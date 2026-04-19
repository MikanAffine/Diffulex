from __future__ import annotations

import os
import re
import subprocess

import torch
import torch.distributed as dist

from diffulex.moe.dispatcher.base_dispatcher import DispatcherOutput, TokenDispatcher
from diffulex.moe.metadata import DeepEPDispatchMetadata, DispatcherStage
from diffulex.moe.mode import DeepEPMode

try:
    from deep_ep import Buffer as DeepEPBuffer

    _DEEP_EP_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # DeepEP can fail with native extension linkage errors.
    DeepEPBuffer = None
    _DEEP_EP_IMPORT_ERROR = exc


def _import_error_message() -> str:
    if _DEEP_EP_IMPORT_ERROR is None:
        return ""
    return (
        "DeepEP is installed but cannot be imported correctly. "
        f"Import error: {type(_DEEP_EP_IMPORT_ERROR).__name__}: {_DEEP_EP_IMPORT_ERROR}. "
        "Use moe_dispatcher_backend='standard' for non-A2A TP MoE, use 'naive' only for A2A debugging, "
        "or rebuild DeepEP for the current CUDA/PyTorch stack."
    )


def _visible_physical_device_id() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    current = torch.cuda.current_device()
    if not visible:
        return current
    devices = [item.strip() for item in visible.split(",") if item.strip()]
    if current >= len(devices):
        return current
    device = devices[current]
    return int(device) if device.isdigit() else current


def _query_nvidia_smi_topology() -> dict[tuple[int, int], str]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "topo", "-m"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        return {}

    matrix: dict[tuple[int, int], str] = {}
    for raw_line in output.splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw_line).strip()
        if not line.startswith("GPU"):
            continue
        parts = line.split()
        row_match = re.fullmatch(r"GPU(\d+)", parts[0])
        if row_match is None:
            continue
        row = int(row_match.group(1))
        for col, link in enumerate(parts[1:]):
            if link in {"X", "CPU"}:
                continue
            matrix[(row, col)] = link
    return matrix


def _ensure_full_nvlink_connectivity(ep_group, ep_size: int) -> None:
    if ep_size <= 1:
        return

    physical_id = _visible_physical_device_id()
    physical_ids = [None for _ in range(ep_size)]
    dist.all_gather_object(physical_ids, physical_id, group=ep_group)

    detail: list[str | None] = [None]
    if dist.get_rank(group=ep_group) == 0:
        topology = _query_nvidia_smi_topology()
        missing: list[str] = []
        for i, src in enumerate(physical_ids):
            for dst in physical_ids[i + 1 :]:
                link = topology.get((int(src), int(dst))) or topology.get((int(dst), int(src))) or "UNKNOWN"
                if not link.startswith("NV"):
                    missing.append(f"GPU{src}<->GPU{dst}={link}")
        if missing:
            detail[0] = (
                "DeepEP native intranode dispatch requires full NVLink connectivity inside the EP group, "
                f"but this machine has non-NVLink pairs: {', '.join(missing[:8])}. "
                "Use moe_dispatcher_backend='standard' for non-A2A TP MoE on this machine; "
                "use 'naive' only for A2A debugging."
            )
    dist.broadcast_object_list(detail, src=dist.get_global_rank(ep_group, 0), group=ep_group)
    if detail[0] is not None:
        raise RuntimeError(detail[0])


class DeepEPDispatcher(TokenDispatcher):
    """Native DeepEP normal-mode dispatcher.

    DeepEP communicates one row per received token. Diffulex's expert runner
    consumes one row per local expert slot, so dispatch expands received tokens
    into expert slots and combine first reduces slot outputs back to DeepEP's
    token rows before calling native combine.
    """

    def __init__(
        self,
        *,
        ep_group,
        ep_size: int,
        num_local_experts: int,
        top_k: int,
        hidden_size: int | None = None,
        num_experts: int | None = None,
        params_dtype: torch.dtype | None = None,
        deepep_mode: str | DeepEPMode = DeepEPMode.AUTO,
        num_max_dispatch_tokens_per_rank: int = 256,
        async_finish: bool = False,
        **_unused,
    ) -> None:
        if DeepEPBuffer is None:
            raise ImportError(_import_error_message())
        if ep_group is None:
            raise RuntimeError(
                "DeepEPDispatcher requires an EP process group. Use moe_dispatcher_backend='standard' "
                "for non-A2A TP MoE, or 'naive' only for A2A debugging."
            )

        self.ep_group = ep_group
        self.ep_size = ep_size
        self.ep_rank = dist.get_rank(group=ep_group)
        self.num_local_experts = num_local_experts
        self.local_expert_start = self.ep_rank * self.num_local_experts
        self.local_expert_end = self.local_expert_start + self.num_local_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.num_experts = num_experts or ep_size * num_local_experts
        self.params_dtype = params_dtype or torch.bfloat16
        self.deepep_mode = DeepEPMode.from_value(deepep_mode)
        self.forward_phase = "unknown"
        self.num_max_dispatch_tokens_per_rank = int(num_max_dispatch_tokens_per_rank)
        if self.deepep_mode.enable_normal():
            _ensure_full_nvlink_connectivity(ep_group, ep_size)
        self.async_finish = async_finish
        self._stage = DispatcherStage.INITIAL
        self._dispatch_intermediate_state = None
        self._combine_intermediate_state = None
        self._buffer_key = (
            id(ep_group),
            self.ep_size,
            self.hidden_size,
            self.params_dtype,
            self.deepep_mode.value,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
        )

    _BUFFER_CACHE: dict[tuple[object, ...], object] = {}

    def set_forward_phase(self, phase: str) -> None:
        self.forward_phase = phase

    def _resolve_mode(self) -> DeepEPMode:
        return DeepEPMode.from_value(self.deepep_mode).resolve(getattr(self, "forward_phase", "unknown"))

    def _update_stage(self, expected: DispatcherStage, next_stage: DispatcherStage) -> None:
        if self._stage is not expected:
            raise RuntimeError(f"Invalid DeepEPDispatcher stage: got {self._stage}, expected {expected}.")
        self._stage = next_stage

    def _get_buffer(self):
        resolved_mode = self._resolve_mode()
        buffer_key = (
            id(self.ep_group),
            self.ep_size,
            self.hidden_size,
            self.params_dtype,
            resolved_mode.value,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
        )
        if buffer_key in self._BUFFER_CACHE:
            return self._BUFFER_CACHE[buffer_key]
        assert DeepEPBuffer is not None
        if self.hidden_size is None:
            raise RuntimeError("DeepEPDispatcher requires hidden_size to initialize native DeepEP Buffer.")

        param_bytes = torch.empty((), dtype=self.params_dtype).element_size()
        hidden_bytes = int(self.hidden_size) * param_bytes
        num_nvl_bytes = 0
        num_rdma_bytes = 0
        low_latency_mode = resolved_mode is DeepEPMode.LOW_LATENCY
        if not low_latency_mode:
            dispatch_config = DeepEPBuffer.get_dispatch_config(self.ep_size)
            combine_config = DeepEPBuffer.get_combine_config(self.ep_size)
            num_nvl_bytes = max(
                dispatch_config.get_nvl_buffer_size_hint(hidden_bytes, self.ep_size),
                combine_config.get_nvl_buffer_size_hint(hidden_bytes, self.ep_size),
            )
            num_rdma_bytes = max(
                dispatch_config.get_rdma_buffer_size_hint(hidden_bytes, self.ep_size),
                combine_config.get_rdma_buffer_size_hint(hidden_bytes, self.ep_size),
            )
        else:
            num_rdma_bytes = DeepEPBuffer.get_low_latency_rdma_size_hint(
                self.num_max_dispatch_tokens_per_rank,
                int(self.hidden_size),
                self.ep_size,
                self.num_experts,
            )
        buffer = DeepEPBuffer(
            self.ep_group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=low_latency_mode,
            num_qps_per_rank=(self.num_experts // self.ep_size if low_latency_mode else DeepEPBuffer.num_sms),
            allow_mnnvl=True,
        )
        self._BUFFER_CACHE[buffer_key] = buffer
        return buffer

    @staticmethod
    def _build_src2dst(reorder_indices: torch.Tensor) -> torch.Tensor:
        src2dst = torch.empty_like(reorder_indices, dtype=torch.int64)
        src2dst[reorder_indices] = torch.arange(
            reorder_indices.numel(),
            device=reorder_indices.device,
            dtype=torch.int64,
        )
        return src2dst

    @staticmethod
    def _expert_major_reorder(
        token_indices: torch.Tensor,
        local_expert_ids: torch.Tensor,
        weights: torch.Tensor,
        *,
        num_local_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        reorder_indices = torch.argsort(local_expert_ids.to(torch.int64), stable=True)
        reordered_token_indices = token_indices[reorder_indices].contiguous()
        reordered_local_expert_ids = local_expert_ids[reorder_indices].contiguous()
        reordered_weights = weights[reorder_indices].contiguous()
        src2dst = DeepEPDispatcher._build_src2dst(reorder_indices)
        expert_ids = torch.arange(
            num_local_experts + 1,
            device=local_expert_ids.device,
            dtype=reordered_local_expert_ids.dtype,
        )
        seg_indptr = torch.searchsorted(reordered_local_expert_ids, expert_ids).to(torch.int64)
        num_recv_tokens_per_expert = (seg_indptr[1:] - seg_indptr[:-1]).to(torch.int32)
        return (
            reordered_token_indices,
            reordered_local_expert_ids,
            reordered_weights,
            reorder_indices,
            src2dst,
            seg_indptr,
            num_recv_tokens_per_expert,
        )

    def _validate_active_inputs(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor | None,
        topk_weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.dtype != torch.bfloat16:
            raise RuntimeError(
                "DeepEP native normal dispatch currently requires bf16 hidden_states. "
                f"Got {hidden_states.dtype}. Use moe_dispatcher_backend='standard' for non-A2A TP MoE, "
                "or 'naive' only for A2A debugging."
            )
        if topk_ids is None or topk_weights is None:
            raise ValueError("DeepEPDispatcher active dispatch requires topk_ids and topk_weights.")
        if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
            raise ValueError(
                "DeepEPDispatcher expects topk_ids/topk_weights with matching [num_tokens, top_k] shape, "
                f"got topk_ids={tuple(topk_ids.shape)}, topk_weights={tuple(topk_weights.shape)}."
            )
        if int(topk_ids.shape[0]) != int(hidden_states.shape[0]):
            raise ValueError(
                "DeepEPDispatcher topk metadata does not match hidden_states: "
                f"hidden_tokens={hidden_states.shape[0]}, topk_ids={tuple(topk_ids.shape)}."
            )
        return topk_ids.to(torch.int64).contiguous(), topk_weights.to(torch.float32).contiguous()

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor | None,
        topk_weights: torch.Tensor | None,
        *,
        active_dispatch: bool = True,
    ) -> DispatcherOutput:
        if self._resolve_mode() is DeepEPMode.LOW_LATENCY:
            self.dispatch_low_latency_a(hidden_states, topk_ids, topk_weights, active_dispatch=active_dispatch)
            return self.dispatch_low_latency_b()
        self.dispatch_a(hidden_states, topk_ids, topk_weights, active_dispatch=active_dispatch)
        return self.dispatch_b()

    def dispatch_low_latency_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor | None,
        topk_weights: torch.Tensor | None,
        *,
        active_dispatch: bool = True,
    ) -> None:
        self._update_stage(DispatcherStage.INITIAL, DispatcherStage.AFTER_DISPATCH_A)
        num_tokens, hidden_size = hidden_states.shape
        if active_dispatch:
            dispatch_hidden = hidden_states.contiguous()
            dispatch_topk_ids, dispatch_topk_weights = self._validate_active_inputs(
                dispatch_hidden,
                topk_ids,
                topk_weights,
            )
        else:
            dispatch_hidden = torch.empty((0, hidden_size), device=hidden_states.device, dtype=hidden_states.dtype)
            dispatch_topk_ids = torch.empty((0, self.top_k), device=hidden_states.device, dtype=torch.int64)
            dispatch_topk_weights = torch.empty((0, self.top_k), device=hidden_states.device, dtype=torch.float32)

        if num_tokens > self.num_max_dispatch_tokens_per_rank:
            raise RuntimeError(
                "DeepEP low-latency dispatch requires num_tokens <= "
                f"num_max_dispatch_tokens_per_rank, got {num_tokens} > {self.num_max_dispatch_tokens_per_rank}."
            )

        buffer = self._get_buffer()
        recv_hidden_states, recv_count, handle, event, hook = buffer.low_latency_dispatch(
            dispatch_hidden,
            dispatch_topk_ids,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
            use_fp8=False,
            async_finish=True,
            return_recv_hook=False,
        )
        event.current_stream_wait()
        self._dispatch_intermediate_state = (
            hidden_states,
            bool(active_dispatch),
            num_tokens,
            recv_hidden_states,
            recv_count,
            handle,
            dispatch_topk_ids,
            dispatch_topk_weights,
        )

    def dispatch_low_latency_b(self) -> DispatcherOutput:
        self._update_stage(DispatcherStage.AFTER_DISPATCH_A, DispatcherStage.AFTER_DISPATCH_B)
        state = self._dispatch_intermediate_state
        self._dispatch_intermediate_state = None
        assert state is not None
        (
            original_hidden_states,
            active_dispatch,
            num_tokens,
            packed_recv_hidden_states,
            recv_count,
            handle,
            dispatch_topk_ids,
            dispatch_topk_weights,
        ) = state
        device = original_hidden_states.device
        dtype = original_hidden_states.dtype
        hidden_size = int(original_hidden_states.shape[1])
        capacity = int(packed_recv_hidden_states.shape[1])
        recv_slot_hidden_states = packed_recv_hidden_states.reshape(-1, hidden_size).contiguous()
        expert_ids = torch.arange(
            self.num_local_experts,
            device=device,
            dtype=torch.int32,
        )
        recv_local_expert = expert_ids[:, None].expand(self.num_local_experts, capacity).reshape(-1).contiguous()
        recv_weights = torch.ones((recv_local_expert.numel(),), device=device, dtype=torch.float32)
        recv_token_indices = torch.arange(recv_local_expert.numel(), device=device, dtype=torch.int32)
        seg_indptr = torch.arange(
            self.num_local_experts + 1,
            device=device,
            dtype=torch.int64,
        ) * capacity

        metadata = DeepEPDispatchMetadata(
            num_tokens=int(num_tokens),
            hidden_size=hidden_size,
            dtype=dtype,
            device=device,
            send_splits=[],
            recv_splits=[],
            recv_hidden_states=recv_slot_hidden_states,
            recv_local_expert=recv_local_expert,
            recv_token_indices=recv_token_indices,
            recv_weights=recv_weights,
            total_recv_slots=int(recv_local_expert.numel()),
            active_dispatch=active_dispatch,
            reordered_token_indices=recv_token_indices,
            reordered_local_expert_ids=recv_local_expert,
            seg_indptr=seg_indptr,
            num_recv_tokens_per_expert=recv_count.to(torch.int32).contiguous(),
            low_latency=True,
            low_latency_handle=handle,
            low_latency_topk_ids=dispatch_topk_ids,
            low_latency_topk_weights=dispatch_topk_weights,
            low_latency_recv_count=recv_count,
            low_latency_capacity=capacity,
        )
        return DispatcherOutput(
            metadata=metadata,
            recv_hidden_states=recv_slot_hidden_states,
            recv_local_expert_ids=recv_local_expert,
            recv_weights=recv_weights,
        )

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor | None,
        topk_weights: torch.Tensor | None,
        *,
        active_dispatch: bool = True,
    ) -> None:
        self._update_stage(DispatcherStage.INITIAL, DispatcherStage.AFTER_DISPATCH_A)
        num_tokens, hidden_size = hidden_states.shape
        if active_dispatch:
            dispatch_hidden = hidden_states.contiguous()
            dispatch_topk_ids, dispatch_topk_weights = self._validate_active_inputs(
                dispatch_hidden,
                topk_ids,
                topk_weights,
            )
        else:
            dispatch_hidden = torch.empty((0, hidden_size), device=hidden_states.device, dtype=hidden_states.dtype)
            dispatch_topk_ids = torch.empty((0, self.top_k), device=hidden_states.device, dtype=torch.int64)
            dispatch_topk_weights = torch.empty((0, self.top_k), device=hidden_states.device, dtype=torch.float32)

        buffer = self._get_buffer()
        previous_event = DeepEPBuffer.capture() if self.async_finish else None
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = buffer.get_dispatch_layout(
            dispatch_topk_ids,
            self.num_experts,
            previous_event=previous_event,
            async_finish=self.async_finish,
            allocate_on_comm_stream=previous_event is not None,
        )
        (
            recv_hidden_states,
            recv_topk_ids,
            recv_topk_weights,
            _num_recv_tokens_per_expert_list,
            handle,
            event,
        ) = buffer.dispatch(
            dispatch_hidden,
            topk_idx=dispatch_topk_ids,
            topk_weights=dispatch_topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            async_finish=self.async_finish,
            allocate_on_comm_stream=(previous_event is not None) and self.async_finish,
            config=DeepEPBuffer.get_dispatch_config(self.ep_size),
        )
        if self.async_finish:
            event.current_stream_wait()
        self._dispatch_intermediate_state = (
            hidden_states,
            bool(active_dispatch),
            num_tokens,
            recv_hidden_states,
            recv_topk_ids,
            recv_topk_weights,
            handle,
            num_tokens_per_rank,
        )

    def dispatch_b(self) -> DispatcherOutput:
        self._update_stage(DispatcherStage.AFTER_DISPATCH_A, DispatcherStage.AFTER_DISPATCH_B)
        state = self._dispatch_intermediate_state
        self._dispatch_intermediate_state = None
        assert state is not None
        (
            original_hidden_states,
            active_dispatch,
            num_tokens,
            recv_hidden_states,
            recv_topk_ids,
            recv_topk_weights,
            handle,
            num_tokens_per_rank,
        ) = state
        device = original_hidden_states.device
        dtype = original_hidden_states.dtype
        hidden_size = int(original_hidden_states.shape[1])
        native_recv_num_tokens = int(recv_hidden_states.shape[0])
        native_recv_topk_ids = (
            torch.empty((0, self.top_k), device=device, dtype=torch.int64)
            if recv_topk_ids is None
            else recv_topk_ids.to(torch.int64).contiguous()
        )
        native_recv_topk_weights = (
            torch.empty((0, self.top_k), device=device, dtype=torch.float32)
            if recv_topk_weights is None
            else recv_topk_weights.to(torch.float32).contiguous()
        )
        reorder_indices = None

        if native_recv_num_tokens == 0:
            recv_token_indices = torch.empty((0,), device=device, dtype=torch.int32)
            recv_local_expert = torch.empty((0,), device=device, dtype=torch.int32)
            recv_weights = torch.empty((0,), device=device, dtype=torch.float32)
            recv_slot_hidden_states = torch.empty((0, hidden_size), device=device, dtype=dtype)
            src2dst = None
            seg_indptr = None
            num_recv_tokens_per_expert = None
        else:
            token_rows = (
                torch.arange(native_recv_num_tokens, device=device, dtype=torch.int32)[:, None]
                .expand_as(native_recv_topk_ids)
                .contiguous()
            )
            local_mask = (native_recv_topk_ids >= self.local_expert_start) & (
                native_recv_topk_ids < self.local_expert_end
            )
            recv_token_indices = token_rows[local_mask].contiguous()
            recv_local_expert = (
                native_recv_topk_ids[local_mask] - self.local_expert_start
            ).to(torch.int32).contiguous()
            recv_weights = native_recv_topk_weights[local_mask].contiguous()
            recv_slot_hidden_states = recv_hidden_states[recv_token_indices.long()].contiguous()

            if recv_local_expert.numel() == 0:
                src2dst = None
                seg_indptr = None
                num_recv_tokens_per_expert = None
            else:
                (
                    recv_token_indices,
                    recv_local_expert,
                    recv_weights,
                    reorder_indices,
                    src2dst,
                    seg_indptr,
                    num_recv_tokens_per_expert,
                ) = self._expert_major_reorder(
                    recv_token_indices,
                    recv_local_expert,
                    recv_weights,
                    num_local_experts=self.num_local_experts,
                )
                recv_slot_hidden_states = recv_slot_hidden_states[reorder_indices].contiguous()

        metadata = DeepEPDispatchMetadata(
            num_tokens=int(num_tokens),
            hidden_size=hidden_size,
            dtype=dtype,
            device=device,
            send_splits=num_tokens_per_rank.cpu().tolist(),
            recv_splits=[],
            recv_hidden_states=recv_slot_hidden_states,
            recv_local_expert=recv_local_expert,
            recv_token_indices=recv_token_indices,
            recv_weights=recv_weights,
            total_recv_slots=int(recv_token_indices.numel()),
            active_dispatch=active_dispatch,
            src2dst=src2dst,
            reorder_indices=reorder_indices if recv_local_expert.numel() > 0 else None,
            reordered_token_indices=recv_token_indices,
            reordered_local_expert_ids=recv_local_expert,
            seg_indptr=seg_indptr,
            num_recv_tokens_per_expert=num_recv_tokens_per_expert,
            native_handle=handle,
            native_recv_num_tokens=native_recv_num_tokens,
            native_recv_topk_ids=native_recv_topk_ids,
            native_recv_topk_weights=native_recv_topk_weights,
        )
        return DispatcherOutput(
            metadata=metadata,
            recv_hidden_states=recv_slot_hidden_states,
            recv_local_expert_ids=recv_local_expert,
            recv_weights=recv_weights,
        )

    @staticmethod
    def _restore_src_order(values: torch.Tensor, metadata: DeepEPDispatchMetadata) -> torch.Tensor:
        if metadata.src2dst is None or values.numel() == 0:
            return values
        return values[metadata.src2dst].contiguous()

    def combine(
        self,
        slot_outputs: torch.Tensor,
        metadata: DeepEPDispatchMetadata,
    ) -> torch.Tensor:
        if metadata.low_latency:
            self.combine_low_latency_a(slot_outputs, metadata)
            return self.combine_low_latency_b()
        self.combine_a(slot_outputs, metadata)
        return self.combine_b()

    def combine_low_latency_a(
        self,
        slot_outputs: torch.Tensor,
        metadata: DeepEPDispatchMetadata,
    ) -> None:
        self._update_stage(DispatcherStage.AFTER_DISPATCH_B, DispatcherStage.AFTER_COMBINE_A)
        if metadata.low_latency_handle is None:
            raise RuntimeError("DeepEP low-latency combine is missing dispatch handle.")
        if metadata.low_latency_topk_ids is None or metadata.low_latency_topk_weights is None:
            raise RuntimeError("DeepEP low-latency combine is missing original top-k metadata.")
        if slot_outputs.dtype != metadata.dtype:
            slot_outputs = slot_outputs.to(metadata.dtype)
        capacity = int(metadata.low_latency_capacity or 0)
        packed_outputs = slot_outputs.contiguous().reshape(
            self.num_local_experts,
            capacity,
            metadata.hidden_size,
        )
        buffer = self._get_buffer()
        combined_hidden_states, event, hook = buffer.low_latency_combine(
            packed_outputs,
            metadata.low_latency_topk_ids,
            metadata.low_latency_topk_weights,
            metadata.low_latency_handle,
            async_finish=True,
            return_recv_hook=False,
        )
        event.current_stream_wait()
        self._combine_intermediate_state = (combined_hidden_states,)

    def combine_low_latency_b(self) -> torch.Tensor:
        self._update_stage(DispatcherStage.AFTER_COMBINE_A, DispatcherStage.INITIAL)
        state = self._combine_intermediate_state
        self._combine_intermediate_state = None
        assert state is not None
        (combined_hidden_states,) = state
        return combined_hidden_states.contiguous()

    def combine_a(
        self,
        slot_outputs: torch.Tensor,
        metadata: DeepEPDispatchMetadata,
    ) -> None:
        self._update_stage(DispatcherStage.AFTER_DISPATCH_B, DispatcherStage.AFTER_COMBINE_A)
        if metadata.native_handle is None:
            raise RuntimeError("DeepEP combine is missing native dispatch handle.")
        if slot_outputs.dtype != metadata.dtype:
            slot_outputs = slot_outputs.to(metadata.dtype)
        slot_outputs = self._restore_src_order(slot_outputs, metadata)
        recv_token_indices = self._restore_src_order(metadata.recv_token_indices, metadata)

        native_recv_num_tokens = int(metadata.native_recv_num_tokens or 0)
        token_outputs = torch.zeros(
            (native_recv_num_tokens, metadata.hidden_size),
            device=metadata.device,
            dtype=torch.float32,
        )
        if slot_outputs.numel() > 0:
            token_outputs.index_add_(0, recv_token_indices.long(), slot_outputs.float())
        token_outputs = token_outputs.to(metadata.dtype).contiguous()

        previous_event = DeepEPBuffer.capture() if self.async_finish else None
        self._combine_intermediate_state = (token_outputs, metadata.native_handle, previous_event)

    def combine_b(self) -> torch.Tensor:
        self._update_stage(DispatcherStage.AFTER_COMBINE_A, DispatcherStage.INITIAL)
        state = self._combine_intermediate_state
        self._combine_intermediate_state = None
        assert state is not None
        token_outputs, handle, previous_event = state
        buffer = self._get_buffer()
        combined_hidden_states, _combined_topk_weights, event = buffer.combine(
            token_outputs,
            handle,
            async_finish=self.async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=previous_event is not None,
            config=DeepEPBuffer.get_combine_config(self.ep_size),
        )
        if self.async_finish:
            event.current_stream_wait()
        return combined_hidden_states.contiguous()

__all__ = ["DeepEPDispatcher"]
