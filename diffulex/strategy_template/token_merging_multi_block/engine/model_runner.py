from __future__ import annotations

import os
from pathlib import Path

import torch
from tqdm import tqdm

from diffulex.attention.metadata import AttnMetaDataBase, reset_warming_up, set_warming_up
from diffulex.strategy_template.token_merging_multi_block.attention.metadata import (
    TokenMergingMultiBlockAttnMetaDataTemplate,
)
from diffulex.strategy_template.multi_block.engine.model_runner import MultiBlockModelRunnerTemplate
from diffulex.strategy_template.token_merging_multi_block.engine.request import TokenMergingMultiBlockReqTemplate


class TokenMergingMultiBlockModelRunnerTemplate(MultiBlockModelRunnerTemplate):
    token_merge_renormalize = True

    @staticmethod
    def _debug_dump_dir() -> str | None:
        return os.getenv("DIFFULEX_DEBUG_DUMP_DIR")

    @staticmethod
    def _debug_dump_include_qkv() -> bool:
        return os.getenv("DIFFULEX_DEBUG_DUMP_QKV", "0") == "1"

    @staticmethod
    def _debug_dump_full_logits() -> bool:
        return os.getenv("DIFFULEX_DEBUG_DUMP_FULL_LOGITS", "0") == "1"

    def _debug_dump_enabled(self) -> bool:
        return bool(self._debug_dump_dir())

    def _set_model_debug_dump_enabled(self) -> None:
        model = getattr(self.model, "model", None)
        enable_fn = getattr(model, "enable_debug_dump", None)
        if enable_fn is not None:
            enable_fn(include_qkv=self._debug_dump_include_qkv())

    def _clear_model_debug_dump(self) -> None:
        model = getattr(self.model, "model", None)
        disable_fn = getattr(model, "disable_debug_dump", None)
        if disable_fn is not None:
            disable_fn()

    def _pop_model_debug_dump(self) -> dict[str, object] | None:
        model = getattr(self.model, "model", None)
        pop_fn = getattr(model, "pop_last_debug_dump", None)
        if pop_fn is None:
            return None
        return pop_fn()

    def _append_engine_debug_dump(
        self,
        reqs: list[TokenMergingMultiBlockReqTemplate],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        logits: torch.Tensor,
        sample_output,
    ) -> None:
        dump_dir = self._debug_dump_dir()
        if not dump_dir or not self.is_model_parallel_root:
            return

        debug_capture = self._pop_model_debug_dump()
        if debug_capture is None:
            return

        attn_metadata: TokenMergingMultiBlockAttnMetaDataTemplate = self.fetch_attn_metadata()
        cu_seqlens_q = attn_metadata.cu_seqlens_q.detach().cpu()
        full_logits = self._debug_dump_full_logits()
        out_dir = Path(dump_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        token_merge_meta = None
        if bool(attn_metadata.token_merge_enabled):
            token_merge_meta = {
                "merge_mask": attn_metadata.token_merge_mask.detach().cpu()
                if attn_metadata.token_merge_mask is not None
                else None,
                "topk_ids": attn_metadata.token_merge_topk_ids.detach().cpu()
                if attn_metadata.token_merge_topk_ids is not None
                else None,
                "topk_probs": attn_metadata.token_merge_topk_probs.detach().cpu()
                if attn_metadata.token_merge_topk_probs is not None
                else None,
                "residual_probs": attn_metadata.token_merge_residual_probs.detach().cpu()
                if attn_metadata.token_merge_residual_probs is not None
                else None,
                "mask_token_id": int(attn_metadata.token_merge_mask_token_id)
                if attn_metadata.token_merge_mask_token_id is not None
                else None,
                "renormalize": bool(attn_metadata.token_merge_renormalize),
                "mode": str(attn_metadata.token_merge_mode),
                "weight": float(attn_metadata.token_merge_weight),
            }

        for req_idx, req in enumerate(reqs):
            start = int(cu_seqlens_q[req_idx].item())
            end = int(cu_seqlens_q[req_idx + 1].item())
            req_num_tokens = end - start
            req_logits = logits[start:end]
            req_input_ids = input_ids[start:end]
            req_positions = positions[start:end]
            top_k = min(10, int(req_logits.shape[-1]))
            topk_probs, topk_ids = torch.topk(torch.softmax(req_logits.float(), dim=-1), k=top_k, dim=-1)
            record = {
                "meta": {
                    "req_id": int(req.req_id),
                    "step_nfe": int(req.nfe) + 1,
                    "req_status": getattr(req.status, "name", str(req.status)),
                    "running_len": int(req.running_len),
                    "chunk_size": int(req.chunk_size),
                    "valid_len": int(req.valid_len),
                    "is_prefilling": bool(req.is_prefilling),
                    "rank": int(self.rank),
                    "dp_rank": int(self.dp_rank),
                },
                "inputs": {
                    "input_ids": req_input_ids.detach().cpu(),
                    "positions": req_positions.detach().cpu(),
                    "running_sequence": torch.tensor(list(req.running_sequence), dtype=torch.int64),
                    "running_position_ids": torch.tensor(list(req.running_position_ids), dtype=torch.int64),
                },
                "blocks": [
                    {
                        "block_id": int(block.block_id),
                        "start": int(block.start),
                        "end": int(block.end),
                        "status": getattr(block.status, "name", str(block.status)),
                        "editable_start": int(getattr(block, "editable_start", 0) or 0),
                        "token_ids": torch.tensor(block.token_ids, dtype=torch.int64),
                    }
                    for block in req.dllm_blocks
                    if getattr(block, "is_active", False)
                ],
                "token_merge_metadata": None,
                "hidden_states": {},
                "logits": {
                    "topk_ids": topk_ids.detach().cpu(),
                    "topk_probs": topk_probs.detach().cpu(),
                },
                "sampler_trace": None,
            }
            if full_logits:
                record["logits"]["full"] = req_logits.detach().cpu()

            if token_merge_meta is not None:
                req_token_merge = {}
                for key, value in token_merge_meta.items():
                    if isinstance(value, torch.Tensor):
                        req_token_merge[key] = value[start:end]
                    else:
                        req_token_merge[key] = value
                record["token_merge_metadata"] = req_token_merge

            for key, value in debug_capture.items():
                if isinstance(value, dict):
                    nested = {}
                    for nested_key, nested_value in value.items():
                        if (
                            torch.is_tensor(nested_value)
                            and nested_value.ndim > 0
                            and int(nested_value.shape[0]) == int(input_ids.shape[0])
                        ):
                            nested[nested_key] = nested_value[start:end].detach().cpu()
                        else:
                            nested[nested_key] = nested_value.detach().cpu()
                    record["hidden_states"][key] = nested
                else:
                    if (
                        torch.is_tensor(value)
                        and value.ndim > 0
                        and int(value.shape[0]) == int(input_ids.shape[0])
                    ):
                        record["hidden_states"][key] = value[start:end].detach().cpu()
                    else:
                        record["hidden_states"][key] = value.detach().cpu()

            if sample_output is not None:
                req_id_str = str(req.req_id)
                record["sampler_trace"] = {
                    "dmax_trace_map": dict(getattr(sample_output, "dmax_trace_map", {}).get(req_id_str, {})),
                    "block_state_map": dict(getattr(sample_output, "block_state_map", {}).get(req_id_str, {})),
                    "edit_writes_map": dict(getattr(sample_output, "edit_writes_map", {}).get(req_id_str, {})),
                }

            file_path = out_dir / f"req_{int(req.req_id)}_step_{int(req.nfe) + 1:03d}_rank_{int(self.rank)}.pt"
            torch.save(record, file_path)

    def prepare_chunked_prefill_token_merging_multi_block(
        self: TokenMergingMultiBlockModelRunnerTemplate,
        reqs: list[TokenMergingMultiBlockReqTemplate],
    ):
        input_ids, positions = self.prepare_chunked_prefill_multi_block(reqs)
        self._init_token_merge_metadata(reqs, positions)
        return input_ids, positions

    def _init_token_merge_metadata(
        self: TokenMergingMultiBlockModelRunnerTemplate,
        reqs: list[TokenMergingMultiBlockReqTemplate],
        positions: torch.Tensor,
    ) -> None:
        attn_metadata: TokenMergingMultiBlockAttnMetaDataTemplate = self.fetch_attn_metadata()

        num_tokens = int(positions.numel())
        if num_tokens == 0:
            attn_metadata.init_token_merging(mask_token_id=self.config.mask_token_id)
            return

        descriptors = []
        max_top_k = 1
        has_merge = False
        for req in reqs:
            for position in req.running_position_ids:
                descriptor = req.token_merge_descriptor_for_position(position)
                descriptors.append(descriptor)
                if descriptor is not None:
                    has_merge = True
                    max_top_k = max(max_top_k, len(descriptor.topk_ids))

        if len(descriptors) != num_tokens:
            raise RuntimeError(
                "Token-merge descriptor alignment failed: "
                f"descriptors={len(descriptors)}, num_tokens={num_tokens}"
            )

        if not has_merge:
            attn_metadata.init_token_merging(mask_token_id=self.config.mask_token_id)
            return

        device = positions.device
        merge_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        topk_ids = torch.full(
            (num_tokens, max_top_k),
            int(self.config.mask_token_id),
            dtype=torch.int64,
            device=device,
        )
        topk_probs = torch.zeros((num_tokens, max_top_k), dtype=torch.float32, device=device)
        residual_probs = torch.zeros((num_tokens, 1), dtype=torch.float32, device=device)

        for idx, descriptor in enumerate(descriptors):
            if descriptor is None:
                continue
            k = len(descriptor.topk_ids)
            merge_mask[idx] = True
            topk_ids[idx, :k] = torch.tensor(descriptor.topk_ids, dtype=torch.int64, device=device)
            topk_probs[idx, :k] = torch.tensor(descriptor.topk_probs, dtype=torch.float32, device=device)
            residual_probs[idx, 0] = float(descriptor.residual_prob)

        attn_metadata.init_token_merging(
            merge_mask=merge_mask,
            topk_ids=topk_ids,
            topk_probs=topk_probs,
            residual_probs=residual_probs,
            mask_token_id=self.config.mask_token_id,
            renormalize=bool(self.config.token_merge_renormalize),
            mode=self.config.token_merge_mode,
            weight=float(self.config.token_merge_weight),
        )

    def _graph_token_merge_top_k(self) -> int:
        return max(1, int(self.config.token_merge_top_k))

    def _init_graph_capture_token_merge_metadata(
        self: TokenMergingMultiBlockModelRunnerTemplate,
        attn_metadata: TokenMergingMultiBlockAttnMetaDataTemplate,
        graph_vars: dict[str, torch.Tensor],
        num_tokens: int,
    ) -> None:
        merge_mask = graph_vars["token_merge_mask"][:num_tokens]
        topk_ids = graph_vars["token_merge_topk_ids"][:num_tokens]
        topk_probs = graph_vars["token_merge_topk_probs"][:num_tokens]
        residual_probs = graph_vars["token_merge_residual_probs"][:num_tokens]

        merge_mask.zero_()
        topk_ids.fill_(int(self.config.mask_token_id))
        topk_probs.zero_()
        residual_probs.zero_()
        if num_tokens > 0:
            # Capture the token-merge branch once; replay-time merge_mask controls no-op behavior.
            merge_mask[0] = True
            residual_probs[0, 0] = 1.0

        attn_metadata.init_token_merging(
            merge_mask=merge_mask,
            topk_ids=topk_ids,
            topk_probs=topk_probs,
            residual_probs=residual_probs,
            mask_token_id=self.config.mask_token_id,
            renormalize=bool(self.config.token_merge_renormalize),
            mode=self.config.token_merge_mode,
            weight=float(self.config.token_merge_weight),
        )

    def _bind_graph_token_merge_metadata(
        self: TokenMergingMultiBlockModelRunnerTemplate,
        attn_metadata: TokenMergingMultiBlockAttnMetaDataTemplate,
        graph_vars: dict[str, torch.Tensor],
        num_tokens: int,
    ) -> None:
        graph_merge_mask = graph_vars["token_merge_mask"][:num_tokens]
        graph_topk_ids = graph_vars["token_merge_topk_ids"][:num_tokens]
        graph_topk_probs = graph_vars["token_merge_topk_probs"][:num_tokens]
        graph_residual_probs = graph_vars["token_merge_residual_probs"][:num_tokens]

        graph_merge_mask.zero_()
        graph_topk_ids.fill_(int(self.config.mask_token_id))
        graph_topk_probs.zero_()
        graph_residual_probs.zero_()

        src_merge_mask = attn_metadata.token_merge_mask
        src_topk_ids = attn_metadata.token_merge_topk_ids
        src_topk_probs = attn_metadata.token_merge_topk_probs
        src_residual_probs = attn_metadata.token_merge_residual_probs
        has_merge = (
            bool(attn_metadata.token_merge_enabled)
            and src_merge_mask is not None
            and src_topk_ids is not None
            and src_topk_probs is not None
            and src_residual_probs is not None
        )

        if has_merge:
            if int(src_merge_mask.numel()) != num_tokens:
                raise RuntimeError(
                    "Token-merge metadata length does not match CUDA graph replay size: "
                    f"merge_mask={src_merge_mask.numel()}, num_tokens={num_tokens}"
                )
            src_top_k = int(src_topk_ids.shape[1])
            graph_top_k = int(graph_topk_ids.shape[1])
            if src_top_k > graph_top_k:
                raise RuntimeError(
                    "Token-merge top-k exceeds CUDA graph capture capacity: "
                    f"src_top_k={src_top_k}, graph_top_k={graph_top_k}"
                )

            graph_merge_mask.copy_(src_merge_mask.to(device=graph_merge_mask.device, dtype=torch.bool))
            graph_topk_ids[:, :src_top_k].copy_(src_topk_ids.to(device=graph_topk_ids.device, dtype=torch.int64))
            graph_topk_probs[:, :src_top_k].copy_(
                src_topk_probs.to(device=graph_topk_probs.device, dtype=torch.float32)
            )
            graph_residual_probs.copy_(
                src_residual_probs.to(device=graph_residual_probs.device, dtype=torch.float32)
            )

        attn_metadata.init_token_merging(
            merge_mask=graph_merge_mask,
            topk_ids=graph_topk_ids,
            topk_probs=graph_topk_probs,
            residual_probs=graph_residual_probs,
            mask_token_id=self.config.mask_token_id,
            renormalize=bool(self.config.token_merge_renormalize),
            mode=self.config.token_merge_mode,
            weight=float(self.config.token_merge_weight),
        )

    def run_multi_block(
        self: TokenMergingMultiBlockModelRunnerTemplate,
        reqs: list[TokenMergingMultiBlockReqTemplate],
    ) -> list[int]:
        return self._run_token_merging_multi_block_subgroup(reqs)

    @torch.inference_mode()
    def run_model_multi_block(
        self: TokenMergingMultiBlockModelRunnerTemplate,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ):
        attn_metadata: AttnMetaDataBase = self.fetch_attn_metadata()

        if (
            (attn_metadata.status_table == 0).any()
            or self.enforce_eager
            or self._debug_dump_enabled()
            or input_ids.size(0) > 512 * (self.config.buffer_size * self.config.block_size)
        ):
            return self.model.compute_logits(self.model(input_ids, positions))

        num_tokens = input_ids.size(0)
        captured_num_tokens = next(x for x in self.graph_bs if x >= num_tokens)
        captured_num_seqs = captured_num_tokens // (self.config.block_size * self.config.buffer_size)
        graph = self.graphs[captured_num_tokens]
        graph_vars = self.graph_vars
        graph_capacity = int(graph_vars["context_lens"].size(0))
        if captured_num_seqs > graph_capacity:
            raise RuntimeError(
                "Captured CUDA graph batch size exceeds allocated graph buffer capacity: "
                f"captured_num_seqs={captured_num_seqs}, graph_capacity={graph_capacity}, "
                f"captured_num_tokens={captured_num_tokens}, num_tokens={num_tokens}, "
                f"max_num_reqs={self.config.max_num_reqs}, block_size={self.config.block_size}, "
                f"buffer_size={self.config.buffer_size}"
            )

        for key, value in graph_vars.items():
            if key == "outputs":
                continue
            if key in ("slot_mapping", "page_tables"):
                value.fill_(-1)
            else:
                value.zero_()

        num_reqs = attn_metadata.num_reqs
        graph_vars["input_ids"][:num_tokens] = input_ids
        graph_vars["positions"][:num_tokens] = positions
        graph_vars["slot_mapping"][:num_tokens] = attn_metadata.slot_mapping
        graph_vars["context_lens"][:num_reqs] = attn_metadata.context_lens
        graph_vars["cu_seqlens_q"][: num_reqs + 1] = attn_metadata.cu_seqlens_q
        graph_vars["cu_seqlens_k"][: num_reqs + 1] = attn_metadata.cu_seqlens_k
        graph_vars["valid_slices"][:num_reqs] = attn_metadata.valid_slices
        graph_vars["status_table"][:num_reqs] = attn_metadata.status_table
        graph_vars["prefix_lens"][:num_reqs] = attn_metadata.prefix_lens
        graph_vars["padded_prefix_lens"][:num_reqs] = attn_metadata.padded_prefix_lens
        pt_w = attn_metadata.page_tables.size(1)
        graph_vars["page_tables"][:num_reqs, :pt_w] = attn_metadata.page_tables
        if pt_w < graph_vars["page_tables"].size(1):
            graph_vars["page_tables"][:, pt_w:].fill_(-1)

        for i in range(num_reqs, captured_num_seqs):
            graph_vars["cu_seqlens_q"][i + 1] = graph_vars["cu_seqlens_q"][i]
            graph_vars["cu_seqlens_k"][i + 1] = graph_vars["cu_seqlens_k"][i]

        attn_metadata.slot_mapping = graph_vars["slot_mapping"]
        attn_metadata.context_lens = graph_vars["context_lens"]
        attn_metadata.cu_seqlens_q = graph_vars["cu_seqlens_q"]
        attn_metadata.cu_seqlens_k = graph_vars["cu_seqlens_k"]
        attn_metadata.valid_slices = graph_vars["valid_slices"]
        attn_metadata.status_table = graph_vars["status_table"]
        attn_metadata.prefix_lens = graph_vars["prefix_lens"]
        attn_metadata.padded_prefix_lens = graph_vars["padded_prefix_lens"]
        attn_metadata.page_tables = graph_vars["page_tables"]
        self._bind_graph_token_merge_metadata(attn_metadata, graph_vars, num_tokens)

        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:num_tokens])

    @torch.inference_mode()
    def capture_cudagraph_token_merging_multi_block(self: TokenMergingMultiBlockModelRunnerTemplate):
        set_warming_up(True)
        config = self.config
        hf_config = config.hf_config
        max_num_seqs = min(self.config.max_num_reqs, 512)
        max_num_pages = (config.max_model_len + self.page_size - 1) // self.page_size
        block_size = config.block_size
        buffer_size = config.buffer_size
        chunk_size = block_size * buffer_size
        max_num_tokens = max_num_seqs * chunk_size
        token_merge_top_k = self._graph_token_merge_top_k()

        input_ids = torch.zeros(max_num_tokens, dtype=torch.int64)
        positions = torch.zeros(max_num_tokens, dtype=torch.int64)
        slot_mapping = torch.full((max_num_tokens,), -1, dtype=torch.int32)
        context_lens = torch.zeros(max_num_seqs, dtype=torch.int32)
        page_tables = torch.zeros(max_num_seqs, max_num_pages, dtype=torch.int32)
        valid_slices = torch.zeros(max_num_seqs, dtype=torch.int32)
        status_table = torch.zeros(max_num_seqs, dtype=torch.int32)
        prefix_lens = torch.zeros(max_num_seqs, dtype=torch.int32)
        padded_prefix_lens = torch.zeros(max_num_seqs, dtype=torch.int32)
        token_merge_mask = torch.zeros(max_num_tokens, dtype=torch.bool)
        token_merge_topk_ids = torch.full(
            (max_num_tokens, token_merge_top_k),
            int(self.config.mask_token_id),
            dtype=torch.int64,
        )
        token_merge_topk_probs = torch.zeros((max_num_tokens, token_merge_top_k), dtype=torch.float32)
        token_merge_residual_probs = torch.zeros((max_num_tokens, 1), dtype=torch.float32)
        outputs = torch.zeros(max_num_tokens, hf_config.hidden_size)

        cu_seqlens_q = torch.zeros(max_num_seqs + 1, dtype=torch.int32)
        for i in range(max_num_seqs + 1):
            cu_seqlens_q[i] = i * chunk_size

        cu_seqlens_k = torch.zeros(max_num_seqs + 1, dtype=torch.int32)
        for i in range(max_num_seqs + 1):
            cu_seqlens_k[i] = i * config.max_model_len

        self.graph_bs = []
        seq_bs_list = self._graph_seq_batch_sizes(max_num_seqs)
        for num_seqs in seq_bs_list:
            self.graph_bs.append(num_seqs * chunk_size)
        self.graphs = {}
        self.graph_pool = None

        for num_tokens in tqdm(reversed(self.graph_bs), desc="Capturing CUDA graphs"):
            num_seqs = num_tokens // chunk_size
            graph = torch.cuda.CUDAGraph()

            self.set_attn_metadata(
                False,
                slot_mapping=slot_mapping[:num_tokens],
                context_lens=context_lens[:num_seqs],
                cu_seqlens_q=cu_seqlens_q[: num_seqs + 1],
                cu_seqlens_k=cu_seqlens_k[: num_seqs + 1],
                max_seqlen_q=chunk_size,
                max_seqlen_k=config.max_model_len,
                page_size=config.kv_cache_page_size,
                page_tables=page_tables[:num_seqs],
                block_size=block_size,
                kv_cache_layout=config.kv_cache_layout,
            )
            attn_metadata: TokenMergingMultiBlockAttnMetaDataTemplate = self.fetch_attn_metadata()
            attn_metadata.init_multi_block(
                valid_slices=valid_slices[:num_seqs],
                buffer_size=buffer_size,
                is_prefix_full=self.is_prefix_full,
                status_table=status_table[:num_seqs],
                prefix_lens=prefix_lens[:num_seqs],
                padded_prefix_lens=padded_prefix_lens[:num_seqs],
            )
            capture_graph_vars = {
                "token_merge_mask": token_merge_mask,
                "token_merge_topk_ids": token_merge_topk_ids,
                "token_merge_topk_probs": token_merge_topk_probs,
                "token_merge_residual_probs": token_merge_residual_probs,
            }
            self._init_graph_capture_token_merge_metadata(attn_metadata, capture_graph_vars, num_tokens)

            outputs[:num_tokens] = self.model(input_ids[:num_tokens], positions[:num_tokens])
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:num_tokens] = self.model(input_ids[:num_tokens], positions[:num_tokens])
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[num_tokens] = graph
            torch.cuda.synchronize()
            self.reset_attn_metadata()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            page_tables=page_tables,
            valid_slices=valid_slices,
            status_table=status_table,
            prefix_lens=prefix_lens,
            padded_prefix_lens=padded_prefix_lens,
            token_merge_mask=token_merge_mask,
            token_merge_topk_ids=token_merge_topk_ids,
            token_merge_topk_probs=token_merge_topk_probs,
            token_merge_residual_probs=token_merge_residual_probs,
            outputs=outputs,
        )
        reset_warming_up()

    def _run_token_merging_multi_block_subgroup(
        self: TokenMergingMultiBlockModelRunnerTemplate,
        reqs: list[TokenMergingMultiBlockReqTemplate],
    ):
        local_reqs = self.filter_local_reqs(reqs)
        if not local_reqs:
            return self.gather_dp_sample_output(None)

        input_ids, positions = self.prepare_chunked_prefill_token_merging_multi_block(local_reqs)
        temperatures = self.prepare_sample(local_reqs) if self.is_model_parallel_root else None
        debug_enabled = self._debug_dump_enabled()
        if debug_enabled:
            self._set_model_debug_dump_enabled()
        try:
            logits = self.run_model_multi_block(input_ids, positions)
            sample_output = self.sampler(local_reqs, logits, temperatures) if self.is_model_parallel_root else None
            if debug_enabled:
                self._append_engine_debug_dump(local_reqs, input_ids, positions, logits, sample_output)
        finally:
            if debug_enabled:
                self._clear_model_debug_dump()
        self.reset_attn_metadata()
        return self.gather_dp_sample_output(sample_output)
