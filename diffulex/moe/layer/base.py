import os
import importlib.util
from abc import ABC
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffulex.attention import fetch_attn_metadata
from diffulex.moe.config import (
    get_moe_intermediate_size,
    get_num_experts,
    get_num_experts_per_tok,
    get_norm_topk_prob,
)
from diffulex.moe.topk import build_topk_router
from diffulex_kernel import fused_moe
from diffulex.layer.activation import SiluAndMul
from diffulex.layer.linear import ColumnParallelLinear, RowParallelLinear

_EXTERNAL_DMAX_FUSED_MOE = None
_EXTERNAL_DMAX_FUSED_MOE_LOAD_ERR: Exception | None = None


def _load_external_dmax_fused_moe():
    """Load DMax/dInfer's fused_moe implementation on demand.

    This is a pure A/B backend for numeric diagnosis. If the external module
    is unavailable, return None and let caller fallback.
    """
    global _EXTERNAL_DMAX_FUSED_MOE, _EXTERNAL_DMAX_FUSED_MOE_LOAD_ERR
    if _EXTERNAL_DMAX_FUSED_MOE is not None:
        return _EXTERNAL_DMAX_FUSED_MOE
    if _EXTERNAL_DMAX_FUSED_MOE_LOAD_ERR is not None:
        return None

    fuse_path = os.environ.get(
        "DIFFULEX_DMAX_FUSED_MOE_PATH",
        "/home/jyj/workspace/denglab-diffulex/DMax/dInfer/tools/fuse_moe.py",
    )
    path = Path(fuse_path)
    if not path.exists():
        _EXTERNAL_DMAX_FUSED_MOE_LOAD_ERR = FileNotFoundError(f"Missing external fused_moe file: {path}")
        return None

    try:
        spec = importlib.util.spec_from_file_location("diffulex_external_dmax_fuse_moe", str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module spec from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[call-arg]
        fn = getattr(module, "fused_moe", None)
        if fn is None:
            raise AttributeError(f"`fused_moe` not found in {path}")
        _EXTERNAL_DMAX_FUSED_MOE = fn
        return _EXTERNAL_DMAX_FUSED_MOE
    except Exception as exc:
        _EXTERNAL_DMAX_FUSED_MOE_LOAD_ERR = exc
        return None


class SharedExpertMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, *, hidden_act: str = "silu") -> None:
        super().__init__()
        if hidden_act != "silu":
            raise NotImplementedError("SharedExpertMLP currently supports only silu.")
        self.gate_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        self.down_proj.tp_all_reduce_kind = "shared_expert_down_proj"
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.act_fn(torch.cat((self.gate_proj(hidden_states), self.up_proj(hidden_states)), dim=-1))
        )

class FusedMoE(nn.Module, ABC):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        *,
        hidden_act: str = "silu",
        norm_topk_prob: bool = True,
        num_shared_experts: int = 0,
        shared_expert_intermediate_size: int | None = None,
    ) -> None:
        super().__init__()

        if hidden_act != "silu":
            raise NotImplementedError("only silu is supported currently")
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_act = hidden_act
        self.norm_topk_prob = norm_topk_prob
        self.num_shared_experts = num_shared_experts

        self.router = build_topk_router(
            "triton",
            top_k=top_k,
            renormalize=norm_topk_prob,
            scoring_func="softmax",
        )
        self.shared_experts = None
        if num_shared_experts > 0:
            shared_intermediate_size = int(shared_expert_intermediate_size or intermediate_size * num_shared_experts)
            self.shared_experts = SharedExpertMLP(
                hidden_size,
                shared_intermediate_size,
                hidden_act=hidden_act,
            )

        self.fetch_attn_metadata = fetch_attn_metadata

    @classmethod
    def from_config(cls, config) -> "FusedMoE":
        return cls(
            hidden_size=config.hidden_size,
            intermediate_size=get_moe_intermediate_size(config),
            num_experts=get_num_experts(config),
            top_k=get_num_experts_per_tok(config),
            hidden_act=getattr(config, "hidden_act", "silu"),
            norm_topk_prob=get_norm_topk_prob(config),
            num_shared_experts=int(getattr(config, "num_shared_experts", 0) or 0),
        )
    
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # return final_hidden_states, router_logits
        raise NotImplementedError

    def add_shared_experts(self, routed_states: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.shared_experts is None:
            return routed_states
        shared_states = self.shared_experts(hidden_states)
        try:
            from diffulex.model.llada2 import _llada2_debug_enabled, _llada2_debug_store_nested
        except Exception:
            _llada2_debug_enabled = None
            _llada2_debug_store_nested = None
        if _llada2_debug_enabled is not None and _llada2_debug_enabled():
            layer_idx = getattr(self, "_llada2_layer_idx", None)
            if layer_idx is not None:
                _llada2_debug_store_nested("moe", f"layer_{layer_idx}_routed_output", routed_states)
                _llada2_debug_store_nested("moe", f"layer_{layer_idx}_shared_output", shared_states)
        return routed_states + shared_states

    @staticmethod
    def _phase_from_prefill_flags(is_prefill) -> str:
        if isinstance(is_prefill, bool):
            return "prefill" if is_prefill else "decode"

        if torch.is_tensor(is_prefill):
            if is_prefill.numel() == 0:
                return "unknown"
            flags = is_prefill.to(dtype=torch.bool)
            all_prefill = bool(flags.all().item())
            any_prefill = bool(flags.any().item())
        else:
            try:
                flags = [bool(flag) for flag in is_prefill]
            except TypeError:
                return "unknown"
            if not flags:
                return "unknown"
            all_prefill = all(flags)
            any_prefill = any(flags)

        if all_prefill:
            return "prefill"
        if not any_prefill:
            return "decode"
        return "mixed"

    def get_current_phase(self) -> str:
        """Return the current inference phase for this forward pass."""
        try:
            attn_metadata = self.fetch_attn_metadata()
        except Exception:
            return "unknown"

        if attn_metadata is None:
            return "unknown"

        phase = self._phase_from_prefill_flags(attn_metadata.is_prefill)
        if phase != "unknown":
            return phase

        status_table = attn_metadata.status_table
        if status_table is None:
            return "unknown"

        if torch.is_tensor(status_table):
            return self._phase_from_prefill_flags(status_table == 0)

        try:
            return self._phase_from_prefill_flags([int(status) == 0 for status in status_table])
        except TypeError:
            return self._phase_from_prefill_flags(int(status_table) == 0)

    def expert_gemm(
        self,
        impl: str,
        hidden_states: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        local_expert_start: int = 0,
        hidden_act: str = "silu"
    ) -> torch.Tensor:
        try:
            from diffulex.model.llada2 import _llada2_debug_enabled, _llada2_debug_store_nested
        except Exception:
            _llada2_debug_enabled = None
            _llada2_debug_store_nested = None
        moe_impl_override = os.getenv("DIFFULEX_MOE_GEMM_IMPL", "").strip().lower()
        if moe_impl_override:
            impl = moe_impl_override
        if os.getenv("DIFFULEX_REFERENCE_MOE_GEMM", "0") == "1":
            impl = "naive"
        if impl == "triton":
            out = fused_moe(
                hidden_states=hidden_states,
                w13=w13,
                w2=w2,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                local_expert_start=local_expert_start,
                hidden_act=hidden_act,
            )
            if _llada2_debug_enabled is not None and _llada2_debug_enabled():
                layer_idx = getattr(self, "_llada2_layer_idx", None)
                if layer_idx is not None:
                    _llada2_debug_store_nested("moe", f"layer_{layer_idx}_expert_gemm_output", out.clone())
            return out
        if impl == "dmax_vllm":
            ext_fused_moe = _load_external_dmax_fused_moe()
            if ext_fused_moe is None:
                # Soft fallback to current kernel so diagnostics can continue.
                out = fused_moe(
                    hidden_states=hidden_states,
                    w13=w13,
                    w2=w2,
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                    local_expert_start=local_expert_start,
                    hidden_act=hidden_act,
                )
            else:
                # External fused_moe expects local expert ids.
                local_topk_ids = topk_ids.to(torch.int64) - int(local_expert_start)
                valid = (local_topk_ids >= 0) & (local_topk_ids < w13.shape[0])
                safe_topk_ids = torch.where(valid, local_topk_ids, torch.zeros_like(local_topk_ids))
                safe_topk_weights = torch.where(valid, topk_weights, torch.zeros_like(topk_weights))
                out = ext_fused_moe(
                    hidden_states=hidden_states,
                    w1=w13,
                    w2=w2,
                    topk_weights=safe_topk_weights,
                    topk_ids=safe_topk_ids,
                    inplace=False,
                )
            if _llada2_debug_enabled is not None and _llada2_debug_enabled():
                layer_idx = getattr(self, "_llada2_layer_idx", None)
                if layer_idx is not None:
                    _llada2_debug_store_nested("moe", f"layer_{layer_idx}_expert_gemm_output", out.clone())
            return out
        if impl == "naive":
            num_tokens, hidden_size = hidden_states.shape
            num_local_experts = w13.shape[0]
            intermediate_size = w13.shape[1] // 2
            final_hidden_states = hidden_states.new_zeros((num_tokens, hidden_size))
            local_topk_ids = topk_ids.to(torch.int64) - int(local_expert_start)
            
            for token_idx in range(num_tokens):
                token_hidden = hidden_states[token_idx]
                token_out = torch.zeros(hidden_size, device=hidden_states.device, dtype=torch.float32)

                for slot_idx in range(topk_ids.shape[1]):
                    local_expert_idx = int(local_topk_ids[token_idx, slot_idx].item())

                    if local_expert_idx < 0 or local_expert_idx >= num_local_experts:
                        continue

                    weight = topk_weights[token_idx, slot_idx]
                    if weight.item() == 0:
                        continue

                    expert_w13 = w13[local_expert_idx]
                    gate_proj = expert_w13[:intermediate_size]
                    up_proj = expert_w13[intermediate_size:]
                    
                    gate = torch.matmul(token_hidden, gate_proj.transpose(0, 1))
                    up = torch.matmul(token_hidden, up_proj.transpose(0, 1))
                    activated = F.silu(gate) * up
                    
                    expert_out = torch.matmul(activated, w2[local_expert_idx].transpose(0, 1))
                    token_out += expert_out.float() * weight.float()
                
                final_hidden_states[token_idx] = token_out.to(hidden_states.dtype)

            if _llada2_debug_enabled is not None and _llada2_debug_enabled():
                layer_idx = getattr(self, "_llada2_layer_idx", None)
                if layer_idx is not None:
                    _llada2_debug_store_nested(
                        "moe",
                        f"layer_{layer_idx}_expert_gemm_output",
                        final_hidden_states.clone(),
                    )
            return final_hidden_states
        raise ValueError(f"Unknown MoE expert_gemm impl: {impl}")


__all__ = ["FusedMoE", "SharedExpertMLP"]
