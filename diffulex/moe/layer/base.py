from abc import ABC

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
    ) -> None:
        super().__init__()

        if hidden_act != "silu":
            raise NotImplementedError("only silu is supported currently")
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.hidden_act = hidden_act
        self.topk = top_k
        self.topk_renormalize = norm_topk_prob
        self.topk_scoring_func = "softmax"

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
        )
    
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # return final_hidden_states, router_logits
        raise NotImplementedError

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

        phase = self._phase_from_prefill_flags(getattr(attn_metadata, "is_prefill", None))
        if phase != "unknown":
            return phase

        status_table = getattr(attn_metadata, "status_table", None)
        if status_table is None:
            return "unknown"

        if torch.is_tensor(status_table):
            return self._phase_from_prefill_flags(status_table == 0)

        try:
            return self._phase_from_prefill_flags([int(status) == 0 for status in status_table])
        except TypeError:
            return self._phase_from_prefill_flags(int(status_table) == 0)

__all__ = ["FusedMoE"]
