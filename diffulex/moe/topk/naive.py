import os

import torch
import torch.nn.functional as F

from diffulex.moe.topk.base import TopKRouter
from diffulex.moe.topk.output import TopKOutput
from diffulex_kernel import fused_topk


def _tile_kernels_topk_gate(scores: torch.Tensor, top_k: int) -> torch.Tensor | None:
    try:
        import tile_kernels  # type: ignore
    except Exception:
        return None
    try:
        return tile_kernels.moe.topk_gate(scores, top_k)
    except Exception:
        return None


class NaiveTopKRouter(TopKRouter):
    def _forward_naive(self, router_logits: torch.Tensor) -> TopKOutput:
        if self.scoring_func == "softmax":
            routing_scores = F.softmax(router_logits, dim=-1, dtype=torch.float)
        elif self.scoring_func == "sigmoid":
            routing_scores = torch.sigmoid(router_logits.float())
        else:
            raise ValueError(f"Unsupported scoring function: {self.scoring_func!r}.")

        top_k = min(self.top_k, routing_scores.shape[-1])
        topk_ids = None
        if (
            router_logits.is_cuda
            and os.getenv("DIFFULEX_MOE_TOPK_IMPL", "").lower() in {"tile", "tilekernels", "tile_kernels"}
        ):
            topk_ids = _tile_kernels_topk_gate(routing_scores, top_k)

        if topk_ids is None:
            topk_weights, topk_ids = torch.topk(routing_scores, top_k, dim=-1, sorted=False)
        else:
            topk_ids = topk_ids.to(torch.int64)
            topk_weights = torch.gather(routing_scores, dim=-1, index=topk_ids)
        if self.renormalize and top_k > 1:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)

        return TopKOutput(
            weights=topk_weights,
            ids=topk_ids.to(torch.int32),
            router_logits=router_logits,
        )

    def forward(self, router_logits: torch.Tensor) -> TopKOutput:
        if not router_logits.is_cuda or os.getenv("DIFFULEX_REFERENCE_MOE_ROUTER", "0") == "1":
            return self._forward_naive(router_logits)

        topk_weights, topk_ids = fused_topk(
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            scoring_func=self.scoring_func,
        )
        return TopKOutput(weights=topk_weights, ids=topk_ids, router_logits=router_logits)
