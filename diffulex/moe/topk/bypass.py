import torch

from diffulex.moe.topk.base import TopKRouter
from diffulex.moe.topk.datatype import TopKOutput


class BypassTopKRouter(TopKRouter):
    """Bypass implemenation, use this if fused moe runner also handles topk"""

    def forward(self, router_logits: torch.Tensor) -> TopKOutput:
        return TopKOutput(
            weights=None,
            ids=None,
            router_logits=router_logits,
        )
