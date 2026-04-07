import torch
import torch.nn.functional as F

from diffulex.moe.topk.base import TopKRouter
from diffulex.moe.topk.datatype import TopKOutput


class TrivialTopKRouter(TopKRouter):
    """Trivial pytorch implementation for reference"""

    def forward(self, router_logits: torch.Tensor) -> TopKOutput:
        if self.scoring_func == "softmax":
            routing_scores = F.softmax(router_logits, dim=-1, dtype=torch.float)
        elif self.scoring_func == "sigmoid":
            routing_scores = torch.sigmoid(router_logits.float())
        else:
            raise ValueError(f"Unsupported scoring function: {self.scoring_func!r}.")

        top_k = min(self.top_k, routing_scores.shape[-1])
        topk_weights, topk_ids = torch.topk(routing_scores, top_k, dim=-1)
        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        return TopKOutput(
            weights=topk_weights,
            ids=topk_ids,
            router_logits=router_logits,
        )
