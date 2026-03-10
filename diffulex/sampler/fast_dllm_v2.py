import torch

from diffulex.sampler.auto_sampler import AutoSampler
from diffulex.sampler.base import DllmSamplerShiftBase


@AutoSampler.register("fast_dllm_v2")
class FastdLLMV2Sampler(DllmSamplerShiftBase):
    def _compute_accepted_ids(
        self,
        block,
        confidence: torch.Tensor,
        initial_confidence: torch.Tensor,
        sampled_tokens: torch.Tensor,
        *,
        threshold: float = 0.95,
        **kwargs,
    ) -> torch.Tensor:
        high_conf_indices = torch.where(initial_confidence > threshold)[0]
        if len(high_conf_indices) == 0:
            max_prob_idx = initial_confidence.argmax()
            return max_prob_idx.view(1)
        max_prob_idx = initial_confidence.argmax()
        return torch.unique(torch.cat([high_conf_indices, max_prob_idx.view(1)]))
