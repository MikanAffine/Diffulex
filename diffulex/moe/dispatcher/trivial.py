import torch
import torch.distributed as dist
import torch.nn.functional as F

from diffulex.moe.dispatcher.base import TokenDispatcher
from diffulex.moe.dispatcher.datatype import CombineInput, DispatchOutput
from diffulex.moe.topk import TopKOutput


class TrivialTokenDispatcher(TokenDispatcher):

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        *,
        hidden_states_scale: torch.Tensor | None = None,
    ) -> DispatchOutput:
        if topk_output.ids is None or topk_output.weights is None:
            raise RuntimeError("Token dispatch requires top-k ids and weights.")

        num_experts = topk_output.router_logits.shape[-1]
        expert_mask = F.one_hot(topk_output.ids, num_classes=num_experts).permute(2, 1, 0)
        expert_token_indices = []
        expert_topk_slot_indices = []

        for expert_idx in range(num_experts):
            topk_slot_idx, token_idx = torch.where(expert_mask[expert_idx])
            expert_token_indices.append(token_idx)
            expert_topk_slot_indices.append(topk_slot_idx)

        return DispatchOutput(
            hidden_states=hidden_states,
            topk_output=topk_output,
            expert_token_indices=tuple(expert_token_indices),
            expert_topk_slot_indices=tuple(expert_topk_slot_indices),
            hidden_states_scale=hidden_states_scale,
        )

    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        final_hidden_states = combine_input.expert_hidden_states[0].new_zeros(
            (combine_input.num_tokens, combine_input.hidden_size)
        )

        for expert_hidden_states, token_idx, topk_slot_idx in zip(
            combine_input.expert_hidden_states,
            combine_input.expert_token_indices,
            combine_input.expert_topk_slot_indices,
            strict=True,
        ):
            if token_idx.numel() == 0:
                continue
            routing_weights = combine_input.topk_weights[token_idx, topk_slot_idx].to(expert_hidden_states.dtype)
            final_hidden_states.index_add_(0, token_idx, expert_hidden_states * routing_weights.unsqueeze(-1))

        if dist.get_world_size() > 1:
            dist.all_reduce(final_hidden_states)

        return final_hidden_states
