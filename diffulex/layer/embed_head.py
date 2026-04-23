import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os

from diffulex.distributed.parallel_state import fetch_parallel_state


LM_HEAD_FP32 = os.environ.get("DIFFULEX_LM_HEAD_FP32", "0") == "1"
LM_HEAD_FP32_GATHER = LM_HEAD_FP32 or os.environ.get("DIFFULEX_LM_HEAD_FP32_GATHER", "0") == "1"


class VocabParallelEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        parallel_state = fetch_parallel_state()
        self.tp_rank = parallel_state.get_tp_rank()
        self.tp_size = parallel_state.get_tp_world_size()
        self.tp_group = parallel_state.get_tp_group()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y, group=self.tp_group)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        super().__init__(num_embeddings, embedding_dim)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_embeddings_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        if LM_HEAD_FP32:
            logits = F.linear(
                x.to(torch.float32),
                self.weight.to(torch.float32),
                self.bias.to(torch.float32) if self.bias is not None else None,
            ).to(x.dtype)
        else:
            logits = F.linear(x, self.weight, self.bias)
        if self.tp_size > 1:
            if LM_HEAD_FP32_GATHER:
                gather_input = logits.to(torch.float32)
                gathered_logits = [torch.empty_like(gather_input) for _ in range(self.tp_size)]
                dist.all_gather(gathered_logits, gather_input, group=self.tp_group)
                logits = torch.cat(gathered_logits, -1).to(logits.dtype) if self.tp_rank == 0 else None
            else:
                gathered_logits = [torch.empty_like(logits) for _ in range(self.tp_size)]
                dist.all_gather(gathered_logits, logits, group=self.tp_group)
                logits = torch.cat(gathered_logits, -1) if self.tp_rank == 0 else None
        return logits
