import os

import torch
import torch.nn as nn

from diffulex.utils.profiler import trace


def _use_reference_rmsnorm() -> bool:
    return os.getenv("DIFFULEX_REFERENCE_RMSNORM", "0") == "1"


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    def rms_forward_reference(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_fp32 = x_fp32 * torch.rsqrt(var + self.eps)
        return x_fp32.to(orig_dtype) * self.weight

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.to(torch.float32).add_(residual.to(torch.float32))
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def add_rms_forward_reference(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x_fp32 = x.to(torch.float32) + residual.to(torch.float32)
        residual_out = x_fp32.to(orig_dtype)
        var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_fp32 = x_fp32 * torch.rsqrt(var + self.eps)
        return x_fp32.to(orig_dtype) * self.weight, residual_out

    @trace
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if _use_reference_rmsnorm():
            if residual is None:
                return self.rms_forward_reference(x)
            return self.add_rms_forward_reference(x, residual)
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
