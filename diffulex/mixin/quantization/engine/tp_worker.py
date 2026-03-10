"""
Quantization mixin for TP worker.

Provides step-local activation quant cache clearing. The engine tp_worker should
inherit TPWorkerQuantizationMixin and call clear_step_act_quant_cache() at the
start of each step() (and step_async path).
"""

from diffulex.quantization.context import clear_act_quant_cache


class DiffulexTPWorkerQuantizationMixin:
    """Mixin for TP worker quantization: clear step-local activation quant cache."""

    def clear_step_act_quant_cache(self) -> None:
        """Clear step-local activation quant cache. Call at the start of each step."""
        clear_act_quant_cache()
