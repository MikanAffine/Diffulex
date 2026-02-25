"""Re-export from diffulex.quantization.strategy."""
from diffulex.quantization.strategy import (
    QuantizationStrategy,
    KVCacheQuantizationStrategy,
    WeightQuantizationStrategy,
    LinearQuantizationStrategy,
)

__all__ = [
    "QuantizationStrategy",
    "kv_cacheQuantizationStrategy",
    "WeightQuantizationStrategy",
    "LinearQuantizationStrategy",
]
