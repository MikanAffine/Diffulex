from diffulex_legacy.layers.attention.ops.triton_decode_attn_clm import (
    causal_lm_decode_attention_fwd as causal_lm_flash_decoding,
)
from diffulex_legacy.layers.attention.ops.triton_decode_attn_dlm import (
    diffusion_lm_flash_decoding,
    CHECK_ATTENTION,
)
from diffulex_legacy.layers.attention.ops.chunked_prefill_decoding_unified_kernel import (
    chunked_prefill_paged_decode as diffusion_lm_parallel_flash_decoding,
)
from diffulex_legacy.layers.attention.ops.kv_cache_kernels import (
    store_kv_cache_distinct_layout,
    store_kv_cache_unified_layout,
    load_kv_cache,
    CHECK_STORING,
    CHECK_LOADING,
)
