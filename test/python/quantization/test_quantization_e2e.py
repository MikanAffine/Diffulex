"""
E2E tests: verify quantization strategy integration in real usage scenarios.
"""
import torch
from types import SimpleNamespace

from diffulex.utils.quantization.factory import QuantizationStrategyFactory
from diffulex.utils.quantization.context import get_kv_cache_strategy
from diffulex.attention.metadata import AttnMetaDataBase


def test_bf16_e2e():
    """E2E test: full BF16 path flow."""
    print("\n=== BF16 E2E test ===")
    
    # 1. Config init
    config = SimpleNamespace(
        kv_cache_dtype="bf16",
        attn_q_dtype="bf16",
    )
    ctx = QuantizationStrategyFactory.create_from_config(config)
    strategy = get_kv_cache_strategy()
    
    # 2. Verify storage dtype
    storage_dtype, itemsize = strategy.get_storage_dtype()
    assert storage_dtype == torch.bfloat16
    print(f"✓ Storage dtype: {storage_dtype}, itemsize: {itemsize}")
    
    # 3. Simulate KV cache allocation (like ModelRunner.allocate_kv_cache)
    num_layers = 2
    num_blocks = 4
    block_size = 32
    num_kv_heads = 8
    head_dim = 128
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Allocate KV cache (unified layout)
    kv_cache = torch.zeros(
        2, num_layers, num_blocks, block_size, num_kv_heads, head_dim,
        dtype=storage_dtype, device=device
    )
    print(f"✓ KV cache allocated: shape={kv_cache.shape}, dtype={kv_cache.dtype}")
    
    # 4. Verify no scales required
    k_scale_init, v_scale_init = strategy.init_scales(num_kv_heads, device)
    assert k_scale_init is None or v_scale_init is None
    print("✓ BF16 does not require scales")
    
    # 5. Simulate attention forward (like Attention.forward)
    seq_len = 16
    k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    
    # Simulate scale update (should be skipped for BF16)
    k_scale, v_scale = None, None
    if strategy.requires_kv_cache_scales:
        k_scale, v_scale = strategy.update_scales(k, v, k_scale, v_scale, num_kv_heads, device)
    assert k_scale is None or v_scale is None
    print("✓ Scale update correctly skipped")
    
    # 6. Simulate metadata setup
    attn_metadata = AttnMetaDataBase()
    strategy.maybe_set_attn_metadata_scales(attn_metadata, k_scale=k_scale, v_scale=v_scale)
    assert attn_metadata.k_scale is None
    assert attn_metadata.v_scale is None
    print("✓ Metadata scales not set (as expected)")
    
    # 7. Verify cache view (should return original cache)
    cache_view = strategy.view_kv_cache_for_kernels(kv_cache[0, 0, 0])
    assert cache_view is kv_cache[0, 0, 0] or torch.equal(cache_view, kv_cache[0, 0, 0])
    print("✓ Cache view correct (returns original cache)")
    
    print("✅ BF16 E2E test passed")


def test_fp8_e2e():
    """E2E test: full FP8 path flow."""
    print("\n=== FP8 E2E test ===")
    
    # Check FP8 support
    has_fp8 = hasattr(torch, "float8_e4m3fn") or hasattr(torch, "float8_e4m3fnuz")
    if not has_fp8:
        print("⚠ Current PyTorch does not support FP8, skipping FP8 E2E test")
        return True
    
    # 1. Config init
    config = SimpleNamespace(
        kv_cache_dtype="fp8",
        attn_q_dtype="bf16",
    )
    ctx = QuantizationStrategyFactory.create_from_config(config)
    strategy = get_kv_cache_strategy()
    
    # 2. Verify storage dtype
    storage_dtype, itemsize = strategy.get_storage_dtype()
    assert storage_dtype == torch.uint8
    assert itemsize == 1
    print(f"✓ Storage dtype: {storage_dtype}, itemsize: {itemsize}")
    
    # 3. Simulate KV cache allocation
    num_layers = 2
    num_blocks = 4
    block_size = 32
    num_kv_heads = 8
    head_dim = 128
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Allocate KV cache (unified layout, uint8 storage)
    kv_cache = torch.zeros(
        2, num_layers, num_blocks, block_size, num_kv_heads, head_dim,
        dtype=storage_dtype, device=device
    )
    print(f"✓ KV cache allocated: shape={kv_cache.shape}, dtype={kv_cache.dtype}")
    
    # 4. Allocate scales (like ModelRunner.allocate_kv_cache)
    k_scale_init, v_scale_init = strategy.init_scales(num_kv_heads, device)
    assert k_scale_init is not None and v_scale_init is not None
    
    k_scale = torch.zeros(num_layers, num_kv_heads, dtype=torch.float32, device=device)
    v_scale = torch.zeros(num_layers, num_kv_heads, dtype=torch.float32, device=device)
    k_scale[:] = k_scale_init[None, :]
    v_scale[:] = v_scale_init[None, :]
    print(f"✓ Scales allocated: k_scale={k_scale.shape}, v_scale={v_scale.shape}")
    
    # 5. Simulate attention forward
    seq_len = 16
    k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    
    # Simulate scale update (like logic in Attention.forward)
    layer_id = 0
    k_scale_layer = k_scale[layer_id]
    v_scale_layer = v_scale[layer_id]
    
    k_scale_updated, v_scale_updated = strategy.update_scales(
        k, v, k_scale_layer, v_scale_layer, num_kv_heads, device
    )
    assert k_scale_updated is not None and v_scale_updated is not None
    assert k_scale_updated.shape == (num_kv_heads,)
    assert v_scale_updated.shape == (num_kv_heads,)
    print(f"✓ Scale update: k_scale range=[{k_scale_updated.min():.4f}, {k_scale_updated.max():.4f}]")
    
    # Update global scales
    k_scale[layer_id] = k_scale_updated
    v_scale[layer_id] = v_scale_updated
    
    # 6. Simulate metadata setup (like Attention.forward)
    attn_metadata = AttnMetaDataBase()
    strategy.maybe_set_attn_metadata_scales(
        attn_metadata, k_scale=k_scale_layer, v_scale=v_scale_layer
    )
    assert attn_metadata.k_scale is not None
    assert attn_metadata.v_scale is not None
    print("✓ Metadata scales set")
    
    # 7. Verify cache view (should return float8 view)
    cache_view = strategy.view_kv_cache_for_kernels(kv_cache[0, 0, 0])
    assert cache_view.dtype != torch.uint8
    print(f"✓ Cache view dtype: {cache_view.dtype}")
    
    # 8. Simulate quantize_kv_for_store (like logic in store_kv_cache)
    k_quantized, v_quantized = strategy.quantize_kv_for_store(
        k, v, k_scale=k_scale_layer, v_scale=v_scale_layer
    )
    assert k_quantized.dtype == torch.uint8
    assert v_quantized.dtype == torch.uint8
    assert k_quantized.shape == k.shape
    assert v_quantized.shape == v.shape
    print(f"✓ KV quantized: k={k_quantized.shape}, v={v_quantized.shape}")
    
    print("✅ FP8 E2E test passed")


if __name__ == "__main__":
    print("Running E2E tests...")
    
    try:
        test_bf16_e2e()
        test_fp8_e2e()
        print("\n✅ All E2E tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

