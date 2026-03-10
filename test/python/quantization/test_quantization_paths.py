"""
Verify quantization paths: bf16 path and bf16+fp8 kv path.
"""
import torch
from types import SimpleNamespace

from diffulex.utils.quantization.factory import QuantizationStrategyFactory
from diffulex.utils.quantization.context import (
    get_kv_cache_strategy,
    get_attn_q_strategy,
    QuantizationContext,
)


def test_bf16_path():
    """Test bf16 path (default, no quantization)."""
    print("\n=== Test BF16 path ===")
    
    # Create config
    config = SimpleNamespace(
        kv_cache_dtype="bf16",
        attn_q_dtype="bf16",
    )
    
    # Initialize quantization context
    ctx = QuantizationStrategyFactory.create_from_config(config)
    
    # Get strategies
    kv_strategy = get_kv_cache_strategy()
    attn_q_strategy = get_attn_q_strategy()
    
    assert kv_strategy is not None, "KV cache strategy should be created"
    assert attn_q_strategy is not None, "Attn-Q strategy should be created"
    
    print(f"KV Cache strategy: {kv_strategy.name}")
    print(f"KV Cache format: {kv_strategy.kv_cache_format}")
    print(f"Requires scales: {kv_strategy.requires_kv_cache_scales}")
    
    print(f"Attn-Q strategy: {attn_q_strategy.name}")
    print(f"Attn-Q format: {attn_q_strategy.attn_q_format}")
    
    # Verify storage dtype
    storage_dtype, itemsize = kv_strategy.get_storage_dtype()
    assert storage_dtype == torch.bfloat16, f"Expected bfloat16, got {storage_dtype}"
    assert itemsize == 2, f"Expected itemsize=2, got {itemsize}"
    print(f"Storage dtype: {storage_dtype}, itemsize: {itemsize}")
    
    # Verify no scales required
    assert not kv_strategy.requires_kv_cache_scales, "BF16 should not require scales"
    
    # Verify scale initialization
    num_kv_heads = 8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    k_scale, v_scale = kv_strategy.init_scales(num_kv_heads, device)
    assert k_scale is None or v_scale is None, "BF16 strategy should return None scales"
    print("✓ BF16 path verified")


def test_bf16_with_fp8_kv_path():
    """Test bf16 + fp8 kv path."""
    print("\n=== Test BF16 + FP8 KV path ===")
    
    # Check FP8 support
    has_fp8 = hasattr(torch, "float8_e4m3fn") or hasattr(torch, "float8_e4m3fnuz")
    if not has_fp8:
        print("⚠ Current PyTorch does not support FP8, skipping FP8 test")
        return True
    
    # Create config
    config = SimpleNamespace(
        kv_cache_dtype="fp8",  # or "fp8_e4m3"
        attn_q_dtype="bf16",
    )
    
    # Initialize quantization context
    ctx = QuantizationStrategyFactory.create_from_config(config)
    
    # Get strategies
    kv_strategy = get_kv_cache_strategy()
    attn_q_strategy = get_attn_q_strategy()
    
    assert kv_strategy is not None, "KV cache strategy should be created"
    assert attn_q_strategy is not None, "Attn-Q strategy should be created"
    
    print(f"KV Cache strategy: {kv_strategy.name}")
    print(f"KV Cache format: {kv_strategy.kv_cache_format}")
    print(f"Requires scales: {kv_strategy.requires_kv_cache_scales}")
    
    print(f"Attn-Q strategy: {attn_q_strategy.name}")
    print(f"Attn-Q format: {attn_q_strategy.attn_q_format}")
    
    # Verify storage dtype (FP8 should use uint8)
    storage_dtype, itemsize = kv_strategy.get_storage_dtype()
    assert storage_dtype == torch.uint8, f"Expected uint8, got {storage_dtype}"
    assert itemsize == 1, f"Expected itemsize=1, got {itemsize}"
    print(f"Storage dtype: {storage_dtype}, itemsize: {itemsize}")
    
    # Verify scales required
    assert kv_strategy.requires_kv_cache_scales, "FP8 should require scales"
    
    # Verify scale initialization
    num_kv_heads = 8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    k_scale, v_scale = kv_strategy.init_scales(num_kv_heads, device)
    assert k_scale is not None and v_scale is not None, "FP8 strategy should return non-None scales"
    assert k_scale.shape == (num_kv_heads,), f"k_scale shape should be ({num_kv_heads},), got {k_scale.shape}"
    assert v_scale.shape == (num_kv_heads,), f"v_scale shape should be ({num_kv_heads},), got {v_scale.shape}"
    print(f"Initial scales shape: k_scale={k_scale.shape}, v_scale={v_scale.shape}")
    
    # Verify scale update logic
    seq_len = 32
    head_dim = 128
    k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    
    # First update (from None)
    k_scale_new, v_scale_new = kv_strategy.update_scales(
        k, v, None, None, num_kv_heads, device
    )
    assert k_scale_new is not None and v_scale_new is not None
    assert k_scale_new.shape == (num_kv_heads,)
    assert v_scale_new.shape == (num_kv_heads,)
    print(f"First scale update: k_scale range=[{k_scale_new.min():.4f}, {k_scale_new.max():.4f}]")
    
    # Second update (using existing scales)
    k_scale_updated, v_scale_updated = kv_strategy.update_scales(
        k, v, k_scale_new, v_scale_new, num_kv_heads, device
    )
    assert k_scale_updated is not None and v_scale_updated is not None
    print(f"Second scale update: k_scale range=[{k_scale_updated.min():.4f}, {k_scale_updated.max():.4f}]")
    
    # Verify view_kv_cache_for_kernels
    cache_u8 = torch.empty((16,), dtype=torch.uint8, device=device)
    cache_view = kv_strategy.view_kv_cache_for_kernels(cache_u8)
    assert cache_view.dtype != torch.uint8, "View should return non-uint8 dtype"
    print(f"Cache view dtype: {cache_view.dtype}")
    
    # Verify quantize_kv_for_store
    k_quantized, v_quantized = kv_strategy.quantize_kv_for_store(
        k, v, k_scale=k_scale_new, v_scale=v_scale_new
    )
    assert k_quantized.dtype == torch.uint8, f"Quantized K should be uint8, got {k_quantized.dtype}"
    assert v_quantized.dtype == torch.uint8, f"Quantized V should be uint8, got {v_quantized.dtype}"
    assert k_quantized.shape == k.shape, "Quantized K shape should be unchanged"
    assert v_quantized.shape == v.shape, "Quantized V shape should be unchanged"
    print(f"Quantized shape: k={k_quantized.shape}, v={v_quantized.shape}")
    
    print("✓ BF16 + FP8 KV path verified")


def test_metadata_integration():
    """Test integration with AttnMetaData."""
    print("\n=== Test Metadata integration ===")
    
    from diffulex.attention.metadata import AttnMetaDataBase
    
    # BF16 path
    config_bf16 = SimpleNamespace(kv_cache_dtype="bf16", attn_q_dtype="bf16")
    QuantizationStrategyFactory.create_from_config(config_bf16)
    kv_strategy_bf16 = get_kv_cache_strategy()
    
    md_bf16 = AttnMetaDataBase()
    kv_strategy_bf16.maybe_set_attn_metadata_scales(md_bf16, k_scale=None, v_scale=None)
    assert md_bf16.k_scale is None, "BF16 should not set scales"
    assert md_bf16.v_scale is None, "BF16 should not set scales"
    print("✓ BF16 metadata integration OK")
    
    # FP8 path (if supported)
    has_fp8 = hasattr(torch, "float8_e4m3fn") or hasattr(torch, "float8_e4m3fnuz")
    if has_fp8:
        config_fp8 = SimpleNamespace(kv_cache_dtype="fp8", attn_q_dtype="bf16")
        QuantizationStrategyFactory.create_from_config(config_fp8)
        kv_strategy_fp8 = get_kv_cache_strategy()
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        k_scale = torch.ones((8,), dtype=torch.float32, device=device)
        v_scale = torch.ones((8,), dtype=torch.float32, device=device) * 2
        
        md_fp8 = AttnMetaDataBase()
        kv_strategy_fp8.maybe_set_attn_metadata_scales(md_fp8, k_scale=k_scale, v_scale=v_scale)
        assert md_fp8.k_scale is k_scale, "FP8 should set k_scale"
        assert md_fp8.v_scale is v_scale, "FP8 should set v_scale"
        print("✓ FP8 metadata integration OK")


if __name__ == "__main__":
    print("Verifying quantization paths...")
    
    try:
        test_bf16_path()
        test_bf16_with_fp8_kv_path()
        test_metadata_integration()
        print("\n✅ All path checks passed!")
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

