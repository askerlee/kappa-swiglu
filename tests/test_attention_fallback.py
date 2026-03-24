"""
Test Flash Attention unified interface against the SDPA fallback.

Run: python -m pytest tests/test_attention_fallback.py -v -s

Note on test structure:
    Tests are split into two classes due to dtype/device constraints:

    1. TestFlashAttentionVsSDPA: Comparison tests that run both an accelerated
       backend and SDPA on the same inputs and verify they produce identical
       results. These require a supported GPU backend and use bfloat16.

    2. TestSDPAOnly: Tests that only exercise the SDPA fallback path. These can
       run on any device (CUDA, CPU, MPS) with the appropriate dtype.
"""
import pytest
import torch

import nanochat.flash_attention as fa_module
from nanochat.engine import KVCache
from nanochat.flash_attention import (
    FLASH_ATTN_BACKEND,
    HAS_FA3,
    HAS_FA4,
    HAS_FLASH_ATTN,
    HAS_FLASH_ATTN_KVCACHE,
    flash_attn,
)


def set_impl(impl):
    """Set the implementation override ('flash', 'fa3', 'fa4', 'sdpa', or None)."""
    fa_module._override_impl = impl


@pytest.fixture(autouse=True)
def reset_flash_override():
    previous = fa_module._override_impl
    try:
        yield
    finally:
        fa_module._override_impl = previous


def run_both_impls(fn):
    """Run a function with both Flash Attention and SDPA, return both outputs."""
    set_impl('flash')
    out_flash = fn()
    set_impl('sdpa')
    out_sdpa = fn()
    set_impl(None)
    return out_flash, out_sdpa


def assert_close(t1, t2, name, atol=1e-2, rtol=1e-2):
    """Assert two tensors are close, with helpful error message."""
    max_diff = (t1 - t2).abs().max().item()
    mean_diff = (t1 - t2).abs().mean().item()
    assert torch.allclose(t1, t2, atol=atol, rtol=rtol), (
        f"{name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
    )
    return max_diff, mean_diff


# =============================================================================
# Flash Attention vs SDPA comparison tests (require accelerated backend)
# =============================================================================
@pytest.mark.skipif(not HAS_FLASH_ATTN, reason="Flash Attention backend required to compare implementations")
class TestFlashAttentionVsSDPA:
    """Compare accelerated Flash Attention and SDPA outputs on supported GPUs."""

    DEVICE = "cuda"
    DTYPE = torch.bfloat16

    def test_basic_causal(self):
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))

        y_flash, y_sdpa = run_both_impls(run)
        max_diff, mean_diff = assert_close(y_flash, y_sdpa, "basic_causal")
        print(f"basic_causal: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_full_context(self):
        B, T, H, D = 2, 128, 4, 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))

        y_flash, y_sdpa = run_both_impls(run)
        max_diff, mean_diff = assert_close(y_flash, y_sdpa, "full_context")
        print(f"full_context: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_sliding_window(self):
        B, T, H, D = 2, 128, 4, 32
        window = 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(window, 0))

        y_flash, y_sdpa = run_both_impls(run)
        max_diff, mean_diff = assert_close(y_flash, y_sdpa, "sliding_window")
        print(f"sliding_window: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_gqa(self):
        B, T, D = 2, 64, 32
        n_heads = 8
        n_kv_heads = 2

        q = torch.randn(B, T, n_heads, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, n_kv_heads, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, n_kv_heads, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))

        y_flash, y_sdpa = run_both_impls(run)
        max_diff, mean_diff = assert_close(y_flash, y_sdpa, "gqa")
        print(f"gqa: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_larger_model(self):
        B, T, H, D = 4, 256, 12, 64
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))

        y_flash, y_sdpa = run_both_impls(run)
        max_diff, mean_diff = assert_close(y_flash, y_sdpa, "larger_model")
        print(f"larger_model: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    @pytest.mark.skipif(not HAS_FLASH_ATTN_KVCACHE, reason="Accelerated KV-cache backend required")
    def test_kvcache_prefill(self):
        B, T_max, H, D = 2, 64, 4, 32
        T_prefill = 16

        q = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            k_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            v_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            cache_seqlens = torch.zeros(B, dtype=torch.int32, device=self.DEVICE)
            return flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache, k=k, v=v,
                cache_seqlens=cache_seqlens,
                causal=True, window_size=(T_max, 0),
            )

        y_flash, y_sdpa = run_both_impls(run)
        max_diff, mean_diff = assert_close(y_flash, y_sdpa, "prefill")
        print(f"prefill: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    @pytest.mark.skipif(not HAS_FLASH_ATTN_KVCACHE, reason="Accelerated KV-cache backend required")
    def test_kvcache_single_token(self):
        B, T_max, H, D = 2, 64, 4, 32
        T_prefill = 16

        k_init = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_init = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        q_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            k_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            v_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            k_cache[:, :T_prefill, :, :] = k_init
            v_cache[:, :T_prefill, :, :] = v_init
            cache_seqlens = torch.full((B,), T_prefill, dtype=torch.int32, device=self.DEVICE)
            return flash_attn.flash_attn_with_kvcache(
                q_single, k_cache, v_cache, k=k_single, v=v_single,
                cache_seqlens=cache_seqlens,
                causal=True, window_size=(T_max, 0),
            )

        y_flash, y_sdpa = run_both_impls(run)
        max_diff, mean_diff = assert_close(y_flash, y_sdpa, "single_token")
        print(f"single_token: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    @pytest.mark.skipif(not HAS_FLASH_ATTN_KVCACHE, reason="Accelerated KV-cache backend required")
    def test_kvcache_single_token_sliding_window(self):
        B, T_max, H, D = 2, 64, 4, 32
        T_prefill = 32
        window = 8

        k_init = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_init = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        q_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            k_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            v_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            k_cache[:, :T_prefill, :, :] = k_init
            v_cache[:, :T_prefill, :, :] = v_init
            cache_seqlens = torch.full((B,), T_prefill, dtype=torch.int32, device=self.DEVICE)
            return flash_attn.flash_attn_with_kvcache(
                q_single, k_cache, v_cache, k=k_single, v=v_single,
                cache_seqlens=cache_seqlens,
                causal=True, window_size=(window, 0),
            )

        y_flash, y_sdpa = run_both_impls(run)
        max_diff, mean_diff = assert_close(y_flash, y_sdpa, "single_token_sliding_window")
        print(f"single_token_sliding_window: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_backward_gradients_match(self):
        B, T, H, D = 2, 32, 4, 16

        q_data = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_data = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_data = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            q = q_data.clone().requires_grad_(True)
            k = k_data.clone().requires_grad_(True)
            v = v_data.clone().requires_grad_(True)
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))
            loss = y.sum()
            loss.backward()
            return y.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach()

        set_impl('flash')
        y_flash, q_grad_flash, k_grad_flash, v_grad_flash = run()
        set_impl('sdpa')
        y_sdpa, q_grad_sdpa, k_grad_sdpa, v_grad_sdpa = run()
        set_impl(None)

        max_diff, mean_diff = assert_close(y_flash, y_sdpa, "backward_output")
        print(f"backward_output: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        max_diff, mean_diff = assert_close(q_grad_flash, q_grad_sdpa, "q_grad", atol=0.05, rtol=0.05)
        print(f"q_grad: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        max_diff, mean_diff = assert_close(k_grad_flash, k_grad_sdpa, "k_grad", atol=0.05, rtol=0.05)
        print(f"k_grad: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        max_diff, mean_diff = assert_close(v_grad_flash, v_grad_sdpa, "v_grad", atol=0.05, rtol=0.05)
        print(f"v_grad: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")


# =============================================================================
# SDPA-only tests (run on any device)
# =============================================================================
class TestSDPAOnly:
    """Test SDPA fallback works correctly. Runs on any device."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def test_basic_forward(self):
        set_impl('sdpa')
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))

        assert y.shape == (B, T, H, D)
        assert not torch.isnan(y).any(), "Output contains NaN"
        set_impl(None)

    def test_backward(self):
        set_impl('sdpa')
        B, T, H, D = 2, 32, 4, 16
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)

        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))
        loss = y.sum()
        loss.backward()

        assert q.grad is not None, "No gradient for q"
        assert k.grad is not None, "No gradient for k"
        assert v.grad is not None, "No gradient for v"
        assert not torch.isnan(q.grad).any(), "NaN in q gradient"
        set_impl(None)

    def test_kvcache(self):
        set_impl('sdpa')
        B, T_max, H, D = 2, 64, 4, 32
        n_layers = 1

        cache = KVCache(
            batch_size=B,
            num_heads=H,
            seq_len=T_max,
            head_dim=D,
            num_layers=n_layers,
            device=self.DEVICE,
            dtype=self.DTYPE,
        )
        k_cache, v_cache = cache.get_layer_cache(0)

        T_prefill = 16
        q = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y = flash_attn.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v,
            cache_seqlens=cache.cache_seqlens,
            causal=True, window_size=(T_max, 0),
        )
        cache.advance(T_prefill)

        assert y.shape == (B, T_prefill, H, D)
        assert cache.get_pos() == T_prefill

        q_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y_single = flash_attn.flash_attn_with_kvcache(
            q_single, k_cache, v_cache, k=k_single, v=v_single,
            cache_seqlens=cache.cache_seqlens,
            causal=True, window_size=(T_max, 0),
        )
        cache.advance(1)

        assert y_single.shape == (B, 1, H, D)
        assert cache.get_pos() == T_prefill + 1
        set_impl(None)


# =============================================================================
# Override mechanism tests
# =============================================================================
class TestOverrideMechanism:
    """Test that the override mechanism works correctly."""

    @pytest.mark.skipif(not HAS_FLASH_ATTN, reason="Flash Attention backend required")
    def test_override_flash(self):
        set_impl('flash')
        assert fa_module._use_flash_attention() is True
        set_impl(None)

    @pytest.mark.skipif(not HAS_FA3, reason="FA3 backend required")
    def test_override_fa3(self):
        set_impl('fa3')
        assert fa_module._use_flash_attention() is True
        set_impl(None)

    @pytest.mark.skipif(not HAS_FA4, reason="FA4 backend required")
    def test_override_fa4(self):
        set_impl('fa4')
        assert fa_module._use_flash_attention() is True
        set_impl(None)

    def test_override_sdpa(self):
        set_impl('sdpa')
        assert fa_module._use_flash_attention() is False
        set_impl(None)

    def test_override_auto(self):
        set_impl(None)
        assert fa_module._use_flash_attention() == HAS_FLASH_ATTN


class TestBackendOutputNormalization:
    def test_flash_backend_tuple_output_is_unwrapped(self, monkeypatch):
        out = torch.randn(2, 4, 3, 8)
        lse = torch.randn(2, 3, 4)

        backend = fa_module._make_backend(
            name='fa4',
            flash_attn_func=lambda *args, **kwargs: (out, lse),
        )

        monkeypatch.setattr(fa_module, '_backend', backend)
        monkeypatch.setattr(fa_module, 'HAS_FLASH_ATTN', True)
        monkeypatch.setattr(fa_module, 'FLASH_ATTN_BACKEND', 'fa4')
        monkeypatch.setattr(fa_module, 'HAS_FA3', False)
        monkeypatch.setattr(fa_module, 'HAS_FA4', True)
        monkeypatch.setattr(fa_module, 'HAS_FLASH_ATTN_KVCACHE', False)
        set_impl('flash')

        q = torch.randn(2, 4, 3, 8)
        y = fa_module.flash_attn_func(q, q, q, causal=True, window_size=(4, 0))

        assert y is out

    def test_kvcache_backend_tuple_output_is_unwrapped(self, monkeypatch):
        out = torch.randn(2, 1, 3, 8)
        lse = torch.randn(2, 3, 1)

        backend = fa_module._make_backend(
            name='fa4',
            flash_attn_func=lambda *args, **kwargs: out,
            flash_attn_with_kvcache=lambda *args, **kwargs: (out, lse),
        )

        monkeypatch.setattr(fa_module, '_backend', backend)
        monkeypatch.setattr(fa_module, 'HAS_FLASH_ATTN', True)
        monkeypatch.setattr(fa_module, 'FLASH_ATTN_BACKEND', 'fa4')
        monkeypatch.setattr(fa_module, 'HAS_FA3', False)
        monkeypatch.setattr(fa_module, 'HAS_FA4', True)
        monkeypatch.setattr(fa_module, 'HAS_FLASH_ATTN_KVCACHE', True)
        set_impl('flash')

        q = torch.randn(2, 1, 3, 8)
        k_cache = torch.randn(2, 4, 3, 8)
        v_cache = torch.randn(2, 4, 3, 8)
        cache_seqlens = torch.zeros(2, dtype=torch.int32)
        y = fa_module.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=q,
            v=q,
            cache_seqlens=cache_seqlens,
            causal=True,
            window_size=(4, 0),
        )

        assert y is out


if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        major, minor = torch.cuda.get_device_capability()
        print(f"Compute capability: {major}.{minor}")
    print(f"HAS_FLASH_ATTN: {HAS_FLASH_ATTN}")
    print(f"HAS_FLASH_ATTN_KVCACHE: {HAS_FLASH_ATTN_KVCACHE}")
    print(f"FLASH_ATTN_BACKEND: {FLASH_ATTN_BACKEND}")
    print(f"HAS_FA3: {HAS_FA3}")
    print(f"HAS_FA4: {HAS_FA4}")
    print()

    pytest.main([__file__, "-v", "-s"])
