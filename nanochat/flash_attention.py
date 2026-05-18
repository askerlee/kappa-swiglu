"""
Unified Flash Attention interface with automatic FA4/FA3/SDPA switching.

Exports `flash_attn` with the same surface area this codebase needs, while
selecting the best available backend at import time:
- Flash Attention 4 on Blackwell / newer GPUs when installed
- Flash Attention 3 on Hopper when available
- PyTorch SDPA everywhere else

Environment overrides:
- NANOCHAT_ATTENTION_IMPL=sdpa|flash|fa3|fa4
- NANOCHAT_ALLOW_FA4_TRAINING=1 to permit FA4 autograd during training

Usage:
    from nanochat.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
import os
import logging
from types import SimpleNamespace

import torch
import torch.nn.functional as F


# =============================================================================
# Detection: Try to load the best available Flash Attention backend
# =============================================================================
def _make_backend(name, flash_attn_func, flash_attn_with_kvcache=None):
    return SimpleNamespace(
        name=name,
        flash_attn_func=flash_attn_func,
        flash_attn_with_kvcache=flash_attn_with_kvcache,
    )


def _unwrap_backend_output(result):
    """Normalize backend outputs to the tensor that callers expect."""
    if isinstance(result, tuple):
        if not result:
            raise RuntimeError("Flash Attention backend returned an empty tuple")
        return result[0]
    return result


def _is_fa4_window_size_type_error(exc):
    """Detect FA4/CUTLASS window-size typing issues and trigger SDPA fallback."""
    msg = str(exc)
    return (
        "window_size_left" in msg
        and "expects argument" in msg
        and "got <class 'int'>" in msg
    )


def _run_with_quiet_hf_request_logs(func):
    """Suppress request-level HF/httpx info logs during kernel resolution."""
    logger_names = ("httpx", "huggingface_hub")
    previous_levels = {}
    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)
        previous_levels[logger_name] = logger.level
        if logger.isEnabledFor(logging.INFO):
            logger.setLevel(logging.WARNING)
    try:
        return func()
    finally:
        for logger_name, level in previous_levels.items():
            logging.getLogger(logger_name).setLevel(level)


def _format_cuda_capability(major, minor):
    return f"sm{major}{minor}"


def _load_flash_attention_4():
    """Try to load Flash Attention 4 (optimized for Hopper / Blackwell)."""
    try:
        from flash_attn.cute import flash_attn_func as flash_attn_func_fa4  # type: ignore[import-not-found]
    except Exception as exc:
        return None, f"Flash Attention 4 import failed: {exc}"

    flash_attn_with_kvcache = None
    try:
        from flash_attn import flash_attn_with_kvcache as flash_attn_with_kvcache_fa  # type: ignore[import-not-found]
        flash_attn_with_kvcache = flash_attn_with_kvcache_fa
    except Exception:
        pass

    return _make_backend(
        name='fa4',
        flash_attn_func=flash_attn_func_fa4,
        flash_attn_with_kvcache=flash_attn_with_kvcache,
    ), None


def _load_flash_attention_3():
    """Try to load Flash Attention 3 (requires Hopper GPU, sm90)."""
    if not torch.cuda.is_available():
        return None, "torch.cuda.is_available() is False"
    try:
        major, minor = torch.cuda.get_device_capability()
        # FA3 kernels are compiled for Hopper (sm90) only
        # Ada (sm89), Blackwell (sm100) need SDPA fallback until FA3 is recompiled
        if major != 9:
            return None, (
                f"Flash Attention 3 requires Hopper (sm90), got {_format_cuda_capability(major, minor)}"
            )
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        interface = _run_with_quiet_hf_request_logs(
            lambda: get_kernel('varunneal/flash-attention-3').flash_attn_interface
        )
        return _make_backend(
            name='fa3',
            flash_attn_func=interface.flash_attn_func,
            flash_attn_with_kvcache=interface.flash_attn_with_kvcache,
        ), None
    except Exception as exc:
        return None, f"Flash Attention 3 kernel load failed: {exc}"


def _load_flash_attention_backend():
    """Load the best Flash Attention backend for the current GPU."""
    if not torch.cuda.is_available():
        return None, "torch.cuda.is_available() is False"

    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception as exc:
        return None, f"Could not query CUDA device capability: {exc}"

    if major >= 10:
        backend, reason = _load_flash_attention_4()
        if backend is not None:
            return backend, None
        return None, reason

    if major == 9:
        reasons = []
        for loader in (_load_flash_attention_3, _load_flash_attention_4):
            backend, reason = loader()
            if backend is not None:
                return backend, None
            if reason:
                reasons.append(reason)
        return None, "; ".join(reasons)

    return None, (
        f"No supported Flash Attention backend for GPU {_format_cuda_capability(major, minor)}"
    )


_backend, FLASH_ATTN_UNAVAILABLE_REASON = _load_flash_attention_backend()
HAS_FLASH_ATTN = _backend is not None
FLASH_ATTN_BACKEND = _backend.name if _backend is not None else None
HAS_FLASH_ATTN_KVCACHE = HAS_FLASH_ATTN and _backend.flash_attn_with_kvcache is not None

# Backward-compatible flags for older callers.
HAS_FA3 = FLASH_ATTN_BACKEND == 'fa3'
HAS_FA4 = FLASH_ATTN_BACKEND == 'fa4'

# Override for testing: set to 'flash', 'fa3', 'fa4', 'sdpa', or None (auto)
_override_impl = os.environ.get("NANOCHAT_ATTENTION_IMPL") or None
_flash_disabled_due_to_oom = False
ALLOW_FA4_TRAINING = os.environ.get("NANOCHAT_ALLOW_FA4_TRAINING", "0") == "1"


def _use_flash_attention(require_kvcache=False):
    """Determine whether to use an accelerated backend based on availability."""
    if _flash_disabled_due_to_oom:
        return False
    if _override_impl in ('flash', 'fa3', 'fa4'):
        assert HAS_FLASH_ATTN, "Cannot override to Flash Attention: no backend is available"
        if _override_impl in ('fa3', 'fa4'):
            assert FLASH_ATTN_BACKEND == _override_impl, (
                f"Cannot override to {_override_impl}: active backend is {FLASH_ATTN_BACKEND!r}"
            )
        if require_kvcache:
            assert HAS_FLASH_ATTN_KVCACHE, (
                f"Cannot override to {FLASH_ATTN_BACKEND}: KV-cache API is not available"
            )
        return True
    if _override_impl == 'sdpa':
        return False
    if not HAS_FLASH_ATTN:
        return False
    if require_kvcache and not HAS_FLASH_ATTN_KVCACHE:
        return False
    return True


def _should_skip_fa4_for_training(q, k, v):
    """Avoid FA4 autograd by default because backward OOM cannot be recovered in this wrapper."""
    if FLASH_ATTN_BACKEND != 'fa4' or ALLOW_FA4_TRAINING:
        return False
    if not torch.is_grad_enabled():
        return False
    return q.requires_grad or k.requires_grad or v.requires_grad


def _use_fa3():
    """Backward-compatible alias for older tests and callers."""
    return _use_flash_attention()


def _is_out_of_memory_error(exc):
    """Best-effort detection for CUDA OOM errors across torch/cuda versions."""
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "cuda error: out of memory" in msg
        or "cuda out of memory" in msg
    )


# =============================================================================
# SDPA helpers
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)
    
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

# =============================================================================
# Public API: Same interface as the flash attention backends used here
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if _use_flash_attention() and not _should_skip_fa4_for_training(q, k, v):
        try:
            y = _backend.flash_attn_func(q, k, v, causal=causal, window_size=window_size)
            return _unwrap_backend_output(y)
        except Exception as exc:
            global _flash_disabled_due_to_oom
            # FA4 can fail in some environments when CUTLASS expects typed Int32 window args.
            # Fall back to SDPA for this call instead of crashing training.
            is_fa4_typed_window_err = FLASH_ATTN_BACKEND == 'fa4' and _is_fa4_window_size_type_error(exc)
            is_fa4_oom = FLASH_ATTN_BACKEND == 'fa4' and _is_out_of_memory_error(exc)
            if is_fa4_oom:
                _flash_disabled_due_to_oom = True
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if not (is_fa4_typed_window_err or is_fa4_oom):
                raise

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA3 updates k_cache/v_cache in-place. Our SDPA fallback does the same.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    if _use_flash_attention(require_kvcache=True):
        try:
            y = _backend.flash_attn_with_kvcache(
                q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
                causal=causal, window_size=window_size
            )
            return _unwrap_backend_output(y)
        except Exception as exc:
            global _flash_disabled_due_to_oom
            # Match flash_attn_func behavior: tolerate FA4 window-size type issues.
            is_fa4_typed_window_err = FLASH_ATTN_BACKEND == 'fa4' and _is_fa4_window_size_type_error(exc)
            is_fa4_oom = FLASH_ATTN_BACKEND == 'fa4' and _is_out_of_memory_error(exc)
            if is_fa4_oom:
                _flash_disabled_due_to_oom = True
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if not (is_fa4_typed_window_err or is_fa4_oom):
                raise

    # SDPA fallback: manually manage KV cache
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place, matching FA3 behavior)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA backends)
# =============================================================================
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
