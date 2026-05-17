import functools
import os
import subprocess
import warnings

import torch


_DISABLE_COMPILE_ENV = "NANOCHAT_DISABLE_COMPILE"
_compile_disabled_reason = None


def _env_flag_is_true(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _iter_exception_chain(exc: BaseException):
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_backend_compile_failure(exc: BaseException) -> bool:
    for current in _iter_exception_chain(exc):
        if isinstance(current, subprocess.CalledProcessError):
            return True
        exc_type = type(current)
        module_name = exc_type.__module__
        type_name = exc_type.__name__
        if type_name in {"BackendCompilerFailed", "InductorError"}:
            return True
        if module_name.startswith("torch._inductor") or module_name.startswith("triton"):
            return True
    return False


def _disable_compile(reason: str) -> None:
    global _compile_disabled_reason
    if _compile_disabled_reason is not None:
        return
    _compile_disabled_reason = reason
    warnings.warn(
        f"nanochat: disabling torch.compile and falling back to eager execution ({reason})",
        RuntimeWarning,
        stacklevel=3,
    )


def compile_is_disabled() -> bool:
    return _compile_disabled_reason is not None or _env_flag_is_true(_DISABLE_COMPILE_ENV)


class _CompiledModuleProxy:
    def __init__(self, original, compiled):
        object.__setattr__(self, "_original", original)
        object.__setattr__(self, "_compiled", compiled)
        object.__setattr__(self, "_compiled_enabled", True)

    def __call__(self, *args, **kwargs):
        if compile_is_disabled() or not object.__getattribute__(self, "_compiled_enabled"):
            return object.__getattribute__(self, "_original")(*args, **kwargs)

        compiled = object.__getattribute__(self, "_compiled")
        original = object.__getattribute__(self, "_original")
        try:
            return compiled(*args, **kwargs)
        except Exception as exc:
            if _is_backend_compile_failure(exc):
                object.__setattr__(self, "_compiled_enabled", False)
                _disable_compile(str(exc))
                return original(*args, **kwargs)
            raise

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_original"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_original"), name, value)


def maybe_compile(obj=None, **compile_kwargs):
    if obj is None:
        return lambda actual_obj: maybe_compile(actual_obj, **compile_kwargs)

    if compile_is_disabled() or not hasattr(torch, "compile"):
        return obj

    try:
        compiled = torch.compile(obj, **compile_kwargs)
    except Exception as exc:
        if _is_backend_compile_failure(exc):
            _disable_compile(str(exc))
            return obj
        raise

    if isinstance(obj, torch.nn.Module):
        return _CompiledModuleProxy(obj, compiled)

    if callable(obj):
        @functools.wraps(obj)
        def wrapped(*args, **kwargs):
            if compile_is_disabled():
                return obj(*args, **kwargs)
            try:
                return compiled(*args, **kwargs)
            except Exception as exc:
                if _is_backend_compile_failure(exc):
                    _disable_compile(str(exc))
                    return obj(*args, **kwargs)
                raise

        return wrapped

    return compiled