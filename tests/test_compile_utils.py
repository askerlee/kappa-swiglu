import subprocess

import torch

from nanochat import compile_utils


def test_maybe_compile_function_falls_back_after_backend_failure(monkeypatch):
    compile_utils._compile_disabled_reason = None

    def eager_add_one(x):
        return x + 1

    class FakeCompiledFunction:
        def __init__(self):
            self.calls = 0

        def __call__(self, x):
            self.calls += 1
            raise subprocess.CalledProcessError(1, ["gcc", "cuda_utils.c"])

    fake_compiled = FakeCompiledFunction()
    monkeypatch.setattr(compile_utils.torch, "compile", lambda obj, **kwargs: fake_compiled)

    wrapped = compile_utils.maybe_compile(eager_add_one, dynamic=False)

    assert wrapped(torch.tensor(2)) == 3
    assert compile_utils.compile_is_disabled() is True
    assert fake_compiled.calls == 1


def test_maybe_compile_module_proxy_falls_back_to_original(monkeypatch):
    compile_utils._compile_disabled_reason = None

    class ToyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.config = {"name": "toy"}

        def forward(self, x):
            self.calls += 1
            return x + 1

    class FakeCompiledModule:
        def __call__(self, x):
            raise subprocess.CalledProcessError(1, ["gcc", "cuda_utils.c"])

    model = ToyModule()
    monkeypatch.setattr(compile_utils.torch, "compile", lambda obj, **kwargs: FakeCompiledModule())

    wrapped = compile_utils.maybe_compile(model, dynamic=False)

    assert wrapped.config == {"name": "toy"}
    assert wrapped(torch.tensor(4)) == 5
    assert model.calls == 1
    assert compile_utils.compile_is_disabled() is True