import multiprocessing
import runpy
import socket
import sys

import nanochat.dataset as dataset
from nanochat.dataset import set_huggingface_dns


def test_huggingface_dns_override(monkeypatch):
    calls = []

    def original_getaddrinfo(host, port, *args, **kwargs):
        calls.append((host, port, args, kwargs))
        return [(host, port)]

    monkeypatch.setattr(socket, "getaddrinfo", original_getaddrinfo)
    set_huggingface_dns("192.0.2.1")

    assert socket.getaddrinfo("huggingface.co", 443, 0, type=socket.SOCK_STREAM) == [("192.0.2.1", 443)]
    assert socket.getaddrinfo("example.com", 80) == [("example.com", 80)]
    assert calls == [
        ("192.0.2.1", 443, (0,), {"type": socket.SOCK_STREAM}),
        ("example.com", 80, (), {}),
    ]


def test_dns_override_requires_cli_flag(monkeypatch):
    pool_calls = []

    class FakePool:
        def __init__(self, **kwargs):
            pool_calls.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

        def map(self, function, items):
            return []

    monkeypatch.setattr(multiprocessing, "Pool", FakePool)
    for options in ([], ["--override-huggingface-dns"], ["--override-huggingface-dns", "--huggingface-ip", "192.0.2.1"]):
        monkeypatch.setattr(sys, "argv", ["dataset.py", "--num-files", "0", *options])
        runpy.run_path(dataset.__file__, run_name="__main__")

    assert pool_calls[0] == {"processes": 4}
    assert pool_calls[1]["initargs"] == ("13.35.36.77",)
    assert pool_calls[2]["initargs"] == ("192.0.2.1",)
    assert all(call["initializer"].__name__ == "set_huggingface_dns" for call in pool_calls[1:])