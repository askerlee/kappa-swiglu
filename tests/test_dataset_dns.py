import socket

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