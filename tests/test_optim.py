import torch

from nanochat.optim import MuonAdamW


def test_adamw_param_update_hook_applies_same_delta_as_default_path():
    base_param = torch.nn.Parameter(torch.tensor([0.5, -1.0, 1.5], dtype=torch.float32))
    hooked_param = torch.nn.Parameter(base_param.detach().clone())
    grad = torch.tensor([0.2, -0.4, 0.6], dtype=torch.float32)
    base_param.grad = grad.clone()
    hooked_param.grad = grad.clone()
    hooked_before = hooked_param.detach().clone()
    captured = {}
    lr = 0.1
    weight_decay = 0.01

    def hook(delta):
        captured['delta'] = delta.detach().clone()
        hooked_param.add_(delta)

    default_optimizer = MuonAdamW([
        dict(kind='adamw', params=[base_param], lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay),
    ])
    hooked_optimizer = MuonAdamW([
        dict(
            kind='adamw', params=[hooked_param], lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay,
            param_update_hooks=[hook],
        ),
    ])

    assert 'param_update_hooks' not in hooked_optimizer.param_groups[0]

    default_optimizer.step()
    hooked_optimizer.step()

    assert torch.allclose(hooked_param, base_param)
    decayed_before = hooked_before * (1 - lr * weight_decay)
    assert torch.allclose(captured['delta'], hooked_param.detach() - decayed_before)


def test_muon_param_update_hook_preserves_group_update_behavior():
    param_a = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10)
    param_b = torch.nn.Parameter(-param_a.detach().clone())
    hooked_a = torch.nn.Parameter(param_a.detach().clone())
    hooked_b = torch.nn.Parameter(param_b.detach().clone())

    grad_a = torch.tensor([
        [0.3, -0.2, 0.1, 0.4],
        [-0.5, 0.2, 0.3, -0.1],
        [0.2, 0.1, -0.4, 0.6],
    ], dtype=torch.float32)
    grad_b = torch.tensor([
        [-0.1, 0.2, -0.3, 0.4],
        [0.3, -0.2, 0.5, -0.4],
        [-0.6, 0.1, 0.2, -0.3],
    ], dtype=torch.float32)

    param_a.grad = grad_a.clone()
    param_b.grad = grad_b.clone()
    hooked_a.grad = grad_a.clone()
    hooked_b.grad = grad_b.clone()
    hooked_a_before = hooked_a.detach().clone()
    hooked_b_before = hooked_b.detach().clone()
    captured = {}

    def hook(param, delta):
        captured['delta'] = delta.detach().clone()
        param.add_(delta)

    hooked_optimizer = MuonAdamW([
        dict(
            kind='muon', params=[hooked_a, hooked_b], lr=0.05, momentum=0.95, ns_steps=3, beta2=0.95, weight_decay=0.0,
            param_update_hooks=[hook, None],
        ),
    ])

    hooked_optimizer.step()

    assert torch.allclose(captured['delta'], hooked_a.detach() - hooked_a_before)
    assert not torch.allclose(hooked_a, hooked_a_before)
    assert not torch.allclose(hooked_b, hooked_b_before)


def test_muon_chunk_size_preserves_full_group_update():
    torch.manual_seed(0)
    full_params = [
        torch.nn.Parameter(torch.randn(3, 4, dtype=torch.float32))
        for _ in range(5)
    ]
    chunked_params = [torch.nn.Parameter(param.detach().clone()) for param in full_params]
    grads = [torch.randn_like(param) for param in full_params]

    for param, grad in zip(full_params, grads):
        param.grad = grad.clone()
    for param, grad in zip(chunked_params, grads):
        param.grad = grad.clone()

    full_optimizer = MuonAdamW([
        dict(kind='muon', params=full_params, lr=0.05, momentum=0.95, ns_steps=3, beta2=0.95, weight_decay=0.01),
    ])
    chunked_optimizer = MuonAdamW([
        dict(kind='muon', params=chunked_params, lr=0.05, momentum=0.95, ns_steps=3, beta2=0.95, weight_decay=0.01, chunk_size=2),
    ])

    full_optimizer.step()
    chunked_optimizer.step()

    for full_param, chunked_param in zip(full_params, chunked_params):
        assert torch.allclose(chunked_param, full_param)


def test_muon_chunked_hooked_group_applies_updates():
    torch.manual_seed(1)
    params = [
        torch.nn.Parameter(torch.randn(3, 4, dtype=torch.float32))
        for _ in range(3)
    ]
    grads = [torch.randn_like(param) for param in params]
    before = [param.detach().clone() for param in params]
    captured = []

    def hook(param, delta):
        captured.append(delta.detach().clone())
        param.add_(delta)

    for param, grad in zip(params, grads):
        param.grad = grad.clone()

    optimizer = MuonAdamW([
        dict(
            kind='muon', params=params, lr=0.05, momentum=0.95, ns_steps=3, beta2=0.95,
            weight_decay=0.01, chunk_size=1, param_update_hooks=[hook, None, hook],
        ),
    ])

    optimizer.step()

    assert len(captured) == 2
    for param, param_before in zip(params, before):
        assert not torch.allclose(param, param_before)