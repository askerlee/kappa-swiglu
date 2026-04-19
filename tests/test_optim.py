import torch

from nanochat.optim import MuonAdamW


def test_adamw_step_updates_parameter_and_state():
    param = torch.nn.Parameter(torch.tensor([0.5, -1.0, 1.5], dtype=torch.float32))
    grad = torch.tensor([0.2, -0.4, 0.6], dtype=torch.float32)
    before = param.detach().clone()
    param.grad = grad.clone()
    lr = 0.1
    weight_decay = 0.01

    optimizer = MuonAdamW([
        dict(
            kind='adamw', params=[param], lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay,
        ),
    ])

    optimizer.step()

    assert not torch.allclose(param, before)
    assert optimizer.state[param]['step'] == 1


def test_muon_group_update_changes_all_params():
    param_a = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10)
    param_b = torch.nn.Parameter(-param_a.detach().clone())

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
    before_a = param_a.detach().clone()
    before_b = param_b.detach().clone()

    optimizer = MuonAdamW([
        dict(
            kind='muon', params=[param_a, param_b], lr=0.05, momentum=0.95, ns_steps=3, beta2=0.95, weight_decay=0.0,
        ),
    ])

    optimizer.step()

    assert not torch.allclose(param_a, before_a)
    assert not torch.allclose(param_b, before_b)


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


def test_muon_chunk_size_one_updates_all_params():
    torch.manual_seed(1)
    params = [
        torch.nn.Parameter(torch.randn(3, 4, dtype=torch.float32))
        for _ in range(3)
    ]
    grads = [torch.randn_like(param) for param in params]
    before = [param.detach().clone() for param in params]

    for param, grad in zip(params, grads):
        param.grad = grad.clone()

    optimizer = MuonAdamW([
        dict(
            kind='muon', params=params, lr=0.05, momentum=0.95, ns_steps=3, beta2=0.95,
            weight_decay=0.01, chunk_size=1,
        ),
    ])

    optimizer.step()

    for param, param_before in zip(params, before):
        assert not torch.allclose(param, param_before)