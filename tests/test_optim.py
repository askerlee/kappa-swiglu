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
    captured = {}

    def hook(param, delta):
        captured['delta'] = delta.detach().clone()
        param.add_(delta)

    default_optimizer = MuonAdamW([
        dict(kind='muon', params=[param_a, param_b], lr=0.05, momentum=0.95, ns_steps=3, beta2=0.95, weight_decay=0.0),
    ])
    hooked_optimizer = MuonAdamW([
        dict(
            kind='muon', params=[hooked_a, hooked_b], lr=0.05, momentum=0.95, ns_steps=3, beta2=0.95, weight_decay=0.0,
            param_update_hooks=[hook, None],
        ),
    ])

    default_optimizer.step()
    hooked_optimizer.step()

    assert torch.allclose(hooked_a, param_a)
    assert torch.allclose(hooked_b, param_b)
    assert torch.allclose(captured['delta'], hooked_a.detach() - hooked_a_before)