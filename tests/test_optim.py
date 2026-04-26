import torch

from nanochat.configuration_nanomoe_gpt import GPTConfig
from nanochat.gpt import GPT, Qwen3MLP, Qwen3MLPExperts
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


def test_setup_optimizer_applies_moe_weight_decay_to_dense_gate_projection():
    config = GPTConfig(
        n_layer=3,
        moe_start_layer=1,
        moe_layer_stride=1,
        n_exp=2,
        n_embd=8,
        n_head=2,
    )
    model = GPT(config)

    optimizer = model.setup_optimizer(
        matrix_lr=0.01,
        weight_decay_dense=0.2,
        weight_decay_moe=0.2,
    )

    moe_params = set()
    dense_params = set()
    for block in model.transformer.h:
        params = set(block.parameters())
        if hasattr(block, 'mlp') and block.mlp.__class__.__name__ == 'MOELayer':
            moe_params.update(params)
        else:
            dense_params.update(params)

    moe_muon_groups = []
    other_muon_groups = []
    for group in optimizer.param_groups:
        if group.get('kind') != 'muon':
            continue
        params = set(group['params'])
        if params and params.issubset(moe_params):
            moe_muon_groups.append(group)
        else:
            other_muon_groups.append(group)

    assert moe_muon_groups
    assert other_muon_groups
    assert all(group['weight_decay'] == 0.2 for group in moe_muon_groups)
    assert all(group['weight_decay'] == 0.2 for group in other_muon_groups)


def test_setup_optimizer_keeps_gate_projection_biases_out_of_muon_groups():
    config = GPTConfig(
        n_layer=4,
        moe_start_layer=1,
        moe_layer_stride=1,
        n_exp=2,
        n_embd=8,
        n_head=2,
        use_exp_gate_proj_bias=True,
        use_dense_gate_proj_bias=True,
    )
    model = GPT(config)

    optimizer = model.setup_optimizer(
        matrix_lr=0.01,
        weight_decay_dense=0.0,
        weight_decay_moe=0.0,
    )

    dense_gate_bias = []
    moe_gate_bias = []
    for block in model.transformer.h:
        mlp = getattr(block, 'mlp', None)
        if hasattr(mlp, 'gate_proj_bias') and mlp.gate_proj_bias is not None:
            dense_gate_bias.append(mlp.gate_proj_bias)
        if hasattr(mlp, 'experts') and getattr(mlp.experts, 'gate_proj_bias', None) is not None:
            moe_gate_bias.append(mlp.experts.gate_proj_bias)

    muon_params = {
        param
        for group in optimizer.param_groups
        if group.get('kind') == 'muon'
        for param in group['params']
    }
    adamw_params = {
        param
        for group in optimizer.param_groups
        if group.get('kind') == 'adamw'
        for param in group['params']
    }

    assert dense_gate_bias
    assert moe_gate_bias
    assert all(param not in muon_params for param in dense_gate_bias)
    assert all(param not in muon_params for param in moe_gate_bias)
    assert all(param in adamw_params for param in dense_gate_bias)
    assert all(param in adamw_params for param in moe_gate_bias)


def test_gate_proj_bias_forward_and_l2_gradients_are_scaled_down():
    torch.manual_seed(0)
    config = GPTConfig(
        n_layer=2,
        moe_start_layer=1,
        moe_layer_stride=1,
        n_exp=2,
        n_embd=8,
        n_head=2,
        use_exp_gate_proj_bias=True,
        use_dense_gate_proj_bias=True,
    )
    x_dense = torch.randn(2, 3, config.n_embd)
    x_moe = torch.randn(config.n_exp, 3, config.n_embd)

    dense_mlp = Qwen3MLP(config)
    with torch.no_grad():
        for parameter in dense_mlp.parameters():
            parameter.normal_(mean=0.0, std=0.02)
    dense_loss = dense_mlp(x_dense).sum() + dense_mlp.get_gate_proj_bias_with_scaled_grad().float().square().mean()
    dense_loss.backward()
    dense_bias = dense_mlp.gate_proj_bias.detach().clone().requires_grad_(True)
    dense_expected_loss = dense_mlp.c_proj(
        (dense_mlp.act_fn(dense_mlp.gate_proj(x_dense)) + dense_bias) * dense_mlp.c_fc(x_dense)
    ).sum() + dense_bias.float().square().mean()
    dense_expected_grad, = torch.autograd.grad(dense_expected_loss, dense_bias)

    moe_mlp = Qwen3MLPExperts(config)
    with torch.no_grad():
        for parameter in moe_mlp.parameters():
            parameter.normal_(mean=0.0, std=0.02)
    moe_loss = moe_mlp(x_moe).sum() + moe_mlp.get_gate_proj_bias_with_scaled_grad().float().square().mean()
    moe_loss.backward()
    moe_bias = moe_mlp.gate_proj_bias.detach().clone().requires_grad_(True)
    moe_gate_out = moe_mlp.act_fn(torch.bmm(x_moe, moe_mlp.gate_proj)) + moe_bias.unsqueeze(1)
    moe_expected_loss = torch.bmm(moe_gate_out * torch.bmm(x_moe, moe_mlp.c_fc), moe_mlp.c_proj).sum() + moe_bias.float().square().mean()
    moe_expected_grad, = torch.autograd.grad(moe_expected_loss, moe_bias)

    assert torch.allclose(dense_mlp.gate_proj_bias.grad, dense_expected_grad * 0.1, atol=1e-6, rtol=1e-5)
    assert torch.allclose(moe_mlp.gate_proj_bias.grad, moe_expected_grad * 0.1, atol=1e-6, rtol=1e-5)