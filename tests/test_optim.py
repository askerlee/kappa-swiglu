import ast
from pathlib import Path

import torch

from nanochat.configuration_nanomoe_gpt import GPTConfig
from nanochat.gpt import GPT, Qwen3MLP, Qwen3MLPExperts
from nanochat import optim as optim_module
from nanochat.optim import DistMuonAdamW, MuonAdamW


def load_base_train_function(name: str):
    base_train_path = Path(__file__).resolve().parents[1] / "scripts" / "base_train.py"
    source = base_train_path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(base_train_path))
    namespace = {}
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            function_module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(function_module)
            exec(compile(function_module, str(base_train_path), "exec"), namespace)
            return namespace[name]
    raise AssertionError(f"Function {name} not found in {base_train_path}")


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


def test_dist_muon_compute_reuses_updated_param_buffer(monkeypatch):
    params = [
        torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4) + offset)
        for offset in (0.0, 10.0)
    ]
    grad_chunk = torch.stack([torch.ones_like(params[0]), torch.full_like(params[0], 2.0)])
    stacked_grads = torch.empty_like(grad_chunk)
    optimizer = DistMuonAdamW([
        dict(kind='muon', params=params, lr=0.05, momentum=0.95, ns_steps=3, beta2=0.95, weight_decay=0.01),
    ])

    class _DoneFuture:
        def wait(self):
            return None

    class _AsyncCollective:
        def __init__(self, output, local):
            self.output = output
            self.local = local

        def get_future(self):
            self.output[:self.local.shape[0]].copy_(self.local)
            return _DoneFuture()

    def fake_all_gather_into_tensor(output, local, async_op=True):
        assert async_op is True
        return _AsyncCollective(output, local)

    def fake_muon_step_fused(grads, updated, momentum_buffer, second_momentum_buffer, *_args):
        updated.sub_(grads)
        momentum_buffer.copy_(grads)
        second_momentum_buffer.zero_()

    original_stack = torch.stack

    def guarded_stack(sequence, *args, **kwargs):
        if sequence and all(item is param for item, param in zip(sequence, params)) and len(sequence) == len(params):
            raise AssertionError('owned params should be copied into the update buffer, not restacked')
        return original_stack(sequence, *args, **kwargs)

    monkeypatch.setattr(optim_module.dist, 'all_gather_into_tensor', fake_all_gather_into_tensor)
    monkeypatch.setattr(optim_module, 'muon_step_fused', fake_muon_step_fused)
    monkeypatch.setattr(torch, 'stack', guarded_stack)

    info = dict(chunk_infos=[dict(
        future=_DoneFuture(),
        params=params,
        chunk_size=len(params),
        grad_chunk=grad_chunk,
        stacked_grads=stacked_grads,
    )])
    gather_list = []

    with torch.inference_mode():
        optimizer._compute_muon(optimizer.param_groups[0], info, gather_list, rank=0)
        optimizer._finish_gathers(gather_list)

    assert len(gather_list) == 1
    assert torch.allclose(params[0], torch.arange(12, dtype=torch.float32).reshape(3, 4) - 1.0)
    assert torch.allclose(params[1], torch.arange(12, dtype=torch.float32).reshape(3, 4) + 8.0)


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

    assert moe_gate_bias
    assert all(param not in muon_params for param in moe_gate_bias)
    assert all(param in adamw_params for param in moe_gate_bias)


def test_setup_optimizer_places_gate_proj_biases_in_scaled_groups():
    config = GPTConfig(
        n_layer=4,
        moe_start_layer=1,
        moe_layer_stride=1,
        n_exp=2,
        n_embd=8,
        n_head=2,
        use_exp_gate_proj_bias=True,
    )
    model = GPT(config)

    optimizer = model.setup_optimizer(
        embedding_lr=0.2,
        matrix_lr=0.01,
        weight_decay_dense=0.0,
        weight_decay_moe=0.0,
        gate_proj_bias_lr_final_scale=1.0,
        gate_proj_bias_lr_warmup_iterations=1000,
    )

    moe_gate_proj_bias_params = []
    for block in model.transformer.h:
        mlp = getattr(block, 'mlp', None)
        experts = getattr(mlp, 'experts', None)
        if getattr(experts, 'gate_proj_bias', None) is not None:
            moe_gate_proj_bias_params.append(experts.gate_proj_bias)

    gate_proj_bias_group = None
    for group in optimizer.param_groups:
        params = set(group['params'])
        if params == set(moe_gate_proj_bias_params):
            gate_proj_bias_group = group

    assert gate_proj_bias_group is not None
    assert gate_proj_bias_group['kind'] == 'adamw'
    dmodel_lr_scale = (config.n_embd / 768) ** -0.5
    assert gate_proj_bias_group['lr'] == 0.0
    assert gate_proj_bias_group['initial_lr'] == gate_proj_bias_group['lr']
    assert gate_proj_bias_group['base_lr'] == 0.2 * dmodel_lr_scale
    assert gate_proj_bias_group['lr_scale_end'] == 1.0
    assert gate_proj_bias_group['lr_scale_warmup_iterations'] == 1000


def test_gate_proj_bias_lr_schedule_warms_then_decays_to_final_scale():
    schedule = load_base_train_function("get_linear_lr_scale")

    assert schedule(0, 100, end_scale=0.2, warmup_iterations=10) == 0.0
    assert schedule(5, 100, end_scale=0.2, warmup_iterations=10) == 0.5
    assert schedule(10, 100, end_scale=0.2, warmup_iterations=10) == 1.0
    assert abs(schedule(55, 100, end_scale=0.2, warmup_iterations=10) - 0.6) < 1e-12
    assert abs(schedule(100, 100, end_scale=0.2, warmup_iterations=10) - 0.2) < 1e-12