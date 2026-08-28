import ast
from pathlib import Path

import pytest
import torch

from nanochat.configuration_nanomoe_gpt import GPTConfig
from nanochat.gpt import GPT, Qwen3MLP, Qwen3MLPExperts
from nanochat import optim as optim_module
from nanochat.optim import AuroraAdamW, DistAuroraAdamW, DistMuonAdamW, MuonAdamW


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


def test_adamw_nonfinite_error_reports_named_parameter_and_source(monkeypatch):
    param = torch.nn.Parameter(torch.ones(3, dtype=torch.float32))
    param.grad = torch.ones_like(param)

    optimizer = AuroraAdamW([
        dict(
            kind='adamw',
            params=[param],
            debug_param_names=['transformer.h.2.mlp.experts.kappa_bias'],
            lr=0.1,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
        ),
    ])

    def fake_adamw_step_fused(p_flat, _grad_flat, exp_avg_flat, exp_avg_sq_flat, *_args):
        exp_avg_flat.zero_()
        exp_avg_sq_flat.zero_()
        p_flat[0] = float('nan')

    monkeypatch.setattr(optim_module, 'adamw_step_fused', fake_adamw_step_fused)

    with pytest.raises(RuntimeError, match='transformer\.h\.2\.mlp\.experts\.kappa_bias') as exc_info:
        optimizer.step()

    message = str(exc_info.value)
    assert 'updated index=(0,) value=nan' in message


def test_adamw_nonfinite_error_reports_preexisting_state(monkeypatch):
    param = torch.nn.Parameter(torch.ones(3, dtype=torch.float32))
    param.grad = torch.ones_like(param)

    optimizer = AuroraAdamW([
        dict(
            kind='adamw',
            params=[param],
            debug_param_names=['transformer.h.2.mlp.experts.kappa_bias'],
            lr=0.1,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
        ),
    ])
    optimizer.state[param]['step'] = 1
    optimizer.state[param]['exp_avg'] = torch.full_like(param, float('nan'))
    optimizer.state[param]['exp_avg_sq'] = torch.zeros_like(param)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError('adamw_step_fused should not run when state is already non-finite')

    monkeypatch.setattr(optim_module, 'adamw_step_fused', fail_if_called)

    with pytest.raises(RuntimeError, match='AdamW received non-finite inputs/state') as exc_info:
        optimizer.step()

    message = str(exc_info.value)
    assert 'exp_avg index=(0,) value=nan' in message


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


def test_bfloat16_parameters_create_bfloat16_optimizer_states():
    adamw_param = torch.nn.Parameter(torch.tensor([0.5, -1.0], dtype=torch.bfloat16))
    muon_param = torch.nn.Parameter(torch.arange(12, dtype=torch.bfloat16).reshape(3, 4) / 10)
    adamw_param.grad = torch.tensor([0.2, -0.4], dtype=torch.bfloat16)
    muon_param.grad = torch.ones_like(muon_param)
    optimizer = MuonAdamW([
        dict(
            kind='adamw', params=[adamw_param], lr=0.1, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
        ),
        dict(
            kind='muon', params=[muon_param], lr=0.05, momentum=0.95, ns_steps=3, beta2=0.95, weight_decay=0.0,
        ),
    ])

    optimizer.step()

    adamw_state = optimizer.state[adamw_param]
    muon_state = optimizer.state[muon_param]
    assert adamw_state['exp_avg'].dtype == torch.bfloat16
    assert adamw_state['exp_avg_sq'].dtype == torch.bfloat16
    assert muon_state['momentum_buffer'].dtype == torch.bfloat16
    assert muon_state['second_momentum_buffer'].dtype == torch.bfloat16
    assert torch.isfinite(adamw_param).all()
    assert torch.isfinite(muon_param).all()


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


def test_aurora_group_update_changes_all_params():
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

    optimizer = AuroraAdamW([
        dict(
            kind='aurora', params=[param_a, param_b], lr=0.05, momentum=0.95,
            pp_iterations=2, pp_beta=0.5, weight_decay=0.0,
        ),
    ])

    optimizer.step()

    assert not torch.allclose(param_a, before_a)
    assert not torch.allclose(param_b, before_b)


def test_aurora_nonfinite_error_reports_param_name_and_source(monkeypatch):
    param_a = torch.nn.Parameter(torch.ones(3, 4, dtype=torch.float32))
    param_b = torch.nn.Parameter(torch.full((3, 4), 2.0, dtype=torch.float32))
    param_a.grad = torch.ones_like(param_a)
    param_b.grad = torch.ones_like(param_b)

    optimizer = AuroraAdamW([
        dict(
            kind='aurora',
            params=[param_a, param_b],
            debug_param_names=['transformer.h.0.attn.c_q.weight', 'transformer.h.0.attn.c_k.weight'],
            lr=0.05,
            momentum=0.95,
            pp_iterations=2,
            pp_beta=0.5,
            weight_decay=0.0,
        ),
    ])

    def fake_aurora_step_fused(_grads, updated, momentum_buffer, *_args):
        momentum_buffer.zero_()
        updated[0, 0, 0] = float('inf')

    monkeypatch.setattr(optim_module, 'aurora_step_fused', fake_aurora_step_fused)

    with pytest.raises(RuntimeError, match='transformer\\.h\\.0\\.attn\\.c_q\\.weight') as exc_info:
        optimizer.step()

    message = str(exc_info.value)
    assert 'updated name=transformer.h.0.attn.c_q.weight' in message
    assert 'grad name=' not in message
    assert 'param name=' not in message


def test_aurora_chunk_size_preserves_full_group_update():
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

    full_optimizer = AuroraAdamW([
        dict(kind='aurora', params=full_params, lr=0.05, momentum=0.95, pp_iterations=2, pp_beta=0.5, weight_decay=0.01),
    ])
    chunked_optimizer = AuroraAdamW([
        dict(kind='aurora', params=chunked_params, lr=0.05, momentum=0.95, pp_iterations=2, pp_beta=0.5, weight_decay=0.01, chunk_size=2),
    ])

    full_optimizer.step()
    chunked_optimizer.step()

    for full_param, chunked_param in zip(full_params, chunked_params):
        assert torch.allclose(chunked_param, full_param)


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


def test_dist_aurora_compute_reuses_updated_param_buffer(monkeypatch):
    params = [
        torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4) + offset)
        for offset in (0.0, 10.0)
    ]
    grad_chunk = torch.stack([torch.ones_like(params[0]), torch.full_like(params[0], 2.0)])
    stacked_grads = torch.empty_like(grad_chunk)
    optimizer = DistAuroraAdamW([
        dict(kind='aurora', params=params, lr=0.05, momentum=0.95, pp_iterations=2, pp_beta=0.5, weight_decay=0.01),
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

    def fake_aurora_step_fused(grads, updated, momentum_buffer, *_args):
        updated.sub_(grads)
        momentum_buffer.copy_(grads)

    original_stack = torch.stack

    def guarded_stack(sequence, *args, **kwargs):
        if sequence and all(item is param for item, param in zip(sequence, params)) and len(sequence) == len(params):
            raise AssertionError('owned params should be copied into the update buffer, not restacked')
        return original_stack(sequence, *args, **kwargs)

    monkeypatch.setattr(optim_module.dist, 'all_gather_into_tensor', fake_all_gather_into_tensor)
    monkeypatch.setattr(optim_module, 'aurora_step_fused', fake_aurora_step_fused)
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
        optimizer._compute_aurora(optimizer.param_groups[0], info, gather_list, rank=0)
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
        weight_decay=0.2,
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


def test_setup_optimizer_includes_router_wg_delta_matrix():
    config = GPTConfig(
        n_layer=3,
        moe_start_layer=1,
        moe_layer_stride=1,
        n_exp=2,
        n_embd=8,
        n_head=2,
    )
    model = GPT(config)
    model.setup_router_wg_delta()

    optimizer = model.setup_optimizer(matrix_lr=0.01)
    optimizer_params = {
        parameter
        for group in optimizer.param_groups
        for parameter in group['params']
    }
    for block in model.transformer.h:
        if hasattr(block.mlp, 'router'):
            assert block.mlp.router.w_g_delta in optimizer_params
            assert not block.mlp.router.w_g.weight.requires_grad
    delta_groups = [
        group for group in optimizer.param_groups
        if group.get('name') == 'router_wg_delta'
    ]
    assert len(delta_groups) == 1
    assert set(delta_groups[0]['params']) == {
        block.mlp.router.w_g_delta
        for block in model.transformer.h
        if hasattr(block.mlp, 'router')
    }
    base_router_groups = [
        group for group in optimizer.param_groups
        if group.get('name') == 'router_wg_base'
    ]
    assert len(base_router_groups) == 1

    delta_group = delta_groups[0]
    delta_param = delta_group['params'][0]
    initial_delta = delta_param.detach().clone()
    delta_param.grad = torch.randn_like(delta_param)
    delta_group['lr'] = 0.0
    optimizer.step()
    torch.testing.assert_close(delta_param, initial_delta)

    optimizer.zero_grad(set_to_none=True)
    delta_param.grad = torch.randn_like(delta_param)
    delta_group['lr'] = 0.01
    optimizer.step()
    assert not torch.equal(delta_param, initial_delta)


def test_setup_optimizer_scalar_lr_is_x0_lr_and_residual_scalars_use_one_tenth():
    config = GPTConfig(
        n_layer=2,
        n_exp=1,
        n_embd=8,
        n_head=2,
    )
    model = GPT(config)

    optimizer = model.setup_optimizer(scalar_lr=0.05)

    resid_group = next(
        group for group in optimizer.param_groups
        if any(param is model.resid_lambdas for param in group['params'])
    )
    x0_group = next(
        group for group in optimizer.param_groups
        if any(param is model.x0_lambdas for param in group['params'])
    )
    source_group = next(
        group for group in optimizer.param_groups
        if any(param is model.ut_source_lambdas for param in group['params'])
    )
    assert resid_group['lr'] == pytest.approx(0.005)
    assert source_group is resid_group
    assert x0_group['lr'] == pytest.approx(0.05)
    assert resid_group['kind'] == 'adamw'
    assert x0_group['kind'] == 'adamw'


def test_setup_optimizer_keeps_kappa_biases_out_of_muon_groups():
    config = GPTConfig(
        n_layer=4,
        moe_start_layer=1,
        moe_layer_stride=1,
        n_exp=2,
        n_embd=8,
        n_head=2,
        use_kappa_swiglu=True,
    )
    model = GPT(config)

    optimizer = model.setup_optimizer(
        matrix_lr=0.01,
        weight_decay=0.0,
    )

    dense_gate_bias = []
    moe_gate_bias = []
    for block in model.transformer.h:
        mlp = getattr(block, 'mlp', None)
        if hasattr(mlp, 'experts') and getattr(mlp.experts, 'kappa_bias', None) is not None:
            moe_gate_bias.append(mlp.experts.kappa_bias)

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


def test_setup_optimizer_selects_aurora_for_matrix_groups():
    config = GPTConfig(
        n_layer=4,
        moe_start_layer=1,
        moe_layer_stride=1,
        n_exp=2,
        n_embd=8,
        n_head=2,
    )
    model = GPT(config)

    optimizer = model.setup_optimizer(
        matrix_lr=0.01,
        weight_decay=0.2,
        matrix_optimizer='aurora',
    )

    matrix_groups = [group for group in optimizer.param_groups if group['kind'] == 'aurora']

    assert isinstance(optimizer, AuroraAdamW)
    assert matrix_groups
    assert all(group['weight_decay'] == 0.2 for group in matrix_groups)



def test_setup_optimizer_places_kappa_params_in_scaled_adamw_group():
    config = GPTConfig(
        n_layer=4,
        moe_start_layer=1,
        moe_layer_stride=1,
        n_exp=2,
        n_embd=8,
        n_head=2,
        use_kappa_swiglu=True,
    )
    model = GPT(config)

    optimizer = model.setup_optimizer(
        embedding_lr=0.2,
        matrix_lr=0.01,
        weight_decay=0.0,
        kappa_lr_final_scale=1.0,
        kappa_bias_lr_warmup_iterations=1000,
    )

    kappa_params = {
        param
        for name, param in model.named_parameters()
        if 'kappa_bias' in name or 'kappa_scale' in name
    }
    kappa_bias_group = next(
        group for group in optimizer.param_groups
        if group.get('name') == 'kappa_bias'
    )

    assert kappa_params
    assert set(kappa_bias_group['params']) == kappa_params
    assert kappa_bias_group['kind'] == 'adamw'
    assert kappa_bias_group['lr'] == 0.0
    assert kappa_bias_group['initial_lr'] == kappa_bias_group['lr']
    assert kappa_bias_group['base_lr'] == 0.05  # default scalar_lr, matches resid/x0 scalar groups
    assert kappa_bias_group['lr_scale_end'] == 1.0
    assert kappa_bias_group['lr_scale_warmup_iterations'] == 1000


def test_kappa_bias_lr_schedule_warms_then_decays_to_final_scale():
    schedule = load_base_train_function("get_linear_lr_scale")

    assert schedule(0, 100, end_scale=0.2, warmup_iterations=10) == 0.0
    assert schedule(5, 100, end_scale=0.2, warmup_iterations=10) == 0.5
    assert schedule(10, 100, end_scale=0.2, warmup_iterations=10) == 1.0
    assert abs(schedule(55, 100, end_scale=0.2, warmup_iterations=10) - 0.6) < 1e-12
    assert abs(schedule(100, 100, end_scale=0.2, warmup_iterations=10) - 0.2) < 1e-12