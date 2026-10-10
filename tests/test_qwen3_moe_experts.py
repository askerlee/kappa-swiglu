import math
import pytest
import torch
import torch.nn.functional as F
from copy import deepcopy

from nanochat.configuration_nanomoe_gpt import GPTConfig
from nanochat.engine import KVCache
from nanochat.gpt import GPT, MANAGER, MOELayer, Qwen3MLP, Qwen3MLPExperts, Router, _UTLossAccum, _UT_LOSS_NAMES, _accumulate_kappa_slope_l2_loss, _chunked_cross_entropy, _save_activations_on_cpu, scale_grad
from nanochat.manager import MOEManager


@pytest.mark.parametrize('masked', [False, True])
@pytest.mark.parametrize('compiled', [False, True])
def test_checkpointed_kappa_regularization_reuses_gate_storage(masked, compiled):
    from nanochat.gpt import _kappa_slope_l2_from_conditioning
    from torch.utils.checkpoint import checkpoint

    raw_gate = torch.randn(2, 8, 16, dtype=torch.bfloat16, requires_grad=True)
    bias = torch.randn(2, 8, 1, requires_grad=True)
    alpha = torch.randn(2, 1, 16, requires_grad=True)
    mask = torch.rand(2, 8) > 0.5 if masked else None
    reference_loss = _kappa_slope_l2_from_conditioning(raw_gate.detach(), bias, alpha, mask)
    reference_grads = torch.autograd.grad(reference_loss, (bias, alpha))
    saved = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    def compute_loss(raw_gate, bias, alpha, mask):
        return checkpoint(
            _kappa_slope_l2_from_conditioning,
            raw_gate.detach(), bias, alpha, mask,
            use_reentrant=False, preserve_rng_state=False,
        )

    if compiled:
        compute_loss = torch.compile(compute_loss, fullgraph=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss = compute_loss(raw_gate, bias, alpha, mask)
    torch.testing.assert_close(loss, reference_loss)
    gate_sized_saved = [tensor for tensor in saved if tensor.shape == raw_gate.shape]
    assert len(gate_sized_saved) == 1
    assert gate_sized_saved[0].dtype == torch.bfloat16
    assert gate_sized_saved[0].data_ptr() == raw_gate.data_ptr()
    loss.backward()
    torch.testing.assert_close(bias.grad, reference_grads[0])
    torch.testing.assert_close(alpha.grad, reference_grads[1])
    assert raw_gate.grad is None


@pytest.mark.parametrize('kappa_input', ['router_probs', 'gate_proj'])
@pytest.mark.parametrize('is_sft', [False, True])
def test_kappa_regularization_blocks_inputs_but_preserves_activation_gradients(monkeypatch, kappa_input, is_sft):
    monkeypatch.setattr('nanochat.gpt.MANAGER', MOEManager())
    torch.manual_seed(42)
    layer = MOELayer(GPTConfig(
        n_exp=2, n_embd=4, use_kappa_swiglu=True, independent_kappa_router=True,
        kappa_input=kappa_input, separate_base_sft_kappa=True,
        use_aux_loss=False, use_router_z_loss=False, router_tie_noise_steps=0,
    ), layer_idx=0)
    layer.experts.kappa_phase = int(is_sft)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.uniform_(-0.3, 0.3)
    inputs = torch.randn(1, 8, 4, requires_grad=True)
    mask = torch.tensor([[True, True, True, True, True, True, False, False]])
    accum = _UTLossAccum(inputs, 1, 2)
    output = layer(inputs, valid_token_mask=mask, loss_accum=accum, router_layer_idx=0)
    loss = accum.losses[_UT_LOSS_NAMES.index('kappa_slope_l2_loss')]
    loss.backward(retain_graph=True)
    assert inputs.grad is None
    assert layer.experts.gate_proj.grad is None
    assert layer.router.w_g.weight.grad is None
    assert layer.kappa_router.weight.grad.abs().sum() > 0
    kappa_parameter = layer.experts.kappa_scale if kappa_input == 'gate_proj' else layer.experts.kappa_bias
    assert kappa_parameter.grad.abs().sum() > 0
    layer.zero_grad(set_to_none=True)
    output.square().sum().backward()
    assert inputs.grad.abs().sum() > 0
    assert layer.experts.gate_proj.grad.abs().sum() > 0
    assert layer.kappa_router.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('kappa_slot', [0, 1])
@pytest.mark.parametrize('granularity', ['per-gate', 'per-expert', 'per-layer', 'global'])
@pytest.mark.parametrize('independent_router', [False, True])
def test_gate_proj_kappa_formula_and_gradients(monkeypatch, training, kappa_slot, granularity, independent_router):
    manager = MOEManager()
    monkeypatch.setattr('nanochat.gpt.MANAGER', manager)
    config = GPTConfig(
        n_exp=2, n_embd=4, use_kappa_swiglu=True, kappa_input='gate_proj',
        independent_kappa_router=independent_router,
        separate_base_sft_kappa=True, global_kappa_param_granularity=granularity,
    )
    experts = Qwen3MLPExperts(config)
    if granularity == 'global':
        if not independent_router:
            experts.bind_shared_kappa_bias(torch.nn.Parameter(torch.empty(2, 1)))
        experts.bind_shared_kappa_scale(torch.nn.Parameter(torch.empty(2, 1)))
    experts.kappa_phase = kappa_slot
    experts.train(training)
    with torch.no_grad():
        for parameter in experts.parameters():
            parameter.uniform_(-0.3, 0.3)
        if not independent_router:
            experts._get_kappa_bias_parameter().uniform_(-0.3, 0.3)
        experts._get_kappa_scale_parameter().uniform_(-0.3, 0.3)
    inputs = torch.randn(2, 3, 4, requires_grad=True)
    scores = torch.randn(2, 3, requires_grad=True)
    raw = torch.bmm(inputs, experts.gate_proj)
    bias = experts._materialize_kappa_bias(kappa_slot)
    scale = experts._materialize_kappa_scale(kappa_slot)
    kappa_raw = scale_grad(raw, 0.1)
    conditioning_bias = scores.unsqueeze(-1) if independent_router else bias.unsqueeze(1)
    kappa = torch.exp(torch.log(experts.kappa_slope_max_scale) * torch.tanh(
        scale.unsqueeze(1) * kappa_raw + conditioning_bias
    ))
    expected = torch.bmm(
        raw * torch.sigmoid(raw * kappa) * torch.bmm(inputs, experts.c_fc),
        experts.c_proj,
    )
    valid_mask = torch.tensor([[True, True, False], [True, False, False]])
    actual = experts(inputs, selected_gate_scores=scores, valid_score_mask=valid_mask)
    torch.testing.assert_close(actual, expected)
    if independent_router:
        assert experts._get_kappa_bias_parameter() is None
    if training:
        slope_loss = manager.aggregate('kappa_slope_l2_loss')
        expected_slope = scale.unsqueeze(1) * kappa_raw + conditioning_bias
        torch.testing.assert_close(slope_loss, expected_slope[valid_mask].square().mean())
        if independent_router:
            score_grad = torch.autograd.grad(slope_loss, scores, retain_graph=True, allow_unused=True)[0]
            assert score_grad is None
    if not independent_router:
        torch.testing.assert_close(experts(inputs, selected_gate_scores=scores * 7), actual)
        torch.testing.assert_close(experts(inputs), actual)
    if not training:
        return
    parameters = (
        inputs, experts.gate_proj,
        experts._get_kappa_scale_parameter(),
        scores if independent_router else experts._get_kappa_bias_parameter(),
    )
    actual_grads = torch.autograd.grad(actual.sum(), parameters, retain_graph=True)
    expected_grads = torch.autograd.grad(expected.sum(), parameters)
    if not independent_router:
        assert torch.autograd.grad(actual.sum(), scores, retain_graph=True, allow_unused=True)[0] is None
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize('independent_router', [False, True])
@pytest.mark.parametrize('granularity', ['per-gate', 'global'])
def test_gate_proj_kappa_dispatch(monkeypatch, independent_router, granularity):
    monkeypatch.setattr('nanochat.gpt.MANAGER', MOEManager())
    config = GPTConfig(
        n_layer=1, n_head=2, n_embd=32, n_exp=3, vocab_size=64,
        sequence_len=8, moe_start_layer=0, use_kappa_swiglu=True,
        kappa_input='gate_proj', independent_kappa_router=independent_router,
        global_kappa_param_granularity=granularity,
    )
    model = GPT(config)
    model.init_weights()
    layer = model.transformer.h[0].mlp
    assert (layer.kappa_router is not None) == independent_router
    logits = torch.randn(4, config.moe_top_k, requires_grad=True)
    probabilities = torch.randn_like(logits, requires_grad=True)
    if independent_router:
        assert layer.kappa_router.bias is None
        latent = torch.randn(4, config.n_embd, requires_grad=True)
        indices = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2]])
        scores = layer._select_kappa_scores(logits, probabilities, latent, indices)
        expected = F.linear(scale_grad(latent, 0.1), layer.kappa_router.weight).gather(-1, indices)
        torch.testing.assert_close(scores, expected)
    else:
        scores = layer._select_kappa_scores(logits, probabilities)
        torch.testing.assert_close(scores, torch.ones_like(probabilities))
        assert not scores.requires_grad
    tokens = torch.randint(0, config.vocab_size, (1, 4))
    with torch.no_grad():
        layer.experts._get_kappa_scale_parameter().fill_(0.2)
        layer.experts.c_proj.uniform_(-0.1, 0.1)
    loss, _ = model(tokens, tokens)
    loss.backward()
    assert layer.experts._get_kappa_scale_parameter().grad.abs().sum() > 0
    assert layer.experts.gate_proj.grad.abs().sum() > 0
    if independent_router:
        assert layer.kappa_router.weight.grad.abs().sum() > 0
        clone = GPT(config)
        clone.init_weights()
        clone.load_state_dict(model.state_dict())
        torch.testing.assert_close(
            clone.transformer.h[0].mlp.experts._get_kappa_scale_parameter(),
            layer.experts._get_kappa_scale_parameter(),
        )
    model.eval()
    with torch.no_grad():
        assert torch.isfinite(model(tokens)).all()


@pytest.mark.parametrize('is_sft', [False, True])
@pytest.mark.parametrize('separate_slots', [False, True])
@pytest.mark.parametrize('independent_router', [False, True])
def test_kappa_slope_is_the_only_kappa_penalty(
    monkeypatch, is_sft, separate_slots, independent_router,
):
    monkeypatch.setattr('nanochat.gpt.MANAGER', MOEManager())
    config = GPTConfig(
        n_layer=2, n_head=2, n_embd=32, n_exp=3, vocab_size=64,
        sequence_len=8, moe_start_layer=0, use_kappa_swiglu=True,
        independent_kappa_router=independent_router, separate_base_sft_kappa=separate_slots,
    )
    model = GPT(config)
    model.init_weights()
    model.set_kappa_training_phase(is_sft)
    weights = []
    if independent_router:
        with torch.no_grad():
            for block in model.transformer.h:
                weight = block.mlp.kappa_router.weight
                weight.fill_(2.0)
                if separate_slots:
                    weight.view(2, config.n_exp, -1)[1].fill_(0.3)
                weights.append(weight)
    tokens = torch.randint(0, config.vocab_size, (1, 4))
    _, losses = model(tokens, tokens)
    assert {name for name in losses if name.startswith('kappa_') and name.endswith('_loss')} == {'kappa_slope_l2_loss'}
    assert torch.isfinite(losses['kappa_slope_l2_loss'])
    if independent_router:
        losses['kappa_slope_l2_loss'].backward()
        for weight in weights:
            assert weight.grad.abs().sum() > 0


@pytest.mark.parametrize('is_sft', [False, True])
def test_independent_sft_kappa_router_is_scaled_base_plus_residual(is_sft):
    config = GPTConfig(
        n_layer=1, n_head=2, n_embd=32, n_exp=3, vocab_size=64,
        sequence_len=8, moe_start_layer=0, use_kappa_swiglu=True,
        independent_kappa_router=True, separate_base_sft_kappa=True,
    )
    model = GPT(config)
    model.init_weights()
    layer = model.transformer.h[0].mlp
    weights = layer.kappa_router.weight.view(2, config.n_exp, config.n_embd)
    assert weights[0].count_nonzero() > 0
    assert weights[1].count_nonzero() == 0
    inputs = torch.randn(2, config.n_embd, requires_grad=True)
    indices = torch.tensor([[0, 1], [1, 2]])
    model.set_kappa_training_phase(True)
    initial_scores = layer._select_kappa_scores(None, None, inputs, indices)
    torch.testing.assert_close(initial_scores, F.linear(inputs, weights[0]).gather(-1, indices))
    with torch.no_grad():
        weights[1].fill_(0.2)
    model.set_kappa_training_phase(is_sft)
    actual = layer._select_kappa_scores(None, None, inputs, indices)
    expected_weight = scale_grad(weights[0], 0.1) + weights[1] if is_sft else weights[0]
    expected = F.linear(scale_grad(inputs, 0.1), expected_weight).gather(-1, indices)
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual.sum(), (inputs, layer.kappa_router.weight), retain_graph=True)
    expected_grads = torch.autograd.grad(expected.sum(), (inputs, layer.kappa_router.weight))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
    slot_grads = actual_grads[1].view_as(weights)
    if is_sft:
        torch.testing.assert_close(slot_grads[0], 0.1 * slot_grads[1])
    else:
        assert slot_grads[1].count_nonzero() == 0
    assert slot_grads[int(is_sft)].count_nonzero() > 0


@pytest.mark.parametrize('granularity', ['per-gate', 'per-expert', 'per-layer', 'global'])
@pytest.mark.parametrize('independent_router', [False, True])
def test_disable_kappa_bias_keeps_scale_active(monkeypatch, granularity, independent_router):
    manager = MOEManager()
    monkeypatch.setattr('nanochat.gpt.MANAGER', manager)
    config = GPTConfig(
        n_exp=2, n_embd=4, use_kappa_swiglu=True, disable_kappa_bias=True,
        global_kappa_param_granularity=granularity,
        independent_kappa_router=independent_router, separate_base_sft_kappa=True,
    )
    experts = Qwen3MLPExperts(config)
    if granularity == 'global':
        experts.bind_shared_kappa_bias(torch.nn.Parameter(torch.full((2, 1), 3.0)))
        if not independent_router:
            experts.bind_shared_kappa_scale(torch.nn.Parameter(torch.full((2, 1), 0.5)))
    with torch.no_grad():
        if experts._get_kappa_bias_parameter() is not None:
            experts._get_kappa_bias_parameter().fill_(3.0)
        if experts._get_kappa_scale_parameter() is not None:
            experts._get_kappa_scale_parameter().fill_(0.5)
    scores = torch.full((2, 3), 0.7, requires_grad=True)
    raw_gate = torch.randn(2, 3, 16)
    for slot in (0, 1):
        bias = experts._materialize_kappa_bias(slot)
        torch.testing.assert_close(bias, torch.zeros(2, 16))
        scale = None if independent_router else experts._materialize_kappa_scale(slot)
        conditioning = scores.unsqueeze(-1) if independent_router else scores.unsqueeze(-1) * scale.unsqueeze(1)
        expected = raw_gate * torch.sigmoid(raw_gate * config.moe_kappa_slope_max_scale ** conditioning.tanh())
        actual = experts._apply_kappa_slope_scaled_activation(raw_gate, bias, scores, kappa_scale=scale, kappa_slot=slot)
        torch.testing.assert_close(actual, expected)
        assert torch.autograd.grad(actual.sum(), scores, retain_graph=True)[0].abs().sum() > 0
        with torch.no_grad():
            experts.eval()
            actual_eval = experts._apply_kappa_slope_scaled_activation(raw_gate, bias, scores, kappa_slot=slot)
            torch.testing.assert_close(actual_eval, expected)
            experts.train()
    assert manager.aggregate('kappa_slope_l2_loss') > 0


@pytest.mark.parametrize('granularity', ['per-gate', 'per-expert', 'per-layer', 'global'])
def test_disable_kappa_bias_dense_activation_is_standard_silu(monkeypatch, granularity):
    manager = MOEManager()
    monkeypatch.setattr('nanochat.gpt.MANAGER', manager)
    config = GPTConfig(
        n_embd=4, use_kappa_swiglu=True, constant_kappa_bias_dense_layers=True,
        disable_kappa_bias=True, global_kappa_param_granularity=granularity,
        separate_base_sft_kappa=True,
    )
    mlp = Qwen3MLP(config)
    assert mlp.has_kappa_swiglu is True
    assert mlp.kappa_swiglu_enabled is True
    mlp.set_kappa_swiglu_enabled(False)
    assert mlp.kappa_swiglu_enabled is False
    mlp.set_kappa_swiglu_enabled(True)
    assert mlp.kappa_swiglu_enabled is True
    if granularity == 'global':
        mlp.bind_shared_kappa_bias(torch.nn.Parameter(torch.full((2, 1), 3.0)))
    else:
        with torch.no_grad():
            mlp.kappa_bias.fill_(3.0)
    inputs = torch.randn(2, 3, 4)
    expected = mlp.c_proj(F.silu(mlp.gate_proj(inputs)) * mlp.c_fc(inputs))
    for slot in (0, 1):
        mlp.kappa_phase = slot
        torch.testing.assert_close(mlp._materialize_kappa_bias(slot), torch.zeros(16))
        torch.testing.assert_close(mlp(inputs), expected)
        with torch.no_grad():
            mlp.eval()
            torch.testing.assert_close(mlp(inputs), expected)
            mlp.train()
    assert manager.aggregate('kappa_slope_l2_loss') == 0


def test_disable_kappa_bias_constant_moe_activation_is_standard_silu():
    experts = Qwen3MLPExperts(GPTConfig(
        n_exp=2, n_embd=4, use_kappa_swiglu=True,
        kappa_input='constant', disable_kappa_bias=True,
    ))
    with torch.no_grad():
        experts.kappa_bias.fill_(3.0)
    raw_gate = torch.randn(2, 3, 16, requires_grad=True)
    scores = torch.ones(2, 3)
    bias = experts._materialize_kappa_bias()
    expected = F.silu(raw_gate)
    actual = experts._apply_kappa_slope_scaled_activation(raw_gate, bias, scores)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert experts.kappa_bias.grad is None
    with torch.no_grad():
        experts.eval()
        torch.testing.assert_close(
            experts._apply_kappa_slope_scaled_activation(raw_gate, bias, scores), expected,
        )


@pytest.mark.parametrize('kappa_input', ['top_logits', 'router_probs'])
@pytest.mark.parametrize('is_sft', [False, True])
@pytest.mark.parametrize('granularity', ['per-gate', 'per-expert', 'per-layer', 'global'])
def test_independent_kappa_router_direct_scale_activation(monkeypatch, kappa_input, is_sft, granularity):
    monkeypatch.setattr('nanochat.gpt.MANAGER', MOEManager())
    config = GPTConfig(
        n_exp=3, n_embd=4, use_kappa_swiglu=True, kappa_input=kappa_input,
        independent_kappa_router=True, separate_base_sft_kappa=True,
        global_kappa_param_granularity=granularity,
    )
    experts = Qwen3MLPExperts(config)
    if granularity == 'global':
        experts.bind_shared_kappa_bias(torch.nn.Parameter(torch.zeros(2, 1)))
    experts.kappa_phase = int(is_sft)
    with torch.no_grad():
        bias_param = experts._get_kappa_bias_parameter()
        bias_param.zero_()
        bias_param[int(is_sft)].fill_(0.2)
    raw_gate = torch.randn(3, 2, 16)
    predicted_scale = torch.randn(3, 2, requires_grad=True)
    bias = experts._materialize_kappa_bias()
    conditioning = 0.2 + predicted_scale.unsqueeze(-1)
    slope = experts.kappa_slope_max_scale ** torch.tanh(conditioning)
    expected = raw_gate * torch.sigmoid(raw_gate * slope)
    actual = experts._apply_kappa_slope_scaled_activation(
        raw_gate, bias, predicted_scale, kappa_slot=int(is_sft),
    )
    torch.testing.assert_close(actual, expected)
    grad_params = (predicted_scale,)
    actual_grads = torch.autograd.grad(actual.square().sum(), grad_params, retain_graph=True)
    expected_grads = torch.autograd.grad(expected.square().sum(), grad_params)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
        assert actual_grad.abs().sum() > 0
    assert experts.kappa_scale is None
    with torch.no_grad():
        experts.eval()
        actual_eval = experts._apply_kappa_slope_scaled_activation(
            raw_gate, bias, predicted_scale, kappa_slot=int(is_sft),
        )
        torch.testing.assert_close(actual_eval, expected)


@pytest.mark.parametrize('disable_bias', [False, True])
@pytest.mark.parametrize('is_sft', [False, True])
def test_kappa_router_bias_follows_disabled_expert_bias(disable_bias, is_sft):
    config = GPTConfig(
        n_exp=3, n_embd=4, use_kappa_swiglu=True,
        independent_kappa_router=True, disable_kappa_bias=disable_bias,
        separate_base_sft_kappa=True,
    )
    layer = MOELayer(config, layer_idx=0)
    assert (layer.kappa_router.bias is not None) == disable_bias
    layer.experts.kappa_phase = int(is_sft)
    with torch.no_grad():
        layer.kappa_router.weight.zero_()
        if disable_bias:
            layer.kappa_router.bias.copy_(torch.arange(6, dtype=torch.float32))
    inputs = torch.randn(2, 4, requires_grad=True)
    indices = torch.tensor([[0, 2], [1, 0]])
    scores = layer._select_kappa_scores(torch.ones(2, 2), torch.ones(2, 2), inputs, indices)
    expected = indices.float() + 3 * int(is_sft) if disable_bias else torch.zeros(2, 2)
    torch.testing.assert_close(scores, expected)
    scores.sum().backward()
    if disable_bias:
        expected_grad = torch.zeros(2, 3)
        expected_grad[int(is_sft)] = torch.tensor([2.0, 1.0, 1.0])
        torch.testing.assert_close(layer.kappa_router.bias.grad.view(2, 3), expected_grad)


@pytest.mark.parametrize('matrix_optimizer', ['muon', 'aurora'])
@pytest.mark.parametrize('is_sft', [False, True])
def test_kappa_router_bias_initialization_loading_and_optimizer(matrix_optimizer, is_sft):
    config = GPTConfig(
        n_layer=1, n_head=2, n_embd=32, n_exp=3, vocab_size=64,
        sequence_len=8, moe_start_layer=0, use_kappa_swiglu=True,
        independent_kappa_router=True, disable_kappa_bias=True,
        separate_base_sft_kappa=True,
    )
    model = GPT(config)
    model.init_weights()
    bias = model.transformer.h[0].mlp.kappa_router.bias
    torch.testing.assert_close(bias, torch.zeros(6))
    state = model.state_dict()
    del state['transformer.h.0.mlp.kappa_router.bias']
    with torch.no_grad():
        bias.fill_(4.0)
    model.load_state_dict(state, strict=True)
    torch.testing.assert_close(bias, torch.zeros(6))
    optimizer = model.setup_optimizer(matrix_optimizer=matrix_optimizer, kappa_param_delay_start_iterations=20)
    group = next(group for group in optimizer.param_groups if any(param is bias for param in group['params']))
    assert group['kind'] == 'adamw'
    assert group['name'] == 'kappa_router'
    assert group['kappa_param_delay_start_iterations'] == 20
    group['active_kappa_slot'] = int(is_sft)
    bias.grad = torch.ones_like(bias)
    group['lr'] = 0.0
    optimizer.step()
    torch.testing.assert_close(bias, torch.zeros(6))
    group['lr'] = group['initial_lr']
    optimizer.step()
    active_bias = bias.view(2, 3)
    assert active_bias[int(is_sft)].count_nonzero() == 3
    torch.testing.assert_close(active_bias[1 - int(is_sft)], torch.zeros(3))


@pytest.mark.parametrize('initial_anchor', [False, True])
@pytest.mark.parametrize('is_sft', [False, True])
@pytest.mark.parametrize('use_loss_accum', [False, True])
def test_independent_kappa_router_scale_l2_uses_combined_slope(
    monkeypatch, initial_anchor, is_sft, use_loss_accum,
):
    manager = MOEManager()
    monkeypatch.setattr('nanochat.gpt.MANAGER', manager)
    config = GPTConfig(
        n_layer=1, n_head=2, n_embd=32, n_exp=2, vocab_size=64,
        sequence_len=8, moe_start_layer=0, use_kappa_swiglu=True,
        independent_kappa_router=True, separate_base_sft_kappa=True,
        refresh_kappa_param_references=initial_anchor,
        router_tie_noise_steps=0,
    )
    model = GPT(config)
    model.init_weights()
    model.set_kappa_training_phase(is_sft)
    layer = model.transformer.h[0].mlp
    weight = layer.kappa_router.weight
    active_slice = slice(int(is_sft) * config.n_exp, (int(is_sft) + 1) * config.n_exp)
    with torch.no_grad():
        weight[active_slice].add_(0.2)
        layer.experts.kappa_bias.uniform_(-0.3, 0.3)
    inputs = torch.randn(1, 3, config.n_embd, requires_grad=True)
    accum = _UTLossAccum(inputs, config.n_layer, config.n_exp) if use_loss_accum else None
    layer(inputs, loss_accum=accum, router_layer_idx=0)
    scale_loss = (
        accum.losses[_UT_LOSS_NAMES.index('kappa_slope_l2_loss')]
        if use_loss_accum else manager.aggregate('kappa_slope_l2_loss')
    )
    with torch.no_grad():
        _, _, _, indices, ranks = layer.router(inputs)
    effective_weight = scale_grad(weight[:config.n_exp], 0.1) + weight[active_slice] if is_sft else weight[active_slice]
    scores = F.linear(inputs.detach().reshape(-1, config.n_embd), effective_weight)
    selected_scores = scores.gather(-1, indices)
    valid_assignments = ranks < layer.router.get_capacity(inputs.size(0) * inputs.size(1))
    bias = layer.experts._materialize_kappa_bias(int(is_sft))
    combined_slope = selected_scores.unsqueeze(-1) + bias[indices]
    expected_loss = combined_slope[valid_assignments].square().mean()
    torch.testing.assert_close(scale_loss, expected_loss)
    expected_weight_grad, expected_bias_grad = torch.autograd.grad(
        expected_loss, (weight, layer.experts.kappa_bias)
    )
    scale_loss.backward()
    torch.testing.assert_close(weight.grad, expected_weight_grad)
    assert inputs.grad is None
    torch.testing.assert_close(layer.experts.kappa_bias.grad, expected_bias_grad)
    assert layer.router.w_g.weight.grad is None
    assert layer.experts.kappa_scale is None
    assert 'transformer.h.0.mlp.initial_kappa_router_weight' not in model.state_dict()


def test_independent_kappa_router_materialized_scale_cache():
    config = GPTConfig(
        n_exp=2, n_embd=4, use_kappa_swiglu=True,
        independent_kappa_router=True,
        separate_base_sft_kappa=True,
    )
    experts = Qwen3MLPExperts(config)
    with pytest.raises(RuntimeError, match='cached kappa router logits'):
        experts._materialize_kappa_scale()
    logits = torch.tensor([[0.0, 2.0, 99.0], [-3.0, 99.0, 99.0]], requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    live_scale = experts._materialize_kappa_scale(selected_gate_scores=logits, valid_score_mask=mask)
    assert live_scale is logits
    live_scale.sum().backward()
    torch.testing.assert_close(logits.grad, torch.ones_like(logits))
    cached = experts._materialize_kappa_scale()
    assert not cached.requires_grad
    assert cached.grad_fn is None
    assert cached.data_ptr() != logits.data_ptr()
    torch.testing.assert_close(cached[mask], logits.detach()[mask])
    assert cached[~mask].isnan().all()
    torch.testing.assert_close(cached.nanmean(), torch.tensor(-1.0 / 3.0))
    torch.testing.assert_close(cached.abs().nanmean(), torch.tensor(5.0 / 3.0))
    experts.kappa_phase = 1
    with pytest.raises(RuntimeError, match='this slot'):
        experts._materialize_kappa_scale()
    experts._materialize_kappa_scale(selected_gate_scores=logits + 1)
    torch.testing.assert_close(experts._materialize_kappa_scale(), logits.detach() + 1)


@pytest.mark.parametrize('exponent', [0.0, 0.5, 1.0])
@pytest.mark.parametrize('kappa_input', ['top_logits', 'router_probs'])
@pytest.mark.parametrize('separate_base_sft_kappa', [False, True])
@pytest.mark.parametrize('is_sft', [False, True])
def test_independent_kappa_router_scales_only_latent_gradients(
    exponent, kappa_input, separate_base_sft_kappa, is_sft,
):
    config = GPTConfig(
        n_exp=3, n_embd=4, use_kappa_swiglu=True, kappa_input=kappa_input,
        independent_kappa_router=True, kappa_input_logit_norm_exponent=exponent,
        separate_base_sft_kappa=separate_base_sft_kappa,
    )
    layer = MOELayer(config, layer_idx=0)
    layer.experts.kappa_phase = int(is_sft)
    kappa_slot = int(is_sft) if separate_base_sft_kappa else 0
    predictor_weights = layer.kappa_router.weight.view(layer.num_kappa_router_slots, 3, 4)
    reference_config = deepcopy(config)
    reference_config.independent_kappa_router = False
    reference = MOELayer(reference_config, layer_idx=0)
    with torch.no_grad():
        effective_weight = predictor_weights[kappa_slot]
        if separate_base_sft_kappa and is_sft:
            effective_weight = effective_weight + predictor_weights[0]
        reference.router.w_g.weight.copy_(effective_weight)
    latent = torch.randn(2, 4, requires_grad=True)
    reference_latent = latent.detach().clone().requires_grad_(True)
    indices = torch.tensor([[2, 0], [1, 2]])
    scores = layer._select_kappa_scores(torch.zeros(2, 2), torch.ones(2, 2), latent, indices)
    reference_logits = reference.router.w_g(reference_latent).gather(-1, indices)
    reference_scores = reference_logits
    torch.testing.assert_close(scores, reference_scores)
    scores.square().sum().backward()
    reference_scores.square().sum().backward()
    torch.testing.assert_close(latent.grad, 0.1 * reference_latent.grad)
    predictor_grads = layer.kappa_router.weight.grad.view_as(predictor_weights)
    torch.testing.assert_close(predictor_grads[kappa_slot], reference.router.w_g.weight.grad)
    if separate_base_sft_kappa:
        assert predictor_grads[1 - kappa_slot].count_nonzero() == 0
    assert layer.router.w_g.weight.grad is None


@pytest.mark.parametrize('kappa_input', ['top_logits', 'router_probs'])
def test_independent_kappa_router_dispatch_and_gradients(monkeypatch, kappa_input):
    manager = MOEManager()
    monkeypatch.setattr('nanochat.gpt.MANAGER', manager)
    torch.manual_seed(42)
    config = GPTConfig(
        n_exp=3, n_embd=4, use_kappa_swiglu=True, kappa_input=kappa_input,
        independent_kappa_router=True, router_tie_noise_steps=0,
        use_aux_loss=False, use_router_z_loss=False, min_capacity=2,
        train_capacity=0.5, eval_capacity=0.5,
    )
    layer = MOELayer(config, layer_idx=0)
    with torch.no_grad():
        for param in layer.experts.parameters():
            param.uniform_(-0.5, 0.5)
        layer.kappa_router.weight.copy_(layer.router.w_g.weight)
        layer.experts.router_confidence_gate_bias_grad_scale.zero_()
    reference_config = deepcopy(config)
    reference_config.independent_kappa_router = False
    reference = MOELayer(reference_config, layer_idx=0)
    reference.load_state_dict({
        name: value for name, value in layer.state_dict().items()
        if name != 'kappa_router.weight'
    }, strict=False)
    with torch.no_grad():
        reference.experts.kappa_scale.fill_(1.0)
    reference._select_kappa_scores = lambda scores, probs, **kwargs: scores
    reference.experts.router_confidence_gate_bias_grad_scale.zero_()
    latent = torch.randn(1, 8, 4, requires_grad=True)
    valid_mask = torch.tensor([[True, True, True, True, True, True, False, False]])
    output = layer(latent, valid_token_mask=valid_mask)
    scale_loss = manager.aggregate('kappa_slope_l2_loss')
    reference_output = reference(latent, valid_token_mask=valid_mask)
    torch.testing.assert_close(output, reference_output)
    cached = layer.experts._materialize_kappa_scale()
    with torch.no_grad():
        _, _, _, indices, ranks = layer.router(latent, valid_token_mask=valid_mask)
        selected_logits = F.linear(latent.reshape(-1, 4), layer.kappa_router.weight).gather(-1, indices)
        valid_assignments = ranks < cached.size(1)
        expected_cache = torch.full_like(cached, float('nan'))
        expected_cache[indices[valid_assignments], ranks[valid_assignments]] = selected_logits[valid_assignments]
    torch.testing.assert_close(cached, expected_cache, equal_nan=True)
    assert valid_assignments.sum() < valid_mask.sum() * config.moe_top_k
    assert not cached.requires_grad
    assert output[:, -2:].count_nonzero() == 0
    expected_logits = F.linear(
        latent.detach().reshape(-1, 4), layer.kappa_router.weight,
    ).gather(-1, indices)
    expected_slope = expected_logits.unsqueeze(-1) + layer.experts._materialize_kappa_bias()[indices]
    expected_scale_loss = expected_slope[valid_assignments].square().mean()
    torch.testing.assert_close(scale_loss, expected_scale_loss)
    actual_scale_grad = torch.autograd.grad(
        scale_loss, layer.kappa_router.weight, retain_graph=True,
    )[0]
    expected_scale_grad = torch.autograd.grad(
        expected_scale_loss, layer.kappa_router.weight,
    )[0]
    torch.testing.assert_close(actual_scale_grad, expected_scale_grad)
    actual_latent_grad = torch.autograd.grad(scale_loss, latent, retain_graph=True, allow_unused=True)[0]
    assert actual_latent_grad is None
    assert actual_scale_grad.abs().sum() > 0
    output.square().sum().backward()
    reference_output.square().sum().backward()
    torch.testing.assert_close(layer.router.w_g.weight.grad, reference.router.w_g.weight.grad)
    assert layer.router.w_g.weight.grad.abs().sum() > 0
    assert layer.kappa_router.weight.grad.abs().sum() > 0
    assert layer.experts.kappa_scale is None
    with torch.no_grad():
        layer.eval()
        reference.eval()
        torch.testing.assert_close(
            layer(latent, valid_token_mask=valid_mask),
            reference(latent, valid_token_mask=valid_mask),
        )
        layer.experts.set_kappa_swiglu_enabled(False)
        layer(latent, valid_token_mask=valid_mask)
        assert layer.experts._cached_kappa_scale is None


@pytest.mark.parametrize('separate_base_sft_kappa', [False, True])
@pytest.mark.parametrize('has_predictor', [False, True])
def test_independent_kappa_router_loads_legacy_checkpoint_and_roundtrips(
    separate_base_sft_kappa, has_predictor,
):
    config = GPTConfig(
        n_layer=1, n_head=2, n_embd=32, n_exp=3, vocab_size=64,
        sequence_len=8, moe_start_layer=0, use_kappa_swiglu=True,
        kappa_input='top_logits',
        separate_base_sft_kappa=separate_base_sft_kappa,
        independent_kappa_router=has_predictor,
    )
    legacy = GPT(config)
    legacy.init_weights()
    with torch.no_grad():
        legacy.transformer.h[0].mlp.router.w_g.weight.normal_()
    state_dict = legacy.state_dict()
    state_dict['transformer.h.0.mlp.experts.kappa_scale'] = torch.ones_like(
        legacy.transformer.h[0].mlp.experts.kappa_bias
    )
    predictor_key = 'transformer.h.0.mlp.kappa_router.weight'
    expected_weight = legacy.transformer.h[0].mlp.router.w_g.weight
    if has_predictor:
        state_dict[predictor_key] = state_dict[predictor_key][:config.n_exp].clone()
        expected_weight = state_dict[predictor_key]
    new_config = deepcopy(config)
    new_config.independent_kappa_router = True
    new_config.refresh_kappa_param_references = True
    with torch.device('meta'):
        model = GPT(new_config)
    model.to_empty(device='cpu')
    model.init_weights()
    model.load_state_dict(state_dict, strict=True, assign=True)
    predictor = model.transformer.h[0].mlp.kappa_router.weight
    slot_weights = predictor.view(-1, config.n_exp, config.n_embd)
    torch.testing.assert_close(slot_weights[0], expected_weight)
    if separate_base_sft_kappa:
        torch.testing.assert_close(slot_weights[1], torch.zeros_like(expected_weight))
    layer = model.transformer.h[0].mlp
    assert layer.experts.initial_kappa_scale is None
    inputs = torch.randn(2, config.n_embd)
    indices = torch.tensor([[0, 1], [1, 2]])
    scores = layer._select_kappa_scores(
        torch.zeros(2, 2), torch.ones(2, 2), inputs, indices,
    )
    accum = MOEManager()
    _accumulate_kappa_slope_l2_loss(scores.unsqueeze(-1), loss_accum=accum)
    torch.testing.assert_close(accum.aggregate('kappa_slope_l2_loss'), scores.square().mean())
    assert predictor.data_ptr() != model.transformer.h[0].mlp.router.w_g.weight.data_ptr()
    with torch.no_grad():
        predictor[:config.n_exp].add_(0.2)
    reloaded = GPT(new_config)
    reloaded.load_state_dict(model.state_dict())
    torch.testing.assert_close(reloaded.transformer.h[0].mlp.kappa_router.weight, predictor)
    assert reloaded.transformer.h[0].mlp.experts.initial_kappa_scale is None
    optimizer = model.setup_optimizer()
    assert any(predictor is param for group in optimizer.param_groups for param in group['params'])


@pytest.mark.parametrize('separate_base_sft_kappa', [False, True])
@pytest.mark.parametrize('matrix_optimizer', ['muon', 'muonh', 'aurora'])
def test_kappa_router_delay_keeps_weights_fixed_and_accumulates_optimizer_state(
    separate_base_sft_kappa, matrix_optimizer,
):
    config = GPTConfig(
        n_layer=1, n_head=2, n_embd=32, n_exp=3, vocab_size=64,
        sequence_len=8, moe_start_layer=0, use_kappa_swiglu=True,
        independent_kappa_router=True, separate_base_sft_kappa=separate_base_sft_kappa,
    )
    model = GPT(config)
    model.init_weights()
    predictor = model.transformer.h[0].mlp.kappa_router.weight
    optimizer = model.setup_optimizer(
        matrix_optimizer=matrix_optimizer, matrix_lr=0.01,
        weight_decay=0.1, kappa_param_delay_start_iterations=20,
    )
    group = next(group for group in optimizer.param_groups if group.get('name') == 'kappa_router')
    assert group['params'] == [predictor]
    assert group['kappa_param_delay_start_iterations'] == 20
    assert ('active_kappa_slot' in group) == separate_base_sft_kappa
    before = predictor.detach().clone()
    for _ in range(2):
        predictor.grad = torch.ones_like(predictor)
        group['lr'] = 0.0
        optimizer.step()
        torch.testing.assert_close(predictor, before, rtol=0, atol=0)
    state = optimizer.state[predictor]
    if separate_base_sft_kappa:
        state = state['slot_states'][0]
    assert any(
        isinstance(value, torch.Tensor) and value.count_nonzero() > 0
        for value in state.values()
    )
    assert predictor.grad is not None
    group['lr'] = group['initial_lr']
    optimizer.step()
    assert not torch.equal(predictor, before)
    if separate_base_sft_kappa:
        torch.testing.assert_close(
            predictor[config.n_exp:], before[config.n_exp:], rtol=0, atol=0,
        )


@pytest.mark.parametrize('is_sft', [False, True])
@pytest.mark.parametrize('matrix_optimizer', ['muon', 'muonh', 'aurora'])
@pytest.mark.parametrize('regularization', [False, True])
@pytest.mark.parametrize('kappa_input', ['top_logits', 'gate_proj'])
def test_independent_kappa_router_checkpointed_ut_matches_gradients(monkeypatch, is_sft, matrix_optimizer, regularization, kappa_input):
    monkeypatch.setattr('nanochat.gpt.MANAGER', MOEManager())
    torch.manual_seed(42)
    config = GPTConfig(
        n_layer=1, n_head=2, n_embd=32, n_exp=3, vocab_size=64,
        sequence_len=8, moe_start_layer=0, use_kappa_swiglu=True,
        kappa_input=kappa_input, independent_kappa_router=True,
        total_ut_steps=2, separate_base_sft_kappa=True, router_tie_noise_steps=0,
        refresh_kappa_param_references=regularization,
    )
    reference = GPT(config)
    reference.init_weights()
    reference.set_kappa_training_phase(is_sft)
    with torch.no_grad():
        experts = reference.transformer.h[0].mlp.experts
        experts.c_proj.normal_(std=0.05)
    checkpoint_config = deepcopy(config)
    checkpoint_config.activation_checkpointing = True
    checkpoint_model = GPT(checkpoint_config)
    checkpoint_model.load_state_dict(reference.state_dict())
    checkpoint_model.set_kappa_training_phase(is_sft)
    tokens = torch.randint(0, 64, (2, 5))
    targets = torch.randint(0, 64, (2, 5))
    reference_loss, _ = reference(tokens, targets)
    checkpoint_loss, _ = checkpoint_model(tokens, targets)
    reference_loss.backward()
    checkpoint_loss.backward()
    torch.testing.assert_close(reference_loss, checkpoint_loss)
    for (name, param), (checkpoint_name, checkpoint_param) in zip(
        reference.named_parameters(), checkpoint_model.named_parameters()
    ):
        assert name == checkpoint_name
        if param.grad is not None:
            torch.testing.assert_close(param.grad, checkpoint_param.grad, rtol=1e-5, atol=1e-6)
    assert reference.transformer.h[0].mlp.kappa_router.weight.grad.abs().sum() > 0
    if regularization:
        assert experts.initial_kappa_bias is not None
        with torch.no_grad():
            reference.eval()
            checkpoint_model.eval()
            reference_logits = reference(tokens)
            checkpoint_logits = checkpoint_model(tokens)
            torch.testing.assert_close(reference_logits, checkpoint_logits)
    predictor_grads = reference.transformer.h[0].mlp.kappa_router.weight.grad.view(2, 3, 32)
    if is_sft:
        torch.testing.assert_close(predictor_grads[0], 0.1 * predictor_grads[1])
    else:
        assert predictor_grads[1].count_nonzero() == 0
    predictor = reference.transformer.h[0].mlp.kappa_router.weight
    optimizer = reference.setup_optimizer(matrix_optimizer=matrix_optimizer, matrix_lr=0.01, weight_decay=0.1)
    predictor_group = next(group for group in optimizer.param_groups if group.get('name') == 'kappa_router')
    assert predictor_group['kind'] == ('muon' if matrix_optimizer == 'muonh' else matrix_optimizer)
    assert predictor_group['lr'] == 0.01
    assert predictor_group['active_kappa_slot'] == int(is_sft)
    before = predictor.detach().clone().view(2, 3, 32)
    optimizer.step()
    if is_sft:
        assert not torch.equal(predictor.view(2, 3, 32)[0], before[0])
    else:
        torch.testing.assert_close(predictor.view(2, 3, 32)[1], before[1], rtol=0, atol=0)
    assert not torch.equal(predictor.view(2, 3, 32)[int(is_sft)], before[int(is_sft)])


def test_dense_gate_projection_is_applied_before_fc_gating():
    torch.manual_seed(0)
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        debug=False,
    )
    experts = Qwen3MLPExperts(config)

    x = torch.randn(config.n_exp, 5, config.n_embd)

    with torch.no_grad():
        experts.gate_proj.copy_(torch.randn_like(experts.gate_proj))
        experts.c_fc.copy_(torch.randn_like(experts.c_fc))
        experts.c_proj.copy_(torch.randn_like(experts.c_proj))
        raw_gate_out = torch.bmm(x, experts.gate_proj)
        expected_gate_out_acts = experts.act_fn(raw_gate_out)

        fc_out = torch.bmm(x, experts.c_fc)
        expected = torch.bmm(expected_gate_out_acts * fc_out, experts.c_proj)

    actual = experts(x)
    torch.testing.assert_close(actual, expected)


def test_dense_qwen3_mlp_keeps_silu_gate_when_moe_bilinear_is_enabled():
    torch.manual_seed(0)
    config = GPTConfig(
        n_embd=4,
        bilinear_mlp_moe=True,
        debug=False,
    )
    mlp = Qwen3MLP(config)
    x = torch.randn(3, 5, config.n_embd)

    with torch.no_grad():
        mlp.gate_proj.weight.copy_(torch.randn_like(mlp.gate_proj.weight))
        mlp.c_fc.weight.copy_(torch.randn_like(mlp.c_fc.weight))
        mlp.c_proj.weight.copy_(torch.randn_like(mlp.c_proj.weight))
        raw_gate_out = mlp.gate_proj(x)
        expected = mlp.c_proj(mlp.act_fn(raw_gate_out) * mlp.c_fc(x))

    actual = mlp(x)
    torch.testing.assert_close(actual, expected)


def test_moe_qwen3_mlp_uses_raw_bilinear_gate_when_enabled():
    torch.manual_seed(0)
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        bilinear_mlp_moe=True,
        debug=False,
    )
    experts = Qwen3MLPExperts(config)
    x = torch.randn(config.n_exp, 5, config.n_embd)

    with torch.no_grad():
        experts.gate_proj.copy_(torch.randn_like(experts.gate_proj))
        experts.c_fc.copy_(torch.randn_like(experts.c_fc))
        experts.c_proj.copy_(torch.randn_like(experts.c_proj))
        raw_gate_out = torch.bmm(x, experts.gate_proj)
        fc_out = torch.bmm(x, experts.c_fc)
        expected = torch.bmm(raw_gate_out * fc_out, experts.c_proj)

    actual = experts(x)
    torch.testing.assert_close(actual, expected)


def test_scale_grad_only_backprops_into_tensor_alpha():
    x_tensor_alpha = torch.tensor([2.0], requires_grad=True)
    alpha_tensor = torch.tensor([3.0], requires_grad=True)

    y_tensor_alpha = scale_grad(x_tensor_alpha, alpha_tensor)

    torch.testing.assert_close(y_tensor_alpha, x_tensor_alpha.detach())
    y_tensor_alpha.backward()

    torch.testing.assert_close(x_tensor_alpha.grad, alpha_tensor.detach())
    torch.testing.assert_close(alpha_tensor.grad, x_tensor_alpha.detach())

    x_scalar_alpha = torch.tensor([2.0], requires_grad=True)
    y_scalar_alpha = scale_grad(x_scalar_alpha, 3.0)

    torch.testing.assert_close(y_scalar_alpha, x_scalar_alpha.detach())
    y_scalar_alpha.backward()

    torch.testing.assert_close(x_scalar_alpha.grad, torch.tensor([3.0]))

    x_nograd_tensor_alpha = torch.tensor([2.0], requires_grad=True)
    alpha_nograd_tensor = torch.tensor([3.0])
    y_nograd_tensor_alpha = scale_grad(x_nograd_tensor_alpha, alpha_nograd_tensor)

    torch.testing.assert_close(y_nograd_tensor_alpha, x_nograd_tensor_alpha.detach())
    y_nograd_tensor_alpha.backward()

    torch.testing.assert_close(x_nograd_tensor_alpha.grad, alpha_nograd_tensor)


def test_kappa_bias_can_rescale_kappa_slope_from_router_probs():
    torch.manual_seed(0)
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_kappa_swiglu=True,
        debug=False,
    )
    experts = Qwen3MLPExperts(config)

    x = torch.randn(config.n_exp, 5, config.n_embd)
    router_probs = torch.rand(config.n_exp, 5)

    with torch.no_grad():
        experts.gate_proj.copy_(torch.randn_like(experts.gate_proj))
        experts.kappa_bias.copy_(torch.randn_like(experts.kappa_bias))
        experts.kappa_scale.copy_(torch.randn_like(experts.kappa_scale))
        experts.c_fc.copy_(torch.randn_like(experts.c_fc))
        experts.c_proj.copy_(torch.randn_like(experts.c_proj))

        raw_gate_out = torch.bmm(x, experts.gate_proj)
        slope_work = experts._materialize_kappa_bias().unsqueeze(1) + (
            router_probs.unsqueeze(-1)
            * experts._materialize_kappa_scale().unsqueeze(1)
        )
        slope_scales = torch.exp(
            torch.log(experts.kappa_slope_max_scale) * torch.tanh(slope_work)
        )
        expected_gate_out_acts = raw_gate_out * torch.sigmoid(raw_gate_out * slope_scales)

        fc_out = torch.bmm(x, experts.c_fc)
        expected = torch.bmm(expected_gate_out_acts * fc_out, experts.c_proj)

    actual = experts(x, selected_gate_scores=router_probs)
    torch.testing.assert_close(actual, expected)

@pytest.mark.parametrize("granularity", ["per-gate", "per-expert", "per-layer", "global"])
@pytest.mark.parametrize("separate_base_sft_kappa", [False, True])
def test_kappa_bias_is_independent_of_scale_and_eval_cache(granularity, separate_base_sft_kappa):
    config = GPTConfig(
        n_exp=2, n_embd=4, use_kappa_swiglu=True,
        global_kappa_param_granularity=granularity,
        total_ut_steps=3, separate_base_sft_kappa=separate_base_sft_kappa,
    )
    experts = Qwen3MLPExperts(config, layer_idx=1)
    if granularity == "global":
        experts.bind_shared_kappa_bias(torch.nn.Parameter(torch.zeros(experts.num_kappa_slots, 1)))
        experts.bind_shared_kappa_scale(torch.nn.Parameter(torch.ones(experts.num_kappa_slots, 1)))
    scale = experts._get_kappa_scale_parameter()
    bias_parameter = experts._get_kappa_bias_parameter()
    assert bias_parameter is not scale
    with torch.no_grad():
        bias_parameter.fill_(2.0)
        scale.fill_(3.0)
    for slot in range(experts.num_kappa_slots):
        bias = experts._materialize_kappa_bias(slot)
        torch.testing.assert_close(bias, torch.full_like(bias, 4.0 if slot == 1 else 2.0))
    bias.sum().backward()
    assert bias_parameter.grad[-1].abs().sum() > 0
    assert scale.grad is None
    experts.eval()
    with torch.no_grad():
        cached = experts._get_kappa_bias_unsqueezed_for_eval(torch.float32, scale.device, 0)
        torch.testing.assert_close(cached, torch.full_like(cached, 2.0))
        bias_parameter.fill_(4.0)
        updated = experts._get_kappa_bias_unsqueezed_for_eval(torch.float32, scale.device, 0)
        torch.testing.assert_close(updated, torch.full_like(updated, 4.0))
        scale.fill_(5.0)
        unchanged = experts._get_kappa_bias_unsqueezed_for_eval(torch.float32, scale.device, 0)
        assert unchanged is updated
        torch.testing.assert_close(unchanged, torch.full_like(unchanged, 4.0))


def test_kappa_bias_and_scale_are_regularized_only_through_combined_slope():
    config = GPTConfig(
        n_exp=2, n_embd=4, use_kappa_swiglu=True,
    )
    experts = Qwen3MLPExperts(config)
    with torch.no_grad():
        experts.kappa_scale.fill_(3.0)
        experts.kappa_bias.fill_(6.0)
    accum = MOEManager()
    raw_gate = torch.ones(2, 3, 16)
    experts._apply_kappa_slope_scaled_activation(
        raw_gate, experts._materialize_kappa_bias(), torch.full((2, 3), -2.0),
        kappa_scale=experts._materialize_kappa_scale(), loss_accum=accum,
    )
    slope_loss = accum.aggregate("kappa_slope_l2_loss")
    assert slope_loss.item() == 0.0
    slope_loss.backward()
    assert experts.kappa_scale.grad.count_nonzero() == 0
    assert experts.kappa_bias.grad.count_nonzero() == 0
    experts.zero_grad()
    accum.reset("kappa_slope_l2_loss")
    experts._apply_kappa_slope_scaled_activation(
        raw_gate, experts._materialize_kappa_bias(), torch.ones(2, 3),
        kappa_scale=experts._materialize_kappa_scale(), loss_accum=accum,
    )
    accum.aggregate("kappa_slope_l2_loss").backward()
    assert experts.kappa_scale.grad.abs().sum() > 0
    assert experts.kappa_bias.grad.abs().sum() > 0


def test_gate_activation_stats_match_logged_formulas():
    torch.manual_seed(0)
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        gate_stats_threshold=0.2,
        gate_stats_topk=3,
        debug=False,
    )
    experts = Qwen3MLPExperts(config)

    x = torch.randn(config.n_exp, 5, config.n_embd)

    with torch.no_grad():
        experts.gate_proj.copy_(torch.randn_like(experts.gate_proj))
        experts.c_fc.copy_(torch.randn_like(experts.c_fc))
        experts.c_proj.copy_(torch.randn_like(experts.c_proj))

    old_collect = MANAGER.collect_load_balancing_stats
    MANAGER.collect_load_balancing_stats = True
    try:
        _ = experts(x)
    finally:
        MANAGER.collect_load_balancing_stats = old_collect

    gate = experts.act_fn(torch.bmm(x, experts.gate_proj)).abs().float()
    gate_sum = gate.sum(dim=-1)
    gate_probs = gate / gate_sum.clamp_min(1e-8).unsqueeze(-1)
    expected_mean_abs_gate = gate.mean()
    expected_active_frac = gate.gt(config.gate_stats_threshold).float().mean()
    expected_topk_share = (
        gate.topk(config.gate_stats_topk, dim=-1).values.sum(dim=-1)
        / gate_sum.clamp_min(1e-8)
    ).mean()
    expected_entropy = -(
        gate_probs * gate_probs.clamp_min(1e-8).log()
    ).sum(dim=-1).mean()

    assert experts.last_gate_stats is not None
    torch.testing.assert_close(experts.last_gate_stats['mean_abs_gate'], expected_mean_abs_gate)
    torch.testing.assert_close(experts.last_gate_stats['active_frac'], expected_active_frac)
    torch.testing.assert_close(experts.last_gate_stats['topk_share'], expected_topk_share)
    torch.testing.assert_close(experts.last_gate_stats['entropy'], expected_entropy)


def test_dynamic_kappa_bias_backprops_into_selected_gate_scores():
    torch.manual_seed(0)
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_kappa_swiglu=True,
        debug=False,
    )
    experts = Qwen3MLPExperts(config)

    x = torch.randn(config.n_exp, 5, config.n_embd, requires_grad=True)
    selected_gate_scores = torch.randn(config.n_exp, 5, requires_grad=True)
    out = experts(x, selected_gate_scores=selected_gate_scores).sum()
    out.backward()

    assert selected_gate_scores.grad is not None


def test_dynamic_kappa_bias_scales_selected_router_score_gradients():
    torch.manual_seed(0)
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_kappa_swiglu=True,
        debug=False,
    )
    experts = Qwen3MLPExperts(config)

    with torch.no_grad():
        experts.gate_proj.fill_(0.1)
        experts.c_fc.fill_(0.2)
        experts.c_proj.fill_(0.3)
        experts.kappa_bias.fill_(0.05)

    x = torch.randn(config.n_exp, 5, config.n_embd)
    selected_gate_scores = torch.randn(config.n_exp, 5)

    experts.router_confidence_gate_bias_grad_scale.fill_(1.0)
    selected_gate_scores_full = selected_gate_scores.clone().requires_grad_(True)
    experts(x, selected_gate_scores=selected_gate_scores_full).sum().backward()
    grad_full = selected_gate_scores_full.grad.clone()

    experts.zero_grad(set_to_none=True)
    experts.router_confidence_gate_bias_grad_scale.fill_(0.25)
    selected_gate_scores_scaled = selected_gate_scores.clone().requires_grad_(True)
    experts(x, selected_gate_scores=selected_gate_scores_scaled).sum().backward()
    grad_scaled = selected_gate_scores_scaled.grad.clone()

    torch.testing.assert_close(grad_scaled, grad_full * 0.25, rtol=1e-4, atol=1e-6)


def test_router_returns_selected_top_k_router_scores():
    torch.manual_seed(0)
    config = GPTConfig(
        n_exp=4,
        moe_top_k=2,
        n_embd=4,
        use_noisy_top_k=False,
        use_aux_loss=False,
        use_router_z_loss=False,
        debug=False,
    )
    router = Router(config)
    x = torch.randn(2, 3, config.n_embd)

    _, router_probs, selected_router_scores, top_k_indices, _ = router(x)

    logits = F.linear(x.view(-1, config.n_embd), router.w_g.weight)
    expected_scores = logits.gather(-1, top_k_indices) * router_probs.gt(0)
    torch.testing.assert_close(selected_router_scores, expected_scores)
    MANAGER._selected_scores_buffer = None
    MANAGER._selected_scores_size = 0


def test_router_uses_unscaled_softmax_and_receives_gradient():
    config = GPTConfig(
        n_exp=3,
        moe_top_k=2,
        n_embd=2,
        eval_capacity=100.0,
        use_aux_loss=False,
        use_router_z_loss=False,
    )
    router = Router(config).eval()
    with torch.no_grad():
        router.w_g.weight.copy_(torch.tensor([[2.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]))
    x = torch.tensor([[[1.0, 0.0]]])

    _, router_probs, _, top_k_indices, _ = router(x)

    assert torch.equal(top_k_indices, torch.tensor([[0, 1]]))
    expected_probs = F.softmax(torch.tensor([[2.0, 1.0]]), dim=-1)
    torch.testing.assert_close(router_probs, expected_probs)

    router_probs.square().sum().backward()
    assert router.w_g.weight.grad is not None
    assert router.w_g.weight.grad.abs().sum() > 0


def test_zero_initialized_router_only_randomly_breaks_first_ten_training_ties():
    torch.manual_seed(0)
    config = GPTConfig(
        n_exp=4,
        moe_top_k=2,
        n_embd=4,
        train_capacity=100.0,
        use_aux_loss=False,
        use_router_z_loss=False,
        debug=False,
    )
    router = Router(config).train()
    with torch.no_grad():
        router.w_g.weight.zero_()
    x = torch.randn(8, 16, config.n_embd)

    router.set_training_step(9)
    early_indices = router(x)[3]
    assert torch.unique(early_indices).numel() == config.n_exp

    router.set_training_step(10)
    rng_before = torch.random.get_rng_state()
    late_indices = router(x)[3]
    rng_after = torch.random.get_rng_state()

    assert torch.equal(late_indices, late_indices[:1].expand_as(late_indices))
    assert torch.equal(rng_after, rng_before)
    MANAGER._selected_scores_buffer = None
    MANAGER._selected_scores_size = 0


def test_dense_qwen3_gate_projection_has_no_bias_parameter():
    config = GPTConfig(
        n_exp=1,
        n_embd=4,
        use_kappa_swiglu=True,
        debug=False,
    )

    mlp = Qwen3MLP(config)

    assert not hasattr(mlp, 'kappa_bias')


def test_router_valid_token_mask_excludes_padding_from_capacity():
    config = GPTConfig(
        n_exp=2,
        moe_top_k=1,
        n_embd=4,
        train_capacity=1.0,
        use_noisy_top_k=False,
        use_aux_loss=False,
        use_router_z_loss=False,
        debug=False,
    )
    router = Router(config)
    router.train()
    x = torch.ones(1, 4, config.n_embd)
    valid_token_mask = torch.tensor([[True, True, False, False]])

    expert_mask, router_probs, _, _, rank = router(
        x, valid_token_mask=valid_token_mask
    )
    capacity = router.get_capacity(x.shape[0] * x.shape[1])

    assert (rank[valid_token_mask.reshape(-1)] < capacity).all()
    assert (rank[~valid_token_mask.reshape(-1)] >= capacity).all()
    assert not expert_mask[~valid_token_mask.reshape(-1)].any()
    assert not router_probs[~valid_token_mask.reshape(-1)].any()


def test_router_projection_receives_gradients_and_roundtrips():
    config = GPTConfig(
        n_layer=3,
        moe_start_layer=1,
        n_exp=2,
        n_embd=8,
        n_head=2,
        use_router_z_loss=False,
    )
    router = Router(config).eval()
    x = torch.randn(2, 3, config.n_embd)
    logits = F.linear(x.view(-1, config.n_embd), router.w_g.weight)
    actual_scores, actual_indices = router(x)[2:4]
    torch.testing.assert_close(actual_scores, logits.gather(-1, actual_indices))
    assert router.w_g.weight.requires_grad

    actual_scores.sum().backward()
    assert router.w_g.weight.grad is not None
    assert router.w_g.weight.grad.abs().sum() > 0
    reloaded_router = Router(config)
    reloaded_router.load_state_dict(router.state_dict(), strict=True)
    torch.testing.assert_close(reloaded_router.w_g.weight, router.w_g.weight)


def test_no_expert_rate_tracks_joint_assignment_drops():
    rank = torch.tensor([
        [0, 0],
        [1, 2],
        [2, 1],
        [2, 2],
        [2, 2],
    ])
    exp_capacity = 2
    valid_token_mask = torch.tensor([[True, True, True, True, False]])
    valid_mask = rank.reshape(-1) < exp_capacity
    flat_top_k_indices = torch.zeros(rank.numel(), dtype=torch.long)
    layer_stub = type("LayerStub", (), {"n_exp": 1})()

    old_collect = MANAGER.collect_load_balancing_stats
    MANAGER.collect_load_balancing_stats = True
    MANAGER.reset("drop_rate_per_ks")
    MANAGER.reset("no_expert_rates")
    try:
        MOELayer._maybe_collect_load_balancing_stats(
            layer_stub,
            rank,
            flat_top_k_indices,
            valid_mask,
            exp_capacity,
            valid_token_mask,
        )
        drop_rates = MANAGER.aggregate("drop_rate_per_ks")
        no_expert_rates = MANAGER.aggregate("no_expert_rates")
    finally:
        MANAGER.collect_load_balancing_stats = old_collect
        MANAGER.reset("drop_rate_per_ks")
        MANAGER.reset("no_expert_rates")

    torch.testing.assert_close(drop_rates, torch.tensor([[0.5, 0.5]]))
    torch.testing.assert_close(no_expert_rates, torch.tensor([0.25]))


@pytest.mark.parametrize(
    "valid_token_mask",
    [
        torch.tensor([[True, True, False], [True, False, False]]),
        torch.tensor([[True, True, True], [True, True, False]]),
    ],
)
def test_router_masked_reductions_match_compacted_reference(valid_token_mask):
    torch.manual_seed(0)
    config = GPTConfig(
        n_exp=3,
        moe_top_k=2,
        n_embd=4,
        use_noisy_top_k=False,
        use_aux_loss=True,
        use_router_z_loss=False,
        debug=False,
    )
    router = Router(config)
    expert_probs = torch.softmax(torch.randn(2, 3, config.n_exp), dim=-1)
    top_k_indices = torch.topk(expert_probs, config.moe_top_k, dim=-1).indices

    actual_aux_loss = router.compute_aux_loss(
        expert_probs, top_k_indices, valid_token_mask
    )
    compact_indices = top_k_indices[valid_token_mask]
    compact_probs = expert_probs[valid_token_mask]
    compact_one_hot = F.one_hot(compact_indices, num_classes=config.n_exp).float()
    expected_aux_loss = config.n_exp * torch.sum(
        compact_probs.mean(dim=0) * compact_one_hot.sum(dim=1).mean(dim=0)
    )
    torch.testing.assert_close(actual_aux_loss, expected_aux_loss)

    router.set_aux_free_load_balancing(True)
    router._accumulate_aux_free_load_balancing_counts(
        top_k_indices.reshape(-1, config.moe_top_k), valid_token_mask.reshape(-1)
    )
    expected_counts = torch.bincount(
        compact_indices.reshape(-1), minlength=config.n_exp
    ).float()
    torch.testing.assert_close(router.tokens_per_expert_counter, expected_counts)


def test_config_allows_constant_dense_kappa_bias_with_router_probs_for_moe_layers():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_kappa_swiglu=True,
        kappa_input="router_probs",
        kappa_input_constant=1.0,
        constant_kappa_bias_dense_layers=True,
        debug=False,
    )

    assert config.kappa_input == "router_probs"
    assert config.kappa_input_constant == pytest.approx(1.0)
    assert config.constant_kappa_bias_dense_layers is True


def test_dense_qwen3_mlp_enables_constant_kappa_bias_when_requested():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_kappa_swiglu=True,
        kappa_input="router_probs",
        kappa_input_constant=1.0,
        constant_kappa_bias_dense_layers=True,
        debug=False,
    )

    mlp = Qwen3MLP(config, layer_idx=0)
    experts = Qwen3MLPExperts(config, layer_idx=0)

    assert mlp.use_kappa_swiglu is True
    assert mlp.kappa_bias is not None
    assert experts.use_kappa_swiglu is True
    assert experts.use_kappa_scale_param is True


def test_kappa_swiglu_runtime_toggle_preserves_parameters():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_kappa_swiglu=True,
        kappa_input="router_probs",
        constant_kappa_bias_dense_layers=True,
        debug=False,
    )
    mlp = Qwen3MLP(config, layer_idx=0)
    experts = Qwen3MLPExperts(config, layer_idx=0)
    mlp_kappa_bias = mlp.kappa_bias
    experts_kappa_bias = experts.kappa_bias

    mlp.set_kappa_swiglu_enabled(False)
    experts.set_kappa_swiglu_enabled(False)

    assert mlp.kappa_swiglu_enabled is False
    assert experts.kappa_swiglu_enabled is False
    assert mlp.kappa_bias is mlp_kappa_bias
    assert experts.kappa_bias is experts_kappa_bias

    mlp.set_kappa_swiglu_enabled(True)
    experts.set_kappa_swiglu_enabled(True)

    assert mlp.kappa_swiglu_enabled is True
    assert experts.kappa_swiglu_enabled is True


def test_disabled_dense_kappa_swiglu_uses_standard_activation_without_l2_loss():
    config = GPTConfig(
        n_embd=4,
        use_kappa_swiglu=True,
        kappa_input="constant",
        constant_kappa_bias_dense_layers=True,
        debug=False,
    )
    mlp = Qwen3MLP(config, layer_idx=0).train()
    with torch.no_grad():
        mlp.kappa_bias.fill_(1.0)
    x = torch.randn(2, 3, config.n_embd)
    gate_out_raw = mlp.gate_proj(x)
    expected = mlp.c_proj(mlp.act_fn(gate_out_raw) * mlp.c_fc(x))
    MANAGER.reset("kappa_slope_l2_loss")

    mlp.set_kappa_swiglu_enabled(False)
    disabled_output = mlp(x)

    torch.testing.assert_close(disabled_output, expected)
    assert MANAGER.aggregate("kappa_slope_l2_loss") == 0

    mlp.set_kappa_swiglu_enabled(True)
    enabled_output = mlp(x)

    assert not torch.allclose(enabled_output, expected)
    assert MANAGER.aggregate("kappa_slope_l2_loss") > 0
    MANAGER.reset("kappa_slope_l2_loss")


def test_dense_qwen3_mlp_uses_placeholder_bias_before_start_layer():
    torch.manual_seed(0)
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_kappa_swiglu=True,
        kappa_input="router_probs",
        kappa_input_constant=1.0,
        constant_kappa_bias_dense_layers=True,
        kappa_start_layer=2,
        debug=False,
    )

    mlp = Qwen3MLP(config, layer_idx=0)
    x = torch.randn(3, 5, config.n_embd)

    assert mlp.use_kappa_swiglu is True
    assert mlp.has_kappa_swiglu is False
    mlp.set_kappa_swiglu_enabled(True)
    assert mlp.kappa_swiglu_enabled is False
    assert not hasattr(mlp, 'kappa_bias')

    with torch.no_grad():
        mlp.gate_proj.weight.copy_(torch.randn_like(mlp.gate_proj.weight))
        mlp.c_fc.weight.copy_(torch.randn_like(mlp.c_fc.weight))
        mlp.c_proj.weight.copy_(torch.randn_like(mlp.c_proj.weight))
        raw_gate_out = mlp.gate_proj(x)
        expected = mlp.c_proj(mlp.act_fn(raw_gate_out) * mlp.c_fc(x))

    actual = mlp(x)
    torch.testing.assert_close(actual, expected)


def test_kappa_bias_lr_scale_defaults_and_overrides_from_config():
    default_config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_kappa_swiglu=True,
        debug=False,
    )
    override_config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_kappa_swiglu=True,
        debug=False,
    )

    default_moe = Qwen3MLPExperts(default_config)
    override_moe = Qwen3MLPExperts(override_config)


def test_gpt_sets_router_confidence_gate_bias_grad_scale_for_all_qwen3_moe_experts():
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=3,
        moe_start_layer=0,
        num_moe_layers=2,
        moe_layer_stride=1,
        n_exp=2,
        n_embd=32,
        n_head=4,
        use_aux_loss=False,
        use_router_z_loss=False,
        use_kappa_swiglu=True,
        use_qwen3_moe_mlp=True,
        debug=False,
    )

    model = GPT(config)
    model.set_router_confidence_gate_bias_grad_scale(0.125)

    found_experts = 0
    for block in model.transformer.h:
        mlp = getattr(block, 'mlp', None)
        if hasattr(mlp, 'experts') and isinstance(mlp.experts, Qwen3MLPExperts):
            found_experts += 1
            assert mlp.experts.router_confidence_gate_bias_grad_scale == 0.125

    assert found_experts == 2


def test_gpt_train_clears_kappa_evaluation_caches():
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=2,
        moe_start_layer=0,
        num_moe_layers=1,
        n_exp=2,
        n_embd=32,
        n_head=4,
        use_kappa_swiglu=True,
        debug=False,
    )
    model = GPT(config)
    cache_attributes = [
        (module, name)
        for module in model.modules()
        for name in vars(module)
        if name.startswith('_eval_kappa_')
    ]
    assert cache_attributes
    for module, name in cache_attributes:
        setattr(module, name, object())

    model.train()

    assert all(getattr(module, name) is None for module, name in cache_attributes)


def test_gpt_total_ut_steps_populates_distinct_kv_cache_layers():
    torch.manual_seed(0)
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=2,
        n_exp=1,
        n_embd=32,
        n_head=4,
        total_ut_steps=2,
        use_aux_loss=False,
        use_router_z_loss=False,
        debug=False,
    )

    model = GPT(config)
    model.init_weights()
    ids = torch.randint(0, config.vocab_size, (1, 3))
    cache = KVCache(
        batch_size=1,
        num_heads=config.n_kv_head,
        seq_len=config.sequence_len,
        head_dim=config.n_embd // config.n_head,
        num_layers=config.n_layer * config.total_ut_steps,
        device="cpu",
        dtype=torch.float32,
    )

    logits = model(ids, kv_cache=cache)

    assert logits.shape == (1, ids.size(1), config.vocab_size)
    assert cache.get_pos() == ids.size(1)

    for layer_idx in range(config.n_layer * config.total_ut_steps):
        k_layer, v_layer = cache.get_layer_cache(layer_idx)
        assert k_layer[:, : ids.size(1)].abs().sum().item() > 0.0
        assert v_layer[:, : ids.size(1)].abs().sum().item() > 0.0


@pytest.mark.parametrize("activation_checkpointing", [False, True])
def test_gpt_total_ut_steps_use_distinct_scalars_with_token_embedding_anchor(
    activation_checkpointing,
):
    torch.manual_seed(0)
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=2,
        n_exp=1,
        n_embd=32,
        n_head=4,
        total_ut_steps=2,
        activation_checkpointing=activation_checkpointing,
        use_aux_loss=False,
        use_router_z_loss=False,
        debug=False,
    )
    model = GPT(config)
    model.init_weights()
    with torch.no_grad():
        model.resid_lambdas.zero_()
        model.x0_lambdas.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        model.ut_source_lambdas.zero_()

    block_inputs = [[], []]
    hooks = []

    def capture_block_input(layer_idx):
        def hook(_module, args):
            block_inputs[layer_idx].append(args[0].detach().clone())
        return hook

    for layer_idx, block in enumerate(model.transformer.h):
        hooks.append(block.register_forward_pre_hook(capture_block_input(layer_idx)))
    try:
        ids = torch.randint(0, config.vocab_size, (1, 3))
        token_x0 = F.rms_norm(model.transformer.wte(ids), (config.n_embd,)).detach()
        model(ids, targets=ids)
    finally:
        for hook in hooks:
            hook.remove()

    assert model.resid_lambdas.shape == (2, 2)
    assert model.x0_lambdas.shape == (2, 2)
    assert [len(inputs) for inputs in block_inputs] == [2, 2]
    torch.testing.assert_close(block_inputs[0][0], token_x0)
    torch.testing.assert_close(block_inputs[1][0], 2.0 * token_x0)
    torch.testing.assert_close(block_inputs[0][1], 3.0 * token_x0)
    torch.testing.assert_close(block_inputs[1][1], 4.0 * token_x0)


def test_gpt_ut_mixes_previous_pass_source_only_at_destination():
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=4,
        n_exp=1,
        n_embd=8,
        n_head=2,
        total_ut_steps=2,
        ut_source=1,
        ut_destination=-2,
        use_aux_loss=False,
        use_router_z_loss=False,
        debug=False,
    )
    model = GPT(config)
    model.init_weights()
    with torch.no_grad():
        model.resid_lambdas.fill_(1.0)
        model.x0_lambdas.zero_()
        model.ut_source_lambdas[1] = 0.5

    for layer_idx, block in enumerate(model.transformer.h):
        offset = float(layer_idx + 1)
        block.forward = lambda x, *args, offset=offset, **kwargs: x + offset

    destination_inputs = []
    downstream_inputs = []
    source_outputs = []
    source_hook = model.transformer.h[1].register_forward_hook(
        lambda _module, _args, output: source_outputs.append(output.detach().clone())
    )
    destination_hook = model.transformer.h[2].register_forward_pre_hook(
        lambda _module, args: destination_inputs.append(args[0].detach().clone())
    )
    downstream_hook = model.transformer.h[3].register_forward_pre_hook(
        lambda _module, args: downstream_inputs.append(args[0].detach().clone())
    )
    try:
        ids = torch.randint(0, config.vocab_size, (1, 3))
        model(ids)
    finally:
        source_hook.remove()
        destination_hook.remove()
        downstream_hook.remove()

    assert len(source_outputs) == 2
    assert len(destination_inputs) == 2
    assert len(downstream_inputs) == 2
    assert model.ut_source_lambdas.shape == (2,)
    torch.testing.assert_close(
        model.ut_source_lambdas,
        torch.tensor([0.0, 0.5]),
    )
    torch.testing.assert_close(
        destination_inputs[1],
        source_outputs[1] + 0.5 * source_outputs[0],
    )
    torch.testing.assert_close(
        downstream_inputs[1],
        destination_inputs[1] + 3.0,
    )


def test_gpt_ut_uses_source_as_only_cross_pass_activation():
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=2,
        n_exp=1,
        n_embd=8,
        n_head=2,
        total_ut_steps=2,
        ut_source=-1,
        ut_destination=0,
        use_aux_loss=False,
        use_router_z_loss=False,
        debug=False,
    )
    model = GPT(config)
    model.init_weights()
    with torch.no_grad():
        model.resid_lambdas.fill_(1.0)
        model.x0_lambdas.zero_()

    for layer_idx, block in enumerate(model.transformer.h):
        offset = float(layer_idx + 1)
        block.forward = lambda x, *args, offset=offset, **kwargs: x + offset

    first_layer_inputs = []
    final_layer_outputs = []
    first_hook = model.transformer.h[0].register_forward_pre_hook(
        lambda _module, args: first_layer_inputs.append(args[0].detach().clone())
    )
    final_hook = model.transformer.h[-1].register_forward_hook(
        lambda _module, _args, output: final_layer_outputs.append(output.detach().clone())
    )
    try:
        ids = torch.randint(0, config.vocab_size, (1, 3))
        model(ids)
    finally:
        first_hook.remove()
        final_hook.remove()

    expected_next_pass_input = first_layer_inputs[0] + final_layer_outputs[0]
    torch.testing.assert_close(first_layer_inputs[1], expected_next_pass_input)


@pytest.mark.parametrize("field", ["ut_source", "ut_destination"])
def test_gpt_config_rejects_out_of_range_ut_layer_indices(field):
    with pytest.raises(ValueError, match=field):
        GPTConfig(n_layer=4, total_ut_steps=2, **{field: 4})


def test_gpt_value_embedding_inputs_have_consistent_grad_state():
    torch.manual_seed(0)
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=3,
        n_exp=1,
        n_embd=32,
        n_head=4,
        use_aux_loss=False,
        use_router_z_loss=False,
        debug=False,
    )
    model = GPT(config)
    model.init_weights()
    ids = torch.randint(0, config.vocab_size, (1, 3))
    ve_requires_grad = []
    router_layer_indices = []

    def capture_block_inputs(_module, args, kwargs):
        ve_requires_grad.append(args[1].requires_grad)
        router_layer_indices.append(kwargs["router_layer_idx"])

    hooks = [
        block.register_forward_pre_hook(capture_block_inputs, with_kwargs=True)
        for block in model.transformer.h
    ]

    model.train()
    model(ids, targets=ids)
    assert ve_requires_grad == [True] * config.n_layer
    assert all(torch.is_tensor(layer_idx) for layer_idx in router_layer_indices)
    assert [layer_idx.item() for layer_idx in router_layer_indices] == list(range(config.n_layer))

    ve_requires_grad.clear()
    router_layer_indices.clear()
    model.eval()
    with torch.inference_mode():
        model(ids)
    assert ve_requires_grad == [False] * config.n_layer
    assert [layer_idx.item() for layer_idx in router_layer_indices] == list(range(config.n_layer))

    for hook in hooks:
        hook.remove()


def test_gpt_total_ut_steps_moe_training_backward_uses_no_persistent_grad_buffers():
    torch.manual_seed(0)
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=3,
        moe_start_layer=0,
        num_moe_layers=-1,
        moe_layer_stride=1,
        n_exp=2,
        moe_top_k=2,
        n_embd=32,
        n_head=4,
        total_ut_steps=2,
        use_aux_loss=False,
        use_router_z_loss=False,
        use_qwen3_moe_mlp=True,
        use_qwen3_dense_mlp=True,
        debug=False,
    )

    model = GPT(config)
    model.init_weights()
    ids = torch.randint(0, config.vocab_size, (2, 5))
    targets = torch.randint(0, config.vocab_size, (2, 5))

    first_loss, losses = model(ids, targets)
    first_loss.backward()

    model.zero_grad(set_to_none=True)
    second_loss, _ = model(ids, targets)
    second_loss.backward()

    assert torch.isfinite(first_loss)
    assert torch.isfinite(second_loss)
    assert losses['ntp_loss'].item() >= 0.0
    for block in model.transformer.h:
        assert block.mlp._expert_inputs_cache is None
        assert block.mlp._expert_gate_scores_cache is None


def test_gpt_total_ut_steps_averages_ntp_loss_from_each_loop(monkeypatch):
    torch.manual_seed(0)
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=1,
        n_exp=1,
        n_embd=32,
        n_head=4,
        total_ut_steps=2,
        use_aux_loss=False,
        use_router_z_loss=False,
        debug=False,
    )
    model = GPT(config)
    model.init_weights()
    ids = torch.randint(0, config.vocab_size, (2, 5))
    targets = torch.randint(0, config.vocab_size, (2, 5))
    loop_losses = []
    recompute_backward_values = []
    original_chunked_cross_entropy = _chunked_cross_entropy

    def capture_loop_loss(*args, **kwargs):
        recompute_backward_values.append(kwargs["recompute_backward"])
        loop_loss = original_chunked_cross_entropy(*args, **kwargs)
        loop_losses.append(loop_loss.detach())
        return loop_loss

    monkeypatch.setattr("nanochat.gpt._chunked_cross_entropy", capture_loop_loss)
    loss, losses = model(ids, targets)

    assert len(loop_losses) == config.total_ut_steps
    assert recompute_backward_values == [True] * config.total_ut_steps
    expected_loss = torch.stack(loop_losses).mean()
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(losses["ntp_loss"], expected_loss)


def test_gpt_total_ut_steps_can_compute_ntp_loss_only_on_final_loop(monkeypatch):
    torch.manual_seed(0)
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=1,
        n_exp=1,
        n_embd=32,
        n_head=4,
        total_ut_steps=2,
        ut_everypass_ntp=False,
        use_aux_loss=False,
        use_router_z_loss=False,
        debug=False,
    )
    model = GPT(config)
    model.init_weights()
    ids = torch.randint(0, config.vocab_size, (2, 5))
    targets = torch.randint(0, config.vocab_size, (2, 5))
    loop_losses = []
    recompute_backward_values = []
    original_chunked_cross_entropy = _chunked_cross_entropy

    def capture_loop_loss(*args, **kwargs):
        recompute_backward_values.append(kwargs["recompute_backward"])
        loop_loss = original_chunked_cross_entropy(*args, **kwargs)
        loop_losses.append(loop_loss.detach())
        return loop_loss

    monkeypatch.setattr("nanochat.gpt._chunked_cross_entropy", capture_loop_loss)
    loss, losses = model(ids, targets)

    assert len(loop_losses) == 1
    assert recompute_backward_values == [True]
    torch.testing.assert_close(loss, loop_losses[0])
    torch.testing.assert_close(losses["ntp_loss"], loop_losses[0])


def test_ut_detach_requires_everypass_ntp():
    with pytest.raises(ValueError, match="ut_detach requires ut_everypass_ntp"):
        GPTConfig(ut_everypass_ntp=False, ut_detach=True)


@pytest.mark.parametrize("ut_detach", [False, True])
@pytest.mark.parametrize("activation_checkpointing", [False, True])
def test_ut_detach_controls_cross_pass_gradient_dependency(
    monkeypatch, ut_detach, activation_checkpointing
):
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=1,
        n_exp=1,
        n_embd=32,
        n_head=4,
        total_ut_steps=2,
        ut_everypass_ntp=True,
        ut_detach=ut_detach,
        activation_checkpointing=activation_checkpointing,
        use_aux_loss=False,
        use_router_z_loss=False,
        debug=False,
    )
    model = GPT(config)
    model.init_weights()
    ids = torch.randint(0, config.vocab_size, (2, 5))
    pass_hidden_states = []
    source_activations = []
    original_chunked_cross_entropy = _chunked_cross_entropy

    def capture_hidden_state(hidden_states, *args, **kwargs):
        pass_hidden_states.append(hidden_states)
        return original_chunked_cross_entropy(hidden_states, *args, **kwargs)

    monkeypatch.setattr("nanochat.gpt._chunked_cross_entropy", capture_hidden_state)
    source_hook = model.transformer.h[-1].register_forward_hook(
        lambda _module, _args, output: source_activations.append(output)
    )
    try:
        model(ids, targets=ids)
    finally:
        source_hook.remove()

    assert len(pass_hidden_states) == config.total_ut_steps
    assert len(source_activations) == config.total_ut_steps
    cross_pass_grad = torch.autograd.grad(
        source_activations[-1].sum(),
        source_activations[0],
        allow_unused=True,
    )[0]
    assert (cross_pass_grad is None) is ut_detach


@pytest.mark.parametrize("total_ut_steps", [1, 2])
def test_gpt_activation_checkpointing_matches_losses_and_gradients_without_replay_side_effects(total_ut_steps):
    torch.manual_seed(0)
    base_config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=2,
        moe_start_layer=0,
        num_moe_layers=-1,
        n_exp=2,
        moe_top_k=2,
        n_embd=32,
        n_head=4,
        total_ut_steps=total_ut_steps,
        use_aux_loss=True,
        use_router_z_loss=True,
        debug=False,
    )
    checkpoint_config = deepcopy(base_config)
    checkpoint_config.activation_checkpointing = True
    reference_model = GPT(base_config)
    reference_model.init_weights()
    checkpoint_model = GPT(checkpoint_config)
    checkpoint_model.load_state_dict(reference_model.state_dict())
    idx = torch.randint(0, base_config.vocab_size, (2, 5))
    targets = torch.randint(0, base_config.vocab_size, (2, 5))

    def run_model(model):
        MANAGER.reset_all()
        loss, losses = model(idx, targets)
        selected_scores_rows_before_backward = MANAGER._selected_scores_size
        objective = loss + model.config.aux_loss_weight * losses["aux_loss"]
        objective.backward()
        selected_scores_rows_after_backward = MANAGER._selected_scores_size
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        return (
            loss.detach(),
            losses,
            gradients,
            selected_scores_rows_before_backward,
            selected_scores_rows_after_backward,
        )

    reference_loss, reference_losses, reference_gradients, reference_rows_before, reference_rows_after = run_model(reference_model)
    checkpoint_loss, checkpoint_losses, checkpoint_gradients, checkpoint_rows_before, checkpoint_rows_after = run_model(checkpoint_model)

    torch.testing.assert_close(checkpoint_loss, reference_loss)
    for name in ("aux_loss", "router_z_loss"):
        torch.testing.assert_close(checkpoint_losses[name], reference_losses[name])
    torch.testing.assert_close(
        checkpoint_losses["selected_scores"],
        reference_losses["selected_scores"],
    )
    assert checkpoint_gradients.keys() == reference_gradients.keys()
    for name in checkpoint_gradients:
        torch.testing.assert_close(
            checkpoint_gradients[name],
            reference_gradients[name],
            rtol=1e-5,
            atol=1e-6,
        )
    assert reference_rows_before == reference_rows_after == 0
    assert checkpoint_rows_before == checkpoint_rows_after == 0


def test_gpt_activation_checkpointing_does_not_replay_aux_free_router_counts():
    torch.manual_seed(0)
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=2,
        moe_start_layer=0,
        num_moe_layers=-1,
        n_exp=2,
        moe_top_k=1,
        n_embd=32,
        n_head=4,
        total_ut_steps=2,
        use_aux_loss=False,
        use_aux_free_load_balancing=True,
        use_router_z_loss=False,
        activation_checkpointing=True,
        debug=False,
    )
    model = GPT(config)
    model.init_weights()
    idx = torch.randint(0, config.vocab_size, (2, 5))
    targets = torch.randint(0, config.vocab_size, (2, 5))

    loss, _ = model(idx, targets)
    counters_before_backward = [
        block.mlp.router.tokens_per_expert_counter.clone()
        for block in model.transformer.h
    ]
    loss.backward()

    for block, expected_counts in zip(model.transformer.h, counters_before_backward):
        torch.testing.assert_close(
            block.mlp.router.tokens_per_expert_counter,
            expected_counts,
        )
        assert expected_counts.sum().item() == idx.numel() * config.total_ut_steps


def test_gpt_activation_offload_matches_loss_and_gradients():
    torch.manual_seed(0)
    base_config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=2,
        moe_start_layer=0,
        num_moe_layers=-1,
        n_exp=2,
        moe_top_k=2,
        n_embd=32,
        n_head=4,
        use_aux_loss=True,
        use_router_z_loss=True,
        debug=False,
    )
    offload_config = deepcopy(base_config)
    offload_config.activation_offload = True
    reference_model = GPT(base_config)
    reference_model.init_weights()
    offload_model = GPT(offload_config)
    offload_model.load_state_dict(reference_model.state_dict())
    idx = torch.randint(0, base_config.vocab_size, (2, 5))
    targets = torch.randint(0, base_config.vocab_size, (2, 5))

    def run_model(model):
        MANAGER.reset_all()
        loss, losses = model(idx, targets)
        objective = loss + model.config.aux_loss_weight * losses["aux_loss"]
        objective.backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        return loss.detach(), losses, gradients

    reference_loss, reference_losses, reference_gradients = run_model(reference_model)
    offload_loss, offload_losses, offload_gradients = run_model(offload_model)

    torch.testing.assert_close(offload_loss, reference_loss)
    for name in ("aux_loss", "router_z_loss", "selected_scores"):
        torch.testing.assert_close(offload_losses[name], reference_losses[name])
    assert offload_gradients.keys() == reference_gradients.keys()
    for name in offload_gradients:
        torch.testing.assert_close(
            offload_gradients[name],
            reference_gradients[name],
            rtol=1e-5,
            atol=1e-6,
        )


def test_gpt_rejects_checkpointing_with_activation_offload():
    with pytest.raises(ValueError, match="mutually exclusive"):
        GPTConfig(activation_checkpointing=True, activation_offload=True)


def test_activation_offload_preserves_saved_tensor_strides():
    class SaveNarrowView(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            ctx.save_for_backward(value)
            return value.sum()

        @staticmethod
        def backward(ctx, grad_output):
            (value,) = ctx.saved_tensors
            assert value.stride() == (12, 1)
            return grad_output.expand_as(value)

    base = torch.randn(8, 12, requires_grad=True)
    narrow = base[:, :4]
    config = GPTConfig(activation_offload=True)

    with _save_activations_on_cpu(base.device.type):
        loss = SaveNarrowView.apply(narrow)
    loss.backward()

    assert config.activation_offload
    torch.testing.assert_close(base.grad[:, :4], torch.ones_like(narrow))
    torch.testing.assert_close(base.grad[:, 4:], torch.zeros_like(base[:, 4:]))


def test_gpt_total_ut_steps_averages_repeated_manager_losses():
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=1,
        n_exp=1,
        n_embd=32,
        n_head=4,
        total_ut_steps=2,
        use_aux_loss=False,
        use_router_z_loss=False,
        debug=False,
    )
    model = GPT(config)
    loss_names = (
        "aux_loss",
        "router_z_loss",
        "kappa_slope_l2_loss",
    )

    for name in loss_names:
        MANAGER.reset(name)
        MANAGER.add(name, torch.tensor(2.0))
        MANAGER.add(name, torch.tensor(4.0))

        actual = model._aggregate_loop_averaged_loss(name)

        torch.testing.assert_close(actual, torch.tensor(3.0))
        assert MANAGER.aggregate(name) == 0


def test_gpt_total_ut_steps_averages_kappa_l2_from_model_forward():
    torch.manual_seed(0)
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=1,
        moe_start_layer=0,
        num_moe_layers=1,
        n_exp=2,
        moe_top_k=2,
        n_embd=32,
        n_head=4,
        total_ut_steps=2,
        use_aux_loss=False,
        use_router_z_loss=False,
        use_kappa_swiglu=True,
        debug=False,
    )
    model = GPT(config)
    model.init_weights()
    with torch.no_grad():
        kappa_bias = model.transformer.h[0].mlp.experts.kappa_bias
        kappa_bias[0].fill_(2.0)

    idx = torch.randint(0, config.vocab_size, (2, 4))
    targets = torch.randint(0, config.vocab_size, (2, 4))
    _, losses = model(idx, targets)

    torch.testing.assert_close(losses["kappa_slope_l2_loss"], torch.tensor(4.0))


@pytest.mark.parametrize("checkpointed", [False, True])
def test_gpt_kappa_losses_and_state_exclude_ema_helpers(checkpointed):
    torch.manual_seed(0)
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=1,
        moe_start_layer=0,
        num_moe_layers=1,
        n_exp=2,
        moe_top_k=2,
        n_embd=32,
        n_head=4,
        total_ut_steps=2,
        use_aux_loss=False,
        use_router_z_loss=False,
        use_kappa_swiglu=True,
        activation_checkpointing=checkpointed,
        debug=False,
    )
    model = GPT(config)
    model.init_weights()
    experts = model.transformer.h[0].mlp.experts
    with torch.no_grad():
        experts.kappa_bias.fill_(0.5)
        experts.kappa_scale.zero_()
    idx = torch.randint(0, config.vocab_size, (2, 4))
    targets = torch.randint(0, config.vocab_size, (2, 4))
    loss, losses = model(idx, targets)
    torch.testing.assert_close(losses["kappa_slope_l2_loss"], torch.tensor(0.25))
    assert not any("ema_rms" in name for name in losses)
    assert not any("ema_rms" in name for name in model.state_dict())
    for module in (model, experts):
        assert not hasattr(module, "set_kappa_bias_ema_rms_reg_step")
        assert not hasattr(module, "set_kappa_bias_ema_rms_reg_total_iterations")
    loss.backward()


def test_moe_functional_dispatch_drops_overflow_without_dynamic_shapes():
    config = GPTConfig(
        n_exp=2,
        moe_top_k=2,
        n_embd=4,
        use_qwen3_moe_mlp=True,
        debug=False,
    )
    layer = MOELayer(config, layer_idx=0)
    x_flat = torch.arange(16, dtype=torch.float32).view(4, 4).requires_grad_(True)
    flat_rank = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    flat_token_indices = torch.arange(4).repeat_interleave(2)
    flat_top_k_indices = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    flat_router_scores = torch.arange(1, 9, dtype=torch.float32, requires_grad=True)

    expert_inputs, expert_gate_scores = layer._build_expert_inputs_functional(
        x_flat,
        flat_rank,
        2,
        flat_token_indices,
        flat_top_k_indices,
        flat_router_scores,
    )

    expected_inputs = torch.stack((x_flat[:2], x_flat[:2]))
    expected_scores = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
    torch.testing.assert_close(expert_inputs, expected_inputs)
    torch.testing.assert_close(expert_gate_scores, expected_scores)

    (expert_inputs.sum() + expert_gate_scores.sum()).backward()
    torch.testing.assert_close(x_flat.grad[:2], torch.full((2, 4), 2.0))
    torch.testing.assert_close(x_flat.grad[2:], torch.zeros(2, 4))
    torch.testing.assert_close(
        flat_router_scores.grad,
        torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    )


def test_gpt_sets_kappa_slope_max_scales_for_dense_and_moe_qwen3_mlps():
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=4,
        moe_start_layer=1,
        num_moe_layers=2,
        moe_layer_stride=1,
        n_exp=2,
        n_embd=32,
        n_head=4,
        use_aux_loss=False,
        use_router_z_loss=False,
        use_kappa_swiglu=True,
        constant_kappa_bias_dense_layers=True,
        use_qwen3_moe_mlp=True,
        use_qwen3_dense_mlp=True,
        debug=False,
    )

    model = GPT(config)
    model.set_kappa_slope_max_scales(moe_kappa_slope_max_scale=2.5, dense_kappa_slope_max_scale=1.75)

    dense_layers = 0
    moe_layers = 0
    for block in model.transformer.h:
        mlp = getattr(block, 'mlp', None)
        if isinstance(mlp, Qwen3MLP):
            dense_layers += 1
            torch.testing.assert_close(mlp.kappa_slope_max_scale, torch.tensor(1.75))
            continue
        experts = getattr(mlp, 'experts', None)
        if isinstance(experts, Qwen3MLPExperts):
            moe_layers += 1
            torch.testing.assert_close(experts.kappa_slope_max_scale, torch.tensor(2.5))

    assert dense_layers == 2
    assert moe_layers == 2


def test_kappa_input_defaults_and_overrides_from_config():
    default_config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_kappa_swiglu=True,
        debug=False,
    )
    override_config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_kappa_swiglu=True,
        kappa_input="router_probs",
        debug=False,
    )

    assert default_config.kappa_input == "router_probs"
    assert override_config.kappa_input == "router_probs"
    assert default_config.kappa_input_logit_norm_exponent == 0.5
    assert override_config.kappa_input_logit_norm_exponent == 0.5


def test_kappa_input_logit_norm_exponent_defaults_and_overrides():
    default_config = GPTConfig(
        n_exp=2,
        n_embd=4,
        debug=False,
    )
    explicit_config = GPTConfig(
        n_exp=2,
        n_embd=4,
        kappa_input_logit_norm_exponent=0.5,
        debug=False,
    )

    assert default_config.kappa_input_logit_norm_exponent == 0.5
    assert explicit_config.kappa_input_logit_norm_exponent == 0.5


def test_moe_select_kappa_scores_can_normalize_top_logits():
    config = GPTConfig(
        n_exp=3,
        n_embd=4,
        moe_top_k=2,
        kappa_input="top_logits",
        kappa_input_logit_norm_exponent=0.5,
        debug=False,
    )
    moe_layer = MOELayer(config, layer_idx=0)

    x_flat = torch.tensor([
        [3.0, 4.0, 0.0, 0.0],
        [0.0, 0.0, 5.0, 12.0],
    ])
    top_k_scores = torch.tensor([
        [15.0, 40.0],
        [130.0, 26.0],
    ])
    router_probs = torch.tensor([
        [0.7, 0.3],
        [0.8, 0.2],
    ])
    top_k_indices = torch.tensor([
        [0, 1],
        [1, 2],
    ])

    with torch.no_grad():
        moe_layer.router.w_g.weight.copy_(torch.tensor([
            [3.0, 4.0, 0.0, 0.0],
            [6.0, 8.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
        ]))

    actual = moe_layer._select_kappa_scores(
        top_k_scores,
        router_probs,
        x_flat=x_flat,
        top_k_indices=top_k_indices,
    )

    router_weight_magnitudes = moe_layer.router.w_g.weight[top_k_indices].norm(dim=-1)
    smoothed_router_weight_magnitudes = torch.sqrt(
        router_weight_magnitudes.square() + moe_layer.top_logit_norm_eps
    )
    scale_compensation = torch.sqrt(
        moe_layer.router.w_g.weight.norm(dim=-1).square() + moe_layer.top_logit_norm_eps
    ).sqrt().mean()
    expected = (top_k_scores * 6.0) / (
        math.sqrt(config.n_embd)
        * smoothed_router_weight_magnitudes.sqrt()
        * scale_compensation
    )

    torch.testing.assert_close(actual, expected)


def test_moe_select_kappa_scores_smooths_tiny_router_weight_norms():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        moe_top_k=1,
        kappa_input="top_logits",
        kappa_input_logit_norm_exponent=0.5,
        debug=False,
    )
    moe_layer = MOELayer(config, layer_idx=0)

    top_k_scores = torch.tensor([[1.0]], dtype=torch.float32)
    router_probs = torch.tensor([[1.0]], dtype=torch.float32)
    top_k_indices = torch.tensor([[0]])

    with torch.no_grad():
        moe_layer.router.w_g.weight.zero_()

    actual = moe_layer._select_kappa_scores(
        top_k_scores,
        router_probs,
        x_flat=torch.zeros(1, config.n_embd),
        top_k_indices=top_k_indices,
    )

    assert torch.isfinite(actual).all()
    smoothed_router_weight_magnitudes = torch.sqrt(
        moe_layer.router.w_g.weight[top_k_indices].norm(dim=-1).square()
        + moe_layer.top_logit_norm_eps
    )
    scale_compensation = torch.sqrt(
        moe_layer.router.w_g.weight.norm(dim=-1).square() + moe_layer.top_logit_norm_eps
    ).sqrt().mean()
    expected = (top_k_scores * 6.0) / (
        math.sqrt(config.n_embd)
        * smoothed_router_weight_magnitudes.sqrt()
        * scale_compensation
    )

    torch.testing.assert_close(actual, expected)


def test_moe_select_kappa_scores_keeps_partial_norm_scale_near_unit():
    config = GPTConfig(
        n_exp=3,
        n_embd=4,
        moe_top_k=2,
        kappa_input="top_logits",
        kappa_input_logit_norm_exponent=0.5,
        debug=False,
    )
    moe_layer = MOELayer(config, layer_idx=0)

    router_probs = torch.tensor([
        [0.7, 0.3],
        [0.6, 0.4],
    ])
    top_k_indices = torch.tensor([
        [0, 1],
        [1, 2],
    ])

    with torch.no_grad():
        moe_layer.router.w_g.weight.copy_(torch.tensor([
            [2.0, 0.0, 0.0, 0.0],
            [8.0, 0.0, 0.0, 0.0],
            [18.0, 0.0, 0.0, 0.0],
        ]))

    target_gate_confidence = torch.ones_like(router_probs)
    router_weight_magnitudes = moe_layer.router.w_g.weight[top_k_indices].norm(dim=-1)
    smoothed_router_weight_magnitudes = torch.sqrt(
        router_weight_magnitudes.square() + moe_layer.top_logit_norm_eps
    )
    scale_compensation = torch.sqrt(
        moe_layer.router.w_g.weight.norm(dim=-1).square() + moe_layer.top_logit_norm_eps
    ).pow(0.5).mean()
    top_k_scores = (
        target_gate_confidence
        * math.sqrt(config.n_embd)
        * smoothed_router_weight_magnitudes.pow(config.kappa_input_logit_norm_exponent)
        * scale_compensation
        / 6.0
    )

    actual = moe_layer._select_kappa_scores(
        top_k_scores,
        router_probs,
        x_flat=torch.zeros(2, config.n_embd),
        top_k_indices=top_k_indices,
    )

    torch.testing.assert_close(actual, target_gate_confidence)

@pytest.mark.parametrize('kappa_input,independent_router', [
    ('router_probs', False), ('top_logits', False), ('constant', False),
    ('gate_proj', False), ('router_probs', True), ('top_logits', True),
    ('gate_proj', True),
])
@pytest.mark.parametrize('empty_mask', [False, True])
def test_kappa_slope_l2_uses_valid_pre_transform_values(kappa_input, independent_router, empty_mask):
    config = GPTConfig(
        n_exp=2, n_embd=4, use_kappa_swiglu=True,
        kappa_input=kappa_input, independent_kappa_router=independent_router,
        kappa_input_constant=2.0,
    )
    experts = Qwen3MLPExperts(config)
    with torch.no_grad():
        for parameter in experts.parameters():
            parameter.fill_(0.5)
    experts.snapshot_kappa_param_references()
    raw_gate = torch.randn(2, 3, 16, requires_grad=True)
    scores = torch.randn(2, 3, requires_grad=True)
    if kappa_input == 'constant':
        scores = torch.full_like(scores, config.kappa_input_constant)
    mask = torch.tensor([[True, False, False], [True, True, False]])
    if empty_mask:
        mask.zero_()
    bias = experts._materialize_kappa_bias()
    scale = None if experts._get_kappa_scale_parameter() is None else experts._materialize_kappa_scale()
    accum = MOEManager()
    experts._apply_kappa_slope_scaled_activation(
        raw_gate, bias, scores, kappa_scale=scale, loss_accum=accum,
        valid_score_mask=mask,
    )
    conditioning = scale_grad(raw_gate, 0.1) if kappa_input == 'gate_proj' else scores.unsqueeze(-1)
    if independent_router and kappa_input == 'gate_proj':
        slope = scale.unsqueeze(1) * conditioning + scores.unsqueeze(-1)
    elif independent_router:
        slope = bias.unsqueeze(1) + conditioning
    elif kappa_input == 'constant':
        slope = bias.unsqueeze(1) * conditioning
    else:
        slope = bias.unsqueeze(1) + scale.unsqueeze(1) * conditioning
    slope = slope.expand_as(raw_gate)
    expected = slope[mask].square().sum() / max(int(mask.sum()) * raw_gate.size(-1), 1)
    loss = accum.aggregate('kappa_slope_l2_loss')
    torch.testing.assert_close(loss, expected)
    loss.backward()
    if kappa_input == 'gate_proj':
        assert raw_gate.grad is None
    if independent_router:
        assert scores.grad is None
    experts.eval()
    experts._apply_kappa_slope_scaled_activation(
        raw_gate.detach(), bias.detach(), scores.detach(), loss_accum=accum,
    )
    torch.testing.assert_close(accum.aggregate('kappa_slope_l2_loss'), loss)


@pytest.mark.parametrize('constant', [0.0, 0.5, 2.0])
def test_dense_kappa_slope_l2_penalizes_pre_transform_value(constant):
    mlp = Qwen3MLP(GPTConfig(
        n_embd=4, use_kappa_swiglu=True, constant_kappa_bias_dense_layers=True,
        kappa_input_constant=constant,
    ))
    with torch.no_grad():
        mlp.kappa_bias.fill_(0.5)
    accum = MOEManager()
    mlp(torch.randn(2, 3, 4), loss_accum=accum)
    loss = accum.aggregate('kappa_slope_l2_loss')
    torch.testing.assert_close(loss, torch.tensor((0.5 * constant) ** 2))
    loss.backward()
    assert mlp.kappa_bias.grad is not None


def test_kappa_slope_l2_losses_are_reported_from_model_forward():
    torch.manual_seed(0)
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=3,
        moe_start_layer=1,
        num_moe_layers=1,
        moe_layer_stride=1,
        n_exp=2,
        n_embd=32,
        n_head=4,
        use_aux_loss=False,
        use_router_z_loss=False,
        use_kappa_swiglu=True,
        debug=False,
    )

    model = GPT(config)
    model.init_weights()

    with torch.no_grad():
        kappa_bias = model.transformer.h[1].mlp.experts.kappa_bias
        kappa_bias[0, 0].fill_(2.0)
        kappa_bias[0, 1].fill_(-2.0)

    idx = torch.randint(0, config.vocab_size, (2, 4))
    targets = torch.randint(0, config.vocab_size, (2, 4))

    _, losses = model(idx, targets)

    assert torch.isfinite(losses['kappa_slope_l2_loss'])
    torch.testing.assert_close(losses['kappa_slope_l2_loss'], torch.tensor(4.0))


@pytest.mark.parametrize("module_class", [Qwen3MLP, Qwen3MLPExperts])
@pytest.mark.parametrize("value", [0.0, 0.25, 2.0])
def test_kappa_regularization_is_mean_squared_slope(module_class, value):
    module = module_class(GPTConfig(
        n_exp=2, n_embd=4, use_kappa_swiglu=True,
        constant_kappa_bias_dense_layers=True,
    ))
    shape = (16,) if module_class is Qwen3MLP else (2, 16)
    parameter = torch.full(shape, value, requires_grad=True)
    accum = MOEManager()
    assert "kappa_bias_ema_rms_reg_loss" not in accum._values
    assert "kappa_scale_ema_rms_reg_loss" not in accum._values
    _accumulate_kappa_slope_l2_loss(parameter, loss_accum=accum)
    loss = accum.aggregate("kappa_slope_l2_loss")
    torch.testing.assert_close(loss, torch.tensor(value ** 2))
    loss.backward()
    torch.testing.assert_close(parameter.grad, torch.full(shape, 2 * value / parameter.numel()))
    assert not hasattr(module, "update_kappa_ema_rms_targets")
    assert not hasattr(module, "kappa_bias_ema_rms_reg_keeper")
    assert not hasattr(module, "kappa_scale_ema_rms_reg_keeper")


def test_kappa_slope_scale_stats_are_logged_and_detached_in_slope_scaler_mode():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_kappa_swiglu=True,
        debug=False,
    )
    experts = Qwen3MLPExperts(config)

    with torch.no_grad():
        experts.kappa_bias.fill_(1.0)

    MANAGER.reset("kappa_slope_scale_abs_mean")

    selected_gate_scores = torch.tensor([
        [1.0, 0.5],
        [0.0, 0.0],
    ], requires_grad=True)
    expected_scale_1 = math.exp(math.log(4.0) * math.tanh(-2.0))
    expected_scale_2 = math.exp(math.log(4.0) * math.tanh(-1.0))
    slope_scales = torch.tensor([
        [[expected_scale_1] * experts.intermediate_size, [expected_scale_2] * experts.intermediate_size],
        [[1.0] * experts.intermediate_size, [1.0] * experts.intermediate_size],
    ], dtype=torch.bfloat16)
    old_collect = MANAGER.collect_load_balancing_stats
    MANAGER.collect_load_balancing_stats = True
    try:
        experts._update_kappa_slope_scale_stats(slope_scales, selected_gate_scores)
    finally:
        MANAGER.collect_load_balancing_stats = old_collect

    shift_abs_mean = MANAGER.aggregate("kappa_slope_scale_abs_mean")
    assert "kappa_slope_scale_abs_mean_normalized" not in MANAGER.tensor_var_names
    assert "kappa_slope_scale_abs_mean_normalized" not in MANAGER._values

    expected_mean = slope_scales[0].float().mean().reshape(1)

    MANAGER.reset("kappa_slope_scale_abs_mean")

    torch.testing.assert_close(shift_abs_mean, expected_mean)
    assert not shift_abs_mean.requires_grad


def test_gate_stats_and_gate_bias_stats_do_not_update_when_collection_disabled():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_kappa_swiglu=True,
        debug=False,
    )
    experts = Qwen3MLPExperts(config)

    with torch.no_grad():
        experts.kappa_bias.fill_(1.0)

    MANAGER.reset("kappa_slope_scale_abs_mean")

    old_collect = MANAGER.collect_load_balancing_stats
    MANAGER.collect_load_balancing_stats = False
    try:
        experts.last_gate_stats = {"mean_abs_gate": torch.tensor(1.0)}
        experts._update_kappa_slope_scale_stats(
            torch.ones(2, 1, 4),
            torch.tensor([[1.0], [0.0]]),
        )
        experts._update_gate_stats(torch.ones(2, 1, 4))
    finally:
        MANAGER.collect_load_balancing_stats = old_collect

    assert MANAGER.aggregate("kappa_slope_scale_abs_mean") is None
    assert experts.last_gate_stats is None

    MANAGER.reset("kappa_slope_scale_abs_mean")


def test_gpt_forward_reports_kappa_slope_scale_abs_mean_metric():
    torch.manual_seed(0)
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=3,
        moe_start_layer=1,
        num_moe_layers=1,
        moe_layer_stride=1,
        n_exp=2,
        n_embd=32,
        n_head=4,
        use_aux_loss=False,
        use_router_z_loss=False,
        use_kappa_swiglu=True,
        debug=False,
    )
    model = GPT(config)
    model.init_weights()

    with torch.no_grad():
        model.transformer.h[1].mlp.experts.kappa_bias.fill_(2.0)

    idx = torch.randint(0, config.vocab_size, (2, 4))
    targets = torch.randint(0, config.vocab_size, (2, 4))

    old_collect = MANAGER.collect_load_balancing_stats
    MANAGER.collect_load_balancing_stats = True
    try:
        _, losses = model(idx, targets)
    finally:
        MANAGER.collect_load_balancing_stats = old_collect

    assert 'kappa_slope_scale_abs_mean' in losses
    assert not any('abs_mean_normalized' in name for name in losses)
    assert 'kappa_slope_scale_abs_mean_1' in losses
    assert torch.isfinite(losses['kappa_slope_scale_abs_mean'])
    assert losses['kappa_slope_scale_abs_mean'].item() >= 0.0
    torch.testing.assert_close(
        losses['kappa_slope_scale_abs_mean'],
        torch.tensor([losses['kappa_slope_scale_abs_mean_1']]),
    )


def test_gpt_forward_reports_kappa_slope_scale_abs_mean_metric_in_slope_scaler_mode():
    torch.manual_seed(0)
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=3,
        moe_start_layer=1,
        num_moe_layers=1,
        moe_layer_stride=1,
        n_exp=2,
        n_embd=32,
        n_head=4,
        use_aux_loss=False,
        use_router_z_loss=False,
        use_kappa_swiglu=True,
        debug=False,
    )
    model = GPT(config)
    model.init_weights()

    with torch.no_grad():
        model.transformer.h[1].mlp.experts.kappa_bias.fill_(2.0)

    idx = torch.randint(0, config.vocab_size, (2, 4))
    targets = torch.randint(0, config.vocab_size, (2, 4))

    old_collect = MANAGER.collect_load_balancing_stats
    MANAGER.collect_load_balancing_stats = True
    try:
        _, losses = model(idx, targets)
    finally:
        MANAGER.collect_load_balancing_stats = old_collect

    assert 'kappa_slope_scale_abs_mean' in losses
    assert not any('abs_mean_normalized' in name for name in losses)
    assert 'kappa_slope_scale_abs_mean_1' in losses
    assert torch.isfinite(losses['kappa_slope_scale_abs_mean'])
    assert losses['kappa_slope_scale_abs_mean'].item() >= 0.0
    torch.testing.assert_close(
        losses['kappa_slope_scale_abs_mean'],
        torch.tensor([losses['kappa_slope_scale_abs_mean_1']]),
    )


    assert losses['kappa_slope_scale_abs_top5p_mean'].numel() == 1
    assert losses['kappa_slope_scale_abs_bottom5p_mean'].numel() == 1
    assert 'kappa_slope_scale_abs_top5p_mean_1' in losses
    assert 'kappa_slope_scale_abs_bottom5p_mean_1' in losses


def test_kappa_param_references_are_not_auto_refreshed_without_config_opt_in():
    torch.manual_seed(0)
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=3,
        moe_start_layer=1,
        num_moe_layers=1,
        moe_layer_stride=1,
        n_exp=2,
        n_embd=32,
        n_head=4,
        use_aux_loss=False,
        use_router_z_loss=False,
        use_kappa_swiglu=True,
        debug=False,
    )
    model = GPT(config)
    model.init_weights()

    assert model.transformer.h[1].mlp.experts.initial_kappa_bias is None
    assert model.transformer.h[1].mlp.experts.initial_kappa_scale is None

    model.refresh_kappa_param_references()

    assert model.transformer.h[1].mlp.experts.initial_kappa_bias is not None
    assert model.transformer.h[1].mlp.experts.initial_kappa_scale is not None


def test_kappa_slope_scale_stats_default_to_zero_when_bias_disabled():
    torch.manual_seed(0)
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=3,
        moe_start_layer=1,
        num_moe_layers=1,
        moe_layer_stride=1,
        n_exp=2,
        n_embd=32,
        n_head=4,
        use_aux_loss=False,
        use_router_z_loss=False,
        use_kappa_swiglu=False,
        debug=False,
    )
    model = GPT(config)
    model.init_weights()

    idx = torch.randint(0, config.vocab_size, (2, 4))
    targets = torch.randint(0, config.vocab_size, (2, 4))

    old_collect = MANAGER.collect_load_balancing_stats
    MANAGER.collect_load_balancing_stats = True
    try:
        _, losses = model(idx, targets)
    finally:
        MANAGER.collect_load_balancing_stats = old_collect

    assert losses['kappa_slope_scale_abs_top5p_mean'].shape == torch.Size([])
    assert losses['kappa_slope_scale_abs_bottom5p_mean'].shape == torch.Size([])
    assert losses['kappa_slope_scale_abs_top5p_mean'].item() == 0.0
    assert losses['kappa_slope_scale_abs_bottom5p_mean'].item() == 0.0
    assert torch.isfinite(losses['kappa_slope_scale_abs_top5p_mean'])
    assert torch.isfinite(losses['kappa_slope_scale_abs_bottom5p_mean'])


def test_kappa_param_references_can_auto_refresh_when_config_enabled():
    torch.manual_seed(0)
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=3,
        moe_start_layer=1,
        num_moe_layers=1,
        moe_layer_stride=1,
        n_exp=2,
        n_embd=32,
        n_head=4,
        use_aux_loss=False,
        use_router_z_loss=False,
        use_kappa_swiglu=True,
        refresh_kappa_param_references=True,
        debug=False,
    )
    model = GPT(config)
    model.init_weights()

    assert model.transformer.h[1].mlp.experts.initial_kappa_bias is not None
    assert model.transformer.h[1].mlp.experts.initial_kappa_scale is not None


def test_dense_gate_projection_has_expected_shape():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        debug=False,
    )

    experts = Qwen3MLPExperts(config)

    assert hasattr(experts, 'gate_proj')
    assert experts.gate_proj.ndim == 3
    assert experts.gate_proj.shape == (config.n_exp, config.n_embd, 4 * config.n_embd)
    assert experts.kappa_bias is None


def test_kappa_bias_has_expected_shape_when_enabled():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_kappa_swiglu=True,
        debug=False,
    )

    experts = Qwen3MLPExperts(config)

    assert experts.kappa_bias is not None
    assert experts.kappa_bias.ndim == 3
    assert experts.kappa_bias.shape == (
        1,
        config.n_exp,
        4 * config.n_embd,
    )


@pytest.mark.parametrize(
    ("granularity", "parameter_shape", "expected_materialized_shape"),
    [
        ("per-gate", (2, 16), (2, 16)),
        ("per-expert", (2,), (2, 16)),
        ("per-layer", (1,), (2, 16)),
    ],
)
def test_kappa_bias_materializes_expected_shape_for_local_granularities(
    granularity,
    parameter_shape,
    expected_materialized_shape,
):
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_kappa_swiglu=True,
        global_kappa_param_granularity=granularity,
        debug=False,
    )

    experts = Qwen3MLPExperts(config)

    assert experts.kappa_bias is not None
    assert tuple(experts.kappa_bias.shape) == (1, *parameter_shape)
    assert tuple(experts._materialize_kappa_bias().shape) == expected_materialized_shape


def test_kappa_bias_materialization_broadcasts_per_expert_values():
    config = GPTConfig(
        n_exp=3,
        n_embd=4,
        use_kappa_swiglu=True,
        global_kappa_param_granularity="per-expert",
        debug=False,
    )

    experts = Qwen3MLPExperts(config)
    with torch.no_grad():
        experts.kappa_bias[0].copy_(torch.tensor([1.0, 2.0, 3.0]))

    materialized = experts._materialize_kappa_bias()

    torch.testing.assert_close(materialized[0], torch.ones(16))
    torch.testing.assert_close(materialized[1], torch.full((16,), 2.0))
    torch.testing.assert_close(materialized[2], torch.full((16,), 3.0))


def test_kappa_parameters_accumulate_gradients_in_one_shared_ut_slot():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        total_ut_steps=2,
        use_kappa_swiglu=True,
        global_kappa_param_granularity="per-expert",
        debug=False,
    )
    experts = Qwen3MLPExperts(config)
    with torch.no_grad():
        experts.kappa_bias.copy_(torch.tensor([[1.0, 2.0]]))
        experts.kappa_scale.copy_(torch.tensor([[5.0, 6.0]]))

    for current_ut in range(config.total_ut_steps):
        materialized_bias = experts._materialize_kappa_bias()
        materialized_scale = experts._materialize_kappa_scale()
        torch.testing.assert_close(materialized_bias[0], torch.full((16,), 1.0))
        torch.testing.assert_close(materialized_bias[1], torch.full((16,), 2.0))
        torch.testing.assert_close(materialized_scale[0], torch.full((16,), 5.0))
        torch.testing.assert_close(materialized_scale[1], torch.full((16,), 6.0))
        (materialized_bias.sum() + materialized_scale.sum()).backward()

    torch.testing.assert_close(experts.kappa_bias.grad, torch.full((1, 2), 32.0))
    torch.testing.assert_close(experts.kappa_scale.grad, torch.full((1, 2), 32.0))


def test_dense_kappa_bias_has_one_shared_ut_slot():
    config = GPTConfig(
        n_embd=4,
        total_ut_steps=2,
        use_kappa_swiglu=True,
        constant_kappa_bias_dense_layers=True,
        global_kappa_param_granularity="per-layer",
        debug=False,
    )
    mlp = Qwen3MLP(config)
    with torch.no_grad():
        mlp.kappa_bias.fill_(2.0)

    assert mlp.kappa_bias.shape == (1, 1)
    materialized = mlp._materialize_kappa_bias()
    torch.testing.assert_close(materialized, torch.full((16,), 2.0))

    materialized.sum().backward()
    torch.testing.assert_close(mlp.kappa_bias.grad, torch.full((1, 1), 16.0))


def test_kappa_bias_global_granularity_shares_one_parameter_across_layers():
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=4,
        moe_start_layer=1,
        num_moe_layers=2,
        moe_layer_stride=1,
        n_exp=2,
        n_embd=8,
        n_head=2,
        total_ut_steps=3,
        use_aux_loss=False,
        use_router_z_loss=False,
        use_kappa_swiglu=True,
        global_kappa_param_granularity="global",
        debug=False,
    )

    model = GPT(config)
    moe_experts = [
        block.mlp.experts
        for block in model.transformer.h
        if hasattr(block.mlp, 'experts') and isinstance(block.mlp.experts, Qwen3MLPExperts)
    ]

    assert model.global_kappa_bias is not None
    assert tuple(model.global_kappa_bias.shape) == (1, 1)
    assert tuple(model.global_kappa_scale.shape) == (1, 1)
    assert all(experts.kappa_bias is None for experts in moe_experts)
    assert all(experts._get_kappa_bias_parameter() is model.global_kappa_bias for experts in moe_experts)
    assert all(tuple(experts._materialize_kappa_bias().shape) == (config.n_exp, 4 * config.n_embd) for experts in moe_experts)


def test_kappa_bias_respects_start_layer_cutoff():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_kappa_swiglu=True,
        kappa_start_layer=3,
        debug=False,
    )

    early_experts = Qwen3MLPExperts(config, layer_idx=2)
    late_experts = Qwen3MLPExperts(config, layer_idx=3)

    assert early_experts.kappa_bias is None
    assert late_experts.kappa_bias is not None


def test_qwen3_experts_use_dense_gate_projection_only():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        debug=False,
    )

    experts = Qwen3MLPExperts(config)

    assert experts.gate_proj.shape == (config.n_exp, config.n_embd, 4 * config.n_embd)
    assert not hasattr(experts, 'gate_proj_a')
    assert not hasattr(experts, 'gate_proj_b')


def test_all_moe_layers_use_dense_gate_projection():
    config = GPTConfig(
        n_layer=6,
        moe_start_layer=2,
        moe_layer_stride=1,
        n_exp=2,
        n_embd=8,
        n_head=2,
        debug=False,
    )

    model = GPT(config)
    observed_gate_ndims = [
        layer.mlp.experts.gate_proj.ndim
        for layer in model.transformer.h
        if hasattr(layer.mlp, 'experts') and isinstance(layer.mlp.experts, Qwen3MLPExperts)
    ]

    assert observed_gate_ndims == [3, 3, 3, 3]


def test_qwen3_experts_do_not_expose_low_rank_gate_factors():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        debug=False,
    )

    experts = Qwen3MLPExperts(config)

    assert hasattr(experts, 'gate_proj')
    assert experts.gate_proj.ndim == 3
    assert not hasattr(experts, 'gate_proj_a')
    assert not hasattr(experts, 'gate_proj_b')