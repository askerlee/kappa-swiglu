import math
import pytest
import torch
import torch.nn.functional as F
from copy import deepcopy

from nanochat.configuration_nanomoe_gpt import GPTConfig
from nanochat.gpt import GPT, MANAGER, GateProjBiasEmaTargetKeeper, Qwen3MLP, Qwen3MLPExperts, Router, scale_grad


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


def test_gate_projection_bias_can_rescale_gate_slope_from_router_probs():
    torch.manual_seed(0)
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_gate_proj_bias=True,
        debug=False,
    )
    experts = Qwen3MLPExperts(config)

    x = torch.randn(config.n_exp, 5, config.n_embd)
    router_probs = torch.rand(config.n_exp, 5)

    with torch.no_grad():
        experts.gate_proj.copy_(torch.randn_like(experts.gate_proj))
        experts.gate_proj_bias.copy_(torch.randn_like(experts.gate_proj_bias))
        experts.c_fc.copy_(torch.randn_like(experts.c_fc))
        experts.c_proj.copy_(torch.randn_like(experts.c_proj))

        raw_gate_out = torch.bmm(x, experts.gate_proj)
        log_kappa = 4 * experts.gate_proj_bias.unsqueeze(1) * router_probs.unsqueeze(-1)
        inv_kappa = torch.exp(torch.log(torch.tensor(4.0)) * torch.tanh(-log_kappa / 2.0))
        expected_gate_out_acts = raw_gate_out * torch.sigmoid(raw_gate_out * inv_kappa)

        fc_out = torch.bmm(x, experts.c_fc)
        expected = torch.bmm(expected_gate_out_acts * fc_out, experts.c_proj)

    actual = experts(x, selected_router_scores=router_probs)
    torch.testing.assert_close(actual, expected)

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


def test_dynamic_gate_projection_bias_backprops_into_selected_router_scores():
    torch.manual_seed(0)
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_gate_proj_bias=True,
        debug=False,
    )
    experts = Qwen3MLPExperts(config)

    x = torch.randn(config.n_exp, 5, config.n_embd, requires_grad=True)
    selected_router_scores = torch.randn(config.n_exp, 5, requires_grad=True)
    out = experts(x, selected_router_scores=selected_router_scores).sum()
    out.backward()

    assert selected_router_scores.grad is not None


def test_dynamic_gate_projection_bias_scales_selected_router_score_gradients():
    torch.manual_seed(0)
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_gate_proj_bias=True,
        debug=False,
    )
    experts = Qwen3MLPExperts(config)

    with torch.no_grad():
        experts.gate_proj.fill_(0.1)
        experts.c_fc.fill_(0.2)
        experts.c_proj.fill_(0.3)
        experts.gate_proj_bias.fill_(0.05)

    x = torch.randn(config.n_exp, 5, config.n_embd)
    selected_router_scores = torch.randn(config.n_exp, 5)

    experts.router_confidence_gate_bias_grad_scale.fill_(1.0)
    selected_router_scores_full = selected_router_scores.clone().requires_grad_(True)
    experts(x, selected_router_scores=selected_router_scores_full).sum().backward()
    grad_full = selected_router_scores_full.grad.clone()

    experts.zero_grad(set_to_none=True)
    experts.router_confidence_gate_bias_grad_scale.fill_(0.25)
    selected_router_scores_scaled = selected_router_scores.clone().requires_grad_(True)
    experts(x, selected_router_scores=selected_router_scores_scaled).sum().backward()
    grad_scaled = selected_router_scores_scaled.grad.clone()

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


def test_dense_qwen3_gate_projection_has_no_bias_parameter():
    config = GPTConfig(
        n_exp=1,
        n_embd=4,
        use_gate_proj_bias=True,
        debug=False,
    )

    mlp = Qwen3MLP(config)

    assert not hasattr(mlp, 'gate_proj_bias')


def test_config_allows_constant_dense_gate_proj_bias_with_router_probs_for_moe_layers():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_gate_proj_bias=True,
        gate_proj_bias_input="router_probs",
        gate_proj_bias_input_constant=0.5,
        constant_gate_proj_bias_dense_layers=True,
        debug=False,
    )

    assert config.gate_proj_bias_input == "router_probs"
    assert config.gate_proj_bias_input_constant == pytest.approx(0.5)
    assert config.constant_gate_proj_bias_dense_layers is True


def test_dense_qwen3_mlp_enables_constant_gate_proj_bias_when_requested():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_gate_proj_bias=True,
        gate_proj_bias_input="router_probs",
        gate_proj_bias_input_constant=0.5,
        constant_gate_proj_bias_dense_layers=True,
        debug=False,
    )

    mlp = Qwen3MLP(config, layer_idx=0)
    experts = Qwen3MLPExperts(config, layer_idx=0)

    assert mlp.use_gate_proj_bias is True
    assert mlp.gate_proj_bias is not None
    assert experts.use_gate_proj_bias is True
    assert experts.use_gate_proj_bias_scale is True


def test_dense_qwen3_mlp_uses_placeholder_bias_before_start_layer():
    torch.manual_seed(0)
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_gate_proj_bias=True,
        gate_proj_bias_input="router_probs",
        gate_proj_bias_input_constant=0.5,
        constant_gate_proj_bias_dense_layers=True,
        gate_proj_bias_start_layer=2,
        debug=False,
    )

    mlp = Qwen3MLP(config, layer_idx=0)
    x = torch.randn(3, 5, config.n_embd)

    assert mlp.use_gate_proj_bias is True
    assert mlp.has_active_gate_proj_bias is False
    assert not hasattr(mlp, 'gate_proj_bias')

    with torch.no_grad():
        mlp.gate_proj.weight.copy_(torch.randn_like(mlp.gate_proj.weight))
        mlp.c_fc.weight.copy_(torch.randn_like(mlp.c_fc.weight))
        mlp.c_proj.weight.copy_(torch.randn_like(mlp.c_proj.weight))
        raw_gate_out = mlp.gate_proj(x)
        expected = mlp.c_proj(mlp.act_fn(raw_gate_out) * mlp.c_fc(x))

    actual = mlp(x)
    torch.testing.assert_close(actual, expected)


def test_gate_proj_bias_lr_scale_defaults_and_overrides_from_config():
    default_config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_gate_proj_bias=True,
        debug=False,
    )
    override_config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_gate_proj_bias=True,
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
        use_gate_proj_bias=True,
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


def test_gate_proj_bias_input_defaults_and_overrides_from_config():
    default_config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_gate_proj_bias=True,
        debug=False,
    )
    override_config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_gate_proj_bias=True,
        gate_proj_bias_input="router_probs",
        debug=False,
    )

    assert default_config.gate_proj_bias_input == "router_probs"
    assert override_config.gate_proj_bias_input == "router_probs"

def test_gate_proj_bias_l2_losses_split_above_and_below_zero():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_gate_proj_bias=True,
        debug=False,
    )
    experts = Qwen3MLPExperts(config)

    MANAGER.reset("gate_proj_bias_l2_loss_above_0")
    MANAGER.reset("gate_proj_bias_l2_loss_below_0")

    gate_proj_bias = torch.tensor([
        [-0.5, 0.5],
        [-0.25, 0.25],
    ])
    selected_router_scores = torch.ones(2, 2)
    slope_scales = torch.tensor([
        [[0.5, 1.0], [1.5, 0.75]],
        [[2.0, 1.0], [1.0, 0.25]],
    ])
    del slope_scales, selected_router_scores
    experts._accumulate_gate_proj_bias_l2_losses(gate_proj_bias)

    above_0 = MANAGER.aggregate("gate_proj_bias_l2_loss_above_0")
    below_0 = MANAGER.aggregate("gate_proj_bias_l2_loss_below_0")

    MANAGER.reset("gate_proj_bias_l2_loss_above_0")
    MANAGER.reset("gate_proj_bias_l2_loss_below_0")

    torch.testing.assert_close(above_0, torch.tensor(0.3125 / 4.0))
    torch.testing.assert_close(below_0, torch.tensor(0.3125 / 4.0))


def test_gate_proj_bias_l2_losses_are_reported_from_gate_proj_biases():
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
        use_gate_proj_bias=True,
        debug=False,
    )

    model = GPT(config)
    model.init_weights()

    with torch.no_grad():
        gate_proj_bias = model.transformer.h[1].mlp.experts.gate_proj_bias
        gate_proj_bias[0].fill_(2.0)
        gate_proj_bias[1].fill_(-2.0)

    idx = torch.randint(0, config.vocab_size, (2, 4))
    targets = torch.randint(0, config.vocab_size, (2, 4))

    _, losses = model(idx, targets)

    assert torch.isfinite(losses['gate_proj_bias_l2_loss'])
    assert torch.isfinite(losses['gate_proj_bias_l2_loss_above_0'])
    assert torch.isfinite(losses['gate_proj_bias_l2_loss_below_0'])
    assert losses['gate_proj_bias_l2_loss_above_0'].item() > 0.0
    assert losses['gate_proj_bias_l2_loss_below_0'].item() > 0.0
    torch.testing.assert_close(
        losses['gate_proj_bias_l2_loss'],
        losses['gate_proj_bias_l2_loss_above_0'] + losses['gate_proj_bias_l2_loss_below_0'],
    )


def test_gate_proj_bias_ema_rms_reg_loss_is_added_on_top_of_l2_loss():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_gate_proj_bias=True,
        gate_proj_bias_ema_rms_reg=True,
        gate_proj_bias_l2_ema_beta=0.99,
        gate_proj_bias_l2_ema_anchor_start=0.0,
        gate_proj_bias_l2_ema_anchor_end=0.0,
        gate_proj_bias_l2_ema_floor_frac=0.8,
        debug=False,
    )
    experts = Qwen3MLPExperts(config)

    MANAGER.reset("gate_proj_bias_l2_loss")
    MANAGER.reset("gate_proj_bias_ema_rms_reg_loss")
    experts.set_gate_proj_bias_ema_rms_reg_total_iterations(1)
    experts.set_gate_proj_bias_ema_rms_reg_step(0)
    experts._accumulate_gate_proj_bias_l2_losses(torch.full((2, 16), 2.0))
    first_l2_loss = MANAGER.aggregate("gate_proj_bias_l2_loss")
    first_ema_rms_reg_loss = MANAGER.aggregate("gate_proj_bias_ema_rms_reg_loss")
    MANAGER.reset("gate_proj_bias_l2_loss")
    MANAGER.reset("gate_proj_bias_ema_rms_reg_loss")

    experts.set_gate_proj_bias_ema_rms_reg_step(1)
    experts._accumulate_gate_proj_bias_l2_losses(torch.full((2, 16), 0.5))
    second_l2_loss = MANAGER.aggregate("gate_proj_bias_l2_loss")
    second_ema_rms_reg_loss = MANAGER.aggregate("gate_proj_bias_ema_rms_reg_loss")
    MANAGER.reset("gate_proj_bias_l2_loss")
    MANAGER.reset("gate_proj_bias_ema_rms_reg_loss")

    torch.testing.assert_close(first_l2_loss, torch.tensor(4.0))
    torch.testing.assert_close(first_ema_rms_reg_loss, torch.tensor(0.0))
    torch.testing.assert_close(second_l2_loss, torch.tensor(0.25))
    torch.testing.assert_close(second_ema_rms_reg_loss, torch.tensor((1.6 - 0.5) ** 2))


def test_gate_proj_bias_ema_target_keeper_raises_on_nonfinite_input():
    keeper = GateProjBiasEmaTargetKeeper(
        beta=0.99,
        anchor_start=0.0,
        anchor_end=1.0,
        floor_frac=0.8,
    )

    with pytest.raises(RuntimeError, match="non-finite value"):
        keeper.update(torch.tensor([float('nan')]), step=0)


def test_gate_proj_bias_ema_target_keeper_raises_on_nonfinite_target_before_loss():
    keeper = GateProjBiasEmaTargetKeeper(
        beta=0.99,
        anchor_start=0.0,
        anchor_end=1.0,
        floor_frac=0.8,
    )
    keeper.target_rms.fill_(float('nan'))
    keeper.target_ready.fill_(True)

    with pytest.raises(RuntimeError, match="non-finite floor"):
        keeper.loss(torch.tensor([1.0]))


def test_gate_proj_bias_ema_target_error_includes_module_source():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_gate_proj_bias=True,
        gate_proj_bias_ema_rms_reg=True,
        debug=False,
    )
    experts = Qwen3MLPExperts(config, layer_idx=3)

    with pytest.raises(RuntimeError, match=r"Qwen3MLPExperts\(layer=3, granularity=per-gate\)\.gate_proj_bias"):
        experts._accumulate_gate_proj_bias_l2_losses(torch.full((2, 16), float('nan')))


def test_gate_proj_bias_ema_target_loss_has_finite_gradient_at_zero():
    keeper = GateProjBiasEmaTargetKeeper(
        beta=0.99,
        anchor_start=0.0,
        anchor_end=1.0,
        floor_frac=0.8,
    )
    keeper.target_rms.fill_(2.0)
    keeper.target_ready.fill_(True)

    value = torch.zeros(4, requires_grad=True)
    loss = keeper.loss(value)
    loss.backward()

    assert torch.isfinite(loss)
    assert value.grad is not None
    assert torch.isfinite(value.grad).all()


def test_gate_proj_bias_scale_ema_rms_reg_loss_is_added_on_top_of_l2_loss():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_gate_proj_bias=True,
        gate_proj_bias_input="router_probs",
        gate_proj_bias_ema_rms_reg=True,
        gate_proj_bias_l2_ema_beta=0.99,
        gate_proj_bias_l2_ema_anchor_start=0.0,
        gate_proj_bias_l2_ema_anchor_end=0.0,
        gate_proj_bias_l2_ema_floor_frac=0.8,
        debug=False,
    )
    experts = Qwen3MLPExperts(config)

    MANAGER.reset("gate_proj_bias_scale_l2_loss")
    MANAGER.reset("gate_proj_bias_scale_ema_rms_reg_loss")
    experts.set_gate_proj_bias_ema_rms_reg_total_iterations(1)
    experts.set_gate_proj_bias_ema_rms_reg_step(0)
    experts._accumulate_gate_proj_bias_scale_l2_losses(torch.full((2, 16), 2.0))
    first_l2_loss = MANAGER.aggregate("gate_proj_bias_scale_l2_loss")
    first_ema_rms_reg_loss = MANAGER.aggregate("gate_proj_bias_scale_ema_rms_reg_loss")
    MANAGER.reset("gate_proj_bias_scale_l2_loss")
    MANAGER.reset("gate_proj_bias_scale_ema_rms_reg_loss")

    experts.set_gate_proj_bias_ema_rms_reg_step(1)
    experts._accumulate_gate_proj_bias_scale_l2_losses(torch.full((2, 16), 0.25))
    second_l2_loss = MANAGER.aggregate("gate_proj_bias_scale_l2_loss")
    second_ema_rms_reg_loss = MANAGER.aggregate("gate_proj_bias_scale_ema_rms_reg_loss")
    MANAGER.reset("gate_proj_bias_scale_l2_loss")
    MANAGER.reset("gate_proj_bias_scale_ema_rms_reg_loss")

    torch.testing.assert_close(first_l2_loss, torch.tensor(4.0))
    torch.testing.assert_close(first_ema_rms_reg_loss, torch.tensor(0.0))
    torch.testing.assert_close(second_l2_loss, torch.tensor(0.0625))
    torch.testing.assert_close(second_ema_rms_reg_loss, torch.tensor((1.6 - 0.25) ** 2))


def test_dense_gate_proj_bias_scale_ema_rms_reg_loss_is_added_on_top_of_l2_loss():
    config = GPTConfig(
        n_embd=4,
        use_gate_proj_bias=True,
        constant_gate_proj_bias_dense_layers=True,
        gate_proj_bias_input="constant",
        gate_proj_bias_ema_rms_reg=True,
        gate_proj_bias_l2_ema_beta=0.99,
        gate_proj_bias_l2_ema_anchor_start=0.0,
        gate_proj_bias_l2_ema_anchor_end=0.0,
        gate_proj_bias_l2_ema_floor_frac=0.8,
        debug=False,
    )
    mlp = Qwen3MLP(config)

    MANAGER.reset("gate_proj_bias_scale_l2_loss")
    MANAGER.reset("gate_proj_bias_scale_ema_rms_reg_loss")
    mlp.set_gate_proj_bias_ema_rms_reg_total_iterations(1)
    mlp.set_gate_proj_bias_ema_rms_reg_step(0)
    mlp._accumulate_gate_proj_bias_scale_l2_losses(torch.full((16,), 2.0))
    first_l2_loss = MANAGER.aggregate("gate_proj_bias_scale_l2_loss")
    first_ema_rms_reg_loss = MANAGER.aggregate("gate_proj_bias_scale_ema_rms_reg_loss")
    MANAGER.reset("gate_proj_bias_scale_l2_loss")
    MANAGER.reset("gate_proj_bias_scale_ema_rms_reg_loss")

    mlp.set_gate_proj_bias_ema_rms_reg_step(1)
    mlp._accumulate_gate_proj_bias_scale_l2_losses(torch.full((16,), 0.25))
    second_l2_loss = MANAGER.aggregate("gate_proj_bias_scale_l2_loss")
    second_ema_rms_reg_loss = MANAGER.aggregate("gate_proj_bias_scale_ema_rms_reg_loss")
    MANAGER.reset("gate_proj_bias_scale_l2_loss")
    MANAGER.reset("gate_proj_bias_scale_ema_rms_reg_loss")

    torch.testing.assert_close(first_l2_loss, torch.tensor(4.0))
    torch.testing.assert_close(first_ema_rms_reg_loss, torch.tensor(0.0))
    torch.testing.assert_close(second_l2_loss, torch.tensor(0.0625))
    torch.testing.assert_close(second_ema_rms_reg_loss, torch.tensor((1.6 - 0.25) ** 2))


def test_gate_proj_bias_ema_target_buffers_load_from_older_checkpoints():
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
        use_gate_proj_bias=True,
        gate_proj_bias_ema_rms_reg=True,
        debug=False,
    )
    model = GPT(config)
    state_dict = {
        name: value
        for name, value in model.state_dict().items()
        if "ema_rms_reg_keeper" not in name
    }

    load_result = model.load_state_dict(state_dict, strict=True)

    assert not load_result.missing_keys
    assert not load_result.unexpected_keys
    experts = model.transformer.h[1].mlp.experts
    assert torch.equal(experts.gate_proj_bias_ema_rms_reg_keeper.ema_rms, torch.zeros(()))
    assert torch.equal(experts.gate_proj_bias_scale_ema_rms_reg_keeper.ema_rms, torch.zeros(()))
    assert not bool(experts.gate_proj_bias_ema_rms_reg_keeper.initialized.item())
    assert not bool(experts.gate_proj_bias_scale_ema_rms_reg_keeper.initialized.item())


def test_gate_proj_bias_ema_anchor_fractions_resolve_against_total_iterations():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_gate_proj_bias=True,
        gate_proj_bias_ema_rms_reg=True,
        gate_proj_bias_l2_ema_beta=0.99,
        gate_proj_bias_l2_ema_anchor_start=0.4,
        gate_proj_bias_l2_ema_anchor_end=0.8,
        gate_proj_bias_l2_ema_floor_frac=0.8,
        debug=False,
    )
    experts = Qwen3MLPExperts(config)
    experts.set_gate_proj_bias_ema_rms_reg_total_iterations(10)

    anchor_start, anchor_end = experts.gate_proj_bias_ema_rms_reg_keeper._resolve_anchor_steps()

    assert anchor_start == 4
    assert anchor_end == 8


def test_gate_proj_bias_ema_rms_reg_is_zero_before_anchor():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_gate_proj_bias=True,
        gate_proj_bias_ema_rms_reg=True,
        gate_proj_bias_l2_ema_beta=0.99,
        gate_proj_bias_l2_ema_anchor_start=0.4,
        gate_proj_bias_l2_ema_anchor_end=0.8,
        gate_proj_bias_l2_ema_floor_frac=0.8,
        debug=False,
    )
    experts = Qwen3MLPExperts(config)
    experts.set_gate_proj_bias_ema_rms_reg_total_iterations(10)

    value = torch.full((2, 16), 2.0)
    MANAGER.reset("gate_proj_bias_l2_loss")
    MANAGER.reset("gate_proj_bias_ema_rms_reg_loss")
    experts.set_gate_proj_bias_ema_rms_reg_step(0)
    experts._accumulate_gate_proj_bias_l2_losses(value)
    l2_loss = MANAGER.aggregate("gate_proj_bias_l2_loss")
    ema_rms_reg_loss = MANAGER.aggregate("gate_proj_bias_ema_rms_reg_loss")
    MANAGER.reset("gate_proj_bias_l2_loss")
    MANAGER.reset("gate_proj_bias_ema_rms_reg_loss")

    torch.testing.assert_close(l2_loss, value.square().mean())
    torch.testing.assert_close(ema_rms_reg_loss, torch.tensor(0.0))
    assert not bool(experts.gate_proj_bias_ema_rms_reg_keeper.target_ready.item())


def test_gate_slope_scale_stats_are_logged_and_detached_in_slope_scaler_mode():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_gate_proj_bias=True,
        debug=False,
    )
    experts = Qwen3MLPExperts(config)

    with torch.no_grad():
        experts.gate_proj_bias.fill_(1.0)

    MANAGER.reset("gate_proj_bias_shift_abs_mean")
    MANAGER.reset("gate_proj_bias_shift_abs_mean_normalized")

    selected_router_scores = torch.tensor([
        [1.0, 0.5],
        [0.0, 0.0],
    ], requires_grad=True)
    slope_scales = experts._compute_gate_slope_scales(
        experts.gate_proj_bias,
        selected_router_scores,
    )
    old_collect = MANAGER.collect_load_balancing_stats
    MANAGER.collect_load_balancing_stats = True
    try:
        experts._update_gate_slope_scale_stats(slope_scales, selected_router_scores)
    finally:
        MANAGER.collect_load_balancing_stats = old_collect

    shift_abs_mean = MANAGER.aggregate("gate_proj_bias_shift_abs_mean")
    normalized_shift_abs_mean = MANAGER.aggregate("gate_proj_bias_shift_abs_mean_normalized")

    expected_scale_1 = math.exp(math.log(4.0) * math.tanh(-2.0))
    expected_scale_2 = math.exp(math.log(4.0) * math.tanh(-1.0))
    expected_mean = torch.tensor([(expected_scale_1 + expected_scale_2) / 2.0])

    MANAGER.reset("gate_proj_bias_shift_abs_mean")
    MANAGER.reset("gate_proj_bias_shift_abs_mean_normalized")

    torch.testing.assert_close(shift_abs_mean, expected_mean)
    torch.testing.assert_close(normalized_shift_abs_mean, expected_mean)


def test_gate_stats_and_gate_bias_stats_do_not_update_when_collection_disabled():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_gate_proj_bias=True,
        debug=False,
    )
    experts = Qwen3MLPExperts(config)

    with torch.no_grad():
        experts.gate_proj_bias.fill_(1.0)

    MANAGER.reset("gate_proj_bias_shift_abs_mean")
    MANAGER.reset("gate_proj_bias_shift_abs_mean_normalized")

    old_collect = MANAGER.collect_load_balancing_stats
    MANAGER.collect_load_balancing_stats = False
    try:
        experts.last_gate_stats = {"mean_abs_gate": torch.tensor(1.0)}
        experts._update_gate_slope_scale_stats(
            torch.ones(2, 1, 4),
            torch.tensor([[1.0], [0.0]]),
        )
        experts._update_gate_stats(torch.ones(2, 1, 4))
    finally:
        MANAGER.collect_load_balancing_stats = old_collect

    assert MANAGER.aggregate("gate_proj_bias_shift_abs_mean") is None
    assert MANAGER.aggregate("gate_proj_bias_shift_abs_mean_normalized") is None
    assert experts.last_gate_stats is None

    MANAGER.reset("gate_proj_bias_shift_abs_mean")
    MANAGER.reset("gate_proj_bias_shift_abs_mean_normalized")


def test_gpt_forward_reports_gate_proj_bias_shift_abs_mean_metric():
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
        use_gate_proj_bias=True,
        debug=False,
    )
    model = GPT(config)
    model.init_weights()

    with torch.no_grad():
        model.transformer.h[1].mlp.experts.gate_proj_bias.fill_(2.0)

    idx = torch.randint(0, config.vocab_size, (2, 4))
    targets = torch.randint(0, config.vocab_size, (2, 4))

    old_collect = MANAGER.collect_load_balancing_stats
    MANAGER.collect_load_balancing_stats = True
    try:
        _, losses = model(idx, targets)
    finally:
        MANAGER.collect_load_balancing_stats = old_collect

    assert 'gate_proj_bias_shift_abs_mean' in losses
    assert 'gate_proj_bias_shift_abs_mean_normalized' in losses
    assert 'gate_proj_bias_shift_abs_mean_1' in losses
    assert 'gate_proj_bias_shift_abs_mean_normalized_1' in losses
    assert torch.isfinite(losses['gate_proj_bias_shift_abs_mean'])
    assert torch.isfinite(losses['gate_proj_bias_shift_abs_mean_normalized'])
    assert losses['gate_proj_bias_shift_abs_mean'].item() >= 0.0
    assert losses['gate_proj_bias_shift_abs_mean_normalized'].item() >= 0.0
    torch.testing.assert_close(
        losses['gate_proj_bias_shift_abs_mean'],
        torch.tensor(losses['gate_proj_bias_shift_abs_mean_1']),
    )
    torch.testing.assert_close(
        losses['gate_proj_bias_shift_abs_mean_normalized'],
        torch.tensor(losses['gate_proj_bias_shift_abs_mean_normalized_1']),
    )


def test_gpt_forward_reports_gate_proj_bias_shift_abs_mean_metric_in_slope_scaler_mode():
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
        use_gate_proj_bias=True,
        debug=False,
    )
    model = GPT(config)
    model.init_weights()

    with torch.no_grad():
        model.transformer.h[1].mlp.experts.gate_proj_bias.fill_(2.0)

    idx = torch.randint(0, config.vocab_size, (2, 4))
    targets = torch.randint(0, config.vocab_size, (2, 4))

    old_collect = MANAGER.collect_load_balancing_stats
    MANAGER.collect_load_balancing_stats = True
    try:
        _, losses = model(idx, targets)
    finally:
        MANAGER.collect_load_balancing_stats = old_collect

    assert 'gate_proj_bias_shift_abs_mean' in losses
    assert 'gate_proj_bias_shift_abs_mean_normalized' in losses
    assert 'gate_proj_bias_shift_abs_mean_1' in losses
    assert 'gate_proj_bias_shift_abs_mean_normalized_1' in losses
    assert torch.isfinite(losses['gate_proj_bias_shift_abs_mean'])
    assert torch.isfinite(losses['gate_proj_bias_shift_abs_mean_normalized'])
    assert losses['gate_proj_bias_shift_abs_mean'].item() >= 0.0
    assert losses['gate_proj_bias_shift_abs_mean_normalized'].item() >= 0.0
    torch.testing.assert_close(
        losses['gate_proj_bias_shift_abs_mean'],
        torch.tensor(losses['gate_proj_bias_shift_abs_mean_1']),
    )
    torch.testing.assert_close(
        losses['gate_proj_bias_shift_abs_mean_normalized'],
        torch.tensor(losses['gate_proj_bias_shift_abs_mean_normalized_1']),
    )


    assert losses['gate_proj_bias_shift_abs_top5p_mean'].numel() == 1
    assert losses['gate_proj_bias_shift_abs_bottom5p_mean'].numel() == 1
    assert 'gate_proj_bias_shift_abs_top5p_mean_1' in losses
    assert 'gate_proj_bias_shift_abs_bottom5p_mean_1' in losses


def test_gate_proj_bias_references_are_not_auto_refreshed_without_config_opt_in():
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
        use_gate_proj_bias=True,
        debug=False,
    )
    model = GPT(config)
    model.init_weights()

    assert model.transformer.h[1].mlp.experts.initial_gate_proj_bias is None

    model.refresh_gate_proj_bias_references()

    assert model.transformer.h[1].mlp.experts.initial_gate_proj_bias is not None


def test_gate_proj_bias_shift_stats_default_to_zero_when_bias_disabled():
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
        use_gate_proj_bias=False,
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

    assert losses['gate_proj_bias_shift_abs_top5p_mean'].shape == torch.Size([])
    assert losses['gate_proj_bias_shift_abs_bottom5p_mean'].shape == torch.Size([])
    assert losses['gate_proj_bias_shift_abs_top5p_mean'].item() == 0.0
    assert losses['gate_proj_bias_shift_abs_bottom5p_mean'].item() == 0.0
    assert torch.isfinite(losses['gate_proj_bias_shift_abs_top5p_mean'])
    assert torch.isfinite(losses['gate_proj_bias_shift_abs_bottom5p_mean'])


def test_gate_proj_bias_references_can_auto_refresh_when_config_enabled():
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
        use_gate_proj_bias=True,
        refresh_gate_proj_bias_references=True,
        debug=False,
    )
    model = GPT(config)
    model.init_weights()

    assert model.transformer.h[1].mlp.experts.initial_gate_proj_bias is not None


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
    assert experts.gate_proj_bias is None


def test_gate_projection_bias_has_expected_shape_when_enabled():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_gate_proj_bias=True,
        debug=False,
    )

    experts = Qwen3MLPExperts(config)

    assert experts.gate_proj_bias is not None
    assert experts.gate_proj_bias.ndim == 2
    assert experts.gate_proj_bias.shape == (config.n_exp, 4 * config.n_embd)


@pytest.mark.parametrize(
    ("granularity", "parameter_shape", "expected_materialized_shape"),
    [
        ("per-gate", (2, 16), (2, 16)),
        ("per-expert", (2,), (2, 16)),
        ("per-layer", (1,), (2, 16)),
    ],
)
def test_gate_projection_bias_materializes_expected_shape_for_local_granularities(
    granularity,
    parameter_shape,
    expected_materialized_shape,
):
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_gate_proj_bias=True,
        global_gate_proj_bias_granularity=granularity,
        debug=False,
    )

    experts = Qwen3MLPExperts(config)

    assert experts.gate_proj_bias is not None
    assert tuple(experts.gate_proj_bias.shape) == parameter_shape
    assert tuple(experts._materialize_gate_proj_bias().shape) == expected_materialized_shape


def test_gate_projection_bias_materialization_broadcasts_per_expert_values():
    config = GPTConfig(
        n_exp=3,
        n_embd=4,
        use_gate_proj_bias=True,
        global_gate_proj_bias_granularity="per-expert",
        debug=False,
    )

    experts = Qwen3MLPExperts(config)
    with torch.no_grad():
        experts.gate_proj_bias.copy_(torch.tensor([1.0, 2.0, 3.0]))

    materialized = experts._materialize_gate_proj_bias()

    torch.testing.assert_close(materialized[0], torch.ones(16))
    torch.testing.assert_close(materialized[1], torch.full((16,), 2.0))
    torch.testing.assert_close(materialized[2], torch.full((16,), 3.0))


def test_gate_projection_bias_global_granularity_shares_one_parameter_across_layers():
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
        use_aux_loss=False,
        use_router_z_loss=False,
        use_gate_proj_bias=True,
        global_gate_proj_bias_granularity="global",
        debug=False,
    )

    model = GPT(config)
    moe_experts = [
        block.mlp.experts
        for block in model.transformer.h
        if hasattr(block.mlp, 'experts') and isinstance(block.mlp.experts, Qwen3MLPExperts)
    ]

    assert model.global_gate_proj_bias is not None
    assert tuple(model.global_gate_proj_bias.shape) == (1,)
    assert all(experts.gate_proj_bias is None for experts in moe_experts)
    assert all(experts._get_gate_proj_bias_parameter() is model.global_gate_proj_bias for experts in moe_experts)
    assert all(tuple(experts._materialize_gate_proj_bias().shape) == (config.n_exp, 4 * config.n_embd) for experts in moe_experts)


def test_gate_projection_bias_respects_start_layer_cutoff():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_gate_proj_bias=True,
        gate_proj_bias_start_layer=3,
        debug=False,
    )

    early_experts = Qwen3MLPExperts(config, layer_idx=2)
    late_experts = Qwen3MLPExperts(config, layer_idx=3)

    assert early_experts.gate_proj_bias is None
    assert late_experts.gate_proj_bias is not None


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