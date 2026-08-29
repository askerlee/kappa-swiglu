import math
import pytest
import torch
import torch.nn.functional as F
from copy import deepcopy

from nanochat.configuration_nanomoe_gpt import GPTConfig
from nanochat.engine import KVCache
from nanochat.gpt import GPT, MANAGER, GateProjBiasEmaTargetKeeper, MOELayer, Qwen3MLP, Qwen3MLPExperts, Router, _chunked_cross_entropy, _save_activations_on_cpu, scale_grad
from nanochat.manager import MOEManager


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


def test_dynamic_kappa_bias_backprops_into_selected_router_scores():
    torch.manual_seed(0)
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_kappa_swiglu=True,
        debug=False,
    )
    experts = Qwen3MLPExperts(config)

    x = torch.randn(config.n_exp, 5, config.n_embd, requires_grad=True)
    selected_router_scores = torch.randn(config.n_exp, 5, requires_grad=True)
    out = experts(x, selected_router_scores=selected_router_scores).sum()
    out.backward()

    assert selected_router_scores.grad is not None


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


def test_kappa_router_softmax_modulates_temperature_and_receives_gradient():
    config = GPTConfig(
        n_exp=3,
        moe_top_k=2,
        n_embd=2,
        eval_capacity=100.0,
        use_aux_loss=False,
        use_router_z_loss=False,
        use_kappa_router_softmax=True,
        router_kappa_slope_max_scale=4.0,
    )
    router = Router(config).eval()
    with torch.no_grad():
        router.w_g.weight.copy_(torch.tensor([[2.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]))
        target_slope = torch.tensor(2.0)
        router.router_softmax_kappa.copy_(
            torch.atanh(torch.log(target_slope) / math.log(config.router_kappa_slope_max_scale))
        )
    x = torch.tensor([[[1.0, 0.0]]])

    _, router_probs, _, top_k_indices, _ = router(x)

    assert torch.equal(top_k_indices, torch.tensor([[0, 1]]))
    expected_probs = F.softmax(torch.tensor([[2.0, 1.0]]) * target_slope, dim=-1)
    torch.testing.assert_close(router_probs, expected_probs)

    router_probs.square().sum().backward()
    assert router.router_softmax_kappa.grad is not None
    assert router.router_softmax_kappa.grad.abs() > 0


def test_kappa_router_softmax_l2_loss_is_mean_square():
    config = GPTConfig(
        n_exp=2,
        moe_top_k=2,
        n_embd=4,
        use_aux_loss=False,
        use_router_z_loss=False,
        use_kappa_router_softmax=True,
    )
    router = Router(config)
    with torch.no_grad():
        router.router_softmax_kappa.fill_(2.0)
    MANAGER.reset("router_softmax_kappa_l2_loss")

    router(torch.randn(1, 2, config.n_embd))
    loss = MANAGER.aggregate("router_softmax_kappa_l2_loss")

    MANAGER.reset("router_softmax_kappa_l2_loss")
    torch.testing.assert_close(loss, torch.tensor(4.0))
    loss.backward()
    torch.testing.assert_close(router.router_softmax_kappa.grad, torch.tensor(4.0))


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


def test_router_full_delta_preserves_logits_and_receives_gradients():
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
    expected_logits = F.linear(x.view(-1, config.n_embd), router.w_g.weight)
    expected_scores, expected_indices = router(x)[2:4]

    router.setup_router_wg_delta()
    actual_logits = F.linear(x.view(-1, config.n_embd), router.effective_w_g_weight())
    actual_scores, actual_indices = router(x)[2:4]
    torch.testing.assert_close(actual_logits, expected_logits)
    torch.testing.assert_close(actual_scores, expected_scores)
    torch.testing.assert_close(actual_indices, expected_indices)
    assert router.w_g.weight.requires_grad
    assert router.w_g_delta.requires_grad

    actual_logits.sum().backward()
    assert router.w_g.weight.grad is not None
    assert router.w_g_delta.grad is not None
    torch.testing.assert_close(router.w_g.weight.grad, router.w_g_delta.grad)

    with torch.no_grad():
        router.w_g_delta.fill_(0.25)
    assert not torch.equal(router.effective_w_g_weight(), router.w_g.weight)
    router.enable_router_wg_delta(False)
    assert router.w_g.weight.requires_grad
    assert not router.w_g_delta.requires_grad
    torch.testing.assert_close(router.effective_w_g_weight(), router.w_g.weight)

    config.router_wg_delta = True
    reloaded_router = Router(config)
    reloaded_router.load_state_dict(router.state_dict(), strict=True)
    torch.testing.assert_close(reloaded_router.w_g_delta, router.w_g_delta)


def test_router_wg_delta_l2_loss_tracks_allocated_delta_and_does_not_touch_base_weights():
    config = GPTConfig(
        sequence_len=4,
        vocab_size=32,
        n_layer=2,
        moe_start_layer=0,
        n_exp=2,
        n_embd=32,
        n_head=4,
        use_aux_loss=False,
        use_router_z_loss=False,
    )
    model = GPT(config, pad_vocab_size_to=1)
    model.init_weights()
    model.setup_router_wg_delta()
    routers = [block.mlp.router for block in model.transformer.h]
    with torch.no_grad():
        routers[0].w_g_delta.fill_(0.25)
        routers[1].w_g_delta.fill_(0.5)

    delta_l2_loss = model.compute_router_wg_delta_l2_loss()
    torch.testing.assert_close(delta_l2_loss, torch.tensor((0.25 ** 2 + 0.5 ** 2) / 2))
    delta_l2_loss.backward()
    assert all(router.w_g_delta.grad is not None for router in routers)
    assert all(router.w_g.weight.grad is None for router in routers)

    model.enable_router_wg_delta(False)
    disabled_loss = model.compute_router_wg_delta_l2_loss()
    torch.testing.assert_close(disabled_loss, delta_l2_loss.detach())
    assert not disabled_loss.requires_grad

    ids = torch.randint(0, config.vocab_size, (1, config.sequence_len))
    _, base_mode_losses = model(ids, ids)
    torch.testing.assert_close(
        base_mode_losses["router_wg_delta_l2_loss"],
        delta_l2_loss.detach(),
    )


def test_gpt_forward_skips_router_wg_delta_l2_when_not_enabled(monkeypatch):
    config = GPTConfig(
        sequence_len=4,
        vocab_size=32,
        n_layer=2,
        moe_start_layer=0,
        n_exp=2,
        n_embd=32,
        n_head=4,
        use_aux_loss=False,
        use_router_z_loss=False,
    )
    model = GPT(config, pad_vocab_size_to=1)
    model.init_weights()

    def fail_if_called():
        raise AssertionError("router delta L2 must not be computed when disabled")

    monkeypatch.setattr(model, "compute_router_wg_delta_l2_loss", fail_if_called)
    ids = torch.randint(0, config.vocab_size, (1, config.sequence_len))
    _, losses = model(ids, ids)

    assert losses["router_wg_delta_l2_loss"] == 0


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
    assert experts.use_kappa_scale is True


def test_dense_qwen3_mlp_uses_placeholder_bias_before_start_layer():
    torch.manual_seed(0)
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_kappa_swiglu=True,
        kappa_input="router_probs",
        kappa_input_constant=1.0,
        constant_kappa_bias_dense_layers=True,
        kappa_bias_start_layer=2,
        debug=False,
    )

    mlp = Qwen3MLP(config, layer_idx=0)
    x = torch.randn(3, 5, config.n_embd)

    assert mlp.use_kappa_swiglu is True
    assert mlp.has_active_kappa_bias is False
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
        assert block.mlp._expert_router_scores_cache is None


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
        "router_softmax_kappa_l2_loss",
        "kappa_bias_l2_loss",
        "kappa_scale_l2_loss",
        "kappa_bias_ema_rms_reg_loss",
        "kappa_scale_ema_rms_reg_loss",
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
        kappa_bias[1].fill_(-2.0)

    idx = torch.randint(0, config.vocab_size, (2, 4))
    targets = torch.randint(0, config.vocab_size, (2, 4))
    _, losses = model(idx, targets)

    torch.testing.assert_close(losses["kappa_bias_l2_loss"], torch.tensor(4.0))


def test_gpt_total_ut_steps_updates_kappa_ema_once_and_applies_loss_each_loop():
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
        kappa_bias_ema_rms_reg=True,
        kappa_bias_l2_ema_beta=0.5,
        kappa_bias_l2_ema_anchor_start=0.0,
        kappa_bias_l2_ema_anchor_end=1.0,
        kappa_bias_l2_ema_floor_frac=0.8,
        debug=False,
    )
    model = GPT(config)
    model.init_weights()
    model.set_kappa_bias_ema_rms_reg_total_iterations(1)
    experts = model.transformer.h[0].mlp.experts
    with torch.no_grad():
        experts.kappa_bias.fill_(2.0)
    model.set_kappa_bias_ema_rms_reg_step(0)
    model._update_kappa_ema_rms_targets()

    with torch.no_grad():
        experts.kappa_bias.fill_(0.5)
    model.set_kappa_bias_ema_rms_reg_step(1)
    idx = torch.randint(0, config.vocab_size, (2, 4))
    targets = torch.randint(0, config.vocab_size, (2, 4))
    _, losses = model(idx, targets)

    torch.testing.assert_close(
        experts.kappa_bias_ema_rms_reg_keeper.ema_rms,
        torch.tensor([1.25, 1.25]),
    )
    torch.testing.assert_close(
        losses["kappa_bias_ema_rms_reg_loss"],
        torch.tensor(0.25),
    )


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

    expert_inputs, expert_router_scores = layer._build_expert_inputs_functional(
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
    torch.testing.assert_close(expert_router_scores, expected_scores)

    (expert_inputs.sum() + expert_router_scores.sum()).backward()
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


def test_moe_select_gate_confidence_can_normalize_top_logits():
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

    actual = moe_layer._select_gate_confidence(
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


def test_moe_select_gate_confidence_smooths_tiny_router_weight_norms():
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

    actual = moe_layer._select_gate_confidence(
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


def test_moe_select_gate_confidence_keeps_partial_norm_scale_near_unit():
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

    actual = moe_layer._select_gate_confidence(
        top_k_scores,
        router_probs,
        x_flat=torch.zeros(2, config.n_embd),
        top_k_indices=top_k_indices,
    )

    torch.testing.assert_close(actual, target_gate_confidence)

def test_kappa_bias_l2_loss_is_mean_square():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_kappa_swiglu=True,
        debug=False,
    )
    experts = Qwen3MLPExperts(config)

    MANAGER.reset("kappa_bias_l2_loss")

    kappa_bias = torch.tensor([
        [-0.5, 0.5],
        [-0.25, 0.25],
    ])
    experts._accumulate_kappa_bias_l2_losses(kappa_bias)

    loss = MANAGER.aggregate("kappa_bias_l2_loss")

    MANAGER.reset("kappa_bias_l2_loss")

    torch.testing.assert_close(loss, kappa_bias.square().mean())


def test_kappa_bias_l2_losses_are_reported_from_kappa_biases():
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

    assert torch.isfinite(losses['kappa_bias_l2_loss'])
    torch.testing.assert_close(losses['kappa_bias_l2_loss'], torch.tensor(4.0))


def test_kappa_bias_ema_rms_reg_loss_is_added_on_top_of_l2_loss():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_kappa_swiglu=True,
        kappa_bias_ema_rms_reg=True,
        kappa_bias_l2_ema_beta=0.99,
        kappa_bias_l2_ema_anchor_start=0.0,
        kappa_bias_l2_ema_anchor_end=0.0,
        kappa_bias_l2_ema_floor_frac=0.8,
        debug=False,
    )
    experts = Qwen3MLPExperts(config)

    MANAGER.reset("kappa_bias_l2_loss")
    MANAGER.reset("kappa_bias_ema_rms_reg_loss")
    experts.set_kappa_bias_ema_rms_reg_total_iterations(1)
    experts.set_kappa_bias_ema_rms_reg_step(0)
    first_value = torch.full((2, 16), 2.0)
    experts.kappa_bias_ema_rms_reg_keeper.update(first_value, step=0)
    experts._accumulate_kappa_bias_l2_losses(first_value)
    first_l2_loss = MANAGER.aggregate("kappa_bias_l2_loss")
    first_ema_rms_reg_loss = MANAGER.aggregate("kappa_bias_ema_rms_reg_loss")
    MANAGER.reset("kappa_bias_l2_loss")
    MANAGER.reset("kappa_bias_ema_rms_reg_loss")

    experts.set_kappa_bias_ema_rms_reg_step(1)
    second_value = torch.full((2, 16), 0.5)
    experts.kappa_bias_ema_rms_reg_keeper.update(second_value, step=1)
    experts._accumulate_kappa_bias_l2_losses(second_value)
    second_l2_loss = MANAGER.aggregate("kappa_bias_l2_loss")
    second_ema_rms_reg_loss = MANAGER.aggregate("kappa_bias_ema_rms_reg_loss")
    MANAGER.reset("kappa_bias_l2_loss")
    MANAGER.reset("kappa_bias_ema_rms_reg_loss")

    torch.testing.assert_close(first_l2_loss, torch.tensor(4.0))
    torch.testing.assert_close(first_ema_rms_reg_loss, torch.tensor(0.0))
    torch.testing.assert_close(second_l2_loss, torch.tensor(0.25))
    torch.testing.assert_close(second_ema_rms_reg_loss, torch.tensor((1.6 - 0.5) ** 2))


def test_moe_manager_registers_kappa_bias_ema_rms_reg_losses_by_default():
    manager = MOEManager()

    manager.add("kappa_bias_ema_rms_reg_loss", torch.tensor(1.25))
    manager.add("kappa_scale_ema_rms_reg_loss", torch.tensor(0.75))

    torch.testing.assert_close(
        manager.aggregate("kappa_bias_ema_rms_reg_loss"),
        torch.tensor(1.25),
    )
    torch.testing.assert_close(
        manager.aggregate("kappa_scale_ema_rms_reg_loss"),
        torch.tensor(0.75),
    )


def test_kappa_bias_ema_target_keeper_raises_on_nonfinite_input():
    keeper = GateProjBiasEmaTargetKeeper(
        beta=0.99,
        anchor_start=0.0,
        anchor_end=1.0,
        floor_frac=0.8,
    )

    with pytest.raises(RuntimeError, match="non-finite value"):
        keeper.update(torch.tensor([float('nan')]), step=0)


def test_kappa_bias_ema_target_keeper_raises_on_nonfinite_target_before_loss():
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


def test_kappa_bias_ema_target_keeper_loss_compiles_without_readiness_graph_break():
    keeper = GateProjBiasEmaTargetKeeper(
        beta=0.99,
        anchor_start=0.0,
        anchor_end=1.0,
        floor_frac=0.8,
    )
    keeper.target_rms.fill_(2.0)
    compiled_loss = torch.compile(keeper.loss, fullgraph=True, backend="eager")

    keeper.target_ready.fill_(False)
    torch.testing.assert_close(compiled_loss(torch.ones(4)), torch.tensor(0.0))

    keeper.target_ready.fill_(True)
    torch.testing.assert_close(compiled_loss(torch.ones(4)), torch.tensor(0.36))


def test_kappa_bias_ema_target_keeper_tracks_ut_passes_independently():
    keeper = GateProjBiasEmaTargetKeeper(
        beta=0.5,
        anchor_start=0.0,
        anchor_end=1.0,
        floor_frac=0.8,
        total_ut_steps=2,
    )

    keeper.update(torch.full((4,), 2.0), step=0, current_ut=0)
    keeper.update(torch.full((4,), 4.0), step=0, current_ut=1)

    torch.testing.assert_close(keeper.ema_rms, torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(
        keeper.loss(torch.full((4,), 1.0), current_ut=0),
        torch.tensor((1.6 - 1.0) ** 2),
    )
    torch.testing.assert_close(
        keeper.loss(torch.full((4,), 1.0), current_ut=1),
        torch.tensor((3.2 - 1.0) ** 2),
    )


def test_kappa_bias_ema_target_error_includes_module_source():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_kappa_swiglu=True,
        kappa_bias_ema_rms_reg=True,
        debug=False,
    )
    experts = Qwen3MLPExperts(config, layer_idx=3)

    with pytest.raises(RuntimeError, match=r"Qwen3MLPExperts\(layer=3, granularity=per-gate\)\.kappa_bias"):
        with torch.no_grad():
            experts.kappa_bias.fill_(float('nan'))
        experts.update_kappa_ema_rms_targets()


def test_kappa_bias_ema_target_loss_has_finite_gradient_at_zero():
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


def test_kappa_scale_ema_rms_reg_loss_is_added_on_top_of_l2_loss():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_kappa_swiglu=True,
        kappa_input="router_probs",
        kappa_bias_ema_rms_reg=True,
        kappa_bias_l2_ema_beta=0.99,
        kappa_bias_l2_ema_anchor_start=0.0,
        kappa_bias_l2_ema_anchor_end=0.0,
        kappa_bias_l2_ema_floor_frac=0.8,
        debug=False,
    )
    experts = Qwen3MLPExperts(config)

    MANAGER.reset("kappa_scale_l2_loss")
    MANAGER.reset("kappa_scale_ema_rms_reg_loss")
    experts.set_kappa_bias_ema_rms_reg_total_iterations(1)
    experts.set_kappa_bias_ema_rms_reg_step(0)
    first_value = torch.full((2, 16), 2.0)
    experts.kappa_scale_ema_rms_reg_keeper.update(first_value, step=0)
    experts._accumulate_kappa_scale_l2_losses(first_value)
    first_l2_loss = MANAGER.aggregate("kappa_scale_l2_loss")
    first_ema_rms_reg_loss = MANAGER.aggregate("kappa_scale_ema_rms_reg_loss")
    MANAGER.reset("kappa_scale_l2_loss")
    MANAGER.reset("kappa_scale_ema_rms_reg_loss")

    experts.set_kappa_bias_ema_rms_reg_step(1)
    second_value = torch.full((2, 16), 0.25)
    experts.kappa_scale_ema_rms_reg_keeper.update(second_value, step=1)
    experts._accumulate_kappa_scale_l2_losses(second_value)
    second_l2_loss = MANAGER.aggregate("kappa_scale_l2_loss")
    second_ema_rms_reg_loss = MANAGER.aggregate("kappa_scale_ema_rms_reg_loss")
    MANAGER.reset("kappa_scale_l2_loss")
    MANAGER.reset("kappa_scale_ema_rms_reg_loss")

    torch.testing.assert_close(first_l2_loss, torch.tensor(4.0))
    torch.testing.assert_close(first_ema_rms_reg_loss, torch.tensor(0.0))
    torch.testing.assert_close(second_l2_loss, torch.tensor(0.0625))
    torch.testing.assert_close(second_ema_rms_reg_loss, torch.tensor((1.6 - 0.25) ** 2))


def test_dense_kappa_scale_ema_rms_reg_loss_is_added_on_top_of_l2_loss():
    config = GPTConfig(
        n_embd=4,
        use_kappa_swiglu=True,
        constant_kappa_bias_dense_layers=True,
        kappa_input="constant",
        kappa_bias_ema_rms_reg=True,
        kappa_bias_l2_ema_beta=0.99,
        kappa_bias_l2_ema_anchor_start=0.0,
        kappa_bias_l2_ema_anchor_end=0.0,
        kappa_bias_l2_ema_floor_frac=0.8,
        debug=False,
    )
    mlp = Qwen3MLP(config)

    MANAGER.reset("kappa_scale_l2_loss")
    MANAGER.reset("kappa_scale_ema_rms_reg_loss")
    mlp.set_kappa_bias_ema_rms_reg_total_iterations(1)
    mlp.set_kappa_bias_ema_rms_reg_step(0)
    first_value = torch.full((16,), 2.0)
    mlp.kappa_scale_ema_rms_reg_keeper.update(first_value, step=0)
    mlp._accumulate_kappa_scale_l2_losses(first_value)
    first_l2_loss = MANAGER.aggregate("kappa_scale_l2_loss")
    first_ema_rms_reg_loss = MANAGER.aggregate("kappa_scale_ema_rms_reg_loss")
    MANAGER.reset("kappa_scale_l2_loss")
    MANAGER.reset("kappa_scale_ema_rms_reg_loss")

    mlp.set_kappa_bias_ema_rms_reg_step(1)
    second_value = torch.full((16,), 0.25)
    mlp.kappa_scale_ema_rms_reg_keeper.update(second_value, step=1)
    mlp._accumulate_kappa_scale_l2_losses(second_value)
    second_l2_loss = MANAGER.aggregate("kappa_scale_l2_loss")
    second_ema_rms_reg_loss = MANAGER.aggregate("kappa_scale_ema_rms_reg_loss")
    MANAGER.reset("kappa_scale_l2_loss")
    MANAGER.reset("kappa_scale_ema_rms_reg_loss")

    torch.testing.assert_close(first_l2_loss, torch.tensor(4.0))
    torch.testing.assert_close(first_ema_rms_reg_loss, torch.tensor(0.0))
    torch.testing.assert_close(second_l2_loss, torch.tensor(0.0625))
    torch.testing.assert_close(second_ema_rms_reg_loss, torch.tensor((1.6 - 0.25) ** 2))


def test_kappa_bias_ema_target_buffers_load_from_older_checkpoints():
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
        kappa_bias_ema_rms_reg=True,
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
    assert torch.equal(experts.kappa_bias_ema_rms_reg_keeper.ema_rms, torch.zeros(1))
    assert torch.equal(experts.kappa_scale_ema_rms_reg_keeper.ema_rms, torch.zeros(1))
    assert not bool(experts.kappa_bias_ema_rms_reg_keeper.initialized.item())
    assert not bool(experts.kappa_scale_ema_rms_reg_keeper.initialized.item())


def test_kappa_bias_ema_scalar_buffers_expand_across_ut_passes_on_load():
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=1,
        moe_start_layer=0,
        num_moe_layers=1,
        n_exp=2,
        n_embd=32,
        n_head=4,
        total_ut_steps=2,
        use_kappa_swiglu=True,
        kappa_bias_ema_rms_reg=True,
        debug=False,
    )
    model = GPT(config)
    state_dict = model.state_dict()
    keeper_prefix = "transformer.h.0.mlp.experts.kappa_bias_ema_rms_reg_keeper"
    state_dict[f"{keeper_prefix}.ema_rms"] = torch.tensor(1.5)
    state_dict[f"{keeper_prefix}.target_rms"] = torch.tensor(1.25)
    state_dict[f"{keeper_prefix}.initialized"] = torch.tensor(True)
    state_dict[f"{keeper_prefix}.target_ready"] = torch.tensor(True)

    load_result = model.load_state_dict(state_dict, strict=True)

    assert not load_result.missing_keys
    assert not load_result.unexpected_keys
    keeper = model.transformer.h[0].mlp.experts.kappa_bias_ema_rms_reg_keeper
    torch.testing.assert_close(keeper.ema_rms, torch.tensor([1.5, 1.5]))
    torch.testing.assert_close(keeper.target_rms, torch.tensor([1.25, 1.25]))
    assert keeper.initialized.tolist() == [True, True]
    assert keeper.target_ready.tolist() == [True, True]


def test_kappa_bias_ema_anchor_fractions_resolve_against_total_iterations():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_kappa_swiglu=True,
        kappa_bias_ema_rms_reg=True,
        kappa_bias_l2_ema_beta=0.99,
        kappa_bias_l2_ema_anchor_start=0.4,
        kappa_bias_l2_ema_anchor_end=0.8,
        kappa_bias_l2_ema_floor_frac=0.8,
        debug=False,
    )
    experts = Qwen3MLPExperts(config)
    experts.set_kappa_bias_ema_rms_reg_total_iterations(10)

    anchor_start, anchor_end = experts.kappa_bias_ema_rms_reg_keeper._resolve_anchor_steps()

    assert anchor_start == 4
    assert anchor_end == 8


def test_kappa_bias_ema_rms_reg_is_zero_before_anchor():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_kappa_swiglu=True,
        kappa_bias_ema_rms_reg=True,
        kappa_bias_l2_ema_beta=0.99,
        kappa_bias_l2_ema_anchor_start=0.4,
        kappa_bias_l2_ema_anchor_end=0.8,
        kappa_bias_l2_ema_floor_frac=0.8,
        debug=False,
    )
    experts = Qwen3MLPExperts(config)
    experts.set_kappa_bias_ema_rms_reg_total_iterations(10)

    value = torch.full((2, 16), 2.0)
    MANAGER.reset("kappa_bias_l2_loss")
    MANAGER.reset("kappa_bias_ema_rms_reg_loss")
    experts.set_kappa_bias_ema_rms_reg_step(0)
    experts.kappa_bias_ema_rms_reg_keeper.update(value, step=0)
    experts._accumulate_kappa_bias_l2_losses(value)
    l2_loss = MANAGER.aggregate("kappa_bias_l2_loss")
    ema_rms_reg_loss = MANAGER.aggregate("kappa_bias_ema_rms_reg_loss")
    MANAGER.reset("kappa_bias_l2_loss")
    MANAGER.reset("kappa_bias_ema_rms_reg_loss")

    torch.testing.assert_close(l2_loss, value.square().mean())
    torch.testing.assert_close(ema_rms_reg_loss, torch.tensor(0.0))
    assert not bool(experts.kappa_bias_ema_rms_reg_keeper.target_ready.item())


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
    MANAGER.reset("kappa_slope_scale_abs_mean_normalized")

    selected_router_scores = torch.tensor([
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
        experts._update_kappa_slope_scale_stats(slope_scales, selected_router_scores)
    finally:
        MANAGER.collect_load_balancing_stats = old_collect

    shift_abs_mean = MANAGER.aggregate("kappa_slope_scale_abs_mean")
    normalized_shift_abs_mean = MANAGER.aggregate("kappa_slope_scale_abs_mean_normalized")

    expected_mean = slope_scales[0].float().mean().reshape(1)

    MANAGER.reset("kappa_slope_scale_abs_mean")
    MANAGER.reset("kappa_slope_scale_abs_mean_normalized")

    torch.testing.assert_close(shift_abs_mean, expected_mean)
    torch.testing.assert_close(normalized_shift_abs_mean, expected_mean)


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
    MANAGER.reset("kappa_slope_scale_abs_mean_normalized")

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
    assert MANAGER.aggregate("kappa_slope_scale_abs_mean_normalized") is None
    assert experts.last_gate_stats is None

    MANAGER.reset("kappa_slope_scale_abs_mean")
    MANAGER.reset("kappa_slope_scale_abs_mean_normalized")


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
    assert 'kappa_slope_scale_abs_mean_normalized' in losses
    assert 'kappa_slope_scale_abs_mean_1' in losses
    assert 'kappa_slope_scale_abs_mean_normalized_1' in losses
    assert torch.isfinite(losses['kappa_slope_scale_abs_mean'])
    assert torch.isfinite(losses['kappa_slope_scale_abs_mean_normalized'])
    assert losses['kappa_slope_scale_abs_mean'].item() >= 0.0
    assert losses['kappa_slope_scale_abs_mean_normalized'].item() >= 0.0
    torch.testing.assert_close(
        losses['kappa_slope_scale_abs_mean'],
        torch.tensor([losses['kappa_slope_scale_abs_mean_1']]),
    )
    torch.testing.assert_close(
        losses['kappa_slope_scale_abs_mean_normalized'],
        torch.tensor([losses['kappa_slope_scale_abs_mean_normalized_1']]),
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
    assert 'kappa_slope_scale_abs_mean_normalized' in losses
    assert 'kappa_slope_scale_abs_mean_1' in losses
    assert 'kappa_slope_scale_abs_mean_normalized_1' in losses
    assert torch.isfinite(losses['kappa_slope_scale_abs_mean'])
    assert torch.isfinite(losses['kappa_slope_scale_abs_mean_normalized'])
    assert losses['kappa_slope_scale_abs_mean'].item() >= 0.0
    assert losses['kappa_slope_scale_abs_mean_normalized'].item() >= 0.0
    torch.testing.assert_close(
        losses['kappa_slope_scale_abs_mean'],
        torch.tensor([losses['kappa_slope_scale_abs_mean_1']]),
    )
    torch.testing.assert_close(
        losses['kappa_slope_scale_abs_mean_normalized'],
        torch.tensor([losses['kappa_slope_scale_abs_mean_normalized_1']]),
    )


    assert losses['kappa_slope_scale_abs_top5p_mean'].numel() == 1
    assert losses['kappa_slope_scale_abs_bottom5p_mean'].numel() == 1
    assert 'kappa_slope_scale_abs_top5p_mean_1' in losses
    assert 'kappa_slope_scale_abs_bottom5p_mean_1' in losses


def test_kappa_bias_references_are_not_auto_refreshed_without_config_opt_in():
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

    model.refresh_kappa_bias_references()

    assert model.transformer.h[1].mlp.experts.initial_kappa_bias is not None


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


def test_kappa_bias_references_can_auto_refresh_when_config_enabled():
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
        refresh_kappa_bias_references=True,
        debug=False,
    )
    model = GPT(config)
    model.init_weights()

    assert model.transformer.h[1].mlp.experts.initial_kappa_bias is not None


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
        config.total_ut_steps,
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
        global_kappa_bias_granularity=granularity,
        debug=False,
    )

    experts = Qwen3MLPExperts(config)

    assert experts.kappa_bias is not None
    assert tuple(experts.kappa_bias.shape) == (config.total_ut_steps, *parameter_shape)
    assert tuple(experts._materialize_kappa_bias().shape) == expected_materialized_shape


def test_kappa_bias_materialization_broadcasts_per_expert_values():
    config = GPTConfig(
        n_exp=3,
        n_embd=4,
        use_kappa_swiglu=True,
        global_kappa_bias_granularity="per-expert",
        debug=False,
    )

    experts = Qwen3MLPExperts(config)
    with torch.no_grad():
        experts.kappa_bias[0].copy_(torch.tensor([1.0, 2.0, 3.0]))

    materialized = experts._materialize_kappa_bias()

    torch.testing.assert_close(materialized[0], torch.ones(16))
    torch.testing.assert_close(materialized[1], torch.full((16,), 2.0))
    torch.testing.assert_close(materialized[2], torch.full((16,), 3.0))


def test_kappa_bias_selects_and_backprops_only_the_current_ut_pass():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        total_ut_steps=2,
        use_kappa_swiglu=True,
        global_kappa_bias_granularity="per-expert",
        debug=False,
    )
    experts = Qwen3MLPExperts(config)
    with torch.no_grad():
        experts.kappa_bias.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        experts.kappa_scale.copy_(torch.tensor([[5.0, 6.0], [7.0, 8.0]]))

    materialized_bias = experts._materialize_kappa_bias(current_ut=1)
    materialized_scale = experts._materialize_kappa_scale(current_ut=1)
    torch.testing.assert_close(materialized_bias[0], torch.full((16,), 3.0))
    torch.testing.assert_close(materialized_bias[1], torch.full((16,), 4.0))
    torch.testing.assert_close(materialized_scale[0], torch.full((16,), 7.0))
    torch.testing.assert_close(materialized_scale[1], torch.full((16,), 8.0))

    (materialized_bias.sum() + materialized_scale.sum()).backward()
    torch.testing.assert_close(experts.kappa_bias.grad[0], torch.zeros(2))
    torch.testing.assert_close(experts.kappa_bias.grad[1], torch.full((2,), 16.0))
    torch.testing.assert_close(experts.kappa_scale.grad[0], torch.zeros(2))
    torch.testing.assert_close(experts.kappa_scale.grad[1], torch.full((2,), 16.0))


def test_dense_kappa_bias_selects_only_the_current_ut_pass():
    config = GPTConfig(
        n_embd=4,
        total_ut_steps=2,
        use_kappa_swiglu=True,
        constant_kappa_bias_dense_layers=True,
        global_kappa_bias_granularity="per-layer",
        debug=False,
    )
    mlp = Qwen3MLP(config)
    with torch.no_grad():
        mlp.kappa_bias.copy_(torch.tensor([[1.0], [2.0]]))

    materialized = mlp._materialize_kappa_bias(current_ut=1)
    torch.testing.assert_close(materialized, torch.full((16,), 2.0))

    materialized.sum().backward()
    torch.testing.assert_close(mlp.kappa_bias.grad[0], torch.zeros(1))
    torch.testing.assert_close(mlp.kappa_bias.grad[1], torch.full((1,), 16.0))


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
        use_aux_loss=False,
        use_router_z_loss=False,
        use_kappa_swiglu=True,
        global_kappa_bias_granularity="global",
        debug=False,
    )

    model = GPT(config)
    moe_experts = [
        block.mlp.experts
        for block in model.transformer.h
        if hasattr(block.mlp, 'experts') and isinstance(block.mlp.experts, Qwen3MLPExperts)
    ]

    assert model.global_kappa_bias is not None
    assert tuple(model.global_kappa_bias.shape) == (config.total_ut_steps, 1)
    assert all(experts.kappa_bias is None for experts in moe_experts)
    assert all(experts._get_kappa_bias_parameter() is model.global_kappa_bias for experts in moe_experts)
    assert all(tuple(experts._materialize_kappa_bias().shape) == (config.n_exp, 4 * config.n_embd) for experts in moe_experts)


def test_kappa_bias_respects_start_layer_cutoff():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_kappa_swiglu=True,
        kappa_bias_start_layer=3,
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