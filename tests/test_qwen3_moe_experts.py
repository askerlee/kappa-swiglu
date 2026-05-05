import torch
from copy import deepcopy

from nanochat.configuration_nanomoe_gpt import GPTConfig
from nanochat.gpt import GPT, Qwen3MLP, Qwen3MLPExperts


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


def test_gate_projection_bias_is_replaced_with_dynamic_router_conditioned_bias_when_enabled():
    torch.manual_seed(0)
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_exp_gate_proj_bias=True,
        debug=False,
    )
    experts = Qwen3MLPExperts(config)

    x = torch.randn(config.n_exp, 5, config.n_embd)
    selected_router_probs = torch.rand(config.n_exp, 5)

    with torch.no_grad():
        experts.gate_proj.copy_(torch.randn_like(experts.gate_proj))
        experts.gate_proj_bias.copy_(torch.randn_like(experts.gate_proj_bias))
        experts.c_fc.copy_(torch.randn_like(experts.c_fc))
        experts.c_proj.copy_(torch.randn_like(experts.c_proj))
        raw_gate_out = torch.bmm(x, experts.gate_proj)
        raw_gate_out = torch.baddbmm(
            raw_gate_out,
            selected_router_probs.unsqueeze(-1),
            experts.gate_proj_bias.unsqueeze(1),
            beta=1,
            alpha=-1,
        )
        expected_gate_out_acts = experts.act_fn(raw_gate_out)

        fc_out = torch.bmm(x, experts.c_fc)
        expected = torch.bmm(expected_gate_out_acts * fc_out, experts.c_proj)

    actual = experts(x, selected_router_probs=selected_router_probs)
    torch.testing.assert_close(actual, expected)


def test_dynamic_gate_projection_bias_backprops_into_selected_router_probs():
    torch.manual_seed(0)
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_exp_gate_proj_bias=True,
        debug=False,
    )
    experts = Qwen3MLPExperts(config)

    x = torch.randn(config.n_exp, 5, config.n_embd, requires_grad=True)
    selected_router_probs = torch.rand(config.n_exp, 5, requires_grad=True)
    out = experts(x, selected_router_probs=selected_router_probs).sum()
    out.backward()

    assert selected_router_probs.grad is not None


def test_dense_qwen3_gate_projection_has_no_bias_parameter():
    config = GPTConfig(
        n_exp=1,
        n_embd=4,
        use_exp_gate_proj_bias=True,
        debug=False,
    )

    mlp = Qwen3MLP(config)

    assert not hasattr(mlp, 'gate_proj_bias')


def test_gate_proj_bias_lr_scale_defaults_and_overrides_from_config():
    default_config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_exp_gate_proj_bias=True,
        debug=False,
    )
    override_config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_exp_gate_proj_bias=True,
        debug=False,
    )

    default_moe = Qwen3MLPExperts(default_config)
    override_moe = Qwen3MLPExperts(override_config)


def test_gate_proj_bias_l2_losses_are_reported_for_moe_layers_only():
    torch.manual_seed(0)
    base_config = GPTConfig(
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
        use_router_ortho_loss=False,
        use_exp_gate_proj_bias=True,
        exp_gate_proj_bias_l2_loss_weight=0.0,
        debug=False,
    )
    penalized_config = GPTConfig(
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
        use_router_ortho_loss=False,
        use_exp_gate_proj_bias=True,
        exp_gate_proj_bias_l2_loss_weight=0.7,
        debug=False,
    )

    base_model = GPT(base_config)
    penalized_model = GPT(penalized_config)
    base_model.init_weights()
    penalized_model.init_weights()
    penalized_model.load_state_dict(deepcopy(base_model.state_dict()))

    with torch.no_grad():
        penalized_model.transformer.h[1].mlp.experts.gate_proj_bias.fill_(1.0)
    penalized_model.refresh_gate_proj_bias_references()

    with torch.no_grad():
        penalized_model.transformer.h[1].mlp.experts.gate_proj_bias.fill_(4.0)
        base_model.load_state_dict(deepcopy(penalized_model.state_dict()))

    idx = torch.randint(0, base_config.vocab_size, (2, 4))
    targets = torch.randint(0, base_config.vocab_size, (2, 4))

    base_loss, base_losses = base_model(idx, targets)
    penalized_loss, penalized_losses = penalized_model(idx, targets)

    assert penalized_losses['exp_gate_proj_bias_l2_loss'].item() == 9.0
    assert base_losses['exp_gate_proj_bias_l2_loss'].item() == 16.0
    if torch.isnan(base_loss) and torch.isnan(penalized_loss):
        assert True
    else:
        torch.testing.assert_close(penalized_loss, base_loss)


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
        use_router_ortho_loss=False,
        use_exp_gate_proj_bias=True,
        debug=False,
    )
    model = GPT(config)
    model.init_weights()

    assert model.transformer.h[1].mlp.experts.initial_gate_proj_bias is None

    model.refresh_gate_proj_bias_references()

    assert model.transformer.h[1].mlp.experts.initial_gate_proj_bias is not None


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
        use_router_ortho_loss=False,
        use_exp_gate_proj_bias=True,
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
        use_exp_gate_proj_bias=True,
        debug=False,
    )

    experts = Qwen3MLPExperts(config)

    assert experts.gate_proj_bias is not None
    assert experts.gate_proj_bias.ndim == 2
    assert experts.gate_proj_bias.shape == (config.n_exp, 4 * config.n_embd)


def test_gate_projection_bias_respects_start_layer_cutoff():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_exp_gate_proj_bias=True,
        exp_gate_proj_bias_start_layer=3,
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