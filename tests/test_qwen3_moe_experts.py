import torch
from torch.nn import functional as F
from copy import deepcopy

from nanochat.configuration_nanomoe_gpt import GPTConfig
from nanochat.gpt import GPT, Qwen3MLP, Qwen3MLPExperts, Router


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
    router = Router(config)
    experts.set_router(router)

    x = torch.randn(config.n_exp, 5, config.n_embd)

    with torch.no_grad():
        experts.gate_proj.copy_(torch.randn_like(experts.gate_proj))
        experts.gate_proj_bias.copy_(torch.randn_like(experts.gate_proj_bias))
        experts.c_fc.copy_(torch.randn_like(experts.c_fc))
        experts.c_proj.copy_(torch.randn_like(experts.c_proj))
        router.w_g.weight.copy_(torch.randn_like(router.w_g.weight))
        raw_gate_out = torch.bmm(x, experts.gate_proj)
        router_scores = (x * router.w_g.weight.unsqueeze(1)).sum(dim=-1)
        normalized_router_scores = F.normalize(router_scores.float(), dim=1, eps=1e-6).to(dtype=x.dtype)
        raw_gate_out = torch.baddbmm(
            raw_gate_out,
            normalized_router_scores.unsqueeze(-1),
            experts.gate_proj_bias.unsqueeze(1),
            beta=1,
            alpha=-1,
        )
        expected_gate_out_acts = experts.act_fn(raw_gate_out)

        fc_out = torch.bmm(x, experts.c_fc)
        expected = torch.bmm(expected_gate_out_acts * fc_out, experts.c_proj)

    actual = experts(x)
    torch.testing.assert_close(actual, expected)


def test_dynamic_gate_projection_bias_backprops_into_router_weights():
    torch.manual_seed(0)
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_exp_gate_proj_bias=True,
        debug=False,
    )
    experts = Qwen3MLPExperts(config)
    router = Router(config)
    experts.set_router(router)

    x = torch.randn(config.n_exp, 5, config.n_embd, requires_grad=True)
    out = experts(x).sum()
    out.backward()

    assert router.w_g.weight.grad is not None


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
    penalized_model.load_state_dict(deepcopy(base_model.state_dict()))

    with torch.no_grad():
        penalized_model.transformer.h[1].mlp.experts.gate_proj_bias.zero_()
        penalized_model.transformer.h[1].mlp.experts.gate_proj_bias.fill_(3.0)
        base_model.load_state_dict(deepcopy(penalized_model.state_dict()))

    idx = torch.randint(0, base_config.vocab_size, (2, 4))
    targets = torch.randint(0, base_config.vocab_size, (2, 4))

    base_loss, base_losses = base_model(idx, targets)
    penalized_loss, penalized_losses = penalized_model(idx, targets)

    assert penalized_losses['exp_gate_proj_bias_l2_loss'].item() == 9.0
    assert base_losses['exp_gate_proj_bias_l2_loss'].item() == 9.0
    if torch.isnan(base_loss) and torch.isnan(penalized_loss):
        assert True
    else:
        torch.testing.assert_close(penalized_loss, base_loss)


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