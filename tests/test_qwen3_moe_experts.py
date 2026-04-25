import torch

from nanochat.configuration_nanomoe_gpt import GPTConfig
from nanochat.gpt import GPT, Qwen3MLP, Qwen3MLPExperts


def test_dense_gate_projection_is_applied_before_fc_gating():
    torch.manual_seed(0)
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_experts_gate_output_loss=False,
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


def test_gate_projection_bias_is_added_after_activation_when_enabled():
    torch.manual_seed(0)
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_exp_gate_proj_bias=True,
        use_experts_gate_output_loss=False,
        debug=False,
    )
    experts = Qwen3MLPExperts(config)

    x = torch.randn(config.n_exp, 5, config.n_embd)

    with torch.no_grad():
        experts.gate_proj.copy_(torch.randn_like(experts.gate_proj))
        experts.gate_proj_bias.copy_(torch.randn_like(experts.gate_proj_bias))
        experts.c_fc.copy_(torch.randn_like(experts.c_fc))
        experts.c_proj.copy_(torch.randn_like(experts.c_proj))
        raw_gate_out = torch.bmm(x, experts.gate_proj)
        expected_gate_out_acts = experts.act_fn(raw_gate_out) + experts.gate_proj_bias.unsqueeze(1)

        fc_out = torch.bmm(x, experts.c_fc)
        expected = torch.bmm(expected_gate_out_acts * fc_out, experts.c_proj)

    actual = experts(x)
    torch.testing.assert_close(actual, expected)


def test_dense_qwen3_gate_projection_bias_is_added_after_activation_when_enabled():
    torch.manual_seed(0)
    config = GPTConfig(
        n_exp=1,
        n_embd=4,
        use_exp_gate_proj_bias=True,
        use_dense_gate_proj_bias=True,
        debug=False,
    )
    mlp = Qwen3MLP(config)

    x = torch.randn(5, config.n_embd)

    with torch.no_grad():
        mlp.gate_proj.weight.copy_(torch.randn_like(mlp.gate_proj.weight))
        mlp.gate_proj_bias.copy_(torch.randn_like(mlp.gate_proj_bias))
        mlp.c_fc.weight.copy_(torch.randn_like(mlp.c_fc.weight))
        mlp.c_proj.weight.copy_(torch.randn_like(mlp.c_proj.weight))
        gate_out = mlp.act_fn(mlp.gate_proj(x)) + mlp.gate_proj_bias
        expected = mlp.c_proj(gate_out * mlp.c_fc(x))

    actual = mlp(x)
    torch.testing.assert_close(actual, expected)


def test_dense_qwen3_gate_projection_bias_has_expected_shape_when_enabled():
    config = GPTConfig(
        n_exp=1,
        n_embd=4,
        use_exp_gate_proj_bias=True,
        use_dense_gate_proj_bias=True,
        debug=False,
    )

    mlp = Qwen3MLP(config)

    assert mlp.gate_proj_bias is not None
    assert mlp.gate_proj_bias.ndim == 1
    assert mlp.gate_proj_bias.shape == (4 * config.n_embd,)


def test_dense_qwen3_gate_projection_bias_stays_disabled_without_dense_flag():
    config = GPTConfig(
        n_exp=1,
        n_embd=4,
        use_exp_gate_proj_bias=True,
        use_dense_gate_proj_bias=False,
        debug=False,
    )

    mlp = Qwen3MLP(config)

    assert mlp.gate_proj_bias is None


def test_dense_gate_projection_has_expected_shape():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        use_experts_gate_output_loss=False,
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
        use_experts_gate_output_loss=False,
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
        use_experts_gate_output_loss=False,
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
        use_experts_gate_output_loss=False,
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
        use_experts_gate_output_loss=False,
        debug=False,
    )

    experts = Qwen3MLPExperts(config)

    assert hasattr(experts, 'gate_proj')
    assert experts.gate_proj.ndim == 3
    assert not hasattr(experts, 'gate_proj_a')
    assert not hasattr(experts, 'gate_proj_b')