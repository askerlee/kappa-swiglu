import torch

from nanochat.configuration_nanomoe_gpt import GPTConfig
from nanochat.gpt import GPT, Qwen3MLPExperts
import math

def test_exp_gate_proj_rank_and_m_factorize_gate_projection_before_fc_gating():
    torch.manual_seed(0)
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        exp_gate_proj_rank=3,
        exp_gate_proj_m=3,
        use_experts_gate_output_loss=False,
        debug=False,
    )
    experts = Qwen3MLPExperts(config)

    x = torch.randn(config.n_exp, 5, config.n_embd)

    with torch.no_grad():
        experts.gate_proj_a.copy_(torch.randn_like(experts.gate_proj_a))
        experts.gate_proj_b.copy_(torch.randn_like(experts.gate_proj_b))
        experts.c_fc.copy_(torch.randn_like(experts.c_fc))
        experts.c_proj.copy_(torch.randn_like(experts.c_proj))
        raw_gate_hidden = torch.einsum('ech,ehrm->ecrm', x, experts.gate_proj_a)
        raw_gate_out = torch.einsum('ecrm,erim->ecim', raw_gate_hidden, experts.gate_proj_b)
        expected_gate_out_acts = experts.act_fn(raw_gate_out).mean(dim=-1) * math.sqrt(config.exp_gate_proj_m)
        actual_gate_out_acts = experts.act_fn(raw_gate_out).mean(dim=-1) * math.sqrt(config.exp_gate_proj_m)
        torch.testing.assert_close(actual_gate_out_acts, expected_gate_out_acts)

        fc_out = torch.bmm(x, experts.c_fc)
        expected = torch.bmm(expected_gate_out_acts * fc_out, experts.c_proj)

    actual = experts(x)
    torch.testing.assert_close(actual, expected)


def test_exp_gate_proj_m_applies_to_all_moe_layers_with_low_rank():
    config = GPTConfig(
        n_layer=6,
        moe_start_layer=2,
        stride=1,
        n_exp=2,
        n_embd=8,
        n_head=2,
        exp_gate_proj_rank=3,
        exp_gate_proj_m=3,
        use_experts_gate_output_loss=False,
        debug=False,
    )

    model = GPT(config)
    observed_ms = [
        layer.mlp.experts.gate_proj_m
        for layer in model.transformer.h
        if hasattr(layer.mlp, 'experts') and isinstance(layer.mlp.experts, Qwen3MLPExperts)
    ]
    observed_ranks = [
        layer.mlp.experts.gate_proj_a.shape[-2] if layer.mlp.experts.gate_proj_a.ndim == 4 else layer.mlp.experts.gate_proj_a.shape[-1]
        for layer in model.transformer.h
        if hasattr(layer.mlp, 'experts') and isinstance(layer.mlp.experts, Qwen3MLPExperts)
    ]

    assert observed_ms == [3, 3, 3, 3]
    assert observed_ranks == [3, 3, 3, 3]


def test_exp_gate_proj_rank_zero_keeps_dense_gate_projection():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        exp_gate_proj_rank=0,
        exp_gate_proj_m=3,
        use_experts_gate_output_loss=False,
        debug=False,
    )

    experts = Qwen3MLPExperts(config)

    assert hasattr(experts, 'gate_proj')
    assert experts.gate_proj.ndim == 4
    assert not hasattr(experts, 'gate_proj_a')
    assert not hasattr(experts, 'gate_proj_b')


def test_exp_gate_proj_logsumexp_aggregation_matches_expected_output():
    torch.manual_seed(0)
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        exp_gate_proj_rank=0,
        exp_gate_proj_m=3,
        exp_gate_proj_aggr_scheme='logsumexp',
        use_experts_gate_output_loss=False,
        debug=False,
    )
    experts = Qwen3MLPExperts(config)

    x = torch.randn(config.n_exp, 5, config.n_embd)

    with torch.no_grad():
        experts.gate_proj.copy_(torch.randn_like(experts.gate_proj))
        experts.c_fc.copy_(torch.randn_like(experts.c_fc))
        experts.c_proj.copy_(torch.randn_like(experts.c_proj))
        raw_gate_out = torch.einsum('ech,ehim->ecim', x, experts.gate_proj)
        expected_gate_out_acts = torch.logsumexp(experts.act_fn(raw_gate_out), dim=-1)
        fc_out = torch.bmm(x, experts.c_fc)
        expected = torch.bmm(expected_gate_out_acts * fc_out, experts.c_proj)

    actual = experts(x)
    torch.testing.assert_close(actual, expected)