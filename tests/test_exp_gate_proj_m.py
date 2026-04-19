import torch

from nanochat.configuration_nanomoe_gpt import GPTConfig
from nanochat.gpt import GPT, Qwen3MLPExperts


def test_exp_gate_proj_m_means_activated_gate_before_fc_gating():
    torch.manual_seed(0)
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        exp_gate_proj_m=3,
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
        expected_gate_out_acts = experts.act_fn(raw_gate_out).mean(dim=-1)
        actual_gate_out_acts = experts.act_fn(raw_gate_out).mean(dim=-1)
        torch.testing.assert_close(actual_gate_out_acts, expected_gate_out_acts)

        fc_out = torch.bmm(x, experts.c_fc)
        expected = torch.bmm(expected_gate_out_acts * fc_out, experts.c_proj)

    actual = experts(x)
    torch.testing.assert_close(actual, expected)


def test_exp_gate_proj_m_only_applies_to_highest_two_moe_layers():
    config = GPTConfig(
        n_layer=6,
        moe_start_layer=2,
        stride=1,
        n_exp=2,
        n_embd=8,
        n_head=2,
        exp_gate_proj_m=3,
        use_experts_gate_output_loss=False,
        debug=False,
    )

    model = GPT(config)
    observed_ms = [
        layer.mlp.experts.gate_proj.shape[-1]
        for layer in model.transformer.h
        if hasattr(layer.mlp, 'experts') and isinstance(layer.mlp.experts, Qwen3MLPExperts)
    ]

    assert observed_ms == [1, 1, 3, 3]