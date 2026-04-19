import torch

from nanochat.configuration_nanomoe_gpt import GPTConfig
from nanochat.gpt import Qwen3MLPExperts


def test_qwen3_gate_proj_m_sums_before_fc_gating():
    config = GPTConfig(
        n_exp=2,
        n_embd=4,
        qwen3_gate_proj_m=3,
        use_experts_gate_output_loss=False,
        debug=False,
    )
    experts = Qwen3MLPExperts(config)

    x = torch.randn(config.n_exp, 5, config.n_embd)

    with torch.no_grad():
        raw_gate_out = torch.einsum('ech,ehim->ecim', x, experts.gate_proj)
        expected_gate_out = raw_gate_out.sum(dim=-1)
        actual_gate_out = raw_gate_out.sum(dim=-1)
        torch.testing.assert_close(actual_gate_out, expected_gate_out)

        fc_out = torch.bmm(x, experts.c_fc)
        expected = torch.bmm(experts.act_fn(expected_gate_out) * fc_out, experts.c_proj)

    actual = experts(x)
    torch.testing.assert_close(actual, expected)