import json
import math
from types import SimpleNamespace

import pytest
import torch

from nanochat.configuration_nanomoe_gpt import GPTConfig
from nanochat.gpt import MOELayer, Qwen3MLP, Qwen3MLPExperts
from scripts import check_kappa_latent_norms
from scripts.check_kappa_latent_norms import LatentNormProbe, ScalarStats


def test_zerosum_kappa_report_ignores_training_weights():
    config = GPTConfig(
        n_embd=4, n_exp=2, use_kappa_swiglu=True, kappa_input='top_logits',
        zerosum_kappa_alpha_weight=0.25, zerosum_kappa_bias_weight=0.5,
    )
    experts = Qwen3MLPExperts(config).eval()
    disabled_experts = Qwen3MLPExperts(config).eval()
    disabled_experts.set_kappa_swiglu_enabled(False)
    model = torch.nn.Sequential(experts, disabled_experts)
    model.config = config
    with torch.no_grad():
        experts.kappa_scale[0, 0].fill_(1)
        experts.kappa_scale[0, 1].fill_(-3)
        experts.kappa_bias.fill_(1)
        disabled_experts.kappa_scale.fill_(100)
        disabled_experts.kappa_bias.fill_(100)
    report = check_kappa_latent_norms.summarize_zerosum_kappa_loss(model)
    assert report == dict(
        expert_count=2, alpha_loss=10.0, bias_loss=2.0, total_loss=12.0,
        mean_alpha_loss=5.0, mean_bias_loss=1.0, mean_total_loss=6.0,
        mean_abs_alpha_mean=2.0, mean_abs_bias_mean=1.0,
    )


def test_expert_summary_uses_equal_weight_and_excludes_empty_experts():
    summaries = {}
    for expert_idx, values in enumerate(([1.0, 3.0], [10.0], [])):
        stats = ScalarStats()
        stats.observe(torch.tensor(values))
        summaries[f"layer.expert{expert_idx}/baseline/output_norm"] = stats.summary()
    aggregate = check_kappa_latent_norms.summarize_experts(["layer"], summaries)["layer/baseline/output_norm"]
    assert aggregate["expert_count"] == 3
    assert aggregate["observed_experts"] == 2
    assert aggregate["mean"]["mean"] == pytest.approx(6)
    assert aggregate["mean"]["std"] == pytest.approx(4)
    assert aggregate["std"]["mean"] == pytest.approx(0.5)
    assert aggregate["std"]["std"] == pytest.approx(0.5)
    assert aggregate["cv"]["mean"] == pytest.approx(0.25)
    assert aggregate["cv"]["std"] == pytest.approx(0.25)


def test_norm_cv_is_scale_invariant_and_detects_uniformization():
    original = ScalarStats()
    scaled = ScalarStats()
    uniform = ScalarStats()
    original.observe(torch.tensor([1.0, 2.0, 3.0]))
    scaled.observe(torch.tensor([10.0, 20.0, 30.0]))
    uniform.observe(torch.tensor([2.0, 2.0, 2.0]))
    assert original.summary()["cv"] == pytest.approx(math.sqrt(2 / 3) / 2)
    assert scaled.summary()["cv"] == pytest.approx(original.summary()["cv"])
    assert uniform.summary()["cv"] == 0


def test_streaming_merge_matches_combined_and_tracks_nonfinite():
    first = ScalarStats()
    second = ScalarStats()
    first.observe(torch.tensor([1.0, 2.0, float("nan")]))
    second.observe(torch.tensor([3.0, 4.0, float("inf")]))
    first.merge(second)
    summary = first.summary()
    assert summary["count"] == 4
    assert summary["nonfinite"] == 2
    assert summary["mean"] == pytest.approx(2.5)
    assert summary["std"] == pytest.approx(math.sqrt(1.25))
    empty = ScalarStats()
    assert empty.summary()["cv"] is None
    empty.observe(torch.zeros(3))
    assert empty.summary()["cv"] is None


class FakeExperts(Qwen3MLPExperts):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.use_kappa_swiglu = True
        self.kappa_swiglu_enabled = True

    def forward(self, inputs, valid_score_mask=None):
        return torch.nn.functional.normalize(inputs, dim=-1) if self.kappa_swiglu_enabled else inputs


def test_probe_pairs_same_inputs_excludes_padding_and_restores_toggle():
    experts = FakeExperts()
    model = torch.nn.Sequential(experts)
    probe = LatentNormProbe(per_expert=True)
    assert probe.install(model) == ["0"]
    inputs = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [999.0, 999.0]]])
    mask = torch.tensor([[True, True, False]])
    try:
        enabled_output = experts(inputs, valid_score_mask=mask)
        assert experts.kappa_swiglu_enabled
        torch.testing.assert_close(enabled_output[0, :2], torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
        experts.set_kappa_swiglu_enabled(False)
        probe.mode = "disabled"
        experts(inputs, valid_score_mask=mask)
        summary = probe.summaries()
        assert summary["0/enabled/input_norm"]["mean"] == 2
        assert summary["0/enabled/output_norm"]["count"] == 2
        assert summary["0/enabled/output_norm"]["cv"] == 0
        assert summary["0/local_disabled/output_norm"]["cv"] == pytest.approx(0.5)
        assert summary["0/local/output_cosine"]["mean"] == pytest.approx(1)
        assert summary["0.expert0/enabled/input_norm"]["count"] == 2
        assert summary["0/disabled/output_norm"]["mean"] == 2
    finally:
        probe.remove()
    assert not experts._forward_hooks


def test_probe_requires_expert_validity_mask():
    experts = FakeExperts()
    probe = LatentNormProbe()
    inputs = torch.ones(1, 2, 2)
    with pytest.raises(RuntimeError, match="validity mask"):
        probe.hook("experts", experts, (inputs,), {}, inputs)


@pytest.mark.parametrize("has_kappa", [False, True])
def test_real_dense_probe_matches_native_ablation(has_kappa):
    config = SimpleNamespace(
        n_embd=4, use_kappa_swiglu=has_kappa, constant_kappa_bias_dense_layers=True,
        kappa_input_constant=1.0,
    )
    dense = Qwen3MLP(config, layer_idx=0).eval()
    if has_kappa:
        with torch.no_grad():
            dense.kappa_bias.fill_(0.5)
    inputs = torch.randn(2, 3, 4)
    with torch.inference_mode():
        enabled = dense(inputs)
        dense.set_kappa_swiglu_enabled(False)
        disabled = dense(inputs)
        dense.set_kappa_swiglu_enabled(True)
        probe = LatentNormProbe()
        probe.install(torch.nn.Sequential(dense))
        try:
            actual = dense(inputs)
        finally:
            probe.remove()
    torch.testing.assert_close(actual, enabled)
    summary = probe.summaries()
    if has_kappa:
        assert summary["0/local_disabled/output_norm"]["mean"] == pytest.approx(disabled.float().norm(dim=-1).mean().item())
    else:
        assert "0/local_disabled/output_norm" not in summary
    assert summary["0/enabled/output_norm"]["mean"] == pytest.approx(enabled.float().norm(dim=-1).mean().item())
    assert dense.kappa_swiglu_enabled is has_kappa


@pytest.mark.parametrize("has_kappa", [False, True])
@pytest.mark.parametrize("mixed_precision", [False, True])
@pytest.mark.parametrize("normalize_output", [False, True])
def test_dense_swiglu_norms_match_explicit_branches(has_kappa, mixed_precision, normalize_output):
    dense = Qwen3MLP(SimpleNamespace(
        n_embd=4, use_kappa_swiglu=has_kappa, constant_kappa_bias_dense_layers=True,
        kappa_input_constant=1.0, swiglu_norm_output=normalize_output,
    ), layer_idx=0).eval()
    if has_kappa:
        with torch.no_grad():
            dense.kappa_bias.fill_(0.5)
    inputs = torch.randn(1, 3, 4)
    mask = torch.tensor([[True, True, False]])
    probe = LatentNormProbe(norm_location="swiglu")
    probe.mode = "enabled" if has_kappa else "baseline"
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=mixed_precision):
        gate = dense.gate_proj(inputs)
        up = dense.c_fc(inputs)
        pre = torch.cat((gate.float(), up.float()), dim=-1)
        disabled_gate = dense.act_fn(gate)
        if has_kappa:
            slope = math.exp(math.log(2.0) * math.tanh(0.5))
            activated_gate = gate * torch.sigmoid(gate * torch.tensor(slope, dtype=gate.dtype))
        else:
            activated_gate = disabled_gate
        if normalize_output:
            activated_gate = torch.nn.functional.rms_norm(activated_gate, (dense.intermediate_size,))
            disabled_gate = torch.nn.functional.rms_norm(disabled_gate, (dense.intermediate_size,))
        product = activated_gate * up
        disabled_product = disabled_gate * up
        expected = dense.c_proj(product)
        try:
            probe.install(torch.nn.Sequential(dense))
            actual = dense(inputs, valid_token_mask=mask)
        finally:
            probe.remove()
    torch.testing.assert_close(actual, expected)
    summaries = probe.summaries()
    mode = probe.mode
    assert summaries[f"0/{mode}/input_norm"]["mean"] == pytest.approx(pre.norm(dim=-1)[mask].mean().item())
    assert summaries[f"0/{mode}/output_norm"]["mean"] == pytest.approx(product.float().norm(dim=-1)[mask].mean().item())
    assert summaries[f"0/{mode}/output_norm"]["count"] == 2
    if has_kappa:
        assert summaries["0/local_disabled/output_norm"]["mean"] == pytest.approx(disabled_product.float().norm(dim=-1)[mask].mean().item())
    assert not dense.gate_proj._forward_hooks
    assert not dense.c_fc._forward_hooks
    assert not dense.c_proj._forward_pre_hooks
    assert not probe.activations


@pytest.mark.parametrize("kappa_input", ["router_probs", "top_logits"])
@pytest.mark.parametrize("has_kappa", [False, True])
@pytest.mark.parametrize("mixed_precision", [False, True])
@pytest.mark.parametrize("normalize_output", [False, True])
def test_expert_swiglu_norms_match_explicit_branches(kappa_input, has_kappa, mixed_precision, normalize_output):
    experts = Qwen3MLPExperts(GPTConfig(
        n_embd=4, n_exp=2, use_kappa_swiglu=has_kappa,
        kappa_input=kappa_input,
        swiglu_norm_output=normalize_output,
    ), layer_idx=0).eval()
    with torch.no_grad():
        for parameter in experts.parameters():
            parameter.uniform_(-0.3, 0.3)
    inputs = torch.randn(2, 3, 4)
    scores = torch.randn(2, 3)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    probe = LatentNormProbe(per_expert=True, norm_location="swiglu")
    probe.mode = "enabled" if has_kappa else "baseline"
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=mixed_precision):
        gate = torch.bmm(inputs, experts.gate_proj)
        up = torch.bmm(inputs, experts.c_fc)
        pre = torch.cat((gate.float(), up.float()), dim=-1)
        disabled_gate = experts.act_fn(gate)
        activated_gate = experts._apply_kappa_slope_scaled_activation_inference(gate, scores) if has_kappa else disabled_gate
        if normalize_output:
            activated_gate = torch.nn.functional.rms_norm(activated_gate, (experts.intermediate_size,))
            disabled_gate = torch.nn.functional.rms_norm(disabled_gate, (experts.intermediate_size,))
        product = activated_gate * up
        disabled_product = disabled_gate * up
        expected = torch.bmm(product, experts.c_proj)
        try:
            probe.install(torch.nn.Sequential(experts))
            actual = experts(inputs, selected_gate_scores=scores, valid_score_mask=mask)
        finally:
            probe.remove()
    torch.testing.assert_close(actual, expected)
    summaries = probe.summaries()
    mode = probe.mode
    assert summaries[f"0/{mode}/input_norm"]["mean"] == pytest.approx(pre.norm(dim=-1)[mask].mean().item())
    assert summaries[f"0/{mode}/output_norm"]["mean"] == pytest.approx(product.float().norm(dim=-1)[mask].mean().item())
    assert summaries[f"0/{mode}/output_norm"]["count"] == 3
    assert summaries[f"0.expert1/{mode}/output_norm"]["count"] == 1
    if has_kappa:
        assert summaries["0/local_disabled/output_norm"]["mean"] == pytest.approx(disabled_product.float().norm(dim=-1)[mask].mean().item())
    assert "_apply_gate_activation" not in experts.__dict__
    assert "_apply_kappa_slope_scaled_activation" not in experts.__dict__
    assert experts.kappa_swiglu_enabled is has_kappa
    assert not probe.activations


@pytest.mark.parametrize("kappa_input", ["router_probs", "top_logits"])
@pytest.mark.parametrize("has_kappa", [False, True])
def test_native_moe_dispatch_with_probe(kappa_input, has_kappa):
    layer = MOELayer(GPTConfig(
        n_embd=4, n_exp=2, moe_top_k=1, use_kappa_swiglu=has_kappa,
        kappa_input=kappa_input,
        use_aux_loss=False, use_router_z_loss=False, router_tie_noise_steps=0,
    ), layer_idx=0).eval()
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.uniform_(-0.3, 0.3)
    inputs = torch.randn(1, 8, 4)
    probe = LatentNormProbe(per_expert=True)
    with torch.inference_mode():
        expected = layer(inputs)
        assert probe.install(layer) == ["experts"]
        try:
            actual = layer(inputs)
            probe.mode = "disabled"
            layer.experts.set_kappa_swiglu_enabled(False)
            layer(inputs)
        finally:
            probe.remove()
    torch.testing.assert_close(actual, expected)
    summaries = probe.summaries()
    assert 0 < summaries["experts/enabled/input_norm"]["count"] <= 8
    if has_kappa:
        assert summaries["experts/local_disabled/output_norm"]["count"] == summaries["experts/enabled/output_norm"]["count"]
    else:
        assert "experts/local_disabled/output_norm" not in summaries
    assert all(stats["nonfinite"] == 0 for stats in summaries.values())


@pytest.mark.parametrize("has_kappa", [False, True])
@pytest.mark.parametrize("per_expert", [False, True])
def test_main_pairs_batches_reports_ntp_loss_and_writes_json(monkeypatch, tmp_path, capsys, has_kappa, per_expert):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = GPTConfig(zerosum_kappa_alpha_weight=0.25, zerosum_kappa_bias_weight=0.5)
            self.experts = FakeExperts()
            self.experts.use_kappa_swiglu = has_kappa
            self.experts.kappa_swiglu_enabled = has_kappa
            self.baseline_layer = FakeExperts()
            self.baseline_layer.use_kappa_swiglu = False
            self.baseline_layer.kappa_swiglu_enabled = False
            self.seen = []

        def forward(self, inputs, targets):
            self.seen.append(inputs.clone())
            mask = torch.ones(inputs.shape[:-1], dtype=torch.bool)
            output = self.experts(inputs, valid_score_mask=mask)
            self.baseline_layer(output, valid_score_mask=mask)
            return torch.tensor(9.0), {"ntp_loss": torch.tensor(2.0)}

    model = Model()
    monkeypatch.setattr(model.experts, "zerosum_kappa_expert_means", lambda: (torch.tensor([2.0, -2.0]), torch.ones(2)))
    output = tmp_path / "stats.json"
    args = SimpleNamespace(
        per_expert=per_expert, source="base", device_type="cpu", model_tag="test", step=None,
        eval_capacity=None, device_batch_size=1, eval_tokens=4, split="val", output=output, norm_location="mlp",
    )
    batches = [(torch.tensor([[[1.0, 0.0], [3.0, 0.0]]]), torch.zeros(1, 2, dtype=torch.long))] * 2
    monkeypatch.setattr(check_kappa_latent_norms, "parse_args", lambda: args)
    monkeypatch.setattr(check_kappa_latent_norms, "compute_init", lambda device_type: (False, 0, 0, 1, "cpu"))
    monkeypatch.setattr(check_kappa_latent_norms, "compute_cleanup", lambda: None)
    monkeypatch.setattr(check_kappa_latent_norms, "load_model", lambda *args, **kwargs: (
        model, None, {"step": 5, "model_config": {"sequence_len": 2}},
    ))
    monkeypatch.setattr(check_kappa_latent_norms, "build_loader", lambda *args: iter(batches))
    check_kappa_latent_norms.main()
    report = json.loads(output.read_text())
    assert report["tokens_per_mode"] == 4
    assert report["zerosum_kappa_loss"]["total_loss"] == (10.0 if has_kappa else 0.0)
    assert report["zerosum_kappa_loss"]["mean_total_loss"] == (5.0 if has_kappa else None)
    mode = "enabled" if has_kappa else "baseline"
    assert report["statistics"][f"model/{mode}/nll"]["mean"] == 2
    if per_expert:
        assert report["statistics"][f"experts.expert0/{mode}/output_norm"]["count"] == 4
        assert report["layer_kappa_enabled"]["experts.expert0"] is has_kappa
        assert report["layer_kappa_enabled"]["baseline_layer.expert0"] is False
    else:
        assert not any(".expert0" in key for key in report["statistics"])
    assert report["expert_summary"][f"experts/{mode}/output_norm"]["observed_experts"] == 1
    text = capsys.readouterr().out
    assert "Zero-sum kappa loss (unweighted):" in text
    assert ".expert0" not in text
    assert "mean +/- population std across observed experts" in text
    assert report["statistics"][f"baseline_layer/{mode}/output_norm"]["count"] == 4
    assert "baseline_layer/local_disabled/output_norm" not in report["statistics"]
    if has_kappa:
        assert report["modes"] == ["enabled", "disabled"]
        assert len(model.seen) == 4
        torch.testing.assert_close(model.seen[0], model.seen[1])
        torch.testing.assert_close(model.seen[2], model.seen[3])
        assert report["statistics"]["baseline_layer/enabled/input_norm"]["mean"] == 1
        assert report["statistics"]["baseline_layer/disabled/input_norm"]["mean"] == 2
    else:
        assert report["modes"] == ["baseline"]
        assert len(model.seen) == 2
        assert not any("local/" in key or "/enabled/" in key for key in report["statistics"])
    assert model.experts.kappa_swiglu_enabled is has_kappa
    assert not model.baseline_layer.kappa_swiglu_enabled
    assert not model.experts._forward_hooks
    assert not model.baseline_layer._forward_hooks


@pytest.mark.parametrize("has_kappa", [False, True])
def test_main_swiglu_records_boundary_definition(monkeypatch, tmp_path, capsys, has_kappa):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = GPTConfig()
            self.mlp = Qwen3MLP(SimpleNamespace(
                n_embd=4, use_kappa_swiglu=has_kappa, constant_kappa_bias_dense_layers=True,
                kappa_input_constant=1.0,
            ), layer_idx=0)
            if has_kappa:
                with torch.no_grad():
                    self.mlp.kappa_bias.fill_(0.5)

        def forward(self, inputs, targets):
            loss = self.mlp(inputs).float().square().mean()
            return loss, {"ntp_loss": loss}

    model = Model()
    output = tmp_path / "swiglu.json"
    args = SimpleNamespace(
        per_expert=False, source="base", device_type="cpu", model_tag="test", step=None,
        eval_capacity=None, device_batch_size=1, eval_tokens=2, split="val", output=output,
        norm_location="swiglu",
    )
    inputs = torch.randn(1, 2, 4)
    with torch.inference_mode():
        pre = torch.cat((model.mlp.gate_proj(inputs), model.mlp.c_fc(inputs)), dim=-1)
    monkeypatch.setattr(check_kappa_latent_norms, "parse_args", lambda: args)
    monkeypatch.setattr(check_kappa_latent_norms, "compute_init", lambda device_type: (False, 0, 0, 1, "cpu"))
    monkeypatch.setattr(check_kappa_latent_norms, "compute_cleanup", lambda: None)
    monkeypatch.setattr(check_kappa_latent_norms, "load_model", lambda *args, **kwargs: (
        model, None, {"step": 5, "model_config": {"sequence_len": 2}},
    ))
    monkeypatch.setattr(check_kappa_latent_norms, "build_loader", lambda *args: iter([(inputs, None)]))
    check_kappa_latent_norms.main()
    report = json.loads(output.read_text())
    mode = "enabled" if has_kappa else "baseline"
    assert report["norm_location"] == "swiglu"
    assert report["norm_definition"]["input"] == "concat(gate_proj(x), up_proj(x))"
    assert report["statistics"][f"mlp/{mode}/input_norm"]["mean"] == pytest.approx(pre.norm(dim=-1).mean().item())
    assert "SwiGLU norms: input=" in capsys.readouterr().out
    assert not model.mlp.gate_proj._forward_hooks


def test_swiglu_mode_rejects_bilinear_experts():
    experts = Qwen3MLPExperts(GPTConfig(n_embd=4, n_exp=2, bilinear_mlp_moe=True)).eval()
    probe = LatentNormProbe(norm_location="swiglu")
    with pytest.raises(ValueError, match="bilinear_mlp_moe"):
        probe.install(torch.nn.Sequential(experts))
    probe.remove()