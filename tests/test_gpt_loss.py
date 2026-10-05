import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.configuration_nanomoe_gpt import GPTConfig
from nanochat.gpt import GPT, _UTLossAccum, _chunked_cross_entropy, _get_loss_chunk_tokens, SoftcapInPlace


def test_separate_base_sft_kappa_uses_same_slot_for_all_ut_passes():
    config = GPTConfig(
        n_layer=2, n_head=2, n_embd=16, vocab_size=32, sequence_len=8,
        n_exp=2, moe_start_layer=1, total_ut_steps=3,
        use_kappa_swiglu=True, constant_kappa_bias_dense_layers=True,
        separate_base_sft_kappa=True,
    )
    model = GPT(config)
    model.init_weights()
    dense = model.transformer.h[0].mlp
    experts = model.transformer.h[1].mlp.experts
    inputs = torch.randn(2, 2, 16)
    for module in (dense, experts):
        assert module.kappa_bias.shape[0] == 2
        assert not any('ema_rms' in name for name in module.state_dict())
        with torch.no_grad():
            module.kappa_bias[0].fill_(-0.4)
            module.kappa_bias[1].fill_(0.4)
            if module is dense:
                module.c_proj.weight.fill_(0.1)
            else:
                module.c_proj.fill_(0.1)
    assert experts.kappa_scale.shape[0] == 2
    for is_sft in (False, True):
        model.set_kappa_training_phase(is_sft)
        model.zero_grad(set_to_none=True)
        for module in (dense, experts):
            outputs = []
            for current_ut in range(3):
                accum = _UTLossAccum(inputs, 2, 2)
                if module is dense:
                    output = module(inputs, loss_accum=accum, current_ut=current_ut)
                else:
                    output = module(inputs, torch.ones(2, 2), loss_accum=accum, current_ut=current_ut)
                outputs.append(output)
            for output in outputs[1:]:
                torch.testing.assert_close(output, outputs[0])
            outputs[0].sum().backward()
            inactive_slot = 1 - int(is_sft)
            assert module.kappa_bias.grad[inactive_slot].count_nonzero() == 0
            assert module.kappa_bias.grad[int(is_sft)].count_nonzero() > 0


@pytest.mark.parametrize('granularity', ['per-gate', 'per-expert', 'per-layer', 'global'])
def test_separate_kappa_eval_cache_and_reference_slots(granularity):
    config = GPTConfig(
        n_layer=2, n_head=2, n_embd=16, vocab_size=32, sequence_len=8,
        n_exp=2, moe_start_layer=1, total_ut_steps=3,
        use_kappa_swiglu=True, constant_kappa_bias_dense_layers=True,
        separate_base_sft_kappa=True, global_kappa_param_granularity=granularity,
        refresh_kappa_param_references=True,
    )
    model = GPT(config)
    model.init_weights()
    dense = model.transformer.h[0].mlp
    experts = model.transformer.h[1].mlp.experts
    for module in (dense, experts):
        bias = module._get_kappa_bias_parameter()
        assert bias.shape[0] == 2
        with torch.no_grad():
            bias[0].fill_(-0.5)
            bias[1].fill_(0.5)
        assert not any('ema_rms' in name for name in module.state_dict())
    model.refresh_kappa_param_references()
    assert experts.initial_kappa_bias.shape[0] == 2
    assert experts.initial_kappa_scale.shape[0] == 2
    inputs = torch.ones(2, 2, 64)
    model.eval()
    outputs = []
    for is_sft in (False, True, False):
        model.set_kappa_training_phase(is_sft)
        with torch.no_grad():
            dense_output = dense._materialize_kappa_slope_scales_for_eval(torch.float32, inputs.device, int(is_sft))
            expert_output = experts._apply_kappa_slope_scaled_activation_inference(
                inputs, torch.ones(2, 2), kappa_slot=int(is_sft),
            )
        outputs.append((dense_output.clone(), expert_output.clone()))
    for module_index in (0, 1):
        torch.testing.assert_close(outputs[0][module_index], outputs[2][module_index])
        assert not torch.allclose(outputs[0][module_index], outputs[1][module_index])
    restored = GPT(GPTConfig(**vars(config)))
    restored.init_weights()
    restored.load_state_dict(model.state_dict())
    assert restored.transformer.h[1].mlp.experts._get_kappa_bias_parameter().shape[0] == 2


def test_default_kappa_layout_remains_per_ut_pass():
    config = GPTConfig(
        n_layer=2, n_head=2, n_embd=16, n_exp=2, moe_start_layer=1,
        total_ut_steps=3, use_kappa_swiglu=True, constant_kappa_bias_dense_layers=True,
    )
    model = GPT(config)
    assert model.transformer.h[0].mlp.kappa_bias.shape[0] == 3
    assert model.transformer.h[1].mlp.experts.kappa_bias.shape[0] == 3


@pytest.mark.parametrize('checkpointed', [False, True])
def test_task_kappa_full_model_training_preserves_other_task(checkpointed):
    config = GPTConfig(
        n_layer=2, n_head=2, n_embd=32, vocab_size=32, sequence_len=8,
        n_exp=2, moe_start_layer=1, total_ut_steps=3,
        use_kappa_swiglu=True, constant_kappa_bias_dense_layers=True,
        separate_base_sft_kappa=True, activation_checkpointing=checkpointed,
    )
    model = GPT(config)
    model.init_weights()
    with torch.no_grad():
        model.transformer.h[0].mlp.c_proj.weight.fill_(0.1)
        model.transformer.h[1].mlp.experts.c_proj.fill_(0.1)
    optimizer = model.setup_optimizer(embedding_lr=0.001, matrix_lr=0.001)
    group = next(group for group in optimizer.param_groups if group.get('name') == 'kappa_params')
    group['lr'] = 0.001
    inputs = torch.randint(0, 32, (2, 4))
    for is_sft in (False, True):
        slot = int(is_sft)
        model.set_kappa_training_phase(is_sft)
        group['active_kappa_slot'] = slot
        before = [param[1 - slot].detach().clone() for param in group['params']]
        loss, _ = model(inputs, inputs)
        assert loss.isfinite()
        loss.backward()
        for param in group['params']:
            assert param.grad[1 - slot].count_nonzero() == 0
        optimizer.step()
        model.zero_grad(set_to_none=True)
        for param, inactive_before in zip(group['params'], before):
            torch.testing.assert_close(param[1 - slot], inactive_before, rtol=0, atol=0)
    for param in group['params']:
        assert optimizer.state[param]['slot_steps'] == [1, 1]


def _full_softcapped_cross_entropy(hidden_states, targets, lm_head, vocab_size, softcap, reduction):
    logits = lm_head(hidden_states.reshape(-1, hidden_states.size(-1)))
    logits = logits[:, :vocab_size]
    logits = SoftcapInPlace.apply(logits, softcap)
    return F.cross_entropy(logits, targets.reshape(-1), ignore_index=-1, reduction=reduction)


def test_chunked_cross_entropy_matches_full_mean_loss():
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=37)
    lm_head = nn.Linear(16, 40, bias=False)
    hidden_states = torch.randn(3, 11, 16)
    targets = torch.randint(0, config.vocab_size, (3, 11))
    targets[0, 0] = -1
    softcap = 15.0

    full_loss = _full_softcapped_cross_entropy(hidden_states, targets, lm_head, config.vocab_size, softcap, 'mean')
    chunked_loss = _chunked_cross_entropy(
        hidden_states,
        targets,
        lm_head,
        config.vocab_size,
        softcap,
        'mean',
        chunk_tokens=7,
    )

    assert torch.allclose(chunked_loss, full_loss)


def test_chunked_cross_entropy_matches_full_mean_loss_gradients():
    torch.manual_seed(2)
    config = GPTConfig(vocab_size=31)
    full_lm_head = nn.Linear(10, 32, bias=False)
    chunked_lm_head = nn.Linear(10, 32, bias=False)
    chunked_lm_head.load_state_dict(full_lm_head.state_dict())
    full_hidden = torch.randn(4, 7, 10, requires_grad=True)
    chunked_hidden = full_hidden.detach().clone().requires_grad_(True)
    targets = torch.randint(0, config.vocab_size, (4, 7))
    targets[0, 2] = -1
    softcap = 15.0

    full_loss = _full_softcapped_cross_entropy(
        full_hidden,
        targets,
        full_lm_head,
        config.vocab_size,
        softcap,
        'mean',
    )
    chunked_loss = _chunked_cross_entropy(
        chunked_hidden,
        targets,
        chunked_lm_head,
        config.vocab_size,
        softcap,
        'mean',
        chunk_tokens=6,
    )

    full_loss.backward()
    chunked_loss.backward()

    assert torch.allclose(chunked_loss, full_loss)
    assert torch.allclose(chunked_hidden.grad, full_hidden.grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(chunked_lm_head.weight.grad, full_lm_head.weight.grad, atol=1e-6, rtol=1e-5)


def test_recompute_chunked_cross_entropy_matches_full_mean_loss_gradients():
    torch.manual_seed(4)
    config = GPTConfig(vocab_size=31)
    full_lm_head = nn.Linear(10, 32, bias=False)
    chunked_lm_head = nn.Linear(10, 32, bias=False)
    chunked_lm_head.load_state_dict(full_lm_head.state_dict())
    full_hidden = torch.randn(4, 7, 10, requires_grad=True)
    chunked_hidden = full_hidden.detach().clone().requires_grad_(True)
    targets = torch.randint(0, config.vocab_size, (4, 7))
    targets[0, 2] = -1
    softcap = 15.0

    full_loss = _full_softcapped_cross_entropy(
        full_hidden,
        targets,
        full_lm_head,
        config.vocab_size,
        softcap,
        'mean',
    )
    chunked_loss = _chunked_cross_entropy(
        chunked_hidden,
        targets,
        chunked_lm_head,
        config.vocab_size,
        softcap,
        'mean',
        chunk_tokens=6,
        recompute_backward=True,
    )

    full_loss.backward()
    chunked_loss.backward()

    assert torch.allclose(chunked_loss, full_loss)
    assert torch.allclose(chunked_hidden.grad, full_hidden.grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(chunked_lm_head.weight.grad, full_lm_head.weight.grad, atol=1e-6, rtol=1e-5)


def test_recompute_chunked_cross_entropy_handles_all_ignored_targets():
    torch.manual_seed(5)
    lm_head = nn.Linear(10, 32, bias=False)
    hidden_states = torch.randn(2, 7, 10, requires_grad=True)
    targets = torch.full((2, 7), -1)

    loss = _chunked_cross_entropy(
        hidden_states,
        targets,
        lm_head,
        vocab_size=31,
        softcap=15.0,
        loss_reduction='mean',
        chunk_tokens=6,
        recompute_backward=True,
    )
    loss.backward()

    torch.testing.assert_close(loss, torch.tensor(0.0))
    torch.testing.assert_close(hidden_states.grad, torch.zeros_like(hidden_states))
    torch.testing.assert_close(lm_head.weight.grad, torch.zeros_like(lm_head.weight))


def test_chunked_cross_entropy_supports_bf16_hidden_with_fp32_lm_head():
    torch.manual_seed(3)
    config = GPTConfig(vocab_size=23)
    lm_head = nn.Linear(8, 24, bias=False)
    hidden_states = torch.randn(2, 5, 8, dtype=torch.bfloat16, requires_grad=True)
    targets = torch.randint(0, config.vocab_size, (2, 5))

    loss = _chunked_cross_entropy(
        hidden_states,
        targets,
        lm_head,
        config.vocab_size,
        softcap=15.0,
        loss_reduction='mean',
        chunk_tokens=3,
        recompute_backward=True,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert hidden_states.grad is not None
    assert lm_head.weight.grad is not None


def test_chunked_cross_entropy_matches_full_none_loss():
    torch.manual_seed(1)
    config = GPTConfig(vocab_size=29)
    lm_head = nn.Linear(12, 32, bias=False)
    hidden_states = torch.randn(2, 9, 12)
    targets = torch.randint(0, config.vocab_size, (2, 9))
    targets[1, 3] = -1
    softcap = 15.0

    full_loss = _full_softcapped_cross_entropy(hidden_states, targets, lm_head, config.vocab_size, softcap, 'none')
    chunked_loss = _chunked_cross_entropy(
        hidden_states,
        targets,
        lm_head,
        config.vocab_size,
        softcap,
        'none',
        chunk_tokens=5,
    )

    assert torch.allclose(chunked_loss, full_loss)


def test_get_loss_chunk_tokens_uses_configured_cap():
    config = GPTConfig(vocab_size=50304, loss_chunk_tokens=256)

    assert _get_loss_chunk_tokens(config, total_tokens=1024) == 256
    assert _get_loss_chunk_tokens(config, total_tokens=128) == 128


def test_get_loss_chunk_tokens_defaults_to_32_mib_logits():
    config = GPTConfig(vocab_size=32768, loss_chunk_tokens=None)

    assert _get_loss_chunk_tokens(config, total_tokens=1024) == 512