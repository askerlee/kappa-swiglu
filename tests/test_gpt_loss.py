import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.configuration_nanomoe_gpt import GPTConfig
from nanochat.gpt import _chunked_cross_entropy, _get_loss_chunk_tokens, SoftcapInPlace


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