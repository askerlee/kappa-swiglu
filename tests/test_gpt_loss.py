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