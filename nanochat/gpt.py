"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

import math
from contextlib import nullcontext

import torch
import torch._dynamo
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F

try:
    from .manager import MANAGER
except ImportError:
    from manager import MANAGER
from transformers.activations import SiLUActivation
from nanochat.common import get_dist_info, print0
from nanochat.optim import AuroraAdamW, DistAuroraAdamW, MuonAdamW, DistMuonAdamW
# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn

def scale_grad(x, alpha):
    if torch.is_tensor(alpha) and alpha.requires_grad:
        return IdentityWithScaledGrad.apply(x, alpha)
    return x.detach() + alpha * (x - x.detach())


def _sum_to_shape(x, shape):
    while x.ndim > len(shape):
        x = x.sum(dim=0)
    for dim, size in enumerate(shape):
        if size == 1 and x.shape[dim] != 1:
            x = x.sum(dim=dim, keepdim=True)
    return x


class IdentityWithScaledGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, alpha):
        ctx.save_for_backward(input_, alpha)
        return input_

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        input_, alpha = ctx.saved_tensors

        if ctx.needs_input_grad[0]:
            alpha_for_grad_input = alpha
            if alpha_for_grad_input.dtype != grad_output.dtype:
                alpha_for_grad_input = alpha_for_grad_input.to(dtype=grad_output.dtype)
            grad_input = grad_output * alpha_for_grad_input
            if grad_input.dtype != input_.dtype:
                grad_input = grad_input.to(dtype=input_.dtype)
        else:
            grad_input = None

        if ctx.needs_input_grad[1]:
            input_for_grad_alpha = input_
            if input_for_grad_alpha.dtype != grad_output.dtype:
                input_for_grad_alpha = input_for_grad_alpha.to(dtype=grad_output.dtype)
            grad_alpha = grad_output * input_for_grad_alpha
            grad_alpha = _sum_to_shape(grad_alpha, alpha.shape)
            if grad_alpha.dtype != alpha.dtype:
                grad_alpha = grad_alpha.to(dtype=alpha.dtype)
        else:
            grad_alpha = None

        return grad_input, grad_alpha

def _mean_extreme_percentile(values: torch.Tensor, fraction: float, largest: bool) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(())
    k = max(1, math.ceil(values.numel() * fraction))
    return values.topk(k, largest=largest).values.mean()


def _mean_extreme_percentile_per_row(
    values: torch.Tensor,
    active_mask: torch.Tensor,
    fraction: float,
    largest: bool,
) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("values must be 2D")
    if active_mask.shape != values.shape:
        raise ValueError("active_mask must have the same shape as values")

    row_means = []
    for row_values, row_mask in zip(values, active_mask):
        active_values = row_values[row_mask]
        if active_values.numel() == 0:
            continue
        row_means.append(_mean_extreme_percentile(active_values, fraction=fraction, largest=largest))

    if not row_means:
        return values.new_zeros(())
    return torch.stack(row_means).mean()


def _mean_extreme_outer_product_percentile_per_row(
    left_values: torch.Tensor,
    right_values: torch.Tensor,
    left_active_mask: torch.Tensor,
    fraction: float,
    largest: bool,
) -> torch.Tensor:
    if left_values.ndim != 2:
        raise ValueError("left_values must be 2D")
    if right_values.ndim != 2:
        raise ValueError("right_values must be 2D")
    if left_active_mask.shape != left_values.shape:
        raise ValueError("left_active_mask must have the same shape as left_values")
    if left_values.size(0) != right_values.size(0):
        raise ValueError("left_values and right_values must have the same number of rows")

    row_means = []
    for row_left_values, row_right_values, row_left_mask in zip(left_values, right_values, left_active_mask):
        active_left_values = row_left_values[row_left_mask]
        if active_left_values.numel() == 0 or row_right_values.numel() == 0:
            continue
        row_products = active_left_values.unsqueeze(-1) * row_right_values.unsqueeze(0)
        row_means.append(
            _mean_extreme_percentile(row_products.reshape(-1), fraction=fraction, largest=largest)
        )

    if not row_means:
        return left_values.new_zeros(())
    return torch.stack(row_means).mean()


class SoftcapInPlace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, softcap):
        softcap = float(softcap)
        input_.div_(softcap)
        input_.tanh_()
        input_.mul_(softcap)
        ctx.softcap = softcap
        ctx.mark_dirty(input_)
        ctx.save_for_backward(input_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        (output,) = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = grad_output
            output.mul_(output)
            output.div_(ctx.softcap * ctx.softcap)
            output.neg_().add_(1.0)
            grad_input.mul_(output)
        else:
            grad_input = None
        return grad_input, None


class ChunkedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def _softcap_logits_(logits, softcap):
        logits.div_(softcap)
        logits.tanh_()
        logits.mul_(softcap)
        return logits

    @staticmethod
    def forward(ctx, hidden_flat, targets_flat, weight, vocab_size, softcap, loss_reduction, chunk_tokens):
        vocab_size = int(vocab_size)
        softcap = float(softcap)
        chunk_tokens = int(chunk_tokens)
        targets_flat = targets_flat.reshape(-1)
        weight_vocab = weight[:vocab_size]
        weight_vocab_for_mm = weight_vocab.to(dtype=hidden_flat.dtype)
        valid_mask = targets_flat.ne(-1)
        valid_tokens = valid_mask.sum()

        if loss_reduction == 'none':
            losses = hidden_flat.new_zeros(targets_flat.shape, dtype=torch.float32)
        else:
            loss_sum = hidden_flat.new_zeros((), dtype=torch.float32)

        for chunk_start in range(0, hidden_flat.size(0), chunk_tokens):
            chunk_end = min(chunk_start + chunk_tokens, hidden_flat.size(0))
            chunk_hidden = hidden_flat[chunk_start:chunk_end]
            chunk_targets = targets_flat[chunk_start:chunk_end]
            chunk_valid_mask = chunk_targets.ne(-1)
            if not bool(chunk_valid_mask.any().item()):
                continue

            logits = F.linear(chunk_hidden, weight_vocab_for_mm)
            logits = ChunkedLinearCrossEntropy._softcap_logits_(logits, softcap)
            logsumexp = torch.logsumexp(logits, dim=-1)
            safe_targets = chunk_targets.clamp_min(0)
            target_logits = logits.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
            chunk_losses = (logsumexp - target_logits).float()
            chunk_losses = chunk_losses * chunk_valid_mask.to(dtype=chunk_losses.dtype)

            if loss_reduction == 'none':
                losses[chunk_start:chunk_end] = chunk_losses
            else:
                loss_sum = loss_sum + chunk_losses.sum()

        ctx.save_for_backward(hidden_flat, targets_flat, weight, valid_tokens)
        ctx.vocab_size = vocab_size
        ctx.softcap = softcap
        ctx.loss_reduction = loss_reduction
        ctx.chunk_tokens = chunk_tokens

        if loss_reduction == 'none':
            return losses
        if loss_reduction == 'sum':
            return loss_sum
        mean_loss = loss_sum / valid_tokens.clamp_min(1).to(dtype=loss_sum.dtype)
        zero_loss = hidden_flat.sum(dtype=loss_sum.dtype) * 0.0
        return torch.where(valid_tokens > 0, mean_loss, zero_loss)

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        hidden_flat, targets_flat, weight, valid_tokens = ctx.saved_tensors
        vocab_size = ctx.vocab_size
        softcap = ctx.softcap
        loss_reduction = ctx.loss_reduction
        chunk_tokens = ctx.chunk_tokens
        weight_vocab = weight[:vocab_size]
        weight_vocab_for_mm = weight_vocab.to(dtype=hidden_flat.dtype)

        grad_hidden = torch.zeros_like(hidden_flat) if ctx.needs_input_grad[0] else None
        grad_weight = torch.zeros_like(weight) if ctx.needs_input_grad[2] else None

        if not bool((valid_tokens > 0).item()):
            return grad_hidden, None, grad_weight, None, None, None, None

        if loss_reduction == 'mean':
            base_scale = grad_output / valid_tokens.to(device=grad_output.device, dtype=grad_output.dtype)
        elif loss_reduction == 'sum':
            base_scale = grad_output
        else:
            base_scale = None

        for chunk_start in range(0, hidden_flat.size(0), chunk_tokens):
            chunk_end = min(chunk_start + chunk_tokens, hidden_flat.size(0))
            chunk_hidden = hidden_flat[chunk_start:chunk_end]
            chunk_targets = targets_flat[chunk_start:chunk_end]
            chunk_valid_mask = chunk_targets.ne(-1)
            if not bool(chunk_valid_mask.any().item()):
                continue

            logits_raw = F.linear(chunk_hidden, weight_vocab_for_mm)
            logits = ChunkedLinearCrossEntropy._softcap_logits_(logits_raw, softcap)
            grad_logits = torch.softmax(logits, dim=-1)
            safe_targets = chunk_targets.clamp_min(0)
            grad_logits.scatter_add_(
                1,
                safe_targets.unsqueeze(1),
                -chunk_valid_mask.to(dtype=grad_logits.dtype).unsqueeze(1),
            )
            grad_logits.mul_(chunk_valid_mask.to(dtype=grad_logits.dtype).unsqueeze(1))

            if loss_reduction == 'none':
                chunk_scale = grad_output[chunk_start:chunk_end].to(dtype=grad_logits.dtype).unsqueeze(1)
            else:
                chunk_scale = base_scale.to(dtype=grad_logits.dtype)
            grad_logits.mul_(chunk_scale)

            softcap_grad = 1.0 - logits.square().div(softcap * softcap)
            grad_logits.mul_(softcap_grad)

            if grad_hidden is not None:
                grad_hidden[chunk_start:chunk_end] = torch.mm(
                    grad_logits.to(dtype=weight_vocab_for_mm.dtype),
                    weight_vocab_for_mm,
                ).to(dtype=grad_hidden.dtype)
            if grad_weight is not None:
                grad_weight[:vocab_size].add_(torch.mm(
                    grad_logits.transpose(0, 1).to(dtype=chunk_hidden.dtype),
                    chunk_hidden,
                ).to(dtype=grad_weight.dtype))

        return grad_hidden, None, grad_weight, None, None, None, None


def _get_loss_chunk_tokens(config, total_tokens: int) -> int:
    """Bound lm_head/loss work to a token chunk that keeps peak vocab activations manageable."""
    configured = getattr(config, "loss_chunk_tokens", None)
    if configured is not None:
        chunk_tokens = int(configured)
        if chunk_tokens <= 0:
            raise ValueError("loss_chunk_tokens must be a positive integer")
        return min(total_tokens, chunk_tokens)

    max_logit_elements = 64 * 1024 * 1024
    # vocab_size: 50304. If loss_chunk_tokens is None, then
    # chunk_tokens = 64 * 1024 * 1024 // 50304 = 1334.
    chunk_tokens = max(1, max_logit_elements // int(config.vocab_size))
    return min(total_tokens, chunk_tokens)


def _chunked_cross_entropy(
    hidden_states: torch.Tensor,
    targets: torch.Tensor,
    lm_head: nn.Module,
    vocab_size: int,
    softcap: float,
    loss_reduction: str,
    chunk_tokens: int,
    recompute_backward: bool = False,
) -> torch.Tensor:
    """Compute next-token loss without materializing full training logits at once."""
    hidden_flat = hidden_states.reshape(-1, hidden_states.size(-1))
    targets_flat = targets.reshape(-1)
    valid_tokens = targets_flat.ne(-1).sum()

    if loss_reduction not in {'mean', 'sum', 'none'}:
        raise ValueError(f"Unsupported loss_reduction: {loss_reduction}")

    if recompute_backward:
        return ChunkedLinearCrossEntropy.apply(
            hidden_flat,
            targets_flat,
            lm_head.weight,
            vocab_size,
            softcap,
            loss_reduction,
            chunk_tokens,
        )

    if chunk_tokens >= hidden_flat.size(0):
        logits = lm_head(hidden_flat)
        logits = logits[:, :vocab_size]
        logits = SoftcapInPlace.apply(logits, softcap)
        if loss_reduction == 'mean':
            loss_sum = F.cross_entropy(logits, targets_flat, ignore_index=-1, reduction='sum')
            mean_loss = loss_sum / valid_tokens.clamp_min(1)
            zero_loss = hidden_flat.sum(dtype=loss_sum.dtype) * 0.0
            return torch.where(valid_tokens > 0, mean_loss, zero_loss)
        return F.cross_entropy(logits, targets_flat, ignore_index=-1, reduction=loss_reduction)

    if loss_reduction == 'none':
        chunk_losses = []
        for chunk_start in range(0, hidden_flat.size(0), chunk_tokens):
            chunk_end = min(chunk_start + chunk_tokens, hidden_flat.size(0))
            chunk_logits = lm_head(hidden_flat[chunk_start:chunk_end])
            chunk_logits = chunk_logits[:, :vocab_size]
            chunk_logits = SoftcapInPlace.apply(chunk_logits, softcap)
            chunk_losses.append(F.cross_entropy(
                chunk_logits,
                targets_flat[chunk_start:chunk_end],
                ignore_index=-1,
                reduction='none',
            ))
        return torch.cat(chunk_losses, dim=0)

    loss_sum = None
    for chunk_start in range(0, hidden_flat.size(0), chunk_tokens):
        chunk_end = min(chunk_start + chunk_tokens, hidden_flat.size(0))
        chunk_logits = lm_head(hidden_flat[chunk_start:chunk_end])
        chunk_logits = chunk_logits[:, :vocab_size]
        chunk_logits = SoftcapInPlace.apply(chunk_logits, softcap)
        chunk_loss = F.cross_entropy(
            chunk_logits,
            targets_flat[chunk_start:chunk_end],
            ignore_index=-1,
            reduction='sum',
        )
        loss_sum = chunk_loss if loss_sum is None else loss_sum + chunk_loss

    if loss_reduction == 'sum':
        return loss_sum

    mean_loss = loss_sum / valid_tokens.clamp_min(1)
    zero_loss = hidden_flat.sum(dtype=loss_sum.dtype) * 0.0
    return torch.where(valid_tokens > 0, mean_loss, zero_loss)

class ReuseMmWithScaledInputGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, left, right, alpha):
        ctx.save_for_backward(left, right, alpha)
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        left, right, alpha = ctx.saved_tensors

        grad_output_for_output = None

        if ctx.needs_input_grad[1]:
            right_for_grad_left = right
            if right_for_grad_left.dtype != grad_output.dtype:
                right_for_grad_left = right_for_grad_left.to(dtype=grad_output.dtype)
            grad_left = torch.mm(grad_output, right_for_grad_left)
            alpha_for_grad_left = alpha
            if alpha_for_grad_left.dtype != grad_left.dtype:
                alpha_for_grad_left = alpha_for_grad_left.to(dtype=grad_left.dtype)
            while alpha_for_grad_left.ndim < grad_left.ndim:
                alpha_for_grad_left = alpha_for_grad_left.unsqueeze(-1)
            grad_left = grad_left * alpha_for_grad_left
            if grad_left.dtype != left.dtype:
                grad_left = grad_left.to(dtype=left.dtype)
        else:
            grad_left = None

        if ctx.needs_input_grad[2]:
            left_for_grad_right = left
            if left_for_grad_right.dtype != grad_output.dtype:
                left_for_grad_right = left_for_grad_right.to(dtype=grad_output.dtype)
            grad_right = torch.mm(grad_output.transpose(0, 1), left_for_grad_right)
            if grad_right.dtype != right.dtype:
                grad_right = grad_right.to(dtype=right.dtype)
        else:
            grad_right = None

        return grad_output_for_output, grad_left, grad_right, None

def compute_z_loss(logits: torch.Tensor, demean_logits: bool = True, 
                   z_loss_penalize_mean_logits: bool = True):
    """
    Computes ST-MoE router z loss (https://arxiv.org/abs/2202.08906)
    See equation (5) on page 7
    """

    # exponentiate logits, sum logits of each expert, take log, and square
    # code below is the same as:
    # > z_loss = torch.log(torch.exp(logits).sum(dim=-1)) ** 2.0
    if demean_logits:
        z_loss = torch.logsumexp(logits - logits.mean(dim=-1, keepdim=True), dim=-1) ** 2.0  # [B, T]
    else:
        z_loss = torch.logsumexp(logits, dim=-1) ** 2.0  # [B, T]

    if z_loss_penalize_mean_logits:
        mean_logit = logits.mean(dim=-1)  # [B, T]
        # Penalize both positive and negative mean logits.
        loss_mean_logit = mean_logit ** 2.0 # [B, T]
        # z_loss: ~[13, 30], loss_mean_logit: ~[0.1, 0.8]. 
        # So it won't dominate the z_loss, but still has a meaningful effect.
        z_loss = z_loss + loss_mean_logit

    # sum over all tokens and divide by total number of tokens
    return torch.mean(z_loss)

def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))

def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def get_moe_layer_indices(config):
    if config.n_exp <= 1:
        return []
    num_moe_layers = int(getattr(config, 'num_moe_layers', -1))
    moe_layer_stride = int(getattr(config, 'moe_layer_stride', 1))
    moe_layers = [
        layer_idx
        for layer_idx in range(config.n_layer)
        if (layer_idx >= config.moe_start_layer) and ((layer_idx + 1) % moe_layer_stride == 0)
    ]
    if num_moe_layers >= 0:
        return moe_layers[:num_moe_layers]
    return moe_layers

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.use_ve = has_ve(layer_idx, config.n_layer)
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        # Branch only on a static module attribute to avoid Dynamo recompiles on ve presence.
        if self.use_ve:
            assert ve is not None, "Expected value embeddings for VE-enabled layer"
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 2)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y

class Router(nn.Module):
    def __init__(self, config):
        super().__init__()

        # router settings
        self.top_k = config.moe_top_k
        self.n_exp = config.n_exp
        assert self.top_k >= 1 and self.top_k <= config.n_exp
        self.use_noisy_top_k = config.use_noisy_top_k
        self.train_capacity = config.train_capacity
        self.eval_capacity = config.eval_capacity
        self.min_capacity = config.min_capacity
        self.router_use_full_prec = config.router_use_full_prec

        # auxiliary / load balancing loss settings
        self.use_aux_loss           = config.use_aux_loss
        self.use_aux_free_load_balancing = bool(
            getattr(config, 'use_aux_free_load_balancing', False)
        )
        self.aux_free_load_balancing_bias_update_speed = float(
            getattr(config, 'aux_free_load_balancing_bias_update_speed', 1e-3)
        )
        self.use_router_z_loss      = config.use_router_z_loss
        self.z_loss_demean_logits = config.z_loss_demean_logits
        self.z_loss_penalize_mean_logits = config.z_loss_penalize_mean_logits
        # linear projection for (noisy) softmax gating
        # no bias is used, see page 4 eq (4) in (https://arxiv.org/abs/1701.06538)
        self.w_g = nn.Linear(config.n_embd, config.n_exp, bias=False)
        self.w_noise = nn.Linear(config.n_embd, config.n_exp, bias=False) if self.use_noisy_top_k else None
        self.router_z_loss_input_grad_scale = config.router_z_loss_input_grad_scale
        self.expert_probs = None
        self.top_k_indices = None
        self.register_buffer(
            'expert_bias',
            torch.zeros(self.n_exp, dtype=torch.float32),
        )
        self.register_buffer(
            'tokens_per_expert_counter',
            torch.zeros(self.n_exp, dtype=torch.float32),
            persistent=False,
        )
        if self.use_aux_loss and self.use_aux_free_load_balancing:
            raise ValueError("use_aux_loss and use_aux_free_load_balancing are mutually exclusive")

    def set_aux_free_load_balancing(self, enabled, bias_update_speed=None):
        self.use_aux_free_load_balancing = bool(enabled)
        self.use_aux_loss = not self.use_aux_free_load_balancing
        if bias_update_speed is not None:
            self.aux_free_load_balancing_bias_update_speed = float(bias_update_speed)
        self.tokens_per_expert_counter.zero_()

    def _get_selection_scores(self, logits):
        if not self.use_aux_free_load_balancing:
            return logits
        expert_bias = self.expert_bias.to(device=logits.device, dtype=logits.dtype)
        return logits + expert_bias

    @torch.no_grad()
    def _accumulate_aux_free_load_balancing_counts(self, top_k_indices):
        if not self.use_aux_free_load_balancing:
            return
        token_counts = torch.bincount(top_k_indices.reshape(-1), minlength=self.n_exp)
        token_counts = token_counts.to(
            device=self.tokens_per_expert_counter.device,
            dtype=self.tokens_per_expert_counter.dtype,
        )
        self.tokens_per_expert_counter.add_(token_counts)

    @torch.no_grad()
    def update_aux_free_load_balancing(self):
        if not self.use_aux_free_load_balancing:
            return
        counts = self.tokens_per_expert_counter
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts)
        if bool((counts.sum() == 0).item()):
            counts.zero_()
            return
        mean_count = counts.mean()
        self.expert_bias.add_(
            self.aux_free_load_balancing_bias_update_speed * torch.sign(mean_count - counts)
        )
        self.expert_bias.sub_(self.expert_bias.mean())
        counts.zero_()

    def forward(self, x):
        """
        Computes routing information for tokens, including which experts to use,
        the weights for their outputs, and their position within the expert's batch.
        This implementation is memory-efficient and avoids quadratic scaling with batch size.
        """
        # The router can be sensitive to precision issues, so we can run it in full float32.
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        ctx = nullcontext() if not self.router_use_full_prec else torch.amp.autocast(device_type=device_type, enabled=False)

        with ctx:
            B, T, C = x.size()
            num_tokens = B * T
            x_flat = x.view(num_tokens, C)

            # 1. GET ROUTING LOGITS
            # ---------------------
            logits_wg = F.linear(x_flat, self.w_g.weight)  # [B*T, n_exp]
            noise = None  # Initialize noise variable

            if self.training and self.use_noisy_top_k:
                noise = F.softplus(self.w_noise(x_flat))
                noise *= torch.randn_like(noise)
            logits = logits_wg if noise is None else logits_wg + noise

            # 2. COMPUTE LOSSES (if training)
            # -------------------------------
            if self.training:
                selection_scores = self._get_selection_scores(logits)
                _, top_k_indices = selection_scores.topk(self.top_k, dim=-1)
                self._accumulate_aux_free_load_balancing_counts(top_k_indices)

                logits_for_router = logits

                # Router Z-loss prevents logits from growing too large
                if self.use_router_z_loss:
                    if self.router_z_loss_input_grad_scale == 1:
                        logits_for_z_loss = logits_for_router
                    else:
                        input_alpha_t = torch.as_tensor(self.router_z_loss_input_grad_scale, device=logits.device, dtype=logits.dtype)
                        logits_wg_for_z_loss = ReuseMmWithScaledInputGrad.apply(
                            logits_wg, x_flat, self.w_g.weight, input_alpha_t
                        )
                        logits_for_z_loss = logits_wg_for_z_loss if noise is None else logits_wg_for_z_loss + noise

                    router_z_loss = compute_z_loss(logits_for_z_loss.view(B, T, -1), 
                                                   demean_logits=self.z_loss_demean_logits,
                                                   z_loss_penalize_mean_logits=self.z_loss_penalize_mean_logits)
                    MANAGER.add("router_z_loss", router_z_loss)

                # Find top-k choices for each token
                top_k_logits = logits_for_router.gather(-1, top_k_indices) # [B*T, k]
                router_probs = F.softmax(top_k_logits, dim=-1) # [B*T, k]
                
                # The auxiliary loss encourages load balancing across experts
                if self.use_aux_loss:
                    # Use the full router distribution here so the balancing loss keeps
                    # a meaningful gradient signal even when top_k = 1.
                    all_probs = F.softmax(logits_for_router, dim=-1)
                    aux_loss = self.compute_aux_loss(all_probs.view(B, T, -1), top_k_indices.view(B, T, -1))
                    MANAGER.add("aux_loss", aux_loss)
                    self.expert_probs = all_probs.view(B, T, -1).detach().clone()
                    self.top_k_indices = top_k_indices.view(B, T, -1).clone()
            else:
                # At inference, we just need the top-k
                selection_scores = self._get_selection_scores(logits)
                _, top_k_indices = selection_scores.topk(self.top_k, dim=-1)
                top_k_logits = logits.gather(-1, top_k_indices)
                router_probs = F.softmax(top_k_logits, dim=-1) # [B*T, k]

            top_k_scores = top_k_logits

            if self.training or MANAGER.collect_load_balancing_stats:
                selected_scores = self.compute_selected_scores(logits.view(B, T, -1), top_k_indices.view(B, T, -1))
                MANAGER.add("selected_scores", selected_scores.detach())

            # 3. COMPUTE ROUTER PROBABILITIES
            # --------------------------------
            # We normalize the probabilities over the top-k experts

            # 4. DETERMINE TOKEN RANKS WITH CAPACITY LIMITING
            # -----------------------------------------------
            exp_capacity = self.get_capacity(num_tokens)
            
            # Create a one-hot mask of the chosen experts for each token. Shape: [B*T, k, n_exp]
            expert_mask_one_hot = F.one_hot(top_k_indices, num_classes=self.n_exp)

            # ANCHOR[id=routing_ranks]
            # This is the critical step to ensure load balancing prioritizes top-1 experts.
            # We flatten the k dimension first, so cumsum processes all top-1 choices, then all top-2, etc.
            # This is the memory-efficient equivalent of the original logic.
            # Because it permutes to `[k, tokens, experts]` before cumsum, we are enforcing:
            # - all **top-1** assignments fill capacity first,
            # - then **top-2** try to use remaining capacity,
            # - etc.
            # That reduces a different pathology (top-2 stealing capacity from top-1), 
            # but it **doesn’t remove within-top-1 ordering bias**: within the top-1 pass, 
            # token order still matters.
            reshaped_mask = expert_mask_one_hot.permute(1, 0, 2).reshape(self.top_k * num_tokens, self.n_exp)
            cumulative_sum = torch.cumsum(reshaped_mask, dim=0)
            
            # Reshape back to the original layout
            position_in_expert = cumulative_sum.reshape(self.top_k, num_tokens, self.n_exp).permute(1, 0, 2)
            
            # The rank is the position, but we only care about the rank for the selected expert.
            # We multiply by the one-hot mask to zero out positions for non-selected experts.
            # NOTE: rank is not vetted with exp_capacity yet. So it includes over-capacity positions.
            rank = (position_in_expert - 1) * expert_mask_one_hot
            
            # 5. GENERATE FINAL MASKS AND RANKS FOR THE MOE LAYER
            # ----------------------------------------------------
            # Create a mask to drop tokens that exceed the expert's capacity
            # rank >= exp_capacity -> drop token 
            # (the current layer outputs zero for that token. 
            # Only relies on the residual connection)
            capacity_mask = rank < exp_capacity

            # The final expert mask includes both the expert choice and the capacity check.
            final_expert_mask = expert_mask_one_hot * capacity_mask # [B*T, k, n_exp]
            
            # Router probabilities are also masked. If a token is dropped, its probability is zero.
            # We check if the token was assigned to any expert in its k-th slot.
            probs_mask = (final_expert_mask.sum(dim=-1) > 0) # [B*T, k]
            router_probs_masked = router_probs * probs_mask
            top_k_scores_masked = top_k_scores * probs_mask

            # The final rank is collapsed to a single value per top-k choice.
            # It adds across the expert dimension, since only one expert per top-k slot is selected,
            # and all other positions are zeros. 
            # NOTE: final_rank is derived from rank, so it also includes 
            # over-capacity positions.
            final_rank = torch.sum(rank, dim=-1) # [B*T, k]

            # The MOELayer will use these tensors to efficiently dispatch and combine tokens.
            # Their memory usage all scale linearly with (B * T).
            return final_expert_mask, router_probs_masked, top_k_scores_masked, top_k_indices, final_rank
    
    def compute_aux_loss(self, expert_probs: torch.Tensor, indices: torch.Tensor):
        """
        Computes Switch Transformer auxiliary loss (https://arxiv.org/abs/2101.03961)
        See equations (4)-(6) on page 7
        """

        # equation (5): compute ratio of tokens allocated to each expert
        # total number of tokens is defined as total tokens in batch * k
        # (k = 1) for the Switch Transformer
        with torch.no_grad():
            one_hot_indices = F.one_hot(indices, num_classes=self.n_exp)  # [B, T, k, n_exp]
            one_hot_indices = torch.sum(one_hot_indices.float(), dim=2)  # [B, T, n_exp] (sum over k dimension)
            tokens_per_expert = torch.mean(one_hot_indices.float(), dim=(0, 1))

        # equation (6): compute ratio of router probability allocated to each expert
        prob_per_expert = torch.mean(expert_probs.float(), dim=(0, 1))

        # equation (4): take a scaled dot product between prob/token allocation vectors
        # multiply the result by the number of experts
        return self.n_exp * torch.sum(prob_per_expert * tokens_per_expert)
        
    def compute_selected_scores(self, logits: torch.Tensor, top_k_indices: torch.Tensor):
        """
        logits: [B, T, n_exp]  (router logits or scores)
        top_k_indices: [B, T, k]
        returns: aux_scores [n_exp]
        """
        with torch.no_grad():
            B, T, n_exp = logits.shape
            k = top_k_indices.shape[-1]

            # counts per expert over (B,T,k)
            one_hot = F.one_hot(top_k_indices, num_classes=n_exp).float()   # [B,T,k,n_exp]
            counts = one_hot.sum(dim=(0, 1, 2))                              # [n_exp]
            total = counts.sum().clamp_min(1.0)

            # frequency over assignments (sums to 1)
            tokens_per_expert = counts / total                               # [n_exp]

            # sum of selected logits per expert
            sel_logits = logits.gather(-1, top_k_indices)                    # [B,T,k]
            score_sum = (sel_logits.unsqueeze(-1) * one_hot).sum(dim=(0,1,2))# [n_exp]

            # mean logit given selected
            mean_selected_scores = score_sum / counts.clamp_min(1.0)          # [n_exp]
            return mean_selected_scores

    def get_capacity(self, tokens_per_batch):
        # expert capacity is given by (tokens_per_batch / num_experts) * capacity_factor
        # see eq (3) in Switch Transformer (https://arxiv.org/abs/2101.03961)
        capacity_factor = self.train_capacity if self.training else self.eval_capacity
        capacity = math.floor(self.top_k * capacity_factor * tokens_per_batch / self.n_exp)
        capacity += capacity % 2 # make sure capacity is an even number
        capacity = max(capacity, self.min_capacity) # use min capacity
        assert capacity > 0
        return int(capacity)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config, layer_idx, use_moe=False):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        if use_moe:
            self.mlp = MOELayer(config, layer_idx)
        elif getattr(config, 'use_qwen3_dense_mlp', True):
            self.mlp = Qwen3MLP(config, layer_idx=layer_idx)
        else:
            self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x

# NOTE: MLPExperts is not used in our default settings. Instead, we always use Qwen3MLPExperts.
class MLPExperts(nn.Module):
    """
    implementation of multiple MLP-based experts that can process input
    in batch -- based upon ColossalAI OpenMoE but simple, has optional bias, and
    uses a bmm instead of a loop over a mm for each expert to improve efficiency
    link: https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/moe/experts.py
    """
    def __init__(self, config):
        # TODO: add param init
        super().__init__()
        self.c_fc = nn.Parameter(torch.empty(config.n_exp, config.n_embd, 4 * config.n_embd))
        self.c_proj = nn.Parameter(torch.empty(config.n_exp, 4 * config.n_embd, config.n_embd))

    def forward(self, x, selected_router_scores=None):
        fc_out = torch.bmm(x, self.c_fc)
        x = F.relu(fc_out).square()
        proj_out = torch.bmm(x, self.c_proj)
        return proj_out


class GateProjBiasEmaTargetKeeper(nn.Module):
    rms_eps = 1e-12

    def __init__(self, beta, anchor_start, anchor_end, floor_frac):
        super().__init__()
        self.beta = float(beta)
        self.anchor_start = float(anchor_start)
        self.anchor_end = float(anchor_end)
        self.floor_frac = float(floor_frac)
        self.register_buffer("ema_rms", torch.zeros(()))
        self.register_buffer("target_rms", torch.zeros(()))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))
        self.register_buffer("target_ready", torch.zeros((), dtype=torch.bool))
        self.register_buffer("total_iterations", torch.ones((), dtype=torch.int64), persistent=False)

    def _raise_if_nonfinite(self, tensor, label, source=None):
        if torch.isfinite(tensor).all():
            return
        bad = (~torch.isfinite(tensor)).nonzero(as_tuple=False)
        index = tuple(int(i) for i in bad[0].tolist()) if bad.numel() > 0 else ()
        value = tensor[index] if index else tensor
        scalar_value = float(value.item()) if value.numel() == 1 else str(value)
        source_suffix = "" if source is None else f" from {source}"
        raise RuntimeError(
            f"GateProjBiasEmaTargetKeeper observed non-finite {label}{source_suffix} at index {index}: {scalar_value}"
        )

    def _compute_rms(self, value):
        mean_sq = value.float().square().mean()
        return (mean_sq + self.rms_eps).sqrt()

    def set_total_iterations(self, total_iterations):
        self.total_iterations.fill_(max(int(total_iterations), 1))

    def _resolve_anchor_steps(self):
        total_iterations = max(int(self.total_iterations.item()), 1)
        anchor_start = min(max(math.ceil(total_iterations * self.anchor_start), 0), total_iterations)
        anchor_end = min(max(math.ceil(total_iterations * self.anchor_end), 0), total_iterations)
        return anchor_start, anchor_end

    @torch.no_grad()
    def update(self, value, step, source=None):
        self._raise_if_nonfinite(value, "value", source=source)
        rms = self._compute_rms(value.detach())
        self._raise_if_nonfinite(rms, "rms", source=source)
        if not bool(self.initialized.item()):
            self.ema_rms.copy_(rms)
            self.initialized.fill_(True)
        else:
            self.ema_rms.mul_(self.beta).add_(rms, alpha=1.0 - self.beta)
        self._raise_if_nonfinite(self.ema_rms, "ema_rms", source=source)
        anchor_start, anchor_end = self._resolve_anchor_steps()
        # If step < anchor_start, we are in the warming-up period, 
        # and we keep target_rms at zero and target_ready at False, 
        # so that the regularization is disabled.
        if anchor_start <= step <= anchor_end:
            self.target_rms.copy_(self.ema_rms)
            self.target_ready.fill_(True)
        # If step > anchor_end, we keep using the target_rms from the anchor period, 
        # and target_ready is still True, so that the regularization remains stable.
        self._raise_if_nonfinite(self.target_rms, "target_rms", source=source)

    def loss(self, value, source=None):
        self._raise_if_nonfinite(value, "value", source=source)
        if not bool(self.target_ready.item()):
            loss = value.new_zeros((), dtype=torch.float32)
            self._raise_if_nonfinite(loss, "loss", source=source)
            return loss
        value_f = value.float()
        current_rms = self._compute_rms(value_f)
        self._raise_if_nonfinite(current_rms, "current_rms", source=source)
        floor = self.target_rms.detach() * self.floor_frac
        self._raise_if_nonfinite(floor, "floor", source=source)
        loss = torch.relu(floor - current_rms).square()
        self._raise_if_nonfinite(loss, "loss", source=source)
        return loss

# Borrowed Qwen3MoeMLP implementation from modeling_qwen3_moe.py.
class Qwen3MLP(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.hidden_size = config.n_embd
        self.intermediate_size = 4 * config.n_embd
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # up_proj -> c_fc, down_proj -> c_proj
        # to ensure minimal code changes when switching between Qwen3MoeMLP and regular MLP.
        self.c_fc = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.c_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = SiLUActivation()
        self.kappa_input = getattr(config, 'kappa_input', 'router_probs')
        self.kappa_input_constant = getattr(config, 'kappa_input_constant', None)
        self.register_buffer(
            'kappa_slope_max_scale',
            torch.tensor(float(getattr(config, 'dense_kappa_slope_max_scale', 2.0))),
            persistent=False,
        )
        self.global_kappa_bias_granularity = getattr(config, 'global_kappa_bias_granularity', 'per-gate')
        self.kappa_bias_ema_rms_reg = bool(getattr(config, 'kappa_bias_ema_rms_reg', False))
        kappa_bias_start_layer = int(getattr(config, 'kappa_bias_start_layer', 0))
        self.use_kappa_swiglu = (
            bool(getattr(config, 'use_kappa_swiglu', False))
            and bool(getattr(config, 'constant_kappa_bias_dense_layers', False))
        )
        self.register_buffer('kappa_bias_ema_rms_reg_step', torch.zeros((), dtype=torch.int64), persistent=False)
        self.has_active_kappa_bias = self.use_kappa_swiglu and (
            layer_idx is None or layer_idx >= kappa_bias_start_layer
        )
        self._shared_kappa_bias = None
        self._eval_kappa_slope_scales_cache = None
        self._eval_kappa_slope_scales_cache_dtype = None
        self._eval_kappa_slope_scales_cache_device = None
        self._eval_kappa_slope_scales_cache_bias_version = None
        self._eval_kappa_slope_scales_cache_scale_version = None
        self.kappa_bias_ema_rms_reg_keeper = None
        self.kappa_scale_ema_rms_reg_keeper = None
        if self.has_active_kappa_bias:
            kappa_bias_shape = self._get_kappa_bias_parameter_shape()
            if self.global_kappa_bias_granularity == 'global':
                self.register_parameter('kappa_bias', None)
            else:
                self.kappa_bias = nn.Parameter(torch.empty(*kappa_bias_shape))
            if self.kappa_bias_ema_rms_reg:
                keeper = GateProjBiasEmaTargetKeeper(
                    beta=getattr(config, 'kappa_bias_l2_ema_beta', 0.99),
                    anchor_start=getattr(config, 'kappa_bias_l2_ema_anchor_start', 0.4),
                    anchor_end=getattr(config, 'kappa_bias_l2_ema_anchor_end', 0.8),
                    floor_frac=getattr(config, 'kappa_bias_l2_ema_floor_frac', 0.8),
                )
                self.kappa_bias_ema_rms_reg_keeper = keeper
                self.kappa_scale_ema_rms_reg_keeper = GateProjBiasEmaTargetKeeper(
                    beta=keeper.beta,
                    anchor_start=keeper.anchor_start,
                    anchor_end=keeper.anchor_end,
                    floor_frac=keeper.floor_frac,
                )
        else:
            # disabled_kappa_bias: placeholder to satisfy _materialize_kappa_bias().
            self.register_buffer(
                'disabled_kappa_bias',
                torch.zeros(self.intermediate_size),
                persistent=False,
            )

    def _get_kappa_bias_parameter_shape(self):
        if self.global_kappa_bias_granularity == 'per-gate':
            return (self.intermediate_size,)
        if self.global_kappa_bias_granularity in {'per-expert', 'per-layer', 'global'}:
            return (1,)
        raise ValueError(
              f"Unsupported kappa bias granularity: {self.global_kappa_bias_granularity!r}"
        )

    def bind_shared_kappa_bias(self, kappa_bias):
        if self.global_kappa_bias_granularity != 'global':
            raise ValueError("Shared kappa_bias binding is only valid for global granularity")
        self._shared_kappa_bias = kappa_bias

    def _get_kappa_bias_parameter(self):
        kappa_bias = getattr(self, 'kappa_bias', None)
        if kappa_bias is not None:
            return kappa_bias
        return self._shared_kappa_bias

    @torch._dynamo.disable
    def _materialize_kappa_bias(self):
        if not self.has_active_kappa_bias:
            return self.disabled_kappa_bias.detach().requires_grad_(True)
        kappa_bias = self._get_kappa_bias_parameter()
        if kappa_bias is None:
            raise RuntimeError("kappa_bias was enabled but no parameter was bound")
        if self.global_kappa_bias_granularity == 'per-gate':
            return kappa_bias + 0
        return kappa_bias.reshape(1).expand(self.intermediate_size) + 0

    def _compute_kappa_slope_scales(self, kappa_bias):
        target_dtype = torch.float32 if self.training else kappa_bias.dtype
        kappa_bias = kappa_bias.to(dtype=target_dtype)
        kappa_slope_max_scale = self.kappa_slope_max_scale.to(device=kappa_bias.device, dtype=target_dtype)
        input_constant = torch.as_tensor(
            self.kappa_input_constant,
            device=kappa_bias.device,
            dtype=target_dtype,
        )
        log_kappa = kappa_bias * input_constant
        return torch.exp(torch.log(kappa_slope_max_scale) * torch.tanh(log_kappa))

    @torch._dynamo.disable
    def _materialize_kappa_slope_scales_for_eval(self, target_dtype, target_device):
        kappa_bias_param = self._get_kappa_bias_parameter() if self.has_active_kappa_bias else None
        bias_version = None if kappa_bias_param is None else kappa_bias_param._version
        scale_version = self.kappa_slope_max_scale._version
        if (
            self._eval_kappa_slope_scales_cache is not None
            and self._eval_kappa_slope_scales_cache_dtype == target_dtype
            and self._eval_kappa_slope_scales_cache_device == target_device
            and self._eval_kappa_slope_scales_cache_bias_version == bias_version
            and self._eval_kappa_slope_scales_cache_scale_version == scale_version
        ):
            return self._eval_kappa_slope_scales_cache
        kappa_bias = self._materialize_kappa_bias().to(device=target_device, dtype=target_dtype)
        slope_scales = self._compute_kappa_slope_scales(kappa_bias)
        self._eval_kappa_slope_scales_cache = slope_scales
        self._eval_kappa_slope_scales_cache_dtype = target_dtype
        self._eval_kappa_slope_scales_cache_device = target_device
        self._eval_kappa_slope_scales_cache_bias_version = bias_version
        self._eval_kappa_slope_scales_cache_scale_version = scale_version
        return slope_scales

    def set_kappa_bias_ema_rms_reg_step(self, step):
        self.kappa_bias_ema_rms_reg_step.fill_(int(step))

    def set_kappa_slope_max_scale(self, kappa_slope_max_scale):
        self.kappa_slope_max_scale.fill_(float(kappa_slope_max_scale))

    def set_kappa_bias_ema_rms_reg_total_iterations(self, total_iterations):
        if self.kappa_bias_ema_rms_reg_keeper is not None:
            self.kappa_bias_ema_rms_reg_keeper.set_total_iterations(total_iterations)
        if self.kappa_scale_ema_rms_reg_keeper is not None:
            self.kappa_scale_ema_rms_reg_keeper.set_total_iterations(total_iterations)

    def _kappa_bias_debug_source(self, suffix):
        owner = self.__class__.__name__
        layer = "unknown" if self.layer_idx is None else str(self.layer_idx)
        granularity = self.global_kappa_bias_granularity
        return f"{owner}(layer={layer}, granularity={granularity}).{suffix}"

    @torch._dynamo.disable
    def _accumulate_kappa_bias_l2_losses(self, kappa_bias):
        kappa_bias = kappa_bias.float()
        MANAGER.add("kappa_bias_l2_loss", kappa_bias.square().mean())
        if self.kappa_bias_ema_rms_reg_keeper is not None:
            self.kappa_bias_ema_rms_reg_keeper.update(
                kappa_bias,
                int(self.kappa_bias_ema_rms_reg_step.item()),
                source=self._kappa_bias_debug_source("kappa_bias"),
            )
            MANAGER.add(
                "kappa_bias_ema_rms_reg_loss",
                self.kappa_bias_ema_rms_reg_keeper.loss(
                    kappa_bias,
                    source=self._kappa_bias_debug_source("kappa_bias"),
                ),
            )

    def _accumulate_kappa_scale_l2_losses(self, kappa_scale):
        kappa_scale = kappa_scale.float()
        MANAGER.add("kappa_scale_l2_loss", kappa_scale.square().mean())
        if self.kappa_scale_ema_rms_reg_keeper is not None:
            self.kappa_scale_ema_rms_reg_keeper.update(
                kappa_scale,
                int(self.kappa_bias_ema_rms_reg_step.item()),
                source=self._kappa_bias_debug_source("kappa_scale"),
            )
            MANAGER.add(
                "kappa_scale_ema_rms_reg_loss",
                self.kappa_scale_ema_rms_reg_keeper.loss(
                    kappa_scale,
                    source=self._kappa_bias_debug_source("kappa_scale"),
                ),
            )

    @torch._dynamo.disable
    def _update_kappa_slope_scale_stats(self, slope_scales):
        if not MANAGER.collect_load_balancing_stats or not self.has_active_kappa_bias:
            return

        slope_scales = slope_scales.detach().float()
        flat_slope_scales = slope_scales.reshape(1, -1)
        flat_mask = torch.ones_like(flat_slope_scales, dtype=torch.bool)
        slope_scale_mean = slope_scales.mean()
        MANAGER.add("kappa_slope_scale_abs_mean", slope_scale_mean)
        MANAGER.add(
            "kappa_slope_scale_abs_top5p_mean",
            _mean_extreme_percentile_per_row(
                flat_slope_scales,
                flat_mask,
                fraction=0.05,
                largest=True,
            ).squeeze(0),
        )
        MANAGER.add(
            "kappa_slope_scale_abs_bottom5p_mean",
            _mean_extreme_percentile_per_row(
                flat_slope_scales,
                flat_mask,
                fraction=0.05,
                largest=False,
            ).squeeze(0),
        )
        MANAGER.add("kappa_slope_scale_abs_mean_normalized", slope_scale_mean)

    def forward(self, x):
        gate_out_raw = self.gate_proj(x)
        kappa_bias = self._materialize_kappa_bias()
        if self.training:
            slope_scales = self._compute_kappa_slope_scales(kappa_bias)
        else:
            slope_scales = self._materialize_kappa_slope_scales_for_eval(
                gate_out_raw.dtype,
                gate_out_raw.device,
            )
        if self.training:
            self._accumulate_kappa_bias_l2_losses(kappa_bias)
        self._update_kappa_slope_scale_stats(slope_scales)
        gate_out = gate_out_raw * torch.sigmoid(
            gate_out_raw * slope_scales.to(dtype=gate_out_raw.dtype)
        )
        down_proj = self.c_proj(gate_out * self.c_fc(x))
        return down_proj

class Qwen3MLPExperts(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.debug = config.debug
        self.n_exp = config.n_exp
        self.hidden_size = config.n_embd
        self.intermediate_size = 4 * config.n_embd
        self.bilinear_mlp_moe = bool(getattr(config, 'bilinear_mlp_moe', False))
        self.kappa_input = getattr(config, 'kappa_input', 'router_probs')
        self.register_buffer(
            'kappa_slope_max_scale',
            torch.tensor(float(getattr(config, 'moe_kappa_slope_max_scale', 3.0))),
            persistent=False,
        )
        self.global_kappa_bias_granularity = getattr(config, 'global_kappa_bias_granularity', 'per-gate')
        self.gate_stats_threshold = float(getattr(config, 'gate_stats_threshold', 0.1))
        self.gate_stats_topk = int(getattr(config, 'gate_stats_topk', 16))
        self.kappa_bias_ema_rms_reg = bool(getattr(config, 'kappa_bias_ema_rms_reg', False))
        kappa_bias_start_layer = int(getattr(config, 'kappa_bias_start_layer', 0))
        self.use_kappa_swiglu = bool(getattr(config, 'use_kappa_swiglu', False)) and (
            layer_idx is None or layer_idx >= kappa_bias_start_layer
        )
        self.log_implicit_gate_proj_bias = bool(getattr(config, 'log_implicit_gate_proj_bias', False))
        self.register_buffer('kappa_bias_ema_rms_reg_step', torch.zeros((), dtype=torch.int64), persistent=False)
        self._shared_kappa_bias = None
        self._shared_kappa_scale = None
        self._eval_kappa_bias_cache = None
        self._eval_kappa_bias_cache_dtype = None
        self._eval_kappa_bias_cache_device = None
        self._eval_kappa_bias_cache_version = None
        self._eval_kappa_bias_unsqueezed_cache = None
        self._eval_kappa_bias_unsqueezed_cache_dtype = None
        self._eval_kappa_bias_unsqueezed_cache_device = None
        self._eval_kappa_bias_unsqueezed_cache_version = None
        self._eval_kappa_scale_cache = None
        self._eval_kappa_scale_cache_dtype = None
        self._eval_kappa_scale_cache_device = None
        self._eval_kappa_scale_cache_version = None
        self._eval_kappa_scale_unsqueezed_cache = None
        self._eval_kappa_scale_unsqueezed_cache_dtype = None
        self._eval_kappa_scale_unsqueezed_cache_device = None
        self._eval_kappa_scale_unsqueezed_cache_version = None
        self._eval_log_kappa_slope_max_scale_cache = None
        self._eval_log_kappa_slope_max_scale_cache_dtype = None
        self._eval_log_kappa_slope_max_scale_cache_device = None
        self._eval_log_kappa_slope_max_scale_cache_version = None
        self.kappa_bias_ema_rms_reg_keeper = None
        self.kappa_scale_ema_rms_reg_keeper = None
        self.gate_proj = nn.Parameter(
            torch.empty(self.n_exp, self.hidden_size, self.intermediate_size)
        )
        self.use_kappa_scale = (
            self.use_kappa_swiglu
            and self.kappa_input in {'top_logits', 'router_probs'}
        )
        if self.use_kappa_swiglu:
            kappa_bias_shape = self._get_kappa_bias_parameter_shape()
            if self.global_kappa_bias_granularity == 'global':
                self.register_parameter('kappa_bias', None)
                self.register_parameter('kappa_scale', None)
            else:
                self.kappa_bias = nn.Parameter(torch.empty(*kappa_bias_shape))
                if self.use_kappa_scale:
                    self.kappa_scale = nn.Parameter(torch.empty(*kappa_bias_shape))
                else:
                    self.register_parameter('kappa_scale', None)
            self.register_parameter('kappa_bias_expert', None)
            self.register_parameter('kappa_bias_intermediate', None)
            self.register_parameter('kappa_bias_residual', None)
            if self.kappa_bias_ema_rms_reg:
                keeper_kwargs = {
                    'beta': getattr(config, 'kappa_bias_l2_ema_beta', 0.99),
                    'anchor_start': getattr(config, 'kappa_bias_l2_ema_anchor_start', 0.4),
                    'anchor_end': getattr(config, 'kappa_bias_l2_ema_anchor_end', 0.8),
                    'floor_frac': getattr(config, 'kappa_bias_l2_ema_floor_frac', 0.8),
                }
                self.kappa_bias_ema_rms_reg_keeper = GateProjBiasEmaTargetKeeper(**keeper_kwargs)
                if self.use_kappa_scale:
                    self.kappa_scale_ema_rms_reg_keeper = GateProjBiasEmaTargetKeeper(**keeper_kwargs)
        else:
            self.register_parameter('kappa_bias', None)
            self.register_parameter('kappa_scale', None)
            self.register_parameter('kappa_bias_expert', None)
            self.register_parameter('kappa_bias_intermediate', None)
            self.register_parameter('kappa_bias_residual', None)
            # disabled_kappa_bias: placeholder to satisfy _materialize_kappa_bias().
            self.register_buffer(
                "disabled_kappa_bias",
                torch.zeros(self.n_exp, self.intermediate_size),
                persistent=False,
            )
        self.register_buffer(
            "disabled_kappa_scale",
            torch.ones(self.n_exp, self.intermediate_size),
            persistent=False,
        )
        self.register_buffer("initial_kappa_bias", None, persistent=False)
        self.c_fc   = nn.Parameter(torch.empty(self.n_exp, self.hidden_size, self.intermediate_size))
        self.c_proj = nn.Parameter(torch.empty(self.n_exp, self.intermediate_size, self.hidden_size))

        self.act_fn = SiLUActivation()
        self.fc_bias = None
        self.proj_bias = None
        self.z_loss_demean_logits = config.z_loss_demean_logits
        self.z_loss_penalize_mean_logits = config.z_loss_penalize_mean_logits
        # Since we update router_confidence_gate_bias_grad_scale 
        # continuously during training, if it's just a float, Dynamo would
        # compile guards around its value and recompile whenever it changes.
        # So we change it to a buffer tensor, and Dynamo will treat it as an input
        # to the compiled region, avoiding recompiles on value changes.
        self.register_buffer(
            "router_confidence_gate_bias_grad_scale",
            torch.tensor(0.1),
            persistent=False,
        )
        self.gate_out_acts_normed = None
        self.last_gate_stats = None
        # Weak reference to the router. Avoid registering it as a child module.
        self._router_ref = None

    def _apply_gate_activation(self, gate_out_raw):
        if self.bilinear_mlp_moe:
            return gate_out_raw
        return self.act_fn(gate_out_raw)

    def _get_kappa_bias_parameter_shape(self):
        if self.global_kappa_bias_granularity == 'per-gate':
            return (self.n_exp, self.intermediate_size)
        if self.global_kappa_bias_granularity == 'per-expert':
            return (self.n_exp,)
        if self.global_kappa_bias_granularity in {'per-layer', 'global'}:
            return (1,)
        raise ValueError(
              f"Unsupported kappa bias granularity: {self.global_kappa_bias_granularity!r}"
        )

    def bind_shared_kappa_bias(self, kappa_bias):
        if self.global_kappa_bias_granularity != 'global':
            raise ValueError("Shared kappa_bias binding is only valid for global granularity")
        self._shared_kappa_bias = kappa_bias

    def bind_shared_kappa_scale(self, kappa_scale):
        if self.global_kappa_bias_granularity != 'global':
            raise ValueError("Shared kappa_scale binding is only valid for global granularity")
        self._shared_kappa_scale = kappa_scale

    def _get_kappa_bias_parameter(self):
        kappa_bias = self.kappa_bias
        if kappa_bias is not None:
            return kappa_bias
        return self._shared_kappa_bias

    def _get_kappa_scale_parameter(self):
        kappa_scale = self.kappa_scale
        if kappa_scale is not None:
            return kappa_scale
        return self._shared_kappa_scale

    @torch.no_grad()
    def snapshot_kappa_bias_reference(self):
        if not self.use_kappa_swiglu:
            self.initial_kappa_bias = None
            return
        self.initial_kappa_bias = self._materialize_kappa_bias().detach().clone()
        return self.initial_kappa_bias

    '''
    Bias-enabled layers use a dense kappa_bias matrix parameter.
    Bias-disabled layers return a zero buffer.
    @torch._dynamo.disable keeps Dynamo from tracing across those representation
    differences and treats the materialized bias matrix as an input tensor instead.
    '''
    @torch._dynamo.disable
    def _materialize_kappa_bias(self):
        if not self.use_kappa_swiglu:
            return self.disabled_kappa_bias.detach().requires_grad_(True)
        kappa_bias = self._get_kappa_bias_parameter()
        if kappa_bias is None:
            raise RuntimeError("kappa_bias was enabled but no parameter was bound")
        if self.global_kappa_bias_granularity == 'per-gate':
            return kappa_bias + 0
        if self.global_kappa_bias_granularity == 'per-expert':
            return kappa_bias.unsqueeze(-1).expand(-1, self.intermediate_size) + 0
        return kappa_bias.reshape(1, 1).expand(self.n_exp, self.intermediate_size) + 0

    @torch._dynamo.disable
    def _materialize_kappa_scale(self):
        if not self.use_kappa_scale:
            return self.disabled_kappa_scale.detach().requires_grad_(True)
        kappa_scale = self._get_kappa_scale_parameter()
        if kappa_scale is None:
            raise RuntimeError("kappa_scale was enabled but no parameter was bound")
        if self.global_kappa_bias_granularity == 'per-gate':
            return kappa_scale + 0
        if self.global_kappa_bias_granularity == 'per-expert':
            return kappa_scale.unsqueeze(-1).expand(-1, self.intermediate_size) + 0
        return kappa_scale.reshape(1, 1).expand(self.n_exp, self.intermediate_size) + 0

    @torch._dynamo.disable
    def _materialize_kappa_bias_for_eval(self, target_dtype, target_device):
        if not self.use_kappa_swiglu:
            return self.disabled_kappa_bias.to(device=target_device, dtype=target_dtype)
        kappa_bias = self._get_kappa_bias_parameter()
        if kappa_bias is None:
            raise RuntimeError("kappa_bias was enabled but no parameter was bound")
        version = kappa_bias._version
        if (
            self._eval_kappa_bias_cache is not None
            and self._eval_kappa_bias_cache_dtype == target_dtype
            and self._eval_kappa_bias_cache_device == target_device
            and self._eval_kappa_bias_cache_version == version
        ):
            return self._eval_kappa_bias_cache
        materialized = self._materialize_kappa_bias().to(device=target_device, dtype=target_dtype)
        self._eval_kappa_bias_cache = materialized
        self._eval_kappa_bias_cache_dtype = target_dtype
        self._eval_kappa_bias_cache_device = target_device
        self._eval_kappa_bias_cache_version = version
        return materialized

    @torch._dynamo.disable
    def _materialize_kappa_scale_for_eval(self, target_dtype, target_device):
        if not self.use_kappa_scale:
            return self.disabled_kappa_scale.to(device=target_device, dtype=target_dtype)
        kappa_scale = self._get_kappa_scale_parameter()
        if kappa_scale is None:
            raise RuntimeError("kappa_scale was enabled but no parameter was bound")
        version = kappa_scale._version
        if (
            self._eval_kappa_scale_cache is not None
            and self._eval_kappa_scale_cache_dtype == target_dtype
            and self._eval_kappa_scale_cache_device == target_device
            and self._eval_kappa_scale_cache_version == version
        ):
            return self._eval_kappa_scale_cache
        materialized = self._materialize_kappa_scale().to(device=target_device, dtype=target_dtype)
        self._eval_kappa_scale_cache = materialized
        self._eval_kappa_scale_cache_dtype = target_dtype
        self._eval_kappa_scale_cache_device = target_device
        self._eval_kappa_scale_cache_version = version
        return materialized

    @torch._dynamo.disable
    def _get_kappa_bias_unsqueezed_for_eval(self, target_dtype, target_device):
        kappa_bias = self._get_kappa_bias_parameter()
        version = -1 if kappa_bias is None else kappa_bias._version
        if (
            self._eval_kappa_bias_unsqueezed_cache is not None
            and self._eval_kappa_bias_unsqueezed_cache_dtype == target_dtype
            and self._eval_kappa_bias_unsqueezed_cache_device == target_device
            and self._eval_kappa_bias_unsqueezed_cache_version == version
        ):
            return self._eval_kappa_bias_unsqueezed_cache
        cached = self._materialize_kappa_bias_for_eval(target_dtype, target_device).unsqueeze(1)
        self._eval_kappa_bias_unsqueezed_cache = cached
        self._eval_kappa_bias_unsqueezed_cache_dtype = target_dtype
        self._eval_kappa_bias_unsqueezed_cache_device = target_device
        self._eval_kappa_bias_unsqueezed_cache_version = version
        return cached

    @torch._dynamo.disable
    def _get_kappa_scale_unsqueezed_for_eval(self, target_dtype, target_device):
        if not self.use_kappa_scale:
            return self.disabled_kappa_scale.to(device=target_device, dtype=target_dtype).unsqueeze(1)
        kappa_scale = self._get_kappa_scale_parameter()
        if kappa_scale is None:
            raise RuntimeError("kappa_scale was enabled but no parameter was bound")
        version = kappa_scale._version
        if (
            self._eval_kappa_scale_unsqueezed_cache is not None
            and self._eval_kappa_scale_unsqueezed_cache_dtype == target_dtype
            and self._eval_kappa_scale_unsqueezed_cache_device == target_device
            and self._eval_kappa_scale_unsqueezed_cache_version == version
        ):
            return self._eval_kappa_scale_unsqueezed_cache
        cached = self._materialize_kappa_scale_for_eval(target_dtype, target_device).unsqueeze(1)
        self._eval_kappa_scale_unsqueezed_cache = cached
        self._eval_kappa_scale_unsqueezed_cache_dtype = target_dtype
        self._eval_kappa_scale_unsqueezed_cache_device = target_device
        self._eval_kappa_scale_unsqueezed_cache_version = version
        return cached

    @torch._dynamo.disable
    def _get_log_kappa_slope_max_scale_for_eval(self, target_dtype, target_device):
        version = self.kappa_slope_max_scale._version
        if (
            self._eval_log_kappa_slope_max_scale_cache is not None
            and self._eval_log_kappa_slope_max_scale_cache_dtype == target_dtype
            and self._eval_log_kappa_slope_max_scale_cache_device == target_device
            and self._eval_log_kappa_slope_max_scale_cache_version == version
        ):
            return self._eval_log_kappa_slope_max_scale_cache
        cached = torch.log(self.kappa_slope_max_scale.to(device=target_device, dtype=target_dtype))
        self._eval_log_kappa_slope_max_scale_cache = cached
        self._eval_log_kappa_slope_max_scale_cache_dtype = target_dtype
        self._eval_log_kappa_slope_max_scale_cache_device = target_device
        self._eval_log_kappa_slope_max_scale_cache_version = version
        return cached

    def _apply_kappa_slope_scaled_activation_training(
        self,
        gate_out_raw,
        kappa_bias,
        selected_router_scores,
        kappa_scale=None,
    ):
        target_dtype = torch.float32
        kappa_bias = kappa_bias.to(dtype=target_dtype).unsqueeze(1)
        slope_work = selected_router_scores.to(dtype=target_dtype).unsqueeze(-1)
        kappa_slope_max_scale = self.kappa_slope_max_scale.to(device=kappa_bias.device, dtype=target_dtype)
        if self.kappa_input in {'top_logits', 'router_probs'}:
            if kappa_scale is None:
                kappa_scale = self._materialize_kappa_scale()
            kappa_scale = kappa_scale.to(dtype=target_dtype).unsqueeze(1)
            slope_work = torch.addcmul(kappa_bias, slope_work, kappa_scale)
        else:
            slope_work = slope_work * kappa_bias
        slope_work = torch.exp(torch.log(kappa_slope_max_scale) * torch.tanh(2.0 * slope_work))
        self._update_kappa_slope_scale_stats(slope_work, selected_router_scores)
        slope_work = slope_work.to(dtype=gate_out_raw.dtype)
        return gate_out_raw * torch.sigmoid(gate_out_raw * slope_work)

    def _apply_kappa_slope_scaled_activation_inference(
        self,
        gate_out_raw,
        selected_router_scores,
    ):
        target_dtype = gate_out_raw.dtype
        kappa_bias = self._get_kappa_bias_unsqueezed_for_eval(target_dtype, gate_out_raw.device)
        slope_work = selected_router_scores.to(dtype=target_dtype).unsqueeze(-1)
        log_kappa_slope_max_scale = self._get_log_kappa_slope_max_scale_for_eval(
            target_dtype,
            kappa_bias.device,
        )
        if self.kappa_input in {'top_logits', 'router_probs'}:
            kappa_scale = self._get_kappa_scale_unsqueezed_for_eval(target_dtype, kappa_bias.device)
            slope_work = torch.addcmul(kappa_bias, slope_work, kappa_scale)
        else:
            slope_work = slope_work * kappa_bias
        slope_work.tanh_()
        slope_work.mul_(log_kappa_slope_max_scale)
        slope_work.exp_()
        self._update_kappa_slope_scale_stats(slope_work, selected_router_scores)
        return gate_out_raw * torch.sigmoid(gate_out_raw * slope_work)

    def _apply_kappa_slope_scaled_activation(
        self,
        gate_out_raw,
        kappa_bias,
        selected_router_scores,
        kappa_scale=None,
    ):
        if self.training:
            return self._apply_kappa_slope_scaled_activation_training(
                gate_out_raw,
                kappa_bias,
                selected_router_scores,
                kappa_scale=kappa_scale,
            )
        return self._apply_kappa_slope_scaled_activation_inference(
            gate_out_raw,
            selected_router_scores,
        )

    def set_kappa_bias_ema_rms_reg_step(self, step):
        self.kappa_bias_ema_rms_reg_step.fill_(int(step))

    def set_kappa_slope_max_scale(self, kappa_slope_max_scale):
        self.kappa_slope_max_scale.fill_(float(kappa_slope_max_scale))

    def set_kappa_bias_ema_rms_reg_total_iterations(self, total_iterations):
        if self.kappa_bias_ema_rms_reg_keeper is not None:
            self.kappa_bias_ema_rms_reg_keeper.set_total_iterations(total_iterations)
        if self.kappa_scale_ema_rms_reg_keeper is not None:
            self.kappa_scale_ema_rms_reg_keeper.set_total_iterations(total_iterations)

    def _kappa_bias_debug_source(self, suffix):
        owner = self.__class__.__name__
        layer = "unknown" if self.layer_idx is None else str(self.layer_idx)
        granularity = self.global_kappa_bias_granularity
        return f"{owner}(layer={layer}, granularity={granularity}).{suffix}"

    @torch._dynamo.disable
    def _accumulate_kappa_bias_l2_losses(self, kappa_bias):
        kappa_bias = kappa_bias.float()
        MANAGER.add(
            "kappa_bias_l2_loss",
            kappa_bias.square().mean(),
        )
        if self.kappa_bias_ema_rms_reg_keeper is not None:
            self.kappa_bias_ema_rms_reg_keeper.update(
                kappa_bias,
                int(self.kappa_bias_ema_rms_reg_step.item()),
                source=self._kappa_bias_debug_source("kappa_bias"),
            )
            MANAGER.add(
                "kappa_bias_ema_rms_reg_loss",
                self.kappa_bias_ema_rms_reg_keeper.loss(
                    kappa_bias,
                    source=self._kappa_bias_debug_source("kappa_bias"),
                ),
            )

    @torch._dynamo.disable
    def _accumulate_kappa_scale_l2_losses(self, kappa_scale):
        kappa_scale = kappa_scale.float()
        MANAGER.add(
            "kappa_scale_l2_loss",
            kappa_scale.square().mean(),
        )
        if self.kappa_scale_ema_rms_reg_keeper is not None:
            self.kappa_scale_ema_rms_reg_keeper.update(
                kappa_scale,
                int(self.kappa_bias_ema_rms_reg_step.item()),
                source=self._kappa_bias_debug_source("kappa_scale"),
            )
            MANAGER.add(
                "kappa_scale_ema_rms_reg_loss",
                self.kappa_scale_ema_rms_reg_keeper.loss(
                    kappa_scale,
                    source=self._kappa_bias_debug_source("kappa_scale"),
                ),
            )

    def _update_gate_stats(self, gate_out_acts):
        if not MANAGER.collect_load_balancing_stats:
            self.last_gate_stats = None
            return
        with torch.no_grad():
            abs_gate = gate_out_acts.detach().abs()
            topk = min(self.gate_stats_topk, abs_gate.size(-1))
            abs_gate_sum = abs_gate.sum(dim=-1, dtype=torch.float32)
            safe_abs_gate_sum = abs_gate_sum.clamp_min(1e-8)
            # xlogx: torch.xlogy(abs_gate, abs_gate)
            entropy = safe_abs_gate_sum.log() - (
                torch.xlogy(abs_gate, abs_gate).sum(dim=-1, dtype=torch.float32) / safe_abs_gate_sum
            )
            # abs_gate: [n_exp, capacity, intermediate_size].
            # topk: 16. topk_share is the sum of the top-16 gate values
            # divided by the sum of all gate values, averaged across tokens.
            # It measures how concentrated the top gates are.
            topk_share = (
                abs_gate.topk(topk, dim=-1).values.sum(dim=-1, dtype=torch.float32)
                / safe_abs_gate_sum
            )
            self.last_gate_stats = {
                'mean_abs_gate': abs_gate.float().mean().detach(),
                'active_frac': abs_gate.gt(self.gate_stats_threshold).float().mean().detach(),
                'topk_share': topk_share.mean().detach(),
                'entropy': entropy.mean().detach(),
            }

    @torch._dynamo.disable
    def _update_kappa_slope_scale_stats(self, slope_scales, selected_router_scores):
        if (
            not MANAGER.collect_load_balancing_stats
            or not self.use_kappa_swiglu
            or selected_router_scores is None
        ):
            return

        active_mask = selected_router_scores.detach().float().abs() > 0
        if not active_mask.any():
            return

        slope_scales = slope_scales.detach().float()
        active_mask_f = active_mask.to(dtype=slope_scales.dtype)
        active_token_counts = active_mask_f.sum(dim=1)
        slope_scale_sum_per_expert = (slope_scales * active_mask_f.unsqueeze(-1)).sum(dim=(1, 2))
        slope_scale_mean_per_expert = slope_scale_sum_per_expert / (
            active_token_counts.clamp_min(1) * slope_scales.size(-1)
        )
        total_active_tokens = active_token_counts.sum()
        slope_scale_mean = (
            slope_scale_mean_per_expert * active_token_counts
        ).sum() / total_active_tokens.clamp_min(1)
        MANAGER.add("kappa_slope_scale_abs_mean", slope_scale_mean.detach())
        # slope_scales: [n_exp, capacity, intermediate_size]
        slope_scales_flat = slope_scales.reshape(slope_scales.size(0), -1)
        active_slope_mask_flat = active_mask.unsqueeze(-1).expand_as(slope_scales).reshape(slope_scales.size(0), -1)
        MANAGER.add(
            "kappa_slope_scale_abs_top5p_mean",
            _mean_extreme_percentile_per_row(
                slope_scales_flat,
                active_slope_mask_flat,
                fraction=0.05,
                largest=True,
            ).detach(),
        )
        MANAGER.add(
            "kappa_slope_scale_abs_bottom5p_mean",
            _mean_extreme_percentile_per_row(
                slope_scales_flat,
                active_slope_mask_flat,
                fraction=0.05,
                largest=False,
            ).detach(),
        )
        active_token_counts_sqrt = active_token_counts.sqrt().clamp_min(1e-8)
        total_active_tokens_sqrt = active_token_counts_sqrt.sum().clamp_min(1)
        normalized_slope_scale_mean = (
            slope_scale_mean_per_expert * active_token_counts_sqrt
        ).sum() / total_active_tokens_sqrt
        MANAGER.add("kappa_slope_scale_abs_mean_normalized", normalized_slope_scale_mean.detach())

    @torch._dynamo.disable
    def _update_implicit_gate_proj_bias_stats(self, x, router_weight, selected_router_scores):
        if (
            not MANAGER.collect_load_balancing_stats
            or not self.log_implicit_gate_proj_bias
            or selected_router_scores is None
            or router_weight is None
        ):
            return

        active_mask = selected_router_scores.detach().float().abs() > 0
        if not active_mask.any():
            return

        x = x.detach().float()
        router_weight = torch.nn.functional.normalize(router_weight.detach().float(), dim=1, eps=1e-12)
        x_norm = torch.nn.functional.normalize(x, dim=2, eps=1e-12)
        routed_token_router_cosine = (x_norm * router_weight.unsqueeze(1)).sum(dim=2)
        active_cosines = routed_token_router_cosine[active_mask]
        MANAGER.add(
            "routed_token_router_weight_cosine_mean",
            active_cosines.mean().detach(),
        )
        MANAGER.add(
            "routed_token_router_weight_cosine_top5p_mean",
            _mean_extreme_percentile(active_cosines, fraction=0.05, largest=True).detach(),
        )
        MANAGER.add(
            "routed_token_router_weight_cosine_bottom5p_mean",
            _mean_extreme_percentile(active_cosines, fraction=0.05, largest=False).detach(),
        )
        exp_gate_parallel_coeff = (self.gate_proj.detach().float() * router_weight.unsqueeze(-1)).sum(dim=1)
        input_parallel = (x * router_weight.unsqueeze(1)).sum(dim=2)
        MANAGER.add(
            "implicit_gate_proj_bias_top5p_mean",
            _mean_extreme_outer_product_percentile_per_row(
                input_parallel,
                exp_gate_parallel_coeff,
                active_mask,
                fraction=0.05,
                largest=True,
            ).detach(),
        )
        MANAGER.add(
            "implicit_gate_proj_bias_bottom5p_mean",
            _mean_extreme_outer_product_percentile_per_row(
                input_parallel,
                exp_gate_parallel_coeff,
                active_mask,
                fraction=0.05,
                largest=False,
            ).detach(),
        )

    def forward(self, x, selected_router_scores=None, router_weight=None):
        # x: [n_exp, capacity, hidden_size]
        # gate_out_raw: [n_exp, capacity, intermediate_size]
        # gate_out_acts: [n_exp, capacity, intermediate_size]
        gate_input = x
        gate_out_raw = torch.bmm(gate_input, self.gate_proj)
        if selected_router_scores is not None and self.use_kappa_swiglu:
            if self.training:
                kappa_bias = self._materialize_kappa_bias()
                self._accumulate_kappa_bias_l2_losses(kappa_bias)
            else:
                kappa_bias = self._materialize_kappa_bias_for_eval(
                    gate_out_raw.dtype,
                    gate_out_raw.device,
                )
            kappa_scale = None
            if self.training and self.use_kappa_scale:
                kappa_scale = self._materialize_kappa_scale()
                self._accumulate_kappa_scale_l2_losses(kappa_scale)
            scaled_selected_router_scores = scale_grad(
                selected_router_scores,
                self.router_confidence_gate_bias_grad_scale,
            )
            gate_out_acts = self._apply_kappa_slope_scaled_activation(
                gate_out_raw,
                kappa_bias,
                scaled_selected_router_scores,
                kappa_scale=kappa_scale,
            )
        else:
            gate_out_acts = self._apply_gate_activation(gate_out_raw)
        if selected_router_scores is not None:
            self._update_implicit_gate_proj_bias_stats(x, router_weight, selected_router_scores)
        self._update_gate_stats(gate_out_acts)

        fc_out = torch.bmm(x, self.c_fc)
        x = gate_out_acts * fc_out
        proj_out = torch.bmm(x, self.c_proj)

        if self.debug:
            breakpoint()

        return proj_out
    
class MOELayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.router = Router(config)
        self.debug = config.debug
        if getattr(config, 'use_qwen3_moe_mlp', False) and config.use_qwen3_moe_mlp:
            self.experts = Qwen3MLPExperts(config, layer_idx=layer_idx)
            self.use_qwen3_moe_mlp = True
        else:
            self.experts = MLPExperts(config)
            self.use_qwen3_moe_mlp = False

        self.n_exp = config.n_exp
        self.top_k = config.moe_top_k
        self.use_aux_loss = config.use_aux_loss
        self.kappa_input = getattr(config, 'kappa_input', 'router_probs')
        self.kappa_input_constant = getattr(config, 'kappa_input_constant', None)
        self.top_logit_norm_exponent = float(getattr(config, 'top_logit_norm_exponent', 0.0))
        self.top_logit_norm_eps = float(getattr(config, 'top_logit_norm_eps', 1e-4))
        self._expert_inputs_cache = None
        self._expert_inputs_cache_dtype = None
        self._expert_inputs_cache_device = None
        self._expert_inputs_cache_capacity = None
        self._expert_router_scores_cache = None
        self._expert_router_scores_cache_dtype = None
        self._expert_router_scores_cache_device = None
        self._expert_router_scores_cache_capacity = None

    def update_aux_free_load_balancing(self):
        self.router.update_aux_free_load_balancing()

    @torch._dynamo.disable
    def _build_expert_inputs(self, x_flat, flat_rank, exp_capacity, flat_token_indices, 
                             flat_top_k_indices, flat_router_scores, expert_inputs, 
                             expert_router_scores):
        valid_mask = flat_rank < exp_capacity
        valid_token_indices = flat_token_indices[valid_mask]
        valid_expert_indices = flat_top_k_indices[valid_mask]
        valid_ranks = flat_rank[valid_mask]
        expert_inputs[valid_expert_indices, valid_ranks] = x_flat[valid_token_indices]
        if expert_router_scores is not None:
            expert_router_scores[valid_expert_indices, valid_ranks] = flat_router_scores[valid_mask]

    @torch._dynamo.disable
    def _get_expert_router_scores_buffer(self, exp_capacity, target_dtype, target_device):
        # Safe only when each forward using this buffer is backpropped before reuse.
        if (
            self._expert_router_scores_cache is None
            or self._expert_router_scores_cache_dtype != target_dtype
            or self._expert_router_scores_cache_device != target_device
            or self._expert_router_scores_cache_capacity != exp_capacity
        ):
            self._expert_router_scores_cache = torch.empty(
                self.n_exp,
                exp_capacity,
                dtype=target_dtype,
                device=target_device,
            )
            self._expert_router_scores_cache_dtype = target_dtype
            self._expert_router_scores_cache_device = target_device
            self._expert_router_scores_cache_capacity = exp_capacity
        else:
            # The cache becomes graph-connected after the indexed writes below.
            # Reuse must start from a detached tensor so the next micro-step does
            # not try to continue the previous autograd graph through this buffer.
            self._expert_router_scores_cache = self._expert_router_scores_cache.detach()
        self._expert_router_scores_cache.zero_()
        return self._expert_router_scores_cache

    @torch._dynamo.disable
    def _get_expert_inputs_buffer(self, exp_capacity, target_dtype, target_device, hidden_size):
        # Safe only when each forward using this buffer is backpropped before reuse.
        if (
            self._expert_inputs_cache is None
            or self._expert_inputs_cache_dtype != target_dtype
            or self._expert_inputs_cache_device != target_device
            or self._expert_inputs_cache_capacity != exp_capacity
            or self._expert_inputs_cache.size(2) != hidden_size
        ):
            self._expert_inputs_cache = torch.empty(
                self.n_exp,
                exp_capacity,
                hidden_size,
                dtype=target_dtype,
                device=target_device,
            )
            self._expert_inputs_cache_dtype = target_dtype
            self._expert_inputs_cache_device = target_device
            self._expert_inputs_cache_capacity = exp_capacity
        else:
            # See _get_expert_router_scores_buffer: the cached dispatch tensor can
            # retain autograd history from the previous forward unless we detach it
            # before overwriting it for the next micro-step.
            self._expert_inputs_cache = self._expert_inputs_cache.detach()
        self._expert_inputs_cache.zero_()
        return self._expert_inputs_cache

    @torch._dynamo.disable
    def _combine_expert_outputs(self, x_flat, expert_outputs, flat_rank, exp_capacity, flat_token_indices, flat_top_k_indices, router_probs, rank):
        valid_mask = flat_rank < exp_capacity
        valid_token_indices = flat_token_indices[valid_mask]
        valid_expert_indices = flat_top_k_indices[valid_mask]
        valid_ranks = flat_rank[valid_mask]
        output_flat = torch.zeros_like(x_flat)
        gated_expert_outputs = expert_outputs[valid_expert_indices, valid_ranks]
        valid_router_probs = router_probs.view(-1)[valid_mask].unsqueeze(1).to(dtype=x_flat.dtype)
        weighted_outputs = gated_expert_outputs * valid_router_probs
        output_flat.scatter_add_(0, valid_token_indices.unsqueeze(1).expand_as(weighted_outputs), weighted_outputs)
        self._maybe_collect_load_balancing_stats(rank, valid_expert_indices, exp_capacity)
        return output_flat

    def _select_gate_confidence(self, top_k_scores, router_probs, x_flat=None, top_k_indices=None):
        if self.kappa_input == 'top_logits':
            if self.top_logit_norm_exponent <= 0.0:
                # No normalization.
                # top_logits are usually 3~4. * 0.3 -> 0.9~1.2. 
                # Similar as the default "constant" setting of 1.
                return 0.3 * top_k_scores
            if top_k_indices is None:
                raise RuntimeError(
                    "top_k_indices are required when top_logit_norm_exponent is enabled"
                )
            # For exp32-d10, router_weight_magnitudes_all has (min, max, std)
            # of (0.24, 0.83, 0.21).
            router_weight_magnitudes_all = torch.linalg.vector_norm(
                self.router.w_g.weight,
                ord=2,
                dim=-1,
                dtype=torch.float32,
            )
            smoothed_router_weight_magnitudes_all = torch.sqrt(
                router_weight_magnitudes_all.square() + self.top_logit_norm_eps
            )
            # Partial normalization using smoothed_router_weight_magnitudes below
            # leaves a residual ||w||^(1-alpha) factor.
            # Calibrate it back to unit scale using the detached layer-wide
            # average router-weight magnitude, so the correction is batch
            # independent while keeping relative per-expert magnitude effects.
            scale_compensation = smoothed_router_weight_magnitudes_all.pow(
                1.0 - self.top_logit_norm_exponent
            ).mean().detach()
            router_weight_magnitudes = router_weight_magnitudes_all[top_k_indices]
            # Witout the top_logit_norm_eps term, if router_weight_magnitudes is
            # close to zero, and top_logit_norm_exponent = 0.5, then
            # the grad w.r.t. router_weight_magnitudes will be huge.
            smoothed_router_weight_magnitudes = torch.sqrt(
                router_weight_magnitudes.square() + self.top_logit_norm_eps
            )
            # Router inputs are RMS-normalized, so each token has L2 norm sqrt(hidden_dim).
            # sqrt(1024) = 32.
            token_magnitude = math.sqrt(self.router.w_g.weight.size(-1))
            normalizer = token_magnitude * smoothed_router_weight_magnitudes.pow(
                self.top_logit_norm_exponent
            ) * scale_compensation

            # Empirically, the average cosine(token embeddings, router weights) is 0.15.
            # top_k_scores / normalizer is roughly the cosine similarity, i.e., ~ 0.15.
            # We should multiply it by 6 to get it to be close to 1.0.
            return 6 * top_k_scores / normalizer.to(dtype=top_k_scores.dtype)
        
        if self.kappa_input == 'router_probs':
            # When top_k = 2, router_probs are typically 0.5. * 2 -> 1.0.
            return router_probs * 2
        if self.kappa_input == 'constant':
            if self.kappa_input_constant is None:
                raise RuntimeError(
                    "kappa_input_constant must be set when kappa_input='constant'"
                )
            return torch.full_like(router_probs, self.kappa_input_constant)
        raise ValueError(
            f"Unsupported kappa_input: {self.kappa_input!r}"
        )

    def forward(self, x: torch.Tensor):
        # x: [64, 2048, 512]
        B, T, C = x.size() # Keep track of original shape

        # --- Get routing information ---
        # Call the router with the ORIGINAL 3D tensor. The router will handle flattening internally
        # and return routing info shaped for a flattened list of tokens.
        # top_k_scores: [B*T, k], raw selected router scores and one possible
        # gate-bias confidence input. router_probs is the other possible input.
        expert_mask, router_probs, top_k_scores, top_k_indices, rank = self.router(x)

        # expert_mask: [B*T, k, n_exp], router_probs/top_k_scores: [B*T, k], etc.
        # Now, flatten the input tensor for the dispatch operation
        x_flat = x.view(B * T, C)

        # --- Dispatch tokens to experts (the "scatter" part) ---
        exp_capacity = self.router.get_capacity(B * T)

        # Get the indices for the valid assignments that are within capacity
        flat_top_k_indices = top_k_indices.view(-1)
        flat_rank = rank.view(-1)
        flat_token_indices = torch.arange(B * T, device=x.device).repeat_interleave(self.top_k)

        expert_inputs = self._get_expert_inputs_buffer(
            exp_capacity,
            x_flat.dtype,
            x_flat.device,
            x_flat.size(1),
        )
        selected_gate_confidence = self._select_gate_confidence(
            top_k_scores,
            router_probs,
            x_flat=x_flat,
            top_k_indices=top_k_indices,
        )
        expert_router_scores = None
        if self.use_qwen3_moe_mlp:
            expert_router_scores = self._get_expert_router_scores_buffer(
                exp_capacity,
                selected_gate_confidence.dtype,
                selected_gate_confidence.device,
            )
        self._build_expert_inputs(
            x_flat,
            flat_rank,
            exp_capacity,
            flat_token_indices,
            flat_top_k_indices,
            selected_gate_confidence.view(-1),
            expert_inputs,
            expert_router_scores,
        )

        # --- Run experts ---
        expert_outputs = self.experts(
            expert_inputs,
            selected_router_scores=expert_router_scores,
            router_weight=self.router.w_g.weight,
        ) # [n_exp, exp_capacity, C]

        # --- Combine expert outputs (the "gather" part) ---
        output_flat = self._combine_expert_outputs(
            x_flat,
            expert_outputs,
            flat_rank,
            exp_capacity,
            flat_token_indices,
            flat_top_k_indices,
            router_probs,
            rank,
        )

        # Reshape output back to the original input shape
        return output_flat.view(B, T, C)

    @torch._dynamo.disable
    def _maybe_collect_load_balancing_stats(self, rank, valid_expert_indices, exp_capacity):
        if MANAGER.collect_load_balancing_stats:
            slot_served = (rank < exp_capacity)                     # [B*T, k]
            # Since k=2, drop_rate_per_k = [drop_rate_0_step, drop_rate_1_step].
            # drop_rate_0_step: fraction of tokens whose top-1 expert assignment overflowed capacity.
            # drop_rate_1_step: fraction of tokens whose top-2 expert assignment overflowed capacity.
            #LINK #routing_ranks
            # for top_k = 2:
            # if top-1 and top-2 both fit, the token is sent to both experts
            # if top-1 fits and top-2 overflows, only top-1 contributes
            # if top-1 overflows and top-2 fits, only top-2 contributes
            # if both overflow, the token gets no MoE contribution from that layer            
            drop_rate_per_k = (~slot_served).float().mean(dim=0)    # [k]
            MANAGER.add("drop_rate_per_ks", drop_rate_per_k.detach())
            # Derive expert utilities: fraction of buffers used per expert.
            expert_util_counts = torch.bincount(valid_expert_indices, minlength=self.n_exp).float()
            expert_utilities = expert_util_counts / exp_capacity  # [n_exp]
            MANAGER.add("expert_utilities", expert_utilities.detach())

class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")

        moe_layer_indices = set(get_moe_layer_indices(config))
        if not moe_layer_indices:
            # create normal transformer blocks
            blocks = nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)])
        else:
            # create transformer blocks, placing MoE blocks at the configured layer indices
            blocks = []
            for layer_idx in range(config.n_layer):
                use_moe = layer_idx in moe_layer_indices
                blocks.append(Block(config, layer_idx, use_moe=use_moe))
            blocks = nn.ModuleList(blocks)

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": blocks,
        })
        self.register_parameter("global_kappa_bias", None)
        self.register_parameter("global_kappa_scale", None)
        self._configure_kappa_bias_sharing()

        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embed_dim = kv_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def _configure_kappa_bias_sharing(self):
        if getattr(self.config, 'global_kappa_bias_granularity', 'per-gate') != 'global':
            return
        bias_enabled_modules = []
        bias_scale_enabled_modules = []
        for block in self.transformer.h:
            mlp = getattr(block, 'mlp', None)
            if isinstance(mlp, MOELayer):
                experts = getattr(mlp, 'experts', None)
                if isinstance(experts, Qwen3MLPExperts) and experts.use_kappa_swiglu:
                    bias_enabled_modules.append(experts)
                    if experts.use_kappa_scale:
                        bias_scale_enabled_modules.append(experts)
            elif isinstance(mlp, Qwen3MLP) and getattr(mlp, 'has_active_kappa_bias', mlp.use_kappa_swiglu):
                bias_enabled_modules.append(mlp)
        if not bias_enabled_modules:
            return
        self.global_kappa_bias = nn.Parameter(torch.empty(1))
        for module in bias_enabled_modules:
            module.bind_shared_kappa_bias(self.global_kappa_bias)
        if bias_scale_enabled_modules:
            self.global_kappa_scale = nn.Parameter(torch.empty(1))
            for module in bias_scale_enabled_modules:
                module.bind_shared_kappa_scale(self.global_kappa_scale)

    def compute_kappa_slope_magnitude_losses(self):
        device = self.transformer.wte.weight.device
        losses = {}
        for name in (
            'kappa_bias_l2_loss',
            'kappa_scale_l2_loss',
            'kappa_bias_ema_rms_reg_loss',
            'kappa_scale_ema_rms_reg_loss',
        ):
            value = MANAGER.aggregate(name)
            losses[name] = value if torch.is_tensor(value) else torch.zeros((), device=device)
            MANAGER.reset(name)
        return losses

    def set_kappa_bias_ema_rms_reg_step(self, step):
        step = int(step)
        for block in self.transformer.h:
            mlp = getattr(block, 'mlp', None)
            if isinstance(mlp, Qwen3MLP):
                mlp.set_kappa_bias_ema_rms_reg_step(step)
                continue
            experts = getattr(mlp, 'experts', None)
            if isinstance(experts, Qwen3MLPExperts):
                experts.set_kappa_bias_ema_rms_reg_step(step)

    def set_kappa_bias_ema_rms_reg_total_iterations(self, total_iterations):
        total_iterations = int(total_iterations)
        for block in self.transformer.h:
            mlp = getattr(block, 'mlp', None)
            if isinstance(mlp, Qwen3MLP):
                mlp.set_kappa_bias_ema_rms_reg_total_iterations(total_iterations)
                continue
            experts = getattr(mlp, 'experts', None)
            if isinstance(experts, Qwen3MLPExperts):
                experts.set_kappa_bias_ema_rms_reg_total_iterations(total_iterations)

    def set_kappa_slope_max_scales(self, moe_kappa_slope_max_scale=None, dense_kappa_slope_max_scale=None):
        for block in self.transformer.h:
            mlp = getattr(block, 'mlp', None)
            if isinstance(mlp, Qwen3MLP) and dense_kappa_slope_max_scale is not None:
                mlp.set_kappa_slope_max_scale(dense_kappa_slope_max_scale)
                continue
            experts = getattr(mlp, 'experts', None)
            if isinstance(experts, Qwen3MLPExperts) and moe_kappa_slope_max_scale is not None:
                experts.set_kappa_slope_max_scale(moe_kappa_slope_max_scale)

    def set_router_confidence_gate_bias_grad_scale(self, grad_scale):
        grad_scale = float(grad_scale)
        for block in self.transformer.h:
            mlp = getattr(block, 'mlp', None)
            if isinstance(mlp, MOELayer):
                experts = getattr(mlp, 'experts', None)
                if isinstance(experts, Qwen3MLPExperts):
                    experts.router_confidence_gate_bias_grad_scale.fill_(grad_scale)

    @torch.no_grad()
    def refresh_kappa_bias_references(self):
        for block in self.transformer.h:
            mlp = getattr(block, 'mlp', None)
            if isinstance(mlp, MOELayer):
                experts = getattr(mlp, 'experts', None)
                if isinstance(experts, Qwen3MLPExperts):
                    experts.snapshot_kappa_bias_reference()

    def _should_refresh_kappa_bias_references(self):
        return bool(getattr(self.config, 'refresh_kappa_bias_references', False))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if strict:
            state_dict = state_dict.copy()
            for name, param in self.state_dict().items():
                # Keep the model’s current value for these parameters 
                # if they are missing in the checkpoint, to avoid loading errors 
                # when changing kappa_bias configuration.
                if ('ema_rms_reg_keeper' in name or 'l2_target_keeper' in name) and name not in state_dict:
                    state_dict[name] = param.clone()
                elif 'kappa_scale' in name and name not in state_dict:
                    state_dict[name] = param.clone()
        load_result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self._should_refresh_kappa_bias_references():
            self.refresh_kappa_bias_references()
        return load_result

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero

            if isinstance(block.mlp, Qwen3MLP):
                torch.nn.init.uniform_(block.mlp.gate_proj.weight, -s, s)
                torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
                torch.nn.init.zeros_(block.mlp.c_proj.weight)
                kappa_bias = getattr(block.mlp, 'kappa_bias', None)
                if kappa_bias is not None:
                    torch.nn.init.zeros_(kappa_bias)
            elif isinstance(block.mlp, MLP):
                torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
                torch.nn.init.zeros_(block.mlp.c_proj.weight)
            elif isinstance(block.mlp, MOELayer):
                experts = block.mlp.experts
                if isinstance(experts, Qwen3MLPExperts):
                    torch.nn.init.uniform_(experts.gate_proj, -s, s)
                    torch.nn.init.uniform_(experts.c_fc, -s, s)
                    torch.nn.init.zeros_(experts.c_proj)
                else:
                    # Ordinary MLPExperts doesn't have gate_proj.
                    torch.nn.init.uniform_(experts.c_fc, -s, s)
                    torch.nn.init.zeros_(experts.c_proj)
        if self.global_kappa_bias is not None:
            torch.nn.init.zeros_(self.global_kappa_bias)
        if self.global_kappa_scale is not None:
            torch.nn.init.zeros_(self.global_kappa_scale)
        for block in self.transformer.h:
            mlp = getattr(block, 'mlp', None)
            if isinstance(mlp, MOELayer):
                experts = getattr(mlp, 'experts', None)
                if isinstance(experts, Qwen3MLPExperts) and experts.kappa_bias is not None:
                    torch.nn.init.zeros_(experts.kappa_bias)
                if isinstance(experts, Qwen3MLPExperts) and experts.kappa_scale is not None:
                    torch.nn.init.zeros_(experts.kappa_scale)
            
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)   # 1.0 => typical residual connections at init
        self.x0_lambdas.fill_(0.1)      # 0.1 => small initial weight for skip connection to input embedding

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init to zero so gates start at sigmoid(0) = 0.5, scaled by 2 -> 1.0 (neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)   # 1.0 => typical residual connections at init
        self.x0_lambdas.fill_(0.0)      # 0.0 => skip connection to input is disabled at init

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to bf16: optimizer can tolerate it and it saves memory
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

        if self._should_refresh_kappa_bias_references():
            self.refresh_kappa_bias_references()

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def get_moe_adjusted_scaling_params(self, n_exp, top_k):
        """Return MoE-adjusted scaling params: transformer matrices plus lm_head.

        For MoE models, expert parameters contribute only a routed-data-adjusted fraction.
        Dense parameters in the transformer stack and lm_head are always active.
        """
        n_params = 0
        seen = set()
        for name, param in self.named_parameters():
            if not (name.startswith('transformer.h.') or name.startswith('lm_head.')):
                continue
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)
            if 'experts' in name:
                # each expert is active for a fraction of tokens proportional to top_k / n_exp.
                # So they tend to be undertrained compared to dense parameters.
                # The sqrt is a heuristic to partially compensate for undertraining.
                n_params += param.numel() * math.sqrt(top_k / n_exp)
            else:
                n_params += param.numel()
        return n_params

    def set_aux_free_load_balancing(self, enabled, bias_update_speed=None):
        enabled = bool(enabled)
        self.config.use_aux_free_load_balancing = enabled
        self.config.use_aux_loss = not enabled
        if bias_update_speed is not None:
            self.config.aux_free_load_balancing_bias_update_speed = float(bias_update_speed)
        for block in self.transformer.h:
            mlp = getattr(block, 'mlp', None)
            if isinstance(mlp, MOELayer):
                mlp.router.set_aux_free_load_balancing(
                    enabled,
                    bias_update_speed=bias_update_speed,
                )

    def update_aux_free_load_balancing(self):
        for block in self.transformer.h:
            mlp = getattr(block, 'mlp', None)
            if isinstance(mlp, MOELayer):
                mlp.update_aux_free_load_balancing()

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0,
                        adam_betas=(0.8, 0.95), scalar_lr=0.5, muon_match_rms_adamw=False,
                        matrix_optimizer='aurora',
                        kappa_bias_lr_final_scale=1.0,
                        kappa_bias_lr_max_scale=1.0,
                        kappa_bias_delay_start_iterations=0,
                        kappa_bias_lr_warmup_iterations=1000):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        dense_matrix_params = []
        dense_nonmatrix_params = []
        moe_matrix_params = []
        moe_nonmatrix_params = []
        kappa_bias_params = []
        seen_param_ids = set()
        param_names = {}

        def append_param(target, param, name=None):
            if param is None:
                return
            param_id = id(param)
            if param_id in seen_param_ids:
                return
            seen_param_ids.add(param_id)
            if name is not None:
                param_names[param_id] = name
            target.append(param)

        def use_matrix_optimizer(param):
            return param.ndim >= 2

        for block_idx, block in enumerate(self.transformer.h):
            mlp = getattr(block, 'mlp', None)
            target_matrix_params = moe_matrix_params if isinstance(mlp, MOELayer) else dense_matrix_params
            target_nonmatrix_params = moe_nonmatrix_params if isinstance(mlp, MOELayer) else dense_nonmatrix_params
            for name, param in block.named_parameters():
                full_name = f'transformer.h.{block_idx}.{name}'
                if name.startswith('mlp.experts.kappa_bias') or name.startswith('mlp.kappa_bias'):
                    append_param(kappa_bias_params, param, full_name)
                elif not use_matrix_optimizer(param):
                    append_param(target_nonmatrix_params, param, full_name)
                else:
                    append_param(target_matrix_params, param, full_name)
        append_param(kappa_bias_params, self.global_kappa_bias, 'global_kappa_bias')
        append_param(kappa_bias_params, self.global_kappa_scale, 'global_kappa_scale')
        value_embeds_params = []
        for param in self.value_embeds.parameters():
            append_param(value_embeds_params, param)
        embedding_params = []
        for param in self.transformer.wte.parameters():
            append_param(embedding_params, param)
        lm_head_params = []
        for param in self.lm_head.parameters():
            append_param(lm_head_params, param)
        resid_params = []
        append_param(resid_params, self.resid_lambdas)
        x0_params = []
        append_param(x0_params, self.x0_lambdas)
        assert len(list(self.parameters())) == (
            len(dense_matrix_params) + len(dense_nonmatrix_params) +
            len(moe_matrix_params) + len(moe_nonmatrix_params) +
            len(kappa_bias_params) +
            len(embedding_params) + len(lm_head_params) + len(value_embeds_params) +
            len(resid_params) + len(x0_params)
        )

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = []
        param_groups.append(
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0)
        )
        param_groups.append(
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0)
        )
        param_groups.append(
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0)
        )
        param_groups.append(
            dict(kind='adamw', params=dense_nonmatrix_params + moe_nonmatrix_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0)
        )
        param_groups.append(
            dict(
                kind='adamw',
                name='kappa_bias',
                params=kappa_bias_params,
                debug_param_names=[param_names[id(p)] for p in kappa_bias_params],
                lr=0.0,
                base_lr=embedding_lr * dmodel_lr_scale,
                lr_scale_end=kappa_bias_lr_final_scale,
                lr_scale_max=kappa_bias_lr_max_scale,
                lr_scale_nolearn_iterations=kappa_bias_delay_start_iterations,
                lr_scale_warmup_iterations=kappa_bias_lr_warmup_iterations,
                betas=adam_betas,
                eps=1e-10,
                weight_decay=0.0,
            )
        )
        param_groups.append(
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0)
        )
        param_groups.append(
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0)
        )  # higher beta1 for x0
        if matrix_optimizer not in ('muon', 'aurora'):
            raise ValueError(f"Unsupported matrix_optimizer: {matrix_optimizer}")

        matrix_kind = matrix_optimizer
        matrix_lr_scaling = "match_rms_adamw" if muon_match_rms_adamw else "original"
        print0(f"{matrix_optimizer.capitalize()} LR scaling: {matrix_lr_scaling}")
        for shape in sorted({p.shape for p in dense_matrix_params}):
            group_params = [p for p in dense_matrix_params if p.shape == shape]
            group_param_names = [param_names[id(p)] for p in group_params]
            param_groups.append(dict(
                kind=matrix_kind, params=group_params, debug_param_names=group_param_names, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, pp_iterations=2, pp_beta=0.5, nesterov=True, weight_decay=weight_decay,
                chunk_size=2,
                match_rms_adamw=muon_match_rms_adamw,
            ))
        for shape in sorted({p.shape for p in moe_matrix_params}):
            group_params = [p for p in moe_matrix_params if p.shape == shape]
            group_param_names = [param_names[id(p)] for p in group_params]
            param_groups.append(dict(
                kind=matrix_kind, params=group_params, debug_param_names=group_param_names, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, pp_iterations=2, pp_beta=0.5, nesterov=True, weight_decay=weight_decay,
                chunk_size=2,
                match_rms_adamw=muon_match_rms_adamw,
            ))
        factory_map = {
            'muon': (DistMuonAdamW if ddp else MuonAdamW),
            'aurora': (DistAuroraAdamW if ddp else AuroraAdamW),
        }
        Factory = factory_map[matrix_optimizer]
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
            group["initial_weight_decay"] = group["weight_decay"]
        return optimizer

    # Adapted from nanoMoE's forward() method.
    # kv_cache hasn't been implemented in nanochat. So we can safely ignore it here.
    # loss_reduction is used in chat_rl.py ('mean') and loss_eval.py ('none') only.
    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx) # embed current token
        x = norm(x)
        x0 = x  # save initial normalized embedding for x0 residual
        ve_placeholder = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            if str(i) in self.value_embeds:
                ve = self.value_embeds[str(i)](idx)
            else:
                if ve_placeholder is None:
                    ve_placeholder = x.new_zeros(B, T, self.value_embed_dim)
                ve = ve_placeholder
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = None
        if targets is None:
            # Always compute logits for all positions at inference time (HuggingFace standard)
            logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
            logits = logits[..., :self.config.vocab_size] # slice to remove padding
            logits = SoftcapInPlace.apply(logits, softcap)

        losses = { 'ntp_loss': 0,
                   'aux_loss': 0,
                   'router_z_loss': 0,
                   'kappa_bias_l2_loss': 0,
                   'kappa_scale_l2_loss': 0,
                   'kappa_bias_ema_rms_reg_loss': 0,
                   'kappa_scale_ema_rms_reg_loss': 0,
                   'gate_grad_scale_mean': None,
                   'kappa_slope_scale_abs_top5p_mean': 0,
                   'kappa_slope_scale_abs_bottom5p_mean': 0,
                   'kappa_slope_scale_abs_mean': 0,
                   'kappa_slope_scale_abs_mean_normalized': 0,
                   'implicit_gate_proj_bias_top5p_mean': 0,
                   'implicit_gate_proj_bias_bottom5p_mean': 0,
                   'routed_token_router_weight_cosine_mean': 0,
                   'routed_token_router_weight_cosine_top5p_mean': 0,
                   'routed_token_router_weight_cosine_bottom5p_mean': 0,
                   'drop_rate_per_ks': None,
                   'expert_utilities': None,
                   'selected_scores': None,
                 }

        # If MANAGER.collect_load_balancing_stats is False, these will return None
        expert_utilities = MANAGER.aggregate("expert_utilities")
        losses['expert_utilities'] = expert_utilities.detach() if expert_utilities is not None else None
        MANAGER.reset("expert_utilities")
        drop_rate_per_ks = MANAGER.aggregate("drop_rate_per_ks")
        losses['drop_rate_per_ks'] = drop_rate_per_ks.detach() if drop_rate_per_ks is not None else None
        MANAGER.reset("drop_rate_per_ks")
        moe_layer_indices = get_moe_layer_indices(self.config)
        selected_scores = MANAGER.aggregate("selected_scores")
        losses['selected_scores'] = selected_scores.detach() if selected_scores is not None else None
        MANAGER.reset("selected_scores")
        gate_grad_scale_mean = MANAGER.aggregate("gate_grad_scale_mean")
        losses['gate_grad_scale_mean'] = (
            gate_grad_scale_mean.detach()
            if gate_grad_scale_mean is not None
            else None
        )
        MANAGER.reset("gate_grad_scale_mean")
        kappa_slope_scale_abs_top5p_mean = MANAGER.aggregate("kappa_slope_scale_abs_top5p_mean")
        losses['kappa_slope_scale_abs_top5p_mean'] = (
            kappa_slope_scale_abs_top5p_mean.detach()
            if kappa_slope_scale_abs_top5p_mean is not None
            else torch.zeros((), device=x.device)
        )
        MANAGER.reset("kappa_slope_scale_abs_top5p_mean")
        kappa_slope_scale_abs_bottom5p_mean = MANAGER.aggregate("kappa_slope_scale_abs_bottom5p_mean")
        losses['kappa_slope_scale_abs_bottom5p_mean'] = (
            kappa_slope_scale_abs_bottom5p_mean.detach()
            if kappa_slope_scale_abs_bottom5p_mean is not None
            else torch.zeros((), device=x.device)
        )
        MANAGER.reset("kappa_slope_scale_abs_bottom5p_mean")
        kappa_slope_scale_abs_mean = MANAGER.aggregate("kappa_slope_scale_abs_mean")
        losses['kappa_slope_scale_abs_mean'] = (
            kappa_slope_scale_abs_mean.detach()
            if kappa_slope_scale_abs_mean is not None
            else torch.zeros((), device=x.device)
        )
        MANAGER.reset("kappa_slope_scale_abs_mean")
        kappa_slope_scale_abs_mean_normalized = MANAGER.aggregate("kappa_slope_scale_abs_mean_normalized")
        losses['kappa_slope_scale_abs_mean_normalized'] = (
            kappa_slope_scale_abs_mean_normalized.detach()
            if kappa_slope_scale_abs_mean_normalized is not None
            else torch.zeros((), device=x.device)
        )
        MANAGER.reset("kappa_slope_scale_abs_mean_normalized")
        implicit_gate_proj_bias_top5p_mean = MANAGER.aggregate("implicit_gate_proj_bias_top5p_mean")
        losses['implicit_gate_proj_bias_top5p_mean'] = (
            implicit_gate_proj_bias_top5p_mean.detach()
            if implicit_gate_proj_bias_top5p_mean is not None
            else torch.zeros((), device=x.device)
        )
        MANAGER.reset("implicit_gate_proj_bias_top5p_mean")
        implicit_gate_proj_bias_bottom5p_mean = MANAGER.aggregate("implicit_gate_proj_bias_bottom5p_mean")
        losses['implicit_gate_proj_bias_bottom5p_mean'] = (
            implicit_gate_proj_bias_bottom5p_mean.detach()
            if implicit_gate_proj_bias_bottom5p_mean is not None
            else torch.zeros((), device=x.device)
        )
        MANAGER.reset("implicit_gate_proj_bias_bottom5p_mean")
        routed_token_router_weight_cosine_mean = MANAGER.aggregate("routed_token_router_weight_cosine_mean")
        losses['routed_token_router_weight_cosine_mean'] = (
            routed_token_router_weight_cosine_mean.detach()
            if routed_token_router_weight_cosine_mean is not None
            else torch.zeros((), device=x.device)
        )
        MANAGER.reset("routed_token_router_weight_cosine_mean")
        routed_token_router_weight_cosine_top5p_mean = MANAGER.aggregate("routed_token_router_weight_cosine_top5p_mean")
        losses['routed_token_router_weight_cosine_top5p_mean'] = (
            routed_token_router_weight_cosine_top5p_mean.detach()
            if routed_token_router_weight_cosine_top5p_mean is not None
            else torch.zeros((), device=x.device)
        )
        MANAGER.reset("routed_token_router_weight_cosine_top5p_mean")
        routed_token_router_weight_cosine_bottom5p_mean = MANAGER.aggregate("routed_token_router_weight_cosine_bottom5p_mean")
        losses['routed_token_router_weight_cosine_bottom5p_mean'] = (
            routed_token_router_weight_cosine_bottom5p_mean.detach()
            if routed_token_router_weight_cosine_bottom5p_mean is not None
            else torch.zeros((), device=x.device)
        )
        MANAGER.reset("routed_token_router_weight_cosine_bottom5p_mean")
        kappa_bias_layer_indices = []
        implicit_gate_proj_bias_layer_indices = []
        for layer_idx, block in enumerate(self.transformer.h):
            mlp = getattr(block, 'mlp', None)
            experts = getattr(mlp, 'experts', None)
            if isinstance(experts, Qwen3MLPExperts) and experts.use_kappa_swiglu:
                kappa_bias_layer_indices.append(layer_idx)
            if isinstance(experts, Qwen3MLPExperts) and experts.log_implicit_gate_proj_bias:
                implicit_gate_proj_bias_layer_indices.append(layer_idx)
            elif isinstance(mlp, Qwen3MLP) and getattr(mlp, 'has_active_kappa_bias', mlp.use_kappa_swiglu):
                kappa_bias_layer_indices.append(layer_idx)
        kappa_bias_layer_to_stats_idx = {
            layer_idx: stats_idx for stats_idx, layer_idx in enumerate(kappa_bias_layer_indices)
        }
        implicit_gate_proj_bias_layer_to_stats_idx = {
            layer_idx: stats_idx for stats_idx, layer_idx in enumerate(implicit_gate_proj_bias_layer_indices)
        }
        kappa_slope_scale_abs_top5p_mean = losses['kappa_slope_scale_abs_top5p_mean']
        kappa_slope_scale_abs_bottom5p_mean = losses['kappa_slope_scale_abs_bottom5p_mean']
        kappa_slope_scale_abs_top5p_count = (
            kappa_slope_scale_abs_top5p_mean.shape[0]
            if kappa_slope_scale_abs_top5p_mean.ndim > 0
            else 0
        )
        kappa_slope_scale_abs_bottom5p_count = (
            kappa_slope_scale_abs_bottom5p_mean.shape[0]
            if kappa_slope_scale_abs_bottom5p_mean.ndim > 0
            else 0
        )
        kappa_slope_scale_abs_mean_count = (
            kappa_slope_scale_abs_mean.shape[0]
            if kappa_slope_scale_abs_mean is not None and kappa_slope_scale_abs_mean.ndim > 0
            else 0
        )
        kappa_slope_scale_abs_mean_normalized_count = (
            kappa_slope_scale_abs_mean_normalized.shape[0]
            if kappa_slope_scale_abs_mean_normalized is not None and kappa_slope_scale_abs_mean_normalized.ndim > 0
            else 0
        )
        implicit_gate_proj_bias_top5p_count = (
            losses['implicit_gate_proj_bias_top5p_mean'].shape[0]
            if losses['implicit_gate_proj_bias_top5p_mean'].ndim > 0
            else 0
        )
        implicit_gate_proj_bias_bottom5p_count = (
            losses['implicit_gate_proj_bias_bottom5p_mean'].shape[0]
            if losses['implicit_gate_proj_bias_bottom5p_mean'].ndim > 0
            else 0
        )
        routed_token_router_weight_cosine_mean_count = (
            losses['routed_token_router_weight_cosine_mean'].shape[0]
            if losses['routed_token_router_weight_cosine_mean'].ndim > 0
            else 0
        )
        routed_token_router_weight_cosine_top5p_count = (
            losses['routed_token_router_weight_cosine_top5p_mean'].shape[0]
            if losses['routed_token_router_weight_cosine_top5p_mean'].ndim > 0
            else 0
        )
        routed_token_router_weight_cosine_bottom5p_count = (
            losses['routed_token_router_weight_cosine_bottom5p_mean'].shape[0]
            if losses['routed_token_router_weight_cosine_bottom5p_mean'].ndim > 0
            else 0
        )
        for layer_idx, kappa_bias_stats_idx in kappa_bias_layer_to_stats_idx.items():
            if kappa_bias_stats_idx < kappa_slope_scale_abs_mean_count:
                losses[f'kappa_slope_scale_abs_mean_{layer_idx}'] = (
                    kappa_slope_scale_abs_mean[kappa_bias_stats_idx].item()
                )
            if kappa_bias_stats_idx < kappa_slope_scale_abs_mean_normalized_count:
                losses[f'kappa_slope_scale_abs_mean_normalized_{layer_idx}'] = (
                    kappa_slope_scale_abs_mean_normalized[kappa_bias_stats_idx].item()
                )
            if kappa_bias_stats_idx < kappa_slope_scale_abs_top5p_count:
                losses[f'kappa_slope_scale_abs_top5p_mean_{layer_idx}'] = (
                    kappa_slope_scale_abs_top5p_mean[kappa_bias_stats_idx].item()
                )
            if kappa_bias_stats_idx < kappa_slope_scale_abs_bottom5p_count:
                losses[f'kappa_slope_scale_abs_bottom5p_mean_{layer_idx}'] = (
                    kappa_slope_scale_abs_bottom5p_mean[kappa_bias_stats_idx].item()
                )
        for layer_idx, implicit_stats_idx in implicit_gate_proj_bias_layer_to_stats_idx.items():
            if implicit_stats_idx < implicit_gate_proj_bias_top5p_count:
                losses[f'implicit_gate_proj_bias_top5p_mean_{layer_idx}'] = (
                    losses['implicit_gate_proj_bias_top5p_mean'][implicit_stats_idx].item()
                )
            if implicit_stats_idx < implicit_gate_proj_bias_bottom5p_count:
                losses[f'implicit_gate_proj_bias_bottom5p_mean_{layer_idx}'] = (
                    losses['implicit_gate_proj_bias_bottom5p_mean'][implicit_stats_idx].item()
                )
            if implicit_stats_idx < routed_token_router_weight_cosine_mean_count:
                losses[f'routed_token_router_weight_cosine_mean_{layer_idx}'] = (
                    losses['routed_token_router_weight_cosine_mean'][implicit_stats_idx].item()
                )
            if implicit_stats_idx < routed_token_router_weight_cosine_top5p_count:
                losses[f'routed_token_router_weight_cosine_top5p_mean_{layer_idx}'] = (
                    losses['routed_token_router_weight_cosine_top5p_mean'][implicit_stats_idx].item()
                )
            if implicit_stats_idx < routed_token_router_weight_cosine_bottom5p_count:
                losses[f'routed_token_router_weight_cosine_bottom5p_mean_{layer_idx}'] = (
                    losses['routed_token_router_weight_cosine_bottom5p_mean'][implicit_stats_idx].item()
                )
        for stats_idx, layer_idx in enumerate(moe_layer_indices):
            experts = getattr(self.transformer.h[layer_idx].mlp, 'experts', None)
            if not isinstance(experts, Qwen3MLPExperts) or experts.last_gate_stats is None:
                continue
            losses[f'mean_abs_gate_{layer_idx}'] = experts.last_gate_stats['mean_abs_gate'].item()
            losses[f'active_frac_gate_{layer_idx}'] = experts.last_gate_stats['active_frac'].item()
            losses[f'topk_share_gate_{layer_idx}'] = experts.last_gate_stats['topk_share'].item()
            losses[f'entropy_gate_{layer_idx}'] = experts.last_gate_stats['entropy'].item()
        
        if targets is not None:
            loss = _chunked_cross_entropy(
                x,
                targets,
                self.lm_head,
                self.config.vocab_size,
                softcap,
                loss_reduction,
                _get_loss_chunk_tokens(self.config, x.size(0) * x.size(1)),
                recompute_backward=bool(getattr(self.config, 'loss_recompute_backward', False)),
            )
            losses['ntp_loss'] = loss.detach()

            if self.config.n_exp > 1 and self.config.use_aux_loss:
                aux_loss = MANAGER.aggregate("aux_loss")
                losses['aux_loss'] = aux_loss
                MANAGER.reset("aux_loss")
            if self.config.n_exp > 1 and self.config.use_router_z_loss:
                router_z_loss = MANAGER.aggregate("router_z_loss")
                loss += self.config.router_z_loss_weight * router_z_loss
                losses['router_z_loss'] = router_z_loss.detach() if isinstance(router_z_loss, torch.Tensor) else router_z_loss
                MANAGER.reset("router_z_loss")

            # Updates losses['kappa_bias_l2_loss'] and losses['kappa_scale_l2_loss'].
            losses.update(self.compute_kappa_slope_magnitude_losses())
        else:
            return logits

        if False and self.global_iter >= 1000:
            self.debug_losses(losses, losses_to_debug=[self.config.router_z_loss_weight * router_z_loss])

        return loss, losses

    # Revised from collect_weight_grad_stats().
    def debug_losses(self, losses, losses_to_debug=[]):
        router_grad_norms = []
        router_grad_self_alignments = []
        router_weight_exp_alignments = []
        exp_gate_grad_norms = []
        expert_utilities = losses.get('expert_utilities', None)
        selected_scores = losses.get('selected_scores', None)
        moe_layer_indices = get_moe_layer_indices(self.config)
        moe_layer_to_stats_idx = {layer_idx: stats_idx for stats_idx, layer_idx in enumerate(moe_layer_indices)}

        for loss in losses_to_debug:
            if loss is not None and isinstance(loss, torch.Tensor):
                loss.backward(retain_graph=True)
            else:
                breakpoint()

        for i in moe_layer_indices:
            layer = self.transformer.h[i]
            if hasattr(layer.mlp, 'experts'):
                # [n_exp, hidden_size]
                router_gate_grad = layer.mlp.router.w_g.weight.grad
                router_grad_norm = router_gate_grad.norm(dim=1)
                router_grad_norms.append(router_grad_norm)
                losses[f'router_grad_norm_{i}'] = router_grad_norm.mean().item()
                exp_gate_grad = layer.mlp.experts.gate_proj.grad
                exp_gate_grad_norm = None if exp_gate_grad is None else torch.linalg.vector_norm(
                    exp_gate_grad,
                    dim=tuple(range(1, exp_gate_grad.ndim)),
                )
                if exp_gate_grad_norm is not None:
                    exp_gate_grad_norms.append(exp_gate_grad_norm)
                    losses[f'exp_gate_grad_norm_{i}'] = exp_gate_grad_norm.mean().item()

                # Compute router grad - router weight alignment
                # Compute router expert - gate weight alignment
                with torch.no_grad():
                    router_weight = layer.mlp.router.w_g.weight  # [n_exp, hidden_size]
                    exp_gate_weight = layer.mlp.experts.gate_proj
                    exp_gate_mean_weight = exp_gate_weight.mean(dim=2)  # [n_exp, hidden_size]
                    # Compute the cosine similarity between router weights and router weight grads.
                    # With SGD: Δw = -lr * ∇w. Since w·Δw = -lr*(w·∇w),
                    # -(w·∇w) is positive when the update has a component along w (tends to increase ||w||),
                    # and negative when it moves against w (tends to decrease ||w||). 
                    rg_rw_alignment = -(router_gate_grad * router_weight).sum(dim=1) / (
                        router_weight.norm(dim=1) * router_gate_grad.norm(dim=1) + 1e-10
                    )  # [n_exp]
                    router_grad_self_alignments.append(rg_rw_alignment)
                    mean_rg_rw_alignment = rg_rw_alignment.mean().item()
                    losses[f'router_grad_self_alignment_{i}'] = mean_rg_rw_alignment

                    # No negative sign here since these are weights, not gradients.
                    rw_ew_alignment = (exp_gate_mean_weight * router_weight).sum(dim=1) / \
                            (router_weight.norm(dim=1) * (exp_gate_mean_weight.norm(dim=1) + 1e-10)) # [n_exp]
                    router_weight_exp_alignments.append(rw_ew_alignment)
                    mean_rw_ew_alignment = rw_ew_alignment.mean().item()
                    losses[f'router_weight_exp_alignment_{i}'] = mean_rw_ew_alignment

                    if expert_utilities is not None:
                        # expert_utilities: Tensor of shape (num_moe_layers, n_exp)
                        exp_utilities = expert_utilities[moe_layer_to_stats_idx[i]]  # [n_exp]
                        half_experts = exp_utilities.shape[0] // 2
                        top_indices    = torch.topk(exp_utilities, k=half_experts, largest=True).indices
                        bottom_indices = torch.topk(exp_utilities, k=half_experts, largest=False).indices

                        top_rg_rw_alignment    = rg_rw_alignment[top_indices].mean().item()
                        bottom_rg_rw_alignment = rg_rw_alignment[bottom_indices].mean().item()
                        losses[f'router_grad_self_alignment_top_{i}']    = top_rg_rw_alignment
                        losses[f'router_grad_self_alignment_bottom_{i}'] = bottom_rg_rw_alignment

                        top_rw_ew_alignment    = rw_ew_alignment[top_indices].mean().item()
                        bottom_rw_ew_alignment = rw_ew_alignment[bottom_indices].mean().item()
                        losses[f'router_weight_exp_alignment_top_{i}']    = top_rw_ew_alignment
                        losses[f'router_weight_exp_alignment_bottom_{i}'] = bottom_rw_ew_alignment

                        top_router_grad_norm    = router_grad_norm[top_indices].mean().item()
                        bottom_router_grad_norm = router_grad_norm[bottom_indices].mean().item()
                        losses[f'router_grad_norm_top_{i}']    = top_router_grad_norm
                        losses[f'router_grad_norm_bottom_{i}'] = bottom_router_grad_norm

                        if selected_scores is not None:
                            # selected_scores: Tensor of shape (num_moe_layers, n_exp)
                            layer_selected_scores = selected_scores[moe_layer_to_stats_idx[i]]  # [n_exp]
                            top_selected_scores    = layer_selected_scores[top_indices].mean().item()
                            bottom_selected_scores = layer_selected_scores[bottom_indices].mean().item()
                            losses[f'selected_scores_top_{i}']    = top_selected_scores
                            losses[f'selected_scores_bottom_{i}'] = bottom_selected_scores

        router_grad_norms = torch.stack(router_grad_norms, dim=0) if router_grad_norms else None
        losses['router_grad_norms'] = router_grad_norms
        router_grad_self_alignments = torch.stack(router_grad_self_alignments, dim=0) if router_grad_self_alignments else None
        losses['router_grad_self_alignments'] = router_grad_self_alignments
        router_weight_exp_alignments = torch.stack(router_weight_exp_alignments, dim=0) if router_weight_exp_alignments else None
        losses['router_weight_exp_alignments'] = router_weight_exp_alignments
        exp_gate_grad_norms = torch.stack(exp_gate_grad_norms, dim=0) if exp_gate_grad_norms else None
        losses['exp_gate_grad_norms'] = exp_gate_grad_norms
        breakpoint()

    # nanochat's generate() is almost identical to nanoMoE's generate(). We only keep nanoMoE's version here.
    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of GPU bfloat16 -> fp32 accum peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.sequence_len
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        # Determine the theoretical peak FLOPs of the current device using a simple lookup.
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0).lower()

            # Very small lookup table of common GPUs and their BF16/FP16 peak throughput (in FLOPs).
            # TODO: add more GPUs
            flops_table = {
                "3090": 71e12,   # RTX 3090
                "4090": 165e12,  # RTX 4090
                "l40s": 362e12,  # L40S
                "a100": 312e12,  # A100 80GB
                "h100": 990e12,  # H100
                "h200": 990e12,  # H200 (assumed same as H100 for BF16/FP16)
                "5070 ti": 176e12,  # RTX 5070 Ti
                "5080": 225e12,  # RTX 5080
                "b200": 2250e12,  # B200
                "rtx 6000 ada": 364e12,
                "rtx a6000": 155e12,   # dense tensor (BF16/FP16) approx; datasheet tensor is 309.7 TFLOPS with sparsity
            }

            # Pick the first entry whose key is a substring of the device name; fall back to 0.
            flops_promised = next((v for k, v in flops_table.items() if k in device_name), 0)
        else:
            # If running on CPU or an unknown accelerator, return -1 
            flops_promised = -1
        try:
            mfu = flops_achieved / flops_promised
        except:
            breakpoint()
        return mfu

    def get_num_active_params(self, n_exp, top_k):
        """
        Return the number of active parameters in the model.
        Active parameters are those that are used during a forward pass.
        In MoE models, only a subset of expert parameters are active per token.
        """
        n_params = 0
        # seen: avoid double-counting tied parameters.
        seen = set()
        for name, param in self.named_parameters():
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)
            if 'experts' in name:
                n_params += param.numel() * top_k / n_exp
            else:
                # Non-expert parameters are always active
                n_params += param.numel()
        return n_params
    
