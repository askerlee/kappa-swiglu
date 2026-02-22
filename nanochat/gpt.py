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
import torch.nn as nn
from torch.nn import functional as F

try:
    from .manager import MANAGER
except ImportError:
    from manager import MANAGER
from transformers.activations import SiLUActivation
from nanochat.common import get_dist_info, print0
from nanochat.optim import MuonAdamW, DistMuonAdamW
# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn

# Revised from RevGrad, by removing the grad negation.
class ScaleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, alpha_, debug=False):
        ctx.save_for_backward(alpha_, debug)
        output = input_
        if debug:
            print(f"input: {input_.abs().mean().detach().item()}")
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        # saved_tensors returns a tuple of tensors.
        alpha_, debug = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_output2 = grad_output * alpha_
            if debug:
                print(f"grad_output2: {grad_output2.abs().mean().detach().item()}")
        else:
            grad_output2 = None
        return grad_output2, None, None

class ReuseBmmWithScaledInputGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, left, right, alpha):
        ctx.save_for_backward(left, right, alpha)
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        left, right, alpha = ctx.saved_tensors

        grad_output_for_output = None

        if ctx.needs_input_grad[1]:
            grad_left = torch.bmm(grad_output, right.transpose(1, 2)) * alpha
        else:
            grad_left = None

        if ctx.needs_input_grad[2]:
            grad_right = torch.bmm(left.transpose(1, 2), grad_output)
        else:
            grad_right = None

        return grad_output_for_output, grad_left, grad_right, None

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
            # output = left @ right.T  => dleft = grad_output @ right
            grad_left = torch.mm(grad_output, right) * alpha
        else:
            grad_left = None

        if ctx.needs_input_grad[2]:
            # output = left @ right.T  => dright = grad_output.T @ left
            grad_right = torch.mm(grad_output.transpose(0, 1), left)
        else:
            grad_right = None

        return grad_output_for_output, grad_left, grad_right, None

class GradientScaler(nn.Module):
    def __init__(self, alpha=1., debug=False, *args, **kwargs):
        """
        A gradient scaling layer.
        This layer has no parameters, and simply scales the gradient in the backward pass.
        """
        super().__init__(*args, **kwargs)

        # Store as Python scalars to avoid meta tensor buffers during lazy/meta init.
        self._alpha = float(alpha)
        self._debug = bool(debug)

    def forward(self, input_):
        _debug = self._debug if hasattr(self, '_debug') else False
        alpha_t = torch.as_tensor(self._alpha, device=input_.device, dtype=input_.dtype)
        debug_t = torch.as_tensor(_debug, device=input_.device)
        return ScaleGrad.apply(input_, alpha_t, debug_t)

def gen_gradient_scaler(alpha, debug=False):
    if alpha == 1:
        return nn.Identity()
    if alpha > 0:
        return GradientScaler(alpha, debug=debug)
    else:
        assert alpha == 0
        # Don't use lambda function here, otherwise the object can't be pickled.
        return torch.detach

def compute_z_loss(logits: torch.Tensor, demean_logits: bool = True, 
                   z_loss_penalize_mean_logits: bool = True):
    """
    Computes ST-MoE router z loss (https://arxiv.org/abs/2202.08906)
    See equation (5) on page 7
    """

    # exponentiate logits, sum logits of each expert, take log, and square
    # code below is the same as:
    # > z_loss = torch.exp(logits)
    # > z_loss = torch.sum(z_loss, dim=-1)
    # > z_loss = torch.log(z_loss) ** 2.0
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
        self.use_router_z_loss      = config.use_router_z_loss
        self.z_loss_demean_logits = config.z_loss_demean_logits
        self.z_loss_penalize_mean_logits = config.z_loss_penalize_mean_logits
        # linear projection for (noisy) softmax gating
        # no bias is used, see page 4 eq (4) in (https://arxiv.org/abs/1701.06538)
        self.w_g = nn.Linear(config.n_embd, config.n_exp, bias=False)
        self.w_noise = nn.Linear(config.n_embd, config.n_exp, bias=False) if self.use_noisy_top_k else None
        self.router_z_loss_input_grad_scale = config.router_z_loss_input_grad_scale

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
            logits = self.w_g(x_flat)  # [B*T, n_exp]
            if self.training and self.use_noisy_top_k:
                noise = F.softplus(self.w_noise(x_flat))
                noise *= torch.randn_like(noise)
                logits += noise

            # 2. COMPUTE LOSSES (if training)
            # -------------------------------
            if self.training:
                # Router Z-loss prevents logits from growing too large
                if self.use_router_z_loss:
                    if self.router_z_loss_input_grad_scale == 1:
                        logits_for_z_loss = logits
                    else:
                        alpha_t = torch.as_tensor(self.router_z_loss_input_grad_scale, device=logits.device, dtype=logits.dtype)
                        # self.w_g(x_flat) => torch.mm(x_flat, self.w_g.weight.t())
                        # So the left is x_flat, the right is self.w_g.weight.
                        # In ReuseMmWithScaledInputGrad, we will scale the gradient 
                        # to the left (input) but not the right (weights), since we want to stabilize the
                        # input representations to the router.
                        logits_for_z_loss = ReuseMmWithScaledInputGrad.apply(logits, x_flat, self.w_g.weight, alpha_t)

                    router_z_loss = compute_z_loss(logits_for_z_loss.view(B, T, -1), 
                                                   demean_logits=self.z_loss_demean_logits,
                                                   z_loss_penalize_mean_logits=self.z_loss_penalize_mean_logits)
                    MANAGER.add("router_z_loss", router_z_loss)

                # Find top-k choices for each token
                top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1) # [B*T, k]
                
                # The auxiliary loss encourages load balancing across experts
                if self.use_aux_loss:
                    # To compute aux loss, we need the full probability distribution,
                    # not just for the top k. We can create this sparsely.
                    all_probs = torch.zeros_like(logits)
                    top_k_logits_for_aux, _ = logits_for_z_loss.topk(self.top_k, dim=-1) # [B*T, k]
                    top_k_probs = F.softmax(top_k_logits_for_aux, dim=-1).to(dtype=all_probs.dtype)
                    all_probs.scatter_(-1, top_k_indices, top_k_probs)
                    aux_loss = self.compute_aux_loss(all_probs.view(B, T, -1), top_k_indices.view(B, T, -1))
                    MANAGER.add("aux_loss", aux_loss)
            else:
                 # At inference, we just need the top-k
                 top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)


            selected_scores = self.compute_selected_scores(logits.view(B, T, -1), top_k_indices.view(B, T, -1))
            MANAGER.add("selected_scores", selected_scores.detach())

            # 3. COMPUTE ROUTER PROBABILITIES
            # --------------------------------
            # We normalize the probabilities over the top-k experts
            router_probs = F.softmax(top_k_logits, dim=-1) # [B*T, k]

            # 4. DETERMINE TOKEN RANKS WITH CAPACITY LIMITING
            # -----------------------------------------------
            exp_capacity = self.get_capacity(num_tokens)
            
            # Create a one-hot mask of the chosen experts for each token. Shape: [B*T, k, n_exp]
            expert_mask_one_hot = F.one_hot(top_k_indices, num_classes=self.n_exp)

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

            # The final rank is collapsed to a single value per top-k choice.
            # It adds across the expert dimension, since only one expert per top-k slot is selected,
            # and all other positions are zeros. 
            # NOTE: final_rank is derived from rank, so it also includes 
            # over-capacity positions.
            final_rank = torch.sum(rank, dim=-1) # [B*T, k]

            # The MOELayer will use these tensors to efficiently dispatch and combine tokens.
            # Their memory usage all scale linearly with (B * T).
            return final_expert_mask, router_probs_masked, top_k_indices, final_rank 
    
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
            self.mlp = MOELayer(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x

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
    
    def forward(self, x):
        x = torch.bmm(x, self.c_fc)
        x = F.relu(x).square()
        x = torch.bmm(x, self.c_proj)
        return x

# Borrowed Qwen3MoeMLP implementation from modeling_qwen3_moe.py.
class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.n_embd
        self.intermediate_size = 4 * config.n_embd
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # up_proj -> c_fc, down_proj -> c_proj
        # to ensure minimal code changes when switching between Qwen3MoeMLP and regular MLP.
        self.c_fc = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.c_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = SiLUActivation()

    def forward(self, x):
        down_proj = self.c_proj(self.act_fn(self.gate_proj(x)) * self.c_fc(x))
        return down_proj

class Qwen3MLPExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_exp = config.n_exp
        self.hidden_size = config.n_embd
        self.intermediate_size = 4 * config.n_embd

        self.gate_proj = nn.Parameter(torch.empty(self.n_exp, self.hidden_size, self.intermediate_size))
        self.c_fc = nn.Parameter(torch.empty(self.n_exp, self.hidden_size, self.intermediate_size))
        self.c_proj = nn.Parameter(torch.empty(self.n_exp, self.intermediate_size, self.hidden_size))

        self.act_fn = SiLUActivation()
        self.fc_bias = None
        self.proj_bias = None
        self.use_experts_gate_output_loss = config.use_experts_gate_output_loss
        self.z_loss_demean_logits = config.z_loss_demean_logits
        self.z_loss_penalize_mean_logits = config.z_loss_penalize_mean_logits
        self.experts_gate_output_loss_input_grad_scale = 0.1

    def forward(self, x):
        # gate_out: [n_exp, capacity, intermediate_size]
        gate_out = torch.bmm(x, self.gate_proj)
        if self.training and self.use_experts_gate_output_loss:
            if self.experts_gate_output_loss_input_grad_scale == 1:
                gate_out_gs = gate_out
            else:
                alpha_t = torch.as_tensor(self.experts_gate_output_loss_input_grad_scale, device=gate_out.device, dtype=gate_out.dtype)
                gate_out_gs = ReuseBmmWithScaledInputGrad.apply(gate_out, x, self.gate_proj, alpha_t)
            # gate_out_gs: [n_exp, capacity, intermediate_size]
            # We treat each (token-slot, intermediate-dim) pair as a routing decision over experts,
            # so expert dimension should be the final logits dimension.
            gate_out_gs = gate_out_gs.permute(1, 2, 0)  # [capacity, intermediate_size, n_exp]
            experts_gate_output_loss = (gate_out_gs ** 2).mean()
            MANAGER.add("experts_gate_output_loss", experts_gate_output_loss)

        fc_out = torch.bmm(x, self.c_fc)
        x = self.act_fn(gate_out) * fc_out
        x = torch.bmm(x, self.c_proj)
        return x
    
class MOELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = Router(config)
        if getattr(config, 'use_qwen3_moe_mlp', False) and config.use_qwen3_moe_mlp:
            self.experts = Qwen3MLPExperts(config)
            self.use_qwen3_moe_mlp = True
        else:
            self.experts = MLPExperts(config)
            self.use_qwen3_moe_mlp = False

        self.n_exp = config.n_exp
        self.top_k = config.moe_top_k
        self.use_router_ortho_loss = config.use_router_ortho_loss
        self.router_ortho_neg_corr_weight = config.router_ortho_neg_corr_weight
        self.router_ortho_loss_leave_one_out = config.router_ortho_loss_leave_one_out
        self.router_ortho_loss_grad_scale = config.router_ortho_loss_grad_scale
        # use_experts_ortho_loss: If set to True, compute experts ortho loss for ablation study.
        # But the computation is slow, so disabled by default.
        # We just don't optimize it unless the weight is set > 0 in the config.
        self.use_experts_ortho_loss = config.use_experts_ortho_loss
        self.use_experts_gate_output_loss = config.use_experts_gate_output_loss
        # scale down gradients back to expert weights by router_ortho_loss_grad_scale during router orthogonality loss computation.
        # Default: 1, no grad scaling.
        self.exp_gate_grad_scaler = gen_gradient_scaler(self.router_ortho_loss_grad_scale) 

    @torch._dynamo.disable
    def _build_expert_inputs(self, x_flat, flat_rank, exp_capacity, flat_token_indices, flat_top_k_indices, expert_inputs):
        valid_mask = flat_rank < exp_capacity
        valid_token_indices = flat_token_indices[valid_mask]
        valid_expert_indices = flat_top_k_indices[valid_mask]
        valid_ranks = flat_rank[valid_mask]
        expert_inputs[valid_expert_indices, valid_ranks] = x_flat[valid_token_indices]

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

    def forward(self, x: torch.Tensor):
        # x: [64, 2048, 512]
        B, T, C = x.size() # Keep track of original shape

        # --- Get routing information ---
        # Call the router with the ORIGINAL 3D tensor. The router will handle flattening internally
        # and return routing info shaped for a flattened list of tokens.
        expert_mask, router_probs, top_k_indices, rank = self.router(x)

        # expert_mask: [B*T, k, n_exp], router_probs: [B*T, k], etc.
        if self.training and self.use_router_ortho_loss:
            router_ortho_loss, router_ortho_losses_by_exp = self.compute_router_ortho_loss()
            # router_ortho_loss will be optimized, so we keep its computation graph.
            MANAGER.add("router_ortho_loss", router_ortho_loss)
            # router_ortho_losses_by_exp is only for logging, so we detach it.
            MANAGER.add("router_ortho_losses_by_exp", router_ortho_losses_by_exp.detach())
            # Always use gate diversity loss when using router orthogonality loss.
            projs_diversity_loss = self.compute_projs_diversity_loss()
            MANAGER.add("projs_diversity_loss", projs_diversity_loss)

        if self.training and self.use_experts_ortho_loss:
            experts_ortho_loss = self.compute_experts_ortho_loss()
            MANAGER.add("experts_ortho_loss", experts_ortho_loss)

        # Now, flatten the input tensor for the dispatch operation
        x_flat = x.view(B * T, C)

        # --- Dispatch tokens to experts (the "scatter" part) ---
        exp_capacity = self.router.get_capacity(B * T)

        # Get the indices for the valid assignments that are within capacity
        flat_top_k_indices = top_k_indices.view(-1)
        flat_rank = rank.view(-1)
        flat_token_indices = torch.arange(B * T, device=x.device).repeat_interleave(self.top_k)

        expert_inputs = torch.zeros(
            self.n_exp, exp_capacity, x_flat.size(1), dtype=x_flat.dtype, device=x_flat.device
        )
        self._build_expert_inputs(
            x_flat, flat_rank, exp_capacity, flat_token_indices, flat_top_k_indices, expert_inputs
        )

        # --- Run experts ---
        expert_outputs = self.experts(expert_inputs) # [n_exp, exp_capacity, C]

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
        if not MANAGER.collect_load_balancing_stats:
            return
        slot_served = (rank < exp_capacity)                     # [B*T, k]
        drop_rate_per_k = (~slot_served).float().mean(dim=0)    # [k]
        MANAGER.add("drop_rate_per_ks", drop_rate_per_k.detach())
        # Derive expert utilities: fraction of buffers used per expert.
        expert_util_counts = torch.bincount(valid_expert_indices, minlength=self.n_exp).float()
        expert_utilities = expert_util_counts / exp_capacity  # [n_exp]
        MANAGER.add("expert_utilities", expert_utilities.detach())


    def compute_router_ortho_loss(self):
        if not self.use_qwen3_moe_mlp:
            # Only apply orthogonality loss when using Qwen3-style MoE MLPs
            return 0, None
        else:
            # Compute orthogonality loss between router weight vectors and expert gate projection vectors
            router_weights = self.router.w_g.weight.unsqueeze(-1)  # [n_exp, n_embd, 1]
            # Totally cutting off gradients may not be optimal.
            # Scale down gradients to expert gate projection weights by alpha, 
            # allows adjusting expert weights slightly, without hurting representation learning too much.
            # gate_proj_weights: [n_exp, n_embd, intermediate_size]
            if self.router_ortho_loss_leave_one_out:
                # Only apply orthogonality loss to all but the first dimension of gate projection weights.
                gate_proj_weights = self.exp_gate_grad_scaler(self.experts.gate_proj[:, :, 1:])
            else:
                gate_proj_weights = self.exp_gate_grad_scaler(self.experts.gate_proj)  

            # ortho_losses: [n_exp, intermediate_size]
            ortho_losses_signed = (router_weights * gate_proj_weights).sum(dim=1)
            ortho_losses_weights = torch.ones_like(ortho_losses_signed)
            # Negative correlations could be more tolerated by setting router_ortho_neg_corr_weight < 1.
            ortho_losses_weights[ortho_losses_signed < 0] = self.router_ortho_neg_corr_weight       
            # Square the dot products to penalize both positive and negative correlations
            ortho_losses = ortho_losses_signed.square()
            # Change mean to sum, otherwise the loss is too small to have effect.
            # sum() is n_exp * intermediate_size times larger than mean()
            # n_exp = 128, intermediate_size = 2048, so the loss is 262144 times larger!!
            ortho_loss = (ortho_losses * ortho_losses_weights).sum()
            ortho_losses_by_exp = ortho_losses_signed.sum(dim=1) # [n_exp]
            return ortho_loss, ortho_losses_by_exp

    # use_rand_estimate: speed up diversity loss computation with stochastic estimate.
    def compute_projs_diversity_loss(self, use_rand_estimate=True, num_rand_probes=1):
        loss = 0

        if not self.use_qwen3_moe_mlp:
            # Only apply orthogonality loss when using Qwen3-style MoE MLPs
            return loss

        for proj_name in ('gate_proj', 'c_fc'):
            # G: [n_exp, n_embd, intermediate_size]
            G = getattr(self.experts, proj_name)
            # Row-normalize: normalize each row vector over intermediate_size
            G = G / (G.norm(dim=2, keepdim=True) + 1e-12)
            E, D, F = G.size()  # n_exp, n_embd, intermediate_size

            if use_rand_estimate:
                # Stochastic Hutchinson trace/Frobenius estimator.
                # 2 probs are not accurate, but slightly faster than the exact method.
                # On the long term, it can still provide useful signal 
                # to improve diversity and suppress collapse.
                K = num_rand_probes
                # Z: [E, D, K]  (±1)
                Z = torch.empty((E, D, K), device=G.device, dtype=torch.int8).random_(2)
                Z = (Z * 2 - 1).to(G.dtype)
                # gt_z = G^T Z: [E, F, K]
                gt_z = torch.bmm(G.transpose(1, 2), Z)
                # Ggt_z = G gt_z: [E, D, K]
                Ggt_z = torch.bmm(G, gt_z)
                Az = Ggt_z - Z
                est_frob2 = Az.square().sum(dim=1).mean(dim=1)  # [E]  sum over D, mean over K
                row_sim_per_expert = est_frob2 / (D * D - D)
            else:
                # Batched Gram: [n_exp, n_embd, n_embd]
                # This computes cosine similarity between all pairs of row vectors of 
                # each expert's gate projection matrix.
                gram = torch.bmm(G, G.transpose(1, 2))
                # Zero out diagonal without materializing eye per expert
                gram = gram - torch.diag_embed(torch.diagonal(gram, dim1=-2, dim2=-1))

                # Mean squared off-diagonal similarity per expert
                # Off-diagonal count = n_embd * n_embd - n_embd
                offdiag_sq_sum = gram.square().sum(dim=(1, 2))       # [n_exp]
                row_sim_per_expert = offdiag_sq_sum / (D * D - D)    # [n_exp]

            loss += row_sim_per_expert.mean()
            return loss
        
    # Compute orthogonality loss between expert weight matrices.
    # This is an ablation study of arXiv:2601.00457.
    def compute_experts_ortho_loss(self):
        if not self.use_qwen3_moe_mlp:
            return torch.tensor(0.0, device=self.experts.c_fc.device)

        W = self.experts.c_fc  # [n_exp, n_embd, 4*n_embd]
        n_exp = W.shape[0]
        if n_exp < 2:
            return W.new_zeros(())

        X = W.reshape(n_exp, -1).float()  # do math in fp32. long vector math is unstable in fp16/bf16.
        X = X / (X.norm(dim=1, keepdim=True) + 1e-12)  # normalize per expert

        G = X @ X.t()  # cosine-sim Gram matrix, diag ~ 1
        offdiag = torch.triu(G, diagonal=1)
        # penalize non-orthogonality without sign cancellation
        loss = (offdiag ** 2).mean()
        return loss.to(W.dtype)


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

        if config.n_exp == 1:
            # create normal transformer blocks
            blocks = nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)])
        else:
            # create transformer blocks, placing an MoE block every <stride> layers
            blocks = []
            for layer_idx in range(config.n_layer):
                use_moe = (layer_idx >= config.moe_start_layer) and ((layer_idx + 1) % config.stride == 0)
                blocks.append(Block(config, layer_idx, use_moe=use_moe))
            blocks = nn.ModuleList(blocks)

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": blocks,
        })

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

            if isinstance(block.mlp, MLP):
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

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, 
                        adam_betas=(0.8, 0.95), scalar_lr=0.5, muon_match_rms_adamw=False):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
        ]
        # Muon groups (matrix params, grouped by shape for stacking)
        muon_lr_scaling = "match_rms_adamw" if muon_match_rms_adamw else "original"
        print0(f"Muon LR scaling: {muon_lr_scaling}")
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
                match_rms_adamw=muon_match_rms_adamw,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
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
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        # Always compute logits for all positions (HuggingFace standard)
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        losses = { 'ntp_loss': 0,
                   'aux_loss': 0,
                   'router_z_loss': 0,
                   'router_ortho_loss': 0,
                   'experts_ortho_loss': 0, 
                   'experts_gate_output_loss': 0,
                   'projs_diversity_loss': 0,
                   'router_ortho_losses_by_exp': None,
                   'drop_rate_per_ks': None,
                   'expert_utilities': None,
                   'selected_scores':  None,
                 }

        # If MANAGER.collect_load_balancing_stats is False, these will return None
        expert_utilities = MANAGER.aggregate("expert_utilities")
        losses['expert_utilities'] = expert_utilities.detach() if expert_utilities is not None else None
        MANAGER.reset("expert_utilities")
        router_ortho_losses_by_exp = MANAGER.aggregate("router_ortho_losses_by_exp")
        losses['router_ortho_losses_by_exp'] = router_ortho_losses_by_exp.detach() if router_ortho_losses_by_exp is not None else None
        MANAGER.reset("router_ortho_losses_by_exp")
        drop_rate_per_ks = MANAGER.aggregate("drop_rate_per_ks")
        losses['drop_rate_per_ks'] = drop_rate_per_ks.detach() if drop_rate_per_ks is not None else None
        MANAGER.reset("drop_rate_per_ks")
        selected_scores = MANAGER.aggregate("selected_scores")
        losses['selected_scores'] = selected_scores.detach() if selected_scores is not None else None
        MANAGER.reset("selected_scores")
        
        if targets is not None:
            # Compute loss when targets are provided
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            losses['ntp_loss'] = loss.detach()

            # add the auxiliary load balancing loss and router z loss to the main loss
            if self.config.n_exp > 1 and self.config.use_aux_loss:
                aux_loss = MANAGER.aggregate("aux_loss")
                loss += self.config.aux_loss_weight * aux_loss
                losses['aux_loss'] = aux_loss.detach() if isinstance(aux_loss, torch.Tensor) else aux_loss
                MANAGER.reset("aux_loss")
            if self.config.n_exp > 1 and self.config.use_router_z_loss:
                router_z_loss = MANAGER.aggregate("router_z_loss")
                # router_z_loss_weight: default 1e-5.
                loss += self.config.router_z_loss_weight * router_z_loss
                losses['router_z_loss'] = router_z_loss.detach() if isinstance(router_z_loss, torch.Tensor) else router_z_loss
                MANAGER.reset("router_z_loss")
            if self.config.n_exp > 1 and self.config.use_router_ortho_loss:
                router_ortho_loss = MANAGER.aggregate("router_ortho_loss")
                loss += self.config.router_ortho_loss_weight * router_ortho_loss 
                losses['router_ortho_loss'] = router_ortho_loss.detach() if isinstance(router_ortho_loss, torch.Tensor) else router_ortho_loss
                MANAGER.reset("router_ortho_loss")
                projs_diversity_loss = MANAGER.aggregate("projs_diversity_loss")
                loss += self.config.projs_diversity_loss_weight * projs_diversity_loss
                losses['projs_diversity_loss'] = projs_diversity_loss.detach() if isinstance(projs_diversity_loss, torch.Tensor) else projs_diversity_loss
                MANAGER.reset("projs_diversity_loss")
            if self.config.n_exp > 1 and self.config.use_experts_ortho_loss:
                experts_ortho_loss = MANAGER.aggregate("experts_ortho_loss")
                loss += self.config.experts_ortho_loss_weight * experts_ortho_loss
                losses['experts_ortho_loss'] = experts_ortho_loss.detach() if isinstance(experts_ortho_loss, torch.Tensor) else experts_ortho_loss
                MANAGER.reset("experts_ortho_loss")
            if self.config.n_exp > 1 and self.config.use_experts_gate_output_loss:
                experts_gate_output_loss = MANAGER.aggregate("experts_gate_output_loss")
                loss += self.config.experts_gate_output_loss_weight * experts_gate_output_loss
                losses['experts_gate_output_loss'] = experts_gate_output_loss.detach() if isinstance(experts_gate_output_loss, torch.Tensor) else experts_gate_output_loss
                MANAGER.reset("experts_gate_output_loss")
        else:
            # inference: just return the logits directly
            return logits

        if False and self.global_iter >= 1000:
            # To debug router z loss, we need the properly weighted, un-detached loss to do manual backward.
            self.debug_losses(losses, losses_to_debug=[self.config.router_z_loss_weight * router_z_loss])

        return loss, losses

    # Revised from collect_grad_stats().
    def debug_losses(self, losses, losses_to_debug=[]):
        router_grad_norms = []
        router_grad_self_alignments = []
        router_weight_exp_alignments = []
        exp_gate_grad_norms = []
        expert_utilities = losses.get('expert_utilities', None)
        selected_scores = losses.get('selected_scores', None)

        for loss in losses_to_debug:
            if loss is not None and isinstance(loss, torch.Tensor):
                loss.backward(retain_graph=True)
            else:
                breakpoint()

        for i in range(self.config.moe_start_layer, self.config.n_layer):
            layer = self.transformer.h[i]
            if hasattr(layer.mlp, 'experts'):
                # [n_exp, hidden_size]
                router_gate_grad = layer.mlp.router.w_g.weight.grad
                router_grad_norm = router_gate_grad.norm(dim=1)
                router_grad_norms.append(router_grad_norm)
                losses[f'router_grad_norm_{i}'] = router_grad_norm.mean().item()
                exp_gate_grad = layer.mlp.experts.gate_proj.grad
                if exp_gate_grad is not None:
                    exp_gate_grad_norm = exp_gate_grad.norm(dim=(1,2))
                    exp_gate_grad_norms.append(exp_gate_grad_norm)
                    losses[f'exp_gate_grad_norm_{i}'] = exp_gate_grad_norm.mean().item()

                # Compute router grad - router weight alignment
                # Compute router expert - gate weight alignment
                with torch.no_grad():
                    router_weight = layer.mlp.router.w_g.weight  # [n_exp, hidden_size]
                    exp_gate_mean_weight = layer.mlp.experts.gate_proj.mean(dim=2)  # [n_exp, hidden_size]
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
                        exp_utilities = expert_utilities[i - self.config.moe_start_layer]  # [n_exp]
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
                            layer_selected_scores = selected_scores[i - self.config.moe_start_layer]  # [n_exp]
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
    
