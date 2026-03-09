"""
Configuration class for NanoMoE GPT models.
"""

from transformers import PretrainedConfig


class GPTConfig:    
    def __init__(
        self,
        sequence_len: int = 2048,
        vocab_size: int = 50304,  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
        n_layer: int = 8,
        n_head: int = 12,
        n_kv_head: int = None,  # if None, n_kv_head = n_head
        n_embd: int = 768,
        # MoE-related configs
        n_exp: int = 64,  # if n_exp = 1 we just use regular MLP layers
        moe_top_k: int = 2,  # renamed from top_k to avoid conflict with generation top_k
        use_aux_loss: bool = True,  # apply auxiliary loss (from Switch Transformer) in router
        use_router_z_loss: bool = True,  # apply router z loss (from ST-MoE)
        z_loss_demean_logits: bool = True,  # fix router z loss bug by removing mean of logits
        z_loss_penalize_mean_logits: bool = True,  # penalize mean logits in router z loss
        use_router_ortho_loss: bool = True,  # apply router orthogonality loss
        use_experts_ortho_loss: bool = False,  # Compute experts orthogonality loss for ablation study
        use_experts_gate_output_loss: bool = True,  # Always compute gate output regularization loss for ablation study
        use_noisy_top_k: bool = False,
        aux_loss_weight: float = 0.01,  # default setting from Switch Transformer (see top of page 8)
        router_z_loss_weight: float = 0.00001,  # Much smaller than the setting used in ST-MoE (see page 8 eq. 6)
        router_z_loss_input_grad_scale: float = 0.1,  # scale down gradients to router input when computing router z loss.
        router_wg_grad_scale: float = 1.0,  # scale gradients for router w_g weights without affecting router inputs.
        router_ortho_loss_weight: float = 0.0001,  # default weight for orthogonality loss
        router_ortho_neg_corr_weight: float = 0.1,  # weight for negative correlations in router-ortho loss
        router_ortho_loss_leave_one_out: bool = False,  # whether to leave one dimension out of the router orthogonality loss.
        router_ortho_loss_grad_scale: float = 1.0,  # scaling factor for gradients to expert gate projection weights during router orthogonality loss computation.
        # experts_ortho_loss is very small due to squared cosine similarities.
        # So its weight is set higher to have a meaningful effect.
        experts_ortho_loss_weight: float = 0.01,
        experts_gate_output_loss_weight: float = 0.00001,  # default weight for gate output regularization loss
        projs_diversity_loss_weight: float = 0.01,  # default weight for gate diversity loss
        train_capacity: float = 1,      # slightly smaller than 1.25, the default setting from ST-MoE (see top of page 6)
        eval_capacity: float = 3.0,     # 3.0 leads slightly better performance than 2.0 on CORE.
        min_capacity: int = 4,  # minimum batch size to send to any single expert
        stride: int = 1,  # one in every stride layers are converted to an MoE
        moe_start_layer: int = 2,  # layer index to start using MoE layers, if n_exp > 1
        use_switch_tfm_init: bool = False,  # use weight init scheme from Switch Transformer
        switch_tfm_init_scale: float = 1.0,
        router_use_full_prec: bool = False,  # use float32 precision in the router
        use_qwen3_moe_mlp: bool = True,  # use Qwen3-style MoE MLPs
        # Sliding window attention pattern string, tiled across layers. Final layer always L.
        # Characters: L=long (full context), S=short (half context)
        # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
        window_pattern: str = "SSSL",
        **kwargs,
    ):        
        self.sequence_len = sequence_len
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        self.n_embd = n_embd
        self.num_hidden_layers = n_layer    # For compatibility with lm-eval
        self.num_attention_heads = n_head   # For compatibility with lm-eval
        self.hidden_size = n_embd           # For compatibility with lm-eval
        self.n_exp = n_exp
        self.moe_top_k = moe_top_k  # Store with moe_ prefix to avoid HF generation conflict
        self.use_aux_loss = use_aux_loss
        self.use_router_z_loss = use_router_z_loss
        self.z_loss_demean_logits = z_loss_demean_logits
        self.z_loss_penalize_mean_logits = z_loss_penalize_mean_logits
        self.use_router_ortho_loss = use_router_ortho_loss
        self.use_experts_ortho_loss = use_experts_ortho_loss
        self.use_experts_gate_output_loss = use_experts_gate_output_loss
        self.use_noisy_top_k = use_noisy_top_k
        self.aux_loss_weight = aux_loss_weight
        self.router_z_loss_weight = router_z_loss_weight
        self.router_z_loss_input_grad_scale = router_z_loss_input_grad_scale
        self.router_wg_grad_scale = router_wg_grad_scale
        self.router_ortho_loss_weight = router_ortho_loss_weight
        self.router_ortho_neg_corr_weight = router_ortho_neg_corr_weight
        self.router_ortho_loss_leave_one_out = router_ortho_loss_leave_one_out
        self.router_ortho_loss_grad_scale = router_ortho_loss_grad_scale
        self.experts_ortho_loss_weight = experts_ortho_loss_weight
        self.experts_gate_output_loss_weight = experts_gate_output_loss_weight
        self.projs_diversity_loss_weight = projs_diversity_loss_weight
        self.train_capacity = train_capacity
        self.eval_capacity = eval_capacity
        self.min_capacity = min_capacity
        self.stride = stride
        self.moe_start_layer = moe_start_layer
        self.use_switch_tfm_init = use_switch_tfm_init
        self.switch_tfm_init_scale = switch_tfm_init_scale
        self.router_use_full_prec = router_use_full_prec
        self.use_qwen3_moe_mlp = use_qwen3_moe_mlp
        self.window_pattern = window_pattern
        