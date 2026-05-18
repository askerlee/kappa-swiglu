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
        use_aux_free_load_balancing: bool = False,  # use DeepSeekV3 auxiliary-loss-free load balancing via expert-selection bias updates
        aux_free_load_balancing_bias_update_speed: float = 1e-3,  # DeepSeekV3 expert-bias update coefficient
        use_router_z_loss: bool = True,  # apply router z loss (from ST-MoE)
        z_loss_demean_logits: bool = True,  # fix router z loss bug by removing mean of logits
        z_loss_penalize_mean_logits: bool = True,  # penalize mean logits in router z loss
        use_exp_gate_proj_bias: bool = False,  # add a learnable bias to Qwen3 expert gate activations after gate_proj and SiLU
        exp_gate_proj_bias_input: str = "router_probs",
        exp_gate_proj_bias_input_constant: float = 0.5,
        constant_gate_proj_bias_all_layers: bool = False,
        global_gate_proj_bias_granularity: str = "per-gate",
        gate_proj_bias_start_layer: int = 0,
        gate_stats_threshold: float = 0.1,
        gate_stats_topk: int = 16,
        gate_proj_bias_l2_loss_weight: float = 0.0,
        refresh_gate_proj_bias_references: bool = False,
        use_noisy_top_k: bool = False,
        aux_loss_weight: float = 0.001,  # default setting from Switch Transformer (see top of page 8)
        # router z loss: around 160~200. So we use a very small weight to avoid overwhelming the main loss, and we also scale down gradients to router inputs when computing z loss to further stabilize training.
        router_z_loss_weight: float = 1e-5,  # Much smaller than the setting used in ST-MoE (see page 8 eq. 6)
        router_z_loss_input_grad_scale: float = 0.1,  # scale down gradients to router input when computing router z loss.
        train_capacity: float = 1,      # slightly smaller than 1.25, the default setting from ST-MoE (see top of page 6)
        eval_capacity: float = 3.0,     # 3.0 leads slightly better performance than 2.0 on CORE.
        min_capacity: int = 4,  # minimum batch size to send to any single expert
        moe_layer_stride: int = 1,  # one in every stride layers are converted to an MoE
        moe_start_layer: int = 2,  # layer index to start using MoE layers, if n_exp > 1
        num_moe_layers: int = -1,  # total number of MoE layers from moe_start_layer onward (-1 = all eligible layers)
        router_use_full_prec: bool = False,  # use float32 precision in the router
        use_qwen3_moe_mlp: bool = True,  # use Qwen3-style MoE MLPs
        use_qwen3_dense_mlp: bool = True,  # use Qwen3-style dense MLPs in non-MoE layers
        bilinear_mlp_moe: bool = False,  # disable SiLU gating in Qwen3-style MoE MLPs and use raw bilinear gating instead
        # Sliding window attention pattern string, tiled across layers. Final layer always L.
        # Characters: L=long (full context), S=short (half context)
        # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
        window_pattern: str = "SSSL",
        loss_chunk_tokens: int | None = None,
        debug: bool = False,
        **kwargs,
    ):        
        kwargs.pop('use_experts_gate_output_loss', None)
        kwargs.pop('experts_gate_output_loss_weight', None)
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
        self.use_aux_free_load_balancing = use_aux_free_load_balancing
        self.aux_free_load_balancing_bias_update_speed = aux_free_load_balancing_bias_update_speed
        self.use_router_z_loss = use_router_z_loss
        self.z_loss_demean_logits = z_loss_demean_logits
        self.z_loss_penalize_mean_logits = z_loss_penalize_mean_logits
        kwargs.pop('use_router_ortho_loss', None)
        kwargs.pop('router_ortho_loss_target', None)
        kwargs.pop('router_ortho_loss_weight', None)
        kwargs.pop('router_ortho_neg_corr_weight', None)
        kwargs.pop('use_dense_gate_proj_bias', None)
        kwargs.pop('dense_gate_proj_bias_l2_loss_weight', None)
        legacy_bilinear_mlp = kwargs.pop('bilinear_mlp', None)
        kwargs.pop('exp_gate_proj_bias_mode', None)
        self.use_exp_gate_proj_bias = bool(use_exp_gate_proj_bias)
        kwargs.pop('gate_proj_bias_residual_l2_loss_weight', None)
        self.exp_gate_proj_bias_mode = "full"
        valid_exp_gate_proj_bias_inputs = {"top_logits", "router_probs", "constant"}
        if exp_gate_proj_bias_input not in valid_exp_gate_proj_bias_inputs:
            raise ValueError(
                "exp_gate_proj_bias_input must be one of "
                f"{sorted(valid_exp_gate_proj_bias_inputs)}, got {exp_gate_proj_bias_input!r}"
            )
        if exp_gate_proj_bias_input == "constant" and exp_gate_proj_bias_input_constant is None:
            raise ValueError(
                "exp_gate_proj_bias_input_constant must be set when exp_gate_proj_bias_input='constant'"
            )
        self.constant_gate_proj_bias_all_layers = bool(constant_gate_proj_bias_all_layers)
        if self.constant_gate_proj_bias_all_layers and exp_gate_proj_bias_input != "constant":
            raise ValueError(
                "constant_gate_proj_bias_all_layers requires exp_gate_proj_bias_input='constant'"
            )
        self.exp_gate_proj_bias_input = exp_gate_proj_bias_input
        self.exp_gate_proj_bias_input_constant = (
            None if exp_gate_proj_bias_input_constant is None else float(exp_gate_proj_bias_input_constant)
        )
        valid_gate_proj_bias_granularities = {"per-gate", "per-expert", "per-layer", "global"}
        if global_gate_proj_bias_granularity not in valid_gate_proj_bias_granularities:
            raise ValueError(
                "global_gate_proj_bias_granularity must be one of "
                f"{sorted(valid_gate_proj_bias_granularities)}, got {global_gate_proj_bias_granularity!r}"
            )
        self.global_gate_proj_bias_granularity = global_gate_proj_bias_granularity
        self.gate_proj_bias_start_layer = int(gate_proj_bias_start_layer)
        if self.constant_gate_proj_bias_all_layers:
            self.gate_proj_bias_start_layer = 0
        if self.gate_proj_bias_start_layer < 0:
            raise ValueError(
                f"gate_proj_bias_start_layer must be >= 0, got {gate_proj_bias_start_layer}"
            )
        self.gate_proj_bias_l2_loss_weight = float(gate_proj_bias_l2_loss_weight)
        self.gate_stats_threshold = float(gate_stats_threshold)
        self.gate_stats_topk = int(gate_stats_topk)
        if self.gate_stats_topk <= 0:
            raise ValueError(f"gate_stats_topk must be > 0, got {gate_stats_topk}")
        self.gate_proj_bias_l2_loss_weight = float(gate_proj_bias_l2_loss_weight)
        self.refresh_gate_proj_bias_references = bool(refresh_gate_proj_bias_references)
        self.use_noisy_top_k = use_noisy_top_k
        self.aux_loss_weight = aux_loss_weight
        self.router_z_loss_weight = router_z_loss_weight
        self.router_z_loss_input_grad_scale = router_z_loss_input_grad_scale
        self.train_capacity = train_capacity
        self.eval_capacity = eval_capacity
        self.min_capacity = min_capacity
        self.moe_layer_stride = moe_layer_stride
        self.moe_start_layer = moe_start_layer
        if int(num_moe_layers) < -1:
            raise ValueError(f"num_moe_layers must be >= -1, got {num_moe_layers}")
        self.num_moe_layers = int(num_moe_layers)
        self.router_use_full_prec = router_use_full_prec
        self.use_qwen3_moe_mlp = use_qwen3_moe_mlp
        self.use_qwen3_dense_mlp = bool(use_qwen3_dense_mlp)
        if legacy_bilinear_mlp is not None and not bilinear_mlp_moe:
            bilinear_mlp_moe = legacy_bilinear_mlp
        self.bilinear_mlp_moe = bool(bilinear_mlp_moe)
        self.window_pattern = window_pattern
        self.loss_chunk_tokens = None if loss_chunk_tokens is None else int(loss_chunk_tokens)
        self.debug = debug
        