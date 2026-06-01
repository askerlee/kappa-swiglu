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
        use_kappa_swiglu: bool = False,  # add a learnable bias to Qwen3 expert gate activations after gate_proj and SiLU
        kappa_input: str = "router_probs",
        kappa_input_constant: float = 1.0,
        normalize_top_logits: bool = False,
        moe_kappa_slope_max_scale: float = 3.0,
        dense_kappa_slope_max_scale: float = 2.0,
        constant_kappa_bias_dense_layers: bool = False,
        global_kappa_bias_granularity: str = "per-gate",
        kappa_bias_start_layer: int = 0,
        log_implicit_gate_proj_bias: bool = False,
        gate_stats_threshold: float = 0.1,
        gate_stats_topk: int = 16,
        kappa_bias_l2_loss_weight: float = 0.0,
        kappa_bias_ema_rms_reg: bool = False,
        kappa_bias_l2_ema_beta: float = 0.99,
        kappa_bias_l2_ema_anchor_start: float = 0.4,
        kappa_bias_l2_ema_anchor_end: float = 0.8,
        kappa_bias_l2_ema_floor_frac: float = 0.8,
        refresh_kappa_bias_references: bool = False,
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
        self.use_kappa_swiglu = bool(use_kappa_swiglu)
        valid_kappa_inputs = {"top_logits", "router_probs", "constant"}
        if kappa_input not in valid_kappa_inputs:
            raise ValueError(
                "kappa_input must be one of "
                f"{sorted(valid_kappa_inputs)}, got {kappa_input!r}"
            )
        if kappa_input == "constant" and kappa_input_constant is None:
            raise ValueError(
                "kappa_input_constant must be set when kappa_input='constant'"
            )
        self.constant_kappa_bias_dense_layers = bool(constant_kappa_bias_dense_layers)
        self.kappa_input = kappa_input
        self.kappa_input_constant = (
            None if kappa_input_constant is None else float(kappa_input_constant)
        )
        self.normalize_top_logits = bool(normalize_top_logits)
        self.moe_kappa_slope_max_scale = float(moe_kappa_slope_max_scale)
        self.dense_kappa_slope_max_scale = float(dense_kappa_slope_max_scale)
        valid_kappa_bias_granularities = {"per-gate", "per-expert", "per-layer", "global"}
        if global_kappa_bias_granularity not in valid_kappa_bias_granularities:
            raise ValueError(
                "global_kappa_bias_granularity must be one of "
                f"{sorted(valid_kappa_bias_granularities)}, got {global_kappa_bias_granularity!r}"
            )
        self.global_kappa_bias_granularity = global_kappa_bias_granularity
        self.kappa_bias_start_layer = int(kappa_bias_start_layer)
        if self.kappa_bias_start_layer < 0:
            raise ValueError(
                f"kappa_bias_start_layer must be >= 0, got {kappa_bias_start_layer}"
            )
        self.log_implicit_gate_proj_bias = bool(log_implicit_gate_proj_bias)
        self.kappa_bias_l2_loss_weight = float(kappa_bias_l2_loss_weight)
        self.kappa_bias_ema_rms_reg = bool(kappa_bias_ema_rms_reg)
        self.kappa_bias_l2_ema_beta = float(kappa_bias_l2_ema_beta)
        if not (0.0 <= self.kappa_bias_l2_ema_beta < 1.0):
            raise ValueError(
                "kappa_bias_l2_ema_beta must satisfy 0 <= beta < 1, got "
                f"{kappa_bias_l2_ema_beta}"
            )
        self.kappa_bias_l2_ema_anchor_start = float(kappa_bias_l2_ema_anchor_start)
        self.kappa_bias_l2_ema_anchor_end = float(kappa_bias_l2_ema_anchor_end)
        if not (0.0 <= self.kappa_bias_l2_ema_anchor_start <= 1.0):
            raise ValueError(
                "kappa_bias_l2_ema_anchor_start must satisfy 0 <= start <= 1, got "
                f"{kappa_bias_l2_ema_anchor_start}"
            )
        if not (0.0 <= self.kappa_bias_l2_ema_anchor_end <= 1.0):
            raise ValueError(
                "kappa_bias_l2_ema_anchor_end must satisfy 0 <= end <= 1, got "
                f"{kappa_bias_l2_ema_anchor_end}"
            )
        if self.kappa_bias_l2_ema_anchor_end < self.kappa_bias_l2_ema_anchor_start:
            raise ValueError(
                "kappa_bias_l2_ema_anchor_end must be >= kappa_bias_l2_ema_anchor_start, got "
                f"start={kappa_bias_l2_ema_anchor_start}, end={kappa_bias_l2_ema_anchor_end}"
            )
        self.kappa_bias_l2_ema_floor_frac = float(kappa_bias_l2_ema_floor_frac)
        if self.kappa_bias_l2_ema_floor_frac < 0.0:
            raise ValueError(
                "kappa_bias_l2_ema_floor_frac must be >= 0, got "
                f"{kappa_bias_l2_ema_floor_frac}"
            )
        self.gate_stats_threshold = float(gate_stats_threshold)
        self.gate_stats_topk = int(gate_stats_topk)
        if self.gate_stats_topk <= 0:
            raise ValueError(f"gate_stats_topk must be > 0, got {gate_stats_topk}")
        self.kappa_bias_l2_loss_weight = float(kappa_bias_l2_loss_weight)
        self.refresh_kappa_bias_references = bool(refresh_kappa_bias_references)
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
        self.bilinear_mlp_moe = bool(bilinear_mlp_moe)
        self.window_pattern = window_pattern
        self.loss_chunk_tokens = None if loss_chunk_tokens is None else int(loss_chunk_tokens)
        self.debug = debug
        