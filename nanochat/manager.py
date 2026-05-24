import torch

class MOEManager:
    """
    basic wrapper class for tracking, storing, and aggregating auxiliary
    losses across multiple MoE layers in the model
    """

    def __init__(self):
        self.collect_load_balancing_stats = False
        self.collect_backward_stats = False
        self._values = {
            "aux_loss": [],
            "router_z_loss": [],
            "gate_proj_bias_l2_loss": [],
            "gate_proj_bias_scale_l2_loss": [],
            "gate_proj_bias_ema_rms_reg_loss": [],
            "gate_proj_bias_scale_ema_rms_reg_loss": [],
            "gate_proj_bias_shift_abs_top5p_mean": [],
            "gate_proj_bias_shift_abs_bottom5p_mean": [],
            "gate_grad_scale_mean": [],
            "drop_rate_per_ks": [],
            "expert_utilities": [],
            "selected_scores": [],
            "gate_proj_bias_shift_abs_mean": [],
            "gate_proj_bias_shift_abs_mean_normalized": [],
            "implicit_gate_proj_bias_top5p_mean": [],
            "implicit_gate_proj_bias_bottom5p_mean": [],
            "routed_token_router_weight_cosine_mean": [],
            "routed_token_router_weight_cosine_top5p_mean": [],
            "routed_token_router_weight_cosine_bottom5p_mean": [],
        }
        self._tensor_var_capacity = 32
        self._drop_rate_buffer = None
        self._drop_rate_size = 0
        self._expert_utilities_buffer = None
        self._expert_utilities_size = 0
        self._selected_scores_buffer = None
        self._selected_scores_size = 0
        self._gate_proj_bias_shift_abs_top5p_mean_buffer = None
        self._gate_proj_bias_shift_abs_top5p_mean_size = 0
        self._gate_proj_bias_shift_abs_bottom5p_mean_buffer = None
        self._gate_proj_bias_shift_abs_bottom5p_mean_size = 0
        self._gate_grad_scale_mean_buffer = None
        self._gate_grad_scale_mean_size = 0
        self._gate_proj_bias_shift_abs_mean_buffer = None
        self._gate_proj_bias_shift_abs_mean_size = 0
        self._gate_proj_bias_shift_abs_mean_normalized_buffer = None
        self._gate_proj_bias_shift_abs_mean_normalized_size = 0
        self._implicit_gate_proj_bias_top5p_mean_buffer = None
        self._implicit_gate_proj_bias_top5p_mean_size = 0
        self._implicit_gate_proj_bias_bottom5p_mean_buffer = None
        self._implicit_gate_proj_bias_bottom5p_mean_size = 0
        self._routed_token_router_weight_cosine_mean_buffer = None
        self._routed_token_router_weight_cosine_mean_size = 0
        self._routed_token_router_weight_cosine_top5p_mean_buffer = None
        self._routed_token_router_weight_cosine_top5p_mean_size = 0
        self._routed_token_router_weight_cosine_bottom5p_mean_buffer = None
        self._routed_token_router_weight_cosine_bottom5p_mean_size = 0
        self.tensor_var_names = \
        set(["drop_rate_per_ks", 
             "expert_utilities",
             "selected_scores",
             "gate_proj_bias_shift_abs_top5p_mean",
             "gate_proj_bias_shift_abs_bottom5p_mean",
             "gate_grad_scale_mean",
             "gate_proj_bias_shift_abs_mean",
             "gate_proj_bias_shift_abs_mean_normalized",
             "implicit_gate_proj_bias_top5p_mean",
             "implicit_gate_proj_bias_bottom5p_mean",
             "routed_token_router_weight_cosine_mean",
             "routed_token_router_weight_cosine_top5p_mean",
             "routed_token_router_weight_cosine_bottom5p_mean"])

    def reset(self, name):
        if name == "drop_rate_per_ks":
            self._drop_rate_size = 0
            return
        if name == "expert_utilities":
            self._expert_utilities_size = 0
            return
        if name == "selected_scores":
            self._selected_scores_size = 0
            return
        if name == "gate_proj_bias_shift_abs_top5p_mean":
            self._gate_proj_bias_shift_abs_top5p_mean_size = 0
            return
        if name == "gate_proj_bias_shift_abs_bottom5p_mean":
            self._gate_proj_bias_shift_abs_bottom5p_mean_size = 0
            return
        if name == "gate_grad_scale_mean":
            self._gate_grad_scale_mean_size = 0
            return
        if name == "gate_proj_bias_shift_abs_mean":
            self._gate_proj_bias_shift_abs_mean_size = 0
            return
        if name == "gate_proj_bias_shift_abs_mean_normalized":
            self._gate_proj_bias_shift_abs_mean_normalized_size = 0
            return
        if name == "implicit_gate_proj_bias_top5p_mean":
            self._implicit_gate_proj_bias_top5p_mean_size = 0
            return
        if name == "implicit_gate_proj_bias_bottom5p_mean":
            self._implicit_gate_proj_bias_bottom5p_mean_size = 0
            return
        if name == "routed_token_router_weight_cosine_mean":
            self._routed_token_router_weight_cosine_mean_size = 0
            return
        if name == "routed_token_router_weight_cosine_top5p_mean":
            self._routed_token_router_weight_cosine_top5p_mean_size = 0
            return
        if name == "routed_token_router_weight_cosine_bottom5p_mean":
            self._routed_token_router_weight_cosine_bottom5p_mean_size = 0
            return
        self._values[name] = []

    def reset_all(self):
        for name in self._values:
            self.reset(name)

    @torch._dynamo.disable
    def add(self, name, value):
        if name == "drop_rate_per_ks":
            with torch.inference_mode(False):
                if self._drop_rate_buffer is None:
                    self._drop_rate_buffer = torch.empty(
                        (self._tensor_var_capacity, value.shape[0]),
                        device=value.device,
                        dtype=value.dtype,
                    )
                new_size = self._drop_rate_size + 1
                self._drop_rate_buffer[self._drop_rate_size:new_size].copy_(value)
                self._drop_rate_size = new_size
            return
        if name == "expert_utilities":
            with torch.inference_mode(False):
                if self._expert_utilities_buffer is None:
                    self._expert_utilities_buffer = torch.empty(
                        (self._tensor_var_capacity, value.shape[0]),
                        device=value.device,
                        dtype=value.dtype,
                    )
                new_size = self._expert_utilities_size + 1
                self._expert_utilities_buffer[self._expert_utilities_size:new_size].copy_(value)
                self._expert_utilities_size = new_size
            return
        if name == "selected_scores":
            with torch.inference_mode(False):
                if self._selected_scores_buffer is None:
                    self._selected_scores_buffer = torch.empty(
                        (self._tensor_var_capacity, value.shape[0]),
                        device=value.device,
                        dtype=value.dtype,
                    )
                new_size = self._selected_scores_size + 1
                self._selected_scores_buffer[self._selected_scores_size:new_size].copy_(value)
                self._selected_scores_size = new_size
            return
        if name == "gate_proj_bias_shift_abs_top5p_mean":
            with torch.inference_mode(False):
                if self._gate_proj_bias_shift_abs_top5p_mean_buffer is None:
                    self._gate_proj_bias_shift_abs_top5p_mean_buffer = torch.empty(
                        (self._tensor_var_capacity,),
                        device=value.device,
                        dtype=value.dtype,
                    )
                new_size = self._gate_proj_bias_shift_abs_top5p_mean_size + 1
                self._gate_proj_bias_shift_abs_top5p_mean_buffer[
                    self._gate_proj_bias_shift_abs_top5p_mean_size:new_size
                ].copy_(value.reshape(1))
                self._gate_proj_bias_shift_abs_top5p_mean_size = new_size
            return
        if name == "gate_proj_bias_shift_abs_bottom5p_mean":
            with torch.inference_mode(False):
                if self._gate_proj_bias_shift_abs_bottom5p_mean_buffer is None:
                    self._gate_proj_bias_shift_abs_bottom5p_mean_buffer = torch.empty(
                        (self._tensor_var_capacity,),
                        device=value.device,
                        dtype=value.dtype,
                    )
                new_size = self._gate_proj_bias_shift_abs_bottom5p_mean_size + 1
                self._gate_proj_bias_shift_abs_bottom5p_mean_buffer[
                    self._gate_proj_bias_shift_abs_bottom5p_mean_size:new_size
                ].copy_(value.reshape(1))
                self._gate_proj_bias_shift_abs_bottom5p_mean_size = new_size
            return
        if name == "gate_grad_scale_mean":
            with torch.inference_mode(False):
                if self._gate_grad_scale_mean_buffer is None:
                    self._gate_grad_scale_mean_buffer = torch.empty(
                        (self._tensor_var_capacity, value.shape[0]),
                        device=value.device,
                        dtype=value.dtype,
                    )
                new_size = self._gate_grad_scale_mean_size + 1
                self._gate_grad_scale_mean_buffer[self._gate_grad_scale_mean_size:new_size].copy_(value)
                self._gate_grad_scale_mean_size = new_size
            return
        if name == "gate_proj_bias_shift_abs_mean":
            with torch.inference_mode(False):
                if self._gate_proj_bias_shift_abs_mean_buffer is None:
                    self._gate_proj_bias_shift_abs_mean_buffer = torch.empty(
                        (self._tensor_var_capacity,),
                        device=value.device,
                        dtype=value.dtype,
                    )
                new_size = self._gate_proj_bias_shift_abs_mean_size + 1
                self._gate_proj_bias_shift_abs_mean_buffer[
                    self._gate_proj_bias_shift_abs_mean_size:new_size
                ].copy_(value.reshape(1))
                self._gate_proj_bias_shift_abs_mean_size = new_size
            return
        if name == "gate_proj_bias_shift_abs_mean_normalized":
            with torch.inference_mode(False):
                if self._gate_proj_bias_shift_abs_mean_normalized_buffer is None:
                    self._gate_proj_bias_shift_abs_mean_normalized_buffer = torch.empty(
                        (self._tensor_var_capacity,),
                        device=value.device,
                        dtype=value.dtype,
                    )
                new_size = self._gate_proj_bias_shift_abs_mean_normalized_size + 1
                self._gate_proj_bias_shift_abs_mean_normalized_buffer[
                    self._gate_proj_bias_shift_abs_mean_normalized_size:new_size
                ].copy_(value.reshape(1))
                self._gate_proj_bias_shift_abs_mean_normalized_size = new_size
            return
        if name == "implicit_gate_proj_bias_top5p_mean":
            with torch.inference_mode(False):
                if self._implicit_gate_proj_bias_top5p_mean_buffer is None:
                    self._implicit_gate_proj_bias_top5p_mean_buffer = torch.empty(
                        (self._tensor_var_capacity,),
                        device=value.device,
                        dtype=value.dtype,
                    )
                new_size = self._implicit_gate_proj_bias_top5p_mean_size + 1
                self._implicit_gate_proj_bias_top5p_mean_buffer[
                    self._implicit_gate_proj_bias_top5p_mean_size:new_size
                ].copy_(value.reshape(1))
                self._implicit_gate_proj_bias_top5p_mean_size = new_size
            return
        if name == "implicit_gate_proj_bias_bottom5p_mean":
            with torch.inference_mode(False):
                if self._implicit_gate_proj_bias_bottom5p_mean_buffer is None:
                    self._implicit_gate_proj_bias_bottom5p_mean_buffer = torch.empty(
                        (self._tensor_var_capacity,),
                        device=value.device,
                        dtype=value.dtype,
                    )
                new_size = self._implicit_gate_proj_bias_bottom5p_mean_size + 1
                self._implicit_gate_proj_bias_bottom5p_mean_buffer[
                    self._implicit_gate_proj_bias_bottom5p_mean_size:new_size
                ].copy_(value.reshape(1))
                self._implicit_gate_proj_bias_bottom5p_mean_size = new_size
            return
        if name == "routed_token_router_weight_cosine_mean":
            with torch.inference_mode(False):
                if self._routed_token_router_weight_cosine_mean_buffer is None:
                    self._routed_token_router_weight_cosine_mean_buffer = torch.empty(
                        (self._tensor_var_capacity,),
                        device=value.device,
                        dtype=value.dtype,
                    )
                new_size = self._routed_token_router_weight_cosine_mean_size + 1
                self._routed_token_router_weight_cosine_mean_buffer[
                    self._routed_token_router_weight_cosine_mean_size:new_size
                ].copy_(value.reshape(1))
                self._routed_token_router_weight_cosine_mean_size = new_size
            return
        if name == "routed_token_router_weight_cosine_top5p_mean":
            with torch.inference_mode(False):
                if self._routed_token_router_weight_cosine_top5p_mean_buffer is None:
                    self._routed_token_router_weight_cosine_top5p_mean_buffer = torch.empty(
                        (self._tensor_var_capacity,),
                        device=value.device,
                        dtype=value.dtype,
                    )
                new_size = self._routed_token_router_weight_cosine_top5p_mean_size + 1
                self._routed_token_router_weight_cosine_top5p_mean_buffer[
                    self._routed_token_router_weight_cosine_top5p_mean_size:new_size
                ].copy_(value.reshape(1))
                self._routed_token_router_weight_cosine_top5p_mean_size = new_size
            return
        if name == "routed_token_router_weight_cosine_bottom5p_mean":
            with torch.inference_mode(False):
                if self._routed_token_router_weight_cosine_bottom5p_mean_buffer is None:
                    self._routed_token_router_weight_cosine_bottom5p_mean_buffer = torch.empty(
                        (self._tensor_var_capacity,),
                        device=value.device,
                        dtype=value.dtype,
                    )
                new_size = self._routed_token_router_weight_cosine_bottom5p_mean_size + 1
                self._routed_token_router_weight_cosine_bottom5p_mean_buffer[
                    self._routed_token_router_weight_cosine_bottom5p_mean_size:new_size
                ].copy_(value.reshape(1))
                self._routed_token_router_weight_cosine_bottom5p_mean_size = new_size
            return
        self._values[name].append(value)

    def aggregate(self, name):
        values = self._values.get(name, [])
        if name == "drop_rate_per_ks":
            if self._drop_rate_buffer is None or self._drop_rate_size == 0:
                return None
            values = self._drop_rate_buffer[:self._drop_rate_size]
            # Keep one [top_k] row per MoE layer; the training loop averages
            # across microsteps separately when it accumulates step losses.
            return values
        elif name == "expert_utilities":
            if self._expert_utilities_buffer is None or self._expert_utilities_size == 0:
                return None
            values = self._expert_utilities_buffer[:self._expert_utilities_size]
            # Return the whole 2D tensor of expert utilities by layer and by exp, 
            # since different layers have different utilities, and averaging them does not make sense.
            return values
        elif name == "selected_scores":
            if self._selected_scores_buffer is None or self._selected_scores_size == 0:
                return None
            values = self._selected_scores_buffer[:self._selected_scores_size]
            return values
        elif name == "gate_proj_bias_shift_abs_top5p_mean":
            if (
                self._gate_proj_bias_shift_abs_top5p_mean_buffer is None
                or self._gate_proj_bias_shift_abs_top5p_mean_size == 0
            ):
                return None
            values = self._gate_proj_bias_shift_abs_top5p_mean_buffer[
                :self._gate_proj_bias_shift_abs_top5p_mean_size
            ]
            return values
        elif name == "gate_proj_bias_shift_abs_bottom5p_mean":
            if (
                self._gate_proj_bias_shift_abs_bottom5p_mean_buffer is None
                or self._gate_proj_bias_shift_abs_bottom5p_mean_size == 0
            ):
                return None
            values = self._gate_proj_bias_shift_abs_bottom5p_mean_buffer[
                :self._gate_proj_bias_shift_abs_bottom5p_mean_size
            ]
            return values
        elif name == "gate_grad_scale_mean":
            if self._gate_grad_scale_mean_buffer is None or self._gate_grad_scale_mean_size == 0:
                return None
            values = self._gate_grad_scale_mean_buffer[:self._gate_grad_scale_mean_size]
            return values
        elif name == "gate_proj_bias_shift_abs_mean":
            if self._gate_proj_bias_shift_abs_mean_buffer is None or self._gate_proj_bias_shift_abs_mean_size == 0:
                return None
            values = self._gate_proj_bias_shift_abs_mean_buffer[:self._gate_proj_bias_shift_abs_mean_size]
            return values
        elif name == "gate_proj_bias_shift_abs_mean_normalized":
            if (
                self._gate_proj_bias_shift_abs_mean_normalized_buffer is None
                or self._gate_proj_bias_shift_abs_mean_normalized_size == 0
            ):
                return None
            values = self._gate_proj_bias_shift_abs_mean_normalized_buffer[
                :self._gate_proj_bias_shift_abs_mean_normalized_size
            ]
            return values
        elif name == "implicit_gate_proj_bias_top5p_mean":
            if (
                self._implicit_gate_proj_bias_top5p_mean_buffer is None
                or self._implicit_gate_proj_bias_top5p_mean_size == 0
            ):
                return None
            values = self._implicit_gate_proj_bias_top5p_mean_buffer[
                :self._implicit_gate_proj_bias_top5p_mean_size
            ]
            return values
        elif name == "implicit_gate_proj_bias_bottom5p_mean":
            if (
                self._implicit_gate_proj_bias_bottom5p_mean_buffer is None
                or self._implicit_gate_proj_bias_bottom5p_mean_size == 0
            ):
                return None
            values = self._implicit_gate_proj_bias_bottom5p_mean_buffer[
                :self._implicit_gate_proj_bias_bottom5p_mean_size
            ]
            return values
        elif name == "routed_token_router_weight_cosine_mean":
            if (
                self._routed_token_router_weight_cosine_mean_buffer is None
                or self._routed_token_router_weight_cosine_mean_size == 0
            ):
                return None
            values = self._routed_token_router_weight_cosine_mean_buffer[
                :self._routed_token_router_weight_cosine_mean_size
            ]
            return values
        elif name == "routed_token_router_weight_cosine_top5p_mean":
            if (
                self._routed_token_router_weight_cosine_top5p_mean_buffer is None
                or self._routed_token_router_weight_cosine_top5p_mean_size == 0
            ):
                return None
            values = self._routed_token_router_weight_cosine_top5p_mean_buffer[
                :self._routed_token_router_weight_cosine_top5p_mean_size
            ]
            return values
        elif name == "routed_token_router_weight_cosine_bottom5p_mean":
            if (
                self._routed_token_router_weight_cosine_bottom5p_mean_buffer is None
                or self._routed_token_router_weight_cosine_bottom5p_mean_size == 0
            ):
                return None
            values = self._routed_token_router_weight_cosine_bottom5p_mean_buffer[
                :self._routed_token_router_weight_cosine_bottom5p_mean_size
            ]
            return values
        else:
            return sum(values)
    
MANAGER = MOEManager()
