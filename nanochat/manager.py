import torch

class MOEManager:
    """
    basic wrapper class for tracking, storing, and aggregating auxiliary
    losses across multiple MoE layers in the model
    """

    def __init__(self, ortho_loss_start_frac=0):
        self.ortho_loss_start_frac = ortho_loss_start_frac
        self.collect_load_balancing_stats = False
        self._values = {
            "aux_loss": [],
            "router_z_loss": [],
            "router_ortho_loss": [],
            "experts_ortho_loss": [],
            "gate_output_loss": [],
            "projs_diversity_loss": [],
            "router_ortho_losses_by_exp": [],
            "drop_rate_per_ks": [],
            "expert_utilities": [],
            "selected_scores": [],
        }
        self._tensor_var_capacity = 32
        self._drop_rate_buffer = None
        self._drop_rate_size = 0
        self._expert_utilities_buffer = None
        self._expert_utilities_size = 0
        self._router_ortho_losses_by_exp_buffer = None
        self._router_ortho_losses_by_exp_size = 0
        self._selected_scores_buffer = None
        self._selected_scores_size = 0
        self._start_frac_names = {
            "router_ortho_loss",
            "experts_ortho_loss",
            "gate_output_loss",
            "projs_diversity_loss",
        }
        self.tensor_var_names = set(["router_ortho_losses_by_exp", 
                                     "drop_rate_per_ks", 
                                     "expert_utilities",
                                     "selected_scores"])

    def reset(self, name):
        if name == "drop_rate_per_ks":
            self._drop_rate_size = 0
            return
        if name == "expert_utilities":
            self._expert_utilities_size = 0
            return
        if name == "router_ortho_losses_by_exp":
            self._router_ortho_losses_by_exp_size = 0
            return
        if name == "selected_scores":
            self._selected_scores_size = 0
            return
        self._values[name] = []

    @torch._dynamo.disable
    def add(self, name, value):
        if name == "drop_rate_per_ks":
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
        if name == "router_ortho_losses_by_exp":
            if self._router_ortho_losses_by_exp_buffer is None:
                self._router_ortho_losses_by_exp_buffer = torch.empty(
                    (self._tensor_var_capacity, value.shape[0]),
                    device=value.device,
                    dtype=value.dtype,
                )
            new_size = self._router_ortho_losses_by_exp_size + 1
            self._router_ortho_losses_by_exp_buffer[self._router_ortho_losses_by_exp_size:new_size].copy_(value)
            self._router_ortho_losses_by_exp_size = new_size
            return
        if name == "selected_scores":
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
        self._values[name].append(value)

    def aggregate(self, name):
        values = self._values.get(name, [])
        if name in self._start_frac_names:
            # If ortho_loss_start_frac = 0.25 and there are 8 moe layers, then 0.25*8 = 2.0, 
            # so start from layer 2, i.e. skip first two layers.
            # But usually we set ortho_loss_start_frac = 0, i.e. sum losses on all layers.
            start_layer = int(len(values) * self.ortho_loss_start_frac)
            values = values[start_layer:]
        if name == "drop_rate_per_ks":
            if self._drop_rate_buffer is None or self._drop_rate_size == 0:
                return None
            values = self._drop_rate_buffer[:self._drop_rate_size]
            return values.mean(dim=0)
        elif name == "expert_utilities":
            if self._expert_utilities_buffer is None or self._expert_utilities_size == 0:
                return None
            values = self._expert_utilities_buffer[:self._expert_utilities_size]
            # Return the whole 2D tensor of expert utilities by layer and by exp, 
            # since different layers have different utilities, and averaging them does not make sense.
            return values
        elif name == "router_ortho_losses_by_exp":
            if self._router_ortho_losses_by_exp_buffer is None or self._router_ortho_losses_by_exp_size == 0:
                return None
            values = self._router_ortho_losses_by_exp_buffer[:self._router_ortho_losses_by_exp_size]
            # Return the whole 2D tensor of router_ortho_losses by layers and by exp, 
            # since different layers have different losses, and averaging them does not make sense.
            return values
        elif name == "selected_scores":
            if self._selected_scores_buffer is None or self._selected_scores_size == 0:
                return None
            values = self._selected_scores_buffer[:self._selected_scores_size]
            return values
        else:
            return sum(values)
    
MANAGER = MOEManager(ortho_loss_start_frac=0.)
