import os
import sys
import time
import numpy as np
import itertools
import logging
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


# def weighted_average_wirelength(self,pin_x, pin_y, flat_net2pin_map, flat_net2pin_start_map, 
#                                  gamma, net_weights=None, net_mask=None):
#     """
#     @brief Compute weighted average wirelength in pure PyTorch
#     @param pin_x tensor of pin x coordinates (num_pins,)
#     @param pin_y tensor of pin y coordinates (num_pins,)
#     @param flat_net2pin_map flat tensor of pin indices for each net
#     @param flat_net2pin_start_map start index tensor for each net in flat_net2pin_map
#     @param gamma smoothing parameter (scalar tensor)
#     @param net_weights optional tensor of net weights
#     @param net_mask optional tensor mask for valid nets
#     @return total weighted average wirelength
#     """
#     device = pin_x.device
#     dtype = pin_x.dtype
    
#     # Handle case when net data is not provided
#     if flat_net2pin_start_map is None or flat_net2pin_map is None:
#         return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
    
#     # Get number of nets
#     if flat_net2pin_start_map.dim() == 0:
#         return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
    
#     num_nets = flat_net2pin_start_map.shape[0] - 1
    
#     if num_nets <= 0:
#         return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
    
#     inv_gamma = 1.0 / gamma
#     total_wl = torch.tensor(0.0, device=device, dtype=dtype)
    
#     for net_id in range(num_nets):
#         # Skip if net is masked
#         if net_mask is not None and not net_mask[net_id]:
#             continue
        
#         start = flat_net2pin_start_map[net_id].item()
#         end = flat_net2pin_start_map[net_id + 1].item()
        
#         # Skip nets with less than 2 pins
#         if end - start < 2:
#             continue
        
#         # Get pin indices for this net
#         pin_indices = self.flat_net2pin_map[start:end]
        
#         # Get pin coordinates for this net
#         net_pin_x = pin_x[pin_indices]
#         net_pin_y = pin_y[pin_indices]
        
#         # Get net weight
#         weight = net_weights[net_id] if net_weights is not None else 1.0
        
#         # WA wirelength for x dimension
#         # For numerical stability, subtract max before exp
#         x_max = net_pin_x.max()
#         x_min = net_pin_x.min()
        
#         # exp((x - x_max) / gamma) for smooth max
#         exp_x_pos = torch.exp((net_pin_x - x_max) * inv_gamma)
#         # exp((x_min - x) / gamma) for smooth min
#         exp_x_neg = torch.exp((x_min - net_pin_x) * inv_gamma)
        
#         # Weighted sums
#         exp_x_pos_sum = exp_x_pos.sum()
#         exp_x_neg_sum = exp_x_neg.sum()
#         xexp_x_pos_sum = (net_pin_x * exp_x_pos).sum()
#         xexp_x_neg_sum = (net_pin_x * exp_x_neg).sum()
        
#         # x_max_smooth - x_min_smooth
#         wl_x = xexp_x_pos_sum / exp_x_pos_sum - xexp_x_neg_sum / exp_x_neg_sum
        
#         # WA wirelength for y dimension
#         y_max = net_pin_y.max()
#         y_min = net_pin_y.min()
        
#         exp_y_pos = torch.exp((net_pin_y - y_max) * inv_gamma)
#         exp_y_neg = torch.exp((y_min - net_pin_y) * inv_gamma)
        
#         exp_y_pos_sum = exp_y_pos.sum()
#         exp_y_neg_sum = exp_y_neg.sum()
#         yexp_y_pos_sum = (net_pin_y * exp_y_pos).sum()
#         yexp_y_neg_sum = (net_pin_y * exp_y_neg).sum()
        
#         wl_y = yexp_y_pos_sum / exp_y_pos_sum - yexp_y_neg_sum / exp_y_neg_sum
        
#         total_wl = total_wl + weight * (wl_x + wl_y)
    
#     return total_wl


class WirelengtData:
    """
    @brief Data structure to hold wirelength computation data
    """
    def __init__(self, num_pins, num_nets, device, dtype=torch.float32):
        """
        @param num_pins total number of pins
        @param num_nets total number of nets
        @param device torch device
        @param dtype data type
        """
        self.num_pins = num_pins
        self.num_nets = num_nets
        self.device = device
        self.dtype = dtype
        
        # Pin coordinates (will be set during computation)
        self.pin_x = None
        self.pin_y = None
        
        # Net to pin mappings
        self.flat_net2pin_map = None
        self.flat_net2pin_start_map = None
        self.pin2net_map = None
        
        # Net properties
        self.net_weights = None
        self.net_mask = None


class PlaceObj(nn.Module):
    """
    @brief Define placement objective:
        wirelength + density_weight * density penalty
    It includes various ops related to global placement as well.
    For Pin Assign, this is the global objective function that aggregates
    wirelength across all edges and density from all edges.
    """
    def __init__(self, density_weight, params, placedb, edges, edge_places, global_place_params=None,
                 flat_net2pin_map=None, flat_net2pin_start_map=None, pin2net_map=None,
                 net_weights=None, net_mask=None, reuse_group=None) -> None:
        """
        @brief initialize ops for placement
        @param density_weight density weight in the objective
        @param params parameters
        @param placedb placement database
        @param edges list of all Edge objects
        @param edge_places list of EdgePlace instances, one per edge
        @param global_place_params global placement parameters for current global placement stage
        @param flat_net2pin_map flat tensor of pin indices for each net
        @param flat_net2pin_start_map start index tensor for each net
        @param pin2net_map tensor mapping pin to net
        @param net_weights optional tensor of net weights
        @param net_mask optional tensor mask for valid nets
        @param reuse_group list of reuse groups, e.g. [[0, 2], [1, 3]] means edges 0,2 share
               the same pin distribution, and edges 1,3 share the same pin distribution.
               The first element in each group is the base edge.
        """
        super(PlaceObj, self).__init__()
        
        ### quadratic penalty
        self.density_quad_coeff = 2000
        self.init_density = None
        ### increase density penalty if slow convergence
        self.density_factor = 1
        
        # For Pin Assign, we don't have fence regions, so use first-order density penalty by default
        self.quad_penalty = False

        ### update mask controls whether stop gradient/updating, 1 represents allow grad/update
        self.update_mask = None

        self.params = params
        self.placedb = placedb
        self.edges = edges
        self.edge_places = edge_places
        
        # ----------------------------------------------------------------
        # Build reuse-edge mappings
        # reuse_group: e.g. [[0, 2]] means edge 0 is the base, edge 2 is a copy
        # is_reuse_edge : set of edge indices that are non-base copies
        # base_edge_of  : reuse_edge_id -> base_edge_id
        # reuse_offset  : reuse_edge_id -> 1-D translation offset (base→reuse)
        # ----------------------------------------------------------------
        self.is_reuse_edge = set()
        self.base_edge_of  = {}   # reuse_id  -> base_id
        self.reuse_offset  = {}   # reuse_id  -> float offset along edge direction
        if reuse_group is not None:
            for group in reuse_group:
                if len(group) < 2:
                    continue
                base_id   = group[0]
                base_edge = edges[base_id]
                for reuse_id in group[1:]:
                    reuse_edge = edges[reuse_id]
                    # Offset along the varying direction of the edge
                    offset = float(reuse_edge.start_point - base_edge.start_point)
                    self.base_edge_of[reuse_id] = base_id
                    self.reuse_offset[reuse_id] = offset
                    self.is_reuse_edge.add(reuse_id)
                    logging.info(
                        "Reuse edge %d is a copy of base edge %d (offset=%.4g)"
                        % (reuse_id, base_id, offset))

        # Edges that contribute an independent density term (reuse edges use base density only)
        n_ep = len(edge_places) if edge_places is not None else 0
        self.base_edge_indices: List[int] = [
            i for i in range(n_ep) if i not in self.is_reuse_edge
        ]
        self._num_density_edges = len(self.base_edge_indices)
        self.init_density_vec: Optional[torch.Tensor] = None
        
        # Set global_place_params with defaults if not provided
        if global_place_params is None:
            global_place_params = {
                "wirelength": "weighted_average",
                "num_bins": getattr(params, 'num_bins', 100),
                "learning_rate": getattr(params, 'learning_rate', 0.01),
                "iteration": getattr(params, 'iteration', 1000),
            }
        self.global_place_params = global_place_params

        self.gpu = getattr(params, 'gpu', 0)
        self.device = torch.device("cuda" if self.gpu else "cpu")
        
        # Wirelength computation data
        # Use passed parameters if provided, otherwise try to get from placedb
        if flat_net2pin_map is not None:
            self.flat_net2pin_map = flat_net2pin_map
            self.flat_net2pin_start_map = flat_net2pin_start_map
            self.pin2net_map = pin2net_map
            self.net_weights = net_weights
            self.net_mask = net_mask
            self.num_pins = placedb.total_pin_count if placedb is not None else 0
            self.num_nets = len(placedb.nets_list) if placedb is not None and hasattr(placedb, 'nets_list') else 0
        elif placedb is not None and hasattr(placedb, 'flat_net2pin_map'):
            self.flat_net2pin_map = placedb.flat_net2pin_map
            self.flat_net2pin_start_map = placedb.flat_net2pin_start_map
            self.pin2net_map = placedb.pin2net_map
            self.net_weights = placedb.net_weights
            self.net_mask = placedb.net_mask
            self.num_pins = placedb.total_pin_count
            self.num_nets = len(placedb.nets_list)
        else:
            self.flat_net2pin_map = None
            self.flat_net2pin_start_map = None
            self.pin2net_map = None
            self.net_weights = None
            self.net_mask = None
            self.num_pins = 0
            self.num_nets = 0
        
        # Per–base-edge density weights (length K = _num_density_edges)
        K = self._num_density_edges
        if K == 0:
            self.density_weight = torch.zeros(0, dtype=torch.float32, device=self.device)
        else:
            self.density_weight = torch.full(
                (K,), float(density_weight), dtype=torch.float32, device=self.device)
        
        # Gamma for wirelength smoothing (WA or LSE)
        self.gamma = torch.tensor(self.base_gamma(params),
                                  dtype=torch.float32,
                                  device=self.device)
        
        # Build wirelength operation
        wirelength_method = global_place_params.get("wirelength", "weighted_average")
        if wirelength_method == "weighted_average":
            self.op_collections_wirelength_op, self.op_collections_update_gamma_op = self.build_weighted_average_wl(
                params, placedb)
        elif wirelength_method == "logsumexp":
            self.op_collections_wirelength_op, self.op_collections_update_gamma_op = self.build_logsumexp_wl(
                params, placedb)
        else:
            assert 0, "unknown wirelength model %s" % wirelength_method
        
        # Build density overflow operation
        self.op_collections_density_overflow_op = self.build_density_overflow_op(params, placedb)
        
        # Build update density weight operation
        # self.op_collections_update_density_weight_op = self.build_update_density_weight_op(params, placedb)
        
        # Build precondition operation
        self.op_collections_precondition_op = self.build_precondition_op(params, placedb)
        
        # Skip density weight initialization if no edge_places
        if edge_places is not None and len(edge_places) > 0:
            self.initialize_density_weight(params, placedb)
    
    def obj_fn(self, all_edge_positions):
        """
        @brief Compute objective.
            wirelength + sum_i density_weight[i] * density_i (base edges only)
        @param all_edge_positions dict mapping edge_id to position tensor for that edge
        @return objective value
        """
        self.wirelength = self.op_collections_wirelength_op(all_edge_positions)
        K = self._num_density_edges
        if K == 0:
            self.density = torch.tensor(0.0, device=self.device, requires_grad=True)
            return self.wirelength

        density_list = []
        for edge_id in self.base_edge_indices:
            edge_place = self.edge_places[edge_id]
            edge_pos = all_edge_positions[edge_id]
            density_list.append(edge_place.op_collections.density_op(edge_pos))

        dens_stack = torch.stack(density_list)
        w = self.density_weight

        if self.init_density is None:
            self.init_density_vec = dens_stack.detach().clone()
            self.init_density = dens_stack.sum().detach().clone()
            tot = self.init_density
            if tot > 0:
                self.density_weight_grad_precond = 1.0 / tot
            else:
                self.density_weight_grad_precond = 1.0
            self.quad_penalty_coeff = self.density_quad_coeff / 2 * self.density_weight_grad_precond

        if self.quad_penalty:
            init_v = self.init_density_vec.clamp(min=1e-18)
            quad_coeff_vec = self.density_quad_coeff / 2.0 / init_v
            dens_transformed = dens_stack * (1.0 + quad_coeff_vec * dens_stack)
            self.density = (w * dens_transformed).sum()
        else:
            self.density = (w * dens_stack).sum()

        return self.wirelength + self.density_factor * self.density
    
    def build_pin_positions(self, all_edge_positions):
        """
        @brief Build global pin_x, pin_y tensors from all edge positions.
               For reuse edges, positions are derived from their base edge via a
               translation offset instead of using their own (non-optimized) pos.
        @param all_edge_positions dict mapping edge_id to position tensor for that edge
        @return pin_x, pin_y tensors of all pin coordinates
        """
        pin_x_list = []
        pin_y_list = []
        
        for edge_id, edge_place in enumerate(self.edge_places):
            edge = edge_place.edge
            
            if edge_id in self.is_reuse_edge:
                # Reuse edge: derive positions from the base edge + offset
                base_id = self.base_edge_of[edge_id]
                offset  = self.reuse_offset[edge_id]
                # base_pos has requires_grad=True; adding a constant keeps the grad_fn
                edge_pos = all_edge_positions[base_id] + offset
            else:
                edge_pos = all_edge_positions[edge_id]  # 1D positions along the edge
            
            if edge.direction == 'horizontal':
                # Horizontal edge: x varies, y = fixed_val
                pin_x_list.append(edge_pos)
                pin_y_list.append(torch.full_like(edge_pos, edge.fixed_val))
            elif edge.direction == 'vertical':
                # Vertical edge: x = fixed_val, y varies
                pin_x_list.append(torch.full_like(edge_pos, edge.fixed_val))
                pin_y_list.append(edge_pos)
            else:
                raise ValueError(f"Unknown edge direction: {edge.direction}")
        
        pin_x = torch.cat(pin_x_list, dim=0)
        pin_y = torch.cat(pin_y_list, dim=0)
        
        return pin_x, pin_y
    
    def obj_wl_test(self, all_edge_positions):
        """
        @brief Compute wirelength only (for testing)
        @param all_edge_positions dict mapping edge_id to position tensor for that edge
        @return wirelength value
        """
        # Build global pin positions from edge positions
        pin_x, pin_y = self.build_pin_positions(all_edge_positions)
        
        # Compute wirelength using weighted average
        wirelength = self.weighted_average_wirelength(
            pin_x, pin_y,
        )
        
        return wirelength
    
    def obj_density_test(self, all_edge_positions):
        """
        @brief Compute density only (for testing).
               Reuse edges share the same density as their base edge and are skipped.
        @param all_edge_positions dict mapping edge_id to position tensor for that edge
        @return density value (sum of base-edge densities only)
        """
        density_list = []
        for edge_id, edge_place in enumerate(self.edge_places):
            if edge_id in self.is_reuse_edge:
                continue  # density already represented by the base edge
            edge_pos = all_edge_positions[edge_id]
            edge_density = edge_place.op_collections.density_op(edge_pos)
            density_list.append(edge_density)
        
        if not density_list:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        total_density = sum(density_list)
        return total_density

    def _normalize_density_weights(
        self,
        density_weight: Union[None, float, int, Sequence[float], torch.Tensor],
    ) -> torch.Tensor:
        """Map scalar or length-K vector to a [K] tensor on device."""
        K = self._num_density_edges
        if K == 0:
            return torch.zeros(0, dtype=torch.float32, device=self.device)
        if density_weight is None:
            return self.density_weight.detach().to(device=self.device, dtype=torch.float32)
        if isinstance(density_weight, torch.Tensor):
            t = density_weight.to(device=self.device, dtype=torch.float32).reshape(-1)
        else:
            t = torch.as_tensor(density_weight, dtype=torch.float32, device=self.device).reshape(-1)
        if t.numel() == 1:
            return torch.full((K,), float(t.item()), dtype=torch.float32, device=self.device)
        if t.numel() != K:
            raise ValueError(
                "density_weight has length %d but num base density edges K=%d"
                % (t.numel(), K)
            )
        return t
    
    def obj_wl_density_test(
        self,
        all_edge_positions,
        density_weight: Union[None, float, int, Sequence[float], torch.Tensor] = 1.0,
        compute_per_edge_norms: bool = False,
    ) -> Tuple[torch.Tensor, float, float, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        @brief wl_weight * WL + sum_i w_i * density_i (base edges). Also reports gradient norms.
        @param density_weight  Scalar (broadcast), length-K vector, or None (use self.density_weight).
        @param compute_per_edge_norms  If True, run per-base-edge density backward for schedule metrics.
        @return objective, wl_grad_norm, density_grad_norm, wl_per_edge, raw_d_per_edge
                Last two entries are None when compute_per_edge_norms is False.
        """
        wl_weight = 1000
        K = self._num_density_edges
        w = self._normalize_density_weights(density_weight)

        wirelength = self.obj_wl_test(all_edge_positions)
        wirelength.backward(retain_graph=True)

        wl_per_edge_out: Optional[np.ndarray] = None
        if compute_per_edge_norms and all_edge_positions:
            max_eid = max(all_edge_positions.keys())
            wl_per_edge_out = np.zeros(max_eid + 1, dtype=np.float64)
            for edge_id, edge_pos in all_edge_positions.items():
                if edge_pos.grad is not None:
                    wl_per_edge_out[edge_id] = edge_pos.grad.norm(p=2).item() * wl_weight
            wl_grad_norm = float(np.sqrt(np.sum(wl_per_edge_out ** 2)))
        else:
            wl_grad_norm = 0.0
            for edge_pos in all_edge_positions.values():
                if edge_pos.grad is not None:
                    wl_grad_norm += edge_pos.grad.norm(p=2).item() ** 2
            wl_grad_norm = float(np.sqrt(wl_grad_norm)) * wl_weight

        for edge_pos in all_edge_positions.values():
            if edge_pos.grad is not None:
                edge_pos.grad.zero_()

        if K == 0:
            weighted_density = wirelength * 0.0
        else:
            dens_list = [
                self.edge_places[eid].op_collections.density_op(all_edge_positions[eid])
                for eid in self.base_edge_indices
            ]
            dens_stack = torch.stack(dens_list)
            weighted_density = (w * dens_stack).sum()

        raw_d_per_edge: Optional[np.ndarray] = None
        density_grad_norm = 0.0

        if K > 0:
            if compute_per_edge_norms:
                raw_d_per_edge = np.zeros(K, dtype=np.float64)
                for ki, eid in enumerate(self.base_edge_indices):
                    edge_pos = all_edge_positions[eid]
                    d_i = self.edge_places[eid].op_collections.density_op(edge_pos)
                    d_i.backward(retain_graph=True)
                    if edge_pos.grad is not None:
                        raw_d_per_edge[ki] = edge_pos.grad.norm(p=2).item()
                    for ep in all_edge_positions.values():
                        if ep.grad is not None:
                            ep.grad.zero_()
                w_np = w.detach().cpu().numpy()
                density_grad_norm = float(np.sqrt(np.sum((w_np * raw_d_per_edge) ** 2)))
            else:
                weighted_density.backward(retain_graph=True)
                for edge_pos in all_edge_positions.values():
                    if edge_pos.grad is not None:
                        density_grad_norm += edge_pos.grad.norm(p=2).item() ** 2
                density_grad_norm = float(np.sqrt(density_grad_norm))
                for edge_pos in all_edge_positions.values():
                    if edge_pos.grad is not None:
                        edge_pos.grad.zero_()

        if not compute_per_edge_norms:
            wl_per_edge_out = None

        for edge_pos in all_edge_positions.values():
            if edge_pos.grad is not None:
                edge_pos.grad.zero_()

        objective = wl_weight * wirelength + weighted_density
        return objective, wl_grad_norm, density_grad_norm, wl_per_edge_out, raw_d_per_edge
    
    def forward(self):
        """
        @brief Compute objective with current locations of cells.
        """
        return self.obj_fn(self.all_edge_positions)
    
    def obj_and_grad_fn(self, all_edge_positions):
        """
        @brief compute objective and gradient.
            wirelength + density_weight * density penalty
        @param all_edge_positions dict mapping edge_id to position tensor for that edge
        @return objective value and gradients
        """
        # Zero gradients for all edge positions
        for edge_pos in all_edge_positions.values():
            if edge_pos.grad is not None:
                edge_pos.grad.zero_()
        
        obj = self.obj_fn(all_edge_positions)
        obj.backward()
        
        # Apply preconditioning to gradients
        all_grads = {}
        for edge_id, edge_pos in all_edge_positions.items():
            if edge_pos.grad is not None:
                grad = self.op_collections_precondition_op(edge_pos.grad, self.density_weight)
                all_grads[edge_id] = grad
            else:
                all_grads[edge_id] = None
        
        return obj, all_grads
    
    def base_gamma(self, params):
        """
        @brief compute base gamma
        @param params parameters
        """
        # For 1D placement, use average edge length as base
        if self.edges is not None and len(self.edges) > 0:
            avg_edge_length = np.mean([edge.length for edge in self.edges])
        else:
            avg_edge_length = 100.0  # default value
        gamma_base = getattr(params, 'gamma', 4.0) * avg_edge_length / 100.0
        return max(gamma_base, 1.0)  # ensure gamma is at least 1
    
    def update_gamma(self, iteration, overflow, base_gamma):
        """
        @brief update gamma in wirelength model
        @param iteration optimization step
        @param overflow evaluated in current step
        @param base_gamma base gamma
        """
        if isinstance(overflow, torch.Tensor):
            overflow_avg = overflow.mean() if overflow.numel() > 1 else overflow
        else:
            overflow_avg = overflow
        
        coef = torch.pow(torch.tensor(10.0), (overflow_avg - 0.1) * 20 / 9 - 1)
        self.gamma.data.fill_((base_gamma * coef).item())
        return True
    
    def initialize_density_weight(self, params, placedb):
        """
        @brief Per–base-edge initial weights: params.density_weight * (wl_g_i / density_g_i).
        """
        K = self._num_density_edges
        if K == 0:
            self.density_weight = torch.zeros(0, dtype=torch.float32, device=self.device)
            return self.density_weight

        dummy_positions = {}
        for edge_id, edge_place in enumerate(self.edge_places):
            dummy_pos = edge_place.pos[0].clone().detach().requires_grad_(True)
            dummy_positions[edge_id] = dummy_pos

        wirelength = self.op_collections_wirelength_op(dummy_positions)
        wirelength.backward()

        wl_g = {}
        for eid, edge_pos in dummy_positions.items():
            if edge_pos.grad is not None:
                wl_g[eid] = edge_pos.grad.norm(p=2).item()
            else:
                wl_g[eid] = 0.0

        eps = 1e-18
        pw = float(getattr(params, "density_weight", 1.0))
        # Edges with zero WL gradient (e.g. not in any 2+ pin net) would get w_i=0; keep a floor.
        w_floor = max(1e-6 * pw, 1e-12)
        weights = []
        for eid in self.base_edge_indices:
            edge_place = self.edge_places[eid]
            edge_pos = edge_place.pos[0].clone().detach().requires_grad_(True)
            density = edge_place.op_collections.density_op(edge_pos)
            density.backward()
            dg = edge_pos.grad.norm(p=2).item() if edge_pos.grad is not None else 0.0
            wl_i = wl_g.get(eid, 0.0)
            if dg > eps:
                wi = pw * (wl_i / dg)
                weights.append(max(wi, w_floor))
            else:
                weights.append(pw)

        self.density_weight = torch.tensor(
            weights, dtype=self.density_weight.dtype, device=self.density_weight.device
        )
        return self.density_weight
    
    def build_weighted_average_wl(self, params, placedb):
        """
        @brief build the op to compute weighted average wirelength
        @param params parameters
        @param placedb placement database
        """
        def build_wirelength_op(all_edge_positions):
            """
            @brief compute weighted average wirelength
            @param all_edge_positions dict mapping edge_id to position tensor for that edge
            """
            # Build global pin positions from edge positions
            pin_x, pin_y = self.build_pin_positions(all_edge_positions)
            
            # Check if we have valid net mappings
            if self.flat_net2pin_map is None or self.flat_net2pin_start_map is None:
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # Compute wirelength using weighted average
            wirelength = self.weighted_average_wirelength(
                pin_x, pin_y,
                
            )
            
            return wirelength
        
        def build_update_gamma_op(iteration, overflow):
            base_gamma = self.base_gamma(params)
            self.update_gamma(iteration, overflow, base_gamma)
        
        return build_wirelength_op, build_update_gamma_op
    
    def weighted_average_wirelength(self, pin_x, pin_y, net_weights=None, net_mask=None):
        """
        @brief Compute weighted average wirelength in pure PyTorch
        @param pin_x tensor of pin x coordinates (num_pins,)
        @param pin_y tensor of pin y coordinates (num_pins,)
        @param flat_net2pin_map flat tensor of pin indices for each net
        @param flat_net2pin_start_map start index tensor for each net in flat_net2pin_map
        @param gamma smoothing parameter (scalar tensor)
        @param net_weights optional tensor of net weights
        @param net_mask optional tensor mask for valid nets
        @return total weighted average wirelength
        """
        device = pin_x.device
        dtype = pin_x.dtype
        # num_nets = len(flat_net2pin_start_map) - 1
        
        if self.num_nets == 0:
            return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
        
        inv_gamma = 1.0 / self.gamma
        total_wl = torch.tensor(0.0, device=device, dtype=dtype)
        
        for net_id in range(self.num_nets):
            # Skip if net is masked
            if net_mask is not None and not net_mask[net_id]:
                continue
            
            start = self.flat_net2pin_start_map[net_id].item()
            end = self.flat_net2pin_start_map[net_id + 1].item()
            
            # Skip nets with less than 2 pins
            if end - start < 2:
                continue
            
            # Get pin indices for this net
            pin_indices = self.flat_net2pin_map[start:end]
            
            # Get pin coordinates for this net
            net_pin_x = pin_x[pin_indices]
            net_pin_y = pin_y[pin_indices]
            
            # Get net weight
            weight = net_weights[net_id] if net_weights is not None else 1.0
            
            # WA wirelength for x dimension
            # For numerical stability, subtract max before exp
            x_max = net_pin_x.max()
            x_min = net_pin_x.min()
            
            # exp((x - x_max) / gamma) for smooth max
            exp_x_pos = torch.exp((net_pin_x - x_max) * inv_gamma)
            # exp((x_min - x) / gamma) for smooth min
            exp_x_neg = torch.exp((x_min - net_pin_x) * inv_gamma)
            
            # Weighted sums
            exp_x_pos_sum = exp_x_pos.sum()
            exp_x_neg_sum = exp_x_neg.sum()
            xexp_x_pos_sum = (net_pin_x * exp_x_pos).sum()
            xexp_x_neg_sum = (net_pin_x * exp_x_neg).sum()
            
            # x_max_smooth - x_min_smooth
            wl_x = xexp_x_pos_sum / exp_x_pos_sum - xexp_x_neg_sum / exp_x_neg_sum
            
            # WA wirelength for y dimension
            y_max = net_pin_y.max()
            y_min = net_pin_y.min()
            
            exp_y_pos = torch.exp((net_pin_y - y_max) * inv_gamma)
            exp_y_neg = torch.exp((y_min - net_pin_y) * inv_gamma)
            
            exp_y_pos_sum = exp_y_pos.sum()
            exp_y_neg_sum = exp_y_neg.sum()
            yexp_y_pos_sum = (net_pin_y * exp_y_pos).sum()
            yexp_y_neg_sum = (net_pin_y * exp_y_neg).sum()
            
            wl_y = yexp_y_pos_sum / exp_y_pos_sum - yexp_y_neg_sum / exp_y_neg_sum
            
            total_wl = total_wl + weight * (wl_x + wl_y)
        
        return total_wl
    
    def build_logsumexp_wl(self, params, placedb):
        """
        @brief build the op to compute log-sum-exp wirelength
        @param params parameters
        @param placedb placement database
        """
        # TODO: implement log-sum-exp wirelength for Pin Assign
        def build_wirelength_op(all_edge_positions):
            # Placeholder: return zero for now
            device = self.density_weight.device
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        def build_update_gamma_op(iteration, overflow):
            base_gamma = self.base_gamma(params)
            self.update_gamma(iteration, overflow, base_gamma)
        
        return build_wirelength_op, build_update_gamma_op
    
    def build_density_overflow_op(self, params, placedb):
        """
        @brief build density overflow operation
        @param params parameters
        @param placedb placement database
        """
        # TODO: implement density overflow calculation
        def density_overflow_op(all_edge_positions):
            # Placeholder: return zero overflow for now
            device = self.density_weight.device
            overflow = torch.tensor(0.0, device=device)
            max_density = torch.tensor(1.0, device=device)
            return overflow, max_density
        return density_overflow_op
    
    def update_density_weight(self, density_weight, wl_grad, density_grad):
        """
        @brief build update density weight operation
        @param params parameters
        @param placedb placement database
        """
        if density_grad > 1.3*wl_grad:
            density_weight = density_weight * 0.9
        elif density_grad < 0.7*wl_grad:
            density_weight = density_weight * 1.1
        return density_weight
    
    def build_precondition_op(self, params, placedb):
        """
        @brief build preconditioning operation
        @param params parameters
        @param placedb placement database
        """
        # TODO: implement gradient preconditioning
        def precondition_op(grad, density_weight, update_mask=None):
            # Placeholder: return grad as is
            return grad
        return precondition_op
    
    def estimate_initial_learning_rate(self, all_edge_positions, lr):
        """
        @brief Estimate initial learning rate by moving a small step.
        @param all_edge_positions dict mapping edge_id to position tensor
        @param lr small step
        """
        # Compute objective and gradient at current position
        obj_k, all_grads_k = self.obj_and_grad_fn(all_edge_positions)
        
        # Move a small step
        all_edge_positions_k1 = {}
        for edge_id, edge_pos in all_edge_positions.items():
            if edge_id in all_grads_k and all_grads_k[edge_id] is not None:
                all_edge_positions_k1[edge_id] = edge_pos - lr * all_grads_k[edge_id]
            else:
                all_edge_positions_k1[edge_id] = edge_pos.clone()
        
        # Compute objective and gradient at new position
        obj_k1, all_grads_k1 = self.obj_and_grad_fn(all_edge_positions_k1)
        
        # Compute learning rate estimate
        x_diff_norm = 0.0
        g_diff_norm = 0.0
        
        for edge_id in all_edge_positions:
            if edge_id in all_grads_k and all_grads_k[edge_id] is not None and \
               edge_id in all_grads_k1 and all_grads_k1[edge_id] is not None:
                x_diff = (all_edge_positions[edge_id] - all_edge_positions_k1[edge_id]).norm(p=2)
                g_diff = (all_grads_k[edge_id] - all_grads_k1[edge_id]).norm(p=2)
                x_diff_norm += x_diff.item() ** 2
                g_diff_norm += g_diff.item() ** 2
        
        if g_diff_norm > 0:
            estimated_lr = np.sqrt(x_diff_norm) / np.sqrt(g_diff_norm)
        else:
            estimated_lr = lr
        
        return torch.tensor(estimated_lr, device=self.density_weight.device)
