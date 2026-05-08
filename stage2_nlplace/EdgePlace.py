import os
import sys
import time
import math
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.fft as fft


class ElectricPotential1D(nn.Module):
    """
    @brief 1D Electric Potential for Pin Assign
    Computes electrostatic potential energy to spread pins along an edge
    and avoid overlaps. Based on ePlace methodology adapted for 1D.
    """
    def __init__(self, pin_widths, edge_length, edge_start, num_bins, 
                 target_density, device, dtype=torch.float32):
        """
        @param pin_widths tensor of pin widths (num_pins,)
        @param edge_length length of the edge
        @param edge_start start position of the edge
        @param num_bins number of bins for density calculation
        @param target_density target density (0.0-1.0)
        @param device torch device
        @param dtype data type
        """
        super(ElectricPotential1D, self).__init__()
        
        self.num_pins = len(pin_widths)
        self.edge_length = edge_length
        self.edge_start = edge_start
        self.num_bins = num_bins
        self.target_density = target_density
        self.device = device
        self.dtype = dtype
        
        # Bin size
        self.bin_size = edge_length / num_bins
        
        # Register pin widths as buffer (not trainable)
        self.register_buffer('pin_widths', pin_widths.to(device).to(dtype))
        
        # Compute bin centers
        bin_centers = torch.linspace(
            edge_start + self.bin_size / 2,
            edge_start + edge_length - self.bin_size / 2,
            num_bins, device=device, dtype=dtype
        )
        self.register_buffer('bin_centers', bin_centers)
        
        # Precompute frequency coefficients for 1D Poisson solver
        # For DCT-II: w_k = (π * k / num_bins)
        k = torch.arange(num_bins, device=device, dtype=dtype)
        w = math.pi * k / num_bins
        w2 = w ** 2
        w2[0] = 1.0  # Avoid division by zero, will be zeroed out later
        self.register_buffer('w', w)
        self.register_buffer('w2', w2)
        self.register_buffer('inv_w2', 1.0 / w2)
        
        # Target density per bin
        total_pin_width = pin_widths.sum().item()
        self.target_density_per_bin = target_density * total_pin_width / num_bins
        
    def compute_density_map(self, pos):
        """
        @brief Compute 1D density map using smooth distribution(bell-shaped,not used now)
        @param pos pin positions (num_pins,)
        @return density_map (num_bins,)
        
        Each pin contributes to nearby bins based on overlap.
        Uses bell-shaped (quadratic) distribution for smoothness.
        """
        # Clamp pin widths to at least sqrt(2) * bin_size for smooth density
        sqrt2 = math.sqrt(2)
        pin_widths_clamped = torch.clamp(self.pin_widths, min=sqrt2 * self.bin_size)
        
        # Compute ratio for area preservation
        ratio = self.pin_widths / pin_widths_clamped
        
        # Initialize density map
        density_map = torch.zeros(self.num_bins, device=self.device, dtype=self.dtype)
        
        # For each pin, compute its contribution to each bin
        # Using triangular/bell-shaped distribution
        # O(N*M) can be time consuming
        for i in range(self.num_pins):
            pin_pos = pos[i]
            pin_width = pin_widths_clamped[i]
            half_width = pin_width / 2.0
            
            # Pin range: [pin_pos - half_width, pin_pos + half_width]
            pin_left = pin_pos - half_width
            pin_right = pin_pos + half_width
            
            # Find affected bins
            # O(M) is not efficient, the bins can be calculated in advance
            for b in range(self.num_bins):
                bin_center = self.bin_centers[b]
                bin_left = bin_center - self.bin_size / 2
                bin_right = bin_center + self.bin_size / 2
                
                # Compute overlap between pin and bin
                overlap_left = max(pin_left, bin_left)
                overlap_right = min(pin_right, bin_right)
                overlap = max(0.0, overlap_right - overlap_left)
                
                if overlap > 0:
                    # Bell-shaped distribution: weight by distance from center
                    # 这个 bell-shaped 好像也不太对
                    dist = abs(bin_center - pin_pos)
                    if dist < half_width:
                        # Triangular distribution
                        weight = 1.0 - dist / half_width
                        density_map[b] += self.pin_widths[i] * weight * ratio[i]
        
        return density_map
    
    def compute_density_map_vectorized(self, pos):
        """
        @brief Vectorized computation of 1D density map
        @param pos pin positions (num_pins,)
        @return density_map (num_bins,)
        
        Uses smooth Gaussian-like spreading for differentiability.
        """
        # Compute distances from each pin to each bin center
        # pos: (num_pins,), bin_centers: (num_bins,)
        # distances: (num_pins, num_bins)
        distances = pos.unsqueeze(1) - self.bin_centers.unsqueeze(0)
        
        # Use smooth bell-shaped distribution
        # sigma = pin_width / 2 (approximate width as sigma)
        # But for simplicity, use a fixed sigma based on bin_size
        sigma = self.bin_size * 1.5  # Smoothing parameter
        
        # Gaussian kernel (normalized)
        weights = torch.exp(-0.5 * (distances / sigma) ** 2)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-10)  # Normalize per pin
        
        # Weight by pin width
        pin_contributions = self.pin_widths.unsqueeze(1) * weights
        
        # Sum contributions from all pins
        density_map = pin_contributions.sum(dim=0)
        
        return density_map
    
    def solve_poisson_1d(self, density_map):
        """
        @brief Solve 1D Poisson equation using DCT
        @param density_map (num_bins,)
        @return potential_map, field_map
        
        Solves: d²φ/dx² = ρ(x) - ρ_target
        Using DCT (Type-II) approach.
        """
        # Subtract target density
        rho = density_map - self.target_density_per_bin
        
        # Compute DCT-II (using torch.fft.rfft with pre/post processing)
        # For simplicity, use direct DCT computation
        rho_dct = self.dct_1d(rho)
        
        # Solve in frequency domain: Φ_k = ρ_k / w_k²
        # Note: k=0 component (DC) is set to 0 (average potential is arbitrary)
        potential_dct = rho_dct * self.inv_w2
        potential_dct[0] = 0.0  # Zero out DC component
        
        # Compute field in frequency domain: E_k = -dφ/dx
        # For DCT, derivative converts to DST with factor w
        field_dct = -rho_dct * self.w / self.w2
        field_dct[0] = 0.0
        
        # Inverse transforms
        potential_map = self.idct_1d(potential_dct)
        field_map = self.idst_1d(field_dct)
        
        return potential_map, field_map
    
    def dct_1d(self, x):
        """
        @brief 1D Discrete Cosine Transform (Type-II)
        @param x input signal (N,)
        @return DCT coefficients (N,)
        """
        N = x.shape[0]
        # Create DCT-II matrix
        n = torch.arange(N, device=x.device, dtype=x.dtype)
        k = torch.arange(N, device=x.device, dtype=x.dtype)
        # DCT-II: X[k] = sum_n x[n] * cos(π*k*(2n+1)/(2N))
        cos_matrix = torch.cos(math.pi * k.unsqueeze(1) * (2 * n.unsqueeze(0) + 1) / (2 * N))
        return torch.mv(cos_matrix, x)
    
    def idct_1d(self, X):
        """
        @brief 1D Inverse Discrete Cosine Transform (Type-III)
        @param X DCT coefficients (N,)
        @return signal (N,)
        """
        N = X.shape[0]
        n = torch.arange(N, device=X.device, dtype=X.dtype)
        k = torch.arange(N, device=X.device, dtype=X.dtype)
        # IDCT: x[n] = X[0]/N + (2/N) * sum_{k=1}^{N-1} X[k] * cos(π*k*(2n+1)/(2N))
        cos_matrix = torch.cos(math.pi * k.unsqueeze(0) * (2 * n.unsqueeze(1) + 1) / (2 * N))
        result = torch.mv(cos_matrix.T, X)
        result = result * 2 / N
        result[0] = X[0] / N  # Handle DC component
        return result
    
    def idst_1d(self, X):
        """
        @brief 1D Inverse Discrete Sine Transform (Type-III)
        @param X DST coefficients (N,)
        @return signal (N,)
        """
        N = X.shape[0]
        n = torch.arange(N, device=X.device, dtype=X.dtype)
        k = torch.arange(N, device=X.device, dtype=X.dtype)
        # IDST: x[n] = (2/N) * sum_{k=1}^{N-1} X[k] * sin(π*k*(2n+1)/(2N))
        sin_matrix = torch.sin(math.pi * k.unsqueeze(0) * (2 * n.unsqueeze(1) + 1) / (2 * N))
        result = torch.mv(sin_matrix.T, X)
        result = result * 2 / N
        return result
    
    def interpolate_field(self, pos, field_map):
        """
        @brief Interpolate electric field at pin positions (not used now)
        @param pos pin positions (num_pins,)
        @param field_map field values at bin centers (num_bins,)
        @return field at pin positions (num_pins,)
        """
        # Convert positions to bin indices (fractional)
        bin_indices = (pos - self.edge_start) / self.bin_size - 0.5
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1.001)
        
        # Linear interpolation
        bin_low = bin_indices.long()
        bin_high = torch.clamp(bin_low + 1, max=self.num_bins - 1)
        frac = bin_indices - bin_low.float()
        
        field_at_pins = (1 - frac) * field_map[bin_low] + frac * field_map[bin_high]
        
        return field_at_pins
    
    def forward(self, pos):
        """
        @brief Compute electric potential energy
        @param pos pin positions (num_pins,)
        @return energy (scalar), density_map, overflow
        """
        # Compute density map
        density_map = self.compute_density_map_vectorized(pos)
        
        # Solve Poisson equation
        potential_map, field_map = self.solve_poisson_1d(density_map)
        
        # Compute deviation from target density (rho = density - target)
        rho = density_map - self.target_density_per_bin
        
        # Compute energy: E = 0.5 * sum(rho * phi)
        # This ensures non-negative energy since rho and phi have same sign pattern
        # (from Poisson equation: ∇²φ = ρ)
        energy = 0.5 * (rho * potential_map).sum()
        
        # Compute overflow (density exceeding target)
        overflow = torch.clamp(density_map - self.target_density_per_bin, min=0).sum()
        max_density = density_map.max() / self.bin_size
        
        return energy, overflow, max_density

class PlaceDataCollection(object):
    """
    @brief A wraper for all data tensors on device for building ops, 优化过程中每个iteration需要用到的变量
    """
    def __init__(self, pos, params, edge, placedb, device) -> None:
        """
        @brief initialization
        @param pos locations of pins
        @param params parameters
        @param edge Edge object containing pins
        @param placedb placement database
        @param device cpu or cuda
        """
        self.device = device
        torch.set_num_threads(params.num_threads)
        # position should be parameter
        self.pos = pos
        
        # Edge information
        self.edge = edge
        self.edge_start_point = edge.start_point
        self.edge_end_point = edge.end_point
        self.edge_length = edge.length
        self.edge_direction = edge.direction
        self.edge_fixed_val = edge.fixed_val if hasattr(edge, 'fixed_val') else None
        
        # Number of pins on this edge
        self.num_pins = len(edge.pins)
        
        with torch.no_grad():
            # Extract pin widths from edge.pins
            if len(edge.pin_widths) > 0:
                pin_widths_array = np.array(edge.pin_widths, dtype=np.float32)
            else:
                pin_widths_array = np.array([pin.width for pin in edge.pins], dtype=np.float32)
            
            self.pin_widths = torch.from_numpy(pin_widths_array).to(device)
            
            # Pin areas (for density calculation, in 1D we use width)
            self.pin_areas = self.pin_widths.clone()
            
            self.target_density = torch.empty(1, dtype=self.pos[0].dtype, device=device)
            self.target_density.data.fill_(params.target_density)
            
            # detect movable macros and scale down the density to avoid halos
            # I use a heuristic that cells whose areas are 10x of the mean area will be regarded movable macros in global placement
            # 这里会识别出一些尺寸特别大的Pin，参考eplace里面调整其density to avoid halos
            if self.target_density < 1:
                mean_width = self.pin_widths.mean() * 10
                self.movable_macro_mask = (self.pin_widths > mean_width)
            else:
                self.movable_macro_mask = None
            
            # Pin to net mapping (will be set by NonLinearPlace)
            # For now, create empty tensors
            self.pin2node_map = torch.empty(self.num_pins, dtype=torch.int32, device=device)
            # pin2net_map还是需要的，好好看一下怎么生成的，好像还有flat版本？
            # self.pin2net_map = torch.from_numpy(placedb.pin2net_map).to(device)
            # Bin centers for density calculation (1D bins)
            # Will be initialized when building density op
            self.bin_centers = None
            self.num_bins = getattr(params, 'num_bins', 100)
            
            # Pin indices on this edge (for mapping to global pin indices)
            self.pin_indices = torch.arange(self.num_pins, dtype=torch.int32, device=device)
            


class PlaceOpCollection(object):
    """
    @brief A wrapper for all ops
    """
    def __init__(self) -> None:
        """
        @brief initialization
        """
        self.pin_pos_op = None
        self.move_boundary_op = None
        self.hpwl_op = None
        self.rmst_wl_op = None
        self.density_overflow_op = None
        self.legality_check_op = None
        self.legalize_op = None
        self.detailed_place_op = None
        self.wirelength_op = None
        self.update_gamma_op = None
        self.density_op = None
        self.update_density_weight_op = None
        self.precondition_op = None
        self.noise_op = None
        self.draw_place_op = None
        self.route_utilization_map_op = None
        self.pin_utilization_map_op = None
        self.nctugr_congestion_map_op = None
        self.adjust_node_area_op = None

class EdgePlace(nn.Module):
    """
    @brief single edge placement engine
    Base placement class for a single edge.
    
    """
    def __init__(self, params, edge, placedb) -> None:
        """
        @brief initialization
        @param params parameter
        @param edge Edge object to be placed
        @param placedb placement database
        """
        torch.manual_seed(params.random_seed)
        super(EdgePlace, self).__init__()
        
        self.edge = edge
        self.num_pins = len(edge.pin_widths)
        self.pin_widths = edge.pin_widths
        self.fixed_val = edge.fixed_val
        self.start_point = edge.start_point
        self.end_point = edge.end_point
        self.length = edge.length
        self.direction = edge.direction
       
        
        # Initialize pin positions
        if hasattr(params, 'random_center_init_flag') and params.random_center_init_flag:
            # logging.info(
            #     "move pins to the center of the edge with random noise")
            center = (edge.start_point + edge.end_point) / 2.0
            scale = abs(edge.end_point - edge.start_point) * 0.01
            self.init_pos = np.random.normal(
                loc=center,
                scale=max(scale, 1e-6),  # Avoid zero scale
                size=self.num_pins)
            # logging.info("centerinit_pos: %s" % self.init_pos)
        else:
            # Random init
            self.init_pos = np.random.uniform(edge.start_point, edge.end_point, self.num_pins)
            logging.info("uniform init_pos: %s" % self.init_pos)
            # center = (edge.start_point + edge.end_point) / 2.0
            # self.init_pos = np.full(self.num_pins, center, dtype=np.float32)
        
        # Ensure pin_widths is initialized
        if len(edge.pin_widths) == 0:
            edge.pin_widths = [pin.width for pin in edge.pins]
        
        self.device = torch.device("cuda" if params.gpu else "cpu")
        # 是否要添加filler node，要添加的话到时候这里还有一段代码

        # position should be parameter defined in EdgePlace
        self.pos = nn.ParameterList(
            [nn.Parameter(torch.from_numpy(self.init_pos).to(self.device))])
        
        # shared data on device for building ops
        # I do not want to construct the data from placedb again and again for each op
        self.data_collections = PlaceDataCollection(self.pos, params, edge, placedb,
                                                    self.device)
        
        # similarly I wrap all ops
        self.op_collections = PlaceOpCollection()
        
        # Build operations
        # bound nodes to layout region，可能需要把超出边界的pin给clip到边界，后面看看这个函数是否需要
        self.op_collections.move_boundary_op = self.build_move_boundary(
            params, edge, self.data_collections, self.device)
        
        # hpwl and density overflow ops for evaluation,标准的HPWL计算方法，不是WA或LSE
        # self.op_collections.hpwl_op = self.build_hpwl(
        #     params, placedb, self.data_collections, self.device)
        # density overflow 好像没啥用啊,
        self.op_collections.density_overflow_op = self.build_density_overflow(
            params, edge, self.data_collections, self.device)
        
        # density op for this edge (1D electrostatic potential)
        self.op_collections.density_op = self.build_density_op(
            params, edge, self.data_collections, self.device)
        
        # legality check
        self.op_collections.legality_check_op = self.build_legality_check(
            params, edge, self.data_collections, self.device)
        
        # legalization
        self.op_collections.legalize_op = self.build_legalization(
            params, edge, self.data_collections, self.device)
    
    def build_move_boundary(self, params, edge, data_collections, device):
        """
        @brief bound pins into edge region
        @param params parameters
        @param edge Edge object
        @param data_collections a collection of all data and variables required for constructing the ops
        @param device cpu or cuda
        """
        # TODO: implement move_boundary operation for 1D pins
        # For now, return a placeholder function
        def move_boundary_op(pos):
            with torch.no_grad():
                # Clip positions to edge boundaries considering pin widths
                half_widths = data_collections.pin_widths / 2.0
                pos_min = edge.start_point + half_widths
                pos_max = edge.end_point - half_widths
                pos.data.clamp_(min=pos_min, max=pos_max)
            return pos
        return move_boundary_op
    
    def build_hpwl(self, params, placedb, data_collections, device):
        """
        @brief compute half-perimeter wirelength
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param device cpu or cuda
        """
        # TODO: implement HPWL calculation
        # This will be called at global level in NonLinearPlace
        def hpwl_op(pos):
            # Placeholder: return zero for now
            return torch.tensor(0.0, device=device, requires_grad=True)
        return hpwl_op
    
    def build_density_overflow(self, params, edge, data_collections, device):
        """
        @brief compute density overflow
        @param params parameters
        @param edge Edge object
        @param data_collections a collection of all data and variables required for constructing the ops
        @param device cpu or cuda
        """
        def density_overflow_op(pos):
            """
            @brief Compute density overflow for this edge
            @param pos pin positions (num_pins,)
            @return overflow, max_density
            """
            if hasattr(self, 'electric_potential'):
                energy, overflow, max_density = self.electric_potential(pos)
                return overflow, max_density
            else:
                # Fallback if electric_potential not initialized
                return torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
        return density_overflow_op
    
    def build_density_op(self, params, edge, data_collections, device):
        """
        @brief build 1D electrostatic potential operation for this edge
        @param params parameters
        @param edge Edge object
        @param data_collections a collection of all data and variables required for constructing the ops
        @param device cpu or cuda
        """
        # Get parameters
        num_bins = getattr(params, 'num_bins', 64)
        target_density = getattr(params, 'target_density', 0.8)
        
        # Get pin widths
        pin_widths = torch.tensor(edge.pin_widths, dtype=torch.float32, device=device)
        
        # Create ElectricPotential1D module
        electric_potential = ElectricPotential1D(
            pin_widths=pin_widths,
            edge_length=edge.length,
            edge_start=edge.start_point,
            num_bins=num_bins,
            target_density=target_density,
            device=device
        )
        
        # Store for later use
        self.electric_potential = electric_potential
        
        def density_op(pos):
            """
            @brief Compute 1D electric potential energy for this edge
            @param pos pin positions (num_pins,)
            @return energy (scalar, differentiable)
            """
            energy, overflow, max_density = electric_potential(pos)
            return energy
        
        return density_op
    
    def build_legality_check(self, params, edge, data_collections, device):
        """
        @brief legality check
        @param params parameters
        @param edge Edge object
        @param data_collections a collection of all data and variables required for constructing the ops
        @param device cpu or cuda
        """
        # TODO: implement legality check (check for pin overlaps)
        def legality_check_op(pos):
            # Placeholder: return True for now
            return True
        return legality_check_op
    
    def build_legalization(self, params, edge, data_collections, device):
        """
        @brief legalization
        @param params parameters
        @param edge Edge object
        @param data_collections a collection of all data and variables required for constructing the ops
        @param device cpu or cuda
        """
        # TODO: implement legalization (remove overlaps)
        def legalize_op(pos):
            # Placeholder: return pos as is
            return pos
        return legalize_op
    
    def plot(self, params, edge, placedb, iteration, pos):
        """
        @brief plot layout
        @param params parameters
        @param edge Edge object
        @param placedb placement database
        @param iteration optimization step
        @param pos locations of pins
        """
        pass
    
    def dump(self, params, edge, placedb, pos, filename):
        """
        @brief dump intermediate solution as compressed pickle file (.pklz)
        @param params parameters
        @param edge Edge object
        @param placedb placement database
        @param pos locations of pins
        @param filename output file name
        """
        pass