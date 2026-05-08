##
# @file   NonLinearPlace.py
# @brief  Nonlinear placement engine for Pin Assign
#         Manages all edges and performs global optimization
#

import os
import sys
import time
import json
import numpy as np
import logging
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
from PIL import Image


from EdgePlace import EdgePlace
from PlaceObj import PlaceObj
from PlaceDB import PlaceDB, Edge
import Params


def density_weight_target_scale(
    iteration: int,
    max_iterations: int,
    scale_high: float = 2.0,
    scale_low: float = 0.05,
) -> float:
    """Monotone schedule: large early (density vs WL), small late."""
    if max_iterations <= 1:
        return scale_low
    t = iteration / float(max_iterations - 1)
    return scale_high * (1.0 - t) + scale_low * t


def update_per_edge_density_weights(
    w: torch.Tensor,
    wl_per_edge: np.ndarray,
    raw_d_per_edge: np.ndarray,
    base_edge_indices: List[int],
    target_scale: float,
    w_floor: Optional[float] = None,
) -> torch.Tensor:
    """Per-edge case2-style rule; thresholds scaled by target_scale(iteration).

    If wl_i == 0, hi/lo would both be 0 and any positive d_w would only shrink w toward 0.
    We use a small WL floor for the thresholds only, and clamp weights after updates.
    """
    w = w.clone()
    K = len(base_edge_indices)
    wl_eps = 1e-9 * max(float(np.max(wl_per_edge)) if len(wl_per_edge) else 1.0, 1.0)
    if w_floor is None:
        w_floor = max(1e-8 * float(w.mean().item()), 1e-12)
    for ki in range(K):
        eid = base_edge_indices[ki]
        wl_i = float(wl_per_edge[eid]) if eid < len(wl_per_edge) else 0.0
        wl_t = max(wl_i, wl_eps)
        d_w = float(w[ki].item()) * float(raw_d_per_edge[ki])
        hi = 1.3 * wl_t * target_scale
        lo = 1.0 * wl_t * target_scale
        if d_w > hi:
            w[ki] *= 0.95
        elif d_w < lo:
            w[ki] *= 1.05
        w[ki].clamp_(min=w_floor)
    return w


class NonLinearPlace(nn.Module):
    """
    @brief Nonlinear placement engine for Pin Assign.
    It takes parameters and placement database and runs placement flow.
    Manages all edges and performs global optimization.
    """
    
    def __init__(self, params, placedb):
        """
        @brief initialization.
        @param params parameters
        @param placedb placement database
        """
        super(NonLinearPlace, self).__init__()

        self.params = params
        self.placedb = placedb
        self.device = torch.device("cuda" if params.gpu else "cpu")

        # TODO: 待重构 —— 主入口改为 test_optimization_from_segment_assignments()
        # 以下旧初始化逻辑（基于 PlaceDB 对象构建 EdgePlace/PlaceObj）暂时保留注释备查

        # # Build pin to net mappings first (needed for EdgePlace initialization)
        # self.pin2net_map, self.net2pin_map, self.flat_net2pin_map, self.flat_net2pin_start_map = \
        #     self.build_pin_net_mappings()
        #
        # # Collect all edges from all modules
        # self.edges = self.collect_all_edges()
        # logging.info("Collected %d edges" % len(self.edges))
        #
        # # Build pin_to_global_idx once and reuse across all edges
        # pin_to_global_idx: Dict[object, int] = {}
        # global_idx = 0
        # for edge in self.edges:
        #     for pin in edge.pins:
        #         pin_to_global_idx[pin] = global_idx
        #         global_idx += 1
        #
        # # Create EdgePlace instance for each edge
        # self.edge_places = []
        # for edge_id, edge in enumerate(self.edges):
        #     self.set_edge_pin_net_map(edge, edge_id, pin_to_global_idx)
        #     if len(edge.pin_widths) == 0:
        #         edge.pin_widths = [pin.width for pin in edge.pins]
        #     edge_place = EdgePlace(params, edge, placedb)
        #     self.edge_places.append(edge_place)
        # logging.info("Created %d EdgePlace instances" % len(self.edge_places))
        #
        # # Create global PlaceObj
        # density_weight = 0.0
        # global_place_params = {
        #     "wirelength": getattr(params, 'wirelength_method', 'weighted_average'),
        #     "num_bins": getattr(params, 'num_bins', 100),
        #     "learning_rate": getattr(params, 'learning_rate', 0.01),
        #     "iteration": getattr(params, 'iteration', 1000),
        # }
        # self.place_obj = PlaceObj(
        #     density_weight, params, placedb, self.edges, self.edge_places, global_place_params
        # ).to(self.device)
        # logging.info("Created PlaceObj")
    
    # TODO: 待重构 —— collect_all_edges 属于旧 PlaceDB 对象流程，暂时保留注释备查
    # def collect_all_edges(self) -> List[Edge]:
    #     all_edges = []
    #     for module in self.placedb.all_modules_list:
    #         module_edges = module.get_all_edges()
    #         for edge in module_edges:
    #             if len(edge.pins) > 0:
    #                 all_edges.append(edge)
    #     return all_edges
    
    # TODO: 待重构 —— build_pin_net_mappings 属于旧 PlaceDB 对象流程，暂时保留注释备查
    # def build_pin_net_mappings(self):
    #     pin_to_global_idx = {}
    #     global_idx = 0
    #     for edge in self.edges:
    #         for pin in edge.pins:
    #             pin_to_global_idx[pin] = global_idx
    #             global_idx += 1
    #     num_pins = global_idx
    #     pin2net_map = np.full(num_pins, -1, dtype=np.int32)
    #     net2pin_map = []
    #     for net_id, net in enumerate(self.placedb.nets_list):
    #         pin_indices = []
    #         for pin in net.pins:
    #             if pin in pin_to_global_idx:
    #                 pin_idx = pin_to_global_idx[pin]
    #                 pin2net_map[pin_idx] = net_id
    #                 pin_indices.append(pin_idx)
    #         net2pin_map.append(pin_indices)
    #     flat_net2pin_map = []
    #     flat_net2pin_start_map = [0]
    #     for net_pins in net2pin_map:
    #         flat_net2pin_map.extend(net_pins)
    #         flat_net2pin_start_map.append(len(flat_net2pin_map))
    #     flat_net2pin_map = np.array(flat_net2pin_map, dtype=np.int32)
    #     flat_net2pin_start_map = np.array(flat_net2pin_start_map, dtype=np.int32)
    #     return pin2net_map, net2pin_map, flat_net2pin_map, flat_net2pin_start_map

    # TODO: 待重构 —— set_edge_pin_net_map 属于旧 PlaceDB 对象流程，暂时保留注释备查
    # def set_edge_pin_net_map(self, edge, edge_id, pin_to_global_idx=None):
    #     if pin_to_global_idx is None:
    #         pin_to_global_idx = {}
    #         global_idx = 0
    #         for e in self.edges:
    #             for pin in e.pins:
    #                 pin_to_global_idx[pin] = global_idx
    #                 global_idx += 1
    #     edge_pin2net_map = []
    #     for pin in edge.pins:
    #         pin_idx = pin_to_global_idx.get(pin, None)
    #         if pin_idx is not None:
    #             edge_pin2net_map.append(int(self.pin2net_map[pin_idx]))
    #         else:
    #             edge_pin2net_map.append(-1)
    #     edge.pin2net_map = np.array(edge_pin2net_map, dtype=np.int32)
    
    # TODO: 待重构 —— __call__ 旧优化循环已废弃，主入口改为 test_optimization_from_segment_assignments()
    # def __call__(self, params, placedb):
    #     iteration = 0
    #     all_metrics = []
    #     if params.global_place_flag:
    #         optimizer_name = getattr(params, 'optimizer', 'adam').lower()
    #         all_params = []
    #         for edge_place in self.edge_places:
    #             all_params.extend(list(edge_place.parameters()))
    #         if optimizer_name == "adam":
    #             optimizer = torch.optim.Adam(all_params, lr=0)
    #         elif optimizer_name == "sgd":
    #             optimizer = torch.optim.SGD(all_params, lr=0)
    #         elif optimizer_name == "sgd_momentum":
    #             optimizer = torch.optim.SGD(all_params, lr=0, momentum=0.9, nesterov=False)
    #         elif optimizer_name == "sgd_nesterov":
    #             optimizer = torch.optim.SGD(all_params, lr=0, momentum=0.9, nesterov=True)
    #         else:
    #             assert 0, "unknown optimizer %s" % optimizer_name
    #         self.place_obj.train()
    #         max_iterations = getattr(params, 'iteration', 1000)
    #         for iteration in range(max_iterations):
    #             all_edge_positions = {eid: ep.pos[0] for eid, ep in enumerate(self.edge_places)}
    #             for ep in self.edge_places:
    #                 ep.op_collections.move_boundary_op(ep.pos[0])
    #             optimizer.zero_grad()
    #             obj, all_grads = self.place_obj.obj_and_grad_fn(all_edge_positions)
    #             for eid, ep in enumerate(self.edge_places):
    #                 if eid in all_grads and all_grads[eid] is not None:
    #                     if ep.pos[0].grad is None:
    #                         ep.pos[0].grad = all_grads[eid]
    #                     else:
    #                         ep.pos[0].grad.data.copy_(all_grads[eid])
    #             optimizer.step()
    #             if iteration % 100 == 0:
    #                 logging.info("iteration %d: objective = %g" % (iteration, obj.item()))
    #         logging.info("optimization completed")
    #     return all_metrics
    
 


class OptimizationVisualizer:
    """
    @brief Visualization tool for optimization process
    Generates animated GIFs and metrics plots
    """
    
    # Elegant color palette
    COLORS = {
        'edges': ['#2C3E50', '#8E44AD', '#16A085', '#D35400', '#2980B9', 
                  '#C0392B', '#27AE60', '#F39C12'],
        'pins': ['#3498DB', '#9B59B6', '#1ABC9C', '#E67E22', '#34495E',
                 '#E74C3C', '#2ECC71', '#F1C40F'],
        'wl': '#E74C3C',      # Wirelength - warm red
        'density': '#3498DB', # Density - cool blue
        'bg': '#FAFAFA',      # Background
        'grid': '#E0E0E0',    # Grid lines
        'text': '#2C3E50',    # Text color
    }
    
    def __init__(self, save_path: str = "optimization.gif"):
        """Initialize the visualizer with empty history"""
        self.history = {
            'iterations': [],
            'wirelength': [],
            'density': [],
            'total_obj': [],
            'positions': [],       # List of {edge_id: positions_array}
            'elapsed_seconds': [], # Wall time from optimization start (cumulative)
            'iter_time': [],       # Time for each recorded interval (seconds)
            'density_weight': [],  # Density penalty weight schedule
        }
        self.edge_places = None
        self.fig_size = (10, 8)
        self.save_path = save_path
    
    def set_edge_places(self, edge_places):
        """Set edge places reference for position visualization"""
        self.edge_places = edge_places
    
    def record(self, iteration: int, wirelength: float = None, density: float = None,
               total_obj: float = None, edge_places=None,
               elapsed: float = None, iter_time: float = None,
               density_weight: float = None):
        """
        @brief Record metrics and positions at current iteration.
        @param iteration       Current iteration number.
        @param wirelength      Wirelength value (optional).
        @param density         Density energy value (optional).
        @param total_obj       Total objective value (optional).
        @param edge_places     List of EdgePlace instances (optional, for position recording).
        @param elapsed         Cumulative wall time from optimization start (seconds).
        @param iter_time       Wall time for the last recorded interval (seconds).
        @param density_weight  Current density penalty weight.
        """
        self.history['iterations'].append(iteration)
        self.history['wirelength'].append(wirelength)
        self.history['density'].append(density)
        self.history['total_obj'].append(total_obj)
        self.history['elapsed_seconds'].append(elapsed)
        self.history['iter_time'].append(iter_time)
        self.history['density_weight'].append(density_weight)

        # Record positions
        if edge_places is not None:
            positions = {}
            for edge_id, ep in enumerate(edge_places):
                pos = ep.pos[0].detach().cpu().numpy().copy()
                positions[edge_id] = pos
            self.history['positions'].append(positions)
    
    def _create_pin_frame(self, frame_idx: int, ax, show_metrics: bool = True):
        """
        @brief Create a single frame for the animation
        @param frame_idx Index of the frame in history
        @param ax Matplotlib axes
        @param show_metrics Whether to show metrics text
        """
        ax.clear()
        
        if self.edge_places is None or frame_idx >= len(self.history['positions']):
            return
        
        positions = self.history['positions'][frame_idx]
        iteration = self.history['iterations'][frame_idx]
        
        all_x, all_y = [], []
        
        for idx, edge_place in enumerate(self.edge_places):
            edge = edge_place.edge
            edge_color = self.COLORS['edges'][idx % len(self.COLORS['edges'])]
            pin_color = self.COLORS['pins'][idx % len(self.COLORS['pins'])]
            
            pin_positions_1d = positions[idx]
            pin_widths = edge_place.pin_widths if hasattr(edge_place, 'pin_widths') else edge.pin_widths
            
            if edge.direction == 'horizontal':
                edge_y = edge.fixed_val
                edge_x_start, edge_x_end = edge.start_point, edge.end_point
                
                # Draw edge line
                ax.plot([edge_x_start, edge_x_end], [edge_y, edge_y], 
                        color=edge_color, linewidth=2.5, solid_capstyle='round',
                        alpha=0.9, zorder=1)
                
                # Draw pins
                for pin_idx, (pin_x, pin_width) in enumerate(zip(pin_positions_1d, pin_widths)):
                    rect = patches.FancyBboxPatch(
                        (pin_x - pin_width / 2, edge_y - 2.5),
                        pin_width, 5,
                        boxstyle=patches.BoxStyle("Round", pad=0.3),
                        linewidth=1, edgecolor='white',
                        facecolor=pin_color, alpha=0.85, zorder=2
                    )
                    ax.add_patch(rect)
                    all_x.extend([pin_x - pin_width / 2, pin_x + pin_width / 2])
                    all_y.extend([edge_y - 3, edge_y + 3])
                
                all_x.extend([edge_x_start, edge_x_end])
                all_y.append(edge_y)
                
            elif edge.direction == 'vertical':
                edge_x = edge.fixed_val
                edge_y_start, edge_y_end = edge.start_point, edge.end_point
                
                ax.plot([edge_x, edge_x], [edge_y_start, edge_y_end], 
                        color=edge_color, linewidth=2.5, solid_capstyle='round',
                        alpha=0.9, zorder=1)
                
                for pin_idx, (pin_y, pin_width) in enumerate(zip(pin_positions_1d, pin_widths)):
                    rect = patches.FancyBboxPatch(
                        (edge_x - 2.5, pin_y - pin_width / 2),
                        5, pin_width,
                        boxstyle=patches.BoxStyle("Round", pad=0.3),
                        linewidth=1, edgecolor='white',
                        facecolor=pin_color, alpha=0.85, zorder=2
                    )
                    ax.add_patch(rect)
                    all_x.extend([edge_x - 3, edge_x + 3])
                    all_y.extend([pin_y - pin_width / 2, pin_y + pin_width / 2])
                
                all_x.append(edge_x)
                all_y.extend([edge_y_start, edge_y_end])
        
        # Set limits
        if all_x and all_y:
            x_margin = (max(all_x) - min(all_x)) * 0.15 + 10
            y_margin = (max(all_y) - min(all_y)) * 0.15 + 10
            ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
            ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
        
        # Style
        ax.set_facecolor(self.COLORS['bg'])
        ax.grid(True, linestyle='-', alpha=0.3, color=self.COLORS['grid'], zorder=0)
        ax.set_aspect('equal', adjustable='box')
        
        # Iteration title
        ax.set_title(f'Iteration {iteration}', fontsize=14, fontweight='600', 
                     color=self.COLORS['text'], pad=10)
        
        # Metrics annotation
        if show_metrics:
            wl = self.history['wirelength'][frame_idx]
            density = self.history['density'][frame_idx]
            metrics_text = []
            if wl is not None:
                metrics_text.append(f'WL: {wl:.2f}')
            if density is not None:
                metrics_text.append(f'Density: {density:.4f}')
            if metrics_text:
                ax.annotate('\n'.join(metrics_text), xy=(0.02, 0.98), xycoords='axes fraction',
                           fontsize=10, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                    edgecolor=self.COLORS['grid'], alpha=0.9))
        
        # Clean axes
        ax.tick_params(axis='both', which='both', length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    def create_animation_gif(self, save_path: str = "optimization.gif", fps: int = 10,
                             sample_interval: int = 1):
        """
        @brief Create animated GIF of pin position changes
        @param save_path Path to save the GIF
        @param fps Frames per second
        @param sample_interval Sample every N frames (for large iteration counts)
        """
        if not self.history['positions'] or self.edge_places is None:
            logging.warning("No position history recorded. Cannot create animation.")
            return
        
        # Sample frames
        total_frames = len(self.history['positions'])
        frame_indices = list(range(0, total_frames, sample_interval))
        if frame_indices[-1] != total_frames - 1:
            frame_indices.append(total_frames - 1)  # Always include final frame
        
        logging.info(f"Creating GIF with {len(frame_indices)} frames...")
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=100)
        fig.patch.set_facecolor(self.COLORS['bg'])
        
        frames = []
        for i, frame_idx in enumerate(frame_indices):
            self._create_pin_frame(frame_idx, ax, show_metrics=True)
            
            # Convert to image
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                       facecolor=self.COLORS['bg'], edgecolor='none')
            buf.seek(0)
            img = Image.open(buf).copy()
            frames.append(img)
            buf.close()
            
            if (i + 1) % 50 == 0:
                logging.info(f"  Processed {i + 1}/{len(frame_indices)} frames")
        
        plt.close(fig)
        save_path = os.path.join(self.save_path, save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save GIF
        if frames:
            # Hold final frame longer
            duration_ms = int(1000 / fps)
            durations = [duration_ms] * len(frames)
            durations[-1] = duration_ms * 3  # Hold final frame 3x longer
            
            frames[0].save(
                save_path,
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=0
            )
            logging.info(f"Animation saved to {save_path}")
    
    def plot_metrics(self, save_path: str = "metrics.png", show: bool = True):
        """
        @brief Plot optimization metrics over iterations.

        Produces a figure with up to three rows:
          Row 1 – Wirelength (left y-axis) + Density (right y-axis, dual-axis).
          Row 2 – Per-interval wall time (bar) + cumulative time (line).  [only if timing recorded]
          Row 3 – Density-weight schedule.                                 [only if dw recorded]

        @param save_path Path to save the figure (empty string → skip saving).
        @param show      Whether to call plt.show().
        @return matplotlib Figure object.
        """
        if not self.history['iterations']:
            logging.warning("No metrics recorded. Cannot plot.")
            return None

        iterations  = np.array(self.history['iterations'])
        wl          = np.array([v if v is not None else np.nan for v in self.history['wirelength']])
        density     = np.array([v if v is not None else np.nan for v in self.history['density']])
        elapsed     = np.array([v if v is not None else np.nan for v in self.history['elapsed_seconds']])
        iter_time   = np.array([v if v is not None else np.nan for v in self.history['iter_time']])
        dw          = np.array([v if v is not None else np.nan for v in self.history['density_weight']])

        has_wl      = not np.all(np.isnan(wl))
        has_density = not np.all(np.isnan(density))
        has_timing  = not np.all(np.isnan(elapsed))
        has_dw      = not np.all(np.isnan(dw))

        # Decide subplot layout
        n_rows = 1 + int(has_timing) + int(has_dw)
        row_heights = [3] + ([1.2] if has_timing else []) + ([1] if has_dw else [])

        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except OSError:
            plt.style.use('seaborn-whitegrid')

        fig, axes = plt.subplots(
            n_rows, 1,
            figsize=(11, 4 * n_rows),
            dpi=120,
            gridspec_kw={'height_ratios': row_heights},
            sharex=False,
        )
        if n_rows == 1:
            axes = [axes]
        fig.patch.set_facecolor('white')

        # ── Row 0: Wirelength + Density ──────────────────────────────────
        ax1 = axes[0]
        ax1.set_facecolor('white')
        lines, labels_leg = [], []

        if has_wl:
            ln1, = ax1.plot(iterations, wl, color=self.COLORS['wl'], linewidth=2,
                            label='Wirelength', alpha=0.9)
            ax1.fill_between(iterations, wl, alpha=0.08, color=self.COLORS['wl'])
            ax1.set_ylabel('Wirelength', color=self.COLORS['wl'], fontsize=11, fontweight='500')
            ax1.tick_params(axis='y', labelcolor=self.COLORS['wl'])
            ax1.scatter([iterations[0], iterations[-1]], [wl[0], wl[-1]],
                        color=self.COLORS['wl'], s=50, zorder=5,
                        edgecolors='white', linewidth=1.5)
            lines.append(ln1); labels_leg.append('Wirelength')

        if has_density:
            ax2 = ax1.twinx() if has_wl else ax1
            ln2, = ax2.plot(iterations, density, color=self.COLORS['density'], linewidth=2,
                            label='Density', alpha=0.9, linestyle='--' if has_wl else '-')
            ax2.fill_between(iterations, density, alpha=0.06, color=self.COLORS['density'])
            ax2.set_ylabel('Density', color=self.COLORS['density'], fontsize=11, fontweight='500')
            ax2.tick_params(axis='y', labelcolor=self.COLORS['density'])
            ax2.scatter([iterations[0], iterations[-1]], [density[0], density[-1]],
                        color=self.COLORS['density'], s=50, zorder=5,
                        edgecolors='white', linewidth=1.5)
            ax2.spines['right'].set_color(self.COLORS['density'])
            ax2.spines['right'].set_linewidth(1.5)
            lines.append(ln2); labels_leg.append('Density')

        ax1.set_title('Optimization Progress', fontsize=14, fontweight='600',
                      color=self.COLORS['text'], pad=12)
        ax1.legend(lines, labels_leg, loc='upper right', frameon=True,
                   fancybox=True, framealpha=0.9, edgecolor=self.COLORS['grid'])
        for sp in ['top']:
            ax1.spines[sp].set_visible(False)
        ax1.spines['left'].set_color(self.COLORS['wl'] if has_wl else self.COLORS['text'])
        ax1.spines['left'].set_linewidth(1.5)
        ax1.spines['bottom'].set_color(self.COLORS['grid'])
        ax1.grid(True, alpha=0.3, linestyle='-', color=self.COLORS['grid'])
        ax1.set_axisbelow(True)

        # Summary annotation
        ann_parts = []
        if has_wl:
            wl_chg = (wl[-1] - wl[0]) / wl[0] * 100 if wl[0] != 0 else 0
            ann_parts.append(f'WL:  {wl[0]:.1f} → {wl[-1]:.1f}  ({wl_chg:+.1f}%)')
        if has_density:
            ann_parts.append(f'Den: {density[0]:.4f} → {density[-1]:.4f}')
        if has_timing:
            total_s = elapsed[~np.isnan(elapsed)][-1]
            ann_parts.append(f'Total time: {total_s:.1f} s')
        if ann_parts:
            ax1.annotate('\n'.join(ann_parts),
                         xy=(0.02, 0.02), xycoords='axes fraction',
                         fontsize=9, fontfamily='monospace', verticalalignment='bottom',
                         bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                   edgecolor=self.COLORS['grid'], alpha=0.9))

        row_idx = 1

        # ── Row 1: Timing ────────────────────────────────────────────────
        if has_timing:
            ax_t = axes[row_idx]; row_idx += 1
            ax_t.set_facecolor('white')

            COLOR_TIME = '#27AE60'
            COLOR_CUM  = '#8E44AD'

            valid = ~np.isnan(iter_time)
            if valid.any():
                ax_t.bar(iterations[valid], iter_time[valid], width=max(iterations[1] - iterations[0], 1) * 0.7,
                         color=COLOR_TIME, alpha=0.6, label='Interval time (s)')
            ax_t.set_ylabel('Interval (s)', color=COLOR_TIME, fontsize=10)
            ax_t.tick_params(axis='y', labelcolor=COLOR_TIME)

            ax_t2 = ax_t.twinx()
            valid_e = ~np.isnan(elapsed)
            if valid_e.any():
                ax_t2.plot(iterations[valid_e], elapsed[valid_e], color=COLOR_CUM,
                           linewidth=2, label='Cumulative (s)')
            ax_t2.set_ylabel('Cumulative (s)', color=COLOR_CUM, fontsize=10)
            ax_t2.tick_params(axis='y', labelcolor=COLOR_CUM)
            ax_t2.spines['right'].set_color(COLOR_CUM)

            h1, l1 = ax_t.get_legend_handles_labels()
            h2, l2 = ax_t2.get_legend_handles_labels()
            ax_t.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=9,
                        frameon=True, framealpha=0.9)
            ax_t.set_title('Iteration Wall Time', fontsize=11, fontweight='500',
                           color=self.COLORS['text'])
            ax_t.grid(True, alpha=0.3, color=self.COLORS['grid'])
            ax_t.set_axisbelow(True)

        # ── Row 2: Density weight ─────────────────────────────────────────
        if has_dw:
            ax_dw = axes[row_idx]; row_idx += 1
            ax_dw.set_facecolor('white')
            COLOR_DW = '#E67E22'
            valid_dw = ~np.isnan(dw)
            ax_dw.plot(iterations[valid_dw], dw[valid_dw], color=COLOR_DW,
                       linewidth=2, label='Density Weight')
            ax_dw.fill_between(iterations[valid_dw], dw[valid_dw],
                                alpha=0.1, color=COLOR_DW)
            ax_dw.set_ylabel('DensityWeight', color=COLOR_DW, fontsize=10)
            ax_dw.tick_params(axis='y', labelcolor=COLOR_DW)
            ax_dw.set_title('Density Penalty Weight Schedule', fontsize=11,
                            fontweight='500', color=self.COLORS['text'])
            ax_dw.set_xlabel('Iteration', fontsize=10, color=self.COLORS['text'])
            ax_dw.grid(True, alpha=0.3, color=self.COLORS['grid'])
            ax_dw.set_axisbelow(True)

        # X-label on bottom-most axis
        axes[-1].set_xlabel('Iteration', fontsize=11, fontweight='500',
                            color=self.COLORS['text'])

        plt.tight_layout(pad=1.5)

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            logging.info("Metrics plot saved to %s", save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig
    
    def clear(self):
        """Clear all recorded history"""
        for key in self.history:
            self.history[key] = []


def build_net_data_structures(nets, num_pins, device):
    """
    @brief Build net data structures for wirelength computation
    @param nets list of nets, each net is a list of pin indices
    @param num_pins total number of pins
    @param device torch device
    @return flat_net2pin_map, flat_net2pin_start_map, pin2net_map, net_weights, net_mask
    """
    # Build flat_net2pin_map and flat_net2pin_start_map
    flat_net2pin_map = []
    flat_net2pin_start_map = [0]
    
    for net in nets:
        flat_net2pin_map.extend(net)
        flat_net2pin_start_map.append(len(flat_net2pin_map))
    
    flat_net2pin_map = torch.tensor(flat_net2pin_map, dtype=torch.int64, device=device)
    flat_net2pin_start_map = torch.tensor(flat_net2pin_start_map, dtype=torch.int64, device=device)
    
    # Build pin2net_map
    pin2net_map = torch.full((num_pins,), -1, dtype=torch.int32, device=device)
    for net_id, net in enumerate(nets):
        for pin_id in net:
            pin2net_map[pin_id] = net_id
    
    # Net weights (all ones by default)
    num_nets = len(nets)
    net_weights = torch.ones(num_nets, dtype=torch.float32, device=device)
    
    # Net mask (all True by default, skip single-pin nets)
    net_mask = torch.tensor([len(net) >= 2 for net in nets], dtype=torch.bool, device=device)
    
    return flat_net2pin_map, flat_net2pin_start_map, pin2net_map, net_weights, net_mask



# ---------------------------------------------------------------------------
# test_wirelength_optimization / test_electric_potential_optimization /
# test_wirelength_and_density_optimization 已废弃删除。
# 主入口统一使用 test_optimization_from_segment_assignments()。
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Build EdgePlaces from segment_assignments JSON (with reuse / isomorphic)
# ---------------------------------------------------------------------------

def build_edge_places_from_segment_assignments(
    segment_assignments_path: str,
    params,
    placedb,
) -> Tuple[
    List[Edge],
    List["EdgePlace"],
    List[List[int]],
    Dict[int, Tuple[int, int]],
    Dict[str, Tuple[int, int]],
    List[List[int]],
]:
    """
    @brief Parse a segment_assignments JSON and construct Edge / EdgePlace objects.

    For each master segment that has N block instances (segment_insts), ONE base
    Edge is created from the first inst, and the remaining N-1 insts become
    *reuse edges* that share the base edge's pin distribution (with a 1-D
    translation offset).

    Pin width comes directly from the ``assigned_pins`` entries in the JSON, so
    pingroup.json is NOT needed.

    Net connectivity is derived from the ``net_id`` field embedded in each
    assigned pin.

    @param segment_assignments_path  Path to segment_assignments_*.json.
    @param params                    Params object (forwarded to EdgePlace).
    @param placedb                   PlaceDB object (forwarded to EdgePlace).

    @return (edges, edge_places, reuse_group, pin_id_to_local, pin_name_to_local, nets)
        edges             – list of Edge objects (base edges first, then their reuse
                            copies, grouped by master segment).
        edge_places       – parallel list of EdgePlace objects.
        reuse_group       – list of groups [[base_edge_id, reuse_id_1, …], …];
                            empty list if no reuse exists.
        pin_id_to_local   – dict  pin_id → (edge_id, local_pin_idx).
        pin_name_to_local – dict  "parent_inst.pingroup_name" → (edge_id, local_pin_idx);
                            used by write_pin_positions_by_name for result export.
        nets              – list of nets, each net is a list of optimisation-global
                            pin indices (ready for build_net_data_structures).
    """

    logging.info("Loading segment assignments from %s", segment_assignments_path)
    with open(segment_assignments_path, "r", encoding="utf-8") as f:
        sa_data = json.load(f)

    seg_assignments = sa_data["segment_assignments"]
    logging.info("Master segments in file: %d", len(seg_assignments))

    # ------------------------------------------------------------------
    # Pass 1: collect inst-level info, ordered by master segment id
    # ------------------------------------------------------------------
    # Each element: (master_seg_id, block_id, inst_data_dict)
    inst_records: List[Tuple[int, int, Dict]] = []
    for seg_id_str, seg_obj in seg_assignments.items():
        master_seg_id = int(seg_id_str)
        for blk_id_str, inst in seg_obj["segment_insts"].items():
            inst_records.append((master_seg_id, int(blk_id_str), inst))

    # Sort by (master_seg_id, block_id) for deterministic ordering
    inst_records.sort(key=lambda r: (r[0], r[1]))

    # ------------------------------------------------------------------
    # Pass 2: create Edge + EdgePlace per inst, build reuse_group
    # ------------------------------------------------------------------
    edges: List[Edge] = []
    edge_places: List[EdgePlace] = []
    reuse_group: List[List[int]] = []
    pin_id_to_local: Dict[int, Tuple[int, int]] = {}
    # net_id -> list of opt-global pin indices
    net_pin_map: Dict[int, List[int]] = {}

    # Track first edge_id per master segment for reuse linkage
    master_to_base_edge: Dict[int, int] = {}

    # Name-based lookup: "parent_inst.pingroup_name" -> (edge_id, local_pin_idx)
    # Used by write_pin_positions_to_result to match pins in pingroup.json
    pin_name_to_local: Dict[str, Tuple[int, int]] = {}

    global_pin_offset = 0  # running offset for opt-global pin indexing

    for master_seg_id, _blk_id, inst in inst_records:
        coords = inst["coordinates"]  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = (float(coords[0]), float(coords[1]),
                           float(coords[2]), float(coords[3]))
        direction_int = inst.get("direction", inst.get("direction", 0))

        if direction_int == 0:
            direction = "horizontal"
            fixed_val = y1
            start_val = min(x1, x2)
            end_val   = max(x1, x2)
        else:
            direction = "vertical"
            fixed_val = x1
            start_val = min(y1, y2)
            end_val   = max(y1, y2)

        # Collect pins on this inst, sorted by pin id for stable ordering
        # Placeholder pins (id == -1) have no 'width'; keep them for pin-count
        # consistency with reuse edges, and fall back to the minimum width.
        _MIN_PIN_WIDTH = 0.04
        raw_pins = inst.get("assigned_pins", [])
        sorted_pins = sorted(raw_pins, key=lambda p: p.get("id", -1))
        pin_widths = [float(p.get("width", _MIN_PIN_WIDTH)) for p in sorted_pins]

        edge_id = len(edges)
        edge = Edge(start_val, end_val, fixed_val, direction, edge_id)
        edge.add_pin_width_list(pin_widths)
        edges.append(edge)
        edge_places.append(EdgePlace(params, edge, placedb))

        # Build pin_id_to_local, pin_name_to_local and net connectivity
        for local_idx, p in enumerate(sorted_pins):
            pid = p.get("id", -1)
            pin_id_to_local[pid] = (edge_id, local_idx)
            pname = p.get("name", "")
            if pname:
                pin_name_to_local[pname] = (edge_id, local_idx)
            opt_global_idx = global_pin_offset + local_idx
            nid = p.get("net_id", -1)
            if nid >= 0:
                net_pin_map.setdefault(nid, []).append(opt_global_idx)

        global_pin_offset += len(pin_widths)

        # Reuse linkage: first inst of a master segment is the base
        if master_seg_id not in master_to_base_edge:
            master_to_base_edge[master_seg_id] = edge_id
        else:
            base_id = master_to_base_edge[master_seg_id]
            # Find or create the reuse group for this master segment
            found = False
            for grp in reuse_group:
                if grp[0] == base_id:
                    grp.append(edge_id)
                    found = True
                    break
            if not found:
                reuse_group.append([base_id, edge_id])

    # ------------------------------------------------------------------
    # Pass 3: build nets list (sorted by net_id for reproducibility)
    # ------------------------------------------------------------------
    nets: List[List[int]] = []
    for _nid in sorted(net_pin_map.keys()):
        pin_list = net_pin_map[_nid]
        if len(pin_list) >= 2:
            nets.append(pin_list)

    num_pins = global_pin_offset
    num_reuse = sum(len(g) - 1 for g in reuse_group)
    logging.info(
        "build_edge_places_from_segment_assignments: "
        "%d edges (%d base + %d reuse), %d pins, %d nets",
        len(edges), len(edges) - num_reuse, num_reuse,
        num_pins, len(nets),
    )
    return edges, edge_places, reuse_group, pin_id_to_local, pin_name_to_local, nets


def write_pin_positions_by_name(
    pingroup_path: str,
    edges: List["Edge"],
    edge_places: List["EdgePlace"],
    pin_name_to_local: Dict[str, Tuple[int, int]],
    result_path: str,
) -> None:
    """
    @brief 将优化后的 pin 坐标写回 pingroup.json 结构，保存为 result.json。

    使用 "parent_inst.pingroup_name" 作为查找键，与
    build_edge_places_from_segment_assignments 返回的 pin_name_to_local 配套。
    适用于 segment_assignments 流程（pin id 为非连续真实 id，不能用顺序计数器匹配）。

    @param pingroup_path       原始 pingroup.json 路径
    @param edges               优化使用的 Edge 对象列表
    @param edge_places         优化完成的 EdgePlace 对象列表
    @param pin_name_to_local   "parent_inst.pingroup_name" → (edge_id, local_pin_idx)
    @param result_path         结果文件保存路径
    """
    with open(pingroup_path, "r", encoding="utf-8") as f:
        pingroup_data = json.load(f)

    unassigned: List[str] = []
    total = 0
    for net_pins in pingroup_data:
        for pin_data in net_pins:
            total += 1
            pname = pin_data.get("parent_inst", "") + "." + pin_data.get("pingroup_name", "")
            if pname in pin_name_to_local:
                edge_id, local_idx = pin_name_to_local[pname]
                edge = edges[edge_id]
                pos_1d = edge_places[edge_id].pos[0].detach().cpu().numpy()
                coord = float(pos_1d[local_idx])
                if edge.direction == "horizontal":
                    x, y = coord, float(edge.fixed_val)
                else:
                    x, y = float(edge.fixed_val), coord
                pin_data["scope"] = [round(x, 6), round(y, 6)]
            else:
                unassigned.append(pname)

    if unassigned:
        logging.warning(
            "write_pin_positions_by_name: %d pin(s) not found in any segment"
            " — scope kept as []: names=%s%s",
            len(unassigned),
            str(unassigned[:20]),
            " ..." if len(unassigned) > 20 else "",
        )

    os.makedirs(os.path.dirname(os.path.abspath(result_path)), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(pingroup_data, f, indent=4, ensure_ascii=False)
    logging.info("Pin positions written to %s  (%d pins total, %d unassigned)",
                 result_path, total, len(unassigned))


def test_optimization_from_segment_assignments(
    segment_assignments_path: str,
    pingroup_path: str = "",
    params_path: str = "params.json",
    max_iterations: int = 600,
    density_weight_init: float = 100.0,
    log_interval: int = 50,
    record_interval: int = 5,
    result_dir: str = "benchmark/result",
    density_weight_update_interval: int = 10,
    target_scale_high: float = 2.0,
    target_scale_low: float = 0.05,
    enable_plot: bool = False,
):
    """
    @brief Full WL + Density optimisation driven by a segment_assignments JSON.

    Builds EdgePlaces (with reuse/isomorphic groups), nets, and runs the
    Adam-based optimisation loop.  Metrics are recorded and plotted; timing
    information is logged and included in the plot.

    @param segment_assignments_path  Path to the segment_assignments_*.json file.
    @param pingroup_path      Path to pingroup.json (only used for result export;
                              leave empty to skip writing result.json).
    @param params_path        Path to the params JSON file.
    @param max_iterations     Number of optimiser iterations.
    @param density_weight_init  Initial multiplier for per-edge density weights.
    @param log_interval       Print a log line every N iterations.
    @param record_interval    Record metrics for plotting every N iterations.
    @param result_dir         Directory for result.json output.
    @param density_weight_update_interval  Adapt per-edge density weights every N iters.
    @param target_scale_high  Upper target ratio at start of schedule.
    @param target_scale_low   Lower target ratio at end of schedule.
    @param enable_plot        If True, save metrics plot to result_dir.
    @return (place_obj, edge_places, visualizer)
    """
    logging.info("=" * 70)
    logging.info("Optimization from segment_assignments JSON")
    logging.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Params & minimal PlaceDB stub
    # ------------------------------------------------------------------
    params = Params.Params()
    params.load(params_path)
    device = torch.device("cuda" if params.gpu else "cpu")

    placedb = PlaceDB.__new__(PlaceDB)
    placedb.all_modules_list = []
    placedb.all_edges_list = []

    # ------------------------------------------------------------------
    # 2. Build EdgePlaces, reuse groups, nets
    # ------------------------------------------------------------------
    edges, edge_places, reuse_group, pin_id_to_local, pin_name_to_local, nets = \
        build_edge_places_from_segment_assignments(
            segment_assignments_path, params, placedb)

    if reuse_group:
        reuse_edge_ids = {rid for grp in reuse_group for rid in grp[1:]}
    else:
        reuse_edge_ids = set()

    # Total optimisation pin count
    num_pins = sum(len(e.pin_widths) for e in edges)
    num_nets = len(nets)
    placedb.total_pin_count = num_pins
    placedb.nets_list = nets
    logging.info("Edges: %d (%d base, %d reuse)  |  Pins: %d  |  Nets: %d",
                 len(edges), len(edges) - len(reuse_edge_ids),
                 len(reuse_edge_ids), num_pins, num_nets)

    # ------------------------------------------------------------------
    # 3. Build flat net data structures
    # ------------------------------------------------------------------
    flat_net2pin_map, flat_net2pin_start_map, pin2net_map, net_weights, net_mask = \
        build_net_data_structures(nets, num_pins, device)

    # ------------------------------------------------------------------
    # 4. PlaceObj
    # ------------------------------------------------------------------
    global_place_params = {
        "wirelength": "weighted_average",
        "learning_rate": 0.1,
    }
    place_obj = PlaceObj(
        density_weight=0.0,
        params=params,
        placedb=placedb,
        edges=edges,
        edge_places=edge_places,
        global_place_params=global_place_params,
        flat_net2pin_map=flat_net2pin_map,
        flat_net2pin_start_map=flat_net2pin_start_map,
        pin2net_map=pin2net_map,
        net_weights=net_weights,
        net_mask=net_mask,
        reuse_group=reuse_group if reuse_group else None,
    ).to(device)

    pw = float(getattr(params, "density_weight", 1.0))
    if pw <= 0:
        pw = 1.0
    place_obj.density_weight.mul_(density_weight_init / pw)

    # ------------------------------------------------------------------
    # 5. Optimizer (skip reuse edges — they share the base edge's params)
    # ------------------------------------------------------------------
    all_opt_params: List = []
    for eid, ep in enumerate(edge_places):
        if eid not in reuse_edge_ids:
            all_opt_params.extend(ep.parameters())
    optimizer = torch.optim.Adam(all_opt_params, lr=0.5)

    # ------------------------------------------------------------------
    # 6. Initial metrics
    # ------------------------------------------------------------------
    all_edge_positions = {eid: ep.pos[0] for eid, ep in enumerate(edge_places)}
    with torch.no_grad():
        initial_wl      = place_obj.obj_wl_test(all_edge_positions).item()
        initial_density = place_obj.obj_density_test(all_edge_positions).item()
    logging.info("Initial  WL=%.4f  Density=%.4f", initial_wl, initial_density)

    # ------------------------------------------------------------------
    # 7. Visualizer
    # ------------------------------------------------------------------
    metrics_save_dir = os.path.dirname(os.path.abspath(segment_assignments_path))
    visualizer = OptimizationVisualizer(save_path=metrics_save_dir)
    visualizer.record(
        iteration=0,
        wirelength=initial_wl,
        density=initial_density,
        total_obj=None,
        elapsed=0.0,
        iter_time=0.0,
        density_weight=float(place_obj.density_weight.mean().item()),
    )

    # ------------------------------------------------------------------
    # 8. Sync helper for reuse edges
    # ------------------------------------------------------------------
    def sync_reuse_positions():
        if not reuse_group:
            return
        with torch.no_grad():
            for grp in reuse_group:
                base_id = grp[0]
                base_start = edge_places[base_id].start_point
                for rid in grp[1:]:
                    offset = edge_places[rid].start_point - base_start
                    edge_places[rid].pos[0].data.copy_(
                        edge_places[base_id].pos[0].data + offset)

    # ------------------------------------------------------------------
    # 9. Optimization loop
    # ------------------------------------------------------------------
    density_w = place_obj.density_weight.detach().clone()
    logging.info("\n--- Starting Optimization (WL + Density) ---")

    t_loop_start  = time.time()
    t_last_record = t_loop_start

    for iteration in range(max_iterations):
        t_iter_start = time.time()

        # Boundary clamp (skip reuse edges)
        for eid, ep in enumerate(edge_places):
            if eid not in reuse_edge_ids:
                ep.op_collections.move_boundary_op(ep.pos[0])

        optimizer.zero_grad()

        all_edge_positions = {eid: ep.pos[0] for eid, ep in enumerate(edge_places)}
        do_pe = (
            density_weight_update_interval > 0
            and iteration > 0
            and iteration % density_weight_update_interval == 0
        )
        objective, wl_grad_norm, density_grad_norm, wl_pe, raw_d = \
            place_obj.obj_wl_density_test(
                all_edge_positions, density_w, compute_per_edge_norms=do_pe)

        objective.backward()
        optimizer.step()

        # Keep reuse edges in sync
        sync_reuse_positions()

        # Per-edge density weight adaptation
        if do_pe and raw_d is not None and wl_pe is not None:
            ts = density_weight_target_scale(
                iteration, max_iterations, target_scale_high, target_scale_low)
            density_w = update_per_edge_density_weights(
                density_w, wl_pe, raw_d,
                place_obj.base_edge_indices, ts)
            place_obj.density_weight.data.copy_(density_w)

        # Recording & logging
        is_record = (iteration % record_interval == 0) or (iteration == max_iterations - 1)
        is_log    = ((iteration + 1) % log_interval == 0) or (iteration == max_iterations - 1)

        if is_record or is_log:
            t_now = time.time()
            elapsed_total = t_now - t_loop_start
            interval_time = t_now - t_last_record
            t_last_record = t_now

            with torch.no_grad():
                wl      = place_obj.obj_wl_test(all_edge_positions).item()
                density = place_obj.obj_density_test(all_edge_positions).item()

            if is_record:
                dw_mean = float(density_w.mean().item())
                visualizer.record(
                    iteration=iteration + 1,
                    wirelength=wl,
                    density=density,
                    total_obj=objective.item(),
                    elapsed=elapsed_total,
                    iter_time=interval_time,
                    density_weight=dw_mean,
                )

            if is_log:
                dw_np = density_w.detach().cpu().numpy()
                logging.info(
                    "Iter %4d | WL=%.4f  Density=%.4f  Total=%.4f  "
                    "DW_mean=%.4f min=%.4f max=%.4f  "
                    "WLGrad=%.4f  DGrad=%.4f  |  +%.1fs  (total %.1fs)",
                    iteration + 1, wl, density, objective.item(),
                    float(dw_np.mean()), float(dw_np.min()), float(dw_np.max()),
                    wl_grad_norm, density_grad_norm,
                    interval_time, elapsed_total,
                )

    total_time = time.time() - t_loop_start
    logging.info("Optimization finished in %.2f s  (avg %.3f s/iter)",
                 total_time, total_time / max(max_iterations, 1))

    # ------------------------------------------------------------------
    # 10. Final metrics
    # ------------------------------------------------------------------
    all_edge_positions = {eid: ep.pos[0] for eid, ep in enumerate(edge_places)}
    with torch.no_grad():
        final_wl      = place_obj.obj_wl_test(all_edge_positions).item()
        final_density = place_obj.obj_density_test(all_edge_positions).item()
    logging.info("\nFinal WL=%.4f  (%.2f%%)  Density=%.4f",
                 final_wl,
                 (final_wl - initial_wl) / max(initial_wl, 1e-12) * 100,
                 final_density)

    # ------------------------------------------------------------------
    # 11. Plot metrics
    # ------------------------------------------------------------------
    if enable_plot:
        os.makedirs(result_dir, exist_ok=True)
        metrics_png = os.path.join(result_dir, "metrics_segment_assignments.png")
        visualizer.plot_metrics(save_path=metrics_png, show=False)

    # ------------------------------------------------------------------
    # 12. (Optional) Write result.json
    # ------------------------------------------------------------------
    if pingroup_path:
        result_json = os.path.join(result_dir, "result.json")
        write_pin_positions_by_name(
            pingroup_path=pingroup_path,
            edges=edges,
            edge_places=edge_places,
            pin_name_to_local=pin_name_to_local,
            result_path=result_json,
        )

    return place_obj, edge_places, visualizer


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)-7s] %(name)s - %(message)s',
                        stream=sys.stdout)

    parser = argparse.ArgumentParser(description="NonLinearPlace optimization")
    parser.add_argument("--segment-assignments", required=True)
    parser.add_argument("--pingroup", default="")
    parser.add_argument("--params", default="params.json")
    parser.add_argument("--result-dir", default="result")
    parser.add_argument("--iterations", type=int, default=600)
    args = parser.parse_args()

    test_optimization_from_segment_assignments(
        segment_assignments_path=args.segment_assignments,
        pingroup_path=args.pingroup,
        params_path=args.params,
        result_dir=args.result_dir,
        max_iterations=args.iterations,
    )