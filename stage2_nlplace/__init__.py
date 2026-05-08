"""
Stage 2: NonLinearPlace continuous optimization.

Determines the exact floating-point coordinate of each pin on its assigned
segment, minimizing total wirelength subject to density constraints.
"""

import sys
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
if _STAGE_DIR not in sys.path:
    sys.path.insert(0, _STAGE_DIR)


def run_nlplace(
    segment_assignments_path: str,
    pingroup_path: str,
    output_dir: str,
    params_path: str = None,
    max_iterations: int = 600,
    density_weight_init: float = 100.0,
    enable_plot: bool = False,
) -> str:
    """Run NonLinearPlace continuous optimization.

    Args:
        segment_assignments_path: Path to segment_assignments.json from Stage 1.
        pingroup_path: Path to pingroup.json (needed for result export).
        output_dir: Directory where result.json will be written.
        params_path: Path to params JSON. If None, uses built-in default.
        max_iterations: Number of optimizer iterations.
        density_weight_init: Initial density weight multiplier.
        enable_plot: Whether to save metrics plots.

    Returns:
        Path to the generated result.json file.
    """
    from NonLinearPlace import test_optimization_from_segment_assignments

    segment_assignments_path = str(Path(segment_assignments_path).resolve())
    pingroup_path = str(Path(pingroup_path).resolve())
    output_dir = str(Path(output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)

    if params_path is None:
        params_path = os.path.join(_STAGE_DIR, "params.json")
    else:
        params_path = str(Path(params_path).resolve())

    if not Path(segment_assignments_path).exists():
        raise FileNotFoundError(
            f"Segment assignments file not found: {segment_assignments_path}"
        )
    if not Path(pingroup_path).exists():
        raise FileNotFoundError(f"Pingroup file not found: {pingroup_path}")

    logger.info("=" * 70)
    logger.info("Stage 2: NonLinearPlace Continuous Optimization")
    logger.info("=" * 70)
    logger.info(f"  segment_assignments: {segment_assignments_path}")
    logger.info(f"  pingroup: {pingroup_path}")
    logger.info(f"  params: {params_path}")
    logger.info(f"  max_iterations: {max_iterations}")

    test_optimization_from_segment_assignments(
        segment_assignments_path=segment_assignments_path,
        pingroup_path=pingroup_path,
        params_path=params_path,
        max_iterations=max_iterations,
        density_weight_init=density_weight_init,
        result_dir=output_dir,
        enable_plot=enable_plot,
    )

    result_path = os.path.join(output_dir, "result.json")
    if not Path(result_path).exists():
        raise RuntimeError(
            f"NonLinearPlace did not produce result.json at: {result_path}"
        )

    logger.info(f"Result written to: {result_path}")
    return result_path
