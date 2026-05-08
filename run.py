#!/usr/bin/env python3
"""
PinAssignFlow - Top-level entry point for the complete Pin Assignment flow.

Usage:
    python run.py --case benchmark/case2 --output output/case2
    python run.py --case benchmark/case2 --output output/case2 --skip-mcts --segment-assignments path/to/file.json
    python run.py --case benchmark/case2 --output output/case2 --skip-mcts --skip-nlplace --result path/to/result.json
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def run_full_flow(
    case_dir: str,
    output_dir: str,
    # MCTS parameters
    num_simulations: int = 1000,
    time_limit: float = 30.0,
    # NonLinearPlace parameters
    nlplace_max_iterations: int = 600,
    nlplace_density_weight: float = 100.0,
    nlplace_params_path: str = None,
    enable_plot: bool = False,
    # Legalization parameters
    keepout: float = 0.0,
    hpwl_thresh: float = 500.0,
    max_outer_iter: int = 30,
    # Skip flags
    skip_mcts: bool = False,
    skip_nlplace: bool = False,
    skip_legalization: bool = False,
    # Pre-computed intermediate files (when skipping stages)
    segment_assignments_path: str = None,
    result_json_path: str = None,
) -> str:
    """Execute the complete Pin Assignment flow.

    Args:
        case_dir: Benchmark case directory (must contain block.json and pingroup.json).
        output_dir: Output directory for all intermediate and final results.
        num_simulations: MCTS simulation count per unit.
        time_limit: MCTS time limit per unit (seconds).
        nlplace_max_iterations: NonLinearPlace optimizer iterations.
        nlplace_density_weight: Initial density weight.
        nlplace_params_path: Custom params JSON for NonLinearPlace.
        enable_plot: Save optimization plots.
        keepout: Legalization minimum pin gap.
        hpwl_thresh: Legalization HPWL threshold.
        max_outer_iter: Legalization max outer iterations.
        skip_mcts: Skip Stage 1, use provided segment_assignments_path.
        skip_nlplace: Skip Stage 2, use provided result_json_path.
        skip_legalization: Skip Stage 3.
        segment_assignments_path: Pre-computed segment assignments (when skip_mcts=True).
        result_json_path: Pre-computed result.json (when skip_nlplace=True).

    Returns:
        Path to the final result file.
    """
    case_dir = str(Path(case_dir).resolve())
    output_dir = str(Path(output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)

    block_json = os.path.join(case_dir, "block.json")
    pingroup_json = os.path.join(case_dir, "pingroup.json")

    if not Path(block_json).exists():
        raise FileNotFoundError(f"block.json not found in case directory: {case_dir}")
    if not Path(pingroup_json).exists():
        raise FileNotFoundError(f"pingroup.json not found in case directory: {case_dir}")

    logging.info("=" * 70)
    logging.info("PinAssignFlow - Complete Pin Assignment Pipeline")
    logging.info("=" * 70)
    logging.info(f"Case directory: {case_dir}")
    logging.info(f"Output directory: {output_dir}")

    t_start = time.time()

    # ------------------------------------------------------------------
    # Stage 1: MCTS Segment Assignment
    # ------------------------------------------------------------------
    if not skip_mcts:
        from stage1_mcts import run_mcts

        seg_assign_path = run_mcts(
            block_json=block_json,
            pingroup_json=pingroup_json,
            output_dir=output_dir,
            num_simulations=num_simulations,
            time_limit=time_limit,
        )
    else:
        if segment_assignments_path is not None:
            seg_assign_path = str(Path(segment_assignments_path).resolve())
            if not Path(seg_assign_path).exists():
                raise FileNotFoundError(
                    f"Segment assignments not found: {seg_assign_path}"
                )
            logging.info(f"[Stage 1 SKIPPED] Using: {seg_assign_path}")
        elif skip_nlplace:
            seg_assign_path = None
            logging.info("[Stage 1 SKIPPED] (not needed since Stage 2 also skipped)")
        else:
            raise ValueError(
                "--skip-mcts requires --segment-assignments to be specified."
            )

    t_stage1 = time.time()
    logging.info(f"Stage 1 elapsed: {t_stage1 - t_start:.1f}s")

    # ------------------------------------------------------------------
    # Stage 2: NonLinearPlace Continuous Optimization
    # ------------------------------------------------------------------
    if not skip_nlplace:
        from stage2_nlplace import run_nlplace

        result_path = run_nlplace(
            segment_assignments_path=seg_assign_path,
            pingroup_path=pingroup_json,
            output_dir=output_dir,
            params_path=nlplace_params_path,
            max_iterations=nlplace_max_iterations,
            density_weight_init=nlplace_density_weight,
            enable_plot=enable_plot,
        )
    else:
        if result_json_path is None:
            raise ValueError("--skip-nlplace requires --result to be specified.")
        result_path = str(Path(result_json_path).resolve())
        if not Path(result_path).exists():
            raise FileNotFoundError(f"Result JSON not found: {result_path}")
        logging.info(f"[Stage 2 SKIPPED] Using: {result_path}")

    t_stage2 = time.time()
    logging.info(f"Stage 2 elapsed: {t_stage2 - t_stage1:.1f}s")

    # ------------------------------------------------------------------
    # Stage 3: QP Legalization
    # ------------------------------------------------------------------
    if not skip_legalization:
        from stage3_legalization import run_legalization

        final_path = run_legalization(
            block_json=block_json,
            pingroup_json=pingroup_json,
            result_json=result_path,
            output_dir=output_dir,
            keepout=keepout,
            hpwl_thresh=hpwl_thresh,
            max_outer_iter=max_outer_iter,
        )
    else:
        final_path = result_path
        logging.info("[Stage 3 SKIPPED]")

    t_stage3 = time.time()
    logging.info(f"Stage 3 elapsed: {t_stage3 - t_stage2:.1f}s")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_time = time.time() - t_start
    logging.info("=" * 70)
    logging.info("PinAssignFlow Complete!")
    logging.info(f"  Total time: {total_time:.1f}s")
    logging.info(f"  Final result: {final_path}")
    logging.info("=" * 70)

    return final_path


def main():
    parser = argparse.ArgumentParser(
        description="PinAssignFlow - Complete Pin Assignment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full flow
  python run.py --case benchmark/case2 --output output/case2

  # Skip MCTS, use existing segment assignments
  python run.py --case benchmark/case2 --output output/case2 \\
      --skip-mcts --segment-assignments path/to/segment_assignments.json

  # Only run legalization on existing result
  python run.py --case benchmark/case2 --output output/case2 \\
      --skip-mcts --skip-nlplace --result path/to/result.json
""",
    )

    parser.add_argument(
        "--case", required=True, help="Benchmark case directory (contains block.json and pingroup.json)"
    )
    parser.add_argument("--output", required=True, help="Output directory")

    # Stage 1 params
    stage1 = parser.add_argument_group("Stage 1: MCTS")
    stage1.add_argument("--num-simulations", type=int, default=1000)
    stage1.add_argument("--time-limit", type=float, default=30.0)
    stage1.add_argument("--skip-mcts", action="store_true")
    stage1.add_argument("--segment-assignments", type=str, default=None,
                        help="Pre-computed segment_assignments.json (requires --skip-mcts)")

    # Stage 2 params
    stage2 = parser.add_argument_group("Stage 2: NonLinearPlace")
    stage2.add_argument("--nlplace-iterations", type=int, default=600)
    stage2.add_argument("--nlplace-density-weight", type=float, default=100.0)
    stage2.add_argument("--nlplace-params", type=str, default=None)
    stage2.add_argument("--enable-plot", action="store_true")
    stage2.add_argument("--skip-nlplace", action="store_true")
    stage2.add_argument("--result", type=str, default=None,
                        help="Pre-computed result.json (requires --skip-nlplace)")

    # Stage 3 params
    stage3 = parser.add_argument_group("Stage 3: Legalization")
    stage3.add_argument("--keepout", type=float, default=0.0)
    stage3.add_argument("--hpwl-thresh", type=float, default=500.0)
    stage3.add_argument("--max-outer-iter", type=int, default=30)
    stage3.add_argument("--skip-legalization", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)-7s] %(name)s - %(message)s",
        stream=sys.stdout,
    )

    final_path = run_full_flow(
        case_dir=args.case,
        output_dir=args.output,
        num_simulations=args.num_simulations,
        time_limit=args.time_limit,
        nlplace_max_iterations=args.nlplace_iterations,
        nlplace_density_weight=args.nlplace_density_weight,
        nlplace_params_path=args.nlplace_params,
        enable_plot=args.enable_plot,
        keepout=args.keepout,
        hpwl_thresh=args.hpwl_thresh,
        max_outer_iter=args.max_outer_iter,
        skip_mcts=args.skip_mcts,
        skip_nlplace=args.skip_nlplace,
        skip_legalization=args.skip_legalization,
        segment_assignments_path=args.segment_assignments,
        result_json_path=args.result,
    )

    print(f"\nFinal result: {final_path}")


if __name__ == "__main__":
    main()
