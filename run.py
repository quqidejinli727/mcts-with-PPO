#!/usr/bin/env python3
"""
PinAssignFlow - Top-level entry point for the complete Pin Assignment flow.

Usage:
    python run.py --case benchmark/case2 --output output/case2
    python run.py --case benchmark/case2 --output output/case2 --evaluate path/to/evaluate.py
    python run.py --case benchmark/case2 --output output/case2 --skip-mcts --segment-assignments path/to/file.json
    python run.py --case benchmark/case2 --output output/case2 --skip-mcts --skip-nlplace --result path/to/result.json --segment-assignments path/to/segment_assignments.json --evaluate path/to/evaluate.py
"""

import argparse
from dataclasses import fields
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

_STAGE1_CONFIG_PATH_FIELDS = {
    "block_json_path",
    "pingroup_json_path",
    "assignment_output_path",
    "results_root",
    "interface_result_dir",
}


def _stage1_config_class():
    """Load Stage 1 RunConfig without changing Stage 2/3 behavior."""
    from stage1_mcts.config import RunConfig

    return RunConfig


def _stage1_arg_name(field_name: str) -> str:
    return "--stage1-" + field_name.replace("_", "-")


def _str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def _add_stage1_config_arguments(stage1: argparse._ArgumentGroup) -> None:
    """Expose Stage 1 RunConfig fields as --stage1-* command-line overrides."""
    RunConfig = _stage1_config_class()
    for config_field in fields(RunConfig):
        name = config_field.name
        if name in _STAGE1_CONFIG_PATH_FIELDS:
            continue
        option = _stage1_arg_name(name)
        dest = f"stage1_{name}"
        default = argparse.SUPPRESS
        current_value = getattr(RunConfig(), name)
        help_text = f"Override Stage 1 config.{name}; default: {current_value!r}."

        if isinstance(current_value, bool):
            stage1.add_argument(
                option,
                dest=dest,
                type=_str_to_bool,
                nargs="?",
                const=True,
                default=default,
                metavar="{true,false}",
                help=help_text,
            )
            stage1.add_argument(
                "--no-stage1-" + name.replace("_", "-"),
                dest=dest,
                action="store_false",
                default=default,
                help=f"Set Stage 1 config.{name}=False.",
            )
        elif isinstance(current_value, int) and not isinstance(current_value, bool):
            stage1.add_argument(option, dest=dest, type=int, default=default, help=help_text)
        elif isinstance(current_value, float):
            stage1.add_argument(option, dest=dest, type=float, default=default, help=help_text)
        elif isinstance(current_value, Path):
            stage1.add_argument(option, dest=dest, type=Path, default=default, help=help_text)
        else:
            stage1.add_argument(option, dest=dest, type=str, default=default, help=help_text)


def _stage1_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Collect only Stage 1 command-line values that the user explicitly set."""
    RunConfig = _stage1_config_class()
    overrides: dict[str, Any] = {}
    for config_field in fields(RunConfig):
        name = config_field.name
        if name in _STAGE1_CONFIG_PATH_FIELDS:
            continue
        attr = f"stage1_{name}"
        if hasattr(args, attr):
            overrides[name] = getattr(args, attr)
    if args.num_simulations is not None:
        overrides["simulations"] = args.num_simulations
    return overrides


class TeeStream:
    """Write text to multiple streams, used to keep terminal output and log files in sync."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return self.streams[0].isatty()

    @property
    def encoding(self):
        return getattr(self.streams[0], "encoding", "utf-8")


def setup_logging(output_dir: str, log_dir: str = None):
    """Configure logging and mirror stdout/stderr to a timestamped run log."""
    log_root = Path(log_dir).resolve() if log_dir else Path(output_dir).resolve() / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"run_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.log"
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)-7s] %(name)s - %(message)s",
        stream=sys.stdout,
        force=True,
    )
    logging.info(f"Run log: {log_path}")

    return log_file, log_path, original_stdout, original_stderr


def run_full_flow(
    case_dir: str,
    output_dir: str,
    # MCTS parameters
    num_simulations: int | None = None,
    time_limit: float = 30.0,
    stage1_config_overrides: dict[str, Any] | None = None,
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
    # External evaluation script
    evaluate_path: str = None,
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
        evaluate_path: Path to external evaluate.py script. If provided, runs after
            Stage 2 (before Stage 3) and again after Stage 3 completes.

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
            num_simulations=(
                num_simulations
                if num_simulations is not None
                else _stage1_config_class()().simulations
            ),
            time_limit=time_limit,
            config_overrides=stage1_config_overrides,
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
    # Evaluate (post-Stage 2)
    # ------------------------------------------------------------------
    if evaluate_path:
        ftpred_path = os.path.join(PROJECT_ROOT, "stage1_mcts", "feedthrough", "build", "ftpred")
        logging.info(f"[Evaluate @ post-Stage2] Running: {evaluate_path}")
        try:
            result = subprocess.run(
                [sys.executable, evaluate_path, result_path, block_json, result_path, ftpred_path],
                capture_output=False,
                check=False,
            )
            if result.returncode == 0:
                logging.info(f"[Evaluate @ post-Stage2] Done (exit 0)")
            else:
                logging.warning(f"[Evaluate @ post-Stage2] Exited with code {result.returncode}")
        except Exception as e:
            logging.error(f"[Evaluate @ post-Stage2] Failed: {e}")

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
            segment_assignments_json=seg_assign_path,
        )
    else:
        final_path = result_path
        logging.info("[Stage 3 SKIPPED]")

    t_stage3 = time.time()
    logging.info(f"Stage 3 elapsed: {t_stage3 - t_stage2:.1f}s")

    # ------------------------------------------------------------------
    # Evaluate (post-Stage 3)
    # ------------------------------------------------------------------
    if evaluate_path:
        ftpred_path = os.path.join(PROJECT_ROOT, "stage1_mcts", "feedthrough", "build", "ftpred")
        logging.info(f"[Evaluate @ post-Stage3] Running: {evaluate_path}")
        try:
            result = subprocess.run(
                [sys.executable, evaluate_path, final_path, block_json, final_path, ftpred_path],
                capture_output=False,
                check=False,
            )
            if result.returncode == 0:
                logging.info(f"[Evaluate @ post-Stage3] Done (exit 0)")
            else:
                logging.warning(f"[Evaluate @ post-Stage3] Exited with code {result.returncode}")
        except Exception as e:
            logging.error(f"[Evaluate @ post-Stage3] Failed: {e}")

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

  # Run full flow with external evaluate script
  python run.py --case benchmark/case2 --output output/case2 \\
      --evaluate path/to/evaluate.py

  # Skip MCTS, use existing segment assignments
  python run.py --case benchmark/case2 --output output/case2 \\
      --skip-mcts --segment-assignments path/to/segment_assignments.json

  # Only run legalization on existing result; pass segment assignments explicitly
  python run.py --case benchmark/case2 --output output/case2 \\
      --skip-mcts --skip-nlplace --result path/to/result.json \\
      --segment-assignments path/to/segment_assignments.json

  # Same as above, with evaluate
  python run.py --case benchmark/case2 --output output/case2 \\
      --skip-mcts --skip-nlplace --result path/to/result.json \\
      --segment-assignments path/to/segment_assignments.json \\
      --evaluate path/to/evaluate.py
""",
    )

    parser.add_argument(
        "--case", required=True, help="Benchmark case directory (contains block.json and pingroup.json)"
    )
    parser.add_argument("--output", required=True, help="Output directory")

    # Stage 1 params
    stage1 = parser.add_argument_group("Stage 1: MCTS")
    stage1.add_argument(
        "--num-simulations",
        type=int,
        default=None,
        help="Compatibility alias for --stage1-simulations; omitted means Stage 1 config default.",
    )
    stage1.add_argument("--time-limit", type=float, default=30.0)
    stage1.add_argument("--skip-mcts", action="store_true")
    stage1.add_argument("--segment-assignments", type=str, default=None,
                        help="Pre-computed segment_assignments.json (requires --skip-mcts)")
    _add_stage1_config_arguments(stage1)

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

    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for per-run log files (default: <output>/logs)",
    )

    parser.add_argument(
        "--evaluate",
        type=str,
        default=None,
        dest="evaluate_path",
        help="Path to external evaluate.py script. If provided, runs after Stage 2 "
             "(before Stage 3) and again after Stage 3 completes.",
    )

    args = parser.parse_args()

    log_file, log_path, original_stdout, original_stderr = setup_logging(
        args.output, args.log_dir
    )
    try:
        final_path = run_full_flow(
            case_dir=args.case,
            output_dir=args.output,
            num_simulations=args.num_simulations,
            time_limit=args.time_limit,
            stage1_config_overrides=_stage1_overrides_from_args(args),
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
            evaluate_path=args.evaluate_path,
        )

        print(f"\nFinal result: {final_path}")
        print(f"Run log: {log_path}")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    main()
