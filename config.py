"""
PinAssignFlow - Default configuration.

All parameters can be overridden via command-line arguments in run.py.
This module provides centralized defaults and preset configurations for
common use cases.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
OUTPUT_DIR = PROJECT_ROOT / "output"

# ---------------------------------------------------------------------------
# Stage 1: MCTS defaults
# ---------------------------------------------------------------------------
MCTS_NUM_SIMULATIONS = 1000
MCTS_TIME_LIMIT = 30.0  # seconds per processing unit

# ---------------------------------------------------------------------------
# Stage 2: NonLinearPlace defaults
# ---------------------------------------------------------------------------
NLPLACE_MAX_ITERATIONS = 600
NLPLACE_DENSITY_WEIGHT_INIT = 100.0
NLPLACE_LOG_INTERVAL = 50
NLPLACE_RECORD_INTERVAL = 5
NLPLACE_DENSITY_WEIGHT_UPDATE_INTERVAL = 10
NLPLACE_TARGET_SCALE_HIGH = 2.0
NLPLACE_TARGET_SCALE_LOW = 0.05
NLPLACE_PARAMS_PATH = str(PROJECT_ROOT / "stage2_nlplace" / "params.json")

# ---------------------------------------------------------------------------
# Stage 3: Legalization defaults
# ---------------------------------------------------------------------------
LEGAL_KEEPOUT = 0.0
LEGAL_HPWL_THRESH = 500.0
LEGAL_MAX_OUTER_ITER = 30
LEGAL_TOL = 1e-3
LEGAL_ENABLE_HARD_ISO = True


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------
PRESETS = {
    "fast": {
        "num_simulations": 200,
        "time_limit": 10.0,
        "nlplace_max_iterations": 300,
        "max_outer_iter": 15,
    },
    "default": {
        "num_simulations": MCTS_NUM_SIMULATIONS,
        "time_limit": MCTS_TIME_LIMIT,
        "nlplace_max_iterations": NLPLACE_MAX_ITERATIONS,
        "max_outer_iter": LEGAL_MAX_OUTER_ITER,
    },
    "quality": {
        "num_simulations": 3000,
        "time_limit": 60.0,
        "nlplace_max_iterations": 1200,
        "max_outer_iter": 50,
    },
}


def get_case_dir(case_name: str) -> str:
    """Get the benchmark case directory path."""
    return str(BENCHMARK_DIR / case_name)


def get_output_dir(case_name: str) -> str:
    """Get the output directory path for a case."""
    return str(OUTPUT_DIR / case_name)
