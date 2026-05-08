"""
Stage 1: MCTS-based segment assignment.

Determines which segment each pin should be placed on.
"""

import sys
import os
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Add this directory to sys.path so internal imports in the MCTS code work
_STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
if _STAGE_DIR not in sys.path:
    sys.path.insert(0, _STAGE_DIR)


def run_mcts(
    block_json: str,
    pingroup_json: str,
    output_dir: str,
    num_simulations: int = 1000,
    time_limit: float = 30.0,
) -> str:
    """Run MCTS segment assignment pipeline.

    Args:
        block_json: Path to block.json.
        pingroup_json: Path to pingroup.json.
        output_dir: Directory where segment_assignments.json will be written.
        num_simulations: Number of MCTS simulations per processing unit.
        time_limit: Time limit in seconds per processing unit.

    Returns:
        Path to the generated segment_assignments.json file.
    """
    from complete_mcts_pipeline import CompleteMCTSPipeline
    from segment_usages_direct_interface import create_segment_usages_direct_interface

    block_json = str(Path(block_json).resolve())
    pingroup_json = str(Path(pingroup_json).resolve())
    output_dir = str(Path(output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)

    if not Path(block_json).exists():
        raise FileNotFoundError(f"Block file not found: {block_json}")
    if not Path(pingroup_json).exists():
        raise FileNotFoundError(f"Pingroup file not found: {pingroup_json}")

    logger.info("=" * 70)
    logger.info("Stage 1: MCTS Segment Assignment")
    logger.info("=" * 70)
    logger.info(f"  block_json: {block_json}")
    logger.info(f"  pingroup_json: {pingroup_json}")
    logger.info(f"  num_simulations: {num_simulations}")
    logger.info(f"  time_limit: {time_limit}")

    pipeline = CompleteMCTSPipeline()
    pipeline.block_file = block_json
    pipeline.pingroup_file = pingroup_json

    success = pipeline.run_complete_mcts_pipeline(
        block_json, pingroup_json, num_simulations, time_limit
    )

    if not success:
        raise RuntimeError("MCTS pipeline failed.")

    # Save detailed results
    detailed_dir = os.path.join(output_dir, "mcts_detailed")
    os.makedirs(detailed_dir, exist_ok=True)
    timestamp = int(time.time())
    detailed_output_file = os.path.join(
        detailed_dir, f"mcts_assignment_results_{timestamp}.json"
    )
    pipeline.save_results(detailed_output_file)
    logger.info(f"Detailed MCTS results saved to: {detailed_output_file}")

    # Convert to segment_assignments format
    interface = create_segment_usages_direct_interface(pipeline.floorplan)
    segment_usages = interface.load_mcts_segment_usages(detailed_output_file)

    if not segment_usages:
        raise RuntimeError("Failed to load segment_usages from MCTS results.")

    final_result = interface.convert_from_mcts_segment_usages(segment_usages)

    output_file = os.path.join(output_dir, "segment_assignments.json")
    interface.export_segment_assignments(final_result, output_file)
    logger.info(f"Segment assignments written to: {output_file}")

    return output_file
