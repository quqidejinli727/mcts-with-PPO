"""
Stage 3: QP-based pin legalization.

Eliminates pin-to-pin overlaps while preserving wirelength quality,
enforcing isomorphic constraints for reused modules.
"""

import sys
import os
import json
import logging
import importlib.util
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

_STAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_module_from_stage(module_name, filename):
    """Load a module explicitly from this stage's directory to avoid conflicts."""
    spec = importlib.util.spec_from_file_location(
        f"stage3_{module_name}", os.path.join(_STAGE_DIR, filename)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_legalization(
    block_json: str,
    pingroup_json: str,
    result_json: str,
    output_dir: str,
    keepout: float = 0.0,
    hpwl_thresh: float = 500.0,
    max_outer_iter: int = 30,
    tol: float = 1e-3,
    enable_hard_iso: bool = True,
) -> str:
    """Run QP pin legalization to eliminate overlaps.

    Args:
        block_json: Path to block.json.
        pingroup_json: Path to pingroup.json.
        result_json: Path to result.json from Stage 2.
        output_dir: Directory where result_legalized.json will be written.
        keepout: Minimum gap between adjacent pins.
        hpwl_thresh: HPWL threshold for successor alignment admission.
        max_outer_iter: Maximum outer QP iterations.
        tol: Convergence tolerance.
        enable_hard_iso: Enable hard isomorphic constraints.

    Returns:
        Path to the generated result_legalized.json file.
    """
    # Set environment variables before importing legalizer modules
    os.environ.setdefault("SUCCESSOR_PAIRS_AFFECT_TEMPLATE_ORDER", "1")
    os.environ.setdefault("ISO_MASTER_REFERENCE", "1")
    os.environ.setdefault("ISO_MASTER_FIXED_TEMPLATE", "0")
    os.environ.setdefault("ISO_MASTER_CONSTRAINT_SCOPE", "master")
    os.environ.setdefault("FOLLOWER_HARDPAIRS_AFFECT_MASTER", "0")
    os.environ.setdefault("ISO_MASTER_OVERRIDES", "")
    os.environ.setdefault("MOVE_ANCHOR_WEIGHT", "0.00001")
    os.environ.setdefault("TRIAL_FAIL_CACHE", "1")
    os.environ.setdefault("TRIAL_FAIL_CACHE_LIMIT", "200000")
    os.environ.setdefault("STRICT_PAIR_HPWL_GATE", "1")
    os.environ.setdefault("PAIR_HPWL_GATE_MARGIN", "0.0")
    os.environ.setdefault("HPWL_GATE_DEBUG", "0")
    os.environ.setdefault("LOCAL_FIXEDPOINT_ITERS", "1")
    os.environ.setdefault("SOLVER_DEBUG_LOG", "0")
    os.environ.setdefault("SOLVER_PROGRESS_LOG", "1")
    os.environ.setdefault("WRITE_FLAT_OUTPUT", "0")
    os.environ.setdefault("WRITE_REPORT_OUTPUT", "0")

    # Load modules explicitly from this stage's directory to avoid conflicts
    # with PlaceDB from stage2_nlplace that may already be in sys.modules
    _placedb_mod = _load_module_from_stage("PlaceDB", "PlaceDB.py")
    PlaceDB = _placedb_mod.PlaceDB

    # pin_legalizer needs PlaceDB in sys.modules to import correctly
    if _STAGE_DIR not in sys.path:
        sys.path.insert(0, _STAGE_DIR)
    # Temporarily replace PlaceDB in sys.modules
    _old_placedb = sys.modules.pop("PlaceDB", None)
    sys.modules["PlaceDB"] = _placedb_mod
    try:
        from pin_legalizer import solve_global_qp_with_outer_order_update, extract_real_segments
    finally:
        if _old_placedb is not None:
            sys.modules["PlaceDB"] = _old_placedb
        else:
            sys.modules.pop("PlaceDB", None)

    block_json = str(Path(block_json).resolve())
    pingroup_json = str(Path(pingroup_json).resolve())
    result_json = str(Path(result_json).resolve())
    output_dir = str(Path(output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)

    if not Path(block_json).exists():
        raise FileNotFoundError(f"Block file not found: {block_json}")
    if not Path(pingroup_json).exists():
        raise FileNotFoundError(f"Pingroup file not found: {pingroup_json}")
    if not Path(result_json).exists():
        raise FileNotFoundError(f"Result file not found: {result_json}")

    logger.info("=" * 70)
    logger.info("Stage 3: QP Pin Legalization")
    logger.info("=" * 70)
    logger.info(f"  block_json: {block_json}")
    logger.info(f"  pingroup_json: {pingroup_json}")
    logger.info(f"  result_json: {result_json}")
    logger.info(f"  keepout: {keepout}")
    logger.info(f"  max_outer_iter: {max_outer_iter}")

    # Load PlaceDB
    db = PlaceDB(block_json, pingroup_json)
    if not db.nets_list:
        raise RuntimeError("No nets found in pingroup.json!")

    # Load initial pin positions from result.json
    with open(result_json, "r", encoding="utf-8") as f:
        result_data = json.load(f)

    result_lookup = {}
    for net_idx, net in enumerate(result_data):
        for pin_idx, pin in enumerate(net):
            key = (pin.get("parent_inst", ""), pin.get("pingroup_name", ""))
            scope = pin.get("scope", None)
            if not isinstance(scope, list) or len(scope) != 2:
                continue
            result_lookup[key] = {
                "x": float(scope[0]),
                "y": float(scope[1]),
                "net_idx": net_idx,
                "pin_idx": pin_idx,
                "raw": pin,
            }

    # Build active pins from result
    mod_segs_cache = {}
    for m in db.all_modules_list:
        if getattr(m, "vertex", None):
            mod_segs_cache[m.name] = extract_real_segments(m)

    active_pins = []
    for net in db.nets_list:
        for pin in net.pins:
            key = (pin.parent_inst, pin.pingroup_name)
            rp = result_lookup.get(key)
            if rp is None:
                continue
            segs = mod_segs_cache.get(pin.parent_inst, [])
            if not segs:
                continue
            # Project pin to nearest segment
            best = None
            safe_margin = max(0.0, float(pin.width) / 2.0)
            for seg in segs:
                if seg.free_axis == "x":
                    proj_s = min(max(float(rp["x"]), float(seg.lo)), float(seg.hi))
                    proj_x, proj_y = proj_s, float(seg.fixed_coord)
                else:
                    proj_s = min(max(float(rp["y"]), float(seg.lo)), float(seg.hi))
                    proj_x, proj_y = float(seg.fixed_coord), proj_s
                dist2 = (proj_x - rp["x"]) ** 2 + (proj_y - rp["y"]) ** 2
                lo, hi = float(seg.lo), float(seg.hi)
                if hi < lo:
                    lo, hi = hi, lo
                if hi - lo < 2.0 * safe_margin:
                    s_init = 0.5 * (lo + hi)
                else:
                    s_init = min(max(proj_s, lo + safe_margin), hi - safe_margin)
                cand = (dist2, seg.id, seg, s_init)
                if best is None or cand[0] < best[0]:
                    best = cand

            if best is not None:
                _, _, best_seg, best_s_init = best
                pin.seg_id = best_seg.id
                pin.s_init = float(best_s_init)
                pin.x = float(rp["x"])
                pin.y = float(rp["y"])
                active_pins.append(pin)

    logger.info(f"Active pins assigned: {len(active_pins)} / {db.total_pin_count}")

    if not active_pins:
        raise RuntimeError("No active pins were initialized from result.json")

    # Run QP solver
    logger.info("Running QP Pin Legalizer...")
    final_state, all_real_segments = solve_global_qp_with_outer_order_update(
        all_modules=db.all_modules_list,
        active_pins=active_pins,
        active_nets=db.nets_list,
        keepout=keepout,
        hpwl_thresh=hpwl_thresh,
        max_outer_iter=max_outer_iter,
        tol=tol,
        enable_hard_iso=enable_hard_iso,
    )

    logger.info(f"Solver complete. Solved {len(final_state)} pins.")

    # Build flat results
    flat_results = []
    for key, st in final_state.items():
        inst, pname = key
        segs = all_real_segments.get(inst, [])
        seg = next((s for s in segs if s.id == st.seg_id), None)
        if seg is None:
            continue
        if seg.free_axis == "x":
            x, y = float(st.s_center), float(seg.fixed_coord)
        else:
            x, y = float(seg.fixed_coord), float(st.s_center)
        flat_results.append({
            "parent_inst": inst,
            "pingroup_name": pname,
            "seg_id": st.seg_id,
            "width": float(st.width),
            "scope": [x, y],
            "free_axis": seg.free_axis,
        })

    flat_lookup = {(p["parent_inst"], p["pingroup_name"]): p for p in flat_results}

    # Rewrite result.json structure with updated coordinates
    result_out = []
    updated_count = 0
    for net in result_data:
        out_net = []
        for pin in net:
            key = (pin.get("parent_inst", ""), pin.get("pingroup_name", ""))
            new_pin = dict(pin)
            fp = flat_lookup.get(key)
            if fp is not None:
                new_pin["scope"] = [float(fp["scope"][0]), float(fp["scope"][1])]
                updated_count += 1
            out_net.append(new_pin)
        result_out.append(out_net)

    output_path = os.path.join(output_dir, "result_legalized.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_out, f, indent=2)

    logger.info(f"Updated {updated_count} pin coordinates.")
    logger.info(f"Legalized result written to: {output_path}")

    return output_path
