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


def _make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _timestamped_name(stem: str, suffix: str, timestamp: str) -> str:
    clean_ts = str(timestamp).strip() or _make_timestamp()
    return f"{stem}_{clean_ts}{suffix}"


def _load_module_from_stage(module_name, filename):
    """Load a module explicitly from this stage's directory to avoid conflicts."""
    spec = importlib.util.spec_from_file_location(
        f"stage3_{module_name}", os.path.join(_STAGE_DIR, filename)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod






# ---------------- Stage2 segment assignment support ----------------

def _set_diag_paths_for_output(output_json_path: str, timestamp: str = "", timestamp_outputs: bool = True):
    """Force diagnostic CSV/JSON files next to result_legalized output.

    If timestamp_outputs=True, filenames are suffixed with the shared timestamp.
    If timestamp_outputs=False, stable non-timestamped names are used.
    """
    out_path = Path(output_json_path).resolve()
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if timestamp_outputs:
        ts = str(timestamp).strip() or _make_timestamp()
        os.environ["CANONICAL_CLUSTER_DIAG_CSV"] = str(out_dir / _timestamped_name("canonical_cluster_slack_diagnostics", ".csv", ts))
        os.environ["CANONICAL_CLUSTER_DIAG_JSON"] = str(out_dir / _timestamped_name("canonical_cluster_slack_diagnostics", ".json", ts))
        os.environ["CANONICAL_ORDER_FALLBACK_JSON"] = str(out_dir / _timestamped_name("canonical_order_fallback_report", ".json", ts))
    else:
        os.environ["CANONICAL_CLUSTER_DIAG_CSV"] = str(out_dir / "canonical_cluster_slack_diagnostics.csv")
        os.environ["CANONICAL_CLUSTER_DIAG_JSON"] = str(out_dir / "canonical_cluster_slack_diagnostics.json")
        os.environ["CANONICAL_ORDER_FALLBACK_JSON"] = str(out_dir / "canonical_order_fallback_report.json")


def _default_result_legalized_path(output_dir: str, timestamp: str = "", timestamp_outputs: bool = True) -> str:
    out_dir = Path(output_dir).resolve()
    if timestamp_outputs:
        return str(out_dir / _timestamped_name("result_legalized", ".json", timestamp))
    return str(out_dir / "result_legalized.json")


def _discover_segment_assignments_path(result_json_path: str, explicit_path: str = None):
    """Find segment_assignments*.json in the same directory as result.json.

    The file can also be specified explicitly with the function argument or
    SEGMENT_ASSIGNMENTS_JSON. Function argument has highest priority.
    """
    explicit = str(explicit_path or "").strip() or os.environ.get("SEGMENT_ASSIGNMENTS_JSON", "").strip()
    if explicit:
        cand = Path(explicit).expanduser().resolve()
        return str(cand) if cand.exists() else None

    result_dir = Path(result_json_path).resolve().parent
    candidates = list(result_dir.glob("segment_assignments*.json"))
    if not candidates:
        return None
    # Prefer latest generated file; timestamped names also sort correctly in most cases.
    candidates.sort(key=lambda x: (x.stat().st_mtime, x.name), reverse=True)
    return str(candidates[0])


def _load_segment_assignment_index(path: str):
    """Build full-pin-name -> assigned small-segment record index.

    Important: the segment_assignments file subdivides one physical edge into many
    small segments. Stage3 uses larger polygon edges. Therefore we do NOT reuse
    segment_id from this file as Stage3 seg_id; we only use block_name +
    coordinates to find the containing Stage3 big edge.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    idx = {}
    duplicates = []
    entries = data.get("segment_assignments", {}) or {}
    seg_inst_count = 0
    assigned_count = 0
    for sid, sent in entries.items():
        sinsts = sent.get("segment_insts", {}) or {}
        for _bid, inst_rec in sinsts.items():
            seg_inst_count += 1
            block_name = inst_rec.get("block_name", "")
            coords = inst_rec.get("coordinates", None)
            pins = inst_rec.get("assigned_pins", []) or []
            for ap in pins:
                name = ap.get("name", "")
                if not name:
                    continue
                assigned_count += 1
                rec = {
                    "pin_name": name,
                    "block_name": block_name,
                    "coordinates": coords,
                    "small_segment_id": inst_rec.get("segment_id", sid),
                    "segment_inst_id": inst_rec.get("segment_inst_id"),
                    "edge_id": inst_rec.get("edge_id"),
                    "width": ap.get("width"),
                    "net_id": ap.get("net_id"),
                    "isomorphic_group_id": ap.get("isomorphic_group_id"),
                }
                if name in idx:
                    duplicates.append(name)
                    continue
                idx[name] = rec

    stats = {
        "path": path,
        "total_segments": data.get("total_segments", len(entries)),
        "segment_entries": len(entries),
        "segment_instances": seg_inst_count,
        "assigned_pins": assigned_count,
        "unique_assigned_pins": len(idx),
        "duplicate_pin_names": len(duplicates),
    }
    return idx, stats


def _coords_to_assignment_axis(coords, tol=1e-6):
    if not isinstance(coords, (list, tuple)) or len(coords) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in coords]
    if abs(y1 - y2) <= tol:
        return ("x", float(y1), min(x1, x2), max(x1, x2))
    if abs(x1 - x2) <= tol:
        return ("y", float(x1), min(y1, y2), max(y1, y2))
    return None


def _match_assignment_to_big_segment(coords, segs, tol=1e-5):
    """Match a Stage2 small segment to the containing Stage3 big edge."""
    axis = _coords_to_assignment_axis(coords, tol=tol)
    if axis is None:
        return None, "bad_assignment_coordinates"
    free_axis, fixed_coord, lo, hi = axis
    mid = 0.5 * (lo + hi)

    containing = []
    near = []
    for seg in segs or []:
        if getattr(seg, "free_axis", None) != free_axis:
            continue
        slo, shi = sorted((float(seg.lo), float(seg.hi)))
        fixed_delta = abs(float(seg.fixed_coord) - fixed_coord)
        if fixed_delta <= tol and lo >= slo - tol and hi <= shi + tol:
            # Prefer the smallest containing Stage3 segment if there are multiple.
            containing.append((shi - slo, repr(seg.id), seg))
        else:
            # Fallback score for near-collinear partial-overlap cases.
            overlap = max(0.0, min(hi, shi) - max(lo, slo))
            mid_dist = 0.0 if slo <= mid <= shi else min(abs(mid - slo), abs(mid - shi))
            if fixed_delta <= max(1.0, tol) and (overlap > 0.0 or mid_dist <= max(1.0, tol)):
                near.append((fixed_delta, -overlap, mid_dist, repr(seg.id), seg))

    if containing:
        containing.sort(key=lambda t: (t[0], t[1]))
        return containing[0][2], None
    if near:
        near.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
        return near[0][4], "near_collinear_fallback"
    return None, "no_containing_big_segment"


def _project_point_to_segment(x: float, y: float, seg):
    if seg.free_axis == "x":
        proj_s = min(max(float(x), float(seg.lo)), float(seg.hi))
        proj_x = proj_s
        proj_y = float(seg.fixed_coord)
    else:
        proj_s = min(max(float(y), float(seg.lo)), float(seg.hi))
        proj_x = float(seg.fixed_coord)
        proj_y = proj_s
    dist2 = (proj_x - float(x)) ** 2 + (proj_y - float(y)) ** 2
    return proj_s, proj_x, proj_y, dist2


def _set_pin_on_segment(pin, seg, x: float, y: float):
    proj_s, _px, _py, _dist2 = _project_point_to_segment(x, y, seg)
    safe_margin = max(0.0, float(pin.width) / 2.0)
    lo = float(seg.lo)
    hi = float(seg.hi)
    if hi < lo:
        lo, hi = hi, lo
    if hi - lo < 2.0 * safe_margin:
        s_init = 0.5 * (lo + hi)
    else:
        s_init = min(max(proj_s, lo + safe_margin), hi - safe_margin)
    pin.seg_id = seg.id
    pin.s_init = float(s_init)
    pin.x = float(x)
    pin.y = float(y)
    return True


def _assign_pin_from_segment_assignment(pin, segs, x: float, y: float, assignment_index: dict):
    if not assignment_index:
        return False, "no_assignment_index"
    full_name = f"{pin.parent_inst}.{pin.pingroup_name}"
    rec = assignment_index.get(full_name)
    if rec is None:
        return False, "pin_not_in_segment_assignments"
    if rec.get("block_name") and rec.get("block_name") != pin.parent_inst:
        return False, "assignment_block_mismatch"
    seg, reason = _match_assignment_to_big_segment(rec.get("coordinates"), segs)
    if seg is None:
        return False, reason or "assignment_segment_match_failed"
    _set_pin_on_segment(pin, seg, x, y)
    return True, reason


def run_legalization(
    block_json: str,
    pingroup_json: str,
    result_json: str,
    output_dir: str,
    keepout: float = 0.0,
    hpwl_thresh: float = 500.0,  # kept for API compatibility; unused in clean no-align route
    max_outer_iter: int = 30,
    tol: float = 1e-3,
    enable_hard_iso: bool = True,
    output_json: str = None,
    segment_assignments_json: str = None,
    output_timestamp: str = None,
    timestamp_outputs: bool = True,
) -> str:
    """Run QP pin legalization to eliminate overlaps.

    Args:
        block_json: Path to block.json.
        pingroup_json: Path to pingroup.json.
        result_json: Path to result.json from Stage 2.
        output_dir: Directory where result_legalized.json will be written.
        keepout: Minimum gap between adjacent pins.
        hpwl_thresh: Kept for API compatibility; unused in clean no-align route.
        max_outer_iter: Maximum outer QP iterations.
        tol: Convergence tolerance.
        enable_hard_iso: Enable hard isomorphic constraints.
        output_json: Optional explicit result_legalized output path.
        segment_assignments_json: Optional explicit Stage2 segment_assignments*.json path.
        output_timestamp: Optional timestamp token shared by all Stage3 output files.
        timestamp_outputs: If False, disable timestamp suffixes for Stage3 output/diagnostic filenames.

    Returns:
        Path to the generated result_legalized.json file.
    """
    # Canonical graph legality-only defaults.
    # Keep this block synchronized with run_pin_legalizer.py.
    os.environ.setdefault("CVXPY_SOLVER", "CLARABEL")
    os.environ.setdefault("ACCEPT_INACCURATE_SOLVE", "0")
    os.environ.setdefault("CLARABEL_MAX_ITER", "10000")
    os.environ.setdefault("SOLVER_DEBUG_LOG", "0")
    os.environ.setdefault("SOLVER_PROGRESS_LOG", "1")
    os.environ.setdefault("WRITE_FLAT_OUTPUT", "0")
    os.environ.setdefault("WRITE_REPORT_OUTPUT", "0")

    # Current production route: canonical graph only.
    os.environ["CANONICAL_GRAPH_LEGALIZE"] = "1"
    os.environ.setdefault("CANONICAL_CLUSTER_MAX_VARS", "300")
    os.environ.setdefault("CANONICAL_CLUSTER_SOLVE_MAX_VARS", "300")
    os.environ["CANONICAL_CLUSTER_OBJECTIVE"] = "move"
    os.environ["CANONICAL_PRINT_TOTAL_HPWL"] = "0"
    os.environ.setdefault("CANONICAL_MOVE_WEIGHT", "1.0")
    os.environ.setdefault("CANONICAL_NO_OVERLAP_EPS", "1e-3")
    os.environ.setdefault("CANONICAL_STRICT_SEGMENT_ORDER", "pure_template")

    # Homology canonicalization.
    os.environ.setdefault("CANONICAL_ENFORCE_HOMOLOGY_SEGMENT", "1")
    os.environ.setdefault("CANONICAL_HOMOLOGY_SEGMENT_POLICY", "canon_inst")
    os.environ["CANONICAL_HOMOLOGY_AUDIT_ENABLE"] = "0"
    os.environ.setdefault("CANONICAL_HOMOLOGY_AUDIT_PREFIX", "canonical_homology")
    os.environ.setdefault("CANONICAL_HOMOLOGY_AUDIT_FAIL", "0")

    # Infeasible cluster diagnostics and local order fallback.
    os.environ["CANONICAL_CLUSTER_DIAG_ENABLE"] = "1"
    os.environ.setdefault("CANONICAL_CLUSTER_DIAG_MAX_CLUSTERS", "20")
    os.environ.setdefault("CANONICAL_CLUSTER_DIAG_MAX_VARS", "0")
    os.environ.setdefault("CANONICAL_CLUSTER_DIAG_TOPK", "50")
    os.environ.setdefault("CANONICAL_CLUSTER_DIAG_CSV", "canonical_cluster_slack_diagnostics.csv")
    os.environ.setdefault("CANONICAL_CLUSTER_DIAG_JSON", "canonical_cluster_slack_diagnostics.json")
    os.environ["CANONICAL_ORDER_FALLBACK_ENABLE"] = "1"
    os.environ.setdefault("CANONICAL_ORDER_FALLBACK_CAUSES", "order_affine_interval_conflict")
    os.environ.setdefault("CANONICAL_ORDER_FALLBACK_MAX_SEGMENTS", "8")
    os.environ.setdefault("CANONICAL_ORDER_FALLBACK_MAX_TRIALS", "16")
    os.environ.setdefault("CANONICAL_ORDER_FALLBACK_JSON", "canonical_order_fallback_report.json")

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
    timestamp_env = str(output_timestamp or os.environ.get("STAGE3_OUTPUT_TIMESTAMP", "")).strip()
    timestamp = (timestamp_env or _make_timestamp()) if timestamp_outputs else timestamp_env
    if output_json:
        output_path = str(Path(output_json).resolve())
    else:
        output_path = _default_result_legalized_path(output_dir, timestamp, timestamp_outputs=timestamp_outputs)
    _set_diag_paths_for_output(output_path, timestamp, timestamp_outputs=timestamp_outputs)

    if not Path(block_json).exists():
        raise FileNotFoundError(f"找不到 block 文件：{block_json}")
    if not Path(pingroup_json).exists():
        raise FileNotFoundError(f"找不到 pingroup 文件：{pingroup_json}")
    if not Path(result_json).exists():
        raise FileNotFoundError(f"找不到 result 文件：{result_json}")

    print(f"[stage3-output] timestamp={'disabled' if not timestamp_outputs else timestamp} output_json={output_path}")
    print(f"[stage3-output] diag_csv={os.environ.get('CANONICAL_CLUSTER_DIAG_CSV')}")
    print(f"[stage3-output] diag_json={os.environ.get('CANONICAL_CLUSTER_DIAG_JSON')}")
    print(f"[stage3-output] order_fallback_json={os.environ.get('CANONICAL_ORDER_FALLBACK_JSON')}")

    # Load PlaceDB
    db = PlaceDB(block_json, pingroup_json)
    if not db.nets_list:
        raise RuntimeError("pingroup.json 中没有找到 net！")

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

    # Optional Stage2 segment assignment file lives next to result.json by default.
    # It subdivides a large edge into small segments. We use its coordinates only
    # to identify the containing Stage3 big edge; its numeric segment_id is NOT
    # reused as Stage3 seg_id.
    assignment_index = {}
    assignment_path = _discover_segment_assignments_path(result_json, explicit_path=segment_assignments_json)
    if assignment_path:
        assignment_index, assignment_stats = _load_segment_assignment_index(assignment_path)
        print(
            f"[segment-assignment] loaded path={assignment_path} "
            f"assigned_pins={assignment_stats.get('assigned_pins')} "
            f"unique={assignment_stats.get('unique_assigned_pins')} "
            f"duplicates={assignment_stats.get('duplicate_pin_names')}"
        )
    else:
        print(f"[segment-assignment] no segment_assignments*.json found next to result_json={result_json}; fallback to nearest-edge binding")

    # Build active pins from result
    mod_segs_cache = {}
    for m in db.all_modules_list:
        if getattr(m, "vertex", None):
            mod_segs_cache[m.name] = extract_real_segments(m)

    active_pins = []
    assigned_from_file = 0
    assigned_nearest = 0
    assignment_fallback_reasons = {}
    for net in db.nets_list:
        for pin in net.pins:
            key = (pin.parent_inst, pin.pingroup_name)
            rp = result_lookup.get(key)
            if rp is None:
                continue
            segs = mod_segs_cache.get(pin.parent_inst, [])
            if not segs:
                continue
            ok = False
            if assignment_index:
                ok, reason = _assign_pin_from_segment_assignment(pin, segs, rp["x"], rp["y"], assignment_index)
                if ok:
                    assigned_from_file += 1
                    active_pins.append(pin)
                    continue
                assignment_fallback_reasons[reason] = assignment_fallback_reasons.get(reason, 0) + 1

            # Fallback only when no assignment exists/matches. This keeps old behavior
            # for partial or missing assignment files.
            best = None
            safe_margin = max(0.0, float(pin.width) / 2.0)
            for seg in segs:
                proj_s, proj_x, proj_y, dist2 = _project_point_to_segment(rp["x"], rp["y"], seg)
                lo, hi = float(seg.lo), float(seg.hi)
                if hi < lo:
                    lo, hi = hi, lo
                if hi - lo < 2.0 * safe_margin:
                    s_init = 0.5 * (lo + hi)
                else:
                    s_init = min(max(proj_s, lo + safe_margin), hi - safe_margin)
                cand = (dist2, repr(seg.id), seg, s_init)
                if best is None or cand[0] < best[0]:
                    best = cand

            if best is not None:
                _, _, best_seg, best_s_init = best
                pin.seg_id = best_seg.id
                pin.s_init = float(best_s_init)
                pin.x = float(rp["x"])
                pin.y = float(rp["y"])
                assigned_nearest += 1
                active_pins.append(pin)

    print(f"[segment-assignment] active_pins_from_file={assigned_from_file} fallback_nearest={assigned_nearest} fallback_reasons={assignment_fallback_reasons}")

    if not active_pins:
        raise RuntimeError("未能从 result.json 初始化任何有效 pin")

    # Clean route does not use net-derived successor or HPWL-objective data.
    active_nets_for_solver = []
    final_state, all_real_segments = solve_global_qp_with_outer_order_update(
        all_modules=db.all_modules_list,
        active_pins=active_pins,
        active_nets=active_nets_for_solver,
        keepout=keepout,
        hpwl_thresh=hpwl_thresh,
        max_outer_iter=max_outer_iter,
        tol=tol,
        enable_hard_iso=enable_hard_iso,
    )


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

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_out, f, indent=2)

    print("合法化已完成。")
    return output_path
