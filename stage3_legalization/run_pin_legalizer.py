"""Run script for the QP pin legalizer.

Inputs:
  - block.json
  - pingroup.json
  - result.json

Outputs:
  - result_legalized.json
  - pin_positions_legalized_flat.json, only when WRITE_FLAT_OUTPUT=1
  - qp_json_legalization_report.txt, only when WRITE_REPORT_OUTPUT=1

Recommended stable defaults are set below via os.environ.setdefault(...).
Override them from the shell before running when needed.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path


def _make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _timestamped_name(stem: str, suffix: str, timestamp: str) -> str:
    clean_ts = str(timestamp).strip() or _make_timestamp()
    return f"{stem}_{clean_ts}{suffix}"


def _output_path(path_str: str, default_stem: str, default_suffix: str, timestamp: str = "", timestamp_outputs: bool = True) -> str:
    """Return an absolute output path, optionally adding a timestamp suffix."""
    p = Path(path_str) if path_str else Path(default_stem + default_suffix)
    directory = p.parent if str(p.parent) not in {"", "."} else Path.cwd()
    if timestamp_outputs:
        return str((directory / _timestamped_name(default_stem, default_suffix, timestamp)).resolve())
    return str((directory / f"{default_stem}{default_suffix}").resolve())

# Canonical graph legality-only defaults.
# Keep this block synchronized with __init__.py.
os.environ.setdefault("CVXPY_SOLVER", "CLARABEL")
os.environ.setdefault("ACCEPT_INACCURATE_SOLVE", "0")
os.environ.setdefault("CLARABEL_MAX_ITER", "10000")
os.environ.setdefault("SOLVER_DEBUG_LOG", "0")
os.environ.setdefault("SOLVER_PROGRESS_LOG", "1")
os.environ.setdefault("WRITE_FLAT_OUTPUT", "1")
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
os.environ.setdefault("CANONICAL_STRICT_CYCLE_FALLBACK", "consensus")

# Keep homology canonicalization/audit enabled.
os.environ.setdefault("CANONICAL_ENFORCE_HOMOLOGY_SEGMENT", "1")
os.environ.setdefault("CANONICAL_HOMOLOGY_SEGMENT_POLICY", "canon_inst")
os.environ["CANONICAL_HOMOLOGY_AUDIT_ENABLE"] = "0"
os.environ.setdefault("CANONICAL_HOMOLOGY_AUDIT_PREFIX", "canonical_homology")
os.environ.setdefault("CANONICAL_HOMOLOGY_AUDIT_FAIL", "0")

# Final real fallback paths remain absent from the clean canonical route.
os.environ["CANONICAL_STRICT_GLOBAL_QP"] = "0"
os.environ["CANONICAL_FINAL_REAL_LEGALIZE"] = "0"
os.environ["CANONICAL_FINAL_REAL_SNAP"] = "0"

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

from PlaceDB import PlaceDB
import pin_legalizer as _solver_mod

solve_global_qp_with_outer_order_update = _solver_mod.solve_global_qp_with_outer_order_update
extract_real_segments = _solver_mod.extract_real_segments





REPORT_PATH = Path(__file__).with_name("pin_legalizer_report.txt")
DEFAULT_BLOCK = "block.json"
DEFAULT_PINGROUP = "pingroup.json"
DEFAULT_RESULT = "result.json"
DEFAULT_OUTPUT = "result_legalized.json"
DEFAULT_FLAT_OUTPUT = "pin_positions_legalized_flat.json"
DEFAULT_KEEP_OUT = 0.0
DEFAULT_MAX_OUTER_ITER = 30
DEFAULT_TOL = 1e-3
DEFAULT_ENABLE_HARD_ISO = True




# ---------------- Stage2 segment assignment support ----------------

def _set_diag_paths_for_output(output_json_path: str, timestamp: str = "", timestamp_outputs: bool = True):
    """Force diagnostic CSV/JSON files next to result_legalized output."""
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


def _discover_segment_assignments_path(result_json_path: str, explicit_path: str = None):
    explicit = str(explicit_path or "").strip() or os.environ.get("SEGMENT_ASSIGNMENTS_JSON", "").strip()
    if explicit:
        cand = Path(explicit).expanduser().resolve()
        return str(cand) if cand.exists() else None
    result_dir = Path(result_json_path).resolve().parent
    candidates = list(result_dir.glob("segment_assignments*.json"))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x.stat().st_mtime, x.name), reverse=True)
    return str(candidates[0])


def _load_segment_assignment_index(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    idx = {}
    duplicates = []
    entries = data.get("segment_assignments", {}) or {}
    seg_inst_count = 0
    assigned_count = 0
    for sid, sent in entries.items():
        for _bid, inst_rec in (sent.get("segment_insts", {}) or {}).items():
            seg_inst_count += 1
            block_name = inst_rec.get("block_name", "")
            coords = inst_rec.get("coordinates", None)
            for ap in (inst_rec.get("assigned_pins", []) or []):
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
    return idx, {
        "path": path,
        "total_segments": data.get("total_segments", len(entries)),
        "segment_entries": len(entries),
        "segment_instances": seg_inst_count,
        "assigned_pins": assigned_count,
        "unique_assigned_pins": len(idx),
        "duplicate_pin_names": len(duplicates),
    }


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
            containing.append((shi - slo, repr(seg.id), seg))
        else:
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


def _set_pin_on_segment(pin, seg, x: float, y: float):
    proj_s, _proj_x, _proj_y, _dist2 = _project_point_to_segment(x, y, seg)
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

def _load_result_lookup(result_json_path: str):
    with open(result_json_path, "r", encoding="utf-8") as f:
        result_data = json.load(f)

    lookup = {}
    for net_idx, net in enumerate(result_data):
        for pin_idx, pin in enumerate(net):
            key = (pin.get("parent_inst", ""), pin.get("pingroup_name", ""))
            scope = pin.get("scope", None)
            if not isinstance(scope, list) or len(scope) != 2:
                continue
            lookup[key] = {
                "x": float(scope[0]),
                "y": float(scope[1]),
                "net_idx": net_idx,
                "pin_idx": pin_idx,
                "raw": pin,
            }
    return result_data, lookup


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


def _assign_pin_to_nearest_segment(pin, segs, x: float, y: float):
    if not segs:
        return False, "no_segments"

    best = None
    safe_margin = max(0.0, float(pin.width) / 2.0)
    for seg in segs:
        proj_s, proj_x, proj_y, dist2 = _project_point_to_segment(x, y, seg)
        lo = float(seg.lo)
        hi = float(seg.hi)
        if hi < lo:
            lo, hi = hi, lo
        if hi - lo < 2.0 * safe_margin:
            s_init = 0.5 * (lo + hi)
        else:
            s_init = min(max(proj_s, lo + safe_margin), hi - safe_margin)
        cand = (dist2, 0 if proj_s == s_init else 1, abs(proj_s - s_init), seg.id, seg, s_init)
        if best is None or cand < best:
            best = cand

    _, _, _, _, best_seg, best_s_init = best
    pin.seg_id = best_seg.id
    pin.s_init = float(best_s_init)
    pin.x = float(x)
    pin.y = float(y)
    return True, None


def _build_active_pins_from_result(db: PlaceDB, result_lookup: dict, assignment_index=None):
    assignment_index = assignment_index or {}
    active_pins = []
    missing_coords = []
    missing_segments = []
    assigned_from_file = 0
    assigned_nearest = 0
    assignment_fallback_reasons = {}
    mod_segs_cache = {}
    for m in db.all_modules_list:
        if getattr(m, "vertex", None):
            mod_segs_cache[m.name] = extract_real_segments(m)

    for net in db.nets_list:
        for pin in net.pins:
            key = (pin.parent_inst, pin.pingroup_name)
            rp = result_lookup.get(key)
            if rp is None:
                missing_coords.append(key)
                continue
            segs = mod_segs_cache.get(pin.parent_inst, [])
            if assignment_index:
                ok, reason = _assign_pin_from_segment_assignment(pin, segs, rp["x"], rp["y"], assignment_index)
                if ok:
                    active_pins.append(pin)
                    assigned_from_file += 1
                    continue
                assignment_fallback_reasons[reason] = assignment_fallback_reasons.get(reason, 0) + 1
            ok, reason = _assign_pin_to_nearest_segment(pin, segs, rp["x"], rp["y"])
            if ok:
                active_pins.append(pin)
                assigned_nearest += 1
            else:
                missing_segments.append((key, reason))

    return active_pins, mod_segs_cache, missing_coords, missing_segments, {
        "assigned_from_file": assigned_from_file,
        "assigned_nearest": assigned_nearest,
        "assignment_fallback_reasons": assignment_fallback_reasons,
    }


def _build_flat_results(final_state, all_real_segments):
    results = []
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
        results.append({
            "parent_inst": inst,
            "pingroup_name": pname,
            "seg_id": st.seg_id,
            "width": float(st.width),
            "scope": [x, y],
            "free_axis": seg.free_axis,
        })
    return results


def _rewrite_result_json_like_input(result_data, flat_lookup):
    out = []
    updated = 0
    missing = 0
    for net in result_data:
        out_net = []
        for pin in net:
            key = (pin.get("parent_inst", ""), pin.get("pingroup_name", ""))
            new_pin = dict(pin)
            fp = flat_lookup.get(key)
            if fp is not None:
                new_pin["scope"] = [float(fp["scope"][0]), float(fp["scope"][1])]
                updated += 1
            else:
                missing += 1
            out_net.append(new_pin)
        out.append(out_net)
    return out, updated, missing




def _no_overlap_report(flat_results, keepout: float, tol: float):
    by_seg = {}
    for p in flat_results:
        seg_id_raw = p.get("seg_id", [])
        if not isinstance(seg_id_raw, list) or len(seg_id_raw) != 2:
            continue
        seg_id = tuple(seg_id_raw)
        free_axis = p.get("free_axis")
        scope = p.get("scope", [None, None])
        if free_axis not in {"x", "y"} or not isinstance(scope, list) or len(scope) != 2:
            continue
        s = float(scope[0]) if free_axis == "x" else float(scope[1])
        by_seg.setdefault(seg_id, []).append((s, float(p.get("width", 0.0)), (p.get("parent_inst", ""), p.get("pingroup_name", ""))))
    violations = []
    for seg_id, arr in by_seg.items():
        arr.sort(key=lambda t: t[0])
        for (s0, w0, k0), (s1, w1, k1) in zip(arr, arr[1:]):
            req = 0.5 * (w0 + w1) + float(keepout)
            gap = s1 - s0
            if gap + tol < req:
                violations.append((req - gap, seg_id, k0, k1, gap, req))
    violations.sort(reverse=True, key=lambda x: x[0])
    return violations


def _displacement_report(flat_lookup, result_lookup):
    vals = []
    for key, fp in flat_lookup.items():
        rp = result_lookup.get(key)
        if rp is None:
            continue
        x0, y0 = float(rp["x"]), float(rp["y"])
        x1, y1 = map(float, fp["scope"])
        vals.append((abs(x1-x0)+abs(y1-y0), max(abs(x1-x0), abs(y1-y0)), key))
    if not vals:
        return 0.0, 0.0, []
    vals.sort(reverse=True, key=lambda x: x[0])
    mean_manhattan = sum(v[0] for v in vals) / len(vals)
    return mean_manhattan, vals[0][0], vals[:10]

def main():
    block_json = os.environ.get("BLOCK_JSON", DEFAULT_BLOCK)
    pingroup_json = os.environ.get("PINGROUP_JSON", DEFAULT_PINGROUP)
    result_json = os.environ.get("RESULT_JSON", DEFAULT_RESULT)
    timestamp_outputs = os.environ.get("STAGE3_TIMESTAMP_OUTPUTS", "1").strip().lower() not in {"0", "false", "no", "off"}
    timestamp_env = str(os.environ.get("STAGE3_OUTPUT_TIMESTAMP", "")).strip()
    timestamp = (timestamp_env or _make_timestamp()) if timestamp_outputs else timestamp_env
    output_json_env = os.environ.get("OUTPUT_JSON", DEFAULT_OUTPUT)
    flat_output_json_env = os.environ.get("FLAT_OUTPUT_JSON", DEFAULT_FLAT_OUTPUT)
    output_json = _output_path(output_json_env, "result_legalized", ".json", timestamp, timestamp_outputs=timestamp_outputs)
    flat_output_json = _output_path(flat_output_json_env, "pin_positions_legalized_flat", ".json", timestamp, timestamp_outputs=timestamp_outputs) if flat_output_json_env else flat_output_json_env
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    _set_diag_paths_for_output(output_json, timestamp, timestamp_outputs=timestamp_outputs)
    keepout = float(os.environ.get("KEEP_OUT", str(DEFAULT_KEEP_OUT)))
    hpwl_thresh = 500.0  # kept only for solver API compatibility; clean route ignores it
    max_outer_iter = int(os.environ.get("MAX_OUTER_ITER", str(DEFAULT_MAX_OUTER_ITER)))
    tol = float(os.environ.get("TOL", str(DEFAULT_TOL)))
    enable_hard_iso = os.environ.get("ENABLE_HARD_ISO", "1" if DEFAULT_ENABLE_HARD_ISO else "0").strip() not in {"0", "false", "False", "no", "NO"}
    write_flat_output = os.environ.get("WRITE_FLAT_OUTPUT", "1").strip() not in {"0", "false", "False", "no", "NO"}
    write_report_output = os.environ.get("WRITE_REPORT_OUTPUT", "0").strip() not in {"0", "false", "False", "no", "NO"}

    db = PlaceDB(block_json, pingroup_json)
    if not db.nets_list:
        raise RuntimeError("pingroup.json 中没有找到 net！")

    result_data, result_lookup = _load_result_lookup(result_json)

    assignment_index = {}
    assignment_path = _discover_segment_assignments_path(result_json)
    assignment_stats = None
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

    report_lines = []
    report_lines.append(f"=== QP Pin Legalization Report @ {datetime.now().isoformat(timespec='seconds')} ===")
    report_lines.append(f"Workspace: {Path.cwd()}")
    report_lines.append(f"block_json: {block_json}")
    report_lines.append(f"pingroup_json: {pingroup_json}")
    report_lines.append(f"result_json: {result_json}")
    report_lines.append(f"output_timestamp: {timestamp if timestamp_outputs else 'disabled'}")
    report_lines.append(f"output_json: {output_json}")
    report_lines.append(f"diag_csv: {os.environ.get('CANONICAL_CLUSTER_DIAG_CSV')}")
    report_lines.append(f"diag_json: {os.environ.get('CANONICAL_CLUSTER_DIAG_JSON')}")
    report_lines.append(f"order_fallback_json: {os.environ.get('CANONICAL_ORDER_FALLBACK_JSON')}")
    report_lines.append(f"segment_assignments_json: {assignment_path}")
    report_lines.append(f"keepout: {keepout}")
    report_lines.append(f"max_outer_iter: {max_outer_iter}")
    report_lines.append(f"enable_hard_iso: {enable_hard_iso}")
    report_lines.append(f"write_flat_output: {write_flat_output}")
    report_lines.append(f"nets: {len(db.nets_list)}")
    report_lines.append(f"pins_in_result_lookup: {len(result_lookup)}")

    active_pins, mod_segs_cache, missing_coords, missing_segments, assign_stats = _build_active_pins_from_result(db, result_lookup, assignment_index=assignment_index)
    report_lines.append(f"active_pins_assigned: {len(active_pins)} / {db.total_pin_count}")
    report_lines.append(f"active_pins_from_segment_assignments: {assign_stats.get('assigned_from_file')}")
    report_lines.append(f"active_pins_from_nearest_fallback: {assign_stats.get('assigned_nearest')}")
    report_lines.append(f"assignment_fallback_reasons: {assign_stats.get('assignment_fallback_reasons')}")
    report_lines.append(f"missing_coords: {len(missing_coords)}")
    report_lines.append(f"missing_segments: {len(missing_segments)}")

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

    report_lines.append(f"solved_pins: {len(final_state)}")

    flat_results = _build_flat_results(final_state, all_real_segments)
    flat_lookup = {(p["parent_inst"], p["pingroup_name"]): p for p in flat_results}
    result_out, updated_count, unchanged_count = _rewrite_result_json_like_input(result_data, flat_lookup)
    report_lines.append(f"updated_result_entries: {updated_count}")
    report_lines.append(f"unchanged_result_entries: {unchanged_count}")

    noov = _no_overlap_report(flat_results, keepout=keepout, tol=tol)
    disp_mean, disp_max, disp_top = _displacement_report(flat_lookup, result_lookup)

    # Optional report, written only when WRITE_REPORT_OUTPUT=1.
    report_lines.append(f"no_overlap_violations: {len(noov)}")
    for item in noov[:10]:
        report_lines.append(f"  no_overlap_violation: excess={item[0]:.6f} seg={item[1]} a={item[2]} b={item[3]} gap={item[4]:.6f} req={item[5]:.6f}")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result_out, f, indent=2)
    report_lines.append(f"output_json_written: {output_json}")

    if write_flat_output:
        with open(flat_output_json, "w", encoding="utf-8") as f:
            json.dump(flat_results, f, indent=2)
        report_lines.append(f"flat_output_json_written: {flat_output_json}")
    else:
        report_lines.append("flat_output_json_written: disabled_by_WRITE_FLAT_OUTPUT")

    if write_report_output:
        report_name = _timestamped_name("pin_legalizer_report", ".txt", timestamp) if timestamp_outputs else "pin_legalizer_report.txt"
        Path(output_json).with_name(report_name).write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("合法化已完成。")


if __name__ == "__main__":
    main()
