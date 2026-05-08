"""Run script for the QP pin legalizer.

Inputs:
  - block.json
  - pingroup.json
  - result.json

Outputs:
  - result_legalized.json
  - qp_json_legalization_report.txt
  - pin_positions_legalized_flat.json, unless WRITE_FLAT_OUTPUT=0

Recommended stable defaults are set below via os.environ.setdefault(...).
Override them from the shell before running when needed.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
# Stable pin legalizer run: abandoned RRR / target-first / target-closure / force-diagnostic paths are removed.
# Reference-master hard iso policy: for reused modules A/B/C/D, A is master; B/C/D follow A.
# Master coordinates initialize the shared template, but the template remains optimizable.
os.environ.setdefault("SUCCESSOR_PAIRS_AFFECT_TEMPLATE_ORDER", "1")
os.environ.setdefault("ISO_MASTER_REFERENCE", "1")
os.environ.setdefault("ISO_MASTER_FIXED_TEMPLATE", "0")
os.environ.setdefault("ISO_MASTER_CONSTRAINT_SCOPE", "master")
os.environ.setdefault("FOLLOWER_HARDPAIRS_AFFECT_MASTER", "0")
# Optional explicit overrides, e.g. F=TOP.U_F;K=TOP.U_K. By default the first instance in each iso group is used.
os.environ.setdefault("ISO_MASTER_OVERRIDES", "")
# Soft secondary wire-preservation only. Alignment/no-overlap feasibility has priority.
os.environ.setdefault("MOVE_ANCHOR_WEIGHT", "0.00001")
# Cache failed trial active sets to avoid repeated identical deterministic infeasible GUROBI solves.
os.environ.setdefault("TRIAL_FAIL_CACHE", "1")
os.environ.setdefault("TRIAL_FAIL_CACHE_LIMIT", "200000")
# Pair-level HPWL gate. Set HPWL_GATE_DEBUG=1 only when debugging skipped candidates.
os.environ.setdefault("STRICT_PAIR_HPWL_GATE", "1")
os.environ.setdefault("PAIR_HPWL_GATE_MARGIN", "0.0")
os.environ.setdefault("HPWL_GATE_DEBUG", "0")
os.environ.setdefault("LOCAL_FIXEDPOINT_ITERS", "1")
# Keep coarse [inner] progress on by default; detailed solver/admission logs remain debug-only.
os.environ.setdefault("SOLVER_DEBUG_LOG", "0")
os.environ.setdefault("SOLVER_PROGRESS_LOG", "1")
# Control whether pin_positions_legalized_flat.json is written. Set WRITE_FLAT_OUTPUT=0 to skip it.
os.environ.setdefault("WRITE_FLAT_OUTPUT", "0")
os.environ.setdefault("WRITE_REPORT_OUTPUT", "0")

from PlaceDB import PlaceDB
from pin_legalizer import (
    solve_global_qp_with_outer_order_update,
    extract_real_segments,
)

REPORT_PATH = Path(__file__).with_name("qp_json_legalization_report.txt")
DEFAULT_BLOCK = "block.json"
DEFAULT_PINGROUP = "pingroup.json"
DEFAULT_RESULT = "result.json"
DEFAULT_OUTPUT = "result_legalized.json"
DEFAULT_FLAT_OUTPUT = "pin_positions_legalized_flat.json"
DEFAULT_KEEP_OUT = 0.0
DEFAULT_HPWL_THRESH = 500.0
DEFAULT_MAX_OUTER_ITER = 30
DEFAULT_TOL = 1e-3
DEFAULT_ENABLE_HARD_ISO = True


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


def _build_active_pins_from_result(db: PlaceDB, result_lookup: dict):
    active_pins = []
    missing_coords = []
    missing_segments = []
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
            ok, reason = _assign_pin_to_nearest_segment(pin, segs, rp["x"], rp["y"])
            if ok:
                active_pins.append(pin)
            else:
                missing_segments.append((key, reason))

    return active_pins, mod_segs_cache, missing_coords, missing_segments


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




def _parse_successor_key(s: str):
    if not isinstance(s, str) or "." not in s:
        return None
    inst, pname = s.rsplit(".", 1)
    return (inst, pname)


def _successor_alignment_report(result_out, flat_lookup, hpwl_thresh: float, tol: float):
    total = 0
    low = 0
    aligned = 0
    misaligned = []
    for net in result_out:
        for pin in net:
            akey = (pin.get("parent_inst", ""), pin.get("pingroup_name", ""))
            af = flat_lookup.get(akey)
            if af is None:
                continue
            for succ in pin.get("successors", []) or []:
                bkey = _parse_successor_key(succ)
                if bkey is None:
                    continue
                bf = flat_lookup.get(bkey)
                if bf is None:
                    continue
                total += 1
                ax, ay = map(float, af["scope"])
                bx, by = map(float, bf["scope"])
                hpwl = abs(ax - bx) + abs(ay - by)
                if hpwl < hpwl_thresh:
                    low += 1
                    if af.get("free_axis") == bf.get("free_axis") == "x":
                        delta = abs(ax - bx)
                    elif af.get("free_axis") == bf.get("free_axis") == "y":
                        delta = abs(ay - by)
                    else:
                        delta = hpwl
                    if delta <= tol:
                        aligned += 1
                    else:
                        misaligned.append((delta, hpwl, akey, bkey))
    misaligned.sort(reverse=True, key=lambda x: x[0])
    return total, low, aligned, misaligned


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
    # block_json = os.environ.get("BLOCK_JSON", DEFAULT_BLOCK)
    # pingroup_json = os.environ.get("PINGROUP_JSON", DEFAULT_PINGROUP)
    # result_json = os.environ.get("RESULT_JSON", DEFAULT_RESULT)
    block_json = r"benchmark\case2\block.json"
    pingroup_json = r"benchmark\case2\pingroup.json"
    result_json = r"benchmark\result\case2\result.json"
    output_json = os.environ.get("OUTPUT_JSON", DEFAULT_OUTPUT)
    flat_output_json = os.environ.get("FLAT_OUTPUT_JSON", DEFAULT_FLAT_OUTPUT)
    keepout = float(os.environ.get("KEEP_OUT", str(DEFAULT_KEEP_OUT)))
    hpwl_thresh = float(os.environ.get("HPWL_THRESH", str(DEFAULT_HPWL_THRESH)))
    max_outer_iter = int(os.environ.get("MAX_OUTER_ITER", str(DEFAULT_MAX_OUTER_ITER)))
    tol = float(os.environ.get("TOL", str(DEFAULT_TOL)))
    enable_hard_iso = os.environ.get("ENABLE_HARD_ISO", "1" if DEFAULT_ENABLE_HARD_ISO else "0").strip() not in {"0", "false", "False", "no", "NO"}
    write_flat_output = os.environ.get("WRITE_FLAT_OUTPUT", "0").strip() not in {"0", "false", "False", "no", "NO"}
    write_report_output = os.environ.get("WRITE_REPORT_OUTPUT", "0").strip() not in {"0", "false", "False", "no", "NO"}

    print(f"Loading PlaceDB from {block_json} and {pingroup_json}...")
    db = PlaceDB(block_json, pingroup_json)
    if not db.nets_list:
        print("No nets found in pingroup.json!")
        return

    print(f"Loading initial pin positions from {result_json}...")
    result_data, result_lookup = _load_result_lookup(result_json)

    report_lines = []
    report_lines.append(f"=== QP Pin Legalization Report @ {datetime.now().isoformat(timespec='seconds')} ===")
    report_lines.append(f"Workspace: {Path.cwd()}")
    report_lines.append(f"block_json: {block_json}")
    report_lines.append(f"pingroup_json: {pingroup_json}")
    report_lines.append(f"result_json: {result_json}")
    report_lines.append(f"keepout: {keepout}")
    report_lines.append(f"hpwl_thresh: {hpwl_thresh}")
    report_lines.append(f"max_outer_iter: {max_outer_iter}")
    report_lines.append(f"enable_hard_iso: {enable_hard_iso}")
    report_lines.append(f"write_flat_output: {write_flat_output}")
    report_lines.append(f"nets: {len(db.nets_list)}")
    report_lines.append(f"pins_in_result_lookup: {len(result_lookup)}")

    active_pins, mod_segs_cache, missing_coords, missing_segments = _build_active_pins_from_result(db, result_lookup)
    print(f"Mapped active pins from result.json: {len(active_pins)} / {db.total_pin_count}")
    report_lines.append(f"active_pins_assigned: {len(active_pins)} / {db.total_pin_count}")
    report_lines.append(f"missing_coords: {len(missing_coords)}")
    report_lines.append(f"missing_segments: {len(missing_segments)}")

    if missing_coords:
        print(f"Pins missing coordinates in result.json: {len(missing_coords)}")
        for key in missing_coords[:10]:
            print("  missing_coord", key)
    if missing_segments:
        print(f"Pins missing usable segments: {len(missing_segments)}")
        for item in missing_segments[:10]:
            print("  missing_segment", item)

    if not active_pins:
        raise RuntimeError("No active pins were initialized from result.json")

    print("Running QP Pin Legalizer (successor HPWL-triggered hard alignment)...")
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

    print(f"Solver process complete. Solved {len(final_state)} pins.")
    report_lines.append(f"solved_pins: {len(final_state)}")

    flat_results = _build_flat_results(final_state, all_real_segments)
    flat_lookup = {(p["parent_inst"], p["pingroup_name"]): p for p in flat_results}
    result_out, updated_count, unchanged_count = _rewrite_result_json_like_input(result_data, flat_lookup)
    report_lines.append(f"updated_result_entries: {updated_count}")
    report_lines.append(f"unchanged_result_entries: {unchanged_count}")

    succ_total, succ_low, succ_aligned, succ_misaligned = _successor_alignment_report(
        result_out, flat_lookup, hpwl_thresh=hpwl_thresh, tol=tol
    )
    noov = _no_overlap_report(flat_results, keepout=keepout, tol=tol)
    disp_mean, disp_max, disp_top = _displacement_report(flat_lookup, result_lookup)
    print(f"[report] successor_edges={succ_total} final_hpwl_lt_{hpwl_thresh:g}={succ_low} aligned_tol={succ_aligned} misaligned={succ_low - succ_aligned}")
    print(f"[report] no_overlap_violations={len(noov)}")
    report_lines.append(f"final_successor_edges_total: {succ_total}")
    report_lines.append(f"final_successor_edges_hpwl_lt_{hpwl_thresh:g}: {succ_low}")
    report_lines.append(f"final_successor_edges_aligned_tol_{tol:g}: {succ_aligned}")
    report_lines.append(f"final_successor_edges_misaligned_tol_{tol:g}: {succ_low - succ_aligned}")
    report_lines.append(f"no_overlap_violations: {len(noov)}")
    for item in noov[:10]:
        report_lines.append(f"  no_overlap_violation: excess={item[0]:.6f} seg={item[1]} a={item[2]} b={item[3]} gap={item[4]:.6f} req={item[5]:.6f}")
    report_lines.append(f"displacement_mean_manhattan: {disp_mean:.6f}")
    report_lines.append(f"displacement_max_manhattan: {disp_max:.6f}")
    for dman, dlinf, key in disp_top:
        report_lines.append(f"  displacement_top: manhattan={dman:.6f} linf={dlinf:.6f} key={key}")
    for delta, hpwl, akey, bkey in succ_misaligned[:10]:
        report_lines.append(f"  misaligned_low_hpwl: delta={delta:.6f} hpwl={hpwl:.6f} a={akey} b={bkey}")

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
        REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
