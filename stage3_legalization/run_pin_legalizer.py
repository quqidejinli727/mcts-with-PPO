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
# Width-pressure admission heuristic. Wide/tight candidate pairs are tried first.
os.environ.setdefault("ADMISSION_WIDTH_PRIORITY", "0")
os.environ.setdefault("COMPONENT_FALLBACK_POLICY", "nearest_fixed_axis")
os.environ.setdefault("FOLLOWER_ANCHOR_CLOSURE", "1")
os.environ.setdefault("LOCAL_FIXEDPOINT_ITERS", "1")
# Adaptive fast-first mode: keep the ordinary outer loop close to the old fast
# path; enter required-closure / policy target-order rescue only if final
# report-style blockers remain.
os.environ.setdefault("ADAPTIVE_STRICT_CLOSURE", "1")
os.environ.setdefault("FINAL_RESCUE_COUNT_GATE", "1")
# Keep coarse [inner] progress on by default; detailed solver/admission logs remain debug-only.
os.environ.setdefault("SOLVER_DEBUG_LOG", "0")
os.environ.setdefault("SOLVER_PROGRESS_LOG", "1")
# Control optional outputs.
os.environ.setdefault("WRITE_FLAT_OUTPUT", "1")
os.environ.setdefault("WRITE_REPORT_OUTPUT", "0")
# Print true same-axis blocker details after excluding cross-axis, pair-empty,
# and component-empty edges intentionally dropped by nearest-fixed-axis policy.
os.environ.setdefault("ALIGN_BLOCKER_REPORT", "1")
os.environ.setdefault("ALIGN_BLOCKER_LIMIT", "50")
# Generic final residual rescue. No module-specific names are used.
os.environ.setdefault("FINAL_RESCUE_ENABLE", "1")
os.environ.setdefault("FINAL_RESCUE_ROUNDS", "2")
os.environ.setdefault("FINAL_RESCUE_SEARCH_CAP", "10")
os.environ.setdefault("FINAL_RESCUE_SWAP_CAP", "3")
os.environ.setdefault("FINAL_RESCUE_REMOVAL_POOL_CAP", "32")
os.environ.setdefault("FINAL_RESCUE_BEAM_WIDTH", "128")
os.environ.setdefault("FINAL_RESCUE_GLOBAL_SWAP", "1")
# Accept final-rescue mutations only when the final residual score improves.
os.environ.setdefault("FINAL_RESCUE_SCORE_ACCEPT", "1")
os.environ.setdefault("FINAL_RESCUE_SCORE_MODE", "count_total_max")
# Diagnostic-only bounded minimum-removal search for remaining blockers.
os.environ.setdefault("FINAL_BLOCKER_DIAGNOSE", "1")
os.environ.setdefault("FINAL_BLOCKER_DIAG_LIMIT", "50")
os.environ.setdefault("FINAL_BLOCKER_DIAG_MAX_REMOVE", "3")
os.environ.setdefault("FINAL_BLOCKER_DIAG_POOL_CAP", "24")
os.environ.setdefault("FINAL_BLOCKER_DIAG_BEAM_WIDTH", "128")
# Admission-layer score/replacement. This uses the same true-blocker objective
# as final rescue during ordinary family admission.
os.environ.setdefault("ADMISSION_SCORE_ACCEPT", "0")
os.environ.setdefault("ADMISSION_REPLACEMENT_ENABLE", "0")
os.environ.setdefault("ADMISSION_REPLACEMENT_MAX_REMOVE", "3")
os.environ.setdefault("ADMISSION_REPLACEMENT_POOL_CAP", "24")
os.environ.setdefault("ADMISSION_REPLACEMENT_BEAM_WIDTH", "64")

# Required-edge hard closure: after ordinary admission/rescue, every actionable same-axis
# residual is either committed as a hard equality / follower-anchor equality, or reported
# as infeasible under bounded replacement.
os.environ.setdefault("REQUIRED_HARD_CLOSURE_ENABLE", "1")
os.environ.setdefault("REQUIRED_HARD_CLOSURE_ROUNDS", "5")
os.environ.setdefault("REQUIRED_HARD_CLOSURE_MAX_REMOVE", "3")
os.environ.setdefault("REQUIRED_HARD_CLOSURE_POOL_CAP", "32")
os.environ.setdefault("REQUIRED_HARD_CLOSURE_BEAM_WIDTH", "128")

# Policy-blocked reference-master edges: first try component-level plan B; if a component cannot be fully anchored, keep the HPWL-shortest edge.
os.environ.setdefault("POLICY_TARGET_FREEZE_BOTH", "1")
os.environ.setdefault("POLICY_TARGET_TRY_MIDPOINT", "1")
os.environ.setdefault("POLICY_COMPONENT_FALLBACK_POLICY", "min_hpwl")
# Policy component fallback tries candidates in HPWL order, but never tries candidates above this limit.
os.environ.setdefault("POLICY_COMPONENT_FALLBACK_HPWL_LIMIT", "500")
# Final rescue for the representative low-HPWL policy blockers that remain after required closure.
os.environ.setdefault("POLICY_MIN_EDGE_CONFLICT_REPLACE_ENABLE", "1")
os.environ.setdefault("POLICY_MIN_EDGE_CONFLICT_REPLACE_ROUNDS", "3")
os.environ.setdefault("POLICY_MIN_EDGE_CONFLICT_REPLACE_MAX_REMOVE", "3")
os.environ.setdefault("POLICY_MIN_EDGE_CONFLICT_REPLACE_POOL_CAP", "32")
os.environ.setdefault("POLICY_MIN_EDGE_CONFLICT_REPLACE_BEAM_WIDTH", "128")
# Final policy representative rescue: scan candidate scalar targets and rebuild
# order/no-overlap constraints around each target before declaring infeasible.
os.environ.setdefault("POLICY_MIN_EDGE_TARGET_SCAN_ENABLE", "1")
os.environ.setdefault("POLICY_MIN_EDGE_TARGET_SCAN_POINTS", "17")
os.environ.setdefault("POLICY_MIN_EDGE_TARGET_ORDER_HINT", "1")
os.environ.setdefault("POLICY_MIN_EDGE_ALIAS_REMOVAL_ENABLE", "1")
os.environ.setdefault("POLICY_MIN_EDGE_CONFLICT_VERBOSE", "0")
os.environ.setdefault("POLICY_MIN_EDGE_CONFLICT_DETAIL_LIMIT", "0")

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


def _make_segment_lookup(all_real_segments):
    lookup = {}
    for inst, segs in all_real_segments.items():
        for seg in segs:
            lookup[tuple(seg.id)] = seg
    return lookup


def _flat_interval(fp, seg_lookup, keepout: float):
    seg_id_raw = fp.get("seg_id", [])
    if not isinstance(seg_id_raw, (list, tuple)) or len(seg_id_raw) != 2:
        return None
    seg = seg_lookup.get(tuple(seg_id_raw))
    if seg is None:
        return None
    width = float(fp.get("width", 0.0))
    return (float(seg.lo) + float(keepout) + 0.5 * width,
            float(seg.hi) - float(keepout) - 0.5 * width)


def _successor_alignment_report(result_out, flat_lookup, all_real_segments, keepout: float, hpwl_thresh: float, tol: float, policy_dropped_edge_keys=None):
    """Report alignment with the production semantics.

    Count as alignable only if:
      - final HPWL < threshold;
      - both endpoints have the same free_axis;
      - the two endpoint legal scalar intervals intersect.

    Cross-axis and pair-empty cases are skipped rather than counted as
    misaligned because they are not valid same-axis hard-align targets.

    Component-empty cases are diagnosed separately. When a component has no
    common scalar intersection, the solver intentionally keeps one edge by the
    nearest-fixed-axis policy and drops the rest from the ordinary misaligned
    count. Those dropped edges are reported as component_empty_dropped_by_policy.
    """
    seg_lookup = _make_segment_lookup(all_real_segments)
    policy_dropped_edge_keys = set(policy_dropped_edge_keys or [])
    stats = {
        "successor_edges": 0,
        "low_hpwl_total": 0,
        "cross_axis_low_hpwl_skipped": 0,
        "same_axis_low_hpwl": 0,
        "same_axis_pair_empty_skipped": 0,
        "same_axis_evaluable": 0,
        "same_axis_aligned": 0,
        "same_axis_misaligned_raw": 0,
        "same_axis_misaligned": 0,
        "true_same_axis_misaligned": 0,
        "component_empty_count": 0,
        "component_empty_edges": 0,
        "component_empty_kept_by_policy": 0,
        "component_empty_dropped_by_policy": 0,
        "component_empty_dropped_misaligned": 0,
        "policy_component_dropped_by_policy": 0,
        "policy_component_dropped_misaligned": 0,
    }
    misaligned_raw = []
    pair_empty = []
    comp_edges = []

    def _fixed_axis_distance_from_scopes(axis, ascope, bscope):
        if axis == "x":
            return abs(float(ascope[1]) - float(bscope[1]))
        if axis == "y":
            return abs(float(ascope[0]) - float(bscope[0]))
        return float("inf")

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
                stats["successor_edges"] += 1
                ax, ay = map(float, af["scope"])
                bx, by = map(float, bf["scope"])
                hpwl = abs(ax - bx) + abs(ay - by)
                if hpwl >= hpwl_thresh:
                    continue
                stats["low_hpwl_total"] += 1

                fa, fb = af.get("free_axis"), bf.get("free_axis")
                if fa != fb or fa not in {"x", "y"}:
                    stats["cross_axis_low_hpwl_skipped"] += 1
                    continue
                stats["same_axis_low_hpwl"] += 1

                ia = _flat_interval(af, seg_lookup, keepout)
                ib = _flat_interval(bf, seg_lookup, keepout)
                if ia is None or ib is None:
                    stats["same_axis_pair_empty_skipped"] += 1
                    continue
                lo = max(float(ia[0]), float(ib[0]))
                hi = min(float(ia[1]), float(ib[1]))
                if lo > hi + 1e-9:
                    stats["same_axis_pair_empty_skipped"] += 1
                    pair_empty.append({
                        "gap": lo - hi,
                        "hpwl": hpwl,
                        "axis": fa,
                        "akey": akey,
                        "bkey": bkey,
                        "lo": lo,
                        "hi": hi,
                    })
                    continue

                stats["same_axis_evaluable"] += 1
                delta = abs(ax - bx) if fa == "x" else abs(ay - by)
                fixed_axis_dist = _fixed_axis_distance_from_scopes(fa, af["scope"], bf["scope"])
                rec = {
                    "edge_key": (akey, bkey),
                    "akey": akey,
                    "bkey": bkey,
                    "axis": fa,
                    "ia": ia,
                    "ib": ib,
                    "lo": lo,
                    "hi": hi,
                    "slack": hi - lo,
                    "hpwl": hpwl,
                    "delta": delta,
                    "fixed_axis_distance": fixed_axis_dist,
                    "a_width": float(af.get("width", 0.0)),
                    "b_width": float(bf.get("width", 0.0)),
                    "a_scope": tuple(map(float, af.get("scope", [0.0, 0.0]))),
                    "b_scope": tuple(map(float, bf.get("scope", [0.0, 0.0]))),
                }
                comp_edges.append(rec)
                if delta <= tol:
                    stats["same_axis_aligned"] += 1
                else:
                    stats["same_axis_misaligned_raw"] += 1
                    misaligned_raw.append(rec)

    # Component-level interval diagnosis for same-axis pair-feasible edges.
    parent = {}
    node_axis = {}
    node_iv = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for rec in comp_edges:
        akey = rec["akey"]
        bkey = rec["bkey"]
        axis = rec["axis"]
        union(akey, bkey)
        node_axis[akey] = axis
        node_axis[bkey] = axis
        node_iv[akey] = rec["ia"]
        node_iv[bkey] = rec["ib"]

    comps = {}
    for n in parent:
        comps.setdefault(find(n), []).append(n)

    component_empty_dropped_edge_keys = set()
    component_empty_dropped = []
    for nodes in comps.values():
        if len(nodes) <= 2:
            continue
        axes = {node_axis.get(n) for n in nodes}
        if len(axes) != 1:
            continue
        lo = max(float(node_iv[n][0]) for n in nodes if n in node_iv)
        hi = min(float(node_iv[n][1]) for n in nodes if n in node_iv)
        if lo <= hi + 1e-9:
            continue

        nset = set(nodes)
        edges = [rec for rec in comp_edges if rec["akey"] in nset and rec["bkey"] in nset]
        if not edges:
            continue
        stats["component_empty_count"] += 1
        stats["component_empty_edges"] += len(edges)

        # Match the solver-side component fallback policy.
        policy = os.environ.get("COMPONENT_FALLBACK_POLICY", "nearest_fixed_axis").strip().lower()
        if policy in {"nearest", "nearest_fixed", "nearest_fixed_axis", "fixed_axis"}:
            kept = min(edges, key=lambda r: (r["fixed_axis_distance"], r["hpwl"], str(r["edge_key"])))
        elif policy in {"hpwl", "shortest", "shortest_hpwl"}:
            kept = min(edges, key=lambda r: (r["hpwl"], str(r["edge_key"])))
        elif policy in {"width", "wide"}:
            kept = min(edges, key=lambda r: (-max(r["a_width"], r["b_width"]), r["hpwl"], str(r["edge_key"])))
        else:
            kept = min(edges, key=lambda r: (
                -max(r["a_width"], r["b_width"]) / max(r["slack"], 1e-6),
                -max(r["a_width"], r["b_width"]),
                r["slack"],
                r["hpwl"],
                str(r["edge_key"]),
            ))
        stats["component_empty_kept_by_policy"] += 1
        for rec in edges:
            if rec is kept:
                continue
            component_empty_dropped_edge_keys.add(rec["edge_key"])
            component_empty_dropped.append({**rec, "component_lo": lo, "component_hi": hi, "component_gap": lo - hi})
            stats["component_empty_dropped_by_policy"] += 1

    true_misaligned = []
    for rec in misaligned_raw:
        norm_edge_key = rec["edge_key"] if rec["edge_key"][0] <= rec["edge_key"][1] else (rec["edge_key"][1], rec["edge_key"][0])
        if rec["edge_key"] in component_empty_dropped_edge_keys:
            stats["component_empty_dropped_misaligned"] += 1
        elif norm_edge_key in policy_dropped_edge_keys:
            stats["policy_component_dropped_misaligned"] += 1
        else:
            true_misaligned.append(rec)

    stats["policy_component_dropped_by_policy"] = len(policy_dropped_edge_keys)
    stats["true_same_axis_misaligned"] = len(true_misaligned)
    # Preserve the old key name, but make it the actionable count after removing
    # component-empty edges intentionally dropped by nearest policy.
    stats["same_axis_misaligned"] = len(true_misaligned)

    true_misaligned.sort(reverse=True, key=lambda r: r["delta"])
    component_empty_dropped.sort(reverse=True, key=lambda r: r["delta"])
    pair_empty.sort(reverse=True, key=lambda r: r["gap"])
    return stats, true_misaligned, pair_empty, component_empty_dropped

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
    output_json = os.environ.get("OUTPUT_JSON", DEFAULT_OUTPUT)
    flat_output_json = os.environ.get("FLAT_OUTPUT_JSON", DEFAULT_FLAT_OUTPUT)
    keepout = float(os.environ.get("KEEP_OUT", str(DEFAULT_KEEP_OUT)))
    hpwl_thresh = float(os.environ.get("HPWL_THRESH", str(DEFAULT_HPWL_THRESH)))
    max_outer_iter = int(os.environ.get("MAX_OUTER_ITER", str(DEFAULT_MAX_OUTER_ITER)))
    tol = float(os.environ.get("TOL", str(DEFAULT_TOL)))
    enable_hard_iso = os.environ.get("ENABLE_HARD_ISO", "1" if DEFAULT_ENABLE_HARD_ISO else "0").strip() not in {"0", "false", "False", "no", "NO"}
    write_flat_output = os.environ.get("WRITE_FLAT_OUTPUT", "1").strip() not in {"0", "false", "False", "no", "NO"}
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

    print("Running pin_legalizer...")
    print("[version] solver=hpwl500-adaptive-fast-target-order")
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

    policy_dropped_edge_keys = getattr(_solver_mod, "LAST_POLICY_COMPONENT_DROPPED_EDGE_KEYS", set())
    succ_stats, succ_misaligned, succ_pair_empty, succ_component_dropped = _successor_alignment_report(
        result_out, flat_lookup, all_real_segments, keepout=keepout, hpwl_thresh=hpwl_thresh, tol=tol,
        policy_dropped_edge_keys=policy_dropped_edge_keys,
    )
    noov = _no_overlap_report(flat_results, keepout=keepout, tol=tol)
    disp_mean, disp_max, disp_top = _displacement_report(flat_lookup, result_lookup)

    # User-facing result summary.  Keep the old solver's terse style: progress
    # lines during optimization, then a compact final result.  Internal categories
    # such as component/policy drops remain available in the optional report file
    # but are not printed by default.
    skipped_non_actionable = (
        succ_stats.get("cross_axis_low_hpwl_skipped", 0)
        + succ_stats.get("same_axis_pair_empty_skipped", 0)
        + succ_stats.get("component_empty_dropped_by_policy", 0)
        + succ_stats.get("policy_component_dropped_by_policy", 0)
    )
    print(
        f"[result] pins={len(final_state)} updated={updated_count} "
        f"successor_edges={succ_stats['successor_edges']} low_hpwl={succ_stats['low_hpwl_total']}"
    )
    print(
        f"[result] same_axis_evaluable={succ_stats['same_axis_evaluable']} "
        f"aligned={succ_stats['same_axis_aligned']} misaligned={succ_stats['true_same_axis_misaligned']} "
        f"skipped={skipped_non_actionable}"
    )
    print(f"[result] no_overlap_violations={len(noov)}")
    print(f"[result] displacement_mean={disp_mean:.6f} displacement_max={disp_max:.6f}")

    # Optional report, written only when WRITE_REPORT_OUTPUT=1.  Keep it compact
    # and avoid exposing internal policy/drop category names.
    report_lines.append(f"successor_edges: {succ_stats['successor_edges']}")
    report_lines.append(f"low_hpwl: {succ_stats['low_hpwl_total']}")
    report_lines.append(f"same_axis_evaluable: {succ_stats['same_axis_evaluable']}")
    report_lines.append(f"aligned: {succ_stats['same_axis_aligned']}")
    report_lines.append(f"misaligned: {succ_stats['true_same_axis_misaligned']}")
    report_lines.append(f"skipped: {skipped_non_actionable}")
    report_lines.append(f"no_overlap_violations: {len(noov)}")
    for item in noov[:10]:
        report_lines.append(f"  no_overlap_violation: excess={item[0]:.6f} seg={item[1]} a={item[2]} b={item[3]} gap={item[4]:.6f} req={item[5]:.6f}")
    report_lines.append(f"displacement_mean_manhattan: {disp_mean:.6f}")
    report_lines.append(f"displacement_max_manhattan: {disp_max:.6f}")
    for dman, dlinf, key in disp_top:
        report_lines.append(f"  displacement_top: manhattan={dman:.6f} linf={dlinf:.6f} key={key}")
    for rec in succ_misaligned[:10]:
        report_lines.append(
            f"  blocker: delta={rec['delta']:.6f} hpwl={rec['hpwl']:.6f} "
            f"axis={rec['axis']} slack={rec['slack']:.6f} "
            f"a={rec['akey']} b={rec['bkey']}"
        )

    align_blocker_report = os.environ.get("ALIGN_BLOCKER_REPORT", "1").strip().lower() not in {"0", "false", "False", "no", "NO"}
    align_blocker_limit = int(os.environ.get("ALIGN_BLOCKER_LIMIT", "50"))
    if align_blocker_report and succ_misaligned:
        shown = min(len(succ_misaligned), max(0, align_blocker_limit))
        print(f"[blocker] count={len(succ_misaligned)} showing={shown}")
        for idx, rec in enumerate(succ_misaligned[:shown], 1):
            print(
                f"[blocker] #{idx:02d} axis={rec['axis']} delta={rec['delta']:.6f} "
                f"hpwl={rec['hpwl']:.6f} slack={rec['slack']:.6f} "
                f"a={rec['akey']} b={rec['bkey']}"
            )

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
