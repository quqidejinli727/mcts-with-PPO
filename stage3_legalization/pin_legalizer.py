"""Canonical graph pin legalizer.

Clean production route:
  - canonical node source of truth for reused/homology pins;
  - pure-template segment order as the default discrete no-overlap order;
  - boundary + no-overlap hard constraints;
  - movement objective;
  - infeasible-cluster slack diagnostics for location/cause reporting;
  - optional infeasible-only, slack-guided local order fallback.

Legacy master/follower, local-batch, strict-global, final real snap,
and HPWL-objective paths are intentionally removed.
"""


import os
import re
import csv
import json
import warnings
warnings.filterwarnings("ignore", message="Solution may be inaccurate.*")
from collections import defaultdict, Counter


def _env_bool(name, default=False):
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in {"1", "true", "yes", "on"}

def _debug_print(*args, **kwargs):
    # Internal solver/admission logs are debug-only by default.
    if _env_bool("SOLVER_DEBUG_LOG", False):
        print(*args, **kwargs)


def _progress_print(*args, **kwargs):
    # Coarse progress logs stay on by default; disable with SOLVER_PROGRESS_LOG=0.
    if _env_bool("SOLVER_PROGRESS_LOG", True):
        print(*args, **kwargs)


try:
    import cvxpy as cp
except Exception:
    cp = None


class Segment:
    def __init__(self, seg_id, inst_name, orient, free_axis, fixed_coord, lo, hi):
        self.id = seg_id
        self.inst_name = inst_name
        self.orient = orient
        self.free_axis = free_axis
        self.fixed_coord = float(fixed_coord)
        self.lo = float(lo)
        self.hi = float(hi)


class CanonSegment:
    def __init__(self, seg_id, orient, free_axis, fixed_coord, lo, hi):
        self.id = seg_id
        self.orient = orient
        self.free_axis = free_axis
        self.fixed_coord = float(fixed_coord)
        self.lo = float(lo)
        self.hi = float(hi)


class RealPinState:
    def __init__(self):
        self.inst_name = ""
        self.pin_name = ""
        self.seg_id = None
        self.width = 0.0
        self.s_center = 0.0
        self.s_min = 0.0
        self.s_max = 0.0


class IsoGroup:
    def __init__(self, name):
        self.name = name
        self.instances = []
        self.canon_inst = None
        self.canon_segments = []
        self.inst_to_canon_seg = {}   # inst -> real_seg_id -> canon_seg_id
        self.canon_to_inst_seg = {}   # inst -> canon_seg_id -> real_seg_id
        self.inst_affine = {}         # inst -> real_seg_id -> (a,b), s_real = a*s_canon + b
        self.template_pins = {}


# ---------------- geometry basics ----------------

def extract_real_segments(module):
    segs = []
    V = module.vertex
    if not V:
        return segs
    n = len(V)
    for k in range(n):
        x1, y1 = V[k]
        x2, y2 = V[(k + 1) % n]
        if y1 == y2:
            segs.append(Segment((module.name, k), module.name, 'H', 'x', y1, min(x1, x2), max(x1, x2)))
        elif x1 == x2:
            segs.append(Segment((module.name, k), module.name, 'V', 'y', x1, min(y1, y2), max(y1, y2)))
    return segs


def find_segment_by_id(seg_list, seg_id):
    for s in seg_list:
        if s.id == seg_id:
            return s
    raise ValueError(f"segment {seg_id} not found")


def build_iso_groups(all_modules):
    groups = {}
    for m in all_modules:
        if not getattr(m, "vertex", None):
            continue
        gname = m.module_name
        if gname not in groups:
            groups[gname] = IsoGroup(gname)
        groups[gname].instances.append(m.name)
    for G in groups.values():
        G.canon_inst = G.instances[0]
    return groups


def _seg_endpoints(seg):
    if seg.orient == 'H':
        return (seg.lo, seg.fixed_coord), (seg.hi, seg.fixed_coord)
    return (seg.fixed_coord, seg.lo), (seg.fixed_coord, seg.hi)


def _bbox_center_from_vertices(V):
    xs = [float(x) for x, _ in V]
    ys = [float(y) for _, y in V]
    return (0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys)))


_D4_TRANSFORMS = {
    "id": lambda x, y: (x, y),
    "mx": lambda x, y: (x, -y),
    "my": lambda x, y: (-x, y),
    "r180": lambda x, y: (-x, -y),
    "dxy": lambda x, y: (y, x),
    "r90": lambda x, y: (-y, x),
    "r270": lambda x, y: (y, -x),
    "dnegxy": lambda x, y: (-y, -x),
}

_D4_MATRICES = {
    "id": ((1, 0), (0, 1)),
    "mx": ((1, 0), (0, -1)),
    "my": ((-1, 0), (0, 1)),
    "r180": ((-1, 0), (0, -1)),
    "dxy": ((0, 1), (1, 0)),
    "r90": ((0, -1), (1, 0)),
    "r270": ((0, 1), (-1, 0)),
    "dnegxy": ((0, -1), (-1, 0)),
}

_DIRECTION_TO_D4 = {
    0: "id",
    1: "r180",
    2: "r90",
    3: "r270",
    4: "my",
    5: "mx",
    6: "dxy",
    7: "dnegxy",
}

_MATRIX_TO_D4 = {mat: name for name, mat in _D4_MATRICES.items()}


def _transpose2(mat):
    return ((mat[0][0], mat[1][0]), (mat[0][1], mat[1][1]))


def _matmul2(a, b):
    return (
        (
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ),
        (
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ),
    )


def _module_direction_value(module):
    direction = getattr(module, "direction", None)
    if direction is None:
        return None
    try:
        return int(direction)
    except Exception:
        return None


def _required_relative_transform_name(canon_module, inst_module):
    canon_dir = _module_direction_value(canon_module)
    inst_dir = _module_direction_value(inst_module)
    if canon_dir is None or canon_dir < 0:
        raise RuntimeError(
            f"module {canon_module.name} missing valid direction metadata; "
            "direction-only matching requires explicit CoordRotation values"
        )
    if inst_dir is None or inst_dir < 0:
        raise RuntimeError(
            f"module {inst_module.name} missing valid direction metadata; "
            "direction-only matching requires explicit CoordRotation values"
        )

    canon_name = _DIRECTION_TO_D4.get(canon_dir)
    inst_name = _DIRECTION_TO_D4.get(inst_dir)
    if canon_name is None or inst_name is None:
        raise RuntimeError(
            f"unsupported direction mapping canon={canon_dir} inst={inst_dir} for "
            f"{canon_module.name} -> {inst_module.name}"
        )

    rel_mat = _matmul2(_D4_MATRICES[inst_name], _transpose2(_D4_MATRICES[canon_name]))
    rel_name = _MATRIX_TO_D4.get(rel_mat)
    if rel_name is None:
        raise RuntimeError(
            f"cannot derive relative transform from direction metadata canon={canon_dir} inst={inst_dir} "
            f"for {canon_module.name} -> {inst_module.name}"
        )
    return rel_name


def _seg_key_from_endpoints(p0, p1, tol=1e-6):
    # orientation-free hashable key with rounding
    a = (round(min(p0[0], p1[0]), 6), round(min(p0[1], p1[1]), 6))
    b = (round(max(p0[0], p1[0]), 6), round(max(p0[1], p1[1]), 6))
    return (a, b)


def _match_transform(canon_module, inst_module, real_segments, tol=1e-6):
    """
    Use direction metadata to map canonical segments to instance segments.
    Returns:
      seg_map: real_seg_id(inst,k) -> canon_seg_id(edge_idx on canonical)
      affine_map: real_seg_id(inst,k) -> (a,b) with s_real = a*s_canon + b
    """
    canon_inst = canon_module.name
    inst_name = inst_module.name
    canon_segs = real_segments[canon_inst]
    inst_segs = real_segments[inst_name]

    if len(canon_segs) != len(inst_segs):
        raise RuntimeError(f"isomorphic instances {canon_inst} / {inst_name} segment count mismatch")

    ccx, ccy = _bbox_center_from_vertices(canon_module.vertex)
    icx, icy = _bbox_center_from_vertices(inst_module.vertex)

    # instance segment dict by endpoint key
    inst_seg_by_key = {}
    for rs in inst_segs:
        p0, p1 = _seg_endpoints(rs)
        inst_seg_by_key[_seg_key_from_endpoints(p0, p1)] = rs

    tname = _required_relative_transform_name(canon_module, inst_module)
    tf = _D4_TRANSFORMS[tname]
    seg_map = {}
    affine_map = {}
    used_real = set()

    for cs in canon_segs:
        c0, c1 = _seg_endpoints(cs)
        c0_rel = (c0[0] - ccx, c0[1] - ccy)
        c1_rel = (c1[0] - ccx, c1[1] - ccy)

        tc0_rel = tf(*c0_rel)
        tc1_rel = tf(*c1_rel)
        tc0 = (tc0_rel[0] + icx, tc0_rel[1] + icy)
        tc1 = (tc1_rel[0] + icx, tc1_rel[1] + icy)

        key = _seg_key_from_endpoints(tc0, tc1)
        rs = inst_seg_by_key.get(key)
        if rs is None or rs.id in used_real:
            raise RuntimeError(
                f"direction-only mapping failed for {canon_inst} -> {inst_name} using transform {tname}; "
                f"no unique target segment for canonical segment {cs.id}"
            )

        # affine map from canonical scalar to real scalar
        if cs.free_axis == 'x':
            u0 = c0[0]
            u1 = c1[0]
        else:
            u0 = c0[1]
            u1 = c1[1]

        if rs.free_axis == 'x':
            v0 = tc0[0]
            v1 = tc1[0]
        else:
            v0 = tc0[1]
            v1 = tc1[1]

        if abs(u1 - u0) <= tol:
            raise RuntimeError(f"degenerate canonical segment {cs.id} while mapping {canon_inst} -> {inst_name}")

        a = (v1 - v0) / (u1 - u0)
        if abs(abs(a) - 1.0) > 1e-6:
            raise RuntimeError(
                f"direction-only mapping produced invalid affine scale a={a} for {canon_inst} -> {inst_name} "
                f"segment {cs.id}"
            )
        a = 1.0 if a > 0 else -1.0
        b = v0 - a * u0

        seg_map[rs.id] = cs.id[1]
        affine_map[rs.id] = (a, b)
        used_real.add(rs.id)

    return seg_map, affine_map


def build_canonical_segment_map(group, real_segments, module_map):
    canon_inst = group.canon_inst
    canon_real_segs = real_segments[canon_inst]

    for s in canon_real_segs:
        group.canon_segments.append(CanonSegment(s.id[1], s.orient, s.free_axis, s.fixed_coord, s.lo, s.hi))

    canon_mod = module_map[canon_inst]
    for inst in group.instances:
        group.inst_to_canon_seg[inst] = {}
        group.canon_to_inst_seg[inst] = {}
        group.inst_affine[inst] = {}

        if inst == canon_inst:
            for s in real_segments[inst]:
                c_id = s.id[1]
                group.inst_to_canon_seg[inst][s.id] = c_id
                group.canon_to_inst_seg[inst][c_id] = s.id
                group.inst_affine[inst][s.id] = (1.0, 0.0)
            continue

        seg_map, affine_map = _match_transform(canon_mod, module_map[inst], real_segments)
        for real_seg_id, canon_seg_id in seg_map.items():
            group.inst_to_canon_seg[inst][real_seg_id] = canon_seg_id
            group.canon_to_inst_seg[inst][canon_seg_id] = real_seg_id
            group.inst_affine[inst][real_seg_id] = affine_map[real_seg_id]


def _find_group_of_inst(groups, inst):
    for G in groups.values():
        if inst in G.instances:
            return G
    return None


def map_real_s_to_canon_s(groups, inst_name, real_seg_id, real_s):
    G = _find_group_of_inst(groups, inst_name)
    if G is None:
        return real_s
    a, b = G.inst_affine[inst_name][real_seg_id]
    if abs(a) < 1e-12:
        return real_s
    return (float(real_s) - b) / a


# ---------------- template init / expand ----------------


def _pname_order(pname):
    s = str(pname)
    m = re.match(r"^(.*?)(\d+)$", s)
    if m:
        return (m.group(1), int(m.group(2)))
    return (s, 10**9)


# ---------------- inner hard solver ----------------


# ---------------- outer loop ----------------


# ---------------- experimental local net-batch solver ----------------

def _state_from_active_pins_local(active_pins, real_segments, keepout):
    """Build independent real-pin states directly from the assigned segment positions.

    This is used only by LOCAL_NET_BATCH_ENABLE=1.  It intentionally avoids
    global hard-iso template variables so local batches can stay small.
    """
    state = {}
    bucket = defaultdict(list)
    for pin in active_pins:
        inst = getattr(pin, "parent_inst", None)
        pname = getattr(pin, "pingroup_name", None)
        if inst is None or pname is None:
            continue
        if not hasattr(pin, "seg_id") or not hasattr(pin, "s_init"):
            continue
        if inst not in real_segments:
            continue
        try:
            seg = find_segment_by_id(real_segments[inst], pin.seg_id)
        except Exception:
            continue
        width = float(getattr(pin, "width", 0.0))
        s_min = float(seg.lo + keepout + width / 2.0)
        s_max = float(seg.hi - keepout - width / 2.0)
        if s_max < s_min:
            mid = 0.5 * (s_min + s_max)
            s_min = mid
            s_max = mid
        s = max(min(float(pin.s_init), s_max), s_min)
        st = RealPinState()
        st.inst_name = inst
        st.pin_name = pname
        st.seg_id = pin.seg_id
        st.width = width
        st.s_center = float(s)
        st.s_init = float(s)
        st.s_min = s_min
        st.s_max = s_max
        key = (inst, pname)
        state[key] = st
        bucket[st.seg_id].append(key)
    return state, dict(bucket)


def _refresh_bucket_local(state):
    bucket = defaultdict(list)
    for key, st in state.items():
        bucket[st.seg_id].append(key)
    return dict(bucket)


# ---------------- canonical graph cluster legalizer ----------------

class _DSU:
    def __init__(self):
        self.parent = {}
        self.size = defaultdict(int)

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1

    def find(self, x):
        self.add(x)
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return ra


def _canon_interval_from_real(a, b, lo, hi):
    lo = float(lo); hi = float(hi)
    if abs(float(a)) < 1e-12:
        return lo, hi
    x0 = (lo - float(b)) / float(a)
    x1 = (hi - float(b)) / float(a)
    return (min(x0, x1), max(x0, x1))


def _make_noverlap_record(akey, bkey, na, nb, gap, seg_id, same_node):
    """Compact no-overlap record.

    Stored as a tuple to reduce large-case memory.  The old dict shape is still
    accepted by _unpack_noverlap_record for backward compatibility with strict
    rebuild/debug paths.
    """
    return (akey, bkey, na, nb, float(gap), seg_id, bool(same_node))


def _unpack_noverlap_record(ov):
    if isinstance(ov, dict):
        return (
            ov.get("a"),
            ov.get("b"),
            ov.get("na"),
            ov.get("nb"),
            float(ov.get("gap", 0.0)),
            ov.get("seg_id"),
            bool(ov.get("same_node", False)),
        )
    try:
        akey, bkey, na, nb, gap, seg_id, same_node = ov
        return akey, bkey, na, nb, float(gap), seg_id, bool(same_node)
    except Exception:
        return None, None, None, None, 0.0, None, False


def _canonical_node_key_for_real_pin(key, st, groups):
    inst, pname = key
    G = _find_group_of_inst(groups, inst)
    if G is None or len(getattr(G, 'instances', []) or []) <= 1:
        return ("R", inst, pname, tuple(st.seg_id)), 1.0, 0.0, None
    try:
        canon_seg_id = G.inst_to_canon_seg[inst][st.seg_id]
        a, b = G.inst_affine[inst][st.seg_id]
        return ("C", G.name, pname, canon_seg_id), float(a), float(b), G
    except Exception:
        # If canonical mapping is unavailable, keep this pin as an independent
        # real node rather than silently using a wrong homology relation.
        return ("R", inst, pname, tuple(st.seg_id)), 1.0, 0.0, None


def _strict_template_order_ranks_from_graph(graph):
    """Build pure canonical/template segment ranks.

    Current production mode is pure_template only: each canonical segment is
    sorted once by canonical scalar and stable pin key.  No real-coordinate
    voting, consensus, or cycle repair is performed in this pruned build.
    """
    nodes = graph.get("nodes", {})

    def base_rank(n):
        rec = nodes.get(n, {})
        pname = n[2] if isinstance(n, tuple) and len(n) > 2 else str(n)
        return (float(rec.get("s_init", rec.get("s_value", 0.0))), _pname_order(pname), repr(n))

    node_sets = defaultdict(set)
    for n in nodes:
        if isinstance(n, tuple) and len(n) >= 4 and n[0] == "C":
            node_sets[(n[1], n[3])].add(n)

    ranks = {}
    for tkey, nset in node_sets.items():
        out = sorted(nset, key=base_rank)
        ranks[tkey] = {n: i for i, n in enumerate(out)}

    return {
        "ranks": ranks,
        "cycles": [],
        "template_segments": len(ranks),
        "source_segments": 0,
        "mode": "pure_template",
    }


def _strict_order_keys_on_segment(keys, graph, real_state):
    """Order one real segment from pure canonical/template ranks."""
    keys = [k for k in keys if k in real_state]
    real_to_node = graph.get("real_to_node", {})
    occurrence_affine = graph.get("occurrence_affine", {})
    order_info = graph.get("strict_template_order")
    if order_info is None:
        order_info = _strict_template_order_ranks_from_graph(graph)
        graph["strict_template_order"] = order_info
    ranks_by_tkey = order_info.get("ranks", {})

    def key_func(k):
        n = real_to_node.get(k)
        fallback = (float(real_state[k].s_center), _pname_order(k[1]), repr(k))
        if not isinstance(n, tuple) or len(n) < 4 or n[0] != "C":
            return (1, fallback[0], fallback[1], fallback[2])
        tkey = (n[1], n[3])
        ranks = ranks_by_tkey.get(tkey)
        if not ranks or n not in ranks:
            return (1, fallback[0], fallback[1], fallback[2])
        idx = ranks[n]
        a = float(occurrence_affine.get(k, (None, 1.0, 0.0))[1])
        if a < 0:
            idx = -idx
        return (0, idx, fallback[0], fallback[1], fallback[2])

    return sorted(keys, key=key_func)


def _real_xy_from_state(st, real_segments):
    """Recover an occurrence's current 2-D coordinate from its segment state."""
    seg = find_segment_by_id(real_segments[st.inst_name], st.seg_id)
    if seg.free_axis == "x":
        return float(st.s_center), float(seg.fixed_coord)
    return float(seg.fixed_coord), float(st.s_center)


def _compute_total_hpwl_from_state(active_nets, state, real_segments):
    """Compute total net HPWL using the same bbox-center definition as evaluate.py."""
    total = 0.0
    nets = 0
    pins = 0
    single_pin_nets = 0
    empty_nets = 0
    for net in active_nets or []:
        coords = []
        for p in getattr(net, "pins", []) or []:
            key = (getattr(p, "parent_inst", None), getattr(p, "pingroup_name", None))
            st = state.get(key)
            if st is None:
                continue
            try:
                coords.append(_real_xy_from_state(st, real_segments))
            except Exception:
                continue
        if not coords:
            empty_nets += 1
            continue
        nets += 1
        pins += len(coords)
        if len(coords) == 1:
            single_pin_nets += 1
            continue
        xs = [float(x) for x, _y in coords]
        ys = [float(y) for _x, y in coords]
        total += (max(xs) - min(xs)) + (max(ys) - min(ys))
    return {
        "total_hpwl": float(total),
        "nets": int(nets),
        "pins": int(pins),
        "single_pin_nets": int(single_pin_nets),
        "empty_nets": int(empty_nets),
    }


def _reset_state_to_segment(st, new_seg_id, real_segments, keepout, *, prefer_xy=None):
    """Move a RealPinState to a new segment by projecting its current 2-D point.

    This is a topology-level correction, not a post-solve real fallback.  It is
    used before canonical variables are created so all homology occurrences of a
    pin can share the same canonical segment and therefore the same canonical
    node.  The subsequent QP still owns the final scalar value.
    """
    old_seg_id = st.seg_id
    old_s = float(st.s_center)
    if prefer_xy is None:
        try:
            prefer_xy = _real_xy_from_state(st, real_segments)
        except Exception:
            prefer_xy = (float(st.s_center), float(st.s_center))
    new_seg = find_segment_by_id(real_segments[st.inst_name], new_seg_id)
    width = float(st.width)
    s_min = float(new_seg.lo + keepout + width / 2.0)
    s_max = float(new_seg.hi - keepout - width / 2.0)
    if s_max < s_min:
        mid = 0.5 * (s_min + s_max)
        s_min = mid
        s_max = mid
    raw_s = float(prefer_xy[0] if new_seg.free_axis == "x" else prefer_xy[1])
    new_s = max(min(raw_s, s_max), s_min)
    st.seg_id = new_seg_id
    st.s_center = float(new_s)
    st.s_init = float(new_s)
    st.s_min = float(s_min)
    st.s_max = float(s_max)
    return {
        "old_seg_id": old_seg_id,
        "new_seg_id": new_seg_id,
        "old_s": old_s,
        "raw_projected_s": raw_s,
        "new_s": float(new_s),
        "new_bounds": [float(s_min), float(s_max)],
    }


def _enforce_homology_segment_consistency(state, real_segments, groups, keepout):
    """Force each reused module pin to a single canonical segment before QP.

    The previous strict-canonical versions still keyed canonical nodes as
    (group, pin, canon_seg).  If the same pingroup_name landed on different
    canonical segments in different reused instances, that pin was split into
    several canonical variables.  The QP could then be overlap-legal while the
    final geometry violated homology.  This function fixes that modelling gap by
    retargeting all occurrences of the same (homology group, pingroup_name) to a
    single canonical segment before node construction.
    """
    stats = {
        "enabled": bool(_env_bool("CANONICAL_ENFORCE_HOMOLOGY_SEGMENT", True)),
        "groups": 0,
        "split_groups_before": 0,
        "reassigned": 0,
        "skipped_no_mapping": 0,
        "skipped_no_current_mapping": 0,
        "records": [],
    }
    if not stats["enabled"]:
        return stats

    policy = os.environ.get("CANONICAL_HOMOLOGY_SEGMENT_POLICY", "canon_inst").strip().lower()
    by_hp = defaultdict(list)
    for key, st in list(state.items()):
        inst, pname = key
        G = _find_group_of_inst(groups, inst)
        if G is None or len(getattr(G, "instances", []) or []) <= 1:
            continue
        cseg = G.inst_to_canon_seg.get(inst, {}).get(st.seg_id)
        by_hp[(G.name, pname)].append((key, st, G, cseg))

    for hkey, items in sorted(by_hp.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        if len(items) <= 1:
            continue
        stats["groups"] += 1
        csegs = [c for _key, _st, _G, c in items if c is not None]
        if len(set(csegs)) > 1:
            stats["split_groups_before"] += 1
        if not csegs:
            stats["skipped_no_current_mapping"] += len(items)
            continue

        target = None
        if policy in {"canon", "canon_inst", "master"}:
            # Prefer the canonical instance's current segment for this pin if it
            # exists.  This makes homology deterministic and avoids letting a
            # majority of already-inconsistent followers redefine the template.
            for key, st, G, cseg in items:
                if key[0] == G.canon_inst and cseg is not None:
                    target = cseg
                    break
        if target is None:
            counts = Counter(csegs)
            # Deterministic majority; ties by segment id repr.
            target = sorted(counts.items(), key=lambda kv: (-kv[1], repr(kv[0])))[0][0]

        for key, st, G, cseg in items:
            if cseg == target:
                continue
            new_seg_id = G.canon_to_inst_seg.get(key[0], {}).get(target)
            if new_seg_id is None:
                stats["skipped_no_mapping"] += 1
                stats["records"].append({
                    "status": "skipped_no_mapping",
                    "group": hkey[0],
                    "pingroup_name": hkey[1],
                    "parent_inst": key[0],
                    "old_canon_seg": cseg,
                    "target_canon_seg": target,
                    "old_seg_id": st.seg_id,
                })
                continue
            old_xy = _real_xy_from_state(st, real_segments)
            rec2 = _reset_state_to_segment(st, new_seg_id, real_segments, keepout, prefer_xy=old_xy)
            stats["reassigned"] += 1
            stats["records"].append({
                "status": "reassigned",
                "group": hkey[0],
                "pingroup_name": hkey[1],
                "parent_inst": key[0],
                "old_canon_seg": cseg,
                "target_canon_seg": target,
                "old_seg_id": rec2["old_seg_id"],
                "new_seg_id": rec2["new_seg_id"],
                "old_s": rec2["old_s"],
                "raw_projected_s": rec2["raw_projected_s"],
                "new_s": rec2["new_s"],
                "new_bounds": rec2["new_bounds"],
            })

    if _env_bool("CANONICAL_HOMOLOGY_AUDIT_ENABLE", True):
        prefix = os.environ.get("CANONICAL_HOMOLOGY_AUDIT_PREFIX", "canonical_homology")
        csv_path = prefix + "_segment_reassignments.csv"
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                fieldnames = [
                    "status", "group", "pingroup_name", "parent_inst", "old_canon_seg",
                    "target_canon_seg", "old_seg_id", "new_seg_id", "old_s",
                    "raw_projected_s", "new_s", "new_bounds",
                ]
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in stats["records"]:
                    rr = dict(r)
                    for k in ["old_seg_id", "new_seg_id", "new_bounds"]:
                        if k in rr:
                            rr[k] = repr(rr[k])
                    w.writerow({k: rr.get(k, "") for k in fieldnames})
        except Exception as e:
            _progress_print(f"[canonical-graph] warning: failed to write {csv_path}: {e}")
    return stats


def _write_canonical_homology_audit(graph):
    """Write an audit of homology group -> canonical node coverage.

    The decisive invariant is: for each (homology group, pingroup_name), all
    reused real occurrences should map to exactly one canonical node.  If not,
    homology is not actually being enforced even if every cluster QP is optimal.
    """
    if not _env_bool("CANONICAL_HOMOLOGY_AUDIT_ENABLE", True):
        return {"enabled": False}
    prefix = os.environ.get("CANONICAL_HOMOLOGY_AUDIT_PREFIX", "canonical_homology")
    real_state = graph.get("real_state", {})
    real_to_node = graph.get("real_to_node", {})
    occurrence_affine = graph.get("occurrence_affine", {})
    nodes = graph.get("nodes", {})
    by_hp = defaultdict(list)
    for key, st in real_state.items():
        n = real_to_node.get(key)
        if not (isinstance(n, tuple) and len(n) >= 4 and n[0] == "C"):
            continue
        by_hp[(n[1], key[1])].append((key, st, n))

    summary_rows = []
    detail_rows = []
    bad = 0
    for (gname, pname), items in sorted(by_hp.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        nodes_set = sorted({n for _key, _st, n in items}, key=repr)
        csegs = sorted({n[3] for n in nodes_set}, key=repr)
        insts = sorted({key[0] for key, _st, _n in items})
        status = "ok" if len(nodes_set) == 1 else "split"
        if status != "ok":
            bad += 1
        summary_rows.append({
            "status": status,
            "group": gname,
            "pingroup_name": pname,
            "occurrences": len(items),
            "instances": len(insts),
            "canonical_node_count": len(nodes_set),
            "canonical_nodes": repr(nodes_set),
            "canonical_segments": repr(csegs),
        })
        for key, st, n in items:
            aff = occurrence_affine.get(key, (n, 1.0, 0.0))
            rec = nodes.get(n, {})
            detail_rows.append({
                "status": status,
                "group": gname,
                "pingroup_name": pname,
                "parent_inst": key[0],
                "seg_id": repr(st.seg_id),
                "canonical_node": repr(n),
                "canonical_segment": n[3],
                "affine_a": float(aff[1]),
                "affine_b": float(aff[2]),
                "real_s": float(st.s_center),
                "real_bounds": repr([float(st.s_min), float(st.s_max)]),
                "node_bounds": repr([float(rec.get("s_min", float("nan"))), float(rec.get("s_max", float("nan")))]),
            })

    summary_csv = prefix + "_audit_summary.csv"
    detail_csv = prefix + "_audit_detail.csv"
    summary_json = prefix + "_audit_summary.json"
    try:
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["status", "group", "pingroup_name", "occurrences", "instances", "canonical_node_count", "canonical_nodes", "canonical_segments"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)
        with open(detail_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["status", "group", "pingroup_name", "parent_inst", "seg_id", "canonical_node", "canonical_segment", "affine_a", "affine_b", "real_s", "real_bounds", "node_bounds"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in detail_rows:
                w.writerow(r)
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary_rows, f, indent=2, ensure_ascii=False)
    except Exception as e:
        _progress_print(f"[canonical-graph] warning: failed to write homology audit files: {e}")

    stats = {
        "enabled": True,
        "groups": len(summary_rows),
        "split_groups": bad,
        "summary_csv": summary_csv,
        "detail_csv": detail_csv,
    }
    if _env_bool("CANONICAL_HOMOLOGY_AUDIT_FAIL", False) and bad > 0:
        raise RuntimeError(f"canonical homology audit failed: split_groups={bad}; see {summary_csv}")
    return stats

def _build_canonical_graph_state(active_pins, real_segments, groups, keepout):
    """Collapse homology occurrences into canonical nodes and build no-overlap graph.

    Nodes are canonical pin variables.  A real pin occurrence maps to a node by
    s_real = a * s_node + b.  This clean build does not construct successor or HPWL-derived edge data.  No-overlap constraints remain
    occurrence-level constraints on fixed real segments.
    """
    real_state, bucket_by_seg = _state_from_active_pins_local(active_pins, real_segments, keepout)
    homology_segment_stats = _enforce_homology_segment_consistency(real_state, real_segments, groups, keepout)
    bucket_by_seg = _refresh_bucket_local(real_state)

    nodes = {}
    real_to_node = {}
    occurrence_affine = {}
    empty_interval_nodes = []

    for rkey, st in real_state.items():
        nkey, a, b, _G = _canonical_node_key_for_real_pin(rkey, st, groups)
        real_to_node[rkey] = nkey
        occurrence_affine[rkey] = (nkey, float(a), float(b))
        c_lo, c_hi = _canon_interval_from_real(a, b, st.s_min, st.s_max)
        c_init = map_real_s_to_canon_s(groups, rkey[0], st.seg_id, st.s_center) if nkey[0] == "C" else float(st.s_center)
        rec = nodes.setdefault(nkey, {
            "key": nkey,
            "occurrences": [],
            "s_min": float(c_lo),
            "s_max": float(c_hi),
            "s_init_num": 0.0,
            "s_init_den": 0.0,
            "width": float(st.width),
            "canon_seg_id": nkey[-1],
        })
        rec["occurrences"].append(rkey)
        rec["s_min"] = max(float(rec["s_min"]), float(c_lo))
        rec["s_max"] = min(float(rec["s_max"]), float(c_hi))
        rec["s_init_num"] += float(c_init)
        rec["s_init_den"] += 1.0
        rec["width"] = max(float(rec.get("width", 0.0)), float(st.width))

    for rec in nodes.values():
        den = max(1.0, float(rec.pop("s_init_den", 1.0)))
        rec["s_init"] = float(rec.pop("s_init_num", 0.0)) / den
        if float(rec["s_max"]) + 1e-9 < float(rec["s_min"]):
            empty_interval_nodes.append(rec["key"])
            # Strict canonical intersection is empty.  Keep a degenerate bound at
            # the robust initial scalar so the run can diagnose and continue;
            # this node is reported in solver logs.
            mid = float(rec["s_init"])
            rec["s_min"] = mid
            rec["s_max"] = mid
        rec["s_init"] = max(min(float(rec["s_init"]), float(rec["s_max"])), float(rec["s_min"]))
        rec["s_value"] = float(rec["s_init"])

    # Occurrence-level no-overlap constraints on already assigned real segments.
    # In no-align production mode, order is pure canonical/template order: sort
    # each canonical segment once, then map that order back to each real
    # occurrence using affine sign.  Legacy real-order consensus remains
    # available via CANONICAL_STRICT_SEGMENT_ORDER=template.
    noverlap = []
    noverlap_graph_edges = []
    eps = float(os.environ.get("CANONICAL_NO_OVERLAP_EPS", "1e-3"))
    order_graph = {
        "real_state": real_state,
        "bucket_by_seg": bucket_by_seg,
        "nodes": nodes,
        "real_to_node": real_to_node,
        "occurrence_affine": occurrence_affine,
    }
    if os.environ.get("CANONICAL_STRICT_SEGMENT_ORDER", "pure_template").strip().lower() not in {"current", "real", "coordinate"}:
        order_graph["strict_template_order"] = _strict_template_order_ranks_from_graph(order_graph)
    segment_orders = {}
    for seg_id, keys0 in list(bucket_by_seg.items()):
        keys = [k for k in keys0 if k in real_state]
        if len(keys) <= 1:
            continue
        keys = _strict_order_keys_on_segment(keys, order_graph, real_state)
        segment_orders[seg_id] = list(keys)
        for i in range(len(keys) - 1):
            akey, bkey = keys[i], keys[i + 1]
            na, nb = real_to_node.get(akey), real_to_node.get(bkey)
            if na is None or nb is None:
                continue
            gap = (float(real_state[akey].width) + float(real_state[bkey].width)) / 2.0 + eps
            # Do NOT drop na == nb.  A repeated canonical node can still appear
            # multiple times on a real segment through different occurrences or
            # affine offsets.  The resulting constraint is either a valid linear
            # self-constraint on the canonical scalar, or a precise diagnostic of
            # an impossible/incorrect grouping.  Dropping it was one source of
            # residual overlaps in group-sync fallback results.
            noverlap.append(_make_noverlap_record(akey, bkey, na, nb, gap, seg_id, na == nb))
            if na != nb:
                noverlap_graph_edges.append((na, nb))

    return {
        "real_state": real_state,
        "bucket_by_seg": bucket_by_seg,
        "nodes": nodes,
        "real_to_node": real_to_node,
        "occurrence_affine": occurrence_affine,
        "real_segments": real_segments,
        "noverlap": noverlap,
        "noverlap_graph_edges": noverlap_graph_edges,
        "segment_orders": segment_orders,
        "order_fallback_records": [],
        "empty_interval_nodes": empty_interval_nodes,
        "strict_template_order": order_graph.get("strict_template_order"),
        "homology_segment_stats": homology_segment_stats,
    }


def _partition_canonical_graph(nodes, noverlap_graph_edges):
    """Partition canonical nodes using only hard no-overlap connectivity."""
    dsu = _DSU()
    for n in nodes:
        dsu.add(n)
    for u, v in noverlap_graph_edges:
        dsu.union(u, v)

    clusters = defaultdict(list)
    for n in nodes:
        clusters[dsu.find(n)].append(n)

    cluster_list = []
    for _root, ns in clusters.items():
        cluster_list.append({"id": len(cluster_list), "nodes": set(ns), "selected_edges": set()})

    node_to_cluster = {}
    for c in cluster_list:
        for n in c["nodes"]:
            node_to_cluster[n] = c["id"]
    return cluster_list, node_to_cluster


def _canonical_real_expr(real_key, var, fixed_values, occurrence_affine):
    nkey, a, b = occurrence_affine[real_key]
    base = var[nkey] if nkey in var else float(fixed_values[nkey])
    return float(a) * base + float(b)


def _canonical_movement_objective_terms(cluster_nodes, var, graph, weight):
    terms = []
    nodes = graph["nodes"]
    for n, v in var.items():
        if n not in cluster_nodes:
            continue
        rec = nodes[n]
        occ_count = max(1, len(rec.get("occurrences", []) or []))
        anchor = float(rec.get("s_init", rec.get("s_value", 0.0)))
        terms.append(float(weight) * float(occ_count) * cp.square(v - anchor))
    return terms


def _solve_canonical_cluster_qp(cluster_nodes, selected_edge_ids, graph, fixed_values=None, apply=True, soft_edge_ids=None):
    """Solve one no-align canonical cluster QP.

    Current production model: bounds + no-overlap constraints + movement
    objective only.  HPWL-objective and successor-driven code paths are absent from
    this clean build.
    """
    if cp is None:
        raise RuntimeError("CVXPY not available; cannot run canonical cluster QP")

    nodes = graph["nodes"]
    occurrence_affine = graph["occurrence_affine"]
    if fixed_values is None:
        fixed_values = {n: float(nodes[n].get("s_value", nodes[n]["s_init"])) for n in nodes}

    cluster_nodes = set(cluster_nodes)
    var = {n: cp.Variable(name="cg_" + str(abs(hash(n)))) for n in sorted(cluster_nodes, key=repr)}

    constraints = []
    for n, v in var.items():
        rec = nodes[n]
        constraints.append(v >= float(rec["s_min"]))
        constraints.append(v <= float(rec["s_max"]))

    for ov in graph.get("noverlap", []) or []:
        akey, bkey, na, nb, gap, _seg_id, _same_node = _unpack_noverlap_record(ov)
        if na not in cluster_nodes or nb not in cluster_nodes:
            continue
        lhs = (
            _canonical_real_expr(bkey, var, fixed_values, occurrence_affine)
            - _canonical_real_expr(akey, var, fixed_values, occurrence_affine)
        )
        constraints.append(lhs >= float(gap))

    move_weight = float(os.environ.get("CANONICAL_MOVE_WEIGHT", os.environ.get("LOCAL_BATCH_MOVE_WEIGHT", "1.0")))
    obj_terms = _canonical_movement_objective_terms(cluster_nodes, var, graph, move_weight)
    if not obj_terms:
        obj_terms.append(0)

    prob = cp.Problem(cp.Minimize(sum(obj_terms)), constraints)
    try:
        with warnings.catch_warnings():
            if not _env_bool("SOLVER_DEBUG_LOG", False):
                warnings.simplefilter("ignore")
            prob.solve(
                solver="CLARABEL",
                verbose=False,
                warm_start=True,
                max_iter=int(os.environ.get(
                    "CANONICAL_CLARABEL_MAX_ITER",
                    os.environ.get("LOCAL_BATCH_CLARABEL_MAX_ITER", os.environ.get("CLARABEL_MAX_ITER", "1000")),
                )),
            )
    except Exception as e:
        return False, "exception:" + str(e), {}

    allow_inaccurate = _env_bool("ACCEPT_INACCURATE_SOLVE", False)
    ok_status = {cp.OPTIMAL}
    if allow_inaccurate:
        ok_status.add(cp.OPTIMAL_INACCURATE)
    if prob.status not in ok_status:
        return False, str(prob.status), {}

    sol = {}
    for n, v in var.items():
        if v.value is None:
            continue
        val = float(v.value)
        val = max(min(val, float(nodes[n]["s_max"])), float(nodes[n]["s_min"]))
        sol[n] = val
    if apply:
        for n, val in sol.items():
            nodes[n]["s_value"] = float(val)
    return True, str(prob.status), sol



def _canonical_diag_segment_info(graph, seg_id):
    """Return human-readable location/capacity info for one no-overlap diagnostic record."""
    real_state = graph.get("real_state", {})
    bucket_by_seg = graph.get("bucket_by_seg", {})
    real_segments = graph.get("real_segments", {})
    eps = float(os.environ.get("CANONICAL_NO_OVERLAP_EPS", "1e-3"))
    out = {
        "location": repr(seg_id),
        "parent_inst": "",
        "segment_index": "",
        "segment_free_axis": "",
        "segment_fixed_coord": "",
        "segment_lo": "",
        "segment_hi": "",
        "segment_available_length": "",
        "segment_pin_count": 0,
        "segment_total_width": 0.0,
        "segment_required_length": 0.0,
        "segment_overflow": 0.0,
    }
    try:
        if isinstance(seg_id, (list, tuple)) and len(seg_id) >= 2:
            out["parent_inst"] = seg_id[0]
            out["segment_index"] = seg_id[1]
            seg = find_segment_by_id(real_segments.get(seg_id[0], []), seg_id)
            out["segment_free_axis"] = getattr(seg, "free_axis", "")
            out["segment_fixed_coord"] = float(getattr(seg, "fixed_coord", 0.0))
            lo = float(getattr(seg, "lo", 0.0))
            hi = float(getattr(seg, "hi", 0.0))
            if hi < lo:
                lo, hi = hi, lo
            out["segment_lo"] = lo
            out["segment_hi"] = hi
            out["segment_available_length"] = hi - lo
    except Exception:
        pass
    try:
        keys = [k for k in bucket_by_seg.get(seg_id, []) if k in real_state]
        widths = [float(real_state[k].width) for k in keys]
        out["segment_pin_count"] = int(len(widths))
        out["segment_total_width"] = float(sum(widths))
        out["segment_required_length"] = float(sum(widths) + max(0, len(widths) - 1) * eps)
        if out["segment_available_length"] != "":
            out["segment_overflow"] = float(out["segment_required_length"] - float(out["segment_available_length"]))
    except Exception:
        pass
    return out


def _canonical_diag_cause_guess(same_node, seg_info, rec_a, rec_b, slack):
    """Classify the likely reason for a relaxed no-overlap slack."""
    if bool(same_node):
        return "same_node_noverlap"
    try:
        if float(seg_info.get("segment_overflow", 0.0) or 0.0) > 1e-7:
            return "segment_capacity_overflow"
    except Exception:
        pass
    try:
        a_len = float(rec_a.get("s_max")) - float(rec_a.get("s_min"))
        b_len = float(rec_b.get("s_max")) - float(rec_b.get("s_min"))
        if min(a_len, b_len) <= max(1e-7, float(slack) + 1e-7):
            return "narrow_canonical_node_bounds"
    except Exception:
        pass
    return "order_affine_interval_conflict"


def _diagnose_canonical_cluster_slack(cluster_id, cluster_nodes, graph, *, topk=20):
    """Run a relaxed legality QP for one infeasible canonical cluster.

    Boundary constraints remain hard. Each in-cluster no-overlap constraint gets
    a nonnegative slack. Largest positive slacks identify the segment/order,
    affine, grouping, or capacity relation responsible for infeasibility.
    """
    if cp is None:
        return {"cluster_id": int(cluster_id), "ok": False, "status": "cvxpy_unavailable", "records": []}

    nodes = graph["nodes"]
    real_state = graph.get("real_state", {})
    occurrence_affine = graph["occurrence_affine"]
    fixed_values = {n: float(nodes[n].get("s_value", nodes[n].get("s_init", 0.0))) for n in nodes}
    cluster_nodes = set(cluster_nodes)
    var = {n: cp.Variable(name="cd_" + str(abs(hash((cluster_id, n))))) for n in sorted(cluster_nodes, key=repr)}

    constraints = []
    for n, v in var.items():
        rec = nodes[n]
        constraints.append(v >= float(rec["s_min"]))
        constraints.append(v <= float(rec["s_max"]))

    ov_items = []
    slack_vars = []
    for idx, ov in enumerate(graph.get("noverlap", []) or []):
        akey, bkey, na, nb, gap, seg_id, same_node = _unpack_noverlap_record(ov)
        if na not in cluster_nodes or nb not in cluster_nodes:
            continue
        lhs = _canonical_real_expr(bkey, var, fixed_values, occurrence_affine) - _canonical_real_expr(akey, var, fixed_values, occurrence_affine)
        sv = cp.Variable(nonneg=True, name="cds_" + str(idx))
        constraints.append(lhs + sv >= float(gap))
        slack_vars.append(sv)
        ov_items.append((akey, bkey, na, nb, float(gap), seg_id, bool(same_node)))

    move_weight = float(os.environ.get("CANONICAL_MOVE_WEIGHT", os.environ.get("LOCAL_BATCH_MOVE_WEIGHT", "1.0")))
    slack_l1 = float(os.environ.get("CANONICAL_CLUSTER_DIAG_SLACK_L1", "1000000.0"))
    slack_l2 = float(os.environ.get("CANONICAL_CLUSTER_DIAG_SLACK_L2", "1000.0"))
    obj_terms = []
    if slack_vars:
        sv_vec = cp.hstack(slack_vars)
        obj_terms.append(slack_l1 * cp.sum(sv_vec))
        obj_terms.append(slack_l2 * cp.sum_squares(sv_vec))
    for n, v in var.items():
        rec = nodes[n]
        occ_count = max(1, len(rec.get("occurrences", []) or []))
        anchor = float(rec.get("s_init", rec.get("s_value", 0.0)))
        obj_terms.append(float(move_weight) * float(occ_count) * cp.square(v - anchor))
    if not obj_terms:
        obj_terms.append(0)

    prob = cp.Problem(cp.Minimize(sum(obj_terms)), constraints)
    try:
        with warnings.catch_warnings():
            if not _env_bool("SOLVER_DEBUG_LOG", False):
                warnings.simplefilter("ignore")
            prob.solve(
                solver="CLARABEL",
                verbose=False,
                warm_start=True,
                max_iter=int(os.environ.get("CANONICAL_CLUSTER_DIAG_CLARABEL_MAX_ITER", os.environ.get("CLARABEL_MAX_ITER", "5000"))),
            )
    except Exception as e:
        return {"cluster_id": int(cluster_id), "ok": False, "status": "exception:" + str(e), "records": []}

    allow_inaccurate = _env_bool("ACCEPT_INACCURATE_SOLVE", False)
    ok_status = {cp.OPTIMAL}
    if allow_inaccurate:
        ok_status.add(cp.OPTIMAL_INACCURATE)
    if prob.status not in ok_status:
        return {"cluster_id": int(cluster_id), "ok": False, "status": str(prob.status), "records": []}

    sol = {}
    for n, v in var.items():
        if v.value is not None:
            sol[n] = float(v.value)

    def _eval_real(rkey):
        n, a, b = occurrence_affine[rkey]
        return float(a) * float(sol.get(n, fixed_values[n])) + float(b)

    records = []
    for item, sv in zip(ov_items, slack_vars):
        try:
            slack = max(0.0, float(sv.value))
        except Exception:
            slack = 0.0
        if slack <= 1e-7:
            continue
        akey, bkey, na, nb, gap, seg_id, same_node = item
        _na, aa, ba = occurrence_affine[akey]
        _nb, ab, bb = occurrence_affine[bkey]
        aval = _eval_real(akey)
        bval = _eval_real(bkey)
        ast = real_state.get(akey)
        bst = real_state.get(bkey)
        rec_a = nodes.get(na, {})
        rec_b = nodes.get(nb, {})
        seg_info = _canonical_diag_segment_info(graph, seg_id)
        cause_guess = _canonical_diag_cause_guess(same_node, seg_info, rec_a, rec_b, slack)
        rec = {
            "cluster_id": int(cluster_id),
            "_seg_id_obj": seg_id,
            "cause_guess": cause_guess,
            "slack": float(slack),
            "gap_required": float(gap),
            "lhs_after_relax": float(bval - aval),
            "seg_id": repr(seg_id),
            "same_node": bool(same_node),
            "a_pin": repr(akey),
            "b_pin": repr(bkey),
            "a_node": repr(na),
            "b_node": repr(nb),
            "a_real_s": float(aval),
            "b_real_s": float(bval),
            "a_width": float(getattr(ast, "width", 0.0)) if ast is not None else 0.0,
            "b_width": float(getattr(bst, "width", 0.0)) if bst is not None else 0.0,
            "a_bounds": f"[{getattr(ast, 's_min', '')},{getattr(ast, 's_max', '')}]" if ast is not None else "",
            "b_bounds": f"[{getattr(bst, 's_min', '')},{getattr(bst, 's_max', '')}]" if bst is not None else "",
            "a_affine": f"{float(aa)}*t+{float(ba)}",
            "b_affine": f"{float(ab)}*t+{float(bb)}",
            "a_node_bounds": f"[{rec_a.get('s_min','')},{rec_a.get('s_max','')}]",
            "b_node_bounds": f"[{rec_b.get('s_min','')},{rec_b.get('s_max','')}]",
            "a_node_occurrences": int(len(rec_a.get("occurrences", []) or [])),
            "b_node_occurrences": int(len(rec_b.get("occurrences", []) or [])),
        }
        rec.update(seg_info)
        records.append(rec)
    records.sort(key=lambda r: -float(r["slack"]))
    top_records = records[:max(0, int(topk))]
    return {
        "cluster_id": int(cluster_id),
        "ok": True,
        "status": str(prob.status),
        "slack_sum": float(sum(float(r["slack"]) for r in records)),
        "slack_max": float(records[0]["slack"]) if records else 0.0,
        "slack_pos": int(len(records)),
        "records": top_records,
    }


def _write_canonical_cluster_diag_files(graph):
    """Write accumulated infeasible-cluster slack diagnostics."""
    diags = list(graph.get("cluster_slack_diagnostics", []) or [])
    if not diags:
        return {"clusters": 0, "records": 0}
    csv_path = os.environ.get("CANONICAL_CLUSTER_DIAG_CSV", "canonical_cluster_slack_diagnostics.csv")
    json_path = os.environ.get("CANONICAL_CLUSTER_DIAG_JSON", "canonical_cluster_slack_diagnostics.json")
    rows = []
    for d in diags:
        for r in d.get("records", []) or []:
            rr = dict(r)
            rr["cluster_status"] = d.get("status", "")
            rr["cluster_slack_sum"] = d.get("slack_sum", 0.0)
            rr["cluster_slack_max"] = d.get("slack_max", 0.0)
            rr["cluster_slack_pos"] = d.get("slack_pos", 0)
            rows.append(rr)
    fieldnames = [
        "cluster_id", "cluster_status", "cluster_slack_sum", "cluster_slack_max", "cluster_slack_pos",
        "cause_guess", "location", "parent_inst", "segment_index", "segment_free_axis",
        "segment_fixed_coord", "segment_lo", "segment_hi", "segment_available_length",
        "segment_pin_count", "segment_total_width", "segment_required_length", "segment_overflow",
        "slack", "gap_required", "lhs_after_relax", "seg_id", "same_node",
        "a_pin", "b_pin", "a_node", "b_node", "a_real_s", "b_real_s",
        "a_width", "b_width", "a_bounds", "b_bounds", "a_affine", "b_affine",
        "a_node_bounds", "b_node_bounds", "a_node_occurrences", "b_node_occurrences",
    ]
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, "") for k in fieldnames})
    except Exception as e:
        _progress_print(f"[canonical-graph] warning: failed to write {csv_path}: {e}")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(diags, f, ensure_ascii=False, indent=2)
    except Exception as e:
        _progress_print(f"[canonical-graph] warning: failed to write {json_path}: {e}")
    return {"clusters": len(diags), "records": len(rows), "csv": csv_path, "json": json_path}



def _segment_noverlap_records_from_order(seg_id, ordered_keys, graph):
    """Build no-overlap records for one real segment from an explicit key order."""
    real_state = graph["real_state"]
    real_to_node = graph["real_to_node"]
    eps = float(os.environ.get("CANONICAL_NO_OVERLAP_EPS", "1e-3"))
    records = []
    graph_edges = []
    keys = [k for k in ordered_keys if k in real_state]
    for akey, bkey in zip(keys, keys[1:]):
        na = real_to_node.get(akey)
        nb = real_to_node.get(bkey)
        if na is None or nb is None:
            continue
        gap = (float(real_state[akey].width) + float(real_state[bkey].width)) / 2.0 + eps
        records.append(_make_noverlap_record(akey, bkey, na, nb, gap, seg_id, na == nb))
        if na != nb:
            graph_edges.append((na, nb))
    return records, graph_edges


def _interval_order_for_segment(seg_id, cluster_nodes, graph):
    """Return an interval-based real-coordinate order for one segment.

    This fallback order is derived from canonical node feasible intervals mapped
    back into the real segment coordinate system.  It does not use current real
    pin coordinate voting and is only used after a cluster has already failed
    the pure-template QP.
    """
    real_state = graph["real_state"]
    occurrence_affine = graph["occurrence_affine"]
    nodes = graph["nodes"]
    keys = [k for k in graph.get("bucket_by_seg", {}).get(seg_id, []) if k in real_state and k in occurrence_affine]
    if len(keys) <= 1:
        return None

    def interval_key(k):
        st = real_state[k]
        n, a, b = occurrence_affine[k]
        rec = nodes.get(n, {})
        try:
            lo, hi = _real_range_from_canon(a, b, rec.get("s_min", st.s_min), rec.get("s_max", st.s_max))
            lo = max(float(lo), float(st.s_min))
            hi = min(float(hi), float(st.s_max))
            if hi < lo:
                # Keep a deterministic key even when the projected interval is empty.
                mid = 0.5 * (float(lo) + float(hi))
            else:
                mid = 0.5 * (float(lo) + float(hi))
        except Exception:
            lo = float(st.s_min)
            hi = float(st.s_max)
            mid = 0.5 * (lo + hi)
        return (mid, hi, lo, _pname_order(k[1]), repr(k))

    return sorted(keys, key=interval_key)


def _eligible_order_fallback_segments(dinfo, graph):
    """Aggregate slack records into ordered candidate segments for local order repair."""
    allowed = set(
        x.strip()
        for x in os.environ.get("CANONICAL_ORDER_FALLBACK_CAUSES", "order_affine_interval_conflict").split(",")
        if x.strip()
    )
    if _env_bool("CANONICAL_ORDER_FALLBACK_ALLOW_NARROW_BOUNDS", False):
        allowed.add("narrow_canonical_node_bounds")
    by_seg = {}
    for r in dinfo.get("records", []) or []:
        cause = str(r.get("cause_guess", ""))
        if cause not in allowed:
            continue
        if bool(r.get("same_node", False)):
            continue
        try:
            if float(r.get("segment_overflow", 0.0) or 0.0) > 1e-7:
                continue
        except Exception:
            pass
        seg_id = r.get("_seg_id_obj")
        if seg_id is None:
            # Older diagnostics only contain repr(seg_id); those cannot be safely
            # edited in-memory.  Fresh diagnostics from this build always carry
            # the object under _seg_id_obj.
            continue
        entry = by_seg.setdefault(seg_id, {"seg_id": seg_id, "slack_sum": 0.0, "slack_max": 0.0, "records": 0, "causes": Counter()})
        slack = float(r.get("slack", 0.0) or 0.0)
        entry["slack_sum"] += slack
        entry["slack_max"] = max(float(entry["slack_max"]), slack)
        entry["records"] += 1
        entry["causes"][cause] += 1
    out = list(by_seg.values())
    out.sort(key=lambda e: (-float(e["slack_sum"]), -float(e["slack_max"]), repr(e["seg_id"])))
    return out


def _apply_segment_order_replacements(graph, base_noverlap, replacements):
    """Replace no-overlap records on selected segments with records from new orders."""
    repl_keys = set(replacements.keys())
    new_noverlap = []
    for ov in base_noverlap:
        _akey, _bkey, _na, _nb, _gap, seg_id, _same_node = _unpack_noverlap_record(ov)
        if seg_id in repl_keys:
            continue
        new_noverlap.append(ov)
    for seg_id, order in replacements.items():
        records, _edges = _segment_noverlap_records_from_order(seg_id, order, graph)
        new_noverlap.extend(records)
    graph["noverlap"] = new_noverlap


def _try_infeasible_order_fallback(cid, cluster_nodes, graph, dinfo, *, progress=True):
    """Try a bounded, slack-guided order repair for one infeasible cluster.

    Default pure_template order remains the source of truth.  This routine only
    runs after the normal cluster QP is infeasible and a slack diagnostic points
    to order/interval conflicts rather than same-node or segment-capacity causes.
    Each candidate order is verified by re-solving the same cluster QP before it
    is committed.
    """
    if not _env_bool("CANONICAL_ORDER_FALLBACK_ENABLE", True):
        return False, "disabled", {}
    if not dinfo or not dinfo.get("ok"):
        return False, "no_valid_slack_diag", {}

    candidates = _eligible_order_fallback_segments(dinfo, graph)
    if not candidates:
        return False, "no_eligible_segments", {}

    max_segments = int(os.environ.get("CANONICAL_ORDER_FALLBACK_MAX_SEGMENTS", "8"))
    max_trials = int(os.environ.get("CANONICAL_ORDER_FALLBACK_MAX_TRIALS", "16"))
    if max_segments > 0:
        candidates = candidates[:max_segments]

    base_noverlap = list(graph.get("noverlap", []) or [])
    old_orders = {seg_id: list(graph.get("segment_orders", {}).get(seg_id, [])) for seg_id in graph.get("segment_orders", {})}
    replacements = {}
    trial_count = 0
    attempted = []

    for cand in candidates:
        if max_trials > 0 and trial_count >= max_trials:
            break
        seg_id = cand["seg_id"]
        new_order = _interval_order_for_segment(seg_id, cluster_nodes, graph)
        if not new_order:
            continue
        old_order = list(graph.get("segment_orders", {}).get(seg_id, []))
        if old_order == new_order:
            continue
        replacements[seg_id] = new_order
        _apply_segment_order_replacements(graph, base_noverlap, replacements)
        trial_count += 1
        ok, status, _sol = _solve_canonical_cluster_qp(cluster_nodes, set(), graph, apply=False, soft_edge_ids=set())
        attempted.append({
            "seg_id": repr(seg_id),
            "slack_sum": float(cand.get("slack_sum", 0.0)),
            "slack_max": float(cand.get("slack_max", 0.0)),
            "records": int(cand.get("records", 0)),
            "status": str(status),
            "ok": bool(ok),
            "old_order": [repr(x) for x in old_order],
            "new_order": [repr(x) for x in new_order],
        })
        if progress:
            _progress_print(
                f"[canonical-graph] cluster={cid} order_fallback trial={trial_count} "
                f"segments={len(replacements)} latest_seg={seg_id} ok={int(bool(ok))} status={status}"
            )
        if ok:
            ok2, status2, _sol2 = _solve_canonical_cluster_qp(cluster_nodes, set(), graph, apply=True, soft_edge_ids=set())
            if ok2:
                graph.setdefault("segment_orders", {}).update({sid: list(order) for sid, order in replacements.items()})
                rec = {
                    "cluster_id": int(cid),
                    "status": str(status2),
                    "segments_reordered": [repr(sid) for sid in replacements.keys()],
                    "trials": int(trial_count),
                    "attempted": attempted,
                }
                graph.setdefault("order_fallback_records", []).append(rec)
                return True, "order_fallback:" + str(status2), rec
            # Extremely unlikely: verification was feasible but apply failed.
            attempted[-1]["apply_status"] = str(status2)

    graph["noverlap"] = base_noverlap
    if "segment_orders" in graph:
        for sid, order in old_orders.items():
            graph["segment_orders"][sid] = order
    rec = {
        "cluster_id": int(cid),
        "status": "failed",
        "segments_considered": [repr(c["seg_id"]) for c in candidates],
        "trials": int(trial_count),
        "attempted": attempted,
    }
    graph.setdefault("order_fallback_records", []).append(rec)
    return False, "order_fallback_failed", rec


def _write_order_fallback_report(graph):
    records = list(graph.get("order_fallback_records", []) or [])
    if not records:
        return {"records": 0}
    path = os.environ.get("CANONICAL_ORDER_FALLBACK_JSON", "canonical_order_fallback_report.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    except Exception as e:
        _progress_print(f"[canonical-graph] warning: failed to write {path}: {e}")
        return {"records": len(records), "error": str(e)}
    return {"records": len(records), "json": path}

def _solve_canonical_clusters(graph, clusters, node_to_cluster):
    """Solve all canonical clusters under the current no-align model."""
    progress = _env_bool("CANONICAL_CLUSTER_PROGRESS_LOG", True)
    solve_enable = _env_bool("CANONICAL_CLUSTER_QP_ENABLE", True)
    max_solve_vars = int(os.environ.get("CANONICAL_CLUSTER_SOLVE_MAX_VARS", "300"))
    diag_enable = _env_bool("CANONICAL_CLUSTER_DIAG_ENABLE", True)
    diag_max_clusters = int(os.environ.get("CANONICAL_CLUSTER_DIAG_MAX_CLUSTERS", "20"))
    diag_max_vars = int(os.environ.get("CANONICAL_CLUSTER_DIAG_MAX_VARS", "0"))
    diag_topk = int(os.environ.get("CANONICAL_CLUSTER_DIAG_TOPK", "50"))
    diag_done = 0
    graph["cluster_slack_diagnostics"] = []

    sizes = [len(c["nodes"]) for c in clusters]
    if progress and sizes:
        _progress_print(
            f"[canonical-graph] cluster_stats count={len(sizes)} max_nodes={max(sizes)} "
            f"avg_nodes={sum(sizes)/len(sizes):.1f} solve_max_vars={max_solve_vars}"
        )

    solved = 0
    infeasible = 0
    skipped = 0
    for c in clusters:
        cid = c["id"]
        nvars = len(c["nodes"])
        if (not solve_enable) or (max_solve_vars > 0 and nvars > max_solve_vars):
            skipped += 1
            c["selected_edges"] = set()
            if progress:
                _progress_print(f"[canonical-graph] cluster={cid} skipped nodes={nvars}")
            continue

        if progress:
            _progress_print(f"[canonical-graph] cluster={cid} start nodes={nvars}")
        ok, status, _sol = _solve_canonical_cluster_qp(c["nodes"], set(), graph, apply=True, soft_edge_ids=set())
        c["selected_edges"] = set()
        if ok:
            solved += 1
        else:
            dinfo = None
            if diag_enable and diag_done < diag_max_clusters and (diag_max_vars <= 0 or nvars <= diag_max_vars):
                dinfo = _diagnose_canonical_cluster_slack(cid, c["nodes"], graph, topk=diag_topk)
                graph["cluster_slack_diagnostics"].append(dinfo)
                diag_done += 1
                if progress:
                    _progress_print(
                        f"[canonical-graph] cluster={cid} slack_diag ok={int(bool(dinfo.get('ok')))} "
                        f"status={dinfo.get('status')} slack_pos={dinfo.get('slack_pos', 0)} "
                        f"slack_max={float(dinfo.get('slack_max', 0.0)):.6g} "
                        f"slack_sum={float(dinfo.get('slack_sum', 0.0)):.6g}"
                    )
                repaired, rstatus, _rinfo = _try_infeasible_order_fallback(
                    cid, c["nodes"], graph, dinfo, progress=progress
                )
                if repaired:
                    ok = True
                    status = rstatus
                    solved += 1
                    if progress:
                        _progress_print(f"[canonical-graph] cluster={cid} order_fallback_applied status={status}")
                else:
                    if progress and _env_bool("CANONICAL_ORDER_FALLBACK_ENABLE", True):
                        _progress_print(f"[canonical-graph] cluster={cid} order_fallback_not_applied reason={rstatus}")
            elif diag_enable and progress and diag_max_vars > 0 and nvars > diag_max_vars:
                _progress_print(
                    f"[canonical-graph] cluster={cid} slack_diag skipped nodes={nvars} "
                    f"diag_max_vars={diag_max_vars}"
                )
            if not ok:
                infeasible += 1
        if progress:
            _progress_print(f"[canonical-graph] cluster={cid} done ok={int(ok)} status={status}")

    if diag_enable:
        wstats = _write_canonical_cluster_diag_files(graph)
        if progress and wstats.get("clusters", 0):
            _progress_print(f"[canonical-graph] cluster_slack_diag_written={wstats}")
    ofstats = _write_order_fallback_report(graph)
    if progress and ofstats.get("records", 0):
        _progress_print(f"[canonical-graph] order_fallback_report_written={ofstats}")
    return defaultdict(set), [], solved, infeasible

def _expand_canonical_graph_state_to_real(graph, real_segments):
    real_state = graph["real_state"]
    nodes = graph["nodes"]
    for rkey, st in real_state.items():
        nkey, a, b = graph["occurrence_affine"][rkey]
        val = float(a) * float(nodes[nkey].get("s_value", nodes[nkey].get("s_init", 0.0))) + float(b)
        val = max(min(val, float(st.s_max)), float(st.s_min))
        st.s_center = float(val)
    return real_state


def solve_canonical_graph_legalization(all_modules, active_pins, active_nets, keepout,
                                        hpwl_thresh=500.0, max_outer_iter=1, tol=1e-3,
                                        enable_hard_iso=True):
    """Canonical no-align legality-only graph legalizer.

    This pruned build keeps only the production path:
      - canonical node source of truth;
      - homology segment retarget;
      - pure-template order;
      - hard no-overlap + boundary constraints;
      - movement objective.
    """
    module_map = {m.name: m for m in all_modules if getattr(m, "vertex", None)}
    real_segments = {m.name: extract_real_segments(m) for m in module_map.values()}
    groups = build_iso_groups(all_modules)

    for G in groups.values():
        try:
            build_canonical_segment_map(G, real_segments, module_map)
        except Exception as e:
            _debug_print(f"[canonical-graph] skip canonical map for group={G.name}: {e}")

    graph = _build_canonical_graph_state(active_pins, real_segments, groups, keepout)

    initial_hpwl_info = None
    if _env_bool("CANONICAL_PRINT_TOTAL_HPWL", False) and active_nets:
        initial_hpwl_info = _compute_total_hpwl_from_state(active_nets, graph["real_state"], real_segments)

    hseg_stats = graph.get("homology_segment_stats", {}) or {}
    hseg_brief = {
        "enabled": hseg_stats.get("enabled"),
        "groups": hseg_stats.get("groups"),
        "split_before": hseg_stats.get("split_groups_before"),
        "reassigned": hseg_stats.get("reassigned"),
        "skipped_no_mapping": hseg_stats.get("skipped_no_mapping"),
    }
    _progress_print(f"[canonical-graph] homology_segment_enforce={hseg_brief}")

    haudit = _write_canonical_homology_audit(graph)
    if haudit.get("enabled"):
        _progress_print(
            f"[canonical-graph] homology_audit groups={haudit.get('groups')} split_groups={haudit.get('split_groups')} "
            f"summary={haudit.get('summary_csv')} detail={haudit.get('detail_csv')}"
        )

    clusters, node_to_cluster = _partition_canonical_graph(
        graph["nodes"], graph["noverlap_graph_edges"]
    )
    order_info = graph.get("strict_template_order") or {}
    _progress_print(
        f"[canonical-graph] nodes={len(graph['nodes'])} real_pins={len(graph['real_state'])} "
        f"clusters={len(clusters)} empty_interval_nodes={len(graph['empty_interval_nodes'])} "
        f"order_mode={os.environ.get('CANONICAL_STRICT_SEGMENT_ORDER', 'pure_template')} "
        f"template_orders={order_info.get('template_segments', 0)} order_cycles={len(order_info.get('cycles', []) or [])}"
    )

    _selected, _cut_edges, solved, infeasible = _solve_canonical_clusters(graph, clusters, node_to_cluster)
    _progress_print(f"[canonical-graph] solved_clusters={solved} infeasible_clusters={infeasible}")

    state = _expand_canonical_graph_state_to_real(graph, real_segments)
    if _env_bool("CANONICAL_PRINT_TOTAL_HPWL", False) and active_nets:
        final_hpwl_info = _compute_total_hpwl_from_state(active_nets, state, real_segments)
        _progress_print(f"[canonical-graph] total_hpwl = {final_hpwl_info.get('total_hpwl', 0.0):.6f}")
    return state, real_segments


def solve_global_qp_with_outer_order_update(all_modules, active_pins, active_nets, keepout,
                                            hpwl_thresh=500.0, max_outer_iter=5, tol=1e-3,
                                            enable_hard_iso=True):
    """Compatibility wrapper for the current canonical no-align route."""
    _progress_print("[canonical-graph] enabled: bypass legacy local-batch/master-follower paths")
    return solve_canonical_graph_legalization(
        all_modules, active_pins, active_nets, keepout,
        hpwl_thresh=hpwl_thresh, max_outer_iter=max_outer_iter, tol=tol,
        enable_hard_iso=enable_hard_iso,
    )
