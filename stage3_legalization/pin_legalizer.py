"""QP pin legalizer.

Stable hard-isomorphism legalization flow.

Core policy:
  - Reused modules use a master-follower model: the selected master initializes
    and drives the shared template; follower instances are affine projections of
    that template.
  - Master/template variables remain optimizable. They are not fixed to the
    initial result.json coordinates.
  - Follower-side successor hard-pairs do not pull the master by default.
  - Hard constraints are bounds, segment order/no-overlap, and admitted
    hard-align equalities.
  - Initial result.json coordinates are used only as a weak soft anchor through
    MOVE_ANCHOR_WEIGHT.
  - Candidate admission uses a dynamic HPWL gate and deterministic
    trial-failure cache to avoid repeated infeasible GUROBI solves.
  - Cross-axis successor edges are not hard-aligned. Pair-level empty legal
    interval intersections are skipped before admission.
  - If several same-axis successor edges form a component whose common interval
    is empty, keep one edge by nearest fixed-axis distance (for example GH to
    the physically nearest GH1/GH2/GH3/GH4 leaf) instead of forcing the whole
    component.
  - Follower-side successor edges that would pull the master are not admitted
    as ordinary equalities; optionally they are handled after the main solve by
    follower-anchor closure: the follower endpoint stays fixed and the other
    endpoint is pulled to that scalar coordinate if feasible.
  - Final rescue can use local swap, but swap acceptance is guarded by a
    residual-alignment score. Remaining blockers can be diagnosed by trying
    direct insertion and bounded minimum-removal searches over nearby active
    hard-pairs.

Removed from this stable flow:
  - RRR rescue
  - target-first admission
  - target-closure rounds
  - force-diagnostic admission
  - non-GUROBI fallback solvers
"""


import math
import os
import re
import numpy as np
import scipy.optimize as opt
from collections import defaultdict
from itertools import combinations

# Exposed for run-script reporting: policy-blocked components that could not
# be fully anchored may intentionally keep only the HPWL-shortest edge.
LAST_POLICY_COMPONENT_DROPPED_EDGE_KEYS = set()
LAST_POLICY_COMPONENT_DROPPED_RECORDS = []
# Policy/free-target edges intentionally removed from required alignment accounting.
LAST_POLICY_FREE_TARGET_DROPPED_EDGE_KEYS = set()
LAST_POLICY_FREE_TARGET_DROPPED_RECORDS = []

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


class TemplatePinVar:
    def __init__(self):
        self.group_name = ""
        self.pin_name = ""
        self.canon_seg_id = None
        self.canon_s = 0.0
        self.width = 0.0


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


class HardAlignPair:
    def __init__(self, pin_a, pin_b, axis):
        self.pin_a = pin_a
        self.pin_b = pin_b
        self.axis = axis

    def as_undirected_key(self):
        a = self.pin_a
        b = self.pin_b
        return (a, b) if a <= b else (b, a)

    def __hash__(self):
        return hash(self.as_undirected_key())

    def __eq__(self, other):
        if not isinstance(other, HardAlignPair):
            return False
        return self.as_undirected_key() == other.as_undirected_key()


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


def find_segment_by_global_id(real_segments, seg_id):
    inst_name = seg_id[0]
    return find_segment_by_id(real_segments[inst_name], seg_id)


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


def _apply_iso_master_overrides(groups):
    """Optionally override the reference/master instance of each iso group.

    Environment format:
        ISO_MASTER_OVERRIDES=F=TOP.U_F;K=TOP.U_K;J1=TOP.U_J.U_J1

    The selected master becomes G.canon_inst. The affine maps are then built
    from that master to every other reused instance. This supports the policy:
    B/C/D follow A, but A is not pulled by constraints coming from B/C/D.
    """
    spec = os.environ.get("ISO_MASTER_OVERRIDES", "").strip()
    if not spec:
        return
    for raw in spec.split(";"):
        raw = raw.strip()
        if not raw or "=" not in raw:
            continue
        gname, inst = raw.split("=", 1)
        gname = gname.strip()
        inst = inst.strip()
        G = groups.get(gname)
        if G is None:
            _debug_print(f"[iso-master] warning: group {gname!r} not found for override {raw!r}")
            continue
        if inst not in G.instances:
            _debug_print(f"[iso-master] warning: instance {inst!r} not in iso group {gname!r}; keep default {G.canon_inst!r}")
            continue
        G.canon_inst = inst
        G.instances = [inst] + [x for x in G.instances if x != inst]
        _debug_print(f"[iso-master] group {gname}: master={inst}")


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


def _get_inst_affine(groups, inst, real_seg_id):
    G = _find_group_of_inst(groups, inst)
    if G is None:
        raise RuntimeError(f"instance {inst} not in any iso group")
    return G.inst_affine[inst][real_seg_id]


def map_real_s_to_canon_s(groups, inst_name, real_seg_id, real_s):
    G = _find_group_of_inst(groups, inst_name)
    if G is None:
        return real_s
    a, b = G.inst_affine[inst_name][real_seg_id]
    if abs(a) < 1e-12:
        return real_s
    return (float(real_s) - b) / a


def map_canon_s_to_real_s(groups, inst_name, real_seg_id, canon_s):
    G = _find_group_of_inst(groups, inst_name)
    if G is None:
        return canon_s
    a, b = G.inst_affine[inst_name][real_seg_id]
    return a * float(canon_s) + b


# ---------------- template init / expand ----------------

def init_template_vars_from_assignment(active_pins, groups, real_segments):
    """Initialize canonical template pins.

    Default historical behavior averaged samples from all reused instances.
    With ISO_MASTER_REFERENCE=1, initialize each template pin from the selected
    master/canonical instance only.  This makes A the reference and makes B/C/D
    follow A instead of letting B/C/D constraints move the shared template.
    """
    master_reference = str(os.environ.get("ISO_MASTER_REFERENCE", "1")).strip().lower() not in {"0", "false", "no"}

    for G in groups.values():
        all_samples = {}
        master_samples = {}
        for pin in active_pins:
            inst = pin.parent_inst
            if inst not in G.instances:
                continue
            if not hasattr(pin, 'seg_id') or not hasattr(pin, 's_init'):
                continue

            pname = pin.pingroup_name
            real_seg_id = pin.seg_id
            real_s = pin.s_init

            if real_seg_id not in G.inst_to_canon_seg[inst]:
                continue

            canon_seg_id = G.inst_to_canon_seg[inst][real_seg_id]
            canon_s = map_real_s_to_canon_s(groups, inst, real_seg_id, real_s)
            sample = (canon_seg_id, canon_s, float(pin.width))
            all_samples.setdefault(pname, []).append(sample)
            if inst == G.canon_inst:
                master_samples.setdefault(pname, []).append(sample)

        pnames = set(all_samples.keys())
        for pname in pnames:
            arr = master_samples.get(pname) if master_reference and master_samples.get(pname) else all_samples.get(pname)
            if not arr:
                continue
            tvar = TemplatePinVar()
            tvar.group_name = G.name
            tvar.pin_name = pname

            cnt = {}
            for item in arr:
                cnt[item[0]] = cnt.get(item[0], 0) + 1
            tvar.canon_seg_id = max(cnt.items(), key=lambda x: x[1])[0]
            tvar.canon_s = float(sum(x[1] for x in arr) / len(arr))
            tvar.width = float(arr[0][2])
            tvar.ref_canon_s = float(tvar.canon_s)
            tvar.ref_canon_seg_id = tvar.canon_seg_id
            G.template_pins[pname] = tvar

def expand_template_to_instances(groups, real_segments, keepout):
    inst_pin_state = {}
    bucket_by_seg = {}
    for G in groups.values():
        for inst in G.instances:
            for pname, tvar in G.template_pins.items():
                if tvar.canon_seg_id not in G.canon_to_inst_seg[inst]:
                    continue
                real_seg_id = G.canon_to_inst_seg[inst][tvar.canon_seg_id]
                real_s = map_canon_s_to_real_s(groups, inst, real_seg_id, tvar.canon_s)
                seg = find_segment_by_id(real_segments[inst], real_seg_id)
                width = float(tvar.width)

                s_min = float(seg.lo + keepout + width / 2.0)
                s_max = float(seg.hi - keepout - width / 2.0)
                if s_max < s_min:
                    mid = 0.5 * (s_min + s_max)
                    s_min = mid
                    s_max = mid

                real_s = max(min(real_s, s_max), s_min)

                st = RealPinState()
                st.inst_name, st.pin_name = inst, pname
                st.seg_id, st.s_center, st.width = real_seg_id, float(real_s), width
                st.s_min, st.s_max = s_min, s_max

                key = (inst, pname)
                inst_pin_state[key] = st
                bucket_by_seg.setdefault(real_seg_id, []).append(key)
    return inst_pin_state, bucket_by_seg




def _pname_order(pname):
    s = str(pname)
    m = re.match(r"^(.*?)(\d+)$", s)
    if m:
        return (m.group(1), int(m.group(2)))
    return (s, 10**9)


def _hard_pair_components(hard_pairs):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for hp in hard_pairs:
        union(hp.pin_a, hp.pin_b)

    comps = defaultdict(list)
    for x in list(parent.keys()):
        comps[find(x)].append(x)
    return comps

def _parse_successor_full_name(full_name):
    if not full_name or '.' not in full_name:
        return None
    inst, pin = full_name.rsplit('.', 1)
    if not inst or not pin:
        return None
    return (inst, pin)


def build_hard_align_pairs(active_nets, inst_pin_state, real_segments, hpwl_thresh):
    """
    Build hard-align candidates from explicit successor edges while preserving
    the existing HPWL trigger rule.

    This keeps the client's "HPWL-triggered" requirement, but no longer drops
    all multi-pin nets on the floor. Each pin->successor edge becomes one
    candidate pair, deduplicated as an undirected edge.
    """
    pairs = []
    seen = set()

    for net in active_nets:
        for p in getattr(net, 'pins', []):
            key1 = (p.parent_inst, p.pingroup_name)
            if key1 not in inst_pin_state:
                continue

            for succ_full in getattr(p, 'successors', []) or []:
                key2 = _parse_successor_full_name(succ_full)
                if key2 is None or key2 not in inst_pin_state:
                    continue

                undirected = (key1, key2) if key1 <= key2 else (key2, key1)
                if undirected in seen:
                    continue
                seen.add(undirected)

                st1, st2 = inst_pin_state[key1], inst_pin_state[key2]
                seg1 = find_segment_by_global_id(real_segments, st1.seg_id)
                seg2 = find_segment_by_global_id(real_segments, st2.seg_id)

                # Keep current model restriction: only same-orientation pairs are
                # represented as 1D hard-align equalities.
                if seg1.orient != seg2.orient:
                    continue

                if seg1.free_axis == 'x':
                    hpwl = abs(st1.s_center - st2.s_center) + abs(seg1.fixed_coord - seg2.fixed_coord)
                else:
                    hpwl = abs(seg1.fixed_coord - seg2.fixed_coord) + abs(st1.s_center - st2.s_center)

                if hpwl < hpwl_thresh:
                    pairs.append(HardAlignPair(key1, key2, seg1.free_axis))

    return pairs


def build_orders(bucket_by_seg, hard_pairs, inst_pin_state):
    """
    Multi-mate aware local order builder.

    For each pin, use the average position of all aligned mates as anchor.
    This avoids the old single-mate overwrite bug.
    """
    mate_pos_lists = defaultdict(list)
    for hp in hard_pairs:
        if hp.pin_a in inst_pin_state and hp.pin_b in inst_pin_state:
            mate_pos_lists[hp.pin_a].append(float(inst_pin_state[hp.pin_b].s_center))
            mate_pos_lists[hp.pin_b].append(float(inst_pin_state[hp.pin_a].s_center))

    orders = {}
    for seg_id, seg_pin_list in bucket_by_seg.items():
        def key_func(k):
            vals = mate_pos_lists.get(k)
            anchor = sum(vals) / len(vals) if vals else float(inst_pin_state[k].s_center)
            return (anchor, float(inst_pin_state[k].s_center), _pname_order(k[1]))

        orders[seg_id] = sorted(seg_pin_list, key=key_func)
    return orders

def topo_sort_with_tie_break(node_set, edge_map, base_rank):
    indeg = {u: 0 for u in node_set}
    for u, vs in edge_map.items():
        indeg.setdefault(u, 0)
        for v in vs:
            indeg[v] = indeg.get(v, 0) + 1

    ready = sorted([u for u in indeg if indeg[u] == 0], key=base_rank)
    out = []
    while ready:
        u = ready.pop(0)
        out.append(u)
        for v in sorted(edge_map.get(u, []), key=base_rank):
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
        ready.sort(key=base_rank)

    if len(out) != len(indeg):
        return None
    return out


def build_orders_template(groups, hard_pairs, inst_pin_state, real_segments):
    """Hard-iso mode: infer template/canonical order from hard-align connected components,
    not from a single overwritten mate pointer.

    Returns:
      (template_orders, cyclic_keys)
    """
    bucket = {}
    for key, st in inst_pin_state.items():
        bucket.setdefault(st.seg_id, []).append(key)

    base_orders = {
        seg_id: sorted(lst, key=lambda k: (float(inst_pin_state[k].s_center), _pname_order(k[1])))
        for seg_id, lst in bucket.items()
    }

    comps = _hard_pair_components(hard_pairs)

    # pin -> component id
    comp_of = {}
    # component id -> { (group_name, canon_seg_id) : template_pin_name }
    comp_targets = {}
    component_conflicts = set()

    for cid, members in comps.items():
        for m in members:
            comp_of[m] = cid

        tmap = {}
        for key in members:
            inst, pname = key
            G = _find_group_of_inst(groups, inst)
            if G is None:
                continue
            if key not in inst_pin_state:
                continue
            if pname not in G.template_pins:
                continue

            seg_id = inst_pin_state[key].seg_id
            canon_seg_id = G.inst_to_canon_seg.get(inst, {}).get(seg_id)
            if canon_seg_id is None:
                continue

            tkey = (G.name, canon_seg_id)
            prev = tmap.get(tkey)
            if prev is None:
                tmap[tkey] = pname
            elif prev != pname:
                # One hard-align connected component maps to multiple template pins
                # on the same canonical segment => fundamentally conflicting.
                component_conflicts.add(tkey)

        comp_targets[cid] = tmap

    graph = {}
    nodes = {}

    for seg_id, order in base_orders.items():
        inst = seg_id[0]
        G = _find_group_of_inst(groups, inst)

        a_self = 1.0
        if G is not None and seg_id in G.inst_affine.get(inst, {}):
            a_self, _ = G.inst_affine[inst][seg_id]

        seq = [comp_of[k] for k in order if k in comp_of]
        if not seq:
            continue

        if a_self < 0:
            seq = list(reversed(seq))

        # Project this segment order to every template target touched by these components.
        per_tkey = defaultdict(list)
        for cid in seq:
            for tkey, pname in comp_targets.get(cid, {}).items():
                if not per_tkey[tkey] or per_tkey[tkey][-1] != pname:
                    per_tkey[tkey].append(pname)

        for tkey, plist in per_tkey.items():
            nodes.setdefault(tkey, set()).update(plist)
            emap = graph.setdefault(tkey, {})
            for p in plist:
                emap.setdefault(p, set())
            for i in range(len(plist) - 1):
                if plist[i] != plist[i + 1]:
                    emap[plist[i]].add(plist[i + 1])

    template_orders = {}
    cyclic_keys = set(component_conflicts)

    for tkey, pset in nodes.items():
        gname, _ = tkey
        G = groups[gname]

        def base_rank(pname):
            tvar = G.template_pins.get(pname)
            return (
                float(tvar.canon_s) if tvar is not None else 1e30,
                _pname_order(pname),
            )

        indeg = {u: 0 for u in pset}
        for u, vs in graph.get(tkey, {}).items():
            indeg.setdefault(u, 0)
            for v in vs:
                indeg[v] = indeg.get(v, 0) + 1

        ready = sorted([u for u in indeg if indeg[u] == 0], key=base_rank)
        out = []
        while ready:
            u = ready.pop(0)
            out.append(u)
            for v in sorted(graph.get(tkey, {}).get(u, []), key=base_rank):
                indeg[v] -= 1
                if indeg[v] == 0:
                    ready.append(v)
            ready.sort(key=base_rank)

        if len(out) != len(indeg):
            cyclic_keys.add(tkey)
        else:
            template_orders[tkey] = out

    return template_orders, cyclic_keys


def _find_cycle_witness_edges(node_set, edge_map, edge_sources):
    """Return one directed cycle as edge witness list, or [] if none."""
    state = {u: 0 for u in node_set}
    stack = []

    def dfs(u):
        state[u] = 1
        stack.append(u)
        for v in sorted(edge_map.get(u, []), key=lambda x: str(x)):
            if state.get(v, 0) == 0:
                cyc = dfs(v)
                if cyc:
                    return cyc
            elif state.get(v, 0) == 1:
                try:
                    i = stack.index(v)
                except ValueError:
                    i = 0
                nodes = stack[i:] + [v]
                out = []
                for a, b in zip(nodes, nodes[1:]):
                    out.append({"u": a, "v": b, "sources": list(edge_sources.get((a, b), []))})
                return out
        stack.pop()
        state[u] = 2
        return []

    for u in sorted(node_set, key=lambda x: str(x)):
        if state.get(u, 0) == 0:
            cyc = dfs(u)
            if cyc:
                return cyc
    return []


def build_orders_template_with_witness(groups, hard_pairs, inst_pin_state, real_segments):
    """Like build_orders_template(), but also returns cycle provenance.

    Returns:
      (template_orders, cyclic_keys, cycle_witnesses)

    cycle_witnesses maps canonical key (group, canon_seg_id) to a list of
    directed precedence edges on the detected cycle. Each edge contains source
    components and active hard-pairs contained in those components.
    """
    bucket = {}
    for key, st in inst_pin_state.items():
        bucket.setdefault(st.seg_id, []).append(key)

    base_orders = {
        seg_id: sorted(lst, key=lambda k: (float(inst_pin_state[k].s_center), _pname_order(k[1])))
        for seg_id, lst in bucket.items()
    }

    comps = _hard_pair_components(hard_pairs)
    comp_of = {}
    comp_targets = {}
    component_conflicts = set()
    conflict_witness = {}

    for cid, members in comps.items():
        for m in members:
            comp_of[m] = cid

        tmap = {}
        for key in members:
            inst, pname = key
            G = _find_group_of_inst(groups, inst)
            if G is None or key not in inst_pin_state or pname not in G.template_pins:
                continue
            seg_id = inst_pin_state[key].seg_id
            canon_seg_id = G.inst_to_canon_seg.get(inst, {}).get(seg_id)
            if canon_seg_id is None:
                continue
            tkey = (G.name, canon_seg_id)
            prev = tmap.get(tkey)
            if prev is None:
                tmap[tkey] = pname
            elif prev != pname:
                component_conflicts.add(tkey)
                conflict_witness.setdefault(tkey, []).append({
                    "u": prev,
                    "v": pname,
                    "sources": [{
                        "kind": "component_conflict",
                        "component_ids": [cid],
                        "hard_pairs": [],
                        "real_seg_id": None,
                        "template_key": tkey,
                    }],
                })
        comp_targets[cid] = tmap

    hp_by_comp = defaultdict(set)
    for hp in hard_pairs:
        cid = comp_of.get(hp.pin_a, comp_of.get(hp.pin_b))
        if cid is not None:
            hp_by_comp[cid].add(hp)

    graph = {}
    nodes = {}
    edge_sources = defaultdict(lambda: defaultdict(list))

    for seg_id, order in base_orders.items():
        inst = seg_id[0]
        G = _find_group_of_inst(groups, inst)
        a_self = 1.0
        if G is not None and seg_id in G.inst_affine.get(inst, {}):
            a_self, _ = G.inst_affine[inst][seg_id]

        seq = [comp_of[k] for k in order if k in comp_of]
        if not seq:
            continue
        if a_self < 0:
            seq = list(reversed(seq))

        per_tkey = defaultdict(list)
        for cid in seq:
            for tkey, pname in comp_targets.get(cid, {}).items():
                if (not per_tkey[tkey]) or per_tkey[tkey][-1][0] != pname:
                    per_tkey[tkey].append((pname, cid))

        for tkey, items in per_tkey.items():
            nodes.setdefault(tkey, set()).update(p for p, _cid in items)
            emap = graph.setdefault(tkey, {})
            esrc = edge_sources[tkey]
            for p_name, _cid in items:
                emap.setdefault(p_name, set())
            for i in range(len(items) - 1):
                p0, c0 = items[i]
                p1, c1 = items[i + 1]
                if p0 == p1:
                    continue
                emap[p0].add(p1)
                src_hps = set(hp_by_comp.get(c0, set())) | set(hp_by_comp.get(c1, set()))
                esrc[(p0, p1)].append({
                    "kind": "projected_segment_order",
                    "real_seg_id": seg_id,
                    "real_inst": inst,
                    "template_key": tkey,
                    "from_component": c0,
                    "to_component": c1,
                    "component_ids": [c0, c1],
                    "hard_pairs": list(src_hps),
                })

    template_orders = {}
    cyclic_keys = set(component_conflicts)
    cycle_witnesses = dict(conflict_witness)

    for tkey, pset in nodes.items():
        gname, _ = tkey
        G = groups[gname]

        def base_rank(pname):
            tvar = G.template_pins.get(pname)
            return (float(tvar.canon_s) if tvar is not None else 1e30, _pname_order(pname))

        indeg = {u: 0 for u in pset}
        for u, vs in graph.get(tkey, {}).items():
            indeg.setdefault(u, 0)
            for v in vs:
                indeg[v] = indeg.get(v, 0) + 1

        ready = sorted([u for u in indeg if indeg[u] == 0], key=base_rank)
        out = []
        while ready:
            u = ready.pop(0)
            out.append(u)
            for v in sorted(graph.get(tkey, {}).get(u, []), key=base_rank):
                indeg[v] -= 1
                if indeg[v] == 0:
                    ready.append(v)
            ready.sort(key=base_rank)

        if len(out) != len(indeg):
            cyclic_keys.add(tkey)
            if tkey not in cycle_witnesses:
                cycle_witnesses[tkey] = _find_cycle_witness_edges(pset, graph.get(tkey, {}), edge_sources[tkey])
        else:
            template_orders[tkey] = out

    return template_orders, cyclic_keys, cycle_witnesses


def build_orders_hard_iso_with_witness(groups, bucket_by_seg, hard_pairs, inst_pin_state, real_segments, template_hard_pairs=None):
    """Build instance-level orders and return template cycle provenance."""
    if template_hard_pairs is None:
        template_hard_pairs = hard_pairs
    template_orders, cyclic_keys, cycle_witnesses = build_orders_template_with_witness(groups, template_hard_pairs, inst_pin_state, real_segments)
    local_base = build_orders(bucket_by_seg, hard_pairs, inst_pin_state)
    orders = {}

    for seg_id, seg_pin_list in bucket_by_seg.items():
        inst = seg_id[0]
        G = _find_group_of_inst(groups, inst)
        if G is None:
            orders[seg_id] = local_base[seg_id]
            continue
        canon_seg_id = G.inst_to_canon_seg.get(inst, {}).get(seg_id)
        if canon_seg_id is None:
            orders[seg_id] = local_base[seg_id]
            continue
        tkey = (G.name, canon_seg_id)
        torder = template_orders.get(tkey)
        if not torder:
            torder = [
                pname for pname, tvar in sorted(
                    G.template_pins.items(),
                    key=lambda item: (float(item[1].canon_s), _pname_order(item[0])),
                ) if tvar.canon_seg_id == canon_seg_id
            ]
            if not torder:
                orders[seg_id] = local_base[seg_id]
                continue
        a, _ = G.inst_affine[inst][seg_id]
        if a < 0:
            torder = list(reversed(torder))
        rank = {pname: i for i, pname in enumerate(torder)}
        base_pos = {k: i for i, k in enumerate(local_base[seg_id])}
        orders[seg_id] = sorted(seg_pin_list, key=lambda k: (rank.get(k[1], 10**9), base_pos[k]))

    return orders, cyclic_keys, cycle_witnesses

def build_orders_hard_iso(groups, bucket_by_seg, hard_pairs, inst_pin_state, real_segments, template_hard_pairs=None):
    """Build instance-level orders from template orders in hard-iso mode.
    Multi-mate aware version.

    Successor hard-align pairs are geometry/equality constraints. They should
    not automatically infer canonical template precedence, because legal
    successors often connect different template pin names on the same canonical
    segment, e.g. src_* -> dst_*. Older versions used all hard_pairs here,
    which treated those successor alignments as template-pin merges and created
    artificial cyclic precedence.

    template_hard_pairs controls only template-order inference. local_base still
    sees all hard_pairs so ordinary local order can use mate anchors.
    """
    if template_hard_pairs is None:
        template_hard_pairs = hard_pairs
    template_orders, cyclic_keys = build_orders_template(groups, template_hard_pairs, inst_pin_state, real_segments)
    local_base = build_orders(bucket_by_seg, hard_pairs, inst_pin_state)
    orders = {}

    for seg_id, seg_pin_list in bucket_by_seg.items():
        inst = seg_id[0]
        G = _find_group_of_inst(groups, inst)
        if G is None:
            orders[seg_id] = local_base[seg_id]
            continue

        canon_seg_id = G.inst_to_canon_seg.get(inst, {}).get(seg_id)
        if canon_seg_id is None:
            orders[seg_id] = local_base[seg_id]
            continue

        tkey = (G.name, canon_seg_id)
        torder = template_orders.get(tkey)
        if not torder:
            torder = [
                pname
                for pname, tvar in sorted(
                    G.template_pins.items(),
                    key=lambda item: (float(item[1].canon_s), _pname_order(item[0])),
                )
                if tvar.canon_seg_id == canon_seg_id
            ]
            if not torder:
                orders[seg_id] = local_base[seg_id]
                continue

        a, _ = G.inst_affine[inst][seg_id]
        if a < 0:
            torder = list(reversed(torder))

        rank = {pname: i for i, pname in enumerate(torder)}
        base_pos = {k: i for i, k in enumerate(local_base[seg_id])}

        # Stable merge: constrained pins follow template rank;
        # unconstrained pins keep local_base relative order.
        orders[seg_id] = sorted(
            seg_pin_list,
            key=lambda k: (rank.get(k[1], 10**9), base_pos[k]),
        )

    return orders, cyclic_keys

# ---------------- inner hard solver ----------------

def solve_inner_qp_dummy(groups, inst_pin_state, orders, hard_pairs, enable_hard_iso=True,
                         keepout=0.0, real_segments=None, prev_values=None, init_values=None,
                         fixed_scalar_constraints=None):
    if cp is None:
        raise RuntimeError("CVXPY not available; cannot enforce hard constraints")

    inst_to_group = {}
    for G in groups.values():
        for inst_name in G.instances:
            inst_to_group[inst_name] = G

    def find_group(inst_name):
        return inst_to_group.get(inst_name)

    var = {}
    if enable_hard_iso:
        for G in groups.values():
            for pname in G.template_pins:
                key = (G.name, pname)
                if key not in var:
                    var[key] = cp.Variable(name=f"t_{G.name}_{pname}")
    else:
        for key in inst_pin_state:
            var[key] = cp.Variable(name=f"s_{key[0]}_{key[1]}")

    # Build instance-level affine expressions.
    real_expr = {}
    if enable_hard_iso:
        for key, st in inst_pin_state.items():
            inst, pname = key
            g = find_group(inst)
            if g is None or pname not in g.template_pins:
                continue
            tkey = (g.name, pname)
            a, b = g.inst_affine[inst][st.seg_id]
            real_expr[key] = a * var[tkey] + b
    else:
        for key in inst_pin_state:
            real_expr[key] = var[key]

    constraints = []
    eps = 1e-3
    if fixed_scalar_constraints is None:
        fixed_scalar_constraints = []

    _progress_print(f"[inner] hard_pairs: {len(list(hard_pairs))}, segments_in_orders: {len(orders)}")

    iso_constraint_scope = os.environ.get("ISO_MASTER_CONSTRAINT_SCOPE", "master").strip().lower()

    def _is_master_instance(inst_name):
        g = find_group(inst_name)
        return g is None or inst_name == g.canon_inst

    def _segment_constraints_enabled(seg_id):
        if not enable_hard_iso or iso_constraint_scope != "master":
            return True
        return _is_master_instance(seg_id[0])

    # Reference-master v12: template variables are free by default.
    # A/master initializes and drives the shared template; B/C/D follow through affine maps.
    # ISO_MASTER_FIXED_TEMPLATE=1 is debug-only and usually too strict.
    master_fixed_template = (
        enable_hard_iso
        and str(os.environ.get("ISO_MASTER_FIXED_TEMPLATE", "0")).strip().lower() not in {"0", "false", "no"}
    )
    if master_fixed_template:
        for G in groups.values():
            for pname, tvar in G.template_pins.items():
                tkey = (G.name, pname)
                if tkey in var:
                    constraints.append(var[tkey] == float(getattr(tvar, "ref_canon_s", tvar.canon_s)))

    # Fill missing bounds if any
    if real_segments is not None:
        for key, st in inst_pin_state.items():
            if (not hasattr(st, "s_min")) or (not hasattr(st, "s_max")) or (st.s_max <= st.s_min):
                seg = find_segment_by_global_id(real_segments, st.seg_id)
                width = float(getattr(st, "width", 0.0))
                st.s_min = float(seg.lo + keepout + width / 2.0)
                st.s_max = float(seg.hi - keepout - width / 2.0)
                if st.s_max < st.s_min:
                    mid = 0.5 * (st.s_min + st.s_max)
                    st.s_min = mid
                    st.s_max = mid

    # bounds + no-overlap
    for seg_id, order in orders.items():
        if not _segment_constraints_enabled(seg_id):
            continue
        seg_vars = []
        seg_mins = []
        seg_maxs = []
        seg_widths = []
        for k in order:
            if k not in real_expr:
                continue
            st = inst_pin_state[k]
            seg_vars.append(real_expr[k])
            seg_mins.append(float(st.s_min))
            seg_maxs.append(float(st.s_max))
            seg_widths.append(float(st.width))

        for v, lo, hi in zip(seg_vars, seg_mins, seg_maxs):
            constraints.append(v >= lo)
            constraints.append(v <= hi)

        for i in range(len(seg_vars) - 1):
            gap = (seg_widths[i] + seg_widths[i + 1]) / 2.0 + eps
            constraints.append(seg_vars[i + 1] - seg_vars[i] >= gap)

    # hard align equalities on instance expressions
    for hp in hard_pairs:
        if hp.pin_a in real_expr and hp.pin_b in real_expr:
            constraints.append(real_expr[hp.pin_a] == real_expr[hp.pin_b])

    # Follower-anchor closure constraints. These are fixed-scalar constraints on
    # only the movable endpoint; the follower endpoint is treated as an anchor
    # and therefore cannot pull the master/template back through an equality.
    for item in fixed_scalar_constraints:
        try:
            pin_key, target_s = item[0], float(item[1])
        except Exception:
            continue
        if pin_key in real_expr:
            constraints.append(real_expr[pin_key] == target_s)

    # Objective priority:
    #   1) Hard constraints above are non-negotiable: bounds, no-overlap/order, hard-align equality.
    #   2) Stay close to the previous accepted iterate, matching the original stable inner-QP behavior.
    #   3) Add a weak soft anchor to the initial result.json positions to reduce unnecessary drift.
    #      This is deliberately NOT a hard displacement cap.
    obj_terms = []

    if prev_values:
        keys = [k for k in var if k in prev_values]
        if keys:
            x = cp.hstack([var[k] for k in keys])
            x0 = np.asarray([float(prev_values[k]) for k in keys], dtype=float)
            obj_terms.append(cp.sum_squares(x - x0))

    move_anchor_weight = float(os.environ.get("MOVE_ANCHOR_WEIGHT", "1e-5"))
    if init_values and move_anchor_weight > 0.0:
        akeys = [k for k in real_expr if k in init_values]
        if akeys:
            avec = cp.hstack([real_expr[k] for k in akeys])
            a0 = np.asarray([float(init_values[k]) for k in akeys], dtype=float)
            obj_terms.append(move_anchor_weight * cp.sum_squares(avec - a0))

    if obj_terms:
        objective = cp.Minimize(sum(obj_terms))
    else:
        objective = cp.Minimize(0)

    prob = cp.Problem(objective, constraints)

    installed = set(cp.installed_solvers())
    gurobi_verbose_retry = str(os.environ.get("GUROBI_VERBOSE_RETRY", "0")).strip().lower() not in {"0", "false", "no"}

    preferred_solvers = []
    if "GUROBI" in installed:
        preferred_solvers.append("GUROBI")
    if not preferred_solvers:
        raise RuntimeError("GUROBI is not installed/visible to CVXPY for hard-constraint solve")

    solved = False
    last_err = None
    last_status = None

    def _acceptable(status):
        return status == cp.OPTIMAL

    for solver_name in preferred_solvers:
        try:
            # v21 is GUROBI-only. CLARABEL / OSQP / SCS fallback paths were
            # removed because the stable flow treats non-GUROBI fallback as abandoned.
            prob.solve(solver=cp.GUROBI, verbose=False, reoptimize=True, warm_start=True)
            last_status = prob.status
            _debug_print(f"[inner] solver {solver_name} status {prob.status}")
            if _acceptable(prob.status):
                solved = True
                break
        except Exception as e:
            last_err = e
            _debug_print(f"[inner] solver {solver_name} exception {repr(e)}")
            if gurobi_verbose_retry:
                try:
                    prob.solve(solver=cp.GUROBI, verbose=True, reoptimize=True, warm_start=True)
                    last_status = prob.status
                    _debug_print(f"[inner] solver GUROBI retry status {prob.status}")
                    if _acceptable(prob.status):
                        solved = True
                        break
                except Exception as e2:
                    last_err = e2
                    _debug_print(f"[inner] solver GUROBI retry exception {repr(e2)}")

    if not solved:
        msg = f"Hard-constraint solve failed. status={last_status or prob.status}"
        if last_err is not None:
            msg += f" last_err={last_err}"
        raise RuntimeError(msg)

    if enable_hard_iso:
        for G in groups.values():
            for pname, tvar in G.template_pins.items():
                val = var[(G.name, pname)].value
                if val is not None:
                    tvar.canon_s = float(val)

        # Update real states from affine mapping so greedy admission can continue on fresh geometry.
        for key, st in inst_pin_state.items():
            inst, pname = key
            g = find_group(inst)
            if g is None or pname not in g.template_pins:
                continue
            canon_s = float(g.template_pins[pname].canon_s)
            a, b = g.inst_affine[inst][st.seg_id]
            st.s_center = a * canon_s + b
    else:
        for key, st in inst_pin_state.items():
            val = var[key].value
            if val is not None:
                st.s_center = float(val)


def snapshot_template(groups):
    snap = {}
    for G in groups.values():
        for pname, tvar in G.template_pins.items():
            snap[(G.name, pname)] = float(tvar.canon_s)
    return snap


def template_diff(groups, snap):
    max_d = 0.0
    for G in groups.values():
        for pname, tvar in G.template_pins.items():
            key = (G.name, pname)
            if key in snap:
                max_d = max(max_d, abs(float(tvar.canon_s) - float(snap[key])))
    return max_d


def orders_equal(o1, o2):
    return o1 == o2


# ---------------- outer loop ----------------


def solve_global_qp_with_outer_order_update(all_modules, active_pins, active_nets, keepout,
                                            hpwl_thresh=500.0, max_outer_iter=5, tol=1e-3,
                                            enable_hard_iso=True):
    global LAST_POLICY_COMPONENT_DROPPED_EDGE_KEYS, LAST_POLICY_COMPONENT_DROPPED_RECORDS
    global LAST_POLICY_FREE_TARGET_DROPPED_EDGE_KEYS, LAST_POLICY_FREE_TARGET_DROPPED_RECORDS
    LAST_POLICY_COMPONENT_DROPPED_EDGE_KEYS = set()
    LAST_POLICY_COMPONENT_DROPPED_RECORDS = []
    LAST_POLICY_FREE_TARGET_DROPPED_EDGE_KEYS = set()
    LAST_POLICY_FREE_TARGET_DROPPED_RECORDS = []
    policy_component_dropped_edge_keys = set()
    policy_component_dropped_records = []
    policy_free_target_dropped_edge_keys = set()
    policy_free_target_dropped_records = []
    module_map = {m.name: m for m in all_modules if getattr(m, "vertex", None)}
    real_segments = {m.name: extract_real_segments(m) for m in module_map.values()}

    groups = build_iso_groups(all_modules)
    _apply_iso_master_overrides(groups)
    for G in groups.values():
        build_canonical_segment_map(G, real_segments, module_map)

    init_template_vars_from_assignment(active_pins, groups, real_segments)
    if enable_hard_iso and str(os.environ.get("ISO_MASTER_FIXED_TEMPLATE", "0")).strip().lower() not in {"0", "false", "no"}:
        fixed_cnt = sum(len(G.template_pins) for G in groups.values())
        _debug_print(f"[iso-master] reference-master mode enabled: template_pins={fixed_cnt} fixed_template={os.environ.get('ISO_MASTER_FIXED_TEMPLATE', '0')}")

    prev_orders = None
    orders = {}
    hard_pairs_active = set()
    # Model fix: successor hard-align pairs are equalities, not template-order
    # evidence by default. Set SUCCESSOR_PAIRS_AFFECT_TEMPLATE_ORDER=1 to recover
    # the old behavior.
    successor_pairs_affect_template_order = str(os.environ.get(
        "SUCCESSOR_PAIRS_AFFECT_TEMPLATE_ORDER", "1"
    )).strip().lower() not in {"0", "false", "no"}

    def _template_order_pairs():
        if not enable_hard_iso:
            return set(hard_pairs_active)
        if successor_pairs_affect_template_order:
            return set(hard_pairs_active)
        return set()

    inst_pin_state = None
    # Fixed original placement/projection anchor used by the wire-preservation objective.
    # Keys are real pins: (parent_inst, pingroup_name) -> scalar coordinate on its assigned segment.
    initial_real_anchor = {}

    family_search_cap = int(os.environ.get("FAMILY_SEARCH_CAP", "6"))
    local_fixedpoint_iters = int(os.environ.get("LOCAL_FIXEDPOINT_ITERS", "1"))
    enable_family_swap = str(os.environ.get("ENABLE_FAMILY_SWAP", "1")).strip().lower() not in {"0", "false", "no"}

    # v4 speed/stability guard:
    # Many admission/search paths retry the same infeasible active hard-pair set.
    # Cache failed trial active sets so identical combinations do not repeatedly
    # re-enter GUROBI. This does not change the mathematical model; it
    # only avoids duplicated infeasible solves.
    trial_fail_cache_enable = str(os.environ.get("TRIAL_FAIL_CACHE", "1")).strip().lower() not in {"0", "false", "no"}
    trial_fail_cache_limit = int(os.environ.get("TRIAL_FAIL_CACHE_LIMIT", "200000"))
    trial_fail_cache = {}

    # Reference-master policy: follower-side hard-pairs should not pull the master.
    # B/C/D follow A through shared template variables; only master-side successor
    # constraints are allowed to alter the template by default.
    follower_hardpairs_affect_master = str(os.environ.get("FOLLOWER_HARDPAIRS_AFFECT_MASTER", "0")).strip().lower() not in {"0", "false", "no"}

    # v15: strict pair-level HPWL gate. A family may be selected because some
    # edges are short, but every individual candidate must still be low-HPWL
    # at the moment it is about to enter admission. This prevents large-HPWL
    # edges (for example ~900 H/H2 edges) from repeatedly entering GUROBI only
    # to be rejected as infeasible.
    strict_pair_hpwl_gate = str(os.environ.get("STRICT_PAIR_HPWL_GATE", "1")).strip().lower() not in {"0", "false", "no"}
    pair_hpwl_gate_margin = float(os.environ.get("PAIR_HPWL_GATE_MARGIN", "0.0"))
    hpwl_gate_debug = str(os.environ.get("HPWL_GATE_DEBUG", "0")).strip().lower() not in {"0", "false", "no"}
    hpwl_gate_skipped_total = 0

    # Coarse progress log gate shared by nested admission/rescue/closure helpers.
    solver_progress_log = _env_bool("SOLVER_PROGRESS_LOG", True)

    # Persistent hard follower-anchor constraints accepted by required closure.
    # Each item is (pin_key, target_scalar).  These constraints are passed to
    # every subsequent QP solve so an accepted policy-blocked edge remains exact.
    required_anchor_active = []

    # Width-pressure admission heuristic is available but default-off; in the
    # latest dataset it performed worse than HPWL-first. Enable explicitly with
    # ADMISSION_WIDTH_PRIORITY=1.
    admission_width_priority = str(os.environ.get("ADMISSION_WIDTH_PRIORITY", "0")).strip().lower() not in {"0", "false", "no"}
    component_fallback_policy = os.environ.get("COMPONENT_FALLBACK_POLICY", "nearest_fixed_axis").strip().lower()
    follower_anchor_closure = str(os.environ.get("FOLLOWER_ANCHOR_CLOSURE", "1")).strip().lower() not in {"0", "false", "no"}

    # Adaptive two-stage mode.  The ordinary outer loop must stay close to the
    # old fast path.  Expensive score/replacement/rescue/target-scan logic is
    # entered only after the report-style final collector proves that real
    # same-axis blockers remain.
    adaptive_strict_closure = str(os.environ.get("ADAPTIVE_STRICT_CLOSURE", "1")).strip().lower() not in {"0", "false", "no"}
    final_rescue_count_gate = str(os.environ.get("FINAL_RESCUE_COUNT_GATE", "1")).strip().lower() not in {"0", "false", "no"}

    # Final residual rescue pass. This is deliberately generic: after the main
    # greedy admission and follower-anchor closure have produced a feasible
    # placement, re-collect the still-misaligned same-axis/pair-feasible targets
    # and try to admit them again under the current geometry. It does not use
    # any module-specific names.
    final_rescue_enable = str(os.environ.get("FINAL_RESCUE_ENABLE", "1")).strip().lower() not in {"0", "false", "no"}
    final_rescue_rounds = int(os.environ.get("FINAL_RESCUE_ROUNDS", "2"))
    final_rescue_search_cap = int(os.environ.get("FINAL_RESCUE_SEARCH_CAP", "10"))
    final_rescue_swap_cap = int(os.environ.get("FINAL_RESCUE_SWAP_CAP", "3"))
    final_rescue_removal_pool_cap = int(os.environ.get("FINAL_RESCUE_REMOVAL_POOL_CAP", "32"))
    final_rescue_beam_width = int(os.environ.get("FINAL_RESCUE_BEAM_WIDTH", "128"))
    final_rescue_global_swap = str(os.environ.get("FINAL_RESCUE_GLOBAL_SWAP", "1")).strip().lower() not in {"0", "false", "no"}
    # Score-guarded rescue: accept final rescue mutations only if the residual
    # blocker score improves. This prevents arbitrary local swaps that are
    # feasible but make the final residual distribution worse.
    final_rescue_score_accept = str(os.environ.get("FINAL_RESCUE_SCORE_ACCEPT", "1")).strip().lower() not in {"0", "false", "no"}
    final_rescue_score_mode = os.environ.get("FINAL_RESCUE_SCORE_MODE", "count_total_max").strip().lower()
    # Diagnostic-only minimum-removal search for the final blockers. This does
    # not change the solution; it explains whether a blocker is directly
    # feasible, or how many nearby active hard-pairs must be removed before it
    # becomes feasible.
    final_blocker_diagnose = str(os.environ.get("FINAL_BLOCKER_DIAGNOSE", "1")).strip().lower() not in {"0", "false", "no"}
    final_blocker_diag_limit = int(os.environ.get("FINAL_BLOCKER_DIAG_LIMIT", "50"))
    final_blocker_diag_max_remove = int(os.environ.get("FINAL_BLOCKER_DIAG_MAX_REMOVE", "3"))
    final_blocker_diag_pool_cap = int(os.environ.get("FINAL_BLOCKER_DIAG_POOL_CAP", "24"))
    final_blocker_diag_beam_width = int(os.environ.get("FINAL_BLOCKER_DIAG_BEAM_WIDTH", "128"))

    # Admission-level score/replacement: use the same true-blocker objective as
    # final rescue while admitting families, so early locally-feasible choices do
    # not block higher-value residual edges.
    admission_score_accept = str(os.environ.get("ADMISSION_SCORE_ACCEPT", "0")).strip().lower() not in {"0", "false", "no"}
    admission_replacement_enable = str(os.environ.get("ADMISSION_REPLACEMENT_ENABLE", "0")).strip().lower() not in {"0", "false", "no"}
    admission_replacement_max_remove = int(os.environ.get("ADMISSION_REPLACEMENT_MAX_REMOVE", "3"))
    admission_replacement_pool_cap = int(os.environ.get("ADMISSION_REPLACEMENT_POOL_CAP", "24"))
    admission_replacement_beam_width = int(os.environ.get("ADMISSION_REPLACEMENT_BEAM_WIDTH", "64"))

    # Required hard closure is different from score rescue: after the ordinary
    # solver converges, every actionable same-axis residual edge is treated as a
    # required equality. Allowed pairs are committed if feasible, possibly after
    # bounded local removal; reference-master-blocked pairs are converted to hard
    # follower-anchor scalar constraints. Delta is not optimized here: success
    # means exact equality within tol, failure is reported as a hard blocker.
    required_hard_closure_enable = str(os.environ.get("REQUIRED_HARD_CLOSURE_ENABLE", "1")).strip().lower() not in {"0", "false", "no"}
    required_hard_closure_rounds = int(os.environ.get("REQUIRED_HARD_CLOSURE_ROUNDS", "5"))
    required_hard_closure_max_remove = int(os.environ.get("REQUIRED_HARD_CLOSURE_MAX_REMOVE", "3"))
    required_hard_closure_pool_cap = int(os.environ.get("REQUIRED_HARD_CLOSURE_POOL_CAP", "32"))
    required_hard_closure_beam_width = int(os.environ.get("REQUIRED_HARD_CLOSURE_BEAM_WIDTH", "128"))

    # Policy-component fallback: only try policy fallback representatives whose
    # current pair HPWL is within this limit. Larger-HPWL policy edges are
    # dropped by policy and are not sent to GUROBI.
    policy_component_fallback_hpwl_limit = float(os.environ.get("POLICY_COMPONENT_FALLBACK_HPWL_LIMIT", "500"))

    # Final conflict replacement for the small remaining policy representatives.
    # After policy-component fallback has selected the HPWL-ordered representative,
    # this pass may remove nearby active hard-pairs and/or old persistent anchors
    # touching the representative endpoints, then retry the hard anchor.
    policy_min_edge_conflict_replace_enable = _env_bool("POLICY_MIN_EDGE_CONFLICT_REPLACE_ENABLE", True)
    policy_min_edge_conflict_replace_rounds = int(os.environ.get("POLICY_MIN_EDGE_CONFLICT_REPLACE_ROUNDS", "3"))
    policy_min_edge_conflict_replace_max_remove = int(os.environ.get("POLICY_MIN_EDGE_CONFLICT_REPLACE_MAX_REMOVE", "3"))
    policy_min_edge_conflict_replace_pool_cap = int(os.environ.get("POLICY_MIN_EDGE_CONFLICT_REPLACE_POOL_CAP", "32"))
    policy_min_edge_conflict_replace_beam_width = int(os.environ.get("POLICY_MIN_EDGE_CONFLICT_REPLACE_BEAM_WIDTH", "128"))
    # Final policy representative rescue knobs.  These are deliberately scoped
    # to the tiny residual policy-min-conflict pass so ordinary admission remains
    # stable and fast.
    policy_min_edge_target_scan_enable = _env_bool("POLICY_MIN_EDGE_TARGET_SCAN_ENABLE", True)
    policy_min_edge_target_scan_points = int(os.environ.get("POLICY_MIN_EDGE_TARGET_SCAN_POINTS", "17"))
    policy_min_edge_target_order_hint = _env_bool("POLICY_MIN_EDGE_TARGET_ORDER_HINT", True)
    policy_min_edge_alias_removal_enable = _env_bool("POLICY_MIN_EDGE_ALIAS_REMOVAL_ENABLE", True)
    policy_min_edge_conflict_verbose = _env_bool("POLICY_MIN_EDGE_CONFLICT_VERBOSE", False)
    policy_min_edge_conflict_detail_limit = int(os.environ.get("POLICY_MIN_EDGE_CONFLICT_DETAIL_LIMIT", "8"))

    # Master-only homology direct target closure.  For E--B where B is an
    # A/B/C/D homology target, B must not move; only E is allowed to move to
    # B's current derived/master scalar.  This avoids sending these edges into
    # the generic policy-component anchor search.
    free_to_homology_target_closure_enable = _env_bool("FREE_TO_HOMOLOGY_TARGET_CLOSURE", True)
    free_to_homology_target_trial_budget = int(os.environ.get("FREE_TO_HOMOLOGY_TARGET_TRIAL_BUDGET", "128"))
    free_to_homology_target_failed_edge_keys = set()

    def _is_master_inst_for_policy(inst_name):
        G = _find_group_of_inst(groups, inst_name)
        return G is None or inst_name == G.canon_inst

    def _is_follower_inst_for_policy(inst_name):
        G = _find_group_of_inst(groups, inst_name)
        return G is not None and inst_name != G.canon_inst

    def _multi_iso_group_of_inst(inst_name):
        """Return the reused/homology group for inst_name, ignoring singleton groups."""
        G = _find_group_of_inst(groups, inst_name)
        if G is None or len(getattr(G, "instances", []) or []) <= 1:
            return None
        return G

    def _pin_role_for_alignment(pin_key):
        """Role for master-only homology alignment semantics.

        For a reused group ABCD, only A/canon_inst is a movable master source;
        B/C/D are derived targets and never produce independent constraints.
        Singleton/non-reused modules are ordinary free sources.
        """
        inst = pin_key[0]
        G = _multi_iso_group_of_inst(inst)
        if G is None:
            return ("free", None)
        if inst == G.canon_inst:
            return ("homology_master", G.name)
        return ("homology_follower", G.name)

    def _alignment_policy_class(hp):
        """Classify a candidate edge under master-only/follower-derived semantics.

        ordinary:
            May enter as a normal equality.  This includes A--A inside the
            master instance and non-homology/free pairs.
        free_to_homology_target:
            One endpoint is a movable source, the other is an A/B/C/D homology
            target.  It must be handled as a terminal fixed-target constraint:
            the target is held fixed, and the movable endpoint is moved to it.
        covered_by_homology:
            Same reused group but not A--A.  B/C/D internal, B--C, and A--B
            observations are covered by A and are not independent requirements.
        ignored_by_policy:
            Both endpoints are derived/non-movable homology targets from
            different groups, or otherwise ambiguous under the no-reverse-pull
            rule.
        """
        ra, ga = _pin_role_for_alignment(hp.pin_a)
        rb, gb = _pin_role_for_alignment(hp.pin_b)

        # Same reused group: only master-instance A--A edges are real required
        # constraints.  Follower copies and copy-to-copy observations are not
        # independent; they are covered by the master A solution.
        if ga is not None and ga == gb:
            if ra == "homology_master" and rb == "homology_master":
                return "ordinary", None, None
            return "covered_by_homology", None, None

        # No reused group involved.
        if ga is None and gb is None:
            return "ordinary", None, None

        def is_source(role):
            return role in {"free", "homology_master"}
        def is_target(role):
            return role in {"homology_master", "homology_follower"}

        # Exactly one side is in a reused group: reused side is the target,
        # outside/free side moves.  This covers E--B and E--A.
        if ga is None and gb is not None:
            return "free_to_homology_target", hp.pin_a, hp.pin_b
        if gb is None and ga is not None:
            return "free_to_homology_target", hp.pin_b, hp.pin_a

        # Different reused groups.  Master--master edges are allowed as normal
        # master-level constraints.  If exactly one side is a derived follower,
        # move the source/master side to that follower target.  Follower--follower
        # has no safe movable side and is ignored by policy.
        if ra == "homology_master" and rb == "homology_master":
            return "ordinary", None, None
        if ra == "homology_follower" and is_source(rb):
            return "free_to_homology_target", hp.pin_b, hp.pin_a
        if rb == "homology_follower" and is_source(ra):
            return "free_to_homology_target", hp.pin_a, hp.pin_b
        return "ignored_by_policy", None, None

    def _hard_pair_allowed_by_ref_master_policy(hp):
        if not enable_hard_iso or follower_hardpairs_affect_master:
            return True
        return _alignment_policy_class(hp)[0] == "ordinary"

    def _template_var_key_for_pin(pin_key):
        """Return the optimization variable identity controlled by a real pin.

        In hard-iso mode many real pins in reused instances share the same
        template variable.  Exact real-pin matching is therefore too weak for
        conflict removal: an old anchor or hard-pair on a sibling instance may
        constrain the same underlying variable.
        """
        if enable_hard_iso and pin_key in inst_pin_state:
            inst, pname = pin_key
            G = _find_group_of_inst(groups, inst)
            if G is not None and pname in G.template_pins:
                return ("template", G.name, pname)
        return ("real", pin_key)

    def _pin_template_scalar_from_real_target(pin_key, target_s):
        """Map a real fixed scalar target to the pin's template scalar."""
        if not enable_hard_iso or pin_key not in inst_pin_state:
            return float(target_s)
        inst, pname = pin_key
        G = _find_group_of_inst(groups, inst)
        if G is None or pname not in G.template_pins:
            return float(target_s)
        st = inst_pin_state[pin_key]
        a, b = G.inst_affine[inst][st.seg_id]
        if abs(float(a)) < 1e-12:
            return float(target_s)
        return (float(target_s) - float(b)) / float(a)

    def _apply_order_hints(order_hint_constraints):
        """Move only the ordering state to target positions before a trial solve.

        The QP solve still enforces the real hard constraints.  This hint only
        lets order/no-overlap constraints be rebuilt around the candidate target
        instead of around the previous, still-misaligned coordinates.  Without
        this, a feasible target can be rejected solely because it would require a
        local legal reordering on the affected segment.
        """
        if not order_hint_constraints:
            return
        for pin_key, target_s in _dedup_fixed_constraints(order_hint_constraints) or []:
            if pin_key not in inst_pin_state:
                continue
            if enable_hard_iso:
                inst, pname = pin_key
                G = _find_group_of_inst(groups, inst)
                if G is not None and pname in G.template_pins:
                    G.template_pins[pname].canon_s = float(_pin_template_scalar_from_real_target(pin_key, target_s))
                    continue
            inst_pin_state[pin_key].s_center = float(target_s)

    def _hp_sig(hp):
        a, b = hp.as_undirected_key()
        return (a[0], a[1], b[0], b[1], getattr(hp, "axis", ""))

    def _active_set_sig(edge_set):
        return tuple(sorted(_hp_sig(hp) for hp in edge_set))

    def build_inst_state_from_active_pins(active_pins_list):
        state = {}
        bucket = {}
        for pin in active_pins_list:
            inst = pin.parent_inst
            pname = pin.pingroup_name
            if not hasattr(pin, "seg_id") or not hasattr(pin, "s_init"):
                continue
            seg_id = pin.seg_id
            seg = find_segment_by_id(real_segments.get(inst, []), seg_id)
            width = float(getattr(pin, "width", 0.0))
            s_center = float(pin.s_init)
            s_min = float(seg.lo + keepout + width / 2.0)
            s_max = float(seg.hi - keepout - width / 2.0)
            if s_max < s_min:
                mid = 0.5 * (s_min + s_max)
                s_min = mid
                s_max = mid
            s_center = max(min(s_center, s_max), s_min)

            st = RealPinState()
            st.inst_name = inst
            st.pin_name = pname
            st.seg_id = seg_id
            st.width = width
            st.s_center = s_center
            st.s_init = s_center
            st.s_min = s_min
            st.s_max = s_max

            key = (inst, pname)
            state[key] = st
            bucket.setdefault(seg_id, []).append(key)
        return state, bucket

    def snapshot_inst_state(state):
        return {k: float(v.s_center) for k, v in state.items()}

    def restore_inst_state(state, snap):
        for k, s in snap.items():
            if k in state:
                state[k].s_center = float(s)

    def inst_state_diff(state, snap):
        md = 0.0
        for k, v in state.items():
            if k in snap:
                md = max(md, abs(float(v.s_center) - float(snap[k])))
        return md

    def restore_template(groups, snap):
        for G in groups.values():
            for pname, tvar in G.template_pins.items():
                key = (G.name, pname)
                if key in snap:
                    tvar.canon_s = float(snap[key])

    def _interval_of(pin_key):
        st = inst_pin_state.get(pin_key)
        if st is None:
            return None
        return float(st.s_min), float(st.s_max)

    def _compute_hpwl(hp):
        if hp.pin_a not in inst_pin_state or hp.pin_b not in inst_pin_state:
            return float("inf")
        st1, st2 = inst_pin_state[hp.pin_a], inst_pin_state[hp.pin_b]
        seg1 = find_segment_by_global_id(real_segments, st1.seg_id)
        seg2 = find_segment_by_global_id(real_segments, st2.seg_id)
        if seg1.free_axis == 'x':
            return abs(st1.s_center - st2.s_center) + abs(seg1.fixed_coord - seg2.fixed_coord)
        return abs(seg1.fixed_coord - seg2.fixed_coord) + abs(st1.s_center - st2.s_center)

    def _alignment_delta(hp):
        if hp.pin_a not in inst_pin_state or hp.pin_b not in inst_pin_state:
            return float("inf")
        st1, st2 = inst_pin_state[hp.pin_a], inst_pin_state[hp.pin_b]
        return abs(float(st1.s_center) - float(st2.s_center))

    def _edge_aligned_after_solve(hp, check_tol=None):
        """Return True only when the target edge is actually aligned now.

        A trial anchor/hard-pair is not considered accepted merely because the
        QP is feasible.  It must also make the target edge's free-axis scalar
        delta vanish within tolerance; otherwise the caller must roll the trial
        back.  This prevents misleading logs such as kept=1 while residual_after
        is unchanged.
        """
        t = tol if check_tol is None else float(check_tol)
        return _alignment_delta(hp) <= t + 1e-7

    def _post_delta(hp):
        return float(_alignment_delta(hp))

    def _fixed_axis_distance(hp):
        """Distance on the non-free axis of a same-free-axis hard-align pair.

        For x-free horizontal pins this is vertical distance; for y-free vertical
        pins this is horizontal distance. Component fallback uses this to keep
        the physically nearest edge when a star-like component has no common
        scalar interval.
        """
        if hp.pin_a not in inst_pin_state or hp.pin_b not in inst_pin_state:
            return float("inf")
        st1, st2 = inst_pin_state[hp.pin_a], inst_pin_state[hp.pin_b]
        seg1 = find_segment_by_global_id(real_segments, st1.seg_id)
        seg2 = find_segment_by_global_id(real_segments, st2.seg_id)
        return abs(float(seg1.fixed_coord) - float(seg2.fixed_coord))

    def _follower_anchor_constraint_from_pair(hp):
        """Return a role-aware target anchor for non-ordinary homology edges.

        Under master-only homology semantics, an edge such as E--B does not move
        B.  B is a derived target, and E is moved to B.  The returned legacy tuple
        is used only for counting/compatibility; the actual hard constraints are
        generated by _policy_target_constraint_sets_from_pair().
        """
        if not enable_hard_iso or not follower_anchor_closure:
            return None
        if hp.pin_a not in inst_pin_state or hp.pin_b not in inst_pin_state:
            return None
        cls, movable, target_pin = _alignment_policy_class(hp)
        if cls != "free_to_homology_target" or movable is None or target_pin is None:
            return None
        mst = inst_pin_state[movable]
        tst = inst_pin_state[target_pin]
        mseg = find_segment_by_global_id(real_segments, mst.seg_id)
        tseg = find_segment_by_global_id(real_segments, tst.seg_id)
        if mseg.free_axis != tseg.free_axis or mseg.free_axis != hp.axis:
            return None
        target = float(tst.s_center)
        if target < float(mst.s_min) - 1e-9 or target > float(mst.s_max) + 1e-9:
            return None
        return (movable, target, hp)

    def _dedup_fixed_constraints(items):
        """Deduplicate fixed scalar constraints, rejecting same-pin conflicts."""
        by_pin = {}
        out = []
        for pin_key, target in items or []:
            target = float(target)
            if pin_key in by_pin:
                if abs(by_pin[pin_key] - target) > 1e-6:
                    return None
                continue
            by_pin[pin_key] = target
            out.append((pin_key, target))
        return out

    def _policy_target_constraint_sets_from_pair(hp, include_scan=False):
        """Return role-aware fixed-scalar target constraints for policy edges.

        Master-only homology semantics:
          * A/B/C/D same homology group: only A--A is ordinary; B/C/D copies are
            covered by A and are not anchored here.
          * E--B (or E--A): B/A is held at its current derived/master scalar and
            E is moved to that scalar.  The target side is also fixed to its
            current value so this trial cannot move the homology group.
          * follower--follower ambiguous edges are ignored by policy.
        """
        if not (enable_hard_iso and follower_anchor_closure):
            return []
        if hp.pin_a not in inst_pin_state or hp.pin_b not in inst_pin_state:
            return []
        sta = inst_pin_state[hp.pin_a]
        stb = inst_pin_state[hp.pin_b]
        sega = find_segment_by_global_id(real_segments, sta.seg_id)
        segb = find_segment_by_global_id(real_segments, stb.seg_id)
        if sega.free_axis != segb.free_axis or sega.free_axis != hp.axis:
            return []

        cls, movable, target_pin = _alignment_policy_class(hp)
        if cls != "free_to_homology_target" or movable is None or target_pin is None:
            return []

        mst = inst_pin_state[movable]
        tst = inst_pin_state[target_pin]
        target = float(tst.s_center)
        if target < float(mst.s_min) - 1e-9 or target > float(mst.s_max) + 1e-9:
            return []

        # Freeze the homology target at its current scalar and move the external
        # source to that scalar.  This enforces E->B without letting E drag B/A.
        cons = _dedup_fixed_constraints([(target_pin, target), (movable, target)])
        return [cons] if cons else []

    def _follower_anchor_fixed_constraints_from_pair(hp):
        """Return the first policy-target constraint set for compatibility."""
        sets = _policy_target_constraint_sets_from_pair(hp)
        return sets[0] if sets else None

    def _pin_width(pin_key):
        st = inst_pin_state.get(pin_key)
        return float(getattr(st, "width", 0.0)) if st is not None else 0.0

    def _edge_width(hp):
        return max(_pin_width(hp.pin_a), _pin_width(hp.pin_b))

    def _edge_interval_slack(hp):
        iva = _interval_of(hp.pin_a)
        ivb = _interval_of(hp.pin_b)
        if iva is None or ivb is None:
            return float("inf")
        lo = max(float(iva[0]), float(ivb[0]))
        hi = min(float(iva[1]), float(ivb[1]))
        return max(0.0, hi - lo)

    def _edge_width_pressure(hp):
        # Larger means more critical. Slack is floored to avoid exploding on
        # near-zero intervals while still ranking them first.
        width = _edge_width(hp)
        slack = max(_edge_interval_slack(hp), 1e-6)
        return width / slack

    def _admission_priority(hp):
        """Sort key for candidate admission. Lower is better.

        With ADMISSION_WIDTH_PRIORITY=1:
          1) larger pin width first;
          2) smaller common interval slack first;
          3) smaller current HPWL first;
          4) deterministic pin name tie-break.

        With ADMISSION_WIDTH_PRIORITY=0, preserve the old HPWL-first behavior.
        """
        hpwl = _compute_hpwl(hp)
        if not admission_width_priority:
            return (hpwl, str(hp.as_undirected_key()))
        return (
            -_edge_width(hp),
            _edge_interval_slack(hp),
            hpwl,
            str(hp.as_undirected_key()),
        )

    def _subset_admission_priority(edges):
        edges = list(edges)
        if not admission_width_priority:
            return (
                sum(_compute_hpwl(e) for e in edges),
                ";".join(str(e.as_undirected_key()) for e in sorted(edges, key=lambda x: str(x.as_undirected_key()))),
            )
        return (
            -sum(_edge_width(e) for e in edges),
            sum(_edge_interval_slack(e) for e in edges),
            sum(_compute_hpwl(e) for e in edges),
            ";".join(str(e.as_undirected_key()) for e in sorted(edges, key=lambda x: str(x.as_undirected_key()))),
        )

    def _component_fallback_priority(hp):
        """Sort key for choosing one edge when a hard-align component has no
        common interval.

        Default policy is nearest_fixed_axis: for a star-like component such as
        GH connected to GH1/GH2/GH3/GH4, keep the GH edge whose non-free-axis
        distance is shortest. This preserves the physically nearest alignment
        instead of forcing all leaves to a single impossible scalar coordinate.
        """
        hpwl = _compute_hpwl(hp)
        policy = component_fallback_policy
        if policy in {"nearest", "nearest_fixed", "nearest_fixed_axis", "fixed_axis"}:
            return (_fixed_axis_distance(hp), hpwl, str(hp.as_undirected_key()))
        if policy in {"hpwl", "shortest", "shortest_hpwl"}:
            return (hpwl, str(hp.as_undirected_key()))
        if policy in {"width", "wide"}:
            return (-_edge_width(hp), hpwl, str(hp.as_undirected_key()))
        # Optional legacy width-pressure policy.
        return (
            -_edge_width_pressure(hp),
            -_edge_width(hp),
            _edge_interval_slack(hp),
            hpwl,
            str(hp.as_undirected_key()),
        )

    def _components_from_pairs(pairs_list):
        parent = {}
        def find(x):
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(a, b):
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[rb] = ra
        for hp in pairs_list:
            union(hp.pin_a, hp.pin_b)
        comps = {}
        for x in parent:
            r = find(x)
            comps.setdefault(r, []).append(x)
        return list(comps.values())

    def _family_key(hp):
        ia, ib = hp.pin_a[0], hp.pin_b[0]
        if ia <= ib:
            return (ia, ib, hp.axis)
        return (ib, ia, hp.axis)

    def _refresh_bucket_from_state():
        nonlocal inst_pin_state, bucket_by_seg
        if enable_hard_iso:
            inst_pin_state, bucket_by_seg = expand_template_to_instances(groups, real_segments, keepout)
        else:
            bucket_by_seg = {}
            for key, st in inst_pin_state.items():
                bucket_by_seg.setdefault(st.seg_id, []).append(key)
        return inst_pin_state, bucket_by_seg

    def _rebuild_orders_current():
        if enable_hard_iso:
            orders_try, cyclic_keys_try = build_orders_hard_iso(groups, bucket_by_seg, hard_pairs_active, inst_pin_state, real_segments, _template_order_pairs())
            if cyclic_keys_try:
                raise RuntimeError(f"cyclic template precedence: {sorted(cyclic_keys_try)}")
            return orders_try
        return build_orders(bucket_by_seg, hard_pairs_active, inst_pin_state)

    def _is_cacheable_trial_error(err):
        """Only cache deterministic infeasibility/cycle failures.

        GUROBI SolverError with status=None is an uncertain solver failure, not a
        proven mathematical infeasibility. Caching those failures can wrongly
        reject later attempts that happen to use the same active hard-pair set.
        """
        s = str(err)
        sl = s.lower()
        if "cyclic template precedence" in sl:
            return True
        if "empty interval" in sl or "interval intersection" in sl:
            return True
        if "status=infeasible" in sl:
            return True
        if "status=infeasible_or_unbounded" in sl:
            return True
        if "status=none" in sl:
            return False
        if "solver 'gurobi' failed" in sl or "solvererror" in sl:
            return False
        return False

    def _attempt_replace(replace_edges, subset_edges):
        nonlocal hard_pairs_active, inst_pin_state, bucket_by_seg, orders
        active_backup = set(hard_pairs_active)
        inst_backup = snapshot_inst_state(inst_pin_state)
        tmpl_backup = snapshot_template(groups)
        bucket_backup = {seg: list(lst) for seg, lst in bucket_by_seg.items()}
        orders_backup = {seg: list(lst) for seg, lst in orders.items()}

        trial_active = (set(hard_pairs_active) - set(replace_edges)) | set(subset_edges)
        trial_sig = _active_set_sig(trial_active)
        if trial_fail_cache_enable and trial_sig in trial_fail_cache:
            return False, RuntimeError(f"cached deterministic infeasible trial: {trial_fail_cache[trial_sig]}")

        try:
            hard_pairs_active = trial_active
            _refresh_bucket_from_state()

            last_orders = None
            for _ in range(max(1, local_fixedpoint_iters)):
                orders_try = _rebuild_orders_current()
                prev_values_try = snapshot_template(groups) if enable_hard_iso else snapshot_inst_state(inst_pin_state)
                solve_inner_qp_dummy(groups, inst_pin_state, orders_try, hard_pairs_active,
                                     enable_hard_iso=enable_hard_iso, keepout=keepout,
                                     real_segments=real_segments, prev_values=prev_values_try,
                                     init_values=initial_real_anchor,
                                     fixed_scalar_constraints=required_anchor_active)
                _refresh_bucket_from_state()
                if last_orders is not None and orders_equal(last_orders, orders_try):
                    orders = orders_try
                    return True, None
                last_orders = orders_try

            orders = _rebuild_orders_current()
            return True, None
        except Exception as e:
            if trial_fail_cache_enable and _is_cacheable_trial_error(e):
                if len(trial_fail_cache) >= trial_fail_cache_limit:
                    # Drop an arbitrary oldest-ish entry. Dict insertion order is
                    # deterministic in modern Python; this keeps memory bounded.
                    try:
                        trial_fail_cache.pop(next(iter(trial_fail_cache)))
                    except StopIteration:
                        pass
                trial_fail_cache[trial_sig] = str(e)
            hard_pairs_active = active_backup
            restore_inst_state(inst_pin_state, inst_backup)
            restore_template(groups, tmpl_backup)
            bucket_by_seg = bucket_backup
            orders = orders_backup
            return False, e

    def _build_triggered_pairs(apply_component_gate=True, force_keep_all_components=False):
        newly_triggered = build_hard_align_pairs(active_nets, inst_pin_state, real_segments, hpwl_thresh)

        if enable_hard_iso and not follower_hardpairs_affect_master:
            before_policy = len(newly_triggered)
            allowed = []
            follower_anchorable = 0
            blocked = 0
            for hp in newly_triggered:
                if _hard_pair_allowed_by_ref_master_policy(hp):
                    allowed.append(hp)
                else:
                    # Do not admit it as an ordinary equality because that would
                    # pull the master/template. It may still be handled after the
                    # main solve by follower-anchor closure if exactly one endpoint
                    # is a follower and the other endpoint can move to the follower
                    # scalar coordinate.
                    if _follower_anchor_constraint_from_pair(hp) is not None:
                        follower_anchorable += 1
                    else:
                        blocked += 1
            newly_triggered = allowed
            dropped_policy = before_policy - len(newly_triggered)
            if dropped_policy:
                _debug_print(
                    f"[iso-master] held follower-side hard_pairs out of ordinary admission: "
                    f"{dropped_policy} anchorable_now={follower_anchorable} blocked_now={blocked}"
                )

        feasible_triggered = []
        skipped = 0
        skipped_samples = []
        for hp in newly_triggered:
            if hp.pin_a not in inst_pin_state or hp.pin_b not in inst_pin_state:
                continue
            a = inst_pin_state[hp.pin_a]
            b = inst_pin_state[hp.pin_b]
            lo = max(float(a.s_min), float(b.s_min))
            hi = min(float(a.s_max), float(b.s_max))
            if lo <= hi + 1e-9:
                feasible_triggered.append(hp)
            else:
                skipped += 1
                if len(skipped_samples) < 10:
                    skipped_samples.append((hp.pin_a, hp.pin_b, lo, hi))
        if skipped and apply_component_gate:
            _debug_print(f"[outer] skipped infeasible newly_triggered hard_pairs: {skipped} (empty interval intersection)")
            for a, b, lo, hi in skipped_samples:
                _debug_print(f"  skip {a} <-> {b} lo={lo:.3f} hi={hi:.3f}")

        newly_triggered = feasible_triggered

        if newly_triggered and apply_component_gate:
            comps = _components_from_pairs(newly_triggered)
            pin_to_comp = {}
            for idx, nodes in enumerate(comps):
                for n in nodes:
                    pin_to_comp[n] = idx
            comp_edges = [[] for _ in range(len(comps))]
            for hp in newly_triggered:
                ci = pin_to_comp.get(hp.pin_a)
                if ci is not None:
                    comp_edges[ci].append(hp)

            kept = []
            dropped_components = 0
            for nodes, edges in zip(comps, comp_edges):
                if not edges:
                    continue
                lo = -1e30
                hi = 1e30
                for n in nodes:
                    iv = _interval_of(n)
                    if iv is None:
                        continue
                    lo = max(lo, float(iv[0]))
                    hi = min(hi, float(iv[1]))
                if lo <= hi + 1e-9 or force_keep_all_components:
                    kept.extend(edges)
                else:
                    dropped_components += 1
                    best = min(edges, key=_component_fallback_priority)
                    kept.append(best)
            if dropped_components:
                _debug_print(
                    f"[outer] component-level infeasible hard-align components: {dropped_components}; "
                    f"kept one edge per component by policy={component_fallback_policy}"
                )
            newly_triggered = kept
        return newly_triggered

    def _hpwl_gate_allows_pair(hp):
        if not strict_pair_hpwl_gate:
            return True
        try:
            return _compute_hpwl(hp) < (hpwl_thresh + pair_hpwl_gate_margin)
        except Exception:
            # If HPWL cannot be evaluated, do not gate it here; let the existing
            # interval/QP checks handle the edge.
            return True

    def _run_family_admission(candidates, search_cap, allow_family_swap,
                              swap_cap=1, removal_pool_cap=8, beam_width=24):
        """Admit a maximum-ish feasible subset of ordinary low-HPWL hard-align pairs.

        This is the only hard-pair admission path in v21.
        """
        nonlocal hpwl_gate_skipped_total
        admitted = 0
        rejected = 0
        if not candidates:
            return admitted, rejected

        # Pair-level current-HPWL gate. This is deliberately inside admission,
        # not only inside build_hard_align_pairs(), because earlier accepted
        # families can move pins before later families are processed.
        if strict_pair_hpwl_gate:
            gated_candidates = []
            skipped_samples = []
            skipped_count = 0
            for hp in candidates:
                try:
                    ha = _compute_hpwl(hp)
                except Exception:
                    ha = None
                if ha is not None and ha >= (hpwl_thresh + pair_hpwl_gate_margin):
                    skipped_count += 1
                    if len(skipped_samples) < 10:
                        skipped_samples.append((hp, ha))
                    continue
                gated_candidates.append(hp)
            if skipped_count:
                hpwl_gate_skipped_total += skipped_count
                if hpwl_gate_debug:
                    _debug_print(f"[outer] pair-hpwl gate skipped={skipped_count} threshold={hpwl_thresh:g} margin={pair_hpwl_gate_margin:g}")
                    for hp, ha in skipped_samples:
                        _debug_print(f"[outer] skip pair by hpwl gate: hpwl={ha:.3f} a={hp.pin_a} b={hp.pin_b}")
            candidates = gated_candidates
            if not candidates:
                return admitted, rejected

        family_to_candidates = defaultdict(list)
        for hp in candidates:
            family_to_candidates[_family_key(hp)].append(hp)

        family_order = sorted(
            family_to_candidates.items(),
            key=lambda kv: (min(_admission_priority(hp) for hp in kv[1]), len(kv[1]))
        )

        total_candidates = len(candidates)
        for fam_key, fam_candidates in family_order:
            family_active = [hp for hp in hard_pairs_active if _family_key(hp) == fam_key]
            union_edges = []
            seen_union = set()
            for hp in sorted(family_active + fam_candidates, key=_admission_priority):
                if hp not in seen_union:
                    seen_union.add(hp)
                    union_edges.append(hp)

            base_active_set = set(family_active)
            chosen_subset = None
            chosen_err = None
            skipped_by_dynamic_hpwl = set()

            if len(union_edges) <= search_cap:
                for k in range(len(union_edges), len(base_active_set) - 1, -1):
                    combos = list(combinations(union_edges, k))
                    combos.sort(key=lambda combo: (
                        # Prefer more admitted critical width-pressure edges when
                        # several subsets have the same cardinality.
                        _subset_admission_priority(combo),
                        -sum(1 for hp in combo if hp in base_active_set),
                    ))
                    found = False
                    for combo in combos:
                        combo_set = set(combo)
                        ok, err = _attempt_replace_admission(base_active_set, combo_set)
                        if ok:
                            chosen_subset = combo_set
                            chosen_err = None
                            found = True
                            break
                        if chosen_err is None:
                            chosen_err = err
                    if found:
                        break
            else:
                current_family_set = set(family_active)
                ordered_candidates = sorted(fam_candidates, key=_admission_priority)

                for hp in ordered_candidates:
                    if hp in current_family_set:
                        continue

                    # Dynamic pair-level HPWL gate. Successful earlier insertions
                    # in the same family can move pins; then later candidates may
                    # no longer be low-HPWL targets.
                    if strict_pair_hpwl_gate:
                        try:
                            ha_now = _compute_hpwl(hp)
                        except Exception:
                            ha_now = None
                        if ha_now is not None and ha_now >= (hpwl_thresh + pair_hpwl_gate_margin):
                            hpwl_gate_skipped_total += 1
                            skipped_by_dynamic_hpwl.add(hp)
                            if hpwl_gate_debug:
                                _debug_print(
                                    f"[outer] skip pair by dynamic hpwl gate: "
                                    f"hpwl={ha_now:.3f} threshold={hpwl_thresh:g} "
                                    f"family={fam_key[:2]} axis={fam_key[2]} a={hp.pin_a} b={hp.pin_b}"
                                )
                            continue

                    ok, err = _attempt_replace_admission(current_family_set, current_family_set | {hp})
                    if ok:
                        current_family_set = {e for e in hard_pairs_active if _family_key(e) == fam_key}
                        continue

                    if allow_family_swap and current_family_set:
                        swapped = False
                        if admission_width_priority:
                            # Try removing low-criticality existing family edges first
                            # to make room for a high-pressure new edge.
                            removal_pool = sorted(
                                current_family_set,
                                key=lambda old_hp: (
                                    _edge_width_pressure(old_hp),
                                    _edge_width(old_hp),
                                    -_compute_hpwl(old_hp),
                                    str(old_hp.as_undirected_key()),
                                )
                            )[:max(1, removal_pool_cap)]
                        else:
                            removal_pool = sorted(
                                current_family_set,
                                key=lambda old_hp: (
                                    -_compute_hpwl(old_hp),
                                    -_alignment_delta(old_hp),
                                )
                            )[:max(1, removal_pool_cap)]

                        trial_removals = []
                        for r in range(1, max(1, swap_cap) + 1):
                            for rem in combinations(removal_pool, r):
                                rem_set = set(rem)
                                if admission_width_priority:
                                    score = (
                                        sum(_edge_width_pressure(e) for e in rem_set),
                                        sum(_edge_width(e) for e in rem_set),
                                        -sum(_compute_hpwl(e) for e in rem_set),
                                    )
                                    # lower score means less critical; try removing it first
                                    trial_removals.append((score, rem_set))
                                else:
                                    score = (
                                        sum(_compute_hpwl(e) for e in rem_set),
                                        sum(_alignment_delta(e) for e in rem_set),
                                    )
                                    trial_removals.append((score, rem_set))
                        trial_removals.sort(key=lambda item: item[0], reverse=(not admission_width_priority))
                        for _, rem_set in trial_removals[:max(1, beam_width)]:
                            trial_set = (current_family_set - rem_set) | {hp}
                            ok2, err2 = _attempt_replace_admission(current_family_set, trial_set)
                            if ok2:
                                current_family_set = {e for e in hard_pairs_active if _family_key(e) == fam_key}
                                swapped = True
                                break
                            if chosen_err is None:
                                chosen_err = err2
                        if (not swapped) and admission_replacement_enable:
                            pool = _candidate_local_removal_pool(hp, admission_replacement_pool_cap)
                            for rem_set in _bounded_removal_combos(pool, admission_replacement_max_remove, admission_replacement_beam_width):
                                ok3, err3 = _attempt_replace_admission(rem_set, {hp})
                                if ok3:
                                    current_family_set = {e for e in hard_pairs_active if _family_key(e) == fam_key}
                                    swapped = True
                                    break
                                if chosen_err is None:
                                    chosen_err = err3
                        if not swapped and chosen_err is None:
                            chosen_err = err
                    else:
                        if admission_replacement_enable:
                            pool = _candidate_local_removal_pool(hp, admission_replacement_pool_cap)
                            replaced = False
                            for rem_set in _bounded_removal_combos(pool, admission_replacement_max_remove, admission_replacement_beam_width):
                                ok3, err3 = _attempt_replace_admission(rem_set, {hp})
                                if ok3:
                                    current_family_set = {e for e in hard_pairs_active if _family_key(e) == fam_key}
                                    replaced = True
                                    break
                                if chosen_err is None:
                                    chosen_err = err3
                            if not replaced and chosen_err is None:
                                chosen_err = err
                        elif chosen_err is None:
                            chosen_err = err

                chosen_subset = {e for e in hard_pairs_active if _family_key(e) == fam_key}

            if chosen_subset is None:
                chosen_subset = base_active_set

            newly_kept = [hp for hp in fam_candidates if hp in chosen_subset and hp not in base_active_set]
            newly_skipped_hpwl = [hp for hp in fam_candidates if hp in skipped_by_dynamic_hpwl and hp not in chosen_subset]
            newly_rejected = [hp for hp in fam_candidates if hp not in chosen_subset and hp not in skipped_by_dynamic_hpwl]
            admitted += len(newly_kept)
            rejected += len(newly_rejected)

            if newly_kept or newly_rejected or newly_skipped_hpwl:
                skip_extra = f" skipped_hpwl={len(newly_skipped_hpwl)}" if newly_skipped_hpwl else ""
                _debug_print(f"[outer] family {fam_key[:2]} axis={fam_key[2]} -> kept={len(newly_kept)} rejected={len(newly_rejected)}{skip_extra} active_now={len(chosen_subset)}")
            if newly_rejected:
                for hp in newly_rejected[:10]:
                    try:
                        ha = _compute_hpwl(hp)
                        iva = _interval_of(hp.pin_a)
                        ivb = _interval_of(hp.pin_b)
                        _debug_print(f"[outer] reject pair due to infeasible inner solve: "
                              f"hpwl={ha:.3f} a={hp.pin_a} b={hp.pin_b} a_iv={iva} b_iv={ivb} err={chosen_err}")
                    except Exception:
                        pass

        if admitted or rejected:
            _debug_print(f"[outer] family admission: admitted={admitted} rejected={rejected} from candidates={total_candidates}")
        return admitted, rejected

    def _collect_true_same_axis_blockers_report_style(include_active=True):
        """Collect the same actionable blockers used by the final report.

        This is the canonical residual collector used by admission scoring,
        final rescue, and blocker diagnosis. It intentionally matches the run
        script's report semantics:
          * HPWL must be below threshold;
          * endpoints must have the same free_axis;
          * pair legal intervals must intersect;
          * component-empty groups keep one edge by the configured fallback
            policy and drop the rest from the ordinary blocker count.
        """
        _refresh_bucket_from_state()
        records = []
        seen = set()
        for net in active_nets:
            for p in getattr(net, 'pins', []) or []:
                key1 = (p.parent_inst, p.pingroup_name)
                if key1 not in inst_pin_state:
                    continue
                for succ_full in getattr(p, 'successors', []) or []:
                    key2 = _parse_successor_full_name(succ_full)
                    if key2 is None or key2 not in inst_pin_state:
                        continue
                    edge_key = (key1, key2) if key1 <= key2 else (key2, key1)
                    if edge_key in seen:
                        continue
                    seen.add(edge_key)
                    st1, st2 = inst_pin_state[key1], inst_pin_state[key2]
                    seg1 = find_segment_by_global_id(real_segments, st1.seg_id)
                    seg2 = find_segment_by_global_id(real_segments, st2.seg_id)
                    if seg1.free_axis not in {'x', 'y'} or seg1.free_axis != seg2.free_axis:
                        continue
                    axis = seg1.free_axis
                    hpwl = (abs(float(st1.s_center) - float(st2.s_center)) +
                            abs(float(seg1.fixed_coord) - float(seg2.fixed_coord)))
                    if hpwl >= (hpwl_thresh + pair_hpwl_gate_margin):
                        continue
                    lo = max(float(st1.s_min), float(st2.s_min))
                    hi = min(float(st1.s_max), float(st2.s_max))
                    if lo > hi + 1e-9:
                        continue
                    hp = HardAlignPair(key1, key2, axis)
                    cls, movable, target_pin = _alignment_policy_class(hp)
                    if enable_hard_iso and cls in {"covered_by_homology", "ignored_by_policy"}:
                        # B/C/D copies and copy-to-copy observations are not
                        # independent required alignments. They are governed by
                        # the master A template and must not become residual
                        # blockers.
                        continue
                    delta = _alignment_delta(hp)
                    fixed_axis_dist = _fixed_axis_distance(hp)
                    records.append({
                        'policy_class': cls,
                        'hp': hp,
                        'edge_key': hp.as_undirected_key(),
                        'delta': float(delta),
                        'hpwl': float(hpwl),
                        'fixed_axis_distance': float(fixed_axis_dist),
                        'slack': float(hi - lo),
                        'lo': float(lo),
                        'hi': float(hi),
                    })

        # Component-empty fallback matching solver policy. Components are built
        # over all same-axis pair-feasible low-HPWL records, then non-kept edges
        # from empty components are excluded from actionable residual blockers.
        parent = {}
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
        for rec in records:
            hp = rec['hp']
            union(hp.pin_a, hp.pin_b)
            node_iv[hp.pin_a] = _interval_of(hp.pin_a)
            node_iv[hp.pin_b] = _interval_of(hp.pin_b)
        comps = {}
        for n in list(parent.keys()):
            comps.setdefault(find(n), []).append(n)
        dropped_component_edges = set()
        for nodes in comps.values():
            if len(nodes) <= 2:
                continue
            ivs = [node_iv[n] for n in nodes if n in node_iv and node_iv[n] is not None]
            if not ivs:
                continue
            lo = max(float(iv[0]) for iv in ivs)
            hi = min(float(iv[1]) for iv in ivs)
            if lo <= hi + 1e-9:
                continue
            nset = set(nodes)
            comp_recs = [rec for rec in records if rec['hp'].pin_a in nset and rec['hp'].pin_b in nset]
            if not comp_recs:
                continue
            kept = min(comp_recs, key=lambda rec: _component_fallback_priority(rec['hp']))
            kept_key = kept['edge_key']
            for rec in comp_recs:
                if rec['edge_key'] != kept_key:
                    dropped_component_edges.add(rec['edge_key'])

        blockers = []
        for rec in records:
            hp = rec['hp']
            if rec['delta'] <= tol:
                continue
            if rec['edge_key'] in dropped_component_edges:
                continue
            if rec['edge_key'] in policy_component_dropped_edge_keys:
                continue
            if rec['edge_key'] in policy_free_target_dropped_edge_keys:
                continue
            if (not include_active) and hp in hard_pairs_active:
                continue
            blockers.append(rec)
        blockers.sort(key=lambda rec: (
            -rec['delta'],
            rec['fixed_axis_distance'],
            -_edge_width(rec['hp']),
            rec['slack'],
            rec['hpwl'],
            str(rec['edge_key']),
        ))
        return blockers

    def _collect_final_rescue_candidates():
        """Collect ordinary residual hard-align candidates under report semantics."""
        blockers = _collect_true_same_axis_blockers_report_style(include_active=False)
        out = []
        seen = set()
        for rec in blockers:
            hp = rec['hp']
            if hp.as_undirected_key() in seen:
                continue
            seen.add(hp.as_undirected_key())
            if hp in hard_pairs_active:
                continue
            if not _hpwl_gate_allows_pair(hp):
                continue
            # Ordinary admission must respect reference-master policy; follower-
            # side pairs are left for follower-anchor closure and diagnosis.
            if not _hard_pair_allowed_by_ref_master_policy(hp):
                continue
            out.append(hp)
        return out

    def _residual_score():
        """Canonical residual score over the same blockers used by final report."""
        residual = _collect_true_same_axis_blockers_report_style(include_active=True)
        if not residual:
            return (0, 0.0, 0.0, 0.0)
        deltas = [float(rec['delta']) for rec in residual]
        hpwls = [float(rec['hpwl']) for rec in residual]
        count = len(residual)
        total_delta = float(sum(deltas))
        max_delta = float(max(deltas))
        hpwl_sum = float(sum(hpwls))
        if final_rescue_score_mode in {"delta", "delta_first", "residual"}:
            return (total_delta, max_delta, count, hpwl_sum)
        if final_rescue_score_mode in {"max", "max_first"}:
            return (max_delta, total_delta, count, hpwl_sum)
        return (count, total_delta, max_delta, hpwl_sum)

    def _candidate_local_removal_pool(hp, limit):
        """Return active hard pairs most likely to conflict with hp.

        This optional global-swap pool is local and generic: it considers active
        pairs sharing a segment with either endpoint of hp, then falls back to
        same family. It is disabled by default because trading one aligned edge
        for another is not always a net improvement, but it is useful for
        diagnosing tight local conflicts.
        """
        if hp.pin_a not in inst_pin_state or hp.pin_b not in inst_pin_state:
            return []
        segs = {inst_pin_state[hp.pin_a].seg_id, inst_pin_state[hp.pin_b].seg_id}
        target_vars = {_template_var_key_for_pin(hp.pin_a), _template_var_key_for_pin(hp.pin_b)}
        pool = []
        for old in hard_pairs_active:
            try:
                old_segs = set()
                if old.pin_a in inst_pin_state:
                    old_segs.add(inst_pin_state[old.pin_a].seg_id)
                if old.pin_b in inst_pin_state:
                    old_segs.add(inst_pin_state[old.pin_b].seg_id)
                old_vars = {_template_var_key_for_pin(old.pin_a), _template_var_key_for_pin(old.pin_b)}
                if (old_segs & segs or _family_key(old) == _family_key(hp) or
                        (policy_min_edge_alias_removal_enable and old_vars & target_vars)):
                    pool.append(old)
            except Exception:
                continue
        pool = sorted(pool, key=lambda e: (
            _edge_width_pressure(e),
            _edge_width(e),
            -_compute_hpwl(e),
            str(e.as_undirected_key()),
        ))
        return pool[:max(0, int(limit))]

    def _snapshot_all_state():
        return {
            "active": set(hard_pairs_active),
            "inst": snapshot_inst_state(inst_pin_state),
            "tmpl": snapshot_template(groups),
            "bucket": {seg: list(lst) for seg, lst in bucket_by_seg.items()},
            "orders": {seg: list(lst) for seg, lst in orders.items()},
            "anchors": list(required_anchor_active),
            "policy_dropped_keys": set(policy_component_dropped_edge_keys),
            "policy_dropped_records": list(policy_component_dropped_records),
            "free_target_dropped_keys": set(policy_free_target_dropped_edge_keys),
            "free_target_dropped_records": list(policy_free_target_dropped_records),
        }

    def _restore_all_state(snap):
        nonlocal hard_pairs_active, bucket_by_seg, orders, required_anchor_active
        nonlocal policy_component_dropped_edge_keys, policy_component_dropped_records
        nonlocal policy_free_target_dropped_edge_keys, policy_free_target_dropped_records
        hard_pairs_active = set(snap["active"])
        restore_inst_state(inst_pin_state, snap["inst"])
        restore_template(groups, snap["tmpl"])
        bucket_by_seg = {seg: list(lst) for seg, lst in snap["bucket"].items()}
        orders = {seg: list(lst) for seg, lst in snap["orders"].items()}
        required_anchor_active = list(snap.get("anchors", []))
        policy_component_dropped_edge_keys = set(snap.get("policy_dropped_keys", set()))
        policy_component_dropped_records = list(snap.get("policy_dropped_records", []))
        policy_free_target_dropped_edge_keys = set(snap.get("free_target_dropped_keys", set()))
        policy_free_target_dropped_records = list(snap.get("free_target_dropped_records", []))

    def _attempt_replace_admission(replace_edges, subset_edges):
        """Admission-layer replacement with optional canonical-score guard."""
        if not admission_score_accept:
            return _attempt_replace(set(replace_edges), set(subset_edges))
        snap = _snapshot_all_state()
        score_before = _residual_score()
        ok, err = _attempt_replace(set(replace_edges), set(subset_edges))
        if not ok:
            return False, err
        score_after = _residual_score()
        if _score_improved(score_before, score_after):
            return True, None
        _restore_all_state(snap)
        return False, RuntimeError(f"admission score did not improve: before={score_before} after={score_after}")

    def _score_improved(old_score, new_score):
        eps = 1e-7
        for old, new in zip(old_score, new_score):
            if new < old - eps:
                return True
            if new > old + eps:
                return False
        return False

    def _attempt_replace_dry(replace_edges, subset_edges):
        """Feasibility probe that always restores the current accepted state."""
        snap = _snapshot_all_state()
        ok, err = _attempt_replace(set(replace_edges), set(subset_edges))
        _restore_all_state(snap)
        return ok, err

    def _attempt_replace_score_accept(replace_edges, subset_edges, old_score=None):
        """Attempt a replacement and accept only if residual score improves.

        This wrapper is used only by the final residual rescue path.  Main greedy
        admission remains cardinality-oriented for speed/stability.
        """
        snap = _snapshot_all_state()
        score_before = _residual_score() if old_score is None else old_score
        ok, err = _attempt_replace(set(replace_edges), set(subset_edges))
        if not ok:
            return False, err, score_before, score_before
        score_after = _residual_score()
        if final_rescue_score_accept and not _score_improved(score_before, score_after):
            _restore_all_state(snap)
            return False, RuntimeError(f"rescue score did not improve: before={score_before} after={score_after}"), score_before, score_after
        return True, None, score_before, score_after

    def _rank_removal_combo(rem_set):
        rem_set = set(rem_set)
        return (
            len(rem_set),
            sum(_edge_width_pressure(e) for e in rem_set),
            sum(_edge_width(e) for e in rem_set),
            -sum(_compute_hpwl(e) for e in rem_set),
            ";".join(str(e.as_undirected_key()) for e in sorted(rem_set, key=lambda x: str(x.as_undirected_key()))),
        )

    def _bounded_removal_combos(pool, max_remove, beam_width):
        combos = []
        max_r = max(0, min(int(max_remove), len(pool)))
        for r in range(1, max_r + 1):
            for rem in combinations(pool, r):
                combos.append((_rank_removal_combo(rem), set(rem)))
        combos.sort(key=lambda x: x[0])
        return [rem for _, rem in combos[:max(0, int(beam_width))]]

    def _diagnose_final_blockers():
        """Print bounded minimum-removal diagnostics for residual blockers.

        This is intentionally diagnostic-only.  It does not commit any new
        hard-pairs or removals.  It answers: direct feasible? if not, can this
        blocker be made feasible by removing 1..N nearby active hard-pairs?
        """
        if not final_blocker_diagnose:
            return
        residual_recs = _collect_true_same_axis_blockers_report_style(include_active=True)
        residual = [rec['hp'] for rec in residual_recs]
        if not residual:
            if solver_progress_log:
                print("[final-diagnose] residual_blockers=0")
            return
        limit = max(0, int(final_blocker_diag_limit))
        shown = min(len(residual), limit)
        if solver_progress_log:
            print(
                f"[final-diagnose] residual_blockers={len(residual)} showing={shown} "
                f"max_remove={final_blocker_diag_max_remove} pool_cap={final_blocker_diag_pool_cap}"
            )
        for idx, hp in enumerate(residual[:shown], 1):
            if not _hard_pair_allowed_by_ref_master_policy(hp):
                if solver_progress_log:
                    print(
                        f"[final-diagnose] #{idx:02d} reason=held_by_reference_master_policy "
                        f"axis={hp.axis} delta={_alignment_delta(hp):.6f} hpwl={_compute_hpwl(hp):.6f} "
                        f"fixed_axis_dist={_fixed_axis_distance(hp):.6f} slack={_edge_interval_slack(hp):.6f} "
                        f"a={hp.pin_a} b={hp.pin_b}"
                    )
                continue
            direct_ok, direct_err = _attempt_replace_dry(set(), {hp})
            pool = _candidate_local_removal_pool(hp, final_blocker_diag_pool_cap)
            min_remove = None
            best_removal = None
            best_err = direct_err
            if not direct_ok and pool and final_blocker_diag_max_remove > 0:
                for rem_set in _bounded_removal_combos(pool, final_blocker_diag_max_remove, final_blocker_diag_beam_width):
                    ok, err = _attempt_replace_dry(rem_set, {hp})
                    if ok:
                        min_remove = len(rem_set)
                        best_removal = rem_set
                        best_err = None
                        break
                    best_err = err
            reason = "direct_feasible" if direct_ok else ("feasible_after_removal" if min_remove is not None else "blocked_under_bounded_removal")
            rem_desc = ""
            if best_removal:
                rem_desc = " remove=[" + "; ".join(str(_hp_sig(e)) for e in sorted(best_removal, key=lambda x: str(x.as_undirected_key()))) + "]"
            err_desc = ""
            if best_err is not None and reason == "blocked_under_bounded_removal":
                msg = str(best_err).replace("\n", " ")
                if len(msg) > 180:
                    msg = msg[:177] + "..."
                err_desc = f" err={msg}"
            if solver_progress_log:
                print(
                    f"[final-diagnose] #{idx:02d} reason={reason} "
                    f"min_remove={0 if direct_ok else (min_remove if min_remove is not None else '>{}'.format(final_blocker_diag_max_remove))} "
                    f"axis={hp.axis} delta={_alignment_delta(hp):.6f} hpwl={_compute_hpwl(hp):.6f} "
                    f"fixed_axis_dist={_fixed_axis_distance(hp):.6f} slack={_edge_interval_slack(hp):.6f} "
                    f"a={hp.pin_a} b={hp.pin_b}{rem_desc}{err_desc}"
                )

    def _run_final_rescue_pass():
        """Generic score-guarded residual rescue for true same-axis blockers.

        The main outer loop is a forward greedy family admission.  Some pairs are
        not candidates at the right moment, or become feasible only after later
        movement/follower-anchor closure.  This pass recomputes residual targets
        from the final geometry and attempts to admit them with a wider search.

        Unlike ordinary admission, every committed mutation must improve the
        final residual score unless FINAL_RESCUE_SCORE_ACCEPT=0.
        """
        nonlocal hard_pairs_active, inst_pin_state, bucket_by_seg, orders
        if not final_rescue_enable:
            _diagnose_final_blockers()
            return 0, 0

        total_admitted = 0
        last_score = None
        for rr in range(max(0, final_rescue_rounds)):
            candidates = _collect_final_rescue_candidates()
            if not candidates:
                break
            score_before = _residual_score()
            if score_before == last_score and rr > 0:
                break
            last_score = score_before

            round_snap = _snapshot_all_state()
            admitted = 0
            rejected = 0
            anchor_kept = 0
            global_added = 0
            score_after_ordinary = score_before

            # First try the existing family admission machinery, but keep it only
            # if the final residual score improves.  This prevents feasible but
            # globally harmful swaps from becoming permanent.
            admitted_tmp, rejected_tmp = _run_family_admission(
                candidates,
                search_cap=final_rescue_search_cap,
                allow_family_swap=True,
                swap_cap=final_rescue_swap_cap,
                removal_pool_cap=final_rescue_removal_pool_cap,
                beam_width=final_rescue_beam_width,
            )
            anchor_tmp, _, _ = _run_follower_anchor_closure()
            score_after_ordinary = _residual_score()
            if (admitted_tmp or anchor_tmp) and (not final_rescue_score_accept or _score_improved(score_before, score_after_ordinary)):
                admitted = admitted_tmp
                anchor_kept = anchor_tmp
                total_admitted += admitted + int(anchor_kept)
            else:
                _restore_all_state(round_snap)

            # Optional local global-swap: try one residual at a time by removing
            # nearby low-criticality active pairs.  Each successful swap must also
            # improve the residual score.
            if final_rescue_global_swap:
                candidates2 = _collect_final_rescue_candidates()
                for hp in candidates2[:max(1, final_rescue_beam_width)]:
                    if _alignment_delta(hp) <= tol or hp in hard_pairs_active:
                        continue
                    old_score = _residual_score()
                    ok, _, _, _ = _attempt_replace_score_accept(set(), {hp}, old_score=old_score)
                    if ok:
                        global_added += 1
                        total_admitted += 1
                        continue

                    pool = _candidate_local_removal_pool(hp, final_rescue_removal_pool_cap)
                    found = False
                    for rem_set in _bounded_removal_combos(pool, final_rescue_swap_cap, final_rescue_beam_width):
                        old_score = _residual_score()
                        ok2, _, _, _ = _attempt_replace_score_accept(rem_set, {hp}, old_score=old_score)
                        if ok2:
                            global_added += 1
                            total_admitted += 1
                            found = True
                            break
                    if found:
                        continue

            # Follower-anchor targets can change after successful ordinary/global
            # rescue. Keep anchor closure only if it improves residual score.
            anchor_snap = _snapshot_all_state()
            score_before_anchor = _residual_score()
            anchor_extra, _, _ = _run_follower_anchor_closure()
            score_after_anchor = _residual_score()
            if anchor_extra and (not final_rescue_score_accept or _score_improved(score_before_anchor, score_after_anchor)):
                anchor_kept += anchor_extra
                total_admitted += int(anchor_extra)
            elif anchor_extra:
                _restore_all_state(anchor_snap)

            score_after = _residual_score()
            if solver_progress_log:
                print(
                    f"[final-rescue] round={rr + 1} score_before={score_before} score_after={score_after} "
                    f"ordinary_admitted={admitted} anchor_kept={anchor_kept} global_added={global_added}"
                )
            if final_rescue_count_gate and score_after[0] >= score_before[0]:
                # Do not spend more rounds when the primary residual count does
                # not decrease.  Tiny total-delta improvements caused the heavy
                # rescue loop to run for a long time on old/easy datasets.
                break
            if not _score_improved(score_before, score_after):
                break

        final_residual = len(_collect_true_same_axis_blockers_report_style(include_active=True))
        if final_rescue_enable and solver_progress_log:
            print(f"[final-rescue] total_progress={total_admitted} residual_after={final_residual} score={_residual_score()}")
        if not required_hard_closure_enable:
            _diagnose_final_blockers()
        return total_admitted, final_residual

    def _run_follower_anchor_closure():
        """Try follower-side low-HPWL edges without letting them pull master.

        The closure is greedy and feasibility-preserving. For each exactly-one-
        follower pair, the follower endpoint supplies a fixed scalar target and
        only the other endpoint receives a hard fixed-scalar constraint. Failed
        constraints are rejected and do not affect the final state.
        """
        nonlocal inst_pin_state, bucket_by_seg, orders
        if not (enable_hard_iso and follower_anchor_closure and not follower_hardpairs_affect_master):
            return 0, 0, 0

        _refresh_bucket_from_state()
        raw_pairs = build_hard_align_pairs(active_nets, inst_pin_state, real_segments, hpwl_thresh)
        candidates = []
        skipped = 0
        for hp in raw_pairs:
            if _hard_pair_allowed_by_ref_master_policy(hp):
                continue
            item = _follower_anchor_constraint_from_pair(hp)
            if item is None:
                skipped += 1
                continue
            candidates.append(hp)

        if not candidates:
            return 0, 0, skipped

        candidates = sorted(
            set(candidates),
            key=lambda hp: (_fixed_axis_distance(hp), _compute_hpwl(hp), str(hp.as_undirected_key()))
        )

        fixed_constraints = []
        fixed_sig = set()
        kept = 0
        rejected = 0

        for hp in candidates:
            item = _follower_anchor_constraint_from_pair(hp)
            if item is None:
                skipped += 1
                continue
            constraints_for_hp = _follower_anchor_fixed_constraints_from_pair(hp)
            if not constraints_for_hp:
                skipped += 1
                continue
            sig_items = tuple((pk, round(float(ts), 6)) for pk, ts in constraints_for_hp)
            if any(si in fixed_sig for si in sig_items):
                continue

            inst_backup = snapshot_inst_state(inst_pin_state)
            tmpl_backup = snapshot_template(groups)
            bucket_backup = {seg: list(lst) for seg, lst in bucket_by_seg.items()}
            orders_backup = {seg: list(lst) for seg, lst in orders.items()}
            trial_constraints = required_anchor_active + fixed_constraints + constraints_for_hp

            try:
                _refresh_bucket_from_state()
                last_orders = None
                for _ in range(max(1, local_fixedpoint_iters)):
                    orders_try = _rebuild_orders_current()
                    prev_values_try = snapshot_template(groups) if enable_hard_iso else snapshot_inst_state(inst_pin_state)
                    solve_inner_qp_dummy(
                        groups, inst_pin_state, orders_try, hard_pairs_active,
                        enable_hard_iso=enable_hard_iso, keepout=keepout,
                        real_segments=real_segments, prev_values=prev_values_try,
                        init_values=initial_real_anchor,
                        fixed_scalar_constraints=trial_constraints,
                    )
                    _refresh_bucket_from_state()
                    if last_orders is not None and orders_equal(last_orders, orders_try):
                        orders = orders_try
                        break
                    last_orders = orders_try
                else:
                    orders = _rebuild_orders_current()

                fixed_constraints.extend(constraints_for_hp)
                fixed_sig.update(sig_items)
                kept += 1
            except Exception:
                rejected += 1
                restore_inst_state(inst_pin_state, inst_backup)
                restore_template(groups, tmpl_backup)
                bucket_by_seg = bucket_backup
                orders = orders_backup

        if kept or rejected or skipped:
            _debug_print(f"[follower-anchor] kept={kept} rejected={rejected} skipped={skipped} candidates={len(candidates)}")
        return kept, rejected, skipped



    def _attempt_required_anchor_constraints(anchor_constraints, order_hint_constraints=None):
        """Commit a set of follower-anchor fixed-scalar constraints exactly.

        This is used only in the terminal required-closure phase.  The list of
        fixed constraints is cumulative, so every accepted anchor remains enforced
        while later anchors are tested.  After this phase no ordinary solve is run,
        so the final state keeps these hard equalities.

        order_hint_constraints is trial-only: before building orders, the hinted
        pins/templates are moved to their target scalar positions.  The QP still
        validates the actual fixed constraints.  This fixes false infeasibility
        caused by stale no-overlap order around a still-misaligned residual edge.
        """
        nonlocal inst_pin_state, bucket_by_seg, orders
        inst_backup = snapshot_inst_state(inst_pin_state)
        tmpl_backup = snapshot_template(groups)
        bucket_backup = {seg: list(lst) for seg, lst in bucket_by_seg.items()}
        orders_backup = {seg: list(lst) for seg, lst in orders.items()}
        try:
            if order_hint_constraints and policy_min_edge_target_order_hint:
                _apply_order_hints(order_hint_constraints)
            _refresh_bucket_from_state()
            last_orders = None
            for _ in range(max(1, local_fixedpoint_iters)):
                orders_try = _rebuild_orders_current()
                prev_values_try = snapshot_template(groups) if enable_hard_iso else snapshot_inst_state(inst_pin_state)
                solve_inner_qp_dummy(
                    groups, inst_pin_state, orders_try, hard_pairs_active,
                    enable_hard_iso=enable_hard_iso, keepout=keepout,
                    real_segments=real_segments, prev_values=prev_values_try,
                    init_values=initial_real_anchor,
                    fixed_scalar_constraints=anchor_constraints,
                )
                _refresh_bucket_from_state()
                if last_orders is not None and orders_equal(last_orders, orders_try):
                    orders = orders_try
                    return True, None
                last_orders = orders_try
            orders = _rebuild_orders_current()
            return True, None
        except Exception as e:
            restore_inst_state(inst_pin_state, inst_backup)
            restore_template(groups, tmpl_backup)
            bucket_by_seg = bucket_backup
            orders = orders_backup
            return False, e

    def _merge_new_anchor_constraints(cons):
        """Return only new fixed constraints, or None if they conflict.

        Existing persistent anchors remain active.  If a requested fixed scalar
        on the same pin disagrees with an existing scalar, the E->target edge is
        not simultaneously satisfiable under current accepted constraints.
        """
        deduped = _dedup_fixed_constraints(cons)
        if deduped is None:
            return None
        existing = {pin: float(target) for pin, target in required_anchor_active}
        new_constraints = []
        for pin_key, target in deduped:
            target = float(target)
            if pin_key in existing:
                if abs(existing[pin_key] - target) > 1e-6:
                    return None
                continue
            new_constraints.append((pin_key, target))
        return new_constraints

    def _free_to_homology_target_precheck(hp):
        """Check whether an E->B fixed-target trial is meaningful.

        Returns (ok, reason, constraints).  ok=False is a deterministic
        non-trial rejection, not a solver failure:
          - role_not_free_to_target: not an E--A/B/C/D edge;
          - missing_state: endpoint missing from current state;
          - axis_mismatch: endpoints do not share the active free axis;
          - target_out_of_movable_interval: E cannot reach B's scalar at all;
          - conflicting_existing_anchor: E or target already has incompatible
            accepted fixed scalar.
        """
        cls, movable, target_pin = _alignment_policy_class(hp)
        if cls != "free_to_homology_target" or movable is None or target_pin is None:
            return False, "role_not_free_to_target", []
        if movable not in inst_pin_state or target_pin not in inst_pin_state:
            return False, "missing_state", []
        mst = inst_pin_state[movable]
        tst = inst_pin_state[target_pin]
        mseg = find_segment_by_global_id(real_segments, mst.seg_id)
        tseg = find_segment_by_global_id(real_segments, tst.seg_id)
        if mseg.free_axis != tseg.free_axis or mseg.free_axis != hp.axis:
            return False, "axis_mismatch", []
        target = float(tst.s_center)
        if target < float(mst.s_min) - 1e-9 or target > float(mst.s_max) + 1e-9:
            return False, "target_out_of_movable_interval", []
        candidate_sets = _policy_target_constraint_sets_from_pair(hp)
        if not candidate_sets:
            return False, "no_candidate_constraints", []
        cons = candidate_sets[0]
        new_constraints = _merge_new_anchor_constraints(cons)
        if new_constraints is None:
            return False, "conflicting_existing_anchor", []
        if not new_constraints and not _edge_aligned_after_solve(hp):
            return False, "existing_anchor_not_aligned", []
        return True, "ok", cons

    def _run_free_to_homology_target_closure(policy_records):
        """Commit E->A/B/C/D fixed-target alignments before generic policy search.

        This implements the rule:
            ABCD are homology copies with A as master; B/C/D are targets.
            If an external free module E wants to align to B, B does not move;
            E alone is constrained to B's current derived scalar.

        Rejected edges are not sent to policy-component fallback in this round.
        Under master-only homology semantics, a rejected fixed-target edge is not
        an independent required equality: the homology target cannot be moved,
        and the movable endpoint cannot legally reach it under the accepted
        constraints.  Therefore rejected free-target edges are dropped from
        required-blocker accounting by policy.
        """
        nonlocal required_anchor_active, inst_pin_state, bucket_by_seg, orders
        nonlocal policy_free_target_dropped_edge_keys, policy_free_target_dropped_records

        def _mark_free_target_dropped(rec, reason):
            hp = rec['hp']
            edge_key = rec.get('edge_key', hp.as_undirected_key())
            if edge_key in policy_free_target_dropped_edge_keys:
                return
            policy_free_target_dropped_edge_keys.add(edge_key)
            policy_free_target_dropped_records.append({
                'edge_key': edge_key,
                'hpwl': float(rec.get('hpwl', _compute_hpwl(hp))),
                'delta': float(rec.get('delta', _alignment_delta(hp))),
                'axis': hp.axis,
                'a': hp.pin_a,
                'b': hp.pin_b,
                'reason': 'free_target_' + str(reason),
            })
        if not (free_to_homology_target_closure_enable and enable_hard_iso and follower_anchor_closure):
            return 0, 0, 0
        if not policy_records:
            return 0, 0, 0

        records = [rec for rec in policy_records if rec.get('policy_class') == "free_to_homology_target"]
        if not records:
            return 0, 0, 0

        records = sorted(records, key=lambda rec: (
            float(rec.get('fixed_axis_distance', _fixed_axis_distance(rec['hp']))),
            float(rec.get('hpwl', _compute_hpwl(rec['hp']))),
            -_edge_width(rec['hp']),
            str(rec.get('edge_key', rec['hp'].as_undirected_key())),
        ))

        kept = 0
        rejected = 0
        skipped = 0
        trials = 0
        budget = max(0, int(free_to_homology_target_trial_budget))
        fail_reasons = {}

        for rec in records:
            hp = rec['hp']
            edge_key = rec.get('edge_key', hp.as_undirected_key())
            if edge_key in free_to_homology_target_failed_edge_keys:
                skipped += 1
                continue
            ok_pre, reason, cons = _free_to_homology_target_precheck(hp)
            if not ok_pre:
                rejected += 1
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
                free_to_homology_target_failed_edge_keys.add(edge_key)
                _mark_free_target_dropped(rec, reason)
                continue
            if _edge_aligned_after_solve(hp):
                kept += 1
                continue
            if budget and trials >= budget:
                skipped += 1
                continue
            trials += 1

            inst_backup = snapshot_inst_state(inst_pin_state)
            tmpl_backup = snapshot_template(groups)
            bucket_backup = {seg: list(lst) for seg, lst in bucket_by_seg.items()}
            orders_backup = {seg: list(lst) for seg, lst in orders.items()}
            before_required = list(required_anchor_active)

            new_constraints = _merge_new_anchor_constraints(cons)
            if new_constraints is None:
                rejected += 1
                fail_reasons["conflicting_existing_anchor"] = fail_reasons.get("conflicting_existing_anchor", 0) + 1
                free_to_homology_target_failed_edge_keys.add(edge_key)
                _mark_free_target_dropped(rec, "conflicting_existing_anchor")
                continue
            trial_constraints = required_anchor_active + new_constraints
            ok_solve, _ = _attempt_required_anchor_constraints(trial_constraints, order_hint_constraints=cons)
            if ok_solve and _edge_aligned_after_solve(hp):
                required_anchor_active = trial_constraints
                kept += 1
                continue

            restore_inst_state(inst_pin_state, inst_backup)
            restore_template(groups, tmpl_backup)
            bucket_by_seg = bucket_backup
            orders = orders_backup
            required_anchor_active = before_required
            rejected += 1
            fail_reasons["solve_failed_or_not_aligned"] = fail_reasons.get("solve_failed_or_not_aligned", 0) + 1
            free_to_homology_target_failed_edge_keys.add(edge_key)
            _mark_free_target_dropped(rec, "solve_failed_or_not_aligned")

        if solver_progress_log and (kept or rejected or skipped):
            detail = " ".join(f"{k}={v}" for k, v in sorted(fail_reasons.items()))
            print(
                f"[free-target] kept={kept} rejected={rejected} skipped={skipped} "
                f"candidates={len(records)} trials={trials} budget={budget} {detail}".rstrip()
            )
        return kept, rejected, skipped

    def _run_required_anchor_closure(policy_records):
        """Force reference-master-blocked residuals via policy-component targets.

        The requested policy is:
          1. Use plan B for policy-blocked edges: do not let the edge pull the
             master/template as an ordinary equality; instead, create hard
             fixed-scalar anchor constraints that make the relevant endpoints
             equal.
          2. If a connected policy component cannot be fully anchored, keep only
             the edge with the smallest current HPWL in that component and mark
             the other component edges as intentionally dropped. This mirrors the
             GH star rule, but for reference-master policy components.

        Accepted constraints are appended to required_anchor_active and are
        passed to every later QP solve.  Dropped policy-component edge keys are
        removed from the ordinary residual-blocker count.
        """
        nonlocal required_anchor_active, policy_component_dropped_edge_keys, policy_component_dropped_records
        if not (enable_hard_iso and follower_anchor_closure and policy_records):
            return 0, 0, len(policy_records)

        accepted_constraints = []
        fixed_by_pin = {}
        for pin_key, target in required_anchor_active:
            fixed_by_pin[pin_key] = float(target)

        def _norm_edge_key(hp):
            return hp.as_undirected_key()

        def _dedup_edge_records(records):
            out = []
            seen_keys = set()
            for rec in records:
                k = rec.get('edge_key', _norm_edge_key(rec['hp']))
                if k in seen_keys:
                    continue
                seen_keys.add(k)
                out.append(rec)
            return out

        policy_records = _dedup_edge_records(policy_records)

        # Connected components are built per axis; edges in different axes must
        # never be forced into one scalar target.
        parent = {}
        def find(x):
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for rec in policy_records:
            hp = rec['hp']
            union((hp.axis, hp.pin_a), (hp.axis, hp.pin_b))

        comps = {}
        for rec in policy_records:
            hp = rec['hp']
            root = find((hp.axis, hp.pin_a))
            comps.setdefault(root, []).append(rec)

        def _constraints_conflict_or_new(cons):
            """Return (conflict, new_constraints)."""
            new_constraints = []
            for pin_key, target in _dedup_fixed_constraints(cons):
                target = float(target)
                if pin_key in fixed_by_pin:
                    if abs(fixed_by_pin[pin_key] - target) > 1e-6:
                        return True, []
                    continue
                new_constraints.append((pin_key, target))
            return False, new_constraints

        def _try_accept_constraints(cons):
            conflict, new_constraints = _constraints_conflict_or_new(cons)
            if conflict or not new_constraints:
                return False
            trial_constraints = required_anchor_active + accepted_constraints + new_constraints
            ok, _ = _attempt_required_anchor_constraints(trial_constraints)
            if not ok:
                return False
            accepted_constraints.extend(new_constraints)
            for pin_key, target in new_constraints:
                fixed_by_pin[pin_key] = float(target)
            return True

        def _rebuild_fixed_by_pin():
            fixed_by_pin.clear()
            for pin_key, target in required_anchor_active:
                fixed_by_pin[pin_key] = float(target)
            for pin_key, target in accepted_constraints:
                fixed_by_pin[pin_key] = float(target)

        def _remove_anchor_constraints_for_pins(pin_set):
            """Remove persistent/current-pass anchors touching pin_set.

            This is used only for policy-component fallback: if a component
            cannot be fully anchored, the requested rule is to keep the HPWL
            shortest edge.  To make that possible, old policy anchors touching
            the same pins may be replaced by the new shortest-edge anchor.
            """
            nonlocal required_anchor_active, accepted_constraints
            pin_set = set(pin_set)
            before_active = list(required_anchor_active)
            before_accepted = list(accepted_constraints)
            required_anchor_active = [c for c in required_anchor_active if c[0] not in pin_set]
            accepted_constraints = [c for c in accepted_constraints if c[0] not in pin_set]
            removed = (len(before_active) - len(required_anchor_active)) + (len(before_accepted) - len(accepted_constraints))
            _rebuild_fixed_by_pin()
            return removed, before_active, before_accepted

        def _restore_anchor_constraint_lists(before_active, before_accepted):
            nonlocal required_anchor_active, accepted_constraints
            required_anchor_active = list(before_active)
            accepted_constraints = list(before_accepted)
            _rebuild_fixed_by_pin()

        def _try_accept_constraints_replacing_pins(cons, replace_pins):
            """Try cons after replacing old anchors on replace_pins.

            If the replacement attempt fails, both anchor lists and the QP state
            are restored by _attempt_required_anchor_constraints / explicit list
            rollback.  If it succeeds, the new constraints stay persistent and
            any old anchors on replace_pins remain removed.
            """
            removed, before_active, before_accepted = _remove_anchor_constraints_for_pins(replace_pins)
            if removed <= 0:
                return _try_accept_constraints(cons)
            ok = _try_accept_constraints(cons)
            if not ok:
                _restore_anchor_constraint_lists(before_active, before_accepted)
                return False
            return True

        def _pins_for_records(records):
            pins = []
            seen = set()
            for rec in records:
                hp = rec['hp']
                for pin in (hp.pin_a, hp.pin_b):
                    if pin not in seen:
                        seen.add(pin)
                        pins.append(pin)
            return pins

        def _component_interval(pins):
            ivs = []
            for pin in pins:
                iv = _interval_of(pin)
                if iv is None:
                    return None
                ivs.append(iv)
            lo = max(float(iv[0]) for iv in ivs)
            hi = min(float(iv[1]) for iv in ivs)
            if lo > hi + 1e-9:
                return None
            return lo, hi

        def _component_targets(records):
            pins = _pins_for_records(records)
            iv = _component_interval(pins)
            if iv is None:
                return []
            lo, hi = iv
            existing = []
            for pin in pins:
                if pin in fixed_by_pin:
                    existing.append(float(fixed_by_pin[pin]))
            # If component already contains fixed endpoints, all existing fixed
            # targets must agree; otherwise the full component cannot be anchored.
            if existing:
                ref = existing[0]
                if any(abs(x - ref) > 1e-6 for x in existing):
                    return []
                if lo - 1e-9 <= ref <= hi + 1e-9:
                    return [ref]
                return []

            vals = []
            for pin in pins:
                st = inst_pin_state.get(pin)
                if st is not None:
                    vals.append(float(st.s_center))
            if not vals:
                vals = [0.5 * (lo + hi)]
            vals_sorted = sorted(vals)
            median = vals_sorted[len(vals_sorted) // 2]
            mean = sum(vals) / len(vals)
            def clamp(x):
                return max(lo, min(hi, float(x)))
            targets = []
            for x in vals + [median, mean, 0.5 * (lo + hi), lo, hi]:
                x = clamp(x)
                if lo - 1e-9 <= x <= hi + 1e-9:
                    targets.append(float(x))
            # Prefer targets with smaller total movement of the component pins.
            uniq = []
            seen = set()
            for x in targets:
                rx = round(float(x), 6)
                if rx in seen:
                    continue
                seen.add(rx)
                movement = sum(abs(float(inst_pin_state[p].s_center) - float(x)) for p in pins if p in inst_pin_state)
                uniq.append((movement, float(x)))
            uniq.sort(key=lambda item: (item[0], item[1]))
            return [x for _, x in uniq]

        def _rollback_anchor_acceptance(before_active, before_accepted):
            _restore_anchor_constraint_lists(before_active, before_accepted)

        def _try_accept_full_component(records):
            # Role-aware E->B/A target edges must not move the homology target.
            # A full-component common target could drag B/A, so use only
            # single-edge fixed-target trials for these records.
            if any(rec.get('policy_class') == "free_to_homology_target" for rec in records):
                return False
            pins = _pins_for_records(records)
            if not pins:
                return False
            hps = [rec['hp'] for rec in records]
            for target in _component_targets(records):
                cons = [(pin, target) for pin in pins]
                before_active = list(required_anchor_active)
                before_accepted = list(accepted_constraints)
                if _try_accept_constraints(cons):
                    if all(_edge_aligned_after_solve(hp) for hp in hps):
                        return True
                    # Feasible but not actually aligned: do not count as kept.
                    _rollback_anchor_acceptance(before_active, before_accepted)
            return False

        def _try_accept_single_edge(rec, allow_replace=False, replace_pins=None):
            hp = rec['hp']
            candidate_sets = _policy_target_constraint_sets_from_pair(hp)
            if replace_pins is None:
                replace_pins = [hp.pin_a, hp.pin_b]
            for cons in candidate_sets:
                before_active = list(required_anchor_active)
                before_accepted = list(accepted_constraints)
                if allow_replace:
                    accepted = _try_accept_constraints_replacing_pins(cons, replace_pins)
                else:
                    accepted = _try_accept_constraints(cons)
                if accepted:
                    if _edge_aligned_after_solve(hp):
                        return True
                    # Feasible QP, but the represented edge still has nonzero
                    # delta. Roll back so this does not become a false kept.
                    _rollback_anchor_acceptance(before_active, before_accepted)
            return False

        kept = 0
        rejected = 0
        skipped = 0
        fallback_kept = 0
        fallback_dropped = 0
        full_component_kept = 0

        ordered_components = sorted(
            comps.values(),
            key=lambda recs: (
                min(float(r.get('hpwl', _compute_hpwl(r['hp']))) for r in recs),
                -len(recs),
                str(min(str(r.get('edge_key', _norm_edge_key(r['hp']))) for r in recs)),
            )
        )

        for recs in ordered_components:
            recs = _dedup_edge_records(recs)
            recs.sort(key=lambda rec: (
                float(rec.get('hpwl', _compute_hpwl(rec['hp']))),
                float(rec.get('fixed_axis_distance', _fixed_axis_distance(rec['hp']))),
                -_edge_width(rec['hp']),
                str(rec.get('edge_key', _norm_edge_key(rec['hp']))),
            ))

            # First try the mathematically strongest plan B: anchor the entire
            # policy component to one common target so every edge in the component
            # has delta = 0.
            if _try_accept_full_component(recs):
                full_component_kept += 1
                kept += len(recs)
                continue

            # If the component cannot be fully anchored, try fallback edges in
            # HPWL order, but never send HPWL-over-limit edges to GUROBI.  The
            # first feasible low-HPWL edge is kept; all other component edges are
            # intentionally dropped from ordinary blocker accounting.  If no
            # low-HPWL edge can be anchored, keep only the lowest-HPWL low-limit
            # representative as unresolved and drop the rest.
            eligible = [
                rec for rec in recs
                if float(rec.get('hpwl', _compute_hpwl(rec['hp']))) <= policy_component_fallback_hpwl_limit + 1e-9
            ]
            over_limit = [rec for rec in recs if rec not in eligible]
            accepted_rec = None
            accepted_replaced_existing_anchor = False
            component_pins = _pins_for_records(recs)
            for cand in eligible:
                accepted = _try_accept_single_edge(cand)
                replaced_existing_anchor = False
                if not accepted:
                    accepted = _try_accept_single_edge(cand, allow_replace=True, replace_pins=component_pins)
                    replaced_existing_anchor = accepted
                if accepted:
                    accepted_rec = cand
                    accepted_replaced_existing_anchor = replaced_existing_anchor
                    break

            if accepted_rec is not None:
                fallback_kept += 1
                kept += 1
                kept_key = accepted_rec.get('edge_key', _norm_edge_key(accepted_rec['hp']))
                for rec in recs:
                    rec_key = rec.get('edge_key', _norm_edge_key(rec['hp']))
                    if rec_key == kept_key:
                        continue
                    if rec_key not in policy_component_dropped_edge_keys:
                        policy_component_dropped_edge_keys.add(rec_key)
                        policy_component_dropped_records.append({
                            'edge_key': rec_key,
                            'hpwl': float(rec.get('hpwl', _compute_hpwl(rec['hp']))),
                            'delta': float(rec.get('delta', _alignment_delta(rec['hp']))),
                            'axis': rec['hp'].axis,
                            'a': rec['hp'].pin_a,
                            'b': rec['hp'].pin_b,
                            'kept_edge': kept_key,
                            'reason': 'policy_component_hpwl_ordered_keep_replace_old_anchor' if accepted_replaced_existing_anchor else 'policy_component_hpwl_ordered_keep',
                        })
                        fallback_dropped += 1
            else:
                # No candidate could be anchored.  Preserve exactly one
                # low-limit representative if available, otherwise drop the whole
                # over-limit component.  This keeps final blocker counts aligned
                # with the user's HPWL-ordered fallback policy.
                keep_key = None
                if eligible:
                    keep_rec = eligible[0]
                    keep_key = keep_rec.get('edge_key', _norm_edge_key(keep_rec['hp']))
                    rejected += 1
                else:
                    skipped += len(recs)
                for rec in recs:
                    rec_key = rec.get('edge_key', _norm_edge_key(rec['hp']))
                    if keep_key is not None and rec_key == keep_key:
                        continue
                    if rec_key not in policy_component_dropped_edge_keys:
                        policy_component_dropped_edge_keys.add(rec_key)
                        policy_component_dropped_records.append({
                            'edge_key': rec_key,
                            'hpwl': float(rec.get('hpwl', _compute_hpwl(rec['hp']))),
                            'delta': float(rec.get('delta', _alignment_delta(rec['hp']))),
                            'axis': rec['hp'].axis,
                            'a': rec['hp'].pin_a,
                            'b': rec['hp'].pin_b,
                            'kept_edge': keep_key,
                            'reason': 'policy_component_hpwl_ordered_all_failed_drop_non_representatives' if keep_key is not None else 'policy_component_all_candidates_over_hpwl_limit',
                        })
                        fallback_dropped += 1

        if accepted_constraints:
            required_anchor_active.extend(accepted_constraints)
        if solver_progress_log and (kept or rejected or skipped or fallback_dropped):
            print(
                f"[required-anchor] kept={kept} rejected={rejected} skipped={skipped} "
                f"persistent={len(required_anchor_active)} candidates={len(policy_records)} "
                f"full_components={full_component_kept} fallback_kept={fallback_kept} "
                f"fallback_dropped={fallback_dropped} "
                f"fallback_hpwl_limit={policy_component_fallback_hpwl_limit:g}"
            )
        return kept, rejected, skipped

    def _commit_required_ordinary_blocker(hp):
        """Force one ordinary blocker as a hard equality if bounded-feasible.

        The acceptance criterion is feasibility, not score improvement.  If the
        edge can be made feasible by removing up to REQUIRED_HARD_CLOSURE_MAX_REMOVE
        nearby active hard-pairs, commit that replacement and let the removed edges
        reappear as residual blockers in later rounds if they are still required.
        """
        if hp in hard_pairs_active and _alignment_delta(hp) <= tol:
            return True, "already_aligned"
        ok, err = _attempt_replace(set(), {hp})
        if ok:
            return True, "direct"
        pool = _candidate_local_removal_pool(hp, required_hard_closure_pool_cap)
        for rem_set in _bounded_removal_combos(pool, required_hard_closure_max_remove, required_hard_closure_beam_width):
            ok2, err2 = _attempt_replace(rem_set, {hp})
            if ok2:
                return True, f"remove_{len(rem_set)}"
            err = err2
        return False, err

    def _remove_policy_anchors_touching_pins(pin_set):
        """Remove persistent required-anchor constraints touching pin_set.

        In hard-iso mode this also removes anchors on alias real pins that share
        the same template optimization variable.
        """
        nonlocal required_anchor_active
        pin_set = set(pin_set or [])
        target_vars = {_template_var_key_for_pin(pin) for pin in pin_set}
        before = list(required_anchor_active)
        kept = []
        for c in required_anchor_active:
            cpin = c[0]
            if cpin in pin_set:
                continue
            if policy_min_edge_alias_removal_enable and _template_var_key_for_pin(cpin) in target_vars:
                continue
            kept.append(c)
        required_anchor_active = kept
        return before, len(before) - len(required_anchor_active)

    def _drop_policy_component_mates_for_kept(kept_rec, reason="policy_component_representative_aligned_drop_mates"):
        """Drop residual policy edges in the same connected component as kept_rec.

        Conflict replacement works on one representative edge at a time.  After
        a representative is really aligned, all remaining policy-blocked residual
        edges connected to it by shared same-axis pins are policy fallbacks, not
        ordinary blockers.  This closes the loop between edge-level acceptance
        and the final collector.
        """
        kept_hp = kept_rec['hp']
        kept_key = kept_rec.get('edge_key', kept_hp.as_undirected_key())
        residual = _collect_true_same_axis_blockers_report_style(include_active=True)
        policy_residual = [
            r for r in residual
            if not _hard_pair_allowed_by_ref_master_policy(r['hp'])
            and r['hp'].axis == kept_hp.axis
        ]
        if not policy_residual:
            return 0

        # Build a same-axis connectivity component over the kept edge plus the
        # current residual policy edges.  This catches the M case where aligning
        # one representative makes another edge sharing the same pin appear.
        parent = {}
        def find(x):
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        all_recs = [kept_rec] + policy_residual
        for r in all_recs:
            hp = r['hp']
            if hp.axis != kept_hp.axis:
                continue
            union((hp.axis, hp.pin_a), (hp.axis, hp.pin_b))

        kept_root = find((kept_hp.axis, kept_hp.pin_a))
        dropped = 0
        for r in policy_residual:
            hp = r['hp']
            if hp.axis != kept_hp.axis:
                continue
            if find((hp.axis, hp.pin_a)) != kept_root:
                continue
            rec_key = r.get('edge_key', hp.as_undirected_key())
            if rec_key == kept_key:
                continue
            if rec_key not in policy_component_dropped_edge_keys:
                policy_component_dropped_edge_keys.add(rec_key)
                policy_component_dropped_records.append({
                    'edge_key': rec_key,
                    'hpwl': float(r.get('hpwl', _compute_hpwl(hp))),
                    'delta': float(r.get('delta', _alignment_delta(hp))),
                    'axis': hp.axis,
                    'a': hp.pin_a,
                    'b': hp.pin_b,
                    'kept_edge': kept_key,
                    'reason': reason,
                })
                dropped += 1
        return dropped

    def _collector_still_contains_edge(rec):
        hp = rec['hp']
        key = rec.get('edge_key', hp.as_undirected_key())
        residual = _collect_true_same_axis_blockers_report_style(include_active=True)
        return any(r.get('edge_key', r['hp'].as_undirected_key()) == key for r in residual)

    def _try_policy_edge_anchor_with_conflict_replacement(rec):
        """Try to force one remaining policy representative to delta=0.

        This is more aggressive than policy-component fallback: it can remove
        nearby active hard-pairs and old persistent anchors touching the two
        endpoint pins, then re-solve with the candidate policy-target hard
        constraints.  It is used only for final low-HPWL representatives that
        survived all earlier closure passes.
        """
        nonlocal hard_pairs_active, required_anchor_active
        hp = rec['hp']
        if float(rec.get('hpwl', _compute_hpwl(hp))) > policy_component_fallback_hpwl_limit + 1e-9:
            return False, "hpwl_over_policy_limit", 0
        candidate_sets = _policy_target_constraint_sets_from_pair(hp, include_scan=True)
        if not candidate_sets:
            return False, "no_policy_target", 0
        pool = _candidate_local_removal_pool(hp, policy_min_edge_conflict_replace_pool_cap)
        removal_sets = [set()]
        if policy_min_edge_conflict_replace_max_remove > 0 and pool:
            removal_sets.extend(_bounded_removal_combos(
                pool,
                policy_min_edge_conflict_replace_max_remove,
                policy_min_edge_conflict_replace_beam_width,
            ))
        # More conservative target sets first. The target generator already
        # orders follower/current/midpoint candidates deterministically.
        trials = 0
        solve_failed = 0
        post_delta_failed = 0
        collector_failed = 0
        last_detail = ""
        for cons in candidate_sets:
            cons = _dedup_fixed_constraints(cons)
            if not cons:
                continue
            cons_pins = {pin for pin, _ in cons}
            cons_vars = [_template_var_key_for_pin(pin) for pin, _ in cons]
            cons_desc = ",".join(f"{pin}->{float(target):.6f}" for pin, target in cons)
            for rem_set in removal_sets:
                trials += 1
                snap = _snapshot_all_state()
                try:
                    hard_pairs_active = set(hard_pairs_active) - set(rem_set)
                    _, removed_anchors = _remove_policy_anchors_touching_pins(cons_pins)
                    trial_constraints = required_anchor_active + cons
                    ok, err = _attempt_required_anchor_constraints(
                        trial_constraints,
                        order_hint_constraints=cons if policy_min_edge_target_order_hint else None,
                    )
                    last_detail = (
                        f"cons=[{cons_desc}] rem={len(rem_set)} removed_anchors={removed_anchors} "
                        f"vars={cons_vars}"
                    )
                    if ok and _edge_aligned_after_solve(hp):
                        required_anchor_active = list(trial_constraints)
                        dropped_mates = _drop_policy_component_mates_for_kept(
                            rec,
                            reason="policy_component_representative_aligned_after_conflict_replace",
                        )
                        # Collector-level postcheck: the accepted representative
                        # must actually disappear from the same final blocker
                        # collector used by reporting.  If not, roll back the
                        # anchors and any policy drops from this trial.
                        if not _collector_still_contains_edge(rec):
                            return True, (
                                f"remove_{len(rem_set)};removed_anchors={removed_anchors};"
                                f"drop_mates={dropped_mates};vars={cons_vars}"
                            ), len(rem_set)
                        collector_failed += 1
                        err = RuntimeError(
                            f"collector_postcheck_failed: delta={_post_delta(hp):.6f} "
                            f"tol={tol:.6f} drop_mates={dropped_mates}"
                        )
                    # Feasible is not enough: the requested representative must
                    # actually become aligned and disappear from the final
                    # residual collector.  Otherwise roll back the trial.
                    elif ok:
                        post_delta_failed += 1
                        err = RuntimeError(f"post_delta_not_aligned: delta={_post_delta(hp):.6f} tol={tol:.6f}")
                    else:
                        solve_failed += 1
                    if err is not None:
                        msg = str(err).replace("\n", " ")
                        if len(msg) > 180:
                            msg = msg[:177] + "..."
                        last_detail += f" err={msg}"
                    _restore_all_state(snap)
                except Exception as e:
                    solve_failed += 1
                    _restore_all_state(snap)
                    msg = str(e).replace("\n", " ")
                    if len(msg) > 180:
                        msg = msg[:177] + "..."
                    last_detail = f"cons=[{cons_desc}] rem={len(rem_set)} vars={cons_vars} err={msg}"
        return False, (
            "blocked_under_conflict_replacement;"
            f"trials={trials};solve_failed={solve_failed};"
            f"post_delta_failed={post_delta_failed};collector_failed={collector_failed};"
            f"last={last_detail}"
        ), 0

    def _run_policy_min_edge_conflict_replacement():
        """Final rescue for remaining low-HPWL policy representatives.

        The policy-component pass may reduce each unresolved component to one
        HPWL-ordered representative. If that representative is still misaligned,
        try to make it exact by replacing local active constraints. This pass is
        feasibility-preserving: failed trials restore the full accepted state.
        """
        if not policy_min_edge_conflict_replace_enable:
            return 0, len(_collect_true_same_axis_blockers_report_style(include_active=True))
        total_kept = 0
        total_removed = 0
        for rr in range(max(0, policy_min_edge_conflict_replace_rounds)):
            residual = _collect_true_same_axis_blockers_report_style(include_active=True)
            policy_residual = [
                rec for rec in residual
                if not _hard_pair_allowed_by_ref_master_policy(rec['hp'])
                and float(rec.get('hpwl', _compute_hpwl(rec['hp']))) <= policy_component_fallback_hpwl_limit + 1e-9
            ]
            if not policy_residual:
                break
            policy_residual.sort(key=lambda rec: (
                float(rec.get('hpwl', _compute_hpwl(rec['hp']))),
                float(rec.get('fixed_axis_distance', _fixed_axis_distance(rec['hp']))),
                -_edge_width(rec['hp']),
                str(rec.get('edge_key', rec['hp'].as_undirected_key())),
            ))
            kept = 0
            removed = 0
            tried = 0
            post_rejected = 0
            collector_rejected = 0
            fail_reasons = defaultdict(int)
            details = []
            for rec in policy_residual:
                tried += 1
                ok, why, nrem = _try_policy_edge_anchor_with_conflict_replacement(rec)
                hp = rec['hp']
                edge_key = rec.get('edge_key', hp.as_undirected_key())
                if ok:
                    kept += 1
                    removed += int(nrem)
                    total_kept += 1
                    total_removed += int(nrem)
                    details.append(("accepted", rec, why))
                else:
                    reason_key = str(why).split(';', 1)[0]
                    fail_reasons[reason_key] += 1
                    if "post_delta_not_aligned" in str(why):
                        post_rejected += 1
                    elif "collector_postcheck_failed" in str(why):
                        collector_rejected += 1
                    details.append(("rejected", rec, why))
            residual_after = len(_collect_true_same_axis_blockers_report_style(include_active=True))
            if solver_progress_log:
                fail_desc = ""
                if fail_reasons:
                    fail_desc = " fail=" + ",".join(f"{k}={v}" for k, v in sorted(fail_reasons.items()))
                print(
                    f"[policy-min-conflict] round={rr + 1} candidates={len(policy_residual)} "
                    f"tried={tried} kept_aligned={kept} post_delta_rejected={post_rejected} "
                    f"collector_rejected={collector_rejected} "
                    f"removed_hard_pairs={removed} residual_after={residual_after} "
                    f"hpwl_limit={policy_component_fallback_hpwl_limit:g}{fail_desc}"
                )
                if policy_min_edge_conflict_verbose and details:
                    for status, rec, why in details[:max(0, policy_min_edge_conflict_detail_limit)]:
                        hp = rec['hp']
                        iv_lo = rec.get('lo', float('nan'))
                        iv_hi = rec.get('hi', float('nan'))
                        print(
                            f"[policy-min-conflict-detail] {status} edge={rec.get('edge_key', hp.as_undirected_key())} "
                            f"axis={hp.axis} delta={_alignment_delta(hp):.6f} hpwl={_compute_hpwl(hp):.6f} "
                            f"iv=[{float(iv_lo):.6f},{float(iv_hi):.6f}] "
                            f"vars=({_template_var_key_for_pin(hp.pin_a)}, {_template_var_key_for_pin(hp.pin_b)}) "
                            f"why={why}"
                        )
            if kept == 0:
                break
            if residual_after == 0:
                break
        return total_kept, len(_collect_true_same_axis_blockers_report_style(include_active=True))

    def _run_required_hard_closure_pass():
        """Terminal hard closure for actionable same-axis residuals.

        Goal: reduce true same-axis misalignment to zero under the current policy.
        Cross-axis, pair-empty, and component-empty dropped-by-policy edges are
        already excluded by the canonical collector.  Remaining ordinary edges are
        forced as equality constraints if feasible; reference-master-blocked edges
        are forced through follower-anchor hard scalar constraints when possible.
        """
        if not required_hard_closure_enable:
            _diagnose_final_blockers()
            return
        total_direct = 0
        total_replaced = 0
        total_anchor = 0
        last_sig = None
        for rr in range(max(0, required_hard_closure_rounds)):
            residual_recs = _collect_true_same_axis_blockers_report_style(include_active=True)
            if not residual_recs:
                if solver_progress_log:
                    print(f"[required-closure] round={rr + 1} residual=0")
                break
            sig = tuple((rec['edge_key'], round(float(rec['delta']), 6)) for rec in residual_recs)
            if sig == last_sig and rr > 0:
                break
            last_sig = sig

            ordinary = [rec for rec in residual_recs if _hard_pair_allowed_by_ref_master_policy(rec['hp'])]
            policy = [rec for rec in residual_recs if not _hard_pair_allowed_by_ref_master_policy(rec['hp'])]

            round_direct = 0
            round_replaced = 0
            round_failed = 0
            # Force ordinary blockers first.  This may move follower anchors, so
            # follower-anchor closure is deliberately delayed until after ordinary
            # hard-pair closure has no more progress.
            for rec in ordinary:
                hp = rec['hp']
                if hp in hard_pairs_active and _alignment_delta(hp) <= tol:
                    continue
                ok, why = _commit_required_ordinary_blocker(hp)
                if ok:
                    if str(why).startswith('remove_'):
                        round_replaced += 1
                    else:
                        round_direct += 1
                else:
                    round_failed += 1
            total_direct += round_direct
            total_replaced += round_replaced
            remaining_after_ordinary = _collect_true_same_axis_blockers_report_style(include_active=True)
            remaining_policy = [rec for rec in remaining_after_ordinary if not _hard_pair_allowed_by_ref_master_policy(rec['hp'])]

            # First handle E->A/B/C/D target edges by the dedicated fixed-target
            # rule: the homology endpoint is frozen; only the external/free side
            # moves.  This prevents E--B from becoming a generic component-level
            # attempt to move B or its master template.
            free_target_kept = free_target_rejected = free_target_skipped = 0
            if round_direct == 0 and round_replaced == 0 and remaining_policy:
                free_target_records = [rec for rec in remaining_policy if rec.get('policy_class') == "free_to_homology_target"]
                free_target_kept, free_target_rejected, free_target_skipped = _run_free_to_homology_target_closure(free_target_records)

            remaining_after_free_target = _collect_true_same_axis_blockers_report_style(include_active=True)
            remaining_policy = [
                rec for rec in remaining_after_free_target
                if (not _hard_pair_allowed_by_ref_master_policy(rec['hp']))
                and rec.get('policy_class') != "free_to_homology_target"
            ]

            # Force the remaining non-free-target policy pairs through hard anchors.
            anchor_kept = anchor_rejected = anchor_skipped = 0
            if round_direct == 0 and round_replaced == 0 and remaining_policy:
                anchor_kept, anchor_rejected, anchor_skipped = _run_required_anchor_closure(remaining_policy)
                total_anchor += anchor_kept
            total_anchor += free_target_kept

            residual_now = len(_collect_true_same_axis_blockers_report_style(include_active=True))
            if solver_progress_log:
                print(
                    f"[required-closure] round={rr + 1} residual_before={len(residual_recs)} "
                    f"ordinary={len(ordinary)} policy={len(policy)} direct={round_direct} "
                    f"replaced={round_replaced} failed={round_failed} "
                    f"free_target_kept={free_target_kept} free_target_rejected={free_target_rejected} "
                    f"free_target_skipped={free_target_skipped} "
                    f"anchor_kept={anchor_kept} anchor_rejected={anchor_rejected} "
                    f"anchor_skipped={anchor_skipped} residual_after={residual_now}"
                )
            if residual_now == 0:
                break
            if round_direct == 0 and round_replaced == 0 and free_target_kept == 0 and anchor_kept == 0:
                break
        conflict_kept, conflict_residual = _run_policy_min_edge_conflict_replacement()
        if solver_progress_log:
            final_res = len(_collect_true_same_axis_blockers_report_style(include_active=True))
            print(
                f"[required-closure] total_direct={total_direct} total_replaced={total_replaced} "
                f"total_anchor={total_anchor} conflict_kept={conflict_kept} residual_after={final_res}"
            )
        _diagnose_final_blockers()

    bucket_by_seg = {}

    for outer_it in range(max_outer_iter):
        old_snap = snapshot_template(groups)

        if enable_hard_iso:
            inst_pin_state, bucket_by_seg = expand_template_to_instances(groups, real_segments, keepout)
        elif inst_pin_state is None:
            inst_pin_state, bucket_by_seg = build_inst_state_from_active_pins(active_pins)
        else:
            bucket_by_seg = {}
            for key, st in inst_pin_state.items():
                bucket_by_seg.setdefault(st.seg_id, []).append(key)

        inst_snap = snapshot_inst_state(inst_pin_state)
        if enable_hard_iso:
            orders, cyclic_keys = build_orders_hard_iso(groups, bucket_by_seg, hard_pairs_active, inst_pin_state, real_segments, _template_order_pairs())
            if cyclic_keys:
                _debug_print(f"[outer] cyclic template precedence on {len(cyclic_keys)} canonical segments; local fallback may be used")
        else:
            orders = build_orders(bucket_by_seg, hard_pairs_active, inst_pin_state)

        prev_values = snapshot_template(groups) if enable_hard_iso else inst_snap
        solve_inner_qp_dummy(groups, inst_pin_state, orders, hard_pairs_active,
                             enable_hard_iso=enable_hard_iso, keepout=keepout,
                             real_segments=real_segments, prev_values=prev_values,
                             init_values=initial_real_anchor,
                             fixed_scalar_constraints=required_anchor_active)

        _refresh_bucket_from_state()
        newly_triggered = _build_triggered_pairs(apply_component_gate=True)

        admitted = 0
        rejected = 0
        if newly_triggered:
            active_candidates = [hp for hp in newly_triggered if hp not in hard_pairs_active]
            if active_candidates:
                admitted, rejected = _run_family_admission(
                    active_candidates,
                    search_cap=family_search_cap,
                    allow_family_swap=enable_family_swap,
                    swap_cap=1,
                    removal_pool_cap=8,
                    beam_width=24,
                )

        no_new_pairs = (admitted == 0)
        if not no_new_pairs:
            prev_orders = orders
            continue

        order_stable = (prev_orders is not None and orders_equal(orders, prev_orders))
        if enable_hard_iso:
            max_change = template_diff(groups, old_snap)
            if no_new_pairs and order_stable and max_change < tol:
                break
        else:
            state_change = inst_state_diff(inst_pin_state, inst_snap)
            if no_new_pairs and order_stable and state_change < tol:
                break
        prev_orders = orders

    # final enforcement
    _refresh_bucket_from_state()

    if enable_hard_iso:
        orders, cyclic_keys = build_orders_hard_iso(groups, bucket_by_seg, hard_pairs_active, inst_pin_state, real_segments, _template_order_pairs())
        if cyclic_keys:
            raise RuntimeError(f"cyclic template precedence in final enforcement: {sorted(cyclic_keys)}")
    else:
        orders = build_orders(bucket_by_seg, hard_pairs_active, inst_pin_state)
    prev_values = snapshot_template(groups) if enable_hard_iso else snapshot_inst_state(inst_pin_state)
    solve_inner_qp_dummy(groups, inst_pin_state, orders, hard_pairs_active,
                         enable_hard_iso=enable_hard_iso, keepout=keepout,
                         real_segments=real_segments, prev_values=prev_values,
                             init_values=initial_real_anchor,
                             fixed_scalar_constraints=required_anchor_active)

    if enable_hard_iso:
        inst_pin_state, bucket_by_seg = expand_template_to_instances(groups, real_segments, keepout)

    # Do not run follower-anchor closure before the adaptive gate.  It can be
    # expensive because it performs many trial QP solves, and old/easy datasets
    # may already have zero actionable residuals after the ordinary fast path.
    # Policy-blocked residuals are handled inside the strict branch below.

    # Adaptive strict closure gate.  This must run before any expensive final
    # rescue / required-closure search.  If the report-style collector already
    # sees no actionable same-axis blockers, return through the final report path
    # without extra GUROBI trial loops.
    residual_before_strict = len(_collect_true_same_axis_blockers_report_style(include_active=True))
    if solver_progress_log:
        print(f"[adaptive] residual_before_strict={residual_before_strict} adaptive={int(adaptive_strict_closure)}")

    if (not adaptive_strict_closure) or residual_before_strict > 0:
        # Terminal hard closure first: this is usually cheaper and more decisive
        # than generic final rescue for policy-blocked residuals.
        _run_required_hard_closure_pass()

        residual_after_required = len(_collect_true_same_axis_blockers_report_style(include_active=True))
        if residual_after_required > 0:
            # Revisit residual same-axis blockers under the final geometry only
            # when strict closure leaves real blockers.
            before_rescue_score = _residual_score()
            _run_final_rescue_pass()
            after_rescue_score = _residual_score()
            if final_rescue_count_gate and after_rescue_score[0] >= before_rescue_score[0]:
                if solver_progress_log:
                    print(f"[adaptive] final-rescue count gate: before={before_rescue_score} after={after_rescue_score}")

            # One more required closure pass may convert rescue-improved geometry
            # into exact hard equality / policy anchor constraints.
            if len(_collect_true_same_axis_blockers_report_style(include_active=True)) > 0:
                _run_required_hard_closure_pass()
    else:
        if solver_progress_log:
            print("[adaptive] skip strict closure: residual=0")
        _diagnose_final_blockers()

    if enable_hard_iso:
        inst_pin_state, bucket_by_seg = expand_template_to_instances(groups, real_segments, keepout)

    LAST_POLICY_COMPONENT_DROPPED_EDGE_KEYS = set(policy_component_dropped_edge_keys) | set(policy_free_target_dropped_edge_keys)
    LAST_POLICY_COMPONENT_DROPPED_RECORDS = list(policy_component_dropped_records) + list(policy_free_target_dropped_records)
    LAST_POLICY_FREE_TARGET_DROPPED_EDGE_KEYS = set(policy_free_target_dropped_edge_keys)
    LAST_POLICY_FREE_TARGET_DROPPED_RECORDS = list(policy_free_target_dropped_records)

    if strict_pair_hpwl_gate:
        _debug_print(f"[outer] pair-hpwl gate total_skipped={hpwl_gate_skipped_total} threshold={hpwl_thresh:g} margin={pair_hpwl_gate_margin:g} debug={int(hpwl_gate_debug)}")
    return inst_pin_state, real_segments
