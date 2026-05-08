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
  - Candidate admission uses a dynamic HPWL gate and deterministic trial-failure
    cache to avoid repeated infeasible GUROBI solves.

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
                         keepout=0.0, real_segments=None, prev_values=None, init_values=None):
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
    for s in ["GUROBI", "CLARABEL", "OSQP", "SCS"]:
        if s in installed:
            preferred_solvers.append(s)
    if not preferred_solvers:
        raise RuntimeError("No supported CVXPY solver available (tried GUROBI, CLARABEL, OSQP, SCS)")

    solved = False
    last_err = None
    last_status = None

    def _acceptable(status):
        return status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

    _SOLVER_KWARGS = {
        "GUROBI":   {"verbose": False, "reoptimize": True, "warm_start": True},
        "CLARABEL": {"verbose": False},
        "OSQP":     {"verbose": False, "warm_start": True, "max_iter": 10000, "eps_abs": 1e-6, "eps_rel": 1e-6},
        "SCS":      {"verbose": False, "max_iters": 10000, "eps": 1e-6},
    }

    for solver_name in preferred_solvers:
        try:
            kwargs = _SOLVER_KWARGS.get(solver_name, {"verbose": False})
            prob.solve(solver=getattr(cp, solver_name), **kwargs)
            last_status = prob.status
            _debug_print(f"[inner] solver {solver_name} status {prob.status}")
            if _acceptable(prob.status):
                solved = True
                break
        except Exception as e:
            last_err = e
            _debug_print(f"[inner] solver {solver_name} exception {repr(e)}")
            if solver_name == "GUROBI" and gurobi_verbose_retry:
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


    def _is_master_inst_for_policy(inst_name):
        G = _find_group_of_inst(groups, inst_name)
        return G is None or inst_name == G.canon_inst

    def _hard_pair_allowed_by_ref_master_policy(hp):
        if not enable_hard_iso or follower_hardpairs_affect_master:
            return True
        return _is_master_inst_for_policy(hp.pin_a[0]) and _is_master_inst_for_policy(hp.pin_b[0])

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
                                     init_values=initial_real_anchor)
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
            newly_triggered = [hp for hp in newly_triggered if _hard_pair_allowed_by_ref_master_policy(hp)]
            dropped_policy = before_policy - len(newly_triggered)
            if dropped_policy:
                _debug_print(f"[iso-master] dropped follower-side hard_pairs that would pull master: {dropped_policy}")

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
                    best = min(edges, key=_compute_hpwl)
                    kept.append(best)
            if dropped_components:
                _debug_print(f"[outer] component-level infeasible hard-align components: {dropped_components}; kept only shortest-hpwl edge per such component")
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
            key=lambda kv: (min(_compute_hpwl(hp) for hp in kv[1]), len(kv[1]))
        )

        total_candidates = len(candidates)
        for fam_key, fam_candidates in family_order:
            family_active = [hp for hp in hard_pairs_active if _family_key(hp) == fam_key]
            union_edges = []
            seen_union = set()
            for hp in sorted(family_active + fam_candidates, key=_compute_hpwl):
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
                        sum(_compute_hpwl(hp) for hp in combo),
                        -sum(1 for hp in combo if hp in base_active_set),
                    ))
                    found = False
                    for combo in combos:
                        combo_set = set(combo)
                        ok, err = _attempt_replace(base_active_set, combo_set)
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
                ordered_candidates = sorted(fam_candidates, key=_compute_hpwl)

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

                    ok, err = _attempt_replace(current_family_set, current_family_set | {hp})
                    if ok:
                        current_family_set = {e for e in hard_pairs_active if _family_key(e) == fam_key}
                        continue

                    if allow_family_swap and current_family_set:
                        swapped = False
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
                                score = (
                                    sum(_compute_hpwl(e) for e in rem_set),
                                    sum(_alignment_delta(e) for e in rem_set),
                                )
                                trial_removals.append((score, rem_set))
                        trial_removals.sort(key=lambda item: item[0], reverse=True)
                        for _, rem_set in trial_removals[:max(1, beam_width)]:
                            trial_set = (current_family_set - rem_set) | {hp}
                            ok2, err2 = _attempt_replace(current_family_set, trial_set)
                            if ok2:
                                current_family_set = {e for e in hard_pairs_active if _family_key(e) == fam_key}
                                swapped = True
                                break
                            if chosen_err is None:
                                chosen_err = err2
                        if not swapped and chosen_err is None:
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
                             init_values=initial_real_anchor)

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
                             init_values=initial_real_anchor)

    if enable_hard_iso:
        inst_pin_state, bucket_by_seg = expand_template_to_instances(groups, real_segments, keepout)

    if strict_pair_hpwl_gate:
        _debug_print(f"[outer] pair-hpwl gate total_skipped={hpwl_gate_skipped_total} threshold={hpwl_thresh:g} margin={pair_hpwl_gate_margin:g} debug={int(hpwl_gate_debug)}")
    return inst_pin_state, real_segments
