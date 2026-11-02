"""The arms: per-arm coefficient maps for one target triple.

A map is {(fold, role): coefficients} or, for a leave-pair-out arm, also
{(fold, pair, role): coefficients}; fold f's entries are what is applied to the target's fold-f
events. self_coeff_maps builds the target's own arms (notwc, self_in, self_cf, self_cf_lpo,
placebo_self), transferred_map a transfer arm from a source's coefficients, xfer_lin_map and
xfer_lin_lpo_map the shape-versus-slope arm of SPEC.md 4 and its guard.
"""
import hashlib
from itertools import combinations

import numpy as np

from ._core import ORDER, TS
from .bank import lpo_events
from .campaigns import N_CAP
from .events import event_keys
from .solve import solve

# The arm names. ARMS_CORE are the arms a cell evaluates by default (SPEC.md 4); notwc_1b is
# the uncorrected reference of a one-board cell: board --xfer-board set to zero, every other
# board on its own fold-matched self_cf. It is built only when --xfer-board is given, because
# only there is SPEC 5.2's "the placebo must reproduce notwc" a comparison against something
# other than the plain notwc arm.
ARMS_CORE = ("notwc", "self_in", "self_cf", "self_cf_lpo", "placebo_self",
             "xfer", "xfer_lpo", "placebo",
             "notwc_1b")
# ARMS_EXT are the extended arms (--arms ext). xfer_lin and its leave-pair-out guard are the
# shape-versus-slope re-score of SPEC.md 4; they are listed here for documentation only, the
# gate that builds them is --xfer-lin.
ARMS_EXT = ("xfer_clip", "xfer_shift", "xfer_rescale", "xfer_calfix", "self_in_rep",
            "xfer_lin", "xfer_lin_lpo")


class ArmMaps(dict):
    """arm -> coefficient map, carrying the source event keys of each arm.

    A plain dict subclass so every existing caller (and every verifier script) that
    treats it as `{arm: cmap}` keeps working; `src_keys[arm]` is what makes the SPEC 1
    invariant checkable, and `in_sample` names the arms that are deliberately fitted
    on the evaluated events (`self_in`, its repeat probe) and are therefore exempt.
    """

    def __init__(self, *a, **kw):
        dict.__init__(self, *a, **kw)
        self.src_keys = {}
        self.in_sample = {"self_in", "self_in_rep"}
        self.lpo_shape = {}    # {drop_role: "sibling" | "own"}, see self_coeff_maps


def self_coeff_maps(tri, folds=(0, 1), root=None, want_lpo=True, neighbor="none",
                    neighbor_logic="OR", lpo_fallback=None):
    """notwc / self_in / self_cf / self_cf_lpo / placebo_self maps for one target.

    `self_cf` IS the self-transfer null of SPEC 5.5 (the source is the complementary
    fold of the same triple), which is why it is also the baseline.

    `lpo_fallback` (SPEC 5.1): `None`/`"sibling"` (default) is the all-or-nothing
    rule - if any of the three dropped roles has no same-pixel sibling combo,
    `self_cf_lpo` is not built AT ALL.  `"own"` builds the guard PER DROP instead:
    where the sibling
    is unavailable, the dropped role's excluded-partner coefficients come from a
    JOINT SOLVE ON THE TARGET'S OWN FOLD-MATCHED EVENTS restricted to the two
    REMAINING roles only - a design that never sees the dropped role's TOT/TOA at
    all, which is what "a solve that excluded board b" (SPEC 5.1) means, built from
    a 2-board design instead of a 3-board sibling combo.  The cheat statistic stays
    well-posed (SPEC 5.1: it regroups into two brackets that each difference arms
    identical on every unguarded board, whatever built the guarded entries); a
    2-board guard is noisier, which SPEC 5.1's `uninformative` rule censors.
    `maps.lpo_shape` records which construction was actually used for each dropped
    role.
    """
    zero = np.zeros(ORDER + 1)
    maps = ArmMaps({"notwc": {}, "self_in": {}, "self_cf": {}, "placebo_self": {}})
    keys_cf = {}
    idx_src_by_fold = {}
    c_in = solve(tri.tot, tri.toa, tri.roles)
    for f in folds:
        other = [g for g in folds if g != f]
        idx_src = np.concatenate([tri.idx_fold(g) for g in other]) if other else np.arange(tri.n)
        idx_src_by_fold[f] = idx_src
        c_cf = solve(tri.tot, tri.toa, tri.roles, idx=idx_src)
        # content-addressed seed (blake2b of the track name and the fold), as in
        # bank_rows_for_triple: independent of Python's per-process hash() salt
        rng = np.random.default_rng(
            int(hashlib.blake2b(repr((tri.name, "self_placebo", f)).encode(), digest_size=8).hexdigest(), 16))
        c_pl = solve(tri.tot, tri.toa, tri.roles, idx=idx_src, permute=True, rng=rng)
        keys_cf[f] = [tri.key_entry(idx_src)]
        for r in tri.roles:
            maps["notwc"][(f, r)] = zero
            maps["self_in"][(f, r)] = c_in[r]
            maps["self_cf"][(f, r)] = c_cf[r]
            maps["placebo_self"][(f, r)] = c_pl[r]
    maps.src_keys["notwc"] = None          # no source: nothing was fitted
    maps.src_keys["self_in"] = None        # deliberately in-sample (optimism arm)
    maps.src_keys["self_cf"] = keys_cf
    maps.src_keys["placebo_self"] = keys_cf
    if want_lpo:
        if lpo_fallback != "own":
            # ---- default: all-or-nothing on the sibling combo; see the `else`
            # branch below for the per-drop fallback.
            lpo = {}
            ok = True
            for drop in tri.roles:
                got = lpo_events(tri, drop, cap=N_CAP, root=root, neighbor=neighbor,
                                 neighbor_logic=neighbor_logic)
                if got is None:
                    ok = False
                    break
                lpo[drop] = got
            if ok:
                m = dict(maps["self_cf"])          # fallback for own-slope bookkeeping
                klpo = {f: list(keys_cf.get(f, [])) for f in folds}
                for f in folds:
                    for drop, (ltot, ltoa, lfile, lroles, sib, nl) in lpo.items():
                        keep = np.where((lfile % 2) != f)[0] if len(folds) > 1 else np.arange(len(lfile))
                        if len(keep) < 50:
                            continue
                        c = TS.solve_timewalk_coefficients({r: ltot[r][keep] for r in lroles},
                                                           {r: ltoa[r][keep] for r in lroles}, lroles)
                        kept = tuple(sorted(r for r in tri.roles if r != drop))
                        klpo[f].append((kept, event_keys(lfile[keep],
                                                         {r: ltoa[r][keep] for r in lroles}, kept)))
                        for a, b in combinations(tri.roles, 2):
                            pair = "%s-%s" % (a, b)
                            if drop == b and a in c:
                                m[(f, pair, a)] = c[a]
                            if drop == a and b in c:
                                m[(f, pair, b)] = c[b]
                maps["self_cf_lpo"] = m
                maps.src_keys["self_cf_lpo"] = klpo
                maps.lpo_shape = {d: "sibling" for d in tri.roles}
        else:
            # ---- --lpo-fallback own: per-drop fallback.  A drop whose sibling combo
            # IS available still uses it (the most informative source); a drop whose
            # sibling is missing gets the target's-own-events 2-role solve instead of
            # voiding the whole triple's guard.
            lpo_sib, lpo_shape = {}, {}
            for drop in tri.roles:
                got = lpo_events(tri, drop, cap=N_CAP, root=root, neighbor=neighbor,
                                 neighbor_logic=neighbor_logic)
                if got is not None:
                    lpo_sib[drop] = got
                    lpo_shape[drop] = "sibling"
                else:
                    lpo_shape[drop] = "own"
            if lpo_shape:
                m = dict(maps["self_cf"])
                klpo = {f: list(keys_cf.get(f, [])) for f in folds}
                for f in folds:
                    idx_src = idx_src_by_fold[f]
                    for drop in tri.roles:
                        if lpo_shape[drop] == "sibling":
                            ltot, ltoa, lfile, lroles, sib, nl = lpo_sib[drop]
                            keep = (np.where((lfile % 2) != f)[0] if len(folds) > 1
                                   else np.arange(len(lfile)))
                            if len(keep) < 50:
                                continue
                            c = TS.solve_timewalk_coefficients(
                                {r: ltot[r][keep] for r in lroles},
                                {r: ltoa[r][keep] for r in lroles}, lroles)
                            kept = tuple(sorted(r for r in tri.roles if r != drop))
                            klpo[f].append((kept, event_keys(
                                lfile[keep], {r: ltoa[r][keep] for r in lroles}, kept)))
                        else:
                            # target's own fold-matched events, 2-role design: this
                            # solve's TOA/TOT design never includes `drop` at all, so
                            # both remaining roles' coefficients are, by construction,
                            # fitted excluding `drop` - exactly SPEC 5.1's guard, just
                            # without a sibling combo to supply the fourth board.
                            remaining = [r for r in tri.roles if r != drop]
                            if len(idx_src) < 50:
                                continue
                            c = solve(tri.tot, tri.toa, remaining, idx=idx_src)
                            kept = tuple(sorted(remaining))
                            klpo[f].append(tri.key_entry(idx_src, kept))
                        for a, b in combinations(tri.roles, 2):
                            pair = "%s-%s" % (a, b)
                            if drop == b and a in c:
                                m[(f, pair, a)] = c[a]
                            if drop == a and b in c:
                                m[(f, pair, b)] = c[b]
                maps["self_cf_lpo"] = m
                maps.src_keys["self_cf_lpo"] = klpo
            maps.lpo_shape = lpo_shape
    return maps


def transferred_map(tri, src_coeffs, folds=(0, 1), base=None, roles_sub=None, variant="plain",
                    src_row=None, tri_stats=None):
    """A transfer arm's coefficient map.

    src_coeffs : {role: raw-TOT coefficients} from the SOURCE (bank row or fit)
    base       : the fold-matched self map used for the boards NOT being replaced
                 (one_board diagnostics); None replaces all three.
    variant    : plain | clip | shift | rescale | calfix   (SPEC 4 extended arms)
    """
    out = {}
    roles_sub = roles_sub or tri.roles
    missing = [r for r in tri.roles if r in roles_sub and r not in src_coeffs]
    if missing and base is None:
        # SPEC 3: a whole-triple transfer needs a coefficient for every board.
        # Filling the gap with zeros would silently turn the missing board into
        # `notwc` and charge the difference to the transfer.  A missing board is a
        # SKIP, never a datum.
        raise KeyError("transferred_map: source has no coefficients for board(s) %s "
                       "(has %s)" % (sorted(missing), sorted(src_coeffs)))
    for f in folds:
        for r in tri.roles:
            if r in roles_sub and r in src_coeffs:
                out[(f, r)] = np.asarray(src_coeffs[r], float)
            elif base is not None:
                out[(f, r)] = base[(f, r)]
            else:                                   # unreachable: guarded above
                raise KeyError("transferred_map: no coefficients for board %s" % r)
    return out


def xfer_lin_map(tri, src_coeffs, folds=(0, 1)):
    """The `xfer_lin` arm (SPEC 4, axis F): shape versus slope.

    The source's QUADRATIC coefficient (index 0 of the raw-TOT `[c2, c1, c0]`
    triple) is held fixed at its transferred value for every board; the LINEAR
    term is refit fold-matched on the TARGET's own events, jointly across all
    three boards, with the SAME unmodified `TS.solve_timewalk_coefficients` the
    rest of this module uses - just called at `order=1` instead of `ORDER` (2).

    Algebra ("the t^2 column moved to the offset"): the apply line is always
    `toa_r + polyval(c_r, tot_r)`, so for
    `c_r = [c2_r, c1_r, c0_r]` with `c2_r` FIXED,

        toa_r + c2_r*tot_r**2 + c1_r*tot_r + c0_r
      = (toa_r + c2_r*tot_r**2) + c1_r*tot_r + c0_r

    i.e. solving for (c1_r, c0_r) alone against `toa_r + c2_r*tot_r**2` at order=1
    is the SAME joint least-squares problem the order-2 solve poses for (c1_r,
    c0_r) given a known c2_r: one call to
    `TS.solve_timewalk_coefficients(tot, toa_tilde, roles, order=1)`.  The sign is
    `+`, matched to the apply line's own ADD convention; `toa - c2*tot**2` would
    invert the sign of the shape term being held fixed (SPEC 4).

    `folds` are handled exactly as `self_coeff_maps`'s `self_cf`: fold f's
    coefficients are fit on the COMPLEMENTARY fold's events only, so `xfer_lin`
    is fold-matched like every other arm and carries no in-sample optimism.

    Returns `{(f, role): ndarray[3]}`, the same shape as any other arm map.
    """
    out = {}
    for f in folds:
        other = [g for g in folds if g != f]
        idx = np.concatenate([tri.idx_fold(g) for g in other]) if other else np.arange(tri.n)
        tot_f = {r: tri.tot[r][idx] for r in tri.roles}
        toa_f = {r: tri.toa[r][idx] for r in tri.roles}
        c2 = {r: float(np.asarray(src_coeffs[r], float)[0]) for r in tri.roles}
        toa_tilde = {r: toa_f[r] + c2[r] * tot_f[r] ** 2 for r in tri.roles}
        c1 = TS.solve_timewalk_coefficients(tot_f, toa_tilde, tri.roles, order=1)
        for r in tri.roles:
            out[(f, r)] = np.array([c2[r], float(c1[r][0]), float(c1[r][1])])
    return out


def xfer_lin_lpo_map(tri, src_coeffs, lpo, folds=(0, 1)):
    """`xfer_lin`'s own leave-pair-out guard (axis F requirement 3: "the arm is
    unguarded" otherwise).

    For target pair (x, y), x's contribution must come from a source solve that
    EXCLUDED y - `lpo[y][x]`, exactly SPEC 5.1's rule for `xfer_lpo`.

    Swapping only the guarded board's quadratic term and REUSING the plain fit's
    (c1, c0) would leave the guard partial: those come from a joint solve in which
    board y's OWN fixed c2 (and hence its `toa_tilde`) entered the SAME pairwise
    design that determined x's (c1, c0).  So this is the genuine minimiser:
    a fresh `xfer_lin_map` call per (pair, direction) with ONLY the guarded
    board's c2 replaced by `lpo[y][x][0]` (every other board keeps its plain,
    unguarded `src_coeffs`), keeping just the guarded board's resulting
    (c1, c0) from THAT solve - 3 pairs x 2 directions x 2 folds = 12 fresh joint
    solves per triple (6 per fold), each exactly as cheap as the plain
    `xfer_lin` solve it replaces one board's c2 in.

    Returns `{(f, pair, role): ndarray[3]}`, SEEDED from the plain `xfer_lin_map`,
    so `resolve_coeff`'s fallback to `cmap[(fold, role)]` finds a board-level entry
    for the pair's OTHER role.

    Residual coupling, stated rather than assumed away: for pair (x, y) guarded
    on x (`a_ = x, b_ = y`), the joint solve inside this call still uses y's own
    UNGUARDED `src_coeffs[y]` to build y's `toa_tilde`, and the pairwise least
    squares couples every board's (c1, c0) through the shared design - so board
    y's (excluded) source events still enter x's fitted (c1, c0) at SECOND order,
    through y's fixed contribution to the pair residual x is fit against. This is
    strictly a smaller leak than doing nothing (x's own c2 no longer depends on y
    AT ALL, and x's c1/c0 depend on y only through a value y's OWN self_cf/xfer
    already carries in every other arm), and it is a real limitation `xfer_lpo`
    does not share: `xfer_lpo`'s guarded coefficient is copied verbatim from a
    source fit that excluded y, with no target-side re-solve to reintroduce y at
    all.
    """
    out = dict(xfer_lin_map(tri, src_coeffs, folds=folds))
    for x, y in combinations(tri.roles, 2):
        pair = "%s-%s" % (x, y)
        for a_, b_ in ((x, y), (y, x)):
            c_g = (lpo.get(b_) or {}).get(a_)
            if c_g is None:
                continue
            guarded = dict(src_coeffs)
            guarded[a_] = np.asarray(c_g, float)
            m_g = xfer_lin_map(tri, guarded, folds=folds)
            for f in folds:
                out[(f, pair, a_)] = m_g[(f, a_)]
    return out
