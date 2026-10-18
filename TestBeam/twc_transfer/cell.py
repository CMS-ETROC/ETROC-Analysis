"""One transfer cell (SPEC.md 3): the source of every target triple, the arms built from it,
and the resumable cell table (SPEC.md 7).

A cell is a mode, a target (campaign, run, combo, a list of target triples) and a source. For
every target triple the cell resolves the source coefficients, builds the target's own arms
(arms.self_coeff_maps) and the transfer arms from the source, evaluates every arm on the same
frozen events (widths.evaluate_arm, which also runs the per-event leakage assert) and appends
one row per (triple, arm, pair or board) to the cell table, a CSV with the fixed columns
CELL_COLUMNS. A run that stops can be started again with the same command: the triples already
in the table are skipped.

What lives here:

  board identity      BoardIdentityError, the role map (--role-map) and its checks
  sources             source_from_bank (--src-bank), source_coeffs_for (a refit from events),
                      pooled_source / pooled_lpo_source (mode e), the in-run split (cmd_cell)
  one-board cells     _one_board_arm (protocols P3 and P4 of SPEC.md 3), _xfer_board_of
  source picking      _pick_sources (--src-k, --src-pick)
  the cell            cmd_cell, _flush, CELL_COLUMNS

The placebo of a source refitted from events (source_coeffs_for) and of the in-run split
(cmd_cell) is seeded from Python's hash() of a tuple of strings, which is salted per process
unless PYTHONHASHSEED is fixed; python -m twc_transfer fixes it to 0. With PYTHONHASHSEED=0
the seeds are those of the study's cells under Python 3.9.12 (LCG_104d); other Python versions
are not verified to give the same seeds. The bank's own placebo (bank.py) is seeded by
blake2b and does not depend on it.
"""
import json
import os
import sys
import time
from itertools import combinations

import numpy as np
import pandas as pd

from . import campaigns
from ._core import ORDER
from .arms import ARMS_CORE, self_coeff_maps, transferred_map, xfer_lin_lpo_map, xfer_lin_map
from .bank import coeffs_from_row, lpo_events, overlap_metrics, tot_stats
from .campaigns import N_CAP
from .events import LeakageError, Triple, assert_disjoint, event_keys
from .naming import (boards_of, log, parse_pixels, pixel_tracks, roles_of, run_dir,
                     top_triples, track_name)
from .solve import VariantCoeff, centred_to_raw_matrix, raw_to_centred, solve
from .widths import evaluate_arm

# The cell table's columns, in the order every cell CSV has them (SPEC.md 7). The writer
# (_flush) refuses a row with any other key and an existing file with any other header, so a
# table can be appended to only by the schema that wrote it.
CELL_COLUMNS = [
    # identity
    "cell", "mode", "campaign", "run", "combo", "tgt_triple", "tgt_key",
    "src_run", "src_combo", "src_triple", "n_fit_src", "rho", "self_transfer",
    "same_pixel", "same_board", "src_file_overlap", "src_keys_checked",
    # was the key intersection meaningful at all: 0 with src_keys_checked = 0 says the
    # source and the target put different boards under the same role name, 1 with 0 says
    # the source read no events (bank-backed).
    "keys_comparable",
    "n_pool_excluded",
    # the pixel -> board protocols (SPEC.md 3). Six of the seven are blank (NaN or "") on a
    # cell that does not name a single transferred board. The seventh, src_pick, records how
    # the source pixel was chosen and is meaningful for every mode c / cb / d cell, all3 ones
    # included, where it reads trkpop (the top-K tracks rule) or explicit (--src-triple).
    # Which board carries the transferred polynomial, and whose it is:
    "xfer_board",       # target role whose polynomial was replaced; "" = all three
    "src_role",         # source role the polynomial came from ( == xfer_board in P3 )
    # how the source pixel was chosen, and the evidence that it is "the most populous":
    "src_pick",         # explicit | pixpop | trkpop
    "src_pix_rc",       # the chosen source pixel on `src_role`, e.g. R14C1
    "src_pix_ntrk",     # how many tracks carry that pixel on that board
    "src_pix_pop",      # its total population proxy (summed track bytes)
    # and how much of the leave-pair-out guard was definable for a one-board transfer:
    "lpo_pairs",
    "ts",
    # the measurement
    "arm", "kind", "name", "value", "v_f0", "v_f1", "n", "n_f0", "n_f1",
    # fidelity / diagnostics
    "ks_d", "ks_p", "accepted", "comps", "grid_step", "iqr", "rms",
    "own_slope", "neg_radicand",
    # coverage / extrapolation
    "cov_C", "hist_O", "lever_X", "lever_X_all", "extrap",
]


# ------------------------------------------------------------------ source lookup
_BANK_CACHE = {}


class BoardIdentityError(RuntimeError):
    """A source board was about to be applied to a DIFFERENT target board.

    Modes a/b/f identify the source triple by the target's role -> pixel map and check
    that the role SETS agree. Role sets are not board identity: March H1 is
    `extra0-ref1-dut2` (IH11 = ref = board 1, IH12 = dut = board 2) while July H1 is
    `extra0-dut1-ref2` (IH11 = dut = board 1, IH12 = ref = board 2), so a mar -> jul cell
    matches every role set while it would put board 2's coefficients into board 1's
    correction and vice versa.

    Raised in every mode; mode (d) is exempt only when a `--role-map` was supplied,
    which is what makes its board -> board transfer a declared intent rather than an
    unnoticed swap.
    """


def _bank(path):
    """The bank parquet `path`, read once per process. A relative path is taken in the read
    root (campaigns.transfer_root()), as `tier0 --bank` does."""
    if not os.path.isabs(path):
        path = os.path.join(campaigns.transfer_root(), path)
    if path not in _BANK_CACHE:
        _BANK_CACHE[path] = pd.read_parquet(path)
    return _BANK_CACHE[path]


def _role_map_of(a, tgt_roles, src_roles=None):
    """Parse and validate `--role-map 'tgt_role:src_role,...'`.

    Returns (mp, None) with `mp[target_role] = source_role`, or (None, reason).
    The map must be a bijection that covers exactly the TARGET's roles: a target role
    the map missed would be filled with zeros downstream, i.e. that board would silently
    become `notwc` and its whole walk would be charged to the transfer.
    """
    if not getattr(a, "role_map", None):
        return None, "no --role-map given"
    try:
        mp = dict(kv.split(":") for kv in a.role_map.split(","))
    except ValueError:
        return None, "--role-map is not 'tgt:src,...': %r" % a.role_map
    miss_t = set(tgt_roles) - set(mp)
    if miss_t:
        return None, "--role-map does not cover target role(s) %s" % sorted(miss_t)
    extra_t = set(mp) - set(tgt_roles)
    if extra_t:
        return None, ("--role-map names target role(s) %s the target does not have"
                      % sorted(extra_t))
    if len(set(mp.values())) != len(mp):
        return None, "--role-map is not a bijection: %r" % a.role_map
    if src_roles is not None:
        miss_s = set(mp.values()) - set(src_roles)
        if miss_s:
            return None, ("--role-map names source role(s) %s the source does not have"
                          % sorted(miss_s))
    return mp, None


def _boards_shared(src_boards, tgt_boards, roles):
    """Do the SOURCE and the TARGET put the same board under each role NAME?

    This, and not the role map, decides whether the per-event content keys (`src_keys`,
    `lpo_keys`) survive a `--role-map`. The keys are `event_keys(file, toa,
    sorted(roles))`: the role tuple is sorted and the TOA column of each role is hashed in
    that order, so two key sets are comparable exactly when both sides mean the same
    physical board by each role name.

      * modes (c) and (d) take their source from the TARGET's own combo, so the
        role -> board dict is identical on both sides whatever the role map permutes:
        the map moves COEFFICIENTS between boards, not columns between role names. The
        SPEC 1 event-level guard is therefore valid and is kept.
      * a cross-campaign pair that reassigned roles (mar `extra0-ref1-dut2` -> jul
        `extra0-dut1-ref2`: `ref` is board 1 there and board 2 here) hashes two
        different board orderings, so the intersection would come back empty for
        free. Only then are the keys dropped, recorded as `src_keys_checked = 0`:
        absent rather than faked.

    `src_boards` is the source's OWN role -> board dict (unpermuted, e.g. `src.boards`
    or `boards_of(src_combo)`), never the permuted one.

    Applied with or without a `--role-map`: the keys' comparability is a property of the
    two board maps, not of the flag. Gated on the flag, the guard would pass vacuously for
    an unmapped mode-d board swap.
    """
    for r in roles:
        if r not in src_boards or src_boards[r] != tgt_boards.get(r):
            return False
    return True


def _role_map_hint(src_boards, tgt_boards, tgt_roles):
    """The COMPLETE `--role-map` that would fix a board-identity mismatch, or None.

    Only a full bijection is ever offered: a partial or two-to-one map printed as the fix
    would be rejected by `_role_map_of` with a different message or, worse, trusted.
    Returns None when no bijection exists, and the caller prints both board maps instead.
    """
    inv = {}
    for s, b in (src_boards or {}).items():
        inv.setdefault(b, []).append(s)
    mp = {}
    for t in sorted(tgt_roles):
        cand = inv.get(tgt_boards.get(t), [])
        if len(cand) != 1:
            return None
        mp[t] = cand[0]
    if len(set(mp.values())) != len(mp):
        return None
    return ",".join("%s:%s" % (t, mp[t]) for t in sorted(mp))


def _board_mismatch(src_boards, tgt_boards, roles):
    """The target roles whose SOURCE board is not the target's; a MISSING one counts.

    Kept in step with `_boards_shared`, which returns False for a role the source does
    not report: a source that reported only two of the target's three roles must fail
    the board-identity assert, since its event keys are dropped as incomparable and the
    one board nobody could vouch for would otherwise be the one nobody checked.

    A source that reports NO boards at all is the modes (e) / `inrun` case (`src_boards`
    is `{}` for the pool, and `inrun` reports the target's own): there is nothing to
    compare and both take their events from the target's own combo, so an EMPTY dict is
    exempt. An INCOMPLETE one is not.
    """
    if not src_boards:
        return []
    return [r for r in roles if src_boards.get(r) != tgt_boards.get(r)]


def _board_identity_fix(src_boards_own, tgt_boards, tgt_roles):
    """The "Fix:" clause of a board-identity error: a COMPLETE map, or both board maps.

    `src_boards_own` must be the source's OWN, UNPERMUTED role -> board dict. The permuted
    `info['src_boards']` would give the RESIDUAL map (what would still have to be applied
    on top of the `--role-map` already given), and pasting that into the failing command
    would make the swap worse instead of fixing it. With no map given the two dicts
    coincide.
    """
    hint = _role_map_hint(src_boards_own, tgt_boards, tgt_roles)
    if hint:
        return "pass --role-map '%s'" % hint
    return ("no consistent map exists - source boards %s vs target boards %s (a role map "
            "must be a bijection of the target's roles onto the source's)"
            % ({r: (src_boards_own or {}).get(r) for r in sorted(src_boards_own or {})},
               {r: tgt_boards.get(r) for r in sorted(tgt_roles)}))


def _src_is_same_run(a):
    """Does this cell's source live in the TARGET's own campaign and run?

    `--src-parity-contrast`'s precondition. Both of its arms refit the SOURCE on a file
    parity and difference the two, which measures shared drift only while the source's
    files ARE the files the target is evaluated on. Across runs or campaigns
    `src_obj.file % 2` has no relation to the target's parity, so `xfer_srcmatch` and
    `xfer_srccross` would differ by fit-sample noise alone while still being named, and
    aggregated, as a shared-drift measurement. Reads the args only (no triple), so
    `cmd_cell` can refuse before any event is read.
    """
    return ((getattr(a, "src_campaign", None) or a.campaign) == a.campaign and
            (getattr(a, "src_run", None) or a.run) == a.run)


def _src_track_name(tri, a, src_combo):
    """The SOURCE triple's file name for a same-pixel mode (a/b/f).

    Without `--role-map` the source carries the target's own role -> pixel map, which is
    right only while both campaigns give the same role to the same board. With
    `--role-map 'tgt:src,...'` the boards are permuted, so the pixel that must appear
    under the SOURCE role `src_r` is the target's pixel for `tgt_r`: the inverse of the
    permutation applied to the coefficients afterwards. Returns (name, None) or
    (None, reason).
    """
    sroles = roles_of(src_combo)
    if getattr(a, "role_map", None):
        mp, why = _role_map_of(a, tri.roles, sroles)
        if mp is None:
            return None, why
        pix = {}
        for tgt_r, src_r in mp.items():
            if tgt_r not in tri.pix:
                return None, "target triple has no pixel for role %s" % tgt_r
            pix[src_r] = tri.pix[tgt_r]
        if set(pix) != set(sroles):
            return None, ("--role-map covers source role(s) %s but the source combo %s "
                          "has %s" % (sorted(pix), src_combo, sorted(sroles)))
        return track_name(pix, sroles), None
    if set(sroles) != set(tri.roles):
        return None, "source combo has different roles"
    return track_name(tri.pix, sroles), None


def _permute_lpo(mp, lpo):
    """Re-key a source-side leave-pair-out map onto the TARGET's role names.

    `lpo[src_drop][src_role]` -> `out[tgt_drop][tgt_role]`, through the SAME permutation
    on BOTH levels: the outer key is the board that was left out of the fit and the inner
    keys are the boards that were kept, and `mp[tgt_r]` is the source's name for the board
    the target calls `tgt_r`. A pair the map does not cover is omitted, as an incomplete
    source LPO map is.
    """
    out = {}
    for tgt_drop, src_drop in mp.items():
        c = lpo.get(src_drop)
        if not c:
            continue
        inner = {tgt_r: c[src_r] for tgt_r, src_r in mp.items() if src_r in c}
        if inner:
            out[tgt_drop] = inner
    return out


def _apply_role_map(a, tri, coeffs, pieces, src_roles):
    """Permute a source's per-role objects onto the TARGET's role names.

    `coeffs`, `pieces['placebo']`, `pieces['stats']`, `pieces['src_pix']`,
    `pieces['src_boards']` and `pieces['lpo']` are re-keyed by the target role they are
    applied to, so that a role-reassigning campaign pair can be transferred at all, in
    modes a/b/c/cb/d/f alike. The LPO map is permuted on both levels (`_permute_lpo`):
    `cmd_cell` indexes it with TARGET roles, and mar `extra0-ref1-dut2` / jul
    `extra0-dut1-ref2` share the role NAME set, so a map left keyed by SOURCE roles would
    succeed on every lookup and build `xfer_lpo` from the wrong board's fits.

    The per-event content keys (`src_keys`, `lpo_keys`) are NOT touched here. This helper
    only records `keys_comparable` (`_boards_shared` on the source's own, unpermuted
    role -> board dict, taken BEFORE the permutation), and the callers drop the keys on
    that flag alone: a bank-backed source has none to begin with (`source_from_bank`
    writes `src_keys=None, lpo_keys=[]`), and the event path drops them in
    `source_coeffs_for`, outside any role-map block, so an UNMAPPED board swap loses them
    too. In modes (c) and (d) the source lives in the target's own combo, both sides mean
    the same board by each role name, and the key intersection stays: it is exactly the
    test that catches a same-events source.

    Returns (coeffs, pieces) or (None, reason).
    """
    mp, why = _role_map_of(a, tri.roles, src_roles)
    if mp is None:
        return None, why
    # BEFORE the permutation: the source's own role -> board dict is what the event
    # keys were hashed under, so the comparability test is made on it, not on the
    # re-keyed copy.
    keys_ok = _boards_shared(pieces.get("src_boards") or {}, tri.boards, tri.roles)
    miss_s = set(mp.values()) - set(coeffs)
    if miss_s:
        return None, ("--role-map names source role(s) %s the source does not have"
                      % sorted(miss_s))
    out = {tgt_r: coeffs[src_r] for tgt_r, src_r in mp.items()}
    pl = pieces.get("placebo")
    if pl:
        pieces["placebo"] = {tgt_r: pl[src_r] for tgt_r, src_r in mp.items() if src_r in pl}
    st = pieces.get("stats") or {}
    pieces["stats"] = {tgt_r: st[src_r] for tgt_r, src_r in mp.items() if src_r in st}
    sp = pieces.get("src_pix") or {}
    pieces["src_pix"] = {tgt_r: sp.get(src_r, (-1, -1)) for tgt_r, src_r in mp.items()}
    sb = pieces.get("src_boards") or {}
    # the source's OWN role -> board dict, kept beside the permuted one so a
    # board-identity error can print the map to PASS rather than the residual left over
    # after the map already given (`_board_identity_fix`).
    pieces["src_boards_own"] = dict(sb)
    pieces["src_boards"] = {tgt_r: sb.get(src_r, -1) for tgt_r, src_r in mp.items()}
    lp = pieces.get("lpo")
    if lp:
        pieces["lpo"] = _permute_lpo(mp, lp)
    # Only the flag is recorded here; dropping the keys is the callers' job (see the
    # docstring): a bank-backed source carries none, and on the event path
    # `source_coeffs_for` sets them itself, from `keys_ok`, outside the role-map block.
    pieces["keys_comparable"] = bool(keys_ok)
    pieces["role_map"] = a.role_map
    pieces["src_role_map"] = dict(mp)
    return out, pieces


def _parity_contrast_arms(a, tri, info, folds, xb=None, sr=None, base=None, nmin_parity=200):
    """The `--src-parity-contrast` arms, role-map aware.

    A same-run source is fitted on the SAME FILES the target is evaluated on, so it
    inherits that file's inter-board common mode (measured: 0.7-6.6 ps rms). Fitting
    the source on the matching file parity (`xfer_srcmatch`) and on the complementary
    one (`xfer_srccross`) and differencing measures how much of a same-run transfer is
    that shared drift.

    These arms are re-solved from `info['src_obj']`, the UNPERMUTED source `Triple`,
    and are indexed by the TARGET's role names, so the map `source_coeffs_for` recorded
    as `src_role_map` is applied here, and the boards the arms actually draw from
    (`pc_boards`) are asserted against the target's: mode (d) WITH a role map exempt,
    exactly as in `cmd_cell` (an unmapped mode-d source is asserted like any other).

    The same-run precondition `--src-parity-contrast` needs is enforced by `cmd_cell`,
    not here, so this helper stays callable on a cross-campaign pair.

    One-board aware. With `--xfer-board xb` (P3: `sr == xb`, P4: `sr != xb`) the SAME
    joint three-board fit is re-solved on the source triple's matching / complementary
    file parity, but only the resulting coefficient for role `srole[sr]` (the board the
    transferred polynomial is fitted on) is USED, and it lands on target role `xb` alone;
    `base` (the target's own `self_cf`, fold-matched) supplies every other board
    unchanged. This is the shape `_one_board_arm` gives `xfer` and `placebo` in a
    one-board cell (SPEC 3): one board's polynomial replaced, the rest the target's own.
    The board-identity check is then scoped to `sr` alone: `sr`, not `xb`, is the role
    whose SOURCE board the coefficient comes from, and `xb` has no source-side
    counterpart to check when `sr != xb` (P4).

    Returns `(arms, keys, pc_boards)`; `arms` is empty when a parity half is too small or
    the source lacks a needed role. Raises `BoardIdentityError`.
    """
    src_o = info.get("src_obj")
    if src_o is None:
        return {}, {}, {}
    rm = info.get("src_role_map") or {}
    srole = {r: rm.get(r, r) for r in tri.roles}
    pc_boards = {r: (src_o.boards or {}).get(srole[r]) for r in tri.roles}
    check_roles = [sr] if xb is not None else sorted(tri.roles)
    if a.mode != "d" or not getattr(a, "role_map", None):
        bad = _board_mismatch(pc_boards, tri.boards, check_roles)
        if bad:
            raise BoardIdentityError(
                "cell %s, mode %s: the --src-parity-contrast arms would take role(s) %s "
                "from source board(s) %s while the target's are %s (source %s %s %s, "
                "target %s %s %s, triple %s).  These arms are re-solved from the "
                "unpermuted source triple, so they must be re-keyed through the same "
                "--role-map as the main transfer; %s"
                % (getattr(a, "cell", "?"), a.mode, bad,
                   {r: pc_boards[r] for r in bad}, {r: tri.boards.get(r) for r in bad},
                   info.get("src_campaign"), info.get("src_run"), info.get("src_combo"),
                   tri.campaign, tri.run, tri.combo, tri.name,
                   _board_identity_fix(src_o.boards, tri.boards, tri.roles)))
    # the content keys follow the same rule as the main transfer's `src_keys`: they are
    # hashes of the SOURCE's own TOA columns under the target's sorted role tuple,
    # comparable exactly when both sides mean the same board by each role name.
    keys_ok = _boards_shared(src_o.boards, tri.boards, tri.roles)
    need_roles = [srole[sr]] if xb is not None else [srole[r] for r in tri.roles]
    arms, keys = {}, {}
    for tag, matched in (("srcmatch", True), ("srccross", False)):
        m_new, kk_ = {}, {}
        ok_pc = True
        for f in folds:
            sel = np.where((src_o.file % 2 == f) if matched
                           else (src_o.file % 2 != f))[0]
            if len(sel) < nmin_parity:
                ok_pc = False
                break
            cpc = solve(src_o.tot, src_o.toa, src_o.roles, idx=sel)
            if not set(cpc) >= set(need_roles):
                ok_pc = False
                break
            if xb is not None:
                m_new[(f, xb)] = cpc[srole[sr]]
            else:
                for r in tri.roles:
                    m_new[(f, r)] = cpc[srole[r]]
            if keys_ok:
                kk_[f] = [src_o.key_entry(sel, tuple(sorted(tri.roles)))]
        if ok_pc:
            if xb is not None:
                m = dict(base)
                m.update(m_new)
            else:
                m = m_new
            arms["xfer_%s" % tag] = m
            keys["xfer_%s" % tag] = kk_ or None
    return arms, keys, pc_boards


def source_from_bank(tri, a, src_name):
    """Read one source triple's coefficients out of the bank ("fit once, evaluate many").

    Returns (coeffs, info) exactly as source_coeffs_for does, or (None, reason). No
    source event is read: the coefficients, the TOT summary the overlap metrics need,
    the placebo fit and the leave-pair-out fits all travel in the bank row.
    """
    b = _bank(a.src_bank)
    sel = b[(b["run"] == (a.src_run or a.run)) & (b["combo"] == (a.src_combo or tri.combo))
            & (b["src_file"] == src_name) & (b["fold"] == "full")]
    if not len(sel):
        return None, "source triple %s not in bank %s" % (src_name, os.path.basename(a.src_bank))
    full = sel[(sel["guard"] == "full") & (sel["placebo"] == 0)]
    if not len(full):
        return None, "no full-guard rows for %s in bank" % src_name
    coeffs = {r["role"]: coeffs_from_row(r) for _, r in full.iterrows()}
    stats = {}
    for _, r in full.iterrows():
        stats[r["role"]] = {k: r[k] for k in ("q005", "q01", "q05", "q25", "q50", "q75", "q95",
                                              "q99", "q995", "mean_tot", "sd_tot_emp", "hist",
                                              "hist_lo", "cal_mean")}
    pl = sel[(sel["guard"] == "full") & (sel["placebo"] == 1)]
    placebo = {r["role"]: coeffs_from_row(r) for _, r in pl.iterrows()} if len(pl) else None
    lpo = {}
    for g_, gg in sel[sel["guard"].str.startswith("lpo:")].groupby("guard"):
        lpo[g_.split(":", 1)[1]] = {r["role"]: coeffs_from_row(r) for _, r in gg.iterrows()}
    info = dict(src_campaign=a.src_campaign or a.campaign, src_run=a.src_run or a.run,
                src_combo=a.src_combo or tri.combo, src_triple=src_name,
                src_key=str(full.iloc[0]["src_triple"]), n_fit=int(full.iloc[0]["n_fit"]),
                self_transfer=(src_name == tri.name and (a.src_run or a.run) == tri.run
                               and (a.src_combo or tri.combo) == tri.combo),
                stats=stats, placebo=placebo, lpo=lpo, from_bank=True,
                src_pix=parse_pixels(src_name), src_boards=boards_of(a.src_combo or tri.combo),
                # the same dict again, under the name the board-identity error reads:
                # `src_boards` is re-keyed by `_apply_role_map` when a --role-map is
                # given, `src_boards_own` never is.
                src_boards_own=boards_of(a.src_combo or tri.combo),
                # recorded although a bank source has no keys either way, so the column
                # tells "no events were read" from "the two sides mean different boards"
                # instead of leaving both as src_keys_checked = 0.
                keys_comparable=_boards_shared(boards_of(a.src_combo or tri.combo),
                                               tri.boards, tri.roles),
                # a bank-backed source reads no events, so there are no keys to
                # intersect: the invariant rests on the structural assert here and the
                # row records src_keys_checked = 0 rather than implying more.
                src_keys=None, lpo_keys=[], src_files=None)
    return coeffs, info


def source_coeffs_for(tri, a, bank=None, src_name=None):
    """Resolve the source coefficients for one target triple.

    Returns (coeffs, info) where info carries the source identity, n_fit, the
    per-board source stats needed by the overlap metrics, the placebo source and
    the source-side LPO map; or (None, reason) when the source is missing.
    """
    mode = a.mode
    src_campaign = a.src_campaign or a.campaign
    src_run = a.src_run or a.run
    src_combo = a.src_combo or tri.combo
    # ONE root for every read of the source tree: the triple itself and the leave-pair-out
    # re-reads alike. A same-campaign source reached through --root (e.g. the August v2
    # tree) has src_root = None, and run_dir / lpo_events would then fall back to
    # CAMPAIGNS[campaign] and read a DIFFERENT tree; the coefficients would come from one
    # tree and the guard from another. A genuine cross-campaign source with no --src-root
    # resolves through CAMPAIGNS.
    src_root = a.src_root or (getattr(a, "root", None)
                              if src_campaign == a.campaign else None)
    # The SOURCE's own per-triple event cap. `--cap` (N_CAP = 6000) applies to every
    # read, and at that value every candidate source track of a DESY / March run is
    # saturated: `n_fit_src = 6000` for the top pixel and for the runner-up alike, so
    # "the most populous pixel" is a deterministic tie-break and not, at this cap, a
    # better-determined polynomial. `--src-cap` raises the source read alone (0 = follow
    # --cap), which changes n_fit_src and nothing about any target's evaluation set.
    s_cap = int(getattr(a, "src_cap", 0) or 0) or a.cap
    info = dict(src_campaign=src_campaign, src_run=src_run, src_combo=src_combo)

    if (getattr(a, "role_map", None) and mode in ("a", "b", "f")
            and src_campaign == tri.campaign and src_run == tri.run
            and src_combo == tri.combo):
        # Modes a/b/f are the SAME-PIXEL modes: the source triple is named by applying the
        # role -> pixel map to the source combo, so with a role map and a source in the
        # TARGET's own run and combo the "source" is a permuted pixel triple of the very
        # same run: a cross-board same-run transfer under a same-pixel label, a disguised
        # self-transfer. Modes (c) pixel -> board, (cb) pixel -> OTHER board and (d)
        # board -> board do that ON PURPOSE, within one run, and by construction land on a
        # different pixel or a different board; they are exempt, and since their source
        # sits in the target's own combo they keep the full event-level key guard
        # (_boards_shared).
        raise LeakageError(
            "cell %s, mode %s: --role-map %r with a source in the TARGET's own run and "
            "combo (%s %s %s).  Modes a/b/f name the source by the target's own "
            "role -> pixel map, so a role map here makes the cell a cross-board transfer "
            "inside one run under a same-pixel label, a disguised self-transfer.  "
            "Use a source in a different run/campaign, drop "
            "--role-map (the board-identity assert then decides whether the geometry is "
            "the same), or state the intent with mode c (pixel -> board) or mode d "
            "(board -> board), which are exempt."
            % (getattr(a, "cell", "?"), mode, a.role_map, src_campaign, src_run, src_combo))

    if mode == "inrun":
        return None, "inrun handled by the caller"

    if mode in ("a", "b", "f"):
        # the source file name comes from the role -> pixel map. With --role-map it is
        # the PERMUTED map, so the pixel asked of the source is the one that sits on the
        # same physical board as the target's (mar ref = jul dut = board 1).
        name, why = _src_track_name(tri, a, src_combo)
        if name is None:
            return None, why
        if getattr(a, "src_bank", None):
            coeffs, info_b = source_from_bank(tri, a, name)
            if coeffs is None:
                return None, info_b
            if getattr(a, "role_map", None):
                coeffs, info_b = _apply_role_map(a, tri, coeffs, info_b,
                                                 roles_of(src_combo))
                if coeffs is None:
                    return None, info_b
            return coeffs, info_b
        p = os.path.join(run_dir(src_campaign, src_run, src_root), src_combo, "time", name + ".parquet")
        if not os.path.exists(p):
            return None, "no matching source triple"
        src = Triple.from_file(p, src_campaign, src_run, src_combo, cap=s_cap,
                               neighbor=a.neighbor, neighbor_logic=a.neighbor_logic)
        if src.n < a.nmin_src:
            return None, "source triple too small (n=%d)" % src.n
    elif mode in ("c", "d", "cb"):
        name = src_name or a.src_triple
        if not name:
            return None, "--src-triple required for mode %s" % mode
        if getattr(a, "src_bank", None) and mode in ("c", "cb"):
            # not a bare `return`: the role map has to be applied on this path too. The
            # bank reports src_boards = boards_of(src_combo), which for mode c's default
            # (--src-combo unset -> the target's own combo) equals tri.boards, so the
            # board-identity assert would not catch an omission: the cell would run with
            # UNPERMUTED coefficients and exit 0.
            coeffs, info_b = source_from_bank(tri, a, name)
            if coeffs is None:
                return None, info_b
            if getattr(a, "role_map", None):
                coeffs, info_b = _apply_role_map(a, tri, coeffs, info_b,
                                                 roles_of(src_combo))
                if coeffs is None:
                    return None, info_b
            return coeffs, info_b
        p = os.path.join(run_dir(src_campaign, src_run, src_root), src_combo, "time",
                         name + ".parquet")
        if not os.path.exists(p):
            return None, "source triple file missing"
        src = Triple.from_file(p, src_campaign, src_run, src_combo, cap=s_cap,
                               neighbor=a.neighbor, neighbor_logic=a.neighbor_logic)
    elif mode == "e":
        return None, "pooled mode handled by the caller"
    else:
        return None, "unknown mode %s" % mode

    self_transfer = (src.campaign == tri.campaign and src.run == tri.run and
                     src.combo == tri.combo and src.name == tri.name)
    coeffs = solve(src.tot, src.toa, src.roles)
    rng = np.random.default_rng(abs(hash(("placebo", src.name, src.run))) % (2 ** 31))
    placebo = solve(src.tot, src.toa, src.roles, permute=True, rng=rng)

    stats = {r: tot_stats(src.tot[r]) for r in src.roles}
    for r, s in stats.items():
        s["cal_mean"] = src.cal.get(r, np.nan)
    src_pix = dict(src.pix)
    src_boards = dict(src.boards)

    rm_active = bool(getattr(a, "role_map", None)) and mode in ("a", "b", "c", "cb", "d", "f")
    # Are the per-event content keys comparable? Decided by the BOARDS alone, never by
    # whether a --role-map was passed. The keys hash the SOURCE's own TOA columns under the
    # TARGET's sorted role tuple, so two key sets mean the same thing exactly when both
    # sides put the same board under each role name: true by construction for a source in
    # the target's own combo (modes c/d), false for a role-reassigning campaign pair,
    # mapped or not. Deciding it by the flag would let mode (d), the one mode the
    # board-identity assert exempts, report `src_keys_checked > 0` for an intersection that
    # is empty by construction (two different board orderings hashed). See `_boards_shared`.
    keys_ok = _boards_shared(src.boards, tri.boards, tri.roles)

    if rm_active:
        # SPEC 3's role map, honoured in every named-source mode, so that a
        # role-reassigning campaign pair (mar `extra0-ref1-dut2` -> jul `extra0-dut1-ref2`)
        # does not transfer board 2's coefficients onto board 1.
        pieces = dict(placebo=placebo, stats=stats, src_pix=src_pix, src_boards=src_boards)
        coeffs, pieces = _apply_role_map(a, tri, coeffs, pieces, src.roles)
        if coeffs is None:
            return None, pieces                       # `pieces` is the reason string
        placebo, stats = pieces["placebo"], pieces["stats"]
        src_pix, src_boards = pieces["src_pix"], pieces["src_boards"]
        info["role_map"] = pieces["role_map"]
        # the target role -> source role permutation, kept so that anything built LATER
        # out of the unpermuted source object (the --src-parity-contrast arms) can be
        # re-keyed through exactly the same map.
        info["src_role_map"] = pieces["src_role_map"]
    info["keys_comparable"] = bool(keys_ok)

    lpo = {}
    lpo_keys = []
    lpo_shape = {}
    if a.lpo:
        # `--lpo-fallback sibling` (the default) is the all-or-nothing rule below: one
        # dropped role without a sibling combo abandons every drop. `--lpo-fallback own`
        # builds a missing drop's guard from the SOURCE's OWN full event set instead: a
        # joint solve on just the two REMAINING roles, which never sees the dropped role's
        # TOT/TOA at all (the same argument as on the target side, arms.self_coeff_maps).
        fb_own = (getattr(a, "lpo_fallback", "sibling") == "own")
        for drop in src.roles:
            # `src_root` (set at the top of this function) is the root the source triple
            # itself was read through, so the coefficients and this guard can never come
            # from two different trees.
            got = lpo_events(src, drop, cap=s_cap, root=src_root, neighbor=a.neighbor,
                             neighbor_logic=a.neighbor_logic)
            if got is not None:
                ltot, ltoa, lfile, lroles, sib, nl = got
                c = solve(ltot, ltoa, lroles)
                lpo[drop] = c
                kept = tuple(sorted(r for r in src.roles if r != drop))
                lpo_keys.append((kept, event_keys(lfile, ltoa, kept)))
                lpo_shape[drop] = "sibling"
            elif fb_own:
                remaining = [r for r in src.roles if r != drop]
                c = solve(src.tot, src.toa, remaining)
                lpo[drop] = c
                kept = tuple(sorted(remaining))
                lpo_keys.append((kept, src.keys(kept)))
                lpo_shape[drop] = "own"
            else:
                lpo, lpo_keys, lpo_shape = {}, [], {}
                break

    if rm_active:
        # the LPO map is built AFTER the permutation block above (it needs `src`), so it
        # is re-keyed here, through the same map and on both levels: `cmd_cell` indexes it
        # with TARGET roles, and a role-reassigning pair shares the role NAME set, so a map
        # left keyed by SOURCE roles would build `xfer_lpo` of a role-mapped cell from the
        # wrong boards. (`lpo_keys` is dropped just below, on the board-comparability rule
        # alone, so an unmapped board swap loses it too.)
        mp_l, why_l = _role_map_of(a, tri.roles, src.roles)
        if mp_l is None:
            return None, why_l
        lpo = _permute_lpo(mp_l, lpo)
        lpo_shape = {tgt_d: lpo_shape[src_d] for tgt_d, src_d in mp_l.items()
                     if src_d in lpo_shape}

    if not keys_ok:
        # outside the role-map block, on the same rule as `src_keys` below: the LPO keys
        # are `event_keys(file, toa, sorted(kept roles))` over the SOURCE's own columns, so
        # they are comparable exactly when both sides mean the same board by each role
        # name, a property of the boards, not of the flag. An unmapped cross-campaign
        # source (only mode d can get that far) would otherwise keep a key list hashed
        # under the source's board ordering: an intersection empty by construction,
        # reported as a guard that ran.
        lpo_keys = []

    # the event identities the transferred coefficients were fitted on (SPEC 1).
    # Comparable whenever the source means the same board by each of the target's role
    # names (`keys_ok`): always without a role map, and under a role map exactly when the
    # map permutes coefficients rather than board identity (modes c/d). Where it is not,
    # the structural assert (different pixel triple) is what stands, recorded as
    # src_keys_checked = 0, not implied.
    src_keys = None
    if keys_ok and set(src.roles) >= set(tri.roles):
        ent = src.key_entry(None, tuple(sorted(tri.roles)))
        src_keys = {0: [ent], 1: [ent]}
    info.update(src_triple=src.name, src_key=src.key, n_fit=src.n, self_transfer=self_transfer,
                stats=stats, placebo=placebo, lpo=lpo, src_obj=src, src_pix=src_pix,
                src_boards=src_boards, src_keys=src_keys, lpo_keys=lpo_keys,
                # `src_boards` above is the PERMUTED dict under a --role-map; this one is
                # the source's own and is what the board-identity error must quote its
                # suggested map from.
                src_boards_own=dict(src.boards),
                src_files=np.unique(src.file),
                # {drop_role: "sibling"|"own"}, keyed by TARGET role once past the
                # rm_active re-key above; empty ({}) under the default
                # --lpo-fallback sibling.
                lpo_shape=lpo_shape)
    return coeffs, info


def _pool_slice(tri, cap_per):
    """The capped, TOA-median-centred contribution of one pool triple (cached)."""
    key = ("_pool", cap_per)
    if key not in tri.meta:
        idx = (np.arange(tri.n) if tri.n <= cap_per
               else np.sort(np.random.default_rng(7).choice(tri.n, cap_per, False)))
        tri.meta[key] = (idx,
                         {r: tri.tot[r][idx] for r in tri.roles},
                         {r: tri.toa[r][idx] - np.median(tri.toa[r][idx]) for r in tri.roles})
    return tri.meta[key]


def pooled_source(a, tri_list, target=None, cap_per=1500, exclude_key=None, scope="board"):
    """(e) pooled board-level TWC, leave-one-PIXEL-out, built ONE BOARD AT A TIME.

    The pool design is assembled per board: for board r, every pool triple whose pixel
    FOR THAT BOARD equals the target's is dropped, and only board r's coefficients are
    taken from that solve. An exclusion on the whole triple key (leave-one-TRIPLE-out)
    would let through a pool triple that carries the target's pixel on one board with
    different partners: measured, 15 of 36 (target, board) slots contaminated that way on
    jul h1_run10 (mar h1_run1 is clean only because its top-40 triples are all diagonal).
    The pooled penalty is the per-pixel advantage, so contamination biases it DOWN.

    Each triple's per-board TOA is median-centred before pooling (so pixel-to-pixel
    offsets cannot masquerade as walk) and events are capped per pixel. The
    alternative solve-then-average is returned beside it, on the same per-board
    selection. `n_excluded` and the fitted event keys travel back for the record
    and for the SPEC 1 invariant.
    """
    tri_list = [t for t in tri_list if exclude_key is None or t.key != exclude_key]
    if not tri_list:
        return None
    roles = tri_list[0].roles
    tpix = (target.pix if target is not None else {})
    c_pool, c_avg, stats, n_excl, nfit, keys = {}, {}, {}, {}, {}, []
    for r in roles:
        # scope="board": leave-one-PIXEL-out, the correct exclusion.
        # scope="triple": leave-one-TRIPLE-out, kept only so that the size of the bias
        # described above can be reproduced rather than asserted.
        sel = [t for t in tri_list
               if scope != "board" or not (r in tpix and t.pix.get(r) == tpix.get(r))]
        n_excl[r] = len(tri_list) - len(sel)
        if len(sel) < 2:
            return None
        tot, toa, idxs, per_coeffs, per_n, per_stats = {}, {}, [], [], [], []
        for t in sel:
            idx, tt_, ta_ = _pool_slice(t, cap_per)
            for q in roles:
                tot.setdefault(q, []).append(tt_[q])
                toa.setdefault(q, []).append(ta_[q])
            idxs.append((t, idx))
            per_coeffs.append(solve(t.tot, t.toa, t.roles))
            per_n.append(t.n)
            per_stats.append({q: (float(t.tot[q].mean()), float(t.tot[q].std())) for q in roles})
        tot = {q: np.concatenate(v) for q, v in tot.items()}
        toa = {q: np.concatenate(v) for q, v in toa.items()}
        c = solve(tot, toa, roles)
        c_pool[r] = c[r]
        stats[r] = tot_stats(tot[r])
        nfit[r] = len(tot[r])
        # solve-then-average in a common centred basis (n-weighted, gauge removed)
        mu = float(np.mean([s[r][0] for s in per_stats]))
        sd = float(np.mean([s[r][1] for s in per_stats])) or 1.0
        B = np.array([raw_to_centred(cc[r], mu, sd) for cc in per_coeffs])
        w = np.asarray(per_n, float)
        b = (B * w[:, None]).sum(0) / w.sum()
        b[-1] = 0.0                     # the gauge constant is unobservable
        c_avg[r] = centred_to_raw_matrix(mu, sd) @ b
        for t, idx in idxs:
            keys.append(t.key_entry(idx, roles))
    return dict(c_pool=c_pool, c_avg=c_avg, stats=stats, n_excluded=n_excl,
                n_fit=int(np.median(list(nfit.values()))), keys=keys)


_LPO_POOL_CACHE = {}


def pooled_lpo_source(a, tri_list, target, cap_per=1500, nmax=6):
    """(e) leave-PAIR-out companion to `pooled_source`: the guard for the pooled arm.

    For pair (a,b) the coefficient applied to board a must come from a solve that never
    saw board b. Pooled, that means pooling the SIBLING-combo events (the four-board
    telescope's leave-one-out combo omitting b) of the pool triples, with the same
    per-board leave-one-pixel-out exclusion. Without it the pooled cell has no `xfer_lpo`
    and its cheat statistic is NaN.

    The sibling events are read once per cell and cached; only the solve is redone per
    target. `nmax` bounds the read: 6 pool triples are enough for a guard number and
    keep the cell inside the CPU budget.
    """
    roles = tri_list[0].roles
    ck = (id(tri_list), nmax)
    if ck not in _LPO_POOL_CACHE:
        store = {}
        for drop in roles:
            got = []
            for t in tri_list[:nmax]:
                g = lpo_events(t, drop, cap=cap_per, root=a.root, neighbor=a.neighbor,
                               neighbor_logic=a.neighbor_logic)
                if g is None:
                    continue
                ltot, ltoa, lfile, lroles, sib, nl = g
                # centred TOA for the pooled solve, RAW TOA for the event keys
                got.append(dict(tri=t, tot=ltot, raw=ltoa, file=lfile, roles=lroles,
                                ctr={r: ltoa[r] - np.median(ltoa[r]) for r in lroles}))
            store[drop] = got
        _LPO_POOL_CACHE[ck] = store
    store = _LPO_POOL_CACHE[ck]
    out, keys = {}, []
    tpix = target.pix
    for drop, got in store.items():
        keep = [r for r in roles if r != drop]
        kept_key = tuple(sorted(keep))
        per = {}
        for r in keep:
            sel = [g for g in got if g["tri"].pix.get(r) != tpix.get(r)]
            if len(sel) < 2:
                continue
            lroles = sel[0]["roles"]
            tot = {q: np.concatenate([g["tot"][q] for g in sel]) for q in lroles}
            toa = {q: np.concatenate([g["ctr"][q] for g in sel]) for q in lroles}
            c = solve(tot, toa, lroles)
            if r in c:
                per[r] = c[r]
            for g in sel:
                keys.append((kept_key, event_keys(g["file"], g["raw"], kept_key)))
        if per:
            out[drop] = per
    return out, keys


# ------------------------------------------------------------------------- cell
def _rows_for_arm(arm, ev, tri, base, meta, ov, per_board=None):
    """Long-format rows: one per (arm, pair) width and one per (arm, board) sigma.

    `per_board` carries the per-board bookkeeping that is a property of the SOURCE,
    not of the arm: same_pixel / same_board and the pooled leave-one-pixel-out
    exclusion count.
    """
    def ovget(role, key):
        return ov.get(role, {}).get(key, np.nan) if ov else np.nan

    def pb(role, key):
        return (per_board or {}).get(role, {}).get(key, np.nan)

    rows = []
    for pair, dd in ev["pooled"].items():
        bf = ev["byfold"].get(pair, {})
        x, y = pair.split("-")
        rows.append(dict(meta, arm=arm, kind="pair", name=pair, value=dd["w"],
                         same_pixel=np.nanmax([pb(x, "same_pixel"), pb(y, "same_pixel")]),
                         same_board=np.nanmax([pb(x, "same_board"), pb(y, "same_board")]),
                         n_pool_excluded=np.nanmax([pb(x, "n_pool_excluded"),
                                                    pb(y, "n_pool_excluded")]),
                         v_f0=bf.get(0, {}).get("w", np.nan), v_f1=bf.get(1, {}).get("w", np.nan),
                         n=dd["n"], n_f0=bf.get(0, {}).get("n", np.nan),
                         n_f1=bf.get(1, {}).get("n", np.nan),
                         ks_d=dd["ks_d"], ks_p=dd["ks_p"], accepted=dd["accepted"], comps=dd["comps"],
                         grid_step=dd["grid_step"], iqr=dd["iqr"], rms=dd["rms"],
                         own_slope=np.nan, neg_radicand=np.nan,
                         cov_C=np.nanmin([ovget(x, "cov_C"), ovget(y, "cov_C")]),
                         hist_O=np.nanmin([ovget(x, "hist_O"), ovget(y, "hist_O")]),
                         lever_X=np.nanmax([ovget(x, "lever_X"), ovget(y, "lever_X")]),
                         lever_X_all=np.nanmax([ovget(x, "lever_X_all"), ovget(y, "lever_X_all")]),
                         extrap=np.nanmax([ovget(x, "extrap"), ovget(y, "extrap")])))
    for r, s_ in ev["sigma"].items():
        rows.append(dict(meta, arm=arm, kind="board", name=r, value=s_,
                         same_pixel=pb(r, "same_pixel"), same_board=pb(r, "same_board"),
                         n_pool_excluded=pb(r, "n_pool_excluded"),
                         v_f0=ev["sigma_fold"].get(0, {}).get(r, np.nan),
                         v_f1=ev["sigma_fold"].get(1, {}).get(r, np.nan),
                         n=tri.n, n_f0=int((tri.fold == 0).sum()), n_f1=int((tri.fold == 1).sum()),
                         ks_d=np.nan, ks_p=np.nan, accepted=np.nan, comps="",
                         grid_step=np.nan, iqr=np.nan, rms=np.nan,
                         own_slope=ev["own_slope"].get(r, np.nan),
                         neg_radicand=ev["neg_radicand"].get(r, np.nan),
                         cov_C=ovget(r, "cov_C"), hist_O=ovget(r, "hist_O"),
                         lever_X=ovget(r, "lever_X"), lever_X_all=ovget(r, "lever_X_all"),
                         extrap=ovget(r, "extrap")))
    return rows


def _merge_keys(*ks):
    """Union of several {fold: [(roles, keys)]} source-key maps (a mixed arm)."""
    out = {}
    for k in ks:
        if not k:
            continue
        for f, v in k.items():
            out.setdefault(f, [])
            out[f] += list(v) if isinstance(v, list) else [v]
    return out or None


def _one_board_arm(tri, base, xb, sr, coeffs, folds, lpo=None, base_lpo=None):
    """SPEC 3's `one_board` shape: the protocols P3 and P4.

    Exactly ONE target board's polynomial is replaced; every other board keeps its own
    fold-matched cross-fit self coefficients:

      P3 (mode c with --xfer-board)  source pixel P1 on board B1 -> board B1 of every
         other target triple. `sr == xb`: the polynomial lands on the board it was
         fitted on, just from a different pixel. B2 and B3 keep their own.
      P4 (mode cb)                   source pixel P1 on board B1 -> board B2.
         `sr != xb`: the polynomial crosses boards. B1 and B3 keep their own.

    base      the fold-matched self map (`self_cf`) the untouched boards use
    xb        TARGET role whose polynomial is replaced
    sr        TARGET-role NAME of the source board the polynomial is taken from. It is a
              target role name, not a source one, because `--role-map` has already
              re-keyed the source's objects onto the target's role names, so after that
              map a role name means the same physical board on both sides.
    coeffs    {role: c} the source's per-board coefficients (already permuted)
    lpo       {drop_role: {role: c}} the source's leave-pair-out solves, or None for
              the unguarded arms
    base_lpo  `self_cf_lpo`: what the OTHER boards use in the guarded arm

    Returns `(arm_map, n_guarded_pairs)`.

    Where the leave-pair-out guard is not definable. For the target pair (xb, y) the
    guard wants xb's polynomial to come from a SOURCE solve that left board y out,
    `lpo[y][sr]`. When `y == sr` there is no such object: `lpo[sr]` is the solve that
    DROPPED board sr, so it carries no coefficient for sr at all. So a P4 cell has a
    guard on exactly one of its two xb-pairs (xb with the third board) and none on the
    pair (xb, sr); a P3 cell (`sr == xb`) has both. The undefinable pair keeps the plain
    transferred polynomial, and the count of the pairs that DID get a guard is written
    into the row as `lpo_pairs`, so "guarded" is a number on the row rather than an
    assumption (SPEC 5.1: the absence of a guard is itself a verdict).
    """
    pairs = {frozenset((x, y)): "%s-%s" % (x, y) for x, y in combinations(tri.roles, 2)}
    m = dict(base_lpo if base_lpo is not None else base)
    c_x = None if coeffs is None else coeffs.get(sr)
    if c_x is None:
        raise KeyError("_one_board_arm: the source has no coefficients for role %s "
                       "(has %s)" % (sr, sorted(coeffs or {})))
    for f in folds:
        m[(f, xb)] = np.asarray(c_x, float)
    n_guard = 0
    if lpo is not None:
        for y in tri.roles:
            if y == xb:
                continue
            pk = pairs[frozenset((xb, y))]
            src_c = None if y == sr else (lpo.get(y) or {}).get(sr)
            for f in folds:
                # the pair-specific entry ALWAYS gets written, even when the guard is not
                # definable: `base_lpo` carries the TARGET's own leave-pair-out
                # coefficient at (f, pair, xb), and leaving that in place would put a self
                # fit into the transfer arm.
                m[(f, pk, xb)] = (np.asarray(src_c, float) if src_c is not None
                                  else np.asarray(c_x, float))
            if src_c is not None:
                n_guard += 1
    return m, n_guard


def _pick_rule(a, board):
    """`--src-pick`, with `auto` resolved: pixpop when a board was named, else trkpop.

    Its own function so that `_pick_sources` and the row's `src_pick` column can never
    disagree about which rule actually ran.
    """
    pick = str(getattr(a, "src_pick", None) or "auto").lower()
    if pick == "auto":
        pick = "pixpop" if board else "trkpop"
    if pick not in ("pixpop", "trkpop"):
        sys.exit("--src-pick must be auto | pixpop | trkpop (got %r)" % pick)
    return pick


def _pick_sources(a, tri_pix_by_name, src_campaign, src_run, src_combo, board=None):
    """SPEC 3 c-f with n_src > 1: K source pixels per cell, chosen deterministically.

    With a single source pixel, `random_effects` quotes an se that has no
    source-to-source component at all, and "this one pixel transfers badly" reads as
    "pixel -> board costs 1.4 ps". The rotation takes K sources that carry DISTINCT
    pixels, so the K sources are genuinely different pixels rather than K neighbours of
    one.

    Two rules, selected by `--src-pick`:

    `trkpop`  top-K TRACKS by file size, keeping only tracks whose pixel is new on every
              board. The pixel is whatever the biggest track happens to carry. This is
              what an `all3` whole-triple transfer needs, because there all three of the
              source's pixels are used at once and no single board is privileged.
    `pixpop`  the protocols' rule: rank the PIXELS of one named board by their total
              population and take the top-K, each contributing its own most populous
              track. Needs a board to rank on, which `--xfer-board` / `--src-role`
              supply, and is meaningless without one.

    `auto` (the default) picks `pixpop` when a board was named and `trkpop` otherwise.

    Returns `(names, meta)`; `meta[name]` carries the pixel-population evidence (`pix`,
    `pop`, `ntrk`) for the `pixpop` rule and is empty for the others, so the row can
    print WHY that source was chosen rather than asserting that it is the most populous
    one.
    """
    chosen, chosen_pix, meta = [], [], {}
    if a.src_triple:
        for nm in [s.strip() for s in a.src_triple.split(",") if s.strip()]:
            chosen.append(nm)
            chosen_pix.append(parse_pixels(nm))
    n_explicit = len(chosen)
    k = max(int(getattr(a, "src_k", 1) or 1), len(chosen))
    if len(chosen) >= k:
        return chosen[:k], meta
    # the same one-root rule source_coeffs_for applies: a same-campaign source reached
    # through --root must have its pixel list read from THAT tree, or the names chosen
    # here would not be the ones the triples are then read from.
    src_root = a.src_root or (getattr(a, "root", None)
                              if src_campaign == getattr(a, "campaign", None) else None)
    pick = _pick_rule(a, board)
    if pick == "pixpop":
        if not board:
            sys.exit("cell %s: --src-pick pixpop ranks the PIXELS of one board by "
                     "population, so it needs a board to rank on - name it with "
                     "--xfer-board (and --src-role for a cross-board transfer).  With "
                     "no board the whole triple is transferred at once and no board is "
                     "privileged; use --src-pick trkpop there."
                     % getattr(a, "cell", "?"))
        # Rank the SOURCE's board, not the target's role name. `board` here is `sr`, a
        # TARGET role name (SPEC 3: both --xfer-board and --src-role are target role
        # names, because a --role-map has already re-keyed the source's OBJECTS onto the
        # target's names). But this function reads FILE NAMES out of the source tree, and
        # `pixel_tracks` resolves them with `parse_pixels(path).get(role)`, keyed by the
        # SOURCE's own role names. Measured on mar h1_run1 `extra0-ref1-dut2`: source-`dut`
        # ranks to R8C3 and source-`ref` to R12C1, so with
        # `--role-map dut:ref,ref:dut,extra:extra` a target role `ref` is fed by source
        # `dut`, and it is source-`dut`'s track that must be picked. The same map governs
        # the distinctness test below, which reads the chosen SOURCE files' own pixel
        # dicts. No map (every same-run cell): src_board is board.
        src_board = board
        if getattr(a, "role_map", None) and getattr(a, "combo", None):
            mp_, _why = _role_map_of(a, roles_of(a.combo))
            if mp_ is not None:
                src_board = mp_.get(board, board)
            if src_board != board:
                log("--role-map: the source pixels are ranked on the SOURCE's board "
                    "%r, which is what target role %r is fed from" % (src_board, board))
        for e in pixel_tracks(src_campaign, src_run, src_combo, src_board, src_root):
            nm = os.path.basename(e["best"])[:-8]
            if nm in chosen:
                continue
            # distinctness is on the SOURCE BOARD's pixel alone: that pixel is the only
            # thing being transferred, so two sources sharing a partner pixel on some
            # other board are still two different sources of the quantity under test.
            if any(e["pix"] == q.get(src_board) for q in chosen_pix):
                continue
            chosen.append(nm)
            chosen_pix.append(parse_pixels(e["best"]))
            meta[nm] = dict(e)
            if len(chosen) >= k:
                break
        if n_explicit:
            log("--src-triple pinned %d source(s); the rest were chosen by pixel "
                "population on board %s" % (n_explicit, src_board))
        return chosen, meta
    for p in top_triples(src_campaign, src_run, src_combo, 12 * k, src_root):
        nm = os.path.basename(p)[:-8]
        if nm in chosen:
            continue
        pix = parse_pixels(p)
        if any(any(pix.get(r) == q.get(r) for r in pix) for q in chosen_pix):
            continue                      # a board-pixel already used by another source
        chosen.append(nm)
        chosen_pix.append(pix)
        if len(chosen) >= k:
            break
    return chosen, meta


def _xfer_board_of(a):
    """Validate `--xfer-board` / `--src-role` and mode `cb`. Returns `(xb, sr)`.

    `(None, None)` is the whole-triple `all3` transfer.

    The rules, all refusals rather than silent no-ops:

    * the two flags mean something only in modes (c) and (cb), the two modes whose
      source is a NAMED pixel triple inside a run. Modes a/b/f are same-pixel by
      construction, (e) pools a board over its pixels and `inrun` splits the target,
      and mode (d) already declares its board -> board intent with `--role-map`;
    * mode `cb` REQUIRES both, and requires them to name different boards. That
      requirement IS its board-identity guard: the source board differs from the
      target board by design, so the role -> board assert (which compares role NAMES,
      and passes trivially for a source in the target's own combo) can never see the
      substitution. What makes it a declared intent rather than an unnoticed swap is
      what makes mode (d)'s exemption legitimate: the mapping DECLARED on the command
      line. Mode cb is therefore not exempt from the role -> board assert at all (it
      still has to be true that a role name means the same board on both sides, which
      is what `--role-map` fixes across campaigns); it cannot run without its own
      declaration;
    * mode (c) with the two flags naming DIFFERENT boards is mode cb, and says so;
    * `--src-role` without `--xfer-board` names the board the polynomial is fitted on but
      no board for it to land on, so the cell would still be the whole-triple `all3`
      transfer while `--src-pick auto` resolves to `pixpop` and picks the source by that
      board's pixels, with `xfer_board` and `src_role` blank on the row. Refused.
    """
    xb = getattr(a, "xfer_board", None) or None
    sr = getattr(a, "src_role", None) or None
    mode = a.mode
    if mode not in ("c", "cb") and (xb or sr):
        sys.exit("cell %s: --xfer-board / --src-role name the ONE target board that "
                 "carries a transferred polynomial, and they are defined only in modes "
                 "c (pixel -> board, same board) and cb (pixel -> OTHER board).  Mode %s "
                 "either has no named source pixel (e, inrun), is same-pixel by "
                 "construction (a, b, f), or already declares its board -> board intent "
                 "with --role-map (d)." % (a.cell, mode))
    if mode == "cb":
        if not xb or not sr:
            sys.exit("cell %s: mode cb (pixel -> OTHER board) needs BOTH --src-role "
                     "(the board the coefficients are fitted on, B1) and --xfer-board "
                     "(the board they are applied to, B2).  The source board differs "
                     "from the target board by design, so the role -> board assert - "
                     "which compares role NAMES and passes trivially inside one combo - "
                     "cannot see the substitution: the DECLARATION is the guard, exactly "
                     "as a --role-map is what exempts mode d.  Got --src-role %r "
                     "--xfer-board %r." % (a.cell, sr, xb))
        if sr == xb:
            sys.exit("cell %s: mode cb with --src-role == --xfer-board (%s) is not a "
                     "cross-board transfer at all - it is protocol P3, one pixel's "
                     "coefficients onto the same board of every other pixel.  Run it as "
                     "--mode c --xfer-board %s." % (a.cell, xb, xb))
    if sr and not xb:
        # see the last rule of the docstring: a whole-triple cell whose source was chosen
        # by a board rule the row cannot name is a silent change of meaning, not a no-op.
        # Measured on mar h1_run1: `--mode c` alone picks track_d-R14C1 (trkpop),
        # `--mode c --src-role dut` would pick track_d-R8C3, a different source of what
        # is still an all3 cell.
        sys.exit("cell %s: --src-role %s without --xfer-board.  --src-role names the "
                 "board the transferred polynomial is FITTED on; with no --xfer-board "
                 "there is no board for it to LAND on, so the cell is still the "
                 "whole-triple `all3` transfer - but --src-pick auto resolves to "
                 "`pixpop` and picks the source by ranking %s's pixels, and the row "
                 "records that rule with xfer_board and src_role blank.  A whole-triple "
                 "cell whose source was chosen by a board rule the row cannot name is a "
                 "silent change of meaning, not a no-op.  Add --xfer-board %s for "
                 "protocol P3, use --mode cb --src-role %s --xfer-board <other board> "
                 "for P4, or drop --src-role and the all3 cell picks its source by "
                 "`trkpop`." % (a.cell, sr, sr, sr, sr))
    if mode == "c" and xb and sr and sr != xb:
        sys.exit("cell %s: mode c applies a pixel's coefficients to the SAME board "
                 "(--src-role %s, --xfer-board %s name two different boards).  A "
                 "cross-board transfer is mode cb, which requires the declaration: "
                 "--mode cb --src-role %s --xfer-board %s."
                 % (a.cell, sr, xb, sr, xb))
    # --src-parity-contrast composes with --xfer-board: `_parity_contrast_arms` re-solves
    # the same joint fit and applies only the transferred board's coefficient, mirroring
    # `_one_board_arm`; `cmd_cell` passes xb/sr/base through.
    if xb and not sr:
        sr = xb
    if xb and a.combo:
        rs = roles_of(a.combo)
        # `sr` defaults to `xb` just above, so a P3 cell naming a role outside the combo
        # would print it twice: deduplicate, order-preserving.
        bad = [r for r in dict.fromkeys((xb, sr)) if r not in rs]
        if bad:
            sys.exit("cell %s: role(s) %s are not in the target combo %s (roles %s)"
                     % (a.cell, bad, a.combo, sorted(rs)))
    return xb, sr


def _triple_name_of(path):
    """`.../track_d-R8C3_....parquet` -> the triple name `events.Triple` reports.

    The dry run uses it to say how many targets a P3 cell will actually measure, which
    means comparing the source triple NAMES against the file list before any Triple is
    built.
    """
    b = os.path.basename(str(path))
    return b[:-8] if b.endswith(".parquet") else b


def _ntrip_of(a):
    """`--ntrip` as an int, with the literal `all` meaning every target triple.

    An uncapped whole-board cell is the point of the two pixel -> board protocols, and
    `--ntrip 2000` would be a guess at the file count, not a statement of intent. Only
    the word `all` is special: `--ntrip 0` means zero and `--ntrip -1` slices. The EVENT
    floors (`--nmin`, `--nmin-fold`) still remove the triples that are too small, and
    `--max-triples` / `--timebudget` are the job-sizing caps on top.
    """
    v = str(getattr(a, "ntrip", 12)).strip()
    if v.lower() == "all":
        return 10 ** 9
    try:
        return int(v)
    except ValueError:
        sys.exit("--ntrip must be an integer or the word 'all' (got %r)" % v)


def _source_text(a):
    """One line saying where a dry run's cell takes its source from."""
    if a.mode in ("a", "b", "f"):
        return ("the target's own pixels in %s %s %s"
                % (a.src_campaign or a.campaign, a.src_run or a.run, a.src_combo or a.combo))
    if a.mode == "e":
        return ("pooled over the %d largest triples of the target run, leave-one-pixel-out "
                "(scope %s)" % (a.pool_ntrip, a.pool_lopo_scope))
    if a.mode == "inrun":
        return "the other %s half of each target triple" % a.split
    return "named source triples in %s %s %s" % (a.src_campaign or a.campaign,
                                                 a.src_run or a.run, a.src_combo or a.combo)


def cmd_cell(a, root):
    """`cell` (alias `transfer`): one transfer cell, every arm, one CSV row per
    (triple, arm, quantity). The table goes to `--out` (a relative one in the write root), else
    <root>/cells/<cell>.csv, where `root` is the write root; an output inside a protected root
    is refused (campaigns.output_path). `--dry-run` prints the plan and writes nothing."""
    # --role-map has no meaning in the two modes with no named source: mode (e) builds its
    # source by pooling the target's own board across pixels and mode `inrun` splits the
    # target itself, so there is no second role -> board convention to reconcile.
    # Accepting and ignoring it would be a silent no-op, so it is an error.
    if getattr(a, "role_map", None) and a.mode in ("e", "inrun"):
        sys.exit("cell %s: --role-map is meaningless in mode %s (mode e pools the "
                 "TARGET's own board over its pixels, mode inrun splits the target "
                 "itself - neither has a source with its own role -> board map).  "
                 "Drop it, or use mode c/cb/d for "
                 "a cross-pixel / cross-board transfer." % (a.cell, a.mode))
    # the one transferred board (protocols P3 / P4), or (None, None) for the whole-triple
    # transfer. Validated before a single event is read; mode cb refuses to exist without
    # its declaration.
    xb, sr = _xfer_board_of(a)
    # --src-parity-contrast is a SAME-RUN measurement: the two arms fit the source on the
    # file parity that matches the evaluation fold and on the complementary one, and their
    # difference is the inter-board common mode the two SHARE because they are the same
    # files. With a source in another run or campaign the parities are unrelated, so the
    # difference would be fit-sample noise under a shared-drift label, written where the
    # aggregation picks the arms up beside genuine ones. Refused, not silently produced.
    # (`_parity_contrast_arms` itself does not enforce this, so the cross-campaign
    # permutation stays exercisable at the helper level.)
    if getattr(a, "src_parity_contrast", False) and not _src_is_same_run(a):
        sys.exit("cell %s: --src-parity-contrast needs a source in the TARGET's own "
                 "campaign and run (target %s %s, source %s %s).  The arms fit the "
                 "source on the matching and the complementary FILE PARITY and "
                 "difference them, which measures shared drift only when the source's "
                 "files are the target's files; across runs the two arms differ by "
                 "fit-sample noise alone.  Drop the flag for this cell."
                 % (a.cell, a.campaign, a.run,
                    getattr(a, "src_campaign", None) or a.campaign,
                    getattr(a, "src_run", None) or a.run))
    # `xfer_lin` (SPEC 4) fixes every board's quadratic term at the SOURCE's and refits
    # one linear coefficient per board jointly, so it needs a source polynomial for EVERY
    # board (the all3 shape) and needs --arms ext. Refused up front, before any event is
    # read, like the other flag-combination checks here.
    if getattr(a, "xfer_lin_arm", False) and xb is not None:
        sys.exit("cell %s: --xfer-lin is only defined for whole-triple (all3) cells: "
                 "it holds every board's quadratic term fixed at the SOURCE's and "
                 "needs a source polynomial for every board to hold fixed.  This is a "
                 "one-board cell (--xfer-board %s).  Drop --xfer-lin, or drop "
                 "--xfer-board and run the shape check as its own all3 cell."
                 % (a.cell, xb))
    if getattr(a, "xfer_lin_arm", False) and a.arms != "ext":
        sys.exit("cell %s: --xfer-lin needs --arms ext (xfer_lin is one of the extended "
                 "arms, SPEC 4)." % a.cell)
    if getattr(a, "xfer_lin_arm", False):
        # the degrees of freedom every verdict is implicitly compared against, once per
        # cell. The order-1 solve fits TWO raw coefficients per board (c1, c0). SPEC 2's
        # per-board gauge zero-mean correction and the per-fold median removal each remove
        # one location dof from every arm alike, so the EFFECTIVE ladder is 2/1/0
        # (self_cf/xfer_lin/xfer), not the raw fit counts 3/2/0.
        log("xfer_lin dof/board: self_cf fits %d raw order-%d coefficients, 2 "
            "effective (the SPEC 2 gauge zero-mean correction and the per-fold "
            "median removal each remove one location dof); xfer fits 0 "
            "(coefficients taken verbatim from the source); xfer_lin fits 2 raw "
            "order-1 coefficients (c1, c0 - c2 held fixed at the source's), 1 "
            "effective.  Effective ladder self_cf/xfer_lin/xfer = 2/1/0; a TOST "
            "verdict against self_cf compares a 2-effective-dof baseline, against "
            "xfer a 0-dof one." % (ORDER + 1, ORDER))
    dest = campaigns.output_path(a.out, root, "cells", "%s.csv" % a.cell)
    done = set()
    if os.path.exists(dest) and not a.overwrite:
        # Whether appending would mis-align the columns is knowable from the header alone,
        # before a single event is read; `_flush` checks it again before every append.
        with open(dest) as fh:
            head0 = fh.readline().strip().split(",")
        if head0 != CELL_COLUMNS:
            sys.exit("cell %s: %s has a different schema (%d cols) from CELL_COLUMNS "
                     "(%d), so appending to it would mis-align every column.  Re-run "
                     "with --overwrite to regenerate it, or point --out at a new file.  "
                     "(Checked at open time - nothing has been computed yet.)"
                     % (a.cell, dest, len(head0), len(CELL_COLUMNS)))
        old = pd.read_csv(dest)
        if "tgt_triple" not in old.columns:
            sys.exit("cell %s: %s is not a cell table - it has no `tgt_triple` column, "
                     "so the set of triples already done cannot be read from it "
                     "(columns: %s).  Point --out at a cell CSV written by "
                     "`python -m twc_transfer cell`, "
                     "or pass --overwrite to start it over."
                     % (a.cell, dest, list(old.columns)[:8]))
        done = set(old["tgt_triple"].unique())
        log("resuming: %d triples already in %s" % (len(done), dest))
    if getattr(a, "triples", None):
        d_ = os.path.join(run_dir(a.campaign, a.run, a.root), a.combo, "time")
        files = [os.path.join(d_, t.strip() + ".parquet") for t in a.triples.split(",")]
        files = [p_ for p_ in files if os.path.exists(p_)]
    else:
        files = top_triples(a.campaign, a.run, a.combo, _ntrip_of(a), a.root)
    if a.chunk:
        k, n = (int(x) for x in a.chunk.split("/"))
        files = files[k - 1::n]

    # ---- the source roster (modes c/cb/d rotate K pixels; a/b are same-pixel by
    #      construction and inrun/e have no named source)
    src_names, src_pick_meta, src_pick_by = [None], {}, {}
    if a.mode in ("c", "cb", "d"):
        # the board the pixels are ranked on is the board the coefficients are FITTED on,
        # i.e. `sr`; `xb` is where they land. For P3 the two coincide.
        src_names, src_pick_meta = _pick_sources(
            a, None, a.src_campaign or a.campaign, a.src_run or a.run,
            a.src_combo or a.combo, board=sr)
        if not src_names:
            sys.exit("cell %s: mode %s needs --src-triple or --src-k" % (a.cell, a.mode))
        _rule = _pick_rule(a, sr)
        _explicit = set(s.strip() for s in (a.src_triple or "").split(",") if s.strip())
        src_pick_by = {nm: ("explicit" if nm in _explicit else _rule) for nm in src_names}
        log("sources (n_src=%d, pick=%s%s): %s"
            % (len(src_names), _rule, (" on board %s" % sr) if sr else "",
               ", ".join(src_names)))
        for nm in src_names:
            e = src_pick_meta.get(nm)
            if e:
                log("  %s: pixel R%dC%d on %s, pop=%d over %d tracks, "
                    "largest track %d B" % (nm, e["pix"][0], e["pix"][1], sr,
                                            e["pop"], e["ntrk"], e["best_pop"]))
    # In P3 (`sr == xb`) the cell's own source triple is NOT one of its targets. For that
    # target `info["self_transfer"]` is true and SPEC 5.5's substitution puts board `sr`'s
    # fold-matched self fit on board `xb`; with `sr == xb` the whole `xfer` arm becomes
    # `self_cf` itself: delta exactly 0 on all three pairs and all three boards, se = 0.
    # There is no leak (the substitution is fold-matched and `assert_keys_disjoint` runs),
    # but a null IDENTICAL to the baseline measures nothing, and its exact zeros would sit
    # in the aggregation's pooled k_cal for the arm. In P4 (`sr != xb`) the source triple
    # stays a target: board `xb` genuinely takes board `sr`'s curve there, which is a real
    # substitution and the cell's own defining measurement.
    _p3_self_skip = set()
    if (xb is not None and sr == xb
            and (a.src_campaign or a.campaign) == a.campaign
            and (a.src_run or a.run) == a.run
            and (a.src_combo or a.combo) == a.combo):
        _p3_self_skip = set(nm for nm in src_names if nm)
        log("P3 (--xfer-board %s, source board %s): the cell's own source triple(s) "
            "%s are EXCLUDED from the target list - with sr == xb their `xfer` arm is "
            "`self_cf` itself (delta == 0 on every pair and board), a null identical "
            "to the baseline whose structural zeros would otherwise sit in agg's "
            "k_cal pool for the arm" % (xb, sr, ", ".join(sorted(_p3_self_skip))))
    if a.dry_run:
        # `notwc_1b` is built ONLY under --xfer-board, so an all3 cell builds 8 core arms
        # and a one-board cell 9; --src-parity-contrast adds xfer_srcmatch and
        # xfer_srccross to either shape.
        n_arms = len(ARMS_CORE) if xb is not None else len(ARMS_CORE) - 1
        if getattr(a, "src_parity_contrast", False):
            n_arms += 2
        print("cell %s: %d target triples%s x %d sources x %d arms; ~%.0f GMM fits%s"
              % (a.cell, len(files) - len(_p3_self_skip & set(
                     _triple_name_of(p_) for p_ in files)),
                 (" (capped at %d)" % a.max_triples) if a.max_triples else "",
                 len(src_names), n_arms,
                 min(len(files), a.max_triples or len(files))
                 * (5 + 3 * len(src_names)) * 9,
                 ("; one transferred board (%s <- %s)" % (xb, sr)) if xb else ""))
        targets = [_triple_name_of(p_) for p_ in files]
        print("  mode %s; target %s %s %s; source: %s"
              % (a.mode, a.campaign, a.run, a.combo, _source_text(a)))
        print("  targets (%d): %s" % (len(targets), ", ".join(targets)))
        if _p3_self_skip:
            print("  excluded as P3 self-transfer targets: %s"
                  % ", ".join(sorted(_p3_self_skip)))
        if done:
            print("  resume: %d target triples already in the table are skipped"
                  % len(done & set(targets)))
        print("  writes %s%s (write root %s)"
              % (dest, " (replacing it, --overwrite)"
                 if a.overwrite and os.path.exists(dest) else "", root))
        print("  dry run: nothing computed, nothing written")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if a.overwrite and os.path.exists(dest):
        os.remove(dest)

    pool_list = None
    if a.mode == "e":
        pool_files = top_triples(a.campaign, a.run, a.combo, a.pool_ntrip, a.root)
        pool_list = []
        for p in pool_files:
            t = Triple.from_file(p, a.campaign, a.run, a.combo, cap=a.pool_cap,
                                 neighbor=a.neighbor, neighbor_logic=a.neighbor_logic)
            if t.n >= a.nmin:
                pool_list.append(t)
        log("pool: %d triples, %d events" % (len(pool_list), sum(t.n for t in pool_list)))

    rows, t0, skipped, censored = [], time.time(), [], {}
    n_done = 0
    n_p3_skipped = 0
    # Per-triple guard provenance under --lpo-fallback own. Not a column of the cell
    # table: `_flush` enforces the fixed CELL_COLUMNS schema, and a new column would make
    # every default-flag table differ from the study's. It is written instead to a
    # side-car, `<out without .csv>.lpo_shape.json`, keyed by `tgt_triple`.
    lpo_shape_by_triple = {}
    _src_cache = {}
    for i, p in enumerate(files):
        tri0 = Triple.from_file(p, a.campaign, a.run, a.combo, cap=a.cap, neighbor=a.neighbor,
                                neighbor_logic=a.neighbor_logic)
        if tri0.name in done:
            continue
        if tri0.name in _p3_self_skip:
            # recorded in `skipped`, so it is printed with the cell's other exclusions at
            # the end of the run and is not a silent drop.
            skipped.append((tri0.name, "P3 self-transfer target: this triple IS the "
                            "cell's source pixel and sr == xb, so its `xfer` arm would "
                            "be `self_cf` itself (delta == 0 on every pair and board) - "
                            "a null identical to the baseline that measures nothing and "
                            "whose structural zeros would enter agg's k_cal pool"))
            n_p3_skipped += 1
            continue
        if tri0.n < a.nmin:
            skipped.append((tri0.name, "n=%d < %d" % (tri0.n, a.nmin)))
            continue
        tri = tri0
        src_specs = []          # (src_name, xfer_c, placebo_c, lpo_c, info, src_stats)

        if a.mode == "inrun":
            # --split block  : contiguous halves (drift charged to the transfer)
            # --split parity : file-stratified halves (SPEC 1's fold rule): the
            #                  same-config stability floor with the drift removed
            kA, kB = ("parityA", "parityB") if a.split == "parity" else ("A", "B")
            iA, iB = tri0.idx_block(kA), tri0.idx_block(kB)
            if len(iA) < a.nmin_src or len(iB) < a.nmin:
                skipped.append((tri0.name, "%s split too small (%d/%d)" % (a.split, len(iA), len(iB))))
                continue
            tA, aA = tri0.sub(iA)
            xfer_c = solve(tA, aA, tri0.roles)
            rng = np.random.default_rng(abs(hash((tri0.name, "pl"))) % (2 ** 31))
            placebo_c = solve(tA, aA, tri0.roles, permute=True, rng=rng)
            src_stats = {r: tot_stats(tA[r]) for r in tri0.roles}
            for r in tri0.roles:
                src_stats[r]["cal_mean"] = tri0.cal.get(r, np.nan)
            ent = tri0.key_entry(iA)
            info = dict(src_campaign=a.campaign, src_run=a.run + ":" + kA, src_combo=a.combo,
                        src_triple=tri0.name, n_fit=len(iA), self_transfer=False,
                        src_keys={0: [ent], 1: [ent]},
                        # source and target are two blocks of the SAME triple, so both
                        # sides mean the same board by each role name by construction.
                        keys_comparable=True,
                        src_files=np.unique(tri0.file[iA]),
                        src_pix=dict(tri0.pix), src_boards=dict(tri0.boards))
            # the leave-pair-out guard for the in-run cell: the sibling combo's events
            # restricted to the SAME source block (without it the cheat statistic of an
            # in-run cell would be NaN)
            lpo_c, lpo_keys = {}, []
            if a.lpo:
                fA = set(np.unique(tri0.file[iA]).tolist())
                for drop in tri0.roles:
                    got = lpo_events(tri0, drop, cap=N_CAP, root=a.root,
                                     neighbor=a.neighbor, neighbor_logic=a.neighbor_logic)
                    if got is None:
                        lpo_c, lpo_keys = {}, []
                        break
                    ltot, ltoa, lfile, lroles, sib, nl = got
                    m = np.where(np.isin(lfile, list(fA)))[0]
                    if len(m) < 50:
                        lpo_c, lpo_keys = {}, []
                        break
                    lpo_c[drop] = solve({r: ltot[r][m] for r in lroles},
                                        {r: ltoa[r][m] for r in lroles}, lroles)
                    kept = tuple(sorted(r for r in tri0.roles if r != drop))
                    lpo_keys.append((kept, event_keys(lfile[m], {r: ltoa[r][m] for r in lroles}, kept)))
            info["lpo_keys"] = lpo_keys
            tot_b, toa_b = tri0.sub(iB)
            tri = Triple(tot_b, toa_b, tri0.roles, file=tri0.file[iB], path=tri0.path,
                         campaign=tri0.campaign, run=tri0.run, combo=tri0.combo, cal=tri0.cal)
            if a.split == "parity":
                # a parity half has a constant file parity: re-derive the cross-fit folds
                # from the RANK of the files actually present, else fold 1 is empty and
                # every self arm degrades to notwc without failing
                tri.refold_by_rank()
                if len(tri.idx_fold(0)) < 50 or len(tri.idx_fold(1)) < 50:
                    skipped.append((tri.name, "parity half has too few distinct files"))
                    continue
            src_specs = [(tri0.name, xfer_c, placebo_c, lpo_c, info, src_stats)]
        elif a.mode == "e":
            got = pooled_source(a, pool_list, target=tri0, cap_per=a.pool_cap,
                                exclude_key=tri0.key, scope=a.pool_lopo_scope)
            if got is None:
                skipped.append((tri0.name, "empty pool after the per-board exclusion"))
                continue
            info = dict(src_campaign=a.campaign, src_run=a.run + ":pooled_LOPO", src_combo=a.combo,
                        src_triple="pooled", n_fit=got["n_fit"], self_transfer=False,
                        c_avg=got["c_avg"], n_pool_excluded=got["n_excluded"],
                        src_keys={0: got["keys"], 1: got["keys"]}, src_files=None,
                        # the pool is drawn from the TARGET's own campaign/run/combo, so
                        # the keys are comparable by construction (no `src_boards` is
                        # reported for a pool; see `_board_mismatch`'s empty-dict case).
                        keys_comparable=True,
                        src_pix={}, src_boards={})
            lpo_c, lpo_keys = {}, []
            if a.lpo:
                lpo_c, lpo_keys = pooled_lpo_source(a, pool_list, tri0, cap_per=a.pool_cap,
                                                    nmax=a.pool_lpo_ntrip)
            info["lpo_keys"] = lpo_keys
            src_specs = [("pooled", got["c_pool"], None, lpo_c, info, got["stats"])]
        else:
            for nm in src_names:
                # the source of a mode a/b/c/d/f cell does not depend on the target (only
                # the self_transfer flag does), so it is resolved ONCE per cell and key:
                # with K rotated sources and N targets it is not re-read and re-solved N
                # times. In modes a/b/f the key holds the target's own track name, so
                # there the source is read once per target.
                ck = (nm, a.src_run or a.run, a.src_combo or a.combo,
                      (_src_track_name(tri0, a, a.src_combo or a.combo)[0] or "")
                      if a.mode in ("a", "b", "f") else "")
                if ck not in _src_cache:
                    _src_cache[ck] = source_coeffs_for(tri0, a, src_name=nm)
                xfer_c, info = _src_cache[ck]
                if xfer_c is None:
                    skipped.append((tri0.name, "src=%s: %s" % (nm, info)))
                    continue
                # a whole-triple transfer needs every board; a one-board transfer needs
                # exactly the board it takes the polynomial FROM. Demanding all three
                # there would skip a target for the absence of a coefficient the cell
                # never applies.
                missing = ({sr} - set(xfer_c)) if xb else (set(tri0.roles) - set(xfer_c))
                if missing:
                    skipped.append((tri0.name, "src=%s has no coefficients for %s"
                                    % (nm, sorted(missing))))
                    continue
                # the self_transfer flag of this target: the source triple IS this target
                # triple, in the same run and combo (SPEC 5.5).
                info = dict(info)
                info["self_transfer"] = bool(
                    info.get("src_triple") == tri0.name
                    and (a.src_run or a.run) == tri0.run
                    and (a.src_combo or a.combo) == tri0.combo)
                src_specs.append((info.get("src_triple", nm), xfer_c, info["placebo"],
                                  info.get("lpo", {}), info, info["stats"]))
        if not src_specs:
            continue

        # ---- SPEC 5 / 8: the evaluation folds must be big enough for the estimator.
        # Below ~300 events per fold the GMM returns sigma = 0 with accepted = 1 and a
        # fake Delta of exactly 0.00 ps, so the whole TRIPLE is dropped: symmetrically,
        # across every arm, on a quality criterion only.
        nf0, nf1 = len(tri.idx_fold(0)), len(tri.idx_fold(1))
        if min(nf0, nf1) < a.nmin_fold:
            skipped.append((tri.name, "fold too small (%d/%d < %d)" % (nf0, nf1, a.nmin_fold)))
            continue

        folds = (0, 1)
        maps = self_coeff_maps(tri, folds=folds, root=a.root, want_lpo=a.lpo,
                               neighbor=a.neighbor, neighbor_logic=a.neighbor_logic,
                               lpo_fallback=getattr(a, "lpo_fallback", "sibling"))
        self_arms = {k: v for k, v in maps.items()}
        self_keys = dict(maps.src_keys)
        if xb is not None and xb in tri.roles:
            # ---- the reference the PLACEBO of a one-board cell is read against (SPEC
            # 5.2). The placebo transfers a permuted-TOT source fit and must come out where
            # an UNCORRECTED board would; in an all3 cell that is the plain `notwc` arm,
            # but here only board `xb` is uncorrected while the other two carry their own
            # self_cf fits, so `notwc` is the wrong yardstick by two boards' worth of
            # correction. `notwc_1b` is that yardstick: zero on `xb`, self_cf elsewhere.
            # It is also the `G` of a one-board retention, R = 1 - Delta/G with
            # G = sigma_notwc_1b - sigma_self_cf.
            m_nz = dict(maps["self_cf"])
            for f in folds:
                m_nz[(f, xb)] = np.zeros(ORDER + 1)
            self_arms["notwc_1b"] = m_nz
            # the two untouched boards ARE fold-matched self fits, so this arm carries
            # self_cf's source keys and the SPEC 1 guard runs on it like any other.
            self_keys["notwc_1b"] = maps.src_keys.get("self_cf")
        if a.arms == "ext":
            self_arms["self_in_rep"] = maps["self_in"]
            self_keys["self_in_rep"] = None

        for src_i, (sname, xfer_c, placebo_c, lpo_c, info, src_stats) in enumerate(src_specs):
            # ---- SPEC 1 invariant, structural pre-filter (it RAISES for inrun: the two
            # halves are passed as the two folds of the SAME id, so a split whose halves
            # coincide dies here)
            sid = (info.get("src_campaign"), info.get("src_run"), info.get("src_combo"),
                   info.get("src_triple"))
            tid = (tri.campaign, tri.run, tri.combo, tri.name)

            # ---- BOARD IDENTITY, unconditional, in every mode but (d) with a role map.
            # Modes a/b/f pick the source triple by the TARGET's role -> pixel map, and
            # role sets are not board identity (see BoardIdentityError). This is an error,
            # never a warning: the fix is --role-map 'tgt:src,...'. Mode (d) is exempt ONLY
            # once a --role-map has been supplied, because a mode-d map deliberately puts
            # board A's coefficients on board B: a declared intent. Mode (d) with NO map
            # applies the source role-name-for-role-name exactly like every other mode, so
            # it is asserted like every other mode.
            if a.mode != "d" or not getattr(a, "role_map", None):
                sb_chk = info.get("src_boards") or {}
                # a role the source does not report counts as a MISMATCH, as in
                # `_boards_shared`; an EMPTY dict (modes e / inrun) is the one exemption,
                # see `_board_mismatch`.
                bad = _board_mismatch(sb_chk, tri.boards, tri.roles)
                if bad:
                    # the suggested map is offered ONLY when a complete bijection exists;
                    # otherwise both board maps are printed. It is computed from the
                    # source's OWN board map (`src_boards_own`), never from the permuted
                    # `src_boards`: with a --role-map already given the latter yields the
                    # RESIDUAL map, which is not what the user must pass.
                    sb_own = info.get("src_boards_own") or sb_chk
                    fix = _board_identity_fix(sb_own, tri.boards, tri.roles)
                    raise BoardIdentityError(
                        "cell %s, mode %s: source (%s %s %s) and target (%s %s %s) put "
                        "DIFFERENT BOARDS under the same role(s) %s - source boards %s "
                        "vs target boards %s, on target triple %s.  Role sets match but "
                        "the board identity does not, so the transfer would apply each "
                        "board's coefficients to the wrong board.  Fix: %s.%s"
                        % (a.cell, a.mode, info.get("src_campaign"), info.get("src_run"),
                           info.get("src_combo"), tri.campaign, tri.run, tri.combo,
                           sorted(bad), {r: sb_chk.get(r) for r in sorted(bad)},
                           {r: tri.boards.get(r) for r in sorted(bad)}, tri.name, fix,
                           ("  Mode (d) is exempt from this assert only once a "
                            "--role-map has DECLARED the board -> board mapping, which "
                            "is why an unmapped mode-d cell is refused like any other."
                            if a.mode == "d" else
                            "  If the two boards really are meant to be swapped, DECLARE "
                            "it with --role-map; otherwise the board maps themselves are "
                            "what to check (the combo strings of the two runs).")))

            if a.mode == "inrun":
                assert_disjoint(tid, tid, kA, kB)
            elif not info.get("self_transfer"):
                assert_disjoint(sid, tid, None, None)

            arms = dict(self_arms) if src_i == 0 else {}
            keys = dict(self_keys) if src_i == 0 else {}
            n_guard = np.nan
            if xb is not None and xb not in tri.roles:
                skipped.append((tri.name, "--xfer-board %s is not a role of %s"
                                % (xb, tri.combo)))
                continue
            if xb is not None:
                # ---- PROTOCOLS P3 / P4: ONE board carries the transferred polynomial and
                # every other board keeps its own fold-matched cross-fit self coefficients,
                # in EVERY transfer-derived arm: xfer, placebo, the leave-pair-out guard and
                # the extended variants alike. Anything less would compare a one-board
                # transfer against a three-board placebo.
                if info.get("self_transfer"):
                    # SPEC 5.5, one board at a time: the source triple IS this target, so
                    # the polynomial put on `xb` must be the FOLD-MATCHED self fit of board
                    # `sr`, which is still a genuine substitution (a different board's
                    # curve, or the same board's own for P3) but contains no evaluated
                    # event. The source's own in-sample fit would leak, and raises.
                    m_sx = dict(maps["self_cf"])
                    for f in folds:
                        m_sx[(f, xb)] = maps["self_cf"][(f, sr)]
                    arms["xfer"] = m_sx
                    keys["xfer"] = maps.src_keys.get("self_cf")
                    placebo_c, lpo_c = None, None
                    if "placebo_self" in maps:
                        m_pl = dict(maps["self_cf"])
                        for f in folds:
                            m_pl[(f, xb)] = maps["placebo_self"][(f, sr)]
                        arms["placebo"] = m_pl
                        keys["placebo"] = maps.src_keys.get("placebo_self")
                    if "self_cf_lpo" in maps:
                        # the guarded version of the SAME substitution: board xb takes
                        # board sr's leave-pair-out fit, every other board keeps its own.
                        # (For P3, sr == xb, and this collapses to `self_cf_lpo` itself.)
                        pr_ = {frozenset((x_, y_)): "%s-%s" % (x_, y_)
                               for x_, y_ in combinations(tri.roles, 2)}
                        m_l = dict(maps["self_cf_lpo"])
                        n_guard = 0
                        for f in folds:
                            m_l[(f, xb)] = maps["self_cf"][(f, sr)]
                        for y_ in tri.roles:
                            if y_ == xb:
                                continue
                            pk_ = pr_[frozenset((xb, y_))]
                            sk_ = (None if y_ == sr
                                   else pr_[frozenset((sr, y_))])
                            got_ = sk_ is not None and all(
                                (f, sk_, sr) in maps["self_cf_lpo"] for f in folds)
                            for f in folds:
                                m_l[(f, pk_, xb)] = (maps["self_cf_lpo"][(f, sk_, sr)]
                                                     if got_ else
                                                     maps["self_cf"][(f, sr)])
                            n_guard += int(got_)
                        arms["xfer_lpo"] = m_l
                        keys["xfer_lpo"] = maps.src_keys.get("self_cf_lpo")
                else:
                    arms["xfer"], _ = _one_board_arm(tri, maps["self_cf"], xb, sr,
                                                     xfer_c, folds)
                    keys["xfer"] = info.get("src_keys")
                    if placebo_c is not None and sr in placebo_c:
                        arms["placebo"], _ = _one_board_arm(tri, maps["self_cf"], xb, sr,
                                                            placebo_c, folds)
                        keys["placebo"] = info.get("src_keys")
                    if lpo_c:
                        m, n_guard = _one_board_arm(
                            tri, maps["self_cf"], xb, sr, xfer_c, folds,
                            lpo=lpo_c, base_lpo=maps.get("self_cf_lpo"))
                        if n_guard:
                            arms["xfer_lpo"] = m
                            lk = list(info.get("lpo_keys") or [])
                            if info.get("src_keys"):
                                lk = list(info["src_keys"].get(0, [])) + lk
                            keys["xfer_lpo"] = {0: lk, 1: lk} if lk else None
                        else:
                            log("%s: no leave-pair-out guard is definable for a "
                                "one-board transfer of %s <- %s (the only source "
                                "solve that excludes a partner also excludes the "
                                "source board itself); xfer_lpo not built"
                                % (tri.name, xb, sr))
            elif info.get("self_transfer"):
                # SPEC 5.5's self-transfer null: the source IS this target, so every arm of
                # it must be the FOLD-MATCHED self version. The source-side placebo and LPO
                # fits were made on all of the source's events, i.e. on the evaluated ones:
                # using them here would leak (and raises).
                arms["xfer"] = maps["self_cf"]
                keys["xfer"] = maps.src_keys.get("self_cf")
                placebo_c, lpo_c = None, None
                if "placebo_self" in maps:
                    arms["placebo"] = maps["placebo_self"]
                    keys["placebo"] = maps.src_keys.get("placebo_self")
                if "self_cf_lpo" in maps:
                    arms["xfer_lpo"] = maps["self_cf_lpo"]
                    keys["xfer_lpo"] = maps.src_keys.get("self_cf_lpo")
            else:
                arms["xfer"] = transferred_map(tri, xfer_c, folds=folds)
                keys["xfer"] = info.get("src_keys")
            # the two whole-triple blocks below are for the `all3` shape only: the
            # one-board branch above has already built its own placebo and its own guard,
            # on the same one-board shape. Letting these run for it would put a
            # THREE-board placebo beside a one-board transfer.
            if xb is None and placebo_c is not None:
                arms["placebo"] = transferred_map(tri, placebo_c, folds=folds)
                keys["placebo"] = info.get("src_keys")
            if xb is None and lpo_c:
                m = dict(arms["xfer"])
                built = False
                # under --lpo-fallback own, count the guard SLOTS actually filled (up to 2
                # per pair x 3 pairs = 6), so that `lpo_pairs` is a number on the row here
                # too (SPEC 5.1: the guard's extent is a number, not an assumption). Under
                # the default flag it stays NaN, as in the study's default-flag cells.
                fb_own = (getattr(a, "lpo_fallback", "sibling") == "own")
                n_guard_all3 = 0 if fb_own else None
                for x, y in combinations(tri.roles, 2):
                    pair = "%s-%s" % (x, y)
                    if y in lpo_c and x in lpo_c[y]:
                        m[(0, pair, x)] = lpo_c[y][x]
                        m[(1, pair, x)] = lpo_c[y][x]
                        built = True
                        if n_guard_all3 is not None:
                            n_guard_all3 += 1
                    if x in lpo_c and y in lpo_c[x]:
                        m[(0, pair, y)] = lpo_c[x][y]
                        m[(1, pair, y)] = lpo_c[x][y]
                        built = True
                        if n_guard_all3 is not None:
                            n_guard_all3 += 1
                if built:
                    arms["xfer_lpo"] = m
                    lk = list(info.get("lpo_keys") or [])
                    if info.get("src_keys"):
                        lk = list(info["src_keys"].get(0, [])) + lk
                    keys["xfer_lpo"] = {0: lk, 1: lk} if lk else None
                    if n_guard_all3 is not None:
                        n_guard = n_guard_all3

            # ---- shared-drift optimism. Built by `_parity_contrast_arms`, which applies
            # the SAME role permutation as the main transfer and asserts that the boards
            # these arms draw from are the target's own: they are re-solved from the
            # unpermuted source Triple.
            if (getattr(a, "src_parity_contrast", False) and info.get("src_obj") is not None
                    and not info.get("self_transfer") and _src_is_same_run(a)):
                pc_arms, pc_keys, _pc_boards = _parity_contrast_arms(
                    a, tri, info, folds, xb=xb, sr=sr,
                    base=(maps["self_cf"] if xb is not None else None))
                arms.update(pc_arms)
                keys.update(pc_keys)

            if a.arms == "ext":
                for v in ("clip", "shift", "rescale", "calfix"):
                    if xb is None:
                        arms["xfer_%s" % v] = {k: VariantCoeff(c, v, src_stats.get(k[1], {}),
                                                               tri.cal.get(k[1]))
                                               for k, c in arms["xfer"].items() if len(k) == 2}
                    else:
                        # only the ONE transferred board gets a variant; the others are
                        # this target's own self fits, and clipping / rescaling them
                        # against the SOURCE's TOT support would be meaningless. The support
                        # that matters is the source BOARD's (`sr`), not the target
                        # board's, which is the point of a cross-board cell.
                        m_v = dict(arms["xfer"])
                        for f in folds:
                            m_v[(f, xb)] = VariantCoeff(arms["xfer"][(f, xb)], v,
                                                        src_stats.get(sr, {}),
                                                        tri.cal.get(xb))
                        arms["xfer_%s" % v] = m_v
                    keys["xfer_%s" % v] = keys.get("xfer")
                if xb is None:
                    for r in tri.roles:
                        m = dict(maps["self_cf"])
                        if r in xfer_c:
                            for f in folds:
                                m[(f, r)] = xfer_c[r]
                        arms["xfer_1b:%s" % r] = m
                        keys["xfer_1b:%s" % r] = _merge_keys(keys.get("xfer"),
                                                             maps.src_keys.get("self_cf"))
                # else: `xfer` IS the one-board arm, so xfer_1b:<board> would be a
                # duplicate of it under a second name (and, for cb, of the wrong shape).
                if a.mode == "e" and info.get("c_avg"):
                    arms["xfer_avg"] = transferred_map(tri, info["c_avg"], folds=folds)
                    keys["xfer_avg"] = keys.get("xfer")
                if getattr(a, "xfer_lin_arm", False) and xb is None:
                    if xfer_c is not None and set(xfer_c) >= set(tri.roles):
                        # SPEC 4's xfer_lin: the source's quadratic term stays fixed, the
                        # linear term is refit fold-matched on the TARGET's own events,
                        # jointly across all three boards (arms.xfer_lin_map, the same
                        # unmodified TS.solve_timewalk_coefficients at order=1).
                        arms["xfer_lin"] = xfer_lin_map(tri, xfer_c, folds=folds)
                        keys["xfer_lin"] = keys.get("xfer")
                        if lpo_c and all(y in lpo_c for y in tri.roles):
                            ml = xfer_lin_lpo_map(tri, xfer_c, lpo_c, folds=folds)
                            if ml:
                                arms["xfer_lin_lpo"] = ml
                                keys["xfer_lin_lpo"] = keys.get("xfer_lpo")
                    else:
                        # --xfer-lin is refused earlier for a one-board cell (xb is
                        # always None here), so the only way in is a source missing a
                        # coefficient for one of this triple's roles; said, not skipped
                        # silently.
                        log("%s: --xfer-lin requested but source %s has no "
                            "coefficient for role(s) %s - xfer_lin/xfer_lin_lpo "
                            "skipped for this triple/source"
                            % (tri.name, sname,
                               sorted(set(tri.roles) - set(xfer_c or {}))))

            ov = {}
            if src_stats:
                if xb is not None:
                    # the coverage / overlap / extrapolation triad of SPEC 5.3 compares the
                    # polynomial's own TOT support with the support it is evaluated over.
                    # For a one-board transfer that is source board `sr`'s support against
                    # TARGET board `xb`'s TOT, across boards in a cb cell, which is exactly
                    # where the EXTRAP flag has to be able to fire.
                    if sr in src_stats and sr in (xfer_c or {}):
                        ov[xb] = overlap_metrics(src_stats[sr], tri.tot[xb], xfer_c[sr])
                else:
                    for r in tri.roles:
                        if r in src_stats and r in (xfer_c or {}):
                            ov[r] = overlap_metrics(src_stats[r], tri.tot[r], xfer_c[r])

            # per-board source bookkeeping
            spix, sboard = info.get("src_pix") or {}, info.get("src_boards") or {}
            nexc = info.get("n_pool_excluded") or {}
            # `same_pixel` means a SAME-RUN, SAME-COMBO refit of that board's own pixel on
            # a different sample of the same run: a disguised self-transfer. For a
            # cross-run or cross-angle cell (modes a/b/f) the same board-pixel is the
            # whole point of the comparison, so the flag does not apply there and stays
            # NaN.
            same_run = (str(info.get("src_run", "")) == str(tri.run)
                        and str(info.get("src_combo", "")) == str(tri.combo))
            per_board = {}
            for r in tri.roles:
                applies = same_run and a.mode not in ("inrun", "e")
                # For a ONE-BOARD transfer only board `xb` has a source at all, and the
                # pixel it must be compared against is the source's pixel on board `sr`
                # (the board the polynomial was fitted on), not on `xb`. In P3 the two
                # coincide; in P4 `same_board` is 0 by construction, which is the cell's
                # defining property rather than a warning. Every other board carries no
                # source, so NaN, not 0.
                s_r = sr if (xb is not None) else r
                on = (xb is None) or (r == xb)
                per_board[r] = dict(
                    same_pixel=(int(spix.get(s_r) == tri.pix.get(r))
                                if applies and on and s_r in spix else np.nan),
                    same_board=(int(sboard.get(s_r) == tri.boards.get(r))
                                if applies and on and s_r in sboard else np.nan),
                    n_pool_excluded=nexc.get(r, np.nan))
            # SPEC 1 bookkeeping beside src_keys_checked: were the source's per-event keys
            # comparable with the target's at all? 0 with keys_comparable = 1 is "no source
            # event was read" (a bank-backed source); 0 with 0 is "the two sides put
            # different boards under the same role name". Blank only where there is no
            # source triple (the self arms).
            kcmp = info.get("keys_comparable")
            kcmp = int(bool(kcmp)) if kcmp is not None else np.nan
            sfiles = info.get("src_files")
            ev_files = set(np.unique(tri.file).tolist())
            overlap = (len(ev_files & set(np.asarray(sfiles).tolist())) / max(len(ev_files), 1)
                       if sfiles is not None else np.nan)

            # ---- which board carries the transferred polynomial, whose it is, and the
            # evidence for calling that source pixel "the most populous". `src_pix` is
            # keyed by TARGET role names (a --role-map has already re-keyed it), so the
            # pixel that supplies board `xb` is the one filed under `sr`.
            pk_e = src_pick_meta.get(sname) or {}
            s_px = spix.get(sr) if xb is not None else None
            meta = dict(cell=a.cell, mode=a.mode, campaign=a.campaign, run=a.run, combo=a.combo,
                        tgt_triple=tri.name, tgt_key=tri.key, src_run=info.get("src_run", ""),
                        src_combo=info.get("src_combo", ""), src_triple=info.get("src_triple", ""),
                        n_fit_src=info.get("n_fit", np.nan), rho=tri.rho(),
                        self_transfer=int(bool(info.get("self_transfer"))),
                        src_file_overlap=overlap, keys_comparable=kcmp,
                        xfer_board=(xb or ""), src_role=(sr if xb is not None else ""),
                        src_pick=src_pick_by.get(sname, ""),
                        src_pix_rc=("R%dC%d" % s_px if s_px is not None else ""),
                        src_pix_ntrk=pk_e.get("ntrk", np.nan),
                        src_pix_pop=pk_e.get("pop", np.nan),
                        lpo_pairs=n_guard,
                        ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
            # `lpo_shape` names which construction (sibling combo or the triple's own
            # events) built each dropped role's guard: `self` is the TARGET side
            # (arms.self_coeff_maps, feeds self_cf_lpo and the self-LPO cost), `xfer` the
            # SOURCE side (the `lpo` block of source_coeffs_for, feeds xfer_lpo). Written
            # to the side-car, not to the row (see `lpo_shape_by_triple` above).
            if getattr(a, "lpo_fallback", "sibling") == "own":
                lpo_shape_by_triple[tri.name] = dict(
                    self=dict(getattr(maps, "lpo_shape", {}) or {}),
                    xfer=dict(info.get("lpo_shape") or {}))
            for arm, cmap in arms.items():
                if (a.arms == "core" and arm not in ARMS_CORE
                        and not arm.startswith("xfer_src")):
                    continue
                try:
                    ev = evaluate_arm(tri, cmap, folds=folds, per_fold=not a.fast,
                                      src_keys=keys.get(arm), arm=arm)
                except LeakageError:
                    raise
                except Exception as e:
                    log("arm %s failed on %s: %r" % (arm, tri.name, e))
                    continue
                nz = sum(v for v in ev["neg_radicand"].values())
                if nz:
                    censored[arm] = censored.get(arm, 0) + int(nz)
                m2 = dict(meta, src_keys_checked=ev.get("n_leak_checks", 0))
                if arm in self_arms and src_i == 0:
                    # a self arm has no source of its own: blank the source identity and
                    # its self_transfer flag, so it is not split into two aggregate groups
                    # just because one of the cell's targets happened to be a source
                    m2.update(src_triple="", src_run="", src_combo="", n_fit_src=np.nan,
                              src_file_overlap=np.nan, self_transfer=0,
                              keys_comparable=np.nan,
                              # ... and the source columns of the protocols with it: a self
                              # arm replaces NO board and was picked by no rule, so these
                              # are blank rather than inheriting the cell's source.
                              xfer_board="", src_role="", src_pick="", src_pix_rc="",
                              src_pix_ntrk=np.nan, src_pix_pop=np.nan, lpo_pairs=np.nan)
                rows += _rows_for_arm(arm, ev, tri, maps["self_cf"], m2,
                                      ov if arm.startswith("xfer") else {},
                                      per_board if arm.startswith("xfer") else None)
        n_done += 1
        log("%s [%d/%d] n=%d src=%d  %.0fs  (done %d)"
            % (tri.name, i + 1, len(files), tri.n, len(src_specs),
               time.time() - t0, n_done))
        if rows and (i + 1) % max(a.flush, 1) == 0:
            _flush(rows, dest)
            rows = []
        # ---- the two job-sizing caps. `--ntrip all` makes the target list the whole
        # board, so the cap that bounds a job is a count of triples ACTUALLY measured (the
        # ones that passed the event floors, not the ones offered) or a wall-clock budget.
        # Both leave the CSV resumable, so a capped job is a prefix of the uncapped one
        # rather than a different cell.
        if a.max_triples and n_done >= a.max_triples:
            log("--max-triples %d reached, stopping (resumable)" % a.max_triples)
            break
        if a.timebudget and time.time() - t0 > a.timebudget:
            log("time budget reached, stopping (resumable)")
            break
    if rows:
        _flush(rows, dest)
    for name, why in skipped:
        log("skipped %s: %s" % (name, why))
    if n_p3_skipped:
        # the cell's own record of the exclusion, beside the per-triple lines above, so a
        # reader of the log does not have to count them.
        log("P3 self-transfer targets skipped: %d of the %d source triple(s) of this "
            "cell appeared in the target list and were excluded (sr == xb, so their "
            "xfer arm is the baseline); no self_cf-identical xfer arm reaches agg"
            % (n_p3_skipped, len(_p3_self_skip)))
    if censored:
        log("degenerate (neg-radicand) board sigmas censored to NaN, per arm: %s"
            % dict(sorted(censored.items())))
    if lpo_shape_by_triple:
        # the guard-provenance side-car (see the accumulator above), merged with whatever
        # an earlier run on the same table already wrote, keyed by tgt_triple, so a
        # resumed cell's side-car is a union as the CSV itself is. It is not written under
        # a lock: two --chunk jobs writing the SAME side-car concurrently can race each
        # other's read-modify-write (the CSV has no such race, `_flush` only appends).
        # --overwrite removes the CSV but not the side-car, whose entries are merged.
        shp = os.path.splitext(dest)[0] + ".lpo_shape.json"
        prior = {}
        if os.path.exists(shp):
            try:
                with open(shp) as fh:
                    prior = json.load(fh)
            except Exception as e:
                log("lpo_shape side-car %s unreadable, overwriting: %r" % (shp, e))
        prior.update(lpo_shape_by_triple)
        with open(shp, "w") as fh:
            json.dump(prior, fh)
        log("lpo_shape guard-provenance side-car: %s (%d triples)" % (shp, len(prior)))
    print("cell %s -> %s  [%.0fs]" % (a.cell, dest, time.time() - t0))


def _flush(rows, dest):
    """Append with a FIXED column list and a header check.

    An append that wrote a header only when the file did not exist would lay new-order
    rows under an old header after any change of the columns and mis-align every column
    from there on. The schema is CELL_COLUMNS, and a mismatch is a hard error that names
    the fix rather than a silent corruption.
    """
    df = pd.DataFrame(rows)
    for c in CELL_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    extra = [c for c in df.columns if c not in CELL_COLUMNS]
    if extra:
        raise SystemExit("_flush: rows carry columns outside CELL_COLUMNS: %s "
                         "(add them to the schema)" % extra)
    df = df[CELL_COLUMNS]
    if os.path.exists(dest):
        with open(dest) as fh:
            head = fh.readline().strip().split(",")
        if head != CELL_COLUMNS:
            raise SystemExit(
                "_flush: %s has a different schema (%d cols) from CELL_COLUMNS (%d). "
                "Appending would mis-align every column - re-run the cell with "
                "--overwrite to regenerate it." % (dest, len(head), len(CELL_COLUMNS)))
        df.to_csv(dest, mode="a", header=False, index=False)
    else:
        df.to_csv(dest, mode="w", header=True, index=False)
