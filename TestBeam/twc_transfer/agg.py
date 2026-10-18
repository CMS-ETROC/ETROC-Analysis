"""The aggregation (SPEC.md 8): from the tables of a group of cells to the paired difference of
every triple against the baseline arm, the aggregate per (cell, arm, quantity) and the error
calibration.

The cell tables of a group (cell.CELL_COLUMNS, plus the lpo_shape side-car of a cell run with
--lpo-fallback own) are read and stacked. Rows are then removed, in this order and only by
these rules, each a function of its own:

  drop_below_nmin_eval    a triple whose board rows were evaluated on fewer than n_eval events
  censor_degenerate       a (triple, quantity) with a degenerate value in any arm, from every arm
  drop_identical_pairs    on a one-board cell, the pair without the transferred board in the
                          arms built on self_cf (is_one_board_arm): identical to the baseline

Every remaining arm is differenced against the baseline arm per (triple, quantity) (deltas).
Two kinds of row then leave the transfer aggregate, the k_cal pool or both, and stay in the
per-triple table: same-run same-pixel refits filed as transfers (same_pixel_mask) and, for
k_cal only, transfer rows whose source triple is one of the cell's targets
(kcal_exclusion_mask). k_cal is pooled per (cell, run, arm, kind) (kcal_by_group); a pool that
still holds a row identical to the baseline on a one-board pair stops the aggregation before
anything is written (structural_zero_census, StructuralZeroError). The aggregate per (cell,
mode, arm, kind, name, self_transfer, extrapolation family) is DerSimonian-Laird with the
source-to-source term, TOST against the margin, the guard verdict of the leave-pair-out
control (guard_verdict, tost_label), the primary flag (primary_flag) and Holm over each cell's
board FAIL claims (holm_board_fails).

The error inflation f is given explicitly, as a campaign's value from campaigns.F_CAL, or read
from the read root's tables/f_cal.json (f_inflate_for).

aggregate() computes the three tables in memory and writes nothing; cmd_agg reads the cell
tables, calls it and writes the tables under the write root.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from . import campaigns
from .cell import CELL_COLUMNS
from .naming import log
from .stats import (holm, kcal_from_cells, margin_for, p_nonequiv, random_effects,
                    source_random_effect, tost)

# arms with no source: never split by self_transfer, never excluded from k_cal on it
SELF_ARMS = ("notwc", "notwc_1b", "self_in", "self_cf", "self_cf_lpo", "placebo_self",
             "self_in_rep")
TRIPLE_KEY = ["cell", "run", "combo", "tgt_triple"]
QUANTITY_KEY = ["cell", "run", "combo", "tgt_triple", "kind", "name"]
DELTA_KEY = ["cell", "mode", "run", "combo", "tgt_triple", "kind", "name"]
KCAL_GROUP = ["cell", "run", "arm", "kind"]
# the columns delta_by_triple.csv adds to the cell table's
DELTA_COLUMNS = ["value_b", "v_f0_b", "v_f1_b", "n_b", "delta", "d_f0", "d_f1", "scale",
                 "k_cal", "f_inflate", "se", "se_halfraw"]
# agg_by_cell.csv, in order; lpo_shape comes before p_ne_holm when the group has side-cars
AGG_COLUMNS = [
    "cell", "mode", "arm", "kind", "name", "extrap_family", "primary", "self_transfer",
    "n_pix", "n_tgt", "n_src", "mu_src_spread", "se_src", "sigma_base", "mu", "se",
    "se_within", "tau", "median", "p95", "chi2_red", "mu_fe", "se_fe", "margin", "tost",
    "tost_raw", "ci_lo", "ci_hi", "p_nonequiv", "has_xfer_lpo", "cheat_stat", "self_lpo_cost",
    "guard_verdict", "has_placebo", "has_self_in_rep", "placebo_minus_notwc1b_ps",
    "frac_over_margin", "R_median", "gate_fail", "k_cal", "f_inflate", "src_file_overlap",
    "n_pool_excluded", "lever_X_all", "p_ne_holm", "fail_holm"]
CALIBRATION_COLUMNS = ["cell", "run", "arm", "kind", "k_cal", "f_inflate", "f_source"]


class AggregationError(RuntimeError):
    """The cell tables cannot be aggregated: no cell table among the files, or no row with a
    baseline to difference against."""


class StructuralZeroError(AggregationError):
    """A k_cal pool of a one-board pair holds a row identical to the baseline."""


# ------------------------------------------------------------------------------ reading
def read_cells(files):
    """The cell tables in `files`, stacked; every column of CELL_COLUMNS present.

    A file without arm, kind, name and value is skipped with a line; a table with another
    column set is aggregated on the shared columns, the missing ones NaN. The side-car
    <table without .csv>.lpo_shape.json of a cell run with --lpo-fallback own is merged in
    memory as the column lpo_shape (one JSON text per tgt_triple, NaN for a triple it does not
    name); it is not a cell-table column."""
    frames = []
    for f in files:
        t = pd.read_csv(f)
        miss = [c for c in ("arm", "kind", "name", "value") if c not in t.columns]
        if miss:
            log("agg: skipping %s (not a cell table: missing %s)" % (os.path.basename(f), miss))
            continue
        if list(t.columns) != CELL_COLUMNS:
            log("agg: %s has another cell schema (%d columns, not the %d of CELL_COLUMNS); "
                "aggregating it on the shared columns only"
                % (os.path.basename(f), len(t.columns), len(CELL_COLUMNS)))
        shp = os.path.splitext(f)[0] + ".lpo_shape.json"
        if os.path.exists(shp) and "tgt_triple" in t.columns:
            try:
                with open(shp) as fh:
                    shape_j = json.load(fh)
                t = t.copy()
                t["lpo_shape"] = t["tgt_triple"].astype(str).map(
                    lambda k: json.dumps(shape_j[k]) if k in shape_j else np.nan)
            except Exception as e:
                log("agg: lpo_shape side-car %s unreadable: %r" % (shp, e))
        frames.append(t)
    if not frames:
        raise AggregationError("agg: none of the %d files is a cell table" % len(files))
    df = pd.concat(frames, ignore_index=True, sort=False)
    for c in CELL_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df


# ---------------------------------------------------------------------- row selection
def drop_below_nmin_eval(df, nmin_eval):
    """Drop every row of a (cell, run, combo, tgt_triple) one of whose board rows has
    n < nmin_eval (SPEC 5, the n_eval floor; 0 disables). Returns (frame, rows dropped)."""
    n_small = 0
    if nmin_eval:
        small = pd.to_numeric(df["n"], errors="coerce") < nmin_eval
        small &= (df["kind"] == "board")
        dead = set(map(tuple, df.loc[small, TRIPLE_KEY].drop_duplicates().to_numpy().tolist()))
        if dead:
            tup = list(map(tuple, df[TRIPLE_KEY].to_numpy().tolist()))
            keep = np.array([t not in dead for t in tup])
            n_small = int((~keep).sum())
            df = df[keep].reset_index(drop=True)
    return df, n_small


def censor_degenerate(df, keys=("cell", "run", "combo", "tgt_triple", "kind", "name")):
    """Drop a (triple, quantity) from every arm as soon as one arm is degenerate.

    Degenerate: a value that is not finite (a degenerate three-board solve is NaN,
    widths._sigma_from_widths) or neg_radicand == 1. Dropping only the arm that failed would
    keep exactly the transfers that happened to work (SPEC 5.6), so the censoring is
    symmetric across arms. Returns (frame, {arm: rows that triggered it}, rows dropped)."""
    d = df.copy()
    bad = ~np.isfinite(pd.to_numeric(d["value"], errors="coerce"))
    if "neg_radicand" in d.columns:
        bad |= (pd.to_numeric(d["neg_radicand"], errors="coerce") == 1)
    trig = d[bad].groupby("arm").size().to_dict()
    kk = [k for k in keys if k in d.columns]
    dead = set(map(tuple, d.loc[bad, kk].drop_duplicates().to_numpy().tolist()))
    if not dead:
        return d, trig, 0
    tup = list(map(tuple, d[kk].to_numpy().tolist()))
    keep = np.array([t not in dead for t in tup])
    return d[keep].reset_index(drop=True), trig, int((~keep).sum())


def is_one_board_arm(arm):
    """Is this arm's untransferred pair identical to the self_cf baseline on a one-board cell?

    True for placebo, notwc_1b and every arm whose name starts with "xfer" except xfer_lpo:
    the arms a one-board cell builds on self_cf with only board xb replaced (xfer, placebo,
    notwc_1b and the --arms ext variants xfer_clip, xfer_shift, xfer_rescale, xfer_calfix).
    False for xfer_lpo, whose base map is self_cf_lpo: its untouched pair differs from the
    baseline by the leave-pair-out cost and is a real measurement. False for placebo_self,
    self_in, self_cf_lpo, notwc and the other arms that move all three boards."""
    a_ = str(arm)
    return (a_ in ("placebo", "notwc_1b")
            or (a_.startswith("xfer") and a_ != "xfer_lpo"))


def xfer_board_by_cell(df):
    """{cell: its transferred board} for every one-board cell (one non-blank xfer_board).

    A cell naming more than one board is left out with a line: the identical-pair drop is then
    not applied to it."""
    xb_by_cell = {}
    for cell_, g_ in df.groupby("cell"):
        vals = sorted(set(str(v) for v in g_["xfer_board"].dropna().tolist()
                          if str(v) not in ("", "nan")))
        if len(vals) == 1:
            xb_by_cell[cell_] = vals[0]
        elif len(vals) > 1:
            log("agg: cell %s names more than one xfer_board %s; a cell is one protocol, so "
                "the identical-pair drop is not applied to it and its k_cal may be diluted; "
                "split it into one cell per board" % (cell_, vals))
    return xb_by_cell


def identical_pair_mask(df, xb_by_cell):
    """The pair rows of a one-board cell, in an is_one_board_arm arm, whose pair does not
    contain the cell's transferred board. The rule is keyed on the cell: notwc_1b is a self
    arm with a blank xfer_board, and its untransferred pair is identical to self_cf too."""
    return np.array([
        (kind_ == "pair") and is_one_board_arm(arm_)
        and (xb_by_cell.get(cell_) is not None)
        and (xb_by_cell.get(cell_) not in str(name_).split("-"))
        for cell_, kind_, name_, arm_ in zip(df["cell"], df["kind"], df["name"], df["arm"])])


def drop_identical_pairs(df, xb_by_cell, baseline="self_cf"):
    """Drop the identical_pair_mask rows (SPEC 3b) before anything is calibrated on them.

    On a one-board cell the pair without the transferred board has a residual array identical
    to self_cf's in those arms, so its delta and d_f0 - d_f1 are both exactly zero: such rows
    would pull k_cal (a median) down and stand as primary rows that cannot fail. The drop is a
    statement about the self_cf baseline only; against another baseline (notwc_1b, say) that
    pair's delta is a real measurement and stays. Returns (frame, rows dropped)."""
    n_id = 0
    if xb_by_cell and str(baseline) == "self_cf":
        drop_id = identical_pair_mask(df, xb_by_cell)
        n_id = int(drop_id.sum())
        if n_id:
            df = df[~drop_id].reset_index(drop=True)
            log("agg: dropped %d one-board pair rows whose pair does not contain the "
                "transferred board: identical to the %s baseline by construction (SPEC 3b). "
                "Cell -> xfer_board: %s" % (n_id, baseline, xb_by_cell))
    elif xb_by_cell:
        log("agg: the baseline is %s, not self_cf, so the identical-pair drop is not applied: "
            "against %s the untransferred pair of a one-board cell is a real measurement. "
            "Cell -> xfer_board: %s" % (baseline, baseline, xb_by_cell))
    return df, n_id


def deltas(df, baseline="self_cf"):
    """Every non-baseline arm joined to the baseline arm on DELTA_KEY (inner join, first
    baseline row of a key): value_b, v_f0_b, v_f1_b, n_b, delta = value - value_b, d_f0, d_f1,
    and scale = value_b / sqrt(n_b) (n_b floored at 1)."""
    base = df[df.arm == baseline].set_index(DELTA_KEY)
    base = base[~base.index.duplicated(keep="first")]
    out = []
    for arm, g in df.groupby("arm"):
        if arm == baseline:
            continue
        m = g.set_index(DELTA_KEY).join(base[["value", "v_f0", "v_f1", "n"]], rsuffix="_b",
                                         how="inner")
        m = m.reset_index()
        m["delta"] = m["value"] - m["value_b"]
        m["d_f0"] = m["v_f0"] - m["v_f0_b"]
        m["d_f1"] = m["v_f1"] - m["v_f1_b"]
        m["scale"] = m["value_b"] / np.sqrt(m["n_b"].clip(lower=1))
        out.append(m)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def same_pixel_mask(d):
    """Transfer rows whose source pixel is the target's own pixel in the same run and combo:
    a same-pixel refit filed as a transfer. They never enter a transfer aggregate or the k_cal
    pool. Across runs or angles (modes a, b, f) the same pixel is the point of the comparison,
    so the rule needs src_run == run and src_combo == combo."""
    sp = pd.to_numeric(d.get("same_pixel"), errors="coerce")
    same_run = (d["src_run"].astype(str) == d["run"].astype(str)) & \
               (d["src_combo"].astype(str) == d["combo"].astype(str))
    is_x = d["arm"].astype(str).str.startswith("xfer")
    return np.asarray(is_x & (sp == 1) & same_run)


def kcal_exclusion_mask(d, drop_sp):
    """The rows that set no k_cal: the same_pixel_mask rows, and the rows of a non-self arm
    with self_transfer == 1 (the source triple is one of the cell's targets, where the xfer
    arm is self_cf). A self arm has no source, so the flag says nothing about it. Returns
    (excluded, the self_transfer part)."""
    st_num = pd.to_numeric(d.get("self_transfer"), errors="coerce")
    is_self_arm = np.asarray(d["arm"].isin(SELF_ARMS))
    kc_st = np.asarray(st_num == 1) & ~is_self_arm
    return drop_sp | kc_st, kc_st


def kcal_by_group(d, excluded):
    """k_cal per (cell, run, arm, kind) (stats.kcal_from_cells) on the rows not excluded; a
    group made only of excluded rows falls back to all of its rows. Returns ({group: k_cal},
    the fallback groups, the number of rows left out)."""
    kc_ex_ser = pd.Series(excluded, index=d.index)
    kc, kc_fallback, kc_trim = {}, [], 0
    for k, g in d.groupby(KCAL_GROUP):
        gk = g[~kc_ex_ser.loc[g.index].to_numpy()]
        kc_trim += len(g) - len(gk)
        if not len(gk):
            gk, kc_fallback = g, kc_fallback + [k]
        kc[k] = kcal_from_cells(gk)
    return kc, kc_fallback, kc_trim


def structural_zero_census(d, excluded, xb_by_cell):
    """Rows identical to the baseline (delta == 0 and d_f0 - d_f1 == 0 exactly).

    Returns (in the pool, outside it, their (cell, arm, kind) groups in the pool, the hard
    ones). A hard group is a one-board pair group (is_one_board_arm, a cell of xb_by_cell):
    such a row deflates the group's median and must not be in the pool."""
    kc_ex_ser = pd.Series(excluded, index=d.index)
    z_all = ((pd.to_numeric(d["delta"], errors="coerce") == 0.0)
             & ((pd.to_numeric(d["d_f0"], errors="coerce")
                 - pd.to_numeric(d["d_f1"], errors="coerce")) == 0.0))
    z_ = z_all & ~kc_ex_ser
    n_z = int(z_.sum())
    n_z_ex = int((z_all & kc_ex_ser).sum())
    zg, hard = [], []
    if n_z:
        zg = sorted(set(zip(d.loc[z_, "cell"].astype(str), d.loc[z_, "arm"].astype(str),
                            d.loc[z_, "kind"].astype(str))))
        hard = [g_ for g_ in zg
                if g_[2] == "pair" and is_one_board_arm(g_[1]) and g_[0] in xb_by_cell]
    return n_z, n_z_ex, zg, hard


def retention_gain(df, baseline="self_cf"):
    """G per (cell, run, combo, tgt_triple, kind, name): notwc - baseline, or notwc_1b -
    baseline where the cell has notwc_1b (a one-board cell, where notwc would overstate what
    the correction buys by the two boards that keep their own fit)."""
    kk = QUANTITY_KEY
    nw = df[df.arm == "notwc"][kk + ["value"]].drop_duplicates(kk).rename(
        columns={"value": "v_notwc"})
    bs = df[df.arm == baseline][kk + ["value"]].drop_duplicates(kk).rename(
        columns={"value": "v_base"})
    g = nw.merge(bs, on=kk, how="inner")
    g["G"] = g["v_notwc"] - g["v_base"]
    n1 = df[df.arm == "notwc_1b"][kk + ["value"]].drop_duplicates(kk).rename(
        columns={"value": "v_notwc_1b"})
    if len(n1):
        g = g.merge(n1, on=kk, how="left")
        use1b = g["v_notwc_1b"].notna()
        g.loc[use1b, "G"] = g.loc[use1b, "v_notwc_1b"] - g.loc[use1b, "v_base"]
        log("agg: retention G taken from notwc_1b (the one-board uncorrected reference) on %d "
            "of %d (triple, quantity) rows" % (int(use1b.sum()), len(g)))
    return g


# ------------------------------------------------------------------------- decisions
def guard_verdict(xf, xl, sl, margin):
    """The leave-pair-out guard of a (cell, kind, name) from the mean deltas of xfer (xf),
    xfer_lpo (xl) and self_cf_lpo (sl): (has_xfer_lpo, cheat_stat, verdict).

    cheat_stat = (xf - xl) + sl (SPEC 5.1), NaN unless all three are finite. The verdict is
    "absent" without an xfer_lpo mean, "uninformative" when the self guard's cost |sl|
    exceeds the margin, "ok" otherwise."""
    has_lpo = int(np.isfinite(xl))
    cheat = (xf - xl) + sl if (np.isfinite(xf) and np.isfinite(xl) and np.isfinite(sl)) else np.nan
    if not has_lpo:
        verdict = "absent"
    elif np.isfinite(sl) and abs(sl) > margin:
        verdict = "uninformative"
    else:
        verdict = "ok"
    return has_lpo, cheat, verdict


def tost_label(arm, tost_raw, verdict, self_transfer):
    """The tost column: "NA-unguarded" for a transfer arm (xfer*) whose guard verdict is not
    "ok", unless self_transfer is 1; the TOST verdict otherwise. Never a PASS behind a guard
    that is not there."""
    if str(arm).startswith("xfer") and verdict != "ok" and int(self_transfer) == 0:
        return "NA-unguarded"
    return tost_raw


def primary_flag(kind, arm, name, xb):
    """1 for a pair row, 0 for a board row; 0 also for xfer_lpo on the pair of a one-board
    cell (transferred board xb) that does not contain xb, where xfer_lpo is identical to the
    self_cf_lpo control and cannot fail. The row is kept and reported either way."""
    prim = int(kind == "pair")
    if prim and str(arm) == "xfer_lpo" and xb is not None and xb not in str(name).split("-"):
        prim = 0
    return prim


def holm_board_fails(agg):
    """p_ne_holm and fail_holm: Holm's step-down over the board rows with tost == "FAIL" of
    each cell, on p_nonequiv (NaN and 0 elsewhere). PASS claims form an intersection-union test
    and take no correction. Sets the two columns in place and returns the frame."""
    agg["p_ne_holm"], agg["fail_holm"] = np.nan, 0
    for cell, gi in agg[(agg.kind == "board") & (agg.tost == "FAIL")].groupby("cell"):
        adj, rej = holm(gi["p_nonequiv"].to_numpy())
        agg.loc[gi.index, "p_ne_holm"] = adj
        agg.loc[gi.index, "fail_holm"] = rej.astype(int)
    return agg


# ----------------------------------------------------------------------------- f_cal
def f_inflate_for(explicit=None, campaign=None, precision="agg", read_root=None):
    """The error inflation f (SPEC 4 errors, 3) and where it came from: (f, f_source).

    In order: `explicit` (--f-inflate); the campaign's value of campaigns.F_CAL (--f-cal),
    at `precision` "agg", the value the study's aggregation ran with (four decimals), or
    "full", the bootstrap's value (the round-1 recomputation's); else the f_cal of
    <read root>/tables/f_cal.json; else 1.0, the uninflated half-split error, with a line."""
    if explicit is not None:
        return float(explicit), "explicit"
    if campaign is not None:
        f_agg, f_full, _ = campaigns.F_CAL[campaign]
        if precision == "agg":
            return f_agg, "f_cal %s, as aggregated" % campaign
        if precision == "full":
            return f_full, "f_cal %s, full precision" % campaign
        raise ValueError("f precision %r is neither 'agg' nor 'full'" % (precision,))
    p = os.path.join(read_root or campaigns.transfer_root(), "tables", "f_cal.json")
    if os.path.exists(p):
        try:
            with open(p) as fh:
                j = json.load(fh)
            return float(j["f_cal"]), "tables/f_cal.json (%s, ref=%s)" % (j.get("run"),
                                                                          j.get("ref"))
        except Exception:
            pass
    log("WARNING: no f given and no tables/f_cal.json in the read root; using f = 1.0 (the "
        "uninflated half-split error)")
    return 1.0, "default 1.0 (uncalibrated)"


# ------------------------------------------------------------------------- aggregate
def aggregate(df, baseline="self_cf", f_inflate=1.0, f_source="explicit", nmin_eval=600):
    """The aggregation of stacked cell tables (read_cells). Writes nothing.

    Returns dict(delta_by_triple, agg_by_cell, calibration, deltas): the three tables, and the
    per-triple frame with G, R and the self arms' self_transfer reset to 0 (what the Tier-0
    calibration joins). Raises AggregationError when no row has a baseline to difference
    against, and StructuralZeroError when a one-board pair's k_cal pool holds a structural
    zero."""
    df, n_small = drop_below_nmin_eval(df, nmin_eval)
    df, cens_by_arm, n_cens = censor_degenerate(df)
    if n_small or n_cens:
        log("agg: dropped %d rows below n_eval=%d and %d rows censored symmetrically for a "
            "degenerate solve (triggered by arm: %s)"
            % (n_small, nmin_eval, n_cens, cens_by_arm or "-"))

    xb_by_cell = xfer_board_by_cell(df)
    df, _ = drop_identical_pairs(df, xb_by_cell, baseline)

    d = deltas(df, baseline)
    if not len(d):
        raise AggregationError("agg: no baseline rows to difference against")
    log("agg: f_inflate = %.3f  [%s]" % (f_inflate, f_source))

    drop_sp = same_pixel_mask(d)
    n_sp = int(drop_sp.sum())
    if n_sp:
        log("agg: %d transfer rows excluded as same-pixel-on-that-board refits" % n_sp)

    kc_excl, kc_st = kcal_exclusion_mask(d, drop_sp)
    n_kx = int(kc_excl.sum())
    kc, kc_fallback, kc_trim = kcal_by_group(d, kc_excl)
    if n_kx:
        log("agg: the k_cal pool leaves out %d row(s) unfit to set a transfer error scale (%d "
            "same-pixel same-run refits, %d self-transfer rows on non-self arms), %d row(s) "
            "across %d k_cal group(s)"
            % (n_kx, n_sp, int((kc_st & ~drop_sp).sum()), kc_trim, len(kc)))
    if kc_fallback:
        log("agg: %d k_cal group(s) consist only of excluded rows and fall back to their own "
            "unfiltered scale: %s" % (len(kc_fallback), kc_fallback[:8]))

    n_z, n_z_ex, zg, hard = structural_zero_census(d, kc_excl, xb_by_cell)
    log("agg: k_cal structural-zero census: %d of %d delta rows in the pool have delta == 0 "
        "and d_f0 - d_f1 == 0 exactly, across %d (cell, run, arm, kind) k_cal groups; %d "
        "such row(s) sit outside the pool and set no scale"
        % (n_z, int(len(d) - n_kx), len(kc), n_z_ex))
    if n_z:
        log("agg: they sit in k_cal group(s) %s" % (zg[:12],))
    if hard:
        raise StructuralZeroError(
            "agg: %d one-board pair k_cal group(s) contain structurally-zero rows: %s. On a "
            "one-board cell such a row is identical to the baseline by construction (SPEC "
            "3b) and must not set the group's scale. Either the identical-pair drop missed "
            "them, or the cell has its own source triple among its targets without "
            "self_transfer = 1 (an older cell table): rerun that cell with --overwrite. "
            "Nothing has been written." % (len(hard), hard))

    d["k_cal"] = [kc.get((c, r, a_, k_), np.nan)
                  for c, r, a_, k_ in zip(d["cell"], d["run"], d["arm"], d["kind"])]
    d["f_inflate"] = f_inflate
    d["se"] = f_inflate * d["k_cal"] * d["scale"]
    d["se_halfraw"] = np.abs(d["d_f0"] - d["d_f1"]) / 2.0
    delta_by_triple = d

    kk = QUANTITY_KEY
    g = retention_gain(df, baseline)
    d = d.merge(g[kk + ["G"]], on=kk, how="left")
    d["R"] = 1.0 - d["delta"] / d["G"]

    # the reset comes after delta_by_triple, which keeps each row's own self_transfer
    d.loc[d["arm"].isin(SELF_ARMS), "self_transfer"] = 0
    d_agg = d[~drop_sp]

    # per (cell, kind, name): the guard's three means and the one-board placebo control
    guard, pl_vs_1b, shape_map = {}, {}, {}
    have_shape_col = "lpo_shape" in d_agg.columns
    for (cell, kind, name), g_ in d_agg.groupby(["cell", "kind", "name"]):
        def mu_(arm):
            x = g_[g_.arm == arm]["delta"].dropna()
            return float(x.mean()) if len(x) else np.nan
        guard[(cell, kind, name)] = (mu_("xfer"), mu_("xfer_lpo"), mu_("self_cf_lpo"))
        # SPEC 5.2 on a one-board cell: the placebo must land on notwc_1b (NaN without it)
        pl_vs_1b[(cell, kind, name)] = mu_("placebo") - mu_("notwc_1b")
        if have_shape_col:
            vals = sorted(set(v for v in g_["lpo_shape"].dropna().astype(str).tolist()
                              if v and v not in ("nan", "{}")))
            shape_map[(cell, kind, name)] = ";".join(vals)

    def _present(arm):
        h = d_agg[(d_agg.arm == arm) & d_agg["delta"].notna()]
        return set(map(tuple, h[["cell", "kind", "name"]].drop_duplicates().to_numpy().tolist()))
    have_pl, have_rep = _present("placebo"), _present("self_in_rep")

    rows, n_lpo_ctrl = [], 0
    for (cell, mode, arm, kind, name, st, ex), g_ in d_agg.groupby(
            ["cell", "mode", "arm", "kind", "name", "self_transfer",
             d_agg["extrap"].fillna(-1).astype(int)]):
        re_ = random_effects(g_["delta"], g_["se"])
        sre = source_random_effect(g_["delta"], g_["se"], g_["src_triple"].fillna(""))
        se_tot = float(np.sqrt(re_["se"] ** 2 + sre["se_src"] ** 2)) if np.isfinite(re_["se"]) else np.nan
        sig = float(g_["value_b"].median())
        marg = margin_for(sig)
        ts = tost(re_["mu"], se_tot, marg)
        # the guard is a property of the cell's (kind, name), the same on every arm's row
        xf, xl, sl = guard.get((cell, kind, name), (np.nan, np.nan, np.nan))
        has_lpo, cheat, verdict = guard_verdict(xf, xl, sl, marg)
        tost_v = tost_label(arm, ts["verdict"], verdict, st)
        n_pl = int((cell, kind, name) in have_pl)
        n_rep = int((cell, kind, name) in have_rep)
        prim = primary_flag(kind, arm, name, xb_by_cell.get(cell))
        if kind == "pair" and not prim:
            n_lpo_ctrl += 1
        row_d = dict(cell=cell, mode=mode, arm=arm, kind=kind, name=name,
                     extrap_family=("extrap" if ex == 1 else "interp" if ex == 0 else "na"),
                     primary=prim,
                     self_transfer=int(st), n_pix=re_["n"],
                     n_tgt=int(g_["tgt_triple"].nunique()), n_src=sre["n_src"],
                     mu_src_spread=sre["mu_src_spread"], se_src=sre["se_src"],
                     sigma_base=sig, mu=re_["mu"], se=se_tot, se_within=re_["se"],
                     tau=re_["tau"],
                     median=re_.get("median", np.nan), p95=re_.get("p95", np.nan),
                     chi2_red=re_["chi2"], mu_fe=re_["mu_fe"], se_fe=re_["se_fe"],
                     margin=marg, tost=tost_v, tost_raw=ts["verdict"],
                     ci_lo=ts["lo"], ci_hi=ts["hi"],
                     p_nonequiv=p_nonequiv(re_["mu"], se_tot, marg),
                     has_xfer_lpo=has_lpo, cheat_stat=cheat, self_lpo_cost=sl,
                     guard_verdict=verdict,
                     has_placebo=int(n_pl > 0), has_self_in_rep=int(n_rep > 0),
                     placebo_minus_notwc1b_ps=pl_vs_1b.get((cell, kind, name), np.nan),
                     frac_over_margin=float(np.mean(np.abs(g_["delta"]) > marg)),
                     R_median=float(np.nanmedian(g_["R"])) if "R" in g_ else np.nan,
                     gate_fail=float(np.mean(g_["accepted"] == 0)) if "accepted" in g_ else np.nan,
                     k_cal=float(np.nanmedian(g_["k_cal"])), f_inflate=f_inflate,
                     src_file_overlap=float(np.nanmedian(g_["src_file_overlap"]))
                     if "src_file_overlap" in g_ else np.nan,
                     n_pool_excluded=float(np.nanmedian(g_["n_pool_excluded"]))
                     if "n_pool_excluded" in g_ else np.nan,
                     lever_X_all=float(np.nanmedian(g_["lever_X_all"]))
                     if "lever_X_all" in g_ else np.nan)
        if have_shape_col:
            row_d["lpo_shape"] = shape_map.get((cell, kind, name), "")
        rows.append(row_d)
    if n_lpo_ctrl:
        log("agg: %d xfer_lpo pair row(s) marked non-primary: the pair without the transferred "
            "board, where xfer_lpo is the self_cf_lpo control by construction" % n_lpo_ctrl)
    agg = pd.DataFrame(rows).sort_values(["cell", "kind", "arm", "name"])
    agg = holm_board_fails(agg)

    cal = pd.DataFrame([dict(cell=c, run=r, arm=a_, kind=k_, k_cal=v, f_inflate=f_inflate,
                             f_source=f_source) for (c, r, a_, k_), v in kc.items()])
    return dict(delta_by_triple=delta_by_triple, agg_by_cell=agg, calibration=cal, deltas=d)


def tier0_calibration(d, t0):
    """The Tier-0 to Tier-2 calibration (SPEC 4, three tiers): the measured delta regressed
    on the Tier-0 P_exc of the same (run, combo, target, source, kind, name).

    `d` is aggregate()'s "deltas", `t0` a tier0 table. Returns (the one-row regression: n,
    slope, intercept, resid_rms, corr; the matched rows), or None with fewer than 4 matched
    rows."""
    j = d.merge(t0.rename(columns={"tgt": "tgt_triple", "src": "src_triple"}),
                on=["run", "combo", "tgt_triple", "src_triple", "kind", "name"],
                how="inner", suffixes=("", "_t0"))
    j = j[np.isfinite(j["delta"]) & np.isfinite(j.get("P_exc", np.nan))]
    if len(j) <= 3:
        log("tier0->tier2: only %d matched rows, no regression written" % len(j))
        return None
    x = j["P_exc"].to_numpy(float)
    y = j["delta"].to_numpy(float)
    A = np.column_stack([x, np.ones(len(x))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    reg = pd.DataFrame([dict(n=len(j), slope=coef[0], intercept=coef[1],
                             resid_rms=float(np.sqrt(np.mean(resid ** 2))),
                             corr=float(np.corrcoef(x, y)[0, 1]))])
    log("tier0->tier2: n=%d slope=%.3f resid rms=%.3f ps corr=%.2f"
        % (len(j), coef[0], float(np.sqrt(np.mean(resid ** 2))),
           float(np.corrcoef(x, y)[0, 1])))
    return reg, j[["cell", "run", "tgt_triple", "src_triple", "kind", "name", "delta", "P",
                   "P_null", "P_exc"]]


# ---------------------------------------------------------------------------- command
def _in_read_root(path, read_root):
    return path if os.path.isabs(path) else os.path.join(read_root, path)


def cmd_agg(a, root):
    """`agg`: aggregate the cell tables matched by --cells into <write root>/tables/ (or
    --out, a relative one in the write root): delta_by_triple.csv, agg_by_cell.csv, calibration.csv, and with --tier0
    tier0_calibration.csv and tier0_vs_tier2.csv."""
    read_root = campaigns.transfer_root()
    outdir = campaigns.output_path(a.out, root, "tables")
    files = sorted(sum([glob.glob(_in_read_root(x, read_root)) for x in a.cells], []))
    if not files:
        sys.exit("agg: no cell CSVs matched")
    if a.f_precision != "agg" and a.f_cal is None:
        sys.exit("agg: --f-precision applies to --f-cal only")
    f_inf, f_src = f_inflate_for(a.f_inflate, a.f_cal, a.f_precision, read_root)
    try:
        res = aggregate(read_cells(files), baseline=a.baseline, f_inflate=f_inf,
                        f_source=f_src, nmin_eval=a.nmin_eval)
    except AggregationError as e:
        sys.exit(str(e))
    os.makedirs(outdir, exist_ok=True)
    res["delta_by_triple"].to_csv(os.path.join(outdir, "delta_by_triple.csv"), index=False)
    agg = res["agg_by_cell"]
    agg.to_csv(os.path.join(outdir, "agg_by_cell.csv"), index=False)
    res["calibration"].to_csv(os.path.join(outdir, "calibration.csv"), index=False)

    if a.tier0:
        p = _in_read_root(a.tier0, read_root)
        if os.path.exists(p):
            t0 = tier0_calibration(res["deltas"], pd.read_parquet(p))
            if t0 is not None:
                t0[0].to_csv(os.path.join(outdir, "tier0_calibration.csv"), index=False)
                t0[1].to_csv(os.path.join(outdir, "tier0_vs_tier2.csv"), index=False)
        else:
            log("agg: --tier0 %s does not exist; no Tier-0 calibration written" % p)

    # SPEC 5.2 on a one-board cell, per cell: the placebo against notwc_1b
    for cell_, gg_ in agg[agg.kind == "board"].groupby("cell"):
        v_ = pd.to_numeric(gg_["placebo_minus_notwc1b_ps"], errors="coerce").dropna()
        if len(v_):
            log("agg: SPEC 5.2 one-board control, cell %s: mu(placebo) - mu(notwc_1b) = "
                "%+.3f ps (median over its %d board rows, range %+.3f .. %+.3f); a healthy "
                "placebo reproduces notwc_1b" % (cell_, float(np.nanmedian(v_)), len(v_),
                                                 float(np.nanmin(v_)), float(np.nanmax(v_))))

    show = ["cell", "arm", "name", "extrap_family", "n_pix", "n_tgt", "n_src", "sigma_base",
            "mu", "se", "tau", "margin", "tost", "guard_verdict", "R_median",
            "placebo_minus_notwc1b_ps"]
    print(agg[agg.kind == "board"][show].round(3).to_string(index=False))
    print("-> %s" % outdir)
