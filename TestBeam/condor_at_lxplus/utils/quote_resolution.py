#!/usr/bin/env python3
"""Quote per-board timing resolutions from step-13 tables - the same recipe for
any beam, campaign and number of boards (>= 3).

WHY A FIXED RECIPE
------------------
The pipeline's smallest unit that always exists is a TRACK: one pixel triple,
one 3-board solve, one bootstrap error (steps 8-13). A three-plane telescope
(e.g. a DESY angle scan) produces exactly one step-13 table; a four-board
telescope produces one table per leave-one-out combination. The recipe below
is defined at the track/pixel/board level so it applies to both, and treats
whatever extra tables exist as a CHECK (partner dependence, pair-width
consistency), never as the definition of the number.

RECIPE (per board, per step-13 table)
1. Clean: drop tracks with single_shot_failed / boot_failed, with fewer than
   --min-boot accepted resamples, and near-degenerate solves - a solved sigma
   below --margin-lo (0.35) or above --margin-hi (0.95) times the smallest of
   the pair widths it was solved from (a sigma of 16 ps next to 60-100 ps
   pair widths is the 3-board formula going almost imaginary, not a chip).
2. Pixel: 1/err^2-weighted mean of res_ over the tracks that use the pixel
   (what plot_resolution_table draws), its error 1/sqrt(sum 1/err^2), and the
   spread of res_ over those tracks.
3. Board value: robust Gaussian mean over the pixels (median/IQR-seeded,
   2.5-sigma clipped, unbinned ML - fit_bootstrap_results.perform_robust_
   unbinned_fit); its width = pixel-to-pixel spread (quoted as spread);
   statistical error = width / sqrt(N_pixels) (negligible in practice).
4. Systematics, always the same list:
   - partner-pixel: median over pixels of sqrt(spread^2 - mean err^2) - how
     much the answer for a pixel depends on WHICH partner pixels it was paired
     with; exists with three boards;
   - partner-board (only when the board appears in >= 2 tables): per pixel,
     half the range of that pixel's per-table values, median over the pixels
     present in >= 2 tables (see 5);
   - definition: FWHM-of-a-mixture core width, --def-syst (default 1 %: the
     residual softness of the converged fit); the core-vs-RMS convention itself
     is stated, not folded in;
   - chip halves: the error-weighted left - right difference, reported;
   - illumination (sub-pixel): tracks whose partner pixels sit at the modal
     (nominal) offset sample one sub-region of the pixel, tracks with an
     off-nominal partner sample the complementary side sub-region - with a
     fractional inter-plane alignment of a few tenths of a pixel (typical),
     these sub-regions are a few hundred um wide, i.e. sizeable fractions of a
     1.3 mm pixel, not charge-sharing edges, and the off-nominal class often
     carries most of the events. The same pixel's sigma differs between the
     two classes (a coarse sub-pixel position dependence, the only one
     available without a tracker). The quoted value is the illumination-
     weighted average of both (the operating resolution under this beam
     geometry and alignment); the tool also reports the 'central' value
     (nominal-offset tracks only), the median off-nominal - central difference
     over pixels, and the event share of the off-nominal tracks (from the
     step-11 nevt CSVs when found next to the tables, else 1/err^2 as a proxy
     for N). The "modal (nominal) offset" itself is EVENT-WEIGHTED (same
     nevt/1-err^2 weights, per partner board, per table) - a plain per-track
     vote flips to a side offset as the water-fill of step 7 adds mostly
     side-offset tracks with selection depth, even though the nominal offset
     keeps the most events (notes/pipeline-pr-list.md 2026-09-07). The modal
     and runner-up weight shares are reported per partner board (warn below
     1.5x), along with the fraction of a board's pixels whose own most-weighted
     partner offset agrees with the table mode (a rigid-translation check), and
     an optional --alignment comparison against the step-6 translation.
5. Combination across tables is PIXEL-CENTRIC, because with an angled beam
   the tables of a multi-board telescope cover different pixel sets (planes
   0-1-2 and 2-3-4 fire on different rows; some combos have no overlap at
   all): all cleaned tracks of all tables are pooled and step 2 is applied to
   the pool (a pixel present in one table only just gets that table's
   tracks), the value is step 3 on that COMBINED map, and the partner-board
   systematic is computed per pixel - half the range of the per-table
   per-pixel values, median over the pixels present in >= 2 tables (so
   different pixel coverage cannot fake a partner shift; if no pixel is in
   two tables it falls back to the partner-pixel term). Coverage (pixels per
   table, in the union, in >= 2 tables) is reported.
6. Quoted number: the combined value +/- stat, +/- partner, +/- definition;
   the map, the spread and the halves alongside. With >= 4 boards, the
   least-squares solve of all pair widths is printed as a consistency check.
7. TWO EVENT THRESHOLDS, always both. The tracks entering the quote are those
   step 12 ran, i.e. >= --nevt-standard (100) events; the same number is also
   recomputed on tracks with >= --nevt-conservative (300), with the pixel count
   of each. 100 events is generous - the converged mixture fit is still ~5 %
   low there (the GMM toy study) - so a run whose two numbers disagree is a run
   whose value rests on low-occupancy pixels, and that is now visible in the
   quote instead of having to be re-derived. Both are stored; the headline
   number is unchanged (it is the standard one).
8. PAIRING COVARIANCE (four boards, optional input). The 3-board algebra
   assumes sigma_ab^2 = sigma_a^2 + sigma_b^2. On 4-board coincidences that
   additivity is rejected (p = 8e-11 / 1e-30 on the two runs measured), and the
   one covariance model the six pair widths CAN constrain - a single g shared by
   the two pairs of one PAIRING of the four boards, 5 parameters and 1 d.o.f.
   left to test it - fits them (chi2/dof 0.59 and 0.35) with g = +60 ps^2 at
   1.5e15 and +619 ps^2 on F1 at 3.5e15. Under that model every 3-board solve is
   v-g, v-g, v+g over the three combos, so the quoted value is low by g/3
   (+0.2 ps at 1.5e15, +2 ps on F1 3.5e15). See notes/fourboard-lsq-test.md.
   This block is computed when a four-board per-quadruple width table is
   supplied (--fourboard, or fourboard_pairs_*.csv next to the input tables;
   utils/fourboard_pairing.py builds it) AND the table clears
   --pairing-min-events (500) events per quadruple on --pairing-min-quads (30)
   quadruples. It is stored ALONGSIDE the standard solve, never instead of it;
   when it cannot be computed the block records why.

USAGE
  python utils/quote_resolution.py -i final_<run>/resolution_table_*.csv -o <outdir> [--label <run>]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from glob import glob
from itertools import combinations

import numpy as np
import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))
sys.path.insert(0, HERE)
import fit_bootstrap_results as fbr          # noqa: E402  (step 13's own robust Gaussian)
from telescope_diagnostics import GRID, pixel_map, set_output_options, add_output_arguments  # noqa: E402
from track_diagnostics import finish, rc  # noqa: E402
from path_finder import PIXEL_PITCH  # noqa: E402  (same convention as apply_geometric_transformation_matrix: col <-> x, row <-> y; used only by --alignment)

ROLES = ("dut", "ref", "trig", "extra")


def combo_of(path):
    m = re.search(r"resolution_table_(.+?)(?:_[^_]*)?\.csv$", os.path.basename(path))
    return m.group(1) if m else os.path.splitext(os.path.basename(path))[0]


def clean(df, roles, a):
    """Apply the cleaning rules; return the cleaned frame and a dict of counts."""
    n0 = len(df)
    keep = pd.Series(True, index=df.index)
    if "boot_failed" in df:
        keep &= df["boot_failed"] == 0
    if "n_boot" in df:
        keep &= df["n_boot"] >= a.min_boot
    for r in roles:
        col = "single_shot_failed_%s" % r
        if col in df:
            keep &= df[col] == 0
        keep &= df["res_%s" % r] > 0
    n_flag = int((~keep).sum())
    # near-degenerate: solved sigma vs the smallest pair width it came from
    pair_cols = [c for c in df.columns if c.startswith("res_pair_")]
    n_deg = 0
    if pair_cols:
        for r in roles:
            mine = [c for c in pair_cols if r in c[len("res_pair_"):].split("-")]
            if not mine:
                continue
            smin = df[mine].min(axis=1)
            ratio = df["res_%s" % r] / smin
            bad = keep & ((ratio < a.margin_lo) | (ratio > a.margin_hi))
            n_deg += int(bad.sum())
            keep &= ~bad
    return df.loc[keep].copy(), dict(n_tracks=n0, dropped_flagged=n_flag, dropped_degenerate=n_deg, kept=int(keep.sum()))


def per_pixel(df, r):
    g = df.groupby(["row_%s" % r, "col_%s" % r])
    w = lambda x: 1.0 / x["err_%s" % r] ** 2
    P = pd.DataFrame({
        "mean": g.apply(lambda x: np.average(x["res_%s" % r], weights=w(x))),
        "err": g.apply(lambda x: 1.0 / np.sqrt(w(x).sum())),
        "spread": g["res_%s" % r].std(ddof=1),
        "mean_err": g["err_%s" % r].mean(),
        "n": g.size(),
    })
    if "_table" in df.columns:
        P["n_tables"] = g["_table"].nunique()
    return P


def board_summary(P, a):
    vals = P["mean"].to_numpy(float)
    st = fbr.perform_robust_unbinned_fit(pd.Series(vals), 2.5)
    mu, spread = float(st["mu"]), float(st["sigma"])
    stat = spread / np.sqrt(len(vals))
    # partner-pixel systematic: excess of the track-to-track spread over the bootstrap error
    m = P["n"] >= 2
    exc = np.sqrt(np.clip(P.loc[m, "spread"] ** 2 - P.loc[m, "mean_err"] ** 2, 0, None))
    partner_pixel = float(np.median(exc)) if m.any() else np.nan
    # chip halves
    cols = np.array([k[1] for k in P.index])
    right, left = cols < 8, cols >= 8
    wts = 1.0 / P["err"].to_numpy(float) ** 2

    def wm(mask):
        return (np.sum(wts[mask] * vals[mask]) / np.sum(wts[mask]), 1.0 / np.sqrt(np.sum(wts[mask]))) if mask.any() else (np.nan, np.nan)
    ml, el = wm(left)
    mr, er = wm(right)
    return dict(value=mu, pixel_spread=spread, stat=float(stat), n_pixels=int(len(vals)),
                partner_pixel=partner_pixel, half_left_minus_right=float(ml - mr), half_err=float(np.hypot(el, er)),
                fit_valid=int(st["valid"]))


def offset_weight(df, r, nevt=None):
    """Weight for the offset vote (central_mask) and the illumination split: nevt (step-11 event count)
    when given, else 1/err_r^2 - the same fallback illumination() has always used as a proxy for N."""
    return nevt if nevt is not None else 1.0 / df["err_%s" % r] ** 2


def central_mask(df, r, roles, weights=None):
    """True for tracks whose partner pixels all sit at the EVENT-WEIGHTED modal (nominal) offset from
    board r's pixel: the (dr, dc) offset carrying the largest summed weight (nevt when available, else
    offset_weight()'s 1/err^2 fallback - the same weights illumination() uses), not the plain track count
    a bare Series.mode() gives. Each pixel has one partner pixel at the nominal offset but several at side
    offsets, so as the water-fill of step 7 adds mostly side-offset tracks with selection depth, an
    unweighted vote can flip to a side offset even though the nominal offset still carries the most events
    (notes/pipeline-pr-list.md 2026-09-07).

    Returns (central_bool_array, diag): diag maps each partner role name to the winning offset and the
    modal/runner-up shares of total weight - the vote margin, for the caller to report/warn on.
    """
    w = weights if weights is not None else offset_weight(df, r)
    central = np.ones(len(df), bool)
    diag = {}
    for o in [x for x in roles if x != r]:
        dr = df["row_%s" % o] - df["row_%s" % r]
        dc = df["col_%s" % o] - df["col_%s" % r]
        # kind="stable" so an exact tie in summed weight resolves the same way on every run, not by
        # whatever order the hash-based default quicksort happens to leave equal keys in.
        wsum = w.groupby([dr, dc]).sum().sort_values(ascending=False, kind="stable")
        mode_dr, mode_dc = wsum.index[0]
        total = float(wsum.sum())
        modal_share = float(wsum.iloc[0] / total) if total > 0 else None
        runnerup_share = float(wsum.iloc[1] / total) if len(wsum) > 1 and total > 0 else 0.0
        diag[o] = dict(offset=[int(mode_dr), int(mode_dc)], modal_share=modal_share, runnerup_share=runnerup_share,
                       ratio=(modal_share / runnerup_share) if (modal_share is not None and runnerup_share > 0) else None)
        central &= ((dr == mode_dr) & (dc == mode_dc)).to_numpy()
    return central, diag


def pixel_mode_match(df, r, roles, weights, diag):
    """Fraction of board r's pixels whose OWN most-weighted partner offset agrees with the table-level
    mode carried in `diag` (per partner board) - a rigid-translation check: a genuine translation moves
    every pixel's own mode together with the table mode, whereas a rotation would leave many pixels
    disagreeing with it even though the table-level vote still has a clear winner."""
    out = {}
    for o in [x for x in roles if x != r]:
        dr = df["row_%s" % o] - df["row_%s" % r]
        dc = df["col_%s" % o] - df["col_%s" % r]
        key = pd.DataFrame({"pr": df["row_%s" % r].to_numpy(), "pc": df["col_%s" % r].to_numpy(),
                            "dr": dr.to_numpy(), "dc": dc.to_numpy(), "w": np.asarray(weights, dtype=float)})
        g = key.groupby(["pr", "pc", "dr", "dc"], as_index=False)["w"].sum()
        top = g.loc[g.groupby(["pr", "pc"])["w"].idxmax()]
        mode_dr, mode_dc = diag[o]["offset"]
        match = (top["dr"] == mode_dr) & (top["dc"] == mode_dc)
        out[o] = dict(fraction=float(match.mean()) if len(top) else float("nan"), n_pixels=int(len(top)))
    return out


def illumination(df, r, roles, nevt=None):
    """Central (all partners at the event-weighted modal offset) vs off-nominal tracks for board r."""
    w = offset_weight(df, r, nevt)
    central, central_diag = central_mask(df, r, roles, weights=w)
    pmm = pixel_mode_match(df, r, roles, w, central_diag)
    c, e = df[central], df[~central]
    if len(c) < 5:
        return dict(value_central=np.nan, edge_minus_central=np.nan, edge_event_share=float(w[~central].sum() / w.sum()),
                    n_central=int(len(c)), central_diag=central_diag, pixel_mode_match=pmm)
    pc = c.groupby(["row_%s" % r, "col_%s" % r])["res_%s" % r].mean()
    stc = fbr.perform_robust_unbinned_fit(pd.Series(pc.values), 2.5)
    pe = e.groupby(["row_%s" % r, "col_%s" % r]).apply(lambda x: np.average(x["res_%s" % r], weights=w.loc[x.index]))
    j = pd.concat([pc.rename("c"), pe.rename("e")], axis=1).dropna()
    return dict(value_central=float(stc["mu"]), edge_minus_central=float((j.e - j.c).median()) if len(j) else np.nan,
                edge_event_share=float(w[~central].sum() / w.sum()), n_central=int(len(c)),
                central_diag=central_diag, pixel_mode_match=pmm)


# --------------------------------------------------------------------------
# optional step-6 alignment cross-check (--alignment)
# --------------------------------------------------------------------------

def load_alignment(path):
    """Load a step-6 --find_alignment yaml (core/path_finder.py) and return per-board-id translations
    (mm; x <-> col, y <-> row, see apply_geometric_transformation_matrix) plus a role->id map inferred by
    parsing the combo-label keys (e.g. 'trig0-ref1-dut2'), since the yaml keys boards by id and
    quote_resolution otherwise only knows role names.

    CONSERVATIVE by construction, because the format leaves real ambiguity for a role-space consumer:
    (1) role<->id is inferred from combo-label text, not a dedicated mapping - a board whose id never
    appears in a combo label is left unmapped and any comparison involving it is skipped with a reason;
    (2) a board absent from the yaml's 'applied' block (the trigger board, which --find_alignment never
    re-estimates, or a board --find_alignment could not align at all) is assumed to sit at its zero
    --config translation - alignment_check() flags this per board (except the trigger, for which it is
    expected) rather than comparing against it silently; (3) rotation is handled separately - see
    alignment_check()'s docstring for what is and is not corrected for. Every early exit / per-board gap
    is reported via a `reason` string rather than silently comparing against a wrong assumption.

    Rejects, rather than silently mis-mapping, the OTHER alignment yaml layout in this repo:
    telescope_diagnostics.py's section_alignment() documents (and reads) a
    {run: {legacy_per_combo: {combo: {board: ...}}, global_relative: {pinned_board, boards: {...}}}}
    layout that the current core/path_finder.py does not write, but that a yaml from a different
    pipeline version could. Its combo-label keys ('legacy_per_combo', 'global_relative') do not match
    the role+id token regex below, which would otherwise silently yield an empty role_to_id.
    """
    try:
        with open(path) as f:
            y = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        return dict(present=False, reason="could not read/parse %s: %s" % (path, e))
    if not isinstance(y, dict) or len(y) != 1:
        return dict(present=False, reason="expected exactly one top-level run name in %s, found %d" % (path, len(y or {})))
    run_name, block = next(iter(y.items()))
    block = block or {}
    if "legacy_per_combo" in block or "global_relative" in block:
        return dict(present=False, reason="%s uses the {legacy_per_combo, global_relative} alignment layout "
                                          "(see utils/telescope_diagnostics.py section_alignment()), which "
                                          "quote_resolution's role<->id combo-label inference does not support - "
                                          "read it with telescope_diagnostics.py instead, or extend load_alignment()"
                                          % path)
    role_to_id = {}
    for combo_label in block:
        if combo_label == "applied":
            continue
        for tok in str(combo_label).split("-"):
            m = re.match(r"^([A-Za-z]+)(\d+)$", tok)
            if m:
                role_to_id.setdefault(m.group(1), int(m.group(2)))
    applied = block.get("applied", {}) or {}
    translations = {bid: (v.get("transformation", {}) or {}).get("translation", {}) or {} for bid, v in applied.items()}
    if not translations:
        # Observed in practice: some alignment yamls carry no 'applied' summary at all (every board's
        # estimate stayed keyed under its combo only). Conservative fallback: take each board's first
        # appearance across the combo blocks as this dict iterates them. NOTE that is NOT reliably
        # path_finder.py's own measurement order: path_finder writes with ruamel's round-trip dumper
        # (order-preserving) today, but a generic yaml.dump (default sort_keys=True) would alphabetize
        # the combo labels instead, and either way this is an approximation of 'applied', not a
        # guarantee of matching what path_finder itself would have picked.
        for combo_label, combo_vals in block.items():
            if combo_label == "applied":
                continue
            for bid, v in (combo_vals or {}).items():
                if bid not in translations:
                    translations[bid] = ((v or {}).get("transformation", {}) or {}).get("translation", {}) or {}
    return dict(present=True, run_name=run_name, role_to_id=role_to_id, translations=translations, source=path)


def load_rotation(config_path, run_name, role_to_id):
    """Load each role's rotation.y (degrees) from a board-config yaml (the -c/--config path_finder.py
    itself takes), so alignment_check() can correct the column axis for a real telescope tilt instead of
    ignoring it. Returns (dict role -> degrees, reason); reason is None on success. Checked against
    board_configs_yaml/DESY_TB_2026May_CE.yaml: run23 has rotation.y = -20 deg on all four boards (a
    telescope tilt, not per-board), run26 has no transformation block at all (all zero) - consistent
    with the row-only default being the common case worth calling out, not a rare edge."""
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        return None, "could not read/parse --config %s: %s" % (config_path, e)
    if not isinstance(cfg, dict) or run_name not in cfg:
        return None, "run '%s' not found in --config %s" % (run_name, config_path)
    run_block = cfg[run_name] or {}
    out = {}
    for role, bid in role_to_id.items():
        board_conf = run_block.get(bid, {}) or {}
        rot = (board_conf.get("transformation", {}) or {}).get("rotation", {}) or {}
        out[role] = float(rot.get("y", 0.0) or 0.0)
    return out, None


def alignment_check(align, r, roles, rot_y=None):
    """Compare the step-6 translation-implied pixel offset against the weighted modal offset, per
    partner board. `align` is load_alignment()'s return; every non-comparable case carries a `reason`
    rather than a silent skip.

    ROW axis (dr): unaffected by a y-axis rotation - apply_geometric_transformation_matrix gives
    y_global = (row - OFFSET)*PITCH + tra.y for any rotation.y, so it is always predicted and compared.

    COLUMN axis (dc): x_global = (col - OFFSET)*PITCH*cos(ry) + tra.x, so a nonzero rotation.y rescales
    it. Without `rot_y` (no --config given) the column is NOT predicted at all - a "small rotation"
    assumption would be wrong on most DESY-May runs (rotation.y from -10 to -60 deg in
    board_configs_yaml/DESY_TB_2026May_CE.yaml; run26 happens to be 0, run23 is -20 on every board).
    With `rot_y` (role -> degrees, from load_rotation()), pred_dc = (tra_r.x - tra_o.x) /
    (PITCH * cos(ry_avg)); ry_avg is exact when both boards share one rotation.y (true on every DESY-May
    run checked - the whole telescope tilts together, not per board) and an approximation, flagged via
    `reason`, when they differ by more than 1 deg.

    Any board mapped to a role but absent from `align["translations"]` (no recorded translation) is
    assumed to sit at zero - expected for the trigger board (--find_alignment never re-estimates it) but
    a red flag for any other board, and flagged as `zero_assumed_boards` / `zero_assumed_reason` rather
    than silently compared against a possibly-wrong zero.
    """
    if align is None or not align.get("present"):
        return dict(present=False, reason=(align or {}).get("reason", "no --alignment given"))
    r2i = align["role_to_id"]
    trig_id = r2i.get("trig")
    if r not in r2i:
        return dict(present=False, reason="role '%s' not found in the alignment yaml's combo labels" % r)
    id_r = r2i[r]
    tr_r_raw = align["translations"].get(id_r)
    zero_r = tr_r_raw is None and id_r != trig_id
    tr_r = tr_r_raw or {}
    out = dict(present=True)
    for o in [x for x in roles if x != r]:
        if o not in r2i:
            out[o] = dict(present=False, reason="role '%s' not found in the alignment yaml's combo labels" % o)
            continue
        id_o = r2i[o]
        tr_o_raw = align["translations"].get(id_o)
        zero_o = tr_o_raw is None and id_o != trig_id
        tr_o = tr_o_raw or {}
        pred_dr = (float(tr_r.get("y", 0.0)) - float(tr_o.get("y", 0.0))) / PIXEL_PITCH
        if rot_y is not None and r in rot_y and o in rot_y:
            ry_r, ry_o = rot_y[r], rot_y[o]
            ry_avg = 0.5 * (ry_r + ry_o)
            cos_avg = float(np.cos(np.deg2rad(ry_avg)))
            pred_dc = ((float(tr_r.get("x", 0.0)) - float(tr_o.get("x", 0.0))) / (PIXEL_PITCH * cos_avg)
                      if cos_avg != 0 else float("nan"))
            entry = dict(present=True, predicted_offset=[round(pred_dr, 2), round(pred_dc, 2)], row_only=False)
            if abs(ry_r - ry_o) > 1.0:
                entry["reason"] = ("boards have different rotation.y (%.1f vs %.1f deg); column prediction "
                                   "uses their average and is approximate" % (ry_r, ry_o))
        else:
            entry = dict(present=True, predicted_offset=[round(pred_dr, 2), None], row_only=True,
                        reason="column not compared: rotation.y unknown without --config, and it is not "
                               "small on most DESY-May runs (see docstring)")
        zero_assumed = [x for x, z in ((r, zero_r), (o, zero_o)) if z]
        if zero_assumed:
            entry["zero_assumed_boards"] = zero_assumed
            entry["zero_assumed_reason"] = ("board(s) %s have no recorded translation in the alignment yaml "
                                            "and are not the trigger board - assumed zero, which may be wrong"
                                            % ", ".join(zero_assumed))
        out[o] = entry
    return out


# --------------------------------------------------------------------------
# dual event threshold
# --------------------------------------------------------------------------

def nevt_quotes(pooled, r, a, has_nevt):
    """The board value at two track-occupancy thresholds, with the pixel counts.

    Returns a dict that always carries `present`; when the step-11 nevt CSV was
    not found next to the tables there is no per-track event count to threshold
    on, and the block says so rather than silently quoting one number twice.
    """
    if not has_nevt or "nevt" not in pooled.columns:
        return dict(present=False, reason="no step-11 nevt_<combo>_*.csv next to the input tables, "
                                          "so no per-track event count to threshold on")
    out = dict(present=True, weights="nevt")
    for name, thr in (("standard", a.nevt_standard), ("conservative", a.nevt_conservative)):
        sub = pooled[pooled["nevt"] >= thr]
        if len(sub) < 5:
            out[name] = dict(nevt_min=int(thr), present=False,
                             reason="only %d track(s) with >= %d events" % (len(sub), thr))
            continue
        P = per_pixel(sub, r)
        st = board_summary(P, a)
        out[name] = dict(nevt_min=int(thr), present=True, value=st["value"], stat=st["stat"],
                         pixel_spread=st["pixel_spread"], n_pixels=st["n_pixels"], n_tracks=int(len(sub)))
    if out.get("standard", {}).get("present") and out.get("conservative", {}).get("present"):
        out["conservative_minus_standard"] = out["conservative"]["value"] - out["standard"]["value"]
    return out


# --------------------------------------------------------------------------
# pairing-covariance fit on 4-board coincidences (notes/fourboard-lsq-test.md)
# --------------------------------------------------------------------------

def _pair_key(a, b):
    return "-".join(sorted((a, b)))


def _fourboard_design(boards, pair_keys):
    X = np.zeros((6, 4))
    for k, key in enumerate(pair_keys):
        for b in key.split("-"):
            X[k, boards.index(b)] = 1.0
    return X


def _calibrate(d, pair_keys):
    """Per-run error scale from the two disjoint halves of each quadruple.

    Under Var(w) ~ 1/n the half-sample difference has sd = 2 * sd(w_n), so
    kcal = median(|w_h0 - w_h1| * sqrt(n) / w) / (2 * 0.6745) converts a
    quadruple's n into the error on its pair widths.  Straight port of
    fb_sum.calib.
    """
    u = []
    for k in pair_keys:
        if "h0_w_%s" % k not in d.columns or "h1_w_%s" % k not in d.columns:
            continue
        m = d[["w_%s" % k, "h0_w_%s" % k, "h1_w_%s" % k, "n"]].dropna()
        if not len(m):
            continue
        u.append((np.abs(m["h0_w_%s" % k] - m["h1_w_%s" % k]) * np.sqrt(m["n"]) / m["w_%s" % k]).to_numpy())
    if not u:
        return None
    return float(np.median(np.concatenate(u)) / (2 * 0.6745))


def _lsq4(y, se, X):
    W = np.diag(1.0 / np.asarray(se, float) ** 2)
    cov = np.linalg.inv(X.T @ W @ X)
    v = cov @ (X.T @ W @ np.asarray(y, float))
    r = np.asarray(y, float) - X @ v
    return v, cov, float(r @ W @ r)


def _pairing_fit(y, se, X, pair_keys, ka, kb):
    """5-parameter model y_ij = v_i + v_j - 2g for the two pairs of ONE pairing.

    A g common to all six pairs is exactly degenerate with v_i -> v_i + g (the
    design has rank 4 of 5), and so is one covariance per pair of a pairing;
    one g per PAIRING is the largest covariance model six widths can constrain -
    rank 5, leaving 1 d.o.f. to test it.  Port of fb_sum.pairing_fit.
    """
    X5 = np.zeros((6, 5))
    X5[:, :4] = X
    X5[pair_keys.index(ka), 4] = -2.0
    X5[pair_keys.index(kb), 4] = -2.0
    W = np.diag(1.0 / np.asarray(se, float) ** 2)
    cov = np.linalg.inv(X5.T @ W @ X5)
    th = cov @ (X5.T @ W @ np.asarray(y, float))
    r = np.asarray(y, float) - X5 @ th
    return th[:4], float(th[4]), float(np.sqrt(cov[4, 4])), float(r @ W @ r)


def pairing_covariance(path, a):
    """Fit the pairing covariance from a four-board per-quadruple width table.

    Input schema (utils/fourboard_pairing.py, and the study's own fit_*.csv):
    one row per pixel quadruple with `n`, `w_<a>-<b>` for the six pair widths in
    ps, and `h0_w_*`/`h1_w_*` for the two disjoint halves (the error source).
    Returns the JSON block; every early exit carries a `reason`.
    """
    from scipy.stats import chi2 as chi2dist
    if path is None:
        return dict(present=False, reason="no four-board per-quadruple width table given "
                                          "(--fourboard, or fourboard_pairs_*.csv next to the tables); "
                                          "build one with utils/fourboard_pairing.py")
    d = pd.read_csv(path)
    if "status" in d.columns:
        d = d[d["status"] == "ok"]
    wcols = [c for c in d.columns if c.startswith("w_") and "-" in c]
    boards = sorted({b for c in wcols for b in c[2:].split("-")})
    if len(boards) != 4 or len(wcols) != 6:
        return dict(present=False, source=path,
                    reason="table has %d board(s) and %d pair-width column(s); the pairing model needs "
                           "exactly 4 boards and their 6 pair widths" % (len(boards), len(wcols)))
    pair_keys = [_pair_key(x, y) for x, y in combinations(boards, 2)]
    n_in = len(d)
    d = d[d["n"] >= a.pairing_min_events].dropna(subset=["w_%s" % k for k in pair_keys])
    if len(d) < a.pairing_min_quads:
        return dict(present=False, source=path,
                    cuts=dict(min_events_per_quadruple=int(a.pairing_min_events),
                              min_quadruples=int(a.pairing_min_quads)),
                    n_quadruples_input=int(n_in), n_quadruples_used=int(len(d)),
                    reason="only %d quadruple(s) have >= %d four-board events (need %d); g is measurable "
                           "below that but loose (+-35 %%), so quote it from a neighbouring run of the same "
                           "configuration instead" % (len(d), a.pairing_min_events, a.pairing_min_quads))
    kcal = _calibrate(d, pair_keys)
    if kcal is None:
        return dict(present=False, source=path,
                    reason="table carries no h0_w_/h1_w_ half-sample columns, so the pair-width errors "
                           "cannot be calibrated and no chi2 would mean anything")
    X = _fourboard_design(boards, pair_keys)
    pairings = [(_pair_key(boards[0], boards[1]), _pair_key(boards[2], boards[3])),
                (_pair_key(boards[0], boards[2]), _pair_key(boards[1], boards[3])),
                (_pair_key(boards[0], boards[3]), _pair_key(boards[1], boards[2]))]
    rng = np.random.default_rng(11)
    Y = np.array([[row["w_%s" % k] ** 2 for k in pair_keys] for _, row in d.iterrows()])
    SE = np.array([[2 * row["w_%s" % k] * (kcal * row["w_%s" % k] / np.sqrt(row["n"])) for k in pair_keys]
                   for _, row in d.iterrows()])
    # 4-parameter closure (the additivity the 3-board algebra assumes)
    c2_add, vv = [], []
    for i in range(len(Y)):
        v, _, c2 = _lsq4(Y[i], SE[i], X)
        c2_add.append(c2)
        vv.append(v)
    vv = np.array(vv)
    c2_add = np.array(c2_add)
    out_pairings = []
    for ka, kb in pairings:
        gs, c2s = [], []
        for i in range(len(Y)):
            _, g, _, c2 = _pairing_fit(Y[i], SE[i], X, pair_keys, ka, kb)
            gs.append(g)
            c2s.append(c2)
        gs, c2s = np.array(gs), np.array(c2s)
        bs = np.array([np.median(rng.choice(gs, len(gs))) for _ in range(400)])
        out_pairings.append(dict(pairing="%s | %s" % (ka, kb), g_ps2=float(np.median(gs)),
                                 g_err_ps2=float(bs.std(ddof=1)), chi2_per_dof=float(np.median(c2s)),
                                 pooled_chi2=float(c2s.sum()), pooled_dof=int(len(c2s)),
                                 pooled_p=float(chi2dist.sf(c2s.sum(), len(c2s)))))
    best = min(out_pairings, key=lambda x: x["pooled_chi2"])
    g = best["g_ps2"]
    sig_lsq = {b: float(np.sqrt(v)) if v > 0 else float("nan")
               for b, v in zip(boards, np.median(vv, axis=0))}
    sig_cor = {b: float(np.sqrt(v + g / 3.0)) if (v + g / 3.0) > 0 else float("nan")
               for b, v in zip(boards, np.median(vv, axis=0))}
    return dict(present=True, source=path,
                cuts=dict(min_events_per_quadruple=int(a.pairing_min_events),
                          min_quadruples=int(a.pairing_min_quads)),
                n_quadruples_input=int(n_in), n_quadruples_used=int(len(d)),
                median_events_per_quadruple=float(d["n"].median()), error_scale_kcal=kcal,
                additivity=dict(chi2_per_dof=float(np.median(c2_add) / 2.0), pooled_chi2=float(c2_add.sum()),
                                pooled_dof=int(2 * len(c2_add)),
                                pooled_p=float(chi2dist.sf(c2_add.sum(), 2 * len(c2_add)))),
                best_pairing=best, all_pairings=out_pairings,
                sigma_lsq_ps=sig_lsq, sigma_pairing_corrected_ps=sig_cor,
                bias_on_3board_quote_ps2=float(g / 3.0),
                note="the 3-board solve is v-g in the two combos containing a board's partner and v+g in "
                     "the one that leaves it out, so the combo-averaged quote is low by g/3; "
                     "sigma_pairing_corrected_ps = sqrt(v_lsq + g/3). Stored alongside, never instead of, "
                     "the standard solve. See notes/fourboard-lsq-test.md.")


def lsq_pairs(tables):
    """Least-squares sigma per board from all pair widths (median over cleaned tracks per table); needs >= 4 boards."""
    rows, y, labels = [], [], []
    boards = sorted({r for t in tables.values() for r in t["roles"]})
    if len(boards) < 4:
        return None
    for combo, t in tables.items():
        for c in [c for c in t["df"].columns if c.startswith("res_pair_")]:
            a, b = c[len("res_pair_"):].split("-")
            v = t["df"][c]
            v = v[v > 0]
            if len(v) < 20:
                continue
            row = np.zeros(len(boards))
            row[boards.index(a)] = 1
            row[boards.index(b)] = 1
            rows.append(row)
            y.append(float(np.median(v)) ** 2)
            labels.append("%s in %s" % (c[len("res_pair_"):], combo))
    if len(rows) < len(boards) + 1:
        return None
    A, yv = np.array(rows), np.array(y)
    s2, *_ = np.linalg.lstsq(A, yv, rcond=None)
    pred = A @ s2
    resid = np.sqrt(np.abs(yv)) - np.sqrt(np.abs(pred))
    return dict(boards=boards, sigma={b: float(np.sqrt(v)) if v > 0 else float("nan") for b, v in zip(boards, s2)},
                rms_residual=float(np.sqrt(np.mean(resid ** 2))), residuals=dict(zip(labels, [float(x) for x in resid])))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--inputs", nargs="+", required=True, help="step-13 resolution_table_*.csv (one per combo; one file for a 3-board telescope)")
    p.add_argument("-o", "--outdir", required=True)
    p.add_argument("--label", default=None, help="run label for the outputs (default: derived from the input directory)")
    p.add_argument("--min-boot", type=int, default=100, dest="min_boot", help="minimum accepted bootstrap resamples per track (default 100)")
    p.add_argument("--margin-lo", type=float, default=0.35, dest="margin_lo", help="near-degenerate: sigma / smallest pair width below this is dropped (default 0.35)")
    p.add_argument("--margin-hi", type=float, default=0.95, dest="margin_hi", help="near-degenerate: sigma / smallest pair width above this is dropped (default 0.95)")
    p.add_argument("--def-syst", type=float, default=0.01, dest="def_syst", help="relative definition/convergence systematic (default 0.01)")
    p.add_argument("--nevt-standard", type=int, default=100, dest="nevt_standard",
                   help="standard track-occupancy threshold for the quote (default 100, step 12's own --minimum_nevt)")
    p.add_argument("--nevt-conservative", type=int, default=300, dest="nevt_conservative",
                   help="conservative track-occupancy threshold, quoted alongside the standard one (default 300)")
    p.add_argument("--fourboard", default=None,
                   help="four-board per-quadruple pair-width table for the pairing-covariance fit "
                        "(utils/fourboard_pairing.py output). Default: fourboard_pairs_*.csv next to the input tables.")
    p.add_argument("--pairing-min-events", type=int, default=500, dest="pairing_min_events",
                   help="minimum four-board events per pixel quadruple for the pairing fit (default 500)")
    p.add_argument("--pairing-min-quads", type=int, default=30, dest="pairing_min_quads",
                   help="minimum qualifying quadruples for the pairing fit (default 30)")
    p.add_argument("--alignment", default=None,
                   help="optional step-6 --find_alignment yaml (core/path_finder.py's <track_label>_alignment.yaml); "
                        "when given, the translation-implied pixel offset is compared to the weighted modal offset "
                        "per partner board and a mismatch is warned on. Conservative: role<->board-id is inferred "
                        "from the yaml's own combo-label keys, a board with no recorded translation that is not the "
                        "trigger is flagged rather than silently assumed zero, and without --config the column axis "
                        "is not compared at all (see alignment_check()'s docstring for why).")
    p.add_argument("--config", default=None,
                   help="optional board-config yaml (the -c/--config path_finder.py itself takes) for --alignment: "
                        "supplies each board's rotation.y so the column-axis prediction can correct for a real "
                        "telescope tilt (see load_rotation()). Without it, --alignment compares rows only.")
    add_output_arguments(p)
    a = p.parse_args()
    set_output_options(a.format, a.split)

    files = sorted(sum([glob(x) for x in a.inputs], []))
    if not files:
        sys.exit("quote_resolution: no input files")
    label = a.label or os.path.basename(os.path.dirname(os.path.abspath(files[0])))
    os.makedirs(a.outdir, exist_ok=True)
    align_data = load_alignment(a.alignment) if a.alignment else None
    if align_data is not None and not align_data.get("present"):
        print("WARNING: quote_resolution: --alignment %s not usable: %s" % (a.alignment, align_data["reason"]), file=sys.stderr)
    rot_y, rot_reason = None, None
    if a.config:
        if align_data is not None and align_data.get("present"):
            rot_y, rot_reason = load_rotation(a.config, align_data["run_name"], align_data["role_to_id"])
        else:
            rot_reason = "--config given but --alignment is not usable, so there is no role<->id map to load rotations for"
        if rot_reason:
            print("WARNING: quote_resolution: --config %s not usable: %s" % (a.config, rot_reason), file=sys.stderr)

    tables = {}
    for f in files:
        df = pd.read_csv(f)
        roles = [r for r in ROLES if "res_%s" % r in df.columns]
        combo = combo_of(f)
        # optional event counts from the step-11 CSV next to the table (nevt_<combo>_*.csv)
        nv = glob(os.path.join(os.path.dirname(os.path.abspath(f)), "nevt_%s_*.csv" % combo))
        if nv:
            keys = ["row_%s" % r for r in roles] + ["col_%s" % r for r in roles]
            df = df.merge(pd.read_csv(nv[0])[keys + ["nevt"]], on=keys, how="left")
        cdf, counts = clean(df, roles, a)
        tables[combo] = dict(df=cdf, roles=roles, counts=counts, has_nevt=bool(nv))

    # per board per table
    result = {"label": label, "recipe": "see utils/quote_resolution.py docstring", "tables": {}, "boards": {}}
    per_board = {}
    for combo, t in tables.items():
        result["tables"][combo] = dict(t["counts"])
        result["tables"][combo]["boards"] = {}
        for r in t["roles"]:
            P = per_pixel(t["df"], r)
            s = board_summary(P, a)
            s["map"] = P
            result["tables"][combo]["boards"][r] = {k: v for k, v in s.items() if k != "map"}
            per_board.setdefault(r, {})[combo] = s

    # quoted number per board: pixel-centric combination across the tables the board appears in
    combined_maps, central_maps = {}, {}
    for r, per_combo in per_board.items():
        pooled = pd.concat([tables[c]["df"].assign(_table=c) for c in per_combo], ignore_index=True)
        Pc = per_pixel(pooled, r)
        sc = board_summary(Pc, a)
        combined_maps[r] = Pc
        ill = {c: illumination(tables[c]["df"], r, tables[c]["roles"], tables[c]["df"]["nevt"] if tables[c]["has_nevt"] else None) for c in per_combo}
        # central-hit map: pool the central tracks of every table (weighted modal offsets are per table)
        cen_pool = pd.concat(
            [tables[c]["df"][central_mask(tables[c]["df"], r, tables[c]["roles"],
                              weights=offset_weight(tables[c]["df"], r, tables[c]["df"]["nevt"] if tables[c]["has_nevt"] else None))[0]
             ].assign(_table=c) for c in per_combo], ignore_index=True)
        Pcen = per_pixel(cen_pool, r) if len(cen_pool) >= 5 else None
        central_maps[r] = Pcen
        scen = board_summary(Pcen, a) if Pcen is not None and len(Pcen) >= 5 else None
        ill_comb = dict(value_central=float(np.nanmean([v["value_central"] for v in ill.values()])),
                        edge_minus_central=float(np.nanmedian([v["edge_minus_central"] for v in ill.values()])),
                        edge_event_share=float(np.mean([v["edge_event_share"] for v in ill.values()])),
                        weights="nevt" if all(tables[c]["has_nevt"] for c in per_combo) else "1/err^2 (no nevt csv found)")
        # modal/runner-up weight-share diagnostics (per table, per partner board); warn when thin
        ratios = []
        for c in per_combo:
            for o, dstat in ill[c]["central_diag"].items():
                ratios.append((dstat["ratio"], c, o, dstat))
                if dstat["ratio"] is not None and dstat["ratio"] < 1.5:
                    print("WARNING: quote_resolution: table %s, board %s vs partner %s: modal/runner-up "
                          "weight ratio %.2f (%.0f%% vs %.0f%%) < 1.5 - the central-hit definition is not "
                          "well resolved (likely a near-half-pixel misalignment)."
                          % (c, r, o, dstat["ratio"], 100 * dstat["modal_share"], 100 * dstat["runnerup_share"]),
                          file=sys.stderr)
        # None means "no runner-up offset at all" (as unambiguous as a vote gets), so it sorts as the
        # LARGEST ratio, not the smallest, when picking the worst (minimum) one below.
        ratios.sort(key=lambda x: float("inf") if x[0] is None else x[0])
        worst = ratios[0] if ratios else None
        pmms = [(v["fraction"], c, o) for c in per_combo for o, v in ill[c]["pixel_mode_match"].items()]
        pmms.sort(key=lambda x: x[0])
        worst_pmm = pmms[0] if pmms else None
        # optional --alignment cross-check, per table. n_compared counts actual comparisons made (not
        # tables/partners visited) so a run where every lookup failed (unparsable yaml, role missing
        # from role_to_id, unsupported yaml layout) is reported as present=False with why, rather than
        # as "agrees with the step-6 alignment" on zero evidence.
        align_summary = None
        if align_data is not None:
            mismatches, reasons, n_compared = [], set(), 0
            if not align_data.get("present"):
                reasons.add(align_data.get("reason", "unknown"))
            for c in per_combo:
                ac = alignment_check(align_data, r, tables[c]["roles"], rot_y=rot_y)
                ill[c]["alignment_check"] = ac
                if not ac.get("present"):
                    reasons.add(ac.get("reason", "unknown"))
                    continue
                for o, av in ac.items():
                    if o == "present" or not isinstance(av, dict):
                        continue
                    if not av.get("present"):
                        reasons.add(av.get("reason", "unknown"))
                        continue
                    if av.get("zero_assumed_boards"):
                        print("WARNING: quote_resolution: table %s, board %s vs partner %s: %s"
                              % (c, r, o, av["zero_assumed_reason"]), file=sys.stderr)
                    n_compared += 1
                    pdr, pdc = av["predicted_offset"]
                    mdr, mdc = ill[c]["central_diag"][o]["offset"]
                    av["modal_offset"] = [mdr, mdc]
                    # row_only (pdc is None, no --config): only the rotation-invariant row axis is checked.
                    av["mismatch"] = bool(round(pdr) != mdr or (pdc is not None and round(pdc) != mdc))
                    if av["mismatch"]:
                        mismatches.append((c, o, pdr, pdc, mdr, mdc, av.get("row_only", False)))
                        print("WARNING: quote_resolution: table %s, board %s vs partner %s: step-6 "
                              "alignment predicts %srow offset %.1f px%s, weighted mode is (%d, %d) px "
                              "- mismatch (--alignment %s)."
                              % (c, r, o, "" if pdc is None else "offset (", pdr,
                                 "" if pdc is None else ", %.1f) px col" % pdc, mdr, mdc, a.alignment),
                              file=sys.stderr)
            if n_compared == 0:
                align_summary = dict(present=False, source=a.alignment,
                                     reason="; ".join(sorted(reasons)) if reasons else "no partner offset could be compared")
            else:
                align_summary = dict(present=True, source=a.alignment, n_compared=n_compared, n_mismatch=len(mismatches),
                                     row_only=(rot_y is None),
                                     mismatches=[dict(table=c, partner=o, predicted_offset=[pdr, pdc], modal_offset=[mdr, mdc],
                                                      row_only=ro) for c, o, pdr, pdc, mdr, mdc, ro in mismatches])
        # partner-board, per pixel: half range of the per-table per-pixel values over pixels seen in >= 2 tables
        maps = {c: per_combo[c]["map"]["mean"] for c in per_combo}
        M = pd.concat(maps, axis=1)             # columns = tables, index = pixels
        multi = M.notna().sum(axis=1) >= 2
        partner_board = float(np.median(0.5 * (M[multi].max(axis=1) - M[multi].min(axis=1)))) if multi.any() else None
        partner_pixel = sc["partner_pixel"]
        result["boards"][r] = dict(
            value=sc["value"], stat=sc["stat"], pixel_spread=sc["pixel_spread"],
            nevt_thresholds=nevt_quotes(pooled, r, a, all(tables[c]["has_nevt"] for c in per_combo)),
            partner_pixel=partner_pixel, partner_board=partner_board,
            partner_quoted=partner_board if partner_board is not None else partner_pixel,
            partner_source=("per pixel across tables, %d px in >= 2 tables" % int(multi.sum())) if partner_board is not None else "partner-pixel spread within the run",
            definition=float(a.def_syst * sc["value"]),
            half_left_minus_right=sc["half_left_minus_right"],
            n_tables=len(per_combo), values_per_table={c: s["value"] for c, s in per_combo.items()},
            coverage=dict(pixels_per_table={c: int(len(per_combo[c]["map"])) for c in per_combo},
                          pixels_union=int(len(Pc)), pixels_in_2_or_more=int(multi.sum())),
            illumination=ill_comb, illumination_per_table=ill,
            central=dict(value=scen["value"], stat=scen["stat"], pixel_spread=scen["pixel_spread"], n_pixels=scen["n_pixels"],
                         half_left_minus_right=scen["half_left_minus_right"]) if scen else None,
            central_diagnostics_summary=(dict(min_modal_runnerup_ratio=worst[0], table=worst[1], partner=worst[2],
                                              modal_share=worst[3]["modal_share"], runnerup_share=worst[3]["runnerup_share"])
                                         if worst else None),
            pixel_mode_match_summary=(dict(min_fraction=worst_pmm[0], table=worst_pmm[1], partner=worst_pmm[2])
                                      if worst_pmm else None),
            alignment_check=align_summary,
        )
        # per-pixel maps as CSV (canonical average, and central) for downstream use
        Pc.rename(columns={"mean": "res", "err": "err"}).to_csv(os.path.join(a.outdir, "%s_map_%s_average.csv" % (label, r)))
        if Pcen is not None:
            Pcen.rename(columns={"mean": "res", "err": "err"}).to_csv(os.path.join(a.outdir, "%s_map_%s_central.csv" % (label, r)))
    result["pair_lsq"] = lsq_pairs(tables)
    fb = a.fourboard
    if fb is None:
        cand = sorted(glob(os.path.join(os.path.dirname(os.path.abspath(files[0])), "fourboard_pairs_*.csv")))
        fb = cand[0] if cand else None
    result["pairing_covariance"] = pairing_covariance(fb, a)

    # ---- outputs: json, markdown, figure
    with open(os.path.join(a.outdir, "%s_resolution_quote.json" % label), "w") as f:
        json.dump(result, f, indent=1, default=float)
    md = ["# %s - quoted resolutions\n" % label,
          "| board | average (canonical) [ps] | stat | partner (%s) | definition | pixel spread | left−right | central-hit [ps] | central pixel spread | off-nominal − central | off-nominal event share | nevt≥%d [ps] (px) | nevt≥%d [ps] (px) | tables | modal/runner-up (min) | pixel-mode match (min) |"
          % ("board / pixel", a.nevt_standard, a.nevt_conservative),
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r, b in result["boards"].items():
        il = b["illumination"]
        cen = b["central"]
        nt = b["nevt_thresholds"]

        def _nt(key):
            if not nt.get("present") or not nt.get(key, {}).get("present"):
                return "n/a"
            return "%.2f (%d)" % (nt[key]["value"], nt[key]["n_pixels"])
        cd = b["central_diagnostics_summary"]
        cd_str = ("%s (%s vs %s)" % ("∞" if cd["min_modal_runnerup_ratio"] is None else "%.2fx" % cd["min_modal_runnerup_ratio"],
                                     cd["table"], cd["partner"])) if cd else "n/a"
        pmm = b["pixel_mode_match_summary"]
        pmm_str = "%.0f %% (%s vs %s)" % (100 * pmm["min_fraction"], pmm["table"], pmm["partner"]) if pmm else "n/a"
        md.append("| %s | %.2f | ±%.2f | ±%.2f (%s) | ±%.2f | %.2f | %+.2f | %s | %s | %+.2f | %.0f %% | %s | %s | %d | %s | %s |" % (
            r, b["value"], b["stat"], b["partner_quoted"], b["partner_source"], b["definition"], b["pixel_spread"], b["half_left_minus_right"],
            ("%.2f ± %.2f" % (cen["value"], cen["stat"])) if cen else "n/a", ("%.2f (%d px)" % (cen["pixel_spread"], cen["n_pixels"])) if cen else "n/a",
            il["edge_minus_central"], 100 * il["edge_event_share"], _nt("standard"), _nt("conservative"), b["n_tables"], cd_str, pmm_str))
    md.append("\nCoverage: " + "; ".join("%s: %s, union %d px, %d px in ≥2 tables" % (
        r, ", ".join("%s %d" % (c, k) for c, k in b["coverage"]["pixels_per_table"].items()), b["coverage"]["pixels_union"], b["coverage"]["pixels_in_2_or_more"]) for r, b in result["boards"].items()))
    md.append("\nPer table: " + "; ".join("%s: %s (kept %d/%d, dropped %d flagged + %d near-degenerate)" % (
        c, ", ".join("%s %.2f" % (r, s["value"]) for r, s in t["boards"].items()), t["kept"], t["n_tracks"], t["dropped_flagged"], t["dropped_degenerate"]) for c, t in result["tables"].items()))
    if result["pair_lsq"]:
        L = result["pair_lsq"]
        md.append("\nConsistency check (≥ 4 boards): least-squares solve of all pair widths → " + ", ".join("%s %.2f" % (b, v) for b, v in L["sigma"].items()) + " ps; rms residual %.2f ps." % L["rms_residual"])
    pc = result["pairing_covariance"]
    if pc.get("present"):
        bp = pc["best_pairing"]
        md.append("\nPairing covariance (4-board coincidences, %d quadruples with ≥ %d events, median %d): best-supported "
                  "pairing %s, g = %+.0f ± %.0f ps², χ²/dof %.2f on the 1 d.o.f. it leaves (plain additivity: χ²/dof %.2f, "
                  "p = %.1g). Per board σ_LSQ → σ corrected for the g/3 bias: %s. Stored alongside the standard solve, not "
                  "instead of it."
                  % (pc["n_quadruples_used"], pc["cuts"]["min_events_per_quadruple"], pc["median_events_per_quadruple"],
                     bp["pairing"], bp["g_ps2"], bp["g_err_ps2"], bp["chi2_per_dof"],
                     pc["additivity"]["chi2_per_dof"], pc["additivity"]["pooled_p"],
                     ", ".join("%s %.2f → %.2f" % (b, pc["sigma_lsq_ps"][b], pc["sigma_pairing_corrected_ps"][b])
                               for b in pc["sigma_lsq_ps"])))
    else:
        md.append("\nPairing covariance: not computed - %s." % pc["reason"])
    nt0 = result["boards"][next(iter(result["boards"]))]["nevt_thresholds"]
    if nt0.get("present"):
        md.append("\nEvent thresholds: the last two columns are the same board value recomputed on tracks with at least "
                  "%d and at least %d events (pixel counts in brackets). The headline number is the standard (%d) one - "
                  "that is the set step 12 ran on; the conservative column exists because the converged mixture width is "
                  "still ~5 %% low at ~300 events, so a board whose two numbers differ is a board resting on "
                  "low-occupancy pixels." % (a.nevt_standard, a.nevt_conservative, a.nevt_standard))
    else:
        md.append("\nEvent thresholds: not computed - %s." % nt0.get("reason", "unknown"))
    md.append("\nTwo board-level numbers: the CANONICAL one is the average map (every track using the pixel, 1/err²-weighted = event-weighted, i.e. the operating resolution under this illumination); the central-hit one uses only tracks whose partners sit at the modal offset (one sub-region of the pixel) and is the more geometry-independent number for chip-to-chip comparisons. Both maps are written as CSV (<label>_map_<board>_{average,central}.csv).")
    md.append("\nIllumination: 'central-only' uses tracks whose partner pixels all sit at the modal offset (one sub-region of the pixel, set by the fractional inter-plane alignment); 'off-nominal − central' is the median over pixels of the same pixel's off-nominal minus central value (the complementary side sub-region, a few hundred µm wide at 1.3 mm pitch - not a charge-sharing edge); the value quoted above is the illumination-weighted average of both (weights: %s), i.e. the operating resolution under this beam geometry and alignment - with a different alignment, beam angle or a tracker the split moves." % result["boards"][next(iter(result["boards"]))]["illumination"]["weights"])
    md.append("\nConvention: FWHM/2.355 of the converged Gaussian mixture of the TWC-corrected pairwise TOA difference (a core width), 3-board solve per track, 1/err²-weighted per pixel, robust Gaussian mean over pixels; the definition systematic is the %.0f %% convergence softness of that width, the core-vs-RMS convention itself is not folded in." % (100 * a.def_syst))
    md.append("\nCentral-hit modal offset: the 'modal (nominal) offset' that defines central vs off-nominal tracks is EVENT-WEIGHTED (nevt when available, else 1/err² - the same weights as the illumination split), not a plain per-track vote; a per-track vote can flip to a side offset as selection depth grows even though the nominal offset keeps the most events. 'modal/runner-up (min)' is the smallest weight-share ratio of the winning offset over the next most common one, over all (table, partner board) pairs - below 1.5x the split is not well resolved (typically a near-half-pixel misalignment; a WARNING is also printed). 'pixel-mode match (min)' is the smallest fraction, over the same pairs, of a board's pixels whose OWN most-weighted partner offset agrees with the table-level mode - a rigid-translation check that a rotation would fail even with a clear table-level vote. Full per-table, per-partner detail is in the JSON (`illumination_per_table.<combo>.central_diag` / `.pixel_mode_match`).")
    if any(b.get("alignment_check") for b in result["boards"].values()):
        ac_lines = []
        any_row_only = False
        for r, b in result["boards"].items():
            ac = b.get("alignment_check")
            if not ac:
                continue
            if not ac.get("present"):
                ac_lines.append("%s: not compared - %s" % (r, ac.get("reason", "unknown")))
                continue

            def _fmt(m):
                if m["predicted_offset"][1] is None:
                    return "%s vs %s predicted row %.1f px (col not compared, row_only), weighted mode (%d, %d) px" % (
                        m["table"], m["partner"], m["predicted_offset"][0], m["modal_offset"][0], m["modal_offset"][1])
                return "%s vs %s predicted (%.1f, %.1f) px, weighted mode (%d, %d) px" % (
                    m["table"], m["partner"], m["predicted_offset"][0], m["predicted_offset"][1],
                    m["modal_offset"][0], m["modal_offset"][1])
            any_row_only = any_row_only or ac.get("row_only", False)
            if ac["n_mismatch"] == 0:
                ac_lines.append("%s: agrees with the step-6 alignment on all %d comparison(s)%s"
                                % (r, ac["n_compared"], " (row axis only)" if ac.get("row_only") else ""))
            else:
                ac_lines.append("%s: %d/%d mismatch(es)%s - %s" % (
                    r, ac["n_mismatch"], ac["n_compared"], " (row axis only)" if ac.get("row_only") else "",
                    "; ".join(_fmt(m) for m in ac["mismatches"])))
        md.append("\nAlignment cross-check (--alignment %s%s, conservative: role<->id inferred from the yaml's own "
                  "combo labels, boards absent from the yaml's translation record flagged rather than assumed "
                  "zero): %s.%s"
                  % (a.alignment, (" --config %s" % a.config) if a.config else "", "; ".join(ac_lines),
                     " Column axis not compared anywhere (no --config, so rotation.y is unknown and, per the "
                     "board configs checked, is not small enough on most DESY-May runs to assume away)."
                     if any_row_only else ""))
    with open(os.path.join(a.outdir, "%s_resolution_quote.md" % label), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    # figure: per board, the per-table values with the quoted band, plus one map per board (first table it appears in)
    boards = list(result["boards"].keys())
    fig, axs = plt.subplots(3, len(boards), figsize=(4.9 * len(boards) + 0.8, 14.6), squeeze=False)
    for j, r in enumerate(boards):
        b = result["boards"][r]
        ax = axs[0, j]
        combos = list(b["values_per_table"].keys())
        ys = [b["values_per_table"][c] for c in combos]
        errs = [per_board[r][c]["stat"] for c in combos]
        ax.errorbar(range(len(combos)), ys, yerr=errs, fmt="o", color=rc(r), ms=6, capsize=3, label="value per table (± stat)")
        ax.axhline(b["value"], color="k", lw=1.4, label="quoted (combined per-pixel map): %.2f ps" % b["value"])
        tot = np.hypot(b["partner_quoted"], b["definition"])
        ax.axhspan(b["value"] - tot, b["value"] + tot, color=rc(r), alpha=0.15, lw=0, label="± partner ⊕ definition = ±%.2f ps" % tot)
        ax.set_xticks(range(len(combos)))
        ax.set_xticklabels([c.replace("-", "\n") for c in combos], fontsize=8)
        ax.set_ylabel("σ_%s [ps]" % r)
        ax.set_title("%s - %.2f ± %.2f (stat) ± %.2f (partner) ± %.2f (def.) ps\npixel spread %.2f, left−right %+.2f ps"
                     % (r, b["value"], b["stat"], b["partner_quoted"], b["definition"], b["pixel_spread"], b["half_left_minus_right"]), fontsize=9)
        ax.legend(fontsize=7.5, loc="best")
        ax.grid(alpha=0.3, axis="y")
        ax = axs[1, j]
        P = combined_maps[r]
        img = np.full((GRID, GRID), np.nan)
        for (rr, cc), v in P["mean"].items():
            img[int(rr), int(cc)] = v
        fin = img[np.isfinite(img)]
        im = pixel_map(ax, img, cmap="viridis", vmin=np.percentile(fin, 2), vmax=np.percentile(fin, 98), nan_blank=True)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("σ_%s per pixel [ps]" % r, fontsize=8.5)
        cov = b["coverage"]
        ax.set_title("%s - AVERAGE map (canonical): all tracks, %d table(s)\n(%d pixels; %d in ≥2 tables; grey = no track after cleaning)"
                     % (r, b["n_tables"], cov["pixels_union"], cov["pixels_in_2_or_more"]), fontsize=9)
        ax = axs[2, j]
        Pn = central_maps[r]
        if Pn is None:
            ax.axis("off")
            ax.text(0.5, 0.5, "no central tracks", ha="center", va="center", transform=ax.transAxes)
        else:
            img2 = np.full((GRID, GRID), np.nan)
            for (rr, cc), v in Pn["mean"].items():
                img2[int(rr), int(cc)] = v
            fin2 = img2[np.isfinite(img2)]
            im2 = pixel_map(ax, img2, cmap="viridis", vmin=np.percentile(fin, 2), vmax=np.percentile(fin, 98), nan_blank=True)   # same colour scale as the average map
            cb2 = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.03)
            cb2.set_label("σ_%s per pixel [ps]" % r, fontsize=8.5)
            cen = b["central"]
            ax.set_title("%s - CENTRAL-hit map: partners at the modal offset only\n(%d pixels; board value %.2f ps; same colour scale as above)"
                         % (r, fin2.size, cen["value"] if cen else float("nan")), fontsize=9)
    fig.suptitle("%s - quoted resolutions (same recipe for any telescope with ≥ 3 boards)" % label, fontsize=11.5, y=0.995)
    cap = ("TOP: per board, the value from each step-13 table it appears in (a three-plane telescope has one table; "
           "leave-one-out combos of a four-board telescope give one each): the robust Gaussian mean over the "
           "per-pixel values (1/err²-weighted over the tracks using each pixel, after dropping flagged and "
           "near-degenerate tracks), with its negligible statistical error; black = the quoted value: the same "
           "robust Gaussian mean over the COMBINED per-pixel map (all tables' tracks pooled per pixel - a pixel "
           "seen in one table only contributes that table's tracks, so partial overlaps of an angled beam are "
           "handled); band = the partner systematic (per pixel across tables: half the range of a pixel's "
           "per-table values, median over pixels present in ≥ 2 tables; with a single table, or no pixel in two "
           "tables, the excess spread over partner pixels within the run) added in quadrature with the definition "
           "systematic (the %.0f %% convergence softness of an FWHM-of-a-mixture width). Titles also give the "
           "pixel-to-pixel spread (chip structure, quoted as a spread not an error) and the error-weighted "
           "left − right chip-half difference. MIDDLE: the canonical AVERAGE per-pixel map (all tracks using the pixel, "
           "1/err²-weighted, tables pooled). BOTTOM: the CENTRAL-hit map on the same colour scale - only tracks whose "
           "partner pixels sit at the modal offset, i.e. one sub-region of the pixel set by the inter-plane alignment; "
           "its board value is the second, more geometry-independent number, and the difference to the average map "
           "is the illumination systematic. The convention "
           "(core width from the converged mixture, 3-board solve per pixel triple) is the same for every "
           "campaign; only the number of tables changes." % (100 * a.def_syst))
    finish(fig, os.path.join(a.outdir, "%s_resolution_quote.png" % label), cap, wspace=0.34, hspace=0.40, top=0.94)
    print("wrote %s/%s_resolution_quote.{json,md,%s} and per-board map CSVs" % (a.outdir, label, a.format))


if __name__ == "__main__":
    main()
