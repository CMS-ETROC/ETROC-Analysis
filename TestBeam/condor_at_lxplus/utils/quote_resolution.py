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
     for N).
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))
sys.path.insert(0, HERE)
import fit_bootstrap_results as fbr          # noqa: E402  (step 13's own robust Gaussian)
from telescope_diagnostics import GRID, pixel_map, set_output_options, add_output_arguments  # noqa: E402
from track_diagnostics import finish, rc  # noqa: E402

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


def central_mask(df, r, roles):
    """True for tracks whose partner pixels all sit at the modal (nominal) offset from board r's pixel."""
    central = np.ones(len(df), bool)
    for o in [x for x in roles if x != r]:
        dr = (df["row_%s" % o] - df["row_%s" % r])
        dc = (df["col_%s" % o] - df["col_%s" % r])
        central &= ((dr == dr.mode().iloc[0]) & (dc == dc.mode().iloc[0])).to_numpy()
    return central


def illumination(df, r, roles, nevt=None):
    """Central (all partners at the modal offset) vs off-nominal tracks for board r."""
    central = central_mask(df, r, roles)
    w = nevt if nevt is not None else 1.0 / df["err_%s" % r] ** 2
    c, e = df[central], df[~central]
    if len(c) < 5:
        return dict(value_central=np.nan, edge_minus_central=np.nan, edge_event_share=float(w[~central].sum() / w.sum()), n_central=int(len(c)))
    pc = c.groupby(["row_%s" % r, "col_%s" % r])["res_%s" % r].mean()
    stc = fbr.perform_robust_unbinned_fit(pd.Series(pc.values), 2.5)
    pe = e.groupby(["row_%s" % r, "col_%s" % r]).apply(lambda x: np.average(x["res_%s" % r], weights=w.loc[x.index]))
    j = pd.concat([pc.rename("c"), pe.rename("e")], axis=1).dropna()
    return dict(value_central=float(stc["mu"]), edge_minus_central=float((j.e - j.c).median()) if len(j) else np.nan,
                edge_event_share=float(w[~central].sum() / w.sum()), n_central=int(len(c)))


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
    add_output_arguments(p)
    a = p.parse_args()
    set_output_options(a.format, a.split)

    files = sorted(sum([glob(x) for x in a.inputs], []))
    if not files:
        sys.exit("quote_resolution: no input files")
    label = a.label or os.path.basename(os.path.dirname(os.path.abspath(files[0])))
    os.makedirs(a.outdir, exist_ok=True)

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
        # central-hit map: pool the central tracks of every table (modal offsets are per table)
        cen_pool = pd.concat([tables[c]["df"][central_mask(tables[c]["df"], r, tables[c]["roles"])].assign(_table=c) for c in per_combo], ignore_index=True)
        Pcen = per_pixel(cen_pool, r) if len(cen_pool) >= 5 else None
        central_maps[r] = Pcen
        scen = board_summary(Pcen, a) if Pcen is not None and len(Pcen) >= 5 else None
        ill_comb = dict(value_central=float(np.nanmean([v["value_central"] for v in ill.values()])),
                        edge_minus_central=float(np.nanmedian([v["edge_minus_central"] for v in ill.values()])),
                        edge_event_share=float(np.mean([v["edge_event_share"] for v in ill.values()])),
                        weights="nevt" if all(tables[c]["has_nevt"] for c in per_combo) else "1/err^2 (no nevt csv found)")
        # partner-board, per pixel: half range of the per-table per-pixel values over pixels seen in >= 2 tables
        maps = {c: per_combo[c]["map"]["mean"] for c in per_combo}
        M = pd.concat(maps, axis=1)             # columns = tables, index = pixels
        multi = M.notna().sum(axis=1) >= 2
        partner_board = float(np.median(0.5 * (M[multi].max(axis=1) - M[multi].min(axis=1)))) if multi.any() else None
        partner_pixel = sc["partner_pixel"]
        result["boards"][r] = dict(
            value=sc["value"], stat=sc["stat"], pixel_spread=sc["pixel_spread"],
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
        )
        # per-pixel maps as CSV (canonical average, and central) for downstream use
        Pc.rename(columns={"mean": "res", "err": "err"}).to_csv(os.path.join(a.outdir, "%s_map_%s_average.csv" % (label, r)))
        if Pcen is not None:
            Pcen.rename(columns={"mean": "res", "err": "err"}).to_csv(os.path.join(a.outdir, "%s_map_%s_central.csv" % (label, r)))
    result["pair_lsq"] = lsq_pairs(tables)

    # ---- outputs: json, markdown, figure
    with open(os.path.join(a.outdir, "%s_resolution_quote.json" % label), "w") as f:
        json.dump(result, f, indent=1, default=float)
    md = ["# %s - quoted resolutions\n" % label,
          "| board | average (canonical) [ps] | stat | partner (%s) | definition | pixel spread | left−right | central-hit [ps] | central pixel spread | off-nominal − central | off-nominal event share | tables |" % ("board / pixel"),
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r, b in result["boards"].items():
        il = b["illumination"]
        cen = b["central"]
        md.append("| %s | %.2f | ±%.2f | ±%.2f (%s) | ±%.2f | %.2f | %+.2f | %s | %s | %+.2f | %.0f %% | %d |" % (
            r, b["value"], b["stat"], b["partner_quoted"], b["partner_source"], b["definition"], b["pixel_spread"], b["half_left_minus_right"],
            ("%.2f ± %.2f" % (cen["value"], cen["stat"])) if cen else "n/a", ("%.2f (%d px)" % (cen["pixel_spread"], cen["n_pixels"])) if cen else "n/a",
            il["edge_minus_central"], 100 * il["edge_event_share"], b["n_tables"]))
    md.append("\nCoverage: " + "; ".join("%s: %s, union %d px, %d px in ≥2 tables" % (
        r, ", ".join("%s %d" % (c, k) for c, k in b["coverage"]["pixels_per_table"].items()), b["coverage"]["pixels_union"], b["coverage"]["pixels_in_2_or_more"]) for r, b in result["boards"].items()))
    md.append("\nPer table: " + "; ".join("%s: %s (kept %d/%d, dropped %d flagged + %d near-degenerate)" % (
        c, ", ".join("%s %.2f" % (r, s["value"]) for r, s in t["boards"].items()), t["kept"], t["n_tracks"], t["dropped_flagged"], t["dropped_degenerate"]) for c, t in result["tables"].items()))
    if result["pair_lsq"]:
        L = result["pair_lsq"]
        md.append("\nConsistency check (≥ 4 boards): least-squares solve of all pair widths → " + ", ".join("%s %.2f" % (b, v) for b, v in L["sigma"].items()) + " ps; rms residual %.2f ps." % L["rms_residual"])
    md.append("\nTwo board-level numbers: the CANONICAL one is the average map (every track using the pixel, 1/err²-weighted = event-weighted, i.e. the operating resolution under this illumination); the central-hit one uses only tracks whose partners sit at the modal offset (one sub-region of the pixel) and is the more geometry-independent number for chip-to-chip comparisons. Both maps are written as CSV (<label>_map_<board>_{average,central}.csv).")
    md.append("\nIllumination: 'central-only' uses tracks whose partner pixels all sit at the modal offset (one sub-region of the pixel, set by the fractional inter-plane alignment); 'off-nominal − central' is the median over pixels of the same pixel's off-nominal minus central value (the complementary side sub-region, a few hundred µm wide at 1.3 mm pitch - not a charge-sharing edge); the value quoted above is the illumination-weighted average of both (weights: %s), i.e. the operating resolution under this beam geometry and alignment - with a different alignment, beam angle or a tracker the split moves." % result["boards"][next(iter(result["boards"]))]["illumination"]["weights"])
    md.append("\nConvention: FWHM/2.355 of the converged Gaussian mixture of the TWC-corrected pairwise TOA difference (a core width), 3-board solve per track, 1/err²-weighted per pixel, robust Gaussian mean over pixels; the definition systematic is the %.0f %% convergence softness of that width, the core-vs-RMS convention itself is not folded in." % (100 * a.def_syst))
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
