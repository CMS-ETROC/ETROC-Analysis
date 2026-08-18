#!/usr/bin/env python3
"""Diagnostics of the resolution extraction (steps 12/13): the anatomy of one
bootstrap trial, the statistical behaviour over all tracks, and the consistency
of the same board / the same pixel pair across different board combinations.

WHAT THIS IS
------------
Step 12 (bootstrap.py) turns one track file (a fixed pixel triple, all its
events after the step-10 cuts, in ps) into a resolution per board by:
  1. time-walk correction (TWC): per board, a 2nd-order polynomial in the
     board's own TOT fitted to (mean TOA of the other two boards) - (own TOA),
     added to the TOA, twice iteratively;
  2. for each of the 3 pairwise TOA differences after TWC, a Gaussian-mixture
     fit (1, 2 or 3 components when < 1500 events, 3 otherwise; the one with
     the smallest KS distance to the data is kept and accepted if its KS
     p-value is >= 1e-3 or its distance <= 0.03; on failure the sample is
     refitted with the p threshold halved per attempt down to 1e-6 - then the
     single-shot gives up, the bootstrap phase after --iteration_limit
     attempts); the pair width is FWHM / 2.355 of the mixture, fitted to
     convergence (EM tol 1e-6);
  3. the 3-board solve sigma_a^2 = (s_ab^2 + s_ac^2 - s_bc^2) / 2 per board;
  4. once on the full sample ("single-shot") and -n times (200 in the H1
     runs) on resamples of the same size drawn WITH replacement ("bootstrap").
Step 13 (fit_bootstrap_results.py) then fits a Gaussian (median/IQR-seeded,
clipped at --sigma_cut = 2.5 of that width, unbinned ML) to the bootstrap values
per board: its mean is the track's `res_<role>`, its width the track's
`err_<role>`; the single-shot value is kept as `single_shot_res_<role>`.  A
track whose single-shot failed (-1 placeholder) still gets res_/err_ from its
bootstrap rows.

This tool re-uses those scripts' OWN functions wherever possible (imported
from core/), so what it draws is what the pipeline computes - with one
unavoidable caveat: the mixture fit is unseeded in the pipeline, so a replay
reproduces the procedure, not bit-identical numbers - typically ~0.1 ps apart
for tracks with thousands of events, but many ps where the mixture is unstable
(a few hundred events); the figure prints both so the difference is visible.
The anatomy replays the event set of a step-12 run with the given
--neighbor-cut (default none) - pass the same options step 12 used.

Subcommands
-----------
  anatomy      one track: every stage drawn with the fit on top of the data:
               the TWC iterations, the pairwise distributions with the mixture
               and its FWHM, the 3-board solve, and the 200 bootstrap values
               with the Gaussian step 13 fits to them.  Reads the step-10 track
               file and (optionally) its step-12 boot file for comparison.
  stats        all tracks of one or more combos: bootstrap error vs number of
               events, pull of the single-shot against the bootstrap
               distribution, completeness, and (from the condor logs) how many
               attempts each track needed - is the machinery healthy?
  consistency  the same board's resolution from different combos (different
               partner boards) per pixel, and the same pixel PAIR's width from
               different combos - does the answer depend on who the partners
               are?
  compare      two step-12 output sets track by track (ratio of single-shot
               resolutions and pair widths vs event count) - what did a
               configuration change (gate, mixture convergence) do?

USAGE
-----
  python utils/bootstrap_diagnostics.py anatomy -f <combo>/time/track_X.parquet \\
      [--boot-file bootstrap_<run>/<combo>/track_X_boot.parquet] [--replay-boot N] -o <outdir>
  python utils/bootstrap_diagnostics.py stats -d bootstrap_<run> --time-base <EOS .../tracks_<run>> \\
      [--log-dir condor_logs/bootstrap/<tag>] -o <outdir>
  python utils/bootstrap_diagnostics.py consistency -d bootstrap_<run> --time-base <EOS .../tracks_<run>> \\
      [--diag-dir <track_diagnostics parquet dir>] -o <outdir>
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap
from glob import glob
from itertools import combinations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kstest, kstwobign, norm
from sklearn.mixture import GaussianMixture

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))
import bootstrap as bs                       # noqa: E402  (the pipeline's own step-12 code)
import fit_bootstrap_results as fbr          # noqa: E402  (the pipeline's own step-13 code)
from telescope_diagnostics import GRID, pixel_map, set_output_options, add_output_arguments   # noqa: E402
from track_diagnostics import finish, parse_pixels, rc, robust_sigma  # noqa: E402

KS_PMIN = bs.KS_PMIN_DEFAULT    # bootstrap.py: accept a pair fit if KS p-value >= KS_PMIN or KS distance <= KS_DMAX
KS_DMAX = bs.KS_DMAX_DEFAULT
PAIR_MIN_EVENTS = 500           # pair-width consistency test: tracks with fewer surviving events are left out
FWHM_TO_SIGMA = 1.0 / 2.355     # bootstrap.py: fit_sigmas[pair] = fwhm / 2.355
NL = "\n"                       # newline inside multi-line tick labels / titles


def die(msg):
    sys.exit("bootstrap_diagnostics: " + msg)


def combo_of(path):
    """bootstrap_<run>/<combo>/track_X_boot.parquet -> <combo>"""
    return os.path.basename(os.path.dirname(os.path.abspath(path)))


# --------------------------------------------------------------------------
# faithful replays of bootstrap.py internals that also hand back what a plot needs
# --------------------------------------------------------------------------

def twc_iterations(df, roles):
    """bootstrap.apply_timewalk_correction, keeping each iteration's deltas and fit.
    Returns (iters, toas): iters = [ {role: (delta_before_fit, coeff)} per iteration ],
    toas = corrected TOAs (identical to what apply_timewalk_correction returns)."""
    tots = {r: df["tot_%s" % r].to_numpy(float) for r in roles}
    toas = {r: df["toa_%s" % r].to_numpy(float).copy() for r in roles}
    iters = []
    for _ in range(2):
        deltas = {}
        for r in roles:
            others = [toas[o] for o in roles if o != r]
            deltas[r] = (0.5 * sum(others)) - toas[r]
        step = {}
        for r in roles:
            coeff = np.polyfit(tots[r], deltas[r], 2)
            step[r] = (deltas[r].copy(), coeff)
            toas[r] += np.poly1d(coeff)(tots[r])
        iters.append(step)
    return iters, toas, tots


def fit_gmm_verbose(data):
    """bootstrap.fit_gmm_and_get_fwhm, keeping every candidate mixture.
    Returns dict(best=(n_comp, ks, fwhm, gmm) or None, candidates=[...])."""
    data_reshaped = data.reshape(-1, 1)
    data_sorted = np.sort(data)
    n_events = len(data)
    components_to_try = [1, 2, 3] if n_events < 1500 else [3]
    best, cands = None, []
    best_ks = 1.0
    for n_comp in components_to_try:
        try:
            gmm = GaussianMixture(n_components=n_comp, n_init=3, tol=bs.GMM_TOL, max_iter=bs.GMM_MAX_ITER).fit(data_reshaped)
            ks_score, ks_p = kstest(data_sorted, lambda x: bs.calculate_gmm_cdf(x, gmm.weights_, gmm.means_, gmm.covariances_))
            x_range = np.linspace(data.min(), data.max(), 1000).reshape(-1, 1)
            pdf_range = np.exp(gmm.score_samples(x_range))
            peak_val = np.max(pdf_range)
            half = np.where(pdf_range >= peak_val / 2.0)[0]
            if len(half) <= 1:
                cands.append((n_comp, ks_score, np.nan, gmm, ks_p))
                continue
            fwhm = float(x_range[half[-1], 0] - x_range[half[0], 0])
            cands.append((n_comp, ks_score, fwhm, gmm, ks_p))
            if ks_score < best_ks:
                best_ks, best = ks_score, (n_comp, ks_score, fwhm, gmm, ks_p)
        except Exception:
            continue
    return dict(best=best, candidates=cands)


def single_shot_replay(df, roles, pmin=KS_PMIN, dmax=KS_DMAX):
    """One pass of bootstrap.run_sample_analysis with everything kept, plus the
    verdict step 12 would give this attempt: accepted, or rejected because a
    pair's fit fails the KS gate (p-value below pmin AND distance above dmax),
    has no usable FWHM, or the 3-board solve is imaginary (run_sample_analysis
    returns None in those cases and the caller retries)."""
    iters, toas, tots = twc_iterations(df, roles)
    pairs = {"%s-%s" % (a, b): toas[a] - toas[b] for a, b in combinations(roles, 2)}
    fits = {p: fit_gmm_verbose(v) for p, v in pairs.items()}
    sig, reasons = {}, []
    for p, f in fits.items():
        if f["best"] is None or not np.isfinite(f["best"][2]) or f["best"][2] == 0:
            sig[p] = np.nan
            reasons.append("pair %s: no usable FWHM" % p)
            continue
        sig[p] = f["best"][2] * FWHM_TO_SIGMA
        if f["best"][4] < pmin and f["best"][1] > dmax:
            reasons.append("pair %s: KS p %.2e < %.0e and D %.4f > %.2f" % (p, f["best"][4], pmin, f["best"][1], dmax))
    if all(np.isfinite(v) for v in sig.values()):
        res = bs.calculate_resolution_from_fit(sig, roles)
        for r, v in res.items():
            if v <= 0:
                reasons.append("σ_%s imaginary/zero (3-board solve)" % r)
    else:
        res = {r: np.nan for r in roles}
    return dict(iters=iters, toas=toas, tots=tots, pairs=pairs, fits=fits, sig=sig, res=res,
                accepted=(len(reasons) == 0), reasons=reasons)


def load_boot(path):
    """-> (roles, bootstrap rows, single-shot row or None). Columns 'pair_<a>-<b>' (pair widths,
    written by bootstrap.py since the KS p-value gate) are kept in the frames but are not roles."""
    d = pd.read_parquet(path)
    cols = [c for c in d.columns if c != "is_bootstrap"]
    roles = sorted(c for c in cols if not c.startswith("pair_") and not c.startswith("ksp"))
    b = d.loc[d.is_bootstrap == True, cols]
    s = d.loc[d.is_bootstrap == False, cols]
    return roles, b, (s.iloc[0] if len(s) else None)


# --------------------------------------------------------------------------
# anatomy
# --------------------------------------------------------------------------

def fig_anatomy_twc(rep, roles, out, title):
    fig, axs = plt.subplots(len(roles), 3, figsize=(16.5, 4.3 * len(roles) + 1.6), squeeze=False)
    for i, r in enumerate(roles):
        tot = rep["tots"][r] / 1e3
        lo, hi = np.percentile(tot, [0.5, 99.5])
        g = np.linspace(lo, hi, 200)
        for k in range(2):
            ax = axs[i, k]
            d, c = rep["iters"][k][r]
            ax.hexbin(tot, d, gridsize=70, cmap="Greys", mincnt=1, bins="log")
            ax.plot(g, np.poly1d(c)(g * 1e3), color=rc(r), lw=2.0,
                    label="fit: %.3g·TOT² %+.3g·TOT %+.1f  (TOT in ps)" % (c[0], c[1], c[2]))
            ax.set_xlim(lo, hi)
            yl = np.percentile(d, [0.5, 99.5])
            ax.set_ylim(yl[0] - 0.1 * (yl[1] - yl[0]), yl[1] + 0.1 * (yl[1] - yl[0]))
            ax.set_xlabel("TOT of %s [ns]" % r)
            ax.set_ylabel("½·(TOA others) − TOA(%s) [ps]\n%s" % (r, "before iteration 1" if k == 0 else "after iteration 1 (residual)"))
            ax.set_title("%s - iteration %d: what is fitted and the fit" % (r, k + 1), fontsize=10)
            ax.legend(fontsize=7.5, loc="upper right")
            ax.grid(alpha=0.3)
        ax = axs[i, 2]
        # what the correction leaves: the iteration-2 fit residual (mean 0 by construction; a
        # constant is irrelevant to any pair width, the SHAPE vs TOT is what matters)
        d2, c2 = rep["iters"][1][r]
        resid = d2 - np.poly1d(c2)(rep["tots"][r])
        ax.hexbin(tot, resid, gridsize=70, cmap="Greys", mincnt=1, bins="log")
        # profile: median in TOT bins
        edges = np.linspace(lo, hi, 21)
        idx = np.digitize(tot, edges)
        cx, cy = [], []
        for b in range(1, len(edges)):
            m = idx == b
            if m.sum() >= 30:
                cx.append(0.5 * (edges[b - 1] + edges[b]))
                cy.append(np.median(resid[m]))
        ax.plot(cx, cy, "o-", color=rc(r), ms=4, lw=1.4, label="median per TOT bin")
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xlim(lo, hi)
        yl = np.percentile(resid, [0.5, 99.5])
        ax.set_ylim(yl[0] - 0.1 * (yl[1] - yl[0]), yl[1] + 0.1 * (yl[1] - yl[0]))
        ax.set_xlabel("TOT of %s [ns]" % r)
        ax.set_ylabel("iteration-2 fit residual [ps]")
        ax.set_title("%s - what the correction leaves: flat in TOT?" % r, fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=11, y=0.995)
    cap = ("The time-walk correction exactly as step 12 applies it, one row per board. Grey: 2D density of the "
           "events (log colour scale). LEFT: the quantity step 12 fits - half the sum of the OTHER two boards' "
           "TOA minus this board's TOA - against this board's TOT, with the 2nd-order polynomial of iteration 1 "
           "on top (numpy polyfit, all events, no weights, no outlier rejection); it is ADDED to this board's TOA. "
           "MIDDLE: the same quantity recomputed with the TOAs corrected once, and iteration 2's polynomial. "
           "RIGHT: the residual of that second fit with its median per TOT bin - flat means the time walk is "
           "gone; a bend at the TOT extremes is what a 2nd-order polynomial cannot follow (and is where the fit "
           "is driven by few events). It is centred on 0 by construction of the fit; the recomputed offset "
           "between a board and its partners' mean is NOT zero after two iterations (each coupled iteration halves "
           "and flips it, a quarter survives) but a constant enters no pair width. Because each board is corrected "
           "against the mean of its two partners, their jitter is in the vertical spread here and their own "
           "time walk leaks into the fit; the horizontal range shown is the 0.5-99.5 percentile of TOT.")
    return finish(fig, out, cap, wspace=0.30, hspace=0.42, top=0.94)


def fig_anatomy_pairs(rep, roles, out, title, boot_ss=None, boot=None, proxy=None):
    pairs = list(rep["pairs"].keys())
    fig, axs = plt.subplots(1, len(pairs) + 1, figsize=(5.4 * (len(pairs) + 1) + 0.6, 6.0), squeeze=False)
    axs = axs.ravel()
    for ax, p in zip(axs, pairs):
        v = rep["pairs"][p]
        f = rep["fits"][p]
        lo, hi = np.percentile(v, [0.2, 99.8])
        bins = np.linspace(lo, hi, 80)
        hvals, _, _ = ax.hist(v, bins=bins, density=True, color="#bbbbbb", label="data: TOA(%s) − TOA(%s), after TWC (n=%d)" % (p.split("-")[0], p.split("-")[1], len(v)))
        x = np.linspace(v.min(), v.max(), 1000)
        ymax = float(np.max(hvals)) if len(hvals) else 1.0
        for n_comp, ks, fwhm, gmm, ksp in f["candidates"]:
            pdf = np.exp(gmm.score_samples(x.reshape(-1, 1)))
            is_best = f["best"] is not None and n_comp == f["best"][0]
            if is_best:
                ymax = max(ymax, float(pdf.max()))
            lab = ("%d-comp. mixture: KS D %.4f (p %.2f), FWHM %.1f ps%s" % (n_comp, ks, ksp, fwhm, "  ← kept" if is_best else "")
                   if np.isfinite(fwhm) else "%d-comp. mixture: KS D %.4f, no usable half-max width (spike) - excluded" % (n_comp, ks))
            ax.plot(x, pdf, color="#c0392b" if is_best else "#888888", lw=2.2 if is_best else 1.0,
                    ls="-" if is_best else "--", alpha=1.0 if is_best else 0.8, label=lab)
        ax.set_ylim(0, 1.35 * ymax)   # a spiked, non-kept candidate must not flatten the data
        if f["best"] is not None:
            n_comp, ks, fwhm, gmm, ksp = f["best"]
            pdf = np.exp(gmm.score_samples(x.reshape(-1, 1)))
            peak = pdf.max()
            half = np.where(pdf >= peak / 2)[0]
            ax.hlines(peak / 2, x[half[0]], x[half[-1]], color="k", lw=1.6)
            ax.text(0.5 * (x[half[0]] + x[half[-1]]), peak / 2, "  FWHM = %.1f ps → σ = %.1f ps" % (fwhm, fwhm * FWHM_TO_SIGMA),
                    fontsize=8.5, va="bottom", ha="center")
            ok = (ksp >= KS_PMIN) or (ks <= KS_DMAX)
            ax.set_title("%s\nKS p %.3f (D %.4f): %s" % (p, ksp, ks, "accepted (p ≥ %.0e or D ≤ %.2f)" % (KS_PMIN, KS_DMAX) if ok
                                                             else "REJECTED → step 12 would refit"), fontsize=9.5)
        ax.set_xlim(lo, hi)
        ax.set_xlabel("ΔTOA [ps]")
        ax.set_ylabel("density")
        ax.legend(fontsize=7.2, loc="upper left")
        ax.grid(alpha=0.3)
    # 3-board solve summary panel
    ax = axs[-1]
    ax.axis("off")
    lines = ["3-board solve  σ_a² = ½(s_ab² + s_ac² − s_bc²)", ""]
    for p, s in rep["sig"].items():
        lines.append("s_%s = %s" % (p, ("%.2f ps" % s) if np.isfinite(s) else "no usable FWHM"))
    lines.append("")
    if rep["accepted"]:
        lines.append("this replay attempt: ACCEPTED by step 12's rules")
    else:
        lines.append("this replay attempt: REJECTED by step 12's rules:")
        lines += ["   " + x for x in rep["reasons"]]
        lines.append("   (step 12 would refit with the p threshold halved,")
        lines.append("    down to 1e-6 on the single-shot; then -1)")
    lines.append("")
    for r in roles:
        v = rep["res"].get(r, np.nan)
        line = "σ_%s = %s  (this replay)" % (r, ("%.2f ps" % v) if np.isfinite(v) and v > 0 else "imaginary / not solved")
        if boot_ss is not None:
            line += ("\n      step-12 single-shot in the boot file: %.2f ps" % boot_ss[r]) if boot_ss[r] > 0 else \
                    "\n      step-12 single-shot: -1 = FAILED all attempts (placeholder)"
        if boot is not None:
            bv = boot[r][boot[r] > 0]
            line += "\n      step-12 bootstrap: median %.2f, std %.2f ps (%d accepted)" % (bv.median(), bv.std(), len(bv))
        if proxy is not None and r in proxy:
            line += "\n      IQR proxy (track_diagnostics): %.2f ps" % proxy[r]
        lines.append(line)
    ax.text(-0.05, 1.0, "\n".join(lines), va="top", ha="left", fontsize=8.8, family="DejaVu Sans Mono", transform=ax.transAxes)
    fig.suptitle(title, fontsize=11, y=0.995)
    cap = ("The second stage of step 12, exactly as coded: for each pair of boards, the histogram of the TWC-"
           "corrected TOA difference (grey) and the Gaussian mixtures fitted to it (sklearn GaussianMixture, "
           "n_init=3; 1, 2 and 3 components are tried when the track has fewer than 1500 events, only 3 "
           "otherwise). Each candidate's KS distance between the mixture CDF and the data is printed; the "
           "smallest KS distance is KEPT (red) and accepted if its KS p-value is at least 1e-3 OR its distance at "
           "most 0.03 - otherwise step 12 refits (the mixture is unseeded, so a refit differs) with the p threshold "
           "halved per attempt down to 1e-6; the single-shot then gives up with a -1 placeholder, which also "
           "happens when the 3-board solve is imaginary every time. The mixture is fitted to convergence (EM "
           "tolerance 1e-6, up to 2000 iterations, as bootstrap.py now does; sklearn's default tolerance stopped "
           "after ~10 iterations and left the width ~12 % too narrow). The pair width is the FULL WIDTH AT HALF "
           "MAXIMUM of the kept mixture (black "
           "bar, read off a 1000-point grid of its density) divided by 2.355 - i.e. a Gaussian-equivalent core "
           "width that ignores tails. RIGHT: the verdict step 12 would give this attempt, the three pair widths "
           "solved for the three boards, next to what the actual step-12 job wrote for this track (single-shot "
           "and the bootstrap median ± std of its accepted resamples) and the IQR-based proxy of "
           "track_diagnostics. Replay and job differ by the unseeded mixture fit only: ~0.1 ps for tracks with "
           "thousands of events, many ps where the mixture is unstable (a few hundred events) - both numbers "
           "are printed so the difference is visible. A track whose single-shot is -1 still gets res_/err_ in "
           "step 13 from its accepted bootstrap resamples, which can be far from physical: check them here.")
    return finish(fig, out, cap, wspace=0.28, top=0.86)


def fig_anatomy_bootstrap(boot, ss, roles, out, title, replay=None, sigma_cut=2.5):
    fig, axs = plt.subplots(1, len(roles), figsize=(5.4 * len(roles) + 0.6, 5.6), squeeze=False)
    axs = axs.ravel()
    for ax, r in zip(axs, roles):
        v = boot[r].to_numpy(float) if boot is not None else np.array([])
        v = v[np.isfinite(v) & (v > 0)]
        if v.size < 5:
            if replay is not None and r in replay and len(replay[r]) >= 5:
                v = np.asarray(replay[r], float)   # no usable job values: show the replay alone
            else:
                ax.text(0.5, 0.5, "bootstrap phase FAILED in the job\n(-1 placeholder, no values to show)",
                        ha="center", va="center", transform=ax.transAxes, fontsize=9.5)
                ax.set_title(r, fontsize=10)
                continue
        st = fbr.perform_robust_unbinned_fit(pd.Series(v), sigma_cut)
        lo, hi = np.percentile(v, [0, 100])
        pad = 0.15 * (hi - lo + 1e-9)
        bins = np.linspace(lo - pad, hi + pad, 40)
        ax.hist(v, bins=bins, density=True, color=rc(r), alpha=0.35, label="step-12 bootstrap values (n=%d)" % len(v))
        if replay is not None and r in replay:
            ax.hist(replay[r], bins=bins, density=True, histtype="step", color="k", lw=1.4, label="replayed here (n=%d)" % len(replay[r]))
        x = np.linspace(bins[0], bins[-1], 400)
        ax.plot(x, norm.pdf(x, st["mu"], st["sigma"]), color="k", lw=2.0,
                label="step-13 fit: μ=%.2f (→ res), σ=%.2f (→ err)\nχ²/ndf %.2f, valid=%d" % (st["mu"], st["sigma"], st["chi2_red"], st["valid"]))
        med = np.median(v)
        q75, q25 = np.percentile(v, [75, 25])
        s0 = (q75 - q25) / 1.349
        ax.axvspan(med - sigma_cut * s0, med + sigma_cut * s0, color="#dddddd", alpha=0.35, lw=0, label="fit window: median ± %.1f·(IQR/1.349)" % sigma_cut)
        if ss is not None:
            ax.axvline(ss[r], color="k", lw=1.6, ls="--", label="single-shot (full sample): %.2f ps" % ss[r])
        ax.set_xlabel("σ_%s per bootstrap resample [ps]" % r)
        ax.set_ylabel("density")
        ax.set_title(r, fontsize=10)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.6)   # headroom for the legend
        ax.legend(fontsize=7.5, loc="upper left")
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=11, y=0.995)
    cap = ("The third stage: each bootstrap resample (same size as the track, drawn with replacement) goes "
           "through the same TWC + mixture + 3-board solve, giving one σ per board per resample; shown are the "
           "values the step-12 job wrote (filled histogram) and, if requested, the same procedure replayed here "
           "(black steps) - agreement between the two says the replay is faithful up to the unseeded mixture "
           "fit. Black curve: what step 13 does with them - an unbinned Gaussian likelihood fit (iminuit) seeded "
           "by the median and IQR/1.349 and restricted to the grey window of ± --sigma_cut of that width; its "
           "mean becomes the track's `res_<role>` and its width the track's `err_<role>` in the resolution "
           "table. Dashed: the single-shot value from the full sample, kept as `single_shot_res_<role>`; any "
           "later fit over pixels uses `res_<role>` (the bootstrap μ), not this. Healthy: a roughly Gaussian bump around the single-shot with a "
           "width of a few % of the value; a skewed or double-peaked histogram means the mixture fit hops "
           "between solutions on resamples.")
    return finish(fig, out, cap, wspace=0.26, top=0.88)


def cmd_anatomy(a):
    df = pd.read_parquet(a.file)
    stem = os.path.splitext(os.path.basename(a.file))[0]
    roles = sorted(r for r in bs.ALL_ROLES if "toa_%s" % r in df.columns)
    if len(roles) != 3:
        die("expected 3 roles in %s, found %s" % (a.file, roles))
    df = bs.apply_neighbor_cut(df, a.neighbor_cut, a.neighbor_logic)
    print("anatomy: %s | roles %s | %d events" % (stem, roles, len(df)))
    rep = single_shot_replay(df, roles)
    boot, ss = None, None
    if a.boot_file:
        if not os.path.isfile(a.boot_file):
            die("--boot-file not found: %s" % a.boot_file)
        broles, boot, ss = load_boot(a.boot_file)
    elif a.replay_boot:
        print("  note: no --boot-file; the bootstrap figure will show the replay alone")
    proxy = None
    if a.diag_parquet and os.path.isfile(a.diag_parquet):
        dg = pd.read_parquet(a.diag_parquet)
        t = dg[(dg.section == "twc_full") & (dg.key == "sig_role") & (dg.track == stem)]
        if len(t):
            proxy = dict(zip(t.role, t.value))
    folder = ("%s__%s" % (a.tag, stem)) if a.tag else stem
    outdir = os.path.join(a.outdir, folder)
    os.makedirs(outdir, exist_ok=True)
    title = "%s%s  (%d events after step 10)" % (("[%s]  " % a.tag) if a.tag else "", stem, len(df))
    written = [fig_anatomy_twc(rep, roles, os.path.join(outdir, "anatomy_twc.png"), title),
               fig_anatomy_pairs(rep, roles, os.path.join(outdir, "anatomy_pairs.png"), title, ss, boot, proxy)]
    replay = None
    if a.replay_boot and a.replay_boot > 0:
        # same resampling as bootstrap.py --reproducible: SeedSequence(42) -> phases -> attempts
        root = np.random.SeedSequence(42)
        phase_seqs = root.spawn(2)
        att = phase_seqs[1].spawn(max(a.replay_boot * 3, 30))
        vals = {r: [] for r in roles}
        n_ok, k = 0, 0
        while n_ok < a.replay_boot and k < len(att):
            rng = np.random.default_rng(att[k])
            k += 1
            sample = df.sample(frac=1.0, replace=True, random_state=rng)
            res, ok = bs.run_sample_analysis(sample, roles, threshold=KS_PMIN, is_boot=True)
            if ok:
                for r in roles:
                    vals[r].append(res[r])
                n_ok += 1
        replay = {r: np.array(v) for r, v in vals.items()}
        print("  replayed %d/%d bootstrap resamples (%d attempts)" % (n_ok, a.replay_boot, k))
    if boot is not None or replay is not None:
        written.append(fig_anatomy_bootstrap(boot, ss, roles, os.path.join(outdir, "anatomy_bootstrap.png"), title, replay, a.sigma_cut))
    for w in written:
        print("  wrote %s" % w)


# --------------------------------------------------------------------------
# stats: all tracks
# --------------------------------------------------------------------------

def collect_boot(bdir, time_base=None, log_dir=None):
    """One row per track: combo, stem, roles' single-shot / boot median / boot std / n_boot, nevt, attempts."""
    rows = []
    combos = sorted(d for d in os.listdir(bdir) if os.path.isdir(os.path.join(bdir, d)) and glob(os.path.join(bdir, d, "*_boot.parquet")))
    if not combos:
        if glob(os.path.join(bdir, "*_boot.parquet")):
            combos = [""]
        else:
            die("no *_boot.parquet under %s" % bdir)
    logs = {}
    if log_dir:
        for f in glob(os.path.join(log_dir, "**", "*.stdout"), recursive=True):
            txt = open(f, errors="ignore").read()
            m = re.search(r"Processing (track_\S+) \|", txt)
            if not m:
                continue
            # single-shot phase lines precede 'Starting Phase: Bootstrap'; count both phases separately
            k = txt.find("Starting Phase: Bootstrap")
            ss_txt, bt_txt = (txt[:k], txt[k:]) if k >= 0 else (txt, "")
            logs[m.group(1)] = dict(rej=len(re.findall(r"Rejecting pair", bt_txt)),
                                    phys=len(re.findall(r"Physics Failure", bt_txt)),
                                    single_rej=len(re.findall(r"Rejecting pair", ss_txt)) + len(re.findall(r"Physics Failure", ss_txt)),
                                    relax=len(re.findall(r"\[Relaxation\]", bt_txt)),
                                    failed=("FAILED" in txt), short=("Only reached" in txt))
    for combo in combos:
        for f in sorted(glob(os.path.join(bdir, combo, "*_boot.parquet"))):
            stem = os.path.basename(f).replace("_boot.parquet", "")
            roles, b, ss = load_boot(f)
            row = dict(combo=combo, track=stem, n_boot=len(b))
            for r in roles:
                v = b[r].to_numpy(float)
                v = v[np.isfinite(v) & (v > 0)]
                row["ss_%s" % r] = float(ss[r]) if ss is not None else np.nan
                row["med_%s" % r] = float(np.median(v)) if v.size else np.nan
                row["std_%s" % r] = float(np.std(v, ddof=1)) if v.size > 1 else np.nan
                row["skew_%s" % r] = float(pd.Series(v).skew()) if v.size > 2 else np.nan
            for pc in [c for c in b.columns if c.startswith("pair_")]:
                v = b[pc].to_numpy(float)
                v = v[np.isfinite(v) & (v > 0)]
                row["ss%s" % pc] = float(ss[pc]) if ss is not None else np.nan       # sspair_<a>-<b>
                row["med%s" % pc] = float(np.median(v)) if v.size else np.nan        # medpair_<a>-<b>
            for r, (rr, cc) in parse_pixels(stem).items():
                row["row_%s" % r] = rr
                row["col_%s" % r] = cc
            if time_base:
                tf = os.path.join(time_base, combo, "time", stem + ".parquet")
                row["nevt"] = pq.read_metadata(tf).num_rows if os.path.isfile(tf) else np.nan
            row.update(logs.get(stem, {}))
            rows.append(row)
    return pd.DataFrame(rows)


def roles_in(T):
    return sorted(c[3:] for c in T.columns if c.startswith("ss_"))


def fig_stats_errors(T, out, label):
    roles = roles_in(T)
    have_n = "nevt" in T.columns and T["nevt"].notna().any()
    fig, axs = plt.subplots(2, 3, figsize=(17.5, 11.6))
    ax = axs[0, 0]
    for r in roles:
        m = T["std_%s" % r].notna() & (T["ss_%s" % r] > 0)
        if have_n:
            ax.scatter(T.loc[m, "nevt"], T.loc[m, "std_%s" % r], s=6, alpha=0.45, color=rc(r), label=r)
    if have_n:
        n = np.geomspace(max(T["nevt"].min(), 50), T["nevt"].max(), 50)
        m = T["std_%s" % roles[0]].notna()
        k = np.nanmedian(T.loc[m, "std_%s" % roles[0]] * np.sqrt(T.loc[m, "nevt"]))
        ax.plot(n, k / np.sqrt(n), "k--", lw=1.4, label="∝ 1/√N (through the %s median)" % roles[0])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("events in the track after step 10")
    if not have_n:
        ax.text(0.5, 0.5, "no --time-base given:\nevent counts unavailable", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel("bootstrap std of σ [ps]\n(≈ 1.05 × err_<role>; err is the 2.5σ-clipped fit width)")
    ax.set_title("bootstrap error vs statistics", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax = axs[0, 1]
    for r in roles:
        rel = T["std_%s" % r] / T["med_%s" % r]
        rel = rel[np.isfinite(rel)]
        ax.hist(rel, bins=np.linspace(0, min(0.3, np.nanpercentile(rel, 99.5) * 1.2), 60), histtype="step", lw=1.6, color=rc(r),
                label="%s: median %.1f%%" % (r, 100 * np.median(rel)))
    ax.set_xlabel("relative bootstrap error  std / median")
    ax.set_ylabel("tracks")
    ax.set_title("relative error per track: bootstrap std ÷ bootstrap median", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axs[0, 2]
    for r in roles:
        pull = (T["ss_%s" % r] - T["med_%s" % r]) / T["std_%s" % r]
        pull = pull[np.isfinite(pull) & (T["ss_%s" % r] > 0)]
        ax.hist(pull, bins=np.linspace(-5, 5, 61), histtype="step", lw=1.6, color=rc(r),
                label="%s: mean %+.2f, robust width %.2f" % (r, pull.mean(), robust_sigma(pull)))
    ax.set_xlabel("(single-shot − bootstrap median) / bootstrap std")
    ax.set_ylabel("tracks")
    ax.set_title("single-shot inside its own bootstrap distribution\n(one entry per track and board)", fontsize=10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    ax = axs[1, 0]
    for r in roles:
        sk = T["skew_%s" % r].dropna()
        ax.hist(sk, bins=np.linspace(-2, 2, 61), histtype="step", lw=1.6, color=rc(r), label="%s: median %+.2f" % (r, sk.median()))
    ax.set_xlabel("skewness of the bootstrap values of a track")
    ax.set_ylabel("tracks")
    ax.set_title("shape of each track's bootstrap distribution\n(one entry per track and board)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axs[1, 1]
    nb = T["n_boot"]
    step = max(1, int(nb.max()) // 50)
    ax.hist(nb, bins=np.arange(-0.5, nb.max() + step + 0.5, step), color="#777777")
    ax.set_yscale("log")
    ax.set_xlabel("bootstrap rows written per track")
    ax.set_ylabel("tracks")
    ax.set_title("completeness: %d of %d tracks reached the target\n%d track(s) with a -1 single-shot placeholder"
                 % ((nb == nb.max()).sum(), len(nb), int((T[["ss_%s" % r for r in roles]] < 0).any(axis=1).sum())), fontsize=10)
    ax.grid(alpha=0.3)
    ax = axs[1, 2]
    if "rej" in T.columns and T["rej"].notna().any():
        rej = T["rej"].fillna(0).clip(upper=300)
        phys = T["phys"].fillna(0).clip(upper=300)
        step = max(1, int(max(rej.max(), phys.max()) // 60))
        bins = np.arange(-0.5, 300 + step + 0.5, step)
        ax.hist([rej, phys], bins=bins, stacked=True, color=["#777777", "#c0392b"],
                label=["mixture KS above threshold", "imaginary 3-board solve"])
        ax.set_yscale("log")
        ax.set_xlabel("rejected bootstrap resamples per track (from the job logs; ≥300 in the last bin)")
        ax.set_ylabel("tracks")
        att = rej + phys
        ax.set_title("rejected resamples: median %.0f, 90%% of tracks ≤ %.0f\n%d track(s) relaxed the KS threshold, %d needed >1 single-shot attempt"
                     % (att.median(), att.quantile(0.9), int((T["relax"].fillna(0) > 0).sum()),
                        int((T["single_rej"].fillna(0) > 0).sum())), fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "no --log-dir given:\nrejection counts unavailable", ha="center", va="center", transform=ax.transAxes)
    fig.suptitle("%s - statistical behaviour of the bootstrap over %d tracks" % (label, len(T)), fontsize=11, y=0.995)
    cap = ("Is the resampling machinery behaving? Every panel has one entry per track (and per board where "
           "coloured). TOP-LEFT: the bootstrap standard deviation of each track's σ against the track's event "
           "count; a healthy estimator follows the dashed 1/√N line. TOP-MIDDLE: the same error divided by the "
           "track's bootstrap median, i.e. the fractional statistical error of one pixel's resolution - it is a "
           "few % rather than the 1/√(2N) < 1 % of a plain Gaussian width because the 3-board solve amplifies the "
           "pair-width errors by about s_pair/σ ≈ 1.5 and the FWHM of a mixture is a less efficient width "
           "estimator than a Gaussian σ; the tail beyond ~10 % is the low-statistics tracks (their per-pixel "
           "numbers carry little weight in any board average). TOP-RIGHT: where the full-sample (single-shot) "
           "value falls inside "
           "the bootstrap distribution, in units of its std - expected centred at 0 (a shift is a bias of the "
           "resampled estimator: mixture + FWHM behave differently on samples with duplicated events). Its "
           "expected width is small, ~0.1 for a symmetric distribution of 200 values (the statistical noise of "
           "the median alone), NOT 1: a width approaching 1 means the bootstrap bias varies from track to track "
           "(mixture hopping, rejection truncation). BOTTOM-LEFT: skewness of each track's bootstrap values - "
           "the third standardised moment (pandas' bias-corrected sample skewness G1): 0 = symmetric, negative "
           "= a tail towards small σ, as when the mixture occasionally locks onto a narrow core. BOTTOM-MIDDLE: "
           "how many bootstrap rows each "
           "track actually has (target = step 12's -n; fewer means the iteration limit was hit; a -1 single-shot "
           "means all 20 full-sample attempts failed, by KS or by an imaginary solve). BOTTOM-RIGHT, from the "
           "condor logs: how many bootstrap resamples were thrown away per track because a pair's mixture failed "
           "the KS threshold (grey) or the 3-board solve went imaginary (red) - many rejections mean the KS "
           "threshold is tight for that track's statistics; the title also counts tracks whose single-shot "
           "needed more than one attempt.")
    return finish(fig, out, cap, wspace=0.32, hspace=0.50, top=0.93)


def fig_stats_ks_gate(T, out, label):
    """Rejected-resample fraction vs N, against what a fixed KS-distance gate does to perfect fits."""
    if "nevt" not in T.columns or "rej" not in T.columns or not T["rej"].notna().any():
        return None
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.6))
    n = T["nevt"].to_numpy(float)
    ks = T["rej"].fillna(0).to_numpy(float)
    ph = T["phys"].fillna(0).to_numpy(float)
    nb = T["n_boot"].to_numpy(float)
    with np.errstate(all="ignore"):
        f_ks = ks / (ks + ph + nb)
        f_ph = ph / (ks + ph + nb)
    a1.scatter(n, f_ks, s=8, alpha=0.6, color="#555555", label="rejected by the KS gate (mixture fit)")
    a1.scatter(n, f_ph, s=8, alpha=0.6, color="#c0392b", label="rejected: imaginary 3-board solve")
    ng = np.geomspace(max(50, np.nanmin(n) * 0.8), np.nanmax(n) * 1.2, 200)
    a1.plot(ng, kstwobign.sf(0.03 * np.sqrt(ng)), "k--", lw=1.4,
            label="perfect fit rejected by the OLD rule D ≤ 0.03: P(K > 0.03·√N)")
    a1.axhline(1e-3, color="k", lw=1.0, ls=":", label="perfect fit rejected by the new gate (p ≥ 1e-3 or D ≤ 0.03): ≤ 0.1 % at any N")
    a1.set_xscale("log")
    a1.set_ylim(-0.02, 1.02)
    a1.set_xlabel("events in the track after step 10")
    a1.set_ylabel("fraction of bootstrap attempts rejected")
    a1.set_title("rejected attempts per track (from the job logs)", fontsize=10)
    a1.legend(fontsize=7.5, loc="upper right")
    a1.grid(alpha=0.3, which="both")
    a2.plot(ng, 0.83 / np.sqrt(ng), color="#1f77b4", lw=2.0, label="median KS distance of a PERFECT fit: 0.83/√N")
    a2.plot(ng, 1.36 / np.sqrt(ng), color="#1f77b4", lw=1.0, ls="--", label="95 % quantile of a perfect fit: 1.36/√N")
    a2.axhline(0.03, color="k", lw=1.4, ls="--", label="old acceptance: D ≤ 0.03")
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlabel("events in the track after step 10")
    a2.set_ylabel("KS distance D")
    a2.set_title("why a fixed KS distance is not a quality gate", fontsize=10)
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3, which="both")
    fig.suptitle("%s - KS acceptance vs statistics" % label, fontsize=11, y=0.995)
    cap = ("Plain words: step 12 draws 200 resamples per track and keeps a resample only if the mixture fit to each "
           "pairwise TOA-difference distribution passes a goodness-of-fit gate; a rejected resample is replaced by "
           "a new draw. LEFT: for each track, the fraction of draws that were thrown away - because a fit failed "
           "the gate (grey) or because the 3-board solve came out imaginary (red; a physics failure, not the gate), "
           "against the number of events in the track. Dashed: what the OLD gate 'KS distance ≤ 0.03' does to a "
           "PERFECT fit: the Kolmogorov probability that a perfect model's distance exceeds 0.03 at that N - it "
           "rejects most good fits below ~800 events (so the kept resamples were the lucky ones, a biased sample) "
           "and nothing above ~3000 (so it tested nothing there). Dotted: the gate now used (p-value ≥ 1e-3, or "
           "D ≤ 0.03) rejects at most 0.1 % of perfect fits at any N. Reading rule: grey points that follow the "
           "dashed curve = this run used the old rule and its low-N tracks are biased; grey points near zero = "
           "the gate only removed broken fits. RIGHT: why a fixed distance cannot be a quality gate - the KS "
           "distance of a good fit shrinks like 0.83/√N (median) and 1.36/√N (95 %), so one number is tight at low "
           "N and meaningless at high N; the p-value is the same test on an N-independent scale.")
    return finish(fig, out, cap, wspace=0.26, top=0.90)


def fig_stats_sigma_vs_n(T, out, label):
    roles = roles_in(T)
    if "nevt" not in T.columns:
        return None
    fig, axs = plt.subplots(1, len(roles), figsize=(5.4 * len(roles) + 0.6, 5.6), squeeze=False, sharey=True)
    axs = axs.ravel()
    for ax, r in zip(axs, roles):
        m = T["ss_%s" % r].gt(0) & T["nevt"].notna()
        n = T.loc[m, "nevt"].to_numpy(float)
        v = T.loc[m, "ss_%s" % r].to_numpy(float)
        med = T.loc[m, "med_%s" % r].to_numpy(float)
        ax.scatter(n, v, s=6, alpha=0.4, color=rc(r), label="single-shot per track")
        edges = np.geomspace(max(50, n.min() * 0.95), n.max() * 1.05, 11)
        idx = np.digitize(n, edges)
        cx, cy, cm = [], [], []
        for b in range(1, len(edges)):
            k = idx == b
            if k.sum() >= 5:
                cx.append(np.sqrt(edges[b - 1] * edges[b]))
                cy.append(np.median(v[k]))
                cm.append(np.median(med[k]))
        ax.plot(cx, cy, "o-", color="k", lw=1.6, ms=4, label="median per N bin (single-shot)")
        ax.plot(cx, cm, "s--", color="k", lw=1.0, ms=3, label="median per N bin (bootstrap median)")
        ax.set_xscale("log")
        ax.set_xlabel("events in the track after step 10")
        ax.set_title(r, fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7.5, loc="upper right")
        lo, hi = np.percentile(v, [1, 99])
        ax.set_ylim(lo - 5, hi + 8)
    axs[0].set_ylabel("σ per track [ps]")
    fig.suptitle("%s - does the extracted resolution depend on the track's statistics?" % label, fontsize=11, y=0.995)
    cap = ("Each track's per-board resolution against its event count, with the median per logarithmic N bin for "
           "the single-shot value (solid) and for the bootstrap median (dashed). The physical resolution of a "
           "pixel does not depend on how many events happened to land there, so a flat line is what a healthy "
           "estimator gives; a rise or fall towards low N is an estimator bias (the FWHM of a mixture fitted to "
           "few events, the KS acceptance selecting fluctuations, or the 3-board solve going marginal), and a "
           "growing gap between solid and dashed at low N is the bootstrap bias of the same estimator. Low-N "
           "tracks are also the edge tracks, so a trend can partly be geometry - compare with the σ maps.")
    return finish(fig, out, cap, wspace=0.16, top=0.88)


def per_pixel_role_all(T, r):
    """Per pixel of role r (one combo's table T): 1/err²-weighted mean single-shot σ, its error, n tracks."""
    sub = T[T["ss_%s" % r].gt(0) & T["std_%s" % r].gt(0)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["mean", "err", "n"])
    sub["w"] = 1.0 / sub["std_%s" % r] ** 2
    g = sub.groupby(["row_%s" % r, "col_%s" % r])
    return pd.DataFrame({"mean": g.apply(lambda x: np.average(x["ss_%s" % r], weights=x["w"])),
                         "err": g.apply(lambda x: 1.0 / np.sqrt(x["w"].sum())), "n": g.size()})


def fig_stats_maps(T, out, label):
    roles = roles_in(T)
    fig, axs = plt.subplots(2, len(roles), figsize=(5.4 * len(roles) + 0.6, 10.6), squeeze=False)
    lines = []
    for j, r in enumerate(roles):
        P = per_pixel_role_all(T, r)
        ax = axs[0, j]
        if P.empty:
            ax.axis("off")
            axs[1, j].axis("off")
            continue
        img = grid_from(P["mean"])
        fin = img[np.isfinite(img)]
        im = pixel_map(ax, img, cmap="viridis", vmin=np.percentile(fin, 2), vmax=np.percentile(fin, 98), nan_blank=True)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("σ_%s [ps]" % r, fontsize=8.5)
        nfail = int((T["ss_%s" % r] < 0).sum())
        ax.set_title("%s - σ per pixel (median %.1f ps; %d pixels%s)" % (r, np.nanmedian(fin), fin.size,
                                                                            "; %d failed track(s) excluded" % nfail if nfail else ""), fontsize=9.5)
        ax = axs[1, j]
        cols_c = np.array([k[1] for k in P.index])
        vals = P["mean"].to_numpy(float)
        errs = P["err"].to_numpy(float)
        right = cols_c < 8       # col<8 = physical RIGHT half
        left = ~right
        bp = ax.boxplot([vals[right], vals[left]], labels=["col < 8" + NL + "(physical right)", "col ≥ 8" + NL + "(physical left)"],
                        showfliers=True, patch_artist=True, widths=0.5, whis=[5, 95])
        for patch in bp["boxes"]:
            patch.set_facecolor(rc(r))
            patch.set_alpha(0.35)
        for med in bp["medians"]:
            med.set_color("k")

        def wm(m):
            w = 1.0 / errs[m] ** 2
            return np.sum(w * vals[m]) / np.sum(w), 1.0 / np.sqrt(np.sum(w))
        mr, er = wm(right)
        ml, el = wm(left)
        d, ed = ml - mr, np.hypot(el, er)
        ax.set_ylabel("σ_%s per pixel [ps]" % r)
        ax.set_title(("%s - chip halves: left − right = %+.2f ± %.2f ps" + NL + "(medians %.1f / %.1f; per-pixel spread %.1f ps)")
                     % (r, d, ed, np.median(vals[left]), np.median(vals[right]), np.std(vals)), fontsize=9.5)
        ax.grid(alpha=0.3, axis="y")
        lines.append("%s: left − right %+.2f ± %.2f ps" % (r, d, ed))
    fig.suptitle("%s - resolution over the array and per chip half" % label, fontsize=11, y=0.995)
    cap = ("TOP: each board's per-pixel resolution (1/err²-weighted mean of the single-shot σ over the tracks "
           "using the pixel; tracks whose single-shot failed are excluded), physical orientation, colour scale "
           "clipped at the 2nd-98th percentile. BOTTOM: the same per-pixel values split by chip half - col < 8 "
           "is the physical RIGHT half, col ≥ 8 the LEFT; the two halves' TDCs sit on different power domains and "
           "board 3 (trig) records ~12 % fewer hits on its left half, so this is where a per-half timing "
           "difference would show. Box = 25-75 %, whiskers 5-95 %, line = median; the title gives the "
           "error-weighted mean difference left − right with its statistical error (pixel-to-pixel spread is "
           "not included in that error: a difference smaller than the spread but larger than the error is a "
           "real but small systematic). " + "; ".join(lines) + ".")
    return finish(fig, out, cap, wspace=0.34, hspace=0.42, top=0.92)


def fig_stats_boards(T, out, label):
    roles = roles_in(T)
    fig, axs = plt.subplots(1, len(roles), figsize=(5.4 * len(roles) + 0.6, 5.6), squeeze=False)
    axs = axs.ravel()
    txt = []
    for ax, r in zip(axs, roles):
        v = T["ss_%s" % r]
        v = v[np.isfinite(v) & (v > 0)]
        m = T["med_%s" % r]
        m = m[np.isfinite(m) & (m > 0)]
        lo, hi = np.percentile(pd.concat([v, m]), [0.5, 99.5])
        bins = np.linspace(lo - 2, hi + 2, 60)
        ax.hist(v, bins=bins, color=rc(r), alpha=0.35, label="single-shot per track (n=%d)" % len(v))
        ax.hist(m, bins=bins, histtype="step", color="k", lw=1.3, label="bootstrap median per track")
        st = fbr.perform_robust_unbinned_fit(v, 2.5)
        stm = fbr.perform_robust_unbinned_fit(m, 2.5)
        x = np.linspace(bins[0], bins[-1], 400)
        ax.plot(x, len(v) * (bins[1] - bins[0]) * norm.pdf(x, st["mu"], st["sigma"]), color=rc(r), lw=2.0,
                label="robust Gaussian over tracks, single-shot: μ=%.2f, σ=%.2f ps" % (st["mu"], st["sigma"]))
        ax.plot(x, len(m) * (bins[1] - bins[0]) * norm.pdf(x, stm["mu"], stm["sigma"]), color="k", lw=1.2, ls=":",
                label="same, bootstrap median (≈ res_): μ=%.2f, σ=%.2f ps" % (stm["mu"], stm["sigma"]))
        ax.set_xlabel("σ_%s [ps]" % r)
        ax.set_ylabel("tracks")
        ax.set_title(r, fontsize=10)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
        ax.legend(fontsize=7.2, loc="upper right")
        ax.grid(alpha=0.3)
        txt.append("%s: μ=%.2f (single-shot) / %.2f (bootstrap median), σ=%.2f ps" % (r, st["mu"], stm["mu"], st["sigma"]))
    fig.suptitle("%s - per-track resolutions over the array" % label, fontsize=11, y=0.995)
    cap = ("Preview of a board-level number: the distribution over tracks of each board's per-track resolution "
           "(single-shot, filled; bootstrap median, black steps) with the robust Gaussian fit that step 13 applies "
           "PER TRACK to its bootstrap values (median/IQR-seeded, 2.5σ-clipped unbinned ML), here applied across "
           "tracks as a preview - the pipeline itself quotes no board number; a downstream fit over pixels would "
           "use res_<role> (the bootstrap μ per track), which the bootstrap median approximates, so both are "
           "fitted. The fit's μ is the board's typical value, its σ the pixel-to-pixel spread - NOT a statistical "
           "error (the bootstrap errors of the individual tracks are not propagated into it). Tails come from "
           "low-statistics or edge tracks. " + "; ".join(txt) + ".")
    return finish(fig, out, cap, wspace=0.26, top=0.88)


def cmd_stats(a):
    T = collect_boot(a.bootdir, a.time_base, a.log_dir)
    os.makedirs(a.outdir, exist_ok=True)
    T.to_csv(os.path.join(a.outdir, "boot_summary.csv"), index=False)
    print("stats: %d tracks in %d combo(s); wrote boot_summary.csv" % (len(T), T["combo"].nunique()))
    for combo, sub in T.groupby("combo"):
        sub = sub.dropna(axis=1, how="all")   # roles absent from this combo
        lab = combo or os.path.basename(a.bootdir)
        d = os.path.join(a.outdir, lab)
        os.makedirs(d, exist_ok=True)
        print("  wrote %s" % fig_stats_errors(sub, os.path.join(d, "stats_errors.png"), lab))
        print("  wrote %s" % fig_stats_boards(sub, os.path.join(d, "stats_boards.png"), lab))
        for fn, name in ((fig_stats_ks_gate, "stats_ks_gate.png"), (fig_stats_sigma_vs_n, "stats_sigma_vs_n.png"),
                         (fig_stats_maps, "stats_maps.png")):
            o = fn(sub, os.path.join(d, name), lab)
            if o:
                print("  wrote %s" % o)


# --------------------------------------------------------------------------
# consistency across combos
# --------------------------------------------------------------------------

def per_pixel_role(T, combo, r):
    """Per pixel of role r in one combo: inverse-variance-weighted mean of the single-shot σ over the
    tracks using that pixel, its error, the plain spread over those tracks, and the count."""
    sub = T[(T.combo == combo) & T["ss_%s" % r].gt(0) & T["std_%s" % r].gt(0)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["mean", "err", "spread", "n"])
    sub["w"] = 1.0 / sub["std_%s" % r] ** 2
    g = sub.groupby(["row_%s" % r, "col_%s" % r])
    out = pd.DataFrame({
        "mean": g.apply(lambda x: np.average(x["ss_%s" % r], weights=x["w"])),
        "err": g.apply(lambda x: 1.0 / np.sqrt(x["w"].sum())),
        "spread": g["ss_%s" % r].std(ddof=1),
        "n": g.size(),
    })
    return out


def grid_from(series):
    img = np.full((GRID, GRID), np.nan)
    for (r, c), v in series.items():
        img[int(r), int(c)] = v
    return img


def fig_consistency_role(T, r, combos, out):
    P = {c: per_pixel_role(T, c, r) for c in combos}
    A, B = combos[0], combos[1]
    J = P[A].join(P[B], lsuffix="_A", rsuffix="_B", how="inner")
    fig = plt.figure(figsize=(18.5, 12.0))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1.0], hspace=0.55, wspace=0.42)
    ax = fig.add_subplot(gs[0, 0])
    ax.errorbar(J["mean_A"], J["mean_B"], xerr=J["err_A"], yerr=J["err_B"], fmt="o", ms=3, lw=0.6, color=rc(r), alpha=0.7)
    lo = min(J["mean_A"].min(), J["mean_B"].min()) - 1
    hi = max(J["mean_A"].max(), J["mean_B"].max()) + 1
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("σ_%s per pixel in A [ps]" % r)
    ax.set_ylabel("σ_%s per pixel in B [ps]" % r)
    ax.set_title("σ_%s per pixel: B against A\n(%d pixels; A, B named in the title)" % (r, len(J)), fontsize=10)
    ax.grid(alpha=0.3)
    d = J["mean_B"] - J["mean_A"]
    e = np.sqrt(J["err_A"] ** 2 + J["err_B"] ** 2)
    ax = fig.add_subplot(gs[0, 1])
    img = grid_from(d)
    v = np.nanpercentile(np.abs(d), 98)
    im = pixel_map(ax, img, cmap="RdBu_r", vmin=-v, vmax=v, nan_blank=True)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("σ_%s(B) − σ_%s(A) [ps]" % (r, r), fontsize=8.5)
    ax.set_title("B − A over the %s array\nmean %+.2f ps, spread %.2f ps" % (r, d.mean(), d.std()), fontsize=10)
    ax = fig.add_subplot(gs[0, 2])
    pull = (d / e).replace([np.inf, -np.inf], np.nan).dropna()
    ax.hist(pull, bins=np.linspace(-6, 6, 49), color=rc(r), alpha=0.5)
    ax.set_xlabel("(B − A) / combined bootstrap error")
    ax.set_ylabel("pixels")
    ax.set_title("pull: mean %+.2f, robust width %.2f\n(null width < 1: shared events, see caption)" % (pull.mean(), robust_sigma(pull)), fontsize=10)
    ax.grid(alpha=0.3)
    ax = fig.add_subplot(gs[0, 3])
    ax.hist(d, bins=40, color=rc(r), alpha=0.5)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("σ_%s(B) − σ_%s(A) [ps]" % (r, r))
    ax.set_ylabel("pixels")
    ax.set_title("B − A: median %+.2f ps\n16-84%%: %+.2f .. %+.2f ps" % (d.median(), d.quantile(.16), d.quantile(.84)), fontsize=10)
    ax.grid(alpha=0.3)
    for j, c in enumerate(combos[:2]):
        ax = fig.add_subplot(gs[1, 2 * j])
        img = grid_from(P[c]["mean"])
        fin = img[np.isfinite(img)]
        im = pixel_map(ax, img, cmap="viridis", vmin=np.percentile(fin, 2), vmax=np.percentile(fin, 98), nan_blank=True)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("σ_%s [ps]" % r, fontsize=8.5)
        ax.set_title("%s: σ_%s per pixel\nmedian %.1f ps" % ("A" if j == 0 else "B", r, np.nanmedian(fin)), fontsize=10)
        ax = fig.add_subplot(gs[1, 2 * j + 1])
        img = grid_from(P[c]["spread"])
        fin = img[np.isfinite(img)]
        im = pixel_map(ax, img, cmap="magma", vmin=0, vmax=np.percentile(fin, 98) if fin.size else 1, nan_blank=True)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("std of σ_%s over the tracks sharing the pixel [ps]" % r, fontsize=8.5)
        ax.set_title("%s: spread of σ_%s over partner pixels\nmedian %.2f ps (per-track bootstrap error %.2f ps)"
                     % ("A" if j == 0 else "B", r, np.nanmedian(fin), T.loc[(T.combo == c) & T["ss_%s" % r].gt(0), "std_%s" % r].median()), fontsize=10)
    fig.suptitle("Does σ_%s depend on who the partner boards are?   A = %s,  B = %s" % (r, A, B), fontsize=11.5, y=0.995)
    cap = ("The same board's resolution obtained in two board combinations. For each pixel of this board, the "
           "single-shot σ of every track that uses the pixel is averaged with weights 1/err² (err = the track's "
           "bootstrap std), separately in each combo (A and B, named in the title). TOP-LEFT: pixel by pixel, B "
           "against A with the combined errors; the dashed line is equality. TOP-MIDDLE: the difference on the "
           "array - a uniform offset is either a partner effect inside the 3-board solve (correlated jitter, a "
           "partner's residual time walk) OR an event-selection effect of the third board (its acceptance and "
           "the per-combo step-10 cuts pick a different event mix); the shared-pair figure separates the two "
           "(pair widths shift → selection; pair widths stable → solve). A pattern over the array means the "
           "effect depends on where the tracks land. TOP-RIGHT: the difference divided by the two bootstrap errors "
           "combined as if independent. They are not: the two combos share most of their events for this board, "
           "so the null width is well below 1 (~0.6-0.9 depending on the overlap) - a width of ~1 or more is a "
           "firm systematic beyond statistics, and a mean far from 0 (relative to 1/sqrt(pixels)) is one "
           "regardless; the size in ps is the last top panel. BOTTOM: per combo, the per-pixel σ itself and the "
           "spread of σ over the different partner pixels that pair with the same pixel inside that combo - "
           "compare it with the per-track bootstrap error (in the title): equal means the partner choice within "
           "a combo does not matter beyond statistics.")
    return finish(fig, out, cap, wspace=0.42, hspace=0.55, top=0.93)


def fig_consistency_pairs(diag_dir, combos, out):
    """Same pixel PAIR width (IQR proxy from track_diagnostics) in two combos that share the pair."""
    D = {}
    for c in combos:
        f = glob(os.path.join(diag_dir, "*%s*_track_diagnostics.parquet" % c))
        if not f:
            continue
        dg = pd.read_parquet(f[0])
        sp = dg[(dg.section == "twc_full") & (dg.key == "sig_pair")]
        px = dg[(dg.section == "tracks") & dg.key.isin(["row", "col"])].pivot_table(index=["track", "role"], columns="key", values="value")
        nev = dg[(dg.section == "twc_full") & (dg.key == "n_events")].set_index("track")["value"].to_dict()
        D[c] = (sp, px, nev)
    if len(D) < 2:
        return None
    (ca, (spa, pxa, NA)), (cb, (spb, pxb, NB)) = list(D.items())[:2]
    shared = sorted(set(spa.role) & set(spb.role))
    if not shared:
        return None
    fig, axs = plt.subplots(1, 2 * len(shared), figsize=(5.6 * 2 * len(shared) + 0.5, 5.6), squeeze=False)
    axs = axs.ravel()
    lines = []
    for k, pr in enumerate(shared):
        a, b = pr.split("-")

        def keyed(sp, px, dg_nev):
            s = sp[sp.role == pr].set_index("track")["value"]
            keys = {}
            for t, v in s.items():
                if dg_nev.get(t, 0) < PAIR_MIN_EVENTS:
                    continue
                try:
                    ka = (int(px.loc[(t, a), "row"]), int(px.loc[(t, a), "col"]))
                    kb = (int(px.loc[(t, b), "row"]), int(px.loc[(t, b), "col"]))
                except KeyError:
                    continue
                keys.setdefault((ka, kb), []).append(v)
            return {kk: np.mean(v) for kk, v in keys.items()}
        KA, KB = keyed(spa, pxa, NA), keyed(spb, pxb, NB)
        common = sorted(set(KA) & set(KB))
        if len(common) < 5:
            axs[2 * k].axis("off")
            axs[2 * k + 1].axis("off")
            lines.append("%s: fewer than 5 shared pixel pairs" % pr)
            continue
        xa = np.array([KA[c] for c in common])
        xb = np.array([KB[c] for c in common])
        ax = axs[2 * k]
        ax.scatter(xa, xb, s=7, alpha=0.6, color="#444444")
        lo, hi = min(xa.min(), xb.min()) - 1, max(xa.max(), xb.max()) + 1
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("width of TOA(%s) − TOA(%s) in %s [ps]" % (a, b, ca))
        ax.set_ylabel("same pixel pair in %s [ps]" % cb)
        ax.set_title("pair %s: %d identical pixel pairs" % (pr, len(common)), fontsize=10)
        ax.grid(alpha=0.3)
        ax = axs[2 * k + 1]
        d = xb - xa
        ax.hist(d, bins=40, color="#777777")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("difference %s − %s [ps]" % (cb, ca))
        ax.set_ylabel("pixel pairs")
        ax.set_title("median %+.2f ps, 16-84%%: %+.2f .. %+.2f" % (np.median(d), np.percentile(d, 16), np.percentile(d, 84)), fontsize=10)
        ax.grid(alpha=0.3)
        lines.append("%s: median %+.2f ps" % (pr, np.median(d)))
    fig.suptitle("Does a pixel PAIR's width depend on the third board?", fontsize=11.5, y=0.995)
    cap = ("The cleanest partner test: a pair of pixels (one on each of two boards) that appears in both combos "
           "gives the width of its TWC-corrected TOA difference twice, from two different event samples - the "
           "events where the third board of combo A also fired versus those where the third board of combo B "
           "fired. The pair's two boards are the same, but their step-10 ToT windows and the TOA-correlation cut "
           "are re-derived per combo and the coupled TWC uses the third board, so a systematic offset means the "
           "third board's requirement selects different events (its position/acceptance biases the track angles "
           "or charge, or the per-combo cuts differ) or its own time walk leaks in through the coupled TWC. If "
           "instead the pair widths agree while the per-board σ (previous figures) does not, the partner effect "
           "sits in the 3-board solve. Widths here are the IQR/1.349 proxy from track_diagnostics (the boot files "
           "only store the solved per-board σ, not the pair widths), from tracks with at least %d events; tracks "
           "sharing the same pixel pair are averaged without weights. " % PAIR_MIN_EVENTS + "; ".join(lines) + ".")
    return finish(fig, out, cap, wspace=0.30, top=0.86)


def fig_consistency_pairs_boot(T, combos, out):
    """Same pixel PAIR: single-shot pair width (FWHM/2.355, from the boot files) in two combos."""
    A, B = combos
    pcols = sorted(c for c in T.columns if c.startswith("sspair_"))
    shared = [c for c in pcols if T.loc[T.combo == A, c].notna().any() and T.loc[T.combo == B, c].notna().any()]
    if not shared:
        return None
    fig, axs = plt.subplots(1, 2 * len(shared), figsize=(5.6 * 2 * len(shared) + 0.5, 5.6), squeeze=False)
    axs = axs.ravel()
    lines = []
    for k, pc in enumerate(shared):
        pr = pc[len("sspair_"):]
        a, b = pr.split("-")

        def keyed(c):
            sub = T[(T.combo == c) & T[pc].gt(0)]
            g = sub.groupby(["row_%s" % a, "col_%s" % a, "row_%s" % b, "col_%s" % b])[pc].mean()
            return g.to_dict()
        KA, KB = keyed(A), keyed(B)
        common = sorted(set(KA) & set(KB))
        if len(common) < 5:
            axs[2 * k].axis("off")
            axs[2 * k + 1].axis("off")
            lines.append("%s: fewer than 5 shared pixel pairs" % pr)
            continue
        xa = np.array([KA[c] for c in common])
        xb = np.array([KB[c] for c in common])
        ax = axs[2 * k]
        ax.scatter(xa, xb, s=7, alpha=0.6, color="#444444")
        lo, hi = min(xa.min(), xb.min()) - 1, max(xa.max(), xb.max()) + 1
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("width of TOA(%s) − TOA(%s) in A [ps]" % (a, b))
        ax.set_ylabel("same pixel pair in B [ps]")
        ax.set_title(("pair %s: %d identical pixel pairs" + NL + "(step-12 pair widths, FWHM/2.355)") % (pr, len(common)), fontsize=10)
        ax.grid(alpha=0.3)
        ax = axs[2 * k + 1]
        d = xb - xa
        ax.hist(d, bins=40, color="#777777")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("difference B − A [ps]")
        ax.set_ylabel("pixel pairs")
        ax.set_title("median %+.2f ps, 16-84%%: %+.2f .. %+.2f" % (np.median(d), np.percentile(d, 16), np.percentile(d, 84)), fontsize=10)
        ax.grid(alpha=0.3)
        lines.append("%s: median %+.2f ps" % (pr, np.median(d)))
    fig.suptitle("Does a pixel PAIR's width depend on the third board?   A = %s,  B = %s" % (A, B), fontsize=11.5, y=0.995)
    cap = ("The same test as the proxy version, now with the pipeline's own numbers: for a pair of pixels present "
           "in both combos, the single-shot pair width step 12 computed (FWHM/2.355 of the kept mixture of the "
           "TWC-corrected TOA difference) in A against B. The two samples differ only in which third board also "
           "fired (plus the per-combo step-10 windows/correlation cut and the coupled TWC). Widths that agree "
           "while the per-board σ differs between the same combos put the partner effect inside the 3-board "
           "solve; widths that shift put it in the event selection. Tracks sharing the pixel pair are averaged "
           "without weights. " + "; ".join(lines) + ".")
    return finish(fig, out, cap, wspace=0.30, top=0.86)


def fig_consistency_summary(T, out):
    """Per board: box plot of per-pixel σ for every combo it appears in."""
    combos = sorted(T["combo"].unique())
    roles = roles_in(T)
    fig, axs = plt.subplots(1, len(roles), figsize=(4.8 * len(roles) + 0.6, 5.8), squeeze=False)
    axs = axs.ravel()
    lines = []
    for ax, r in zip(axs, roles):
        data, labels = [], []
        for c in combos:
            sub = T[(T.combo == c)]
            if r not in roles_in(sub.dropna(axis=1, how="all")):
                continue
            P = per_pixel_role_all(sub, r)
            if P.empty:
                continue
            data.append(P["mean"].to_numpy(float))
            labels.append(c.replace("-", NL) + NL + "(%d px)" % len(P))
            lines.append("%s in %s: median %.2f ps" % (r, c, np.median(P["mean"])))
        if not data:
            ax.axis("off")
            continue
        bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True, widths=0.55, whis=[5, 95])
        for patch in bp["boxes"]:
            patch.set_facecolor(rc(r))
            patch.set_alpha(0.35)
        for med in bp["medians"]:
            med.set_color("k")
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_ylabel("σ_%s per pixel [ps]" % r)
        ax.set_title(r, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("The same board's resolution from every combo it appears in", fontsize=11.5, y=0.995)
    cap = ("Per board, the distribution over pixels of the per-pixel single-shot σ (1/err²-weighted over the "
           "tracks using the pixel), one box per board combination the board is part of (box 25-75 %, whiskers "
           "5-95 %, line = median). In an ideal telescope every box of a board sits at the same value: the "
           "resolution of a chip does not depend on which other two chips were used to measure it. Boxes that "
           "move with the partners are the partner systematic; which partner moves them (see the per-pair "
           "figures) points at the mechanism. " + "; ".join(lines) + ".")
    return finish(fig, out, cap, wspace=0.34, top=0.88)


def cmd_consistency(a):
    T = collect_boot(a.bootdir, a.time_base, None)
    combos = sorted(T["combo"].unique())
    if len(combos) < 2:
        die("consistency needs at least two combos under %s" % a.bootdir)
    os.makedirs(a.outdir, exist_ok=True)
    print("  wrote %s" % fig_consistency_summary(T, os.path.join(a.outdir, "consistency_summary.png")))
    roles_by_combo = {c: roles_in(T[T.combo == c].dropna(axis=1, how="all")) for c in combos}
    for r in sorted(set.union(*[set(v) for v in roles_by_combo.values()])):
        cs = [c for c in combos if r in roles_by_combo[c]]
        for pair in combinations(cs, 2):
            out = os.path.join(a.outdir, "consistency_%s__%s__vs__%s.png" % (r, pair[0], pair[1]))
            print("  wrote %s" % fig_consistency_role(T, r, list(pair), out))
    for pair in combinations(combos, 2):
        out = fig_consistency_pairs_boot(T, list(pair), os.path.join(a.outdir, "consistency_pairs__%s__vs__%s.png" % pair))
        if out:
            print("  wrote %s" % out)
        elif a.diag_dir:
            out = fig_consistency_pairs(a.diag_dir, list(pair), os.path.join(a.outdir, "consistency_pairs_proxy__%s__vs__%s.png" % pair))
            if out:
                print("  wrote %s" % out)



# --------------------------------------------------------------------------
# compare: two step-12 output sets (e.g. two configurations) track by track
# --------------------------------------------------------------------------

def fig_compare_gate(J, roles, out, labA, labB):
    """What an acceptance rule does: rejected fraction vs N for A and B, and the effect of the
    biased selection on the extracted numbers (bootstrap spread and central value, B relative to A)."""
    have = all(c in J.columns for c in ["rej_A", "rej_B", "n_boot_A", "n_boot_B", "nevt_A"])
    if not have:
        return None
    n = J["nevt_A"].to_numpy(float)
    with np.errstate(all="ignore"):
        fA = (J["rej_A"].fillna(0) / (J["rej_A"].fillna(0) + J["phys_A"].fillna(0) + J["n_boot_A"])).to_numpy(float)
        fB = (J["rej_B"].fillna(0) / (J["rej_B"].fillna(0) + J["phys_B"].fillna(0) + J["n_boot_B"])).to_numpy(float)
    fig, axs = plt.subplots(1, 3, figsize=(17.0, 5.9))
    ax = axs[0]
    ax.scatter(n, fA, s=9, alpha=0.6, color="#999999", label="A: %s" % labA)
    ax.scatter(n, fB, s=9, alpha=0.6, color="#c0392b", label="B: %s" % labB)
    ng = np.geomspace(max(50, np.nanmin(n) * 0.8), np.nanmax(n) * 1.2, 200)
    ax.plot(ng, kstwobign.sf(0.03 * np.sqrt(ng)), "k--", lw=1.4, label="what the OLD rule (D ≤ 0.03) does to a PERFECT fit")
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("events in the track after step 10")
    ax.set_ylabel("fraction of bootstrap resamples thrown away by the mixture-fit gate")
    ax.set_title("how many resamples each rule rejects", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, which="both")
    for k, (key, lab, ttl) in enumerate([("std", "bootstrap std (B) / bootstrap std (A)", "effect on the quoted error"),
                                          ("med", "bootstrap median (B) / bootstrap median (A)", "effect on the central value")]):
        ax = axs[k + 1]
        for r in roles:
            m = J["%s_%s_A" % (key, r)].gt(0) & J["%s_%s_B" % (key, r)].gt(0)
            ratio = (J.loc[m, "%s_%s_B" % (key, r)] / J.loc[m, "%s_%s_A" % (key, r)]).to_numpy(float)
            nn = J.loc[m, "nevt_A"].to_numpy(float)
            ax.scatter(nn, ratio, s=6, alpha=0.35, color=rc(r))
            edges = np.geomspace(max(50, nn.min() * 0.95), nn.max() * 1.05, 9)
            idx = np.digitize(nn, edges)
            cx = [np.sqrt(edges[b - 1] * edges[b]) for b in range(1, len(edges)) if (idx == b).sum() >= 5]
            cy = [np.median(ratio[idx == b]) for b in range(1, len(edges)) if (idx == b).sum() >= 5]
            ax.plot(cx, cy, "o-", color=rc(r), lw=1.6, ms=4, label="%s: median per N bin" % r)
        ax.axhline(1, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("events in the track after step 10")
        ax.set_ylabel(lab)
        ax.set_title(ttl, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        lo, hi = np.nanpercentile(np.concatenate([(J.loc[J["%s_%s_A" % (key, r)].gt(0) & J["%s_%s_B" % (key, r)].gt(0), "%s_%s_B" % (key, r)] /
                                                    J.loc[J["%s_%s_A" % (key, r)].gt(0) & J["%s_%s_B" % (key, r)].gt(0), "%s_%s_A" % (key, r)]).to_numpy(float) for r in roles]), [1, 99])
        ax.set_ylim(min(lo, 0.9) - 0.03, max(hi, 1.1) + 0.03)
    fig.suptitle("What the acceptance rule of the mixture fit does   (A = %s, B = %s)" % (labA, labB), fontsize=11, y=0.995)
    cap = ("Plain words first. Step 12 draws 200 resamples per track and, for each, fits a Gaussian mixture to the "
           "three pairwise TOA-difference distributions; a resample is only KEPT if every fit passes a "
           "goodness-of-fit gate, otherwise it is thrown away and another one is drawn. If the gate throws away "
           "resamples for statistical reasons rather than because the fit is broken, the kept ones are a biased "
           "sample - the ones that happened to look most Gaussian - and both the quoted error (spread of the "
           "kept values) and the central value shift. LEFT: for each track, the fraction of resamples the gate "
           "threw away, under rule A (grey) and rule B (red), against the track's number of events; the dashed "
           "curve is what the OLD rule 'KS distance ≤ 0.03' does to a PERFECT fit (the KS distance of a good fit "
           "shrinks like 0.83/√N, so a fixed 0.03 rejects most good fits below ~800 events and nothing above "
           "~3000). MIDDLE and RIGHT: the consequence, track by track - the ratio B/A of the bootstrap spread "
           "(the error step 13 quotes) and of the bootstrap median (the value); coloured lines are medians per "
           "N bin. Where the old rule rejected heavily, the kept resamples were too alike: the spread was "
           "under-estimated and the value shifted; the new rule (p-value ≥ 1e-3 or D ≤ 0.03) rejects only broken "
           "fits at any N. A and B here differ ONLY in the acceptance rule if both were made with the same "
           "mixture-convergence settings; otherwise the middle/right ratios also contain that difference.")
    return finish(fig, out, cap, wspace=0.30, top=0.90)


def cmd_compare(a):
    TA = collect_boot(a.bootdir_a, a.time_base, a.log_dir_a)
    TB = collect_boot(a.bootdir_b, a.time_base, a.log_dir_b)
    key = ["combo", "track"]
    J = TA.merge(TB, on=key, suffixes=("_A", "_B"))
    if J.empty:
        die("no common tracks between %s and %s" % (a.bootdir_a, a.bootdir_b))
    os.makedirs(a.outdir, exist_ok=True)
    roles = sorted(c[3:] for c in TA.columns if c.startswith("ss_"))
    pairs = sorted(c[len("sspair_"):] for c in TA.columns if c.startswith("sspair_") and c in TB.columns)
    ncol = len(roles) + (1 if pairs else 0)
    fig, axs = plt.subplots(2, ncol, figsize=(4.9 * ncol + 1.0, 9.6), squeeze=False)
    lines = []
    have_n = "nevt_A" in J.columns and J["nevt_A"].notna().any()
    for j, r in enumerate(roles):
        m = J["ss_%s_A" % r].gt(0) & J["ss_%s_B" % r].gt(0)
        ratio = (J.loc[m, "ss_%s_B" % r] / J.loc[m, "ss_%s_A" % r]).to_numpy(float)
        ax = axs[0, j]
        if have_n:
            ax.scatter(J.loc[m, "nevt_A"], ratio, s=6, alpha=0.45, color=rc(r))
            ax.set_xscale("log")
            ax.set_xlabel("events in the track after step 10")
        ax.axhline(1, color="k", lw=0.8)
        ax.set_ylabel("σ_%s(B) / σ_%s(A), single-shot" % (r, r))
        ax.set_title("%s: median ratio %.3f (16-84%%: %.3f .. %.3f)" % (r, np.median(ratio), np.percentile(ratio, 16), np.percentile(ratio, 84)), fontsize=9.5)
        ax.grid(alpha=0.3, which="both")
        lo, hi = np.percentile(ratio, [1, 99])
        ax.set_ylim(lo - 0.05, hi + 0.05)
        ax = axs[1, j]
        ax.hist(ratio, bins=60, color=rc(r), alpha=0.5)
        ax.axvline(1, color="k", lw=0.8)
        ax.set_xlabel("σ_%s(B) / σ_%s(A)" % (r, r))
        ax.set_ylabel("tracks")
        ax.grid(alpha=0.3)
        lines.append("%s %.3f" % (r, np.median(ratio)))
    if pairs:
        ax = axs[0, -1]
        for pr in pairs:
            m = J["sspair_%s_A" % pr].gt(0) & J["sspair_%s_B" % pr].gt(0)
            ratio = (J.loc[m, "sspair_%s_B" % pr] / J.loc[m, "sspair_%s_A" % pr]).to_numpy(float)
            if have_n:
                ax.scatter(J.loc[m, "nevt_A"], ratio, s=5, alpha=0.4, label="pair %s: median %.3f" % (pr, np.median(ratio)))
            axs[1, -1].hist(ratio, bins=60, histtype="step", lw=1.4, label="pair %s" % pr)
            lines.append("pair %s %.3f" % (pr, np.median(ratio)))
        ax.set_xscale("log")
        ax.axhline(1, color="k", lw=0.8)
        ax.set_xlabel("events in the track after step 10")
        ax.set_ylabel("pair width (B) / pair width (A), single-shot")
        ax.set_title("pair widths", fontsize=9.5)
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.3, which="both")
        axs[1, -1].axvline(1, color="k", lw=0.8)
        axs[1, -1].set_xlabel("pair width ratio B / A")
        axs[1, -1].set_ylabel("tracks")
        axs[1, -1].legend(fontsize=7.5)
        axs[1, -1].grid(alpha=0.3)
    fig.suptitle("Step-12 output B relative to A, track by track   (A = %s, B = %s; %d common tracks)"
                 % (os.path.basename(a.bootdir_a.rstrip("/")), os.path.basename(a.bootdir_b.rstrip("/")), len(J)), fontsize=11, y=0.995)
    cap = ("Two step-12 output sets for the same tracks (typically two configurations of bootstrap.py - e.g. the "
           "mixture fitted with sklearn's default tolerance versus to convergence, or two acceptance rules): for "
           "each track, the ratio of the single-shot resolution per board (top: against the event count; bottom: "
           "distribution) and, when both sets carry them, of the pair widths. A ratio different from 1 by more "
           "than the bootstrap error (~4 %% per track) is a systematic of the configuration, not statistics; a "
           "ratio that depends on N says the effect is statistical in origin (fit convergence and estimator bias "
           "both grow with the amount of structure the fit can resolve). Medians: %s." % "; ".join(lines))
    print("  wrote %s" % finish(fig, os.path.join(a.outdir, "compare_%s__vs__%s.png" % (os.path.basename(a.bootdir_a.rstrip("/")), os.path.basename(a.bootdir_b.rstrip("/")))), cap, wspace=0.34, hspace=0.34, top=0.92))
    J.to_csv(os.path.join(a.outdir, "compare_%s__vs__%s.csv" % (os.path.basename(a.bootdir_a.rstrip("/")), os.path.basename(a.bootdir_b.rstrip("/")))), index=False)
    if a.log_dir_a and a.log_dir_b:
        o = fig_compare_gate(J, roles, os.path.join(a.outdir, "compare_gate_%s__vs__%s.png" % (os.path.basename(a.bootdir_a.rstrip("/")), os.path.basename(a.bootdir_b.rstrip("/")))),
                             os.path.basename(a.bootdir_a.rstrip("/")), os.path.basename(a.bootdir_b.rstrip("/")))
        if o:
            print("  wrote %s" % o)

# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("anatomy", help="one track: every stage of step 12/13 drawn with the fits on the data")
    s.add_argument("-f", "--file", required=True, help="step-10 track parquet (<combo>/time/track_*.parquet)")
    s.add_argument("--boot-file", default=None, dest="boot_file", help="that track's step-12 *_boot.parquet, for comparison")
    s.add_argument("--diag-parquet", default=None, dest="diag_parquet", help="track_diagnostics parquet of the combo (IQR proxy for comparison)")
    s.add_argument("--replay-boot", type=int, default=0, dest="replay_boot", help="also replay N bootstrap resamples here (slow: ~0.6 s each)")
    s.add_argument("--neighbor-cut", nargs="+", default=["none"], dest="neighbor_cut", help="same as step 12's --neighbor_cut (default none)")
    s.add_argument("--neighbor-logic", default="OR", dest="neighbor_logic", help="same as step 12's --neighbor_logic")
    s.add_argument("--sigma-cut", type=float, default=2.5, dest="sigma_cut", help="same as step 13's --sigma_cut (default 2.5)")
    s.add_argument("--tag", default=None, help="short label prefixed to the output folder and title, e.g. highest-stat / median-stat / lowest-stat / failed")
    s.add_argument("-o", "--outdir", required=True)
    s.set_defaults(fn=cmd_anatomy)
    q = sp.add_parser("stats", help="all tracks: statistical behaviour of the bootstrap")
    q.add_argument("-d", "--bootdir", required=True, help="step-12 output dir (bootstrap_<run>, combos auto-detected)")
    q.add_argument("--time-base", default=None, dest="time_base", help="step-9/10 base dir holding <combo>/time/ (for event counts)")
    q.add_argument("--log-dir", default=None, dest="log_dir", help="condor_logs/bootstrap/<tag> (for attempt counts)")
    q.add_argument("-o", "--outdir", required=True)
    q.set_defaults(fn=cmd_stats)
    c = sp.add_parser("consistency", help="same board / same pixel pair across combos")
    c.add_argument("-d", "--bootdir", required=True)
    c.add_argument("--time-base", default=None, dest="time_base")
    c.add_argument("--diag-dir", default=None, dest="diag_dir", help="dir with the track_diagnostics parquets (pair-width test)")
    c.add_argument("-o", "--outdir", required=True)
    c.set_defaults(fn=cmd_consistency)
    k = sp.add_parser("compare", help="two step-12 output sets track by track (e.g. two bootstrap.py configurations)")
    k.add_argument("-a", "--bootdir-a", required=True, dest="bootdir_a", help="reference step-12 output dir (combos auto-detected)")
    k.add_argument("-b", "--bootdir-b", required=True, dest="bootdir_b", help="step-12 output dir to compare against A")
    k.add_argument("--time-base", default=None, dest="time_base")
    k.add_argument("--log-dir-a", default=None, dest="log_dir_a", help="condor log dir of A (with --log-dir-b: adds the acceptance-rule before/after figure)")
    k.add_argument("--log-dir-b", default=None, dest="log_dir_b")
    k.add_argument("-o", "--outdir", required=True)
    k.set_defaults(fn=cmd_compare)
    for sub_ in (s, q, c, k):
        add_output_arguments(sub_)
    a = p.parse_args()
    set_output_options(a.format, a.split)
    a.fn(a)


if __name__ == "__main__":
    main()
