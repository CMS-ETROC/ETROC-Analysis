#!/usr/bin/env python3
"""Per-track diagnostics after steps 9/10: stability along the run, and the
time-walk correction (TWC) across the pixel array and across boards.

WHAT THIS IS
------------
A second run-level health check, complementary to telescope_diagnostics.py
(which looks at the telescope and the beam after step 7). This one looks at the
per-track files - step 9's ``<combo>/tracks/track_*.parquet`` (raw TDC codes,
one file per pixel triple, every event of the run) and step 10's
``<combo>/time/track_*.parquet`` (the same tracks after the TDC cuts, in
picoseconds, with the step-8 ``file`` index and the raw CAL carried through),
and asks two families of questions:

  1. Is the run STABLE along its length?  Every step-9/10 row carries the index
     of the raw file it came from (``file`` = loop_N).  The DAQ closes a raw file
     when it reaches a fixed size, so the file index is "accumulated data" and,
     under a constant-rate assumption, a clock.  Events per file, cut survival,
     mean CAL (the TDC bin size step 10 used), TOT and inter-board TOA offsets
     are followed file by file; the TWC and a resolution proxy are followed in
     bins of a few files (a 2nd-order fit needs a few hundred events).

  2. Is the TIME-WALK CORRECTION consistent?  Step 12 (bootstrap.py) fits, per
     track and per board, a 2nd-order polynomial in that board's TOT to the
     board's TOA offset from the mean of the other two boards, twice
     iteratively, and adds it to the TOA.  This tool re-runs exactly that fit
     (same formula, same iteration count) on every track and reports the
     EFFECTIVE polynomial (the two iterations summed) so it can be compared
     pixel to pixel across one board, board to board, and bin to bin along the
     run.

It has two subcommands:

  summarize  reads one combo's tracks/ + time/ directories, computes everything
             once per track, and writes ONE tidy parquet
             (<outdir>/<label>_track_diagnostics.parquet).  Slow part (reads
             every track file twice over EOS); parallelised over tracks.
  plot       reads that parquet and writes the figures into <outdir>/<label>/
             (what each figure shows is explained in the README, section 10b).
             Fast; re-run freely.

TIME AXIS
---------
The pipeline never records wall-clock time per event.  What exists is the raw
file index (``file``) and, in the DAQ's run_metadata.yaml, the run start time
and the file-size chunking.  ``plot`` therefore labels the x axis in file index
and, if you give it ``--t0 <ISO> --duration-min <D>`` (run start and duration:
e.g. the DAQ's max_run_time when the run ran to its cap), adds a second axis in
hours under the explicit assumption of a constant data rate.  If the raw
binaries' create/modify times are available, ``--file-times <csv>`` (columns
``file,time``) replaces the assumption by the measured mapping.

CONVENTIONS
-----------
* Roles come from the toa_/tot_ columns of the step-10 file; the pixel of each
  role comes from the track filename (t/d/r/e = trig/dut/ref/extra, the same
  regex steps 11 and 13 use).  Pixel maps use the physical orientation of
  telescope_diagnostics.py (looking down at the ETROC, wire-bond pads at the
  bottom edge, pixel (0,0) bottom-RIGHT).
* "Effective TWC polynomial" P_r(TOT) = a2*TOT^2 + a1*TOT + a0 [ps, TOT in ps]
  is the total correction step 12 ADDS to toa_r; a2 and a1 of the two iterations
  are summed (both iterations evaluate at the same TOT, so their sum is the
  polynomial actually applied).  Because raw a0/a1/a2 are strongly correlated,
  the comparable quantities reported are: the correction's SLOPE at the board's
  median TOT (ps per ns of TOT), its CURVATURE 2*a2 (ps per ns^2), and its RANGE
  P(TOT_p90) - P(TOT_p10) (ps) - how much time walk is being removed across the
  central 80% of that board's TOT distribution.
* "Resolution proxy": after the TWC, the robust width (IQR/1.349) of each
  pairwise TOA difference and the usual 3-board solve
  sigma_a^2 = (s_ab^2 + s_ac^2 - s_bc^2)/2.  It is NOT the step-12 number
  (which uses a GMM FWHM and bootstrap), only a cheap stand-in expected to
  follow it in trends and comparisons.
* The TWC is fitted on the same events a step-12 run WITHOUT --neighbor_cut
  would use (every surviving step-10 row of the track).

USAGE
-----
  python utils/track_diagnostics.py summarize \\
      --tracks-dir <combo>/tracks --time-dir <combo>/time \\
      --label h1_run1_ref1-dut2-trig3 -o <outdir> \\
      [--files-per-bin 5] [--min-events-twc 3000] [--workers 4] [--max-tracks N]

  python utils/track_diagnostics.py plot -i <parquet> -o <outdir> \\
      [--t0 2026-03-11T19:02:56 --duration-min 720] [--file-times times.csv]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from natsort import natsorted

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from telescope_diagnostics import GRID, pixel_map, save_figure, set_output_options, add_output_arguments  # noqa: E402

NICK = {"t": "trig", "d": "dut", "r": "ref", "e": "extra"}
FNAME_RE = re.compile(r"(\w)-R(\d+)C(\d+)")
N_TWC_ITER = 2          # bootstrap.py: `for _ in range(2)`
TWC_ORDER = 2           # bootstrap.py: np.polyfit(..., 2)
MIN_BIN_EVENTS = 150    # below this a 2nd-order fit per bin is not attempted
IQR_TO_SIGMA = 1.0 / 1.349

# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def die(msg):
    sys.exit("track_diagnostics: " + msg)


def parse_pixels(stem):
    """'track_t-R0C0_d-R0C0_r-R0C0' -> {'trig': (0,0), 'dut': (0,0), 'ref': (0,0)}"""
    out = {}
    for n, r, c in FNAME_RE.findall(stem):
        if n in NICK:
            out[NICK[n]] = (int(r), int(c))
    return out


def robust_sigma(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 20:
        return np.nan
    q75, q25 = np.percentile(x, [75, 25])
    return (q75 - q25) * IQR_TO_SIGMA


def pair_name(a, b):
    return "%s-%s" % (a, b)


def twc_fit(tots, toas, roles):
    """bootstrap.apply_timewalk_correction, but returning the coefficients.

    tots/toas: dict role -> ndarray (ps).  Returns (eff, corrected_toas) where
    eff[role] = summed polynomial coefficients [a2, a1, a0] over the
    N_TWC_ITER iterations, i.e. the correction actually added to toa_role."""
    toas = {r: np.array(toas[r], float, copy=True) for r in roles}
    eff = {r: np.zeros(TWC_ORDER + 1) for r in roles}
    for _ in range(N_TWC_ITER):
        deltas = {}
        for r in roles:
            others = [toas[o] for o in roles if o != r]
            deltas[r] = 0.5 * sum(others) - toas[r]
        for r in roles:
            c = np.polyfit(tots[r], deltas[r], TWC_ORDER)
            eff[r] += c
            toas[r] += np.poly1d(c)(tots[r])
    return eff, toas


def pair_sigmas(toas, roles):
    out = {}
    for i, a in enumerate(roles):
        for b in roles[i + 1:]:
            out[pair_name(a, b)] = robust_sigma(toas[a] - toas[b])
    return out


def solve_three_board(sig_pairs, roles):
    """sigma_a^2 = 0.5 (s_ab^2 + s_ac^2 - s_bc^2); NaN if not solvable."""
    out = {}
    if len(roles) != 3:
        return {r: np.nan for r in roles}

    def s2(a, b):
        v = sig_pairs.get(pair_name(a, b), sig_pairs.get(pair_name(b, a), np.nan))
        return v * v

    for r in roles:
        o = [x for x in roles if x != r]
        v = 0.5 * (s2(r, o[0]) + s2(r, o[1]) - s2(o[0], o[1]))
        out[r] = np.sqrt(v) if np.isfinite(v) and v > 0 else np.nan
    return out


# --------------------------------------------------------------------------
# summarize: one worker per track
# --------------------------------------------------------------------------

class Rec:
    """Tidy accumulator: one row per (track, section, role, file, bin, key)."""

    def __init__(self):
        self.rows = []

    def add(self, track, section, key, value, role="", file=-1, bin=-1):
        self.rows.append((track, section, role, int(file), int(bin), key, float(value)))

    def frame(self):
        df = pd.DataFrame(self.rows, columns=["track", "section", "role", "file", "bin", "key", "value"])
        df["file"] = df["file"].astype(np.int32)
        df["bin"] = df["bin"].astype(np.int32)
        return df


def process_track(job):
    """job = (track_file, time_file, files_per_bin, min_events_twc).  Returns list of rows."""
    tf, mf, files_per_bin, min_events_twc = job
    stem = os.path.splitext(os.path.basename(mf if mf else tf))[0]
    R = Rec()
    pix = parse_pixels(stem)
    d9 = pd.read_parquet(tf, columns=["file"])

    if mf is None:
        # step 10 filtered this track to nothing (no time/ file): it still ENTERED
        # step 10, so its step-9 rows count in n_in with n_out = 0.
        for r, (rr, cc) in pix.items():
            R.add(stem, "tracks", "row", rr, role=r)
            R.add(stem, "tracks", "col", cc, role=r)
        R.add(stem, "tracks", "n_in", len(d9))
        R.add(stem, "tracks", "n_out", 0)
        for f, v in d9.groupby("file").size().items():
            R.add(stem, "file", "n_in", v, file=f)
        return R.rows

    d10 = pd.read_parquet(mf)
    roles = sorted(c[4:] for c in d10.columns if c.startswith("toa_"))
    if not roles:
        return R.rows

    for r in roles:
        if r in pix:
            R.add(stem, "tracks", "row", pix[r][0], role=r)
            R.add(stem, "tracks", "col", pix[r][1], role=r)
    R.add(stem, "tracks", "n_in", len(d9))
    R.add(stem, "tracks", "n_out", len(d10))

    # ---- per file: counts in / out
    n_in = d9.groupby("file").size()
    n_out = d10.groupby("file").size()
    for f, v in n_in.items():
        R.add(stem, "file", "n_in", v, file=f)
    for f, v in n_out.items():
        R.add(stem, "file", "n_out", v, file=f)

    if len(d10) < 20:
        return R.rows

    tots = {r: d10["tot_%s" % r].to_numpy(float) for r in roles}
    toas = {r: d10["toa_%s" % r].to_numpy(float) for r in roles}
    files = d10["file"].to_numpy()

    # ---- per file: mean CAL (= the bin-size CAL step 10 used on its default
    #      cut-then-convert path), median TOA, MEAN TOT (TOT is quantised in
    #      2-bin = ~40 ps steps, so a per-file median just hops between codes;
    #      the mean is the usable drift indicator), raw pairwise TOA offset (no
    #      TWC) and each board's raw offset against the mean of its two partners.
    g = d10.groupby("file")
    for r in roles:
        for f, v in g["cal_%s" % r].mean().items():
            R.add(stem, "file", "cal_mean", v, role=r, file=f)
        for f, v in g["toa_%s" % r].median().items():
            R.add(stem, "file", "toa_med", v, role=r, file=f)
        for f, v in g["tot_%s" % r].mean().items():
            R.add(stem, "file", "tot_mean", v, role=r, file=f)
        R.add(stem, "tracks", "cal_mean_all", d10["cal_%s" % r].mean(), role=r)
        R.add(stem, "tracks", "toa_med_all", np.median(toas[r]), role=r)
        R.add(stem, "tracks", "tot_mean_all", tots[r].mean(), role=r)
    for i, a in enumerate(roles):
        for b in roles[i + 1:]:
            dd = pd.Series(toas[a] - toas[b])
            pr = pair_name(a, b)
            for f, v in dd.groupby(files).mean().items():
                R.add(stem, "file", "dtoa_mean", v, role=pr, file=f)
            R.add(stem, "tracks", "dtoa_mean_all", dd.mean(), role=pr)

    # ---- raw offset of each board against the mean of its partners (no TWC):
    #      the quantity the TWC constant term is fitted to; its per-bin change
    #      is the direct "offset drift" (the TWC constant itself is not: the
    #      coupled 2-iteration fit maps a drift D of one board to -D/2 on that
    #      board and +D/4 on each partner).
    off = {}
    if len(roles) == 3:
        for r in roles:
            o = [toas[x] for x in roles if x != r]
            off[r] = toas[r] - 0.5 * (o[0] + o[1])
            R.add(stem, "tracks", "offset_all", off[r].mean(), role=r)

    # ---- full-run TWC (exactly bootstrap.py's fit) and its comparable summaries
    eff, corr = twc_fit(tots, toas, roles)
    ref_pts = {}
    for r in roles:
        p10, p50, p90 = np.percentile(tots[r], [10, 50, 90])
        ref_pts[r] = (p10, p50, p90)
        a2, a1, a0 = eff[r]
        P = np.poly1d(eff[r])
        R.add(stem, "twc_full", "a2", a2, role=r)
        R.add(stem, "twc_full", "a1", a1, role=r)
        R.add(stem, "twc_full", "a0", a0, role=r)
        R.add(stem, "twc_full", "tot_p10", p10, role=r)
        R.add(stem, "twc_full", "tot_med", p50, role=r)
        R.add(stem, "twc_full", "tot_p90", p90, role=r)
        R.add(stem, "twc_full", "slope_med", (2 * a2 * p50 + a1) * 1e3, role=r)      # ps per ns
        R.add(stem, "twc_full", "curvature", 2 * a2 * 1e6, role=r)                   # ps per ns^2
        R.add(stem, "twc_full", "corr_range", P(p90) - P(p10), role=r)               # ps
    sp = pair_sigmas(corr, roles)
    for pr, v in sp.items():
        R.add(stem, "twc_full", "sig_pair", v, role=pr)
    for r, v in solve_three_board(sp, roles).items():
        R.add(stem, "twc_full", "sig_role", v, role=r)
    R.add(stem, "twc_full", "n_events", len(d10))

    # ---- per bin of files: TWC refit + widths (only tracks with enough events)
    if len(d10) >= min_events_twc:
        bins = files // files_per_bin
        for b in np.unique(bins):
            m = bins == b
            nb = int(m.sum())
            if nb < MIN_BIN_EVENTS:
                continue
            tb = {r: tots[r][m] for r in roles}
            ob = {r: toas[r][m] for r in roles}
            R.add(stem, "twc_bin", "n_events", nb, bin=b)
            for r in off:
                R.add(stem, "twc_bin", "d_offset", off[r][m].mean() - off[r].mean(), role=r, bin=b)
            # (a) full-run TWC applied to this bin's events
            ca = {r: ob[r] + np.poly1d(eff[r])(tb[r]) for r in roles}
            spa = pair_sigmas(ca, roles)
            for pr, v in spa.items():
                R.add(stem, "twc_bin", "sig_pair_fullTWC", v, role=pr, bin=b)
            for r, v in solve_three_board(spa, roles).items():
                R.add(stem, "twc_bin", "sig_role_fullTWC", v, role=r, bin=b)
            # (b) TWC refit on this bin alone
            try:
                effb, cb = twc_fit(tb, ob, roles)
            except (np.linalg.LinAlgError, ValueError):
                continue
            spb = pair_sigmas(cb, roles)
            for pr, v in spb.items():
                R.add(stem, "twc_bin", "sig_pair_binTWC", v, role=pr, bin=b)
            for r, v in solve_three_board(spb, roles).items():
                R.add(stem, "twc_bin", "sig_role_binTWC", v, role=r, bin=b)
            for r in roles:
                p10, p50, p90 = ref_pts[r]
                Pb, Pf = np.poly1d(effb[r]), np.poly1d(eff[r])
                a2b, a1b, _ = effb[r]
                a2f, a1f, _ = eff[r]
                R.add(stem, "twc_bin", "d_slope_med", ((2 * a2b * p50 + a1b) - (2 * a2f * p50 + a1f)) * 1e3, role=r, bin=b)
                R.add(stem, "twc_bin", "d_corr_range", (Pb(p90) - Pb(p10)) - (Pf(p90) - Pf(p10)), role=r, bin=b)
                R.add(stem, "twc_bin", "d_corr_med", Pb(p50) - Pf(p50), role=r, bin=b)
                R.add(stem, "twc_bin", "d_corr_p10", Pb(p10) - Pf(p10), role=r, bin=b)
                R.add(stem, "twc_bin", "d_corr_p90", Pb(p90) - Pf(p90), role=r, bin=b)
    return R.rows


def cmd_summarize(a):
    tdir, mdir = a.tracks_dir, a.time_dir
    if not os.path.isdir(tdir):
        die("--tracks-dir not found: %s" % tdir)
    if not os.path.isdir(mdir):
        die("--time-dir not found: %s" % mdir)
    mfiles = natsorted(glob(os.path.join(mdir, "track_*.parquet")))
    tfiles = natsorted(glob(os.path.join(tdir, "track_*.parquet")))
    if not mfiles:
        die("no track_*.parquet in %s" % mdir)
    by_name = {os.path.basename(f): f for f in tfiles}
    jobs = []
    missing = 0
    for mf in mfiles:
        tf = by_name.get(os.path.basename(mf))
        if tf is None:
            missing += 1
            continue
        jobs.append((tf, mf, a.files_per_bin, a.min_events_twc))
    have_time = {os.path.basename(f) for f in mfiles}
    emptied = [(by_name[nm], None, a.files_per_bin, a.min_events_twc) for nm in by_name if nm not in have_time]
    if a.max_tracks:
        jobs = jobs[: a.max_tracks]
        emptied = emptied[: a.max_tracks]
    print("summarize: %d tracks with step-10 output, %d step-9 tracks that step 10 filtered to empty "
          "(counted as entering, 0 surviving), %d time files without a tracks counterpart skipped, %d workers"
          % (len(jobs), len(emptied), missing, a.workers))
    jobs = jobs + emptied
    rows = []
    done = 0
    failed = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(process_track, j): j for j in jobs}
        for fu in as_completed(futs):
            try:
                rows.extend(fu.result())
            except Exception as e:  # one bad track must not kill the run
                j = futs[fu]
                failed.append(os.path.basename(j[1] or j[0]))
                print("  !! failed: %s: %s" % (failed[-1], e), flush=True)
            done += 1
            if done % 100 == 0 or done == len(jobs):
                print("  %d/%d tracks" % (done, len(jobs)), flush=True)
    if failed:
        print("  %d track(s) FAILED and are missing from the parquet: %s%s"
              % (len(failed), ", ".join(failed[:10]), " ..." if len(failed) > 10 else ""))
    if not rows:
        die("no track produced any output")
    R = Rec()
    R.rows = rows
    df = R.frame()
    df.insert(0, "run", a.label)
    # run-level meta as rows too, so `plot` needs nothing but the parquet
    meta = Rec()
    meta.add("", "meta", "files_per_bin", a.files_per_bin)
    meta.add("", "meta", "min_events_twc", a.min_events_twc)
    meta.add("", "meta", "n_tracks", len(jobs))
    meta.add("", "meta", "n_tracks_emptied_by_step10", len(emptied))
    meta.add("", "meta", "n_tracks_failed", len(failed))
    meta.add("", "meta", "n_files", int(df.loc[df.section == "file", "file"].max()) + 1 if (df.section == "file").any() else 0)
    mdf = meta.frame()
    mdf.insert(0, "run", a.label)
    df = pd.concat([mdf, df], ignore_index=True)
    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, "%s_track_diagnostics.parquet" % a.label)
    df.to_parquet(out, index=False, compression="lz4")
    print("wrote %s (%d rows)" % (out, len(df)))


# --------------------------------------------------------------------------
# plot helpers
# --------------------------------------------------------------------------

class TimeMap:
    """file index -> hours since run start.  None if no mapping was given."""

    def __init__(self, t0, duration_min, n_files, file_times_csv):
        self.desc = None
        self.f = None
        self.n_files = n_files
        if file_times_csv:
            t = pd.read_csv(file_times_csv)
            t["time"] = pd.to_datetime(t["time"], utc=True)
            t = t.sort_values("file")
            h = (t["time"] - t["time"].iloc[0]).dt.total_seconds().to_numpy() / 3600.0
            fx = t["file"].to_numpy(float)
            self.f = lambda x: np.interp(x, fx, h)
            self.desc = ("hours since first raw file, from the measured file times in %s"
                         % os.path.basename(file_times_csv))
            self.axis_label = "hours since first raw file (measured, see caption)"
        elif duration_min:
            D = duration_min / 60.0
            N = float(n_files)
            self.f = lambda x: np.round((np.asarray(x, float) + 0.5) / N * D, 9) + 0.0
            self.desc = ("hours since run start%s, ASSUMING a constant data rate over %d files in %.0f min"
                         % ((" (%s)" % t0) if t0 else "", n_files, duration_min))
            self.axis_label = "hours since run start (constant-rate assumption, see caption)"

    def add_top_axis(self, ax):
        if self.f is None:
            return
        ax.set_xlim(-0.5, self.n_files - 0.5)
        f = self.f
        # secondary axis needs an inverse; the map is monotone so invert numerically
        xs = np.linspace(-0.5, self.n_files + 0.5, max(2 * self.n_files + 3, 200))
        ys = f(xs)
        inv = lambda y: np.interp(y, ys, xs)
        sec = ax.secondary_xaxis("top", functions=(f, inv))
        sec.set_xlabel(self.axis_label, fontsize=8.5)
        sec.tick_params(labelsize=8)


def meta_val(df, key, default=np.nan):
    s = df.loc[(df.section == "meta") & (df.key == key), "value"]
    return float(s.iloc[0]) if len(s) else default


def roles_of(df):
    r = df.loc[(df.section == "tracks") & (df.key == "row"), "role"].unique()
    return sorted(r)


def pairs_of(df):
    return sorted(df.loc[(df.section == "twc_full") & (df.key == "sig_pair"), "role"].unique())


def wide(df, section, key, role=None, index="file"):
    """track x file (or bin) matrix for one (section, key[, role])."""
    m = (df.section == section) & (df.key == key)
    if role is not None:
        m &= df.role == role
    sub = df.loc[m]
    if sub.empty:
        return None
    W = sub.pivot_table(index="track", columns=index, values="value", aggfunc="first")
    return None if W.empty else W


def median_band(W, min_n=5):
    """per-column median, p16, p84 across tracks (rows); NaN where < min_n tracks."""
    x = W.columns.to_numpy(float)
    v = W.to_numpy(float)
    n = np.isfinite(v).sum(axis=0)
    with np.errstate(all="ignore"):
        med = np.nanmedian(v, axis=0)
        lo = np.nanpercentile(v, 16, axis=0)
        hi = np.nanpercentile(v, 84, axis=0)
    bad = n < min_n
    med[bad] = lo[bad] = hi[bad] = np.nan
    return x, med, lo, hi, n


def band_plot(ax, x, med, lo, hi, color, label=None, lw=1.6):
    if not np.isfinite(med).any():
        ax.text(0.5, 0.5, "fewer than 5 tracks with data:\nno median / band drawn", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="#777777")
        return
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0)
    ax.plot(x, med, color=color, lw=lw, label=label)


CAP_FS = 8.6           # caption font size (pt)
CAP_LINE_IN = 0.158    # line pitch of the caption at CAP_FS (inches)
CAP_CHAR_IN = 0.066    # mean glyph advance of DejaVu Sans at CAP_FS (inches)


def finish(fig, out, cap, wspace=None, hspace=None, top=None, xlabel_in=0.62):
    """Wrap the caption to the figure width, reserve exactly the room it needs at
    the bottom (plus the x-label band), apply the requested panel spacing, save."""
    fw, fh = fig.get_size_inches()
    width_chars = max(60, int((fw - 0.5) / CAP_CHAR_IN))
    lines = textwrap.wrap(cap, width_chars)
    cap_h = len(lines) * CAP_LINE_IN + 0.12
    bottom = (cap_h + xlabel_in) / fh
    kw = {"bottom": min(bottom, 0.6)}
    if wspace is not None:
        kw["wspace"] = wspace
    if hspace is not None:
        kw["hspace"] = hspace
    if top is not None:
        kw["top"] = top
    fig.subplots_adjust(**kw)
    fig.text(0.25 / fw, 0.08 / fh, "\n".join(lines), fontsize=CAP_FS, va="bottom", ha="left",
             family="DejaVu Sans", linespacing=1.25)
    out = save_figure(fig, out, dpi=130)
    plt.close(fig)
    return out


ROLE_COL = {"dut": "#c0392b", "ref": "#1f77b4", "trig": "#2ca02c", "extra": "#8e44ad"}


def rc(role):
    return ROLE_COL.get(role, "#555555")


def n_distinct_pixels(df, r):
    t = df.loc[(df.section == "tracks") & (df.role == r) & df.key.isin(["row", "col"])]
    if t.empty:
        return 0
    rc = t.pivot_table(index="track", columns="key", values="value", aggfunc="first")
    return int(rc.drop_duplicates().shape[0])


def label_role(df, r):
    """'dut (R,C)' if the pixel is unique across tracks (fixed board), else 'dut (varying pixel)'."""
    n = n_distinct_pixels(df, r)
    return "%s (%d distinct pixel%s in this combo)" % (r, n, "" if n == 1 else "s")


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def fig_events_per_file(df, out, tm):
    Win = wide(df, "file", "n_in")
    Wout = wide(df, "file", "n_out")
    if Win is None or Wout is None:
        return None
    # a track with entering rows but no survivor in a file is a survival of 0, not a gap
    Wout = Wout.reindex(index=Win.index, columns=Win.columns).fillna(0).where(Win.notna())
    x = Win.columns.to_numpy(float)
    tin = np.nansum(Win.to_numpy(float), axis=0)
    tout = np.nansum(Wout.to_numpy(float), axis=0)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11.5, 8.2), sharex=True,
                                 gridspec_kw={"height_ratios": [1.15, 1]})
    a1.plot(x, tin, "o-", ms=3, lw=1.2, color="#555555", label="entering step 10 (step-9 rows)")
    a1.plot(x, tout, "o-", ms=3, lw=1.2, color="#c0392b", label="surviving step 10")
    a1.set_ylabel("events in this file, all tracks pooled")
    a1.set_title("Events per raw file entering and surviving the TDC cuts", fontsize=11)
    a1.legend(fontsize=9, loc="lower left")
    a1.grid(alpha=0.3)
    a1.set_ylim(bottom=0)
    tm.add_top_axis(a1)
    with np.errstate(all="ignore"):
        pooled = tout / tin
        R = Wout.to_numpy(float) / Win.to_numpy(float)
        R[Win.to_numpy(float) < 20] = np.nan
    a2.plot(x, pooled, "-", color="#c0392b", lw=1.8, label="pooled: surviving / entering")
    xx, med, lo, hi, n = median_band(pd.DataFrame(R, index=Win.index, columns=Win.columns))
    a2.fill_between(xx, lo, hi, color="#c0392b", alpha=0.15, lw=0, label="16-84% of tracks")
    a2.set_ylabel("survival fraction")
    a2.set_xlabel("raw file index (loop_N)")
    a2.grid(alpha=0.3)
    a2.legend(fontsize=9, loc="lower left")
    lo_all, hi_all = np.nanmin(pooled), np.nanmax(pooled)
    a2.set_ylim(max(0, lo_all - 0.08), min(1.0, hi_all + 0.08))
    n_empt = int(meta_val(df, "n_tracks_emptied_by_step10", 0))
    cap = ("TOP: for each raw file, the number of (track, event) rows across all tracks of this combo that "
           "entered step 10 (grey, = step-9 output, including the %d track(s) step 10 filtered to nothing) and "
           "survived its ToT windows / TOA window / TOA-correlation cut (red). The DAQ closes a raw file at a "
           "fixed size, so a flat grey line means a steady hit rate and beam; a step means the beam or the DAQ "
           "changed. BOTTOM: the surviving fraction per file, all tracks pooled (line) and the 16-84%% spread over "
           "individual tracks (band; a track with entering rows but no survivor in a file counts as 0; tracks "
           "with <20 entering events in a file are left out of the band). A drift here means the per-file ToT "
           "windows, the whole-track TOA-correlation cut, or the underlying distributions moved. The last file is "
           "normally a partial chunk (the run stopped mid-file), so its counts are low and its ratio noisy. Pooled "
           "survival over the run: %.1f%% (%d of %d rows)."
           % (n_empt, 100 * tout.sum() / tin.sum(), int(tout.sum()), int(tin.sum())))
    if tm.desc:
        cap += " Top axis: " + tm.desc + "."
    a1.set_xlim(-0.5, int(meta_val(df, "n_files", len(x))) - 0.5)
    return finish(fig, out, cap, hspace=0.12, top=0.90)


def fig_cal_drift(df, out, tm):
    roles = roles_of(df)
    fig, axs = plt.subplots(len(roles), 1, figsize=(11.5, 3.0 * len(roles) + 2.2), sharex=True, squeeze=False)
    axs = axs.ravel()
    typ = {}
    for ax, r in zip(axs, roles):
        W = wide(df, "file", "cal_mean", role=r)
        ref = df.loc[(df.section == "tracks") & (df.key == "cal_mean_all") & (df.role == r)].set_index("track")["value"]
        if W is None:
            continue
        D = W.sub(ref.reindex(W.index), axis=0)
        x, med, lo, hi, n = median_band(D)
        band_plot(ax, x, med, lo, hi, rc(r))
        ax.axhline(0, color="k", lw=0.7)
        ax.set_ylabel("Δ mean CAL [codes]\n(file − run average, per track)")
        typ[r] = float(np.nanmedian(ref))
        ax.set_title("%s - run-average CAL of its pixels: %.0f codes (bin %.1f ps)"
                     % (label_role(df, r), typ[r], 3125.0 / typ[r]), fontsize=10)
        ax.grid(alpha=0.3)
    axs[-1].set_xlabel("raw file index (loop_N)")
    tm.add_top_axis(axs[0])
    cap = ("For each board and each raw file: the mean CAL code of the surviving events of a track in that "
           "file minus the same track's run-average CAL, then the median (line) and 16-84% spread (band) over "
           "all tracks. On step 10's default cut-then-convert path this is exactly the CAL it used to set the "
           "TDC bin for that file (bin = 3.125 ns / mean CAL of the surviving rows, per file); with "
           "--convert-first step 10 averaged over all rows before the cuts instead. CAL counts delay-line cells "
           "flipping in the fixed 3.125 ns reference, so a drift here is the delay line (the bin) speeding up or "
           "slowing down as the chip warms or the supply moves, not the clock. Scale: at CAL ~155 one code "
           "changes the bin by 0.13 ps, which is up to ~80 ps at the far end of the 12.5 ns TOA range - a slow "
           "drift is absorbed by the per-file mean, a change WITHIN a file is not. A flat line at 0 within a "
           "fraction of a code is what a stable run looks like.")
    if tm.desc:
        cap += " Top axis: " + tm.desc + "."
    axs[0].set_xlim(-0.5, int(meta_val(df, "n_files", 1)) - 0.5)
    return finish(fig, out, cap, hspace=0.32, top=0.90)



CAL_REF_FILES = 5     # the t=0 reference of the per-pixel CAL evolution = mean over the first N files
MIN_CELL_EVENTS = 20  # a (pixel, file) mean built from fewer events than this is not shown (partial last file etc.)


def per_pixel_file_matrix(df, role, key, section="file"):
    """(pixel-index sorted by row,col) x file matrix of the mean over tracks sharing that
    pixel of a per-(track,file) quantity; returns (matrix, row_of_pixel, col_of_pixel)."""
    W = wide(df, section, key, role=role)
    N = wide(df, "file", "n_out")
    if W is None or N is None:
        return None, None, None
    row = df.loc[(df.section == "tracks") & (df.role == role) & (df.key == "row")].set_index("track")["value"]
    col = df.loc[(df.section == "tracks") & (df.role == role) & (df.key == "col")].set_index("track")["value"]
    pix = (row.reindex(W.index) * GRID + col.reindex(W.index)).astype("Int64")
    N = N.reindex(index=W.index, columns=W.columns).fillna(0)
    # event-weighted mean over the tracks that share the pixel, and the events behind each cell
    num = (W * N).assign(pix=pix.to_numpy()).dropna(subset=["pix"]).groupby("pix").sum(min_count=1)
    den = N.assign(pix=pix.to_numpy()).dropna(subset=["pix"]).groupby("pix").sum()
    M = (num / den).where(den >= MIN_CELL_EVENTS)
    M = M.reindex(range(GRID * GRID))
    prow = np.arange(GRID * GRID) // GRID
    pcol = np.arange(GRID * GRID) % GRID
    return M, prow, pcol


def fig_cal_pixels_vs_time(df, out, tm):
    """All pixels of each board: per-file mean CAL relative to the start of the run."""
    roles = roles_of(df)
    n_files = int(meta_val(df, "n_files", 0)) or int(df.loc[df.section == "file", "file"].max()) + 1
    fig, axs = plt.subplots(len(roles), 3, figsize=(17.5, 4.9 * len(roles) + 1.6), squeeze=False,
                            gridspec_kw={"width_ratios": [1.35, 1.35, 1.0]})
    cmap = plt.get_cmap("coolwarm")
    offenders = []
    for i, r in enumerate(roles):
        M, prow, pcol = per_pixel_file_matrix(df, r, "cal_mean")
        if M is None:
            for ax in axs[i]:
                ax.axis("off")
            continue
        files = M.columns.to_numpy(float)
        V = M.to_numpy(float)                                   # pixel x file
        ref_cols = files < CAL_REF_FILES
        with np.errstate(all="ignore"):
            ref = np.nanmean(V[:, ref_cols], axis=1)
        D = V - ref[:, None]
        have = np.isfinite(D).sum(axis=1) > 0
        # (a) one line per pixel
        ax = axs[i, 0]
        for k in np.where(have)[0]:
            ax.plot(files, D[k], color=cmap((pcol[k] + 0.5) / GRID), lw=0.6, alpha=0.45)
        med = np.nanmedian(D[have], axis=0)
        ax.plot(files, med, color="k", lw=1.8, label="median over pixels")
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(CAL_REF_FILES - 0.5, color="k", lw=0.6, ls=":")
        ax.set_xlim(-0.5, n_files - 0.5)
        ax.set_ylabel("mean CAL of the pixel in this file\n− its mean over the first %d files  [codes]" % CAL_REF_FILES)
        ax.set_title("%s - %d pixels, one line each (colour = column, blue col 0 .. red col 15)"
                     % (r, int(have.sum())), fontsize=9.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
        if i == len(roles) - 1:
            ax.set_xlabel("raw file index (loop_N)")
        tm.add_top_axis(ax)
        # (b) pixel x file heat map
        ax = axs[i, 1]
        fin = np.abs(D[np.isfinite(D)])
        vmax = max(0.5, float(np.percentile(fin, 99.5))) if fin.size else 0.5
        im = ax.imshow(D, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       extent=[files.min() - 0.5, files.max() + 0.5, -0.5, GRID * GRID - 0.5],
                       interpolation="nearest")
        ax.set_yticks(np.arange(0, GRID * GRID, GRID * 2))
        ax.set_yticklabels(["row %d" % (k // GRID) for k in np.arange(0, GRID * GRID, GRID * 2)], fontsize=8)
        ax.set_ylabel("pixel (row-major: row*16 + col)")
        ax.set_title("%s - same numbers as a pixel x file image" % r, fontsize=9.5)
        if i == len(roles) - 1:
            ax.set_xlabel("raw file index (loop_N)")
        cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("Δ mean CAL [codes]", fontsize=8.5)
        cb.ax.tick_params(labelsize=8)
        # (c) map of the largest excursion per pixel
        ax = axs[i, 2]
        with np.errstate(all="ignore"):
            exc = np.nanmax(np.abs(D), axis=1)
        img = np.full((GRID, GRID), np.nan)
        img[prow[have], pcol[have]] = exc[have]
        im = pixel_map(ax, img, cmap="magma", vmin=0, vmax=max(0.5, float(np.nanpercentile(exc[have], 98))), nan_blank=True)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("largest |Δ mean CAL| over the run [codes]", fontsize=8.5)
        cb.ax.tick_params(labelsize=8)
        ax.set_title("%s - largest excursion per pixel\nmedian %.2f, max %.2f codes" % (r, np.nanmedian(exc[have]), np.nanmax(exc[have])), fontsize=9.5)
        bad = [(int(prow[k]), int(pcol[k]), float(exc[k])) for k in np.where(have)[0] if exc[k] > 1.0]
        if bad:
            offenders.append("%s: %s" % (r, ", ".join("(%d,%d) %.1f" % b for b in sorted(bad, key=lambda t: -t[2])[:12])
                                            + (" ..." if len(bad) > 12 else "")))
    cap = ("Sanity check of the per-file CAL means step 10 uses, pixel by pixel. For every board and every pixel "
           "of it that a track uses: the mean CAL code of that pixel's surviving events in each raw file, minus "
           "the same pixel's mean over the first %d files (~ the start of the run, so t=0 is the origin and the "
           "evolution reads forwards; the first-file mean alone would be a noisy origin - a file holds only "
           "~50-100 events per pixel, so a single file scatters by ~0.1-0.2 code by statistics alone). Where a "
           "pixel is used by several tracks its values are event-weighted averages; a (pixel, file) cell built "
           "from fewer than %d events (e.g. the partial last file) is left blank. LEFT: one line per pixel, coloured by "
           "column (blue = col 0, physical right; red = col 15, physical left), black = median over pixels. "
           "MIDDLE: the same matrix as an image, pixels row-major from the bottom (row 0) up, so a misbehaving "
           "pixel is a horizontal streak and a run-wide event is a vertical stripe. RIGHT: the largest |Δ| each "
           "pixel reached during the run, on the array. A healthy chip: all lines within a few tenths of a code, "
           "no streaks, a flat dark map. Pixels whose excursion exceeds 1 code (bin change 0.13 ps, i.e. up to "
           "~80 ps at the far end of the TOA range) are listed here: %s." % (CAL_REF_FILES, MIN_CELL_EVENTS, "; ".join(offenders) if offenders else "none"))
    if tm.desc:
        cap += " Top axis: " + tm.desc + "."
    return finish(fig, out, cap, wspace=0.30, hspace=0.36, top=0.93)

def fig_toa_tot_drift(df, out, tm):
    roles = roles_of(df)
    pairs = pairs_of(df)
    n = max(len(roles), len(pairs))
    fig, axs = plt.subplots(2, n, figsize=(4.9 * n + 1.5, 8.6), sharex=True, squeeze=False)
    for j, r in enumerate(roles):
        ax = axs[0, j]
        W = wide(df, "file", "tot_mean", role=r)
        ref = df.loc[(df.section == "tracks") & (df.key == "tot_mean_all") & (df.role == r)].set_index("track")["value"]
        if W is not None:
            D = W.sub(ref.reindex(W.index), axis=0)
            x, med, lo, hi, nn = median_band(D)
            band_plot(ax, x, med, lo, hi, rc(r))
        ax.axhline(0, color="k", lw=0.7)
        ax.set_title("%s\nrun-mean TOT %.2f ns" % (r, np.nanmedian(ref) / 1e3), fontsize=10)
        if j == 0:
            ax.set_ylabel("Δ mean TOT [ps]\n(file − run, per track)")
        ax.grid(alpha=0.3)
        tm.add_top_axis(ax)
    for j in range(len(roles), n):
        axs[0, j].axis("off")
    for j, pr in enumerate(pairs):
        ax = axs[1, j]
        W = wide(df, "file", "dtoa_mean", role=pr)
        ref = df.loc[(df.section == "tracks") & (df.key == "dtoa_mean_all") & (df.role == pr)].set_index("track")["value"]
        if W is not None:
            D = W.sub(ref.reindex(W.index), axis=0)
            x, med, lo, hi, nn = median_band(D)
            band_plot(ax, x, med, lo, hi, "#444444")
        ax.axhline(0, color="k", lw=0.7)
        ax.set_title("TOA(%s) − TOA(%s)" % tuple(pr.split("-")), fontsize=10)
        if j == 0:
            ax.set_ylabel("Δ mean TOA difference [ps]\n(file − run, per track; before TWC)")
        ax.set_xlabel("raw file index (loop_N)")
        ax.grid(alpha=0.3)
    for j in range(len(pairs), n):
        axs[1, j].axis("off")
    cap = ("TOP ROW, one panel per board: the MEAN TOT (in ps, after step 10's conversion) of a track's "
           "surviving events in each raw file minus that track's run mean; median over tracks (line) and "
           "16-84% spread (band). (The mean, not the median: TOT is quantised in ~40 ps steps, so a per-file "
           "median only hops between codes.) TOT is the charge proxy, so a trend here is the sensor's signal or "
           "the discriminator threshold moving (HV, temperature, radiation). BOTTOM ROW, one panel per board "
           "pair: the mean raw TOA difference between the two boards per file, again relative to each track's "
           "run value. Anything common to all boards cancels; what remains is the RELATIVE timing offset "
           "between the two chips (clock phase, cable/latency, temperature of one chip). A static per-track "
           "offset is absorbed by the TWC's constant term; a drift within the run is not - step 12 fits one "
           "constant per track over the whole run, so a drift adds its RMS in quadrature to the pair widths "
           "(a linear end-to-end drift D adds D/sqrt(12); see the resolution-proxy figure).")
    if tm.desc:
        cap += " Top axis: " + tm.desc + "."
    for ax in axs.ravel():
        ax.set_xlim(-0.5, int(meta_val(df, "n_files", 1)) - 0.5)
    return finish(fig, out, cap, wspace=0.32, hspace=0.30, top=0.88)


def curves_data(df, roles):
    """per role: (tot grid in ps, matrix track x grid of P(tot) - P(tot_med_global), colour values, tot_med_global)."""
    out = {}
    T = df.loc[df.section == "twc_full"]
    for r in roles:
        sub = T.loc[T.role == r].pivot_table(index="track", columns="key", values="value", aggfunc="first")
        if sub.empty or not {"a2", "a1", "a0", "tot_p10", "tot_p90", "tot_med"} <= set(sub.columns):
            continue
        lo = np.nanpercentile(sub["tot_p10"], 10)
        hi = np.nanpercentile(sub["tot_p90"], 90)
        tmed = float(np.nanmedian(sub["tot_med"]))
        grid = np.linspace(lo, hi, 60)
        A = sub[["a2", "a1", "a0"]].to_numpy(float)
        Y = A[:, 0:1] * grid[None, :] ** 2 + A[:, 1:2] * grid[None, :] + A[:, 2:3]
        Y0 = A[:, 0] * tmed ** 2 + A[:, 1] * tmed + A[:, 2]
        Y = Y - Y0[:, None]
        # draw each track only inside its own central-80% TOT range: outside it the
        # polynomial is extrapolation and would fan out for no physical reason
        p10 = sub["tot_p10"].to_numpy(float)[:, None]
        p90 = sub["tot_p90"].to_numpy(float)[:, None]
        Y[(grid[None, :] < p10) | (grid[None, :] > p90)] = np.nan
        col = df.loc[(df.section == "tracks") & (df.role == r) & (df.key == "col")].set_index("track")["value"]
        row = df.loc[(df.section == "tracks") & (df.role == r) & (df.key == "row")].set_index("track")["value"]
        out[r] = dict(grid=grid, Y=Y, tracks=sub.index.to_numpy(), col=col.reindex(sub.index).to_numpy(),
                      row=row.reindex(sub.index).to_numpy(), tmed=tmed, lo=lo, hi=hi)
    return out


def fig_twc_curves(df, out, tm):
    roles = roles_of(df)
    cd = curves_data(df, roles)
    if not cd:
        return None
    fig, axs = plt.subplots(1, len(roles), figsize=(5.4 * len(roles) + 1.0, 6.4), squeeze=False)
    axs = axs.ravel()
    cmap = plt.get_cmap("coolwarm")
    for ax, r in zip(axs, roles):
        if r not in cd:
            ax.axis("off")
            continue
        d = cd[r]
        g = d["grid"] / 1e3
        ncol = len(np.unique(d["col"][np.isfinite(d["col"])]))
        for i in range(d["Y"].shape[0]):
            c = cmap((d["col"][i] + 0.5) / 16.0) if np.isfinite(d["col"][i]) else "#888888"
            ax.plot(g, d["Y"][i], color=c, lw=0.6, alpha=0.25 if ncol > 1 else 0.6)
        med = np.nanmedian(d["Y"], axis=0)
        ax.plot(g, med, color="k", lw=2.2, label="median over %d tracks" % d["Y"].shape[0])
        ax.axvline(d["tmed"] / 1e3, color="k", lw=0.7, ls=":")
        ax.set_xlabel("TOT [ns]")
        ax.set_ylabel("TWC correction added to TOA [ps],\nrelative to its value at the median TOT")
        ax.set_title(label_role(df, r), fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5, loc="upper right")
        if ncol > 1:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-0.5, 15.5))
            cb = plt.colorbar(sm, ax=ax, fraction=0.05, pad=0.02)
            cb.set_label("pixel column on this board (0 = physical right)", fontsize=8.5)
            cb.ax.tick_params(labelsize=8)
    cap = ("One curve per track: the effective time-walk correction that step 12 would add to this board's "
           "TOA as a function of the board's TOT (the two iterations of its 2nd-order fit summed, on the same "
           "events a step-12 run without --neighbor_cut uses), shifted so every curve is 0 at the dotted line "
           "= the median over tracks of each track's median TOT - the constant term is a per-track "
           "cable/latency offset and is not comparable, the SHAPE is. Each curve is drawn only over its own "
           "track's central 80% of TOT (10th to 90th percentile); the panel spans the 10th percentile over "
           "tracks of those lower ends to the 90th percentile of the upper ends. Black: median curve. Colour: "
           "the pixel column on that board (only meaningful for the board whose pixel varies across tracks), "
           "running from blue (col 0, physical RIGHT edge) to red (col 15, physical LEFT edge); the TDC power "
           "domains meet at col 7|8, the white middle of the scale. Tight bundle "
           "= one correction would serve the whole array; a colour gradient = the correction depends on where "
           "on the chip; a wide bundle with no colour order = pixel-to-pixel scatter or fit noise. Note the "
           "correction for one board is fitted against the average of the other two boards, so their jitter "
           "and their own time walk feed into it: the curves are what the pipeline applies, not the intrinsic "
           "time walk of one chip alone.")
    return finish(fig, out, cap, wspace=0.55, top=0.92)


def per_pixel_grid(df, role, key, section="twc_full", weight_key="n_events"):
    """16x16 arrays: weighted mean over tracks at that pixel, spread over tracks, n tracks."""
    T = df.loc[df.section == section]
    v = T.loc[(T.role == role) & (T.key == key)].set_index("track")["value"]
    w = T.loc[T.key == weight_key].set_index("track")["value"]
    row = df.loc[(df.section == "tracks") & (df.role == role) & (df.key == "row")].set_index("track")["value"]
    col = df.loc[(df.section == "tracks") & (df.role == role) & (df.key == "col")].set_index("track")["value"]
    d = pd.DataFrame({"v": v, "w": w.reindex(v.index), "row": row.reindex(v.index), "col": col.reindex(v.index)}).dropna()
    mean = np.full((GRID, GRID), np.nan)
    sd = np.full((GRID, GRID), np.nan)
    n = np.zeros((GRID, GRID))
    for (r, c), g in d.groupby(["row", "col"]):
        r, c = int(r), int(c)
        ww = g["w"].to_numpy(float)
        vv = g["v"].to_numpy(float)
        mean[r, c] = np.average(vv, weights=ww) if ww.sum() > 0 else np.nan
        sd[r, c] = np.std(vv, ddof=1) if len(vv) > 1 else np.nan
        n[r, c] = len(vv)
    return mean, sd, n


def fig_twc_array_maps(df, out, tm):
    roles = roles_of(df)
    # only boards whose pixel varies across tracks are worth mapping
    var_roles = [r for r in roles if df.loc[(df.section == "tracks") & (df.role == r) & (df.key == "row"), "value"].nunique() > 1
                 or df.loc[(df.section == "tracks") & (df.role == r) & (df.key == "col"), "value"].nunique() > 1]
    if not var_roles:
        return None
    specs = [("slope_med", "slope of the correction at the median TOT [ps per ns of TOT]"),
             ("corr_range", "correction removed across the central 80% of TOT [ps]"),
             ("curvature", "curvature 2·a2 [ps per ns²]")]
    fig, axs = plt.subplots(len(specs) + 1, len(var_roles), figsize=(5.4 * len(var_roles) + 0.6, 4.6 * (len(specs) + 1) + 1.0),
                            squeeze=False)
    for j, r in enumerate(var_roles):
        for i, (key, lab) in enumerate(specs):
            ax = axs[i, j]
            mean, sd, n = per_pixel_grid(df, r, key)
            fin = mean[np.isfinite(mean)]
            if fin.size == 0:
                ax.axis("off")
                continue
            lo, hi = np.percentile(fin, [2, 98])
            im = pixel_map(ax, mean, cmap="viridis", vmin=lo, vmax=hi, nan_blank=True)
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=8)
            ax.set_title("%s - %s\nmedian %.1f, 16-84%%: %.1f .. %.1f  (%d pixels)"
                         % (r, lab, np.median(fin), np.percentile(fin, 16), np.percentile(fin, 84), fin.size), fontsize=9)
        ax = axs[len(specs), j]
        mean, sd, n = per_pixel_grid(df, r, "slope_med")
        fin = sd[np.isfinite(sd)]
        if fin.size:
            im = pixel_map(ax, sd, cmap="magma", vmin=0, vmax=np.percentile(fin, 98), nan_blank=True)
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=8)
            ax.set_title("%s - spread of the slope over the tracks that share a pixel [ps/ns]\n"
                         "median %.1f (pixels with >=2 tracks: %d; tracks per pixel: median %.0f)"
                         % (r, np.median(fin), fin.size, np.median(n[n > 0])), fontsize=9)
        else:
            ax.axis("off")
    cap = ("Maps over the pixel array of the board(s) whose pixel varies from track to track, in the physical "
           "orientation (pixel (0,0) bottom-right, wire-bond pads at the bottom). Each pixel shows the "
           "event-weighted mean over the tracks that use it; colour scales are clipped at the 2nd-98th percentile "
           "of the pixel values. ROW 1: the slope of the TWC polynomial at each track's own median TOT (ps of "
           "correction per ns of TOT). Sign: the correction is ADDED to the TOA and fitted to (mean of the "
           "other two boards) - (this board), so a positive slope means this board's TOA comes EARLIER with "
           "increasing TOT relative to its partners - the usual time walk; the correction slopes the opposite "
           "way to the raw TOA-vs-TOT trend. "
           "ROW 2: how many ps of correction the polynomial spans between the 10th and 90th percentile of the "
           "board's TOT - the size of the time walk being removed. ROW 3: the curvature term. ROW 4: the "
           "standard deviation of the slope over the tracks that share the same pixel (same pixel, different "
           "partner pixels on the other boards) - if this is as large as the pixel-to-pixel structure above, "
           "the structure is fit noise, not the chip. Grey pixels have no track. A uniform map means one "
           "correction would serve the whole chip; a left/right step at the col 7|8 boundary would follow the "
           "power domains; a smooth gradient would follow temperature or the beam's charge distribution. "
           "CAVEAT of the coupled fit: each board is corrected against the mean of the other two, so a real "
           "row/column feature of one chip is partly mirrored, with the opposite sign, onto its partners. "
           "Compare the same board across the other combos it appears in: a feature that persists with "
           "different partners belongs to that chip; one that flips sign or migrates to a partner is the "
           "fit's bookkeeping, not the silicon.")
    return finish(fig, out, cap, wspace=0.35, hspace=0.42, top=0.95, xlabel_in=0.7)


def fig_twc_board_compare(df, out, tm):
    roles = roles_of(df)
    T = df.loc[df.section == "twc_full"]
    specs = [("slope_med", "slope at median TOT [ps / ns]"),
             ("corr_range", "correction across central 80% of TOT [ps]"),
             ("curvature", "curvature 2·a2 [ps / ns²]"),
             ("sig_role", "resolution proxy after TWC [ps]")]
    fig, axs = plt.subplots(1, len(specs), figsize=(4.4 * len(specs) + 0.8, 5.6), squeeze=False)
    axs = axs.ravel()
    lines = []
    for ax, (key, lab) in zip(axs, specs):
        data, labels, cols = [], [], []
        for r in roles:
            v = T.loc[(T.role == r) & (T.key == key), "value"].to_numpy(float)
            v = v[np.isfinite(v)]
            if v.size:
                data.append(v)
                labels.append("%s\n(n=%d)" % (r, v.size))
                cols.append(rc(r))
                if key in ("slope_med", "sig_role"):
                    lines.append("%s %s %.0f %s (16-84%%: %.0f..%.0f)" % (r, "slope" if key == "slope_med" else "sigma-proxy",
                                                                          np.median(v), "ps/ns" if key == "slope_med" else "ps",
                                                                          np.percentile(v, 16), np.percentile(v, 84)))
        if not data:
            ax.axis("off")
            continue
        bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True, widths=0.55, whis=[5, 95])
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_alpha(0.35)
        for med in bp["medians"]:
            med.set_color("k")
        ax.set_title(lab, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    cap = ("Distribution over tracks of the same TWC summaries, one box per board (box = 25-75%, whiskers "
           "= 5-95%, line = median). Boards are compatible if their boxes overlap; a board whose slope or range "
           "differs is a chip with a different time-walk behaviour (threshold, gain, sensor) - or, for the "
           "board whose pixel is fixed in this combo, a single pixel that need not be typical. Right: the "
           "resolution proxy per board (robust width of the TWC-corrected pairwise TOA differences, 3-board "
           "solve; NOT the step-12 GMM/bootstrap number, only a stand-in; tracks whose 3-board solve is imaginary "
           "are excluded from that box, hence its smaller n). Medians over tracks: " + "; ".join(lines) + ".")
    return finish(fig, out, cap, wspace=0.28, top=0.92)


def bin_centers(df, bins):
    fpb = meta_val(df, "files_per_bin", 5)
    return (np.asarray(bins, float) + 0.5) * fpb - 0.5


def fig_twc_vs_time(df, out, tm):
    roles = roles_of(df)
    if not (df.section == "twc_bin").any():
        return None
    specs = [("d_slope_med", "Δ slope at median TOT [ps / ns]"),
             ("d_corr_range", "Δ correction across central 80% of TOT [ps]"),
             ("d_offset", "Δ raw offset vs mean of the other two boards [ps]\n(before TWC; what the constant term is fitted to)")]
    fig, axs = plt.subplots(len(specs), len(roles), figsize=(4.9 * len(roles) + 1.5, 3.1 * len(specs) + 2.6),
                            sharex=True, squeeze=False)
    ntr = None
    for j, r in enumerate(roles):
        for i, (key, lab) in enumerate(specs):
            ax = axs[i, j]
            W = wide(df, "twc_bin", key, role=r, index="bin")
            if W is None:
                ax.axis("off")
                continue
            ntr = W.shape[0]
            b, med, lo, hi, n = median_band(W)
            x = bin_centers(df, b)
            band_plot(ax, x, med, lo, hi, rc(r))
            ax.axhline(0, color="k", lw=0.7)
            if i == 0:
                ax.set_title(r, fontsize=10)
                tm.add_top_axis(ax)
            if j == 0:
                ax.set_ylabel(lab, fontsize=9)
            ax.grid(alpha=0.3)
        axs[-1, j].set_xlabel("raw file index (bin centre)")
    fpb = int(meta_val(df, "files_per_bin", 5))
    mev = int(meta_val(df, "min_events_twc", 0))
    cap = ("Is the time-walk correction stable along the run?  For every track with at least %d surviving "
           "events, the TWC fit was redone on each bin of %d consecutive raw files (>= %d events per bin) and "
           "compared with the same track's full-run fit, at the full-run reference TOTs. Median over tracks "
           "(line) and 16-84%% spread (band); %d tracks used. ROW 1: change of the slope at the median TOT. "
           "ROW 2: change of the correction span across the central 80%% of TOT - if this stays within a few "
           "ps the correction SHAPE is stable and one full-run fit is adequate. ROW 3: change, per bin, of the "
           "board's mean raw TOA minus the mean of the other two boards' TOA (before any TWC) relative to the "
           "run value - the drift of the offset the TWC's constant term is fitted to. It is shown directly, "
           "not via the fitted constant, because the coupled 2-iteration fit maps a drift D of one board to "
           "-D/2 on that board and +D/4 on each partner (the fitted change at the median TOT is stored in the "
           "parquet as d_corr_med for anyone who wants it). A trend in row 3 with flat rows 1-2 means the "
           "clocks/latencies moved, not the time walk. Bins with fewer than 5 tracks are blank."
           % (mev, fpb, MIN_BIN_EVENTS, ntr or 0))
    if tm.desc:
        cap += " Top axis: " + tm.desc + "."
    for ax in axs.ravel():
        ax.set_xlim(-0.5, int(meta_val(df, "n_files", 1)) - 0.5)
    return finish(fig, out, cap, wspace=0.28, hspace=0.22, top=0.90)


def fig_resolution_proxy_vs_time(df, out, tm):
    roles = roles_of(df)
    if not (df.section == "twc_bin").any():
        return None
    fig, axs = plt.subplots(1, len(roles), figsize=(4.9 * len(roles) + 1.5, 5.6), sharey=True, squeeze=False)
    axs = axs.ravel()
    full = df.loc[(df.section == "twc_full") & (df.key == "sig_role")]
    ntr = 0
    for ax, r in zip(axs, roles):
        Wa = wide(df, "twc_bin", "sig_role_fullTWC", role=r, index="bin")
        Wb = wide(df, "twc_bin", "sig_role_binTWC", role=r, index="bin")
        if Wa is None:
            ax.axis("off")
            continue
        ntr = Wa.shape[0]
        b, med, lo, hi, n = median_band(Wa)
        x = bin_centers(df, b)
        band_plot(ax, x, med, lo, hi, rc(r), label="full-run TWC applied to the bin")
        if Wb is not None:
            b2, med2, lo2, hi2, n2 = median_band(Wb)
            ax.plot(bin_centers(df, b2), med2, color="k", lw=1.4, ls="--", label="TWC refitted in the bin")
        fr = full.loc[(full.role == r) & full.track.isin(Wa.index), "value"]
        fr = fr[np.isfinite(fr)]
        if len(fr):
            ax.axhline(np.median(fr), color=rc(r), lw=1.0, ls=":", label="full run, same tracks: %.1f ps" % np.median(fr))
        ax.set_title(r, fontsize=10)
        ax.set_xlabel("raw file index (bin centre)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
        tm.add_top_axis(ax)
    axs[0].set_ylabel("resolution proxy [ps]\n(robust width of TWC-corrected ΔTOA, 3-board solve)")
    fpb = int(meta_val(df, "files_per_bin", 5))
    cap = ("Does the timing hold along the run, and does a drifting TWC cost anything?  Per bin of %d raw "
           "files and per track (%d tracks with enough events), the pairwise TOA differences after time-walk "
           "correction are reduced to a robust width (IQR/1.349) and solved for the three boards; shown is "
           "the median over tracks (line) and 16-84%% spread (band). Solid/band: the FULL-RUN TWC applied to "
           "that bin's events - what step 12 effectively does. Dashed: the TWC re-fitted inside the bin. If "
           "the two agree, the correction's drift is irrelevant to the resolution; a dashed line systematically "
           "lower is an UPPER bound on what a time-dependent TWC could recover, because the in-bin refit also "
           "fits noise (a few tenths of a ps at the smallest bins is that in-sample optimism, not drift). "
           "Dotted: the same proxy over the whole run for the same tracks. Bin values sitting slightly BELOW "
           "it are expected: a slow drift of the inter-board offset (see the ΔTOA and TWC-vs-time figures) adds "
           "in quadrature over the whole run but not within a bin - the gap is that drift's cost. This proxy is "
           "not the step-12 number (GMM FWHM, bootstrap): read it for trends and comparisons, not for the "
           "absolute value." % (fpb, ntr))
    if tm.desc:
        cap += " Top axis: " + tm.desc + "."
    for ax in axs.ravel():
        ax.set_xlim(-0.5, int(meta_val(df, "n_files", 1)) - 0.5)
    return finish(fig, out, cap, wspace=0.22, top=0.90)


FIGS = [
    ("events_per_file.png", "How many events per raw file enter and survive the TDC cuts?", fig_events_per_file),
    ("cal_drift.png", "Does the TDC bin (mean CAL) drift along the run?", fig_cal_drift),
    ("cal_pixels_vs_time.png", "Does any single pixel's CAL misbehave along the run?", fig_cal_pixels_vs_time),
    ("toa_tot_drift.png", "Do TOT and the inter-board TOA offsets drift along the run?", fig_toa_tot_drift),
    ("twc_curves.png", "What does the time-walk correction look like, track by track?", fig_twc_curves),
    ("twc_array_maps.png", "Is the time-walk correction uniform across the pixel array?", fig_twc_array_maps),
    ("twc_board_compare.png", "Are the boards' time-walk corrections compatible?", fig_twc_board_compare),
    ("twc_vs_time.png", "Is the time-walk correction stable along the run?", fig_twc_vs_time),
    ("resolution_proxy_vs_time.png", "Does the timing hold along the run (proxy)?", fig_resolution_proxy_vs_time),
]


def cmd_plot(a):
    df = pd.read_parquet(a.input)
    label = str(df["run"].iloc[0])
    outdir = os.path.join(a.outdir, label)
    os.makedirs(outdir, exist_ok=True)
    n_files = int(meta_val(df, "n_files", 0)) or int(df.loc[df.section == "file", "file"].max()) + 1
    tm = TimeMap(a.t0, a.duration_min, n_files, a.file_times)
    written = []
    for fname, question, fn in FIGS:
        try:
            out = fn(df, os.path.join(outdir, fname), tm)
        except Exception as e:  # keep going; report at the end
            print("  !! %s failed: %s" % (fname, e))
            out = None
        if out:
            written.append((fname, question))
            print("  wrote %s" % out)
    print("done: %d figures in %s" % (len(written), outdir))


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("summarize", help="compute per-track summaries -> tidy parquet")
    s.add_argument("--tracks-dir", required=True, dest="tracks_dir", help="step-9 <combo>/tracks directory")
    s.add_argument("--time-dir", required=True, dest="time_dir", help="step-10 <combo>/time directory")
    s.add_argument("--label", required=True, help="run/combo label used in the parquet and output names")
    s.add_argument("-o", "--outdir", required=True)
    s.add_argument("--files-per-bin", type=int, default=5, dest="files_per_bin",
                   help="raw files per bin for the TWC-vs-time and resolution-vs-time studies (default 5)")
    s.add_argument("--min-events-twc", type=int, default=3000, dest="min_events_twc",
                   help="only tracks with at least this many surviving events enter the per-bin TWC study (default 3000)")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--max-tracks", type=int, default=0, dest="max_tracks", help="debug: process only the first N tracks")
    s.set_defaults(fn=cmd_summarize)
    q = sp.add_parser("plot", help="figures from the summarize parquet")
    q.add_argument("-i", "--input", required=True)
    q.add_argument("-o", "--outdir", required=True)
    q.add_argument("--t0", default=None, help="run start (ISO), only used to label the time axis")
    q.add_argument("--duration-min", type=float, default=None, dest="duration_min",
                   help="run duration in minutes; with it a top axis in hours is added under a constant-rate assumption")
    q.add_argument("--file-times", default=None, dest="file_times",
                   help="CSV with columns file,time (raw-file timestamps): exact file->time mapping, overrides --duration-min")
    add_output_arguments(q)
    q.set_defaults(fn=cmd_plot)
    a = p.parse_args()
    if a.cmd == "plot":
        set_output_options(a.format, a.split)
    a.fn(a)


if __name__ == "__main__":
    main()
