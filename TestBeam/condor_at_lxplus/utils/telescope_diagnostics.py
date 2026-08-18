#!/usr/bin/env python3
"""Per-run telescope diagnostics for the ETROC test-beam analysis pipeline.

WHAT THIS IS
------------
A run-level health check of the telescope geometry and of the track candidates the
pipeline produced, meant to be run once per run AFTER step 7 (path_finder.py has
written ``tracks_<combo>.parquet`` and select_tracks_by_coverage.py the matching
``tracks_<combo>_reduced.parquet``).

It has two subcommands:

* ``summarize`` reads one run's step-6/step-7 outputs (plus, optionally, the raw
  decoded feather files, the CAL table and the alignment yaml) and condenses
  everything into ONE tidy long parquet, ``<label>_diagnostics.parquet``. That file
  is small (a few thousand rows) and is the only thing that needs to be kept or
  copied around; the multi-hundred-MB inputs are not needed again.
* ``plot`` renders that parquet. Given ONE file it writes one figure per plain
  question - ``hit_maps.png`` (where do particles hit each board), ``occupancy.png``
  (how busy is each board), ``cal_tot_halves.png`` (CAL, TOT code and TOT in
  ns per pixel, and the two chip halves), ``track_landing.png`` (where a track lands relative to the trigger
  board), ``beam_tilt.png`` (how tilted the tracks are), plus ``board_rotation.png``
  with ``--extended`` - each carrying its question as the title, a caption in
  ordinary words (the same explanations are in the README, section 7b). Given SEVERAL
  files it draws cross-run trends (e.g. an HV scan), so comparing twenty runs is just
  ``plot -i */*_diagnostics.parquet``.

TIDY SCHEMA (one row = one number)
----------------------------------
``run, section, combo, board, pair, half, row, col, key, value, text``

``value`` is the number, ``text`` carries string payloads (board name/role). Empty
string / NA marks a dimension that does not apply to that row. Everything a plot
needs is stored, including the binned profiles and the 2D offset histograms, so
``plot`` never has to touch the original track files.

CONVENTIONS (pixel orientation)
-------------------------------
On every ETROC board the pixel (row 0, col 0) sits at the BOTTOM-RIGHT of the chip
and (15, 15) at the TOP-LEFT. Increasing ``col`` therefore runs physically to the
LEFT and increasing ``row`` physically UP. Consequently every panel with a ``col``
(or ``dcol``) axis in this tool is drawn with that axis REVERSED, so the picture is
what an observer sees looking DOWN at the ETROC from above - the view in which the
chip's wire-bond pads run along the bottom edge and the bump-bond pads face you (an
LGAD bump-bonded on top would then cover the ETROC except its lower balcony with the
wire-bond pads): col 15 at the left edge, col 0 at
the right edge, row 0 at the bottom. A positive ``dcol`` means "physically
leftward". The two readout halves are ``col<8`` = the PHYSICAL RIGHT half and
``col>=8`` = the PHYSICAL LEFT half.

Nothing numeric changes because of this: all stored values (means, peaks, slopes,
ratios) remain in raw col/row units with their original signs. Likewise the
internal ``x = (col - 7.5) * 1.3`` used by core/path_finder.py is a MIRROR of the
physical x, but every cut in the pipeline is mirror-symmetric (radii, |dcol| windows),
so no pipeline result depends on the handedness - only the rendering and the wording
here do.

PHYSICS CONVENTIONS
-------------------
* One pixel = ``PIXEL_PITCH`` = 1.3 mm in the plane; pixel centres are
  ``(index - 7.5) * pitch`` (same model as core/path_finder.py).
* Board index is used as the z axis. The physical plane spacing along the beam is
  generally UNKNOWN, so slopes are quoted per PLANE GAP, never per mm or in mrad.
* Every candidate row carries a ``count`` (how many events shared that exact pixel
  pattern), so all means/sigmas/modes/fits below are COUNT-WEIGHTED, and their
  uncertainties use the Kish effective N - repeated identical patterns are not
  independent measurements.

Author's note on which estimator to read: for a skewed offset distribution the
count-weighted MEAN is pulled by the combinatorial tail, while the MODE (refined
by the centroid over mode +/- 1 px, stored as ``peak_*``) tracks the real particle
population. Both are stored; prefer ``peak_*`` for geometry.
"""

import argparse
import os
import re
import sys
import textwrap
from glob import glob

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyBboxPatch, Rectangle
from matplotlib.transforms import Affine2D
from natsort import natsorted
from tabulate import tabulate

# --- geometry constants, kept identical to core/path_finder.py -----------------
PIXEL_PITCH = 1.3    # mm per pixel
PIXEL_OFFSET = 7.5   # pixel index of the array centre
GRID = 16            # 16 x 16 pixels per board

# Output options shared by every diagnostics tool that imports this module:
# figure file format and, on request, one file per panel next to the compound figure.
OUTPUT = {"format": "png", "split": False}


def set_output_options(fmt="png", split=False):
    OUTPUT["format"] = fmt
    OUTPUT["split"] = bool(split)


def add_output_arguments(parser):
    """--format / --split for a plot-style subcommand."""
    parser.add_argument("--format", choices=["png", "pdf"], default="png",
                        help="file format of the figures (default png)")
    parser.add_argument("--split", action="store_true",
                        help="also write every panel of a compound figure as its own file, "
                             "<figure>__pNN.<format> (a colour bar stays with its panel)")


def save_figure(fig, out, dpi=140):
    """Save `fig` as <out stem>.<format>; with OUTPUT['split'] also each panel on its own.
    Returns the path of the compound figure."""
    stem = os.path.splitext(out)[0]
    fmt = OUTPUT["format"]
    path = "%s.%s" % (stem, fmt)
    fig.savefig(path, dpi=dpi)
    if OUTPUT["split"]:
        from matplotlib.transforms import Bbox
        renderer = fig.canvas.get_renderer()
        cbars = [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]
        panels = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
        for k, ax in enumerate(panels):
            bb = ax.get_tightbbox(renderer)
            if bb is None:
                continue
            for cax in cbars:
                parents = getattr(cax, "_colorbar_info", {}).get("parents", [])
                if ax in parents:
                    cb_bb = cax.get_tightbbox(renderer)
                    if cb_bb is not None:
                        bb = Bbox.union([bb, cb_bb])
            inch = bb.transformed(fig.dpi_scale_trans.inverted())
            inch = Bbox.from_extents(inch.x0 - 0.12, inch.y0 - 0.12, inch.x1 + 0.12, inch.y1 + 0.12)
            fig.savefig("%s__p%02d.%s" % (stem, k + 1, fmt), dpi=dpi, bbox_inches=inch)
    return path

HALF_LO, HALF_HI = "col<8", "col>=8"  # the two readout halves of an ETROC2 chip
# Display-only labels. The stored parquet values stay HALF_LO / HALF_HI.
# Pixel (0,0) is at the chip's BOTTOM-RIGHT, so col<8 is the physical RIGHT half.
HALF_LBL = {HALF_LO: "col<8 (phys. right half)", HALF_HI: "col>=8 (phys. left half)"}
PHYS_NOTE = "physical view: pixel (0,0) bottom-right"
DCOL_NOTE = "(+ = physically leftward)"
CAL_REF_NS = 3.125   # the fixed reference interval whose delay-cell gates CAL counts
# Honesty guards for a narrow, centred beam (the usual CERN/DESY testbeam case).
MIN_D1_SPREAD = 0.3  # px: below this spread of the first-gap offset the straightness
                     # (plane-spacing) fit has no lever arm and is reported as NaN
MIN_ROT_SPAN = 8     # anchor rows/cols that must be illuminated before a rotation fit
NANC = "#d6d6d6"     # colour for a pixel with NO measurement (NaN) in a pixel map

# --- plot styling, carried over from the reference viz scripts -----------------
CAT = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]  # Okabe-Ito, CVD-safe
INK, INK2, GRIDC = "#1b1b1b", "#555555", "#d9d9d9"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "figure.dpi": 130,
    "axes.edgecolor": GRIDC, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    # Nothing in a figure is allowed below 9 pt: these are read on a laptop, on
    # a projector and printed, by people who did not write the tool.
    "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
    "figure.titlesize": 13.5,
    "axes.grid": False, "grid.color": GRIDC, "grid.linewidth": 0.6, "grid.alpha": 0.7,
    "axes.axisbelow": True, "legend.frameon": False,
})

SCHEMA = ["run", "section", "combo", "board", "pair", "half", "row", "col",
          "key", "value", "text"]


# =============================================================================
# count-weighted statistics
# =============================================================================
def wmean(x, w):
    return float(np.sum(w * x) / np.sum(w))


def wstd(x, w):
    m = wmean(x, w)
    return float(np.sqrt(np.sum(w * (x - m) ** 2) / np.sum(w)))


def wskew(x, w):
    """Third standardized moment. Non-zero skew is why mean != mode for the
    pairwise offsets: the combinatorial background sits on one side only."""
    m, s = wmean(x, w), wstd(x, w)
    if s == 0:
        return 0.0
    return float(np.sum(w * (x - m) ** 3) / np.sum(w) / s ** 3)


def wmode(x, w):
    """Integer-valued mode: the value carrying the largest total weight."""
    vals, inv = np.unique(x, return_inverse=True)
    return float(vals[np.argmax(np.bincount(inv, weights=w))])


def wpeak(x, w):
    """Sub-pixel refinement of the mode: count-weighted centroid over mode +/- 1 px.
    (path_finder.py --find_alignment reports the peak bin of a 30-bin histogram of the
    same shifts in mm, a coarser estimator of the same peak.)"""
    m = wmode(x, w)
    k = np.abs(x - m) <= 1
    if not k.any() or np.sum(w[k]) <= 0:
        return m
    return float(np.average(x[k], weights=w[k]))


def neff(w):
    """Kish effective sample size: identical repeated patterns are correlated."""
    return float(np.sum(w) ** 2 / np.sum(w ** 2))


def wlsq(x, y, w):
    """Weighted OLS y = a x + b, with the slope error rescaled to the Kish N_eff."""
    W = np.sum(w)
    mx, my = np.sum(w * x) / W, np.sum(w * y) / W
    sxx = np.sum(w * (x - mx) ** 2)
    if sxx <= 0:
        return np.nan, np.nan, np.nan, neff(w)
    a = np.sum(w * (x - mx) * (y - my)) / sxx
    b = my - a * mx
    s2 = np.sum(w * (y - (a * x + b)) ** 2) / W
    ne = neff(w)
    sa = float(np.sqrt(s2 / max(ne * sxx / W, 1e-12)))
    return float(a), float(b), sa, ne


def weighted_profile(xv, yv, w, nbins=GRID):
    """<y> vs integer x bin, error = weighted sigma / sqrt(N_eff)."""
    xs, ys, es = [], [], []
    for i in range(nbins):
        k = xv == i
        if not k.any() or np.sum(w[k]) <= 0:
            continue
        ne = neff(w[k])
        if ne < 2:
            continue
        xs.append(float(i))
        ys.append(wmean(yv[k], w[k]))
        es.append(wstd(yv[k], w[k]) / np.sqrt(ne))
    return np.array(xs), np.array(ys), np.array(es)


def fit_profile(x, y, e):
    """Straight-line fit through a profile, weighted by 1/err^2."""
    if len(x) < 2:
        return np.nan, np.nan
    ww = 1.0 / np.clip(e, 1e-9, None) ** 2
    p = np.polyfit(x, y, 1, w=np.sqrt(ww))
    return float(p[0]), float(p[1])


# =============================================================================
# small helpers
# =============================================================================
def board_ids_from_columns(df):
    return sorted(int(c.split("_")[1]) for c in df.columns if c.startswith("row_"))


def pair_name(a, b):
    return "%dv%d" % (a, b)


def percentiles(v, qs):
    """Percentiles of the per-candidate 'count' column itself (NOT count-weighted:
    the question here is 'how big are the candidate counts', one entry per pattern)."""
    if len(v) == 0:
        return dict((q, np.nan) for q in qs)
    return dict((q, float(np.percentile(v, q))) for q in qs)


def tot_code_to_ns(tot, cal):
    """TOT code -> nanoseconds, per pixel.

    CAL is the number of delay-cell gates that flip in the fixed CAL_REF_NS
    reference interval, so the TDC bin of that pixel is CAL_REF_NS / CAL and every
    TOA or TOT code is a COUNT OF THOSE BINS. Where CAL is higher the bin is
    smaller, so the same physical time reads a HIGHER code - which is why a TOT
    code has to be multiplied by the pixel's own bin before two pixels (or two chip
    halves) can be compared. (2 * tot - tot // 32) is the standard ETROC2
    linearisation of the TOT code before that multiplication.
    """
    tot = np.asarray(tot, dtype=float)
    cal = np.asarray(cal, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (2.0 * tot - np.floor(tot / 32.0)) * (CAL_REF_NS / cal)


def die(msg):
    sys.stderr.write("ERROR: %s\n" % msg)
    sys.exit(1)


class Rows(object):
    """Accumulator for the tidy long table."""

    def __init__(self, run):
        self.run = run
        self._rows = []

    def add(self, section, key, value=np.nan, combo="", board=None, pair="",
            half="", row=None, col=None, text=""):
        self._rows.append({
            "run": self.run, "section": section, "combo": combo, "board": board,
            "pair": pair, "half": half, "row": row, "col": col, "key": key,
            "value": np.nan if value is None else float(value), "text": text,
        })

    def frame(self):
        df = pd.DataFrame(self._rows, columns=SCHEMA)
        for c in ("board", "row", "col"):
            df[c] = pd.array(df[c], dtype="Int64")
        df["value"] = df["value"].astype("float64")
        for c in ("run", "section", "combo", "pair", "half", "key", "text"):
            df[c] = df[c].fillna("").astype(str)
        return df


# =============================================================================
# summarize: section builders
# =============================================================================
def load_run_config(config_path, run_name):
    """Board metadata for one run out of the board_configs_yaml file."""
    from ruamel.yaml import YAML
    yaml = YAML(typ="safe")
    with open(config_path, "r") as fh:
        cfg = yaml.load(fh)
    if run_name not in cfg:
        die("run '%s' not found in %s" % (run_name, config_path))
    return dict((int(b), v) for b, v in cfg[run_name].items())


def section_meta(R, run_config, sampling_note):
    """Per-board bookkeeping (name/role/HV/offset/irradiation) so a diagnostics
    parquet is self-describing and cross-run plots can label points by HV."""
    for bid in sorted(run_config):
        info = run_config[bid] or {}
        for key in ("name", "short", "role", "irrad", "note"):
            if key in info and info[key] is not None:
                R.add("meta", key, board=bid, text=str(info[key]))
        for key in ("HV", "offset"):
            if key in info and info[key] is not None:
                try:
                    R.add("meta", key, value=float(info[key]), board=bid)
                except (TypeError, ValueError):
                    R.add("meta", key, board=bid, text=str(info[key]))
    R.add("meta", "pixel_pitch_mm", value=PIXEL_PITCH)
    R.add("meta", "n_boards", value=len(run_config))
    R.add("meta", "sampling_note", text=sampling_note)


def pair_offsets(df, ba, bb):
    """dcol/drow of board ba relative to board bb, plus the candidate weights."""
    dc = (df["col_%d" % ba].astype(int) - df["col_%d" % bb].astype(int)).to_numpy(float)
    dr = (df["row_%d" % ba].astype(int) - df["row_%d" % bb].astype(int)).to_numpy(float)
    return dc, dr, df["count"].to_numpy(float)


def all_pairs_within(df, board_ids, max_diff_pixel):
    """Per-candidate mask: EVERY board pair sits within max_diff_pixel of that
    pair's modal offset. This is the diagnostic twin of path_finder's radius cut:
    it says what fraction of the surviving candidates are geometrically consistent
    with a single particle rather than a random coincidence."""
    w = df["count"].to_numpy(float)
    ok = np.ones(len(df), dtype=bool)
    per_pair = {}
    for i, ba in enumerate(board_ids):
        for bb in board_ids[i + 1:]:
            dc, dr, _ = pair_offsets(df, ba, bb)
            mc, mr = wmode(dc, w), wmode(dr, w)
            good = np.hypot(dc - mc, dr - mr) <= max_diff_pixel
            per_pair[pair_name(ba, bb)] = dict(
                mode_dcol=mc, mode_drow=mr,
                frac_within=float(np.sum(w[good]) / np.sum(w)))
            ok &= good
    return ok, per_pair, w


def section_candidates(R, tracks, max_diff_pixel):
    """All step-6 candidates: how many, how peaked their counts are, and how many
    are geometrically self-consistent."""
    for combo, df in sorted(tracks.items()):
        bids = board_ids_from_columns(df)
        cnt = df["count"].to_numpy(float)
        R.add("candidates", "n_candidates", value=len(df), combo=combo)
        R.add("candidates", "sum_count", value=float(cnt.sum()), combo=combo)
        pct = percentiles(cnt, [10, 50, 90])
        for q in (10, 50, 90):
            R.add("candidates", "count_p%d" % q, value=pct[q], combo=combo)
        R.add("candidates", "count_max", value=float(cnt.max()) if len(cnt) else np.nan,
              combo=combo)
        ok, per_pair, w = all_pairs_within(df, bids, max_diff_pixel)
        R.add("candidates", "frac_all_pairs_within_maxdiff",
              value=float(np.sum(w[ok]) / np.sum(w)), combo=combo)
        for pn, st in sorted(per_pair.items()):
            for k in ("mode_dcol", "mode_drow", "frac_within"):
                R.add("candidates", k, value=st[k], combo=combo, pair=pn)


def section_selected(R, reduced, max_diff_pixel):
    """Step-7 selection: size, count distribution, and per-board pixel coverage.
    coverage_frac is the fraction of the 256 pixels backed by >=1 selected
    candidate - the guarantee select_tracks_by_coverage.py is built around."""
    for combo, df in sorted(reduced.items()):
        bids = board_ids_from_columns(df)
        cnt = df["count"].to_numpy(float)
        R.add("selected", "n_selected", value=len(df), combo=combo)
        pct = percentiles(cnt, [10, 50, 90])
        for q in (10, 50, 90):
            R.add("selected", "count_p%d" % q, value=pct[q], combo=combo)
        R.add("selected", "count_min", value=float(cnt.min()) if len(cnt) else np.nan,
              combo=combo)
        R.add("selected", "count_max", value=float(cnt.max()) if len(cnt) else np.nan,
              combo=combo)
        # Low-count candidates are the ones step 8 will struggle to time-align.
        R.add("selected", "n_count_lt100", value=float((cnt < 100).sum()), combo=combo)
        ok, _, w = all_pairs_within(df, bids, max_diff_pixel)
        R.add("selected", "frac_all_pairs_within_maxdiff",
              value=float(np.sum(w[ok]) / np.sum(w)), combo=combo)
        for b in bids:
            px = set(zip(df["row_%d" % b].astype(int), df["col_%d" % b].astype(int)))
            R.add("selected", "coverage_frac", value=len(px) / float(GRID * GRID),
                  combo=combo, board=b)
            hot = df[df["count"] >= 100]
            pxh = set(zip(hot["row_%d" % b].astype(int), hot["col_%d" % b].astype(int)))
            R.add("selected", "coverage_frac_count_ge100",
                  value=len(pxh) / float(GRID * GRID), combo=combo, board=b)


def pick_hitmap_source(tracks, board, trig_id):
    """Which combo's candidates to use for a board's beam profile.

    Prefer a combo that contains the trigger board (its candidates are the ones the
    radius cut is anchored on, so the profile is least contaminated by
    combinatorics); among those take the one with the most total counts, i.e. the
    best statistics. Fall back to any combo containing the board.
    """
    have = [c for c, d in tracks.items() if "row_%d" % board in d.columns]
    if not have:
        return None
    withtrig = [c for c in have
                if trig_id is not None and "row_%d" % trig_id in tracks[c].columns]
    pool = withtrig if withtrig else have
    return sorted(pool, key=lambda c: (-float(tracks[c]["count"].sum()), c))[0]


def section_hitmap(R, tracks, board_ids, trig_id):
    """Count-weighted beam profile per board.

    Under the flat halo illumination of an off-axis telescope the centroid is not
    'where the beam is' in a Gaussian sense - it measures the FLUX GRADIENT across
    the 20.8 mm array. Its drift from plane to plane is the mean projected track
    direction (see the 'angles' section for the cleaner handle)."""
    for b in board_ids:
        combo = pick_hitmap_source(tracks, b, trig_id)
        if combo is None:
            continue
        df = tracks[combo]
        w = df["count"].to_numpy(float)
        rr = df["row_%d" % b].to_numpy(int)
        cc = df["col_%d" % b].to_numpy(int)
        m = np.zeros((GRID, GRID))
        np.add.at(m, (rr, cc), w)
        for r in range(GRID):
            for c in range(GRID):
                if m[r, c] != 0:
                    R.add("hitmap", "hitmap", value=m[r, c], combo=combo,
                          board=b, row=r, col=c)
        col_c, row_c = wmean(cc, w), wmean(rr, w)
        R.add("hitmap", "centroid_col", value=col_c, combo=combo, board=b)
        R.add("hitmap", "centroid_row", value=row_c, combo=combo, board=b)
        R.add("hitmap", "centroid_x_mm", value=(col_c - PIXEL_OFFSET) * PIXEL_PITCH,
              combo=combo, board=b)
        R.add("hitmap", "centroid_y_mm", value=(row_c - PIXEL_OFFSET) * PIXEL_PITCH,
              combo=combo, board=b)
        R.add("hitmap", "sigma_col", value=wstd(cc, w), combo=combo, board=b)
        R.add("hitmap", "sigma_row", value=wstd(rr, w), combo=combo, board=b)
        lo, hi = m[:, :8].sum(), m[:, 8:].sum()
        R.add("hitmap", "sum_count", value=float(lo), combo=combo, board=b, half=HALF_LO)
        R.add("hitmap", "sum_count", value=float(hi), combo=combo, board=b, half=HALF_HI)
        # An imbalance across the col-8 boundary can be a flux gradient across the
        # telescope and/or a chip-half effect (the two halves' TDCs sit on
        # different supplies: digital for col<8, analog for col>=8). Stored as the
        # col<8 (physical RIGHT half) over
        # col>=8 (physical LEFT half) hit ratio.
        R.add("hitmap", "ratio_collt8_over_colge8", value=float(lo / hi) if hi else np.nan,
              combo=combo, board=b)


def section_cal(R, cal_path):
    """Per-pixel CAL code: the number of delay-cell gates that flip in the fixed
    3.125 ns reference interval, i.e. the pixel's TDC bin is 3.125 ns / CAL. Its
    spread within a board is delay-line process variation; a step across the col-8
    half boundary is expected, because the TDCs of the two halves are powered from
    different supplies (ETROC2 manual rev 0.6, sec 3.7.1 Table 20: col<8, the
    physical RIGHT half, off the digital supply; col>=8, the physical LEFT half,
    off the analog/discriminator supply). Because every TOA/TOT code is counted in
    units of that bin, a CAL step makes the codes step too - a unit change, not a
    threshold or gain difference."""
    cal = pd.read_csv(cal_path)
    for b, g in cal.groupby("board"):
        b = int(b)
        vals = g["cal_mode"].to_numpy(float)
        for _, rec in g.iterrows():
            if np.isfinite(rec["cal_mode"]):
                R.add("cal", "cal_mode", value=float(rec["cal_mode"]), board=b,
                      row=int(rec["row"]), col=int(rec["col"]))
        good = vals[np.isfinite(vals)]
        R.add("cal", "n_nan", value=float(np.sum(~np.isfinite(vals))), board=b)
        if good.size:
            R.add("cal", "cal_median", value=float(np.median(good)), board=b)
            R.add("cal", "cal_min", value=float(good.min()), board=b)
            R.add("cal", "cal_max", value=float(good.max()), board=b)
            R.add("cal", "cal_std", value=float(good.std()), board=b)
        for hname, sub in ((HALF_LO, g[g["col"] < 8]), (HALF_HI, g[g["col"] >= 8])):
            v = sub["cal_mode"].to_numpy(float)
            vg = v[np.isfinite(v)]
            R.add("cal", "n_nan", value=float(np.sum(~np.isfinite(v))), board=b, half=hname)
            if vg.size:
                R.add("cal", "cal_median", value=float(np.median(vg)), board=b, half=hname)
                R.add("cal", "cal_min", value=float(vg.min()), board=b, half=hname)
                R.add("cal", "cal_max", value=float(vg.max()), board=b, half=hname)
                R.add("cal", "cal_std", value=float(vg.std()), board=b, half=hname)


def section_raw(R, raw_dir, n_files, cal_path=None):
    """Unselected decoded hits: the only place where per-board efficiency/noise is
    visible before any coincidence requirement. f0/f1/f2plus are the fractions of
    events with 0 / exactly 1 / >=2 hits on that board - f2plus is the pile-up and
    noise budget that the single-hit requirement of step 6 throws away.

    With cal_path given, every hit is also converted to physical time: the CAL
    table's per-pixel mode gives that pixel's TDC bin (3.125 ns / CAL) and
    tot_ns_median is the median over the hits of that pixel of the linearised TOT
    code times its own bin. The raw TOT code is a bin COUNT, so the code alone
    cannot be compared between pixels or chip halves whose CAL differs; the ns
    value can."""
    files = natsorted(glob(os.path.join(raw_dir, "*.feather")))
    if not files:
        die("no *.feather files in --raw-dir %s" % raw_dir)
    files = files[:n_files]
    parts = []
    for k, f in enumerate(files):
        d = pd.read_feather(f, columns=["evt", "board", "row", "col", "tot", "cal"])
        # evt numbering restarts in every loop file: offset so events stay unique.
        d["evt"] = d["evt"].astype(np.int64) + k * 10 ** 9
        parts.append(d)
    raw = pd.concat(parts, ignore_index=True)
    del parts
    if cal_path and os.path.isfile(cal_path):
        cal = pd.read_csv(cal_path)[["board", "row", "col", "cal_mode"]].copy()
        for c in ("board", "row", "col"):
            cal[c] = cal[c].astype(np.int64)
            raw[c] = raw[c].astype(np.int64)
        raw = raw.merge(cal, on=["board", "row", "col"], how="left")
        # NaN wherever that pixel has no CAL entry, which propagates to tot_ns.
        raw["tot_ns"] = tot_code_to_ns(raw["tot"].to_numpy(),
                                       raw["cal_mode"].to_numpy())
    n_ev = int(raw["evt"].nunique())
    R.add("raw", "n_events_read", value=float(n_ev))
    R.add("raw", "n_files_read", value=float(len(files)))
    for b, sb in raw.groupby("board"):
        b = int(b)
        per = sb.groupby("evt").size()
        R.add("raw", "hits", value=float(len(sb)), board=b)
        R.add("raw", "events", value=float(len(per)), board=b)
        R.add("raw", "hits_per_event", value=float(len(sb)) / n_ev, board=b)
        R.add("raw", "f0", value=float(n_ev - len(per)) / n_ev, board=b)
        R.add("raw", "f1", value=float((per == 1).sum()) / n_ev, board=b)
        R.add("raw", "f2plus", value=float((per >= 2).sum()) / n_ev, board=b)
        lo = int((sb["col"] < 8).sum())
        hi = int((sb["col"] >= 8).sum())
        # col<8 (physical RIGHT half) over col>=8 (physical LEFT half) hit ratio.
        R.add("raw", "ratio_collt8_over_colge8", value=float(lo) / hi if hi else np.nan,
              board=b)
        m = np.zeros((GRID, GRID))
        np.add.at(m, (sb["row"].to_numpy(int), sb["col"].to_numpy(int)), 1.0)
        for r in range(GRID):
            for c in range(GRID):
                if m[r, c] != 0:
                    R.add("raw", "rawmap", value=m[r, c], board=b, row=r, col=c)
        # Per-pixel median raw ToT: the charge proxy, one number per pixel, so
        # the CAL/ToT figure can show both quantities on the same 16x16 grid.
        for (r, c), v in sb.groupby(["row", "col"])["tot"].median().items():
            if np.isfinite(v):
                R.add("raw", "tot_median", value=float(v), board=b,
                      row=int(r), col=int(c))
        # Per-pixel median TOT in NANOSECONDS: the same hits, each code multiplied
        # by its own pixel's TDC bin (3.125 ns / CAL). This is the quantity that is
        # comparable across the chip halves; the code alone is not.
        if "tot_ns" in sb.columns:
            for (r, c), v in sb.groupby(["row", "col"])["tot_ns"].median().items():
                if np.isfinite(v):
                    R.add("raw", "tot_ns_median", value=float(v), board=b,
                          row=int(r), col=int(c))
        for hname, sub in ((HALF_LO, sb[sb["col"] < 8]), (HALF_HI, sb[sb["col"] >= 8])):
            R.add("raw", "hits", value=float(len(sub)), board=b, half=hname)
            if len(sub):
                R.add("raw", "tot_mean", value=float(sub["tot"].mean()), board=b, half=hname)
                R.add("raw", "tot_median", value=float(sub["tot"].median()), board=b, half=hname)
                R.add("raw", "cal_std", value=float(sub["cal"].std()), board=b, half=hname)
                if "tot_ns" in sub.columns:
                    v = float(sub["tot_ns"].median())
                    if np.isfinite(v):
                        R.add("raw", "tot_ns_median", value=v, board=b, half=hname)
        # Per-column slices expose the col-8 step directly (occupancy and ToT).
        for c, sc in sb.groupby("col"):
            R.add("raw", "hits_per_col_mean", value=float(len(sc)) / GRID,
                  board=b, col=int(c))
            R.add("raw", "tot_mean_col", value=float(sc["tot"].mean()),
                  board=b, col=int(c))


def section_pairwise(R, tracks, max_diff_pixel):
    """Pixel offset between every pair of boards in a combo.

    mean vs peak: the offset distribution is skewed (combinatorial coincidences
    populate one side), so the mean is biased away from the particle population.
    peak_* = centroid over mode +/- 1 px is the estimator to compare with the yaml
    alignment translations."""
    edges = np.arange(-6.5, 7.5, 1.0)
    for combo, df in sorted(tracks.items()):
        bids = board_ids_from_columns(df)
        for i, ba in enumerate(bids):
            for bb in bids[i + 1:]:
                dc, dr, w = pair_offsets(df, ba, bb)
                pn = pair_name(ba, bb)
                stats = dict(
                    mode_dcol=wmode(dc, w), mode_drow=wmode(dr, w),
                    peak_dcol=wpeak(dc, w), peak_drow=wpeak(dr, w),
                    mean_dcol=wmean(dc, w), mean_drow=wmean(dr, w),
                    sigma_dcol=wstd(dc, w), sigma_drow=wstd(dr, w),
                    skew_dcol=wskew(dc, w), skew_drow=wskew(dr, w),
                )
                good = np.hypot(dc - stats["mode_dcol"], dr - stats["mode_drow"]) <= max_diff_pixel
                stats["frac_within_maxdiff"] = float(np.sum(w[good]) / np.sum(w))
                for k in sorted(stats):
                    R.add("pairwise", k, value=stats[k], combo=combo, pair=pn)
                keep = (np.abs(dc) <= 6) & (np.abs(dr) <= 6)
                H, _, _ = np.histogram2d(dc[keep], dr[keep], bins=[edges, edges],
                                         weights=w[keep])
                for ic in range(H.shape[0]):
                    for ir in range(H.shape[1]):
                        if H[ic, ir] != 0:
                            R.add("pairwise", "hist2d", value=float(H[ic, ir]),
                                  combo=combo, pair=pn, row=ir - 6, col=ic - 6)


def section_angles(R, tracks, reduced):
    """Projected track slope, in pixels per PLANE GAP.

    For a combo the outer-board offset (last minus first) divided by the lever arm
    (number of plane gaps between them) is the mean projected direction. With the
    z spacing unknown this is NOT convertible to mrad; multiply by 1.3 for mm per
    plane gap. Both the step-7 selected tracks ('_sel', cleaner) and all step-6
    candidates ('_all', includes combinatorics) are stored."""
    for combo in sorted(tracks):
        bids = board_ids_from_columns(tracks[combo])
        first, last = bids[0], bids[-1]
        lever = float(last - first)  # board index difference = number of plane gaps
        if lever <= 0:
            continue
        R.add("angles", "lever_planes", value=lever, combo=combo)
        for suffix, source in (("_sel", reduced.get(combo)), ("_all", tracks.get(combo))):
            if source is None or len(source) == 0:
                continue
            dc, dr, w = pair_offsets(source, last, first)
            R.add("angles", "mean_dcol_per_plane" + suffix, value=wmean(dc / lever, w), combo=combo)
            R.add("angles", "sigma_dcol_per_plane" + suffix, value=wstd(dc / lever, w), combo=combo)
            R.add("angles", "skew_dcol" + suffix, value=wskew(dc, w), combo=combo)
            R.add("angles", "mean_drow_per_plane" + suffix, value=wmean(dr / lever, w), combo=combo)
            R.add("angles", "sigma_drow_per_plane" + suffix, value=wstd(dr / lever, w), combo=combo)
            R.add("angles", "skew_drow" + suffix, value=wskew(dr, w), combo=combo)
            # Discrete outer-offset distribution (fractions), enough to redraw both
            # the 2D pattern and the 1D per-plane histograms without the tracks.
            keep = (np.abs(dc) <= 6) & (np.abs(dr) <= 6)
            tot = np.sum(w)
            key = "outer_offset_frac" + suffix
            code = (dc[keep].astype(int) + 6) * 13 + (dr[keep].astype(int) + 6)
            codes, inv = np.unique(code, return_inverse=True)
            sums = np.bincount(inv, weights=w[keep], minlength=len(codes))
            for cd, s in zip(codes, sums):
                R.add("angles", key, value=float(s / tot), combo=combo,
                      row=int(cd % 13) - 6, col=int(cd // 13) - 6)


def anchor_board(bids, roles_by_id):
    """The trigger board when present (path_finder anchors its spatial cut there
    too); without one, the first of ref/dut/extra, else the lowest board id
    (path_finder uses the combo's median board id instead)."""
    by_role = dict((roles_by_id.get(b, ""), b) for b in bids)
    for role in ("trig", "ref", "dut", "extra"):
        if role in by_role:
            return by_role[role]
    return bids[0]


def section_rotation(R, tracks, roles_by_id):
    """Rotation / projective probe.

    If two planes are rotated about the beam axis relative to each other, the
    offset in one coordinate varies linearly with the position in the OTHER
    coordinate: d(row) grows with column, d(col) grows with row. The fitted slope
    in px/px is tan(relative rotation); the profile points themselves are stored so
    the plot subcommand can redraw the fit without the track files.

    A narrow, centred beam is the honest limit of this probe: the fit needs the beam
    to illuminate a range of anchor rows/columns. profile_span_* records how many of
    the 16 are lit, and below MIN_ROT_SPAN of them the slope/angle are stored as NaN
    with a 'status' row saying so, instead of a fit through two points."""
    for combo, df in sorted(tracks.items()):
        bids = board_ids_from_columns(df)
        anc = anchor_board(bids, roles_by_id)
        ar = df["row_%d" % anc].to_numpy(int)
        ac = df["col_%d" % anc].to_numpy(int)
        for b in bids:
            if b == anc:
                continue
            dc, dr, w = pair_offsets(df, b, anc)
            # pair label is always ordered (a<b) but the offsets are always board - anchor
            pn = pair_name(min(b, anc), max(b, anc))
            for key, xv, yv in (("drow_vs_col", ac, dr), ("dcol_vs_row", ar, dc)):
                X, Y, E = weighted_profile(xv, yv, w)
                # How much of the array the beam lights up: the number of anchor
                # rows/cols holding at least a couple of independent tracks.
                span = int(len(X))
                what = "columns" if key.endswith("_col") else "rows"
                R.add("rotation", "profile_span_" + key, value=float(span),
                      combo=combo, pair=pn)
                if span < MIN_ROT_SPAN:
                    slope = np.nan
                    R.add("rotation", "status", combo=combo, pair=pn,
                          text="beam too narrow for a rotation estimate (%s): only "
                               "%d of %d anchor %s illuminated"
                               % (key, span, GRID, what))
                else:
                    slope, _ = fit_profile(X, Y, E)
                R.add("rotation", "slope_" + key, value=slope, combo=combo, pair=pn)
                R.add("rotation", "angle_deg_from_" + ("drow" if key.startswith("drow") else "dcol"),
                      value=float(np.degrees(np.arctan(slope))) if np.isfinite(slope) else np.nan,
                      combo=combo, pair=pn)
                for xb, yb, eb in zip(X, Y, E):
                    R.add("rotation", "profile_" + key, value=float(yb),
                          combo=combo, pair=pn, col=int(xb))
                    R.add("rotation", "profile_err_" + key, value=float(eb),
                          combo=combo, pair=pn, col=int(xb))
        R.add("rotation", "anchor_board", value=float(anc), combo=combo)


def section_spacing(R, tracks, board_ids):
    """Relative plane spacing from track straightness.

    For a straight track through three planes, d2 = (last - mid) offset and
    d1 = (mid - first) offset satisfy d2 = a * d1 + b with a = z-gap ratio. Two
    caveats, both real here:
      * errors-in-variables: pixelisation noise on d1 ATTENUATES the plain OLS
        slope, so the OLS number is an artifact and is stored only as such. The
        forward fit and the inverse fit bracket the truth; the geometric mean of
        the two (GM / reduced-major-axis) is the estimator used for the solve.
      * the finite 16x16 aperture gives the combinatorial background a rhombic
        support, which imposes a spurious anti-correlation.
      * the fit needs a lever arm: if the tracks have almost no angular spread the
        first-gap offset d1 is always the same number and there is nothing to
        regress against. The count-weighted spread of d1 is stored as
        spread_d1_<axis>, and below MIN_D1_SPREAD pixels the slopes are stored as
        NaN with a 'status' row reading "not determinable: tracks have too little
        angular spread" - a narrow, centred beam is exactly that case.
    The resulting spacing ratios are good to roughly a factor two - treat them as
    a sanity check on the mechanical survey, not a measurement.
    """
    idx = dict((b, i) for i, b in enumerate(sorted(board_ids)))
    slopes = {}
    for combo, df in sorted(tracks.items()):
        bids = board_ids_from_columns(df)
        if len(bids) != 3:
            continue
        f, m, l = bids[0], bids[1], bids[2]
        w = df["count"].to_numpy(float)
        per_axis = {}
        for nm in ("col", "row"):
            v1 = (df["%s_%d" % (nm, m)].astype(int) - df["%s_%d" % (nm, f)].astype(int)).to_numpy(float)
            v2 = (df["%s_%d" % (nm, l)].astype(int) - df["%s_%d" % (nm, m)].astype(int)).to_numpy(float)
            s1 = wstd(v1, w)
            R.add("spacing", "spread_d1_%s" % nm, value=s1, combo=combo)
            if not np.isfinite(s1) or s1 < MIN_D1_SPREAD:
                # No angular spread, no lever arm: refuse to quote a spacing.
                per_axis[nm] = dict(a=np.nan, sa=np.nan, a_gm=np.nan, s_gm=np.nan,
                                    r=np.nan)
                for k in ("ols_slope_%s", "gm_slope_%s", "r_%s"):
                    R.add("spacing", k % nm, value=np.nan, combo=combo)
                R.add("spacing", "status", combo=combo,
                      text="not determinable: tracks have too little angular spread "
                           "(%s spread of the first-gap offset is %.2f px, below "
                           "%.1f px)" % (nm, s1, MIN_D1_SPREAD))
                continue
            a, _, sa, _ = wlsq(v1, v2, w)
            ar, _, _, _ = wlsq(v2, v1, w)
            # inverse fit: 1/a_rev is the un-attenuated end of the EIV bracket
            a_hi = 1.0 / ar if np.isfinite(ar) and ar != 0 else np.nan
            ok = np.isfinite(a) and np.isfinite(a_hi) and a * a_hi > 0
            a_gm = float(np.sqrt(a * a_hi)) if ok else np.nan
            r = float(np.sign(a) * np.sqrt(a / a_hi)) if ok else np.nan
            s_gm = 0.5 * abs(a_hi - a) if ok else np.nan  # bracket half-width
            per_axis[nm] = dict(a=a, sa=sa, a_gm=a_gm, s_gm=s_gm, r=r)
            R.add("spacing", "ols_slope_%s" % nm, value=a, combo=combo)
            R.add("spacing", "gm_slope_%s" % nm, value=a_gm, combo=combo)
            R.add("spacing", "r_%s" % nm, value=r, combo=combo)
        slopes[combo] = dict(boards=(idx[f], idx[m], idx[l]), axes=per_axis)

    # Solve for the two unknown gap ratios (d01 fixed to 1). Only defined for a
    # 4-plane telescope; with more planes there are more unknowns than this
    # two-parameter grid scan handles, so the per-combo slopes above are all we store.
    if len(board_ids) != 4 or not slopes:
        return
    obs = []
    for combo, s in slopes.items():
        for nm in ("col", "row"):
            a = s["axes"][nm]["a_gm"]
            e = float(np.hypot(s["axes"][nm]["s_gm"], s["axes"][nm]["sa"]))
            if np.isfinite(a) and np.isfinite(e) and e > 0:
                obs.append((s["boards"], a, e))
    if len(obs) < 3:
        return
    ug, vg = np.meshgrid(np.linspace(0.05, 6.0, 800), np.linspace(0.05, 6.0, 800))

    def predicted(bidx, u, v):
        z = [0.0 * u, 1.0 + 0.0 * u, 1.0 + u, 1.0 + u + v]
        f, m, l = bidx
        return (z[l] - z[m]) / (z[m] - z[f])

    chi2 = np.zeros_like(ug)
    for bidx, a, e in obs:
        chi2 += ((predicted(bidx, ug, vg) - a) / e) ** 2
    i = np.unravel_index(np.argmin(chi2), chi2.shape)
    u0, v0, c0 = float(ug[i]), float(vg[i]), float(chi2[i])
    dof = max(len(obs) - 2, 1)
    # Rescale the 1-sigma contour by chi2/dof: the model does not fit perfectly and
    # pretending otherwise would quote an absurdly tight bracket.
    inside = (chi2 - c0) <= max(c0 / dof, 1.0)
    R.add("spacing", "d12_over_d01", value=u0)
    R.add("spacing", "d12_over_d01_lo", value=float(ug[inside].min()))
    R.add("spacing", "d12_over_d01_hi", value=float(ug[inside].max()))
    R.add("spacing", "d23_over_d01", value=v0)
    R.add("spacing", "d23_over_d01_lo", value=float(vg[inside].min()))
    R.add("spacing", "d23_over_d01_hi", value=float(vg[inside].max()))
    R.add("spacing", "chi2", value=c0)
    R.add("spacing", "dof", value=float(dof))


def section_alignment(R, align_path, run_name):
    """Translations written by path_finder.py --find_alignment, stored per combo/board.

    Reads the layout path_finder writes ({run: {legacy_per_combo: {combo: {board:
    {transformation: {translation}}}}, global_relative: {pinned_board, boards: {...}}}});
    the global_relative block is stored under combo 'global_relative'. A flat
    {run: {combo: {board: ...}}} layout is accepted too."""
    from ruamel.yaml import YAML
    yaml = YAML(typ="safe")
    with open(align_path, "r") as fh:
        doc = yaml.load(fh)
    block = doc.get(run_name, doc) or {}

    def store(combo, boards):
        for bid, payload in (boards or {}).items():
            tr = ((payload or {}).get("transformation", {}) or {}).get("translation", {}) or {}
            for axis in ("x", "y", "z"):
                if axis in tr and tr[axis] is not None:
                    R.add("alignment", axis, value=float(tr[axis]),
                          combo=str(combo), board=int(bid))

    if "legacy_per_combo" in block or "global_relative" in block:
        for combo, boards in (block.get("legacy_per_combo") or {}).items():
            store(combo, boards)
        glob_block = block.get("global_relative") or {}
        store("global_relative", glob_block.get("boards"))
        if glob_block.get("pinned_board") is not None:
            R.add("alignment", "pinned_board", value=float(glob_block["pinned_board"]),
                  combo="global_relative")
    else:
        for combo, boards in block.items():
            store(combo, boards)


# =============================================================================
# summarize driver
# =============================================================================
def load_tracks(tracks_dir):
    """Returns ({combo: all-candidates df}, {combo: selected df})."""
    tracks, reduced = {}, {}
    for path in sorted(glob(os.path.join(tracks_dir, "tracks_*.parquet"))):
        stem = os.path.basename(path)[len("tracks_"):-len(".parquet")]
        if stem.endswith("_reduced"):
            reduced[stem[:-len("_reduced")]] = pd.read_parquet(path)
        else:
            tracks[stem] = pd.read_parquet(path)
    return tracks, reduced


def cmd_summarize(args):
    if not os.path.isdir(args.tracks_dir):
        die("--tracks-dir not found: %s" % args.tracks_dir)
    for path, flag in ((args.cal_table, "--cal-table"), (args.config, "-c")):
        if not os.path.isfile(path):
            die("%s not found: %s" % (flag, path))

    tracks, reduced = load_tracks(args.tracks_dir)
    if not tracks:
        die("no tracks_<combo>.parquet files in %s" % args.tracks_dir)
    missing = sorted(set(tracks) - set(reduced))
    if missing:
        sys.stderr.write("WARNING: no *_reduced.parquet for combo(s): %s "
                         "(run select_tracks_by_coverage.py first)\n" % ", ".join(missing))

    run_config = load_run_config(args.config, args.runName)
    roles_by_id = dict((b, (info or {}).get("role", "")) for b, info in run_config.items())
    trig_id = next((b for b, r in roles_by_id.items() if r == "trig"), None)
    board_ids = sorted(run_config)

    R = Rows(args.label)
    sampling_note = ("candidates from %s; %d combo(s); max_diff_pixel=%g px used for the "
                     "consistency fractions" % (os.path.basename(args.tracks_dir.rstrip("/")),
                                                len(tracks), args.max_diff_pixel))
    section_meta(R, run_config, sampling_note)
    section_candidates(R, tracks, args.max_diff_pixel)
    section_selected(R, reduced, args.max_diff_pixel)
    section_hitmap(R, tracks, board_ids, trig_id)
    section_cal(R, args.cal_table)
    if args.raw_dir:
        if not os.path.isdir(args.raw_dir):
            die("--raw-dir not found: %s" % args.raw_dir)
        section_raw(R, args.raw_dir, args.n_raw_files, args.cal_table)
    section_pairwise(R, tracks, args.max_diff_pixel)
    section_angles(R, tracks, reduced)
    section_rotation(R, tracks, roles_by_id)
    section_spacing(R, tracks, board_ids)
    if args.alignment:
        if not os.path.isfile(args.alignment):
            die("--alignment not found: %s" % args.alignment)
        section_alignment(R, args.alignment, args.runName)

    df = R.frame()
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "%s_diagnostics.parquet" % args.label)
    df.to_parquet(out, index=False)
    print_summary(df, args.label, out)
    return 0


def print_summary(df, label, out):
    """Compact human-readable digest of the parquet just written."""
    def scal(section, key, **kw):
        q = df[(df["section"] == section) & (df["key"] == key)]
        for k, v in kw.items():
            q = q[q[k] == v]
        return float(q["value"].iloc[0]) if len(q) else float("nan")

    boards = sorted(set(df.loc[df["section"] == "meta", "board"].dropna().astype(int)))
    rows = []
    for b in boards:
        meta = df[(df["section"] == "meta") & (df["board"] == b)]
        role = meta.loc[meta["key"] == "role", "text"]
        short = meta.loc[meta["key"] == "short", "text"]
        rows.append([b, role.iloc[0] if len(role) else "", short.iloc[0] if len(short) else "",
                     "%.0f" % scal("meta", "HV", board=b),
                     "%.2f" % scal("hitmap", "centroid_col", board=b),
                     "%.2f" % scal("hitmap", "centroid_row", board=b),
                     "%.0f" % scal("cal", "cal_median", board=b),
                     "%.3f" % scal("raw", "hits_per_event", board=b),
                     "%.2f" % scal("hitmap", "ratio_collt8_over_colge8", board=b)])
    print("\n=== telescope diagnostics: %s ===" % label)
    print(tabulate(rows, headers=["board", "role", "chip", "HV", "cen.col", "cen.row",
                                  "CAL med", "raw h/ev", "col<8/col>=8"],
                   tablefmt="simple"))
    print("  (%s / %s; hit ratio col<8 over col>=8)" % (HALF_LBL[HALF_LO], HALF_LBL[HALF_HI]))

    combos = natsorted(set(df.loc[df["section"] == "angles", "combo"]))
    rows = []
    for c in combos:
        rows.append([c, "%.0f" % scal("angles", "lever_planes", combo=c),
                     "%.0f" % scal("candidates", "n_candidates", combo=c),
                     "%.0f" % scal("selected", "n_selected", combo=c),
                     "%.3f" % scal("candidates", "frac_all_pairs_within_maxdiff", combo=c),
                     "%+.3f" % scal("angles", "mean_dcol_per_plane_sel", combo=c),
                     "%+.3f" % scal("angles", "mean_drow_per_plane_sel", combo=c),
                     "%.2f" % scal("spacing", "gm_slope_col", combo=c),
                     "%.2f" % scal("spacing", "gm_slope_row", combo=c)])
    print(tabulate(rows, headers=["combo", "lever", "cand", "sel", "consistent",
                                  "<dcol>/plane", "<drow>/plane", "GM col", "GM row"],
                   tablefmt="simple"))
    u, v = scal("spacing", "d12_over_d01"), scal("spacing", "d23_over_d01")
    if np.isfinite(u):
        print("spacing ratio d01:d12:d23 = 1 : %.2f : %.2f  (GM regression, factor-2 accurate)"
              % (u, v))
    for _, rr in df[(df["section"] == "spacing") & (df["key"] == "status")].iterrows():
        print("plane spacing [%s]: %s" % (rr["combo"], rr["text"]))
    for _, rr in df[(df["section"] == "rotation") & (df["key"] == "status")].iterrows():
        print("rotation [%s %s]: %s" % (rr["combo"], rr["pair"], rr["text"]))
    print("rows written: %d -> %s" % (len(df), out))


# =============================================================================
# plot helpers
# =============================================================================
def q(df, section=None, key=None, combo=None, board=None, pair=None, half=None,
      run=None, pixels=None):
    """Filter the tidy table. pixels=True keeps only per-pixel rows, False only scalars."""
    m = pd.Series(True, index=df.index)
    for col, val in (("section", section), ("key", key), ("combo", combo),
                     ("pair", pair), ("half", half), ("run", run)):
        if val is not None:
            m &= df[col] == val
    if board is not None:
        m &= df["board"] == board
    if pixels is True:
        m &= df["row"].notna() & df["col"].notna()
    elif pixels is False:
        m &= df["row"].isna() & df["col"].isna()
    return df[m]


def one(df, section, key, **kw):
    r = q(df, section=section, key=key, **kw)
    return float(r["value"].iloc[0]) if len(r) else float("nan")


def text_of(df, section, key, **kw):
    r = q(df, section=section, key=key, **kw)
    return str(r["text"].iloc[0]) if len(r) else ""


def grid_of(df, section, key, board=None, combo=None, fill=0.0):
    """Rebuild a 16x16 (row, col) image from per-pixel rows; None if absent.

    fill is what an unlisted pixel means: 0 for counts (nothing was recorded
    there), NaN for per-pixel measurements like CAL or TOT (not measured)."""
    r = q(df, section=section, key=key, board=board, combo=combo, pixels=True)
    if len(r) == 0:
        return None
    rr = r["row"].astype(int).to_numpy()
    cc = r["col"].astype(int).to_numpy()
    img = np.full((GRID, GRID), float(fill))
    img[rr, cc] = r["value"].to_numpy(float)
    return img


def board_label(df, b):
    role = text_of(df, "meta", "role", board=b)
    short = text_of(df, "meta", "short", board=b)
    return "board %d - %s%s" % (b, role or "?", " (%s)" % short if short else "")


def boards_of(df):
    return sorted(set(df.loc[df["section"] == "meta", "board"].dropna().astype(int)))


def board_by_role(df, role):
    for b in boards_of(df):
        if text_of(df, "meta", "role", board=b) == role:
            return b
    return None


def phys_col_axis(ax, annotate=False):
    """Render a col / dcol x-axis in PHYSICAL orientation.

    Pixel (0,0) is at the bottom-right of the chip, so increasing col runs to the
    LEFT. Reversing the x-axis puts col 15 (or +dcol) on the left of the panel and
    col 0 (or -dcol) on the right; tick labels stay plain col numbers and no stored
    number is touched. Call this AFTER the artists and any set_xlim()."""
    if ax.get_xlim()[0] < ax.get_xlim()[1]:
        ax.invert_xaxis()
    if annotate:
        ax.text(0.99, 0.01, PHYS_NOTE, transform=ax.transAxes, ha="right",
                va="bottom", fontsize=9, color=INK2)
    return ax


# =============================================================================
# plot: figure furniture (captions, schematics, dual px/mm labelling)
# =============================================================================
# Every single-run figure is built the same way: a two-line suptitle whose first
# line is the QUESTION the figure answers in plain words, the panels, and a
# caption box at the bottom that says in ordinary sentences what the reader is
# looking at. finish() reserves the space for both and writes the file.
CAP_BOX = dict(fc="#f5f5f5", ec=GRIDC, boxstyle="round,pad=0.5")
ANN_BOX = dict(fc="white", ec=GRIDC, alpha=0.94, boxstyle="round,pad=0.4")
CAP_FS = 10.8   # caption font size; the band is drawn edge to edge to afford it
PX_MM = "1 pixel = %.1f mm" % PIXEL_PITCH
WIREBOND = "wire-bond pads (bottom edge)"


def finish(fig, out, question, sub, caption, wrap=None, dpi=140, w_pad=1.6,
           h_pad=1.8, cap_fs=CAP_FS):
    """Suptitle = question + run label, caption band at the bottom, then save.

    The caption band is a rectangle spanning the figure edge to edge (2% margin
    each side) with the text drawn on top of it, rather than a box that hugs the
    text: that way the reading guide can be set at cap_fs pt (>= 10.5) and still
    fit. The wrap width is therefore DERIVED from the figure width and the font
    size instead of being hard-coded per figure, so the lines really do reach the
    edges of the band.

    NOTE: tight_layout silently gives up ("Axes not compatible") if any gridspec
    in the figure carries its own hspace/wspace, and the caption then lands on
    top of the panels - so ask for outer spacing through h_pad / w_pad here and
    leave fig.add_gridspec() free of hspace/wspace."""
    w, h = [float(v) for v in fig.get_size_inches()]
    if wrap is None:
        # 0.503 em is the mean glyph advance of DejaVu Sans over English prose;
        # aim the longest line at ~94% of the figure width.
        wrap = max(60, int(0.94 * w * 72.0 / (0.503 * cap_fs)))
    body = textwrap.fill(" ".join(caption.split()), wrap)
    nlines = body.count("\n") + 1
    band = 0.36 + nlines * 1.55 * cap_fs / 72.0   # inches of caption band
    y0 = 0.06 / h                                 # band bottom, figure fraction
    top = 1.0 - 0.82 / h                          # inches reserved for the suptitle
    fig.suptitle("%s\n%s" % (question, sub), fontsize=13.5, y=1.0 - 0.16 / h,
                 va="top")
    # Panels are kept clear of the band by 0.16 in, so nothing can overlap it.
    fig.tight_layout(rect=[0.008, (0.06 + band + 0.16) / h, 0.992, top],
                     w_pad=w_pad, h_pad=h_pad)
    fig.add_artist(FancyBboxPatch((0.02, y0), 0.96, band / h,
                                  boxstyle="round,pad=0,rounding_size=0.006",
                                  transform=fig.transFigure, clip_on=False,
                                  fc="#f5f5f5", ec=GRIDC, lw=0.8, zorder=0))
    fig.text(0.5, y0 + 0.5 * band / h, body, ha="center", va="center",
             fontsize=cap_fs, color=INK, linespacing=1.55, zorder=1)
    out = save_figure(fig, out, dpi=dpi)
    plt.close(fig)
    return out


def pixel_map(ax, img, cmap="viridis", vmin=None, vmax=None, nan_blank=False):
    """16x16 (row, col) image in the physical orientation, col axis reversed.

    With nan_blank, a pixel that carries NO measurement is drawn in flat grey
    (NANC) rather than in a colour from the scale - on a TOT map those are the
    pixels no hit landed on, which is most of the array when the beam is a narrow
    spot."""
    data = np.where(np.isfinite(img), img, np.nan) if nan_blank else img
    if nan_blank:
        cmap = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap.copy()
        cmap.set_bad(NANC)
    im = ax.imshow(data, origin="lower", cmap=cmap, aspect="equal",
                   extent=[-0.5, 15.5, -0.5, 15.5], vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_xlabel("col [pixel index]")
    ax.set_ylabel("row [pixel index]")
    ax.set_xticks([0, 4, 8, 12, 15])
    ax.set_yticks([0, 4, 8, 12, 15])
    phys_col_axis(ax)
    return im


def wirebond_strip(ax, label=WIREBOND):
    """Grey bar along the bottom edge of a pixel map: the chip's wire-bond side.
    It is the one landmark that fixes the orientation for a reader holding the
    board - pads at the bottom, pixel (0,0) at the bottom-RIGHT."""
    ax.set_ylim(-2.3, 15.5)
    ax.add_patch(Rectangle((-0.5, -1.85), 16.0, 0.85, fc="#9e9e9e", ec="#6f6f6f",
                           lw=0.7, clip_on=False, zorder=6))
    ax.text(7.5, -1.42, label, ha="center", va="center", fontsize=9,
            color="white", zorder=7, clip_on=False)


def scale_bar(ax, px=5, y=1.35, x0=1.0):
    """A 5-pixel rule labelled in mm, so a pixel figure is also readable in mm.
    Drawn at low col and low row: the visually BOTTOM-RIGHT corner of the map."""
    ax.plot([x0, x0 + px], [y, y], "-", color="white", lw=3.0, solid_capstyle="butt",
            zorder=6)
    ax.text(x0 + 0.5 * px, y + 0.45, "%d px = %.1f mm" % (px, px * PIXEL_PITCH),
            ha="center", va="bottom", fontsize=9, color="white", zorder=6,
            bbox=dict(fc="black", alpha=0.5, ec="none", pad=1.5))


def half_boundary(ax, lw=1.7, alpha=1.0):
    """The col 7|8 line: the border between the chip's two readout halves."""
    ax.axvline(7.5, color="crimson", lw=lw, alpha=alpha, zorder=5)
    return ax


def board_title(df, b):
    """'board 3 - trigger - ETROC HPK-13 (PT_IH13)' - role spelled out."""
    role = text_of(df, "meta", "role", board=b) or "?"
    long_role = {"trig": "trigger", "ref": "reference", "dut": "device under test",
                 "extra": "extra plane"}.get(role, role)
    name = text_of(df, "meta", "name", board=b)
    short = text_of(df, "meta", "short", board=b)
    chip = name or short
    if name and short:
        chip = "%s (%s)" % (name, short)
    return "board %d - %s%s" % (b, long_role, "\n%s" % chip if chip else "")


def combo_boards(combo):
    """Board indices out of a combo label such as 'extra0-ref1-trig3'."""
    ids = []
    for tok in str(combo).split("-"):
        digits = re.findall(r"\d+", tok)
        if digits:
            ids.append(int(digits[-1]))
    return ids


def combo_plain(df, combo):
    """'boards 0-1-3 (extra, ref, trig)' - no pipeline jargon."""
    ids = combo_boards(combo)
    if not ids:
        return str(combo)
    roles = [text_of(df, "meta", "role", board=b) or "?" for b in ids]
    return "boards %s (%s)" % ("-".join(str(i) for i in ids), ", ".join(roles))


def combo_counts(df, combo):
    v = one(df, "candidates", "sum_count", combo=combo)
    return 0.0 if not np.isfinite(v) else v

def pairs_vs_anchor(df, section="pairwise"):
    """(combo, pair) entries, trigger-containing pairs first (the alignment-relevant
    ones), then the rest - deterministic order for the panel grids."""
    trig = board_by_role(df, "trig")
    seen = q(df, section=section)[["combo", "pair"]].drop_duplicates()
    items = [(r["combo"], r["pair"]) for _, r in seen.iterrows() if r["pair"]]
    def rank(it):
        a, b = [int(x) for x in it[1].split("v")]
        return (0 if trig in (a, b) else 1, it[0], it[1])
    return sorted(items, key=rank)


def outer_offset_arrays(df, combo, suffix):
    """Stored (dcol, drow, fraction) of the outermost-board offset per track."""
    r = q(df, section="angles", key="outer_offset_frac" + suffix, combo=combo, pixels=True)
    if len(r) == 0:
        return None
    return (r["col"].astype(int).to_numpy(float), r["row"].astype(int).to_numpy(float),
            r["value"].to_numpy(float))


# =============================================================================
# 1. hit_maps.png - where do particles hit each board?
# =============================================================================
def plot_hit_maps(df, out, sub):
    bs = boards_of(df)
    bs = [b for b in bs if grid_of(df, "hitmap", "hitmap", board=b) is not None]
    if not bs:
        return None
    ncol = 2 if len(bs) > 1 else 1
    nrow = int(np.ceil(len(bs) / float(ncol)))
    fig, axs = plt.subplots(nrow, ncol, figsize=(6.1 * ncol, 5.9 * nrow + 1.1),
                            squeeze=False)
    axs = axs.ravel()
    brighter = []
    for ax, b in zip(axs, bs):
        img = grid_of(df, "hitmap", "hitmap", board=b)
        im = pixel_map(ax, img)
        cc = one(df, "hitmap", "centroid_col", board=b)
        rc = one(df, "hitmap", "centroid_row", board=b)
        ax.plot(cc, rc, "x", color="crimson", ms=13, mew=2.6, zorder=6)
        ax.annotate("red x = centre of gravity:\n"
                    "mean col and mean row of this\n"
                    "map, weighted by the counts\n"
                    "col,row = %.1f, %.1f px\nx,y = %+.1f, %+.1f mm"
                    % (cc, rc, one(df, "hitmap", "centroid_x_mm", board=b),
                       one(df, "hitmap", "centroid_y_mm", board=b)),
                    xy=(cc, rc), xytext=(0.035, 0.975),
                    textcoords="axes fraction", fontsize=9, color=INK,
                    va="top", ha="left", bbox=ANN_BOX, zorder=7,
                    arrowprops=dict(arrowstyle="-", color="crimson", lw=1.2,
                                    shrinkB=9))
        scale_bar(ax)
        wirebond_strip(ax)
        ax.set_title(board_title(df, b), fontsize=10.5)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("sampled events hitting this pixel", fontsize=9.5)
        cb.ax.tick_params(labelsize=9)
        ratio = one(df, "hitmap", "ratio_collt8_over_colge8", board=b)
        if np.isfinite(ratio):
            brighter.append(ratio)
    for ax in axs[len(bs):]:
        ax.axis("off")

    if brighter:
        r = float(np.mean(brighter))
        side, sidecol = ("right", "col<8") if r > 1 else ("left", "col>=8")
        halftxt = ("the physically %s half (%s) carries %.2f times the counts of "
                   "the other half"
                   % (side, sidecol, max(r, 1.0 / r) if r else 1.0))
    else:
        halftxt = ""
    sc = [one(df, "hitmap", "sigma_col", board=b) for b in bs]
    sr = [one(df, "hitmap", "sigma_row", board=b) for b in bs]
    sc = [v for v in sc if np.isfinite(v)]
    sr = [v for v in sr if np.isfinite(v)]
    if sc and sr:
        spottxt = ("the counts sit within %.1f px in col and %.1f px in row of the "
                   "centre of gravity (count-weighted standard deviation, so %.0f%% "
                   "of the 16-pixel width in col)"
                   % (np.mean(sc), np.mean(sr), 100.0 * np.mean(sc) / GRID))
    else:
        spottxt = ""
    meas = "; ".join([t for t in (spottxt, halftxt) if t])
    meas = ("Measured here: %s, averaged over the boards shown." % meas) if meas else ""
    caption = (
        "Each cell adds up the sampled events - the events step 6 read - that "
        "gave a clean three-board coincidence with the hit on that pixel of this "
        "board, so one track candidate contributes as many counts as the events "
        "that share its pixel pattern. The red cross is the centre of gravity of "
        "the counts, computed as the mean col and the mean row of this map weighted "
        "by the counts (in mm: measured from the array centre, pixel 7.5). One "
        "pixel is %.1f mm, so the full array is %.1f x %.1f mm. "
        "How to read it: a flat map means this plane sits in a diffuse halo rather "
        "than in a focused beam, which is what a telescope placed off the beam axis "
        "sees; a bright compact spot means a focused beam crosses here. A gradient "
        "across the map, or one chip half brighter than the other, means a flux "
        "gradient across the telescope, and/or the difference between the two chip "
        "halves shown in the CAL/TOT figure. Dark rows or columns along an edge are "
        "the geometric acceptance of the coincidence with the neighbouring boards, "
        "not dead pixels. What the cross means depends on the illumination: when the "
        "counts form a compact spot, as they do for a narrow beam centred on the "
        "array, the centre of gravity IS the beam centre; under flat halo "
        "illumination it is not a beam position at all and mostly reflects whatever "
        "flux gradient is present. The spread measured below tells the two apart - a "
        "few pixels means a spot, half the array means flat illumination. %s"
        % (PIXEL_PITCH, GRID * PIXEL_PITCH, GRID * PIXEL_PITCH, meas))
    return finish(fig, out, "Where do particles hit each board?", sub, caption)


# =============================================================================
# 2. occupancy.png - how busy is each board?
# =============================================================================
def plot_occupancy(df, out, sub):
    bs = [b for b in boards_of(df) if grid_of(df, "raw", "rawmap", board=b) is not None]
    if not bs:
        return None
    n = len(bs)
    fig = plt.figure(figsize=(4.55 * n, 11.6))
    gs = fig.add_gridspec(2, 2 * n, height_ratios=[1.15, 0.85])
    for j, b in enumerate(bs):
        ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
        img = grid_of(df, "raw", "rawmap", board=b)
        im = pixel_map(ax, img)
        wirebond_strip(ax)
        ax.set_title("%s\n%d hits recorded" % (board_title(df, b),
                                               int(one(df, "raw", "hits", board=b))),
                     fontsize=10)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("hits on this pixel", fontsize=9.5)
        cb.ax.tick_params(labelsize=9)

    half = max(n, 1)
    ax = fig.add_subplot(gs[1, 0:half])
    y = [one(df, "raw", "hits_per_event", board=b) for b in bs]
    ax.bar(np.arange(n), y, width=0.6,
           color=[CAT[i % len(CAT)] for i in range(n)], edgecolor="white")
    for i, v in enumerate(y):
        ax.text(i, v, " %.2f" % v, ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(["board %d\n%s" % (b, text_of(df, "meta", "role", board=b))
                        for b in bs], fontsize=9.5)
    ax.set_ylabel("hits per event")
    ax.set_ylim(0, max(y) * 1.25 if max(y) > 0 else 1)
    ax.set_title("How many hits does a board record per event?\n"
                 "(all hits that board recorded / number of events read)",
                 fontsize=10.5)
    ax.grid(axis="y", alpha=0.3)

    ax = fig.add_subplot(gs[1, half:2 * n])
    keys = [("f0", "no hit", "#b0b0b0"), ("f1", "exactly 1 hit", CAT[2]),
            ("f2plus", "2 or more hits", CAT[1])]
    base = np.zeros(n)
    for key, lab, colr in keys:
        v = np.array([one(df, "raw", key, board=b) for b in bs], dtype=float)
        v = np.where(np.isfinite(v), v, 0.0)
        ax.bar(np.arange(n), v, bottom=base, width=0.6, color=colr, label=lab,
               edgecolor="white")
        for i in range(n):
            if v[i] > 0.06:
                ax.text(i, base[i] + 0.5 * v[i], "%.0f%%" % (100 * v[i]),
                        ha="center", va="center", fontsize=9.5,
                        color="white" if colr != "#b0b0b0" else INK)
        base = base + v
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(["board %d" % b for b in bs], fontsize=9.5)
    ax.set_ylabel("fraction of events")
    ax.set_ylim(0, 1.02)
    ax.set_title("What fraction of events have 0 / 1 / 2+ hits?\n"
                 "(each bar is a fraction of all events read, so they add to 1)",
                 fontsize=10.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9.5,
              frameon=False)
    ax.grid(axis="y", alpha=0.3)

    nfil = int(one(df, "raw", "n_files_read"))
    nev = int(one(df, "raw", "n_events_read"))
    caption = (
        "Top row: every hit the decoder wrote out for the first %d file(s) of the "
        "run, with no track requirement at all. Those %d file(s) hold %d events, "
        "and every number below is divided by that same event count. Bottom left: "
        "hits per event = all hits that board recorded in those file(s), divided by "
        "the %d events read. Bottom right: the fraction of those same %d events in "
        "which the board gave no hit, exactly one hit, or two or more, so the three "
        "pieces of a bar add up to 1 by construction. "
        "How to read it: hits per event around 1, with some 10-20%% of the events "
        "showing 2 or more hits, is a busy halo - one particle plus the occasional "
        "extra hit. Hits per event well above 1, or single pixels far brighter than "
        "their neighbours in the top row, points instead at noise or a badly "
        "behaved channel, worth checking against the pixel masks. A large no-hit "
        "fraction is expected rather than alarming: at IRRAD the trigger is the AND "
        "of two boards (which two varies from run to run) and it is ONE whole-chip "
        "trigger bit, not a per-pixel one, so a board can be triggered on and still "
        "record nothing in an event; on top of that the DAQ drops or truncates "
        "events, and the chip's own fast-readout mode omits the data words of an "
        "event once the L1 buffer is 96 or more deep. Those three together account "
        "for the zero-hit fractions, and since the analysis requires a hit on the "
        "pixels of interest, incomplete events fall out downstream anyway. A step in "
        "hit rate at the col 7 | 8 line is a chip-half effect - the two halves' "
        "TDCs sit on different supplies and their discriminators can differ "
        "slightly; note that a step in the TOT CODE at that line is only the change "
        "of TDC bin size and means nothing about threshold (see the CAL/TOT "
        "figure). The 2-or-more slice is the pile-up and noise that the single-hit "
        "requirement of the track search throws away."
        % (nfil, nfil, nev, nev, nev))
    return finish(fig, out, "How busy is each board?", sub, caption)


# =============================================================================
# 3. cal_tot_halves.png - CAL and TOT per pixel, and the two chip halves
# =============================================================================
def half_medians(img):
    """(median over the physical RIGHT half col<8, median over the LEFT half
    col>=8) of a 16x16 map, ignoring pixels with no measurement."""
    if img is None:
        return np.nan, np.nan
    outs = []
    for part in (img[:, :8], img[:, 8:]):
        good = np.isfinite(part)
        outs.append(float(np.median(part[good])) if good.any() else np.nan)
    return outs[0], outs[1]


def ratio(a, b):
    """a / b, NaN-safe - used for the right-half over left-half numbers."""
    if np.isfinite(a) and np.isfinite(b) and b != 0:
        return float(a) / float(b)
    return np.nan


def col_average(img):
    """Mean over rows of a 16x16 map, ignoring empty pixels: value vs col."""
    with np.errstate(invalid="ignore"):
        good = np.isfinite(img) & (img != 0)
        n = good.sum(axis=0)
        s = np.where(good, img, 0.0).sum(axis=0)
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def plot_cal_tot_halves(df, out, sub):
    bs = [b for b in boards_of(df)
          if grid_of(df, "cal", "cal_mode", board=b, fill=np.nan) is not None]
    if not bs:
        return None
    nrow = len(bs)
    fig = plt.figure(figsize=(20.4, 4.55 * nrow + 1.6))
    gs = fig.add_gridspec(nrow, 4, width_ratios=[1.0, 1.0, 1.0, 1.45])

    def mapcol(slot, img, title, cblabel):
        """One 16x16 panel, half boundary drawn, unmeasured pixels left grey."""
        ax = fig.add_subplot(slot)
        data = np.where(np.isfinite(img) & (img > 0), img, np.nan)
        pos = data[np.isfinite(data)]
        im = pixel_map(ax, data, nan_blank=True,
                       vmin=np.percentile(pos, 2) if pos.size else None,
                       vmax=np.percentile(pos, 98) if pos.size else None)
        half_boundary(ax)
        ax.set_title(title, fontsize=10)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(cblabel, fontsize=9.5)
        cb.ax.tick_params(labelsize=9)
        return ax

    def missing(slot, what):
        ax = fig.add_subplot(slot)
        ax.text(0.5, 0.5, "no %s in this parquet\n(re-run summarize with --raw-dir)"
                % what, ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color=INK2)
        ax.axis("off")

    rcal, rtot, rns = [], [], []
    for i, b in enumerate(bs):
        cal = grid_of(df, "cal", "cal_mode", board=b, fill=np.nan)
        tot = grid_of(df, "raw", "tot_median", board=b, fill=np.nan)
        tns = grid_of(df, "raw", "tot_ns_median", board=b, fill=np.nan)
        clo = one(df, "cal", "cal_median", board=b, half=HALF_LO)
        chi = one(df, "cal", "cal_median", board=b, half=HALF_HI)

        mapcol(gs[i, 0], cal,
               "%s\nCAL code per pixel   |   median %.0f (right half) vs %.0f "
               "(left half)" % (board_title(df, b).replace("\n", " - "), clo, chi),
               "CAL code (TDC bin = %.3f ns / CAL)" % CAL_REF_NS)

        if tot is None:
            missing(gs[i, 1], "raw TOT")
            tlo = thi = np.nan
        else:
            tlo, thi = half_medians(tot)
            mapcol(gs[i, 1], tot,
                   "TOT code per pixel (median of the raw hits)\nmedian %.0f (right "
                   "half) vs %.0f (left half)" % (tlo, thi),
                   "TOT code (counts in TDC bins)")
        if tns is None:
            missing(gs[i, 2], "TOT in ns (needs the CAL table at summarize time)")
            nlo = nhi = np.nan
        else:
            nlo, nhi = half_medians(tns)
            mapcol(gs[i, 2], tns,
                   "TOT in ns per pixel (code x that pixel's bin)\nmedian %.2f "
                   "(right half) vs %.2f ns (left half)" % (nlo, nhi),
                   "TOT in ns (code x per-pixel bin)")
        rcal.append(ratio(clo, chi))
        rtot.append(ratio(tlo, thi))
        rns.append(ratio(nlo, nhi))

        # Column averages: CAL on top, and below it the TOT code together with the
        # SAME TOT in ns on its own axis - the code trace steps with CAL, the ns
        # trace does not, which is the whole point of the figure.
        inner = gs[i, 3].subgridspec(2, 1, hspace=0.5)
        axc = fig.add_subplot(inner[0])
        axt = fig.add_subplot(inner[1], sharex=axc)
        cols = np.arange(GRID, dtype=float)
        axc.plot(cols, col_average(cal), "o-", ms=5, color=CAT[0], lw=1.6)
        axc.set_ylabel("CAL code\n(column average)", fontsize=9.5, color=CAT[0])
        axc.tick_params(labelbottom=False)
        axc.set_title("Column average across the half boundary\nCAL median %.0f "
                      "right vs %.0f left (left minus right = %+.0f, right/left "
                      "= %.3f)" % (clo, chi, chi - clo, ratio(clo, chi)), fontsize=10)
        axn = axt.twinx()
        hs, ls = [], []
        if tot is not None:
            h, = axt.plot(cols, col_average(tot), "s-", ms=5, color=CAT[1], lw=1.6)
            hs.append(h)
            ls.append("TOT code (counts of TDC bins, left axis)")
        if tns is not None:
            h, = axn.plot(cols, col_average(tns), "^--", ms=5, color=CAT[2], lw=1.6)
            hs.append(h)
            ls.append("TOT in ns (code x per-pixel bin, right axis)")
        axt.set_ylabel("TOT code\n(column average)", fontsize=9.5, color=CAT[1])
        axn.set_ylabel("TOT [ns]\n(column average)", fontsize=9.5, color=CAT[2])
        axn.tick_params(axis="y", labelsize=9, colors=CAT[2])
        axt.set_xlabel("col [pixel index]   (col 15 is physically left)")
        if np.isfinite(ratio(tlo, thi)) or np.isfinite(ratio(nlo, nhi)):
            axt.set_title("TOT code right/left = %.3f (it follows CAL);   TOT in ns "
                          "right/left = %.3f (unit removed)"
                          % (ratio(tlo, thi), ratio(nlo, nhi)), fontsize=10)
        if hs:
            # Headroom first, so the legend sits over empty space instead of over
            # the traces or the col 7 | 8 label at the bottom of the panel.
            for a in (axt, axn):
                y0, y1 = a.get_ylim()
                a.set_ylim(y0, y1 + 0.95 * (y1 - y0))
            axt.legend(hs, ls, loc="upper left", bbox_to_anchor=(0.0, 0.99),
                       fontsize=9, frameon=True, framealpha=0.92)
        for a in (axc, axt):
            half_boundary(a, lw=1.4, alpha=0.85)
            a.set_xlim(-0.6, 15.6)
            a.grid(alpha=0.3)
            a.text(7.5, 0.03, "col 7 | 8", transform=a.get_xaxis_transform(),
                   ha="center", va="bottom", fontsize=9, color="crimson",
                   bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0), zorder=6)
            phys_col_axis(a)

    def avg(v):
        v = [x for x in v if np.isfinite(x)]
        return float(np.mean(v)) if v else np.nan
    rc, rt, rn = avg(rcal), avg(rtot), avg(rns)
    meas = ""
    if np.isfinite(rc) and np.isfinite(rt):
        meas = ("Measured here, averaged over the boards shown: CAL right/left = "
                "%.3f and TOT code right/left = %.3f, the two agreeing to %.1f%% - "
                "the code step IS the CAL step." % (rc, rt, 100.0 * abs(rt / rc - 1.0)))
        if np.isfinite(rn):
            meas += (" Converted to nanoseconds the same TOT gives right/left = "
                     "%.3f, so the two halves agree in physical time to %.1f%% and "
                     "the step is gone." % (rn, 100.0 * abs(rn - 1.0)))
    caption = (
        "Left column: CAL, the per-pixel mode (the commonest code on that pixel) of "
        "the TDC calibration code the pipeline itself uses. Second column: the "
        "per-pixel median of the raw TOT code. Third column: that same TOT "
        "converted to nanoseconds pixel by pixel, each code multiplied by its own "
        "pixel's TDC bin. Right column: all three averaged down each column (mean "
        "over the rows of that column, unmeasured pixels ignored), so a difference "
        "between the chip's two readout halves shows up as a step at the red col "
        "7 | 8 line; col<8 is the physical right half of the chip and col>=8 the "
        "physical left half, and the quoted step is left minus right. Pixels with "
        "no measurement are left grey: on the TOT maps those are pixels that no hit "
        "landed on in the files read, so a narrow beam leaves most of the array grey. "
        "How to read it: CAL is the number of delay-cell gates that flip in a fixed "
        "%.3f ns reference interval, so that pixel's TDC bin is %.3f ns / CAL and "
        "every TOA and TOT code is a COUNT OF THOSE BINS - where CAL is higher the "
        "bin is smaller and the SAME physical time reads a HIGHER code. A CAL step "
        "exactly at the col 7 | 8 line is therefore an expected chip-half feature, "
        "not a fault: per the ETROC2 Reference Manual (rev 0.6, sec 3.7.1, Table 20) "
        "the TDCs of the right half (col<8) are powered from the DIGITAL supply and "
        "those of the left half (col>=8) from the ANALOG (discriminator) supply. A "
        "smooth grade in CAL from column to column or row to row is fabrication "
        "process variation of the delay line - a slightly faster or slower corner of "
        "the array counts a slightly different number of gates. A TOT-code step that "
        "follows the CAL step is just the change of unit: convert with the per-pixel "
        "CAL and it disappears, which is exactly what the ns map and the flat ns "
        "trace show. A step that REMAINS in ns would be a real threshold or gain "
        "difference between the halves, and that is the only version of this "
        "observation worth chasing. Isolated pixels sitting far from their "
        "neighbours in any of the maps are pixels to watch, and candidates for "
        "masking. None of this biases the timing: the pipeline uses the per-pixel "
        "CAL, so every pixel gets its own bin width. %s"
        % (CAL_REF_NS, CAL_REF_NS, meas))
    return finish(fig, out, "CAL and TOT per pixel - the two chip halves", sub,
                  caption, w_pad=2.0, h_pad=2.2)

# =============================================================================
# 4. track_landing.png - where does a track land relative to the trigger board?
# =============================================================================
def landing_source(df, b, anchor):
    """The trigger-containing combo with the most counts that holds both boards,
    the same rule summarize uses to pick a board's hit-map source. Returns
    (combo, pair_name) or (None, None)."""
    pn = pair_name(min(b, anchor), max(b, anchor))
    pool = []
    for c in natsorted(set(q(df, section="pairwise")["combo"])):
        if not c:
            continue
        if len(q(df, section="pairwise", key="hist2d", combo=c, pair=pn)):
            pool.append(c)
    if not pool:
        return None, None
    return sorted(pool, key=lambda c: (-combo_counts(df, c), c))[0], pn


def landing_schematic(ax, b, anchor, anchor_role):
    """Two chip squares, the anchor behind and this board in front, with the
    arrow whose components ARE (dcol, drow).

    Layout rule: nothing is labelled where it is drawn. The two squares carry one
    corner label each, the two pixels are identified by a legend block at the
    bottom, and the arrow label sits in the empty wedge between the squares - so
    no two pieces of text can land on top of each other whatever the board numbers
    are. The axes box is only slightly taller than it is wide, which keeps the
    column narrow and leaves no dead space against the map next to it."""
    # The frame is 10 x 13 units: with an equal aspect the sketch is width-limited,
    # so those proportions match the tall narrow column it sits in and leave no dead
    # space beside the map.
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 13.0)
    ax.set_aspect("equal")
    ax.axis("off")
    # anchor plane, drawn behind and up-right
    ax.add_patch(Rectangle((3.7, 5.9), 5.1, 5.1, fc="#eaeaea", ec="#9a9a9a",
                           lw=1.4, zorder=1))
    ax.text(8.8, 11.15, "board %d (%s)" % (anchor, anchor_role), ha="right",
            va="bottom", fontsize=9.5, color="#6f6f6f", zorder=2)
    # this board, in front and down-left
    ax.add_patch(Rectangle((1.0, 3.1), 5.1, 5.1, fc="white", ec=CAT[0], lw=2.0,
                           zorder=3))
    ax.text(1.15, 3.25, "board %d\n(this row)" % b, ha="left", va="bottom",
            fontsize=9.5, color=CAT[0], zorder=4)
    # The anchor pixel deliberately sits in the part of the back square the front
    # square does not cover, so it stays visible.
    ax.add_patch(Rectangle((7.2, 9.1), 0.8, 0.8, fc="#8a8a8a", ec="none", zorder=2))
    ax.add_patch(Rectangle((4.4, 6.3), 0.8, 0.8, fc=CAT[1], ec="none", zorder=4))
    ax.annotate("", xy=(4.9, 6.8), xytext=(7.5, 9.4),
                arrowprops=dict(arrowstyle="-|>", color=CAT[1], lw=2.2,
                                shrinkA=0, shrinkB=0), zorder=5)
    ax.text(6.55, 7.55, "(dcol, drow)", fontsize=10, color=CAT[1], ha="left",
            va="center", zorder=6)
    ax.text(5.0, 0.25,
            "grey square = pixel that fired on board %d\n"
            "orange square = pixel that fired on board %d\n"
            "(0, 0) = same pixel index on both boards" % (anchor, b),
            ha="center", va="bottom", fontsize=9, color=INK, bbox=ANN_BOX,
            zorder=7)


def plot_track_landing(df, out, sub):
    trig = board_by_role(df, "trig")
    ref = board_by_role(df, "ref")
    bs = [b for b in boards_of(df) if b != trig]
    rows = []
    for b in bs:
        anchor, role = trig, "trigger"
        combo, pn = (landing_source(df, b, anchor) if anchor is not None
                     else (None, None))
        if combo is None and ref is not None and ref != b:
            anchor, role = ref, "reference"
            combo, pn = landing_source(df, b, anchor)
        if combo is not None:
            rows.append((b, anchor, role, combo, pn))
    if not rows:
        return None

    n = len(rows)
    # Row height, not column width, is what sets the size of an equal-aspect map,
    # so the maps are enlarged by giving each row more height; the schematic column
    # is narrow and the widths are chosen so the map axes come out roughly square,
    # which is what removes the dead space that used to sit beside the sketch.
    fig = plt.figure(figsize=(16.6, 5.95 * n + 1.7))
    gs = fig.add_gridspec(n, 3, width_ratios=[0.85, 1.35, 1.55])
    fallback = False
    for i, (b, anchor, role, combo, pn) in enumerate(rows):
        if role != "trigger":
            fallback = True
        landing_schematic(fig.add_subplot(gs[i, 0]), b, anchor, role)

        # stored dcol is always col(lower board) - col(higher board); we want
        # this board minus the anchor board.
        sgn = 1.0 if b < anchor else -1.0
        r = q(df, section="pairwise", key="hist2d", combo=combo, pair=pn)
        img = np.zeros((13, 13))
        img[r["row"].astype(int) + 6, r["col"].astype(int) + 6] = \
            r["value"].to_numpy(float)
        if sgn < 0:
            img = img[::-1, ::-1]
        pc = sgn * one(df, "pairwise", "peak_dcol", combo=combo, pair=pn)
        pr = sgn * one(df, "pairwise", "peak_drow", combo=combo, pair=pn)
        sc = one(df, "pairwise", "sigma_dcol", combo=combo, pair=pn)
        sr = one(df, "pairwise", "sigma_drow", combo=combo, pair=pn)
        sk = sgn * one(df, "pairwise", "skew_dcol", combo=combo, pair=pn)

        ax = fig.add_subplot(gs[i, 1])
        im = ax.imshow(img, origin="lower", cmap="viridis", aspect="equal",
                       extent=[-6.5, 6.5, -6.5, 6.5], interpolation="nearest")
        ax.plot(pc, pr, "+", color="white", ms=17, mew=2.8, zorder=5)
        ax.axhline(0, color="white", lw=0.7, alpha=0.5)
        ax.axvline(0, color="white", lw=0.7, alpha=0.5)
        ax.set_xlabel("dcol [pixels]   (+ = physically left)")
        ax.set_ylabel("drow [pixels]")
        ax.set_title("%s\nboard set used: %s" % (board_title(df, b).replace("\n", " - "),
                                                 combo_plain(df, combo)), fontsize=10)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("sampled events", fontsize=9.5)
        cb.ax.tick_params(labelsize=9)
        phys_col_axis(ax)

        # The numbers live in their own panel above the projections: nothing is
        # written on top of the map, so no annotation can hide data.
        inner = gs[i, 2].subgridspec(3, 1, height_ratios=[0.42, 1.0, 1.0],
                                     hspace=0.62)
        lean = ("skewed toward +dcol (left)" if sk > 0.15 else
                "skewed toward -dcol (right)" if sk < -0.15 else
                "symmetric in dcol")
        axi = fig.add_subplot(inner[0])
        axi.axis("off")
        axi.text(0.0, 1.0,
                 "board %d minus board %d (%s)\n"
                 "peak    %+.2f, %+.2f px   =   %+.2f, %+.2f mm\n"
                 "spread  %.2f, %.2f px\n%s\n"
                 "(peak and spread are defined in the caption)"
                 % (b, anchor, role, pc, pr, pc * PIXEL_PITCH, pr * PIXEL_PITCH,
                    sc, sr, lean),
                 transform=axi.transAxes, ha="left", va="top", fontsize=9.8,
                 color=INK, bbox=ANN_BOX, zorder=6)

        axd = fig.add_subplot(inner[1])
        axr = fig.add_subplot(inner[2])
        ks = np.arange(-6, 7, dtype=float)
        for a, prof, nm, colr, pk, sig in (
                (axd, img.sum(axis=0), "dcol", CAT[0], pc, sc),
                (axr, img.sum(axis=1), "drow", CAT[2], pr, sr)):
            tot = prof.sum()
            a.bar(ks, prof / tot if tot else prof, width=0.82, color=colr,
                  edgecolor="white")
            a.axvline(0, color=INK2, lw=0.9, ls=":")
            a.set_xlabel("%s [pixels]%s" % (nm, "   (+ = physically left)"
                                            if nm == "dcol" else ""))
            a.set_ylabel("fraction of events", fontsize=9.5)
            a.axvline(pk, color="crimson", lw=1.5, ls="--",
                      label="peak %+.2f px = %+.2f mm" % (pk, pk * PIXEL_PITCH))
            a.set_xlim(-6.6, 6.6)
            top = float((prof / tot if tot else prof).max())
            a.set_ylim(0, top * 1.45 if top > 0 else 1)
            a.grid(axis="y", alpha=0.3)
            a.legend(loc="upper left", fontsize=9, frameon=True, framealpha=0.9)
            if nm == "dcol":
                phys_col_axis(a)
        axd.set_title("Sideways landing (dcol)", fontsize=10.5)
        axr.set_title("Vertical landing (drow)", fontsize=10.5)

    extra = (" One board is never measured together with the trigger board, so "
             "the reference board is used as the anchor for that row and the row "
             "title says so." if fallback else "")
    caption = (
        "For every event that gave a clean coincidence, we take the pixel that "
        "fired on this board and subtract the pixel that fired on the trigger "
        "board; the sketch on the left shows exactly that subtraction, and the "
        "numbers for the row sit above the two projections on the right. (0, 0) "
        "means the track hit the same pixel index on both boards, and the width of "
        "the blob is how far a single track wanders between the two planes. %s. "
        "Each board is shown with the board set that has the most statistics among "
        "those containing the trigger board, named in the panel title.%s "
        "How to read it: a peak at (0, 0) means the two boards are aligned and the "
        "tracks cross them perpendicularly. A shifted peak means the boards sit at "
        "different places in the beam and/or the tracks cross them at an angle - "
        "the two cannot be separated without a mechanical survey; the pipeline's radius "
        "cut is centred by the yaml translation, and this peak is the value to put "
        "there. A wider blob for a pair with a "
        "longer lever arm is the angular spread of the tracks; a blob only one or "
        "two pixels wide means the beam is narrow and the tracks nearly parallel, "
        "which is the normal case for a small centred spot. A tail on one side "
        "only means tracks tilted predominantly one way, as for a telescope sitting "
        "beside the beam rather than in it (combinatorial background feeds such a "
        "tail too). A peak far from (0, 0) together with a narrow blob is a pure "
        "mechanical offset. "
        "How the numbers are computed: the map and both histograms are over the "
        "sampled events step 6 read, with each track candidate weighted by how many "
        "events share its pixel pattern. 'peak' is the most common whole-pixel "
        "shift, refined by the count-weighted average of the shifts within +-1 "
        "pixel of it (a peak estimator, so the combinatorial tail does not pull it). "
        "'spread' is the count-weighted standard deviation of the shifts. 'skewed "
        "toward' is the sign of the count-weighted skewness of the dcol "
        "distribution (a lopsided tail, usually the combinatorial background)."
        % (PX_MM, extra))
    return finish(fig, out,
                  "Where does a track land on each board relative to the "
                  "trigger board?", sub, caption, w_pad=2.0, h_pad=2.6)

# =============================================================================
# 5. beam_tilt.png - how tilted are the tracks through the telescope?
# =============================================================================
def tilt_schematic(ax, bs):
    """Side view: the planes as vertical bars along the beam, one tilted track,
    and the shift the track picks up per plane gap."""
    ax.set_xlim(-1.6, 3.9)
    ax.set_ylim(-4.2, 3.4)
    ax.axis("off")
    n = len(bs) if bs else 4
    slope = 0.62
    for i in range(n):
        ax.plot([i, i], [-2.5, 2.5], "-", color="#9a9a9a", lw=6, solid_capstyle="butt",
                zorder=1)
        ax.text(i, 2.75, "board %d" % (bs[i] if bs else i), ha="center", va="bottom",
                fontsize=9.5, color=INK2)
    xs = np.arange(n, dtype=float)
    ys = (xs - (n - 1) / 2.0) * slope
    ax.plot(xs, ys, "-", color=CAT[1], lw=2.4, zorder=3)
    ax.plot(xs, ys, "o", color=CAT[1], ms=7, zorder=4)
    ax.annotate("", xy=(1.0, ys[1]), xytext=(1.0, ys[0]),
                arrowprops=dict(arrowstyle="<|-|>", color=CAT[0], lw=1.6))
    ax.text(1.12, 0.5 * (ys[0] + ys[1]), "shift per\nplane gap", fontsize=9.5,
            color=CAT[0], va="center", ha="left")
    ax.annotate("", xy=(n - 1 + 0.02, -3.0), xytext=(0.0, -3.0),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.4))
    ax.text(0.5 * (n - 1), -3.25, "beam direction (downstream)", ha="center",
            va="top", fontsize=9.5, color=INK2)
    ax.text(-1.55, ys[0], "particle\ntrack", color=CAT[1], fontsize=9.5,
            va="center", ha="left")
    ax.set_title("Side view: the shift a track picks up\nfrom one plane to the "
                 "next", fontsize=10.5)


def plot_beam_tilt(df, out, sub):
    combos = [c for c in natsorted(set(q(df, section="angles")["combo"])) if c]
    combos = [c for c in combos
              if len(q(df, section="angles", key="outer_offset_frac_sel", combo=c))]
    if not combos:
        return None
    fig = plt.figure(figsize=(16.5, 10.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.62, 1.5])
    tilt_schematic(fig.add_subplot(gs[:, 0]), boards_of(df))

    axc = fig.add_subplot(gs[0, 1])
    axr = fig.add_subplot(gs[1, 1])
    nc = len(combos)
    w = 0.9 * PIXEL_PITCH / 3.0 / max(nc, 1)
    span = {"dcol": [], "drow": []}
    for i, c in enumerate(combos):
        arr = outer_offset_arrays(df, c, "_sel")
        if arr is None:
            continue
        dc, dr, frac = arr
        lev = one(df, "angles", "lever_planes", combo=c)
        off = (i - (nc - 1) / 2.0) * w
        for a, v, nm in ((axc, dc, "dcol"), (axr, dr, "drow")):
            x = v / lev * PIXEL_PITCH
            ux, inv = np.unique(x, return_inverse=True)
            uy = np.bincount(inv, weights=frac, minlength=len(ux))
            m = one(df, "angles", "mean_d%s_per_plane_sel" % nm[1:], combo=c)
            s = one(df, "angles", "sigma_d%s_per_plane_sel" % nm[1:], combo=c)
            a.bar(ux + off, uy, width=w, color=CAT[i % len(CAT)],
                  edgecolor="white", lw=0.4,
                  label="%s\nmean %+.2f +- %.2f mm/gap  =  %+.3f +- %.3f px/gap"
                        % (combo_plain(df, c), m * PIXEL_PITCH, s * PIXEL_PITCH, m, s))
            keep = uy > 2e-3
            if keep.any():
                span[nm] += [float(ux[keep].min()), float(ux[keep].max())]
    for a, nm, ttl in ((axc, "col", "Sideways tilt (col direction)"),
                       (axr, "row", "Vertical tilt (row direction)")):
        a.axvline(0.0, color=INK2, lw=1.0, ls=":")
        a.set_xlabel("shift per plane gap in the %s direction [mm]%s"
                     % (nm, "   (+ = track moves physically LEFT going downstream)"
                        if nm == "col" else ""))
        a.set_ylabel("fraction of tracks")
        sp = span["dcol" if nm == "col" else "drow"]
        lo = min(sp + [-0.4]) - 0.35 if sp else -1.6
        hi = max(sp + [0.4]) + 0.35 if sp else 2.2
        a.set_xlim(lo, hi)
        a.grid(axis="y", alpha=0.3)
        a.set_title(ttl, fontsize=10.5)
        a.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=9,
                 frameon=False, labelspacing=0.9, handlelength=1.2)
        if nm == "col":
            phys_col_axis(a)

    # Average tilt over the board sets, each set weighted by its own statistics:
    # one number per axis that the caption can quote as a measurement.
    acc, wsum = {"col": 0.0, "row": 0.0}, 0.0
    for c in combos:
        cw = combo_counts(df, c)
        mc = one(df, "angles", "mean_dcol_per_plane_sel", combo=c)
        mr = one(df, "angles", "mean_drow_per_plane_sel", combo=c)
        if cw > 0 and np.isfinite(mc) and np.isfinite(mr):
            acc["col"] += cw * mc
            acc["row"] += cw * mr
            wsum += cw
    tc = acc["col"] / wsum if wsum else np.nan
    tr = acc["row"] / wsum if wsum else np.nan
    if np.isfinite(tc) and np.isfinite(tr):
        meas = ("Measured here: averaged over the board sets (each weighted by its "
                "own candidate statistics) the tilt is %+.3f px per plane gap in "
                "col, i.e. %+.2f mm per gap, and %+.3f px per plane gap in row, "
                "i.e. %+.2f mm per gap."
                % (tc, tc * PIXEL_PITCH, tr, tr * PIXEL_PITCH))
    else:
        meas = ""
    caption = (  # noqa: E501
        "Each selected track is compared between the two OUTERMOST boards of its "
        "board set, and that pixel shift is divided by the number of plane gaps in "
        "between, then turned into millimetres with %s. Zero means the track hits "
        "the same pixel on every plane. A positive col shift means the track moves "
        "physically LEFT as it travels downstream. Shifts are whole pixels, which "
        "is why the bars sit at discrete positions. An angle in mrad would need the "
        "spacing between the planes along the beam, so the tilt is quoted per plane "
        "gap, in pixels and in mm. %s "
        "How to read it: as a rule of thumb, a tilt below about 0.1 px per plane "
        "gap means the tracks run along the telescope axis to within the pixel "
        "resolution; around 0.3 px per gap - the sort of value a telescope standing "
        "beside the beam rather than in it sees - is clearly tilted; above 1 px per "
        "gap means either a large angle or a board that is not where it is assumed "
        "to be. Board sets that agree with each other mean all of them see the same "
        "beam and the planes are roughly equally spaced; board sets that disagree "
        "mean unequal spacing or a misplaced board. Wide bars mean a large angular "
        "spread among the tracks, which is not the same thing as a tilt of the beam "
        "as a whole; bars concentrated in one or two positions mean a narrow, nearly "
        "parallel beam. How the numbers in the legend are computed: 'mean' is "
        "the count-weighted mean over the selected tracks of that board set (each "
        "track candidate weighted by how many events share its pixel pattern), and "
        "the '+-' after it is the count-weighted standard deviation of the same "
        "distribution - the SPREAD of the tracks themselves, not the uncertainty "
        "on the mean, which would be far smaller." % (PX_MM, meas))
    return finish(fig, out, "How tilted are the tracks (the beam) through the "
                            "telescope?", sub, caption, w_pad=2.0, h_pad=3.2)


# =============================================================================
# 6. board_rotation.png (--extended) - are the boards rotated relative to each
#    other?
# =============================================================================
def rotation_schematic(ax, deg=9.0):
    ax.set_xlim(-5.6, 9.6)
    ax.set_ylim(-7.6, 6.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((-4.0, -4.0), 8.0, 8.0, fc="#eeeeee", ec="#9a9a9a",
                           lw=1.5, zorder=1))
    rot = Rectangle((-4.0, -4.0), 8.0, 8.0, fc="none", ec=CAT[1], lw=2.2, zorder=3)
    rot.set_transform(Affine2D().rotate_deg(deg) + ax.transData)
    ax.add_patch(rot)
    rr, th = 6.4, np.radians(deg)
    ax.plot([0, rr], [0, 0], ":", color="#9a9a9a", lw=1.2, zorder=2)
    ax.plot([0, rr * np.cos(th)], [0, rr * np.sin(th)], ":", color=CAT[1], lw=1.2,
            zorder=4)
    ax.add_patch(Arc((0, 0), 2 * 5.4, 2 * 5.4, theta1=0.0, theta2=deg,
                     color=CAT[1], lw=2.0, zorder=5))
    ax.text(6.7, 0.55, "rotation\nangle", color=CAT[1], fontsize=9.5, ha="left",
            va="center")
    ax.text(0.0, -5.1, "grey = one board\norange = the other, rotated by a small\n"
                       "angle about the beam axis", ha="center", va="top",
            fontsize=9.5, color=INK)
    ax.set_title("What a relative rotation looks like", fontsize=10.5)


def panel_rotation(ax, df, combo, pn, key):
    """Mean offset in one coordinate against position in the OTHER coordinate on
    the anchor board - a straight rising line is a relative rotation."""
    r = q(df, section="rotation", combo=combo, pair=pn)
    prof = r[r["key"] == "profile_" + key].sort_values("col")
    err = r[r["key"] == "profile_err_" + key].sort_values("col")
    if len(prof) == 0:
        ax.axis("off")
        return np.nan
    X = prof["col"].astype(int).to_numpy(float)
    Y = prof["value"].to_numpy(float)
    E = err["value"].to_numpy(float)
    ax.errorbar(X, Y, yerr=E, fmt="o", ms=5, color="#31688e", capsize=2.5, lw=1.2)
    slope = one(df, "rotation", "slope_" + key, combo=combo, pair=pn)
    ang = one(df, "rotation", "angle_deg_from_" + ("drow" if key.startswith("drow")
                                                   else "dcol"), combo=combo, pair=pn)
    # How much of the array the beam lights up. summarize refuses the fit below
    # MIN_ROT_SPAN lit anchor rows/cols, and the panel says so instead of drawing a
    # line through a couple of points.
    span = one(df, "rotation", "profile_span_" + key, combo=combo, pair=pn)
    if not np.isfinite(span):
        span = float(len(X))
    lit = "columns" if key.endswith("_col") else "rows"
    if span < MIN_ROT_SPAN:
        ax.text(0.5, 0.5, "beam too narrow for a rotation estimate\n"
                          "(only %d of %d anchor %s illuminated)"
                % (int(span), GRID, lit), transform=ax.transAxes, ha="center",
                va="center", fontsize=10, color="crimson", bbox=ANN_BOX, zorder=6)
    if np.isfinite(slope) and len(X) > 1:
        b0 = np.average(Y - slope * X)
        xx = np.array([0.0, 15.0])
        ax.plot(xx, slope * xx + b0, "-", color="crimson", lw=1.8,
                label="fit %+.4f px of offset per px\n  of position  =  %+.2f deg"
                      % (slope, ang))
        ax.legend(loc="best", fontsize=9, frameon=True, framealpha=0.9)
    on_col = key.startswith("drow")
    a, b = [int(x) for x in pn.split("v")]
    ax.set_xlabel("column looked at on board %d [pixels]" % min(a, b) if on_col
                  else "row looked at on board %d [pixels]" % min(a, b))
    ax.set_ylabel("mean row offset [pixels]" if on_col
                  else "mean col offset [pixels]   (+ = left)")
    ax.grid(alpha=0.3)
    ax.set_title("boards %d and %d   (board set %s)\n%s\nbeam lights %d of %d "
                 "anchor %s"
                 % (a, b, combo_plain(df, combo).split(" ")[1],
                    "row offset vs column" if on_col else "col offset vs row",
                    int(span), GRID, lit),
                 fontsize=10)
    if on_col:
        phys_col_axis(ax)
    return ang


def plot_board_rotation(df, out, sub):
    items = pairs_vs_anchor(df, section="rotation")[:4]
    if not items:
        return None
    n = len(items)
    fig = plt.figure(figsize=(4.4 + 4.4 * n, 11.8))
    gs = fig.add_gridspec(2, n + 1, width_ratios=[1.05] + [1.0] * n)
    rotation_schematic(fig.add_subplot(gs[:, 0]))
    angs = []
    for j, (combo, pn) in enumerate(items):
        angs.append(panel_rotation(fig.add_subplot(gs[0, j + 1]), df, combo, pn,
                                   "dcol_vs_row"))
        angs.append(panel_rotation(fig.add_subplot(gs[1, j + 1]), df, combo, pn,
                                   "drow_vs_col"))
    good = [abs(a) for a in angs if np.isfinite(a)]
    worst = max(good) if good else float("nan")
    if np.isfinite(worst):
        worsttxt = ("Measured here: the largest angle in this figure is %.2f deg, "
                    "which moves the far edge of the array by %.2f px."
                    % (worst, GRID * np.tan(np.radians(worst))))
    else:
        worsttxt = ("Measured here: no panel of this figure yielded a fitted angle, "
                    "so no rotation is quoted for this run.")
    caption = (
        "If two boards are rotated with respect to each other about the beam axis, "
        "then the row offset between them grows steadily with the column you look "
        "at, and the col offset grows with the row. Each panel is that test, and "
        "the red line is a straight-line fit converted to an angle in degrees. "
        "How the numbers are computed: "
        "each point is the count-weighted mean offset of the tracks sitting in one "
        "column (or row) of the anchor board, and its error bar is the weighted "
        "spread in that bin divided by the square root of the effective number of "
        "independent tracks in it. The fitted slope says, in plain words, how much "
        "the offset changes per pixel of position on the anchor board: in the top "
        "row, how much the sideways (col) shift changes per row of the anchor "
        "board; in the bottom row, how much the vertical (row) shift changes per "
        "column. The quoted degrees are arctan of the slope. "
        "How to read it: flat profiles mean no relative rotation between the two "
        "boards. A clear slope means that board is rotated about the beam axis with "
        "respect to the anchor, and the slope read as radians is that angle. A "
        "rigid rotation by a small angle theta gives a slope of about -theta "
        "radians in the top row and about +theta in the bottom row - opposite "
        "signs, same size; slopes of opposite sign but clearly unequal size mean a "
        "shear or perspective term (tilted planes, for instance) rather than a "
        "rigid rotation. A panel can also be undecidable: this test needs the beam "
        "to illuminate a range of rows and columns on the anchor board, so when "
        "fewer than %d of the %d anchor rows (or columns) hold tracks - a narrow, "
        "centred beam - no line is fitted and the panel says so. "
        "What an angle means in practice: a rotation theta shifts the far edge of "
        "the 16-pixel array by 16 px x tan(theta), so 0.5 deg is 0.14 px, 2 deg is "
        "0.56 px and 4 deg is 1.1 px. Anything below about 1 deg is therefore "
        "negligible against a radius window of a few pixels (path_finder's "
        "--max_diff_pixel) and starts to matter only if that window is 1 pixel. %s"
        % (MIN_ROT_SPAN, GRID, worsttxt))
    return finish(fig, out, "Are the boards rotated relative to each other?", sub,
                  caption, w_pad=2.0, h_pad=3.0)


# =============================================================================
# figure index written next to the PNGs
# =============================================================================
FIG_DOC = [
    ("hit_maps.png", "Where do particles hit each board?",
     "One count-weighted map per board, drawn as you would see the chip looking "
     "down at it with the wire-bond pads along the bottom edge.",
     "How to read it: a flat map means the plane sits in a diffuse halo rather "
     "than in a focused beam (what a telescope off the beam axis sees); a bright "
     "compact spot means a focused beam crosses there; a gradient, or one chip half "
     "brighter than the other, means a flux gradient across the telescope and/or "
     "the chip-half difference shown in `cal_tot_halves.png`; dark edge "
     "rows or columns are the geometric acceptance of the coincidence with the "
     "neighbouring boards. What the cross means depends on the illumination: with a "
     "compact spot, as a narrow beam centred on the array gives, the centre of "
     "gravity IS the beam centre; under flat halo illumination it is not a beam "
     "position at all and mostly reflects the flux gradient. The count-weighted "
     "spread quoted in the caption tells the two apart - a few pixels means a spot, "
     "half the array means flat illumination. 1 pixel = 1.3 mm, "
     "full array 20.8 x 20.8 mm.",
     "How the numbers are computed: each cell adds up the sampled events - the "
     "events step 6 read - that gave a clean coincidence with the hit on that "
     "pixel, so one track candidate contributes as many counts as the events that "
     "share its pixel pattern. The centroid (red x) is the centre of gravity of "
     "that map: the mean col and the mean row of the map, weighted by the counts, "
     "turned into mm with 1 pixel = 1.3 mm measured from the array centre (pixel "
     "7.5)."),
    ("occupancy.png", "How busy is each board?",
     "Raw decoded hits with no track requirement, plus hits per event and the "
     "0 / 1 / 2+ hit fractions per board.",
     "How to read it: hits per event around 1 with some 10-20% of events showing "
     "2 or more hits is a busy halo; hits per event well above 1, or single pixels "
     "far brighter than their neighbours in the raw map, points at noise or a badly "
     "behaved channel; a large no-hit fraction is expected rather than alarming - "
     "at IRRAD the trigger is the AND of two boards (which two varies by run) and it "
     "is ONE whole-chip trigger bit, not a per-pixel one, so a triggered board can "
     "still record nothing in an event; on top of that the DAQ drops or truncates "
     "events, and the chip's fast-readout mode omits an event's data words once the "
     "L1 buffer is 96 or more deep. Since the analysis requires a hit on the pixels "
     "of interest, incomplete events fall out downstream. A step in HIT RATE at the "
     "col 7 | 8 line is a chip-half effect (the halves' TDCs sit on different "
     "supplies and their discriminators can differ slightly); a step in the TOT CODE "
     "at that line is only the change of TDC bin size and says nothing about "
     "threshold.",
     "How the numbers are computed: nothing here is track-selected. The number of "
     "files read and the number of events they hold are quoted in the caption. "
     "hits/event = all hits that board recorded in those files, divided by that "
     "number of events read. The 0 / 1 / 2+ percentages are fractions of the same "
     "event count, so the three add up to 1 by construction."),
    ("cal_tot_halves.png", "What do CAL and TOT look like per pixel, and do the "
     "two chip halves differ?",
     "Three per-pixel maps - CAL mode, median raw TOT code, and that TOT converted "
     "to nanoseconds with each pixel's own TDC bin - plus all three averaged down "
     "the columns across the col 7 | 8 boundary.",
     "How to read it: CAL is the number of delay-cell gates that flip in the fixed "
     "3.125 ns reference interval, so that pixel's TDC bin is 3.125 ns / CAL and "
     "every TOA and TOT code is a COUNT OF THOSE BINS - where CAL is higher the bin "
     "is smaller and the SAME physical time reads a HIGHER code. A CAL step exactly "
     "at the col 7 | 8 line is an expected chip-half feature: per the ETROC2 "
     "Reference Manual (rev 0.6, sec 3.7.1, Table 20) the TDCs of the right half "
     "(col<8) are powered from the DIGITAL supply and those of the left half "
     "(col>=8) from the ANALOG (discriminator) supply. A smooth column-to-column or "
     "row-to-row grade in CAL is fabrication process variation of the delay line. A "
     "TOT-code step that follows the CAL step is just the change of unit: convert "
     "with the per-pixel CAL and it disappears, which is what the ns map and the "
     "flat ns trace show. A step that REMAINS in ns would be a real threshold or "
     "gain difference between the halves - that is the only version worth chasing. "
     "Isolated odd pixels in any map are ones to watch for masking, and pixels with "
     "no measurement are left grey (on the TOT maps, pixels no hit landed on, which "
     "is most of the array when the beam is a narrow spot). The pipeline uses the "
     "per-pixel CAL, so neither the step nor the grade biases the timing.",
     "How the numbers are computed: CAL per pixel is the mode, i.e. the commonest "
     "CAL code seen on that pixel (exactly the per-pixel value the pipeline uses), "
     "and the TDC bin width it implies is 3.125 ns divided by CAL. TOT per pixel "
     "is the median of the raw TOT codes recorded on that pixel. TOT in ns per pixel "
     "is the median over that pixel's hits of (2 * tot - floor(tot / 32)) x 3.125 ns "
     "/ CAL, using that pixel's own CAL mode, and is NaN where the pixel has no CAL "
     "entry. The per-half numbers in the titles are medians over the pixels of each "
     "half, the column curves are the mean over the rows of each column with "
     "unmeasured pixels ignored, and the quoted step is the left half minus the "
     "right half; the right/left ratios quoted in the caption are measurements of "
     "this run."),
    ("track_landing.png", "Where does a track land on each board relative to the "
     "trigger board?",
     "Pixel on this board minus pixel on the trigger board, as a 2D map and as "
     "sideways / vertical projections, with a sketch of the subtraction.",
     "How to read it: (0,0) is the same pixel index on both boards. A peak there "
     "means the boards are aligned and the tracks perpendicular; a shifted peak "
     "means a mechanical offset and/or tracks crossing at an angle - the two are "
     "not separable without a survey; the pipeline's radius cut is centred by the "
     "yaml translation, and this peak is the value to put there; a wider blob for a pair with a longer lever arm is angular spread, "
     "while a blob only one or two pixels wide means a narrow, nearly parallel beam; "
     "a tail on one side only means tracks tilted predominantly one way; a peak far "
     "from (0,0) together with a narrow blob is a pure mechanical offset.",
     "How the numbers are computed: the map and both histograms are over the "
     "sampled events step 6 read, with each track candidate weighted by how many "
     "events share its pixel pattern. peak = the most common whole-pixel shift, "
     "refined by the count-weighted average of the shifts within +-1 pixel of it "
     "(a peak estimator, so the combinatorial tail does not pull it). spread = the "
     "count-weighted standard deviation of the shifts. 'skewed toward' is the sign "
     "of the count-weighted skewness of the dcol distribution."),
    ("beam_tilt.png", "How tilted are the tracks (the beam) through the telescope?",
     "Shift between the outermost boards of each board set, divided by the number "
     "of plane gaps and converted to mm.",
     "How to read it: the tilt is quoted per plane gap, in pixels and in mm, since "
     "an angle in mrad would need the plane spacing along the beam. As a rule of "
     "thumb, |tilt| below about 0.1 px per gap means the tracks run along the "
     "telescope axis to within the pixel resolution; around 0.3 px per gap - what a "
     "telescope standing beside the beam sees - is clearly tilted; above 1 px per "
     "gap means a large angle or a board out of place. Agreement between the board "
     "sets means roughly equally spaced planes and the same beam seen by all of "
     "them, disagreement means unequal spacing or a misplaced board; wide bars mean "
     "a large angular spread among the tracks, while bars concentrated in one or two "
     "positions mean a narrow, nearly parallel beam. A positive col "
     "shift means the track moves physically left going downstream. The caption "
     "quotes the measured average tilt of this run in both units.",
     "How the numbers are computed: for each selected track the pixel shift between "
     "the two outermost boards of its board set is divided by the number of plane "
     "gaps between them and converted with 1 pixel = 1.3 mm. The quoted mean is the "
     "count-weighted mean over those selected tracks, and the '+-' after it is the "
     "count-weighted standard deviation of the same distribution - the SPREAD of "
     "the tracks, not the uncertainty on the mean."),
    ("board_rotation.png", "Are the boards rotated relative to each other? "
     "(written only with --extended)",
     "Mean offset in one coordinate against position in the other, with a fit "
     "converted to degrees, next to a sketch of what a rotation does.",
     "How to read it: flat profiles mean no relative rotation; a slope means that "
     "board is rotated about the beam axis with respect to the anchor, and the "
     "slope read as radians is the angle; slopes of opposite sign but clearly "
     "unequal size mean a shear or perspective term (tilted planes, for instance) "
     "rather than a rigid rotation. What an angle means in practice: a rotation "
     "theta shifts the far edge of the 16-pixel array by 16 px x tan(theta), so "
     "0.5 deg is 0.14 px, 2 deg is 0.56 px and 4 deg is 1.1 px - below about 1 deg "
     "it is negligible against a radius window of a few pixels (path_finder's "
     "--max_diff_pixel) and matters only if that window is 1 pixel. The test also "
     "needs the "
     "beam to illuminate a range of anchor rows and columns: each panel states how "
     "many of the 16 are lit, and with fewer than 8 no line is fitted and the panel "
     "reads 'beam too narrow for a rotation estimate' (the stored slope and angle "
     "are NaN).",
     "How the numbers are computed: each point is the count-weighted mean offset of "
     "the tracks sitting in one column (or row) of the anchor board, with an error "
     "bar of the weighted spread in that bin divided by the square root of the "
     "effective number of independent tracks in it. The fitted slope is how much "
     "the offset changes per pixel of position on the anchor board: how much the "
     "sideways (col) shift changes per row of the anchor board, and how much the "
     "vertical (row) shift changes per column. A rigid rotation by a small angle "
     "theta gives a slope of about -theta radians in the col-shift-vs-row panel and "
     "about +theta in the row-shift-vs-column panel - opposite signs, same size - "
     "so a pair of panels that does not mirror itself like that is not a pure rigid "
     "rotation; the quoted degrees are arctan of the slope."),
]


def plot_single(df, outdir, extended=False):
    run = df["run"].iloc[0]
    d = os.path.join(outdir, run)
    os.makedirs(d, exist_ok=True)
    sub = "run %s" % run
    written = [plot_hit_maps(df, os.path.join(d, "hit_maps.png"), sub)]
    if (df["section"] == "raw").any():
        written.append(plot_occupancy(df, os.path.join(d, "occupancy.png"), sub))
    written.append(plot_cal_tot_halves(df, os.path.join(d, "cal_tot_halves.png"), sub))
    written.append(plot_track_landing(df, os.path.join(d, "track_landing.png"), sub))
    written.append(plot_beam_tilt(df, os.path.join(d, "beam_tilt.png"), sub))
    if extended:
        written.append(plot_board_rotation(df, os.path.join(d, "board_rotation.png"),
                                          sub))
    print("wrote %d figure(s) to %s" % (len([w for w in written if w]), d))


# =============================================================================
# plot: cross-run comparison
# =============================================================================
def run_annotation(df, run, label_key):
    """'<label_key>=<value>' averaged over boards, plus the irradiation tag."""
    sub = df[(df["run"] == run) & (df["section"] == "meta")]
    v = sub.loc[sub["key"] == label_key, "value"].dropna()
    txt = sub.loc[sub["key"] == label_key, "text"]
    irr = sub.loc[sub["key"] == "irrad", "text"]
    parts = []
    if len(v):
        parts.append("%s=%g" % (label_key, v.iloc[0] if v.nunique() == 1 else v.mean()))
    elif len(txt) and txt.iloc[0]:
        parts.append("%s=%s" % (label_key, txt.iloc[0]))
    if len(irr) and irr.iloc[0]:
        parts.append(irr.iloc[0])
    return " ".join(parts)


def plot_compare(df, outdir, label_key):
    runs = natsorted(set(df["run"]))
    d = os.path.join(outdir, "compare")
    os.makedirs(d, exist_ok=True)
    x = np.arange(len(runs), dtype=float)
    ann = [run_annotation(df, r, label_key) for r in runs]
    xt = ["%s\n%s" % (r, a) for r, a in zip(runs, ann)]

    def series(section, key, **kw):
        return np.array([one(df[df["run"] == r], section, key, **kw) for r in runs])

    combos = natsorted(set(df.loc[df["section"] == "angles", "combo"]))
    combos = [c for c in combos if c]
    boards = boards_of(df[df["run"] == runs[0]])
    trig = board_by_role(df[df["run"] == runs[0]], "trig")
    has_raw = (df["section"] == "raw").any()

    fig, axs = plt.subplots(4, 2, figsize=(18, 19))

    # (1) beam tilt, quoted in mm per plane gap (stored values are px; x1.3)
    ax = axs[0, 0]
    for i, c in enumerate(combos):
        for key, ls, mk, nm in (("dcol", "-", "o", "sideways (col)"),
                                ("drow", "--", "s", "vertical (row)")):
            y = series("angles", "mean_%s_per_plane_sel" % key, combo=c) * PIXEL_PITCH
            e = series("angles", "sigma_%s_per_plane_sel" % key, combo=c) * PIXEL_PITCH
            ax.errorbar(x, y, yerr=e, ls=ls, marker=mk, ms=5, capsize=2,
                        color=CAT[i % len(CAT)], alpha=0.9,
                        label="%s - %s" % (combo_plain(df, c), nm))
    ax.axhline(0, color=INK2, lw=0.8, ls=":")
    ax.set_ylabel("mean shift per plane gap [mm]")
    ax.set_title("(1) Beam tilt per plane gap [mm]", fontsize=11)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)

    # (2) pairwise peak offset vs the trigger board
    ax = axs[0, 1]
    for i, b in enumerate(boards):
        if trig is None or b == trig:
            continue
        pn = pair_name(min(b, trig), max(b, trig))
        combo = None
        for c in natsorted(set(df.loc[df["section"] == "pairwise", "combo"])):
            if len(q(df[df["run"] == runs[0]], section="pairwise", combo=c, pair=pn)):
                combo = c
                break
        if combo is None:
            continue
        sgn = 1.0 if b < trig else -1.0
        for key, mk, nm in (("peak_dcol", "o", "sideways (dcol)"),
                            ("peak_drow", "s", "vertical (drow)")):
            ax.plot(x, sgn * series("pairwise", key, combo=combo, pair=pn), marker=mk,
                    color=CAT[i % len(CAT)], label="board %d - %s" % (b, nm))
    ax.axhline(0, color=INK2, lw=0.8, ls=":")
    ax.set_ylabel("landing peak, board minus trigger [px]")
    ax.set_title("(2) Track landing peak vs trigger board [px]", fontsize=11)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)

    # (3) CAL median per board per readout half
    ax = axs[1, 0]
    for i, b in enumerate(boards):
        for half, ls in ((HALF_LO, "-"), (HALF_HI, "--")):
            ax.plot(x, series("cal", "cal_median", board=b, half=half), ls=ls, marker="o",
                    ms=4, color=CAT[i % len(CAT)],
                    label="board %d - %s half" % (
                        b, "right (col<8)" if half == HALF_LO else "left (col>=8)"))
    ax.set_ylabel("CAL median [code]")
    ax.set_title("(3) CAL mode per chip half", fontsize=11)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)

    # (4) raw occupancy and the col<8 (phys. right) / col>=8 (phys. left) balance
    ax = axs[1, 1]
    if has_raw:
        for i, b in enumerate(boards):
            ax.plot(x, series("raw", "hits_per_event", board=b), marker="o",
                    color=CAT[i % len(CAT)], label="board %d hits/event" % b)
        ax2 = ax.twinx()
        for i, b in enumerate(boards):
            ax2.plot(x, series("raw", "ratio_collt8_over_colge8", board=b),
                     ls=":", marker="^", ms=4,
                     color=CAT[i % len(CAT)])
        ax2.set_ylabel("right-half / left-half hit ratio (dotted)")
        ax.set_ylabel("raw hits per event")
        ax.legend(loc="center left", bbox_to_anchor=(1.14, 0.5), fontsize=9)
    else:
        ax.text(0.5, 0.5, "no raw hits stored for these runs", ha="center",
                va="center", transform=ax.transAxes, fontsize=10)
    ax.set_title("(4) Hits per event\n"
                 "(dotted: physical right half col<8 over left half col>=8)",
                 fontsize=11)

    # (5) selection size and typical candidate count
    ax = axs[2, 0]
    for i, c in enumerate(combos):
        ax.plot(x, series("selected", "n_selected", combo=c), marker="o",
                color=CAT[i % len(CAT)], label=combo_plain(df, c))
    ax2 = ax.twinx()
    for i, c in enumerate(combos):
        ax2.plot(x, series("selected", "count_p50", combo=c), ls=":", marker="s", ms=4,
                 color=CAT[i % len(CAT)])
    ax.set_ylabel("tracks kept for timing")
    ax2.set_ylabel("typical events per track (dotted)")
    ax.set_title("(5) Selected tracks per board set", fontsize=11)
    ax.legend(loc="center left", bbox_to_anchor=(1.14, 0.5), fontsize=9)

    # (6) geometric consistency of the candidate sample
    ax = axs[2, 1]
    for i, c in enumerate(combos):
        ax.plot(x, series("candidates", "frac_all_pairs_within_maxdiff", combo=c),
                marker="o", color=CAT[i % len(CAT)], label=combo_plain(df, c))
    ax.set_ylabel("fraction with every board pair consistent")
    ax.set_ylim(0, 1.02)
    ax.set_title("(6) Fraction of geometrically consistent patterns", fontsize=11)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)

    # (7) spacing ratios with their brackets
    ax = axs[3, 0]
    for i, (k, lab) in enumerate((("d12_over_d01", "gap 1-2 over gap 0-1"),
                                 ("d23_over_d01", "gap 2-3 over gap 0-1"))):
        y = series("spacing", k)
        lo = series("spacing", k + "_lo")
        hi = series("spacing", k + "_hi")
        ax.errorbar(x, y, yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)],
                    marker="o", capsize=3, color=CAT[i], label=lab)
    ax.set_ylabel("spacing ratio")
    ax.set_title("(7) Relative plane spacing (factor-2 accurate)", fontsize=11)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)

    axs[3, 1].axis("off")
    for ax in axs.ravel()[:7]:
        ax.set_xticks(x)
        ax.set_xticklabels(xt, fontsize=9.5, rotation=25, ha="right")
        ax.grid(alpha=0.3)
    fig.suptitle("How do these runs compare?  (%d runs, x axis labelled run + %s)"
                 % (len(runs), label_key), fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.86, 0.965])
    out = save_figure(fig, os.path.join(d, "trends.png"), dpi=140)
    plt.close(fig)

    # A machine-readable digest next to the figure: one row per run.
    keys = [("angles", "mean_dcol_per_plane_sel"), ("angles", "mean_drow_per_plane_sel"),
            ("angles", "sigma_dcol_per_plane_sel"), ("angles", "sigma_drow_per_plane_sel"),
            ("candidates", "frac_all_pairs_within_maxdiff"), ("selected", "n_selected"),
            ("selected", "count_p50"), ("pairwise", "peak_dcol"), ("pairwise", "peak_drow"),
            ("cal", "cal_median"), ("raw", "hits_per_event"),
            ("raw", "ratio_collt8_over_colge8"),
            ("hitmap", "centroid_col"), ("hitmap", "centroid_row"),
            ("spacing", "d12_over_d01"), ("spacing", "d23_over_d01"), ("meta", "HV")]
    recs = []
    for r in runs:
        sub = df[df["run"] == r]
        rec = {"run": r, "annotation": run_annotation(df, r, label_key)}
        for section, key in keys:
            rows = q(sub, section=section, key=key, pixels=False)
            for _, rr in rows.iterrows():
                tag = "%s.%s" % (section, key)
                for dim in ("combo", "pair", "half"):
                    if rr[dim]:
                        tag += ".%s" % rr[dim]
                if pd.notna(rr["board"]):
                    tag += ".b%d" % int(rr["board"])
                rec[tag] = rr["value"]
        recs.append(rec)
    table = pd.DataFrame(recs).set_index("run")
    csv = os.path.join(d, "summary_table.csv")
    table.to_csv(csv)
    print("wrote %s and %s (%d runs x %d columns)" % (out, csv, table.shape[0], table.shape[1]))


def cmd_plot(args):
    set_output_options(getattr(args, "format", "png"), getattr(args, "split", False))
    frames = []
    for path in args.inputs:
        if not os.path.isfile(path):
            die("input not found: %s" % path)
        frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    os.makedirs(args.outdir, exist_ok=True)
    runs = sorted(set(df["run"]))
    if len(runs) == 1:
        plot_single(df, args.outdir, extended=getattr(args, "extended", False))
    else:
        plot_compare(df, args.outdir, args.label_key)
    return 0


# =============================================================================
# CLI
# =============================================================================
def build_parser():
    p = argparse.ArgumentParser(
        prog="telescope_diagnostics.py",
        description="Per-run telescope diagnostics: condense one run's step-6/step-7 "
                    "outputs into a single tidy parquet ('summarize'), then draw the "
                    "per-run dashboard or cross-run trends from it ('plot').",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="command")

    s = sp.add_parser("summarize", help="build <label>_diagnostics.parquet for one run")
    s.add_argument("--tracks-dir", required=True, dest="tracks_dir",
                   help="Per-run step-6 output directory holding tracks_<combo>.parquet "
                        "and the matching tracks_<combo>_reduced.parquet from step 7.")
    s.add_argument("--cal-table", required=True, dest="cal_table",
                   help="CAL table CSV written by path_finder.py (columns row,col,board,cal_mode).")
    s.add_argument("-c", "--config", required=True,
                   help="Board configuration YAML (TestBeam/board_configs_yaml/*.yaml).")
    s.add_argument("-r", "--runName", required=True, dest="runName",
                   help="Run key inside the configuration YAML, e.g. h1_run1.")
    s.add_argument("--label", required=True,
                   help="Short run label used for the 'run' column, the output filename "
                        "and the plot subdirectory, e.g. h1_run1.")
    s.add_argument("-o", "--outdir", required=True,
                   help="Directory to write <label>_diagnostics.parquet into (created if needed).")
    s.add_argument("--alignment", default=None,
                   help="Optional alignment YAML from 'path_finder.py --find_alignment'; "
                        "its translations are stored in the parquet (section 'alignment') for reference.")
    s.add_argument("--raw-dir", default=None, dest="raw_dir",
                   help="Optional directory of decoded *.feather files; enables the raw "
                        "occupancy / ToT / CAL-spread section.")
    s.add_argument("--n-raw-files", type=int, default=5, dest="n_raw_files",
                   help="How many naturally-sorted feather files to read from --raw-dir "
                        "(default: 5). Raw numbers are per-file-sample, not whole-run.")
    s.add_argument("--max-diff-pixel", type=float, default=4.0, dest="max_diff_pixel",
                   help="Radius in pixels around a pair's modal offset used for the "
                        "geometric-consistency fractions (default: 4). Diagnostic only; it "
                        "does not re-cut anything.")
    s.set_defaults(func=cmd_summarize)

    d = sp.add_parser("plot", help="draw the dashboard (one input) or trends (several)")
    d.add_argument("-i", "--inputs", required=True, nargs="+",
                   help="One or more *_diagnostics.parquet files. One input writes the "
                        "per-run figure set (one figure per question) into <outdir>/<run>/; "
                        "several draw cross-run trend panels into <outdir>/compare/.")
    d.add_argument("-o", "--outdir", required=True,
                   help="Output directory (created if needed).")
    d.add_argument("--label-key", default="HV", dest="label_key",
                   help="Meta key annotated on the cross-run x axis (default: HV).")
    d.add_argument("--extended", action="store_true",
                   help="Also draw the specialist figures that are not part of the "
                        "everyday set: board_rotation.png (relative rotation of the "
                        "planes about the beam axis).")
    add_output_arguments(d)
    d.set_defaults(func=cmd_plot)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "func", None) is None:
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
