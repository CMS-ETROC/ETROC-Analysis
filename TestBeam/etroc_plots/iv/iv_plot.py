"""iv_plot.py -- shared IV-curve panel drawing for the builder-A figure scripts
(fig21/fig25/fig29/fig31/fig39). Not part of the loader contract (iv_data.py); a small drawing
helper so the marker-thinning / colour / legend conventions are identical across five scripts
instead of retyped five times.

Also carries two safety checks every IV-curve script must run after drawing (2026-09 fig21/fig25
gap-vs-legend post-mortem: a legend box drawn over a climbing curve looked exactly like a missing
scan range, and only re-rendering without the legend proved the underlying data was complete):

* `assert_no_gaps` -- fails loudly if a drawn curve has a real hole in its own V range, so a
  genuinely broken scan can never again be mistaken for a rendering artefact (or the reverse).
* `check_legend_overlap` -- reports every (legend, line) pair where a legend's window extent
  covers an actual data point, so a legend can never again silently sit on top of a curve.
"""
import json
import os

import numpy as np
from matplotlib.legend import Legend
from matplotlib.lines import Line2D

from . import iv_data as ivd
from .. import talk_style as ts

YLABEL_I = r"Leakage current [$\mu$A]"

XMAX_V = 620.0  # shared upper bias limit: overall data maximum (610 V) plus one 10 V bin
XLIM_FULL = (0.0, XMAX_V)


def draw_curves(ax, curves, xlim, ylim=None, yscale="log", ylabel=None, thin=True,
                point_scale=1.0):
    """Draw one telescope's IV panel.

    `curves`: list of dicts with v, i, chip, fluence, and optional color (override fluence
    colour), shade_k (lighten via ivd.shade, for a later look at the same fluence), open_marker,
    line (default True). Sets x/y limits, scale, labels and the repo tick/grid treatment;
    legends are the caller's job since which legends a panel needs varies figure to figure.

    `thin` (default True, unchanged behaviour): markevery thins to ~12 markers per curve via
    ivd.markevery. `thin=False` draws every measured point instead (markevery=1) -- the "every
    data-taking point" convention for fig21b/c/d (2026-09-18): small, un-thinned open markers so
    no measured point is silently dropped from the rendered curve. `point_scale` shrinks the
    marker/line size (ts.point_style's own `scale`) for that dense-point case; default 1.0 keeps
    every other caller's marker size unchanged.
    """
    for c in curves:
        color = c.get("color")
        if color is None:
            color = ts.fluence_color(c["fluence"])
            if c.get("shade_k"):
                color = ivd.shade(color, c["shade_k"])
        kw = ts.point_style(c["chip"], c["fluence"], color=color, scale=point_scale,
                            open_marker=c.get("open_marker", False), line=c.get("line", True))
        kw.pop("capsize", None)
        kw.pop("elinewidth", None)          # errorbar-only; these curves carry no error bars
        # a curve may set its own markevery (the low-V views space markers ~10 V apart
        # via ivp.lowv_markevery, which neither the ~12-marker thinning nor "every point"
        # can express); otherwise `thin` decides.
        kw["markevery"] = c.get("markevery") or (ivd.markevery(len(c["v"])) if thin else 1)
        ax.plot(c["v"], c["i"], **kw)
    ax.set_xlim(*xlim)
    if yscale == "log":
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel(ts.XLABEL_BIAS)
    ax.set_ylabel(ylabel or YLABEL_I)
    ts.style_axes(ax)


def shade_handles(labels_colors, scale=1.0):
    """Small legend for a "time after irradiation" shade ramp: [(label, color), ...]."""
    return [Line2D([], [], color=c, linewidth=3.0 * scale, linestyle="-", label=l)
            for l, c in labels_colors]


# ------------------------------------------------------------------------ v2: distinct timing hues
# v2 review (2026-09-16): several-scans-at-one-fluence figures used ivd.shade() (lighter tints of
# the same hue) to mark "later look at the same fluence"; the user flagged this as hard to read
# ("different colours, not lighter shades"). TIMING_COLORS replaces the shade ramp: the first look
# (right after / March) keeps the fluence-ladder colour, every later look gets its own qualitative
# hue from this list, in order -- never a lighter tint of the same colour. Used by fig25, fig28,
# fig29, fig31, fig39 (v2 only; v1 keeps the shade ramp).
TIMING_COLORS = ["#1b9e9e", "#c03fa0", "#8a9a1c", "#6e6e6e", "#2aa7d6", "#8c5a2b"]  # orange removed 2026-09-19 (now the 3.5e15 ladder colour); old brown takes the last slot


def timing_color(index, base_color):
    """index=0 (right after / March) -> base_color (the fluence ladder colour); index>=1 (a later
    look at the same fluence) -> TIMING_COLORS[index-1], cycling if a sequence ever runs past 6
    later looks (none currently does -- fig31's 7-look sequence uses exactly 6 later colours)."""
    if index <= 0:
        return base_color
    return TIMING_COLORS[(index - 1) % len(TIMING_COLORS)]


# --------------------------------------------------------------------- v2: before/after ratio row
RATIO_MIN_OVERLAP_V = 50.0
RATIO_STEP_V = 1.0


def curve_ratio(v_before, i_before, v_after, i_after, step=RATIO_STEP_V,
               min_overlap=RATIO_MIN_OVERLAP_V):
    """|I(after)| / |I(before)| for one chip, on a common `step`-V grid restricted to the OVERLAP
    of the two scans' bias ranges (no extrapolation past either scan's own range). Returns
    (v_grid, ratio, lo, hi); v_grid/ratio are empty arrays (caller draws no line) when the overlap
    is under `min_overlap` volts, but lo/hi are always the true overlap bounds so the values JSON
    can record why a chip was skipped.
    """
    v_before = np.asarray(v_before, dtype=float)
    v_after = np.asarray(v_after, dtype=float)
    lo = max(float(np.min(v_before)), float(np.min(v_after)))
    hi = min(float(np.max(v_before)), float(np.max(v_after)))
    if hi - lo < min_overlap:
        return np.array([]), np.array([]), lo, hi
    grid = np.arange(np.ceil(lo), np.floor(hi) + 1e-9, step)
    ib = np.interp(grid, v_before, np.abs(np.asarray(i_before, dtype=float)))
    ia = np.interp(grid, v_after, np.abs(np.asarray(i_after, dtype=float)))
    return grid, ia / ib, lo, hi


# 2026-09-18: H1 and F1 (and fig25 vs fig39, same underlying ratio) must share one explicit
# y range instead of each panel autoscaling to its own data -- union of both telescopes' ratio
# values across both "after 4-month rest" stems is [0.121, 1.762] (fig25_rest_iv_1p5e15_v2 and
# fig39_iv_campaign_v2 draw the identical July+4mo/March 1.5e15 ratio); padded to clean round
# numbers that comfortably clear both ends.
RATIO_YLIM = (0.0, 1.9)


def draw_ratio_panel(ax, tel, pairs, xlim, ylabel="I(after rest) / I(before rest)",
                     ylim=None, extra=None):
    """One ratio-vs-bias panel: `pairs` = [(chip, v_before, i_before, v_after, i_after, color)].
    Marker/linestyle per chip (ts.chip_marker/chip_linestyle), one shared colour (the figure's
    "after" colour, passed in per pair since fig25 and fig39 each compute their own). Skips a
    chip whose scan-overlap is under RATIO_MIN_OVERLAP_V and returns its info for the caller's
    footer note + values JSON instead of drawing a line for it.

    `ylim` (default RATIO_YLIM): explicit, no per-panel autoscale, so the H1 and F1 ratio panels
    of every "after 4-month rest" ratio stem (fig25, fig39) share one y range and compare
    directly with each other -- 2026-09-18 fix, see RATIO_YLIM for how it was chosen.
    """
    if ylim is None:
        ylim = RATIO_YLIM
    ax.axhline(1.0, color=ts.INK_MUTED, linestyle="--", linewidth=1.4, zorder=1)
    per_chip, skipped = [], []
    for chip, v_before, i_before, v_after, i_after, color in pairs:
        grid, ratio, lo, hi = curve_ratio(v_before, i_before, v_after, i_after)
        entry = dict(chip=chip, overlap_lo_V=lo, overlap_hi_V=hi, overlap_V=hi - lo)
        if len(grid) == 0:
            skipped.append(chip)
            entry["v_grid_V"], entry["ratio"] = [], []
            per_chip.append(entry)
            continue
        kw = dict(color=color, marker=ts.chip_marker(chip), linestyle=ts.chip_linestyle(chip),
                  linewidth=2.0, markersize=7.0, markevery=ivd.markevery(len(grid)))
        ax.plot(grid, ratio, **kw)
        entry["v_grid_V"], entry["ratio"] = grid.tolist(), ratio.tolist()
        per_chip.append(entry)
    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
    ax.set_xlabel(ts.XLABEL_BIAS)
    ax.set_ylabel(ylabel)
    ts.style_axes(ax)
    # `extra` defaults to nothing: on the shared geometry the compound header tag
    # already says this is a ratio figure, and a second "after / before ratio" clause on
    # the panel title pushed the merged title line left into the experiment text.
    ts.panel_title(ax, tel, extra=extra)
    return per_chip, skipped


# ---------------------------------------------------------------------------- A1: gap assertion
MAX_GAP_BIN_WIDTHS = 2.5


LOCAL_WINDOW = 4      # gaps on each side used for the local-neighbourhood comparison
LOCAL_FACTOR = 4.0    # a gap must also clear this multiple of its own neighbourhood to count


def assert_no_gaps(curves, tag="", skip_below_V=0.0):
    """Fail loudly if any curve has a hole bigger than 2.5 bin widths inside its own V range.

    Every curve dict must carry `bin_width_V` (ivd.bin_width_march / ivd.bin_july; 10 V for a
    quick scan, 0.1 V for a fine scan) -- a curve without one is skipped, not silently passed, but
    every builder in this family sets it. `skip_below_V` drops points below that bias before
    checking (the campaign's fine scans ramp through the first ~10 V on a coarser, ~1 V native
    step regardless of chip or fluence -- real DAQ behaviour at negligible current, not a hole;
    fig21's low-V knee panels pass 10.0 here since that is exactly the range they zoom into).

    A gap only counts once it clears BOTH (a) 2.5x the scan's nominal bin width and (b) 4x the
    median of its own neighbouring gaps. (b) matters because even above skip_below_V a "fine"
    scan is not uniformly 0.1 V end to end -- July's binning coarsens past ~250 V where raw
    samples thin out, which (a) alone would flag on every such curve. A genuine hole (the
    fig21/fig25 2026-09 post-mortem: a legend box drawn over a climbing curve looked exactly like
    a ~200 V missing range) stands out from ITS OWN neighbours however coarse the local sampling
    is, so (b) still catches it. Raises AssertionError immediately -- a real gap must never be
    "fixed" by loosening this check instead of the data, and a legend sitting on top of good data
    must never be "fixed" by loosening it instead of moving the legend (see check_legend_overlap).
    """
    bad = []
    for c in curves:
        bw = c.get("bin_width_V")
        v = np.asarray(c["v"], dtype=float)
        if skip_below_V:
            v = v[v >= skip_below_V]
        if bw is None or len(v) < 2:
            continue
        gaps = np.diff(np.sort(v))
        floor = MAX_GAP_BIN_WIDTHS * bw
        n = len(gaps)
        for gi in range(n):
            g = gaps[gi]
            if g <= floor:
                continue
            lo, hi = max(0, gi - LOCAL_WINDOW), min(n, gi + LOCAL_WINDOW + 1)
            nbhd = np.concatenate([gaps[lo:gi], gaps[gi + 1:hi]])
            local_typical = float(np.median(nbhd)) if len(nbhd) else bw
            if g > LOCAL_FACTOR * max(local_typical, bw):
                bad.append("%s: chip=%s fluence=%s look=%s max_gap=%.3gV > %.3gV floor and "
                          "%.1fx its local %.3gV neighbourhood"
                          % (tag, c.get("chip"), c.get("fluence"), c.get("look"), g, floor,
                             LOCAL_FACTOR, local_typical))
    if bad:
        raise AssertionError("gap check FAILED:\n  " + "\n  ".join(bad))
    print("  gap check on %s: %d curves, no gap > %.1fx bin width" % (tag, len(curves),
                                                                       MAX_GAP_BIN_WIDTHS))
    return []


# ---------------------------------------------------------------------- A3: legend overlap check
def _axes_title_texts(ax):
    """The Text artists ts.panel_title / ax.set_title can populate: centre, left and right.

    matplotlib stores loc="left"/"right" titles separately from the centre one (`ax._left_title`,
    `ax._right_title`) -- `ax.title` alone is the centre slot, empty whenever panel_title's
    loc="left" is used, so checking only `ax.title` silently passes even when the *actual*, visible
    left-aligned title sits right under a legend. Only text artists carrying real text are
    returned, so an unused slot's tiny/undefined bbox can't produce a spurious hit.
    """
    out = []
    for attr in ("title", "_left_title", "_right_title"):
        t = getattr(ax, attr, None)
        if t is not None and t.get_text():
            out.append(t)
    return out


def check_legend_overlap(fig, tag=""):
    """Every legend's window extent must clear every drawn line's actual data points AND the
    axes' own title text (ts.panel_title draws a loc="left" title above the axes; a legend tall
    enough to reach past the axes' top edge sits on top of it -- 2026-09 fig21 low-V-row
    post-mortem: check_no_clipping and the line-vs-legend test above both passed while the title
    was plainly grayed out under the fluence legend in the rendered PNG).

    Compares each Legend artist's bbox (display/pixel coords, after fig.canvas.draw()) against
    the display-space xy of every plotted Line2D's real points (get_xydata through ax.transData)
    on the same axes, and against the axes' title text bbox(es) (see _axes_title_texts). Legend
    *handle* Line2D objects (built with Line2D([], [], ...) for fluence_handles/chip_handles/
    shade_handles) are never added to an axes, so they cannot spuriously match their own legend.
    Returns the list of offending descriptions (empty = clean); does not raise -- the fix is a
    placement change, not a data bug.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    problems = []
    for ai, ax in enumerate(fig.axes):
        legends = [c for c in ax.get_children() if isinstance(c, Legend)]
        lines = ax.get_lines()
        titles = _axes_title_texts(ax)
        for leg in legends:
            lbb = leg.get_window_extent(renderer=r)
            lt = leg.get_title()
            lname = (lt.get_text() if lt else "") or ", ".join(
                t.get_text() for t in leg.get_texts()[:2])
            for ln in lines:
                xy = ln.get_xydata()
                if len(xy) == 0:
                    continue
                disp = ax.transData.transform(xy)
                inside = ((disp[:, 0] >= lbb.x0) & (disp[:, 0] <= lbb.x1) &
                          (disp[:, 1] >= lbb.y0) & (disp[:, 1] <= lbb.y1))
                if inside.any():
                    problems.append("axes[%d] legend %r overlaps line %r (%d/%d points)"
                                    % (ai, lname, ln.get_label(), int(inside.sum()), len(xy)))
            for t in titles:
                tbb = t.get_window_extent(renderer=r)
                if not (tbb.x1 < lbb.x0 or lbb.x1 < tbb.x0 or tbb.y1 < lbb.y0 or lbb.y1 < tbb.y0):
                    problems.append("axes[%d] legend %r overlaps title text %r"
                                    % (ai, lname, t.get_text()))
    print("  legend overlap check on %s: %s" % (tag, "nothing overlaps" if not problems
                                                 else "%d PROBLEM(S)" % len(problems)))
    for p in problems:
        print("      %s" % p)
    return problems


# --------------------------------------------------------------------------------- A2: JSON tags
# The IV/V_gl family draws binned scans, not tracks/combos -- ts.write_values' built-in
# "conventions" block is the resolution-figure template, so every script in this family calls
# write_values() below instead, which swaps in the conventions that actually apply here.
IV_CONVENTIONS = {
    "curves": "binned scans; |V| in V (negative bias applied), |I| in uA; median per bin",
    "binning": "quick = 10 V bins, fine = 0.1 V bins",
    "scan_time": "UTC; March raw logs are recorded local CET = UTC+1",
    "fluence": "facility bookkeeping, p/cm2, 24 GeV protons; no n_eq",
    "v_gl": "gain-layer depletion voltage, k-factor peak of the fine low-V scan",
}


# The block follows the figure KIND: IV_CONVENTIONS describes binned IV scans, and talk_style's
# declares a track selection; a bias-current timeline passes this one instead.
CURRENTS_CONVENTIONS = {
    "current": "HV supply channel current readback, uA, the slow-control log in 60 s bins; "
               "channel 0-3 = the telescope's chips in board order",
    "time": "hours since the first 60 s bin inside the run window [start_utc, start_utc + "
            "max_min] of the per-run current log",
    "run_order": "run order, per-run class and flag as in the campaign's per-run resolution "
                 "table (COMBO_CHECK_JSON in the campaign module)",
    "fluence": "facility bookkeeping, p/cm2, 24 GeV protons; no n_eq",
}


def write_values(out_dir, stem, payload, script=None, inputs=None, conventions=IV_CONVENTIONS):
    """ts.write_values with a figure-kind conventions block in place of the resolution template."""
    path = ts.write_values(out_dir, stem, payload, script=script, inputs=inputs)
    with open(path) as fh:
        doc = json.load(fh)
    doc["conventions"] = conventions
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=False)
    return path


# ==========================================================================================
# SHARED TALK LAYER (2026-09-19) -- the geometry, legend and label conventions of the three
# reference figures the user approved (fig21b_iv_march_v2 / fig21bc_iv_march.py,
# fig21d_iv_march_lowv_v2 / fig21d_iv_march_lowv.py, fig22_currents_march_v2 /
# fig22_currents_march.py), lifted here ONCE so every IV-family script draws through the same
# code path instead of retyping (and drifting from) the template.
#
# Panels matter as much as compounds: single panels are placed side by side on slides, so every
# panel of every figure is exported on the SAME canvas (PANEL_FIGSIZE at 200 dpi = 1900 x 1500
# px) with the SAME axes box (PANEL_AXES), and every 1 x 2 compound on COMPOUND_FIGSIZE with
# COMPOUND_ADJUST.  Nothing in this family may open its own figure.
# ==========================================================================================
import matplotlib.pyplot as _plt
import pandas as _pd

COMPOUND_FIGSIZE = (15.0, 7.6)
COMPOUND_ADJUST = dict(left=0.075, right=0.985, top=0.91, bottom=0.12, wspace=0.24)
PANEL_FIGSIZE = (9.5, 7.5)          # 1.267:1, clears the 1.35:1 panel cap -> 1900 x 1500 px
PANEL_AXES = [0.13, 0.11, 0.82, 0.74]
HEADER_PAD = 8

# low-V knee view, shared by every low-V figure (fig21d's own window)
LOWV_XLIM = (0.0, 60.0)
LOWV_MASK_V = 65.0                  # keep autoscale honest for the 0-60 V zoom
LOWV_MARKER_STEP_V = 10.0           # a marker roughly every 10 V, as fig21d does

# legend placement, straight from the reference scripts (empirically validated there)
LOC_LOOK_LIN, LOC_CHIP_LIN, SCALE_LIN = (0.02, 0.62), (0.02, 0.24), 0.58
LOC_LOOK_LOG, LOC_CHIP_LOG, SCALE_LOG = (0.66, 0.26), (0.66, 0.02), 0.50
LOC_LOOK_LOWV, LOC_CHIP_LOWV, SCALE_LOWV = (0.02, 0.62), (0.02, 0.20), 0.55


# ------------------------------------------------------------------ look labels and colours
def look_label(fluence, when=None):
    """The one label form this family uses (user rule 2026-09-19): '<fluence>, <time after the
    step>', e.g. '1.5x10^15 p/cm2, +2 days'.  Pre-irradiation has no step to count from, so it
    stays 'pre-irradiation'.  No campaign tags ('(Mar)', '(Jul)', '(end)') anywhere.
    """
    k = ts.fluence_key(fluence)
    base = ts.fluence_label(fluence)
    if k == 0.0 or not when:
        return base
    return u"%s, %s" % (base, when)


def look_series(looks, chip_dose=False):
    """[(fluence, when), ...] in drawing order -> [(label, color), ...].

    Within ONE fluence step the first look keeps the fluence-ladder colour and every later look
    gets its own hue from TIMING_COLORS (never a lighter/darker shade of the same colour --
    user rule).  Across different fluence steps each step starts again at its own ladder colour,
    so a campaign figure reads as the fluence ladder while a single-step figure reads as a time
    sequence.

    `chip_dose=True` swaps the label text for the on-chip fluence number (chip_dose_label) while
    the colour/grouping stays keyed to the bookkeeping fluence step -- see CHIP_DOSE_FACTOR.
    """
    label_fn = chip_dose_label if chip_dose else look_label
    seen, out = {}, []
    for fluence, when in looks:
        k = ts.fluence_key(fluence)
        idx = seen.get(k, 0)
        seen[k] = idx + 1
        out.append((label_fn(fluence, when), timing_color(idx, ts.FLUENCE_COLOR[k])))
    return out



# ---- campaign catalogue (see iv/campaigns/) --------------------------------
# These names live in the campaign module now; they are re-bound here so every
# reference in this file and in the figure scripts keeps working unchanged.
from .campaigns import active as _campaign
from .campaigns import check_chips as _check_chips
CHIP_DOSE_FACTOR = _campaign.CHIP_DOSE_FACTOR
CHIP_DOSE_FOOTER = _campaign.CHIP_DOSE_FOOTER
_CHIP_DOSE_TEXT = _campaign._CHIP_DOSE_TEXT
# every panel draws talk_style.TELESCOPE_CHIPS; stop here if the campaign's chips differ
_check_chips(ts.TELESCOPE_CHIPS)



def chip_dose_label(fluence, when=None):
    """look_label's on-chip-fluence counterpart: same '<value>, <time after the step>' form, the
    on-chip fluence number tagged with the IRRAD bookkeeping step it was measured against,
    instead of the bookkeeping fluence value on its own."""
    k = ts.fluence_key(fluence)
    if k == 0.0:
        base = _CHIP_DOSE_TEXT[k]
    else:
        base = "%s (IRRAD step %s) %s" % (_CHIP_DOSE_TEXT[k], ts.fluence_label(fluence, unit=False), ts.FLUENCE_UNIT)
    if k == 0.0 or not when:
        return base
    return u"%s, %s" % (base, when)


def chip_dose_fluence_handles(steps, scale=1.0):
    """ts.fluence_handles' on-chip-fluence counterpart: same colours/steps, on-chip fluence label text."""
    out = []
    for f in sorted({ts.fluence_key(x) for x in steps}):
        out.append(Line2D([], [], color=ts.FLUENCE_COLOR[f], linewidth=3.0 * scale,
                          linestyle="-", label=chip_dose_label(f)))
    return out


def look_handles(pairs, scale=1.0):
    """Legend handles for [(label, color), ...] -- one coloured line per look."""
    return [Line2D([], [], color=c, linewidth=3.0 * scale, linestyle="-", label=l)
            for l, c in pairs]


def two_look_legends(ax, looks, chips, scale=SCALE_LIN, loc_f=LOC_LOOK_LIN, loc_c=LOC_CHIP_LIN,
                     ncol_f=1, ncol_c=1):
    """The reference two-legend layout (ts.two_legends' geometry): the look/fluence legend at
    `loc_f`, the chip legend at `loc_c`, same font scale.  `looks` is [(label, color), ...] from
    look_series(); the labels already carry the fluence, so the box takes no title (a 'fluence'
    title over '2x10^15 p/cm2, +2 days' would name only half of what the entry says).
    """
    s = ts.sizes(scale)
    lf = ax.legend(handles=look_handles(looks, scale), loc=loc_f, fontsize=s["legend"],
                   ncol=ncol_f)
    ax.add_artist(lf)
    lc = ax.legend(handles=ts.chip_handles(chips, scale), loc=loc_c, fontsize=s["legend"],
                   ncol=ncol_c)
    return lf, lc


# ------------------------------------------------------------------ day counting (Part 2 audit)
def days_after_step(fluence, start_utc):
    """Days between a fluence step's end and a scan's own start.

    The step end is iv_data.RAD_STOP_UTC (for the July steps that is the first post-irradiation
    IV scan, an upper bound on the true stop -- so a July day count is a slight UNDER-estimate).
    Returns None for pre-irradiation (no step to count from) or an unparsable time.
    """
    k = ts.fluence_key(fluence)
    if k == 0.0 or k not in ivd.RAD_STOP_UTC or start_utc is None:
        return None
    try:
        t0 = _pd.Timestamp(ivd.RAD_STOP_UTC[k])
        t1 = _pd.Timestamp(start_utc)
    except Exception:
        return None
    return float((t1 - t0).total_seconds()) / 86400.0


def check_days(entries, tag=""):
    """Print (and return) the measured day count of every drawn scan next to the label it was
    given, so a '+4 days' label can never silently sit on a scan taken at +9 days.

    `entries`: [dict(look=, when=, fluence=, scan=, start_utc=), ...].  Reports, never raises:
    the labels are the user's fixed vocabulary ('right after', '+2 days', '+4 months'), and the
    measured number is the evidence recorded beside them in the values JSON.
    """
    rows = []
    for e in entries:
        d = days_after_step(e.get("fluence"), e.get("start_utc"))
        rows.append(dict(look=e.get("look"), when=e.get("when"), fluence=e.get("fluence"),
                         scan=e.get("scan"), start_utc=str(e.get("start_utc")),
                         days_after_step=(round(d, 3) if d is not None else None)))
    print("  day check on %s:" % tag)
    for r in rows:
        print("      %-44s %-12s %s" % (r["look"], r["scan"] or "", 
                                        "n/a (pre-irradiation)" if r["days_after_step"] is None
                                        else "%.2f d" % r["days_after_step"]))
    return rows


# ------------------------------------------------------------------ the shared IV panel
def lowv_mask(v, i, max_v=LOWV_MASK_V):
    v = np.asarray(v, dtype=float)
    i = np.asarray(i, dtype=float)
    m = v <= max_v
    return v[m], i[m]


def lowv_markevery(v, step_V=LOWV_MARKER_STEP_V):
    """~one marker every `step_V` volts on a 0.1 V fine scan (fig21d's own marker density)."""
    v = np.asarray(v, dtype=float)
    if len(v) < 2:
        return 1
    dv = float(np.median(np.diff(np.sort(v))))
    if not np.isfinite(dv) or dv <= 0:
        return 1
    return max(1, int(round(step_V / dv)))


def draw_iv_panel(ax, tel, curves, xlim, ylim=None, yscale="linear", looks=None,
                  legend_scale=SCALE_LIN, loc_f=LOC_LOOK_LIN, loc_c=LOC_CHIP_LIN,
                  point_scale=0.28, thin=False, ylabel=None, extra=None, ncol_f=1):
    """One telescope's IV panel, exactly as the reference figures draw it: every measured point
    drawn as a small open marker on a thin line (point_scale ~0.28 for the dense full-range
    view, ~0.55 for the low-V zoom), the repo tick/grid treatment, the telescope line on the
    header right, and the two-legend layout (look legend + chip legend).
    """
    draw_curves(ax, curves, xlim, ylim, yscale=yscale, ylabel=ylabel or YLABEL_I,
                thin=thin, point_scale=point_scale)
    ts.panel_title(ax, tel, extra=extra)
    if looks:
        two_look_legends(ax, looks, ts.TELESCOPE_CHIPS[tel], scale=legend_scale,
                         loc_f=loc_f, loc_c=loc_c, ncol_f=ncol_f)
    return ax


# ------------------------------------------------------------------ the shared currents panel
CURRENTS_XLABEL = "Elapsed time in run [min]"
CURRENTS_MARKEVERY_MIN = 30      # one marker roughly every 30 minutes (60 s bins)
CURRENTS_POINT_SCALE = 0.275     # fig22's halved markers
CURRENTS_XPAD = 1.30             # room to the right of the last sample for the end labels
CURRENTS_YPAD = 1.45             # empty band above the data for the two legends


def currents_end_labels(ax, entries, min_gap_frac=0.06, leader_thresh_frac=0.02, fontsize=9.0):
    """One bias-only label per run group at its own line end, in the group's own colour
    (fig22_currents_march._end_of_line_labels, lifted unchanged except that the colour now comes
    from the entry instead of being re-derived from the fluence -- the currents figures at one
    fluence colour by BIAS, not by fluence).

    `entries`: [dict(x_end, y_end, color, text), ...].  Collision resolution: sort by end-y,
    push apart from the median outward with a minimum spacing; a label displaced by more than
    `leader_thresh_frac` of the y range gets a thin dotted leader back to its own end point.
    No run numbers ever appear -- a run is described by its condition (bias), per the talk rule.
    """
    if not entries:
        return dict(mode="none", n=0)
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    yrange = yhi - ylo
    min_gap = min_gap_frac * yrange
    n = len(entries)
    rows = [dict(e, x0=e["x_end"], y0=e["y_end"]) for e in entries]
    rows.sort(key=lambda r: r["y0"])
    for r in rows:
        r["y"] = r["y0"]
    mid = n // 2
    for i in range(mid - 1, -1, -1):
        if rows[i + 1]["y"] - rows[i]["y"] < min_gap:
            rows[i]["y"] = rows[i + 1]["y"] - min_gap
    for i in range(mid + 1, n):
        if rows[i]["y"] - rows[i - 1]["y"] < min_gap:
            rows[i]["y"] = rows[i - 1]["y"] + min_gap
    leader_thresh = leader_thresh_frac * yrange
    n_displaced = 0
    for r in rows:
        y = max(ylo, min(r["y"], yhi))
        color = r["color"]
        x = r["x0"] + 0.015 * (xhi - xlo)
        ha, dx = "left", 4
        if x > xhi - 0.10 * (xhi - xlo):
            x, ha, dx = xhi - 0.01 * (xhi - xlo), "right", -4
        if abs(y - r["y0"]) > leader_thresh:
            ax.plot([r["x0"], x], [r["y0"], y], color=color, linewidth=0.7, linestyle=":",
                    alpha=0.7, zorder=1, clip_on=False)
            n_displaced += 1
        ax.annotate(r["text"], xy=(x, y), xytext=(dx, 0), textcoords="offset points",
                    fontsize=fontsize, color=color, va="center", ha=ha, clip_on=False,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0))
    return dict(mode="end_of_line_by_run", n=n, n_displaced=n_displaced)


def currents_legends(ax, swatches, chips, narrow=True):
    """fig22's two stacked legends for a currents panel: the colour legend (one entry per
    fluence step, or per bias on a single-fluence figure) upper LEFT and the chip legend upper
    RIGHT, in the empty band draw_currents_panel leaves above the data.  One legend entry per
    distinct colour -- never a shade ramp.
    """
    s = ts.sizes(1.0)
    # one column always: a currents legend entry carries a mathtext fluence AND a bias
    # ("2x10^15 p/cm2, 540/550 V"), so a two-column box grew wide enough to meet the chip legend
    # coming in from the right (legend-box audit, 2026-09-19).
    ncol_f = 1
    ncol_c = 2 if narrow else 4
    tight = dict(handlelength=1.3, handletextpad=0.4, columnspacing=0.8,
                 borderaxespad=0.3) if narrow else {}
    lf = ax.legend(handles=look_handles(swatches), loc="upper left", ncol=ncol_f,
                   fontsize=s["legend"], **tight)
    ax.add_artist(lf)
    lc = ax.legend(handles=ts.chip_handles(chips), loc="upper right", ncol=ncol_c,
                   fontsize=s["legend"], **tight)
    return lf, lc


def draw_currents_panel(ax, tel, lines, groups=None, swatches=None, markevery=CURRENTS_MARKEVERY_MIN,
                        point_scale=CURRENTS_POINT_SCALE, xpad=CURRENTS_XPAD, ypad=CURRENTS_YPAD,
                        ylabel=None, extra=None, xlabel=None, ylim=None):
    """One telescope's current-vs-time panel, exactly as fig22_currents_march draws it.

    `lines`:  [dict(x, y, chip, color), ...] -- one drawn series per (run, chip); colour is the
              caller's (fluence on a multi-step figure, bias on a single-step one), marker and
              line style come from the chip.
    `groups`: [dict(x_end, y_end, color, text), ...] -- one bias-only end label per run group.
    `swatches`: [(label, color), ...] for the colour legend; None draws no legends (the caller
              adds them, e.g. a panel export that needs a different column count).
    """
    ydata_max = 0.0
    xmax = 0.0
    for ln in lines:
        # a line may carry its own open/filled marker and line style on top of the chip style:
        # the before/after-the-rest figure draws the same chip twice on one axis (open + dashed
        # before, filled + solid after), the convention the V_gl figures use.
        kw = ts.point_style(ln["chip"], 0.0, color=ln["color"], line=True,
                            open_marker=bool(ln.get("open_marker")))
        kw.pop("capsize", None)
        kw.pop("elinewidth", None)
        if ln.get("linestyle"):
            kw["linestyle"] = ln["linestyle"]
        kw["markersize"] = kw["markersize"] * point_scale
        ax.plot(ln["x"], ln["y"], markevery=markevery, **kw)
        if len(ln["y"]):
            ydata_max = max(ydata_max, float(np.max(ln["y"])))
            xmax = max(xmax, float(np.max(ln["x"])))
    ax.set_xlabel(xlabel or CURRENTS_XLABEL)
    ax.set_ylabel(ylabel or YLABEL_I)
    ax.set_ylim(*(ylim if ylim is not None else (0, ydata_max * ypad if ydata_max > 0 else 1.0)))
    ax.set_xlim(0, xmax * xpad if xmax > 0 else 1.0)
    label_info = currents_end_labels(ax, groups or [])
    ts.style_axes(ax)
    ts.panel_title(ax, tel, extra=extra)
    if swatches:
        currents_legends(ax, swatches, ts.TELESCOPE_CHIPS[tel], narrow=True)
    return label_info


# ------------------------------------------------------------------ compound + panel assembly
def compound_1x2(stem, out_dir, draw, tag, footer_text, line1=None, data=None, pad=HEADER_PAD,
                 values_fn=None, inputs=None, script=None, subjects=None):
    """The 1 x 2 compound of the reference figures: one figure of COMPOUND_FIGSIZE with
    COMPOUND_ADJUST, H1 left / F1 right, the experiment text immediately above the axes (no
    header axis, no dead band), the footer through ts.footer(), then the three audits.

    `draw(ax, tel)` draws one panel and returns whatever the caller wants recorded; its return
    values are handed back so the caller can build the values JSON.
    """
    ts.apply_style(1.0)
    fig, (ax_h, ax_f) = _plt.subplots(1, 2, figsize=COMPOUND_FIGSIZE)
    fig.subplots_adjust(**COMPOUND_ADJUST)
    res_h = draw(ax_h, "h1")
    res_f = draw(ax_f, "f1")
    ts.compound_header([ax_h, ax_f], tag=tag, data=data,
                       line1=(ivd.LINE1_PARKED if line1 is None else line1), pad=pad,
                       subjects=subjects)
    # footer_text / inputs may be callables of the two panels' own results: several figures can
    # only write their footer (a fallback note, a quick-scan note) once the panels have been
    # drawn and know which scans they actually used.
    ts.footer(fig, footer_text(res_h, res_f) if callable(footer_text) else footer_text)
    if callable(inputs):
        inputs = inputs(res_h, res_f)
    ts.lower_footer(fig)  # the footer's final band, so the audits see the saved layout
    problems = ts.check_no_clipping(fig, stem)
    legend_problems = check_legend_overlap(fig, tag=stem)
    box_problems = check_legend_boxes(fig, tag=stem)
    ts.save_figure(fig, out_dir, stem, audit=False)
    _plt.close(fig)
    values = None
    if values_fn is not None:
        values = dict(values_fn(res_h, res_f))
        values.update(overlap_problems=problems, legend_overlap_problems=legend_problems,
                      legend_box_problems=box_problems)
        write_values(out_dir, stem, values, script=script, inputs=inputs)
    print("wrote %s  problems=%d  legend_problems=%d  legend_box_problems=%d"
          % (stem, len(problems), len(legend_problems), len(box_problems)))
    return values if values is not None else dict(
        overlap_problems=problems, legend_overlap_problems=legend_problems,
        legend_box_problems=box_problems)


def export_panels_1x2(stem, out_dir, chosen, line1=None, tag=None, scale=1.0, pad=HEADER_PAD,
                      name=None):
    """Single-panel export on the reference canvas (PANEL_FIGSIZE / PANEL_AXES), so any two
    panels from any two figures in this family tile cleanly on a slide.  The compound's
    '(a)/(b)' panel letter is dropped (it only means something next to its sibling); the panel's
    own subject text goes in through `data`.
    """
    ts.apply_style(1.0)
    written = []
    for index, pname, draw, title in chosen:
        fig = _plt.figure(figsize=PANEL_FIGSIZE)
        ax = fig.add_axes(PANEL_AXES)
        draw(ax, None)
        ts.talk_header(ax, name=name, tag=tag,
                       line1=(ivd.LINE1_PARKED if line1 is None else line1),
                       pad=pad, data=title or None, scale=scale)
        ts.check_no_clipping(fig, "%s panel %s" % (stem, pname))
        check_legend_overlap(fig, tag="%s panel %s" % (stem, pname))
        check_legend_boxes(fig, tag="%s panel %s" % (stem, pname))
        paths = ts.save_panel(fig, out_dir, stem, index, pname)
        _plt.close(fig)
        written += paths
        print("  wrote panel %02d %-10s %s" % (index, pname, paths[0]))
    return written


# ------------------------------------------------------------------ A4: legend-vs-legend audit
def check_legend_boxes(fig, tag=""):
    """Every pair of legends on one axes must not overlap each other.

    check_no_clipping() only scans ax.get_legend(), which is whichever legend was registered
    LAST; the first legend of a two-legend panel is kept alive through ax.add_artist() and is
    invisible to that audit, so two boxes meeting was never caught (fig32 carried its own copy
    of this check -- lifted here so every figure in the family runs it).  Reports, never raises:
    the fix is a placement change.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    problems = []
    for ai, ax in enumerate(fig.axes):
        legs = [c for c in ax.get_children() if isinstance(c, Legend)]
        for i in range(len(legs)):
            for j in range(i + 1, len(legs)):
                a = legs[i].get_window_extent(renderer=r)
                b = legs[j].get_window_extent(renderer=r)
                if not (a.x1 < b.x0 or b.x1 < a.x0 or a.y1 < b.y0 or b.y1 < a.y0):
                    dx = min(a.x1, b.x1) - max(a.x0, b.x0)
                    dy = min(a.y1, b.y1) - max(a.y0, b.y0)
                    problems.append("axes[%d] two legend boxes overlap by %.0f x %.0f px"
                                    % (ai, dx, dy))
    print("  legend box check on %s: %s" % (tag, "boxes clear" if not problems
                                            else "%d PROBLEM(S)" % len(problems)))
    for p in problems:
        print("      %s" % p)
    return problems


# ------------------------------------------------------------------ top-band audit (user rule)
def top_band_px(png_path, dpi_threshold=200, sample_every=4):
    """First image row carrying any pixel below `dpi_threshold` in luminance, sampling every
    `sample_every`th column -- the blank band above the figure's topmost ink.  The talk rule
    caps this at 60 px on a 200 dpi compound.
    """
    from PIL import Image
    im = Image.open(png_path).convert("L")
    w, h = im.size
    px = im.load()
    for y in range(h):
        for x in range(0, w, sample_every):
            if px[x, y] < dpi_threshold:
                return y
    return h


# ------------------------------------------------------------------ one loader for every look
def load_looks(tel, looks, mask_V=None, tag="", chip_dose=False):
    """Load one telescope's curves for an ordered list of LOOKS, whatever their source.

    Each look is a dict:
      source="march"      fluence, role ("fine_ra"/"fine_2d"/"quick_last"), when
      source="july"       fluence, key (into ivd.JULY_SCANS), when
      source="july_fine"  fluence, key (into ivd.JULY_FINE_SCANS, raw legacy logs binned to
                          0.1 V by ivd.load_july_fine), when
    plus an optional `when` of None for pre-irradiation (no step to count from).

    Returns (curves, used, swatches): `curves` carries the family's drawing keys (v, i, chip,
    fluence, look, color, bin_width_V, open_marker); `used` one provenance row per look,
    including the scan's own start time so ivp.check_days can measure the day count the label
    claims; `swatches` is look_series()'s [(label, color), ...] for the legend, in draw order.
    One code path for every figure: a look's colour, label and bin width can never be decided
    differently by two scripts.
    """
    swatches = look_series([(lk["fluence"], lk.get("when")) for lk in looks],
                          chip_dose=chip_dose)
    curves, used = [], []
    for lk, (label, color) in zip(looks, swatches):
        f = lk["fluence"]
        src = lk["source"]
        if src == "march":
            path = ivd.MARCH_SCANS[tel][f][lk["role"]]
            data = ivd.load_march(path, tel)
            bw = ivd.bin_width_march(lk["role"])
            start = ivd.march_scan_start_utc(path)
            prov = dict(path=path, scan=os.path.basename(path), role=lk["role"])
        elif src == "july":
            # `scan_by_tel` names the iv_curves.json key per telescope directly, for the one
            # look whose two telescopes do not share a catalogue entry: the 1.5e15 "+4 months"
            # scan, where H1's first-after-rest quick scan carries an IH13 diagnostic-switch
            # noise interval and the scan two days later (still +4 months after the step) is
            # used instead, F1's unchanged.
            key = (lk["scan_by_tel"][tel] if lk.get("scan_by_tel")
                   else ivd.JULY_SCANS[lk["key"]][tel])
            data, meta = ivd.load_july(key, tel)
            bw = ivd.bin_width_july(meta["type"])
            start = meta.get("start_utc")
            prov = dict(scan=key, meta=meta, type=meta["type"])
        elif src == "july_fine":
            data, meta = ivd.load_july_fine(lk["key"], tel)
            bw = ivd.BIN_WIDTH_FINE_V
            start = meta.get("start")
            prov = dict(scan=meta.get("scan"), meta=meta, type="fine")
        else:
            raise ValueError("unknown look source %r" % src)
        for chip, (v, i) in data.items():
            if mask_V is not None:
                v, i = lowv_mask(v, i, mask_V)
            if not len(v):
                continue
            curves.append(dict(v=v, i=i, chip=chip, fluence=f, look=label, color=color,
                               bin_width_V=bw, open_marker=True))
        used.append(dict(fluence=f, when=lk.get("when"), look=label, source=src,
                         start_utc=str(start) if start is not None else None,
                         bin_width_V=bw, **prov))
    return curves, used, swatches


def look_inputs(used):
    """The provenance paths/scan keys of a `used` list, for write_values(inputs=...)."""
    out = []
    for u in used:
        out.append(u.get("path") or u.get("scan"))
    return [x for x in out if x]


# ------------------------------------------------------------------ one builder for every look figure
def look_figure(stem, out_dir, looks, tag, footer, xlim, yscale="linear", ylim=None,
                mask_V=None, marker_step_V=None, point_scale=0.28, legend_scale=None,
                loc_f=None, loc_c=None, ncol_f=1, panel=None, script=None,
                gap_skip_below_V=0.0, panel_subject="", line1=None, data=None, ylabel=None,
                extra_values=None, chip_dose=False):
    """Build one 1 x 2 IV-curve figure (or export its panels) from a LOOKS list.

    Every IV-curve figure in this family is the same object: a list of looks, a bias window, a
    y scale, and the shared panel.  This is that object, so a new figure is a LOOKS list and a
    footer -- not another copy of the reference script's assembly code, which is how the family
    drifted apart in the first place.

    `ylim` may be a callable(curves) for the linear views that autoscale to their own data.
    `mask_V` switches on the low-V treatment (drop points above it, low-V legend placement);
    `marker_step_V` spaces markers by that many volts (the low-V convention, ~10 V).
    """
    if legend_scale is None:
        legend_scale = SCALE_LOWV if mask_V else (SCALE_LIN if yscale == "linear" else SCALE_LOG)
    if loc_f is None:
        loc_f = LOC_LOOK_LOWV if mask_V else (LOC_LOOK_LIN if yscale == "linear" else LOC_LOOK_LOG)
    if loc_c is None:
        loc_c = LOC_CHIP_LOWV if mask_V else (LOC_CHIP_LIN if yscale == "linear" else LOC_CHIP_LOG)

    def panel_data(tel):
        curves, used, swatches = load_looks(tel, looks, mask_V=mask_V, tag=stem,
                                            chip_dose=chip_dose)
        assert_no_gaps(curves, tag="%s %s" % (stem, tel), skip_below_V=gap_skip_below_V)
        days = check_days(used, tag="%s %s" % (stem, tel))
        for u, d in zip(used, days):
            u["days_after_step"] = d["days_after_step"]
        return curves, used, swatches

    def draw(ax, tel):
        curves, used, swatches = panel_data(tel)
        if marker_step_V:
            for c in curves:
                c["markevery"] = lowv_markevery(c["v"], marker_step_V)
        yl = ylim(curves) if callable(ylim) else ylim
        draw_iv_panel(ax, tel, curves, xlim, yl, yscale=yscale, looks=swatches,
                      legend_scale=legend_scale, loc_f=loc_f, loc_c=loc_c,
                      point_scale=point_scale, thin=False, ylabel=ylabel)
        return curves, used, yl

    def values_of(curves):
        return [dict(chip=c["chip"], fluence=c["fluence"], look=c["look"],
                     i_at_150V_uA=ivd.value_at(c["v"], c["i"], 150.0),
                     v_max_V=float(c["v"].max()) if len(c["v"]) else None,
                     i_max_uA=float(c["i"].max()) if len(c["i"]) else None)
                for c in curves]

    panels = {
        "h1": (lambda ax, ctx: draw(ax, "h1"), panel_subject),
        "f1": (lambda ax, ctx: draw(ax, "f1"), panel_subject),
    }
    chosen = ts.resolve_panels(panel, panels)
    if chosen:
        return export_panels_1x2(stem, out_dir, chosen, line1=line1)

    def values_fn(rh, rf):
        out = dict(h1=dict(scans=rh[1], curves=values_of(rh[0]),
                           ylim_uA=list(rh[2]) if rh[2] else None),
                   f1=dict(scans=rf[1], curves=values_of(rf[0]),
                           ylim_uA=list(rf[2]) if rf[2] else None),
                   xlim_V=list(xlim), yscale=yscale,
                   looks=[dict(fluence=lk["fluence"], when=lk.get("when"), source=lk["source"])
                          for lk in looks])
        if extra_values:
            out.update(extra_values)
        return out

    return compound_1x2(stem, out_dir, draw, tag=tag, footer_text=footer, line1=line1, data=data,
                        values_fn=values_fn, script=script,
                        inputs=lambda rh, rf: look_inputs(rh[1]) + look_inputs(rf[1]))


# ------------------------------------------------------------------ before/after ratio figure
def ratio_figure(stem, out_dir, before, after, xlim, tag, footer, panel=None, script=None,
                 ylabel=None, panel_subject="after / before ratio", line1=None):
    """The family's before/after ratio figure: |I(after)| / |I(before)| per chip on a common
    1 V grid over each chip's own scan overlap, H1 left / F1 right, on the shared geometry.

    `before` / `after` are LOOKS dicts (see load_looks).  The ratio line takes the AFTER look's
    own colour, so it matches the curve that look draws in the IV figures above it; `footer`
    may be a callable(skipped) when the skipped-chip note has to name chips.
    """
    swatches = look_series([(before["fluence"], before.get("when")),
                            (after["fluence"], after.get("when"))])
    after_color = swatches[1][1]

    def pairs_for(tel):
        cb, ub, _ = load_looks(tel, [before])
        ca, ua, _ = load_looks(tel, [after])
        by_b = {c["chip"]: c for c in cb}
        by_a = {c["chip"]: c for c in ca}
        pairs = []
        for chip in ts.TELESCOPE_CHIPS[tel]:
            b, a = by_b.get(chip), by_a.get(chip)
            if b is None or a is None:
                continue
            pairs.append((chip, b["v"], b["i"], a["v"], a["i"], after_color))
        return pairs, ub + ua

    def draw(ax, tel):
        pairs, used = pairs_for(tel)
        per_chip, skipped = draw_ratio_panel(ax, tel, pairs, xlim,
                                             ylabel=ylabel or "I(after) / I(before)")
        s = ts.sizes(SCALE_LOWV)
        ax.legend(handles=ts.chip_handles(ts.TELESCOPE_CHIPS[tel], SCALE_LOWV),
                  loc="upper right", fontsize=s["legend"])
        return per_chip, skipped, used

    panels = {"h1": (lambda ax, ctx: draw(ax, "h1"), panel_subject),
              "f1": (lambda ax, ctx: draw(ax, "f1"), panel_subject)}
    chosen = ts.resolve_panels(panel, panels)
    if chosen:
        return export_panels_1x2(stem, out_dir, chosen, line1=line1)

    return compound_1x2(
        stem, out_dir, draw, tag=tag,
        footer_text=(lambda rh, rf: footer(["h1 %s" % c for c in rh[1]] +
                                           ["f1 %s" % c for c in rf[1]])) if callable(footer)
                    else footer,
        line1=line1, script=script,
        values_fn=lambda rh, rf: dict(
            h1=dict(ratio=rh[0], skipped=rh[1], scans=rh[2]),
            f1=dict(ratio=rf[0], skipped=rf[1], scans=rf[2]),
            xlim_V=list(xlim),
            looks=dict(before=dict(fluence=before["fluence"], when=before.get("when")),
                       after=dict(fluence=after["fluence"], when=after.get("when")))),
        inputs=lambda rh, rf: look_inputs(rh[2]) + look_inputs(rf[2]))
