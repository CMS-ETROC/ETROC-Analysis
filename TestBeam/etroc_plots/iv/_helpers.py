"""Internal helpers: limits, legends, panel tags, smoothing."""

import os

import numpy as np
import pandas as pd
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import mplhep as hep

from ._ivstyle import *

def _lims(spec, axis="x", as_date=False):
    """Normalise a limit spec to a kwargs dict for set_xlim / set_ylim."""
    lo_key, hi_key = ("left", "right") if axis == "x" else ("bottom", "top")
    if spec is None:
        return {}
    if isinstance(spec, dict):
        out = {lo_key: spec.get(lo_key, spec.get("min")),
               hi_key: spec.get(hi_key, spec.get("max"))}
    elif isinstance(spec, (tuple, list)) and len(spec) == 2:
        out = {lo_key: spec[0], hi_key: spec[1]}
    else:
        raise ValueError(f"Cannot interpret {axis}lims={spec!r}; "
                         f"use (lo, hi) or {{'{lo_key}': .., '{hi_key}': ..}}")
    if as_date:
        out = {k: (pd.to_datetime(v) if isinstance(v, str) else v)
               for k, v in out.items()}
    return {k: v for k, v in out.items() if v is not None}


def _apply_lims(ax, xlims=None, ylims=None, x_is_date=False):
    xl = _lims(xlims, "x", as_date=x_is_date)
    yl = _lims(ylims, "y")
    if xl:
        ax.set_xlim(**xl)
    if yl:
        ax.set_ylim(**yl)


_GRID_MODE = "both"          # module-level default: gridlines on both axes


def set_grid(mode):
    """
    Choose which gridlines every subsequent plot draws.

    mode : "both" (default), "x" (vertical lines only), "y" (horizontal lines
           only), or None / "none" / False for no grid.

    One global knob, deliberately: call it once in the notebook's config cell
    and every figure from every plot function follows, including the twin-axes
    timelines. Call it again before a specific plot to change just that one,
    and restore afterwards.
    """
    global _GRID_MODE
    allowed = {"both", "x", "y", "none", None, False}
    if mode not in allowed:
        raise ValueError(f'grid mode must be one of "both", "x", "y", None; got {mode!r}')
    _GRID_MODE = "none" if mode in (None, False) else mode


# expose as etroc_plots.iv.set_grid without touching __init__: the package is
# already registered in sys.modules while its submodules import
import sys as _sys
if _sys.modules.get(__package__) is not None:
    setattr(_sys.modules[__package__], "set_grid", set_grid)


def _apply_grid(ax):
    """Draw gridlines on ax according to the global mode (dashed, ALPHA_GRID)."""
    ax.grid(False, which="both")
    if _GRID_MODE in ("both", "x"):
        ax.grid(True, axis="x", linestyle="--", alpha=ALPHA_GRID, which="both")
    if _GRID_MODE in ("both", "y"):
        ax.grid(True, axis="y", linestyle="--", alpha=ALPHA_GRID, which="both")


def _style_axes(ax, xlabel=None, ylabel=None, title=None, cms=False,
                grid=True, minor_labels=False):
    """Common formatting: CMS header, right-aligned title, dashed grid, minor ticks."""
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if cms:
        hep.cms.text(EXPERIMENT, loc=0, ax=ax, fontsize=FS_CMS)
    if title:
        ax.set_title(title, loc="right", size=FS_TITLE)
    ax.minorticks_on()
    if minor_labels:
        ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
        ax.tick_params(axis="both", which="minor", labelsize=FS_MINOR)
    if grid:
        _apply_grid(ax)


def _ch_label(channel_names, ch):
    channel_names = channel_names or {}
    return channel_names.get(ch, channel_names.get(str(ch), f"Ch {ch}"))


def _two_tier_legend(ax, scans, seen_channels, channel_names,
                     anchor_channels=None, anchor_scans=None,
                     title_channels="Devices", title_scans="Conditions",
                     fontsize=None, strip_threshold=4):
    """
    Tier 1 = channels (marker/linestyle), tier 2 = datasets (colour).

    With no anchors given, the pair is auto-placed on the corner covering the
    fewest drawn points and stacked there as a unit. Passing anchor_channels
    pins the pair; anchor_scans additionally splits the tiers apart (for callers
    that want two independent boxes).
    """
    fs = fontsize or FS_LEGEND
    ch_handles = [mlines.Line2D([], [], color="black",
                                marker=MARKERS[c % len(MARKERS)],
                                ls=LINESTYLES[c % len(LINESTYLES)],
                                markersize=MARKERSIZE,
                                label=_ch_label(channel_names, c))
                  for c in sorted(seen_channels)]
    sc_handles = [mlines.Line2D([], [], color=_scan_color(i), ls=_scan_dash(i),
                                lw=LW_LEGEND_KEY,
                                label=s.get("label", f"Dataset {i+1}"))
                  for i, s in enumerate(scans)]

    if not sc_handles and ch_handles:
        _place_legend(ax, ch_handles, title=title_channels,
                      anchor=anchor_channels, fontsize=fs)
        return
    if not ch_handles:
        if sc_handles:
            _place_legend(ax, sc_handles, title=title_scans,
                          anchor=anchor_scans or anchor_channels, fontsize=fs)
        return

    if anchor_channels is not None and anchor_scans is not None:
        # fully pinned, independent boxes
        def side(a): return (f"{'upper' if a[1]>0.5 else 'lower'} "
                             f"{'left' if a[0]<0.5 else 'right'}",
                             "left" if a[0] < 0.5 else "right")
        loc, al = side(anchor_channels)
        lg1 = ax.legend(handles=ch_handles, title=title_channels, fontsize=fs,
                        title_fontsize=fs, loc=loc, bbox_to_anchor=anchor_channels,
                        frameon=True, alignment=al)
        ax.add_artist(lg1)
        loc, al = side(anchor_scans)
        ax.legend(handles=sc_handles, title=title_scans, fontsize=fs,
                  title_fontsize=fs, loc=loc, bbox_to_anchor=anchor_scans,
                  frameon=True, alignment=al)
        return

    if len(sc_handles) > strip_threshold:
        # many conditions: their long labels fill any corner, so the
        # Conditions tier moves to a strip under the figure (like a caption)
        # and the compact Devices tier stays inside the axes, unless the
        # axes are so saturated (hours of ~1 Hz monitoring) that even the
        # best Devices spot would sit on dozens of points, in which case it
        # joins the strip too
        lg = _place_legend(ax, ch_handles, title=title_channels,
                           anchor=anchor_channels, fontsize=fs,
                           give_up_cover=30)
        fig = ax.get_figure()
        if lg is None:
            fig.legend(handles=ch_handles, title=title_channels, fontsize=fs,
                       title_fontsize=fs, loc="upper right",
                       bbox_to_anchor=(0.99, 0.0), ncols=len(ch_handles) if
                       len(ch_handles) <= 4 else 2, frameon=True)
            fig.legend(handles=sc_handles, title=title_scans, fontsize=fs,
                       title_fontsize=fs, loc="upper left",
                       bbox_to_anchor=(0.01, 0.0), ncols=2, frameon=True)
        else:
            fig.legend(handles=sc_handles, title=title_scans, fontsize=fs,
                       title_fontsize=fs, loc="upper center",
                       bbox_to_anchor=(0.5, 0.0), ncols=2, frameon=True)
        return

    _stacked_legends(ax, ch_handles, title_channels, sc_handles, title_scans,
                     anchor=anchor_channels, fontsize=fs)


def _mfactor(unit):
    return {"nA": 1e9, "uA": 1e6, "mA": 1e3, "A": 1.0}[unit]


def _panel_tag(ax, text, loc="upper left", size=FS_GRID_TAG):
    """
    Device name *inside* the panel. Keeps the axes-title strip free for the CMS
    header and the run title, so the three never collide.
    """
    x, ha = (0.03, "left") if "left" in loc else (0.97, "right")
    y, va = (0.96, "top") if "upper" in loc else (0.04, "bottom")
    return ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=size,
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85),
                   zorder=5)


def _grid_fonts(ax):
    """Shrink CMS-sized text to fit a grid cell."""
    ax.xaxis.label.set_size(FS_GRID_LABEL)
    ax.yaxis.label.set_size(FS_GRID_LABEL)
    ax.tick_params(axis="both", which="major", labelsize=FS_GRID_TICK)


def _smooth_volts(v, y, width_v):
    """
    Boxcar whose width is given in VOLTS, not samples.

    The window is rebuilt at every point from the local voltage spacing, because
    the fine scans are not uniformly stepped: 20260720_190823 runs 0.1 V from 10
    to 60 V and then 0.2 / 1 / 5 / 10 V out to 510 V. A single sample count
    derived from the median step would average over 1 V down low and 100 V up
    high.

    Averaging is restricted to a contiguous run of samples so an up-and-down
    sweep never mixes its two legs together.
    """
    y = np.asarray(y, dtype=float)
    if width_v is None or width_v <= 0 or len(v) < 3:
        return y
    v = np.asarray(v, dtype=float)
    half = width_v / 2.0
    n = len(v)
    out = np.empty(n, dtype=float)
    for j in range(n):
        lo = j
        while lo > 0 and abs(v[lo - 1] - v[j]) <= half:
            lo -= 1
        hi = j
        while hi < n - 1 and abs(v[hi + 1] - v[j]) <= half:
            hi += 1
        seg = y[lo:hi + 1]
        out[j] = np.nanmean(seg) if np.isfinite(seg).any() else np.nan
    return out


def _parse_window(value, ts):
    """
    Parse a start/end window compatibly with the data's own timestamp style.

    Time-only legacy logs land on pandas' dummy date (1900-01-01ff), so a
    window given as "07:55:00" must land there too; a bare
    pd.to_datetime("07:55:00") would resolve onto *today* and silently select
    nothing. Detection: if the window parses with no date and the data lives
    on the dummy date, re-anchor the window onto the data's own day(s),
    choosing the day that actually contains that time of day when the log
    crosses midnight.
    """
    t = pd.to_datetime(value)
    if not len(ts):
        return t
    data_day0 = ts.dropna().dt.normalize().min()
    if data_day0 is pd.NaT or data_day0.year != 1900:
        return t                     # full-date data: use the window as given
    # data is time-only; interpret the window as time-of-day
    tod = t - t.normalize()
    for day in pd.unique(ts.dropna().dt.normalize()):
        cand = day + tod
        if ts.min() <= cand <= ts.max():
            return cand
    return data_day0 + tod


def _elapsed_origin(tidy, scan, t0_mode="window"):
    """
    Zero point for an elapsed-time axis.

    "window" -> first reading inside this dataset's start/end slice, so several
                slices carved out of the same long log overlay from zero
    "file"   -> first reading in the file, preserving the real offsets
    """
    t = tidy["timestamp"]
    if t0_mode == "window":
        if scan.get("start") is not None:
            t = t[t >= _parse_window(scan["start"], tidy["timestamp"])]
        if scan.get("end") is not None:
            t = t[t <= _parse_window(scan["end"], tidy["timestamp"])]
    return t.min() if len(t) else tidy["timestamp"].min()


def _markevery(n, spec="auto"):
    """Thin markers on dense traces so a long log doesn't become a solid block."""
    if spec is None:
        return None
    if spec == "auto":
        return max(1, n // MARKEVERY_N)
    return spec


# scan-config keys that route to loading / cleaning rather than plotting
SCAN_LOAD_KEYS  = ("fmt", "format")
SCAN_CLEAN_KEYS = ("start", "end", "i_max")
SCAN_CURVE_KEYS = ("bins",)


def _as_scan_list(scans):
    """
    Accept a path, a config dict, or a list of either. Recognised keys:

        file    (required)   path to the scan
        label                legend label
        format / fmt         "new" | "old" | "auto"   (default: by extension)
        start / end          time window; essential for legacy logs that hold
                             many sweeps in one file
        i_max                legacy compliance-current cut [uA]
        bins                 legacy voltage binning (e.g. legacy.QUICK_BINS)
    """
    if isinstance(scans, (str, os.PathLike, dict)):
        scans = [scans]
    out = []
    for i, s in enumerate(scans):
        s = {"file": s} if not isinstance(s, dict) else dict(s)
        base = os.path.basename(str(s["file"]))
        s.setdefault("label", base.rsplit(".", 1)[0])
        out.append(s)
    return out


def _scan_load_kw(s):
    fmt = s.get("fmt", s.get("format", "auto"))
    fmt = {"new": "new", "sqlite": "new", "old": "old", "legacy": "old",
           "csv": "old", "auto": "auto"}[fmt]
    return {"fmt": fmt}


def _scan_clean_kw(s):
    return {k: v for k, v in s.items() if k in SCAN_CLEAN_KEYS}


def _scan_color(idx):
    """Dataset colour; past the palette end, wrap but switch to dashed so two
    datasets never share an identical (colour, linestyle) pair."""
    return COLORS[idx % len(COLORS)]


def _scan_dash(idx):
    return "--" if idx >= len(COLORS) else "-"


def _twin_axes(ax):
    """Axes in the same figure occupying the same position (twinx/twiny)."""
    p = ax.get_position()
    out = []
    for a in ax.get_figure().axes:
        if a is ax:
            continue
        q = a.get_position()
        if abs(q.x0 - p.x0) < 1e-6 and abs(q.y0 - p.y0) < 1e-6 and            abs(q.x1 - p.x1) < 1e-6 and abs(q.y1 - p.y1) < 1e-6:
            out.append(a)
    return out


def _artist_points_px(ax, renderer, include_twins=True):
    """
    Pixel coordinates of every drawn data point on ax and, by default, on its
    twinned axes: a legend hosted on ax_v also has to dodge the current
    trace living on ax_i.

    Points come from Line2D.get_xydata(), which is already unit-converted to
    float data coordinates. Converting get_xdata() with np.asarray(dtype=
    float) is NOT safe: datetime64 silently casts to nanoseconds since epoch
    rather than raising, every transformed point lands ~1e13 pixels off
    screen, and the cover count reads zero for any absolute-time axis, so a
    legend could sit on the voltage trace of a timeline plot while the audit
    passed.
    """
    axes = [ax] + (_twin_axes(ax) if include_twins else [])
    pts = []
    for a in axes:
        for l in a.lines:
            try:
                xy = np.asarray(l.get_xydata(), dtype=float)
            except (TypeError, ValueError):
                continue
            if xy.ndim != 2 or len(xy) < 2:
                continue
            ok = np.isfinite(xy).all(axis=1)
            if not ok.any():
                continue
            try:
                pts.append(a.transData.transform(xy[ok]))
            except Exception:
                pass
    return np.vstack(pts) if pts else np.empty((0, 2))


def _place_legend(ax, handles, title=None, anchor=None, fontsize=None,
                  candidates=((0.02, 0.98), (0.98, 0.98), (0.02, 0.02),
                              (0.98, 0.02), (0.40, 0.98), (0.98, 0.55)),
                  pad_px=4, give_up_cover=None, ncols=1):
    """
    Draw a legend at the candidate corner covering the fewest data points.

    A fixed anchor cannot suit every dataset: a log IV plot fills the upper
    left, a cooling-down hold fills the upper right, and a nine-condition
    overlay fills most corners. So unless the caller pins `anchor`, each
    candidate is measured against the axes' actual drawn points (markers
    included) and the least-covered corner wins; ties break toward the first
    candidate. loc/alignment follow the chosen corner automatically.
    """
    fs = fontsize or FS_LEGEND

    def mkleg(anc):
        loc = f"{'upper' if anc[1] > 0.5 else 'lower'} {'left' if anc[0] < 0.5 else 'right'}"
        lg = ax.legend(handles=handles, title=title, fontsize=fs, title_fontsize=fs,
                       loc=loc, bbox_to_anchor=anc, frameon=True, ncols=ncols,
                       alignment="left" if anc[0] < 0.5 else "right")
        return lg

    if anchor is not None:
        return mkleg(anchor)

    fig = ax.get_figure()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pts = _artist_points_px(ax, r)
    axbb = ax.get_window_extent(r)

    # Measure the legend's footprint once (any anchor), then slide that
    # footprint over a coarse grid of anchor positions and count covered
    # points at each. Corner candidates alone fail on long time-series where
    # every corner has a trace passing through but mid-height regions are
    # empty; a 13 x 9 grid finds those. Grid positions are scored with a
    # slight preference for the conventional corners so sparse plots keep
    # their usual look; the probe corners are tried first and any zero-cover
    # corner wins outright.
    probe = mkleg(candidates[0])
    fig.canvas.draw()
    pw = probe.get_window_extent(r).width
    ph = probe.get_window_extent(r).height
    probe.remove()

    def cover_at(anc):
        # anchor -> pixel box the legend would occupy
        x0 = axbb.x0 + anc[0] * axbb.width - (0 if anc[0] < 0.5 else pw)
        y0 = axbb.y0 + anc[1] * axbb.height - (0 if anc[1] < 0.5 else ph)
        x1, y1 = x0 + pw, y0 + ph
        if x0 < axbb.x0 - pad_px or x1 > axbb.x1 + pad_px or            y0 < axbb.y0 - pad_px or y1 > axbb.y1 + pad_px:
            return 10_000
        if not len(pts):
            return 0
        return int(((pts[:, 0] > x0 - pad_px) & (pts[:, 0] < x1 + pad_px) &
                    (pts[:, 1] > y0 - pad_px) & (pts[:, 1] < y1 + pad_px)).sum())

    best, best_cost = None, None
    for anc in candidates:                       # conventional corners first
        c = cover_at(anc)
        if c == 0:
            return mkleg(anc)
        if best_cost is None or c * 10 < best_cost:
            best, best_cost = anc, c * 10
    for gx in np.linspace(0.02, 0.98, 13):       # then the full grid
        for gy in np.linspace(0.02, 0.98, 9):
            # corner preference must never beat actual emptiness: a zero-cover
            # grid spot (cost 2) always wins over a 2-point corner (cost 20)
            c = cover_at((gx, gy)) * 10 + 2
            if c < best_cost:
                best, best_cost = (gx, gy), c
    best_cost //= 10
    if give_up_cover is not None and best_cost > give_up_cover:
        return None                              # axes saturated: caller strips
    return mkleg(best)


def _stacked_legends(ax, tier1_handles, tier1_title, tier2_handles, tier2_title,
                     anchor=None, fontsize=None, gap=0.02):
    """
    Two-tier legend (Devices over Conditions) as one unit.

    The probe legend holds BOTH tiers' entries so its footprint equals the
    final stack; the corner covering the fewest drawn points wins. The second
    tier is then anchored strictly below the first, and if that would leave
    the axes (tall stacks near the bottom edge), the whole pair flips to
    growing upward instead, so the stack never pokes outside.
    """
    fs = fontsize or FS_LEGEND
    fig = ax.get_figure()

    probe = _place_legend(ax, tier1_handles + tier2_handles, title=tier1_title,
                          anchor=anchor, fontsize=fs)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pbb = probe.get_window_extent(r)
    axbb = ax.get_window_extent(r)
    left = (pbb.x0 - axbb.x0) < (axbb.x1 - pbb.x1)
    top_anchor_disp = (pbb.x0 if left else pbb.x1, pbb.y1)
    anc = tuple(ax.transAxes.inverted().transform(top_anchor_disp))
    probe.remove()

    loc_x = "left" if left else "right"
    lg1 = ax.legend(handles=tier1_handles, title=tier1_title, fontsize=fs,
                    title_fontsize=fs, loc=f"upper {loc_x}", bbox_to_anchor=anc,
                    frameon=True, alignment=loc_x)
    ax.add_artist(lg1)
    fig.canvas.draw()
    b1 = lg1.get_window_extent(r).transformed(ax.transAxes.inverted())

    lg2 = ax.legend(handles=tier2_handles, title=tier2_title, fontsize=fs,
                    title_fontsize=fs, loc=f"upper {loc_x}",
                    bbox_to_anchor=(anc[0], b1.y0 - gap), frameon=True,
                    alignment=loc_x)
    fig.canvas.draw()
    b2 = lg2.get_window_extent(r)
    if b2.y0 < axbb.y0 - 2:
        # stack leaves the bottom: rebuild growing upward from the floor, and
        # if the pair is simply taller than the axes, shrink the fonts one
        # step so it fits rather than poking out either end
        lg2.remove()
        lg1.remove()

        def build_up(fsz):
            lo = ax.legend(handles=tier2_handles, title=tier2_title, fontsize=fsz,
                           title_fontsize=fsz, loc=f"lower {loc_x}",
                           bbox_to_anchor=(anc[0], 0.02), frameon=True,
                           alignment=loc_x)
            ax.add_artist(lo)
            fig.canvas.draw()
            h2 = lo.get_window_extent(r).transformed(ax.transAxes.inverted()).height
            hi = ax.legend(handles=tier1_handles, title=tier1_title, fontsize=fsz,
                           title_fontsize=fsz, loc=f"lower {loc_x}",
                           bbox_to_anchor=(anc[0], 0.02 + h2 + gap), frameon=True,
                           alignment=loc_x)
            fig.canvas.draw()
            return lo, hi

        lo, hi = build_up(fs)
        if hi.get_window_extent(r).y1 > axbb.y1 + 2:      # taller than the axes
            hi.remove()
            lo.remove()
            build_up(max(FS_GRID_LEG, fs - 5))
