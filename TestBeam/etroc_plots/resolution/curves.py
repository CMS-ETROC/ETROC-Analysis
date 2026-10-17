"""Time resolution against bias and against time (the resolution notebook).

Points: one per (run, board), from the published board tables (BOARD_TABLE for single runs,
BOARD_TABLE_MERGED for merges, read by resolution.tables) and runs_summary.csv (each board's chip,
bias, bookkeeping fluence, threshold offset and RFSel; the March RFSel, which runs_summary leaves
empty, is MARCH_RFSEL). A point is the board's value in its anointed combination, its bar the
pixel-to-pixel spread there; at a step of BAND_STEPS it carries the band too (tables.band) and
is drawn as one.

Populations, chosen in the notebook with these helpers:

  standard      threshold offset 20 and RFSel 2 (or none recorded): the standard operating point
  merge_rule    a merge replaces its runs only when every run of the merge is in the figure's
                display-list class and the good-run list marks the merge good
  good_list     the good-run list of one campaign and telescope, merges included ("3+4")

Clocks, from the run starts of RUN_TIMES_CSV (overview.timeline.read_run_times):

  hours_after_stop   run start minus RAD_STOP_UTC[step], the earliest stop the records allow
                     (frontend.blnw.time_after_stop); stop_band is the allowed-stop range
                     (0 to latest minus earliest, from RAD_STOP_BOUNDS_UTC), drawn at the origin
                     when it spans STOP_BAND_MIN_H or more
  hours_since_first  run start minus the start of the first run at the step (first_run_at)

Colour (color_rule): several looks (campaign, fluence step) on the axes -> colour = look, the
fluence ladder (FLUENCE_COLOR) with a later look at the same step in the later-look hue
(iv_plot.timing_color); one look -> colour = board (etroc_style.BOARD_COLOR), the step named in
the header. Marker and line style = chip on every figure. No jitter: points at one x share it;
the line of a series runs through the mean at each x.

Primitives, words as arguments: bias_scan and time_series (return the legend handles), brace.
Figures (each returns the figure and the numbers drawn):

  res_vs_bias_march_preirrad   the March pre-irradiation bias scan, colour = board
  res_vs_bias_march_steps      the March bias scans after each step, colour = fluence
  res_1p5e15_march_vs_july     the March 1.5e15 scan and the July 1.5e15 points
  res_july_1p5e15_stability    July 1.5e15 against the hours since the first run there
  res_2e15_cooling_down        July 2e15 against the hours after the stop, in-run currents
"""
import re

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from ..campaigns import active as campaign, check_required
from .. import style
from ..etroc_style import BOARD_COLOR
from ..frontend import blnw
from ..iv.iv_plot import timing_color
from . import tables
from .settings import DIVIDER_GID, LABEL_GID

check_required("resolution")

SCALE = 0.8                 # font and marker scale of the resolution figures
STANDARD_RFSEL = 2          # the standard operating point (standard())
STANDARD_OFFSET = 20
STOP_BAND_MIN_H = 0.25      # an allowed-stop range at least this long (h) is drawn as a band
CAMP_ORDER = ("march", "july")     # the order of the looks at one fluence step
_MERGE_RUN = re.compile(r"_grp_run(\d+(?:_\d+)+)$")


def _close(a, b):
    return abs(float(a) - float(b)) <= 1e-6 * max(abs(float(b)), 1.0)


def run_label(run):
    """A run of a figure as text: 12 -> "12", (3, 4) -> "3+4" (the good-run list's spelling)."""
    if isinstance(run, (tuple, list)):
        return "+".join(str(int(r)) for r in run)
    return str(int(run))


# --------------------------------------------------------------------------------- points
def points(bt, bm, rs, telescope, runs, camp, variant="top", floor="300"):
    """One dict per (run, board) of `runs` (ints for single runs, tuples of ints for merges) of
    telescope `telescope` ("h1") in campaign `camp` ("march" / "july"):

      tel, camp, run (the table's run: int, or the merge name "h1_grp_run3_4"), runs (the run
      numbers), label ("3+4"), board_idx, chip, value_ps (the anointed combination's value),
      bar_ps (its pixel-to-pixel spread), n_pixels, hv, fluence (bookkeeping), offset, rfsel,
      rfsel_source ("runs_summary" or "MARCH_RFSEL"), merged, band (tables.band at a step of
      BAND_STEPS, else None), y_ps and yerr_ps (what is drawn: the band centre and the anointed
      spread where there is a band, else the value and its spread).

    A merge takes its boards' settings from runs_summary through tables.common_settings, which
    raises when its runs differ in fluence, RFSel, offset, chip or bias. A run without anointed
    rows in the table, or whose chips differ from runs_summary, raises ValueError."""
    tc, tk = campaign.TABLE_CAMPAIGN[camp], campaign.TABLE_TEL[telescope]
    out = []
    for run in runs:
        merged = isinstance(run, (tuple, list))
        members = tuple(int(r) for r in run) if merged else (int(run),)
        if merged:
            tables.common_settings(rs, tc, tk, members)
            key, table = tables.merge_name(telescope, members), bm
        else:
            key, table = members[0], bt
        boards = {b["board_idx"]: b for b in tables.run_settings(rs, tc, tk, members[0])}
        rows = tables.board_rows(table, telescope, key, variant, floor, campaign=tc)
        if not len(rows):
            raise ValueError("%s %s run %s: no anointed rows (variant %s, floor %s) in the %s "
                             "board table" % (telescope, camp, run_label(members), variant,
                                              floor, "merged" if merged else "single-run"))
        for _, r in rows.iterrows():
            i = int(r["board_idx"])
            b = boards[i]
            chip = style.chip_short(b["chip"])
            if style.chip_short(r["board_chip"]) != chip:
                raise ValueError("%s run %s board %d: %s in the board table, %s in runs_summary"
                                 % (telescope, run_label(members), i, r["board_chip"], chip))
            rfsel, source = b["rfsel"], "runs_summary"
            if rfsel is None and camp == "march":
                rfsel, source = campaign.MARCH_RFSEL, "MARCH_RFSEL"
            band = None
            if any(_close(b["fluence"], s) for s in campaign.BAND_STEPS):
                band = tables.band(table, telescope, key, i, variant, floor, campaign=tc)
            p = dict(tel=telescope, camp=camp, run=key, runs=list(members),
                     label=run_label(members), board_idx=i, chip=chip,
                     value_ps=float(r["value_ps"]), bar_ps=float(r["pixel_spread_ps"]),
                     n_pixels=int(r["n_pixels"]), hv=b["hv"], fluence=b["fluence"],
                     offset=b["offset"], rfsel=rfsel, rfsel_source=source, merged=merged,
                     band=band)
            p["y_ps"] = band["marker_ps"] if band else p["value_ps"]
            p["yerr_ps"] = band["bar_spread_ps"] if band else p["bar_ps"]
            out.append(p)
    return out


def standard(pts, rfsel=STANDARD_RFSEL, offset=STANDARD_OFFSET):
    """The points at the standard operating point: threshold offset `offset` and RFSel `rfsel`,
    or no RFSel recorded."""
    return [p for p in pts if p["offset"] == offset and (p["rfsel"] is None
                                                          or p["rfsel"] == rfsel)]


def table_runs(bt, telescope, camp, variant="top", floor="300"):
    """The single runs of one telescope and campaign in a board table, sorted."""
    sel = ((bt["campaign"] == campaign.TABLE_CAMPAIGN[camp]) & (bt["telescope"] == telescope)
           & (bt["variant"] == variant) & (bt["floor"] == str(floor)))
    return sorted({int(r) for r in bt.loc[sel, "run"]})


def table_merges(bm, telescope, camp):
    """The merges of one telescope and campaign in the merged board table, as sorted tuples of
    run numbers ("h1_grp_run3_4" -> (3, 4))."""
    sel = (bm["campaign"] == campaign.TABLE_CAMPAIGN[camp]) & (bm["telescope"] == telescope)
    out = set()
    for name in bm.loc[sel, "run"].astype(str).unique():
        m = _MERGE_RUN.search(name)
        if m is None:
            raise ValueError("merged-table run %r is not a merge name" % name)
        out.add(tuple(sorted(int(r) for r in m.group(1).split("_"))))
    return sorted(out)


def runs_at_step(rs, telescope, camp, step):
    """The runs of runs_summary whose boards all sit at bookkeeping fluence `step`, sorted."""
    tc, tk = campaign.TABLE_CAMPAIGN[camp], campaign.TABLE_TEL[telescope]
    runs = rs.loc[(rs["campaign"] == tc) & (rs["telescope"] == tk), "run"]
    return sorted(int(r) for r in runs
                  if all(_close(b["fluence"], step) for b in tables.run_settings(rs, tc, tk, r)))


# ------------------------------------------------------------------------ good runs, merges
def _good_flag(value, where):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if str(value).strip() in ("True", "False"):
        return str(value).strip() == "True"
    raise ValueError("%s: good = %r is not True or False" % (where, value))


def good_list(camp, telescope, path=None):
    """{label: (good, reason)} of the good-run list (GOOD_RUNS_CSV) for one campaign ("march" /
    "july") and telescope, single runs ("12") and merges ("3+4") alike."""
    path = path or campaign.GOOD_RUNS_CSV
    df = pd.read_csv(path, dtype={"run": str})
    name = {"march": campaign.GOOD_RUNS_CAMPAIGN_MARCH, "july": campaign.GOOD_RUNS_CAMPAIGN}[camp]
    df = df[(df["campaign"] == name) & (df["telescope"] == campaign.GOOD_RUNS_TEL[telescope])]
    out = {}
    for _, row in df.iterrows():
        label = "+".join(str(int(r)) for r in str(row["run"]).split("+"))
        if label in out:
            raise ValueError("%s: %s %s run %s is listed twice" % (path, name, telescope, label))
        out[label] = (_good_flag(row["good"], "%s, %s run %s" % (path, telescope, label)),
                      row["reason"])
    return out


def not_good(runs, good):
    """{label: reason} for every run of `runs` (ints or merge tuples) that `good` (good_list)
    does not mark good; a run missing from the list counts as not good."""
    out = {}
    for run in runs:
        flag, reason = good.get(run_label(run), (False, "not on the good-run list"))
        if not flag:
            out[run_label(run)] = reason
    return out


def merge_rule(runs, merges, display_class, good):
    """Which merges replace their runs on a result figure. A merge replaces its runs only when
    every run of the merge is in the figure's display-list class (`display_class`, the run
    numbers of the class) and the good-run list (`good`, good_list) marks the merge good.

    runs     the figure's runs (ints), in drawing order
    merges   the merges available (tuples of run numbers; table_merges)
    Returns (drawn, taken, refused): drawn = `runs` with each taken merge in place of its runs
    (where its first run stood), taken = {merge: [the runs of `runs` it replaced]},
    refused = {merge: reason}."""
    cls = {int(r) for r in display_class}
    taken, refused = {}, {}
    for m in merges:
        m = tuple(sorted(int(r) for r in m))
        why = []
        off = [r for r in m if r not in cls]
        if off:
            why.append("run%s %s not in the display class" % ("s" if len(off) > 1 else "",
                                                             ", ".join(map(str, off))))
        flag, reason = good.get(run_label(m), (False, "not on the good-run list"))
        if not flag:
            why.append("not good on the good-run list (%s)" % reason)
        if why:
            refused[m] = "; ".join(why)
        else:
            taken[m] = [int(r) for r in runs if int(r) in m]
    drawn, done = [], set()
    for r in runs:
        m = next((m for m in taken if int(r) in m), None)
        if m is None:
            drawn.append(int(r))
        elif m not in done:
            drawn.append(m)
            done.add(m)
    return drawn, taken, refused


def display_class(path, step, rfsel=STANDARD_RFSEL, offset=STANDARD_OFFSET):
    """The display-list class (tables.classes_at_step) at RFSel `rfsel` and offset `offset` of
    one step: (runs {telescope: [run, ...]}, preferred {telescope: run or None})."""
    for c in tables.classes_at_step(path, step):
        if c.rfsel == rfsel and tables.os_offsets(c.os) == (offset,):
            return c.runs, c.preferred
    return {}, {}


# ------------------------------------------------------------------------------- clocks
def stop_band(step, min_h=STOP_BAND_MIN_H):
    """(0, span) in hours: the allowed range of the stop of `step` (RAD_STOP_BOUNDS_UTC: latest
    minus earliest) as seen on an axis counted from the earliest (RAD_STOP_UTC); None when the
    range is shorter than `min_h`."""
    lo, hi = (pd.Timestamp(t) for t in campaign.RAD_STOP_BOUNDS_UTC[step])
    span = (hi - lo).total_seconds() / 3600.0
    return (0.0, span) if span >= min_h else None


def hours_after_stop(run_times, telescope, runs, step, camp="july"):
    """{run: hours} from RAD_STOP_UTC[step] (the earliest stop the records allow) to each run
    start (frontend.blnw.time_after_stop on run_times). A run's time after the true stop lies
    between its hours minus the stop_band span and its hours."""
    _, hours = blnw.time_after_stop(run_times, camp, telescope, runs, campaign.RAD_STOP_UTC[step])
    return hours


def first_run_at(run_times, rs, telescope, step, camp="july"):
    """(run, start): the run of runs_summary at bookkeeping fluence `step` that started first."""
    runs = runs_at_step(rs, telescope, camp, step)
    rt = run_times[(run_times["campaign"] == camp) & (run_times["tel"] == telescope)
                   & run_times["run_num"].isin(runs)]
    if not len(rt):
        raise ValueError("no run-times row for %s %s at %g" % (telescope, camp, step))
    first = rt.loc[rt["start_utc"].idxmin()]
    return int(first["run_num"]), first["start_utc"]


def hours_since_first(run_times, rs, telescope, runs, step, camp="july"):
    """{run: hours} from the start of the first run at `step` (first_run_at) to each run start,
    and that first run: ({run: hours}, first)."""
    first, start = first_run_at(run_times, rs, telescope, step, camp)
    _, hours = blnw.time_after_stop(run_times, camp, telescope, runs, start)
    return hours, first


def inrun_currents(path=None):
    """{(telescope, run, chip): i_inrun_uA} from JULY_REF_CSV: the bias current of each board
    during the run, the median over the run's HV plateau; None where the table has none."""
    df = pd.read_csv(path or campaign.JULY_REF_CSV, usecols=["tel", "run_num", "board",
                                                              "i_inrun_uA"])
    out = {}
    for _, r in df.iterrows():
        k = (str(r["tel"]).lower(), int(r["run_num"]), style.chip_short(r["board"]))
        if k in out:
            raise ValueError("in-run currents: %s run %d %s listed twice" % k)
        out[k] = None if pd.isna(r["i_inrun_uA"]) else float(r["i_inrun_uA"])
    return out


# -------------------------------------------------------------------------------- colour
def look(p):
    """(campaign, fluence step) of a point: one look at the chips."""
    return (p["camp"], style.fluence_key(p["fluence"]))


def looks(pts):
    """The looks of `pts`, by step, then March before July."""
    return sorted({look(p) for p in pts}, key=lambda k: (k[1], CAMP_ORDER.index(k[0])))


def color_rule(pts):
    """"fluence" when `pts` hold several looks (campaign, fluence step), else "board"."""
    return "fluence" if len(looks(pts)) > 1 else "board"


def look_colors(pts):
    """{look: colour}: a step's first look its FLUENCE_COLOR, a later look at the same step the
    later-look hue (iv_plot.timing_color)."""
    out, seen = {}, {}
    for k in looks(pts):
        i = seen.get(k[1], 0)
        out[k] = timing_color(i, campaign.FLUENCE_COLOR[k[1]])
        seen[k[1]] = i + 1
    return out


def _color_of(pts, color_by):
    """(rule, colour function) for the points `pts` of a whole figure: the look colours depend
    on every look the figure holds, not on one panel's."""
    rule = color_by or color_rule(pts)
    if rule == "board":
        return rule, lambda p: BOARD_COLOR[p["board_idx"]]
    if rule != "fluence":
        raise ValueError("color_by %r: 'fluence' or 'board'" % (color_by,))
    colors = look_colors(pts)
    return rule, lambda p: colors[look(p)]


def _handles(pts, figure_pts, rule, color_of, scale):
    """The chip handles of `pts` and, when colour = look, the handles of the looks in `pts`
    (coloured as on the whole figure, `figure_pts`)."""
    present = set(looks(pts))
    return dict(chips=_chip_handles(pts, rule, color_of, scale),
                looks=[Line2D([], [], color=c, linewidth=3.0 * scale, label=look_text(k))
                       for k, c in look_colors(figure_pts).items() if k in present]
                if rule == "fluence" else [])


def look_text(k):
    """Legend text of a look: "March, 1.5e15 p/cm2" in mathtext."""
    return u"%s, %s" % (k[0].capitalize(), style.fluence_label(k[1]))


# ------------------------------------------------------------------------------ drawing
def _draw(ax, pts, x_of, color_of, open_of, series_of, half_width, scale):
    """Every point at x_of(p): a band where it has one, else the marker and its bar; then one
    line per series (series_of) through the mean at each x."""
    for p in pts:
        c = color_of(p)
        x = x_of(p)
        kw = style.point_style(p["chip"], p["fluence"], merged=p["merged"], scale=scale,
                               line=False, open_marker=bool(open_of(p)), color=c)
        if p["band"]:
            tables.draw_band(ax, x, p["band"], c, half_width=half_width, capsize=2.5 * scale,
                             marker=style.chip_marker(p["chip"]), markersize=8.5 * scale,
                             markerfacecolor=kw["markerfacecolor"], alpha=0.3)
        else:
            ax.errorbar([x], [p["y_ps"]], yerr=[p["yerr_ps"]], zorder=3, **kw)
    series = {}
    for p in pts:
        series.setdefault(series_of(p), []).append(p)
    for s in series.values():
        xs = sorted({x_of(p) for p in s})
        if len(xs) < 2:
            continue
        ys = [np.mean([p["y_ps"] for p in s if x_of(p) == x]) for x in xs]
        ax.plot(xs, ys, color=color_of(s[0]), linestyle=style.chip_linestyle(s[0]["chip"]),
                linewidth=1.8 * scale, zorder=2)


def _chip_handles(pts, rule, color_of, scale):
    chips = []
    for p in sorted(pts, key=lambda p: p["board_idx"]):
        if p["chip"] not in [c for c, _ in chips]:
            chips.append((p["chip"], p))
    out = []
    for chip, p in chips:
        c = color_of(p) if rule == "board" else style.INK
        out.append(Line2D([], [], color=c, marker=style.chip_marker(chip),
                          markersize=8.5 * scale, markerfacecolor=c, markeredgecolor=c,
                          linestyle=style.chip_linestyle(chip), linewidth=1.8 * scale,
                          label=chip))
    return out


def bias_scan(ax, pts, color_by=None, open_of=None, half_width=None, figure_pts=None,
              scale=SCALE):
    """Draw `pts` (one telescope) against the bias: x = each board's own bias (p["hv"]), points
    at one bias stacked at that bias (no jitter). Colour by look or by board (`color_by`:
    "fluence", "board" or None for color_rule over `figure_pts`, the whole figure's points,
    default `pts`), marker and line style by chip, an open marker where open_of(p), the thick
    black edge on merged runs; a band where a point has one (`half_width` in V, default 0.8 % of
    the axis span). One line per (chip, look) through the mean at each bias. Returns the legend
    handles: dict(chips=[...], looks=[...])."""
    open_of = open_of or (lambda p: False)
    figure_pts = figure_pts or pts
    rule, color_of = _color_of(figure_pts, color_by)
    if half_width is None:
        half_width = 0.008 * abs(np.diff(ax.get_xlim())[0])
    _draw(ax, pts, lambda p: p["hv"], color_of, open_of,
          lambda p: (p["chip"], look(p)), half_width, scale)
    return _handles(pts, figure_pts, rule, color_of, scale)


def time_series(ax, pts, hours, stop_band=None, color_by=None, open_of=None, half_width=None,
                figure_pts=None, scale=SCALE):
    """Draw `pts` (one telescope) against time: x = hours[p["run"]], every board of a run at its
    run's x (no jitter). Colour, markers, open markers and bands as bias_scan. One line per
    (chip, look, bias): a series breaks where the bias of its board changes. `stop_band` (0,
    span) in hours shades the allowed range of the irradiation stop at the origin (stop_band()).
    Returns the legend handles: dict(chips=[...], looks=[...])."""
    open_of = open_of or (lambda p: False)
    figure_pts = figure_pts or pts
    rule, color_of = _color_of(figure_pts, color_by)
    if stop_band:
        ax.axvspan(stop_band[0], stop_band[1], color=style.INK_MUTED, alpha=0.18, linewidth=0,
                   zorder=0, gid=DIVIDER_GID)
    if half_width is None:
        half_width = 0.008 * abs(np.diff(ax.get_xlim())[0])
    _draw(ax, pts, lambda p: hours[p["run"]], color_of, open_of,
          lambda p: (p["chip"], look(p), p["hv"]), half_width, scale)
    return _handles(pts, figure_pts, rule, color_of, scale)


def brace(ax, x0, x1, y, height, text, scale=SCALE):
    """A square brace over [x0, x1] at `y` (data units), `height` tall, with `text` centred
    above it (an in-axes label: the layout audit checks it against the data)."""
    xm = 0.5 * (x0 + x1)
    verts = [(x0, y), (x0, y + 0.5 * height), (xm, y + 0.5 * height), (xm, y + height),
             (xm, y + 0.5 * height), (x1, y + 0.5 * height), (x1, y)]
    ax.add_patch(PathPatch(Path(verts, [Path.MOVETO] + [Path.LINETO] * 6), linewidth=1.3,
                           edgecolor=style.INK, facecolor="none", zorder=5))
    ax.text(xm, y + 1.25 * height, text, ha="center", va="bottom", color=style.INK,
            fontsize=style.sizes(scale)["ann"], gid=LABEL_GID, zorder=6)


def kind_handles(merged=False, open_label=None, scale=SCALE):
    """Legend handles for the marker kinds: merged runs (filled, thick black edge, as point_style
    draws them; style.merged_handle is hollow and would read as the open marker beside it) and,
    with `open_label`, the open marker."""
    out = [Line2D([], [], color=style.INK_MUTED, marker="o", markersize=8.5 * scale,
                  markerfacecolor=style.INK_MUTED, markeredgecolor="#000000",
                  markeredgewidth=style.MERGED_EDGE_WIDTH * scale, linestyle="none",
                  label="merged runs")] if merged else []
    if open_label:
        out.append(Line2D([], [], color=style.INK, marker="o", markersize=8.5 * scale,
                          markerfacecolor=style.SURFACE, markeredgewidth=1.8 * scale,
                          linestyle="none", label=open_label))
    return out


# ------------------------------------------------------------------------------- figures
def _footer_in(text, scale):
    """The footer's height in inches (line height 1.2 x the footer font)."""
    return (text.count("\n") + 1) * 1.2 * style.sizes(scale)["ann"] / 72.0


def _row(n, footer, scale, aw=7.6, ah=4.6, gx=1.05):
    """`n` panels in one row, each aw x ah inches; the bottom margin holds the tick labels, the
    axis label and the footer."""
    fig, axes = blnw._grid(1, n, aw=aw, ah=ah, ml=1.2, mr=0.3, mt=0.62,
                           mb=0.7 + _footer_in(footer, scale), gx=gx, gy=0.0, scale=scale)
    return fig, axes[0]


def _axis(ax, xlim, ylim, xlabel, scale):
    # round x ticks (steps of 1, 2, 5, 10 x a power of ten): a tick at the very left edge of an
    # axis would touch the lowest y tick label
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 5, 10]))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(style.YLABEL_RES)
    return style.style_axes(ax, scale)


def _legend(ax, handles, loc, scale, ncol=1, title=None):
    s = style.sizes(scale)
    lg = ax.legend(handles=handles, loc=loc, ncol=ncol, fontsize=s["legend"], title=title,
                   title_fontsize=s["legend_title"], handletextpad=0.4, columnspacing=1.0,
                   borderaxespad=0.5, labelspacing=0.35, handlelength=2.2)
    ax.add_artist(lg)
    return lg


def _span(xs, pad, min_pad):
    lo, hi = min(xs), max(xs)
    d = max(pad * (hi - lo), min_pad)
    return (lo - d, hi + d)


def _numbers(pts, x_of, color_of, open_of):
    keys = ("tel", "camp", "run", "runs", "label", "board_idx", "chip", "value_ps", "bar_ps",
            "n_pixels", "hv", "fluence", "offset", "rfsel", "rfsel_source", "merged", "band",
            "y_ps", "yerr_ps")
    return [dict({k: p[k] for k in keys}, x=x_of(p), color=color_of(p),
                 open_marker=bool(open_of(p))) for p in pts]


def _scan(pts_by_tel, *, tel_titles, xlabel, ylim, tag, data, footer, color_by=None,
          open_of=None, xlims=None, kinds=None, kinds_loc="lower left", scale=SCALE):
    """A row of bias-scan panels, one per telescope of `tel_titles` (left to right): the chip
    legend upper left (one row), the look legend upper right (when colour = look), the marker
    kinds (`kinds`: handles) at `kinds_loc`. Returns (figure, numbers)."""
    tels = list(tel_titles)
    open_of = open_of or (lambda p: False)
    every = [p for t in tels for p in pts_by_tel[t]]
    rule, color_of = _color_of(every, color_by)
    fig, axes = _row(len(tels), footer, scale)
    numbers = []
    for ax, tel in zip(axes, tels):
        tp = pts_by_tel[tel]
        xlim = (xlims or {}).get(tel) or _span([p["hv"] for p in tp], 0.1, 15.0)
        _axis(ax, xlim, ylim, xlabel, scale)
        h = bias_scan(ax, tp, color_by=rule, open_of=open_of, figure_pts=every, scale=scale)
        _legend(ax, h["chips"], "upper left", scale, ncol=len(h["chips"]))
        if h["looks"]:
            _legend(ax, h["looks"], "upper right", scale)
        if kinds and kinds.get(tel):
            _legend(ax, kinds[tel], kinds_loc, scale)
        numbers += _numbers(tp, lambda p: p["hv"], color_of, open_of)
    style.compound_header(axes, tag=tag, data=data, scale=scale,
                          subjects=[tel_titles[t] for t in tels])
    style.footer(fig, footer, scale)
    return fig, dict(points=numbers, color_rule=rule, ylim_ps=list(ylim),
                     xlim={t: list(ax.get_xlim()) for t, ax in zip(tels, axes)})


def res_vs_bias_march_preirrad(pts_by_tel, **words):
    """re01: the March pre-irradiation bias scan, one panel per telescope, a series per chip,
    colour = board (one look; the step goes in the header through `data`). `words`: tel_titles,
    xlabel, ylim, tag, data, footer (and the optional xlims, scale). Returns (figure, numbers)."""
    return _scan(pts_by_tel, color_by="board", **words)


def res_vs_bias_march_steps(pts_by_tel, **words):
    """re02: the March bias scans after each irradiation step, colour = fluence step, marker =
    chip. `words` as res_vs_bias_march_preirrad. Returns (figure, numbers)."""
    return _scan(pts_by_tel, color_by="fluence", **words)


def res_1p5e15_march_vs_july(pts_by_tel, off_list, *, open_label, **words):
    """re03: the March scan at 1.5e15 and the July points at 1.5e15, colour = look (March the
    step's colour, July the later-look hue). `off_list` {telescope: {run, ...}}: runs drawn with
    an open marker (off the display-run list), named `open_label` in the marker legend; merged
    runs keep the thick black edge. Returns (figure, numbers)."""
    def open_of(p):
        return p["camp"] == "july" and p["run"] in off_list.get(p["tel"], ())
    kinds = {t: kind_handles(merged=any(p["merged"] for p in pts),
                             open_label=open_label if any(open_of(p) for p in pts) else None,
                             scale=words.get("scale", SCALE))
             for t, pts in pts_by_tel.items()}
    return _scan(pts_by_tel, color_by="fluence", open_of=open_of, kinds=kinds, **words)


def _time(pts_by_tel, hours, *, tel_titles, xlabel, ylim, tag, data, footer, run_labels,
          label_y, bold=None, stop=None, open_of=None, open_legend=None, xlims=None,
          scale=SCALE):
    """A row of time-series panels, one per telescope of `tel_titles`: x = hours[tel][run],
    colour = board, the chip legend upper left (one row); under each run its label
    (run_labels[tel][run], at `label_y` ps, bold where (tel, run) is in `bold`); `stop` shades
    the allowed-stop range (stop_band); `open_legend` {tel: (handles, loc, title)}."""
    tels = list(tel_titles)
    open_of = open_of or (lambda p: False)
    bold = bold or set()
    s = style.sizes(scale)
    fig, axes = _row(len(tels), footer, scale)
    numbers, labels = [], {}
    for ax, tel in zip(axes, tels):
        tp = pts_by_tel[tel]
        hr = hours[tel]
        xlim = (xlims or {}).get(tel) or _span([0.0] + [hr[p["run"]] for p in tp], 0.07, 1.0)
        _axis(ax, xlim, ylim, xlabel, scale)
        _, color_of = _color_of(tp, "board")
        h = time_series(ax, tp, hr, stop_band=stop, color_by="board", open_of=open_of,
                        scale=scale)
        _legend(ax, h["chips"], "upper left", scale, ncol=len(h["chips"]))
        if open_legend and open_legend.get(tel):
            handles, loc, title = open_legend[tel]
            _legend(ax, handles, loc, scale, title=title)
        for run in sorted({p["run"] for p in tp}, key=lambda r: hr[r]):
            text = run_labels[tel][run]
            ax.text(hr[run], label_y, text, ha="center", va="bottom", color=style.INK,
                    fontsize=s["ann"], gid=LABEL_GID, zorder=6, linespacing=1.15,
                    fontweight="bold" if (tel, run) in bold else "normal")
            labels["%s run %s" % (tel, run)] = dict(text=text, x=hr[run],
                                                    bold=(tel, run) in bold)
        numbers += _numbers(tp, lambda p: hr[p["run"]], color_of, open_of)
    style.compound_header(axes, tag=tag, data=data, scale=scale,
                          subjects=[tel_titles[t] for t in tels])
    style.footer(fig, footer, scale)
    return fig, axes, dict(points=numbers, color_rule="board", ylim_ps=list(ylim),
                           xlim={t: list(ax.get_xlim()) for t, ax in zip(tels, axes)},
                           run_labels=labels, stop_band_h=list(stop) if stop else None)


def res_july_1p5e15_stability(pts_by_tel, hours, *, open_runs, open_legend, **words):
    """re04: July 1.5e15 against the hours since the first run at the step, colour = board.
    `open_runs` {telescope: {run, ...}} are drawn with open markers and named by `open_legend`
    {telescope: (handles, loc, title)}. `words`: tel_titles, xlabel, ylim, tag, data, footer,
    run_labels, label_y, bold (and the optional xlims, scale). Returns (figure, numbers)."""
    def open_of(p):
        return p["run"] in open_runs.get(p["tel"], ())
    fig, _, numbers = _time(pts_by_tel, hours, open_of=open_of, open_legend=open_legend,
                            **words)
    return fig, numbers


def res_2e15_cooling_down(pts_by_tel, hours, *, stop, braces, **words):
    """re05: July 2e15 against the hours after the stop, colour = board, the allowed-stop range
    `stop` (stop_band) shaded at the origin; `braces` {telescope: (runs, text)}: a brace over
    those runs at 3 ps above their highest bar. Returns (figure, numbers)."""
    fig, axes, numbers = _time(pts_by_tel, hours, stop=stop, **words)
    numbers["braces"] = {}
    for ax, tel in zip(axes, list(words["tel_titles"])):
        if tel not in braces:
            continue
        runs, text = braces[tel]
        hr = hours[tel]
        tp = [p for p in pts_by_tel[tel] if p["run"] in runs]
        x0, x1 = min(hr[r] for r in runs), max(hr[r] for r in runs)
        y = max(p["y_ps"] + p["yerr_ps"] for p in tp) + 3.0
        brace(ax, x0, x1, y, 2.5, text, words.get("scale", SCALE))
        numbers["braces"][tel] = dict(runs=sorted(runs), x0=x0, x1=x1, y_ps=y, text=text)
    return fig, numbers
