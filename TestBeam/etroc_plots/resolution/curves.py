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
  one_per_condition
                one run per bias pattern (every board's bias) of a display class: the preferred
                run where it is in the group, else the latest run; then the merge rule
  merge_override
                a figure that overrides the merge rule says so in its footer: its runs stay
                single runs, and the merges the rule would take are named
  display_singles
                every display run of every step, as single runs (a summary over whole classes)

Clocks, from the run starts of RUN_TIMES_CSV (overview.timeline.read_run_times):

  hours_after_stop   run start minus RAD_STOP_UTC[step], the earliest stop the records allow
                     (frontend.blnw.time_after_stop); stop_band is the allowed-stop range
                     (0 to latest minus earliest, from RAD_STOP_BOUNDS_UTC), drawn at the origin
                     when it spans STOP_BAND_MIN_H or more
  hours_since_first  run start minus the start of the first run at the step (first_run_at)

Pixels (low_pixels): a point built from fewer than LOW_PIXELS pixels (half the chip) is drawn
as a small open marker on every figure here, the legend says "under 128 pixels" and the
figure's numbers list those points (low_pixels); the footer gives their counts.

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
  res_3p5e15_vs_bias_and_hours July 3.5e15 against the bias and against the hours after the
                               stop, bands, one row of two panels per telescope
  res_vs_bias_summary          every March standard run and the July runs the notebook gives,
                               colour = fluence step

The fluence ladder (best resolution per step, pixel yield) is resolution.ladder; the numbers
table is resolution.numbers_table; the combination split at one step is
resolution.combo_split.
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
LOW_PIXELS = 128            # a point from fewer pixels (half the chip) gets a small open marker
LOW_MARKER = 5.5            # its marker size, x the figure's scale (the small open markers)
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


def merge_override(runs, merges, display_class, good):
    """The merge rule overridden on one figure, which says so in its footer: `runs` are drawn
    as single runs. Arguments as merge_rule. Returns (drawn, overridden): drawn = `runs` as ints,
    in order; overridden = {merge: [the runs of `runs` it would have replaced]}, the merges
    merge_rule would have drawn in their place."""
    _, taken, _ = merge_rule(runs, merges, display_class, good)
    return [int(r) for r in runs], taken


def display_singles(classes, telescope):
    """Every display run of `classes` ({step: (runs {telescope: [run, ...]}, preferred)}, as
    display_class gives them per step) on one telescope, as single runs: by step, then by run
    number. No run of a class is left out (no one_per_condition) and no merge replaces any
    (merge_override names the merges a figure leaves out)."""
    return [int(r) for f in sorted(classes) for r in sorted(classes[f][0].get(telescope, []))]


def display_class(path, step, rfsel=STANDARD_RFSEL, offset=STANDARD_OFFSET):
    """The display-list class (tables.classes_at_step) at RFSel `rfsel` and offset `offset` of
    one step: (runs {telescope: [run, ...]}, preferred {telescope: run or None})."""
    for c in tables.classes_at_step(path, step):
        if c.rfsel == rfsel and tables.os_offsets(c.os) == (offset,):
            return c.runs, c.preferred
    return {}, {}


def one_per_condition(rs, telescope, runs, preferred=None, camp="july"):
    """One run per condition among `runs` (the runs of one display class): runs whose boards
    all sit at the same bias, threshold offset and RFSel (runs_summary, board by board) form a
    group; a group keeps the preferred run (`preferred`) where it is in the group, else its
    latest run (the highest run number: the runs of a campaign count up in time). Returns
    (kept, dropped): the kept runs sorted, dropped {run: why}. Apply merge_rule to the kept
    runs afterwards: a merge then replaces the kept run it holds."""
    tc, tk = campaign.TABLE_CAMPAIGN[camp], campaign.TABLE_TEL[telescope]
    groups = {}
    for r in sorted({int(x) for x in runs}):
        key = tuple((b["board_idx"], b["hv"], b["offset"], b["rfsel"])
                    for b in tables.run_settings(rs, tc, tk, r))
        groups.setdefault(key, []).append(r)
    kept, dropped = [], {}
    for g in groups.values():
        if preferred is not None and int(preferred) in g:
            keep, why = int(preferred), "the preferred run"
        else:
            keep, why = g[-1], "the later run"
        kept.append(keep)
        dropped.update({r: "same bias on every board as run %d, %s" % (keep, why)
                        for r in g if r != keep})
    return sorted(kept), dropped


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


# -------------------------------------------------------------------------------- pixels
def low_pixels(p, threshold=LOW_PIXELS):
    """True when the point `p` is built from fewer than `threshold` pixels (p["n_pixels"])."""
    return p["n_pixels"] < threshold


def low_pixel_points(pts, threshold=LOW_PIXELS):
    """The points of `pts` built from fewer than `threshold` pixels, as the values file lists
    them: [dict(tel, label, board_idx, chip, n_pixels), ...] in the order of `pts`."""
    return [dict(tel=p["tel"], label=p["label"], board_idx=p["board_idx"], chip=p["chip"],
                 n_pixels=p["n_pixels"]) for p in pts if low_pixels(p, threshold)]


def low_pixel_handle(scale=SCALE, threshold=LOW_PIXELS):
    """The legend handle of the small open marker: "under 128 pixels"."""
    return Line2D([], [], color=style.INK, marker="o", markersize=LOW_MARKER * scale,
                  markerfacecolor=style.SURFACE, markeredgewidth=1.4 * scale, linestyle="none",
                  label=u"under %d pixels" % threshold)


def _with_low(numbers, pts):
    """`numbers` with the points under LOW_PIXELS added (key low_pixels), when there are any."""
    low = low_pixel_points(pts)
    if low:
        numbers["low_pixels"] = dict(threshold=LOW_PIXELS, marker="small open", points=low)
    return numbers


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


def _handles(pts, figure_pts, rule, color_of, scale, look_label=None):
    """The chip handles of `pts` and, when colour = look, the handles of the looks in `pts`
    (coloured as on the whole figure, `figure_pts`), labelled by `look_label` (default
    look_text)."""
    look_label = look_label or look_text
    present = set(looks(pts))
    return dict(chips=_chip_handles(pts, rule, color_of, scale),
                looks=[Line2D([], [], color=c, linewidth=3.0 * scale, label=look_label(k))
                       for k, c in look_colors(figure_pts).items() if k in present]
                if rule == "fluence" else [])


def look_text(k):
    """Legend text of a look: "March, 1.5e15 p/cm2" in mathtext."""
    return u"%s, %s" % (k[0].capitalize(), style.fluence_label(k[1]))


def chip_look_text(k):
    """Legend text of a look with the fluence on the chips (CHIP_FLUENCE_TEXT, CHIP_FLUENCE_FACTOR
    x the bookkeeping step): "March, 1.12e15 p/cm2 on the chips"."""
    if k[1] == 0.0:
        return u"%s, %s" % (k[0].capitalize(), campaign.CHIP_FLUENCE_TEXT[0.0])
    return u"%s, %s %s on the chips" % (k[0].capitalize(), campaign.CHIP_FLUENCE_TEXT[k[1]],
                                        campaign.FLUENCE_UNIT)


# ------------------------------------------------------------------------------ drawing
def _draw(ax, pts, x_of, color_of, open_of, series_of, half_width, scale, linestyle_of=None):
    """Every point at x_of(p): a band where it has one, else the marker and its bar; then one
    line per series (series_of) through the mean at each x, in the chip's line style unless
    `linestyle_of` (a function of the series' first point) says otherwise. A point under
    LOW_PIXELS pixels (low_pixels) is a small open marker."""
    linestyle_of = linestyle_of or (lambda p: style.chip_linestyle(p["chip"]))
    for p in pts:
        c = color_of(p)
        x = x_of(p)
        small = low_pixels(p)
        kw = style.point_style(p["chip"], p["fluence"], merged=p["merged"], scale=scale,
                               line=False, open_marker=bool(open_of(p)) or small, color=c)
        if small:
            kw.update(markersize=LOW_MARKER * scale, markerfacecolor=style.SURFACE,
                      markeredgewidth=1.4 * scale)
        if p["band"]:
            tables.draw_band(ax, x, p["band"], c, half_width=half_width, capsize=2.5 * scale,
                             marker=style.chip_marker(p["chip"]),
                             markersize=(LOW_MARKER if small else 8.5) * scale,
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
        ax.plot(xs, ys, color=color_of(s[0]), linestyle=linestyle_of(s[0]),
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
              scale=SCALE, look_label=None):
    """Draw `pts` (one telescope) against the bias: x = each board's own bias (p["hv"]), points
    at one bias stacked at that bias (no jitter). Colour by look or by board (`color_by`:
    "fluence", "board" or None for color_rule over `figure_pts`, the whole figure's points,
    default `pts`), marker and line style by chip, an open marker where open_of(p), the thick
    black edge on merged runs; a band where a point has one (`half_width` in V, default 0.8 % of
    the axis span). One line per (chip, look) through the mean at each bias. Returns the legend
    handles: dict(chips=[...], looks=[...]), the looks labelled by `look_label` (default
    look_text)."""
    open_of = open_of or (lambda p: False)
    figure_pts = figure_pts or pts
    rule, color_of = _color_of(figure_pts, color_by)
    if half_width is None:
        half_width = 0.008 * abs(np.diff(ax.get_xlim())[0])
    _draw(ax, pts, lambda p: p["hv"], color_of, open_of,
          lambda p: (p["chip"], look(p)), half_width, scale)
    return _handles(pts, figure_pts, rule, color_of, scale, look_label)


def time_series(ax, pts, hours, stop_band=None, color_by=None, open_of=None, half_width=None,
                figure_pts=None, scale=SCALE):
    """Draw `pts` (one telescope) against time: x = hours[p["run"]], every board of a run at its
    run's x (no jitter). Colour, markers, open markers and bands as bias_scan. One line per
    (chip, look, bias, threshold offset): a series breaks where the bias or the offset of its
    board changes. `stop_band` (0, span) in hours shades the allowed range of the irradiation
    stop at the origin (stop_band()). Returns the legend handles: dict(chips=[...],
    looks=[...])."""
    open_of = open_of or (lambda p: False)
    figure_pts = figure_pts or pts
    rule, color_of = _color_of(figure_pts, color_by)
    if stop_band:
        ax.axvspan(stop_band[0], stop_band[1], color=style.INK_MUTED, alpha=0.18, linewidth=0,
                   zorder=0, gid=DIVIDER_GID)
    if half_width is None:
        half_width = 0.008 * abs(np.diff(ax.get_xlim())[0])
    _draw(ax, pts, lambda p: hours[p["run"]], color_of, open_of,
          lambda p: (p["chip"], look(p), p["hv"], p["offset"]), half_width, scale)
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


def kind_handles(merged=False, open_label=None, low=False, scale=SCALE):
    """Legend handles for the marker kinds: merged runs (filled, thick black edge, as point_style
    draws them; style.merged_handle is hollow and would read as the open marker beside it), with
    `open_label` the open marker, with `low` the small open marker (low_pixel_handle)."""
    out = [Line2D([], [], color=style.INK_MUTED, marker="o", markersize=8.5 * scale,
                  markerfacecolor=style.INK_MUTED, markeredgecolor="#000000",
                  markeredgewidth=style.MERGED_EDGE_WIDTH * scale, linestyle="none",
                  label="merged runs")] if merged else []
    if open_label:
        out.append(Line2D([], [], color=style.INK, marker="o", markersize=8.5 * scale,
                          markerfacecolor=style.SURFACE, markeredgewidth=1.8 * scale,
                          linestyle="none", label=open_label))
    if low:
        out.append(low_pixel_handle(scale))
    return out


# ------------------------------------------------------------------------------- figures
def _footer_in(text, scale):
    """The footer's height in inches (line height 1.2 x the footer font)."""
    return (text.count("\n") + 1) * 1.2 * style.sizes(scale)["ann"] / 72.0


def _row(n, footer, scale, aw=7.6, ah=4.6, gx=1.05, mr=0.3):
    """`n` panels in one row, each aw x ah inches; the bottom margin holds the tick labels, the
    axis label and the footer; the right margin `mr` inches (a legend outside the last panel)."""
    fig, axes = blnw._grid(1, n, aw=aw, ah=ah, ml=1.2, mr=mr, mt=0.62,
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
          open_of=None, xlims=None, kinds=None, kinds_loc="lower left", look_label=None,
          side=None, mr=0.3, scale=SCALE):
    """A row of bias-scan panels, one per telescope of `tel_titles` (left to right): the chip
    legend upper left (one row), the look legend upper right (when colour = look), the marker
    kinds (`kinds`: handles, and the small open marker where the panel has a point under
    LOW_PIXELS) at `kinds_loc`. With `side` (a list of extra handles, possibly empty) the look
    legend is drawn once, outside the last panel on its right, the looks of the whole figure
    followed by `side`; `mr` inches of right margin hold it. Returns (figure, numbers)."""
    tels = list(tel_titles)
    open_of = open_of or (lambda p: False)
    every = [p for t in tels for p in pts_by_tel[t]]
    rule, color_of = _color_of(every, color_by)
    fig, axes = _row(len(tels), footer, scale, mr=mr)
    numbers = []
    for ax, tel in zip(axes, tels):
        tp = pts_by_tel[tel]
        xlim = (xlims or {}).get(tel) or _span([p["hv"] for p in tp], 0.1, 15.0)
        _axis(ax, xlim, ylim, xlabel, scale)
        h = bias_scan(ax, tp, color_by=rule, open_of=open_of, figure_pts=every, scale=scale,
                      look_label=look_label)
        _legend(ax, h["chips"], "upper left", scale, ncol=len(h["chips"]))
        if h["looks"] and side is None:
            _legend(ax, h["looks"], "upper right", scale)
        kh = list((kinds or {}).get(tel) or [])
        if side is None and any(low_pixels(p) for p in tp):
            kh.append(low_pixel_handle(scale))
        if kh:
            _legend(ax, kh, kinds_loc, scale)
        numbers += _numbers(tp, lambda p: p["hv"], color_of, open_of)
    if side is not None:
        s = style.sizes(scale)
        handles = _handles(every, every, rule, color_of, scale, look_label)["looks"] + list(side)
        lg = axes[-1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.03, 1.0),
                             borderaxespad=0.0, fontsize=s["legend"], handletextpad=0.4,
                             labelspacing=0.35, handlelength=2.2)
        axes[-1].add_artist(lg)
    style.compound_header(axes, tag=tag, data=data, scale=scale,
                          subjects=[tel_titles[t] for t in tels])
    style.footer(fig, footer, scale)
    return fig, _with_low(dict(points=numbers, color_rule=rule, ylim_ps=list(ylim),
                               xlim={t: list(ax.get_xlim()) for t, ax in zip(tels, axes)}),
                          every)


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
          low_loc="upper right", scale=SCALE):
    """A row of time-series panels, one per telescope of `tel_titles`: x = hours[tel][run],
    colour = board, the chip legend upper left (one row); under each run its label
    (run_labels[tel][run], at `label_y` ps, bold where (tel, run) is in `bold`); `stop` shades
    the allowed-stop range (stop_band); `open_legend` {tel: (handles, loc, title)}; the small
    open marker's legend at `low_loc` on a panel with a point under LOW_PIXELS."""
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
        if any(low_pixels(p) for p in tp):
            _legend(ax, [low_pixel_handle(scale)], low_loc, scale)
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
    return fig, axes, _with_low(dict(points=numbers, color_rule="board", ylim_ps=list(ylim),
                                     xlim={t: list(ax.get_xlim()) for t, ax in zip(tels, axes)},
                                     run_labels=labels, stop_band_h=list(stop) if stop else None),
                                [p for t in tels for p in pts_by_tel[t]])


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


def _place(where, n):
    """(loc, ncol) of a legend: `where` is a loc, a (loc, ncol) pair or None (upper left, `n`
    columns: one row)."""
    if where is None:
        return "upper left", n
    return (where, n) if isinstance(where, str) else tuple(where)


def res_3p5e15_vs_bias_and_hours(bias_pts, time_pts, hours, *, tel_titles, ylim, tag, data,
                                 footer, xlabel_bias, xlabel_hours, run_labels, label_y,
                                 groups, group_y, stop=None, open_runs=None, open_legend=None,
                                 legend_locs=None, xlims=None, scale=SCALE):
    """re06: one step (July 3.5e15), one row per telescope of `tel_titles`, top to bottom: on
    the left the resolution against the bias (bias_pts {tel: points}), on the right against the
    hours after the stop (time_pts {tel: points}, x = hours[tel][run]). Every point is a band
    (BAND_STEPS); colour = board, marker = chip. On the time panels: under each run its label
    (run_labels[tel][run], at `label_y` ps) and, at `group_y` ps, one text under each group of
    runs (groups[tel]: [(runs, text), ...]), centred on the group; runs of open_runs[tel] are
    drawn with open markers and named by open_legend {tel: (handles, loc, title)}; `stop`
    shades the allowed-stop range (stop_band). legend_locs {(tel, "bias" or "hours"): loc or
    (loc, ncol)} places the chip legend (default upper left, one row); xlims {(tel, panel): (lo,
    hi)}. Returns (figure, numbers)."""
    tels = list(tel_titles)
    open_runs = open_runs or {}
    open_legend = open_legend or {}
    legend_locs = legend_locs or {}
    xlims = xlims or {}
    s = style.sizes(scale)
    fig, axes = blnw._grid(len(tels), 2, aw=7.6, ah=4.6, ml=1.2, mr=0.3, mt=0.62,
                           mb=0.7 + _footer_in(footer, scale), gx=1.05, gy=1.3, scale=scale)
    numbers, labels = [], {}
    for row, tel in zip(axes, tels):
        ax_b, ax_t = row
        def open_of(p, tel=tel):
            return p["run"] in open_runs.get(tel, ())
        bp, tp, hr = bias_pts[tel], time_pts[tel], hours[tel]
        _, color_of = _color_of(bp + tp, "board")
        _axis(ax_b, xlims.get((tel, "bias")) or _span([p["hv"] for p in bp], 0.1, 15.0), ylim,
              xlabel_bias, scale)
        h = bias_scan(ax_b, bp, color_by="board", open_of=open_of, scale=scale)
        hb = h["chips"] + ([low_pixel_handle(scale)] if any(low_pixels(p) for p in bp) else [])
        loc, ncol = _place(legend_locs.get((tel, "bias")), len(hb))
        _legend(ax_b, hb, loc, scale, ncol=ncol)
        numbers += [dict(q, panel="bias") for q in _numbers(bp, lambda p: p["hv"], color_of,
                                                            open_of)]
        _axis(ax_t, xlims.get((tel, "hours"))
              or _span([0.0] + [hr[p["run"]] for p in tp], 0.07, 1.0), ylim, xlabel_hours, scale)
        h = time_series(ax_t, tp, hr, stop_band=stop, color_by="board", open_of=open_of,
                        scale=scale)
        ht = h["chips"] + ([low_pixel_handle(scale)] if any(low_pixels(p) for p in tp) else [])
        loc, ncol = _place(legend_locs.get((tel, "hours")), len(ht))
        _legend(ax_t, ht, loc, scale, ncol=ncol)
        if open_legend.get(tel):
            handles, loc, title = open_legend[tel]
            _legend(ax_t, handles, loc, scale, title=title)
        for run in sorted({p["run"] for p in tp}, key=lambda r: hr[r]):
            ax_t.text(hr[run], label_y, run_labels[tel][run], ha="center", va="bottom",
                      color=style.INK, fontsize=s["ann"], gid=LABEL_GID, zorder=6)
            labels["%s run %s" % (tel, run)] = dict(text=run_labels[tel][run], x=hr[run],
                                                    y_ps=label_y)
        for runs, text in groups.get(tel, []):
            x = 0.5 * (min(hr[r] for r in runs) + max(hr[r] for r in runs))
            ax_t.text(x, group_y, text, ha="center", va="bottom", color=style.INK,
                      fontsize=s["ann"], gid=LABEL_GID, zorder=6)
            labels["%s runs %s" % (tel, ", ".join(map(str, runs)))] = dict(text=text, x=x,
                                                                         y_ps=group_y)
        numbers += [dict(q, panel="hours") for q in _numbers(tp, lambda p: hr[p["run"]],
                                                             color_of, open_of)]
    s = style.compound_header(axes[0], tag=tag, data=data, scale=scale,
                              subjects=[tel_titles[tels[0]]] * 2)
    for row, tel in zip(axes[1:], tels[1:]):
        for ax in row:
            style.right_lines(ax, [tel_titles[tel]], s["title"])
    style.footer(fig, footer, scale)
    return fig, _with_low(dict(points=numbers, color_rule="board", ylim_ps=list(ylim),
                               xlim={"%s %s" % (t, k): list(ax.get_xlim())
                                     for t, row in zip(tels, axes)
                                     for k, ax in zip(("bias", "hours"), row)},
                               labels=labels, stop_band_h=list(stop) if stop else None),
                          [p for t in tels for p in bias_pts[t] + time_pts[t]])


def res_vs_bias_summary(pts_by_tel, *, chip_fluence=False, mr=None, **words):
    """re07: the bias scans of several steps and campaigns on one figure, one panel per
    telescope, the points the notebook gives (the resolution notebook: every March standard run
    and every July display run as a single run, display_singles); colour = fluence step (a later
    look at a step in the later-look hue), marker = chip, merged runs (where given) with the
    thick black edge, points under LOW_PIXELS small and open. The look legend stands once,
    outside the last panel, with the marker kinds the figure holds; with `chip_fluence` its
    labels give the fluence on the chips (chip_look_text) instead of the bookkeeping fluence.
    `mr` inches of right margin hold that legend (default: wide enough for either labelling).
    `words` as res_vs_bias_march_preirrad. Returns (figure, numbers)."""
    scale = words.get("scale", SCALE)
    every = [p for pts in pts_by_tel.values() for p in pts]
    merged = any(p["merged"] for p in every)
    low = any(low_pixels(p) for p in every)
    return _scan(pts_by_tel, color_by="fluence",
                 side=kind_handles(merged=merged, low=low, scale=scale),
                 look_label=chip_look_text if chip_fluence else look_text,
                 mr=mr or (3.7 if chip_fluence else 2.9), **words)
