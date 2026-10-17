"""Time resolution against the front-end settings at one fluence step (the settings notebook).

At 3.5e15 (SETTINGS_STEP of the campaign module) the July runs stepped two front-end settings: the
preamp feedback setting RFSel (the feedback resistor R_f, style.RFSEL_RF_KOHM) and the
discriminator threshold offset above the baseline. A settings class is one (RFSel, offset) of the
display-run list (resolution.tables.classes_at_step). Its offset is the list's text: "10", or
"8/10" for runs whose boards sit at different offsets; each board's own offset comes from
runs_summary (tables.run_settings). Every board of every run is drawn as a band (tables.band; the
step is one of BAND_STEPS). Given the merged runs' board table, class_points draws a class's
accepted merge (tables.classes_at_step) in place of its runs: one band per board from the merged
table, the point's run the merge's name and its runs in p["runs"]; merge_x_over_nw gives the
merge's x_B.

Figures, the words on them passed in by the notebook:

  res_by_offset         resolution against each board's own threshold offset, one RFSel (st01)
  res_by_class          resolution against the settings class, classes grouped by offset (st02)
  rfsel_step            the RFSel step at (nearly) one offset: four quarters on one axis (st03)
  threshold_over_noise  x_B per board, x = chip, the classes side by side in each chip's slot
                        (st04)
  res_vs_x              resolution against x_B, one band per board, no x bars (st05)

One drawing primitive serves st01-st03 (place and draw_slots): category slots, the runs of a slot
side by side in run order, the boards of a run side by side in board order, one band per board;
colour = RFSel (etroc_style.RFSEL_COLOR), marker = chip. Three position functions place the slots
(offset_slots, class_slots, quarter_slots); st04 uses place too, with the chip as the slot, the
class in place of the run and the run in place of the board. st04 is the one settings figure whose
marker is not the chip: the chip is on its axis, so the marker is the offset class
(offset_markers).

settings_guard checks, before anything is drawn, the run facts the figures rest on: the display
list, runs_summary and the autocalibration scan agree on RFSel and on the offsets. x_over_nw is
the threshold offset over the board's noise width (x_B), from the BL/NW table; spread_text
quotes the spread of its linearised pixel-to-pixel bar over a set of boards.
"""
import numpy as np
from matplotlib.lines import Line2D

from ..campaigns import active as campaign, check_required
from .. import style
from ..checks import axes_legends
from ..etroc_style import RFSEL_COLOR
from ..frontend import blnw
from . import tables

check_required("settings")

CAMP = "july"           # the display-run list (DISPLAY_RUNS_JUL_CSV) lists July runs
SCALE = 0.8             # font and marker scale of the settings figures
SLOT_WIDTH = 0.85       # the width the fullest slot fills, in slot units (slots 1 apart)
RUN_GAP = 1.5           # cells between two runs of one slot (one cell per board)
MIN_RUN = 3             # cells a run takes at least: the labels of one-board runs stay apart
LABEL_GID = "settings_label"     # in-axes texts (run labels, quarter labels): the layout audit
DIVIDER_GID = "settings_divider"  # the dividers are not data: the layout audit skips them


def _close(a, b):
    return abs(float(a) - float(b)) <= 1e-6 * abs(float(b))


# ------------------------------------------------------------------------- points and checks
def class_points(bt, rs, display, fluence, variant, floor, bt_merged=None):
    """Every board of every run of the display-run list `display` (a path) at `fluence`: one dict
    per (telescope, run, board) with tel, run, board_idx, chip, rfsel and os (the run's class in
    the list), offset and hv (the board's own, from runs_summary `rs`), preferred (the run is its
    class's preferred run) and band (tables.band of the board table `bt` at `variant`, `floor`).
    Ordered by telescope (TELESCOPE_CHIPS), class (classes_at_step), run and board.

    With the merged runs' board table `bt_merged`, a class's accepted merges
    (tables.accepted_merges, tables.classes_at_step) stand in place of their runs: run is the
    merge's name (tables.merge_name), runs its runs, rfsel and offset the runs' common ones
    (tables.common_settings, which also stops on a chip or bias that differs between the runs),
    preferred when it holds the class's preferred run, band from `bt_merged` by the same rule."""
    if not any(_close(fluence, f) for f in campaign.BAND_STEPS):
        raise ValueError("%g is not one of BAND_STEPS %s: the settings figures draw every board "
                         "as a band" % (fluence, campaign.BAND_STEPS))
    tc = campaign.TABLE_CAMPAIGN[CAMP]
    merges = None if bt_merged is None else tables.accepted_merges(bt_merged, tc)
    classes = tables.classes_at_step(display, fluence, merges)
    out = []
    for tel, chips in campaign.TELESCOPE_CHIPS.items():
        for c in classes:
            for entry in tables.class_entries(c, tel):
                merge = isinstance(entry, tuple)
                runs = entry if merge else (entry,)
                run, table = (tables.merge_name(tel, runs), bt_merged) if merge else (entry, bt)
                if merge:
                    s = tables.common_settings(rs, tc, campaign.TABLE_TEL[tel], runs)
                    if (s["rfsel"], (s["offset"],)) != (c.rfsel, tables.os_offsets(c.os)):
                        raise ValueError("%s merge %s: RFSel %s, offset %s in runs_summary; the "
                                         "class is RFSel %d, offset %s" % (
                                             tel, run, s["rfsel"], s["offset"], c.rfsel, c.os))
                for b in tables.run_settings(rs, tc, campaign.TABLE_TEL[tel], runs[0]):
                    chip = style.chip_short(b["chip"])
                    if chip != chips[b["board_idx"]]:
                        raise ValueError("%s run %d board %d: runs_summary chip %s, the campaign "
                                         "has %s" % (tel, runs[0], b["board_idx"], b["chip"],
                                                     chips[b["board_idx"]]))
                    p = dict(tel=tel, run=run, board_idx=b["board_idx"], chip=chip,
                             rfsel=c.rfsel, os=c.os, offset=b["offset"], hv=b["hv"],
                             preferred=c.preferred.get(tel) in runs,
                             band=tables.band(table, tel, run, b["board_idx"], variant, floor,
                                              campaign=tc))
                    if merge:
                        p["runs"] = list(runs)
                    out.append(p)
    return out


def run_order(p):
    """A point's place in run order: its run, or a merge's first run (p["runs"])."""
    return p["runs"][0] if "runs" in p else p["run"]


def x_over_nw(blnw_table, campaign_key, tel, run, chip):
    """The threshold over the noise of one board in one run, from the BL/NW table
    (frontend.blnw.read_blnw): x = offset / nw_mean, the threshold offset (DAC) over the board's
    mean autocalibration noise width (DAC) over its live pixels (noise width above 0, n_live of
    them); bar = offset * nw_std / nw_mean**2, the pixel-to-pixel spread of the noise width
    carried to x to first order (a spread, not an uncertainty). Returns dict(x, bar, offset,
    nw_mean, nw_std, n_live)."""
    t = blnw_table
    r = t[(t["campaign"] == campaign_key) & (t["tel"] == tel) & (t["run"] == int(run))
          & (t["chip"] == chip)]
    if len(r) != 1:
        raise ValueError("%d BL/NW rows for %s %s run %s %s" % (len(r), campaign_key, tel, run,
                                                                 chip))
    r = r.iloc[0]
    off, mean, std = float(r["offset"]), float(r["nw_mean"]), float(r["nw_std"])
    if not (np.isfinite(mean) and mean > 0):
        raise ValueError("%s %s run %s %s: no noise width (nw_mean %r)" % (campaign_key, tel, run,
                                                                           chip, mean))
    return dict(x=off / mean, bar=off * std / mean ** 2, offset=off, nw_mean=mean, nw_std=std,
                n_live=int(r["n_pixels_nw"]))


def merge_x_over_nw(blnw_table, campaign_key, tel, runs, chip):
    """x_B of one board in the merge of `runs`, each run with its own autocalibration scan: x and
    bar the means of the runs' x_over_nw, offset their common offset, nw_mean and nw_std the
    means, n_live the fewest; per_run keeps each run's x_over_nw (with its run). Raises
    ValueError when the runs' offsets differ."""
    per = [dict(x_over_nw(blnw_table, campaign_key, tel, r, chip), run=int(r)) for r in runs]
    if len({p["offset"] for p in per}) != 1:
        raise ValueError("%s %s runs %s: offsets %s in the BL/NW table"
                         % (campaign_key, tel, list(runs), [p["offset"] for p in per]))
    mean = {k: float(np.mean([p[k] for p in per])) for k in ("x", "bar", "nw_mean", "nw_std")}
    return dict(mean, offset=per[0]["offset"], n_live=min(p["n_live"] for p in per),
                per_run=per)


def settings_guard(rs, display, blnw_table, fluence=None):
    """Check the run facts the settings figures rest on, for every run of the display-run list
    `display` (a path) at `fluence` (default SETTINGS_STEP):

      (a) every board's RFSel in runs_summary `rs` equals the run's class RFSel in the list;
      (b) the set of the boards' threshold offsets in runs_summary equals the class's offset
          text ("8/10" = {8, 10});
      (c) every board's RFSel in runs_summary equals the RFSel the BL/NW table `blnw_table` read
          from the autocalibration scan (rows with rfsel_src "scan"), an independent record.

    Raises ValueError listing every failure, each naming its run; returns the counts checked:
    dict(runs, boards, boards_scan_rfsel)."""
    fluence = campaign.SETTINGS_STEP if fluence is None else fluence
    tc = campaign.TABLE_CAMPAIGN[CAMP]
    t = blnw_table
    bad, n_runs, n_boards, n_scan = [], 0, 0, 0
    for c in tables.classes_at_step(display, fluence):
        want = set(tables.os_offsets(c.os))
        for tel, runs in c.runs.items():
            for run in runs:
                n_runs += 1
                where = ("%s run %d (RFSel %d, offset %s in the display list)"
                         % (tel.upper(), run, c.rfsel, c.os))
                boards = tables.run_settings(rs, tc, campaign.TABLE_TEL[tel], run)
                if any(b["rfsel"] != c.rfsel for b in boards):
                    bad.append("%s: runs_summary RFSel per board %s"
                               % (where, [b["rfsel"] for b in boards]))
                if {b["offset"] for b in boards} != want:
                    bad.append("%s: runs_summary offset per board %s"
                               % (where, [b["offset"] for b in boards]))
                sub = t[(t["campaign"] == CAMP) & (t["tel"] == tel) & (t["run"] == run)
                        & (t["rfsel_src"] == "scan")]
                for b in boards:
                    n_boards += 1
                    s = sub[sub["board_idx"] == b["board_idx"]]
                    if not len(s):
                        continue
                    n_scan += 1
                    if int(s["rfsel"].iloc[0]) != b["rfsel"]:
                        bad.append("%s board %d (%s): runs_summary RFSel %s, autocalibration "
                                   "scan RFSel %d" % (where, b["board_idx"], b["chip"],
                                                      b["rfsel"], int(s["rfsel"].iloc[0])))
    if bad:
        raise ValueError("settings guard, %d problem(s):\n  %s" % (len(bad), "\n  ".join(bad)))
    return dict(runs=n_runs, boards=n_boards, boards_scan_rfsel=n_scan)


def good_run_drops(classes):
    """{(telescope, run): reason} for every run of `classes` (classes_at_step) that the good-run
    list (GOOD_RUNS_CSV, its GOOD_RUNS_CAMPAIGN rows: the July runs) does not mark good."""
    from ..iv.iv_timeseries import good_runs
    drops = {}
    for tel in sorted({t for c in classes for t in c.runs}):
        good, reasons = good_runs(tel)
        for c in classes:
            for run in c.runs.get(tel, []):
                if run not in good:
                    drops[(tel, run)] = reasons.get(run, "not on the good-run list")
    return drops


def run_facts(rs, run_times, classes, stop_utc):
    """{(telescope, run): dict(bias, hours, time)} for every run of `classes`: the bias of its
    boards as text (frontend.blnw.bias_text, from runs_summary `rs`) and the time from
    `stop_utc` (UTC) to the run start (frontend.blnw.time_after_stop on `run_times`), in hours
    and as text."""
    tc = campaign.TABLE_CAMPAIGN[CAMP]
    out = {}
    for c in classes:
        for tel, runs in c.runs.items():
            for run in runs:
                text, hours = blnw.time_after_stop(run_times, CAMP, tel, [run], stop_utc)
                hv = [b["hv"] for b in tables.run_settings(rs, tc, campaign.TABLE_TEL[tel], run)]
                out[(tel, run)] = dict(bias=blnw.bias_text(hv), hours=hours[run], time=text)
    return out


# ---------------------------------------------------------------------------- positions
def offset_slots(offsets):
    """st01: one slot per threshold offset, 1 apart in the order given. {offset: x}."""
    return {o: float(i) for i, o in enumerate(offsets)}


def class_slots(classes, group_gap=0.5):
    """st02: one slot per class (rfsel, os text) in the order given, 1 apart within an offset
    group (one os text) and 1 + `group_gap` apart where the os text changes.
    {(rfsel, os): x}."""
    out, x, prev = {}, 0.0, None
    for i, (rfsel, os_text) in enumerate(classes):
        if i:
            x += 1.0 + (group_gap if os_text != prev else 0.0)
        out[(rfsel, os_text)] = x
        prev = os_text
    return out


def quarter_slots(quarters, quarter=2.3, pitch=1.0):
    """st03: `quarters` = [(key, [offset, ...]), ...] in axis order, each quarter `quarter`
    wide, its offset columns `pitch` apart and centred on it. Returns ({(key, offset): x},
    [the quarters' centres])."""
    out, centres = {}, []
    for q, (key, offsets) in enumerate(quarters):
        c = (q + 0.5) * quarter
        centres.append(c)
        for j, o in enumerate(offsets):
            out[(key, o)] = c + (j - (len(offsets) - 1) / 2.0) * pitch
    return out, centres


# ---------------------------------------------------------------------- drawing primitive
def place(pts, slot_x, width=SLOT_WIDTH, run_gap=RUN_GAP, min_run=MIN_RUN):
    """Stamp p["x"] on every point; each carries p["slot"], a key of `slot_x` (slot -> centre).
    The runs of a slot stand side by side in run order (run_order: a merge at its first run), the
    boards of a run side by side in board order, one cell per board and `run_gap` cells between
    runs, centred on the slot; a run
    of fewer than `min_run` boards takes `min_run` cells, its boards centred in them. One cell for
    the whole figure: the fullest slot fills `width`. Returns (cell, {(slot, run): the x of the
    run's centre})."""
    groups = {}
    for p in pts:
        groups.setdefault(p["slot"], {}).setdefault(p["run"], []).append(p)

    def n_cells(runs):
        return sum(max(len(v), min_run) for v in runs.values()) + run_gap * (len(runs) - 1)

    cell = width / max([n_cells(r) for r in groups.values()] + [4.0])
    centres = {}
    for slot, runs in groups.items():
        x = slot_x[slot] - n_cells(runs) * cell / 2.0
        for run in sorted(runs, key=lambda r: run_order(runs[r][0])):
            boards = sorted(runs[run], key=lambda p: p["board_idx"])
            span = max(len(boards), min_run)
            x0 = x + (span - len(boards)) * cell / 2.0
            for k, p in enumerate(boards):
                p["x"] = x0 + (k + 0.5) * cell
            centres[(slot, run)] = x + span * cell / 2.0
            x += (span + run_gap) * cell
    return cell, centres


def draw_slots(ax, pts, cell, centres, run_labels, ylim, scale=SCALE):
    """One band per board at p["x"] (tables.draw_band): box and marker face in the RFSel colour,
    marker = chip; under each run its label (`run_labels[(tel, run)]`, bold for the preferred
    run of its class), at the bottom of the axis."""
    s = style.sizes(scale)
    for p in pts:
        c = RFSEL_COLOR[p["rfsel"]]
        tables.draw_band(ax, p["x"], p["band"], c, half_width=0.4 * cell, capsize=2.5,
                         marker=style.chip_marker(p["chip"]), markersize=8.5 * scale,
                         markerfacecolor=c, alpha=0.3)
    preferred = {(p["tel"], p["run"]) for p in pts if p["preferred"]}
    tel_of = {(p["slot"], p["run"]): p["tel"] for p in pts}
    y = ylim[0] + 0.025 * (ylim[1] - ylim[0])
    for key, x in centres.items():
        tr = (tel_of[key], key[1])
        ax.text(x, y, run_labels[tr], ha="center", va="bottom", fontsize=s["ann"] * 0.85,
                color=style.INK, fontweight="bold" if tr in preferred else "normal",
                linespacing=1.15, gid=LABEL_GID, zorder=4)


def divider(ax, x, solid=False):
    ax.axvline(x, color=style.INK if solid else style.INK_MUTED, linewidth=1.2 if solid else 0.9,
               linestyle="-" if solid else ":", zorder=1, gid=DIVIDER_GID)


def chip_handles(chips, scale=SCALE):
    """Legend handles: the chip's marker, ink on white (the drawn marker's face is the RFSel)."""
    return [Line2D([], [], color=style.INK, marker=style.chip_marker(c), markersize=8.5 * scale,
                   markerfacecolor=style.SURFACE, markeredgewidth=1.6, linestyle="none", label=c)
            for c in chips]


def rfsel_handles(rfsels, label=style.rfsel_text, scale=SCALE):
    """Legend handles: one per RFSel, a square in its colour, labelled label(rfsel)."""
    return [Line2D([], [], color=RFSEL_COLOR[r], marker="s", markersize=11.0 * scale,
                   markeredgecolor=style.INK, markeredgewidth=1.0, linestyle="none",
                   label=label(r)) for r in rfsels]


def _axis(ax, ticks, xlim, ylim, xlabel, scale):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([x for x, _ in ticks])
    ax.set_xticklabels([t for _, t in ticks])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(style.YLABEL_RES)
    s = style.style_axes(ax, scale)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.tick_params(axis="x", labelsize=s["tick"])
    for t in ax.get_xticklabels():
        t.set_linespacing(1.2)
    return s


def _legends(ax, tel, pts, legend_title, rfsel_label, scale, rfsel_ncol=None):
    """The chip legend upper left (one row) and the RFSel legend upper right, `rfsel_ncol`
    columns (None: one row)."""
    s = style.sizes(scale)
    chips = [c for c in campaign.TELESCOPE_CHIPS[tel] if any(p["chip"] == c for p in pts)]
    lc = ax.legend(handles=chip_handles(chips, scale), loc="upper left", ncol=len(chips),
                   fontsize=s["legend"], handletextpad=0.2, columnspacing=1.0, borderaxespad=0.5)
    ax.add_artist(lc)
    rfsels = [r for r in campaign.RFSEL_ORDER if any(p["rfsel"] == r for p in pts)]
    ax.legend(handles=rfsel_handles(rfsels, rfsel_label, scale), loc="upper right",
              ncol=rfsel_ncol or len(rfsels), title=legend_title, fontsize=s["legend"],
              title_fontsize=s["legend_title"], handletextpad=0.2, columnspacing=1.0,
              borderaxespad=0.5)


def _stack(n, aw, ah, ml, mr, mt, mb, gy, scale):
    """`n` axes stacked, each aw x ah inches; margins in inches."""
    import matplotlib.pyplot as plt
    style.apply_style(scale)
    W, H = ml + aw + mr, mb + n * ah + (n - 1) * gy + mt
    fig = plt.figure(figsize=(W, H))
    return fig, [fig.add_axes([ml / W, (mb + (n - 1 - i) * (ah + gy)) / H, aw / W, ah / H])
                 for i in range(n)]


def _footer_in(text, scale):
    """The footer's height in inches (line height 1.2 x the footer font). The bottom margin is
    this plus room for the tick labels and the axis label; save_figure then adds the footer's own
    band (style.lower_footer)."""
    return (text.count("\n") + 1) * 1.2 * style.sizes(scale)["ann"] / 72.0


def _panels(pts, slot_x, xlims, ticks, dividers, *, tel_titles, run_labels, xlabel, tag, data,
            legend_title, rfsel_label, footer, ylim, scale):
    """The stacked telescope panels of st01 and st02: one axis per telescope of `tel_titles`,
    top to bottom."""
    tels = list(tel_titles)
    cell, centres = place(pts, slot_x)
    fig, axes = _stack(len(tels), aw=14.0, ah=4.0, ml=1.25, mr=0.3, mt=0.8,
                       mb=0.7 + _footer_in(footer, scale), gy=1.3, scale=scale)
    for ax, tel in zip(axes, tels):
        tp = [p for p in pts if p["tel"] == tel]
        _axis(ax, ticks[tel], xlims[tel], ylim, xlabel, scale)
        for x, solid in dividers.get(tel, []):
            divider(ax, x, solid)
        draw_slots(ax, tp, cell, {k: v for k, v in centres.items() if k[0][0] == tel},
                   run_labels, ylim, scale)
        _legends(ax, tel, tp, legend_title, rfsel_label, scale)
    s = style.compound_header(axes[:1], tag=tag, data=data, scale=scale,
                              subjects=[tel_titles[tels[0]]])
    for ax, tel in zip(axes[1:], tels[1:]):
        style.right_lines(ax, [tel_titles[tel]], s["title"])
    style.footer(fig, footer, scale)
    return fig, dict(cell=cell, run_centres=centres)


# ----------------------------------------------------------------------------- figures
def res_by_offset(pts, offsets, *, tel_titles, tick_labels, run_labels, xlabel, tag, data,
                  legend_title, footer, ylim, rfsel_label=style.rfsel_text, scale=SCALE):
    """st01: resolution against each board's own threshold offset. One panel per telescope
    (`tel_titles` {telescope: subject}, top to bottom), one slot per offset of `offsets` (the
    same on every panel, labelled tick_labels[tel][offset]), each board in the slot of its own
    offset (a run whose boards sit at two offsets stands in both). `run_labels` {(tel, run):
    text} under each run; `tag` and `data` join the header's facility line. Returns (figure,
    layout: the cell, the run centres and the slots)."""
    xs = offset_slots(offsets)
    for p in pts:
        p["slot"] = (p["tel"], p["offset"])
    slot_x = {(t, o): x for t in tel_titles for o, x in xs.items()}
    xlim = (-0.6, len(offsets) - 0.4)
    fig, lay = _panels(pts, slot_x, {t: xlim for t in tel_titles},
                       {t: [(xs[o], tick_labels[t][o]) for o in offsets] for t in tel_titles}, {},
                       tel_titles=tel_titles, run_labels=run_labels, xlabel=xlabel, tag=tag,
                       data=data, legend_title=legend_title, rfsel_label=rfsel_label,
                       footer=footer, ylim=ylim, scale=scale)
    lay["slots"] = {"%s offset %s" % k: x for k, x in slot_x.items()}
    return fig, lay


def res_by_class(pts, classes, *, tel_titles, tick_labels, run_labels, xlabel, tag, data,
                 legend_title, footer, ylim, group_gap=0.5, rfsel_label=style.rfsel_text,
                 scale=SCALE):
    """st02: resolution against the settings class. One panel per telescope (as res_by_offset),
    one slot per class of `classes[tel]` ([(rfsel, os text), ...] in axis order), the classes of
    one offset text together and `group_gap` wider apart where it changes (a dotted line between
    the groups), labelled tick_labels[tel][(rfsel, os)]. Every panel spans the widest panel's
    range, so one slot has one width everywhere. Returns (figure, layout)."""
    for p in pts:
        p["slot"] = (p["tel"], (p["rfsel"], p["os"]))
    slot_x, xlims, ticks, dividers = {}, {}, {}, {}
    pos = {t: class_slots(classes[t], group_gap) for t in tel_titles}
    span = max(max(v.values()) for v in pos.values())
    for t in tel_titles:
        slot_x.update({(t, k): x for k, x in pos[t].items()})
        xlims[t] = (-0.6, span + 0.6)
        ticks[t] = [(pos[t][k], tick_labels[t][k]) for k in classes[t]]
        dividers[t] = [((pos[t][a] + pos[t][b]) / 2.0, False)
                       for a, b in zip(classes[t], classes[t][1:]) if a[1] != b[1]]
    fig, lay = _panels(pts, slot_x, xlims, ticks, dividers, tel_titles=tel_titles,
                       run_labels=run_labels, xlabel=xlabel, tag=tag, data=data,
                       legend_title=legend_title, rfsel_label=rfsel_label, footer=footer,
                       ylim=ylim, scale=scale)
    lay["slots"] = {"%s RFSel %d offset %s" % (t, k[0], k[1]): x
                    for (t, k), x in slot_x.items()}
    return fig, lay


def rfsel_step(pts, quarters, *, quarter_labels, tick_labels, run_labels, xlabel, name, tag,
               data, footer, ylim, quarter=2.3, scale=SCALE):
    """st03: the RFSel step at (nearly) one offset, on one axis of four equal quarters.
    `quarters` = [((tel, rfsel), [offset, ...]), ...] in axis order, one offset column each
    (each board in the column of its own offset); quarter_labels {(tel, rfsel): text} written at
    the top of each quarter in the RFSel colour (the colour key), tick_labels {(tel, rfsel,
    offset): text}; a solid line between the telescopes, dotted between the RFSel quarters of
    one telescope; one chip legend per telescope under its quarter labels. `name` is the header
    subject; `tag` and `data` join the facility line. Returns (figure, layout)."""
    import matplotlib.pyplot as plt
    slot_x, centres_q = quarter_slots(quarters, quarter)
    for p in pts:
        p["slot"] = ((p["tel"], p["rfsel"]), p["offset"])
    cell, centres = place(pts, slot_x)
    style.apply_style(scale)
    aw, ah, ml, mr, mt = 14.0, 4.6, 1.25, 0.3, 0.8
    mb = 0.7 + _footer_in(footer, scale)
    W, H = ml + aw + mr, mb + ah + mt
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([ml / W, mb / H, aw / W, ah / H])
    s = _axis(ax, [(x, tick_labels[(k[0][0], k[0][1], k[1])]) for k, x in slot_x.items()],
              (0.0, quarter * len(quarters)), ylim, xlabel, scale)
    tels = []
    for q, ((tel, rfsel), _) in enumerate(quarters):
        if q:
            divider(ax, q * quarter, solid=tel != quarters[q - 1][0][0])
        if tel not in tels:
            tels.append(tel)
        ax.text(centres_q[q], ylim[1] - 0.03 * (ylim[1] - ylim[0]), quarter_labels[(tel, rfsel)],
                ha="center", va="top", fontsize=s["ann"], color=RFSEL_COLOR[rfsel],
                fontweight="bold", gid=LABEL_GID, zorder=4)
    draw_slots(ax, pts, cell, centres, run_labels, ylim, scale)
    for tel in tels:
        q0 = [q for q, ((t, _), _) in enumerate(quarters) if t == tel][0]
        chips = [c for c in campaign.TELESCOPE_CHIPS[tel] if any(p["chip"] == c and
                                                                 p["tel"] == tel for p in pts)]
        lg = ax.legend(handles=chip_handles(chips, scale), loc="upper left", ncol=len(chips),
                       bbox_to_anchor=(q0 / float(len(quarters)), 0.88), fontsize=s["legend"],
                       handletextpad=0.2, columnspacing=1.0, borderaxespad=0.5)
        ax.add_artist(lg)
    style.compound_header([ax], tag=tag, data=data, scale=scale, subjects=[name])
    style.footer(fig, footer, scale)
    return fig, dict(cell=cell, run_centres=centres,
                     slots={"%s RFSel %d offset %s" % (k[0][0], k[0][1], k[1]): x
                            for k, x in slot_x.items()},
                     quarter_centres=centres_q)


OFFSET_MARKERS = ("v", "D", "o", "s", "^", "P", "X")   # st04, in the offset classes' order


def offset_markers(os_texts):
    """st04: {os text: marker}, one marker per offset class, in the order of
    tables.os_offsets ("6/8" before "8/10" before "10" ...); the same text gets the same marker
    on every panel, so pass the classes of every telescope."""
    texts = sorted(set(os_texts), key=tables.os_offsets)
    if len(texts) > len(OFFSET_MARKERS):
        raise ValueError("%d offset classes, %d markers" % (len(texts), len(OFFSET_MARKERS)))
    return {t: OFFSET_MARKERS[i] for i, t in enumerate(texts)}


def spread_text(bars, fmt, q=(10, 90)):
    """`fmt` filled with the spread of x_B's linearised bars over a set of boards: fields lo and
    hi (the q[0] and q[1] percentiles), min, max and n, the values rounded to 0.01."""
    b = np.asarray(list(bars), dtype=float)
    lo, hi = np.percentile(b, q)
    return fmt % dict(lo=round(float(lo), 2), hi=round(float(hi), 2),
                      min=round(float(b.min()), 2), max=round(float(b.max()), 2), n=len(b))


def _row(n, footer, scale, aw=7.6, ah=4.6, gx=1.05):
    """`n` panels in one row, each aw x ah inches; the bottom margin holds the tick labels, the
    axis label and the footer."""
    return blnw._grid(1, n, aw=aw, ah=ah, ml=1.2, mr=0.3, mt=0.62,
                      mb=0.7 + _footer_in(footer, scale), gx=gx, gy=0.0, scale=scale)


def threshold_over_noise(pts, classes, *, markers, class_labels, tel_titles, xlabel, ylabel, tag,
                         data, footer, ylim, run_gap=1.2, scale=SCALE):
    """st04: x_B of every board, one panel per telescope of `tel_titles` (left to right), x =
    chip. In each chip's slot the classes of `classes[tel]` ([(rfsel, os text), ...] in axis
    order) stand side by side, the runs of a class side by side in run order, `run_gap` cells
    between classes. Point = x_B with its linearised pixel-to-pixel bar (p["x_b"], p["bar"]);
    colour = RFSel, marker = markers[os text]; dotted lines between the chips. One legend entry
    per class, class_labels[(tel, rfsel, os)], in axis order. Returns (figure, layout)."""
    tels = list(tel_titles)
    proxies, slot_x = [], {}
    for tel in tels:
        chips = campaign.TELESCOPE_CHIPS[tel]
        slot_x.update({(tel, c): float(i) for i, c in enumerate(chips)})
        order = {k: i for i, k in enumerate(classes[tel])}
        for p in pts:
            if p["tel"] != tel:
                continue
            k = order[(p["rfsel"], p["os"])]
            runs = [r for _, r in sorted({(run_order(q), q["run"]) for q in pts if q["tel"] == tel
                                          and (q["rfsel"], q["os"]) == (p["rfsel"], p["os"])})]
            proxies.append(dict(slot=(tel, p["chip"]), run=k, board_idx=runs.index(p["run"]),
                                ref=p))
    cell, centres = place(proxies, slot_x, run_gap=run_gap, min_run=1)
    for q in proxies:
        q["ref"]["x"] = q["x"]
    fig, axes = _row(len(tels), footer, scale)
    axes = axes[0]
    s = style.sizes(scale)
    for ax, tel in zip(axes, tels):
        chips = campaign.TELESCOPE_CHIPS[tel]
        for p in [p for p in pts if p["tel"] == tel]:
            c = RFSEL_COLOR[p["rfsel"]]
            ax.errorbar([p["x"]], [p["x_b"]], yerr=[p["bar"]], color=c, ecolor=c,
                        marker=markers[p["os"]], markersize=7.5 * scale, markerfacecolor=c,
                        markeredgecolor=style.INK, markeredgewidth=0.9, linestyle="none",
                        elinewidth=1.3, capsize=2.5, zorder=3)
        for i in range(1, len(chips)):
            divider(ax, i - 0.5)
        ax.set_xlim(-0.5, len(chips) - 0.5)
        ax.set_ylim(*ylim)
        ax.set_xticks(range(len(chips)))
        ax.set_xticklabels(chips)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        style.style_axes(ax, scale)
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        h = [Line2D([], [], color=RFSEL_COLOR[r], marker=markers[o], markersize=7.5 * scale,
                    markerfacecolor=RFSEL_COLOR[r], markeredgecolor=style.INK,
                    markeredgewidth=0.9, linestyle="none", label=class_labels[(tel, r, o)])
             for r, o in classes[tel]]
        ax.legend(handles=h, loc="upper left", fontsize=s["legend"] * 0.85, handletextpad=0.3,
                  labelspacing=0.35, borderaxespad=0.5, borderpad=0.4)
    style.compound_header(axes, tag=tag, data=data, scale=scale,
                          subjects=[tel_titles[t] for t in tels])
    style.footer(fig, footer, scale)
    return fig, dict(cell=cell, class_centres={"%s %s class %d" % (k[0][0], k[0][1], k[1]): x
                                               for k, x in centres.items()})


def res_vs_x(pts, *, tel_titles, xlabel, xlim, ylim, legend_title, tag, data, footer,
             half_width=0.015, rfsel_label=style.rfsel_text, scale=SCALE):
    """st05: resolution against x_B (p["x_b"]), one panel per telescope of `tel_titles` (left to
    right), one band per board (tables.draw_band) at its x_B, no x bar: colour = RFSel, marker =
    chip; the chip legend upper left (one row), the RFSel legend (`legend_title`, one column)
    upper right. `xlim` and `ylim` the same on every panel. Returns the figure."""
    tels = list(tel_titles)
    fig, axes = _row(len(tels), footer, scale)
    axes = axes[0]
    for ax, tel in zip(axes, tels):
        tp = [p for p in pts if p["tel"] == tel]
        for p in tp:
            c = RFSEL_COLOR[p["rfsel"]]
            tables.draw_band(ax, p["x_b"], p["band"], c, half_width=half_width, capsize=2.5,
                             marker=style.chip_marker(p["chip"]), markersize=8.5 * scale,
                             markerfacecolor=c, alpha=0.3)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(style.YLABEL_RES)
        style.style_axes(ax, scale)
        _legends(ax, tel, tp, legend_title, rfsel_label, scale, rfsel_ncol=1)
    style.compound_header(axes, tag=tag, data=data, scale=scale,
                          subjects=[tel_titles[t] for t in tels])
    style.footer(fig, footer, scale)
    return fig


# ------------------------------------------------------------------------- layout audit
def layout_boxes(fig, dpi=200):
    """The pixel boxes [x0, y0, x1, y1] (PNG coordinates at `dpi`: origin top left) that
    layout_problems compares: the canvas, and per axes its frame, tick labels, legends, in-axes
    labels and data marks (band boxes, markers, bars and caps); plus every other text of the
    figure. Call it after style.save_figure, which lowers the footer."""
    fig.set_dpi(dpi)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    H = fig.bbox.height

    def box(artist):
        bb = artist.get_window_extent(renderer=r)
        return [round(bb.x0, 1), round(H - bb.y1, 1), round(bb.x1, 1), round(H - bb.y0, 1)]

    out = dict(canvas=[round(fig.bbox.width, 1), round(H, 1)], axes=[], texts=[])
    for t in fig.texts:
        if t.get_visible() and str(t.get_text()).strip():
            out["texts"].append(dict(text=t.get_text(), box=box(t)))
    for ax in fig.axes:
        if not ax.axison:
            continue
        frame = box(ax)
        ticks = []
        for k, (lim, labels) in enumerate(((ax.get_xlim(), ax.get_xticklabels()),
                                           (ax.get_ylim(), ax.get_yticklabels()))):
            lo, hi = sorted(lim)
            tol = 1e-9 * (hi - lo)
            for t in labels:
                if not (t.get_visible() and str(t.get_text()).strip()):
                    continue
                if not lo - tol <= t.get_position()[k] <= hi + tol:
                    continue                # a tick outside the view is not drawn
                ticks.append(dict(text=t.get_text(), box=box(t)))
        others = [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label,
                  getattr(ax, "_etroc_right_top", None)]
        others += [t for t in ax.texts if t.get_gid() != LABEL_GID]
        out["texts"] += [dict(text=t.get_text(), box=box(t)) for t in others
                         if t is not None and t.get_visible() and str(t.get_text()).strip()]
        marks = [a for a in list(ax.patches) + list(ax.lines) + list(ax.collections)
                 if a.get_gid() != DIVIDER_GID and a.get_visible()]
        out["axes"].append(dict(
            frame=frame, ticklabels=ticks,
            legends=[box(lg) for lg in axes_legends(ax)],
            labels=[dict(text=t.get_text(), box=box(t)) for t in ax.texts
                    if t.get_gid() == LABEL_GID],
            marks=[box(a) for a in marks]))
    return out


def _hit(a, b, pad):
    return a[0] < b[2] + pad and b[0] < a[2] + pad and a[1] < b[3] + pad and b[1] < a[3] + pad


def layout_problems(boxes, pad=2.0):
    """What layout_boxes shows touching, as a list of strings: a tick label or in-axes label off
    the canvas, a tick label touching its axes frame, another tick label or any other text, a
    legend touching data or an in-axes label, an in-axes label touching data or another in-axes
    label. `pad` pixels count as touching."""
    W, H = boxes["canvas"]
    probs = []
    every_tick = [(i, t) for i, a in enumerate(boxes["axes"]) for t in a["ticklabels"]]
    for i, a in enumerate(boxes["axes"]):
        for t in a["ticklabels"] + a["labels"]:
            b = t["box"]
            if b[0] < 0 or b[1] < 0 or b[2] > W or b[3] > H:
                probs.append("axes %d: %r runs off the canvas %s" % (i, t["text"], b))
        for t in a["ticklabels"]:
            if _hit(t["box"], a["frame"], pad):
                probs.append("axes %d: tick label %r touches the axes frame" % (i, t["text"]))
            for u in boxes["texts"]:
                if _hit(t["box"], u["box"], pad):
                    probs.append("axes %d: tick label %r touches %r" % (i, t["text"], u["text"]))
        for lg in a["legends"]:
            if any(_hit(lg, m, 0.0) for m in a["marks"]):
                probs.append("axes %d: a legend touches data" % i)
            for t in a["labels"]:
                if _hit(lg, t["box"], pad):
                    probs.append("axes %d: a legend touches the label %r" % (i, t["text"]))
        for k, t in enumerate(a["labels"]):
            if any(_hit(t["box"], m, 0.0) for m in a["marks"]):
                probs.append("axes %d: label %r touches data" % (i, t["text"]))
            for u in a["labels"][k + 1:]:
                if _hit(t["box"], u["box"], pad):
                    probs.append("axes %d: labels %r and %r touch" % (i, t["text"], u["text"]))
    for k, (i, t) in enumerate(every_tick):
        for j, u in every_tick[k + 1:]:
            if _hit(t["box"], u["box"], pad):
                probs.append("tick labels %r (axes %d) and %r (axes %d) touch"
                             % (t["text"], i, u["text"], j))
    return probs
