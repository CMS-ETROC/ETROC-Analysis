"""Time resolution against the front-end settings at one fluence step (the settings notebook).

At 3.5e15 (SETTINGS_STEP of the campaign module) the July runs stepped two front-end settings: the
preamp feedback setting RFSel (the feedback resistor R_f, style.RFSEL_RF_KOHM) and the
discriminator threshold offset above the baseline. A settings class is one (RFSel, offset) of the
display-run list (resolution.tables.classes_at_step). Its offset is the list's text: "10", or
"8/10" for runs whose boards sit at different offsets; each board's own offset comes from
runs_summary (tables.run_settings). Every board of every run is drawn as a band (tables.band; the
step is one of BAND_STEPS).

Figures, the words on them passed in by the notebook:

  res_by_offset   resolution against each board's own threshold offset, one RFSel (st01)
  res_by_class    resolution against the settings class, classes grouped by offset (st02)
  rfsel_step      the RFSel step at (nearly) one offset: four quarters on one axis (st03)

One drawing primitive serves the three (place and draw_slots): category slots, the runs of a slot
side by side in run order, the boards of a run side by side in board order, one band per board;
colour = RFSel (etroc_style.RFSEL_COLOR), marker = chip. Three position functions place the slots
(offset_slots, class_slots, quarter_slots).

settings_guard checks, before anything is drawn, the run facts the figures rest on: the display
list, runs_summary and the autocalibration scan agree on RFSel and on the offsets. x_over_nw is
the threshold offset over the board's noise width, from the BL/NW table.
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
def class_points(bt, rs, display, fluence, variant, floor):
    """Every board of every run of the display-run list `display` (a path) at `fluence`: one dict
    per (telescope, run, board) with tel, run, board_idx, chip, rfsel and os (the run's class in
    the list), offset and hv (the board's own, from runs_summary `rs`), preferred (the run is its
    class's preferred run) and band (tables.band of the board table `bt` at `variant`, `floor`).
    Ordered by telescope (TELESCOPE_CHIPS), class (classes_at_step), run and board."""
    if not any(_close(fluence, f) for f in campaign.BAND_STEPS):
        raise ValueError("%g is not one of BAND_STEPS %s: the settings figures draw every board "
                         "as a band" % (fluence, campaign.BAND_STEPS))
    tc = campaign.TABLE_CAMPAIGN[CAMP]
    classes = tables.classes_at_step(display, fluence)
    out = []
    for tel, chips in campaign.TELESCOPE_CHIPS.items():
        for c in classes:
            for run in c.runs.get(tel, []):
                for b in tables.run_settings(rs, tc, campaign.TABLE_TEL[tel], run):
                    chip = style.chip_short(b["chip"])
                    if chip != chips[b["board_idx"]]:
                        raise ValueError("%s run %d board %d: runs_summary chip %s, the campaign "
                                         "has %s" % (tel, run, b["board_idx"], b["chip"],
                                                     chips[b["board_idx"]]))
                    out.append(dict(tel=tel, run=run, board_idx=b["board_idx"], chip=chip,
                                    rfsel=c.rfsel, os=c.os, offset=b["offset"], hv=b["hv"],
                                    preferred=run == c.preferred.get(tel),
                                    band=tables.band(bt, tel, run, b["board_idx"], variant,
                                                     floor, campaign=tc)))
    return out


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
    The runs of a slot stand side by side in run order, the boards of a run side by side in
    board order, one cell per board and `run_gap` cells between runs, centred on the slot; a run
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
        for run in sorted(runs):
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


def _legends(ax, tel, pts, legend_title, rfsel_label, scale):
    s = style.sizes(scale)
    chips = [c for c in campaign.TELESCOPE_CHIPS[tel] if any(p["chip"] == c for p in pts)]
    lc = ax.legend(handles=chip_handles(chips, scale), loc="upper left", ncol=len(chips),
                   fontsize=s["legend"], handletextpad=0.2, columnspacing=1.0, borderaxespad=0.5)
    ax.add_artist(lc)
    rfsels = [r for r in campaign.RFSEL_ORDER if any(p["rfsel"] == r for p in pts)]
    ax.legend(handles=rfsel_handles(rfsels, rfsel_label, scale), loc="upper right",
              ncol=len(rfsels), title=legend_title, fontsize=s["legend"],
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
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            if not (t.get_visible() and str(t.get_text()).strip()):
                continue
            b = box(t)
            if t in ax.get_yticklabels() and (b[3] < frame[1] or b[1] > frame[3]):
                continue                    # a y tick outside the view is not drawn
            ticks.append(dict(text=t.get_text(), box=b))
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
