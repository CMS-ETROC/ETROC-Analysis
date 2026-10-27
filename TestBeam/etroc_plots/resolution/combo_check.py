"""The combination checks (re16 to re18): each board's time resolution from three choices of
3-board combination, run by run, all read from the single-run board table's combination rows.

Rows, for board b with HV partner p (PARTNERS: boards 0-1 and 2-3 are HV pairs):

  reference         the board's reference combination (boards 0-1-2 for boards 0-2, 1-2-3 for
                    board 3): the value every result figure draws.
  partner_excluded  the one combination omitting p: board 0 from boards 0-2-3, board 1 from
                    1-2-3, board 2 from 0-1-2, board 3 from 0-1-3.
  midpoint          w_mid = sqrt((w_present^2 + w_absent^2) / 2), w_present the RMS of the board's
                    values in the two combinations holding both b and p, w_absent the
                    partner-excluded value; the bar by the same rule on the pixel spreads
                    (sp_present the RMS of the two present spreads); n_pixels the fewest of the
                    three combinations.

A combination is named by its boards (combo_split.combo_boards), never by the role words, which
change from run to run. A run-board whose combination is missing from the table (a run where a
board took no data has one combination only) gets no row; the estimators list it in `missing`.

The run axis of re18 carries what happened between consecutive runs: an HV cycle (hv_cycles: a
powered board's bias below HV_CYCLE_FRAC of its set value in at least one sample of the 60 s
log between the end of the previous run, in time, and the start of this one), an LV cycle or an
irradiation step (the campaign's HV/LV-cycle table) and a shared autocalibration scan
(autocal_groups: consecutive runs whose chips all carry the same scan id).

The figures: bias_check (re16, re17: three rows of bias panels, one column per telescope) and
run_check (re18: the same rows against the run). No band on any row: the figures compare the
combinations themselves (CONVENTIONS). Colour = fluence step, marker = chip; a point under
curves.LOW_PIXELS pixels is a small marker.
"""
import math

import pandas as pd
from matplotlib.lines import Line2D

from ..campaigns import active as campaign, check_required
from .. import style
from ..frontend import blnw
from . import curves, tables
from .combo_split import _bottom, combo_boards
from .settings import DIVIDER_GID, LABEL_GID

check_required("resolution")

PARTNERS = {0: 1, 1: 0, 2: 3, 3: 2}
KEYS = ["campaign", "telescope", "run", "board_idx"]
ROWS = ("reference", "partner_excluded", "midpoint")
HV_CYCLE_FRAC = 0.20
SCALE = curves.SCALE
EVENT_STYLE = {"hv": dict(color="#ff7f0e", linestyle="--", dx=-0.56),
               "lv": dict(color="#17becf", linestyle="-.", dx=-0.44),
               "irradiation": dict(color="#7b3fb5", linestyle="-", dx=-0.44)}
EVENT_TEXT = {"hv": u"HV cycle before the run", "lv": u"LV cycle before the run",
              "irradiation": u"irradiation step before the run"}
BOARD_DX = {0: -0.30, 1: -0.10, 2: 0.10, 3: 0.30}     # a board's x offset in its run's slot
LIGHT_ALPHA = 0.45


# ------------------------------------------------------------------------------ estimators
def _combos(bt):
    """The rows of `bt` with `boards` (the combination's boards, a sorted tuple) added; one row
    per (KEYS, boards), else ValueError."""
    t = bt.assign(boards=bt["combo"].map(combo_boards))
    dup = t.duplicated(KEYS + ["boards"], keep=False)
    if dup.any():
        raise ValueError("combinations listed twice: %s"
                         % t.loc[dup, KEYS + ["combo"]].values.tolist()[:6])
    return t


def _missing(key, row, boards):
    return dict(zip(KEYS, key), row=row,
                combination=u"boards %s" % u"-".join(str(b) for b in boards))


def _value(r):
    return dict(chip=r["board_chip"], combo=r["combo"], value_ps=float(r["value_ps"]),
                bar_ps=float(r["pixel_spread_ps"]), n_pixels=int(r["n_pixels"]))


def _by_board(bt, boards_of):
    """For each run-board of `bt` (its reference row names it): the row of the combination
    boards_of(board) gives. Returns (frame KEYS + chip, combo, value_ps, bar_ps, n_pixels,
    missing)."""
    t = _combos(bt)
    rows, missing = [], []
    for key, g in t.groupby(KEYS, sort=True):
        want = boards_of(int(key[3]))
        hit = g[g["boards"] == want]
        if len(hit):
            rows.append(dict(zip(KEYS, key), **_value(hit.iloc[0])))
        else:
            missing.append(_missing(key, None, want))
    return pd.DataFrame(rows, columns=KEYS + ["chip", "combo", "value_ps", "bar_ps",
                                             "n_pixels"]), missing


def reference_rows(bt):
    """Row 1: each run-board's reference combination (tables.REFERENCE_COLUMN true) of the
    combination table `bt` (one variant and event floor). Returns (frame, missing); a run-board
    without a reference row is in `missing`."""
    t = _combos(bt)
    rows, missing = [], []
    for key, g in t.groupby(KEYS, sort=True):
        hit = g[g[tables.REFERENCE_COLUMN].astype(str).str.lower() == "true"]
        if len(hit) > 1:
            raise ValueError("%s: %d reference combinations" % (key, len(hit)))
        if len(hit):
            rows.append(dict(zip(KEYS, key), **_value(hit.iloc[0])))
        else:
            missing.append(dict(zip(KEYS, key), row="reference", combination="reference"))
    return pd.DataFrame(rows, columns=KEYS + ["chip", "combo", "value_ps", "bar_ps",
                                             "n_pixels"]), missing


def excluded_boards(board, n_boards=4):
    """The boards of the combination omitting `board`'s HV partner: 0 -> (0, 2, 3)."""
    return tuple(b for b in range(n_boards) if b != PARTNERS[board])


def present_boards(board, n_boards=4):
    """The two combinations holding both `board` and its partner: 0 -> [(0, 1, 2), (0, 1, 3)]."""
    pair = {board, PARTNERS[board]}
    return [tuple(sorted(pair | {o})) for o in range(n_boards) if o not in pair]


def partner_excluded(bt):
    """Row 2: each run-board's value in the combination omitting its HV partner, from the
    combination table `bt` (one variant and event floor; every combination of every run-board).
    Returns (frame KEYS + chip, combo, value_ps, bar_ps, n_pixels, missing [dict(KEYS, row,
    combination)])."""
    out, missing = _by_board(bt, excluded_boards)
    for m in missing:
        m["row"] = "partner_excluded"
    return out, missing


def midpoint(bt):
    """Row 3: each run-board's pair-corrected midpoint (module docstring) from the combination
    table `bt`. Returns (frame KEYS + chip, value_ps, bar_ps, n_pixels, w_present, w_absent,
    sp_present, sp_absent, combos_present, combo_absent; missing): a run-board lacking any of
    its three combinations is in `missing`, one entry per combination lacking."""
    t = _combos(bt)
    rows, missing = [], []
    for key, g in t.groupby(KEYS, sort=True):
        b = int(key[3])
        need = present_boards(b) + [excluded_boards(b)]
        hits = [g[g["boards"] == w] for w in need]
        lack = [w for w, h in zip(need, hits) if not len(h)]
        if lack:
            missing += [_missing(key, "midpoint", w) for w in lack]
            continue
        p1, p2, a = (h.iloc[0] for h in hits)
        w_p = math.sqrt((float(p1["value_ps"]) ** 2 + float(p2["value_ps"]) ** 2) / 2.0)
        s_p = math.sqrt((float(p1["pixel_spread_ps"]) ** 2
                         + float(p2["pixel_spread_ps"]) ** 2) / 2.0)
        w_a, s_a = float(a["value_ps"]), float(a["pixel_spread_ps"])
        rows.append(dict(zip(KEYS, key), chip=a["board_chip"],
                         value_ps=math.sqrt((w_p ** 2 + w_a ** 2) / 2.0),
                         bar_ps=math.sqrt((s_p ** 2 + s_a ** 2) / 2.0),
                         n_pixels=int(min(p1["n_pixels"], p2["n_pixels"], a["n_pixels"])),
                         w_present=w_p, w_absent=w_a, sp_present=s_p, sp_absent=s_a,
                         combos_present=[p1["combo"], p2["combo"]], combo_absent=a["combo"]))
    return pd.DataFrame(rows), missing


def combination_rows(bt, variant="top", floor="300"):
    """The three rows (ROWS) of the board table `bt` at one track variant and event floor:
    {row: (frame, missing)}."""
    t = bt[(bt["variant"] == variant) & (bt["floor"].astype(str) == str(floor))]
    return {"reference": reference_rows(t), "partner_excluded": partner_excluded(t),
            "midpoint": midpoint(t)}


def row_points(pts, frame):
    """The points `pts` (curves.points, single runs) with value_ps, bar_ps and n_pixels from
    `frame` (one row's estimator frame), drawn as they are (band None, y = value): (points,
    missing [the (tel, run, board_idx) of `pts` the frame has no row for])."""
    idx = {(r["campaign"], r["telescope"], str(r["run"]), int(r["board_idx"])): r
           for _, r in frame.iterrows()}
    out, missing = [], []
    for p in pts:
        r = idx.get((campaign.TABLE_CAMPAIGN[p["camp"]], p["tel"], str(p["run"]),
                     p["board_idx"]))
        if r is None:
            missing.append((p["tel"], p["run"], p["board_idx"]))
            continue
        if style.chip_short(r["chip"]) != p["chip"]:
            raise ValueError("%s run %s board %d: %s in the row, %s in the point"
                             % (p["tel"], p["run"], p["board_idx"], r["chip"], p["chip"]))
        q = dict(p, value_ps=float(r["value_ps"]), bar_ps=float(r["bar_ps"]),
                 n_pixels=int(r["n_pixels"]), band=None)
        q.update(y_ps=q["value_ps"], yerr_ps=q["bar_ps"])
        for k in ("w_present", "w_absent", "sp_present", "sp_absent"):
            if k in r and not pd.isna(r[k]):
                q[k] = float(r[k])
        out.append(q)
    return out, missing


# ---------------------------------------------------------------------------- annotations
def hv_cycles(windows, timeline, status, boards=range(4), frac=HV_CYCLE_FRAC):
    """Per run of one telescope, per board, whether the HV cycled before it: {run: dict(
    prev_run, boards={board: True, False, "off", "no data" or "n/a"})}. `windows` {run: (start,
    end)}; `timeline` a frame of the 60 s log of that telescope (tb, channel = board, v, vset);
    `status` {(run, board): status}. Runs are taken in time order; the gap of a run runs from
    the previous run's end to its start (tb > end, tb <= start). A board whose status is
    "hv_off_by_design" or that has no status is "off"; a gap without samples of the board is
    "no data"; the first run is "n/a"; otherwise True when any sample with vset > 0 has v <
    `frac` x vset."""
    out, prev = {}, None
    for run in sorted(windows, key=lambda r: windows[r][0]):
        start, end = windows[run]
        rec = dict(prev_run=prev[0] if prev else None, boards={})
        for b in boards:
            if prev is None:
                rec["boards"][b] = "n/a"
                continue
            st = status.get((run, b))
            if st is None or st == "hv_off_by_design":
                rec["boards"][b] = "off"
                continue
            win = timeline[(timeline["channel"] == b) & (timeline["tb"] > prev[1])
                           & (timeline["tb"] <= start)]
            if not len(win):
                rec["boards"][b] = "no data"
                continue
            rec["boards"][b] = bool(((win["v"] < frac * win["vset"]) & (win["vset"] > 0)).any())
        out[run] = rec
        prev = (run, end)
    return out


def autocal_groups(ids, runs, chips):
    """Consecutive runs of `runs` (the figure's order) sharing their autocalibration scan:
    (groups [[run, ...], ...] of at least two runs where every chip of `chips` carries the same id
    as in the run before; partial {run: [chips sharing the previous run's id]} where some but
    not all do; no_id [runs without any id]). `ids` {run: {chip: id}}."""
    no_id = [r for r in runs if not ids.get(r)]
    groups, partial, cur = [], {}, list(runs[:1])
    for a, b in zip(runs, runs[1:]):
        pa, pb = ids.get(a) or {}, ids.get(b) or {}
        shared = [c for c in chips if c in pa and c in pb and pa[c] == pb[c]]
        if pa and pb and len(shared) == len(chips):
            cur.append(b)
            continue
        if len(cur) > 1:
            groups.append(cur)
        cur = [b]
        if pa and pb and shared:
            partial[b] = shared
    if len(cur) > 1:
        groups.append(cur)
    return groups, partial, no_id


# --------------------------------------------------------------------------------- figures
def _side_legend(ax, handles, scale):
    s = style.sizes(scale)
    lg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.03, 1.0),
                   borderaxespad=0.0, fontsize=s["legend"], handletextpad=0.4, labelspacing=0.4,
                   handlelength=2.2)
    ax.add_artist(lg)
    return lg


def _chip_handles(chips, scale):
    return [Line2D([], [], color=style.INK, marker=style.chip_marker(c), markersize=8.5 * scale,
                   markerfacecolor=style.INK, linestyle="none", label=c) for c in chips]


def _numbers(pts, row, x_of, color_of, extra=()):
    keys = ("tel", "camp", "run", "label", "board_idx", "chip", "value_ps", "bar_ps",
            "n_pixels", "hv", "fluence", "offset", "rfsel")
    return [dict({k: p[k] for k in keys}, row=row, x=x_of(p), color=color_of(p),
                 small=curves.low_pixels(p),
                 **{k: p[k] for k in ("w_present", "w_absent", "sp_present", "sp_absent")
                    if k in p}, **{k: p[k] for k in extra if k in p})
            for p in pts]


def _headers(axes, tels, tel_titles, row_titles, tag, data, scale):
    s = style.compound_header(axes[0], tag=tag, data=data, scale=scale,
                              subjects=[u"%s: %s" % (tel_titles[t], row_titles[0]) for t in tels])
    for row, title in zip(axes[1:], row_titles[1:]):
        for ax, t in zip(row, tels):
            style.right_lines(ax, [u"%s: %s" % (tel_titles[t], title)], s["title"])


def bias_check(rows, *, tel_titles, row_titles, xlabel, ylim, tag, data, footer,
               scale=SCALE):
    """re16, re17: one row of panels per entry of `rows` ({row: {tel: points}}, in ROWS order,
    top to bottom), one column per telescope of `tel_titles`; x = each board's bias, colour =
    look (curves.look_colors over every row's points), marker = chip, one line per chip and look
    through the mean at each bias, no band. The look, chip and small-marker legend stands
    outside the top right panel. Returns (figure, numbers)."""
    tels = list(tel_titles)
    names = [r for r in ROWS if r in rows]
    every = [p for r in names for t in tels for p in rows[r][t]]
    rule, color_of = curves._color_of(every, "fluence")
    fig, axes = blnw._grid(len(names), len(tels), aw=7.6, ah=4.2, ml=1.2, mr=2.9, mt=0.62,
                           mb=0.7 + curves._footer_in(footer, scale), gx=1.05, gy=1.0,
                           scale=scale)
    numbers, xlims = [], {}
    for t in tels:
        xlims[t] = curves._span([p["hv"] for r in names for p in rows[r][t]], 0.1, 15.0)
    for axrow, r in zip(axes, names):
        for ax, t in zip(axrow, tels):
            curves._axis(ax, xlims[t], ylim, xlabel, scale)
            curves.bias_scan(ax, rows[r][t], color_by=rule, figure_pts=every, scale=scale)
            numbers += _numbers(rows[r][t], r, lambda p: p["hv"], color_of)
    handles = curves._handles(every, every, rule, color_of, scale)["looks"]
    handles += _chip_handles(list(dict.fromkeys(p["chip"] for p in sorted(
        every, key=lambda p: (p["tel"] != tels[0], p["board_idx"])))), scale)
    if any(curves.low_pixels(p) for p in every):
        handles.append(curves.low_pixel_handle(scale))
    _side_legend(axes[0][-1], handles, scale)
    _headers(axes, tels, tel_titles, [row_titles[r] for r in names], tag, data, scale)
    style.footer(fig, footer, scale)
    return fig, dict(points=numbers, color_rule=rule, ylim_ps=list(ylim),
                     xlim={t: list(v) for t, v in xlims.items()})


def run_slots(runs):
    """{run: x}: the runs, sorted by number, one slot each at x = 1, 2, ..."""
    return {r: i + 1.0 for i, r in enumerate(sorted(runs))}


def _bracket(ax, x0, x1, y, depth, text, size):
    """An under-axis bracket from x0 to x1 (data x) at axes height `y`, its text below."""
    tr = ax.get_xaxis_transform()
    ax.plot([x0, x0, x1, x1], [y, y - depth, y - depth, y], transform=tr, color=style.INK_MUTED,
            linewidth=1.0, clip_on=False)
    return ax.text(0.5 * (x0 + x1), y - depth - 0.012, text, transform=tr, ha="center",
                   va="top", fontsize=size, color=style.INK_MUTED)


def run_check(rows, slots, events, autocal, *, tel_titles, row_titles, ylim, tag, data, footer,
              slot_in=0.46, scale=SCALE):
    """re18: the rows of bias_check against the run: one row of panels per entry of `rows`, one
    column per telescope; x = the run's slot (slots {tel: run_slots(...)}) plus the board's
    BOARD_DX. Colour = look, marker = chip, no lines, no band; each point's "kind" ("standard",
    "other" or "not good"): filled, filled at LIGHT_ALPHA, open. events {tel: {run: [kind, ...]}}
    (kinds of EVENT_STYLE): a vertical line in the gap before the run on every row.
    autocal {tel: (groups, partial, no_id)} (autocal_groups): under the bottom row, a bracket
    under each group of runs sharing an autocalibration scan, "partial" or "no id" under a run
    where the footer needs it. A point whose bar reaches past the top of `ylim` is an arrow at
    the top, open or filled by its kind, with its value beside it. Returns (figure, numbers)."""
    tels = list(tel_titles)
    names = [r for r in ROWS if r in rows]
    every = [p for r in names for t in tels for p in rows[r][t]]
    rule, color_of = curves._color_of(every, "fluence")
    s = style.sizes(scale)
    n = max(len(slots[t]) for t in tels)
    tick_in = 1.2 * s["tick"] / 72.0 + 0.12
    under = tick_in + 0.55            # the bottom row: its tick labels, then the brackets
    fig, axes = blnw._grid(len(names), len(tels), aw=slot_in * (n + 1), ah=4.2, ml=1.2, mr=3.6,
                           mt=0.62, mb=0.25 + under + curves._footer_in(footer, scale),
                           gx=0.9, gy=0.75, scale=scale)
    numbers = []
    for axrow, r in zip(axes, names):
        for ax, t in zip(axrow, tels):
            xs = slots[t]
            ax.set_xlim(0.3, len(xs) + 0.7)
            ax.set_ylim(*ylim)
            ax.set_xticks([xs[k] for k in sorted(xs)])
            ax.set_xticklabels([str(k) for k in sorted(xs)])
            ax.set_ylabel(style.YLABEL_RES)
            style.style_axes(ax, scale)
            ax.tick_params(axis="x", which="minor", bottom=False, top=False)
            for run, kk in sorted(events.get(t, {}).items()):
                for k in kk:
                    e = EVENT_STYLE[k]
                    ax.axvline(xs[run] + e["dx"], color=e["color"], linestyle=e["linestyle"],
                               linewidth=1.3, zorder=1, gid=DIVIDER_GID)
            for p in rows[r][t]:
                c = color_of(p)
                kind = p["kind"]
                small = curves.low_pixels(p)
                x = xs[p["runs"][0]] + BOARD_DX[p["board_idx"]]
                if p["y_ps"] + p["yerr_ps"] > ylim[1]:          # off scale: an arrow, its value
                    y = ylim[1] - 0.04 * (ylim[1] - ylim[0])
                    ax.plot([x], [y], marker="^", markersize=7.5 * scale, color=c,
                            markerfacecolor=style.SURFACE if kind == "not good" else c,
                            markeredgewidth=1.5 * scale, linestyle="none", zorder=3)
                    ax.text(x + 0.12, y, u"%.0f ps" % p["y_ps"], ha="left", va="center",
                            fontsize=s["ann"], color=style.INK, gid=LABEL_GID, zorder=6)
                    continue
                ax.errorbar([x], [p["y_ps"]],
                            yerr=[p["yerr_ps"]], color=c, marker=style.chip_marker(p["chip"]),
                            markersize=(curves.LOW_MARKER if small else 7.5) * scale,
                            markerfacecolor=style.SURFACE if kind == "not good" else c,
                            markeredgecolor=c, markeredgewidth=1.5 * scale, linestyle="none",
                            capsize=1.8 * scale, elinewidth=1.1 * scale, zorder=3,
                            alpha=LIGHT_ALPHA if kind == "other" else 1.0)
            numbers += [dict(q, off_scale=q["value_ps"] + q["bar_ps"] > ylim[1]) for q in
                        _numbers(rows[r][t], r,
                                 lambda p: xs[p["runs"][0]] + BOARD_DX[p["board_idx"]],
                                 color_of, extra=("kind",))]
    brackets = {}
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, t in zip(axes[-1], tels):
        xs = slots[t]
        groups, partial, no_id = autocal[t]
        y = _bottom(ax.get_xticklabels(), ax, renderer) - 0.02
        brackets[t] = []
        for g in groups:
            _bracket(ax, xs[g[0]] - 0.4, xs[g[-1]] + 0.4, y, 0.03, u"same autocalibration",
                     s["ann"])
            brackets[t].append(dict(runs=g, text=u"same autocalibration"))
        for run, word in [(r, u"partial") for r in partial] + [(r, u"no id") for r in no_id]:
            ax.text(xs[run], y - 0.012, word, transform=ax.get_xaxis_transform(), ha="center",
                    va="top", fontsize=s["ann"], color=style.INK_MUTED)
            brackets[t].append(dict(runs=[run], text=word))
    handles = curves._handles(every, every, rule, color_of, scale)["looks"]
    handles += _chip_handles(list(dict.fromkeys(p["chip"] for p in sorted(
        every, key=lambda p: (p["tel"] != tels[0], p["board_idx"])))), scale)
    ms = 7.5 * scale
    handles += [Line2D([], [], color=style.INK, marker="o", markersize=ms,
                       markerfacecolor=style.INK, linestyle="none",
                       label=u"good run, RFSel 2, offset 20"),
                Line2D([], [], color=style.INK, marker="o", markersize=ms,
                       markerfacecolor=style.INK, alpha=LIGHT_ALPHA, linestyle="none",
                       label=u"good run, other settings"),
                Line2D([], [], color=style.INK, marker="o", markersize=ms,
                       markerfacecolor=style.SURFACE, markeredgewidth=1.5 * scale,
                       linestyle="none", label=u"not good on the good-run list")]
    if any(curves.low_pixels(p) for p in every):
        h = curves.low_pixel_handle(scale)
        h.set_label(u"small: under %d pixels" % curves.LOW_PIXELS)
        handles.append(h)
    present = sorted({k for t in tels for kk in events.get(t, {}).values() for k in kk},
                     key=list(EVENT_STYLE).index)
    handles += [Line2D([], [], color=EVENT_STYLE[k]["color"],
                       linestyle=EVENT_STYLE[k]["linestyle"], linewidth=1.3,
                       label=EVENT_TEXT[k]) for k in present]
    _side_legend(axes[0][-1], handles, scale)
    _headers(axes, tels, tel_titles, [row_titles[r] for r in names], tag, data, scale)
    style.footer(fig, footer, scale)
    return fig, dict(points=numbers, color_rule=rule, ylim_ps=list(ylim),
                     slots={t: {str(r): x for r, x in slots[t].items()} for t in tels},
                     events={t: {str(r): kk for r, kk in sorted(events.get(t, {}).items())}
                             for t in tels},
                     autocal=brackets)
