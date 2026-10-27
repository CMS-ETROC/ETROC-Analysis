"""The fluence ladder: each board's time resolution at each fluence step (re08) and the pixels
behind the same points (re10). The numbers table (re09) lists the same points
(resolution.numbers_table).

The points, one per (telescope, step, board), come from step_winners:

  March   MARCH_DISPLAY_RUNS, one run per step: the standard run (threshold offset 20, RFSel 2 or
          none recorded) at the highest bias of the step. step_winners derives that run from
          the board table and runs_summary (standard_runs, highest_bias_run) and raises where
          the list disagrees.
  July    the preferred run of each step in the display-run list at RFSel 2 and offset 20,
          replaced by its merge where curves.merge_rule takes one.

Both looks at 1.5e15 stay: March, and July after four months of cooling down. Each point is the
board's value in its reference combination; at a step of BAND_STEPS it is a band (tables.band),
as on every resolution figure.

The x axis is the bookkeeping fluence, linear, in units of X_UNIT (1e15 p/cm2), ticked at the
steps. Where two looks share a step they stand DODGE to either side of it (March left, July
right): a fixed offset, stated in the footer; nothing else moves (no jitter).

Colour = board (etroc_style.BOARD_COLOR), marker = chip; line style = look (March solid, July
dashed): one line per (board, look) joins its steps, never across looks. The colour already
names the chip (one chip per board), so the line style is free for the look.

pixel_yield draws the pixel counts of the same points: at the points' event floor (solid, the
look's markers) and with any events (floor "none", dashed; floor_none_pixels), plus extra runs
as small open markers off the lines.
"""
import numpy as np
from matplotlib.lines import Line2D

from ..campaigns import active as campaign, check_required
from .. import style
from ..etroc_style import BOARD_COLOR
from . import curves, tables

check_required("resolution")

SCALE = curves.SCALE
X_UNIT = 1e15                  # the x axis counts bookkeeping fluence in these units
DODGE = 0.06                   # x offset (in X_UNIT) of two looks at one step
LOOK_LINESTYLE = {"march": "-", "july": "--"}
LOOK_TEXT = {"march": u"March", "july": u"July"}
XLABEL = u"Bookkeeping fluence [$10^{15}$ %s]" % campaign.FLUENCE_UNIT
YLABEL_PIXELS = u"Pixels with a resolution"


# ------------------------------------------------------------------------------- picks
def standard_runs(bt, rs, telescope, camp, step, variant="top", floor="300"):
    """The single runs of the board table (at `variant`, `floor`) whose boards all sit at
    bookkeeping fluence `step` and at the standard operating point (threshold offset
    curves.STANDARD_OFFSET, RFSel curves.STANDARD_RFSEL or none recorded), sorted."""
    tc, tk = campaign.TABLE_CAMPAIGN[camp], campaign.TABLE_TEL[telescope]
    at_step = set(curves.runs_at_step(rs, telescope, camp, step))
    out = []
    for r in curves.table_runs(bt, telescope, camp, variant, floor):
        if r not in at_step:
            continue
        if all(b["offset"] == curves.STANDARD_OFFSET
               and b["rfsel"] in (None, curves.STANDARD_RFSEL)
               for b in tables.run_settings(rs, tc, tk, r)):
            out.append(r)
    return out


def highest_bias_run(rs, telescope, runs, camp="march"):
    """The run of `runs` in which every board sits at the highest bias it has in any of `runs`
    (runs_summary). ValueError when no run, or more than one, is highest on every board."""
    tc, tk = campaign.TABLE_CAMPAIGN[camp], campaign.TABLE_TEL[telescope]
    hv = {int(r): {b["board_idx"]: b["hv"] for b in tables.run_settings(rs, tc, tk, r)}
          for r in runs}
    if not hv:
        raise ValueError("%s %s: no run to choose from" % (telescope, camp))
    boards = sorted(set().union(*(h.keys() for h in hv.values())))
    top = {i: max(h[i] for h in hv.values() if h.get(i) is not None) for i in boards}
    best = [r for r, h in sorted(hv.items()) if all(h.get(i) == top[i] for i in boards)]
    if len(best) != 1:
        raise ValueError("%s %s runs %s: %s at the highest bias on every board; one run is "
                         "needed" % (telescope, camp, sorted(hv), best or "no run"))
    return best[0]


def step_winners(bt, bm, rs, telescope, class_by_step, good, merges, march_runs=None,
                 variant="top", floor="300"):
    """One point per (step, board) of telescope `telescope`: curves.points dicts plus step (the
    bookkeeping step), pick (how the run was chosen) and preferred (July: the run, or the merge
    holding it, is the step's preferred run; March: None, March has no display list).

      March  the runs of `march_runs` (default MARCH_DISPLAY_RUNS[telescope]), one per step,
             each checked to be the highest-bias standard run of its step (standard_runs,
             highest_bias_run); a disagreement raises ValueError
      July   per step of `class_by_step` ({step: (runs {tel: [...]}, preferred {tel: run})},
             the display class of each step at RFSel 2, offset 20; curves.display_class), the
             preferred run, replaced by its merge where curves.merge_rule takes one (`merges`
             the July merges, `good` the July good_list)

    Returns (points, picks): picks one dict per (campaign, step) with the run label, the rule,
    and for July the merges taken and refused."""
    tc = {c: campaign.TABLE_CAMPAIGN[c] for c in ("march", "july")}
    tk = campaign.TABLE_TEL[telescope]
    march_runs = campaign.MARCH_DISPLAY_RUNS[telescope] if march_runs is None else march_runs
    pts, picks, seen = [], [], set()
    for run in march_runs:
        step = style.fluence_key(tables.common_settings(rs, tc["march"], tk, [run])["fluence"])
        if step in seen:
            raise ValueError("%s March: two listed runs at step %g" % (telescope, step))
        seen.add(step)
        cand = standard_runs(bt, rs, telescope, "march", step, variant, floor)
        best = highest_bias_run(rs, telescope, cand, "march")
        if best != int(run):
            raise ValueError("%s March step %g: the listed run %d is not the highest-bias "
                             "standard run of the step (run %d, of runs %s)"
                             % (telescope, step, int(run), best, cand))
        rule = u"highest-bias standard run of the step (of runs %s)" % u", ".join(map(str, cand))
        for p in curves.points(bt, bm, rs, telescope, [int(run)], "march", variant, floor):
            pts.append(dict(p, step=step, pick=rule, preferred=None))
        picks.append(dict(camp="march", step=step, run=str(int(run)), rule=rule,
                          standard_runs=cand))
    for step in sorted(class_by_step):
        runs, preferred = class_by_step[step]
        pref = preferred.get(telescope)
        if pref is None:
            raise ValueError("%s July step %g: the display class has no preferred run"
                             % (telescope, step))
        drawn, taken, refused = curves.merge_rule([pref], merges, runs[telescope], good)
        run = drawn[0]
        rule = (u"merge %s of the preferred run %d (merge rule)" % (curves.run_label(run), pref)
                if isinstance(run, tuple) else u"preferred run of the step")
        pp = curves.points(bt, bm, rs, telescope, [run], "july", variant, floor)
        if curves.standard(pp) != pp or any(style.fluence_key(p["fluence"]) != step
                                            for p in pp):
            raise ValueError("%s July run %s is not at the standard point at step %g"
                             % (telescope, curves.run_label(run), step))
        pts += [dict(p, step=step, pick=rule, preferred=True) for p in pp]
        picks.append(dict(camp="july", step=step, run=curves.run_label(run), rule=rule,
                          preferred=pref, display_runs=list(runs[telescope]),
                          taken={curves.run_label(m): r for m, r in taken.items()},
                          refused={curves.run_label(m): w for m, w in refused.items()}))
    return pts, picks


def floor_none_pixels(bt, bm, pts, variant="top", floor="none"):
    """{(tel, camp, label, board_idx): n_pixels} of each point's run and board at event floor
    `floor` (default "none": tracks with any events), reference combination, from the same table
    as the point (the merged table for a merge). A missing row gives None, never 0; two rows
    raise ValueError."""
    out = {}
    for p in pts:
        rows = tables.board_rows(bm if p["merged"] else bt, p["tel"], p["run"], variant, floor,
                                 campaign=campaign.TABLE_CAMPAIGN[p["camp"]],
                                 board_idx=p["board_idx"])
        if len(rows) > 1:
            raise ValueError("%s %s run %s board %d: %d reference rows at floor %s"
                             % (p["tel"], p["camp"], p["label"], p["board_idx"], len(rows),
                                floor))
        out[(p["tel"], p["camp"], p["label"], p["board_idx"])] = (
            int(rows.iloc[0]["n_pixels"]) if len(rows) else None)
    return out


# ------------------------------------------------------------------------------- drawing
def look_x(pts, dodge=DODGE):
    """(x_of, shared): x_of(p) = p["step"] in X_UNIT, moved `dodge` left (March) or right
    (July) at the steps where `pts` (the whole figure's points) hold both looks; shared = those
    steps."""
    shared = sorted(s for s in {p["step"] for p in pts}
                    if len({p["camp"] for p in pts if p["step"] == s}) > 1)

    def x_of(p):
        x = p["step"] / X_UNIT
        if p["step"] in shared:
            x += dodge if curves.CAMP_ORDER.index(p["camp"]) else -dodge
        return x
    return x_of, shared


def _board_color(p):
    return BOARD_COLOR[p["board_idx"]]


def chip_handles(pts, scale=SCALE):
    """Marker-only handles, one per chip of `pts` in board order, in the board's colour (the
    lines on these figures carry the look or the floor, not the chip)."""
    out, seen = [], set()
    for p in sorted(pts, key=lambda p: p["board_idx"]):
        if p["chip"] in seen:
            continue
        seen.add(p["chip"])
        c = _board_color(p)
        out.append(Line2D([], [], color=c, marker=style.chip_marker(p["chip"]),
                          markersize=8.5 * scale, markerfacecolor=c, markeredgecolor=c,
                          linestyle="none", label=p["chip"]))
    return out


def look_handles(pts, labels=None, scale=SCALE):
    """One ink line per look (campaign) of `pts` in its line style (LOOK_LINESTYLE), labelled
    by `labels` {campaign: text} (default LOOK_TEXT)."""
    labels = labels or LOOK_TEXT
    return [Line2D([], [], color=style.INK, linestyle=LOOK_LINESTYLE[c], linewidth=1.8 * scale,
                   label=labels[c]) for c in curves.CAMP_ORDER if any(p["camp"] == c for p in pts)]


def ladder(ax, pts, x_of, half_width=0.045, scale=SCALE):
    """Draw `pts` (one telescope) on the fluence ladder: x = x_of(p) (look_x), colour = board,
    marker = chip, a band where a point has one (half `half_width` wide, in X_UNIT), the thick
    black edge on merged runs, a small open marker under curves.LOW_PIXELS pixels; one line per
    (board, look) in the look's line style."""
    curves._draw(ax, pts, x_of, _board_color, lambda p: False,
                 lambda p: (p["board_idx"], p["camp"]), half_width, scale,
                 linestyle_of=lambda p: LOOK_LINESTYLE[p["camp"]])


def pixel_yield(ax, pts, any_counts, x_of, extras=(), scale=SCALE):
    """Draw the pixel counts of `pts` (one telescope): n_pixels at the points' floor as filled
    chip markers in the board's colour joined by a solid line per (board, look); the count with
    any events (any_counts[(tel, camp, label, board_idx)], floor_none_pixels) as a dashed line
    per (board, look), broken where a count is None; `extras` (points of other runs) as small
    open markers at x_of(p), off the lines."""
    for p in pts:
        kw = style.point_style(p["chip"], p["fluence"], merged=p["merged"], scale=scale,
                               line=False, color=_board_color(p))
        kw.pop("capsize"), kw.pop("elinewidth")
        ax.plot([x_of(p)], [p["n_pixels"]], zorder=3, **kw)
    for p in extras:
        c = _board_color(p)
        ax.plot([x_of(p)], [p["n_pixels"]], marker=style.chip_marker(p["chip"]),
                markersize=5.5 * scale, markerfacecolor=style.SURFACE, markeredgecolor=c,
                markeredgewidth=1.4 * scale, linestyle="none", zorder=3)
    series = {}
    for p in pts:
        series.setdefault((p["board_idx"], p["camp"]), []).append(p)
    for s in series.values():
        s = sorted(s, key=x_of)
        c = _board_color(s[0])
        ax.plot([x_of(p) for p in s], [p["n_pixels"] for p in s], color=c, linestyle="-",
                linewidth=1.8 * scale, zorder=2)
        ys = [any_counts.get((p["tel"], p["camp"], p["label"], p["board_idx"])) for p in s]
        ax.plot([x_of(p) for p in s], [np.nan if y is None else y for y in ys], color=c,
                linestyle="--", linewidth=1.5 * scale, zorder=2)


# ------------------------------------------------------------------------------- figures
def _steps_axis(ax, steps, ylim, xlabel, ylabel, scale):
    xs = [s / X_UNIT for s in steps]
    curves._axis(ax, (min(xs) - 0.25, max(xs) + 0.35), ylim, xlabel, scale)
    ax.set_xticks(xs)
    ax.set_xticklabels([u"%g" % x for x in xs])
    ax.set_ylabel(ylabel)


def _ladder_numbers(pts, x_of, **extra):
    out = []
    for q, p in zip(curves._numbers(pts, x_of, _board_color, lambda p: False), pts):
        q.update(step=p["step"], pick=p["pick"], preferred=p["preferred"])
        q.update({k: v(p) for k, v in extra.items()})
        out.append(q)
    return out


def best_res_vs_fluence(pts_by_tel, *, tel_titles, ylim, tag, data, footer, xlabel=XLABEL,
                        legend_locs=None, half_width=0.045, scale=SCALE):
    """re08: the step winners (step_winners) on the fluence ladder, one panel per telescope of
    `tel_titles` (left to right): colour = board, marker = chip, line style = look, the two
    looks at a shared step dodged by DODGE. The chip legend sits upper left (one row), the look
    and merged-run legend lower right unless legend_locs {(tel, "chips" or "looks"): loc} says
    otherwise. Returns (figure, numbers)."""
    tels = list(tel_titles)
    legend_locs = legend_locs or {}
    every = [p for t in tels for p in pts_by_tel[t]]
    x_of, shared = look_x(every)
    steps = sorted({p["step"] for p in every})
    fig, axes = curves._row(len(tels), footer, scale)
    numbers = []
    for ax, tel in zip(axes, tels):
        tp = pts_by_tel[tel]
        _steps_axis(ax, steps, ylim, xlabel, style.YLABEL_RES, scale)
        ladder(ax, tp, x_of, half_width, scale)
        ch = chip_handles(tp, scale)
        curves._legend(ax, ch, legend_locs.get((tel, "chips"), "upper left"), scale,
                       ncol=len(ch))
        curves._legend(ax, look_handles(tp, scale=scale)
                       + curves.kind_handles(merged=any(p["merged"] for p in tp),
                                             low=any(curves.low_pixels(p) for p in tp),
                                             scale=scale),
                       legend_locs.get((tel, "looks"), "lower right"), scale)
        numbers += _ladder_numbers(tp, x_of)
    style.compound_header(axes, tag=tag, data=data, scale=scale,
                          subjects=[tel_titles[t] for t in tels])
    style.footer(fig, footer, scale)
    return fig, curves._with_low(dict(points=numbers, color_rule="board", line_style="look",
                                      dodge=DODGE, dodged_steps=shared, x_unit=X_UNIT,
                                      ylim_ps=list(ylim),
                                      xlim={t: list(ax.get_xlim()) for t, ax in zip(tels, axes)}),
                                 every)


def pixel_yield_vs_fluence(pts_by_tel, any_counts, extras_by_tel, *, tel_titles, ylim, tag,
                           data, footer, floor_label, any_label, extra_label, xlabel=XLABEL,
                           legend_locs=None, scale=SCALE):
    """re10: the pixel counts of the step winners on the fluence ladder (pixel_yield), one panel
    per telescope: solid at the points' floor (`floor_label`), dashed with any events
    (`any_label`, any_counts), extras_by_tel {tel: points} as small open markers
    (`extra_label`). The chip legend sits lower left (one row) and the line and marker kinds
    lower right, under the counts, unless legend_locs {(tel, "chips" or "kinds"): loc} says
    otherwise. Returns (figure, numbers)."""
    tels = list(tel_titles)
    legend_locs = legend_locs or {}
    every = [p for t in tels for p in pts_by_tel[t]]
    x_of, shared = look_x(every)
    steps = sorted({p["step"] for p in every})
    fig, axes = curves._row(len(tels), footer, scale)
    numbers, extras_out = [], []
    for ax, tel in zip(axes, tels):
        tp, ex = pts_by_tel[tel], list(extras_by_tel.get(tel, ()))
        _steps_axis(ax, steps, ylim, xlabel, YLABEL_PIXELS, scale)
        pixel_yield(ax, tp, any_counts, x_of, ex, scale)
        ch = chip_handles(tp, scale)
        curves._legend(ax, ch, legend_locs.get((tel, "chips"), "lower left"), scale,
                       ncol=len(ch))
        kinds = [Line2D([], [], color=style.INK, linestyle="-", linewidth=1.8 * scale,
                        marker="o", markersize=8.5 * scale, label=floor_label),
                 Line2D([], [], color=style.INK, linestyle="--", linewidth=1.5 * scale,
                        label=any_label)]
        if ex:
            kinds.append(Line2D([], [], color=style.INK, marker="o", markersize=5.5 * scale,
                                markerfacecolor=style.SURFACE, markeredgewidth=1.4 * scale,
                                linestyle="none", label=extra_label))
        kinds += curves.kind_handles(merged=any(p["merged"] for p in tp), scale=scale)
        curves._legend(ax, kinds, legend_locs.get((tel, "kinds"), "lower right"), scale)

        def n_any(p):
            return any_counts.get((p["tel"], p["camp"], p["label"], p["board_idx"]))
        numbers += _ladder_numbers(tp, x_of, n_pixels_any=n_any)
        extras_out += [dict(tel=p["tel"], camp=p["camp"], run=p["label"], label=p["label"],
                            board_idx=p["board_idx"], chip=p["chip"], step=p["step"],
                            hv=p["hv"], x=x_of(p), n_pixels=p["n_pixels"],
                            n_pixels_any=n_any(p)) for p in ex]
    style.compound_header(axes, tag=tag, data=data, scale=scale,
                          subjects=[tel_titles[t] for t in tels])
    style.footer(fig, footer, scale)
    return fig, dict(points=numbers, extras=extras_out, color_rule="board",
                     line_style="floor", dodge=DODGE, dodged_steps=shared, x_unit=X_UNIT,
                     ylim=list(ylim), xlim={t: list(ax.get_xlim()) for t, ax in zip(tels, axes)})
