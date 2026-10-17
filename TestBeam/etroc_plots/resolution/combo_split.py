"""The combination split at one fluence step (re11): each board's time resolution in every
3-board combination containing it, run by run.

Runs (split_runs), per telescope: the reference runs, the preferred display run at RFSel 2 and
threshold offset 20 of each reference step (1.5e15 and 2e15 for IRRAD 2026), then every display
run of the step (3.5e15), class by class in the order of tables.classes_at_step, by run number
within a class. Every run is a single run: the figure compares the combinations of one run, and
a merge pools runs. So F1 runs 46 and 47 stay apart here, where the settings figures draw their
accepted merge.

Points (combo_points): one per (run, board, combination containing the board), from the
single-run board table at one track variant and event floor, every combination and not only the
anointed one: value_ps, bar_ps (the pixel-to-pixel spread of the board's map in that
combination), n_pixels, anointed; the board's chip and bias from runs_summary.

The figure (combo_split): one row per telescope, one panel per board. In a panel the runs stand
side by side, grouped by step and settings class (a dotted line between classes, a solid one
between steps), the combinations of a run side by side at COMBO_DX apart. Colour =
combination, by its three boards (COMBO_COLOR); filled = the board's anointed combination, open
= the others; marker = chip, as everywhere. No band: the figure compares the combinations
themselves (CONVENTIONS). Under each panel: each run's number and this board's bias, then the
class, then the step. A point under curves.LOW_PIXELS pixels is a small marker, filled or open
by the same rule.
"""
import re

from matplotlib.lines import Line2D

from ..campaigns import active as campaign, check_required
from .. import style
from ..frontend import blnw
from . import curves, tables
from .settings import divider

check_required("resolution")

SCALE = 0.7                  # font and marker scale: eight panels on one figure
COMBO_DX = 0.25              # x offset between the combinations of one run (run slots 1 apart)
GROUP_MIN = 1.6              # the narrowest class group, in run slots: its two-line label fits
TICK_SCALE = 0.85            # run/bias tick labels at this fraction of the tick size: 15 runs a panel
COMBO_COLOR = {(0, 1, 2): "#0072B2", (0, 1, 3): "#56B4E9", (0, 2, 3): "#009E73",
               (1, 2, 3): "#CC79A7"}     # no fluence hue: the steps are on the axis, not colours
STEP_RFSEL, STEP_OFFSET = curves.STANDARD_RFSEL, curves.STANDARD_OFFSET


def combo_boards(combo):
    """A combination's board indices, sorted: "extra0-dut1-ref2" -> (0, 1, 2)."""
    return tuple(sorted(int(re.search(r"\d+$", part).group()) for part in combo.split("-")))


def combo_text(boards):
    """"boards 0-1-2"."""
    return u"boards %s" % u"-".join(str(b) for b in boards)


def split_runs(path, ref_steps, step, rfsel=STEP_RFSEL, offset=STEP_OFFSET):
    """The runs of the split, per telescope, in x order: {telescope: [dict(run, step, rfsel, os,
    ref, preferred), ...]}. First the reference runs, one per step of `ref_steps`: the preferred
    run of the display-run list `path` in the class at RFSel `rfsel`, offset `offset`; then every
    run listed at `step`, class by class (tables.classes_at_step order), by run number within a
    class, each as a single run (no merge). `os` is the list's offset text ("20", "8/10"). A
    reference step without a preferred run on a telescope raises ValueError."""
    out = {}
    for f in ref_steps:
        runs, preferred = curves.display_class(path, f, rfsel, offset)
        for tel in sorted(set(runs) | set(preferred)):
            if preferred.get(tel) is None:
                raise ValueError("%s: no preferred run at %g, RFSel %d, offset %d"
                                 % (tel, f, rfsel, offset))
            out.setdefault(tel, []).append(dict(run=int(preferred[tel]), step=f, rfsel=rfsel,
                                                os=str(offset), ref=True, preferred=True))
    for c in tables.classes_at_step(path, step):
        for tel, runs in c.runs.items():
            out.setdefault(tel, []).extend(
                dict(run=int(r), step=step, rfsel=c.rfsel, os=c.os, ref=False,
                     preferred=c.preferred.get(tel) == r) for r in sorted(runs))
    return out


def combo_points(bt, rs, telescope, entries, camp="july", variant="top", floor="300"):
    """One dict per (entry, board, combination containing the board) of the single-run board
    table `bt`, for the runs `entries` (split_runs) of one telescope: the entry's keys (run,
    step, rfsel, os, ref, preferred) and tel, board_idx, chip, hv (runs_summary), combo,
    boards (combo_boards), value_ps, bar_ps, n_pixels, anointed. A run whose board has no
    anointed combination, or one whose chips differ from runs_summary, raises ValueError."""
    tc, tk = campaign.TABLE_CAMPAIGN[camp], campaign.TABLE_TEL[telescope]
    out = []
    for e in entries:
        boards = {b["board_idx"]: b for b in tables.run_settings(rs, tc, tk, e["run"])}
        rows = tables.board_rows(bt, telescope, e["run"], variant, floor, campaign=tc,
                                 reference=None)
        for i, b in sorted(boards.items()):
            chip = style.chip_short(b["chip"])
            mine = rows[rows["board_idx"] == i].sort_values("combo")
            if not (mine["anointed"] == True).any():          # noqa: E712  (a pandas mask)
                raise ValueError("%s run %d board %d: no anointed combination (variant %s, "
                                 "floor %s)" % (telescope, e["run"], i, variant, floor))
            for _, r in mine.iterrows():
                if style.chip_short(r["board_chip"]) != chip:
                    raise ValueError("%s run %d board %d: %s in the board table, %s in "
                                     "runs_summary" % (telescope, e["run"], i, r["board_chip"],
                                                       chip))
                out.append(dict(e, tel=telescope, board_idx=i, chip=chip, hv=b["hv"],
                                combo=r["combo"], boards=combo_boards(r["combo"]),
                                value_ps=float(r["value_ps"]),
                                bar_ps=float(r["pixel_spread_ps"]), n_pixels=int(r["n_pixels"]),
                                anointed=bool(r["anointed"])))
    return out


def group_key(e):
    """The group of an entry on the axis: (step, RFSel, offset text)."""
    return (e["step"], e["rfsel"], e["os"])


def layout(entries, group_min=GROUP_MIN):
    """x positions of the runs `entries` (split_runs, one telescope): consecutive entries of one
    group (group_key) form a group `group_min` run slots wide or one slot per run, whichever is
    wider, its runs 1 apart and centred in it. Returns dict(x={run: x}, groups=[(key, x0, x1,
    [runs])], width)."""
    groups = []
    for e in entries:
        if groups and groups[-1][0] == group_key(e):
            groups[-1][1].append(e["run"])
        else:
            groups.append((group_key(e), [e["run"]]))
    x, out, x0 = {}, [], 0.0
    for key, runs in groups:
        w = max(float(len(runs)), group_min)
        for i, r in enumerate(runs):
            x[r] = x0 + (w - len(runs)) / 2.0 + 0.5 + i
        out.append((key, x0, x0 + w, runs))
        x0 += w
    return dict(x=x, groups=out, width=x0)


def combo_x(pts, x_of_run, dx=COMBO_DX):
    """{(run, board_idx, combo): x}: the combinations of one board in one run side by side,
    `dx` apart, centred on the run's x, in combination-name order."""
    out = {}
    for (run, i) in sorted({(p["run"], p["board_idx"]) for p in pts}):
        combos = sorted(p["combo"] for p in pts if p["run"] == run and p["board_idx"] == i)
        k = len(combos)
        for j, c in enumerate(combos):
            out[(run, i, c)] = x_of_run[run] + (j - (k - 1) / 2.0) * dx
    return out


def _draw(ax, pts, xs, scale):
    """Each point at its x: the chip's marker and the pixel-spread bar in the combination's
    colour, filled for the anointed combination, open otherwise, small under LOW_PIXELS."""
    for p in pts:
        c = COMBO_COLOR[p["boards"]]
        small = curves.low_pixels(p)
        ax.errorbar([xs[(p["run"], p["board_idx"], p["combo"])]], [p["value_ps"]],
                    yerr=[p["bar_ps"]], color=c, marker=style.chip_marker(p["chip"]),
                    markersize=(curves.LOW_MARKER if small else 8.5) * scale,
                    markerfacecolor=c if p["anointed"] else style.SURFACE, markeredgecolor=c,
                    markeredgewidth=1.6 * scale, linestyle="none", capsize=2.0 * scale,
                    elinewidth=1.2 * scale, zorder=3)


def _tier(ax, texts, y_axes, size, linespacing=1.15):
    """Texts under the axis at axes-fraction height `y_axes` (top edge), each at its data x:
    [(x, text), ...]. Returns the text artists."""
    tr = ax.get_xaxis_transform()
    return [ax.text(x, y_axes, t, transform=tr, ha="center", va="top", fontsize=size,
                    color=style.INK, linespacing=linespacing) for x, t in texts]


def _bottom(artists, ax, renderer):
    """The lowest edge of `artists`, as an axes fraction of `ax`."""
    ab = ax.get_window_extent(renderer=renderer)
    y0 = min(a.get_window_extent(renderer=renderer).y0 for a in artists)
    return (y0 - ab.y0) / ab.height


def low_points(pts):
    """The points of `pts` under curves.LOW_PIXELS pixels, as the values file lists them."""
    return [{k: p[k] for k in ("tel", "run", "board_idx", "chip", "combo", "anointed",
                               "n_pixels")} for p in pts if curves.low_pixels(p)]


def combo_split(pts_by_tel, layouts, *, tel_titles, class_text, step_text, ylim, tag, data,
                footer, low_label=None, aw=7.0, ah=3.6, scale=SCALE):
    """re11: one row per telescope of `tel_titles` (top to bottom), one panel per board (left to
    right, board order), the points `pts_by_tel` (combo_points) at the x of `layouts`
    {tel: layout(...)}. Under each panel: tick labels "run" and this board's bias, then
    class_text(key) under each group (key = group_key), then step_text(step) under each step's
    groups. The legend (combinations, filled / open, and the small marker, labelled
    `low_label`, when a point is under LOW_PIXELS) stands outside the top right panel. Returns
    (figure, numbers)."""
    tels = list(tel_titles)
    s = style.sizes(scale)
    n_b = max(len(campaign.TELESCOPE_CHIPS[t]) for t in tels)
    # under each axis: two tick lines, two class lines, two step lines, two 0.03 gaps, the pads
    gap_y = (2 * 1.2 * TICK_SCALE * s["tick"] + 4 * 1.2 * s["ann"]) / 72.0 + 0.06 * ah + 0.3
    fig, axes = blnw._grid(len(tels), n_b, aw=aw, ah=ah, ml=1.0, mr=2.75, mt=0.62,
                           mb=0.25 + gap_y + curves._footer_in(footer, scale), gx=0.62,
                           gy=gap_y + 0.35, scale=scale)
    numbers, tiers = [], {}
    renderer = fig.canvas.get_renderer()
    for row, tel in zip(axes, tels):
        lay = layouts[tel]
        tp = pts_by_tel[tel]
        xs = combo_x(tp, lay["x"])
        for i, ax in enumerate(row):
            bp = [p for p in tp if p["board_idx"] == i]
            ax.set_xlim(0.0, lay["width"])
            ax.set_ylim(*ylim)
            runs = sorted(lay["x"], key=lambda r: lay["x"][r])
            hv = {p["run"]: p["hv"] for p in bp}
            ax.set_xticks([lay["x"][r] for r in runs])
            ax.set_xticklabels([u"%d\n%s" % (r, blnw.bias_text([hv[r]])) for r in runs])
            if i == 0:
                ax.set_ylabel(style.YLABEL_RES)
            style.style_axes(ax, scale)
            ax.tick_params(axis="x", which="minor", bottom=False, top=False)
            ax.tick_params(axis="x", which="major", labelsize=TICK_SCALE * s["tick"])
            for t in ax.get_xticklabels():
                t.set_linespacing(1.15)
            for k, (key, x0, x1, _) in enumerate(lay["groups"]):
                if k:
                    divider(ax, x0, solid=key[0] != lay["groups"][k - 1][0][0])
            _draw(ax, bp, xs, scale)
            fig.canvas.draw()
            y = _bottom(ax.get_xticklabels(), ax, renderer) - 0.03
            cls = _tier(ax, [(0.5 * (x0 + x1), class_text(key))
                             for key, x0, x1, _ in lay["groups"]], y, s["ann"])
            fig.canvas.draw()
            steps = []
            for key, x0, x1, _ in lay["groups"]:
                if steps and steps[-1][0] == key[0]:
                    steps[-1][2] = x1
                else:
                    steps.append([key[0], x0, x1])
            _tier(ax, [(0.5 * (x0 + x1), step_text(f)) for f, x0, x1 in steps],
                  _bottom(cls, ax, renderer) - 0.03, s["ann"])
            tiers["%s board %d" % (tel, i)] = dict(
                classes=[dict(text=class_text(key), x=0.5 * (x0 + x1))
                         for key, x0, x1, _ in lay["groups"]],
                steps=[dict(text=step_text(f), x=0.5 * (x0 + x1)) for f, x0, x1 in steps])
            numbers += [dict({k: p[k] for k in ("tel", "run", "step", "rfsel", "os", "ref",
                                                "preferred", "board_idx", "chip", "hv", "combo",
                                                "value_ps", "bar_ps", "n_pixels", "anointed")},
                             boards=list(p["boards"]),
                             x=xs[(p["run"], p["board_idx"], p["combo"])],
                             color=COMBO_COLOR[p["boards"]], small=curves.low_pixels(p))
                        for p in bp]
    every = [p for t in tels for p in pts_by_tel[t]]
    present = sorted({p["boards"] for p in every})
    handles = [Line2D([], [], color=COMBO_COLOR[b], marker="o", markersize=8.5 * scale,
                      markerfacecolor=COMBO_COLOR[b], linestyle="none", label=combo_text(b))
               for b in present]
    handles += [Line2D([], [], color=style.INK, marker="o", markersize=8.5 * scale,
                       markerfacecolor=style.INK, linestyle="none",
                       label=u"filled: anointed combination"),
                Line2D([], [], color=style.INK, marker="o", markersize=8.5 * scale,
                       markerfacecolor=style.SURFACE, markeredgewidth=1.6 * scale,
                       linestyle="none", label=u"open: other combinations")]
    low = low_points(every)
    if low:
        h = curves.low_pixel_handle(scale)
        if low_label:
            h.set_label(low_label)
        handles.append(h)
    lg = axes[0][-1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.04, 1.0),
                            borderaxespad=0.0, fontsize=s["legend"], handletextpad=0.4,
                            labelspacing=0.45)
    axes[0][-1].add_artist(lg)
    subjects = {t: [u"%s, board %d, %s" % (tel_titles[t], i, c)
                    for i, c in enumerate(campaign.TELESCOPE_CHIPS[t])] for t in tels}
    hs = style.compound_header(axes[0], tag=tag, data=data, scale=scale,
                               subjects=subjects[tels[0]])
    for row, tel in zip(axes[1:], tels[1:]):
        for ax, subj in zip(row, subjects[tel]):
            style.right_lines(ax, [subj], hs["title"])
    style.footer(fig, footer, scale)
    out = dict(points=numbers, colour="combination (COMBO_COLOR)", ylim_ps=list(ylim),
               layout={t: dict(x={str(r): x for r, x in layouts[t]["x"].items()},
                               groups=[dict(step=k[0], rfsel=k[1], os=k[2], x0=x0, x1=x1,
                                            runs=runs)
                                       for k, x0, x1, runs in layouts[t]["groups"]],
                               width=layouts[t]["width"]) for t in tels},
               tiers=tiers)
    if low:
        out["low_pixels"] = dict(threshold=curves.LOW_PIXELS, marker="small", points=low)
    return fig, out
