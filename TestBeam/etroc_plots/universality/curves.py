"""TWC curves: one pixel's time-walk correction against TOT in several runs, overlaid.

twc_curves draws a row of panels, each one pixel's correction delta(TOT) in several runs (the
curves of transfer.curve, mean-removed over the source run's TOT range), with the TOT
distribution of each run below it and the panel's runs and conditions written under the column.
Every word on the figure comes in as an argument; the notebook chooses the runs, the pixel, the
colours and the words.
"""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from .. import style
from .transfer import HIST_BIN_WIDTH_PS

PS_PER_NS = 1000.0
BAND_ALPHA = 0.6
SOURCE_FILL = "#b8b8b4"            # the source run's TOT distribution
LEGEND_LOCS = ("upper right", "lower left", "upper left", "lower right")
LW_SOURCE, LW_OTHER = 3.2, 2.2     # line widths at scale 0.7


def ramp_color(value, vmin, vmax, cmap_name):
    """A colour on a continuous ramp (angle, bias), kept off both ends of the colour map."""
    import matplotlib
    cmap = (matplotlib.colormaps[cmap_name] if hasattr(matplotlib, "colormaps")
            else matplotlib.cm.get_cmap(cmap_name))
    t = 0.5 if vmax <= vmin else 0.12 + 0.76 * (value - vmin) / (vmax - vmin)
    return cmap(t)


def _points_in(ax, legend, curves):
    """How many drawn curve points the legend box covers (display coordinates, 4 px pad)."""
    bb = legend.get_window_extent(renderer=ax.figure.canvas.get_renderer()).padded(4)
    n = 0
    for c in curves:
        xy = ax.transData.transform(np.column_stack([c["x"] / PS_PER_NS, c["y"]]))
        n += int(((xy[:, 0] >= bb.x0) & (xy[:, 0] <= bb.x1)
                  & (xy[:, 1] >= bb.y0) & (xy[:, 1] <= bb.y1)).sum())
    return n


def _legend(ax, handles, size, loc):
    lg = ax.legend(handles=handles, loc=loc, fontsize=size, frameon=True, framealpha=0.9,
                   borderpad=0.35, labelspacing=0.35, handlelength=1.8, handletextpad=0.5,
                   borderaxespad=0.4)
    return lg


def _legend_clear_of_curves(ax, handles, size, curves, max_raise=6):
    """Put the legend in the first corner of LEGEND_LOCS where it covers no curve point; when
    every corner covers some, raise the top of the y axis by a fifth of the data range and try
    again (at most `max_raise` times), then keep the place that covers the fewest points.
    Returns the number of points covered (0 when clear)."""
    lo, hi = ax.get_ylim()
    best = None
    for k in range(max_raise + 1):
        ax.set_ylim(lo, hi + 0.2 * k * (hi - lo))
        for loc in LEGEND_LOCS:
            lg = _legend(ax, handles, size, loc)
            ax.figure.canvas.draw()
            n = _points_in(ax, lg, curves)
            if n == 0:
                return 0
            if best is None or n < best[0]:
                best = (n, loc, k)
    n, loc, k = best
    ax.set_ylim(lo, hi + 0.2 * k * (hi - lo))
    _legend(ax, handles, size, loc)
    return n


def _xlim(panel):
    return (min(c["tot_min"] for c in panel["curves"]) / PS_PER_NS,
            max(c["tot_max"] for c in panel["curves"]) / PS_PER_NS)


def _lw(c, scale):
    return (LW_SOURCE if c["is_source"] else LW_OTHER) * scale / 0.7


def _draw_top(ax, panel, scale, band_label, legend_size):
    for c in sorted(panel["curves"], key=lambda c: c["is_source"]):
        ax.plot(c["x"] / PS_PER_NS, c["y"], color=c["color"], ls=c.get("ls", "-"),
                lw=_lw(c, scale), zorder=5 if c["is_source"] else 3, solid_capstyle="round")
    labels = panel.get("band_labels") or [band_label] * len(panel["bands"])
    bands = [(b[0], b[1], b[2], b[3] if len(b) > 3 else BAND_ALPHA) for b in panel["bands"]]
    for lo, hi, color, alpha in bands:
        ax.axvspan(lo / PS_PER_NS, hi / PS_PER_NS, color=color, alpha=alpha, zorder=1, lw=0)
    ax.set_xlim(*_xlim(panel))
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    style.style_axes(ax, scale)
    handles = [Line2D([], [], color=c["color"], lw=_lw(c, scale), ls=c.get("ls", "-"),
                      label=c["label"]) for c in panel["curves"]]
    handles += [Patch(facecolor=color, alpha=alpha, label=label)
                for (_, _, color, alpha), label in zip(bands, labels)]
    return _legend_clear_of_curves(ax, handles, legend_size, panel["curves"])


def _draw_bottom(ax, panel, scale):
    top = 0.0
    for c in panel["curves"]:
        counts = np.asarray(c["counts"], dtype=float)
        frac = counts / counts.sum() if counts.sum() > 0 else counts
        centers = (np.asarray(c["edges"][:-1]) + HIST_BIN_WIDTH_PS / 2.0) / PS_PER_NS
        top = max(top, float(frac.max()) if len(frac) else 0.0)
        if c["is_source"]:
            ax.fill_between(centers, 0, frac, step="mid", color=SOURCE_FILL, alpha=0.9,
                            edgecolor=style.INK_MUTED, linewidth=1.0, zorder=2)
        else:
            ax.step(centers, frac, where="mid", color=c["color"], ls=c.get("ls", "-"),
                    lw=1.8 * scale / 0.7, zorder=3)
    ax.set_xlim(*_xlim(panel))
    ax.set_ylim(0.0, 1.12 * (top or 1.0))
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    style.style_axes(ax, scale)


def twc_curves(panels, *, xlabel, ylabel_top, ylabel_bottom, band_label, line1, footer,
               tag=None, data=None, scale=0.7):
    """One column per panel: the curves on top, the TOT distributions below, the caption under.

    panels  [dict(subject, caption, curves, bands, band_labels=None), ...]
            subject      the panel's header line (axis, chip, pixel): one short line
            caption      the panel's runs and conditions, lines joined by "\\n", written under
                         the column
            curves       [dict(label, color, ls, is_source, x, y, tot_min, tot_max, edges,
                         counts), ...] (transfer.curve plus the words and the look), TOT in ps;
                         the source is drawn thicker on top and filled grey below
            bands        [(lo, hi, colour[, alpha]), ...] TOT ranges in ps shaded on top (the
                         source's 5-95 % TOT range), labelled band_labels or `band_label`;
                         alpha BAND_ALPHA unless given
    xlabel, ylabel_top, ylabel_bottom  the axis titles (the y titles on the first column only)
    line1, tag, data   the header's facility line (style.compound_header). The campaign's second
                       header line is left out: line1 names every campaign drawn.
    footer             the footer text.
    Returns (figure, top axes, bottom axes, legend points covered per panel: 0 when clear)."""
    s = style.sizes(scale)
    style.apply_style(scale)
    import matplotlib.pyplot as plt
    n = len(panels)
    line_in = 1.25 * s["ann"] / 72.0
    cap_lines = max(p["caption"].count("\n") + 1 for p in panels)
    foot_in = (footer.count("\n") + 1) * 1.2 * s["ann"] / 72.0
    aw, ah_top, ah_bot = 4.6, 4.7, 3.2
    ml, mr, mt, gx, gy = 1.3, 0.35, 0.62, 0.8, 1.0
    xlab_in = 0.62                       # tick labels and the x title under a panel
    mb = xlab_in + cap_lines * line_in + 0.15 + max(0.0, foot_in + 0.1 - style.FOOTER_GAP_IN)
    W = ml + n * aw + (n - 1) * gx + mr
    H = mb + ah_bot + gy + ah_top + mt
    fig = plt.figure(figsize=(W, H))
    top_axes, bot_axes, covered = [], [], []
    for j, panel in enumerate(panels):
        x0 = ml + j * (aw + gx)
        ax_t = fig.add_axes([x0 / W, (mb + ah_bot + gy) / H, aw / W, ah_top / H])
        ax_b = fig.add_axes([x0 / W, mb / H, aw / W, ah_bot / H])
        _draw_bottom(ax_b, panel, scale)
        ax_t.set_xlabel(xlabel)
        ax_b.set_xlabel(xlabel)
        if j == 0:
            ax_t.set_ylabel(ylabel_top)
            ax_b.set_ylabel(ylabel_bottom)
        covered.append(_draw_top(ax_t, panel, scale, band_label, s["legend"] * 0.85))
        fig.text(x0 / W, (mb - xlab_in) / H, panel["caption"], ha="left", va="top",
                 fontsize=s["ann"], color=style.INK, linespacing=1.25)
        top_axes.append(ax_t)
        bot_axes.append(ax_b)
    style.compound_header(top_axes, tag=tag, data=data, line1=line1, line2="", scale=scale,
                          subjects=[p["subject"] for p in panels])
    style.footer(fig, footer, scale)
    return fig, top_axes, bot_axes, covered
