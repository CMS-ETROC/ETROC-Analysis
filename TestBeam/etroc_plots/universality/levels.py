"""Pair-penalty figures: where a stored time-walk correction transfers and where it does not.

Every number drawn here is a PAIR penalty of the TWC-transfer study (transfer.py describes the
study): for one comparison (a cell) and one pair of boards of a target triple, the width of that
pair with the source's correction minus its width with the target's own correction fitted on the
other half of the events (the baseline), on the same triples and events, in ps; with its error
se. A row is `interp` when the target's TOT lies within the source's (coverage at least 0.98 and
lever at most 5 ps), else `extrap`. The margin of a row is max(1 ps, 2 % of the baseline width),
read from the tables; the 90 % interval is mu +- Z90 se. No figure here computes a verdict: the
verdicts drawn are the study's (transfer.read_master applies its override).

un02_coverage_lever, un03_penalty_per_cell, un04_linear_refit, un05_settings_steps and
un06_controls draw one figure each from rows the notebook selects with the data helpers below;
every word on a figure comes in as an argument. Each returns (figure, axes, drawn, covered):
`drawn` holds what was put on the figure beyond the input rows (positions, limits), `covered` the
number of data points the legends cover (0 when clear), for the values file.
"""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from .. import style
from .transfer import family_rows, primary_rows

Z90 = 1.6448536269514722          # the two-sided 90 % interval of a normal: mu +- Z90 se
VERDICT_COLOR = {"PASS": "#1a9641", "FAIL": "#d62728", "NA-UNGUARDED": "#8c5a2b"}
VERDICT_MARKER = {"PASS": "o", "FAIL": "s", "NA-UNGUARDED": "D"}
# one colour per comparison axis of the study (A-E, I-K), apart from each other
AXIS_COLOR = {"A": "#2a78d6", "B": "#1baf7a", "C": "#eda100", "D": "#c0392b",
              "E": "#8e44ad", "I": "#e377c2", "J": "#7f6000", "K": "#34495e"}
REFIT_COLOR = {"xfer": style.INK_MUTED, "xfer_lin": style.ALERT}
MARGIN_COLOR = style.GRID
MARGIN_ALPHA = 0.85
LEGEND_LOCS = ("upper right", "upper left", "lower right", "lower left")

# What the numbers of these figures are, for the values files; the notebook adds per figure.
CONVENTIONS = dict(
    penalty="paired difference of PAIR widths on the same target triples and events, in ps: the "
            "pair width with the transferred correction (arm xfer, or xfer_lin) minus the width "
            "with the target's own correction fitted on the other half of its events (self_cf)",
    pair="a pair = two boards of a target triple (one pixel per board, the tracks through all "
         "three); role words (ref, trig, dut, extra) are the study's bookkeeping keys of the "
         "boards in the target run's combination",
    families="interp: target TOT within the source's (coverage C >= 0.98 of the target's TOT "
             "inside the source's 0.5-99.5 % range, and lever <= 5 ps, the rms change of the "
             "correction when the target TOT is clipped to that range); extrap: otherwise",
    margin="TOST equivalence margin max(1 ps, 2 % of the baseline width), from the tables",
    interval="mu +- Z90 se, Z90 = %.16g (two-sided 90 %%)" % Z90,
    verdicts="the study's verdicts as read by transfer.read_master (override: an unguarded "
             "cell that is not a FAIL reads NA-UNGUARDED; FAIL+UNGUARDED reads FAIL)",
    rows="the primary error factor of each campaign (f_variant_is_primary), pair rows, arm xfer "
         "unless stated",
)


# ------------------------------------------------------------------------------- data helpers
def coverage_lever_rows(axis_r2):
    """(extrap, interp): the primary pair rows of the transferred correction of the comparison
    axes, extrap rows with a coverage, a lever and a penalty, and every interp row."""
    rows = primary_rows(axis_r2)
    extrap = family_rows(rows, "extrap").dropna(subset=["cov_C", "lever_X", "mu_ps"])
    return extrap, family_rows(rows, "interp")


def interp_band(interp, lo=16, hi=84):
    """The median and the lo-hi percentiles of the penalties of the interp rows (rows without a
    penalty left out): dict(median, lo, hi, n_rows, n_values)."""
    v = interp.mu_ps.dropna().values
    p_lo, p_hi = np.percentile(v, [lo, hi])
    return dict(median=float(np.median(v)), lo=float(p_lo), hi=float(p_hi),
                percentiles=[lo, hi], n_rows=int(len(interp)), n_values=int(len(v)))


def refit_rows(axis_f):
    """The pair rows of axis F with both arms present (the transferred correction as is, xfer,
    and with its linear term refitted on the target, xfer_lin), interp rows first, then by cell
    number and pair; with fnum ("F3") and the cell number fn."""
    rows = axis_f[(axis_f.kind == "pair") & axis_f.family.isin(["interp", "extrap"])
                  & (axis_f.xfer_present == True)                  # noqa: E712
                  & (axis_f.xfer_lin_present == True)].copy()      # noqa: E712
    rows["fnum"] = rows.cell.str.extract(r"^(F\d+)_", expand=False)
    rows["fn"] = rows.fnum.str[1:].astype(int)
    return rows.sort_values(["family", "fn", "board_or_role"],
                            ascending=[False, True, True]).reset_index(drop=True)


def family_bars(rows, cells):
    """Per cell of `cells` (in order) and family (interp, then extrap): the pixel-weighted mean
    penalty of its rows, mu = sum(n mu) / sum(n), and se = sqrt(sum(n se^2) / sum(n)), with n
    the row's n_pix. A family without rows gives no bar."""
    out = []
    for cell in cells:
        for fam in ("interp", "extrap"):
            g = rows[(rows.cell == cell) & (rows.family == fam)]
            if g.empty:
                continue
            w = g.n_pix.values.astype(float)
            out.append(dict(cell=cell, family=fam, n_rows=int(len(g)), n_pix=int(w.sum()),
                            mu_ps=float(np.average(g.mu_ps.values, weights=w)),
                            se_ps=float(np.sqrt(np.average(g.se_ps.values ** 2, weights=w)))))
    return out


def spread_offsets(groups, step):
    """Offsets that spread points sharing an x: within each group (a list of keys, in drawing
    order) the k-th of n points moves by (k - (n - 1) / 2) step. Returns {key: offset}."""
    out = {}
    for keys in groups:
        n = len(keys)
        for k, key in enumerate(keys):
            out[key] = (k - (n - 1) / 2.0) * step
    return out


# ------------------------------------------------------------------------------ drawing helpers
def _new_figure(placed, scale):
    """A figure with axes at their sizes in inches (positions set again by _layout):
    placed = [(x, y, w, h)] from the lower left of the axes block."""
    import matplotlib.pyplot as plt
    style.apply_style(scale)
    cw = max(x + w for x, y, w, h in placed)
    ch = max(y + h for x, y, w, h in placed)
    ml, mb, mr, mt = 2.0, 1.5, 0.5, 1.0
    W, H = ml + cw + mr, mb + ch + mt
    fig = plt.figure(figsize=(W, H))
    axes = [fig.add_axes([(ml + x) / W, (mb + y) / H, w / W, h / H]) for x, y, w, h in placed]
    fig._etroc_block = (placed, cw, ch, (ml, mr, mb, mt))
    return fig, axes


def _layout(fig, pad=0.15, top_blank=0.16):
    """Size the figure around its axes block, from the drawn ink: tick labels, axis titles,
    legends outside the axes and the header lines get `pad` inches to the left, right and bottom
    edges and `top_blank` inches above; the bottom margin also takes a footer taller than the
    band style.lower_footer adds. The axes keep their sizes in inches."""
    placed, cw, ch, (ml, mr, mb, mt) = fig._etroc_block
    axes = list(fig.axes)
    feet = [t for t in fig.texts if getattr(t, "_etroc_footer", False)]
    for _ in range(4):
        W, H = ml + cw + mr, mb + ch + mt
        fig.set_size_inches(W, H, forward=False)
        for ax, (x, y, w, h) in zip(axes, placed):
            ax.set_position([(ml + x) / W, (mb + y) / H, w / W, h / H])
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        dpi = fig.dpi
        boxes = [ax.get_tightbbox(r) for ax in axes]
        x0 = min(b.x0 for b in boxes) / dpi
        x1 = max(b.x1 for b in boxes) / dpi
        y0 = min(b.y0 for b in boxes) / dpi
        y1 = max(b.y1 for b in boxes) / dpi
        foot = max([t.get_window_extent(renderer=r).height / dpi for t in feet] + [0.0])
        new = (ml - x0 + pad, mr - (W - x1) + pad,
               mb - y0 + pad + max(0.0, foot + 0.1 - style.FOOTER_GAP_IN),
               mt - (H - y1) + top_blank)
        done = max(abs(a - b) for a, b in zip(new, (ml, mr, mb, mt))) < 0.003
        ml, mr, mb, mt = new
        if done:
            break
    W, H = ml + cw + mr, mb + ch + mt
    fig.set_size_inches(W, H, forward=False)
    for ax, (x, y, w, h) in zip(axes, placed):
        ax.set_position([(ml + x) / W, (mb + y) / H, w / W, h / H])
    return W, H


def _fit_footer(fig, pad=0.15):
    """Break the footer's lines again at spaces where one runs past the figure's right edge less
    `pad` inches (the line breaks it already has are kept; a $...$ span is never split), then
    lay the figure out again for the taller footer."""
    feet = [t for t in fig.texts if getattr(t, "_etroc_footer", False)]
    r = fig.canvas.get_renderer()
    W = fig.get_size_inches()[0]
    changed = False
    for t in feet:
        text = t.get_text()
        avail = W - pad - t.get_window_extent(renderer=r).x0 / fig.dpi
        out = []
        for line in text.split("\n"):
            words, cur = [], ""
            for w in line.split(" "):              # keep $...$ spans whole
                cur = (cur + " " + w) if cur else w
                if cur.count("$") % 2 == 0:
                    words.append(cur)
                    cur = ""
            if cur:
                words.append(cur)
            cur = ""
            for w in words:
                trial = (cur + " " + w) if cur else w
                t.set_text(trial)
                if cur and t.get_window_extent(renderer=r).width / fig.dpi > avail:
                    out.append(cur)
                    cur = w
                else:
                    cur = trial
            out.append(cur)
        t.set_text("\n".join(out))
        changed |= t.get_text() != text
    if changed:
        _layout(fig)


def _covered(ax, box, points):
    """How many of `points` ([(x, y) arrays] in data coordinates) fall in display box `box`."""
    n = 0
    for xy in points:
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if not len(xy):
            continue
        d = ax.transData.transform(xy)
        n += int(((d[:, 0] >= box.x0) & (d[:, 0] <= box.x1)
                  & (d[:, 1] >= box.y0) & (d[:, 1] <= box.y1)).sum())
    return n


def place_legend(ax, handles, points, size, locs=LEGEND_LOCS, **kw):
    """The legend in the first of `locs` where its box (padded 4 px) covers none of `points`,
    else where it covers the fewest. Returns the number of points covered (0 when clear)."""
    kw = dict(dict(frameon=True, framealpha=0.92, borderpad=0.4, labelspacing=0.4,
                   handletextpad=0.5, borderaxespad=0.5), **kw)
    best = None
    for loc in locs:
        lg = ax.legend(handles=handles, loc=loc, fontsize=size, **kw)
        ax.figure.canvas.draw()
        box = lg.get_window_extent(renderer=ax.figure.canvas.get_renderer()).padded(4)
        n = _covered(ax, box, points)
        if n == 0:
            return 0
        if best is None or n < best[0]:
            best = (n, loc)
    ax.legend(handles=handles, loc=best[1], fontsize=size, **kw)
    return best[0]


def trim_headroom(ax, pad=0.05):
    """With the legend in the upper half of the axes, lower the axis top to the least that keeps
    everything drawn (the axes' data limits) `pad` of the axes height below the legend's box."""
    lg = ax.get_legend()
    if lg is None:
        return
    box = lg.get_window_extent(renderer=ax.figure.canvas.get_renderer())
    y0 = ax.transAxes.inverted().transform((box.x0, box.y0))[1]
    if y0 < 0.5:
        return
    lo, hi = ax.get_ylim()
    top = lo + (ax.dataLim.y1 - lo) / (y0 - pad)
    if top < hi:
        ax.set_ylim(lo, top)


def _segment(lo, hi, y, n=9, horizontal=True):
    """n points along an interval drawn at y (or at x = y when vertical), for the legend count."""
    t = np.linspace(lo, hi, n)
    return np.column_stack([t, np.full(n, y)] if horizontal else [np.full(n, y), t])


def _symlog_ticks(lo, hi, candidates):
    return [t for t in candidates if lo <= t <= hi]


def _rows_axis(ax, labels, bold, scale, size):
    """y ticks at 0..n-1, labelled `labels`, top to bottom; the `bold` ones are group headers
    with no tick mark and no grid line."""
    n = len(labels)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels, fontsize=size)
    ax.set_ylim(n - 0.4, -0.8)
    for t, b in zip(ax.get_yticklabels(), bold):
        if b:
            t.set_fontweight("bold")
    for tk, b in zip(ax.yaxis.get_major_ticks(), bold):
        if b:
            tk.tick1line.set_visible(False)
            tk.tick2line.set_visible(False)
            tk.gridline.set_visible(False)


def _margin_bar(ax, at, margin, half, horizontal=True):
    """The row's own margin, +-margin, as a grey bar behind the point (nothing when the row has
    no margin)."""
    if margin is None or not np.isfinite(margin):
        return
    if horizontal:
        ax.add_patch(Rectangle((-margin, at - half), 2 * margin, 2 * half, color=MARGIN_COLOR,
                               alpha=MARGIN_ALPHA, zorder=0, lw=0))
    else:
        ax.add_patch(Rectangle((at - half, -margin), 2 * half, 2 * margin, color=MARGIN_COLOR,
                               alpha=MARGIN_ALPHA, zorder=0, lw=0))


def _verdict(v):
    v = str(v).upper()
    if v not in VERDICT_COLOR:
        raise ValueError("verdict %r has no colour: %s" % (v, sorted(VERDICT_COLOR)))
    return VERDICT_COLOR[v], VERDICT_MARKER[v]


def verdict_handles(words, ms):
    """Legend handles of the verdicts, in the order of `words` ({verdict: label})."""
    return [Line2D([], [], color=VERDICT_COLOR[v], marker=VERDICT_MARKER[v], ls="none", ms=ms,
                   label=label) for v, label in words.items()]


def margin_handle(label):
    """Legend swatch of the grey margin bar."""
    return Patch(facecolor=MARGIN_COLOR, alpha=MARGIN_ALPHA, edgecolor="none", label=label)


# ------------------------------------------------------------------------------------- un02
def un02_coverage_lever(extrap, band, *, axis_words, xlabels, ylabel, legend_title, band_label,
                        median_label, line1, subjects, footer, axis_colors=AXIS_COLOR,
                        axis_markers=None, xscales=("linear", "log"), scale=0.85):
    """Two panels side by side: the pair penalty of every extrap row against 1 - coverage (left)
    and against the lever (right), one colour per comparison axis, marker area growing with the
    row's n_pix; behind, the median (dashed) and the band of the interp rows' penalties.

    extrap        rows with axis, n_pix, cov_C, lever_X, mu_ps (coverage_lever_rows)
    band          interp_band() of the interp rows
    axis_words    {axis letter: legend label}, legend order; every axis of `extrap` needs one
    xlabels       (left, right) x titles; ylabel the y title (left panel)
    legend_title, band_label, median_label   legend words; median_label may hold one %-format
                  for the median in ps
    line1, subjects, footer   the facility line, the two panel subjects, the footer
    axis_markers  {letter: marker} for axes drawn apart (default "o")
    xscales       the two x scales
    """
    axis_markers = axis_markers or {}
    missing = sorted(set(extrap.axis) - set(axis_words))
    if missing:
        raise ValueError("no legend words for axes %s" % missing)
    s = style.sizes(scale)
    aw, ah, gap = 8.2, 6.4, 1.15
    fig, axes = _new_figure([(0, 0, aw, ah), (aw + gap, 0, aw, ah)], scale)
    k = (scale / 0.85) ** 2

    def area(n):
        return k * (16 + 40 * np.sqrt(np.clip(np.asarray(n, dtype=float), 1, None) / 300.0))

    xs = [1.0 - extrap.cov_C, extrap.lever_X]
    for ax, x, xl, xsc in zip(axes, xs, xlabels, xscales):
        ax.axhspan(band["lo"], band["hi"], color=MARGIN_COLOR, alpha=0.5, zorder=1, lw=0)
        ax.axhline(band["median"], color=style.INK_MUTED, lw=1.4, ls="--", zorder=2)
        for letter in axis_words:
            g = extrap[extrap.axis == letter]
            if g.empty:
                continue
            ax.scatter(x[g.index], g.mu_ps, s=area(g.n_pix), c=axis_colors[letter],
                       marker=axis_markers.get(letter, "o"), alpha=0.8, edgecolors="white",
                       linewidths=0.5, zorder=3)
        ax.set_xscale(xsc)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel(xl)
        style.style_axes(ax, scale)
    axes[0].set_ylabel(ylabel)
    lo = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    hi = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    yt = _symlog_ticks(lo, hi, [-100, -10, -1, 0, 1, 10, 100])
    for ax in axes:
        ax.set_ylim(lo, hi)
        ax.set_yticks(yt)
        ax.set_yticklabels(["%g" % t for t in yt])
    present = [a for a in axis_words if (extrap.axis == a).any()]
    ms = 9 * scale / 0.85
    handles = [Line2D([], [], marker=axis_markers.get(a, "o"), color="none",
                      markerfacecolor=axis_colors[a], markeredgecolor="white", markersize=ms,
                      label=axis_words[a]) for a in present]
    handles += [Patch(facecolor=MARGIN_COLOR, alpha=0.5, label=band_label),
                Line2D([], [], color=style.INK_MUTED, ls="--", lw=1.4,
                       label=(median_label % band["median"]) if "%" in median_label
                       else median_label)]
    axes[1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.03, 1.0),
                   borderaxespad=0.0, fontsize=s["legend"] * 0.8, title=legend_title,
                   title_fontsize=s["legend_title"] * 0.85, frameon=False)
    style.compound_header(axes, line1=line1, line2="", scale=scale, subjects=subjects)
    style.footer(fig, footer, scale)
    _layout(fig)
    _fit_footer(fig)
    box = axes[1].get_legend().get_window_extent(renderer=fig.canvas.get_renderer())
    covered = sum(_covered(ax, box, [np.column_stack([x, extrap.mu_ps])])
                  for ax, x in zip(axes, xs))
    drawn = dict(x_left="1 - cov_C", x_right="lever_X", xscales=list(xscales),
                 marker_area_pt2="%.4g * (16 + 40 sqrt(max(n_pix, 1) / 300))" % k,
                 ylim=[float(lo), float(hi)], axes_drawn=present)
    return fig, axes, drawn, covered


# ------------------------------------------------------------------------------------- un03
def un03_penalty_per_cell(master, *, labels, level_titles, subjects, xlabel, verdict_words,
                          margin_word, no_rows_text, share_fmt, line1, footer, scale=0.8):
    """A forest of the level cells, grouped by level (a header row each): left, the interp
    family (mu_ivw, se_ivw, margin, verdict_pair_interp, and the share of interp differences at
    the right edge); right, all target triples (all_ivw_mu, all_ivw_se, all_margin,
    all_verdict). Verdict sets marker and colour; each cell's own margin is a grey bar; a cell
    with no interp row reads `no_rows_text` in the left panel.

    master        transfer.read_master() rows, in level order
    labels        {cell: row label}; level_titles {level: header row}
    subjects      the two panel subjects; xlabel the x title of both panels
    verdict_words {verdict: legend label}; margin_word the margin's legend label
    share_fmt     the share's text, one %-format of interp_pct (e.g. "%.0f %% interp.")
    """
    s = style.sizes(scale)
    tick_size = s["ann"] * 0.9
    lab, bold, cell_y = [], [], {}
    seps = []
    for lev in master.level.drop_duplicates():
        if lab:
            seps.append(len(lab) - 0.5)
        lab.append(level_titles[lev])
        bold.append(True)
        for cell in master[master.level == lev].cell:
            cell_y[cell] = len(lab)
            lab.append(labels[cell])
            bold.append(False)
    pitch = 0.36
    aw, gap = 7.4, 0.35
    ah = pitch * (len(lab) + 0.4)
    fig, axes = _new_figure([(0, 0, aw, ah), (aw + gap, 0, aw, ah)], scale)
    specs = [("mu_ivw", "se_ivw", "margin", "verdict_pair_interp"),
             ("all_ivw_mu", "all_ivw_se", "all_margin", "all_verdict")]
    xlims = []
    for i, (m, e, g, _) in enumerate(specs):     # each panel its own range; the left one keeps
        lo = min((master[m] - Z90 * master[e]).min(), -master[g].max())   # room for the share
        hi = (master[m] + Z90 * master[e]).max()
        xlims.append((min(-6.0, 1.3 * lo), max(6.0, (5.0 if i == 0 else 2.2) * hi)))
    points = [[], []]
    ms = 8 * scale / 0.8
    for i, (ax, (mc, sc_, gc, vc)) in enumerate(zip(axes, specs)):
        for _, r in master.iterrows():
            y = cell_y[r.cell]
            _margin_bar(ax, y, r[gc], 0.32)
            if np.isfinite(r[gc]):
                points[i].append(_segment(-r[gc], r[gc], y, n=5))
            mu, se = r[mc], r[sc_]
            if not np.isfinite(mu):
                ax.annotate(no_rows_text, (0.0, y), xycoords="data", xytext=(4, 0),
                            textcoords="offset points", fontsize=s["ann"] * 0.78,
                            color=style.INK_MUTED, ha="left", va="center")
                continue
            col, mk = _verdict(r[vc])
            ax.errorbar(mu, y, xerr=Z90 * se, fmt=mk, ms=ms, color=col, ecolor=col,
                        elinewidth=2.0 * scale / 0.8, capsize=4 * scale / 0.8, zorder=5)
            points[i].append(_segment(mu - Z90 * se, mu + Z90 * se, y))
            if i == 0 and np.isfinite(r.interp_pct):
                ax.annotate(share_fmt % r.interp_pct, (0.985, y),
                            xycoords=("axes fraction", "data"), fontsize=s["ann"] * 0.7,
                            color=style.INK_MUTED, va="center", ha="right")
        ax.axvline(0, color=style.INK, lw=1.2, zorder=1)
        for y in seps:
            ax.axhline(y, color=style.GRID, lw=1.0, ls=":")
        ax.set_xscale("symlog", linthresh=2.0, linscale=1.1)
        ax.set_xlim(*xlims[i])
        ticks = _symlog_ticks(xlims[i][0], xlims[i][1], [-100, -10, -2, 0, 2, 10, 100, 1000])
        ax.set_xticks(ticks)
        ax.set_xticklabels(["%g" % t for t in ticks])
        ax.set_xlabel(xlabel)
        style.style_axes(ax, scale, minor=False)
        _rows_axis(ax, lab, bold, scale, tick_size)
        ax.grid(False, axis="y")
    axes[1].set_yticklabels([])
    handles = verdict_handles(verdict_words, ms) + [margin_handle(margin_word)]
    covered = place_legend(axes[1], handles, points[1], s["ann"] * 0.85,
                           locs=("lower right", "upper right", "center right"))
    style.compound_header(axes, line1=line1, line2="", scale=scale, subjects=subjects)
    style.footer(fig, footer, scale)
    _layout(fig)
    _fit_footer(fig)
    drawn = dict(xlim=[[float(v) for v in xl] for xl in xlims], x_scale="symlog, linthresh 2 ps",
                 row_y={c: int(y) for c, y in cell_y.items()})
    return fig, axes, drawn, covered


# ------------------------------------------------------------------------------------- un04
def un04_linear_refit(rows, *, row_labels, family_titles, xlabel, legend_words, line1, subject,
                      footer, scale=0.72):
    """One row per pair row of refit_rows(): the penalty with the source's correction as is
    (xfer, grey) and with its linear term refitted on the target (xfer_lin, red), joined by a
    line, each with its 90 % interval; filled markers for interp rows, open for extrap; the
    row's margin (margin_ps) as a grey bar. Rows grouped by family under a header row each.

    row_labels     one label per row of `rows`, in order
    family_titles  {"interp": header, "extrap": header}
    legend_words   dict(xfer, xfer_lin, open, margin): the legend labels
    """
    s = style.sizes(scale)
    lab, bold, ys = [], [], []
    for fam in ("interp", "extrap"):
        idx = np.flatnonzero((rows.family == fam).values)
        if not len(idx):
            continue
        lab.append(family_titles[fam])
        bold.append(True)
        for i in idx:
            ys.append(len(lab))
            lab.append(row_labels[i])
            bold.append(False)
    ys = np.array(ys)
    pitch = 0.36
    aw = 9.0
    ah = pitch * (len(lab) + 0.4)
    fig, (ax,) = _new_figure([(0, 0, aw, ah)], scale)
    lo = min((rows.xfer_mu_ps - Z90 * rows.xfer_se_ps).min(),
             (rows.xfer_lin_mu_ps - Z90 * rows.xfer_lin_se_ps).min(), -rows.margin_ps.max())
    hi = max((rows.xfer_mu_ps + Z90 * rows.xfer_se_ps).max(),
             (rows.xfer_lin_mu_ps + Z90 * rows.xfer_lin_se_ps).max())
    xlim = (min(-3.0, 1.3 * lo), 4.0 * hi)
    ms = 8 * scale / 0.72
    lw = 1.3 * scale / 0.72
    points = []
    for y, (_, r) in zip(ys, rows.iterrows()):
        _margin_bar(ax, y, r.margin_ps, 0.3)
        points.append(_segment(-r.margin_ps, r.margin_ps, y, n=5))
        filled = r.family == "interp"
        ax.plot([r.xfer_mu_ps, r.xfer_lin_mu_ps], [y, y], color=style.INK_MUTED, lw=lw, zorder=2)
        for arm in ("xfer", "xfer_lin"):
            mu, se, col = r[arm + "_mu_ps"], r[arm + "_se_ps"], REFIT_COLOR[arm]
            ax.errorbar(mu, y, xerr=Z90 * se, fmt="none", ecolor=col, elinewidth=lw,
                        capsize=2.5 * scale / 0.72, zorder=3)
            ax.plot([mu], [y], color=col, marker="o", ms=ms, mfc=col if filled else "white",
                    mew=1.6 * scale / 0.72, ls="none", zorder=5 if arm == "xfer_lin" else 4)
            points.append(_segment(mu - Z90 * se, mu + Z90 * se, y))
    ax.axvline(0, color=style.INK, lw=1.2, zorder=1)
    ax.set_xscale("symlog", linthresh=1.0, linscale=1.1)
    ax.set_xlim(*xlim)
    ticks = _symlog_ticks(xlim[0], xlim[1], [-10, -2, -1, 0, 1, 2, 10, 50, 100])
    ax.set_xticks(ticks)
    ax.set_xticklabels(["%g" % t for t in ticks])
    ax.set_xlabel(xlabel)
    style.style_axes(ax, scale, minor=False)
    _rows_axis(ax, lab, bold, scale, s["ann"] * 0.82)
    ax.grid(False, axis="y")
    if bold.count(True) == 2:
        ax.axhline(bold.index(True, 1) - 0.5, color=style.INK, lw=1.0, ls=":")
    hms = 9 * scale / 0.72
    handles = [Line2D([], [], color=REFIT_COLOR["xfer"], marker="o", mfc=REFIT_COLOR["xfer"],
                      ms=hms, ls="none", label=legend_words["xfer"]),
               Line2D([], [], color=REFIT_COLOR["xfer_lin"], marker="o",
                      mfc=REFIT_COLOR["xfer_lin"], ms=hms, ls="none",
                      label=legend_words["xfer_lin"]),
               Line2D([], [], color=style.INK_MUTED, marker="o", mfc="white", ms=hms, mew=1.6,
                      ls="none", label=legend_words["open"]),
               margin_handle(legend_words["margin"])]
    covered = place_legend(ax, handles, points, s["ann"] * 0.9,
                           locs=("lower right", "upper right", "center right"))
    style.header(ax, name=subject, line1=line1, line2="", scale=scale)
    style.footer(fig, footer, scale)
    _layout(fig)
    _fit_footer(fig)
    drawn = dict(xlim=[float(v) for v in xlim], x_scale="symlog, linthresh 1 ps",
                 row_y=[int(y) for y in ys])
    return fig, [ax], drawn, covered


# ------------------------------------------------------------------------------------- un05
def un05_settings_steps(bars, *, cell_labels, cell_colors, legend, family_words, ylabel, line1,
                        subject, footer, scale=0.8):
    """Bars per cell and family (family_bars): the pixel-weighted mean penalty with its 90 %
    interval, solid for interp, hatched for extrap, coloured per cell.

    cell_labels   {cell: x tick label}; cell_colors {cell: colour}
    legend        [(colour, label), ...] the colour classes
    family_words  dict(interp, extrap): legend labels of solid and hatched
    """
    s = style.sizes(scale)
    cells = list(dict.fromkeys(b["cell"] for b in bars))
    n_bars = len(bars)
    aw, ah = max(6.5, 1.0 * n_bars + 1.2 * len(cells)), 5.6
    fig, (ax,) = _new_figure([(0, 0, aw, ah)], scale)
    x, xt, pos = 0.0, [], []
    points = []
    for cell in cells:
        first = x
        for b in bars:
            if b["cell"] != cell:
                continue
            hatch = "" if b["family"] == "interp" else "///"
            ax.bar(x, b["mu_ps"], width=0.8, color=cell_colors[cell], hatch=hatch,
                   edgecolor=style.INK, linewidth=0.6, zorder=3)
            ax.errorbar(x, b["mu_ps"], yerr=Z90 * b["se_ps"], fmt="none", ecolor=style.INK,
                        elinewidth=1.2, capsize=3 * scale / 0.8, zorder=4)
            pos.append(dict(cell=cell, family=b["family"], x=x))
            lo, hi = sorted([0.0, b["mu_ps"]])
            for xx in (x - 0.4, x, x + 0.4):
                points.append(_segment(lo, hi, xx, n=7, horizontal=False))
            points.append(_segment(b["mu_ps"] - Z90 * b["se_ps"], b["mu_ps"] + Z90 * b["se_ps"],
                                   x, n=7, horizontal=False))
            x += 1.0
        xt.append((first + x - 1.0) / 2.0)
        x += 0.8
    ax.axhline(0, color=style.INK, lw=1.0, zorder=2)
    ax.set_xticks(xt)
    ax.set_xticklabels([cell_labels[c] for c in cells], fontsize=s["ann"] * 0.9)
    ax.set_xlim(-0.9, x - 0.9)
    ax.set_ylabel(ylabel)
    style.style_axes(ax, scale, minor=False)
    ax.grid(False, axis="x")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.15 * (hi - lo), hi + 0.6 * (hi - lo))
    ms = 11 * scale / 0.8
    handles = [Line2D([], [], marker="s", color="none", markerfacecolor=c,
                      markeredgecolor=style.INK,
                      markersize=ms, label=l) for c, l in legend]
    handles += [Patch(facecolor="white", edgecolor=style.INK, label=family_words["interp"]),
                Patch(facecolor="white", edgecolor=style.INK, hatch="///",
                      label=family_words["extrap"])]
    covered = place_legend(ax, handles, points, s["legend"] * 0.8,
                           locs=("upper left", "upper right", "lower left", "lower right"))
    trim_headroom(ax)
    style.header(ax, name=subject, line1=line1, line2="", scale=scale)
    style.footer(fig, footer, scale)
    _layout(fig)
    _fit_footer(fig)
    drawn = dict(bar_x=pos, ylim=[float(v) for v in ax.get_ylim()])
    return fig, [ax], drawn, covered


# ------------------------------------------------------------------------------------- un06
def un06_controls(panels, *, ylabel, line1, footer, scale=0.78):
    """Up to three panels of level cells, each cell mu +- Z90 se with its own margin as a grey
    box: categorical panels (one x slot per cell, its label under it) and one panel against a
    number (x given per cell).

    panels  [dict(subject, rows, xlabel=None, legend=None, width), ...], rows = [dict(cell,
            mu, se, margin, color, marker, x=None, label=None), ...]: a panel whose rows carry x
            is drawn against it (xlabel under it; boxes `box_half` wide in x units); else the
            rows sit at 0, 1, ... under their labels, turned by `rotation` degrees (35; 0 for
            labels broken over lines). legend = [(colour, marker, label), ...].
    """
    s = style.sizes(scale)
    ah, gap = 5.8, 0.75
    placed, x0 = [], 0.0
    for p in panels:
        placed.append((x0, 0, p["width"], ah))
        x0 += p["width"] + gap
    fig, axes = _new_figure(placed, scale)
    ms = 10 * scale / 0.78
    covered, drawn = 0, []
    for ax, p in zip(axes, panels):
        rows = p["rows"]
        numeric = all(r.get("x") is not None for r in rows)
        xs = [r["x"] for r in rows] if numeric else list(range(len(rows)))
        half = p.get("box_half", 0.3)
        points = []
        for xx, r in zip(xs, rows):
            _margin_bar(ax, xx, r["margin"], half, horizontal=False)
            ax.errorbar(xx, r["mu"], yerr=Z90 * r["se"], fmt=r["marker"], ms=ms,
                        color=r["color"], ecolor=r["color"], elinewidth=1.8 * scale / 0.78,
                        capsize=4 * scale / 0.78, ls="none", zorder=4)
            points.append(_segment(r["mu"] - Z90 * r["se"], r["mu"] + Z90 * r["se"], xx, n=7,
                                   horizontal=False))
            if r["margin"] is not None and np.isfinite(r["margin"]):
                points.append(np.array([[xx - half, -r["margin"]], [xx + half, -r["margin"]],
                                        [xx - half, r["margin"]], [xx + half, r["margin"]]]))
        ax.axhline(0, color=style.INK, lw=1.0, zorder=1)
        if numeric:
            span = max(xs) - min(xs)
            ax.set_xlim(min(xs) - 0.12 * span - half, max(xs) + 0.12 * span + half)
            ax.set_xlabel(p["xlabel"])
            style.style_axes(ax, scale)
        else:
            ax.set_xticks(xs)
            rot = p.get("rotation", 35)
            ax.set_xticklabels([r["label"] for r in rows], rotation=rot,
                               ha="right" if rot else "center", rotation_mode="anchor",
                               fontsize=s["ann"] * 0.85)
            ax.set_xlim(-0.7, len(rows) - 0.3)
            style.style_axes(ax, scale, minor=False)
            ax.grid(False, axis="x")
        lo, hi = ax.get_ylim()
        if p.get("legend"):
            ax.set_ylim(lo, hi + 0.45 * (hi - lo))
            handles = [Line2D([], [], color=c, marker=m, ls="none", ms=ms, label=l)
                       for c, m, l in p["legend"]]
            covered += place_legend(ax, handles, points, s["legend"] * 0.78,
                                    locs=("upper right", "upper left", "lower right",
                                          "lower left"))
            trim_headroom(ax)
        drawn.append(dict(subject=p["subject"], x=[float(v) for v in xs],
                          ylim=[float(v) for v in ax.get_ylim()]))
    axes[0].set_ylabel(ylabel)
    style.compound_header(axes, line1=line1, line2="", scale=scale,
                          subjects=[p["subject"] for p in panels])
    style.footer(fig, footer, scale)
    _layout(fig)
    _fit_footer(fig)
    return fig, axes, drawn, covered
