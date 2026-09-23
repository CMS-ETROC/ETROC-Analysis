"""wafer_plots.py -- the wafer figures, drawn from the tables of
wafer_tables.py (plot_wafer.py runs both). One PNG per figure; a
figure whose data the runs did not take (no full scan, no QInj) is left
out, and an older copy of it is removed so a folder never mixes results.

Wafer maps draw every die at its wafer_map.csv row and column,
row 0 at the top as on the station's map display, and print the value in
the cell; a die without a value is hatched grey. Colour scales span the
2nd to 98th percentile of the PASSED dies, so one bad die does not flatten
the rest; values beyond take the end colours. Pixel maps draw row 0 at
the top and column 0 at the left. The note (which runs, when plotted) sits
in a band under the figure, clear of the axis labels.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import ListedColormap, Normalize, to_rgba  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402

from wafer_tables import grade_counts, injected_pixels, offset_trend  # noqa: E402

# grade -> colour; the legend lists them in the bin order of the grading in station.py
GRADE_COLOURS = {
    "PASSED": "#2ca02c", "POWER_SHORT": "#d62728", "I2C_NACK": "#ff7f0e",
    "I2C_PIXELS": "#e6b800", "NO_LINK_OR_DATA": "#9467bd", "OTHER_FAIL": "#8c564b",
    "NOT_TESTED": "#dcdcdc", "RAIL_OPEN": "#1f77b4", "BL_NW_ZERO": "#e377c2", "EFUSE_FAIL": "#17becf",
    "EFUSE_TRAILER_FAIL": "#006d5b",
}
MAIN_RAILS = ("analog", "digital")
ZERO_COLOUR = "#e41a1c"
FULL_SCAN_PIXELS = 256

FIGURES = ("grades", "currents", "currents_small_rails", "current_hists", "baseline_maps",
           "baseline_hists", "alignment", "fullscan_baseline", "fullscan_noise_width",
           "pixel_issues", "qinj_overview", "qinj_cal", "qinj_toa", "qinj_tot", "qinj_pixels")


# ---------------------------------------------------------------- helpers

def _num(dies, column):
    """dies[column] as floats, NaN where missing; all NaN without the column."""
    if column not in dies:
        return np.full(len(dies), np.nan)
    return pd.to_numeric(dies[column], errors="coerce").to_numpy(dtype=float)


def _passed(dies):
    return (dies["grade"] == "PASSED").to_numpy()


def _shape(dies):
    return int(dies["die_row"].max()) + 1, int(dies["die_col"].max()) + 1


def value_range(values, centre=None):
    """(lo, hi): the 2nd and 98th percentile of the finite values, made
    symmetric about `centre` when one is given."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(v, [2, 98])
    if centre is not None:
        half = max(abs(lo - centre), abs(hi - centre)) or 1.0
        return centre - half, centre + half
    if hi - lo < 1e-9:
        pad = abs(hi) * 0.05 or 1.0
        return lo - pad, hi + pad
    return float(lo), float(hi)


def _edges(values, target=40):
    """Histogram bin edges on a round step, about `target` bins."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 10
    lo, hi = v.min(), v.max()
    raw = max(hi - lo, 1e-9) / target
    step = next((s for s in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100) if s >= raw), 100)
    start = np.floor(lo / step) * step
    return np.arange(start, hi + step * 1.001, step)


def _signed(v):
    """'+ 1.23' or '− 1.23', for a term of a printed formula."""
    return f"{'+' if v >= 0 else '−'} {abs(v):.2f}"


def _ink(colour):
    r, g, b = to_rgba(colour)[:3]
    return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.5 else "white"


def _wafer_axes(ax, shape):
    nrow, ncol = shape
    ax.set_xlim(-0.5, ncol - 0.5)
    ax.set_ylim(nrow - 0.5, -0.5)
    ax.set_xticks(range(ncol))
    ax.set_yticks(range(nrow))
    ax.tick_params(labelsize=7, length=0)
    ax.set_xlabel("column", fontsize=8)
    ax.set_ylabel("row", fontsize=8)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _pixel_axes(ax):
    ax.set_xticks(range(0, 16, 3))
    ax.set_yticks(range(0, 16, 3))
    ax.tick_params(labelsize=7)
    ax.set_xlabel("pixel column", fontsize=8)
    ax.set_ylabel("pixel row", fontsize=8)


def _colourbar(fig, ax, norm, cmap, label, values=None, cax=None, integer=False):
    extend = "neither"
    if values is not None:
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        low, high = bool((v < norm.vmin).any()), bool((v > norm.vmax).any())
        extend = {(True, True): "both", (True, False): "min", (False, True): "max"}.get((low, high), "neither")
    kw = {"cax": cax} if cax is not None else {"ax": ax, "shrink": 0.82, "pad": 0.02}
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), extend=extend, **kw)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    if integer:
        cb.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    return cb


def _suptitle(fig, title, text, fontsize):
    """The figure title "<title>: <text>", title being the wafer's labels. On
    a narrow figure (two panels, one rail) the labels can push that first
    line past the figure's edges: when it takes more than 95 % of the figure
    width, the labels get a line of their own."""
    heading = fig.suptitle(f"{title}: {text}", fontsize=fontsize)
    first = heading.get_text().split("\n")[0]
    width, _, _ = fig.canvas.get_renderer().get_text_width_height_descent(
        first, heading.get_fontproperties(), ismath=False)
    if width > 0.95 * fig.bbox.width:
        heading.set_text(f"{title}\n{text}")
    return heading


def wafer_map(ax, dies, values, *, title, label="", fmt="{:.0f}", cmap="viridis",
              ref=None, centre=None, vrange=None, frame=None, fontsize=6.5, integer=False):
    """Colour each die of the dies table by `values` (one per row), print
    the value in the cell, hatch the dies without one. `ref` (a boolean
    mask) picks the dies that set the colour range, `vrange` fixes it,
    `frame` (a mask) outlines dies in red; `integer` puts whole numbers on
    the colour bar (counts)."""
    values = np.asarray(values, dtype=float)
    rows = dies["die_row"].to_numpy(dtype=int)
    cols = dies["die_col"].to_numpy(dtype=int)
    if vrange is None:
        use = values[ref] if ref is not None and np.isfinite(values[ref]).any() else values
        vrange = value_range(use, centre)
    norm = Normalize(*vrange)
    cmap = plt.get_cmap(cmap)
    _wafer_axes(ax, _shape(dies))
    for r, c, v in zip(rows, cols, values):
        if np.isfinite(v):
            colour = cmap(norm(v))
            ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, facecolor=colour, edgecolor="white", lw=0.6))
            ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=fontsize, color=_ink(colour))
        else:
            ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, facecolor="#f4f4f4", edgecolor="#c8c8c8",
                                   hatch="////", lw=0.5))
    if frame is not None:
        for r, c in zip(rows[frame], cols[frame]):
            ax.add_patch(Rectangle((c - .44, r - .44), 0.88, 0.88, fill=False, edgecolor=ZERO_COLOUR, lw=1.5))
    _colourbar(ax.figure, ax, norm, cmap, label, values, integer=integer)
    ax.set_title(title, fontsize=10)


def _pixel_values(dies, qinj, pixel, column):
    """dies-ordered values of one qinj column for one pixel (NaN elsewhere)."""
    r, c = pixel
    per_die = qinj[(qinj["pix_row"] == r) & (qinj["pix_col"] == c)].set_index("die")[column]
    return dies["die"].map(per_die).to_numpy(dtype=float)


# ---------------------------------------------------------------- figures

def fig_grades(dies, title):
    fig, ax = plt.subplots(figsize=(9.8, 6.8), layout="constrained")
    _wafer_axes(ax, _shape(dies))
    for d in dies.itertuples(index=False):
        colour = GRADE_COLOURS.get(d.grade, "white")
        ax.add_patch(Rectangle((d.die_col - .5, d.die_row - .5), 1, 1, facecolor=colour,
                               edgecolor="white", lw=0.8))
        retry = str(getattr(d, "map_text", "")).endswith("_retry")
        ax.text(d.die_col, d.die_row, f"{d.die}{'*' if retry else ''}", ha="center", va="center",
                fontsize=7, color=_ink(colour))
    counts, passed, tested = grade_counts(dies)
    handles = [Patch(facecolor=GRADE_COLOURS.get(g, "white"), label=f"{g}: {n}") for g, n in counts]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
    share = f" ({100 * passed / tested:.1f} %)" if tested else ""
    ax.set_title(f"{title}: grade per die, PASSED {passed} of {tested} tested{share}\n"
                 "die number in each cell; * = graded on the retry", fontsize=10)
    return fig


def fig_currents(dies, title):
    rails = [r for r in MAIN_RAILS if f"{r}_I_on" in dies]
    if not rails:
        return None
    ok = _passed(dies)
    fig, axes = plt.subplots(len(rails), 3, figsize=(17, 5.0 * len(rails)), squeeze=False,
                             layout="constrained")
    for i, rail in enumerate(rails):
        on = _num(dies, f"{rail}_I_on") * 1e3
        high = _num(dies, f"{rail}_I_high") * 1e3
        wafer_map(axes[i, 0], dies, on, title=f"{rail}: power-on current", label="mA", ref=ok)
        wafer_map(axes[i, 1], dies, high, title=f"{rail}: high-power current", label="mA", ref=ok)
        wafer_map(axes[i, 2], dies, high - on, title=f"{rail}: high power minus power-on",
                  label="mA", ref=ok)
    _suptitle(fig, title, "rail currents, median over each run phase "
                          "(first sweep of every phase dropped)", fontsize=12)
    return fig


def fig_small_rails(dies, title):
    rails = [c[:-len("_I_high")] for c in dies.columns if c.endswith("_I_high")]
    rails = [r for r in rails if r not in MAIN_RAILS and np.isfinite(_num(dies, f"{r}_I_high")).any()]
    if not rails:
        return None
    ok = _passed(dies)
    fig, axes = plt.subplots(1, len(rails), figsize=(5.7 * len(rails), 5.0), squeeze=False,
                             layout="constrained")
    for ax, rail in zip(axes[0], rails):
        wafer_map(ax, dies, _num(dies, f"{rail}_I_high") * 1e3, title=f"{rail}: high-power current",
                  label="mA", fmt="{:.1f}", ref=ok)
    _suptitle(fig, title, "the other rails at high power", fontsize=12)
    return fig


def fig_current_hists(dies, title):
    rails = [r for r in MAIN_RAILS if f"{r}_I_on" in dies]
    if not rails:
        return None
    fig, axes = plt.subplots(1, len(rails), figsize=(6.6 * len(rails), 4.6), squeeze=False,
                             layout="constrained")
    for ax, rail in zip(axes[0], rails):
        series = [("power-on", _num(dies, f"{rail}_I_on") * 1e3, "#1f77b4"),
                  ("high power", _num(dies, f"{rail}_I_high") * 1e3, "#ff7f0e")]
        edges = _edges(np.concatenate([v for _, v, _ in series]))
        for name, v, colour in series:
            v = v[np.isfinite(v)]
            ax.hist(v, bins=edges, histtype="step", lw=1.6, color=colour, label=f"{name}: {len(v)} dies")
        for name, column, colour, style in (("short check", "abort_above", "#d62728", "--"),
                                            ("open-rail floor", "abort_below", "#555555", ":")):
            used = pd.Series(_num(dies, f"{rail}_I_{column}") * 1e3).dropna().round(0).value_counts()
            for value, n in used.sort_index().items():
                ax.axvline(value, color=colour, ls=style, lw=1, label=f"{name}: {value:.0f} mA ({n} dies)")
        ax.set_xlabel(f"{rail} current (mA)", fontsize=9)
        ax.set_ylabel("dies", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ax.set_title(rail, fontsize=10)
    _suptitle(fig, title, "rail currents per die, median over each run phase "
                          "(lines: the check thresholds the runs used)", fontsize=11)
    return fig


def fig_baseline_maps(dies, title):
    if not np.isfinite(_num(dies, "bl_mean")).any():
        return None
    ok = _passed(dies)
    zero = _num(dies, "n_zero_pixels") > 0
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 10.2), layout="constrained")
    for ax, column, what, fmt in ((axes[0, 0], "bl_mean", "baseline mean", "{:.0f}"),
                                  (axes[0, 1], "bl_std", "baseline std", "{:.1f}"),
                                  (axes[1, 0], "nw_mean", "noise-width mean", "{:.1f}"),
                                  (axes[1, 1], "nw_std", "noise-width std", "{:.1f}")):
        wafer_map(ax, dies, _num(dies, column), title=what, label="DAC code", fmt=fmt, ref=ok, frame=zero)
    sizes = pd.Series(_num(dies, "n_pixels")).dropna().astype(int).value_counts().sort_index()
    mix = ", ".join(f"{k} pixels: {n} dies" for k, n in sizes.items())
    _suptitle(fig, title, "baseline and noise width per die, over its calibrated pixels that did not "
                          f"read zero\n({mix}; red frame: some pixels read zero, see pixel_issues.png)", fontsize=11)
    return fig


def fig_baseline_hists(pixels, title):
    """Pixels pooled over dies, the quick-test dies and the full-scan dies
    in rows of their own: a full scan brings 256 pixels per die and would
    swamp the quick test's few."""
    if pixels.empty:
        return None
    zero = (pixels["baseline"] == 0) | (pixels["noise_width"] == 0)
    full = pixels.groupby("die")["pix_row"].transform("size") >= FULL_SCAN_PIXELS
    groups = [(name, pixels[~zero & sel]) for name, sel in (("quick test", ~full), ("full scan", full))]
    groups = [(name, g) for name, g in groups if not g.empty]
    if not groups:
        return None
    fig, axes = plt.subplots(len(groups), 2, figsize=(12.5, 4.2 * len(groups)), squeeze=False,
                             layout="constrained")
    for row, (name, g) in zip(axes, groups):
        for ax, column, what in ((row[0], "baseline", "baseline"), (row[1], "noise_width", "noise width")):
            v = g[column].to_numpy(dtype=float)
            ax.hist(v, bins=np.arange(v.min() - 0.5, v.max() + 1.5, 1), histtype="stepfilled",
                    color="#1f77b4", alpha=0.7)
            ax.set_xlabel(f"{what} (DAC code)", fontsize=9)
            ax.set_ylabel("pixels per code", fontsize=9)
            spread = f", std {v.std(ddof=1):.2f}" if len(v) > 1 else ""
            ax.set_title(f"{name}, {what}: {len(v)} pixels of {g['die'].nunique()} dies, "
                         f"mean {v.mean():.2f}{spread}", fontsize=10)
    _suptitle(fig, title, "calibrated pixels pooled over dies, one bin per DAC code "
                          f"({int(zero.sum())} zero readings left out)", fontsize=11)
    return fig


def fig_alignment(dies, title):
    dx, dy = _num(dies, "dx_um"), _num(dies, "dy_um")
    if not (np.isfinite(dx).any() or np.isfinite(dy).any()):
        return None
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 10.0), layout="constrained")
    for ax, v, axis in ((axes[0, 0], dx, "x"), (axes[0, 1], dy, "y")):
        v = v[np.isfinite(v)]
        ax.hist(v, bins=_edges(v, 30), histtype="stepfilled", color="#1f77b4", alpha=0.7)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel(f"chuck {axis} minus map {axis} (µm)", fontsize=9)
        ax.set_ylabel("dies", fontsize=9)
        spread = f", std {v.std(ddof=1):.1f}" if len(v) > 1 else ""
        ax.set_title(f"d{axis}: mean {v.mean():.1f} µm{spread} µm, {len(v)} dies", fontsize=10)
    wafer_map(axes[1, 0], dies, dx, title="dx per die", label="µm", fmt="{:.1f}", cmap="RdBu_r", centre=0.0)
    wafer_map(axes[1, 1], dies, dy, title="dy per die", label="µm", fmt="{:.1f}", cmap="RdBu_r", centre=0.0)
    trends = []
    for axis in ("x", "y"):
        t = offset_trend(dies, f"d{axis}_um")
        if t is not None:
            a, b, c, s, n = t
            trends.append(f"d{axis} = {a:.1f} {_signed(b)} × column {_signed(c)} × row, residual std {s:.1f}")
    fit = f"\nplane fitted over the wafer (µm): {'; '.join(trends)}" if trends else ""
    _suptitle(fig, title, f"chuck position read at contact minus the station's map position of the die{fit}",
                          fontsize=11)
    return fig


def fig_fullscan(dies, pixels, title, column, what):
    """Every full-scan die's 16 x 16 map at its place on the wafer; zero
    readings in red; the other dies as grey frames."""
    full = set(dies.loc[_num(dies, "n_pixels") >= FULL_SCAN_PIXELS, "die"])
    if not full:
        return None
    nrow, ncol = _shape(dies)
    sel = pixels[pixels["die"].isin(full)]
    zero = (sel["baseline"] == 0) | (sel["noise_width"] == 0)
    norm = Normalize(*value_range(sel.loc[~zero, column].to_numpy(dtype=float)))
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(ncol * 1.0 + 1.4, nrow * 1.0 + 1.2))
    left, right, top, bottom = 0.02, 0.89, 0.91, 0.04
    cw, ch = (right - left) / ncol, (top - bottom) / nrow
    for d in dies.itertuples(index=False):
        ax = fig.add_axes([left + d.die_col * cw + 0.05 * cw, top - (d.die_row + 1) * ch + 0.03 * ch,
                           0.9 * cw, 0.8 * ch])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, 15.5)
        ax.set_ylim(15.5, -0.5)
        ax.set_aspect("equal")
        if d.die in full:
            p = sel[sel["die"] == d.die]
            img = np.full((16, 16), np.nan)
            img[p["pix_row"].to_numpy(int), p["pix_col"].to_numpy(int)] = p[column].to_numpy(float)
            bad = np.zeros((16, 16), dtype=bool)
            pz = p[(p["baseline"] == 0) | (p["noise_width"] == 0)]
            bad[pz["pix_row"].to_numpy(int), pz["pix_col"].to_numpy(int)] = True
            ax.imshow(np.ma.masked_where(bad | np.isnan(img), img), cmap=cmap, norm=norm,
                      interpolation="nearest")
            if bad.any():
                ax.imshow(np.ma.masked_where(~bad, np.ones((16, 16))), cmap=ListedColormap([ZERO_COLOUR]),
                          vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(str(d.die), fontsize=6.5, pad=1.5)
        else:
            for spine in ax.spines.values():
                spine.set_color("#dddddd")
            ax.text(0.5, 0.5, str(d.die), transform=ax.transAxes, ha="center", va="center",
                    fontsize=6, color="#bbbbbb")
    _colourbar(fig, None, norm, cmap, f"{what} (DAC code)", sel.loc[~zero, column],
               cax=fig.add_axes([0.915, 0.3, 0.015, 0.4]))
    _suptitle(fig, title, f"{what} of every pixel, {len(full)} full-scan dies at their wafer positions\n"
                          "each die: row 0 at the top, column 0 at the left; red = read zero; grey frame = no full scan",
                          fontsize=11)
    return fig


def fig_pixel_issues(dies, pixels, title):
    fails = np.zeros((16, 16))
    for text in dies.get("failed_pixels", pd.Series(dtype=object)).dropna():
        for r, c in (json.loads(text) if text else []):
            fails[r, c] += 1
    checked = int(dies.get("pixel_id_ok", pd.Series(dtype=object)).notna().sum())
    zeros = np.zeros((16, 16))
    calibrated = np.zeros((16, 16))
    if not pixels.empty:
        np.add.at(calibrated, (pixels["pix_row"].to_numpy(int), pixels["pix_col"].to_numpy(int)), 1)
        z = pixels[(pixels["baseline"] == 0) | (pixels["noise_width"] == 0)]
        np.add.at(zeros, (z["pix_row"].to_numpy(int), z["pix_col"].to_numpy(int)), 1)
    if checked == 0 and pixels.empty:
        return None
    panels = [(fails, f"pixel-ID check failed ({checked} dies checked)", "dies", True),
              (zeros, "baseline or noise width read zero", "dies", True)]
    full = pixels[pixels["die"].isin(dies.loc[_num(dies, "n_pixels") >= FULL_SCAN_PIXELS, "die"])]
    good = full[(full["baseline"] > 0) & (full["noise_width"] > 0)]
    if not good.empty:
        g = good.groupby(["pix_row", "pix_col"])["baseline"]
        mean, std = np.full((16, 16), np.nan), np.full((16, 16), np.nan)
        for (r, c), v in g.mean().items():
            mean[r, c] = v
        for (r, c), v in g.std().items():
            std[r, c] = v
        n_full = full["die"].nunique()
        panels += [(mean, f"baseline mean over {n_full} full-scan dies", "DAC code", False),
                   (std, f"baseline std over {n_full} full-scan dies", "DAC code", False)]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5.0), squeeze=False,
                             layout="constrained")
    for ax, (img, what, label, counts) in zip(axes[0], panels):
        if counts:
            norm, cmap = Normalize(0, max(1.0, float(np.nanmax(img)))), plt.get_cmap("Reds")
        else:
            norm, cmap = Normalize(*value_range(img)), plt.get_cmap("viridis")
        ax.imshow(img, cmap=cmap, norm=norm, interpolation="nearest")
        if counts:
            for r, c in zip(*np.nonzero(img)):
                ax.text(c, r, f"{img[r, c]:.0f}", ha="center", va="center", fontsize=6,
                        color=_ink(cmap(norm(img[r, c]))))
        _pixel_axes(ax)
        _colourbar(fig, ax, norm, cmap, label, img, integer=counts)
        ax.set_title(what, fontsize=10)
    _suptitle(fig, title, "per pixel, over dies (a pixel counts for zero readings only in dies that "
                          f"calibrated it: {int(calibrated.max())} dies at most)", fontsize=11)
    return fig


def fig_qinj_overview(dies, qinj, title):
    if not np.isfinite(_num(dies, "qinj_events")).any():
        return None
    n = len(injected_pixels(qinj))
    # rounded down, so a die below 100 % never prints 100
    eff = np.floor(_num(dies, "qinj_min_eff") * 100 + 1e-9)
    ok = _passed(dies)
    flagged, ea = _num(dies, "qinj_flagged_trailers"), _num(dies, "qinj_ea_words")
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 10.2), layout="constrained")
    wafer_map(axes[0, 0], dies, _num(dies, "qinj_events"), title="events read",
              label="events", ref=ok)
    wafer_map(axes[0, 1], dies, eff, title="lowest efficiency over the injected pixels, rounded down",
              label="%", vrange=(0, 100))
    wafer_map(axes[1, 0], dies, flagged, title="frame trailers with a nonzero chip status",
              label="trailers", cmap="Reds", vrange=(0, max(1.0, float(np.nanmax(flagged)))), integer=True)
    wafer_map(axes[1, 1], dies, ea, title="hit words with a nonzero EA flag", label="words",
              cmap="Reds", vrange=(0, max(1.0, float(np.nanmax(ea)))), integer=True)
    _suptitle(fig, title, "charge injection, the files after the first of each qinj/ run (older runs: all of qinj_run2/)\n"
                          "efficiency = hits with EA = 0 per event; injected pixels as the run recorded them, "
                          f"else the wafer's {n} if the run expected {n} hits per event", fontsize=11)
    return fig


def fig_qinj_maps(dies, qinj, title, quantity):
    injected = injected_pixels(qinj)
    if not injected:
        return None
    ncols = 3
    nrows = -(-len(injected) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.9 * nrows), squeeze=False,
                             layout="constrained")
    for ax, pixel in zip(axes.flat, injected):
        wafer_map(ax, dies, _pixel_values(dies, qinj, pixel, f"{quantity}_mean"),
                  title=f"pixel {pixel}", label=f"{quantity.upper()} code", fontsize=5.5, ref=_passed(dies))
    for ax in axes.flat[len(injected):]:
        ax.axis("off")
    _suptitle(fig, title, f"mean {quantity.upper()} code per die and injected pixel, the files after the first of each qinj/ run (older runs: all of qinj_run2/),\n"
                          "hits with EA = 0 and |CAL - its most common value| < 3", fontsize=11)
    return fig


def fig_qinj_pixels(qinj, title):
    injected = injected_pixels(qinj)
    if not injected:
        return None
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.6), layout="constrained")
    for j, quantity in enumerate(("cal", "toa", "tot")):
        for i, stat in enumerate(("mean", "std")):
            ax = axes[i, j]
            for k, (r, c) in enumerate(injected):
                v = qinj.loc[(qinj["pix_row"] == r) & (qinj["pix_col"] == c), f"{quantity}_{stat}"]
                v = v.dropna().to_numpy(dtype=float)
                ax.scatter(k + rng.uniform(-0.18, 0.18, len(v)), v, s=14, edgecolor="none", alpha=0.8,
                           color="#1f77b4" if c < 8 else "#ff7f0e")
            ax.set_xticks(range(len(injected)))
            ax.set_xticklabels([f"({r},{c})" for r, c in injected], fontsize=7, rotation=45)
            ax.set_ylabel(f"{quantity.upper()} {stat} (code)", fontsize=9)
            ax.tick_params(labelsize=7)
    handles = [Patch(color="#1f77b4", label="pixel columns 0-7"), Patch(color="#ff7f0e", label="pixel columns 8-15")]
    axes[0, 0].legend(handles=handles, fontsize=7, frameon=False)
    _suptitle(fig, title, "CAL, TOA and TOT per injected pixel, one dot per die "
                          f"({qinj['die'].nunique()} dies; same selection as the maps)", fontsize=11)
    return fig


def plot_all(out_dir, dies, pixels, qinj, title, note="", dpi=130):
    """Write every figure the tables support into out_dir and remove the
    older copy of any figure they do not; returns the paths written."""
    makers = {
        "grades": lambda: fig_grades(dies, title),
        "currents": lambda: fig_currents(dies, title),
        "currents_small_rails": lambda: fig_small_rails(dies, title),
        "current_hists": lambda: fig_current_hists(dies, title),
        "baseline_maps": lambda: fig_baseline_maps(dies, title),
        "baseline_hists": lambda: fig_baseline_hists(pixels, title),
        "alignment": lambda: fig_alignment(dies, title),
        "fullscan_baseline": lambda: fig_fullscan(dies, pixels, title, "baseline", "baseline"),
        "fullscan_noise_width": lambda: fig_fullscan(dies, pixels, title, "noise_width", "noise width"),
        "pixel_issues": lambda: fig_pixel_issues(dies, pixels, title),
        "qinj_overview": lambda: fig_qinj_overview(dies, qinj, title),
        "qinj_cal": lambda: fig_qinj_maps(dies, qinj, title, "cal"),
        "qinj_toa": lambda: fig_qinj_maps(dies, qinj, title, "toa"),
        "qinj_tot": lambda: fig_qinj_maps(dies, qinj, title, "tot"),
        "qinj_pixels": lambda: fig_qinj_pixels(qinj, title),
    }
    written = []
    for name in FIGURES:
        path = Path(out_dir) / f"{name}.png"
        fig = makers[name]()
        if fig is None:
            path.unlink(missing_ok=True)
            continue
        if note:
            # below the figure's bottom edge: the tight bounding box of savefig
            # takes it in, and no axis label can reach it
            fig.text(1.0, 0.0, note, ha="right", va="top", fontsize=6.5, color="#777777",
                     transform=fig.transFigure + ScaledTranslation(0, -8 / 72, fig.dpi_scale_trans))
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written
