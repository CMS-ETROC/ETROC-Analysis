"""lot_compare_plots.py -- the figures of what UBM and bumping did to the
wafers of a lot, drawn from lot_dies.csv rows (lot_compare.py;
plot_lots.py runs both). Per lot: the outcome of each wafer's dies
(yield), a map of each wafer coloured by what happened to each die
(transitions), the new failures by die position (edges), the current
changes (currents) and the readings of the shorted dies (shorts); and
across lots, one figure comparing them (lots). The maps follow
wafer_plots.py: row 0 at the top, the wafer seen as on the station's map
display, invalid dies grey with INVALID across. Only the valid dies count.
"""
import math
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402

from lot_compare import (CLASSES, NEW_FAILURES, at_risk, counts, median_by_position, passed,  # noqa: E402
                         per_position, profile, relative_change, stage_text, valid_positions)
from stage_compare import MAIN_RAILS  # noqa: E402
from wafer_plots import _ink, _invalid_cell, _shape, _suptitle, _wafer_axes, wafer_map  # noqa: E402

CLASS_COLOURS = {
    "passed both": "#2ca02c", "recovered": "#98df8a", "new analog short": "#d62728",
    "new digital short": "#ff7f0e", "new short, other or unknown rail": "#8c564b", "new other failure": "#9467bd",
    "failed both": "#7f7f7f", "untested in a stage": "#dcdcdc",
}
LOT_FIGURES = ("yield", "transitions", "edges", "currents", "shorts")
WAFER_COLOURS = plt.get_cmap("tab10").colors
ERRORS = "68 % Wilson intervals"


def _wafers(lot):
    return list(dict.fromkeys(lot["wafer"]))


def _valid(lot):
    return lot[~lot["invalid"].astype(bool)]


def _class_handles(lot, classes=CLASSES):
    n = counts(lot)
    return [Patch(facecolor=CLASS_COLOURS[c], edgecolor="#555555", lw=0.4, label=f"{c}: {n[c]}")
            for c in classes if n[c] or c in NEW_FAILURES]


def fig_yield(lot, title):
    """Per wafer, the valid dies that did not pass both stages, stacked by
    what happened to them, with PASSED before -> after beside each bar."""
    wafers = _wafers(lot)
    shown = [c for c in CLASSES if c != "passed both"]
    fig, ax = plt.subplots(figsize=(12, 1.2 + 0.55 * len(wafers)), layout="constrained")
    widest = 0
    for i, w in enumerate(wafers):
        sub = _valid(lot[lot["wafer"] == w])
        left = 0
        for c in shown:
            n = int((sub["transition"] == c).sum())
            if n:
                ax.barh(i, n, left=left, color=CLASS_COLOURS[c], edgecolor="white", lw=0.8)
                ax.text(left + n / 2, i, str(n), ha="center", va="center", fontsize=8, color=_ink(CLASS_COLOURS[c]))
                left += n
        widest = max(widest, left)
        ax.text(left + 0.4, i, f"PASSED {passed(sub, 'pre')} → {passed(sub, 'post')} of {len(sub)}",
                va="center", fontsize=8.5)
    ax.set_xlim(0, widest + 9)
    ax.set_yticks(range(len(wafers)), wafers, fontsize=9)
    ax.set_ylim(len(wafers) - 0.5, -0.5)
    ax.set_xlabel("valid dies that did not pass both stages", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(handles=_class_handles(lot, shown), title=f"lot, {len(wafers)} wafers", loc="lower right",
              fontsize=8, title_fontsize=8, frameon=False)
    _suptitle(fig, title, f"what UBM and bumping did to each wafer's dies; PASSED before → after UBM, "
                          f"{passed(lot, 'pre')} → {passed(lot, 'post')} over the lot", fontsize=11)
    return fig


def fig_transitions(lot, title):
    """One wafer map per wafer, each die coloured by what happened to it."""
    wafers = _wafers(lot)
    ncols = min(4, len(wafers))
    nrows = math.ceil(len(wafers) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 4.0 * nrows + 1.2), squeeze=False,
                             layout="constrained")
    for ax, w in zip(axes.flat, wafers):
        sub = lot[lot["wafer"] == w]
        _wafer_axes(ax, _shape(sub))
        ax.tick_params(labelsize=5)
        ax.set_xlabel("")
        ax.set_ylabel("")
        for d in sub.itertuples(index=False):
            if d.invalid:
                _invalid_cell(ax, d.die_col, d.die_row)
                continue
            colour = CLASS_COLOURS[d.transition]
            ax.add_patch(Rectangle((d.die_col - .5, d.die_row - .5), 1, 1, facecolor=colour,
                                   edgecolor="white", lw=0.6))
            ax.text(d.die_col, d.die_row, str(d.die), ha="center", va="center", fontsize=4.5, color=_ink(colour))
        ax.set_title(f"{w}: PASSED {passed(sub, 'pre')} → {passed(sub, 'post')}", fontsize=9)
    for ax in axes.flat[len(wafers):]:
        ax.axis("off")
    fig.legend(handles=_class_handles(lot), loc="outside lower center", ncol=4, fontsize=8, frameon=False)
    _suptitle(fig, title, "what happened to each die after UBM; row 0 at the top, as on the station's map",
              fontsize=11)
    return fig


def _profile_axes(ax, lot, by, label):
    """The fraction of the dies at risk that became a new analog short, and
    that failed newly in any way, per value of `by`, with Wilson bars."""
    for classes, name, colour, dx in ((("new analog short",), "new analog short", CLASS_COLOURS["new analog short"],
                                       -0.12),
                                      (NEW_FAILURES, "any new failure", "#333333", 0.12)):
        p = profile(lot, by, classes)
        x = p.index.to_numpy(dtype=float) + dx
        ax.errorbar(x, 100 * p["fraction"], yerr=[100 * (p["fraction"] - p["lo"]), 100 * (p["hi"] - p["fraction"])],
                    fmt="o", ms=4, color=colour, capsize=2, lw=1, label=name)
    n = profile(lot, by)["n"]
    top = ax.get_ylim()[1]
    for x, count in n.items():
        ax.text(x, top, str(count), ha="center", va="bottom", fontsize=6, color="#555555")
    ax.set_ylim(0, top)
    ax.set_xticks(n.index)
    ax.tick_params(labelsize=7)
    ax.set_xlabel(f"{label} (above each: dies at risk, all wafers)", fontsize=8)
    ax.set_ylabel("% of the dies at risk", fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc="upper left")


def fig_edges(lot, title):
    """Where on the wafer the new failures are: per die position, on how
    many wafers it failed newly (analog shorts; any new failure), and the
    new-failure fraction per map row and column."""
    positions = valid_positions(lot)
    n_wafers = lot["wafer"].nunique()
    risk_per_die = at_risk(lot).groupby("die").size()
    at_risk_n = positions["die"].map(risk_per_die).fillna(0).to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 11.5), layout="constrained")
    for ax, classes, name in ((axes[0, 0], ("new analog short",), "new analog shorts"),
                              (axes[0, 1], NEW_FAILURES, "any new failure")):
        values = per_position(lot, classes, positions)
        values = np.where(at_risk_n > 0, values, np.nan)
        wafer_map(ax, positions, values, title=f"{name}: wafers per die position (of {n_wafers})", label="wafers",
                  fmt="{:.0f}", cmap="Reds", vrange=(0, max(1.0, np.nanmax(values) if np.isfinite(values).any() else 1)),
                  integer=True)
    _profile_axes(axes[1, 0], lot, "die_row", "map row (0 at the top)")
    _profile_axes(axes[1, 1], lot, "die_col", "map column (0 at the left)")
    ring = profile(lot, "edge_ring", ("new analog short",))
    ring_text = "; ".join(f"{'edge ring' if flag else 'interior'}: {int(r.k)} of {int(r.n)} "
                          f"({100 * r.fraction:.1f} %)" for flag, r in ring.iterrows())
    _suptitle(fig, title, "where on the wafer the dies PASSED before UBM failed after it (dies at risk: valid, "
                          "PASSED before, tested after)\n"
                          f"new analog shorts by position on the map, {ring_text}; bars: {ERRORS}; "
                          "a die with no wafer at risk hatched", fontsize=11)
    return fig


def _boxes(ax, lot, column, wafers, offset, colour, label):
    data = [pd.to_numeric(lot.loc[lot["wafer"] == w, column], errors="coerce").dropna().to_numpy() * 1e3
            for w in wafers]
    keep = [i for i, d in enumerate(data) if len(d)]
    if not keep:
        return
    parts = ax.boxplot([data[i] for i in keep], positions=[i + offset for i in keep], widths=0.34,
                       patch_artist=True, showfliers=True, flierprops={"ms": 2.5, "mec": colour},
                       medianprops={"color": "black"})
    for box in parts["boxes"]:
        box.set_facecolor(colour)
        box.set_alpha(0.7)
    ax.plot([], [], "s", color=colour, alpha=0.7, label=label)


def fig_currents(lot, title, setpoints):
    """The rail current changes of the dies PASSED in both stages: per
    wafer, and, with each wafer's median change taken out, per die position
    and per map row."""
    wafers = _wafers(lot)
    if "d_analog_I_on" not in lot:
        return None
    phased = lot[np.isfinite(pd.to_numeric(lot["d_analog_I_on"], errors="coerce"))]
    if phased.empty:
        return None
    fig, axes = plt.subplots(2, 3, figsize=(19, 11.5), layout="constrained")
    for ax, rail in zip(axes[0, :2], MAIN_RAILS):
        _boxes(ax, phased, f"d_{rail}_I_on", wafers, -0.19, "#1f77b4", "power-on")
        _boxes(ax, phased, f"d_{rail}_I_high", wafers, 0.19, "#ff7f0e", "high power")
        ax.axhline(0, color="#777777", lw=1, ls="--")
        ax.set_xticks(range(len(wafers)), wafers, fontsize=8)
        ax.set_ylabel(f"{rail} current after minus before UBM (mA)", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.legend(fontsize=7, frameon=False)
        ax.set_title(f"{rail}: per wafer, dies PASSED in both stages", fontsize=10)
    ax = axes[0, 2]
    rows = []
    for w in wafers:
        v = setpoints.get(w, {})
        pre, post = v.get("pre_ubm", {}), v.get("post_ubm", {})
        rows.append([w] + [f"{pre[r]:.3f} → {post[r]:.3f}" if r in pre and r in post else "–" for r in MAIN_RAILS])
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["wafer", "analog (V)", "digital (V)"], loc="upper center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    ax.set_title("rail voltage at power-on, median of the PASSED dies, before → after", fontsize=10)
    positions = valid_positions(lot)
    for ax, phase, name in ((axes[1, 0], "on", "power-on"), (axes[1, 1], "high", "high-power")):
        rel = relative_change(phased, f"d_analog_I_{phase}") * 1e3
        wafer_map(ax, positions, median_by_position(phased, rel, positions),
                  title=f"analog {name} current change minus its wafer's median: median over the wafers",
                  label="mA", fmt="{:+.0f}", cmap="RdBu_r", centre=0.0)
    ax = axes[1, 2]
    for phase, name, colour, dx in (("on", "power-on", "#1f77b4", -0.12), ("high", "high power", "#ff7f0e", 0.12)):
        rel = relative_change(phased, f"d_analog_I_{phase}") * 1e3
        g = rel.groupby(phased["die_row"])
        med, n = g.median(), g.size()
        err = 1.2533 * 0.7413 * (g.quantile(0.75) - g.quantile(0.25)) / np.sqrt(n)
        ax.errorbar(med.index + dx, med, yerr=err, fmt="o", ms=4, color=colour, capsize=2, lw=1, label=name)
    ax.axhline(0, color="#777777", lw=1, ls="--")
    ax.set_xticks(sorted(phased["die_row"].unique()))
    ax.tick_params(labelsize=7)
    ax.set_xlabel("map row (0 at the top)", fontsize=8)
    ax.set_ylabel("analog current change minus its wafer's median (mA)", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("per map row: median, bars 1.25 × IQR / 1.35 / √n", fontsize=10)
    _suptitle(fig, title, "rail current changes after UBM, dies PASSED in both stages with the power phases "
                          "logged in both", fontsize=11)
    return fig


def fig_shorts(lot, title):
    """Each valid die tested in both stages and graded POWER_SHORT after
    UBM: the voltage and current of each main rail at the check that
    stopped it (the high-power one for a detail ending _high), new shorts
    filled, dies failed before as well open, the edge-ring dies square."""
    post = _valid(lot[(lot["grade_post"] == "POWER_SHORT") & (lot["transition"] != "untested in a stage")])
    if post.empty:
        return None
    wafers = _wafers(lot)
    colours = {w: WAFER_COLOURS[i % len(WAFER_COLOURS)] for i, w in enumerate(wafers)}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), layout="constrained")
    plotted = set()
    for ax, rail in zip(axes, MAIN_RAILS):
        for d in post.itertuples(index=False):
            phase = "high" if str(d.detail_post or "").endswith("_high") else "on"
            v = pd.to_numeric(getattr(d, f"{rail}_V_{phase}_post", np.nan), errors="coerce")
            i = pd.to_numeric(getattr(d, f"{rail}_I_{phase}_post", np.nan), errors="coerce")
            if not (np.isfinite(v) and np.isfinite(i)):
                continue
            new = d.transition != "failed both"
            plotted.add((d.wafer, d.die))
            ax.scatter(v, i * 1e3, s=34, marker="s" if d.edge_ring else "o", linewidths=1.2,
                       facecolors=colours[d.wafer] if new else "none", edgecolors=colours[d.wafer])
            ax.annotate(str(d.die), (v, i * 1e3), xytext=(3, 2), textcoords="offset points", fontsize=6,
                        color="#444444")
        ok = pd.to_numeric(lot.loc[lot["grade_post"] == "PASSED", f"{rail}_V_on_post"], errors="coerce").median()
        if np.isfinite(ok):
            ax.axvline(ok, color="#777777", lw=1, ls="--")
            ax.text(ok, ax.get_ylim()[1], f" PASSED dies' median {ok:.3f} V", fontsize=7, color="#555555",
                    va="top", rotation=90, ha="right")
        ax.set_xlabel(f"{rail} rail voltage (V)", fontsize=9)
        ax.set_ylabel(f"{rail} rail current (mA)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_title(f"{rail} rail", fontsize=10)
    handles = [Line2D([], [], ls="", marker="o", color=colours[w], label=w) for w in wafers
               if (post["wafer"] == w).any()]
    handles += [Line2D([], [], ls="", marker="o", mfc="#555555", mec="#555555", label="new after UBM"),
                Line2D([], [], ls="", marker="o", mfc="none", mec="#555555", label="failed before as well"),
                Line2D([], [], ls="", marker="s", mfc="none", mec="#555555", label="edge-ring die")]
    fig.legend(handles=handles, loc="outside right upper", fontsize=8, frameon=False)
    _suptitle(fig, title, f"the {len(plotted)} valid dies graded POWER_SHORT after UBM and tested before, each "
                          "rail read at the check that stopped the die (power-on, or high power for a _high "
                          "short);\na supply at its current limit reads the limit, and a rail pulled far below its "
                          "setpoint is a hard short", fontsize=11)
    return fig


def fig_lots(lots, title="lots"):
    """The lots side by side: the new failures of the dies at risk by
    class, and the new analog shorts per map row and column."""
    labels = list(lots)
    colours = {label: WAFER_COLOURS[i % len(WAFER_COLOURS)] for i, label in enumerate(labels)}
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.8), layout="constrained")
    ax = axes[0]
    for i, label in enumerate(labels):
        lot = lots[label]
        risk = at_risk(lot)
        bottom = 0.0
        for c in NEW_FAILURES:
            f = 100 * (risk["transition"] == c).mean() if len(risk) else 0.0
            ax.bar(i, f, bottom=bottom, color=CLASS_COLOURS[c], edgecolor="white", lw=0.8)
            bottom += f
        lo, hi = profile(lot.assign(all=0), "all").loc[0, ["lo", "hi"]] if len(risk) else (np.nan, np.nan)
        ax.errorbar(i, bottom, yerr=[[bottom - 100 * lo], [100 * hi - bottom]], color="black", capsize=3, lw=1)
    ax.set_xticks(range(len(labels)), [f"{textwrap.fill(label, 16)}\n{lots[label]['wafer'].nunique()} wafers\nPASSED "
                                       f"{passed(lots[label], 'pre')} → {passed(lots[label], 'post')}"
                                       for label in labels], fontsize=8)
    ax.set_ylabel("% of the dies at risk failing newly after UBM", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(handles=[Patch(facecolor=CLASS_COLOURS[c], label=c) for c in NEW_FAILURES], fontsize=7, frameon=False,
              loc="upper left")
    ax.set_title(f"new failures by kind; bar: {ERRORS} of the total", fontsize=10)
    for ax, by, name in ((axes[1], "die_row", "map row (0 at the top)"), (axes[2], "die_col", "map column (0 at the left)")):
        for k, label in enumerate(labels):
            p = profile(lots[label], by, ("new analog short",))
            dx = (k - (len(labels) - 1) / 2) * 0.12
            ax.errorbar(p.index + dx, 100 * p["fraction"],
                        yerr=[100 * (p["fraction"] - p["lo"]), 100 * (p["hi"] - p["fraction"])],
                        fmt="o-", ms=4, lw=1, capsize=2, color=colours[label], label=label)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=7)
        ax.set_xlabel(name, fontsize=8)
        ax.set_ylabel("% of the dies at risk becoming a new analog short", fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        ax.set_title(f"new analog shorts per {name.split(' (')[0]}; bars: {ERRORS}", fontsize=10)
    stages = "\n".join(f"{label}: before {stage_text(lots[label], 'pre')}; after {stage_text(lots[label], 'post')}"
                       for label in labels)
    _suptitle(fig, title, "what UBM and bumping did to each lot, over the dies at risk (valid, PASSED before, "
                          "tested after)\n" + stages, fontsize=10)
    return fig


def _save(fig, path, note, dpi):
    if note:
        fig.text(1.0, 0.0, note, ha="right", va="top", fontsize=6.5, color="#777777", wrap=True,
                 transform=fig.transFigure + ScaledTranslation(0, -8 / 72, fig.dpi_scale_trans))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_lot(out_dir, prefix, lot, setpoints, title, note="", dpi=130):
    """Write the figures of one lot into out_dir as <prefix>_<figure>.png,
    removing the older copy of one the data do not support; returns the
    paths written."""
    makers = {
        "yield": lambda: fig_yield(lot, title),
        "transitions": lambda: fig_transitions(lot, title),
        "edges": lambda: fig_edges(lot, title),
        "currents": lambda: fig_currents(lot, title, setpoints),
        "shorts": lambda: fig_shorts(lot, title),
    }
    written = []
    for name in LOT_FIGURES:
        path = out_dir / f"{prefix}_{name}.png"
        fig = makers[name]()
        if fig is None:
            path.unlink(missing_ok=True)
            continue
        _save(fig, path, note, dpi)
        written.append(path)
    return written


def plot_lots(out_dir, lots, note="", dpi=130):
    """Write lots.png comparing the lots ({label: lot rows}); returns its path."""
    path = out_dir / "lots.png"
    _save(fig_lots(lots), path, note, dpi)
    return path
