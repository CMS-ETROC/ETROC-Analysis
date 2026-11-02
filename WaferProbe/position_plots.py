"""position_plots.py -- the figures of whether the dies at one place on the
wafer behave alike on the wafers of a lot (position_compare.py;
plot_positions.py runs both), for one stage:

  position_maps    per die-level quantity, the median over the wafers of
                   each die's value with its wafer's median subtracted, at
                   the die's place, with the mean pairwise r and its range
                   with the positions shuffled
  position_pairs   per quantity, the pairwise r of every two wafers, and
                   each wafer's values against the die number, the order
                   the dies were tested in
  pixel_patterns   per pixel quantity, the pattern every chip shares, the
                   map r of dies at the same place on two wafers against
                   dies at different places (and the same die in the other
                   stage, and the same place with the tilt across each die
                   taken out), the median same-place map r at each place,
                   and the tilt across the die at each place

The maps follow wafer_plots.py: row 0 at the top, the wafer seen notch up
as on the station's map display, invalid dies grey with INVALID across.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from lot_compare import median_by_position  # noqa: E402
from position_compare import DIE_QUANTITIES, MAP_QUANTITIES, centred, measured  # noqa: E402
from wafer_plots import _colourbar, _pixel_axes, _suptitle, add_note, number, value_range, wafer_map  # noqa: E402

FIGURES = ("position_maps", "position_pairs", "pixel_patterns")
UNITS = {"baseline": "DAC code", "noise_width": "DAC code", "cal": "TDC code", "toa": "TDC code",
         "tot": "TDC code"}
FORMATS = {"baseline": "{:+.0f}", "noise_width": "{:+.2f}", "cal": "{:+.1f}", "toa": "{:+.1f}",
           "tot": "{:+.1f}"}
WAFER_COLOURS = plt.get_cmap("tab20").colors
PIXEL_FRAME = "each die seen notch up: pixel (0, 0) top left (bottom right in the chip's own frame)"
R_BINS = np.linspace(-0.6, 1.0, 81)


def _shuffle_text(shuffled):
    lo, hi = np.nanpercentile(shuffled, [2.5, 97.5])
    return f"shuffled positions: 95 % within {number('{:+.2f}', lo)} to {number('{:+.2f}', hi)}"


def fig_position_maps(values, positions, stats, title):
    """One wafer map per quantity at least two wafers measured."""
    quantities = measured(values)
    if not quantities:
        return None
    ncols = min(3, len(quantities))
    nrows = -(-len(quantities) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 6.6 * nrows), squeeze=False, layout="constrained")
    for ax, q in zip(axes.flat, quantities):
        n = values.loc[np.isfinite(values[q].astype(float)), "wafer"].nunique()
        mean, shuffled = stats[q]
        wafer_map(ax, positions, median_by_position(values, centred(values, q), positions),
                  title=f"{DIE_QUANTITIES[q]} minus its wafer's median, median over {n} wafers\n"
                        f"pairwise r {number('{:+.2f}', mean)} ({_shuffle_text(shuffled)})",
                  label=UNITS[q], fmt=FORMATS[q], cmap="RdBu_r", centre=0.0)
    for ax in axes.flat[len(quantities):]:
        ax.axis("off")
    _suptitle(fig, title, "do the dies at one place behave alike on every wafer?\n"
                          "each die's value minus its wafer's median over its dies (offsets between wafers and "
                          "test setups taken out); hatched: no wafer measured it\npairwise r: the correlation "
                          "of two wafers' values over the places both measured, mean over the pairs",
              fontsize=11)
    return fig


def fig_position_pairs(values, matrices, title):
    """Per quantity: the pairwise r matrix and each wafer's centred values
    against the die number."""
    quantities = measured(values)
    if not quantities:
        return None
    wafers = list(dict.fromkeys(values["wafer"]))
    colours = {w: WAFER_COLOURS[k % len(WAFER_COLOURS)] for k, w in enumerate(wafers)}
    size = max(4.5, 0.42 * len(wafers) + 1.5)
    fig, axes = plt.subplots(len(quantities), 2, figsize=(size + 10, max(size, 4.2) * len(quantities)),
                             squeeze=False, layout="constrained", gridspec_kw={"width_ratios": [size, 10]})
    for (ax_r, ax_v), q in zip(axes, quantities):
        r = matrices[q].reindex(index=wafers, columns=wafers)
        ax_r.imshow(r.to_numpy(float), cmap="RdBu_r", vmin=-1, vmax=1)
        for i in range(len(wafers)):
            for j in range(len(wafers)):
                v = r.iat[i, j]
                if i != j and np.isfinite(v):
                    ax_r.text(j, i, number("{:.2f}", v).replace("0.", "."), ha="center", va="center", fontsize=6.5,
                              color="white" if abs(v) > 0.6 else "black")
        ax_r.set_xticks(range(len(wafers)))
        ax_r.set_xticklabels(wafers, rotation=90, fontsize=7)
        ax_r.set_yticks(range(len(wafers)))
        ax_r.set_yticklabels(wafers, fontsize=7)
        ax_r.set_title(f"{DIE_QUANTITIES[q]}: pairwise r of two wafers", fontsize=10)
        c = centred(values, q)
        for w in wafers:
            sel = (values["wafer"] == w).to_numpy() & np.isfinite(c.to_numpy())
            order = np.argsort(values.loc[sel, "die"].to_numpy())
            ax_v.plot(values.loc[sel, "die"].to_numpy()[order], c.to_numpy()[sel][order], "-o", ms=2.2, lw=0.6,
                      color=colours[w], label=w)
        ax_v.set_ylim(*value_range(c.to_numpy(), 0.0))
        ax_v.axhline(0, color="#999999", lw=0.6)
        ax_v.set_xlabel("die number = test order in each pass (rows of the wafer map from the top, serpentine)",
                        fontsize=8)
        ax_v.set_ylabel(f"{DIE_QUANTITIES[q]} minus the wafer's median ({UNITS[q]})", fontsize=8)
        ax_v.tick_params(labelsize=7)
        ax_v.set_title(f"{DIE_QUANTITIES[q]} against the order the dies were tested in: a drift during a pass "
                       "would be a trend here", fontsize=10)
    axes[0, 1].legend(fontsize=7, ncol=2, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _suptitle(fig, title, "the pairwise r of every two wafers, and each wafer's values in test order", fontsize=11)
    return fig


def fig_pixel_patterns(maps, positions, title):
    """Per pixel quantity (maps {quantity: plot_positions' dict of pattern,
    same, other, by_die, same_die, same_untilted, tilt}): the shared
    pattern, the map r histograms, the median same-place map r at each
    place and the tilt at each place."""
    quantities = [q for q in MAP_QUANTITIES if q in maps and len(maps[q]["same"])]
    if not quantities:
        return None
    fig, axes = plt.subplots(len(quantities), 4, figsize=(27, 6.4 * len(quantities)), squeeze=False,
                             layout="constrained", gridspec_kw={"width_ratios": [5.5, 7, 7, 7]})
    for (ax_p, ax_h, ax_m, ax_t), q in zip(axes, quantities):
        m = maps[q]
        norm = Normalize(*value_range(m["pattern"].ravel(), 0.0))
        ax_p.imshow(m["pattern"], cmap="RdBu_r", norm=norm, interpolation="nearest")
        _pixel_axes(ax_p)
        _colourbar(fig, ax_p, norm, plt.get_cmap("RdBu_r"), f"{MAP_QUANTITIES[q]} minus the die's mean "
                   f"({UNITS[q]})", m["pattern"].ravel())
        ax_p.set_title(f"{MAP_QUANTITIES[q]}: the pattern every chip shares\n(median over the dies of each "
                       "pixel minus its die's mean)", fontsize=10)
        series = [(m["same"], "same place, two wafers", "pairs", "#d62728", "-"),
                  (m["same_untilted"], "same place, tilt across each die taken out", "pairs", "#d62728", ":"),
                  (m["other"], "different places, two wafers", "pairs", "#7f7f7f", "-")]
        if len(m["same_die"]):
            series.append((m["same_die"], "same die, other stage", "dies", "#1f77b4", "-"))
        for v, label, unit, colour, style in series:
            if not len(v):
                continue
            ax_h.hist(v, bins=R_BINS, density=True, histtype="step", lw=1.6, color=colour, ls=style,
                      label=f"{label}: median {number('{:+.2f}', np.median(v))} ({len(v)} {unit})")
            ax_h.axvline(np.median(v), color=colour, lw=0.8, ls="--")
        ax_h.set_ylim(0, ax_h.get_ylim()[1] * 1.45)   # room for the legend above the peaks
        ax_h.set_xlabel("map r: correlation of two dies' pixels, after their means and the shared pattern are "
                        "taken out", fontsize=8)
        ax_h.set_ylabel("pairs (normalised)", fontsize=8)
        ax_h.tick_params(labelsize=7)
        ax_h.legend(fontsize=7.5, frameon=False, loc="upper left")
        ax_h.set_title(f"{MAP_QUANTITIES[q]}: do two dies' pixel maps look alike?", fontsize=10)
        wafer_map(ax_m, positions, positions["die"].map(m["by_die"]).to_numpy(float),
                  title=f"{MAP_QUANTITIES[q]}: median same-place map r at each place", label="map r",
                  fmt="{:.2f}", cmap="RdBu_r", vrange=(-1.0, 1.0))
        tilt = m["tilt"].reindex(positions["die"])
        size = np.hypot(tilt["col"], tilt["row"]).to_numpy(float)
        wafer_map(ax_t, positions, 15 * size, title=f"{MAP_QUANTITIES[q]}: tilt across each die, median over the "
                  f"wafers; arrow: the way it rises\ncosine with the outward radius: median "
                  f"{number('{:+.2f}', tilt['outward'].median())} over {int(np.isfinite(tilt['outward']).sum())} places",
                  label=f"rise across the die, pixel 0 to 15 ({UNITS[q]})", fmt="", cmap="YlOrRd",
                  vrange=(0.0, max(value_range(15 * size)[1], 1e-6)))
        draw = np.isfinite(size) & (size > 0) & ~positions["invalid"].to_numpy(bool)
        ax_t.quiver(positions["die_col"].to_numpy(float)[draw], positions["die_row"].to_numpy(float)[draw],
                    (tilt["col"].to_numpy(float) / size)[draw], (tilt["row"].to_numpy(float) / size)[draw],
                    angles="xy", scale_units="xy", scale=1 / 0.7, pivot="middle", width=0.005, color="black")
    _suptitle(fig, title, "do the dies at one place have the same pixel map on every wafer?\n"
                          f"full scans only, pixels reading 0 left out; pixel maps {PIXEL_FRAME}; tilt: the "
                          "least-squares plane over each die's pixels, after the shared pattern", fontsize=11)
    return fig


def _save(fig, path, note, dpi):
    if note:
        add_note(fig, note, wrap=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_positions(out_dir, prefix, values, positions, stats, matrices, maps, title, note="", dpi=130):
    """Write the figures of one lot and stage into out_dir as
    <prefix>_<figure>.png, removing the older copy of one the data do not
    support; returns the paths written."""
    makers = {"position_maps": lambda: fig_position_maps(values, positions, stats, title),
              "position_pairs": lambda: fig_position_pairs(values, matrices, title),
              "pixel_patterns": lambda: fig_pixel_patterns(maps, positions, title)}
    written = []
    for name in FIGURES:
        path = out_dir / f"{prefix}_{name}.png"
        fig = makers[name]()
        if fig is None:
            path.unlink(missing_ok=True)
            continue
        _save(fig, path, note, dpi)
        written.append(path)
    return written
