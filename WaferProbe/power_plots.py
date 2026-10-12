"""power_plots.py -- the figures of plot_power.py, drawn from its
power_dies.csv rows: the power of each wafer's PASSED valid dies at
power-on and at high power, one panel per stage, every wafer at the same
place in both (power), and the mean power of each rail per lot and stage
(power_rails)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

from plot_power import NOMINAL_V, PHASES, POWER_RAILS, usable  # noqa: E402
from position_compare import months_tested  # noqa: E402
from stage_compare import STAGE_ORDER, STAGE_TEXT  # noqa: E402
from wafer_plots import add_note  # noqa: E402

PHASE_COLOURS = {"on": "#9ecae1", "high": "#08519c"}
RAIL_COLOURS = {"analog": "#d62728", "digital": "#1f77b4", "ws_analog": "#ff7f0e", "ws_digital": "#2ca02c",
                "vref": "#9467bd"}
# ETROC2 Reference Manual rev 0.6, Table 21: the power per chip it expects with the
# preamplifiers at their lower setting (our power-on) and their high setting (high
# power, IBSel), within +-20 % from process variation
MANUAL_W = {"on": 0.77, "high": 0.97}
MANUAL_SPREAD = 0.2
MANUAL_TEXT = ("dash-dot lines and shading: the ETROC2 manual's estimate (rev 0.6, Table 21), "
               f"{MANUAL_W['on']:g} W at the preamp's lower and {MANUAL_W['high']:g} W at its high setting, "
               f"+-{MANUAL_SPREAD:.0%}")


def _manual_estimates(ax, vertical=False):
    """The manual's two estimates as lines, their +-20 % as shading."""
    span, at = (ax.axvspan, ax.axvline) if vertical else (ax.axhspan, ax.axhline)
    for phase, watts in MANUAL_W.items():
        span(watts * (1 - MANUAL_SPREAD), watts * (1 + MANUAL_SPREAD), color=PHASE_COLOURS[phase],
             alpha=0.22 if phase == "on" else 0.10, lw=0, zorder=0)
        at(watts, color=PHASE_COLOURS[phase], ls="-.", lw=1.0, zorder=1)
LOT_GAP = 1.0
WHISKERS = (2, 98)  # percentiles, as the legend says


def _lots(power):
    return list(dict.fromkeys(power["lot"]))


def _stages(power):
    return [s for s in STAGE_ORDER if (power["stage"] == s).any()]


def _layout(power):
    """({(lot, wafer): x}, [(first x, last x, lot)]): the wafers in the
    order read, the lots apart by LOT_GAP."""
    xs, spans, x = {}, [], 0.0
    for lot in _lots(power):
        first = x
        for wafer in dict.fromkeys(power.loc[power["lot"] == lot, "wafer"]):
            xs[(lot, wafer)] = x
            x += 1
        spans.append((first, x - 1, lot))
        x += LOT_GAP
    return xs, spans


def _rails_text(rows):
    """The median analog and digital supply voltage at high power, "1.36 / 1.30 V"."""
    volts = [rows[f"{rail}_V_high"].median() for rail in ("analog", "digital")]
    return " / ".join("?" if pd.isna(v) else f"{v:.2f}" for v in volts) + " V"


def fig_power(power):
    """One panel per stage: per wafer, boxes of the power of its PASSED
    valid dies at power-on and at high power, each lot headed by the months
    its stage was tested and its supply voltages."""
    stages = _stages(power)
    xs, spans = _layout(power)
    width = max(9.0, 1.5 + 0.44 * (max(xs.values()) + 1))
    fig, axes = plt.subplots(len(stages), 1, figsize=(width, 1.0 + 3.4 * len(stages)), sharex=True, sharey=True,
                             layout="constrained", squeeze=False)
    good = usable(power)
    for ax, stage in zip(axes[:, 0], stages):
        rows = good[good["stage"] == stage]
        heading = blended_transform_factory(ax.transData, ax.transAxes)
        for (lot, wafer), x in xs.items():
            die = rows[(rows["lot"] == lot) & (rows["wafer"] == wafer)]
            ax.text(x, 0.02, die["power_high_W"].notna().sum() if len(die) else "-", transform=heading,
                    ha="center", va="bottom", fontsize=8, color="#555555")
            for k, phase in enumerate(PHASES):
                values = die[f"power_{phase}_W"].dropna()
                if values.empty:
                    continue
                ax.boxplot([values], positions=[x + (k - 0.5) * 0.38], widths=0.34, whis=WHISKERS,
                           patch_artist=True, manage_ticks=False,
                           boxprops={"facecolor": PHASE_COLOURS[phase], "edgecolor": "#333333", "lw": 0.6},
                           medianprops={"color": "black", "lw": 1.0},
                           whiskerprops={"lw": 0.6}, capprops={"lw": 0.6},
                           flierprops={"marker": ".", "markersize": 3, "markeredgecolor": "#555555"})
        for first, last, lot in spans:
            lot_rows = rows[rows["lot"] == lot]
            if lot_rows.empty:
                continue
            ax.text((first + last) / 2, 0.98, f"{lot}\n{months_tested(lot_rows)}\n{_rails_text(lot_rows)}",
                    transform=heading, ha="center", va="top", fontsize=8)
        for first, _, _ in spans[1:]:
            ax.axvline(first - (LOT_GAP + 1) / 2, color="#bbbbbb", lw=0.6)
        _manual_estimates(ax)
        for phase, watts in MANUAL_W.items():
            ax.text(1.0, watts, f" manual\n {watts:g} W", transform=blended_transform_factory(ax.transAxes, ax.transData),
                    ha="left", va="center", fontsize=8.5, color=PHASE_COLOURS[phase])
        ax.set_title(STAGE_TEXT[stage], fontsize=12, loc="center")
        ax.set_ylabel(f"power at {NOMINAL_V:g} V (W)", fontsize=10)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", color="#e5e5e5", lw=0.5)
        ax.set_axisbelow(True)
        ax.set_xticks(list(xs.values()), [wafer for _, wafer in xs], fontsize=9)
    top = max(MANUAL_W["high"] * (1 + MANUAL_SPREAD), good[[f"power_{p}_W" for p in PHASES]].max().max())
    low = good[[f"power_{p}_W" for p in PHASES]].min().min()
    axes[0, 0].set_ylim(low - 0.1 * (top - low), top + 0.45 * (top - low))
    fig.suptitle(f"Power per chip, PASSED valid dies: {NOMINAL_V:g} V x the current summed over "
                 f"{', '.join(POWER_RAILS)}", fontsize=12)
    fig.legend(handles=[Patch(facecolor=PHASE_COLOURS[p], edgecolor="#333333", lw=0.6, label=t)
                        for p, t in PHASES.items()],
               loc="outside lower center", ncol=len(PHASES), fontsize=10, frameon=False,
               title=f"boxes: quartiles and median; whiskers: 2nd to 98th percentile; small numbers: the dies "
                     "with a high-power reading; under each lot: test months, analog / digital supply voltage as read back\n"
                     + MANUAL_TEXT,
               title_fontsize=9)
    return fig


def fig_power_rails(power):
    """Per lot and stage, the mean power of each rail at power-on (pale,
    upper bar) and at high power (lower bar) over its PASSED valid dies with
    a power in both phases, stacked, the sum at the end."""
    good = usable(power).dropna(subset=[f"power_{p}_W" for p in PHASES])
    groups = [(lot, stage) for lot in _lots(power) for stage in _stages(power)
              if ((good["lot"] == lot) & (good["stage"] == stage)).any()]
    fig, ax = plt.subplots(figsize=(10.0, 1.4 + 0.62 * len(groups)), layout="constrained")
    labels = {}
    for g, (lot, stage) in enumerate(groups):
        rows = good[(good["lot"] == lot) & (good["stage"] == stage)]
        for k, phase in enumerate(PHASES):
            y = -(3 * g + k)
            left = 0.0
            for rail in POWER_RAILS:
                mean = rows[f"{rail}_P_{phase}"].mean()
                ax.barh(y, mean, left=left, height=0.85, color=RAIL_COLOURS[rail], alpha=0.45 if phase == "on" else 1.0,
                        edgecolor="white", lw=0.4)
                left += mean
            ax.text(left + 0.008, y, f"{left:.2f} W {PHASES[phase]}", ha="left", va="center", fontsize=8)
        labels[-(3 * g + 0.5)] = f"{lot}, {STAGE_TEXT[stage]}\n{months_tested(rows)}, {len(rows)} dies"
    ax.set_yticks(list(labels), list(labels.values()), fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(f"mean power at {NOMINAL_V:g} V (W)", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    _manual_estimates(ax, vertical=True)
    totals = [good[[f"{rail}_P_{phase}" for rail in POWER_RAILS]].groupby([good["lot"], good["stage"]]).mean().sum(axis=1)
              for phase in PHASES]
    ax.set_xlim(0, 1.05 * max(MANUAL_W["high"] * (1 + MANUAL_SPREAD), *(t.max() for t in totals)))
    ax.set_ylim(-(3 * len(groups) - 1) - 0.6, 0.6)
    ax.grid(axis="x", color="#e5e5e5", lw=0.5)
    ax.set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor=RAIL_COLOURS[r], label=r) for r in POWER_RAILS],
               loc="outside lower center", ncol=len(POWER_RAILS), fontsize=9, frameon=False,
               title="pale upper bar: power-on; lower bar: high power\n" + MANUAL_TEXT, title_fontsize=9)
    fig.suptitle(f"Power per rail, mean over the PASSED valid dies of each lot and stage\n({NOMINAL_V:g} V x the "
                 "current of the rail)", fontsize=12)
    return fig


def _save(fig, path, note, dpi):
    if note:
        add_note(fig, note, wrap=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_power(out_dir, power, note="", dpi=130):
    """Write power.png and power_rails.png into out_dir; returns their
    paths."""
    power = power.copy()
    power["invalid"] = power["invalid"].astype(bool)
    written = []
    for name, maker in (("power", fig_power), ("power_rails", fig_power_rails)):
        path = out_dir / f"{name}.png"
        _save(maker(power), path, note, dpi)
        written.append(path)
    return written
