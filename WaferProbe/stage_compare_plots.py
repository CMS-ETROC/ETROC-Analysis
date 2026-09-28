"""stage_compare_plots.py -- the pre_ubm vs post_ubm figures, drawn from
the tables of stage_compare.py. One PNG per figure; a figure whose data
one of the stages did not take (no pixels calibrated in both, no QInj in
both) is left out, and an older copy of it removed. The wafer maps follow
wafer_plots.py: row 0 at the top, invalid dies grey with INVALID across,
a die without a value hatched.
"""
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from stage_compare import (MAIN_RAILS, QINJ_QUANTITIES, SAME_CHIP_R, STAGE_ORDER, qinj_compared,  # noqa: E402
                           transitions, unchecked_text)
from wafer_plots import (GRADE_COLOURS, _edges, _ink, _invalid, _invalid_cell, _num, _shape,  # noqa: E402
                         _suptitle, _wafer_axes, add_note, wafer_map)
from wafer_tables import UNTESTED  # noqa: E402

FIGURES = ("grade_changes", "current_changes", "baseline_changes", "qinj_changes")

# one letter per grade for the before -> after text in a changed die's cell
GRADE_LETTERS = {
    "PASSED": "P", "POWER_SHORT": "S", "I2C_NACK": "N", "I2C_PIXELS": "X", "NO_LINK_OR_DATA": "L",
    "OTHER_FAIL": "O", "NOT_TESTED": "-", "RAIL_OPEN": "R", "BL_NW_ZERO": "Z", "EFUSE_FAIL": "E",
    "EFUSE_TRAILER_FAIL": "T", "TEST_FAILURE": "F", "CONTACT_FAILURE": "C",
}
EFF_DROP = 0.01  # a die whose lowest pixel efficiency fell by more than this is named


def _stages_text(setpoints):
    """"analog 1.363 → 1.363 V, digital 1.255 → 1.298 V": the median
    power-on voltage of each rail, pre → post."""
    pre, post = (setpoints.get(s, {}) for s in STAGE_ORDER)
    return ", ".join(f"{rail} {pre[rail]:.3f} → {post[rail]:.3f} V" for rail in MAIN_RAILS
                     if rail in pre and rail in post)


def fig_grade_changes(changes, title):
    fig, ax = plt.subplots(figsize=(10.4, 6.8), layout="constrained")
    _wafer_axes(ax, _shape(changes))
    for d, out in zip(changes.itertuples(index=False), _invalid(changes)):
        if out:
            _invalid_cell(ax, d.die_col, d.die_row)
            continue
        colour = GRADE_COLOURS.get(d.grade_post, "white")
        ax.add_patch(Rectangle((d.die_col - .5, d.die_row - .5), 1, 1, facecolor=colour,
                               edgecolor="white", lw=0.8))
        text = str(d.die)
        if d.changed:
            ax.add_patch(Rectangle((d.die_col - .44, d.die_row - .44), 0.88, 0.88, fill=False,
                                   edgecolor="black", lw=1.6))
            text += f"\n{GRADE_LETTERS.get(d.grade_pre, '?')}→{GRADE_LETTERS.get(d.grade_post, '?')}"
        ax.text(d.die_col, d.die_row, text, ha="center", va="center", fontsize=6.5, color=_ink(colour))
    valid = changes[~_invalid(changes)]
    counts = valid["grade_post"].value_counts()
    handles = [Patch(facecolor=GRADE_COLOURS.get(g, "white"),
                     label=f"{GRADE_LETTERS.get(g, '?')} {g}: {counts.get(g, 0)} after "
                           f"({int((valid['grade_pre'] == g).sum())} before)")
               for g in GRADE_COLOURS if g in counts or (valid["grade_pre"] == g).any()]
    moves = transitions(changes)
    for before, after, dies in moves:
        label = textwrap.fill(f"{before} → {after} ({len(dies)}): " + ", ".join(map(str, dies)), 46,
                              subsequent_indent="   ")
        handles.append(Patch(facecolor="none", edgecolor="black", label=label))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=7.5)
    passed = [int((valid[f"grade_{s}"] == "PASSED").sum()) for s in ("pre", "post")]
    tested = [int((~valid[f"grade_{s}"].isin(UNTESTED)).sum()) for s in ("pre", "post")]
    unchecked = [(s, unchecked_text(changes, tag)) for s, tag in zip(STAGE_ORDER, ("pre", "post"))]
    ax.set_title(f"{title}: grade per die after UBM, PASSED {passed[0]} of {tested[0]} before, "
                 f"{passed[1]} of {tested[1]} after\n"
                 f"framed = grade changed, with before → after; {sum(len(d) for *_, d in moves)} valid dies "
                 "changed"
                 + "".join(f"\n{s}: {text}" for s, text in unchecked if text), fontsize=10)
    return fig


def fig_current_changes(changes, title, setpoints):
    rails = [r for r in MAIN_RAILS if np.isfinite(_num(changes, f"d_{r}_I_on")).any()]
    if not rails:
        return None
    fig, axes = plt.subplots(len(rails), 3, figsize=(17, 5.0 * len(rails)), squeeze=False,
                             layout="constrained")
    for i, rail in enumerate(rails):
        for j, (phase, name) in enumerate((("on", "power-on"), ("high", "high-power"))):
            wafer_map(axes[i, j], changes, _num(changes, f"d_{rail}_I_{phase}") * 1e3,
                      title=f"{rail}: {name} current, after minus before", label="mA", fmt="{:+.0f}",
                      cmap="RdBu_r", centre=0.0)
        ax = axes[i, 2]
        for phase, name, colour in (("on", "power-on", "#1f77b4"), ("high", "high power", "#ff7f0e")):
            keep = np.isfinite(_num(changes, f"d_{rail}_I_{phase}")) & ~_invalid(changes)
            x = _num(changes, f"{rail}_I_{phase}_pre") * 1e3
            y = _num(changes, f"{rail}_I_{phase}_post") * 1e3
            if keep.any():
                ax.scatter(x[keep], y[keep], s=12, color=colour, alpha=0.8,
                           label=f"{name}: {int(keep.sum())} dies, median change "
                                 f"{np.median(y[keep] - x[keep]):+.1f} mA")
        lo, hi = ax.get_xlim()
        lo, hi = min(lo, ax.get_ylim()[0]), max(hi, ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], color="#777777", lw=1, ls="--", label="after = before")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{rail} current before UBM (mA)", fontsize=9)
        ax.set_ylabel(f"{rail} current after UBM (mA)", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ax.set_title(f"{rail}: dies PASSED before and after", fontsize=10)
    volts = _stages_text(setpoints)
    _suptitle(fig, title, "rail current changes, dies PASSED in both stages with the power phases logged in both"
                          + (f"\nrail voltages (median at power-on) before → after: {volts}" if volts else ""),
              fontsize=12)
    return fig


def fig_baseline_changes(changes, pixel_rows, title):
    if pixel_rows.empty:
        return None
    valid = ~_invalid(changes)
    flag = changes["same_chip_flag"].fillna(False).to_numpy(dtype=bool) & valid
    n = changes.loc[(changes["n_common_pixels"] > 0) & valid, "n_common_pixels"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.2), layout="constrained")
    for ax, column, what in ((axes[0, 0], "d_bl_mean", "baseline"), (axes[0, 1], "d_nw_mean", "noise width")):
        wafer_map(ax, changes, _num(changes, column), title=f"{what}: mean change over the common pixels",
                  label="DAC codes", fmt="{:+.1f}", cmap="RdBu_r", centre=0.0, frame=flag)
    for ax, column, what in ((axes[1, 0], "d_baseline", "baseline"), (axes[1, 1], "d_noise_width", "noise width")):
        v = pixel_rows[column].to_numpy(dtype=float)
        ax.hist(v, bins=_edges(v), histtype="stepfilled", color="#4c72b0", alpha=0.8)
        ax.axvline(0, color="#777777", lw=1, ls="--")
        ax.set_xlabel(f"{what} after minus before (DAC codes)", fontsize=9)
        ax.set_ylabel("pixels", fontsize=9)
        ax.set_title(f"{what}: {len(v)} pixels, median {np.median(v):+.1f}", fontsize=10)
    flagged = changes.loc[flag, "die"].tolist()
    same = (f"; red frame = within-die baseline pattern r < {SAME_CHIP_R} before vs after "
            f"(another chip?): {', '.join(map(str, flagged))}" if flagged else
            f"; every die's within-die baseline pattern agrees (r ≥ {SAME_CHIP_R}) where it can be checked")
    _suptitle(fig, title, f"baseline and noise-width changes at the pixels calibrated in both stages "
                          f"({len(n)} dies, {int(n.median()) if len(n) else 0} pixels a die (median); "
                          "pixels reading 0 left out)" + same, fontsize=11)
    return fig


def fig_qinj_changes(changes, qinj_rows, title):
    both = qinj_compared(changes)
    if not both.any():
        return None
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.2), layout="constrained")
    for j, q in enumerate(QINJ_QUANTITIES):
        wafer_map(axes[0, j], changes, _num(changes, f"d_{q}_median"),
                  title=f"{q.upper()}: median change over the injected pixels", label="codes",
                  fmt="{:+.1f}", cmap="RdBu_r", centre=0.0)
        v = qinj_rows[f"d_{q}"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        ax = axes[1, j]
        ax.hist(v, bins=_edges(v), histtype="stepfilled", color="#4c72b0", alpha=0.8)
        ax.axvline(0, color="#777777", lw=1, ls="--")
        ax.set_xlabel(f"mean {q.upper()} code, after minus before", fontsize=9)
        ax.set_ylabel("pixels", fontsize=9)
        ax.set_title(f"{q.upper()}: {len(v)} pixels, median {np.median(v) if len(v) else np.nan:+.1f}",
                     fontsize=10)
    drop = changes[both & ((changes["eff_min_pre"] - changes["eff_min_post"]) > EFF_DROP).to_numpy()]
    eff = ("; lowest pixel efficiency fell by more than 1 % on die(s) "
           + ", ".join(f"{d.die} ({d.eff_min_pre:.0%} → {d.eff_min_post:.0%})" for d in drop.itertuples())
           if len(drop) else "; no die's lowest pixel efficiency fell by more than 1 %")
    _suptitle(fig, title, f"charge-injection changes at the pixels injected and hit in both stages "
                          f"({int(both.sum())} dies with QInj in both)" + eff, fontsize=11)
    return fig


def plot_changes(out_dir, changes, pixel_rows, qinj_rows, setpoints, title, note="", dpi=130):
    """Write every comparison figure the tables support into out_dir and
    remove the older copy of any they do not; returns the paths written."""
    makers = {
        "grade_changes": lambda: fig_grade_changes(changes, title),
        "current_changes": lambda: fig_current_changes(changes, title, setpoints),
        "baseline_changes": lambda: fig_baseline_changes(changes, pixel_rows, title),
        "qinj_changes": lambda: fig_qinj_changes(changes, qinj_rows, title),
    }
    written = []
    for name in FIGURES:
        path = Path(out_dir) / f"{name}.png"
        fig = makers[name]()
        if fig is None:
            path.unlink(missing_ok=True)
            continue
        if note:
            add_note(fig, note)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written
