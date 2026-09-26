"""stage_compare.py -- what changed on one wafer between its two stages,
pre_ubm and post_ubm, from the tables wafer_tables.collect gives for each
(plot_wafer.py writes them into <wafer folder>/pre_vs_post/ whenever both
stage folders exist; stage_compare_plots.py draws the figures).

Only what both stages measured is compared. A stage probed with the quick
test calibrated 9 pixels a die (8 in the February 2026 import, whose
baselines are not in pixels.csv at all), a full scan 256, so baselines
compare at the pixels calibrated in both stages; a stage without QInj
leaves the QInj changes empty. Currents compare for the dies that PASSED
in both stages, since a failed die's current says more about the failure
than about the change. The supply setpoints may differ between the
stages (the digital rail went from 1.255 V to 1.298 V between April and
September 2026), so the median power-on voltage of each rail is reported
beside the current changes.

Currents compare only where both stages logged the station's power
phases (a high-power current in the dies table). The February 2026
before_bump import holds one current per rail instead, the median over a
sequence of its own: 463 mA analog on the PASSED dies of N62H30 01D4,
against 299 mA at power-on and 444 mA at high power on the same dies after
UBM, so no station phase matches it. The April 2026 N62M23 runs imported
into pre_ubm were station runs with the phases logged, and compare like
any other. A stage with no I2C record (the February import) is graded on
its power readings and baselines alone (8 pixels, and none at all on five
of the ten wafers; the report's I2C findings are the only I2C record it
has): a die graded I2C_PIXELS after UBM may have been one before.
"""
import numpy as np
import pandas as pd

from wafer_tables import injected_by_die

STAGE_ORDER = ("pre_ubm", "post_ubm")
MAIN_RAILS = ("analog", "digital")
PHASES = ("on", "high")
QINJ_QUANTITIES = ("toa", "tot", "cal")

# Same-chip check: within a die the pixel-to-pixel baseline pattern is the
# chip's own. Over the 8 February pixels, read from the imported runs'
# BaselineHistory.sqlite by hand (pixels.csv does not hold them, so this
# package cannot repeat it), the post-UBM dies of N62H30 01D4 and 02C7
# (2026-09-26) correlated with their own pre-UBM baselines at median r 0.98
# and 0.97, and with the other wafer's at 0.45, so a die below SAME_CHIP_R
# is flagged. MIN_R_PIXELS common pixels are needed for an r.
SAME_CHIP_R = 0.7
MIN_R_PIXELS = 6

PIXEL_KEYS = ["die", "pix_row", "pix_col"]


def _good_pixels(pixels):
    """The pixels whose baseline and noise width were read and are not 0
    (a 0 is a finding the grade already takes, not a value)."""
    p = pixels.dropna(subset=["baseline", "noise_width"])
    return p[(p["baseline"] != 0) & (p["noise_width"] != 0)]


def pixel_changes(pre, post):
    """One row per pixel calibrated in both stages: its baseline and noise
    width in each and the change (post minus pre)."""
    cols = PIXEL_KEYS + ["baseline", "noise_width"]
    both = _good_pixels(pre)[cols].merge(_good_pixels(post)[cols], on=PIXEL_KEYS, suffixes=("_pre", "_post"))
    for q in ("baseline", "noise_width"):
        both[f"d_{q}"] = both[f"{q}_post"] - both[f"{q}_pre"]
    return both


def same_chip_r(pixel_rows):
    """Per die, the correlation over its common pixels of each pixel's
    baseline minus the die's mean baseline, pre against post; NaN with fewer
    than MIN_R_PIXELS pixels or a flat pattern in either stage."""
    out = {}
    for die, g in pixel_rows.groupby("die"):
        a = g["baseline_pre"] - g["baseline_pre"].mean()
        b = g["baseline_post"] - g["baseline_post"].mean()
        if len(g) >= MIN_R_PIXELS and a.std() > 0 and b.std() > 0:
            out[die] = float(np.corrcoef(a, b)[0, 1])
    return pd.Series(out, dtype=float)


def injected_rows(dies, qinj):
    """The rows of a stage's qinj table at the pixels each die's QInj run
    injected (wafer_tables.injected_by_die): a hit elsewhere is not an
    injected pixel's."""
    keys = {(int(die), int(r), int(c)) for die, pixels in zip(dies["die"], injected_by_die(dies, qinj))
            for r, c in pixels or ()}
    keep = [(int(d), int(r), int(c)) in keys for d, r, c in zip(qinj["die"], qinj["pix_row"], qinj["pix_col"])]
    return qinj[pd.Series(keep, index=qinj.index, dtype=bool)]


def qinj_changes(pre_dies, pre, post_dies, post):
    """One row per pixel injected and hit in both stages: efficiency and
    the mean TOA, TOT and CAL codes in each, and their changes (post minus
    pre). A pixel without a hit in a stage has no row: its efficiency 0
    shows in the dies tables' qinj_min_eff."""
    cols = PIXEL_KEYS + ["eff"] + [f"{q}_mean" for q in QINJ_QUANTITIES]
    both = injected_rows(pre_dies, pre)[cols].merge(injected_rows(post_dies, post)[cols], on=PIXEL_KEYS,
                                                    suffixes=("_pre", "_post"))
    for q in QINJ_QUANTITIES:
        both[f"d_{q}"] = both[f"{q}_mean_post"] - both[f"{q}_mean_pre"]
    return both


def setpoints(dies):
    """{rail: median power-on voltage over the PASSED dies} for the main
    rails: the supply setpoint the stage ran with."""
    ok = dies[dies["grade"] == "PASSED"]
    out = {}
    for rail in MAIN_RAILS:
        column = f"{rail}_V_on"
        if column in ok and pd.to_numeric(ok[column], errors="coerce").notna().any():
            out[rail] = float(pd.to_numeric(ok[column], errors="coerce").median())
    return out


def _column(dies, column):
    if column not in dies:
        return pd.Series(np.nan, index=dies["die"].to_numpy())
    return pd.Series(pd.to_numeric(dies[column], errors="coerce").to_numpy(), index=dies["die"].to_numpy())


def _present(dies, column):
    if column not in dies:
        return pd.Series(False, index=dies["die"].to_numpy())
    return pd.Series(dies[column].notna().to_numpy(), index=dies["die"].to_numpy())


def die_changes(pre, post, pixel_rows, qinj_rows):
    """One row per die of the wafer map: its grade in each stage, the rail
    currents in each and their change, and the per-die summary of the pixel
    and QInj changes, with each stage's lowest pixel efficiency
    (qinj_min_eff of its dies table). The current changes (d_<rail>_I_<phase>) are filled
    in for the dies PASSED in both stages whose rail has a high-power
    current in both (the power phases logged); i2c_pre / i2c_post say
    whether the die has an I2C record in that stage."""
    out = post[["die", "die_row", "die_col", "invalid"]].copy().reset_index(drop=True)
    grades = {name: dict(zip(stage["die"], stage["grade"])) for name, stage in (("pre", pre), ("post", post))}
    out["grade_pre"] = out["die"].map(grades["pre"]).fillna("NOT_TESTED")
    out["grade_post"] = out["die"].map(grades["post"]).fillna("NOT_TESTED")
    out["changed"] = out["grade_pre"] != out["grade_post"]
    both = (out["grade_pre"] == "PASSED") & (out["grade_post"] == "PASSED")
    out["passed_both"] = both
    for name, stage in (("pre", pre), ("post", post)):
        out[f"i2c_{name}"] = out["die"].map(_present(stage, "pixel_id_ok")).fillna(False).astype(bool)
    for rail in MAIN_RAILS:
        phased = (out["die"].map(_column(pre, f"{rail}_I_high")).notna()
                  & out["die"].map(_column(post, f"{rail}_I_high")).notna())
        for phase in PHASES:
            column = f"{rail}_I_{phase}"
            before = out["die"].map(_column(pre, column))
            after = out["die"].map(_column(post, column))
            out[f"{column}_pre"] = before
            out[f"{column}_post"] = after
            out[f"d_{column}"] = (after - before).where(both & phased)

    by_die = pixel_rows.groupby("die")
    out["n_common_pixels"] = out["die"].map(by_die.size()).fillna(0).astype(int)
    out["d_bl_mean"] = out["die"].map(by_die["d_baseline"].mean())
    out["d_nw_mean"] = out["die"].map(by_die["d_noise_width"].mean())
    out["same_chip_r"] = out["die"].map(same_chip_r(pixel_rows))
    out["same_chip_flag"] = out["same_chip_r"] < SAME_CHIP_R

    by_die = qinj_rows.groupby("die")
    out["n_common_qinj_pixels"] = out["die"].map(by_die.size()).fillna(0).astype(int)
    for q in QINJ_QUANTITIES:
        out[f"d_{q}_median"] = out["die"].map(by_die[f"d_{q}"].median())
    out["eff_min_pre"] = out["die"].map(_column(pre, "qinj_min_eff"))
    out["eff_min_post"] = out["die"].map(_column(post, "qinj_min_eff"))
    return out


def qinj_compared(changes):
    """Mask of the valid dies with a lowest pixel efficiency in both
    stages: QInj ran in both and its injected pixels are known."""
    return (changes["eff_min_pre"].notna() & changes["eff_min_post"].notna()
            & ~changes["invalid"].astype(bool)).to_numpy(dtype=bool)


def passed_unchecked(changes, tag):
    """The valid dies PASSED in the stage `tag` (pre or post) without an
    I2C record there: passed on their power readings and, where taken,
    baselines alone."""
    rows = changes[(changes[f"grade_{tag}"] == "PASSED") & ~changes[f"i2c_{tag}"] & ~changes["invalid"].astype(bool)]
    return sorted(rows["die"].tolist())


def transitions(changes):
    """[(grade before, grade after, [dies]), ...] for the valid dies whose
    grade changed, the most common change first."""
    moved = changes[changes["changed"] & ~changes["invalid"].astype(bool)]
    groups = [(a, b, sorted(g["die"].tolist())) for (a, b), g in moved.groupby(["grade_pre", "grade_post"])]
    return sorted(groups, key=lambda t: (-len(t[2]), t[0], t[1]))


def compare(stages):
    """(changes, pixel_rows, qinj_rows, setpoints) from {stage: (dies,
    pixels, qinj)} holding both STAGE_ORDER stages."""
    (pre_dies, pre_pix, pre_q), (post_dies, post_pix, post_q) = (stages[s] for s in STAGE_ORDER)
    pixel_rows = pixel_changes(pre_pix, post_pix)
    qinj_rows = qinj_changes(pre_dies, pre_q, post_dies, post_q)
    changes = die_changes(pre_dies, post_dies, pixel_rows, qinj_rows)
    return changes, pixel_rows, qinj_rows, {s: setpoints(stages[s][0]) for s in STAGE_ORDER}
