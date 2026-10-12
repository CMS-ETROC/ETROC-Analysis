"""lot_compare.py -- what UBM and bumping did to the wafers of a lot, from
the two stages of each wafer, pre_ubm and post_ubm, read as plot_wafer.py
reads them (the newest run of each die); plot_lots.py writes the table
(lot_dies.csv) and draws the figures of lot_compare_plots.py.

A lot here is a set of wafers that went through UBM and bumping together:
by default the wafers of one batch, but wafers of one batch sent for UBM
at different times are separate lots (plot_lots.py --lot). Each die of
every wafer gets one row: its grade in each stage and what happened to it
(transition_class), its rail currents in each stage and their change
(stage_compare.compare), and whether it sits on the edge ring of the wafer
map (edge_ring).

Like with like: the quick test calibrates the 9 pixels of QUICK_PIXELS
(8 before 2026-09-22, and in the February 2026 import, which has none on
the dies without baselines), the full scan (--doFullScan) all 256, either
may run QInj (--doQinj) and burn or verify the eFuse, and the high-power
short check came in on 2026-09-26, so a die's run in one stage can test
more than its run in the other, and fail on what the other never looked
at. Each die is graded in both stages with the test both of its runs made
(like_with_like, common_grade): a run loses its zero pixels outside the
pixels the other calibrated or carried over from another test
(calibrated), its QInj stage where the other had none, and its eFuse record where the other neither burned nor
verified; a run without the high-power check gets one made from its own
high-power currents against the other run's limits. The measured grade of
a die so regraded is kept in measured_pre or measured_post. A die left
stuck past its time limit (TEST_FAILURE) keeps its grade unless it was
stuck in a QInj stage taken out: where a longer test stopped says nothing
of where a shorter one would have. I2C findings compare as recorded: the
February import's I2C record is its campaign report's findings
(stage_compare). Where both runs made the same test, the grades are the
station's.
"""
import copy
import json
import math
import re

import numpy as np
import pandas as pd

from stage_compare import MAIN_RAILS, STAGE_ORDER, compare
from station import QUICK_PIXELS, grade, zero_pixels
from wafer_tables import UNTESTED, _pixel, collect

ALL_PIXELS = frozenset((r, c) for r in range(16) for c in range(16))
TAGS = dict(zip(STAGE_ORDER, ("pre", "post")))

# what happened to a die, in the order the figures stack them
CLASSES = ("passed both", "recovered", "new analog short", "new digital short", "new short, other or unknown rail",
           "new I2C NACK", "new I2C pixel or register failure", "new BL/NW = 0", "new other failure",
           "failed both", "untested in a stage")
NEW_FAILURES = tuple(c for c in CLASSES if c.startswith("new "))
# the new failures named by the grade after UBM, a short aside
NEW_BY_GRADE = {"I2C_NACK": "new I2C NACK", "I2C_PIXELS": "new I2C pixel or register failure",
                "BL_NW_ZERO": "new BL/NW = 0"}
STAGE_COLUMNS = ("detail", "run", "start", "fullscan", "qinj", "imported",
                 "analog_V_on", "analog_V_high", "digital_V_on", "digital_V_high")


def calibrated(summary):
    """The pixels whose BL or NW a run's summary.json holds: the pixels its
    calibration read, and any carried over from another test
    (calibration.carried_over: the February import's bl_nw_zero finding
    of a pixel the full scan before UBM read 0 and the quick test after
    did not read), which the run counts as zero."""
    carried = (summary.get("calibration") or {}).get("carried_over") or []
    return _read(summary) | frozenset(tuple(_pixel(p)) for p in carried)


def _read(summary):
    """The pixels a run's summary.json says it calibrated: all 256 for a
    full scan; else by its calibration record's n_pixels, QUICK_PIXELS for
    9 and their first 8 for 8 (the quick test before (15, 15) came in on
    2026-09-22, as in the February 2026 import); none for a run completed
    without a record (a February die without baselines); for a run stopped
    before calibrating, the pixels its test would have calibrated."""
    n = (summary.get("calibration") or {}).get("n_pixels")
    if summary.get("fullscan") or n == len(ALL_PIXELS):
        return ALL_PIXELS
    if n:
        if n not in (8, 9):
            raise ValueError(f"a quick test of {n} pixels: which ones is not known")
        return frozenset(QUICK_PIXELS[:n])
    if summary.get("status") == "completed":
        return frozenset()
    return frozenset(QUICK_PIXELS[:8] if summary.get("imported") else QUICK_PIXELS)


def _efuse(summary):
    return bool(summary.get("efuse_burn") or summary.get("efuse_verify"))


def common_grade(summary, other, high=None):
    """The station's Grade for a run's summary.json (summary) as the test
    both it and the other stage's run of the die (other, its summary.json)
    made would have given it: the calibration's zero pixels outside the
    pixels other calibrated dropped; without QInj in other, the QInj stage
    taken out, a run that failed or was stopped in it counting as
    completed; without an eFuse burn or verify in other, the eFuse record
    dropped; and where other made the high-power short check and this run,
    having reached high power, did not, one made from its high-power
    currents, high ({rail: amps}), against other's limits."""
    s = copy.deepcopy(summary)
    calibration = s.get("calibration")
    if calibration:
        keep = calibrated(other)
        calibration["zero_pixels"] = [p for p in zero_pixels(s) if tuple(_pixel(p)) in keep]
        calibration["n_pixels"] = len(keep & calibrated(summary))
    phases = s.get("phases") or {}
    if "qinj_start" in phases and not other.get("qinj"):
        phases.pop("qinj_start")
        s.pop("qinj_check", None)
        if s.get("status") in ("failed", "stuck"):
            s["status"] = "completed"
    if not _efuse(other):
        s.pop("efuse", None)
    limits = {rail: c["abort_above"] for rail, c in (other.get("short_check_high") or {}).items()
              if c.get("abort_above") is not None}
    if limits and "short_check_high" not in s and "high_power" in phases and s.get("status") != "over_current":
        s["short_check_high"] = {rail: {"current": amps, "abort_above": limits[rail], "ok": amps < limits[rail],
                                        "fault": "over_current"}
                                 for rail, amps in (high or {}).items() if rail in limits and amps == amps}
        if not all(c["ok"] for c in s["short_check_high"].values()):
            s["status"] = "over_current"
    return grade(s)


def common_text(a, b):
    """The test two runs' summary.json (a, b) both made: "8 pixels, no
    QInj" and the like."""
    n = len(calibrated(a) & calibrated(b))
    parts = [f"{n} pixels" if n else "no pixels", "QInj" if a.get("qinj") and b.get("qinj") else "no QInj"]
    if _efuse(a) != _efuse(b):
        parts.append("no eFuse")
    if bool(a.get("short_check_high")) != bool(b.get("short_check_high")):
        parts.append("high-power check from the currents")
    return ", ".join(parts)


def _tested(dies):
    """{die: its row index} for the dies with a run that tested them (not
    UNTESTED)."""
    if "run" not in dies:
        return {}
    ran = dies[dies["run"].notna() & ~dies["grade"].isin(UNTESTED)]
    return dict(zip(ran["die"], ran.index))


def like_with_like(stage_dirs, tables):
    """Grade every die tested in both stages as the test both runs made
    would have (common_grade), in the dies tables of `tables` ({stage:
    (dies, pixels, qinj)}), in place. stage_dirs maps each stage to its
    folder. Returns [(stage, die, measured grade, regraded, common_text of
    the common test)] for the dies whose grade changed."""
    changed = []
    tested = {stage: _tested(tables[stage][0]) for stage in STAGE_ORDER}
    for die in sorted(set(tested[STAGE_ORDER[0]]) & set(tested[STAGE_ORDER[1]])):
        runs = {}
        for stage in STAGE_ORDER:
            run = tables[stage][0].at[tested[stage][die], "run"]
            runs[stage] = json.loads((stage_dirs[stage] / f"die{die:03d}" / run / "summary.json").read_text())
        for stage, other in zip(STAGE_ORDER, reversed(STAGE_ORDER)):
            dies, i = tables[stage][0], tested[stage][die]
            high = {rail: dies.at[i, f"{rail}_I_high"] for rail in MAIN_RAILS if f"{rail}_I_high" in dies}
            g = common_grade(runs[stage], runs[other], high)
            if g.name != dies.at[i, "grade"]:
                changed.append((stage, die, dies.at[i, "grade"], g.name, common_text(*runs.values())))
                dies.loc[i, ["grade", "bin", "detail"]] = [g.name, g.bin, g.detail]
    return changed


def short_rails(detail):
    """The rails a POWER_SHORT grade's detail names, in its order:
    "analog_650mA-digital_275mA_high" -> ["analog", "digital"]."""
    return re.findall(r"(?:^|-)([a-z_]+?)_\d+mA", detail or "")


def transition_class(pre, post, detail_post=""):
    """What happened to a die between its grade before UBM (pre) and after
    (post), one of CLASSES. A new short is an analog one when its detail
    names the analog rail, whatever else it names, else a digital one when
    it names the digital rail; any other new failure is named by its grade
    after UBM (NEW_BY_GRADE), else counted as a new other failure."""
    if pre in UNTESTED or post in UNTESTED:
        return "untested in a stage"
    if pre != "PASSED":
        return "recovered" if post == "PASSED" else "failed both"
    if post == "PASSED":
        return "passed both"
    if post != "POWER_SHORT":
        return NEW_BY_GRADE.get(post, "new other failure")
    rails = short_rails(detail_post)
    if "analog" in rails:
        return "new analog short"
    return "new digital short" if "digital" in rails else "new short, other or unknown rail"


def edge_ring(wafer_map):
    """The dies of the wafer map's outer ring: dies with at least one of
    the four neighbouring positions off the map."""
    cells = set(wafer_map.values())
    return {die for die, (r, c) in wafer_map.items()
            if any((r + dr, c + dc) not in cells for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))}


def wafer_rows(wafer_dir, wafer_map, invalid=()):
    """(changes, setpoints, regraded, warnings) for one wafer folder with
    both stages: the stage_compare changes table of its dies, graded like
    with like (like_with_like), with the grade details, run, scan type and
    power-on and high-power rail voltages of each stage, the measured grade
    of a regraded die (measured_<pre|post>, "" for the others), the
    transition class and edge_ring; the median rail voltages of each stage
    (stage_compare.setpoints); the regraded dies; the collect warnings."""
    tables, warnings = {}, []
    for stage in STAGE_ORDER:
        dies, pixels, qinj, warn = collect(wafer_dir / stage, wafer_map, with_qinj=False, invalid=invalid)
        tables[stage] = (dies, pixels, qinj)
        warnings += [f"{stage}: {w}" for w in warn]
    regraded = like_with_like({s: wafer_dir / s for s in STAGE_ORDER}, tables)
    changes, _, _, volts = compare(tables)
    for stage, tag in TAGS.items():
        dies = tables[stage][0].set_index("die")
        for column in STAGE_COLUMNS:
            if column in dies:
                changes[f"{column}_{tag}"] = changes["die"].map(dies[column])
        measured = {die: m for s, die, m, *_ in regraded if s == stage}
        changes[f"measured_{tag}"] = changes["die"].map(measured).fillna("")
    changes["transition"] = [transition_class(a, b, d) for a, b, d in
                             zip(changes["grade_pre"], changes["grade_post"],
                                 changes["detail_post"] if "detail_post" in changes else [""] * len(changes))]
    changes["edge_ring"] = changes["die"].isin(edge_ring(wafer_map))
    return changes, volts, regraded, warnings


def at_risk(lot):
    """The valid dies PASSED before UBM and tested after it: the ones UBM
    could make fail."""
    return lot[~lot["invalid"].astype(bool) & (lot["grade_pre"] == "PASSED") & ~lot["grade_post"].isin(UNTESTED)]


def wilson(k, n, z=1.0):
    """The Wilson score interval (lo, hi) of k successes in n trials, 68 %
    for z = 1; (nan, nan) for n = 0. The bounds hold k / n, which rounding
    alone would move them past at k = 0 and k = n."""
    if n == 0:
        return math.nan, math.nan
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, min(centre - half, p)), min(1.0, max(centre + half, p))


def profile(lot, by, classes=NEW_FAILURES):
    """Per value of the column `by` (die_row, die_col, edge_ring, lot, ...)
    over the dies at risk (at_risk) of all the lot's wafers: n of them, k
    of them in `classes`, the fraction k / n and its 68 % Wilson interval
    (lo, hi)."""
    risk = at_risk(lot)
    rows = []
    for value, g in risk.groupby(by, sort=True):
        n, k = len(g), int(g["transition"].isin(classes).sum())
        lo, hi = wilson(k, n)
        rows.append({by: value, "n": n, "k": k, "fraction": k / n, "lo": lo, "hi": hi})
    return pd.DataFrame(rows, columns=[by, "n", "k", "fraction", "lo", "hi"]).set_index(by)


def relative_change(lot, column):
    """lot[column] minus its median over the same wafer's dies: a current
    change with each wafer's own shift (setpoints, supply) taken out."""
    values = pd.to_numeric(lot[column], errors="coerce")
    return values - values.groupby(lot["wafer"]).transform("median")


def _ran(lot, tag):
    """The rows of the dies a run tested in the stage `tag` (pre or post)."""
    return (lot[lot[f"run_{tag}"].notna() & ~lot[f"grade_{tag}"].isin(UNTESTED)] if f"run_{tag}" in lot
            else lot.iloc[0:0])


def _days(lot, tag):
    return pd.to_datetime(_ran(lot, tag).get(f"start_{tag}"), errors="coerce").dropna()


def stage_months(lot, tag):
    """When the dies of a stage were tested: "MM/YYYY", or "MM/YYYY-MM/YYYY"
    from the first month to the last; "?" without dates."""
    days = _days(lot, tag)
    if days.empty:
        return "?"
    first, last = f"{days.min():%m/%Y}", f"{days.max():%m/%Y}"
    return first if first == last else f"{first}-{last}"


def lot_batch(lot):
    """The batch name of the lot's wafers ("N62C72")."""
    return "+".join(dict.fromkeys(b.split("_Name_")[-1] for b in lot["batch"]))


def lot_name(lot):
    """The lot's batch and when each stage was tested, as the figures name
    it: "N62C72, pre-UBM test 09/2025-10/2025, post-UBM test 02/2026"."""
    return f"{lot_batch(lot)}, pre-UBM test {stage_months(lot, 'pre')}, post-UBM test {stage_months(lot, 'post')}"


def lot_names(lots):
    """{label: lot_name} for the lots ({label: lot rows}), the label added
    to the names two lots share."""
    names = {label: lot_name(lot) for label, lot in lots.items()}
    shared = {name for name in names.values() if list(names.values()).count(name) > 1}
    return {label: name + (f" ({label})" if name in shared else "") for label, name in names.items()}


def chronology(lot):
    """A sort key putting lots in the order they were tested: by the first
    test after UBM, then the first before (lots without dates last)."""
    return tuple(_days(lot, tag).min() if len(_days(lot, tag)) else pd.Timestamp.max for tag in ("post", "pre"))


def stage_text(lot, tag):
    """How a stage of the lot was tested: "116 quick tests, 0 with QInj
    (116 imported), 2026-02-10 to 2026-02-19", over the dies a run tested."""
    ran = _ran(lot, tag)
    if ran.empty:
        return "no runs"
    full = ran[f"fullscan_{tag}"].fillna(False).astype(bool)
    parts = [f"{n} {name}" for n, name in ((int(full.sum()), "full scans"), (int((~full).sum()), "quick tests")) if n]
    qinj = int(ran[f"qinj_{tag}"].fillna(False).astype(bool).sum()) if f"qinj_{tag}" in ran else 0
    text = ", ".join(parts) + f", {qinj} with QInj"
    if f"imported_{tag}" in ran:
        imported = int(ran[f"imported_{tag}"].fillna(False).astype(bool).sum())
        if imported:
            text += f" ({imported} imported)"
    days = pd.to_datetime(ran.get(f"start_{tag}"), errors="coerce").dropna()
    if len(days):
        first, last = days.min().date(), days.max().date()
        text += f", {first}" + (f" to {last}" if last != first else "")
    return text


def counts(lot):
    """{class: number of valid dies} over the lot, every class present."""
    valid = lot[~lot["invalid"].astype(bool)]
    return {c: int((valid["transition"] == c).sum()) for c in CLASSES}


def passed(lot, tag):
    """The valid dies PASSED in the stage `tag` (pre or post)."""
    return int(((lot[f"grade_{tag}"] == "PASSED") & ~lot["invalid"].astype(bool)).sum())


def valid_positions(lot):
    """One row per die position of the lot's wafer map (die, die_row,
    die_col, invalid), for the lot maps."""
    return lot.drop_duplicates("die").sort_values("die")[["die", "die_row", "die_col", "invalid"]].reset_index(drop=True)


def per_position(lot, classes, positions):
    """positions-ordered: the number of the lot's wafers on which the die
    at each position is in `classes`."""
    k = lot[lot["transition"].isin(classes)].groupby("die").size()
    return positions["die"].map(k).fillna(0).to_numpy(dtype=float)


def median_by_position(lot, values, positions):
    """positions-ordered: the median of `values` (one per lot row) over the
    wafers, NaN where no wafer has one."""
    s = pd.Series(np.asarray(values, dtype=float), index=lot.index)
    med = s.groupby(lot["die"]).median()
    return positions["die"].map(med).to_numpy(dtype=float)
