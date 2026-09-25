"""wafer_tables.py -- one wafer's results as three flat tables, for the
wafer figures (wafer_plots.py; plot_wafer.py runs both) and for anyone
who wants the numbers.

Reads only what master_run_script.run_die writes, under
<path>/BatchID_<id>_Name_<batch>/WaferID_<id>_Name_<wafer>/die<nnn>/run_<k>_<suffix>/:
summary.json, power.parquet, baseline.parquet and, after --doQinj, the .nem
files of the QInj run (qinj_files).

The run that stands for a die is the newest run folder holding a
summary.json, passing over runs aborted with Ctrl+C: the run whose grade
the station map shows, since a retry is newer than its first attempt. Two
exceptions: a run made by hand with master_run_script.py counts here but
never reaches the map, and a pass stopped with Ctrl+C during a die's retry
leaves the first attempt standing here but no grade on the map. With
`before`, it is the newest such run that
started before that time, to see the wafer as it stood then. Every die of
the wafer map gets a row; a die without such a run grades NOT_TESTED, as
in wafer_run.py.

  dies    one row per die: position, run, grade and map text, status and
          error, the median current and voltage of every rail in each run
          phase, the I2C verdicts, the baseline and noise-width statistics,
          the QInj verdict, event counts and lowest pixel efficiency,
          the chuck-minus-map offset at contact, and a note on a baseline
          or noise width that stands out (bl_nw_notes)
  pixels  one row per calibrated pixel: baseline and noise width
  qinj    one row per pixel with hits in the QInj data (qinj_files): hits,
          efficiency, and CAL, TOA and TOT mean and std over the hits
          with |CAL - the pixel's most common CAL value| < 3
"""
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from station import grade, result_text
from station import nem_files

# power.parquet phase -> column suffix in the dies table
PHASES = {"power_on": "on", "high_power": "high", "qinj_start": "qinj"}

# QInj hit selection, as in the 2025 engineering-wafer report:
# |CAL - the pixel's most common CAL code| < CAL_WINDOW
CAL_WINDOW = 3

# BL/NW notes (bl_nw_notes): a die whose baseline or noise width stands out
# gets a note, which the figures mark; a note never changes a grade.
NOTE_PIXEL_BL = 100   # a pixel's baseline further than this (DAC codes) from its die's median
NOTE_PIXEL_NW = 8     # a pixel's noise width further than this from its die's median
NOTE_DIE_SIGMA = 5    # a die's baseline or noise-width mean further than this many robust
                      # sigmas (1.4826 x the median absolute deviation) from the median of the
                      # PASSED dies calibrated over as many pixels (quick test or full scan)
NOTE_MIN_DIES = 20    # such PASSED dies with a mean that the die check needs
NOTE_PIXELS_LISTED = 3  # pixels a note names, the furthest from the median first
DIE_MEANS = (("bl_mean", "baseline mean", "{:.0f}"), ("nw_mean", "noise-width mean", "{:.2f}"))

PIXEL_COLUMNS = ["die", "pix_row", "pix_col", "baseline", "noise_width"]
QINJ_COLUMNS = ["die", "pix_row", "pix_col", "hits", "eff", "cal_mode", "n_sel",
                "toa_mean", "toa_std", "tot_mean", "tot_std", "cal_mean", "cal_std"]

_RUN_FOLDER = re.compile(r"run_(\d+)_")


def run_dirs(die_dir):
    """The run folders of one die, oldest first by run number (a plain
    sort would put run_100 before run_99)."""
    found = []
    for p in Path(die_dir).iterdir():
        m = _RUN_FOLDER.match(p.name)
        if m and p.is_dir():
            found.append((int(m.group(1)), p))
    return [p for _, p in sorted(found)]


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _read_parquet(path):
    """The table, or None when the run did not write it."""
    return pd.read_parquet(path) if Path(path).is_file() else None


def start_time(summary):
    """When the run started, as a datetime, or None: its earliest phase
    time, which is power_on, or power_off for a run that stopped before
    it powered the die."""
    times = []
    for t in ((summary or {}).get("phases") or {}).values():
        try:
            times.append(datetime.fromisoformat(t))
        except (TypeError, ValueError):
            continue
    return min(times) if times else None


def pick_run(die_dir, before=None):
    """(run_dir, summary, passed_over) for one die folder: the newest run
    with a summary.json that was not aborted with Ctrl+C (the wafer loop
    stops before it grades such a run); with `before` (a datetime), the
    newest such run that started before it. passed_over names the newer
    run folders left out, each with its reason: no summary.json (a run
    killed before writing one; not listed with `before`, where such a
    folder cannot be dated) or aborted."""
    passed_over = []
    for run_dir in reversed(run_dirs(die_dir)):
        summary = _read_json(run_dir / "summary.json")
        if summary is None:
            if before is None:
                passed_over.append(f"{run_dir.name} (no summary.json)")
            continue
        if before is not None:
            t = start_time(summary)
            if t is None or t >= before:
                continue
        if summary.get("status") == "user_abort":
            passed_over.append(f"{run_dir.name} (aborted)")
            continue
        return run_dir, summary, passed_over
    return None, None, passed_over


def phase_values(power):
    """{(rail, phase): (current_A, voltage_V)}: the median of each rail's
    samples in each phase of one power.parquet, without the phase's first
    sample when it has more than one. The phase label switches as the
    command goes out and the rails follow a few hundred ms later: the first
    high_power sweep can still read the low-power current, and the first
    power_on sweep catches vref still ramping (0.33 V, then 1.00 V)."""
    out = {}
    if power is None or power.empty or "phase" not in power.columns:
        return out
    for (rail, phase), g in power.groupby(["description", "phase"], sort=False):
        g = g.sort_values("timestamp")
        if len(g) > 1:
            g = g.iloc[1:]
        out[(rail, phase)] = (float(g["current"].median()), float(g["voltage"].median()))
    return out


def _rail_columns(summary, power):
    checks = summary.get("short_check") or {}
    values = phase_values(power)
    rails = list(checks) + [r for r, _ in values if r not in checks]
    out = {}
    for rail in dict.fromkeys(rails):
        check = checks.get(rail) or {}
        for phase, tag in PHASES.items():
            current, voltage = values.get((rail, phase), (None, None))
            if phase == "power_on" and current is None:
                # no power log for the run: the power-on rail check reading
                current, voltage = check.get("current"), check.get("voltage")
            out[f"{rail}_I_{tag}"] = current
            out[f"{rail}_V_{tag}"] = voltage
        out[f"{rail}_I_abort_above"] = check.get("abort_above")
        out[f"{rail}_I_abort_below"] = check.get("abort_below")
    return out


def _pixel(p):
    return [int(p["row"]), int(p["col"])] if isinstance(p, dict) else [int(p[0]), int(p[1])]


def _i2c_columns(summary):
    """Pass flags as the grading in station.py reads them (a check fails when its
    result says passed: false), and the failure lists of all chips."""
    i2c = summary.get("i2c") or {}
    if not i2c:
        return {}
    ok = {"pixel_id": True, "peripheral": True}
    failed_pixels, failed_registers = [], []
    for checks in i2c.values():
        for name in ok:
            result = checks.get(name)
            if result and not result.get("passed", True):
                ok[name] = False
        failed_pixels += [_pixel(p) for p in (checks.get("pixel_id") or {}).get("failed_pixels") or []]
        failed_registers += list((checks.get("peripheral") or {}).get("failed_registers") or [])
    return {"pixel_id_ok": ok["pixel_id"], "peripheral_ok": ok["peripheral"],
            "failed_pixels": json.dumps(failed_pixels) if failed_pixels else "",
            "failed_registers": json.dumps(failed_registers) if failed_registers else ""}


def _calibration_columns(baseline):
    """Baseline and noise-width statistics over the pixels that did not
    read zero (a zero baseline or noise width is a failed auto-calibration,
    listed apart; summary.json's own means include them)."""
    if baseline is None or baseline.empty:
        return {}
    zero = (baseline["baseline"] == 0) | (baseline["noise_width"] == 0)
    good = baseline[~zero]
    return {"n_pixels": int(len(baseline)), "n_zero_pixels": int(zero.sum()),
            "zero_pixels": json.dumps([[int(r), int(c)] for r, c in
                                       zip(baseline["row"][zero], baseline["col"][zero])]) if zero.any() else "",
            "bl_mean": float(good["baseline"].mean()) if len(good) else None,
            "bl_std": float(good["baseline"].std()) if len(good) > 1 else None,
            "nw_mean": float(good["noise_width"].mean()) if len(good) else None,
            "nw_std": float(good["noise_width"].std()) if len(good) > 1 else None}


def _qinj_columns(summary):
    """The QInj verdict of the run and the pixels it injected, as the run
    recorded them (older runs lack the pixel list, and runs from before
    the QInj check lack the verdict too)."""
    out = {}
    check = summary.get("qinj_check") or {}
    if check:
        out.update({"qinj_check_ok": check.get("ok"), "qinj_check_events": check.get("events"),
                    "qinj_check_bad": check.get("bad"), "qinj_check_reason": check.get("reason"),
                    "qinj_hits_expected": check.get("hits_expected")})
    pixels = summary.get("qinj_pixels")
    if pixels:
        out["qinj_pixels"] = json.dumps([_pixel(p) for p in pixels])
    return out


def _prober_columns(summary):
    """The station's map position of the die and the chuck position read at
    contact (wafer_run.py records both); d = chuck - map."""
    prober = summary.get("prober") or {}
    if not prober:
        return {}
    velox, chuck = prober.get("velox") or {}, prober.get("chuck") or {}
    out = {"map_x_um": velox.get("x_um"), "map_y_um": velox.get("y_um"),
           "chuck_x_um": chuck.get("x_um"), "chuck_y_um": chuck.get("y_um"),
           "z_contact_um": chuck.get("z_contact_um"), "retouched": prober.get("retouched")}
    for axis in ("x", "y"):
        a, b = out[f"chuck_{axis}_um"], out[f"map_{axis}_um"]
        out[f"d{axis}_um"] = a - b if a is not None and b is not None else None
    return out


def die_record(die, position, run_dir=None, summary=None, baseline=None, power=None):
    """One row of the dies table; a die without a run grades NOT_TESTED."""
    g = grade(summary)
    record = {"die": die, "die_row": position[0], "die_col": position[1],
              "grade": g.name, "bin": g.bin, "detail": g.detail}
    if summary is None:
        return record
    attempt = summary.get("attempt") or 1
    t = start_time(summary)
    record.update({
        "map_text": result_text(g, attempt),
        "run": run_dir.name if run_dir is not None else None,
        "start": t.isoformat(sep=" ", timespec="seconds") if t else None,
        "wafer_stage": summary.get("wafer_stage"),
        "status": summary.get("status"), "error": summary.get("error"),
        "attempt": attempt, "retry_reason": summary.get("retry_reason"),
        "fullscan": bool(summary.get("fullscan")), "qinj": bool(summary.get("qinj")),
        "elapsed_s": summary.get("elapsed_s"),
    })
    record.update(_rail_columns(summary, power))
    record.update(_i2c_columns(summary))
    record.update(_calibration_columns(baseline))
    record.update(_qinj_columns(summary))
    record.update(_prober_columns(summary))
    return record


def qinj_files(run_dir):
    """The .nem files that hold a run's QInj data, in file order, or None
    when the run has no QInj folder. The station writes one run, qinj/,
    whose first file can hold malformed events from the start of the run
    and is left out, so a qinj/ run that wrote a single file gives an
    empty list; older runs wrote a flush run, qinj_run1/, and then the
    data run, qinj_run2/, read whole."""
    if run_dir is None:
        return None
    if (run_dir / "qinj").is_dir():
        return nem_files(run_dir / "qinj")[1:]
    if (run_dir / "qinj_run2").is_dir():
        return nem_files(run_dir / "qinj_run2")
    return None


def read_nem_hits(files):
    """(events, flagged, hits) over the given .nem files, in order: the
    number of events (EH records), the number of frame
    trailers (T) whose chip status field is not 0, and one row per hit
    record (D) as an int array with the columns ea, row, col, toa, tot, cal.
    Field positions as the EtrocReceiver translation writes them:
    D <channel> <ea> <row> <col> <toa> <tot> <cal> and
    T <channel> <chip id> <status> <hits> <crc>."""
    events = flagged = 0
    rows = []
    for path in files:
        with open(path) as f:
            for line in f:
                t = line.split()
                if not t:
                    continue
                if t[0] == "D" and len(t) >= 8:
                    rows.append(t[2:8])
                elif t[0] == "EH":
                    events += 1
                elif t[0] == "T" and len(t) >= 4 and t[3] != "0":
                    flagged += 1
    hits = np.array(rows, dtype=np.int64).reshape(-1, 6)
    return events, flagged, hits


def qinj_pixel_stats(hits, events):
    """One dict per pixel with hits whose EA field is 0 (a nonzero EA is the
    chip's own anomaly flag): hits, efficiency = hits / events, the most
    common CAL code, and over the hits with |CAL - that code| < CAL_WINDOW
    n_sel and the mean and sample std of TOA, TOT and CAL."""
    good = hits[hits[:, 0] == 0]
    out = []
    for r, c in np.unique(good[:, 1:3], axis=0).tolist():
        h = good[(good[:, 1] == r) & (good[:, 2] == c)]
        cal = h[:, 5]
        mode = int(np.bincount(cal).argmax())
        sel = h[np.abs(cal - mode) < CAL_WINDOW]
        record = {"pix_row": int(r), "pix_col": int(c), "hits": int(len(h)),
                  "eff": len(h) / events if events else float("nan"),
                  "cal_mode": mode, "n_sel": int(len(sel))}
        for name, k in (("toa", 3), ("tot", 4), ("cal", 5)):
            v = sel[:, k].astype(float)
            record[f"{name}_mean"] = float(v.mean()) if len(v) else float("nan")
            record[f"{name}_std"] = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
        out.append(record)
    return out


def injected_pixels(qinj):
    """The wafer's injected pixels as the data show them: hit in at least
    half of the events of at least half of the QInj dies."""
    if qinj.empty:
        return []
    n_dies = qinj["die"].nunique()
    hot = qinj[qinj["eff"] >= 0.5].groupby(["pix_row", "pix_col"])["die"].nunique()
    return sorted((int(r), int(c)) for (r, c), n in hot.items() if n >= n_dies / 2)


def _recorded(value):
    """The pixel list of a qinj_pixels cell, or None."""
    return [tuple(p) for p in json.loads(value)] if isinstance(value, str) and value else None


def lowest_efficiency(dies, qinj):
    """Per die (dies order), the lowest efficiency over the pixels its QInj
    run injected; a pixel without a hit counts 0. The pixels are the list
    the run recorded; for a run that recorded none, the wafer's injected
    pixels when its QInj check expected that many hits per event. NaN
    without a QInj run, or when the injected pixels are not known."""
    wafer_set = injected_pixels(qinj)
    eff = {(int(d), int(r), int(c)): e for d, r, c, e in
           qinj[["die", "pix_row", "pix_col", "eff"]].itertuples(index=False)}
    out = []
    for d in dies.itertuples(index=False):
        events = getattr(d, "qinj_events", None)
        pixels = _recorded(getattr(d, "qinj_pixels", None))
        if pixels is None and wafer_set and getattr(d, "qinj_hits_expected", None) == len(wafer_set):
            pixels = wafer_set
        if events is None or not np.isfinite(events) or pixels is None:
            out.append(float("nan"))
        elif events == 0:
            out.append(0.0)
        else:
            out.append(min(eff.get((int(d.die), r, c), 0.0) for r, c in pixels))
    return out


def offset_trend(dies, column):
    """The least-squares plane column = a + b * die_col + c * die_row over
    the dies with a value: (a, b, c, residual std, dies), or None with
    fewer than 4 dies."""
    if column not in dies:
        return None
    d = dies[["die_col", "die_row", column]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 4:
        return None
    a = np.c_[np.ones(len(d)), d["die_col"], d["die_row"]]
    v = d[column].to_numpy(dtype=float)
    coef = np.linalg.lstsq(a, v, rcond=None)[0]
    residual = v - a @ coef
    return float(coef[0]), float(coef[1]), float(coef[2]), float(residual.std(ddof=3)), len(d)


def wafer_spread(dies, column):
    """(median, robust sigma) of a dies column over the PASSED dies, the
    robust sigma being 1.4826 x the median absolute deviation; None when
    fewer than NOTE_MIN_DIES of them have a value or the values do not
    spread."""
    if column not in dies:
        return None
    values = pd.to_numeric(dies.loc[dies["grade"] == "PASSED", column], errors="coerce").dropna()
    if len(values) < NOTE_MIN_DIES:
        return None
    median = float(values.median())
    sigma = 1.4826 * float((values - median).abs().median())
    return (median, sigma) if sigma > 0 else None


def _pixel_groups(dies):
    """{n_pixels: dies calibrated over that many pixels}: the die check
    compares a die with its own group only, since a mean over the 9 pixels
    of a quick test scatters more than one over the 256 of a full scan."""
    if "n_pixels" not in dies:
        return {}
    return {int(n): group for n, group in dies.groupby("n_pixels")}


def unchecked_die_means(dies):
    """[(n_pixels, [dies column, ...])] for the groups of dies (_pixel_groups)
    whose means bl_nw_notes could not check (wafer_spread gave None)."""
    out = []
    for n, group in _pixel_groups(dies).items():
        columns = [column for column, _, _ in DIE_MEANS if wafer_spread(group, column) is None]
        if columns:
            out.append((n, columns))
    return out


def bl_nw_notes(dies, pixels):
    """A note per die of the dies table ("" for none), whatever its grade,
    on what stands out in its baseline or noise width, for the figures to
    mark: pixels whose baseline is further than NOTE_PIXEL_BL DAC codes,
    or whose noise width is further than NOTE_PIXEL_NW, from the die's
    median, and a baseline or noise-width mean further than
    NOTE_DIE_SIGMA robust sigmas from the median of the PASSED dies
    calibrated over as many pixels (_pixel_groups, wafer_spread). Zero
    readings stay out: they are a failed calibration, which the grade
    already takes. A note marks a die, it never grades it."""
    found = {}
    good = pixels[(pixels["baseline"] > 0) & (pixels["noise_width"] > 0)]
    for die, p in good.groupby("die"):
        for column, what, limit in (("baseline", "baseline", NOTE_PIXEL_BL),
                                    ("noise_width", "noise width", NOTE_PIXEL_NW)):
            median = float(p[column].median())
            off = p.assign(dev=(p[column] - median).abs())
            off = off[off["dev"] > limit].sort_values("dev", ascending=False, kind="stable")
            if off.empty:
                continue
            listed = [f"({r},{c}) at {v}" for r, c, v in
                      zip(off["pix_row"], off["pix_col"], off[column])][:NOTE_PIXELS_LISTED]
            more = f" and {len(off) - len(listed)} more" if len(off) > len(listed) else ""
            found.setdefault(die, []).append(
                f"die median {what} {median:g}, pixel{'s' if len(off) > 1 else ''} {', '.join(listed)}{more}")
    for n, group in _pixel_groups(dies).items():
        for column, what, fmt in DIE_MEANS:
            spread = wafer_spread(group, column)
            if spread is None:
                continue
            median, sigma = spread
            for die, v in zip(group["die"], pd.to_numeric(group[column], errors="coerce")):
                z = (v - median) / sigma
                if abs(z) > NOTE_DIE_SIGMA:  # NaN compares False
                    found.setdefault(die, []).append(
                        f"{what} {fmt.format(v)}, {z:+.1f} sigma from the median {fmt.format(median)} "
                        f"of the PASSED {n}-pixel dies")
    return dies["die"].map(lambda die: "; ".join(found.get(die, [])))


def collect(wafer_dir, wafer_map, before=None, with_qinj=True):
    """(dies, pixels, qinj, warnings) for one wafer folder. wafer_map maps
    the die number to its (row, col) on the station map, as
    station.load_wafer_map reads wafer_map.csv."""
    wafer_dir = Path(wafer_dir)
    warnings = []
    listed = {f"die{d:03d}" for d in wafer_map}
    for p in sorted(wafer_dir.glob("die*")):
        if p.is_dir() and p.name not in listed:
            warnings.append(f"{p.name} is not in the wafer map, left out")
    dies, pixels, qinj = [], [], []
    for die in sorted(wafer_map):
        die_dir = wafer_dir / f"die{die:03d}"
        run_dir = summary = baseline = power = None
        if die_dir.is_dir():
            run_dir, summary, passed_over = pick_run(die_dir, before)
            if passed_over:
                warnings.append(f"die {die}: {', '.join(passed_over)} left out, "
                                f"{run_dir.name if run_dir else 'no earlier run'} used")
        if run_dir is not None:
            baseline = _read_parquet(run_dir / "baseline.parquet")
            power = _read_parquet(run_dir / "power.parquet")
        record = die_record(die, wafer_map[die], run_dir, summary, baseline, power)
        if baseline is not None and not baseline.empty:
            pixels.append(pd.DataFrame({"die": die,
                                        "pix_row": baseline["row"].astype(int),
                                        "pix_col": baseline["col"].astype(int),
                                        "baseline": baseline["baseline"].astype(int),
                                        "noise_width": baseline["noise_width"].astype(int)}))
        files = qinj_files(run_dir)
        # a run whose only file was left out has no QInj data: NaN, not zeros
        if with_qinj and files:
            events, flagged, hits = read_nem_hits(files)
            record.update({"qinj_events": events, "qinj_flagged_trailers": flagged,
                           "qinj_ea_words": int((hits[:, 0] != 0).sum())})
            qinj += [{"die": die, **s} for s in qinj_pixel_stats(hits, events)]
        dies.append(record)
    dies = pd.DataFrame(dies)
    pixels = pd.concat(pixels, ignore_index=True) if pixels else pd.DataFrame(columns=PIXEL_COLUMNS)
    qinj = pd.DataFrame(qinj, columns=QINJ_COLUMNS)
    if "qinj_events" in dies:
        dies["qinj_min_eff"] = lowest_efficiency(dies, qinj)
    dies["bl_nw_note"] = bl_nw_notes(dies, pixels)
    return dies, pixels, qinj, warnings


def grade_counts(dies):
    """([(grade name, count)] in bin order, passed, tested) for a dies table."""
    counts = dies.groupby(["bin", "grade"]).size()
    rows = [(name, int(n)) for (_, name), n in counts.sort_index().items()]
    tested = int((dies["grade"] != "NOT_TESTED").sum())
    passed = int((dies["grade"] == "PASSED").sum())
    return rows, passed, tested


def write_tables(out_dir, dies, pixels, qinj):
    """dies.csv, pixels.csv and qinj.csv in out_dir; returns their paths."""
    paths = []
    for name, table in (("dies", dies), ("pixels", pixels), ("qinj", qinj)):
        path = Path(out_dir) / f"{name}.csv"
        table.to_csv(path, index=False)
        paths.append(path)
    return paths
