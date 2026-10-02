"""Synthetic wafer results laid out as master_run_script.run_die writes
them, for tests/test_wafer_tables.py and tests/test_plot_wafer.py:
<wafer folder>/<stage>/die<nnn>/run_<k>_noEfuse/ with summary.json, power.parquet,
baseline.parquet and qinj_run2/file_<n>.nem. Needs pandas and pyarrow."""
import json
from pathlib import Path

import pandas as pd

QUICK_PIXELS = [(2, 2), (2, 10), (10, 2), (10, 10), (5, 5), (5, 13), (13, 5), (13, 13), (15, 15)]
FULL_PIXELS = [(r, c) for r in range(16) for c in range(16)]
RAILS = ("analog", "digital", "ws_analog")


def event(pixels, status=0, junk=0):
    """The lines of one translated QInj event (EH, H, one D per pixel, T,
    ET), D as <channel> <ea> <row> <col> <toa> <tot> <cal>. `junk` adds hit
    words at pixel (0, 0) with EA 2 and the trailer takes `status`, as on
    the dies of 2026-09-22 whose QInj readout broke."""
    lines = ["EH 7897 0 10 13", "H 0 130 0 719"]
    lines += [f"D 0 0 {r} {c} {250 + k} {70 + k % 3} {180 + k % 2}" for k, (r, c) in enumerate(pixels)]
    lines += ["D 0 2 0 0 256 0 0"] * junk
    n = len(pixels) + junk
    lines += [f"T 0 0x17f0f {status} {n} 44", f"ET {n} 0 0 0"]
    return [line + "\n" for line in lines]


def summary(die, row, col, *, status="completed", attempt=1, power_on="2026-09-22 15:00:00.000", **extra):
    """A summary.json record; `extra` adds or replaces top-level keys."""
    record = {
        "batch": "B", "wafer": "W", "wafer_stage": "pre_ubm", "die": die, "row": row, "col": col, "status": status, "error": None,
        "attempt": attempt, "fullscan": False, "qinj": False, "phases": {"power_on": power_on},
        "short_check": {rail: {"voltage": 1.2, "current": 0.3, "abort_above": 0.54, "abort_below": 0.1,
                               "ok": True} for rail in ("analog", "digital")},
        "i2c": {"0x60": {"pixel_id": {"passed": True, "failed_pixels": [], "error": None},
                         "peripheral": {"passed": True, "failed_registers": [], "error": None}}},
    }
    record.update(extra)
    return record


def power(samples):
    """power.parquet rows from (rail, phase, current[, voltage]) tuples, one
    second apart; the voltage defaults to 1.2 V."""
    t0 = pd.Timestamp("2026-09-22 15:00:00")
    return pd.DataFrame([{"timestamp": t0 + pd.Timedelta(seconds=k), "description": rail, "channel": 1,
                          "voltage": voltage[0] if voltage else 1.2, "current": current, "phase": phase}
                         for k, (rail, phase, current, *voltage) in enumerate(samples)])


def run_power(on=0.31, high=0.45):
    """Two power-on and three high-power sweeps of every rail; the first
    high-power sweep still reads the power-on current, as it does on the
    station. ws_analog draws a tenth."""
    samples = []
    for phase, currents in (("power_on", (on, on)), ("high_power", (on, high, high))):
        for current in currents:
            samples += [(rail, phase, current / 10 if rail == "ws_analog" else current) for rail in RAILS]
    return power(samples)


def baseline(pixels, zero=()):
    """baseline.parquet rows; the pixels in `zero` read baseline 0."""
    return pd.DataFrame([{"row": r, "col": c, "baseline": 0 if (r, c) in zero else 400 + r + c,
                          "noise_width": 0 if (r, c) in zero else 6 + (r + c) % 3,
                          "timestamp": "2026-09-22 15:00:05", "chip_name": "0x60"} for r, c in pixels])


def write_run(wafer_dir, die, run, summary=None, power=None, baseline=None, nem=None):
    """One run folder of one die; `nem` maps a subfolder (qinj, or qinj_run2
    as older runs have it) to the list of files, each a list of lines.
    Returns the run folder."""
    run_dir = Path(wafer_dir) / f"die{die:03d}" / f"run_{run:02d}_noEfuse"
    run_dir.mkdir(parents=True)
    if summary is not None:
        (run_dir / "summary.json").write_text(json.dumps(summary))
    if power is not None:
        power.to_parquet(run_dir / "power.parquet")
    if baseline is not None:
        baseline.to_parquet(run_dir / "baseline.parquet")
    for folder, files in (nem or {}).items():
        (run_dir / folder).mkdir()
        for k, lines in enumerate(files):
            (run_dir / folder / f"file_{k}.nem").write_text("".join(lines))
    return run_dir


def write_map(path, wafer_map, invalid=()):
    """A wafer map csv as wafer_map.csv has it, the dies in `invalid` marked."""
    rows = ["location_id,row,col,invalid"] + [f"{d},{r},{c},{int(d in invalid)}"
                                               for d, (r, c) in sorted(wafer_map.items())]
    Path(path).write_text("\n".join(rows) + "\n")
