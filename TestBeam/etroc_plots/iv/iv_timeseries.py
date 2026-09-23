#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Currents-vs-time inputs for the IV figures: March raw-log reduction + July timeline reader.

March: reduces the six ~1 Hz IV-monitor raw logs (one per telescope x fluence step, each covering
two runs) to 60 s median bins, restricted to each run's plateau window and to bins whose readback
voltage sits within 6 V of that run's bias.  Raw `Date` is local CET = UTC+1 (no DST); every window
boundary below is UTC, so raw timestamps are shifted by -1 h before any comparison.  Output:
INPUTS/march/march_inrun_60s.csv (tel, run, chip, t_utc, elapsed_min, v_V, i_uA), elapsed_min
measured from that run's own start (H1: the run start; F1: the same start_utc used for its window).

July: INPUTS/july/timeline_60s.csv.gz is already 60 s bins in UTC; this module only selects runs
(2e15 and 3.5e15: the display_runs_jul.csv-listed RFSel 2 / Disc offset 20 runs at that fluence;
a selection by plateau length alone is blind to RFSel/offset and would mix other settings classes
into a figure meant to show one) using the per-run current table (JULY_REF_CSV) and filters to
each run's plateau window + 6 V bias tolerance.

Assertions (both campaigns) compare the median current per chip over the reference sub-window
against the campaign's own numbers, with a 5% tolerance, and are never tuned away. The March
reduction prints every comparison, marks a bigger disagreement "<-- FAIL", still writes its CSV and
returns ok=False; the IV notebook's July figures drop a run that disagrees and list it as rejected.

CLI:
  iv_timeseries.py --reduce-one h1_3e14        (single raw file, a quick sanity check)
  iv_timeseries.py --reduce-all                (all six files -> INPUTS/march/march_inrun_60s.csv)
  iv_timeseries.py --assert-march [csv]        (re-check an already-reduced march CSV)
  iv_timeseries.py --assert-july                (build + check the July run selection)
"""
import argparse
import csv
import json
import os
from datetime import timedelta

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- campaign catalogue (see campaigns/) --------------------------------
# These names are defined by the campaign module and re-bound here, so this module and the
# notebook use them as tsdata.<NAME>.
from ..campaigns import active as _campaign
INPUTS = _campaign.INPUTS            # the campaign's input folder (ETROC_IV_INPUTS overrides it)
EOS_ROOT = _campaign.EOS_ROOT
DISPLAY_RUNS_JUL_CSV = _campaign.DISPLAY_RUNS_JUL_CSV
TELESCOPE_CHIPS = _campaign.TELESCOPE_CHIPS
MARCH_H1_RUNS = _campaign.MARCH_H1_RUNS
MARCH_F1_RUN_NUMS = _campaign.MARCH_F1_RUN_NUMS
F1_WINDOWS_CSV = _campaign.F1_WINDOWS_CSV
MARCH_RAW_FILES = _campaign.MARCH_RAW_FILES
RUNS_AT_FLUENCE = _campaign.RUNS_AT_FLUENCE
JULY_TIMELINE = _campaign.JULY_TIMELINE
JULY_REF_CSV = _campaign.JULY_REF_CSV
JULY_TEL = _campaign.JULY_TEL
GOOD_RUNS_CSV = _campaign.GOOD_RUNS_CSV
GOOD_RUNS_CAMPAIGN = _campaign.GOOD_RUNS_CAMPAIGN
GOOD_RUNS_TEL = _campaign.GOOD_RUNS_TEL
JULY_YAML = _campaign.JULY_YAML
JULY_1P5E15_RFSEL = _campaign.JULY_1P5E15_RFSEL
JULY_1P5E15_OFFSET = _campaign.JULY_1P5E15_OFFSET
JULY_TIMELINE_JSON = _campaign.JULY_TIMELINE_JSON
JULY_STEPS = _campaign.JULY_STEPS
JULY_STEPS_RFSEL = _campaign.JULY_STEPS_RFSEL
JULY_STEPS_OFFSET = _campaign.JULY_STEPS_OFFSET
JULY_MIN_PLATEAU_H = _campaign.JULY_MIN_PLATEAU_H
FLUENCE_TEXT_PLAIN = _campaign.FLUENCE_TEXT_PLAIN
PREIRRAD_LOG_UTC_OFFSET_H = _campaign.PREIRRAD_LOG_UTC_OFFSET_H





BIAS_TOL_V = 6.0    # V: a bin counts only while the HV readback is this close to the run bias
SETTLE_MIN = 15.0   # minutes after the HV step dropped from every in-run window: the readback
                    # is in tolerance there, the current is not
BIN = "60S"


def _f1_windows():
    """run -> dict(start_utc, end_utc, bias, ref_uA{chip}) from F1_WINDOWS_CSV.

    win_start_utc/win_end_utc there are start_utc+15min / +plateau_h, i.e. exactly the assertion
    sub-window; start_utc/window_h give the run's own full plotted window.
    """
    df = pd.read_csv(F1_WINDOWS_CSV)
    df = df.set_index("run")
    out = {}
    for run, rname in MARCH_F1_RUN_NUMS.items():
        row = df.loc[rname]
        out[run] = dict(
            start_utc=pd.Timestamp(row["start_utc"]),
            end_utc=pd.Timestamp(row["start_utc"]) + timedelta(hours=float(row["window_h"])),
            win_start_utc=pd.Timestamp(row["win_start_utc"]),
            win_end_utc=pd.Timestamp(row["win_end_utc"]),
            bias={c: float(row["Vread_%s_V" % c]) for c in TELESCOPE_CHIPS["f1"]},
            ref_uA={c: float(row["I_%s_uA" % c]) for c in TELESCOPE_CHIPS["f1"]},
        )
    return out


def _reduce_one_file(tel, fluence, verbose=True):
    """Read one raw log once, bin to 60 s medians, restrict to that step's two runs + bias tol.

    Returns a DataFrame (tel, run, chip, t_utc, elapsed_min, v_V, i_uA).
    """
    path = MARCH_RAW_FILES[(tel, fluence)]
    chips = TELESCOPE_CHIPS[tel]
    usecols = ["Date"] + ["Re(%smeas[%d]) [%s]" % (kind, c, unit)
                          for c in range(len(chips)) for kind, unit in (("V", "V"), ("I", "A"))]
    if verbose:
        print("  reading %s ..." % path)
    df = pd.read_csv(path, usecols=usecols)
    df = df.dropna(subset=usecols[1:], how="all")
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %H:%M:%S.%f", errors="coerce")
    df = df.dropna(subset=["Date"])
    df["t_utc"] = df["Date"] - timedelta(hours=PREIRRAD_LOG_UTC_OFFSET_H)   # local -> UTC
    df["bin_ts"] = df["t_utc"].dt.floor(BIN)
    grouped = df.groupby("bin_ts")

    # per-run windows for this telescope/step
    runs = RUNS_AT_FLUENCE[fluence]
    if tel == "h1":
        windows = {r: dict(start=pd.Timestamp(MARCH_H1_RUNS[r]["start_utc"]),
                            end=pd.Timestamp(MARCH_H1_RUNS[r]["start_utc"])
                                + timedelta(hours=MARCH_H1_RUNS[r]["plateau_h"]),
                            bias=MARCH_H1_RUNS[r]["bias"])
                   for r in runs}
    else:
        f1w = _f1_windows()
        windows = {r: dict(start=f1w[r]["start_utc"], end=f1w[r]["end_utc"], bias=f1w[r]["bias"])
                   for r in runs}

    rows = []
    for c, chip in enumerate(chips):
        vcol, icol = "Re(Vmeas[%d]) [V]" % c, "Re(Imeas[%d]) [A]" % c
        sub = grouped[[vcol, icol]].median().reset_index()
        sub["v_V"] = sub[vcol].abs()
        sub["i_uA"] = sub[icol].abs() * 1.0e6
        sub = sub.dropna(subset=["v_V", "i_uA"])
        for r in runs:
            w = windows[r]
            m = (sub["bin_ts"] >= w["start"]) & (sub["bin_ts"] <= w["end"])
            m &= (sub["v_V"] - w["bias"][chip]).abs() <= BIAS_TOL_V
            picked = sub.loc[m]
            if picked.empty:
                continue
            elapsed = (picked["bin_ts"] - w["start"]).dt.total_seconds() / 60.0
            rows.append(pd.DataFrame(dict(
                tel=tel, run=r, chip=chip,
                t_utc=picked["bin_ts"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
                elapsed_min=elapsed.round(2), v_V=picked["v_V"].round(3),
                i_uA=picked["i_uA"].round(3))))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["tel", "run", "chip", "t_utc", "elapsed_min", "v_V", "i_uA"])
    return out


def _assert_table(reduced, label_windows, ref_lookup):
    """Print + return [(run, chip, drawn_median, ref, pct_diff)] for the assertion sub-window."""
    results = []
    ok = True
    print("  %-4s %-6s %10s %10s %8s" % ("run", "chip", "drawn_uA", "ref_uA", "%diff"))
    for run in sorted(reduced["run"].unique()):
        w = label_windows[run]
        for chip in TELESCOPE_CHIPS[reduced.loc[reduced["run"] == run, "tel"].iloc[0]]:
            sub = reduced[(reduced["run"] == run) & (reduced["chip"] == chip)]
            t = pd.to_datetime(sub["t_utc"])
            m = (t >= w["assert_start"]) & (t <= w["assert_end"])
            drawn = sub.loc[m, "i_uA"].median()
            ref = ref_lookup[run][chip]
            pct = 100.0 * (drawn - ref) / ref if pd.notna(drawn) else float("nan")
            flag = "" if pd.notna(pct) and abs(pct) <= 5.0 else "  <-- FAIL"
            if not flag == "":
                ok = False
            print("  %-4d %-6s %10.1f %10.1f %7.2f%%%s" % (run, chip, drawn, ref, pct, flag))
            results.append(dict(run=run, chip=chip, drawn_uA=round(float(drawn), 2),
                                ref_uA=ref, pct_diff=round(float(pct), 3)))
    return results, ok


def reduce_all(out_csv):
    """Reduce every March raw log (MARCH_RAW_FILES: each telescope x RUNS_AT_FLUENCE step) to
    60 s medians, write them to out_csv, then compare each run's per-chip median with the
    campaign's reference (H1: MARCH_H1_RUNS ref_uA; F1: F1_WINDOWS_CSV). A disagreement above 5%
    is printed with "<-- FAIL"; the CSV is written either way.
    Returns (reduced frame, comparison rows, True when every comparison passed)."""
    all_parts = []
    for tel in TELESCOPE_CHIPS:
        for fluence in RUNS_AT_FLUENCE:
            part = _reduce_one_file(tel, fluence)
            all_parts.append(part)
            print("    %s %.0e -> %d rows" % (tel, fluence, len(part)))
    full = pd.concat(all_parts, ignore_index=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    full.to_csv(out_csv, index=False)
    print("wrote %s (%d rows)" % (out_csv, len(full)))

    # assertion: H1 against MARCH_H1_RUNS ref_uA, F1 against F1_WINDOWS_CSV
    h1_windows = {r: dict(assert_start=pd.Timestamp(MARCH_H1_RUNS[r]["start_utc"]) + timedelta(minutes=SETTLE_MIN),
                          assert_end=pd.Timestamp(MARCH_H1_RUNS[r]["start_utc"])
                                     + timedelta(hours=MARCH_H1_RUNS[r]["plateau_h"]))
                 for r in MARCH_H1_RUNS}
    h1_ref = {r: MARCH_H1_RUNS[r]["ref_uA"] for r in MARCH_H1_RUNS}
    f1w = _f1_windows()
    f1_windows = {r: dict(assert_start=f1w[r]["win_start_utc"], assert_end=f1w[r]["win_end_utc"])
                 for r in f1w}
    f1_ref = {r: f1w[r]["ref_uA"] for r in f1w}

    print("\nAssertion table - H1 (vs the campaign's MARCH_H1_RUNS ref_uA):")
    h1_res, h1_ok = _assert_table(full[full["tel"] == "h1"], h1_windows, h1_ref)
    print("\nAssertion table - F1 (vs %s):" % os.path.basename(F1_WINDOWS_CSV))
    f1_res, f1_ok = _assert_table(full[full["tel"] == "f1"], f1_windows, f1_ref)
    print("\nAssertion: %s" % ("ALL PASS" if h1_ok and f1_ok else "FAILURES ABOVE - see <-- FAIL"))
    return full, h1_res + f1_res, (h1_ok and f1_ok)


def load_march(csv_path=None):
    csv_path = csv_path or os.path.join(INPUTS, "march", "march_inrun_60s.csv")
    return pd.read_csv(csv_path, parse_dates=["t_utc"])





def load_july_timeline(path=None):
    path = path or JULY_TIMELINE
    df = pd.read_csv(path, parse_dates=["tb"])
    return df


def _parse_good_flag(value, where):
    """A good-run flag as a bool: a bool, the numbers 1 / 0, or the strings true / false / 1 / 0
    (any case). Anything else (an empty cell, "yes", ...) raises ValueError naming `where`."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("true", "false", "1", "0"):
        return value.strip().lower() in ("true", "1")
    raise ValueError("%s: good = %r is not true/false or 1/0; fix that cell" % (where, value))


def good_runs(tel_key, path=None, good_runs_campaign=None):
    """(good, reasons) for one telescope from the good-run list: `good` is the set of plain run
    numbers (int) marked good==True, `reasons` maps every not-good run number to its CSV reason.
    `good_runs_campaign` picks the list's campaign column (default GOOD_RUNS_CAMPAIGN, the July
    runs). Merged-run rows ("46+47") are skipped: the per-run selections never see them.
    """
    path = path or GOOD_RUNS_CSV
    df = pd.read_csv(path)
    df = df[(df["campaign"] == (good_runs_campaign or GOOD_RUNS_CAMPAIGN))
            & (df["telescope"] == GOOD_RUNS_TEL[tel_key])]
    good, reasons = set(), {}
    for _, row in df.iterrows():
        run_field = str(row["run"])
        if "+" in run_field:
            continue
        run_num = int(run_field)
        if _parse_good_flag(row["good"], "%s, %s run %s" % (path, GOOD_RUNS_TEL[tel_key],
                                                              run_field)):
            good.add(run_num)
        else:
            reasons[run_num] = row["reason"]
    return good, reasons


def _run_num(run_field):
    """'h1_run48' -> 48."""
    return int(str(run_field).rsplit("run", 1)[-1])


def _load_display_runs_jul_rows():
    """DISPLAY_RUNS_JUL_CSV rows as dicts, via the stdlib csv module rather than pandas: the
    file's `note` free-text field has unquoted commas (e.g. the H1 3.5e15 RFSel2/offset "8/10"
    run 56 row), which pandas' C parser rejects outright as a field-count mismatch for the whole
    file. csv.DictReader tolerates that ragged trailing field; every column _display_runs_jul
    reads (telescope/fluence/rfsel/os/run) sits before the ragged tail so it is unaffected.
    """
    with open(DISPLAY_RUNS_JUL_CSV, newline="") as fh:
        return list(csv.DictReader(fh))


def _setting_is(value, want):
    """True when a display-list setting cell (rfsel, os) is the number `want`; a cell holding
    several settings ("8/10") is not."""
    try:
        return float(str(value).strip()) == float(want)
    except ValueError:
        return False


def _display_runs_jul(fluence):
    """{tel_key: set(run_num)}: the DISPLAY_RUNS_JUL_CSV rows at `fluence` with RFSel
    JULY_STEPS_RFSEL and threshold offset JULY_STEPS_OFFSET."""
    out = {tel_key: set() for tel_key in JULY_TEL}
    for row in _load_display_runs_jul_rows():
        if float(row["fluence"]) != fluence:
            continue
        if not (_setting_is(row["rfsel"], JULY_STEPS_RFSEL)
                and _setting_is(row["os"], JULY_STEPS_OFFSET)):
            continue
        tel_key = str(row["telescope"]).strip().lower()
        if tel_key in out:
            out[tel_key].add(int(row["run"]))
    return out


def _cut_reason(rows, min_plateau_h=None):
    """Why the status / plateau cut drops these per-run current table rows (one run or board)."""
    ok = rows[rows["status"] == "ok"]
    if ok.empty:
        return "status %s" % "/".join(sorted(set(rows["status"].astype(str))))
    plateau = ok["plateau_h"].astype(float)
    if plateau.isna().all():
        return "no HV plateau"
    return "HV plateau %.1f h < %g h" % (plateau.max(), min_plateau_h)


def july_run_selection(good_runs_csv=None, steps=None):
    """Per telescope, the July runs drawn at the irradiation steps `steps` (default: JULY_STEPS).

    A run is drawn when it is on the good-run list (GOOD_RUNS_CSV) and, at its fluence, on the
    display-run list (DISPLAY_RUNS_JUL_CSV) with RFSel JULY_STEPS_RFSEL and threshold offset
    JULY_STEPS_OFFSET. Only the boards with status "ok" and an HV plateau of at least
    JULY_MIN_PLATEAU_H hours in the per-run current table (JULY_REF_CSV) are drawn: a run with no
    such board goes to `rejected` with the reason, a drawn run's other boards to `dropped_boards`.
    A good run off the display list goes to `rejected` with that reason.

    Returns dict tel -> dict(chosen=[run...], rejected=["run (fluence: reason)"...],
    not_good=[(run, reason)...], candidates=[dict(run, fluence, plateau_h, good, kept)...],
    dropped_boards=["run/chip (fluence: reason)"...]), plus per-run/board window + bias +
    reference dict keyed (tel, run, chip) -> dict(start, end, bias_V, ref_uA, plateau_h).
    """
    steps = JULY_STEPS if steps is None else tuple(steps)
    ref = pd.read_csv(JULY_REF_CSV)
    ref = ref[ref["fluence_p_cm2"].astype(float).isin(list(steps))]
    df = ref[(ref["status"] == "ok") & ref["plateau_h"].notna()]
    df = df[df["plateau_h"].astype(float) >= JULY_MIN_PLATEAU_H]
    info = {}
    selection = {}
    for tel_key, tel_name in JULY_TEL.items():
        good, reasons = good_runs(tel_key, good_runs_csv)
        sub = df[df["tel"] == tel_name]
        ref_tel = ref[ref["tel"] == tel_name]
        chosen_runs = []
        rejected_runs = []
        not_good = []
        candidates = []
        dropped_boards = []
        for fluence in steps:
            fluence_label = FLUENCE_TEXT_PLAIN.get(fluence, "%g" % fluence)
            fsub = sub[sub["fluence_p_cm2"].astype(float) == fluence]
            fref = ref_tel[ref_tel["fluence_p_cm2"].astype(float) == fluence]
            per_run_plateau_all = fsub.groupby("run")["plateau_h"].max().sort_values(ascending=False)
            for r, p_h in per_run_plateau_all.items():
                rn = _run_num(r)
                candidates.append(dict(run=r, fluence=fluence_label, plateau_h=float(p_h),
                                       good=(rn in good), kept=False))
                if rn not in good:
                    not_good.append((r, reasons.get(rn, "not on the good-run list")))
            per_run_plateau = per_run_plateau_all[[
                (_run_num(r) in good) for r in per_run_plateau_all.index]]
            runs = list(per_run_plateau.index)
            display_runs = _display_runs_jul(fluence).get(tel_key, set())
            keep = [r for r in runs if _run_num(r) in display_runs]
            drop = [r for r in runs if _run_num(r) not in display_runs]
            chosen_runs += keep
            rejected_runs += ["%s (%s: not on the display-run list at RFSel %s / offset %s)"
                              % (r, fluence_label, JULY_STEPS_RFSEL, JULY_STEPS_OFFSET)
                              for r in drop]
            for run in sorted(set(fref["run"]) - set(fsub["run"]), key=_run_num):
                rejected_runs.append("%s (%s: %s)" % (
                    run, fluence_label, _cut_reason(fref[fref["run"] == run], JULY_MIN_PLATEAU_H)))
            for run in keep:
                for idx, row in fref[fref["run"] == run].iterrows():
                    if idx not in fsub.index:
                        dropped_boards.append("%s/%s (%s: %s)" % (
                            run, _chip_from_board(row["board"]), fluence_label,
                            _cut_reason(fref.loc[[idx]], JULY_MIN_PLATEAU_H)))
        kept_set = set(chosen_runs)
        for c in candidates:
            c["kept"] = c["run"] in kept_set
        selection[tel_key] = dict(chosen=chosen_runs, rejected=rejected_runs,
                                  not_good=not_good, candidates=candidates,
                                  dropped_boards=dropped_boards)
        for run in chosen_runs:
            rrows = sub[sub["run"] == run]
            for _, row in rrows.iterrows():
                chip = row["board"].split("_")[-1] if "_" in str(row["board"]) else row["board"]
                start = pd.Timestamp(row["plateau_start_utc"])
                end = pd.Timestamp(row["plateau_end_used_utc"])
                info[(tel_key, run, chip)] = dict(
                    start=start, end=end, assert_start=start + timedelta(minutes=SETTLE_MIN),
                    assert_end=end, bias_V=float(row["v_plateau_V"]),
                    ref_uA=float(row["i_inrun_uA"]), plateau_h=float(row["plateau_h"]),
                    fluence=row["fluence_p_cm2"])
    return selection, info


_YAML_CACHE = {}


def _board_yaml(path):
    if path not in _YAML_CACHE:
        import yaml
        with open(path) as fh:
            _YAML_CACHE[path] = yaml.safe_load(fh)
    return _YAML_CACHE[path]


def run_settings(run_key, yaml_path=None):
    """{(RFSel, offset), ...} over the four boards of one run ("h1_run17"), from a board-config
    yaml (default: the campaign's July one). RFSel is None where the yaml does not record it; a
    run missing from the yaml gives an empty set."""
    cfg = _board_yaml(yaml_path or JULY_YAML).get(run_key, {})
    if not isinstance(cfg, dict):
        return set()
    return {(d.get("RFSel"), d.get("offset")) for d in cfg.values() if isinstance(d, dict)}


def july_run_settings(run_key):
    """run_settings() of one July run."""
    return run_settings(run_key, JULY_YAML)


def check_offset(run_keys, yaml_path, offset):
    """Raise RuntimeError unless every board of every run in `run_keys` ("h1_run17") ran at
    threshold offset `offset` in the board-config yaml `yaml_path`. A footer that states the
    offset calls this for the runs it draws."""
    bad = {}
    for key in run_keys:
        got = sorted({off for _rfsel, off in run_settings(key, yaml_path)}, key=str)
        if got != [offset]:
            bad[key] = got
    if bad:
        raise RuntimeError("threshold offset is not %s on every board in %s: %s (an empty list "
                           "means the run is not in the yaml)"
                           % (offset, os.path.basename(yaml_path), bad))


def july_1p5e15_selection(good_runs_csv=None):
    """Same (selection, info) shape as july_run_selection(), for the July 1.5e15 runs only: the
    good runs with RFSel JULY_1P5E15_RFSEL and offset JULY_1P5E15_OFFSET on every board (the
    campaign yaml), boards with status "ok" in the per-run current table (no plateau minimum).
    A run with no such board goes to `rejected` with its status, a drawn run's other boards to
    `dropped_boards`."""
    ref_all = pd.read_csv(JULY_REF_CSV)
    ref_all = ref_all[ref_all["fluence_p_cm2"].astype(float) == 1.5e15]
    ref = ref_all[ref_all["status"] == "ok"]
    selection, info = {}, {}
    for tel_key, tel_name in JULY_TEL.items():
        good, reasons = good_runs(tel_key, good_runs_csv)
        sub = ref[ref["tel"] == tel_name]
        sub_all = ref_all[ref_all["tel"] == tel_name]
        chosen, rejected, not_good, candidates, dropped_boards = [], [], [], [], []
        for run in sorted(set(sub_all["run"]) - set(sub["run"]), key=_run_num):
            rejected.append("%s (%s)" % (run, _cut_reason(sub_all[sub_all["run"] == run])))
        for run in sorted(sub["run"].unique(), key=_run_num):
            rn = _run_num(run)
            settings = july_run_settings(run)
            ok_settings = settings == {(JULY_1P5E15_RFSEL, JULY_1P5E15_OFFSET)}
            rows = sub[sub["run"] == run]
            candidates.append(dict(run=run, fluence="1.5e15",
                                   plateau_h=float(rows["plateau_h"].max()),
                                   good=(rn in good), settings=sorted(map(list, settings)),
                                   kept=False))
            if rn not in good:
                not_good.append((run, reasons.get(rn, "not on the good-run list")))
                rejected.append(run)
                continue
            if not ok_settings:
                rejected.append(run + " (not RFSel %d / offset %d on every board)"
                                % (JULY_1P5E15_RFSEL, JULY_1P5E15_OFFSET))
                continue
            chosen.append(run)
            for idx, row in sub_all[sub_all["run"] == run].iterrows():
                if idx not in rows.index:
                    dropped_boards.append("%s/%s (%s)" % (run, _chip_from_board(row["board"]),
                                                          _cut_reason(sub_all.loc[[idx]])))
            for _, row in rows.iterrows():
                chip = row["board"].split("_")[-1] if "_" in str(row["board"]) else row["board"]
                start = pd.Timestamp(row["plateau_start_utc"])
                end = pd.Timestamp(row["plateau_end_used_utc"])
                info[(tel_key, run, chip)] = dict(
                    start=start, end=end, assert_start=start + timedelta(minutes=SETTLE_MIN),
                    assert_end=end, bias_V=float(row["v_plateau_V"]),
                    ref_uA=float(row["i_inrun_uA"]), plateau_h=float(row["plateau_h"]),
                    fluence=1.5e15)
        for c in candidates:
            c["kept"] = c["run"] in chosen
        selection[tel_key] = dict(chosen=chosen, rejected=rejected, not_good=not_good,
                                  candidates=candidates, dropped_boards=dropped_boards)
    return selection, info


def check_holds_last_to(holds, chips, t_end, what):
    """Raise RuntimeError unless the hold of every chip in `chips` ({chip: dict(end=timestamp)})
    lasts until t_end: a current-limit figure whose footer says the ramp-down after the hold is
    not drawn must end its panels no later than that."""
    early = ["%s (hold ends %s)" % (c, holds[c]["end"]) for c in chips if holds[c]["end"] < t_end]
    if early:
        raise RuntimeError("%s: the panels run to %s, after the hold of %s ends, so its ramp-down "
                           "would be drawn; end the panels earlier or reword the footer"
                           % (what, t_end, ", ".join(early)))


SPIKE_SETTLE_MIN = SETTLE_MIN     # the spike scan skips the same settle time
SPIKE_FACTOR = 1.25         # |I| > this x the run's chip median counts as "above"
SPIKE_MAX_MIN = 30.0        # a contiguous above-threshold group this long or longer is an
                            # "excursion" (real physics/HV behaviour); shorter is a "spike"


def _chip_from_board(board):
    """'PT_IH12' -> 'IH12' (board strings in july_timeline.json's per_run table)."""
    s = str(board)
    return s.split("_", 1)[-1] if "_" in s else s


def _per_run_windows(per_run=None):
    """july_timeline.json 'per_run' (status=='ok' rows only, each already carrying both plateau
    bounds) -> {(tel_key, run, chip): dict(tel, channel, settle_start, end, bias_V)}.

    settle_start = plateau_start_utc + 15 min, matching the assert_start convention
    of july_run_selection and of the in-run current figures (iv21-iv23);
    bias_V is v_plateau_V, falling back to hv_nominal_V on the rare row missing it.
    """
    if per_run is None:
        with open(JULY_TIMELINE_JSON) as fh:
            per_run = json.load(fh)["per_run"]
    out = {}
    for e in per_run:
        if e.get("status") != "ok":
            continue
        p_start, p_end = e.get("plateau_start_utc"), e.get("plateau_end_used_utc")
        if not p_start or not p_end:
            continue
        bias = e.get("v_plateau_V")
        if bias is None:
            bias = e.get("hv_nominal_V")
        if bias is None:
            continue
        start = pd.Timestamp(p_start)
        end = pd.Timestamp(p_end)
        settle_start = start + timedelta(minutes=SPIKE_SETTLE_MIN)
        if settle_start >= end:
            continue
        out[(e["tel"].lower(), e["run"], _chip_from_board(e["board"]))] = dict(
            tel=e["tel"], channel=int(e["channel"]), settle_start=settle_start, end=end,
            bias_V=float(bias))
    return out


def detect_july_spikes(timeline=None, per_run=None):
    """In-run current spike/excursion scan over every July run's 60 s median series (thresholds:
    the SPIKE_* constants). A spike is a contiguous group of post-settle bins where |I| exceeds
    SPIKE_FACTOR x the run's own chip median (over the same post-settle window); a group shorter
    than SPIKE_MAX_MIN minutes is a "spike", one that long or longer is an "excursion" (kept
    distinct: an excursion is more likely a real, sustained HV/physics effect than a brief glitch,
    but neither is filtered here; that judgement is the caller's).

    Returns a list of dicts (tel, run, chip, kind, start_utc, end_utc, duration_min, peak_uA,
    median_uA), one row per contiguous above-threshold group, sorted by (tel, run, chip,
    start_utc). Never tunes a window to make a run "look clean"; a run with no group above
    threshold contributes no rows.
    """
    timeline = load_july_timeline() if timeline is None else timeline
    windows = _per_run_windows(per_run)
    rows = []
    for (tel_key, run, chip), w in sorted(windows.items()):
        m = ((timeline["tel"] == w["tel"]) & (timeline["channel"] == w["channel"])
             & (timeline["tb"] >= w["settle_start"]) & (timeline["tb"] <= w["end"])
             & ((timeline["v"] - w["bias_V"]).abs() <= BIAS_TOL_V))
        sub = timeline.loc[m].sort_values("tb").reset_index(drop=True)
        if sub.empty:
            continue
        med = float(sub["i_uA"].median())
        if not (med == med) or med == 0:      # NaN or zero baseline: nothing to compare against
            continue
        above = (sub["i_uA"].abs() > SPIKE_FACTOR * abs(med)).to_numpy()
        i, n = 0, len(above)
        while i < n:
            if not above[i]:
                i += 1
                continue
            j = i
            while j < n and above[j]:
                j += 1
            grp = sub.iloc[i:j]
            start_ts, end_ts = grp["tb"].iloc[0], grp["tb"].iloc[-1]
            duration_min = (end_ts - start_ts).total_seconds() / 60.0 + 1.0   # inclusive of both bins
            kind = "excursion" if duration_min >= SPIKE_MAX_MIN else "spike"
            rows.append(dict(tel=w["tel"], run=run, chip=chip, kind=kind,
                             start_utc=start_ts.strftime("%Y-%m-%dT%H:%M:%S"),
                             end_utc=end_ts.strftime("%Y-%m-%dT%H:%M:%S"),
                             duration_min=round(duration_min, 1),
                             peak_uA=round(float(grp["i_uA"].abs().max()), 2),
                             median_uA=round(med, 2)))
            i = j
    return rows


def assert_july():
    selection, info = july_run_selection()
    timeline = load_july_timeline()
    results = []
    ok = True
    for tel_key, tel_name in JULY_TEL.items():
        print("\n%s: chosen %s, rejected %s" % (
            tel_key.upper(), selection[tel_key]["chosen"], selection[tel_key]["rejected"]))
        for run in selection[tel_key]["chosen"]:
            for c, chip in enumerate(TELESCOPE_CHIPS[tel_key]):
                key = (tel_key, run, chip)
                if key not in info:
                    continue
                w = info[key]
                m = ((timeline["tel"] == tel_name) & (timeline["channel"] == c)
                     & (timeline["tb"] >= w["start"]) & (timeline["tb"] <= w["end"])
                     & ((timeline["v"] - w["bias_V"]).abs() <= BIAS_TOL_V))
                sub = timeline.loc[m]
                am = (sub["tb"] >= w["assert_start"]) & (sub["tb"] <= w["assert_end"])
                drawn = sub.loc[am, "i_uA"].median()
                ref = w["ref_uA"]
                pct = 100.0 * (drawn - ref) / ref if pd.notna(drawn) else float("nan")
                flag = "" if pd.notna(pct) and abs(pct) <= 5.0 else "  <-- FAIL"
                if flag:
                    ok = False
                print("  %-8s %-6s %10.1f %10.1f %7.2f%%%s" % (run, chip, drawn, ref, pct, flag))
                results.append(dict(tel=tel_key, run=run, chip=chip,
                                    drawn_uA=round(float(drawn), 2), ref_uA=ref,
                                    pct_diff=round(float(pct), 3)))
    print("\nJuly assertion: %s" % ("ALL PASS" if ok else "FAILURES ABOVE"))
    return selection, results, ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reduce-one", choices=["%s_%s" % (tel, FLUENCE_TEXT_PLAIN[fluence])
                                             for tel, fluence in MARCH_RAW_FILES])
    ap.add_argument("--reduce-all", action="store_true")
    ap.add_argument("--assert-march", nargs="?", const="__default__")
    ap.add_argument("--assert-july", action="store_true")
    ap.add_argument("--detect-spikes", action="store_true",
                    help="scan every July run for in-run current spikes/excursions "
                         "(see detect_july_spikes); prints a summary, writes --out if given")
    ap.add_argument("--out", default=os.path.join(INPUTS, "march", "march_inrun_60s.csv"))
    a = ap.parse_args()

    if a.detect_spikes:
        spikes = detect_july_spikes()
        print("found %d spike/excursion group(s) across %d (tel,run,chip)"
             % (len(spikes), len(set((r["tel"], r["run"], r["chip"]) for r in spikes))))
        for r in spikes:
            print("  %-2s %-10s %-6s %-9s %s -> %s  %5.1f min  peak=%.1f uA  median=%.1f uA"
                 % (r["tel"], r["run"], r["chip"], r["kind"], r["start_utc"], r["end_utc"],
                    r["duration_min"], r["peak_uA"], r["median_uA"]))
        if a.out and a.out != os.path.join(INPUTS, "march", "march_inrun_60s.csv"):
            pd.DataFrame(spikes).to_csv(a.out, index=False)
            print("wrote %s (%d rows)" % (a.out, len(spikes)))

    if a.reduce_one:
        tel, flu_s = a.reduce_one.split("_", 1)
        fluence = float(flu_s)
        part = _reduce_one_file(tel, fluence)
        print(part.groupby(["run", "chip"]).size())
        if tel == "h1":
            windows = {r: dict(assert_start=pd.Timestamp(MARCH_H1_RUNS[r]["start_utc"]) + timedelta(minutes=SETTLE_MIN),
                               assert_end=pd.Timestamp(MARCH_H1_RUNS[r]["start_utc"])
                                          + timedelta(hours=MARCH_H1_RUNS[r]["plateau_h"]))
                      for r in RUNS_AT_FLUENCE[fluence]}
            ref = {r: MARCH_H1_RUNS[r]["ref_uA"] for r in RUNS_AT_FLUENCE[fluence]}
        else:
            f1w = _f1_windows()
            windows = {r: dict(assert_start=f1w[r]["win_start_utc"], assert_end=f1w[r]["win_end_utc"])
                      for r in RUNS_AT_FLUENCE[fluence]}
            ref = {r: f1w[r]["ref_uA"] for r in RUNS_AT_FLUENCE[fluence]}
        _assert_table(part, windows, ref)

    if a.reduce_all:
        reduce_all(a.out)

    if a.assert_march:
        path = None if a.assert_march == "__default__" else a.assert_march
        full = load_march(path)
        h1_windows = {r: dict(assert_start=pd.Timestamp(MARCH_H1_RUNS[r]["start_utc"]) + timedelta(minutes=SETTLE_MIN),
                              assert_end=pd.Timestamp(MARCH_H1_RUNS[r]["start_utc"])
                                         + timedelta(hours=MARCH_H1_RUNS[r]["plateau_h"]))
                     for r in MARCH_H1_RUNS}
        h1_ref = {r: MARCH_H1_RUNS[r]["ref_uA"] for r in MARCH_H1_RUNS}
        f1w = _f1_windows()
        f1_windows = {r: dict(assert_start=f1w[r]["win_start_utc"], assert_end=f1w[r]["win_end_utc"])
                     for r in f1w}
        f1_ref = {r: f1w[r]["ref_uA"] for r in f1w}
        _assert_table(full[full["tel"] == "h1"], h1_windows, h1_ref)
        _assert_table(full[full["tel"] == "f1"], f1_windows, f1_ref)

    if a.assert_july:
        assert_july()
