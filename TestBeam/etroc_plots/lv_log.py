"""The LV power log: when each telescope's LV was off.

The DAQ logs the power board's 9 V input in files of test_run_power_<local stamp>.sqlite (the
first days as .parquet), one folder per telescope, a record about every 15 s, stamped in UTC. It
records only while the LV is on and starts a new file each time the LV comes back on, so every
gap between one file's last record and the next file's first is an LV cycle, however short (in
July, from 7 s to 59 h). lv_off_windows() returns the gaps longer than the campaign's
LV_OFF_MIN_H, the LV-off periods long enough to draw on a campaign timeline.

build_lv_spans() reads the folders once into a small table (the campaign's LV_SPANS_JUL_CSV);
everything else reads the table. Read by overview.timeline (the hatched LV-off spans) and
iv.current_vs_run (the LV-on time after each irradiation step).
"""
import os
import sqlite3
import warnings

import pandas as pd

from .campaigns import active as campaign

LV_SPANS_COLUMNS = ["campaign", "tel", "file", "first_utc", "last_utc", "n_records"]


def build_lv_spans(out_csv=None, dirs=None):
    """The LV-spans table (LV_SPANS_COLUMNS): one row per test_run_power_* file in each folder
    of `dirs` (default the campaign's LV_LOG_DIRS, {(campaign, telescope): folder}), with its
    first and last record (UTC, to the second) and its number of records; a stamp with a time
    zone is converted to UTC, one without is taken as UTC. A file without records is left out
    with a warning; two files of one name stem (a .sqlite and a .parquet copy) stop the build.
    Written to `out_csv` when given. LV_SPANS_JUL_CSV was made with

        python -c "from etroc_plots import lv_log; lv_log.build_lv_spans('lv_spans_jul.csv')"

    run from TestBeam/ with read access to the folders."""
    dirs = campaign.LV_LOG_DIRS if dirs is None else dirs
    rows = []
    for (camp, tel), folder in dirs.items():
        stems = {}
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name)
            if not name.startswith("test_run_power_"):
                continue
            if name.endswith(".sqlite"):
                con = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
                stamps = pd.read_sql("SELECT timestamp FROM power_log", con)["timestamp"]
                con.close()
            elif name.endswith(".parquet"):
                stamps = pd.read_parquet(path, columns=["timestamp"])["timestamp"]
            else:
                continue
            stem = os.path.splitext(name)[0]
            if stem in stems:
                raise ValueError("%s: %s and %s are the same log file" % (folder, stems[stem], name))
            stems[stem] = name
            t = pd.to_datetime(stamps)
            if t.dt.tz is not None:
                t = t.dt.tz_convert("UTC")
            if t.isna().all():
                warnings.warn("%s: no records, left out" % path)
                continue
            rows.append(dict(campaign=camp, tel=tel, file=name,
                             first_utc=t.min().strftime("%Y-%m-%d %H:%M:%S"),
                             last_utc=t.max().strftime("%Y-%m-%d %H:%M:%S"), n_records=len(t)))
    df = pd.DataFrame(rows, columns=LV_SPANS_COLUMNS)
    df = df.sort_values(["campaign", "tel", "first_utc"]).reset_index(drop=True)
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df


def read_lv_spans(path=None):
    """LV_SPANS_JUL_CSV (or `path`) with first_utc and last_utc as UTC Timestamps."""
    df = pd.read_csv(path or campaign.LV_SPANS_JUL_CSV)
    missing = [c for c in LV_SPANS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("LV-spans table lacks the columns %s" % missing)
    for col in ("first_utc", "last_utc"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df


def lv_off_windows(tel, camp="july", min_hours=None, path=None):
    """[(start, end)] UTC Timestamps of telescope `tel`'s LV-off periods: the gaps longer than
    `min_hours` (default the campaign's LV_OFF_MIN_H) from the latest record so far to the next
    file's first, in time order (the latest, so that overlapping files make no gap)."""
    df = read_lv_spans(path)
    df = df[(df["campaign"] == camp) & (df["tel"] == tel)].sort_values("first_utc")
    limit = pd.Timedelta(hours=campaign.LV_OFF_MIN_H if min_hours is None else min_hours)
    last, first = list(df["last_utc"].cummax()), list(df["first_utc"])
    return [(a, b) for a, b in zip(last[:-1], first[1:]) if b - a > limit]


def lv_on_before(tel, t, camp="july", path=None):
    """The end of telescope `tel`'s last LV-off period longer than the campaign's LV_OFF_MIN_H
    (lv_off_windows) at or before `t` (a UTC time; one without a time zone is taken as UTC), or
    None when there is none."""
    t = pd.Timestamp(t)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    ends = [b for _, b in lv_off_windows(tel, camp=camp, path=path) if b <= t]
    return ends[-1] if ends else None
