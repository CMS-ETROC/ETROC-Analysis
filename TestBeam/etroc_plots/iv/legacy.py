"""
Legacy data format: the March-2026 IRRAD slow-control CSVs.

Wide layout, one column pair per channel:

    Date, Re(Vmeas[0]) [V], Re(Imeas[0]) [A], ..., Re(Vmeas[3]) [V], Re(Imeas[3]) [A]

sampled continuously (~1 Hz) while the supply ramps, so one file can span days
and hold many sweeps. Two consequences the new SQLite format doesn't have:

* a scan is selected by a **time window**, not by the file boundary;
* there is no `voltage_set`, so points are collapsed onto a **voltage binning**
  and the per-bin median taken (which also rejects beam-spill spikes).

`load_iv_legacy` returns the same tidy frame as `load_iv`, with the columns the
old format cannot supply (v_set, status bits) marked absent, so every downstream
function works unchanged. Binning happens later, in `build_iv_curve`, the same
way as `process_binned_iv` of the March-2026 analysis: pass `bins=` or a
`"bins"` key in the scan config.
"""

import os

import numpy as np
import pandas as pd

# The March-2026 analysis binnings, kept unchanged so results of this package and
# of that analysis are directly comparable. Negative and descending, matching the
# reverse-bias convention of the raw data.
FINE_BINS = -1 * np.array(
    np.arange(1, 10, 1).tolist()
    + np.arange(10, 60, 0.1).tolist()
    + np.arange(60, 70, 0.2).tolist()
    + np.arange(70, 80, 1).tolist()
    + np.arange(80, 130, 5).tolist()
    + np.arange(130, 520, 10).tolist()
)[::-1]

QUICK_BINS = -1 * np.array(np.arange(10, 520, 10).tolist())[::-1]


def load_iv_legacy(path, n_channels=8):
    """
    Read a legacy wide CSV into the tidy frame:

        timestamp | channel | v_set | v_meas | i_meas | i_limit
                  | hv_on | ramping | in_compliance | low_range | fault

    v_set and i_limit come back NaN and every status bit False: the old format
    records none of them. Cleaning cuts that key on status bits are no-ops, and
    the hand-tuned `i_max` cut works via build_iv_curve.
    """
    raw = pd.read_csv(path)
    if "Date" not in raw.columns:
        raise ValueError(f"{path}: no 'Date' column: not a legacy IV log")
    ts = _parse_legacy_dates(raw["Date"], path)

    frames = []
    for ch in range(n_channels):
        v_col, i_col = f"Re(Vmeas[{ch}]) [V]", f"Re(Imeas[{ch}]) [A]"
        if v_col not in raw.columns or i_col not in raw.columns:
            continue
        sub = pd.DataFrame({
            "timestamp":     ts,
            "channel":       ch,
            "v_set":         np.nan,
            "v_meas":        pd.to_numeric(raw[v_col], errors="coerce"),
            "i_meas":        pd.to_numeric(raw[i_col], errors="coerce"),
            "i_limit":       np.nan,
            "hv_on":         True,
            "ramping":       False,
            "in_compliance": False,
            "low_range":     False,
            "fault":         False,
        })
        # the logger writes rows even when a channel reports nothing
        frames.append(sub.dropna(subset=["v_meas", "i_meas"], how="all"))

    if not frames:
        raise ValueError(
            f"{path}: no 'Re(Vmeas[i]) [V]' / 'Re(Imeas[i]) [A]' column pairs found")

    tidy = (pd.concat(frames, ignore_index=True)
              .sort_values(["channel", "timestamp"])
              .reset_index(drop=True))
    tidy.attrs["source"] = os.path.basename(str(path))
    tidy.attrs["format"] = "legacy-csv"
    return tidy


def up_sweep_window(t, v_abs, tol_V=0.5):
    """
    (first, last) time of the single up-sweep in one channel's readings.

    t and v_abs are the channel's time stamps and |V|, equal length, in time order, no NaN.
    A legacy log often holds a ramp down before the sweep and the ramp back down after it, and
    those samples pass through the same low voltages. The series is split wherever |V| falls
    more than tol_V below the running maximum of the current piece, so every ramp down breaks
    into short pieces of small |V| span; the piece with the largest |V| span (max - min) is the
    sweep, and on a tie the one with more samples wins.
    """
    t = np.asarray(t)
    v = np.asarray(v_abs, dtype=float)
    if len(v) == 0 or len(t) != len(v):
        raise ValueError("up_sweep_window: t and |V| must be non-empty and of equal length "
                         "(got %d and %d)" % (len(t), len(v)))
    if not np.isfinite(v).all():
        raise ValueError("up_sweep_window: |V| holds NaN or inf; drop those samples first")
    best = None                       # (span, n_samples, first index, last index)
    first, run_max, run_min = 0, v[0], v[0]
    for k in range(1, len(v) + 1):
        if k == len(v) or v[k] < run_max - tol_V:
            piece = (run_max - run_min, k - first, first, k - 1)
            if best is None or piece[:2] > best[:2]:
                best = piece
            if k < len(v):
                first, run_max, run_min = k, v[k], v[k]
        else:
            run_max = max(run_max, v[k])
            run_min = min(run_min, v[k])
    return t[best[2]], t[best[3]]


def bin_iv(df, bins, agg="median", min_count=1):
    """
    Collapse cleaned readings of ONE channel onto a voltage binning.

    This is the March-2026 analysis binning: pd.cut on the measured voltage, then
    the per-bin median current (median so a beam spill during the dwell at one
    voltage doesn't drag the point). Returns V_mean, I_filtered, n, i_spread,
    the same schema build_iv_curve produces for the new format.
    """
    d = df.dropna(subset=["v_meas", "i_meas"])
    if d.empty or bins is None or len(bins) < 2:
        return None
    out = (d.assign(_bin=pd.cut(d["v_meas"], bins=np.sort(np.asarray(bins))))
             .groupby("_bin", observed=False)
             .agg(V_mean=("v_meas", "mean"),
                  I_filtered=("i_meas", agg),
                  n=("i_meas", "size"),
                  i_spread=("i_meas", "std"))
             .dropna(subset=["V_mean", "I_filtered"])
             .reset_index(drop=True))
    out = out[out["n"] >= min_count]
    return out.sort_values("V_mean").reset_index(drop=True)


def _parse_legacy_dates(col, path=""):
    """
    The old logger wrote two timestamp styles:

        03/25/2026 06:05:45.000      full date + time
        07:51:25.000                 time of day only (no date at all)

    Time-only stamps are parsed onto a dummy date (pandas puts them on
    1900-01-01), with midnight wraps detected by backwards jumps and rolled
    into the next day so a log crossing 00:00 stays monotonic. start/end
    windows then also need to be time-only for such files ("07:55:00", not
    "03/25/2026 07:55:00"), and clean_channel handles that by re-parsing the
    window with the same rule.

    A silent all-NaT parse is refused: it would turn every start/end window
    into a no-op and quietly bin unrelated historical data into the curve.
    """
    ts = pd.to_datetime(col, format="%m/%d/%Y %H:%M:%S.%f", errors="coerce")
    if ts.notna().any():
        return ts
    ts = pd.to_datetime(col, format="%H:%M:%S.%f", errors="coerce")
    if ts.isna().all():
        ts = pd.to_datetime(col, errors="coerce")
        if ts.isna().all():
            raise ValueError(
                f"{path}: cannot parse any 'Date' value "
                f"(first is {col.dropna().iloc[0]!r}); "
                f"expected 'MM/DD/YYYY HH:MM:SS.fff' or 'HH:MM:SS.fff'")
        return ts
    # time-only: roll midnight wraps forward so the axis stays monotonic
    wraps = (ts.diff().dt.total_seconds() < -1).cumsum().fillna(0)
    return ts + pd.to_timedelta(wraps, unit="D")


def infer_bins(df, plateau_tol=0.05, min_dwell=3, pad_frac=0.5):
    """
    Derive a voltage binning from the sweep's own dwell structure.

    The old logger samples ~1 Hz while the supply dwells at each set point, so
    a sweep shows up as plateaus in v_meas. This walks the readings in time
    order, calls a new plateau whenever the voltage moves more than plateau_tol
    from the running plateau mean, and places one bin edge halfway between
    neighbouring plateaus. The result adapts to any step scheme (0.1 V here,
    2 V there) with exactly one bin per dwell, where a fixed FINE_BINS /
    QUICK_BINS grid would merge dwells in its coarse regions and leave empty
    bins in its fine ones.

    df         : cleaned readings of ONE channel (use clean_channel first so a
                 historical head/tail doesn't contribute plateaus)
    plateau_tol: max |v - plateau mean| still counted as the same dwell [V]
    min_dwell  : plateaus with fewer samples than this are ignored (transits)
    pad_frac   : outermost edges extend past the end plateaus by this fraction
                 of the neighbouring step

    Returns sorted ascending bin edges for bin_iv / build_iv_curve.
    """
    d = df.dropna(subset=["v_meas"]).sort_values("timestamp")
    v = d["v_meas"].to_numpy(dtype=float)
    if len(v) < min_dwell * 2:
        return None

    plateaus, cur = [], [v[0]]
    for x in v[1:]:
        if abs(x - np.mean(cur)) > plateau_tol:
            if len(cur) >= min_dwell:
                plateaus.append(np.mean(cur))
            cur = [x]
        else:
            cur.append(x)
    if len(cur) >= min_dwell:
        plateaus.append(np.mean(cur))

    p = np.unique(np.asarray(plateaus))
    if len(p) < 2:
        return None
    mid = (p[:-1] + p[1:]) / 2.0
    first = p[0] - pad_frac * (p[1] - p[0])
    last = p[-1] + pad_frac * (p[-1] - p[-2])
    return np.concatenate([[first], mid, [last]])


def interpolate_legacy_csv(in_path, out_path):
    """
    Produce the *_interpolated companion of a raw slow-control CSV.

    The logger writes V rows and I rows asynchronously (~half the cells of any
    row are empty), and the usual pipeline fills them by linear interpolation
    in TIME per column before analysis. Some raw files ship without that step:
    202607160020_fineIV_interpolate.csv, despite its name, is byte-for-byte
    the raw log plus a longer tail. This reimplements the fill, validated
    against the 202607152150 raw/interpolated pair: every raw token is
    preserved as written, every filled cell matches the reference file to within
    one instrument readback quantum (V: 0.01 V, I: 5e-9 A; the residuals sit
    where the reference tool interpolated over the logger's occasionally
    out-of-order timestamps, which are sorted here first).

    Handles both timestamp styles (time-only or with a date part). No
    extrapolation: cells before a column's first sample or after its last stay
    empty.
    """
    lines = open(in_path).read().splitlines()
    hdr, rows = lines[0], [l.split(",") for l in lines[1:] if l.strip()]
    dates = [r[0] for r in rows]
    fmt = "%m/%d/%Y %H:%M:%S.%f" if "/" in dates[0] else "%H:%M:%S.%f"
    t = pd.to_datetime(pd.Series(dates), format=fmt).astype("int64").to_numpy() / 1e9
    vals = np.full((len(rows), 8), np.nan)
    for i, r in enumerate(rows):
        for j in range(8):
            cell = r[j + 1].strip()
            if cell:
                vals[i, j] = float(cell)
    order = np.argsort(t, kind="stable")
    filled = vals.copy()
    for j in range(8):
        col = vals[:, j]
        if (~np.isnan(col)).sum() < 2:
            continue
        vs = col[order]
        ok = ~np.isnan(vs)
        y = np.interp(t, t[order][ok], vs[ok])
        y[t < t[order][ok][0]] = np.nan
        y[t > t[order][ok][-1]] = np.nan
        filled[:, j] = np.where(np.isnan(col), y, col)
    out = [hdr]
    for i, r in enumerate(rows):
        cells = [r[0]]
        for j in range(8):
            raw = r[j + 1].strip()
            if raw:
                cells.append(r[j + 1])            # raw token as written
            elif np.isnan(filled[i, j]):
                cells.append("")
            else:
                x = filled[i, j]
                cells.append(f"{x:.5f}" if j % 2 == 0 else f"{x:.6e}")
        out.append(",".join(cells))
    open(out_path, "w").write("\n".join(out) + "\n")
    print(f"Interpolated {len(rows)} rows -> {out_path}")
