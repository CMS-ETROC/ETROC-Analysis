"""Data access: new SQLite format -> tidy frame, plus cleaning and curves."""

import os
import sqlite3

import numpy as np
import pandas as pd

from ._helpers import _ch_label, _as_scan_list, _mfactor, _parse_window

# any of these bits means the reading should not be trusted
_FAULT_BITS = ["emergency_off", "trip_exceeded", "input_error", "ext_inhibit",
               "current_limit_exceeded", "voltage_limit_exceeded",
               "current_bounds_exceeded", "voltage_bounds_exceeded"]


def load_iv(path, fmt="auto", **kwargs):
    """
    Read a scan of either format into the common tidy frame.

    fmt: "auto"   -> .sqlite/.db => new format, .csv/.txt => legacy
         "new"    -> force the SQLite reader
         "old"    -> force the legacy CSV reader
    """
    if fmt == "auto":
        ext = os.path.splitext(str(path))[1].lower()
        fmt = "new" if ext in (".sqlite", ".sqlite3", ".db") else               "old" if ext in (".csv", ".txt") else None
        if fmt is None:
            raise ValueError(f"Cannot guess the format of {path}; pass fmt='new' or 'old'")
    if fmt in ("new", "sqlite"):
        return load_iv_sqlite(path, **kwargs)
    if fmt in ("old", "legacy", "csv"):
        from .legacy import load_iv_legacy
        return load_iv_legacy(path, **kwargs)
    raise ValueError(f"Unknown format {fmt!r}; use 'new', 'old' or 'auto'")


def load_iv_sqlite(path, table=None):
    """
    Read one *_IVscan.sqlite into a tidy frame:

        timestamp | channel | v_set | v_meas | i_meas | i_limit
                  | hv_on | ramping | in_compliance | low_range | fault
    """
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)   # read-only: safe on live DAQ files
    try:
        if table is None:
            names = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table'", con)["name"].tolist()
            if not names:
                raise ValueError(f"No tables found in {path}")
            table = "iv_scan" if "iv_scan" in names else names[0]
        raw = pd.read_sql_query(f'SELECT * FROM "{table}"', con)
    finally:
        con.close()

    def bit(name):
        if name in raw.columns:
            return raw[name].fillna(0).astype(int).astype(bool)
        return pd.Series(False, index=raw.index)

    fault = pd.Series(False, index=raw.index)
    for b in _FAULT_BITS:
        fault |= bit(b)

    tidy = pd.DataFrame({
        "timestamp":     pd.to_datetime(raw["timestamp"], errors="coerce"),
        "channel":       raw["channel"].astype(int),
        "v_set":         raw["voltage_set"].astype(float),
        "v_meas":        raw["voltage_measured"].astype(float),
        "i_meas":        raw["current_measured"].astype(float),
        "i_limit":       raw["current_limit"].astype(float) if "current_limit" in raw else np.nan,
        "hv_on":         bit("on"),
        "ramping":       bit("ramping"),
        "in_compliance": bit("controlled_current"),   # regulating current, not voltage
        "low_range":     bit("low_current_range"),    # ammeter autorange state
        "fault":         fault,
    }).sort_values(["channel", "timestamp"]).reset_index(drop=True)

    tidy.attrs["source"] = os.path.basename(str(path))
    return tidy


def clean_channel(tidy, channel, start=None, end=None, drop_compliance=True,
                  drop_hv_off=True, drop_ramping=False, drop_fault=True,
                  v_tol=5.0, i_max=None, i_max_unit=1e-6, label="", verbose=True):
    """
    Split one channel into (good, excluded) readings.

    drop_compliance : the supply switched to current regulation, so v_meas has
                      folded back and no longer reflects the demand
    drop_hv_off     : the zero-bias readings bracketing every scan
    drop_ramping    : reading taken mid-ramp; off by default, since the readback
                      is usually already settled
    v_tol           : backstop for the compliance case: drop points whose
                      readback sits further than this from the demand (None = off)
    i_max           : legacy compliance proxy: drop points with |I| above this
                      (in i_max_unit, default uA). The old format has no status
                      bits, so the hand-tuned current cut is the only way to
                      remove compliance points there; on the new format prefer
                      the controlled_current bit and leave this None.
    """
    d = tidy[tidy["channel"] == channel].copy()
    if d.empty:
        return d, d, {}

    n_in, reasons, bad = len(d), [], pd.Series(False, index=d.index)

    def flag(mask, why):
        nonlocal bad
        mask = mask.reindex(d.index, fill_value=False) & ~bad
        if mask.any():
            reasons.append(f"{int(mask.sum())} {why}")
            bad |= mask

    if start is not None:
        flag(d["timestamp"] < _parse_window(start, d["timestamp"]), "before start")
    if end is not None:
        flag(d["timestamp"] > _parse_window(end, d["timestamp"]), "after end")
    if drop_hv_off:
        flag(~d["hv_on"], "HV off")
    if drop_ramping:
        flag(d["ramping"], "still ramping")
    if drop_fault:
        flag(d["fault"], "fault bit set")
    if drop_compliance:
        v_first = d.loc[d["in_compliance"] & ~bad, "v_set"].abs().min()
        flag(d["in_compliance"], "in current compliance")
        if reasons and np.isfinite(v_first) and "compliance" in reasons[-1]:
            reasons[-1] += f" (from |V_set| = {v_first:.0f} V)"
    if v_tol is not None and d["v_set"].notna().any():
        flag((d["v_meas"] - d["v_set"]).abs() > v_tol, f"|V_meas - V_set| > {v_tol:g} V")
    if i_max is not None:
        flag(d["i_meas"].abs() > i_max * i_max_unit, f"|I| > {i_max:g} (i_max)")
    flag(d[["v_meas", "i_meas"]].isna().any(axis=1), "missing V or I")

    good, excluded = d[~bad], d[bad]
    info = {"n_in": n_in, "n_good": len(good), "reasons": reasons}

    if verbose:
        msg = f"    ch{channel} {label}: {n_in} rows -> {len(good)} points"
        if reasons:
            msg += "  [dropped: " + "; ".join(reasons) + "]"
        print(msg)
    return good, excluded, info


def build_iv_curve(tidy, channel, group_by="auto", bins=None, agg="median",
                   **clean_kw):
    """
    Clean one channel and collapse it to an IV curve.

    Returns V_mean, I_filtered, n, i_spread (signed, in V and A), or None.

    group_by : "auto"  -> v_set when recorded (new format); else `bins` when
                          given (legacy format); else the raw readings
               "v_set" -> force grouping on the demand voltage
               "bins"  -> force pd.cut(v_meas, bins), as for legacy-format logs
               None    -> no grouping (hysteresis loops)
    bins     : voltage bin edges for the legacy path, e.g. legacy.QUICK_BINS.
               A continuous slow-control log has no set points, so without bins
               every raw sample becomes its own point.
    """
    good, _, _ = clean_channel(tidy, channel, **clean_kw)
    if good.empty:
        return None

    mode = group_by
    if mode == "auto":
        if good["v_set"].notna().all():
            mode = "v_set"
        elif bins is not None and (isinstance(bins, str) or len(bins) >= 2):
            mode = "bins"
        else:
            mode = None

    if mode == "bins":
        from .legacy import bin_iv, infer_bins
        if isinstance(bins, str) and bins == "auto":
            bins = infer_bins(good)
            if bins is None:
                return None
        return bin_iv(good, bins, agg=agg)

    key = good["v_set"] if mode == "v_set" else good["v_meas"]
    curve = (good.assign(_key=key)
                 .groupby("_key", observed=False)
                 .agg(V_mean=("v_meas", "mean"),
                      I_filtered=("i_meas", agg),
                      n=("i_meas", "size"),
                      i_spread=("i_meas", "std"))
                 .dropna(subset=["V_mean", "I_filtered"])
                 .sort_values("V_mean")
                 .reset_index(drop=True))
    return curve[curve["n"] > 0].reset_index(drop=True)


def scan_summary(scans, channel_names=None, unit="uA", v_ref=None, verbose=False):
    """Per-channel overview of one or more scans: reach, compliance onset, I at v_ref."""
    mfactor, rows = _mfactor(unit), []
    for s in _as_scan_list(scans):
        tidy = load_iv(s["file"])
        for ch in sorted(tidy["channel"].unique()):
            good, excl, _ = clean_channel(tidy, ch, verbose=verbose,
                                          **{k: v for k, v in s.items()
                                             if k in ("start", "end")})
            ch_rows = tidy[tidy["channel"] == ch]
            comp = ch_rows.loc[ch_rows["in_compliance"], "v_set"].abs()
            # per channel: channels can be configured with different limits
            lim = ch_rows["i_limit"].abs()
            lim = lim.max() * mfactor if lim.notna().any() else np.nan
            row = {
                "scan": s["label"],
                "channel": ch,
                "device": _ch_label(channel_names, ch),
                "start": tidy["timestamp"].min(),
                "n_good": len(good),
                "|V| max good [V]": good["v_meas"].abs().max() if len(good) else np.nan,
                "|V| compliance [V]": comp.min() if len(comp) else np.nan,
                f"I limit [{unit}]": lim,
            }
            if v_ref is not None and len(good):
                near = good.iloc[(good["v_meas"].abs() - abs(v_ref)).abs().argsort()[:1]]
                row[f"|I| @ {abs(v_ref):.0f} V [{unit}]"] = near["i_meas"].abs().iloc[0] * mfactor
            rows.append(row)
    return pd.DataFrame(rows).set_index(["scan", "channel"])

