"""iv_data.py: loaders and scan catalogue for the IV and V_gl figures.

Two on-disk formats, one in-memory shape. March slow-control scans are pre-binned CSVs with
columns V_ch0,I_ch0,...,V_ch3,I_ch3 (volts negative, amps negative, one row per voltage bin,
channel columns not aligned row-for-row; each channel is its own series in ascending V, largest
|V| first, padded to a common row count, so a row is NOT a shared voltage point across
channels). July scans are already binned, positive, in INPUTS/july/iv_curves.json. Both loaders
return {chip_name: (v_abs_volts, i_abs_uA)}, ascending in V, so every figure draws off one code
path regardless of source.

The March tree holds no binned CSV of the F1 1.5e15 scans. The inputs folder holds the three that
legacy.binned_table makes from the raw logs of that step (`python -m etroc_plots.iv.legacy`),
INPUTS/march/<stem>_binned_iv_data.csv, in this same 8-column format; they load through the
same March loader as everything else.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


# ---- campaign catalogue (see campaigns/) --------------------------------
# These names are defined by the campaign module and re-bound here, so this module and the
# notebook use them as ivd.<NAME>.
from ..campaigns import active as _campaign
INPUTS = _campaign.INPUTS            # the campaign's input folder (ETROC_IV_INPUTS overrides it)
INPUTS_JULY = _campaign.INPUTS_JULY
INPUTS_MARCH = _campaign.INPUTS_MARCH
INPUTS_VGL = _campaign.INPUTS_VGL
MARCH_EOS_WEEK2 = _campaign.MARCH_EOS_WEEK2
MARCH_EOS_15E14 = _campaign.MARCH_EOS_15E14
TELESCOPE_CHIPS = _campaign.TELESCOPE_CHIPS
LINE1_PARKED = _campaign.LINE1_PARKED
RAD_STOP_UTC = _campaign.RAD_STOP_UTC
MARCH_SCANS = _campaign.MARCH_SCANS
INPUTS_JULY_FINE = _campaign.INPUTS_JULY_FINE
JULY_FINE_SCANS = _campaign.JULY_FINE_SCANS
JULY_SCANS = _campaign.JULY_SCANS
PREIRRAD_LOG_UTC_OFFSET_H = _campaign.PREIRRAD_LOG_UTC_OFFSET_H















FINE_LOWV_MAX_V = 80.0   # bin ceiling (V); the scans coarsen past ~70 V
MIN_N_JULY_FINE = 3     # minimum per-bin sample count (~3 s dwell); thinner bins are interpolated




# ---------------------------------------------------------------------------- inputs
def _read_problem(path):
    """None if `path` (a file or a folder) can be read, else "missing" or "unreadable (why)"."""
    try:
        if os.path.isdir(path):
            os.listdir(path)
        else:
            with open(path, "rb") as fh:
                fh.read(1)
    except (FileNotFoundError, NotADirectoryError):
        return "missing"
    except OSError as err:
        return "unreadable (%s)" % (err.strerror or err)
    return None


def check_inputs(inputs=None, manifest=None, raw_inputs=None):
    """Check every input the campaign reads before any figure is drawn.

    Two sets: the tables in INPUTS, against the campaign's checksum list (INPUTS_MANIFEST), and
    the raw files read in place (the campaign's RAW_INPUTS: the March scans and slow-control
    logs, the July board-config yaml, any table moved by its own environment variable). Every
    file must exist and be readable; all problems are collected, missing apart from unreadable
    (no permission), and raised together. A table whose content differs from the published one
    (a cache you rebuilt, say) is reported and the run goes on, but the figures drawn from it
    will differ. The arguments default to the campaign's INPUTS, INPUTS_MANIFEST and RAW_INPUTS.
    """
    inputs = INPUTS if inputs is None else inputs
    manifest = _campaign.INPUTS_MANIFEST if manifest is None else manifest
    raw_inputs = _campaign.RAW_INPUTS if raw_inputs is None else raw_inputs
    with open(manifest) as fh:
        listed = [line.split(None, 1) for line in fh if line.strip()]
    missing, unreadable, changed = [], [], []

    def note(path, problem):
        if problem == "missing":
            missing.append(path)
        else:
            unreadable.append("%s: %s" % (path, problem))

    for md5, name in listed:
        name = name.strip()
        path = os.path.join(inputs, name)
        problem = _read_problem(path)
        if problem:
            note(path, problem)
            continue
        digest = hashlib.md5()
        try:
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    digest.update(block)
        except OSError as err:
            note(path, "unreadable (%s)" % (err.strerror or err))
            continue
        if digest.hexdigest() != md5:
            changed.append(name)
    for path in raw_inputs:
        problem = _read_problem(path)
        if problem:
            note(path, problem)
    if missing or unreadable:
        raise RuntimeError(
            "input check failed for campaign %s (tables folder %s, %d tables and %d raw inputs "
            "checked):\n%s\nMissing: point the campaign's input variables at a full copy; "
            "campaigns/%s_inputs.md lists every variable and its default. "
            "Unreadable: ask the owner of that area for read access."
            % (_campaign.__name__, inputs, len(listed), len(raw_inputs),
               "\n".join(["  missing: %s" % p for p in missing]
                         + ["  unreadable: %s" % p for p in unreadable]),
               _campaign.__name__.split(".")[-1]))
    for name in changed:
        print("note: %s differs from the published file, so figures drawn from it will too"
              % name)


# ---------------------------------------------------------------------------- loaders
def load_march(path, tel):
    """8-column binned March CSV -> {chip: (v_abs_V, i_abs_uA)}, ascending in V.

    Each channel column pair is its own series (rows are NOT a shared voltage point across
    channels; see module docstring); NaNs (padding / empty bins) are dropped per channel.
    """
    df = pd.read_csv(path)
    chips = TELESCOPE_CHIPS[tel]
    out = {}
    for ch, chip in enumerate(chips):
        vcol, icol = "V_ch%d" % ch, "I_ch%d" % ch
        if vcol not in df.columns or icol not in df.columns:
            continue
        v = pd.to_numeric(df[vcol], errors="coerce").to_numpy(dtype=float)
        i = pd.to_numeric(df[icol], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(v) & np.isfinite(i)
        v, i = np.abs(v[m]), np.abs(i[m]) * 1e6
        order = np.argsort(v)
        out[chip] = (v[order], i[order])
    return out


_JULY_CURVES_CACHE = None


def july_curves():
    global _JULY_CURVES_CACHE
    if _JULY_CURVES_CACHE is None:
        with open(os.path.join(INPUTS_JULY, "iv_curves.json")) as fh:
            _JULY_CURVES_CACHE = json.load(fh)
    return _JULY_CURVES_CACHE


def load_july(tel_scan_key, tel):
    """iv_curves.json[tel_scan_key] -> ({chip: (v_V, i_uA)}, meta dict). Already positive, uA."""
    entry = july_curves()[tel_scan_key]
    chips = TELESCOPE_CHIPS[tel]
    out = {}
    for chip in chips:
        c = entry["curves"].get("PT_" + chip)
        if c is None:
            continue
        v = np.asarray(c["v"], dtype=float)
        i = np.asarray(c["i_uA"], dtype=float)
        order = np.argsort(v)
        out[chip] = (v[order], i[order])
    return out, entry["meta"]


_JULY_FINE_CACHE = {}


def load_july_fine(key, tel):
    """Bin a raw July legacy slow-control CSV (JULY_FINE_SCANS) onto 0.1 V bins over the
    genuinely finely-stepped part of the scan (kfactor.fine_step_span, the same
    contiguous-<=0.5V-run detector the iv06_vgl_method figure uses: these logs run fine 0.1 V steps
    from a few V up to ~70 V, then open out to ~2-5 V steps on the way to breakdown), capped
    at FINE_LOWV_MAX_V. Time-windowed to the scan's own start first, each channel then cut to its
    single up-sweep (legacy.up_sweep_window) and binned with bin_iv, median aggregation: the
    recipe of legacy.binned_table, which bins the F1 1.5e15 March logs. The logs also hold the
    ramp down before the sweep and the ramp down after it, which pass through the same low
    voltages. meta["n_ramp_dropped"] counts, per chip, the samples cut there that fall inside the
    binned range.
    Returns {chip: (v_abs_V, i_abs_uA)}, meta dict shaped like load_july's own (type/scan), so
    iv_plot.load_looks draws both sources through the same code path.
    """
    cache_key = (key, tel)
    if cache_key in _JULY_FINE_CACHE:
        return _JULY_FINE_CACHE[cache_key]
    from .legacy import load_iv_legacy, bin_iv, up_sweep_window          # noqa: E402
    from .kfactor import fine_step_span        # noqa: E402
    from ._helpers import _parse_window        # noqa: E402

    spec = JULY_FINE_SCANS[key][tel]
    tidy = load_iv_legacy(spec["path"])
    tidy = tidy[tidy["timestamp"] >= _parse_window(spec["start"], tidy["timestamp"])]
    chips = TELESCOPE_CHIPS[tel]
    out = {}
    fine_hi_by_chip = {}
    n_filled_by_chip = {}
    n_ramp_dropped_by_chip = {}
    for ch, chip in enumerate(chips):
        sub = tidy[tidy["channel"] == ch]
        sub = sub[np.isfinite(sub["v_meas"].to_numpy(dtype=float))].sort_values("timestamp")
        if sub.empty:
            raise ValueError("%s: channel %d (%s) has no voltage readings after %s"
                             % (spec["path"], ch, chip, spec["start"]))
        v_abs = sub["v_meas"].abs().to_numpy(dtype=float)
        try:
            t0, t1 = up_sweep_window(sub["timestamp"].to_numpy(), v_abs)
        except ValueError as err:
            raise ValueError("%s: channel %d (%s): %s" % (spec["path"], ch, chip, err)) from None
        in_sweep = ((sub["timestamp"] >= t0) & (sub["timestamp"] <= t1)).to_numpy()
        ramp_v = v_abs[~in_sweep]
        sub = sub[in_sweep]
        v_raw = sub["v_meas"].dropna().to_numpy(dtype=float)
        _, hi = fine_step_span(v_raw, max_step=0.5)
        v_max = min(FINE_LOWV_MAX_V, hi) if np.isfinite(hi) else FINE_LOWV_MAX_V
        fine_hi_by_chip[chip] = float(v_max)
        n_ramp_dropped_by_chip[chip] = int((ramp_v <= v_max).sum())
        bins = -1 * np.arange(0.0, v_max + 0.05, 0.1)
        b = bin_iv(sub, bins, agg="median")
        if b is None or b.empty:
            continue
        # Bins under MIN_N_JULY_FINE (~3 s of 1 Hz dwell) are a single (or two) raw sample(s),
        # noisy enough to read as spurious spikes on a figure. Drop them and, where they sit
        # strictly between two well-sampled bins, fill by linear interpolation between those
        # neighbours only (never extrapolated past the good range); this suppresses the artefact
        # without smoothing the knee, which sits in the densely-sampled part of the scan.
        good = b[b["n"] >= MIN_N_JULY_FINE]
        if good.empty:
            continue
        v_good = good["V_mean"].to_numpy(dtype=float)
        i_good = good["I_filtered"].to_numpy(dtype=float)
        gorder = np.argsort(v_good)
        v_good, i_good = v_good[gorder], i_good[gorder]
        bad = b[b["n"] < MIN_N_JULY_FINE]
        v_bad = bad["V_mean"].to_numpy(dtype=float)
        inside = (v_bad > v_good.min()) & (v_bad < v_good.max())
        v_fill = v_bad[inside]
        i_fill = np.interp(v_fill, v_good, i_good)
        n_filled_by_chip[chip] = int(inside.sum())
        v_all = np.concatenate([v_good, v_fill])
        i_all = np.concatenate([i_good, i_fill])
        v = np.abs(v_all)
        i = np.abs(i_all) * 1e6
        order = np.argsort(v)
        out[chip] = (v[order], i[order])
    meta = dict(type="fine", scan=os.path.basename(spec["path"]), start=spec["start"],
               window_V=[0.0, FINE_LOWV_MAX_V], fine_hi_V=fine_hi_by_chip,
               n_filled_V=n_filled_by_chip, n_ramp_dropped=n_ramp_dropped_by_chip)
    result = (out, meta)
    _JULY_FINE_CACHE[cache_key] = result
    return result


_JULY_SCAN_TABLE_CACHE = None


def july_scan_table():
    """INPUTS/july/july_iv_scans.csv: |I| at 150/400/450/480/530/550 V per scan x board."""
    global _JULY_SCAN_TABLE_CACHE
    if _JULY_SCAN_TABLE_CACHE is None:
        _JULY_SCAN_TABLE_CACHE = pd.read_csv(os.path.join(INPUTS_JULY, "july_iv_scans.csv"))
    return _JULY_SCAN_TABLE_CACHE


def load_vgl_points():
    """INPUTS/vgl/vgl_points.csv -> DataFrame(tel, chip, fluence_p_cm2, timing, vgl_V)."""
    return pd.read_csv(os.path.join(INPUTS_VGL, "vgl_points.csv"), comment="#")


VGL_TABLE_TOL_V = 0.05   # campaign V_gl tables vs vgl_points.csv


def check_vgl_table(tel, table, points=None, tol_V=VGL_TABLE_TOL_V):
    """Check a campaign V_gl table (VGL_H1 / VGL_F1) against vgl_points.csv.

    table rows are (fluence in 1e15 p/cm2, look label, [V_gl per board in TELESCOPE_CHIPS[tel] order]).
    Every entry needs exactly one row of `points` (default: load_vgl_points()) with the same
    chip, fluence and look (timing) whose vgl_V is within tol_V; every mismatch or missing
    counterpart is listed in one ValueError. Returns the number of entries checked.
    """
    points = load_vgl_points() if points is None else points
    problems = []
    n = 0
    for phi, label, values in table:
        for chip, vgl in zip(TELESCOPE_CHIPS[tel], values):
            n += 1
            rows = points[(points["chip"] == chip)
                          & np.isclose(points["fluence_p_cm2"].astype(float), phi * 1e15,
                                       rtol=1e-6, atol=0.0)
                          & (points["timing"] == label)]
            if len(rows) != 1:
                problems.append("%s %s %.2fe15 '%s': %d rows in vgl_points.csv, expected one"
                                % (tel.upper(), chip, phi, label, len(rows)))
                continue
            csv_v = float(rows["vgl_V"].iloc[0])
            if not abs(csv_v - vgl) <= tol_V + 1e-9:
                problems.append("%s %s %.2fe15 '%s': table %.2f V, vgl_points.csv %.2f V"
                                % (tel.upper(), chip, phi, label, vgl, csv_v))
    if problems:
        raise ValueError("V_gl table of campaign %s disagrees with %s (tolerance %.2f V); fix "
                         "the table or the file:\n  %s"
                         % (_campaign.__name__, os.path.join(INPUTS_VGL, "vgl_points.csv"),
                            tol_V, "\n  ".join(problems)))
    return n


def check_vgl_agreement(rows, tol_V, what):
    """Raise AssertionError unless every row's drawn V_gl (knee_V) is within tol_V of its
    tabulated value (table_vgl_V). rows: dicts with tel, chip, timing, knee_V, table_vgl_V;
    a missing value on either side fails. `what` names the figure in the message."""
    bad = []
    for r in rows:
        knee, table = r.get("knee_V"), r.get("table_vgl_V")
        ok = (knee is not None and table is not None and np.isfinite(knee) and np.isfinite(table)
              and abs(knee - table) <= tol_V)
        if not ok:
            bad.append("%s %s '%s': drawn %s V, vgl_points.csv %s V"
                       % (str(r.get("tel", "")).upper(), r.get("chip"), r.get("timing"), knee,
                          table))
    if bad:
        raise AssertionError("%s: V_gl differs from vgl_points.csv by more than %.2f V:\n  %s"
                             % (what, tol_V, "\n  ".join(bad)))


# ---------------------------------------------------------------------------- small numeric helpers
def value_at(v, i, v0):
    """Linear interpolation of i(v) at v0; NaN if v0 is outside the measured range."""
    v = np.asarray(v, dtype=float)
    i = np.asarray(i, dtype=float)
    if len(v) == 0 or v0 < v.min() or v0 > v.max():
        return float("nan")
    return float(np.interp(v0, v, i))


_MARCH_STEM_RE = __import__("re").compile(r"_(\d{2})(\d{2})(\d{4})_(\d{4})(?:\D|$)")


def march_scan_start_utc(path):
    """A March scan's own start time, in UTC, from its file stem (..._MMDDYYYY_HHMM_...).

    March slow-control logs are written in local time, PREIRRAD_LOG_UTC_OFFSET_H hours ahead of
    UTC (IV_CONVENTIONS["scan_time"]), so that offset is subtracted here; RAD_STOP_UTC is
    already UTC, which makes the two directly subtractable for a day count.  Returns None when
    the stem carries no timestamp.
    """
    m = _MARCH_STEM_RE.search(os.path.basename(str(path)))
    if not m:
        return None
    mm, dd, yyyy, hhmm = m.groups()
    local = pd.Timestamp("%s-%s-%s %s:%s:00" % (yyyy, mm, dd, hhmm[:2], hhmm[2:]))
    return local - pd.Timedelta(hours=PREIRRAD_LOG_UTC_OFFSET_H)


def markevery(n, target=12):
    """~`target` markers spread over `n` points on a dense curve."""
    return max(1, n // target)


# ---------------------------------------------------------------------------- bin widths
# quick scans bin at 10 V, fine scans at 0.1 V; every IV curve carries its bin width so
# iv_plot.assert_no_gaps can tell a real hole in the scan from a curve that is legitimately
# short (a scan that stopped early).
BIN_WIDTH_QUICK_V = 10.0
BIN_WIDTH_FINE_V = 0.1


def bin_width_march(role):
    """MARCH_SCANS role ('quick_last', 'fine_ra', 'fine_2d') -> bin width in V."""
    return BIN_WIDTH_QUICK_V if role.startswith("quick") else BIN_WIDTH_FINE_V


def bin_width_july(scan_type):
    """load_july's meta['type'] ('quick'/'fine') -> bin width in V."""
    return BIN_WIDTH_QUICK_V if scan_type == "quick" else BIN_WIDTH_FINE_V


SHADE_K_MAX = 0.65   # keep at least 35% of the base colour even at the lightest step


def shade(color, k):
    """Lighten a hex/named colour towards white by fraction k in [0, SHADE_K_MAX].

    k=0 keeps the colour (the "right after" point/curve at a fluence step); increasing k marks a
    later look at the same fluence ("+2 days", "+4 months") without leaving the fluence's hue.
    Capped at SHADE_K_MAX so the lightest step in a long ramp (seven looks, say) never
    fades into near-white.
    """
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(color)
    k = max(0.0, min(SHADE_K_MAX, k))
    return (r + (1.0 - r) * k, g + (1.0 - g) * k, b + (1.0 - b) * k)
