"""Autocalibration baseline (BL) and noise width (NW) of every board in every run.

Two tables, described in the campaign's inputs document: BLNW_CSV, one row per board per run
(columns BLNW_COLUMNS), and BLNW_PIXELS_CSV, one row per pixel of every board with a scan in every
run (columns BLNW_PIXEL_COLUMNS: the scan's baseline and noise width of the pixel, whether
live_pixels() keeps it, the scan's start). build_blnw_table() makes both from two read-only records:

  scans   the autocalibration scan histories of the telescope PCs (BLNW_SQLITES): one scan = one
          sweep of one chip, a baseline and a noise width per pixel, in DAC codes
  runs    the DAQ's run_metadata.yaml of every run (RUN_METADATA_DIRS): run start, and per chip
          its threshold offset and the threshold DAC written to each pixel

Join rule. The DAQ programs each pixel's threshold DAC = baseline + threshold offset, from the
scan taken just before the run. So for the scan a run used, DAC - offset equals the scan's
baseline map pixel for pixel. Among the chip's scans that started before the run, the one with
the fewest mismatching pixels is taken, ties going to the latest. Pixels at the DAC rail are left
out of the comparison (DAC >= 1020 in the run metadata, or baseline >= 1020 in the scan): the
DAQ writes the rail there, not baseline + offset. `match` says how well the maps agree: exact (no
mismatch), near_exact (1 to 3 pixels, a failed single-pixel autocalibration), nearest (no scan
agrees; the closest scan before the run is used) or none (no scan before the run).

Statistics, both kept: over all 256 pixels (`*_allpix`; a railed or dead
pixel counted as it was stored, a missing noise width as 0) and over the pixels with a real
value: for BL the pixels with 50 < baseline < 1020 DAC and not railed in the run metadata, for NW
the pixels with a noise width above 0; `n_pixels_bl` and `n_pixels_nw` count them. Mean and
standard deviation over the pixels: the standard deviation is the pixel-to-pixel spread, not an
uncertainty. Over the same pixels also the median and the 16th and 84th percentiles (`bl_median`,
`bl_p16`, `bl_p84`, `nw_median`, `nw_p16`, `nw_p84`; numpy's linear interpolation between pixels),
the pixel-to-pixel spread again, not an uncertainty.

RFSel: the RFSel the scan history records where it records one, else "RFSel<N>" in the run name,
else the run-config yaml; `rfsel_src` says which, and the March records carry none of the three.
The chip is the board's chip in TELESCOPE_CHIPS, checked against the run-config yaml; the DAQ's
own chip names are used only to pair a run's chips with the scan histories.

Scan times: the scan clock runs BLNW_SQLITE_UTC_OFFSET_H hours ahead of UTC; every time in the
table is UTC.

The drawing (ladder_vs_fluence, rest_pair) colours by board (etroc_style.BOARD_COLOR, marker by
chip): the fluence step, the look or the RFSel setting (the settings notebook's st06) is the x
axis. The words on the figures are arguments.
"""
import glob
import os
import re
import sqlite3
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D

from ..campaigns import active as campaign
from .. import style
from ..etroc_style import BOARD_COLOR

NPIX = 16
BL_DEAD_MAX = 50.0      # a baseline at or below this is a failed autocalibration (stored as 0)
RAIL = 1020.0           # a baseline or DAC at or above this is the rail, not a measurement
NEAR_EXACT_MAX = 3      # mismatching pixels still called near_exact

BLNW_COLUMNS = [
    "campaign", "tel", "run", "board_idx", "chip", "start_utc", "rfsel", "rfsel_src", "offset",
    "hv", "scan_utc", "scan_gap_min", "match", "n_mismatch",
    "bl_mean_allpix", "bl_std_allpix", "nw_mean_allpix", "nw_std_allpix",
    "bl_mean", "bl_std", "n_pixels_bl", "nw_mean", "nw_std", "n_pixels_nw",
    "bl_median", "bl_p16", "bl_p84", "nw_median", "nw_p16", "nw_p84",
]
# one row per pixel of every board of every run with a scan; row, col in the chip frame (pixel
# (0, 0) at the bottom right of a map); live = live_pixels() (1 or 0)
BLNW_PIXEL_COLUMNS = ["campaign", "tel", "run", "board_idx", "chip", "row", "col", "bl", "nw",
                      "live", "scan_utc"]
_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


# ------------------------------------------------------------------------------- records
def _clock(text):
    """A scan time stamp ('2026-03-23 15:29:17', fractions, 'T' or '+00:00' accepted) as a
    pandas Timestamp without a zone, in the scan clock; None when empty."""
    if text is None or not str(text).strip():
        return None
    t = pd.Timestamp(str(text).strip())
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


def load_scans(paths):
    """{DAQ chip name: [scan, ...]} from the scan-history sqlites `paths`, each scan a dict: bl
    and nw (16 x 16, NaN where the pixel has no row), t_start (its first pixel's time stamp, scan
    clock) and gain_mode (None where the history has no such column), sorted by t_start.

    A scan is the set of rows sharing one save marker: save_notes in the older schema,
    saving_timestamp_utc in the newer one (the per-pixel time stamps differ within a scan)."""
    groups = defaultdict(lambda: defaultdict(list))
    for i, path in enumerate(paths):
        con = sqlite3.connect("file:%s?mode=ro&immutable=1" % path, uri=True)
        try:
            cols = [r[1] for r in con.execute("PRAGMA table_info(baselines)")]
            new = "saving_timestamp_utc" in cols
            tcol = "pixel_timestamp_utc" if new else "timestamp"
            key = "saving_timestamp_utc" if new else "save_notes"
            gain = "gain_mode" if "gain_mode" in cols else "NULL"
            q = ("SELECT row, col, baseline, noise_width, %s, chip_name, %s, %s FROM baselines"
                 % (tcol, key, gain))
            for r in con.execute(q):
                groups[r[5]][(i, r[6])].append(r)
        finally:
            con.close()
    scans = {}
    for chip, by_key in groups.items():
        lst = []
        for rows in by_key.values():
            bl = np.full((NPIX, NPIX), np.nan)
            nw = np.full((NPIX, NPIX), np.nan)
            for r in rows:
                bl[r[0], r[1]] = r[2]
                nw[r[0], r[1]] = r[3]
            stamps = sorted(x[4] for x in rows if x[4])
            gm = rows[0][7]
            lst.append(dict(bl=bl, nw=nw, t_start=_clock(stamps[0]) if stamps else None,
                            gain_mode=None if gm in (None, "") else gm))
        lst.sort(key=lambda s: (s["t_start"] is None, s["t_start"] or pd.Timestamp(0)))
        scans[chip] = lst
    return scans


def load_runs(folder):
    """[run, ...] of the run_<NNN>_*/run_metadata.yaml files in `folder`, each a dict: run,
    run_name, start (UTC, no zone) and chips {DAQ chip name: dict(offset, dac (16 x 16), elink)}.
    A run start without a time zone is an error."""
    runs = []
    for path in sorted(glob.glob(os.path.join(folder, "run_*", "run_metadata.yaml"))):
        with open(path) as fh:
            meta = yaml.load(fh, Loader=_LOADER)
        info = meta["run_info"]
        start = pd.Timestamp(str(info["timestamp"]))
        if start.tzinfo is None:
            raise ValueError("%s: run_info.timestamp %r has no time zone"
                             % (path, info["timestamp"]))
        chips = {}
        for name, c in (meta.get("etroc_chips") or {}).items():
            dac = np.full((NPIX, NPIX), np.nan)
            for k, v in ((c.get("pixels") or {}).get("dac_values") or {}).items():
                r, cc = re.match(r"\(?(\d+),\s*(\d+)\)?", str(k)).groups()
                dac[int(r), int(cc)] = v
            chips[name] = dict(offset=c.get("threshold_offset"), dac=dac, elink=c.get("elink_id"))
        runs.append(dict(run=int(re.search(r"run_(\d+)", info["run_name"]).group(1)),
                         run_name=info["run_name"], chips=chips,
                         start=start.tz_convert("UTC").tz_localize(None)))
    return runs


def board_configs(path):
    """(per run {(telescope, run, board_idx): board}, per telescope {(telescope, chip): board})
    from a run-config yaml: the <tel>_run<N> entries and the telescope blocks (the defaults of a
    run without an entry), keyed by the chip of TELESCOPE_CHIPS."""
    with open(path) as fh:
        y = yaml.load(fh, Loader=_LOADER)
    per_run = {}
    for key, boards in y.items():
        m = re.match(r"^([a-z]\w*)_run(\d+)$", str(key))
        if m and isinstance(boards, dict):
            for idx, b in boards.items():
                if isinstance(b, dict):
                    per_run[(m.group(1), int(m.group(2)), int(idx))] = b
    defaults = {}
    for tel in campaign.TELESCOPE_CHIPS:
        for b in (y.get(tel.upper()) or {}).values():
            if isinstance(b, dict) and b.get("short"):
                defaults[(tel, style.chip_short(b["short"]))] = b
    return per_run, defaults


# ------------------------------------------------------------------------------- the join
def live_pixels(bl, railed):
    """Pixels with a real baseline: 50 < baseline < 1020 DAC and not railed in the run."""
    return np.isfinite(bl) & (bl > BL_DEAD_MAX) & (bl < RAIL) & ~railed


def match_scan(start, dac, offset, scans, utc_offset_h):
    """The scan a run used, for one chip: (scan, n_mismatch, gap in minutes, match), or
    (None, nan, nan, "none") when no scan started before the run. See the module docstring."""
    if not scans or offset is None:
        return None, np.nan, np.nan, "none"
    target = dac - offset
    railed = dac >= RAIL
    best, best_key = None, None
    for s in scans:
        if s["t_start"] is None:
            continue
        gap = (start - (s["t_start"] - timedelta(hours=utc_offset_h))).total_seconds() / 60.0
        if gap < 0:
            continue
        mask = np.isfinite(s["bl"]) & np.isfinite(target) & ~railed & (s["bl"] < RAIL)
        if not mask.any():
            continue
        n_mis = int(np.sum(np.abs(s["bl"] - target)[mask] > 0))
        if best_key is None or (n_mis, gap) < best_key:
            best_key, best = (n_mis, gap), s
    if best is None:
        return None, np.nan, np.nan, "none"
    n_mis, gap = best_key
    flag = "exact" if n_mis == 0 else ("near_exact" if n_mis <= NEAR_EXACT_MAX else "nearest")
    return best, n_mis, gap, flag


def board_stats(bl, nw, live):
    """Both statistics of one board's scan (the module docstring says which pixels each uses),
    and the median and 16th / 84th percentiles over the pixels of the second."""
    def ms(a):
        return (float(np.mean(a)), float(np.std(a))) if a.size else (np.nan, np.nan)

    def quantiles(a):
        return tuple(float(v) for v in np.percentile(a, [50, 16, 84])) if a.size else (np.nan,) * 3
    b_all = bl[np.isfinite(bl)]
    n_all = np.nan_to_num(nw, nan=0.0).ravel()
    b_live = bl[live]
    n_live = nw[np.isfinite(nw) & (nw > 0)]
    out = {}
    out["bl_mean_allpix"], out["bl_std_allpix"] = ms(b_all)
    out["nw_mean_allpix"], out["nw_std_allpix"] = ms(n_all)
    out["bl_mean"], out["bl_std"] = ms(b_live)
    out["nw_mean"], out["nw_std"] = ms(n_live)
    out["n_pixels_bl"], out["n_pixels_nw"] = int(b_live.size), int(n_live.size)
    out["bl_median"], out["bl_p16"], out["bl_p84"] = quantiles(b_live)
    out["nw_median"], out["nw_p16"], out["nw_p84"] = quantiles(n_live)
    return out


def pixel_rows(base, bl, nw, live, scan_utc):
    """BLNW_PIXEL_COLUMNS rows of one board's scan: `base` (campaign, tel, run, board_idx, chip)
    plus each pixel's bl and nw as stored (NaN where the scan has no row), live and scan_utc."""
    return [dict(base, row=r, col=c, bl=float(bl[r, c]), nw=float(nw[r, c]),
                 live=int(bool(live[r, c])), scan_utc=scan_utc)
            for r in range(NPIX) for c in range(NPIX)]


def build_blnw_table(out_csv=None, metadata_dirs=None, sqlites=None, run_yamls=None, runs=None,
                     out_pixels_csv=None):
    """The BL/NW table (BLNW_COLUMNS), one row per board of every run in the run metadata.

    metadata_dirs  {(campaign, telescope): folder of run folders}, default RUN_METADATA_DIRS
    sqlites        {(campaign, telescope): [scan-history sqlite, ...]}, default BLNW_SQLITES
    run_yamls      {campaign: run-config yaml}, default RUN_LIST_YAML
    runs           {(campaign, telescope): runs} to keep only those runs; default every run
    Written to `out_csv` when given; the per-pixel table (BLNW_PIXEL_COLUMNS, the pixels of every
    board with a scan) to `out_pixels_csv` when given (gzip when it ends in .gz). BLNW_CSV and
    BLNW_PIXELS_CSV were made with `python -m etroc_plots.frontend.rebuild --out <folder> --part
    blnw`."""
    metadata_dirs = campaign.RUN_METADATA_DIRS if metadata_dirs is None else metadata_dirs
    sqlites = campaign.BLNW_SQLITES if sqlites is None else sqlites
    run_yamls = campaign.RUN_LIST_YAML if run_yamls is None else run_yamls
    configs = {camp: board_configs(path) for camp, path in run_yamls.items()}
    rows, pix_rows = [], []
    for (camp, tel), folder in metadata_dirs.items():
        keep = None if runs is None else set(runs.get((camp, tel), ()))
        if keep is not None and not keep:
            continue
        scans = load_scans(sqlites[(camp, tel)])
        per_run, defaults = configs[camp]
        utc_off = campaign.BLNW_SQLITE_UTC_OFFSET_H[camp]
        for r in load_runs(folder):
            if keep is not None and r["run"] not in keep:
                continue
            for daq_name, c in r["chips"].items():
                idx = c["elink"] // 4
                chip = campaign.TELESCOPE_CHIPS[tel][idx]
                cfg = per_run.get((tel, r["run"], idx)) or defaults.get((tel, chip)) or {}
                if cfg.get("short") and style.chip_short(cfg["short"]) != chip:
                    raise ValueError("%s %s run %d board %d: the run-config yaml has %s, "
                                     "TELESCOPE_CHIPS has %s" % (camp, tel, r["run"], idx,
                                                                 cfg["short"], chip))
                s, n_mis, gap, flag = match_scan(r["start"], c["dac"], c["offset"],
                                                 scans.get(daq_name, []), utc_off)
                rf = re.search(r"RFSel(\d)", r["run_name"])
                if s is not None and s["gain_mode"] is not None:
                    rfsel, src = int(s["gain_mode"]), "scan"
                elif rf:
                    rfsel, src = int(rf.group(1)), "run_name"
                elif cfg.get("RFSel") not in (None, ""):
                    rfsel, src = int(cfg["RFSel"]), "run_yaml"
                else:
                    rfsel, src = np.nan, "none"
                row = dict(campaign=camp, tel=tel, run=r["run"], board_idx=idx, chip=chip,
                           start_utc=r["start"].strftime("%Y-%m-%d %H:%M:%S"),
                           rfsel=rfsel, rfsel_src=src, offset=c["offset"],
                           hv=cfg.get("HV", np.nan),
                           scan_utc=((s["t_start"] - timedelta(hours=utc_off))
                                     .strftime("%Y-%m-%d %H:%M:%S") if s is not None else ""),
                           scan_gap_min=round(gap, 2) if np.isfinite(gap) else np.nan,
                           match=flag, n_mismatch=n_mis)
                if s is not None:
                    live = live_pixels(s["bl"], c["dac"] >= RAIL)
                    row.update(board_stats(s["bl"], s["nw"], live))
                    pix_rows += pixel_rows({k: row[k] for k in BLNW_PIXEL_COLUMNS[:5]},
                                           s["bl"], s["nw"], live, row["scan_utc"])
                rows.append(row)
    df = pd.DataFrame(rows, columns=BLNW_COLUMNS)
    df = df.sort_values(["campaign", "tel", "run", "board_idx"]).reset_index(drop=True)
    if out_csv:
        df.to_csv(out_csv, index=False, float_format="%.6g")
    if out_pixels_csv:
        pix = pd.DataFrame(pix_rows, columns=BLNW_PIXEL_COLUMNS)
        pix = pix.sort_values(["campaign", "tel", "run", "board_idx", "row", "col"])
        gz = str(out_pixels_csv).endswith(".gz")       # mtime 0: the same rows, the same bytes
        pix.to_csv(out_pixels_csv, index=False, float_format="%.6g",
                   compression=dict(method="gzip", mtime=0) if gz else None)
    return df


def read_blnw(path=None):
    """The BL/NW table (default BLNW_CSV)."""
    df = pd.read_csv(path or campaign.BLNW_CSV)
    missing = [c for c in BLNW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("BL/NW table lacks the columns %s" % missing)
    return df


def read_blnw_pixels(path=None):
    """The per-pixel BL/NW table (default BLNW_PIXELS_CSV), `live` as bool."""
    df = pd.read_csv(path or campaign.BLNW_PIXELS_CSV)
    missing = [c for c in BLNW_PIXEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("per-pixel BL/NW table lacks the columns %s" % missing)
    df["live"] = df["live"].astype(bool)
    return df


# ------------------------------------------------------------------------------- times
def time_text(hours):
    """'+17 h' below 48 h, then days: '+2.5 d' below 10 days, '+116 d' above."""
    if hours < 48.0:
        return "+%.0f h" % hours
    days = hours / 24.0
    return ("+%.1f d" % days) if days < 10.0 else ("+%.0f d" % days)


def time_after_stop(run_times, camp, tel, runs, stop_utc):
    """Time from an irradiation stop (`stop_utc`, UTC) to the start of each run (run_times: the
    run-times table read by overview.timeline.read_run_times), as text: one value, or the range
    over `runs` ('+11 to +23 h'). Returns (text, {run: hours})."""
    stop = pd.Timestamp(stop_utc)
    stop = stop.tz_localize("UTC") if stop.tzinfo is None else stop.tz_convert("UTC")
    hours = {}
    for run in runs:
        r = run_times[(run_times["campaign"] == camp) & (run_times["tel"] == tel)
                      & (run_times["run_num"] == int(run))]
        if len(r) != 1:
            raise ValueError("%d run-times rows for %s %s run %s" % (len(r), camp, tel, run))
        hours[int(run)] = (r["start_utc"].iloc[0] - stop).total_seconds() / 3600.0
    lo, hi = time_text(min(hours.values())), time_text(max(hours.values()))
    if lo == hi:
        return lo, hours
    if lo[-2:] == hi[-2:]:                      # one unit for both ends
        return "%s to %s" % (lo[:-2], hi), hours
    return "%s to %s" % (lo, hi), hours


# ------------------------------------------------------------------------------- points
def ladder_runs(display_runs, extra_runs):
    """The runs one telescope's fluence ladder draws, step by step: {step: [run, ...]}, the
    display runs of each step (`display_runs`, {step: [run, ...]}) followed by that step's extras
    (`extra_runs`, {step: [run, ...]}, the campaign's LADDER_EXTRA_RUNS), so the display runs
    keep their places. An extra at a step with no display run raises ValueError."""
    orphans = sorted(set(extra_runs) - set(display_runs))
    if orphans:
        raise ValueError("ladder extras at a step with no display run: %s"
                         % "; ".join("run %s at %g" % (", ".join(str(r) for r in extra_runs[s]), s)
                                     for s in orphans))
    return {step: list(runs) + list(extra_runs.get(step, []))
            for step, runs in display_runs.items()}


def points(table, steps, quantities, stats, span=0.5, pitch=None):
    """Every point a BL/NW figure draws: one per step, telescope, run, board, quantity and
    statistic, as dicts (the values-file payload).

    steps       [dict(campaign, runs={telescope: [run, ...]}), ...], one per x position
    quantities  [dict(key="bl" or "nw", ...), ...]
    stats       [dict(key="allpix" or "live", ...), ...]: "allpix" reads <q>_mean_allpix and
                <q>_std_allpix over 256 pixels, "live" reads <q>_mean and <q>_std over
                n_pixels_<q>
    The runs of one step are offset side by side over `span`, or `pitch` apart when given (then
    a step of few runs keeps them close); the four boards of one run share that offset. A board
    missing from the table raises ValueError."""
    out = []
    for i, st in enumerate(steps):
        for tel, runs in st["runs"].items():
            runs = sorted(runs)
            dx = ({runs[0]: 0.0} if len(runs) == 1 else
                  {r: (j - (len(runs) - 1) / 2.0) * (span / (len(runs) - 1) if pitch is None
                                                     else pitch)
                   for j, r in enumerate(runs)})
            for run in runs:
                sub = table[(table["campaign"] == st["campaign"]) & (table["tel"] == tel)
                            & (table["run"] == int(run))]
                for idx, chip in enumerate(campaign.TELESCOPE_CHIPS[tel]):
                    b = sub[sub["board_idx"] == idx]
                    if len(b) != 1 or not np.isfinite(b["bl_mean_allpix"].iloc[0]):
                        raise ValueError("BL/NW table: no scan for %s %s run %s board %d"
                                         % (st["campaign"], tel, run, idx))
                    b = b.iloc[0]
                    for q in quantities:
                        for s in stats:
                            sfx = "_allpix" if s["key"] == "allpix" else ""
                            n = 256 if s["key"] == "allpix" else int(b["n_pixels_" + q["key"]])
                            out.append(dict(step=i, tel=tel, campaign=st["campaign"],
                                            run=int(run), dx=dx[run], board_idx=idx, chip=chip,
                                            quantity=q["key"], stat=s["key"],
                                            mean=float(b[q["key"] + "_mean" + sfx]),
                                            std=float(b[q["key"] + "_std" + sfx]),
                                            n_pixels=n, hv=None if pd.isna(b["hv"]) else
                                            float(b["hv"]), match=b["match"]))
    return out


def excluded_range(pts, quantity):
    """(fewest, most) pixels left out of the "live" statistic of `quantity` over the boards in
    `pts`, 256 minus n_pixels."""
    n = [256 - p["n_pixels"] for p in pts if p["quantity"] == quantity and p["stat"] == "live"]
    return (min(n), max(n)) if n else (0, 0)


def bias_text(hv):
    """'400 V', '530/550 V' or '450-480 V' (more than two values)."""
    hv = sorted({int(round(v)) for v in hv if v is not None})
    if not hv:
        return ""
    if len(hv) > 2:
        return "%d-%d V" % (hv[0], hv[-1])
    return "%s V" % "/".join(str(v) for v in hv)


# ------------------------------------------------------------------------------- drawing
STAT_DX = 0.03          # x offset of the two statistics of one board, in step units


def _draw_points(ax, pts, quantity, stats, scale, x_of):
    order = [s["key"] for s in stats]
    for p in pts:
        if p["quantity"] != quantity:
            continue
        k = order.index(p["stat"])
        x = x_of(p) + (k - (len(order) - 1) / 2.0) * 2 * STAT_DX
        kw = style.point_style(p["chip"], 0.0, scale=scale, line=False,
                               open_marker=bool(stats[k].get("open")),
                               color=BOARD_COLOR[p["board_idx"]])
        ax.errorbar([x], [p["mean"]], yerr=[[p["std"]], [p["std"]]], zorder=3 + k, **kw)


def board_handles(tel, scale=1.0):
    """Legend handles: one per board of `tel`, its colour and marker, named by its chip."""
    out = []
    for idx, chip in enumerate(campaign.TELESCOPE_CHIPS[tel]):
        kw = style.point_style(chip, 0.0, scale=scale, line=False, color=BOARD_COLOR[idx])
        out.append(Line2D([], [], color=kw["color"], marker=kw["marker"],
                          markersize=kw["markersize"], linestyle="none", label=chip))
    return out


def stat_handles(stats, scale=1.0):
    """Legend handles: filled or open marker per statistic, labelled by the caller's words."""
    out = []
    for s in stats:
        out.append(Line2D([], [], color=style.INK_MUTED, marker="o", markersize=8.0 * scale,
                          markerfacecolor=style.SURFACE if s.get("open") else style.INK_MUTED,
                          markeredgecolor=style.INK_MUTED, markeredgewidth=1.6 * scale,
                          linestyle="none", label=s["label"]))
    return out


def _ylim(ax, pts, quantity, headroom):
    lo = [p["mean"] - p["std"] for p in pts if p["quantity"] == quantity]
    hi = [p["mean"] + p["std"] for p in pts if p["quantity"] == quantity]
    a, b = min(lo), max(hi)
    span = (b - a) or 1.0
    ax.set_ylim(a - 0.06 * span, b + headroom * span)


def _grid(nrows, ncols, aw, ah, ml, mr, mt, mb, gx, gy, scale):
    import matplotlib.pyplot as plt
    style.apply_style(scale)
    W = ml + ncols * aw + (ncols - 1) * gx + mr
    H = mb + nrows * ah + (nrows - 1) * gy + mt
    fig = plt.figure(figsize=(W, H))
    axes = [[fig.add_axes([(ml + j * (aw + gx)) / W, (mb + (nrows - 1 - i) * (ah + gy)) / H,
                           aw / W, ah / H]) for j in range(ncols)] for i in range(nrows)]
    return fig, axes


def ladder_vs_fluence(pts, steps, quantities, stats, *, tick_labels, xlabel, tel_titles, tag,
                      data, footer, boundary_after=None, scale=0.72):
    """BL and NW of every board against the fluence step: rows = `quantities` (BL, NW), columns =
    the telescopes, x = `steps` (one category each, labelled `tick_labels[tel][i]`, the bottom
    row's axis `xlabel`), both statistics side by side (filled / open marker, `stats`), bar =
    pixel spread. Colour = board, marker = chip. A dotted line after step `boundary_after` (the
    campaign change); the bias of the runs drawn at each step written above it on the top row.

    pts         points(table, steps, quantities, stats)
    quantities  [dict(key, ylabel), ...]; stats [dict(key, label, open), ...]
    tel_titles  {telescope: panel subject}; tag, data: the header's facility line
    footer      the footer text; the bottom margin grows for more than four lines
    `steps` may be any categories (the RFSel settings, say): with boundary_after None no line is
    drawn. Returns the figure."""
    tels = list(tel_titles)
    n = len(steps)
    fig, axes = _grid(len(quantities), len(tels), aw=7.4, ah=3.6, ml=1.25, mr=0.3, mt=0.62,
                      mb=1.05 + style.footer_extra_in(footer, scale, fits=4), gx=0.85, gy=0.95,
                      scale=scale)
    s = style.sizes(scale)
    for i, q in enumerate(quantities):
        row_pts = [p for p in pts if p["quantity"] == q["key"]]
        for j, tel in enumerate(tels):
            ax = axes[i][j]
            tp = [p for p in row_pts if p["tel"] == tel]
            _draw_points(ax, tp, q["key"], stats, scale, lambda p: p["step"] + p["dx"])
            _ylim(ax, row_pts, q["key"], 0.62 if i == 0 else 0.12)
            if boundary_after is not None:
                ax.axvline(boundary_after + 0.5, color=style.INK_MUTED, linewidth=0.9,
                           linestyle=":", zorder=1)
            ax.set_xlim(-0.6, n - 0.4)
            ax.set_xticks(range(n))
            ax.set_xticklabels(tick_labels[tel], fontsize=s["tick"] * 0.78)
            ax.minorticks_off()
            ax.set_ylabel(q["ylabel"] if j == 0 else "")
            if i == len(quantities) - 1:
                ax.set_xlabel(xlabel)
            style.style_axes(ax, scale, minor=False)
            ax.tick_params(axis="x", labelsize=s["tick"] * 0.78)
            if i == 0:
                ylo, yhi = ax.get_ylim()
                pad = 0.03 * (yhi - ylo)
                for k in range(n):
                    grp = [p for p in tp if p["step"] == k]
                    if grp:
                        ax.text(k, max(p["mean"] + p["std"] for p in grp) + pad,
                                bias_text([p["hv"] for p in grp]), ha="center", va="bottom",
                                fontsize=s["ann"] * 0.85, color=style.INK_MUTED, zorder=4)
                lg = ax.legend(handles=board_handles(tel, scale), loc="upper left", ncol=4,
                               fontsize=s["legend"] * 0.9, handletextpad=0.2, columnspacing=0.9,
                               borderaxespad=0.4)
                ax.add_artist(lg)
                ax.legend(handles=stat_handles(stats, scale), loc="upper right",
                          fontsize=s["legend"] * 0.9, handletextpad=0.2, borderaxespad=0.4)
    style.compound_header(axes[0], tag=tag, data=data, scale=scale,
                          subjects=[tel_titles[t] for t in tels])
    style.footer(fig, footer, scale)
    return fig


def rest_pair(pts, looks, quantities, stats, *, tick_labels, xlabel, tel_titles, tag, data,
              footer, link_color="#9aa2ad", scale=0.72):
    """BL and NW of every board at its two looks (before and after cooling down): rows =
    `quantities`, columns = the telescopes, x = the two looks (`looks`, one step each in `pts`,
    labelled `tick_labels[tel]`, the bottom row's axis `xlabel`). Colour = board, marker = chip,
    filled / open = the two statistics; a thin line joins each board's two looks per statistic
    (solid filled, dashed open); bar = pixel spread. Returns the figure."""
    tels = list(tel_titles)
    fig, axes = _grid(len(quantities), len(tels), aw=6.2, ah=3.6, ml=1.25, mr=0.3, mt=0.62,
                      mb=1.05, gx=0.85, gy=0.95, scale=scale)
    s = style.sizes(scale)
    jitter = 0.2            # between the boards of one look, in look units

    def x_of(p):
        return p["step"] + (p["board_idx"] - 1.5) * jitter

    order = [st["key"] for st in stats]
    for i, q in enumerate(quantities):
        row_pts = [p for p in pts if p["quantity"] == q["key"]]
        for j, tel in enumerate(tels):
            ax = axes[i][j]
            tp = [p for p in row_pts if p["tel"] == tel]
            for idx in range(len(campaign.TELESCOPE_CHIPS[tel])):
                for k, sk in enumerate(order):
                    pair = sorted([p for p in tp if p["board_idx"] == idx and p["stat"] == sk],
                                  key=lambda p: p["step"])
                    off = (k - (len(order) - 1) / 2.0) * 2 * STAT_DX
                    ax.plot([x_of(p) + off for p in pair], [p["mean"] for p in pair],
                            color=link_color, linewidth=1.0,
                            linestyle="--" if stats[k].get("open") else "-", zorder=2)
            _draw_points(ax, tp, q["key"], stats, scale, x_of)
            _ylim(ax, row_pts, q["key"], 0.55 if i == 0 else 0.12)
            ax.set_xlim(-0.55, len(looks) - 0.45)
            ax.set_xticks(range(len(looks)))
            ax.set_xticklabels(tick_labels[tel], fontsize=s["tick"] * 0.8)
            ax.minorticks_off()
            ax.set_ylabel(q["ylabel"] if j == 0 else "")
            if i == len(quantities) - 1:
                ax.set_xlabel(xlabel)
            style.style_axes(ax, scale, minor=False)
            ax.tick_params(axis="x", labelsize=s["tick"] * 0.8)
            if i == 0:
                lg = ax.legend(handles=board_handles(tel, scale), loc="upper left", ncol=4,
                               fontsize=s["legend"] * 0.9, handletextpad=0.2, columnspacing=0.9,
                               borderaxespad=0.4)
                ax.add_artist(lg)
                ax.legend(handles=stat_handles(stats, scale), loc="upper right",
                          fontsize=s["legend"] * 0.9, handletextpad=0.2, borderaxespad=0.4)
    style.compound_header(axes[0], tag=tag, data=data, scale=scale,
                          subjects=[tel_titles[t] for t in tels])
    style.footer(fig, footer, scale)
    return fig
