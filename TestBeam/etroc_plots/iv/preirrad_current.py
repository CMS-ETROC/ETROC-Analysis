"""Pre-irradiation bias and leakage-current stability, from the HV supply's slow-control log.

The figure: HV readback (top) and sensor leakage current (bottom, log scale) per board over the
whole pre-irradiation phase, each beam run's window shaded and labelled with its bias, its in-run
median current and its drift (both quoted as the range over the boards), and the runs with
current spikes marked.

Two stages:
  build_cache(cache)  raw slow-control log + conditions table -> three small files in `cache`
                        preirrad_log_10s.csv.gz     the log as 10 s medians, UTC
                        preirrad_current_stats.csv  per run and board: median, spread, drift, spikes
                        preirrad_run_windows.csv    the run windows actually used, with a verdict
  plot(out, cache)    those three files alone -> preirrad_current_stability.{png,pdf} and its
                      _values.json (the numbers drawn, with provenance) in `out`

    python -m etroc_plots.iv.preirrad_current --out DIR [--cache DIR] [--rebuild]

The cache defaults to the campaign's PREIRRAD_CACHE_DIR, in the campaign's input folder, so
drawing the figure needs no access to the raw log. --rebuild first regenerates the cache from the
raw log, in place (or into --cache). The rebuild is the authority, and it writes the same bytes
when nothing has changed, so comparing the result with the campaign's checksum list
(INPUTS_MANIFEST, md5sum format) shows whether the published cache was stale.

Inputs of the rebuild, named in the campaign file:
  PREIRRAD_LOG             the iseg log: Date (local time), Re(Vmeas[c]) [V], Re(Imeas[c]) [A]
                           for channels c = 0-3 (reverse bias, so both are negative)
  PREIRRAD_CONDITIONS_CSV  one row per run: run, run_number, run_start_utc, fluence_p_cm2
                           ("pre-irrad" selects the runs drawn), board0_hv_V,
                           board0_threshold_offset

Method:
- The raw ~0.5 s stream carries frequent short readback transients (HV excursions of +-10 V with
  the current jumping by up to an order of magnitude for a second or two, in >10 % of the
  samples), so everything is computed on the log downsampled to 10 s medians. Medians taken on
  the raw stream are up to 20 % high.
- Run windows come from the log, not from the run list: [run start + 15 min, end of the HV
  plateau], where the plateau needs every channel's readback within 6 V of the run's nominal
  bias, and the window is capped at the DAQ run length (PREIRRAD_MAX_RUN_MIN). The cap matters:
  on March run 1 the HV stayed up ~2 h past the end of data taking while the current was still
  climbing, and the uncapped median would be 33 % high.
- Every run drifts (beam damage accumulating), so the plain MAD measures the drift, not the
  noise. The noise is the MAD of the residual after a 30 min rolling-median detrend; a spike is
  a residual beyond both 5 noise-MADs and 10 % of the run median.
- Verdict per run, from the worst board: "drifting" if |drift| >= 10 % of the median, "noisy" if
  the detrended noise MAD >= 1 % of it, plus the spike count; "stable" if none of these.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .. import etroc_style as es
from .. import style
from ..campaigns import active as campaign

LOG_TIME_FORMAT = "%m/%d/%Y %H:%M:%S.%f"
TOL_V = 6.0          # plateau tolerance on the HV readback
SETTLE_MIN = 15      # minutes skipped after the run start
MAD_K = 1.4826       # MAD -> Gaussian-sigma equivalent
DETREND_PTS = 181    # 30 min at 10 s sampling: rolling-median window used to detrend a run
EXCURSION_K = 5.0    # excursion: > 5 x the detrended noise MAD ...
EXCURSION_FRAC = 0.10  # ... AND > 10 % of the run median, so ordinary ripple is not flagged

LOG_10S = "preirrad_log_10s.csv.gz"
STATS_CSV = "preirrad_current_stats.csv"
WINDOWS_CSV = "preirrad_run_windows.csv"
FIG_STEM = "preirrad_current_stability"

CONVENTIONS = {
    "current": "HV supply channel current readback, uA, the slow-control log as 10 s medians; "
               "channel 0-3 = the telescope's chips in board order",
    "time": "UTC (the log's local time shifted by the campaign's PREIRRAD_LOG_UTC_OFFSET_H)",
    "run_window": "[run start + %d min, the earlier of the HV plateau end (every channel within "
                  "%g V of the nominal bias) and the DAQ run end]" % (SETTLE_MIN, TOL_V),
    "hv_spread": "max - min of the channels' HV readbacks per 10 s sample, pooled over every "
                 "in-run sample",
}


def verdict(g):
    """One-line verdict for a run, from its worst board (thresholds in the module docstring)."""
    drift = g.I_drift_pct_of_median.abs().max()
    noise = g.I_noise_MAD_detrended_pct.max()
    nexc = int(g.n_excursions.max())
    tags = []
    if drift >= 10: tags.append("drifting")
    if noise >= 1: tags.append("noisy")
    if nexc: tags.append(f"{nexc} spike{'s' if nexc > 1 else ''}")
    return " + ".join(tags) if tags else "stable"


def build_cache(cache=None, log_csv=None, conditions_csv=None, channels=None, max_run_min=None,
                utc_offset_h=None):
    """Raw slow-control log + conditions table -> the three cache files in `cache`.

    Every argument defaults to the active campaign's value (the cache to PREIRRAD_CACHE_DIR).
    `channels` maps the HV channel index to (role, chip). Returns (stats, windows, log_10s).
    """
    cache = cache or campaign.PREIRRAD_CACHE_DIR
    log_csv = log_csv or campaign.PREIRRAD_LOG
    conditions_csv = conditions_csv or campaign.PREIRRAD_CONDITIONS_CSV
    ch2board = channels or campaign.PREIRRAD_CHANNELS
    max_run_min = max_run_min or campaign.PREIRRAD_MAX_RUN_MIN
    utc_offset_h = campaign.PREIRRAD_LOG_UTC_OFFSET_H if utc_offset_h is None else utc_offset_h
    chans = sorted(ch2board)
    os.makedirs(cache, exist_ok=True)

    # ------------------------------------------------------------ load
    d = pd.read_csv(log_csv)
    d["t"] = pd.to_datetime(d["Date"], format=LOG_TIME_FORMAT) - pd.Timedelta(hours=utc_offset_h)
    for ch in chans:
        d[f"V{ch}"] = d[f"Re(Vmeas[{ch}]) [V]"].abs()          # reverse bias, reported negative
        d[f"I{ch}"] = d[f"Re(Imeas[{ch}]) [A]"].abs() * 1e6    # A -> uA
    d = d[["t"] + [f"{q}{c}" for c in chans for q in ("V", "I")]].dropna()
    n_raw = len(d)
    # 10 s median downsample: kills the readback transients
    d = d.set_index("t").resample("10s").median().dropna().reset_index()

    cond = pd.read_csv(conditions_csv)
    pre = cond[cond.fluence_p_cm2 == "pre-irrad"].copy()
    pre["start"] = pd.to_datetime(pre.run_start_utc)
    pre = pre.sort_values("start").reset_index(drop=True)

    # ------------------------------------------------------------ run windows from the log
    rows, windows = [], []
    for i, r in pre.iterrows():
        bias = float(r.board0_hv_V)
        t0 = r.start + pd.Timedelta(minutes=SETTLE_MIN)
        # search until the next run start (or the end of the log for the last run)
        t_end_search = pre.start[i + 1] if i + 1 < len(pre) else d.t.iloc[-1]
        seg = d[(d.t >= t0) & (d.t < t_end_search)]
        on = np.all([(seg[f"V{c}"] - bias).abs() <= TOL_V for c in chans], axis=0)
        if not on.any():
            raise RuntimeError(f"no plateau found for run {r.run}")
        # first contiguous plateau block (allow <60 s dropouts inside it)
        idx = np.flatnonzero(on)
        ts = seg.t.values[idx]
        brk = np.flatnonzero(np.diff(ts).astype("timedelta64[s]").astype(float) > 60)
        last = idx[brk[0]] if len(brk) else idx[-1]
        w0, plateau_end = seg.t.iloc[idx[0]], seg.t.iloc[last]
        daq_end = r.start + pd.Timedelta(minutes=max_run_min[int(r.run_number)])
        w1 = min(plateau_end, daq_end)          # DAQ run length AND HV plateau must both hold
        win = seg[(seg.t >= w0) & (seg.t <= w1)]
        windows.append(dict(run=r.run, run_number=int(r.run_number), bias_V=bias,
                            offset=int(r.board0_threshold_offset), start_utc=r.start,
                            window_start=w0, window_end=w1,
                            daq_end=daq_end, plateau_end=plateau_end,
                            end_set_by="DAQ max_run_time" if daq_end <= plateau_end else "HV plateau",
                            window_h=round((w1 - w0).total_seconds() / 3600, 2),
                            plateau_h=round((plateau_end - w0).total_seconds() / 3600, 2),
                            n_samples=len(win)))
        # per-board statistics inside the window
        n = len(win)
        for c in chans:
            role, chip = ch2board[c]
            cur = win[f"I{c}"].values
            med = float(np.median(cur))
            mad = float(np.median(np.abs(cur - med)) * MAD_K)   # includes the in-run drift
            first, last10 = cur[:max(n // 10, 1)], cur[-max(n // 10, 1):]
            drift = float(np.median(last10) - np.median(first))
            # detrend with a 30 min rolling median; the residual is the sample-to-sample
            # stability, and excursions are defined on it
            trend = pd.Series(cur).rolling(DETREND_PTS, center=True, min_periods=1).median().values
            res = cur - trend
            nmad = float(np.median(np.abs(res - np.median(res))) * MAD_K)
            thr = max(EXCURSION_K * nmad, EXCURSION_FRAC * med)
            exc = np.abs(res - np.median(res)) > thr
            rows.append(dict(
                run=r.run, run_number=int(r.run_number), HV_V=bias,
                thr_offset_DAC=int(r.board0_threshold_offset), channel=c, board=role, chip=chip,
                window_start_utc=w0.strftime("%Y-%m-%d %H:%M:%S"),
                window_end_utc=w1.strftime("%Y-%m-%d %H:%M:%S"),
                window_h=round((w1 - w0).total_seconds() / 3600, 2), n_samples=n,
                I_median_uA=round(med, 4), I_MAD_uA=round(mad, 4),
                I_MAD_over_median_pct=round(100 * mad / med, 2) if med else np.nan,
                I_noise_MAD_detrended_uA=round(nmad, 5),
                I_noise_MAD_detrended_pct=round(100 * nmad / med, 3) if med else np.nan,
                I_drift_last_minus_first_tenth_uA=round(drift, 4),
                I_drift_pct_of_median=round(100 * drift / med, 2) if med else np.nan,
                I_min_uA=round(float(cur.min()), 4), I_max_uA=round(float(cur.max()), 4),
                n_excursions=int(exc.sum()),
                excursion_frac_pct=round(100 * exc.mean(), 3),
                excursion_threshold_uA=round(float(thr), 5),
                max_excursion_pct_of_median=round(100 * float(np.abs(res).max()) / med, 1) if med else np.nan,
                V_median_V=round(float(np.median(win[f"V{c}"].values)), 2)))

    stats = pd.DataFrame(rows)
    wins = pd.DataFrame(windows)
    v = (stats.groupby("run")[["I_drift_pct_of_median", "I_noise_MAD_detrended_pct", "n_excursions"]]
         .apply(verdict).rename("verdict"))
    stats = stats.merge(v, on="run")
    wins = wins.merge(v, on="run")
    stats.to_csv(os.path.join(cache, STATS_CSV), index=False)
    wins.to_csv(os.path.join(cache, WINDOWS_CSV), index=False)
    # a fixed gzip timestamp keeps an unchanged rebuild byte-identical
    d.to_csv(os.path.join(cache, LOG_10S), index=False, compression={"method": "gzip", "mtime": 0})
    print("log span (UTC): %s -> %s | %d raw samples -> %d at 10 s"
          % (d.t.min(), d.t.max(), n_raw, len(d)))
    return stats, wins, d


def check_against_reference(stats, wins, ref_medians=None, ref_plateau_h=None):
    """Print the in-run medians and window lengths beside an independent reference table.

    The March references are the in-run medians and HV-plateau lengths of the campaign's IV
    analysis, computed separately from this code; a new campaign without one sets both to None.
    """
    ref_medians = campaign.PREIRRAD_REF_MEDIANS if ref_medians is None else ref_medians
    ref_plateau_h = campaign.PREIRRAD_REF_PLATEAU_H if ref_plateau_h is None else ref_plateau_h
    if not ref_medians:
        return
    chans = sorted(stats.channel.unique())
    print(f"{'run':>4} {'HV':>4} {'plateau h (mine/note)':>22}   median uA mine vs note (max |diff|)")
    for rn, g in stats.groupby("run_number"):
        mine = [g[g.channel == c].I_median_uA.iloc[0] for c in chans]
        ref = ref_medians[rn]
        dmax = max(abs(a - b) for a, b in zip(mine, ref))
        w = wins[wins.run_number == rn].iloc[0]
        print(f"{rn:>4} {w.bias_V:>4.0f} {w.window_h:>10.2f} / {ref_plateau_h[rn]:<9.1f}  "
              f"{[round(x,3) for x in mine]} vs {list(ref)}   dmax={dmax:.3f}")


def hv_spread(log, wins):
    """Median and maximum of the per-sample HV spread between channels, over every in-run sample."""
    vcols = [c for c in log.columns if c.startswith("V")]
    spread = (log[vcols].max(axis=1) - log[vcols].min(axis=1)).values
    inrun = np.zeros(len(log), bool)
    for _, w in wins.iterrows():
        inrun |= ((log.t >= w.window_start) & (log.t <= w.window_end)).values
    return float(np.median(spread[inrun])), float(spread[inrun].max())


def plot(out, cache=None, text=None, ramp_note=None, spike_label_y=None, stem=FIG_STEM):
    """The three cache files in `cache` -> preirrad_current_stability.{png,pdf} + _values.json
    in `out`.

    cache          the cache directory
    text           figure strings (campaign, subject, hybrid, footer)
    ramp_note      the one annotation naming the between-run IV ramps; {} draws none
    spike_label_y  per spike run, in time order: (label y, arrow-tip y) in uA
    Each defaults to the active campaign's PREIRRAD_* value.
    """
    cache = cache or campaign.PREIRRAD_CACHE_DIR
    text = text or campaign.PREIRRAD_TEXT
    ramp_note = campaign.PREIRRAD_RAMP_NOTE if ramp_note is None else ramp_note
    LABEL_Y = spike_label_y or campaign.PREIRRAD_SPIKE_LABEL_Y

    SCALE = 0.85                 # panel is ~0.85 of the repo's reference (11, 10) CMS panel
    A = 1.39                     # annotation point sizes, scaled with the figure
    S = es.apply_style(SCALE)
    FIG_NAME = "Pre-irradiation Bias and Leakage Current"

    os.makedirs(out, exist_ok=True)
    log = pd.read_csv(os.path.join(cache, LOG_10S), parse_dates=["t"])
    stats = pd.read_csv(os.path.join(cache, STATS_CSV))
    wins = pd.read_csv(os.path.join(cache, WINDOWS_CSV),
                       parse_dates=["window_start", "window_end", "start_utc"])
    chips = stats.drop_duplicates("channel").set_index("channel").chip.to_dict()
    COND = ("iseg slow-control log, 10 s medians; shaded: the %d pre-irradiation beam runs"
            % len(wins))
    spread_med, spread_max = hv_spread(log, wins)

    t0 = log.t.min() - pd.Timedelta(hours=2)
    t1 = log.t.max() + pd.Timedelta(hours=2)

    # margins in inches: 0.736 above the axes for the two header lines and no more (top band
    # <= 60 px, see checks.py), 0.908 below them for the tick labels and the axis label;
    # style.save_figure then adds the footer's own band below that, 12 in high in the end
    height = 12.0 - style.FOOTER_GAP_IN
    fig, (axv, axi) = plt.subplots(2, 1, figsize=(21.5, height), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1, 1.6], hspace=0.09,
                                                    left=0.125, right=0.9,
                                                    top=1 - 0.736 / height,
                                                    bottom=(1.408 - style.FOOTER_GAP_IN) / height))

    # ------------------------------------------------------------ run bands (both panels)
    for ax in (axv, axi):
        for _, w in wins.iterrows():
            ax.axvspan(w.window_start, w.window_end, color="#c9d8ec", alpha=0.45, lw=0, zorder=0)

    # ------------------------------------------------------------ top: HV per channel
    for ch in sorted(chips):
        axv.plot(log.t, log[f"V{ch}"], lw=1.6, color=es.BOARD_COLOR[ch], alpha=0.9,
                 label=chips[ch])
    axv.set_ylabel("HV readback [V]")
    axv.set_ylim(-15, 430)
    es.style_axes(axv, SCALE)
    axv.legend(loc="lower left", ncol=4, fontsize=S["legend"], columnspacing=1.2, handlelength=1.6,
               title=text["hybrid"], title_fontsize=S["legend_title"], framealpha=0.97)
    style.header(axv, name=FIG_NAME, tag=text["subject"], line1=text["campaign"], line2=COND,
                 scale=SCALE, pad=12)

    # the one quantitative statement about the HV traces, stated where they are drawn
    axv.text(0.014, 0.968,
             "the four HV traces overlie: separate iseg channels driven to one common set point,\n"
             f"in-run spread between them {spread_med:.3f} V (median), {spread_max:.2f} V (maximum)",
             transform=axv.transAxes, ha="left", va="top", fontsize=8.6 * A, color=es.INK_MUTED,
             linespacing=1.4, zorder=7)

    # run labels above each band, in the top panel
    for _, w in wins.iterrows():
        mid = w.window_start + (w.window_end - w.window_start) / 2
        axv.annotate(f"r{w.run_number}\n{w.bias_V:.0f} V" + ("\nOS 10" if w.offset == 10 else ""),
                     (mid, 322), ha="center", va="top", fontsize=8.6 * A, color=es.INK, zorder=5,
                     bbox=dict(boxstyle="round,pad=0.28", fc=es.SURFACE, ec="#b9c6d8", lw=0.7))

    # name the between-run IV ramps once, after the first run
    if ramp_note:
        ramp_t = wins.iloc[0].window_end + pd.Timedelta(hours=ramp_note["after_first_run_h"])
        axv.annotate(ramp_note["text"], (ramp_t, ramp_note["y"]),
                     xytext=(ramp_t + pd.Timedelta(hours=ramp_note["text_dt_h"]),
                             ramp_note["text_y"]),
                     fontsize=8.3 * A, color=es.INK_MUTED, ha="left", va="center", zorder=6,
                     arrowprops=dict(arrowstyle="->", color=es.INK_MUTED, lw=1.1))

    # ------------------------------------------------------------ bottom: leakage current
    for ch in sorted(chips):
        axi.plot(log.t, log[f"I{ch}"], lw=1.4, color=es.BOARD_COLOR[ch], alpha=0.9,
                 label=chips[ch])
    axi.set_yscale("log")
    axi.set_ylabel("Sensor leakage current [µA]")
    axi.set_xlabel("Time [UTC], 2026")
    axi.set_ylim(4e-3, 30)
    es.style_axes(axi, SCALE)
    axi.grid(True, which="minor", alpha=0.22, zorder=0)
    axi.legend(loc="upper right", ncol=4, fontsize=S["legend"], columnspacing=1.2, handlelength=1.6,
               title=text["hybrid"], title_fontsize=S["legend_title"])

    # under each band: the in-run median and the drift, quoted as the RANGE over the four boards
    for _, w in wins.iterrows():
        g = stats[stats.run == w.run]
        mid = w.window_start + (w.window_end - w.window_start) / 2
        ilo, ihi = g.I_median_uA.min(), g.I_median_uA.max()
        dlo, dhi = g.I_drift_pct_of_median.min(), g.I_drift_pct_of_median.max()
        drift = (f"{dlo:+.0f} %" if round(dlo) == round(dhi)
                 else f"{dlo:+.0f}…{dhi:+.0f} %")
        axi.annotate(f"{ilo:.3f}–{ihi:.3f} µA\n{drift}",
                     (mid, 6.0e-3), ha="center", va="bottom", fontsize=8.2 * A, color=es.INK,
                     zorder=5,
                     bbox=dict(boxstyle="round,pad=0.25", fc=es.SURFACE, ec="#b9c6d8", lw=0.7))

    # mark the excursions, on the runs that have them
    exc_runs = list(stats[stats.n_excursions > 0].run.unique())
    NOTE = "#5d5d5a"                                  # neutral annotation colour
    for k, r in enumerate(exc_runs):
        w = wins[wins.run == r].iloc[0]
        n = int(stats[stats.run == r].n_excursions.max())
        mid = w.window_start + (w.window_end - w.window_start) / 2
        ytxt, ytip = LABEL_Y[k]
        axi.annotate(f"{n} spike{'s' if n > 1 else ''} > 10 % of the run median (worst board),\n"
                     f"largest {stats[stats.run == r].max_excursion_pct_of_median.max():.0f} % of it",
                     (mid, ytip), xytext=(mid, ytxt), ha="center", va="bottom", fontsize=8.2 * A,
                     color=NOTE, zorder=6,
                     arrowprops=dict(arrowstyle="->", color=NOTE, lw=1.2),
                     bbox=dict(boxstyle="round,pad=0.25", fc=es.SURFACE, ec=NOTE, lw=0.8))

    axi.set_xlim(t0, t1)
    axi.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    axi.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    plt.setp(axi.get_xticklabels(), fontsize=S["tick"])

    # the footer's point size is 8.8 * A; style.footer takes it as a scale of the reference size
    style.footer(fig, text["footer"], scale=8.8 * A / style.sizes(1.0)["ann"], x=0.125)
    _, problems = style.save_figure(fig, out, stem, dpi=200)   # footer band, audits, PNG + PDF
    plt.close(fig)
    print("wrote %s.{png,pdf}" % stem)
    print(wins[["run", "bias_V", "window_start", "window_end", "window_h", "end_set_by",
                "verdict"]].to_string(index=False))
    return write_values(out, cache, stats, wins, spread_med, spread_max, stem=stem,
                        problems=problems)


def write_values(out, cache, stats, wins, spread_med, spread_max, *, problems, stem=FIG_STEM):
    """The numbers the figure draws, with provenance, as <FIG_STEM>_values.json."""
    # imported here, after the figure is saved: iv_plot sets the IV family's rcParams on import
    from . import iv_plot as ivp
    runs = []
    for _, w in wins.iterrows():
        g = stats[stats.run == w.run]
        runs.append(dict(
            run=w.run, run_number=int(w.run_number), bias_V=float(w.bias_V), offset=int(w.offset),
            window_start_utc=str(w.window_start), window_end_utc=str(w.window_end),
            window_h=float(w.window_h), end_set_by=w.end_set_by, verdict=w.verdict,
            I_median_uA_range=[float(g.I_median_uA.min()), float(g.I_median_uA.max())],
            I_drift_pct_range=[float(g.I_drift_pct_of_median.min()),
                               float(g.I_drift_pct_of_median.max())],
            n_spikes_worst_board=int(g.n_excursions.max()),
            max_spike_pct_of_median=float(g.max_excursion_pct_of_median.max())))
    payload = {"hv_spread_V": {"median": spread_med, "max": spread_max}, "runs": runs,
               "overlap_problems": list(problems)}
    inputs = [os.path.join(cache, f) for f in (LOG_10S, STATS_CSV, WINDOWS_CSV)]
    return ivp.write_values(out, stem, payload,
                            script="TestBeam/etroc_plots/iv/preirrad_current.py", inputs=inputs,
                            conventions=CONVENTIONS)


def main(argv=None):
    matplotlib.use("Agg")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=".", help="output directory for the figure and its sidecar")
    ap.add_argument("--cache", default=None,
                    help="cache directory (default: the campaign's PREIRRAD_CACHE_DIR)")
    ap.add_argument("--rebuild", action="store_true",
                    help="regenerate the cache from the raw log (PREIRRAD_LOG) before drawing")
    a = ap.parse_args(argv)
    if a.rebuild:
        stats, wins, _ = build_cache(a.cache)
        check_against_reference(stats, wins)
    plot(a.out, cache=a.cache)


if __name__ == "__main__":
    main()
