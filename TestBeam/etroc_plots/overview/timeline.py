"""Campaign timeline: every run of a campaign as a bar from its start to its configured end, one
row per telescope, over the campaign's fluence steps.

What the figure draws and where each piece comes from (names of the campaign module):

  runs           the <tel>_run<N> keys of RUN_LIST_YAML[campaign], the run-config yaml; a run's
                 bookkeeping fluence is the "irrad" of its boards, which must all agree
  start, end     RUN_TIMES_CSV: the DAQ's start stamp and configured run length
                 (max_run_time_minutes). The DAQ records no stop time, so every bar ends at start +
                 configured length. build_run_times() makes the table from the DAQ's
                 run_metadata.yaml files (RUN_METADATA_DIRS).
  fluence steps  background shading, one colour per step. A step ends at its successor's
                 RAD_STOP_UTC entry (the earliest stop the records allow), drawn as a dotted
                 line with its time.
  LV off         hatched spans on a telescope's row: the caller's windows, such as
                 lv_log.lv_off_windows() (the gaps in the LV power log), those that reach into
                 the drawn time range
  DAQ restart    a red line at the start of a run: daq_restarts(), from the HV / LV cycle table
  display runs   filled bars, "p" above or below the preferred ones; which runs these are is the
                 caller's choice (read_display_runs() reads a display-run list)

Times are UTC throughout.
"""
import os
import re

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib.lines import Line2D

from ..campaigns import active as campaign
from .. import style

RUN_TIMES_COLUMNS = ["campaign", "tel", "run_num", "run_name", "start_utc", "max_run_time_min"]
_RUN_DIR = re.compile(r"^run_(\d+)_")
_RUN_KEY = re.compile(r"^([a-z]\w*)_run(\d+)$")
BAR_H = 0.5             # bar height, rows 1 apart
LABEL_SIZE = 9.5        # run numbers
P_SIZE = 10.5           # the "p" of a preferred run
STEP_ALPHA = 0.16       # background shading of the fluence steps


def _utc(t):
    """A pandas Timestamp in UTC; a stamp without a time zone is taken as UTC."""
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


# ------------------------------------------------------------------------------- inputs
def read_run_metadata(path):
    """(run name, start as a UTC Timestamp, configured length in minutes) of one DAQ
    run_metadata.yaml: run_info.run_name, run_info.timestamp and
    acquisition_settings.max_run_time_minutes. A timestamp without a time zone is an error."""
    with open(path) as fh:
        meta = yaml.load(fh, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    info = meta["run_info"]
    start = pd.Timestamp(info["timestamp"])
    if start.tzinfo is None:
        raise ValueError("%s: run_info.timestamp %r has no time zone" % (path, info["timestamp"]))
    return info["run_name"], start.tz_convert("UTC"), int(
        meta["acquisition_settings"]["max_run_time_minutes"])


def build_run_times(out_csv=None, dirs=None):
    """The run-times table (RUN_TIMES_COLUMNS) from the DAQ's run folders: one row per
    run_<NNN>_* folder holding a run_metadata.yaml, in each folder of `dirs` (default the
    campaign's RUN_METADATA_DIRS, {(campaign, telescope): folder}); start_utc to the second.
    Written to `out_csv` when given. RUN_TIMES_CSV was made with

        python -c "from etroc_plots.overview import timeline; timeline.build_run_times('run_times.csv')"

    run from TestBeam/ with read access to the folders."""
    dirs = campaign.RUN_METADATA_DIRS if dirs is None else dirs
    rows = []
    for (camp, tel), folder in dirs.items():
        for name in sorted(os.listdir(folder)):
            m = _RUN_DIR.match(name)
            path = os.path.join(folder, name, "run_metadata.yaml")
            if not m or not os.path.isfile(path):
                continue
            run_name, start, length = read_run_metadata(path)
            rows.append(dict(campaign=camp, tel=tel, run_num=int(m.group(1)), run_name=run_name,
                             start_utc=start.strftime("%Y-%m-%d %H:%M:%S"),
                             max_run_time_min=length))
    df = pd.DataFrame(rows, columns=RUN_TIMES_COLUMNS)
    dup = df[df.duplicated(["campaign", "tel", "run_num"], keep=False)]
    if len(dup):
        raise ValueError("two run folders share a run number:\n%s" % dup.to_string())
    df = df.sort_values(["campaign", "tel", "run_num"]).reset_index(drop=True)
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df


def read_run_times(path=None):
    """RUN_TIMES_CSV with start_utc as UTC Timestamps."""
    df = pd.read_csv(path or campaign.RUN_TIMES_CSV)
    missing = [c for c in RUN_TIMES_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("run-times table lacks the columns %s" % missing)
    df["start_utc"] = pd.to_datetime(df["start_utc"], utc=True)
    return df


def run_list(camp, path=None):
    """[(telescope, run number, bookkeeping fluence)] of the campaign's run-config yaml (default
    RUN_LIST_YAML[camp]): every <tel>_run<N> key of a telescope in TELESCOPE_CHIPS. The fluence
    is the "irrad" all boards of the run carry, "pre-irrad" = 0, snapped to the fluence ladder."""
    with open(path or campaign.RUN_LIST_YAML[camp]) as fh:
        cfg = yaml.safe_load(fh)
    out = []
    for key, boards in cfg.items():
        m = _RUN_KEY.match(str(key))
        if not m or m.group(1) not in campaign.TELESCOPE_CHIPS:
            continue
        irrad = {str(b.get("irrad")) for b in boards.values()}
        if len(irrad) != 1:
            raise ValueError("%s: its boards disagree on the fluence: %s" % (key, sorted(irrad)))
        f = irrad.pop()
        out.append((m.group(1), int(m.group(2)),
                    0.0 if f == "pre-irrad" else style.fluence_key(float(f))))
    return out


def campaign_runs(camp, run_times):
    """The campaign's runs (run_list) with their times from `run_times` (read_run_times): a
    DataFrame of tel, run_num, fluence, start_utc, end_utc (start + configured length), sorted by
    start. Raises when a run has no row in `run_times`."""
    rt = run_times[run_times["campaign"] == camp].set_index(["tel", "run_num"])
    rows, missing = [], []
    for tel, num, fluence in run_list(camp):
        if (tel, num) not in rt.index:
            missing.append("%s run %d" % (tel, num))
            continue
        r = rt.loc[(tel, num)]
        rows.append(dict(tel=tel, run_num=num, fluence=fluence, start_utc=r["start_utc"],
                         end_utc=r["start_utc"] + pd.Timedelta(minutes=float(r["max_run_time_min"]))))
    if missing:
        raise ValueError("%s: no run times for %s" % (camp, ", ".join(missing)))
    return pd.DataFrame(rows).sort_values(["start_utc", "tel"]).reset_index(drop=True)


def steps_and_stops(runs):
    """(steps, stops): the fluence steps of `runs`, lowest first, and for every step after the
    first the end of the irradiation that led to it, RAD_STOP_UTC[step], as UTC Timestamps."""
    steps = sorted(set(runs["fluence"]))
    return steps, [_utc(campaign.RAD_STOP_UTC[f]) for f in steps[1:]]


def runs_off_step(runs, steps, stops):
    """The runs whose bookkeeping fluence is not the step in force at their start (the step
    after the last stop before it): a check, empty when the yaml and the stops agree."""
    out = []
    for r in runs.itertuples():
        k = sum(1 for s in stops if s <= r.start_utc)
        if steps[k] != r.fluence:
            out.append("%s run %d" % (r.tel.upper(), r.run_num))
    return out


def daq_restarts(tel, path=None):
    """Sorted run numbers of telescope `tel` flagged as the first run after a DAQ restart (the
    restart column of the HV / LV cycle table, default the campaign's HV_CYCLES_JUL_CSV)."""
    df = pd.read_csv(path or campaign.HV_CYCLES_JUL_CSV)
    sel = df[(df["telescope"] == tel) & (df["restart"].astype(str).str.lower() == "true")]
    return sorted(int(r) for r in sel["run"])


def read_display_runs(path):
    """{(telescope, run): preferred} of a display-run list (columns telescope, run, preferred;
    a run listed several times is preferred if any of its rows is)."""
    df = pd.read_csv(path, usecols=["telescope", "run", "preferred"])
    out = {}
    for tel, run, pref in zip(df["telescope"], df["run"], df["preferred"]):
        key = (str(tel).lower(), int(run))
        out[key] = out.get(key, False) or int(pref) == 1
    return out


# ------------------------------------------------------------------------------- drawing
def row_y():
    """{telescope: row height}: the campaign's telescopes top to bottom in TELESCOPE_CHIPS order."""
    tels = list(campaign.TELESCOPE_CHIPS)
    return {t: float(len(tels) - 1 - i) for i, t in enumerate(tels)}


def draw_campaign(ax, runs, lv_off=None, restarts=None, display=None, stop_label_y=None):
    """One campaign on `ax`: fluence-step shading, radiation stops (dotted, with their time),
    LV-off windows {tel: [(start, end)]}, DAQ restarts {tel: [run]}, and a bar per run of `runs`
    (campaign_runs), filled for the runs in `display` {(tel, run): preferred}, with "p" on the
    preferred ones. Returns (records, label_items, lv_drawn): one record per run for the values
    file, the run-number labels for place_run_labels, which needs the laid-out axes, and the
    LV-off windows that reach into the drawn time range {tel: [(start, end)]}."""
    lv_off, restarts, display = lv_off or {}, restarts or {}, display or {}
    ys = row_y()
    top = max(ys.values())
    steps, stops = steps_and_stops(runs)
    t_lo, t_hi = runs["start_utc"].min(), runs["end_utc"].max()
    pad = (t_hi - t_lo) * 0.02
    x_lo, x_hi = mdates.date2num(t_lo - pad), mdates.date2num(t_hi + pad)

    bounds = [x_lo] + [mdates.date2num(s) for s in stops] + [x_hi]
    for i, f in enumerate(steps):
        ax.axvspan(bounds[i], bounds[i + 1], color=style.fluence_color(f), alpha=STEP_ALPHA,
                   zorder=0, linewidth=0)

    lv_drawn = {tel: [(a, b) for a, b in windows if b > t_lo - pad and a < t_hi + pad]
                for tel, windows in lv_off.items()}
    for tel, windows in lv_drawn.items():
        spans = [(mdates.date2num(a), mdates.date2num(b) - mdates.date2num(a)) for a, b in windows]
        ax.broken_barh(spans, (ys[tel] - BAR_H / 2, BAR_H), facecolor="none",
                       edgecolor=style.INK_MUTED, hatch="////", linewidth=0.6, zorder=1)

    records, labels = [], []
    for r in runs.itertuples():
        y = ys[r.tel]
        x0 = mdates.date2num(r.start_utc)
        w = mdates.date2num(r.end_utc) - x0
        pref = display.get((r.tel, r.run_num))
        filled = pref is not None
        ax.broken_barh([(x0, w)], (y - BAR_H / 2, BAR_H),
                       facecolor=style.INK if filled else "none", edgecolor=style.INK,
                       linewidth=1.0, zorder=3)
        if pref:
            above = y == top          # the top row's "p" above its bar, the others' below
            ax.annotate("p", xy=(x0 + w / 2, y + BAR_H / 2 + 0.06 if above else y - BAR_H / 2 - 0.06),
                        ha="center", va="bottom" if above else "top", fontsize=P_SIZE,
                        color=style.INK, zorder=4, fontweight="bold")
        labels.append(dict(x0=x0, w=w, y=y, tel=r.tel, run_num=r.run_num, top=y == top))
        records.append(dict(tel=r.tel.upper(), run_num=int(r.run_num), fluence=float(r.fluence),
                            start=r.start_utc.isoformat(), end=r.end_utc.isoformat(),
                            display=filled, preferred=bool(pref)))

    by_run = {(r.tel, r.run_num): r for r in runs.itertuples()}
    for tel, nums in restarts.items():
        for num in nums:
            r = by_run.get((tel, num))
            if r is None:
                continue
            x0 = mdates.date2num(r.start_utc)
            ax.plot([x0, x0], [ys[tel] - BAR_H / 2 - 0.14, ys[tel] + BAR_H / 2 + 0.14],
                    color=style.ALERT, linewidth=1.8, zorder=5)

    label_y = top + 0.55 if stop_label_y is None else stop_label_y
    for s in stops:
        x0 = mdates.date2num(s)
        ax.axvline(x0, color=style.INK_MUTED, linestyle=":", linewidth=1.3, zorder=2)
        ax.annotate(s.strftime("%m-%d %H:%M"), xy=(x0, label_y), xytext=(3, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=8.0,
                    color=style.INK_MUTED, rotation=90, zorder=6)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-0.95, top + 0.95)
    ax.set_yticks(sorted(ys.values()))
    ax.set_yticklabels([t.upper() for t, _ in sorted(ys.items(), key=lambda kv: kv[1])])
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    style.style_axes(ax, grid=True)
    ax.grid(axis="y", visible=False)
    return records, labels, lv_drawn


def place_run_labels(fig, ax, items):
    """Write each run's number below its bar where it fits within the bar's width; returns
    (placed, skipped) as "TEL/run" strings. The top row's number goes in the gap below its bar
    (its "p" is above); a lower row's "p" is below its bar, so its number sits further down."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed, skipped = [], []
    for it in items:
        text = str(it["run_num"])
        x0_px = ax.transData.transform((it["x0"], 0))[0]
        x1_px = ax.transData.transform((it["x0"] + it["w"], 0))[0]
        probe = ax.text(it["x0"] + it["w"] / 2, 0, text, fontsize=LABEL_SIZE, ha="center",
                        va="top", alpha=0)
        width = probe.get_window_extent(renderer=renderer).width
        probe.remove()
        tag = "%s/%s" % (it["tel"].upper(), text)
        if width <= x1_px - x0_px:
            y = it["y"] - BAR_H / 2 - (0.06 if it["top"] else 0.24)
            ax.annotate(text, xy=(it["x0"] + it["w"] / 2, y), ha="center", va="top",
                        fontsize=LABEL_SIZE, color=style.INK, zorder=4)
            placed.append(tag)
        else:
            skipped.append(tag)
    return placed, skipped


def legend_handles(steps, lv_off=False, restarts=False):
    """Legend entries for one panel: its fluence steps, then the marks it draws."""
    out = [mpatches.Patch(facecolor=style.fluence_color(f), alpha=STEP_ALPHA, edgecolor="none",
                          label=style.fluence_label(f)) for f in steps]
    if lv_off:
        out.append(mpatches.Patch(facecolor="none", edgecolor=style.INK_MUTED, hatch="////",
                                  linewidth=0.6, label="LV off"))
    out.append(Line2D([], [], color=style.INK_MUTED, linestyle=":", linewidth=1.4,
                      label="radiation stop"))
    if restarts:
        out.append(Line2D([], [], color=style.ALERT, linewidth=1.8, label="DAQ restart"))
    out.append(mpatches.Patch(facecolor=style.INK, edgecolor=style.INK,
                              label="display run (hollow: other runs)"))
    return out


def timeline_figure(panels, tag, data, footer_text, figsize=(15.5, 12.0)):
    """One stacked panel per campaign. `panels` is a list of dicts, top first: title (the
    panel's header line), runs (campaign_runs), and optionally lv_off, restarts, display (see
    draw_campaign). `tag` and `data` join the facility line of the header. Returns (fig,
    payload), payload = per panel the records drawn, the steps, the stops, the marks, the
    run-number labels placed and skipped, and runs_off_step."""
    style.apply_style(1.0)
    fig, axes = plt.subplots(len(panels), 1, figsize=figsize, squeeze=False)
    axes = list(axes[:, 0])
    fig.subplots_adjust(left=0.06, right=0.985, top=0.945, bottom=0.14, hspace=0.60)
    drawn = []
    for ax, p in zip(axes, panels):
        drawn.append(draw_campaign(ax, p["runs"], p.get("lv_off"), p.get("restarts"),
                                   p.get("display")))
    payload = {}
    for ax, p, (records, labels, lv_drawn) in zip(axes, panels, drawn):
        placed, skipped = place_run_labels(fig, ax, labels)
        steps, stops = steps_and_stops(p["runs"])
        handles = legend_handles(steps, lv_off=bool(p.get("lv_off")),
                                 restarts=bool(p.get("restarts")))
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                  ncol=len(handles), fontsize=style.sizes()["legend"] * 0.78, frameon=False)
        payload[p["title"]] = dict(
            runs=records, fluence_steps=steps,
            radiation_stops_utc=[s.isoformat() for s in stops],
            lv_off_windows_utc={t.upper(): [[a.isoformat(), b.isoformat()] for a, b in w]
                                for t, w in lv_drawn.items()},
            daq_restarts={t.upper(): list(n) for t, n in (p.get("restarts") or {}).items()},
            run_number_labels_placed=placed, run_number_labels_skipped=skipped,
            runs_off_step=runs_off_step(p["runs"], steps, stops))
    s = style.compound_header([axes[0]], tag=tag, data=data, subjects=[panels[0]["title"]])
    for ax, p in zip(axes[1:], panels[1:]):
        style.right_lines(ax, [p["title"]], s["title"])
    style.footer(fig, footer_text, scale=0.72)
    return fig, payload
