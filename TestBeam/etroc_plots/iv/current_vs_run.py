"""Per-run bias current vs elapsed time in the run: one row of small panels per telescope.

The panels run in the SAME order as the resolution-vs-run figure (res_vs_run_combo_check_jul),
so the two figures can be read column by column. That order, and each run's class, flag,
fluence, threshold offset and RFSel, are read from that figure's values file
(campaign.COMBO_CHECK_JSON, made outside this repository; the campaign's inputs file,
campaigns/<campaign>_inputs.md, says where it comes from).

Inputs, all named in the campaign file:
  JULY_REF_CSV       per-run log: start time (UTC), run length, per-board bias voltage and
                     logging status (ok / no_hv_log / hv_off_by_design), radiation-stop stamps
  JULY_TIMELINE      the HV slow-control log in 60 s bins (tb, channel, i_uA, tel)
  HV_CYCLES_JUL_CSV  HV / LV cycle, irradiation-step and DAQ-restart marks per run
  DISPLAY_RUNS_JUL_CSV  the display runs (ink triangle; preferred ones get "(p)")
Channel 0-3 = the telescope's chips in board order (the campaign's TELESCOPE_CHIPS).

    python -m etroc_plots.iv.current_vs_run --out DIR [--tel h1|f1|both] [--no-panels]

writes DIR/current_vs_run_jul_<tel>.{png,pdf}, its _values.json, and one small panel per run
under DIR/panels/. The notebook draws the same figures (iv24_current_vs_run_h1,
iv25_current_vs_run_f1) through build_one().
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .. import style
from . import iv_plot as ivp
from ..campaigns import active as campaign

OUT = "."                                  # main() sets both from --out
PANEL_ROOT = os.path.join(OUT, "panels")

CURRENTS_CSV = campaign.JULY_REF_CSV
TIMELINE_60S = campaign.JULY_TIMELINE
HV_CYCLES_CSV = campaign.HV_CYCLES_JUL_CSV
# Display-run marks and logged-radiation-stop ticks, drawn as on the resolution-vs-run figure;
# the display-run list is read here directly.
DISPLAY_RUNS_JUL_CSV = campaign.DISPLAY_RUNS_JUL_CSV

TELS = ("h1", "f1")
SPIKE_RUNS = campaign.CURRENT_SPIKE_RUNS
FLUENCE_TEXT_PLAIN = campaign.FLUENCE_TEXT_PLAIN

SPIKE_COLOR = "#c0392b"
NOLOG_COLOR = style.INK_MUTED
# fixed per-chip-slot colour: board order 0..3, same order as TELESCOPE_CHIPS and the style
# module's line-style slots; NOT etroc_style.BOARD_COLOR.
CHIP_COLOR = {0: "#1f77b4", 1: "#d62728", 2: "#2ca02c", 3: "#9467bd"}

# HV-cycle/LV-cycle/irradiation-step marks: colours and legend text are those of the
# resolution-vs-run figure (legend text: its values file's hv_cycle_legend / lv_cycle_legend /
# irradiation_step_legend; colours: its tick swatches).
HVCYCLE_COLOR = "#ff7f0e"
LVCYCLE_COLOR = "#17becf"
IRRAD_COLOR = "#7b3fb5"
HVCYCLE_LEGEND = "HV cycle before run"
LVCYCLE_LEGEND = "LV cycle before run"
IRRAD_LEGEND = "irradiation step before run"
DAQ_RESTART_WORD = "DAQ restart"          # as on the resolution-vs-run figure; text only

# A small filled ink triangle marks every run of display_runs_jul.csv; preferred runs
# additionally get "(p)" in the run-number label. Legend text as on the resolution-vs-run figure.
DISPLAY_RUN_LEGEND_TEXT = "display run; (p) = preferred when one run is shown"
# An extra ink-coloured tick (distinct from the purple irradiation-step tick)
# beside each irradiation-step mark, quoting the logged radiation-stop stamp + LV-on time.
RAD_STOP_COLOR = style.INK
RAD_STOP_LEGEND_TEXT = "logged radiation stop"
RAD_STOP_SOURCE_TEXT = campaign.RAD_STOP_SOURCE_TEXT
# No file names on figures: the same sentence in plain words for the drawn footer;
# RAD_STOP_SOURCE_TEXT keeps them because it is stored as the values file's rad_stop_source
# provenance field.
RAD_STOP_SOURCE_TEXT_FIGURE = campaign.RAD_STOP_SOURCE_TEXT_FIGURE


def fl_text(fl):
    return FLUENCE_TEXT_PLAIN.get(style.fluence_key(fl), str(fl))


def _find_combo_json():
    p = campaign.COMBO_CHECK_JSON
    if os.path.isfile(p):
        return p, os.path.dirname(p)
    raise FileNotFoundError("%s not found (campaign COMBO_CHECK_JSON): campaigns/%s_inputs.md "
                            "says where it comes from and which variable moves it"
                            % (p, campaign.__name__.split(".")[-1]))


def load_combo_meta():
    """Run order (ascending run number, per telescope) plus per-run class/flag/fluence/offset/RFSel,
    read from the 'top' rows of the campaign's COMBO_CHECK_JSON. Returns
    (runs_by_tel, meta_by_tel_run, combo_json_path)."""
    path, _ = _find_combo_json()
    with open(path) as fh:
        doc = json.load(fh)
    top = doc["values"]["top"]
    by_run = defaultdict(list)
    for e in top:
        by_run[(e["tel"], e["run"])].append(e)
    runs_by_tel = {}
    meta = {}
    for tel in TELS:
        runs = sorted({r for (t, r) in by_run if t == tel})
        runs_by_tel[tel] = runs
        for run in runs:
            boards = by_run[(tel, run)]
            classes = sorted({b["class"] for b in boards})
            flags = sorted({b["flag"] for b in boards if b.get("flag")})
            fluence = boards[0].get("fluence")
            offsets = sorted({b["offset"] for b in boards if b.get("offset") is not None})
            rfsels = sorted({b["rfsel"] for b in boards if b.get("rfsel") is not None})
            meta[(tel, run)] = dict(
                cls=classes[0] if len(classes) == 1 else "/".join(classes),
                cls_uniform=(len(classes) == 1),
                flag=flags[0] if flags else "",
                fluence=fluence, offsets=offsets, rfsels=rfsels,
            )
    return runs_by_tel, meta, path


def load_currents_meta():
    """Per (tel, run) -> start_utc (datetime), max_min (float); per (tel, run, board_idx) ->
    dict(hv_V (the hv_nominal_V column), status, board). From JULY_REF_CSV
    (july/july_inrun_currents.csv in the inputs folder)."""
    run_win = {}
    board_meta = {}
    with open(CURRENTS_CSV) as fh:
        for row in csv.DictReader(fh):
            tel = row["tel"].lower()
            run = int(row["run_num"])
            idx = int(row["channel"])
            key = (tel, run)
            if key not in run_win:
                run_win[key] = dict(
                    start_utc=datetime.strptime(row["start_utc"], "%Y-%m-%d %H:%M:%S"),
                    max_min=float(row["max_min"]),
                )
            board_meta[(tel, run, idx)] = dict(
                hv_V=float(row["hv_nominal_V"]) if row["hv_nominal_V"] else None,
                status=row["status"],
                board=row["board"],
            )
    return run_win, board_meta


def _restart_flag(value, where):
    """The restart column as a bool: exactly "True" or "False"; anything else raises ValueError
    naming `where`, so a mistyped cell cannot silently read as no restart."""
    if value not in ("True", "False"):
        raise ValueError("%s: restart = %r is not True or False; fix that cell" % (where, value))
    return value == "True"


def load_hv_cycles():
    """Per (tel, run) -> dict(hv_cycled (bool: ANY board with status "ok" has b*_cycled "True"),
    lv_event (str: none / lv_cycle / irradiation_step / n/a / 'no data'), restart (bool), plus the
    raw CSV fields (gap_h, b_status / b_gap_min_V / b_cycled per board 0-3, lv_on_utc / lv_off_utc
    / lv_off_h, prev_run) for the values file. Read from the campaign's HV_CYCLES_JUL_CSV, never
    recomputed."""
    out = {}
    with open(HV_CYCLES_CSV) as fh:
        for row in csv.DictReader(fh):
            tel = row["telescope"].lower()
            run = int(row["run"])
            b_status, b_gap_min_V, b_cycled = [], [], []
            any_ok_cycled = False
            for i in range(len(campaign.TELESCOPE_CHIPS[tel])):
                st = row["b%d_status" % i]
                gv = row["b%d_gap_min_V" % i]
                cy = row["b%d_cycled" % i]
                b_status.append(st)
                b_gap_min_V.append(float(gv) if gv else None)
                cyc_bool = (cy == "True") if cy in ("True", "False") else None
                b_cycled.append(cyc_bool)
                if st == "ok" and cyc_bool:
                    any_ok_cycled = True
            out[(tel, run)] = dict(
                prev_run=(int(row["prev_run"]) if row["prev_run"] else None),
                gap_h=row["gap_h"],
                b_status=b_status, b_gap_min_V=b_gap_min_V, b_cycled=b_cycled,
                hv_cycled=any_ok_cycled,
                restart=_restart_flag(row["restart"], "%s, %s run %d" % (HV_CYCLES_CSV, tel, run)),
                lv_event=row["lv_event"],
                lv_on_utc=row["lv_on_utc"], lv_off_utc=row["lv_off_utc"],
                lv_off_h=(float(row["lv_off_h"]) if row["lv_off_h"] else None),
            )
    return out


def hv_cycle_marks(tel, run, hv_cycles):
    """(mark_text_or_None, restart_text_or_None, tick_colors[list]) for one run, combining the HV
    cycle / LV cycle / irradiation step flags exactly as the resolution-vs-run figure renders
    them: 'HV cycle' alone, 'HV+LV cycle' (HV cycle + lv_event=='lv_cycle'), 'HV cycle ·
    irradiation step' (HV cycle + lv_event=='irradiation_step'), or 'LV cycle'/'irradiation step'
    alone if HV did not cycle; 'DAQ restart' is always its own separate line, never merged (as
    on H1 run 18: 'HV cycle' then 'DAQ restart' as two lines). Ticks stack orange-then-teal/purple,
    no tick for DAQ restart (the resolution-vs-run figure's legend has none either)."""
    row = hv_cycles.get((tel, run))
    if row is None:
        return None, None, []
    hv = row["hv_cycled"]
    lv = row["lv_event"]
    ticks = []
    if hv:
        ticks.append(HVCYCLE_COLOR)
    if lv == "lv_cycle":
        ticks.append(LVCYCLE_COLOR)
    elif lv == "irradiation_step":
        ticks.append(IRRAD_COLOR)
    if hv and lv == "lv_cycle":
        mark = "HV+LV cycle"
    elif hv and lv == "irradiation_step":
        mark = u"HV cycle · irradiation step"
    elif hv:
        mark = "HV cycle"
    elif lv == "lv_cycle":
        mark = "LV cycle"
    elif lv == "irradiation_step":
        mark = "irradiation step"
    else:
        mark = None
    restart = DAQ_RESTART_WORD if row["restart"] else None
    return mark, restart, ticks


def load_display_runs():
    """(all_runs, preferred_runs): {tel: set(run)} from display_runs_jul.csv. Columns used:
    telescope (H1/F1), run (int), preferred ('0'/'1')."""
    all_runs = {"h1": set(), "f1": set()}
    pref_runs = {"h1": set(), "f1": set()}
    with open(DISPLAY_RUNS_JUL_CSV) as fh:
        for row in csv.DictReader(fh):
            tel = row["telescope"].strip().lower()
            if tel not in all_runs:
                continue
            run = int(row["run"])
            all_runs[tel].add(run)
            if row.get("preferred", "").strip() in ("1", "True", "true"):
                pref_runs[tel].add(run)
    return all_runs, pref_runs


def load_rad_stop_utc():
    """{(tel, run_num): rad_stop_utc string} from JULY_REF_CSV (the file load_currents_meta()
    reads; the first row per (tel, run) wins, as there). The stamps differ per telescope:
    H1 run11 2026-07-20 18:37:57, H1 run14 2026-07-25 16:08:57, F1 run10 2026-07-20 18:39:00,
    F1 run13 2026-07-25 16:09:11."""
    out = {}
    with open(CURRENTS_CSV) as fh:
        for row in csv.DictReader(fh):
            key = (row["tel"].lower(), int(row["run_num"]))
            if key not in out and row.get("rad_stop_utc"):
                out[key] = row["rad_stop_utc"]
    return out


def _fmt_utc_stamp(s):
    """'2026-07-20 18:37:57' -> '07-20 18:37' (the resolution-vs-run figure's stamp format)."""
    if not s:
        return None
    try:
        dt = datetime.strptime(s[:16], "%Y-%m-%d %H:%M")
    except ValueError:
        return s
    return dt.strftime("%m-%d %H:%M")


def rad_stop_lv_on_map(hv_cycles):
    """{(tel, run): (rad_stop_fmt, lv_on_fmt)} for exactly the runs hv_cycles_jul.csv marks
    lv_event=='irradiation_step' (H1 runs 11/14, F1 runs 10/13), requiring both a
    rad_stop_utc (july_inrun_currents.csv) and an lv_on_utc (hv_cycles_jul.csv, already loaded by
    load_hv_cycles()) to be present."""
    rad_stop = load_rad_stop_utc()
    out = {}
    for (tel, run), row in hv_cycles.items():
        if row.get("lv_event") != "irradiation_step":
            continue
        rs = _fmt_utc_stamp(rad_stop.get((tel, run)))
        lv = _fmt_utc_stamp(row.get("lv_on_utc"))
        if rs and lv:
            out[(tel, run)] = (rs, lv)
    return out


def load_timeline():
    """tb (naive UTC datetime, the same clock as july_inrun_currents.csv's start_utc), channel
    (0-3), i_uA, tel (H1/F1 upper)."""
    import pandas as pd
    df = pd.read_csv(TIMELINE_60S, parse_dates=["tb"])
    return df


def build_panel_data(tel, runs, run_win, board_meta, timeline):
    """{run: dict(t0, length_h, nolog(bool), chips: {chip: dict(status, elapsed_h[], i_uA[])})}"""
    chips = campaign.TELESCOPE_CHIPS[tel]
    tel_up = tel.upper()
    sub = timeline[timeline["tel"] == tel_up]
    out = {}
    for run in runs:
        win = run_win.get((tel, run))
        if win is None:
            out[run] = dict(t0=None, length_h=0.0, nolog=True, chips={})
            continue
        t0_start = win["start_utc"]
        t1 = t0_start + __import__("datetime").timedelta(minutes=win["max_min"])
        run_sub = sub[(sub["tb"] >= t0_start) & (sub["tb"] <= t1)]
        chip_series = {}
        statuses = []
        for idx, chip in enumerate(chips):
            bm = board_meta.get((tel, run, idx), {})
            status = bm.get("status", "unknown")
            statuses.append(status)
            ch_rows = run_sub[run_sub["channel"] == idx].sort_values("tb")
            chip_series[chip] = dict(status=status, hv_V=bm.get("hv_V"),
                                      tb=list(ch_rows["tb"]), i_uA=list(ch_rows["i_uA"]))
        nolog = all(s == "no_hv_log" for s in statuses)
        t0 = None
        all_tb = [t for c in chip_series.values() for t in c["tb"]]
        if all_tb:
            t0 = min(all_tb)
        for chip in chips:
            cs = chip_series[chip]
            if t0 is not None and cs["tb"]:
                cs["elapsed_h"] = [(t - t0).total_seconds() / 3600.0 for t in cs["tb"]]
            else:
                cs["elapsed_h"] = []
        length_h = win["max_min"] / 60.0
        out[run] = dict(t0=t0, length_h=length_h, nolog=nolog, chips=chip_series)
    return out


def compute_ylim(tel, panel_data):
    all_i = []
    for run, pd_ in panel_data.items():
        for chip, cs in pd_["chips"].items():
            if cs["status"] == "ok":
                all_i.extend(cs["i_uA"])
    all_i = np.asarray(all_i, dtype=float)
    all_i = all_i[np.isfinite(all_i)]
    if all_i.size == 0:
        return 100.0, "max", 0.0, 0.0
    raw_max = float(all_i.max())
    p99 = float(np.percentile(all_i, 99))
    if raw_max > 1.5 * p99 and raw_max > 0:
        y_max = math.ceil(p99 / 50.0) * 50.0
        mode = "p99"
    else:
        y_max = math.ceil(raw_max / 50.0) * 50.0
        mode = "max"
    return y_max, mode, raw_max, p99


def compute_xlim(panel_data):
    lengths = [pd_["length_h"] for pd_ in panel_data.values() if pd_["length_h"] > 0]
    return (max(lengths) * 1.03) if lengths else 1.0, (max(lengths) if lengths else 0.0)


def label_lines(tel, run, meta_row, board_meta, chips, hv_cycles, is_preferred=False,
               rad_stop_line=None):
    """5-line stacked label (the run-label convention of the resolution-vs-run figure):
    run N / fluence / HV tuple / RFSel . OS / flag-or-spike word, plus up to three more lines
    appended after it: the HV/LV/irradiation-step mark, the radiation-stop line, then 'DAQ
    restart'. Returns (lines, last_color, is_spike, hv_txt, flag_idx, tick_colors); flag_idx is
    the index of the flag-or-spike line, the only line drawn in last_color; every other line,
    the appended ones included, is drawn in muted ink (style.INK_MUTED). The resolution-vs-run
    figure likewise colours only the tick marks, not the mark text.

    is_preferred appends '(p)' to the run-number line; rad_stop_line, when given, is inserted
    right after the HV/LV/irradiation-step mark line and before 'DAQ restart' (it belongs to the
    irradiation-step mark it sits beside)."""
    hv_vals = []
    for idx in range(len(chips)):
        bm = board_meta.get((tel, run, idx))
        if bm is None:
            hv_vals.append("-")
        elif bm.get("status") == "hv_off_by_design":
            hv_vals.append("off")
        elif bm.get("hv_V") is not None:
            hv_vals.append("%g" % bm["hv_V"])
        else:
            hv_vals.append("-")
    hv_txt = "/".join(hv_vals) + " V"
    fl_txt = fl_text(meta_row["fluence"]) if meta_row["fluence"] is not None else "-"
    off_txt = "/".join("%g" % o for o in meta_row["offsets"]) if meta_row["offsets"] else "-"
    rf_txt = "/".join("%g" % r for r in meta_row["rfsels"]) if meta_row["rfsels"] else "-"
    is_spike = run in SPIKE_RUNS[tel]
    last = meta_row["flag"] if meta_row["flag"] else ("spike" if is_spike else "")
    # red whenever the run is on the campaign's spike list (CURRENT_SPIKE_RUNS), whether the
    # word shown is the not-good flag word (which may itself read "spike": the resolution-vs-run
    # values file flags a "current"-reason not-good run with that word) or the bare spike label
    last_color = SPIKE_COLOR if is_spike else style.INK
    run_txt = "run %d%s" % (run, " (p)" if is_preferred else "")
    lines = [run_txt, fl_txt, hv_txt, u"RFSel %s · OS %s" % (rf_txt, off_txt), last]
    flag_idx = len(lines) - 1
    mark, restart, tick_colors = hv_cycle_marks(tel, run, hv_cycles)
    if mark:
        lines.append(mark)
    if rad_stop_line:
        lines.append(rad_stop_line)
    if restart:
        lines.append(restart)
    return lines, last_color, is_spike, hv_txt, flag_idx, tick_colors


def _chip_style(tel, chip):
    idx = campaign.TELESCOPE_CHIPS[tel].index(chip)
    return dict(color=CHIP_COLOR[idx], linestyle=style.chip_linestyle(chip), linewidth=1.1)


def _draw_one_panel(ax, tel, run, pd_, meta_row, board_meta, y_max, x_max, fs, hv_cycles,
                    is_display=False, is_preferred=False, rad_stop_line=None, title_pad=3.0):
    chips = campaign.TELESCOPE_CHIPS[tel]
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(0.0, y_max)
    lines, last_color, is_spike, hv_txt, flag_idx, tick_colors = label_lines(
        tel, run, meta_row, board_meta, chips, hv_cycles, is_preferred=is_preferred,
        rad_stop_line=rad_stop_line)

    if pd_["nolog"]:
        ax.text(0.5, 0.5, "no HV log", transform=ax.transAxes, ha="center", va="center",
               fontsize=fs["nolog"], color=NOLOG_COLOR, style="italic")
    else:
        for chip in chips:
            cs = pd_["chips"][chip]
            if cs["status"] != "ok" or not cs["elapsed_h"]:
                continue
            xs = np.asarray(cs["elapsed_h"])
            ys = np.asarray(cs["i_uA"], dtype=float)
            clipped = bool((ys > y_max).any())
            ys_draw = np.clip(ys, None, y_max)
            chip_kw = _chip_style(tel, chip)
            ax.plot(xs, ys_draw, **chip_kw, zorder=3)
            if clipped:
                xtop = xs[np.argmax(ys)]
                # centred on the clipped peak, but hung inward within 10 % of a side of the
                # panel, which keeps it off the frame at these panel widths (the compound
                # figure's spines rule reports a label that still touches it)
                where = xtop / x_max
                ha, dx = (("left", 2) if where < 0.1 else ("right", -2) if where > 0.9
                          else ("center", 0))
                ax.annotate("clipped", xy=(xtop, y_max), xytext=(dx, -2),
                           textcoords="offset points", ha=ha, va="top",
                           fontsize=fs["clip"], color=chip_kw["color"], weight="bold",
                           arrowprops=dict(arrowstyle="-", color=chip_kw["color"], lw=0.6))

    # title above panel: "run N" + HV tuple
    # This run-title (matplotlib centre title) shares the same base title row as
    # compound_header's CMS text (axes[0]) and right_lines' facility/subject line (last axes
    # and, via subjects[], others); all default to the axes-top title row. A big pad lifts the
    # run-title clear above that header block so their bounding boxes do not share a y-range
    # (the header stays immediately above the axes).
    # explicit y (not pad=): matplotlib shares ONE title-offset transform across the
    # loc='left'/'center'/'right' titles of an axes, so a later right_lines() call on this
    # same axes (loc='right', pad=None) would silently reset this pad back to the rcParams
    # default and re-collide the two; an explicit y (axes fraction) is independent of that
    # shared transform, so this run-title stays clear of any header content right_lines adds
    # later on axes[0]/axes[-1].
    ax_h_in = ax.get_position().height * ax.figure.get_size_inches()[1]
    y_title = 1.0 + (title_pad / 72.0) / ax_h_in
    ax.set_title(u"run %d\n%s" % (run, hv_txt), fontsize=fs["title"], color=style.INK,
               y=y_title, verticalalignment="bottom")

    # small filled ink triangle above the panel for every
    # display_runs_jul.csv run (preferred runs already carry their own "(p)" in the label stack).
    if is_display:
        ax.plot([0.5], [1.055], transform=ax.transAxes, marker="^", color=style.INK,
               markersize=4.0, clip_on=False, linestyle="none", zorder=5)

    # stacked label block below panel (the resolution-vs-run figure's 5 lines); pitch tuned
    # to the panel's own axes height in inches so it does not depend on the compound-vs-single
    # aspect ratio (fs["stack_pitch"]/fs["stack_gap"] are set by the caller from the axes bbox).
    pitch = fs.get("stack_pitch", 0.030)
    gap = fs.get("stack_gap", 0.022)
    for i, txt in enumerate(lines):
        color = last_color if i == flag_idx else style.INK_MUTED
        ax.text(0.5, -gap - pitch * i, txt, transform=ax.transAxes, ha="center", va="top",
               fontsize=fs["stack"], color=color, clip_on=False)

    # red spike tick: short, thin (~1pt), centred on the panel's own bottom axis, not a bar
    # across the whole axis. HV-cycle/LV-cycle/irradiation-step ticks reuse the same
    # short-centred-tick geometry, stacked below it in the resolution-vs-run figure's colours
    # (orange then teal/purple); no tick for 'DAQ restart' (text only; that figure's legend
    # has no restart tick either).
    tick_y = -0.012
    if is_spike:
        ax.plot([0.44, 0.56], [tick_y, tick_y], transform=ax.transAxes, color=SPIKE_COLOR,
               linewidth=1.0, clip_on=False, solid_capstyle="butt")
        tick_y -= 0.010
    for color in tick_colors:
        ax.plot([0.44, 0.56], [tick_y, tick_y], transform=ax.transAxes, color=color,
               linewidth=1.0, clip_on=False, solid_capstyle="butt")
        tick_y -= 0.010
    # extra ink-coloured tick (distinct from the purple irradiation-step tick above) for the
    # logged radiation-stop stamp, on the irradiation-step runs that have one.
    if rad_stop_line:
        ax.plot([0.44, 0.56], [tick_y, tick_y], transform=ax.transAxes, color=RAD_STOP_COLOR,
               linewidth=1.0, clip_on=False, solid_capstyle="butt")
        tick_y -= 0.010

    ax.tick_params(axis="both", labelsize=fs["tick"], length=2.5)
    ax.set_xticks([0.0, x_max])
    ax.set_xticklabels(["0", "%.0fh" % x_max], fontsize=fs["tick"])
    ax.grid(True, axis="y", alpha=0.35, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)


def draw_compound(tel, runs, panel_data, meta, board_meta, y_max, y_mode, x_max, write_panels,
                 hv_cycles, stem=None):
    n = len(runs)
    W_IN, DPI = 39.4, 200
    H_IN = 6.3
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)
    fs = dict(header=11.0, title=5.6, stack=4.7, tick=4.6, nolog=7.0, clip=4.2, footer=7.5,
             legend=7.5, stack_pitch=0.032, stack_gap=0.022)
    stem = stem or ("current_vs_run_jul_%s" % tel)

    display_runs, pref_runs = load_display_runs()
    rad_stop_map = rad_stop_lv_on_map(hv_cycles)

    left, right = 0.018, 0.997
    bottom, top = 0.335, 0.84  # top: no blank band above the header
    w = (right - left) / n
    axes = []
    for i, run in enumerate(runs):
        ax = fig.add_axes([left + i * w, bottom, w * 0.90, top - bottom])
        axes.append(ax)
        rs_line = None
        if (tel, run) in rad_stop_map:
            rs, lv = rad_stop_map[(tel, run)]
            rs_line = "rad stop %s UTC / LV on %s UTC" % (rs, lv)
        _draw_one_panel(ax, tel, run, panel_data[run], meta[(tel, run)], board_meta, y_max, x_max,
                       fs, hv_cycles, is_display=(run in display_runs.get(tel, ())),
                       is_preferred=(run in pref_runs.get(tel, ())), rad_stop_line=rs_line,
                       title_pad=36.0)
        if i == 0:
            ax.set_ylabel(u"bias current [µA]", fontsize=fs["title"])
        else:
            ax.set_yticklabels([])

    # header: CMS text on the first run panel, immediately above its axes; facility line
    # (telescope + campaign) on the last panel's top right; the y/x-range convention rides as
    # that panel's subject.
    style.compound_header(
        axes, tag=campaign.TELESCOPE_TITLE[tel], data="July 2026",
        subjects=[None] * (len(axes) - 1) + [
            u"y: 0–%d µA (%s), x: 0–%.1f h" % (int(y_max), y_mode, x_max)],
        scale=fs["header"] / 18.0)

    # chip legend (colour = fixed chip-slot colour, marker/linestyle = chip); chip ids only, plus
    # the HV/LV/irradiation-step tick legend (labels and colours of the resolution-vs-run figure)
    chips = campaign.TELESCOPE_CHIPS[tel]
    handles = [Line2D([], [], color=CHIP_COLOR[i], linestyle=style.chip_linestyle(c),
                     marker=style.chip_marker(c), markersize=3.5, linewidth=1.4, label=c)
             for i, c in enumerate(chips)]
    handles.append(Line2D([], [], color=SPIKE_COLOR, linewidth=2.2, label="in-run current spike"))
    handles.append(Line2D([], [], color=HVCYCLE_COLOR, linewidth=2.2, label=HVCYCLE_LEGEND))
    handles.append(Line2D([], [], color=LVCYCLE_COLOR, linewidth=2.2, label=LVCYCLE_LEGEND))
    handles.append(Line2D([], [], color=IRRAD_COLOR, linewidth=2.2, label=IRRAD_LEGEND))
    handles.append(Line2D([], [], color="none", marker="^", markerfacecolor=style.INK,
                         markeredgecolor=style.INK, markersize=5.5, linestyle="none",
                         label=DISPLAY_RUN_LEGEND_TEXT))
    handles.append(Line2D([], [], color=RAD_STOP_COLOR, linewidth=2.2, label=RAD_STOP_LEGEND_TEXT))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
              fontsize=fs["legend"], bbox_to_anchor=(0.5, 0.205))

    n_clip = 0
    for run in runs:
        for chip, cs in panel_data[run]["chips"].items():
            if cs["status"] == "ok" and any(v > y_max for v in cs["i_uA"]):
                n_clip += 1

    import textwrap
    wrapped = []
    for line in _footer_text(tel, y_max, y_mode, x_max, n_clip).split("\n"):
        wrapped.extend(textwrap.wrap(line, width=260) or [""])
    # drawn through style.footer() (not a raw fig.text()) so it carries the _etroc_footer
    # marker: save_figure()'s lower_footer() needs that marker to find and shift this text into
    # its own band. scale is picked so sizes(scale)["ann"] gives this figure's 7.5 pt footer.
    style.footer(fig, "\n".join(wrapped), fs["footer"] / 14.0, x=0.012)

    os.makedirs(OUT, exist_ok=True)
    paths, problems = style.save_figure(fig, OUT, stem, dpi=DPI)
    png, pdf = paths
    plt.close(fig)

    n_panel_problems = 0
    if write_panels:
        n_panel_problems = draw_singles(tel, runs, panel_data, meta, board_meta, y_max, x_max, fs,
                                       stem, hv_cycles)
    return png, pdf, problems, n_panel_problems


def draw_singles(tel, runs, panel_data, meta, board_meta, y_max, x_max, fs, stem, hv_cycles):
    d = os.path.join(PANEL_ROOT, stem)
    os.makedirs(d, exist_ok=True)
    total_problems = 0
    fs1 = dict(fs)
    fs1.update(title=9.0, stack=8.0, tick=8.0, nolog=13.0, clip=7.5,
              stack_pitch=0.095, stack_gap=0.065)
    display_runs, pref_runs = load_display_runs()
    rad_stop_map = rad_stop_lv_on_map(hv_cycles)
    for run in runs:
        fig, ax = plt.subplots(figsize=(3.2, 3.0), dpi=200)
        fig.subplots_adjust(left=0.20, right=0.97, top=0.80, bottom=0.34)
        rs_line = None
        if (tel, run) in rad_stop_map:
            rs, lv = rad_stop_map[(tel, run)]
            rs_line = "rad stop %s UTC / LV on %s UTC" % (rs, lv)
        _draw_one_panel(ax, tel, run, panel_data[run], meta[(tel, run)], board_meta, y_max, x_max,
                       fs1, hv_cycles, is_display=(run in display_runs.get(tel, ())),
                       is_preferred=(run in pref_runs.get(tel, ())), rad_stop_line=rs_line)
        ax.set_ylabel(u"bias current [µA]", fontsize=fs1["title"])
        # a per-run tile carries no CMS header, so only the overlap audit applies
        problems = style.check_no_clipping(fig, "%s run%02d" % (stem, run))
        total_problems += len(problems)
        base = os.path.join(d, "run%02d" % run)
        fig.savefig(base + ".png", dpi=200, facecolor=style.SURFACE)
        fig.savefig(base + ".pdf", facecolor=style.SURFACE)
        plt.close(fig)
    return total_problems


_NUM_WORD = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
            6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}

# The two mechanism sentences are the resolution-vs-run values file's hv_cycle_rule /
# lv_cycle_rule fields, word for word; the DAQ-restart source sentence (daq_restart_source)
# completes the provenance of the footer.
_HV_CYCLE_SENTENCE = (
    "HV cycle before a run: any powered board's timeline bias v drops below 20% of that sample's "
    "own vset for >=1 60 s sample in the gap between the previous run's end and this run's start "
    "(off boards: no rule; no timeline samples in the gap: 'no data'; first run: 'n/a')."
)
_LV_CYCLE_SENTENCE = (
    "LV cycle / irradiation step: from the LV log (power-board 9 V input, ~15 s cadence, one file "
    "per LV-on period); a file boundary between the previous run's end and this run's start at the "
    "same fluence = LV cycle, at a higher fluence = irradiation step."
)
_DAQ_RESTART_SENTENCE = campaign.DAQ_RESTART_SENTENCE
# No file names on figures: a plain-worded copy for the drawn footer; _DAQ_RESTART_SENTENCE
# keeps them because it is stored as the values file's daq_restart_source provenance field.
_DAQ_RESTART_SENTENCE_FIGURE = campaign.DAQ_RESTART_SENTENCE_FIGURE
# _DISPLAY_RUN_SENTENCE is only ever drawn (not echoed into the values file), so it carries no
# file name.
_DISPLAY_RUN_SENTENCE = (
    "%s. Source: the July display-run list." % DISPLAY_RUN_LEGEND_TEXT
)


def _footer_text(tel, y_max, y_mode, x_max, n_clip):
    """The footer: a plain data statement; file names and provenance details stay in the
    values file."""
    chips_txt = "/".join(campaign.TELESCOPE_CHIPS[tel])
    if y_mode == "p99":
        n_word = _NUM_WORD.get(n_clip, str(n_clip))
        y_desc = ("99th percentile of all samples, %s excursion%s clipped and marked"
                 % (n_word, "" if n_clip == 1 else "s"))
    else:
        y_desc = "max"
    line1 = (
        "bias current: the July HV-monitor log in 60 s bins (raw, first 15 min included); HV "
        "tuple: nominal bias per board from the per-run current table; run set, run order, "
        "fluence, RFSel and offset from the resolution-vs-run values file; channel 0-3 = %s; "
        "spike runs: the campaign's in-run current spike list; y limit %d uA (%s); x limit "
        "%.2f h = longest run"
        % (chips_txt, int(y_max), y_desc, x_max)
    )
    return "\n".join([line1, _HV_CYCLE_SENTENCE, _LV_CYCLE_SENTENCE, _DAQ_RESTART_SENTENCE_FIGURE,
                     _DISPLAY_RUN_SENTENCE, RAD_STOP_SOURCE_TEXT_FIGURE])


def build_values_json(tel, runs, panel_data, meta, board_meta, y_max, y_mode, x_max, raw_max, p99,
                     combo_json_path, no_log_runs, clipped_runs, hv_cycles, stem=None, *,
                     problems):
    chips = campaign.TELESCOPE_CHIPS[tel]
    display_runs, pref_runs = load_display_runs()
    rad_stop_map = rad_stop_lv_on_map(hv_cycles)
    rad_stop_raw = load_rad_stop_utc()
    per_run = {}
    for run in runs:
        pd_ = panel_data[run]
        m = meta[(tel, run)]
        chip_rows = {}
        for chip in chips:
            cs = pd_["chips"].get(chip, {})
            i_vals = [v for v in cs.get("i_uA", []) if v is not None]
            chip_rows[chip] = dict(
                status=cs.get("status"),
                n_samples=len(i_vals),
                min_uA=min(i_vals) if i_vals else None,
                median_uA=float(np.median(i_vals)) if i_vals else None,
                max_uA=max(i_vals) if i_vals else None,
            )
        mark, restart_word, tick_colors = hv_cycle_marks(tel, run, hv_cycles)
        hv_row = hv_cycles.get((tel, run), {})
        is_display = run in display_runs.get(tel, ())
        is_preferred = run in pref_runs.get(tel, ())
        per_run[str(run)] = dict(
            tel=tel, run=run, class_=m["cls"], flag=m["flag"], fluence=m["fluence"],
            offsets=m["offsets"], rfsels=m["rfsels"], is_spike=(run in SPIKE_RUNS[tel]),
            run_length_h=pd_["length_h"], no_hv_log=pd_["nolog"], chips=chip_rows,
            is_display_run=is_display, preferred=is_preferred,
            rad_stop_utc=(rad_stop_raw.get((tel, run)) if (tel, run) in rad_stop_map else None),
            hv_cycle=dict(
                prev_run=hv_row.get("prev_run"), gap_h=hv_row.get("gap_h"),
                b_status=hv_row.get("b_status"), b_gap_min_V=hv_row.get("b_gap_min_V"),
                b_cycled=hv_row.get("b_cycled"), hv_cycled=hv_row.get("hv_cycled"),
                lv_event=hv_row.get("lv_event"), restart=hv_row.get("restart"),
                lv_on_utc=hv_row.get("lv_on_utc"), lv_off_utc=hv_row.get("lv_off_utc"),
                lv_off_h=hv_row.get("lv_off_h"), mark_text=mark, restart_text=restart_word,
                tick_colors=tick_colors,
            ),
        )
    payload = dict(
        figure="current_vs_run_jul_%s" % tel,
        telescope=tel,
        run_order=runs,
        chip_channel_map={c: i for i, c in enumerate(chips)},
        y_limit_uA=y_max, y_limit_mode=y_mode, y_raw_max_uA=raw_max, y_p99_uA=p99,
        x_limit_h=x_max,
        spike_runs=sorted(SPIKE_RUNS[tel]),
        long_excursion_clip=sorted(list(clipped_runs)),
        no_hv_log_runs=sorted(no_log_runs),
        source_currents_csv=CURRENTS_CSV,
        source_timeline_60s=TIMELINE_60S,
        source_combo_json=combo_json_path,
        source_hv_cycles_csv=HV_CYCLES_CSV,
        hv_cycle_legend=HVCYCLE_LEGEND, lv_cycle_legend=LVCYCLE_LEGEND,
        irradiation_step_legend=IRRAD_LEGEND, daq_restart_mark=DAQ_RESTART_WORD,
        hv_cycle_color=HVCYCLE_COLOR, lv_cycle_color=LVCYCLE_COLOR, irradiation_step_color=IRRAD_COLOR,
        hv_cycle_rule=_HV_CYCLE_SENTENCE, lv_cycle_rule=_LV_CYCLE_SENTENCE,
        daq_restart_source=_DAQ_RESTART_SENTENCE,
        display_run_legend=DISPLAY_RUN_LEGEND_TEXT,
        display_runs_jul_csv=DISPLAY_RUNS_JUL_CSV,
        display_runs={t: sorted(display_runs.get(t, ())) for t in TELS},
        display_runs_preferred={t: sorted(pref_runs.get(t, ())) for t in TELS},
        rad_stop_legend=RAD_STOP_LEGEND_TEXT, rad_stop_source=RAD_STOP_SOURCE_TEXT,
        rad_stop_runs={"%s/%d" % (t, r): {"rad_stop_utc": rad_stop_raw.get((t, r)),
                                         "lv_on_utc": hv_cycles.get((t, r), {}).get("lv_on_utc")}
                      for (t, r) in rad_stop_map},
        per_run=per_run,
        overlap_problems=list(problems),
    )
    return ivp.write_values(OUT, stem or ("current_vs_run_jul_%s" % tel), payload,
                            script="TestBeam/etroc_plots/iv/current_vs_run.py",
                            inputs=[CURRENTS_CSV, TIMELINE_60S, combo_json_path, HV_CYCLES_CSV],
                            conventions=ivp.CURRENTS_CONVENTIONS)


def build_one(tel, runs_by_tel, run_win, board_meta, timeline, hv_cycles, meta, combo_json_path,
             write_panels=True, stem=None):
    """One telescope's compound figure + values JSON: main()'s per-tel body, callable directly
    (bypassing argparse) with only the output stem overridden, the way iv/preirrad_current.plot()
    takes a stem= argument. main() below calls this with stem=None (the default name,
    current_vs_run_jul_<tel>)."""
    runs = runs_by_tel[tel]
    panel_data = build_panel_data(tel, runs, run_win, board_meta, timeline)
    y_max, y_mode, raw_max, p99 = compute_ylim(tel, panel_data)
    x_max, x_max_raw = compute_xlim(panel_data)

    no_log_runs = {r for r, pd_ in panel_data.items() if pd_["nolog"]}
    clipped_runs = set()
    for r, pd_ in panel_data.items():
        for chip, cs in pd_["chips"].items():
            if cs["status"] == "ok" and any(v > y_max for v in cs["i_uA"]):
                clipped_runs.add((r, chip))

    png, pdf, problems, n_panel_prob = draw_compound(tel, runs, panel_data, meta, board_meta,
                                                     y_max, y_mode, x_max, write_panels,
                                                     hv_cycles, stem=stem)
    jpath = build_values_json(tel, runs, panel_data, meta, board_meta, y_max, y_mode, x_max,
                             raw_max, p99, combo_json_path, no_log_runs, clipped_runs, hv_cycles,
                             stem=stem, problems=problems)
    return dict(tel=tel, n_runs=len(runs), y_max=y_max, y_mode=y_mode, raw_max=raw_max, p99=p99,
               x_max=x_max, x_max_raw=x_max_raw, no_log_runs=sorted(no_log_runs),
               clipped=sorted(clipped_runs), png=png, pdf=pdf, json=jpath, problems=len(problems),
               panel_problems=n_panel_prob)


def main(argv=None):
    global OUT, PANEL_ROOT
    matplotlib.use("Agg")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".", help="output directory (figures, sidecars, panels/)")
    ap.add_argument("--tel", choices=["h1", "f1", "both"], default="both")
    ap.add_argument("--no-panels", action="store_true", help="skip per-run single-panel files")
    a = ap.parse_args(argv)
    OUT, PANEL_ROOT = a.out, os.path.join(a.out, "panels")
    tels = TELS if a.tel == "both" else (a.tel,)

    style.apply_style(1.0)
    runs_by_tel, meta, combo_json_path = load_combo_meta()
    run_win, board_meta = load_currents_meta()
    timeline = load_timeline()
    hv_cycles = load_hv_cycles()

    total_problems = 0
    report = []
    for tel in tels:
        rec = build_one(tel, runs_by_tel, run_win, board_meta, timeline, hv_cycles, meta,
                        combo_json_path, write_panels=not a.no_panels)
        total_problems += rec["problems"] + rec["panel_problems"]
        report.append(rec)

    print(json.dumps(report, indent=1, default=str))
    print("TOTAL audit problems: %d" % total_problems)


if __name__ == "__main__":
    main()
