# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
# ---

# %% [markdown]
# # Settings: time resolution against RFSel and the threshold offset at 3.5e15, CERN IRRAD 2026
#
# At the last July fluence step, bookkeeping fluence 3.5e15 p/cm2, the runs stepped two
# front-end settings of the ETROC2 chips: the preamp feedback setting RFSel, which picks the
# feedback resistor R_f of the preamplifier and so its preamp gain (RFSel 1, 2, 3 = 10, 5.7,
# 4.4 kOhm), and the discriminator threshold, which the DAQ sets per pixel at the pixel's
# baseline plus a threshold offset in DAC codes. These figures draw the time resolution of every
# board in those runs against the two settings:
#
# * Figure st01: at RFSel 2, against each board's own threshold offset.
# * Figure st02: against the settings class (RFSel, offset), every class with a display run.
# * Figure st03: the step from RFSel 2 to RFSel 3 at nearly one offset.
# * Figure st04: the threshold over the noise, x_B = threshold offset / noise width, of every
#   board, per chip and class.
# * Figure st05: the time resolution against x_B.
# * Figure st06: the baseline and noise width of every board against RFSel.
# * Figure st07: what RFSel changes: where it acts in the front end, the TOT of one chip at
#   RFSel 2 and 3, and the measured numbers.
#
# The runs are the display runs of the step (`july/display_runs_jul.csv`): 9 on H1 and 11 on F1,
# 80 boards. Every board is drawn as a band, the rule at this step (`../CONVENTIONS.md`,
# "Resolution and TDC figures"): its value over every 3-board combination containing it. F1 runs
# 46 and 47 form a merge that the merged runs' table accepts and the good-run list marks good,
# and st02, st04 and st05 draw the merge in their place: 76 boards there. On st01 to st05 a board
# whose resolution comes from fewer than 128 pixels (half the chip) is a small open marker, named
# with its count in the footer.
#
# The IRRAD 2026 loops behind these figures were recorded with the readout saturated: 47 to 81 %
# of the ETROC2 packets lost their header and were dropped at decoding, so every number below is
# a measurement on the surviving hits, selected by buffer state, with the selection bias open.
# Details, and what it means per figure type: the "Data quality" section of the package README
# (`../README.md`).
#
# Every figure is saved as PNG (200 dpi) and PDF, with `<name>_values.json` beside it: every
# number drawn, the input files read and the commit of the code.
#
# To run it:
#
# 1. `source TestBeam/condor_at_lxplus/envs/load_python39.sh` (LCG_104d, the environment this
#    notebook is tested in).
# 2. The figures read small tables that are not in the repository, from
#    `/eos/user/m/musafdar/ETROC_plot_inputs/irrad_2026` (the list below).
#    `../campaigns/irrad_2026_inputs.md` lists every table with where it came from. To read
#    your own copy, copy that whole `irrad_2026` folder and point `ETROC_INPUTS` at the copy.
#    The first code cell checks that every input exists and can be read, and stops with a list
#    of all that cannot; a table whose checksum differs from the published list is named, and
#    the run goes on.
# 3. Open `settings.ipynb` in Jupyter and run all cells, or run it headless with
#    `jupyter nbconvert --to notebook --execute settings.ipynb --output-dir <somewhere>`.
#
# Figures go to `$ETROC_FIGURES/settings/`, by default `figures/settings/` next to this notebook.
#
# The tables read:
#
# * `tables/irrad/board_table_irrad.csv`: the time resolution of every board in every
#   3-board combination of every run (the bands).
# * `tables/merged/board_table_merged.csv`: the same for the merged runs, with the run-merging
#   study's verdict on each merge (`runs_merged`, `accepted`): the merges of st02, st04 and st05.
# * `tables/runs_summary.csv`: per run and board the chip, bias, bookkeeping fluence, RFSel and
#   threshold offset (each board's own offset; the bias in the footers).
# * `july/display_runs_jul.csv`: the July display runs, their settings class and the preferred
#   run of each class.
# * `july/good_runs.csv`: the good-run list; every run and every merge drawn has to be good.
# * `tables/run_times.csv`: the start of every run, for the time after the irradiation.
# * `frontend/blnw_by_run.csv`: the autocalibration scan before each run, per board: the RFSel
#   it recorded (an independent check of the RFSel in `runs_summary.csv`), the threshold offset
#   (x_B of st04 and st05), the baseline and noise width (st06).
# * `frontend/blnw_by_pixel.csv.gz`: the noise width of every pixel in each of those scans (x_B
#   of st04 and st05 and its bar).
# * `frontend/rfsel_pair_hist.csv`: the TOT of one F1 chip in the pairs of runs that differ in
#   RFSel only (st07; the table of the frontend notebook's fe15).
#
# No other table: st04-st07 read the tables above that st01-st03 read, the RFSel pair table and
# the per-pixel BL/NW table; the merged runs' table is read for st02, st04 and st05 only.
#
# The drawing lives in `../resolution/settings.py` (st01-st05), `../frontend/blnw.py` (st06) and
# `../overview/cartoons.py` (st07, with the curves of `../frontend/tot_shapes.py`); the classes
# and bands in `../resolution/tables.py`. This notebook holds the choices made per figure: the
# runs, the words and the axis ranges.
#
# `settings.ipynb` and `settings.py` are the same notebook, kept in step by jupytext. Edit
# either one, then run `python3 -m jupytext --sync settings.ipynb` and
# `python3 -m nbstripout settings.ipynb` in this folder before committing. Neither tool is in
# LCG_104d (the install line is in `../README.md`); running the notebook needs neither.
#
# After a run, `python3 -m etroc_plots.checks <figures>`, run from `TestBeam/` with `<figures>`
# the folder this notebook wrote to (`etroc_plots/notebooks/figures/settings` by default),
# checks every saved figure against the rules in `../CONVENTIONS.md`.

# %%
import os
import sys

import matplotlib
matplotlib.use("Agg")   # draw off-screen; show() displays the PNG exactly as saved
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display

# The folder that holds the package (TestBeam/) is two folders up. Jupyter starts the kernel in
# the notebook's folder; `python settings.py` has __file__.
here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
sys.path.insert(0, os.path.normpath(os.path.join(here, "..", "..")))

CAMPAIGN = "irrad_2026"                       # campaigns/<CAMPAIGN>.py
# An exported ETROC_CAMPAIGN wins (an empty one counts as unset); it is read once, on the first
# etroc_plots import.
if not os.environ.get("ETROC_CAMPAIGN"):
    os.environ["ETROC_CAMPAIGN"] = CAMPAIGN

from etroc_plots import style                                 # noqa: E402
from etroc_plots import campaigns                             # noqa: E402
from etroc_plots.inputs import check_inputs                   # noqa: E402
from etroc_plots.frontend import blnw, tot_shapes             # noqa: E402
from etroc_plots.overview import cartoons, timeline           # noqa: E402
from etroc_plots.resolution import settings, tables           # noqa: E402

if campaigns.NAME != CAMPAIGN:
    raise RuntimeError("the package is running campaign %r (ETROC_CAMPAIGN) but CAMPAIGN = %r: "
                       "set one to match the other, then restart the kernel (the campaign is "
                       "read once, on the first import)" % (campaigns.NAME, CAMPAIGN))
campaign = campaigns.active

# stops, listing every missing or unreadable input; reports any table that differs from the
# published copy.
check_inputs(only=("tables/irrad/board_table_irrad.csv", "tables/merged/board_table_merged.csv",
                   "tables/runs_summary.csv", "tables/run_times.csv", "july/display_runs_jul.csv",
                   "july/good_runs.csv", "frontend/blnw_by_run.csv",
                   "frontend/blnw_by_pixel.csv.gz", "frontend/rfsel_pair_hist.csv"))

NOTEBOOK = "TestBeam/etroc_plots/notebooks/settings.ipynb"   # "script" in every values file
OUT = os.path.join(os.path.abspath(os.environ.get("ETROC_FIGURES") or "figures"), "settings")
os.makedirs(OUT, exist_ok=True)
INPUTS = [campaign.BOARD_TABLE, campaign.RUNS_SUMMARY_CSV, campaign.DISPLAY_RUNS_JUL_CSV,
          campaign.GOOD_RUNS_CSV, campaign.RUN_TIMES_CSV, campaign.BLNW_CSV,
          campaign.RFSEL_PAIR_HIST_CSV]
INPUTS_MERGED = INPUTS + [campaign.BOARD_TABLE_MERGED]   # st02, st04 and st05: the merges too
INPUTS_XB = INPUTS_MERGED + [campaign.BLNW_PIXELS_CSV]   # st04 and st05: the pixels' noise widths

VARIANT, FLOOR = "top", "300"          # the track variant and event floor of every band
STEP = campaign.SETTINGS_STEP          # 3.5e15
TELS = list(campaign.TELESCOPE_CHIPS)
BT = tables.read_board_table(campaign.BOARD_TABLE)
BT_MERGED = tables.read_board_table(campaign.BOARD_TABLE_MERGED)
RS = tables.read_runs_summary(campaign.RUNS_SUMMARY_CSV)
BLNW = blnw.read_blnw()
RUN_TIMES = timeline.read_run_times()


def show(stem):
    """Display a saved figure, exactly as written to disk."""
    display(Image(filename=os.path.join(OUT, stem + ".png"), width=900))


def save(fig, stem, payload, conventions, inputs=INPUTS):
    """Save the figure and its values file. Beside the house audits (overlap_problems), the
    values file keeps the pixel boxes of the tick labels, legends, labels and data marks
    (layout_px) and what touches (layout_problems)."""
    _, problems = style.save_figure(fig, OUT, stem)
    boxes = settings.layout_boxes(fig)
    plt.close(fig)
    payload["overlap_problems"] = problems
    payload["layout_problems"] = settings.layout_problems(boxes)
    payload["layout_px"] = boxes
    style.write_values(OUT, stem, payload, conventions=conventions, script=NOTEBOOK,
                       inputs=inputs)
    print("%s: audit %s, layout %s" % (stem, problems or "clean",
                                       payload["layout_problems"] or "clean"))
    show(stem)


# %% [markdown]
# ## The runs and their checks
#
# The settings classes come from the display-run list: one class per (RFSel, offset) at 3.5e15,
# with its runs and its preferred run. A class spelled "8/10" holds runs whose boards sit at two
# offsets (H1 run 56: IH7 and IH11 at 8, IH12 and IH13 at 10; H1 runs 64 and 67: IH11 at 6, the
# others at 8). Each board's own offset comes from `runs_summary.csv`.
#
# Before anything is drawn:
#
# * every run drawn has to be good on the good-run list;
# * the settings guard: for every run, the RFSel of every board in `runs_summary.csv` equals the
#   class's RFSel; the boards' offsets are the class's offsets; and where the autocalibration
#   scan before the run recorded the RFSel (`rfsel_src` = scan in `blnw_by_run.csv`), it equals
#   the RFSel in `runs_summary.csv`. The guard checks every run of the display-run list at this
#   step and stops the notebook, naming the run, on any mismatch. Until 2026-10-02
#   `runs_summary.csv` listed H1 runs 64, 67 and 70 at RFSel 2 where the DAQ ran RFSel 3; on that
#   copy (`tables/superseded/runs_summary_2026-09-16.csv`) the guard stops on runs 64 and 67, the
#   two of them on the display-run list (run 70 is not a display run).
#
# F1 has no class at RFSel 3 and offsets 6/8: its one run there, 59, is not good on the
# good-run list (the cell below prints the reason).
#
# Merged runs: the merged runs' board table lists every merge with the run-merging study's
# verdict (`runs_merged`, `accepted`), and the good-run list marks merges good or not as it does
# single runs. A merge is drawn only when it passes both, accepted in the merged table and good
# on the good-run list (`tables.good_merges`, the one merge verdict of the settings and resolution
# figures): H1 6+7 and 18+19 are accepted but not good. A class whose runs on a telescope include
# every run of such a merge draws the merge in their place on st02, st04 and st05, one band per
# board from the merged table by the rule of the single runs (`tables.classes_at_step`,
# `settings.class_points`): at this step F1 runs 46 and 47 (RFSel 1, offset 20). The merge's
# RFSel and offset are the two runs' common ones in `runs_summary.csv`, as the merged notebook
# reads them. A merge with a run outside the class is not drawn; the cell below lists the merges
# that pass both verdicts and touch a class, and which of them stand in for their runs. st01 and
# st03 hold no RFSel 1 run, and st06 draws one autocalibration scan per run, so these keep the
# runs.
#
# The runs of one class were taken at different times after the end of the 3.5e15 irradiation,
# and on F1 at two biases: the cell below lists both for every run; the figures write the time
# under each run and the bias in the footer.

# %%
# accepted in the merged runs' table and good on the good-run list
MERGES = tables.good_merges(BT_MERGED, settings.CAMP)
CLASSES = tables.classes_at_step(campaign.DISPLAY_RUNS_JUL_CSV, STEP, MERGES)
for c in CLASSES:
    print("RFSel %d, offset %-5s %s" % (c.rfsel, c.os, "; ".join(
        "%s runs %s (preferred %s)" % (t.upper(), c.runs[t], c.preferred[t]) for t in TELS
        if t in c.runs)))
# the merges (accepted and good) with a run in a class, and whether the class draws them
for t in TELS:
    for m in MERGES.get(t, []):
        for c in CLASSES:
            if set(m) & set(c.runs.get(t, [])):
                print("merge %s %s (accepted, good): RFSel %d, offset %s class, %s" % (
                    t.upper(), "+".join(map(str, m)), c.rfsel, c.os,
                    "drawn in place of its runs" if m in c.merges.get(t, []) else
                    "not drawn, runs %s are not display runs of the class"
                    % sorted(set(m) - set(c.runs[t]))))
MERGES_DRAWN = {t: ["+".join(map(str, m)) for c in CLASSES for m in c.merges.get(t, [])]
                for t in TELS}
drops = settings.good_run_drops(CLASSES)
N_RUNS = sum(len(r) for c in CLASSES for r in c.runs.values())
print("display runs drawn: %d, not good: %s" % (N_RUNS, drops or "none"))
if drops:
    raise RuntimeError("display runs not good on the good-run list: %s" % drops)

from etroc_plots.iv.iv_timeseries import good_runs   # noqa: E402
_, f1_reasons = good_runs("f1")
print("F1 run 59 on the good-run list: not good, %r" % f1_reasons.get(59))

GUARD = settings.settings_guard(RS, campaign.DISPLAY_RUNS_JUL_CSV, BLNW)
print("settings guard passed: %(runs)d runs, %(boards)d boards, %(boards_scan_rfsel)d of them "
      "with an RFSel from the autocalibration scan" % GUARD)

FACTS = settings.run_facts(RS, RUN_TIMES, CLASSES, campaign.RAD_STOP_UTC[STEP])
for (tel, run), f in sorted(FACTS.items(), key=lambda kv: (TELS.index(kv[0][0]), kv[0][1])):
    print("%s run %3d  bias %-10s  after the stop %-8s (%.1f h)"
          % (tel.upper(), run, f["bias"], f["time"], f["hours"]))

PTS = settings.class_points(BT, RS, campaign.DISPLAY_RUNS_JUL_CSV, STEP, VARIANT, FLOOR)
# st02, st04 and st05: a class's merge (accepted and good) in place of its runs
PTS_M = settings.class_points(BT, RS, campaign.DISPLAY_RUNS_JUL_CSV, STEP, VARIANT, FLOOR,
                              bt_merged=BT_MERGED)
print("boards drawn: %d; with the merges in place of their runs: %d" % (len(PTS), len(PTS_M)))

# %% [markdown]
# ## The words every figure shares
#
# The selection, the band rule, the run labels (a merge's: its runs joined by "+", and the time
# of each) and the bias line, and what every values file records.

# %%
YLIM = (0.0, 150.0)            # ps, every figure: the legends and run labels sit in the margins
DATA_TEXT = u"July, bookkeeping fluence %s %s" % (campaign.FLUENCE_TEXT[STEP],
                                                  campaign.FLUENCE_UNIT)
TEL_TITLES = dict(campaign.TELESCOPE_TITLE)
LEGEND_TITLE = "preamp feedback setting"
# one footer line each: a footer line stays under about 180 characters to fit the figure width
SELECTION = (u"Selection: %s, tracks with at least %s events, each chip from its anointed "
             u"combination (boards 0-1-2 for boards 0-2, boards 1-2-3 for board 3).\n"
             u"Widths = FWHM/2.355 of the TOA-difference core, as the result tables carry them."
             % (tables.VARIANT_TEXT[VARIANT], FLOOR))
BAND_TEXT = u"Bands: %s." % tables.BAND_FOOTER
RUN_TEXT = (u"Under each run: its number (bold: the preferred run of its class) and the time "
            u"from the end of the 3.5e15 irradiation to the run start.")
RUN_LABELS = {k: u"%d\n%s" % (k[1], f["time"]) for k, f in FACTS.items()}
# a merge (st02): its runs joined by "+", and the time of each
RUN_LABELS.update({(t, tables.merge_name(t, m)): u"%s\n%s" % (
    u"+".join(map(str, m)), u", ".join(FACTS[(t, r)]["time"] for r in m))
    for c in CLASSES for t, mm in c.merges.items() for m in mm})


def run_name(p):
    """A point's run as the figures write it: its number, or a merge's runs joined by "+"."""
    return u"+".join(map(str, p["runs"])) if "runs" in p else u"%d" % p["run"]


def runs_of(pts):
    """The display runs of `pts` per telescope, the runs of a merge included."""
    return {t: sorted({r for p in pts if p["tel"] == t for r in p.get("runs", [p["run"]])})
            for t in TELS}


def entries_of(pts):
    """What `pts` draws per telescope, in run order: (name, run) of each run and merge."""
    return {t: [(n, r) for _, n, r in sorted({(settings.run_order(p), run_name(p), p["run"])
                                              for p in pts if p["tel"] == t})] for t in TELS}


def counts_text(pts):
    runs, entries = runs_of(pts), entries_of(pts)
    n = [p["band"]["n_pixels"] for p in pts]
    return (u"%d runs (%s), %d boards, %d-%d pixels per board"
            % (sum(len(r) for r in runs.values()),
               u", ".join(u"%s %s" % (t.upper(), u", ".join(n for n, _ in entries[t]))
                          for t in TELS if runs[t]), len(pts), min(n), max(n)))


def merge_text(pts):
    """The merges among `pts` in words: "F1 runs 46 and 47 as their merge"."""
    mm = sorted({(TELS.index(p["tel"]), tuple(p["runs"])) for p in pts if "runs" in p})
    return u"; ".join(u"%s runs %s and %d as their merge" % (TELS[i].upper(), u", ".join(
        map(str, m[:-1])), m[-1]) for i, m in mm)


def low_marks(pts):
    """Set p["low"] on the boards of `pts` whose resolution comes from fewer than
    tables.LOW_PIXELS pixels (settings.low_pixels): the figure draws them small and open. Returns
    the values entry: the threshold and those boards."""
    for p in pts:
        p["low"] = settings.low_pixels(p)
    low = sorted((p for p in pts if p["low"]),
                 key=lambda p: (TELS.index(p["tel"]), settings.run_order(p), p["board_idx"]))
    return dict(threshold=tables.LOW_PIXELS, marker="small open", n_boards=len(pts),
                points=[dict(tel=p["tel"], run=run_name(p), board_idx=p["board_idx"],
                             chip=p["chip"], n_pixels=p["band"]["n_pixels"]) for p in low])


SURVIVING_TEXT = u"Pixel counts are the surviving readout sample."


def low_text(low):
    """The footer lines of low_marks' boards: the rule, how many, and each board's count; then
    that the counts are those of the surviving readout sample."""
    head = (u"Small open markers: resolution from under %d pixels (half the chip), %d of %d boards"
            % (low["threshold"], len(low["points"]), low["n_boards"]))
    if not low["points"]:
        return head + u". " + SURVIVING_TEXT
    runs = {}
    for q in low["points"]:
        runs.setdefault(q["tel"], {}).setdefault(q["run"], []).append(
            u"%s %d" % (q["chip"], q["n_pixels"]))
    return head + u": %s.\n%s" % (u"; ".join(
        u"%s %s" % (t.upper(), u", ".join(u"run %s (%s)" % (r, u", ".join(c))
                                         for r, c in rr.items()))
        for t, rr in runs.items()), SURVIVING_TEXT)


LOW_CONVENTION = ("a board whose resolution comes from fewer than %d pixels (board table n_pixels "
                  "of the anointed combination; resolution.tables.LOW_PIXELS, half the chip) is "
                  "a small open marker" % tables.LOW_PIXELS)


def bias_text(pts):
    """The bias of the runs drawn, per telescope: one value when every run shares it."""
    parts = []
    for t, runs in runs_of(pts).items():
        groups = {}
        for r in runs:
            groups.setdefault(FACTS[(t, r)]["bias"], []).append(r)
        if len(groups) == 1:
            parts.append(u"%s %s on every run" % (t.upper(), list(groups)[0]))
        elif groups:
            parts.append(u"%s %s" % (t.upper(), u", ".join(
                u"runs %s at %s" % (u", ".join(map(str, r)), b) for b, r in groups.items())))
    return u"Bias: " + u"; ".join(parts) + u"."


def values(pts, lay):
    keys = ("tel", "run", "board_idx", "chip", "rfsel", "os", "offset", "hv", "preferred", "x")
    runs, entries = runs_of(pts), entries_of(pts)
    return dict(points=[dict({k: p[k] for k in keys}, **dict(
                    {"runs": p["runs"]} if "runs" in p else {}, **p["band"])) for p in pts],
                run_labels={u"%s run %s" % (t, n): RUN_LABELS[(t, r)] for t in TELS
                            for n, r in entries[t]},
                run_facts={u"%s run %d" % (t, r): FACTS[(t, r)] for t in TELS for r in runs[t]},
                layout={k: (v if k != "run_centres" else
                            {u"%s run %s" % (str(s), r): x for (s, r), x in v.items()})
                        for k, v in lay.items()},
                ylim_ps=YLIM, settings_guard=GUARD, step=STEP,
                good_run_drops={u"%s run %d" % k: v for k, v in drops.items()})


def conventions():
    c = tables.conventions(VARIANT, FLOOR)
    c.update(band=tables.BAND_FOOTER,
             width="FWHM/2.355 of the TOA-difference core per pair, as the board table carries it",
             colour="RFSel (etroc_style.RFSEL_COLOR), marker = chip",
             offset="each board's own threshold offset, runs_summary.csv",
             times="run start (tables/run_times.csv) minus RAD_STOP_UTC of the step, hours",
             class_order="by offset, then RFSel (resolution.tables.classes_at_step)")
    return c


MERGE_CONVENTION = ("a class whose runs include every run of a merge accepted in the merged runs' "
                    "board table (runs_merged, accepted) and good on the good-run list "
                    "(tables.good_merges) draws the merge in place of those runs: "
                    "one band per board from the merged table by the rule of the single runs; "
                    "RFSel and offset the runs' common ones in runs_summary.csv")


# %% [markdown]
# ## Figure st01: resolution against the threshold offset at RFSel 2
#
# Every board of every display run at RFSel 2, in the cell of its own threshold offset: 12 runs,
# 48 boards (H1 runs 18, 33, 52, 53 and 56; F1 runs 13, 14, 18, 32, 50, 51 and 52). H1 run 56
# stands in two cells, IH7 and IH11 at offset 8, IH12 and IH13 at 10. One RFSel only, so the
# offset is the one setting that changes between the cells. As on st02, a board whose resolution
# comes from fewer than 128 pixels is a small open marker: here the four boards of H1 run 56.
#
# It is not the only thing that changes. The H1 boards at offset 10 ran 2.3 to 7.5 days after
# the end of the irradiation, its offset-20 run 37 h after; on F1 the offset-20 runs came first
# (4 h to 46 h) and runs 13 and 14 ran at 370/390 V, every other F1 run at 450/480 V. Read a
# difference between cells together with the times under the runs and the bias line of the
# footer.

# %%
p01 = [dict(p) for p in PTS if p["rfsel"] == 2]
LOW_01 = low_marks(p01)
OFFSETS_01 = [o for o in campaign.SETTINGS_OFFSETS if any(p["offset"] == o for p in p01)]
split = {(p["tel"], p["run"]) for p in p01 if p["os"] != str(p["offset"])}
split_text = u"; ".join(u"%s run %d: %s" % (t.upper(), r, u", ".join(
    u"%s at %d" % (p["chip"], p["offset"]) for p in p01 if (p["tel"], p["run"]) == (t, r)))
    for t, r in sorted(split))
fig, lay = settings.res_by_offset(
    p01, OFFSETS_01, tel_titles=TEL_TITLES,
    tick_labels={t: {o: u"%d" % o for o in OFFSETS_01} for t in TELS},
    run_labels=RUN_LABELS, xlabel=u"Threshold offset above the baseline [DAC], each board at "
                                  u"its own",
    tag=style.settings_text(rfsel=2, offset=None), data=DATA_TEXT, legend_title=LEGEND_TITLE,
    ylim=YLIM,
    footer=u"\n".join([
        SELECTION,
        u"Every display run at RFSel 2: %s." % counts_text(p01),
        u"Each board in the cell of its own offset (%s)." % split_text,
        BAND_TEXT, low_text(LOW_01), RUN_TEXT, bias_text(p01)]))
payload = values(p01, lay)
payload.update(offsets=OFFSETS_01, split_runs=split_text, low_pixels=LOW_01)
save(fig, "st01_res_vs_offset_rfsel2", payload, dict(conventions(), low_pixels=LOW_CONVENTION))

# %% [markdown]
# ## Figure st02: resolution against the settings class
#
# Every display run, 20 runs, in the slot of its settings class: 76 boards, F1 runs 46 and 47 as
# their merge (four bands from the merged runs' board table, by the rule of the single runs; the
# cell "The runs and their checks"). The classes are grouped by offset (a dotted line between the
# groups), RFSel rising within a group; each telescope shows the classes it has. A class at
# "8/10" or "6/8" holds runs whose boards sit at two offsets (the cell above). Under the merge:
# both runs and the time of each. A board whose resolution comes from fewer than 128 pixels (half
# the chip; the board table's count of the anointed combination) is a small open marker: at this
# step seven boards of H1 runs 56 and 67, each named with its count in the footer.
#
# The classes differ in more than the setting on the axis. The RFSel 3 runs came last, 9.0 to
# 10.0 days after the end of the irradiation, against 4 h to 7.5 days for the RFSel 1 and 2
# runs; F1 runs 13 and 14 ran at 370/390 V, every other F1 run at 450/480 V, and every H1 run at
# 530/550 V. The time under each run and the footer's bias line give both for every run.

# %%
p02 = [dict(p) for p in PTS_M]
LOW_02 = low_marks(p02)
CLASSES_02 = {t: [(c.rfsel, c.os) for c in CLASSES if t in c.runs] for t in TELS}
fig, lay = settings.res_by_class(
    p02, CLASSES_02, tel_titles=TEL_TITLES,
    tick_labels={t: {k: u"offset %s\nRFSel %d" % (k[1], k[0]) for k in CLASSES_02[t]}
                 for t in TELS},
    run_labels=RUN_LABELS, xlabel=u"Settings class: threshold offset above the baseline [DAC] "
                                  u"and RFSel",
    tag=style.settings_text(rfsel=None, offset=None), data=DATA_TEXT, legend_title=LEGEND_TITLE,
    ylim=YLIM,
    footer=u"\n".join([
        SELECTION,
        u"Every display run: %s." % counts_text(p02),
        u"%s: one band per board from the merged runs' board table, by the rule of the single "
        u"runs; under it both runs and their times." % merge_text(p02),
        u"Classes grouped by offset, a dotted line between the groups; a class \"8/10\" or "
        u"\"6/8\" holds boards at two offsets. F1 has no good run at RFSel 3, offset 6/8.",
        BAND_TEXT, low_text(LOW_02), RUN_TEXT, bias_text(p02)]))
payload = values(p02, lay)
payload.update(classes={t: [u"RFSel %d / offset %s" % k for k in v]
                        for t, v in CLASSES_02.items()}, merges=MERGES_DRAWN, low_pixels=LOW_02)
save(fig, "st02_res_vs_settings_class", payload,
     dict(conventions(), merge=MERGE_CONVENTION, low_pixels=LOW_CONVENTION), inputs=INPUTS_MERGED)

# %% [markdown]
# ## Figure st03: the step from RFSel 2 to RFSel 3
#
# The two classes of each telescope that step RFSel from 2 (R_f = 5.7 kOhm) to 3 (4.4 kOhm) at
# nearly one offset (`RFSEL_STEP_CLASSES` in the campaign module): one axis, four quarters, H1
# RFSel 2 | H1 RFSel 3 | F1 RFSel 2 | F1 RFSel 3, each board in the column of its own offset;
# 28 boards. On F1 every board keeps offset 10 across the step. On H1 only IH7 keeps its offset
# (8 and 8); IH11 goes from 8 to 6, IH12 and IH13 from 10 to 8, so the H1 half compares the RFSel
# step at nearly the same offsets, not at the same ones.
#
# The RFSel 3 runs also came later after the end of the irradiation than their RFSel 2
# partners (H1: run 56 at 7.5 days, runs 64 and 67 at 9.0 and 9.6 days; F1: runs 32 and 52 at
# 2.3 and 7.5 days, runs 71 and 86 at 9.6 and 10.0 days): the step mixes the RFSel with the
# time after the irradiation. Figure st05 draws every board against its offset over its noise
# width (x_B), which puts the boards at different offsets on one axis. As on st02, a board whose
# resolution comes from fewer than 128 pixels is a small open marker: the seven boards of H1 runs
# 56 and 67.

# %%
p03 = [dict(p) for p in PTS if (p["rfsel"], p["os"]) in campaign.RFSEL_STEP_CLASSES[p["tel"]]]
LOW_03 = low_marks(p03)
QUARTERS = [((t, r), sorted({p["offset"] for p in p03 if (p["tel"], p["rfsel"]) == (t, r)}))
            for t in TELS for r, _ in campaign.RFSEL_STEP_CLASSES[t]]
step_offsets = []
for t in TELS:
    (r2, _), (r3, _) = campaign.RFSEL_STEP_CLASSES[t]
    for chip in campaign.TELESCOPE_CHIPS[t]:
        o = [sorted({p["offset"] for p in p03 if p["tel"] == t and p["chip"] == chip
                     and p["rfsel"] == r}) for r in (r2, r3)]
        step_offsets.append((t, chip, o[0], o[1]))
same = [u"%s" % c for t, c, a, b in step_offsets if a == b]
moved = [u"%s %s to %s" % (c, u"/".join(map(str, a)), u"/".join(map(str, b)))
         for t, c, a, b in step_offsets if a != b]
fig, lay = settings.rfsel_step(
    p03, QUARTERS,
    quarter_labels={k: u"%s, %s" % (k[0].upper(), style.rfsel_text(k[1])) for k, _ in QUARTERS},
    tick_labels={(t, r, o): u"offset %d" % o for (t, r), offs in QUARTERS for o in offs},
    run_labels=RUN_LABELS, xlabel=u"Telescope and RFSel; threshold offset above the baseline "
                                  u"[DAC], each board at its own",
    name=u"H1 (HPK) and F1 (FBK) telescopes", tag=style.settings_text(rfsel=None, offset=None),
    data=DATA_TEXT, ylim=YLIM,
    footer=u"\n".join([
        SELECTION,
        u"The classes of each telescope that step RFSel from 2 to 3: %s. Colour = RFSel, as the "
        u"quarter labels." % counts_text(p03),
        u"Offsets kept across the step: %s; changed: %s." % (u", ".join(same),
                                                             u", ".join(moved)),
        BAND_TEXT, low_text(LOW_03), RUN_TEXT, bias_text(p03)]))
payload = values(p03, lay)
payload.update(quarters=[dict(tel=k[0], rfsel=k[1], offsets=v) for k, v in QUARTERS],
               offsets_across_step=[dict(tel=t, chip=c, rfsel2=a, rfsel3=b)
                                    for t, c, a, b in step_offsets], low_pixels=LOW_03)
save(fig, "st03_rfsel2_vs_3", payload, dict(conventions(), low_pixels=LOW_CONVENTION))

# %% [markdown]
# ## The threshold over the noise, x_B
#
# x_B is a board's threshold offset in units of its noise: the offset (DAC) over the median noise
# width (DAC) of the board's live pixels, the pixels whose noise width is above 0 (253 to 256 of
# 256 on these boards), both from the autocalibration scan the DAQ set the run's thresholds from:
# the offset from `blnw_by_run.csv`, the noise width of every pixel from `blnw_by_pixel.csv.gz`
# (`settings.pixel_x_over_nw`). Its bar runs from the 16th to the 84th percentile of the ratio
# offset / noise width over the same pixels: how much x_B varies over the pixels of the board, a
# spread and not an uncertainty. The noise width is an integer DAC (its median is 5 to 9 on these
# boards), so the medians are coarse and the bars asymmetric, one side empty where the median is
# also a percentile: they are drawn as they are.
#
# The autocalibration scan runs before the threshold offset is applied, so a run's noise width
# does not depend on its offset. It does depend on RFSel: the noise widths at RFSel 1 are
# larger than at RFSel 2, so one offset gives a smaller x_B at RFSel 1.
#
# A merge's runs each had their own autocalibration scan: the merge's x_B and bar pool the pixels
# of its runs' scans before the median and the percentiles (a scan two runs shared would count
# once; the cell prints which), and the values files keep each scan's pixel count.
#
# The cell below computes x_B for every board drawn in st04 and st05, the merge included. It
# checks that the scan's offset is the board's offset in `runs_summary.csv`, and the per-pixel
# table against `blnw_by_run.csv`: for every single run, the pixels with a noise width above 0
# are the board's `n_pixels_nw` and their median is its `nw_median`.

# %%
PIXELS = blnw.read_blnw_pixels()
for p in PTS_M:
    off = settings.scan_offset(BLNW, settings.CAMP, p["tel"], p.get("runs", [p["run"]]),
                               p["chip"])
    if off != p["offset"]:
        raise ValueError("%s run %s %s: offset %s in the autocalibration table, %s in "
                         "runs_summary" % (p["tel"].upper(), run_name(p), p["chip"], off,
                                           p["offset"]))
XB = settings.pixel_x_over_nw(PIXELS, pd.DataFrame(
    [dict(tel=p["tel"], runs=tuple(p.get("runs", [p["run"]])), chip=p["chip"], offset=p["offset"])
     for p in PTS_M]))
P_XB = []
for p, x in zip(PTS_M, XB.to_dict("records")):
    if "runs" not in p:
        r = BLNW[(BLNW["campaign"] == settings.CAMP) & (BLNW["tel"] == p["tel"])
                 & (BLNW["run"] == p["run"]) & (BLNW["chip"] == p["chip"])].iloc[0]
        if (x["n_pixels"], x["nw_median"]) != (int(r["n_pixels_nw"]), float(r["nw_median"])):
            raise ValueError("%s run %d %s: %d pixels, median noise width %g in the per-pixel "
                             "table; n_pixels_nw %d, nw_median %g in blnw_by_run.csv"
                             % (p["tel"].upper(), p["run"], p["chip"], x["n_pixels"],
                                x["nw_median"], r["n_pixels_nw"], r["nw_median"]))
    P_XB.append(dict(p, x_b=x["x"], xb_lo=x["lo"], xb_hi=x["hi"], nw_median=x["nw_median"],
                     n_pixels_nw=x["n_pixels"], n_pixels_scan=x["n_pixels_scan"],
                     n_scans=x["n_scans"], shared_scan=bool(x["shared_scan"])))
# the live pixels of every scan behind the points: a merge's runs each had their own
LIVE = [n for p in P_XB for n in p["n_pixels_scan"]]
N_LIVE = (min(LIVE), max(LIVE))
for p in P_XB:
    if "runs" in p:
        print("%s merge %s %s: x_B %.4f, bar %.4f-%.4f, median noise width %g over %d pixels of "
              "%d scan(s) (%s per scan)%s" % (
                  p["tel"].upper(), run_name(p), p["chip"], p["x_b"], p["xb_lo"], p["xb_hi"],
                  p["nw_median"], p["n_pixels_nw"], p["n_scans"],
                  ", ".join(map(str, p["n_pixels_scan"])),
                  ", the runs share one scan" if p["shared_scan"] else ""))
XB_TEXT = (u"x$_\\mathrm{B}$ = offset / median noise width of the live pixels; bar = 16th to 84th "
           u"percentile of offset / noise width over the pixels (live: noise width above 0, "
           u"%d-%d of 256 per scan).\nBoth from the autocalibration scan the DAQ set the run's "
           u"thresholds from (DAC), taken before the offset is applied; integer noise widths: "
           u"coarse medians, asymmetric bars." % N_LIVE)
print("x_B over %d boards: %.2f-%.2f, bars %.2f-%.2f; median noise width %s; live pixels %d-%d "
      "per scan" % (len(P_XB), min(p["x_b"] for p in P_XB), max(p["x_b"] for p in P_XB),
                    min(p["xb_lo"] for p in P_XB), max(p["xb_hi"] for p in P_XB),
                    sorted({p["nw_median"] for p in P_XB}), N_LIVE[0], N_LIVE[1]))


def pool_text(pts):
    """How the merges among `pts` pool their autocalibration scans, in words."""
    shared = sorted({p["chip"] for p in pts if "runs" in p and p["shared_scan"]})
    if not shared:
        return u"the pixels of its runs' scans pooled, one scan per run"
    return u"the pixels of its runs' scans pooled, a scan the runs share counted once (%s)" % (
        u", ".join(shared))


def xb_values(pts):
    keys = ("tel", "run", "board_idx", "chip", "rfsel", "os", "offset", "hv", "x_b", "xb_lo",
            "xb_hi", "nw_median", "n_pixels_nw", "n_pixels_scan", "n_scans", "shared_scan")
    return [dict({k: p[k] for k in keys}, **({"runs": p["runs"]} if "runs" in p else {}))
            for p in pts]


# %% [markdown]
# ## Figure st04: x_B of every board, per chip and class
#
# Every display run, 20 runs, 76 boards (F1 runs 46 and 47 as their merge, x_B and bar over the
# pixels of both runs' scans): one panel per telescope, x = chip. In each chip's slot the settings
# classes stand side by side in the order of the legend (by offset, then RFSel), the runs of a class
# side by side in run order. Colour = RFSel, as on every settings figure; the marker is the offset
# class, one marker per offset text. The chip is on the axis, so here the marker is free for the
# offset: the one settings figure whose marker is not the chip (`../CONVENTIONS.md`, "Settings
# figures"). The bar runs from the 16th to the 84th percentile of offset / noise width over the
# board's pixels (the cell above). The small open markers are the boards whose resolution (st02,
# st05) comes from fewer than 128 pixels; their x_B takes the live pixels like every other.
#
# H1 has five classes and F1 five. The sixth F1 cell the scan intended, RFSel 3 at offsets 6/8,
# is run 59, which the good-run list excludes (the setup cell prints the reason). On H1 the
# class RFSel 2, offset 20 is run 18 alone, the one display run there.
#
# The runs of one class were taken at different times after the end of the irradiation, and F1
# runs 13 and 14 (RFSel 2, offset 20) at a lower bias than run 18 of the same class: the footer
# lists the time of every run of a class with more than one run, and the bias.

# %%
YLIM_04 = (0.0, 6.0)          # the class legend sits above the highest bar
CLASSES_04 = {t: [(c.rfsel, c.os) for c in CLASSES if t in c.runs] for t in TELS}
MARKERS_04 = settings.offset_markers([c.os for c in CLASSES])
LABELS_04 = {(t, c.rfsel, c.os): u"%s, baseline + %s (%s %s%s)"
             % (style.rfsel_text(c.rfsel), c.os, u"runs" if len(c.runs[t]) > 1 else u"run",
                u", ".join(u"+".join(map(str, e)) if isinstance(e, tuple) else u"%d" % e
                           for e in tables.class_entries(c, t)),
                u" merged" if c.merges.get(t) else u"")
             for c in CLASSES for t in TELS if t in c.runs}


def class_times(tel):
    """The time after the stop of every run of each class of `tel` with more than one run."""
    out = []
    for c in CLASSES:
        runs = c.runs.get(tel, [])
        if len(runs) > 1:
            out.append(u"RFSel %d / offset %s: %s" % (c.rfsel, c.os, u", ".join(
                u"%d %s" % (r, FACTS[(tel, r)]["time"]) for r in runs)))
    return u"%s %s" % (tel.upper(), u"; ".join(out))


XB_CONVENTION = ("threshold offset (blnw_by_run.csv) / median noise width over the board's live "
                 "pixels (nw > 0, blnw_by_pixel.csv.gz; settings.pixel_x_over_nw)")
XB_BAR_CONVENTION = ("16th to 84th percentile of offset / nw over the same pixels (numpy's linear "
                     "interpolation): a pixel-to-pixel spread, not an uncertainty")
XB_MERGE_CONVENTION = ("; x_B and bar over the pooled pixels of its runs' autocalibration scans, a "
                       "scan the runs share once")
p04 = [dict(p) for p in P_XB]
LOW_04 = low_marks(p04)
fig, lay = settings.threshold_over_noise(
    p04, CLASSES_04, markers=MARKERS_04, class_labels=LABELS_04, tel_titles=TEL_TITLES,
    xlabel=u"Chip", ylabel=u"x$_\\mathrm{B}$ = threshold offset / median noise width",
    tag=style.settings_text(rfsel=None, offset=None), data=DATA_TEXT, ylim=YLIM_04,
    footer=u"\n".join([
        XB_TEXT,
        u"Every display run, %d runs and %d boards (the runs of each class in the legend). "
        u"Colour = RFSel, marker = offset class; in each chip's slot the classes in legend "
        u"order." % (sum(len(r) for r in runs_of(p04).values()), len(p04)),
        u"%s: one point per board, x$_\\mathrm{B}$ and bar over %s." % (merge_text(p04),
                                                                          pool_text(p04)),
        low_text(LOW_04),
        u"Time from the end of the 3.5e15 irradiation to the run start, the classes with "
        u"several runs:",
        u"%s;" % class_times("h1"),
        u"%s." % class_times("f1"),
        bias_text(p04)]))
payload = dict(points=xb_values(p04), layout=lay, ylim=YLIM_04,
               markers={o: m for o, m in MARKERS_04.items()},
               classes={t: [u"RFSel %d / offset %s" % k for k in v] for t, v in CLASSES_04.items()},
               run_facts={u"%s run %d" % k: v for k, v in FACTS.items()},
               live_pixels=N_LIVE, settings_guard=GUARD, step=STEP,
               good_run_drops={u"%s run %d" % k: v for k, v in drops.items()},
               merges=MERGES_DRAWN, low_pixels=LOW_04)
c = conventions()
c.update(x_b=XB_CONVENTION, bar=XB_BAR_CONVENTION,
         colour="RFSel (etroc_style.RFSEL_COLOR), marker = offset class (chip on the axis)",
         merge=MERGE_CONVENTION + XB_MERGE_CONVENTION,
         low_pixels=LOW_CONVENTION + ", its edge in the RFSel colour (no band behind it)")
save(fig, "st04_threshold_over_noise", payload, c, inputs=INPUTS_XB)

# %% [markdown]
# ## Figure st05: resolution against x_B
#
# Every board of every display run as a band, at its x_B: 20 runs, 76 boards (F1 runs 46 and 47
# as their merge), one panel per telescope. Colour = RFSel, marker = chip. The x bar, in the
# RFSel colour, runs from the 16th to the 84th percentile of offset / noise width over the
# board's pixels (the x_B cell); the y bar is the resolution's pixel-to-pixel spread, as on every
# band.
#
# x and y do not take the same pixels: x_B the board's live pixels (253 to 256), the resolution
# the pixels above the event floor of the track selection (the footer gives how many per board,
# and the fewest).
#
# F1 runs 46 and 47 (RFSel 1, offset 20) are an accepted merge (`f1_grp_run46_47` in the merged
# runs' board table) and are drawn as the merge, as on st02 and st04: one band per board from the
# merged table by the rule of the single runs, at the x_B of the pixels of both runs' scans
# pooled (each run had its own autocalibration scan; the x_B cell prints the counts).
#
# What else differs between the points: F1 runs 13 and 14 ran at a lower bias than F1 run 18
# (RFSel 2, offset 20), and every RFSel 3 run came later after the end of the irradiation than
# its RFSel 2 partners (the footer gives the times). As on st02, a board whose resolution comes
# from fewer than 128 pixels is a small open marker.

# %%
XLIM_05 = (0.8, 4.2)          # every x bar inside; the first tick (1.0) clear of the y tick labels
p05 = [dict(p) for p in P_XB]
LOW_05 = low_marks(p05)
if min(p["xb_lo"] for p in p05) <= XLIM_05[0] or max(p["xb_hi"] for p in p05) >= XLIM_05[1]:
    raise ValueError("an x bar of st05 reaches outside XLIM_05 %s" % (XLIM_05,))
fewest = min(p05, key=lambda p: (p["band"]["n_pixels"], TELS.index(p["tel"]),
                                 settings.run_order(p), p["board_idx"]))


def step_times(tel):
    """The RFSel step classes of `tel` (RFSEL_STEP_CLASSES): each class's runs and their times
    after the stop."""
    return u", ".join(u"RFSel %d %s" % (r, u", ".join(
        u"%d %s" % (run, FACTS[(tel, run)]["time"]) for c in CLASSES
        if (c.rfsel, c.os) == (r, o) for run in c.runs.get(tel, [])))
        for r, o in campaign.RFSEL_STEP_CLASSES[tel])


fig = settings.res_vs_x(
    p05, tel_titles=TEL_TITLES,
    xlabel=u"x$_\\mathrm{B}$ = threshold offset / median noise width", xlim=XLIM_05, ylim=YLIM,
    legend_title=LEGEND_TITLE, tag=style.settings_text(rfsel=None, offset=None), data=DATA_TEXT,
    footer=u"\n".join([
        SELECTION,
        u"Every display run: %s." % counts_text(p05),
        u"%s: one band per board from the merged runs' board table, by the rule of the single "
        u"runs, at the x$_\\mathrm{B}$ of %s." % (merge_text(p05), pool_text(p05)),
        BAND_TEXT,
        low_text(LOW_05),
        XB_TEXT,
        u"x takes the live pixels, y the pixels above the event floor (fewest: %s run %s %s, %d "
        u"pixels)." % (fewest["tel"].upper(), run_name(fewest), fewest["chip"],
                       fewest["band"]["n_pixels"]),
        u"Time from the end of the 3.5e15 irradiation to the run start, the RFSel 2 to 3 step: "
        u"H1 %s; F1 %s." % (step_times("h1"), step_times("f1")),
        bias_text(p05)]))
runs05 = runs_of(p05)
payload = dict(points=[dict(xb, **p["band"]) for xb, p in zip(xb_values(p05), p05)],
               run_facts={u"%s run %d" % (t, r): FACTS[(t, r)] for t in TELS for r in runs05[t]},
               xlim=XLIM_05, ylim_ps=YLIM, live_pixels=N_LIVE,
               fewest_pixels=dict(tel=fewest["tel"], run=fewest["run"], chip=fewest["chip"],
                                  n_pixels=fewest["band"]["n_pixels"]),
               settings_guard=GUARD, step=STEP,
               good_run_drops={u"%s run %d" % k: v for k, v in drops.items()},
               merges=MERGES_DRAWN, low_pixels=LOW_05)
c = conventions()
c.update(x="x_B = " + XB_CONVENTION, x_bar=XB_BAR_CONVENTION,
         y="band per board: " + tables.BAND_FOOTER,
         merge=MERGE_CONVENTION + XB_MERGE_CONVENTION, low_pixels=LOW_CONVENTION)
save(fig, "st05_res_vs_threshold_over_noise", payload, c, inputs=INPUTS_XB)

# %% [markdown]
# ## Figure st06: baseline and noise width against RFSel
#
# The baseline (BL) and noise width (NW) of every board in every display run, against the RFSel
# of the run: three categories, the runs of one category side by side in run order (offset
# included: the category holds every offset of that RFSel). One point per board and run from the
# autocalibration scan the DAQ set the run's thresholds from. Both statistics over the board's
# pixels are drawn, as in the frontend notebook's fe01: filled over all 256 pixels, open over the
# pixels with a value. The bar is the pixel-to-pixel standard deviation. Colour = board, marker
# = chip: the BL/NW figures colour by board (`../CONVENTIONS.md`).
#
# The scan runs before the discriminator offset is applied, so the offset of a run is no
# confound here. The bias and the time after the irradiation are: the small text over each
# category is the bias of its runs, the footer lists the times.
#
# F1 runs 46 and 47, the accepted merge that st02, st04 and st05 draw as one point per board,
# stay two runs here: st06 compares the autocalibration scans, one per run, not the points of the
# resolution figures. The footer says so.

# %%
QUANTITIES_06 = [dict(key="bl", ylabel="Baseline [DAC]"),
                 dict(key="nw", ylabel="Noise width [DAC]")]
STATS_06 = [dict(key="allpix", label="all 256 pixels", open=False),
            dict(key="live", label="pixels with a value", open=True)]
STEPS_06 = [dict(campaign=settings.CAMP, rfsel=r,
                 runs={t: sorted(run for c in CLASSES if c.rfsel == r for run in c.runs.get(t, []))
                       for t in TELS})
            for r in campaign.RFSEL_ORDER]
for st in STEPS_06:
    print("RFSel %d: %s" % (st["rfsel"], "; ".join("%s runs %s" % (t.upper(), st["runs"][t])
                                                    for t in TELS)))
# the fullest category (7 F1 runs at RFSel 2) spans 0.8; every category keeps that pitch
N_MAX_06 = max(len(st["runs"][t]) for st in STEPS_06 for t in TELS)
p06 = blnw.points(BLNW, STEPS_06, QUANTITIES_06, STATS_06, pitch=0.8 / (N_MAX_06 - 1))
EXCL_06 = {q: blnw.excluded_range(p06, q) for q in ("bl", "nw")}
# the merges st02, st04 and st05 draw stay their runs here: st06 compares scans, not points
SHARED_06 = {(p["tel"], run_name(p)) for p in P_XB if "runs" in p and p["shared_scan"]}
MERGES_06 = [u"%s runs %s %s" % (t.upper(), u" and ".join(m.split("+")),
                                 u"(one scan, drawn once per run)" if (t, m) in SHARED_06 else
                                 u"(a scan each)")
             for t in TELS for m in MERGES_DRAWN[t]]
MERGE_LINE_06 = ([u"%s stay separate runs here: st02, st04 and st05 draw the merge as one point "
                  u"per board; st06 compares scans, not points." % u"; ".join(MERGES_06)]
                 if MERGES_06 else [])


def rfsel_times(tel):
    return u"%s: %s" % (tel.upper(), u"; ".join(u"RFSel %d %s" % (st["rfsel"], u", ".join(
        u"%d %s" % (r, FACTS[(tel, r)]["time"]) for r in st["runs"][tel])) for st in STEPS_06))


fig = blnw.ladder_vs_fluence(
    p06, STEPS_06, QUANTITIES_06, STATS_06,
    tick_labels={t: [style.rfsel_text(st["rfsel"]) for st in STEPS_06] for t in TELS},
    xlabel=u"Preamp feedback setting", tel_titles=TEL_TITLES,
    tag=style.settings_text(rfsel=None, offset=None), data=DATA_TEXT, boundary_after=None,
    scale=0.72,
    footer=u"\n".join([
        u"Filled: mean over all 256 pixels (a railed or dead pixel as stored, a missing NW as 0). "
        u"Bar: pixel-to-pixel standard deviation.",
        u"Open: mean over the pixels with a value (BL: 50 < baseline < 1020 DAC and not railed, "
        u"%d-%d of 256 left out per board; NW: above 0, %d-%d left out). One autocalibration "
        u"scan per board and run, the scan the DAQ set the thresholds from."
        % (EXCL_06["bl"] + EXCL_06["nw"]),
        u"Every display run by its RFSel, every offset: %d runs, %d boards; the runs of a "
        u"category side by side in run order. Small text: sensor bias."
        % (sum(len(st["runs"][t]) for st in STEPS_06 for t in TELS),
           len({(p["tel"], p["run"], p["board_idx"]) for p in p06})),
        *MERGE_LINE_06,
        u"The autocalibration scan runs before the discriminator offset is applied, so the offset "
        u"within a category is no confound; bias and time after the stop differ between runs:",
        u"time from the end of the 3.5e15 irradiation to the run start, %s;" % rfsel_times("h1"),
        u"%s." % rfsel_times("f1")]))
save(fig, "st06_blnw_vs_rfsel",
     dict(points=p06, steps=[dict(rfsel=st["rfsel"], runs=st["runs"]) for st in STEPS_06],
          excluded_pixels=EXCL_06,
          run_facts={u"%s run %d" % k: v for k, v in FACTS.items()}),
     dict(statistics="allpix: mean and standard deviation over all 256 pixels; live: over the "
                     "pixels with a value (BL 50 < baseline < 1020 DAC, not railed; NW > 0)",
          error_bar="pixel-to-pixel standard deviation, not an uncertainty",
          categories="RFSel of the display runs at %g, every offset" % STEP,
          colour="board (etroc_style.BOARD_COLOR), marker = chip",
          times="run start (tables/run_times.csv) minus RAD_STOP_UTC of the step, hours"))

# %% [markdown]
# ## Figure st07: what RFSel changes, and what was measured
#
# Three panels. Left: where RFSel acts, the feedback resistor R_f of the preamplifier (the
# schematic of the overview notebook). Middle: the TOT of one F1 chip at offset 10 in the pair of
# runs that steps RFSel from 2 to 3 (`RFSEL_TOT_PAIRS` in the campaign module), from the RFSel
# pair table, drawn with the code of the frontend notebook's fe15: the pixels counted in both
# runs, every event of each, 0.1 ns bins, each curve as the fraction of its events, a tick at
# each median. Right: what RFSel changes by design, then the numbers measured here:
#
# * the median TOT shift of both RFSel pairs (offset 20: RFSel 1 to 2; offset 10: RFSel 2 to 3),
#   computed from the pair table's medians;
# * the resolution of the same chip in the two runs of the middle panel and in the other runs
#   of their classes (band centres, the bands of st02);
# * the time after the irradiation of the two runs: RFSel 3 came later, so the step mixes RFSel
#   with the time cooling down.
#
# The design lines follow the ETROC2 design description (a collaboration paper draft, not quoted
# here); the values file keeps, in this package's words, the statement each line rests on
# (`source_quotes`, with the draft's section and line) and what was left out for want of a source.

# %%
RF_HIST = pd.read_csv(campaign.RFSEL_PAIR_HIST_CSV)
if len(campaign.RFSEL_TOT_PAIRS) != 1:
    raise ValueError("st07 reads the RFSel pairs of one telescope; RFSEL_TOT_PAIRS has %d"
                     % len(campaign.RFSEL_TOT_PAIRS))
((RF_CAMP, RF_TEL), RF_PAIRS), = campaign.RFSEL_TOT_PAIRS.items()
# the pair drawn: the one whose runs are in the RFSel step classes of the telescope (RFSel 2 to
# 3 at one offset)
STEP_CLASSES = campaign.RFSEL_STEP_CLASSES[RF_TEL]
STEP_RUNS = [[r for c in CLASSES if (c.rfsel, c.os) == k for r in c.runs.get(RF_TEL, [])]
             for k in STEP_CLASSES]
(PAIR_07,) = [pr for pr in RF_PAIRS if pr[0] in STEP_RUNS[0] and pr[1] in STEP_RUNS[1]]
SHIFTS = [tot_shapes.median_shift(RF_HIST, RF_CAMP, RF_TEL, pr) for pr in RF_PAIRS]
RF_CHIP = SHIFTS[0]["chip"]
for sh in SHIFTS:
    print("runs %s %s: RFSel %s offset %s, median %s ns, shift %+.4f ns, %d pixels" % (
        sh["runs"], sh["chip"], sh["rfsel"], sh["offset"], sh["median_ns"], sh["shift_ns"],
        sh["n_pixels"]))
panel07 = tot_shapes.rfsel_curves(
    RF_HIST, RF_CAMP, RF_TEL, PAIR_07, RS, RUN_TIMES, campaign.RAD_STOP_UTC[STEP],
    label=u"%(rfsel)s, run %(run)d, %(time)s\nmedian %(median).2f ns, N = %(n)d",
    note=u"%(chip)s, the %(n_pixels)d pixels counted in both runs", width=0.1)
RES_07 = {}
for k, runs in zip(STEP_CLASSES, STEP_RUNS):
    for r in runs:
        (p,) = [p for p in PTS if (p["tel"], p["run"], p["chip"]) == (RF_TEL, r, RF_CHIP)]
        RES_07[r] = p["band"]
print("resolution of %s (band centres): %s" % (RF_CHIP, {r: round(b["marker_ps"], 2)
                                                        for r, b in RES_07.items()}))

# What RFSel changes by design: (head, body, the source statements it rests on)
RF_DESIGN = [
    (u"Preamp gain / amplitude", u"higher R$_\\mathrm{f}$ → more preamp gain, compensating "
     u"the LGAD signal loss after irradiation", "q_rf_tradeoff, q_vga_irrad"),
    (u"Return to baseline / TOT length", u"higher R$_\\mathrm{f}$ → slower trailing edge "
     u"→ longer TOT; lower R$_\\mathrm{f}$ → shorter TOT", "q_rf_tradeoff"),
    (u"Front-end noise / jitter", u"the preamp gain is set to balance timing performance "
     u"against noise-induced jitter", "q_gain_jitter"),
    (u"Discriminator threshold", u"set at the baseline + a programmable offset, independent of "
     u"RFSel", "q_disc_threshold"),
    (u"Rate capability", u"a longer TOT (higher R$_\\mathrm{f}$) keeps the front end busy longer "
     u"after each hit", "q_rf_tradeoff (inferred from the TOT length)"),
]
# Provenance of RF_DESIGN: the statement of the ETROC2 design description (a collaboration paper
# draft) each line rests on, in this package's words rather than the draft's, with the draft's
# section and line as the pointer; and the RFSel to R_f map.
SOURCE_QUOTES = {
    "q_gain_jitter": dict(
        text="The preamplifier is the first stage on the small signal of the DC-coupled LGAD: it "
             "needs enough preamp gain with little noise-induced jitter, and its settings trade "
             "timing performance against power and make up for the LGAD signal lost to "
             "irradiation over the detector lifetime.",
        source="ETROC2 design description (paper draft), design section, line 25"),
    "q_tia": dict(
        text="The preamplifier is a buffered transimpedance amplifier, an NMOS-input cascode with "
             "resistive feedback; separate bias currents set the leading-edge slew and the "
             "input-transistor transconductance.",
        source="ETROC2 design description (paper draft), design section, line 27"),
    "q_rf_tradeoff": dict(
        text="The programmable feedback resistor sets the trailing edge the TOT measures and the "
             "transimpedance: a lower value gives a faster trailing edge while the LGAD signal "
             "suffices, a higher value more preamp gain to make up for the LGAD signal lost to "
             "irradiation; a programmable load capacitor also shapes the bandwidth and the pulse.",
        source="ETROC2 design description (paper draft), design section, line 29"),
    "q_disc_threshold": dict(
        text="The discriminator turns the amplified signal into a digital pulse when it crosses a "
             "programmable threshold, set relative to the pixel baseline with a margin above the "
             "noise so that the efficiency stays high and noise hits are suppressed.",
        source="ETROC2 design description (paper draft), design section, line 44"),
    "q_pixel_config": dict(
        text="Preamplifier, discriminator and TDC are configurable in every pixel: the preamp bias "
             "current trades timing performance against power, the feedback resistance sets the "
             "preamp gain and the fall time of the output pulse, and the discriminator hysteresis "
             "is programmable.",
        source="ETROC2 design description (paper draft), design section, line 497"),
    "q_tdc": dict(
        text="One delay line digitizes both the TOA and the TOT of each discriminator pulse.",
        source="ETROC2 design description (paper draft), design section, line 73"),
    "q_vga_irrad": dict(
        text="A variable-gain amplifier (VGA) at the sampler input takes the smaller signals "
             "expected after the LGAD signal shrinks with irradiation.",
        source="ETROC2 design description (paper draft), design section, line 510"),
    "q_rfsel_kohm": dict(
        text="RFSel 0/1/2/3 -> feedback resistor 20.0 / 10.0 / 5.7 / 4.4 kOhm.",
        source="etroc_plots style.RFSEL_RF_KOHM"),
}
OMITTED_UNSUPPORTED = (
    "A leakage-current-driven baseline shift against RFSel is not drawn: neither the design "
    "description nor the project's digest of the ETROC2 reference manual states it. The design "
    "description covers irradiation's effect on the front end only through the LGAD signal loss "
    "(q_rf_tradeoff, q_vga_irrad), which the first design line uses instead.")


def kohm(r):
    return u"%g" % style.RFSEL_RF_KOHM[r]


MEASURED_07 = [u"Offset %d, R$_\\mathrm{f}$ %s → %s k$\\Omega$ (runs %d → %d): median TOT "
               u"%.2f → %.2f ns, the %d pixels counted in both runs"
               % (sh["offset"][0], kohm(sh["rfsel"][0]), kohm(sh["rfsel"][1]), sh["runs"][0],
                  sh["runs"][1], sh["median_ns"][0], sh["median_ns"][1], sh["n_pixels"])
               for sh in SHIFTS]
others = [r for runs in STEP_RUNS for r in runs if r not in PAIR_07]
MEASURED_07.append(
    u"Resolution of %s at offset %s (band centres): RFSel %d %.1f ps (run %d) → RFSel %d "
    u"%.1f ps (run %d); the other runs of these classes: %s"
    % (RF_CHIP, STEP_CLASSES[0][1], STEP_CLASSES[0][0], RES_07[PAIR_07[0]]["marker_ps"],
       PAIR_07[0], STEP_CLASSES[1][0], RES_07[PAIR_07[1]]["marker_ps"], PAIR_07[1],
       u", ".join(u"run %d %.1f ps" % (r, RES_07[r]["marker_ps"]) for r in others)))
t_pair = [FACTS[(RF_TEL, r)] for r in PAIR_07]
MEASURED_07.append(
    u"Run %d started %s after the end of the irradiation, run %d %s: the RFSel step also adds "
    u"%.1f days of cooling down (bias %s in both runs)"
    % (PAIR_07[0], t_pair[0]["time"], PAIR_07[1], t_pair[1]["time"],
       (t_pair[1]["hours"] - t_pair[0]["hours"]) / 24.0,
       t_pair[0]["bias"] if t_pair[0]["bias"] == t_pair[1]["bias"] else
       u"%s, %s" % (t_pair[0]["bias"], t_pair[1]["bias"])))
for m in MEASURED_07:
    print(m)

fig = cartoons.rfsel_measured(
    panel07, xrange=tot_shapes.common_range(panel07["hists"], 0.1, floor=0.0),
    top=1.75 * max(float(np.max(cv["y"])) for cv in panel07["curves"]),
    xlabel=u"TOT [ns]", ylabel=u"Fraction of events per 0.1 ns bin",
    lines=[(h, b) for h, b, _ in RF_DESIGN],
    measured_head=u"Measured on %s:" % RF_CHIP,
    measured=MEASURED_07, closing=None,
    subjects=[u"Where RFSel acts", u"Measured: %s, baseline + %s" % (RF_CHIP, STEP_CLASSES[0][1]),
              u"What RFSel changes"],
    tag=style.settings_text(rfsel=None, offset=None), data=DATA_TEXT,
    footer=u"\n".join([
        u"Left: schematic of the front end, not to scale. Right: the design lines, then the "
        u"numbers measured on %s: median TOT of every event of the pixels counted in both runs "
        u"of each pair (unbinned)." % RF_CHIP,
        u"Middle: TOT of %s in runs %d and %d, the %d pixels counted in both runs (a pixel "
        u"counts when its triple holds >= %d events), every event of each; 0.1 ns bins, each "
        u"curve as the fraction of its events; tick on the axis: the median."
        % (RF_CHIP, PAIR_07[0], PAIR_07[1], panel07["records"][0]["n_pixels"], tot_shapes.FLOOR),
        u"Legend: RFSel, run, time from the end of the 3.5e15 irradiation to the run start, "
        u"median TOT and events. Bias of %s: %s." % (RF_CHIP, blnw.bias_text(panel07["hv"])),
        SELECTION, BAND_TEXT]))
save(fig, "st07_rfsel_measured",
     dict(curves=panel07["records"], pair=list(PAIR_07), chip=RF_CHIP, telescope=RF_TEL,
          campaign=RF_CAMP, median_shifts=SHIFTS,
          resolution={u"run %d" % r: b for r, b in RES_07.items()},
          measured_lines=MEASURED_07,
          design_lines=[dict(head=h, body=b, supports=q) for h, b, q in RF_DESIGN],
          source_quotes=SOURCE_QUOTES, omitted_unsupported=OMITTED_UNSUPPORTED,
          run_facts={u"%s run %d" % (RF_TEL, r): FACTS[(RF_TEL, r)] for r in PAIR_07}),
     dict(pairs="RFSEL_TOT_PAIRS of the campaign module: runs that differ in RFSel only; the "
                "middle panel draws the pair in the RFSEL_STEP_CLASSES of the telescope",
          pixels="per pair, the chip's pixels counted (triple >= %d events) in both runs; every "
                 "event of each pixel's triple" % tot_shapes.FLOOR,
          bins="0.1 ns (summed from the stored 0.05 ns), fraction of the curve's events",
          median_ns="median TOT of every event of the curve (from the table, unbinned)",
          resolution="band centre: " + tables.BAND_FOOTER,
          times="run start minus RAD_STOP_UTC of the step, hours"))
