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
#
# The runs are the display runs of the step (`july/display_runs_jul.csv`): 9 on H1 and 11 on F1,
# 80 boards. Every board is drawn as a band, the rule at this step (`../CONVENTIONS.md`,
# "Resolution and TDC figures"): its value over every 3-board combination containing it.
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
# * `tables/runs_summary.csv`: per run and board the chip, bias, bookkeeping fluence, RFSel and
#   threshold offset (each board's own offset; the bias in the footers).
# * `july/display_runs_jul.csv`: the July display runs, their settings class and the preferred
#   run of each class.
# * `july/good_runs.csv`: the good-run list; every run drawn has to be good.
# * `tables/run_times.csv`: the start of every run, for the time after the irradiation.
# * `frontend/blnw_by_run.csv`: the RFSel the autocalibration scan before each run recorded, an
#   independent check of the RFSel in `runs_summary.csv`.
#
# The drawing lives in `../resolution/settings.py`, the classes and bands in
# `../resolution/tables.py`. This notebook holds the choices made per figure: the runs, the
# words and the y range.
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
from etroc_plots.frontend import blnw                         # noqa: E402
from etroc_plots.overview import timeline                     # noqa: E402
from etroc_plots.resolution import settings, tables           # noqa: E402

if campaigns.NAME != CAMPAIGN:
    raise RuntimeError("the package is running campaign %r (ETROC_CAMPAIGN) but CAMPAIGN = %r: "
                       "set one to match the other, then restart the kernel (the campaign is "
                       "read once, on the first import)" % (campaigns.NAME, CAMPAIGN))
campaign = campaigns.active

# stops, listing every missing or unreadable input; reports any table that differs from the
# published copy.
check_inputs(only=("tables/irrad/board_table_irrad.csv", "tables/runs_summary.csv",
                   "tables/run_times.csv", "july/display_runs_jul.csv", "july/good_runs.csv",
                   "frontend/blnw_by_run.csv"))

NOTEBOOK = "TestBeam/etroc_plots/notebooks/settings.ipynb"   # "script" in every values file
OUT = os.path.join(os.path.abspath(os.environ.get("ETROC_FIGURES") or "figures"), "settings")
os.makedirs(OUT, exist_ok=True)
INPUTS = [campaign.BOARD_TABLE, campaign.RUNS_SUMMARY_CSV, campaign.DISPLAY_RUNS_JUL_CSV,
          campaign.GOOD_RUNS_CSV, campaign.RUN_TIMES_CSV, campaign.BLNW_CSV]

VARIANT, FLOOR = "top", "300"          # the track variant and event floor of every band
STEP = campaign.SETTINGS_STEP          # 3.5e15
TELS = list(campaign.TELESCOPE_CHIPS)
BT = tables.read_board_table(campaign.BOARD_TABLE)
RS = tables.read_runs_summary(campaign.RUNS_SUMMARY_CSV)
BLNW = blnw.read_blnw()
RUN_TIMES = timeline.read_run_times()


def show(stem):
    """Display a saved figure, exactly as written to disk."""
    display(Image(filename=os.path.join(OUT, stem + ".png"), width=900))


def save(fig, stem, payload, conventions):
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
                       inputs=INPUTS)
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
#   the RFSel in `runs_summary.csv`. The guard stops the notebook, naming the run, on any
#   mismatch (it caught H1 runs 64, 67 and 70, which `runs_summary.csv` listed at RFSel 2 until
#   2026-10-02).
#
# F1 has no class at RFSel 3 and offsets 6/8: its one run there, 59, is not good on the
# good-run list (the cell below prints the reason).
#
# The runs of one class were taken at different times after the end of the 3.5e15 irradiation,
# and on F1 at two biases: the cell below lists both for every run; the figures write the time
# under each run and the bias in the footer.

# %%
CLASSES = tables.classes_at_step(campaign.DISPLAY_RUNS_JUL_CSV, STEP)
for c in CLASSES:
    print("RFSel %d, offset %-5s %s" % (c.rfsel, c.os, "; ".join(
        "%s runs %s (preferred %s)" % (t.upper(), c.runs[t], c.preferred[t]) for t in TELS
        if t in c.runs)))
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
print("boards drawn: %d" % len(PTS))

# %% [markdown]
# ## The words every figure shares
#
# The selection, the band rule, the run labels and the bias line, and what every values file
# records.

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


def runs_of(pts):
    return {t: sorted({p["run"] for p in pts if p["tel"] == t}) for t in TELS}


def counts_text(pts):
    runs = runs_of(pts)
    n = [p["band"]["n_pixels"] for p in pts]
    return (u"%d runs (%s), %d boards, %d-%d pixels per board"
            % (sum(len(r) for r in runs.values()),
               u", ".join(u"%s %s" % (t.upper(), u", ".join(map(str, runs[t]))) for t in TELS
                          if runs[t]), len(pts), min(n), max(n)))


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
    runs = runs_of(pts)
    return dict(points=[dict({k: p[k] for k in keys}, **p["band"]) for p in pts],
                run_labels={u"%s run %d" % (t, r): RUN_LABELS[(t, r)] for t in TELS
                            for r in runs[t]},
                run_facts={u"%s run %d" % (t, r): FACTS[(t, r)] for t in TELS for r in runs[t]},
                layout={k: (v if k != "run_centres" else
                            {u"%s run %d" % (str(s), r): x for (s, r), x in v.items()})
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


# %% [markdown]
# ## Figure st01: resolution against the threshold offset at RFSel 2
#
# Every board of every display run at RFSel 2, in the cell of its own threshold offset: 12 runs,
# 48 boards (H1 runs 18, 33, 52, 53 and 56; F1 runs 13, 14, 18, 32, 50, 51 and 52). H1 run 56
# stands in two cells, IH7 and IH11 at offset 8, IH12 and IH13 at 10. One RFSel only, so the
# offset is the one setting that changes between the cells.
#
# It is not the only thing that changes. The H1 boards at offset 10 ran 2.3 to 7.5 days after
# the end of the irradiation, its offset-20 run 37 h after; on F1 the offset-20 runs came first
# (4 h to 46 h) and runs 13 and 14 ran at 370/390 V, every other F1 run at 450/480 V. Read a
# difference between cells together with the times under the runs and the bias line of the
# footer.

# %%
p01 = [dict(p) for p in PTS if p["rfsel"] == 2]
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
        BAND_TEXT, RUN_TEXT, bias_text(p01)]))
payload = values(p01, lay)
payload.update(offsets=OFFSETS_01, split_runs=split_text)
save(fig, "st01_res_vs_offset_rfsel2", payload, conventions())

# %% [markdown]
# ## Figure st02: resolution against the settings class
#
# Every board of every display run, 20 runs and 80 boards, in the slot of its run's settings
# class. The classes are grouped by offset (a dotted line between the groups), RFSel rising
# within a group; each telescope shows the classes it has. A class at "8/10" or "6/8" holds runs
# whose boards sit at two offsets (the cell above).
#
# The classes differ in more than the setting on the axis. The RFSel 3 runs came last, 9.0 to
# 10.0 days after the end of the irradiation, against 4 h to 7.5 days for the RFSel 1 and 2
# runs; F1 runs 13 and 14 ran at 370/390 V, every other F1 run at 450/480 V, and every H1 run at
# 530/550 V. The time under each run and the footer's bias line give both for every run.

# %%
p02 = [dict(p) for p in PTS]
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
        u"Classes grouped by offset, a dotted line between the groups; a class \"8/10\" or "
        u"\"6/8\" holds boards at two offsets. F1 has no good run at RFSel 3, offset 6/8.",
        BAND_TEXT, RUN_TEXT, bias_text(p02)]))
payload = values(p02, lay)
payload.update(classes={t: [u"RFSel %d / offset %s" % k for k in v]
                        for t, v in CLASSES_02.items()})
save(fig, "st02_res_vs_settings_class", payload, conventions())

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
# time after the irradiation. Figure st05, added with the threshold-over-noise figures, draws
# the same boards against each board's offset over its noise width: the view free of the offset
# difference.

# %%
p03 = [dict(p) for p in PTS if (p["rfsel"], p["os"]) in campaign.RFSEL_STEP_CLASSES[p["tel"]]]
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
        BAND_TEXT, RUN_TEXT, bias_text(p03)]))
payload = values(p03, lay)
payload.update(quarters=[dict(tel=k[0], rfsel=k[1], offsets=v) for k, v in QUARTERS],
               offsets_across_step=[dict(tel=t, chip=c, rfsel2=a, rfsel3=b)
                                    for t, c, a, b in step_offsets])
save(fig, "st03_rfsel2_vs_3", payload, conventions())
