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
# # Resolution: time resolution against bias, fluence and time after the stop, CERN IRRAD 2026
#
# The time resolution of every board of the two telescopes (H1: HPK sensors on chips IH7, IH11,
# IH12, IH13; F1: FBK sensors on LF10, LF12, LF17, LF18) against the bias, across the March
# fluence steps, and against the time after an irradiation step or after the telescopes went
# back into the beam:
#
# * Figure re01: the March bias scan before irradiation.
# * Figure re02: the March bias scans after each irradiation step (3e14, 9e14 and 1.5e15 p/cm2,
#   bookkeeping fluence).
# * Figure re03: the March bias scan at 1.5e15 beside the July points at the same fluence, four
#   months of cooling down later.
# * Figure re04: July at 1.5e15 against the hours since the first run there: a stability check
#   after the telescopes were reinstalled.
# * Figure re05: July at 2e15 against the hours after the end of the irradiation step, with each
#   chip's in-run bias current.
# * Figure re06: July at 3.5e15 against the bias and against the hours after the end of the
#   irradiation step.
# * Figure re07: every bias scan of both campaigns on one figure per telescope, every July display
#   run as a single run.
# * Figure re08: the fluence ladder, each board's resolution at each step at its highest bias.
# * Figure re09: the numbers behind re08, as a table (.csv, .md and a table figure).
# * Figure re10: the pixels behind the re08 points, with at least 300 events and with any events.
# * Figure re11: the combination split at 3.5e15: each board's resolution in every 3-board
#   combination containing it, run by run.
#
# The appendix holds figures of the July talk drawn from the points above, each by the rules of
# the figure it derives from:
#
# * Figures re12, re13 and re14: re07's points at fewer steps: July at 1.5e15 and 2e15; July at
#   1.5e15, 2e15 and 3.5e15; March.
# * Figure re15: re06's hours panel at the standard operating point alone: 3.5e15 against the
#   hours after the end of the irradiation step.
#
# Every point is one board in one run: its value in its anointed 3-board combination (boards
# 0-1-2 for boards 0, 1 and 2; boards 1-2-3 for board 3), the most-populated track per pixel,
# tracks with at least 300 events; the width is FWHM/2.355 of the TOA-difference core, as the
# result tables carry it. The bar is the pixel-to-pixel spread of the board's map. Every point at
# 3.5e15 is a band: a shaded box from the lowest to the highest value of the board over every
# 3-board combination containing it, the marker at its centre, the bar the anointed
# combination's pixel-to-pixel spread (`../CONVENTIONS.md`, "Resolution and TDC figures").
# re01 to re05 have no 3.5e15 point, so no band; re11 compares the combinations themselves and
# draws each one.
#
# A point built from fewer than 128 pixels (half the chip; `curves.LOW_PIXELS`) is drawn as a
# small open marker on every figure, the legend says "under 128 pixels" and the footer gives the
# counts (on re11 the small marker keeps its fill: filled is the anointed combination there).
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
# 3. Open `resolution.ipynb` in Jupyter and run all cells, or run it headless with
#    `jupyter nbconvert --to notebook --execute resolution.ipynb --output-dir <somewhere>`.
#
# Figures go to `$ETROC_FIGURES/resolution/`, by default `figures/resolution/` next to this
# notebook.
#
# The tables read:
#
# * `tables/irrad/board_table_irrad.csv`: the time resolution of every board in every 3-board
#   combination of every single run.
# * `tables/merged/board_table_merged.csv`: the same for the merged runs.
# * `tables/runs_summary.csv`: per run and board the chip, bias, bookkeeping fluence, RFSel and
#   threshold offset.
# * `july/display_runs_jul.csv`: the July display runs and the preferred run of each step and
#   settings class.
# * `july/good_runs.csv`: the good-run list, single runs and merges; every run drawn has to be
#   good.
# * `tables/run_times.csv`: the start of every run, for the hours on the time axes.
# * `july/july_inrun_currents.csv`: each chip's bias current during each July run (re05).
#
# The drawing lives in `../resolution/curves.py` (bias and time axes),
# `../resolution/ladder.py` (the fluence ladder and the pixel yield),
# `../resolution/numbers_table.py` (the table) and `../resolution/combo_split.py` (re11), the
# tables and bands in
# `../resolution/tables.py`. This notebook holds the choices made per figure: the runs, the
# words and the axis ranges. The table of re09 is written as `re09_numbers_table.csv` and `.md`
# beside its figure.
#
# `resolution.ipynb` and `resolution.py` are the same notebook, kept in step by jupytext. Edit
# either one, then run `python3 -m jupytext --sync resolution.ipynb` and
# `python3 -m nbstripout resolution.ipynb` in this folder before committing. Neither tool is in
# LCG_104d (the install line is in `../README.md`); running the notebook needs neither.
#
# After a run, `python3 -m etroc_plots.checks <figures>`, run from `TestBeam/` with `<figures>`
# the folder this notebook wrote to (`etroc_plots/notebooks/figures/resolution` by default),
# checks every saved figure against the rules in `../CONVENTIONS.md`.

# %%
import os
import sys

import matplotlib
matplotlib.use("Agg")   # draw off-screen; show() displays the PNG exactly as saved
import matplotlib.pyplot as plt
from IPython.display import Image, display

# The folder that holds the package (TestBeam/) is two folders up. Jupyter starts the kernel in
# the notebook's folder; `python resolution.py` has __file__.
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
from etroc_plots.resolution import curves, settings, tables   # noqa: E402
from etroc_plots.resolution import combo_split, ladder, numbers_table   # noqa: E402

if campaigns.NAME != CAMPAIGN:
    raise RuntimeError("the package is running campaign %r (ETROC_CAMPAIGN) but CAMPAIGN = %r: "
                       "set one to match the other, then restart the kernel (the campaign is "
                       "read once, on the first import)" % (campaigns.NAME, CAMPAIGN))
campaign = campaigns.active

# stops, listing every missing or unreadable input; reports any table that differs from the
# published copy.
check_inputs(only=("tables/irrad/board_table_irrad.csv", "tables/merged/board_table_merged.csv",
                   "tables/runs_summary.csv", "tables/run_times.csv", "july/display_runs_jul.csv",
                   "july/good_runs.csv", "july/july_inrun_currents.csv"))

NOTEBOOK = "TestBeam/etroc_plots/notebooks/resolution.ipynb"   # "script" in every values file
OUT = os.path.join(os.path.abspath(os.environ.get("ETROC_FIGURES") or "figures"), "resolution")
os.makedirs(OUT, exist_ok=True)
INPUTS = [campaign.BOARD_TABLE, campaign.BOARD_TABLE_MERGED, campaign.RUNS_SUMMARY_CSV,
          campaign.DISPLAY_RUNS_JUL_CSV, campaign.GOOD_RUNS_CSV, campaign.RUN_TIMES_CSV,
          campaign.JULY_REF_CSV]

VARIANT, FLOOR = "top", "300"          # the track variant and event floor of every point
TELS = list(campaign.TELESCOPE_CHIPS)
BT = tables.read_board_table(campaign.BOARD_TABLE)
BM = tables.read_board_table(campaign.BOARD_TABLE_MERGED)
RS = tables.read_runs_summary(campaign.RUNS_SUMMARY_CSV)
RUN_TIMES = timeline.read_run_times()
GOOD = {(c, t): curves.good_list(c, t) for c in ("march", "july") for t in TELS}
MERGES = {t: curves.table_merges(BM, t, "july") for t in TELS}
CURRENTS = curves.inrun_currents()


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
# ## The runs, the display list and the merge rule
#
# March has no display-run list: the bias scans (re01-re03) draw every March run of the step at
# the standard operating point, RFSel 2 (`MARCH_RFSEL`: the March runs record no RFSel) and
# threshold offset 20; the runs at offset 10 are named in the footers.
#
# The July points come from the display-run list, at the standard operating point (RFSel 2,
# offset 20): at each step, its display runs and its preferred run. Where a merge of runs exists,
# the merge rule decides whether a result figure draws it: a merge replaces its runs only when
# every run of the merge is in the step's display class and the good-run list marks the merge
# good (`curves.merge_rule`). So H1 3+4 (1.5e15) and 11+12 (2e15) replace the preferred runs 4
# and 12; H1 6+7 and 18+19 are not good as merges, H1 14+15 and F1 3+4 hold runs off the display
# class, F1 46+47 sits at RFSel 1. At 3.5e15 no merge applies: the result run is 18 on both
# telescopes. The cell below prints the rule's outcome for every merge at every July step.
#
# A figure whose axis is time draws single runs: a merge spans the times of its runs (re04,
# re05). Two figures override the rule and say so in their footers: re03 and re07 draw the
# populations of the deck's figures, H1 run 4 and the other display runs as single runs
# (`curves.merge_override` names the merges they leave out); the fluence ladder (re08 to re10)
# draws the merges by the rule. re11 draws every settings class at 3.5e15 and takes the rule of
# the settings figures: an accepted merge whose runs are all in one class stands in for them (F1
# 46+47). The appendix figures draw the points of the figures they derive from (re12 to re14
# those of re07, re15 those of re06), so the same choices. Every run drawn, single or merged, has
# to be good on the good-run list; the cells stop otherwise.

# %%
STEPS_JUL = (1.5e15, 2e15, 3.5e15)
CLASS = {f: curves.display_class(campaign.DISPLAY_RUNS_JUL_CSV, f) for f in STEPS_JUL}
RESULT, MERGE_EFFECT = {}, {}
for f in STEPS_JUL:
    runs, preferred = CLASS[f]
    for t in TELS:
        drawn, taken, refused = curves.merge_rule([preferred[t]], MERGES[t], runs[t],
                                                  GOOD[("july", t)])
        RESULT[(t, f)] = drawn[0]
        MERGE_EFFECT["%s %g" % (t, f)] = dict(
            display_runs=runs[t], preferred=preferred[t], drawn=curves.run_label(drawn[0]),
            taken={curves.run_label(m): r for m, r in taken.items()},
            refused={curves.run_label(m): why for m, why in refused.items()})
        print("%s %g: display runs %s, preferred %d -> result run %s" % (
            t.upper(), f, runs[t], preferred[t], curves.run_label(drawn[0])))
        for m, why in refused.items():
            print("    merge %s refused: %s" % (curves.run_label(m), why))
bad = {t: curves.not_good([RESULT[(t, f)] for f in STEPS_JUL], GOOD[("july", t)]) for t in TELS}
if any(bad.values()):
    raise RuntimeError("result runs not good on the good-run list: %s" % bad)

# %% [markdown]
# ## The words every figure shares

# %%
YLIM = (20.0, 80.0)            # ps, every figure
TEL_TITLES = dict(campaign.TELESCOPE_TITLE)
TAG = style.settings_text(rfsel=2, offset=20)
# one footer line each: a footer line stays under about 180 characters to fit the figure width
SELECTION = (u"Selection: %s, tracks with at least %s events, each chip from its anointed "
             u"combination (boards 0-1-2 for boards 0-2, boards 1-2-3 for board 3).\n"
             u"Widths = FWHM/2.355 of the TOA-difference core, as the result tables carry them; "
             u"bar = the pixel-to-pixel spread of the board's map." % (tables.VARIANT_TEXT[VARIANT],
                                                                     FLOOR))
FLUENCE_TEXT = (u"Fluence: the facility's bookkeeping fluence of 24 GeV protons (on the chips "
                u"%.3g times that, from the dosimetry)." % campaign.CHIP_FLUENCE_FACTOR)


def step_text(f):
    return style.fluence_label(f)


def runs_of(pts, tel):
    return sorted({p["label"] for p in pts if p["tel"] == tel},
                  key=lambda s: [int(x) for x in s.split("+")])


def bias_text(pts, tel):
    """The runs of `tel` in `pts` with their bias, runs of one bias pattern together:
    "1 (180 V), 2 (170 V)" or "11, 12 (IH7 540 V, the others 550 V)"."""
    groups = {}
    for label in runs_of(pts, tel):
        chips = tuple((p["chip"], p["hv"]) for p in sorted(pts, key=lambda p: p["board_idx"])
                      if p["tel"] == tel and p["label"] == label)
        groups.setdefault(chips, []).append(label)
    parts = []
    for chips, labels in groups.items():
        by = {}
        for c, v in chips:
            by.setdefault(v, []).append(c)
        if len(by) == 1:
            b = blnw.bias_text(list(by))
        else:
            main = max(by, key=lambda v: len(by[v]))
            b = u", ".join(u"%s %d V" % (u", ".join(cs), v) for v, cs in by.items() if v != main)
            b += u", the others %d V" % main
        parts.append(u"%s (%s)" % (u", ".join(labels), b))
    return (u", " if all(len(l) == 1 for l in groups.values()) else u"; ").join(parts)


def run_list(tel, runs):
    """"H1 run 5" / "F1 runs 3, 4"."""
    runs = list(runs)
    return u"%s run%s %s" % (tel.upper(), u"s" if len(runs) > 1 else u"", u", ".join(map(str, runs)))


def counts_text(pts):
    n = [p["n_pixels"] for p in pts]
    return u"%d boards, %d-%d pixels per board" % (len(pts), min(n), max(n))


def low_line(pts):
    """[the footer line of the points under LOW_PIXELS] or [] when there are none: "Small open
    markers, under 128 pixels: H1 run 11 (IH7 45, IH11 35, IH12 40, IH13 79)."."""
    runs = {}
    for q in curves.low_pixel_points(pts):
        chips = runs.setdefault((q["tel"], q["label"]), [])
        if u"%s %d" % (q["chip"], q["n_pixels"]) not in chips:
            chips.append(u"%s %d" % (q["chip"], q["n_pixels"]))
    if not runs:
        return []
    return [u"Small open markers, under %d pixels (half the chip): %s." % (
        curves.LOW_PIXELS, u"; ".join(u"%s run %s (%s)" % (t.upper(), r, u", ".join(c))
                                      for (t, r), c in runs.items()))]


def check_good(pts, camp):
    bad = {t: curves.not_good({tuple(p["runs"]) if p["merged"] else p["runs"][0]
                               for p in pts if p["tel"] == t}, GOOD[(camp, t)]) for t in TELS}
    if any(bad.values()):
        raise RuntimeError("%s runs drawn that are not good: %s" % (camp, bad))
    return {t: u"every run good" for t in TELS}


def conventions(**extra):
    c = tables.conventions(VARIANT, FLOOR)
    c.update(width="FWHM/2.355 of the TOA-difference core per pair, as the board table carries "
                   "it", bar="pixel-to-pixel spread of the anointed combination",
             marker="chip (style.chip_marker); open marker and black edge as the legends say",
             x_repeats="no jitter: points at one x share it; lines join the mean at each x",
             fluence="bookkeeping, p/cm2; on chip = %g x bookkeeping (CHIP_FLUENCE_FACTOR)"
                     % campaign.CHIP_FLUENCE_FACTOR)
    c.update(extra)
    return c


def payload(numbers, footer, **extra):
    numbers = dict(numbers)
    numbers.update(footer=footer, **extra)
    return numbers


# %% [markdown]
# ## Figure re01: the March bias scan before irradiation
#
# Every March run before irradiation at the standard operating point (RFSel 2, threshold offset
# 20), each board at its run's bias: H1 runs 1, 2, 4, 5, 6, 7, 9 from 90 to 190 V; F1 runs 1, 2,
# 4, 5, 6, 9 from 150 to 240 V. The runs at offset 10 (H1 run 8, F1 runs 7 and 8) are left out
# and named in the footer: the offset changes the resolution by itself. The population is
# derived from the tables, not listed by hand: every single run of the board table at
# pre-irradiation, kept when every board is at the standard point.
#
# One fluence step, so the colour is the board (one colour per chip, as the bias-current
# figures) and the step is named in the header. The lines join each chip's points in bias order;
# no run repeats a bias here.

# %%
def march_points(step):
    """{tel: standard points} and {tel: {run: why not drawn}} of the March single runs at a
    fluence step."""
    keep, dropped = {}, {}
    for t in TELS:
        runs = [r for r in curves.table_runs(BT, t, "march", VARIANT, FLOOR)
                if r in curves.runs_at_step(RS, t, "march", step)]
        pts = curves.points(BT, BM, RS, t, runs, "march", VARIANT, FLOOR)
        keep[t] = curves.standard(pts)
        dropped[t] = {r: u"threshold offset %s" % u"/".join(
            str(o) for o in sorted({p["offset"] for p in pts if p["run"] == r}))
            for r in runs if not any(p["run"] == r for p in keep[t])}
    return keep, dropped


P01, DROP01 = march_points(0.0)
GOOD01 = check_good([p for t in TELS for p in P01[t]], "march")
footer01 = u"\n".join([
    SELECTION,
    u"Every March pre-irradiation run at RFSel 2 (the March RFSel), threshold offset 20: %s." % (
        counts_text([p for t in TELS for p in P01[t]])),
    u"H1 runs %s; F1 runs %s." % (bias_text(P01["h1"], "h1"), bias_text(P01["f1"], "f1")),
    u"Not drawn: %s." % u"; ".join(u"%s run %d (%s)" % (t.upper(), r, why) for t in TELS
                                    for r, why in sorted(DROP01[t].items()))] + low_line(
    [p for t in TELS for p in P01[t]]) + [
    u"Lines join each chip's points in bias order."])
fig, n01 = curves.res_vs_bias_march_preirrad(
    P01, tel_titles=TEL_TITLES, xlabel=style.XLABEL_BIAS, ylim=YLIM, tag=TAG,
    data=u"March, pre-irradiation", footer=footer01)
save(fig, "re01_res_vs_bias_march_preirrad",
     payload(n01, footer01, not_drawn={t: {str(r): w for r, w in DROP01[t].items()} for t in TELS},
             good_runs=GOOD01), conventions(colour="board (etroc_style.BOARD_COLOR): one step"))

# %% [markdown]
# ## Figure re02: the March bias scans after each irradiation step
#
# Every March run at the standard operating point after the three March steps: 3e14 (H1 and F1
# runs 10, 11), 9e14 (runs 12 to 15) and 1.5e15 (runs 16, 17, 19). Three steps share the axes,
# so the colour is the fluence step and the marker the chip; one line per chip and step.
#
# The runs of a step were taken in the order of the run numbers, over the hours after the step's
# irradiation: the bias and the time after the stop change together within a step, and a scan
# point is not a repeat at one condition. H1 run 19 ran IH7 at 465 V and the other chips at
# 470 V; each board is drawn at its own bias.

# %%
P02, DROP02 = {t: [] for t in TELS}, {t: {} for t in TELS}
for f in (3e14, 9e14, 1.5e15):
    k, d = march_points(f)
    for t in TELS:
        P02[t] += k[t]
        DROP02[t].update(d[t])
GOOD02 = check_good([p for t in TELS for p in P02[t]], "march")
STEPS02 = sorted({style.fluence_key(p["fluence"]) for t in TELS for p in P02[t]})
footer02 = u"\n".join([
    SELECTION,
    u"Every March run after each irradiation step at RFSel 2 (the March RFSel), threshold offset "
    u"20: %s." % counts_text([p for t in TELS for p in P02[t]]),
    u"H1 runs %s." % bias_text(P02["h1"], "h1"),
    u"F1 runs %s." % bias_text(P02["f1"], "f1"),
    (u"Not drawn: %s." % u"; ".join(u"%s run %d (%s)" % (t.upper(), r, why) for t in TELS
                                     for r, why in sorted(DROP02[t].items())))
    if any(DROP02.values()) else u"No run of these steps is at another operating point."] + (
    low_line([p for t in TELS for p in P02[t]])) + [
    FLUENCE_TEXT + u" Lines join each chip's points of one step in bias order."])
fig, n02 = curves.res_vs_bias_march_steps(
    P02, tel_titles=TEL_TITLES, xlabel=style.XLABEL_BIAS, ylim=YLIM, tag=TAG,
    data=u"March, after each irradiation step", footer=footer02)
save(fig, "re02_res_vs_bias_march_steps",
     payload(n02, footer02, steps=STEPS02, good_runs=GOOD02,
             not_drawn={t: {str(r): w for r, w in DROP02[t].items()} for t in TELS}),
     conventions(colour="fluence step (FLUENCE_COLOR)"))

# %% [markdown]
# ## Figure re03: 1.5e15 in March and in July
#
# The March scan at 1.5e15 (runs 16, 17, 19 on both telescopes, as in re02, in the step's
# colour) beside the July points at the same bookkeeping fluence, measured after four months of
# cooling down, in the later-look colour (teal; the March red stays with March). The July
# points are the deck's population, and this figure overrides the merge rule to draw it (the
# footer says so): the preferred display run of each telescope as a single run, H1 run 4 at
# 540 V (the rule would draw the merge 3+4 in its place, as re08 does) and F1 run 9 at 460 V.
#
# Two more July points are drawn with open markers: H1 run 10 at 550 V and F1 run 4 at 480 V.
# They are off the display-run list; they are the deck's second July point per telescope, kept
# because the display runs sit at one bias per telescope. To drop them, set `EXTRA03 = {}` in
# the cell below; to draw the merge rule's result instead, use `RESULT` in place of
# `curves.merge_override`.
#
# March and July differ in more than the time: the telescopes were reinstalled, and the July
# biases sit above every March bias. Read the gap between the scans as "four months later, at a
# higher bias", not as cooling down alone.

# %%
EXTRA03 = {"h1": [10], "f1": [4]}     # off the display list: the second July point per telescope
P03 = march_points(1.5e15)[0]
OVER03 = {}                           # the merges the override leaves out
for t in TELS:
    runs, preferred = CLASS[1.5e15]
    single, over = curves.merge_override([preferred[t]], MERGES[t], runs[t], GOOD[("july", t)])
    OVER03[t] = {curves.run_label(m): r for m, r in over.items()}
    P03[t] = P03[t] + curves.standard(curves.points(
        BT, BM, RS, t, single + EXTRA03.get(t, []), "july", VARIANT, FLOOR))
check_good([p for t in TELS for p in P03[t] if p["camp"] == "march"], "march")
GOOD03 = check_good([p for t in TELS for p in P03[t] if p["camp"] == "july"], "july")
off03 = {t: set(EXTRA03.get(t, [])) for t in TELS}
july03 = {t: [p for p in P03[t] if p["camp"] == "july"] for t in TELS}
footer03 = u"\n".join([
    SELECTION,
    u"March: every run at 1.5e15 at RFSel 2, threshold offset 20 (H1 runs %s; F1 runs %s)." % (
        bias_text([p for p in P03["h1"] if p["camp"] == "march"], "h1"),
        bias_text([p for p in P03["f1"] if p["camp"] == "march"], "f1")),
    u"July, after four months of cooling down: the preferred display run of the step as a single "
    u"run (H1 %s; F1 %s)." % (
        bias_text([p for p in july03["h1"] if not p["run"] in off03["h1"]], "h1"),
        bias_text([p for p in july03["f1"] if not p["run"] in off03["f1"]], "f1")),
    u"This figure overrides the merge rule and draws single runs: the rule would draw %s in "
    u"place of %s." % (
        u"; ".join(u"%s %s" % (t.upper(), m) for t in TELS for m in OVER03[t]) or u"no merge",
        u"; ".join(u"%s run %s" % (t.upper(), u", ".join(map(str, r)))
                   for t in TELS for r in OVER03[t].values()) or u"any run"),
    u"Open markers: July runs off the display-run list, kept as a second bias point (H1 %s; "
    u"F1 %s)." % (bias_text([p for p in july03["h1"] if p["run"] in off03["h1"]], "h1"),
                   bias_text([p for p in july03["f1"] if p["run"] in off03["f1"]], "f1"))] + (
    low_line([p for t in TELS for p in P03[t]])) + [
    u"%s. %s" % (counts_text([p for t in TELS for p in P03[t]]), FLUENCE_TEXT)])
fig, n03 = curves.res_1p5e15_march_vs_july(
    P03, off03, open_label=u"off the display list", tel_titles=TEL_TITLES,
    xlabel=style.XLABEL_BIAS, ylim=YLIM, tag=TAG,
    data=u"March and July, bookkeeping fluence %s" % step_text(1.5e15), footer=footer03)
save(fig, "re03_res_1p5e15_march_vs_july",
     payload(n03, footer03, off_list={t: sorted(v) for t, v in off03.items()},
             merge_rule={k: v for k, v in MERGE_EFFECT.items() if k.endswith("1.5e+15")},
             merge_override=OVER03, good_runs=GOOD03),
     conventions(colour="look: March 1.5e15 FLUENCE_COLOR, July the later-look hue "
                        "(iv_plot.timing_color)"))

# %% [markdown]
# ## Figure re04: July at 1.5e15, a stability check after the reinstall
#
# The July runs at 1.5e15 at the standard operating point, against the hours since the first
# run at that fluence (run 3 on both telescopes, from the run starts of `tables/run_times.csv`):
# H1 runs 3, 4, 5 at 540 V; F1 runs 3 and 4 at 480 V (open markers) and 5 to 9 at 460 V. This is the design the user chose for a stability check after the
# telescopes went back into the beam, four months after the March irradiation; it is not a
# cooling-down curve (no irradiation step precedes it).
#
# The population is wider than the display list: the display runs (H1 3, 4; F1 7, 8, 9) are
# bold in the run labels, the others are named in the footer. The time axis draws single runs,
# so H1 3+4 is drawn as runs 3 and 4. The other H1 runs at 1.5e15 are left out and named in the
# footer: they sit at other biases (runs 6 to 9 at 530 V with IH13 at 0 V, run 10 at 550 V, the
# open point of re03).
#
# On F1 the bias changes from 480 to 460 V between runs 4 and 5, so a step there mixes bias and
# time; the lines break at the change.

# %%
STABILITY = {"h1": [3, 4, 5], "f1": [3, 4, 5, 6, 7, 8, 9]}    # the user's design
P04, H04, FIRST04 = {}, {}, {}
for t in TELS:
    P04[t] = curves.standard(curves.points(BT, BM, RS, t, STABILITY[t], "july", VARIANT, FLOOR))
    if sorted({p["run"] for p in P04[t]}) != STABILITY[t]:
        raise RuntimeError("%s: a run of the stability set is not at the standard point" % t)
    H04[t], FIRST04[t] = curves.hours_since_first(RUN_TIMES, RS, t, STABILITY[t], 1.5e15)
GOOD04 = check_good([p for t in TELS for p in P04[t]], "july")
DISPLAY04 = {t: set(CLASS[1.5e15][0][t]) for t in TELS}
OFF04 = {t: [r for r in STABILITY[t] if r not in DISPLAY04[t]] for t in TELS}
HV04 = {t: {r: sorted({p["hv"] for p in P04[t] if p["run"] == r}) for r in STABILITY[t]}
        for t in TELS}
OPEN04 = {"f1": {r for r in STABILITY["f1"] if HV04["f1"][r] == [480.0]}}
others04 = {t: [r for r in curves.runs_at_step(RS, t, "july", 1.5e15) if r not in STABILITY[t]]
            for t in TELS}
print("July 1.5e15 runs not in the stability set:", others04)
# their biases, from runs_summary (bias_text reads tel, label, chip, board_idx, hv)
NOT04 = {t: [dict(tel=t, label=str(r), chip=style.chip_short(b["chip"]), board_idx=b["board_idx"],
                  hv=b["hv"])
             for r in others04[t]
             for b in tables.run_settings(RS, campaign.TABLE_CAMPAIGN["july"],
                                          campaign.TABLE_TEL[t], r)] for t in TELS}


def hv_groups(tel):
    """"runs 3, 4 at 480 V (open markers), runs 5, 6, 7, 8, 9 at 460 V" from HV04."""
    by = {}
    for r in STABILITY[tel]:
        by.setdefault(tuple(HV04[tel][r]), []).append(r)
    return u", ".join(u"run%s %s at %s%s" % (u"s" if len(rs) > 1 else u"", u", ".join(map(str, rs)),
                                           u"/".join(u"%d V" % v for v in hv),
                                           u" (open markers)" if set(rs) <= OPEN04.get(tel, set())
                                           else u"")
                      for hv, rs in by.items())


def bias_handle(tel, hv, open_marker):
    runs = [r for r in STABILITY[tel] if HV04[tel][r] == [hv]]
    return plt.Line2D([], [], color=style.INK, marker="o", linestyle="none",
                      markersize=8.5 * curves.SCALE,
                      markerfacecolor=style.SURFACE if open_marker else style.INK,
                      markeredgewidth=(1.8 if open_marker else 1.0) * curves.SCALE,
                      label=u"%d V (runs %s)" % (hv, u", ".join(map(str, runs))))


open_legend04 = {"f1": ([bias_handle("f1", 480.0, True), bias_handle("f1", 460.0, False)],
                        "upper right", u"bias")}
footer04 = u"\n".join([
    SELECTION,
    u"July runs at 1.5e15 at RFSel 2, threshold offset 20: H1 %s; F1 %s." % (hv_groups("h1"),
                                                                             hv_groups("f1")),
    u"x = hours from the start of the first run at 1.5e15 (run %d on both telescopes), from the "
    u"run starts; single runs on this time axis." % FIRST04["h1"],
    u"Under each run its number, bold for the display runs; off the display-run list: %s." % (
        u", ".join(run_list(t, OFF04[t]) for t in TELS if OFF04[t])),
    u"Not drawn, at other biases: %s." % u"; ".join(
        u"%s runs %s" % (t.upper(), bias_text(NOT04[t], t)) for t in TELS if NOT04[t])] + (
    low_line([p for t in TELS for p in P04[t]])) + [
    u"%s. %s" % (counts_text([p for t in TELS for p in P04[t]]), FLUENCE_TEXT)])
fig, n04 = curves.res_july_1p5e15_stability(
    P04, H04, open_runs=OPEN04, open_legend=open_legend04, tel_titles=TEL_TITLES,
    xlabel=u"Hours since the first run at %s" % step_text(1.5e15), ylim=YLIM, tag=TAG,
    data=u"July, bookkeeping fluence %s, stability check after the reinstall" % step_text(1.5e15),
    footer=footer04,
    run_labels={t: {r: u"%d" % r for r in STABILITY[t]} for t in TELS}, label_y=22.0,
    bold={(t, r) for t in TELS for r in STABILITY[t] if r in DISPLAY04[t]})
save(fig, "re04_res_july_1p5e15_stability",
     payload(n04, footer04, hours={t: {str(r): h for r, h in H04[t].items()} for t in TELS},
             first_run=FIRST04, off_list=OFF04, not_in_set=others04,
             not_in_set_bias={t: bias_text(NOT04[t], t) for t in TELS if NOT04[t]},
             good_runs=GOOD04),
     conventions(colour="board (etroc_style.BOARD_COLOR): one step",
                 times="run start (tables/run_times.csv) minus the start of the first run at "
                       "1.5e15, hours"))

# %% [markdown]
# ## Figure re05: cooling down after 2e15
#
# The 2e15 display runs at the standard operating point, as single runs (H1 11, 12; F1 10, 11,
# 12), against the hours after the end of the irradiation step. The merge 11+12, which replaces
# H1 run 12 on the result figures, is not drawn: a merge spans the times of its runs, and the
# axis is time.
#
# The clock: the records allow the 2e15 stop anywhere from 16:30 to 18:17:43 UTC on 2026-07-20
# (`RAD_STOP_BOUNDS_UTC`: an operator's "beam is stopped" message at about 17:00 UTC, minus 30
# min, and the return of the F1 LV, which comes back only once the telescopes leave the
# irradiation zone). The hours count from the earliest, 16:30 UTC (`RAD_STOP_UTC`); the shaded
# band at the origin spans the allowed range, 0 to +1.80 h, so each run's true time after the
# stop lies up to 1.80 h below its x.
#
# Under each run: its number and each chip's in-run bias current (the median over the run's HV
# plateau, `july/july_inrun_currents.csv`). The current falls from run to run as the sensors cool
# down; F1 ran every run at one bias per chip (the brace), H1 too (IH7 540 V, the others 550 V).
# The resolution can change with the time after the stop and with the current; these runs do
# not separate the two.

# %%
runs05, pref05 = CLASS[2e15]
P05, H05 = {}, {}
for t in TELS:
    P05[t] = curves.standard(curves.points(BT, BM, RS, t, runs05[t], "july", VARIANT, FLOOR))
    H05[t] = curves.hours_after_stop(RUN_TIMES, t, runs05[t], 2e15)
GOOD05 = check_good([p for t in TELS for p in P05[t]], "july")
STOP05 = curves.stop_band(2e15)
I05 = {t: {r: {p["chip"]: CURRENTS[(t, r, p["chip"])] for p in P05[t] if p["run"] == r}
           for r in runs05[t]} for t in TELS}
for t in TELS:
    for p in P05[t]:
        p["i_inrun_uA"] = I05[t][p["run"]][p["chip"]]
LABELS05 = {t: {r: u"run %d\n%s" % (r, u"\n".join(u"%s %.0f" % (c, i)
                                                   for c, i in I05[t][r].items()))
                for r in runs05[t]} for t in TELS}


def same_bias(t, runs):
    """The brace text when every chip keeps its bias over `runs`, else None."""
    per_chip = {}
    for p in P05[t]:
        if p["run"] in runs:
            per_chip.setdefault(p["chip"], set()).add(p["hv"])
    if any(len(v) != 1 for v in per_chip.values()):
        return None
    groups = {}
    for c, v in per_chip.items():
        groups.setdefault(v.pop(), []).append(c)
    return u"same bias in runs %s: %s" % (u", ".join(map(str, runs)), u", ".join(
        u"%s %d V" % (u", ".join(cs), hv) for hv, cs in sorted(groups.items(), reverse=True)))


BRACE05 = {"f1": (runs05["f1"], same_bias("f1", runs05["f1"]))}
footer05 = u"\n".join([
    SELECTION,
    u"The display runs at 2e15, RFSel 2, threshold offset 20, as single runs (a merge spans "
    u"the times of its runs): H1 %s; F1 %s." % (bias_text(P05["h1"], "h1"),
                                                bias_text(P05["f1"], "f1")),
    u"x = hours from %s UTC, the earliest end of the irradiation the records allow; shaded: "
    u"the allowed range of the end, 0 to +%.2f h (latest: the F1 LV back on)." % (
        campaign.RAD_STOP_UTC[2e15][11:16], STOP05[1]),
    u"Under each run: its number and each chip's in-run bias current in uA, the median over the "
    u"run's HV plateau. %s." % counts_text([p for t in TELS for p in P05[t]])] + low_line(
    [p for t in TELS for p in P05[t]]) + [
    FLUENCE_TEXT])
fig, n05 = curves.res_2e15_cooling_down(
    P05, H05, stop=STOP05, braces=BRACE05, tel_titles=TEL_TITLES,
    xlabel=u"Hours after the end of the %s irradiation" % step_text(2e15), ylim=YLIM, tag=TAG,
    data=u"July, bookkeeping fluence %s" % step_text(2e15), footer=footer05,
    run_labels=LABELS05, label_y=21.0, bold=set())
save(fig, "re05_res_2e15_cooling_down",
     payload(n05, footer05, hours={t: {str(r): h for r, h in H05[t].items()} for t in TELS},
             stop_utc=campaign.RAD_STOP_UTC[2e15],
             stop_bounds_utc=list(campaign.RAD_STOP_BOUNDS_UTC[2e15]),
             in_run_current_uA={t: {str(r): v for r, v in I05[t].items()} for t in TELS},
             merge_rule={k: v for k, v in MERGE_EFFECT.items() if k.endswith("2e+15")},
             good_runs=GOOD05),
     conventions(colour="board (etroc_style.BOARD_COLOR): one step",
                 times="run start (tables/run_times.csv) minus RAD_STOP_UTC[2e15] (the earliest "
                       "allowed stop), hours; the allowed range is RAD_STOP_BOUNDS_UTC",
                 in_run_current="july_inrun_currents.csv i_inrun_uA: median bias current over "
                                "the run's HV plateau"))

# %% [markdown]
# ## Figure re06: 3.5e15 against the bias and against the hours after the stop
#
# The July display runs at 3.5e15 at the standard operating point (RFSel 2, threshold offset 20):
# H1 run 18 (IH7 at 530 V, the other chips at 550 V); F1 runs 13 and 14 (LF17 at 370 V, the
# others at 390 V) and run 18 (LF17 at 450 V, the others at 480 V). One row per telescope: on the
# left against the bias, on the right against the hours after the end of the irradiation step.
# One fluence step, so the colour is the board and the step is named in the header. Every point
# is a band (see the top of this notebook).
#
# On the F1 hours panel two more runs are drawn with open markers: runs 50 and 51, the display
# class at threshold offset 15 (LF10 and LF17 at 450 V, LF12 and LF18 at 480 V), a week after
# the stop. They differ from run 18 in the offset and in each chip's bias at once, so the
# offset legend names them; they are not on the bias panel, which holds the offset-20 runs only.
#
# The clock: the records fix the 3.5e15 stop to between 15:49:26 and 15:51:26 UTC on 2026-07-25
# (`RAD_STOP_BOUNDS_UTC`); the hours count from 15:49:26 (`RAD_STOP_UTC`). The allowed range is
# 2 min, below the 0.25 h that earns a shaded band, so none is drawn.
#
# Read with care: on F1 the bias rises from 390/370 V (runs 13, 14) to 480/450 V (run 18) while
# the hours after the stop grow from 4 to 46, so the two panels show the same three runs from
# two sides; these runs do not separate the bias from the cooling down. Under each run on the
# hours panels: its number, and under each group of runs at one bias pattern that bias.
#
# re15 (appendix) draws the hours panels alone at the standard point: the same points and clock.

# %%
runs06, pref06 = CLASS[3.5e15]
runs15 = curves.display_class(campaign.DISPLAY_RUNS_JUL_CSV, 3.5e15, offset=15)[0]
B06, T06, H06, OS15 = {}, {}, {}, {}
for t in TELS:
    B06[t] = curves.standard(curves.points(BT, BM, RS, t, runs06[t], "july", VARIANT, FLOOR))
    if sorted({p["run"] for p in B06[t]}) != sorted(runs06[t]):
        raise RuntimeError("%s: a 3.5e15 display run is not at the standard point" % t)
    OS15[t] = list(runs15.get(t, []))
    p15 = curves.points(BT, BM, RS, t, OS15[t], "july", VARIANT, FLOOR)
    if any(p["offset"] != 15 or p["rfsel"] != 2 for p in p15):
        raise RuntimeError("%s: a run of the offset-15 class is not at RFSel 2, offset 15" % t)
    T06[t] = B06[t] + p15
    H06[t] = curves.hours_after_stop(RUN_TIMES, t, list(runs06[t]) + OS15[t], 3.5e15)
GOOD06 = check_good([p for t in TELS for p in T06[t]], "july")
STOP06 = curves.stop_band(3.5e15)        # None: the allowed range is 2 min


def bias_groups(pts, hours):
    """[(runs, bias text)]: the runs of `pts` in time order, consecutive runs at one bias and
    offset on every board together, with blnw.bias_text of their biases."""
    out = []
    for r in sorted({p["run"] for p in pts}, key=lambda r: hours[r]):
        pattern = tuple((p["board_idx"], p["hv"], p["offset"])
                        for p in sorted(pts, key=lambda p: p["board_idx"]) if p["run"] == r)
        if out and out[-1][0] == pattern:
            out[-1][1].append(r)
        else:
            out.append((pattern, [r]))
    return [(runs, blnw.bias_text([hv for _, hv, _ in pattern])) for pattern, runs in out]


GROUPS06 = {t: bias_groups(T06[t], H06[t]) for t in TELS}
open_legend06 = {t: ([plt.Line2D([], [], color=style.INK, marker="o", linestyle="none",
                                 markersize=8.5 * curves.SCALE, markerfacecolor=style.INK,
                                 label=u"20 (runs %s)" % u", ".join(map(str, runs06[t]))),
                      plt.Line2D([], [], color=style.INK, marker="o", linestyle="none",
                                 markersize=8.5 * curves.SCALE, markerfacecolor=style.SURFACE,
                                 markeredgewidth=1.8 * curves.SCALE,
                                 label=u"15 (runs %s)" % u", ".join(map(str, OS15[t])))],
                     "upper center", u"threshold offset") for t in TELS if OS15[t]}
LOCS06 = {("h1", "bias"): "lower center", ("f1", "bias"): "lower right",
          ("h1", "hours"): "upper left", ("f1", "hours"): ("upper right", 2)}
merge06 = {k: v for k, v in MERGE_EFFECT.items() if k.endswith("3.5e+15")}
footer06 = u"\n".join([
    SELECTION,
    u"Every point a band: %s." % tables.BAND_FOOTER,
    u"The display runs at 3.5e15, RFSel 2, threshold offset 20: H1 %s; F1 %s. No merge applies "
    u"at this step (merge rule)." % (bias_text(B06["h1"], "h1"), bias_text(B06["f1"], "f1")),
    u"Open markers, hours panel only: the display class at threshold offset 15, %s." % (
        u"; ".join(u"%s runs %s" % (t.upper(), bias_text([p for p in T06[t] if p["offset"] == 15],
                                                          t))
                   for t in TELS if OS15[t])),
    u"x (right) = hours from %s UTC, the earliest end of the irradiation the records allow (the "
    u"latest is 2 min later); under each run its number, under each group its bias." % (
        campaign.RAD_STOP_UTC[3.5e15][11:19])] + low_line(
    [p for t in TELS for p in B06[t] + T06[t]]) + [
    u"%s. %s" % (counts_text([p for t in TELS for p in T06[t]]), FLUENCE_TEXT)])
fig, n06 = curves.res_3p5e15_vs_bias_and_hours(
    B06, T06, H06, tel_titles=TEL_TITLES, ylim=YLIM,
    tag=style.settings_text(rfsel=2, offset=[20, 15]),
    data=u"July, bookkeeping fluence %s" % step_text(3.5e15), footer=footer06,
    xlabel_bias=style.XLABEL_BIAS,
    xlabel_hours=u"Hours after the end of the %s irradiation" % step_text(3.5e15),
    run_labels={t: {r: u"%d" % r for r in H06[t]} for t in TELS}, label_y=27.0,
    groups=GROUPS06, group_y=21.0, stop=STOP06, open_runs={t: set(OS15[t]) for t in TELS},
    open_legend=open_legend06, legend_locs=LOCS06)
save(fig, "re06_res_3p5e15_vs_bias_and_hours",
     payload(n06, footer06, hours={t: {str(r): h for r, h in H06[t].items()} for t in TELS},
             stop_utc=campaign.RAD_STOP_UTC[3.5e15],
             stop_bounds_utc=list(campaign.RAD_STOP_BOUNDS_UTC[3.5e15]),
             offset15_runs=OS15, merge_rule=merge06, good_runs=GOOD06),
     conventions(colour="board (etroc_style.BOARD_COLOR): one step",
                 band=tables.BAND_FOOTER,
                 times="run start (tables/run_times.csv) minus RAD_STOP_UTC[3.5e15] (the "
                       "earliest allowed stop), hours; the allowed range is RAD_STOP_BOUNDS_UTC"))

# %% [markdown]
# ## Figure re07: every bias scan, one figure per telescope
#
# The summary of re01 to re06 against the bias: every March run at the standard operating point
# at every step (the populations of re01 and re02, the March scan at 1.5e15 of re03), and every
# July display run at RFSel 2, threshold offset 20, each as a single run
# (`curves.display_singles`): H1 runs 3, 4 (1.5e15), 11, 12 (2e15), 18 (3.5e15); F1 runs 7 to
# 14 and 18. This is the deck's population, and the figure overrides the merge rule to draw it
# (the footer says so): no merge replaces a run here (the rule would draw H1 3+4 and 11+12, as
# re08 does), and no run is left out where two runs sit at one bias on every board (F1 13 and
# 14 at 3.5e15): such runs stack at one bias. `curves.one_per_condition` picks one run per
# condition for a figure that wants that.
#
# The colour is the fluence step, with July 1.5e15 in the later-look colour (teal) beside the
# March red; the marker is the chip. The legend names the bookkeeping fluence; H1 run 11 sits
# under 128 pixels on four chips (small open markers, counts in the footer). The option `CHIP_FLUENCE07 = True` writes the fluence on the chips
# (`CHIP_FLUENCE_FACTOR` times the bookkeeping fluence) in the legend instead; the figure is the
# same otherwise.
#
# The July points sit at higher bias than any March point of the same telescope, four months
# later: the gap between March and July at 1.5e15 mixes the bias, the cooling down and the
# reinstall (re03). re12 to re14 (appendix) draw parts of this figure with its colours.

# %%
CHIP_FLUENCE07 = False          # True: the legend gives the fluence on the chips
P07, DROP07M = {t: [] for t in TELS}, {t: {} for t in TELS}
for f in (0.0, 3e14, 9e14, 1.5e15):
    k, d = march_points(f)
    for t in TELS:
        P07[t] += k[t]
        DROP07M[t].update(d[t])
OVER07, DRAWN07 = {t: {} for t in TELS}, {}
for t in TELS:
    DRAWN07[t] = curves.display_singles(CLASS, t)
    for f in STEPS_JUL:
        runs = CLASS[f][0]
        single, over = curves.merge_override([r for r in DRAWN07[t] if r in runs[t]], MERGES[t],
                                             runs[t], GOOD[("july", t)])
        pts = curves.points(BT, BM, RS, t, single, "july", VARIANT, FLOOR)
        if curves.standard(pts) != pts:
            raise RuntimeError("%s %g: a July run drawn is not at the standard point" % (t, f))
        P07[t] += pts
        OVER07[t].update({curves.run_label(m): r for m, r in over.items()})
check_good([p for t in TELS for p in P07[t] if p["camp"] == "march"], "march")
GOOD07 = check_good([p for t in TELS for p in P07[t] if p["camp"] == "july"], "july")
july07 = [p for t in TELS for p in P07[t] if p["camp"] == "july"]

footer07 = u"\n".join([
    SELECTION,
    u"March: every run at RFSel 2 (the March RFSel), threshold offset 20, at every step; not "
    u"drawn: %s." % u"; ".join(u"%s run %d (%s)" % (t.upper(), r, why) for t in TELS
                                for r, why in sorted(DROP07M[t].items())),
    u"July: every display run at RFSel 2, offset 20, each a single run: H1 %s; F1 %s." % (
        bias_text([p for p in P07["h1"] if p["camp"] == "july"], "h1"),
        bias_text([p for p in P07["f1"] if p["camp"] == "july"], "f1")),
    u"This figure overrides the merge rule and draws single runs, two runs at one bias stacked: "
    u"the rule would draw %s in place of their runs." % u", ".join(
        u"%s %s" % (t.upper(), m) for t in TELS for m in OVER07[t]),
    u"At 3.5e15 every point is a band: %s." % tables.BAND_FOOTER] + low_line(july07) + [
    u"%s. %s" % (counts_text([p for t in TELS for p in P07[t]]), FLUENCE_TEXT)])
fig, n07 = curves.res_vs_bias_summary(
    P07, chip_fluence=CHIP_FLUENCE07, tel_titles=TEL_TITLES, xlabel=style.XLABEL_BIAS,
    ylim=YLIM, tag=TAG, data=u"March and July, every step", footer=footer07)
save(fig, "re07_res_vs_bias_summary",
     payload(n07, footer07, july_drawn=DRAWN07, merge_override=OVER07,
             not_drawn_march={t: {str(r): w for r, w in DROP07M[t].items()} for t in TELS},
             chip_fluence_legend=CHIP_FLUENCE07, good_runs=GOOD07),
     conventions(colour="fluence step (FLUENCE_COLOR), a later look at a step in the later-look "
                        "hue (iv_plot.timing_color)",
                 july_population="every display run of every step as a single run "
                                 "(curves.display_singles); the merge rule overridden "
                                 "(curves.merge_override)",
                 band=tables.BAND_FOOTER))

# %% [markdown]
# ## Figure re08: the fluence ladder
#
# Each board's resolution at each fluence step, one point per step: the run at the highest bias
# the step reached, where the resolution is best. March: the highest-bias run of each step at
# the standard operating point (`MARCH_DISPLAY_RUNS`: H1 runs 7, 11, 14, 19; F1 runs 2, 11, 14,
# 19), checked against the tables by `ladder.step_winners`. July: the preferred display run of
# each step at RFSel 2, threshold offset 20, replaced by its merge under the merge rule: H1 3+4
# (1.5e15), 11+12 (2e15), 18 (3.5e15); F1 9, 12, 18.
#
# The x axis is the bookkeeping fluence, linear. Both looks at 1.5e15 are drawn: March, and July
# after four months of cooling down, March 0.06e15 left of the step and July 0.06e15 right of
# it so the two do not cover each other. The colour is the board, the marker the chip; the line
# style is the campaign (March solid, July dashed), and lines join a board's steps within one
# campaign only. At 3.5e15 the points are bands.
#
# The highest bias rose from step to step, so this is not the resolution at one bias: each point
# carries its own bias (re09 lists them).
#
# The maps notebook draws H1 at 1.5e15 and 2e15 from the single runs 4 and 12, the preferred
# display runs (its ladder and March-against-July maps, maps15 to maps18), where this ladder and
# re10 draw the merges 3+4 and 11+12 by the merge rule; the values of a board differ between the
# two notebooks at those two steps for that reason.

# %%
CLASS08 = {f: CLASS[f] for f in STEPS_JUL}
PREF08 = tables.preferred_runs(campaign.DISPLAY_RUNS_JUL_CSV, rfsel=2, offset=20)
P08, PICKS08 = {}, {}
for t in TELS:
    if [CLASS08[f][1][t] for f in sorted(CLASS08)] != PREF08[t]:
        raise RuntimeError("%s: the preferred runs of the display classes %s differ from the "
                           "list's %s" % (t, [CLASS08[f][1][t] for f in sorted(CLASS08)],
                                          PREF08[t]))
    P08[t], PICKS08[t] = ladder.step_winners(BT, BM, RS, t, CLASS08, GOOD[("july", t)],
                                             MERGES[t], variant=VARIANT, floor=FLOOR)
check_good([p for t in TELS for p in P08[t] if p["camp"] == "march"], "march")
GOOD08 = check_good([p for t in TELS for p in P08[t] if p["camp"] == "july"], "july")


def pick_text(t, camp):
    """"7 (190 V), 11 (280 V)" for the picks of `t` in `camp`, merges marked."""
    pts = [p for p in P08[t] if p["camp"] == camp]
    out = []
    for k in PICKS08[t]:
        if k["camp"] != camp:
            continue
        text = bias_text([p for p in pts if p["label"] == k["run"]], t)
        if k.get("taken"):
            text = text[:-1] + u", replacing %d)" % k["preferred"]
        out.append(text)
    return u", ".join(out)


footer08 = u"\n".join([
    SELECTION,
    u"March: the highest-bias run of each step at RFSel 2, threshold offset 20; H1 %s; F1 %s." % (
        pick_text("h1", "march"), pick_text("f1", "march")),
    u"July: the preferred display run of each step at RFSel 2, offset 20, merges by the merge "
    u"rule (thick black edge); H1 %s;" % pick_text("h1", "july"),
    u"F1 %s." % pick_text("f1", "july"),
    u"Both looks at 1.5e15: March %.2fe15 left of the step, July %.2fe15 right of it; lines join "
    u"a board's steps within one campaign." % (ladder.DODGE, ladder.DODGE),
    u"At 3.5e15 every point is a band: %s." % tables.BAND_FOOTER] + low_line(
    [p for t in TELS for p in P08[t]]) + [
    u"%s. %s" % (counts_text([p for t in TELS for p in P08[t]]), FLUENCE_TEXT)])
fig, n08 = ladder.best_res_vs_fluence(
    P08, tel_titles=TEL_TITLES, ylim=YLIM, tag=TAG,
    data=u"March and July, highest bias per step", footer=footer08)
save(fig, "re08_best_res_vs_fluence",
     payload(n08, footer08, picks={t: PICKS08[t] for t in TELS}, good_runs=GOOD08),
     conventions(colour="board (etroc_style.BOARD_COLOR): one point per board and step",
                 line_style="campaign: March solid, July dashed (ladder.LOOK_LINESTYLE)",
                 x="bookkeeping fluence in 1e15 p/cm2; the two looks at 1.5e15 dodged by "
                   "+/- %g" % ladder.DODGE,
                 band=tables.BAND_FOOTER))

# %% [markdown]
# ## Figure re09: the numbers table
#
# The numbers behind re08, one row per telescope, campaign, step and board, written as
# `re09_numbers_table.csv` (every digit) and `re09_numbers_table.md` (0.1 ps) beside the table
# figure. Per row: the chip, the run (a merge as "3+4"), its bias, offset and RFSel, the pixels
# behind the point, the value drawn on the ladder with its pixel-to-pixel spread ("top": the
# most-populated track per pixel), the same board and run with the central track per pixel
# ("central"), and at 3.5e15 the band's ends and the anointed combination's own value. At 3.5e15
# both "top" and "central" are band centres, each of its own track choice, and the anointed
# value is given for both choices. The table shown in September put the central anointed value
# in its central column at 3.5e15; here that number is the column "central anointed value". The
# column "preferred run" says, for July, whether the run is the step's preferred display run
# or the merge holding it.
#
# The table shown in September had 48 of these 56 rows: the eight H1 cells of July 1.5e15 and
# 2e15 were missing, because the merges were applied there before the display-run filter, which
# then found neither the merge nor its single runs. Here those cells hold the merges 3+4 and
# 11+12, as on re08.

# %%
ALL08 = [p for t in TELS for p in P08[t]]
CENTRAL09 = numbers_table.central_values(BT, BM, ALL08, FLOOR)
T09 = numbers_table.numbers_table(ALL08, CENTRAL09)
TABLE09 = numbers_table.write_table(T09, OUT, "re09_numbers_table")
print("table: %d rows -> %s" % (len(T09), ", ".join(os.path.basename(p) for p in TABLE09)))
footer09 = u"\n".join([
    u"Per cell: top = the value on the fluence ladder, %s, with its pixel-to-pixel spread; "
    u"central = the same with the central track per pixel." % tables.VARIANT_TEXT[VARIANT],
    u"Each chip from its anointed combination (boards 0-1-2 for boards 0-2, boards 1-2-3 for "
    u"board 3), tracks with at least %s events, widths = FWHM/2.355 of the TOA-difference core." % (
        FLOOR),
    u"Runs: March, the highest-bias run of each step at RFSel 2, threshold offset 20; July, the "
    u"preferred display run, merges by the merge rule (merged: the merge of the runs named).",
    u"At 3.5e15: top and central are band centres; band = the lowest to the highest value of the "
    u"board over every combination containing it; anointed = the anointed combination's value.",
    u"At 3.5e15 the anointed combination's value is given for both track choices: anointed "
    u"(most-populated track) and central anointed (central track).",
    FLUENCE_TEXT])
fig, n09 = numbers_table.table_figure(
    T09, name=u"Resolution per chip and step", tag=TAG,
    data=u"March and July, highest bias per step", footer=footer09)
save(fig, "re09_numbers_table",
     payload(dict(rows=numbers_table.records(T09), cells=n09["cells"], columns=n09["columns"],
                  table_rows=n09["rows"], files=[os.path.basename(p) for p in TABLE09]),
             footer09),
     conventions(top="the value drawn on re08: the anointed combination's value, or the band "
                     "centre at a step of BAND_STEPS",
                 central="variant central at the same floor, anointed combination; the band "
                         "centre at a step of BAND_STEPS",
                 band=tables.BAND_FOOTER))

# %% [markdown]
# ## Figure re10: the pixels behind the ladder
#
# For each re08 point, the pixels of the board with a resolution: with tracks of at least 300
# events (solid lines and the chip markers: the pixels behind the ladder's values) and with
# tracks of any number of events (dashed lines, the same runs at event floor "none"). A pixel
# counts once it has one track through the anointed combination. The gap between the two lines
# is the pixels the 300-event floor removes; where the solid line falls, the ladder's values come
# from fewer pixels.
#
# The small open markers at 3.5e15 are the other F1 display runs at that step, 13 and 14 (LF17
# at 370 V, the others at 390 V), at least 300 events: the display runs at a lower bias than
# run 18, drawn off the lines. A display run at the same bias as the ladder's run adds no
# marker.
#
# The x axis is as on re08: bookkeeping fluence, both looks at 1.5e15 dodged.

# %%
TC_JUL = campaign.TABLE_CAMPAIGN["july"]


def bias_pattern(t, run):
    return tuple((b["board_idx"], b["hv"]) for b in tables.run_settings(
        RS, TC_JUL, campaign.TABLE_TEL[t], run))


EXTRA10 = {t: [] for t in TELS}
for f in STEPS_JUL:
    runs, _ = CLASS[f]
    for t in TELS:
        pick = [p for p in P08[t] if p["camp"] == "july" and p["step"] == f]
        own = bias_pattern(t, pick[0]["runs"][0])
        others = [r for r in runs[t] if r not in pick[0]["runs"] and bias_pattern(t, r) != own]
        extra = curves.standard(curves.points(BT, BM, RS, t, others, "july", VARIANT, FLOOR))
        EXTRA10[t] += [dict(p, step=style.fluence_key(p["fluence"])) for p in extra]
check_good([p for t in TELS for p in EXTRA10[t]], "july")
ANY10 = ladder.floor_none_pixels(BT, BM, ALL08 + [p for t in TELS for p in EXTRA10[t]],
                                 variant=VARIANT)
extra_runs10 = {t: sorted({p["run"] for p in EXTRA10[t]}) for t in TELS}
footer10 = u"\n".join([
    u"Pixels with a resolution per board, anointed combination, %s: solid = tracks with at "
    u"least %s events (the points of the fluence ladder), dashed = tracks with any events." % (
        tables.VARIANT_TEXT[VARIANT], FLOOR),
    u"Runs as on the fluence ladder: March, the highest-bias run of each step at RFSel 2, "
    u"threshold offset 20; July, the preferred display run, merges by the merge rule.",
    u"Small open markers: %s at 3.5e15, display runs at a lower bias than the ladder's run, at "
    u"least %s events." % (u"; ".join(u"%s runs %s" % (t.upper(), bias_text(EXTRA10[t], t))
                                      for t in TELS if EXTRA10[t]), FLOOR),
    u"Both looks at 1.5e15: March %.2fe15 left of the step, July %.2fe15 right of it. %s" % (
        ladder.DODGE, ladder.DODGE, FLUENCE_TEXT)])
fig, n10 = ladder.pixel_yield_vs_fluence(
    P08, ANY10, EXTRA10, tel_titles=TEL_TITLES, ylim=(0.0, 280.0), tag=TAG,
    data=u"March and July, highest bias per step", footer=footer10,
    floor_label=u"at least %s events" % FLOOR, any_label=u"any events",
    extra_label=u"other display runs")
save(fig, "re10_pixel_yield_vs_fluence",
     payload(n10, footer10, floors=dict(solid=FLOOR, dashed="none"),
             extra_runs={t: extra_runs10[t] for t in TELS}),
     conventions(colour="board (etroc_style.BOARD_COLOR)",
                 line_style="event floor: solid at least %s events, dashed any events" % FLOOR,
                 n_pixels_any="n_pixels of the anointed row at floor 'none', same variant and "
                              "table; None where the table has no row"))

# %% [markdown]
# ## Figure re11: the combination split at 3.5e15
#
# Each board's resolution in every 3-board combination containing it (three per board), run by
# run: the spread between the combinations is what the 3.5e15 bands of re06 to re08 span. One
# row per telescope, one panel per board. The runs (`combo_split.split_runs`): first the
# references, the preferred display run of 1.5e15 and of 2e15 at RFSel 2, offset 20 (H1 runs 4
# and 12, F1 runs 9 and 12), then every display run at 3.5e15, settings class by settings class
# (the display list's order), by run number within a class. H1 run 19 is not on the display list
# at 3.5e15 and is not drawn.
#
# The merge rule of the settings figures applies: an accepted merge of the merged board table
# (`tables.accepted_merges`) whose runs are all in one settings class at 3.5e15 stands in for
# them, its combinations read from the merged table. So F1 46+47 (RFSel 1, offset 20) is drawn
# as one merge, as on the settings notebook's st02, st04 and st05, and runs 46 and 47 are not
# drawn apart. No other accepted merge has all its runs in one class at 3.5e15 (H1 18+19: run 19
# is not on the display list there). The references are the preferred runs alone, H1 4 and 12 as
# on the maps notebook, so H1 3+4 and 11+12 are not drawn.
#
# The colour is the combination (by its three boards; no fluence hue: the steps are on the
# axis), filled for the board's anointed combination and open for the other two, the marker the
# chip, the bar the pixel-to-pixel spread of the board's map in that combination. No band: the
# figure draws the combinations themselves. Under each panel: each run's number and this
# board's bias, then the settings class, then the step. A point from fewer than 128 pixels is a
# small marker, filled or open by the same rule (many 3.5e15 points of H1 and some of F1 run 86;
# the footer counts them).

# %%
YLIM11 = (20.0, 120.0)          # ps: H1 run 56 (offset 8/10) reaches past 100 with its bar
ACCEPTED11 = tables.accepted_merges(BM, campaign.TABLE_CAMPAIGN["july"])
SPLIT11 = combo_split.split_runs(campaign.DISPLAY_RUNS_JUL_CSV, (1.5e15, 2e15), 3.5e15,
                                 merges=ACCEPTED11)
P11, LAYOUT11 = {}, {}
for t in TELS:
    runs11 = [e["run"] for e in SPLIT11[t]]
    if len(set(runs11)) != len(runs11):
        raise RuntimeError("%s: a run twice on the combination split: %s" % (t, runs11))
    bad11 = curves.not_good(runs11, GOOD[("july", t)])
    if bad11:
        raise RuntimeError("%s runs on the combination split that are not good: %s" % (t, bad11))
    P11[t] = combo_split.combo_points(BT, RS, t, SPLIT11[t], "july", VARIANT, FLOOR, bm=BM)
    LAYOUT11[t] = combo_split.layout(SPLIT11[t])
GOOD11 = {t: u"every run good" for t in TELS}
MERGED11 = {t: [e["run"] for e in SPLIT11[t] if isinstance(e["run"], tuple)] for t in TELS}
out11 = [p for t in TELS for p in P11[t]
         if not (YLIM11[0] <= p["value_ps"] - p["bar_ps"] and p["value_ps"] + p["bar_ps"] <= YLIM11[1])]
if out11:
    raise RuntimeError("combination split points outside YLIM11: %s" % [
        (p["tel"], p["run"], p["chip"], p["combo"], p["value_ps"]) for p in out11])


def split_text(t):
    """"references 4 (1.5e15), 12 (2e15); at 3.5e15: RFSel 3, offset 6/8: 64, 67; ..." for one
    telescope, the fluences in mathtext."""
    ref = [u"%d (%s)" % (e["run"], style.fluence_label(e["step"], unit=False))
           for e in SPLIT11[t] if e["ref"]]
    at = style.fluence_label(3.5e15, unit=False)
    parts, last = [], None
    for e in SPLIT11[t]:
        if e["ref"]:
            continue
        key = (e["rfsel"], e["os"])
        if key != last:
            parts.append([u"RFSel %d, offset %s:" % key, []])
            last = key
        parts[-1][1].append(curves.run_label(e["run"]))
    return u"references %s; at %s: %s" % (u", ".join(ref), at, u"; ".join(
        u"%s %s" % (k, u", ".join(rs)) for k, rs in parts))


def low_line11(pts):
    """The footer line of the points under LOW_PIXELS, per telescope: points, then runs."""
    low = [p for p in pts if curves.low_pixels(p)]
    if not low:
        return []
    by = {}
    for p in low:     # a run as a tuple of its runs: single runs and merges sort together
        run = p["run"] if isinstance(p["run"], tuple) else (p["run"],)
        by.setdefault(p["tel"], set()).add(run)
    return [u"Small markers, under %d pixels (half the chip): %d of %d points (%d anointed); %s." % (
        curves.LOW_PIXELS, len(low), len(pts), sum(p["anointed"] for p in low), u"; ".join(
            u"%s run%s %s" % (t.upper(), u"s" if len(by[t]) > 1 else u"",
                              u", ".join(map(curves.run_label, sorted(by[t]))))
            for t in TELS if t in by))]


all11 = [p for t in TELS for p in P11[t]]
footer11 = u"\n".join([
    u"Selection: %s, tracks with at least %s events; every 3-board combination containing the "
    u"board, filled = the board's anointed combination, open = the other two; no band." % (
        tables.VARIANT_TEXT[VARIANT], FLOOR),
    u"Widths = FWHM/2.355 of the TOA-difference core, as the result tables carry them; bar = the "
    u"pixel-to-pixel spread of the board's map in that combination.",
    u"H1: %s." % split_text("h1"),
    u"F1: %s." % split_text("f1"),
    u"An accepted merge whose runs are all in one settings class stands in for them, its "
    u"combinations from the merged table (the settings figures' rule): %s. Under each panel: run "
    u"and this board's bias, then the settings class, then the step." % u"; ".join(
        u"%s %s" % (t.upper(), u", ".join(map(curves.run_label, MERGED11[t])))
        for t in TELS if MERGED11[t])] + low_line11(
    all11) + [
    u"%d points, %d runs. %s" % (len(all11), sum(len(SPLIT11[t]) for t in TELS), FLUENCE_TEXT)])
fig, n11 = combo_split.combo_split(
    P11, LAYOUT11, tel_titles=TEL_TITLES,
    class_text=lambda k: u"offset %s\nRFSel %d" % (k[2], k[1]),
    step_text=lambda f: u"%s\n%s" % (style.fluence_label(f, unit=False), campaign.FLUENCE_UNIT),
    ylim=YLIM11,
    tag=style.settings_text(rfsel=(1, 2, 3), offset=None),
    data=u"July, bookkeeping fluence %s, references at %s and %s" % (
        step_text(3.5e15), style.fluence_label(1.5e15, unit=False),
        style.fluence_label(2e15, unit=False)),
    footer=footer11, low_label=u"small: under %d pixels" % curves.LOW_PIXELS)
save(fig, "re11_combo_split_3p5e15",
     payload(n11, footer11, runs={t: SPLIT11[t] for t in TELS}, good_runs=GOOD11,
             merges={t: [curves.run_label(m) for m in MERGED11[t]] for t in TELS}),
     conventions(colour="combination (combo_split.COMBO_COLOR), by its three boards",
                 marker="chip (style.chip_marker); filled = anointed combination, open = the "
                        "others; small under curves.LOW_PIXELS pixels",
                 bar="pixel-to-pixel spread of the board's map in that combination",
                 band="none: every combination drawn",
                 population="references = the preferred display runs at 1.5e15 and 2e15 (RFSel "
                            "2, offset 20), then every display run at 3.5e15; an accepted merge "
                            "(tables.accepted_merges) whose runs are all in one settings class "
                            "stands in for them"))

# %% [markdown]
# # Appendix: figures of the July talk drawn from the points above
#
# Each appendix figure draws the points of the figure it derives from, by that figure's rules
# (population, merge rule, colours, bands, the under-128 rule), and estimates nothing new.

# %% [markdown]
# ## Figures re12, re13 and re14: re07 at fewer steps
#
# re07's points restricted to some of its looks, with re07's colours (July 1.5e15 in the
# later-look colour, teal, as beside the March red on re07): re12 the July points at 1.5e15 and
# 2e15, re13 the July points at 1.5e15, 2e15 and 3.5e15 (the 3.5e15 points bands), both on
# re07's bias axes; re14 the March points at every step, on its own axes. re07's population rule
# holds: every July display run at RFSel 2, offset 20 as a single run, runs at one bias stacked,
# the merge rule overridden as re07's footer says (`curves.res_vs_bias_summary` with `keep`).


# %%
def subset_footer(pts_by_tel, what):
    """The footer of a figure drawing part of re07's points (re12 to re14)."""
    pts = [p for t in TELS for p in pts_by_tel[t]]
    july = [p for p in pts if p["camp"] == "july"]
    lines = [SELECTION, u"re07's %s, with its population rule and colours." % what]
    if july:
        lines.append(u"July: every display run at RFSel 2, offset 20, each a single run: H1 %s; "
                     u"F1 %s." % (bias_text([p for p in july if p["tel"] == "h1"], "h1"),
                                  bias_text([p for p in july if p["tel"] == "f1"], "f1")))
        over = [u"%s %s" % (t.upper(), m) for t in TELS for m, r in OVER07[t].items()
                if set(r) <= {p["runs"][0] for p in july if p["tel"] == t}]
        lines.append(u"As re07, this figure overrides the merge rule and draws single runs, two "
                     u"runs at one bias stacked%s." % (
                         u": the rule would draw %s in place of their runs" % u", ".join(over)
                         if over else u""))
    if any(p["band"] for p in pts):
        lines.append(u"At 3.5e15 every point is a band: %s." % tables.BAND_FOOTER)
    if len(july) < len(pts):
        lines.append(u"March: every run at RFSel 2 (the March RFSel), threshold offset 20, at "
                     u"every step; not drawn: %s." % u"; ".join(
                         u"%s run %d (%s)" % (t.upper(), r, why) for t in TELS
                         for r, why in sorted(DROP07M[t].items())))
    return u"\n".join(lines + low_line(pts) + [
        u"%s. %s" % (counts_text(pts), FLUENCE_TEXT)])


def keep_looks(camp, steps):
    """The points of re07 in campaign `camp` at the fluence steps `steps`."""
    return lambda p: p["camp"] == camp and style.fluence_key(p["fluence"]) in steps


SUBSETS = [
    ("re12_res_vs_bias_july_1p5_2e15", "july", (1.5e15, 2e15), u"July points at 1.5e15 and 2e15",
     u"July, bookkeeping fluence %s and %s" % (style.fluence_label(1.5e15, unit=False),
                                               step_text(2e15)), n07["xlim"]),
    ("re13_res_vs_bias_july_1p5_2_3p5e15", "july", (1.5e15, 2e15, 3.5e15),
     u"July points at 1.5e15, 2e15 and 3.5e15",
     u"July, bookkeeping fluence %s, %s and %s" % (style.fluence_label(1.5e15, unit=False),
                                                   style.fluence_label(2e15, unit=False),
                                                   step_text(3.5e15)), n07["xlim"]),
    ("re14_res_vs_bias_march", "march", (0.0, 3e14, 9e14, 1.5e15),
     u"March points at every step", u"March, every step", None),
]
N_SUB = {}
for stem, camp, steps, what, data, xlims in SUBSETS:
    keep = keep_looks(camp, steps)
    kept = {t: [p for p in P07[t] if keep(p)] for t in TELS}
    footer = subset_footer(kept, what)
    fig, N_SUB[stem] = curves.res_vs_bias_summary(
        P07, keep=keep, chip_fluence=CHIP_FLUENCE07, tel_titles=TEL_TITLES,
        xlabel=style.XLABEL_BIAS, ylim=YLIM, tag=TAG, data=data, footer=footer, xlims=xlims)
    save(fig, stem,
         payload(N_SUB[stem], footer, derived_from="re07_res_vs_bias_summary",
                 kept=dict(campaign=camp, steps=list(steps)),
                 axes="re07's x ranges" if xlims else "its own x ranges",
                 chip_fluence_legend=CHIP_FLUENCE07, good_runs=GOOD07),
         conventions(colour="fluence step (FLUENCE_COLOR), a later look at a step in the "
                            "later-look hue (iv_plot.timing_color), as on re07",
                     population="re07's points (every March standard run, every July display "
                                "run as a single run), kept by campaign and step",
                     band=tables.BAND_FOOTER))

# %% [markdown]
# ## Figure re15: 3.5e15 against the hours after the stop
#
# re06's hours panel at the standard operating point alone: the July display runs at 3.5e15 at
# RFSel 2, threshold offset 20 (H1 run 18, F1 runs 13, 14 and 18) against the hours after the
# end of the irradiation step, one panel per telescope, colour = board (each board one chip),
# every point a band. Its points and its clock are re06's: the hours count from the earliest end
# of the irradiation the records allow (`RAD_STOP_UTC`, 2 min before the latest), no shaded
# band. re06's open-marker runs at offset 15 (F1 50 and 51) are not drawn. Under each run: its
# number and its bias.

# %%
P15 = {t: list(B06[t]) for t in TELS}
H15 = {t: {r: H06[t][r] for r in sorted({p["run"] for p in P15[t]})} for t in TELS}
LABELS15 = {t: {r: u"%d\n%s" % (r, blnw.bias_text([p["hv"] for p in P15[t] if p["run"] == r]))
                for r in H15[t]} for t in TELS}
footer15 = u"\n".join([
    SELECTION,
    u"Every point a band: %s." % tables.BAND_FOOTER,
    u"re06's hours panel at the standard point, its points and its clock; not drawn: its "
    u"open-marker runs at threshold offset 15 (%s). No merge applies at this step." % u"; ".join(
        run_list(t, OS15[t]) for t in TELS if OS15[t]),
    u"The display runs at 3.5e15, RFSel 2, threshold offset 20: H1 %s; F1 %s." % (
        bias_text(P15["h1"], "h1"), bias_text(P15["f1"], "f1")),
    u"x = hours from %s UTC, the earliest end of the irradiation the records allow (the latest is "
    u"2 min later); under each run its number and its bias." % (
        campaign.RAD_STOP_UTC[3.5e15][11:19])] + low_line(
    [p for t in TELS for p in P15[t]]) + [
    u"%s. %s" % (counts_text([p for t in TELS for p in P15[t]]), FLUENCE_TEXT)])
fig, n15 = curves.res_3p5e15_vs_hours(
    P15, H15, tel_titles=TEL_TITLES,
    xlabel=u"Hours after the end of the %s irradiation" % step_text(3.5e15), ylim=YLIM,
    tag=TAG, data=u"July, bookkeeping fluence %s" % step_text(3.5e15), footer=footer15,
    run_labels=LABELS15, label_y=21.0, stop=STOP06,
    legend_locs={"f1": "upper right"})      # F1 runs 13 and 14 reach the top on the left
save(fig, "re15_res_vs_hours_after_3p5e15",
     payload(n15, footer15, derived_from="re06_res_3p5e15_vs_bias_and_hours (hours panel)",
             hours={t: {str(r): h for r, h in H15[t].items()} for t in TELS},
             stop_utc=campaign.RAD_STOP_UTC[3.5e15],
             stop_bounds_utc=list(campaign.RAD_STOP_BOUNDS_UTC[3.5e15]),
             not_drawn_offset15=OS15, merge_rule=merge06, good_runs=GOOD06),
     conventions(colour="board (etroc_style.BOARD_COLOR): one step",
                 band=tables.BAND_FOOTER,
                 times="run start (tables/run_times.csv) minus RAD_STOP_UTC[3.5e15] (the "
                       "earliest allowed stop), hours, as on re06"))
