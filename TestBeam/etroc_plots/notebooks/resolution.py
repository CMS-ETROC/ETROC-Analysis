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
#
# Every point is one board in one run: its value in its anointed 3-board combination (boards
# 0-1-2 for boards 0, 1 and 2; boards 1-2-3 for board 3), the most-populated track per pixel,
# tracks with at least 300 events; the width is FWHM/2.355 of the TOA-difference core, as the
# result tables carry it. The bar is the pixel-to-pixel spread of the board's map. None of these
# figures has a 3.5e15 point, so none has a band (`../CONVENTIONS.md`, "Resolution and TDC
# figures").
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
# The drawing lives in `../resolution/curves.py`, the tables and bands in
# `../resolution/tables.py`. This notebook holds the choices made per figure: the runs, the
# words and the axis ranges.
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
# class, F1 46+47 sits at RFSel 1. The cell below prints the rule's outcome for every merge.
#
# A figure whose axis is time draws single runs: a merge spans the times of its runs (re04,
# re05). Every run drawn, single or merged, has to be good on the good-run list; the cells stop
# otherwise.

# %%
STEPS_JUL = (1.5e15, 2e15)
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
                                    for r, why in sorted(DROP01[t].items())),
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
    if any(DROP02.values()) else u"No run of these steps is at another operating point.",
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
# cooling down, in the later-look colour (teal; the March red stays with March). The July result
# point of each telescope follows the display list and the merge rule: H1 runs 3+4 merged at
# 540 V (both display runs; the merge replaces the preferred run 4), F1 run 9 at 460 V (the
# preferred display run).
#
# Two more July points are drawn with open markers: H1 run 10 at 550 V and F1 run 4 at 480 V.
# They are off the display-run list; they are the deck's second July point per telescope, kept
# because the display runs sit at one bias per telescope. To drop them, set `EXTRA03 = {}` in
# the cell below.
#
# March and July differ in more than the time: the telescopes were reinstalled, and the July
# biases sit above every March bias. Read the gap between the scans as "four months later, at a
# higher bias", not as cooling down alone.

# %%
EXTRA03 = {"h1": [10], "f1": [4]}     # off the display list: the second July point per telescope
P03 = march_points(1.5e15)[0]
for t in TELS:
    P03[t] = P03[t] + curves.standard(curves.points(
        BT, BM, RS, t, [RESULT[(t, 1.5e15)]] + EXTRA03.get(t, []), "july", VARIANT, FLOOR))
check_good([p for t in TELS for p in P03[t] if p["camp"] == "march"], "march")
GOOD03 = check_good([p for t in TELS for p in P03[t] if p["camp"] == "july"], "july")
off03 = {t: set(EXTRA03.get(t, [])) for t in TELS}
july03 = {t: [p for p in P03[t] if p["camp"] == "july"] for t in TELS}
footer03 = u"\n".join([
    SELECTION,
    u"March: every run at 1.5e15 at RFSel 2, threshold offset 20 (H1 runs %s; F1 runs %s)." % (
        bias_text([p for p in P03["h1"] if p["camp"] == "march"], "h1"),
        bias_text([p for p in P03["f1"] if p["camp"] == "march"], "f1")),
    u"July, after four months of cooling down: the display run of the step, merges by the merge "
    u"rule (H1 %s; F1 %s); thick black edge: merged runs." % (
        bias_text([p for p in july03["h1"] if not p["run"] in off03["h1"]], "h1"),
        bias_text([p for p in july03["f1"] if not p["run"] in off03["f1"]], "f1")),
    u"Open markers: July runs off the display-run list, kept as a second bias point (H1 %s; "
    u"F1 %s)." % (bias_text([p for p in july03["h1"] if p["run"] in off03["h1"]], "h1"),
                   bias_text([p for p in july03["f1"] if p["run"] in off03["f1"]], "f1")),
    u"%s. %s" % (counts_text([p for t in TELS for p in P03[t]]), FLUENCE_TEXT)])
fig, n03 = curves.res_1p5e15_march_vs_july(
    P03, off03, open_label=u"off the display list", tel_titles=TEL_TITLES,
    xlabel=style.XLABEL_BIAS, ylim=YLIM, tag=TAG,
    data=u"March and July, bookkeeping fluence %s" % step_text(1.5e15), footer=footer03)
save(fig, "re03_res_1p5e15_march_vs_july",
     payload(n03, footer03, off_list={t: sorted(v) for t, v in off03.items()},
             merge_rule={k: v for k, v in MERGE_EFFECT.items() if k.endswith("1.5e+15")},
             good_runs=GOOD03),
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
        u"%s runs %s" % (t.upper(), bias_text(NOT04[t], t)) for t in TELS if NOT04[t]),
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
    u"run's HV plateau. %s." % counts_text([p for t in TELS for p in P05[t]]),
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
