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
# # Front end: baseline, noise width, TDC codes, TOT and TOA, CERN IRRAD 2026
#
# What the irradiation did to the ETROC2 front end, up to the TOA difference of two boards:
#
# * Figure fe01: the autocalibration baseline (BL) and noise width (NW) of every board at every
#   fluence step, March and July.
# * Figure fe02: the same at 1.5e15 for one run pair per telescope, before and after the four
#   months of cooling down between the campaigns.
# * Figures fe03-fe04: the CAL, TOA and TOT of one pixel per chip in that run pair, H1 and F1.
# * Figures fe05-fe06: the CAL code of every pixel in that run pair, and its change, H1 and F1.
# * Figure fe07: the median TOT of every board at every fluence step, March and July.
# * Figure fe08: the same with one run per step, beside the most frequent TOT of the board.
# * Figures fe09-fe10: the TOT distribution of each board, and of one pixel per board, at every
#   step.
# * Figures fe11-fe12: the distribution of the TOA difference of two boards after the time-walk
#   correction, every pixel pooled and one pixel pair, at every step.
# * Figures fe13-fe14: the TOT and TOA difference of one pixel in the three July runs, H1 and F1,
#   with the TOA difference at 25 ps, about one TOA code.
# * Figure fe15: the TOT of one F1 chip at 3.5e15 in two pairs of July runs that differ in the
#   RFSel setting only.
# * Figure fe16: the TOT of that chip in three July runs at threshold offset 10, the first two at
#   RFSel 2 and days apart, the third at RFSel 3: the change over time at fixed settings beside
#   the RFSel step of fe15.
#
# Every figure is saved as PNG (200 dpi) and PDF, with `<name>_values.json` beside it: every
# number drawn, the input files read and the commit of the code.
#
# To run it:
#
# 1. `source TestBeam/condor_at_lxplus/envs/load_python39.sh` (LCG_104d, the environment this
#    notebook is tested in).
# 2. The figures read small tables that are not in the repository, from
#    `/eos/user/m/musafdar/ETROC_plot_inputs/irrad_2026`: the tables of `frontend/`,
#    `tables/runs_summary.csv`, `tables/run_times.csv` and `july/display_runs_jul.csv` (the list
#    below says which figure reads which). `../campaigns/irrad_2026_inputs.md` lists every table
#    with where it came from;
#    `python -m etroc_plots.frontend.rebuild` (run from `TestBeam/`) remakes the `frontend/`
#    tables from the autocalibration scan histories, the run metadata, the merged hits and the
#    reconstructed tracks (about an hour for the tracks, on one lxplus core). To
#    read your own copy, copy that whole `irrad_2026` folder and point `ETROC_INPUTS` at the
#    copy. The first code cell checks that every input exists and can be read, and stops with a
#    list of all that cannot; a table whose checksum differs from the published list is named,
#    and the run goes on.
# 3. Open `frontend.ipynb` in Jupyter and run all cells, or run it headless with
#    `jupyter nbconvert --to notebook --execute frontend.ipynb --output-dir <somewhere>`.
#
# Figures go to `$ETROC_FIGURES/frontend/`, by default `figures/frontend/` next to this notebook.
#
# The tables read, in `frontend/` unless named otherwise:
#
# * `blnw_by_run.csv`: the baseline and noise width of every pixel's last autocalibration scan
#   before each run (fe01-fe02).
# * `tdc_pixel_hists.csv`: the CAL, TOA and TOT codes of one pixel per chip in the run pair
#   (fe03-fe04).
# * `tdc_cal_modes.csv`: the most frequent CAL code of every pixel in the run pair (fe05-fe06).
# * `tot_pixel_medians.csv`: the median TOT and event count of every pixel's triple in every run
#   (fe07; the fixed pixel of fe10; the pixels and the chip of fe15-fe16).
# * `tot_pooled_summary.csv`: per board and run, the median, spread and most frequent TOT and the
#   counts (fe07-fe09; the control of fe15-fe16).
# * `tot_pooled_hist.csv`: the TOT of every counted pixel of a board together, per run (fe09; the
#   control of fe15-fe16).
# * `dtoa_triples.csv`: per pixel triple and board pair, the TOA-difference counts and the CAL
#   medians (fe11-fe14).
# * `dtoa_pooled_hist.csv`: the TOA difference of every counted triple of a board pair together
#   (fe11).
# * `pixel_hists.csv`: the TOT (in ns and per TOT code) and TOA difference of the one pixel or
#   pixel pair drawn (fe10, fe12-fe14).
# * `rfsel_pair_hist.csv`: the TOT of the chosen chip in each run of the RFSel pairs (fe15) and
#   of their time control (fe16).
# * `triples.csv`: the triple chosen for every pixel. No figure draws it; the other track tables
#   were made from it, and the first code cell checks it with the other tables.
# * `tables/runs_summary.csv`: the chips, bias, fluence step, RFSel and threshold offset of every
#   run (every figure).
# * `tables/run_times.csv`: the start of every run, for the time after irradiation in the
#   legends.
# * `july/display_runs_jul.csv`: the July display runs of each step and the preferred one
#   (fe01, fe07-fe15).
#
# The drawing lives in `../frontend/blnw.py` (figures fe01-fe02), `../frontend/tdc_codes.py`
# (figures fe03-fe06) and `../frontend/tot_shapes.py` (figures fe07-fe16). This notebook holds
# the choices made per figure: the runs, the colour scales and the words on each figure.
#
# `frontend.ipynb` and `frontend.py` are the same notebook, kept in step by jupytext. Edit
# either one, then run `python3 -m jupytext --sync frontend.ipynb` and
# `python3 -m nbstripout frontend.ipynb` in this folder before committing. Neither tool is in
# LCG_104d (the install line is in `../README.md`); running the notebook needs neither.
#
# After a run, `python3 -m etroc_plots.checks <figures>`, run from `TestBeam/` with `<figures>`
# the folder this notebook wrote to (`etroc_plots/notebooks/figures/frontend` by default),
# checks every saved figure against the rules in `../CONVENTIONS.md`.

# %%
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")   # draw off-screen; show() displays the PNG exactly as saved
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display
from matplotlib.colors import Normalize

# The folder that holds the package (TestBeam/) is two folders up. Jupyter starts the kernel in
# the notebook's folder; `python frontend.py` has __file__.
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
from etroc_plots.frontend import blnw, tdc_codes, tot_shapes  # noqa: E402
from etroc_plots.iv import iv_plot                            # noqa: E402
from etroc_plots.overview import timeline                     # noqa: E402
from etroc_plots.resolution import pixel_maps, tables         # noqa: E402

if campaigns.NAME != CAMPAIGN:
    raise RuntimeError("the package is running campaign %r (ETROC_CAMPAIGN) but CAMPAIGN = %r: "
                       "set one to match the other, then restart the kernel (the campaign is "
                       "read once, on the first import)" % (campaigns.NAME, CAMPAIGN))
campaign = campaigns.active

INPUTS_BLNW = [campaign.BLNW_CSV, campaign.RUNS_SUMMARY_CSV, campaign.RUN_TIMES_CSV,
               campaign.DISPLAY_RUNS_JUL_CSV]
INPUTS_TDC = [campaign.TDC_PIXEL_HISTS_CSV, campaign.TDC_CAL_MODES_CSV, campaign.RUNS_SUMMARY_CSV,
              campaign.RUN_TIMES_CSV]
# stops, listing every missing or unreadable input; reports any table that differs from the
# published copy.
check_inputs(only=("frontend/", "tables/runs_summary.csv", "tables/run_times.csv",
                   "july/display_runs_jul.csv"))

NOTEBOOK = "TestBeam/etroc_plots/notebooks/frontend.ipynb"   # recorded as "script" in every values file
OUT = os.path.join(os.path.abspath(os.environ.get("ETROC_FIGURES") or "figures"), "frontend")
os.makedirs(OUT, exist_ok=True)

TELS = list(campaign.TELESCOPE_CHIPS)
RS = pd.read_csv(campaign.RUNS_SUMMARY_CSV)
RUN_TIMES = timeline.read_run_times()
SETTINGS = campaign.FRONTEND_JULY_SETTINGS     # RFSel 2, threshold offset 20
# March runs record no RFSel; they ran at the campaign's MARCH_RFSEL, the July setting, so every
# figure here states one RFSel for both campaigns.
assert campaign.MARCH_RFSEL == SETTINGS["rfsel"], "March and July ran at different RFSel"
SETTINGS_TEXT = style.settings_text(rfsel=SETTINGS["rfsel"], offset=SETTINGS["offset"])


def show(stem):
    """Display a saved figure, exactly as written to disk."""
    display(Image(filename=os.path.join(OUT, stem + ".png"), width=900))


def save(fig, stem, payload, conventions, inputs=()):
    _, problems = style.save_figure(fig, OUT, stem)
    plt.close(fig)
    payload["overlap_problems"] = problems
    style.write_values(OUT, stem, payload, conventions=conventions, script=NOTEBOOK,
                       inputs=list(inputs))
    print("%s: audit %s" % (stem, problems or "clean"))
    show(stem)


def step_fluence(camp, tel, run):
    """The bookkeeping fluence step of a run (every board of a telescope shares it)."""
    f = {style.fluence_key(b["fluence"]) for b in tables.run_settings(
        RS, campaign.TABLE_CAMPAIGN[camp], campaign.TABLE_TEL[tel], run)}
    if len(f) != 1:
        raise ValueError("%s %s run %d: boards at different fluences %s" % (camp, tel, run, f))
    return f.pop()


def run_bias(camp, tel, run):
    """The sensor bias of a run's boards, as text ('400 V' or '530/550 V')."""
    return blnw.bias_text([b["hv"] for b in tables.run_settings(
        RS, campaign.TABLE_CAMPAIGN[camp], campaign.TABLE_TEL[tel], run)])


# %% [markdown]
# ## The runs
#
# * The fluence ladder (fe01). March: at each step the highest-bias run at threshold offset 20
#   (`MARCH_DISPLAY_RUNS` in the campaign module, the runs of the maps notebook). July: every run
#   of the display-run list (`july/display_runs_jul.csv`) at RFSel 2 and threshold offset 20,
#   preferred or not, side by side, then the step's extra runs.
# * An extra run (`LADDER_EXTRA_RUNS` in the campaign module) is drawn beside the display runs of
#   its step on the ladders of fe01 and fe07 only, as the CERN IRRAD 2026 talk drew it. It is not
#   a display run and no other figure reads it. The one extra is H1 run 19 at 3.5e15, the repeat
#   of run 18 eight hours later at the same settings; it reused run 18's autocalibration scan, so
#   its fe01 points coincide with run 18's.
# * The pair at 1.5e15 (fe02-fe06), `COOLDOWN_PAIRS` in the campaign module: before cooling
#   down, the March run of the 1.5e15 step whose leakage current matches the July run's (its
#   lowest bias, run 17 on both telescopes); after cooling down, the July preferred display run
#   at 1.5e15.
#
# The tick under each irradiated step gives the time from the end of that step's irradiation
# (`RAD_STOP_UTC` in the campaign module, the earliest stop the records allow) to the start of
# the run, or the range over the runs drawn there; the July 1.5e15 runs count from the end of
# the March 1.5e15 step. Run starts come from `tables/run_times.csv`.

# %%
july_display = tables.display_runs(campaign.DISPLAY_RUNS_JUL_CSV, **SETTINGS)
JULY_EXTRAS = {t: campaign.LADDER_EXTRA_RUNS.get(("july", t), {}) for t in TELS}
# fe01 and fe07 describe the extra runs in words; other extras need other words there
if JULY_EXTRAS != {"h1": {3.5e15: [19]}, "f1": {}}:
    raise ValueError("ladder extras %s: update the words of fe01 and fe07" % JULY_EXTRAS)
july_ladder = {t: blnw.ladder_runs(july_display[t], JULY_EXTRAS[t]) for t in TELS}
march_steps = {t: {step_fluence("march", t, r): r for r in campaign.MARCH_DISPLAY_RUNS[t]}
               for t in TELS}
LADDER = ([dict(campaign="march", fluence=f, runs={t: [march_steps[t][f]] for t in TELS})
           for f in sorted(march_steps[TELS[0]])]
          + [dict(campaign="july", fluence=f, runs={t: july_ladder[t][f] for t in TELS})
             for f in sorted(july_display[TELS[0]])])
EXTRA_RUNS = {t: {campaign.FLUENCE_TEXT_PLAIN[f]: rs for f, rs in e.items()}
              for t, e in JULY_EXTRAS.items()}      # as the values files of fe01 and fe07 give them
print("July extra runs on the ladders of fe01 and fe07:", EXTRA_RUNS)
PAIR = [dict(campaign=camp, fluence=1.5e15,
             runs={t: [campaign.COOLDOWN_PAIRS[t][k]] for t in TELS})
        for k, camp in enumerate(("march", "july"))]

HOURS = {}      # (campaign, telescope, run) -> hours after the end of the step's irradiation


def step_time(step, tel):
    if step["fluence"] == 0.0:
        return None
    text, hours = blnw.time_after_stop(RUN_TIMES, step["campaign"], tel, step["runs"][tel],
                                       campaign.RAD_STOP_UTC[step["fluence"]])
    HOURS.update({(step["campaign"], tel, r): round(h, 2) for r, h in hours.items()})
    return text


for st in LADDER:
    for t in TELS:
        print("%-6s %-10s %s runs %s, %s, after the step: %s"
              % (st["campaign"], campaign.FLUENCE_TEXT_PLAIN[st["fluence"]], t, st["runs"][t],
                 "/".join(run_bias(st["campaign"], t, r) for r in st["runs"][t]),
                 step_time(st, t) or "-"))

# %% [markdown]
# ## Figure fe01: baseline and noise width against fluence
#
# One point per board and run: the autocalibration scan the DAQ used for that run (the scan
# whose baseline map equals the run's threshold DAC minus its threshold offset; the table's
# `match` column says how well, and the cell prints any run that is not an exact match). Two
# statistics over the board's pixels, both drawn:
#
# * filled: over all 256 pixels, a railed or dead pixel as the scan stored it and a missing noise
#   width as 0;
# * open: over the pixels with a real value: for BL 50 < baseline < 1020 DAC and not railed in
#   the run, for NW a noise width above 0. The footer gives how many pixels that leaves out.
#
# The bar is the pixel-to-pixel standard deviation, not an uncertainty. Colour = board, marker =
# chip; the small text over each step is the sensor bias of the runs drawn there.

# %%
QUANTITIES = [dict(key="bl", ylabel="Baseline [DAC]"),
              dict(key="nw", ylabel="Noise width [DAC]")]
STATS = [dict(key="allpix", label="all 256 pixels", open=False),
         dict(key="live", label="pixels with a value", open=True)]
TEL_TITLES = dict(campaign.TELESCOPE_TITLE)
XLABEL_LADDER = "Bookkeeping fluence [%s]" % campaign.FLUENCE_UNIT

table = blnw.read_blnw()


def blnw_footer(pts, what):
    bl, nw = blnw.excluded_range(pts, "bl"), blnw.excluded_range(pts, "nw")
    return (u"Filled: mean over all 256 pixels (a railed or dead pixel as stored, a missing NW as "
            u"0). Bar: pixel-to-pixel standard deviation.\nOpen: mean over the pixels with a "
            u"value (BL: 50 < baseline < 1020 DAC and not railed, %d-%d of 256 left out per "
            u"board; NW: above 0, %d-%d left out). One autocalibration scan per board and run, "
            u"the scan the DAQ set the thresholds from.\n%s" % (bl + nw + (what,)))


pts = blnw.points(table, LADDER, QUANTITIES, STATS)
ticks = {t: [style.fluence_label(st["fluence"], unit=False)
             + ("\n" + step_time(st, t) if st["fluence"] else "") for st in LADDER] for t in TELS}
off = sorted({(p["campaign"], p["tel"], p["run"], p["chip"], p["match"]) for p in pts
              if p["match"] != "exact"})
print("boards not on an exact scan match:", off or "none")
fig = blnw.ladder_vs_fluence(
    pts, LADDER, QUANTITIES, STATS, tick_labels=ticks, xlabel=XLABEL_LADDER,
    tel_titles=TEL_TITLES, tag=SETTINGS_TEXT, data=None,
    boundary_after=sum(st["campaign"] == "march" for st in LADDER) - 1,
    footer=blnw_footer(pts, u"March: one run per step, the highest-bias run at threshold offset "
                            u"20; July: every display run at the step, side by side. Dotted "
                            u"line: March | July. Small text: sensor bias.\nH1 3.5e15: run 19, "
                            u"the repeat of run 18 eight hours later, drawn beside it; it reused "
                            u"run 18's autocalibration scan, so its points coincide. Not a "
                            u"display run. Tick: time from the end of the step's irradiation to "
                            u"the run start."))
save(fig, "fe01_blnw_vs_fluence",
     dict(points=pts, steps=LADDER, tick_labels=ticks,
          hours_after_stop={"%s %s run %d" % k: v for k, v in sorted(HOURS.items())},
          excluded_pixels={q: blnw.excluded_range(pts, q) for q in ("bl", "nw")},
          not_exact=off, extra_runs=EXTRA_RUNS),
     dict(fluence=campaign.FLUENCE_CONVENTION,
          statistics="allpix: mean and standard deviation over all 256 pixels; live: over the "
                     "pixels with a value (BL 50 < baseline < 1020 DAC, not railed; NW > 0)",
          error_bar="pixel-to-pixel standard deviation, not an uncertainty",
          times="run start (tables/run_times.csv) minus RAD_STOP_UTC of the step, hours",
          rfsel="July runs at RFSel %d by the display-run list (the extra run, H1 19, repeats run "
                "18 at its settings); March runs record no RFSel and ran at RFSel %d"
                % (SETTINGS["rfsel"], campaign.MARCH_RFSEL)),
     inputs=INPUTS_BLNW)

# %% [markdown]
# ## Figure fe02: baseline and noise width before and after cooling down
#
# The run pair at 1.5e15: each board at its two looks, joined by a thin line (solid for the
# all-pixel statistic, dashed for the other). Same statistics, colours and markers as fe01.

# %%
pair_pts = blnw.points(table, PAIR, QUANTITIES, STATS)
LOOK_WORDS = ("before cooling down", "after cooling down")
pair_ticks = {t: ["%s\n%s run %d, %s" % (LOOK_WORDS[i], st["campaign"].capitalize(),
                                         st["runs"][t][0], step_time(st, t))
                  for i, st in enumerate(PAIR)] for t in TELS}
fig = blnw.rest_pair(
    pair_pts, PAIR, QUANTITIES, STATS, tick_labels=pair_ticks,
    xlabel=u"%s %s, time after the step's irradiation"
           % (style.fluence_label(1.5e15, unit=False), campaign.FLUENCE_UNIT),
    tel_titles=TEL_TITLES, tag=SETTINGS_TEXT, data=None,
    footer=blnw_footer(pair_pts, u"Before cooling down: March, the 1.5e15 run whose leakage "
                                 u"current matches the July run's; after: July, four months "
                                 u"later, the preferred display run. Bias, March / July: %s."
                       % "; ".join("%s %s / %s" % (t.upper(), *(run_bias(st["campaign"], t,
                                                                          st["runs"][t][0])
                                                                 for st in PAIR))
                                   for t in TELS)))
save(fig, "fe02_blnw_cooling_down",
     dict(points=pair_pts, pairs=campaign.COOLDOWN_PAIRS, tick_labels=pair_ticks,
          excluded_pixels={q: blnw.excluded_range(pair_pts, q) for q in ("bl", "nw")}),
     dict(fluence=campaign.FLUENCE_CONVENTION,
          statistics="as fe01", error_bar="pixel-to-pixel standard deviation, not an uncertainty",
          times="run start (tables/run_times.csv) minus RAD_STOP_UTC of the 1.5e15 step, hours"),
     inputs=INPUTS_BLNW)

# %% [markdown]
# ## Figures fe03-fe04: CAL, TOA and TOT of one pixel per chip
#
# The run pair at 1.5e15, one pixel per chip: the most populated pixel in the July run, the same
# pixel in March. Every loop file of the run; hits with the error flag clear (ea = 0) and a CAL
# code within 3 of the pixel's CAL mode in that file (the autocalibration drifts from file to
# file). TOA = 12.5 ns - TOA code x 3.125 ns / CAL mode and TOT = (2 TOT code - floor(TOT code /
# 32)) x 3.125 ns / CAL mode, with that same per-file mode.
#
# Bins: CAL in native codes; TOA in 0.3 ns and TOT in 0.2 ns, summed from the 0.1 ns bins of the
# table (finer bins show the comb of the TDC codes). Each histogram is the fraction of its hits in
# the range drawn per bin (the bins sum to 1); the hits outside it are counted in the values
# file. The box gives the CAL mode (most frequent code) and the TOA and TOT medians (lower edge of
# the 0.1 ns bin holding the 50 % point), and N, the hits in the range drawn.

# %%
hists = tdc_codes.read_hists()
cal_modes = tdc_codes.read_cal_modes()
BEFORE_COLOR = iv_plot.timing_color(0, style.fluence_color(1.5e15))   # the IV notebook's looks
AFTER_COLOR = iv_plot.timing_color(1, style.fluence_color(1.5e15))
PANELS = [dict(key="cal", width=1, range=None, margin=8, xlabel="CAL code"),
          dict(key="toa", width=0.3, range=(0.0, 12.3), xlabel="TOA [ns]"),
          dict(key="tot", width=0.2, range=(0.0, 10.0), xlabel="TOT [ns]")]
STAT_WORDS = {"cal": "CAL mode", "toa": "TOA median", "tot": "TOT median"}


def looks(tel):
    out = []
    for i, st in enumerate(PAIR):
        run = st["runs"][tel][0]
        when = "March" if st["campaign"] == "march" else "July"
        out.append(dict(campaign=st["campaign"], run=run, label=when,
                        color=(BEFORE_COLOR, AFTER_COLOR)[i], fill=(i == 0),
                        legend="%s, run %d, %s: %s" % (when, run, run_bias(st["campaign"], tel, run),
                                                       LOOK_WORDS[i])))
    return out


def pixels_of(tel):
    s = hists[(hists["tel"] == tel) & (hists["campaign"] == "july")
              & (hists["run"] == campaign.COOLDOWN_PAIRS[tel][1])]
    return {int(b): (int(g["row"].iloc[0]), int(g["col"].iloc[0]))
            for b, g in s.groupby("board_idx")}


TDC_FOOTER = (u"One pixel per chip, the most populated in the July run; every loop file; ea = 0 "
              u"and |CAL - CAL mode of the pixel in that file| <= 3. TOA = 12.5 ns - TOA code x "
              u"3.125 ns / CAL mode, TOT = (2 TOT code - floor(TOT code / 32)) x 3.125 ns / CAL "
              u"mode.\nBins: CAL native codes, TOA 0.3 ns, TOT 0.2 ns; each histogram the fraction of "
              u"its hits in the range drawn, per bin. Box: CAL mode, TOA and TOT median; N: hits "
              u"in the range drawn.")
for k, tel in enumerate(TELS):
    pix = pixels_of(tel)
    fig, payload = tdc_codes.hists(
        hists, tel, looks(tel), PANELS, pixels=pix, tag=SETTINGS_TEXT,
        data="%s %s" % (style.fluence_label(1.5e15, unit=False), campaign.FLUENCE_UNIT),
        footer=TDC_FOOTER, stat_words=STAT_WORDS)
    for rec in payload:
        print("%s %s pixel (%d, %d) %s: %s" % (tel, rec["chip"], rec["row"], rec["col"],
                                              rec["panel"], ", ".join(
            "run %d %g (outside %d)" % (lk["run"], lk["value"], lk["hits_outside"])
            for lk in rec["looks"])))
    save(fig, "fe%02d_tdc_codes_%s" % (3 + k, tel),
         dict(panels=payload, pixels={b: list(v) for b, v in pix.items()},
              runs=[dict(campaign=lk["campaign"], run=lk["run"]) for lk in looks(tel)]),
         dict(selection="one pixel per chip (most populated in the July run of COOLDOWN_PAIRS); "
                        "all loop files; ea == 0; |cal - per-file CAL mode| <= 3",
              toa_ns="12.5 - toa * 3.125 / CAL mode (per file)",
              tot_ns="(2 tot - floor(tot / 32)) * 3.125 / CAL mode (per file)",
              bins="CAL native code; TOA 0.3 ns over 0-12.3 ns; TOT 0.2 ns over 0-10 ns; "
                   "fraction of the hits in the range drawn, per bin",
              median="lower edge of the 0.1 ns bin holding the 50 % point",
              fluence=campaign.FLUENCE_CONVENTION),
         inputs=INPUTS_TDC)

# %% [markdown]
# ## Figures fe05-fe06: CAL code maps
#
# Each pixel's CAL mode in the run (the median over the loop files of its most frequent CAL code
# in each file), before and after cooling down, and the change (July minus March); the pixel of
# figures fe03-fe04 is ringed. One CAL colour scale for every CAL map of this notebook, set below
# from all the maps drawn (0.5th to 99.5th percentile of the modes), so the CAL maps compare by
# colour across both telescopes. The differences take one symmetric scale per telescope, from
# that telescope's 99th percentile of |July - March| (11 codes for F1, 2 for H1): one shared
# scale would draw every H1 difference in nearly one colour. A pixel outside a scale takes the
# end colour and is counted ("sat N" in its title). Grey: no CAL mode. Maps in the chip frame:
# pixel (0, 0) at the bottom right.
#
# What the difference maps show: the July CAL modes of LF10 and LF12 sit about 10 codes below
# March over the whole chip, while LF17 and LF18 moved by about 2 codes and the H1 chips by 0 to
# 1 (per-chip medians; the cell prints them with the 5th and 95th percentiles, and each footer
# states them). This is an observation; no cause is drawn from it here. CAL counts the
# delay-line steps in one 3.125 ns clock period, so it moves with the chip's temperature and
# supply.

# %%
grids = {}
for tel in TELS:
    for st in PAIR:
        for b in range(len(campaign.TELESCOPE_CHIPS[tel])):
            grids[(tel, st["campaign"], b)] = tdc_codes.mode_grid(
                cal_modes, st["campaign"], tel, st["runs"][tel][0], b)
modes = np.concatenate([g[np.isfinite(g)] for g in grids.values()])
CAL_NORM = Normalize(vmin=float(np.floor(np.percentile(modes, 0.5))),
                     vmax=float(np.ceil(np.percentile(modes, 99.5))))
DIFF_LIM, CAL_SHIFT = {}, {}
for t in TELS:
    d = {chip: grids[(t, "july", b)] - grids[(t, "march", b)]
         for b, chip in enumerate(campaign.TELESCOPE_CHIPS[t])}
    alld = np.concatenate([g[np.isfinite(g)] for g in d.values()])
    DIFF_LIM[t] = max(1.0, float(np.ceil(np.percentile(np.abs(alld), 99))))
    # per chip: median and 5th / 95th percentile of July - March over its pixels (+ 0.0: no -0)
    CAL_SHIFT[t] = {chip: [float(v) + 0.0 for v in np.percentile(g[np.isfinite(g)], [50, 5, 95])]
                    for chip, g in d.items()}
print("CAL scale %g-%g; difference scale %s" % (
    CAL_NORM.vmin, CAL_NORM.vmax, ", ".join("%s +-%g" % (t, DIFF_LIM[t]) for t in TELS)))
print("July minus March per chip, [median, 5th, 95th percentile]:", CAL_SHIFT)


def shift_text(tel):
    return "; ".join(u"%s %+g (%+g to %+g)" % ((c,) + tuple(v)) for c, v in CAL_SHIFT[tel].items())


def map_footer(tel):
    return (u"CAL mode: median over the loop files of the pixel's most frequent CAL code in each "
            u"file (ea = 0). Difference: July minus March. Grey: no CAL mode.\n"
            u"One CAL scale for every CAL map of both telescopes; one difference scale per "
            u"telescope, symmetric, from its 99th percentile of |July - March| (here ±%g).\n"
            u"\"sat N\" in a title: N pixels outside that map's scale. Ringed: the pixel of the "
            u"CAL, TOA and TOT histograms. Maps in the %s.\n"
            u"July minus March per chip, median (5th to 95th percentile of its pixels), codes: "
            u"%s.\nCAL counts the delay-line steps in one 3.125 ns clock period, so it moves "
            u"with the chip's temperature and supply; no cause is drawn from these shifts."
            % (DIFF_LIM[tel], pixel_maps.FRAME_TEXT, shift_text(tel)))


for k, tel in enumerate(TELS):
    lk = looks(tel)
    words = ["%s, run %d" % (x["label"], x["run"]) for x in lk] + ["July minus March"]
    fig, payload = tdc_codes.cal_maps(
        {(c, b): g for (t, c, b), g in grids.items() if t == tel}, tel, lk,
        pixels=pixels_of(tel), norm=CAL_NORM, diff_norm=tdc_codes.symmetric_norm(DIFF_LIM[tel]),
        titles=lambda chip, j: "%s, %s" % (chip, words[j]),
        cbar_labels=("CAL mode [code]", "CAL mode, July minus March [code]"),
        tag=SETTINGS_TEXT,
        data="%s %s" % (style.fluence_label(1.5e15, unit=False), campaign.FLUENCE_UNIT),
        footer=map_footer(tel))
    for rec in payload:
        print("%s %s: saturated before %d, after %d, difference %d" % (
            tel, rec["chip"], rec["before"]["saturated"], rec["after"]["saturated"],
            rec["difference"]["saturated"]))
    save(fig, "fe%02d_cal_maps_%s" % (5 + k, tel),
         dict(maps=payload, cal_scale=[CAL_NORM.vmin, CAL_NORM.vmax],
              difference_scale=[-DIFF_LIM[tel], DIFF_LIM[tel]],
              difference_median=CAL_SHIFT[tel],
              runs=[dict(campaign=x["campaign"], run=x["run"]) for x in lk]),
         dict(cal_mode="median over the loop files of the per-file mode of the CAL code (ea == 0)",
              scales="CAL: 0.5th-99.5th percentile of every CAL mode of fe05-fe06 (one scale for "
                     "both); difference: 99th percentile of |July - March| over this telescope's "
                     "chips, symmetric (one scale per telescope)",
              difference_median="per chip: median, 5th and 95th percentile over its pixels of "
                                "July - March, codes",
              frame=pixel_maps.FRAME_TEXT, fluence=campaign.FLUENCE_CONVENTION),
         inputs=INPUTS_TDC)

# %% [markdown]
# ## The track tables (figures fe07-fe14)
#
# Figures fe07-fe14 read the reconstructed tracks of each run, through the tables of `frontend/`
# made from them (`python -m etroc_plots.frontend.rebuild --part tracks`). A track file holds the
# events of one triple: one pixel on each of three boards, boards 0-1-2 or boards 1-2-3. Each
# pixel of a board takes ONE triple, the one with the most events among the triples of its
# board's combination (boards 0-1-2 for boards 0-2, boards 1-2-3 for board 3), and counts when
# that triple holds at least 300 events. Nothing is pooled across the triples of one pixel.
#
# The triple matters: a pixel's TOT median depends on which of its triples is read. The partner
# pixels on the other two boards fix where the track, at 60 deg to the chip, crosses the pixel,
# and corridors grazing the pixel read a lower TOT: in July F1 run 13, board 3 pixel (10, 14),
# two corridors of 1,675 and 1,669 events give medians of 3.01 and 2.02 ns (the other four
# corridors of that pixel 2.28-2.97 ns). The board statistics barely move with the rule: ranking
# each pixel's triples by file size instead (`--rank bytes`, the rule of the CERN IRRAD 2026 talk
# figures) changes the triple of 554 of 22,472 pixels (2.5 %) and moves the board values of
# fe07-fe09 by at most 0.0135 ns (median of the pixel medians) and 0.0029 ns (pooled median) over
# 88 boards, and the most frequent TOT of one board by one 0.1 ns bin (comparison of the two
# rebuilds, 2026-10-02). fe07 and fe08 state this in their footers.
#
# * TOT: the time over threshold of every event of that triple, in ns, as reconstructed.
# * dTOA: the TOA difference of two boards of one triple after the time-walk correction solved
#   jointly over the triple's three boards, centred on that triple's own median, so a pooled
#   curve is the pair's timing shape and not the map of pixel-to-pixel offsets.
#
# The runs: fe07 the runs of fe01; fe08-fe12 one run per step (March: the run of fe01; July: the
# preferred display run, `tot_shapes.shape_runs`); fe13-fe14 the three July runs of those. The
# fixed pixels of fe10, fe12 and fe13-fe14 were chosen by the rebuild over the same runs; this
# cell checks the choice against the per-pixel table.

# %%
INPUTS_TRACKS = [campaign.TOT_PIXEL_MEDIANS_CSV, campaign.TOT_POOLED_SUMMARY_CSV,
                 campaign.TOT_POOLED_HIST_CSV, campaign.DTOA_TRIPLES_CSV,
                 campaign.DTOA_POOLED_HIST_CSV, campaign.PIXEL_HISTS_CSV,
                 campaign.RUNS_SUMMARY_CSV, campaign.RUN_TIMES_CSV, campaign.DISPLAY_RUNS_JUL_CSV]
FLOOR = tot_shapes.FLOOR
pix_medians = pd.read_csv(campaign.TOT_PIXEL_MEDIANS_CSV)
SUMMARY = pd.read_csv(campaign.TOT_POOLED_SUMMARY_CSV).set_index(
    ["campaign", "tel", "run", "board_idx"])
TOT_POOLED = dict(list(pd.read_csv(campaign.TOT_POOLED_HIST_CSV).groupby(
    ["campaign", "tel", "run", "board_idx"])))
dtoa = pd.read_csv(campaign.DTOA_TRIPLES_CSV)
DTOA_POOLED = dict(list(pd.read_csv(campaign.DTOA_POOLED_HIST_CSV).groupby(
    ["campaign", "tel", "run", "pair"])))
PIXEL_HISTS = dict(list(pd.read_csv(campaign.PIXEL_HISTS_CSV)
                        .fillna({"board_b": -1}).astype({"board_b": int})
                        .groupby(["pick", "quantity", "campaign", "tel", "run", "board_a",
                                  "board_b"])))
july_pref = tables.preferred_runs(campaign.DISPLAY_RUNS_JUL_CSV, **SETTINGS)
NBOARDS = {t: len(campaign.TELESCOPE_CHIPS[t]) for t in TELS}
# The largest change of a board statistic when each pixel's triple is its largest file instead of
# its most populated one (the two rebuilds of 2026-10-02 compared over 88 boards; markdown above).
RANK_SHIFT = dict(pixmed_median_ns=0.01349, median_ns=0.00290, mode_boards_moved=1, boards=88)
RANK_SHIFT_TEXT = "%.3f ns" % (np.ceil(RANK_SHIFT["pixmed_median_ns"] * 1e3) / 1e3)  # at most
CORRIDOR_TEXT = (u"A pixel's TOT depends on which of its triples is read: the partner pixels fix "
                 u"where the 60 deg track crosses it, and corridors grazing the pixel read a lower "
                 u"TOT (medians up to 1 ns apart for corridors of equal count).")
SET_WORDS = "RFSel %d, threshold offset %d" % (SETTINGS["rfsel"], SETTINGS["offset"])
COMBO_WORDS = "boards 0-1-2 for boards 0-2, boards 1-2-3 for board 3"
MARCH_STEPS = {t: {step_fluence("march", t, r) for r in campaign.MARCH_DISPLAY_RUNS[t]}
               for t in TELS}


def board_bias(camp, tel, run, boards):
    hv = {b["board_idx"]: b["hv"] for b in tables.run_settings(
        RS, campaign.TABLE_CAMPAIGN[camp], campaign.TABLE_TEL[tel], run)}
    return blnw.bias_text([hv[b] for b in boards])


def shape_steps(tel):
    """One dict per run of the shape figures, lowest step first: its step, the time from the end
    of the step's irradiation and its colour (a July run at a March step is the later look)."""
    out = []
    for camp, run in tot_shapes.shape_runs(tel):
        f = step_fluence(camp, tel, run)
        text, hours = (None, {}) if f == 0.0 else blnw.time_after_stop(
            RUN_TIMES, camp, tel, [run], campaign.RAD_STOP_UTC[f])
        later = camp == "july" and f in MARCH_STEPS[tel]
        out.append(dict(campaign=camp, run=int(run), fluence=f, time=text,
                        hours=None if f == 0.0 else round(hours[run], 2), later=later,
                        color=AFTER_COLOR if later else style.fluence_color(f)))
    return out


def step_label(st, tel, boards):
    words = [style.fluence_label(st["fluence"], unit=False), st["time"],
             board_bias(st["campaign"], tel, st["run"], boards)]
    return ", ".join(w for w in words if w)


def fixed_pixel(tel, board, runs):
    """(pixel, {run: events}, candidates) by the rebuild's rule, from the per-pixel table: among
    the pixels of `board` at or above the floor in every run of `runs`, the one whose smallest
    count over the runs is largest."""
    per = {}
    for camp, run in runs:
        p = pix_medians[(pix_medians["campaign"] == camp) & (pix_medians["tel"] == tel)
                        & (pix_medians["run"] == run) & (pix_medians["board_idx"] == board)
                        & pix_medians["passes_floor"]]
        per[(camp, run)] = {(int(r), int(c)): int(n) for r, c, n in zip(p["row"], p["col"], p["n"])}
    common = set.intersection(*(set(d) for d in per.values()))
    best = max(sorted(common), key=lambda px: min(d[px] for d in per.values()))
    return best, {"%s %d" % k: d[best] for k, d in per.items()}, len(common)


def stored_pixel(pick, quantity, tel, runs, a, b=-1):
    """The (row, col) of board `a` in the pixel histograms of `runs`; it must be one pixel."""
    px = {(int(g["row_a"].iloc[0]), int(g["col_a"].iloc[0]))
          for g in (PIXEL_HISTS[(pick, quantity, c, tel, r, a, b)] for c, r in runs)}
    if len(px) != 1:
        raise ValueError("%s %s board %d: pixels %s differ between the runs" % (pick, tel, a, px))
    return px.pop()


SHAPE = {t: shape_steps(t) for t in TELS}
for t in TELS:
    print("%s shape runs: %s" % (t, ", ".join("%s %d (%s)" % (st["campaign"], st["run"],
                                                              st["time"] or "pre")
                                             for st in SHAPE[t])))

# %% [markdown]
# ## Figure fe07: median TOT against fluence
#
# One point per board and run: the median, over the board's counted pixels, of each pixel's
# median TOT; the bar spans the 16th to 84th percentile of those pixel medians (the
# pixel-to-pixel spread, not an uncertainty). Colour = board, marker = chip, as in fe01: the
# figure is about the boards, and the fluence is on the axis. March: the run of each step; July:
# every display run at RFSel 2 and offset 20 and the extra run of fe01 (H1 run 19, the repeat of
# run 18), the preferred one large and joined, the others small beside it.

# %%
RUN_SPAN = 0.27       # x offset between the runs drawn at one step, in step units


def preferred_of(st, tel):
    runs = st["runs"][tel]
    if st["campaign"] == "march":
        return runs[0]
    (run,) = [r for r in runs if r in july_pref[tel]]
    return run


def tot_point(camp, tel, run, b, kind, step, dx=0.0):
    s = SUMMARY.loc[(camp, tel, run, b)]
    return dict(tel=tel, step=step, campaign=camp, run=int(run), board_idx=b, chip=s["chip"],
                kind=kind, value=round(float(s["pixmed_median_ns"]), 4),
                lo=round(float(s["pixmed_p16_ns"]), 4), hi=round(float(s["pixmed_p84_ns"]), 4),
                n_pixels=int(s["n_pixels"]), n_events=int(s["n_events"]), dx=dx)


pts07 = []
for i, st in enumerate(LADDER):
    for t in TELS:
        runs, main = st["runs"][t], preferred_of(st, t)
        for k, run in enumerate(runs):
            dx = (k - (len(runs) - 1) / 2.0) * RUN_SPAN
            for b in range(NBOARDS[t]):
                pts07.append(tot_point(st["campaign"], t, run, b,
                                       "main" if run == main else "other", i, dx))
npix07 = {k: (min(p["n_pixels"] for p in pts07 if p["kind"] == k),
              max(p["n_pixels"] for p in pts07 if p["kind"] == k)) for k in ("main", "other")}
TOT_RULE = (u"A pixel counts when its triple holds >= %d events; one triple per pixel, the one "
            u"with the most events in its board's combination (%s)." % (FLOOR, COMBO_WORDS))
fig = tot_shapes.board_vs_fluence(
    pts07, {t: LADDER for t in TELS},
    {"main": dict(label="March run; July preferred display run", line=True),
     "other": dict(label={t: "other July runs at the step\n(display runs%s)"
                             % ("; H1 run 19, the repeat of 18" if t == "h1" else "")
                          for t in TELS}, size=0.62)},
    tick_labels=ticks, xlabel=XLABEL_LADDER, ylabel="Median of the pixel TOT medians [ns]",
    tel_titles=TEL_TITLES, tag=SETTINGS_TEXT, data=None, board_dx=0.06,
    boundary_after=sum(st["campaign"] == "march" for st in LADDER) - 1,
    footer=(u"Point: median over the board's pixels of each pixel's median TOT. Bar: 16-84 %% "
            u"of those pixel medians (the pixel-to-pixel spread, not an uncertainty); %d-%d "
            u"pixels per board, %d-%d in the other runs.\n%s Ranking by file size "
            u"instead moves these board values by at most %s.\n%s\nMarch: one run per step. "
            u"July: every run drawn at the step (%s) side by side, the preferred one large and "
            u"joined. Dotted line: March | July. Tick: time from the end of the step's "
            u"irradiation to the run start."
            % (npix07["main"] + npix07["other"]
               + (TOT_RULE, RANK_SHIFT_TEXT, CORRIDOR_TEXT, SET_WORDS))))
save(fig, "fe07_tot_median_vs_fluence",
     dict(points=pts07, steps=LADDER, tick_labels=ticks,
          pixels_per_board={k: list(v) for k, v in npix07.items()},
          preferred_july={t: july_pref[t] for t in TELS},
          hours_after_stop={"%s %s run %d" % k: v for k, v in sorted(HOURS.items())},
          rank_shift=RANK_SHIFT, extra_runs=EXTRA_RUNS),
     dict(fluence=campaign.FLUENCE_CONVENTION,
          statistic="median over the counted pixels of each pixel's median TOT (ns)",
          error_bar="16th-84th percentile of the pixel medians, not an uncertainty",
          pixels="a pixel counts when its triple (the one with the most events in boards 0-1-2, "
                 "board 3: 1-2-3) holds >= %d events" % FLOOR,
          triple_rule="one triple per pixel, the most populated; a pixel's TOT median depends on "
                      "the triple (track corridor) read, the board statistic barely: rank_shift "
                      "is the largest change of a board value when the triple is the largest "
                      "file instead (two rebuilds of 2026-10-02 compared)",
          times="run start (tables/run_times.csv) minus RAD_STOP_UTC of the step, hours",
          rfsel="July runs at RFSel %d by the display-run list (the extra run, H1 19, repeats run "
                "18 at its settings); March runs record no RFSel and ran at RFSel %d"
                % (SETTINGS["rfsel"], campaign.MARCH_RFSEL)),
     inputs=INPUTS_TRACKS)

# %% [markdown]
# ## Figure fe08: median and most frequent TOT against fluence
#
# One run per step (March: the run of each step; July: the preferred display run). Filled: the
# statistic of fe07, with its bar. Open: the most frequent TOT of the board, every event of its
# counted pixels together in 0.1 ns bins (the TOT is reconstructed per event, so there is no
# common code grid at board level; the mode is the centre of the fullest 0.1 ns bin).

# %%
STEPS_08 = [dict(campaign=st["campaign"], fluence=st["fluence"],
                 runs={t: [preferred_of(st, t)] for t in TELS}) for st in LADDER]
ticks08 = {t: [style.fluence_label(st["fluence"], unit=False)
               + ("\n" + step_time(st, t) if st["fluence"] else "") for st in STEPS_08]
           for t in TELS}
pts08 = []
for i, st in enumerate(STEPS_08):
    for t in TELS:
        run = st["runs"][t][0]
        for b in range(NBOARDS[t]):
            p = tot_point(st["campaign"], t, run, b, "pixmed", i, dx=-0.13)
            s = SUMMARY.loc[(st["campaign"], t, run, b)]
            pts08 += [p, dict(p, kind="mode", value=round(float(s["mode_ns_0.10"]), 4), lo=None,
                              hi=None, dx=0.13)]
npix08 = (min(p["n_pixels"] for p in pts08), max(p["n_pixels"] for p in pts08))
fig = tot_shapes.board_vs_fluence(
    pts08, {t: STEPS_08 for t in TELS},
    {"pixmed": dict(label="median of the pixel TOT medians, bar 16-84 %", line=True),
     "mode": dict(label="most frequent TOT, all events, 0.1 ns bins", open=True)},
    tick_labels=ticks08, xlabel=XLABEL_LADDER, ylabel="TOT of the board [ns]",
    tel_titles=TEL_TITLES, tag=SETTINGS_TEXT, data=None, board_dx=0.06,
    boundary_after=sum(st["campaign"] == "march" for st in STEPS_08) - 1,
    footer=(u"Filled, joined: median over the board's pixels of each pixel's median TOT; bar: "
            u"16-84 %% of those pixel medians (spread, not an uncertainty); %d-%d pixels per "
            u"board.\nOpen: the most frequent TOT of every event of the board's counted pixels "
            u"together, centre of the fullest 0.1 ns bin.\n%s\nRanking by file size instead "
            u"moves the medians by at most %s and the most frequent TOT of %d board in %d by one "
            u"bin.\n%s\nOne run per step: March, the run of each step; July, the preferred "
            u"display run (%s). Dotted line: March | July. Tick: time from the end of the step's "
            u"irradiation to the run start."
            % (npix08 + (TOT_RULE, RANK_SHIFT_TEXT, RANK_SHIFT["mode_boards_moved"],
                         RANK_SHIFT["boards"], CORRIDOR_TEXT, SET_WORDS))))
save(fig, "fe08_tot_board_vs_fluence",
     dict(points=pts08, steps=STEPS_08, tick_labels=ticks08, pixels_per_board=list(npix08),
          rank_shift=RANK_SHIFT),
     dict(fluence=campaign.FLUENCE_CONVENTION,
          pixmed="median over the counted pixels of each pixel's median TOT (ns); bar 16th-84th "
                 "percentile of the pixel medians",
          mode="centre of the fullest 0.1 ns bin of every event of the counted pixels (ns)",
          pixels="as fe07", triple_rule="as fe07", rfsel="as fe07"),
     inputs=INPUTS_TRACKS)

# %% [markdown]
# ## Figures fe09-fe10: TOT distributions at every step
#
# fe09: every event of every counted pixel of a board together; fe10: one pixel per board, the
# same pixel in every run (among the pixels counted in all seven runs, the one whose smallest
# event count over the runs is largest). One curve per run, each as the fraction of its events
# per bin: fe09 in 0.1 ns bins; fe10 one bin per TOT code, on the code ladder of the pixel's CAL
# mode in the run (one code = 2 x 3.125 ns / CAL mode, the code at each multiple of 32 half of
# that), so the edges differ run to run. One x range per telescope (0.5 to 99.5 % of
# its curves). The legend gives the step,
# the time from the end of its irradiation to the run start (the July 1.5e15 run after cooling
# down, timed from the end of the March 1.5e15 step) and the board's bias.

# %%
TOT_W, DT_W = 0.1, tot_shapes.DT_BIN_PS
SHAPE_RUNS_TEXT = (u"One run per step: March, the run of each step; July, the preferred display "
                   u"run (%s). Legend: step, time from the end of its irradiation to the run "
                   u"start (July 1.5e15: from the end of the March step, after cooling down), "
                   u"bias." % SET_WORDS)


def pooled_tot(camp, tel, run, b):
    e, c = tot_shapes.hist_of(TOT_POOLED[(camp, tel, run, b)], "bin_lo_ns",
                              tot_shapes.TOT_BIN_NS, 0.0, tot_shapes.TOT_HI_NS)
    return tot_shapes.rebin(e, c, TOT_W)


def pixel_tot(pick, camp, tel, run, b):
    """A pixel's TOT, one bin per TOT code from its lowest code to its highest: (edges in ns,
    counts, CAL mode, its tot_ns row). The edges are the code ladder at the pixel's CAL mode in
    the run (tot_shapes.code_edges_ns); the median and the events outside come from the tot_ns
    row, the TOT in ns of the same events."""
    d = PIXEL_HISTS[(pick, "tot_code", camp, tel, run, b, -1)]
    row = PIXEL_HISTS[(pick, "tot_ns", camp, tel, run, b, -1)].iloc[0]
    code = d["bin_lo"].to_numpy(int)
    c = np.zeros(code.max() - code.min() + 1, dtype=np.int64)
    np.add.at(c, code - code.min(), d["count"].to_numpy(np.int64))
    (mode,) = d["cal_mode"].astype(int).unique()
    if c.sum() != int(row["n_events"]) or int(d["n_events"].iloc[0]) != int(row["n_events"]):
        raise ValueError("%s %s %s run %d board %d: %d events in the TOT codes, %d in ns" % (
            pick, camp, tel, run, b, c.sum(), row["n_events"]))
    return (tot_shapes.code_edges_ns(code.min(), code.max(), mode), c, int(mode), row)


def tot_code_ps(cal_mode):
    """One TOT code in ps at a CAL mode: 2 x 3.125 ns / CAL mode."""
    return round(2 * tot_shapes.TOA_CODE_PS / cal_mode, 1)


def code_bins_text(modes):
    """The width of a TOT code over the CAL modes drawn, for the footers."""
    lo, hi = min(modes), max(modes)
    if lo == hi:
        cal, width = "%d" % lo, "%.1f" % tot_code_ps(lo)
    else:
        cal, width = "%d-%d" % (lo, hi), "%.1f-%.1f" % (tot_code_ps(hi), tot_code_ps(lo))
    return (u"2 x 3.125 ns / CAL mode (CAL mode %s: %s ps), except the code at each multiple of "
            u"32, half of that" % (cal, width))


panels, ranges, rec09 = {}, {}, []
for t in TELS:
    panels[t], hs = [], []
    for b, chip in enumerate(campaign.TELESCOPE_CHIPS[t]):
        curves, npix = [], []
        for st in SHAPE[t]:
            s = SUMMARY.loc[(st["campaign"], t, st["run"], b)]
            e, c = pooled_tot(st["campaign"], t, st["run"], b)
            curves.append(dict(label=step_label(st, t, [b]), color=st["color"], edges=e,
                               y=c / float(s["n_events"])))
            hs.append((e, c))
            npix.append(int(s["n_pixels"]))
            rec09.append(dict(tel=t, board_idx=b, chip=chip, campaign=st["campaign"],
                              run=st["run"], n_pixels=int(s["n_pixels"]),
                              n_events=int(s["n_events"]),
                              n_outside=int(s["n_outside_hist"]),
                              median_ns=round(float(s["median_ns"]), 4),
                              mode_ns=round(float(s["mode_ns_0.10"]), 4),
                              pixmed_median_ns=round(float(s["pixmed_median_ns"]), 4)))
        panels[t].append(dict(subject="%s, %s" % (t.upper(), chip), curves=curves,
                              note="pixels %d-%d" % (min(npix), max(npix))))
    ranges[t] = tot_shapes.common_range(hs, TOT_W, floor=0.0)
fig = tot_shapes.shape_grid(
    panels, xlabel="TOT [ns]", ylabel="Fraction of events / 0.1 ns", ranges=ranges,
    tag=SETTINGS_TEXT, data=None,
    footer=(u"Every event of every counted pixel of the board together, 0.1 ns bins, as a "
            u"fraction of those events; pixels per board in each legend title.\n%s\n%s"
            % (TOT_RULE, SHAPE_RUNS_TEXT)))
save(fig, "fe09_tot_shape_board_vs_fluence",
     dict(curves=rec09, ranges=ranges, runs={t: SHAPE[t] for t in TELS}),
     dict(fluence=campaign.FLUENCE_CONVENTION, bins="0.1 ns, fraction of the board's events",
          pixels="as fe07", median_ns="median of every event of the counted pixels",
          mode_ns="centre of the fullest 0.1 ns bin", range="0.5-99.5 % of the curves, per "
          "telescope"),
     inputs=INPUTS_TRACKS)

# %%
panels, ranges, rec10, cand10 = {}, {}, [], {}
for t in TELS:
    panels[t], hs = [], []
    runs = [(st["campaign"], st["run"]) for st in SHAPE[t]]
    for b, chip in enumerate(campaign.TELESCOPE_CHIPS[t]):
        px = stored_pixel("pixel", "tot_ns", t, runs, b)
        mine, counts, ncand = fixed_pixel(t, b, runs)
        if mine != px:
            raise ValueError("%s board %d: pixel %s in the table, %s by the rule"
                             % (t, b, px, mine))
        cand10["%s %s" % (t, chip)] = ncand
        curves, ns = [], []
        for st in SHAPE[t]:
            e, c, mode, row = pixel_tot("pixel", st["campaign"], t, st["run"], b)
            curves.append(dict(label=step_label(st, t, [b]), color=st["color"], edges=e,
                               y=c / float(row["n_events"])))
            hs.append((e, c))
            ns.append(int(row["n_events"]))
            rec10.append(dict(tel=t, board_idx=b, chip=chip, row=px[0], col=px[1],
                              campaign=st["campaign"], run=st["run"],
                              n_events=int(row["n_events"]), n_outside=int(row["n_outside"]),
                              median_ns=round(float(row["median"]), 4), file=row["file"],
                              cal_mode=mode, code_width_ps=tot_code_ps(mode)))
        panels[t].append(dict(subject="%s, %s" % (t.upper(), chip), curves=curves,
                              note="pixel (%d, %d), N %d-%d" % (px + (min(ns), max(ns)))))
    ranges[t] = tot_shapes.common_range(hs, TOT_W, floor=0.0)
print("candidates per board:", cand10)
fig = tot_shapes.shape_grid(
    panels, xlabel="TOT [ns]", ylabel="Fraction of events / TOT code", ranges=ranges,
    tag=SETTINGS_TEXT, data=None,
    footer=(u"One pixel per board, the same in every run: among the pixels whose triple holds "
            u">= %d events in all seven runs (%d-%d candidates per board), the one whose "
            u"smallest event count over the runs is largest.\nPer run its TOT from its triple, "
            u"the one with the most events in its board's combination (%s), as a fraction of "
            u"its events.\nBins: one TOT code = %s; the edges differ run to run by the run's CAL "
            u"mode.\n%s"
            % (FLOOR, min(cand10.values()), max(cand10.values()), COMBO_WORDS,
               code_bins_text([r["cal_mode"] for r in rec10]), SHAPE_RUNS_TEXT)))
save(fig, "fe10_tot_shape_pixel_vs_fluence",
     dict(curves=rec10, candidates=cand10, ranges=ranges, runs={t: SHAPE[t] for t in TELS}),
     dict(fluence=campaign.FLUENCE_CONVENTION,
          bins="one TOT code, fraction of the pixel's events; edges tot_shapes.code_edges_ns at "
               "cal_mode (pixel_hists.csv tot_code rows)",
          cal_mode="the most frequent CAL code of the pixel's events in its triple, per run",
          code_width_ps="one TOT code, 2 x 3125 ps / cal_mode; the code at each multiple of 32 "
                        "is half of it",
          pixel="per board, among the pixels >= %d events in every run drawn, the largest "
                "smallest count; the same pixel in every run" % FLOOR,
          median_ns="median TOT of the pixel's events in its triple (tot_ns rows, unbinned)"),
     inputs=INPUTS_TRACKS)

# %% [markdown]
# ## Figures fe11-fe12: TOA difference distributions at every step
#
# The TOA difference of two boards after the time-walk correction, centred on each triple's own
# median. fe11: every counted triple of the first board's pixels pooled (pairs 0-1 and 0-2 from
# board 0's triples, 1-2 from board 1's, 2-3 from board 2's triples of boards 1-2-3); fe12: one
# pixel pair per board pair, the same in every run. 25 ps bins: one TOA code is 3.125 ns / CAL
# (CAL about 140 to 164 codes), so one bin is about one code and never finer than the coarsest.

# %%
PAIRS = tot_shapes.PAIRS
# the dTOA curves peak in the middle, under the legend: a higher y scale and a smaller legend
DT_GRID = dict(ytop=1.9, legend_scale=0.88)


def pair_text(pair):
    return "%d-%d" % pair


def pooled_dt(camp, tel, run, pair):
    return tot_shapes.hist_of(DTOA_POOLED[(camp, tel, run, pair_text(pair))], "bin_lo_ps",
                              DT_W, -tot_shapes.DT_HALF_PS, tot_shapes.DT_HALF_PS)


cal_runs = dtoa[[(c, r) in {(st["campaign"], st["run"]) for st in SHAPE[t]}
                 for c, t, r in zip(dtoa["campaign"], dtoa["tel"], dtoa["run"])]]
cal = pd.concat([cal_runs["cal_a"], cal_runs["cal_b"]])
CODE_TEXT = (u"One TOA code is 3.125 ns / CAL = %.1f-%.1f ps over these runs (triple CAL medians "
             u"%.0f-%.0f), so one 25 ps bin is about one code."
             % (tot_shapes.TOA_CODE_PS / cal.max(), tot_shapes.TOA_CODE_PS / cal.min(),
                cal.min(), cal.max()))
print(CODE_TEXT)
DT_TEXT = (u"dTOA = TOA(first chip) - TOA(second chip) after a time-walk correction solved jointly "
           u"over the three boards of the triple, centred on the triple's own median.")
panels, ranges, rec11 = {}, {}, []
for t in TELS:
    chips = campaign.TELESCOPE_CHIPS[t]
    panels[t], hs = [], []
    for pair in PAIRS:
        curves, ntr = [], []
        for st in SHAPE[t]:
            sel = dtoa[(dtoa["campaign"] == st["campaign"]) & (dtoa["tel"] == t)
                       & (dtoa["run"] == st["run"]) & (dtoa["pair"] == pair_text(pair))]
            e, c = pooled_dt(st["campaign"], t, st["run"], pair)
            n = int(sel["n_events"].sum())
            curves.append(dict(label=step_label(st, t, list(pair)), color=st["color"], edges=e,
                               y=c / float(n)))
            hs.append((e, c))
            ntr.append(len(sel))
            rec11.append(dict(tel=t, pair=pair_text(pair), chips=[chips[pair[0]], chips[pair[1]]],
                              campaign=st["campaign"], run=st["run"], n_triples=len(sel),
                              n_events=n, n_outside=int(sel["n_outside"].sum())))
        panels[t].append(dict(subject="%s, %s - %s" % (t.upper(), chips[pair[0]], chips[pair[1]]),
                              curves=curves, note="triples %d-%d" % (min(ntr), max(ntr))))
    ranges[t] = tot_shapes.common_range(hs, DT_W, symmetric=True)
fig = tot_shapes.shape_grid(
    panels, xlabel="dTOA [ps]", ylabel="Fraction of events / 25 ps", ranges=ranges,
    tag=SETTINGS_TEXT, data=None,
    footer=(u"%s Every triple of the first chip's counted pixels pooled (triples per run in each "
            u"legend title): pairs 0-1 and 0-2 from board 0's triples, 1-2 from board 1's, 2-3 "
            u"from board 2's triples of boards 1-2-3.\n%s 25 ps bins, as a fraction of the "
            u"events. %s\n%s" % (DT_TEXT, TOT_RULE, CODE_TEXT, SHAPE_RUNS_TEXT)), **DT_GRID)
save(fig, "fe11_dtoa_shape_board_vs_fluence",
     dict(curves=rec11, ranges=ranges, code_width_ps=[round(tot_shapes.TOA_CODE_PS / cal.max(), 2),
                                                      round(tot_shapes.TOA_CODE_PS / cal.min(), 2)],
          runs={t: SHAPE[t] for t in TELS}),
     dict(fluence=campaign.FLUENCE_CONVENTION,
          dtoa="TOA(a) - TOA(b) after the joint time-walk correction of the triple (three boards), "
               "minus the triple's median, ps",
          bins="25 ps over +-2000 ps, fraction of the pair's events; n_outside beyond +-2000 ps",
          pixels="as fe07, by the first board's pixel"),
     inputs=INPUTS_TRACKS)

# %%
panels, ranges, rec12 = {}, {}, []
for t in TELS:
    chips = campaign.TELESCOPE_CHIPS[t]
    runs = [(st["campaign"], st["run"]) for st in SHAPE[t]]
    panels[t], hs = [], []
    for pair in PAIRS:
        curves, ns, pxb = [], [], set()
        pxa = stored_pixel("pixel_pair", "dtoa_ps", t, runs, pair[0], pair[1])
        for st in SHAPE[t]:
            d = PIXEL_HISTS[("pixel_pair", "dtoa_ps", st["campaign"], t, st["run"], pair[0],
                             pair[1])]
            row = d.iloc[0]
            pxb.add((int(row["row_b"]), int(row["col_b"])))
            e, c = tot_shapes.hist_of(d, "bin_lo", DT_W, -tot_shapes.DT_HALF_PS,
                                      tot_shapes.DT_HALF_PS)
            curves.append(dict(label=step_label(st, t, list(pair)), color=st["color"], edges=e,
                               y=c / float(row["n_events"])))
            hs.append((e, c))
            ns.append(int(row["n_events"]))
            rec12.append(dict(tel=t, pair=pair_text(pair), campaign=st["campaign"],
                              run=st["run"], pixel_a=list(pxa),
                              pixel_b=[int(row["row_b"]), int(row["col_b"])],
                              n_events=int(row["n_events"]), n_outside=int(row["n_outside"]),
                              dt_median_ps=float(row["median"]), file=row["file"]))
        if len(pxb) != 1:
            raise ValueError("%s pair %s: board-b pixels %s differ" % (t, pair, pxb))
        pb = pxb.pop()
        panels[t].append(dict(subject="%s, %s - %s" % (t.upper(), chips[pair[0]], chips[pair[1]]),
                              curves=curves, note="(%d, %d) - (%d, %d), N %d-%d"
                              % (pxa + pb + (min(ns), max(ns)))))
    ranges[t] = tot_shapes.common_range(hs, DT_W, symmetric=True)
fig = tot_shapes.shape_grid(
    panels, xlabel="dTOA [ps]", ylabel="Fraction of events / 25 ps", ranges=ranges,
    tag=SETTINGS_TEXT, data=None,
    footer=(u"%s One pixel pair per board pair, the same in every run: pairs 0-1, 0-2 and 1-2 "
            u"join the single pixels of the TOT shapes\n(per board, the largest smallest event "
            u"count over the seven runs, >= %d in each); pair 2-3 keeps board 3's pixel and "
            u"takes the board-2 pixel tracked with it in every run whose best triple has the "
            u"largest smallest count.\nPer run the triple with the most events holding both "
            u"pixels. 25 ps bins, as a fraction of the events. Legend title: pixels (row, col) "
            u"and events per run.\n%s" % (DT_TEXT, FLOOR, SHAPE_RUNS_TEXT)), **DT_GRID)
save(fig, "fe12_dtoa_shape_pixel_vs_fluence",
     dict(curves=rec12, ranges=ranges, runs={t: SHAPE[t] for t in TELS}),
     dict(fluence=campaign.FLUENCE_CONVENTION, dtoa="as fe11; dt_median_ps is the median the "
          "curve was centred on", bins="25 ps over +-2000 ps, fraction of the pair's events",
          pixels="pairs 0-1, 0-2, 1-2: the fixed pixels of fe10; 2-3: board 3's fixed pixel and "
                 "the board-2 pixel tracked with it in every run with the largest smallest count"),
     inputs=INPUTS_TRACKS)

# %% [markdown]
# ## Figures fe13-fe14: one pixel through the July steps
#
# One board-1 pixel per telescope, the same in the three July runs (among the pixels counted in
# all three, the one whose smallest event count is largest): its TOT and its TOA difference with
# board 2 from its triple with the most events in each run, in events per bin. The board-2 pixel
# is the one in that triple. Bins: TOT one TOT code, on the code ladder of the pixel's CAL mode in
# the run as in fe10; dTOA 25 ps, the bin the table stores and about one TOA code of these two
# pixels (3.125 ns / CAL, with each board's median CAL in the triple from `dtoa_triples.csv`), as
# in fe11-fe12. At 100 ps, the bin of the CERN IRRAD 2026 talk, the curve fills four bins.

# %%
for k, t in enumerate(TELS):
    chips = campaign.TELESCOPE_CHIPS[t]
    july = [st for st in SHAPE[t] if st["campaign"] == "july"]
    runs = [(st["campaign"], st["run"]) for st in july]
    px = stored_pixel("july_pixel", "tot_ns", t, runs, 1)
    mine, counts, ncand = fixed_pixel(t, 1, runs)
    if mine != px:
        raise ValueError("%s: July pixel %s in the table, %s by the rule" % (t, px, mine))
    columns, rec, hs, cals = [], [], {"tot": [], "dtoa": []}, []
    for st in july:
        e, c, mode, row = pixel_tot("july_pixel", "july", t, st["run"], 1)
        d = PIXEL_HISTS[("july_pixel", "dtoa_ps", "july", t, st["run"], 1, 2)]
        drow = d.iloc[0]
        ed, cd = tot_shapes.hist_of(d, "bin_lo", DT_W, -tot_shapes.DT_HALF_PS,
                                    tot_shapes.DT_HALF_PS)
        # the same triple in the dTOA table (pair 1-2 of board 1's triple): its CAL medians
        tr = dtoa[(dtoa["campaign"] == "july") & (dtoa["tel"] == t) & (dtoa["run"] == st["run"])
                  & (dtoa["pair"] == "1-2") & (dtoa["row_1"] == px[0]) & (dtoa["col_1"] == px[1])]
        if len(tr) != 1 or (int(tr["row_2"].iloc[0]), int(tr["col_2"].iloc[0]),
                            int(tr["n_events"].iloc[0])) != (int(drow["row_b"]),
                                                             int(drow["col_b"]),
                                                             int(drow["n_events"])):
            raise ValueError("%s run %d: the July pixel's triple is not the dTOA table's" % (
                t, st["run"]))
        cal_ab = [float(tr["cal_a"].iloc[0]), float(tr["cal_b"].iloc[0])]
        cals += cal_ab
        hs["tot"].append((e, c))
        hs["dtoa"].append((ed, cd))
        columns.append(dict(
            subject="Run %d, %s %s, %s" % (st["run"], style.fluence_label(st["fluence"], unit=False),
                                           campaign.FLUENCE_UNIT, st["time"]),
            color=st["color"], hists={"tot": (e, c), "dtoa": (ed, cd)},
            notes={"tot": "N = %d" % row["n_events"],
                   "dtoa": "%s pixel (%d, %d)\nN = %d" % (chips[2], drow["row_b"], drow["col_b"],
                                                          drow["n_events"])}))
        rec.append(dict(run=st["run"], hours=st["hours"], n_events=int(row["n_events"]),
                        tot_median_ns=round(float(row["median"]), 4),
                        tot_outside=int(row["n_outside"]), pixel_b=[int(drow["row_b"]),
                                                                    int(drow["col_b"])],
                        dt_events=int(drow["n_events"]), dt_median_ps=float(drow["median"]),
                        dt_outside=int(drow["n_outside"]), cal_medians=cal_ab, file=row["file"],
                        cal_mode=mode, tot_code_width_ps=tot_code_ps(mode)))
    code_ps = [round(tot_shapes.TOA_CODE_PS / max(cals), 1),
               round(tot_shapes.TOA_CODE_PS / min(cals), 1)]
    rows = [dict(key="tot", xlabel="TOT, %s pixel (%d, %d) [ns]" % ((chips[1],) + px),
                 ylabel="Events / TOT code", range=tot_shapes.common_range(hs["tot"], TOT_W,
                                                                            floor=0.0)),
            dict(key="dtoa", xlabel="dTOA, %s - %s [ps]" % (chips[1], chips[2]),
                 ylabel="Events / 25 ps", zero_line=True,
                 range=tot_shapes.common_range(hs["dtoa"], DT_W, symmetric=True))]
    fig = tot_shapes.run_columns(
        columns, rows, tag=SETTINGS_TEXT, data=None,
        footer=(u"One %s pixel, the same in the three July runs: among the %d pixels whose "
                u"triple holds >= %d events in all three, the one whose smallest event count is "
                u"largest. Per run its triple with the most events (boards 0-1-2); its TOT, and "
                u"dTOA = TOA(%s) - TOA(%s)\nafter a time-walk correction solved jointly over the "
                u"triple's three boards, centred on its median; events per bin.\nBins: TOT one "
                u"code = %s; dTOA 25 ps, about one TOA code (3.125 ns / CAL = %.1f-%.1f ps for "
                u"these two pixels).\nJuly preferred display runs (%s); time from the end of the "
                u"step's irradiation to the run start (1.5e15: from the end of the March step, "
                u"after cooling down)."
                % (chips[1], ncand, FLOOR, chips[1], chips[2],
                   code_bins_text([r["cal_mode"] for r in rec]), code_ps[0], code_ps[1],
                   SET_WORDS)))
    save(fig, "fe%02d_tot_dtoa_shape_vs_fluence_jul_%s" % (13 + k, t),
         dict(pixel=list(px), board=1, chip=chips[1], candidates=ncand,
              counts_at_pick=counts, runs=rec, code_width_ps=code_ps),
         dict(fluence=campaign.FLUENCE_CONVENTION,
              pixel="board 1, among the pixels >= %d events in all three July runs, the largest "
                    "smallest count" % FLOOR,
              tot="one TOT code, events; edges tot_shapes.code_edges_ns at cal_mode "
                  "(pixel_hists.csv tot_code rows)",
              cal_mode="the most frequent CAL code of the pixel's events in its triple, per run",
              tot_code_width_ps="one TOT code, 2 x 3125 ps / cal_mode; the code at each "
                                "multiple of 32 is half of it",
              dtoa="as fe11, 25 ps bins (as stored), events",
              code_width_ps="3.125 ns / CAL over the median CAL of both boards of the triple in "
                            "the three runs (dtoa_triples.csv cal_a, cal_b)",
              times="run start minus RAD_STOP_UTC of the step, hours"),
         inputs=INPUTS_TRACKS)

# %% [markdown]
# ## Figure fe15: the TOT at two RFSel settings
#
# What the preamp feedback resistor R_f (RFSel) does to the TOT at a fixed threshold offset,
# from the pairs of runs that differ in RFSel only (`RFSEL_TOT_PAIRS` in the campaign module:
# July F1 at 3.5e15, run 47 at RFSel 1 against run 18 at RFSel 2, and run 52 at RFSel 2 against
# run 86 at RFSel 3). One chip, the one whose smallest total of counted events over the four runs
# is largest (`rfsel_pair_hist.csv` holds it; this cell checks the choice against the per-pixel
# table). Per pair, the pixels of that chip counted in both runs, every event of each pixel's
# triple. The RFSel, threshold offset and bias of each run come from `tables/runs_summary.csv`.
#
# Control: each curve must be its run's pooled TOT of the chip (`tot_pooled_hist.csv`, the table
# of fe08-fe09) less the pixels not counted in the other run of its pair. The cell checks that the
# curve is nowhere above the pooled TOT, that the difference holds exactly those pixels' events,
# and that the curve holds the events the table states. One run of
# the pairs is also the preferred display run of its step (run 18, the 3.5e15 run of fe08-fe09);
# its median over all its counted pixels, from `tot_pooled_summary.csv`, is stated on the figure.
#
# Colour = RFSel here (the fluence is one step). 0.1 ns bins, each curve as the fraction of its
# events; a short tick on the x axis marks each curve's median.

# %%
INPUTS_RFSEL = [campaign.RFSEL_PAIR_HIST_CSV, campaign.TOT_POOLED_HIST_CSV,
                campaign.TOT_POOLED_SUMMARY_CSV, campaign.TOT_PIXEL_MEDIANS_CSV,
                campaign.RUNS_SUMMARY_CSV, campaign.RUN_TIMES_CSV, campaign.DISPLAY_RUNS_JUL_CSV]
rfsel_hist = pd.read_csv(campaign.RFSEL_PAIR_HIST_CSV)
if len(campaign.RFSEL_TOT_PAIRS) != 1:
    raise ValueError("fe15 draws the RFSel pairs of one telescope; RFSEL_TOT_PAIRS has %d"
                     % len(campaign.RFSEL_TOT_PAIRS))
((RF_CAMP, RF_TEL), RF_PAIRS), = campaign.RFSEL_TOT_PAIRS.items()
RF_RUNS = sorted({r for p in RF_PAIRS for r in p})
(RF_BOARD,) = rfsel_hist["board_idx"].unique()
RF_BOARD = int(RF_BOARD)
RF_CHIP = campaign.TELESCOPE_CHIPS[RF_TEL][RF_BOARD]


def counted(camp, tel, run, b):
    """{(row, col): events of its triple} for the counted pixels of board b in a run."""
    p = pix_medians[(pix_medians["campaign"] == camp) & (pix_medians["tel"] == tel)
                    & (pix_medians["run"] == run) & (pix_medians["board_idx"] == b)
                    & pix_medians["passes_floor"]]
    return {(int(r), int(c)): int(n) for r, c, n in zip(p["row"], p["col"], p["n"])}


def board_settings(camp, tel, run, b):
    (s,) = [x for x in tables.run_settings(RS, campaign.TABLE_CAMPAIGN[camp],
                                           campaign.TABLE_TEL[tel], run) if x["board_idx"] == b]
    return s


# the chip: the board whose smallest total of counted events over the runs is largest (ties to
# the first chip name), as the rebuild chose it
chips = campaign.TELESCOPE_CHIPS[RF_TEL]
RF_TOTALS = {chips[b]: {r: sum(counted(RF_CAMP, RF_TEL, r, b).values()) for r in RF_RUNS}
             for b in range(NBOARDS[RF_TEL])}
best = max(sorted(RF_TOTALS), key=lambda c: min(RF_TOTALS[c].values()))
if best != RF_CHIP:
    raise ValueError("RFSel chip %s in the table, %s by the rule" % (RF_CHIP, best))
RF_FLUENCE = {step_fluence(RF_CAMP, RF_TEL, r) for r in RF_RUNS}
if len(RF_FLUENCE) != 1:
    raise ValueError("RFSel runs at different fluences: %s" % RF_FLUENCE)
(RF_FLUENCE,) = RF_FLUENCE

panels15, rec15, hs15, control15, hv15 = [], [], [], [], []
for pair in RF_PAIRS:
    curves, npix, offsets = [], set(), set()
    for run, partner in ((pair[0], pair[1]), (pair[1], pair[0])):
        g = rfsel_hist[(rfsel_hist["campaign"] == RF_CAMP) & (rfsel_hist["tel"] == RF_TEL)
                       & (rfsel_hist["run"] == run) & (rfsel_hist["partner_run"] == partner)]
        row = g.iloc[0]
        s = board_settings(RF_CAMP, RF_TEL, run, RF_BOARD)
        if (s["rfsel"], s["offset"]) != (int(row["rfsel"]), int(row["offset"])):
            raise ValueError("run %d: RFSel / offset %s in the run summary, %s in the table" % (
                run, (s["rfsel"], s["offset"]), (int(row["rfsel"]), int(row["offset"]))))
        e, c = tot_shapes.hist_of(g, "bin_lo_ns", tot_shapes.TOT_BIN_NS, 0.0, tot_shapes.TOT_HI_NS)
        # positive control: the chip's pooled TOT in this run less the pixels not counted in the
        # partner run is this curve, bin by bin
        mine, other = counted(RF_CAMP, RF_TEL, run, RF_BOARD), counted(RF_CAMP, RF_TEL, partner,
                                                                         RF_BOARD)
        left = sorted(set(mine) - set(other))
        summ = SUMMARY.loc[(RF_CAMP, RF_TEL, run, RF_BOARD)]
        _, c_all = tot_shapes.hist_of(TOT_POOLED[(RF_CAMP, RF_TEL, run, RF_BOARD)], "bin_lo_ns",
                                      tot_shapes.TOT_BIN_NS, 0.0, tot_shapes.TOT_HI_NS)
        ctl = dict(run=int(run), partner_run=int(partner), board_pixels=int(summ["n_pixels"]),
                   board_events=int(summ["n_events"]), pixels_left_out=len(left),
                   events_left_out=sum(mine[p] for p in left),
                   common_pixels=len(set(mine) & set(other)),
                   min_bin_difference=int((c_all - c).min()),
                   difference_events=int((c_all - c).sum()), curve_events=int(c.sum()))
        if (ctl["min_bin_difference"] < 0 or ctl["common_pixels"] != int(row["n_pixels"])
                or ctl["difference_events"] != ctl["events_left_out"]
                or ctl["curve_events"] != int(row["n_events"])
                or ctl["board_pixels"] - ctl["pixels_left_out"] != int(row["n_pixels"])
                or ctl["board_events"] - ctl["events_left_out"] != int(row["n_events"])):
            raise ValueError("run %d: the RFSel curve is not the pooled TOT less the pixels left "
                             "out: %s, table %d pixels %d events" % (run, ctl, row["n_pixels"],
                                                                     row["n_events"]))
        control15.append(ctl)
        text, hours = blnw.time_after_stop(RUN_TIMES, RF_CAMP, RF_TEL, [run],
                                           campaign.RAD_STOP_UTC[RF_FLUENCE])
        e10, c10 = tot_shapes.rebin(e, c, TOT_W)
        hs15.append((e10, c10))
        med = float(row["median_ns"])
        curves.append(dict(rfsel=s["rfsel"], color=tot_shapes.RFSEL_COLOR[s["rfsel"]], edges=e10,
                           y=c10 / float(row["n_events"]), median=med,
                           label=u"%s, run %d, %s\nmedian %.2f ns, N = %d"
                                 % (style.rfsel_text(s["rfsel"]), run, text, med,
                                    int(row["n_events"]))))
        rec15.append(dict(pair=list(pair), run=int(run), partner_run=int(partner),
                          chip=RF_CHIP, board_idx=RF_BOARD, rfsel=s["rfsel"],
                          rf_kohm=style.RFSEL_RF_KOHM[s["rfsel"]], offset=s["offset"],
                          hv=s["hv"], hours=round(hours[run], 2), n_pixels=int(row["n_pixels"]),
                          n_events=int(row["n_events"]), median_ns=round(med, 4)))
        npix.add(int(row["n_pixels"]))
        offsets.add(s["offset"])
        hv15.append(s["hv"])
    if len(npix) != 1 or len(offsets) != 1:
        raise ValueError("pair %s: pixels %s, offsets %s" % (pair, npix, offsets))
    curves.sort(key=lambda cv: cv["rfsel"])
    panels15.append(dict(subject=style.settings_text(rfsel=[cv["rfsel"] for cv in curves],
                                                     offset=offsets.pop()),
                         note="%s, the %d pixels counted in both runs" % (RF_CHIP, npix.pop()),
                         curves=curves))
# the run of the pairs that is also a preferred display run: its board median as fe08-fe09 use it
ctl_runs = [r for r in RF_RUNS if r in july_pref[RF_TEL]] if RF_CAMP == "july" else []
POS_CONTROL = [dict(run=int(r), chip=RF_CHIP, n_pixels=int(SUMMARY.loc[(RF_CAMP, RF_TEL, r,
                                                                         RF_BOARD)]["n_pixels"]),
                    n_events=int(SUMMARY.loc[(RF_CAMP, RF_TEL, r, RF_BOARD)]["n_events"]),
                    median_ns=round(float(SUMMARY.loc[(RF_CAMP, RF_TEL, r, RF_BOARD)]["median_ns"]),
                                    4)) for r in ctl_runs]
for x in rec15:
    print("pair %s: run %d %s RFSel %d offset %d %s: %d pixels, %d events, median %.4f ns" % (
        x["pair"], x["run"], x["chip"], x["rfsel"], x["offset"], blnw.bias_text([x["hv"]]),
        x["n_pixels"], x["n_events"], x["median_ns"]))
print("control (pooled TOT less the pixels left out):", control15)
print("positive control:", POS_CONTROL)
print("events above the floor per chip and run:", RF_TOTALS)
pc_text = "; ".join(u"Run %d is also the %s run of the TOT figures against fluence, where its %d "
                    u"counted %s pixels give a median of %.4f ns" % (
                        p["run"], campaign.FLUENCE_TEXT_PLAIN[RF_FLUENCE], p["n_pixels"],
                        p["chip"], p["median_ns"]) for p in POS_CONTROL)
fig = tot_shapes.rfsel_pairs(
    panels15, xlabel="TOT [ns]", ylabel="Fraction of events / 0.1 ns",
    xrange=tot_shapes.common_range(hs15, TOT_W, floor=0.0), tag=None,
    data="%s, %s, %s, %s" % (campaign.TELESCOPE_TITLE[RF_TEL], RF_CHIP,
                             style.fluence_label(RF_FLUENCE), RF_CAMP.capitalize()),
    footer=(u"%s\nPer pair, the %s pixels counted in both runs, every event of each; %s is the "
            u"chip whose smallest total of counted events over the four runs is largest. Bias of "
            u"%s: %s in every run.\n0.1 ns bins, each curve as the fraction of its events; tick "
            u"on the axis: the median. Legend: RFSel, run, time from the end of the step's "
            u"irradiation to the run start, median TOT and events.\nControl: in every bin each "
            u"curve is at most its run's pooled TOT of %s, and the difference holds exactly the "
            u"events of the pixels not counted in the other run.%s"
            % (TOT_RULE, RF_CHIP, RF_CHIP, RF_CHIP, blnw.bias_text(hv15), RF_CHIP,
               ("\n" + pc_text + ".") if pc_text else "")))
save(fig, "fe15_rfsel_tot_pairs",
     dict(curves=rec15, chip=RF_CHIP, board_idx=RF_BOARD, telescope=RF_TEL, campaign=RF_CAMP,
          chip_choice=RF_TOTALS, control=control15, positive_control=POS_CONTROL),
     dict(fluence=campaign.FLUENCE_CONVENTION,
          pairs="RFSEL_TOT_PAIRS of the campaign module: runs that differ in RFSel only",
          settings="RFSel, threshold offset and bias of the chip per run from runs_summary.csv",
          pixels="per pair, the chip's pixels counted (triple >= %d events) in both runs; every "
                 "event of each pixel's triple" % FLOOR,
          chip="the board whose smallest total of counted events over the four runs is largest",
          bins="0.1 ns (summed from the stored 0.05 ns), fraction of the curve's events",
          median_ns="median TOT of every event of the curve (from the table, unbinned)",
          control="per run: the chip's pooled TOT (tot_pooled_hist.csv) less this curve is >= 0 "
                  "in every 0.05 ns bin and sums to the events of the pixels not counted in the "
                  "partner run; the curve sums to its n_events",
          positive_control="the board median of tot_pooled_summary.csv for the run of the pairs "
                           "that is a preferred display run",
          times="run start minus RAD_STOP_UTC of the step, hours"),
     inputs=INPUTS_RFSEL)

# %% [markdown]
# ## Figure fe16: the TOT of the RFSel chip over time at fixed settings
#
# The second RFSel pair of fe15, run 52 at RFSel 2 against run 86 at RFSel 3 (threshold offset
# 10), was taken days after the end of the 3.5e15 step, while the chips were cooling down, and
# at fixed settings the TOT can move over days. The time control (`TOT_TIME_CONTROL` in the
# campaign module) adds run 32, at RFSel 2 and offset 10 like run 52 but days earlier: run 32 to
# run 52 is the change over time at fixed settings, run 52 to run 86 the RFSel step.
#
# The chip of fe15, the pixels counted in all three runs, every event of each pixel's triple
# (`rfsel_pair_hist.csv`, the rows of group `32+52+86`). The control is fe15's: each curve must
# be its run's pooled TOT of the chip less the pixels not counted in the other two runs, bin by
# bin. The figure states the bias of the chip and of any other board of its triples whose bias
# differs between the runs (`tables/runs_summary.csv`), and how far the medians of runs 52 and
# 86 here are from fe15's, over the pixels of that pair alone. 0.1 ns bins, each curve as the
# fraction of its events; a short tick on the x axis marks each curve's median; run 32 dashed.

# %%
INPUTS_TIME = [campaign.RFSEL_PAIR_HIST_CSV, campaign.TOT_POOLED_HIST_CSV,
               campaign.TOT_POOLED_SUMMARY_CSV, campaign.TOT_PIXEL_MEDIANS_CSV,
               campaign.RUNS_SUMMARY_CSV, campaign.RUN_TIMES_CSV]
if (list(campaign.TOT_TIME_CONTROL) != [(RF_CAMP, RF_TEL)]
        or len(campaign.TOT_TIME_CONTROL[(RF_CAMP, RF_TEL)]) != 1):
    raise ValueError("fe16 draws one time-control group, on the telescope of the RFSel pairs; "
                     "TOT_TIME_CONTROL is %s" % campaign.TOT_TIME_CONTROL)
(TC_RUNS,) = campaign.TOT_TIME_CONTROL[(RF_CAMP, RF_TEL)]
TC_RUNS = [int(r) for r in TC_RUNS]
TC_NAME = "+".join(str(r) for r in TC_RUNS)
TC = tot_shapes.group_hists(rfsel_hist, RF_CAMP, RF_TEL, TC_NAME, TOT_W)
R_EARLY, R_SAME, R_NEXT = TC_RUNS
if {step_fluence(RF_CAMP, RF_TEL, r) for r in TC_RUNS} != {RF_FLUENCE}:
    raise ValueError("time control %s: not all at the step of the RFSel pairs (%g)"
                     % (TC_NAME, RF_FLUENCE))
if not (TC[R_EARLY]["rfsel"] == TC[R_SAME]["rfsel"] == TC[R_NEXT]["rfsel"] - 1
        and len({TC[r]["offset"] for r in TC_RUNS}) == 1):
    raise ValueError("time control %s: not two runs at one RFSel and a third at the next, at one "
                     "offset: %s" % (TC_NAME, {r: (TC[r]["rfsel"], TC[r]["offset"])
                                               for r in TC_RUNS}))
TC_OFFSET = TC[R_SAME]["offset"]
tc_rows = rfsel_hist[(rfsel_hist["campaign"] == RF_CAMP) & (rfsel_hist["tel"] == RF_TEL)
                     & (rfsel_hist["group"].astype(str) == TC_NAME)]
TC_COMBO = tot_shapes.ANOINTED[RF_BOARD]          # the boards of the chip's triples
counted16 = {r: counted(RF_CAMP, RF_TEL, r, RF_BOARD) for r in TC_RUNS}
common16 = set.intersection(*(set(v) for v in counted16.values()))
hv16 = {b: {r: board_settings(RF_CAMP, RF_TEL, r, b)["hv"] for r in TC_RUNS} for b in TC_COMBO}


def runs_words(runs):
    return ("run %d" % runs[0] if len(runs) == 1
            else "runs %s and %d" % (", ".join(str(r) for r in runs[:-1]), runs[-1]))


def hv_words(b):
    """'480 V in all three runs' or '480 V in run 32 and 450 V in runs 52 and 86'."""
    by = {}
    for r in TC_RUNS:
        by.setdefault(blnw.bias_text([hv16[b][r]]), []).append(r)
    if len(by) == 1:
        return "%s in all three runs" % list(by)[0]
    return " and ".join("%s in %s" % (v, runs_words(rs)) for v, rs in by.items())


curves16, rec16, hs16, control16, times16 = [], [], [], [], {}
for run in TC_RUNS:
    cv = TC[run]
    s = board_settings(RF_CAMP, RF_TEL, run, RF_BOARD)
    if (s["rfsel"], s["offset"]) != (cv["rfsel"], cv["offset"]):
        raise ValueError("run %d: RFSel / offset %s in the run summary, %s in the table" % (
            run, (s["rfsel"], s["offset"]), (cv["rfsel"], cv["offset"])))
    # positive control, as fe15: the chip's pooled TOT in this run less the pixels not counted
    # in the other two runs is this curve, bin by bin
    _, c = tot_shapes.hist_of(tc_rows[tc_rows["run"] == run], "bin_lo_ns",
                              tot_shapes.TOT_BIN_NS, 0.0, tot_shapes.TOT_HI_NS)
    mine = counted16[run]
    left = sorted(set(mine) - common16)
    summ = SUMMARY.loc[(RF_CAMP, RF_TEL, run, RF_BOARD)]
    _, c_all = tot_shapes.hist_of(TOT_POOLED[(RF_CAMP, RF_TEL, run, RF_BOARD)], "bin_lo_ns",
                                  tot_shapes.TOT_BIN_NS, 0.0, tot_shapes.TOT_HI_NS)
    ctl = dict(run=run, board_pixels=int(summ["n_pixels"]), board_events=int(summ["n_events"]),
               pixels_left_out=len(left), events_left_out=sum(mine[p] for p in left),
               common_pixels=len(common16), min_bin_difference=int((c_all - c).min()),
               difference_events=int((c_all - c).sum()), curve_events=int(c.sum()))
    if (ctl["min_bin_difference"] < 0 or ctl["common_pixels"] != cv["n_pixels"]
            or ctl["difference_events"] != ctl["events_left_out"]
            or ctl["curve_events"] != cv["n_events"]
            or ctl["board_pixels"] - ctl["pixels_left_out"] != cv["n_pixels"]
            or ctl["board_events"] - ctl["events_left_out"] != cv["n_events"]):
        raise ValueError("run %d: the time-control curve is not the pooled TOT less the pixels "
                         "left out: %s, table %d pixels %d events" % (run, ctl, cv["n_pixels"],
                                                                     cv["n_events"]))
    control16.append(ctl)
    text, hours = blnw.time_after_stop(RUN_TIMES, RF_CAMP, RF_TEL, [run],
                                       campaign.RAD_STOP_UTC[RF_FLUENCE])
    times16[run] = text
    hs16.append((cv["edges"], cv["y"]))
    curves16.append(dict(color=tot_shapes.RFSEL_COLOR[cv["rfsel"]], edges=cv["edges"], y=cv["y"],
                         median=cv["median_ns"], linestyle="--" if run == R_EARLY else "-",
                         label=u"%s, run %d, %s\nmedian %.2f ns, N = %d"
                               % (style.rfsel_text(cv["rfsel"]), run, text, cv["median_ns"],
                                  cv["n_events"])))
    rec16.append(dict(run=run, chip=RF_CHIP, board_idx=RF_BOARD, rfsel=cv["rfsel"],
                      rf_kohm=style.RFSEL_RF_KOHM[cv["rfsel"]], offset=cv["offset"],
                      hv={campaign.TELESCOPE_CHIPS[RF_TEL][b]: hv16[b][run] for b in TC_COMBO},
                      hours=round(hours[run], 2), n_pixels=cv["n_pixels"],
                      n_events=cv["n_events"], median_ns=round(cv["median_ns"], 4)))
SHIFTS16 = dict(time_ns=round(TC[R_SAME]["median_ns"] - TC[R_EARLY]["median_ns"], 4),
                rfsel_ns=round(TC[R_NEXT]["median_ns"] - TC[R_SAME]["median_ns"], 4))
# the runs of the group that fe15 draws as an RFSel pair: their medians over the pair's pixels
VS15 = [dict(run=x["run"], partner_run=x["partner_run"], pair_pixels=x["n_pixels"],
             pair_median_ns=x["median_ns"], group_pixels=TC[x["run"]]["n_pixels"],
             group_median_ns=round(TC[x["run"]]["median_ns"], 4),
             # both medians at fe15's 4 decimals, so equal medians give 0, never -0
             difference_ns=round(round(TC[x["run"]]["median_ns"], 4) - x["median_ns"], 4) + 0.0)
        for x in sorted(rec15, key=lambda x: x["run"])
        if x["run"] in TC_RUNS and x["partner_run"] in TC_RUNS]
BIAS16 = "; ".join(["%s at %s" % (RF_CHIP, hv_words(RF_BOARD))]
                   + ["%s, board %d of the triple, %s" % (campaign.TELESCOPE_CHIPS[RF_TEL][b], b,
                                                          hv_words(b))
                      for b in TC_COMBO if b != RF_BOARD and len(set(hv16[b].values())) > 1])
for x in rec16:
    print("time control %s: run %d %s RFSel %d offset %d: %d pixels, %d events, median %.4f ns"
          % (TC_NAME, x["run"], x["chip"], x["rfsel"], x["offset"], x["n_pixels"],
             x["n_events"], x["median_ns"]))
print("bias:", BIAS16)
print("control (pooled TOT less the pixels left out):", control16)
print("median shifts:", SHIFTS16)
print("medians over the RFSel pair's pixels against these:", VS15)
foot16 = [TOT_RULE,
          u"The %d %s pixels counted in all three runs, every event of each; %s is the chip of "
          u"the RFSel pairs, the one whose smallest total of counted events over their four runs "
          u"is largest. Bias: %s." % (len(common16), RF_CHIP, RF_CHIP, BIAS16),
          u"0.1 ns bins, each curve as the fraction of its events; tick on the axis: the median; "
          u"run %d dashed. Legend: RFSel, run, time from the end of the step's irradiation to "
          u"the run start, median TOT and events." % R_EARLY,
          u"Control: in every bin each curve is at most its run's pooled TOT of %s, and the "
          u"difference holds exactly the events of the pixels not counted in the other two runs."
          % RF_CHIP,
          u"Fixed settings (RFSel %d, threshold offset %d): the median moves by %+.2f ns from run "
          u"%d (%s) to run %d (%s); the RFSel %d -> %d step, run %d to run %d (%s), moves it by "
          u"%+.2f ns." % (TC[R_SAME]["rfsel"], TC_OFFSET, SHIFTS16["time_ns"], R_EARLY,
                          times16[R_EARLY], R_SAME, times16[R_SAME], TC[R_SAME]["rfsel"],
                          TC[R_NEXT]["rfsel"], R_SAME, R_NEXT, times16[R_NEXT],
                          SHIFTS16["rfsel_ns"])]
if VS15:
    if len({v["pair_pixels"] for v in VS15}) != 1:
        raise ValueError("RFSel pair of the time control: pixels %s" % VS15)
    foot16.append(u"Over the %d pixels counted in both %s, run %d not required (their RFSel "
                  u"pair), the medians are %s ns; the medians here differ from those by %s ns."
                  % (VS15[0]["pair_pixels"], runs_words([v["run"] for v in VS15]),
                     [r for r in TC_RUNS if r not in {v["run"] for v in VS15}][0],
                     " and ".join("%.2f" % v["pair_median_ns"] for v in VS15),
                     " and ".join("%+.2f" % v["difference_ns"] for v in VS15)))
FOOT16_W = 135      # characters per footer line under one 8.2 in panel
fig = tot_shapes.rfsel_pairs(
    [dict(subject=style.settings_text(rfsel=sorted({TC[r]["rfsel"] for r in TC_RUNS}),
                                      offset=TC_OFFSET),
          note="%s, the %d pixels counted in all three runs" % (RF_CHIP, len(common16)),
          curves=curves16)],
    xlabel="TOT [ns]", ylabel="Fraction of events / 0.1 ns",
    xrange=tot_shapes.common_range(hs16, TOT_W, floor=0.0), tag=None,
    data="%s, %s, %s, %s" % (campaign.TELESCOPE_TITLE[RF_TEL], RF_CHIP,
                             style.fluence_label(RF_FLUENCE), RF_CAMP.capitalize()),
    footer="\n".join(textwrap.fill(p, FOOT16_W) for p in foot16))
save(fig, "fe16_rfsel_tot_time_control",
     dict(curves=rec16, chip=RF_CHIP, board_idx=RF_BOARD, telescope=RF_TEL, campaign=RF_CAMP,
          group=TC_NAME, control=control16, shifts=SHIFTS16, pair_medians=VS15,
          pixels=dict(all_three_runs=len(common16),
                      counted={str(r): len(v) for r, v in counted16.items()})),
     dict(fluence=campaign.FLUENCE_CONVENTION,
          group="TOT_TIME_CONTROL of the campaign module: two runs at one RFSel and threshold "
                "offset, then one at the next RFSel; the rows of this group in "
                "rfsel_pair_hist.csv",
          settings="RFSel, threshold offset and bias per run and board from runs_summary.csv",
          pixels="the chip's pixels counted (triple >= %d events) in all three runs; every event "
                 "of each pixel's triple" % FLOOR,
          bins="0.1 ns (summed from the stored 0.05 ns), fraction of the curve's events",
          median_ns="median TOT of every event of the curve (from the table, unbinned)",
          control="per run: the chip's pooled TOT (tot_pooled_hist.csv) less this curve is >= 0 "
                  "in every 0.05 ns bin and sums to the events of the pixels not counted in the "
                  "other two runs; the curve sums to its n_events",
          shifts="time_ns: the second run's median minus the first's (same settings); rfsel_ns: "
                 "the third minus the second (the RFSel step)",
          pair_medians="the runs of the group that fe15 draws as an RFSel pair: fe15's median "
                       "over the pair's pixels, this figure's over the pixels of all three runs, "
                       "and the difference",
          times="run start minus RAD_STOP_UTC of the step, hours"),
     inputs=INPUTS_TIME)
