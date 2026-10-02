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
# * Figures fe13-fe14: the TOT and TOA difference of one pixel in the three July runs, H1 and F1.
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
#    `tables/runs_summary.csv`, `tables/run_times.csv` and `july/display_runs_jul.csv`.
#    `../campaigns/irrad_2026_inputs.md` lists every table with where it came from;
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
# The drawing lives in `../frontend/blnw.py` (figures fe01-fe02), `../frontend/tdc_codes.py`
# (figures fe03-fe06) and `../frontend/tot_shapes.py` (figures fe07-fe14). This notebook holds
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
# March runs record no RFSel; they ran at RFSel 2 (the user's ruling), so every figure here
# states RFSel 2.
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
#   preferred or not, side by side.
# * The pair at 1.5e15 (fe02-fe06), `REST_PAIRS` in the campaign module: before cooling down, the
#   March run of the 1.5e15 step whose leakage current matches the July run's (its lowest bias,
#   run 17 on both telescopes); after cooling down, the July preferred display run at 1.5e15.
#
# The tick under each irradiated step gives the time from the end of that step's irradiation
# (`RAD_STOP_UTC` in the campaign module, the earliest stop the records allow) to the start of
# the run, or the range over the runs drawn there; the July 1.5e15 runs count from the end of
# the March 1.5e15 step. Run starts come from `tables/run_times.csv`.

# %%
july_display = tables.display_runs(campaign.DISPLAY_RUNS_JUL_CSV, **SETTINGS)
march_steps = {t: {step_fluence("march", t, r): r for r in campaign.MARCH_DISPLAY_RUNS[t]}
               for t in TELS}
LADDER = ([dict(campaign="march", fluence=f, runs={t: [march_steps[t][f]] for t in TELS})
           for f in sorted(march_steps[TELS[0]])]
          + [dict(campaign="july", fluence=f, runs={t: july_display[t][f] for t in TELS})
             for f in sorted(july_display[TELS[0]])])
PAIR = [dict(campaign="march", fluence=1.5e15, runs={t: [campaign.REST_PAIRS[t][0]] for t in TELS}),
        dict(campaign="july", fluence=1.5e15, runs={t: [campaign.REST_PAIRS[t][1]] for t in TELS})]

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
                            u"20; July: every display run at the step, side by side.\nDotted "
                            u"line: March | July. Small text: sensor bias. Tick: time from the "
                            u"end of the step's irradiation to the run start."))
save(fig, "fe01_blnw_vs_fluence",
     dict(points=pts, steps=LADDER, tick_labels=ticks,
          hours_after_stop={"%s %s run %d" % k: v for k, v in sorted(HOURS.items())},
          excluded_pixels={q: blnw.excluded_range(pts, q) for q in ("bl", "nw")},
          not_exact=off),
     dict(fluence=campaign.FLUENCE_CONVENTION,
          statistics="allpix: mean and standard deviation over all 256 pixels; live: over the "
                     "pixels with a value (BL 50 < baseline < 1020 DAC, not railed; NW > 0)",
          error_bar="pixel-to-pixel standard deviation, not an uncertainty",
          times="run start (tables/run_times.csv) minus RAD_STOP_UTC of the step, hours",
          rfsel="July runs at RFSel 2 by the display-run list; March runs record no RFSel and ran "
                "at RFSel 2"),
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
     dict(points=pair_pts, pairs=campaign.REST_PAIRS, tick_labels=pair_ticks,
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
# the 0.1 ns bin holding the 50 % point).

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
              & (hists["run"] == campaign.REST_PAIRS[tel][1])]
    return {int(b): (int(g["row"].iloc[0]), int(g["col"].iloc[0]))
            for b, g in s.groupby("board_idx")}


TDC_FOOTER = (u"One pixel per chip, the most populated in the July run; every loop file; ea = 0 "
              u"and |CAL - CAL mode of the pixel in that file| <= 3. TOA = 12.5 ns - TOA code x "
              u"3.125 ns / CAL mode, TOT = (2 TOT code - floor(TOT code / 32)) x 3.125 ns / CAL "
              u"mode.\nBins: CAL native codes, TOA 0.3 ns, TOT 0.2 ns; each histogram the fraction of "
              u"its hits in the range drawn, per bin. Box: CAL mode, TOA and TOT median.")
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
         dict(selection="one pixel per chip (most populated in the July run of REST_PAIRS); all "
                        "loop files; ea == 0; |cal - per-file CAL mode| <= 3",
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
# figures fe03-fe04 is ringed. One CAL colour scale for every CAL map of this notebook and one
# symmetric scale for every difference, both set below from all the maps drawn (0.5th to 99.5th
# percentile of the modes, 99th percentile of the absolute differences); a pixel outside a scale
# takes the end colour and is counted ("sat N" in its title). Grey: no CAL mode. Maps in the chip
# frame: pixel (0, 0) at the bottom right.

# %%
grids = {}
for tel in TELS:
    for st in PAIR:
        for b in range(len(campaign.TELESCOPE_CHIPS[tel])):
            grids[(tel, st["campaign"], b)] = tdc_codes.mode_grid(
                cal_modes, st["campaign"], tel, st["runs"][tel][0], b)
modes = np.concatenate([g[np.isfinite(g)] for g in grids.values()])
diffs = np.concatenate([(grids[(t, "july", b)] - grids[(t, "march", b)]).ravel()
                        for t in TELS for b in range(len(campaign.TELESCOPE_CHIPS[t]))])
diffs = diffs[np.isfinite(diffs)]
CAL_NORM = Normalize(vmin=float(np.floor(np.percentile(modes, 0.5))),
                     vmax=float(np.ceil(np.percentile(modes, 99.5))))
DIFF_LIM = max(1.0, float(np.ceil(np.percentile(np.abs(diffs), 99))))
DIFF_NORM = tdc_codes.symmetric_norm(DIFF_LIM)
print("CAL scale %g-%g, difference scale +-%g" % (CAL_NORM.vmin, CAL_NORM.vmax, DIFF_LIM))

MAP_FOOTER = (u"CAL mode: median over the loop files of the pixel's most frequent CAL code in "
              u"each file (ea = 0). Difference: July minus March. One CAL scale for every CAL map "
              u"of these figures, one for every difference; sat N: pixels outside it.\nGrey: no "
              u"CAL mode. Ringed: the pixel of the CAL, TOA and TOT histograms. Maps in the %s."
              % pixel_maps.FRAME_TEXT)
for k, tel in enumerate(TELS):
    lk = looks(tel)
    words = ["%s, run %d" % (x["label"], x["run"]) for x in lk] + ["July minus March"]
    fig, payload = tdc_codes.cal_maps(
        {(c, b): g for (t, c, b), g in grids.items() if t == tel}, tel, lk,
        pixels=pixels_of(tel), norm=CAL_NORM, diff_norm=DIFF_NORM,
        titles=lambda chip, j: "%s, %s" % (chip, words[j]),
        cbar_labels=("CAL mode [code]", "CAL mode, July minus March [code]"),
        tag=SETTINGS_TEXT,
        data="%s %s" % (style.fluence_label(1.5e15, unit=False), campaign.FLUENCE_UNIT),
        footer=MAP_FOOTER)
    for rec in payload:
        print("%s %s: saturated before %d, after %d, difference %d" % (
            tel, rec["chip"], rec["before"]["saturated"], rec["after"]["saturated"],
            rec["difference"]["saturated"]))
    save(fig, "fe%02d_cal_maps_%s" % (5 + k, tel),
         dict(maps=payload, cal_scale=[CAL_NORM.vmin, CAL_NORM.vmax],
              difference_scale=[-DIFF_LIM, DIFF_LIM],
              runs=[dict(campaign=x["campaign"], run=x["run"]) for x in lk]),
         dict(cal_mode="median over the loop files of the per-file mode of the CAL code (ea == 0)",
              scales="CAL: 0.5th-99.5th percentile of every CAL mode of fe05-fe06; difference: "
                     "99th percentile of every |July - March|, symmetric",
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
# every display run at RFSel 2 and offset 20, the preferred one large and joined, the others
# small beside it.

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
     "other": dict(label="other July display runs", size=0.62)},
    tick_labels=ticks, xlabel=XLABEL_LADDER, ylabel="Median of the pixel TOT medians [ns]",
    tel_titles=TEL_TITLES, tag=SETTINGS_TEXT, data=None, board_dx=0.06,
    boundary_after=sum(st["campaign"] == "march" for st in LADDER) - 1,
    footer=(u"Point: median over the board's pixels of each pixel's median TOT. Bar: 16-84 %% "
            u"of those pixel medians (the pixel-to-pixel spread, not an uncertainty); %d-%d "
            u"pixels per board, %d-%d in the other display runs.\n%s\nMarch: one run per step. "
            u"July: every display run at the step (RFSel 2, offset 20) side by side, the "
            u"preferred one large and joined. Dotted line: March | July. Tick: time from the end "
            u"of the step's irradiation to the run start."
            % (npix07["main"] + npix07["other"] + (TOT_RULE,))))
save(fig, "fe07_tot_median_vs_fluence",
     dict(points=pts07, steps=LADDER, tick_labels=ticks,
          pixels_per_board={k: list(v) for k, v in npix07.items()},
          preferred_july={t: july_pref[t] for t in TELS},
          hours_after_stop={"%s %s run %d" % k: v for k, v in sorted(HOURS.items())}),
     dict(fluence=campaign.FLUENCE_CONVENTION,
          statistic="median over the counted pixels of each pixel's median TOT (ns)",
          error_bar="16th-84th percentile of the pixel medians, not an uncertainty",
          pixels="a pixel counts when its triple (the one with the most events in boards 0-1-2, "
                 "board 3: 1-2-3) holds >= %d events" % FLOOR,
          times="run start (tables/run_times.csv) minus RAD_STOP_UTC of the step, hours",
          rfsel="July runs at RFSel 2 by the display-run list; March runs record no RFSel and ran "
                "at RFSel 2"),
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
            u"16-84 %% of those pixel medians (spread, not an uncertainty). Open: the most "
            u"frequent TOT of every event of the board's counted pixels together, centre of the "
            u"fullest 0.1 ns bin.\n%s %d-%d pixels per board.\nOne run per step: March, the run "
            u"of each step; July, the preferred display run (RFSel 2, offset 20). Dotted line: "
            u"March | July. Tick: time from the end of the step's irradiation to the run start."
            % ((TOT_RULE,) + npix08)))
save(fig, "fe08_tot_board_vs_fluence",
     dict(points=pts08, steps=STEPS_08, tick_labels=ticks08, pixels_per_board=list(npix08)),
     dict(fluence=campaign.FLUENCE_CONVENTION,
          pixmed="median over the counted pixels of each pixel's median TOT (ns); bar 16th-84th "
                 "percentile of the pixel medians",
          mode="centre of the fullest 0.1 ns bin of every event of the counted pixels (ns)",
          pixels="as fe07", rfsel="as fe07"),
     inputs=INPUTS_TRACKS)

# %% [markdown]
# ## Figures fe09-fe10: TOT distributions at every step
#
# fe09: every event of every counted pixel of a board together; fe10: one pixel per board, the
# same pixel in every run (among the pixels counted in all seven runs, the one whose smallest
# event count over the runs is largest). One curve per run, 0.1 ns bins, each as the fraction of
# its events; one x range per telescope (0.5 to 99.5 % of its curves). The legend gives the step,
# the time from the end of its irradiation to the run start (the July 1.5e15 run after cooling
# down, timed from the end of the March 1.5e15 step) and the board's bias.

# %%
TOT_W, DT_W = 0.1, tot_shapes.DT_BIN_PS
SHAPE_RUNS_TEXT = (u"One run per step: March, the run of each step; July, the preferred display "
                   u"run (RFSel 2, offset 20). Legend: step, time from the end of its irradiation "
                   u"to the run start (July 1.5e15: from the end of the March step, after cooling "
                   u"down), bias.")


def pooled_tot(camp, tel, run, b):
    e, c = tot_shapes.hist_of(TOT_POOLED[(camp, tel, run, b)], "bin_lo_ns",
                              tot_shapes.TOT_BIN_NS, 0.0, tot_shapes.TOT_HI_NS)
    return tot_shapes.rebin(e, c, TOT_W)


def pixel_tot(pick, camp, tel, run, b):
    d = PIXEL_HISTS[(pick, "tot_ns", camp, tel, run, b, -1)]
    e, c = tot_shapes.hist_of(d, "bin_lo", tot_shapes.TOT_BIN_NS, 0.0, tot_shapes.TOT_HI_NS)
    return tot_shapes.rebin(e, c, TOT_W) + (d.iloc[0],)


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
            e, c, row = pixel_tot("pixel", st["campaign"], t, st["run"], b)
            curves.append(dict(label=step_label(st, t, [b]), color=st["color"], edges=e,
                               y=c / float(row["n_events"])))
            hs.append((e, c))
            ns.append(int(row["n_events"]))
            rec10.append(dict(tel=t, board_idx=b, chip=chip, row=px[0], col=px[1],
                              campaign=st["campaign"], run=st["run"],
                              n_events=int(row["n_events"]), n_outside=int(row["n_outside"]),
                              median_ns=round(float(row["median"]), 4), file=row["file"]))
        panels[t].append(dict(subject="%s, %s" % (t.upper(), chip), curves=curves,
                              note="pixel (%d, %d), N %d-%d" % (px + (min(ns), max(ns)))))
    ranges[t] = tot_shapes.common_range(hs, TOT_W, floor=0.0)
print("candidates per board:", cand10)
fig = tot_shapes.shape_grid(
    panels, xlabel="TOT [ns]", ylabel="Fraction of events / 0.1 ns", ranges=ranges,
    tag=SETTINGS_TEXT, data=None,
    footer=(u"One pixel per board, the same in every run: among the pixels whose triple holds "
            u">= %d events in all seven runs (%d-%d candidates per board), the one whose "
            u"smallest event count over the runs is largest.\nPer run its TOT from its triple, "
            u"the one with the most events in its board's combination (%s); 0.1 ns bins, as a "
            u"fraction of its events.\n%s"
            % (FLOOR, min(cand10.values()), max(cand10.values()), COMBO_WORDS,
               SHAPE_RUNS_TEXT)))
save(fig, "fe10_tot_shape_pixel_vs_fluence",
     dict(curves=rec10, candidates=cand10, ranges=ranges, runs={t: SHAPE[t] for t in TELS}),
     dict(fluence=campaign.FLUENCE_CONVENTION, bins="0.1 ns, fraction of the pixel's events",
          pixel="per board, among the pixels >= %d events in every run drawn, the largest "
                "smallest count; the same pixel in every run" % FLOOR,
          median_ns="median TOT of the pixel's events in its triple"),
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
# board 2 from its triple with the most events in each run, in events per bin (0.1 ns and
# 100 ps). The board-2 pixel is the one in that triple.

# %%
for k, t in enumerate(TELS):
    chips = campaign.TELESCOPE_CHIPS[t]
    july = [st for st in SHAPE[t] if st["campaign"] == "july"]
    runs = [(st["campaign"], st["run"]) for st in july]
    px = stored_pixel("july_pixel", "tot_ns", t, runs, 1)
    mine, counts, ncand = fixed_pixel(t, 1, runs)
    if mine != px:
        raise ValueError("%s: July pixel %s in the table, %s by the rule" % (t, px, mine))
    columns, rec, hs = [], [], {"tot": [], "dtoa": []}
    for st in july:
        e, c, row = pixel_tot("july_pixel", "july", t, st["run"], 1)
        d = PIXEL_HISTS[("july_pixel", "dtoa_ps", "july", t, st["run"], 1, 2)]
        drow = d.iloc[0]
        ed, cd = tot_shapes.rebin(*tot_shapes.hist_of(d, "bin_lo", DT_W, -tot_shapes.DT_HALF_PS,
                                                      tot_shapes.DT_HALF_PS), 100.0)
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
                        dt_outside=int(drow["n_outside"]), file=row["file"]))
    rows = [dict(key="tot", xlabel="TOT, %s pixel (%d, %d) [ns]" % ((chips[1],) + px),
                 ylabel="Events / 0.1 ns", range=tot_shapes.common_range(hs["tot"], TOT_W,
                                                                          floor=0.0)),
            dict(key="dtoa", xlabel="dTOA, %s - %s [ps]" % (chips[1], chips[2]),
                 ylabel="Events / 100 ps", zero_line=True,
                 range=tot_shapes.common_range(hs["dtoa"], 100.0, symmetric=True))]
    fig = tot_shapes.run_columns(
        columns, rows, tag=SETTINGS_TEXT, data=None,
        footer=(u"One %s pixel, the same in the three July runs: among the %d pixels whose "
                u"triple holds >= %d events in all three, the one whose smallest event count is "
                u"largest. Per run its triple with the most events (boards 0-1-2); its TOT, and "
                u"dTOA = TOA(%s) - TOA(%s)\nafter a time-walk correction solved jointly over the "
                u"triple's three boards, centred on its median. Bins 0.1 ns and 100 ps, events "
                u"per bin. July preferred display runs (RFSel 2, offset 20); time from the end of "
                u"the step's irradiation to the run start (1.5e15: from the end of the March "
                u"step, after cooling down)."
                % (chips[1], ncand, FLOOR, chips[1], chips[2])))
    save(fig, "fe%02d_tot_dtoa_shape_vs_fluence_jul_%s" % (13 + k, t),
         dict(pixel=list(px), board=1, chip=chips[1], candidates=ncand,
              counts_at_pick=counts, runs=rec),
         dict(fluence=campaign.FLUENCE_CONVENTION,
              pixel="board 1, among the pixels >= %d events in all three July runs, the largest "
                    "smallest count" % FLOOR,
              tot="0.1 ns bins, events", dtoa="as fe11, 100 ps bins, events",
              times="run start minus RAD_STOP_UTC of the step, hours"),
         inputs=INPUTS_TRACKS)
