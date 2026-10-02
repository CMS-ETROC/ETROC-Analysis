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
# # Front end: baseline, noise width and raw TDC codes, CERN IRRAD 2026
#
# What the irradiation did to the ETROC2 front end before any timing is computed:
#
# * Figure fe01: the autocalibration baseline (BL) and noise width (NW) of every board at every
#   fluence step, March and July.
# * Figure fe02: the same at 1.5e15 for one run pair per telescope, before and after the four
#   months of cooling down between the campaigns.
# * Figures fe03-fe04: the CAL, TOA and TOT of one pixel per chip in that run pair, H1 and F1.
# * Figures fe05-fe06: the CAL code of every pixel in that run pair, and its change, H1 and F1.
#
# Every figure is saved as PNG (200 dpi) and PDF, with `<name>_values.json` beside it: every
# number drawn, the input files read and the commit of the code.
#
# To run it:
#
# 1. `source TestBeam/condor_at_lxplus/envs/load_python39.sh` (LCG_104d, the environment this
#    notebook is tested in).
# 2. The figures read small tables that are not in the repository, from
#    `/eos/user/m/musafdar/ETROC_plot_inputs/irrad_2026`: the three tables of `frontend/`,
#    `tables/runs_summary.csv`, `tables/run_times.csv` and `july/display_runs_jul.csv`.
#    `../campaigns/irrad_2026_inputs.md` lists every table with where it came from;
#    `python -m etroc_plots.frontend.rebuild` (run from `TestBeam/`) remakes the `frontend/`
#    tables from the autocalibration scan histories, the run metadata and the merged hits. To
#    read your own copy, copy that whole `irrad_2026` folder and point `ETROC_INPUTS` at the
#    copy. The first code cell checks that every input exists and can be read, and stops with a
#    list of all that cannot; a table whose checksum differs from the published list is named,
#    and the run goes on.
# 3. Open `frontend.ipynb` in Jupyter and run all cells, or run it headless with
#    `jupyter nbconvert --to notebook --execute frontend.ipynb --output-dir <somewhere>`.
#
# Figures go to `$ETROC_FIGURES/frontend/`, by default `figures/frontend/` next to this notebook.
#
# The drawing lives in `../frontend/blnw.py` (figures fe01-fe02) and `../frontend/tdc_codes.py`
# (figures fe03-fe06). This notebook holds the choices made per figure: the runs, the colour
# scales and the words on each figure.
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
from etroc_plots.frontend import blnw, tdc_codes              # noqa: E402
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
# table (finer bins show the comb of the TDC codes). Each histogram is normalised to unit area
# over the range drawn; the hits outside it are counted in the values file. The box gives the CAL
# mode (most frequent code) and the TOA and TOT medians (lower edge of the 0.1 ns bin holding the
# 50 % point).

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
              u"mode.\nBins: CAL native codes, TOA 0.3 ns, TOT 0.2 ns; each histogram normalised "
              u"to unit area over the range drawn. Box: CAL mode, TOA and TOT median.")
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
              bins="CAL native code; TOA 0.3 ns over 0-12.3 ns; TOT 0.2 ns over 0-10 ns; unit "
                   "area over the range drawn",
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
