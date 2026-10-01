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
# # Overview: CERN IRRAD 2026, the setup, the method and the two campaigns
#
# The figures that frame the results:
#
# * Figures 1-3, the setup: the two telescopes in the beam, the beam profile over a chip, and
#   the boards of each telescope.
# * Figures 4-5, the method: how one time resolution per pixel and one number per board are
#   chosen, and what the ETROC2 RFSel setting changes in the front end.
# * Figure 6: every run of the March and July campaigns on a time line, with the fluence steps.
#
# Figures 1-5 are schematic and draw no measured data. The chips, sensors and beam profile they
# show are in the campaign module, `../campaigns/irrad_2026.py`; the other setup facts on them
# (beam, cold box, sensor and readout notes) are written in this notebook's cells. Every figure
# is saved as PNG (200 dpi) and PDF, with `<name>_values.json` beside it: every number drawn,
# the input files read and the commit of the code.
#
# To run it:
#
# 1. `source TestBeam/condor_at_lxplus/envs/load_python39.sh` (LCG_104d, the environment this
#    notebook is tested in).
# 2. Figure 6 reads four small tables that are not in the repository, from
#    `/eos/user/m/musafdar/ETROC_plot_inputs/irrad_2026`: `tables/run_times.csv`,
#    `july/lv_spans_jul.csv`, `july/hv_cycles_jul.csv` and `july/display_runs_jul.csv`, and the
#    two run-config yamls in `TestBeam/board_configs_yaml/`. `../campaigns/irrad_2026_inputs.md`
#    lists every table with where it came from. To read your own copy, copy that whole
#    `irrad_2026` folder and point `ETROC_INPUTS` at the copy. The first code cell checks that
#    every input exists and can be read, and stops with a list of all that cannot; a table whose
#    checksum differs from the published list is named, and the run goes on.
# 3. Open `overview.ipynb` in Jupyter and run all cells, or run it headless with
#    `jupyter nbconvert --to notebook --execute overview.ipynb --output-dir <somewhere>`.
#
# Figures go to `$ETROC_FIGURES/overview/`, by default `figures/overview/` next to this
# notebook. Setting `PANELS = True` in the setup cell below also writes each panel of figures 4
# and 5 as a file of its own next to the figure, named `<stem>_01_anointed_combo.png` and so
# on. A panel gets the text-overlap audit of its figure, printed in the notebook's output; it
# has no values file of its own.
#
# The drawing lives in `../overview/cartoons.py` (figures 1-5) and `../overview/timeline.py`
# (figure 6). This notebook holds the choices made per figure: the runs and the words on each
# figure.
#
# `overview.ipynb` and `overview.py` are the same notebook, kept in step by jupytext. Edit
# either one, then run `python3 -m jupytext --sync overview.ipynb` and
# `python3 -m nbstripout overview.ipynb` in this folder before committing. Neither tool is in
# LCG_104d (the install line is in `../README.md`); running the notebook needs neither.
#
# After a run, `python3 -m etroc_plots.checks <figures>`, run from `TestBeam/` with `<figures>`
# the folder this notebook wrote to (`etroc_plots/notebooks/figures/overview` by default),
# checks every saved figure against the rules in `../CONVENTIONS.md`.

# %%
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")   # draw off-screen; show() displays the PNG exactly as saved
import matplotlib.pyplot as plt
from IPython.display import Image, display

# The folder that holds the package (TestBeam/) is two folders up. Jupyter starts the kernel in
# the notebook's folder; `python overview.py` has __file__.
here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
sys.path.insert(0, os.path.normpath(os.path.join(here, "..", "..")))

CAMPAIGN = "irrad_2026"                       # campaigns/<CAMPAIGN>.py
# An exported ETROC_CAMPAIGN wins (an empty one counts as unset); it is read once, on the first
# etroc_plots import.
if not os.environ.get("ETROC_CAMPAIGN"):
    os.environ["ETROC_CAMPAIGN"] = CAMPAIGN

from etroc_plots import style                                 # noqa: E402
from etroc_plots import campaigns                             # noqa: E402
from etroc_plots import lv_log                                # noqa: E402
from etroc_plots.inputs import check_inputs                   # noqa: E402
from etroc_plots.overview import cartoons, timeline           # noqa: E402

if campaigns.NAME != CAMPAIGN:
    raise RuntimeError("the package is running campaign %r (ETROC_CAMPAIGN) but CAMPAIGN = %r: "
                       "set one to match the other, then restart the kernel (the campaign is "
                       "read once, on the first import)" % (campaigns.NAME, CAMPAIGN))
campaign = campaigns.active

INPUTS_TIMELINE = [campaign.RUN_TIMES_CSV, campaign.LV_SPANS_JUL_CSV,
                   campaign.HV_CYCLES_JUL_CSV, campaign.DISPLAY_RUNS_JUL_CSV]
RUN_YAMLS = sorted(set(campaign.RUN_LIST_YAML.values()))
# stops, listing every missing or unreadable input; reports any table that differs from the
# published copy. A table moved out of the inputs folder by its own variable is checked as a
# file read in place.
_rel = [os.path.relpath(p, campaign.INPUTS) for p in INPUTS_TIMELINE]
check_inputs(only=tuple(r for r in _rel if not r.startswith("..")),
             raw_inputs=RUN_YAMLS + [p for p, r in zip(INPUTS_TIMELINE, _rel) if r.startswith("..")])

NOTEBOOK = "TestBeam/etroc_plots/notebooks/overview.ipynb"   # recorded as "script" in every values file
OUT = os.path.join(os.path.abspath(os.environ.get("ETROC_FIGURES") or "figures"), "overview")
os.makedirs(OUT, exist_ok=True)

PANELS = False   # True: also write the single panels of figures 4 and 5 (see above)
CARTOON = {"data": "none: a schematic figure; every number on it is listed in these values"}


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


def save_panels(stem, figs):
    """figs: [(name, figure), ...] -> <stem>_01_<name>.png and so on."""
    paths = []
    for index, (name, fig) in enumerate(figs, start=1):
        print(style.check_no_clipping(fig, "%s panel %s" % (stem, name)))
        paths += style.save_panel(fig, OUT, stem, index, name)
        plt.close(fig)
    style.flatten_panels(OUT, stem, paths)


# %% [markdown]
# ## Figure 1: the telescopes in the beam
#
# The H1 (HPK) and F1 (FBK) telescopes one after the other along the beam, four chips each, in
# one cold box. Each chip is drawn as one plane (sensor and ETROC together) at the beam's 60
# degree incidence, coloured by its board index. At the entrance, the beam profile of figure 2
# stood on its side, with the chip window. Not to scale.

# %%
BEAM_GEV, INCIDENCE_DEG, COLD_BOX_C = 24, 60, -25   # proton energy, incidence, cold-box set point
fig = cartoons.telescope_layout(
    name="Telescope layout", tag="H1 then F1 along the beam, one cold box, not to scale",
    telescope_labels={"h1": "H1 (HPK)", "f1": "F1 (FBK)"},
    beam_text=u"%d GeV protons\n%d° beam incidence" % (BEAM_GEV, INCIDENCE_DEG),
    box_text=u"cold box, −%d °C" % -COLD_BOX_C,
    profile_texts=("beam\nprofile", u"±%g mm" % (10 * campaign.BEAM_WINDOW_CM)),
    incidence_deg=INCIDENCE_DEG)
save(fig, "overview01_telescope_layout",
     dict(chips=campaign.TELESCOPE_CHIPS, beam_gev=BEAM_GEV, incidence_deg=INCIDENCE_DEG,
          cold_box_c=COLD_BOX_C, beam_exposures_cm=campaign.BEAM_EXPOSURES,
          beam_fwhm_cm=campaign.BEAM_FWHM_CM, chip_window_cm=campaign.BEAM_WINDOW_CM), CARTOON)

# %% [markdown]
# ## Figure 2: the beam profile
#
# The beam is a Gaussian of 1.1 cm FWHM (sigma 0.47 cm). To spread the fluence evenly over a
# chip, it was delivered as three exposures, at x = +0.6, 0 and -0.6 cm with 40 %, 20 % and 40 %
# of the protons. Each curve is normalised so its protons sum to one; the thick curve is their
# sum, the grey one a single centred exposure of the same protons. The box gives the sum's mean
# (Avg), its RMS deviation from that mean (RMSE) and max / min over the chip window
# (-0.5 to 0.5 cm); the facility's table of beam options quotes Avg and RMSE. The cell also
# recomputes every row of that table (`BEAM_TABLE`): with sigma = FWHM / 2.355 all six agree
# with it to within 0.002 in Avg and 0.0003 in RMSE. The figure shows this model's numbers; the
# table's own sampling of the window is not documented.

# %%
fig, metrics = cartoons.beam_profile(
    name="Beam profile model",
    tag=u"three exposures, beam FWHM %g cm (σ %.2f cm)" % (campaign.BEAM_FWHM_CM,
                                                           cartoons.beam_sigma_cm()),
    legend_texts=dict(window=u"chip window  [−%g, %g] cm" % (campaign.BEAM_WINDOW_CM,
                                                             campaign.BEAM_WINDOW_CM),
                      single="one centred exposure of the same protons",
                      component="x = {x} cm, {share} % of the protons",
                      total="sum of the three exposures"))
table_check = cartoons.beam_table_check()
for row in table_check:
    print("  %-18s Avg %.4f (table %.4f)   RMSE %.5f (table %.5f)"
          % (row["name"], row["avg"], row["avg_table"], row["rmse"], row["rmse_table"]))
save(fig, "overview02_beam_profile",
     dict(beam_fwhm_cm=campaign.BEAM_FWHM_CM, beam_sigma_cm=cartoons.beam_sigma_cm(),
          exposures_cm_share=campaign.BEAM_EXPOSURES, chip_window_cm=campaign.BEAM_WINDOW_CM,
          window_metrics=metrics, facility_table_check=table_check),
     dict(CARTOON, profile="each exposure a unit-area Gaussian times its share of the protons; "
                           "Avg and RMSE over x in the chip window, sampled at 20001 points"))

# %% [markdown]
# ## Figure 3: the boards
#
# Per telescope, the chip of each board, the sensor under it (wafer and die) and the HV channel
# that biases it. The board index sets the colour used for the board in figures 1 and 4.

# %%
TELESCOPE_NOTES = {"h1": "HPK sensors, thin ET2.02 chips",
                   "f1": "FBK carbon-enriched sensors, thick ET2.01 chips"}
BOARD_NOTES = ["HV channel = iseg channel = column of the slow-control log.",
               "H1 sensors randomly selected; F1 the best available, two with dead pixels.",
               u"Pixel array 16×16 at 1.3 mm pitch; active sensor thickness 50 µm (HPK and FBK).",
               "Readout: lpGBT1 + follower lpGBT2, VTRx+."]
fig = cartoons.board_facts(name="Board summary", telescope_notes=TELESCOPE_NOTES,
                           notes=BOARD_NOTES)
save(fig, "overview03_board_summary",
     dict(chips=campaign.TELESCOPE_CHIPS, sensors=campaign.TELESCOPE_SENSORS,
          hv_channel="the board index", telescope_notes=TELESCOPE_NOTES, notes=BOARD_NOTES),
     CARTOON)

# %% [markdown]
# ## Figure 4: one track per pixel, one number per board
#
# (a) A board's time resolution is solved from three boards together: boards 0, 1 and 2 from
# the combination 0-1-2, board 3 from 1-2-3 (the anointed combinations). (b) Several tracks, one
# pixel on each board of the combination, pass through a pixel; the one with the most events
# gives the pixel its value, if it has at least 300. The event counts drawn are illustrative.
# (c) The board's number is the mean of a robust Gaussian fit (2.5 sigma clip) over its pixel
# values; its error bar is the pixel-to-pixel spread.

# %%
fig, figs = cartoons.track_selection(
    name="Track selection", tag="one track per pixel per board", line1="Analysis method",
    line2="cartoon, no data",
    footer_text=("The selection behind every pixel map and board value; the event counts in (b) "
                 "are illustrative. A board's value comes from its anointed combo: the board "
                 "and two partner boards."),
    panels=PANELS)
save(fig, "overview04_track_selection",
     dict(anointed_combos={"0-1-2": [0, 1, 2], "1-2-3": [3]},
          pixels_per_board=cartoons.TRACK_PIXELS_PER_SIDE ** 2,
          event_floor=cartoons.TRACK_EVENT_FLOOR, robust_fit_clip_sigma=cartoons.TRACK_CLIP_SIGMA,
          illustrative_event_counts=[n for _, n, _ in cartoons.TRACK_EXAMPLE]), CARTOON)
if PANELS:
    save_panels("overview04_track_selection", figs)

# %% [markdown]
# ## Figure 5: what RFSel changes
#
# RFSel selects the feedback resistor R_f of the preamplifier (a transimpedance amplifier):
# RFSel 0/1/2/3 = 20/10/5.7/4.4 kOhm. A larger R_f gives more preamp gain and a slower trailing
# edge, so a longer TOT; the leading edge is set by the preamp bias current. The pulses are
# schematic: their heights and decay times only draw that trend.

# %%
RF_LINES = [
    ("Preamp gain / amplitude", u"higher R$_\\mathrm{f}$ → more preamp gain, compensating "
                                u"LGAD signal loss after irradiation"),
    ("Return to baseline / TOT length", u"higher R$_\\mathrm{f}$ → slower trailing edge "
                                        u"→ longer TOT; lower R$_\\mathrm{f}$ → shorter TOT"),
    ("Front-end noise / jitter", "preamp gain is set to balance timing performance against "
                                 "noise-induced jitter"),
    ("Discriminator threshold", "set at baseline + programmable offset, independent of RFSel"),
]
RF_CLOSING = (u"Standard setting in both campaigns: %s; July also took runs at RFSel 1 and 3"
              % style.settings_text(rfsel=2, offset=None))
fig, figs = cartoons.rfsel_explainer(
    name="What RFSel selects", line1="ETROC2 front end",
    titles=["Where RFSel acts", "Same input charge, four R$_\\mathrm{f}$", "What it changes"],
    lines=RF_LINES, closing=RF_CLOSING,
    footer_text=("Schematic, not measured data: the pulse shapes draw the trend only. R_f per "
                 "RFSel code from the ETROC2 reference manual."),
    standard=2, panels=PANELS, panel_line2="schematic, not measured data")
save(fig, "overview05_rfsel",
     dict(rfsel_rf_kohm=style.RFSEL_RF_KOHM, standard_rfsel=2,
          pulse_amplitude_arb=cartoons.PULSE_AMP, pulse_decay_arb=cartoons.PULSE_TAU,
          threshold_arb=cartoons.PULSE_THRESHOLD, text_panel=[list(x) for x in RF_LINES],
          closing=RF_CLOSING),
     dict(CARTOON, pulses="illustrative amplitudes and decay times, not measured"))
if PANELS:
    save_panels("overview05_rfsel", figs)

# %% [markdown]
# ## Figure 6: the campaigns, run by run
#
# March 2026 (top) and July 2026 (bottom), one row per telescope, one bar per run from its start
# to start + its configured length: the DAQ records no stop time, so a run stopped early is
# still drawn to its configured end. The runs are those of the campaign's run-config yaml, each
# at the bookkeeping fluence the yaml gives it; the start and length come from the DAQ's run
# metadata (`tables/run_times.csv`, made by `timeline.build_run_times`).
#
# * Background: the fluence step. A step ends at the radiation stop of the next one
#   (`RAD_STOP_UTC` in the campaign module), the dotted line with its time: the earliest stop
#   the records allow (shift logbook, operators' messages, LV power log).
# * Filled bars: display runs, the runs the result figures use; "p" marks the preferred run.
#   July: the campaign's display-run list (`july/display_runs_jul.csv`). March: at each step
#   the highest-bias run at threshold offset 20 (`MARCH_DISPLAY_RUNS` in the campaign module,
#   the runs of the maps notebook).
# * Hatched: the LV off for longer than 1 h (`LV_OFF_MIN_H` in the campaign module), a gap in the
#   LV power log, which records only while the LV is on; from `july/lv_spans_jul.csv` (made by
#   `lv_log.build_lv_spans`). Every gap in the log is an LV cycle; the shorter ones, of seconds
#   to minutes, are too short to see here, and figures 24 and 25 of the IV notebook mark those
#   between two runs. Between the first and the last July run the LV was off through both
#   irradiation steps and once more on 07-29, for 6.9 h on both telescopes; the cell prints
#   every window drawn with its length.
# * Red: the first run after the DAQ restart of 2026-07-27, from the HV / LV cycle table
#   (`july/hv_cycles_jul.csv`).
#
# A run number that does not fit under its bar is left out and listed in the values file.

# %%
TIMELINE_FOOTER = (
    u"Bar: run start to start + configured run length (the DAQ records no stop time). Filled: "
    u"display run, p: preferred; March display runs: the highest-bias run at threshold offset "
    u"20 of each step.\nShading: bookkeeping fluence step, ending at the radiation stop (dotted; "
    u"the earliest stop the records allow). LV off: gaps longer than %g h in the LV power log. "
    u"Times UTC." % campaign.LV_OFF_MIN_H)

run_times = timeline.read_run_times()
tels = list(campaign.TELESCOPE_CHIPS)
panels = [
    dict(title="March 2026", runs=timeline.campaign_runs("march", run_times),
         display={(t, r): True for t, runs in campaign.MARCH_DISPLAY_RUNS.items() for r in runs}),
    dict(title="July 2026", runs=timeline.campaign_runs("july", run_times),
         lv_off={t: lv_log.lv_off_windows(t) for t in tels},
         restarts={t: timeline.daq_restarts(t) for t in tels},
         display=timeline.read_display_runs(campaign.DISPLAY_RUNS_JUL_CSV)),
]
fig, payload = timeline.timeline_figure(panels, tag="Campaign timeline", data="runs by telescope",
                                        footer_text=TIMELINE_FOOTER)
for title, p in payload.items():
    print("%s: %d runs, labels placed %d, skipped %s; runs off their step: %s"
          % (title, len(p["runs"]), len(p["run_number_labels_placed"]),
             p["run_number_labels_skipped"] or "none", p["runs_off_step"] or "none"))
    for tel, windows in p["lv_off_windows_utc"].items():
        print("  LV off %s: %s" % (tel, "; ".join(
            "%s to %s (%.1f h)" % (a[5:16].replace("T", " "), b[5:16].replace("T", " "),
                                   (datetime.fromisoformat(b) - datetime.fromisoformat(a))
                                   .total_seconds() / 3600) for a, b in windows)))
save(fig, "overview06_campaign_timeline", payload,
     dict(times="UTC", run_end="start + acquisition_settings.max_run_time_minutes (configured "
                               "length; no stop time is recorded)",
          fluence=campaign.FLUENCE_CONVENTION,
          display_runs="July: the display-run list; March: the campaign's MARCH_DISPLAY_RUNS, all "
                       "preferred",
          lv_off="gaps longer than LV_OFF_MIN_H = %g h between consecutive files of the LV "
                 "power log, from one file's last record to the next file's first"
                 % campaign.LV_OFF_MIN_H),
     inputs=INPUTS_TIMELINE + RUN_YAMLS)
