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
# # Pixel maps: CERN IRRAD 2026, March and July
#
# The time resolution of every pixel of every chip, as 16 x 16 maps, for one run per telescope
# and fluence step:
#
# * Figures 1-14, one per run: the four chips' maps in a row, and below each map the histogram
#   of its pixel values with a Gaussian fit.
# * Figures 15-16, one per telescope: every chip (rows) at every fluence step (columns).
# * Figures 17-18, one per telescope: each chip at 1.5e15 p/cm$^2$ in March and again in July,
#   after four months of cooling down.
#
# Every figure is saved as PNG (200 dpi) and PDF, with `<name>_values.json` beside it: every
# number drawn, the pixel maps included, the input files read and the commit of the code.
#
# To run it:
#
# 1. `source TestBeam/condor_at_lxplus/envs/load_python39.sh` (LCG_104d, the environment this
#    notebook is tested in).
# 2. The inputs are not in the repository: the tables are read from the `tables/` folder (69 MB)
#    of `/eos/user/m/musafdar/ETROC_plot_inputs/irrad_2026`. `../campaigns/irrad_2026_inputs.md`
#    lists every table with where it came from. To read your own copy, copy that whole
#    `irrad_2026` folder and point `ETROC_INPUTS` at the copy (the folder that holds `tables/`).
#    The first code cell checks that every table exists and can be read, and stops with a list of
#    all that cannot; a table whose checksum differs from the published list is named, and the run
#    goes on.
# 3. Open `maps.ipynb` in Jupyter and run all cells, or run it headless with
#    `jupyter nbconvert --to notebook --execute maps.ipynb --output-dir <somewhere>`.
#
# Figures go to `$ETROC_FIGURES/maps/`, by default `figures/maps/` next to this notebook.
# Setting `PANELS = True` in the setup cell below also writes each panel of a figure as a file of
# its own next to the compound figure, named `<stem>_01_ih7_map.png`, `<stem>_02_ih7_hist.png`
# and so on. A panel gets the text-overlap audit of its figure, printed in the notebook's
# output; it has no values file of its own.
#
# The campaign's table paths live in `../campaigns/irrad_2026.py`, the table readers in
# `../resolution/tables.py` and the map drawing in `../resolution/pixel_maps.py`. This notebook
# holds the choices made per figure: the runs and the words on each figure.
#
# `maps.ipynb` and `maps.py` are the same notebook, kept in step by jupytext. Edit either one,
# then run `python3 -m jupytext --sync maps.ipynb` and `python3 -m nbstripout maps.ipynb` in
# this folder before committing. Neither tool is in LCG_104d (the install line is in
# `../README.md`); running the notebook needs neither.
#
# After a run, `python3 -m etroc_plots.checks <figures>`, run from `TestBeam/` with `<figures>`
# the folder this notebook wrote to (`etroc_plots/notebooks/figures/maps` by default), checks
# every saved figure against the rules in `../CONVENTIONS.md`.

# %%
import os
import sys

import matplotlib
matplotlib.use("Agg")   # draw off-screen; show() displays the PNG exactly as saved
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, display

# The folder that holds the package (TestBeam/) is two folders up. Jupyter starts the kernel in
# the notebook's folder; `python maps.py` has __file__.
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
from etroc_plots.resolution import tables, pixel_maps         # noqa: E402

if campaigns.NAME != CAMPAIGN:
    raise RuntimeError("the package is running campaign %r (ETROC_CAMPAIGN) but CAMPAIGN = %r: "
                       "set one to match the other, then restart the kernel (the campaign is "
                       "read once, on the first import)" % (campaigns.NAME, CAMPAIGN))
campaign = campaigns.active

# stops, listing every missing or unreadable table; reports any table that differs from the
# published copy
check_inputs(only=("tables/",))

NOTEBOOK = "TestBeam/etroc_plots/notebooks/maps.ipynb"   # recorded as "script" in every values file
OUT = os.path.join(os.path.abspath(os.environ.get("ETROC_FIGURES") or "figures"), "maps")
os.makedirs(OUT, exist_ok=True)

PANELS = False   # True: also write each figure's single panels (see above)


def show(stem):
    """Display a saved figure, exactly as written to disk."""
    display(Image(filename=os.path.join(OUT, stem + ".png"), width=900))



# %% [markdown]
# ## The tables and the runs
#
# The resolution tables (`../resolution/tables.py` describes their columns) hold a time
# resolution for every board of every 3-board combination of a run, per pixel and per board. The
# maps use, for each board:
#
# * the anointed combination: boards 0-1-2 for boards 0, 1 and 2, boards 1-2-3 for board 3;
# * the most-populated track through each pixel (`VARIANT = "top"`);
# * tracks with at least 300 events (`FLOOR`); a pixel without such a track has no value.
#
# A pixel's value is the time resolution the table quotes for it, its error the fit error of
# that quote. The board value is the table's robust Gaussian mean over the board's pixel map
# (2.5-sigma clip). At 3.5e15 p/cm$^2$ (`BAND_STEPS` in the campaign module) the board value
# depends on the combination, so the figures give the centre of the board's values over every
# combination containing it, with their range; the pixels stay the anointed combination's. The
# histogram boxes are the exception: they give the anointed combination's value, the one their
# histogram draws, then the range and centre.
#
# `RUNS` names one run per fluence step, lowest fluence first:
#
# * March 2026: at each step, the highest-bias run among those at the standard threshold
#   offset, 20 (runs_summary);
# * July 2026: the preferred run of the campaign's display-run list
#   (`july/display_runs_jul.csv` in the inputs folder) at RFSel 2 and threshold offset 20.
#
# Figures 17-18 take their runs from `PAIR_RUNS` instead: in March the run of the 1.5e15 step
# whose leakage current matches the July run's, its lowest-bias run (run 17: H1 400 V, F1 265 V);
# in July the display run of `RUNS` at that step.
#
# Each run's bookkeeping fluence, threshold offset and RFSel are read from runs_summary, and
# every board of a run has to share them. runs_summary does not record the RFSel of the March
# runs, which all ran at the default, RFSel 2 (`DEFAULT_RFSEL`); each values file records where
# its RFSel came from (`rfsel_source`). Every fluence is the bookkeeping fluence, the
# irradiation facility's number.

# %%
VARIANT = "top"        # the most-populated track through each pixel
FLOOR = "300"          # at least 300 events per track
RUNS = {               # (campaign, telescope) -> one run per fluence step, lowest fluence first
    ("march", "h1"): [7, 11, 14, 19],
    ("march", "f1"): [2, 11, 14, 19],
    ("july", "h1"): [4, 12, 18],
    ("july", "f1"): [9, 12, 18],
}
PAIR_FLUENCE = 1.5e15  # figures 17-18: one run per campaign at this fluence
PAIR_RUNS = {          # telescope -> {campaign: run}, figures 17-18 (March: July's leakage current)
    "h1": {"march": 17, "july": 4},
    "f1": {"march": 17, "july": 9},
}
DEFAULT_RFSEL = {"march": 2}   # campaign -> the RFSel of every run, where runs_summary has none
CAMPAIGN_TEXT = {"march": "March 2026", "july": "July 2026"}   # the campaigns, in drawing order
RUNS_TEXT = (u"Runs: March, the highest-bias run at threshold offset 20 of each fluence step; "
             u"July, the preferred display run")
PAIR_RUNS_TEXT = (u"Runs: March, the run of the step at the leakage current of the July run (its "
                  u"lowest bias); July, the preferred display run")
BLOCK_TITLE = {"march": u"March 2026, run %d", "july": u"July 2026, run %d, after four months of "
                                                     u"cooling down"}
for _tel, _pair in PAIR_RUNS.items():   # PAIR_RUNS_TEXT: the July run is the display run of RUNS
    if _pair["july"] not in RUNS[("july", _tel)]:
        raise ValueError("PAIR_RUNS[%r]: July run %d is not in RUNS" % (_tel, _pair["july"]))

INPUTS_RES = [campaign.BOARD_TABLE, campaign.PIXEL_TABLE, campaign.RUNS_SUMMARY_CSV]

bt = tables.read_board_table(campaign.BOARD_TABLE)
runs_summary = tables.read_runs_summary(campaign.RUNS_SUMMARY_CSV)

grids = {}
for _camp in CAMPAIGN_TEXT:
    _runs = {t: sorted(set(r) | {PAIR_RUNS[t][c]}) for (c, t), r in RUNS.items() if c == _camp}
    _px = tables.read_pixel_table(campaign.PIXEL_TABLE, _runs, VARIANT, FLOOR,
                                  campaign.TABLE_CAMPAIGN[_camp])
    grids.update(pixel_maps.pixel_grids(_px))

BAND_KEYS = {style.fluence_key(f) for f in campaign.BAND_STEPS}   # snapped to the ladder


def settings_of(camp, tel, run):
    """The run's bookkeeping fluence, RFSel and threshold offset (tables.common_settings: one
    value for every board), and where the RFSel came from."""
    s = tables.common_settings(runs_summary, campaign.TABLE_CAMPAIGN[camp],
                               campaign.TABLE_TEL[tel], [run])
    s["rfsel_source"] = "runs_summary"
    if s["rfsel"] is None and camp in DEFAULT_RFSEL:
        s.update(rfsel=DEFAULT_RFSEL[camp], rfsel_source="DEFAULT_RFSEL (not in runs_summary)")
    if s["rfsel"] is None:
        raise ValueError("%s %s run %d: no RFSel in runs_summary" % (camp, tel, run))
    return s


def shared_settings(settings):
    """The one RFSel and threshold offset of a figure's runs, for its header and values file."""
    out = {}
    for k in ("rfsel", "offset"):
        v = {s[k] for s in settings}
        if len(v) != 1:
            raise ValueError("the runs of one figure differ in %s: %s" % (k, sorted(v)))
        out[k] = v.pop()
    out["rfsel_source"] = "; ".join(sorted({s["rfsel_source"] for s in settings}))
    return out


def run_of(camp, tel, run):
    """The run's boards (pixel_maps.run_boards, with the combination band at BAND_STEPS) and
    its settings."""
    s = settings_of(camp, tel, run)
    boards = pixel_maps.run_boards(bt, grids, runs_summary, campaign.TABLE_CAMPAIGN[camp], tel,
                                   run, VARIANT, FLOOR,
                                   band=style.fluence_key(s["fluence"]) in BAND_KEYS)
    return boards, s


def steps_in_order(camp, tel):
    """[(fluence, run), ...] of RUNS[(camp, tel)], checked to be distinct steps, lowest first."""
    steps = [(style.fluence_key(settings_of(camp, tel, r)["fluence"]), r)
             for r in RUNS[(camp, tel)]]
    fl = [f for f, _ in steps]
    if fl != sorted(set(fl)):
        raise ValueError("RUNS[%r]: fluence steps %s, expected distinct and rising"
                         % ((camp, tel), fl))
    return steps


# %% [markdown]
# ## The words on the figures

# %%
FRAME_TEXT = u"chip frame, pixel (0, 0) at the bottom right"
MAPS_TEXT = (u"Maps: %s; one colour scale, %d-%d ps, for every map; grey = no value"
             % (FRAME_TEXT, pixel_maps.VMIN, pixel_maps.VMAX))
SAT_TEXT = u"sat = pixels outside the colour scale"
CELLS_TEXT = u"in each cell the pixel's time resolution and its fit error"
HIST_TEXT = (u"Histograms: pixel values in 1 ps bins over the board value %s 6 pixel spreads, "
             u"Poisson errors, binned Gaussian fit (fit %s, %s) from %d pixels up; board value = "
             u"robust Gaussian mean of the map (2.5%s clip)"
             % (pixel_maps.PM, pixel_maps.MU, pixel_maps.SIGMA, pixel_maps.N_MIN_FIT,
                pixel_maps.SIGMA))
BAND_STEPS_TEXT = u" and ".join(style.fluence_label(f) for f in campaign.BAND_STEPS)
BAND_TEXT = (u"At %s: board value = centre of the chip's values over every combination containing "
             u"it (range given); pixels from the anointed combination" % BAND_STEPS_TEXT)
HIST_BAND_TEXT = (u"At %s: maps and histograms from the anointed combination, its board value in "
                  u"the box; combinations = range and centre of the chip's values over every "
                  u"combination containing it" % BAND_STEPS_TEXT)
LADDER_CAPTION_TEXT = u"under each map: bias, board value, pixels with a value, sat"
PANEL_TEXT = (u"%s; colour scale %d-%d ps; grey = no value"
              % (FRAME_TEXT, pixel_maps.VMIN, pixel_maps.VMAX))
PANEL_BAND_TEXT = (u"board value = centre of the chip's values over every combination containing "
                   u"it; pixels from the anointed combination")


def fluence_text(fluence):
    return u"bookkeeping fluence %s" % style.fluence_label(fluence)


def data_text(camp, tel, run, fluence):
    return u"%s, %s run %d, %s" % (campaign.TELESCOPE_TITLE[tel], CAMPAIGN_TEXT[camp], run,
                                   fluence_text(fluence))


def settings_tag(s):
    return style.settings_text(rfsel=s["rfsel"], offset=s["offset"])


def subject(b):
    return u"%s, %d V" % (b["chip"], int(round(b["hv"])))


def value_lines(b, prefix=u"board value ", anointed=True):
    """The board value as drawn: at BAND_STEPS the band centre, then the band's range and, with
    `anointed`, the value of the combination the pixels come from."""
    band = b.get("band")
    if band is None:
        return [u"%s%.1f ps" % (prefix, b["value_ps"])]
    rng = u"combinations %.1f-%.1f ps" % (band["band_lo_ps"], band["band_hi_ps"])
    if anointed:
        rng += u", this one %.1f" % band["anointed_ps"]
    return [u"%s%.1f ps" % (prefix, band["marker_ps"]), rng]


def pixels_text(b):
    return u"combo %s, %d px, sat %d" % (b["combo_chips"], b["n_pixels"], b["saturated_count"])


def hist_caption(b):
    """The box of a histogram. At BAND_STEPS it gives the value of the combination drawn, then
    the band's range and centre."""
    band = b.get("band")
    if band is None:
        return u"\n".join([pixels_text(b)] + value_lines(b))
    return u"\n".join([pixels_text(b),
                       u"board value %.1f ps (this combination)" % band["anointed_ps"],
                       u"combinations %.1f-%.1f ps, centre %.1f"
                       % (band["band_lo_ps"], band["band_hi_ps"], band["marker_ps"])])


def panel_caption(b):
    """The footer of a single map panel."""
    lines = [u"; ".join([pixels_text(b)] + value_lines(b)), PANEL_TEXT, SAT_TEXT]
    if "band" in b:
        lines.append(PANEL_BAND_TEXT)
    return u"\n".join(lines)


def selection_text(boards):
    """The selection, each anointed combination named by its chips with the chips it stands
    for."""
    combos = {}
    for b in boards:
        combos.setdefault(b["combo_chips"], []).append(b["chip"])
    return (u"Selection: %s, tracks with at least %s events, each chip from its anointed "
            u"combination: %s" % (tables.VARIANT_TEXT[VARIANT], FLOOR, u"; ".join(
                u"%s for %s" % (c, u", ".join(chips)) for c, chips in combos.items())))


def footer(boards, *lines, banded=False, band_text=BAND_TEXT):
    return u"\n".join([selection_text(boards)] + list(lines) + ([band_text] if banded else []))


def conventions():
    c = tables.conventions(VARIANT, FLOOR)
    c["maps"] = ("value_map_ps / error_map_ps [row][col]: pixel time resolution and its fit "
                 "error, None where the pixel has no value; drawn in the chip frame, pixel (0, 0) "
                 "at the bottom right")
    c["colour_scale_ps"] = [pixel_maps.VMIN, pixel_maps.VMAX]
    c["error_bar"] = "none drawn; error_map_ps holds each pixel's fit error, written in its cell"
    c["band"] = ("at the campaign's BAND_STEPS: band_lo_ps / band_hi_ps = min / max of the "
                 "board's value over every combination containing it, marker_ps = their centre "
                 "(the board value of the map captions and the ladder), anointed_ps = the "
                 "anointed combination's value (the board value of the histogram boxes)")
    c["histogram"] = ("fit: hist_counts in hist_bin_ps bins from hist_lo_ps (board value +/- 6 "
                      "pixel spreads), n_outside pixels beyond that range; Gaussian area fixed "
                      "to n_in_hist")
    return c


def save(fig, stem, payload):
    _, problems = style.save_figure(fig, OUT, stem)
    plt.close(fig)
    payload["overlap_problems"] = problems
    style.write_values(OUT, stem, payload, conventions=conventions(), script=NOTEBOOK,
                       inputs=INPUTS_RES)
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
# ## Figures 1-14: one run, four chips
#
# Top: each chip's map in the chip frame (pixel row 0, column 0 at the bottom right), every
# pixel's value and fit error written in its cell, on one colour scale, 0-120 ps, for every map;
# a pixel outside it is saturated, a pixel without a value grey. Below each map: the histogram
# of the chip's pixel values in 1 ps bins over the board value +/- 6 pixel spreads, with Poisson
# errors (the box counts any pixel outside that range), a binned Gaussian fit when the histogram
# holds at least 30 pixels, and the pulls of the fit. The fit shows the shape of the
# distribution; the board value, in the box, is the table's robust mean (at 3.5e15, the
# anointed combination's, then the range and centre over every combination).

# %%
def run_compound(stem, camp, tel, run):
    boards, s = run_of(camp, tel, run)
    banded = "band" in boards[0]
    data = data_text(camp, tel, run, s["fluence"])
    fig = pixel_maps.compound(
        boards, subjects=[subject(b) for b in boards], tag=settings_tag(s), data=data,
        footer=footer(boards, u"; ".join((MAPS_TEXT, SAT_TEXT, CELLS_TEXT)), HIST_TEXT,
                      banded=banded, band_text=HIST_BAND_TEXT),
        hist_captions=[hist_caption(b) for b in boards])
    save(fig, stem, dict(campaign=campaign.TABLE_CAMPAIGN[camp], telescope=tel, run=run,
                         fluence_p_cm2=s["fluence"], settings=shared_settings([s]),
                         boards=[pixel_maps.board_values(b) for b in boards]))
    if PANELS:
        figs = []
        for b in boards:
            chip = b["chip"].lower()
            figs.append(("%s_map" % chip, pixel_maps.map_panel(
                b, subject=subject(b), tag=settings_tag(s), data=data,
                caption=panel_caption(b))))
            figs.append(("%s_hist" % chip, pixel_maps.hist_panel(
                b, subject=subject(b), tag=settings_tag(s), data=data, caption=hist_caption(b))))
        save_panels(stem, figs)


_n = 0
for _camp, _tel in RUNS:
    for _fluence, _run in steps_in_order(_camp, _tel):
        _n += 1
        run_compound("maps%02d_%s_%s_run%d" % (_n, _tel, _camp, _run), _camp, _tel, _run)


# %% [markdown]
# ## Figures 15-16: every chip at every fluence step
#
# One map per chip (rows) and run of `RUNS` (columns), March then July, without the values in
# the cells; the caption under each map gives the chip's bias, board value, number of pixels
# with a value and of saturated pixels.

# %%
def ladder_caption(b):
    lines = value_lines(b, prefix=u"", anointed=False)
    lines[0] = u"%d V, %s, %d px, sat %d" % (int(round(b["hv"])), lines[0], b["n_pixels"],
                                            b["saturated_count"])
    return lines


def run_ladder(stem, tel):
    groups, columns, settings = [], [], []
    for camp in CAMPAIGN_TEXT:
        cols = []
        for fluence, run in steps_in_order(camp, tel):
            boards, s = run_of(camp, tel, run)
            settings.append(s)
            cols.append(dict(label=u"%s, run %d" % (style.fluence_label(fluence), run),
                             boards=boards, captions=[ladder_caption(b) for b in boards]))
            columns.append(dict(campaign=campaign.TABLE_CAMPAIGN[camp], run=run,
                                fluence_p_cm2=s["fluence"],
                                boards=[pixel_maps.board_values(b) for b in boards]))
        groups.append((CAMPAIGN_TEXT[camp], cols))
    banded = any("band" in b for _, cols in groups for c in cols for b in c["boards"])
    shared = shared_settings(settings)
    fig = pixel_maps.ladder(
        groups, campaign.TELESCOPE_CHIPS[tel], name=campaign.TELESCOPE_TITLE[tel],
        tag=settings_tag(shared), data=u" and ".join(CAMPAIGN_TEXT.values()),
        footer=footer(groups[0][1][0]["boards"],
                      u"; ".join((MAPS_TEXT, SAT_TEXT, LADDER_CAPTION_TEXT)), RUNS_TEXT,
                      banded=banded))
    save(fig, stem, dict(telescope=tel, chips=campaign.TELESCOPE_CHIPS[tel], settings=shared,
                         columns=columns))


run_ladder("maps15_h1_ladder", "h1")
run_ladder("maps16_f1_ladder", "f1")


# %% [markdown]
# ## Figures 17-18: March and July at 1.5e15
#
# Each chip at 1.5e15 p/cm$^2$ (`PAIR_FLUENCE`) in the runs of `PAIR_RUNS`: in March, right after
# irradiation, and in July, after four months of cooling down.

# %%
def run_pair(stem, tel):
    halves, blocks, settings = [], [], []
    for camp, run in PAIR_RUNS[tel].items():
        boards, s = run_of(camp, tel, run)
        if style.fluence_key(s["fluence"]) != style.fluence_key(PAIR_FLUENCE):
            raise ValueError("PAIR_RUNS: %s %s run %d is at %g, not %g"
                             % (camp, tel, run, s["fluence"], PAIR_FLUENCE))
        settings.append(s)
        halves.append((camp, run, boards))
        blocks.append((BLOCK_TITLE[camp] % run, boards, [subject(b) for b in boards]))
    shared = shared_settings(settings)
    head = dict(name=campaign.TELESCOPE_TITLE[tel], tag=settings_tag(shared),
                data=fluence_text(PAIR_FLUENCE))
    fig = pixel_maps.pair(blocks, footer=footer(halves[0][2], MAPS_TEXT + u"; " + CELLS_TEXT,
                                                PAIR_RUNS_TEXT), **head)
    save(fig, stem, dict(telescope=tel, fluence_p_cm2=PAIR_FLUENCE, settings=shared, blocks=[
        dict(campaign=campaign.TABLE_CAMPAIGN[camp], run=run, title=title,
             boards=[pixel_maps.board_values(b) for b in boards])
        for (camp, run, boards), (title, _, _) in zip(halves, blocks)]))
    if PANELS:
        block_footer = PANEL_TEXT + u";\n" + CELLS_TEXT
        figs = [("%s_block" % camp, pixel_maps.block_panel(*blk, footer=block_footer, **head))
                for (camp, _, _), blk in zip(halves, blocks)]
        for (camp, run, boards), s in zip(halves, settings):
            for b in boards:
                figs.append(("%s_%s_map" % (b["chip"].lower(), camp), pixel_maps.map_panel(
                    b, subject=subject(b), tag=settings_tag(s),
                    data=data_text(camp, tel, run, s["fluence"]), caption=panel_caption(b))))
        save_panels(stem, figs)


run_pair("maps17_h1_march_vs_july", "h1")
run_pair("maps18_f1_march_vs_july", "f1")
