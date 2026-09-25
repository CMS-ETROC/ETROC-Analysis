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
# # Merged runs: CERN IRRAD 2026, July
#
# Two runs taken back to back at the same conditions can be analysed as one, a merge, which
# roughly doubles the events behind each pixel. This notebook checks, for each July merge in use,
# that the merge agrees with its single runs:
#
# * Figures 1-6, one per merge: the time resolution of every board in each single run and in the
#   merge (top), and for each chip the 16 pixels with the most events in the merge, merged value
#   next to the single-run values (bottom).
# * Figure 7: the cost of merging, per chip and merge.
#
# Every figure is saved as PNG (200 dpi) and PDF, with `<name>_values.json` beside it: every
# number drawn, the input files read and the commit of the code.
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
# 3. Open `merged.ipynb` in Jupyter and run all cells, or run it headless with
#    `jupyter nbconvert --to notebook --execute merged.ipynb --output-dir <somewhere>`.
#
# Figures go to `$ETROC_FIGURES/merged/`, by default `figures/merged/` next to this notebook.
# Setting `PANELS = True` in the setup cell below also writes each panel of a figure as a file of
# its own next to the compound figure, named `<stem>_01_board_level.png`,
# `<stem>_02_pixels_ih7.png` and so on. A panel gets the text-overlap audit of its figure, printed
# in the notebook's output; it has no values file of its own.
#
# The campaign's table paths live in `../campaigns/irrad_2026.py`, the table readers and the
# selections every resolution figure shares in `../resolution/tables.py`. This notebook holds the
# choices made per figure.
#
# `merged.ipynb` and `merged.py` are the same notebook, kept in step by jupytext. Edit either one,
# then run `python3 -m jupytext --sync merged.ipynb` and `python3 -m nbstripout merged.ipynb` in
# this folder before committing. Neither tool is in LCG_104d (the install line is in
# `../README.md`); running the notebook needs neither.
#
# After a run, `python3 -m etroc_plots.checks <figures>`, run from `TestBeam/` with `<figures>`
# the folder this notebook wrote to (`etroc_plots/notebooks/figures/merged` by default), checks
# every saved figure against the rules in `../CONVENTIONS.md`.

# %%
import os
import sys

import matplotlib
matplotlib.use("Agg")   # draw off-screen; show() displays the PNG exactly as saved
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from IPython.display import Image, display

# The folder that holds the package (TestBeam/) is two folders up. Jupyter starts the kernel in
# the notebook's folder; `python merged.py` has __file__.
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
from etroc_plots.resolution import tables                     # noqa: E402

if campaigns.NAME != CAMPAIGN:
    raise RuntimeError("the package is running campaign %r (ETROC_CAMPAIGN) but CAMPAIGN = %r: "
                       "set one to match the other, then restart the kernel (the campaign is "
                       "read once, on the first import)" % (campaigns.NAME, CAMPAIGN))
campaign = campaigns.active

# stops, listing every missing or unreadable table; reports any table that differs from the
# published copy
check_inputs(only=("tables/",))

NOTEBOOK = "TestBeam/etroc_plots/notebooks/merged.ipynb"   # recorded as "script" in every values file
OUT = os.path.join(os.path.abspath(os.environ.get("ETROC_FIGURES") or "figures"), "merged")
os.makedirs(OUT, exist_ok=True)

PANELS = False   # True: also write each figure's single panels (see above)


def show(stem):
    """Display a saved figure, exactly as written to disk."""
    display(Image(filename=os.path.join(OUT, stem + ".png"), width=900))


# %% [markdown]
# ## The tables and the merges
#
# The resolution tables (`../resolution/tables.py` describes their columns) hold a time
# resolution for every board of every 3-board combination of a run, per pixel and per board, as
# the resolution quote (`TestBeam/condor_at_lxplus/utils/quote_resolution.py`) gives it. A track
# is one path of hit pixels through the combination's three boards, one pixel per board; its
# events are the particles recorded along it. The figures here use one value per board:
#
# * the anointed combination: boards 0-1-2 for boards 0, 1 and 2, boards 1-2-3 for board 3 (the
#   footers name each by its chips);
# * the most-populated track through each pixel (`VARIANT = "top"`);
# * tracks with at least 300 events (`FLOOR`).
#
# A board's value is a robust Gaussian mean over its pixel map, its bar the pixel-to-pixel
# spread. A merge sits in the merged tables under the name `<telescope>_grp_run<a>_<b>`.
#
# At 3.5e15 p/cm$^2$ (`BAND_STEPS` in the campaign module) a board's width depends on the
# combination it is solved in, so every board value at that step, of a single run or of a merge,
# is drawn as a band instead of a point: the shaded box spans the board's value over every
# combination containing it, the chip's marker sits at the centre of the box, the bar is the
# anointed combination's pixel spread. Single runs and merge are then compared on the same
# footing. Pixel values are always the anointed combination's.
#
# Each merge's runs share one fluence, RFSel and threshold offset, and each board keeps its chip
# and bias (`tables.common_settings` reads them from `runs_summary.csv` and stops if they differ);
# the header states RFSel and offset, the legend the fluence.
#
# `MERGES` lists the six July merges drawn, one figure each. H1 runs 14 and 15 were merged too,
# but neither run is used for results (superseded by runs 18 and 19 at the same step), so that
# merge is left out. The merged tables also hold three DESY August 2026 merges, which belong to
# another campaign.

# %%
VARIANT = "top"        # the most-populated track through each pixel
FLOOR = "300"          # at least 300 events per track
N_PIX = 16             # pixels per chip in the lower row: the 16 with the most events in the merge
TABLE_CAMPAIGN = campaign.TABLE_CAMPAIGN["july"]
DATA_TEXT = "July 2026"   # the data, as the figure headers name it
BAND_KEYS = {style.fluence_key(f) for f in campaign.BAND_STEPS}   # snapped to the ladder
MERGES =[("h1", (3, 4)), ("h1", (6, 7)), ("h1", (11, 12)), ("h1", (18, 19)),
          ("f1", (3, 4)), ("f1", (46, 47))]


def merge_settings(tel, runs):
    """The merge's common fluence, RFSel and threshold offset (tables.common_settings); every
    resolution figure states the RFSel, so a run without one stops here."""
    settings = tables.common_settings(runs_summary, TABLE_CAMPAIGN, campaign.TABLE_TEL[tel], runs)
    if settings["rfsel"] is None:
        raise ValueError("%s runs %s: no RFSel in runs_summary" % (tel, runs))
    return settings


INPUTS_RES = [campaign.BOARD_TABLE_MERGED, campaign.BOARD_TABLE, campaign.PIXEL_TABLE_MERGED,
              campaign.PIXEL_TABLE, campaign.RUNS_SUMMARY_CSV]

bt_merged = tables.read_board_table(campaign.BOARD_TABLE_MERGED)
bt_single = tables.read_board_table(campaign.BOARD_TABLE)
runs_summary = tables.read_runs_summary(campaign.RUNS_SUMMARY_CSV)

_single_runs, _merge_names = {}, {}
for _tel, _runs in MERGES:
    _single_runs.setdefault(_tel, []).extend(_runs)
    _merge_names.setdefault(_tel, []).append(tables.merge_name(_tel, _runs))
px_single = tables.read_pixel_table(campaign.PIXEL_TABLE, _single_runs, VARIANT, FLOOR,
                                    TABLE_CAMPAIGN)
px_merged = tables.read_pixel_table(campaign.PIXEL_TABLE_MERGED, _merge_names, VARIANT, FLOOR,
                                    TABLE_CAMPAIGN)


# %% [markdown]
# ## Figures 1-6: merged against single runs
#
# One figure per merge. Top: every board's value in each single run and in the merge (the
# rightmost column; a point, not a band, has a thick black marker edge there), boards side by
# side within a column in slot order. Bottom, one panel per chip: the 16 pixels with the most
# events in the merge, in (row, col) order, merged value with its fit error, the single-run
# values beside it, lighter, in run order from left to right. The footer gives the selection
# and, per anointed combination, the share of the merge's tracks the quote drops as degenerate (a
# solved width below 0.35 or above 0.95 of the smallest pair width it came from;
# `TestBeam/condor_at_lxplus/README.md`). The run-merging study judged each merge on this share:
# a time step between the runs raises it above the single runs' shares, which these tables do
# not carry.

# %%
def merge_data(tel, runs):
    """Everything figures 1-6 draw for one merge: the board points of each single run and of the
    merge, the lower-row pixels, the runs' common settings and the degenerate shares of the
    merge's anointed combinations."""
    group = tables.merge_name(tel, runs)
    settings = merge_settings(tel, runs)
    fluence = settings["fluence"]
    banded = style.fluence_key(fluence) in BAND_KEYS
    mrows = tables.board_rows(bt_merged, tel, group, VARIANT, FLOOR, TABLE_CAMPAIGN)
    if not len(mrows) or mrows.board_idx.duplicated().any():
        raise ValueError("%s: %d anointed merged board rows, boards %s"
                         % (group, len(mrows), list(mrows.board_idx)))
    for r in runs:
        sboards = tables.board_rows(bt_single, tel, r, VARIANT, FLOOR, TABLE_CAMPAIGN).board_idx
        if sorted(sboards) != sorted(mrows.board_idx):
            raise ValueError("%s run %d has anointed boards %s, the merge %s"
                             % (tel, r, sorted(sboards), sorted(mrows.board_idx)))

    boards, combos = [], {}
    for _, mr in mrows.iterrows():
        b, chip = int(mr.board_idx), mr.board_chip
        if style.chip_short(chip) != campaign.TELESCOPE_CHIPS[tel][b]:
            # combo_chips names a combo's chips from TELESCOPE_CHIPS
            raise ValueError("%s board %d is %s, TELESCOPE_CHIPS says %s"
                             % (group, b, chip, campaign.TELESCOPE_CHIPS[tel][b]))
        combos[mr.combo] = float(mr.degenerate_share)
        points = []
        for r in runs:
            srow = tables.board_rows(bt_single, tel, r, VARIANT, FLOOR, TABLE_CAMPAIGN, board_idx=b)
            if len(srow) != 1 or srow.iloc[0].board_chip != chip:
                raise ValueError("%s run %d board %d: %d anointed rows, chip %s in the merge"
                                 % (tel, r, b, len(srow), chip))
            srow = srow.iloc[0]
            pt = dict(run=str(r), value_ps=float(srow.value_ps),
                      pixel_spread_ps=float(srow.pixel_spread_ps), n_pixels=int(srow.n_pixels),
                      merged=False)
            if banded:
                pt["band"] = tables.band(bt_single, tel, r, b, VARIANT, FLOOR, TABLE_CAMPAIGN)
            points.append(pt)
        pt = dict(run=group, value_ps=float(mr.value_ps),
                  pixel_spread_ps=float(mr.pixel_spread_ps), n_pixels=int(mr.n_pixels),
                  merged=True)
        if banded:
            pt["band"] = tables.band(bt_merged, tel, group, b, VARIANT, FLOOR, TABLE_CAMPAIGN)
        points.append(pt)
        boards.append(dict(board_idx=b, chip=chip, combo=mr.combo, points=points))

    mpx = px_merged[(px_merged.telescope == tel) & (px_merged.run == group)]
    spx = px_single[px_single.telescope == tel]
    pixel_boards = []
    for bd in boards:
        top = (mpx[mpx.board_idx == bd["board_idx"]]
               .sort_values(["nevt_sum", "row", "col"], ascending=[False, True, True],
                            kind="mergesort")
               .drop_duplicates(subset=["row", "col"])
               .head(N_PIX)
               .sort_values(["row", "col"]))
        if not len(top):
            raise ValueError("%s %s: no pixels in the merged pixel table" % (group, bd["chip"]))
        pixels = []
        for _, pr in top.iterrows():
            row, col = int(pr.row), int(pr.col)
            singles = []
            for r in runs:
                s = spx[(spx.run.astype(str) == str(r)) & (spx.board_idx == bd["board_idx"])
                        & (spx.row == row) & (spx.col == col)]
                if len(s):
                    singles.append(dict(run=r, value_ps=float(s.iloc[0].value_ps),
                                        err_ps=float(s.iloc[0].err_ps)))
                else:
                    singles.append(dict(run=r, value_ps=None, err_ps=None, missing=True))
            pixels.append(dict(row=row, col=col, nevt_sum=float(pr.nevt_sum),
                               merged_value_ps=float(pr.value_ps),
                               merged_err_ps=float(pr.err_ps), singles=singles))
        pixel_boards.append(dict(board_idx=bd["board_idx"], chip=bd["chip"], pixels=pixels))

    return dict(telescope=tel, group=group, runs=list(runs), fluence=fluence, settings=settings,
                boards=boards, pixel_boards=pixel_boards, combos=combos, banded=banded)


def settings_text(d):
    return style.settings_text(rfsel=d["settings"]["rfsel"], offset=d["settings"]["offset"])


def draw_board_panel(ax, d, scale=1.0, legend_rows=1):
    """The board values of each run and of the merge; the chip legend in `legend_rows` rows,
    one row by default."""
    style.apply_style(scale)
    style.style_axes(ax, scale=scale)
    n = len(d["boards"])
    offsets = np.linspace(-0.18, 0.18, n) if n > 1 else [0.0]
    labels = [str(r) for r in d["runs"]] + ["+".join(str(r) for r in d["runs"])]
    xpos = list(range(len(labels)))
    chips, lo, hi = [], [], []
    for off, b in zip(offsets, d["boards"]):
        chips.append(b["chip"])
        for x, pt in zip(xpos, b["points"]):
            if "band" in pt:
                bd = pt["band"]
                tables.draw_band(ax, x + off, bd, style.fluence_color(d["fluence"]),
                                 half_width=0.05, marker=style.chip_marker(b["chip"]),
                                 markersize=8.0 * scale)
                lo.append(min(bd["band_lo_ps"], bd["marker_ps"] - bd["bar_spread_ps"]))
                hi.append(max(bd["band_hi_ps"], bd["marker_ps"] + bd["bar_spread_ps"]))
                continue
            kw = style.point_style(b["chip"], d["fluence"], merged=pt["merged"], line=False,
                                   scale=scale)
            ax.errorbar([x + off], [pt["value_ps"]], yerr=[pt["pixel_spread_ps"]], **kw)
            lo.append(pt["value_ps"] - pt["pixel_spread_ps"])
            hi.append(pt["value_ps"] + pt["pixel_spread_ps"])
    # the legends take the top of the axis: keep the data below them
    ymin, ymax = np.nanmin(lo), np.nanmax(hi)
    span = ymax - ymin
    ax.set_ylim(ymin - 0.05 * span, ymax + 0.4 * span)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, len(xpos) - 0.5)
    ax.set_xlabel("Run")
    ax.set_ylabel(style.YLABEL_RES)
    style.panel_title(ax, d["telescope"])
    # bands carry no merged marker: the merge is the rightmost column
    n_entries = len(chips) + (0 if d["banded"] else 1)
    style.two_legends(ax, steps=[d["fluence"]], chips=chips, scale=scale, merged=not d["banded"],
                      ncol_c=-(-n_entries // legend_rows))


def draw_pixel_panel(ax, d, pb, scale=0.62):
    style.style_axes(ax, scale=scale)
    pixels = pb["pixels"]
    xs = list(range(1, len(pixels) + 1))
    mkw = style.point_style(pb["chip"], d["fluence"], merged=True, line=False, scale=scale)
    n_singles = len(d["runs"])
    sdx = np.linspace(-0.16, 0.16, n_singles) if n_singles > 1 else [0.0]
    for x, px in zip(xs, pixels):
        ax.errorbar([x], [px["merged_value_ps"]], yerr=[px["merged_err_ps"]], **mkw)
        for dx, sv in zip(sdx, px["singles"]):
            if sv.get("missing"):
                continue
            skw = style.point_style(pb["chip"], d["fluence"], merged=False, line=False, scale=scale)
            skw["alpha"] = 0.55
            ax.errorbar([x + dx], [sv["value_ps"]], yerr=[sv["err_ps"]], **skw)
    ax.set_xticks(xs)
    ax.set_xticklabels(["%d,%d" % (p["row"], p["col"]) for p in pixels], rotation=90,
                       fontsize=style.sizes(scale)["tick"] * 0.85)
    ax.set_xlabel("Pixel (row,col)")
    ax.set_ylabel(style.YLABEL_RES)
    style.panel_title(ax, d["telescope"], extra=style.chip_short(pb["chip"]), scale=scale)


def runs_text(d):
    return "+".join(str(r) for r in d["runs"])


def merged_footer(d):
    runs = [str(r) for r in d["runs"]]
    # each anointed combination by its chips, with the chips it stands for and its share
    anointed = []
    for c, v in d["combos"].items():
        chips = [style.chip_short(b["chip"]) for b in d["boards"] if b["combo"] == c]
        anointed.append("%s for %s (%.2f %%)" % (tables.combo_chips(c, d["telescope"]),
                                                 ", ".join(chips), v * 100.0))
    n_short = min(len(pb["pixels"]) for pb in d["pixel_boards"])
    lines = ["Selection: %s, tracks with at least %s events, each chip from its anointed "
             "combination; runs %s and %s merged" % (tables.VARIANT_TEXT[VARIANT], FLOOR,
                                                     ", ".join(runs[:-1]), runs[-1]),
             "Top: board value, bar = pixel-to-pixel spread. Bottom: the %s%d pixels with the "
             "most events in the merge, merged value (black edge) and each run's (lighter, in "
             "run order left to right), bar = fit error"
             % ("" if n_short == N_PIX else "(at most) ", N_PIX),
             "Anointed combinations, with the share of the merge's tracks dropped as degenerate: "
             + "; ".join(anointed)]
    if d["banded"]:
        lines.append("Bands at %s: %s" % (style.fluence_label(d["fluence"]), tables.BAND_FOOTER))
    return "\n".join(lines)


def merged_compound(d):
    """Board panel on top, one pixel panel per chip below it, footer under both."""
    n_chips = len(d["boards"])
    W = 17.0
    H_TOP, H_ROWGAP, H_BOT, H_BOTMARGIN = 5.2, 1.35, 4.2, 1.2
    H = H_TOP + H_ROWGAP + H_BOT + H_BOTMARGIN
    fig = plt.figure(figsize=(W, H))
    ml, mr = 1.05, 0.35
    top_y0 = (H_BOTMARGIN + H_BOT + H_ROWGAP) / H
    ax_top = fig.add_axes([ml / W, top_y0, (W - ml - mr) / W, H_TOP / H * 0.845])
    draw_board_panel(ax_top, d, scale=1.0)
    # right_lines merges this line with the panel title; the last call sets the size of both
    style.right_lines(ax_top, ["%s runs %s, merged vs single" % (DATA_TEXT, runs_text(d)),
                               settings_text(d)], style.sizes(0.85)["title"])
    style.compound_header([ax_top], scale=0.85)

    bot_y0 = H_BOTMARGIN / H
    gap = 0.032
    aw = ((W - ml - mr) / W - gap * (n_chips - 1)) / n_chips
    for i, pb in enumerate(d["pixel_boards"]):
        ax = fig.add_axes([ml / W + i * (aw + gap), bot_y0, aw, H_BOT / H])
        draw_pixel_panel(ax, d, pb, scale=0.62)
    style.footer(fig, merged_footer(d), scale=0.7, x=ml / W, y=0.01)
    return fig


def merged_panels(d, stem):
    """PANELS=True: the board panel and each pixel panel as figures of their own."""
    paths = []
    subject = "%s runs %s" % (DATA_TEXT, runs_text(d))
    W, H = 11.5, 5.95
    ml, mr, mt, mb = 1.1, 0.4, 0.7, 1.15
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([ml / W, mb / H, (W - ml - mr) / W, (H - mt - mb) / H])
    draw_board_panel(ax, d, scale=1.0, legend_rows=2)
    style.header(ax, name=subject, tag=settings_text(d), scale=0.55, pad=6)
    print(style.check_no_clipping(fig, "%s panel board_level" % stem))
    paths += style.save_panel(fig, OUT, stem, 1, "board_level")
    plt.close(fig)
    for i, pb in enumerate(d["pixel_boards"], start=2):
        name = "pixels_%s" % style.chip_short(pb["chip"]).lower()
        W, H = 11.5, 6.35
        ml, mr, mt, mb = 1.1, 0.4, 0.7, 1.7
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes([ml / W, mb / H, (W - ml - mr) / W, (H - mt - mb) / H])
        draw_pixel_panel(ax, d, pb, scale=1.0)
        style.header(ax, name=subject, tag=settings_text(d), scale=0.55, pad=6)
        print(style.check_no_clipping(fig, "%s panel %s" % (stem, name)))
        paths += style.save_panel(fig, OUT, stem, i, name)
        plt.close(fig)
    style.flatten_panels(OUT, stem, paths)


def merged_values(d, problems):
    boards = [dict(board_idx=b["board_idx"], chip=b["chip"], combo=b["combo"],
                   degenerate_share=d["combos"][b["combo"]], points=b["points"])
              for b in d["boards"]]
    pixels = [dict(chip=pb["chip"], n_pixels=len(pb["pixels"]), pixels=pb["pixels"])
              for pb in d["pixel_boards"]]
    return dict(telescope=d["telescope"], group=d["group"], singles=d["runs"],
                fluence_p_cm2=d["fluence"], rfsel=d["settings"]["rfsel"],
                threshold_offset=d["settings"]["offset"], combos=d["combos"], boards=boards,
                pixels=pixels, campaign=TABLE_CAMPAIGN,
                n_band_points=sum("band" in p for b in d["boards"] for p in b["points"]),
                band_definition=tables.BAND_FOOTER, overlap_problems=problems)


def merged_vs_single(stem, tel, runs):
    if (tel, tuple(runs)) not in MERGES:
        raise ValueError("%s runs %s is not in MERGES, whose pixels were read above"
                         % (tel, runs))
    d = merge_data(tel, runs)
    fig = merged_compound(d)
    _, problems = style.save_figure(fig, OUT, stem)
    plt.close(fig)
    conventions = tables.conventions(VARIANT, FLOOR)
    conventions["pixel_error_bar"] = "lower row: the fit error of the pixel's width (err_ps)"
    if d["banded"]:
        conventions["band"] = ("a point with a band is drawn at band.marker_ps; its value_ps "
                               "stays the anointed combination's")
    style.write_values(OUT, stem, merged_values(d, problems), conventions=conventions,
                       script=NOTEBOOK, inputs=INPUTS_RES)
    print("%s: audit %s" % (stem, problems or "clean"))
    show(stem)
    if PANELS:
        merged_panels(d, stem)


# %% [markdown]
# ### Figure 1: H1, runs 3+4 (1.5e15 p/cm$^2$)

# %%
merged_vs_single("merged01_h1_runs3_4", "h1", (3, 4))

# %% [markdown]
# ### Figure 2: H1, runs 6+7 (1.5e15 p/cm$^2$)
#
# IH13 was not biased in these runs, so the merge has three boards.

# %%
merged_vs_single("merged02_h1_runs6_7", "h1", (6, 7))

# %% [markdown]
# ### Figure 3: H1, runs 11+12 (2e15 p/cm$^2$)

# %%
merged_vs_single("merged03_h1_runs11_12", "h1", (11, 12))

# %% [markdown]
# ### Figure 4: H1, runs 18+19 (3.5e15 p/cm$^2$)
#
# Every board value is a band, in the single runs and in the merge.

# %%
merged_vs_single("merged04_h1_runs18_19", "h1", (18, 19))

# %% [markdown]
# ### Figure 5: F1, runs 3+4 (1.5e15 p/cm$^2$)

# %%
merged_vs_single("merged05_f1_runs3_4", "f1", (3, 4))

# %% [markdown]
# ### Figure 6: F1, runs 46+47 (3.5e15 p/cm$^2$, RFSel 1)
#
# Every board value is a band, as in figure 4. These two runs were taken at RFSel 1, the other
# merges at RFSel 2.

# %%
merged_vs_single("merged06_f1_runs46_47", "f1", (46, 47))

# %% [markdown]
# ## Figure 7: the cost of merging
#
# For each pixel present in the merge and in each of its runs,
# cost$_\mathrm{px}$ = σ$_\mathrm{merged}$ − sqrt(Σ$_r$ p$_r$ σ$_r^2$), where σ$_r$ is the pixel's
# width in run r and p$_r$ that run's share of the merge's tracks with at least 300 events (a
# share of tracks, not of events). The pixel maps here are not the ones of figures 1-6: each is
# the resolution quote's own per-pixel map (`quote_resolution.per_pixel`), with every combination
# containing the board pooled and tracks of at least 300 events. One panel per merge, chips in
# slot order: the
# median cost$_\mathrm{px}$ with a bar from the 16th to the 84th percentile (filled circle), and
# the same difference taken on the board values (hollow diamond). A merge of runs at one condition
# should cost nothing; the medians and board differences stay within about 1 ps of zero.
#
# The table (`tables/merged/per_pixel_cost_summary.csv`) was made by `build_per_pixel_cost.py`
# of the run-merging study, which is not in this repository yet
# (`../campaigns/irrad_2026_inputs.md` lists it with the other builders).

# %%
COST_STEM = "merged07_merge_cost"
COST_SCALE = 0.6       # text and tick size of the grid, times the single-panel reference
COST_NCOL = 4          # panels per row; the legend takes the slot after the last merge
COST_FLOOR = 300       # the event floor build_per_pixel_cost.py applied to its maps
cost = pd.read_csv(campaign.MERGE_COST_CSV)

COST_CONVENTIONS = {
    "selection": "per pixel: the quote's per-pixel map (quote_resolution.per_pixel), every "
                 "combination containing the board pooled, tracks with at least %d events; "
                 "pixels present in the merge and in each of its runs" % COST_FLOOR,
    "cost_px": "sigma_merged - sqrt(sum_r p_r sigma_r^2), p_r = run r's share of the merge's "
               "tracks with at least %d events" % COST_FLOOR,
    "marker": "median of cost_px over the common pixels, bar = 16th to 84th percentile",
    "board_cost": "merged board value - sqrt(sum_r p_r value_r^2), on the quote's board values "
                  "at the same event floor",
    "fluence": campaign.FLUENCE_CONVENTION}
COST_FOOTER = (r"Cost per pixel = $\sigma_\mathrm{merged} - \sqrt{\Sigma_r\, p_r \sigma_r^2}$, "
               r"$p_r$ = run r's share of the merge's tracks, over the pixels present in the "
               "merge and in each run; per-pixel maps pooled over every combination containing "
               "the chip, tracks with at least %d events\n" % COST_FLOOR +
               "Filled circle = median over the pixels, bar = 16th to 84th percentile; hollow "
               "diamond = the same difference on the board values")
COST_MEDIAN = dict(marker="o", markersize=7.0, linestyle="none", capsize=3.0, elinewidth=1.4)
COST_BOARD = dict(marker="D", markersize=6.5, markerfacecolor=style.SURFACE,
                  markeredgewidth=1.6, linestyle="none")


def cost_slots(rows, tel):
    """One merge's rows as (board_idx, row), in slot order."""
    chips = campaign.TELESCOPE_CHIPS[tel]
    return sorted(((chips.index(style.chip_short(r.chip)), r) for _, r in rows.iterrows()),
                  key=lambda x: x[0])


def draw_cost_panel(ax, rows, tel, fluence, scale):
    style.style_axes(ax, scale=scale)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    color = style.fluence_color(fluence)
    slots = cost_slots(rows, tel)
    xs = list(range(1, len(slots) + 1))
    for x, (_, r) in zip(xs, slots):
        if r.status != "ok" or pd.isna(r.median_cost_px_ps):
            ax.text(x, 0, "no map", ha="center", va="center", fontsize=style.sizes(scale)["ann"],
                    color=style.INK_MUTED)
            continue
        lo = r.median_cost_px_ps - r.p16_cost_px_ps
        hi = r.p84_cost_px_ps - r.median_cost_px_ps
        ax.errorbar([x], [r.median_cost_px_ps], yerr=[[lo], [hi]], color=color,
                    markerfacecolor=color, markeredgecolor=color, **COST_MEDIAN)
        if not pd.isna(r.board_cost_official_ps):
            ax.plot([x + 0.15], [r.board_cost_official_ps], markeredgecolor=color, **COST_BOARD)
    ax.axhline(0.0, color=style.INK_MUTED, lw=0.8, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([style.chip_short(r.chip) for _, r in slots])
    ax.set_xlim(0.5, len(xs) + 0.5)
    ax.set_xlabel("Chip")
    ax.set_ylabel("Merge cost [ps]")
    return slots


def cost_legend(ax, fluences, scale):
    """The legend slot: the fluence colours and the two markers."""
    s = style.sizes(scale)
    ax.axis("off")
    lf = ax.legend(handles=style.fluence_handles(fluences, scale), loc="upper left",
                   fontsize=s["legend"], title="fluence", title_fontsize=s["legend_title"])
    ax.add_artist(lf)
    median = Line2D([], [], marker=COST_MEDIAN["marker"], markersize=COST_MEDIAN["markersize"],
                    linestyle="none", color=style.INK,
                    label="cost per pixel: median,\nbar 16th to 84th percentile")
    board = Line2D([], [], color=style.INK, markeredgecolor=style.INK,
                   label="the same on the board values",
                   **COST_BOARD)
    # stacked under the fluence legend, whose height depends on the number of steps
    box = lf.get_window_extent(ax.figure.canvas.get_renderer())
    bottom = ax.transAxes.inverted().transform(box)[0][1]
    ax.legend(handles=[median, board], loc="upper left", bbox_to_anchor=(0.0, bottom - 0.04),
              fontsize=s["legend"])


def _num(v, cast=float):
    return None if pd.isna(v) else cast(v)


cost_merges = []
for tel, runs in MERGES:
    group = tables.merge_name(tel, runs)
    rows = cost[cost.group == group]
    if not len(rows) or rows.chip.duplicated().any():
        raise ValueError("%s: %d merge-cost rows, chips %s" % (group, len(rows), list(rows.chip)))
    settings = merge_settings(tel, runs)
    runs_label = "runs %s" % "+".join(str(r) for r in runs)
    vendor = campaign.TELESCOPE_TITLE[tel].split(", ")[-1]   # "H1 telescope, HPK" -> "HPK"
    # short subjects: the first shares its line with the experiment text; the header's
    # facility line gives each RFSel's feedback resistor
    cost_merges.append(dict(tel=tel, runs=runs, group=group, rows=rows, settings=settings,
                            runs_label=runs_label,
                            subject="%s (%s), %s, RFSel %d" % (tel.upper(), vendor, runs_label,
                                                               settings["rfsel"])))
rfsels = sorted({m["settings"]["rfsel"] for m in cost_merges})
offsets = sorted({m["settings"]["offset"] for m in cost_merges})
cost_tag = style.settings_text(rfsel=rfsels[0] if len(rfsels) == 1 else rfsels,
                               offset=offsets[0] if len(offsets) == 1 else offsets)

style.apply_style(COST_SCALE)
W = 18.0
ML, MR, GAP = 1.0, 0.3, 1.0        # inches: left margin, right margin, gap between columns
MT, RGAP, MB, AH = 0.6, 1.15, 0.85, 2.7   # top, gap between rows, bottom, panel height
AW = (W - ML - MR - (COST_NCOL - 1) * GAP) / COST_NCOL
n_rows = -(-(len(MERGES) + 1) // COST_NCOL)
H = MT + n_rows * AH + (n_rows - 1) * RGAP + MB
fig = plt.figure(figsize=(W, H))
axes = [fig.add_axes([(ML + (i % COST_NCOL) * (AW + GAP)) / W,
                      (MB + (n_rows - 1 - i // COST_NCOL) * (AH + RGAP)) / H, AW / W, AH / H])
        for i in range(len(MERGES) + 1)]
cost_groups = []
for ax, m in zip(axes, cost_merges):
    slots = draw_cost_panel(ax, m["rows"], m["tel"], m["settings"]["fluence"], COST_SCALE)
    cost_groups.append(dict(
        group=m["group"], telescope=m["tel"], runs=list(m["runs"]),
        fluence_p_cm2=m["settings"]["fluence"], rfsel=m["settings"]["rfsel"],
        threshold_offset=m["settings"]["offset"], boards=[
            dict(board_idx=b, chip=r.chip, status=r.status,
                 median_cost_px_ps=_num(r.median_cost_px_ps),
                 p16_cost_px_ps=_num(r.p16_cost_px_ps), p84_cost_px_ps=_num(r.p84_cost_px_ps),
                 board_cost_official_ps=_num(r.board_cost_official_ps),
                 n_pixels_common=_num(r.n_pixels_common, int))
            for b, r in slots]))
cost_legend(axes[len(MERGES)], [m["settings"]["fluence"] for m in cost_merges], COST_SCALE)
top_row = axes[:COST_NCOL]
style.compound_header(top_row, tag=cost_tag, data=DATA_TEXT, scale=0.5,
                      subjects=[m["subject"] for m in cost_merges[:COST_NCOL]])
for ax, m in zip(axes[COST_NCOL:], cost_merges[COST_NCOL:]):
    style.right_lines(ax, [m["subject"]], style.sizes(0.5)["title"])
style.footer(fig, COST_FOOTER, scale=0.7, x=ML / W, y=0.01)
_, cost_problems = style.save_figure(fig, OUT, COST_STEM)
plt.close(fig)
style.write_values(OUT, COST_STEM, dict(groups=cost_groups, overlap_problems=cost_problems),
                   conventions=COST_CONVENTIONS, script=NOTEBOOK,
                   inputs=[campaign.MERGE_COST_CSV, campaign.RUNS_SUMMARY_CSV])
print("%s: audit %s" % (COST_STEM, cost_problems or "clean"))
show(COST_STEM)

if PANELS:
    style.apply_style(1.0)
    paths = []
    for index, m in enumerate(cost_merges, start=1):
        pw, ph = 11.0, 5.51
        pml, pmr, pmt, pmb = 1.45, 0.4, 0.66, 0.9
        pfig = plt.figure(figsize=(pw, ph))
        pax = pfig.add_axes([pml / pw, pmb / ph, (pw - pml - pmr) / pw, (ph - pmt - pmb) / ph])
        draw_cost_panel(pax, m["rows"], m["tel"], m["settings"]["fluence"], 1.0)
        style.panel_title(pax, m["tel"])
        style.header(pax, name="%s %s" % (DATA_TEXT, m["runs_label"]),
                     tag=style.settings_text(rfsel=m["settings"]["rfsel"],
                                             offset=m["settings"]["offset"]), scale=0.55, pad=6)
        print(style.check_no_clipping(pfig, "%s panel %s" % (COST_STEM, m["group"])))
        paths += style.save_panel(pfig, OUT, COST_STEM, index, m["group"])
        plt.close(pfig)
    style.flatten_panels(OUT, COST_STEM, paths)
