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
# # Universality of the time-walk correction, DESY and CERN IRRAD 2026
#
# Can a time-walk correction (TWC) fitted on one source (a run, a board, a pixel) be applied to
# another target without losing time resolution? The correction of a pixel is a quadratic in
# TOT, delta(TOT) = c2 TOT^2 + c1 TOT + c0, fitted per run on the pixel's tracks. The notebook
# first shows the curves themselves: one pixel's correction in runs that differ along one axis,
# and how far each target run's TOT reaches beyond the source's. Then the cost of transferring:
# the pair penalty, how much wider a pair of boards gets with the source's correction than with
# the target's own, on the same events.
#
# * Figure un01: the correction of one pixel against TOT along four axes, with the TOT of the
#   fitted events below: the angle to the beam (DESY May 2026, 0 to 60 deg), the sensor bias
#   (CERN IRRAD March 2026, H1), the fluence step (March, H1, the runs of the fluence ladder)
#   and the four months between March and July at 1.5e15 (H1 and F1).
# * Figure un02: the penalty of every extrapolating pair row of the comparison axes against the
#   share of the target's TOT the source covers and against the lever, with the interpolating
#   rows as a band.
# * Figure un03: the penalty of each of the 43 level cells, interpolating family and all target
#   triples, with the study's verdicts and each cell's margin.
# * Figure un04: the penalty before and after refitting the linear term of the correction on
#   the target, per pair row of the eleven re-scored cells.
# * Figure un05: the threshold-offset and RFSel steps of August, against a floor at the same
#   settings.
# * Figure un06: three controls: the floor on one day, a board correction pooled over pixels
#   against the pixel's own, and March to July against the bias step.
#
# No comparison with an IRRAD run at 2e15 or 3.5e15 is drawn; the cell before un02 says what
# that leaves out.
#
# Every figure is saved as PNG (200 dpi) and PDF, with `<name>_values.json` beside it: every
# number drawn, the input files read and the commit of the code.
#
# To run it:
#
# 1. `source TestBeam/condor_at_lxplus/envs/load_python39.sh` (LCG_104d, the environment this
#    notebook is tested in).
# 2. The figures read small tables that are not in the repository, from
#    `/eos/user/m/musafdar/ETROC_plot_inputs/irrad_2026`: the six tables of `universality/`,
#    `tables/runs_summary.csv` and `july/good_runs.csv`. `../campaigns/irrad_2026_inputs.md`
#    lists every table with where it came from. The `universality/` tables come from the
#    TWC-transfer study (its specification of 2026-09-09): its coefficient bank, one file of
#    fits per run, and its result tables; `python -m etroc_plots.universality.reduce` (run from
#    `TestBeam/`) remakes them from those (15 s on one lxplus core). To read your
#    own copy, copy that whole `irrad_2026` folder and point `ETROC_INPUTS` at the copy. The
#    first code cell checks that every input exists and can be read, and stops with a list of
#    all that cannot; a table whose checksum differs from the published list is named, and the
#    run goes on.
# 3. Open `universality.ipynb` in Jupyter and run all cells, or run it headless with
#    `jupyter nbconvert --to notebook --execute universality.ipynb --output-dir <somewhere>`.
#
# Figures go to `$ETROC_FIGURES/universality/`, by default `figures/universality/` next to this
# notebook.
#
# The tables read:
#
# * `universality/twc_bank_best.csv`: per run and pixel, the one fit drawn: its coefficients,
#   the TOT range and quantiles of its events and their TOT histogram in 25 ps bins (un01).
# * `universality/univ_cells.csv`: every comparison (cell) with its source and target run
#   (un02-un06). `univ_master_cells.csv`: per level cell the inverse-variance means, margins and
#   verdicts (un03, un06). `univ_axis_r2.csv`: the pair rows of the comparison axes with their
#   coverage and lever (un02, un05). `univ_axis_F_wide.csv` and `univ_axis_F_long.csv`: axis F,
#   the linear refit, and its margins (un04).
# * `tables/runs_summary.csv`: the chips, bias, fluence step, RFSel and threshold offset of
#   every IRRAD run (un01-un06).
# * `july/good_runs.csv`: the good-run list; every IRRAD run drawn must be on it as good.
# * The DESY run settings (chip, role, bias, threshold offset, angle, board temperature) come
#   from the campaign's board-config yaml in `TestBeam/board_configs_yaml/`, read by
#   `../campaigns/desy_2026.py`.
#
# The arithmetic lives in `../universality/transfer.py` (the readers, the verdict override, the
# pixel choice and the curves) and `../universality/levels.py` (the rows and bars un02-un06
# draw), the drawing in `../universality/curves.py` and `levels.py`. This notebook holds the
# choices made per figure: the runs, the colours and the words on each figure.
# `../tests/test_universality.py` tests the verdict override and the axis-F rows.
#
# `universality.ipynb` and `universality.py` are the same notebook, kept in step by jupytext.
# Edit either one, then run `python3 -m jupytext --sync universality.ipynb` and
# `python3 -m nbstripout universality.ipynb` in this folder before committing. Neither tool is
# in LCG_104d (the install line is in `../README.md`); running the notebook needs neither.
#
# After a run, `python3 -m etroc_plots.checks <figures>`, run from `TestBeam/` with `<figures>`
# the folder this notebook wrote to (`etroc_plots/notebooks/figures/universality` by default),
# checks every saved figure against the rules in `../CONVENTIONS.md`.

# %%
import os
import sys

import matplotlib
matplotlib.use("Agg")   # draw off-screen; show() displays the PNG exactly as saved
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, display

# The folder that holds the package (TestBeam/) is two folders up. Jupyter starts the kernel in
# the notebook's folder; `python universality.py` has __file__.
here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
sys.path.insert(0, os.path.normpath(os.path.join(here, "..", "..")))

CAMPAIGN = "irrad_2026"                       # campaigns/<CAMPAIGN>.py
# An exported ETROC_CAMPAIGN wins (an empty one counts as unset); it is read once, on the first
# etroc_plots import.
if not os.environ.get("ETROC_CAMPAIGN"):
    os.environ["ETROC_CAMPAIGN"] = CAMPAIGN

from etroc_plots import style                                 # noqa: E402
from etroc_plots import campaigns                             # noqa: E402
from etroc_plots.campaigns import desy_2026                   # noqa: E402
from etroc_plots.etroc_style import BOARD_COLOR               # noqa: E402
from etroc_plots.inputs import check_inputs                   # noqa: E402
from etroc_plots.universality import curves, levels, transfer  # noqa: E402

if campaigns.NAME != CAMPAIGN:
    raise RuntimeError("the package is running campaign %r (ETROC_CAMPAIGN) but CAMPAIGN = %r: "
                       "set one to match the other, then restart the kernel (the campaign is "
                       "read once, on the first import)" % (campaigns.NAME, CAMPAIGN))
campaign = campaigns.active

# stops, listing every missing or unreadable input; reports any table that differs from the
# published copy.
check_inputs(only=("universality/", "tables/runs_summary.csv", "july/good_runs.csv"))

NOTEBOOK = "TestBeam/etroc_plots/notebooks/universality.ipynb"   # "script" in every values file
OUT = os.path.join(os.path.abspath(os.environ.get("ETROC_FIGURES") or "figures"), "universality")
os.makedirs(OUT, exist_ok=True)

RS = pd.read_csv(campaign.RUNS_SUMMARY_CSV)
BANK = transfer.read_bank_best()


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


# %% [markdown]
# ## The runs
#
# * Angle (DESY May 2026): one run per angle of the telescope to the beam, 0 to 60 deg; the
#   source is the 30 deg run. The 60 deg run, 44, was taken on the offset holder, the second
#   mounting of the angled telescope (`OFFSET_HOLDER_RUNS` of `desy_2026`): it is drawn, marked,
#   and never pooled with the aligned runs.
# * Bias (March, H1): the H1 runs at threshold offset 20 from 190 V to 470 V, one per bias. The
#   bias rises with the fluence step, so each curve is labelled with both. There is no natural
#   source on a bias ramp: the source is the run of median bias among those that carry the
#   pixel.
# * Fluence (March, H1): the runs of the fluence ladder, `MARCH_DISPLAY_RUNS` of the campaign
#   module (the runs of the maps notebook); the source is the pre-irradiation run.
# * March to July (H1 and F1): per telescope, a March run and a July run at 1.5e15 after the
#   four months of cooling down; the March run is the source.
#
# Every IRRAD run drawn is checked against the good-run list; DESY runs are not on it.

# %%
ANGLE_RUNS = {0: 26, 10: 24, 20: 20, 30: 9, 45: 13, 60: 44}   # deg -> DESY May run
ANGLE_SOURCE = 30
assert list(ANGLE_RUNS) == sorted(ANGLE_RUNS)
BIAS_RUNS = {"h1": [7, 10, 11, 12, 14, 17, 19]}                # March runs
FLUENCE_RUNS = campaign.MARCH_DISPLAY_RUNS["h1"]
assert FLUENCE_RUNS == [7, 11, 14, 19], FLUENCE_RUNS
MARCH_JULY_PAIRS = {"h1": (19, 3), "f1": (19, 5)}              # telescope -> (March, July) run
HIST_MERGE = 2     # TOT histograms drawn at 2 x 25 ps = 50 ps, about one TOT code: 25 ps bins
                   # show the comb of the TOT codes
HIST_BIN_PS = HIST_MERGE * transfer.HIST_BIN_WIDTH_PS

used = ([("mar", "h1_run%d" % r) for r in BIAS_RUNS["h1"] + FLUENCE_RUNS]
        + [(code, "%s_run%d" % (tel, r)) for tel, pair in MARCH_JULY_PAIRS.items()
           for code, r in zip(("mar", "jul"), pair)])
drops = transfer.good_run_drops(pd.DataFrame(sorted(set(used)), columns=["campaign", "run"]),
                                pairs=(("campaign", "run"),))
print("IRRAD runs drawn: %d, not good: %s" % (len(set(used)), drops or "none"))
if drops:
    raise RuntimeError("runs not good on the good-run list: %s" % drops)

# %% [markdown]
# ## Figure un01: TWC curves along four axes
#
# Per panel one pixel: of the pixels with a fit in the panel's runs, the one fitted in the most
# runs, then the one with the most fitted events summed over them (a tie keeps the order the
# pixels were met in); the fluence panel keeps the bias panel's pixel when every fluence run
# carries it. Per run and pixel the table holds one fit, on the pixel's tracks over the whole
# run, the one with the most events. Each curve is that fit over its own TOT range, minus its
# mean over the source run's 5-95 % TOT range, so the curves compare in shape where the source
# has its events; below, the TOT of the events each fit used, as a fraction per 50 ps bin (two
# of the table's 25 ps bins summed, about one TOT code; 25 ps bins show the comb of the codes).

# %%
UN01_INPUTS = [campaign.TWC_BANK_BEST_CSV, campaign.RUNS_SUMMARY_CSV, campaign.GOOD_RUNS_CSV,
               desy_2026.MAY_YAML]


def best(code, run):
    return transfer.best_rows(BANK, code, run)


def conditions(code, run, board):
    return transfer.board_conditions(RS, code, run, board)


def run_num(run):
    return int(run.rsplit("run", 1)[-1])


def make_curve(rows, pix, src, label, color, is_source, ls="-", **extra):
    c = transfer.curve(rows.loc[pix], float(src.q05), float(src.q95), merge=HIST_MERGE)
    c.update(label=label, color=color, ls=ls, is_source=is_source, **extra)
    return c


def settings_lines(conds):
    """The operating point in at most two lines: the RFSel, then the threshold and the preamp
    power (a clause the campaign did not record is left out)."""
    lines = transfer.settings_text(conds, sep="\n").split("\n")
    rfsel = [l for l in lines if l.startswith("RFSel")]
    others = [l for l in lines if not l.startswith("RFSel")]
    return rfsel + ([", ".join(others)] if others else [])


def values_text(values, unit, fmt="%g"):
    """'220 V' when every value is equal, else the values joined by '/'."""
    vals = sorted(set(values))
    return "/".join(fmt % v for v in vals) + " " + unit


def ranking_record(rank, n=5):
    return [dict(runs_with_fit=int(c), n_fit_sum=int(s), board_idx=int(p[0]), row=int(p[1]),
                 col=int(p[2])) for c, s, p in rank[:n]] if rank else None


# ---- angle: DESY May 2026, source 30 deg
angle_best = {d: best("desy", "run%d" % r) for d, r in ANGLE_RUNS.items()}
angle_rank = transfer.pick_pixel(angle_best, require_keys=[ANGLE_SOURCE],
                                 prefer_keys=list(ANGLE_RUNS))
a_pix = angle_rank[0][2]
a_src = angle_best[ANGLE_SOURCE].loc[a_pix]
a_curves, a_cond = [], {}
for d, r in ANGLE_RUNS.items():
    if a_pix not in angle_best[d].index:
        continue
    a_cond[r] = conditions("desy", "run%d" % r, a_pix[0])
    holder = r in desy_2026.OFFSET_HOLDER_RUNS
    a_curves.append(make_curve(
        angle_best[d], a_pix, a_src,
        "%d deg, run %d%s" % (d, r, " (offset holder)" if holder else ""),
        curves.ramp_color(d, min(ANGLE_RUNS), max(ANGLE_RUNS), "viridis"), d == ANGLE_SOURCE,
        ls="--" if holder else "-", campaign="desy", run="run%d" % r, angle=d,
        offset_holder=holder))
a_missing = [d for d in ANGLE_RUNS if a_pix not in angle_best[d].index]
a_chip = {c["chip"] for c in a_cond.values()}
assert len(a_chip) == 1, a_chip
a_chip = a_chip.pop()
a_hv = {r: c["hv"] for r, c in a_cond.items()}
a_hv_main = max(set(a_hv.values()), key=list(a_hv.values()).count)
a_hv_text = "%g V" % a_hv_main + "".join(", run %d at %g V" % (r, v) for r, v in a_hv.items()
                                         if v != a_hv_main)
angle_panel = dict(
    subject="Angle: %s, pixel (%d, %d)" % (a_chip, a_pix[1], a_pix[2]),
    caption="\n".join(
        ["%s, %s, runs %s" % (desy_2026.CAMPAIGN_TEXT["may"], desy_2026.BEAM_TEXT,
                              "/".join(str(r) for r in a_cond)),
         "%s at %s, board sensor at %s" % (a_chip, a_hv_text,
                                           values_text([c["temp"] for c in a_cond.values()],
                                                       "C")),
         ] + settings_lines(list(a_cond.values()))),
    curves=a_curves, bands=[(float(a_src.q05), float(a_src.q95), style.GRID)])

# ---- bias: March, H1, source = the run of median bias
b_runs = BIAS_RUNS["h1"]
bias_best = {r: best("mar", "h1_run%d" % r) for r in b_runs}
bias_rank = transfer.pick_pixel(bias_best, prefer_keys=b_runs)
b_pix = bias_rank[0][2]
b_cond = {r: conditions("mar", "h1_run%d" % r, b_pix[0]) for r in b_runs
          if b_pix in bias_best[r].index}
b_order = sorted(b_cond, key=lambda r: b_cond[r]["hv"])
b_src_run = b_order[len(b_order) // 2]
b_src = bias_best[b_src_run].loc[b_pix]
hv_lo, hv_hi = min(c["hv"] for c in b_cond.values()), max(c["hv"] for c in b_cond.values())
b_curves = [make_curve(bias_best[r], b_pix, b_src,
                       "%g V, %s" % (b_cond[r]["hv"], style.fluence_label(b_cond[r]["fluence"])),
                       curves.ramp_color(b_cond[r]["hv"], hv_lo, hv_hi, "plasma"),
                       r == b_src_run, campaign="mar", run="h1_run%d" % r)
            for r in b_order]
b_chip = b_cond[b_src_run]["chip"]
bias_panel = dict(
    subject="Bias: %s, pixel (%d, %d)" % (b_chip, b_pix[1], b_pix[2]),
    caption="\n".join(
        ["CERN IRRAD March 2026, H1 at %s, runs %s" % (campaign.ANGLE_TEXT,
                                                      "/".join(str(r) for r in b_order)),
         "%s; source run %d, the median bias (%g V)" % (b_chip, b_src_run,
                                                        b_cond[b_src_run]["hv"]),
         ] + settings_lines(list(b_cond.values()))),
    curves=b_curves, bands=[(float(b_src.q05), float(b_src.q95), style.GRID)])

# ---- fluence: March, H1, the bias panel's pixel when every fluence run carries it
flu_best = {r: best("mar", "h1_run%d" % r) for r in FLUENCE_RUNS}
f_reused = all(b_pix in flu_best[r].index for r in FLUENCE_RUNS)
f_rank = None if f_reused else transfer.pick_pixel(flu_best, prefer_keys=FLUENCE_RUNS)
f_pix = b_pix if f_reused else f_rank[0][2]
f_cond = {r: conditions("mar", "h1_run%d" % r, f_pix[0]) for r in FLUENCE_RUNS}
f_src_run = FLUENCE_RUNS[0]
assert style.fluence_key(f_cond[f_src_run]["fluence"]) == 0.0, f_cond[f_src_run]
f_src = flu_best[f_src_run].loc[f_pix]
f_curves = [make_curve(flu_best[r], f_pix, f_src,
                       "%s, %g V" % (style.fluence_label(f_cond[r]["fluence"]), f_cond[r]["hv"]),
                       style.fluence_color(f_cond[r]["fluence"]), r == f_src_run,
                       campaign="mar", run="h1_run%d" % r)
            for r in FLUENCE_RUNS]
f_chip = f_cond[f_src_run]["chip"]
fluence_panel = dict(
    subject="Fluence: %s, pixel (%d, %d)" % (f_chip, f_pix[1], f_pix[2]),
    caption="\n".join(
        ["CERN IRRAD March 2026, H1 at %s, runs %s" % (campaign.ANGLE_TEXT,
                                                      "/".join(str(r) for r in FLUENCE_RUNS)),
         "%s; source run %d, pre-irradiation" % (f_chip, f_src_run),
         ] + settings_lines(list(f_cond.values()))),
    curves=f_curves, bands=[(float(f_src.q05), float(f_src.q95), style.GRID)])

# ---- March to July at 1.5e15: per telescope its own pixel, the March run as source
mj_curves, mj_bands, mj_band_labels, mj_lines, mj_groups, mj_cond = [], [], [], [], [], []
for k, (tel, (r_mar, r_jul)) in enumerate(MARCH_JULY_PAIRS.items()):
    s_run, t_run = "%s_run%d" % (tel, r_mar), "%s_run%d" % (tel, r_jul)
    bs, bt = best("mar", s_run), best("jul", t_run)
    rank = transfer.pick_pixel({s_run: bs, t_run: bt}, require_keys=[s_run, t_run])
    pix = rank[0][2]
    cs, ct = conditions("mar", s_run, pix[0]), conditions("jul", t_run, pix[0])
    assert style.fluence_key(cs["fluence"]) == style.fluence_key(ct["fluence"]) == 1.5e15
    src = bs.loc[pix]
    color = BOARD_COLOR[k]
    T = tel.upper()
    mj_curves += [make_curve(bs, pix, src, "%s March, run %d" % (T, r_mar), color, True,
                             campaign="mar", run=s_run, telescope=tel, fill=color),
                  make_curve(bt, pix, src, "%s July, run %d" % (T, r_jul), color, False,
                             ls="--", campaign="jul", run=t_run, telescope=tel)]
    mj_bands.append((float(src.q05), float(src.q95), color, 0.18))
    mj_band_labels.append("%s March TOT, 5-95 %%" % T)
    mj_lines.append("%s %s pixel (%d, %d): run %d at %g V, run %d at %g V"
                    % (T, cs["chip"], pix[1], pix[2], r_mar, cs["hv"], r_jul, ct["hv"]))
    mj_cond += [cs, ct]
    mj_groups.append(dict(telescope=tel, chip=cs["chip"], pixel=pix, rank=rank,
                          march_run=s_run, july_run=t_run))
mj_settings = {camp: settings_lines([c for c in mj_cond if c["campaign"] == camp])
               for camp in ("march", "july")}
if mj_settings["march"] == mj_settings["july"]:
    mj_set_lines = ["March and July: " + mj_settings["march"][0]] + mj_settings["march"][1:]
else:
    mj_set_lines = ["%s: %s" % (camp.capitalize(), "; ".join(lines))
                    for camp, lines in mj_settings.items()]
march_july_panel = dict(
    subject="March to July, %s" % style.fluence_label(1.5e15),
    caption="\n".join(["CERN IRRAD March and July 2026, %s at %s"
                       % (" and ".join(g["telescope"].upper() for g in mj_groups),
                          campaign.ANGLE_TEXT)]
                      + mj_lines + mj_set_lines),
    curves=mj_curves, bands=mj_bands, band_labels=mj_band_labels)

PANELS = dict(angle=angle_panel, bias=bias_panel, fluence=fluence_panel,
              march_july=march_july_panel)
for name, p in PANELS.items():
    print("%-10s %s | source TOT 5-95 %%: %s" % (
        name, p["subject"], ", ".join("%.0f-%.0f ps" % (b[0], b[1]) for b in p["bands"])))
    for c in p["curves"]:
        print("    %-34s n_fit %5d  mean removed %8.2f ps%s" % (
            c["label"], c["row"].n_fit, c["mean_removed"], "  source" if c["is_source"] else ""))
print("angle runs without the pixel:", a_missing or "none")
print("fluence panel reuses the bias pixel:", f_reused)

# %%
holders = [c for c in a_curves if c["offset_holder"]]
UN01_FOOTER = (
    u"Per panel one pixel, pixel (row, column): of the pixels with a fit in the panel's runs, "
    u"the one fitted in the most runs, then the most fitted events summed over them. Per run the "
    u"pixel's fit on its tracks over the whole run, the one with the most events.\n"
    u"Each curve is that one fit, delta(TOT) = c2 TOT^2 + c1 TOT + c0, over its own TOT range, "
    u"minus its mean over the source run's 5-95 %% TOT range (shaded; the source curve is "
    u"thicker). Below: the TOT of the events each fit used, as a fraction per %g ps bin; the "
    u"source filled, grey or in its telescope's colour.\n"
    u"Bias: each March run is at its own fluence step. Fluence: %s. March to July: each "
    u"telescope's own pixel, mean removed over its March run's range (shaded in its colour).%s\n"
    u"DESY temperatures: the board sensor values recorded with each run."
    % (HIST_BIN_PS, "the bias panel's pixel" if f_reused else "its own pixel",
       "".join(u" Run %s (%d deg) was taken on the offset holder, the second mounting of the "
               u"angled telescope; it is not part of the aligned runs and is never pooled with "
               u"them." % (c["run"][3:], c["angle"]) for c in holders)))

fig, top_axes, bot_axes, covered = curves.twc_curves(
    list(PANELS.values()), xlabel="TOT [ns]", ylabel_top="Time-walk correction [ps]",
    ylabel_bottom="Fraction / %g ps" % HIST_BIN_PS, band_label="Source TOT, 5-95 %",
    line1="%s, %s; %s, %s" % (desy_2026.CAMPAIGN_TEXT["may"], desy_2026.BEAM_TEXT,
                              campaign.CAMPAIGN, campaign.BEAM_TEXT),
    footer=UN01_FOOTER)


def curve_record(c, cond):
    r = c["row"]
    return dict(label=c["label"], campaign=c["campaign"], run=c["run"],
                is_source=bool(c["is_source"]), n_fit=int(r.n_fit), combo=str(r.combo),
                c2=float(r.c2), c1=float(r.c1), c0=float(r.c0), q05=float(r.q05),
                q95=float(r.q95), tot_min=c["tot_min"], tot_max=c["tot_max"],
                mean_removed_ps=c["mean_removed"], hist_lo=float(r.hist_lo),
                hist_bin_ps=HIST_BIN_PS, hist_counts=[int(v) for v in c["counts"]],
                hist_counts_25ps=[int(v) for v in c["counts_table"]], conditions=cond)


def pixel_record(pix, chip):
    return dict(board_idx=int(pix[0]), row=int(pix[1]), col=int(pix[2]), chip=chip)


un01 = dict(
    angle=dict(subject=angle_panel["subject"], pixel=pixel_record(a_pix, a_chip),
               source=dict(campaign="desy", run="run%d" % ANGLE_RUNS[ANGLE_SOURCE],
                           q05=float(a_src.q05), q95=float(a_src.q95)),
               ranking=ranking_record(angle_rank), runs_without_pixel=a_missing,
               offset_holder_runs=sorted(desy_2026.OFFSET_HOLDER_RUNS),
               curves=[curve_record(c, a_cond[run_num(c["run"])]) for c in a_curves]),
    bias=dict(subject=bias_panel["subject"], pixel=pixel_record(b_pix, b_chip),
              source=dict(campaign="mar", run="h1_run%d" % b_src_run, q05=float(b_src.q05),
                          q95=float(b_src.q95), rule="median bias of the runs with the pixel"),
              ranking=ranking_record(bias_rank),
              curves=[curve_record(c, b_cond[run_num(c["run"])]) for c in b_curves]),
    fluence=dict(subject=fluence_panel["subject"], pixel=pixel_record(f_pix, f_chip),
                 reused_bias_pixel=f_reused,
                 source=dict(campaign="mar", run="h1_run%d" % f_src_run, q05=float(f_src.q05),
                             q95=float(f_src.q95)),
                 ranking=ranking_record(f_rank),
                 curves=[curve_record(c, f_cond[run_num(c["run"])]) for c in f_curves]),
    march_july=dict(subject=march_july_panel["subject"], groups=[
        dict(telescope=g["telescope"], pixel=pixel_record(g["pixel"], g["chip"]),
             source=dict(campaign="mar", run=g["march_run"], q05=b[0], q95=b[1]),
             target=dict(campaign="jul", run=g["july_run"]), ranking=ranking_record(g["rank"]))
        for g, b in zip(mj_groups, mj_bands)],
        curves=[curve_record(c, cnd) for c, cnd in zip(mj_curves, mj_cond)]),
    angles=dict((str(d), r) for d, r in ANGLE_RUNS.items()), bias_runs=BIAS_RUNS,
    fluence_runs=FLUENCE_RUNS, march_july_pairs=MARCH_JULY_PAIRS,
    good_run_drops=sorted("%s %s" % k for k in drops),
    legend_overlap_problems=["%s: the legend covers %d curve points" % (name, n)
                             for name, n in zip(PANELS, covered) if n])
save(fig, "un01_twc_curves", un01,
     dict(curve="one pixel's own fit per curve: delta(TOT) = c2 TOT^2 + c1 TOT + c0, TOT and "
                "delta in ps (TOT drawn in ns), over the fit's TOT range [tot_min, tot_max]",
          fit="per run and pixel the fit with the most events (n_fit) of twc_bank_best.csv: "
              "the pixel's tracks over the whole run",
          mean_removed="each curve minus its mean over a uniform 500-point grid on the source "
                       "run's q05-q95 TOT range (5 and 95 % TOT quantiles of the source fit); "
                       "shaded: that range",
          histograms="the fit's TOT histogram (the table's 25 ps bins from hist_lo, "
                     "hist_counts_25ps) drawn in 50 ps bins from hist_lo, two bins summed "
                     "(hist_counts; a last odd bin gets an empty partner), as the fraction of "
                     "its events per bin",
          pixel="per panel the pixel fitted in the most runs of the panel, then the largest "
                "n_fit summed over them; a tie keeps the order the pixels were met in",
          settings="IRRAD: chip, bias, fluence step, threshold offset and RFSel from "
                   "runs_summary.csv, the March RFSel from MARCH_RFSEL; DESY: from the board-"
                   "config yaml through desy_2026.run_settings, temperature = board sensor value",
          fluence=campaign.FLUENCE_CONVENTION),
     inputs=UN01_INPUTS)

# %% [markdown]
# ## The comparisons and their pair penalties
#
# The study ran its comparisons ("cells") in two sets. The level cells (a to h) ask the
# pre-registered questions: (a) the floor, the same correction on twin runs or on two halves of
# one run; (b) the angle; (b') the angle on the offset holder; (c) one pixel's correction for a
# whole board; (d) one board's for another; (e) a correction pooled over a board's pixels against
# the pixel's own, in August and in a May control; (g) the fluence; (h) March to July at
# 1.5e15. The comparison axes (A to K) follow single knobs: A a pixel's or a board's correction
# for a whole board within one run, B time with the settings held, C an angle step, D a bias
# step at one fluence, E the threshold offset and RFSel, I the sensor vendor, J the cooling down,
# K the IH2 board sensor -20 to -10 C, 45 deg, bias +10 V; axis F re-scores eleven cells with the
# linear term of the correction refitted on the target.
#
# Every number drawn below is a pair penalty: the width of a pair of boards of a target triple
# with the source's correction, minus its width with the target's own correction fitted on the
# other half of the events, on the same events (ps). A row is interpolating when the target's
# TOT lies within the source's, else extrapolating; transfer is claimed on interpolating rows
# only. The margin of each row is max(1 ps, 2 % of the baseline width), from the tables.
#
# **No comparison with an IRRAD run at 2e15 or 3.5e15 is drawn** (`UNIV_EXCLUDED_STEPS` of the
# campaign module). In the study those are the July threshold-offset and RFSel steps (axis E,
# cells E1-E12) and the cooling-down cells (axis J), all at 3.5e15; the cell below counts what
# that leaves out of each figure. Every IRRAD run of a cell drawn is checked against the good-run
# list; the DESY runs are not on that list, so the check does not apply to them.

# %%
UNIV_INPUTS = [campaign.UNIV_CELLS_CSV, campaign.UNIV_MASTER_CSV, campaign.UNIV_AXIS_R2_CSV,
               campaign.UNIV_AXIS_F_WIDE_CSV, campaign.UNIV_AXIS_F_LONG_CSV,
               campaign.TWC_BANK_BEST_CSV, campaign.RUNS_SUMMARY_CSV, campaign.GOOD_RUNS_CSV,
               desy_2026.MAY_YAML, desy_2026.AUG_YAML]
CELLS = transfer.read_cells()
MASTER = transfer.read_master()
AXIS_R2 = transfer.read_axis_r2()
AXIS_F = transfer.read_axis_f()
CELL = CELLS.set_index("cell")

LEFT_OUT = transfer.cells_at_steps(CELLS, RS, campaign.UNIV_EXCLUDED_STEPS)
left_by_group = CELLS[CELLS.cell.isin(LEFT_OUT)].groupby("level_or_axis").cell.apply(
    lambda c: sorted(c, key=transfer.natkey))
STEPS_TEXT = " and ".join(("%g" % s).replace("e+", "e") for s in campaign.UNIV_EXCLUDED_STEPS)
print("cells with an IRRAD run at %s: %d" % (STEPS_TEXT, len(LEFT_OUT)))
for grp, cells_ in left_by_group.items():
    print("    %s: %s" % (grp, ", ".join(cells_)))


def assert_none_left_out(cells_, name):
    hit = sorted(set(cells_) & set(LEFT_OUT), key=transfer.natkey)
    if hit:
        raise RuntimeError("%s would draw cells with an IRRAD run at %s: %s"
                           % (name, campaign.UNIV_EXCLUDED_STEPS, hit))


def good_run_check(cells_, name):
    """The IRRAD source and target runs of `cells_` that are not good on the good-run list."""
    sub = CELLS[CELLS.cell.isin(set(cells_))]
    irrad = sorted({(c, r) for p in (("src_campaign", "src_run"), ("tgt_campaign", "tgt_run"))
                    for c, r in zip(sub[p[0]], sub[p[1]])
                    if transfer.is_irrad(c) and isinstance(r, str)})
    found = transfer.good_run_drops(sub)
    print("%s: %d IRRAD runs checked against the good-run list, %d not good%s"
          % (name, len(irrad), len(found), (": %s" % found) if found else ""))
    if found:
        raise RuntimeError("%s: runs not good on the good-run list: %s" % (name, found))
    return irrad


def cell_runs(cells_):
    """(campaign code, run) of every source and target run of `cells_`."""
    sub = CELLS[CELLS.cell.isin(set(cells_))]
    return sorted({(c, r) for p in (("src_campaign", "src_run"), ("tgt_campaign", "tgt_run"))
                   for c, r in zip(sub[p[0]], sub[p[1]]) if isinstance(r, str)})


# The words on the figures. Campaigns: the study's codes.
MONTH = {"desy": "May", "aug": "Aug", "mar": "March", "jul": "July"}


def facility_line(codes):
    """The campaigns drawn, e.g. 'DESY May and Aug 2026, 4 GeV electrons; CERN IRRAD March
    2026, 24 GeV p+'."""
    codes = set(codes)
    desy = [MONTH[c] for c in ("desy", "aug") if c in codes]
    irrad = [MONTH[c] for c in ("mar", "jul") if c in codes]
    parts = []
    if desy:
        parts.append("DESY %s 2026, %s" % (" and ".join(desy), desy_2026.BEAM_TEXT))
    if irrad:
        parts.append("CERN IRRAD %s 2026, %s" % (" and ".join(irrad), campaign.BEAM_TEXT))
    return "; ".join(parts)


SETTINGS_NAME = {"desy": "DESY May", "aug": "DESY Aug", "mar": "IRRAD March",
                 "jul": "IRRAD July"}


def settings_footer(runs):
    """One line: per campaign, the operating point of every board of the runs drawn
    (transfer.settings_text; DESY May records no RFSel and no preamp power, so the clauses are
    left out there)."""
    by = {}
    for code, run in runs:
        by.setdefault(code, []).extend(transfer.run_conditions(RS, code, run))
    return "Settings, " + "; ".join("%s: %s" % (SETTINGS_NAME[c], transfer.settings_text(by[c]))
                                    for c in ("desy", "aug", "mar", "jul") if c in by) + "."


PENALTY_LABEL = r"Pair penalty $\Delta\sigma_{\mathrm{pair}}$ [ps]"
MARGIN_WORD = "margin, max(1 ps, 2 % of the baseline width)"
VERDICT_WORDS = {"PASS": "equivalent within the margin (TOST)",
                 "FAIL": "not shown equivalent",
                 "NA-UNGUARDED": "no leave-pair-out guard"}
SETTINGS_RULE = ("IRRAD: runs_summary.csv per board, the March RFSel from MARCH_RFSEL; DESY: the "
                 "board-config yaml through desy_2026.run_settings, the August RFSel from "
                 "DEFAULT_RFSEL where the yaml has none, no RFSel recorded in May")

# %% [markdown]
# ## Figure un02: penalty against coverage and lever
#
# Every extrapolating pair row of the comparison axes (the transferred correction, the primary
# error factor), its penalty against how much of the target's TOT the source covers and against
# the lever: the rms change of the correction when the target's TOT is clipped to the source's
# 0.5-99.5 % range, over the target events outside it. The grey band and the dashed line are the
# interpolating rows of the same axes, as a reference. The axis E and J rows of the IRRAD runs at
# 3.5e15 are left out of both; the band is computed without them.
#
# Six extrapolating rows of axis A (cells A4, A5 and A6, two pairs each, one pixel each) are a
# second primary row of a pair whose full row is also drawn. The study's caveat: its per-cell
# summary excludes the source-equals-target (self-transfer) duplicates and the axis dumps do not,
# and the axis tables carry no column that marks them. They are drawn as the table has them and
# listed under rows_flagged in the values file.

# %%
AXIS_WORDS = {"A": "A: one run, pixel or board to a whole board",
              "B": "B: time, 3 h to 6.8 days, same settings",
              "C": "C: angle, 20 and 30 deg",
              "D": "D: bias, at 9e14 and at 1.5e15",
              "E": "E: threshold offset and RFSel together",
              "I": "I: sensor vendor, HPK and FBK, before irradiation",
              "K": "K: IH2 board sensor -20 to -10 C, 45 deg, bias +10 V"}
UN02_MARKERS = {"E": "X"}         # axis E changes the front end, not the sensor: drawn apart

extrap_all, interp_all = levels.coverage_lever_rows(AXIS_R2)
un02_out = {fam: rows[rows.cell.isin(LEFT_OUT)] for fam, rows in (("extrap", extrap_all),
                                                                 ("interp", interp_all))}
extrap = extrap_all[~extrap_all.cell.isin(LEFT_OUT)]
interp = interp_all[~interp_all.cell.isin(LEFT_OUT)]
assert set(pd.concat(un02_out.values()).axis) <= {"E", "J"}, "only axes E and J are left out"
band_all, band = levels.interp_band(interp_all), levels.interp_band(interp)
print("un02 rows: extrap %d of %d, interp %d of %d (left out: %s)" % (
    len(extrap), len(extrap_all), len(interp), len(interp_all),
    {fam: r.groupby("axis").size().to_dict() for fam, r in un02_out.items()}))
print("interp median %.4f ps, 16-84 %%: %.4f to %.4f (all rows: %.4f, %.4f to %.4f)" % (
    band["median"], band["lo"], band["hi"], band_all["median"], band_all["lo"], band_all["hi"]))
un02_cells = sorted(set(extrap.cell) | set(interp.cell), key=transfer.natkey)
PAIR_KEY = ["cell", "kind", "board_or_role", "family"]
FLAG_KEYS = PAIR_KEY + ["n_pix", "mu_ps"]
un02_drawn = pd.concat([extrap, interp])
un02_dup = un02_drawn[un02_drawn.duplicated(PAIR_KEY, keep=False)]
UN02_FLAGGED = un02_dup[un02_dup.n_pix < un02_dup.groupby(PAIR_KEY).n_pix.transform("max")]
assert len(UN02_FLAGGED) == 6 and set(UN02_FLAGGED.cell) == {
    "A4_P4_run26_0deg", "A5_P4_run33_30deg", "A6_P4_run44_60deg"}, UN02_FLAGGED[FLAG_KEYS]
print("un02 rows flagged (second primary row of a pair key):\n%s"
      % UN02_FLAGGED[FLAG_KEYS].to_string())
assert_none_left_out(un02_cells, "un02")
un02_irrad = good_run_check(un02_cells, "un02")
un02_runs = cell_runs(un02_cells)

fig, axes, drawn, covered = levels.un02_coverage_lever(
    extrap, band, axis_words=AXIS_WORDS, axis_markers=UN02_MARKERS,
    xlabels=("1 - coverage of the target TOT", "Lever beyond the source TOT range [ps]"),
    ylabel=PENALTY_LABEL, legend_title="Comparison axis",
    band_label="interpolating rows, 16-84 %", median_label="interpolating median, %.2f ps",
    line1=facility_line(c for c, _ in un02_runs),
    subjects=["Against the TOT coverage", "Against the lever"],
    footer=("Extrapolating pair rows of the comparison axes (target TOT beyond the source's): "
            "%d rows; marker area grows with the pixels pooled in the row. Grey band and dashed "
            "line: the %d interpolating rows, 16-84 %% range and median.\n"
            "Coverage: the share of the target's TOT inside the source's 0.5-99.5 %% TOT range. "
            "Lever: the rms change of the correction when the target's TOT is clipped to that "
            "range, over the target events outside it.\n"
            "No IRRAD run at %s is drawn: %d extrapolating and %d interpolating rows "
            "of axes E and J (July, 3.5e15) are left out.\n%s"
            % (len(extrap), band["n_values"], STEPS_TEXT.replace(" and ", " or "),
               len(un02_out["extrap"]), len(un02_out["interp"]),
               settings_footer(un02_runs))))
ROW_KEYS = ["axis", "cell", "board_or_role", "family", "n_pix", "cov_C", "lever_X", "mu_ps",
            "se_ps"]
save(fig, "un02_coverage_lever",
     dict(n_extrap_rows=int(len(extrap)), n_interp_rows=int(len(interp)),
          axes=sorted(extrap.axis.unique()), interp_median_ps=band["median"],
          interp_16_84_ps=[band["lo"], band["hi"]], interp_band=band,
          rows=extrap[ROW_KEYS].to_dict("records"),
          rows_flagged=UN02_FLAGGED[FLAG_KEYS].to_dict("records"),
          rows_flagged_rule=("drawn: a second primary row on the same (cell, kind, board_or_role, "
                             "family) with fewer pixels; the study's per-cell summary excludes "
                             "self-transfer duplicates, the axis dumps do not"),
          left_out=dict(rule="cells with an IRRAD source or target run at %s" % STEPS_TEXT,
                        extrap_rows=int(len(un02_out["extrap"])),
                        interp_rows=int(len(un02_out["interp"])),
                        cells=sorted(set(pd.concat(un02_out.values()).cell), key=transfer.natkey),
                        interp_band_with_them=band_all),
          irrad_runs_checked=["%s %s" % k for k in un02_irrad], good_run_drops=[],
          drawn=drawn,
          legend_overlap_problems=(["the legend covers %d points" % covered] if covered else [])),
     dict(levels.CONVENTIONS, figure="x: 1 - cov_C (left), lever_X in ps (right); y: mu_ps of "
                                     "the extrap rows; band: median and 16-84 % percentiles of "
                                     "the interp rows' mu_ps (rows without mu left out)",
          settings=SETTINGS_RULE),
     inputs=UNIV_INPUTS)

# %% [markdown]
# ## Figure un03: the penalty of every level cell
#
# One row per level cell, grouped by level. Left, the interpolating family, the claim the study
# registered before it ran: the inverse-variance mean of the cell's interpolating pair rows,
# its 90 % interval, the cell's verdict, and at the right edge the share of the cell's
# differences that are interpolating. Right, all target triples, both families pooled
# (descriptive). The verdicts are the study's: a cell without a leave-pair-out guard that is not
# a FAIL reads "no leave-pair-out guard". The level cells hold no IRRAD run at 2e15 or 3.5e15
# (asserted).

# %%
LABEL = {
    "a1_twin_26to27": "twin 26 -> 27, 0 deg", "a2_twin_27to26": "twin 27 -> 26, 0 deg",
    "a3_floor_parity_run26": "in-run parity floor, run 26",
    "a5_floor_block_run26": "in-run block floor, run 26",
    "a6_twin_24to25": "twin 24 -> 25, 10 deg", "a7_twin_20to21": "twin 20 -> 21, 20 deg",
    "b1_30to0": "30 -> 0 deg", "b2_30to10": "30 -> 10 deg", "b3_30to20": "30 -> 20 deg",
    "b4_30to45": "30 -> 45 deg", "b5_ctrl_30to30": "30 -> 30 deg (control)",
    "bp1_off30to60_r44": "offset holder 30 -> 60 deg (run 44)",
    "bp2_off30to60_r45": "offset holder 30 -> 60 deg (run 45)",
    "bp3_ctrl_off60to60": "offset holder 60 -> 60 deg (control)",
    "bp4_twin_off60": "offset holder twin 44 -> 45, 60 deg",
    "c1_pix2board_run26_0deg": "pixel -> board, 0 deg",
    "c2_pix2board_run33_30deg": "pixel -> board, 30 deg",
    "c3_pix2board_run44_60deg": "pixel -> board, 60 deg",
    "d1_board2board_run26": "board -> board, 0 deg",
    "d2_board2board_run33": "board -> board, 30 deg",
    "e1_pooled_aug_run23": "pooled, Aug run 23", "e2_pooled_aug_run18": "pooled, Aug run 18",
    "e3_pooled_aug_run8": "pooled, Aug run 8", "e4_pooled_aug_run5": "pooled, Aug run 5",
    "e5_pooled_may_run26": "pooled, May run 26 (control)",
    "g1_h1_7to10": "H1 0 -> 3e14 (210 V)", "g2_h1_7to11": "H1 0 -> 3e14 (280 V, bias matched)",
    "g4_h1_7to19": "H1 0 -> 1.5e15 (demonstrator)", "g5_h1_11to14": "H1 3e14 -> 9e14",
    "g6_h1_14to19": "H1 9e14 -> 1.5e15", "g7_f1_2to10": "F1 0 -> 3e14 (190 V)",
    "g8_f1_2to11": "F1 0 -> 3e14 (260 V, bias matched)", "g9_f1_2to19": "F1 0 -> 1.5e15",
    "g10_f1_11to14": "F1 3e14 -> 9e14", "g11_f1_14to19": "F1 9e14 -> 1.5e15",
    "h1_mar19_to_jul3": "H1 Mar 470 V -> Jul 540 V (run 3, few events per pixel)",
    "h2_mar19_to_jul4": "H1 Mar 470 V -> Jul 540 V (run 4)",
    "h3_mar19_to_jul5": "H1 Mar 470 V -> Jul 540 V (run 5)",
    "h4_mar17_to_jul10": "H1 Mar 400 V -> Jul 550 V (run 10)",
    "h5_f1_19to5": "F1 Mar 350 V -> Jul 460 V (run 5)",
    "h6_f1_19to6": "F1 Mar 350 V -> Jul 460 V (run 6)",
    "h7_f1_19to7": "F1 Mar 350 V -> Jul 460 V (run 7)",
    "h8_f1_17to5": "F1 Mar 265 V -> Jul 460 V (run 5)",
}
LEVTITLE = {"a": "(a) floor", "b": "(b) angle", "bp": "(b') angle, offset holder",
            "c": "(c) pixel to board", "d": "(d) board to board",
            "e": "(e) pooled board correction", "g": "(g) fluence", "h": "(h) March to July"}

assert set(MASTER.cell) <= set(LABEL), sorted(set(MASTER.cell) - set(LABEL))
assert_none_left_out(MASTER.cell, "un03")
un03_irrad = good_run_check(MASTER.cell, "un03")
un03_runs = cell_runs(MASTER.cell)
fig, axes, drawn, covered = levels.un03_penalty_per_cell(
    MASTER, labels=LABEL, level_titles=LEVTITLE,
    subjects=["Interpolating family, the registered claim", "All target triples, descriptive"],
    xlabel=PENALTY_LABEL, verdict_words=VERDICT_WORDS, margin_word=MARGIN_WORD,
    no_rows_text="no interpolating triple", share_fmt="%.0f %% interp.",
    line1=facility_line(c for c, _ in un03_runs),
    footer=("Per cell the inverse-variance mean of its primary pair rows with its 90 %% interval "
            "(mu +- 1.645 se), marker and colour the cell's verdict; grey bar: the cell's own "
            "margin. Left: the interpolating rows (target TOT within the source's), with the "
            "share of the cell's differences in that family; right: every target triple, both "
            "families pooled. %d cells, of %d DESY and %d IRRAD runs.\n%s"
            % (len(MASTER), sum(1 for c, _ in un03_runs if not transfer.is_irrad(c)),
               len(un03_irrad), settings_footer(un03_runs))))
M_KEYS = ["cell", "level", "mu_ivw", "se_ivw", "margin", "verdict_pair_interp", "all_ivw_mu",
          "all_ivw_se", "all_margin", "all_verdict", "n_interp", "n_extrap", "interp_pct",
          "guard_verdict", "final"]
save(fig, "un03_penalty_per_cell",
     dict(n_cells=int(len(MASTER)), levels=list(MASTER.level.drop_duplicates()),
          rows=[dict(r, label=LABEL[r["cell"]]) for r in MASTER[M_KEYS].to_dict("records")],
          irrad_runs_checked=["%s %s" % k for k in un03_irrad], good_run_drops=[],
          drawn=drawn,
          legend_overlap_problems=(["the legend covers %d points" % covered] if covered else [])),
     dict(levels.CONVENTIONS, figure="per cell: left mu_ivw +- Z90 se_ivw (interp family), margin, "
                                     "verdict_pair_interp, interp_pct = 100 n_interp / (n_interp "
                                     "+ n_extrap); right all_ivw_mu +- Z90 all_ivw_se, "
                                     "all_margin, all_verdict; a cell without interp rows has "
                                     "no mu_ivw and no margin (none drawn)",
          settings=SETTINGS_RULE),
     inputs=UNIV_INPUTS)

# %% [markdown]
# ## Figure un04: the linear refit
#
# Axis F re-scores eleven cells with a second transferred correction: the source's quadratic
# term kept and the linear term refitted on the target (fold-matched, one more parameter fitted
# on the target than the correction as is). Per pair row the penalty before (grey) and after
# (red); filled markers are interpolating rows, open ones extrapolating. A pair is named by the
# chips of its two boards in the target run (the study's combination of that run, as the
# coefficient table records it). Four extrapolating board-to-board rows (one pair at 0 deg, all
# three at 30 deg) have five pixels and a penalty of exactly 0.0 before the refit. The study's
# caveat: its per-cell summary excludes the source-equals-target (self-transfer) duplicates and
# the axis dumps do not, and these rows have that pattern. They are drawn as the table has them
# and listed under rows_flagged in the values file.

# %%
F_ROWS = levels.refit_rows(AXIS_F)
assert len(F_ROWS) == 49 and (F_ROWS.family == "interp").sum() == 16, F_ROWS.family.value_counts()
f_cells = sorted(set(F_ROWS.cell), key=transfer.natkey)
assert_none_left_out(f_cells, "un04")
un04_irrad = good_run_check(f_cells, "un04")
un04_runs = cell_runs(f_cells)
F_ROWS["chips"] = [transfer.pair_chips(RS, BANK, CELL.loc[c, "tgt_campaign"],
                                       CELL.loc[c, "tgt_run"], b)
                   for c, b in zip(F_ROWS.cell, F_ROWS.board_or_role)]
F_ROWS["label"] = ["%s: %s" % (LABEL[b], ch) for b, ch in zip(F_ROWS.base_cell, F_ROWS.chips)]
zero_rows = F_ROWS[F_ROWS.xfer_mu_ps == 0.0][PAIR_KEY + ["xfer_n_pix", "xfer_mu_ps"]].rename(
    columns={"xfer_n_pix": "n_pix", "xfer_mu_ps": "mu_ps"})
assert len(zero_rows) == 4, zero_rows
print("un04 rows flagged (penalty exactly 0.0 before the refit):\n%s" % zero_rows.to_string())
fig, axes, drawn, covered = levels.un04_linear_refit(
    F_ROWS, row_labels=list(F_ROWS.label),
    family_titles={"interp": "Target TOT within the source's",
                   "extrap": "Target TOT beyond the source's"},
    xlabel=PENALTY_LABEL,
    legend_words=dict(xfer="source correction as is", xfer_lin="linear term refitted on target",
                      open="open: target TOT beyond the source's", margin="margin, max(1 ps, 2 %)"),
    line1=facility_line(c for c, _ in un04_runs), subject="Before and after a linear TOT refit",
    footer=("Per pair row of the eleven re-scored cells: the penalty with the source's correction "
            "as is and with its quadratic term kept and its linear term refitted on the target "
            "(fold-matched), 90 %% intervals; grey bar: the row's margin. %d rows: %d "
            "interpolating, %d extrapolating. A pair is named by the chips of its two boards in "
            "the target run.\n%s"
            % (len(F_ROWS), (F_ROWS.family == "interp").sum(), (F_ROWS.family == "extrap").sum(),
               settings_footer(un04_runs))))
F_KEYS = ["fnum", "cell", "base_cell", "kind", "board_or_role", "chips", "family", "xfer_n_pix",
          "xfer_mu_ps", "xfer_se_ps", "xfer_tost", "xfer_lin_n_pix", "xfer_lin_mu_ps",
          "xfer_lin_se_ps", "xfer_lin_tost", "xfer_lin_lpo_tost", "delta_xferlin_minus_xfer_mu_ps",
          "margin_ps"]
save(fig, "un04_linear_refit",
     dict(n_rows=int(len(F_ROWS)), n_interp=int((F_ROWS.family == "interp").sum()),
          n_extrap=int((F_ROWS.family == "extrap").sum()),
          base_cells={f: b for f, b in zip(F_ROWS.fnum, F_ROWS.base_cell)},
          rows=[dict(r, label=l) for r, l in zip(F_ROWS[F_KEYS].to_dict("records"), F_ROWS.label)],
          rows_flagged=zero_rows.to_dict("records"),
          rows_flagged_rule=("drawn: xfer_mu_ps exactly 0.0; the study's per-cell summary excludes "
                             "self-transfer duplicates, the axis dumps do not"),
          irrad_runs_checked=["%s %s" % k for k in un04_irrad], good_run_drops=[], drawn=drawn,
          legend_overlap_problems=(["the legend covers %d points" % covered] if covered else [])),
     dict(levels.CONVENTIONS, figure="per axis-F pair row with both arms: xfer_mu_ps +- Z90 "
                                     "xfer_se_ps and xfer_lin_mu_ps +- Z90 xfer_lin_se_ps; "
                                     "margin_ps joined from the long table's xfer row on cell, "
                                     "kind, board_or_role, family and n_pix",
          pairs="chips of the two boards of board_or_role in the target run: the role -> board "
                "map of the coefficient table's role column (transfer.role_boards), the chip "
                "from runs_summary.csv (IRRAD) or the board-config yaml (DESY)",
          settings=SETTINGS_RULE),
     inputs=UNIV_INPUTS)

# %% [markdown]
# ## Figure un05: settings steps
#
# Axis E steps the front end: the threshold offset and the RFSel. The IRRAD settings steps of the
# study exist only at 3.5e15 (July, H1 and F1, cells E1-E12), so none is drawn; what remains are
# the three August cells, two steps that change the offset and the RFSel together and a control
# at the same settings. Per cell and family a bar: the pixel-weighted mean penalty of its pair
# rows (weights n_pix) with its 90 % interval.

# %%
E_ALL = transfer.primary_rows(AXIS_R2)
E_ALL = E_ALL[(E_ALL.axis == "E") & E_ALL.family.isin(["interp", "extrap"])]
E_CELLS = sorted(set(E_ALL.cell) - set(LEFT_OUT), key=transfer.natkey)
E_OUT = sorted(set(E_ALL.cell) & set(LEFT_OUT), key=transfer.natkey)
assert [c.split("_")[0] for c in E_CELLS] == ["E13", "E14", "E15"], E_CELLS
un05_irrad = good_run_check(E_CELLS, "un05")
un05_runs = cell_runs(E_CELLS)


def step_words(cell):
    """'runs 19 -> 21' and the settings of the triple's boards that change, from the yaml."""
    c = CELL.loc[cell]
    roles = transfer.role_boards(BANK, c.tgt_campaign, c.tgt_run)
    boards = sorted(roles[r] for r in ("dut", "ref", "trig"))
    src = [transfer.board_conditions(RS, c.src_campaign, c.src_run, b) for b in boards]
    tgt = [transfer.board_conditions(RS, c.tgt_campaign, c.tgt_run, b) for b in boards]
    lines = ["%s runs %s -> %s" % (MONTH[c.tgt_campaign], c.src_run[3:], c.tgt_run[3:])]
    for key, word in (("offset", "threshold offset"), ("rfsel", "RFSel")):
        a = "/".join("%g" % v for v in sorted({x[key] for x in src}))
        b = "/".join("%g" % v for v in sorted({x[key] for x in tgt}))
        if a != b:
            lines.append("%s %s -> %s" % (word, a, b))
    if len(lines) == 1:
        lines.append("settings held")
    return "\n".join(lines), len(lines) > 2


E_LABEL, E_COLOR = {}, {}
for cell in E_CELLS:
    E_LABEL[cell], both = step_words(cell)
    E_COLOR[cell] = levels.AXIS_COLOR["E"] if both else style.INK_MUTED
E_BARS = levels.family_bars(E_ALL, E_CELLS)
for b in E_BARS:
    print("    %-30s %-6s n_rows %d n_pix %3d  mu %+.3f  se %.3f ps" % (
        b["cell"], b["family"], b["n_rows"], b["n_pix"], b["mu_ps"], b["se_ps"]))
fig, axes, drawn, covered = levels.un05_settings_steps(
    E_BARS, cell_labels=E_LABEL, cell_colors=E_COLOR,
    legend=[(levels.AXIS_COLOR["E"], "offset and RFSel step together"),
            (style.INK_MUTED, "same settings (floor)")],
    family_words=dict(interp="solid: target TOT within the source's",
                      extrap="hatched: target TOT beyond it"),
    ylabel="Pair penalty, pixel-weighted [ps]", line1=facility_line(c for c, _ in un05_runs),
    subject="Settings steps; in IRRAD only at 3.5e15",
    footer=("Per cell and family the pixel-weighted mean of the pair penalties (weights: the "
            "pixels pooled per row) and its 90 %% interval, se the root of the pixel-weighted "
            "mean se^2.\nThe IRRAD threshold-offset and RFSel steps of the study exist only at "
            "3.5e15 "
            "(July, H1 and F1, %d cells), a fluence these figures do not draw.\n%s"
            % (len(E_OUT), settings_footer(un05_runs))))
save(fig, "un05_settings_steps",
     dict(n_bars=len(E_BARS), cells=E_CELLS, bars=E_BARS,
          labels={c: E_LABEL[c].replace("\n", "; ") for c in E_CELLS},
          rows=E_ALL[E_ALL.cell.isin(E_CELLS)][
              ["cell", "board_or_role", "family", "n_pix", "mu_ps", "se_ps", "margin_ps", "tost",
               "guard_verdict"]].to_dict("records"),
          left_out=dict(cells=E_OUT, rows=int(E_ALL.cell.isin(E_OUT).sum()),
                        bars=len(levels.family_bars(E_ALL, E_OUT))),
          irrad_runs_checked=["%s %s" % k for k in un05_irrad], good_run_drops=[], drawn=drawn,
          legend_overlap_problems=(["the legend covers %d points" % covered] if covered else [])),
     dict(levels.CONVENTIONS, figure="bars: per cell and family mu = sum(n_pix mu_ps) / "
                                     "sum(n_pix), se = sqrt(sum(n_pix se_ps^2) / sum(n_pix)), "
                                     "error bar Z90 se",
          settings=SETTINGS_RULE),
     inputs=UNIV_INPUTS)

# %% [markdown]
# ## Figure un06: three controls
#
# The same numbers as the right panel of un03 (all target triples, both families), for three
# questions: the floor on one day (twin runs, and the parity and block halves of run 26); a
# correction pooled over a board's pixels against the pixel's own, in the August runs and in the
# May control (run 26); and March to July at 1.5e15 on the same chips, against the bias step
# between the two runs (the July bias minus the March bias, the median over the boards). Points
# at one bias step are spread by 5 V.

# %%
TEL_COLOR = {"h1": BOARD_COLOR[0], "f1": BOARD_COLOR[1]}       # as in un01's March-to-July panel
TEL_MARKER = {"h1": "o", "f1": "s"}
E_CONTROL_COLOR = "#1a9641"
A_WRAP = {c: LABEL[c].replace(", ", "\n").replace("twin ", "twin\n").replace("in-run ", "in-run\n")
          for c in MASTER[MASTER.level == "a"].cell}
E_WRAP = {c: LABEL[c].replace("pooled, ", "").replace(" run", "\nrun").replace(" (", "\n(")
          for c in MASTER[MASTER.level == "e"].cell}
mrow = MASTER.set_index("cell")
h_cells = list(MASTER[MASTER.level == "h"].cell)
h_tel = {c: CELL.loc[c, "tgt_run"].split("_run")[0] for c in h_cells}
h_dbias = {c: float(CELL.loc[c, "dbias_v"]) for c in h_cells}
groups = {}
for c in h_cells:
    groups.setdefault(h_dbias[c], []).append(c)
h_shift = levels.spread_offsets(groups.values(), 5.0)
un06_cells = list(MASTER[MASTER.level.isin(["a", "e", "h"])].cell)
assert_none_left_out(un06_cells, "un06")
un06_irrad = good_run_check(un06_cells, "un06")
un06_runs = cell_runs(un06_cells)


def ctrl_row(cell, color, marker, **kw):
    r = mrow.loc[cell]
    return dict(cell=cell, mu=float(r.all_ivw_mu), se=float(r.all_ivw_se),
                margin=float(r.all_margin), color=color, marker=marker, **kw)


panels = [
    dict(subject="Same day (May)", width=6.4, rotation=0,
         rows=[ctrl_row(c, style.INK_MUTED, "o", label=A_WRAP[c])
               for c in MASTER[MASTER.level == "a"].cell]),
    dict(subject="Pooled board correction", width=4.4, rotation=0,
         rows=[ctrl_row(c, E_CONTROL_COLOR if CELL.loc[c, "tgt_campaign"] == "desy"
                        else style.INK_MUTED, "o", label=E_WRAP[c])
               for c in MASTER[MASTER.level == "e"].cell],
         legend=[(style.INK_MUTED, "o", "August runs"), (E_CONTROL_COLOR, "o", "May control")]),
    dict(subject="March to July, %s" % style.fluence_label(1.5e15), width=5.6, box_half=2.4,
         xlabel="Bias step, March to July [V]",
         rows=[ctrl_row(c, TEL_COLOR[h_tel[c]], TEL_MARKER[h_tel[c]],
                        x=h_dbias[c] + h_shift[c]) for c in h_cells],
         legend=[(TEL_COLOR[t], TEL_MARKER[t], "%s telescope" % t.upper()) for t in ("h1", "f1")]),
]
fig, axes, drawn, covered = levels.un06_controls(
    panels, ylabel=PENALTY_LABEL, line1=facility_line(c for c, _ in un06_runs),
    footer=("Per cell the inverse-variance mean over all target triples (both families) with its "
            "90 %% interval; grey box: the cell's own margin. Pooled: one correction per board, "
            "pooled over its pixels, against each pixel's own, in one run.\nMarch to July: March "
            "source, July target, same chips at 1.5e15; x = the July bias minus the March bias "
            "(median over the boards), points at one bias step spread by 5 V.\n%s"
            % settings_footer(un06_runs)))
C_KEYS = ["cell", "level", "all_ivw_mu", "all_ivw_se", "all_margin", "all_verdict"]
save(fig, "un06_controls",
     dict(same_day=[dict(r, label=LABEL[r["cell"]]) for r in
                    MASTER[MASTER.level == "a"][C_KEYS].to_dict("records")],
          pooled=[dict(r, label=LABEL[r["cell"]]) for r in
                  MASTER[MASTER.level == "e"][C_KEYS].to_dict("records")],
          march_july=[dict(r, dbias_v=h_dbias[r["cell"]], x=h_dbias[r["cell"]] + h_shift[r["cell"]],
                           telescope=h_tel[r["cell"]]) for r in
                      MASTER[MASTER.level == "h"][C_KEYS].to_dict("records")],
          irrad_runs_checked=["%s %s" % k for k in un06_irrad], good_run_drops=[], drawn=drawn,
          legend_overlap_problems=(["the legend covers %d points" % covered] if covered else [])),
     dict(levels.CONVENTIONS, figure="per cell all_ivw_mu +- Z90 all_ivw_se, all_margin as a "
                                     "box; March to July x = dbias_v of univ_cells (July target "
                                     "bias minus March source bias, medians over the boards) "
                                     "plus a 5 V spread within one dbias_v",
          settings=SETTINGS_RULE),
     inputs=UNIV_INPUTS)
