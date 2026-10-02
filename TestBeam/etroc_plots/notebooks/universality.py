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
# TOT, delta(TOT) = c2 TOT^2 + c1 TOT + c0, fitted per run on the pixel's tracks. Before any
# resolution number, this notebook shows the curves themselves: one pixel's correction in runs
# that differ along one axis, and how far each target run's TOT reaches beyond the source's.
#
# * Figure un01: the correction of one pixel against TOT along four axes, with the TOT of the
#   fitted events below: the angle to the beam (DESY May 2026, 0 to 60 deg), the sensor bias
#   (CERN IRRAD March 2026, H1), the fluence step (March, H1, the runs of the fluence ladder)
#   and the four months between March and July at 1.5e15 (H1 and F1).
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
# * `universality/univ_cells.csv`, `univ_master_cells.csv`, `univ_axis_r2.csv`,
#   `univ_axis_F_wide.csv` and `univ_axis_F_long.csv`: the study's comparisons and their
#   resolution penalties. No figure of this notebook draws them yet; the first code cell checks
#   them with the other tables.
# * `tables/runs_summary.csv`: the chips, bias, fluence step, RFSel and threshold offset of
#   every IRRAD run (un01).
# * `july/good_runs.csv`: the good-run list; every IRRAD run drawn must be on it as good.
# * The DESY run settings (chip, role, bias, threshold offset, angle, board temperature) come
#   from the campaign's board-config yaml in `TestBeam/board_configs_yaml/`, read by
#   `../campaigns/desy_2026.py`.
#
# The arithmetic lives in `../universality/transfer.py` (the readers, the pixel choice and the
# curves) and the drawing in `../universality/curves.py`. This notebook holds the choices made
# per figure: the runs, the colours and the words on each figure.
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
from etroc_plots.universality import curves, transfer         # noqa: E402

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
# has its events; below, the TOT of the events each fit used, 25 ps bins, as a fraction per bin.

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
    c = transfer.curve(rows.loc[pix], float(src.q05), float(src.q95))
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
        ["CERN IRRAD March 2026, H1, runs %s" % "/".join(str(r) for r in b_order),
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
        ["CERN IRRAD March 2026, H1, runs %s" % "/".join(str(r) for r in FLUENCE_RUNS),
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
                             campaign="mar", run=s_run, telescope=tel),
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
    caption="\n".join(["CERN IRRAD March and July 2026, %s"
                       % " and ".join(g["telescope"].upper() for g in mj_groups)]
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
    u"thicker). Below: the TOT of the events each fit used, 25 ps bins, as a fraction per bin; "
    u"the source filled grey.\n"
    u"Bias: each March run is at its own fluence step. Fluence: %s. March to July: each "
    u"telescope's own pixel, mean removed over its March run's range (shaded in its colour).%s\n"
    u"DESY temperatures: the board sensor values recorded with each run."
    % ("the bias panel's pixel" if f_reused else "its own pixel",
       "".join(u" Run %s (%d deg) was taken on the offset holder, the second mounting of the "
               u"angled telescope; it is not part of the aligned runs and is never pooled with "
               u"them." % (c["run"][3:], c["angle"]) for c in holders)))

fig, top_axes, bot_axes, covered = curves.twc_curves(
    list(PANELS.values()), xlabel="TOT [ns]", ylabel_top="Time-walk correction [ps]",
    ylabel_bottom="Fraction / 25 ps", band_label="Source TOT, 5-95 %",
    line1="%s, %s; %s" % (desy_2026.CAMPAIGN_TEXT["may"], desy_2026.BEAM_TEXT,
                          campaign.HEADER_LINE_1),
    footer=UN01_FOOTER)


def curve_record(c, cond):
    r = c["row"]
    return dict(label=c["label"], campaign=c["campaign"], run=c["run"],
                is_source=bool(c["is_source"]), n_fit=int(r.n_fit), combo=str(r.combo),
                c2=float(r.c2), c1=float(r.c1), c0=float(r.c0), q05=float(r.q05),
                q95=float(r.q95), tot_min=c["tot_min"], tot_max=c["tot_max"],
                mean_removed_ps=c["mean_removed"], hist_lo=float(r.hist_lo),
                hist_counts=[int(v) for v in c["counts"]], conditions=cond)


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
          histograms="the fit's TOT histogram, 25 ps bins from hist_lo, drawn as the fraction "
                     "of its events per bin",
          pixel="per panel the pixel fitted in the most runs of the panel, then the largest "
                "n_fit summed over them; a tie keeps the order the pixels were met in",
          settings="IRRAD: chip, bias, fluence step, threshold offset and RFSel from "
                   "runs_summary.csv, the March RFSel from MARCH_RFSEL; DESY: from the board-"
                   "config yaml through desy_2026.run_settings, temperature = board sensor value",
          fluence=campaign.FLUENCE_CONVENTION),
     inputs=UN01_INPUTS)
