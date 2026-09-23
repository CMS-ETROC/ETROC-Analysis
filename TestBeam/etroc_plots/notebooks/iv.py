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
# # IV scans and bias currents: CERN IRRAD 2026
#
# The leakage-current figures of the CERN IRRAD 2026 campaign, for the H1 (HPK) and F1 (FBK)
# telescopes. Each figure has a markdown cell saying what it shows and a code cell that draws
# it. Every figure is saved as PNG (200 dpi) and PDF, with `<name>_values.json` beside it:
# every number drawn, the input files read and the commit of the code.
#
# To run it:
#
# 1. `source TestBeam/condor_at_lxplus/envs/load_python39.sh` (LCG_104d, the environment this
#    notebook is tested in).
# 2. The inputs are not in the repository. They are read from two EOS locations: the input
#    tables (31 MB) from `/eos/user/m/musafdar/ETROC_plot_inputs/irrad_2026`, the raw March scans
#    and slow-control logs from `/eos/user/m/musafdar/CERN_IRRAD_Mar2026/IVCurves`.
#    `../iv/campaigns/irrad_2026_inputs.md` lists every table with where it came from, and ends
#    with a table of the environment variables that point the notebook at your own copies
#    (`ETROC_IV_INPUTS` for the tables, `ETROC_IV_EOS_MARCH` for the raw tree). The first code
#    cell checks every table against the published checksums and every raw file for existence
#    and read access, and stops listing all the problems it found.
# 3. Open `iv.ipynb` in Jupyter and run all cells, or run it headless with
#    `jupyter nbconvert --to notebook --execute iv.ipynb --output-dir <somewhere>`.
#
# Figures go to `$ETROC_FIGURES/iv/`, by default `figures/iv/` next to this notebook. Setting
# `PANELS = True` in the setup cell below also writes each panel of a figure as a file of its own
# next to the compound figure, named `<stem>_01_h1.png`, `<stem>_02_f1.png` and so on. Figures 24
# and 25 write their per-run panels instead; figures 5 and 34-39 have no single-panel export.
#
# The campaign's scan catalogue, run lists and input paths live in
# `../iv/campaigns/irrad_2026.py`. This notebook holds the choices made per figure: which scan
# stands for each irradiation step, axis ranges, legend placement.
#
# `iv.ipynb` and `iv.py` are the same notebook, kept in step by jupytext. Edit either one, then
# run `jupytext --sync iv.ipynb` and `nbstripout iv.ipynb` before committing. Neither tool is in
# LCG_104d (`pip install --user jupytext nbstripout`); running the notebook needs neither.

# %%
import os
import sys

import matplotlib
matplotlib.use("Agg")   # draw off-screen; show() displays the PNG exactly as saved
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.signal import find_peaks
from IPython.display import Image, display

# The folder that holds the package (TestBeam/) is two folders up. Jupyter starts the kernel in
# the notebook's folder; `python iv.py` has __file__.
here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
sys.path.insert(0, os.path.normpath(os.path.join(here, "..", "..")))

CAMPAIGN = "irrad_2026"                       # iv/campaigns/<CAMPAIGN>.py
# An exported ETROC_IV_CAMPAIGN wins; it is read once, on the first etroc_plots.iv import.
os.environ.setdefault("ETROC_IV_CAMPAIGN", CAMPAIGN)

from etroc_plots import talk_style as ts                      # noqa: E402
from etroc_plots.iv import campaigns                          # noqa: E402
from etroc_plots.iv import iv_data as ivd                     # noqa: E402
from etroc_plots.iv import iv_plot as ivp                     # noqa: E402
from etroc_plots.iv import iv_timeseries as tsdata            # noqa: E402
from etroc_plots.iv import preirrad_current                   # noqa: E402
from etroc_plots.iv.kfactor import k_factor, fine_step_span   # noqa: E402

if campaigns.NAME != CAMPAIGN:
    raise RuntimeError("the package is running campaign %r (ETROC_IV_CAMPAIGN) but CAMPAIGN = %r: "
                       "set one to match the other, then restart the kernel (the campaign is "
                       "read once, on the first import)" % (campaigns.NAME, CAMPAIGN))

# stops, listing every missing or unreadable table or raw file; reports any table that differs
# from the published copy
ivd.check_inputs()

NOTEBOOK = "TestBeam/etroc_plots/notebooks/iv.ipynb"   # recorded as "script" in every values file
OUT = os.path.join(os.path.abspath(os.environ.get("ETROC_FIGURES", "figures")), "iv")
os.makedirs(OUT, exist_ok=True)

PANELS = False   # True: also write each figure's single panels, where it has them (see above)


def show(stem):
    """Display a saved figure, exactly as written to disk."""
    display(Image(filename=os.path.join(OUT, stem + ".png"), width=900))


def flatten_panels(stem, paths):
    """Move ts.save_panel's nested panels/<stem>/NN_name.{png,pdf} next to the compound figure,
    flat-named <stem>_NN_name.* (this notebook's single-panel naming), and remove the now-empty
    nested folder."""
    nested_dir = None
    for p in paths:
        nested_dir = os.path.dirname(p)
        base = os.path.basename(p)
        name, ext = base.rsplit(".", 1)
        os.replace(p, os.path.join(OUT, "%s_%s.%s" % (stem, name, ext)))
    if nested_dir and os.path.isdir(nested_dir) and not os.listdir(nested_dir):
        os.rmdir(nested_dir)
        parent = os.path.dirname(nested_dir)
        if os.path.isdir(parent) and not os.listdir(parent):
            os.rmdir(parent)


def export_panels(stem, draw, panel_subject, line1=None):
    """PANELS=True, for a plain compound_1x2 figure: h1/f1 single panels through the same
    draw(ax, tel) the compound figure used."""
    panels = {"h1": (lambda ax, ctx: draw(ax, "h1"), panel_subject),
              "f1": (lambda ax, ctx: draw(ax, "f1"), panel_subject)}
    chosen = ts.resolve_panels(["all"], panels)
    flatten_panels(stem, ivp.export_panels_1x2(stem, OUT, chosen, line1=line1))


# %% [markdown]
# ## March 2026: IV curves after each irradiation step
#
# In March both telescopes were irradiated in three steps, to 3e14, 9e14 and 1.5e15 p/cm2, and
# every scan was taken parked (no beam). For each step these figures use the fine scan taken
# about two days after the step. Before irradiation there is no step yet, so the fine scan taken
# right before the first step stands in, labelled by fluence alone.
#
# Every label reads "<fluence>, <time after the step>". `ivp.check_days` prints each scan's
# measured time since the step next to the label it was given. If a fine scan stops at or below
# 60 V, that step falls back to its quick scan and the footer says so; no March scan needs this
# on the present data.

# %%
FALLBACK_MAXV_V = 60.0

# (fluence, time after the step, scan role in ivd.MARCH_SCANS)
MARCH_LOOKS = [(0.0, None, "fine_ra"),
               (3e14, "+2 days", "fine_2d"),
               (9e14, "+2 days", "fine_2d"),
               (1.5e15, "+2 days", "fine_2d")]
MARCH_SWATCHES = ivp.look_series([(f, w) for f, w, _ in MARCH_LOOKS])


def march_scan(tel, fluence, role):
    """The catalogued scan for this step, or the step's quick scan when the fine scan stops at
    or below FALLBACK_MAXV_V. Returns (role, path, data, fell_back)."""
    path = ivd.MARCH_SCANS[tel][fluence][role]
    data = ivd.load_march(path, tel)
    maxv = max((float(v.max()) for v, i in data.values() if len(v)), default=0.0)
    if maxv > FALLBACK_MAXV_V:
        return role, path, data, False
    path = ivd.MARCH_SCANS[tel][fluence]["quick_last"]
    return "quick_last", path, ivd.load_march(path, tel), True


def march_curves(tel, mask_V=None):
    """One curve per (step, chip), coloured by look. `mask_V` drops points above that bias.
    Returns (curves, scans used, fluences that fell back to a quick scan)."""
    curves, used, fell_back = [], [], []
    for (f, when, role), (label, color) in zip(MARCH_LOOKS, MARCH_SWATCHES):
        role, path, data, fb = march_scan(tel, f, role)
        if fb:
            fell_back.append(f)
        bw = ivd.bin_width_march(role)
        start = ivd.march_scan_start_utc(path)
        for chip, (v, i) in data.items():
            if mask_V is not None:
                v, i = ivp.lowv_mask(v, i, mask_V)
            curves.append(dict(v=v, i=i, chip=chip, fluence=f, bin_width_V=bw, look=label,
                               color=color, open_marker=True))
        used.append(dict(fluence=f, when=when, look=label, role=role, path=path,
                         scan=os.path.basename(path), fallback=fb,
                         start_utc=str(start) if start is not None else None,
                         n_points_per_chip={c: int(len(vv)) for c, (vv, ii) in data.items()}))
    ivp.check_days([dict(look=u["look"], when=u["when"], fluence=u["fluence"], scan=u["scan"],
                         start_utc=u["start_utc"]) for u in used], tag="March %s" % tel)
    return curves, used, fell_back


def scan_paths(res_h, res_f):
    """Input files of a two-panel figure, for its values file."""
    return [u["path"] for res in (res_h, res_f) for u in res[1]]


# %% [markdown]
# ### Figures 1 and 2: full range, linear and log y
#
# The fine scans over their full measured range (200-540 V), H1 left and F1 right, with every
# data-taking point drawn (0.1 V bins). Figure 1 has a linear y axis, 0-4000 µA. Figure 2 has a
# log y axis from each telescope's lowest pre-irradiation current up to 4000 µA. Both use the
# x range shared by every full-range IV figure, 0-620 V.

# %%
FULL_FOOTER = ("full-range fine scans, ~+2 days after each step (pre-irradiation: right before "
               "the first step); every data-taking point drawn (0.1 V bins); parked, no beam")


def full_values(curves):
    return [dict(chip=c["chip"], fluence=c["fluence"], look=c["look"],
                 i_at_150V_uA=ivd.value_at(c["v"], c["i"], 150.0),
                 v_max_V=float(c["v"].max()) if len(c["v"]) else None,
                 i_max_uA=float(c["i"].max()) if len(c["v"]) else None)
            for c in curves]


def log_floor(curves):
    """Lowest positive pre-irradiation current in the panel: the bottom of the log axis."""
    lows = []
    for c in curves:
        i = np.asarray(c["i"], dtype=float)
        i = i[i > 0]
        if c["fluence"] == 0.0 and len(i):
            lows.append(float(i.min()))
    return min(lows) if lows else 0.05


def full_footer(fell_back_h, fell_back_f):
    notes = ["%s %s: fine scan under %g V, quick scan used instead"
             % (name, ", ".join(ts.fluence_label(f, unit=False) for f in steps), FALLBACK_MAXV_V)
             for name, steps in (("H1", fell_back_h), ("F1", fell_back_f)) if steps]
    return FULL_FOOTER + ("  ·  " + "; ".join(notes) if notes else "")


def march_full_range(stem, yscale, title):
    def draw(ax, tel):
        curves, used, fell_back = march_curves(tel)
        # Some fine scans step ~1 V at low current up to ~75-80 V (DAQ behaviour, not a hole),
        # so the gap check starts at 80 V. Every point is still drawn.
        ivp.assert_no_gaps(curves, tag="%s %s" % (stem, tel), skip_below_V=80.0)
        if yscale == "linear":
            ylim, loc_f, loc_c, scale = ((0.0, 4000.0), ivp.LOC_LOOK_LIN, ivp.LOC_CHIP_LIN,
                                         ivp.SCALE_LIN)
        else:
            ylim, loc_f, loc_c, scale = ((log_floor(curves), 4000.0), ivp.LOC_LOOK_LOG,
                                         ivp.LOC_CHIP_LOG, ivp.SCALE_LOG)
        ivp.draw_iv_panel(ax, tel, curves, ivp.XLIM_FULL, ylim, yscale=yscale,
                          looks=MARCH_SWATCHES, legend_scale=scale, loc_f=loc_f, loc_c=loc_c,
                          point_scale=0.28, thin=False)
        return curves, used, fell_back, ylim

    ivp.compound_1x2(
        stem, OUT, draw, tag=title, script=NOTEBOOK, inputs=scan_paths,
        footer_text=lambda rh, rf: full_footer(rh[2], rf[2]),
        values_fn=lambda rh, rf: dict(
            h1=dict(scans=rh[1], curves=full_values(rh[0]), ylim_uA=list(rh[3])),
            f1=dict(scans=rf[1], curves=full_values(rf[0]), ylim_uA=list(rf[3])),
            xlim_V=list(ivp.XLIM_FULL), yscale=yscale))
    show(stem)
    if PANELS:
        export_panels(stem, draw, "full range, %s y" % yscale)


march_full_range("iv01_march", "linear", "IV curves per irradiation step, full range (linear y)")
march_full_range("iv02_march_log", "log", "IV curves per irradiation step, full range (log y)")

# %% [markdown]
# ### Figure 3: the low-V knee
#
# The same scans, 0-60 V on a linear y axis. The knee of the low-V scan is where the gain-layer
# depletion voltage is read off. It moves by a few volts from step to step, which the 10 V steps
# of a quick scan cannot resolve, so this view uses fine scans (0.1 V bins) only, with one marker
# about every 10 V so the markers still identify the chip.

# %%
LOWV_FOOTER = ("fine scans (0.1 V bins), low-V knee, ~+2 days after each step "
               "(pre-irradiation: right before the first step); one marker every ~10 V; "
               "parked, no beam")


def lowv_values(curves):
    return [dict(chip=c["chip"], fluence=c["fluence"], look=c["look"],
                 v_max_V=float(c["v"].max()) if len(c["v"]) else None,
                 i_max_uA=float(c["i"].max()) if len(c["v"]) else None)
            for c in curves]


def draw_march_lowv(ax, tel):
    curves, used, _ = march_curves(tel, mask_V=ivp.LOWV_MASK_V)
    ivp.assert_no_gaps(curves, tag="iv03_march_lowv %s" % tel, skip_below_V=10.0)
    for c in curves:
        c["markevery"] = ivp.lowv_markevery(c["v"])
    ivp.draw_iv_panel(ax, tel, curves, ivp.LOWV_XLIM, yscale="linear", looks=MARCH_SWATCHES,
                      legend_scale=ivp.SCALE_LOWV, loc_f=ivp.LOC_LOOK_LOWV,
                      loc_c=ivp.LOC_CHIP_LOWV, point_scale=0.55, thin=False)
    return curves, used


ivp.compound_1x2(
    "iv03_march_lowv", OUT, draw_march_lowv, tag="IV curves per irradiation step, low-V knee",
    footer_text=LOWV_FOOTER, script=NOTEBOOK, inputs=scan_paths,
    values_fn=lambda rh, rf: dict(h1=dict(scans=rh[1], curves=lowv_values(rh[0])),
                                  f1=dict(scans=rf[1], curves=lowv_values(rf[0])),
                                  xlim_V=list(ivp.LOWV_XLIM)))
show("iv03_march_lowv")
if PANELS:
    export_panels("iv03_march_lowv", draw_march_lowv, "low-V knee")

# %% [markdown]
# ## In-run bias currents
#
# ### Figure 4: in-run leakage current, March steps
#
# The HV monitor during the six March in-run current runs, H1 left and F1 right, read from the
# reduced table `march/march_inrun_60s.csv` in the inputs folder: leakage current [µA] against the
# time since each run's own start [min], linear y from 0. Colour is the fluence step, marker and
# line style the chip. Each run group carries one label at its line end giving only its bias, so a
# run is named by its condition rather than its run number. The six runs are the campaign module's
# list (`RUNS_AT_FLUENCE`, two per fluence step); no good-run list is applied to them.

# %%
IV04_SUBJECT = "in-run leakage current, March steps"
IV04_PANEL_SUBJECT = "in-run current"
IV04_FOOTER = ("HV monitor, 60 s medians, readback within 6 V of the run bias  ·  "
               "three fluence steps, two bias points per step, "
               "Disc threshold = baseline + 20")
IV04_RUN_INFO = tsdata.MARCH_H1_RUNS   # fluence + per-chip bias per run (shared H1/F1 numbering)


def iv04_bias(tel, run, chip):
    return (IV04_RUN_INFO[run]["bias"][chip] if tel == "h1"
            else tsdata._f1_windows()[run]["bias"][chip])


def iv04_panel_data(tel, df):
    """One telescope's drawn series, end-label groups, colour swatches and per-run stats."""
    d = df[df["tel"] == tel]
    lines, groups, stats = [], [], {}
    fluences = []
    for run in sorted(d["run"].unique()):
        fl = IV04_RUN_INFO[run]["fluence"]
        color = ts.fluence_color(fl)
        fluences.append(fl)
        x_ends, y_ends, biases = [], [], []
        for chip in tsdata.CHIPS[tel]:
            sub = d[(d["run"] == run) & (d["chip"] == chip)].sort_values("elapsed_min")
            if sub.empty:
                continue
            lines.append(dict(x=sub["elapsed_min"].values, y=sub["i_uA"].values,
                              chip=chip, color=color))
            bias = iv04_bias(tel, run, chip)
            stats["%d_%s" % (run, chip)] = dict(
                median=float(sub["i_uA"].median()),
                first_h=float(sub.loc[sub["elapsed_min"] <= 60, "i_uA"].median()),
                last_h=float(sub.loc[sub["elapsed_min"] >= sub["elapsed_min"].max() - 60,
                                     "i_uA"].median()),
                bias=bias)
            x_ends.append(float(sub["elapsed_min"].iloc[-1]))
            y_ends.append(float(sub["i_uA"].iloc[-1]))
            biases.append(bias)
        if x_ends:
            groups.append(dict(x_end=max(x_ends), y_end=sum(y_ends) / len(y_ends), color=color,
                               text="/".join(str(v) for v in sorted({int(round(b))
                                                                     for b in biases})) + " V"))
    swatches = [(ts.fluence_label(f), ts.fluence_color(f))
                for f in sorted({ts.fluence_key(f) for f in fluences})]
    return lines, groups, swatches, stats


def iv04_draw(ax, tel, df):
    lines, groups, swatches, stats = iv04_panel_data(tel, df)
    info = ivp.draw_currents_panel(ax, tel, lines, groups=groups, swatches=swatches)
    return dict(stats=stats, label_info=info, n_lines=len(lines), n_groups=len(groups))


_iv04_march_csv = os.path.join(tsdata.INPUTS, "march", "march_inrun_60s.csv")
_iv04_df = tsdata.load_march(_iv04_march_csv)


def iv04_draw_wrap(ax, tel):
    return iv04_draw(ax, tel, _iv04_df)


ivp.compound_1x2(
    "iv04_currents_march", OUT, iv04_draw_wrap, tag=IV04_SUBJECT, footer_text=IV04_FOOTER,
    line1=ts.HEADER_LINE_1, script=NOTEBOOK,
    inputs=[_iv04_march_csv, tsdata.F1_WINDOWS_CSV],
    values_fn=lambda res_h, res_f: dict(
        h1=res_h, f1=res_f, runs=sorted(int(r) for r in _iv04_df["run"].unique()),
        notes=("the six March in-run runs listed in the campaign module (RUNS_AT_FLUENCE, two "
              "per fluence step); one bias-only label per run group, no run numbers on the "
              "figure")))
show("iv04_currents_march")
if PANELS:
    export_panels("iv04_currents_march", iv04_draw_wrap, IV04_PANEL_SUBJECT,
                  line1=ts.HEADER_LINE_1)

# %% [markdown]
# ### Figure 5: pre-irradiation current stability
#
# HV readback (top) and sensor leakage current (bottom) during the pre-irradiation beam runs, from
# the iseg slow-control log (10 s medians), one trace per board. The figure is drawn by the package
# module `iv/preirrad_current.py` through its `plot()` function, the same one its command line runs
# (`python -m etroc_plots.iv.preirrad_current --out DIR`); only the output name differs here, so
# the figure lands beside the others.

# %%
preirrad_current.plot(out=OUT, stem="iv05_preirrad_current_stability")
show("iv05_preirrad_current_stability")
if PANELS:
    print("iv05_preirrad_current_stability: preirrad_current.plot() draws no single panels, "
          "so none are written")

# %% [markdown]
# ## V_gl: extraction method and evolution
#
# ### Figure 6: V_gl extraction method
#
# How the gain-layer depletion voltage V_gl is read off one example IV scan (H1 IH12, March
# 3e14 p/cm2 step, the fine low-V scan taken +2 days after that step). Left: |I| vs |V|, 0-60 V,
# every measured point, with V_gl marked as a vertical dashed line. Right: the k-factor k =
# (V/I)(dI/dV) for the same scan; V_gl is the peak of k. The k-factor code (`kfactor.k_factor`,
# `fine_step_span`, the prominence-peak convention) and the per-scan floor (min_v = 4 V) are the
# ones that produced this scan's value in `vgl_points.csv` (IH12, 3e14, +2 d: 31.4 V), so the V_gl
# drawn here reproduces the tabulated number rather than a re-derived one.

# %%
IV06_TEL = "h1"
IV06_CHIP = "IH12"
IV06_FLUENCE = 3e14
IV06_ROLE = "fine_2d"          # +2 day fine low-V scan
IV06_SCAN_DATE = "2026-03-18"  # from the scan filename (03182026_1654, local time)

# k-factor extraction parameters (min_v=4 V for every non-pre-irradiation March fine scan;
# prominence=0.3 is the setting of the campaign IV notebook, ExtractIVCurves.ipynb, that wrote
# vgl_points.csv).
IV06_SMOOTH_V = 1.0
IV06_MIN_V = 4.0
IV06_VGL_PROMINENCE = 0.3
IV06_FINE_MAX_STEP = 0.5
IV06_VGL_AGREE_V = 0.1   # the drawn V_gl must match vgl_points.csv this closely

IV06_XLIM = (0.0, 60.0)
IV06_COLOR = ts.fluence_color(IV06_FLUENCE)
IV06_FORMULA = r"$k = \dfrac{V}{I}\,\dfrac{dI}{dV}$"
IV06_ANNOTATION = (r"gain layer fully depleted at $V_{gl}$: multiplication" "\n"
                   r"turns on, the current rises steeply")
# compound-only: the same wording wrapped to 3 short lines and right-anchored so the block's own
# right edge stays left of the dashed V_gl line on the narrower half-width compound panel
IV06_ANNOTATION_COMPOUND = (r"gain layer fully depleted at $V_{gl}$:" "\n"
                            r"multiplication turns on," "\n"
                            r"the current rises steeply")
IV06_CAPTION_LEAKAGE = "Leakage vs bias"
IV06_CAPTION_KFACTOR = "k-factor"
IV06_HEADER_DATA_FULL = r"H1 IH12, $3\times10^{14}$ p/cm$^2$, +2 d fine scan"   # compound
IV06_HEADER_DATA_SHORT = r"IH12, $3\times10^{14}$ +2 d"                        # panel exports


def iv06_load_and_compute():
    """Load the example scan and run the same k-factor extraction as the kfactor package."""
    path = ivd.MARCH_SCANS[IV06_TEL][IV06_FLUENCE][IV06_ROLE]
    data = ivd.load_march(path, IV06_TEL)
    v, i = data[IV06_CHIP]                                # v in V, i in uA, ascending in V
    k = k_factor(v, i, smooth_v=IV06_SMOOTH_V)

    lo, hi = fine_step_span(v, max_step=IV06_FINE_MAX_STEP)
    lo = max(lo, IV06_MIN_V) if np.isfinite(lo) else np.nan
    if np.isfinite(lo) and np.isfinite(hi):
        sel = (v >= lo) & (v <= hi) & np.isfinite(k)
    else:
        raise RuntimeError("no finely stepped span in %s %s (%s): a peak searched over the whole "
                           "scan lands on the breakdown ramp, not V_gl"
                           % (IV06_TEL, IV06_CHIP, path))
    v_sel, k_sel = v[sel], k[sel]
    pk, _ = find_peaks(k_sel, prominence=IV06_VGL_PROMINENCE)
    if len(pk) == 0:
        raise RuntimeError("no k-factor peak found for %s %s (%s)" % (IV06_TEL, IV06_CHIP, path))
    best = pk[np.argmax(k_sel[pk])]
    v_gl, k_peak = float(v_sel[best]), float(k_sel[best])

    tab = ivd.load_vgl_points()
    row = tab[(tab["tel"] == IV06_TEL) & (tab["chip"] == IV06_CHIP) &
             (tab["fluence_p_cm2"] == IV06_FLUENCE) & (tab["timing"] == "+2 d")]
    vgl_table = float(row["vgl_V"].iloc[0]) if len(row) else None
    ivd.check_vgl_agreement([dict(tel=IV06_TEL, chip=IV06_CHIP, timing="+2 d", knee_V=v_gl,
                                  table_vgl_V=vgl_table)], IV06_VGL_AGREE_V, "figure 6")

    return dict(v=v, i=i, k=k, v_gl=v_gl, k_peak=k_peak, path=path,
               fine_lo=float(lo) if np.isfinite(lo) else None,
               fine_hi=float(hi) if np.isfinite(hi) else None,
               vgl_table_V=vgl_table)


def iv06_draw_leakage(ax, d, compound=False):
    m = d["v"] <= 62.0
    ax.plot(d["v"][m], d["i"][m], marker="o", markersize=3.2, markerfacecolor="none",
           markeredgecolor=IV06_COLOR, markeredgewidth=1.0, color=IV06_COLOR, linewidth=1.0,
           linestyle="-", alpha=0.9)
    ax.axvline(d["v_gl"], color=ts.INK, linestyle="--", linewidth=1.8, zorder=3)
    ymax = 1.10 * float(np.max(d["i"][m]))
    ax.text(d["v_gl"] + 1.2, 0.94 * ymax, r"$V_{gl}$ = %.1f V" % d["v_gl"], color=ts.INK,
           fontsize=ts.sizes()["ann"], va="top", ha="left")
    i_at_vgl = float(np.interp(d["v_gl"], d["v"], d["i"]))
    if compound:
        ax.annotate(IV06_ANNOTATION_COMPOUND, xy=(d["v_gl"], i_at_vgl), xycoords="data",
                   xytext=(28.5, 44.0), textcoords="data",
                   fontsize=ts.sizes()["ann"] * 0.72, color=ts.INK, va="top", ha="right",
                   arrowprops=dict(arrowstyle="->", color=ts.INK, lw=1.2))
    else:
        ax.annotate(IV06_ANNOTATION, xy=(d["v_gl"], i_at_vgl), xycoords="data",
                   xytext=(5.0, 0.86 * ymax), textcoords="data",
                   fontsize=ts.sizes()["ann"] * 0.9, color=ts.INK, va="top", ha="left",
                   arrowprops=dict(arrowstyle="->", color=ts.INK, lw=1.2))
    ax.set_xlim(*IV06_XLIM)
    ax.set_ylim(0.0, ymax)
    ax.set_xlabel(ts.XLABEL_BIAS)
    ax.set_ylabel(ivp.YLABEL_I)
    ts.style_axes(ax)
    return d


def iv06_draw_kfactor(ax, d):
    lo = d["fine_lo"] if d["fine_lo"] is not None else IV06_MIN_V
    m = (d["v"] >= lo) & (d["v"] <= IV06_XLIM[1])
    ax.plot(d["v"][m], d["k"][m], marker="o", markersize=2.6, markerfacecolor="none",
           markeredgecolor=IV06_COLOR, markeredgewidth=0.9, color=IV06_COLOR, linewidth=1.0,
           linestyle="-", alpha=0.9, markevery=ivd.markevery(int(m.sum())))
    ax.axvline(d["v_gl"], color=ts.INK, linestyle="--", linewidth=1.8, zorder=3)
    ax.plot([d["v_gl"]], [d["k_peak"]], marker="*", markersize=16, color=ts.INK,
           markeredgecolor=ts.INK, linestyle="none", zorder=4)
    ax.annotate(r"$V_{gl}$ = peak of k", xy=(d["v_gl"], d["k_peak"]), xycoords="data",
               xytext=(38.0, 0.92 * d["k_peak"]), textcoords="data",
               fontsize=ts.sizes()["ann"], color=ts.INK, va="center", ha="left",
               arrowprops=dict(arrowstyle="->", color=ts.INK, lw=1.2))
    ax.text(0.05, 0.94, IV06_FORMULA, transform=ax.transAxes, fontsize=ts.sizes()["label"] * 1.1,
           color=ts.INK, va="top", ha="left")
    ax.set_xlim(*IV06_XLIM)
    ax.set_ylim(0.0, 1.30 * d["k_peak"])
    ax.set_xlabel(ts.XLABEL_BIAS)
    ax.set_ylabel("k = (V/I)(dI/dV)")
    ts.style_axes(ax)
    return d


def iv06_footer_text(d):
    return ("smoothing: %.1f V boxcar; peak finder: find_peaks (prominence=%.1f), region >= %.0f V\n"
           "%s fine IV scan at 3e14 p/cm2, %s\n"
           "fine scan, 0.1 V steps, +2 d after the 3e14 step, parked, no beam, T = -25 C"
           % (IV06_SMOOTH_V, IV06_VGL_PROMINENCE, IV06_MIN_V, IV06_TEL.upper(), IV06_SCAN_DATE))


_iv06_d = iv06_load_and_compute()
ts.apply_style(1.0)
_iv06_fig, (_iv06_ax_l, _iv06_ax_k) = plt.subplots(1, 2, figsize=(11.5, 6.2))
_iv06_fig.subplots_adjust(left=0.08, right=0.97, top=0.89, bottom=0.20, wspace=0.28)

iv06_draw_leakage(_iv06_ax_l, _iv06_d, compound=True)
iv06_draw_kfactor(_iv06_ax_k, _iv06_d)

ts.right_lines(_iv06_ax_l, [IV06_CAPTION_LEAKAGE], ts.sizes(0.72)["title"])
ts.right_lines(_iv06_ax_k, [IV06_CAPTION_KFACTOR], ts.sizes()["title"])
ts.compound_header([_iv06_ax_l, _iv06_ax_k], line1=ivd.LINE1_PARKED, data=IV06_HEADER_DATA_FULL)
ts.footer(_iv06_fig, iv06_footer_text(_iv06_d), scale=0.85)

ts.lower_footer(_iv06_fig)   # the footer's final band, so the audits see the saved layout
_iv06_problems = ts.check_no_clipping(_iv06_fig, "iv06_vgl_method")
_iv06_legend_problems = ivp.check_legend_overlap(_iv06_fig, tag="iv06_vgl_method")
ts.save_figure(_iv06_fig, OUT, "iv06_vgl_method", audit=False)

_iv06_values = dict(tel=IV06_TEL, chip=IV06_CHIP, fluence_p_cm2=IV06_FLUENCE, timing="+2 d",
                    scan_path=_iv06_d["path"], scan_date=IV06_SCAN_DATE,
                    v_gl_V=round(_iv06_d["v_gl"], 2), k_peak=round(_iv06_d["k_peak"], 3),
                    vgl_points_csv_V=_iv06_d["vgl_table_V"],
                    params=dict(smooth_v_V=IV06_SMOOTH_V, min_v_V=IV06_MIN_V,
                               prominence=IV06_VGL_PROMINENCE, fine_lo_V=_iv06_d["fine_lo"],
                               fine_hi_V=_iv06_d["fine_hi"]),
                    overlap_problems=_iv06_problems, legend_overlap_problems=_iv06_legend_problems)
ivp.write_values(OUT, "iv06_vgl_method", _iv06_values, script=NOTEBOOK,
                 inputs=[_iv06_d["path"], os.path.join(ivd.INPUTS_VGL, "vgl_points.csv")])
plt.close(_iv06_fig)
show("iv06_vgl_method")

if PANELS:
    IV06_PANEL_CAPTION = {"leakage": IV06_CAPTION_LEAKAGE, "kfactor": IV06_CAPTION_KFACTOR}
    ts.apply_style(1.0)
    for _iv06_index, _iv06_name, _iv06_drawfn in (
            (1, "leakage", lambda ax: iv06_draw_leakage(ax, _iv06_d)),
            (2, "kfactor", lambda ax: iv06_draw_kfactor(ax, _iv06_d))):
        _iv06_pfig = plt.figure(figsize=(9.2, 7.0))
        _iv06_pax = _iv06_pfig.add_axes([0.11, 0.13, 0.85, 0.775])
        _iv06_drawfn(_iv06_pax)
        ts.talk_header(_iv06_pax, name=IV06_PANEL_CAPTION[_iv06_name], line1=ivd.LINE1_PARKED,
                       data=IV06_HEADER_DATA_SHORT)
        ts.check_no_clipping(_iv06_pfig, "iv06_vgl_method panel %s" % _iv06_name)
        ivp.check_legend_overlap(_iv06_pfig, tag="iv06_vgl_method panel %s" % _iv06_name)
        for _iv06_ext in ("png", "pdf"):
            _iv06_p = os.path.join(OUT, "iv06_vgl_method_%02d_%s.%s"
                                   % (_iv06_index, _iv06_name, _iv06_ext))
            _iv06_pfig.savefig(_iv06_p, dpi=200 if _iv06_ext == "png" else None,
                               facecolor=ts.SURFACE)
        plt.close(_iv06_pfig)

# %% [markdown]
# ### Figure 7: V_gl vs fluence, March 2026
#
# Gain-layer depletion voltage vs fluence through the March steps to 1.5e15: open markers are the
# scans right after each step, filled markers the scans after cooling down (+2 days). Colour is the
# fluence (the same convention as the IV curves), marker the chip. The x axis is the bookkeeping
# fluence, and no line joins points at different fluences.

# %%
IV07_YLABEL_VGL = r"$V_{gl}$ [V]"
# every V_gl plot uses the same fixed y axis, 0-55 V, so H1 and F1 panels stay directly comparable
IV07_YLIM = {"h1": (0.0, 55.0), "f1": (0.0, 55.0)}

VGL_FSCALE = 1.0e15
# x label and fluence-legend title: bookkeeping fluence, or fluence on chip when a variant sets
# chip_dose (every fluence then scaled by ivp.CHIP_DOSE_FACTOR)
VGL_XLABEL_FLUENCE_BOOKKEEPING = r"Bookkeeping fluence [$10^{15}$ p/cm$^2$]"
VGL_XLABEL_FLUENCE_ONCHIP = r"Fluence on chip [$10^{15}$ p/cm$^2$]"
VGL_LOC_STYLE = "upper right"
VGL_LOC_FLUENCE = "lower left"
VGL_COMPOUND_LEGEND_SCALE = 0.6

# Figures 7, 12 and 31-33 are all drawn by _vgl_draw(), set up per figure by an entry here: the
# fluence range, whether the 4-month rest points are drawn, the header text and, optionally,
# chip_dose=True for fluence on chip. Figures 31-33 add their entries in their own cell below.
VGL_VARIANTS = {
    "iv07_vgl_march": dict(fluence_max=1.7e15, include_4mo=False, data="March 2026"),
    "iv12_vgl_rest": dict(fluence_max=1.7e15, include_4mo=True, data="March 2026 + 4 months"),
}


def _vgl_clean(kw):
    kw = dict(kw)
    kw.pop("capsize", None)
    kw.pop("elinewidth", None)
    return kw


def _vgl_style_handles(rest_color, scale=1.0):
    h = [
        Line2D([], [], color=ts.INK, marker="o", markersize=9.0 * scale, linestyle="none",
              markerfacecolor=ts.SURFACE, markeredgecolor=ts.INK, markeredgewidth=1.8 * scale,
              label="right after (open)"),
        Line2D([], [], color=ts.INK, marker="o", markersize=9.0 * scale, linestyle="none",
              markerfacecolor=ts.INK, markeredgecolor=ts.INK,
              label="after cooling down (+2 d / +4 d)"),
    ]
    if rest_color:
        h.append(Line2D([], [], color=rest_color, marker="o", markersize=9.0 * scale,
                        linestyle="none", markerfacecolor=rest_color,
                        markeredgecolor=rest_color, label="+4 months rest (July 2026)"))
    return h


def _vgl_chip_handles(chips, scale=1.0):
    return [Line2D([], [], color=ts.INK, marker=ts.chip_marker(c), markersize=9.0 * scale,
                   linestyle="none", label=ts.chip_short(c)) for c in chips]


def _vgl_draw(ax, tel, spec, extra=None, legend_scale=VGL_COMPOUND_LEGEND_SCALE,
              skip_title=False):
    chip_dose = bool(spec.get("chip_dose"))
    dose_factor = ivp.CHIP_DOSE_FACTOR if chip_dose else 1.0
    rest_color = ivp.timing_color(1, None) if spec["include_4mo"] else None
    df = ivd.load_vgl_points()
    sub = df[df["tel"] == tel]
    chips = ts.TELESCOPE_CHIPS[tel]
    fmax = spec["fluence_max"]
    used = []
    for chip in chips:
        csub = sub[(sub["chip"] == chip) & (sub["fluence_p_cm2"] <= fmax)]
        ra = csub[csub["timing"] == "right after"].sort_values("fluence_p_cm2")
        for _, row in ra.iterrows():
            kw = ts.point_style(chip, row["fluence_p_cm2"], line=False, open_marker=True,
                                scale=legend_scale)
            ax.plot([row["fluence_p_cm2"] * dose_factor / VGL_FSCALE], [row["vgl_V"]],
                   **_vgl_clean(kw))
        ann = csub[csub["timing"].isin(["+2 d", "+4 d"])].sort_values("fluence_p_cm2")
        for _, row in ann.iterrows():
            kw = ts.point_style(chip, row["fluence_p_cm2"], line=False, open_marker=False,
                                scale=legend_scale)
            ax.plot([row["fluence_p_cm2"] * dose_factor / VGL_FSCALE], [row["vgl_V"]],
                   **_vgl_clean(kw))
        mo = csub[csub["timing"] == "4 months"] if spec["include_4mo"] else csub.iloc[0:0]
        for _, row in mo.iterrows():
            kw = ts.point_style(chip, row["fluence_p_cm2"], line=False, open_marker=False,
                                scale=legend_scale, color=rest_color)
            ax.plot([row["fluence_p_cm2"] * dose_factor / VGL_FSCALE], [row["vgl_V"]],
                   **_vgl_clean(kw))
        used.append(dict(chip=chip,
                         right_after=ra[["fluence_p_cm2", "vgl_V"]].to_dict("records"),
                         after_cooling_down=ann[["fluence_p_cm2", "vgl_V", "timing"]].to_dict("records"),
                         four_months=mo[["fluence_p_cm2", "vgl_V"]].to_dict("records")))

    ax.set_xlim(-0.03 * fmax * dose_factor / VGL_FSCALE, fmax * dose_factor / VGL_FSCALE)
    ax.set_ylim(*IV07_YLIM[tel])
    ax.set_xlabel(VGL_XLABEL_FLUENCE_ONCHIP if chip_dose else VGL_XLABEL_FLUENCE_BOOKKEEPING)
    ax.set_ylabel(IV07_YLABEL_VGL)
    ts.style_axes(ax)
    if not skip_title:
        ts.panel_title(ax, tel, extra=extra, scale=legend_scale)

    s = ts.sizes(legend_scale)
    combined = (_vgl_chip_handles(chips, legend_scale)
                + _vgl_style_handles(rest_color, legend_scale))
    l1 = ax.legend(handles=combined, loc=VGL_LOC_STYLE, fontsize=s["legend"] * 0.9, ncol=2,
                   columnspacing=0.8, handletextpad=0.5)
    ax.add_artist(l1)
    present = [f for f in ts.FLUENCE_STEPS if f <= fmax]
    fluence_handles_fn = ivp.chip_dose_fluence_handles if chip_dose else ts.fluence_handles
    fluence_fontsize = s["legend"] * (0.68 if chip_dose else 0.9)
    ax.legend(handles=fluence_handles_fn(present, legend_scale), loc=VGL_LOC_FLUENCE,
             fontsize=fluence_fontsize, ncol=2,
             title=("fluence on chip" if chip_dose else "bookkeeping fluence"),
             title_fontsize=s["legend_title"], columnspacing=0.8, handletextpad=0.5)

    return used


def vgl_figure(stem):
    spec = VGL_VARIANTS[stem]
    ts.apply_style(1.0)
    fig, (ax_h1, ax_f1) = plt.subplots(1, 2, figsize=ivp.COMPOUND_FIGSIZE)
    fig.subplots_adjust(**dict(ivp.COMPOUND_ADJUST, bottom=0.16))

    u_h1 = _vgl_draw(ax_h1, "h1", spec)
    u_f1 = _vgl_draw(ax_f1, "f1", spec)

    ts.compound_header([ax_h1, ax_f1], line1=ivd.LINE1_PARKED, data=spec["data"], pad=6)
    footer_text = ("gain-layer depletion voltage, from the k-factor peak of the fine low-V "
                  "scan; open = right after, filled = after cooling down")
    if spec.get("chip_dose"):
        footer_text += u"\n" + ivp.CHIP_DOSE_FOOTER
    ts.footer(fig, footer_text)

    ts.lower_footer(fig)   # the footer's final band, so the audits see the saved layout
    problems = ts.check_no_clipping(fig, stem)
    legend_problems = ivp.check_legend_overlap(fig, tag=stem)
    box_problems = ivp.check_legend_boxes(fig, tag=stem)
    ts.save_figure(fig, OUT, stem, audit=False)

    values = dict(h1=u_h1, f1=u_f1, overlap_problems=problems,
                 legend_overlap_problems=legend_problems, legend_box_problems=box_problems)
    ivp.write_values(OUT, stem, values, script=NOTEBOOK,
                     inputs=[os.path.join(ivd.INPUTS_VGL, "vgl_points.csv")])
    plt.close(fig)
    return values


def vgl_export_panels(stem):
    spec = VGL_VARIANTS[stem]
    ts.apply_style(1.0)
    for index, name in ((1, "h1"), (2, "f1")):
        fig = plt.figure(figsize=ivp.PANEL_FIGSIZE)
        ax = fig.add_axes(ivp.PANEL_AXES)
        _vgl_draw(ax, name, spec, extra="", legend_scale=0.9, skip_title=True)
        ts.talk_header(ax, name=ts.TELESCOPE_TITLE[name], line1=ivd.LINE1_PARKED,
                       data=spec["data"], pad=6)
        ts.check_no_clipping(fig, "%s panel %s" % (stem, name))
        ivp.check_legend_overlap(fig, tag="%s panel %s" % (stem, name))
        ivp.check_legend_boxes(fig, tag="%s panel %s" % (stem, name))
        for ext in ("png", "pdf"):
            p = os.path.join(OUT, "%s_%02d_%s.%s" % (stem, index, name, ext))
            fig.savefig(p, dpi=200 if ext == "png" else None, facecolor=ts.SURFACE)
        plt.close(fig)


vgl_figure("iv07_vgl_march")
show("iv07_vgl_march")
if PANELS:
    vgl_export_panels("iv07_vgl_march")

# %% [markdown]
# ## The four-month rest at 1.5e15
#
# ### Figures 8 and 9: IV before/after the four-month rest, 1.5e15 (linear and log y)
#
# Both curves of a chip are at the same fluence: the colour marks when the scan was taken (the
# March 1.5e15 fluence colour before the rest, the after-the-rest teal after it), not a fluence
# difference. H1's after-the-rest scan is the one taken +2 days into the July campaign (the first
# quick scan after the rest carries an IH13 diagnostic-switch noise interval that ends it at
# 430 V); F1's is its first scan after the rest. Figure 8 is linear y, figure 9 is log y.

# %%
IV0809_FLUENCE = 1.5e15
IV0809_XLIM = ivp.XLIM_FULL
IV0809_LOG_YLIM = (1.0, 5000.0)

# H1's own after-the-rest scan (see markdown above); F1 keeps the catalogued one.
IV0809_H1_AFTER_REST_SCAN = "H1/20260719_042243_quick_IVscan"
IV0809_SCAN_CHOICE_IH13 = dict(
    checked_chip="IH13",
    rule="frac_dec (consecutive |I| decreases above 10 V) <= 10% and v_max_V >= 530",
    candidates=[dict(scan="H1/20260717_1517_quick_IVscan", label="first after the rest",
                     frac_dec=0.368, v_max_V=430.0, qualifies=False),
                dict(scan="H1/20260717_1643_quick_IVscan", label="+1 h", frac_dec=0.375,
                     v_max_V=500.0, qualifies=False),
                dict(scan=IV0809_H1_AFTER_REST_SCAN, label="+2 d into the July campaign, still "
                     "+4 months after the 1.5e15 step", frac_dec=0.0, v_max_V=540.0,
                     qualifies=True)],
    chosen=IV0809_H1_AFTER_REST_SCAN,
    f1_check=dict(scan="F1/20260717_1702_quick_IVscan", frac_dec=0.0, qualifies=True,
                  switched=False))

IV08_BEFORE = dict(source="march", fluence=IV0809_FLUENCE, role="quick_last", when="+2 days")
IV08_AFTER = dict(source="july", key="4mo_1p5e15", fluence=IV0809_FLUENCE, when="+4 months",
                  scan_by_tel={"h1": IV0809_H1_AFTER_REST_SCAN,
                              "f1": ivd.JULY_SCANS["4mo_1p5e15"]["f1"]})
IV0809_LOOKS = [IV08_BEFORE, IV08_AFTER]

IV0809_FOOTER = ("same fluence on both curves, colour marks when the scan was taken; quick scans, "
                 "10 V bins; H1's after-the-rest scan avoids an IH13 diagnostic-switch noise interval")

_iv0809_fl = ts.fluence_label(IV0809_FLUENCE)


def _iv0809_lin_ylim(curves):
    return (0.0, 1.1 * max(float(c["i"].max()) for c in curves))


def _iv08_build(panel=None):
    return ivp.look_figure(
        "iv08_rest_1p5e15", OUT, IV0809_LOOKS,
        tag="IV before and after the four-month rest, %s (linear y)" % _iv0809_fl,
        footer=IV0809_FOOTER, xlim=IV0809_XLIM, yscale="linear", ylim=_iv0809_lin_ylim,
        script=NOTEBOOK, panel_subject="full range, linear y",
        extra_values=dict(scan_choice=IV0809_SCAN_CHOICE_IH13), panel=panel)


_iv08_build()
show("iv08_rest_1p5e15")
if PANELS:
    flatten_panels("iv08_rest_1p5e15", _iv08_build(panel=["all"]))


def _iv09_build(panel=None):
    return ivp.look_figure(
        "iv09_rest_1p5e15_log", OUT, IV0809_LOOKS,
        tag="IV before and after the four-month rest, %s (log y)" % _iv0809_fl,
        footer=IV0809_FOOTER, xlim=IV0809_XLIM, yscale="log", ylim=IV0809_LOG_YLIM,
        script=NOTEBOOK, panel_subject="full range, log y",
        extra_values=dict(scan_choice=IV0809_SCAN_CHOICE_IH13), panel=panel)


_iv09_build()
show("iv09_rest_1p5e15_log")
if PANELS:
    flatten_panels("iv09_rest_1p5e15_log", _iv09_build(panel=["all"]))

# %% [markdown]
# ### Figure 10: IV after/before ratio, 1.5e15
#
# The ratio of the two looks figures 8 and 9 draw, |I(after)| / |I(before)| per chip, on a common
# 1 V grid restricted to each chip's own scan overlap (no extrapolation past either scan's range).
# The ratio line takes the after-the-rest look's colour, so it matches the curve it comes from. It
# reads the same BEFORE/AFTER looks as figures 8 and 9, so the figures cannot disagree about which
# scans they compare.

# %%
def _iv10_footer(skipped):
    base = ("after-the-rest / before ratio, common 1 V grid over each chip's scan overlap; "
           "H1's after-the-rest scan avoids an IH13 diagnostic-switch noise interval")
    if skipped:
        base += "; no ratio line (overlap < 50 V): %s" % ", ".join(skipped)
    return base


def _iv10_build(panel=None):
    return ivp.ratio_figure(
        "iv10_rest_1p5e15_ratio", OUT, IV08_BEFORE, IV08_AFTER, IV0809_XLIM,
        tag="IV after / before the four-month rest, %s" % ts.fluence_label(IV08_BEFORE["fluence"]),
        footer=_iv10_footer, ylabel="I(after rest) / I(before rest)", script=NOTEBOOK, panel=panel)


_iv10_build()
show("iv10_rest_1p5e15_ratio")
if PANELS:
    flatten_panels("iv10_rest_1p5e15_ratio", _iv10_build(panel=["all"]))

# %% [markdown]
# ### Figure 11: IV at the low-V knee, before and after the four-month rest
#
# The gain-layer knee at 1.5e15, seen twice with fine (0.1 V) scans on both sides: before (March,
# +2 days after the step) and after (+4 months, from the two 2026-07-16 legacy slow-control logs).
# A sanity check reproduces each drawn curve's k-factor-peak V_gl against the tabulated
# `vgl_points.csv` value (to within 0.1 V for the March curves, about 2 V for the July curves) on
# the scan's full fine range, not the drawn window; a chip past 3 V, or a curve with no knee or no
# tabulated value, stops the notebook.

# %%
IV11_FLUENCE = 1.5e15
IV11_YLIM = (0.0, 160.0)
IV11_MARCH_ROLE = "fine_2d"
IV11_JULY_FINE_KEY = "4mo_1p5e15_fine"

IV11_LOOKS = [
    dict(source="march", fluence=IV11_FLUENCE, role=IV11_MARCH_ROLE, when="+2 days"),
    dict(source="july_fine", key=IV11_JULY_FINE_KEY, fluence=IV11_FLUENCE, when="+4 months"),
]

IV11_FOOTER = ("fine scans on both sides, 0.1 V bins, one marker every ~10 V; the knee is the "
              "gain-layer depletion voltage;\n"
              "same sensors, same fluence, colour marks when the scan was taken "
              "(+2 days vs. +4 months after the step).")

IV11_VGL_SMOOTH_V = 1.0
IV11_VGL_MIN_V = 4.0
IV11_VGL_PROMINENCE = 0.3
IV11_VGL_FINE_MAX_STEP = 0.5
IV11_VGL_TABLE_TIMING = {"+2 days": "+2 d", "+4 months": "4 months"}
IV11_VGL_DISAGREE_V = 3.0


def _iv11_knee_V(v, i, what):
    """k-factor peak (V_gl estimate) for one curve, by the same recipe as figure 6. `what` names
    the curve in the error raised when it has no finely stepped span."""
    v = np.asarray(v, dtype=float)
    i = np.asarray(i, dtype=float)
    k = k_factor(v, i, smooth_v=IV11_VGL_SMOOTH_V)
    lo, hi = fine_step_span(v, max_step=IV11_VGL_FINE_MAX_STEP)
    lo = max(lo, IV11_VGL_MIN_V) if np.isfinite(lo) else np.nan
    if np.isfinite(lo) and np.isfinite(hi):
        sel = (v >= lo) & (v <= hi) & np.isfinite(k)
    else:
        raise RuntimeError("figure 11: no finely stepped span in %s: a peak searched over the "
                           "whole scan lands on the breakdown ramp, not V_gl" % what)
    v_sel, k_sel = v[sel], k[sel]
    if len(v_sel) < 3:
        return float("nan")
    pk, _ = find_peaks(k_sel, prominence=IV11_VGL_PROMINENCE)
    if len(pk) == 0:
        return float("nan")
    return float(v_sel[pk[np.argmax(k_sel[pk])]])


def iv11_sanity_check():
    """Every drawn curve's k-factor knee against vgl_points.csv, on the scans' FULL fine range
    (not the drawn window)."""
    df = ivd.load_vgl_points()
    rows = []
    for tel in ("h1", "f1"):
        curves, _used, _sw = ivp.load_looks(tel, IV11_LOOKS)
        for c in curves:
            when = None
            for lk, (label, _col) in zip(IV11_LOOKS, ivp.look_series(
                    [(lk["fluence"], lk.get("when")) for lk in IV11_LOOKS])):
                if label == c["look"]:
                    when = lk.get("when")
            timing = IV11_VGL_TABLE_TIMING.get(when)
            sub = df[(df["tel"] == tel) & (df["chip"] == c["chip"]) &
                     (df["fluence_p_cm2"] == IV11_FLUENCE) & (df["timing"] == timing)]
            table_V = float(sub["vgl_V"].iloc[0]) if len(sub) else float("nan")
            knee_V = _iv11_knee_V(c["v"], c["i"], "%s %s %s" % (tel.upper(), c["chip"], when))
            diff = (knee_V - table_V) if (np.isfinite(knee_V) and np.isfinite(table_V)) else None
            row = dict(tel=tel, chip=c["chip"], when=when, timing=timing,
                       knee_V=round(knee_V, 2) if np.isfinite(knee_V) else None,
                       table_vgl_V=table_V if np.isfinite(table_V) else None,
                       diff_V=round(diff, 2) if diff is not None else None)
            rows.append(row)
    ivd.check_vgl_agreement(rows, IV11_VGL_DISAGREE_V, "figure 11")
    return rows


def _iv11_build(panel=None):
    return ivp.look_figure(
        "iv11_lowv_rest", OUT, IV11_LOOKS,
        tag="IV at the low-V knee, before and after the four-month rest, %s"
            % ts.fluence_label(IV11_FLUENCE),
        footer=IV11_FOOTER, xlim=ivp.LOWV_XLIM, yscale="linear", ylim=IV11_YLIM,
        mask_V=ivp.LOWV_MASK_V, marker_step_V=ivp.LOWV_MARKER_STEP_V, point_scale=0.55,
        gap_skip_below_V=10.0, script=NOTEBOOK, panel_subject="low-V knee",
        extra_values=dict(vgl_sanity_check=iv11_sanity_check()), panel=panel)


_iv11_build()
show("iv11_lowv_rest")
if PANELS:
    flatten_panels("iv11_lowv_rest", _iv11_build(panel=["all"]))

# %% [markdown]
# ### Figure 12: V_gl vs fluence, plus the four-month rest
#
# Figure 7 plus the 1.5e15 points measured four months after the step, drawn in the teal of the
# after-the-rest looks so they are told apart by colour alone. It uses the `VGL_VARIANTS` and
# `_vgl_draw` code defined under figure 7.

# %%
vgl_figure("iv12_vgl_rest")
show("iv12_vgl_rest")
if PANELS:
    vgl_export_panels("iv12_vgl_rest")

# %% [markdown]
# ### Figure 13: in-run leakage current, before and after the rest
#
# Leakage current vs time in the run at the 1.5e15 step, the same sensors before and after the
# four-month rest, H1 left / F1 right. Exactly one run per telescope per look: before the rest,
# the March run at 1.5e15 (H1 400 V, F1 265 V), about +17 hours after the step; after the rest,
# the July H1 run at 540 V / July F1 run at 460 V, about +4 months after the step. x = hours from
# each run's own start, so the two looks are read on one clock. July curves drop the first 15
# minutes after the HV step (the readback is in tolerance there, the current is not).

# %%
IV13_FLUENCE = 1.5e15
IV13_SETTLE_MIN = 15.0
IV13_MARCH_RUN = {"h1": 17, "f1": 17}
IV13_JULY_RUN = {"h1": "h1_run5", "f1": "f1_run9"}
IV13_WHEN_BEFORE = "+17 hours"
IV13_WHEN_AFTER = "+4 months"
IV13_XLABEL = "Time since run start [h]"
IV13_SUBJECT = "in-run leakage current, before and after the rest"
IV13_PANEL_SUBJECT = "before / after the rest"
IV13_FOOTER = ("HV monitor, 60 s medians, readback within 6 V of the run bias, first 15 min after "
              "the HV step dropped  ·  one good run per look, Disc threshold = baseline + 20")


def iv13_march_bias(tel, run, chip):
    if tel == "h1":
        return tsdata.MARCH_H1_RUNS[run]["bias"][chip]
    return tsdata._f1_windows()[run]["bias"][chip]


def iv13_bias_text(biases):
    return "/".join(str(v) for v in sorted({int(round(b)) for b in biases})) + " V"


def iv13_context():
    march = tsdata.load_march()
    selection, info = tsdata.july_1p5e15_selection()
    timeline = tsdata.load_july_timeline()
    for tel, run in IV13_JULY_RUN.items():
        if run not in selection[tel]["chosen"]:
            raise RuntimeError("%s is not a good RFSel 2 / offset 20 run at 1.5e15 "
                               "(tsdata.july_1p5e15_selection); choose another in IV13_JULY_RUN"
                               % run)
    return dict(march=march, info=info, timeline=timeline, selection=selection)


def iv13_panel_data(tel, ctx):
    chips = tsdata.CHIPS[tel]
    colors = [ivp.timing_color(0, ts.fluence_color(IV13_FLUENCE)),
             ivp.timing_color(1, ts.fluence_color(IV13_FLUENCE))]
    lines, groups, stats, days = [], [], {}, {}

    # ------------------------------------------------------------ before the rest (March)
    run = IV13_MARCH_RUN[tel]
    d = ctx["march"]
    d = d[(d["tel"] == tel) & (d["run"] == run)]
    x_ends, y_ends, biases = [], [], []
    origin = (pd.Timestamp(tsdata.MARCH_H1_RUNS[run]["start_utc"]) if tel == "h1"
             else pd.Timestamp(tsdata._f1_windows()[run]["start_utc"]))
    days["before"] = ivp.days_after_step(IV13_FLUENCE, origin)
    for chip in chips:
        sub = d[d["chip"] == chip].sort_values("t_utc")
        if sub.empty:
            continue
        hours = (sub["t_utc"] - origin).dt.total_seconds() / 3600.0
        bias = iv13_march_bias(tel, run, chip)
        biases.append(bias)
        lines.append(dict(x=hours.values, y=sub["i_uA"].values, chip=chip, color=colors[0]))
        stats["before_%s" % chip] = dict(
            bias_V=float(bias), median_uA=float(sub["i_uA"].median()),
            first_h_uA=float(sub.loc[hours <= hours.min() + 1.0, "i_uA"].median()),
            last_h_uA=float(sub.loc[hours >= hours.max() - 1.0, "i_uA"].median()))
        x_ends.append(float(hours.iloc[-1]))
        y_ends.append(float(sub["i_uA"].iloc[-1]))
    if x_ends:
        groups.append(dict(x_end=max(x_ends), y_end=sum(y_ends) / len(y_ends), color=colors[0],
                           text=iv13_bias_text(biases)))
    before_bias = iv13_bias_text(biases) if biases else "n/a"

    # ------------------------------------------------------------- after the rest (July)
    run = IV13_JULY_RUN[tel]
    tel_name = tsdata.JULY_TEL[tel]
    keys = [(tel, run, c) for c in chips if (tel, run, c) in ctx["info"]]
    origin = min(ctx["info"][k]["start"] for k in keys)
    days["after"] = ivp.days_after_step(IV13_FLUENCE, origin)
    x_ends, y_ends, biases = [], [], []
    for k in keys:
        chip = k[2]
        w = ctx["info"][k]
        c_idx = chips.index(chip)
        tl = ctx["timeline"]
        m = ((tl["tel"] == tel_name) & (tl["channel"] == c_idx) & (tl["tb"] >= w["start"])
            & (tl["tb"] <= w["end"]) & ((tl["v"] - w["bias_V"]).abs() <= tsdata.BIAS_TOL_V))
        sub_all = tl.loc[m].sort_values("tb")
        if sub_all.empty:
            continue
        sub = sub_all.loc[sub_all["tb"] >= w["start"] + pd.Timedelta(minutes=IV13_SETTLE_MIN)]
        if sub.empty:
            continue
        hours = (sub["tb"] - origin).dt.total_seconds() / 3600.0
        biases.append(w["bias_V"])
        lines.append(dict(x=hours.values, y=sub["i_uA"].values, chip=chip, color=colors[1]))
        stats["after_%s" % chip] = dict(
            bias_V=float(w["bias_V"]), median_uA=float(sub["i_uA"].median()),
            first_h_uA=float(sub.loc[hours <= hours.min() + 1.0, "i_uA"].median()),
            last_h_uA=float(sub.loc[hours >= hours.max() - 1.0, "i_uA"].median()),
            dropped_settle_min=int(len(sub_all) - len(sub)))
        x_ends.append(float(hours.iloc[-1]))
        y_ends.append(float(sub["i_uA"].iloc[-1]))
    if x_ends:
        groups.append(dict(x_end=max(x_ends), y_end=sum(y_ends) / len(y_ends), color=colors[1],
                           text=iv13_bias_text(biases)))
    after_bias = iv13_bias_text(biases) if biases else "n/a"

    swatches = [(ivp.look_label(IV13_FLUENCE, IV13_WHEN_BEFORE), colors[0]),
               (ivp.look_label(IV13_FLUENCE, IV13_WHEN_AFTER), colors[1])]
    rec = dict(stats=stats, days=days, before_bias=before_bias, after_bias=after_bias,
              march_run=IV13_MARCH_RUN[tel], july_run=IV13_JULY_RUN[tel], n_lines=len(lines))
    return lines, groups, swatches, rec


def _iv13_ylim(ctx):
    """Both panels share one explicit y range: the union of the two telescopes' data."""
    top = 0.0
    for tel in ("h1", "f1"):
        for ln in iv13_panel_data(tel, ctx)[0]:
            if len(ln["y"]):
                top = max(top, float(max(ln["y"])))
    return (0.0, top * ivp.CURRENTS_YPAD if top > 0 else 1.0)


def iv13_draw(ax, tel, ctx, ylim):
    lines, groups, swatches, rec = iv13_panel_data(tel, ctx)
    rec["label_info"] = ivp.draw_currents_panel(ax, tel, lines, groups=groups, swatches=swatches,
                                                xlabel=IV13_XLABEL, ylim=ylim)
    return rec


_iv13_ctx = iv13_context()
_iv13_ylim_val = _iv13_ylim(_iv13_ctx)


def iv13_draw_wrap(ax, tel):
    return iv13_draw(ax, tel, _iv13_ctx, _iv13_ylim_val)


ivp.compound_1x2(
    "iv13_current_vs_time_rest", OUT, iv13_draw_wrap, tag=IV13_SUBJECT, footer_text=IV13_FOOTER,
    line1=ts.HEADER_LINE_1, script=NOTEBOOK,
    inputs=[tsdata.JULY_TIMELINE, tsdata.JULY_REF_CSV, tsdata.GOOD_RUNS_CSV,
            os.path.join(tsdata.INPUTS, "march", "march_inrun_60s.csv"), tsdata.F1_WINDOWS_CSV,
            tsdata.JULY_YAML],
    values_fn=lambda res_h, res_f: dict(
        fluence_p_cm2=IV13_FLUENCE, h1=res_h, f1=res_f, ylim_uA=list(_iv13_ylim_val),
        looks=[ivp.look_label(IV13_FLUENCE, IV13_WHEN_BEFORE),
              ivp.look_label(IV13_FLUENCE, IV13_WHEN_AFTER)]))
show("iv13_current_vs_time_rest")
if PANELS:
    export_panels("iv13_current_vs_time_rest", iv13_draw_wrap, IV13_PANEL_SUBJECT,
                  line1=ts.HEADER_LINE_1)

# %% [markdown]
# ## July 2026: IV at 2e15
#
# ### Figures 14 and 15: full range, linear and log y
#
# The three July scans of the 2e15 step over their whole measured range, H1 left / F1 right: right
# after the step, then again at +2 days and +3 days. The "+2 days" look is the step's fine (0.1 V)
# scan, chosen over the 10 V quick scan of the same age because it carries more points; the other
# two looks are 10 V quick scans (stated in the footer). Figure 14 is linear y, figure 15 is log y.

# %%
IV1415_FLUENCE = 2e15
IV1415_XLIM = ivp.XLIM_FULL
IV1415_LOG_YLIM = (1.0, 5000.0)

IV1415_LOOKS = [
    dict(source="july", key="2e15_ra", fluence=IV1415_FLUENCE, when="right after"),
    dict(source="july", key="2e15_2d", fluence=IV1415_FLUENCE, when="+2 days"),
    dict(source="july", key="2e15_3d", fluence=IV1415_FLUENCE, when="+3 days"),
]

IV1415_FOOTER = ("full-range scans, one colour per look; 10 V quick scans except the +2 days "
                 "look, which is a 0.1 V fine scan; July 2026, parked, no beam")


def _iv1415_lin_ylim(curves):
    return (0.0, 1.1 * max(float(c["i"].max()) for c in curves))


def _iv14_build(panel=None):
    return ivp.look_figure(
        "iv14_2e15", OUT, IV1415_LOOKS,
        tag="IV at %s: right after / +2 / +3 days (linear y)" % ts.fluence_label(IV1415_FLUENCE),
        footer=IV1415_FOOTER, xlim=IV1415_XLIM, yscale="linear", ylim=_iv1415_lin_ylim,
        script=NOTEBOOK, panel_subject="full range, linear y", panel=panel)


_iv14_build()
show("iv14_2e15")
if PANELS:
    flatten_panels("iv14_2e15", _iv14_build(panel=["all"]))


def _iv15_build(panel=None):
    return ivp.look_figure(
        "iv15_2e15_log", OUT, IV1415_LOOKS,
        tag="IV at %s: right after / +2 / +3 days (log y)" % ts.fluence_label(IV1415_FLUENCE),
        footer=IV1415_FOOTER, xlim=IV1415_XLIM, yscale="log", ylim=IV1415_LOG_YLIM,
        script=NOTEBOOK, panel_subject="full range, log y", panel=panel)


_iv15_build()
show("iv15_2e15_log")
if PANELS:
    flatten_panels("iv15_2e15_log", _iv15_build(panel=["all"]))

# %% [markdown]
# ### Figure 16: the low-V knee
#
# The 2e15 step has exactly two fine scans, right after the step and +2 days, and both are drawn
# here, 0-60 V linear y, H1 left / F1 right. The "+3 days" look of figures 14 and 15 has no fine
# scan, so it is not drawn. Fine scans only (0.1 V bins): the gain-layer depletion voltage is read
# off the knee of a low-V scan, and it moves by a few volts, which a 10 V quick scan cannot
# resolve.

# %%
IV16_FLUENCE = 2e15
IV16_LOOKS = [
    dict(source="july", key="2e15_ra_fine", fluence=IV16_FLUENCE, when="right after"),
    dict(source="july", key="2e15_2d_fine", fluence=IV16_FLUENCE, when="+2 days"),
]

IV16_FOOTER = ("fine scans, 0.1 V bins, one marker every ~10 V; the +3 days look has no fine scan "
              "and is not drawn here; July 2026, parked, no beam")


def _iv16_build(panel=None):
    return ivp.look_figure(
        "iv16_2e15_lowv", OUT, IV16_LOOKS,
        tag="IV at %s, low-V knee" % ts.fluence_label(IV16_FLUENCE),
        footer=IV16_FOOTER, xlim=ivp.LOWV_XLIM, yscale="linear", mask_V=ivp.LOWV_MASK_V,
        marker_step_V=ivp.LOWV_MARKER_STEP_V, point_scale=0.55, gap_skip_below_V=10.0,
        script=NOTEBOOK, panel_subject="low-V knee", panel=panel)


_iv16_build()
show("iv16_2e15_lowv")
if PANELS:
    flatten_panels("iv16_2e15_lowv", _iv16_build(panel=["all"]))

# %% [markdown]
# ## July 2026: IV at 3.5e15
#
# ### Figures 17 and 18: full range, linear and log y
#
# The three looks of the 3.5e15 step (right after, +4 days, +10 days) over their full range, 10 V
# quick scans, H1 left / F1 right. On H1's "+10 days" scan (5 August), IH12 shows a current step of
# about 3.6x at 570-580 V, seen in both 5 August scans; it is drawn as recorded and noted in the
# footer. F1 shows no such step. Figure 17 is linear y, figure 18 is log y.

# %%
IV1718_FLUENCE = 3.5e15
IV1718_XLIM = ivp.XLIM_FULL
IV1718_LOG_YLIM = (1.0, 8000.0)

IV1718_LOOKS = [
    dict(source="july", key="3p5e15_ra", fluence=IV1718_FLUENCE, when="right after"),
    dict(source="july", key="3p5e15_4d", fluence=IV1718_FLUENCE, when="+4 days"),
    dict(source="july", key="3p5e15_end", fluence=IV1718_FLUENCE, when="+10 days"),
]

IV1718_FOOTER = ("full-range quick scans, 10 V bins, one colour per look\n"
                 "H1 +10 days: IH12 shows a reproducible current step at 570-580 V in both 5 "
                 "August scans, drawn as recorded. F1 +10 days is its 5 August scan.")


def _iv1718_lin_ylim(curves):
    return (0.0, 1.1 * max(float(c["i"].max()) for c in curves))


_iv1718_fl = ts.fluence_label(IV1718_FLUENCE)


def _iv17_build(panel=None):
    return ivp.look_figure(
        "iv17_3p5e15", OUT, IV1718_LOOKS,
        tag="IV at %s: right after / +4 days / +10 days (linear y)" % _iv1718_fl,
        footer=IV1718_FOOTER, xlim=IV1718_XLIM, yscale="linear", ylim=_iv1718_lin_ylim,
        script=NOTEBOOK, panel_subject="full range, linear y", panel=panel)


_iv17_build()
show("iv17_3p5e15")
if PANELS:
    flatten_panels("iv17_3p5e15", _iv17_build(panel=["all"]))


def _iv18_build(panel=None):
    return ivp.look_figure(
        "iv18_3p5e15_log", OUT, IV1718_LOOKS,
        tag="IV at %s: right after / +4 days / +10 days (log y)" % _iv1718_fl,
        footer=IV1718_FOOTER, xlim=IV1718_XLIM, yscale="log", ylim=IV1718_LOG_YLIM,
        script=NOTEBOOK, panel_subject="full range, log y", panel=panel)


_iv18_build()
show("iv18_3p5e15_log")
if PANELS:
    flatten_panels("iv18_3p5e15_log", _iv18_build(panel=["all"]))

# %% [markdown]
# ### Figure 19: the low-V knee
#
# All three fine scans of the 3.5e15 step are drawn: right after, +4 days and +10 days, 0-60 V
# linear y, H1 left / F1 right. The right-after and +10 days fine scans were stopped at 35 V by the
# DAQ and the +4 days one at 60 V, so two of the three curves end inside the window; that is the
# reach of those scans, not a gap.

# %%
IV19_FLUENCE = 3.5e15
IV19_LOOKS = [
    dict(source="july", key="3p5e15_ra_fine", fluence=IV19_FLUENCE, when="right after"),
    dict(source="july", key="3p5e15_4d_fine", fluence=IV19_FLUENCE, when="+4 days"),
    dict(source="july", key="3p5e15_10d_fine", fluence=IV19_FLUENCE, when="+10 days"),
]

IV19_FOOTER = ("fine scans, 0.1 V bins, one marker every ~10 V; the right-after and +10 days "
              "scans reach 35 V, the +4 days scan 60 V; July 2026, parked, no beam")


def _iv19_build(panel=None):
    return ivp.look_figure(
        "iv19_3p5e15_lowv", OUT, IV19_LOOKS,
        tag="IV at %s, low-V knee" % ts.fluence_label(IV19_FLUENCE),
        footer=IV19_FOOTER, xlim=ivp.LOWV_XLIM, yscale="linear", mask_V=ivp.LOWV_MASK_V,
        marker_step_V=ivp.LOWV_MARKER_STEP_V, point_scale=0.55, gap_skip_below_V=10.0,
        script=NOTEBOOK, panel_subject="low-V knee", panel=panel)


_iv19_build()
show("iv19_3p5e15_lowv")
if PANELS:
    flatten_panels("iv19_3p5e15_lowv", _iv19_build(panel=["all"]))

# %% [markdown]
# ### Figure 20: current at 480 V vs hours after the step
#
# The cooling-down effect as a number rather than a curve shape: the tabulated current at 480 V
# of each 3.5e15 quick scan, against hours since the step ended, H1 left / F1 right. Symlog x with
# a 1 h linear threshold so the right-after scan sits at 0; the step end is the first post-step IV
# scan, an upper bound, so every hour count is a slight under-estimate (stated in the footer).
# Colour = look, marker = chip, markers only, no connecting line between a chip's looks (one line
# cannot carry three different look colours). Uses the same three-scan sequence as figures 17 and
# 18, so the two views can never disagree about which scans the sequence contains.

# %%
IV20_V_AT = 480.0
IV20_POINT_SCALE = 0.5
IV20_XLABEL = "Hours after irradiation stop [h]"
IV20_XTICKS = (0, 1, 10, 100)
IV20_LINTHRESH = 1.0
IV20_LEGEND_SCALE = 0.70
IV20_LOOKS = IV1718_LOOKS   # the same three looks as figures 17 and 18

IV20_FOOTER = ("quick scans; tabulated current at %d V against hours after the step ended "
              "(symlog x; the step end is the first scan after it, an upper bound)"
              % int(IV20_V_AT))


def _iv20_rows(tel):
    df = ivd.july_scan_table()
    df = df[df["tel"] == tel.upper()]
    rad_stop = pd.Timestamp(ivd.RAD_STOP_UTC[IV1718_FLUENCE])
    rows = []
    for lk, (label, _color) in zip(IV20_LOOKS, ivp.look_series(
            [(x["fluence"], x.get("when")) for x in IV20_LOOKS])):
        scan_id = ivd.JULY_SCANS[lk["key"]][tel].split("/", 1)[1]
        sub = df[df["scan"] == scan_id]
        if sub.empty:
            continue
        hours = max((pd.Timestamp(sub.iloc[0]["start_utc"]) - rad_stop).total_seconds() / 3600.0,
                    0.0)
        for chip in ts.TELESCOPE_CHIPS[tel]:
            r = sub[sub["board"] == "PT_" + chip]
            if r.empty or pd.isna(r.iloc[0]["I_%dV_uA" % int(IV20_V_AT)]):
                continue
            rows.append(dict(chip=chip, look=label, when=lk.get("when"), hours=hours,
                             i_uA=float(r.iloc[0]["I_%dV_uA" % int(IV20_V_AT)])))
    return rows


def _iv20_draw(ax, tel):
    rows = _iv20_rows(tel)
    look_colors = dict(ivp.look_series([(x["fluence"], x.get("when")) for x in IV20_LOOKS]))
    for chip in ts.TELESCOPE_CHIPS[tel]:
        sub = sorted([r for r in rows if r["chip"] == chip], key=lambda r: r["hours"])
        for r in sub:
            kw = ts.point_style(chip, IV1718_FLUENCE, color=look_colors[r["look"]],
                                scale=IV20_POINT_SCALE, line=False)
            kw.pop("capsize", None)
            kw.pop("elinewidth", None)
            ax.plot([r["hours"]], [r["i_uA"]], **kw)
    ax.set_xscale("symlog", linthresh=IV20_LINTHRESH)
    ax.set_xticks(IV20_XTICKS)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel(IV20_XLABEL)
    ax.set_ylabel(r"$|I|$ at %d V [$\mu$A]" % int(IV20_V_AT))
    ts.style_axes(ax)
    ts.panel_title(ax, tel)
    s = ts.sizes(IV20_LEGEND_SCALE)
    lf = ax.legend(handles=ivp.look_handles(ivp.look_series(
        [(x["fluence"], x.get("when")) for x in IV20_LOOKS]), IV20_LEGEND_SCALE),
        loc="upper right", fontsize=s["legend"])
    ax.add_artist(lf)
    ax.legend(handles=ts.chip_handles(ts.TELESCOPE_CHIPS[tel], IV20_LEGEND_SCALE),
              loc="lower left", fontsize=s["legend"])
    return rows


def _iv20_build(panel=None):
    panels = {"h1": (lambda ax, ctx: _iv20_draw(ax, "h1"), "current at %d V" % int(IV20_V_AT)),
             "f1": (lambda ax, ctx: _iv20_draw(ax, "f1"), "current at %d V" % int(IV20_V_AT))}
    chosen = ts.resolve_panels(panel, panels)
    if chosen:
        return ivp.export_panels_1x2("iv20_3p5e15_hours", OUT, chosen)
    return ivp.compound_1x2(
        "iv20_3p5e15_hours", OUT, lambda ax, tel: _iv20_draw(ax, tel),
        tag="Current at %d V vs time after the %s step"
            % (int(IV20_V_AT), ts.fluence_label(IV1718_FLUENCE)),
        footer_text=IV20_FOOTER, script=NOTEBOOK,
        values_fn=lambda rh, rf: dict(h1=rh, f1=rf, v_at_V=IV20_V_AT),
        inputs=[os.path.join(ivd.INPUTS_JULY, "july_iv_scans.csv")])


_iv20_build()
show("iv20_3p5e15_hours")
if PANELS:
    flatten_panels("iv20_3p5e15_hours", _iv20_build(panel=["all"]))

# %% [markdown]
# ## July 2026: in-run bias currents
#
# ### Figures 21-23: in-run leakage current, by fluence step
#
# The HV monitor during the July beam runs, one figure per fluence step (1.5e15, 2e15, 3.5e15), H1
# left / F1 right. Each bias condition has its own colour (the fluence colour for the highest bias,
# then the timing colours) and exactly one legend entry. The 1.5e15 step, the March sensors
# measured again after the four-month rest, is drawn from every good RFSel 2 / Disc offset 20 run
# at that step; 2e15 and 3.5e15 draw the good RFSel 2 / Disc offset 20 runs of the campaign's
# display-run list (`july/display_runs_jul.csv` in the inputs folder), and of those only the boards
# whose HV plateau in the per-run current table (`july/july_inrun_currents.csv`) lasts at least
# 4 h. Every run or board left out is listed in the values file with the reason (`rejected`,
# `not_good`, `dropped_boards`). A run whose drawn median disagrees with the reference current by
# more than 5% is dropped and reported. Every curve drops the first 15 minutes after the HV step
# (the readback is already in tolerance there, the current is not).

# %%
IV2123_ASSERT_TOL_PCT = 5.0
# h1_run57's IH7 channel shows a genuine excursion that disagrees with the reference current by
# -63%; the run is also off the good-run list, so the good-run gate catches it upstream as well.
IV2123_EXCLUDED_ASSERTION_FAIL = {("h1", "h1_run57")}
# The 1.5e15 step's true rad-stop is the March 2026 irradiation, about 4 months before these July
# runs, and there is no documented freezer-removal time, so the earliest logged July activity for
# either telescope is used as the July-restart proxy instead.
IV2123_JULY_1P5E15_RESTART_UTC = "2026-07-17 01:43:00"
IV2123_IH7_EXCURSION_NOTE = "IH7 excursions left as logged"

IV2123_SUBJECT = "in-run leakage current"
IV2123_PANEL_SUBJECT = "in-run current"
IV2123_FOOTER = ("HV monitor, 60 s medians, readback within %g V of the run bias, first 15 min "
                 "after the HV step dropped  ·  good runs only, RFSel %s, "
                 "Disc threshold = baseline + %s")   # tolerance, RFSel, offset
# second footer line at the irradiation steps (july_run_selection's plateau minimum)
IV2123_PLATEAU_FOOTER = ("boards with an HV plateau shorter than %g h in the per-run current "
                         "table are not drawn" % tsdata.JULY_MIN_PLATEAU_H)

IV2123_VARIANTS = {
    "iv21_currents_july_1p5e15": dict(fluence=1.5e15),
    "iv22_currents_july_2e15": dict(fluence=2e15),
    "iv23_currents_july_3p5e15": dict(fluence=3.5e15),
}


def _iv2123_select(fluence):
    """(selection, info, timeline, checked, excluded) for one July fluence step."""
    if ts.fluence_key(fluence) == ts.fluence_key(1.5e15):
        selection, info = tsdata.july_1p5e15_selection()
    else:
        step = [f for f in tsdata.JULY_STEPS if ts.fluence_key(f) == ts.fluence_key(fluence)]
        if not step:
            raise ValueError("%g p/cm2 is neither the 1.5e15 re-measurement nor a July "
                             "irradiation step %s" % (fluence, tsdata.JULY_STEPS))
        selection, info = tsdata.july_run_selection(steps=step)
    timeline = tsdata.load_july_timeline()
    checked, excluded = {}, []
    for tel_key, tel_name in tsdata.JULY_TEL.items():
        for run in list(selection[tel_key]["chosen"]):
            chip_res, run_fail = {}, False
            for c, chip in enumerate(tsdata.CHIPS[tel_key]):
                key = (tel_key, run, chip)
                if key not in info:
                    continue
                w = info[key]
                m = ((timeline["tel"] == tel_name) & (timeline["channel"] == c)
                     & (timeline["tb"] >= w["start"]) & (timeline["tb"] <= w["end"])
                     & ((timeline["v"] - w["bias_V"]).abs() <= tsdata.BIAS_TOL_V))
                sub = timeline.loc[m]
                am = (sub["tb"] >= w["assert_start"]) & (sub["tb"] <= w["assert_end"])
                drawn = sub.loc[am, "i_uA"].median()
                ref = w["ref_uA"]
                pct = 100.0 * (drawn - ref) / ref if pd.notna(drawn) else float("nan")
                chip_res[chip] = dict(drawn_uA=None if pd.isna(drawn) else round(float(drawn), 2),
                                      ref_uA=ref,
                                      pct_diff=None if pd.isna(pct) else round(float(pct), 3))
                if not (pd.notna(pct) and abs(pct) <= IV2123_ASSERT_TOL_PCT):
                    run_fail = True
            checked["%s/%s" % (tel_key, run)] = chip_res
            if run_fail or (tel_key, run) in IV2123_EXCLUDED_ASSERTION_FAIL:
                excluded.append("%s/%s" % (tel_key, run))
                selection[tel_key]["chosen"].remove(run)
                selection[tel_key]["rejected"].append(run + " (assertion fail)")
    return selection, info, timeline, checked, excluded


def _iv2123_run_step_reference(fluence):
    if ts.fluence_key(fluence) == ts.fluence_key(1.5e15):
        return pd.Timestamp(IV2123_JULY_1P5E15_RESTART_UTC)
    return pd.Timestamp(ivd.RAD_STOP_UTC[ts.fluence_key(fluence)])


def _iv2123_hours_label(bias_txt, hours, fluence):
    if ts.fluence_key(fluence) == ts.fluence_key(1.5e15) and hours > 48.0:
        return "%s, +%d d" % (bias_txt, int(round(hours / 24.0)))
    if hours < 10.0:
        return "%s, +%.1f h" % (bias_txt, hours)
    return "%s, +%.0f h" % (bias_txt, hours)


def _iv2123_bias_text(biases):
    return "/".join(str(v) for v in sorted({int(round(b)) for b in biases})) + " V"


def _iv2123_panel_data(tel, fluence, selection, info, timeline):
    tel_name = tsdata.JULY_TEL[tel]
    runs = []
    for run in selection[tel]["chosen"]:
        keys = [(tel, run, c) for c in tsdata.CHIPS[tel] if (tel, run, c) in info]
        keys = [k for k in keys
                if ts.fluence_key(float(info[k]["fluence"])) == ts.fluence_key(fluence)]
        if keys:
            runs.append((run, keys))
    keys_of = dict(runs)
    ref_utc = _iv2123_run_step_reference(fluence)
    run_bias = {run: max(info[k]["bias_V"] for k in keys) for run, keys in runs}
    order = sorted(run_bias, key=lambda r: -run_bias[r])
    color_of = {run: ivp.timing_color(i, ts.fluence_color(fluence)) for i, run in enumerate(order)}
    hours_of = {run: (min(info[k]["start"] for k in keys_of[run]) - ref_utc).total_seconds()
                     / 3600.0
                for run in order}

    lines, stats, excluded_min = [], {}, {}
    ends = {}
    for run, keys in runs:
        color = color_of[run]
        x_ends, y_ends, dropped = [], [], []
        for k in keys:
            chip = k[2]
            c = tsdata.CHIPS[tel].index(chip)
            w = info[k]
            m = ((timeline["tel"] == tel_name) & (timeline["channel"] == c)
                 & (timeline["tb"] >= w["start"]) & (timeline["tb"] <= w["end"])
                 & ((timeline["v"] - w["bias_V"]).abs() <= tsdata.BIAS_TOL_V))
            sub_all = timeline.loc[m].sort_values("tb")
            if sub_all.empty:
                continue
            sub = sub_all.loc[sub_all["tb"] >= w["assert_start"]]
            dropped.append(int(len(sub_all) - len(sub)))
            if sub.empty:
                continue
            elapsed = (sub["tb"] - w["start"]).dt.total_seconds() / 60.0
            lines.append(dict(x=elapsed.values, y=sub["i_uA"].values, chip=chip, color=color))
            stats["%s_%s" % (run, chip)] = dict(
                median=float(sub["i_uA"].median()),
                first_h=float(sub.loc[elapsed <= 60, "i_uA"].median()),
                last_h=float(sub.loc[elapsed >= elapsed.max() - 60, "i_uA"].median()),
                bias=float(w["bias_V"]), plateau_h=float(w["plateau_h"]),
                excluded_min=int(len(sub_all) - len(sub)), hours_after_step=hours_of[run])
            x_ends.append(float(elapsed.iloc[-1]))
            y_ends.append(float(sub["i_uA"].iloc[-1]))
        if dropped:
            excluded_min[run] = int(round(sum(dropped) / len(dropped)))
        if x_ends:
            ends[run] = (max(x_ends), sum(y_ends) / len(y_ends))
    groups = [dict(x_end=ends[run][0], y_end=ends[run][1], color=color_of[run],
                   text=_iv2123_bias_text([info[k]["bias_V"] for k in keys_of[run]]))
              for run in order if run in ends]
    swatches = [(_iv2123_hours_label(_iv2123_bias_text([info[k]["bias_V"] for k in keys_of[run]]),
                                     hours_of[run], fluence), color_of[run])
                for run in order if run in ends]
    return lines, groups, swatches, dict(stats=stats, excluded_min=excluded_min,
                                         runs=[r for r, _k in runs],
                                         hours_after_step={r: round(hours_of[r], 2) for r in order})


IV2123_YPAD_SWATCH_BASE = 2       # the default ypad already clears this many legend rows
IV2123_YPAD_PER_EXTRA_SWATCH = 0.12


def _iv2123_draw(ax, tel, fluence, ctx):
    lines, groups, swatches, rec = _iv2123_panel_data(tel, fluence, ctx["selection"], ctx["info"],
                                                       ctx["timeline"])
    extra_rows = max(0, len(swatches) - IV2123_YPAD_SWATCH_BASE)
    ypad = ivp.CURRENTS_YPAD + IV2123_YPAD_PER_EXTRA_SWATCH * extra_rows
    rec["label_info"] = ivp.draw_currents_panel(ax, tel, lines, groups=groups, swatches=swatches,
                                                ypad=ypad)
    rec["n_lines"] = len(lines)
    return rec


def _iv2123_build(stem, fluence, panel=None):
    selection, info, timeline, checked, excluded = _iv2123_select(fluence)
    ctx = dict(selection=selection, info=info, timeline=timeline)

    def draw(ax, tel):
        return _iv2123_draw(ax, tel, fluence, ctx)

    def values_fn(res_h, res_f):
        return dict(fluence_p_cm2=fluence, h1=res_h, f1=res_f,
                    selection={k: dict(chosen=v["chosen"], rejected=v["rejected"],
                                       not_good=[list(x) for x in v["not_good"]],
                                       dropped_boards=v["dropped_boards"])
                               for k, v in selection.items()},
                    assertion=dict(tol_pct=IV2123_ASSERT_TOL_PCT, checked=checked,
                                   excluded=excluded),
                    notes=IV2123_IH7_EXCURSION_NOTE)

    if panel:
        panels = {"h1": (lambda ax, c: _iv2123_draw(ax, "h1", fluence, ctx), IV2123_PANEL_SUBJECT),
                 "f1": (lambda ax, c: _iv2123_draw(ax, "f1", fluence, ctx), IV2123_PANEL_SUBJECT)}
        chosen = ts.resolve_panels(panel, panels)
        return ivp.export_panels_1x2(stem, OUT, chosen, line1=ts.HEADER_LINE_1)

    at_step = any(ts.fluence_key(fluence) == ts.fluence_key(f) for f in tsdata.JULY_STEPS)
    rfsel, offset = ((tsdata.JULY_STEPS_RFSEL, tsdata.JULY_STEPS_OFFSET) if at_step
                     else (tsdata.JULY_1P5E15_RFSEL, tsdata.JULY_1P5E15_OFFSET))
    footer = (IV2123_FOOTER % (tsdata.BIAS_TOL_V, rfsel, offset)
              + ("\n" + IV2123_PLATEAU_FOOTER if at_step else ""))
    inputs = [tsdata.JULY_TIMELINE, tsdata.JULY_REF_CSV, tsdata.GOOD_RUNS_CSV]
    inputs += [tsdata.DISPLAY_RUNS_JUL_CSV] if at_step else [tsdata.JULY_YAML]
    return ivp.compound_1x2(stem, OUT, draw, tag=IV2123_SUBJECT, footer_text=footer,
                            line1=ts.HEADER_LINE_1, values_fn=values_fn, script=NOTEBOOK,
                            inputs=inputs)


for _iv2123_stem, _iv2123_spec in IV2123_VARIANTS.items():
    _iv2123_build(_iv2123_stem, _iv2123_spec["fluence"])
    show(_iv2123_stem)
    if PANELS:
        flatten_panels(_iv2123_stem, _iv2123_build(_iv2123_stem, _iv2123_spec["fluence"],
                                                    panel=["all"]))

# %% [markdown]
# ### Figures 24 and 25: bias current vs run number
#
# Per-run bias current vs elapsed time in the run, one row of small panels per telescope: H1 in
# figure 24, F1 in figure 25. The runs (those in the `top` rows of
# `july/res_vs_run_combo_check_jul_values.json` in the inputs folder, a values file made outside
# this repository, see `../iv/campaigns/irrad_2026_inputs.md`) are drawn in ascending run number,
# each with its class, flag, fluence, threshold offset and RFSel from that file. The currents come
# from the July HV-monitor log (`july/timeline_60s.csv.gz`), the HV and LV cycle marks from
# `july/hv_cycles_jul.csv`. The figures are drawn by the
# package module `iv/current_vs_run.py` through its `build_one()` function, the same one its
# command line (`python -m etroc_plots.iv.current_vs_run`) runs per telescope; only the output name
# differs here. Its panels are per run rather than the H1/F1 panels of the other figures, so with
# `PANELS = True` this cell writes them through the module's own export.

# %%
from etroc_plots.iv import current_vs_run as cvr              # noqa: E402

ts.apply_style(1.0)
_iv2425_runs_by_tel, _iv2425_meta, _iv2425_combo_json = cvr.load_combo_meta()
_iv2425_run_win, _iv2425_board_meta = cvr.load_currents_meta()
_iv2425_timeline = cvr.load_timeline()
_iv2425_hv_cycles = cvr.load_hv_cycles()
cvr.OUT = OUT
cvr.PANEL_ROOT = os.path.join(OUT, "panels")

cvr.build_one("h1", _iv2425_runs_by_tel, _iv2425_run_win, _iv2425_board_meta, _iv2425_timeline,
             _iv2425_hv_cycles, _iv2425_meta, _iv2425_combo_json, write_panels=PANELS,
             stem="iv24_current_vs_run_h1")
show("iv24_current_vs_run_h1")

cvr.build_one("f1", _iv2425_runs_by_tel, _iv2425_run_win, _iv2425_board_meta, _iv2425_timeline,
             _iv2425_hv_cycles, _iv2425_meta, _iv2425_combo_json, write_panels=PANELS,
             stem="iv25_current_vs_run_f1")
show("iv25_current_vs_run_f1")

# %% [markdown]
# ## The full campaign
#
# ### Figures 26 and 27: IV across the campaign, full range (linear and log y)
#
# The whole campaign on one axis per telescope, H1 left / F1 right: pre-irradiation, the three
# March steps at +2 days each, the same 1.5e15 sensors again after the four-month rest, then the
# two July steps. Colour is the fluence ladder, except the second look at 1.5e15 (after the rest),
# which takes the after-the-rest teal so the before/after pair at that step reads as a pair. The
# March looks are the fine scans figures 1 and 2 draw, so the March half of this figure is
# identical to those rather than a second, differently aged set of curves. The July looks are the
# step's quick scans: 2e15 at +3 days (its only +2 days scan is the fine one figure 16 draws) and
# 3.5e15 at +4 days (the last full-range look before the IH12 current step; the +10 days scan stays
# in figures 17 and 18). Figure 26 is linear y, figure 27 is log y.

# %%
IV2627_XLIM = ivp.XLIM_FULL
IV2627_LOG_YLIM = (0.05, 5000.0)
IV2627_LEGEND_SCALE = 0.45          # seven looks: a smaller box than the four-look reference needs

IV2627_LOOKS = [
    dict(source="march", fluence=0.0, role="fine_ra", when=None),
    dict(source="march", fluence=3e14, role="fine_2d", when="+2 days"),
    dict(source="march", fluence=9e14, role="fine_2d", when="+2 days"),
    dict(source="march", fluence=1.5e15, role="fine_2d", when="+2 days"),
    IV08_AFTER,                                    # 1.5e15, +4 months: the look of figures 8/9
    dict(source="july", key="2e15_3d", fluence=2e15, when="+3 days"),
    dict(source="july", key="3p5e15_4d", fluence=3.5e15, when="+4 days"),
]

IV2627_FOOTER = ("one look per step: March steps at +2 days (fine scans, full range), the 1.5e15 "
                 "sensors again after the four-month rest, then the July steps (quick scans)")
IV28_FOOTER = IV2627_FOOTER + u"\n" + ivp.CHIP_DOSE_FOOTER


def _iv2627_lin_ylim(curves):
    return (0.0, 1.1 * max(float(c["i"].max()) for c in curves))


def _iv26_build(panel=None):
    return ivp.look_figure(
        "iv26_campaign", OUT, IV2627_LOOKS,
        tag="IV across the campaign, one curve per step (linear y)",
        footer=IV2627_FOOTER, xlim=IV2627_XLIM, yscale="linear", ylim=_iv2627_lin_ylim,
        legend_scale=IV2627_LEGEND_SCALE, gap_skip_below_V=80.0, script=NOTEBOOK,
        panel_subject="full range, linear y", panel=panel)


_iv26_build()
show("iv26_campaign")
if PANELS:
    flatten_panels("iv26_campaign", _iv26_build(panel=["all"]))


def _iv27_build(panel=None):
    return ivp.look_figure(
        "iv27_campaign_log", OUT, IV2627_LOOKS,
        tag="IV across the campaign, one curve per step (log y)",
        footer=IV2627_FOOTER, xlim=IV2627_XLIM, yscale="log", ylim=IV2627_LOG_YLIM,
        legend_scale=IV2627_LEGEND_SCALE, gap_skip_below_V=80.0, script=NOTEBOOK,
        panel_subject="full range, log y", panel=panel)


_iv27_build()
show("iv27_campaign_log")
if PANELS:
    flatten_panels("iv27_campaign_log", _iv27_build(panel=["all"]))

# %% [markdown]
# ### Figure 28: IV across the campaign, fluence on chip
#
# The same figure as 26, with each legend entry giving the fluence actually seen on chip
# (`ivp.CHIP_DOSE_FACTOR` times the IRRAD bookkeeping fluence) instead of the bookkeeping fluence.
# The x axis stays bias voltage; looks, colours and timing labels are those of figure 26.

# %%
def _iv28_build(panel=None):
    return ivp.look_figure(
        "iv28_campaign_chipdose", OUT, IV2627_LOOKS,
        tag="IV across the campaign, one curve per step (linear y, fluence on chip)",
        footer=IV28_FOOTER, xlim=IV2627_XLIM, yscale="linear", ylim=_iv2627_lin_ylim,
        legend_scale=IV2627_LEGEND_SCALE, gap_skip_below_V=80.0, script=NOTEBOOK,
        panel_subject="full range, linear y", chip_dose=True, panel=panel)


_iv28_build()
show("iv28_campaign_chipdose")
if PANELS:
    flatten_panels("iv28_campaign_chipdose", _iv28_build(panel=["all"]))

# %% [markdown]
# ### Figure 29: the low-V knee across the campaign
#
# The gain-layer knee at every irradiation step on one axis, fine scans only: the March +2 days
# fine scans (the same scans figure 3 draws), the 1.5e15 step again after the four-month rest (the
# 2026-07-16 legacy fine slow-control logs, the scans the after-the-rest V_gl values of
# `vgl_points.csv` were extracted from by the campaign IV notebooks, since the next day's fine scan
# reaches only 11-13 V), then the two July steps' fine scans. Pre-irradiation is not drawn here;
# its fine low-V scan is in figure 3.

# %%
IV29_LEGEND_SCALE = 0.45          # six looks: a smaller box than the four-look reference needs

IV29_LOOKS = [
    dict(source="march", fluence=3e14, role="fine_2d", when="+2 days"),
    dict(source="march", fluence=9e14, role="fine_2d", when="+2 days"),
    dict(source="march", fluence=1.5e15, role="fine_2d", when="+2 days"),
    dict(source="july_fine", key="4mo_1p5e15_fine", fluence=1.5e15, when="+4 months"),
    dict(source="july", key="2e15_2d_fine", fluence=2e15, when="+2 days"),
    dict(source="july", key="3p5e15_4d_fine", fluence=3.5e15, when="+4 days"),
]

IV29_FOOTER = ("fine scans only, 0.1 V bins, one marker every ~10 V; the after-the-rest look is "
              "the 2026-07-16 legacy fine scan; pre-irradiation is not drawn (its fine scan is "
              "in figure 3)")


def _iv29_build(panel=None):
    return ivp.look_figure(
        "iv29_campaign_lowv", OUT, IV29_LOOKS, tag="IV across the campaign, low-V knee",
        footer=IV29_FOOTER, xlim=ivp.LOWV_XLIM, yscale="linear", mask_V=ivp.LOWV_MASK_V,
        marker_step_V=ivp.LOWV_MARKER_STEP_V, point_scale=0.55, gap_skip_below_V=10.0,
        legend_scale=IV29_LEGEND_SCALE, script=NOTEBOOK, panel_subject="low-V knee", panel=panel)


_iv29_build()
show("iv29_campaign_lowv")
if PANELS:
    flatten_panels("iv29_campaign_lowv", _iv29_build(panel=["all"]))

# %% [markdown]
# ### Figure 30: IV after/before ratio, 1.5e15
#
# The same ratio, same scans and same code path as figure 10 (both read the BEFORE/AFTER looks
# defined under figures 8/9 through `ivp.ratio_figure`): this places that ratio next to the full
# campaign curves rather than next to the four-month-rest curves, so the two can never disagree
# about which scans they compare.

# %%
def _iv30_footer(skipped):
    base = ("after-the-rest / before ratio at the 1.5e15 step, common 1 V grid over each chip's "
           "scan overlap; H1's after-the-rest scan avoids an IH13 diagnostic-switch interval")
    if skipped:
        base += "; no ratio line (overlap < 50 V): %s" % ", ".join(skipped)
    return base


def _iv30_build(panel=None):
    return ivp.ratio_figure(
        "iv30_campaign_ratio", OUT, IV08_BEFORE, IV08_AFTER, IV0809_XLIM,
        tag="IV after / before the four-month rest, %s" % ts.fluence_label(IV08_BEFORE["fluence"]),
        footer=_iv30_footer, ylabel="I(after rest) / I(before rest)", script=NOTEBOOK, panel=panel)


_iv30_build()
show("iv30_campaign_ratio")
if PANELS:
    flatten_panels("iv30_campaign_ratio", _iv30_build(panel=["all"]))

# %% [markdown]
# ## V_gl across the full campaign
#
# ### Figures 31-33: campaign-wide V_gl vs fluence, on-chip fluence, and the ladder to 2e15
#
# Figure 31 is the V_gl-vs-fluence ladder over the whole campaign: every March and July step's
# right-after and after-cooling-down scans, plus the 1.5e15 four-month-rest point, out to 3.5e15.
# As in figure 7, the x axis is the bookkeeping fluence and the per-chip points are not joined by a
# line. Figure 32 is the same data with every fluence value, axis label and legend switched to
# fluence on chip (`ivp.CHIP_DOSE_FACTOR`) instead of bookkeeping fluence. Figure 33 stops the
# ladder at 2e15, with no 3.5e15 point.
#
# All three use the `VGL_VARIANTS` and `_vgl_draw` code defined under figure 7.

# %%
VGL_VARIANTS.update({
    "iv31_vgl_campaign": dict(fluence_max=3.7e15, include_4mo=True, data="full campaign"),
    "iv32_vgl_campaign_chipdose": dict(fluence_max=3.7e15, include_4mo=True, chip_dose=True,
                                       data="full campaign"),
    "iv33_vgl_to_2e15": dict(fluence_max=2.2e15, include_4mo=True, data="March 2026 to 2e15"),
})

vgl_figure("iv31_vgl_campaign")
show("iv31_vgl_campaign")
if PANELS:
    vgl_export_panels("iv31_vgl_campaign")

vgl_figure("iv32_vgl_campaign_chipdose")
show("iv32_vgl_campaign_chipdose")
if PANELS:
    vgl_export_panels("iv32_vgl_campaign_chipdose")

vgl_figure("iv33_vgl_to_2e15")
show("iv33_vgl_to_2e15")
if PANELS:
    vgl_export_panels("iv33_vgl_to_2e15")

# %% [markdown]
# ## V_gl fits and comparison with published measurements
#
# ### Figures 34-39: per-board V_gl fits, the published slopes, and H1 vs F1
#
# Figures 34 and 37 are the per-board V_gl-vs-fluence fits with their residuals, H1 left and F1
# right, linear and log y. Colour is the board, not the fluence: fluence is already the x axis, and
# the four boards' separate fits are the point. An exponential is not the only shape these points
# allow (the single-exponential fits' chi2/ndf, printed on figures 35, 36, 38 and 39, is far above
# 1), so the log view is drawn beside the linear one rather than instead of it: the linear view
# keeps the 0-55 V range of every V_gl plot and carries the fit table, the log view is the shape
# check. Open markers are the scan right after a step, filled are after cooling down (2 to 4
# days), half-filled is the 4-month-later look.
#
# Figures 35 and 38 compare the fitted slope to published removal-constant measurements: the
# average fit curve and every reference curve are forced through this campaign's own fitted V0, so
# only the slope is compared, and a slope strip under each panel restates the comparison as
# numbers, colour-matched to double as the curves' legend. Figures 36 and 39 put both telescopes
# side by side, V_gl normalised to each one's own fitted V0, and V_gl itself.
#
# Legend and fit-table positions are set per panel, because the two telescopes' data sit in
# different parts of the frame.

# %%
from etroc_plots.iv import vgl_plot as vglp, vgl, vgl_style           # noqa: E402
from etroc_plots.iv.campaigns import active as vglc                  # noqa: E402
from matplotlib.gridspec import GridSpec                             # noqa: E402

IV3439_TELS = [("h1", vglc.VGL_H1, (35.0, 0.4)), ("f1", vglc.VGL_F1, (45.0, 0.2))]
IV3439_DATA = "March + July 2026"

IV3439_PANEL = {
    "linear": {"h1": dict(legend_loc="upper right", legend_ncol=2, stats_top=0.755),
              "f1": dict(legend_loc="upper right", stats_top=0.40)},
    "log": {"h1": dict(stats_box=False),
           "f1": dict(stats_box=False, legend_loc="lower right")},
}
IV3439_LINE_KEYS = {"h1": (("-", "PS protons, 23-24 GeV"), ("-.", "reactor neutrons"),
                          (":", "if on-chip 1.00 / 0.50")),
                    "f1": (("-", "PS protons, 24 GeV"), ("-.", "reactor neutrons"),
                          ("--", "500 MeV protons"))}


def _iv3439_fits(tel, table, p0):
    return vglp.fit_boards(table, vglc.TEL_CHIPS[tel], p0)


def _iv3439_refs(tel):
    return vgl.reference_rows(vglc.VGL_VENDOR[tel], vglc.CHIP_HIT_FRACTION, vglc.CONV_FULL,
                              vglc.NIEL_24GEV, vglc.CHIP_HIT_REL_ERR, vglc.VGL_REF_ALTERNATES[tel])


def _iv3439_finish(fig, axes, stem, values, subjects=None):
    ts.compound_header(axes, line1=vglc.LINE1_PARKED, data=IV3439_DATA, pad=6, subjects=subjects)
    ts.lower_footer(fig)   # the footer's final band, so the audits see the saved layout
    problems = ts.check_no_clipping(fig, stem)
    legend_problems = ivp.check_legend_overlap(fig, tag=stem)
    box_problems = ivp.check_legend_boxes(fig, tag=stem)
    ts.save_figure(fig, OUT, stem, audit=False)
    values = dict(values, overlap_problems=problems, legend_overlap_problems=legend_problems,
                  legend_box_problems=box_problems)
    ivp.write_values(OUT, stem, values, script=NOTEBOOK,
                     inputs=[os.path.join(ivd.INPUTS_VGL, "vgl_points.csv")])
    plt.close(fig)


def _iv3439_build_fit(stem, yscale):
    ts.apply_style(1.0)
    fig = plt.figure(figsize=ivp.COMPOUND_FIGSIZE)
    adj = dict(ivp.COMPOUND_ADJUST, bottom=0.16)
    gs = GridSpec(2, 2, height_ratios=[3.0, 1.15], hspace=0.08,
                 left=adj["left"], right=adj["right"], top=adj["top"],
                 bottom=adj["bottom"], wspace=adj["wspace"])
    axes, values = [], {}
    for col, (tel, table, p0) in enumerate(IV3439_TELS):
        boards = vglc.TEL_CHIPS[tel]
        ax = fig.add_subplot(gs[0, col])
        axr = fig.add_subplot(gs[1, col], sharex=ax)
        values[tel] = vglp.draw_fit_panel(
            ax, axr, table, boards,
            colors=vgl_style.by_board(boards),
            xoff=vgl_style.by_board(boards, vgl_style.XOFFSETS),
            fits=_iv3439_fits(tel, table, p0), yscale=yscale, **IV3439_PANEL[yscale][tel])
        ts.panel_title(ax, tel)
        axes.append(ax)
    ts.footer(fig, vglp.FOOTER)
    _iv3439_finish(fig, axes, stem, values)


def _iv3439_build_compare(stem, yscale):
    ts.apply_style(1.0)
    fig = plt.figure(figsize=(ivp.COMPOUND_FIGSIZE[0], 12.0))
    gs = GridSpec(2, 2, height_ratios=[3.0, 2.3], hspace=0.45, left=0.075, right=0.985,
                 top=0.935, bottom=0.085, wspace=0.24)
    axes, values, chi2 = [], {}, {}
    for col, (tel, table, p0) in enumerate(IV3439_TELS):
        fits, refs = _iv3439_fits(tel, table, p0), _iv3439_refs(tel)
        ax = fig.add_subplot(gs[0, col])
        values[tel] = vglp.draw_compare_panel(ax, table, fits, refs, yscale=yscale,
                                              line_keys=IV3439_LINE_KEYS[tel])
        ts.panel_title(ax, tel)
        axc = fig.add_subplot(gs[1, col])
        pos = axc.get_position()
        axc.set_position([pos.x0 + 0.155, pos.y0, pos.width - 0.215, pos.height])
        vglp.draw_slope_strip(axc, fits, refs, tel.upper())
        chi2[tel] = values[tel]["chi2_ndf"]
        axes.append(ax)
    ts.footer(fig, (
        "curves forced through each telescope's own fitted V0: only the slope c is compared; "
        "published c x %.2f per p, x %.3f per n$_{eq}$; band = $\\pm$%d %% on-chip fluence or "
        "wafer spread\n"
        "c is an effective slope: single-exponential fits give $\\chi^2$/ndf = %.0f (H1), %.0f "
        "(F1) at %.1f V per point; open = right after, filled = after cooling down, half = "
        "4 months"
        % (vglc.CHIP_HIT_FRACTION, vglc.CONV_FULL, round(100 * vglc.CHIP_HIT_REL_ERR),
           chi2["h1"], chi2["f1"], vgl.SIGMA_V)))
    _iv3439_finish(fig, axes, stem, values)


def _iv3439_build_h1_vs_f1(stem, yscale):
    ts.apply_style(1.0)
    fig = plt.figure(figsize=ivp.COMPOUND_FIGSIZE)
    fig.subplots_adjust(**dict(ivp.COMPOUND_ADJUST, bottom=0.16))
    axn, axa = fig.subplots(1, 2)
    sets, chi2 = [], {}
    for tel, table, p0 in IV3439_TELS:
        fa = _iv3439_fits(tel, table, p0)["all"]["average"]
        phi, v = vgl.avg_series(table, (vglc.PROMPT,))
        label = "%s (%s)" % (tel.upper(), vglc.VGL_VENDOR[tel])
        sets.append(dict(tel=tel, label=label, phi=phi, v=v, V0=fa["V0"], c_1e16=fa["c_1e16"],
                         c_1e16_err=fa["c_1e16_err"]))
        chi2[tel] = fa["chi2_ndf_sigma0p2"]
    values = vglp.draw_h1_vs_f1(axn, axa, sets, yscale=yscale)
    ts.footer(fig, (
        "board-average scans right after each step (open markers) and their single-exponential "
        "fits; colour and marker = telescope; H1 and F1 sat back to back in the same box\n"
        "the chip-hit fraction and NIEL cancel in c(H1)/c(F1); c is an effective slope, "
        "$\\chi^2$/ndf = %.0f (H1), %.0f (F1) at %.1f V per point"
        % (chi2["h1"], chi2["f1"], vgl.SIGMA_V)))
    _iv3439_finish(fig, [axn, axa], stem, values,
                  subjects=[r"normalised to own fitted $V_0$", "in volts"])


# the campaign's V_gl tables must match vgl_points.csv, which figures 6, 7, 11, 12 and 31-33 draw
for _tel, _table, _p0 in IV3439_TELS:
    ivd.check_vgl_table(_tel, _table)

_iv3439_build_fit("iv34_vgl_fit_campaign_linear", "linear")
show("iv34_vgl_fit_campaign_linear")
_iv3439_build_compare("iv35_vgl_compare_linear", "linear")
show("iv35_vgl_compare_linear")
_iv3439_build_h1_vs_f1("iv36_vgl_h1_vs_f1_linear", "linear")
show("iv36_vgl_h1_vs_f1_linear")
_iv3439_build_fit("iv37_vgl_fit_campaign_log", "log")
show("iv37_vgl_fit_campaign_log")
_iv3439_build_compare("iv38_vgl_compare_log", "log")
show("iv38_vgl_compare_log")
_iv3439_build_h1_vs_f1("iv39_vgl_h1_vs_f1_log", "log")
show("iv39_vgl_h1_vs_f1_log")

# %% [markdown]
# ## HV-monitor current-limit holds
#
# ### Figure 40: the March week-1 spark test
#
# Before the week-1 HPK setup (boards IH18/IH19/IH21/IH22, 3e15 p/cm2) was swapped for the
# campaign's H1/F1 telescopes, its four channels sat pinned at the HV supply's 4 mA current limit
# for about an hour. A channel counts as held if its last unbroken stretch of 60 s medians at or
# above 3.9 mA lasts at least 15 minutes; the panels run from the earliest held channel's hold to
# the end of the latest one, and the notebook stops if any held channel's hold ends earlier, so no
# ramp-down is drawn. The raw log's timestamps are local time (CET) and are shifted back an hour,
# so the times in the values file are UTC like those of the other figures.

# %%
IV40_CHIPS = {0: "IH18", 1: "IH19", 2: "IH21", 3: "IH22"}
IV40_SLOT_MARKER = ("o", "s", "^", "D")
IV40_SLOT_LINESTYLE = ("-", "--", "-.", ":")
IV40_CHIP_MARKER = {chip: IV40_SLOT_MARKER[i] for i, chip in IV40_CHIPS.items()}
IV40_CHIP_LINESTYLE = {chip: IV40_SLOT_LINESTYLE[i] for i, chip in IV40_CHIPS.items()}
IV40_DAY_LO, IV40_DAY_HI = "2026-03-11 06:00:00", "2026-03-11 09:30:00"
IV40_CURRENT_LIMIT_UA = 4000.0
IV40_HOLD_DETECT_UA = 3900.0
IV40_HOLD_GAP_S = 90.0
IV40_HOLD_MIN_MIN = 15.0
IV40_MARKEVERY = 40
IV40_YLABEL_V = "Bias voltage [V]"
IV40_FOOTER = "HV monitor, 60 s medians, before the H1/F1 setup swap; ramp-down after hold not shown"


def _iv40_load():
    df = pd.read_csv(vglc.MARCH_SPARK_CSV)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %H:%M:%S.%f", errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df["tb"] = df["Date"] - pd.Timedelta(hours=vglc.PREIRRAD_LOG_UTC_OFFSET_H)   # local -> UTC
    df = df[(df["tb"] >= IV40_DAY_LO) & (df["tb"] <= IV40_DAY_HI)]
    return df


def _iv40_binned(df, ch):
    vcol, icol = "Re(Vmeas[%d]) [V]" % ch, "Re(Imeas[%d]) [A]" % ch
    sub = df[["tb", vcol, icol]].dropna()
    g = sub.set_index("tb").resample("60s").median(numeric_only=True).dropna().reset_index()
    g["v"] = g[vcol].abs()
    g["i_uA"] = g[icol].abs() * 1.0e6
    return g[["tb", "v", "i_uA"]]


def _iv40_detect(sub):
    sub = sub.reset_index(drop=True)
    mask = (sub["i_uA"] >= IV40_HOLD_DETECT_UA).to_numpy()
    if not mask.any():
        return None
    true_idx = [i for i, m in enumerate(mask) if m]
    last = true_idx[-1]
    start = last
    tb = sub["tb"]
    while (start - 1 >= 0 and mask[start - 1]
          and (tb.iloc[start] - tb.iloc[start - 1]).total_seconds() <= IV40_HOLD_GAP_S):
        start -= 1
    dur_min = (tb.iloc[last] - tb.iloc[start]).total_seconds() / 60.0
    return dict(start=tb.iloc[start], end=tb.iloc[last], dur_min=float(dur_min),
               v_min=float(sub["v"].iloc[start:last + 1].min()),
               v_start=float(sub["v"].iloc[start]), v_end=float(sub["v"].iloc[last]))


def _iv40_data():
    df = _iv40_load()
    binned = {chip: _iv40_binned(df, ch) for ch, chip in IV40_CHIPS.items()}
    holds = {chip: _iv40_detect(binned[chip]) for chip in IV40_CHIPS.values()}
    held_chips = [c for c in IV40_CHIPS.values()
                 if holds[c] and holds[c]["dur_min"] >= IV40_HOLD_MIN_MIN]
    if not held_chips:
        raise RuntimeError("figure 40: no channel of %s stays at or above %.0f uA for %.0f min "
                           "between %s and %s UTC; check the file and the IV40_* settings"
                           % (vglc.MARCH_SPARK_CSV, IV40_HOLD_DETECT_UA, IV40_HOLD_MIN_MIN,
                              IV40_DAY_LO, IV40_DAY_HI))
    t0 = min(holds[c]["start"] for c in held_chips)
    hold_end = max(holds[c]["end"] for c in held_chips)
    tsdata.check_holds_last_to(holds, held_chips, hold_end, "figure 40")
    series = {}
    for c, chip in IV40_CHIPS.items():
        g = binned[chip]
        g = g[(g["tb"] >= t0) & (g["tb"] <= hold_end)]
        elapsed = (g["tb"] - t0).dt.total_seconds() / 60.0
        series[chip] = dict(x=elapsed.to_numpy(), v=g["v"].to_numpy(), i=g["i_uA"].to_numpy(),
                            held=(chip in held_chips), color=ivp.TIMING_COLORS[c])
    return dict(t0=t0, hold_end=hold_end, held_chips=held_chips, holds=holds, series=series)


def _iv40_style(chip, s):
    if s["held"]:
        return dict(color=s["color"], marker=IV40_CHIP_MARKER[chip],
                   linestyle=IV40_CHIP_LINESTYLE[chip], linewidth=1.8, markersize=6.0)
    return dict(color="0.6", marker=IV40_CHIP_MARKER[chip], linestyle=IV40_CHIP_LINESTYLE[chip],
               linewidth=1.0, markersize=4.5, alpha=0.85)


def _iv40_draw_v(ax, data):
    for chip, s in data["series"].items():
        ax.plot(s["x"], s["v"], markevery=IV40_MARKEVERY, **_iv40_style(chip, s))
    vmax = max(float(s["v"].max()) for s in data["series"].values())
    ax.set_ylim(0.0, vmax * 1.12)
    ax.set_ylabel(IV40_YLABEL_V)
    ts.style_axes(ax)
    handles = [Line2D([], [], **{k: v for k, v in _iv40_style(c, s).items() if k != "alpha"},
                      label="%s (%s)" % (c, "held" if s["held"] else "not held"))
              for c, s in data["series"].items()]
    ax.legend(handles=handles, loc="lower left", fontsize=ts.sizes(1.0)["legend"], ncol=2,
             framealpha=0.9)


def _iv40_draw_i(ax, data):
    for chip, s in data["series"].items():
        ax.plot(s["x"], s["i"], markevery=IV40_MARKEVERY, **_iv40_style(chip, s))
    ax.axhline(IV40_CURRENT_LIMIT_UA, color=ts.INK, linestyle="--", linewidth=1.3, zorder=1)
    ax.set_ylim(0.0, IV40_CURRENT_LIMIT_UA * 1.10)
    xmax = max(float(s["x"].max()) for s in data["series"].values())
    ax.set_xlim(0.0, xmax * 1.02)
    ax.set_xlabel("Minutes since the start of the hold")
    ax.set_ylabel(ivp.YLABEL_I)
    ts.style_axes(ax)
    ax.annotate("4 mA limit", xy=(xmax * 1.02, IV40_CURRENT_LIMIT_UA), xytext=(-4, 5),
               textcoords="offset points", ha="right", va="bottom",
               fontsize=ts.sizes(1.0)["ann"], color=ts.INK, annotation_clip=False)


def _iv40_footer_text(data):
    held = data["held_chips"]
    durs = [data["holds"][c]["dur_min"] for c in held]
    vmins = [data["holds"][c]["v_min"] for c in held]
    vmin_chip = min(held, key=lambda c: data["holds"][c]["v_min"])
    return (IV40_FOOTER + u"  ·  %d chips, %.0f-%.0f min, to %.0f V (%s)"
           % (len(held), min(durs), max(durs), min(vmins), vmin_chip))


def _iv40_values(data):
    return dict(t0_utc=str(data["t0"]), held_chips=data["held_chips"],
               holds={c: dict(start_utc=str(h["start"]), end_utc=str(h["end"]),
                              dur_min=round(h["dur_min"], 1), v_start=round(h["v_start"], 2),
                              v_end=round(h["v_end"], 2), v_min=round(h["v_min"], 2))
                      for c, h in data["holds"].items() if h is not None})


def iv40_build():
    stem = "iv40_spark_march"
    data = _iv40_data()
    ts.apply_style(1.0)
    fig, (ax_v, ax_i) = plt.subplots(2, 1, figsize=(9.5, 11.5))
    fig.subplots_adjust(left=0.115, right=0.97, top=0.945, bottom=0.09, hspace=0.26)
    _iv40_draw_v(ax_v, data)
    ts.right_lines(ax_v, ["March week-1 HPK setup"], ts.sizes(1.0)["title"])
    _iv40_draw_i(ax_i, data)
    ts.compound_header([ax_v], line1=ivd.LINE1_PARKED, line2="", data="March 2026")
    ts.footer(fig, _iv40_footer_text(data), scale=0.75)

    ts.lower_footer(fig)   # the footer's final band, so the audits see the saved layout
    problems = ts.check_no_clipping(fig, stem)
    legend_problems = ivp.check_legend_overlap(fig, tag=stem)
    box_problems = ivp.check_legend_boxes(fig, tag=stem)
    ts.save_figure(fig, OUT, stem, audit=False)
    values = dict(**_iv40_values(data), overlap_problems=problems,
                 legend_overlap_problems=legend_problems, legend_box_problems=box_problems)
    ivp.write_values(OUT, stem, values, script=NOTEBOOK, inputs=[vglc.MARCH_SPARK_CSV])
    plt.close(fig)


def iv40_export_panels():
    stem = "iv40_spark_march"
    data = _iv40_data()
    ts.apply_style(1.0)
    fig, (ax_v, ax_i) = plt.subplots(2, 1, figsize=(8.4, 9.2))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.93, bottom=0.09, hspace=0.32)
    _iv40_draw_v(ax_v, data)
    ts.talk_header(ax_v, name="March week-1 HPK setup", line1=ivd.LINE1_PARKED, line2="",
                   tag="current-limit hold")
    _iv40_draw_i(ax_i, data)
    ts.check_no_clipping(fig, "%s panel h2" % stem)
    ivp.check_legend_overlap(fig, tag="%s panel h2" % stem)
    ivp.check_legend_boxes(fig, tag="%s panel h2" % stem)
    paths = ts.save_panel(fig, OUT, stem, 1, "h2")
    plt.close(fig)
    flatten_panels(stem, paths)


iv40_build()
show("iv40_spark_march")
if PANELS:
    iv40_export_panels()

# %% [markdown]
# ### Figure 41: the July campaign's end-of-campaign spark test
#
# In the last logged HV-monitor minutes of the July campaign, sensors on both telescopes sat
# pinned at the power supply's 4 mA current limit while the readback bias drooped. A chip counts
# as held if its last unbroken stretch of 60 s medians at or above 3.99 mA lasts at least 15
# minutes (figure 40 uses the same rule at 3.9 mA). Each telescope's panels (H1 left, F1 right)
# start at its earliest held chip's hold and run for a fixed 32 minutes; the notebook stops if a
# held chip's hold ends inside that window, so the ramp-down afterwards is never drawn.
# Chips that are not held are drawn too. The footer names each telescope's held chips, the range
# over them of drawn 60 s medians at the limit, and the lowest bias they reach.

# %%
IV41_CURRENT_LIMIT_UA = 4000.0
IV41_HOLD_DETECT_UA = 3990.0
IV41_HOLD_GAP_S = 90.0
IV41_HOLD_MIN_MIN = 15.0
IV41_MARKEVERY = 4
IV41_WINDOW_MAX_MIN = 32.0
IV41_YLABEL_V = "Bias voltage [V]"
IV41_FOOTER = ("HV monitor, 60 s medians; end-of-campaign current-limit hold; ramp-down after "
              "hold not shown")


def _iv41_detect(sub):
    sub = sub.reset_index(drop=True)
    mask = (sub["i_uA"] >= IV41_HOLD_DETECT_UA).to_numpy()
    if not mask.any():
        return None
    true_idx = [i for i, m in enumerate(mask) if m]
    last = true_idx[-1]
    start = last
    tb = sub["tb"]
    while (start - 1 >= 0 and mask[start - 1]
          and (tb.iloc[start] - tb.iloc[start - 1]).total_seconds() <= IV41_HOLD_GAP_S):
        start -= 1
    dur_min = (tb.iloc[last] - tb.iloc[start]).total_seconds() / 60.0
    return dict(start=tb.iloc[start], end=tb.iloc[last], dur_min=float(dur_min),
               v_min=float(sub["v"].iloc[start:last + 1].min()),
               v_start=float(sub["v"].iloc[start]), v_end=float(sub["v"].iloc[last]))


def _iv41_telescope_data(tel, timeline):
    chips = tsdata.CHIPS[tel]
    tsub = timeline[timeline["tel"] == tsdata.JULY_TEL[tel]]
    holds = {}
    for c, chip in enumerate(chips):
        sub = tsub[tsub["channel"] == c].sort_values("tb")
        holds[chip] = _iv41_detect(sub)
    held_chips = [c for c in chips if holds[c] and holds[c]["dur_min"] >= IV41_HOLD_MIN_MIN]
    if not held_chips:
        raise RuntimeError("figure 41, %s: no chip stays at or above %.0f uA for %.0f min in %s; "
                           "check the file and the IV41_* settings"
                           % (tel.upper(), IV41_HOLD_DETECT_UA, IV41_HOLD_MIN_MIN,
                              tsdata.JULY_TIMELINE))
    t0 = min(holds[c]["start"] for c in held_chips)
    window_end = t0 + pd.Timedelta(minutes=IV41_WINDOW_MAX_MIN)
    tsdata.check_holds_last_to(holds, held_chips, window_end, "figure 41, %s" % tel.upper())
    series = {}
    for c, chip in enumerate(chips):
        sub = tsub[(tsub["channel"] == c) & (tsub["tb"] >= t0)
                  & (tsub["tb"] <= window_end)].sort_values("tb")
        if sub.empty:
            continue
        elapsed = (sub["tb"] - t0).dt.total_seconds() / 60.0
        series[chip] = dict(x=elapsed.to_numpy(), v=sub["v"].to_numpy(), i=sub["i_uA"].to_numpy(),
                            color=ivp.TIMING_COLORS[c])
    # (60 s medians at the limit, 60 s medians drawn) per drawn chip
    at_limit = {chip: (int((s["i"] >= IV41_HOLD_DETECT_UA).sum()), len(s["i"]))
                for chip, s in series.items()}
    return dict(t0=t0, window_end=window_end, held_chips=held_chips, holds=holds, series=series,
                at_limit=at_limit)


def _iv41_handle(chip, color):
    return Line2D([], [], color=color, marker=ts.chip_marker(chip),
                  linestyle=ts.chip_linestyle(chip), linewidth=1.8, markersize=6.0, label=chip)


def _iv41_draw_v(ax, data):
    for chip, s in data["series"].items():
        kw = dict(color=s["color"], marker=ts.chip_marker(chip), linestyle=ts.chip_linestyle(chip),
                  linewidth=1.8, markersize=6.0)
        ax.plot(s["x"], s["v"], markevery=IV41_MARKEVERY, **kw)
    vmax = max(float(s["v"].max()) for s in data["series"].values())
    ax.set_ylim(0.0, vmax * 1.12)
    ax.set_ylabel(IV41_YLABEL_V)
    ts.style_axes(ax)
    handles = [_iv41_handle(c, data["series"][c]["color"]) for c in data["series"]]
    ax.legend(handles=handles, loc="lower left", fontsize=ts.sizes(1.0)["legend"], ncol=2,
             framealpha=0.9)


def _iv41_draw_i(ax, data):
    for chip, s in data["series"].items():
        kw = dict(color=s["color"], marker=ts.chip_marker(chip), linestyle=ts.chip_linestyle(chip),
                  linewidth=1.8, markersize=6.0)
        ax.plot(s["x"], s["i"], markevery=IV41_MARKEVERY, **kw)
    ax.axhline(IV41_CURRENT_LIMIT_UA, color=ts.INK, linestyle="--", linewidth=1.3, zorder=1)
    ax.set_ylim(0.0, IV41_CURRENT_LIMIT_UA * 1.10)
    xmax = max(float(s["x"].max()) for s in data["series"].values())
    ax.set_xlim(0.0, xmax * 1.02)
    ax.set_xlabel("Minutes since the start of the hold")
    ax.set_ylabel(ivp.YLABEL_I)
    ts.style_axes(ax)
    ax.annotate("4 mA limit", xy=(xmax * 1.02, IV41_CURRENT_LIMIT_UA), xytext=(-4, 5),
               textcoords="offset points", ha="right", va="bottom",
               fontsize=ts.sizes(1.0)["ann"], color=ts.INK, annotation_clip=False)


def _iv41_footer_text(tel_data):
    def span(xs):
        return "%d" % min(xs) if min(xs) == max(xs) else "%d-%d" % (min(xs), max(xs))

    parts = []
    for tel, data in tel_data.items():
        held = data["held_chips"]
        vmins = {c: float(data["series"][c]["v"].min()) for c in held}
        vmin_chip = min(vmins, key=vmins.get)
        n_at = [data["at_limit"][c][0] for c in held]
        n_drawn = [data["at_limit"][c][1] for c in held]
        parts.append(u"%s: %s held, at the limit in %s of the %s medians drawn, bias down to "
                     u"%.0f V%s" % (tel.upper(), ", ".join(held), span(n_at), span(n_drawn),
                                    vmins[vmin_chip],
                                    " (%s)" % vmin_chip if len(held) > 1 else ""))
    return IV41_FOOTER + u"\n" + u"  ·  ".join(parts)


def _iv41_values(tel_data):
    out = {}
    for tel, data in tel_data.items():
        out[tel] = dict(
            t0_utc=str(data["t0"]), window_end_utc=str(data["window_end"]),
            held_chips=data["held_chips"],
            medians_at_limit={c: dict(at_limit=n, drawn=m)
                              for c, (n, m) in data["at_limit"].items()},
            holds={c: dict(start_utc=str(h["start"]), end_utc=str(h["end"]),
                           dur_min=round(h["dur_min"], 1), v_start=round(h["v_start"], 2),
                           v_end=round(h["v_end"], 2), v_min=round(h["v_min"], 2))
                   for c, h in data["holds"].items() if h is not None})
    return out


def iv41_build():
    stem = "iv41_spark_july"
    timeline = tsdata.load_july_timeline()
    tels = ("h1", "f1")
    tel_data = {tel: _iv41_telescope_data(tel, timeline) for tel in tels}

    ts.apply_style(1.0)
    ncol = len(tels)
    fig, axes = plt.subplots(2, ncol, figsize=(7.6 * ncol, 10.2), squeeze=False)
    for j, tel in enumerate(tels):
        data = tel_data[tel]
        ax_v, ax_i = axes[0][j], axes[1][j]
        _iv41_draw_v(ax_v, data)
        ts.panel_title(ax_v, tel)
        _iv41_draw_i(ax_i, data)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.935, bottom=0.09, hspace=0.30,
                        wspace=0.24 if ncol > 1 else 0.0)
    ts.compound_header([axes[0][j] for j in range(ncol)], line1=ivd.LINE1_PARKED,
                       data="July 2026, campaign end")
    ts.footer(fig, _iv41_footer_text(tel_data), scale=0.78)

    ts.lower_footer(fig)   # the footer's final band, so the audits see the saved layout
    problems = ts.check_no_clipping(fig, stem)
    legend_problems = ivp.check_legend_overlap(fig, tag=stem)
    box_problems = ivp.check_legend_boxes(fig, tag=stem)
    ts.save_figure(fig, OUT, stem, audit=False)
    values = dict(tel=_iv41_values(tel_data), overlap_problems=problems,
                 legend_overlap_problems=legend_problems, legend_box_problems=box_problems)
    ivp.write_values(OUT, stem, values, script=NOTEBOOK, inputs=[tsdata.JULY_TIMELINE])
    plt.close(fig)


def iv41_export_panels():
    stem = "iv41_spark_july"
    timeline = tsdata.load_july_timeline()
    paths = []
    for index, tel in enumerate(("h1", "f1"), start=1):
        data = _iv41_telescope_data(tel, timeline)
        ts.apply_style(1.0)
        fig, (ax_v, ax_i) = plt.subplots(2, 1, figsize=(8.4, 9.2))
        fig.subplots_adjust(left=0.14, right=0.97, top=0.93, bottom=0.09, hspace=0.32)
        _iv41_draw_v(ax_v, data)
        ts.talk_header(ax_v, name=ts.TELESCOPE_TITLE[tel], line1=ivd.LINE1_PARKED,
                       tag="current-limit hold")
        _iv41_draw_i(ax_i, data)
        ts.check_no_clipping(fig, "%s panel %s" % (stem, tel))
        ivp.check_legend_overlap(fig, tag="%s panel %s" % (stem, tel))
        ivp.check_legend_boxes(fig, tag="%s panel %s" % (stem, tel))
        paths += ts.save_panel(fig, OUT, stem, index, tel)
        plt.close(fig)
    flatten_panels(stem, paths)


iv41_build()
show("iv41_spark_july")
if PANELS:
    iv41_export_panels()

# %% [markdown]
# ### Figure 42: the whole July campaign at a glance
#
# The same panel layout as figure 41 (bias-voltage row over current row, H1 left, F1 right),
# extended over the whole July campaign on a calendar-time x axis, to show that both telescopes
# sat at high bias for the whole month despite the logging interruptions. Each chip's line is cut
# wherever its own channel has a gap longer than five minutes between consecutive HV-monitor bins,
# so gaps are never bridged; the footer counts the logging sessions of the monitor as a whole (a
# session ends only where no channel logged for five minutes). The shaded bands mark the periods
# at 2e15 and at 3.5e15, from each step's logged radiation-stop timestamp to the next (or, for
# 3.5e15, to the end of the window); the interval before the first July stop, still at 1.5e15
# carried over from March, is left unshaded.

# %%
import matplotlib.dates as mdates                                    # noqa: E402
import matplotlib.patches as mpatches                                # noqa: E402

IV42_SESSION_GAP_S = 300.0
IV42_CURRENT_LIMIT_UA = 4000.0
IV42_YLABEL_V = "Bias voltage [V]"
IV42_BEAM_ALPHA = 0.16
IV42_BEAM_STEPS = (2e15, 3.5e15)


def _iv42_sessions(tb):
    tb = tb.sort_values().reset_index(drop=True)
    if len(tb) == 0:
        return []
    gap_s = tb.diff().dt.total_seconds()
    breaks = [i for i in range(1, len(tb)) if gap_s.iloc[i] > IV42_SESSION_GAP_S]
    bounds = [0] + breaks + [len(tb)]
    return [(tb.iloc[bounds[k]], tb.iloc[bounds[k + 1] - 1]) for k in range(len(bounds) - 1)]


def _iv42_beam_windows():
    stops = ivd.RAD_STOP_UTC
    t2 = pd.Timestamp(stops[2e15])
    t35 = pd.Timestamp(stops[3.5e15])
    return {2e15: (t2, t35), 3.5e15: (t35, None)}


def _iv42_telescope_data(tel, timeline):
    chips = tsdata.CHIPS[tel]
    tsub = timeline[timeline["tel"] == tsdata.JULY_TEL[tel]]
    sessions = _iv42_sessions(tsub["tb"].drop_duplicates())
    t_lo, t_hi = tsub["tb"].min(), tsub["tb"].max()
    series = {}
    for c, chip in enumerate(chips):
        sub_all = tsub[tsub["channel"] == c].sort_values("tb")
        if len(sub_all) == 0:
            raise RuntimeError("figure 42: no HV-monitor rows for %s (%s channel %d) in %s"
                               % (chip, tel.upper(), c, tsdata.JULY_TIMELINE))
        chip_sessions = _iv42_sessions(sub_all["tb"])
        segs = [sub_all[(sub_all["tb"] >= s0) & (sub_all["tb"] <= s1)]
                for (s0, s1) in chip_sessions]
        series[chip] = dict(segments=segs, sessions=chip_sessions, color=ivp.TIMING_COLORS[c])
    return dict(sessions=sessions, series=series, t_lo=t_lo, t_hi=t_hi)


def _iv42_handle(chip, color):
    return Line2D([], [], color=color, linestyle=ts.chip_linestyle(chip), linewidth=1.6,
                  label=chip)


def _iv42_beam_handles(windows):
    out = []
    for fl in IV42_BEAM_STEPS:
        s0, s1 = windows[fl]
        if s0 is None or s1 is None:
            continue
        out.append(mpatches.Patch(facecolor=ts.fluence_color(fl), edgecolor="none",
                                  alpha=IV42_BEAM_ALPHA * 2.2, label="at %s" % ts.fluence_label(fl)))
    return out


def _iv42_shade_beam(ax, windows):
    for fl in IV42_BEAM_STEPS:
        s0, s1 = windows[fl]
        if s0 is None or s1 is None:
            continue
        ax.axvspan(s0, s1, color=ts.fluence_color(fl), alpha=IV42_BEAM_ALPHA, zorder=0, linewidth=0)


def _iv42_draw_v(ax, data, windows):
    _iv42_shade_beam(ax, windows)
    for chip, s in data["series"].items():
        for seg in s["segments"]:
            ax.plot(seg["tb"], seg["v"], color=s["color"], linestyle=ts.chip_linestyle(chip),
                    linewidth=1.1)
    vmax = max(float(seg["v"].max()) for s in data["series"].values() for seg in s["segments"])
    ax.set_ylim(0.0, vmax * 1.12)
    ax.set_ylabel(IV42_YLABEL_V)
    ts.style_axes(ax)
    handles = [_iv42_handle(c, data["series"][c]["color"]) for c in data["series"]]
    handles += _iv42_beam_handles(windows)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3,
             fontsize=max(ts.sizes(1.0)["legend"], 12), framealpha=0.9, borderaxespad=0.0)


def _iv42_draw_i(ax, data, windows, t_hi):
    _iv42_shade_beam(ax, windows)
    for chip, s in data["series"].items():
        for seg in s["segments"]:
            ax.plot(seg["tb"], seg["i_uA"], color=s["color"], linestyle=ts.chip_linestyle(chip),
                    linewidth=1.1)
    ax.axhline(IV42_CURRENT_LIMIT_UA, color=ts.INK, linestyle="--", linewidth=1.0, zorder=1)
    ax.set_ylim(0.0, IV42_CURRENT_LIMIT_UA * 1.10)
    ax.annotate("4 mA limit", xy=(t_hi, IV42_CURRENT_LIMIT_UA), xytext=(-4, 5),
               textcoords="offset points", ha="right", va="bottom",
               fontsize=ts.sizes(1.0)["ann"], color=ts.INK, annotation_clip=False)
    ax.set_xlabel("Calendar time (UTC)")
    ax.set_ylabel(ivp.YLABEL_I)
    ts.style_axes(ax)


def _iv42_apply_date_axis(ax, t_lo, t_hi, labels=True):
    pad = (t_hi - t_lo) * 0.02
    ax.set_xlim(t_lo - pad, t_hi + pad)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    if labels:
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    else:
        ax.tick_params(labelbottom=False)


def _iv42_footer_text(n_sessions, windows):
    parts = ["HV monitor, 60 s medians", "%d H1 + %d F1 sessions (gaps not interpolated)"
            % (n_sessions["h1"], n_sessions["f1"])]
    if windows[2e15][0] is not None or windows[3.5e15][0] is not None:
        parts.append("shading = period at the labelled fluence between logged radiation-stop "
                     "stamps, not a measured beam-on window")
    return "  ·  ".join(parts)


def _iv42_session_list(sessions):
    return [dict(start_utc=str(s0), end_utc=str(s1),
                 dur_min=round((s1 - s0).total_seconds() / 60.0, 1)) for s0, s1 in sessions]


def _iv42_session_values(data):
    return dict(n_sessions=len(data["sessions"]),
               t_lo_utc=str(data["t_lo"]), t_hi_utc=str(data["t_hi"]),
               sessions=_iv42_session_list(data["sessions"]),
               chip_sessions={chip: _iv42_session_list(s["sessions"])
                              for chip, s in data["series"].items()})


def iv42_build():
    stem = "iv42_spark_july_full"
    timeline = tsdata.load_july_timeline()
    tel_data = {tel: _iv42_telescope_data(tel, timeline) for tel in ("h1", "f1")}
    t_lo = min(d["t_lo"] for d in tel_data.values())
    t_hi = max(d["t_hi"] for d in tel_data.values())
    windows = _iv42_beam_windows()
    windows[3.5e15] = (windows[3.5e15][0], t_hi)
    n_sessions = {tel: len(d["sessions"]) for tel, d in tel_data.items()}

    ts.apply_style(1.0)
    tels = ["h1", "f1"]
    fig, axes = plt.subplots(2, 2, figsize=(7.6 * 2, 10.2), squeeze=False)
    for j, tel in enumerate(tels):
        data = tel_data[tel]
        ax_v, ax_i = axes[0][j], axes[1][j]
        _iv42_draw_v(ax_v, data, windows)
        ts.panel_title(ax_v, tel)
        _iv42_draw_i(ax_i, data, windows, t_hi)
        _iv42_apply_date_axis(ax_v, t_lo, t_hi, labels=False)
        _iv42_apply_date_axis(ax_i, t_lo, t_hi)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.935, bottom=0.115, hspace=0.46,
                        wspace=0.24)
    ts.compound_header([axes[0][j] for j in range(2)], line1=ts.HEADER_LINE_1,
                       data="July 2026, full campaign")
    ts.footer(fig, _iv42_footer_text(n_sessions, windows), scale=0.78)

    ts.lower_footer(fig)   # the footer's final band, so the audits see the saved layout
    problems = ts.check_no_clipping(fig, stem)
    legend_problems = ivp.check_legend_overlap(fig, tag=stem)
    box_problems = ivp.check_legend_boxes(fig, tag=stem)
    ts.save_figure(fig, OUT, stem, audit=False)
    values = dict(tel={tel: _iv42_session_values(tel_data[tel]) for tel in tels},
                 session_gap_threshold_s=IV42_SESSION_GAP_S,
                 beam_windows={("%g" % fl): dict(start_utc=str(windows[fl][0]),
                                             end_utc=str(windows[fl][1]))
                              for fl in IV42_BEAM_STEPS if windows[fl][0] is not None},
                 overlap_problems=problems, legend_overlap_problems=legend_problems,
                 legend_box_problems=box_problems)
    ivp.write_values(OUT, stem, values, script=NOTEBOOK, inputs=[tsdata.JULY_TIMELINE])
    plt.close(fig)


def iv42_export_panels():
    stem = "iv42_spark_july_full"
    timeline = tsdata.load_july_timeline()
    tel_data = {tel: _iv42_telescope_data(tel, timeline) for tel in ("h1", "f1")}
    t_lo = min(d["t_lo"] for d in tel_data.values())
    t_hi = max(d["t_hi"] for d in tel_data.values())
    windows = _iv42_beam_windows()
    windows[3.5e15] = (windows[3.5e15][0], t_hi)
    paths = []
    for index, tel in enumerate(("h1", "f1"), start=1):
        data = tel_data[tel]
        ts.apply_style(1.0)
        fig, (ax_v, ax_i) = plt.subplots(2, 1, figsize=(8.4, 9.2))
        fig.subplots_adjust(left=0.14, right=0.97, top=0.93, bottom=0.13, hspace=0.46)
        _iv42_draw_v(ax_v, data, windows)
        ts.talk_header(ax_v, name=ts.TELESCOPE_TITLE[tel], line1=ts.HEADER_LINE_1,
                       tag="full July campaign")
        _iv42_draw_i(ax_i, data, windows, t_hi)
        _iv42_apply_date_axis(ax_v, t_lo, t_hi, labels=False)
        _iv42_apply_date_axis(ax_i, t_lo, t_hi)
        ts.check_no_clipping(fig, "%s panel %s" % (stem, tel))
        ivp.check_legend_overlap(fig, tag="%s panel %s" % (stem, tel))
        ivp.check_legend_boxes(fig, tag="%s panel %s" % (stem, tel))
        paths += ts.save_panel(fig, OUT, stem, index, tel)
        plt.close(fig)
    flatten_panels(stem, paths)


iv42_build()
show("iv42_spark_july_full")
if PANELS:
    iv42_export_panels()
