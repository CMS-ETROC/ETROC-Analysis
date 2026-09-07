# %% [markdown]
# # Four bare dies on one bench: ETROC2 **v2.01** and **v2.03** charge injection, FNAL, July 2024
#
# **Chips** `ET2p03_BARE1`, `ET2p03_BARE4`, `ET2p03_BARE9` (three ETROC2 **version 2.03** bare dies)
# and `ET2p01_BARE10` (one **version 2.01** bare die). No LGAD is bonded to any of them.
# **Campaign** the FNAL bench, seven days between 2024-07-15 and 2024-07-30, one die per day.
# **Pixels** 6: (row, col) ∈ {(0,0), (0,14), (3,0), (3,14), (12,0), (12,14)} — the same six as the
# August-2024 X-ray campaign of notebook 03.
# **Charges** 5, 6, 8, 15, 22 and 27 fC — the same six as the two January-2024 campaigns of
# notebooks 01 and 02.
#
# "Gain" in this notebook always means the **preamp gain**: the charge-to-threshold response of
# the front end, in threshold-DAC LSB per fC. No sensor is involved anywhere here.
#
# This is the fourth measurement notebook of the folder, and it exists for three reasons the
# census (notebook 00, row 5) already named:
#
# 1. **It is the only place where a v2.01 die and v2.03 dies were scanned on the same bench,
#    days apart, on the same pixels and charges.** Notebook 04 compares the two revisions across
#    two campaigns, two sites and two temperatures; here everything but the die is held fixed.
# 2. **`BARE9`, the X-ray die of notebook 03, was scanned here on 29 and 30 July — four days
#    before its first day at CERN.** That is a pre-irradiation reference in an independent setup.
# 3. **`BARE4` is the die whose internal voltage reference notebook 00 measured in the climate
#    chamber** (section 7.6 there). Its three bench days are the only charge injection that die
#    ever took.
#
# It imports the same `qinj_common.py` and therefore the same turn-off estimator, weighted gain
# fit, shared-σ S-curve fit and quantisation floor as notebooks 01–03. One thing this campaign
# turned up applies retroactively to all three of them, and it has its own section (5): the
# pixel (0,14) has a **compressive charge response on every one of six dies of both revisions** —
# its turn-off saturates around 80–100 DAC above baseline while the other pixels reach 200–300 —
# so it is not a representative pixel, and the pooled gains of notebooks 01–03 are pulled low by
# its inclusion. Its cause is not resolved here.
#
# ## What this campaign did not record
#
# No dose (none was given), no rail setting, no temperature, no elapsed time — the directory names
# carry only chip, pixel, charge and threshold — and **no register read-back exists anywhere in
# the tree**, so the analog-rail setting and every front-end register (RfSel, IBSel, QSel …) are
# unknown for all seven days. Every absolute DAC number below is therefore *this die, on this day,
# at whatever the bench was set to*; comparisons within a day and slopes are what survive that.

# %%
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
import qinj_common as Q
from qinj_style import apply_style

apply_style(0.55)
pd.set_option("display.width", 200, "display.max_columns", 80)

LEDGER = []
RESULTS = {}

print("DAC LSB (assumed)    %.6f mV" % Q.LSB_MV)
print("TDC reference clock  %.3f ns" % Q.CLK_NS)
print("electrons per fC     %.3f" % Q.E_PER_FC)
print("operating point      threshold = BL + %d DAC" % Q.OP_OFFSET)
print("charges injected     %s fC" % Q.JUL_CHARGES)
print("pixels scanned       %s" % Q.JUL_PIXELS)
print("gain-fit range       %d-%d fC (notebooks 01 and 02)" % (Q.JUL_QLO, Q.JUL_QHI))

# %% [markdown]
# ## 1. Provenance
#
# Everything read here is under `data/bench_jul2024/`. It is the reduction of one raw tree,
# `fermilab-etroc3:/run/media/daq/X9/et2p03_qinj_data/2024-07-*_Array_Test_Results/` (48 659
# files, 38.8 GB), copied to LPC and to LPC EOS
# (`/eos/uscms/store/user/lpcmtdstudies/fnal_Jul2024_bench/raw/`) with file count, byte sum and a
# 146-file md5 sample verified at each hop. `DECODE_README.md` is the reduction's own
# documentation, shipped verbatim; `reduce_bench.py` is the reduction script, which is
# notebook 03's `reduce_tid.py` with a flat directory walk and every per-point function copied
# unchanged. The decoder it calls is this repository's `DecodeBinary`, byte for byte.
#
# | file | RAW / DERIVED | what it is |
# |---|---|---|
# | `jul_et2p03_points.feather` | RAW + DERIVED | per-point TOA/TOT/CAL moments, all-hit and CAL-windowed, one row per decoded point directory, 22 756 × 42 |
# | `jul_et2p03_turnoff.csv` | RAW | the 252 discriminator binary searches, whole search as `search_trace` |
# | `jul_et2p03_noiseonly.csv` | RAW | the 42 `quietCB` noise scans, one per day and pixel |
# | `jul_et2p03_configs.csv` | RAW + DERIVED | one row per bench day |
# | `jul_et2p03_blnw.csv` | RAW | the single autocalibration map of the campaign, `BARE4` on 07-17, 256 pixels |
#
# Unlike notebook 03 there is no independent hand-decode cross-check for this set; the July
# `DECODE_README.md` says so. The ledger in section 12 checks the tables against each other
# and against the README's own counts instead.

# %%
prov = pd.DataFrame([Q.file_row(p) for p in sorted(Q.JUL_DIR.iterdir()) if p.is_file()])
prov["MB"] = (prov.bytes / 1e6).round(3)
print(prov[["file", "bytes", "MB", "md5"]].to_string(index=False))
print("\nthe repository's own decoder, TestBeam/condor_at_lxplus/core/decoding.py, md5 289ef6cb7b7e48589e02d817de5060ed")
print("staged ../bare_et2p03_tid_aug2024/code/decoding.py (the copy reduce_bench.py imports)   md5",
      Q.md5(Q.ET2P03_DIR / "code" / "decoding.py"))

# %%
PTS = Q.load_jul_points()
TOF = Q.load_jul_turnoff()
NOI = Q.load_jul_noiseonly()
CFG = Q.load_jul_configs()
MAP = Q.load_jul_blnw()
ACQS = list(CFG.acq)            # the seven acquisitions, in date order
DIE_OF = dict(zip(CFG.acq, CFG.chip))
print("acquisitions:", ACQS)

# %% [markdown]
# ## 2. The campaign: seven days, four dies
#
# One die per day; `BARE4` for three consecutive days and `BARE9` for two. The first five days
# were written as `.nem` text, the last two as `.bin` — the same two formats notebook 03 reads
# with the same two code paths. Two of the seven days are flawed, and the table says how:
#
# * **07-18 (`BARE4`): the noise-only scans are unusable** — every one of the six reports its peak
#   at DAC 1, i.e. the counter never rose — and pixel (3,0)'s binary search sat at the DAC ceiling
#   (1019) for every charge up to 15 fC. The other five pixels' searches on that day are fine and
#   reproduce 07-16 and 07-17 to a DAC; they are kept as a third repeat of `BARE4`, but nothing
#   that needs a baseline is taken from 07-18.
# * **07-17 (`BARE4`): a short day** — 1 101 threshold points against 3 200–4 100 on the others, so its
#   money-plot coverage is thin (one to three pixels per threshold offset); its turn-offs and
#   noise scans are complete.
#
# `BARE9` on 07-29 and 07-30 sits at two very different absolute DAC ranges (344–610 against
# 585–846). Section 7 shows that the whole difference is a baseline offset at unchanged slope.

# %%
day_rows = []
for acq in ACQS:
    p = PTS[PTS.acq == acq]
    t = TOF[TOF.acq == acq]
    n = NOI[NOI.acq == acq]
    day_rows.append(dict(
        acquisition=acq, chip=DIE_OF[acq], format=p.source_format.iloc[0],
        n_points=len(p), n_turnoff=len(t), n_noise=len(n),
        n_point_dirs_README=int(CFG[CFG.acq == acq].n_point_dirs.iloc[0]),
        dac_min=int(p.dac.min()), dac_max=int(p.dac.max()),
        hits=int(p.n_hits.sum()),
        n_at_ceiling=int((t.turnoff_dac >= 1019).sum()),
        noise_peak_ok=int((n.dac_at_triggerbits_max > 100).sum()),
    ))
DAYS = pd.DataFrame(day_rows)
DAYS["dirs_accounted"] = DAYS.n_points + DAYS.n_turnoff + DAYS.n_noise
print(DAYS.to_string(index=False))
print("\ntotal points %d, turn-off searches %d, noise scans %d, hits %d"
      % (DAYS.n_points.sum(), DAYS.n_turnoff.sum(), DAYS.n_noise.sum(), DAYS.hits.sum()))
for _, r in DAYS.iterrows():
    Q.check("point directories accounted for, %s" % r.acquisition, r.dirs_accounted,
            r.n_point_dirs_README, 0, "dirs", LEDGER)
Q.check("decoded points, all days", DAYS.n_points.sum(), 22756, 0, "points", LEDGER)
Q.check("turn-off searches, all days", DAYS.n_turnoff.sum(), 252, 0, "searches", LEDGER)
Q.check("noise scans, all days", DAYS.n_noise.sum(), 42, 0, "scans", LEDGER)
Q.check("decode errors", int((PTS.decode_error.fillna("") != "").sum()), 0, 0, "points", LEDGER)
DAYS.to_csv(Q.OUTDIR / "jul_days.csv", index=False)

# %% [markdown]
# ## 3. Baseline and noise width, measured inside every day
#
# As in notebook 03, the `quietCB` noise scan of each pixel gives the **baseline** (the DAC at
# the top of the noise peak) and a **noise width** (its FWHM), both taken in the same
# configuration as the threshold scans. 07-18 has no usable scan and is dropped from everything
# baseline-referred.
#
# Two things are visible before any fit. The v2.01 die sits ~225 DAC below three of the v2.03
# acquisitions — but `BARE9` on 07-29 sits ~240 DAC below itself on 07-30, and *below* the v2.01
# die. A reference-voltage difference between the revisions (1.0 V external against ~0.89 V
# internal, section 6) would account for at most ~65 of the 225 DAC if the DAC scale followed
# it, so the absolute baselines here are dominated by the same unattributed offset that
# section 7 finds within one die, and they are not read as a revision property. The noise FWHM is
# 2–3 DAC on every die, with column 14 consistently wider than column 0 — which matters for the
# control used in section 5.

# %%
NOI["fwhm"] = [Q.noise_peak(t)[2] for t in NOI.scan_trace]
NOI["BL"] = NOI.dac_at_triggerbits_max.astype(float)
NOI.loc[NOI.BL < 100, ["BL", "fwhm"]] = np.nan          # 07-18: the counter never rose
BL = NOI.set_index(["acq", "row", "col"]).BL
print("baseline (noise-peak DAC), per acquisition and pixel")
print(NOI.pivot_table(index="acq", columns=["row", "col"], values="BL").reindex(ACQS).to_string())
print("\nnoise FWHM [DAC]")
print(NOI.pivot_table(index="acq", columns=["row", "col"], values="fwhm").reindex(ACQS).round(2).to_string())
BASE = NOI.groupby("acq").agg(chip=("chip", "first"), BL_mean=("BL", "mean"), BL_rms=("BL", "std"),
                              fwhm_mean=("fwhm", "mean"), fwhm_rms=("fwhm", "std"), n=("BL", "count")).reindex(ACQS)
print("\n" + BASE.round(2).to_string())
BASE.reset_index().to_csv(Q.OUTDIR / "jul_baselines.csv", index=False)
NOI[["acq", "chip", "date", "row", "col", "BL", "fwhm"]].to_csv(Q.OUTDIR / "jul_baseline_by_pixel.csv", index=False)

# %% [markdown]
# The one autocalibration map of the campaign, `BARE4` on 07-17, agrees with the noise scans of
# the same day on the six scanned pixels and adds the other 250. Its baseline rises smoothly by
# ~20 DAC from column 0 to column 15 and from row 0 to row 15, while its noise width is flat
# across the array. Both facts are used in section 5.

# %%
m6 = MAP.merge(NOI[NOI.acq == "v2.03_BARE4 07-17"][["row", "col", "BL", "fwhm"]], on=["row", "col"])
print("BARE4 07-17: autocalibration map against the noise scan, six pixels")
print(m6[["row", "col", "bl_dac", "BL", "nw_dac", "fwhm"]].rename(
    columns={"bl_dac": "map_BL", "BL": "scan_BL", "nw_dac": "map_NW", "fwhm": "scan_FWHM"}).to_string(index=False))
Q.check("BARE4 map vs scan baseline, mean |diff| over six pixels", float((m6.bl_dac - m6.BL).abs().mean()), 0, 3, "DAC", LEDGER)
print("\nmap baseline by row   ", MAP.groupby("row").bl_dac.mean().round(1).values)
print("map baseline by column", MAP.groupby("col").bl_dac.mean().round(1).values)
print("map noise width by row   ", MAP.groupby("row").nw_dac.mean().round(2).values)
print("map noise width by column", MAP.groupby("col").nw_dac.mean().round(2).values)

# %%
fig, axs = plt.subplots(1, 3, figsize=(22, 6.2))
ax = axs[0]
for i, acq in enumerate([a for a in ACQS if a != "v2.03_BARE4 07-18"]):
    g = NOI[NOI.acq == acq]
    ax.plot(range(6), [g[(g.row == r) & (g.col == c)].BL.iloc[0] for r, c in Q.JUL_PIXELS],
            "-o", lw=1.8, ms=7, label=acq, color=plt.cm.tab10(i))
ax.set_xticks(range(6))
ax.set_xticklabels(["(%d,%d)" % p for p in Q.JUL_PIXELS])
ax.set_xlabel("pixel (row, col)")
ax.set_ylabel("baseline, noise-peak DAC")
ax.legend(fontsize=10)
ax.set_title("baseline per pixel", loc="left", fontsize=13)
ax = axs[1]
for i, acq in enumerate([a for a in ACQS if a != "v2.03_BARE4 07-18"]):
    g = NOI[NOI.acq == acq]
    ax.plot(range(6), [g[(g.row == r) & (g.col == c)].fwhm.iloc[0] for r, c in Q.JUL_PIXELS],
            "-o", lw=1.8, ms=7, color=plt.cm.tab10(i))
ax.set_xticks(range(6))
ax.set_xticklabels(["(%d,%d)" % p for p in Q.JUL_PIXELS])
ax.set_xlabel("pixel (row, col)")
ax.set_ylabel("noise FWHM  [DAC]")
ax.set_ylim(0, 5)
ax.set_title("noise width per pixel", loc="left", fontsize=13)
ax = axs[2]
im = ax.imshow(MAP.pivot(index="row", columns="col", values="bl_dac").values, origin="lower", cmap="viridis")
ax.set_xlabel("column")
ax.set_ylabel("row")
ax.set_title("BARE4 07-17 autocalibration baseline, 256 pixels", loc="left", fontsize=13)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="baseline  [DAC]")
fig.tight_layout()
Q.fig_header(fig, "Baseline and noise width on four dies, and one 256-pixel map",
             tag="noise-peak baseline and FWHM from the quietCB scans; 07-18 excluded (no usable scan)",
             scale=0.62, campaign=Q.JUL_CAMPAIGN)
Q.save_fig(fig, "jul_fig1_baselines")
plt.show()

# %% [markdown]
# ## 4. Turn-off against charge, and the preamp gain of every pixel on every die
#
# The turn-off is the discriminator binary search's own converged DAC (RAW), referred to the
# noise-peak baseline of the same day and pixel. The gain is `Q.gain_fit` over 8–27 fC, the
# range and estimator of notebooks 01 and 02. Where 07-18 has no baseline the fit is still made
# (the slope does not need one) and only the intercept is left undefined.
#
# One thing the fit table says and the earlier notebooks did not dwell on: with the 0.5 DAC
# y-error the straight line is rejected on every pixel of every acquisition (χ²/ndf 10–250 on
# two degrees of freedom). The response is visibly convex over 8–27 fC — the 15–27 fC chord is
# steeper than the 8–15 fC one by 10–20 % — so every "gain" in this folder is a **fixed-range
# chord slope**, comparable between data sets only when the range is the same, and the
# per-pixel fit errors in `jul_gain_fits.csv` are understated by √(χ²/ndf). The table carries
# both the nominal error and the scaled one; section 7 is where the range matters.

# %%
TO = TOF.merge(NOI[["acq", "row", "col", "BL"]], on=["acq", "row", "col"], how="left")
TO["turnoff_offset"] = TO.turnoff_dac - TO.BL
TO["at_ceiling"] = TO.turnoff_dac >= 1019
print("turn-off above baseline [DAC], per acquisition, pixel and charge (07-18 has no baseline)")
print(TO.pivot_table(index=["acq", "charge_fC"], columns=["row", "col"], values="turnoff_offset").round(0).to_string())

gf = []
for (acq, r, c), g in TO.groupby(["acq", "row", "col"]):
    g = g[~g.at_ceiling]
    bl = float(g.BL.iloc[0]) if g.BL.notna().any() else np.nan
    f = Q.gain_fit(g, bl if np.isfinite(bl) else 0.0, q_lo=Q.JUL_QLO, q_hi=Q.JUL_QHI,
                   turnoff_col="turnoff_dac", charge_col="charge_fC")
    if not f:
        continue
    if not np.isfinite(bl):
        f["intercept_offset_dac"] = np.nan
        f["Qmin_at_off_fC"] = np.nan
    gf.append(dict(acq=acq, chip=DIE_OF[acq], row=r, col=c, BL=bl, **f))
GF = pd.DataFrame(gf)
GF["gain_err_scaled"] = GF.gain_dac_per_fc_err * np.sqrt(GF.chi2_ndf.clip(lower=1))
# the two chords, to make the convexity a number
ch = []
for (acq, r, c), g in TO.groupby(["acq", "row", "col"]):
    t = g.set_index("charge_fC").turnoff_dac
    if all(q in t.index for q in (8, 15, 27)) and not g.at_ceiling.any():
        ch.append(dict(acq=acq, row=r, col=c, chord_8_15=(t[15] - t[8]) / 7.0, chord_15_27=(t[27] - t[15]) / 12.0))
CH = pd.DataFrame(ch)
CH["convexity"] = CH.chord_15_27 / CH.chord_8_15
GF = GF.merge(CH, on=["acq", "row", "col"], how="left")
GF.to_csv(Q.OUTDIR / "jul_gain_fits.csv", index=False)
print("15-27 fC chord over 8-15 fC chord, the five pixels other than (0,14): median %.2f, range %.2f-%.2f"
      % (CH[~((CH.row == 0) & (CH.col == 14))].convexity.median(),
         CH[~((CH.row == 0) & (CH.col == 14))].convexity.min(), CH[~((CH.row == 0) & (CH.col == 14))].convexity.max()))
print("the same at (0,14): median %.2f" % CH[(CH.row == 0) & (CH.col == 14)].convexity.median())
print("fit error scaled by sqrt(chi2/ndf), median factor %.1f" % np.sqrt(GF.chi2_ndf.clip(lower=1)).median())
print("\npreamp gain [DAC/fC], 8-27 fC, per acquisition and pixel")
print(GF.pivot_table(index="acq", columns=["row", "col"], values="gain_dac_per_fc").reindex(ACQS).round(2).to_string())
print("\nchi2/ndf of the straight line")
print(GF.pivot_table(index="acq", columns=["row", "col"], values="chi2_ndf").reindex(ACQS).round(1).to_string())

# %%
fig, axs = plt.subplots(2, 4, figsize=(24, 11), sharey=True)
PANELS = [a for a in ACQS]
PIXMK = ["o", "s", "^", "D", "v", "P"]      # pixel markers; Q.CMK is the folder's charge->marker map
for k, acq in enumerate(PANELS):
    ax = axs.flat[k]
    g = TO[(TO.acq == acq) & ~TO.at_ceiling]
    for i, (r, c) in enumerate(Q.JUL_PIXELS):
        s = g[(g.row == r) & (g.col == c)].sort_values("charge_fC")
        y = s.turnoff_offset if s.turnoff_offset.notna().any() else s.turnoff_dac - s.turnoff_dac.min() + 10
        ax.plot(s.charge_fC, y, "-", marker=PIXMK[i], ms=7, lw=1.6,
                color=plt.cm.tab10(i), label="(%d,%d)" % (r, c))
    ax.set_title(acq + ("   (no baseline: DAC - min + 10)" if acq == "v2.03_BARE4 07-18" else ""),
                 loc="left", fontsize=12)
    ax.axhline(Q.OP_OFFSET, color=Q.C_MAG, ls="-.", lw=1.5)
    ax.set_xlim(3, 29)
    if k >= 4:
        ax.set_xlabel("injected charge  [fC]")
    if k % 4 == 0:
        ax.set_ylabel("turn-off above baseline  [DAC]")
    if k == 0:
        ax.legend(title="pixel", fontsize=10, title_fontsize=10, ncol=2)
axs.flat[7].axis("off")
fig.tight_layout()
Q.fig_header(fig, "Turn-off against injected charge, six pixels, seven acquisitions",
             tag="binary-search 50 % point above the noise-peak baseline; dash-dot = BL+20; pixel (0,14) is the saturating line on every die",
             scale=0.62, campaign=Q.JUL_CAMPAIGN)
Q.save_fig(fig, "jul_fig2_turnoff_vs_charge")
plt.show()

# %% [markdown]
# ## 5. Pixel (0,14): a compressive response on six dies, and what the noise measurements can and cannot say
#
# On every acquisition above, the (0,14) line is the one that bends over. At 5–8 fC it sits
# close to the other pixels (on `BARE9` within 2 DAC at 5 fC); above 8 fC it falls away and
# saturates at **80–100 DAC above baseline** while the other five reach 200–300. Fitted as a
# chord over 8–27 fC that reads as a slope of 2.3–3.5 DAC/fC against 8.1–13.0, a ratio of about
# 0.3 — and it is the same pixel on `BARE1`, `BARE4`, `BARE9` and `BARE10` here, and, reading
# notebooks 01, 02 and 03's own per-pixel fit tables, on `W36_6` (v2.01, January), `Bar4` (the
# v2.01 hybrid, January) and `BARE9` again at every rail in August. Six dies, two revisions, three
# sites, one pixel. Whatever it is, it is not a defect of a die.
#
# The cell after the table tests the simplest reading, **a constant charge deficit at the
# injector**: that would map the (0,14) curve onto the others' with one constant scale factor
# *k*. It does not — the implied *k* drifts from ~0.7 at 8 fC to ~0.4 at 27 fC on every die, and
# the local slope at (0,14) falls from 5–6 DAC/fC at 5–8 fC to ~2 at 22–27 fC while the other
# pixels' rises. The response is **compressive**, not scaled. Two things could compress it: an
# injection pulse whose amplitude saturates, or a front end whose output does. The noise
# measurements are the discriminator between those, and they have to be read for what they can
# resolve:
#
# * The **256-pixel autocalibration noise width `NW`** is an integer with an array RMS of
#   0.7–1.0 DAC on a mean of 3.5–4.4, so a single pixel's NW can exclude a 3× deficit of
#   small-signal gain but has no power below ~30 %. On four maps from four dies (`BARE4` here,
#   `BARE9` in August, `W36_6` and `Bar4` in January) (0,14) reads 5, 4, 3 and 3 against array
#   means of 3.8, 4.1, 3.5 and 4.4. So: **not a 3× small-signal deficit** — consistent with the
#   near-normal turn-offs at 5–8 fC — and nothing finer.
# * The **noise-scan FWHM** is finer, and its proper control is the other two column-14 pixels,
#   because section 3 shows column 14 wider than column 0 on every die. Against that control
#   (0,14) reads 0.6–0.8 (mean 0.68), and the shared S-curve width at (0,14) over 8–27 fC is
#   0.40–0.72 (median 0.54) of the other five's. Both are what a front end whose output
#   compresses would show at the operating point: noise projected onto the threshold axis
#   through a shrinking local slope. A saturating injector with a normal front end would leave
#   the small-signal FWHM alone. The weight is therefore on the **front end at (0,14) behaving
#   differently**, not on the injector — and since the same DAQ software configured every one of
#   the six dies, a per-pixel configuration difference (a front-end register that the software
#   sets differently for that pixel) is the leading candidate. A register read-back at (0,14)
#   settles it; none exists in any of these campaigns.
#
# The consequence runs backwards through the folder regardless of cause. Notebooks 01–03 pooled
# their gains over all scanned pixels, (0,14) included: with six pixels that pulls the mean down
# by ~12 %, with twelve by ~6 %, and inflates the pixel-to-pixel RMS everywhere. The dose *ratios*
# are paired per pixel and unaffected; the absolute gains, ENCs and pixel spreads are not, and
# (0,14) should be excluded from them as an unrepresentative pixel of unresolved cause. The last
# cell of this section recomputes each earlier campaign's pooled gain with and without (0,14)
# from the tables those notebooks wrote, so the size of the correction is a number rather than a
# warning. In January's twelve pixels the whole of row 0 also reads low (column 8 at 0.7 of its
# neighbours), so a position dependence is not confined to one corner; only (0,14) is common to
# all campaigns and only it is treated here.

# %%
rows = []
for acq, g in GF.groupby("acq"):
    a = g[(g.row == 0) & (g.col == 14)].gain_dac_per_fc
    b = g[~((g.row == 0) & (g.col == 14))].gain_dac_per_fc
    rows.append(dict(campaign="FNAL bench Jul-2024", acquisition=acq, chip=DIE_OF[acq],
                     gain_0_14=float(a.iloc[0]) if len(a) else np.nan, gain_others=b.mean(),
                     others_rms=b.std(ddof=1), n_others=len(b)))
bj = pd.read_csv(Q.OUTDIR / "bare_gain_fits.csv")
for (d, t), g in bj.groupby(["dose", "temp"]):
    a = g[(g.row == 0) & (g.col == 14)].gain_dac_per_fc
    b = g[~((g.row == 0) & (g.col == 14))].gain_dac_per_fc
    rows.append(dict(campaign="CERN Jan-2024 TID (nb 01)", acquisition="%d MRad, %d C" % (d, t), chip="ET2p01_W36_6",
                     gain_0_14=float(a.iloc[0]) if len(a) else np.nan, gain_others=b.mean(),
                     others_rms=b.std(ddof=1), n_others=len(b)))
hj = pd.read_csv(Q.OUTDIR / "hybrid_gain_fits.csv")
for cfg, g in hj.groupby("config"):
    a = g[(g.row == 0) & (g.col == 14)].gain_dac_per_fc
    b = g[~((g.row == 0) & (g.col == 14))].gain_dac_per_fc
    if len(a) and len(b) >= 4:
        rows.append(dict(campaign="CERN Jan-2024 CC hybrid (nb 02)", acquisition=cfg, chip="ET2p01_Bar4",
                         gain_0_14=float(a.iloc[0]), gain_others=b.mean(), others_rms=b.std(ddof=1), n_others=len(b)))
ej = pd.read_csv(Q.OUTDIR / "et2p03_gain_fits.csv")
for cfg, g in ej[ej.chip == "ET2p03_BARE9"].groupby("config"):
    a = g[(g.row == 0) & (g.col == 14)].gain_dac_per_fc
    b = g[~((g.row == 0) & (g.col == 14))].gain_dac_per_fc
    if len(a) and len(b) >= 4:
        rows.append(dict(campaign="CERN Aug-2024 X-ray (nb 03)", acquisition=cfg, chip="ET2p03_BARE9",
                         gain_0_14=float(a.iloc[0]), gain_others=b.mean(), others_rms=b.std(ddof=1), n_others=len(b)))
P14 = pd.DataFrame(rows)
P14["ratio"] = P14.gain_0_14 / P14.gain_others
P14.to_csv(Q.OUTDIR / "jul_pixel_0_14_across_campaigns.csv", index=False)
print("pixel (0,14) against the mean of the other scanned pixels, every acquisition of every campaign in the folder\n")
print(P14.round(2).to_string(index=False))
print("\nratio (0,14) / others: median %.2f, range %.2f-%.2f over %d acquisitions on %d dies"
      % (P14.ratio.median(), P14.ratio.min(), P14.ratio.max(), len(P14), P14.chip.nunique()))
g14 = GF[(GF.row == 0) & (GF.col == 14)].gain_dac_per_fc
g5 = GF[~((GF.row == 0) & (GF.col == 14))].gain_dac_per_fc
print("July 8-27 fC chord slopes: (0,14) %.2f-%.2f DAC/fC, the other five %.2f-%.2f" % (g14.min(), g14.max(), g5.min(), g5.max()))
RESULTS["pixel_0_14_ratio_median"] = float(P14.ratio.median())
RESULTS["pixel_0_14_n_dies"] = int(P14.chip.nunique())

# %%
# Is (0,14) a constant charge deficit? Solve others(k*Q) = turnoff_0_14(Q) per charge by interpolation.
krows = []
for acq, g in TO[TO.acq != "v2.03_BARE4 07-18"].groupby("acq"):
    o = g[~((g.row == 0) & (g.col == 14))].groupby("charge_fC").turnoff_offset.mean()
    p = g[(g.row == 0) & (g.col == 14)].set_index("charge_fC").turnoff_offset
    for q in (8, 15, 22, 27):
        if q in p.index and np.isfinite(p[q]):
            k = np.interp(p[q], o.values, o.index.values) / q      # the charge at which the others cross (0,14)'s turn-off
            krows.append(dict(acq=acq, charge_fC=q, turnoff_0_14=p[q], others_mean=o[q], implied_k=k))
    for lo, hi in ((5, 8), (22, 27)):
        if lo in p.index and hi in p.index:
            krows.append(dict(acq=acq, charge_fC=lo, chord=(hi - lo), local_slope_0_14=(p[hi] - p[lo]) / (hi - lo),
                              local_slope_others=(o[hi] - o[lo]) / (hi - lo)))
K14 = pd.DataFrame(krows)
K14.to_csv(Q.OUTDIR / "jul_pixel_0_14_constant_k_test.csv", index=False)
print("implied charge scale k = Q_others / Q at which the other pixels reach (0,14)'s turn-off:")
print(K14[K14.implied_k.notna()].pivot_table(index="acq", columns="charge_fC", values="implied_k").round(2).to_string())
print("\nlocal slope [DAC/fC] at (0,14) against the other five, 5-8 fC and 22-27 fC chords:")
print(K14[K14.local_slope_0_14.notna()].pivot_table(index="acq", columns="charge_fC", values=["local_slope_0_14", "local_slope_others"]).round(1).to_string())
sat = TO[(TO.row == 0) & (TO.col == 14) & (TO.charge_fC == 27) & TO.turnoff_offset.notna()]
print("\n(0,14) turn-off above baseline at 27 fC: %.0f-%.0f DAC on %d acquisitions; the other pixels: %.0f-%.0f"
      % (sat.turnoff_offset.min(), sat.turnoff_offset.max(), len(sat),
         TO[~((TO.row == 0) & (TO.col == 14)) & (TO.charge_fC == 27)].turnoff_offset.min(),
         TO[~((TO.row == 0) & (TO.col == 14)) & (TO.charge_fC == 27)].turnoff_offset.max()))
RESULTS["pixel_0_14_k_at_8"] = float(K14[K14.charge_fC == 8].implied_k.median())
RESULTS["pixel_0_14_k_at_27"] = float(K14[K14.charge_fC == 27].implied_k.median())

# %%
# S-curve width at (0,14) against the other five, per July acquisition, and on W36_6 at 0 MRad
PRB = Q.et2p03_probes(TOF[TOF.acq != "v2.03_BARE4 07-18"], keys=("acq", "row", "col", "charge_fC"))
srows = []
for acq, g in PRB.groupby("acq"):
    g = g[g.charge_fC >= Q.JUL_QLO]
    for lab, m in [("(0,14)", (g.row == 0) & (g.col == 14)), ("other five", ~((g.row == 0) & (g.col == 14)))]:
        cur, _ = Q.curves_from(g[m], keys=("row", "col", "charge_fC"))
        s, sd, _, _ = Q.scurve_sigma(cur, boot=100)
        srows.append(dict(acquisition=acq, pixels=lab, n_curves=len(cur), sigma_dac=s, sigma_sd=sd))
bp = Q.load_bare_probes()
b0 = bp[(bp.dose == 0) & (bp.charge >= Q.JUL_QLO)]
for lab, m in [("(0,14)", (b0.row == 0) & (b0.col == 14)), ("other eleven", ~((b0.row == 0) & (b0.col == 14)))]:
    cur, _ = Q.curves_from(b0[m], keys=("temp", "row", "col", "charge"))
    s, sd, _, _ = Q.scurve_sigma(cur, boot=100)
    srows.append(dict(acquisition="W36_6 0 MRad (nb 01)", pixels=lab, n_curves=len(cur), sigma_dac=s, sigma_sd=sd))
S14 = pd.DataFrame(srows)
S14["pixels"] = S14.pixels.replace({"other eleven": "others", "other five": "others"})
sp = S14.pivot_table(index="acquisition", columns="pixels", values="sigma_dac")
sp["ratio"] = sp["(0,14)"] / sp["others"]
S14.to_csv(Q.OUTDIR / "jul_pixel_0_14_sigma.csv", index=False)
print("shared S-curve sigma [DAC], charges >= 8 fC")
print(sp.round(2).to_string())
jr = sp.loc[[a for a in sp.index if "W36_6" not in a], "ratio"]
print("\nJuly sigma ratio (0,14)/others: %.2f-%.2f, median %.2f" % (jr.min(), jr.max(), jr.median()))
# noise-scan FWHM at (0,14) against the column-14 control (3,14) and (12,14)
fw = NOI[NOI.acq != "v2.03_BARE4 07-18"]
f14 = fw[(fw.row == 0) & (fw.col == 14)].set_index("acq").fwhm
fc = fw[(fw.col == 14) & (fw.row != 0)].groupby("acq").fwhm.mean()
f00 = fw[(fw.row == 0) & (fw.col == 0)].set_index("acq").fwhm
FW = pd.DataFrame(dict(fwhm_0_14=f14, fwhm_col14_control=fc, fwhm_0_0=f00))
FW["ratio_to_col14"] = FW.fwhm_0_14 / FW.fwhm_col14_control
FW.to_csv(Q.OUTDIR / "jul_pixel_0_14_fwhm_control.csv")
print("\nnoise-scan FWHM [DAC] at (0,14) against the column-14 control:\n" + FW.round(2).to_string())
RESULTS["pixel_0_14_fwhm_ratio_col14"] = float(FW.ratio_to_col14.mean())
RESULTS["pixel_0_14_sigma_ratio_median"] = float(jr.median())

# %%
# The four 256-pixel noise-width maps: is (0,14), row 0 or column 14 special?
maps = [("BARE4 (v2.03), 2024-07-17", MAP.rename(columns={"bl_dac": "bl", "nw_dac": "nw"}))]
am = Q.load_et2p03_tid_blnw()
maps.append(("BARE9 (v2.03), 2024-08-03", am[am.chip == "ET2p03_BARE9"].rename(columns={"baseline_DAC": "bl", "noise_width_DAC": "nw"})))
jb = Q.load_bare_blnw()
maps.append(("W36_6 (v2.01), %s" % jb.day.iloc[0], jb[jb.day == jb.day.iloc[0]].rename(columns={"baseline_dac": "bl", "noise_width_dac": "nw"})))
hb = Q.load_hybrid_blnw()
h0 = hb[(hb.day == hb.day.iloc[0]) & (hb.map_chip == hb.map_chip.iloc[0])]
h0 = h0.drop_duplicates(["row", "col"])
maps.append(("Bar4 hybrid (v2.01), %s" % hb.day.iloc[0][:10], h0.rename(columns={"baseline_dac": "bl", "noise_width_dac": "nw"})))
nw_rows = []
for name, m in maps:
    m = m[["row", "col", "bl", "nw"]].astype(float)
    nw_rows.append(dict(map=name, n=len(m), NW_mean=m.nw.mean(), NW_rms=m.nw.std(ddof=1),
                        NW_0_14=float(m[(m.row == 0) & (m.col == 14)].nw.iloc[0]),
                        NW_0_0=float(m[(m.row == 0) & (m.col == 0)].nw.iloc[0]),
                        NW_row0=m[m.row == 0].nw.mean(), NW_col14=m[m.col == 14].nw.mean(),
                        BL_row0_minus_row15=m[m.row == 0].bl.mean() - m[m.row == 15].bl.mean(),
                        BL_col0_minus_col15=m[m.col == 0].bl.mean() - m[m.col == 15].bl.mean()))
NWM = pd.DataFrame(nw_rows)
NWM.to_csv(Q.OUTDIR / "jul_nw_map_uniformity.csv", index=False)
print("autocalibration noise width NW [DAC] on four 256-pixel maps\n")
print(NWM.round(2).to_string(index=False))
NWM["NW_0_14_over_mean"] = NWM.NW_0_14 / NWM.NW_mean
print("\n(0,14) over the array mean, per map: %s" % ", ".join("%.2f" % v for v in NWM.NW_0_14_over_mean))
print("array RMS over mean, per map: %s  -> a one-pixel NW excludes a ~3x deficit, not a ~30 %% one"
      % ", ".join("%.2f" % v for v in NWM.NW_rms / NWM.NW_mean))
RESULTS["nw_maps_0_14_over_mean"] = float(NWM.NW_0_14_over_mean.mean())

# %%
fig, axs = plt.subplots(1, 3, figsize=(22, 6.4))
ax = axs[0]
for i, (name, m) in enumerate(maps):
    m = m[["row", "col", "nw"]].astype(float)
    vals, cnt = np.unique(m.nw.values, return_counts=True)
    ax.plot(vals + 0.08 * (i - 1.5), cnt / cnt.sum(), "-o", ms=6, lw=1.4, color=plt.cm.tab10(i), label=name)
    ax.plot([float(m[(m.row == 0) & (m.col == 14)].nw.iloc[0]) + 0.08 * (i - 1.5)], [0.02 + 0.03 * i], "*", ms=14,
            color=plt.cm.tab10(i), mec="black")
ax.set_xlim(0, 8)
ax.set_xlabel("autocalibration NW  [DAC], integer")
ax.set_ylabel("fraction of the 256 pixels")
ax.legend(fontsize=9, title="stars: the (0,14) value", title_fontsize=9)
ax.set_title("NW distribution on four dies, with (0,14) marked", loc="left", fontsize=12)
ax = axs[1]
lab = P14.campaign + " " + P14.acquisition.str.slice(0, 22)
ax.barh(range(len(P14)), P14.ratio, color=[Q.C_ORANGE if "Jul" in c else Q.C_BLUE for c in P14.campaign])
ax.set_yticks(range(len(P14)))
ax.set_yticklabels(lab, fontsize=7)
ax.axvline(1, color="black", lw=1)
ax.set_xlabel("slope (0,14) / mean slope of the other pixels")
ax.set_xlim(0, 1.2)
ax.set_title("the (0,14) slope ratio, every acquisition in the folder", loc="left", fontsize=12)
ax = axs[2]
for i, acq in enumerate([a for a in ACQS if a != "v2.03_BARE4 07-18"]):
    g = PRB[(PRB.acq == acq) & (PRB.row == 0) & (PRB.charge_fC == 15)]
    for (c, mk) in [(14, "o"), (0, "s")]:
        s = g[g.col == c].sort_values("dac")
        bl0 = BL.get((acq, 0, c), np.nan)
        ax.plot(s.dac - bl0, s.frac_c, "-" if c == 14 else "--", marker=mk, ms=4, lw=1.2,
                color=plt.cm.tab10(i), label=(acq if c == 14 else None))
ax.set_xlabel("threshold above baseline  [DAC]")
ax.set_ylabel("trigger-bit efficiency")
ax.set_xlim(0, 160)
ax.legend(fontsize=8, title="solid (0,14), dashed (0,0); 15 fC", title_fontsize=8)
ax.set_title("the 15 fC S-curve at (0,14) against (0,0): steeper, and lower", loc="left", fontsize=12)
fig.tight_layout()
Q.fig_header(fig, "Pixel (0,14): a compressive response on six dies of two revisions",
             tag="left: four 256-pixel NW maps; middle: per-acquisition 8-27 fC chord ratio, July in orange; right: binary-search probes",
             scale=0.62, campaign=Q.JUL_CAMPAIGN)
Q.save_fig(fig, "jul_fig3_pixel_0_14")
plt.show()

# %%
# The pooled gains of notebooks 01-03, with and without (0,14), from their own per-pixel tables
prow = []
for (d, t), g in bj.groupby(["dose", "temp"]):
    prow.append(dict(notebook="01 W36_6", condition="%d MRad, %d C" % (d, t),
                     gain_all=g.gain_dac_per_fc.mean(), rms_all=g.gain_dac_per_fc.std(ddof=1), n_all=len(g),
                     gain_excl=g[~((g.row == 0) & (g.col == 14))].gain_dac_per_fc.mean(),
                     rms_excl=g[~((g.row == 0) & (g.col == 14))].gain_dac_per_fc.std(ddof=1)))
for cfg, g in hj.groupby("config"):
    if len(g) < 6:
        continue
    prow.append(dict(notebook="02 Bar4 hybrid", condition=cfg,
                     gain_all=g.gain_dac_per_fc.mean(), rms_all=g.gain_dac_per_fc.std(ddof=1), n_all=len(g),
                     gain_excl=g[~((g.row == 0) & (g.col == 14))].gain_dac_per_fc.mean(),
                     rms_excl=g[~((g.row == 0) & (g.col == 14))].gain_dac_per_fc.std(ddof=1)))
for cfg, g in ej[ej.chip == "ET2p03_BARE9"].groupby("config"):
    prow.append(dict(notebook="03 BARE9", condition=cfg,
                     gain_all=g.gain_dac_per_fc.mean(), rms_all=g.gain_dac_per_fc.std(ddof=1), n_all=len(g),
                     gain_excl=g[~((g.row == 0) & (g.col == 14))].gain_dac_per_fc.mean(),
                     rms_excl=g[~((g.row == 0) & (g.col == 14))].gain_dac_per_fc.std(ddof=1)))
for acq, g in GF.groupby("acq"):
    prow.append(dict(notebook="05 " + acq, condition="bench",
                     gain_all=g.gain_dac_per_fc.mean(), rms_all=g.gain_dac_per_fc.std(ddof=1), n_all=len(g),
                     gain_excl=g[~((g.row == 0) & (g.col == 14))].gain_dac_per_fc.mean(),
                     rms_excl=g[~((g.row == 0) & (g.col == 14))].gain_dac_per_fc.std(ddof=1)))
POOL = pd.DataFrame(prow)
POOL["excl_over_all"] = POOL.gain_excl / POOL.gain_all
POOL.to_csv(Q.OUTDIR / "jul_pooled_gain_with_without_0_14.csv", index=False)
print("pooled preamp gain [DAC/fC] with and without pixel (0,14)\n")
print(POOL.round(2).to_string(index=False))
RESULTS["pooled_gain_bias_6px"] = float(POOL[POOL.n_all == 6].excl_over_all.median())
RESULTS["pooled_gain_bias_12px"] = float(POOL[POOL.n_all == 12].excl_over_all.median()) if (POOL.n_all == 12).any() else np.nan
print("\nexcluding (0,14) raises a six-pixel pooled gain by a median factor %.3f and a twelve-pixel one by %.3f"
      % (RESULTS["pooled_gain_bias_6px"], RESULTS["pooled_gain_bias_12px"]))

# %% [markdown]
# ## 6. v2.01 against v2.03 on the same bench
#
# From here on every pooled number is over the **five pixels other than (0,14)**, and the
# per-pixel comparison between dies is made pixel by pixel.
#
# The question notebook 04 could not settle is whether the threshold-DAC scale follows the
# reference voltage. If the DAC full scale is proportional to VRef and the front ends of the two
# revisions are the same circuit at the same rail and registers, then a v2.01 die on its external
# 1.0 V reference should read a slope **0.86–0.89 ×** that of a v2.03 die on its internal
# reference, because the internal reference measures 0.86–0.89 V (notebook 03 section 15, `BARE9`
# at −23 °C; notebook 00 section 7.6, `BARE4` at 0 °C plus its +0.57 mV/°C; 0.891 V is used for
# the line below, `BARE4` extrapolated to ~22 °C) — a larger LSB, fewer DAC per fC. The S-curve
# width and the noise FWHM in DAC would scale the same way, and the ENC would not move at all.
#
# What the bench shows is mixed, and it is reported as mixed. The v2.01 die's **slope** and
# **S-curve width** sit inside the die-to-die spread of the three v2.03 dies with ratios near
# 1.0; its **noise FWHM** ratio lands on the VRef prediction (0.88) rather than on 1.0. Three
# observables, two of which favour one reading and one the other. Two caveats make the slope
# result weaker than its error bar: the error is a spread over three v2.03 dies (two degrees of
# freedom) applied to one v2.01 die, so the 1.7 σ is a Student-t on 2 dof, about 1.2 σ in
# Gaussian terms; and **the analog rail was hand-set and unrecorded on every day**. Notebook 03's
# rail scan gives d ln(gain)/dVA ≈ 2.5 per volt at this operating point, so the whole gap
# between the observed 1.05 and the predicted 0.89 corresponds to ~65 mV of supply difference
# between the 07-22 setup and the v2.03 days — well inside what a bench supply set by hand can
# differ by. This comparison therefore **cannot resolve** the DAC-scale question on its own; it
# fixes the size of the rail systematic that the transfer-function measurement (the bench
# plan's first item, with the rail and the registers read back) must control. What does
# survive is narrower: the two `BARE4` days and the two `BARE9` days reproduce each other's slope
# to 0.1 %, so the 8 % spread among the v2.03 dies is a genuine die-to-die spread and not a
# day-to-day one.

# %%
VREF_INT_V = 0.891     # BARE4 in the chamber, 0.878 V at 0 C, +0.573 mV/C to ~22 C (notebook 00, 7.6)
VREF_INT_BAND = (0.864, 0.891)   # BARE9 at -23 C (notebook 03, 15) to BARE4 at ~22 C
VREF_EXT_V = 1.000     # the external reference of the v2.01 boards
PRED_RATIO = VREF_INT_V / VREF_EXT_V
PRED_BAND = (VREF_INT_BAND[0] / VREF_EXT_V, VREF_INT_BAND[1] / VREF_EXT_V)
# the rail sensitivity of the chord slope, from notebook 03's BARE9 0 MRad rail scan (supply 0.85 -> 1.20 V)
_rs = pd.read_csv(Q.OUTDIR / "et2p03_gain_by_condition.csv")
_rs = _rs[(_rs.chip == "ET2p03_BARE9") & (_rs.dose == 0)].set_index("va").gain_dac_per_fc
DLNG_DVA = float(np.log(_rs[1.20] / _rs[0.85]) / (1.20 - 0.85))
print("d ln(gain)/dVA from notebook 03's rail scan: %.2f per volt (BARE9, 0 MRad, supply 0.85 -> 1.20 V)" % DLNG_DVA)
G5 = GF[~((GF.row == 0) & (GF.col == 14))]
die_rows = []
for acq in ACQS:
    g = G5[G5.acq == acq]
    p = Q.pooled(g.gain_dac_per_fc)
    n = NOI[NOI.acq == acq]
    n5 = n[~((n.row == 0) & (n.col == 14))]
    pr = PRB[(PRB.acq == acq) & (PRB.charge_fC >= Q.JUL_QLO) & ~((PRB.row == 0) & (PRB.col == 14))]
    if len(pr):
        cur, _ = Q.curves_from(pr, keys=("row", "col", "charge_fC"))
        s, sd, _, _ = Q.scurve_sigma(cur, boot=100)
    else:
        s, sd, cur = np.nan, np.nan, []
    die_rows.append(dict(acquisition=acq, chip=DIE_OF[acq], revision=DIE_OF[acq][:6].replace("ET2p0", "v2.0"),
                         n_pixels=p["n"], gain_dac_per_fc=p["mean"], gain_sem=p["sem"], gain_pixel_rms=p["rms"],
                         BL_mean=n5.BL.mean(), fwhm_mean=n5.fwhm.mean(), fwhm_rms=n5.fwhm.std(ddof=1),
                         n_curves=len(cur), sigma_dac=s, sigma_sd=sd,
                         ENC_e=Q.enc_electrons(s, p["mean"]) if np.isfinite(s) else np.nan))
DIE = pd.DataFrame(die_rows)
DIE.to_csv(Q.OUTDIR / "jul_gain_by_die.csv", index=False)
print("per acquisition, five pixels (excluding (0,14)):\n")
print(DIE.round(2).to_string(index=False))

# %%
ok = DIE[DIE.acquisition != "v2.03_BARE4 07-18"]
v203 = ok[ok.revision == "v2.03"].groupby("chip").agg(gain=("gain_dac_per_fc", "mean"), sigma=("sigma_dac", "mean"),
                                                        fwhm=("fwhm_mean", "mean"), ENC=("ENC_e", "mean"))
v201 = ok[ok.revision == "v2.01"].iloc[0]
print("v2.03 dies, day-averaged:\n" + v203.round(2).to_string())
print("\nv2.01 BARE10: gain %.2f, sigma %.2f, FWHM %.2f, ENC %.0f" % (v201.gain_dac_per_fc, v201.sigma_dac, v201.fwhm_mean, v201.ENC_e))
r_gain = v201.gain_dac_per_fc / v203.gain.mean()
r_gain_err = r_gain * np.sqrt((v203.gain.std(ddof=1) / np.sqrt(len(v203)) / v203.gain.mean()) ** 2
                              + (v203.gain.std(ddof=1) / v203.gain.mean()) ** 2)   # the one v2.01 die gets the v2.03 spread
r_sig = v201.sigma_dac / v203.sigma.mean()
r_fwhm = v201.fwhm_mean / v203.fwhm.mean()
r_fwhm_err = r_fwhm * np.sqrt((v203.fwhm.std(ddof=1) / v203.fwhm.mean()) ** 2 + (v203.fwhm.std(ddof=1) / np.sqrt(len(v203)) / v203.fwhm.mean()) ** 2)
r_enc = v201.ENC_e / v203.ENC.mean()
r_enc_err = r_enc * np.sqrt((v203.ENC.std(ddof=1) / v203.ENC.mean()) ** 2 + (v203.ENC.std(ddof=1) / np.sqrt(len(v203)) / v203.ENC.mean()) ** 2)
# pixel-by-pixel: BARE10 against the mean of the three v2.03 dies at the same pixel
pp = G5.groupby(["chip", "row", "col"]).gain_dac_per_fc.mean().unstack("chip")
pp = pp[pp.index.get_level_values(0).notna()]
pp["v2.03_mean"] = pp[["ET2p03_BARE1", "ET2p03_BARE4", "ET2p03_BARE9"]].mean(axis=1)
pp["ratio_2p01_over_2p03"] = pp["ET2p01_BARE10"] / pp["v2.03_mean"]
print("\npixel by pixel, day-averaged gain [DAC/fC]:\n" + pp.round(2).to_string())
REV = pd.DataFrame([
    dict(quantity="preamp gain, v2.01 / mean v2.03", observed=r_gain, err=r_gain_err,
         predicted_if_DAC_follows_VRef=PRED_RATIO, predicted_if_same_scale=1.0),
    dict(quantity="preamp gain, pixel-by-pixel median ratio", observed=pp.ratio_2p01_over_2p03.median(),
         # the pixel-wise ratio removes the injection gradient but not the one v2.01 die's own place in the die-to-die spread
         err=np.hypot(pp.ratio_2p01_over_2p03.std(ddof=1) / np.sqrt(len(pp)),
                      pp.ratio_2p01_over_2p03.median() * v203.gain.std(ddof=1) / v203.gain.mean()),
         predicted_if_DAC_follows_VRef=PRED_RATIO, predicted_if_same_scale=1.0),
    dict(quantity="S-curve sigma, v2.01 / mean v2.03", observed=r_sig, err=r_sig * np.sqrt((v201.sigma_sd / v201.sigma_dac) ** 2 + (v203.sigma.std(ddof=1) / v203.sigma.mean()) ** 2),
         predicted_if_DAC_follows_VRef=PRED_RATIO, predicted_if_same_scale=1.0),
    dict(quantity="noise FWHM, v2.01 / mean v2.03", observed=r_fwhm, err=r_fwhm_err,
         predicted_if_DAC_follows_VRef=PRED_RATIO, predicted_if_same_scale=1.0),
    dict(quantity="ENC, v2.01 / mean v2.03", observed=r_enc, err=r_enc_err,
         predicted_if_DAC_follows_VRef=1.0, predicted_if_same_scale=1.0),
])
from scipy import stats
REV["t_from_VRef_prediction"] = (REV.observed - REV.predicted_if_DAC_follows_VRef) / REV.err
REV["t_from_same_scale"] = (REV.observed - REV.predicted_if_same_scale) / REV.err
# the die spread comes from three dies: two degrees of freedom, so read t on a Student-t(2)
REV["gauss_sigma_equiv_VRef"] = stats.norm.isf(stats.t.sf(REV.t_from_VRef_prediction.abs(), df=2))
REV["rail_equiv_mV"] = 1e3 * np.log(REV.observed / REV.predicted_if_DAC_follows_VRef) / DLNG_DVA
REV.loc[REV.quantity.str.startswith("ENC"), "rail_equiv_mV"] = np.nan
V_PER_FC_RATIO = r_gain / PRED_RATIO
REV.to_csv(Q.OUTDIR / "jul_revision_comparison.csv", index=False)
print("\nv2.01 against v2.03 on the same bench (predicted ratio %.3f if the DAC scale follows VRef, %.3f V / %.3f V)\n"
      % (PRED_RATIO, VREF_INT_V, VREF_EXT_V))
print(REV.round(3).to_string(index=False))
print("\nif the DAC scale does follow VRef (band %.3f-%.3f), the v2.01 front end delivers %.2f x the v2.03 volts per fC"
      % (PRED_BAND[0], PRED_BAND[1], V_PER_FC_RATIO))
print("the gain gap 1.05 vs %.3f is equivalent to %.0f mV of analog-supply difference at d ln(gain)/dVA = %.2f /V; the rail was not recorded"
      % (PRED_RATIO, REV.rail_equiv_mV.iloc[0], DLNG_DVA))
RESULTS["rev_gain_rail_equiv_mV"] = float(REV.rail_equiv_mV.iloc[0])
RESULTS["rev_gain_gauss_sigma_equiv"] = float(REV.gauss_sigma_equiv_VRef.iloc[0])
RESULTS["rev_v_per_fc_ratio_if_vref"] = float(V_PER_FC_RATIO)
RESULTS["rev_fwhm_ratio"] = float(r_fwhm)
RESULTS["rev_gain_ratio"] = float(r_gain)
RESULTS["rev_gain_ratio_err"] = float(r_gain_err)
RESULTS["rev_pred_ratio_vref"] = float(PRED_RATIO)
RESULTS["rev_sigma_ratio"] = float(r_sig)

# %%
fig, axs = plt.subplots(1, 3, figsize=(22, 6.4))
ax = axs[0]
chips = ["ET2p01_BARE10", "ET2p03_BARE1", "ET2p03_BARE4", "ET2p03_BARE9"]
cols = {"ET2p01_BARE10": Q.C_ORANGE, "ET2p03_BARE1": Q.C_BLUE, "ET2p03_BARE4": Q.C_AQUA, "ET2p03_BARE9": Q.C_MAG}
pix5 = [p for p in Q.JUL_PIXELS if p != (0, 14)]
for k, ch in enumerate(chips):
    g = G5[G5.chip == ch].groupby(["row", "col"]).gain_dac_per_fc.agg(["mean", "std", "count"])
    y = [g.loc[p, "mean"] if p in g.index else np.nan for p in pix5]
    ax.plot(np.arange(5) + 0.12 * (k - 1.5), y, "o", ms=9, color=cols[ch], label=ch.replace("ET2p0", "v2.0"))
ax.set_xticks(range(5))
ax.set_xticklabels(["(%d,%d)" % p for p in pix5])
ax.set_xlabel("pixel (row, col)")
ax.set_ylabel("preamp gain  [DAC/fC]")
ax.set_ylim(0, 15)
ax.legend(fontsize=10)
ax.set_title("gain per pixel, four dies (days averaged)", loc="left", fontsize=12)
ax = axs[1]
o = DIE[DIE.acquisition != "v2.03_BARE4 07-18"]
for k, ch in enumerate(chips):
    s = o[o.chip == ch]
    ax.errorbar(s.gain_dac_per_fc, s.sigma_dac, xerr=s.gain_sem, yerr=s.sigma_sd, fmt="o", ms=9,
                color=cols[ch], label=ch.replace("ET2p0", "v2.0"), capsize=3)
gg = np.linspace(6, 14, 10)
for enc, ls in [(600, ":"), (800, "--")]:
    ax.plot(gg, enc / Q.E_PER_FC * gg, ls, color=Q.GREY, lw=1.2)
    ax.text(13.6, enc / Q.E_PER_FC * 13.6, "%d e" % enc, fontsize=9, color=Q.GREY, va="bottom", ha="right")
ax.set_xlabel("preamp gain  [DAC/fC], five pixels")
ax.set_ylabel("shared S-curve σ  [DAC]")
ax.set_xlim(6, 14)
ax.set_ylim(0.6, 1.7)
ax.legend(fontsize=10)
ax.set_title("width against gain: both scale together, ENC ~600-800 e", loc="left", fontsize=12)
ax = axs[2]
ax.bar(range(len(REV)), REV.observed, yerr=REV.err, color=[Q.C_ORANGE, Q.C_ORANGE, Q.C_BLUE, Q.C_BLUE, Q.C_AQUA], capsize=4)
ax.axhline(1.0, color="black", lw=1.2, label="same DAC scale")
ax.axhspan(PRED_BAND[0], PRED_BAND[1], color=Q.C_MAG, alpha=0.25, lw=0, label="DAC scale follows VRef (%.2f-%.2f); ENC 1.0 either way" % PRED_BAND)
ax.set_xticks(range(len(REV)))
ax.set_xticklabels(["gain", "gain\n(pixel-wise)", "σ", "FWHM", "ENC"], fontsize=10)
ax.set_ylabel("v2.01 / v2.03")
ax.set_ylim(0.6, 1.4)
ax.legend(fontsize=10)
ax.set_title("the revision ratios against the two predictions", loc="left", fontsize=12)
fig.tight_layout()
Q.fig_header(fig, "v2.01 against v2.03 on one bench: gain, S-curve width, noise width and ENC",
             tag="pixel (0,14) excluded throughout; error bars are die spreads (three v2.03 dies, applied to the one v2.01 die), not fit errors; the rail was not recorded",
             scale=0.62, campaign=Q.JUL_CAMPAIGN)
Q.save_fig(fig, "jul_fig4_revision_comparison")
plt.show()

# %% [markdown]
# ## 7. `BARE9` across its move: two bench days and the first CERN day
#
# `BARE9` was scanned on 29 and 30 July at FNAL and again on 3 August at CERN (notebook 03's
# untagged `1p2VD_1p2VA` acquisition, the one it uses as its repeatability reference). Three
# acquisitions of one die in five days, in two setups. The charge ladders differ — bench
# {5, 6, 8, 15, 22, 27}, CERN {6, 8, 10, 15, 22} — and on the CERN day three of the six pixels
# reached the DAC ceiling at 22 fC, so an 8–22 fC fit is silently an 8–15 fC one on half the
# pixels there. Because the response is convex (section 4), the comparison is made on the one
# chord all three acquisitions share, **8 → 15 fC**, and quoted with its per-pixel spread. Read
# pixel by pixel:
#
# * the **8–15 fC slopes agree in the mean to ~2 % with a per-pixel spread of ±7 %** across all
#   three — the preamp response of the die did not measurably change between the bench and the
#   X-ray setup, and notebook 03's reference condition ("VA 1.20 V on the supply, ~1.0 V at the
#   chip") reproduces the FNAL bench slope, whatever the bench rail was;
# * the **baselines step by ~240 DAC between the two bench days and by another ~240 between
#   the second bench day and CERN**, at constant slope. Notebook 03's rail scan puts numbers on
#   what that excludes: the baseline moves ~370 DAC per volt of analog supply and the gain by
#   e^2.5 ≈ 12 per volt, so a 240 DAC step from the rail would need ~0.65 V and would multiply
#   the slope by ~5; conversely the ~2 % chord agreement between the two bench days bounds any
#   rail difference to ~10 mV, i.e. a few DAC of baseline. **These steps are not the rail.**
#   They are offsets of the discriminator's operating point — a register (a per-pixel or global
#   threshold offset), a reference wiring change, or a difference in what the DAQ software
#   wrote (both bench days are `.bin`, so the file format does not separate them) — and with no
#   register read-back for any of the three, which one is not decidable here. Between the two
#   bench days the noise FWHM and the S-curve width also moved (2.0 → 2.7 and 0.98 → 1.12 DAC at a
#   slope constant to 0.1 %), which is a change of input-referred noise and so of the front end's
#   state, not a pure DC offset — one more reason to read the registers back.

# %%
a_to = Q.load_et2p03_turnoff()
a_no = Q.load_et2p03_noiseonly()
a_to = a_to[a_to.config == "1p2VD_1p2VA"]
a_no = a_no[a_no.config == "1p2VD_1p2VA"]
a_bl = a_no.set_index(["row", "col"]).dac_at_triggerbits_max.astype(float)
a_to = a_to.assign(BL=[a_bl.get((r, c), np.nan) for r, c in zip(a_to.row, a_to.col)])
a_to["turnoff_offset"] = a_to.turnoff_dac - a_to.BL
a_to["acq"] = "v2.03_BARE9 08-03 CERN"
a_to["at_ceiling"] = a_to.turnoff_dac >= 1019
B9 = pd.concat([TO[TO.chip == "ET2p03_BARE9"][["acq", "row", "col", "charge_fC", "turnoff_dac", "BL", "turnoff_offset", "at_ceiling"]],
                a_to[["acq", "row", "col", "charge_fC", "turnoff_dac", "BL", "turnoff_offset", "at_ceiling"]]])
print("charge ladder and ceiling-censored searches per acquisition:")
for acq, g in B9.groupby("acq"):
    print("  %-24s charges %s   at ceiling: %d (%s)" % (acq, [int(q) for q in sorted(g.charge_fC.unique())], int(g.at_ceiling.sum()),
          ", ".join("(%d,%d)@%d" % (r, c, q) for r, c, q in zip(g[g.at_ceiling].row, g[g.at_ceiling].col, g[g.at_ceiling].charge_fC)) or "none"))
b9 = []
for (acq, r, c), g in B9.groupby(["acq", "row", "col"]):
    t = g[~g.at_ceiling].set_index("charge_fC").turnoff_dac
    if 8 in t.index and 15 in t.index:
        b9.append(dict(acq=acq, row=r, col=c, BL=float(g.BL.iloc[0]), chord_8_15=(t[15] - t[8]) / 7.0,
                       chord_8_22=(t[22] - t[8]) / 14.0 if 22 in t.index else np.nan))
B9F = pd.DataFrame(b9)
B9F.to_csv(Q.OUTDIR / "jul_bare9_three_acquisitions.csv", index=False)
print("\nBARE9, three acquisitions: baseline and the 8 -> 15 fC chord per pixel (the one chord all three share)\n")
print("baseline [DAC]:\n" + B9F.pivot_table(index="acq", columns=["row", "col"], values="BL").to_string())
print("\n8-15 fC chord [DAC/fC]:\n" + B9F.pivot_table(index="acq", columns=["row", "col"], values="chord_8_15").round(2).to_string())
print("\n8-22 fC chord where it exists (NaN = ceiling at 22 fC):\n" + B9F.pivot_table(index="acq", columns=["row", "col"], values="chord_8_22").round(2).to_string())
piv = B9F[~((B9F.row == 0) & (B9F.col == 14))].pivot_table(index="acq", columns=["row", "col"], values="chord_8_15")
rat = piv.div(piv.loc["v2.03_BARE9 07-30"], axis=1)
print("\n8-15 fC chord relative to 07-30, five pixels:\n" + rat.round(3).to_string())
for a in ("v2.03_BARE9 07-29", "v2.03_BARE9 08-03 CERN"):
    print("  %-24s mean %.3f, per-pixel range %.3f-%.3f" % (a, rat.loc[a].mean(), rat.loc[a].min(), rat.loc[a].max()))
RESULTS["bare9_gain_cern_over_bench"] = float(rat.loc["v2.03_BARE9 08-03 CERN"].mean())
RESULTS["bare9_gain_cern_over_bench_spread"] = float(rat.loc["v2.03_BARE9 08-03 CERN"].std(ddof=1))
RESULTS["bare9_gain_0729_over_0730"] = float(rat.loc["v2.03_BARE9 07-29"].mean())
# what the rail would have to do to move the baseline by the observed step
_bl = Q.load_et2p03_noiseonly()
_bl = _bl[(_bl.chip == "ET2p03_BARE9") & (_bl.dose_MRad == 0)].groupby("va_V").dac_at_triggerbits_max.mean()
DBL_DVA = float((_bl[1.20] - _bl[0.85]) / 0.35)
print("\nnotebook 03 rail scan, BARE9 0 MRad: baseline moves %.0f DAC per volt of analog supply, gain by a factor %.1f per volt"
      % (DBL_DVA, np.exp(DLNG_DVA)))
RESULTS["bare9_rail_for_240_dac_V"] = 240.0 / DBL_DVA
RESULTS["bare9_gain_factor_for_that_rail"] = float(np.exp(DLNG_DVA * 240.0 / DBL_DVA))
print("a 240 DAC baseline step from the rail would need %.2f V and would multiply the slope by %.1f; the slopes agree to %.1f %%"
      % (RESULTS["bare9_rail_for_240_dac_V"], RESULTS["bare9_gain_factor_for_that_rail"], 100 * abs(RESULTS["bare9_gain_0729_over_0730"] - 1)))
blp = B9F.pivot_table(index="acq", columns=["row", "col"], values="BL")
RESULTS["bare9_BL_shift_0730_minus_0729"] = float((blp.loc["v2.03_BARE9 07-30"] - blp.loc["v2.03_BARE9 07-29"]).mean())
RESULTS["bare9_BL_shift_cern_minus_0730"] = float((blp.loc["v2.03_BARE9 08-03 CERN"] - blp.loc["v2.03_BARE9 07-30"]).mean())
print("\nmean 8-15 fC chord ratio CERN/07-30 %.3f, 07-29/07-30 %.3f; baseline shifts 07-30 - 07-29 = %+.0f DAC, CERN - 07-30 = %+.0f DAC"
      % (RESULTS["bare9_gain_cern_over_bench"], RESULTS["bare9_gain_0729_over_0730"],
         RESULTS["bare9_BL_shift_0730_minus_0729"], RESULTS["bare9_BL_shift_cern_minus_0730"]))

# %%
fig, axs = plt.subplots(2, 3, figsize=(22, 11), sharex=True)
acq3 = ["v2.03_BARE9 07-29", "v2.03_BARE9 07-30", "v2.03_BARE9 08-03 CERN"]
c3 = [Q.C_BLUE, Q.C_AQUA, Q.C_ORANGE]
for k, (r, c) in enumerate(Q.JUL_PIXELS):
    ax = axs.flat[k]
    for a, col in zip(acq3, c3):
        s = B9[(B9.acq == a) & (B9.row == r) & (B9.col == c) & ~B9.at_ceiling].sort_values("charge_fC")
        ax.plot(s.charge_fC, s.turnoff_offset, "-o", ms=7, lw=1.6, color=col, label=a)
    ax.set_title("pixel (%d,%d)" % (r, c), loc="left", fontsize=12)
    ax.axhline(Q.OP_OFFSET, color=Q.C_MAG, ls="-.", lw=1.4)
    if k >= 3:
        ax.set_xlabel("injected charge  [fC]")
    if k % 3 == 0:
        ax.set_ylabel("turn-off above baseline  [DAC]")
    if k == 0:
        ax.legend(fontsize=10)
fig.tight_layout()
Q.fig_header(fig, "BARE9 before and after its move: turn-off above baseline on three days in two setups",
             tag="FNAL bench 07-29 and 07-30, CERN 08-03 (notebook 03's untagged 1p2VD_1p2VA run); absolute baselines differ by ~240 DAC per step; CERN ladder 6-22 fC",
             scale=0.62, campaign=Q.JUL_CAMPAIGN)
Q.save_fig(fig, "jul_fig5_bare9_three_acquisitions")
plt.show()

# %% [markdown]
# ## 8. The threshold edge: S-curve width, noise width and ENC, per die
#
# The shared-σ fit of notebooks 01–03 over the five good pixels and the four charges from 8 fC,
# per acquisition; the table is the one section 6 built. `BARE9` on 07-29 reads the narrowest
# edge and the narrowest noise FWHM of the campaign, on the day with the low baseline. With the
# slope constant to 0.1 % between the two days, a width that changes is a change of
# input-referred noise (ENC 626 → 716 e), i.e. of the front end's state — see section 7.

# %%
print(DIE[["acquisition", "n_curves", "sigma_dac", "sigma_sd", "fwhm_mean", "fwhm_rms", "gain_dac_per_fc", "ENC_e"]]
      .round(2).to_string(index=False))
DIE[["acquisition", "chip", "n_curves", "sigma_dac", "sigma_sd", "fwhm_mean", "ENC_e"]].to_csv(Q.OUTDIR / "jul_scurve_sigma.csv", index=False)

# %% [markdown]
# ## 9. The money plot, per die
#
# TOA jitter against threshold above baseline, one curve per charge, CAL-windowed moments,
# median over the five pixels with the inter-quartile band, and the quantisation floor
# `LSB/√12` at each die's own CAL. The construction is notebook 03 section 7's. All four dies
# sit at CAL ≈ 170, i.e. a TOA code of 18.4 ps and a floor of 5.3 ps. At BL+20 and 15 fC the
# v2.03 dies read 6.6–8.1 ps and the v2.01 die 5.8 ps — within a factor 1.5 of the floor, and by
# 22 fC the medians reach it; the revisions are indistinguishable here. A jitter *below* the
# floor (07-30 at 27 fC) means the TOA distribution has collapsed into one code and is a limit,
# not a measurement. 07-18 (no usable baseline) and 07-17 (too few pixels per offset for a
# median) are both absent from the figure; the sixth panel is left blank.

# %%
M = PTS[(PTS.n_hits_cal >= 500) & (PTS.acq != "v2.03_BARE4 07-18")].copy()
M = M.merge(NOI[["acq", "row", "col", "BL"]], on=["acq", "row", "col"], how="left")
M = M[~((M.row == 0) & (M.col == 14))]
M = Q.add_tdc_derived(M, toa="toa_mean_cal", toa_sd="toa_std_cal", tot="tot_mean_cal", tot_sd="tot_std_cal",
                      cal="cal_mean_cal", nhit="n_hits_cal")
M["offset"] = M.dac - M.BL
CALM = M.groupby("acq").cal_mean_cal.median()
print("median CAL per acquisition:\n" + CALM.round(1).to_string())
JIT = Q.pooled_at_offsets(M, ["acq"], ["jitter_ps", "tot_ns", "tot_mean_cal", "cal_mean_cal"], bl_col="BL",
                          charge_col="charge_fC", offsets=Q.OFFS)
JIT["jitter_median"] = [M[(M.acq == a) & (M.charge_fC == q) & (M.offset == o)].jitter_ps.median()
                        for a, q, o in zip(JIT.acq, JIT.charge_fC, JIT.offset)]
JIT.to_csv(Q.OUTDIR / "jul_jitter_by_offset.csv", index=False)
j20 = JIT[JIT.offset == 20]
print("\nTOA jitter at BL+20 [ps], mean over pixels\n" + j20.pivot_table(index="charge_fC", columns="acq", values="jitter_ps_mean").round(1).to_string())
print("\nthe same, median\n" + j20.pivot_table(index="charge_fC", columns="acq", values="jitter_median").round(1).to_string())
print("\npixels contributing\n" + j20.pivot_table(index="charge_fC", columns="acq", values="jitter_ps_n").to_string())

# %%
fig, axs = plt.subplots(2, 3, figsize=(22, 11))
PAN = ["v2.03_BARE1 07-15", "v2.03_BARE4 07-16", "v2.01_BARE10 07-22", "v2.03_BARE9 07-29", "v2.03_BARE9 07-30"]
axs.flat[5].axis("off")    # 07-17 (short day) and 07-18 (no baseline) are not shown
for k, acq in enumerate(PAN):
    ax = axs.flat[k]
    g = M[M.acq == acq]
    _, fl = Q.quantisation_floor_ps(g.cal_mean_cal.median())
    for ch in Q.JUL_CHARGES:
        sub = g[g.charge_fC == ch]
        if not len(sub):
            continue
        grp = sub.groupby("offset").jitter_ps
        med, q1, q3, n = grp.median(), grp.quantile(0.25), grp.quantile(0.75), grp.size()
        keep = n >= 3
        ax.plot(med.index[keep], med[keep], "-", marker=Q.CMK[ch], ms=5, lw=2.0, color=Q.CCOL[ch], label="%d fC" % ch)
        ax.fill_between(med.index[keep], q1[keep], q3[keep], color=Q.CCOL[ch], alpha=0.15, lw=0)
    ax.axhline(fl, color="black", ls="--", lw=2.0)
    ax.axvline(Q.OP_OFFSET, color=Q.C_MAG, ls="-.", lw=1.8)
    ax.set_yscale("log")
    ax.set_ylim(2.5, 400)
    ax.set_xlim(0, 260)
    ax.set_title("%s   floor %.1f ps" % (acq, fl), loc="left", fontsize=13)
    if k >= 3:
        ax.set_xlabel("threshold above baseline  [DAC LSB]")
    if k % 3 == 0:
        ax.set_ylabel("TOA jitter  [ps]")
    if k == 2:
        ax.legend(title="injected charge", ncol=2, fontsize=11, title_fontsize=11)
fig.tight_layout()
Q.fig_header(fig, "The money plot: TOA jitter against threshold, four dies of two revisions",
             tag="five pixels, median with inter-quartile band, at least three pixels per point; dashed = TDC quantisation floor",
             scale=0.62, campaign=Q.JUL_CAMPAIGN)
Q.save_fig(fig, "jul_fig6_money_plot")
plt.show()

# %% [markdown]
# ## 10. Time over threshold at the operating point
#
# TOT at BL+20 against charge, per die, in ns through each point's CAL. From 15 fC up all four
# dies agree to within ~0.15 ns. At 5–6 fC the TOT scatters by ~0.7 ns across the acquisitions
# with no revision ordering — the two `BARE9` days differ from each other by as much as either
# differs from the v2.01 die — because the low-charge TOT is set by where BL+20 cuts each pulse,
# and section 7 shows that operating point moved between days.

# %%
t20 = JIT[JIT.offset == 20]
print("TOT at BL+20 [ns], mean over pixels\n" + t20.pivot_table(index="charge_fC", columns="acq", values="tot_ns_mean").round(2).to_string())
t20[["acq", "charge_fC", "tot_ns_mean", "tot_ns_rms", "tot_ns_n", "tot_mean_cal_mean"]].to_csv(Q.OUTDIR / "jul_tot_at_bl20.csv", index=False)
fig, ax = plt.subplots(figsize=(11, 6.4))
for k, acq in enumerate(PAN):
    s = t20[t20.acq == acq].sort_values("charge_fC")
    ax.errorbar(s.charge_fC, s.tot_ns_mean, yerr=s.tot_ns_rms, fmt="-o", ms=7, lw=1.6, capsize=3,
                color=(Q.C_ORANGE if "v2.01" in acq else plt.cm.Blues(0.45 + 0.13 * k)), label=acq)
Q.charge_axis(ax, "x", Q.JUL_CHARGES)
ax.set_xlabel("injected charge  [fC]")
ax.set_ylabel("TOT at BL+20  [ns]")
ax.legend(fontsize=10)
fig.tight_layout()
Q.fig_header(fig, "Time over threshold at BL+20 against charge, four dies",
             tag="five pixels, mean with pixel RMS; CAL-windowed moments", scale=0.62, campaign=Q.JUL_CAMPAIGN)
Q.save_fig(fig, "jul_fig7_tot")
plt.show()

# %% [markdown]
# ## 11. Summary
#
# Everything pooled is over five pixels, (0,14) excluded, and "per die" averages the good days
# of that die.

# %%
b10 = DIE[DIE.chip == "ET2p01_BARE10"].iloc[0]
v3 = DIE[(DIE.revision == "v2.03") & (DIE.acquisition != "v2.03_BARE4 07-18")]
j15 = j20[j20.charge_fC == 15].set_index("acq").jitter_ps_mean
SUMMARY = pd.DataFrame([
    dict(n=1, quantity="pixel (0,14) 8-27 fC chord relative to the other pixels, all campaigns", value="%.2f (range %.2f-%.2f)" % (P14.ratio.median(), P14.ratio.min(), P14.ratio.max()),
         unit="ratio", label="DERIVED; six dies, two revisions; a compressive response, not a scale (k %.2f at 8 fC -> %.2f at 27 fC)" % (RESULTS["pixel_0_14_k_at_8"], RESULTS["pixel_0_14_k_at_27"])),
    dict(n=2, quantity="(0,14) noise: NW over array mean (four maps); FWHM over the column-14 control; S-curve sigma over the other pixels",
         value="%.2f; %.2f; %.2f" % (RESULTS["nw_maps_0_14_over_mean"], RESULTS["pixel_0_14_fwhm_ratio_col14"], RESULTS["pixel_0_14_sigma_ratio_median"]),
         unit="ratios", label="RAW maps exclude a 3x small-signal deficit only; FWHM and sigma say the front end's operating point differs"),
    dict(n=3, quantity="bias of a six-pixel pooled gain from including (0,14)", value="x %.3f" % RESULTS["pooled_gain_bias_6px"],
         unit="factor (excl/incl)", label="DERIVED; applies to notebook 03's absolute gains; cause of the pixel's behaviour unresolved"),
    dict(n=4, quantity="preamp gain, v2.03 dies (BARE1, BARE4, BARE9), five pixels", value="%.1f, %.1f, %.1f" % tuple(v203.gain.values),
         unit="DAC/fC", label="DERIVED; die-to-die spread %.0f %%" % (100 * v203.gain.std(ddof=1) / v203.gain.mean())),
    dict(n=5, quantity="preamp gain, v2.01 die (BARE10), five pixels", value="%.1f ± %.1f" % (b10.gain_dac_per_fc, b10.gain_sem),
         unit="DAC/fC", label="DERIVED"),
    dict(n=6, quantity="gain ratio v2.01 / v2.03, same bench", value="%.2f ± %.2f" % (r_gain, r_gain_err),
         unit="ratio", label="DERIVED; %.2f-%.2f if the DAC scale follows VRef, 1.00 if not; %.1f sigma (Student-t, 2 dof) from the former, or %.0f mV of unrecorded rail" % (PRED_BAND[0], PRED_BAND[1], RESULTS["rev_gain_gauss_sigma_equiv"], RESULTS["rev_gain_rail_equiv_mV"])),
    dict(n=7, quantity="S-curve sigma and noise FWHM, v2.01 / v2.03", value="%.2f, %.2f" % (r_sig, r_fwhm), unit="ratios", label="DERIVED; same two predictions; sigma near 1.0, FWHM on the VRef prediction"),
    dict(n=8, quantity="ENC, v2.03 dies / v2.01 die", value="%.0f, %.0f, %.0f / %.0f" % (*v203.ENC.values, b10.ENC_e),
         unit="e", label="DERIVED; under the assumed LSB the ratio is LSB-free"),
    dict(n=9, quantity="BARE9 8-15 fC chord, CERN 08-03 / bench 07-30, five pixels", value="%.3f (pixel spread %.3f)" % (RESULTS["bare9_gain_cern_over_bench"], RESULTS["bare9_gain_cern_over_bench_spread"]),
         unit="ratio", label="DERIVED; on the one chord the two ladders share; the die's response survived the move"),
    dict(n=10, quantity="BARE9 baseline steps, 07-29 -> 07-30 -> CERN 08-03", value="%+.0f, %+.0f" % (RESULTS["bare9_BL_shift_0730_minus_0729"], RESULTS["bare9_BL_shift_cern_minus_0730"]),
         unit="DAC", label="RAW; from the rail this would need ~%.1f V and a x%.0f slope change — not the rail; cause unknown, no register read-back" % (RESULTS["bare9_rail_for_240_dac_V"], RESULTS["bare9_gain_factor_for_that_rail"])),
    dict(n=11, quantity="TOA jitter at BL+20, 15 fC, per die (BARE1, BARE4, BARE10, BARE9 x2)", value=", ".join("%.1f" % j15[a] for a in PAN if a in j15.index),
         unit="ps", label="DERIVED; 1.1-1.5 x the %.1f ps floor; at the floor from 22 fC" % Q.quantisation_floor_ps(CALM.median())[1]),
    dict(n=12, quantity="BARE4 map baseline gradient, row 0 -> 15 and col 0 -> 15", value="%+.0f, %+.0f" % (
        MAP.groupby("row").bl_dac.mean().iloc[-1] - MAP.groupby("row").bl_dac.mean().iloc[0],
        MAP.groupby("col").bl_dac.mean().iloc[-1] - MAP.groupby("col").bl_dac.mean().iloc[0]),
         unit="DAC", label="RAW; noise width flat"),
])
print(SUMMARY.to_string(index=False))
SUMMARY.to_csv(Q.OUTDIR / "jul_summary_numbers.csv", index=False)
RESULTS["v203_gain_by_die"] = {k: float(v) for k, v in v203.gain.items()}
RESULTS["v201_gain"] = float(b10.gain_dac_per_fc)
RESULTS["enc_by_die"] = {k: float(v) for k, v in v203.ENC.items()}
RESULTS["enc_v201"] = float(b10.ENC_e)
with open(Q.OUTDIR / "jul_results.json", "w") as f:
    json.dump(RESULTS, f, indent=1)

# %% [markdown]
# ## 12. Reproduction ledger
#
# The July reduction has no hand-decode cross-check, so the ledger is internal: the per-day
# directory counts in `jul_et2p03_configs.csv` against the three tables, the README's totals,
# the decode-error count, and the `BARE4` autocalibration map against the same day's noise scans.

# %%
LED = pd.DataFrame(LEDGER)
LED["|diff|"] = LED["diff"].abs()
print(LED[["quantity", "recomputed", "published", "|diff|", "tol", "unit", "agrees"]].round(3).to_string(index=False))
LED.to_csv(Q.OUTDIR / "jul_reproduction_ledger.csv", index=False)
print("\n%d of %d checks agree within tolerance." % (int(LED.agrees.sum()), len(LED)))

# %% [markdown]
# ---
#
# ## 13. Caveats
#
# 1. **No register read-back, no rail, no temperature, for any day.** The absolute baselines,
#    and the ~240 DAC steps `BARE9` took between its three acquisitions, cannot be attributed.
#    Slopes, widths and everything referred to a same-day baseline are what this campaign can
#    compare; absolute DACs across days are not.
# 2. **One v2.01 die.** The revision comparison of section 6 has a die-to-die spread of ±8 % on the
#    v2.03 side and nothing on the v2.01 side; it is a tension with the DAC-follows-VRef reading,
#    not a rejection. The transfer-function measurement in the bench plan is the direct test.
# 3. **Pixel (0,14) is excluded from every pooled number here**, on the evidence of section 5,
#    and its cause is not resolved: the response compresses above ~8 fC on six dies, the noise
#    measurements point at the front end's operating point rather than the injector, and no
#    register read-back exists to say why. January's twelve-pixel data also put all of row 0 low,
#    so a position dependence is not confined to one corner; only same-pixel ratios between dies
#    are free of it.
# 6. **Every "gain" is a chord slope over a stated range.** The straight line is rejected on every
#    pixel (section 4), so slopes over different ranges are not interchangeable — which is why
#    section 7 uses the one chord its three acquisitions share, and why the per-pixel fit errors
#    are quoted with and without the √(χ²/ndf) scaling.
# 4. **07-18 has no usable noise scan** and 07-17 is a short day. `BARE4`'s numbers rest on 07-16
#    with 07-17 as the repeat.
# 5. **The DAC LSB is assumed** (0.3906 mV) wherever a millivolt or an electron appears, exactly as
#    in notebooks 01–03, and section 6 is the first measurement in the folder that bears on
#    whether it is the same number on both revisions.
