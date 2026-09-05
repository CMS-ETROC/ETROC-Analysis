"""Shared loaders, constants and fits for the ETROC2 charge-injection notebooks.

Every notebook in this folder imports this module. `01_bare_et2p01_tid_jan2024.ipynb` (the bare
ET2.01 die) and `02_hybrid_et2p01_cc_jan2024.ipynb` (the ET2.01+LGAD hybrid) go through
*identical* code paths: the same turn-off estimator, the same weighted gain fit, the same
shared-sigma S-curve fit, the same TOT log law and the same quantisation floor. Any difference
between the two sets of results is then a difference in the data, not a difference in the
analysis. `00_census_and_aux_data.ipynb` uses the loaders in the census section at the foot of
this file and none of the fits.

Conventions used everywhere:

* **RAW** means a column of, or a direct byte of, a staged source file.
  **DERIVED** means it was computed here. Nothing else is quoted.
* Thresholds are quoted as `offset = dac - BL`, the threshold above the autocalibrated
  baseline, because that is the axis the beam operating point (`threshold = BL + 20`) lives on
  and because the absolute DAC-to-mV scale is chip-revision dependent.
* No cuts are applied at any stage. Coverage varies (a low charge simply turns off below
  BL+20 on some pixels); every pooled quantity therefore carries its own `n`.

Nothing in this module touches a remote system: it reads only the staged files under `data/`.
"""

from __future__ import annotations

import hashlib
import os
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special

# ---------------------------------------------------------------- paths
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
BARE_DIR = DATA / "bare_et2p01_tid_jan2024"
HYB_DIR = DATA / "hybrid_et2p01_cc_jan2024"
CENSUS_DIR = DATA / "census"
FIGDIR = HERE / "figures"
OUTDIR = HERE / "outputs"
FIGDIR.mkdir(exist_ok=True)
OUTDIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- constants
# All four are code-verified against `tamalero/tamalero/ETROC.py` (DAC_min 600 mV,
# DAC_max 1000 mV at lines 64-66; `set_QInj` writing `QSel = C - 1` at lines 949-952),
# not taken from a citation chain.
LSB_MV = 400.0 / 1024          # 0.390625 mV per threshold-DAC LSB
CLK_NS = 3.125                 # TDC reference clock period, ns
E_PER_FC = 6241.509            # electrons per fC
QINJ_RANGE_FC = (1, 32)        # the internal injector's integer range

CHARGES = [5, 6, 8, 15, 22, 27]   # the six injected charges present in both campaigns
OFFS = [10, 15, 20]               # threshold offsets above baseline that are tabulated
BL_OFFSET = 5                     # `full_scurve_scan` starts the sweep at BL + 5
OP_OFFSET = 20                    # the beam pipeline's operating point, threshold = BL + 20
CAL_GUARD = (100, 220)            # the beam pipeline's CAL-code guard band (an absolute
                                  # sanity band on CAL, NOT a per-hit window)
CAL_WINDOW = 3                    # the repo pipeline's per-hit window, |CAL - cal_mode| <= 3
                                  # (core/path_finder.py, core/extract_events_by_path.py)
TOT_SLOPE = 2.0 - 1.0 / 32.0      # d(TOT_ns)/d(TOT_code): TOT_ns = (2*TOT - floor(TOT/32))*LSB

# Bytes-per-hit calibration of the sweep point-directory sizes, from the bare-die campaign:
# 19 exact line counts obtained server-side, residual RMS 61 hits over 3500-9400 hits.
BYTES_PER_HIT = 72.46

# ---------------------------------------------------------------- palette
# Six fixed categorical slots for charge (ascending), an ordinal blue ramp for threshold
# offset, and marker shape repeating the same identity so the figures survive colour-vision
# deficiency and greyscale printing.
CCOL = {5: "#2a78d6", 6: "#eb6834", 8: "#1baf7a", 15: "#eda100", 22: "#e87ba4", 27: "#008300"}
CMK = {5: "o", 6: "s", 8: "^", 15: "D", 22: "v", 27: "P"}
OCOL = {10: "#86b6ef", 15: "#3987e5", 20: "#184f95"}
OMK = {10: "^", 15: "s", 20: "o"}
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
C_YELLOW, C_MAG, C_GREEN = "#eda100", "#e87ba4", "#008300"
GREY = "#898781"

# ---------------------------------------------------------------- bare-campaign metadata
# Which day directory on EOS supplied which (dose, temperature) block. Established in the
# staged provenance note by matching pixel sets, charge sets and point counts.
BARE_DAY_BLOCK = {
    "2024-01-11": dict(dose=0, temp=12, kind="sweep+turnoff"),
    "2024-01-12": dict(dose=0, temp=-25, kind="turnoff"),
    "2024-01-15": dict(dose=200, temp=-25, kind="sweep+turnoff"),
    "2024-01-16": dict(dose=200, temp=-25, kind="sweep"),
    "2024-01-18": dict(dose=400, temp=-25, kind="sweep+turnoff"),
    "2024-01-19": dict(dose=400, temp=-25, kind="sweep"),
}
BARE_CONDITIONS = [(0, 12), (0, -25), (200, -25), (400, -25)]


def cond_label(dose, temp):
    return f"{int(dose)} MRad, {int(temp):+d} °C"


# ---------------------------------------------------------------- small utilities
def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_row(path):
    p = Path(path)
    return dict(file=p.name, bytes=p.stat().st_size, md5=md5(p))


def pooled(series):
    """Mean, pixel-to-pixel RMS, standard error and n over pixels, NaNs dropped."""
    s = pd.Series(series).dropna().astype(float)
    n = len(s)
    rms = s.std(ddof=1) if n > 1 else np.nan
    return dict(mean=s.mean() if n else np.nan, rms=rms,
                sem=rms / np.sqrt(n) if n > 1 else np.nan, n=n)


def save_fig(fig, name, dpi_png=110):
    """Write a figure as both PNG (for the notebook and quick viewing) and PDF (for a note)."""
    fig.savefig(FIGDIR / f"{name}.png", bbox_inches="tight", dpi=dpi_png)
    fig.savefig(FIGDIR / f"{name}.pdf", bbox_inches="tight")
    return FIGDIR / f"{name}.png"


# ---------------------------------------------------------------- derived TDC quantities
def add_tdc_derived(df, toa="toa_mean", toa_sd="toa_std", tot="tot_mean",
                    tot_sd="tot_std", cal="cal_mean", nhit=None):
    """Add the DERIVED per-point TDC columns, identically for both campaigns.

    `toa_ns  = TOA * 3.125 / CAL`
    `tot_ns  = (2*TOT - floor(TOT/32)) * 3.125 / CAL`
    `jitter_ps = TOA_std * 3.125 / CAL * 1000`
    `toa_lsb_ps = 3.125 / CAL * 1000`  — one TOA code in ps at this point's CAL
    `tot_std_ns = TOT_std * (2 - 1/32) * 3.125 / CAL`

    The `floor(./32)` is applied to the *mean* TOT code rather than per hit; that is a DERIVED
    approximation whose bias is below one TOT LSB. The spread is converted with the *derivative*
    of that law, `d(TOT_ns)/d(TOT_code) = 2 - 1/32 = TOT_SLOPE`, not with the factor 2 that the
    leading term alone would give: the two differ by 1.6 %, which is small but is a bias rather
    than a spread. CAL jitter is neglected in the conversions (it is at the few-per-mille level
    in both campaigns).
    """
    d = df.copy()
    d["toa_ns"] = d[toa] * CLK_NS / d[cal]
    d["toa_lsb_ps"] = CLK_NS / d[cal] * 1e3
    d["jitter_ps"] = d[toa_sd] * CLK_NS / d[cal] * 1e3
    d["tot_ns"] = (2 * d[tot] - np.floor(d[tot] / 32)) * CLK_NS / d[cal]
    d["tot_std_ns"] = d[tot_sd] * TOT_SLOPE * CLK_NS / d[cal]
    if nhit is not None and nhit in d:
        with np.errstate(divide="ignore", invalid="ignore"):
            d["jitter_ps_err"] = d.jitter_ps / np.sqrt(2 * d[nhit].clip(lower=1))
    return d


def quantisation_floor_ps(cal_code):
    """The TDC quantisation floor: one TOA LSB is 3.125 ns / CAL, and a uniformly filled bin
    has RMS LSB/sqrt(12). Returns (lsb_ps, floor_ps). A measured jitter at this value is a
    limit, not a resolution."""
    lsb = CLK_NS / float(cal_code) * 1e3
    return lsb, lsb / np.sqrt(12.0)


# ---------------------------------------------------------------- loaders: bare campaign
def load_bare_summary():
    """RAW: `ET2p01_W36_6_TDC_summary_allMRad.feather`, one row per
    (dose, temp, pixel, charge, threshold). 22 055 x 12, all RAW columns."""
    return pd.read_feather(BARE_DIR / "ET2p01_W36_6_TDC_summary_allMRad.feather")


def load_bare_probes():
    """RAW: every probe point of the discriminator binary search, parsed from the 288
    `FPGA_Data.dat` files (also staged verbatim in `fpga_turnoff_dat.tar.gz`).

    Column map, verified against the search's own behaviour: index 3 = TDC-hit counter,
    index 4 = `header_max` (the L1A denominator, ~11 223), index 5 = trigger-bit counter,
    index 6 = the threshold DAC. Adds the DERIVED efficiency fractions.
    """
    p = pd.read_csv(BARE_DIR / "fpga_turnoff_probes.csv")
    p["frac"] = p.trig_count / p.header_max
    p["frac_tdc"] = p.tdc_count / p.header_max
    p["frac_c"] = p.frac.clip(upper=1.0)
    p["dose"] = p.day.map(lambda d: BARE_DAY_BLOCK[d]["dose"])
    p["temp"] = p.day.map(lambda d: BARE_DAY_BLOCK[d]["temp"])
    return p


def load_bare_probes_from_raw_tar():
    """Re-parse the probes straight from the staged raw `FPGA_Data.dat` bytes.

    Used once in notebook 01 to prove that `fpga_turnoff_probes.csv` is a faithful
    transcription of the raw files rather than something that has to be taken on trust.
    """
    import io
    import re
    rows = []
    with tarfile.open(BARE_DIR / "fpga_turnoff_dat.tar.gz") as tf:
        for m in tf.getmembers():
            if not m.name.endswith(".dat") or os.path.basename(m.name).startswith("._"):
                continue
            day, name = os.path.basename(m.name)[:-4].split("__", 1)
            g = re.search(r"Pixel_C(\d+)_R(\d+)_QInj_(\d+)", name)
            col, row, q = int(g.group(1)), int(g.group(2)), int(g.group(3))
            a = np.loadtxt(io.BytesIO(tf.extractfile(m).read()), delimiter=",", ndmin=2)
            for r in a:
                rows.append((day, row, col, q, int(r[3]), int(r[4]), int(r[5]), int(r[6])))
    return pd.DataFrame(rows, columns=["day", "row", "col", "charge",
                                       "tdc_count", "header_max", "trig_count", "dac"])


def load_bare_sweep_dirs():
    """RAW: one row per scanned sweep point directory (day, pixel, charge, DAC, size in
    bytes), from a single `xrdfs ls -l` per day directory. 23 416 rows. The size is what makes
    the hit count and the censoring flag recoverable without touching the 16 GB of `.nem`."""
    d = pd.read_csv(BARE_DIR / "sweep_point_dirs.csv.gz")
    d["dose"] = d.day.map(lambda x: BARE_DAY_BLOCK[x]["dose"])
    d["temp"] = d.day.map(lambda x: BARE_DAY_BLOCK[x]["temp"])
    d["nhit_est"] = (d.dir_bytes / BYTES_PER_HIT).round().astype(int)
    return d


def load_bare_blnw():
    """RAW: the autocalibration baseline and noise-width maps (16 x 16 pixels) for the four
    days on EOS that carry one."""
    return pd.read_csv(BARE_DIR / "blnw_maps.csv")


# ---------------------------------------------------------------- loaders: hybrid campaign
def load_hybrid_points():
    """RAW + DERIVED: one row per (config, pixel, charge, threshold) of the climate-chamber
    campaign, 13 281 x 22, with the per-point TOA/TOT/CAL mean and std in raw codes."""
    return pd.read_feather(HYB_DIR / "cc_points.feather")


def load_hybrid_probes():
    """RAW: every probe point of the hybrid binary searches, same columns as the bare set."""
    p = pd.read_csv(HYB_DIR / "cc_fpga_probes.csv.gz")
    p = p[(p.kind == "TurnOff") & (p.dac >= 0)].copy()
    p["frac"] = p.trig_count / p.header_max
    p["frac_tdc"] = p.tdc_count / p.header_max
    p["frac_c"] = p.frac.clip(upper=1.0)
    return p


def load_hybrid_turnoff():
    return pd.read_csv(HYB_DIR / "cc_turnoff.csv")


def load_hybrid_configs():
    return pd.read_csv(HYB_DIR / "cc_configs.csv").set_index("config")


def load_hybrid_blnw():
    return pd.read_csv(HYB_DIR / "cc_blnw.csv.gz")


def load_hybrid_noiseonly():
    return pd.read_csv(HYB_DIR / "cc_noiseonly.csv")


def load_hybrid_tdc_dirs():
    return pd.read_csv(HYB_DIR / "cc_tdc_dirs.csv.gz")


def load_hybrid_aux_power():
    return pd.read_csv(HYB_DIR / "table_aux_power_lv.csv")


def load_hybrid_summary_numbers():
    import json
    with open(HYB_DIR / "summary_numbers.json") as fh:
        return json.load(fh)


# ---------------------------------------------------------------- the turn-off estimators
def turnoff_50pct(dac, frac, method="interp"):
    """The 50 % point of a discriminator binary search, from its own probe points.

    The search sets `a = DAC` when the counter is at or above `header_max/2` and `b = DAC`
    otherwise, so it converges on a genuine 50 % crossing with a real denominator.

    `method="interp"` (the default, used for both campaigns): take the highest probed DAC still
    at or above 50 %, the lowest probed DAC above it that is below 50 %, and linearly
    interpolate the efficiency to 0.5. Sub-LSB, and the estimator the bare-die tables were
    built with.

    `method="maxdac"`: simply the highest probed DAC at or above 50 %. Coarser by construction
    -- it lands about half a probe step low -- and it is the estimator the hybrid tables were
    built with. Both are provided so the two published tables can be reproduced exactly and
    the difference between them shown rather than argued about.
    """
    d = np.asarray(dac, float)
    f = np.asarray(frac, float)
    order = np.argsort(d)
    d, f = d[order], f[order]
    above = np.where(f >= 0.5)[0]
    if len(above) == 0:
        return np.nan
    i = above[-1]
    if method == "maxdac":
        return float(d[i])
    hi = np.where((d > d[i]) & (f < 0.5))[0]
    if len(hi) == 0:
        return float(d[i])
    j = hi[int(np.argmin(d[hi]))]
    return float(d[i] + (0.5 - f[i]) * (d[j] - d[i]) / (f[j] - f[i]))


def turnoff_table(probes, keys, method="interp"):
    """Both turn-off estimators for every (…, pixel, charge) group of a probe table.

    Returns `trig_turnoff_dac` (the discriminator / trigger-bit 50 % point -- a proper
    efficiency, but it saturates at a charge-independent floor below 8 fC on both campaigns)
    and `tdc_bs_turnoff_dac` (the same crossing read off the TDC counter of the same search).
    """
    out = []
    for k, g in probes.groupby(list(keys)):
        rec = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        rec["trig_turnoff_dac"] = turnoff_50pct(g.dac, g.frac, method)
        rec["tdc_bs_turnoff_dac"] = turnoff_50pct(g.dac, g.frac_tdc, method)
        rec["n_probe"] = len(g)
        rec["header_max"] = float(g.header_max.median())
        out.append(rec)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- weighted line fit
def wls(x, y, sy):
    """Weighted least-squares straight-line fit. Returns
    (slope, intercept, slope_err, intercept_err, chi2, residuals)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = 1.0 / np.asarray(sy, float) ** 2
    A = np.vstack([x, np.ones_like(x)]).T
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    beta = cov @ (A.T @ (w * y))
    res = y - A @ beta
    chi2 = float((res ** 2 * w).sum())
    return beta[0], beta[1], np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1]), chi2, res


def gain_fit(g, bl, q_lo=8, q_hi=27, sy=0.5, off_target=OP_OFFSET,
             turnoff_col="trig_turnoff_dac", charge_col="charge"):
    """The **preamp gain** of one pixel -- the charge-to-threshold response in DAC LSB per fC:
    a weighted straight line through discriminator turn-off DAC against injected charge over
    `q_lo`-`q_hi` fC.

    "Gain" everywhere in this folder means this and only this. **No sensor multiplication is
    involved in any measurement here**, including notebook 02's hybrid: the charge is injected by
    the chip's own internal injector directly at the preamplifier input, so an LGAD bonded to the
    pad contributes its capacitance to the input node and nothing else. Nothing in this folder
    measures or depends on LGAD gain.

    y-error 0.5 DAC, because the binary search brackets the crossing to 1 DAC.

    The fit range starts at 8 fC deliberately: below it the discriminator turn-off saturates
    at a charge-independent floor on both campaigns, so its slope there is not a gain.
    Returns the slope in DAC/fC and mV/fC, the intercept referred to the baseline (a pure
    proportionality would put it at 0), chi2/ndf, and the charge that just reaches BL+`off_target`.
    """
    h = g[(g[charge_col] >= q_lo) & (g[charge_col] <= q_hi) & g[turnoff_col].notna()]
    h = h.sort_values(charge_col)
    if len(h) < 3:
        return {}
    s, i, se, ie, chi2, res = wls(h[charge_col].values, h[turnoff_col].values,
                                  np.full(len(h), sy))
    return dict(gain_dac_per_fc=s, gain_dac_per_fc_err=se,
                gain_mv_per_fc=s * LSB_MV, gain_mv_per_fc_err=se * LSB_MV,
                intercept_dac=i, intercept_dac_err=ie, intercept_offset_dac=i - bl,
                chi2_ndf=chi2 / max(len(h) - 2, 1), npts=len(h),
                Qmin_at_off_fC=(bl + off_target - i) / s if s > 0 else np.nan)


# ---------------------------------------------------------------- S-curve: shared sigma
SIG_GRID = np.arange(0.25, 4.0001, 0.01)
_PROF_CACHE = {}


def _curve_profile(dac, frac):
    """For every sigma on the grid, the least-squares SSE of one Gaussian-CDF S-curve,
    minimised over its own free 50 % point mu (coarse 0.25-DAC scan, then a 0.01-DAC refine).
    Vectorised; no optimiser, so it cannot fail to converge."""
    key = (dac.tobytes(), frac.tobytes())
    if key in _PROF_CACHE:
        return _PROF_CACHE[key]
    s2 = SIG_GRID[:, None, None] * np.sqrt(2)
    mu_c = np.arange(dac.min() - 2, dac.max() + 2.001, 0.25)
    sse = ((frac[None, None, :] - 0.5 * special.erfc((dac[None, None, :] - mu_c[None, :, None]) / s2)) ** 2).sum(-1)
    m0 = mu_c[sse.argmin(1)]
    mu_f = m0[:, None] + np.linspace(-0.3, 0.3, 61)[None, :]
    sse_f = ((frac[None, None, :] - 0.5 * special.erfc((dac[None, None, :] - mu_f[:, :, None]) / s2)) ** 2).sum(-1)
    k = sse_f.argmin(1)
    out = (sse_f[np.arange(len(SIG_GRID)), k], mu_f[np.arange(len(SIG_GRID)), k])
    _PROF_CACHE[key] = out
    return out


def scurve_sigma(curves, boot=200, seed=1):
    """Fit a stack of S-curves with ONE shared Gaussian width sigma and a free 50 % point per
    curve, profiled out. Returns (sigma, bootstrap sd over curves, per-curve mu, SSE profile).

    Why not the simpler recipe: aligning each curve on its own fitted 50 % point and then
    fitting the stack biases sigma low, because the alignment absorbs part of the spread. The
    two recipes are therefore not interchangeable, and this one is applied to both campaigns.
    """
    profs = np.array([_curve_profile(d, f)[0] for d, f in curves])
    mus_all = np.array([_curve_profile(d, f)[1] for d, f in curves])
    kbest = profs.sum(0).argmin()
    sbest = SIG_GRID[kbest]
    mus = list(mus_all[:, kbest])
    sd = np.nan
    if boot:
        rng = np.random.default_rng(seed)
        bs = [SIG_GRID[profs[rng.integers(0, len(curves), len(curves))].sum(0).argmin()]
              for _ in range(boot)]
        sd = float(np.std(bs))
    return sbest, sd, mus, profs.sum(0)


def curves_from(df, keys=("row", "col", "charge"), daccol="dac", fraccol="frac_c"):
    """Split a probe table into (dac, frac) S-curves, keeping only groups whose probes
    actually straddle the edge: at least one point *above* 2 % efficiency and at least one
    *below* 98 %, which is what the two conditions below test."""
    out, meta = [], []
    for k, g in df.groupby(list(keys)):
        g = g.sort_values(daccol)
        if (g[fraccol] > 0.02).sum() == 0 or (g[fraccol] < 0.98).sum() == 0:
            continue
        out.append((g[daccol].values.astype(float), g[fraccol].values.astype(float)))
        meta.append(k)
    return out, meta


def gauss_cdf(dac, mu, sigma):
    """The Gaussian-CDF S-curve, falling with threshold: efficiency 1 well below `mu`."""
    return 0.5 * special.erfc((np.asarray(dac, float) - mu) / (sigma * np.sqrt(2)))


def enc_electrons(sigma_dac, gain_dac_per_fc):
    """ENC = sigma / preamp gain, expressed in electrons. A ratio of two measured quantities, so it
    moves if either the noise edge or the volts-per-fC moves -- which of the two did it is
    always worth stating alongside."""
    return sigma_dac / gain_dac_per_fc * E_PER_FC


# ---------------------------------------------------------------- TOT log law
def logfit(q, y, sy):
    """Weighted least squares of `TOT = a + b ln(Q/fC)`. Returns
    (a, b, a_err, b_err, chi2/ndf, residuals). A linear fit in ln Q, so it is exact, not
    iterative. A large chi2/ndf here is a statement about the shape, not about convergence."""
    q = np.asarray(q, float)
    y = np.asarray(y, float)
    sy = np.asarray(sy, float)
    b, a, be, ae, chi2, res = wls(np.log(q), y, sy)
    return a, b, ae, be, chi2 / max(len(q) - 2, 1), res


# ---------------------------------------------------------------- pooling helper
def pooled_at_offsets(df, group_keys, value_cols, bl_col="BL", offsets=OFFS,
                      charge_col="charge", tol=0.0):
    """Pick, for each (group, charge) and each threshold offset in `offsets`, the point at
    exactly `BL + offset`, then pool the requested columns over the group.

    Coverage is not assumed: a charge whose turn-off sits below BL+20 simply has no point
    there, so each pooled value reports its own `n`.
    """
    rows = []
    for off in offsets:
        sel = df[np.isclose(df[bl_col] + off, df.dac, atol=tol)]
        for k, g in sel.groupby(list(group_keys) + [charge_col]):
            rec = dict(zip(list(group_keys) + [charge_col], k))
            rec["offset"] = off
            for c in value_cols:
                p = pooled(g[c])
                rec[f"{c}_mean"], rec[f"{c}_rms"], rec[f"{c}_n"] = p["mean"], p["rms"], p["n"]
            rows.append(rec)
    return pd.DataFrame(rows)


def check(name, got, want, tol, unit="", store=None):
    """Compare a recomputed number against a published one and print the verdict.

    Every headline number in both notebooks goes through this, so a disagreement is visible in
    the executed output rather than buried in prose.
    """
    ok = (want is not None) and abs(got - want) <= tol
    mark = "OK  " if ok else "DIFF"
    print(f"[{mark}] {name:<52s} recomputed {got:10.4f}  published {want:10.4f}  "
          f"|d| {abs(got - want):8.4f} (tol {tol:g}) {unit}")
    if store is not None:
        store.append(dict(quantity=name, recomputed=got, published=want,
                          diff=got - want, tol=tol, unit=unit, agrees=bool(ok)))
    return ok


def charge_axis(ax, axis="x", charges=None):
    """Put a log axis on the six injected charges with clean labels.

    Matplotlib's default log locator adds minor labels such as `2 x 10^1` between the charge
    ticks; this removes them so the axis reads as the six values that were actually injected.
    """
    import matplotlib.ticker as mt
    charges = charges or CHARGES
    a = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    a.set_major_locator(mt.FixedLocator(charges))
    a.set_major_formatter(mt.FixedFormatter([str(c) for c in charges]))
    a.set_minor_locator(mt.NullLocator())
    a.set_minor_formatter(mt.NullFormatter())
    return ax


# ---------------------------------------------------------------- hybrid auxiliary logs
def load_hybrid_lv():
    """RAW: the low-voltage rails from `PowerHistory_v2.sqlite` (two files, table `power_v2`),
    one sample per rail per minute out of the 233 576 recorded at ~5 s. Unit strings (`1.279V`)
    are stripped at extraction; samples below 0.5 V are power-down excursions and are dropped
    where a median is taken, never silently from the series itself."""
    d = pd.read_csv(HYB_DIR / "aux_lv_timeseries.csv.gz", parse_dates=["timestamp"])
    d["P_mW"] = d.V * d.I_A * 1e3
    return d


def load_hybrid_bl_history():
    """RAW, aggregated: `BaselineHistory_CC_Jan2024_CERN.sqlite`, 33 170 autocalibration rows,
    averaged per (note, minute) over all pixels and over the nine primary-config pixels. The
    sqlite has no HV and no temperature column; its `note` records the run and a coarse thermal
    condition, and nothing else."""
    return pd.read_csv(HYB_DIR / "aux_baseline_history.csv.gz", parse_dates=["timestamp"])


# NOTE: scan windows are NOT recoverable. The directory modification times in both the EOS
# original and the LPC scratch copy are upload/copy times (2024-01-27 and 2026-09 respectively),
# not acquisition times, and no scan file carries a wall-clock stamp. The only real timestamps in
# this campaign are in the two sqlite logs, which is why the auxiliary figures anchor on those.


# ---------------------------------------------------------------- figure header
BARE_CAMPAIGN = "CERN TID campaign, January 2024"
HYBRID_CAMPAIGN = "CERN climate chamber, January 2024"
CENSUS_CAMPAIGN = "Data census, 2026-09"


def fig_header(fig, name, tag=None, scale=0.62, campaign=None, y=1.012, pad=6):
    """The repo header on a zero-height axis spanning the figure width.

    A thin wrapper over `qinj_style.cms_header` so that each notebook can name its own campaign
    in the header's right block without editing `qinj_style.py`. `qinj_style.py` is a new file
    written for this folder, modelled on the repo's `TestBeam/BeamTestHelpers/plotting_basic.py`
    (mplhep CMS style, `hep.cms.text(loc=0)` at the header left, the figure name via
    `ax.set_title(loc="right")`); its experiment label, "ETL ETROC Charge Injection", is this
    folder's own member of that family rather than the repo's "ETL ETROC Test Beam".
    Call AFTER `tight_layout`.
    """
    from qinj_style import cms_header
    hax = fig.add_axes([0.0, y, 1.0, 1e-6])
    hax.set_xticks([])
    hax.set_yticks([])
    hax.patch.set_visible(False)
    hax.grid(False)
    for sp in hax.spines.values():
        sp.set_visible(False)
    kw = {} if campaign is None else dict(campaign=campaign)
    cms_header(hax, name, tag=tag, scale=scale, pad=pad, **kw)
    return hax


# ---------------------------------------------------------------- the census (notebook 00)
# Notebook 00 is an inventory, not a measurement, so it uses only the loaders below. The two
# census tables are transcriptions of the two written census documents; the three `aux_tid_*`
# files are extractions from the CERN Jan-2024 TID sqlite logs on LPC EOS; the two
# `aux_et2p03_*` files are copies of the ET2.03 reduction's own aux tables, byte for byte.
# `data/census/PROVENANCE.md` says where each one came from and how.

# Which calendar day of the CERN Jan-2024 TID campaign sat at which (dose, plate temperature).
# The six scan days are the ones `BARE_DAY_BLOCK` already carries, and they are re-derived
# independently in notebook 00 from the dose tag written into every scan-directory name on EOS.
# The four remaining days carry an auxiliary log but no scan of the primary chip; their
# condition is what the surrounding days establish, and it is labelled as such.
TID_LV_DAY = {
    "2024-01-10": dict(dose=0, temp=12, scans="17 TurnOff + 16 NoiseOnly + 12 TurnOn",
                       cond_source="untagged; by continuity with 01-11"),
    "2024-01-11": dict(dose=0, temp=12, scans="7856 TDC + 72 TurnOff", cond_source="directory names"),
    "2024-01-12": dict(dose=0, temp=-25, scans="72 TurnOff", cond_source="directory names"),
    "2024-01-13": dict(dose=0, temp=-25, scans="none", cond_source="between-days"),
    "2024-01-14": dict(dose=0, temp=-25, scans="none", cond_source="between-days"),
    "2024-01-15": dict(dose=200, temp=-25, scans="4535 TDC + 72 TurnOff", cond_source="directory names"),
    "2024-01-16": dict(dose=200, temp=-25, scans="3314 TDC", cond_source="directory names"),
    "2024-01-17": dict(dose=200, temp=-25, scans="none", cond_source="between-days"),
    "2024-01-18": dict(dose=400, temp=-25, scans="3481 TDC + 72 TurnOff", cond_source="directory names"),
    "2024-01-19": dict(dose=400, temp=-25, scans="4230 TDC", cond_source="directory names"),
}


def load_census():
    """The data-set census, one row per data set. A transcription of the two written census
    documents, with the columns the notebook plots. Every cell is a statement about what a
    scan found; `unknown` means no evidence was found and is never a guess."""
    d = pd.read_csv(CENSUS_DIR / "qinj_census.csv", dtype=str).fillna("")
    d["size_GB_num"] = pd.to_numeric(d.size_GB, errors="coerce")
    d["date_start_dt"] = pd.to_datetime(d.date_start)
    d["date_end_dt"] = pd.to_datetime(d.date_end)
    return d


def load_aux_matrix():
    """The auxiliary-data matrix: one row per campaign, one column per kind of monitoring."""
    return pd.read_csv(CENSUS_DIR / "qinj_aux_matrix.csv", dtype=str).fillna("")


AUX_KINDS = ["lv_supply_log", "blnw_maps", "vtemp", "hv_log", "vref_record",
             "chamber_log", "register_dump", "logbook"]
AUX_LABELS = {"lv_supply_log": "LV supply log", "blnw_maps": "BL/NW maps", "vtemp": "VTemp",
              "hv_log": "HV log", "vref_record": "VRef record", "chamber_log": "chamber log",
              "register_dump": "register dump", "logbook": "logbook"}
# `logged-missing` is the sixth value and it is not a synonym for any of the others: the campaign
# demonstrably wrote the log (an analysis notebook reads the file by name) and the file has not
# been located on any storage this census covers. It is neither `no` nor `unknown`.
AUX_VALUES = ["yes", "partial", "logged-missing", "no", "na", "unknown"]
AUX_COLOUR = {"yes": "#1baf7a", "partial": "#eda100", "logged-missing": "#c0567a",
              "no": "#e0e0e0", "na": "#f7f7f7", "unknown": "#c0c8d8"}


def load_tid_lv():
    """RAW, downsampled: the analog and digital low-voltage rails of the CERN Jan-2024 TID
    campaign, from the three `PowerHistory_v2.sqlite` files in that tree on LPC EOS. One
    sample per rail per minute out of the 231 120 recorded at about 5 s. Unit strings
    (`1.238V`, `0.3394A`) are stripped at extraction. Adds the day's dose and plate
    temperature from `TID_LV_DAY`, and the instantaneous power."""
    d = pd.read_csv(CENSUS_DIR / "aux_tid_lv_timeseries.csv.gz", parse_dates=["timestamp"])
    d["day"] = d.timestamp.dt.strftime("%Y-%m-%d")
    d["dose_MRad"] = d.day.map(lambda x: TID_LV_DAY.get(x, {}).get("dose", np.nan))
    d["temp_C"] = d.day.map(lambda x: TID_LV_DAY.get(x, {}).get("temp", np.nan))
    d["P_mW"] = d.V * d.I_A * 1e3
    return d


def load_tid_lv_actions():
    """RAW: the `actions_v2` Start/Stop rows of the same three files. These are what date the
    stretches at ~0 V, so a per-day median can be taken over powered samples only."""
    return pd.read_csv(CENSUS_DIR / "aux_tid_lv_actions.csv", parse_dates=["timestamp"])


def load_tid_bl_history():
    """RAW: `BaselineHistory_TID_Jan2024_CERN.sqlite`, 1 024 rows — two full-array
    autocalibration repeats on 2024-01-14, and the whole of that campaign's baseline log."""
    return pd.read_csv(CENSUS_DIR / "aux_tid_baseline_history.csv", parse_dates=["timestamp"])


def load_et2p03_chipconfig():
    """RAW, decoded: the three ET2.03 August-2024 `ChipConfig/*.pckl` register read-backs, as
    decoded fields plus the raw `PeriCfg0..31` bytes. Copied byte for byte from the ET2.03
    reduction; md5 `5aef7758fd020bddf206430880301a20`."""
    return pd.read_csv(CENSUS_DIR / "aux_et2p03_chipconfig.csv")


def load_et2p03_blnw():
    """RAW: the ET2.03 `BARE9` autocalibration baseline and noise-width maps of 2024-08-03,
    all 256 pixels. md5 `96da68fed9fef08cf920002f4bfa5023`."""
    return pd.read_csv(CENSUS_DIR / "aux_et2p03_blnw.csv")


# ---------------------------------------------------------------- ET2.03 bare-die TID campaign
# Notebook 03. The CERN X-ray August-2024 total-ionising-dose campaign on two ETROC2 **v2.03**
# bare dies, `ET2p03_BARE9` and `ET2p03_BARE7`. The staged tables are the reduction described in
# `data/bare_et2p03_tid_aug2024/DECODE_README.md`, built to the same per-point definitions as the
# two January-2024 campaigns above so that the three can be differenced directly.
#
# The additions below are strictly additive: nothing above this line changed when they were
# written, so notebooks 00, 01 and 02 are unaffected.
ET2P03_DIR = DATA / "bare_et2p03_tid_aug2024"
ET2P03_CAMPAIGN = "CERN X-ray TID campaign, August 2024"
XCAMP_CAMPAIGN = "ET2.01 and ET2.03, January and August 2024"

# The five charges and the six pixels this campaign scanned, at every one of its conditions.
ET2P03_CHARGES = [6, 8, 10, 15, 22]
ET2P03_PIXELS = [(0, 0), (0, 14), (3, 0), (3, 14), (12, 0), (12, 14)]
# The gain fit range. 8 fC is the same lower edge notebooks 01 and 02 use, so the three campaigns'
# gains are slopes over the same charges; 22 fC is simply the highest charge this campaign
# injected (01 and 02 reach 27 fC).
ET2P03_QLO, ET2P03_QHI = 8, 22

# ---------------------------------------------------------------- ET2.03 rails: SUPPLY, not chip
# `va_V` and `vd_V` in every ET2.03 table are the numbers set on the **bench power supply**, not
# the voltages at the chip. The campaign's own logbook says so, in one line, and it is staged
# verbatim as `data/bare_et2p03_tid_aug2024/info_BARE{7,9}_voltage_scan.txt`:
#
#     Nominal Analog 1.2V = 1.4V on power supply
#     Nominal Digital 1.2V = 1.25V on power supply
#
# So the intended operating point on BOTH rails is 1.2 V at the chip, and the difference is the IR
# drop over the cables and the board: 0.20 V on the analog rail and 0.05 V on the digital one at
# the currents of that measurement. Subtracting a constant drop is an APPROXIMATION -- the drop is
# I*R and the rail currents were never logged on this campaign -- so every at-chip number derived
# from it is marked "~" and must not be read at better than the ~50 mV level.
VA_SUPPLY_DROP_V = 0.20        # analog: supply 1.40 V <-> 1.2 V nominal at the chip
VD_SUPPLY_DROP_V = 0.05        # digital: supply 1.25 V <-> 1.2 V nominal at the chip
VA_NOMINAL_SUPPLY_V = 1.40     # the ONE supply setting that is nominal at the chip
VD_NOMINAL_SUPPLY_V = 1.25


def va_at_chip(va_supply):
    """Approximate analog voltage AT THE CHIP for a supply setting. See VA_SUPPLY_DROP_V."""
    return np.asarray(va_supply, dtype=float) - VA_SUPPLY_DROP_V


def vd_at_chip(vd_supply):
    """Approximate digital voltage AT THE CHIP for a supply setting. See VD_SUPPLY_DROP_V."""
    return np.asarray(vd_supply, dtype=float) - VD_SUPPLY_DROP_V


def va_label(va_supply, chip=True):
    """`VA 1.20 V (supply, ~1.00 V at chip)` -- the label form used on every figure and table."""
    return ("VA %.2f V (supply, ~%.2f V at chip)" % (va_supply, float(va_at_chip(va_supply)))
            if chip else "VA %.2f V (supply)" % va_supply)


# The four analog settings of this campaign, supply against approximate at-chip value.
VA_SUPPLY_TABLE = [(0.85, 0.65), (1.00, 0.80), (1.20, 1.00), (1.40, 1.20)]

# 10 fC exists only in this campaign, so it needs its own categorical slot. Adding a key to the
# two palette dicts cannot affect notebooks 00-02: they index them, never iterate them.
CCOL[10] = "#7b4fc9"
CMK[10] = "X"


def load_et2p03_points(union=True):
    """RAW + DERIVED: one row per decoded point directory of the ET2.03 campaign.

    `union=True` (the default, and the only one any analysis should use for a dose comparison)
    reads `tid_et2p03_points_all.feather`, the union of the `TID_August` working copy and the
    `X9` master copy, 29 922 x 43 — the only table that carries the **200 and 400 MRad money
    plots**. `union=False` reads the working copy alone, where the dosed data are the 10 fC rail
    scans only; it is kept staged so that the difference between the two is demonstrable rather
    than asserted.

    **`va_V` and `vd_V` are POWER-SUPPLY settings, not voltages at the chip.** Nominal 1.2 V at the
    chip is 1.40 V on the analog supply and 1.25 V on the digital one (the campaign logbook,
    staged as `info_BARE{7,9}_voltage_scan.txt`); see `VA_SUPPLY_DROP_V` above.

    Two families of row live in this table and must never be mixed: `scan_family == 'money_plot'`
    is a threshold-scan point at one pixel, and `scan_family == 'voltage_scan'` is one fixed
    (unrecorded) threshold acquisition pooled over sixteen pixels with `dac` NaN.

    Each point carries TWO sets of moments. `toa_mean`, `toa_std`, `tot_*`, `cal_*` are over every
    decoded hit, exactly as the two January feathers define them. The `*_cal` columns
    (`n_hits_cal`, `frac_rejected`, `cal_mode`, `toa_mean_cal`, `toa_std_cal`, `tot_mean_cal`,
    `tot_std_cal`, `cal_mean_cal`, `cal_std_cal`) are the same moments over the hits inside the
    per-hit CAL window `|CAL - cal_mode| <= CAL_WINDOW`, the repository test-beam pipeline's own
    criterion. Notebook 03 quotes the windowed set: one TOA code is 3.125 ns / CAL, so a hit with
    a wrong CAL carries a wrong TOA. See `DECODE_README.md` and notebook 03 section 4.4.
    """
    name = "tid_et2p03_points_all.feather" if union else "tid_et2p03_points.feather"
    return pd.read_feather(ET2P03_DIR / name)


def load_et2p03_turnoff(union=True):
    """RAW: one row per (config, pixel, charge) binary search — 360 of them.

    `turnoff_dac` is the campaign's own definition (the DAC on the last line of `FPGA_Data.dat`);
    `search_trace` is the whole search as `dac:triggerbits;...`, which is what lets the same
    50 % estimator used in notebooks 01 and 02 be applied here.
    """
    name = "tid_et2p03_turnoff_all.csv" if union else "tid_et2p03_turnoff.csv"
    return pd.read_csv(ET2P03_DIR / name)


def load_et2p03_noiseonly(union=True):
    """RAW: the `quietCB` noise scans, one row per (config, pixel), 72 of them.

    `scan_trace` is a *dense* one-DAC-step scan of the trigger-bit counter across the noise peak,
    so unlike the two January campaigns this one carries an in-configuration baseline and noise
    measurement at **every** condition, not only where an autocalibration map survived.
    """
    name = "tid_et2p03_noiseonly_all.csv" if union else "tid_et2p03_noiseonly.csv"
    return pd.read_csv(ET2P03_DIR / name)


def load_et2p03_configs(union=True):
    """RAW + DERIVED: the conditions grid, one row per run or voltage-scan sub-run."""
    name = "tid_et2p03_configs_all.csv" if union else "tid_et2p03_configs.csv"
    return pd.read_csv(ET2P03_DIR / name)


def load_et2p03_tid_blnw():
    """RAW: the `ET2p03_BARE9` autocalibration baseline and noise-width maps of 2024-08-03, all
    256 pixels. Byte-identical to `data/census/aux_et2p03_blnw.csv`
    (md5 `96da68fed9fef08cf920002f4bfa5023`), staged in both places so each notebook's data
    directory is self-contained."""
    return pd.read_csv(ET2P03_DIR / "tid_et2p03_blnw.csv")


def load_et2p03_tid_chipconfig():
    """RAW, decoded: the three `ChipConfig/*.pckl` register read-backs. Byte-identical to
    `data/census/aux_et2p03_chipconfig.csv` (md5 `5aef7758fd020bddf206430880301a20`)."""
    return pd.read_csv(ET2P03_DIR / "tid_et2p03_chipconfig.csv")


def parse_trace(trace):
    """`'dac:count;dac:count;...'` -> a DAC-sorted (dac, count) float array pair.

    Both the binary-search `search_trace` and the noise-scan `scan_trace` are written in this
    form by the reduction, so one parser serves both.
    """
    d, v = [], []
    for tok in str(trace).split(";"):
        if ":" not in tok:
            continue
        a, b = tok.split(":")
        d.append(float(a))
        v.append(float(b))
    o = np.argsort(d)
    return np.asarray(d, float)[o], np.asarray(v, float)[o]


def et2p03_probes(TO, keys=("config", "row", "col", "charge_fC")):
    """Expand the ET2.03 `search_trace` column into the same long probe table that notebooks 01
    and 02 fit — one row per probe, with `dac`, `trig_count`, `header_max` and the efficiency
    fractions — so that `turnoff_table`, `curves_from` and `scurve_sigma` apply unchanged.

    The trigger-bit counter plateaus at either 1 or 2 counts per Level-1 accept, on **both** the
    January and the August campaigns (it depends on how many clock cycles the trigger bit is
    asserted for, not on the discriminator). `frac_c` therefore clips at 1, exactly as in the two
    earlier notebooks, and the 50 % crossing is taken on the raw fraction — which is the search's
    own bracket criterion, `counter >= header_max / 2`, and so reproduces what the search itself
    converged on regardless of the plateau level.

    The searches record no TDC counter per probe (only at the final point), so `frac_tdc` is NaN
    here and the TDC-readout turn-off that notebook 01 carries as a low-charge cross-check has no
    counterpart in this campaign.
    """
    rows = []
    for _, r in TO.iterrows():
        d, v = parse_trace(r.search_trace)
        base = {k: r[k] for k in keys}
        for dd, vv in zip(d, v):
            rec = dict(base)
            rec.update(dac=dd, trig_count=vv, header_max=float(r.l1a_max))
            rows.append(rec)
    p = pd.DataFrame(rows)
    p["frac"] = p.trig_count / p.header_max
    p["frac_c"] = p.frac.clip(upper=1.0)
    p["frac_tdc"] = np.nan
    return p


def noise_peak(trace, floor=0.0):
    """The baseline and the noise width read off one dense `quietCB` noise scan.

    Returns `(peak_dac, peak_count, fwhm_dac, n_steps)`. `peak_dac` is the threshold at which the
    trigger-bit counter is largest — the top of the noise peak, i.e. the **baseline** — and
    `fwhm_dac` is the full width at half that maximum, obtained by linear interpolation on each
    flank. Both are measurements *inside the configuration*, which is what makes a baseline and a
    noise width available at every dose here.

    FWHM and the autocalibration `NW` are not the same definition, so the two are compared in the
    notebook rather than substituted for one another.
    """
    d, v = parse_trace(trace)
    if len(d) < 3 or v.max() <= floor:
        return np.nan, np.nan, np.nan, len(d)
    i = int(np.argmax(v))
    half = v[i] / 2.0
    lo = np.nan
    for j in range(i, 0, -1):
        if v[j - 1] < half <= v[j]:
            lo = d[j - 1] + (half - v[j - 1]) * (d[j] - d[j - 1]) / (v[j] - v[j - 1])
            break
    hi = np.nan
    for j in range(i, len(d) - 1):
        if v[j + 1] < half <= v[j]:
            hi = d[j] + (v[j] - half) * (d[j + 1] - d[j]) / (v[j] - v[j + 1])
            break
    return float(d[i]), float(v[i]), float(hi - lo), len(d)


# ------------------------------------------- CERN Aug-2024 climate-chamber monitoring (notebook 00)
# `ET2p03_BARE4`, a bare ET2.03 die, in the CERN climate chamber 2024-08-04 to 08-14. It took NO
# charge-injection data and it was NEVER irradiated (`save_notes` says 0 MRad on all 140 maps). It
# is in this folder because it is the only place where an ET2.03 chip's INTERNAL voltage reference
# and its on-chip temperature sensor were logged against a real temperature sweep.
#
# The four source sqlite files are the CERN_Aug_2024_CC_CTS folder of the etroclaptop3 backup on
# the X9 removable disk; `data/census/PROVENANCE.md` carries their paths and md5 values. Added
# strictly below everything else in this module, so nothing above changed.
CCTS_CAMPAIGN = "CERN climate chamber CTS, August 2024"

# Physical windows, applied only where a median or a fit is taken, never to the plotted series.
# The instrument is a bench multimeter shared between two inputs; when a channel is open or the
# board is off it reports values far outside the chip's range.
CCTS_VREF_WINDOW = (0.5, 1.2)      # the ET2.03 internal reference sits near 0.87 V
CCTS_VTEMP_WINDOW = (0.3, 1.0)     # the on-chip temperature sensor sits near 0.55 V


def load_ccts_vol():
    """RAW, downsampled: the `vol_monitoring` table, one sample per instrument per minute out of
    336 166 recorded at about 5 s. `instrument` is `VRef` (the ET2.03 **internal** reference, read
    on the pin) or `VTemp` (the on-chip temperature sensor output). Both are bench-multimeter
    voltages in volts; neither is calibrated to a temperature by anything in the campaign."""
    return pd.read_csv(CENSUS_DIR / "aux_ccts_vol.csv.gz", parse_dates=["timestamp"])


def load_ccts_chamber():
    """RAW, downsampled: the `ti_hdc1080` climate-chamber air temperature and relative humidity,
    one sample per minute out of 85 779. This is chamber air, not a cold plate and not the die."""
    return pd.read_csv(CENSUS_DIR / "aux_ccts_chamber.csv.gz", parse_dates=["timestamp"])


def load_ccts_lv():
    """RAW, downsampled: the `power_v2` low-voltage rails, one sample per (instrument, rail) per
    minute out of 350 489. Three instruments appear: `Power1` (2024-08-04 to 08-13) and then
    `Power` and `WS_Power` (08-13 to 08-14), so the series must be grouped by instrument as well
    as by rail. Unit strings (`1.3752`, `0.5110`) are stripped at extraction. **These are
    supply-terminal voltages**, not voltages at the chip."""
    d = pd.read_csv(CENSUS_DIR / "aux_ccts_lv.csv.gz", parse_dates=["timestamp"])
    d["P_mW"] = d.V * d.I_A * 1e3
    return d


def load_ccts_lv_actions():
    """RAW: the `actions_v2` Start/Stop and channel On/Off rows, 469 of them."""
    return pd.read_csv(CENSUS_DIR / "aux_ccts_lv_actions.csv", parse_dates=["timestamp"])


def load_ccts_blnw():
    """RAW: every row of the campaign's `baselines` table, 35 840 = 140 full 16x16 maps.

    The `save_notes` free-text field is the campaign's condition record, and it parses exactly
    (`+20.3C, 0 MRad, A: 1.2V, D: 0.9V`, 90 distinct values, none unparsed). The four parsed
    columns are added here: `note_temp_C`, `note_dose_MRad`, `note_va_nominal_V`,
    `note_vd_nominal_V`. The rail figures in the note are the **nominal at-chip settings** the
    operator was aiming at, not supply-terminal readings -- notebook 00 shows the two side by
    side."""
    return pd.read_csv(CENSUS_DIR / "aux_ccts_blnw.csv.gz", parse_dates=["timestamp"])


def load_tid_et2p03_timeline():
    """RAW, transcribed: the ET2.03 August-2024 irradiation start/stop markers, read off the
    analysis notebook that consumed the (currently unlocated) monitoring folder. `source_in_notebook`
    names the cell each marker came from and records the one disagreement between them."""
    return pd.read_csv(CENSUS_DIR / "aux_tid_et2p03_timeline.csv", parse_dates=["timestamp"])


def load_x9_history_folders():
    """RAW: one `ls` per per-campaign history folder of the etroclaptop3 backup on the X9 disk,
    read-only, 2026-09-04. Listing only: none of these campaigns is analysed anywhere here."""
    return pd.read_csv(CENSUS_DIR / "census_x9_history_folders.csv").fillna("")
