"""Gain-layer depletion voltage: the fit machinery shared by every V_gl figure.

V_gl(phi) is modelled as a single exponential, V_gl = V0 * exp(-c * phi), with c the acceptor
removal constant. Everything here is campaign- and sensor-agnostic: the measurement tables, the
beam constants and the literature comparison values belong to a campaign module or to the figure
that draws them.

The fit starting guess (p0), the chi2 sigma (sigma_v) and the fluence table are arguments:
they are what differs between H1 and F1.
"""
import textwrap

import numpy as np
from scipy.optimize import curve_fit
from matplotlib.ticker import FixedLocator, FixedFormatter

# Per-point reading uncertainty on V_gl from a fine IV scan, used only to quote a chi2/ndf. The
# parameter uncertainties come from the unweighted fit with the covariance scaled by the observed
# residual scatter, so they do not depend on this number.
SIGMA_V = 0.2  # V

YLABEL = r"gain-layer depletion voltage  $V_{\mathrm{gl}}$  [V]"


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------
def model(phi, v0, c):
    """phi in 1e15 p/cm2, c in 1e-15 cm2 per bookkeeping proton."""
    return v0 * np.exp(-c * phi)


def fit_exp(phi, vgl, p0=(35.0, 0.4), sigma_v=SIGMA_V):
    """Unweighted least squares. Returns a dict with parameters in physical units.

    c is reported in 1e-16 cm2 per bookkeeping proton (the fit works in 1e-15,
    hence the factor 10). Parameter errors come from the covariance scaled by
    the residual variance, which is the right thing to do when the per-point
    uncertainty is not independently known.

    p0 is the starting guess (V0 in volts, c in 1e-15 cm2 per proton); a gain layer
    that removes more slowly needs a different one. sigma_v is used only to quote
    chi2/ndf, never to weight the fit.
    """
    phi = np.asarray(phi, dtype=float)
    vgl = np.asarray(vgl, dtype=float)
    popt, pcov = curve_fit(model, phi, vgl, p0=list(p0), maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    resid = vgl - model(phi, *popt)
    ndf = len(phi) - 2
    return {
        "n_points": int(len(phi)),
        "ndf": int(ndf),
        "V0": float(popt[0]),
        "V0_err": float(perr[0]),
        "c_1e16": float(popt[1] * 10.0),
        "c_1e16_err": float(perr[1] * 10.0),
        "rms_resid_V": float(np.sqrt(np.mean(resid ** 2))),
        "max_abs_resid_V": float(np.max(np.abs(resid))),
        "chi2_ndf_sigma0p2": float(np.sum((resid / sigma_v) ** 2) / ndf),
        "residuals_V": [float(r) for r in resid],
    }


def fit_exp_log(phi, vgl):
    """Same model fitted in log space, that is with equal fractional weight per point.

    An unweighted fit in linear space is dominated by the high-voltage end whenever
    V_gl falls by a large factor over the campaign. The log-space fit is the natural
    straight-line fit on a logarithmic V_gl axis, and the difference between the two
    is carried as a fit-weighting systematic on c.
    """
    phi = np.asarray(phi, dtype=float)
    vgl = np.asarray(vgl, dtype=float)
    slope, intercept = np.polyfit(phi, np.log(vgl), 1)
    v0 = float(np.exp(intercept))
    c = float(-slope)
    resid = vgl - model(phi, v0, c)
    frac = np.log(vgl) - (intercept + slope * phi)
    ndf = len(phi) - 2
    return {
        "n_points": int(len(phi)),
        "ndf": int(ndf),
        "V0": v0,
        "c_1e16": c * 10.0,
        "rms_resid_V": float(np.sqrt(np.mean(resid ** 2))),
        "rms_frac_resid": float(np.sqrt(np.mean(frac ** 2))),
        "residuals_V": [float(r) for r in resid],
    }


def local_c(phi, vgl):
    """Removal constant inferred interval by interval, in 1e-16 cm2 per proton.

    A single exponential requires this to be constant; it is the sharpest test of
    the model that does not depend on any fit weighting.
    """
    phi = np.asarray(phi, dtype=float)
    vgl = np.asarray(vgl, dtype=float)
    return -np.diff(np.log(vgl)) / np.diff(phi) * 10.0


def series(data, board_index, labels, drop_above=None):
    """Return (fluence, V_gl) for one board restricted to the given timing labels.

    data is the campaign's V_gl table: (fluence, look label, [V_gl per board]).
    drop_above, when set, drops every point at or above that fluence.
    """
    phi, v = [], []
    for f, lab, vals in data:
        if lab not in labels:
            continue
        if drop_above is not None and f >= drop_above:
            continue
        phi.append(f)
        v.append(vals[board_index])
    return np.array(phi), np.array(v)


def avg_series(data, labels, drop_above=None):
    phi, v = [], []
    for f, lab, vals in data:
        if lab not in labels:
            continue
        if drop_above is not None and f >= drop_above:
            continue
        phi.append(f)
        v.append(float(np.mean(vals)))
    return np.array(phi), np.array(v)


def set_log_ticks(ax, ticks, labels=None):
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter(
        labels if labels is not None else [str(t) for t in ticks]))
    ax.yaxis.set_minor_locator(FixedLocator([]))


def wrap(text, width=118):
    return "\n".join(textwrap.wrap(" ".join(text.split()), width=width))


# ---- published gain-layer removal constants ---------------------------------------------------
# All in 1e-16 cm2. unit "p" = per incident proton (converted with the on-chip fluence factor
# only); "neq" = per n_eq (converted with NIEL x that factor). band "onchip" draws the +- on-chip
# fluence uncertainty, None draws none, "spread" draws the min-max over the listed wafers.
# Single proton constants carry the on-chip band, reactor-neutron constants none, multi-wafer
# constants their spread.
REFERENCES = {
    "HPK": [
        # Curras et al. 2023, arXiv:2306.11760: HPK type 2, 50 um LGAD, k-factor V_gl.
        dict(name="Curras 24 GeV/c p", c=9.51, unit="neq", color="#D55E00", ls="-", band="onchip"),
        dict(name="Curras reactor n", c=3.85, unit="neq", color="#0072B2", ls="-.", band=None),
        # Kraus et al. 2026, arXiv:2602.01800: HPK prototype 2, 23 GeV PS-IRRAD protons, quoted
        # per incident proton (no NIEL step by the authors).
        dict(name="Kraus W25", c=6.42, unit="p", color="#009E73", ls="-", band="onchip"),
        dict(name="Kraus W36", c=7.64, unit="p", color="#7B3294", ls="-", band="onchip"),
    ],
    "FBK": [
        # Ferrero et al. 2019, NIM A 919, 16 (arXiv:1802.01745): FBK UFSD2, 24 GeV/c CERN PS
        # protons quoted per incident proton; reactor neutrons quoted per n_eq.
        dict(name="Ferrero 24 GeV/c p, boron", c=6.5, unit="p", color="#D55E00", ls="-",
             band="onchip"),
        dict(name="Ferrero 24 GeV/c p, boron+C", c=3.3, unit="p", color="#009E73", ls="-",
             band="onchip"),
        dict(name="Ferrero reactor n, boron", c=5.4, unit="neq", color="#D55E00", ls="-.",
             band=None),
        dict(name="Ferrero reactor n, boron+C", c=2.1, unit="neq", color="#009E73", ls="-.",
             band=None),
        # Sorenson et al. 2023, arXiv:2311.02027: FBK4 wafers, 500 MeV protons at LANSCE, quoted
        # per n_eq after the authors' own kappa = 0.78; one value per wafer.
        dict(name="Sorenson 500 MeV p, deep+C", c=(6.2, 4.5, 4.2), unit="neq", color="#7B3294",
             ls="--", band="spread"),
        dict(name="Sorenson 500 MeV p, shallow", c=(5.4, 5.0, 5.7), unit="neq", color="#0072B2",
             ls="--", band="spread"),
    ],
}


def _num(c):
    return "/".join("%g" % v for v in np.atleast_1d(c))


def reference_rows(vendor, fluence_factor, conv_neq, niel, rel_err, alternates=()):
    """The published constants as c per BOOKKEEPING proton, one dict per curve.

    Each row carries lo / c / hi (the band, or None), `values` (the converted value per wafer),
    `arith` (the conversion as printed) and `alt`. `fluence_factor` is the on-chip fluence per
    bookkeeping proton, `conv_neq` the n_eq on the chip per bookkeeping proton. `alternates` is
    ((name, factor), ...): the same constant under an on-chip fluence factor the measurement
    excludes, drawn dotted; a per-n_eq constant then converts with NIEL x that factor.
    """
    rows = []
    todo = [(ref, None) for ref in REFERENCES[vendor]]
    todo += [(next(r for r in REFERENCES[vendor] if r["name"] == name), frac)
             for name, frac in alternates]
    for ref, frac in todo:
        per_p = ref["unit"] == "p"
        if frac is None:
            f = fluence_factor if per_p else conv_neq
        else:
            f = frac if per_p else niel * frac
        vals = sorted(float(v) * f for v in np.atleast_1d(ref["c"]))
        cen = float(np.mean(vals))
        if frac is not None or ref["band"] is None:
            lo = hi = None
        elif ref["band"] == "spread":
            lo, hi = vals[0], vals[-1]
        else:
            lo, hi = cen * (1.0 - rel_err), cen * (1.0 + rel_err)
        unit = "/p" if per_p else r"/n$_{\mathrm{eq}}$"
        name = ref["name"] if frac is None else "%s, if on-chip %.2f" % (ref["name"], frac)
        rows.append(dict(name=name, arith="%s %s x %g" % (_num(ref["c"]), unit, f), lo=lo,
                         c=cen, hi=hi, values=vals, color=ref["color"], ls=ref["ls"],
                         alt=frac is not None))
    return rows
