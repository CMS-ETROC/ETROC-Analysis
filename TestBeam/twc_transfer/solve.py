"""Coefficient primitives: the pipeline's joint least-squares solve, the SPEC.md 2 gauge, the apply
line toa + polyval(c, tot), the raw <-> centred basis change, the auxiliary coefficient covariance,
and the transfer variants (clip, shift, rescale, calfix).
"""
from itertools import combinations

import numpy as np

from ._core import ORDER, TS


def solve(tot, toa, roles, idx=None, permute=False, rng=None):
    """The pipeline's joint least-squares TWC (twc_solver), optionally on a subset
    and optionally with the TOT columns permuted (the placebo source)."""
    t = {r: (tot[r] if idx is None else tot[r][idx]) for r in roles}
    a = {r: (toa[r] if idx is None else toa[r][idx]) for r in roles}
    if permute:
        rng = rng or np.random.default_rng(0)
        t = {r: rng.permutation(t[r]) for r in roles}
    return TS.solve_timewalk_coefficients(t, a, roles)


def gauge_to(c, tot):
    """SPEC 2 gauge: remove the (unobservable) mean of the polynomial over the
    events it is applied to, so no arm can differ from another by a location."""
    c = np.asarray(c, float).copy()
    if len(tot):
        c[-1] -= float(np.polyval(c, tot).mean())
    return c


def apply_c(toa, tot, c):
    """The pipeline's apply line."""
    return np.asarray(toa, float) + np.polyval(np.asarray(c, float), np.asarray(tot, float))


def raw_to_centred(c, mu, sd, order=ORDER):
    """Raw-TOT coefficients -> coefficients in u=(t-mu)/sd (the comparable form)."""
    M = centred_to_raw_matrix(mu, sd, order)
    return np.linalg.solve(M, np.asarray(c, float))


def centred_to_raw_matrix(mu, sd, order=ORDER):
    M = np.zeros((order + 1, order + 1))
    for j in range(order + 1):
        e = np.zeros(order + 1)
        e[j] = 1.0
        M[:, j] = TS._compose_to_raw(e, mu, sd, order)
    return M


def coeff_cov(tot, toa, roles, coeffs, order=ORDER):
    """Auxiliary error propagation for the stored coefficients (NOT part of the fit).

    Rebuilds the same design twc_solver builds (centred basis, three pair blocks),
    takes the residual variance of the solved system and returns sigma^2 * pinv(A'A)
    per board, mapped to the raw-TOT basis.  Approximate: the three pair rows of one
    event are correlated and this OLS form ignores that, so it is used as a scale
    (the alpha/n_fit cross-check in SPEC 4), never as a quoted error.
    """
    roles = list(roles)
    n = len(toa[roles[0]])
    nc = order + 1
    mu, sd, V = {}, {}, {}
    for r in roles:
        t = np.asarray(tot[r], float)
        mu[r] = float(t.mean())
        s = float(t.std())
        sd[r] = s if (np.isfinite(s) and s > 0) else 1.0
        V[r] = TS._basis(t, order, mu[r], sd[r])
    pairs = list(combinations(roles, 2))
    A = np.zeros((len(pairs) * n, len(roles) * nc))
    y = np.empty(len(pairs) * n)
    for k, (a, b) in enumerate(pairs):
        sl = slice(k * n, (k + 1) * n)
        ia, ib = roles.index(a) * nc, roles.index(b) * nc
        A[sl, ia:ia + nc] = V[a]
        A[sl, ib:ib + nc] = -V[b]
        y[sl] = np.asarray(toa[b], float) - np.asarray(toa[a], float)
    x = np.concatenate([raw_to_centred(coeffs[r], mu[r], sd[r], order) for r in roles])
    res = A @ x - y
    dof = max(len(y) - (len(roles) * nc - 1), 1)
    s2 = float(res @ res) / dof
    C = s2 * np.linalg.pinv(A.T @ A)
    out = {}
    for i, r in enumerate(roles):
        blk = C[i * nc:(i + 1) * nc, i * nc:(i + 1) * nc]
        M = centred_to_raw_matrix(mu[r], sd[r], order)
        out[r] = M @ blk @ M.T
    return out, mu, sd


# ------------------------------------------------------------ transfer variants
def apply_variant(c, tot, variant, src_row=None, tgt_cal=None):
    """Return the ps correction array for a transfer variant (SPEC 4 extended arms)."""
    c = np.asarray(c, float)
    t = np.asarray(tot, float)
    if variant == "plain" or src_row is None:
        return np.polyval(c, t)
    if variant == "clip":
        return np.polyval(c, np.clip(t, float(src_row["q005"]), float(src_row["q995"])))
    if variant == "shift":
        return np.polyval(c, t - (float(np.median(t)) - float(src_row["q50"])))
    if variant == "rescale":
        sd_s = float(src_row["sd_tot_emp"]) or 1.0
        sd_t = float(np.std(t)) or 1.0
        return np.polyval(c, float(src_row["q50"]) + (t - float(np.median(t))) * sd_s / sd_t)
    if variant == "calfix":
        cs, ct = float(src_row.get("cal_mean", np.nan)), float(tgt_cal or np.nan)
        if not (np.isfinite(cs) and np.isfinite(ct) and ct > 0):
            return np.polyval(c, t)
        return np.polyval(c, t / ct * cs)
    raise ValueError(variant)


class VariantCoeff(object):
    """A coefficient object whose polyval is redefined by a transfer variant."""

    def __init__(self, c, variant, src_row, tgt_cal=None):
        self.c, self.variant, self.src_row, self.tgt_cal = np.asarray(c, float), variant, src_row, tgt_cal


def _corr_of(c, tot, tgt_cal=None):
    if isinstance(c, VariantCoeff):
        return apply_variant(c.c, tot, c.variant, c.src_row, c.tgt_cal)
    return np.polyval(np.asarray(c, float), np.asarray(tot, float))


def _gauge_corr(c, tot, tgt_cal=None):
    """Correction array with its mean over the applied events removed (SPEC 2 gauge)."""
    v = _corr_of(c, tot, tgt_cal)
    return v - (v.mean() if len(v) else 0.0)

