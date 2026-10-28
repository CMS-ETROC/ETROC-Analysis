"""The estimator (SPEC.md 4, Tier-2): the pipeline's width (core/bootstrap GMM FWHM/2.355, seeded
on the residuals), the IQR companion, the three-board sigma solve with degenerate cases censored,
the evaluation of one arm on a target triple, the residuals and own-TOT slopes behind it, and the
partner-aware positive control.
"""
import hashlib
from itertools import combinations

import numpy as np

from ._core import BS, KS_DMAX, KS_PMIN
from .campaigns import PS_PER_NS
from .events import assert_keys_disjoint
from .solve import _gauge_corr


def width_gmm(x, seed_salt=0):
    """The pipeline's width: core/bootstrap.fit_gmm_and_get_fwhm, FWHM/2.355.

    `core/bootstrap.py`'s `GaussianMixture` is DELIBERATELY left unseeded there
    (it is shared with pipeline step 13, so that file is never edited here) -
    `random_state=None` reads sklearn's global numpy RandomState, so a caller CAN
    make it reproducible by seeding that global state first.  It is seeded
    content-addressed on the residual array itself (not on `tri.name`/fold/arm
    identity), so two arms that see the BIT-IDENTICAL residual array (e.g. an
    untransferred board's pair in a one-board cell, SPEC 3b) get the same GMM
    restart sequence and therefore the same fitted width - which is what
    "bit-identical residual array -> bit-identical row" (SPEC 3b) requires once
    the estimator itself is seeded.  Unseeded, two runs of the estimator on the
    same input were measured to differ by mean 0.16 ps / p99 0.70 ps / max 17 ps
    (thin triples, n just above 1500) on the primary delta's GMM value column.
    A consequence: the width depends on the exact bytes of the residuals, so a
    change in the last bit of a coefficient changes the seed and can move a
    width by the restart jitter, not only by the float difference.

    `self_in_rep` (SPEC 5.7) exists to measure the UNSEEDED estimator's own
    restart jitter on the identical residuals `self_in` used; seeding by content
    alone would make it bit-identical to `self_in` by construction, a jitter
    probe turned into a structural zero.  `seed_salt` (default 0, every other
    caller) is mixed into the hashed bytes so a deliberate re-draw
    (`evaluate_arm` passes `seed_salt=1` for `self_in_rep` alone) gets a
    DIFFERENT but still reproducible seed from the same input.

    The seed is PROCESS-GLOBAL numpy state (`np.random.seed`, not a local
    `Generator`): any code importing this module that later calls `np.random.*`
    directly - not through this function - is reseeded as a side effect of the
    most recent `width_gmm` call, every time one runs.
    """
    x = np.asarray(x, float)
    digest_src = np.ascontiguousarray(x).tobytes()
    if seed_salt:
        digest_src += int(seed_salt).to_bytes(8, "little", signed=False)
    np.random.seed(int(hashlib.blake2b(digest_src, digest_size=4).hexdigest(), 16))
    fwhm, ks_d, ks_p = BS.fit_gmm_and_get_fwhm(x)
    w = fwhm / 2.355 if fwhm else np.nan
    acc = int(np.isfinite(w) and w > 0 and (ks_p >= KS_PMIN or ks_d <= KS_DMAX))
    comps = "3" if len(x) >= 1500 else "1,2,3"
    grid_step = (float(x.max()) - float(x.min())) / 999.0 if len(x) > 1 else np.nan
    return dict(w=w, ks_d=float(ks_d), ks_p=float(ks_p), accepted=acc, n=len(x),
                grid_step=grid_step, comps=comps,
                iqr=float(np.subtract(*np.percentile(x, [75, 25])) / 1.349),
                rms=float(np.std(x, ddof=1)) if len(x) > 1 else np.nan)


def width_iqr(x):
    """Tier-1 companion width: IQR/1.349 on the same pooled residuals (SPEC 4)."""
    x = np.asarray(x, float)
    return float(np.subtract(*np.percentile(x, [75, 25])) / 1.349)


def resolve_coeff(cmap, fold, pair, role):
    if (fold, pair, role) in cmap:
        return cmap[(fold, pair, role)]
    return cmap[(fold, role)]


def _sigma_from_widths(w, roles):
    """Three-board sigma solve, with the degenerate case CENSORED rather than floored.

    `BS.calculate_resolution_from_fit` returns 0.0 - not NaN - when the radicand
    0.5*(w_ab^2 + w_ac^2 - w_bc^2) is <= 0, and also when a width is NaN (`NaN > 0`
    is False).  Differenced against a ~35 ps baseline that 0.0 becomes a -35 ps
    "improvement" and, being a point estimate with a pooled per-cell error, it is
    not down-weighted (measured: -53.8 and -50.2 ps entering two LPO cells).
    Step 13 rejects such a sample outright,
    so the toolkit does too: the sigma is NaN and `neg_radicand` records why.
    """
    def sq(a, b):
        v = w.get("%s-%s" % (a, b), w.get("%s-%s" % (b, a), np.nan))
        return float(v) ** 2 if np.isfinite(v) else np.nan

    neg, rad = {}, {}
    for r in roles:
        o = [q for q in roles if q != r]
        v = 0.5 * (sq(r, o[0]) + sq(r, o[1]) - sq(o[0], o[1]))
        rad[r] = v
        neg[r] = int((not np.isfinite(v)) or v <= 0)
    try:
        raw = BS.calculate_resolution_from_fit(w, list(roles))
    except Exception:
        raw = {}
    sig = {r: (np.nan if neg[r] else float(raw.get(r, np.nan))) for r in roles}
    # a floored 0.0 that slipped through (all widths finite, radicand > 0 but the
    # solve still returned 0) is censored too - a 0 ps board sigma is not a datum
    for r in roles:
        if np.isfinite(sig[r]) and sig[r] <= 0:
            sig[r], neg[r] = np.nan, 1
    return sig, neg, rad


def evaluate_arm(tri, cmap, folds=(0, 1), per_fold=True, src_keys=None, arm=""):
    """One arm -> pair widths (pooled and per fold), per-board sigmas, diagnostics.

    `src_keys` (optional) are the event identities the coefficients were fitted on;
    when given, the SPEC 1 hard invariant is checked here, on every arm of every
    mode, and a violation raises LeakageError.  See assert_keys_disjoint.

    `self_in_rep` is `self_in`'s IDENTICAL coefficient map (SPEC 5.7's
    restart-jitter probe), so seeding `width_gmm` by content alone would make it
    bit-identical to `self_in` - a TOST that cannot fail and a structural zero.
    `self_in_rep` alone gets `seed_salt=1`, a deliberate, reproducible re-draw;
    every other arm is unsalted.
    """
    n_leak_checks = assert_keys_disjoint(tri, src_keys, folds=folds, arm=arm)
    res = arm_residuals(tri, cmap, folds=folds)
    seed_salt = 1 if arm == "self_in_rep" else 0
    pooled, byfold = {}, {}
    for pair, per in res.items():
        if not per:
            continue
        pooled[pair] = width_gmm(np.concatenate([per[f] for f in sorted(per)]), seed_salt=seed_salt)
        if per_fold:
            byfold[pair] = {f: width_gmm(per[f], seed_salt=seed_salt)
                            for f in sorted(per) if len(per[f]) > 20}
    w = {p: pooled[p]["w"] for p in pooled}
    sig, neg, rad = _sigma_from_widths(w, tri.roles)
    sig_fold, neg_fold = {}, {}
    if per_fold and byfold:
        for f in sorted({f for p in byfold for f in byfold[p]}):
            wf = {p: byfold[p][f]["w"] for p in byfold if f in byfold[p]}
            if len(wf) == 3:
                sig_fold[f], neg_fold[f], _ = _sigma_from_widths(wf, tri.roles)
    slopes = arm_own_slope(tri, cmap, folds=folds)
    return dict(pooled=pooled, byfold=byfold, sigma=sig, sigma_fold=sig_fold,
                neg_radicand=neg, neg_radicand_fold=neg_fold, radicand=rad,
                own_slope=slopes, n_leak_checks=n_leak_checks)


# ------------------------------------------------------- residuals and slopes
def arm_residuals(tri, cmap, folds=(0, 1)):
    """Corrected pair differences, per fold, each de-medianed then pooled.

    The per-fold median removal is the cross-fit convention (a per-fold
    offset is a nuisance the in-sample fit gets for free); the SPEC 2 gauge is
    applied to every coefficient at the moment of application, so arms differ by
    shape only, never by a location."""
    out = {}
    for a, b in combinations(tri.roles, 2):
        pair = "%s-%s" % (a, b)
        per = {}
        for f in folds:
            m = np.where(tri.fold == f)[0] if len(folds) > 1 else np.arange(tri.n)
            if len(m) < 2:
                continue
            ca = resolve_coeff(cmap, f, pair, a)
            cb = resolve_coeff(cmap, f, pair, b)
            d = ((tri.toa[a][m] + _gauge_corr(ca, tri.tot[a][m], tri.cal.get(a)))
                 - (tri.toa[b][m] + _gauge_corr(cb, tri.tot[b][m], tri.cal.get(b))))
            per[f] = d - np.median(d)
        out[pair] = per
    return out


def arm_own_slope(tri, cmap, folds=(0, 1)):
    """own_tot_slope of the corrected delta against each board's own TOT:
    exactly 0 for a converged self solve, nonzero when a transferred shape is wrong."""
    out = {}
    if any(len(k) == 3 for k in cmap):
        # a pair-specific (leave-pair-out) arm has no single per-board correction,
        # so a per-board own-TOT slope is not defined for it
        return {r: np.nan for r in tri.roles}
    for r in tri.roles:
        sl = []
        for f in folds:
            m = np.where(tri.fold == f)[0] if len(folds) > 1 else np.arange(tri.n)
            if len(m) < 10:
                continue
            corr = {}
            for q in tri.roles:
                if (f, q) not in cmap:
                    return {}
                corr[q] = tri.toa[q][m] + _gauge_corr(cmap[(f, q)], tri.tot[q][m], tri.cal.get(q))
            delta = 0.5 * sum(corr[o] for o in tri.roles if o != r) - corr[r]
            sl.append(own_tot_slope(tri.tot[r][m], delta))
        out[r] = float(np.mean(sl)) if sl else np.nan
    return out


def own_tot_slope(tot_r, delta_r):
    """OLS (mean) slope of the residual against the board's own TOT, ps/ns.
    A converged least-squares quadratic must leave this at zero.

    Vendored unchanged from own_tot_slope in tv_core.py, the module of the study's TWC
    verification (twcverify/), which is not in this repository; copied 2026-10-18.
    """
    return float(np.polyfit(np.asarray(tot_r, float) / PS_PER_NS, delta_r, 1)[0])


# ------------------------------------------------------------------ controls
def leaky_pairwidths(tri, xfold=True, folds=(0, 1)):
    """The PARTNER-AWARE positive control of the leave-pair-out guard (SPEC 5.1).

    Per pair (a,b), least-squares a correction p(tot_a) + p(tot_b) directly on the
    pair difference.  With `xfold` the fit is on the complementary fold, so there is
    no event leakage whatsoever - only partner information, which is exactly the
    channel a leave-pair-out guard exists to forbid (the channel that exposed a
    neural-network TWC, at -4.26 ps).  A guard with
    teeth must show this arm improving while `xfer` does not; if both are ~0 the
    guard is untested, not clean.
    """
    w = {}
    for a, b in combinations(tri.roles, 2):
        per = []
        for f in folds:
            m = np.where(tri.fold == f)[0]
            mf = np.where(tri.fold != f)[0] if xfold else m
            if len(m) < 20 or len(mf) < 20:
                continue
            y = tri.toa[a] - tri.toa[b]

            def design(idx):
                ta, tb = tri.tot[a][idx], tri.tot[b][idx]
                sa, sb = (ta.std() or 1.0), (tb.std() or 1.0)
                ua, ub = (ta - ta.mean()) / sa, (tb - tb.mean()) / sb
                return np.column_stack([np.ones(len(idx)), ua, ua ** 2, ub, ub ** 2])

            beta, _, _, _ = np.linalg.lstsq(design(mf), y[mf], rcond=None)
            d = y[m] - design(m) @ beta
            per.append(d - np.median(d))
        w["%s-%s" % (a, b)] = width_gmm(np.concatenate(per))["w"] if per else np.nan
    return w

