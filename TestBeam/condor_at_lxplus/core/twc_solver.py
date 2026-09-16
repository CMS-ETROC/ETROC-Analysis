"""Time-walk correction: the JOINT least-squares solve, in one place.

WHAT THIS REPLACES
------------------
Every time-walk correction in this repository used to be the same alternating
two-pass loop::

    for _ in range(2):                        # <- the defect
        delta_r = mean(toa of the others) - toa_r
        toa_r  += poly2(tot_r) fitted to delta_r

That loop is a fixed-point iteration whose fixed point is exactly the joint
least-squares problem solved here, and it was stopped after TWO passes.  The
leftover after k passes is `s1 * rho^(k-1)`, where rho is the inter-board TOT
correlation (notes/twc-convergence-verify.md measured that contraction law on
seven runs, better than 15 % on every one).  At rho ~ 0.35 - the H1 3.5e15
July family - two passes leave 26-30 ps/ns of un-corrected own-TOT slope and
the pair widths come out 7-16 ps too wide; at rho ~ 0.1 (March, 1.5e15) the
same defect is worth < 0.25 ps.  The alternation is therefore not a numerical
detail, it is a run-dependent bias.

THE FIX: solve all coefficients at once
---------------------------------------
For roles r = 1..N with per-board correction polynomials p_r (order 2, so 3
coefficients each, 9 in total for a three-board track), minimise

    S = sum_{a<b} sum_events [ (toa_a + p_a(tot_a)) - (toa_b + p_b(tot_b)) ]^2

which is linear in the coefficients: one least-squares problem on a design of
(n_pairs * n_events) rows by (N * (order+1)) columns.  For N = 3 the
stationarity condition of S is precisely the alternation's fixed point
(sum_r ||delta_r||^2 = 0.75 * sum_pairs ||toa_a - toa_b||^2), so the joint
solve IS the converged loop - verified on real data to pair widths agreeing
within 0.00 ps and residual own-TOT slope 0.000 ps/ns (against 26-30 at two
passes).  It has no iteration count to tune, no per-run rho to worry about,
and costs one solve instead of 2*N polynomial fits.

GAUGE.  Adding the same constant to every board's polynomial leaves every pair
difference unchanged, so the design is rank-deficient by exactly one (the
global constant).  `np.linalg.lstsq` returns the minimum-norm solution, which
is deterministic; every quantity the pipeline computes downstream is a pair
difference, hence gauge invariant.

MEMORY.  The design is never materialised.  This same function is called both
on a per-track bootstrap sample (a few thousand rows, core/bootstrap.py) and on
a whole merged run through the notebook helper (BeamTestHelpers/twc.py), where
n is 1e6-1e7 and a dense (n_pairs*n) x 9 array would be 0.2-2 GB before lstsq
makes its own copies.  Only the 9x9 (strictly (ncol+1)^2) triangular factor R
of the augmented design [A | y] enters the answer, so the rows are streamed in
blocks of `_CHUNK_EVENTS` events and folded into a running R by QR.  Because
R^T R = [A|y]^T [A|y] exactly, R poses the identical least-squares problem -
same singular values, same rank deficiency, hence the same minimum-norm
solution - and the same `rcond` the dense call would have used is passed
explicitly so the gauge direction is discarded at the same threshold.  Peak
memory is O(_CHUNK_EVENTS) instead of O(n).

CONDITIONING.  TOT is in ps (~1500-4000), so a raw [tot^2, tot, 1] basis spans
seven orders of magnitude between its columns.  The fit is therefore done in a
centred and scaled variable u = (tot - mean)/sd and the coefficients are
composed back into the raw-TOT polynomial before they are returned, so callers
(and the diagnostics that print "a*TOT^2 + b*TOT + c, TOT in ps") keep the
convention they had.

This module deliberately depends on numpy only, so that the pipeline worker
(core/bootstrap.py), the two diagnostics tools (utils/) and the standalone
notebook helper (BeamTestHelpers/twc.py) can all share this one implementation
instead of carrying four copies of the loop that had to be found and moved
together.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

__all__ = [
    'TWC_ORDER',
    'solve_timewalk_coefficients',
    'apply_timewalk_correction_arrays',
    'timewalk_deltas',
    'selfcheck_against_iteration',
]

TWC_ORDER = 2       # quadratic in TOT; unchanged by decision (see the run plan)

# Events per streamed block of the design (see MEMORY above).  With three
# boards a block is 3*100k = 300k rows of 10 doubles, ~24 MiB, and peak memory
# stays there however long the run is.  The answer does not depend on it.
_CHUNK_EVENTS = 100_000


def _basis(x, order, mu, sd):
    """Vandermonde in u = (x-mu)/sd, highest power first (np.polyfit order)."""
    u = (x - mu) / sd
    return np.vander(u, order + 1)


def _compose_to_raw(coef_desc, mu, sd, order):
    """Coefficients of p(u), u = (x-mu)/sd, expressed as a polynomial in x.

    Returned highest-power-first, length order+1, so callers can keep using
    np.polyval / np.poly1d on raw TOT in ps.
    """
    asc = np.asarray(coef_desc, float)[::-1]            # ascending powers of u
    out = np.zeros(order + 1)                            # ascending powers of x
    # u^k = ((x - mu)/sd)^k  ->  binomial expansion
    for k, c in enumerate(asc):
        if c == 0.0:
            continue
        for j in range(k + 1):
            out[j] += c * _binom(k, j) * (1.0 / sd) ** k * (-mu) ** (k - j)
    return out[::-1]


def _binom(n, k):
    from math import comb
    return float(comb(n, k))


def timewalk_deltas(toas, roles):
    """delta_r = mean(TOA of the other boards) - TOA_r.

    Kept because the diagnostics plot it (it is what the old loop fitted, and
    what the joint solve's residual is measured against).  For three boards
    this is the historical `0.5*sum(others) - toa_r`.
    """
    n_other = len(roles) - 1
    return {r: (sum(toas[o] for o in roles if o != r) / float(n_other)) - toas[r]
            for r in roles}


def solve_timewalk_coefficients(tots, toas, roles, order: int = TWC_ORDER):
    """The joint least-squares time-walk solve.

    tots/toas : dict role -> 1-D array (TOT and TOA in ps, same length)
    roles     : list of roles participating in the solve (>= 2)
    order     : polynomial order in TOT (2 = the pipeline's quadratic)

    Returns {role: ndarray of order+1 polynomial coefficients in RAW TOT,
    highest power first} - i.e. the correction to ADD to that role's TOA is
    np.polyval(coeff[role], tot_role).
    """
    roles = list(roles)
    if len(roles) < 2:
        raise ValueError("time-walk solve needs at least two boards")
    n = len(np.asarray(toas[roles[0]]))
    ncoef = order + 1
    if n < ncoef:
        # Not enough events to constrain even one polynomial - leave the TOAs
        # alone rather than returning a fit of noise (the old loop's polyfit
        # would have raised or produced garbage here).
        return {r: np.zeros(ncoef) for r in roles}

    tot = {r: np.asarray(tots[r], dtype=float) for r in roles}
    toa = {r: np.asarray(toas[r], dtype=float) for r in roles}

    scale = {}
    for r in roles:
        mu = float(tot[r].mean())
        sd = float(tot[r].std())
        if not np.isfinite(sd) or sd <= 0:
            sd = 1.0
        scale[r] = (mu, sd)

    pairs = list(combinations(roles, 2))
    ncol = len(roles) * ncoef
    col0 = {r: roles.index(r) * ncoef for r in roles}
    nrow = len(pairs) * n

    # Stream the design [A | y] in blocks of events and keep only its running
    # triangular factor R (see MEMORY in the module docstring): after each block
    # R^T R equals the Gram matrix of every row seen so far, so the final R is
    # the same least-squares problem as the full A, never built.
    R = np.zeros((0, ncol + 1))
    for start in range(0, n, _CHUNK_EVENTS):
        stop = min(start + _CHUNK_EVENTS, n)
        m = stop - start
        blk = np.zeros((len(pairs) * m, ncol + 1))
        V = {r: _basis(tot[r][start:stop], order, *scale[r]) for r in roles}
        for k, (a, b) in enumerate(pairs):
            sl = slice(k * m, (k + 1) * m)
            blk[sl, col0[a]:col0[a] + ncoef] = V[a]
            blk[sl, col0[b]:col0[b] + ncoef] = -V[b]
            # residual to minimise is (toa_a + p_a) - (toa_b + p_b), so the target is
            # minus the raw pair difference
            blk[sl, ncol] = toa[b][start:stop] - toa[a][start:stop]
        R = np.linalg.qr(np.vstack((R, blk)), mode='r')

    # rcond is what np.linalg.lstsq(A, y, rcond=None) would have used on the
    # full design, passed explicitly so that the one rank deficiency (the global
    # gauge constant) is cut at the same threshold on the small system.
    sol, *_ = np.linalg.lstsq(R[:, :ncol], R[:, ncol],
                              rcond=np.finfo(float).eps * max(nrow, ncol))

    out = {}
    for i, r in enumerate(roles):
        mu, sd = scale[r]
        out[r] = _compose_to_raw(sol[i * ncoef:(i + 1) * ncoef], mu, sd, order)
    return out


def apply_timewalk_correction_arrays(tots, toas, roles, order: int = TWC_ORDER,
                                     return_coefficients: bool = False):
    """Joint solve + apply.  Returns {role: corrected TOA}, or (corrected, coeffs)."""
    coeffs = solve_timewalk_coefficients(tots, toas, roles, order=order)
    corrected = {r: np.asarray(toas[r], dtype=float) + np.polyval(coeffs[r], np.asarray(tots[r], dtype=float))
                 for r in roles}
    return (corrected, coeffs) if return_coefficients else corrected


def _iterate(tots, toas, roles, n_iter, order=TWC_ORDER):
    """The OLD alternating loop, kept only so the equivalence can be asserted."""
    cur = {r: np.asarray(toas[r], dtype=float).copy() for r in roles}
    for _ in range(n_iter):
        d = timewalk_deltas(cur, roles)
        for r in roles:
            cur[r] = cur[r] + np.polyval(np.polyfit(np.asarray(tots[r], float), d[r], order), np.asarray(tots[r], float))
    return cur


def selfcheck_against_iteration(tots, toas, roles, n_iter: int = 40, order: int = TWC_ORDER):
    """Discrepancy (ps) between the joint solve and the alternating loop run to
    convergence, per pair difference.

    The alternation's fixed point IS this least-squares problem, so a converged
    loop and the joint solve must give the same pair differences - UP TO A
    CONSTANT.  The correction is only determined modulo one global offset (add
    the same constant to every board's polynomial and every pair difference is
    unchanged), and the two methods land on different representatives of that
    gauge: `lstsq` returns the minimum-norm solution, the loop returns wherever
    it walked to.  A constant offset moves no width and no resolution, so what
    is checked is the CENTRED discrepancy.

    Returns (max_centred_ps, max_raw_ps): the first is the physical agreement
    and is ~1e-3 ps or below on real tracks; the second includes the meaningless
    gauge constant and is typically a few tenths of a ps.
    """
    joint = apply_timewalk_correction_arrays(tots, toas, roles, order=order)
    itr = _iterate(tots, toas, roles, n_iter, order=order)
    worst_c, worst_r = 0.0, 0.0
    for a, b in combinations(roles, 2):
        d = (joint[a] - joint[b]) - (itr[a] - itr[b])
        worst_r = max(worst_r, float(np.abs(d).max()))
        worst_c = max(worst_c, float(np.abs(d - d.mean()).max()))
    return worst_c, worst_r
