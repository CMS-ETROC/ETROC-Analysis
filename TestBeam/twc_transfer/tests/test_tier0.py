"""Tests of the analytic Tier-0 change of RMS^2 against the variances it predicts.

Run from TestBeam/:  python -m pytest twc_transfer/tests
"""
import os
import sys
from itertools import combinations

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "..")))
from twc_transfer import tier0 as T0     # noqa: E402


def test_tier0_matches_the_variances_of_the_coefficient_change():
    rng = np.random.default_rng(5)
    roles = ["extra", "ref", "dut"]
    n = 5000
    common = rng.normal(size=n)                              # correlated TOTs: covariances matter
    tot = {r: 3000.0 + 600.0 * (0.6 * common + rng.normal(size=n)) for r in roles}
    d = {"extra": np.array([1.5e-6, -0.012, 3.0]), "ref": np.array([-0.8e-6, 0.007, -1.0]),
         "dut": np.array([2.2e-6, -0.020, 0.5])}
    mom = {r: tuple(float(np.mean(tot[r] ** k)) for k in (1, 2, 3, 4)) for r in roles}
    cross = {}
    for a, b in combinations(roles, 2):
        ta, tb = tot[a], tot[b]
        cross["%s-%s" % (a, b)] = (np.mean(ta * tb), np.mean(ta ** 2 * tb),
                                   np.mean(ta * tb ** 2), np.mean(ta ** 2 * tb ** 2))
    pair, board, V = T0.tier0_delta(d, mom, cross, roles)
    w = {r: np.polyval(d[r], tot[r]) for r in roles}
    for r in roles:
        assert V[r] == pytest.approx(np.var(w[r]), rel=1e-6)
    for a, b in combinations(roles, 2):
        assert pair["%s-%s" % (a, b)] == pytest.approx(np.var(w[a] - w[b]), rel=1e-6)
    for r in roles:
        o = [q for q in roles if q != r]
        cov = lambda x, y: float(np.mean((x - x.mean()) * (y - y.mean())))    # noqa: E731
        want = np.var(w[r]) + cov(w[o[0]], w[o[1]]) - cov(w[r], w[o[0]]) - cov(w[r], w[o[1]])
        assert board[r] == pytest.approx(want, rel=1e-6)
