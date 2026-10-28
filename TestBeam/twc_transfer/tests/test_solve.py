"""Tests of the coefficient primitives: the gauge, the apply line, the raw <-> centred basis.

Run from TestBeam/:  python -m pytest twc_transfer/tests
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "..")))
from twc_transfer import solve as S     # noqa: E402

C = np.array([2.0e-6, -0.03, 150.0])     # a raw-TOT quadratic in ps, highest power first


def test_gauge_to_removes_the_mean_and_round_trips():
    tot = np.random.default_rng(1).uniform(1000.0, 6000.0, 500)
    g = S.gauge_to(C, tot)
    assert np.array_equal(g[:2], C[:2])                       # only the constant moves
    assert abs(np.polyval(g, tot).mean()) < 1e-9              # zero mean over the events
    # the removed location is the mean correction: adding it back recovers the input
    np.testing.assert_allclose(g[-1] + np.polyval(C, tot).mean(), C[-1], rtol=0, atol=1e-9)
    np.testing.assert_allclose(S.gauge_to(g, tot), g, rtol=0, atol=1e-9)   # idempotent
    assert np.array_equal(S.gauge_to(C, np.array([])), C)     # no events, nothing removed


def test_apply_c_is_toa_plus_the_quadratic():
    rng = np.random.default_rng(2)
    tot = rng.uniform(1000.0, 6000.0, 200)
    toa = rng.normal(10000.0, 40.0, 200)
    want = toa + C[0] * tot ** 2 + C[1] * tot + C[2]
    np.testing.assert_allclose(S.apply_c(toa, tot, C), want, rtol=1e-13, atol=1e-9)
    np.testing.assert_array_equal(S.apply_c(toa, tot, C), toa + np.polyval(C, tot))


def test_raw_and_centred_bases_are_inverse():
    mu, sd = 3200.0, 650.0
    b = S.raw_to_centred(C, mu, sd)
    np.testing.assert_allclose(S.centred_to_raw_matrix(mu, sd) @ b, C, rtol=1e-12, atol=1e-15)
    # the centred coefficients are the same polynomial in u = (t - mu) / sd
    t = np.linspace(1000.0, 6000.0, 11)
    np.testing.assert_allclose(np.polyval(b, (t - mu) / sd), np.polyval(C, t), rtol=1e-12, atol=1e-9)
