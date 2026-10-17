"""Tests of the decision statistics on hand-made numbers: TOST, Holm, the margin, k_cal.

Run from TestBeam/:  python -m pytest twc_transfer/tests
"""
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "..")))
from twc_transfer import stats as St     # noqa: E402

Z95 = 1.6448536269514722     # the one-sided 95 % normal quantile: a 90 % two-sided interval


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def test_tost_against_a_hand_computation():
    r = St.tost(0.2, 0.3, 1.0)
    assert r["verdict"] == "PASS"
    assert r["lo"] == pytest.approx(0.2 - Z95 * 0.3, abs=1e-12)
    assert r["hi"] == pytest.approx(0.2 + Z95 * 0.3, abs=1e-12)
    assert r["p"] == pytest.approx(max(1.0 - _phi(1.2 / 0.3), _phi(-0.8 / 0.3)), rel=1e-9)
    assert r["margin"] == 1.0
    assert St.tost(0.6, 0.3, 1.0)["verdict"] == "FAIL"        # hi = 1.09 > 1
    assert St.tost(-0.6, 0.3, 1.0)["verdict"] == "FAIL"       # lo = -1.09 < -1
    assert St.tost(0.2, 0.0, 1.0)["verdict"] == "NA"
    assert St.tost(np.nan, 0.3, 1.0)["verdict"] == "NA"


def test_holm_against_a_hand_computation():
    # sorted finite p: 0.005, 0.01, 0.03, 0.04 -> 4 x 0.005, 3 x 0.01, 2 x 0.03, 1 x 0.04,
    # each at least the previous one: 0.02, 0.03, 0.06, 0.06
    adj, rej = St.holm([0.01, 0.04, 0.03, np.nan, 0.005])
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, np.nan, 0.02], rtol=1e-12)
    assert list(rej) == [True, False, False, False, True]
    adj, rej = St.holm([np.nan, np.nan])
    assert np.isnan(adj).all() and not rej.any()


def test_margin_is_one_ps_or_two_percent():
    assert St.margin_for(30.0) == 1.0
    assert St.margin_for(50.0) == 1.0
    assert St.margin_for(100.0) == pytest.approx(2.0, rel=1e-12)
    assert St.margin_for(np.nan) == 1.0


def test_kcal_from_cells_on_a_small_frame():
    df = pd.DataFrame(dict(d_f0=[1.0, 0.0, 3.0, 4.0, np.nan],
                           d_f1=[0.0, 2.0, 0.0, 0.0, 1.0],
                           scale=[1.0, 1.0, 1.0, 2.0, 1.0]))
    # |d_f0 - d_f1| / scale = 1, 2, 3, 2 (the NaN row is dropped): median 2
    assert St.kcal_from_cells(df) == pytest.approx(2.0 / (2 * 0.6745), rel=1e-12)
    assert np.isnan(St.kcal_from_cells(df.iloc[4:]))
