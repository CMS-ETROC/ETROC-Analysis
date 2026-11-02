"""Tests of the three-board sigma solve and its censoring of degenerate cases.

Run from TestBeam/:  python -m pytest twc_transfer/tests
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "..")))
from twc_transfer import widths as W     # noqa: E402

ROLES = ["a", "b", "c"]


def test_sigma_from_widths_solves_three_boards():
    # boards of 1, 2 and 3 ps: pair widths sqrt(1+4), sqrt(1+9), sqrt(4+9)
    for w in ({"a-b": np.sqrt(5.0), "a-c": np.sqrt(10.0), "b-c": np.sqrt(13.0)},
              {"b-a": np.sqrt(5.0), "c-a": np.sqrt(10.0), "c-b": np.sqrt(13.0)}):
        sig, neg, rad = W._sigma_from_widths(w, ROLES)
        np.testing.assert_allclose([sig[r] for r in ROLES], [1.0, 2.0, 3.0], rtol=1e-12)
        np.testing.assert_allclose([rad[r] for r in ROLES], [1.0, 4.0, 9.0], rtol=1e-12)
        assert neg == {"a": 0, "b": 0, "c": 0}


def test_sigma_from_widths_censors_a_negative_radicand_as_nan():
    sig, neg, rad = W._sigma_from_widths({"a-b": 1.0, "a-c": 1.0, "b-c": 3.0}, ROLES)
    assert rad["a"] == 0.5 * (1.0 + 1.0 - 9.0)
    assert np.isnan(sig["a"]) and neg["a"] == 1               # never the core's floored 0.0
    np.testing.assert_allclose([sig["b"], sig["c"]], [np.sqrt(4.5), np.sqrt(4.5)], rtol=1e-12)
    assert neg["b"] == 0 and neg["c"] == 0
    # a NaN width enters every radicand, so every board is censored
    sig, neg, _ = W._sigma_from_widths({"a-b": np.nan, "a-c": 2.0, "b-c": 2.0}, ROLES)
    assert all(np.isnan(sig[r]) for r in ROLES) and all(neg[r] == 1 for r in ROLES)
