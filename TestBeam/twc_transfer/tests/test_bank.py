"""Tests of the bank rows of a triple: the column set, the folds, the stored coefficients.

Run from TestBeam/:  python -m pytest twc_transfer/tests
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "..")))
from twc_transfer import bank as B       # noqa: E402
from twc_transfer import events as E     # noqa: E402
from twc_transfer import solve as S      # noqa: E402

ROLES = ["extra", "ref", "dut"]


def toy_triple(n=900, seed=4):
    """A quadratic walk per board plus 30 ps of noise, 45 files of 20 events, no file read."""
    rng = np.random.default_rng(seed)
    walk = {"extra": [2.0e-6, -0.030, 60.0], "ref": [1.5e-6, -0.025, 40.0],
            "dut": [2.5e-6, -0.035, 80.0]}
    tot = {r: rng.uniform(1500.0, 5000.0, n) for r in ROLES}
    toa = {r: 10000.0 - np.polyval(walk[r], tot[r]) + rng.normal(0.0, 30.0, n) for r in ROLES}
    return E.Triple(tot, toa, ROLES, file=np.arange(n) // 20,
                    path="track_d-R14C1_r-R14C1_e-R14C1.parquet",
                    campaign="toy", run="toy", combo="extra0-ref1-dut2")


def test_bank_columns_are_the_columns_of_bank_rows():
    tri = toy_triple()
    df = pd.DataFrame(B.bank_rows_for_triple(tri, folds="both", guards=("full",), placebo=True))
    assert tuple(df.columns) == B.BANK_COLUMNS
    assert len(B.BANK_COLUMNS) == 56 and len(set(B.BANK_COLUMNS)) == 56
    assert set(B.BANK_META) <= set(B.BANK_COLUMNS)
    # five folds x (fit, placebo) x three boards
    assert len(df) == 30
    assert set(df["fold"]) == {"full", "0", "1", "blockA", "blockB"}
    assert sorted(df["placebo"].unique()) == [0, 1]
    full = df[(df["fold"] == "full") & (df["placebo"] == 0)].set_index("role")
    c = S.solve(tri.tot, tri.toa, tri.roles)
    for r in ROLES:
        np.testing.assert_array_equal(full.loc[r, ["c2", "c1", "c0"]].to_numpy(float), c[r])
        assert full.loc[r, "board_id"] == tri.boards[r]
