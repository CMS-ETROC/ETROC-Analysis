"""Tests of the universality readers: the study's verdict override on the level cells, and the
axis-F pair rows the linear-refit figure draws.

Run from TestBeam/:  python -m pytest etroc_plots/tests
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "..")))
if not os.environ.get("ETROC_CAMPAIGN"):
    os.environ["ETROC_CAMPAIGN"] = "irrad_2026"

from etroc_plots.campaigns import active as campaign            # noqa: E402
from etroc_plots.universality import levels, transfer          # noqa: E402


def synthetic_master():
    """Master-cell rows covering each branch of the override."""
    return pd.DataFrame(dict(
        cell=["unguarded_pass", "uninformative_supporting", "absent_final_fail",
              "absent_final_fail_all_pass", "guarded"],
        guard_verdict=["absent", "uninformative", "absent", "absent", "PASS"],
        final=["PASS", "PASS (supporting)", "FAIL (guarded)", "FAIL", "PASS"],
        all_verdict=["PASS", "FAIL", "FAIL", "PASS", "PASS"],
        verdict_pair_interp=["PASS", "FAIL+UNGUARDED", "FAIL+UNGUARDED", "NO-ROWS", "FAIL"],
    ))


def test_unguarded_cell_that_is_not_a_fail_reads_na_unguarded():
    out = transfer.override_verdicts(synthetic_master()).set_index("cell")
    assert out.loc["unguarded_pass", "all_verdict"] == "NA-UNGUARDED"
    assert out.loc["uninformative_supporting", "all_verdict"] == "NA-UNGUARDED"


def test_final_fail_keeps_its_all_verdict():
    out = transfer.override_verdicts(synthetic_master()).set_index("cell")
    assert out.loc["absent_final_fail", "all_verdict"] == "FAIL"
    assert out.loc["absent_final_fail_all_pass", "all_verdict"] == "PASS"


def test_fail_plus_unguarded_reads_fail():
    out = transfer.override_verdicts(synthetic_master())
    assert list(out.verdict_pair_interp) == ["PASS", "FAIL", "FAIL", "NO-ROWS", "FAIL"]


def test_guarded_cell_and_input_frame_untouched():
    df = synthetic_master()
    out = transfer.override_verdicts(df).set_index("cell")
    assert out.loc["guarded", "all_verdict"] == "PASS"
    assert df.equals(synthetic_master())


def tables_reachable():
    return all(os.access(p, os.R_OK) for p in (campaign.UNIV_AXIS_F_WIDE_CSV,
                                               campaign.UNIV_AXIS_F_LONG_CSV,
                                               campaign.UNIV_CELLS_CSV))


@pytest.mark.skipif(not tables_reachable(), reason="the universality tables are not reachable")
def test_axis_f_pair_rows_with_both_arms():
    rows = levels.refit_rows(transfer.read_axis_f())
    assert len(rows) == 49
    assert (rows.family == "interp").sum() == 16
    assert not rows.duplicated(["cell", "kind", "board_or_role", "family"]).any()
    assert rows.margin_ps.notna().all()
    assert rows.base_cell.notna().all()
