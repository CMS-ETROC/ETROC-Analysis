"""Tests of the universality readers: the study's verdict override on the level cells, the axis
tables the reduction composes (a rerun's file replacing the rows of its cells), the
self-transfer flag (the reduction's join that recovers it, its check of the flag columns a wide
axis-F table carries, the rows the notebook leaves out by it), the axis-F pair rows the
linear-refit figure draws, the DESY beam of a set of runs, and the reduction's default study
folder.

Run from TestBeam/:  python -m pytest etroc_plots/tests
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "..")))
if not os.environ.get("ETROC_CAMPAIGN"):
    os.environ["ETROC_CAMPAIGN"] = "irrad_2026"

from etroc_plots.campaigns import active as campaign            # noqa: E402
from etroc_plots.campaigns import desy_2026                    # noqa: E402
from etroc_plots.universality import levels, reduce, transfer  # noqa: E402


def synthetic_agg():
    """Aggregation rows of one pair: the row over 40 pixels and the self-transfer row over one,
    and one board row whose arm uses no source."""
    return pd.DataFrame(dict(
        cell="A4", arm=["xfer", "xfer", "notwc"], kind=["pair", "pair", "board"],
        board_or_role=["ref-dut", "ref-dut", "dut"], family=["extrap", "extrap", "na"],
        n_pix=[40, 1, 40], self_transfer=[0, 1, 0], mu=[0.5, 1.8, np.nan], se=[0.1, 0.4, np.nan],
        agg_tag="A_desy_f0p8153"))


def axis_rows(agg):
    """The axis-table rows of `agg`, as the study's axis-table code wrote them."""
    return agg.drop(columns="self_transfer").rename(columns={"mu": "mu_ps", "se": "se_ps"})


def test_self_transfer_join_flags_each_row():
    agg = synthetic_agg()
    assert reduce.self_transfer_flags(axis_rows(agg).iloc[[1, 0, 2]], agg, "t") == [1, 0, 0]


def test_self_transfer_join_rejects_a_non_unique_match():
    agg = synthetic_agg()
    agg.loc[1, "n_pix"] = 40          # both groups of the pair on one key
    with pytest.raises(SystemExit, match="A4/xfer/pair/ref-dut/extrap/A_desy_f0p8153/40: 2 rows"):
        reduce.self_transfer_flags(axis_rows(synthetic_agg()).iloc[:1], agg, "t")


def test_self_transfer_join_rejects_other_numbers():
    rows = axis_rows(synthetic_agg())
    rows.loc[0, "mu_ps"] = 0.6       # an aggregation table rewritten under the same tag
    with pytest.raises(SystemExit, match="1 rows do not join"):
        reduce.self_transfer_flags(rows, synthetic_agg(), "t")


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        fh.write(text)


def synthetic_results(root):
    """A study folder with one axis, X: its round-2 table, of cells X1, X2 and X3, and a rerun of
    X2 with two more columns (one quoted, holding a comma) and one more row."""
    for name in reduce.TSVS:
        write_text(os.path.join(root, "condor", name), "")
    for name in ("axis_F.csv", "axis_F_long.csv", "summary_F.md"):
        write_text(os.path.join(root, "results_r3", name), "")
    write_text(os.path.join(root, "results_r2", "axis_X.csv"),
               "axis,cell,arm,mu_ps\nX,X1,xfer,1.0\nX,X2,xfer,2.0\nX,X2,notwc,2.5\n"
               "X,X3,xfer,3.0\n")
    write_text(os.path.join(root, "results_r3", "guardfix", "axis_X_gf.csv"),
               'axis,cell,arm,mu_ps,lpo_shape,lpo_pairs\nX,X2,xfer,2.25,"a,b",4\n'
               "X,X2,notwc,2.75,,0\nX,X2,xfer_lpo,2.5,a,4\n")


def test_axis_sources_rerun_replaces_the_rows_of_its_cells(tmp_path, monkeypatch):
    synthetic_results(str(tmp_path))
    monkeypatch.setattr(reduce.campaign, "UNIV_AXIS_SOURCES",
                        {"X": ("results_r2/axis_X.csv", "results_r3/guardfix/axis_X_gf.csv")})
    files = reduce._results_paths(str(tmp_path))["axis:X"]
    header, lines, df, replaced = reduce.axis_table(files)
    assert header == "axis,cell,arm,mu_ps"
    assert lines == ["X,X1,xfer,1.0", "X,X2,xfer,2.25", "X,X2,notwc,2.75", "X,X2,xfer_lpo,2.5",
                     "X,X3,xfer,3.0"]
    assert replaced == {"X2": (2, 3)}
    assert list(df.columns) == ["axis", "cell", "arm", "mu_ps"]
    assert list(df.mu_ps) == [1.0, 2.25, 2.75, 2.5, 3.0]


def test_axis_sources_rerun_needs_the_round2_columns_first(tmp_path):
    synthetic_results(str(tmp_path))
    rerun = os.path.join(str(tmp_path), "results_r3", "guardfix", "axis_X_gf.csv")
    write_text(rerun, "axis,cell,mu_ps,arm\nX,X2,2.25,xfer\n")
    with pytest.raises(SystemExit, match="do not start with those of"):
        reduce.axis_table([os.path.join(str(tmp_path), "results_r2", "axis_X.csv"), rerun])


F_TAG = "F_desy_f0p8153"
# the axis-F rows of the synthetic tables: (cell, arm, kind, pair, family, n_pix, self_transfer,
# mu, se); the pair ref-extra holds a transfer row and a self-transfer row of xfer on one key
F_ROWS = [("F2_d1", "xfer", "pair", "ref-extra", "extrap", 69, 0, "5.32", "1.1"),
          ("F2_d1", "xfer", "pair", "ref-extra", "extrap", 5, 1, "0.0", "0.2"),
          ("F2_d1", "xfer_lin", "pair", "ref-extra", "extrap", 69, 0, "4.1", "1.0"),
          ("F2_d1", "xfer", "pair", "ref-trig", "extrap", 5, 1, "0.0", "0.2")]
# the wide rows: per pair the rows its arms hold (indices into F_ROWS), the others absent
F_WIDE_HELD = [("ref-extra", {"xfer": 0, "xfer_lin": 2}), ("ref-trig", {"xfer": 3})]
# the flag columns of the wide table, and the flags the reduction writes per wide row
F_WIDE_FLAG_NAMES = ["%s_self_transfer" % arm for arm in reduce.WIDE_ARMS]
F_WIDE_FLAGS = [["0", "0", "", "", ""], ["1", "", "", "", ""]]


def synthetic_axis_f(root, port_flags=None):
    """A study folder for the axes part: one round-2 axis, X, of one row; the long and the wide
    axis-F tables of F_ROWS; the aggregation table behind them, under root/tables. With
    port_flags (one list per wide row, in the order of F_WIDE_FLAG_NAMES) the wide table carries
    each arm's <arm>_self_transfer after its <arm>_present, as twc_transfer's axes writes it.
    Returns the paths of reduce._results_paths that build_axes reads."""
    long_head = "axis,cell,arm,kind,board_or_role,family,agg_tag,n_pix,mu_ps,se_ps"
    write_text(os.path.join(root, "results_r2", "axis_X.csv"),
               long_head + "\nX,X1,xfer,pair,ref-extra,extrap,%s,40,0.5,0.1\n" % F_TAG)
    write_text(os.path.join(root, "results_r3", "axis_F_long.csv"), long_head + "\n" + "".join(
        "F,%s,%s,%s,%s,%s,%s,%d,%s,%s\n" % (c, a, k, p, f, F_TAG, n, mu, se)
        for c, a, k, p, f, n, _, mu, se in F_ROWS))
    head = ["cell", "kind", "board_or_role", "family", "agg_tag"]
    for arm in reduce.WIDE_ARMS:
        head += ["%s_%s" % (arm, x) for x in ("n_pix", "mu_ps", "se_ps", "present")]
        head += ["%s_self_transfer" % arm] if port_flags else []
    lines = [",".join(head + ["delta_xferlin_minus_xfer_mu_ps"])]
    for r, (pair, held) in enumerate(F_WIDE_HELD):
        fields = ["F2_d1", "pair", pair, "extrap", F_TAG]
        for a, arm in enumerate(reduce.WIDE_ARMS):
            if arm in held:
                _, _, _, _, _, n, _, mu, se = F_ROWS[held[arm]]
                fields += [str(n), mu, se, "True"]
            else:
                fields += ["", "", "", "False"]
            fields += [port_flags[r][a]] if port_flags else []
        lines.append(",".join(fields + ["-1.22" if r == 0 else ""]))
    write_text(os.path.join(root, "results_r3", "axis_F.csv"), "\n".join(lines) + "\n")
    write_text(os.path.join(root, "tables", "agg_" + F_TAG, "agg_by_cell.csv"),
               "cell,arm,kind,name,extrap_family,n_pix,self_transfer,mu,se\n"
               "X1,xfer,pair,ref-extra,extrap,40,0,0.5,0.1\n" + "".join(
                   "%s,%s,%s,%s,%s,%d,%d,%s,%s\n" % r for r in F_ROWS))
    return {"axis:X": [os.path.join(root, "results_r2", "axis_X.csv")],
            "F_long": os.path.join(root, "results_r3", "axis_F_long.csv"),
            "F_wide": os.path.join(root, "results_r3", "axis_F.csv")}


def reduce_axis_f(tmp_path, monkeypatch, port_flags=None):
    """The text of univ_axis_F_wide.csv the axes part writes from synthetic_axis_f."""
    root, out = str(tmp_path / "study"), str(tmp_path / "out")
    os.makedirs(out)
    paths = synthetic_axis_f(root, port_flags)
    monkeypatch.setattr(reduce.campaign, "UNIV_AXIS_SOURCES", {"X": ("results_r2/axis_X.csv",)})
    reduce.build_axes(out, paths, root)
    with open(os.path.join(out, "univ_axis_F_wide.csv"), newline="") as fh:
        return fh.read()


def expected_axis_f_wide(tmp_path):
    """The study's wide table of synthetic_axis_f, its lines as written, with the flags the
    reduction appends."""
    root = str(tmp_path / "expected")
    with open(synthetic_axis_f(root)["F_wide"], newline="") as fh:
        lines = fh.read().splitlines()
    return "".join("%s,%s\n" % (line, ",".join(f)) for line, f in
                   zip(lines, [F_WIDE_FLAG_NAMES] + F_WIDE_FLAGS))


def test_axis_f_wide_without_flag_columns_gets_the_flags_appended(tmp_path, monkeypatch):
    text = reduce_axis_f(tmp_path, monkeypatch)
    assert text == expected_axis_f_wide(tmp_path)
    assert text.splitlines()[1].endswith(",-1.22,0,0,,,")


def test_axis_f_wide_flag_columns_that_agree_are_written_once_at_the_end(tmp_path, monkeypatch):
    text = reduce_axis_f(tmp_path, monkeypatch, port_flags=F_WIDE_FLAGS)
    assert text == expected_axis_f_wide(tmp_path)
    assert text.splitlines()[0].count("_self_transfer") == len(reduce.WIDE_ARMS)


def test_axis_f_wide_flag_column_that_differs_from_the_join_stops(tmp_path, monkeypatch):
    flags = [list(f) for f in F_WIDE_FLAGS]
    flags[1][0] = "0"            # the self-transfer row of ref-trig marked as a transfer row
    with pytest.raises(SystemExit, match="1 of its <arm>_self_transfer values differ.*\n  "
                                         "F2_d1/pair/ref-trig/extrap xfer_self_transfer: the "
                                         "table holds '0', the join gives '1'"):
        reduce_axis_f(tmp_path, monkeypatch, port_flags=flags)


def test_without_self_transfer_drops_flagged_rows():
    long = pd.DataFrame(dict(cell=["a", "a", "b"], self_transfer=[0, 1, 0]))
    assert list(transfer.without_self_transfer(long).index) == [0, 2]
    wide = pd.DataFrame(dict(cell=["a", "b", "c"], xfer_self_transfer=[1, 0, np.nan],
                             xfer_lin_lpo_self_transfer=[0, 0, 0]))
    assert list(transfer.without_self_transfer(wide).index) == [1, 2]


def test_without_self_transfer_needs_the_column():
    with pytest.raises(KeyError, match="no self_transfer column"):
        transfer.without_self_transfer(pd.DataFrame(dict(cell=["a"], n_pix=[3])))


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
                                               campaign.UNIV_AXIS_R2_CSV,
                                               campaign.UNIV_CELLS_CSV))


# the cells of axes E, J and K the guard-fix rerun redid (results_r3/guardfix/README.md)
GUARDFIX_CELLS = ["E10_f1_RF2toRF1", "E2_h1_off10to20_early", "E4_h1_RF2toRF1", "E5_h1_RF1toRF2",
                  "J1_h1_0h_to_141h", "J2_h1_141h_to_0h", "J3_h1_27h_to_141h", "J4_h1_0h_to_27h",
                  "K1_m25C_to_m10C_45deg", "K2_m10C_to_m25C_45deg", "K3_floor_17to18_m10C_45deg"]


@pytest.mark.skipif(not tables_reachable(), reason="the universality tables are not reachable")
def test_axis_r2_holds_the_guardfix_rows_of_its_cells():
    d = transfer.read_axis_r2()
    rerun = d.agg_tag.str.contains("_guardfix_")
    assert sorted(d[rerun].cell.unique()) == GUARDFIX_CELLS
    assert rerun[d.cell.isin(GUARDFIX_CELLS)].all()
    assert d.groupby("axis").size().to_dict() == {"A": 800, "B": 620, "C": 660, "D": 1089,
                                                  "E": 1395, "I": 393, "J": 370, "K": 684}


@pytest.mark.skipif(not tables_reachable(), reason="the universality tables are not reachable")
def test_axis_f_pair_rows_with_both_arms():
    every = transfer.read_axis_f()
    rows = levels.refit_rows(transfer.without_self_transfer(every))
    assert len(rows) == 49 and len(levels.refit_rows(every)) == 49
    assert rows.cell.nunique() == every.cell.nunique() == 11
    assert (rows.family == "interp").sum() == 16
    assert not rows.duplicated(["cell", "kind", "board_or_role", "family"]).any()
    assert rows.margin_ps.notna().all()
    assert rows.base_cell.notna().all()


def test_beam_text_follows_the_august_3gev_runs():
    assert desy_2026.beam_gev("aug", 21) == 3 and desy_2026.beam_gev("aug", "run22") == 4
    assert desy_2026.beam_gev("desy", "run26") == 4          # the bank's code of DESY May
    assert desy_2026.beam_text("may", [9, 26, 44]) == "4 GeV electrons"
    assert desy_2026.beam_text("aug", ["run19", "run21"]) == "3 GeV electrons"      # E13
    assert desy_2026.beam_text("aug", [18, 23]) == "3 GeV electrons"                # E14
    assert desy_2026.beam_text("aug", [5, 8]) == "3 and 4 GeV electrons (run 5 at 4 GeV)"
    assert (desy_2026.beam_text("aug", [23, 18, 8, 5]) ==
            "3 and 4 GeV electrons (run 5 at 4 GeV)")                                # e1-e4
    assert (desy_2026.beam_text("aug", [5, 6, 22, 7]) ==
            "3 and 4 GeV electrons (runs 5, 6 and 22 at 4 GeV)")
    with pytest.raises(ValueError):
        desy_2026.beam_gev("aug", 3)            # no beam energy recorded for that run


def test_desy_chips_keep_their_full_names():
    """The DESY chips are PT-IH5, PT-IH3, PT-IH2 (and PT-IH27, B1-P1-6, B1-P1-9): legends and pair
    names print them as the yaml does, never cut to IH5, which reads like an IRRAD chip."""
    assert [c["chip"] for c in transfer.run_conditions(None, "desy", "run26")] == [
        "PT-IH27", "PT-IH5", "PT-IH3", "PT-IH2"]
    assert [c["chip"] for c in transfer.run_conditions(None, "aug", "run19")] == [
        "B1-P1-9", "PT-IH3", "PT-IH5", "B1-P1-6"]
    bank = pd.DataFrame(dict(campaign="desy", run="run26", board_id=[0, 1, 2, 3],
                             role=["extra", "ref", "trig", "dut"]))
    assert transfer.pair_chips(None, bank, "desy", "run26", "ref-dut") == "PT-IH5 / PT-IH2"
    assert transfer.pair_chips(None, bank, "desy", "run26", "trig") == "PT-IH3"


def test_reduction_reads_its_study_files_from_the_inputs_by_default():
    if os.environ.get("ETROC_TWC_RESULTS_ROOT"):
        pytest.skip("ETROC_TWC_RESULTS_ROOT moves the study folder")
    root = campaign.TWC_RESULTS_ROOT
    assert root == os.path.join(campaign.INPUTS_UNIVERSALITY, "results")
    if not os.path.isdir(root):
        pytest.skip("the universality inputs are not reachable")
    paths = reduce._results_paths(root)       # exits naming every file it lacks
    assert len(paths) == len(reduce.TSVS) + 3 + len(campaign.UNIV_AXIS_SOURCES)
