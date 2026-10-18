"""Tests of the axis tables (axes.py) on toy aggregation tables: the join of an aggregation row
with the means over its target triples, the folders and labels of axis A, the self-transfer
rows of the wide table of axis F, the guardfix overlay and the cell table of axis F.

Run from TestBeam/:  python -m pytest twc_transfer/tests
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

TESTBEAM = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, TESTBEAM)
from twc_transfer import axes as X        # noqa: E402
from twc_transfer import campaigns as C   # noqa: E402
from twc_transfer import jobs as J        # noqa: E402

KEY = ("xfer", "pair", "dut-ref")


def toy_agg(cell, f, families=("interp", "extrap")):
    """agg_by_cell rows of one cell: xfer on one pair per family, and notwc (family "na")."""
    rows = [(KEY[0], KEY[1], KEY[2], fam) for fam in families] + [("notwc", "pair", "dut-ref",
                                                                   "na")]
    return pd.DataFrame([dict(cell=cell, mode="a", arm=arm, kind=kind, name=name,
                              extrap_family=fam, primary=int(arm == "xfer"), self_transfer=0,
                              n_pix=3, n_tgt=3, n_src=1, mu=0.1 * i, se=0.05, tau=0.0,
                              margin=1.0, tost="PASS", guard_verdict="ok", R_median=0.99,
                              placebo_minus_notwc1b_ps=np.nan, f_inflate=f, lever_X_all=7.0)
                         for i, (arm, kind, name, fam) in enumerate(rows)])


def toy_triples(cell):
    """delta_by_triple rows: three interpolating xfer triples (cov_C 0.2, 0.4, 0.9), one
    extrapolating, and a notwc triple without an extrap flag (cov_C 0.33)."""
    rows = [(0.0, 0.2, 1.0), (0.0, 0.4, 2.0), (0.0, 0.9, 6.0), (1.0, 0.7, 4.0)]
    out = [dict(cell=cell, arm=KEY[0], kind=KEY[1], name=KEY[2], extrap=e, cov_C=c, lever_X=x,
                lever_X_all=x) for e, c, x in rows]
    out.append(dict(cell=cell, arm="notwc", kind="pair", name="dut-ref", extrap=np.nan,
                    cov_C=0.33, lever_X=np.nan, lever_X_all=np.nan))
    return pd.DataFrame(out)


def test_an_axis_row_is_the_agg_row_with_the_mean_cov_C_and_lever_X_of_its_triples():
    m = X.merge_triples(toy_agg("c1", 0.8153), toy_triples("c1"))
    D = X.axis_table("Z", [("Z_desy_f0p8153", True, m)]).set_index(["arm", "family"])
    assert list(D.reset_index().columns) == ["arm", "family"] + [
        c for c in X.AXIS_COLUMNS if c not in ("arm", "family")]
    assert D.loc[("xfer", "interp"), "cov_C"] == pytest.approx(0.5)
    assert D.loc[("xfer", "interp"), "lever_X"] == pytest.approx(3.0)
    assert D.loc[("xfer", "extrap"), "cov_C"] == pytest.approx(0.7)
    assert D.loc[("notwc", "na"), "cov_C"] == pytest.approx(0.33)       # no flag: family na
    # agg_by_cell's own lever_X_all and the triples' mean collide in the join: empty, as in the
    # study's axis tables
    assert D.lever_X_all.isna().all()
    assert set(D.agg_tag) == {"Z_desy_f0p8153"} and D.f_variant_is_primary.all()
    assert list(D.reset_index().arm) == ["xfer", "xfer", "notwc"]      # ARM_ORDER


def write_tag(root, folder, cell, f):
    d = os.path.join(root, "tables", folder)
    os.makedirs(d)
    toy_agg(cell, f).to_csv(os.path.join(d, "agg_by_cell.csv"), index=False)
    toy_triples(cell).to_csv(os.path.join(d, "delta_by_triple.csv"), index=False)


def test_axis_A_reads_the_r2_folders_and_labels_its_rows_A_desy(tmp_path):
    root = str(tmp_path / "T")
    for f, tag in ((0.8153, "f0p8153"), (1.0, "f1p0")):
        write_tag(root, "agg_A_r2_desy_" + tag, "A1_P3_run26_0deg", f)
        write_tag(root, "agg_A_desy_" + tag, "A8_P3_run26_pc", f)      # the stage-3 cells
    X.build(root, str(tmp_path / "out"), axes=["A"])
    D = pd.read_csv(tmp_path / "out" / "results_r2" / "axis_A.csv")
    assert set(D.cell) == {"A1_P3_run26_0deg"}
    assert sorted(set(D.agg_tag)) == ["A_desy_f0p8153", "A_desy_f1p0"]
    assert set(D[D.f_variant_is_primary].f_inflate) == {0.8153}


def test_the_wide_table_keeps_the_transfer_row_and_flags_a_self_transfer_row(tmp_path):
    root = str(tmp_path / "T")
    for folder, _, f, _ in C.AXIS_TAGS["F"]:
        write_tag(root, folder, "F3_lin_d2", f)
    # the primary DESY table: beside the transfer row of the interp key, a self-transfer row
    # that differs only in its flag and numbers, after it; on the extrap key the self-transfer
    # row alone
    p = os.path.join(root, "tables", "agg_F_desy_f0p8153", "agg_by_cell.csv")
    agg = pd.read_csv(p)
    xfer = agg[agg.arm == "xfer"]
    real = xfer[xfer.extrap_family == "interp"].assign(n_pix=69, mu=5.32, tost="FAIL")
    null = xfer.assign(self_transfer=1, n_pix=5, mu=0.0, tost="PASS")
    pd.concat([real, null, agg[agg.arm != "xfer"]]).to_csv(p, index=False)
    X.build(root, str(tmp_path / "out"), axes=["F"])
    W = pd.read_csv(tmp_path / "out" / "results_r3" / "axis_F.csv")
    W = W[W.agg_tag == "F_desy_f0p8153"].set_index("family")
    assert W.loc["interp", ["xfer_n_pix", "xfer_mu_ps", "xfer_tost", "xfer_self_transfer"]] \
        .tolist() == [69, 5.32, "FAIL", 0]
    assert W.loc["extrap", ["xfer_n_pix", "xfer_mu_ps", "xfer_self_transfer"]].tolist() == \
        [5, 0.0, 1]
    assert W.loc["interp", "xfer_lin_present"] == False                  # noqa: E712
    assert np.isnan(W.loc["interp", "xfer_lin_self_transfer"])
    # the long table keeps both rows, in the study's columns
    FL = pd.read_csv(tmp_path / "out" / "results_r3" / "axis_F_long.csv")
    assert list(FL.columns) == X.AXIS_COLUMNS
    d = FL[(FL.agg_tag == "F_desy_f0p8153") & (FL.arm == "xfer")]
    assert sorted(d.groupby("family").size().items()) == [("extrap", 1), ("interp", 2)]


def test_the_guardfix_overlay_replaces_its_cells_in_place_and_drops_its_two_columns():
    base = pd.DataFrame(dict(axis="E", cell=["a", "a", "b", "c", "b", "a"], arm="xfer",
                             mu_ps=[1.0, 2.0, 3.0, 5.0, 4.0, 6.0]))
    gf = pd.DataFrame(dict(axis="E", cell=["b", "b", "b", "d"], arm="xfer",
                           mu_ps=[30.0, 31.0, 32.0, 60.0], lpo_shape="own", lpo_pairs=2.0))
    out, counts = X.overlay(base, gf)
    assert list(out.columns) == list(base.columns)
    # every row of b gives way to the guardfix rows, at b's first row; the other rows stay
    assert list(out.cell) == ["a", "a", "b", "b", "b", "c", "a", "d"]
    assert list(out.mu_ps) == [1.0, 2.0, 30.0, 31.0, 32.0, 5.0, 6.0, 60.0]
    assert counts == {"b": (2, 3), "d": (0, 1)}


def test_the_axis_F_cell_table_names_each_level_F_cell_of_the_stage3_list_once(tmp_path):
    lines = X.f_cell_table(J.list_path("cells_stage3.tsv"))
    assert lines[0] == X.F_CELL_HEADING and len(lines) == 4 + len(C.AXIS_F_CELLS) + 1
    assert lines[4].startswith("| F1 | c1, run26, all3, mode c,")
    extra = tmp_path / "cells_stage3.tsv"
    extra.write_text(open(J.list_path("cells_stage3.tsv")).read()
                     + "F12_lin_x\tF12_lin_x\tF\tF_desy\t5\t2000\tcell --cell F12_lin_x\n")
    with pytest.raises(X.AxesError, match="F12_lin_x"):
        X.f_cell_table(str(extra))
