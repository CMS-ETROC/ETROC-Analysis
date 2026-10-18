"""Tests of the aggregation (agg.py) on toy cell tables: the symmetric censoring, the one-board
identical-pair drop, the same-pixel rule, the k_cal exclusions, the structural-zero stop, the
guard verdict, the primary flag, Holm over board FAIL claims, the choice of f, and the three
tables `agg` writes.

A toy cell is a few target triples of one run, every arm with its three pair widths and three
board sigmas, each arm a fixed offset from self_cf plus noise.

Run from TestBeam/:  python -m pytest twc_transfer/tests
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

TESTBEAM = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, TESTBEAM)
from twc_transfer import agg as A         # noqa: E402
from twc_transfer import campaigns as C   # noqa: E402
from twc_transfer import cell as CL       # noqa: E402
from twc_transfer import cli              # noqa: E402
from twc_transfer import stats as St      # noqa: E402

COMBO = "dut1-ref2-trig3"
PAIRS = ("dut-ref", "dut-trig", "ref-trig")
BOARDS = ("dut", "ref", "trig")
OFFSET = dict(notwc=20.0, notwc_1b=19.5, self_cf=0.0, self_in=-0.2, self_cf_lpo=0.2, xfer=0.3,
              xfer_lpo=0.4, placebo=19.0, xfer_clip=0.5)
ONE_BOARD_ARMS = ("xfer", "xfer_lpo", "placebo", "xfer_clip")   # rows that name xfer_board


def toy_cell(cell="c1", arms=("notwc", "self_cf", "xfer", "xfer_lpo", "self_cf_lpo", "placebo"),
             ntgt=4, xb=None, seed=0):
    """A cell table (CELL_COLUMNS). With `xb`, a one-board cell: the transfer-derived arms name
    xfer_board = xb, and their pair without xb is exactly self_cf's (SPEC 3b)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(ntgt):
        tgt = "track_d-R%dC1_r-R%dC1_t-R%dC1" % (i, i, i)
        base = {q: 40.0 + rng.normal(0.0, 2.0) for q in PAIRS + BOARDS}
        err = {q: rng.normal(0.0, 0.1) for q in PAIRS + BOARDS}
        for arm in arms:
            for kind, names in (("pair", PAIRS), ("board", BOARDS)):
                for nm in names:
                    same = (xb is not None and kind == "pair" and xb not in nm.split("-")
                            and A.is_one_board_arm(arm))
                    if arm == "self_cf" or same:
                        v, e = base[nm], err[nm]
                    else:
                        v, e = base[nm] + OFFSET[arm] + rng.normal(0.0, 0.1), rng.normal(0.0, 0.1)
                    rows.append(dict(
                        cell=cell, mode="a", campaign="jul", run="r1", combo=COMBO,
                        tgt_triple=tgt, src_run="r0", src_combo=COMBO, src_triple="track_src",
                        self_transfer=0, same_pixel=0,
                        xfer_board=xb if (xb and arm in ONE_BOARD_ARMS) else np.nan,
                        arm=arm, kind=kind, name=nm, value=v, v_f0=v + e, v_f1=v - e, n=1000,
                        accepted=1, neg_radicand=0, extrap=0, src_file_overlap=0.0,
                        n_pool_excluded=0.0, lever_X_all=0.0, ts="2026-10-18T00:00:00"))
    return pd.DataFrame(rows).reindex(columns=CL.CELL_COLUMNS)


def test_a_degenerate_value_censors_its_quantity_in_every_arm():
    df = toy_cell(arms=("notwc", "self_cf", "xfer"), ntgt=2)
    t1, t2 = sorted(df.tgt_triple.unique())
    nan_row = (df.arm == "xfer") & (df.tgt_triple == t1) & (df.kind == "board") & (df.name == "dut")
    neg_row = (df.arm == "notwc") & (df.tgt_triple == t2) & (df.kind == "board") & (df.name == "ref")
    df.loc[nan_row, "value"] = np.nan
    df.loc[neg_row, "neg_radicand"] = 1
    out, trig, n = A.censor_degenerate(df)
    assert trig == {"xfer": 1, "notwc": 1}
    assert n == 6 and len(out) == len(df) - 6           # two quantities, three arms each
    q = set(zip(out.tgt_triple, out.kind, out.name))
    assert (t1, "board", "dut") not in q and (t2, "board", "ref") not in q
    assert (t1, "board", "ref") in q and (t2, "board", "dut") in q
    assert out.groupby("arm").size().to_dict() == {"notwc": 10, "self_cf": 10, "xfer": 10}
    assert A.censor_degenerate(toy_cell(ntgt=1))[2] == 0


def test_a_triple_below_the_n_eval_floor_leaves_every_arm():
    df = toy_cell(ntgt=2)
    t1 = sorted(df.tgt_triple.unique())[0]
    df.loc[(df.tgt_triple == t1) & (df.arm == "xfer") & (df.kind == "board"), "n"] = 599
    out, n = A.drop_below_nmin_eval(df, 600)
    assert n == (df.tgt_triple == t1).sum() and t1 not in set(out.tgt_triple)
    assert A.drop_below_nmin_eval(df, 0)[1] == 0


def test_one_board_cell_drops_the_pair_without_the_transferred_board():
    arms = ("notwc", "notwc_1b", "self_cf", "xfer", "xfer_lpo", "self_cf_lpo", "placebo",
            "xfer_clip")
    one = toy_cell("c1", arms, ntgt=2, xb="dut")
    all3 = toy_cell("c2", arms, ntgt=2)
    two = toy_cell("c3", arms, ntgt=2, xb="dut")
    two.loc[(two.arm == "xfer_clip"), "xfer_board"] = "ref"
    df = pd.concat([one, all3, two], ignore_index=True)
    xb = A.xfer_board_by_cell(df)
    assert xb == {"c1": "dut"}                          # c3 names two boards, c2 none
    mask = A.identical_pair_mask(df, xb)
    hit = df[mask]
    assert set(hit.cell) == {"c1"} and set(hit.kind) == {"pair"} and set(hit.name) == {"ref-trig"}
    assert set(hit.arm) == {"xfer", "placebo", "notwc_1b", "xfer_clip"}
    assert len(hit) == 2 * 4
    out, n = A.drop_identical_pairs(df, xb, "self_cf")
    assert n == 8 and len(out) == len(df) - 8
    kept = out[(out.cell == "c1") & (out.name == "ref-trig")]
    assert set(kept.arm) == {"notwc", "self_cf", "xfer_lpo", "self_cf_lpo"}
    out, n = A.drop_identical_pairs(df, xb, "notwc_1b")
    assert n == 0 and len(out) == len(df)
    assert A.is_one_board_arm("xfer_1b:dut") and not A.is_one_board_arm("placebo_self")


def test_same_pixel_refits_are_transfer_rows_of_the_same_run_and_combo_only():
    d = pd.DataFrame(dict(
        arm=["xfer", "xfer", "xfer", "placebo", "xfer", "xfer_lin"],
        same_pixel=[1, 1, 1, 1, 0, 1],
        run=["r1"] * 6, src_run=["r1", "r0", "r1", "r1", "r1", "r1"],
        combo=[COMBO] * 6, src_combo=[COMBO, COMBO, "x", COMBO, COMBO, COMBO]))
    assert list(A.same_pixel_mask(d)) == [True, False, False, False, False, True]


def test_kcal_leaves_out_self_transfer_and_same_pixel_rows_but_never_a_self_arm():
    def grp(arm, kind, diffs, st):
        return pd.DataFrame(dict(cell="c1", run="r1", arm=arm, kind=kind, d_f0=diffs,
                                 d_f1=0.0, scale=1.0, self_transfer=st))
    d = pd.concat([grp("xfer", "pair", [1.0, 2.0, 3.0, 100.0, 50.0], [0, 0, 0, 1, 0]),
                   grp("self_in", "pair", [1.0, 3.0, 10.0], [1, 1, 0]),
                   grp("placebo", "board", [2.0, 4.0], [1, 1])], ignore_index=True)
    drop_sp = np.zeros(len(d), bool)
    drop_sp[4] = True                                    # the xfer row of 50: a same-pixel refit
    excl, st = A.kcal_exclusion_mask(d, drop_sp)
    assert list(np.where(excl)[0]) == [3, 4, 8, 9] and list(np.where(st)[0]) == [3, 8, 9]
    kc, fallback, trim = A.kcal_by_group(d, excl)
    k = 2 * 0.6745
    assert kc[("c1", "r1", "xfer", "pair")] == pytest.approx(2.0 / k, rel=1e-12)
    assert kc[("c1", "r1", "self_in", "pair")] == pytest.approx(3.0 / k, rel=1e-12)
    # every row excluded: the group falls back to all of its rows
    assert kc[("c1", "r1", "placebo", "board")] == pytest.approx(3.0 / k, rel=1e-12)
    assert fallback == [("c1", "r1", "placebo", "board")] and trim == 4


def _write_cell(root, df, name="c1"):
    p = os.path.join(root, "cells", name, name + ".csv")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    df.to_csv(p, index=False)
    return p


def _roots(monkeypatch, tmp_path):
    monkeypatch.setenv("ETROC_TWC_TRANSFER_ROOT", str(tmp_path / "T"))
    monkeypatch.setenv("ETROC_TWC_OUT", str(tmp_path / "out"))
    return str(tmp_path / "T"), str(tmp_path / "out")


def test_a_structural_zero_in_a_one_board_pair_pool_stops_before_anything_is_written(
        monkeypatch, tmp_path):
    df = toy_cell(arms=("notwc", "self_cf", "xfer", "xfer_lpo", "self_cf_lpo"), ntgt=3, xb="dut")
    t1 = sorted(df.tgt_triple.unique())[0]
    base = df[(df.arm == "self_cf") & (df.tgt_triple == t1) & (df.name == "dut-ref")]
    zero = (df.arm == "xfer") & (df.tgt_triple == t1) & (df.name == "dut-ref")
    for c in ("value", "v_f0", "v_f1"):                  # the target's own source: xfer is self_cf
        df.loc[zero, c] = base[c].to_numpy()
    with pytest.raises(A.StructuralZeroError, match=r"\('c1', 'xfer', 'pair'\)"):
        A.aggregate(df, f_inflate=1.0)
    T, out = _roots(monkeypatch, tmp_path)
    _write_cell(T, df)
    with pytest.raises(SystemExit, match="Nothing has been written"):
        cli.main(["agg", "--cells", "cells/*/*.csv", "--f-inflate", "1.0"])
    assert not os.path.exists(os.path.join(out, "tables", "agg_by_cell.csv"))
    assert not os.listdir(os.path.join(out, "tables"))
    # flagged as the cell's own source triple, the row sets no scale and the aggregation runs
    df.loc[(df.tgt_triple == t1) & ~df.arm.isin(A.SELF_ARMS), "self_transfer"] = 1
    res = A.aggregate(df, f_inflate=1.0)
    st = res["agg_by_cell"].query("arm == 'xfer' and kind == 'pair'").self_transfer
    assert set(st) == {0, 1}


def test_guard_verdict_and_no_transfer_pass_without_a_guard():
    has, cheat, v = A.guard_verdict(1.0, np.nan, 0.5, 1.0)
    assert (has, v) == (0, "absent") and np.isnan(cheat)
    assert A.guard_verdict(1.0, 0.8, 1.5, 1.0) == (1, pytest.approx(1.7), "uninformative")
    assert A.guard_verdict(1.0, 0.8, -1.5, 1.0)[2] == "uninformative"
    assert A.guard_verdict(1.0, 0.8, 0.5, 1.0) == (1, pytest.approx(0.7), "ok")
    has, cheat, v = A.guard_verdict(1.0, 0.8, np.nan, 1.0)
    assert (has, v) == (1, "ok") and np.isnan(cheat)
    assert A.tost_label("xfer", "PASS", "absent", 0) == "NA-unguarded"
    assert A.tost_label("xfer_clip", "FAIL", "uninformative", 0) == "NA-unguarded"
    assert A.tost_label("xfer", "PASS", "absent", 1) == "PASS"
    assert A.tost_label("placebo", "PASS", "absent", 0) == "PASS"
    assert A.tost_label("xfer", "FAIL", "ok", 0) == "FAIL"
    res = A.aggregate(toy_cell(arms=("notwc", "self_cf", "xfer", "placebo")), f_inflate=1.0)
    x = res["agg_by_cell"].query("arm == 'xfer'")
    assert set(x.guard_verdict) == {"absent"} and set(x.tost) == {"NA-unguarded"}
    assert set(x.tost_raw) <= {"PASS", "FAIL"}
    assert set(res["agg_by_cell"].query("arm == 'placebo'").tost) <= {"PASS", "FAIL"}


def test_primary_is_a_pair_row_except_xfer_lpo_on_the_untransferred_pair():
    assert A.primary_flag("pair", "xfer", "dut-ref", None) == 1
    assert A.primary_flag("board", "xfer", "dut", None) == 0
    assert A.primary_flag("pair", "xfer_lpo", "ref-trig", "dut") == 0
    assert A.primary_flag("pair", "xfer_lpo", "dut-ref", "dut") == 1
    assert A.primary_flag("pair", "xfer_lpo", "ref-trig", None) == 1
    assert A.primary_flag("pair", "xfer", "ref-trig", "dut") == 1
    res = A.aggregate(toy_cell(ntgt=3, xb="dut"), f_inflate=1.0)
    lpo = res["agg_by_cell"].query("arm == 'xfer_lpo' and kind == 'pair'").set_index("name")
    assert lpo.primary.to_dict() == {"dut-ref": 1, "dut-trig": 1, "ref-trig": 0}


def test_holm_runs_over_each_cells_board_fail_claims_only():
    agg = pd.DataFrame(dict(
        cell=["c1", "c1", "c1", "c1", "c2"], kind=["board", "board", "board", "pair", "board"],
        tost=["FAIL", "FAIL", "PASS", "FAIL", "FAIL"],
        p_nonequiv=[0.01, 0.04, 0.001, 0.001, 0.03]))
    out = A.holm_board_fails(agg)
    np.testing.assert_allclose(out.p_ne_holm, [0.02, 0.04, np.nan, np.nan, 0.03], rtol=1e-12)
    assert list(out.fail_holm) == [1, 1, 0, 0, 1]


def test_f_inflate_is_explicit_or_the_campaigns_four_decimal_value_or_f_cal_json(tmp_path):
    assert C.F_CAL["aug"][:2] == (0.8979, 0.8979111975615606)
    assert A.f_inflate_for(0.5, None) == (0.5, "explicit")
    assert A.f_inflate_for(None, "aug")[0] == 0.8979
    assert A.f_inflate_for(None, "aug", "full")[0] == 0.8979111975615606
    assert A.f_inflate_for(None, "jul")[0] == 0.9259
    with pytest.raises(ValueError):
        A.f_inflate_for(None, "aug", "rounded")
    for c, (four, full, _) in C.F_CAL.items():
        assert four == round(full, 4), c
    tables = tmp_path / "tables"
    tables.mkdir()
    assert A.f_inflate_for(None, None, read_root=str(tmp_path)) == (1.0, "default 1.0 (uncalibrated)")
    (tables / "f_cal.json").write_text('{"f_cal": 0.8152579445510614, "run": "run26", "ref": "sub"}')
    f, src = A.f_inflate_for(None, None, read_root=str(tmp_path))
    assert f == 0.8152579445510614 and src == "tables/f_cal.json (run26, ref=sub)"
    a = cli.parse_command(["agg", "--cells", "x.csv", "--f-cal", "aug"])
    assert (a.f_cal, a.f_precision, a.f_inflate) == ("aug", "agg", None)
    with pytest.raises(ValueError, match="not allowed with"):
        cli.parse_command(["agg", "--cells", "x.csv", "--f-cal", "aug", "--f-inflate", "1"])


def test_agg_defaults_are_the_studys():
    a = vars(cli.parse_command(["agg", "--cells", "a.csv", "b.csv"]))
    a.pop("func")
    assert a == dict(cmd="agg", cells=["a.csv", "b.csv"], baseline="self_cf", f_inflate=None,
                     f_cal=None, f_precision="agg", nmin_eval=600, tier0=None, out=None)


def test_agg_writes_the_three_tables_under_the_write_root(monkeypatch, tmp_path):
    T, out = _roots(monkeypatch, tmp_path)
    df = toy_cell(ntgt=5)
    df.loc[df.tgt_triple == sorted(df.tgt_triple.unique())[4], "same_pixel"] = 1
    df["src_run"] = "r1"                                 # same run: the same-pixel rule applies
    p = _write_cell(T, df)
    shape = {t: {"self": {"dut": "own"}} for t in sorted(df.tgt_triple.unique())[:2]}
    with open(os.path.splitext(p)[0] + ".lpo_shape.json", "w") as fh:
        fh.write(json.dumps(shape))
    assert cli.main(["agg", "--cells", "cells/*/*.csv", "--f-cal", "jul"]) == 0
    tabs = os.path.join(out, "tables")
    dbt = pd.read_csv(os.path.join(tabs, "delta_by_triple.csv"))
    agg = pd.read_csv(os.path.join(tabs, "agg_by_cell.csv"))
    cal = pd.read_csv(os.path.join(tabs, "calibration.csv"))
    rest = [c for c in CL.CELL_COLUMNS if c not in A.DELTA_KEY]
    assert list(dbt.columns) == A.DELTA_KEY + rest + ["lpo_shape"] + A.DELTA_COLUMNS
    assert list(agg.columns) == A.AGG_COLUMNS[:-2] + ["lpo_shape"] + A.AGG_COLUMNS[-2:]
    assert list(cal.columns) == A.CALIBRATION_COLUMNS
    assert set(cal.f_source) == {"f_cal jul, as aggregated"} and set(cal.f_inflate) == {0.9259}
    assert len(dbt) == 5 * 5 * 6 and "self_cf" not in set(dbt.arm)
    np.testing.assert_allclose(dbt.delta, dbt.value - dbt.value_b, rtol=0, atol=1e-12)
    np.testing.assert_allclose(dbt.se, 0.9259 * dbt.k_cal * dbt.value_b / np.sqrt(dbt.n_b),
                               rtol=1e-12)
    # the same-pixel triple is in the per-triple table, in no transfer aggregate and in no pool
    x = agg.query("arm == 'xfer'")
    assert set(x.n_pix) == {4} and set(agg.query("arm == 'notwc'").n_pix) == {5}
    g = dbt.query("arm == 'xfer' and kind == 'pair' and same_pixel == 0")
    k = St.kcal_from_cells(g)
    assert cal.query("arm == 'xfer' and kind == 'pair'").k_cal.iloc[0] == pytest.approx(k, rel=1e-12)
    assert set(agg.lpo_shape.fillna("")) == {'{"self": {"dut": "own"}}'}
    assert list(agg[["cell", "kind", "arm", "name"]].itertuples(index=False)) == \
        sorted(agg[["cell", "kind", "arm", "name"]].itertuples(index=False))
