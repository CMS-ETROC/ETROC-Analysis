"""Tests of the source-count ensemble (SPEC.md 3c): the seeded subsample of a cell's source
(--src-cap, --src-seed), the full-range rule of its family (--src-range), the cell table's
ensemble columns, and the job list of the rung x seed grid (ensemble.py).

The toy is one run of three pixel triples, each a quadratic walk per board plus noise, written
as step-10 track files under tmp_path; the source triple's TOT has a sparse high tail, so a
small draw of it covers a narrower range than the whole. The cells are protocol P3 (mode c,
--xfer-board dut) with the largest triple as the source, as in the study's ladder. The width
estimator is replaced by the sample standard deviation (as in test_cell.py), so a cell runs in
seconds.

Run from TestBeam/:  python -m pytest twc_transfer/tests
"""
import os
import shlex
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
from twc_transfer import ensemble as E    # noqa: E402
from twc_transfer import jobs as J        # noqa: E402
from twc_transfer import widths as W      # noqa: E402

COMBO = "dut1-ref2-trig3"
ROLES = ("dut", "ref", "trig")
WALK = {"dut": [2.5e-6, -0.035, 80.0], "ref": [1.5e-6, -0.025, 40.0],
        "trig": [2.0e-6, -0.030, 60.0]}
TRIPLES = (("track_t-R1C1_d-R1C1_r-R1C1", 2400), ("track_t-R2C2_d-R2C2_r-R2C2", 1400),
           ("track_t-R3C3_d-R3C3_r-R3C3", 1300))
SRC = TRIPLES[0][0]
N_SRC = TRIPLES[0][1]
TARGETS = sorted(n for n, _ in TRIPLES[1:])
SELF_ARMS = ("notwc", "notwc_1b", "self_in", "self_cf")
LABEL = ["cov_C", "hist_O", "lever_X", "lever_X_all", "extrap"]


def _tot(rng, n, tail):
    t = rng.uniform(1500.0, 4500.0, n)
    k = rng.random(n) < tail
    t[k] = rng.uniform(4500.0, 6500.0, int(k.sum()))
    return t


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("data"))
    rng = np.random.default_rng(7)
    d = os.path.join(root, "tracks_r1", COMBO, "time")
    os.makedirs(d)
    for i, (name, n) in enumerate(TRIPLES):
        cols = {}
        for r in ROLES:
            tot = _tot(rng, n, 0.03 if i == 0 else 0.01)
            cols["tot_%s" % r] = tot
            cols["toa_%s" % r] = 10000.0 - np.polyval(WALK[r], tot) + rng.normal(0.0, 30.0, n)
        cols["file"] = np.arange(n) // 30
        pd.DataFrame(cols).to_parquet(os.path.join(d, name + ".parquet"))
    return root


def _sd_width(x, seed_salt=0):
    x = np.asarray(x, float)
    return dict(w=float(np.std(x, ddof=1)), ks_d=0.0, ks_p=1.0, accepted=1, n=len(x),
                grid_step=0.0, comps="3",
                iqr=float(np.subtract(*np.percentile(x, [75, 25])) / 1.349),
                rms=float(np.std(x, ddof=1)))


@pytest.fixture
def env(data, tmp_path, monkeypatch):
    monkeypatch.setattr(W, "width_gmm", _sd_width)
    monkeypatch.setenv("ETROC_TWC_TRANSFER_ROOT", str(tmp_path / "T"))
    monkeypatch.setenv("ETROC_TWC_OUT", str(tmp_path / "out"))
    return dict(data=data, out=str(tmp_path / "out"))


def _p3(env, name, *extra):
    """A P3 cell of the toy: the source triple's dut polynomial onto the other two triples."""
    cli.main(["cell", "--cell", name, "--mode", "c", "--xfer-board", "dut", "--src-triple", SRC,
              "--src-k", "1", "--campaign", "toy", "--run", "r1", "--combo", COMBO,
              "--root", env["data"], "--ntrip", "3"] + list(extra))
    return pd.read_csv(os.path.join(env["out"], "cells", "%s.csv" % name))


def _arm(t, arm):
    return t[t.arm == arm].sort_values(["tgt_triple", "kind", "name"]).reset_index(drop=True)


def test_a_capped_source_is_a_seeded_draw_and_the_family_comes_from_the_full_source(env):
    plain = _p3(env, "plain")
    full = _p3(env, "all_s0", "--src-seed", "0")
    s1 = _p3(env, "n100_s1", "--src-cap", "100", "--src-seed", "1")
    s1b = _p3(env, "n100_s1_again", "--src-cap", "100", "--src-seed", "1")
    s2 = _p3(env, "n100_s2", "--src-cap", "100", "--src-seed", "2")
    s3 = _p3(env, "n400_s3", "--src-cap", "400", "--src-seed", "3")
    capped = _p3(env, "n100_s1_capped", "--src-cap", "100", "--src-seed", "1",
                 "--src-range", "capped")

    # the schema: the study's fifty columns, and the ensemble columns after them only for a
    # subsampled source
    assert list(plain.columns) == CL.CELL_COLUMNS
    for t in (full, s1, s2, s3, capped):
        assert list(t.columns) == CL.CELL_COLUMNS + CL.ENSEMBLE_COLUMNS
    assert sorted(plain.tgt_triple.unique()) == TARGETS

    # the cap and the seed are recorded on every row; the fit saw exactly the cap
    for t, cap, seed in ((full, C.N_CAP, 0), (s1, 100, 1), (s2, 100, 2), (s3, 400, 3)):
        assert (t.src_cap == cap).all() and (t.src_seed == seed).all()
        x = _arm(t, "xfer")
        assert (x.n_fit_src == min(cap, N_SRC)).all()
        assert (x.n_src_full == N_SRC).all()

    # the targets are untouched: every arm without a source is the same in every cell
    for arm in SELF_ARMS:
        ref = _arm(plain, arm)["value"].to_numpy()
        for t in (full, s1, s2, s3):
            np.testing.assert_array_equal(_arm(t, arm)["value"].to_numpy(), ref)

    # the draw is seeded: two seeds at one cap fit different events, one seed twice the same
    x1, x2 = _arm(s1, "xfer"), _arm(s2, "xfer")
    assert not np.allclose(x1.value, x2.value)
    np.testing.assert_array_equal(x1.value, _arm(s1b, "xfer").value)
    # seed 0 at the rung 'all' is the cell without the flags
    np.testing.assert_array_equal(_arm(full, "xfer").value, _arm(plain, "xfer").value)

    # the family: taken from the full source, so the same at every cap and seed, and the one
    # of the uncapped cell; the capped draw's own label moves with the draw
    ref_label = _arm(plain, "xfer")[LABEL]
    for t in (full, s1, s2, s3):
        x = _arm(t, "xfer")
        pd.testing.assert_frame_equal(x[LABEL], ref_label)
        pd.testing.assert_frame_equal(
            x[["%s_full" % k for k in LABEL]].set_axis(LABEL, axis=1), ref_label)
    cap1 = _arm(s1, "xfer")[["%s_capped" % k for k in LABEL]]
    cap2 = _arm(s2, "xfer")[["%s_capped" % k for k in LABEL]]
    assert not np.allclose(cap1["cov_C_capped"], cap2["cov_C_capped"])
    assert not np.allclose(cap1["cov_C_capped"], ref_label["cov_C"])
    # at the rung 'all' the two labels coincide
    xa = _arm(full, "xfer")
    for k in LABEL:
        np.testing.assert_array_equal(xa["%s_full" % k], xa["%s_capped" % k])

    # --src-range capped: the table's label is the draw's own (the study's stage-3 ladder)
    xc = _arm(capped, "xfer")
    assert (capped.src_range == "capped").all()
    pd.testing.assert_frame_equal(xc[LABEL],
                                  xc[["%s_capped" % k for k in LABEL]].set_axis(LABEL, axis=1))
    np.testing.assert_array_equal(xc.value, x1.value)

    # a self arm carries no label and no source count
    assert _arm(s1, "self_cf")[["cov_C_full", "cov_C_capped", "n_src_full"]].isna().all().all()

    # the aggregation reads the ensemble tables as cell tables: the family of every target is
    # the full-range one, at every cap and seed
    res = A.aggregate(A.read_cells([os.path.join(env["out"], "cells", n + ".csv")
                                    for n in ("all_s0", "n100_s1", "n100_s2", "n400_s3")]))
    fam = res["agg_by_cell"].query("arm == 'xfer' and kind == 'board'")
    assert fam.groupby("cell")["extrap_family"].apply(tuple).nunique() == 1


def test_a_subsampled_source_is_refused_where_no_source_is_read(env):
    for extra, why in ((["--mode", "e"], "mode e"), (["--mode", "inrun"], "mode inrun"),
                       (["--mode", "c", "--src-triple", SRC, "--src-bank", "b.parquet"],
                        "--src-bank")):
        argv = ["cell", "--cell", "r", "--campaign", "toy", "--run", "r1", "--combo", COMBO,
                "--root", env["data"], "--src-cap", "100"] + extra
        with pytest.raises(SystemExit, match="reads no such source") as e:
            cli.main(argv)
        assert why in str(e.value)
    with pytest.raises(SystemExit, match=">= 0"):
        _p3(env, "neg", "--src-cap", "100", "--src-seed", "-1")


def test_a_resumed_ensemble_table_keeps_its_schema(env):
    _p3(env, "res", "--src-cap", "100", "--src-seed", "4", "--max-triples", "1", "--flush", "1")
    t = _p3(env, "res", "--src-cap", "100", "--src-seed", "4", "--flush", "1")
    assert sorted(t.tgt_triple.unique()) == TARGETS
    assert not t.duplicated(["tgt_triple", "arm", "kind", "name"]).any()
    # the same table under the plain schema is refused before anything is computed
    with pytest.raises(SystemExit, match="different schema"):
        _p3(env, "res")


STAGE3 = "cells_stage3.tsv"


def _template(job_id):
    return [j for j in J.read_jobs(J.list_path(STAGE3)) if j["job_id"] == job_id][0]


def test_the_grid_of_the_stage3_p3_ladder():
    rungs = E.parse_rungs(",".join(cli.ENSEMBLE_RUNGS))
    assert rungs == [100, 200, 400, 800, 1600, 3200, 0]
    for tid in ("G_P3_run26_nall", "G_P3_run26_n100"):
        tmpl = _template(tid)
        rows = E.ensemble_jobs(tmpl, rungs, 20)
        assert len(rows) == 6 * 20 + 1
        ids = [j["cell_id"] for j in rows]
        assert len(set(ids)) == len(ids) and ids[0] == "G_P3_run26_n100_s00"
        assert ids[-1] == "G_P3_run26_nall_s00" and ids[-2] == "G_P3_run26_n3200_s19"
        assert {j["group"] for j in rows} == {"Gens_desy"}
        grid = [(n, s) for n in rungs for s in (range(20) if n else (0,))]
        same = None
        for j, (n, s) in zip(rows, grid):
            toks = shlex.split(J.expand(j["command"], C.CAMPAIGNS, "/read", "/write"))
            ns = cli.parse_command(toks)
            assert ns.cell == j["cell_id"] == E.cell_id("G_P3_run26", n, s)
            assert (ns.src_cap, ns.src_seed, ns.src_range) == (n, s, "full")
            assert ns.out == "/write/cells/%s/%s.csv" % (ns.cell, ns.cell)
            # everything else is the template's: the same target, source rule and floors
            rest = {k: v for k, v in vars(ns).items()
                    if k not in ("cell", "src_cap", "src_seed", "src_range", "out", "func")}
            same = same or rest
            assert rest == same
        assert (same["mode"], same["xfer_board"], same["src_pick"], same["src_k"],
                same["run"], same["combo"], same["ntrip"], same["cap"]) == (
                    "c", "dut", "pixpop", 1, "run26", "ref1-trig2-dut3", "60", 6000)
    smoke = E.smoke_job(_template("G_P3_run26_nall"), rungs)
    ns = cli.parse_command(shlex.split(J.expand(smoke["command"], C.CAMPAIGNS, "/r", "/w")))
    assert (smoke["level"], ns.src_cap, ns.src_seed, ns.ntrip) == ("smoke", 100, 1, "3")


def test_the_ensemble_command_writes_a_list_the_driver_takes(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ETROC_TWC_TRANSFER_ROOT", str(tmp_path / "read"))
    monkeypatch.setenv("ETROC_TWC_OUT", str(tmp_path / "out"))
    cli.main(["ensemble", "--list", STAGE3, "--template", "G_P3_run26_nall", "--est-min", "30",
              "--write-dag", "dag_g"])
    path = str(tmp_path / "out" / "ensemble" / "cells_G_P3_run26_ensemble.tsv")
    rows = J.read_jobs(path)
    assert len(rows) == 122 and rows[0]["job_id"] == "SMOKE"
    assert sum(j["est_min"] for j in rows[1:]) == 121 * 30
    dag = open(str(tmp_path / "out" / "dag_g" / "transfer_G_P3_run26_ensemble.dag")).read()
    assert dag.count("\nJOB ") == 121 + 3
    text = capsys.readouterr().out
    assert "write-dag: 121 jobs + 1 smoke + 1 prep + 1 agg" in text
    assert "nothing was submitted" in text
    plan = J.agg_plan(J.expanded_jobs(path, C.CAMPAIGNS, str(tmp_path / "read"),
                                      str(tmp_path / "out"), cli.parse_command),
                      str(tmp_path / "out"))
    assert [p["tag"] for p in plan] == ["agg_Gens_desy_f0p8153", "agg_Gens_desy_f1p0"]
    assert all(len(p["cells"]) == 121 for p in plan)
    # one folder per cell, so the aggregation reads every table once
    pats = [pat for p in plan[:1] for pats in p["cells"].values() for pat in pats]
    assert len(set(pats)) == len(pats) == 121
    with pytest.raises(SystemExit, match="exists"):
        cli.main(["ensemble", "--list", STAGE3, "--template", "G_P3_run26_nall"])


def test_a_grid_whose_source_floor_exceeds_a_rung_is_refused(tmp_path, monkeypatch):
    # mode a checks --nmin-src on the drawn source, so every source of a rung below it is
    # refused; the grid says so before any job is written
    monkeypatch.setenv("ETROC_TWC_TRANSFER_ROOT", str(tmp_path / "read"))
    monkeypatch.setenv("ETROC_TWC_OUT", str(tmp_path / "out"))
    with pytest.raises(SystemExit, match="source floor --nmin-src 1000"):
        cli.main(["ensemble", "--list", "cells.tsv", "--template", "a1_twin_26to27",
                  "--rungs", "100,all", "--seeds", "2"])
    assert not os.path.exists(str(tmp_path / "out" / "ensemble"))
    cli.main(["ensemble", "--list", "cells.tsv", "--template", "a1_twin_26to27",
              "--rungs", "1000,all", "--seeds", "2"])
    assert len(J.read_jobs(str(tmp_path / "out" / "ensemble" / "cells_a1_twin_26to27_ensemble.tsv"))) == 4
