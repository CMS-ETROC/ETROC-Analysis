"""Tests of the cell command (cell.py): the self_transfer flag, the line naming a self-transfer
target's in-sample arms, the role-map refusal of the modes without a named source, resuming a
cell table, the dry run, a relative --out, and the table's columns.

The toy is a campaign of two runs with the same three pixel triples, each a quadratic walk per
board plus noise, written as step-10 track files under tmp_path. The width estimator is
replaced by the sample standard deviation so that a cell runs in seconds; what is tested is the
cell's bookkeeping, not the width fit (test_widths.py covers that).

Run from TestBeam/:  python -m pytest twc_transfer/tests
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

TESTBEAM = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, TESTBEAM)
from twc_transfer import cell as CL       # noqa: E402
from twc_transfer import cli              # noqa: E402
from twc_transfer import widths as W      # noqa: E402

COMBO = "dut1-ref2-trig3"
ROLES = ("dut", "ref", "trig")
WALK = {"dut": [2.5e-6, -0.035, 80.0], "ref": [1.5e-6, -0.025, 40.0],
        "trig": [2.0e-6, -0.030, 60.0]}
# three triples per run, distinct pixels on every board, largest first
TRIPLES = (("track_t-R1C1_d-R1C1_r-R1C1", 1500), ("track_t-R2C2_d-R2C2_r-R2C2", 1400),
           ("track_t-R3C3_d-R3C3_r-R3C3", 1300))
A_NAME = TRIPLES[0][0]

# the cell table of the study, in its order
STUDY_COLUMNS = [
    "cell", "mode", "campaign", "run", "combo", "tgt_triple", "tgt_key", "src_run",
    "src_combo", "src_triple", "n_fit_src", "rho", "self_transfer", "same_pixel", "same_board",
    "src_file_overlap", "src_keys_checked", "keys_comparable", "n_pool_excluded", "xfer_board",
    "src_role", "src_pick", "src_pix_rc", "src_pix_ntrk", "src_pix_pop", "lpo_pairs", "ts",
    "arm", "kind", "name", "value", "v_f0", "v_f1", "n", "n_f0", "n_f1", "ks_d", "ks_p",
    "accepted", "comps", "grid_step", "iqr", "rms", "own_slope", "neg_radicand", "cov_C",
    "hist_O", "lever_X", "lever_X_all", "extrap",
]


def _write_run(data, run, seed):
    rng = np.random.default_rng(seed)
    d = os.path.join(data, "tracks_%s" % run, COMBO, "time")
    os.makedirs(d)
    for name, n in TRIPLES:
        cols = {}
        for r in ROLES:
            tot = rng.uniform(1500.0, 5000.0, n)
            cols["tot_%s" % r] = tot
            cols["toa_%s" % r] = 10000.0 - np.polyval(WALK[r], tot) + rng.normal(0.0, 30.0, n)
        cols["file"] = np.arange(n) // 30
        pd.DataFrame(cols).to_parquet(os.path.join(d, name + ".parquet"))


def _sd_width(x, seed_salt=0):
    x = np.asarray(x, float)
    return dict(w=float(np.std(x, ddof=1)), ks_d=0.0, ks_p=1.0, accepted=1, n=len(x),
                grid_step=0.0, comps="3",
                iqr=float(np.subtract(*np.percentile(x, [75, 25])) / 1.349),
                rms=float(np.std(x, ddof=1)))


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("data"))
    _write_run(root, "r1", 1)
    _write_run(root, "r2", 2)
    return root


@pytest.fixture
def env(data, tmp_path, monkeypatch):
    monkeypatch.setattr(W, "width_gmm", _sd_width)
    monkeypatch.setenv("ETROC_TWC_TRANSFER_ROOT", str(tmp_path / "T"))
    monkeypatch.setenv("ETROC_TWC_OUT", str(tmp_path / "out"))
    return dict(data=data, out=str(tmp_path / "out"))


def _cell(env, name, *extra):
    return ["cell", "--cell", name, "--campaign", "toy", "--run", "r1", "--combo", COMBO,
            "--root", env["data"], "--ntrip", "3"] + list(extra)


def _table(env, name):
    return pd.read_csv(os.path.join(env["out"], "cells", "%s.csv" % name))


def _tree(top):
    out = []
    for d, _, fs in os.walk(top):
        out += [(os.path.relpath(os.path.join(d, f), top),
                 os.path.getsize(os.path.join(d, f))) for f in fs]
    return sorted(out)


def test_self_transfer_is_the_source_triple_in_the_targets_own_run_and_combo(env):
    # mode c, source = triple A of the target run: A is its own source, B and C are not
    cli.main(_cell(env, "c_same", "--mode", "c", "--src-triple", A_NAME, "--src-k", "1"))
    t = _table(env, "c_same")
    x = t[t["arm"] == "xfer"]
    assert sorted(x["tgt_triple"].unique()) == sorted(n for n, _ in TRIPLES)
    flag = x.groupby("tgt_triple")["self_transfer"].agg(["min", "max"])
    assert flag.loc[A_NAME].tolist() == [1, 1]
    assert (flag.drop(A_NAME) == 0).all().all()
    # the self-transfer target's xfer arm is its own fold-matched self fit (SPEC 5.5)
    a = t[t["tgt_triple"] == A_NAME].set_index(["arm", "kind", "name"])["value"]
    np.testing.assert_array_equal(a.loc["xfer"].sort_index().to_numpy(),
                                  a.loc["self_cf"].sort_index().to_numpy())
    # a self arm carries no source, so no flag
    assert (t[t["arm"] == "self_cf"]["self_transfer"] == 0).all()

    # the same source triple name in the OTHER run is a transfer, not a self-transfer
    cli.main(_cell(env, "c_other", "--mode", "c", "--src-run", "r2", "--src-triple", A_NAME,
                   "--src-k", "1"))
    cli.main(_cell(env, "a_other", "--mode", "a", "--src-run", "r2"))
    for name in ("c_other", "a_other"):
        t = _table(env, name)
        x = t[t["arm"] == "xfer"]
        assert len(x["tgt_triple"].unique()) == 3
        assert (x["self_transfer"] == 0).all()
        assert (x["src_run"] == "r2").all()


@pytest.mark.parametrize("mode", ["e", "inrun"])
def test_role_map_is_refused_in_the_modes_without_a_named_source(env, mode):
    with pytest.raises(SystemExit, match="--role-map is meaningless in mode %s" % mode):
        cli.main(_cell(env, "rm_" + mode, "--mode", mode,
                       "--role-map", "dut:ref,ref:dut,trig:trig"))
    assert not os.path.exists(os.path.join(env["out"], "cells", "rm_%s.csv" % mode))


def test_resume_appends_the_missing_triples_and_never_duplicates_one(env):
    argv = _cell(env, "res", "--mode", "a", "--src-run", "r2", "--flush", "1")
    dest = os.path.join(env["out"], "cells", "res.csv")
    cli.main(argv + ["--max-triples", "1"])
    first = _table(env, "res")
    assert first["tgt_triple"].unique().tolist() == [A_NAME]
    with open(dest) as fh:
        first_text = fh.read()
    cli.main(argv)
    second = _table(env, "res")
    per = second.groupby("tgt_triple").size()
    assert sorted(per.index) == sorted(n for n, _ in TRIPLES)
    assert per.loc[A_NAME] == len(first)
    with open(dest) as fh:
        assert fh.read().startswith(first_text)
    assert not second.duplicated(["tgt_triple", "arm", "kind", "name"]).any()
    size = os.path.getsize(dest)
    cli.main(argv)
    assert os.path.getsize(dest) == size
    # --overwrite starts the table over
    cli.main(argv + ["--overwrite", "--max-triples", "1"])
    assert _table(env, "res")["tgt_triple"].unique().tolist() == [A_NAME]


def test_dry_run_prints_the_plan_and_writes_nothing(env, capsys):
    before = _tree(env["data"])
    cli.main(_cell(env, "dry", "--mode", "c", "--src-triple", A_NAME, "--src-k", "1",
                   "--dry-run"))
    text = capsys.readouterr().out
    assert "cell dry: 3 target triples x 1 sources x 8 arms" in text
    assert "mode c; target toy r1 %s" % COMBO in text
    assert "writes %s" % os.path.join(env["out"], "cells", "dry.csv") in text
    assert "nothing computed, nothing written" in text
    assert not os.path.exists(env["out"])
    assert _tree(env["data"]) == before
    # an existing table survives a dry run that names --overwrite
    cli.main(_cell(env, "dry", "--mode", "a", "--src-run", "r2", "--max-triples", "1"))
    dest = os.path.join(env["out"], "cells", "dry.csv")
    size = os.path.getsize(dest)
    cli.main(_cell(env, "dry", "--mode", "a", "--src-run", "r2", "--overwrite", "--dry-run"))
    assert os.path.getsize(dest) == size


def test_a_relative_out_is_taken_in_the_write_root(env, capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cli.main(_cell(env, "rel", "--mode", "a", "--src-run", "r2", "--out",
                   os.path.join("sub", "rel.csv"), "--dry-run"))
    assert "writes %s " % os.path.join(env["out"], "sub", "rel.csv") in capsys.readouterr().out


def test_a_self_transfer_target_names_its_in_sample_arms(env, capsys):
    cli.main(_cell(env, "c_ext", "--mode", "c", "--src-triple", A_NAME, "--src-k", "1",
                   "--arms", "ext", "--xfer-lin"))
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "self-transfer target" in ln]
    assert len(lines) == 1 and A_NAME in lines[0]
    assert lines[0].endswith("in-sample: xfer_1b:dut, xfer_1b:ref, xfer_1b:trig, xfer_lin")
    t = _table(env, "c_ext")
    assert set(t[t.self_transfer == 1].tgt_triple) == {A_NAME}


def test_cell_table_columns_are_the_fifty_of_the_study(env):
    assert CL.CELL_COLUMNS == STUDY_COLUMNS
    assert len(STUDY_COLUMNS) == 50 and len(set(STUDY_COLUMNS)) == 50
    cli.main(_cell(env, "cols", "--mode", "a", "--src-run", "r2", "--max-triples", "1"))
    with open(os.path.join(env["out"], "cells", "cols.csv")) as fh:
        assert fh.readline().strip().split(",") == STUDY_COLUMNS
    with pytest.raises(SystemExit, match="outside CELL_COLUMNS"):
        CL._flush([dict(cell="x", not_a_column=1)], os.path.join(env["out"], "x.csv"))
