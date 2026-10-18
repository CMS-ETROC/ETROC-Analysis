"""Tests of the command line: the write-root rule, the job-list dry run, the parser against the
study's commands, and the PYTHONHASHSEED restart of python -m twc_transfer.

Run from TestBeam/:  python -m pytest twc_transfer/tests
"""
import os
import shlex
import subprocess
import sys

import pytest

TESTBEAM = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, TESTBEAM)
from twc_transfer import __main__ as M    # noqa: E402
from twc_transfer import campaigns as C   # noqa: E402
from twc_transfer import cli              # noqa: E402
from twc_transfer import jobs as J        # noqa: E402

WRITING = [["index", "--campaign", "mar", "--run", "no_such_run"],
           ["bank", "--campaign", "mar", "--run", "no_such_run"],
           ["fit", "--campaign", "mar", "--run", "no_such_run"],
           ["tier0", "--bank", "bank/no_such_bank.parquet", "--tag", "t"],
           ["cell", "--cell", "x", "--mode", "a", "--campaign", "mar", "--run", "no_such_run"],
           ["transfer", "--cell", "x", "--mode", "a", "--campaign", "mar", "--run",
            "no_such_run"],
           ["agg", "--cells", "cells/no_such_cell/*.csv"]]


@pytest.mark.parametrize("argv", WRITING, ids=lambda a: a[0])
def test_writing_subcommands_refuse_a_missing_or_forbidden_write_root(argv, monkeypatch,
                                                                       tmp_path):
    monkeypatch.setenv("ETROC_TWC_TRANSFER_ROOT", str(tmp_path / "T"))
    monkeypatch.setitem(C.CAMPAIGNS, "mar", str(tmp_path / "data"))
    monkeypatch.delenv("ETROC_TWC_OUT", raising=False)
    with pytest.raises(SystemExit, match="ETROC_TWC_OUT is not set"):
        cli.main(argv)
    monkeypatch.setenv("ETROC_TWC_OUT", str(tmp_path / "T"))
    with pytest.raises(SystemExit, match="is the read root"):
        cli.main(argv)
    monkeypatch.setenv("ETROC_TWC_OUT", str(tmp_path / "data"))
    with pytest.raises(SystemExit, match="data root"):
        cli.main(argv)
    assert not os.listdir(tmp_path)


def test_jobs_dry_run_expands_and_parses_and_writes_nothing(monkeypatch, tmp_path, capsys):
    toy = tmp_path / "toy.tsv"
    toy.write_text(
        "a\ta\ta\tdesy\t1\t2000\tcell --cell a --mode a --run r --root ${ETROC_TWC_ROOT_DESY}"
        " --out ${TT_OUT}/cells/a/a.csv\n"
        "t\tt\tf\tnone\t1\t2000\ttier0 --bank ${TT_OUT}/bank/coeffs_desy_r.parquet --tag r\n")
    monkeypatch.setenv("ETROC_TWC_TRANSFER_ROOT", str(tmp_path / "T"))
    monkeypatch.setenv("ETROC_TWC_OUT", str(tmp_path / "out"))
    monkeypatch.setitem(C.CAMPAIGNS, "desy", "/data/desy")
    assert cli.main(["jobs", "--list", str(toy), "--dry-run"]) == 0
    text = capsys.readouterr().out
    assert "--root /data/desy --out %s/cells/a/a.csv" % (tmp_path / "out") in text
    assert "--bank %s/bank/coeffs_desy_r.parquet" % (tmp_path / "T") in text
    assert "toy.tsv: 2 jobs (0 smoke), 2 parsed, 0 failed" in text
    assert sorted(os.listdir(tmp_path)) == ["toy.tsv"]
    with pytest.raises(SystemExit, match="only --dry-run"):
        cli.main(["jobs", "--list", str(toy)])
    toy.write_text(toy.read_text().replace("--cell a ", "--cell b "))
    with pytest.raises(SystemExit, match="1 of 2 commands failed"):
        cli.main(["jobs", "--list", str(toy), "--dry-run"])
    assert "differs from the row's cell_id a" in capsys.readouterr().out


# the defaults of the study's own parser (tt.py), per subcommand
CELL_DEFAULTS = dict(
    cmd="cell", campaign="mar", run=None, combo=None, root=None, cap=6000, neighbor="none",
    neighbor_logic="OR", out=None, src_campaign=None, src_run=None, src_combo=None,
    src_triple=None, src_root=None, src_bank=None, role_map=None, src_parity_contrast=False,
    split="block", xfer_board=None, src_role=None, src_pick="auto", src_cap=0, arms="core",
    xfer_lin_arm=False, lpo=False, lpo_fallback="sibling", ntrip="12", triples=None, nmin=600,
    nmin_src=1000, nmin_fold=250, src_k=5, pool_ntrip=40, pool_cap=1500,
    pool_lopo_scope="board", pool_lpo_ntrip=6, chunk=None, flush=2, fast=False,
    overwrite=False, dry_run=False, max_triples=0, timebudget=0.0)
TIER0_DEFAULTS = dict(cmd="tier0", tag="all", max_src=0, out=None)
INDEX_DEFAULTS = dict(cmd="index", campaign="mar", run=None, combo=None, root=None, cap=6000,
                      neighbor="none", neighbor_logic="OR", out=None, ntrip=100000)
BANK_DEFAULTS = dict(INDEX_DEFAULTS, cmd="bank", ntrip=40, nmin=600, folds="parity",
                     guards="full", placebo=False, angle="nan", telescope="", hv="nan",
                     temp="nan", chunk=None, append=False)
DEFAULTS = {"cell": CELL_DEFAULTS, "tier0": TIER0_DEFAULTS, "index": INDEX_DEFAULTS,
            "bank": BANK_DEFAULTS, "fit": BANK_DEFAULTS}
SWITCHES = {"--lpo", "--overwrite", "--src-parity-contrast", "--xfer-lin", "--fast", "--dry-run",
            "--placebo", "--append"}
INTS = {"--cap", "--nmin", "--nmin-src", "--nmin-fold", "--src-k", "--src-cap", "--pool-ntrip",
        "--pool-cap", "--pool-lpo-ntrip", "--flush", "--max-triples", "--max-src"}
DESTS = {"--xfer-lin": "xfer_lin_arm"}

# one command of each of the seven lists (two of cells.tsv), as jobs/ holds them
SAMPLE = [
    "tier0 --bank ${TT_OUT}/bank/coeffs_desy_run13.parquet --tag may_cross_run13 --max-src 20",
    "cell --cell h1_mar19_to_jul3 --mode a --campaign jul --run h1_run3 --combo dut1-ref2-trig3 "
    "--root ${ETROC_TWC_ROOT_JUL} --src-campaign mar --src-run h1_run19 --src-combo "
    "ref1-dut2-trig3 --src-root ${ETROC_TWC_ROOT_MAR} --role-map dut:ref,ref:dut,trig:trig "
    "--ntrip 15 --cap 6000 --nmin 600 --nmin-src 1000 --nmin-fold 250 --arms core --lpo "
    "--overwrite --out ${TT_OUT}/cells/h1_mar19_to_jul3/h1_mar19_to_jul3.csv",
    "cell --cell I1_h1run7_from_f1run2 --mode d --campaign mar --run h1_run7 --combo "
    "ref1-dut2-trig3 --root ${ETROC_TWC_ROOT_MAR} --src-run f1_run2 --src-combo extra0-dut1-ref2 "
    "--role-map dut:dut,ref:extra,trig:ref --src-k 5 --ntrip 100 --cap 6000 --nmin 600 "
    "--nmin-src 1000 --nmin-fold 250 --lpo --overwrite --arms ext --chunk 1/2 --out "
    "${TT_OUT}/cells/I1_h1run7_from_f1run2/I1_h1run7_from_f1run2_c1of2.csv",
    "cell --cell B1_same_26to3 --mode a --campaign desy --run run3 --combo trig1-ref2-dut3 "
    "--root ${ETROC_TWC_ROOT_DESY} --src-run run26 --src-combo ref1-trig2-dut3 --role-map "
    "trig:ref,ref:trig,dut:dut --ntrip 100 --cap 6000 --nmin 600 --nmin-src 1000 --nmin-fold "
    "250 --lpo --overwrite --arms core --out ${TT_OUT}/cells/B1_same_26to3/B1_same_26to3.csv",
    "cell --cell A8_P3_run26_pc --mode c --xfer-board dut --src-pick pixpop --src-k 1 "
    "--campaign desy --run run26 --combo ref1-trig2-dut3 --root ${ETROC_TWC_ROOT_DESY} --ntrip "
    "100 --cap 6000 --nmin 600 --nmin-src 1000 --nmin-fold 250 --lpo --overwrite --arms core "
    "--src-parity-contrast --chunk 1/3 --out "
    "${TT_OUT}/cells/A8_P3_run26_pc/A8_P3_run26_pc_c1of3.csv",
    "cell --cell F1_lin_c1_all3_run26 --mode c --campaign desy --run run26 --combo "
    "ref1-trig2-dut3 --root ${ETROC_TWC_ROOT_DESY} --src-k 5 --src-pick trkpop "
    "--src-parity-contrast --ntrip 15 --cap 6000 --nmin 600 --nmin-src 1000 --nmin-fold 250 "
    "--lpo --overwrite --arms ext --xfer-lin --out "
    "${TT_OUT}/cells/F1_lin_c1_all3_run26/F1_lin_c1_all3_run26.csv",
    "cell --cell A4_P4_run26_0deg --mode cb --xfer-board dut --src-role ref --src-pick pixpop "
    "--src-k 1 --campaign desy --run run26 --combo ref1-trig2-dut3 --root "
    "${ETROC_TWC_ROOT_DESY} --ntrip all --cap 6000 --nmin 600 --nmin-src 1000 --nmin-fold 250 "
    "--lpo --overwrite --arms core --chunk 1/11 --out "
    "${TT_OUT}/cells/A4_P4_run26_0deg/A4_P4_run26_0deg_c1of11.csv",
    "cell --cell E2_h1_off10to20_early --mode a --campaign jul --run h1_run19 --combo "
    "dut1-ref2-trig3 --root ${ETROC_TWC_ROOT_JUL} --src-run h1_run33 --ntrip 100 --cap 6000 "
    "--nmin 600 --nmin-src 1000 --nmin-fold 250 --lpo --overwrite --arms ext --chunk 1/2 --out "
    "${TT_OUT}/cells_guardfix/E2_h1_off10to20_early/E2_h1_off10to20_early_c1of2.csv "
    "--lpo-fallback own",
]


def _expected(tokens):
    want = dict(DEFAULTS[tokens[0]], cmd=tokens[0])
    i = 1
    while i < len(tokens):
        flag = tokens[i]
        dest = DESTS.get(flag, flag[2:].replace("-", "_"))
        if flag in SWITCHES:
            want[dest], i = True, i + 1
        else:
            v = tokens[i + 1]
            is_int = flag in INTS or (flag == "--ntrip" and tokens[0] != "cell")
            want[dest], i = (int(v) if is_int else v), i + 2
    return want


def _typed(ns):
    """{dest: (type, value)}, a float NaN written as the string "nan" (NaN != NaN)."""
    return {k: (type(v), "nan" if v != v else v) for k, v in ns.items()}


def test_the_parser_reads_the_study_commands_with_the_study_defaults():
    roots = {"mar": "/d/mar", "jul": "/d/jul", "desy": "/d/desy", "aug": "/d/aug"}
    bare = ["cell --cell x --mode a --run r", "tier0 --bank b.parquet", "index --run r",
            "bank --run r", "fit --campaign mar --run f1_run12 --ntrip 40 --folds both --placebo"]
    for cmd in SAMPLE + bare:
        tokens = shlex.split(J.expand(cmd, roots, "/read", "/write"))
        got = {k: v for k, v in vars(cli.parse_command(tokens)).items() if k != "func"}
        want = {k: (float("nan") if v == "nan" else v) for k, v in _expected(tokens).items()}
        assert _typed(got) == _typed(want), tokens[:3]
        if J.OUT_TOKEN in cmd:
            assert (got["out"] or "/write/").startswith("/write/")
            assert (got.get("bank") or "/read/").startswith("/read/")
    with pytest.raises(ValueError, match="unrecognized arguments"):
        cli.parse_command(["tier0", "--bank", "b.parquet", "--no-such-flag"])


def test_the_hashseed_restart_happens_only_when_the_seed_is_not_zero():
    calls = []
    fake = lambda exe, args: calls.append((exe, args))   # noqa: E731
    env = {}
    assert M.ensure_hashseed(env, ["__main__.py", "jobs", "--list", "x"], execv=fake,
                             executable="/py") is True
    assert env["PYTHONHASHSEED"] == "0"
    assert calls == [("/py", ["/py", "-m", "twc_transfer", "jobs", "--list", "x"])]
    assert M.ensure_hashseed({"PYTHONHASHSEED": "1"}, ["m"], execv=fake, executable="/py")
    assert M.ensure_hashseed({"PYTHONHASHSEED": "0"}, ["m"], execv=fake) is False
    assert len(calls) == 2
    env = {}
    M.pin_threads(env)
    assert env == {k: "1" for k in M.THREAD_VARS} and "OMP_NUM_THREADS" in env


def test_python_m_twc_transfer_restarts_once_with_hashseed_zero():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}

    def run(e):
        return subprocess.run([sys.executable, "-m", "twc_transfer", "jobs", "--help"],
                              cwd=TESTBEAM, env=e, capture_output=True, text=True, timeout=120)

    r = run(env)
    assert r.returncode == 0 and "--dry-run" in r.stdout, r.stderr
    assert r.stderr.count("restarting with PYTHONHASHSEED=0") == 1
    r = run(dict(env, PYTHONHASHSEED="0"))
    assert r.returncode == 0 and "restarting" not in r.stderr
