"""Tests of the condor driver: jobs --write-dag, the PREP input check and the AGG plan.

Run from TestBeam/:  python -m pytest twc_transfer/tests
"""
import os
import re
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "..")))
from twc_transfer import campaigns as C   # noqa: E402
from twc_transfer import cli              # noqa: E402
from twc_transfer import jobs as J        # noqa: E402
from twc_transfer.__main__ import THREAD_VARS  # noqa: E402

JUL = "--campaign jul --run r1 --combo c1 --root ${ETROC_TWC_ROOT_JUL} --src-run r2"
DESY = "--campaign desy --run r9 --combo c9 --root ${ETROC_TWC_ROOT_DESY}"
TOY = [
    "\t".join(J.FIELDS),
    "SMOKE\tsmoke_x\tsmoke\tnone\t4\t2000\tcell --cell smoke_x --mode a %s --ntrip 5"
    " --overwrite --out ${TT_OUT}/cells_t/smoke_x/smoke_x.csv" % JUL,
    "x__c1of2\tx\tE\tE_jul\t30\t2000\tcell --cell x --mode a %s --chunk 1/2 --overwrite"
    " --out ${TT_OUT}/cells_t/x/x_c1of2.csv" % JUL,
    "x__c2of2\tx\tE\tE_jul\t30\t2000\tcell --cell x --mode a %s --chunk 2/2 --overwrite"
    " --out ${TT_OUT}/cells_t/x/x_c2of2.csv" % JUL,
    "y\ty\tK\tK_desy_guardfix\t95\t3000\tcell --cell y --mode c %s"
    " --src-bank ${TT_OUT}/bank/coeffs_desy_r9.parquet --out ${TT_OUT}/cells_t/y/y.csv" % DESY,
    "z\tz\tK\tK_desy_guardfix\t20\t2000\tcell --cell z --mode c %s --src-k 5" % DESY,
    "t0\tt0\tf\tnone\t5\t2000\ttier0 --bank ${TT_OUT}/bank/coeffs_desy_r9.parquet --tag t",
]


@pytest.fixture
def toy(tmp_path):
    """(list path, campaign roots, read root, write root); the jul run folders exist, the desy
    run folder and the bank file do not."""
    roots = {c: str(tmp_path / "data" / c) for c in ("mar", "jul", "desy", "aug")}
    os.makedirs(os.path.join(roots["jul"], "tracks_r1", "c1", "time"))
    os.makedirs(os.path.join(roots["jul"], "tracks_r2"))
    path = tmp_path / "cells_toy.tsv"
    path.write_text("\n".join(TOY) + "\n")
    return str(path), roots, str(tmp_path / "read"), str(tmp_path / "out")


def _write(toy, dag_dir="dag", **kw):
    path, roots, read, write = toy
    return J.write_dag(path, dag_dir, roots, read, write, cli.parse_command, **kw)


def _dag_nodes(text):
    jobs = re.findall(r"^JOB +(\S+) ", text, re.M)
    vars_ = {m.group(1): dict(re.findall(r'(\w+)="([^"]*)"', m.group(2)))
             for m in re.finditer(r"^VARS (\S+) (.*)$", text, re.M)}
    return jobs, vars_


def test_write_dag_one_node_per_row(toy, capsys):
    written = _write(toy)
    text = open(written["transfer_toy.dag"]).read()
    jobs, vars_ = _dag_nodes(text)
    assert jobs == ["SMOKE", "PREP", "x__c1of2", "x__c2of2", "y", "z", "t0", "AGG"]
    assert list(vars_) == jobs[2:-1]
    for jid, v in vars_.items():
        assert list(v) == ["job_id", "cell_id", "mem", "flavour", "retry"]
        assert v["job_id"] == jid and v["retry"] == "$(RETRY)"
    assert vars_["x__c1of2"]["cell_id"] == vars_["x__c2of2"]["cell_id"] == "x"
    assert [vars_[j]["flavour"] for j in jobs[2:-1]] == ["longlunch", "longlunch", "workday",
                                                         "workday", "longlunch"]
    assert vars_["y"]["mem"] == "3000"
    assert "PARENT SMOKE CHILD PREP" in text
    assert "PARENT PREP CHILD x__c1of2 x__c2of2 y z t0" in text
    assert "PARENT x__c1of2 x__c2of2 y z t0 CHILD AGG" in text
    assert re.findall(r"^RETRY (\S+) 1$", text, re.M) == jobs
    out = capsys.readouterr().out
    assert "inputs: 4 named, 2 missing" in out and "nothing was submitted" in out
    assert sorted(os.listdir(os.path.dirname(written["transfer_toy.dag"]))) == [
        "cells_toy.tsv", "logs", "manifest_transfer_toy.txt", "sub", "transfer_toy.dag",
        "wrapper.sh"]
    assert sorted(os.listdir(os.path.join(toy[3], "dag", "sub"))) == [
        "transfer_toy_%s.sub" % n for n in ("agg", "cell", "prep", "smoke")]


def test_write_dag_without_prep_node_and_retry(toy):
    text = open(_write(toy, prep_node=False, retry=3)["transfer_toy.dag"]).read()
    jobs, _ = _dag_nodes(text)
    assert jobs == ["SMOKE", "x__c1of2", "x__c2of2", "y", "z", "t0", "AGG"]
    assert "PARENT SMOKE CHILD x__c1of2" in text and "PREP" not in text
    assert len(re.findall(r"^RETRY \S+ 3$", text, re.M)) == len(jobs)


def test_out_expands_to_the_write_root_never_the_read_root(toy):
    path, roots, read, write = toy
    written = _write(toy)
    rows = J.read_jobs(written["cells_toy.tsv"])
    assert [r["job_id"] for r in rows] == [r["job_id"] for r in J.read_jobs(path)]
    for r in rows:
        assert "${" not in r["command"]
        outs = re.findall(r"--out (\S+)", r["command"])
        for o in outs:
            assert o.startswith(write + "/") and not o.startswith(read)
    by = {r["job_id"]: r["command"] for r in rows}
    assert "--out %s/cells_t/x/x_c1of2.csv" % write in by["x__c1of2"]
    assert "--root %s " % roots["jul"] in by["x__c1of2"]
    assert "--src-bank %s/bank/coeffs_desy_r9.parquet" % read in by["y"]
    assert "--bank %s/bank/coeffs_desy_r9.parquet" % read in by["t0"]
    for name in ("transfer_toy.dag", os.path.join("sub", "transfer_toy_cell.sub")):
        assert read not in open(written[name]).read()


def test_wrapper_checks_seed_and_threads(toy):
    path, roots, read, write = toy
    text = open(_write(toy)["wrapper.sh"]).read()
    codes = [int(m.group(1)) for m in re.finditer(r"exit (9\d)\b", text)]
    assert codes == [90, 91, 92, 98, 93, 94, 95, 96, 97]
    run = text.index("python3 -m twc_transfer \"$@\"")
    assert text.index("export PYTHONHASHSEED=0\n") < run
    threads = re.search(r"^export ((?:\w+=1 ?)+)$", text, re.M).group(1).split()
    assert threads == ["%s=1" % v for v in THREAD_VARS]
    assert text.index(threads[0]) < run
    assert "export ETROC_TWC_OUT=%s\n" % write in text
    assert "export ETROC_TWC_TRANSFER_ROOT=%s\n" % read in text
    for c, r in roots.items():
        assert "export ETROC_TWC_ROOT_%s=%s\n" % (c.upper(), r) in text
    assert "TWC_EXPECT_SHA=%s\n" % J.package_sha() in text
    assert "sed 's/ --overwrite//g'" in text and "nice -n 19 python3 -m twc_transfer" in text


@pytest.mark.skipif(not shutil.which("sha256sum"), reason="needs sha256sum")
def test_wrapper_runs_its_checks(toy, monkeypatch, tmp_path):
    dag_dir = os.path.dirname(_write(toy)["wrapper.sh"])
    cwd = tmp_path / "worker"
    cwd.mkdir()

    def run(*args):
        return subprocess.run(["bash", os.path.join(dag_dir, "wrapper.sh")] + list(args),
                              cwd=str(cwd), capture_output=True, text=True)
    assert run().returncode == 90
    helped = run("--help")
    assert helped.returncode == 0 and "92 the package is not the version" in helped.stdout
    nope = run("nope", "0")
    assert nope.returncode == 94, nope.stdout     # 91 and 92 passed: the shell sha is package_sha
    monkeypatch.setattr(J, "package_sha", lambda pkg_dir=J.PKG_DIR: "0" * 16)
    stale = os.path.dirname(_write(toy, dag_dir="dag2")["wrapper.sh"])
    got = subprocess.run(["bash", os.path.join(stale, "wrapper.sh"), "nope", "0"],
                         cwd=str(cwd), capture_output=True, text=True)
    assert got.returncode == 92
    assert not os.path.exists(os.path.join(toy[3], "cells"))


@pytest.mark.skipif(not (shutil.which("sha256sum") and shutil.which("sha1sum")),
                    reason="needs sha256sum and sha1sum")
def test_the_wrapper_exits_98_when_the_core_changed_after_the_dag_was_written(toy, monkeypatch,
                                                                              tmp_path):
    core = tmp_path / "core"
    core.mkdir()
    for name in ("twc_solver.py", "bootstrap.py", "io_utils.py"):
        (core / name).write_text("# %s\n" % name)
    monkeypatch.setenv("ETROC_TWC_CORE", str(core))
    written = _write(toy)
    sha = " ".join("%s=%s" % kv for kv in sorted(J.core_files_sha1(str(core)).items()))
    manifest = open(written["manifest_transfer_toy.txt"]).read()
    assert "core           %s  %s" % (core, sha) in manifest
    cwd = tmp_path / "worker"
    cwd.mkdir()

    def run():
        return subprocess.run(["bash", written["wrapper.sh"], "nope", "0"], cwd=str(cwd),
                              capture_output=True, text=True)
    assert run().returncode == 94                    # the core is the one the DAG was written for
    (core / "twc_solver.py").write_text("# twc_solver.py, edited\n")
    got = run()
    assert got.returncode == 98, got.stdout
    assert "core changed since the DAG was written" in got.stdout


def test_write_dag_refusals(toy, tmp_path):
    path, roots, read, write = toy
    _write(toy)
    with pytest.raises(ValueError, match="exists"):
        _write(toy)
    with pytest.raises(ValueError, match="not inside the write root"):
        _write(toy, dag_dir=str(tmp_path / "elsewhere"))
    lines = open(path).read().splitlines()
    nosmoke = tmp_path / "cells_nosmoke.tsv"
    nosmoke.write_text("\n".join(lines[:1] + lines[2:]) + "\n")
    with pytest.raises(ValueError, match="0 level=smoke rows"):
        J.write_dag(str(nosmoke), "d3", roots, read, write, cli.parse_command)
    reserved = tmp_path / "cells_reserved.tsv"
    reserved.write_text("\n".join(lines + [lines[-1].replace("t0\tt0", "AGG\tt0", 1)]) + "\n")
    with pytest.raises(ValueError, match="AGG are the DAG's own node names"):
        J.write_dag(str(reserved), "d4", roots, read, write, cli.parse_command)
    nocamp = tmp_path / "cells_nocamp.tsv"
    nocamp.write_text("\n".join(lines).replace("\tE_jul\t", "\tE_x\t") + "\n")
    with pytest.raises(ValueError, match="'E_x' names no campaign"):
        J.write_dag(str(nocamp), "d5", roots, read, write, cli.parse_command)
    assert sorted(os.listdir(write)) == ["dag"]


def test_two_lists_never_share_a_dag_folder(toy, tmp_path):
    path, roots, read, write = toy
    first = _write(toy)
    other = tmp_path / "cells_other.tsv"
    other.write_text(open(path).read())
    with pytest.raises(ValueError, match="not empty"):
        J.write_dag(str(other), "dag", roots, read, write, cli.parse_command)
    dag = os.path.dirname(first["wrapper.sh"])
    assert sorted(os.listdir(dag)) == ["cells_toy.tsv", "logs", "manifest_transfer_toy.txt",
                                       "sub", "transfer_toy.dag", "wrapper.sh"]
    assert "for the job table cells_toy.tsv." in open(first["wrapper.sh"]).read()


def test_group_campaign():
    assert [J.group_campaign(g) for g in ("desy", "D_mar", "E_aug", "K_desy_guardfix",
                                          "E_jul_guardfix", "none", "guardfix")] == [
        "desy", "mar", "aug", "desy", "jul", None, None]


def test_agg_plan_and_run(toy, capsys):
    path, roots, read, write = toy
    jobs = J.expanded_jobs(path, roots, read, write, cli.parse_command)
    plan = J.agg_plan(jobs, write)
    assert [(p["tag"], p["f"]) for p in plan] == [
        ("agg_E_jul_f0p9259", C.F_CAL["jul"][0]),
        ("agg_K_desy_guardfix_f0p8153", C.F_CAL["desy"][0]),
        ("agg_K_desy_guardfix_f1p0", 1.0)]
    assert plan[0]["cells"] == {"x": [os.path.join(write, "cells_t", "x", "*.csv")]}
    assert plan[1]["cells"] == {"y": [os.path.join(write, "cells_t", "y", "*.csv")],
                                "z": [os.path.join(write, "cells", "z.csv")]}
    assert plan[2]["out"] == os.path.join(write, "tables", "agg_K_desy_guardfix_f1p0")
    os.makedirs(os.path.join(write, "cells_t", "x"))
    open(os.path.join(write, "cells_t", "x", "x_c1of2.csv"), "w").close()
    calls = []
    assert J.run_agg(plan, calls.append) == []
    assert calls == [["agg", "--cells", os.path.join(write, "cells_t", "x", "*.csv"),
                      "--f-inflate", "0.9259", "--out",
                      os.path.join(write, "tables", "agg_E_jul_f0p9259")]]
    out = capsys.readouterr().out
    assert "agg_K_desy_guardfix_f0p8153: no table yet for 2 cell(s): y, z" in out

    def fails(argv):
        raise SystemExit("agg: no cell CSVs matched")
    assert J.run_agg(plan[:1], fails) == ["agg_E_jul_f0p9259"]


def test_check_inputs(toy, capsys):
    path, roots, read, write = toy
    jobs = J.expanded_jobs(path, roots, read, write, cli.parse_command)
    n, missing = J.check_inputs(jobs, read)
    assert n == 4
    assert missing == sorted([os.path.join(read, "bank", "coeffs_desy_r9.parquet"),
                              os.path.join(roots["desy"], "tracks_r9", "c9", "time")])
    assert capsys.readouterr().out.count("MISSING\t") == 2


def test_default_dag_name_and_flavour():
    assert J.default_dag_name("cells.tsv") == "transfer_cells"
    assert J.default_dag_name("x/cells_guardfix_rerun.tsv") == "transfer_guardfix_rerun"
    assert [J.flavour_for(90), J.flavour_for(91), J.flavour_for(10, "--src-k 4"),
            J.flavour_for(10, "--src-k 5")] == ["longlunch", "workday", "longlunch", "workday"]


def test_cli_write_dag(toy, monkeypatch, capsys):
    path, roots, read, write = toy
    monkeypatch.setenv("ETROC_TWC_TRANSFER_ROOT", read)
    monkeypatch.setenv("ETROC_TWC_OUT", write)
    for c in ("jul", "desy"):
        monkeypatch.setitem(C.CAMPAIGNS, c, roots[c])
    assert cli.main(["jobs", "--list", path, "--write-dag", "dag", "--dag-name",
                     "transfer_cells"]) == 0
    assert "nothing was submitted" in capsys.readouterr().out
    assert sorted(os.listdir(os.path.join(write, "dag", "sub"))) == [
        "agg.sub", "cell.sub", "prep.sub", "smoke.sub"]
    assert os.path.isfile(os.path.join(write, "dag", "manifest.txt"))
    with pytest.raises(SystemExit, match="--job goes with --dry-run only"):
        cli.main(["jobs", "--list", path, "--write-dag", "dag2", "--job", "y"])
    with pytest.raises(SystemExit, match="2 input"):
        cli.main(["jobs", "--list", path, "--check-inputs"])
