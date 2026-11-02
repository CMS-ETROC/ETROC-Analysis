"""Tests of the campaign facts: the write root (never unset, never inside the read root, a data
root, the study's trees or the checkout, always absolute), f per campaign, the thin rule, the
levels, the axis descriptions and the version tables.

Run from TestBeam/:  python -m pytest twc_transfer/tests
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "..")))
from twc_transfer import campaigns as C   # noqa: E402


def test_write_root_refuses_unset_read_root_and_data_roots(monkeypatch, tmp_path):
    monkeypatch.delenv("ETROC_TWC_OUT", raising=False)
    monkeypatch.delenv("ETROC_TWC_TRANSFER_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="ETROC_TWC_OUT is not set"):
        C.write_root()
    monkeypatch.setenv("ETROC_TWC_OUT", "")
    with pytest.raises(RuntimeError, match="ETROC_TWC_OUT is not set"):
        C.write_root()
    monkeypatch.setenv("ETROC_TWC_OUT", C.transfer_root() + "/")    # the default read root
    with pytest.raises(RuntimeError, match="read root"):
        C.write_root()
    monkeypatch.setenv("ETROC_TWC_TRANSFER_ROOT", str(tmp_path / "T"))
    monkeypatch.setenv("ETROC_TWC_OUT", str(tmp_path / "T"))
    with pytest.raises(RuntimeError, match="read root"):
        C.write_root()
    for root in C.CAMPAIGNS.values():
        monkeypatch.setenv("ETROC_TWC_OUT", root)
        with pytest.raises(RuntimeError, match="data root"):
            C.write_root()
    monkeypatch.setenv("ETROC_TWC_OUT", str(tmp_path / "out"))
    assert C.write_root() == str(tmp_path / "out")


def test_write_root_refuses_a_folder_inside_a_protected_root_by_its_real_path(monkeypatch,
                                                                              tmp_path):
    home = tmp_path / "eos" / "home-m"
    (home / "T").mkdir(parents=True)
    alias = tmp_path / "user"                 # a second name of the same tree, as /eos/user/m
    alias.symlink_to(home)
    monkeypatch.setenv("ETROC_TWC_TRANSFER_ROOT", str(alias / "T"))
    monkeypatch.setitem(C.CAMPAIGNS, "mar", str(tmp_path / "data"))
    monkeypatch.setattr(C, "_RERUN_SEEDED_ROOT", str(tmp_path / "seeded"))
    checkout = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(C.__file__))))
    refused = ((home / "T", "is the read root"), (home / "T" / "sub", "inside the read root"),
               (alias / "T" / "sub", "inside the read root"),
               (tmp_path / "data" / "tracks_run1", "inside the step-10 data root"),
               (tmp_path / "seeded" / "cells", "inside an output tree of the study's job lists"),
               (os.path.join(checkout, "TestBeam", "out"), "inside this package's checkout"))
    for out, why in refused:
        monkeypatch.setenv("ETROC_TWC_OUT", str(out))
        with pytest.raises(RuntimeError, match=why):
            C.write_root()
    monkeypatch.setenv("ETROC_TWC_OUT", str(home / "T2"))          # a sibling of the read root
    assert C.write_root() == str(home / "T2")
    assert sorted(os.listdir(home)) == ["T"]


def test_a_relative_write_root_is_made_absolute(monkeypatch, tmp_path):
    monkeypatch.setenv("ETROC_TWC_TRANSFER_ROOT", str(tmp_path / "T"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ETROC_TWC_OUT", os.path.join("rel", "out"))
    assert C.write_root() == os.path.join(os.getcwd(), "rel", "out")


def test_f_cal_keys_gate_and_agg_f():
    assert set(C.F_CAL) == set(C.AGG_F) == {"desy", "aug", "mar", "jul"}
    lo, hi = C.F_GATE
    for camp, (four, full, why) in C.F_CAL.items():
        assert round(full, 4) == four and lo <= full <= hi
        assert "rms convention" in why and "gate PASS" in why
        assert C.AGG_F[camp][0][1] == four
    assert C.AGG_F["desy"] == (("f0p8153", 0.8153), ("f1p0", 1.0))
    assert C.AGG_F["aug"] == (("fmeas", 0.8979), ("f0p8153", 0.8153), ("f1p0", 1.0))
    # round 1's seven tables are the AGG_F tags of its four groups
    assert C.ROUND1_AGG_TAGS == tuple(("agg_%s_%s" % (g, suffix), g, f)
                                      for g in ("desy", "aug", "mar", "jul")
                                      for suffix, f in C.AGG_F[g])


# (median_n, n_ge_600, n_ge_1500) of the study's census, primary combo
CENSUS = {("aug", "run5"): (74, 352, 39), ("aug", "run8"): (90, 372, 34),
          ("aug", "run18"): (90, 435, 131), ("aug", "run23"): (116, 443, 124),
          ("jul", "h1_run3"): (524, 509, 98), ("desy", "run32"): (454, 608, 510),
          ("desy", "run26"): (78581, 1094, 1057), ("jul", "h1_run4"): (2252, 1006, 716)}


def test_thin_rule_on_the_census_and_a_toy_frame():
    for run in C.THIN_RUNS:
        assert C.thin_verdict(*CENSUS[run], ntrip=30) == "THIN"
    assert C.thin_verdict(*CENSUS[("desy", "run32")], ntrip=30) == "THIN"
    assert C.thin_verdict(*CENSUS[("desy", "run26")], ntrip=30) == "THICK"
    pd = pytest.importorskip("pandas")
    rows = []

    def add(camp, run, counts):
        rows.extend(dict(campaign=camp, run=run, combo="c", nevt=n) for n in counts)
    add("x", "edge", [600] * 40 + [1500] * 30 + [600] * 9)  # median 600, 30 at 1500: THICK at 10
    add("x", "median", [599] * 60 + [1500] * 50)             # median 599: THIN
    add("x", "few1500", [1000] * 80 + [1500] * 29)           # 29 < 3 x 10: THIN
    add("x", "empty", [599] * 100 + [700] * 39)              # 39 triples of 600 or more: EXCLUDED
    t = C.thin_table(pd.DataFrame(rows), ntrip=10).set_index("run")
    assert t.loc["edge", "median_n"] == 600 and t.loc["edge", "n_ge_1500"] == 30
    assert t["verdict"].to_dict() == {"edge": "THICK", "empty": "EXCLUDED", "few1500": "THIN",
                                      "median": "THIN"}
    assert C.THIN_RULE == {"median_min": 600, "n1500_per_ntrip": 3, "exclude_below": 40}


def test_level_names():
    assert set(C.LEVEL_TITLES) == set(C.ROUND1_PREREG) == {"a", "b", "bp", "c", "d", "e", "f",
                                                          "g", "h"}
    for lev, title in C.LEVEL_TITLES.items():
        assert title.startswith("(%s) " % ("b'" if lev == "bp" else lev))
    from twc_transfer import jobs as J
    levels = {j["level"] for j in J.read_jobs(J.list_path("cells.tsv"))} - {"smoke"}
    assert levels == set(C.LEVEL_TITLES)


def test_cell_desc_complete():
    assert set(C.CELL_DESC) == set(C.AXIS_TAGS) | set(C.OTHER_AXIS_TAGS)
    assert C.AXIS_F_CELLS is C.CELL_DESC["F"]
    assert [lab for lab, _, _ in C.AXIS_F_CELLS] == ["F%d" % i for i in range(1, 12)]
    for rows in C.CELL_DESC.values():
        assert rows and all(len(r) == 3 and all(isinstance(x, str) for x in r) for r in rows)


def test_versions_shas_per_wave_and_tables():
    names = [w[0] for w in C.WAVES]
    assert len(names) == len(set(names))                     # one sha per wave
    for w in C.WAVES:
        assert re.fullmatch(r"[0-9a-f]{8,}", w[3]), w
        assert set(w[8]) <= set(C.PUBLISHED_MD5)
    assert C.CELL_VERSIONS == tuple((w[0], w[2], w[3], w[4]) for w in C.WAVES)
    lists = {jl for jl, _ in C.JOB_LISTS} - {"cells_stage2_held.tsv"}
    assert lists <= {w[2] for w in C.WAVES}
    tables = [t[0] for t in C.TABLE_VERSIONS]
    assert len(tables) == len(set(tables))
    assert set(tables) == set(C.STUDY_TABLE_MD5) | set(C.PUBLISHED_MD5)
    for table, cmd, waves, made_from, _ in C.TABLE_VERSIONS:
        assert re.fullmatch(r"[0-9a-f]{32}", C.table_md5(table))
        assert cmd.startswith("python -m ") and made_from
        assert waves and set(waves) <= set(names)
    for pub in C.PUBLISHED_MD5:
        assert any(pub in w[8] for w in C.WAVES), pub


def test_readme_carries_the_version_tables():
    readme = os.path.join(os.path.dirname(C.__file__), "README.md")
    assert C.versions_markdown() in open(readme).read()
