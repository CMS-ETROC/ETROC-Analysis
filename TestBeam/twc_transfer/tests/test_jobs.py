"""Tests of the job lists: the reader, the root expansion, and the seven lists of jobs/.

Run from TestBeam/:  python -m pytest twc_transfer/tests
"""
import hashlib
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "..")))
from twc_transfer import campaigns as C   # noqa: E402
from twc_transfer import jobs as J        # noqa: E402

HEADER = "\t".join(J.FIELDS)
TOY_ROWS = [
    "SMOKE\tsmoke_x\tsmoke\tnone\t4\t2000\tcell --cell smoke_x --mode a --run r1",
    "x__c1of2\tx\ta\tdesy\t30\t2000\tcell --cell x --mode a --run r1 --chunk 1/2",
    "x__c2of2\tx\ta\tdesy\t30\t2000\tcell --cell x --mode a --run r1 --chunk 2/2",
    "t0\tt0\tf\tnone\t5\t2000\ttier0 --bank ${TT_OUT}/bank/coeffs_desy_run9.parquet --tag t",
]

# sha256 of the job rows of the study's own lists (the seven fields joined by tabs, the rows by
# newlines; comment and header lines left out), measured on the study's files
STUDY_SHA256 = {
    "cells.tsv": "c39232664da9250591afe7f486a81004f14085e16e09f63aeec85df3bc5b9724",
    "cells_stage1.tsv": "27e53312e3401bac193cce90d4bf5e0ba7a9c7b497f9595d0d29893d426abb6e",
    "cells_stage2.tsv": "58181279ddf0ef78398a0543646b2dbdf02167dd932efccd1ee83e522041156c",
    "cells_stage2_held.tsv": "cf5e5b3d4024ba31acd8926f26e47723da2cbbcf060b48e2ed7481efb4b27dc4",
    "cells_stage3.tsv": "7a5e1da6925def4895ffa81801046e88a5bfb4749769eb30d3edc84affae257e",
    "cells_rerun.tsv": "ed087d0d815cc1ac9d9a1277d8e86d8ab06327850edb380d2ca126615be09daa",
    "cells_guardfix_rerun.tsv": "5a76c7465c249b0a1cd950d8ad5f97e2c7c28fbabb4b48d01debe4647541a61e",
}


def _write(path, lines):
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_reader_skips_header_and_comments_and_keeps_chunks(tmp_path):
    with_header = _write(tmp_path / "a.tsv", ["# a comment", HEADER, "", *TOY_ROWS, "# end"])
    without = _write(tmp_path / "b.tsv", ["#", *TOY_ROWS])
    for path in (with_header, without):
        jobs = J.read_jobs(path)
        assert [j["job_id"] for j in jobs] == ["SMOKE", "x__c1of2", "x__c2of2", "t0"]
        assert jobs[0] == dict(job_id="SMOKE", cell_id="smoke_x", level="smoke", group="none",
                               est_min=4, mem_mb=2000,
                               command="cell --cell smoke_x --mode a --run r1")
        cells = J.by_cell(jobs)
        assert list(cells) == ["smoke_x", "x", "t0"]
        assert [j["job_id"] for j in cells["x"]] == ["x__c1of2", "x__c2of2"]


def test_reader_refuses_a_short_row_and_a_repeated_job_id(tmp_path):
    with pytest.raises(ValueError, match="line 2: 6 tab-separated fields"):
        J.read_jobs(_write(tmp_path / "short.tsv", [TOY_ROWS[0], "a\tb\tc\td\t1\t2"]))
    with pytest.raises(ValueError, match="job_id x__c1of2 repeated"):
        J.read_jobs(_write(tmp_path / "rep.tsv", [TOY_ROWS[1], TOY_ROWS[1]]))


ROOTS = {"desy": "/data/desy", "mar": "/data/mar/"}


def test_expand_sends_inputs_to_the_read_root_and_outputs_to_the_write_root():
    cmd = ("cell --cell x --root ${ETROC_TWC_ROOT_DESY} --src-root ${ETROC_TWC_ROOT_MAR} "
           "--src-bank ${TT_OUT}/bank/coeffs_mar_r.parquet --out ${TT_OUT}/cells/x/x.csv")
    got = J.expand(cmd, ROOTS, "/read", "/write/")
    assert got == ("cell --cell x --root /data/desy --src-root /data/mar "
                   "--src-bank /read/bank/coeffs_mar_r.parquet --out /write/cells/x/x.csv")
    assert J.expand("tier0 --bank ${TT_OUT}/b.parquet --tag t", ROOTS, "/read", "/write") == \
        "tier0 --bank /read/b.parquet --tag t"
    assert J.expand("x --out=${TT_OUT}/o.csv", ROOTS, "/read", "/write") == "x --out=/write/o.csv"
    # a None root keeps its token; the spacing of the command is kept
    assert J.expand("a  --root ${ETROC_TWC_ROOT_MAR} --out ${TT_OUT}/o", ROOTS, None, None) == \
        "a  --root /data/mar --out ${TT_OUT}/o"


def test_expand_never_puts_an_output_in_the_read_root():
    cmd = "cell --cell x --out ${TT_OUT}/cells/x/x.csv"
    out = J.expand(cmd, ROOTS, "/read/T", "/work/out").split()[-1]
    assert out == "/work/out/cells/x/x.csv" and not out.startswith("/read/T")
    with pytest.raises(ValueError, match="is the read root"):
        J.expand(cmd, ROOTS, "/read/T", "/read/T/")
    with pytest.raises(ValueError, match="neither an input flag"):
        J.expand("cell --log ${TT_OUT}/x.log", ROOTS, "/read/T", "/work/out")
    with pytest.raises(ValueError, match="no campaign 'jul'"):
        J.expand("cell --root ${ETROC_TWC_ROOT_JUL}", ROOTS, "/read/T", "/work/out")
    with pytest.raises(ValueError, match="unknown placeholder"):
        J.expand("cell --root ${HOME}/x", ROOTS, "/read/T", "/work/out")


def _rows(name, roots):
    out = []
    for j in J.read_jobs(J.list_path(name)):
        out.append("\t".join([j["job_id"], j["cell_id"], j["level"], j["group"],
                              str(j["est_min"]), str(j["mem_mb"]),
                              J.expand(j["command"], roots, None, None)]))
    return out


def test_the_lists_expanded_with_the_default_roots_are_the_study_commands():
    assert [n for n, _ in C.JOB_LISTS] == list(STUDY_SHA256)
    assert sorted(os.listdir(J.JOBS_DIR)) == sorted(STUDY_SHA256)
    for name, digest in STUDY_SHA256.items():
        rows = _rows(name, C._DEFAULT_ROOTS)
        assert hashlib.sha256("\n".join(rows).encode()).hexdigest() == digest, name


def test_the_lists_hold_placeholders_not_paths():
    n = {"rows": 0, "smoke": 0, "cell": 0, "tier0": 0}
    for name, _ in C.JOB_LISTS:
        text = open(J.list_path(name)).read()
        assert not re.search(r"^#", text, re.M), name
        for j in J.read_jobs(J.list_path(name)):
            toks = j["command"].split()
            n["rows"] += 1
            n["smoke"] += j["level"] == "smoke"
            n[toks[0]] += 1
            assert not [t for t in toks if t.startswith("/")], j["job_id"]
            for t in re.findall(r"\$\{[^}]*\}", j["command"]):
                assert t == J.OUT_TOKEN or t[len("${ETROC_TWC_ROOT_"):-1].lower() in C.CAMPAIGNS
    assert n == {"rows": 247, "smoke": 6, "cell": 236, "tier0": 11}
