"""Tests of the round-1 chain (master.py) on toy frames: the closed list of round-1 tables, the
two precisions of f, and the all-triples headline of the master table, which leaves out the
self-transfer rows.

Run from TestBeam/:  python -m pytest twc_transfer/tests
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

TESTBEAM = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, TESTBEAM)
from twc_transfer import campaigns as C   # noqa: E402
from twc_transfer import master as M      # noqa: E402

PAIR = "dut-ref"


def test_the_chain_reads_only_the_closed_list_of_round1_tables(tmp_path):
    with pytest.raises(M.MasterError, match="not one of the round-1"):
        M.read_round1_table(str(tmp_path), "agg_desy_f0p9999")
    with pytest.raises(M.MasterError, match="not one of the round-1"):
        M.stacked_tables({"agg_V2_desy_v2w1_f0p8153": pd.DataFrame({"cell": ["x"]})})
    assert [M.round1_tag(t)[0] for t, _, _ in C.ROUND1_AGG_TAGS] == [
        "desy", "desy", "aug", "aug", "aug", "mar", "jul"]
    assert M.f_variants("aug") == ["agg_aug_fmeas", "agg_aug_f0p8153", "agg_aug_f1p0"]
    assert M.f_variants("mar") == ["agg_mar_f0p9524"]


def toy_cell(nt=4, offset=0.3):
    """One round-1 cell of a single source: self_cf and xfer on `nt` triples, one pair width and
    one board sigma each, the xfer delta the same on every triple (no between-triple spread),
    its two folds apart by a different amount on each triple."""
    rows = []
    for i in range(nt):
        tri = "track_d-R%dC1_r-R%dC1_t-R%dC1" % (i, i, i)
        for arm in ("self_cf", "xfer"):
            for kind, name in (("pair", PAIR), ("board", "dut")):
                v = 40.0 + i + (offset if arm == "xfer" else 0.0)
                half = 0.1 * (1 + i) if arm == "xfer" else 0.0
                rows.append(dict(cell="c1", mode="a", run="run2", combo="dut1-ref2-trig3",
                                 tgt_triple=tri, kind=kind, name=name, arm=arm, value=v,
                                 v_f0=v + half, v_f1=v - half, n=2000,
                                 neg_radicand=0, self_transfer=0, same_pixel=0,
                                 src_run="run1", src_combo="dut1-ref2-trig3",
                                 src_triple="track_d-R9C9_r-R9C9_t-R9C9" if arm == "xfer"
                                 else np.nan,
                                 extrap=0.0 if arm == "xfer" else np.nan))
    return pd.DataFrame(rows)


def test_round1_f_is_four_decimals_in_the_tables_and_full_in_the_recomputation(tmp_path):
    # the tables' f, as their folders name it
    for tag, group, f in C.ROUND1_AGG_TAGS:
        suffix = tag.rsplit("_", 1)[1]
        want = C.F_CAL["aug"][0] if suffix == "fmeas" else float(suffix[1:].replace("p", "."))
        assert f == want, tag
    # the recomputation's f: the full bootstrap value of the cell's group
    os.makedirs(tmp_path / "cells" / "c1")
    toy_cell().to_csv(tmp_path / "cells" / "c1" / "c1.csv", index=False)
    jobs = pd.DataFrame([dict(job_id="c1", cell="c1", level="a", group="aug")])
    empty = pd.DataFrame({c: pd.Series(dtype=t) for c, t in (
        ("cell", object), ("arm", object), ("kind", object), ("name", object),
        ("self_transfer", "int64"), ("extrap_family", object), ("n_pix", "int64"),
        ("mu", float), ("se", float), ("tau", float), ("margin", float), ("tost_raw", object))})
    out = M.recompute(jobs, str(tmp_path / "cells"), {g: empty for g in C.ROUND1_PRIMARY_TAG})
    ra = out["alltriples_by_cell"]
    se = float(ra[(ra.arm == "xfer") & (ra.kind == "pair")].se.iloc[0])
    d1, _, _ = M.recompute_deltas(toy_cell(), 1.0)
    p1 = d1[d1.kind == "pair"]
    se1 = M.dl(p1.delta, p1.se)["se"]
    assert se == pytest.approx(C.F_CAL["aug"][1] * se1, rel=1e-12)
    assert se != pytest.approx(C.F_CAL["aug"][0] * se1, rel=1e-9)


def alltriples_rows():
    """alltriples_by_cell rows of one cell: per kind, an ordinary xfer row and a self-transfer
    xfer row further from equivalence."""
    rows = []
    for kind, names in (("pair", ("dut-ref", "dut-trig")), ("board", ("dut", "ref"))):
        for st, name, mu in ((0, names[0], 0.2), (1, names[1], 5.0)):
            rows.append(dict(cell="c1", group="desy", arm="xfer", kind=kind, name=name,
                             self_transfer=st, n_pix=10 + st, mu=mu, se=0.1, tau=0.0,
                             margin=1.0, tost="PASS" if st == 0 else "FAIL", ci_lo=mu - 0.2,
                             ci_hi=mu + 0.2, sigma_base=40.0 + st, n_interp=7, n_extrap=3,
                             n_censored=0, n_below_nmin=0))
    return pd.DataFrame(rows)


def test_the_all_triples_headline_leaves_out_the_self_transfer_rows():
    alltr = alltriples_rows()
    pair = M.all_triples_headline(alltr, "pair").set_index("cell").loc["c1"]
    assert (pair.all_bind_name, pair.all_mu, pair.all_npix) == ("dut-ref", 0.2, 10)
    assert pair.all_verdict == "PASS" and (pair.n_interp, pair.n_extrap) == (7, 3)
    assert pair.all_ivw_mu == pytest.approx(0.2)
    board = M.all_triples_headline(alltr, "board").set_index("cell").loc["c1"]
    assert (board.allb_name, board.allb_mu, board.allb_tost) == ("dut", 0.2, "PASS")
