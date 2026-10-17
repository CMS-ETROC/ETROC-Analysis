"""Tests of the frontend builders: the TOT code inversion and its bins, the RFSel group rows, the
runs of the fluence ladder, the run set of the rebuild, the hits stated in the code boxes, the
colour range of the CAL difference maps and the BL/NW tables built from synthetic scans.

Run from TestBeam/:  python -m pytest etroc_plots/tests
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "..")))
if not os.environ.get("ETROC_CAMPAIGN"):
    os.environ["ETROC_CAMPAIGN"] = "irrad_2026"

from etroc_plots.campaigns import active as campaign            # noqa: E402
from etroc_plots.frontend import blnw, rebuild, tdc_codes, tot_shapes   # noqa: E402


# ------------------------------------------------------------------------------- TOT codes
def step10_tot_ps(codes, cal, file_idx):
    """TOT in ps as the pipeline's step 10 writes it: per loop file one bin, 3.125 ns / the mean
    CAL of that file's events, and TOT = (2 code - floor(code / 32)) x bin."""
    out = np.empty(len(codes))
    for f in np.unique(file_idx):
        sel = file_idx == f
        bin_ns = 3.125 / np.mean(cal[sel])
        out[sel] = (2 * codes[sel] - np.floor(codes[sel] / 32.0)) * bin_ns * 1e3
    return out


def synthetic_events():
    """Three loop files: 3 with codes 4-140 across the multiples of 32 and a CAL whose mean is
    not its median, 7 with a single event, 9 with two events of the same TOT."""
    codes3 = np.concatenate([np.arange(4, 141, 7), [31, 32, 33, 63, 64, 95, 96, 127, 128]])
    cal3 = np.resize(np.array([140, 146, 146, 152, 146, 151, 152, 149]), len(codes3))
    codes = np.concatenate([codes3, [33], [64, 64]])
    cal = np.concatenate([cal3, [147], [143, 151]]).astype(float)
    files = np.array([3] * len(codes3) + [7] + [9, 9])
    return codes, cal, files


def test_tot_codes_invert_step_10_per_loop_file():
    codes, cal, files = synthetic_events()
    assert np.mean(cal[files == 3]) != np.median(cal[files == 3])
    got = tot_shapes.tot_codes(step10_tot_ps(codes, cal, files), files, cal)
    np.testing.assert_array_equal(got, codes)


def test_tot_off_the_code_ladder_raises_naming_the_loop_file():
    codes, cal, files = synthetic_events()
    tot = step10_tot_ps(codes, cal, files)
    bin_ps = 3.125 / 147.0 * 1e3
    for m in (61.0, 40.5):          # 61: an integer no code reaches; 40.5: between two codes
        bad = tot.copy()
        bad[files == 7] = m * bin_ps
        with pytest.raises(ValueError, match="loop file 7"):
            tot_shapes.tot_codes(bad, files, cal)


def test_code_edges_are_ladder_values_with_half_bins_at_multiples_of_32():
    cal_mode = 146
    edges = tot_shapes.code_edges_ns(20, 70, cal_mode)
    assert len(edges) == 70 - 20 + 2
    codes = np.arange(20, 71)
    np.testing.assert_allclose(edges[1:], (2 * codes - codes // 32) * 3.125 / cal_mode)
    widths = np.diff(edges)
    step = 2 * 3.125 / cal_mode
    half = codes % 32 == 0
    assert list(codes[half]) == [32, 64]
    np.testing.assert_allclose(widths[half], 0.5 * step)
    np.testing.assert_allclose(widths[~half], step)
    np.testing.assert_allclose(tot_shapes.code_edges_ns(32, 33, cal_mode), edges[12:15])
    np.testing.assert_allclose(np.diff(tot_shapes.code_edges_ns(0, 2, cal_mode)),
                               [0.5 * step, step, step])


# ------------------------------------------------------------------------------- RFSel rows
# board-1 pixels and the events of their triples per run: above the floor (300) in all three
# runs only P1 and P2; in the pair 52-86 also P3
PIXELS = {"P1": (1, 1), "P2": (1, 2), "P3": (2, 1), "P4": (2, 2), "P5": (3, 3)}
EVENTS = {32: dict(P1=400, P2=500, P3=100, P4=350, P5=320),
          52: dict(P1=400, P2=500, P3=400, P4=200, P5=330),
          86: dict(P1=300, P2=310, P3=400, P4=400, P5=50)}
RFSEL = {32: 2, 52: 2, 86: 3}
TOT_EVENTS = 7


def synthetic_triples():
    rows = []
    for run, by_pixel in EVENTS.items():
        for name, n in by_pixel.items():
            (r, c) = PIXELS[name]
            rec = dict(campaign="july", tel="f1", run=run, boards="0-1-2",
                       combo="extra0-dut1-ref2", n_rows=n, n_bytes=10 * n,
                       file="track_%s_run%d.parquet" % (name, run))
            for b in range(4):
                inside = b < 3
                rec["chip_%d" % b] = campaign.TELESCOPE_CHIPS["f1"][b] if inside else None
                rec["row_%d" % b] = (r if b == 1 else 5) if inside else None
                rec["col_%d" % b] = (c if b == 1 else 5) if inside else None
            rows.append(rec)
    return tot_shapes.as_triples(pd.DataFrame(rows, columns=tot_shapes.TRIPLES_COLUMNS))


def test_rfsel_group_rows_count_the_pixels_above_the_floor_in_every_run(monkeypatch):
    monkeypatch.setattr(tot_shapes, "run_roles", lambda camp, tel, run: ["extra", "dut", "ref",
                                                                          "trig"])
    monkeypatch.setattr(tot_shapes, "read_tot", lambda path, role: np.full(TOT_EVENTS, 2500.0))
    monkeypatch.setattr(tot_shapes.tables, "run_settings", lambda rs, camp, tel, run: [
        dict(board_idx=b, rfsel=RFSEL[run], offset=10) for b in range(4)])
    rows = pd.DataFrame(tot_shapes.rfsel_rows(synthetic_triples(), None, "july", "f1",
                                              [(52, 86), (32, 52, 86)], 1))
    assert set(rows["group"]) == {"52+86", "32+52+86"}
    assert (rows["board_idx"] == 1).all()
    group = rows[rows["group"] == "32+52+86"]
    assert sorted(group["run"]) == [32, 52, 86]
    assert (group["n_pixels"] == 2).all()
    assert (group["n_events"] == 2 * TOT_EVENTS).all()
    assert group["partner_run"].isna().all()
    pair = rows[rows["group"] == "52+86"].set_index("run")
    assert (pair["n_pixels"] == 3).all()
    assert pair.loc[52, "partner_run"] == 86 and pair.loc[86, "partner_run"] == 52
    assert list(group.set_index("run").loc[[32, 52, 86], "rfsel"]) == [2, 2, 3]


def test_group_hists_reads_one_group_as_fractions_of_its_events():
    rows = []
    for run, grp, partner, n, med, bins in (
            (52, "52+86", 86, 10, 2.50, {2.40: 4, 2.45: 6}),
            (86, "52+86", 52, 8, 3.00, {2.95: 8}),
            (32, "32+52+86", None, 6, 2.20, {2.10: 1, 2.15: 2, 2.20: 3}),
            (52, "32+52+86", None, 5, 2.45, {2.40: 2, 2.45: 3}),
            (86, "32+52+86", None, 4, 3.05, {3.00: 4})):
        rows += [dict(campaign="july", tel="f1", run=run, board_idx=1, chip="LF12",
                      rfsel=RFSEL[run], offset=10, partner_run=partner, n_pixels=2, n_events=n,
                      median_ns=med, bin_lo_ns=lo, count=c, group=grp) for lo, c in bins.items()]
    out = tot_shapes.group_hists(pd.DataFrame(rows), "july", "f1", "32+52+86", 0.1)
    assert list(out) == [32, 52, 86]
    for run, n, med in ((32, 6, 2.20), (52, 5, 2.45), (86, 4, 3.05)):
        assert out[run]["n_events"] == n
        assert out[run]["median_ns"] == med
        assert out[run]["y"].sum() == pytest.approx(1.0)
        np.testing.assert_allclose(np.diff(out[run]["edges"]), 0.1)
    assert (out[32]["rfsel"], out[86]["rfsel"]) == (2, 3)


# ------------------------------------------------------------------------------- run sets
def test_ladder_extras_follow_the_display_runs_of_their_step():
    display = {1.5e15: [4], 3.5e15: [17, 18]}
    assert blnw.ladder_runs(display, {3.5e15: [19]}) == {1.5e15: [4], 3.5e15: [17, 18, 19]}
    assert blnw.ladder_runs(display, {}) == display
    with pytest.raises(ValueError, match="run 19"):
        blnw.ladder_runs(display, {2e15: [19]})


def test_frontend_run_set_holds_the_extras_the_time_control_and_the_display_runs(monkeypatch):
    stub = {(2, 20): {"h1": {1.5e15: [4], 3.5e15: [18]}, "f1": {1.5e15: [9], 3.5e15: [18, 60]}},
            (2, 10): {"f1": {3.5e15: [77]}}}
    monkeypatch.setattr(rebuild.tables, "display_runs",
                        lambda path, rfsel, offset: stub.get((rfsel, offset), {}))
    runs = rebuild.frontend_runs()
    assert 19 in runs[("july", "h1")]
    assert 32 in runs[("july", "f1")]
    for tel, steps in stub[(2, 20)].items():
        assert set(r for rs in steps.values() for r in rs) <= set(runs[("july", tel)])
    assert 77 not in runs[("july", "f1")]


# ------------------------------------------------------------------------------- code boxes
def native_hits():
    """0.1 ns bins (lower edges, hits) of one look: 23 hits inside 0-12.3 ns, the last in the
    bin that starts at 12.2 ns, and 49 outside (below 0, from 12.3 ns, far above)."""
    lo = np.array([-0.1, 0.0, 6.1, 12.2, 12.3, 20.0])
    counts = np.array([19, 5, 7, 11, 13, 17])
    return lo, counts


def test_hits_in_range_counts_the_drawn_bins_only():
    lo, counts = native_hits()
    assert tdc_codes.hits_in_range(lo, counts, 0.3, 0.0, 12.3) == 23
    edges, inside, outside = tdc_codes.rebin(lo, counts, 0.3, 0.0, 12.3, 0.1)
    assert inside.sum() == 23 and outside == 49
    assert tdc_codes.hits_in_range(lo, counts, 0.3, 0.0, 12.0) == 12


def test_box_lines_state_each_looks_hits_in_range():
    lo, counts = native_hits()
    recs = [dict(value=3.204, hits_in_range=tdc_codes.hits_in_range(lo, counts, 0.3, 0.0, 12.3)),
            dict(value=2.9, hits_in_range=tdc_codes.hits_in_range(lo, 2 * counts, 0.3, 0.0, 12.3))]
    text = tdc_codes.box_lines("median", [dict(label="March"), dict(label="July")], recs, "%.2f ns")
    assert text.split("\n") == ["median", "March: 3.20 ns, N = 23", "July: 2.90 ns, N = 46"]


# ------------------------------------------------------------------------------- CAL difference range
def synthetic_diffs():
    """July - March per chip, two chips per telescope. H1: 90 pixels at 0, 9 at +1, 1 at -2 codes;
    F1: 45 at -2, 50 at -10, 5 at -12.5; the second chip of each holds only pixels without a CAL
    mode in one of the runs (NaN), which no range may see."""
    h1 = np.array([0.0] * 90 + [1.0] * 9 + [-2.0]).reshape(10, 10)
    f1 = np.array([-2.0] * 45 + [-10.0] * 50 + [-12.5] * 5).reshape(10, 10)
    return {"h1": [h1, np.full((2, 2), np.nan)], "f1": [f1, np.array([np.nan])]}


def test_difference_range_is_one_pooled_99th_percentile_rounded_up_on_both_telescopes():
    # pooled |diff| (200 pixels): the 99th percentile sits among the five 12.5 values -> 13
    assert tdc_codes.diff_ranges(synthetic_diffs()) == {"h1": 13.0, "f1": 13.0}
    text = tdc_codes.diff_range_text()
    assert "shared by both telescopes" in text and "99th percentile" in text
    assert "rounded up" in text


def test_difference_range_rule_switches_pooling_rounding_and_keeps_a_floor():
    per_tel = dict(tdc_codes.DIFF_RANGE, pool="per_telescope")
    # H1 alone: the 99th percentile of 100 values lies between 1 and 2 (1.01) -> 2
    assert tdc_codes.diff_ranges(synthetic_diffs(), per_tel) == {"h1": 2.0, "f1": 13.0}
    assert "per telescope" in tdc_codes.diff_range_text(per_tel)
    unrounded = dict(tdc_codes.DIFF_RANGE, rounding="none")
    assert tdc_codes.diff_ranges(synthetic_diffs(), unrounded) == {"h1": 12.5, "f1": 12.5}
    flat = {"h1": [np.zeros((4, 4))], "f1": [np.zeros((4, 4))]}
    assert tdc_codes.diff_ranges(flat) == {"h1": 1.0, "f1": 1.0}
    with pytest.raises(ValueError, match="pool"):
        tdc_codes.diff_ranges(synthetic_diffs(), dict(tdc_codes.DIFF_RANGE, pool="both"))


# ------------------------------------------------------------------------------- BL/NW tables
def synthetic_scan():
    """One board's scan (16 x 16, flat index i = 16 row + col) and the run's railed pixels.
    BL: 101 live pixels at 400..500 DAC, 100 dead (0), 50 at the rail in the scan (1020), 5 at 999
    DAC but railed in the run. NW: 201 pixels at 0.5, 1.0, ..., 100.5 DAC, 55 at 0."""
    i = np.arange(256)
    bl = np.where(i < 101, 400.0 + i, np.where(i < 201, 0.0, np.where(i < 251, 1020.0, 999.0)))
    railed = i >= 251
    nw = np.where(i < 201, 0.5 * (i + 1), 0.0)
    return bl.reshape(16, 16), nw.reshape(16, 16), railed.reshape(16, 16)


def test_board_stats_median_and_16_84_percentiles_use_the_live_pixels_of_each_quantity():
    bl, nw, railed = synthetic_scan()
    st = blnw.board_stats(bl, nw, blnw.live_pixels(bl, railed))
    assert (st["n_pixels_bl"], st["n_pixels_nw"]) == (101, 201)
    assert (st["bl_median"], st["bl_p16"], st["bl_p84"]) == pytest.approx((450.0, 416.0, 484.0))
    assert (st["nw_median"], st["nw_p16"], st["nw_p84"]) == pytest.approx((50.5, 16.5, 84.5))
    assert st["bl_mean"] == pytest.approx(450.0)
    assert st["nw_mean"] == pytest.approx(50.5)
    assert st["bl_mean_allpix"] == pytest.approx(np.mean(bl))
    for k in ("bl_median", "bl_p16", "bl_p84", "nw_median", "nw_p16", "nw_p84"):
        assert k in blnw.BLNW_COLUMNS


def write_scans(path, chip, scans):
    """A scan history in the newer schema: scans = [(saving time, bl, nw), ...]."""
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE baselines (row INTEGER, col INTEGER, baseline REAL, noise_width "
                "REAL, pixel_timestamp_utc TEXT, chip_name TEXT, saving_timestamp_utc TEXT, "
                "gain_mode TEXT)")
    for t, bl, nw in scans:
        con.executemany("INSERT INTO baselines VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        [(r, c, float(bl[r, c]), float(nw[r, c]), t, chip, t, "2")
                         for r in range(16) for c in range(16)])
    con.commit()
    con.close()


def write_run(folder, run, start, chips):
    """run_<NNN>_x/run_metadata.yaml: chips = {DAQ chip: (elink, offset, dac 16 x 16)}."""
    d = os.path.join(folder, "run_%03d_x" % run)
    os.makedirs(d)
    meta = dict(run_info=dict(timestamp=start, run_name="run_%03d_x" % run),
                etroc_chips={name: dict(threshold_offset=off, elink_id=el, pixels=dict(
                    dac_values={"(%d, %d)" % (r, c): float(dac[r, c])
                                for r in range(16) for c in range(16)}))
                    for name, (el, off, dac) in chips.items()})
    with open(os.path.join(d, "run_metadata.yaml"), "w") as fh:
        yaml.safe_dump(meta, fh)


def test_blnw_tables_hold_the_scan_the_run_used_pixel_by_pixel(tmp_path):
    bl, nw, railed = synthetic_scan()
    offset = 20
    dac = np.where(railed, 1023.0, bl + offset)
    # the scan the run used (05:00) and an earlier one whose baselines are all one code higher
    write_scans(str(tmp_path / "hist.sqlite"), "CHIP_A",
                [("2026-07-27 04:00:00", bl + 1.0, nw + 7.0), ("2026-07-27 05:00:00", bl, nw)])
    folder = str(tmp_path / "meta")
    write_run(folder, 4, "2026-07-27T06:00:00+00:00",
              {"CHIP_A": (0, offset, dac), "CHIP_B": (4, offset, dac)})   # CHIP_B: no scan
    with open(str(tmp_path / "runs.yaml"), "w") as fh:
        yaml.safe_dump({"h1_run4": {0: {"HV": 400}, 1: {"HV": 400}}}, fh)
    out, out_pix = str(tmp_path / "by_run.csv"), str(tmp_path / "by_pixel.csv.gz")
    blnw.build_blnw_table(out, metadata_dirs={("july", "h1"): folder},
                          sqlites={("july", "h1"): [str(tmp_path / "hist.sqlite")]},
                          run_yamls={"july": str(tmp_path / "runs.yaml")}, out_pixels_csv=out_pix)
    by_run = blnw.read_blnw(out).set_index("board_idx")
    assert list(by_run["match"]) == ["exact", "none"]
    assert by_run.loc[0, "scan_utc"] == "2026-07-27 05:00:00"
    assert by_run.loc[0, "bl_median"] == pytest.approx(450.0)
    assert by_run.loc[0, "nw_p84"] == pytest.approx(84.5)
    pix = blnw.read_blnw_pixels(out_pix)
    assert len(pix) == 256 and set(pix["board_idx"]) == {0}
    assert set(pix["chip"]) == {campaign.TELESCOPE_CHIPS["h1"][0]}
    assert set(pix["scan_utc"]) == {"2026-07-27 05:00:00"}
    assert pix["live"].dtype == bool
    pix = pix.sort_values(["row", "col"])
    np.testing.assert_array_equal(pix["bl"].to_numpy(), bl.ravel())
    np.testing.assert_array_equal(pix["nw"].to_numpy(), nw.ravel())
    np.testing.assert_array_equal(pix["live"].to_numpy(), blnw.live_pixels(bl, railed).ravel())
    live = pix[pix["live"]]
    assert len(live) == by_run.loc[0, "n_pixels_bl"]
    assert live["bl"].median() == by_run.loc[0, "bl_median"]
