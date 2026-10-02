"""Tests of the settings helpers: the settings classes of a display-run list, the settings guard,
the threshold over the noise and the slot positions.

Run from TestBeam/:  python -m pytest etroc_plots/tests
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "..")))
if not os.environ.get("ETROC_CAMPAIGN"):
    os.environ["ETROC_CAMPAIGN"] = "irrad_2026"

from etroc_plots.campaigns import active as campaign            # noqa: E402
from etroc_plots.resolution import settings, tables             # noqa: E402

STEP = 3.5e15
# (run, rfsel, os text, preferred, per-board offsets): H1 runs of three classes, two of them
# with boards at two offsets
RUNS = [(33, 2, "10", 1, (10, 10, 10, 10)), (56, 2, "8/10", 1, (8, 8, 10, 10)),
        (64, 3, "6/8", 1, (8, 6, 8, 8)), (67, 3, "6/8", 0, (8, 6, 8, 8))]


@pytest.fixture
def display(tmp_path):
    rows = ["telescope,fluence,rfsel,os,run,preferred,combo,bias_note,note"]
    rows += ["H1,3.5e15,%d,%s,%d,%d,band,," % (rfsel, os_text, run, pref)
             for run, rfsel, os_text, pref, _ in RUNS]
    rows.append("H1,1.5e15,2,20,5,1,,,")          # another step: never read at 3.5e15
    p = tmp_path / "display.csv"
    p.write_text("\n".join(rows) + "\n")
    return str(p)


def runs_summary():
    rows = []
    for run, rfsel, _, _, offsets in RUNS:
        r = dict(campaign=campaign.TABLE_CAMPAIGN["july"], telescope=campaign.TABLE_TEL["h1"],
                 run=run)
        for i, chip in enumerate(campaign.TELESCOPE_CHIPS["h1"]):
            r.update({"board%d_chip" % i: "PT_" + chip, "board%d_hv" % i: 550.0,
                      "board%d_irrad" % i: "3.5e15", "board%d_offset" % i: offsets[i],
                      "board%d_rfsel" % i: float(rfsel)})
        rows.append(r)
    return pd.DataFrame(rows)


def blnw_table():
    rows = [dict(campaign="july", tel="h1", run=run, board_idx=i, chip=chip, rfsel=rfsel,
                 rfsel_src="scan", offset=offsets[i], nw_mean=4.0, nw_std=1.0, n_pixels_nw=255)
            for run, rfsel, _, _, offsets in RUNS
            for i, chip in enumerate(campaign.TELESCOPE_CHIPS["h1"])]
    return pd.DataFrame(rows)


def test_classes_at_step_keeps_two_offset_classes_as_text(display):
    classes = tables.classes_at_step(display, STEP)
    assert [(c.rfsel, c.os) for c in classes] == [(3, "6/8"), (2, "8/10"), (2, "10")]
    by = {c.os: c for c in classes}
    assert by["6/8"].runs == {"h1": [64, 67]} and by["6/8"].preferred == {"h1": 64}
    assert by["8/10"].runs == {"h1": [56]}
    assert tables.os_offsets("8/10") == (8, 10)


def test_guard_passes_on_agreeing_tables(display):
    out = settings.settings_guard(runs_summary(), display, blnw_table(), fluence=STEP)
    assert out == dict(runs=4, boards=16, boards_scan_rfsel=16)


def test_guard_a_runs_summary_rfsel_differs_from_the_display_list(display):
    rs = runs_summary()
    rs.loc[rs["run"] == 64, ["board%d_rfsel" % i for i in range(4)]] = 2.0
    bl = blnw_table()
    bl.loc[bl["run"] == 64, "rfsel_src"] = "run_yaml"     # check (c) does not see run 64
    with pytest.raises(ValueError,
                       match=r"H1 run 64 .*runs_summary RFSel per board \[2, 2, 2, 2\]"):
        settings.settings_guard(rs, display, bl, fluence=STEP)


def test_guard_b_board_offsets_differ_from_the_class_text(display):
    rs = runs_summary()
    rs.loc[rs["run"] == 56, "board2_offset"] = 8
    rs.loc[rs["run"] == 56, "board3_offset"] = 8
    with pytest.raises(ValueError, match=r"H1 run 56 .*offset per board \[8, 8, 8, 8\]"):
        settings.settings_guard(rs, display, blnw_table(), fluence=STEP)


def test_guard_c_runs_summary_rfsel_differs_from_the_scan(display):
    bl = blnw_table()
    bl.loc[(bl["run"] == 33) & (bl["board_idx"] == 1), "rfsel"] = 3
    with pytest.raises(ValueError, match=r"H1 run 33 .*board 1 \(PT_IH11\): runs_summary RFSel 2, "
                                         r"autocalibration scan RFSel 3"):
        settings.settings_guard(runs_summary(), display, bl, fluence=STEP)


def test_x_over_nw_arithmetic():
    bl = blnw_table()
    out = settings.x_over_nw(bl, "july", "h1", 56, "IH12")
    assert out == dict(x=10.0 / 4.0, bar=10.0 * 1.0 / 16.0, offset=10.0, nw_mean=4.0,
                       nw_std=1.0, n_live=255)
    bl.loc[(bl["run"] == 56) & (bl["chip"] == "IH12"), "nw_mean"] = 0.0
    with pytest.raises(ValueError, match="no noise width"):
        settings.x_over_nw(bl, "july", "h1", 56, "IH12")


def test_class_slots_group_by_offset_with_a_wider_gap():
    classes = [(3, "6/8"), (2, "8/10"), (2, "10"), (3, "10"), (1, "20"), (2, "20")]
    assert settings.class_slots(classes, group_gap=0.5) == {
        (3, "6/8"): 0.0, (2, "8/10"): 1.5, (2, "10"): 3.0, (3, "10"): 4.0, (1, "20"): 5.5,
        (2, "20"): 6.5}


def test_quarter_slots_and_runs_side_by_side():
    xs, centres = settings.quarter_slots([("a", [8, 10]), ("b", [6, 8]), ("c", [10])],
                                         quarter=2.0, pitch=1.0)
    assert centres == [1.0, 3.0, 5.0]
    assert xs == {("a", 8): 0.5, ("a", 10): 1.5, ("b", 6): 2.5, ("b", 8): 3.5, ("c", 10): 5.0}
    pts = [dict(slot="s", run=run, board_idx=i) for run in (2, 1) for i in range(4)]
    cell, run_x = settings.place(pts, {"s": 0.0}, width=0.95, run_gap=1.5)
    assert cell == pytest.approx(0.95 / 9.5)
    assert run_x[("s", 1)] == pytest.approx(-2.75 * cell) and run_x[("s", 2)] == pytest.approx(
        2.75 * cell)
    xs1 = [p["x"] for p in sorted(pts, key=lambda p: (p["run"], p["board_idx"]))]
    assert xs1 == sorted(xs1)                     # run 1 left of run 2, boards in order


def test_place_gives_a_one_board_run_three_cells():
    pts = [dict(slot="a", run=r, board_idx=1) for r in (64, 67)]
    pts += [dict(slot="b", run=56, board_idx=i) for i in range(4)]
    pts += [dict(slot="b", run=57, board_idx=i) for i in range(4)]
    cell, run_x = settings.place(pts, {"a": 0.0, "b": 10.0}, width=0.95, run_gap=1.5, min_run=3)
    assert cell == pytest.approx(0.95 / 9.5)      # the fullest slot, b, sets the cell
    assert run_x[("a", 67)] - run_x[("a", 64)] == pytest.approx(4.5 * cell)
    assert [p["x"] for p in pts[:2]] == pytest.approx([run_x[("a", 64)], run_x[("a", 67)]])
