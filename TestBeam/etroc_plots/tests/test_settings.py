"""Tests of the settings helpers: the settings classes of a display-run list and the accepted
merges they draw in place of their runs, the settings guard, the threshold over the noise from
the per-pixel noise widths (a merge's pooled scans too) and the offset its scan records, the slot
positions, the offset markers, the layout audit, the BL/NW point offsets and the RFSel pair
readers (frontend.tot_shapes).

Run from TestBeam/:  python -m pytest etroc_plots/tests
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "..")))
if not os.environ.get("ETROC_CAMPAIGN"):
    os.environ["ETROC_CAMPAIGN"] = "irrad_2026"

from etroc_plots.campaigns import active as campaign            # noqa: E402
from etroc_plots import style                                    # noqa: E402
from etroc_plots.frontend import blnw, tot_shapes               # noqa: E402
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


def test_accepted_merges_read_runs_merged_and_the_verdict():
    camp = campaign.TABLE_CAMPAIGN["july"]
    rows = [dict(campaign=camp, telescope="h1", run="h1_grp_run64_67", runs_merged="64+67",
                 accepted=True)] * 2
    rows += [dict(campaign=camp, telescope="h1", run="h1_grp_run33_52", runs_merged="33+52",
                  accepted=False),
             dict(campaign="Aug", telescope="offset", run="grp_run5_6", runs_merged="5+6",
                  accepted=True)]
    assert tables.accepted_merges(pd.DataFrame(rows), camp) == {"h1": [(64, 67)]}
    rows[0] = dict(rows[0], runs_merged="64+68")
    with pytest.raises(ValueError, match="h1_grp_run64_67"):
        tables.accepted_merges(pd.DataFrame(rows), camp)


def test_classes_at_step_hold_an_accepted_merge_in_place_of_its_runs(display):
    plain = {(c.rfsel, c.os): c for c in tables.classes_at_step(display, STEP)}
    # (33, 99): run 99 is not a run of the class, so that merge is not the class's
    by = {(c.rfsel, c.os): c for c in tables.classes_at_step(
        display, STEP, merges={"h1": [(64, 67), (33, 99)]})}
    merged = by[(3, "6/8")]
    assert merged.merges == {"h1": [(64, 67)]}
    assert tables.class_entries(merged, "h1") == [(64, 67)]
    assert merged.runs == {"h1": [64, 67]} and merged.preferred == {"h1": 64}
    for k in [(2, "8/10"), (2, "10")]:
        assert by[k].merges == {} and by[k][:4] == plain[k][:4]
        assert tables.class_entries(by[k], "h1") == plain[k].runs["h1"]


def pixel_table(scans):
    """A per-pixel BL/NW table (frontend.blnw.BLNW_PIXEL_COLUMNS) of H1 IH7: scans = [(run,
    scan_utc, [nw, ...], [live, ...]), ...], one row per value."""
    rows = [dict(campaign="july", tel="h1", run=run, board_idx=0, chip="IH7", row=i // 16,
                 col=i % 16, bl=500.0, nw=v, live=live[i], scan_utc=utc)
            for run, utc, nws, live in scans for i, v in enumerate(nws)]
    return pd.DataFrame(rows, columns=blnw.BLNW_PIXEL_COLUMNS)


def xb_points(runs, offset=10.0):
    return pd.DataFrame([dict(tel="h1", runs=tuple(runs), chip="IH7", offset=offset)])


def test_pixel_x_over_nw_median_and_percentiles_over_the_noise_width_population():
    # a missing and a zero noise width are left out; a pixel off the baseline rule (live False)
    # with a noise width counts, as in n_pixels_nw
    nws = [4.0, 4.0, 5.0, 6.0, 8.0, 0.0, np.nan]
    live = [True, True, False, True, True, True, False]
    out = settings.pixel_x_over_nw(pixel_table([(56, "2026-07-29 10:00:00", nws, live)]),
                                   xb_points([56])).iloc[0]
    assert (out["nw_median"], out["x"], out["n_pixels"]) == (5.0, 2.0, 5)
    # offset / nw sorted: 1.25, 1.667, 2, 2.5, 2.5; numpy's linear interpolation
    assert out["lo"] == pytest.approx(1.25 + 0.64 * (10.0 / 6.0 - 1.25))
    assert out["hi"] == pytest.approx(2.5)
    assert (out["n_scans"], out["shared_scan"], out["n_pixels_scan"]) == (1, False, [5])
    assert (out["tel"], out["chip"], out["runs"], out["offset"]) == ("h1", "IH7", (56,), 10.0)


def test_pixel_x_over_nw_pools_the_scans_of_a_merge_before_the_median():
    pix = pixel_table([(64, "2026-07-30 04:00:00", [4.0, 4.0, 4.0], [True] * 3),
                       (67, "2026-07-30 17:00:00", [8.0, 8.0], [True] * 2)])
    out = settings.pixel_x_over_nw(pix, xb_points([64, 67], offset=8.0)).iloc[0]
    assert (out["nw_median"], out["x"], out["n_pixels"]) == (4.0, 2.0, 5)   # not (2 + 1) / 2
    # offset / nw sorted: 1, 1, 2, 2, 2
    assert out["lo"] == pytest.approx(1.0) and out["hi"] == pytest.approx(2.0)
    assert (out["n_scans"], out["shared_scan"], out["n_pixels_scan"]) == (2, False, [3, 2])


def test_pixel_x_over_nw_counts_a_scan_two_runs_share_once():
    one = (18, "2026-07-25 08:00:00", [4.0, 5.0, 9.0], [True] * 3)
    pix = pixel_table([one, (19,) + one[1:]])          # run 19 reused run 18's scan
    both = settings.pixel_x_over_nw(pix, xb_points([18, 19])).iloc[0]
    alone = settings.pixel_x_over_nw(pix, xb_points([18])).iloc[0]
    assert (both["n_pixels"], both["n_scans"], both["shared_scan"]) == (3, 1, True)
    assert [both[k] for k in ("x", "lo", "hi")] == [alone[k] for k in ("x", "lo", "hi")]


def test_pixel_x_over_nw_stops_on_a_missing_or_doubled_scan():
    pix = pixel_table([(56, "2026-07-29 10:00:00", [4.0, 5.0], [True] * 2),
                       (56, "2026-07-29 11:00:00", [4.0, 5.0], [True] * 2)])
    with pytest.raises(ValueError, match="run 56 IH7: 2 scans"):
        settings.pixel_x_over_nw(pix, xb_points([56]))
    with pytest.raises(ValueError, match="run 57 IH7: 0 scans"):
        settings.pixel_x_over_nw(pix[pix["scan_utc"] < "2026-07-29 11"], xb_points([56, 57]))
    zero = pixel_table([(56, "2026-07-29 10:00:00", [0.0, np.nan], [True] * 2)])
    with pytest.raises(ValueError, match="no pixel with a noise width above 0"):
        settings.pixel_x_over_nw(zero, xb_points([56]))


def test_scan_offset_is_the_runs_common_offset():
    bl = blnw_table()
    assert settings.scan_offset(bl, "july", "h1", [64, 67], "IH7") == 8.0
    with pytest.raises(ValueError, match="offsets"):
        settings.scan_offset(bl, "july", "h1", (56, 64), "IH11")           # IH11 at 8, then 6
    with pytest.raises(ValueError, match="0 BL/NW rows"):
        settings.scan_offset(bl, "july", "h1", [99], "IH7")


def test_place_puts_a_merge_at_its_first_run():
    pts = [dict(slot="s", run=r, board_idx=0) for r in (48, 13)]
    pts.append(dict(slot="s", run="f1_grp_run46_47", runs=[46, 47], board_idx=0))
    _, run_x = settings.place(pts, {"s": 0.0}, min_run=1)
    assert sorted(run_x, key=run_x.get) == [("s", 13), ("s", "f1_grp_run46_47"), ("s", 48)]


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


def test_offset_markers_follow_the_offsets_not_the_text():
    m = settings.offset_markers(["10", "6/8", "20", "8/10", "10", "15"])
    assert m == {"6/8": "v", "8/10": "D", "10": "o", "15": "s", "20": "^"}
    with pytest.raises(ValueError, match="8 offset classes"):
        settings.offset_markers([str(v) for v in range(1, 9)])


def test_layout_boxes_skip_ticks_outside_the_view():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_xticks([0.5, 1.0, 2.0, 3.0, 4.0])      # set_xticks widens the range: limits after
    ax.set_yticks([500, 600, 700, 800])
    ax.set_xlim(0.9, 3.6)
    ax.set_ylim(550, 795)                         # "800" sits 5 units above the top: not drawn
    ticks = [t["text"] for t in settings.layout_boxes(fig, dpi=50)["axes"][0]["ticklabels"]]
    plt.close(fig)
    assert ticks == ["1.0", "2.0", "3.0", "600", "700"]


def test_points_keep_the_pitch_between_runs():
    rows = [dict(campaign="july", tel="h1", run=run, board_idx=i, bl_mean_allpix=1.0,
                 bl_std_allpix=0.1, nw_mean_allpix=4.0, nw_std_allpix=1.0, hv=550.0,
                 match="scan")
            for run in (31, 32, 33) for i in range(4)]
    steps = [dict(campaign="july", runs={"h1": [33, 31, 32]}),
             dict(campaign="july", runs={"h1": [31, 33]})]
    q, st = [dict(key="nw")], [dict(key="allpix")]
    for kw, want in ((dict(span=0.8), [-0.4, 0.0, 0.4, -0.4, 0.4]),
                     (dict(pitch=0.25), [-0.25, 0.0, 0.25, -0.125, 0.125])):
        pts = [p for p in blnw.points(pd.DataFrame(rows), steps, q, st, **kw)
               if p["board_idx"] == 0]
        assert [(p["step"], p["run"]) for p in pts] == [(0, 31), (0, 32), (0, 33), (1, 31),
                                                        (1, 33)]
        assert [p["dx"] for p in pts] == pytest.approx(want)


# An RFSel pair of H1 board 0 (IH7, threshold offset 8 in both runs of RUNS): run 56 at RFSel 2
# and run 64 at RFSel 3. Each run's rows of RFSEL_PAIR_HIST_CSV: 3 pixels counted in both runs,
# the events in two stored 0.05 ns bins of one 0.1 ns bin, and one more bin.
PAIR = (56, 64)
PAIR_ROWS = {56: dict(rfsel=2, median=3.2, bins={3.0: 30, 3.05: 50, 3.3: 20}),
             64: dict(rfsel=3, median=2.44, bins={2.4: 60, 2.6: 20})}


def pair_hist(n_pixels=(3, 3), offset=8):
    rows = []
    for (run, partner), npx in zip(((56, 64), (64, 56)), n_pixels):
        d = PAIR_ROWS[run]
        rows += [dict(campaign="july", tel="h1", run=run, partner_run=partner, board_idx=0,
                      chip="IH7", rfsel=d["rfsel"], offset=offset, n_pixels=npx,
                      n_events=sum(d["bins"].values()), median_ns=d["median"], bin_lo_ns=lo,
                      count=n)
                 for lo, n in d["bins"].items()]
    return pd.DataFrame(rows)


def test_median_shift_reads_both_runs_of_a_pair():
    out = tot_shapes.median_shift(pair_hist(), "july", "h1", PAIR)
    assert {k: v for k, v in out.items() if k != "shift_ns"} == dict(
        runs=[56, 64], chip="IH7", board_idx=0, rfsel=[2, 3], offset=[8, 8], n_pixels=3,
        n_events=[100, 80], median_ns=[3.2, 2.44])
    assert out["shift_ns"] == pytest.approx(2.44 - 3.2)
    with pytest.raises(ValueError, match="pixels"):
        tot_shapes.median_shift(pair_hist(n_pixels=(3, 4)), "july", "h1", PAIR)
    with pytest.raises(ValueError, match="no rows"):
        tot_shapes.median_shift(pair_hist(), "july", "h1", (56, 67))


def test_rfsel_curves_normalise_each_run_and_check_the_settings():
    stop = pd.Timestamp("2026-07-01 00:00", tz="UTC")
    run_times = pd.DataFrame(dict(campaign="july", tel="h1", run_num=[56, 64],
                                  start_utc=[stop + pd.Timedelta(hours=10),
                                             stop + pd.Timedelta(hours=72)]))
    out = tot_shapes.rfsel_curves(pair_hist(), "july", "h1", (64, 56), runs_summary(),
                                  run_times, stop, label="%(rfsel)s run %(run)d %(time)s "
                                  "%(median).2f %(n)d", note="%(chip)s %(n_pixels)d")
    assert [c["rfsel"] for c in out["curves"]] == [2, 3]          # by RFSel, not by `pair`
    c2 = out["curves"][0]
    assert c2["label"] == "%s run 56 +10 h 3.20 100" % style.rfsel_text(2)
    assert c2["median"] == 3.2 and c2["edges"][1] - c2["edges"][0] == pytest.approx(0.1)
    i = int(np.argmax(c2["y"]))
    assert c2["edges"][i] == pytest.approx(3.0) and c2["y"][i] == pytest.approx(0.8)
    assert [float(np.sum(c["y"])) for c in out["curves"]] == pytest.approx([1.0, 1.0])
    assert out["note"] == "IH7 3"
    assert out["subject"] == style.settings_text(rfsel=[2, 3], offset=8)
    assert [(r["run"], r["partner_run"], r["hours"], r["n_events"]) for r in out["records"]] == [
        (64, 56, 72.0, 80), (56, 64, 10.0, 100)]
    with pytest.raises(ValueError, match="in the run summary"):
        tot_shapes.rfsel_curves(pair_hist(offset=10), "july", "h1", PAIR, runs_summary(),
                                run_times, stop, label="", note="")
