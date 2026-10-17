"""Tests of the resolution helpers: the merge rule, the good-run list with merges, the points of
single and merged runs, the standard operating point, the stop clock with its bounds, the hours
since the first run at a step, the colour rule; the fluence ladder's picks (step_winners), the
one-run-per-condition selector, the floor-"none" pixel lookup and the numbers table.

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
from etroc_plots.etroc_style import BOARD_COLOR                  # noqa: E402
from etroc_plots.iv.iv_plot import TIMING_COLORS                 # noqa: E402
from etroc_plots.resolution import curves, ladder, numbers_table, tables   # noqa: E402

JUL, MAR = campaign.TABLE_CAMPAIGN["july"], campaign.TABLE_CAMPAIGN["march"]
CHIPS = campaign.TELESCOPE_CHIPS["h1"]


def board_table(runs, camp=JUL, variant="top", floor="300", base=40.0, n0=250):
    """Anointed rows for every board of each run (a run number or a merge name), value_ps =
    `base` + board index, plus one non-anointed row per board (5 ps higher)."""
    rows = []
    for run in runs:
        for i, chip in enumerate(CHIPS):
            for anointed, combo in ((True, "a"), (False, "b")):
                rows.append(dict(campaign=camp, telescope="h1", run=str(run), combo=combo,
                                 board_idx=i, board_chip="PT_" + chip, variant=variant,
                                 floor=floor, value_ps=base + i + (0 if anointed else 5),
                                 pixel_spread_ps=3.0, n_pixels=n0 + i, anointed=anointed))
    return pd.DataFrame(rows)


def runs_summary(rows):
    """rows: (campaign, run, fluence text, hv, offset, rfsel or None) for H1; hv is one bias for
    every board or one per board."""
    out = []
    for camp, run, irrad, hv, offset, rfsel in rows:
        r = dict(campaign=camp, telescope=campaign.TABLE_TEL["h1"], run=run)
        for i, chip in enumerate(CHIPS):
            r.update({"board%d_chip" % i: "PT_" + chip,
                      "board%d_hv" % i: hv[i] if isinstance(hv, (list, tuple)) else hv,
                      "board%d_irrad" % i: irrad, "board%d_offset" % i: offset,
                      "board%d_rfsel" % i: float("nan") if rfsel is None else float(rfsel)})
        out.append(r)
    return pd.DataFrame(out)


@pytest.fixture
def good_csv(tmp_path):
    rows = ["campaign,telescope,run,good,reason"]
    rows += ["%s,H1_HPK,%s,%s,r%s" % (campaign.GOOD_RUNS_CAMPAIGN, run, good, run)
             for run, good in (("3", True), ("4", True), ("3+4", True), ("14+15", True),
                               ("18+19", False), ("18", True), ("19", False))]
    rows.append("%s,F1_FBK,3+4,True,x" % campaign.GOOD_RUNS_CAMPAIGN)     # other telescope
    p = tmp_path / "good.csv"
    p.write_text("\n".join(rows) + "\n")
    return str(p)


# ------------------------------------------------------------------------------ merge rule
def test_merge_rule_takes_3_4_and_refuses_18_19_and_14_15(good_csv):
    bm = board_table([tables.merge_name("h1", m) for m in ((3, 4), (14, 15), (18, 19))])
    merges = curves.table_merges(bm, "h1", "july")
    assert merges == [(3, 4), (14, 15), (18, 19)]
    good = curves.good_list("july", "h1", path=good_csv)
    assert good["3+4"][0] is True and good["18+19"][0] is False and "46+47" not in good

    # 1.5e15: display class {3, 4}, preferred 4 -> the merge 3+4 replaces it
    drawn, taken, refused = curves.merge_rule([4], merges, [3, 4], good)
    assert drawn == [(3, 4)] and taken == {(3, 4): [4]}
    assert set(refused) == {(14, 15), (18, 19)}
    # 3.5e15: display class {18} (19 off the list) -> 18 stays single
    drawn, taken, refused = curves.merge_rule([18], merges, [18], good)
    assert drawn == [18] and taken == {}
    assert "19 not in the display class" in refused[(18, 19)]
    assert "not good" in refused[(18, 19)]
    assert refused[(14, 15)].startswith("runs 14, 15 not in the display class")


def test_merge_rule_needs_both_conditions(good_csv):
    good = curves.good_list("july", "h1", path=good_csv)
    # good as a merge but a run off the class: refused
    drawn, taken, _ = curves.merge_rule([14], [(14, 15)], [14], good)
    assert drawn == [14] and not taken
    # every run in the class but not good as a merge: refused
    drawn, taken, refused = curves.merge_rule([18, 19], [(18, 19)], [18, 19], good)
    assert drawn == [18, 19] and not taken and "not good" in refused[(18, 19)]
    # both runs drawn and the merge taken: one entry where the first run stood
    drawn, taken, _ = curves.merge_rule([3, 4, 5], [(3, 4)], [3, 4, 5], good)
    assert drawn == [(3, 4), 5] and taken == {(3, 4): [3, 4]}
    assert curves.not_good([(3, 4), 19, 7], good) == {"19": "r19", "7": "not on the good-run list"}


# ---------------------------------------------------------------------------------- points
def test_points_single_merge_and_march_rfsel():
    bt = pd.concat([board_table([4]), board_table([2], camp=MAR)])
    bm = board_table([tables.merge_name("h1", (3, 4))])
    rs = runs_summary([(JUL, 3, "1.5e15", 540.0, 20, 2), (JUL, 4, "1.5e15", 540.0, 20, 2),
                       (MAR, 2, "pre-irrad", 170.0, 20, None)])
    pts = curves.points(bt, bm, rs, "h1", [4, (3, 4)], "july")
    assert [(p["run"], p["board_idx"]) for p in pts][:2] == [(4, 0), (4, 1)]
    single, merged = pts[0], pts[4]
    assert single["value_ps"] == 40.0 and single["bar_ps"] == 3.0 and single["n_pixels"] == 250
    assert not single["merged"] and single["band"] is None and single["y_ps"] == 40.0
    assert merged["run"] == "h1_grp_run3_4" and merged["label"] == "3+4" and merged["merged"]
    assert merged["chip"] == "IH7" and merged["hv"] == 540.0 and merged["rfsel"] == 2
    march = curves.points(bt, bm, rs, "h1", [2], "march")
    assert {p["rfsel"] for p in march} == {campaign.MARCH_RFSEL}
    assert {p["rfsel_source"] for p in march} == {"MARCH_RFSEL"}
    assert {p["fluence"] for p in march} == {0.0}
    with pytest.raises(ValueError):
        curves.points(bt, bm, rs, "h1", [3], "july")      # no single-run rows for run 3


def test_standard_operating_point():
    pts = [dict(offset=20, rfsel=2), dict(offset=20, rfsel=None), dict(offset=10, rfsel=2),
           dict(offset=20, rfsel=3), dict(offset=20, rfsel=1)]
    assert curves.standard(pts) == pts[:2]


# --------------------------------------------------------------------------------- clocks
def test_hours_after_stop_with_bounds():
    rt = pd.DataFrame(dict(campaign=["july", "july"], tel=["h1", "h1"], run_num=[11, 12],
                           start_utc=pd.to_datetime(["2026-07-21 03:42:40",
                                                     "2026-07-20 18:30:00"], utc=True)))
    hours = curves.hours_after_stop(rt, "h1", [11, 12], 2e15)
    # counted from the earliest allowed stop, 16:30:00 UTC
    assert hours[12] == pytest.approx(2.0)
    assert hours[11] == pytest.approx(11.0 + 12.0 / 60 + 40.0 / 3600)
    lo, hi = curves.stop_band(2e15)
    assert lo == 0.0 and hi == pytest.approx(1.0 + 47.0 / 60 + 43.0 / 3600)   # to 18:17:43
    assert curves.stop_band(3.5e15) is None                                  # 2 min: no band


def test_hours_since_first_run_at_the_step():
    rt = pd.DataFrame(dict(campaign=["july"] * 3, tel=["h1"] * 3, run_num=[3, 4, 9],
                           start_utc=pd.to_datetime(["2026-07-16 16:42:04", "2026-07-16 19:07:08",
                                                     "2026-07-16 10:00:00"], utc=True)))
    rs = runs_summary([(JUL, 3, "1.5e15", 540.0, 20, 2), (JUL, 4, "1.5e15", 540.0, 20, 2),
                       (JUL, 9, "2e15", 540.0, 20, 2)])
    hours, first = curves.hours_since_first(rt, rs, "h1", [3, 4], 1.5e15)
    assert first == 3                       # run 9 started earlier but at another step
    assert hours[3] == 0.0 and hours[4] == pytest.approx(2.0 + 25.0 / 60 + 4.0 / 3600)


# --------------------------------------------------------------------------------- colour
def test_colour_rule():
    one = [dict(camp="march", fluence=3e14, board_idx=i) for i in range(4)]
    steps = one + [dict(camp="march", fluence=9e14, board_idx=0)]
    looks = [dict(camp="march", fluence=1.5e15, board_idx=0),
             dict(camp="july", fluence=1.5e15, board_idx=0)]
    assert curves.color_rule(one) == "board"
    assert curves.color_rule(steps) == "fluence"
    assert curves.color_rule(looks) == "fluence"          # one step, two looks
    colors = curves.look_colors(looks)
    assert colors[("march", 1.5e15)] == campaign.FLUENCE_COLOR[1.5e15]
    assert colors[("july", 1.5e15)] == TIMING_COLORS[0]
    _, color_of = curves._color_of(one, None)
    assert [color_of(p) for p in one] == [BOARD_COLOR[i] for i in range(4)]


# ------------------------------------------------------------------------ the fluence ladder
F1_LOW, F1_HIGH = (390.0, 390.0, 370.0, 390.0), (480.0, 480.0, 450.0, 480.0)


def ladder_inputs():
    """A small campaign: March pre-irradiation runs 1 (170 V), 2 (190 V) and 5 (190 V at offset
    10), 3e14 run 10; July 1.5e15 runs 3, 4 (the merge 3+4 exists) and 3.5e15 run 18 (the merge
    18+19 exists, not good); top and central rows at floor 300."""
    singles = [(MAR, r) for r in (1, 2, 5, 10)] + [(JUL, r) for r in (3, 4, 18)]
    merges = [tables.merge_name("h1", m) for m in ((3, 4), (18, 19))]
    bt = pd.concat([board_table([r], camp=c, variant=v, base=b)
                    for c, r in singles for v, b in (("top", 40.0), ("central", 30.0))])
    bm = pd.concat([board_table(merges, variant=v, base=b)
                    for v, b in (("top", 50.0), ("central", 35.0))])
    rs = runs_summary([(MAR, 1, "pre-irrad", 170.0, 20, None),
                       (MAR, 2, "pre-irrad", 190.0, 20, None),
                       (MAR, 5, "pre-irrad", 190.0, 10, None),
                       (MAR, 10, "0.3e15", 280.0, 20, None),
                       (JUL, 3, "1.5e15", 540.0, 20, 2), (JUL, 4, "1.5e15", 540.0, 20, 2),
                       (JUL, 18, "3.5e15", (530.0, 550.0, 550.0, 550.0), 20, 2),
                       (JUL, 19, "3.5e15", (530.0, 550.0, 550.0, 550.0), 20, 2)])
    classes = {1.5e15: ({"h1": [3, 4]}, {"h1": 4}), 3.5e15: ({"h1": [18]}, {"h1": 18})}
    return bt, bm, rs, classes


def test_step_winners_march_highest_bias_july_preferred_or_merge(good_csv):
    bt, bm, rs, classes = ladder_inputs()
    good = curves.good_list("july", "h1", path=good_csv)
    merges = curves.table_merges(bm, "h1", "july")
    pts, picks = ladder.step_winners(bt, bm, rs, "h1", classes, good, merges, march_runs=[2, 10])
    by = {(p["camp"], p["step"]): p["label"] for p in pts}
    # March: run 2 is the highest-bias standard run (run 5 at 190 V sits at offset 10)
    assert by == {("march", 0.0): "2", ("march", 3e14): "10", ("july", 1.5e15): "3+4",
                  ("july", 3.5e15): "18"}
    assert len(pts) == 4 * 4
    j15 = [p for p in pts if p["label"] == "3+4"]
    assert all(p["merged"] and p["preferred"] for p in j15) and j15[0]["value_ps"] == 50.0
    assert all(p["band"] is not None for p in pts if p["step"] == 3.5e15)
    assert all(p["preferred"] is None for p in pts if p["camp"] == "march")
    july = {k["step"]: k for k in picks if k["camp"] == "july"}
    assert july[1.5e15]["taken"] == {"3+4": [4]}
    assert "not good" in july[3.5e15]["refused"]["18+19"]
    # a listed March run that is not the highest-bias standard run of its step is refused
    with pytest.raises(ValueError, match="not the highest-bias"):
        ladder.step_winners(bt, bm, rs, "h1", classes, good, merges, march_runs=[1, 10])
    assert ladder.highest_bias_run(rs, "h1", [1, 2], "march") == 2
    with pytest.raises(ValueError):
        ladder.highest_bias_run(rs, "h1", [2, 5], "march")      # two runs at 190 V


def test_numbers_table_rows_and_restored_merge_cells(good_csv, tmp_path):
    bt, bm, rs, classes = ladder_inputs()
    good = curves.good_list("july", "h1", path=good_csv)
    pts, _ = ladder.step_winners(bt, bm, rs, "h1", classes, good,
                                 curves.table_merges(bm, "h1", "july"), march_runs=[2, 10])
    central = numbers_table.central_values(bt, bm, pts, "300")
    df = numbers_table.numbers_table(pts, central)
    assert list(df.columns) == numbers_table.COLUMNS and len(df) == 16
    merged = df[df["run"] == "3+4"]
    assert len(merged) == 4 and set(merged["merged"]) == {"yes"}
    assert set(merged["preferred run"]) == {"yes"}
    assert list(merged["central value [ps]"]) == [35.0, 36.0, 37.0, 38.0]
    assert set(df.loc[df["campaign"] == "March", "preferred run"]) == {""}
    band = df[df["bookkeeping fluence [p/cm2]"] == 3.5e15]
    # band rows: top and central at the band centre (anointed 40 + i, other combo 45 + i)
    assert list(band["top value [ps]"]) == [42.5, 43.5, 44.5, 45.5]
    assert list(band["central value [ps]"]) == [32.5, 33.5, 34.5, 35.5]
    assert list(band["anointed value [ps]"]) == [40.0, 41.0, 42.0, 43.0]
    assert list(band["central anointed value [ps]"]) == [30.0, 31.0, 32.0, 33.0]
    assert df.loc[df["bookkeeping fluence [p/cm2]"] != 3.5e15, "band low [ps]"].isna().all()
    paths = numbers_table.write_table(df, str(tmp_path), "t")
    assert [os.path.basename(p) for p in paths] == ["t.csv", "t.md"]
    assert len(open(paths[1]).read().splitlines()) == 2 + 16
    assert len(pd.read_csv(paths[0])) == 16


def test_one_per_condition_preferred_or_later_then_merge(good_csv):
    rs = runs_summary([(JUL, 3, "1.5e15", 540.0, 20, 2), (JUL, 4, "1.5e15", 540.0, 20, 2),
                       (JUL, 13, "3.5e15", F1_LOW, 20, 2), (JUL, 14, "3.5e15", F1_LOW, 20, 2),
                       (JUL, 18, "3.5e15", F1_HIGH, 20, 2)])
    good = curves.good_list("july", "h1", path=good_csv)
    # two runs at one bias, the preferred one among them: it stays, then its merge replaces it
    kept, dropped = curves.one_per_condition(rs, "h1", [3, 4], preferred=4)
    assert kept == [4] and dropped == {3: "same bias on every board as run 4, the preferred run"}
    drawn, taken, _ = curves.merge_rule(kept, [(3, 4)], [3, 4], good)
    assert drawn == [(3, 4)] and taken == {(3, 4): [4]}
    # 13 and 14 share every board's bias, the preferred run 18 does not: 14, the later run
    kept, dropped = curves.one_per_condition(rs, "h1", [13, 14, 18], preferred=18)
    assert kept == [14, 18]
    assert dropped == {13: "same bias on every board as run 14, the later run"}
    assert curves.one_per_condition(rs, "h1", [14, 13], preferred=None)[0] == [14]


def test_floor_none_pixels_missing_row_is_none():
    bt = pd.concat([board_table([18], floor="300"), board_table([18], floor="none", n0=255)])
    bt = bt[~((bt["floor"] == "none") & (bt["board_idx"] == 3))]       # board 3: no row
    pts = [dict(tel="h1", camp="july", run=18, label="18", board_idx=i, merged=False)
           for i in range(4)]
    n = ladder.floor_none_pixels(bt, bt.iloc[:0], pts)
    assert n == {("h1", "july", "18", 0): 255, ("h1", "july", "18", 1): 256,
                 ("h1", "july", "18", 2): 257, ("h1", "july", "18", 3): None}
    with pytest.raises(ValueError):
        ladder.floor_none_pixels(pd.concat([bt, bt]), bt.iloc[:0], pts)
