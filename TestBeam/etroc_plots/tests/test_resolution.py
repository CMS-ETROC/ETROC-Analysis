"""Tests of the resolution helpers: the merge rule and its per-figure override, the good-run list
with merges, every display run as single runs, the points of single and merged runs, the
standard operating point, the stop clock with its bounds, the hours since the first run at a
step, the colour rule, the low-pixel threshold and the good-run list reader the settings figures
share; the fluence ladder's picks (step_winners), the one-run-per-condition selector, the
floor-"none" pixel lookup and the numbers table; the
combination split's runs, points and layout, with and without merges; the bias summary drawn in
part (re12 to re14) with the whole figure's colours; the combination checks (re16 to re18): the
partner-excluded combination, the pair-corrected midpoint, a missing combination reported, the HV
cycle between runs and the shared autocalibration scan.

Run from TestBeam/:  python -m pytest etroc_plots/tests
"""
import inspect
import math
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
from etroc_plots.resolution import (combo_check, combo_split, curves, ladder,  # noqa: E402
                                    numbers_table, settings, tables)

JUL, MAR = campaign.TABLE_CAMPAIGN["july"], campaign.TABLE_CAMPAIGN["march"]
CHIPS = campaign.TELESCOPE_CHIPS["h1"]


def board_table(runs, camp=JUL, variant="top", floor="300", base=40.0, n0=250):
    """Reference-combination rows for every board of each run (a run number or a merge name),
    value_ps = `base` + board index, plus one row per board in another combination (5 ps
    higher)."""
    rows = []
    for run in runs:
        for i, chip in enumerate(CHIPS):
            for ref, combo in ((True, "a"), (False, "b")):
                rows.append(dict(campaign=camp, telescope="h1", run=str(run), combo=combo,
                                 board_idx=i, board_chip="PT_" + chip, variant=variant,
                                 floor=floor, value_ps=base + i + (0 if ref else 5),
                                 pixel_spread_ps=3.0, n_pixels=n0 + i,
                                 **{tables.REFERENCE_COLUMN: ref}))
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
    # band rows: top and central at the band centre (reference combination 40 + i, other combo 45 + i)
    assert list(band["top value [ps]"]) == [42.5, 43.5, 44.5, 45.5]
    assert list(band["central value [ps]"]) == [32.5, 33.5, 34.5, 35.5]
    assert list(band["reference combination value [ps]"]) == [40.0, 41.0, 42.0, 43.0]
    assert list(band["central reference combination value [ps]"]) == [30.0, 31.0, 32.0, 33.0]
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


# ------------------------------------------------------------------- population overrides
def test_merge_override_draws_single_runs_and_names_the_merges(good_csv):
    good = curves.good_list("july", "h1", path=good_csv)
    merges = [(3, 4), (14, 15), (18, 19)]
    # 1.5e15, preferred run 4: the rule would draw 3+4, the override keeps 4 and names 3+4
    drawn, over = curves.merge_override([4], merges, [3, 4], good)
    assert drawn == [4] and over == {(3, 4): [4]}
    # both display runs: both stay single, the merge named with the runs it would replace
    drawn, over = curves.merge_override([3, 4], merges, [3, 4], good)
    assert drawn == [3, 4] and over == {(3, 4): [3, 4]}
    # no merge the rule would take: nothing to name
    assert curves.merge_override([18], merges, [18], good) == ([18], {})


def test_display_singles_every_run_of_every_step():
    classes = {2e15: ({"h1": [12, 11], "f1": [10, 11, 12]}, {"h1": 12, "f1": 12}),
               1.5e15: ({"h1": [4, 3], "f1": [7, 8, 9]}, {"h1": 4, "f1": 9}),
               3.5e15: ({"h1": [18], "f1": [18, 14, 13]}, {"h1": 18, "f1": 18})}
    # by step, then by run; runs at one bias (F1 13, 14) and the runs of a merge (H1 3, 4) all kept
    assert curves.display_singles(classes, "h1") == [3, 4, 11, 12, 18]
    assert curves.display_singles(classes, "f1") == [7, 8, 9, 10, 11, 12, 13, 14, 18]
    assert curves.display_singles(classes, "x1") == []


def test_one_good_run_list_reader_for_settings_and_resolution(good_csv):
    # the merge verdict of the settings figures (tables.good_merges) and the merge rule here read
    # the good-run list through one function
    assert curves.good_list is tables.good_list
    assert curves.good_list("july", "h1", path=good_csv)["18+19"] == (False, "r18+19")


# ------------------------------------------------------------------------------ low pixels
def test_low_pixels_below_128_only():
    assert curves.LOW_PIXELS == 128
    # one threshold for the settings and resolution figures: the settings figures mark a board
    # through settings.low_pixels, whose threshold defaults to tables.LOW_PIXELS
    assert curves.LOW_PIXELS == tables.LOW_PIXELS
    threshold = inspect.signature(settings.low_pixels).parameters["threshold"].default
    assert threshold == curves.LOW_PIXELS
    pts = [dict(tel="h1", label="11", board_idx=i, chip=c, n_pixels=n)
           for i, (c, n) in enumerate(zip(CHIPS, (127, 128, 45, 256)))]
    assert [curves.low_pixels(p) for p in pts] == [True, False, True, False]
    assert [q["n_pixels"] for q in curves.low_pixel_points(pts)] == [127, 45]
    assert curves._with_low({}, pts)["low_pixels"]["points"] == curves.low_pixel_points(pts)
    assert "low_pixels" not in curves._with_low({}, pts[1::2])


# ------------------------------------------------------------------------ combination split
@pytest.fixture
def split_display(tmp_path):
    """H1 and F1 rows of a display-run list: the references at 1.5e15 and 2e15 and the runs at
    3.5e15 in five classes; H1 run 19 is not listed."""
    rows = ["telescope,fluence,rfsel,os,run,preferred,combo,bias_note,note"]
    for tel, f, rfsel, os_text, run, pref in (
            ("H1", "1.5e15", 2, "20", 3, 0), ("H1", "1.5e15", 2, "20", 4, 1),
            ("H1", "2e15", 2, "20", 11, 0), ("H1", "2e15", 2, "20", 12, 1),
            ("H1", "3.5e15", 2, "20", 18, 1), ("H1", "3.5e15", 2, "10", 33, 1),
            ("H1", "3.5e15", 2, "8/10", 56, 1), ("H1", "3.5e15", 3, "6/8", 67, 0),
            ("H1", "3.5e15", 3, "6/8", 64, 1), ("H1", "3.5e15", 1, "20", 48, 1),
            ("H1", "3.5e15", 1, "20", 47, 0),
            ("F1", "1.5e15", 2, "20", 9, 1), ("F1", "2e15", 2, "20", 12, 1),
            ("F1", "3.5e15", 1, "20", 47, 1), ("F1", "3.5e15", 1, "20", 46, 0),
            ("F1", "3.5e15", 2, "20", 18, 1), ("F1", "3.5e15", 2, "20", 13, 0),
            ("F1", "3.5e15", 2, "20", 14, 0)):
        rows.append("%s,%s,%d,%s,%d,%d,band,," % (tel, f, rfsel, os_text, run, pref))
    p = tmp_path / "display.csv"
    p.write_text("\n".join(rows) + "\n")
    return str(p)


def test_split_runs_references_then_every_run_by_class(split_display, tmp_path):
    split = combo_split.split_runs(split_display, (1.5e15, 2e15), 3.5e15)
    h1, f1 = split["h1"], split["f1"]
    # references: the preferred runs only; then the classes in offset order, runs by number
    assert [e["run"] for e in h1] == [4, 12, 64, 67, 56, 33, 47, 48, 18]
    assert [e["ref"] for e in h1] == [True, True] + [False] * 7
    assert [(e["rfsel"], e["os"]) for e in h1[2:5]] == [(3, "6/8"), (3, "6/8"), (2, "8/10")]
    assert 19 not in [e["run"] for e in h1]
    # F1 46 and 47: two single runs, never their merge; 13 and 14 both kept
    assert [e["run"] for e in f1] == [9, 12, 46, 47, 13, 14, 18]
    assert [e["preferred"] for e in f1 if e["step"] == 3.5e15] == [False, True, False, False,
                                                                     True]
    # a reference step without a preferred run on a telescope stops the figure
    p = tmp_path / "nopref.csv"
    p.write_text(open(split_display).read().replace("H1,2e15,2,20,12,1", "H1,2e15,2,20,12,0"))
    with pytest.raises(ValueError):
        combo_split.split_runs(str(p), (1.5e15, 2e15), 3.5e15)


def test_combo_points_every_combination_and_layout():
    bt = board_table([18], n0=126)
    bt["combo"] = bt[tables.REFERENCE_COLUMN].map({True: "extra0-dut1-ref2", False: "dut1-ref2-trig3"})
    rs = runs_summary([(JUL, 18, "3.5e15", 550.0, 20, 2)])
    e = dict(run=18, step=3.5e15, rfsel=2, os="20", ref=False, preferred=True)
    pts = combo_split.combo_points(bt, rs, "h1", [e])
    assert len(pts) == 8 and [p["board_idx"] for p in pts] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert {p["boards"] for p in pts} == {(0, 1, 2), (1, 2, 3)}
    assert [p["reference"] for p in pts if p["board_idx"] == 0] == [False, True]
    assert [p["value_ps"] for p in pts if p["board_idx"] == 1] == [46.0, 41.0]
    assert [curves.low_pixels(p) for p in pts if p["reference"]] == [True, True, False, False]
    with pytest.raises(ValueError):          # a board without a reference combination
        combo_split.combo_points(bt[~(bt[tables.REFERENCE_COLUMN] & (bt["board_idx"] == 2))], rs, "h1", [e])
    # x: a lone run gets GROUP_MIN slots, three runs of one class one slot each
    ents = [dict(e, run=4, step=1.5e15, ref=True), dict(e, run=13), dict(e, run=14), dict(e)]
    lay = combo_split.layout(ents)
    g = combo_split.GROUP_MIN
    assert lay["width"] == pytest.approx(g + 3)
    assert (lay["x"][4], lay["x"][13]) == pytest.approx((g / 2, g + 0.5))
    assert [len(runs) for _, _, _, runs in lay["groups"]] == [1, 3]


def test_split_runs_an_accepted_merge_stands_in_for_its_class_runs(split_display):
    merges = {"f1": [(3, 4), (46, 47)], "h1": [(3, 4), (18, 19), (47, 48)]}
    split = combo_split.split_runs(split_display, (1.5e15, 2e15), 3.5e15, merges=merges)
    h1, f1 = split["h1"], split["f1"]
    # F1 46+47: both runs in the RFSel 1 class, one entry holding the class's preferred run 47
    assert [e["run"] for e in f1] == [9, 12, (46, 47), 13, 14, 18]
    assert [e["preferred"] for e in f1 if e["step"] == 3.5e15] == [True, False, False, True]
    # H1 47+48 likewise; 18+19 has run 19 off the list; 3+4 is no class's at 3.5e15 and the
    # reference stays the preferred run alone
    assert [e["run"] for e in h1] == [4, 12, 64, 67, 56, 33, (47, 48), 18]


def test_combo_points_of_a_merge_read_the_merged_table():
    bt = board_table([46, 47], base=40.0)
    bm = board_table([tables.merge_name("h1", (46, 47))], base=60.0)
    for t in (bt, bm):
        t["combo"] = t[tables.REFERENCE_COLUMN].map({True: "extra0-dut1-ref2", False: "dut1-ref2-trig3"})
    rs = runs_summary([(JUL, 46, "3.5e15", 480.0, 20, 1), (JUL, 47, "3.5e15", 480.0, 20, 1),
                       (JUL, 13, "3.5e15", 390.0, 20, 2)])
    e = dict(run=(46, 47), step=3.5e15, rfsel=1, os="20", ref=False, preferred=True)
    pts = combo_split.combo_points(bt, rs, "h1", [e], bm=bm)
    assert len(pts) == 8 and {p["run"] for p in pts} == {(46, 47)}
    assert [p["value_ps"] for p in pts if p["board_idx"] == 0] == [65.0, 60.0]
    assert {p["hv"] for p in pts} == {480.0}
    with pytest.raises(ValueError):          # a merge needs the merged table
        combo_split.combo_points(bt, rs, "h1", [e])
    with pytest.raises(ValueError):          # runs of one merge at different settings
        combo_split.combo_points(bt, rs, "h1", [dict(e, run=(13, 46))], bm=bm)
    # single runs and a merge side by side on one axis
    single = combo_split.combo_points(bt, rs, "h1", [dict(e, run=46)])
    lay = combo_split.layout([dict(e, run=46), e])
    xs = combo_split.combo_x(single + pts, lay["x"])
    assert xs[((46, 47), 0, "extra0-dut1-ref2")] > xs[(46, 0, "extra0-dut1-ref2")]


def test_bias_summary_in_part_keeps_the_whole_figures_colours():
    import matplotlib
    matplotlib.use("Agg")

    def pt(camp, fluence, i, hv):
        return dict(tel="h1", camp=camp, run=1, runs=[1], label="1", board_idx=i, chip=CHIPS[i],
                    value_ps=40.0, bar_ps=3.0, n_pixels=250, hv=hv, fluence=fluence, offset=20,
                    rfsel=2, rfsel_source="runs_summary", merged=False, band=None, y_ps=40.0,
                    yerr_ps=3.0)
    pts = {"h1": [pt("march", 1.5e15, i, 400.0) for i in range(4)]
           + [pt("july", 1.5e15, i, 540.0) for i in range(4)]}
    words = dict(tel_titles={"h1": "H1"}, xlabel="bias", ylim=(20.0, 80.0), tag="t", data="d",
                 footer="f")
    _, whole = curves.res_vs_bias_summary(pts, **words)
    _, part = curves.res_vs_bias_summary(pts, keep=lambda p: p["camp"] == "july", **words)
    assert [q["camp"] for q in part["points"]] == ["july"] * 4
    # July 1.5e15 is the later measurement at the step: teal, as beside March
    assert {q["color"] for q in part["points"]} == {TIMING_COLORS[0]}
    assert [q["color"] for q in part["points"]] == [q["color"] for q in whole["points"]
                                                    if q["camp"] == "july"]


COMBOS = {(0, 1, 2): "extra0-dut1-ref2", (0, 1, 3): "ref0-dut1-trig3",
          (0, 2, 3): "ref0-dut2-trig3", (1, 2, 3): "dut1-ref2-trig3"}


def combination_table(run=18, drop=()):
    """Every 3-board combination of H1 run `run` but those in `drop`: each board's row in each
    combination, value_ps = 40 + 10 x the board left out + the board index, pixel_spread_ps = 2 +
    the board left out, n_pixels = 200 + 10 x the board left out + the board index; the reference
    combination is boards 0-1-2 for boards 0 to 2 and 1-2-3 for board 3."""
    rows = []
    for boards, combo in COMBOS.items():
        if boards in drop:
            continue
        out = ({0, 1, 2, 3} - set(boards)).pop()
        for b in boards:
            rows.append(dict(campaign=JUL, telescope="h1", run=str(run), combo=combo, board_idx=b,
                             board_chip="PT_" + CHIPS[b], variant="top", floor="300",
                             value_ps=40.0 + 10 * out + b, pixel_spread_ps=2.0 + out,
                             n_pixels=200 + 10 * out + b,
                             **{tables.REFERENCE_COLUMN:
                                boards == ((0, 1, 2) if b < 3 else (1, 2, 3))}))
    return pd.DataFrame(rows)


def test_partner_excluded_takes_the_combination_without_the_hv_partner():
    t = combination_table()
    out, missing = combo_check.partner_excluded(t)
    assert missing == [] and out["board_idx"].tolist() == [0, 1, 2, 3]
    # partners 0-1 and 2-3: board 0 from 0-2-3, 1 from 1-2-3, 2 from 0-1-2, 3 from 0-1-3
    assert [combo_split.combo_boards(c) for c in out["combo"]] == [(0, 2, 3), (1, 2, 3),
                                                                   (0, 1, 2), (0, 1, 3)]
    assert out["value_ps"].tolist() == [50.0, 41.0, 72.0, 63.0]
    assert out["chip"].tolist() == ["PT_" + c for c in CHIPS]
    reference, none = combo_check.reference_rows(t)
    assert none == [] and reference["value_ps"].tolist() == [70.0, 71.0, 72.0, 43.0]


def test_midpoint_is_the_rms_of_the_present_mean_and_the_absent_value():
    out, missing = combo_check.midpoint(combination_table())
    assert missing == []
    r = out.set_index("board_idx")
    # board 0: present 0-1-2 (70 ps, spread 5) and 0-1-3 (60, 4), absent 0-2-3 (50, 3)
    w_present = math.sqrt((70.0 ** 2 + 60.0 ** 2) / 2)
    assert r.loc[0, "w_present"] == pytest.approx(w_present) and r.loc[0, "w_absent"] == 50.0
    assert r.loc[0, "value_ps"] == pytest.approx(math.sqrt((w_present ** 2 + 50.0 ** 2) / 2))
    sp_present = math.sqrt((5.0 ** 2 + 4.0 ** 2) / 2)
    assert r.loc[0, "bar_ps"] == pytest.approx(math.sqrt((sp_present ** 2 + 3.0 ** 2) / 2))
    assert r.loc[0, "n_pixels"] == 210                  # the fewest of the three
    # board 3: present 0-2-3 (53) and 1-2-3 (43), absent 0-1-3 (63)
    assert r.loc[3, "value_ps"] == pytest.approx(
        math.sqrt(((53.0 ** 2 + 43.0 ** 2) / 2 + 63.0 ** 2) / 2))
    assert [combo_split.combo_boards(c) for c in r.loc[3, "combos_present"]] == [(0, 2, 3),
                                                                               (1, 2, 3)]


def test_a_missing_combination_is_reported_not_guessed():
    t = combination_table(drop=[(0, 2, 3)])
    out, missing = combo_check.partner_excluded(t)
    assert out["board_idx"].tolist() == [1, 2, 3]
    assert missing == [dict(campaign=JUL, telescope="h1", run="18", board_idx=0,
                            row="partner_excluded", combination="boards 0-2-3")]
    mid, lacking = combo_check.midpoint(t)
    assert mid["board_idx"].tolist() == [1]
    assert [(m["board_idx"], m["combination"]) for m in lacking] == [
        (0, "boards 0-2-3"), (2, "boards 0-2-3"), (3, "boards 0-2-3")]
    # a run with combination 0-1-2 only (board 3 took no data)
    one, gone = combo_check.partner_excluded(combination_table(drop=[(0, 1, 3), (0, 2, 3),
                                                                     (1, 2, 3)]))
    assert one["board_idx"].tolist() == [2]
    assert [(m["board_idx"], m["combination"]) for m in gone] == [(0, "boards 0-2-3"),
                                                                  (1, "boards 1-2-3")]
    # the figure drops the point the row lacks and names it
    pts = [dict(camp="july", tel="h1", run=18, board_idx=b, chip=CHIPS[b], value_ps=1.0,
                bar_ps=1.0, n_pixels=1, band=(0.0, 2.0)) for b in range(4)]
    drawn, lost = combo_check.row_points(pts, out)
    assert [p["board_idx"] for p in drawn] == [1, 2, 3] and lost == [("h1", 18, 0)]
    assert [p["value_ps"] for p in drawn] == [41.0, 72.0, 63.0]
    assert {p["band"] for p in drawn} == {None}


def test_hv_cycle_between_runs_by_board():
    t0 = pd.Timestamp("2026-07-10 10:00", tz="UTC")

    def m(k):
        return t0 + pd.Timedelta(minutes=k)
    windows = {5: (m(0), m(30)), 6: (m(60), m(90)), 7: (m(120), m(150))}
    rows = []
    # board 0 at 50 V of 300 V (under 20 %) between runs 5 and 6, at 10 V in run 6 itself (the
    # end of run 6 is not in the next gap); board 1 set to 0 V; board 2 without samples
    for k, v in ((40, 50.0), (60, 300.0), (90, 10.0), (100, 300.0), (120, 280.0)):
        rows += [dict(tb=m(k), channel=0, v=v, vset=300.0), dict(tb=m(k), channel=1, v=0.0,
                                                               vset=0.0)]
    status = {(r, b): "on" for r in windows for b in range(3)}
    status.update({(r, 3): "hv_off_by_design" for r in windows})
    del status[(7, 1)]
    hv = combo_check.hv_cycles(windows, pd.DataFrame(rows), status)
    assert hv[5] == dict(prev_run=None, boards={0: "n/a", 1: "n/a", 2: "n/a", 3: "n/a"})
    assert hv[6] == dict(prev_run=5, boards={0: True, 1: False, 2: "no data", 3: "off"})
    assert hv[7] == dict(prev_run=6, boards={0: False, 1: "off", 2: "no data", 3: "off"})


def test_autocal_groups_need_every_chip_on_the_same_scan():
    a, b = CHIPS[0], CHIPS[1]
    ids = {3: {a: "s1", b: "s1"}, 4: {a: "s1", b: "s1"}, 5: {a: "s2", b: "s1"}, 6: {},
           7: {a: "s3", b: "s3"}, 8: {a: "s3", b: "s3"}, 9: {a: "s3", b: "s3"}}
    groups, partial, no_id = combo_check.autocal_groups(ids, [3, 4, 5, 6, 7, 8, 9], [a, b])
    assert groups == [[3, 4], [7, 8, 9]]
    assert partial == {5: [b]} and no_id == [6]
