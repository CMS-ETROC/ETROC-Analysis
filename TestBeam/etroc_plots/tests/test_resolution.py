"""Tests of the resolution curves helpers: the merge rule, the good-run list with merges, the
points of single and merged runs, the standard operating point, the stop clock with its bounds,
the hours since the first run at a step and the colour rule.

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
from etroc_plots.resolution import curves, tables               # noqa: E402

JUL, MAR = campaign.TABLE_CAMPAIGN["july"], campaign.TABLE_CAMPAIGN["march"]
CHIPS = campaign.TELESCOPE_CHIPS["h1"]


def board_table(runs, camp=JUL):
    """Anointed rows for every board of each run (a run number or a merge name), value_ps =
    40 + board index, plus one non-anointed row per board."""
    rows = []
    for run in runs:
        for i, chip in enumerate(CHIPS):
            for anointed, combo in ((True, "a"), (False, "b")):
                rows.append(dict(campaign=camp, telescope="h1", run=str(run), combo=combo,
                                 board_idx=i, board_chip="PT_" + chip, variant="top",
                                 floor="300", value_ps=40.0 + i + (0 if anointed else 5),
                                 pixel_spread_ps=3.0, n_pixels=250 + i, anointed=anointed))
    return pd.DataFrame(rows)


def runs_summary(rows):
    """rows: (campaign, run, fluence text, hv, offset, rfsel or None) for H1."""
    out = []
    for camp, run, irrad, hv, offset, rfsel in rows:
        r = dict(campaign=camp, telescope=campaign.TABLE_TEL["h1"], run=run)
        for i, chip in enumerate(CHIPS):
            r.update({"board%d_chip" % i: "PT_" + chip, "board%d_hv" % i: hv,
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
