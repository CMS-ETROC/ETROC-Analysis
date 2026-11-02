"""What plot_lots.py makes of the wafers of a lot (lot_compare.py,
lot_compare_plots.py), on synthetic wafers: quick tests before UBM and
full scans after on one wafer, full scans in both stages on another."""
import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

HAVE_PLOTS = all(importlib.util.find_spec(m) for m in ("numpy", "pandas", "pyarrow", "matplotlib"))
if HAVE_PLOTS:
    import pandas as pd
    from lot_compare import chronology, edge_ring, lot_name, lot_names, profile, short_rails, transition_class, wilson
    from lot_compare_plots import _class_handles
    from plot_lots import main, parse_lot
    from tests.wafer_results import run_power, summary, write_map, write_run

WAFER_MAP = {1: (0, 1), 2: (0, 2), 3: (1, 0), 4: (1, 1), 5: (1, 2), 6: (2, 1)}
QINJ_PHASES = {"power_on": "2026-09-22 15:00:00.000", "i2c_start": "2026-09-22 15:00:02.000",
               "high_power": "2026-09-22 15:00:03.000", "qinj_start": "2026-09-22 15:00:20.000"}


def rails(analog=None, digital=None):
    """A short_check with the analog and/or digital rail at its limit."""
    check = {}
    for rail, at in (("analog", analog), ("digital", digital)):
        check[rail] = ({"voltage": 0.6, "current": at, "abort_above": at - 0.002, "ok": False, "fault": "over_current"}
                       if at else {"voltage": 1.3, "current": 0.3, "abort_above": 0.698, "ok": True})
    return check


@unittest.skipUnless(HAVE_PLOTS, "needs numpy, pandas, pyarrow and matplotlib (the wafer-daq venv)")
class LotHelpersTest(unittest.TestCase):
    def test_short_rails_reads_the_rails_off_the_grade_detail(self):
        self.assertEqual(short_rails("analog_650mA-digital_275mA_high"), ["analog", "digital"])
        self.assertEqual(short_rails("ws_analog_90mA"), ["ws_analog"])
        self.assertEqual(short_rails("over-current"), [])

    def test_transition_classes(self):
        self.assertEqual(transition_class("PASSED", "POWER_SHORT", "digital_300mA-analog_700mA"), "new analog short")
        self.assertEqual(transition_class("PASSED", "POWER_SHORT", "digital_300mA"), "new digital short")
        self.assertEqual(transition_class("PASSED", "POWER_SHORT", "ws_analog_90mA"), "new short, other or unknown rail")
        self.assertEqual(transition_class("PASSED", "POWER_SHORT", "over-current"), "new short, other or unknown rail")
        self.assertEqual(transition_class("PASSED", "BL_NW_ZERO"), "new BL/NW = 0")
        self.assertEqual(transition_class("PASSED", "I2C_NACK"), "new I2C NACK")
        self.assertEqual(transition_class("PASSED", "I2C_PIXELS"), "new I2C pixel or register failure")
        self.assertEqual(transition_class("PASSED", "NO_LINK_OR_DATA"), "new other failure")
        self.assertEqual(transition_class("I2C_NACK", "PASSED"), "recovered")
        self.assertEqual(transition_class("POWER_SHORT", "POWER_SHORT"), "failed both")
        self.assertEqual(transition_class("PASSED", "CONTACT_FAILURE"), "untested in a stage")

    def test_edge_ring_and_wilson(self):
        self.assertEqual(edge_ring(WAFER_MAP), {1, 2, 3, 5, 6})
        lo, hi = wilson(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.0909, places=4)

    def test_the_wilson_interval_holds_the_fraction_at_0_and_n(self):
        for n in range(1, 401):
            self.assertEqual(wilson(0, n)[0], 0.0, n)
            self.assertEqual(wilson(n, n)[1], 1.0, n)

    def lot(self, pre, post):
        """Lot rows of three dies: tested before UBM at the times pre (one
        die NOT_TESTED), after at post."""
        return pd.DataFrame({"batch": "BatchID_X_Name_N62C72", "run_pre": 1, "run_post": 1,
                             "grade_pre": ["PASSED", "PASSED", "NOT_TESTED"], "grade_post": "PASSED",
                             "start_pre": [*pre, "2027-01-01 00:00:00"], "start_post": post})

    def test_a_lot_is_named_by_its_batch_and_the_months_it_was_tested(self):
        feb = self.lot(["2025-09-30 10:00:00", "2025-10-02 09:00:00"], "2026-02-10 12:00:00")
        self.assertEqual(lot_name(feb), "N62C72, pre-UBM test 09/2025-10/2025, post-UBM test 02/2026")
        apr = self.lot(["2025-10-20 10:00:00", "2025-10-21 10:00:00"], "2026-04-02 12:00:00")
        late = self.lot(["2026-02-03 10:00:00", "2026-02-04 10:00:00"], "2026-09-25 12:00:00")
        lots = {"late": late, "apr": apr, "feb": feb, "feb again": feb.copy()}
        self.assertEqual(sorted(lots, key=lambda label: chronology(lots[label])), ["feb", "feb again", "apr", "late"])
        names = lot_names(lots)
        self.assertEqual(names["apr"], "N62C72, pre-UBM test 10/2025, post-UBM test 04/2026")
        self.assertEqual(names["feb again"], "N62C72, pre-UBM test 09/2025-10/2025, post-UBM test 02/2026 (feb again)")

    def test_the_legends_leave_out_empty_classes_and_name_the_other_failures(self):
        lot = pd.DataFrame({"invalid": [False, False, False, True],
                            "transition": ["passed both", "new other failure", "new I2C NACK", "new analog short"],
                            "grade_post": ["PASSED", "NO_LINK_OR_DATA", "I2C_NACK", "POWER_SHORT"]})
        self.assertEqual([h.get_label() for h in _class_handles(lot)],
                         ["passed both: 1", "new I2C NACK: 1", "new other failure (NO_LINK_OR_DATA): 1"])

    def test_parse_lot(self):
        self.assertEqual(parse_lot("N62C72"), ("N62C72", "N62C72", None))
        self.assertEqual(parse_lot("early UBM=N62C72:02G4, 03F5"), ("early UBM", "N62C72", ["02G4", "03F5"]))


@unittest.skipUnless(HAVE_PLOTS, "needs numpy, pandas, pyarrow and matplotlib (the wafer-daq venv)")
class PlotLotsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.out = self.root / "lots"
        write_map(self.root / "map.csv", WAFER_MAP)

    def die(self, batch, wafer, stage, die, full=False, qinj=None, high=0.45, **extra):
        row, col = WAFER_MAP[die]
        folder = self.root / f"BatchID_X_Name_{batch}" / f"WaferID_X_Name_{wafer}" / stage
        qinj = full if qinj is None else qinj
        extra.setdefault("calibration", {"zero_pixels": [], "n_pixels": 256 if full else 9})
        write_run(folder, die, 1, summary(die, row, col, wafer_stage=stage, fullscan=full, qinj=qinj, **extra),
                  power=run_power(on=0.31 if stage == "pre_ubm" else 0.30, high=high))

    def usual_lots(self):
        # W1: quick tests before, full scans after
        for die in WAFER_MAP:
            self.die("B", "W1", "pre_ubm", die)
        self.die("B", "W1", "post_ubm", 1, full=True, calibration={"zero_pixels": [[0, 0]], "n_pixels": 256})
        self.die("B", "W1", "post_ubm", 2, full=True, calibration={"zero_pixels": [[2, 2]], "n_pixels": 256})
        self.die("B", "W1", "post_ubm", 3, full=True, status="failed", phases=dict(QINJ_PHASES),
                 qinj_check={"ok": False, "events": 0, "reason": "no data"})
        self.die("B", "W1", "post_ubm", 4, full=True, status="over_current", short_check=rails(analog=0.70))
        self.die("B", "W1", "post_ubm", 5, full=True, status="over_current", short_check=rails(digital=0.30))
        self.die("B", "W1", "post_ubm", 6, full=True)
        # W2: full scans in both stages
        bad_i2c = {"0x60": {"pixel_id": {"passed": False, "failed_pixels": [[3, 3]]},
                            "peripheral": {"passed": True, "failed_registers": []}}}
        for die in WAFER_MAP:
            pre = {"status": "over_current", "short_check": rails(analog=0.70)} if die == 2 else \
                  {"i2c": bad_i2c} if die == 3 else {}
            self.die("B", "W2", "pre_ubm", die, full=True, **pre)
            post = {"calibration": {"zero_pixels": [[0, 0]], "n_pixels": 256}} if die == 1 else \
                   {"status": "over_current", "short_check": rails(analog=0.70)} if die == 2 else {}
            self.die("B", "W2", "post_ubm", die, full=True, **post)
        # W9: before UBM only
        self.die("B", "W9", "pre_ubm", 1)
        # C / W3: quick tests in both stages, one new short
        for die in WAFER_MAP:
            self.die("C", "W3", "pre_ubm", die)
            self.die("C", "W3", "post_ubm", die, **({"status": "over_current", "short_check": rails(analog=0.70)}
                                                    if die == 5 else {}))

    def run_lots(self, *lots, extra=()):
        argv = ["--path", str(self.root), "--waferMap", str(self.root / "map.csv"), "--out", str(self.out)]
        for lot in lots:
            argv += ["--lot", lot]
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            rc = main(argv + list(extra))
        return rc, printed.getvalue()

    def table(self):
        return pd.read_csv(self.out / "lot_dies.csv").set_index(["wafer", "die"])

    def test_a_full_scan_is_graded_as_a_quick_test_where_the_other_stage_was_one(self):
        self.usual_lots()
        rc, printed = self.run_lots("B", extra=["--tables-only"])
        self.assertEqual(rc, 0, printed)
        t = self.table()
        self.assertEqual(t.loc[("W1", 1), ["grade_post", "measured_post"]].tolist(), ["PASSED", "BL_NW_ZERO"])
        self.assertEqual(t.loc[("W1", 2), "grade_post"], "BL_NW_ZERO")   # its zero pixel is a quick one
        self.assertEqual(t.loc[("W1", 3), ["grade_post", "measured_post"]].tolist(), ["PASSED", "NO_LINK_OR_DATA"])
        self.assertEqual(t.loc[("W2", 1), "grade_post"], "BL_NW_ZERO")   # full scans in both stages: as measured
        self.assertTrue(pd.isna(t.loc[("W2", 1), "measured_post"]))
        self.assertIn("W1 post_ubm die 1: BL_NW_ZERO as measured, PASSED with the test both stages made "
                      "(9 pixels, no QInj)", printed)
        self.assertIn("B: WaferID_X_Name_W9 left out: no post_ubm runs", printed)

    def test_a_qinj_run_on_the_quick_pixels_keeps_the_qinj_of_a_full_scan(self):
        for die in WAFER_MAP:
            self.die("D", "W5", "pre_ubm", die, qinj=True)
        self.die("D", "W5", "post_ubm", 1, full=True, calibration={"zero_pixels": [[0, 0]], "n_pixels": 256})
        self.die("D", "W5", "post_ubm", 2, full=True, status="failed", phases=dict(QINJ_PHASES),
                 qinj_check={"ok": False, "events": 0, "reason": "no data"})
        rc, printed = self.run_lots("D", extra=["--tables-only"])
        self.assertEqual(rc, 0, printed)
        t = self.table()
        self.assertEqual(t.loc[("W5", 1), "grade_post"], "PASSED")
        self.assertEqual(t.loc[("W5", 2), "grade_post"], "NO_LINK_OR_DATA")
        self.assertIn("(9 pixels, QInj)", printed)

    def test_pixels_the_other_stage_never_calibrated_do_not_count(self):
        # before UBM: the February quick test, 8 pixels, and none on a die without baselines
        self.die("E", "W6", "pre_ubm", 1, calibration={"zero_pixels": [], "n_pixels": 8})
        self.die("E", "W6", "pre_ubm", 2, calibration={})
        self.die("E", "W6", "post_ubm", 1, calibration={"zero_pixels": [[15, 15]], "n_pixels": 9})
        self.die("E", "W6", "post_ubm", 2, calibration={"zero_pixels": [[2, 2]], "n_pixels": 9})
        rc, printed = self.run_lots("E", extra=["--tables-only"])
        self.assertEqual(rc, 0, printed)
        t = self.table()
        for die in (1, 2):
            self.assertEqual(t.loc[("W6", die), ["grade_post", "measured_post"]].tolist(), ["PASSED", "BL_NW_ZERO"])
        self.assertIn("W6 post_ubm die 1: BL_NW_ZERO as measured, PASSED with the test both stages made "
                      "(8 pixels, no QInj)", printed)
        self.assertIn("W6 post_ubm die 2: BL_NW_ZERO as measured, PASSED with the test both stages made "
                      "(no pixels, no QInj)", printed)

    def test_a_zero_pixel_carried_over_counts_in_both_stages(self):
        # the February import: a zero the full scan before UBM read, carried over to the 8-pixel test after
        self.die("F", "W8", "pre_ubm", 1, full=True, calibration={"zero_pixels": [[15, 15]], "n_pixels": 256})
        self.die("F", "W8", "post_ubm", 1, calibration={"zero_pixels": [[15, 15]], "n_pixels": 8,
                                                        "carried_over": [[15, 15]]})
        rc, printed = self.run_lots("F", extra=["--tables-only"])
        self.assertEqual(rc, 0, printed)
        t = self.table()
        self.assertEqual(t.loc[("W8", 1), ["grade_pre", "grade_post", "transition"]].tolist(),
                         ["BL_NW_ZERO", "BL_NW_ZERO", "failed both"])
        self.assertTrue(t.loc[("W8", 1), ["measured_pre", "measured_post"]].isna().all())   # neither regraded

    def test_a_run_without_the_high_power_check_is_checked_on_its_currents(self):
        limits = {"analog": {"voltage": 1.3, "current": 0.45, "abort_above": 0.698, "ok": True},
                  "digital": {"voltage": 1.3, "current": 0.45, "abort_above": 0.9, "ok": True}}
        phases = {"power_on": "2026-09-22 15:00:00.000", "high_power": "2026-09-22 15:00:03.000"}
        for die, high in ((1, 0.45), (2, 0.71)):
            self.die("F", "W7", "pre_ubm", die, high=high, phases=dict(phases))
            self.die("F", "W7", "post_ubm", die, phases=dict(phases), short_check_high=limits)
        rc, printed = self.run_lots("F", extra=["--tables-only"])
        self.assertEqual(rc, 0, printed)
        t = self.table()
        self.assertEqual(t.loc[("W7", 1), "grade_pre"], "PASSED")
        self.assertEqual(t.loc[("W7", 2), ["grade_pre", "detail_pre", "measured_pre"]].tolist(),
                         ["POWER_SHORT", "analog_710mA_high", "PASSED"])
        self.assertEqual(t.loc[("W7", 2), "transition"], "recovered")

    def test_a_stage_that_did_not_test_the_die_regrades_nothing(self):
        self.die("G", "W8", "pre_ubm", 1, status="contact_failure", error="no contact")
        self.die("G", "W8", "post_ubm", 1, full=True, status="failed", phases=dict(QINJ_PHASES),
                 qinj_check={"ok": False, "events": 0, "reason": "no data"})
        rc, printed = self.run_lots("G", extra=["--tables-only"])
        self.assertEqual(rc, 0, printed)
        t = self.table()
        self.assertEqual(t.loc[("W8", 1), "grade_post"], "NO_LINK_OR_DATA")
        self.assertTrue(pd.isna(t.loc[("W8", 1), "measured_post"]))
        self.assertEqual(t.loc[("W8", 1), "transition"], "untested in a stage")

    def test_a_run_stuck_in_a_qinj_stage_the_other_did_not_make_counts_as_completed(self):
        self.die("H", "W10", "pre_ubm", 1)
        self.die("H", "W10", "pre_ubm", 2)
        self.die("H", "W10", "post_ubm", 1, full=True, status="stuck", error="past its time limit",
                 phases=dict(QINJ_PHASES))
        self.die("H", "W10", "post_ubm", 2, full=True, status="stuck", error="past its time limit",
                 phases={k: v for k, v in QINJ_PHASES.items() if k != "qinj_start"})
        rc, printed = self.run_lots("H", extra=["--tables-only"])
        self.assertEqual(rc, 0, printed)
        t = self.table()
        self.assertEqual(t.loc[("W10", 1), ["grade_post", "measured_post"]].tolist(), ["PASSED", "TEST_FAILURE"])
        self.assertEqual(t.loc[("W10", 2), "grade_post"], "TEST_FAILURE")

    def test_each_die_gets_its_transition(self):
        self.usual_lots()
        rc, printed = self.run_lots("B", extra=["--tables-only"])
        self.assertEqual(rc, 0, printed)
        t = self.table()["transition"]
        self.assertEqual(t.loc["W1"].to_dict(), {1: "passed both", 2: "new BL/NW = 0", 3: "passed both",
                                                 4: "new analog short", 5: "new digital short", 6: "passed both"})
        self.assertEqual(t.loc["W2"].to_dict(), {1: "new BL/NW = 0", 2: "failed both", 3: "recovered",
                                                 4: "passed both", 5: "passed both", 6: "passed both"})

    def test_the_new_analog_shorts_by_row_count_the_dies_at_risk(self):
        self.usual_lots()
        self.run_lots("B", extra=["--tables-only"])
        p = profile(pd.read_csv(self.out / "lot_dies.csv"), "die_row", ("new analog short",))
        # at risk: W1 all 6, W2 dies 1, 4, 5, 6 (2 failed before, 3 failed before)
        self.assertEqual(p["n"].to_dict(), {0: 3, 1: 5, 2: 2})
        self.assertEqual(p["k"].to_dict(), {0: 0, 1: 1, 2: 0})

    def test_a_listed_wafer_that_is_not_there_stops_it(self):
        self.usual_lots()
        rc, printed = self.run_lots("B:W1,W7", extra=["--tables-only"])
        self.assertEqual(rc, 2)
        self.assertIn("no wafer folder W7 of batch B", printed)

    def test_lots_sharing_a_file_name_or_a_wafer_stop_it(self):
        self.usual_lots()
        rc, printed = self.run_lots("B", "B_=B:W1", extra=["--tables-only"])
        self.assertEqual(rc, 2)
        self.assertIn("two lots share a label, or its file-name form", printed)
        rc, printed = self.run_lots("B", "again=B:W1", extra=["--tables-only"])
        self.assertEqual(rc, 2)
        self.assertIn("again: WaferID_X_Name_W1 is in B already", printed)

    def test_the_figures_of_each_lot_and_the_lot_comparison(self):
        self.usual_lots()
        rc, printed = self.run_lots("B", "late=C:W3")
        self.assertEqual(rc, 0, printed)
        figures = {p.name for p in self.out.glob("*.png")}
        expected = {f"{prefix}_{name}.png" for prefix in ("B", "late")
                    for name in ("yield", "transitions", "edges", "currents", "shorts")}
        self.assertEqual(figures, expected | {"lots.png"})
        self.assertEqual(sorted(pd.read_csv(self.out / "lot_dies.csv")["lot"].unique()), ["B", "late"])


if __name__ == "__main__":
    unittest.main()
