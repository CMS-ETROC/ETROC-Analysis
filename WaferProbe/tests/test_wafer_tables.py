"""The wafer tables (wafer_tables.py) on synthetic run folders laid out
as master_run_script.run_die writes them (tests/wafer_results.py)."""
import importlib.util
import json
import math
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

HAVE_TABLES = all(importlib.util.find_spec(m) for m in ("numpy", "pandas", "pyarrow"))
if HAVE_TABLES:
    import numpy as np
    import pandas as pd

    from wafer_tables import (CURRENT_NOTE_FACTOR, NOTE_MIN_DIES, NOTE_PIXEL_BL, NOTE_PIXEL_NW, bl_nw_notes,
                              collect, current_notes, grade_counts,
                              die_record, lowest_efficiency, offset_trend, phase_values, pick_run,
                              qinj_files, qinj_pixel_stats, read_nem_hits, unchecked_die_means)
    from tests.wafer_results import QUICK_PIXELS, baseline, event, power, run_power, summary, write_run

needs_tables = unittest.skipUnless(HAVE_TABLES, "needs numpy, pandas and pyarrow (the wafer-daq venv)")


class TempWafer(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.wafer = Path(tmp.name) / "B" / "W"


@needs_tables
class RunChoiceTest(TempWafer):
    def test_the_newest_run_with_a_summary_stands_for_the_die(self):
        write_run(self.wafer, 3, 1, summary(3, 0, 7))
        write_run(self.wafer, 3, 2, summary(3, 0, 7, attempt=2))
        write_run(self.wafer, 3, 3)  # killed before it wrote summary.json
        run_dir, s, passed_over = pick_run(self.wafer / "die003")
        self.assertEqual((run_dir.name, s["attempt"], passed_over),
                         ("run_02_noEfuse", 2, ["run_03_noEfuse (no summary.json)"]))

    def test_a_run_aborted_with_ctrl_c_is_passed_over(self):
        write_run(self.wafer, 3, 1, summary(3, 0, 7))
        write_run(self.wafer, 3, 2, summary(3, 0, 7, status="user_abort"))
        run_dir, _, passed_over = pick_run(self.wafer / "die003")
        self.assertEqual((run_dir.name, passed_over), ("run_01_noEfuse", ["run_02_noEfuse (aborted)"]))

    def test_before_dates_a_run_that_stopped_before_power_on(self):
        write_run(self.wafer, 3, 1, summary(3, 0, 7, power_on="2026-09-22 14:00:00.000"))
        write_run(self.wafer, 3, 2, summary(3, 0, 7, status="failed",
                                            phases={"power_off": "2026-09-22 15:00:00.000"}))
        run_dir, _, _ = pick_run(self.wafer / "die003", before=datetime(2026, 9, 22, 16))
        self.assertEqual(run_dir.name, "run_02_noEfuse")

    def test_run_numbers_sort_as_numbers(self):
        write_run(self.wafer, 3, 99, summary(3, 0, 7))
        write_run(self.wafer, 3, 100, summary(3, 0, 7))
        self.assertEqual(pick_run(self.wafer / "die003")[0].name, "run_100_noEfuse")

    def test_before_takes_the_newest_run_powered_before_it(self):
        for run, t in ((1, "14:00"), (2, "15:00"), (3, "17:00")):
            write_run(self.wafer, 3, run, summary(3, 0, 7, power_on=f"2026-09-22 {t}:00.000"))
        run_dir, _, passed_over = pick_run(self.wafer / "die003", before=datetime(2026, 9, 22, 16))
        self.assertEqual((run_dir.name, passed_over), ("run_02_noEfuse", []))


@needs_tables
class PhaseValuesTest(unittest.TestCase):
    def test_the_first_high_power_sample_is_dropped(self):
        p = power([("analog", "power_on", 0.31), ("analog", "high_power", 0.30),
                   ("analog", "high_power", 0.45), ("analog", "high_power", 0.46)])
        values = phase_values(p)
        self.assertAlmostEqual(values[("analog", "high_power")][0], 0.455)
        self.assertAlmostEqual(values[("analog", "power_on")][0], 0.31)

    def test_the_first_sample_of_every_phase_is_dropped(self):
        # the first power-on sweep catches vref still ramping (N60R91 die 5: 0.33 V, then 1.00 V)
        p = power([("vref", "power_on", 0.0, 0.33), ("vref", "power_on", 0.0, 1.00),
                   ("vref", "qinj_start", 0.0, 0.90), ("vref", "qinj_start", 0.0, 1.00),
                   ("vref", "qinj_start", 0.0, 1.00)])
        values = phase_values(p)
        self.assertAlmostEqual(values[("vref", "power_on")][1], 1.00)
        self.assertAlmostEqual(values[("vref", "qinj_start")][1], 1.00)

    def test_a_lone_high_power_sample_is_kept(self):
        values = phase_values(power([("analog", "high_power", 0.30)]))
        self.assertAlmostEqual(values[("analog", "high_power")][0], 0.30)


@needs_tables
class DieRecordTest(TempWafer):
    def test_a_retried_die_with_its_rails_and_offsets(self):
        s = summary(3, 0, 7, attempt=2, prober={"velox": {"x_um": 100.0, "y_um": 200.0},
                                                "chuck": {"x_um": 102.5, "y_um": 190.0, "z_contact_um": 50.0}})
        run_dir = write_run(self.wafer, 3, 2, s)
        rec = die_record(3, (0, 7), run_dir, s, power=run_power(on=0.31, high=0.45))
        self.assertEqual((rec["grade"], rec["map_text"], rec["run"]), ("PASSED", "PASSED_retry", "run_02_noEfuse"))
        self.assertEqual((rec["dx_um"], rec["dy_um"]), (2.5, -10.0))
        self.assertAlmostEqual(rec["analog_I_on"], 0.31)
        self.assertAlmostEqual(rec["analog_I_high"], 0.45)
        self.assertAlmostEqual(rec["ws_analog_I_high"], 0.045)
        self.assertEqual((rec["analog_I_abort_above"], rec["analog_I_abort_below"]), (0.54, 0.1))

    def test_without_a_power_log_the_power_on_check_reading_stands(self):
        s = summary(3, 0, 7)
        rec = die_record(3, (0, 7), write_run(self.wafer, 3, 1, s), s)
        self.assertEqual((rec["analog_I_on"], rec["analog_I_high"]), (0.3, None))

    def test_zero_readings_are_listed_apart_from_the_statistics(self):
        s = summary(3, 0, 7)
        bl = baseline([(0, 0), (0, 1), (1, 1)], zero=[(0, 1)])
        rec = die_record(3, (0, 7), write_run(self.wafer, 3, 1, s), s, baseline=bl)
        self.assertEqual((rec["n_pixels"], rec["n_zero_pixels"], json.loads(rec["zero_pixels"])), (3, 1, [[0, 1]]))
        self.assertEqual((rec["bl_mean"], rec["nw_mean"]), (401.0, 7.0))

    def test_a_failed_qinj_behind_a_zero_pixel_is_kept_beside_the_grade(self):
        # 09A0 post-UBM die 45: a zero pixel, then no QInj data at all
        failed = {"status": "failed", "error": "QInj data: only file_0.nem, and the first file is not checked",
                  "phases": {"power_on": "2026-09-25 17:47:20.000", "qinj_start": "2026-09-25 17:47:29.000"},
                  "qinj_check": {"ok": False, "events": 0, "bad": 0},
                  "calibration": {"n_pixels": 2, "zero_pixels": [[15, 14]]}}
        for name, extra, also in (
                ("no QInj data", failed, "NO_LINK_OR_DATA (no data)"),
                ("events off the pattern", {**failed, "qinj_check": {"ok": False, "events": 9, "bad": 3}},
                 "NO_LINK_OR_DATA (off pattern)"),
                ("trailer chip ID alone", {**failed, "qinj_check": {"ok": False, "events": 9, "bad": 0,
                                                                   "chip_id_bad": 9}}, "EFUSE_TRAILER_FAIL"),
                ("QInj OK", {"phases": failed["phases"], "calibration": failed["calibration"],
                             "qinj_check": {"ok": True, "events": 9, "bad": 0}}, ""),
                ("stopped before QInj", {**failed, "phases": {"power_on": "2026-09-25 17:47:20.000",
                                                              "i2c_start": "2026-09-25 17:47:22.000"}}, "")):
            with self.subTest(name):
                s = summary(45, 5, 12, **extra)
                bl = baseline([(15, 13), (15, 14)], zero=[(15, 14)])
                rec = die_record(45, (5, 12), write_run(self.wafer / name, 45, 1, s), s, baseline=bl)
                self.assertEqual((rec["grade"], rec["qinj_also_failed"]), ("BL_NW_ZERO", also))

    def test_the_other_findings_before_qinj_keep_a_failed_qinj_beside_them_too(self):
        failed = {"status": "failed", "error": "QInj data: only file_0.nem",
                  "phases": {"power_on": "2026-09-25 17:47:20.000", "qinj_start": "2026-09-25 17:47:29.000"},
                  "qinj_check": {"ok": False, "events": 0, "bad": 0},
                  "calibration": {"n_pixels": 2, "zero_pixels": []}}
        for name, extra in (
                ("I2C_PIXELS", {"i2c": {"0x60": {"pixel_id": {"passed": False, "failed_pixels": [[7, 3]]}}}}),
                ("EFUSE_FAIL", {"efuse": {"0x60": {"verified": False, "reason": "reads 0x0"}}})):
            with self.subTest(name):
                s = summary(45, 5, 12, **failed, **extra)
                rec = die_record(45, (5, 12), write_run(self.wafer / name, 45, 1, s), s,
                                 baseline=baseline([(15, 13), (15, 14)]))
                self.assertEqual((rec["grade"], rec["qinj_also_failed"]), (name, "NO_LINK_OR_DATA (no data)"))

    def test_a_die_graded_by_its_qinj_failure_has_nothing_beside_it(self):
        s = summary(45, 5, 12, status="failed", error="QInj data: only file_0.nem",
                    phases={"power_on": "2026-09-25 17:47:20.000", "qinj_start": "2026-09-25 17:47:29.000"})
        rec = die_record(45, (5, 12), write_run(self.wafer, 45, 1, s), s, baseline=baseline([(15, 14)]))
        self.assertEqual((rec["grade"], rec["qinj_also_failed"]), ("NO_LINK_OR_DATA", ""))

    def test_a_die_without_a_run_is_not_tested(self):
        rec = die_record(5, (1, 2))
        self.assertEqual((rec["grade"], rec["die_row"], rec["die_col"]), ("NOT_TESTED", 1, 2))
        self.assertNotIn("run", rec)


@needs_tables
class QInjTest(TempWafer):
    def test_events_flagged_trailers_and_hit_fields(self):
        pixels = QUICK_PIXELS[:2]
        run_dir = write_run(self.wafer, 3, 1, nem={"qinj_run2": [
            event(pixels) + event(pixels) + event(pixels, status=8, junk=2), event(pixels)]})
        events, flagged, hits = read_nem_hits(qinj_files(run_dir))
        self.assertEqual((events, flagged, hits.shape), (4, 1, (10, 6)))
        self.assertEqual(hits[0].tolist(), [0, 2, 2, 250, 70, 180])  # ea row col toa tot cal
        self.assertEqual(hits[6].tolist(), [2, 0, 0, 256, 0, 0])

    def test_the_first_file_of_a_single_run_stays_out(self):
        pixels = QUICK_PIXELS[:2]
        run_dir = write_run(self.wafer, 3, 1, nem={"qinj": [
            event(pixels, status=8, junk=2) * 2, event(pixels) * 3, event(pixels)]})
        files = qinj_files(run_dir)
        self.assertEqual([f.name for f in files], ["file_1.nem", "file_2.nem"])
        events, flagged, hits = read_nem_hits(files)
        self.assertEqual((events, flagged, hits.shape), (4, 0, (8, 6)))

    def test_a_run_without_qinj_has_no_files(self):
        run_dir = write_run(self.wafer, 3, 1)
        self.assertIsNone(qinj_files(run_dir))
        self.assertIsNone(qinj_files(None))

    def test_ea_flagged_hits_and_cal_outliers_stay_out(self):
        good = [[0, 2, 2, 250, 70, 180]] * 8
        rows = good + [[0, 2, 2, 290, 90, 183], [0, 2, 2, 260, 72, 182], [3, 2, 2, 250, 70, 180]]
        (stats,) = qinj_pixel_stats(np.array(rows), events=10)
        self.assertEqual((stats["hits"], stats["eff"], stats["cal_mode"], stats["n_sel"]), (10, 1.0, 180, 9))
        self.assertAlmostEqual(stats["toa_mean"], (8 * 250 + 260) / 9)
        self.assertAlmostEqual(stats["toa_std"], 10 / 3)  # sample std
        self.assertAlmostEqual(stats["cal_mean"], (8 * 180 + 182) / 9)

    def test_lowest_efficiency_over_the_pixels_each_run_injected(self):
        dies = pd.DataFrame([
            {"die": 1, "qinj_events": 10, "qinj_pixels": "[[2, 2], [5, 5]]"},
            {"die": 2, "qinj_events": 10, "qinj_hits_expected": 2},    # older run: count only
            {"die": 3, "qinj_events": 10},                             # older run: nothing recorded
            {"die": 4, "qinj_events": 10, "qinj_pixels": "[[2, 2], [15, 15]]"},
            {"die": 5},                                                # no QInj run
        ])
        qinj = pd.DataFrame([(1, 2, 2, 1.0), (1, 5, 5, 0.8), (2, 2, 2, 1.0), (2, 5, 5, 0.9), (3, 2, 2, 1.0),
                             (4, 2, 2, 1.0)], columns=["die", "pix_row", "pix_col", "eff"])
        low = lowest_efficiency(dies, qinj)
        self.assertEqual(low[:2], [0.8, 0.9])
        self.assertTrue(math.isnan(low[2]) and math.isnan(low[4]))
        self.assertEqual(low[3], 0.0)  # (15, 15) injected, no hit


@needs_tables
class OffsetTrendTest(unittest.TestCase):
    def test_a_plane_is_recovered(self):
        grid = [(r, c) for r in range(3) for c in range(4)]
        dies = pd.DataFrame({"die_row": [r for r, _ in grid], "die_col": [c for _, c in grid],
                             "dx_um": [3.0 - 2.0 * c + 0.5 * r for r, c in grid]})
        a, b, c, s, n = offset_trend(dies, "dx_um")
        self.assertEqual([round(v, 6) for v in (a, b, c, s)], [3.0, -2.0, 0.5, 0.0])
        self.assertEqual(n, 12)

    def test_three_dies_are_too_few(self):
        dies = pd.DataFrame({"die_row": [0, 1, 2], "die_col": [0, 1, 2], "dx_um": [1.0, 2.0, 3.0]})
        self.assertIsNone(offset_trend(dies, "dx_um"))


@needs_tables
class CollectTest(TempWafer):
    def test_every_map_die_gets_a_row(self):
        write_run(self.wafer, 1, 1, summary(1, 0, 0, qinj=True, qinj_pixels=[list(p) for p in QUICK_PIXELS]),
                  power=run_power(), baseline=baseline(QUICK_PIXELS),
                  nem={"qinj_run2": [event(QUICK_PIXELS) * 3]})
        write_run(self.wafer, 3, 1, summary(3, 1, 0, status="over_current"))
        write_run(self.wafer, 9, 1, summary(9, 5, 5))  # a folder the map does not list
        dies, pixels, qinj, warnings = collect(self.wafer, {1: (0, 0), 2: (0, 1), 3: (1, 0)})
        self.assertEqual(dies["grade"].tolist(), ["PASSED", "NOT_TESTED", "POWER_SHORT"])
        self.assertEqual(warnings, ["die009 is not in the wafer map, left out"])
        self.assertEqual((len(pixels), len(qinj)), (9, 9))
        self.assertEqual((dies.loc[0, "qinj_events"], dies.loc[0, "qinj_min_eff"]), (3, 1.0))
        self.assertTrue(math.isnan(dies.loc[2, "qinj_min_eff"]))

    def test_a_die_the_station_would_not_step_to_is_a_contact_failure(self):
        # the summary wafer_run.py writes for it, alone in its run folder
        write_run(self.wafer, 1, 1, summary(1, 0, 0), power=run_power())
        write_run(self.wafer, 2, 1, {
            "status": "contact_failure", "error": "StepToDie 2: [896] [Automation] Pattern matching could not "
                                                  "locate wafer model", "batch": "B", "wafer": "W",
            "batch_id": None, "wafer_id": None, "wafer_stage": "pre_ubm", "die": 2, "row": 0, "col": 1,
            "phases": {"step_refused": "2026-09-26 13:34:49.949"}, "attempt": 1, "retry_reason": None,
            "prober": {"batch": "B", "wafer": "W", "die": 2, "expected_row": 0, "expected_col": 1,
                       "error_code": 896}, "run": "run_01_noEfuse"})
        dies, pixels, qinj, warnings = collect(self.wafer, {1: (0, 0), 2: (0, 1)})
        self.assertEqual(dies["grade"].tolist(), ["PASSED", "CONTACT_FAILURE"])
        self.assertEqual((dies.loc[1, "bin"], dies.loc[1, "map_text"]), (12, "CONTACT_FAILURE"))
        self.assertTrue(math.isnan(dies.loc[1, "analog_I_on"]))
        self.assertEqual(grade_counts(dies)[1:], (1, 1))  # passed, tested: no test ran on die 2

    def test_a_qinj_run_that_wrote_a_single_file_has_no_qinj_data(self):
        pixels = [list(p) for p in QUICK_PIXELS]
        write_run(self.wafer, 1, 1, summary(1, 0, 0, qinj=True, qinj_pixels=pixels),
                  nem={"qinj": [event(QUICK_PIXELS, status=8, junk=2) * 3]})
        write_run(self.wafer, 2, 1, summary(2, 0, 1, qinj=True, qinj_pixels=pixels),
                  nem={"qinj": [event(QUICK_PIXELS), event(QUICK_PIXELS) * 2]})
        dies, _, qinj, _ = collect(self.wafer, {1: (0, 0), 2: (0, 1)})
        for column in ("qinj_events", "qinj_flagged_trailers", "qinj_ea_words", "qinj_min_eff"):
            self.assertTrue(math.isnan(dies.loc[0, column]), column)
        self.assertEqual((dies.loc[1, "qinj_events"], dies.loc[1, "qinj_min_eff"]), (2, 1.0))
        self.assertEqual(set(qinj["die"]), {2})


@needs_tables
class BlNwNoteTest(unittest.TestCase):
    """A note on a die whose baseline or noise width stands out: pixels
    far from their die's median, die means far from the PASSED dies'."""

    def wafer(self, n=None, first=1, n_pixels=256, nw_offset=0.0):
        """n PASSED dies (NOTE_MIN_DIES by default) from die `first`,
        calibrated over n_pixels, whose means spread a little."""
        dies = range(first, first + (n or NOTE_MIN_DIES))
        return pd.DataFrame({"die": list(dies), "grade": "PASSED", "n_pixels": n_pixels,
                             "bl_mean": [520.0 + d % 7 for d in dies],
                             "nw_mean": [7.0 + nw_offset + 0.05 * (d % 5) for d in dies]})

    def pixels(self, die, odd=None):
        """Nine pixels of one die at baseline 550 and noise width 7; `odd`
        maps a pixel to its own (baseline, noise width)."""
        cells = [(r, c) for r in range(3) for c in range(3)]
        values = [(odd or {}).get(cell, (550, 7)) for cell in cells]
        return pd.DataFrame({"die": die, "pix_row": [r for r, _ in cells], "pix_col": [c for _, c in cells],
                             "baseline": [b for b, _ in values], "noise_width": [w for _, w in values]})

    def notes(self, dies, pixels):
        return dict(zip(dies["die"], bl_nw_notes(dies, pixels)))

    def test_pixels_far_from_their_die_median_are_noted(self):
        notes = self.notes(self.wafer(), self.pixels(3, {(2, 2): (257, 7), (0, 1): (550, 16)}))
        self.assertEqual(notes[3], "die median baseline 550, pixel (2,2) at 257; "
                                   "die median noise width 7, pixel (0,1) at 16")
        self.assertEqual({die for die, note in notes.items() if note}, {3})

    def test_a_pixel_at_the_limits_is_not_noted(self):
        odd = {(2, 2): (550 - NOTE_PIXEL_BL, 7), (0, 1): (550, 7 + NOTE_PIXEL_NW)}
        self.assertEqual(self.notes(self.wafer(), self.pixels(3, odd))[3], "")

    def test_zero_readings_are_left_to_the_grade(self):
        self.assertEqual(self.notes(self.wafer(), self.pixels(3, {(2, 2): (0, 0), (1, 1): (0, 7)}))[3], "")

    def test_the_furthest_pixels_are_listed_and_the_rest_counted(self):
        odd = {(0, 0): (700, 7), (0, 1): (900, 7), (0, 2): (800, 7), (1, 0): (1000, 7)}
        self.assertEqual(self.notes(self.wafer(), self.pixels(3, odd))[3],
                         "die median baseline 550, pixels (1,0) at 1000, (0,1) at 900, (0,2) at 800 and 1 more")

    def test_a_die_mean_far_from_the_passed_dies_is_noted(self):
        dies = self.wafer()
        dies.loc[dies["die"] == 5, "nw_mean"] = 9.4
        notes = self.notes(dies, self.pixels(5))
        self.assertRegex(notes[5], r"^noise-width mean 9\.40, \+\d+\.\d sigma from the median 7\.10 "
                                   r"of the PASSED 256-pixel dies$")
        self.assertEqual({die for die, note in notes.items() if note}, {5})

    def test_the_spread_comes_from_the_passed_dies_only(self):
        failed = pd.DataFrame({"die": range(101, 131), "grade": "POWER_SHORT", "n_pixels": 256,
                               "bl_mean": 900.0, "nw_mean": 7.1})
        notes = self.notes(pd.concat([self.wafer(), failed], ignore_index=True), self.pixels(1))
        self.assertEqual({die for die, note in notes.items() if note}, set(range(101, 131)))
        self.assertTrue(notes[101].startswith("baseline mean 900, +"))

    def test_quick_test_and_full_scan_dies_are_checked_apart(self):
        dies = pd.concat([self.wafer(30), self.wafer(first=31, n_pixels=9, nw_offset=3.0)], ignore_index=True)
        self.assertEqual({die for die, note in self.notes(dies, self.pixels(1)).items() if note}, set())

    def test_the_die_check_needs_enough_passed_dies(self):
        dies = self.wafer(NOTE_MIN_DIES - 1)
        dies.loc[dies["die"] == 5, "nw_mean"] = 9.4
        self.assertEqual(self.notes(dies, self.pixels(5))[5], "")
        self.assertEqual(unchecked_die_means(dies), [(256, ["bl_mean", "nw_mean"])])


@needs_tables
class CurrentNoteTest(unittest.TestCase):
    """A note on a PASSED die drawing far more current than the PASSED dies."""

    def wafer(self, n=None):
        """n valid PASSED dies (NOTE_MIN_DIES by default) at 0.30 A analog and
        0.13 A digital at power-on, 0.40 and 0.12 A at high power."""
        dies = list(range(1, (n or NOTE_MIN_DIES) + 1))
        return pd.DataFrame({"die": dies, "grade": "PASSED", "invalid": False,
                             "analog_I_on": 0.30, "analog_I_high": 0.40,
                             "digital_I_on": 0.13, "digital_I_high": 0.12})

    def notes(self, dies):
        return dict(zip(dies["die"], current_notes(dies)))

    def test_a_passed_die_far_above_the_median_is_noted(self):
        # N62H30 02C7 post-UBM die 73
        dies = self.wafer()
        dies.loc[dies["die"] == 7, ["analog_I_on", "analog_I_high"]] = [0.571, 0.687]
        notes = self.notes(dies)
        self.assertEqual(notes[7], "analog power-on 571 mA, 1.90 x the median 300 mA; "
                                   "analog high-power 687 mA, 1.72 x the median 400 mA")
        self.assertEqual({die for die, note in notes.items() if note}, {7})

    def test_a_current_at_the_factor_is_not_noted(self):
        dies = self.wafer()
        dies.loc[dies["die"] == 7, "digital_I_on"] = 0.13 * CURRENT_NOTE_FACTOR
        self.assertEqual(self.notes(dies)[7], "")

    def test_only_valid_passed_dies_are_noted_and_make_the_median(self):
        shorts = pd.DataFrame({"die": range(101, 131), "grade": "POWER_SHORT", "invalid": False,
                               "analog_I_on": 0.65, "analog_I_high": None, "digital_I_on": 0.13,
                               "digital_I_high": None})
        invalid = pd.DataFrame({"die": [57, 58], "grade": "PASSED", "invalid": True, "analog_I_on": 0.65,
                                "analog_I_high": 0.70, "digital_I_on": 0.30, "digital_I_high": 0.30})
        dies = pd.concat([self.wafer(), shorts, invalid], ignore_index=True)
        dies.loc[dies["die"] == 7, "analog_I_on"] = 0.46
        notes = self.notes(dies)
        self.assertEqual({die for die, note in notes.items() if note}, {7})
        self.assertEqual(notes[7], "analog power-on 460 mA, 1.53 x the median 300 mA")

    def test_the_check_needs_enough_passed_dies(self):
        dies = self.wafer(NOTE_MIN_DIES - 1)
        dies.loc[dies["die"] == 7, "analog_I_on"] = 0.60
        self.assertEqual(self.notes(dies)[7], "")


if __name__ == "__main__":
    unittest.main()
