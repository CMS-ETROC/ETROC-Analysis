"""The pre_ubm vs post_ubm comparison plot_wafer.py writes when a wafer has
both stages (stage_compare.py, stage_compare_plots.py), on a synthetic
wafer: a full scan with QInj before, quick scans after."""
import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

HAVE_PLOTS = all(importlib.util.find_spec(m) for m in ("numpy", "pandas", "pyarrow", "matplotlib"))
if HAVE_PLOTS:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from plot_wafer import main
    from stage_compare_plots import fig_grade_changes, fig_qinj_changes
    from tests.wafer_results import (FULL_PIXELS, QUICK_PIXELS, baseline, event, run_power, summary,
                                     write_map, write_run)

WAFER_MAP = {1: (0, 1), 2: (0, 2), 3: (1, 0), 4: (1, 1), 5: (1, 2), 6: (2, 1)}


def shifted(pixels, d_bl=3, d_nw=1, scramble=False):
    """baseline.parquet rows: the pixel pattern of wafer_results.baseline
    moved by d_bl and d_nw, or, with `scramble`, another chip's pattern."""
    rows = []
    for r, c in pixels:
        bl = 400 + ((7 * r + 3 * c) % 11 if scramble else r + c) + d_bl
        rows.append({"row": r, "col": c, "baseline": bl, "noise_width": 6 + (r + c) % 3 + d_nw,
                     "timestamp": "2026-09-26 09:00:05", "chip_name": "0x60"})
    return pd.DataFrame(rows)


@unittest.skipUnless(HAVE_PLOTS, "needs numpy, pandas, pyarrow and matplotlib (the wafer-daq venv)")
class StageCompareTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.wafer = self.root / "BatchID_X_Name_B" / "WaferID_X_Name_W"
        self.out = self.root / "plots"
        write_map(self.root / "map.csv", WAFER_MAP, invalid=())

    def pre_die(self, die, qinj=True, pixels=True, hits=QUICK_PIXELS):
        row, col = WAFER_MAP[die]
        write_run(self.wafer / "pre_ubm", die, 1,
                  summary(die, row, col, fullscan=True, qinj=qinj, qinj_pixels=[list(p) for p in QUICK_PIXELS]),
                  power=run_power(on=0.31, high=0.45), baseline=baseline(FULL_PIXELS) if pixels else None,
                  nem={"qinj": [event(hits)] * 3} if qinj else None)

    def post_die(self, die, qinj=False, hits=QUICK_PIXELS, **extra):
        row, col = WAFER_MAP[die]
        scramble = extra.pop("scramble", False)
        write_run(self.wafer / "post_ubm", die, 1,
                  summary(die, row, col, wafer_stage="post_ubm", qinj=qinj,
                          qinj_pixels=[list(p) for p in QUICK_PIXELS], **extra),
                  power=run_power(on=0.33, high=0.46), baseline=shifted(QUICK_PIXELS, scramble=scramble),
                  nem={"qinj": [event(hits)] * 3} if qinj else None)

    def short_after(self, die):
        row, col = WAFER_MAP[die]
        short = {"analog": {"voltage": 0.6, "current": 0.65, "abort_above": 0.585, "ok": False,
                            "fault": "over_current"}}
        write_run(self.wafer / "post_ubm", die, 1,
                  summary(die, row, col, wafer_stage="post_ubm", status="over_current", short_check=short),
                  power=run_power(on=0.65, high=0.65))

    def plot(self, *extra, stage="post_ubm"):
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            rc = main(["--path", str(self.root), "--waferStage", stage, "--batchName", "B", "--waferName", "W",
                       "--waferMap", str(self.root / "map.csv"), "--out", str(self.out), *extra])
        self.assertEqual(rc, 0, printed.getvalue())
        compared = self.out / "pre_vs_post"
        files = {p.name for p in compared.iterdir()} if compared.is_dir() else set()
        changes = pd.read_csv(compared / "changes.csv").set_index("die") if "changes.csv" in files else None
        return printed.getvalue(), files, changes

    def usual_wafer(self):
        for die in (1, 2, 3, 4, 5):
            self.pre_die(die)
        for die in (1, 2, 3, 6):
            self.post_die(die)
        self.post_die(4, scramble=True)
        self.short_after(5)

    def test_quick_scans_after_a_full_scan_compare_at_the_common_pixels(self):
        self.usual_wafer()
        printed, files, changes = self.plot()
        self.assertEqual(files, {"changes.csv", "grade_changes.png", "current_changes.png", "baseline_changes.png"})
        self.assertEqual(changes.loc[[1, 2, 3, 4], "n_common_pixels"].tolist(), [9, 9, 9, 9])
        self.assertEqual(changes.loc[[1, 2, 3], "d_bl_mean"].tolist(), [3.0, 3.0, 3.0])
        self.assertEqual(changes.loc[[1, 2, 3], "d_nw_mean"].tolist(), [1.0, 1.0, 1.0])
        self.assertIn("no die ran QInj in both stages: no QInj comparison", printed)

    def test_currents_change_only_for_the_dies_passed_in_both_stages(self):
        self.usual_wafer()
        _, _, changes = self.plot("--tables-only")
        self.assertAlmostEqual(changes.loc[1, "d_analog_I_on"], 0.02)
        self.assertAlmostEqual(changes.loc[1, "d_analog_I_high"], 0.01)
        self.assertTrue(np.isnan(changes.loc[5, "d_analog_I_on"]))   # short after
        self.assertTrue(np.isnan(changes.loc[6, "d_analog_I_on"]))   # not tested before
        self.assertAlmostEqual(changes.loc[5, "analog_I_on_post"], 0.65)

    def test_the_grade_changes_are_listed_and_written_in_the_cells(self):
        self.usual_wafer()
        printed, _, changes = self.plot("--tables-only")
        self.assertIn("pre_ubm vs post_ubm: 2 valid dies changed grade\n"
                      "  NOT_TESTED -> PASSED (1): 6\n  PASSED -> POWER_SHORT (1): 5\n", printed)
        fig = fig_grade_changes(changes.reset_index(), "W")
        texts = sorted(t.get_text() for t in fig.axes[0].texts)
        plt.close(fig)
        self.assertEqual(texts, ["1", "2", "3", "4", "5\nP→S", "6\n-→P"])

    def test_a_die_whose_pixel_pattern_is_another_chips_is_flagged(self):
        self.usual_wafer()
        _, _, changes = self.plot("--tables-only")
        self.assertEqual(changes.index[changes["same_chip_flag"]].tolist(), [4])
        self.assertTrue((changes.loc[[1, 2, 3], "same_chip_r"] > 0.99).all())

    def test_qinj_in_both_stages_is_compared(self):
        for die in (1, 2):
            self.pre_die(die)
            self.post_die(die, qinj=True)
        _, files, changes = self.plot()
        self.assertIn("qinj_changes.png", files)
        self.assertEqual(changes.loc[[1, 2], "n_common_qinj_pixels"].tolist(), [9, 9])
        self.assertEqual(changes.loc[[1, 2], "d_toa_median"].tolist(), [0.0, 0.0])

    def test_a_pixel_or_die_that_lost_its_hits_sets_the_lowest_efficiency_and_is_named(self):
        stray = next((r, c) for r in range(16) for c in range(16) if (r, c) not in QUICK_PIXELS)
        self.pre_die(1)
        self.pre_die(2)
        self.pre_die(3, hits=QUICK_PIXELS + [stray])  # hit in both stages, never injected
        for die, hits in ((1, QUICK_PIXELS[1:]), (2, []), (3, QUICK_PIXELS + [stray])):
            self.post_die(die, qinj=True, hits=hits)
        _, files, changes = self.plot()
        self.assertIn("qinj_changes.png", files)
        self.assertEqual(changes.loc[[1, 2, 3], "eff_min_pre"].tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(changes.loc[[1, 2, 3], "eff_min_post"].tolist(), [0.0, 0.0, 1.0])
        # neither the pixel that lost its hits nor the one never injected is compared
        self.assertEqual(changes.loc[[1, 2, 3], "n_common_qinj_pixels"].tolist(), [8, 0, 9])
        rows = pd.DataFrame({f"d_{q}": [0.0] for q in ("toa", "tot", "cal")})
        fig = fig_qinj_changes(changes.reset_index(), rows, "W")
        self.addCleanup(plt.close, fig)
        self.assertIn("(3 dies with QInj in both); lowest pixel efficiency fell by more than 1 % on die(s) "
                      "1 (100% → 0%), 2 (100% → 0%)", fig._suptitle.get_text())

    def test_a_stage_without_power_phases_or_i2c_compares_by_grade_only(self):
        # as the February 2026 before_bump import writes a die whose log shows no switch to high power:
        # one reading per rail, a power.parquet without phases (left out here, to the same effect), no
        # baseline.parquet, no QInj, and no I2C record but a failed pixel ID the report names (die 3)
        checks = {rail: {"voltage": 1.36, "current": current, "abort_above": 0.63, "ok": True}
                  for rail, current in (("analog", 0.463), ("digital", 0.117))}
        report = {"chip": {"pixel_id": {"passed": False, "failed_pixels": [[0, 13]], "error": None}}}
        for die, i2c in ((1, {}), (2, {}), (3, report)):
            row, col = WAFER_MAP[die]
            write_run(self.wafer / "pre_ubm", die, 1,
                      summary(die, row, col, short_check=checks, i2c=i2c, imported={"source": "Wafer_N62H30_02C7"}))
            self.post_die(die)
        printed, files, changes = self.plot()
        self.assertEqual(files, {"changes.csv", "grade_changes.png"})
        self.assertEqual(changes.loc[[1, 2, 3], "grade_pre"].tolist(), ["PASSED", "PASSED", "I2C_PIXELS"])
        self.assertEqual(changes.loc[[1, 2], "analog_I_on_pre"].tolist(), [0.463, 0.463])
        self.assertTrue(changes["d_analog_I_on"].isna().all())
        self.assertEqual(changes.loc[[1, 2, 3], "i2c_pre"].tolist(), [False, False, True])
        self.assertEqual(changes.loc[[1, 2, 3], "i2c_post"].tolist(), [True, True, True])
        self.assertIn("pre_ubm logged no power phases: its currents are not compared", printed)
        self.assertIn("pre_ubm: 2 dies PASSED without an I2C record", printed)
        self.assertNotIn("post_ubm: 3 dies PASSED without", printed)
        self.assertIn("no pixel was calibrated in both stages: no baseline comparison", printed)

    def test_an_imported_stage_with_the_power_phases_compares_its_currents(self):
        # as the April 2026 N62M23 station runs imported into pre_ubm
        for die in (1, 2):
            row, col = WAFER_MAP[die]
            write_run(self.wafer / "pre_ubm", die, 1,
                      summary(die, row, col, fullscan=True, imported={"source": "N62M23 April 2026"}),
                      power=run_power(on=0.31, high=0.45), baseline=baseline(FULL_PIXELS))
            self.post_die(die)
        printed, files, changes = self.plot()
        self.assertIn("current_changes.png", files)
        self.assertEqual(changes.loc[[1, 2], "d_analog_I_on"].round(6).tolist(), [0.02, 0.02])
        self.assertEqual(changes.loc[[1, 2], "d_analog_I_high"].round(6).tolist(), [0.01, 0.01])
        self.assertNotIn("logged no power phases", printed)
        self.assertNotIn("without an I2C record", printed)

    def test_the_comparison_is_the_same_from_either_stage(self):
        self.usual_wafer()
        _, _, from_post = self.plot("--tables-only")
        _, _, from_pre = self.plot("--tables-only", stage="pre_ubm")
        pd.testing.assert_frame_equal(from_post, from_pre)

    def test_one_stage_alone_or_no_compare_writes_no_comparison(self):
        self.post_die(1)
        _, files, _ = self.plot()
        self.assertEqual(files, set())
        self.pre_die(1)
        _, files, _ = self.plot("--no-compare")
        self.assertEqual(files, set())

    def test_a_run_of_another_stage_in_the_other_stage_folder_stops_the_comparison(self):
        self.usual_wafer()
        row, col = WAFER_MAP[1]
        write_run(self.wafer / "pre_ubm", 1, 2, summary(1, row, col, wafer_stage="post_ubm",
                                                        power_on="2026-09-23 10:00:00.000"), power=run_power())
        printed, files, _ = self.plot("--tables-only")
        self.assertIn("ran as another stage, the stages are not compared: die 1 (post_ubm)", printed)
        self.assertEqual(files, set())

    def test_tables_only_warns_about_the_comparison_figures_it_leaves(self):
        self.usual_wafer()
        self.plot()
        printed, _, _ = self.plot("--tables-only")
        self.assertIn(f"figures already in {self.out / 'pre_vs_post'} were not redrawn", printed)

    def test_an_invalid_die_is_left_out_of_the_grade_changes_printed(self):
        write_map(self.root / "map.csv", WAFER_MAP, invalid=(5,))
        self.usual_wafer()
        printed, _, _ = self.plot("--tables-only")
        self.assertIn("pre_ubm vs post_ubm: 1 valid dies changed grade\n  NOT_TESTED -> PASSED (1): 6\n", printed)


if __name__ == "__main__":
    unittest.main()
