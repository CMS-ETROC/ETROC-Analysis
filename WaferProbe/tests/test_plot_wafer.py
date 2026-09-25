"""plot_wafer.py end to end on a synthetic wafer (tests/wafer_results.py):
which tables and figures it writes, and which it leaves out."""
import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HAVE_PLOTS = all(importlib.util.find_spec(m) for m in ("numpy", "pandas", "pyarrow", "matplotlib"))
if HAVE_PLOTS:
    import matplotlib.pyplot as plt
    from plot_wafer import main
    from station import load_wafer_map
    from wafer_plots import FIGURES, fig_fullscan, fig_pixel_issues
    from wafer_tables import collect
    from tests.wafer_results import (FULL_PIXELS, QUICK_PIXELS, baseline, event, run_power, summary,
                                     write_map, write_run)

TABLES = {"dies.csv", "pixels.csv", "qinj.csv"}
WAFER_MAP = {1: (0, 1), 2: (0, 2), 3: (1, 0), 4: (1, 1), 5: (1, 2), 6: (2, 1)}
STAGE = "pre_ubm"


@unittest.skipUnless(HAVE_PLOTS, "needs numpy, pandas, pyarrow and matplotlib (the wafer-daq venv)")
class PlotWaferTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.wafer = self.root / "BatchID_X_Name_B" / "WaferID_X_Name_W"
        self.out = self.root / "plots"
        write_map(self.root / "map.csv", WAFER_MAP)

    def quick_die(self, die, **extra):
        row, col = WAFER_MAP[die]
        prober = {"velox": {"x_um": 1000.0 * col, "y_um": 1000.0 * row},
                  "chuck": {"x_um": 1000.0 * col + 2.0 * col, "y_um": 1000.0 * row - 1.0, "z_contact_um": 50.0}}
        write_run(self.wafer / STAGE, die, 1, summary(die, row, col, prober=prober, **extra),
                  power=run_power(), baseline=baseline(QUICK_PIXELS))

    def full_qinj_die(self, die):
        row, col = WAFER_MAP[die]
        write_run(self.wafer / STAGE, die, 1, summary(die, row, col, fullscan=True, qinj=True,
                                              qinj_pixels=[list(p) for p in QUICK_PIXELS]),
                  power=run_power(), baseline=baseline(FULL_PIXELS, zero=[(4, 10)]),
                  nem={"qinj": [event(QUICK_PIXELS, status=8, junk=3) * 2, event(QUICK_PIXELS) * 5,
                                event(QUICK_PIXELS, status=8, junk=1)]})

    def plot(self, *extra, select=("--batchName", "B", "--waferName", "W"), stage=STAGE):
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            rc = main(["--path", str(self.root), "--waferStage", stage, *select,
                       "--waferMap", str(self.root / "map.csv"), "--out", str(self.out), *extra])
        return rc, printed.getvalue(), {p.name for p in self.out.iterdir()} if self.out.is_dir() else set()

    def test_a_wafer_with_every_kind_of_run_gets_every_figure(self):
        for die in (1, 3, 4):
            self.quick_die(die)
        self.full_qinj_die(2)
        write_run(self.wafer / STAGE, 5, 1, summary(5, 1, 2, status="over_current"))
        rc, printed, written = self.plot()
        self.assertEqual(rc, 0)
        self.assertIn("BatchID_X_Name_B / WaferID_X_Name_W / pre_ubm: 5 of 6 dies tested, PASSED 4 (80.0 %)", printed)
        self.assertEqual(written, TABLES | {f"{name}.png" for name in FIGURES})

    def test_pixel_maps_are_drawn_notch_up(self):
        # seen with the notch up the chip is upside down: pixel (0, 0) top left, (15, 15) bottom right
        self.full_qinj_die(2)
        dies, pixels, _, _ = collect(self.wafer / STAGE, load_wafer_map(self.root / "map.csv"))
        for fig in (fig_fullscan(dies, pixels, "t", "baseline", "baseline"), fig_pixel_issues(dies, pixels, "t")):
            maps = [ax for ax in fig.axes if ax.images]
            self.assertTrue(maps)
            for ax in maps:
                (x0, y0), (x15, y15) = ax.transData.transform([(0, 0), (15, 15)])
                self.assertLess(x0, x15)
                self.assertGreater(y0, y15)
            self.assertIn("notch up", fig._suptitle.get_text())
            plt.close(fig)

    def test_figures_without_data_are_left_out_and_older_copies_removed(self):
        self.out.mkdir()
        for name in ("qinj_cal.png", "fullscan_baseline.png", "alignment.png"):
            (self.out / name).write_bytes(b"from an earlier plot")
        write_run(self.wafer / STAGE, 1, 1, summary(1, 0, 1), power=run_power(), baseline=baseline(QUICK_PIXELS))
        rc, _, written = self.plot()
        self.assertEqual(rc, 0)
        left_out = {"alignment", "fullscan_baseline", "fullscan_noise_width", "qinj_overview", "qinj_cal",
                    "qinj_toa", "qinj_tot", "qinj_pixels"}
        self.assertEqual(written, TABLES | {f"{name}.png" for name in FIGURES if name not in left_out})

    def test_tables_only(self):
        self.quick_die(1)
        rc, printed, written = self.plot("--tables-only")
        self.assertEqual((rc, written), (0, TABLES))
        self.assertNotIn("not redrawn", printed)

    def test_tables_only_warns_about_figures_it_leaves(self):
        self.quick_die(1)
        self.out.mkdir()
        (self.out / "grades.png").write_bytes(b"from an earlier plot")
        rc, printed, written = self.plot("--tables-only")
        self.assertEqual((rc, written), (0, TABLES | {"grades.png"}))
        self.assertIn("the 1 figures already in", printed)

    def test_every_station_grade_has_a_colour(self):
        import station
        from wafer_plots import GRADE_COLOURS
        names = {name for name, value in vars(station).items()
                 if name.isupper() and not name.startswith("_") and name != "N_BINS" and isinstance(value, int)}
        self.assertEqual(names - set(GRADE_COLOURS), set())

    def test_every_dies_column_the_figures_read_is_in_the_table(self):
        import re
        import pandas as pd
        for die in (1, 3, 4):
            self.quick_die(die)
        self.full_qinj_die(2)
        self.plot("--tables-only")
        columns = set(pd.read_csv(self.out / "dies.csv").columns)
        source = (Path(__file__).resolve().parents[1] / "wafer_plots.py").read_text()
        read = set(re.findall(r'_num\(dies, "(\w+)"\)', source)) | set(re.findall(r'dies\["(\w+)"\]', source))
        self.assertEqual(read - columns, set())

    def test_an_efficiency_below_100_percent_never_prints_100(self):
        import matplotlib.pyplot as plt
        import pandas as pd
        from wafer_plots import fig_qinj_overview
        from wafer_tables import QINJ_COLUMNS
        dies = pd.DataFrame([{"die": 1, "die_row": 0, "die_col": 0, "grade": "PASSED", "qinj_events": 8544,
                              "qinj_min_eff": 8540 / 8544, "qinj_flagged_trailers": 0, "qinj_ea_words": 0}])
        qinj = pd.DataFrame([{"die": 1, "pix_row": 2, "pix_col": 2, "hits": 8540, "eff": 8540 / 8544}],
                            columns=QINJ_COLUMNS)
        fig = fig_qinj_overview(dies, qinj, "t")
        self.addCleanup(plt.close, fig)
        self.assertEqual([t.get_text() for t in fig.axes[1].texts], ["99"])

    def test_only_the_stage_asked_for_is_read_and_it_names_the_wafer(self):
        import pandas as pd
        self.quick_die(1)
        write_run(self.wafer / "post_ubm", 3, 1, summary(3, *WAFER_MAP[3], wafer_stage="post_ubm"),
                  power=run_power(), baseline=baseline(QUICK_PIXELS))
        rc, printed, _ = self.plot("--tables-only", stage="post_ubm")
        self.assertEqual(rc, 0)
        self.assertIn(f"reading {self.wafer / 'post_ubm'}\n", printed)
        self.assertIn("BatchID_X_Name_B / WaferID_X_Name_W / post_ubm: 1 of 6 dies tested", printed)
        ran = pd.read_csv(self.out / "dies.csv").dropna(subset=["run"])
        self.assertEqual((ran["die"].tolist(), ran["wafer_stage"].tolist()), ([3], ["post_ubm"]))

    def test_a_run_of_another_stage_in_a_stage_folder_is_refused(self):
        self.quick_die(1)
        write_run(self.wafer / STAGE, 3, 1, summary(3, *WAFER_MAP[3], wafer_stage="post_ubm"),
                  power=run_power(), baseline=baseline(QUICK_PIXELS))
        rc, printed, written = self.plot("--tables-only")
        self.assertEqual((rc, written), (2, set()))
        self.assertIn("ran as another stage, nothing written: die 3 (post_ubm)", printed)

    def test_a_wafer_without_the_stage_folder_is_refused_and_loose_die_folders_are_named(self):
        write_run(self.wafer, 1, 1, summary(1, 0, 1), power=run_power(), baseline=baseline(QUICK_PIXELS))
        rc, printed, written = self.plot("--tables-only")
        self.assertEqual((rc, written), (2, set()))
        self.assertIn(f"warning: 1 die folders directly in {self.wafer}, outside a stage folder", printed)
        self.assertIn(f"no pre_ubm folder in {self.wafer}; the stages there: none", printed)

    def test_a_run_without_a_recorded_stage_is_read_with_a_warning(self):
        self.quick_die(1, wafer_stage=None)
        rc, printed, _ = self.plot("--tables-only")
        self.assertEqual(rc, 0)
        self.assertIn("warning: 1 dies ran without a recorded stage", printed)

    def test_the_stage_is_required(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
            main(["--path", str(self.root), "--batchName", "B", "--waferName", "W"])
        self.assertIn("--waferStage", err.getvalue())

    def test_a_missing_results_folder_is_refused(self):
        rc, printed, written = self.plot("--waferName", "nowhere")
        self.assertEqual((rc, written), (2, set()))
        self.assertIn("0 results folders match BatchID_*_Name_B/WaferID_*_Name_nowhere", printed)

    def test_the_folder_is_found_by_the_names_whatever_the_ids(self):
        self.wafer = self.root / "BatchID_0_Name_B" / "WaferID_3_Name_W"
        (self.root / "BatchID_0_Name_BB" / "WaferID_4_Name_W").mkdir(parents=True)
        self.quick_die(1)
        rc, printed, written = self.plot("--tables-only")
        self.assertEqual(rc, 0)
        self.assertIn(f"reading {self.wafer}", printed)

    def test_two_folders_for_one_wafer_are_refused(self):
        self.quick_die(1)
        self.wafer = self.root / "BatchID_0_Name_B" / "WaferID_3_Name_W"
        self.quick_die(1)
        rc, printed, written = self.plot("--tables-only")
        self.assertEqual((rc, written), (2, set()))
        self.assertIn("2 results folders match BatchID_*_Name_B/WaferID_*_Name_W", printed)

    def use_ids(self):
        """The wafer under test in BatchID_0_Name_B/WaferID_3_Name_W, beside two
        wafers of the same batch (IDs 4 and 43) and one of another batch with
        wafer ID 3."""
        self.wafer = self.root / "BatchID_0_Name_B" / "WaferID_3_Name_W"
        self.quick_die(1)
        (self.root / "BatchID_0_Name_B" / "WaferID_4_Name_W2").mkdir(parents=True)
        (self.root / "BatchID_0_Name_B" / "WaferID_43_Name_W3").mkdir(parents=True)
        (self.root / "BatchID_1_Name_B1" / "WaferID_3_Name_W").mkdir(parents=True)

    def test_the_folder_is_found_by_the_ids(self):
        self.use_ids()
        rc, printed, _ = self.plot("--tables-only", select=("--batchID", "0", "--waferID", "3"))
        self.assertEqual(rc, 0)
        self.assertIn(f"reading {self.wafer}", printed)

    def test_names_and_ids_can_be_mixed(self):
        self.use_ids()
        for select in (("--batchName", "B", "--waferID", "3"), ("--batchID", "0", "--waferName", "W"),
                       ("--batchName", "B", "--batchID", "0", "--waferName", "W", "--waferID", "3")):
            with self.subTest(select=select):
                rc, printed, _ = self.plot("--tables-only", select=select)
                self.assertEqual(rc, 0)
                self.assertIn(f"reading {self.wafer}", printed)

    def test_a_name_and_an_id_that_disagree_are_refused(self):
        self.use_ids()
        rc, printed, written = self.plot("--tables-only", select=("--batchName", "B", "--batchID", "1",
                                                                  "--waferName", "W"))
        self.assertEqual((rc, written), (2, set()))
        self.assertIn("0 results folders match BatchID_1_Name_B/WaferID_*_Name_W", printed)
        self.assertIn(f"the wafer folders there:\n  {self.wafer}\n", printed)

    def test_an_x_folder_is_not_found_by_an_id(self):
        self.quick_die(1)
        rc, printed, written = self.plot("--tables-only", select=("--batchName", "B", "--waferID", "0"))
        self.assertEqual((rc, written), (2, set()))
        self.assertIn(f"the wafer folders there:\n  {self.wafer}\n", printed)

    def test_the_ids_pick_one_of_two_folders_with_the_same_names(self):
        self.use_ids()
        (self.root / "BatchID_2_Name_B" / "WaferID_5_Name_W").mkdir(parents=True)
        rc, printed, _ = self.plot("--tables-only")
        self.assertEqual(rc, 2)
        self.assertIn("2 results folders match BatchID_*_Name_B/WaferID_*_Name_W", printed)
        self.assertIn(f"  {self.root / 'BatchID_2_Name_B' / 'WaferID_5_Name_W'}\n", printed)
        rc, printed, _ = self.plot("--tables-only", "--waferID", "3")
        self.assertEqual(rc, 0)
        self.assertIn(f"reading {self.wafer}", printed)

    def test_each_level_needs_a_name_or_an_id(self):
        self.use_ids()
        for select in (("--batchName", "B"), ("--waferID", "3")):
            with self.subTest(select=select), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.plot("--tables-only", select=select)

    def test_the_header_title_and_note_name_the_wafer_by_its_labels(self):
        self.use_ids()
        with patch("wafer_plots.plot_all", return_value=[]) as plot_all:
            rc, printed, _ = self.plot(select=("--batchID", "0", "--waferID", "3"))
        self.assertEqual(rc, 0)
        self.assertIn("BatchID_0_Name_B / WaferID_3_Name_W / pre_ubm: 1 of 6 dies tested", printed)
        self.assertEqual(plot_all.call_args.kwargs["title"], "BatchID_0_Name_B / WaferID_3_Name_W / pre_ubm")
        self.assertTrue(plot_all.call_args.kwargs["note"].startswith(
            "BatchID_0_Name_B / WaferID_3_Name_W / pre_ubm: newest run of each die; plot_wafer.py "))

    def test_a_title_too_wide_for_its_figure_puts_the_labels_on_a_line_of_their_own(self):
        import matplotlib.pyplot as plt
        from wafer_plots import _suptitle
        label = "BatchID_0_Name_N62M23 / WaferID_3_Name_08A5"
        for inches, expected in ((20, f"{label}: the other rails at high power"),
                                 (5, f"{label}\nthe other rails at high power")):
            fig = plt.figure(figsize=(inches, 3))
            self.addCleanup(plt.close, fig)
            heading = _suptitle(fig, label, "the other rails at high power", fontsize=12)
            self.assertEqual(heading.get_text(), expected)

    def test_a_die_with_a_bl_nw_note_gets_a_letter_and_a_footnote(self):
        import matplotlib.pyplot as plt
        import pandas as pd
        from wafer_plots import fig_grades
        note = "die median baseline 550, pixel (8,8) at 257"
        dies = pd.DataFrame({"die": [1, 2, 3], "die_row": [0, 0, 1], "die_col": [0, 1, 0],
                             "grade": ["PASSED", "PASSED", "POWER_SHORT"], "bin": [0, 0, 1],
                             "map_text": ["PASSED_retry", "PASSED_retry", "POWER_SHORT"],
                             "bl_nw_note": ["", note, ""]})
        fig = fig_grades(dies, "B / W")
        self.addCleanup(plt.close, fig)
        self.assertEqual([t.get_text() for t in fig.axes[0].texts], ["1*", r"2*$^{\rm a}$", "3"])
        self.assertIn(f"a: die 2 (row 0, col 1): {note}", "\n".join(t.get_text() for t in fig.texts))
        dies["bl_nw_note"] = ""
        plain = fig_grades(dies, "B / W")
        self.addCleanup(plt.close, plain)
        self.assertEqual([t.get_text() for t in plain.axes[0].texts], ["1*", "2*", "3"])
        self.assertNotIn("letters:", "\n".join(t.get_text() for t in plain.texts))


if __name__ == "__main__":
    unittest.main()
