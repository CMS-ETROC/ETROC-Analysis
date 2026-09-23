"""plot_wafer.py end to end on a synthetic wafer (tests/wafer_results.py):
which tables and figures it writes, and which it leaves out."""
import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

HAVE_PLOTS = all(importlib.util.find_spec(m) for m in ("numpy", "pandas", "pyarrow", "matplotlib"))
if HAVE_PLOTS:
    from plot_wafer import main
    from wafer_plots import FIGURES
    from tests.wafer_results import (FULL_PIXELS, QUICK_PIXELS, baseline, event, run_power, summary,
                                     write_map, write_run)

TABLES = {"dies.csv", "pixels.csv", "qinj.csv"}
WAFER_MAP = {1: (0, 1), 2: (0, 2), 3: (1, 0), 4: (1, 1), 5: (1, 2), 6: (2, 1)}


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
        write_run(self.wafer, die, 1, summary(die, row, col, prober=prober, **extra),
                  power=run_power(), baseline=baseline(QUICK_PIXELS))

    def full_qinj_die(self, die):
        row, col = WAFER_MAP[die]
        write_run(self.wafer, die, 1, summary(die, row, col, fullscan=True, qinj=True,
                                              qinj_pixels=[list(p) for p in QUICK_PIXELS]),
                  power=run_power(), baseline=baseline(FULL_PIXELS, zero=[(4, 10)]),
                  nem={"qinj": [event(QUICK_PIXELS, status=8, junk=3) * 2, event(QUICK_PIXELS) * 5,
                                event(QUICK_PIXELS, status=8, junk=1)]})

    def plot(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            rc = main(["--path", str(self.root), "--batchName", "B", "--waferName", "W",
                       "--waferMap", str(self.root / "map.csv"), "--out", str(self.out), *extra])
        return rc, printed.getvalue(), {p.name for p in self.out.iterdir()} if self.out.is_dir() else set()

    def test_a_wafer_with_every_kind_of_run_gets_every_figure(self):
        for die in (1, 3, 4):
            self.quick_die(die)
        self.full_qinj_die(2)
        write_run(self.wafer, 5, 1, summary(5, 1, 2, status="over_current"))
        rc, printed, written = self.plot()
        self.assertEqual(rc, 0)
        self.assertIn("B / W: 5 of 6 dies tested, PASSED 4 (80.0 %)", printed)
        self.assertEqual(written, TABLES | {f"{name}.png" for name in FIGURES})

    def test_figures_without_data_are_left_out_and_older_copies_removed(self):
        self.out.mkdir()
        for name in ("qinj_cal.png", "fullscan_baseline.png", "alignment.png"):
            (self.out / name).write_bytes(b"from an earlier plot")
        write_run(self.wafer, 1, 1, summary(1, 0, 1), power=run_power(), baseline=baseline(QUICK_PIXELS))
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

    def test_a_missing_results_folder_is_refused(self):
        rc, printed, written = self.plot("--waferName", "nowhere")
        self.assertEqual((rc, written), (2, set()))
        self.assertIn("0 results folders for B / nowhere", printed)

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
        self.assertIn("2 results folders for B / W", printed)


if __name__ == "__main__":
    unittest.main()
