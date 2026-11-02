"""What plot_qinj.py makes of the qinj.csv and dies.csv tables plot_wafer.py
writes, on synthetic tables: the conversion to time, the H-tree prediction
and which dies and pixels it keeps."""
import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

HAVE_PLOTS = all(importlib.util.find_spec(m) for m in ("numpy", "pandas", "matplotlib"))
if HAVE_PLOTS:
    import pandas as pd
    from plot_qinj import htree_toa_ns, main, to_ns

INJECTED = {(2, 2): 200.0, (5, 5): 210.0, (2, 10): 190.0}  # TOA code of each injected pixel
CAL = {(2, 2): 160.0, (5, 5): 160.0, (2, 10): 140.0}


def tables(n=4, grades=None):
    """(dies.csv, qinj.csv) of n dies as plot_wafer.py writes them; die n is
    invalid, and die 1 has one stray hit on pixel (7, 8)."""
    dies = pd.DataFrame({"die": range(1, n + 1), "grade": [(grades or {}).get(d, "PASSED") for d in range(1, n + 1)],
                         "invalid": [d == n for d in range(1, n + 1)], "start": "2026-09-22 15:00:00"})
    rows = [{"die": d, "pix_row": r, "pix_col": c, "eff": 1.0, "toa_mean": toa, "tot_mean": 50.0, "cal_mean": CAL[(r, c)]}
            for d in range(1, n + 1) for (r, c), toa in INJECTED.items()]
    rows.append({"die": 1, "pix_row": 7, "pix_col": 8, "eff": 1e-4, "toa_mean": 992.0, "tot_mean": 85.0,
                 "cal_mean": 400.0})
    return dies, pd.DataFrame(rows)


@unittest.skipUnless(HAVE_PLOTS, "needs numpy, pandas and matplotlib (the wafer-daq venv)")
class ConversionTest(unittest.TestCase):
    def test_toa_counts_back_from_12p5_ns_and_tot_doubles_the_code(self):
        ns = to_ns(pd.DataFrame({"cal_mean": [160.0], "toa_mean": [200.0], "tot_mean": [50.0]}))
        tdc_bin = 3.125 / 160
        self.assertAlmostEqual(ns.loc[0, "bin_ps"], tdc_bin * 1e3)
        self.assertAlmostEqual(ns.loc[0, "toa_ns"], 12.5 - 200 * tdc_bin)
        self.assertAlmostEqual(ns.loc[0, "tot_ns"], (2 * 50 - 1) * tdc_bin)

    def test_the_htree_prediction_follows_the_parity_of_row_and_column(self):
        self.assertAlmostEqual(htree_toa_ns(5, 5) - htree_toa_ns(2, 2), -0.18)
        self.assertAlmostEqual(htree_toa_ns(13, 13) - htree_toa_ns(2, 10), -0.18)
        self.assertAlmostEqual(htree_toa_ns(5, 2) - htree_toa_ns(2, 2), -0.11)
        self.assertAlmostEqual(htree_toa_ns(4, 4) - htree_toa_ns(2, 2), -0.01)
        self.assertAlmostEqual(htree_toa_ns(2, 3) - htree_toa_ns(2, 1), 0.01)


@unittest.skipUnless(HAVE_PLOTS, "needs numpy, pandas and matplotlib (the wafer-daq venv)")
class PlotQinjTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.out = self.root / "qinj"

    def write(self, wafer, stage, dies, qinj):
        folder = self.root / "BatchID_X_Name_B" / f"WaferID_X_Name_{wafer}" / stage / "plots"
        folder.mkdir(parents=True)
        dies.to_csv(folder / "dies.csv", index=False)
        if qinj is not None:
            qinj.to_csv(folder / "qinj.csv", index=False)

    def run_qinj(self, *lots):
        argv = ["--path", str(self.root), "--out", str(self.out)]
        for lot in lots:
            argv += ["--lot", lot]
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            rc = main(argv)
        return rc, printed.getvalue()

    def test_the_injected_pixels_of_the_passed_valid_dies_and_the_figure(self):
        self.write("W1", "pre_ubm", *tables(grades={2: "POWER_SHORT"}))
        self.write("W1", "post_ubm", tables()[0], None)
        rc, printed = self.run_qinj("L=B")
        self.assertEqual(rc, 0, printed)
        self.assertIn("W1 post_ubm left out", printed)
        table = pd.read_csv(self.out / "qinj_pixels_ns.csv")
        self.assertEqual(sorted(table["die"].unique()), [1, 3])
        self.assertEqual(sorted(set(zip(table["pix_row"], table["pix_col"]))), sorted(INJECTED))
        row = table[(table["die"] == 1) & (table["pix_row"] == 5)].iloc[0]
        self.assertAlmostEqual(row["dtoa_ns"], -(210 - 200) * 3.125 / 160)
        self.assertAlmostEqual(row["htree_dtoa_ns"], -0.18)
        self.assertEqual({p.name for p in self.out.glob("*.png")}, {"L_pre_ubm_qinj.png"})

    def test_a_stage_without_the_reference_pixel_or_a_passed_die_is_left_out(self):
        dies, qinj = tables()
        self.write("W1", "pre_ubm", dies, qinj[(qinj["pix_row"] != 2) | (qinj["pix_col"] != 2)])
        self.write("W2", "pre_ubm", dies.assign(grade="POWER_SHORT"), qinj)
        self.write("W3", "pre_ubm", dies, qinj)
        rc, printed = self.run_qinj("L=B")
        self.assertEqual(rc, 0, printed)
        self.assertIn("W1 pre_ubm left out: the reference pixel (2, 2) is not among its injected pixels", printed)
        self.assertIn("W2 pre_ubm left out: no PASSED valid die with charge injection", printed)
        self.assertEqual(set(pd.read_csv(self.out / "qinj_pixels_ns.csv")["wafer"]), {"W3"})


if __name__ == "__main__":
    unittest.main()
