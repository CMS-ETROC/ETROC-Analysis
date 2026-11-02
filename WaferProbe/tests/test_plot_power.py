"""What plot_power.py makes of the dies.csv tables plot_wafer.py writes,
on synthetic tables: one wafer with both stages, one before UBM only."""
import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

HAVE_PLOTS = all(importlib.util.find_spec(m) for m in ("numpy", "pandas", "matplotlib"))
if HAVE_PLOTS:
    import pandas as pd
    from plot_power import NOMINAL_V, POWER_RAILS, die_power, main

RAIL_CURRENTS = {"analog": (0.30, 0.45), "digital": (0.13, 0.12), "ws_analog": (0.03, 0.03),
                 "ws_digital": (0.0, 0.0), "vref": (0.0, 0.0)}
VOLTS = {"analog": 1.36, "digital": 1.30, "ws_analog": 1.22, "ws_digital": 1.21, "vref": 1.0}


def dies_table(n=4, grades=None, drop=()):
    """A dies.csv as plot_wafer.py writes it, the columns plot_power reads;
    the rails in `drop` were not logged."""
    rows = []
    for die in range(1, n + 1):
        row = {"die": die, "die_row": 0, "die_col": die, "grade": (grades or {}).get(die, "PASSED"),
               "invalid": die == n, "start": "2026-09-22 15:00:00"}
        for rail, (on, high) in RAIL_CURRENTS.items():
            if rail in drop:
                continue
            row.update({f"{rail}_I_on": on, f"{rail}_V_on": VOLTS[rail],
                        f"{rail}_I_high": high, f"{rail}_V_high": VOLTS[rail]})
        rows.append(row)
    return pd.DataFrame(rows)


@unittest.skipUnless(HAVE_PLOTS, "needs numpy, pandas and matplotlib (the wafer-daq venv)")
class DiePowerTest(unittest.TestCase):
    def test_the_power_is_the_nominal_voltage_times_the_current_summed_over_the_rails(self):
        power = die_power(dies_table(n=1))
        self.assertEqual(NOMINAL_V, 1.2)
        expected = {phase: sum(1.2 * RAIL_CURRENTS[r][k] for r in POWER_RAILS)
                    for k, phase in enumerate(("on", "high"))}
        self.assertAlmostEqual(power.loc[0, "power_on_W"], expected["on"])
        self.assertAlmostEqual(power.loc[0, "power_high_W"], expected["high"])
        self.assertAlmostEqual(power.loc[0, "analog_P_high"], 1.2 * 0.45)

    def test_a_rail_the_run_did_not_log_leaves_the_die_without_a_power(self):
        power = die_power(dies_table(n=1, drop=("ws_analog",)))
        self.assertTrue(pd.isna(power.loc[0, "power_high_W"]))
        self.assertAlmostEqual(power.loc[0, "analog_P_high"], 1.2 * 0.45)


@unittest.skipUnless(HAVE_PLOTS, "needs numpy, pandas and matplotlib (the wafer-daq venv)")
class PlotPowerTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.out = self.root / "power"

    def write(self, batch, wafer, stage, table):
        folder = self.root / f"BatchID_X_Name_{batch}" / f"WaferID_X_Name_{wafer}" / stage / "plots"
        folder.mkdir(parents=True)
        table.to_csv(folder / "dies.csv", index=False)

    def run_power(self, *lots):
        argv = ["--path", str(self.root), "--out", str(self.out)]
        for lot in lots:
            argv += ["--lot", lot]
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            rc = main(argv)
        return rc, printed.getvalue()

    def test_every_stage_with_a_dies_table_and_the_figures(self):
        self.write("B", "W1", "pre_ubm", dies_table(grades={2: "POWER_SHORT"}))
        self.write("B", "W1", "post_ubm", dies_table())
        self.write("B", "W2", "pre_ubm", dies_table())
        (self.root / "BatchID_X_Name_B" / "WaferID_X_Name_W2" / "post_ubm").mkdir()
        rc, printed = self.run_power("B")
        self.assertEqual(rc, 0, printed)
        self.assertIn("W2 post_ubm left out", printed)
        self.assertIn("W1 pre_ubm: 2 PASSED valid dies", printed)
        table = pd.read_csv(self.out / "power_dies.csv")
        self.assertEqual(sorted(set(zip(table["wafer"], table["stage"]))),
                         [("W1", "post_ubm"), ("W1", "pre_ubm"), ("W2", "pre_ubm")])
        self.assertEqual({p.name for p in self.out.glob("*.png")}, {"power.png", "power_rails.png"})

    def test_no_passed_die_with_a_power_gives_the_table_and_no_figures(self):
        self.write("B", "W1", "pre_ubm", dies_table(grades={1: "POWER_SHORT", 2: "POWER_SHORT", 3: "POWER_SHORT"}))
        rc, printed = self.run_power("B")
        self.assertEqual(rc, 2)
        self.assertIn("no PASSED valid die of the lots has a power", printed)
        self.assertTrue((self.out / "power_dies.csv").is_file())
        self.assertEqual(list(self.out.glob("*.png")), [])

    def test_a_listed_wafer_that_is_not_there_stops_it(self):
        self.write("B", "W1", "pre_ubm", dies_table())
        rc, printed = self.run_power("B:W1,W7")
        self.assertEqual(rc, 2)
        self.assertIn("no wafer folder W7", printed)


if __name__ == "__main__":
    unittest.main()
