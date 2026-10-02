"""What plot_positions.py (position_compare.py, position_plots.py) makes of
synthetic wafers whose dies do or do not share a pattern by place."""
import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

HAVE_PLOTS = all(importlib.util.find_spec(m) for m in ("numpy", "pandas", "pyarrow", "matplotlib"))
if HAVE_PLOTS:
    import numpy as np
    import pandas as pd
    from plot_positions import main
    from position_compare import (MIN_QINJ_HITS, centred, die_values, injected_notes, map_pairs, mean_r, measured,
                                  pairwise_r, place_tilts, residual_maps, same_die_r, shuffled_r, tilts)
    from tests.wafer_results import FULL_PIXELS, QUICK_PIXELS, event, run_power, summary, write_map, write_run

GRID = {d: ((d - 1) // 5, (d - 1) % 5) for d in range(1, 26)}   # 25 dies, 5 x 5
COMMON = [(2, 2), (2, 10), (10, 2), (10, 10), (5, 5), (5, 13), (13, 5), (13, 13)]


def dies_table(wafer_dies, invalid=()):
    return pd.DataFrame({"die": d, "die_row": GRID[d][0], "die_col": GRID[d][1], "n_pixels": 256,
                         "invalid": d in invalid, "start": "2026-09-22 15:00:00", "imported": False}
                        for d in wafer_dies)


def pixels_table(baseline):
    """baseline: {die: 16 x 16 array}; noise width = baseline / 100."""
    return pd.DataFrame([{"die": d, "pix_row": r, "pix_col": c, "baseline": float(m[r, c]),
                          "noise_width": float(m[r, c]) / 100}
                         for d, m in baseline.items() for r in range(16) for c in range(16)])


CENTRE = (2, 2)   # of GRID
PIX_ROW, PIX_COL = np.divmod(np.arange(256).reshape(16, 16), 16) if HAVE_PLOTS else (None, None)


def lot(n_wafers, place=0.0, shared=30.0, own=0.0, noise=2.0, place_die=0.0, tilt=0.0, seed=1, invalid=()):
    """{wafer: (dies, pixels, qinj)}: every die's pixels = 500 + a pattern
    all chips share (amplitude `shared`) + a pixel pattern of its place
    (`place`) + one of its own (`own`) + noise + a tilt of `tilt` per pixel
    times its place's (row, column) offset from CENTRE, rising outward, and
    its whole die offset by `place_die` times a value of its place plus 100
    per wafer."""
    rng = np.random.default_rng(seed)
    common = shared * np.sin(np.arange(256).reshape(16, 16) / 7.0)
    outward = {d: tilt * ((r - CENTRE[0]) * (PIX_ROW - 7.5) + (c - CENTRE[1]) * (PIX_COL - 7.5))
               for d, (r, c) in GRID.items()}
    by_place = {d: rng.normal(0, 1, (16, 16)) for d in GRID}
    die_offset = {d: rng.normal(0, 1) for d in GRID}
    tables = {}
    for k in range(n_wafers):
        maps = {d: 500 + 100 * k + common + place * by_place[d] + own * rng.normal(0, 1, (16, 16)) + outward[d]
                + noise * rng.normal(0, 1, (16, 16)) + place_die * die_offset[d] + rng.normal(0, 1) for d in GRID}
        qinj = pd.DataFrame(columns=["die", "pix_row", "pix_col", "hits", "eff", "cal_mode", "n_sel", "toa_mean",
                                     "toa_std", "tot_mean", "tot_std", "cal_mean", "cal_std"])
        tables[f"W{k}"] = (dies_table(GRID, invalid), pixels_table(maps), qinj)
    return tables


def qinj_table(pixels, toa=200.0, extra=None, few=None, eff=1.0):
    """QInj rows of every die for `pixels`, hit in a fraction `eff` of the
    events: TOA `toa`, or for the pixel of extra = (pixel, toa) that one;
    the (die, pixel) `few` gets fewer hits than MIN_QINJ_HITS."""
    return pd.DataFrame([{"die": d, "pix_row": r, "pix_col": c, "hits": 100, "eff": eff, "cal_mode": 180,
                          "n_sel": MIN_QINJ_HITS - 1 if (d, (r, c)) == few else 100,
                          "toa_mean": extra[1] if extra and (r, c) == extra[0] else toa, "toa_std": 1.0,
                          "tot_mean": 70.0, "tot_std": 1.0, "cal_mean": 180.0, "cal_std": 0.5}
                         for d in GRID for r, c in pixels])


@unittest.skipUnless(HAVE_PLOTS, "needs numpy, pandas, pyarrow and matplotlib (the wafer-daq venv)")
class PositionCompareTest(unittest.TestCase):
    def test_dies_alike_by_place_give_a_pairwise_r_above_the_shuffled_level(self):
        values = die_values(lot(4, place_die=20.0))
        r = mean_r(pairwise_r(values, "baseline"))
        shuffled = shuffled_r(values, "baseline", n=100)
        self.assertGreater(r, 0.9)
        self.assertLess(np.percentile(shuffled, 97.5), 0.5)
        values = die_values(lot(4, place_die=0.0))
        self.assertLess(abs(mean_r(pairwise_r(values, "baseline"))), 0.3)

    def test_the_shuffled_level_keeps_each_wafer_on_the_places_it_measured(self):
        rng = np.random.default_rng(3)   # baseline on 25 places per wafer, 35 more places with a TOA value only
        values = pd.DataFrame({"wafer": w, "die": d, "baseline": rng.normal() if d <= 25 else np.nan, "toa": 1.0}
                              for w in ("W0", "W1", "W2") for d in range(1, 61))
        self.assertTrue(np.isfinite(shuffled_r(values, "baseline", n=20)).all())

    def test_each_wafer_is_centred_on_its_median_and_invalid_dies_are_left_out(self):
        tables = lot(3, invalid=(7,))
        for wafer, (dies, pixels, _) in tables.items():   # QInj data on every die, the invalid one too
            tables[wafer] = (dies, pixels, qinj_table(COMMON))
        values = die_values(tables)
        self.assertNotIn(7, set(values["die"]))
        self.assertTrue(np.isfinite(values["toa"]).all())
        c = centred(values, "baseline")
        for wafer, v in c.groupby(values["wafer"]):
            self.assertAlmostEqual(float(v.median()), 0.0, places=9, msg=wafer)

    def test_the_pattern_every_chip_shares_does_not_make_two_dies_alike(self):
        index, maps, pattern = residual_maps(lot(3, shared=30.0), "baseline")
        same, other, _ = map_pairs(index, maps)
        self.assertLess(abs(np.median(same)), 0.15)
        self.assertLess(abs(np.median(other)), 0.15)
        self.assertGreater(np.ptp(pattern), 40)   # the shared pattern is found, not lost

    def test_a_pixel_pattern_of_the_place_makes_the_same_place_alike(self):
        index, maps, _ = residual_maps(lot(3, place=5.0), "baseline")
        same, other, by_die = map_pairs(index, maps)
        self.assertGreater(np.median(same), 0.7)
        self.assertLess(abs(np.median(other)), 0.15)
        self.assertEqual(len(same), 3 * len(GRID))   # three wafer pairs per place
        self.assertEqual(len(other), 3 * len(GRID) * (len(GRID) - 1))   # two wafers, never one
        self.assertEqual(set(by_die.index), set(GRID))
        self.assertGreater(by_die.min(), 0.5)
        _, flat = tilts(maps)
        self.assertGreater(np.median(map_pairs(index, flat)[0]), 0.7)   # not a tilt: taking the tilt out keeps it

    def test_a_tilt_rising_outward_is_found_and_taken_out(self):
        index, maps, _ = residual_maps(lot(3, tilt=0.5), "baseline")
        slopes, flat = tilts(maps)
        positions = dies_table(GRID)
        tilt = place_tilts(index, slopes, positions)
        for d, (r, c) in GRID.items():
            self.assertAlmostEqual(tilt.at[d, "row"], 0.5 * (r - CENTRE[0]), delta=0.1, msg=d)
            self.assertAlmostEqual(tilt.at[d, "col"], 0.5 * (c - CENTRE[1]), delta=0.1, msg=d)
        outward = tilt["outward"].drop(13)   # die 13, at the centre, has no outward direction
        self.assertGreater(outward.min(), 0.99)
        self.assertTrue(np.isnan(tilt.at[13, "outward"]))
        self.assertGreater(np.median(map_pairs(index, maps)[0]), 0.7)
        self.assertLess(abs(np.median(map_pairs(index, flat)[0])), 0.15)

    def test_the_same_die_in_the_other_stage_matches_itself(self):
        pre = lot(3, own=5.0, seed=4)
        post = {w: (dies, pixels.assign(baseline=pixels["baseline"] + np.random.default_rng(9).normal(
                    0, 2, len(pixels))), qinj) for w, (dies, pixels, qinj) in reversed(pre.items()) if w != "W1"}
        dies, pixels, qinj = post["W0"]
        post["W0"] = (dies, pixels[pixels["die"] > 5], qinj)   # the other stage: W2 and W0 without dies 1-5
        index, maps, _ = residual_maps(pre, "baseline")
        o_index, o_maps, _ = residual_maps(post, "baseline")
        r = same_die_r(index, maps, o_index, o_maps)
        self.assertEqual(len(r), 2 * len(GRID) - 5)
        self.assertGreater(np.median(r), 0.8)
        same, _, _ = map_pairs(index, maps)
        self.assertLess(abs(np.median(same)), 0.15)   # no likeness by place

    def test_the_qinj_means_take_the_pixels_every_wafer_injected(self):
        tables = lot(2)
        for k, (wafer, extra, toa) in enumerate((("W0", (15, 15), 900.0), ("W1", (7, 8), 50.0))):
            dies, pixels, _ = tables[wafer]
            tables[wafer] = (dies, pixels, qinj_table(COMMON + [extra], toa=200.0 + k, extra=(extra, toa),
                                                      few=(3, (5, 5))))
        values = die_values(tables).set_index(["wafer", "die"])
        self.assertEqual(values.loc[("W0", 1), "toa"], 200.0)
        self.assertEqual(values.loc[("W1", 1), "toa"], 201.0)
        self.assertTrue(np.isnan(values.loc[("W0", 3), "toa"]))   # one pixel below MIN_QINJ_HITS
        self.assertFalse(np.isnan(values.loc[("W0", 3), "baseline"]))
        self.assertEqual(injected_notes(tables), ["W0: injected (15,15) too, left out of the QInj means",
                                                  "W1: injected (7,8) too, left out of the QInj means"])

    def test_a_wafer_without_injected_pixels_empties_the_qinj_means_and_says_so(self):
        tables = lot(2)
        for wafer, eff in (("W0", 1.0), ("W1", 0.4)):
            dies, pixels, _ = tables[wafer]
            tables[wafer] = (dies, pixels, qinj_table(COMMON, eff=eff))
        self.assertEqual(injected_notes(tables), ["W1: no pixel hit in half the events of half its QInj dies",
                                                  "no pixel injected on every wafer with QInj data: CAL, TOA and TOT "
                                                  "left out"])
        self.assertEqual(measured(die_values(tables)), ["baseline", "noise_width"])


@unittest.skipUnless(HAVE_PLOTS, "needs numpy, pandas, pyarrow and matplotlib (the wafer-daq venv)")
class PlotPositionsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.map_csv = self.root / "wafer_map.csv"
        write_map(self.map_csv, GRID)

    def tearDown(self):
        self.tmp.cleanup()

    def wafer(self, name, stage, pixels, seed, qinj=None):
        """The runs of wafer `name`: a calibration of `pixels` on every die
        and, with `qinj`, 12 QInj events hitting those pixels (after a
        first .nem file, which the reader leaves out)."""
        rng = np.random.default_rng(seed)
        place = np.random.default_rng(0).normal(0, 5, (26, 16, 16))
        for d, (r, c) in GRID.items():
            base = pd.DataFrame([{"row": pr, "col": pc, "baseline": int(round(500 + place[d, pr, pc]
                                                                              + rng.normal(0, 1))),
                                  "noise_width": 6, "timestamp": "2026-09-22 15:00:05", "chip_name": "0x60"}
                                 for pr, pc in pixels])
            extra = {"qinj": True, "qinj_pixels": [list(p) for p in qinj]} if qinj else {}
            record = summary(d, r, c, batch="B1", wafer=name, wafer_stage=stage, fullscan=len(pixels) == 256,
                             power_on=f"2026-09-22 15:{d:02d}:00.000", **extra)
            write_run(self.root / "BatchID_X_Name_B1" / f"WaferID_X_Name_{name}" / stage, d, 1, record,
                      run_power(), base, nem={"qinj": [event(qinj) * 2, event(qinj) * 12]} if qinj else None)

    def run_positions(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = main(["--path", str(self.root), "--waferMap", str(self.map_csv), "--out", str(self.root / "out"),
                       *argv])
        return rc, out.getvalue()

    def test_the_tables_and_figures_of_a_lot(self):
        for k, name in enumerate(("A1", "A2", "A3")):
            self.wafer(name, "pre_ubm", FULL_PIXELS, seed=k)
        self.wafer("A1", "post_ubm", FULL_PIXELS, seed=7)
        self.wafer("Q1", "pre_ubm", QUICK_PIXELS, seed=8)
        rc, printed = self.run_positions("--stage", "pre_ubm", "--lot", "B1", "--no-qinj")
        self.assertEqual(rc, 0, printed)
        self.assertIn("Q1 left out: no full scan or QInj data in pre_ubm", printed)
        self.assertIn("same die other stage", printed)
        self.assertIn("same-place map r with the tilt taken out", printed)
        out = self.root / "out"
        for name in ("positions.csv", "position_maps.png", "position_pairs.png", "pixel_patterns.png"):
            self.assertTrue((out / f"B1_pre_ubm_{name}").is_file(), name)
        table = pd.read_csv(out / "B1_pre_ubm_positions.csv")
        self.assertEqual(sorted(table["wafer"].unique()), ["A1", "A2", "A3"])
        self.assertEqual(len(table), 3 * len(GRID))

    def test_the_qinj_path_takes_the_common_pixels_and_keeps_a_wafer_with_qinj_data_only(self):
        self.wafer("A1", "pre_ubm", FULL_PIXELS, seed=0, qinj=QUICK_PIXELS)
        self.wafer("A2", "pre_ubm", FULL_PIXELS, seed=1, qinj=QUICK_PIXELS)
        self.wafer("A3", "pre_ubm", QUICK_PIXELS, seed=2, qinj=QUICK_PIXELS[:8] + [(7, 8)])
        rc, printed = self.run_positions("--stage", "pre_ubm", "--lot", "B1")
        self.assertEqual(rc, 0, printed)
        self.assertNotIn("left out: no full scan", printed)
        self.assertIn("A1: injected (15,15) too, left out of the QInj means", printed)
        self.assertIn("A3: injected (7,8) too, left out of the QInj means", printed)
        self.assertIn("TOA mean", printed)
        table = pd.read_csv(self.root / "out" / "B1_pre_ubm_positions.csv")
        a3 = table[table["wafer"] == "A3"]
        self.assertEqual(len(a3), len(GRID))
        self.assertTrue(a3["toa"].notna().all() and a3["baseline"].isna().all())

    def test_a_lot_needs_two_wafers_with_data(self):
        self.wafer("A1", "pre_ubm", FULL_PIXELS, seed=0)
        self.wafer("Q1", "pre_ubm", QUICK_PIXELS, seed=1)
        rc, printed = self.run_positions("--stage", "pre_ubm", "--lot", "B1", "--no-qinj")
        self.assertEqual(rc, 2)
        self.assertIn("1 wafer with data in pre_ubm; a comparison needs two", printed)


if __name__ == "__main__":
    unittest.main()
