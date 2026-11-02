"""gelpack.analyse_level4 and gelpack.chip_domes on synthetic images (needs numpy, scipy and Pillow)."""
import unittest

import numpy as np

import gelpack

ROT = 0.4                                       # degrees, like the scanned packs (+0.25 to +0.5)


def bump4(kind):
    """15 x 15 level-4 grey offset of a solder bump (highlight upper left, shadow lower right, as in the scans) or of
    a bare pad (dark disk, light rim)"""
    yy, xx = np.mgrid[-7:8, -7:8]
    if kind == "bump":
        top = np.exp(-((xx + 1) ** 2 + (yy + 1) ** 2) / (2 * 1.2 ** 2))
        shadow = np.exp(-((xx - 2) ** 2 + (yy - 2) ** 2) / (2 * 1.6 ** 2))
        return 90 * top - 60 * shadow
    r = np.hypot(xx, yy)
    return -80 * (r <= 4) + 30 * ((r > 4) & (r <= 5.5))


def level4_pack(kinds, empty=()):
    """chips side by side on a dark gel pack; each a 16 x 16 grid plus the bottom row, rotated by ROT"""
    rng = np.random.default_rng(1)
    g = np.full((1300, 100 + 1250 * len(kinds)), 40.0)
    th = np.radians(ROT)
    for c, kind in enumerate(kinds):
        x0, y0 = 100 + 1250 * c, 100
        g[y0:y0 + 1100, x0:x0 + 1030] = 150
        cx, cy = x0 + 515, y0 + 550
        sites = [(j, i, x0 + 35 + gelpack.PITCH * i, y0 + 45 + gelpack.PITCH * j) for j in range(16) for i in range(16)]
        sites += [(16, i, x0 + 34 + gelpack.PITCH * i, y0 + 45 + gelpack.PITCH * 15 + 35) for i in range(16)]
        for j, i, x, y in sites:
            if (c, j, i) in empty:
                continue
            u, w = x - cx, y - cy
            x, y = int(round(cx + np.cos(th) * u - np.sin(th) * w)), int(round(cy + np.sin(th) * u + np.cos(th) * w))
            g[y - 7:y + 8, x - 7:x + 8] += bump4(kind)
    g += rng.normal(0, 3, g.shape)
    return np.repeat(np.clip(g, 0, 255).astype(np.uint8)[..., None], 3, axis=2)


class Level4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chips = gelpack.analyse_level4(level4_pack(["bump", "bump", "pad"], empty={(0, 5, 9)}))

    def test_chips_and_angle(self):
        self.assertEqual(len(self.chips), 3)
        for c in self.chips:
            self.assertAlmostEqual(c["rot"], ROT, delta=0.06)

    def test_empty_site(self):
        S = self.chips[0]["S"]
        self.assertLess(S[5, 9], gelpack.EMPTY)
        others = np.delete(S.ravel(), 5 * 16 + 9)
        self.assertGreater(others.min(), 0.5)
        self.assertGreater(self.chips[1]["S"].min(), 0.5)
        self.assertGreater(self.chips[0]["BS"].min(), 0.5)

    def test_unbumped_chip(self):
        self.assertEqual([c["unbumped"] for c in self.chips], [False, False, True])


class Level4Edges(unittest.TestCase):
    def test_too_few_chips_to_judge(self):
        self.assertIsNone(gelpack.analyse_level4(level4_pack(["pad"]))[0]["unbumped"])

    def test_chip_cut_by_the_edge(self):
        rgb = level4_pack(["bump", "bump"])[:, :100 + 1250 + 950]
        chips = gelpack.analyse_level4(rgb)
        self.assertEqual([c["box"][2] < 1250 for c in chips], [True])

    def test_no_chip(self):
        self.assertEqual(gelpack.analyse_level4(np.full((800, 800, 3), 60, np.uint8)), [])

    def test_chip_order(self):
        """2 x 2 chips, the right chip of the top row a little higher than the left one"""
        g = np.full((2900, 2700), 40.0)
        for y0, x0 in [(296, 100), (276, 1350), (1600, 100), (1610, 1350)]:
            g[y0:y0 + 1100, x0:x0 + 1030] = 150
        boxes = gelpack.find_chips(g)                     # (y0, y1, x0, x1): top left, top right, bottom left, ...
        self.assertEqual([(b[0] < 1000, b[2] < 1000) for b in boxes],
                         [(True, True), (True, False), (False, True), (False, False)])


class Level1(unittest.TestCase):
    def test_discoloured_top(self):
        rng = np.random.default_rng(2)
        n, red, empty = 20, 7, 12
        offsets = {k: tuple(rng.integers(-15, 16, 2)) for k in range(n)}
        sites = [("a", 0, k, 100.0 * k, 50.0) for k in range(n)]
        index = {(int(round(x4 * 8)), int(round(y4 * 8))): k for _, _, k, x4, y4 in sites}
        size = 2 * gelpack.W1 + 1
        yy, xx = np.mgrid[:size, :size] - gelpack.W1

        def read_window(x, y):
            k = index[(x, y)]
            dx, dy = offsets[k]
            r = np.hypot(xx - dx, yy - dy)
            img = np.empty((size, size, 3))
            img[:] = (150, 140, 170)
            if k == empty:
                return np.clip(img + rng.normal(0, 4, img.shape), 0, 255)
            img[r <= 22] = (110, 115, 135)
            top = np.exp(-r ** 2 / (2 * 4 ** 2))[..., None]
            img += top * (np.array([120, 60, 0]) if k == red else np.array([110, 120, 120]))
            return np.clip(img + rng.normal(0, 4, img.shape), 0, 255)

        rows, _ = gelpack.chip_domes(read_window, sites)
        self.assertEqual([rows[k][2] for k in gelpack.discoloured(rows)], [red])
        self.assertLess(rows[empty][7], gelpack.EMPTY)
        for kind, j, i, x1, y1, dx, dy, ncc, br, patch in rows:   # re-centred to within a pixel
            if i == empty:
                continue
            self.assertLessEqual(max(abs(dx - offsets[i][0]), abs(dy - offsets[i][1])), 1)
            self.assertGreater(ncc, 0.8)


if __name__ == "__main__":
    unittest.main()
