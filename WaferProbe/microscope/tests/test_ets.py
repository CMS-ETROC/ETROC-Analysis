"""ets.ETS on small .ets files written in the cellSens layout (SIS header, ETS header, chunk table, raw tiles)."""
import os
import struct
import tempfile
import unittest

import numpy as np

from ets import ETS

TX, TY, C = 4, 3, 3  # tile width, height, channels


def write_ets(path, levels, mids=((0,),), skip=()):
    """levels: {level: uint8 image (H, W, C)}, H and W multiples of the tile size. Every middle coordinate tuple in
    mids gets a copy of the image plus 10 x its index in mids; tile keys in skip are left out."""
    ndim = 3 + len(mids[0])
    tiles = []
    for L, img in levels.items():
        for k, mid in enumerate(mids):
            for ty in range(img.shape[0] // TY):
                for tx in range(img.shape[1] // TX):
                    key = (tx, ty) + tuple(mid) + (L,)
                    if key not in skip:
                        t = img[ty * TY:(ty + 1) * TY, tx * TX:(tx + 1) * TX] + 10 * k
                        tiles.append((key, t.astype(np.uint8).tobytes()))
    tiles.reverse()  # the files keep tiles in scan order, not in position order
    add = b"ETS\0" + struct.pack("<9i", 0x30001, 2, C, 4, 0, 90, TX, TY, 1)
    chunk_off = 64 + len(add)
    off = chunk_off + (4 * ndim + 20) * len(tiles)
    table = blobs = b""
    for key, b in tiles:
        table += struct.pack("<i%diqii" % ndim, 0, *key, off, len(b), 0)
        blobs += b
        off += len(b)
    head = struct.pack("<4siiiqiiqii", b"SIS\0", 64, 2, ndim, 64, len(add), 0, chunk_off, len(tiles), 0)
    with open(path, "wb") as f:
        f.write(head.ljust(64, b"\0") + add + table + blobs)


class TestETS(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "frame_t_0.ets")
        rng = np.random.default_rng(1)
        self.img0 = rng.integers(0, 200, (3 * TY, 3 * TX, C), dtype=np.uint8)
        self.img1 = rng.integers(0, 200, (2 * TY, 2 * TX, C), dtype=np.uint8)

    def tearDown(self):
        self.dir.cleanup()

    def open(self, **kw):
        write_ets(self.path, {0: self.img0, 1: self.img1}, **kw)
        e = ETS(self.path)
        self.addCleanup(e.f.close)
        return e

    def test_header_and_levels(self):
        e = self.open()
        self.assertEqual((e.ndim, e.tile_x, e.tile_y, e.size_c, e.compression), (4, TX, TY, C, 0))
        self.assertEqual(e.levels, [0, 1])
        self.assertEqual(e.level_grid(0), (9, (0, 2), (0, 2), [(0,)]))

    def test_whole_levels(self):
        e = self.open()
        np.testing.assert_array_equal(e.read(0), self.img0)
        np.testing.assert_array_equal(e.read(1), self.img1)

    def test_box_across_tile_edges(self):
        e = self.open()
        np.testing.assert_array_equal(e.read(0, 1, 2, 11, 8), self.img0[2:8, 1:11])

    def test_box_cut_at_the_tile_grid(self):
        e = self.open()
        np.testing.assert_array_equal(e.read(0, 5, 4, 100, 100), self.img0[4:, 5:])

    def test_missing_tile_reads_white(self):
        e = self.open(skip={(1, 1, 0, 0)})
        want = self.img0.copy()
        want[TY:2 * TY, TX:2 * TX] = 255
        np.testing.assert_array_equal(e.read(0), want)

    def test_two_middle_coordinates(self):
        e = self.open(mids=((0, 0), (1, 0)))
        self.assertEqual(e.ndim, 5)
        np.testing.assert_array_equal(e.read(0), self.img0)
        np.testing.assert_array_equal(e.read(0, mid=(1, 0)), self.img0 + 10)


if __name__ == "__main__":
    unittest.main()
