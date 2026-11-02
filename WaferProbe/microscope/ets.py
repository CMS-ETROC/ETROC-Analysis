"""Minimal reader for Olympus cellSens .ets tile files (layout as in Bio-Formats CellSensReader).

python ets.py info <frame_t_0.ets>
python ets.py level <frame_t_0.ets> <level> <out.png> [x0 y0 x1 y1]   (pixel box in that level)

A tile's coordinates are (x, y, <middle coordinates>, level); ndim counts them all. The middle ones index extra
image planes and are 0 in every file seen so far: one for the wafer scans (ndim 4), two for the
gel-pack scans (ndim 5). mid=None reads all-zero middle coordinates for either.
"""
import struct
import sys

import numpy as np


class ETS:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        h = self.f.read(64)
        magic, hsize, version, ndim, add_off, add_size, _r, chunk_off, nchunks, _r2 = struct.unpack("<4siiiqii q i i", h[:48])
        assert magic[:3] == b"SIS", magic
        self.ndim, self.nchunks = ndim, nchunks
        self.f.seek(add_off)
        a = self.f.read(add_size)
        assert a[:3] == b"ETS", a[:4]
        (self.version, self.pixel_type, self.size_c, self.colorspace, self.compression,
         self.quality, self.tile_x, self.tile_y, self.tile_z) = struct.unpack("<9i", a[4:40])
        self.f.seek(chunk_off)
        rec = 4 + 4 * ndim + 8 + 4 + 4
        raw = self.f.read(rec * nchunks)
        self.tiles = {}
        self.nbytes = set()
        for i in range(nchunks):
            r = raw[i * rec:(i + 1) * rec]
            coord = struct.unpack("<%di" % ndim, r[4:4 + 4 * ndim])
            off, nb = struct.unpack("<qi", r[4 + 4 * ndim:4 + 4 * ndim + 12])
            self.tiles[coord] = (off, nb)
            self.nbytes.add(nb)
        self.levels = sorted({c[-1] for c in self.tiles})

    def level_grid(self, level):
        cs = [c for c in self.tiles if c[-1] == level]
        xs = [c[0] for c in cs]
        ys = [c[1] for c in cs]
        return len(cs), (min(xs), max(xs)), (min(ys), max(ys)), sorted({c[2:-1] for c in cs})

    def tile(self, x, y, level, mid=None):
        if mid is None:
            mid = (0,) * (self.ndim - 3)
        key = (x, y) + tuple(mid) + (level,)
        if key not in self.tiles:
            return None
        off, nb = self.tiles[key]
        self.f.seek(off)
        buf = self.f.read(nb)
        if self.compression != 0:
            raise NotImplementedError("compression %d" % self.compression)
        return np.frombuffer(buf, np.uint8).reshape(self.tile_y, self.tile_x, self.size_c)

    def read(self, level, x0=0, y0=0, x1=None, y1=None, mid=None):
        """uint8 array (y1 - y0, x1 - x0, size_c) of one level, x0, y0 >= 0; the box is cut at the level's tile grid,
        and missing tiles read as 255"""
        n, (tx0, tx1), (ty0, ty1), _ = self.level_grid(level)
        W, H = (tx1 + 1) * self.tile_x, (ty1 + 1) * self.tile_y
        x1 = W if x1 is None else min(x1, W)
        y1 = H if y1 is None else min(y1, H)
        out = np.full((y1 - y0, x1 - x0, self.size_c), 255, np.uint8)
        for ty in range(y0 // self.tile_y, (y1 - 1) // self.tile_y + 1):
            for tx in range(x0 // self.tile_x, (x1 - 1) // self.tile_x + 1):
                t = self.tile(tx, ty, level, mid)
                if t is None:
                    continue
                gx, gy = tx * self.tile_x, ty * self.tile_y
                sx0, sy0 = max(x0, gx), max(y0, gy)
                sx1, sy1 = min(x1, gx + self.tile_x), min(y1, gy + self.tile_y)
                out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = t[sy0 - gy:sy1 - gy, sx0 - gx:sx1 - gx]
        return out


def main():
    cmd, path = sys.argv[1], sys.argv[2]
    e = ETS(path)
    if cmd == "info":
        print("version %#x pixel_type %d size_c %d colorspace %d compression %d quality %d tile %dx%dx%d ndim %d chunks %d nbytes %s"
              % (e.version, e.pixel_type, e.size_c, e.colorspace, e.compression, e.quality,
                 e.tile_x, e.tile_y, e.tile_z, e.ndim, e.nchunks, sorted(e.nbytes)))
        for L in e.levels:
            n, xr, yr, mids = e.level_grid(L)
            print("level %d: %d tiles, x %s, y %s, px %d x %d, mid coords %s" % (
                L, n, xr, yr, (xr[1] + 1) * e.tile_x, (yr[1] + 1) * e.tile_y, mids[:5]))
    elif cmd == "level":
        from PIL import Image
        L = int(sys.argv[3])
        box = [int(v) for v in sys.argv[5:9]] if len(sys.argv) > 8 else []
        img = e.read(L, *box)
        Image.fromarray(img).save(sys.argv[4])
        print("wrote", sys.argv[4], img.shape)


if __name__ == "__main__":
    main()
