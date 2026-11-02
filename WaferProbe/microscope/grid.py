"""Fit the die lattice of wafer images at level 4 and store it in grids_L4.json.

python grid.py <wafer> [<wafer> ...]

The wafer is the bright disk (grey > 90): its bounding box gives the centre cx, cy and the radius R. Scribe streets
are dark lines, so the profile of a central band across the rows (columns), high-passed with a 15 px running mean,
has a minimum every die pitch. Pitches within +-6 px of the nominal ones (0.1 px steps) and every integer phase
within one pitch are tried; the lowest mean of the profile at the street positions wins. The die at map row r,
column c starts at x = phx + c * Px, y = phy + r * Py (level-4 px; times 16 for level 0).
The phases are only known modulo one pitch: look at a few dies with crop_die.py before using a new fit.
"""
import json
import sys

import numpy as np

from ets import ETS
from wafer_files import GRIDS, image_path

NOMINAL_PY, NOMINAL_PX = 345.6, 316.8  # starting pitches in level-4 px; fits so far 345.4-345.6 and 315.8-315.9


def hp(p, k=15):
    return p - np.convolve(p, np.ones(k) / k, mode="same")


def fit_streets(h, p0, lo, hi):
    """(mean profile at the streets, pitch, phase) with the lowest mean, streets counted between lo and hi"""
    best = None
    for P in np.arange(p0 - 6, p0 + 6, 0.1):
        for ph in range(int(P)):
            pos = np.arange(ph, len(h), P).astype(int)
            pos = pos[(pos > lo) & (pos < hi)]
            s = h[pos].mean()
            if best is None or s < best[0]:
                best = (s, P, ph)
    return best


def fit_wafer(g):
    """grid fit of one level-4 grey image"""
    ys, xs = np.nonzero(g > 90)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    r = ((xs.max() - xs.min()) + (ys.max() - ys.min())) / 4
    hy = hp(g[:, int(cx - 1200):int(cx + 1200)].mean(axis=1))
    hx = hp(g[int(cy - 1200):int(cy + 1200), :].mean(axis=0))
    _, Py, phy = fit_streets(hy, NOMINAL_PY, int(cy - r + 100), int(cy + r - 100))
    _, Px, phx = fit_streets(hx, NOMINAL_PX, int(cx - r + 100), int(cx + r - 100))
    return {k: float(v) for k, v in dict(cx=cx, cy=cy, R=r, Py=Py, phy=phy, Px=Px, phx=phx).items()}


def main():
    with open(GRIDS) as f:
        grids = json.load(f)
    for w in sys.argv[1:]:
        grids[w] = fit_wafer(ETS(image_path(w)).read(4).astype(float).mean(axis=2))
        print(w, "  ".join("%s %.1f" % kv for kv in grids[w].items()), flush=True)
        with open(GRIDS, "w") as f:
            json.dump(grids, f, indent=1)


if __name__ == "__main__":
    main()
