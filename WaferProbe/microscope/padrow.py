"""Measure the pad row of one die at level 0: rectangular-pad x positions and whether each pad has its octagon
near (505 um) or far (723 um), to pin the chip-to-image orientation against Table 22 of the ETROC2 manual.

python padrow.py <wafer> <die>   (die corner from out/<wafer>_peri/dies.csv)
"""
import csv
import sys

import numpy as np
from scipy import ndimage
from skimage.feature import match_template

from anomaly import REGIONS
from ets import ETS
from wafer_files import image_path

w, d = sys.argv[1], int(sys.argv[2])
L, dx0, dy0, W, H = REGIONS["peri"]
m = {int(r["die"]): r for r in csv.DictReader(open("out/%s_peri/dies.csv" % w))}[d]
X = int(m["x0"]) - int(m["sx"]) - dx0
Y = int(m["y0"]) - int(m["sy"]) - dy0
e = ETS(image_path(w))
x_lo, y_lo = X - 100, Y + 60
g = e.read(0, x_lo, y_lo, X + 5150, Y + 320).mean(2).astype(np.float32)

gd = ndimage.gaussian_filter(g, 1.0) - ndimage.gaussian_filter(g, 6.0)
tpl = gd[198 - 13:198 + 14, 113 - 13:113 + 14]
ncc = match_template(gd, tpl, pad_input=True)
cand = []
for yb, tag in ((145, "N"), (198, "F")):
    band = ncc[yb - 6:yb + 7].max(0)
    pk = np.where((band == ndimage.maximum_filter1d(band, 9)) & (band > 0.3))[0]
    cand += [(band[x], x, tag) for x in pk]
octs = []
for v, x, tag in sorted(cand, reverse=True):  # joint non-max suppression over both rows: side lobes sit ~10 px away
    if all(abs(x - o[1]) > 20 for o in octs):
        octs.append((v, x, tag))
octs = sorted((x + x_lo - X, tag, v) for v, x, tag in octs if v > 0.45)
print("die", d, "X Y", X, Y, "octagons", len(octs))
print("".join(o[1] for o in octs))
for i, (x, tag, v) in enumerate(octs):
    print("%3d x=%6d %s ncc=%.2f dx=%d" % (i, x, tag, v, x - octs[i - 1][0] if i else 0))
