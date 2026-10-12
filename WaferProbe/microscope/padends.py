"""L0 crops of the pad row at both ends of a die (image frame), to fix the chip-to-image orientation.

python padends.py <wafer> <die> [<die> ...]   -> out/<wafer>_peri/padends_dieNNN.png
"""
import csv
import sys

import numpy as np
from PIL import Image

from anomaly import REGIONS
from ets import ETS
from wafer_files import image_path

w = sys.argv[1]
L, dx0, dy0, W, H = REGIONS["peri"]
outd = "out/%s_peri" % w
meta = {int(r["die"]): r for r in csv.DictReader(open(outd + "/dies.csv"))}
e = ETS(image_path(w))
for d in [int(x) for x in sys.argv[2:]]:
    m = meta[d]
    X = int(m["x0"]) - int(m["sx"]) - dx0
    Y = int(m["y0"]) - int(m["sy"]) - dy0
    a = e.read(0, X - 150, Y - 60, X + 1050, Y + 300)
    b = e.read(0, X + 3950, Y - 60, X + 5150, Y + 300)
    Image.fromarray(np.vstack([a, np.full((10, a.shape[1], 3), 255, np.uint8), b])).save("%s/padends_die%03d.png" % (outd, d))
    print(d, X, Y)
