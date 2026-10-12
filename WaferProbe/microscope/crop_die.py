"""Crop dies from a wafer .ets using the L4 grid fit.

python crop_die.py <wafer> <level> <margin_px_L4> <die> [<die> ...]   -> out/<wafer>_dieNNN_L<level>.png
"""
import os
import sys

from PIL import Image

from ets import ETS
from wafer_files import grid_fit, image_path, wafer_map

w, L, m = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
dies = [int(x) for x in sys.argv[4:]]
g = grid_fit(w)
mp = wafer_map()
e = ETS(image_path(w))
s = 2 ** (4 - L)
os.makedirs("out", exist_ok=True)
for d in dies:
    r, c = mp[d]
    y0 = (g["phy"] + r * g["Py"] - m) * s; y1 = (g["phy"] + (r + 1) * g["Py"] + m) * s
    x0 = (g["phx"] + c * g["Px"] - m) * s; x1 = (g["phx"] + (c + 1) * g["Px"] + m) * s
    img = e.read(L, int(max(0, x0)), int(max(0, y0)), int(x1), int(y1))
    out = "out/%s_die%03d_L%d.png" % (w, d, L)
    Image.fromarray(img).save(out)
    print(out, img.shape, "box", int(x0), int(y0), int(x1), int(y1))
