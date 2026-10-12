"""Save the dome templates (DoG patches) of die 64 of N62H30_02C7 so every wafer is matched with the same templates.

python templates.py   -> templates.npz next to this script (oct, pix); needs out/N62H30_02C7_peri/sites_pix_die064.csv
"""
import numpy as np

from census import HT, PADW, dogp, patch
from ets import ETS
from sites import PIX_GUESS, TPL_UV, corner, die_image
from wafer_files import TEMPLATES, image_path

w, d = "N62H30_02C7", 64
e = ETS(image_path(w))
X, Y = corner(w, d)
g = die_image(e, X, Y).mean(2).astype(np.float32)
pix = np.loadtxt("out/%s_peri/sites_pix_die%03d.csv" % (w, d), delimiter=",", skiprows=1)
pix = pix[pix[:, 2] >= 0.7]
up, vp, _ = pix[np.argmin(np.hypot(pix[:, 0] - PIX_GUESS[0], pix[:, 1] - PIX_GUESS[1]))]
out = {}
for k, (u, v) in (("oct", TPL_UV), ("pix", (int(up), int(vp)))):
    out[k] = dogp(patch(g, u, v, HT + PADW))[PADW:-PADW, PADW:-PADW]
np.savez(TEMPLATES, **out)
print({k: v.shape for k, v in out.items()}, "pix at", up, vp)
