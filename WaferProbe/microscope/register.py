"""Register every die of a wafer on its pad row (unique along the die, unlike the 1.3 mm-periodic bump array).

python register.py <wafer> <ref_die>
Reads out/<wafer>_peri/dies.csv (anomaly.py output: grades, and the corner of the reference die, which defines the
die frame of the site lists), matches the reference die's pad-row strip (with both scribe streets) at level 2 in a +-600 x +-300 px (L0) window
around each die's predicted corner, refines at level 0 (+-12 px), and rewrites dies.csv with the new corners
(x0 = X - 400, y0 = Y - 60, sx = sy = 0, so sites.corner() gives X, Y). The anomaly.py version is kept as
dies_anomaly.csv; reg_peak (L2) and reg_peak0 (L0) are added.
"""
import csv
import os
import shutil
import sys

import numpy as np
from skimage.feature import match_template

from anomaly import REGIONS
from ets import ETS
from wafer_files import grid_fit, image_path

U2, V2 = -150, -100          # template at L2: u -150..5150, v -100..400 (L0) from the die corner, so both scribe
W2, H2 = 1325, 125           # streets break the pad-pitch periodicity (u 0..4400 alone locked 4-16 pitches off)
SU, SV = 600, 300           # search half-window at L0
TU0, TU1, TV0, TV1 = 0, 600, 60, 330  # L0 refinement template: left end of the pad row


def main():
    w, ref = sys.argv[1], int(sys.argv[2])
    L, dx0, dy0, W, H = REGIONS["peri"]
    outd = "out/%s_peri/" % w
    src = outd + "dies_anomaly.csv"
    if not os.path.exists(src):
        shutil.copy(outd + "dies.csv", src)
    rows = list(csv.DictReader(open(src)))
    meta = {int(r["die"]): r for r in rows}
    g = grid_fit(w)
    e = ETS(image_path(w))

    def grid(r, c):
        return (g["phx"] + c * g["Px"]) * 16, (g["phy"] + r * g["Py"]) * 16

    m = meta[ref]
    Xr, Yr = int(m["x0"]) - int(m["sx"]) - dx0, int(m["y0"]) - int(m["sy"]) - dy0
    gxr, gyr = grid(int(m["row"]), int(m["col"]))
    t2 = e.read(2, (Xr + U2) // 4, (Yr + V2) // 4, (Xr + U2) // 4 + W2, (Yr + V2) // 4 + H2).mean(2).astype(np.float32)
    t0 = e.read(0, Xr + TU0, Yr + TV0, Xr + TU1, Yr + TV1).mean(2).astype(np.float32)
    for d, m in sorted(meta.items()):
        gx, gy = grid(int(m["row"]), int(m["col"]))
        # prediction: grid step from the reference die plus the affine lattice offsets of anomaly.py (py, px)
        px = Xr + int(round(gx - gxr)) - (int(m["px"]) - int(meta[ref]["px"]))
        py = Yr + int(round(gy - gyr)) - (int(m["py"]) - int(meta[ref]["py"]))
        bx, by = (px + U2 - SU) // 4, (py + V2 - SV) // 4
        box = e.read(2, bx, by, bx + 2 * SU // 4 + W2, by + 2 * SV // 4 + H2)
        if box.shape[:2] != (2 * SV // 4 + H2, 2 * SU // 4 + W2) or (box.min(2) > 235).mean() > 0.3:
            m.update(reg_peak="", reg_peak0="")
            continue
        c2 = match_template(box.mean(2).astype(np.float32), t2)
        iv, iu = np.unravel_index(np.argmax(c2), c2.shape)
        X, Y = (bx + iu) * 4 - U2, (by + iv) * 4 - V2
        b0 = e.read(0, X + TU0 - 12, Y + TV0 - 12, X + TU1 + 12, Y + TV1 + 12).mean(2).astype(np.float32)
        c0 = match_template(b0, t0)
        jv, ju = np.unravel_index(np.argmax(c0), c0.shape)
        X, Y = X + ju - 12, Y + jv - 12
        m.update(x0=X + dx0, y0=Y + dy0, sx=0, sy=0, reg_peak=round(float(c2.max()), 3), reg_peak0=round(float(c0.max()), 3))
        print(d, X, Y, "L2 %.2f L0 %.2f" % (c2.max(), c0.max()), flush=True)
    fields = list(rows[0].keys()) + ["reg_peak", "reg_peak0"]
    with open(outd + os.environ.get("REG_OUT", "dies.csv"), "w", newline="") as f:  # REG_OUT: a test run
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    main()
