"""Find every UBM dome of one kind on one die at level 0 by template matching on the DoG image.

python sites.py <wafer> <die> <oct|pix> [thr]  -> out/<wafer>_peri/sites_<kind>_dieNNN.csv (u, v, ncc) + overview PNG
python sites.py <wafer> <die> auto     -> sites_pix_dieNNN.csv and padrow_dieNNN.csv (pad, u, v) with the
                                           templates.npz of templates.py (any wafer; census.py needs both files)
Die coordinates: u, v = L0 pixels from the die corner (X, Y) of out/<wafer>_peri/dies.csv.
"""
import csv
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.feature import match_template

from anomaly import REGIONS
from ets import ETS
from wafer_files import TEMPLATES, image_path

DIE_W, DIE_H, M = 5060, 5540, 40
TPL_UV = (13, 258)  # far octagon of the leftmost main pad of die 64 of N62H30_02C7
PIX_GUESS = (760, 1032)  # a pixel bump dome of die 64 (read off the 1/4 overview)


def corner(w, d):
    L, dx0, dy0, W, H = REGIONS["peri"]
    m = {int(r["die"]): r for r in csv.DictReader(open("out/%s_peri/dies.csv" % w))}[d]
    return int(m["x0"]) - int(m["sx"]) - dx0, int(m["y0"]) - int(m["sy"]) - dy0


def die_image(e, X, Y):
    return e.read(0, X - M, Y - M, X + DIE_W + M, Y + DIE_H + M)


def dog(g):
    return ndimage.gaussian_filter(g, 1.0) - ndimage.gaussian_filter(g, 6.0)


def template(gd, g, kind):
    """oct: the far octagon of the leftmost main pad; pix: the pixel bump dome nearest PIX_GUESS (darkest point)"""
    if kind == "oct":
        u, v = TPL_UV
    else:
        u0, v0 = PIX_GUESS
        win = ndimage.gaussian_filter(g[v0 + M - 40:v0 + M + 41, u0 + M - 40:u0 + M + 41], 2)
        dv, du = np.unravel_index(np.argmin(win), win.shape)
        u, v = u0 - 40 + du, v0 - 40 + dv
        print("pixel template at", u, v)
    return gd[v + M - 13:v + M + 14, u + M - 13:u + M + 14].copy()


def find_padrow(ncc, pitch=34.2):
    """124 main-pad octagons, near row v ~ 205 and far row v ~ 258 (die coordinates), pad 124 leftmost (image = chip
    rotated 180). Octagons are found by template matching with joint non-max suppression over both rows (side lobes
    sit ~10 px away); the row is then taken as a lattice (the pitch is regular, 143 um) anchored on the leftmost
    octagon that has a neighbour one pitch to its right, and near/far alternate (majority phase). The right end of the
    row is not used as an anchor: the metal block right of pad 1 gives false matches there."""
    band = ncc[M + 180:M + 286]
    mx = ndimage.maximum_filter(band, 9)
    vs, us = np.nonzero((band == mx) & (band > 0.3))
    octs = []
    for c, u, v in sorted(zip(band[vs, us], us - M, vs + 180), reverse=True):
        if c > 0.45 and -80 < u < 4500 and all(abs(u - o[0]) > 20 for o in octs):
            octs.append((u, v, c))
    octs.sort()
    u_all = np.array([o[0] for o in octs], float)
    first = next(i for i in range(len(octs) - 1) if 30 < octs[i + 1][0] - octs[i][0] < 38)
    u0 = u_all[first]
    k = np.round((u_all - u0) / pitch)
    good = (k >= 0) & (k <= 123) & (np.abs(u_all - u0 - k * pitch) < 8)
    a_, b_ = np.polyfit(k[good], u_all[good], 1)[::-1]
    near = np.array([abs(o[1] - 205) < abs(o[1] - 258) for o in octs])
    phase = int(round(np.mean((near[good] == (k[good] % 2 == 0)))))  # 1: slot 0 is near
    res = u_all[good] - (a_ + b_ * k[good])
    print("pad row: %d octagons on the lattice, pitch %.2f px, residual rms %.1f px, slot-0 %s, agreement with "
          "alternation %.2f" % (good.sum(), b_, res.std(), "near" if phase else "far",
                                np.mean(near[good] == ((k[good] % 2 == 0) == bool(phase)))))
    return [(124 - kk, int(round(a_ + b_ * kk)), 205 if ((kk % 2 == 0) == bool(phase)) else 258) for kk in range(124)]


if __name__ == "__main__":
    w, d, kind = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    thr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.45
    e = ETS(image_path(w))
    X, Y = corner(w, d)
    img = die_image(e, X, Y)
    g = img.mean(2).astype(np.float32)
    gd = dog(g)
    if kind == "auto":
        t = np.load(TEMPLATES)
        pads = find_padrow(match_template(gd, t["oct"], pad_input=True))
        with open("out/%s_peri/padrow_die%03d.csv" % (w, d), "w") as f:
            f.write("pad,u,v\n")
            f.writelines("%d,%d,%d\n" % p for p in pads)
        ncc = match_template(gd, t["pix"], pad_input=True)
        pk = (ncc == ndimage.maximum_filter(ncc, 31)) & (ncc > 0.6)
        vs, us = np.nonzero(pk)
        with open("out/%s_peri/sites_pix_die%03d.csv" % (w, d), "w") as f:
            f.write("u,v,ncc\n")
            f.writelines("%d,%d,%.3f\n" % (u - M, v - M, ncc[v, u]) for v, u in zip(vs, us))
        c = ncc[pk]
        print("pixel-template sites: >=0.7 %d, >=0.8 %d" % ((c >= 0.7).sum(), (c >= 0.8).sum()))
        sys.exit(0)
    ncc = match_template(gd, template(gd, g, kind), pad_input=True)
    pk = (ncc == ndimage.maximum_filter(ncc, 31)) & (ncc > thr)
    vs, us = np.nonzero(pk)
    with open("out/%s_peri/sites_%s_die%03d.csv" % (w, kind, d), "w") as f:
        f.write("u,v,ncc\n")
        for v, u in zip(vs, us):
            f.write("%d,%d,%.3f\n" % (u - M, v - M, ncc[v, u]))
    print(len(us), "sites >", thr, "hist", np.histogram(ncc[pk], bins=[0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01])[0])
    im = Image.fromarray(img[::4, ::4])
    dr = ImageDraw.Draw(im)
    for v, u in zip(vs, us):
        c = (255, 0, 0) if ncc[v, u] < 0.6 else (255, 255, 0)
        dr.ellipse([u // 4 - 4, v // 4 - 4, u // 4 + 4, v // 4 + 4], outline=c)
    im.save("out/%s_peri/sites_%s_die%03d.png" % (w, kind, d))
