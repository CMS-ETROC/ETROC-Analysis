"""Per-die census of every UBM dome at level 0: main-pad octagons (124), the periphery dome row, pixel bumps (256).

python census.py <wafer> <ref_die>
Sites come from the reference die (sites_pix and padrow CSVs of sites.py); every die is read at
its registered corner (out/<wafer>_peri/dies.csv) and each site is searched +-R px for the best template match.
Per site: ncc, shift, mean RGB of the dome disk (r <= 6) and of a ring around it (9 <= r <= 12).
Then each (die, site) is compared with the same site on the dies that PASSED before and after UBM.
Outputs in out/<wafer>_peri/: census.npz (all features), census_sites.csv, census_dies.csv, census_flags.csv.
"""
import csv
import os
import sys

import numpy as np
from scipy import ndimage
from skimage.feature import match_template

from ets import ETS
from sites import M, corner, die_image
from wafer_files import TEMPLATES, image_path

HT, PADW = 13, 24  # template half size, DoG context
# search radius: octagons sit 34 px apart (+-12 is unambiguous); pixel domes 310 px apart, and deep in the die the
# mosaic places them up to ~25 px off the position predicted from the pad-row registration
RADIUS = {"oct": 12, "pix": 40, "prow": 40}


def load_sites(w, ref):
    """(u, v, kind) in die coordinates of the reference die"""
    d = "out/%s_peri/" % w
    pix = np.loadtxt(d + "sites_pix_die%03d.csv" % ref, delimiter=",", skiprows=1, ndmin=2)
    s = []
    for u, v, c in pix:
        if c >= 0.7 and 500 <= v <= 700 and 0 <= u < 5060:
            s.append((int(u), int(v), "prow"))
        elif c >= 0.7 and 700 < v <= 5450 and 0 <= u < 5060:
            s.append((int(u), int(v), "pix"))
    for r in csv.DictReader(open(d + "padrow_die%03d.csv" % ref)):
        s.append((int(r["u"]), int(r["v"]), "oct%s" % r["pad"]))
    return s


def patch(img, u, v, h):
    return img[v + M - h:v + M + h + 1, u + M - h:u + M + h + 1]


def dogp(p):
    return ndimage.gaussian_filter(p, 1.0) - ndimage.gaussian_filter(p, 6.0)


def main():
    w, ref = sys.argv[1], int(sys.argv[2])
    e = ETS(image_path(w))
    sites = load_sites(w, ref)
    dies_meta = {int(r["die"]): r for r in csv.DictReader(open("out/%s_peri/dies.csv" % w))}
    dies = sorted(dies_meta)
    if os.environ.get("DIES"):  # test on a subset
        dies = [d for d in dies if d in {int(x) for x in os.environ["DIES"].split(",")}]
    yy, xx = np.mgrid[-HT:HT + 1, -HT:HT + 1]
    rr = np.hypot(yy, xx)
    disk, ring = rr <= 6, (rr >= 9) & (rr <= 12)

    # the same templates for every wafer (die 64 of N62H30_02C7, templates.py)
    t = np.load(TEMPLATES)
    tpl = {"oct": t["oct"], "pix": t["pix"], "prow": t["pix"]}

    NF = 10  # ncc, du, dv, disk R G B, ring R G B
    F = np.full((len(dies), len(sites), NF), np.nan, np.float32)
    for i, d in enumerate(dies):
        X, Y = corner(w, d)
        img = die_image(e, X, Y)
        g = img.mean(2).astype(np.float32)
        white = (img.min(2) > 235)
        for j, (u, v, k) in enumerate(sites):
            kk = "oct" if k.startswith("oct") else k
            R = RADIUS[kk]
            p = patch(g, u, v, HT + R + PADW)
            if p.shape != (2 * (HT + R + PADW) + 1,) * 2 or patch(white, u, v, HT).mean() > 0.2:
                continue  # off the image or off the wafer
            c = match_template(dogp(p)[PADW:-PADW, PADW:-PADW], tpl[kk])
            dv, du = np.unravel_index(np.argmax(c), c.shape)
            du, dv = du - R, dv - R
            q = patch(img, u + du, v + dv, HT).astype(np.float32)
            F[i, j, :3] = c.max(), du, dv
            F[i, j, 3:6] = q[disk].mean(0)
            F[i, j, 6:9] = q[ring].mean(0)
        print("die", d, "median ncc %.2f" % np.nanmedian(F[i, :, 0]), flush=True)

    outd = "out/%s_peri/" % w
    np.savez_compressed(outd + "census.npz", F=F, dies=np.array(dies), sites=np.array(sites, dtype=object))
    with open(outd + "census_sites.csv", "w") as f:
        f.write("site,u,v,kind\n")
        for j, (u, v, k) in enumerate(sites):
            f.write("%d,%d,%d,%s\n" % (j, u, v, k))

    # comparison with the dies that passed before and after UBM
    ok = np.array([dies_meta[d]["pre"] == "PASSED" and dies_meta[d]["post"] == "PASSED" for d in dies])
    ncc = F[..., 0]
    # shift relative to the die's local field-of-view shift: median over sites within 600 px
    uv = np.array([(u, v) for u, v, k in sites], float)
    near = [np.nonzero(np.hypot(*(uv - uv[j]).T) < 600)[0] for j in range(len(sites))]
    rel = np.full(ncc.shape + (2,), np.nan, np.float32)
    for j in range(len(sites)):
        rel[:, j] = F[:, j, 1:3] - np.nanmedian(F[:, near[j], 1:3], axis=1)
    lum = F[..., 3:6].mean(-1) / np.maximum(F[..., 6:9].mean(-1), 1)  # dome / surround brightness
    col = F[..., 3:6] / np.maximum(F[..., 3:6].sum(-1, keepdims=True), 1)  # dome chromaticity
    def z(a):
        m = np.nanmedian(a[ok], axis=0)
        s = 1.4826 * np.nanmedian(np.abs(a[ok] - m), axis=0)
        return (a - m) / np.maximum(s, 1e-3)
    zn, zl = z(ncc), z(lum)
    zc = np.stack([z(col[..., c]) for c in range(3)], -1)
    shift = np.hypot(rel[..., 0], rel[..., 1])
    flags = []
    rows = []
    kinds = np.array(["oct" if k.startswith("oct") else k for u, v, k in sites])
    for i, d in enumerate(dies):
        m = dies_meta[d]
        fl = {}
        for j, (u, v, k) in enumerate(sites):
            if np.isnan(ncc[i, j]):
                continue
            why = []
            if ncc[i, j] < 0.5 and zn[i, j] < -6:
                why.append("shape ncc=%.2f" % ncc[i, j])
            if abs(zl[i, j]) > 6:
                why.append("lum z=%.1f" % zl[i, j])
            if np.nanmax(np.abs(zc[i, j])) > 6:
                why.append("colour z=%.1f" % zc[i, j][np.nanargmax(np.abs(zc[i, j]))])
            if shift[i, j] > 4:
                why.append("shift %.1f px" % shift[i, j])
            if why:
                fl[j] = why
                flags.append(dict(die=d, row=m["row"], col=m["col"], post=m["post"], detail_post=m["detail_post"],
                                  site=j, kind=k, u=u, v=v, why="; ".join(why)))
        cnt = {kk: sum(1 for j in fl if kinds[j] == kk) for kk in ("oct", "prow", "pix")}
        rows.append(dict(die=d, row=m["row"], col=m["col"], pre=m["pre"], post=m["post"], detail_post=m["detail_post"],
                         measured=int(np.isfinite(ncc[i]).sum()), flag_oct=cnt["oct"], flag_prow=cnt["prow"],
                         flag_pix=cnt["pix"], med_ncc=round(float(np.nanmedian(ncc[i])), 3),
                         med_lum_pix=round(float(np.nanmedian(lum[i, kinds == "pix"])), 3),
                         med_lum_oct=round(float(np.nanmedian(lum[i, kinds == "oct"])), 3)))
    for name, rr_ in (("census_dies.csv", rows), ("census_flags.csv", flags)):
        with open(outd + name, "w", newline="") as f:
            if rr_:
                wr = csv.DictWriter(f, fieldnames=list(rr_[0]))
                wr.writeheader()
                wr.writerows(rr_)
    print("reference dies", int(ok.sum()), "flags", len(flags))


if __name__ == "__main__":
    main()
