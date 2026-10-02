"""Per-die anomaly maps: every die of a wafer registered to one reference die, compared with the
per-pixel median and MAD of the dies that PASSED both before and after UBM.

python anomaly.py <wafer> <region> <ref_die> [<lot_dies.csv>]
region: peri = die top strip at level 0 (pad rows + first pixel-bump row)
        full = whole die at level 1
lot_dies.csv (default: ./lot_dies.csv) is the die table of WaferProbe/plot_lots.py; its grade_pre and grade_post
choose the reference dies.
Outputs in out/<wafer>_<region>/: dies.csv (per-die anomaly stats), blobs.csv, median.npy, sigma.npy, and a z-map PNG
per die (z_dieNNN.png). The registered stack (0.9 GB for peri, 1.9 GB for full) lives in a file under $TMPDIR
(default /tmp) that is unlinked at once, so it is freed when the process ends, however it ends.
"""
import csv
import os
import sys
import tempfile

import numpy as np
from PIL import Image
from scipy import ndimage

from ets import ETS
from wafer_files import grid_fit, image_path, wafer_map

REGIONS = {  # level, dx0, dy0, width, height in that level's pixels, relative to the grid corner of the die
    "peri": (0, -400, -60, 5600, 700),
    "full": (1, -200, -40, 2800, 2860),
}
MAXSHIFT = 40


def dog(gray):
    return ndimage.gaussian_filter(gray, 1.0) - ndimage.gaussian_filter(gray, 6.0)


def _prep(g, f, s1, s2):
    if f > 1:
        h, w = g.shape[0] // f * f, g.shape[1] // f * f
        g = g[:h, :w].reshape(h // f, f, w // f, f).mean((1, 3))
    g = ndimage.gaussian_filter(g, s1) - ndimage.gaussian_filter(g, s2)
    g = g - g.mean()
    return g * np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))


def _xcorr(a, b):
    c = np.fft.irfft2(np.fft.rfft2(a) * np.conj(np.fft.rfft2(b)), s=a.shape)
    return c / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9)


def _wrap(i, n):
    return i - n if i > n // 2 else i


def register(ref_gray, gray, f=4, fine=6, maxc=None):
    """integer (dy, dx) that moves gray onto ref_gray: coarse cross-correlation at 1/f (within +-maxc px),
    then +-fine px at full res"""
    c = _xcorr(_prep(ref_gray, f, 1, 8), _prep(gray, f, 1, 8))
    if maxc is not None:
        r = maxc // f
        cm = np.full_like(c, -np.inf)
        cm[:r + 1, :r + 1] = c[:r + 1, :r + 1]
        cm[-r:, :r + 1] = c[-r:, :r + 1]
        cm[:r + 1, -r:] = c[:r + 1, -r:]
        cm[-r:, -r:] = c[-r:, -r:]
        c = cm
    cy, cx = np.unravel_index(np.argmax(c), c.shape)
    cy, cx = _wrap(cy, c.shape[0]) * f, _wrap(cx, c.shape[1]) * f
    c2 = _xcorr(_prep(ref_gray, 1, 1.5, 8), _prep(gray, 1, 1.5, 8))
    best = None
    for dy in range(cy - fine, cy + fine + 1):
        for dx in range(cx - fine, cx + fine + 1):
            v = c2[dy % c2.shape[0], dx % c2.shape[1]]
            if best is None or v > best[0]:
                best = (v, dy, dx)
    return best[1], best[2], float(c.max()), float(best[0])


def local_register(ref_dd, dd, P=128, S=192, R=8, min_peak=0.2):
    """per-cell shifts of dd onto ref_dd (the microscope mosaic places each field of view with its own error);
    returns the re-assembled dd and the fraction of cells that found a confident shift"""
    H, W = dd.shape
    out = np.zeros_like(dd)
    win = np.outer(np.hanning(S), np.hanning(S)).astype(np.float32)
    ok = n = 0
    for y in range(0, H, P):
        for x in range(0, W, P):
            yc, xc = min(max(y + P // 2, S // 2), H - S // 2), min(max(x + P // 2, S // 2), W - S // 2)
            a = ref_dd[yc - S // 2:yc + S // 2, xc - S // 2:xc + S // 2]
            b = dd[yc - S // 2:yc + S // 2, xc - S // 2:xc + S // 2]
            a = (a - a.mean()) * win
            b = (b - b.mean()) * win
            c = np.fft.irfft2(np.fft.rfft2(a) * np.conj(np.fft.rfft2(b)), s=a.shape)
            c /= np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9
            sub = np.roll(np.roll(c, R, 0), R, 1)[:2 * R + 1, :2 * R + 1]
            k = np.unravel_index(np.argmax(sub), sub.shape)
            n += 1
            if sub[k] >= min_peak:
                sy, sx = k[0] - R, k[1] - R
                ok += 1
            else:
                sy = sx = 0
            y1, x1 = min(y + P, H), min(x + P, W)
            ys0, xs0 = y - sy, x - sx
            cy0, cx0 = max(0, -ys0), max(0, -xs0)
            cy1, cx1 = (y1 - y) - max(0, ys0 + (y1 - y) - H), (x1 - x) - max(0, xs0 + (x1 - x) - W)
            out[y + cy0:y + cy1, x + cx0:x + cx1] = dd[ys0 + cy0:ys0 + cy1, xs0 + cx0:xs0 + cx1]
    return out, ok / max(n, 1)


def main():
    w, region, ref_die = sys.argv[1], sys.argv[2], int(sys.argv[3])
    lot_dies = sys.argv[4] if len(sys.argv) > 4 else "lot_dies.csv"
    L, dx0, dy0, W, H = REGIONS[region]
    s = 2 ** (4 - L)
    g = grid_fit(w)
    mp = wafer_map()
    with open(lot_dies) as f:
        grades = {int(r["die"]): r for r in csv.DictReader(f) if r["wafer"] == w.split("_")[1]}
    if not grades:
        sys.exit("no dies of wafer %s in %s" % (w.split("_")[1], lot_dies))
    e = ETS(image_path(w))
    outd = "out/%s_%s" % (w, region)
    os.makedirs(outd, exist_ok=True)
    dies = sorted(mp)
    tmp = os.path.join(tempfile.gettempdir(), "ubm_%s_%s_%d.i16" % (w, region, os.getpid()))
    stack = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.int16, shape=(len(dies), H, W))
    os.remove(tmp)  # the mapping keeps the data until the process ends

    # coarse offsets of every die from the lattice (the wafer sits slightly rotated in the image), at level 3
    m3 = 40
    def box3(d):
        r, c = mp[d]
        x = int(round((g["phx"] + c * g["Px"]) * 2)) - m3
        y = int(round((g["phy"] + r * g["Py"]) * 2)) - m3
        bw, bh = int(g["Px"] * 2) + 2 * m3, int(g["Py"] * 2) + 2 * m3
        img = e.read(3, max(0, x), max(0, y), x + bw, y + bh).mean(2).astype(np.float32)
        out = np.full((bh, bw), 255, np.float32)
        out[max(0, -y):max(0, -y) + img.shape[0], max(0, -x):max(0, -x) + img.shape[1]] = img
        return out
    ref3 = box3(ref_die)
    pred = {}
    for d in dies:
        b = box3(d)
        if b.shape != ref3.shape:
            pred[d] = (0, 0)
            continue
        c3 = _xcorr(_prep(ref3, 1, 1, 8), _prep(b, 1, 1, 8))
        py, px = np.unravel_index(np.argmax(c3), c3.shape)
        pred[d] = (_wrap(py, c3.shape[0]) * 8 // 2 ** L, _wrap(px, c3.shape[1]) * 8 // 2 ** L)

    # the bump array repeats every 1.3 mm, so a single die can lock one period off: use a robust affine
    # fit of the offsets over the lattice (the wafer is a rigid body in the image) instead of per-die guesses
    A = np.array([[1.0, mp[d][0], mp[d][1]] for d in dies])
    for k in (0, 1):
        v = np.array([pred[d][k] for d in dies], float)
        keep = np.ones(len(dies), bool)
        for _ in range(5):
            coef, *_ = np.linalg.lstsq(A[keep], v[keep], rcond=None)
            res = v - A @ coef
            mad = np.median(np.abs(res[keep])) + 1.0
            keep = np.abs(res) < 4 * mad
        fit = A @ coef
        print("lattice fit", "dy" if k == 0 else "dx", np.round(coef, 2), "outliers", int((~keep).sum()), flush=True)
        for i, d in enumerate(dies):
            pred[d] = (int(round(fit[i])), pred[d][1]) if k == 0 else (pred[d][0], int(round(fit[i])))

    def load(d):
        r, c = mp[d]
        x0 = int(round((g["phx"] + c * g["Px"]) * s)) + dx0 - pred[d][1]
        y0 = int(round((g["phy"] + r * g["Py"]) * s)) + dy0 - pred[d][0]
        img = e.read(L, max(0, x0), max(0, y0), x0 + W, y0 + H)
        pad = np.full((H, W, 3), 255, np.uint8)
        pad[max(0, -y0):max(0, -y0) + img.shape[0], max(0, -x0):max(0, -x0) + img.shape[1]] = img
        return pad, x0, y0

    ref_img, _, _ = load(ref_die)
    ref_gray = ref_img.mean(2).astype(np.float32)
    ref_dd = dog(ref_gray)
    meta = {}
    for i, d in enumerate(dies):
        img, x0, y0 = load(d)
        gray = img.mean(2).astype(np.float32)
        dd = dog(gray)
        sy, sx, peak, peak_fine = register(ref_gray, gray, f=2, fine=4, maxc=MAXSHIFT)
        if abs(sy) > MAXSHIFT or abs(sx) > MAXSHIFT:
            sy = sx = 0
        reg = ndimage.shift(dd, (sy, sx), order=0, mode="constant", cval=0)
        reg, local_ok = local_register(ref_dd, reg)
        stack[i] = np.clip(reg * 100, -32000, 32000).astype(np.int16)
        white = ndimage.shift((gray > 235).astype(np.uint8), (sy, sx), order=0, cval=1)  # off-wafer / blank
        meta[d] = dict(x0=x0, y0=y0, py=pred[d][0], px=pred[d][1], sy=sy, sx=sx, peak=peak, peak_fine=peak_fine, local_ok=round(local_ok, 3), offwafer=float(white.mean()))
        print("load", d, sy, sx, "%.3f %.3f %.3f" % (peak, peak_fine, local_ok), flush=True)
    stack.flush()

    ref_set = [i for i, d in enumerate(dies)
               if d in grades and grades[d]["grade_pre"] == "PASSED" and grades[d]["grade_post"] == "PASSED"
               and meta[d]["peak"] > 0.3 and meta[d]["offwafer"] < 0.01]
    print("reference dies:", len(ref_set), flush=True)
    med = np.zeros((H, W), np.float32)
    mad = np.zeros((H, W), np.float32)
    for x in range(0, W, 400):
        blk = stack[ref_set, :, x:x + 400].astype(np.float32)
        m = np.median(blk, axis=0)
        med[:, x:x + 400] = m
        mad[:, x:x + 400] = np.median(np.abs(blk - m), axis=0)
    sig = 1.4826 * mad + 30.0  # floor: 0.3 grey levels of DoG
    np.save(outd + "/median.npy", med)
    np.save(outd + "/sigma.npy", sig)

    rows, blobs = [], []
    for i, d in enumerate(dies):
        z = (stack[i].astype(np.float32) - med) / sig
        hot = np.abs(z) > 8
        hot[:MAXSHIFT] = hot[-MAXSHIFT:] = False
        hot[:, :MAXSHIFT] = hot[:, -MAXSHIFT:] = False
        lab, n = ndimage.label(ndimage.binary_dilation(hot, iterations=2))
        objs = ndimage.find_objects(lab)
        areas = ndimage.sum(hot, lab, range(1, n + 1)) if n else []
        zmax = ndimage.maximum(np.abs(z), lab, range(1, n + 1)) if n else []
        big = 0
        for k, sl in enumerate(objs):
            if areas[k] < 6:
                continue
            big += 1
            blobs.append(dict(die=d, y=(sl[0].start + sl[0].stop) // 2, x=(sl[1].start + sl[1].stop) // 2,
                              h=sl[0].stop - sl[0].start, w=sl[1].stop - sl[1].start,
                              area=int(areas[k]), zmax=round(float(zmax[k]), 1)))
        gr = grades.get(d, {})
        rows.append(dict(die=d, row=mp[d][0], col=mp[d][1], pre=gr.get("grade_pre", ""), post=gr.get("grade_post", ""),
                         detail_post=gr.get("detail_post", ""), hot_px=int(hot.sum()), blobs=big,
                         in_ref=int(i in ref_set), **meta[d]))
        m = np.zeros((H, W, 3), np.uint8)
        m[..., 0] = np.clip(np.abs(z) * 20, 0, 255).astype(np.uint8)
        m[hot] = (255, 255, 0)
        Image.fromarray(m[::2, ::2]).save("%s/z_die%03d.png" % (outd, d))
    with open(outd + "/dies.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with open(outd + "/blobs.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["die", "y", "x", "h", "w", "area", "zmax"])
        wr.writeheader()
        wr.writerows(blobs)
    print("done", outd)


if __name__ == "__main__":
    main()
