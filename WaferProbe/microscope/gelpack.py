"""Bump checks on gel-pack images of diced chips (Olympus cellSens .ets, about 1.25 um per pixel at level 0).

python gelpack.py bumps <frame_t_0.ets>
python gelpack.py domes <frame_t_0.ets> [chip ...]

bumps (level 4, about 20 um per pixel): finds the chips, fits each chip's 16 x 16 bump grid and the bump row along
the pad row, and scores every site. score = normalised cross-correlation (NCC) with the pack's median bump, best
within +-3 px of the site: an empty site scores below 0.3. highlight = brightest pixel within +-3 px of the site minus
the 15 px box mean: a chip whose highlight is far below the other chips of the pack has no solder bumps.
domes (level 1, about 2.5 um per pixel; needs bumps): a patch of every bump, re-centred on the chip's median bump;
NCC with that median and the colour of the bump top (blue minus red). A top that is redder than the chip median by
more than 4 robust sigma is flagged as discoloured; a site whose NCC is below 0.3 has no bump to judge. Writes a
sheet of every bump per chip.

Outputs go to ./out/gelpack_<name>/, <name> being the image folder without its leading and trailing underscores
(_BadChips_111_N62C72_14_ -> BadChips_111_N62C72_14). Chips are numbered from 0 by rows of the scan, top to bottom,
then left to right; sites (j, i) count bump rows from the top and columns from the left from 0, as the chip lies in
the scan (pads at the bottom), row 16 being the row along the pad row.
"""
import csv
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.signal import fftconvolve

from ets import ETS

PITCH = 64.2                   # bump pitch at level 4 (1.3 mm)
CHIP_W, CHIP_H = (900, 1150), (900, 1250)   # accepted chip box sizes at level 4
BRIGHT = 120                   # smoothed grey above which a pixel belongs to a chip
H4 = 7                         # half size of the level-4 bump template
W1, R1 = 110, 40               # level-1 search window and patch half sizes
EMPTY, UNBUMPED, DISCOLOURED = 0.3, 0.6, 4.0   # thresholds: NCC, highlight / pack median, robust sigma


def image_name(path):
    return os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(path)))).strip("_")


def out_dir(path):
    d = os.path.join("out", "gelpack_" + image_name(path))
    os.makedirs(d, exist_ok=True)
    return d


def find_chips(g):
    """(y0, y1, x0, x1) of every chip-sized bright region, by rows of the scan then left to right"""
    m = ndi.binary_erosion(ndi.uniform_filter(g, 61) > BRIGHT, np.ones((51, 51)))
    lab, _ = ndi.label(m)
    out = []
    for sy, sx in ndi.find_objects(lab):
        y0, y1, x0, x1 = sy.start - 25, sy.stop + 25, sx.start - 25, sx.stop + 25
        if CHIP_W[0] < x1 - x0 < CHIP_W[1] and CHIP_H[0] < y1 - y0 < CHIP_H[1]:
            out.append((y0, y1, x0, x1))
    rows = []
    for b in sorted(out):                       # a new row of chips starts at a gap of half a chip height
        if not rows or b[0] - rows[-1][-1][0] > CHIP_H[0] / 2:
            rows.append([])
        rows[-1].append(b)
    return [b for r in rows for b in sorted(r, key=lambda b: b[2])]


def resultant(v, pitches):
    """the pitch whose circular mean of v has the longest resultant: (length, pitch, phase)"""
    z = np.exp(2j * np.pi * v[None, :] / pitches[:, None]).mean(axis=1)
    k = np.abs(z).argmax()
    return abs(z[k]), pitches[k], (np.angle(z[k]) / (2 * np.pi) * pitches[k]) % pitches[k]


def fit_lattice(xs, ys, cx, cy):
    """angle, x and y pitch and phase of the lattice through the points, rotated about (cx, cy)"""
    pitches = np.arange(PITCH - 1.2, PITCH + 1.3, 0.02)
    best = None
    for th in np.radians(np.arange(-1.5, 1.51, 0.05)):
        c, s = np.cos(th), np.sin(th)
        rx = resultant(c * (xs - cx) + s * (ys - cy), pitches)
        ry = resultant(-s * (xs - cx) + c * (ys - cy), pitches)
        if best is None or rx[0] * ry[0] > best[0]:
            best = (rx[0] * ry[0], th, rx, ry)
    return best


def site_grid(lat, ki, kj, n=16):
    """level-4 x, y (n x n) of the lattice sites, index shift ki, kj"""
    th, (_, px, phx), (_, py, phy), cx, cy, u0, w0 = lat
    i, j = np.meshgrid(np.arange(n) + ki, np.arange(n) + kj)
    u, w = u0 + phx + px * i, w0 + phy + py * j
    c, s = np.cos(th), np.sin(th)
    return cx + c * u - s * w, cy + s * u + c * w


def near_max(m, x, y, r=3):
    """(max, x, y) of m within +-r px of (x, y)"""
    x, y = int(round(x)), int(round(y))
    w = m[y - r:y + r + 1, x - r:x + r + 1]
    dy, dx = np.unravel_index(w.argmax(), w.shape)
    return w.max(), x - r + dx, y - r + dy


def chip_lattice(resp, box, npk=300):
    """lattice of one chip from the npk strongest response peaks in its top 85 % (the pad row excluded), and the
    index shift that puts the chip's 16 x 16 sites on the highest summed row and column medians of resp; None if
    the box cannot hold the sites (a chip cut by the edge of the scan)"""
    y0, y1, x0, x1 = box
    sub = resp[y0 + 20:y0 + int(0.85 * (y1 - y0)), x0 + 20:x1 - 20]
    ys, xs = np.nonzero(sub == ndi.maximum_filter(sub, 31))
    o = np.argsort(sub[ys, xs])[-npk:]
    xs, ys = xs[o] + x0 + 20.0, ys[o] + y0 + 20.0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    _, th, rx, ry = fit_lattice(xs, ys, cx, cy)
    lat = (th, rx, ry, cx, cy, -rx[1] * np.ceil((cx - x0) / rx[1]), -ry[1] * np.ceil((cy - y0) / ry[1]))
    best = None
    for ki in range(-1, 3):
        for kj in range(-1, 3):
            X, Y = site_grid(lat, ki, kj)
            if X.min() < x0 or X.max() > x1 or Y.min() < y0 or Y.max() > y1:
                continue
            M = np.array([[near_max(resp, X[j, i], Y[j, i])[0] for i in range(16)] for j in range(16)])
            sc = np.median(M, axis=0).sum() + np.median(M, axis=1).sum()
            if best is None or sc > best[0]:
                best = (sc, ki, kj)
    return None if best is None else (lat, best[1], best[2])


def ncc_map(g, t):
    """normalised cross-correlation of g with the zero-mean, unit-sigma template t (same size as g)"""
    k = t.shape[0]
    num = fftconvolve(g, t[::-1, ::-1], mode="same") / t.size
    mu = ndi.uniform_filter(g, k)
    sd = np.sqrt(np.clip(ndi.uniform_filter(g * g, k) - mu * mu, 1e-6, None))
    return num / sd


def analyse_level4(rgb):
    """per chip: box, angle, pitches, the 16 x 16 sites X, Y and their score S and highlight HL, the bottom row
    BX, BY, BS, BHL (all level-4 px of rgb), and unbumped (None with fewer than 3 chips to compare)"""
    g = rgb.astype(float).mean(axis=2)
    var = ndi.uniform_filter(g * g, 5) - ndi.uniform_filter(g, 5) ** 2
    boxes, pats = [], []                        # pass 1: sites from local-variance peaks, then the pack's median bump
    for box in find_chips(g):
        fit = chip_lattice(var, box)
        if fit is None:
            print("skipped the chip at level-4 x %d-%d, y %d-%d: too small for the bump grid" % (*box[2:], *box[:2]))
            continue
        boxes.append(box)
        X, Y = site_grid(*fit)
        for x, y in zip(X.ravel(), Y.ravel()):
            _, xx, yy = near_max(var, x, y)
            pats.append(g[yy - H4:yy + H4 + 1, xx - H4:xx + H4 + 1])
    if not boxes:
        return []
    t = np.median(pats, axis=0)
    ncc = ncc_map(g, (t - t.mean()) / t.std())
    bg = ndi.uniform_filter(g, 15)

    def highlight(x, y):
        x, y = int(np.rint(x)), int(np.rint(y))
        return max(g[y - 3:y + 4, x - 3:x + 4].max() - bg[y, x], 0)

    chips = []
    for box in boxes:                           # pass 2: the same on the NCC peaks
        fit = chip_lattice(ncc, box)
        if fit is None:
            print("skipped the chip at level-4 x %d-%d, y %d-%d: too small for the bump grid" % (*box[2:], *box[:2]))
            continue
        lat = fit[0]
        X, Y = site_grid(*fit)
        S = np.array([[near_max(ncc, X[j, i], Y[j, i])[0] for i in range(16)] for j in range(16)])
        HL = np.array([[highlight(X[j, i], Y[j, i]) for i in range(16)] for j in range(16)])
        BS, BX, BY = np.zeros(16), np.zeros(16), np.zeros(16)
        for i in range(16):                     # bottom row: about half a pitch below row 15
            xc, yc = int(round(X[15, i])), int(round(Y[15, i]))
            w = ncc[yc + 20:yc + 51, xc - 20:xc + 21]
            dy, dx = np.unravel_index(w.argmax(), w.shape)
            BS[i], BX[i], BY[i] = w.max(), xc - 20 + dx, yc + 20 + dy
        BHL = np.array([highlight(x, y) for x, y in zip(BX, BY)])
        chips.append(dict(box=box, rot=np.degrees(lat[0]), px=lat[1][1], py=lat[2][1], X=X, Y=Y, S=S, HL=HL,
                          BX=BX, BY=BY, BS=BS, BHL=BHL))
    pack_hl = np.median([np.median(c["HL"]) for c in chips]) if chips else 0
    for c in chips:
        c["unbumped"] = None if len(chips) < 3 else bool(np.median(c["HL"]) < UNBUMPED * pack_hl)
    return chips


def bumps(path):
    e = ETS(path)
    _, (xa, xb), (ya, yb), _ = e.level_grid(4)
    ox, oy = xa * e.tile_x, ya * e.tile_y
    rgb = e.read(4, ox, oy, (xb + 1) * e.tile_x, (yb + 1) * e.tile_y)
    chips = analyse_level4(rgb)
    if not chips:
        sys.exit("no chip found at level 4: different magnification? (PITCH, CHIP_W, CHIP_H, BRIGHT)")
    d, name = out_dir(path), image_name(path)
    pack_hl = np.median([np.median(c["HL"]) for c in chips])
    if len(chips) < 3:
        print("%s: %d chip(s), too few to judge solder bumps by the pack median: compare the highlight with a pack "
              "of bumped chips scanned the same way" % (name, len(chips)))
    vis = Image.fromarray(rgb)
    dr = ImageDraw.Draw(vis)
    with open(os.path.join(d, "chips.csv"), "w", newline="") as fc, \
            open(os.path.join(d, "bumps.csv"), "w", newline="") as fb:
        wc, wb = csv.writer(fc), csv.writer(fb)
        wc.writerow(["chip", "x0", "y0", "x1", "y1", "rot_deg", "pitch_x", "pitch_y", "score_median", "score_min",
                     "n_empty", "highlight_median", "highlight_p10", "unbumped"])
        wb.writerow(["chip", "j", "i", "x4", "y4", "score", "highlight"])
        for k, c in enumerate(chips):
            y0, y1, x0, x1 = c["box"]
            S = np.concatenate([c["S"].ravel(), c["BS"]])
            hl, unbumped = np.median(c["HL"]), c["unbumped"]
            wc.writerow([k, x0 + ox, y0 + oy, x1 + ox, y1 + oy, "%.2f" % c["rot"], "%.2f" % c["px"], "%.2f" % c["py"],
                         "%.2f" % np.median(S), "%.2f" % S.min(), int((S < EMPTY).sum()), "%.0f" % hl,
                         "%.0f" % np.percentile(c["HL"], 10), "" if unbumped is None else int(unbumped)])
            sites = [(j, i, c["X"][j, i], c["Y"][j, i], c["S"][j, i], c["HL"][j, i])
                     for j in range(16) for i in range(16)]
            sites += [(16, i, c["BX"][i], c["BY"][i], c["BS"][i], c["BHL"][i]) for i in range(16)]
            empty = []
            for j, i, x, y, s, h in sites:
                wb.writerow([k, j, i, "%.4f" % (x + ox), "%.4f" % (y + oy), "%.3f" % s, "%.0f" % h])
                if s < EMPTY:
                    empty.append((j, i))
                    dr.ellipse([x - 14, y - 14, x + 14, y + 14], outline=(255, 0, 0), width=3)
            dr.rectangle([x0, y0, x1, y1], outline=(255, 0, 0) if unbumped else (0, 255, 0), width=3)
            dr.text((x0 + 30, y0 + 30), "chip %d" % k, fill=(255, 255, 0))
            print("%s chip %d: rot %+.2f deg, score median %.2f min %.2f, highlight %.0f (pack median %.0f)%s, "
                  "empty sites %s" % (name, k, c["rot"], np.median(S), S.min(), hl, pack_hl,
                                      " NO SOLDER BUMPS?" if unbumped else "", empty), flush=True)
            if abs(c["rot"]) > 1.45:
                print("  chip %d is rotated by the search limit (1.5 deg): the outer sites may be off" % k)
    vis.save(os.path.join(d, "bumps_L4.png"))


def chip_domes(read_window, sites):
    """level-1 check of one chip. read_window(x, y) -> float RGB (2 W1 + 1, 2 W1 + 1) centred on the level-1 pixel
    (x, y); sites: (kind, j, i, x4, y4) with kind 'a' (16 x 16 grid) or 'b' (bottom row).
    Returns rows (kind, j, i, x1, y1, dx, dy, ncc, b_minus_r, patch) and the chip's median bump per kind."""
    wins = []
    for kind, j, i, x4, y4 in sites:            # pass 1: centre on the local-variance maximum
        x, y = int(round(x4 * 8)), int(round(y4 * 8))
        rgb = read_window(x, y)
        g = rgb.mean(axis=2)
        v = ndi.uniform_filter(ndi.uniform_filter(g * g, 15) - ndi.uniform_filter(g, 15) ** 2, 9)
        cy, cx = np.unravel_index(v[R1:-R1, R1:-R1].argmax(), (2 * (W1 - R1) + 1,) * 2)
        wins.append((kind, j, i, x, y, rgb, g, cy + R1, cx + R1))
    tmpl = {}
    for k in {w[0] for w in wins}:
        tmpl[k] = np.median([w[6][cy - R1:cy + R1 + 1, cx - R1:cx + R1 + 1] for w in wins if w[0] == k
                             for cy, cx in [(w[7], w[8])]], axis=0)
    rows = []
    for kind, j, i, x, y, rgb, g, _, _ in wins:  # pass 2: centre on the NCC maximum with the chip's median bump
        t = tmpl[kind]
        ncc = ncc_map(g, (t - t.mean()) / t.std())[R1:-R1, R1:-R1]
        cy, cx = np.unravel_index(ncc.argmax(), ncc.shape)
        cy, cx = cy + R1, cx + R1
        core = rgb[cy - 8:cy + 9, cx - 8:cx + 9].reshape(-1, 3).mean(axis=0)
        rows.append((kind, j, i, x - W1 + cx, y - W1 + cy, cx - W1, cy - W1, ncc.max(), core[2] - core[0],
                     rgb[cy - R1:cy + R1 + 1, cx - R1:cx + R1 + 1]))
    return rows, tmpl


def discoloured(rows):
    """indices of the rows whose bump top is redder than the chip median by more than DISCOLOURED robust sigma,
    among the rows with a bump (NCC at least EMPTY)"""
    ok = [k for k, r in enumerate(rows) if r[7] >= EMPTY]
    br = np.array([rows[k][8] for k in ok])
    med = np.median(br)
    mad = 1.4826 * np.median(np.abs(br - med))
    return [k for k, v in zip(ok, br) if v < med - DISCOLOURED * mad]


def domes(path, only=None):
    d, name = out_dir(path), image_name(path)
    fb = os.path.join(d, "bumps.csv")
    if not os.path.exists(fb):
        sys.exit("run gelpack.py bumps %s first" % path)
    with open(fb) as f:
        rows = list(csv.DictReader(f))
    e = ETS(path)

    def read_window(x, y):
        return e.read(1, x - W1, y - W1, x + W1 + 1, y + W1 + 1).astype(float)

    with open(os.path.join(d, "domes.csv"), "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["chip", "j", "i", "x1", "y1", "dx", "dy", "score4", "ncc", "b_minus_r", "discoloured"])
        for c in sorted({int(r["chip"]) for r in rows}):
            if only and c not in only:
                continue
            mine = [r for r in rows if int(r["chip"]) == c]
            sites = [("b" if r["j"] == "16" else "a", int(r["j"]), int(r["i"]), float(r["x4"]), float(r["y4"]))
                     for r in mine]
            res, _ = chip_domes(read_window, sites)
            bad = set(discoloured(res))
            sheet = np.full((17 * 83, 16 * 83, 3), 255, np.uint8)
            for k, (kind, j, i, x1, y1, dx, dy, ncc, br, p) in enumerate(res):
                w.writerow([c, j, i, x1, y1, dx, dy, mine[k]["score"], "%.3f" % ncc, "%.0f" % br, int(k in bad)])
                sheet[j * 83:j * 83 + 81, i * 83:i * 83 + 81] = np.clip(p, 0, 255).astype(np.uint8)
            Image.fromarray(sheet).save(os.path.join(d, "sheet_chip%d.png" % c))
            print("%s chip %d: ncc median %.2f min %.2f, discoloured tops %s, no bump found %s" % (
                name, c, np.median([r[7] for r in res]), min(r[7] for r in res),
                [(res[k][1], res[k][2]) for k in sorted(bad)], [(r[1], r[2]) for r in res if r[7] < EMPTY]),
                flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("bumps", "domes"):
        sys.exit(__doc__)
    if sys.argv[1] == "bumps":
        bumps(sys.argv[2])
    else:
        domes(sys.argv[2], {int(c) for c in sys.argv[3:]})
