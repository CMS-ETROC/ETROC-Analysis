"""Side-by-side patches of the strongest anomaly blobs: this die | the same spot on the reference die.

python patches.py <wafer> <region> <ref_die> <nblobs> <half> <die> [<die> ...]
"""
import csv
import sys

import numpy as np
from PIL import Image, ImageDraw

from anomaly import REGIONS
from ets import ETS
from wafer_files import image_path

w, region, ref_die, nb, half = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
dies = [int(x) for x in sys.argv[6:]]
L = REGIONS[region][0]
outd = "out/%s_%s" % (w, region)
meta = {int(r["die"]): r for r in csv.DictReader(open(outd + "/dies.csv"))}
blobs = [r for r in csv.DictReader(open(outd + "/blobs.csv"))]
e = ETS(image_path(w))


def crop(d, x, y):
    m = meta[d]
    ox = int(m["x0"]) + x - int(m["sx"])
    oy = int(m["y0"]) + y - int(m["sy"])
    return e.read(L, ox - half, oy - half, ox + half, oy + half)


for d in dies:
    bs = sorted([b for b in blobs if int(b["die"]) == d], key=lambda b: -int(b["area"]) * float(b["zmax"]))[:nb]
    if not bs:
        print(d, "no blobs")
        continue
    tiles = []
    for b in bs:
        x, y = int(b["x"]), int(b["y"])
        a, r = crop(d, x, y), crop(ref_die, x, y)
        row = np.full((2 * half, 4 * half + 10, 3), 255, np.uint8)
        row[:, :2 * half] = a
        row[:, 2 * half + 10:] = r
        im = Image.fromarray(row)
        dr = ImageDraw.Draw(im)
        dr.text((4, 4), "die %d  x%d y%d  area %s z %s" % (d, x, y, b["area"], b["zmax"]), fill=(255, 255, 0))
        dr.text((2 * half + 14, 4), "ref die %d" % ref_die, fill=(255, 255, 0))
        dr.rectangle([half - int(b["w"]) // 2 - 4, half - int(b["h"]) // 2 - 4, half + int(b["w"]) // 2 + 4,
                      half + int(b["h"]) // 2 + 4], outline=(255, 0, 0))
        tiles.append(np.asarray(im))
    Image.fromarray(np.vstack(tiles)).save("%s/patch_die%03d.png" % (outd, d))
    print(d, len(bs))
