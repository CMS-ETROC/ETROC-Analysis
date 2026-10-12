"""Summarise census_dies.csv / census_flags.csv of one wafer: failing vs passing dies, and a wafer map.

python census_report.py <dir with census_*.csv> <wafer label>
"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

d, label = sys.argv[1], sys.argv[2]
dies = [r for r in csv.DictReader(open(d + "/census_dies.csv"))]
flags = [r for r in csv.DictReader(open(d + "/census_flags.csv"))]
INVALID = {57, 58}
dies = [r for r in dies if int(r["die"]) not in INVALID and int(r["measured"]) > 300]
new = [r for r in dies if r["pre"] == "PASSED" and r["post"] != "PASSED"]
ok = [r for r in dies if r["pre"] == "PASSED" and r["post"] == "PASSED"]
print("dies measured (>300 sites): %d; new failures %d; passed both %d" % (len(dies), len(new), len(ok)))
for key in ("flag_oct", "flag_prow", "flag_pix", "med_lum_pix", "med_lum_oct", "med_ncc"):
    a = np.array([float(r[key]) for r in new])
    b = np.array([float(r[key]) for r in ok])
    p = mannwhitneyu(a, b).pvalue
    print("%-12s new-fail median %7.3f (IQR %.3f-%.3f)   passed median %7.3f (IQR %.3f-%.3f)   MWU p=%.3g" % (
        key, np.median(a), *np.percentile(a, [25, 75]), np.median(b), *np.percentile(b, [25, 75]), p))

# flag reasons by kind, failing vs passing (per die)
for grp, rows in (("new-fail", new), ("passed", ok)):
    ids = {r["die"] for r in rows}
    f = [x for x in flags if x["die"] in ids]
    for kind in ("oct", "prow", "pix"):
        fk = [x for x in f if (x["kind"].startswith("oct") if kind == "oct" else x["kind"] == kind)]
        why = {}
        for x in fk:
            for w in x["why"].split("; "):
                why[w.split()[0]] = why.get(w.split()[0], 0) + 1
        print("%-8s %-4s flags/die %.1f  by reason/die %s" % (grp, kind, len(fk) / len(rows),
              {k: round(v / len(rows), 1) for k, v in sorted(why.items())}))

# wafer maps
fig, axs = plt.subplots(1, 3, figsize=(15, 5.2))
for ax, key in zip(axs, ("flag_oct", "flag_pix", "med_lum_oct")):
    grid = np.full((12, 13), np.nan)
    for r in dies:
        grid[int(r["row"]), int(r["col"])] = float(r[key])
    im = ax.imshow(grid, cmap="viridis")
    fig.colorbar(im, ax=ax, shrink=0.8)
    for r in new:
        ax.add_patch(plt.Rectangle((int(r["col"]) - 0.5, int(r["row"]) - 0.5), 1, 1, fill=False, ec="red", lw=2))
    ax.set_title(key)
    ax.set_xlabel("map column")
    ax.set_ylabel("map row")
fig.suptitle("%s: dome census per die (red = new post-UBM failure), wafer view notch up" % label)
fig.tight_layout()
fig.savefig(d + "/census_map.png", dpi=80)
