"""Raw TDC codes (CAL, TOA, TOT) of one pixel per chip, and every pixel's CAL code, per run.

gather() reads every loop file of a run from the merged hits (DECODED_DIRS in the campaign
module) and histograms, per board and pixel, after these cuts:

  ea == 0                 the hit's error flag clear
  CAL window              |cal - m| <= 3 and m > 0, with m the pixel's CAL mode in THAT file
                          (the autocalibration drifts from file to file)

TOA and TOT in ns use that same per-file CAL mode m: TOA = 12.5 - toa * 3.125 / m and
TOT = (2 tot - floor(tot / 32)) * 3.125 / m. Codes are kept native; ns in 0.1 ns bins, not
clipped to a range. A run's CAL mode per pixel is the median of its per-file modes.

Two tables (columns TDC_HISTS_COLUMNS and TDC_CAL_COLUMNS, described in the campaign's inputs
document), made by `python -m etroc_plots.frontend.rebuild --part tdc`:

  TDC_PIXEL_HISTS_CSV   the histograms of one pixel per chip: the chip's most populated pixel in
                        the July run of its COOLDOWN_PAIRS pair (hits after the cuts), the same
                        pixel in every other run; one row per non-empty bin
  TDC_CAL_MODES_CSV     every pixel of every board of every run: CAL mode and hit counts before
                        the ea cut, after it and after the CAL window

The drawing (hists, cal_maps) takes its words as arguments.
"""
import os
import re
import warnings

import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator

from ..campaigns import active as campaign
from .. import style
from ..resolution import pixel_maps

NPIX = 16
NPIXELS = NPIX * NPIX
NBOARDS = 4
CAL_WINDOW = 3
NCODE = 1024                     # CAL, TOA and TOT are 10-bit codes
NS_BIN = 0.1                     # ns bin of the tables
QUANTITIES = ("cal_code", "toa_code", "tot_code", "toa_ns", "tot_ns")
TDC_HISTS_COLUMNS = ["campaign", "tel", "run", "board_idx", "chip", "row", "col", "quantity",
                     "bin_lo", "count"]
TDC_CAL_COLUMNS = ["campaign", "tel", "run", "board_idx", "chip", "row", "col", "cal_mode",
                   "n_hits", "n_hits_ea0", "n_hits_all", "n_files"]
COLUMNS = ["board", "row", "col", "cal", "toa", "tot", "ea"]


# ------------------------------------------------------------------------------- gather
def loop_files(camp, tel, run, dirs=None):
    """Every loop_<N>.feather of a run, in loop order."""
    folder = (dirs or campaign.DECODED_DIRS)[camp].format(TEL=tel.upper(), tel=tel.lower(),
                                                          run=int(run))
    loops = []
    for name in os.listdir(folder):
        m = re.match(r"^loop_(\d+)\.feather$", name)
        if m:
            loops.append(int(m.group(1)))
    if not loops:
        raise FileNotFoundError("no loop_<N>.feather in %s" % folder)
    return [os.path.join(folder, "loop_%d.feather" % i) for i in sorted(loops)]


def file_modes(pix, cal):
    """Per pixel (256), the most frequent CAL code among `cal` (NaN for a pixel without hits);
    ties go to the lowest code."""
    counts = np.bincount(pix * NCODE + cal, minlength=NPIXELS * NCODE).reshape(NPIXELS, NCODE)
    mode = np.full(NPIXELS, np.nan)
    have = counts.sum(axis=1) > 0
    mode[have] = np.argmax(counts[have], axis=1)
    return mode


_KEY = 10 ** 7          # ns histograms are keyed pixel * _KEY + bin + _KEY_OFF
_KEY_OFF = 10 ** 6


def _ns_counts(pix, x):
    """pd.Series {pixel * _KEY + 0.1 ns bin index + _KEY_OFF: hits}."""
    key = pix * _KEY + np.floor(x / NS_BIN).astype(np.int64) + _KEY_OFF
    k, c = np.unique(key, return_counts=True)
    return pd.Series(c, index=k, dtype=np.int64)


def gather(camp, tel, run, dirs=None, log=None):
    """Histograms of one run (module docstring): dict of
    hist[quantity]   (4, 256, 1024) count arrays for the codes; for the ns quantities one
                     pd.Series per board, {pixel * _KEY + floor(value / 0.1 ns) + _KEY_OFF: hits}
    cal_mode         (4, 256) median over files of the per-file CAL mode
    n_hits, n_hits_ea0, n_hits_all   (4, 256) hits after the CAL window, after the ea cut, all
    n_files          loop files read
    `log(text)` is told about each file."""
    paths = loop_files(camp, tel, run, dirs)
    codes = {q: np.zeros((NBOARDS, NPIXELS, NCODE), dtype=np.int64)
             for q in ("cal_code", "toa_code", "tot_code")}
    ns = {q: [pd.Series(dtype=np.int64) for _ in range(NBOARDS)] for q in ("toa_ns", "tot_ns")}
    n = {k: np.zeros((NBOARDS, NPIXELS), dtype=np.int64)
         for k in ("n_hits", "n_hits_ea0", "n_hits_all")}
    modes = np.full((NBOARDS, NPIXELS, len(paths)), np.nan)
    for fi, path in enumerate(paths):
        df = pd.read_feather(path, columns=COLUMNS)
        board = df["board"].to_numpy(np.int64)
        pix_all = df["row"].to_numpy(np.int64) * NPIX + df["col"].to_numpy(np.int64)
        ea0 = df["ea"].to_numpy() == 0
        for b in range(NBOARDS):
            on = board == b
            n["n_hits_all"][b] += np.bincount(pix_all[on], minlength=NPIXELS)
            sel = on & ea0
            pix = pix_all[sel]
            if not pix.size:
                continue
            cal = df["cal"].to_numpy(np.int64)[sel]
            toa = df["toa"].to_numpy(np.float64)[sel]
            tot = df["tot"].to_numpy(np.float64)[sel]
            n["n_hits_ea0"][b] += np.bincount(pix, minlength=NPIXELS)
            mode = file_modes(pix, cal)
            modes[b, :, fi] = mode
            m = mode[pix]
            ok = (np.abs(cal - m) <= CAL_WINDOW) & (m > 0)       # NaN mode: False
            pix, cal, toa, tot, m = pix[ok], cal[ok], toa[ok], tot[ok], m[ok]
            n["n_hits"][b] += np.bincount(pix, minlength=NPIXELS)
            for q, v in (("cal_code", cal), ("toa_code", toa), ("tot_code", tot)):
                codes[q][b] += np.bincount(pix * NCODE + np.floor(v).astype(np.int64),
                                           minlength=NPIXELS * NCODE).reshape(NPIXELS, NCODE)
            for q, v in (("toa_ns", 12.5 - toa * 3.125 / m),
                         ("tot_ns", (2.0 * tot - np.floor(tot / 32.0)) * 3.125 / m)):
                ns[q][b] = ns[q][b].add(_ns_counts(pix, v), fill_value=0).astype(np.int64)
        if log:
            log("%s %s run %s: %s (%d of %d)" % (camp, tel, run, os.path.basename(path), fi + 1,
                                                 len(paths)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)         # a pixel without hits: NaN
        cal_mode = np.nanmedian(modes, axis=-1)
    return dict(hist=dict(codes, **ns), cal_mode=cal_mode, n_files=len(paths), **n)


def busiest_pixels(g):
    """{board: (row, col)}: the pixel with the most hits after the cuts (lowest index on a tie)."""
    return {b: divmod(int(np.argmax(g["n_hits"][b])), NPIX) for b in range(NBOARDS)}


def hist_rows(g, camp, tel, run, pixels):
    """TDC_HISTS_COLUMNS rows of one run for the pixels {board: (row, col)}: non-empty bins
    only; bin_lo is the code, or the bin's lower edge in ns."""
    rows = []
    for b, (r, c) in pixels.items():
        p = r * NPIX + c
        base = dict(campaign=camp, tel=tel, run=int(run), board_idx=b,
                    chip=campaign.TELESCOPE_CHIPS[tel][b], row=r, col=c)
        for q in QUANTITIES:
            h = g["hist"][q][b]
            if q.endswith("_code"):
                items = [(k, int(v)) for k, v in enumerate(h[p]) if v]
            else:
                sel = h[(h.index // _KEY) == p]
                items = sorted((int(k % _KEY - _KEY_OFF), int(v)) for k, v in sel.items() if v)
            for k, v in items:
                rows.append(dict(base, quantity=q,
                                 bin_lo=k if q.endswith("_code") else round(k * NS_BIN, 1),
                                 count=v))
    return rows


def cal_rows(g, camp, tel, run):
    """TDC_CAL_COLUMNS rows of one run: every pixel of every board."""
    rows = []
    for b in range(NBOARDS):
        chip = campaign.TELESCOPE_CHIPS[tel][b]
        for p in range(NPIXELS):
            r, c = divmod(p, NPIX)
            rows.append(dict(campaign=camp, tel=tel, run=int(run), board_idx=b, chip=chip,
                             row=r, col=c, cal_mode=g["cal_mode"][b, p],
                             n_hits=int(g["n_hits"][b, p]), n_hits_ea0=int(g["n_hits_ea0"][b, p]),
                             n_hits_all=int(g["n_hits_all"][b, p]), n_files=g["n_files"]))
    return rows


def read_hists(path=None):
    return pd.read_csv(path or campaign.TDC_PIXEL_HISTS_CSV)


def read_cal_modes(path=None):
    return pd.read_csv(path or campaign.TDC_CAL_MODES_CSV)


# ------------------------------------------------------------------------------- selections
def pixel_hist(h, camp, tel, run, board_idx, quantity):
    """(bin_lo array, count array) of one pixel's histogram from the hists table."""
    s = h[(h["campaign"] == camp) & (h["tel"] == tel) & (h["run"] == int(run))
          & (h["board_idx"] == board_idx) & (h["quantity"] == quantity)].sort_values("bin_lo")
    return s["bin_lo"].to_numpy(float), s["count"].to_numpy(np.int64)


def rebin(lo, counts, width, x0, x1, native):
    """Counts in bins of `width` from x0 to x1, summed from bins of `native` width whose lower
    edges are `lo`; (edges, counts in range, counts outside)."""
    nb = int(round((x1 - x0) / width))
    k = np.floor((lo - x0) / native + 1e-6).astype(np.int64) // int(round(width / native))
    inside = (k >= 0) & (k < nb)
    out = np.bincount(k[inside], weights=counts[inside], minlength=nb).astype(np.int64)
    return x0 + np.arange(nb + 1) * width, out, int(counts[~inside].sum())


def median_lo(lo, counts, native):
    """Median of a binned distribution: the lower edge of the native bin holding the 50 % point."""
    if counts.sum() <= 0:
        return float("nan")
    cum = np.cumsum(counts)
    return float(lo[int(np.searchsorted(cum, 0.5 * cum[-1], side="left"))])


def mode_grid(cm, camp, tel, run, board_idx):
    """16 x 16 CAL modes of one board in one run [row, col]; NaN where the pixel has no mode or a
    mode of 0."""
    s = cm[(cm["campaign"] == camp) & (cm["tel"] == tel) & (cm["run"] == int(run))
           & (cm["board_idx"] == board_idx)]
    g = np.full((NPIX, NPIX), np.nan)
    g[s["row"].to_numpy(int), s["col"].to_numpy(int)] = s["cal_mode"].to_numpy(float)
    g[~(g > 0)] = np.nan
    return g


# ------------------------------------------------------------------------------- drawing
HIST_SPECS = {   # quantity drawn -> (table quantity, native bin, unit)
    "cal": ("cal_code", 1.0, "code"),
    "toa": ("toa_ns", NS_BIN, "ns"),
    "tot": ("tot_ns", NS_BIN, "ns"),
}


def _grid(nrows, ncols, aw, ah, ml, mr, mt, mb, gx, gy, scale):
    import matplotlib.pyplot as plt
    style.apply_style(scale)
    W = ml + ncols * aw + (ncols - 1) * gx + mr
    H = mb + nrows * ah + (nrows - 1) * gy + mt
    fig = plt.figure(figsize=(W, H))
    axes = [[fig.add_axes([(ml + j * (aw + gx)) / W, (mb + (nrows - 1 - i) * (ah + gy)) / H,
                           aw / W, ah / H]) for j in range(ncols)] for i in range(nrows)]
    return fig, axes, W, H


def hists(h, tel, looks, panels, *, pixels, tag, data, footer, stat_words, scale=0.6):
    """One telescope: a row per chip (its pixel `pixels[board]`), a column per quantity
    (`panels`: [dict(key "cal"/"toa"/"tot", width, range or None, xlabel), ...]), each look of
    `looks` ([dict(campaign, run, label, color, fill), ...], before cooling down first) as a
    unit-area histogram. CAL keeps native codes over the populated codes +-8; TOA and TOT are
    summed from the 0.1 ns bins of the table into `width` over a fixed `range`. In each panel the
    looks' CAL mode or median in a box (`stat_words` = {"cal": "CAL mode", "toa": "median", ...}).
    Returns (figure, payload): per chip and panel, every look's hits in range, hits outside it and
    the box value."""
    nb = len(campaign.TELESCOPE_CHIPS[tel])
    fig, axes, W, H = _grid(nb, len(panels), aw=4.2, ah=2.75, ml=0.95, mr=0.3, mt=0.62, mb=0.85,
                            gx=0.75, gy=0.85, scale=scale)
    s = style.sizes(scale)
    payload = []
    for b in range(nb):
        r, c = pixels[b]
        chip = campaign.TELESCOPE_CHIPS[tel][b]
        for j, spec in enumerate(panels):
            ax = axes[b][j]
            q, native, unit = HIST_SPECS[spec["key"]]
            drawn = []
            for lk in looks:
                lo, cnt = pixel_hist(h, lk["campaign"], tel, lk["run"], b, q)
                drawn.append((lk, lo, cnt))
            if spec.get("range") is None:      # native codes, populated range +- margin
                nz = np.concatenate([lo[cnt > 0] for _, lo, cnt in drawn])
                x0, x1 = float(nz.min()) - spec.get("margin", 8), float(nz.max()) + 1 + spec.get("margin", 8)
            else:
                x0, x1 = spec["range"]
            ymax = 0.0
            rec = dict(chip=chip, board_idx=b, row=r, col=c, panel=spec["key"], looks=[])
            for lk, lo, cnt in drawn:
                edges, cin, cout = rebin(lo, cnt, spec["width"], x0, x1, native)
                dens = cin / cin.sum() if cin.sum() else cin.astype(float)   # fraction per bin
                x = np.repeat(edges, 2)[1:-1]
                y = np.repeat(dens, 2)
                if lk.get("fill"):
                    ax.fill_between(x, y, 0, color=lk["color"], alpha=0.35, linewidth=0, zorder=1)
                    ax.plot(x, y, color=lk["color"], linewidth=1.2, zorder=2)
                else:
                    ax.plot(x, y, color=lk["color"], linewidth=1.9, zorder=3)
                ymax = max(ymax, float(dens.max()) if dens.size else 0.0)
                if spec["key"] == "cal":
                    v = float(lo[np.argmax(cnt)]) if cnt.size else float("nan")
                else:
                    v = median_lo(lo, cnt, native)
                rec["looks"].append(dict(campaign=lk["campaign"], run=lk["run"],
                                         hits_in_range=int(cin.sum()), hits_outside=cout,
                                         value=v, bin_width=spec["width"], range=[x0, x1]))
            payload.append(rec)
            ax.set_xlim(x0, x1)
            if spec["key"] == "cal":
                ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            ax.set_ylim(0, (ymax or 1.0) * 1.42)
            ax.set_xlabel(spec["xlabel"])
            ax.set_ylabel("Fraction of hits / %g %s" % (spec["width"], unit), fontsize=s["label"] * 0.8)
            style.style_axes(ax, scale)
            fmt = "%.0f" if spec["key"] == "cal" else "%.2f ns"
            box = "\n".join("%s: %s" % (lk["label"], fmt % lr["value"])
                            for lk, lr in zip(looks, rec["looks"]))
            ax.text(0.03, 0.95, "%s\n%s" % (stat_words[spec["key"]], box), transform=ax.transAxes,
                    ha="left", va="top", fontsize=s["ann"] * 0.95, color=style.INK,
                    linespacing=1.3, zorder=8)
            if b > 0:
                style.right_lines(ax, ["%s, pixel (%d, %d)" % (chip, r, c)], s["title"])
    subjects = ["%s, pixel (%d, %d)" % (campaign.TELESCOPE_CHIPS[tel][0], *pixels[0])] * len(panels)
    style.compound_header(axes[0], tag=tag, data=data, scale=scale, subjects=subjects)
    handles = [Patch(facecolor=lk["color"], edgecolor=lk["color"], alpha=0.6, label=lk["legend"])
               if lk.get("fill") else Line2D([], [], color=lk["color"], linewidth=2.2,
                                             label=lk["legend"]) for lk in looks]
    axes[0][-1].legend(handles=handles, loc="upper right", fontsize=s["legend"] * 0.85,
                       borderaxespad=0.4)
    style.footer(fig, footer, scale)
    return fig, payload


def ring(ax, row, col, color="#ff2d2d"):
    """A square around pixel (row, col) of a map drawn by pixel_maps.draw_map."""
    ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor=color,
                           linewidth=1.8, zorder=6))


def cal_maps(grids, tel, looks, *, pixels, norm, diff_norm, titles, cbar_labels, tag, data,
             footer, cmap=None, diff_cmap="RdBu_r", scale=0.6):
    """One telescope's CAL-mode maps: a row per chip, columns = the two looks of `looks`
    ([dict(campaign, run, ...), before, after]) and their difference (after - before), each
    16 x 16 in the chip frame (pixel_maps.draw_map), the pixel of the histograms ringed.
    grids {(campaign, board): 16 x 16}; norm, diff_norm: the notebook's colour scales (one for
    every CAL map, one symmetric for every difference); titles(chip, column index) -> subject;
    cbar_labels (CAL, difference). Saturated pixels (outside the scale) counted per map.
    Returns (figure, payload)."""
    import matplotlib.pyplot as plt
    cmap = cmap or pixel_maps.CMAP
    if isinstance(diff_cmap, str):          # a pixel without a mode is grey on every map
        diff_cmap = plt.get_cmap(diff_cmap).copy()
        diff_cmap.set_bad(color=pixel_maps.GREY_MISSING)
    nb = len(campaign.TELESCOPE_CHIPS[tel])
    aw, ml, gx = 4.2, 0.75, 0.7
    mb = 1.75 + style.footer_extra_in(footer, scale, fits=3)   # room for a longer footer
    fig, axes, W, H = _grid(nb, 3, aw=aw, ah=aw, ml=ml, mr=0.3, mt=0.62, mb=mb, gx=gx, gy=0.75,
                            scale=scale)
    s = style.sizes(scale)
    payload = []
    ims = [None, None]
    for b in range(nb):
        chip = campaign.TELESCOPE_CHIPS[tel][b]
        before = grids[(looks[0]["campaign"], b)]
        after = grids[(looks[1]["campaign"], b)]
        diff = after - before
        rec = dict(chip=chip, board_idx=b, pixel=list(pixels[b]))
        for j, (g, nm, cm) in enumerate(((before, norm, cmap), (after, norm, cmap),
                                         (diff, diff_norm, diff_cmap))):
            ax = axes[b][j]
            im = pixel_maps.draw_map(ax, np.ma.masked_invalid(g), cmap=cm, norm=nm)
            ims[0 if j < 2 else 1] = im
            ring(ax, *pixels[b])
            sat = pixel_maps.saturated_cells(g, nm.vmin, nm.vmax)
            key = ("before", "after", "difference")[j]
            rec[key] = dict(grid=np.where(np.isfinite(g), g, np.nan).tolist(),
                            n_pixels=int(np.isfinite(g).sum()), saturated=len(sat))
            subj = titles(chip, j) + (", sat %d" % len(sat) if sat else "")
            if b > 0:
                style.right_lines(ax, [subj], s["title"])
            else:
                rec["_subject"] = subj
        payload.append(rec)
    subjects = []
    for j in range(3):
        g = (grids[(looks[0]["campaign"], 0)], grids[(looks[1]["campaign"], 0)],
             grids[(looks[1]["campaign"], 0)] - grids[(looks[0]["campaign"], 0)])[j]
        nm = diff_norm if j == 2 else norm
        sat = pixel_maps.saturated_cells(g, nm.vmin, nm.vmax)
        subjects.append(titles(campaign.TELESCOPE_CHIPS[tel][0], j)
                        + (", sat %d" % len(sat) if sat else ""))
    for rec in payload:
        rec.pop("_subject", None)
    style.compound_header(axes[0], tag=tag, data=data, scale=scale, subjects=subjects)
    style.footer(fig, footer, scale)
    # The footer band first: a colour bar's axes keep their figure fraction when the figure grows
    # (matplotlib places them by a locator), so they are placed in the final figure.
    style.lower_footer(fig)
    W2, H2 = fig.get_size_inches()
    y = (mb - 1.13 + (H2 - H)) / H2        # 1.13 in below the bottom row of maps
    # colour bars under the bottom row: CAL under the two look columns, difference under the third
    for k, (x0, x1, lab) in enumerate(((ml, ml + 2 * aw + gx, cbar_labels[0]),
                                       (ml + 2 * (aw + gx), ml + 2 * (aw + gx) + aw,
                                        cbar_labels[1]))):
        cax = fig.add_axes([x0 / W2, y, (x1 - x0) / W2, 0.13 / H2])
        cb = fig.colorbar(ims[k], cax=cax, orientation="horizontal", extend="both")
        cb.set_label(lab, fontsize=s["label"] * 0.75, color=style.INK, labelpad=2)
        cb.ax.xaxis.set_major_locator(MaxNLocator(integer=True))    # CAL codes are integers
        cb.ax.tick_params(labelsize=s["tick"] * 0.8)
    return fig, payload


def symmetric_norm(lim):
    return Normalize(vmin=-lim, vmax=lim)
