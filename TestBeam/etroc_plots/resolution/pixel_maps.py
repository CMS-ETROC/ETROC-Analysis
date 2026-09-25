"""Per-pixel time-resolution maps: one 16 x 16 map per board, read from the pixel table.

A board's map holds, for every pixel with a quote, the pixel's time resolution (value_ps) and
its fit error (err_ps) from the pixel table, anointed combination, one track variant and event
floor (resolution/tables.py). The board value and pixel spread come from the board table. The
maps are drawn in the chip frame (both axes inverted: pixel row 0, column 0 at the bottom
right), on one fixed colour scale, VMIN to VMAX, for every map of every figure; a pixel outside
it is saturated and counted, a pixel without a quote is grey.

Three figure layouts, each with its single panels:

compound     one run: the four boards' maps in a row with one colour bar, and below each map
             the histogram of its pixel values with a binned Gaussian fit and its pulls
ladder       one telescope over runs: a row per board, a column per run (fluence step),
             columns grouped by campaign
pair         one telescope, two runs side by side, each a 2 x 2 block of the four boards

The layouts are drawn for telescopes of four boards (TELESCOPE_CHIPS in the campaign module).

The words on the figures (header lines, captions, footers) are arguments: the notebooks hold
them. Board dictionaries (run_boards) carry the numbers every layout draws; the drawing adds
the histogram fit to them under "fit".
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.optimize import curve_fit

from ..campaigns import active as _campaign
from .. import style
from . import tables

NPIX = 16
VMIN, VMAX = 0.0, 120.0            # ps, every map of every figure
NORM = Normalize(vmin=VMIN, vmax=VMAX)
GREY_MISSING = (0.85, 0.85, 0.85, 1.0)
CMAP = plt.get_cmap("viridis").copy()      # every map; a pixel without a quote is grey
CMAP.set_bad(color=GREY_MISSING)
N_MIN_FIT = 30                     # fewer pixels than this: the histogram is drawn without a fit
HIST_MAIN_FRAC = 0.68              # compound histogram row: share of the height above the pulls
HIST_PULLS_GAP_IN = 0.10
HIST_YLABEL = "Pixels / 1 ps"
HIST_LABEL_PT = 9.4               # both axis labels of a histogram
MU = u"\N{GREEK SMALL LETTER MU}"
SIGMA = u"\N{GREEK SMALL LETTER SIGMA}"
PM = u"\N{PLUS-MINUS SIGN}"


# ---------------------------------------------------------------------------- data
def pixel_grids(px):
    """{(campaign, telescope, run, board_idx): (value grid, error grid)} from pixel-table rows,
    each grid NPIX x NPIX indexed [row, col], NaN where the pixel has no quote. `run` is the
    table's run as text."""
    grids = {}
    for key, g in px.groupby(["campaign", "telescope", "run", "board_idx"]):
        val = np.full((NPIX, NPIX), np.nan)
        err = np.full((NPIX, NPIX), np.nan)
        r, c = g["row"].to_numpy(int), g["col"].to_numpy(int)
        val[r, c] = g["value_ps"].to_numpy(float)
        err[r, c] = g["err_ps"].to_numpy(float)
        grids[(key[0], key[1], str(key[2]), int(key[3]))] = (val, err)
    return grids


def saturated_cells(grid):
    return [(r, c) for r in range(NPIX) for c in range(NPIX)
            if np.isfinite(grid[r, c]) and not (VMIN <= grid[r, c] <= VMAX)]


def run_boards(bt, grids, rs, campaign, telescope, run, variant, floor, band=False):
    """The boards of one run, in slot order, as dictionaries: chip, board_idx, bias (hv, from
    runs_summary), combo and combo_chips (the anointed combination), value_ps, pixel_spread_ps,
    stat_ps, n_pixels (board table), grid and err_grid (pixel_grids), saturated_count. With
    `band`, also "band": tables.band, the board's value over every combination containing it.
    Raises ValueError when a board is missing or its map does not hold n_pixels quotes."""
    rows = tables.board_rows(bt, telescope, run, variant, floor, campaign)
    chips = _campaign.TELESCOPE_CHIPS[telescope]
    if list(rows["board_idx"]) != list(range(len(chips))):
        raise ValueError("%s %s run %s: anointed boards %s, expected one row per board 0-%d"
                         % (campaign, telescope, run, list(rows["board_idx"]), len(chips) - 1))
    settings = {b["board_idx"]: b for b in
                tables.run_settings(rs, campaign, _campaign.TABLE_TEL[telescope], run)}
    boards = []
    for _, r in rows.iterrows():
        i = int(r["board_idx"])
        empty = np.full((NPIX, NPIX), np.nan)
        grid, err = grids.get((campaign, telescope, str(run), i), (empty, empty.copy()))
        chip = style.chip_short(r["board_chip"])
        # the layouts label rows, combinations and biases by slot: every source must agree
        rs_chip = style.chip_short(settings[i]["chip"]) if i in settings else None
        if not chip == rs_chip == chips[i]:
            raise ValueError("%s %s run %s board %d: board table %s, runs_summary %s, "
                             "TELESCOPE_CHIPS %s" % (campaign, telescope, run, i, chip, rs_chip,
                                                     chips[i]))
        if settings[i]["hv"] is None:
            raise ValueError("%s %s run %s %s: no bias in runs_summary"
                             % (campaign, telescope, run, chip))
        n_found = int(np.isfinite(grid).sum())
        if n_found != int(r["n_pixels"]):
            raise ValueError("%s %s run %s %s: the pixel table has %d quotes, the board table "
                             "n_pixels = %d" % (campaign, telescope, run, chip, n_found,
                                                int(r["n_pixels"])))
        b = dict(chip=chip, board_idx=i, hv=settings[i]["hv"],
                 combo=r["combo"], combo_chips=tables.combo_chips(r["combo"], telescope),
                 value_ps=float(r["value_ps"]), pixel_spread_ps=float(r["pixel_spread_ps"]),
                 stat_ps=float(r["stat_ps"]), n_pixels=int(r["n_pixels"]), grid=grid,
                 err_grid=err, saturated_count=len(saturated_cells(grid)))
        if band:
            b["band"] = tables.band(bt, telescope, run, i, variant, floor, campaign)
            if b["band"]["n_combo_rows"] != len(chips) - 1:   # every 3-board combination with it
                raise ValueError("%s %s run %s %s: band over %d combinations, expected %d"
                                 % (campaign, telescope, run, chip, b["band"]["n_combo_rows"],
                                    len(chips) - 1))
        boards.append(b)
    return boards


def board_values(b):
    """A board dictionary as the values file stores it: the numbers, with the maps as lists of
    rows (row 0 first, None where the pixel has no quote)."""
    out = {k: v for k, v in b.items() if k not in ("grid", "err_grid")}
    for k, name in (("grid", "value_map_ps"), ("err_grid", "error_map_ps")):
        out[name] = [[None if not np.isfinite(v) else round(float(v), 4) for v in row]
                     for row in b[k]]
    return out


# ---------------------------------------------------------------------------- one map
def label_color(rgba):
    """Ink on a light cell, white on a dark one, from the cell's luminance."""
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "#ffffff" if lum < 0.55 else style.INK


def draw_map(ax, grid, cmap=CMAP, norm=NORM):
    """One 16 x 16 map in the chip frame: row 0 drawn at the top (origin "upper"), then both
    axes inverted."""
    im = ax.imshow(grid, cmap=plt.get_cmap(cmap), norm=norm, interpolation="nearest",
                   origin="upper")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xticks(range(NPIX))
    ax.set_yticks(range(NPIX))
    lab = [str(v) if v % 3 == 0 else "" for v in range(NPIX)]
    ax.set_xticklabels(lab, fontsize=8.2)
    ax.set_yticklabels(lab, fontsize=8.2)
    ax.set_xlabel("Column", fontsize=10.2, labelpad=3)
    ax.set_ylabel("Row", fontsize=10.2, labelpad=2)
    ax.tick_params(axis="both", which="both", length=2.6)
    ax.minorticks_off()
    for sp in ax.spines.values():
        sp.set_color(style.INK)
    return im


def value_font_size(fig, ax):
    """Point size of the two-line "value / error" text in a cell of `ax`."""
    cell_pt = ax.get_position().width * fig.get_figwidth() * 72.0 / NPIX
    return float(max(4.0, min(6.4, 0.235 * cell_pt)))


def annotate_map(ax, grid, err_grid, fs, cmap=CMAP, norm=NORM):
    """Write each pixel's value and error into its cell, in ink or white by the cell's colour."""
    cm = plt.get_cmap(cmap)
    for r in range(NPIX):
        for c in range(NPIX):
            v = grid[r, c]
            if not np.isfinite(v):
                continue
            e = err_grid[r, c] if err_grid is not None else np.nan
            t = (u"%.1f\n%s%.1f" % (v, PM, e)) if np.isfinite(e) else (u"%.1f" % v)
            ax.text(c, r, t, ha="center", va="center", fontsize=fs, linespacing=0.95,
                    color=label_color(cm(norm(v))), zorder=7)


def colorbar(fig, cax, im, label_size, tick_size):
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
    cb.set_label(style.YLABEL_RES, fontsize=label_size, color=style.INK, labelpad=2)
    cb.ax.tick_params(labelsize=tick_size)
    cb.outline.set_edgecolor(style.INK)
    return cb


def header(ax, *, name=None, tag=None, data=None, scale=1.0):
    """style.header with `data` on the facility line after the campaign's HEADER_LINE_2, where
    style.compound_header puts it too: the subject line (`name`, `tag`) sits beside the
    experiment text and has to stay short."""
    line2 = ", ".join(x for x in (_campaign.HEADER_LINE_2, data) if x)
    return style.header(ax, name=name, tag=tag, line2=line2, scale=scale)


def header_axis(fig, ml_frac, y_frac, width_frac):
    """An invisible axis spanning the header band, to carry style.header above a grid of maps."""
    hax = fig.add_axes([ml_frac, y_frac, width_frac, 1e-6])
    hax.set_xticks([])
    hax.set_yticks([])
    for sp in hax.spines.values():
        sp.set_visible(False)
    return hax


# ---------------------------------------------------------------------------- histogram
def gauss(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2.0 * np.pi))


def draw_hist(ax_main, ax_pulls, b, scale=1.0, caption=None):
    """The board's pixel values in 1 ps bins, Poisson errors, over the board value +/- 6 pixel
    spreads; a binned Gaussian fit (area fixed to the number of pixels in the histogram, from
    N_MIN_FIT of them up) with its uncertainty band (200 draws from the fit covariance, seed 0)
    and the pulls below. `caption` goes in a box at the top right, followed by the number of
    pixels outside the histogram when there are any. Returns the fit and the histogram, also
    stored as b["fit"]."""
    vals = b["grid"][np.isfinite(b["grid"])]
    n = len(vals)
    fit_lo = b["value_ps"] - 6.0 * b["pixel_spread_ps"]
    fit_hi = b["value_ps"] + 6.0 * b["pixel_spread_ps"]
    lo = max(np.floor(vals.min()) - 0.5, np.floor(fit_lo) - 0.5) if n else fit_lo - 0.5
    hi = min(np.ceil(vals.max()) + 0.5, np.ceil(fit_hi) + 0.5) if n else fit_hi + 0.5
    bins = np.arange(lo, hi + 1.0, 1.0)
    counts, edges = np.histogram(vals, bins=bins)
    n_hist = int(counts.sum())
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]
    errs = np.sqrt(np.maximum(counts, 0.0))

    ax_main.errorbar(centers, counts, yerr=errs, fmt="o", ms=4.2, color="steelblue",
                     mfc="steelblue", mec="steelblue", ecolor="steelblue", capsize=1.5,
                     capthick=1.2, elinewidth=1.2, zorder=3, label="Pixels")

    fit = dict(n_pixels=int(n), n_in_hist=n_hist, n_outside=int(n) - n_hist,
               hist_lo_ps=float(edges[0]), hist_bin_ps=float(bin_width),
               hist_counts=[int(c) for c in counts], n_min_fit=N_MIN_FIT, fit_ok=False)
    if n_hist >= N_MIN_FIT:
        amplitude = n_hist * bin_width

        def _model(x, mu, sigma):
            return amplitude * gauss(x, mu, sigma)

        sigma_for_fit = np.maximum(errs, 1.0)
        try:
            popt, pcov = curve_fit(_model, centers, counts,
                                   p0=[b["value_ps"], max(b["pixel_spread_ps"], 0.5)],
                                   sigma=sigma_for_fit, absolute_sigma=True, maxfev=8000,
                                   bounds=([-np.inf, 1e-3], [np.inf, np.inf]))
            mu_fit, sigma_fit = float(popt[0]), float(abs(popt[1]))
            finite_cov = pcov is not None and np.all(np.isfinite(pcov))
            mu_err = float(np.sqrt(pcov[0, 0])) if finite_cov else float("nan")
            sigma_err = float(np.sqrt(pcov[1, 1])) if finite_cov else float("nan")

            x_range = np.linspace(edges[0], edges[-1], 300)
            y_fit = amplitude * gauss(x_range, mu_fit, sigma_fit)

            if finite_cov:
                rng = np.random.default_rng(0)
                samples = rng.multivariate_normal([mu_fit, sigma_fit], pcov, 200)
                sampled = [amplitude * gauss(x_range, s0, s1) for s0, s1 in samples if s1 > 0]
                if sampled:
                    spread = np.nanstd(np.vstack(sampled), axis=0)
                    ax_main.fill_between(x_range, y_fit - spread, y_fit + spread,
                                         color="hotpink", alpha=0.28, zorder=1,
                                         label="Fit uncertainty")

            ax_main.plot(x_range, y_fit, color="hotpink", ls="-", lw=2.2, alpha=0.9, zorder=2,
                         label=(u"fit %s = %.2f %s %.2f ps" % (MU, mu_fit, PM, mu_err)
                                if np.isfinite(mu_err) else u"fit %s = %.2f ps" % (MU, mu_fit)))
            ax_main.plot([], [], " ",
                         label=(u"fit %s = %.2f %s %.2f ps" % (SIGMA, sigma_fit, PM, sigma_err)
                                if np.isfinite(sigma_err)
                                else u"fit %s = %.2f ps" % (SIGMA, sigma_fit)))

            model_at_centers = amplitude * gauss(centers, mu_fit, sigma_fit)
            pulls = (counts - model_at_centers) / np.maximum(errs, 1.0)
            ax_pulls.axhline(0, c=style.INK, lw=1.0)
            ax_pulls.axhline(1, c=style.INK, lw=0.6, ls="--")
            ax_pulls.axhline(-1, c=style.INK, lw=0.6, ls="--")
            ax_pulls.bar(centers, pulls, width=bin_width * 0.9, fc="royalblue", alpha=0.75)
            ax_pulls.set_ylim(-3, 3)
            ax_pulls.set_yticks([-2, 0, 2])
            style.style_axes(ax_pulls, scale * 0.9, grid=True)
            # smaller than style_axes makes them: the label and the tick labels of the pulls
            # have to fit the gap between two histograms of a row
            ax_pulls.set_ylabel("Pulls", fontsize=style.sizes(0.6 * scale)["label"])
            ax_pulls.tick_params(axis="y", labelsize=style.sizes(0.7 * scale)["tick"])

            fit.update(fit_ok=True, mu_ps=mu_fit, mu_err_ps=mu_err, sigma_ps=sigma_fit,
                       sigma_err_ps=sigma_err)
        except (RuntimeError, ValueError) as exc:
            print("%s: histogram fit failed: %s" % (b["chip"], exc))
            ax_main.text(0.03, 0.95, u"fit failed (%d pixels)" % n_hist,
                         transform=ax_main.transAxes, ha="left", va="top", fontsize=7.6,
                         color=style.ALERT)
            ax_pulls.axis("off")
            ax_pulls.text(0.5, 0.5, "no fit", transform=ax_pulls.transAxes, ha="center",
                          va="center", fontsize=8.0, color=style.INK)
            fit["fit_error"] = str(exc)
    else:
        ax_main.text(0.03, 0.95, u"no fit: %d pixels, fewer than %d" % (n_hist, N_MIN_FIT),
                     transform=ax_main.transAxes, ha="left", va="top", fontsize=7.6,
                     color=style.ALERT)
        ax_pulls.axis("off")
        ax_pulls.text(0.5, 0.5, "no fit", transform=ax_pulls.transAxes,
                      ha="center", va="center", fontsize=8.0, color=style.INK)

    ax_main.set_xlim(edges[0], edges[-1])
    ax_pulls.set_xlim(edges[0], edges[-1])
    if fit["fit_ok"]:
        plt.setp(ax_main.get_xticklabels(), visible=False)
        ax_pulls.set_xlabel(style.YLABEL_RES, fontsize=HIST_LABEL_PT * scale)
    else:
        ax_main.set_xlabel(style.YLABEL_RES, fontsize=HIST_LABEL_PT * scale)
    if fit["n_outside"]:
        caption = u"\n".join(x for x in (caption, u"%d px outside the histogram range"
                                                   % fit["n_outside"]) if x)
    if caption:
        ax_main.text(0.97, 0.95, caption, transform=ax_main.transAxes, ha="right", va="top",
                     fontsize=8.6 * scale, color=style.INK,
                     bbox=dict(boxstyle="round,pad=0.25", facecolor=style.SURFACE,
                               edgecolor=style.GRID, alpha=0.9))
    if fit["fit_ok"]:
        ax_main.legend(fontsize=7.4 * scale, loc="upper left", framealpha=0.9)
    style.style_axes(ax_main, scale, grid=True)
    b["fit"] = fit
    return fit


# ---------------------------------------------------------------------------- compound
def compound_geometry():
    """Inches: the four maps in a row, the colour bar, the histogram row."""
    W = 22.0
    ML, MR = 0.95, 0.65
    MT = 0.80
    map_side, gap = 4.30, 0.55
    cb_gap, cb_h = 0.55, 0.22
    gap_rows = 1.25
    hist_h = 3.35
    MB = 1.55
    BODY = W - ML - MR
    row_w = 4.0 * map_side + 3.0 * gap
    x0 = ML + (BODY - row_w) / 2.0
    H = MT + map_side + cb_gap + cb_h + gap_rows + hist_h + MB
    return dict(W=W, H=H, ML=ML, MR=MR, MT=MT, map_side=map_side, gap=gap, cb_gap=cb_gap,
                cb_h=cb_h, gap_rows=gap_rows, hist_h=hist_h, MB=MB, x0=x0, row_w=row_w)


def _adder(fig, W, H):
    """add(x, y, w, h) in inches from the top left corner."""
    def add(x_in, y_in, w_in, h_in):
        return fig.add_axes([x_in / W, 1.0 - (y_in + h_in) / H, w_in / W, h_in / H])
    return add


def compound(boards, *, subjects, tag, data, footer, hist_captions, header_scale=0.85):
    """One run: the boards' maps in a row (one colour bar under them), each map's histogram
    below it. The header (style.compound_header): "CMS" + EXP_TEXT above the first map,
    `subjects[k]` on the right above map k, and above them on the right the campaign's facility
    line with `tag` and `data`. `hist_captions[k]` goes in the box of histogram k."""
    style.apply_style(1.0)
    geo = compound_geometry()
    W, H = geo["W"], geo["H"]
    fig = plt.figure(figsize=(W, H))
    add = _adder(fig, W, H)

    ims, map_axes = [], []
    for k, b in enumerate(boards):
        ax = add(geo["x0"] + k * (geo["map_side"] + geo["gap"]), geo["MT"], geo["map_side"],
                 geo["map_side"])
        im = draw_map(ax, b["grid"])
        annotate_map(ax, b["grid"], b["err_grid"], value_font_size(fig, ax))
        ims.append(im)
        map_axes.append(ax)

    style.compound_header(map_axes, tag=tag, data=data, scale=header_scale, subjects=subjects)

    cb_w = geo["row_w"] * 0.5
    cax = add(geo["x0"] + (geo["row_w"] - cb_w) / 2.0,
              geo["MT"] + geo["map_side"] + geo["cb_gap"], cb_w, geo["cb_h"])
    colorbar(fig, cax, ims[0], 10.0, 9.0)

    hist_y = geo["MT"] + geo["map_side"] + geo["cb_gap"] + geo["cb_h"] + geo["gap_rows"]
    hist_main_h = geo["hist_h"] * HIST_MAIN_FRAC
    hist_pulls_h = geo["hist_h"] - hist_main_h - HIST_PULLS_GAP_IN
    for k, b in enumerate(boards):
        x = geo["x0"] + k * (geo["map_side"] + geo["gap"])
        ax_m = add(x, hist_y, geo["map_side"], hist_main_h)
        ax_p = add(x, hist_y + hist_main_h + HIST_PULLS_GAP_IN, geo["map_side"], hist_pulls_h)
        draw_hist(ax_m, ax_p, b, caption=hist_captions[k])
        if k == 0:
            ax_m.set_ylabel(HIST_YLABEL, fontsize=HIST_LABEL_PT)

    style.footer(fig, footer, 1.0)
    return fig


def map_panel(b, *, subject, tag, data, caption, scale=0.5):
    """One board's map as a figure of its own, with its colour bar; `caption` below it. The
    header (header()): the facility line ending in `data`, then `subject` and `tag`."""
    style.apply_style(1.0)
    ML, MR, MB = 0.85, 0.65, 1.55
    MT = 1.05
    aw = 7.50
    W = ML + aw + MR
    cb_gap, cb_h = 0.55, 0.22
    H = MT + aw + cb_gap + cb_h + MB
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([ML / W, (MB + cb_gap + cb_h) / H, aw / W, aw / H])
    cax = fig.add_axes([(ML + 0.30) / W, MB / H, (aw - 0.60) / W, cb_h / H])
    header(ax, name=subject, tag=tag, data=data, scale=scale)
    im = draw_map(ax, b["grid"])
    annotate_map(ax, b["grid"], b["err_grid"], value_font_size(fig, ax))
    colorbar(fig, cax, im, 9.4, 8.4)
    style.footer(fig, caption, scale=0.7)
    return fig


def hist_panel(b, *, subject, tag, data, caption, scale=0.55):
    """One board's histogram, fit and pulls as a figure of its own."""
    style.apply_style(1.0)
    W, ML, MR, MB = 11.0, 0.90, 0.60, 1.10
    MT = 1.05
    aw_h = 6.40
    pulls_gap = 0.12
    pulls_h = 1.55
    main_h = aw_h - pulls_gap - pulls_h
    H = MT + aw_h + MB
    fig = plt.figure(figsize=(W, H))
    aw = (W - ML - MR)
    ax_m = fig.add_axes([ML / W, (MB + pulls_h + pulls_gap) / H, aw / W, main_h / H])
    ax_p = fig.add_axes([ML / W, MB / H, aw / W, pulls_h / H])
    header(ax_m, name=subject, tag=tag, data=data, scale=scale)
    draw_hist(ax_m, ax_p, b, caption=caption)
    ax_m.set_ylabel(HIST_YLABEL)
    for ax in (ax_m, ax_p):   # one label size and one tick size for both axes of the panel
        style.style_axes(ax, 0.6)
    return fig


# ---------------------------------------------------------------------------- ladder
def ladder_geometry(groups, n_rows):
    """Inches. `groups`: the number of columns in each campaign group, left to right."""
    cell = 2.20
    gap_x = 0.28
    camp_gap = 0.85
    row_w = 0.0
    for g, n in enumerate(groups):
        if g:
            row_w += camp_gap
        row_w += n * cell
        row_w += (n - 1) * gap_x
    ML, MR = 1.35, 0.65
    MT = 0.95
    col_label_h = 0.80
    gap_y = 0.50
    cb_gap, cb_h = 0.55, 0.24
    MB = 1.55
    W = ML + row_w + MR
    H = MT + col_label_h + n_rows * cell + (n_rows - 1) * gap_y + cb_gap + cb_h + MB
    return dict(W=W, H=H, ML=ML, MR=MR, MT=MT, cell=cell, gap_x=gap_x, camp_gap=camp_gap,
                groups=list(groups), row_w=row_w, col_label_h=col_label_h, gap_y=gap_y,
                cb_gap=cb_gap, cb_h=cb_h, MB=MB, n_rows=n_rows)


def ladder_col_x(geo, group, j):
    """x of column j of campaign group `group`, from the left edge of the grid."""
    cell, gap_x = geo["cell"], geo["gap_x"]
    x = 0.0
    for n in geo["groups"][:group]:
        x = x + n * (cell + gap_x) - gap_x + geo["camp_gap"]
    return x + j * (cell + gap_x) if group else j * (cell + gap_x)


def ladder(groups, row_labels, *, name, tag, data, footer, header_scale=0.85):
    """One telescope over runs: a row per board (`row_labels`, the chips), a column per run.
    `groups`: [(group label, [column, ...]), ...], left to right; a column is a dict with
    "label" (above it), "boards" (run_boards, one per row) and "captions" (one list of caption
    lines per row, drawn under the cell). Header (header()): `name` and `tag` on the subject
    line, `data` on the facility line."""
    style.apply_style(1.0)
    geo = ladder_geometry([len(cols) for _, cols in groups], len(row_labels))
    W, H = geo["W"], geo["H"]
    fig = plt.figure(figsize=(W, H))
    add = _adder(fig, W, H)
    hax = header_axis(fig, geo["ML"] / W, 1.0 - 0.8 * geo["MT"] / H,
                      (W - geo["ML"] - geo["MR"]) / W)
    header(hax, name=name, tag=tag, data=data, scale=header_scale)

    grid_top = geo["MT"] + geo["col_label_h"]
    im0 = None
    for g, (group_label, cols) in enumerate(groups):
        for j, col in enumerate(cols):
            cx = ladder_col_x(geo, g, j)
            if j == 0:
                n_this = len(cols)
                group_w = n_this * geo["cell"] + (n_this - 1) * geo["gap_x"]
                lx = geo["ML"] + cx
                fig.text((lx + group_w / 2.0) / W, 1.0 - (geo["MT"] + 0.30) / H, group_label,
                         ha="center", va="top", fontsize=12.5, color=style.INK,
                         fontweight="bold")
            fig.text((geo["ML"] + cx + geo["cell"] / 2.0) / W,
                     1.0 - (geo["MT"] + geo["col_label_h"] - 0.06) / H, col["label"],
                     ha="center", va="bottom", fontsize=10.0, color=style.INK)
            for row, b in enumerate(col["boards"]):
                ry = grid_top + row * (geo["cell"] + geo["gap_y"])
                ax = add(geo["ML"] + cx, ry, geo["cell"], geo["cell"])
                im = draw_map(ax, b["grid"])
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_xticks([])
                ax.set_yticks([])
                if im0 is None:
                    im0 = im
                cap_y = ry + geo["cell"] + 0.07
                for k, (line, fs) in enumerate(zip(col["captions"][row], (6.8, 6.2))):
                    fig.text((geo["ML"] + cx + geo["cell"] / 2.0) / W,
                             1.0 - (cap_y + 0.15 * k) / H, line, ha="center", va="top",
                             fontsize=fs, color=style.INK)
                if g == 0 and j == 0:
                    fig.text((geo["ML"] - 0.10) / W, 1.0 - (ry + geo["cell"] / 2.0) / H,
                             row_labels[row], ha="right", va="center", fontsize=11.5,
                             color=style.INK, fontweight="bold")

    cb_w = geo["row_w"] * 0.5
    cax = add(geo["ML"] + (geo["row_w"] - cb_w) / 2.0,
              grid_top + geo["n_rows"] * geo["cell"] + (geo["n_rows"] - 1) * geo["gap_y"]
              + geo["cb_gap"], cb_w, geo["cb_h"])
    colorbar(fig, cax, im0, 10.5, 9.2)
    style.footer(fig, footer, 1.0)
    return fig


# ---------------------------------------------------------------------------- pair of runs
POS = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}   # board_idx -> (row, col) in a 2 x 2 block


def pair_geometry(n_blocks=2):
    map_side = 3.30
    gap = 0.55              # between the two columns of one block: the Row label and ticks
    block_gap = 0.95        # between the blocks
    ML, MR = 0.95, 0.65
    MT = 1.05
    block_title_h = 0.30
    title_gap = 0.40        # room for the map title above the row-0 maps
    row_gap = 0.85          # the row-0 Column label and the map title above the row-1 maps
    cb_gap, cb_h = 0.55, 0.24
    MB = 1.55
    block_w = 2 * map_side + gap
    row_w = n_blocks * block_w + (n_blocks - 1) * block_gap
    W = ML + row_w + MR
    H = MT + block_title_h + title_gap + map_side + row_gap + map_side + cb_gap + cb_h + MB
    return dict(W=W, H=H, ML=ML, MR=MR, MT=MT, map_side=map_side, gap=gap, block_gap=block_gap,
                block_title_h=block_title_h, title_gap=title_gap, row_gap=row_gap, cb_gap=cb_gap,
                cb_h=cb_h, MB=MB, block_w=block_w, row_w=row_w, x0=ML)


def _block(fig, add, geo, block_index, boards, title, map_titles):
    """One 2 x 2 block of maps at `block_index` (0 = left), `map_titles` above the maps and
    `title` centred above the block. Returns the first map's image."""
    bx0 = geo["x0"] + block_index * (geo["block_w"] + geo["block_gap"])
    row0_y = geo["MT"] + geo["block_title_h"] + geo["title_gap"]
    row_y = {0: row0_y, 1: row0_y + geo["map_side"] + geo["row_gap"]}
    s = style.sizes(1.0)
    ims = []
    for bidx, b in enumerate(boards):
        r, c = POS[bidx]
        x = bx0 + c * (geo["map_side"] + geo["gap"])
        ax = add(x, row_y[r], geo["map_side"], geo["map_side"])
        im = draw_map(ax, b["grid"])
        annotate_map(ax, b["grid"], b["err_grid"], value_font_size(fig, ax))
        ax.set_title(map_titles[bidx], loc="center", fontsize=s["title"], color=style.INK, pad=6)
        ims.append(im)
    fig.text((bx0 + geo["block_w"] / 2.0) / geo["W"], 1.0 - geo["MT"] / geo["H"], title,
             ha="center", va="top", fontsize=s["title"], fontweight="bold", color=style.INK)
    return ims[0]


def _block_colorbar(fig, add, geo, im, width_frac, x_center_in):
    y = (geo["MT"] + geo["block_title_h"] + geo["title_gap"] + 2 * geo["map_side"]
         + geo["row_gap"] + geo["cb_gap"])
    cb_w = geo["row_w"] * width_frac if x_center_in is None else geo["block_w"] * width_frac
    cx = ((geo["row_w"] - cb_w) / 2.0 + geo["x0"] if x_center_in is None
          else x_center_in - cb_w / 2.0)
    colorbar(fig, add(cx, y, cb_w, geo["cb_h"]), im, 10.0, 9.0)


def _pair_header(fig, geo, name, tag, data, scale):
    W, H = geo["W"], geo["H"]
    hax = header_axis(fig, geo["ML"] / W, 1.0 - 0.8 * geo["MT"] / H,
                      (W - geo["ML"] - geo["MR"]) / W)
    header(hax, name=name, tag=tag, data=data, scale=scale)


def pair(blocks, *, name, tag, data, footer, header_scale=0.85, footer_scale=0.72):
    """Two runs of one telescope side by side: `blocks` = [(title, boards, map_titles), ...],
    each a 2 x 2 block of the boards in slot order, map_titles[k] above board k. Header: `name`
    and `tag` on the subject line, `data` on the facility line."""
    style.apply_style(1.0)
    geo = pair_geometry(n_blocks=len(blocks))
    W, H = geo["W"], geo["H"]
    fig = plt.figure(figsize=(W, H))
    add = _adder(fig, W, H)
    ims = [_block(fig, add, geo, k, boards, title, map_titles)
           for k, (title, boards, map_titles) in enumerate(blocks)]
    _pair_header(fig, geo, name, tag, data, header_scale)
    _block_colorbar(fig, add, geo, ims[0], 0.20, None)
    style.footer(fig, footer, footer_scale)
    return fig


def block_panel(title, boards, map_titles, *, name, tag, data, footer, footer_scale=0.75):
    """One block of a pair() figure as a figure of its own."""
    style.apply_style(1.0)
    geo = pair_geometry(n_blocks=1)
    W, H = geo["W"], geo["H"]
    fig = plt.figure(figsize=(W, H))
    add = _adder(fig, W, H)
    im = _block(fig, add, geo, 0, boards, title, map_titles)
    _pair_header(fig, geo, name, tag, data, 0.5)
    _block_colorbar(fig, add, geo, im, 0.36, geo["x0"] + geo["block_w"] / 2.0)
    style.footer(fig, footer, scale=footer_scale)
    return fig
