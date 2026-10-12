"""Setup and method cartoons: schematic figures, no measured data.

  telescope_layout  the two telescopes along the beam in one cold box, with the beam profile
  beam_profile      the flattened beam profile: three Gaussian exposures and their sum
  board_facts       the chips, sensors and HV channels of each telescope
  track_selection   how one resolution value per pixel and one number per board are chosen
  rfsel_explainer   what the ETROC2 RFSel setting changes in the front end

The chips, sensors and beam come from the campaign module (TELESCOPE_CHIPS, TELESCOPE_SENSORS,
BEAM_*) and R_f from style.RFSEL_RF_KOHM; the illustrative numbers (the tracks of figure 4, the
pulses of figure 5) and the selection they draw (TRACK_*) are constants of this module. Every
other word on a figure is an argument, so the notebook holds it. Each function returns the
figure (and, where it has panels, the single-panel figures on request) and draws nothing to
disk.
"""
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from ..campaigns import active as campaign
from .. import style
from ..etroc_style import BOARD_COLOR

INK, INK_MUTED, SURFACE, ALERT, GRID = (style.INK, style.INK_MUTED, style.SURFACE, style.ALERT,
                                        style.GRID)
BEAM_GREY, BEAM_DARK, BOX_BLUE, BOX_EDGE = "#8c8c88", "#5d5d5a", "#41556f", "#7f94b0"


def _clean(ax):
    """No ticks, no frame: a drawing canvas."""
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


# ============================================================================ beam profile
def beam_sigma_cm():
    """The beam's Gaussian sigma in cm, from its FWHM."""
    return campaign.BEAM_FWHM_CM / 2.355


def beam_profile_sum(x, exposures=None):
    """(components, weights): the unit-area Gaussian of each exposure, scaled by its share of
    the protons (shares normalised to sum 1), at positions x [cm]. Sum the components for the
    profile."""
    exposures = campaign.BEAM_EXPOSURES if exposures is None else exposures
    sig = beam_sigma_cm()
    w = np.array([e[1] for e in exposures], float)
    w /= w.sum()
    comps = np.array([wi * np.exp(-0.5 * ((x - c) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
                      for (c, _), wi in zip(exposures, w)])
    return comps, w


def window_metrics(exposures=None, n=20001):
    """Over the chip window [-BEAM_WINDOW_CM, BEAM_WINDOW_CM]: the profile's mean ("Avg"), RMS
    deviation from it ("RMSE") and max / min, as the facility's table defines them."""
    xw = np.linspace(-campaign.BEAM_WINDOW_CM, campaign.BEAM_WINDOW_CM, n)
    f = beam_profile_sum(xw, exposures)[0].sum(axis=0)
    avg = f.mean()
    return dict(avg=float(avg), rmse=float(np.sqrt(((f - avg) ** 2).mean())),
                max_over_min=float(f.max() / f.min()))


def beam_table_check():
    """Every row of the campaign's BEAM_TABLE recomputed with window_metrics: [dict(name, avg,
    avg_table, rmse, rmse_table)]. With sigma = FWHM / 2.355 the rows agree with the table to
    within 0.002 in avg and 0.0003 in rmse (the table's own sampling is not documented); with
    1.1 cm read as sigma they would not come close (see BEAM_TABLE in the campaign module)."""
    out = []
    for name, exposures, avg_t, rmse_t in campaign.BEAM_TABLE:
        m = window_metrics(exposures)
        out.append(dict(name=name, avg=round(m["avg"], 4), avg_table=avg_t,
                        rmse=round(m["rmse"], 5), rmse_table=rmse_t))
    return out


def beam_profile(name, tag, legend_texts, scale=0.90):
    """The flattened profile: each exposure, their sum, and one centred exposure of the same
    protons for comparison, over the chip window. `legend_texts`: dict with keys window, single,
    component (a format with {x} and {share}), total. Returns (fig, metrics of the sum)."""
    A = 1.375
    style.apply_style(scale)
    win = campaign.BEAM_WINDOW_CM
    x = np.linspace(-4, 4, 3201)
    comps, wts = beam_profile_sum(x)
    single = beam_profile_sum(x, ((0.0, 1),))[0][0]
    m = window_metrics()

    fig, ax = plt.subplots(figsize=(13.2, 9.0))
    ax.axvspan(-win, win, color="#c9c9c4", alpha=0.42, zorder=0,
               label=legend_texts["window"])
    for edge in (-win, win):
        ax.axvline(edge, color=BEAM_GREY, lw=1.0, ls="--", alpha=0.9, zorder=1)
    ax.plot(x, single, lw=1.6, color="#9a9a95", zorder=2, label=legend_texts["single"])
    for (c, _), w, comp, col in zip(campaign.BEAM_EXPOSURES, wts, comps,
                                    ("#2a78d6", "#199e70", "#4a3aa7")):
        ax.plot(x, comp, lw=1.9, color=col, alpha=0.95, zorder=3,
                label=legend_texts["component"].format(
                    x=("%+.1f" % c).replace("+0.0", "0.0").replace("-", u"−"),
                    share="%.0f" % (100 * w)))
    ax.plot(x, comps.sum(axis=0), lw=4.0, color=INK, zorder=4, label=legend_texts["total"])

    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 1.13)
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("Normalized probability density")
    ax.set_xticks(np.arange(-4, 5, 1))
    style.style_axes(ax, scale)
    leg = ax.legend(loc="upper left", handletextpad=0.7, fontsize=8.6 * A)
    leg._legend_box.align = "left"
    ax.text(0.978, 0.962,
            u"over the chip window,\n"
            u"x ∈ [−%.1f, %.1f] cm, this sum has\n"
            u"    Avg       = %.4f\n"
            u"    RMSE      = %.5f\n"
            u"    max / min = %.3f\n"
            u"    (%.1f %% spread over the window)"
            % (win, win, m["avg"], m["rmse"], m["max_over_min"], 100 * (m["max_over_min"] - 1)),
            transform=ax.transAxes, ha="right", va="top", ma="left", fontsize=8.8 * A,
            color=INK, linespacing=1.55, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#dcdcd8", lw=0.8))
    style.header(ax, name=name, tag=tag, scale=scale, pad=10)
    fig.tight_layout(pad=0.8)
    return fig, m


# ======================================================================== telescope layout
def telescope_layout(name, tag, telescope_labels, beam_text, box_text, profile_texts,
                     incidence_deg=60.0, scale=0.72):
    """The telescopes one after the other along the beam, each chip a plane at `incidence_deg`
    to the beam, all in one cold box, with the beam profile (beam_profile_sum) drawn at the
    entrance. telescope_labels {tel: text}; profile_texts: (label, window label)."""
    A = 1.507
    style.apply_style(scale)
    half, step, gap = 0.62, 1.15, 1.05
    fig, ax = plt.subplots(figsize=(22.0, 4.15))
    _clean(ax)
    tels = list(campaign.TELESCOPE_CHIPS)
    x0s, x = [], 1.62
    for _ in tels:
        x0s.append(x)
        x += 4 * step + gap
    xs_all = [x0 + i * step for x0 in x0s for i in range(4)]
    xmax = xs_all[-1] + half + 1.30
    ax.set_xlim(-2.05, xmax)
    ax.set_ylim(-0.95, 1.50)
    ax.set_aspect("equal")

    lo, hi = xs_all[0] - half - 0.42, xs_all[-1] + half + 0.42
    ax.add_patch(FancyBboxPatch((lo, -0.66), hi - lo, 1.82,
                                boxstyle="round,pad=0.08,rounding_size=0.16",
                                fc="#eef3fa", ec=BOX_EDGE, lw=1.4, ls="--", zorder=0))
    ax.text(hi - 0.06, -0.58, box_text, ha="right", va="bottom", fontsize=9.6 * A,
            color=BOX_BLUE, fontweight="bold")
    ax.add_patch(FancyArrowPatch((-1.98, 0), (xmax - 0.10, 0), arrowstyle="-|>",
                                 mutation_scale=24, lw=2.4, color=BEAM_GREY, zorder=1,
                                 alpha=0.85))
    ax.text(-1.98, 0.16, beam_text, ha="left", va="bottom", fontsize=9.5 * A, color=INK,
            fontweight="bold", linespacing=1.4)

    # the beam profile at the entrance, drawn vertically and peak-normalised
    ycm = np.linspace(-1.6, 1.6, 400)
    prof = beam_profile_sum(ycm)[0].sum(axis=0)
    prof /= prof.max()
    px, pamp, pscale = -0.35, 0.40, 0.5      # base x, amplitude, data units per cm
    yy = ycm * pscale
    ax.fill_betweenx(yy, px, px + pamp * prof, color=BEAM_GREY, alpha=0.32, lw=0, zorder=2)
    ax.plot(px + pamp * prof, yy, color=BEAM_DARK, lw=1.3, zorder=3)
    ax.plot([px, px], [yy[0], yy[-1]], color=BEAM_GREY, lw=0.8, ls=":", zorder=2)
    win = campaign.BEAM_WINDOW_CM
    for s_ in (-win, win):
        ax.plot([px - 0.05, px + pamp * 1.03], [s_ * pscale] * 2, color=BEAM_DARK, lw=0.7,
                ls="--", zorder=2)
    ax.text(px + pamp / 2, yy[-1] + 0.05, profile_texts[0], ha="center", va="bottom",
            fontsize=7.8 * A, color=BOX_BLUE, linespacing=1.2)
    ax.text(px + pamp + 0.05, win * pscale + 0.02, profile_texts[1], ha="left", va="bottom",
            fontsize=7.2 * A, color=BEAM_DARK)

    # planes lean towards the beam: top end upstream
    ang = np.radians(90.0 - incidence_deg)
    d = np.array([np.cos(ang), -np.sin(ang)])
    for tel, x0 in zip(tels, x0s):
        xs = [x0 + i * step for i in range(4)]
        for bi, (chip, xc) in enumerate(zip(campaign.TELESCOPE_CHIPS[tel], xs)):
            p0, p1 = np.array([xc, 0]) - half * d, np.array([xc, 0]) + half * d
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], lw=11.0, color=BOARD_COLOR[bi],
                    solid_capstyle="butt", zorder=4)
            ax.plot(xc, 0, "o", ms=5.0, mfc=SURFACE, mec=BOARD_COLOR[bi], mew=1.5, zorder=5)
            ax.annotate(chip, (p0[0], p0[1]), xytext=(-1, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9.4 * A, color=INK, zorder=6)
        ax.text(0.5 * (xs[0] + xs[-1]), 0.92, telescope_labels[tel], ha="center", va="bottom",
                fontsize=12.2 * A, color=INK, fontweight="bold")
    style.header(ax, name=name, tag=tag, scale=scale, pad=10)
    fig.tight_layout()
    return fig


# ============================================================================ board facts
def board_facts(name, telescope_notes, notes, columns=("board", "chip", "sensor (wafer, die)", "HV ch"),
                scale=0.75):
    """The boards of each telescope: index, chip, sensor (TELESCOPE_SENSORS) and HV channel
    (= the board index), a colour key per board index. telescope_notes {tel: text beside the
    telescope's name}; notes: the lines under the table."""
    A = 1.585
    style.apply_style(scale)
    fig, ax = plt.subplots(figsize=(13.0, 9.6))
    _clean(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    cols = [0.45, 1.55, 2.55, 4.05, 6.95, 8.55]
    for x, h in zip([cols[0], cols[2], cols[3], cols[5]], columns):
        ax.text(x, 9.35, h, fontsize=8.6 * A, color=INK_MUTED, fontweight="bold", va="center")
    ax.plot([0.1, 9.9], [9.02, 9.02], color=GRID, lw=1.2)
    y = 8.62
    for tel, chips in campaign.TELESCOPE_CHIPS.items():
        ax.text(0.10, y, "%s telescope" % tel.upper(), fontsize=9.0 * A, color=INK,
                fontweight="bold", va="center")
        ax.text(cols[3], y, telescope_notes[tel], fontsize=8.3 * A, color=INK_MUTED, va="center")
        y -= 0.62
        for bi, (chip, sensor) in enumerate(zip(chips, campaign.TELESCOPE_SENSORS[tel])):
            ax.add_patch(Rectangle((0.12, y - 0.19), 0.26, 0.38, fc=BOARD_COLOR[bi], ec="none"))
            ax.text(cols[0], y, "b%d" % bi, fontsize=8.6 * A, color=INK, va="center")
            ax.text(cols[2], y, chip, fontsize=8.6 * A, color=INK, va="center")
            ax.text(cols[3], y, sensor, fontsize=8.6 * A, color=INK_MUTED, va="center")
            ax.text(cols[5], y, "%d" % bi, fontsize=8.6 * A, color=INK, va="center")
            y -= 0.55
        y -= 0.30
    note = ax.text(0.1, y + 0.05, "\n".join(notes), fontsize=8.3 * A, color=INK_MUTED,
                   va="top", linespacing=1.55)
    style.header(ax, name=name, scale=scale, pad=10)
    # end the figure under the notes, keeping the inches per data unit
    fig.canvas.draw()
    bottom = note.get_window_extent().transformed(ax.transData.inverted()).y0 - 0.25
    ax.set_ylim(bottom, 10)
    fig.set_size_inches(13.0, 9.6 * (10 - bottom) / 10)
    fig.tight_layout()
    return fig


# ======================================================================== track selection
GOOD, OTHER = "#2a7f3f", "#9a9a9a"
COMBO_COLOR = "#7a3e9d"


def _track_combos(ax):
    """(a) the anointed combination of each board: 0-1-2 for boards 0-2, 1-2-3 for board 3."""
    _clean(ax)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(-1.5, 1.95)
    ax.set_aspect("equal")
    for i in range(4):
        ax.add_patch(Rectangle((i - 0.08, -0.8), 0.16, 1.6, fc=BOARD_COLOR[i], ec=INK, lw=0.8))
        ax.text(i, -0.98, "board %d" % i, ha="center", va="top", fontsize=9, color=INK)
    ax.set_anchor("N")
    ax.annotate("", xy=(3.55, 1.05), xytext=(-0.45, 1.05),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    ax.text(1.5, 1.12, "beam", ha="center", va="bottom", fontsize=9, color=INK)

    def bracket(x0, x1, y, txt, col):
        ax.plot([x0, x0, x1, x1], [y + 0.08, y, y, y + 0.08], color=col, lw=1.4)
        ax.text((x0 + x1) / 2, y - 0.06, txt, ha="center", va="top", fontsize=8.5, color=col)
    bracket(-0.25, 2.25, -1.2, u"combo 0-1-2  →  the number of boards 0, 1, 2", GOOD)
    bracket(0.75, 3.25, 1.62, u"combo 1-2-3  →  the number of board 3", COMBO_COLOR)


def _grid(ax, x0, y0, n, cell, col):
    for r in range(n):
        for c in range(n):
            ax.add_patch(Rectangle((x0 + c * cell, y0 + r * cell), cell, cell, fc="white",
                                   ec=col, lw=0.5, alpha=0.9))


# The selection figure 4 draws: pixels per board side, the event floor of a track and the clip
# of the robust Gaussian over a board's pixel values, as in the resolution chain
# (condor_at_lxplus/utils/quote_resolution.py) and the maps notebook (floor "300").
TRACK_PIXELS_PER_SIDE = 16
TRACK_EVENT_FLOOR = 300
TRACK_CLIP_SIGMA = 2.5
# Illustrative tracks through one pixel, cell (2, 2) of the middle board: (cells on the three
# boards, events, chosen). The counts are not measured.
TRACK_EXAMPLE = [([(2, 3), (2, 2), (2, 2)], 1840, True),
                 ([(3, 2), (2, 2), (1, 3)], 410, False),
                 ([(1, 1), (2, 2), (3, 3)], 120, False)]


def _track_most_populated(ax):
    """(b) the tracks through one pixel, with their event counts; the most populated one (at
    least TRACK_EVENT_FLOOR events) gives the pixel its value. The counts are illustrative."""
    _clean(ax)
    ax.set_xlim(-0.3, 7.7)
    ax.set_ylim(-1.6, 2.6)
    ax.set_aspect("equal")
    n, cell = 5, 0.34
    origins = [(0.0, 0.0), (2.3, 0.0), (4.6, 0.0)]
    labels = ["partner board (before)", u"this board · pixel (r, c)", "partner board (after)"]
    for (x0, y0), lab, col in zip(origins, labels, BOARD_COLOR.values()):
        _grid(ax, x0, y0, n, cell, col)
        ax.text(x0 + n * cell / 2, y0 - 0.12, lab, ha="center", va="top", fontsize=8, color=col)
    for cells, nevt, chosen in TRACK_EXAMPLE:
        col, ls, lw = (GOOD, "-", 2.0) if chosen else (OTHER, "--", 1.4)
        cx = [origins[i][0] + (cells[i][1] + 0.5) * cell for i in range(3)]
        cy = [origins[i][1] + (cells[i][0] + 0.5) * cell for i in range(3)]
        for i in (0, 2):
            ax.add_patch(Rectangle((origins[i][0] + cells[i][1] * cell,
                                    origins[i][1] + cells[i][0] * cell), cell, cell,
                                   fc=col if chosen else "none", ec=col, lw=1.6 if chosen else 1.4,
                                   ls="-" if chosen else "--", alpha=0.85 if chosen else 1.0))
        ax.plot(cx, cy, color=col, lw=lw, ls=ls, zorder=5 if chosen else 4)
        ax.text(origins[2][0] + n * cell + 0.12, cy[2], "%d events" % nevt, ha="left", va="center", fontsize=7.5,
                color=col, fontweight="bold" if chosen else "normal")
    ax.add_patch(Rectangle((origins[1][0] + 2 * cell, origins[1][1] + 2 * cell), cell, cell,
                           fc=GOOD, ec=GOOD, lw=1.6, alpha=0.85))
    ax.text(3.3, 1.9, u"the most-populated track through the pixel\n(≥ %d events) gives "
            u"its value" % TRACK_EVENT_FLOOR, ha="center", va="bottom", fontsize=8.5, color=GOOD)
    ax.text(3.3, -0.95, "the other tracks through it: not used for its value", ha="center",
            va="top", fontsize=8.5, color=OTHER)
    ax.annotate("", xy=(6.6, 2.5), xytext=(-0.1, 2.5),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0))
    ax.text(6.7, 2.5, "beam", ha="left", va="center", fontsize=9, color=INK)
    ax.set_anchor("N")


def _box(ax, x, y, w, h, txt, fc="white", ec=INK, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=INK)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                 color=INK, lw=1.0))


def _track_one_number(ax):
    """(c) 256 pixels -> one value each (those with a track above the floor) -> robust Gaussian
    over them -> the board's number."""
    _clean(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    fs = 8
    n = TRACK_PIXELS_PER_SIDE
    _box(ax, 0.15, 2.6, 2.2, 1.2, u"one board:\n%d × %d\n= %d pixels" % (n, n, n * n),
         fc="#f3f3f1", fs=fs)
    _arrow(ax, 2.4, 3.2, 2.85, 3.2)
    _box(ax, 2.9, 2.6, 3.9, 1.2, u"per pixel: ONE track,\nthe most populated\n(≥ %d events)"
         u"\n→ one resolution value" % TRACK_EVENT_FLOOR, fc="#eaf5ec", ec=GOOD, fs=fs)
    _arrow(ax, 6.85, 3.2, 7.3, 3.2)
    _box(ax, 7.35, 2.6, 2.5, 1.2, "the map: up to %d\nvalues, no pooling\nacross combos" % (n * n),
         fc="#f3f3f1", fs=fs)
    _arrow(ax, 8.6, 2.55, 8.6, 1.9)
    _box(ax, 4.0, 0.55, 5.85, 1.3, u"robust Gaussian over the map's values (%g σ clip)\n"
         u"→ μ ± σ_pixel\nμ = the board number · σ_pixel = "
         u"the error bar" % TRACK_CLIP_SIGMA, fc="#fbf1e6", ec="#b3661e", fs=fs)
    ax.text(0.15, 1.2, "the bar on a result plot is the\npixel-to-pixel spread of that board,\n"
            "not the error on the mean", ha="left", va="center", fontsize=8,
            color=INK_MUTED)


TRACK_PANELS = [("anointed_combo", "Anointed combo per board", _track_combos, 4.0, (8.0, 4.6)),
                ("most_populated_track", "The most-populated track", _track_most_populated, 5.6,
                 (8.0, 4.8)),
                ("one_number", "From 256 pixels to one number", _track_one_number, 4.6,
                 (8.6, 3.9))]


def track_selection(name, tag, line1, line2, footer_text, panels=False):
    """Three panels: the anointed combos, the most-populated track through a pixel, the funnel
    from a board's pixel values to one number. Returns (fig, [(name, panel figure)]), the
    list empty unless `panels`."""
    style.apply_style(1.0)
    W, H = 15.6, 4.9
    ML, MR, MB, MT = 0.4, 0.3, 0.45, 1.05
    fig = plt.figure(figsize=(W, H))
    hax = style.header_axis(fig, ML / W, 1 - (MT - 0.35) / H, (W - ML - MR) / W)
    style.header(hax, name=name, tag=tag, line1=line1, line2=line2, scale=0.8)
    x, gap = ML, 0.35
    for i, (_key, title, draw, w, _size) in enumerate(TRACK_PANELS):
        ax = fig.add_axes([x / W, MB / H, w / W, (H - MB - MT) / H])
        draw(ax)
        ax.set_title("(%s) %s" % ("abc"[i], title), loc="center", fontsize=10, color=INK)
        x += w + gap
    style.footer(fig, footer_text, scale=0.6)
    out = []
    if panels:
        for i, (key, title, draw, _w, (pw, ph)) in enumerate(TRACK_PANELS):
            pfig = plt.figure(figsize=(pw, ph))
            ml, mr, mb, mt = 0.3, 0.25, 0.3, 0.95
            phax = style.header_axis(pfig, ml / pw, 1 - (mt - 0.35) / ph, (pw - ml - mr) / pw)
            style.header(phax, name=name, tag=title[0].lower() + title[1:], line1=line1,
                         line2=line2, scale=0.8)
            draw(pfig.add_axes([ml / pw, mb / ph, (pw - ml - mr) / pw, (ph - mb - mt) / ph]))
            out.append((key, pfig))
    return fig, out


# ======================================================================== RFSel explainer
# A grey ramp, dark to light as R_f falls: the colours of the fluence ladder stay with fluence.
RFSEL_COLOR = {0: "#252525", 1: "#525252", 2: "#7f7f7f", 3: "#b0b0b0"}
# Schematic pulses: peak amplitude and decay time per RFSel (arbitrary units). They are not
# measured and not taken from any source: they only draw the trend, a larger R_f giving more
# preamp gain and a slower trailing edge, with the same leading edge (it is set by the preamp bias
# current, not by R_f).
PULSE_AMP = {0: 1.00, 1: 0.78, 2: 0.58, 3: 0.46}
PULSE_TAU = {0: 3.6, 1: 2.6, 2: 1.7, 3: 1.15}
PULSE_THRESHOLD = 0.30


def _rf_block_diagram(ax):
    """(1) the signal chain, with R_f parallel to C_f in the preamplifier's feedback."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    rf = style.RFSEL_RF_KOHM
    box_style = dict(boxstyle="round,pad=0.06,rounding_size=0.12", linewidth=1.8,
                     edgecolor=INK, facecolor="#f2f2f0")
    pad = 0.06                                        # the box outline lies this far outside

    def box(cx, cy, w, h, text, fs=10.5):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, **box_style))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.35)
        return cx - w / 2, cx + w / 2

    def arrow(x0, x1, y, label=None):
        """From the outline of the box ending at x0 to the outline of the one starting at x1."""
        ax.add_patch(FancyArrowPatch((x0 + pad, y), (x1 - pad, y), arrowstyle="-|>",
                                     mutation_scale=14, linewidth=1.6, color=INK, shrinkA=0,
                                     shrinkB=0))
        if label:
            ax.text((x0 + x1) / 2.0, y + 0.12, label, ha="center", va="bottom", fontsize=10.0,
                    color=INK_MUTED)

    y0, h_box, gap = 2.7, 1.5, 0.5
    widths = dict(sensor=1.3, preamp=2.2, disc=2.3, tdc=1.5)
    x = 0.3
    _, x = box(x + widths["sensor"] / 2, y0, widths["sensor"], h_box, "LGAD\npixel")
    arrow(x, x + 0.8, y0, "Q$_{in}$")
    x += 0.8
    pL, pR = box(x + widths["preamp"] / 2, y0, widths["preamp"], h_box, "Preamplifier\n(TIA)")
    x = pR
    top = y0 + h_box / 2 + pad
    fy = y0 + 1.15
    fx0, fx1 = pL + 0.22, pR - 0.22
    ax.plot([pL + 0.13, pL + 0.13, fx0], [top, fy, fy], color=INK, linewidth=1.3)
    ax.plot([fx1, pR - 0.13, pR - 0.13], [fy, fy, top], color=INK, linewidth=1.3)
    zx = np.linspace(fx0, fx0 + (fx1 - fx0) * 0.42, 7)
    zy = fy + 0.14 * np.array([0, 1, -1, 1, -1, 1, 0])
    ax.plot(zx, zy, color=INK, linewidth=1.5)
    ax.text((zx[0] + zx[-1]) / 2, fy + 0.36, r"R$_\mathrm{f}$", ha="center", fontsize=11.0,
            color=INK)
    ax.add_patch(FancyArrowPatch((zx[0] - 0.05, fy - 0.30), (zx[-1] + 0.05, fy + 0.30),
                                 arrowstyle="-|>", mutation_scale=10, linewidth=1.3, color=ALERT))
    cx0 = fx0 + (fx1 - fx0) * 0.58
    ax.plot([cx0, cx0], [fy - 0.14, fy + 0.14], color=INK, linewidth=1.8)
    ax.plot([cx0 + 0.12, cx0 + 0.12], [fy - 0.14, fy + 0.14], color=INK, linewidth=1.8)
    ax.plot([fx0 + (fx1 - fx0) * 0.42, cx0], [fy, fy], color=INK, linewidth=1.3)
    ax.plot([cx0 + 0.12, fx1], [fy, fy], color=INK, linewidth=1.3)
    ax.text(cx0 + 0.06, fy + 0.36, r"C$_\mathrm{f}$", ha="center", fontsize=11.0, color=INK)
    ax.text((pL + pR) / 2, fy + 0.58, "RFSel %s = %s k$\\Omega$"
            % ("/".join(str(r) for r in sorted(rf)), " / ".join("%g" % rf[r] for r in sorted(rf))),
            ha="center", va="bottom", fontsize=10.5, color=ALERT, weight="bold")
    arrow(x, x + gap, y0)
    x += gap
    _, x = box(x + widths["disc"] / 2, y0, widths["disc"], h_box,
               "Discriminator\nthreshold =\nbaseline + offset", fs=10.0)
    arrow(x, x + gap, y0)
    x += gap
    box(x + widths["tdc"] / 2, y0, widths["tdc"], h_box, "TDC\nTOA, TOT")
    ax.text(5.0, 0.35, "schematic, not to scale", ha="center", va="bottom", fontsize=10.0,
            color=INK_MUTED, style="italic")


def pulse(t, rfsel):
    """The schematic preamp output for one RFSel: a linear rise from t = 1.2 to 2.0, then an
    exponential decay (PULSE_AMP, PULSE_TAU)."""
    a, tau = PULSE_AMP[rfsel], PULSE_TAU[rfsel]
    y = np.zeros_like(t)
    rise = (t >= 1.2) & (t < 2.0)
    y[rise] = a * (t[rise] - 1.2) / 0.8
    fall = t >= 2.0
    y[fall] = a * np.exp(-(t[fall] - 2.0) / tau)
    return y


def _rf_pulses(ax, standard):
    """(2) the schematic pulses of the four RFSel on one axis; TOA and TOT marked on the
    `standard` one."""
    rf = style.RFSEL_RF_KOHM
    t = np.linspace(0, 10, 600)
    for r in sorted(rf):
        std = r == standard
        ax.plot(t, pulse(t, r), color=RFSEL_COLOR[r], linewidth=3.2 if std else 2.0,
                zorder=5 if std else 3,
                label="RFSel %d (%g k$\\Omega$)%s" % (r, rf[r], ", standard" if std else ""))
    ax.axhline(PULSE_THRESHOLD, color=INK_MUTED, linestyle="--", linewidth=1.4, zorder=1)
    ax.text(9.7, PULSE_THRESHOLD + 0.03, "threshold", ha="right", va="bottom", fontsize=10.5,
            color=INK_MUTED)
    y = pulse(t, standard)
    idx = np.where(y >= PULSE_THRESHOLD)[0]
    t_toa, t_end = t[idx[0]], t[idx[-1]]
    for tx in (t_toa, t_end):
        ax.axvline(tx, color=INK, linestyle=":", linewidth=1.2, ymax=0.62)
    ax.annotate("", xy=(t_end, PULSE_THRESHOLD - 0.09), xytext=(t_toa, PULSE_THRESHOLD - 0.09),
                arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.3))
    ax.text((t_toa + t_end) / 2, PULSE_THRESHOLD - 0.16, "TOT", ha="center", va="top",
            fontsize=10.5, color=INK)
    ax.annotate("TOA", xy=(t_toa, PULSE_THRESHOLD), xytext=(0.75, 0.55), ha="center",
                va="center", fontsize=10.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.0, shrinkA=3, shrinkB=2))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("time", fontsize=12.5, color=INK)
    ax.set_ylabel("preamp output (arb.)", fontsize=12.5, color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper right", fontsize=9.5, frameon=False)


def _wrap(text, width=60):
    """Wrap `text` at `width` shown characters, never inside a $...$ math segment, which counts
    as the characters it shows (R$_\\mathrm{f}$ as two)."""
    def shown(s):
        return len(re.sub(r"\\[a-zA-Z]+|[{}_^$]", "", s))
    parts = text.split("$")
    for k in range(1, len(parts), 2):
        parts[k] = parts[k].replace(" ", "\0")
    lines, cur = [], []
    for word in "$".join(parts).split():
        if cur and shown(" ".join(cur + [word])) > width:
            lines.append(" ".join(cur))
            cur = []
        cur.append(word)
    lines.append(" ".join(cur))
    return "\n".join(lines).replace("\0", " ")


def _rf_text(ax, lines, closing):
    """(3) one line per effect: [(head, body)], then `closing` under a rule. The text is wrapped
    here, to the panel's width (matplotlib's own wrapping stops only at the figure's edge)."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    y, dy = 0.97, 0.155
    for head, body in lines:
        ax.text(0.0, y, u"• %s:" % head, ha="left", va="top", fontsize=12.0, color=INK,
                weight="bold", transform=ax.transAxes)
        ax.text(0.0, y - 0.058, _wrap(body), ha="left", va="top", fontsize=10.8, color=INK_MUTED,
                transform=ax.transAxes)
        y -= dy
    ax.axhline(y + 0.02, xmin=0.0, xmax=1.0, color=GRID, linewidth=1.0)
    ax.text(0.0, y - 0.06, _wrap(closing), ha="left", va="top", fontsize=10.8, color=INK,
            style="italic", transform=ax.transAxes)


def rfsel_explainer(name, line1, titles, lines, closing, footer_text, standard=2, panels=False,
                    panel_line2=""):
    """Three panels: where RFSel acts, schematic pulses for the four RFSel, what it changes.
    titles: the three panel titles; lines, closing: the text panel (see _rf_text); panel_line2:
    the header's second line on the single panels, which carry no footer. Returns
    (fig, [(name, panel figure)]), the list empty unless `panels`."""
    style.apply_style(1.0)
    W, H = 16.0, 9.0
    ML, MR, MT, MB = 0.55, 0.35, 1.6, 0.85
    gap = 0.45
    pw_, ph_ = (W - ML - MR - 2 * gap) / 3.0, H - MT - MB
    draws = [_rf_block_diagram, lambda ax: _rf_pulses(ax, standard),
             lambda ax: _rf_text(ax, lines, closing)]
    fig = plt.figure(figsize=(W, H))
    hax = style.header_axis(fig, ML / W, (H - 0.85) / H, (W - ML - MR) / W)
    style.header(hax, name=name, line1=line1, line2="")
    for i, (draw, title) in enumerate(zip(draws, titles)):
        ax = fig.add_axes([(ML + i * (pw_ + gap)) / W, MB / H, pw_ / W, ph_ / H])
        draw(ax)
        ax.set_title(title, fontsize=13.5, color=INK, pad=10, loc="center")
    style.footer(fig, footer_text, scale=1.0)
    out = []
    if panels:
        keys = ["where_rfsel_acts", "same_charge_four_rf", "what_it_changes"]
        for key, draw, title in zip(keys, draws, titles):
            pfig = plt.figure(figsize=(11.0, 6.19))
            ax = pfig.add_axes([0.09, 0.11, 0.86, 0.62])
            draw(ax)
            ax.set_title(title, fontsize=15.0, color=INK, pad=12, loc="center")
            phax = style.header_axis(pfig, 0.09, 1.0 - 0.80 / 6.19, 0.86)
            style.header(phax, name=name, line1=line1, line2=panel_line2, scale=0.9)
            out.append((key, pfig))
    return fig, out
