# -*- coding: utf-8 -*-
"""House-style drawing of the V_gl-vs-fluence fit figures.

The notebook's own V_gl figures (iv07, iv12, iv31-iv33) fix the house treatment: marker and line
style per chip, the "CMS  ETL ETROC IRRAD" header written directly above the frame, a one-line
footer in its own band, and the no-clipping / legend-overlap audits before saving. This module
draws the *fit* view in that treatment, and is shared by H1 and F1.

Two house conventions are deliberately not carried over, because this figure's subject is
different from those figures', and both deviations are stated on the figure itself:

* colour is the BOARD, not the fluence. Fluence is already the x axis here, so the fluence
  colour ladder would be redundant, while the four boards' separate fits are the whole point.
  style.point_style takes `color=` for exactly this reason.
* a logarithmic view is drawn BESIDE the house 0-55 V linear one, not instead of it. Over this
  fluence range the data do not choose between an exponential and a straight line: H1 favours
  the exponential by only ~0.15 V rms while F1 favours a straight line by ~0.35 V rms, and
  chi2/ndf is 25-69 either way, so showing only the log view would assert a model the data do
  not support. The linear view keeps the house range and carries the fit numbers; the log view
  is the shape check, with one range (YLIM_LOG, 7-55 V) for both telescopes, although F1's V_gl
  roughly halves over the campaign where H1's falls fourfold.

The open/filled sense DOES follow the house: open = scan right after the step, filled = after
cooling down. The third look, 4 months later, is half-filled: the house gives it a colour of its
own, but here colour already means the board, and a marker of its own would collide with one
board's marker.
"""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

from .. import style
from . import vgl
from ..campaigns import active as campaign

YLABEL = r"$V_{gl}$ [V]"
# A short label: the full wording (facility bookkeeping 24 GeV/c proton fluence) overflows a
# compound panel. The bookkeeping-vs-on-chip distinction stays in the footer.
XLABEL = r"Bookkeeping fluence [$10^{15}$ p/cm$^2$]"
RESID_YLABEL = "data - fit  [V]"
LOG_TICKS = (8, 10, 12, 15, 20, 25, 30, 40, 50)
AVG_COLOR = style.INK

# style.point_style returns ERRORBAR keywords; Line2D rejects the error-bar-only ones.
_ERRORBAR_ONLY = ("capsize", "elinewidth", "ecolor", "barsabove", "errorevery")


def _line_kw(kw):
    return {k: v for k, v in kw.items() if k not in _ERRORBAR_ONLY}


def fit_boards(table, boards, p0):
    """Per-board and four-board-average fits of the right-after scans, plus the no-last-step variant."""
    out = {"all": {}, "no_last": {}, "log": {}}
    for i, b in enumerate(boards):
        s = vgl.series(table, i, (campaign.PROMPT,))
        out["all"][b] = vgl.fit_exp(*s, p0=p0)
        out["no_last"][b] = vgl.fit_exp(
            *vgl.series(table, i, (campaign.PROMPT,), drop_above=campaign.VGL_DROP_ABOVE), p0=p0)
        out["log"][b] = vgl.fit_exp_log(*s)
    s = vgl.avg_series(table, (campaign.PROMPT,))
    out["all"]["average"] = vgl.fit_exp(*s, p0=p0)
    out["no_last"]["average"] = vgl.fit_exp(
        *vgl.avg_series(table, (campaign.PROMPT,), drop_above=campaign.VGL_DROP_ABOVE), p0=p0)
    out["log"]["average"] = vgl.fit_exp_log(*s)
    return out


def _curve(f):
    return f["V0"], f["c_1e16"] / 10.0


# Fixed linear range for every V_gl panel: H1 and F1 stay directly comparable. The log view has
# its own range (0 has no place on a log axis), but it too is ONE range for both telescopes, for
# direct visual comparison. The comparison figure's range is wider: it must also hold the
# published curves, which reach 3.6 V at the end of the grid (Kraus W36 with its band).
YLIM_LINEAR = (0.0, 55.0)
YLIM_LOG = (7.0, 55.0)
YLIM_LOG_COMPARE = (3.3, 80.0)
LOG_TICKS_COMPARE = (4, 5, 7, 10, 15, 20, 30, 40, 50, 70)
# One residual range for both telescopes, so their scatter compares directly. It is wide enough
# for F1's cooling-down and 4-month points at 1.5e15, which sit +3.9 to +4.4 V above the
# right-after fit; a range sized for H1 alone, (-2.3, 2.75), would hide them.
RESID_YLIM = (-2.6, 5.4)


def look_of(label):
    """The look a V_gl table label belongs to: "right_after" (the campaign's PROMPT),
    "after_cooling_down" (LATE_LABELS) or "four_months" (VERY_LATE); ValueError otherwise."""
    if label == campaign.PROMPT:
        return "right_after"
    if label in campaign.LATE_LABELS:
        return "after_cooling_down"
    if label == campaign.VERY_LATE:
        return "four_months"
    raise ValueError("V_gl table label %r is none of the campaign's looks (PROMPT %r, "
                     "LATE_LABELS %r, VERY_LATE %r); add it to the campaign module"
                     % (label, campaign.PROMPT, campaign.LATE_LABELS, campaign.VERY_LATE))


def draw_fit_panel(ax, axr, table, boards, colors, xoff, fits, scale=0.9, yscale="linear",
                   ylim=None, log_ticks=LOG_TICKS, resid_ylim=RESID_YLIM, stats_box=True,
                   legend_loc="upper right", legend_ncol=1, stats_top=0.68):
    """One telescope's V_gl fit panel (ax) over its residual panel (axr).

    yscale picks the view: "linear" is the house 0-55 V axis, "log" is the straight-line-if-
    exponential view. Both are drawn for the same data because over
    this fluence range the data do not discriminate between an exponential and a straight line
    (for F1 a straight line fits better), so neither view alone is honest.

    `fits` is what fit_boards() returned. Returns the values dict for the sidecar.
    """
    if ylim is None:
        ylim = YLIM_LOG if yscale == "log" else YLIM_LINEAR
    s = style.sizes(scale)
    xgrid = np.linspace(0.0, campaign.VGL_XGRID_MAX, campaign.VGL_XGRID_N)
    used = []

    for i, b in enumerate(boards):
        col, dx = colors[b], xoff[b]
        f = fits["all"][b]
        v0, c = _curve(f)
        ax.plot(xgrid, vgl.model(xgrid, v0, c), color=col, lw=1.2 * scale, alpha=0.75,
                ls=style.chip_linestyle(b), zorder=2)
        points = {"right_after": [], "after_cooling_down": [], "four_months": []}
        for phi, lab, vals in table:
            value = vals[i]
            resid = value - vgl.model(phi, v0, c)
            # house sense: open = right after the step, filled = after cooling down
            look = look_of(lab)
            if look == "right_after":
                kw = style.point_style(b, phi, scale=scale, line=False, open_marker=True, color=col)
            elif look == "after_cooling_down":
                kw = style.point_style(b, phi, scale=scale, line=False, color=col)
            else:
                kw = style.point_style(b, phi, scale=scale, line=False, color=col)
                kw.update(fillstyle="left", markerfacecoloralt=style.SURFACE)   # half-filled
            points[look].append((phi, value))
            ax.plot([phi + dx], [value], **_line_kw(kw))
            axr.plot([phi + dx], [resid], **_line_kw(kw))
        used.append(dict(chip=b, V0=f["V0"], V0_err=f["V0_err"], c_1e16=f["c_1e16"],
                         c_1e16_err=f["c_1e16_err"], **points))

    fa, fa0 = fits["all"]["average"], fits["no_last"]["average"]
    ax.plot(xgrid, vgl.model(xgrid, *_curve(fa)), color=AVG_COLOR, lw=2.4 * scale, zorder=3)
    ax.plot(xgrid, vgl.model(xgrid, *_curve(fa0)), color=AVG_COLOR, lw=2.0 * scale, ls="--",
            dashes=(5, 3), zorder=3)

    ax.set_yscale(yscale)
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.13, campaign.VGL_XGRID_MAX)
    if yscale == "log":
        vgl.set_log_ticks(ax, list(log_ticks))
    ax.set_ylabel(YLABEL)
    ax.tick_params(labelbottom=False)
    style.style_axes(ax)

    axr.axhline(0.0, color="0.4", lw=0.9)
    axr.set_ylim(*resid_ylim)
    axr.yaxis.set_major_locator(MultipleLocator(2.0))
    axr.set_ylabel(RESID_YLABEL)
    axr.set_xlabel(XLABEL)
    axr.set_xlim(*ax.get_xlim())
    style.style_axes(axr)
    axr.yaxis.label.set_fontsize(s["legend"] * 0.8)   # after style_axes, which resets it

    board_handles = [Line2D([], [], color=colors[b], marker=style.chip_marker(b),
                            ls=style.chip_linestyle(b), ms=8.0 * scale, lw=1.4 * scale, label=b)
                     for b in boards]
    leg1 = ax.legend(handles=board_handles, loc="lower left", fontsize=s["legend"] * 0.85,
                     title="board", title_fontsize=s["legend"] * 0.85, ncol=2)
    ax.add_artist(leg1)

    look_handles = [
        Line2D([], [], color="0.25", marker="o", ms=8.0 * scale, mfc=style.SURFACE, mew=1.8 * scale,
               ls="none", label="right after (fitted)"),
        Line2D([], [], color="0.25", marker="o", ms=8.0 * scale, ls="none",
               label="after cooling down"),
        Line2D([], [], color="0.25", marker="o", ms=8.0 * scale, ls="none", fillstyle="left",
               markerfacecoloralt=style.SURFACE, mew=1.8 * scale, label="4 months later"),
        Line2D([], [], color=AVG_COLOR, lw=2.4 * scale, label="average fit"),
        Line2D([], [], color=AVG_COLOR, lw=2.0 * scale, ls="--", dashes=(5, 3),
               label="average, last step dropped"),
    ]
    ax.legend(handles=look_handles, loc=legend_loc, ncol=legend_ncol, fontsize=s["legend"] * 0.85)

    if stats_box:
        rows = ["board      V0 [V]        c [1e-16 cm2/p]"]
        for b in list(boards) + ["average"]:
            f = fits["all"][b]
            rows.append("%-9s %5.2f +- %4.2f    %5.2f +- %4.2f"
                        % (b, f["V0"], f["V0_err"], f["c_1e16"], f["c_1e16_err"]))
        rows.append("%-9s %5.2f +- %4.2f    %5.2f +- %4.2f"
                    % ("no last", fa0["V0"], fa0["V0_err"], fa0["c_1e16"], fa0["c_1e16_err"]))
        ax.text(0.985, stats_top, "\n".join(rows), transform=ax.transAxes, ha="right", va="top",
                ma="left", fontsize=s["legend"] * 0.66, family="monospace", linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.5", fc=style.SURFACE, ec="0.7", lw=0.8))

    return dict(boards=used,
                average=dict(V0=fa["V0"], V0_err=fa["V0_err"], c_1e16=fa["c_1e16"],
                             c_1e16_err=fa["c_1e16_err"]),
                average_without_last_step=dict(V0=fa0["V0"], c_1e16=fa0["c_1e16"]),
                average_log_weighted_c_1e16=fits["log"]["average"]["c_1e16"])


FOOTER = ("gain-layer depletion voltage, from the k-factor peak of the fine low-V scan; "
          "open = right after the step, filled = after cooling down, 2 to 4 days, "
          "half-filled = 4 months later\n"
          "colour and marker = board; facility bookkeeping fluence; points offset in fluence "
          "for legibility")


# ============================================================================ comparison figures
# The published constants are drawn through each telescope's own fitted V0, so only the slope is
# compared. Below each panel a slope strip restates the same comparison as numbers; it is colour-
# matched to the curves, so it doubles as their legend and carries the conversion arithmetic.
REF_XMAX = 6.6          # one strip range for both telescopes: holds every reference and its band
AVG_OFFSETS = {"right_after": 0.0, "after_cooling_down": 0.05, "four_months": -0.05}
H1F1_COLORS = {"h1": "#B2182B", "f1": "#2166AC"}     # H1 red, F1 blue (iv36, iv39)
H1F1_MARKERS = {"h1": "o", "f1": "s"}


def avg_points(table):
    """Board-average V_gl per look: {look: (fluence, V_gl)}."""
    return {"right_after": vgl.avg_series(table, (campaign.PROMPT,)),
            "after_cooling_down": vgl.avg_series(table, campaign.LATE_LABELS),
            "four_months": vgl.avg_series(table, (campaign.VERY_LATE,))}


def _look_kw(look, color, scale):
    kw = dict(marker="o", ms=8.0 * scale, mec=color, mew=1.8 * scale, ls="none")
    if look == "right_after":
        kw.update(mfc=style.SURFACE)
    elif look == "after_cooling_down":
        kw.update(mfc=color)
    else:
        kw.update(mfc=color, fillstyle="left", markerfacecoloralt=style.SURFACE)
    return kw


def draw_compare_panel(ax, table, fits, refs, scale=0.9, yscale="linear", ylim=None,
                       log_ticks=LOG_TICKS_COMPARE, legend_loc="upper right", line_keys=(),
                       key_loc="lower left"):
    """Board-average points, the average fit and every published curve through the same V0.

    `line_keys` is ((linestyle, meaning), ...), a second legend at `key_loc`: colour identifies
    the reference in the slope strip, the line style says which beam. Returns the values dict for
    the sidecar.
    """
    s = style.sizes(scale)
    fa = fits["all"]["average"]
    v0, c_lin, c_log = fa["V0"], fa["c_1e16"], fits["log"]["average"]["c_1e16"]
    xg = np.linspace(0.0, campaign.VGL_XGRID_MAX, campaign.VGL_XGRID_N)

    def through_v0(c16):
        return v0 * np.exp(-(c16 / 10.0) * xg)

    ax.fill_between(xg, through_v0(min(c_lin, c_log)), through_v0(max(c_lin, c_log)),
                    color="0.55", alpha=0.30, lw=0, zorder=1)
    for r in refs:
        if r["lo"] is not None:
            ax.fill_between(xg, through_v0(r["hi"]), through_v0(r["lo"]), color=r["color"],
                            alpha=0.16, lw=0, zorder=2)
        ax.plot(xg, through_v0(r["c"]), color=r["color"], ls=":" if r["alt"] else r["ls"],
                lw=(1.3 if r["alt"] else 1.7) * scale, zorder=4)
    ax.plot(xg, through_v0(c_lin), color=AVG_COLOR, lw=2.4 * scale, zorder=6)

    points = {}
    for look, (phi, v) in avg_points(table).items():
        ax.plot(phi + AVG_OFFSETS[look], v, zorder=8, **_look_kw(look, AVG_COLOR, scale))
        points[look] = [[float(a), float(b)] for a, b in zip(phi, v)]

    ax.set_yscale(yscale)
    ax.set_ylim(*(ylim or (YLIM_LOG_COMPARE if yscale == "log" else YLIM_LINEAR)))
    ax.set_xlim(-0.13, campaign.VGL_XGRID_MAX)
    if yscale == "log":
        vgl.set_log_ticks(ax, list(log_ticks))
    ax.set_ylabel(YLABEL)
    ax.set_xlabel(XLABEL)
    style.style_axes(ax)

    handles = [Line2D([], [], label=lab, **_look_kw(look, "0.25", scale)) for look, lab in
               (("right_after", "right after (fitted)"),
                ("after_cooling_down", "after cooling down"),
                ("four_months", "4 months later"))]
    handles += [Line2D([], [], color=AVG_COLOR, lw=2.4 * scale, label="average fit"),
                Patch(facecolor="0.55", alpha=0.30, edgecolor="none",
                      label="fit weighting, linear to log")]
    keys = [Line2D([], [], color="0.35", ls=ls, lw=1.7 * scale, label=lab)
            for ls, lab in line_keys]
    if keys:
        ax.add_artist(ax.legend(handles=keys, loc=key_loc, fontsize=s["legend"] * 0.78))
    ax.legend(handles=handles, loc=legend_loc, ncol=2, fontsize=s["legend"] * 0.78)

    return dict(V0=v0, c_1e16=c_lin, c_1e16_err=fa["c_1e16_err"], c_1e16_log_weighting=c_log,
                chi2_ndf=fa["chi2_ndf_sigma0p2"], points=points,
                references=[{k: r[k] for k in ("name", "arith", "lo", "c", "hi", "values", "alt")}
                            for r in refs])


def draw_slope_strip(axc, fits, refs, who, scale=0.9, xmax=REF_XMAX):
    """The comparison as slopes: our c with its fit error, each published value with its band."""
    s = style.sizes(scale)
    style.style_axes(axc)
    fa = fits["all"]["average"]
    c, ce = fa["c_1e16"], fa["c_1e16_err"]
    axc.axvspan(c - ce, c + ce, color="0.55", alpha=0.30, lw=0, zorder=1)
    axc.axvline(c, color=AVG_COLOR, lw=1.1 * scale, zorder=2)
    rows = [("%s this work\nfour-board average" % who, c - ce, c, c + ce, AVG_COLOR, False,
             "%.2f +- %.2f" % (c, ce))]
    for r in refs:
        val = "%.2f" % r["c"] if r["lo"] is None else "%.2f to %.2f" % (r["lo"], r["hi"])
        rows.append((r["name"] + "\n" + r["arith"], r["lo"], r["c"], r["hi"], r["color"],
                     r["alt"], val))
    n = len(rows)
    for k, (_, lo, cen, hi, col, alt, val) in enumerate(rows):
        yy = n - 1 - k
        if lo is not None:
            axc.plot([lo, hi], [yy, yy], color=col, lw=5.0 * scale, alpha=0.32,
                     solid_capstyle="butt", zorder=3)
        axc.plot([cen], [yy], marker="o", ms=7.0 * scale, mfc=style.SURFACE if alt else col,
                 mec=col, mew=1.5 * scale, ls="none", zorder=5)
        axc.text(1.02, yy, val, transform=axc.get_yaxis_transform(), ha="left", va="center",
                 fontsize=s["legend"] * 0.7)
    axc.set_ylim(-0.65, n - 0.35)
    axc.set_xlim(0.0, xmax)
    axc.set_yticks(np.arange(n))
    axc.set_yticklabels([r[0] for r in rows][::-1], fontsize=s["legend"] * 0.6,
                        linespacing=1.2)
    for tl, r in zip(axc.get_yticklabels(), rows[::-1]):
        tl.set_color(r[4])
    axc.tick_params(axis="y", length=0)
    axc.grid(False, axis="y")
    axc.set_xlabel(r"$c$  [$10^{-16}$ cm$^2$ per bookkeeping proton]",
                   fontsize=s["legend"] * 0.85)
    f = campaign.CHIP_FLUENCE_FACTOR
    top = axc.secondary_xaxis("top", functions=(lambda v: v / f, lambda v: v * f))
    top.set_xlabel(r"$c$ per proton that crossed the chip  [$10^{-16}$ cm$^2$]",
                   fontsize=s["legend"] * 0.85)
    top.tick_params(labelsize=s["tick"])


def draw_h1_vs_f1(axn, axa, sets, scale=0.9, yscale="linear"):
    """The two telescopes' board-average right-after points and fits: V_gl / V0 (axn), V (axa).

    `sets` is [dict(tel, label, phi, v, V0, c_1e16, c_1e16_err), ...], H1 first.
    """
    s = style.sizes(scale)
    xg = np.linspace(0.0, campaign.VGL_XGRID_MAX, campaign.VGL_XGRID_N)
    handles = []
    for st in sets:
        col, mk = H1F1_COLORS[st["tel"]], H1F1_MARKERS[st["tel"]]
        pk = dict(marker=mk, ms=8.5 * scale, mfc=style.SURFACE, mec=col, mew=1.9 * scale,
                  ls="none", zorder=5)
        axn.plot(xg, np.exp(-(st["c_1e16"] / 10.0) * xg), color=col, lw=2.2 * scale, zorder=3)
        axn.plot(st["phi"], st["v"] / st["V0"], **pk)
        axa.plot(xg, vgl.model(xg, st["V0"], st["c_1e16"] / 10.0), color=col, lw=2.2 * scale,
                 zorder=3)
        axa.plot(st["phi"], st["v"], **pk)
        handles.append(Line2D([], [], color=col, lw=2.2 * scale, marker=mk, ms=pk["ms"],
                              mfc=style.SURFACE, mec=col, mew=pk["mew"],
                              label="%s:  V0 = %.1f V,  c = %.2f +- %.2f"
                              % (st["label"], st["V0"], st["c_1e16"], st["c_1e16_err"])))
    if yscale == "log":
        axn.set_yscale("log")
        axn.set_ylim(0.19, 1.22)
        vgl.set_log_ticks(axn, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                          ["0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])
        axa.set_yscale("log")
        axa.set_ylim(6.8, 58.0)
        vgl.set_log_ticks(axa, [7, 9, 12, 16, 22, 30, 40, 50])
    else:
        axn.set_ylim(0.0, 1.1)
        axa.set_ylim(*YLIM_LINEAR)
    for ax in (axn, axa):
        ax.set_xlim(-0.13, campaign.VGL_XGRID_MAX)
        ax.set_xlabel(XLABEL)
        style.style_axes(ax)
    axn.set_ylabel(r"$V_{gl}\,/\,V_0$  (own fitted $V_0$)")
    axa.set_ylabel(YLABEL)
    axn.legend(handles=handles, loc="lower left", fontsize=s["legend"] * 0.8,
               title=r"c in $10^{-16}$ cm$^2$ per bookkeeping proton",
               title_fontsize=s["legend"] * 0.8)
    h, f = sets
    ratio = h["c_1e16"] / f["c_1e16"]
    ratio_err = ratio * np.hypot(h["c_1e16_err"] / h["c_1e16"], f["c_1e16_err"] / f["c_1e16"])
    axn.text(0.975, 0.965, "c(H1) / c(F1) = %.2f +- %.2f" % (ratio, ratio_err),
             transform=axn.transAxes, ha="right", va="top", fontsize=s["legend"] * 0.85,
             bbox=dict(boxstyle="round,pad=0.5", fc=style.SURFACE, ec="0.7", lw=0.8))
    return dict(c_ratio_h1_over_f1=ratio, c_ratio_h1_over_f1_err=float(ratio_err),
                fraction_of_V_gl_left_at_3p5e15={st["tel"]: float(st["v"][-1] / st["v"][0])
                                                 for st in sets},
                telescopes={st["tel"]: dict(V0=st["V0"], c_1e16=st["c_1e16"],
                                            c_1e16_err=st["c_1e16_err"]) for st in sets})
