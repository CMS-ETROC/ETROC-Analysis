"""Shared style for the 2026 ETL group talk figures (TestBeam/talk_2026/).

One module, four sessions: every talk figure imports this and nothing else for its look.  It
re-exports the walkthrough house style (TestBeam/analysis_walkthrough/etroc_style.py: mplhep CMS
style, point sizes, the overlap audit, the PANELS registry and single-panel export) and adds the
conventions fixed for the talk in notes/talk-figures-2026.md section 3:

* colour = FLUENCE, one six-step ladder shared by March and July (facility bookkeeping in p/cm2,
  24 GeV protons; no n_eq anywhere);
* marker + line style = CHIP; legends name chips only, the vendor sits in the panel title;
* merged runs replace single runs and are drawn with a thick black marker edge (v1 review aid);
* error bars on board points = pixel-to-pixel sigma of the robust fit (stat error goes to JSON);
* header: "CMS  ETL ETROC IRRAD" alone on the left, immediately above the axes; every other
  line (facility lines, telescope/panel title, data line) on the right (user rule 2026-09-18);
  60 deg in every header
  that shows resolution or TDC data;
* axis labels and the 20-80 ps window for standard-point bias scans.

Style changes go through session e7 only.
"""
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))

import mplhep as hep                                   # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402
from . import etroc_style as es                               # noqa: E402
from .etroc_style import (                              # noqa: E402,F401  (re-exports)
    apply_style, sizes, style_axes, data_line, check_no_clipping,
    add_panel_arg, resolve_panels, panel_subject, save_panel, header_axis,
    INK, INK_MUTED, GRID, SURFACE, ALERT,
)

audit_title_fit = check_no_clipping     # the name the team file uses for the same audit

# ------------------------------------------------------------------ identity of the campaign
EXP_TEXT = "ETL ETROC IRRAD"      # every talk figure is an IRRAD test; hep.cms.text adds "CMS".
                                  # Rule (user, 2026-09-18): this is the ONLY text on the header left,
                                  # written immediately above the axes; all other lines go right.
CAMPAIGN = "CERN IRRAD 2026"
HEADER_LINE_1 = "CERN IRRAD 2026, 24 GeV p+, 60 deg"      # beam line (IV scans override it)
HEADER_LINE_2 = "17 cm off axis, T = -25 C"               # position / temperature

TELESCOPE_TITLE = {"h1": "H1 telescope, HPK", "f1": "F1 telescope, FBK"}   # short: it shares the
# bottom header line with tag and data, next to the experiment text (about 40 characters fit)
TELESCOPE_CHIPS = {"h1": ["IH7", "IH11", "IH12", "IH13"], "f1": ["LF10", "LF12", "LF17", "LF18"]}

# ------------------------------------------------------------------ colour = fluence
# Ladder in p/cm2 (facility bookkeeping).  Colours: black, then five hues that stay apart for
# deuteranopes on a white ground (green/blue/red/purple/brown, no two at the same lightness).
FLUENCE_STEPS = (0.0, 3e14, 9e14, 1.5e15, 2e15, 3.5e15)
FLUENCE_COLOR = {
    0.0:    "#000000",   # pre-irradiation
    3e14:   "#1a9641",   # green
    9e14:   "#2b6fd6",   # blue
    1.5e15: "#d62728",   # red
    2e15:   "#7b3fb5",   # purple
    3.5e15: "#ff7f0e",   # orange (was brown #8c5a2b until 2026-09-19)
}
_FLUENCE_TEXT = {
    0.0: "pre-irradiation", 3e14: r"$3\times10^{14}$", 9e14: r"$9\times10^{14}$",
    1.5e15: r"$1.5\times10^{15}$", 2e15: r"$2\times10^{15}$", 3.5e15: r"$3.5\times10^{15}$",
}
FLUENCE_UNIT = r"p/cm$^2$"


def fluence_key(f, rtol=0.2):
    """Snap a bookkeeping fluence (1.5e15, 15e14, '1.5e15') to a step of FLUENCE_STEPS.

    None, nan and zero (or below) are pre-irradiation, 0.0. A value within `rtol` (relative) of
    two steps goes to the nearer one, nearness measured the same way, |f - step| / step. Raises
    ValueError on anything that is not a number, and on a number within `rtol` of no step, so a
    new irradiation step is added to the ladder deliberately rather than drawn in a default colour.
    """
    if f is None:
        return 0.0
    try:
        x = float(f)
    except (TypeError, ValueError):
        raise ValueError("fluence %r is not a number" % (f,))
    if x != x or x <= 0:                # nan or zero -> pre-irradiation
        return 0.0
    near = [step for step in FLUENCE_STEPS[1:] if abs(x - step) <= rtol * step]
    if not near:
        raise ValueError("fluence %r is not on the fluence ladder %s" % (f, FLUENCE_STEPS))
    return min(near, key=lambda step: abs(x - step) / step)


def fluence_color(f):
    return FLUENCE_COLOR[fluence_key(f)]


def fluence_label(f, unit=True):
    """Legend text for a ladder step: 'pre-irradiation' or '3e14 p/cm2' in mathtext."""
    k = fluence_key(f)
    if k == 0.0 or not unit:
        return _FLUENCE_TEXT[k]
    return "%s %s" % (_FLUENCE_TEXT[k], FLUENCE_UNIT)


# ------------------------------------------------------------------ marker + line = chip
# Marker shape follows the chip's slot in its telescope (same shape set the walkthrough uses for
# board index), line style repeats the same identity so nothing rides on the marker alone.
_SLOT_MARKER = ("o", "s", "^", "D")
_SLOT_LINESTYLE = ("-", "--", "-.", ":")
CHIP_MARKER = {}
CHIP_LINESTYLE = {}
for _tel, _chips in TELESCOPE_CHIPS.items():
    for _i, _c in enumerate(_chips):
        CHIP_MARKER[_c] = _SLOT_MARKER[_i]
        CHIP_LINESTYLE[_c] = _SLOT_LINESTYLE[_i]


def chip_short(name):
    """'PT_IH12' / 'LF10_FBK' / 'IH12' -> 'IH12' / 'LF10': the chip name as legends print it."""
    s = str(name)
    for tok in s.replace("-", "_").split("_"):
        if tok[:2] in ("IH", "LF") and tok[2:].isdigit():
            return tok
    return s


def chip_marker(chip):
    return CHIP_MARKER[chip_short(chip)]


def chip_linestyle(chip):
    return CHIP_LINESTYLE[chip_short(chip)]


# merged runs replace singles on every result plot; v1 marks them with a thick black edge
MERGED_EDGE = dict(markeredgecolor="#000000")
MERGED_EDGE_WIDTH = 2.2          # times scale, applied in point_style


def point_style(chip, fluence, merged=False, scale=1.0, line=True, open_marker=False, color=None):
    """Keyword arguments for ax.errorbar / ax.plot: one board series at one fluence step.

    colour from the fluence ladder (or `color`, for the one-fluence offset ramp of fig 30), marker
    and line style from the chip, thick black marker edge when the point comes from a merged run.
    `open_marker=True` draws a hollow marker: the convention where March and July share an axis
    (March filled, July open).  `line=False` draws markers only (fluence axes).
    """
    c = color or fluence_color(fluence)
    # a merged point is always filled in the fluence colour so the thick black edge stays readable
    open_marker = open_marker and not merged
    kw = dict(color=c, marker=chip_marker(chip), markersize=9.0 * scale,
              markerfacecolor=SURFACE if open_marker else c, markeredgecolor=c,
              markeredgewidth=(1.8 if open_marker else 1.0) * scale,
              linestyle=chip_linestyle(chip) if line else "none", linewidth=2.0 * scale,
              capsize=3.0 * scale, elinewidth=1.4 * scale)
    if merged:
        kw.update(MERGED_EDGE)
        kw["markeredgewidth"] = MERGED_EDGE_WIDTH * scale
    return kw


# ------------------------------------------------------------------ header, labels, ranges
XLABEL_BIAS = "Bias voltage [V]"
YLABEL_RES = "Time resolution [ps]"
XLABEL_FLUENCE = "Fluence [%s]" % FLUENCE_UNIT
YRANGE_STANDARD = (20.0, 80.0)      # standard-point bias scans; off-standard scans choose their own


def talk_header(ax, name=None, tag=None, data=None, scale=1.0, pad=None, line1=None, line2=None):
    """The talk header (user rule 2026-09-18): "CMS  ETL ETROC IRRAD" alone on the left, written
    immediately above the axes; on the right AT MOST TWO lines:

        top     facility conditions: (line1 or HEADER_LINE_1) + ", " + (line2 or HEADER_LINE_2)
        bottom  subject: name, tag and data joined by ", " (omitted when all three are empty)

    name   telescope / panel name ("H1 telescope, HPK sensors") or, on a compound's header axis,
           the figure name
    tag    conditions, e.g. "RFSel 2" or "RFSel 2, threshold offset 20"
    data   the data drawn, e.g. "March 2026" or a data_line() string
    line1  replaces the beam line (IV scans: "CERN IRRAD 2026, parked, no beam")
    line2  replaces the position/temperature line (DESY-only or mixed figures)

    The bottom line shares its height with the experiment text on the left, so it has to be
    short (about 44 characters at the single-panel title size); the overlap audit flags a clash.
    """
    s = sizes(scale)
    hep.cms.text(loc=0, ax=ax, text=EXP_TEXT, fontsize=s["header"])
    facility = ", ".join(x for x in ((line1 or HEADER_LINE_1),
                                     (HEADER_LINE_2 if line2 is None else line2)) if x)
    subject = ", ".join(str(x) for x in (name, tag, data) if x)
    right_lines(ax, [facility] + ([subject] if subject else []), s["title"], pad=pad)
    return s


def right_lines(ax, lines, size, pad=None):
    """Write `lines` on the header RIGHT, merged with the lines the axis already carries
    (talk_header and panel_title may both write to one axis, in either order) and capped at
    TWO lines: the facility line (the one naming IRRAD or DESY) on top, everything else joined
    into one bottom line with ", ".  The bottom line is the axis's right title, so it sits level
    with the experiment text; the top line is a separate text artist above it, so the overlap
    audit measures each line on its own.  The header left never carries anything but the
    experiment text (user rule, 2026-09-18)."""
    import matplotlib as _mpl
    prior = list(getattr(ax, "_talk_right_lines", []))
    merged = prior + [l for l in lines if l and l not in prior]
    fac = [l for l in merged if "IRRAD" in l or "DESY" in l]
    rest = [l for l in merged if l not in fac]
    merged = ([", ".join(fac)] if fac else []) + ([", ".join(rest)] if rest else [])
    ax._talk_right_lines = merged
    top_pad = float(_mpl.rcParams["axes.titlepad"] if pad is None else pad)
    kw = {} if pad is None else {"pad": pad}
    ax.set_title("", loc="left")
    ax.set_title(merged[-1] if merged else "", loc="right", size=size, color=INK, **kw)
    old = getattr(ax, "_talk_right_top", None)
    if old is not None:
        old.remove()
        ax._talk_right_top = None
    if len(merged) == 2:
        ax._talk_right_top = ax.annotate(
            merged[0], xy=(1.0, 1.0), xycoords="axes fraction",
            xytext=(0.0, top_pad + 1.3 * size), textcoords="offset points",
            ha="right", va="bottom", fontsize=size, color=INK, annotation_clip=False)

def compound_header(axes, tag=None, data=None, line1=None, line2=None, scale=1.0, subjects=None,
                    pad=None):
    """Header for a COMPOUND figure WITHOUT a header axis (user rule 2026-09-18: the experiment
    text sits immediately above the axes, so nothing may sit between them).  `axes` = the top
    row of panels, left to right.  The experiment text goes on the first panel; the facility
    line (line1/line2 defaults, then tag and data appended: the compound top line is wide) goes
    on the top right of the LAST panel and may extend leftwards over the other panels' header
    space; every panel gets its own subject on its bottom right: subjects[i], or whatever
    panel_title() already wrote there.  Give EVERY panel a subject, so the facility line is
    always the top line and never drops level with the experiment text.  Returns sizes(scale).
    """
    s = sizes(scale)
    axes = list(axes)
    hep.cms.text(loc=0, ax=axes[0], text=EXP_TEXT, fontsize=s["header"])
    facility = ", ".join(str(x) for x in ((line1 or HEADER_LINE_1),
                                          (HEADER_LINE_2 if line2 is None else line2), tag, data) if x)
    for i, ax in enumerate(axes):
        subj = subjects[i] if subjects and i < len(subjects) and subjects[i] else None
        lines = ([facility] if i == len(axes) - 1 else []) + ([subj] if subj else [])
        if lines:
            right_lines(ax, lines, s["title"], pad=pad)
    return s


def panel_title(ax, telescope, extra=None, scale=1.0):
    """Panel title 'H1 telescope, HPK sensors' (+ ' · extra'), written on the header RIGHT
    (below any facility lines already there).  Nothing is ever written on the left below the
    experiment text, so "CMS ETL ETROC IRRAD" stays immediately above the axes."""
    s = sizes(scale)
    t = TELESCOPE_TITLE[telescope.lower()]
    if extra:
        t = u"%s  ·  %s" % (t, extra)
    right_lines(ax, [t], s["title"])
    return s


# ------------------------------------------------------------------ legends
def fluence_handles(steps, scale=1.0):
    """Legend handles for the fluence ladder: a coloured line per step present (no marker)."""
    out = []
    for f in sorted({fluence_key(x) for x in steps}):
        out.append(Line2D([], [], color=FLUENCE_COLOR[f], linewidth=3.0 * scale,
                          linestyle="-", label=fluence_label(f)))
    return out


def chip_handles(chips, scale=1.0, color=INK):
    """Legend handles for chips: black marker + line style per chip, chip name only."""
    out = []
    for c in chips:
        c = chip_short(c)
        out.append(Line2D([], [], color=color, marker=CHIP_MARKER[c], markersize=9.0 * scale,
                          linestyle=CHIP_LINESTYLE[c], linewidth=2.0 * scale, label=c))
    return out


# threshold-offset ramp for a figure that shows ONE fluence at several offsets (fig 30): the
# fluence colour for the standard offset 20, lighter browns for the off-standard offsets.
OFFSET_COLOR = {20: FLUENCE_COLOR[3.5e15], 15: "#ff9a3c", 10: "#ffb266", 8: "#ffc98f", 6: "#ffe0b8"}


def offset_color(offset):
    return OFFSET_COLOR[int(round(float(offset)))]


def offset_handles(offsets, scale=1.0):
    out = []
    for o in sorted({int(round(float(x))) for x in offsets}, reverse=True):
        out.append(Line2D([], [], color=OFFSET_COLOR[o], linewidth=3.0 * scale, linestyle="-",
                          label="offset %d%s" % (o, "  (standard)" if o == 20 else "")))
    return out


def campaign_handles(scale=1.0, color=INK):
    """Filled marker = March 2026, open marker = July 2026 (used where both share an axis)."""
    return [Line2D([], [], color=color, marker="o", markersize=9.0 * scale, linestyle="none",
                   markerfacecolor=color, label="March 2026"),
            Line2D([], [], color=color, marker="o", markersize=9.0 * scale, linestyle="none",
                   markerfacecolor=SURFACE, markeredgewidth=1.8 * scale, label="July 2026")]


def merged_handle(scale=1.0):
    return Line2D([], [], color=INK_MUTED, marker="o", markersize=9.0 * scale,
                  markerfacecolor=SURFACE, markeredgecolor="#000000",
                  markeredgewidth=MERGED_EDGE_WIDTH * scale, linestyle="none",
                  label="merged runs (July, filled)")


def two_legends(ax, steps, chips, scale=1.0, merged=False, loc_f="upper left",
                loc_c="upper right"):
    """Fluence legend and chip legend on one axis (matplotlib keeps only the last ax.legend,
    so the first is re-added as an artist)."""
    s = sizes(scale)
    lf = ax.legend(handles=fluence_handles(steps, scale), loc=loc_f, fontsize=s["legend"],
                   title="fluence", title_fontsize=s["legend_title"])
    ax.add_artist(lf)
    hc = chip_handles(chips, scale) + ([merged_handle(scale)] if merged else [])
    lc = ax.legend(handles=hc, loc=loc_c, fontsize=s["legend"])
    return lf, lc


# ------------------------------------------------------------------ footer, outputs, provenance
FOOTER_GAP_IN = 0.5   # blank band added between the axes and the footer (user request 2026-09-18)


def footer(fig, text, scale=1.0, x=0.01, y=0.008):
    """Selection / veto / counts line at the bottom of a COMPOUND figure (panels get none).
    save_figure() later pushes it down into its own band (see lower_footer) so the footer can be
    cropped away on a slide without touching the axes."""
    t = fig.text(x, y, text, ha="left", va="bottom", fontsize=sizes(scale)["ann"],
                 color=INK_MUTED)
    t._talk_footer = True
    return t


def lower_footer(fig, gap=None):
    """Grow the figure by `gap` inches at the bottom and move every axes and figure-level text
    (except the footer) up by that amount, so the plot is pixel-identical and the footer sits
    alone in the new band.  Idempotent; no-op when the figure has no footer() text."""
    feet = [t for t in fig.texts if getattr(t, "_talk_footer", False)]
    if not feet or getattr(fig, "_talk_footer_lowered", False):
        return False
    fig._talk_footer_lowered = True
    gap = FOOTER_GAP_IN if gap is None else gap
    W, H = fig.get_size_inches()
    H2 = H + gap
    k, d = H / H2, gap / H2
    inv = fig.transFigure.inverted()
    legends = [(lg, lg.get_bbox_to_anchor().transformed(inv)) for lg in fig.legends]
    if hasattr(fig, "set_layout_engine"):
        fig.set_layout_engine("none")
    fig.set_size_inches(W, H2, forward=False)
    for ax in fig.axes:
        if ax.get_axes_locator() is not None:
            continue  # placed relative to a parent axes; moves with it
        b = ax.get_position(original=True)
        ax.set_position([b.x0, b.y0 * k + d, b.width, b.height * k], which="both")
    for t in fig.texts:
        if t in feet:
            continue
        x, y = t.get_position()
        t.set_position((x, y * k + d))
    for name in ("_suptitle", "_supxlabel", "_supylabel"):
        t = getattr(fig, name, None)
        if t is not None:
            x, y = t.get_position()
            t.set_position((x, y * k + d))
    for lg, bf in legends:
        lg.set_bbox_to_anchor((bf.x0, bf.y0 * k + d, bf.width, bf.height * k),
                              transform=fig.transFigure)
    return True


def git_hash(short=True):
    """The repository's HEAD commit, with "-dirty" appended when tracked files have uncommitted
    changes and "-unknown" when that could not be checked; "" when git or the repository is not
    available."""
    try:
        # the revision is required: `git rev-parse --short` without one exits 128
        cmd = ["git", "-C", _HERE, "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
        out = subprocess.run(cmd,
                             capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return ""
        commit = out.stdout.strip()
    except Exception:
        return ""
    try:
        # exit status 1 = tracked files differ from HEAD (untracked files do not count)
        diff = subprocess.run(["git", "-C", _HERE, "diff", "--quiet", "HEAD", "--"],
                              capture_output=True, timeout=60)
    except Exception:
        return commit + "-unknown"
    return commit + {0: "", 1: "-dirty"}.get(diff.returncode, "-unknown")


def save_figure(fig, out_dir, stem, dpi=200, audit=True):
    """<out_dir>/<stem>.png + .pdf; runs the overlap audit first. Returns (paths, problems)."""
    os.makedirs(out_dir, exist_ok=True)
    lower_footer(fig)
    problems = check_no_clipping(fig, stem) if audit else []
    paths = []
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, "%s.%s" % (stem, ext))
        fig.savefig(p, dpi=dpi if ext == "png" else None, facecolor=SURFACE)
        paths.append(p)
    return paths, problems


# ---- track selection declared in every values.json (user decision 2026-09-22) -------------------
# Most-populated is the default: it is the selection the simplified result scheme uses (KB, switch
# made 2026-09-16). Figures that deliberately compare against central tracks pass SELECTION_CENTRAL
# explicitly, so the sidecar states what that figure actually did instead of inheriting a constant.
SELECTION_TOP = ("most-populated track per pixel, anointed combo (0-1-2 for boards 0-2, "
                 "1-2-3 for board 3), one track per pixel per board, event floor 300")
SELECTION_CENTRAL = ("central tracks, anointed combo (0-1-2 for boards 0-2, "
                     "1-2-3 for board 3), one track per pixel per board, event floor 300")


def write_values(out_dir, stem, payload, script=None, inputs=None,
                 selection=SELECTION_TOP):
    """<out_dir>/<stem>_values.json with provenance (script, commit, inputs, time)."""
    os.makedirs(out_dir, exist_ok=True)
    doc = {"figure": stem, "script": script, "commit": git_hash(),
           "inputs": inputs or [], "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "conventions": {"selection": selection,
                           "board_value": "robust Gaussian mean over the pixel map "
                                          "(quote_resolution.board_summary)",
                           "error_bar": "pixel-to-pixel sigma; stat = sigma/sqrt(n_pixels)",
                           "fluence": "facility bookkeeping, p/cm2, 24 GeV protons; no n_eq"},
           "values": payload}
    p = os.path.join(out_dir, "%s_values.json" % stem)
    with open(p, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=False)
    return p


# ---- front-end operating point disclosure (user rule 2026-09-18) --------------------------------
# Every figure or panel that shows resolution numbers or TDC data states, in its right title,
# the preamp feedback setting WITH the resistor value, the discriminator threshold relative to
# the baseline, and the preamp power mode (slide-12 style).  ETROC2 RFSel -> feedback resistor:
RFSEL_RF_KOHM = {0: 20.0, 1: 10.0, 2: 5.7, 3: 4.4}
PREAMP_POWER = "high"        # run_metadata power_mode: high on every 2026 IRRAD and DESY run


def _rf_text(rfsel):
    r = RFSEL_RF_KOHM[int(rfsel)]
    r = ("%d" % r) if float(r).is_integer() else ("%.1f" % r)
    return r"RFSel %d (R$_\mathrm{f}$ = %s k$\Omega$)" % (int(rfsel), r)


def settings_text(rfsel=2, offset=20, power=None, sep=", "):
    """Operating-point string for a title line, e.g.
    'RFSel 2 (R_f = 5.7 kOhm), Disc threshold = baseline + 20, preamp high power'.
    rfsel: int, or a list/tuple for figures that compare settings ('RFSel 1/2/3 (10/5.7/4.4 kOhm)'),
           or None when the RFSel is the scanned variable (clause omitted; say it on the axis).
    offset: int, or a (lo, hi) range -> 'baseline + 6-8', or a list -> 'baseline + 10/20',
           or None when the offset is the scanned variable (clause omitted).
    power: None -> PREAMP_POWER."""
    parts = []
    if rfsel is not None:
        if isinstance(rfsel, (list, tuple)):
            rs = [int(r) for r in rfsel]
            ks = ["%g" % RFSEL_RF_KOHM[r] for r in rs]
            parts.append(r"RFSel %s (R$_\mathrm{f}$ = %s k$\Omega$)" % ("/".join(map(str, rs)), "/".join(ks)))
        else:
            parts.append(_rf_text(rfsel))
    if offset is not None:
        if isinstance(offset, tuple) and len(offset) == 2:
            o = "%d-%d" % (int(offset[0]), int(offset[1]))
        elif isinstance(offset, (list, tuple)):
            o = "/".join("%d" % int(v) for v in offset)
        else:
            o = "%d" % int(offset)
        parts.append("Disc threshold = baseline + %s" % o)
    parts.append("preamp %s power" % (PREAMP_POWER if power is None else power))
    return sep.join(parts)
