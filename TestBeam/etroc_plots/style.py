"""Figure style shared by every notebook: the house look, headers, legends, footers, output.

Builds on etroc_style (the mplhep CMS style, point sizes, the overlap audit, single-panel export)
and on checks (the house layout rules), and adds the conventions every figure follows:

* colour = FLUENCE: one ladder per campaign, the campaign module's FLUENCE_STEPS and
  FLUENCE_COLOR;
* marker + line style = CHIP, by the chip's slot in its telescope; legends name chips only, the
  vendor sits in the panel title;
* header: "CMS" and the campaign's EXP_TEXT alone on the left, immediately above the axes; every
  other line (facility line, telescope or panel title, data line) on the right, at most two lines;
* footer: selection, vetoes and counts in a band of its own below the axes, so it can be cropped
  away without touching the plot;
* every figure is saved as PNG and PDF, with <stem>_values.json beside it (the numbers drawn, the
  inputs read, the commit of the code).

What the campaign is (experiment text, header lines, telescopes and chips, fluence ladder) comes
from the active campaign module, etroc_plots.campaigns.
"""
import json
import os
import subprocess
import time

_HERE = os.path.dirname(os.path.abspath(__file__))

import mplhep as hep                                   # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402
from .campaigns import active as campaign, check_required   # noqa: E402
from . import checks                                   # noqa: E402
from .etroc_style import (                              # noqa: E402,F401  (re-exports)
    apply_style, sizes, style_axes, check_no_clipping,
    add_panel_arg, resolve_panels, panel_subject, save_panel,
    INK, INK_MUTED, GRID, SURFACE, ALERT,
)

check_required("style")


# ------------------------------------------------------------------ colour = fluence
def fluence_key(f, rtol=0.2):
    """Snap a bookkeeping fluence (1.5e15, 15e14, '1.5e15') to a step of the campaign's
    FLUENCE_STEPS.

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
    near = [step for step in campaign.FLUENCE_STEPS[1:] if abs(x - step) <= rtol * step]
    if not near:
        raise ValueError("fluence %r is not on the fluence ladder %s" % (f, campaign.FLUENCE_STEPS))
    return min(near, key=lambda step: abs(x - step) / step)


def fluence_color(f):
    return campaign.FLUENCE_COLOR[fluence_key(f)]


def fluence_label(f, unit=True):
    """Legend text for a ladder step: 'pre-irradiation' or '3e14 p/cm2' in mathtext."""
    k = fluence_key(f)
    if k == 0.0 or not unit:
        return campaign.FLUENCE_TEXT[k]
    return "%s %s" % (campaign.FLUENCE_TEXT[k], campaign.FLUENCE_UNIT)


# ------------------------------------------------------------------ marker + line = chip
# Marker shape follows the chip's slot in its telescope (the shapes etroc_style.BOARD_MARKER uses
# for the board index); line style repeats the same identity so nothing rides on the marker alone.
_SLOT_MARKER = ("o", "s", "^", "D")
_SLOT_LINESTYLE = ("-", "--", "-.", ":")
CHIP_MARKER = {}
CHIP_LINESTYLE = {}
for _tel, _chips in campaign.TELESCOPE_CHIPS.items():
    for _i, _c in enumerate(_chips):
        CHIP_MARKER[_c] = _SLOT_MARKER[_i]
        CHIP_LINESTYLE[_c] = _SLOT_LINESTYLE[_i]
# the name prefixes of the campaign's chips ("IH", "LF"): chip_short finds a chip name by them
_CHIP_PREFIXES = {c.rstrip("0123456789") for chips in campaign.TELESCOPE_CHIPS.values()
                  for c in chips}


def chip_short(name):
    """'PT_IH12' / 'LF10_FBK' / 'IH12' -> 'IH12' / 'LF10': the chip name as legends print it."""
    s = str(name)
    for tok in s.replace("-", "_").split("_"):
        head = tok.rstrip("0123456789")
        if head in _CHIP_PREFIXES and head != tok:
            return tok
    return s


def chip_marker(chip):
    return CHIP_MARKER[chip_short(chip)]


def chip_linestyle(chip):
    return CHIP_LINESTYLE[chip_short(chip)]


# a point from merged runs is drawn with a thick black marker edge
MERGED_EDGE = dict(markeredgecolor="#000000")
MERGED_EDGE_WIDTH = 2.2          # times scale, applied in point_style


def point_style(chip, fluence, merged=False, scale=1.0, line=True, open_marker=False, color=None):
    """Keyword arguments for ax.errorbar / ax.plot: one board series at one fluence step.

    colour from the fluence ladder (or `color`, where colour means something else on the
    figure), marker and line style from the chip, thick black marker edge when the point comes
    from merged runs.  `open_marker=True` draws a hollow marker, for a second data set sharing
    the axis.  `line=False` draws markers only (fluence axes).
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


# ------------------------------------------------------------------ header, labels
XLABEL_BIAS = "Bias voltage [V]"
YLABEL_RES = "Time resolution [ps]"
XLABEL_FLUENCE = "Fluence [%s]" % campaign.FLUENCE_UNIT


def header(ax, name=None, tag=None, data=None, scale=1.0, pad=None, line1=None, line2=None):
    """The figure header: "CMS" and the campaign's EXP_TEXT alone on the left, written
    immediately above the axes; on the right AT MOST TWO lines:

        top     facility line: (line1 or HEADER_LINE_1) + ", " + (line2 or HEADER_LINE_2), the
                campaign's two header lines unless replaced
        bottom  subject: name, tag and data joined by ", " (omitted when all three are empty)

    name   telescope / panel name ("H1 telescope, HPK") or, on a compound's header axis,
           the figure name
    tag    conditions, e.g. "RFSel 2" or "RFSel 2, threshold offset 20"
    data   the data drawn, e.g. "March 2026"
    line1  replaces HEADER_LINE_1 (IV scans: the campaign's LINE1_PARKED)
    line2  replaces HEADER_LINE_2; "" drops it

    The bottom line shares its height with the experiment text on the left, so it has to be
    short (about 44 characters at the single-panel title size); the overlap audit flags a clash.
    """
    s = sizes(scale)
    hep.cms.text(loc=0, ax=ax, text=campaign.EXP_TEXT, fontsize=s["header"])
    facility = ", ".join(x for x in ((line1 or campaign.HEADER_LINE_1),
                                     (campaign.HEADER_LINE_2 if line2 is None else line2)) if x)
    subject = ", ".join(str(x) for x in (name, tag, data) if x)
    right_lines(ax, [subject] if subject else [], s["title"], pad=pad, facility=[facility])
    return s


def right_lines(ax, lines, size, pad=None, facility=()):
    """Write header lines on the RIGHT of `ax`, merged with the lines the axis already carries
    (header and panel_title may both write to one axis, in either order), at most TWO lines: the
    `facility` lines joined on top, the other `lines` joined into one bottom line with ", ".
    The bottom line is the axis's right title, so it sits level with the experiment text; the
    top line is a separate text artist above it, so the overlap audit measures each line on its
    own.  The header left never carries anything but the experiment text."""
    import matplotlib as _mpl
    fac, rest = getattr(ax, "_etroc_right_lines", ([], []))
    fac = fac + [l for l in facility if l and l not in fac]
    rest = rest + [l for l in lines if l and l not in rest]
    ax._etroc_right_lines = (fac, rest)
    merged = ([", ".join(fac)] if fac else []) + ([", ".join(rest)] if rest else [])
    top_pad = float(_mpl.rcParams["axes.titlepad"] if pad is None else pad)
    kw = {} if pad is None else {"pad": pad}
    ax.set_title("", loc="left")
    ax.set_title(merged[-1] if merged else "", loc="right", size=size, color=INK, **kw)
    old = getattr(ax, "_etroc_right_top", None)
    if old is not None:
        old.remove()
        ax._etroc_right_top = None
    if len(merged) == 2:
        ax._etroc_right_top = ax.annotate(
            merged[0], xy=(1.0, 1.0), xycoords="axes fraction",
            xytext=(0.0, top_pad + 1.3 * size), textcoords="offset points",
            ha="right", va="bottom", fontsize=size, color=INK, annotation_clip=False)

def compound_header(axes, tag=None, data=None, line1=None, line2=None, scale=1.0, subjects=None,
                    pad=None):
    """Header for a COMPOUND figure WITHOUT a header axis (the experiment text sits immediately
    above the axes, so nothing may sit between them).  `axes` = the top
    row of panels, left to right.  The experiment text goes on the first panel; the facility
    line (line1/line2 defaults, then tag and data appended: the compound top line is wide) goes
    on the top right of the LAST panel and may extend leftwards over the other panels' header
    space; every panel gets its own subject on its bottom right: subjects[i], or whatever
    panel_title() already wrote there.  Give EVERY panel a subject, so the facility line is
    always the top line and never drops level with the experiment text.  Returns sizes(scale).
    """
    s = sizes(scale)
    axes = list(axes)
    hep.cms.text(loc=0, ax=axes[0], text=campaign.EXP_TEXT, fontsize=s["header"])
    facility = ", ".join(str(x) for x in ((line1 or campaign.HEADER_LINE_1),
                                          (campaign.HEADER_LINE_2 if line2 is None else line2),
                                          tag, data) if x)
    for i, ax in enumerate(axes):
        subj = subjects[i] if subjects and i < len(subjects) and subjects[i] else None
        last = i == len(axes) - 1
        if last or subj:
            right_lines(ax, [subj] if subj else [], s["title"], pad=pad,
                        facility=[facility] if last else [])
    return s


def panel_title(ax, telescope, extra=None, scale=1.0):
    """Panel title, the campaign's TELESCOPE_TITLE ('H1 telescope, HPK', + ' · extra'), written
    on the header RIGHT (below any facility line already there).  Nothing is ever written on the
    left below the experiment text, so it stays immediately above the axes."""
    s = sizes(scale)
    t = campaign.TELESCOPE_TITLE[telescope.lower()]
    if extra:
        t = u"%s  ·  %s" % (t, extra)
    right_lines(ax, [t], s["title"])
    return s


# ------------------------------------------------------------------ legends
def fluence_handles(steps, scale=1.0):
    """Legend handles for the fluence ladder: a coloured line per step present (no marker)."""
    out = []
    for f in sorted({fluence_key(x) for x in steps}):
        out.append(Line2D([], [], color=campaign.FLUENCE_COLOR[f], linewidth=3.0 * scale,
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


def merged_handle(scale=1.0, label="merged runs"):
    return Line2D([], [], color=INK_MUTED, marker="o", markersize=9.0 * scale,
                  markerfacecolor=SURFACE, markeredgecolor="#000000",
                  markeredgewidth=MERGED_EDGE_WIDTH * scale, linestyle="none", label=label)


def two_legends(ax, steps, chips, scale=1.0, merged=False, loc_f="upper left",
                loc_c="upper right", ncol_c=1):
    """Fluence legend and chip legend on one axis (matplotlib keeps only the last ax.legend,
    so the first is re-added as an artist). `ncol_c` is the chip legend's number of columns: 1
    stacks the entries, more lay them out in rows that take less of the axis height."""
    s = sizes(scale)
    lf = ax.legend(handles=fluence_handles(steps, scale), loc=loc_f, fontsize=s["legend"],
                   title="fluence", title_fontsize=s["legend_title"])
    ax.add_artist(lf)
    hc = chip_handles(chips, scale) + ([merged_handle(scale)] if merged else [])
    lc = ax.legend(handles=hc, loc=loc_c, fontsize=s["legend"], ncol=ncol_c)
    return lf, lc


# ------------------------------------------------------------------ footer, outputs, provenance
FOOTER_GAP_IN = 0.5   # inches of blank band between the axes and the footer


def footer(fig, text, scale=1.0, x=0.01, y=0.008):
    """Selection / veto / counts line at the bottom of a COMPOUND figure (panels get none).
    save_figure() later pushes it down into its own band (see lower_footer) so the footer can be
    cropped away without touching the axes."""
    t = fig.text(x, y, text, ha="left", va="bottom", fontsize=sizes(scale)["ann"],
                 color=INK_MUTED)
    t._etroc_footer = True
    return t


def lower_footer(fig, gap=None):
    """Grow the figure by `gap` inches at the bottom and move every axes and figure-level text
    (except the footer) up by that amount, so the plot is pixel-identical and the footer sits
    alone in the new band.  Idempotent; no-op when the figure has no footer() text."""
    feet = [t for t in fig.texts if getattr(t, "_etroc_footer", False)]
    if not feet or getattr(fig, "_etroc_footer_lowered", False):
        return False
    fig._etroc_footer_lowered = True
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


def audit_figure(fig, name):
    """Every check a figure gets before it is saved: the text-overlap audit (check_no_clipping)
    and the house layout rules (checks.layout_rules). Returns one list of problems; the notebooks
    store it in the values file as `overlap_problems`. Call it after lower_footer()."""
    return check_no_clipping(fig, name) + checks.layout_rules(fig, name)


def save_figure(fig, out_dir, stem, dpi=200, audit=True):
    """<out_dir>/<stem>.png + .pdf; runs audit_figure first. Returns (paths, problems)."""
    os.makedirs(out_dir, exist_ok=True)
    lower_footer(fig)
    problems = audit_figure(fig, stem) if audit else []
    paths = []
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, "%s.%s" % (stem, ext))
        fig.savefig(p, dpi=dpi if ext == "png" else None, facecolor=SURFACE)
        paths.append(p)
    return paths, problems


def flatten_panels(out_dir, stem, paths):
    """Move save_panel's nested panels/<stem>/NN_name.{png,pdf} next to the compound figure,
    flat-named <out_dir>/<stem>_NN_name.* (the notebooks' single-panel naming), and remove the
    nested folder once it is empty."""
    nested_dir = None
    for p in paths:
        nested_dir = os.path.dirname(p)
        name, ext = os.path.basename(p).rsplit(".", 1)
        os.replace(p, os.path.join(out_dir, "%s_%s.%s" % (stem, name, ext)))
    if nested_dir and os.path.isdir(nested_dir) and not os.listdir(nested_dir):
        os.rmdir(nested_dir)
        parent = os.path.dirname(nested_dir)
        if os.path.isdir(parent) and not os.listdir(parent):
            os.rmdir(parent)


def write_values(out_dir, stem, payload, *, conventions, script=None, inputs=None):
    """<out_dir>/<stem>_values.json: the numbers drawn (`payload`) with provenance (script,
    commit, inputs, time) and `conventions`, a dict saying what the numbers are (units, binning,
    selection) for this kind of figure."""
    os.makedirs(out_dir, exist_ok=True)
    doc = {"figure": stem, "script": script, "commit": git_hash(),
           "inputs": inputs or [], "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "conventions": conventions,
           "values": payload}
    p = os.path.join(out_dir, "%s_values.json" % stem)
    with open(p, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=False)
    return p


# ---- front-end operating point ------------------------------------------------------------------
# Every figure or panel that shows resolution numbers or TDC data states, in its right title,
# the preamp feedback setting WITH the resistor value, the discriminator threshold relative to
# the baseline, and the preamp power mode.  ETROC2 RFSel -> feedback resistor:
RFSEL_RF_KOHM = {0: 20.0, 1: 10.0, 2: 5.7, 3: 4.4}


def rfsel_text(rfsel):
    """One RFSel with its feedback resistor, e.g. 'RFSel 2 (R_f = 5.7 kOhm)'."""
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
    power: None -> the campaign's PREAMP_POWER."""
    parts = []
    if rfsel is not None:
        if isinstance(rfsel, (list, tuple)):
            rs = [int(r) for r in rfsel]
            ks = ["%g" % RFSEL_RF_KOHM[r] for r in rs]
            parts.append(r"RFSel %s (R$_\mathrm{f}$ = %s k$\Omega$)" % ("/".join(map(str, rs)), "/".join(ks)))
        else:
            parts.append(rfsel_text(rfsel))
    if offset is not None:
        if isinstance(offset, tuple) and len(offset) == 2:
            o = "%d-%d" % (int(offset[0]), int(offset[1]))
        elif isinstance(offset, (list, tuple)):
            o = "/".join("%d" % int(v) for v in offset)
        else:
            o = "%d" % int(offset)
        parts.append("Disc threshold = baseline + %s" % o)
    parts.append("preamp %s power" % (campaign.PREAMP_POWER if power is None else power))
    return sep.join(parts)
