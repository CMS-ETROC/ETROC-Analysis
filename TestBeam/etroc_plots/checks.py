"""House-rule checks for figures made with etroc_plots.

Two entry points:

* layout_rules(fig, name): rules read off a live matplotlib figure. style.audit_figure() runs
  them together with the text-overlap audit, and the notebooks store what they find in each
  figure's <stem>_values.json (the audit lists), so a finding is recorded, never silently fixed.
* check_outputs(folder), also from the shell after a notebook has run:

      python -m etroc_plots.checks OUT_DIR [OUT_DIR ...]

  measures every saved PNG and reads back every values file. Exit status 1 when anything fails.

The rules (CONVENTIONS.md gives the reasons):

  header     the left of the header is "CMS" and the campaign's EXP_TEXT, nothing else, and no
             axes carries a left title; the right side is at most two lines, so neither right
             line may contain a line break
  spines     no text touches an axes frame: it sits wholly inside or wholly outside it (the
             CMS header itself stands on the top frame line, by design); an annotation is
             measured by its text, since its arrow may reach a frame
  footer     footer text (style.footer) lies below every axes, its labels and every figure
             legend, in its own band
  callbacks  no figure or slide numbers, notes folders, script or notebook names, or facility
             and home paths in any figure text
  top band   the blank band above the topmost ink of a saved PNG is at most 60 px at 200 dpi
  values     every PNG has its values file (a single panel, <stem>_NN_<name>.png or
             panels/<stem>/<name>.png, shares its figure's); every values file records the
             audit (overlap_problems) and names its inputs and conventions block, and every
             audit list in it is empty. A missing commit, uncommitted code or an empty inputs
             list is a warning, not a failure.

Every rule walks the child axes too (a secondary axis, an inset), which fig.axes leaves out.
The layout rules are stored in the values file as the figure is drawn; the top band and the
values rules are measured by check_outputs on the saved files. The audits of single panels
(PANELS = True in a notebook) are printed as the notebook runs; a panel has no values file of
its own, so check_outputs measures its top band and looks for its figure's values file.
"""
import glob
import json
import os
import re
import sys

from .campaigns import active as campaign
from .etroc_style import axes_legends, named_axes, text_extent

TOP_BAND_MAX_PX = 60      # at 200 dpi; scaled with the dpi a PNG was saved at
AUDIT_KEYS = ("overlap_problems", "legend_overlap_problems", "legend_box_problems")
CALLBACK = re.compile(r"\b(fig(ure)?|slide)s?\.?\s*\d|\bnotes/|\.(py|ipynb)\b"
                      r"|/(eos|afs|store|Users|home|nashome)/|~/", re.IGNORECASE)


def _texts(fig, renderer):
    """(text, owner) for every visible, non-empty text of the figure, tick labels included
    when their tick lies inside the drawn view."""
    out = [(t, "figure") for t in fig.texts]
    for lg in fig.legends:
        out += [(t, "figure legend") for t in list(lg.get_texts()) + [lg.get_title()]]
    for who, ax in named_axes(fig):
        out += [(t, who) for t in ax.texts]
        out += [(t, who) for t in (ax.title, ax._left_title, ax._right_title,
                                   ax.xaxis.label, ax.yaxis.label)]
        for lg in axes_legends(ax):          # every legend, not only ax.get_legend()
            out += [(t, who + " legend") for t in list(lg.get_texts()) + [lg.get_title()]]
        ab = ax.get_window_extent(renderer=renderer)
        for t in ax.get_xticklabels():
            bb = t.get_window_extent(renderer=renderer)
            if bb.x0 >= ab.x0 - 2 and bb.x1 <= ab.x1 + 2:
                out.append((t, who + " tick"))
        for t in ax.get_yticklabels():
            bb = t.get_window_extent(renderer=renderer)
            if bb.y0 >= ab.y0 - 2 and bb.y1 <= ab.y1 + 2:
                out.append((t, who + " tick"))
    return [(t, w) for t, w in out
            if t is not None and t.get_visible() and str(t.get_text()).strip()]


def _touches_frame(bb, ab, pad=1.0):
    """True when box `bb`, grown by `pad` px, meets one of the four frame lines of box `ab`."""
    x0, y0, x1, y1 = bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad
    rows = y1 >= ab.y0 and y0 <= ab.y1          # overlaps the frame's height
    cols = x1 >= ab.x0 and x0 <= ab.x1          # overlaps the frame's width
    return ((rows and (x0 <= ab.x0 <= x1 or x0 <= ab.x1 <= x1))
            or (cols and (y0 <= ab.y0 <= y1 or y0 <= ab.y1 <= y1)))


def layout_rules(fig, name):
    """The header, spines, footer and callbacks rules on a drawn figure (call it after
    style.lower_footer, so the footer sits where it is saved). Returns the list of problems and
    prints it."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    problems = []
    texts = _texts(fig, r)

    # header
    axes = named_axes(fig)                # child axes too: a secondary axis, an inset
    suffixes = [t for _, ax in axes for t in ax.texts if type(t).__name__ == "ExpSuffix"]
    if not suffixes:
        problems.append("header: no CMS header on any axes")
    for t in suffixes:
        if t.get_text() != campaign.EXP_TEXT:
            problems.append("header: left text %r is not the campaign's %r"
                            % (t.get_text(), campaign.EXP_TEXT))
    for who, ax in axes:
        if ax._left_title.get_text().strip():
            problems.append("header: %s has a left title %r" % (who, ax._left_title.get_text()))
        for t in (ax._right_title, getattr(ax, "_etroc_right_top", None)):
            if t is not None and "\n" in t.get_text():
                problems.append("header: a right line of %s breaks into several: %r"
                                % (who, t.get_text()))

    # spines
    frames = [ax.get_window_extent(renderer=r) for _, ax in axes
              if ax.axison and any(s.get_visible() for s in ax.spines.values())]
    for t, who in texts:
        if type(t).__name__ in ("ExpText", "ExpSuffix"):
            continue
        bb = text_extent(t, r)
        if bb.width <= 0 or bb.height <= 0:
            continue
        if any(_touches_frame(bb, ab) for ab in frames):
            problems.append("spines: %s text %r touches an axes frame"
                            % (who, " ".join(str(t.get_text()).split())[:46]))

    # footer
    feet = [t for t in fig.texts if getattr(t, "_etroc_footer", False) and t.get_visible()]
    if feet:
        floor = min([ax.get_tightbbox(r).y0 for ax in fig.axes if ax.get_visible()]
                    + [lg.get_window_extent(renderer=r).y0 for lg in fig.legends
                       if lg.get_visible()])
        for t in feet:
            if t.get_window_extent(renderer=r).y1 > floor:
                problems.append("footer: %r reaches up into the axes or a figure legend"
                                % " ".join(t.get_text().split())[:46])

    # callbacks
    for t, who in texts:
        m = CALLBACK.search(str(t.get_text()))
        if m:
            problems.append("callbacks: %s text %r mentions %r"
                            % (who, " ".join(str(t.get_text()).split())[:46], m.group(0)))

    print("  layout rules on %s: %s" % (name, "all hold" if not problems
                                        else "%d PROBLEM(S)" % len(problems)))
    for p in problems:
        print("      %s" % p)
    return problems


def top_band_px(png_path, luminance=200, every=4):
    """Height of the blank band above the topmost ink: the first pixel row holding a pixel darker
    than `luminance` (0-255), sampling every `every`-th column."""
    from PIL import Image
    im = Image.open(png_path).convert("L")
    w, h = im.size
    px = im.load()
    for y in range(h):
        for x in range(0, w, every):
            if px[x, y] < luminance:
                return y
    return h


def check_outputs(folder):
    """Measure every PNG in `folder` and read back every values file. Returns (failures,
    warnings), two lists of strings."""
    from PIL import Image
    failures, warnings = [], []
    if not os.path.isdir(folder):
        failures.append("%s: no such folder" % folder)
        print("%s: no such folder" % folder)
        return failures, warnings
    pngs = sorted(glob.glob(os.path.join(folder, "*.png")))
    # single panels kept in a folder of their figure's name: panels/<stem>/<name>.png
    tiles = sorted(glob.glob(os.path.join(folder, "panels", "*", "*.png")))
    value_files = sorted(glob.glob(os.path.join(folder, "*_values.json")))
    if not pngs:
        failures.append("%s: no PNG files (is it the folder the figures were written to?)"
                        % folder)
    stems = [os.path.basename(p)[:-len("_values.json")] for p in value_files]
    for p in pngs + tiles:
        name = os.path.relpath(p, folder)
        if p in tiles:
            has_values = os.path.basename(os.path.dirname(p)) in stems
        else:
            stem = os.path.basename(p)[:-len(".png")]
            has_values = stem in stems or any(re.match(re.escape(s) + r"_\d\d_[^_]", stem)
                                              for s in stems)
        if not has_values:
            failures.append("%s: no values file" % name)
        # a PNG stores pixels per metre, so 200 dpi reads back as 199.9996
        dpi = round(float(Image.open(p).info.get("dpi", (200, 200))[0]))
        limit = TOP_BAND_MAX_PX * dpi / 200.0
        band = top_band_px(p)
        if band > limit:
            failures.append("%s: top band %d px, more than %.0f px at %.0f dpi"
                            % (name, band, limit, dpi))
    for p in value_files:
        f = os.path.basename(p)
        with open(p) as fh:
            doc = json.load(fh)
        values = doc.get("values") or {}
        if "overlap_problems" not in values:
            failures.append("%s: no audit recorded (overlap_problems): the figure was not "
                            "checked with style.audit_figure" % f)
        for k in AUDIT_KEYS:
            for item in values.get(k) or []:
                failures.append("%s: %s: %s" % (f, k, item))
        if not isinstance(doc.get("inputs"), list):
            failures.append("%s: no inputs list" % f)
        elif not doc["inputs"]:
            warnings.append("%s: the inputs list is empty" % f)
        if not doc.get("conventions"):
            failures.append("%s: no conventions block" % f)
        commit = doc.get("commit") or ""
        if not commit:
            warnings.append("%s: no commit recorded (not run from a git checkout?)" % f)
        elif commit.endswith("-dirty") or commit.endswith("-unknown"):
            warnings.append("%s: made from uncommitted code (%s)" % (f, commit))
    print("%s: %d PNGs, %d values files, %d failure(s), %d warning(s)"
          % (folder, len(pngs) + len(tiles), len(value_files), len(failures), len(warnings)))
    return failures, warnings


def main(argv=None):
    folders = (sys.argv[1:] if argv is None else argv) or ["."]
    bad = 0
    for folder in folders:
        failures, warnings = check_outputs(folder)
        for w in warnings:
            print("  warning: " + w)
        for f in failures:
            print("  FAIL: " + f)
        bad += len(failures)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
