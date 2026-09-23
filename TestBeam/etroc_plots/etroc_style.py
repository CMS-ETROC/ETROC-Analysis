"""Shared style for the DESY May 2026 analysis-walkthrough figures.

House style: the ETROC-Analysis repo convention (TestBeam/BeamTestHelpers/plotting_basic.py) —
`mplhep` CMS style, the experiment label written at the header left with
`hep.cms.text(loc=0, text="ETL ETROC Test Beam")`, the figure name written at the header right with
`ax.set_title(..., loc="right")`, axis labels at 25 pt / tick labels at 17 pt / header 18 pt /
right title 16 pt on a single (11, 10) panel, short ticks on all four sides.  Multi-panel figures
keep the same proportions through `scale`, so text stays legible at 200 dpi.

Palette: dataviz skill categorical slots 1/2/3(dark step)/7, validated all-pairs with the ported
validator (validate_palette.py) on the CMS white ground: CVD dE 8.4, normal-vision dE 16.3,
contrast PASS.  Colour follows the BOARD INDEX (physical slot in the telescope), never its rank.
Marker shape repeats the same identity so nothing is carried by colour alone.
"""
import os
import matplotlib as mpl
import mplhep as hep

# ------------------------------------------------------------------ identity of the campaign
EXP_TEXT = "ETL ETROC Test Beam"      # the repo's own experiment label for test-beam figures
CAMPAIGN = "DESY, May 2026"

# board index 0..3 -> colour / marker.  Identity is (colour, shape), used in every figure.
BOARD_COLOR = {0: "#2a78d6", 1: "#eb6834", 2: "#199e70", 3: "#4a3aa7"}
BOARD_MARKER = {0: "o", 1: "s", 2: "^", 3: "D"}

# board index -> (role, chip) for the DESY May 2026 telescope, from the campaign run
# configuration.  The board index IS the order along the beam.
# The role word is PIPELINE BOOK-KEEPING ONLY (it is how the analysis names the four slots of a
# board combination); all four boards are devices under test and no figure labels one "DUT".
# It is kept here because the alignment files key their board combinations on it, and it is
# carried into value JSONs as a plain field for traceability — never into a figure.
DESY_BOARDS = {0: ("extra", "IH27"), 1: ("ref", "IH5"), 2: ("trig", "IH3"), 3: ("dut", "IH2")}
ROLE_TO_IDX = {v[0]: k for k, v in DESY_BOARDS.items()}

# the hybrid family, used where a panel has to name its own telescope
HYBRID = "ETROC2.02 + HPK LGAD hybrids"
SUBJECT = "DESY May 2026 telescope"

INK = "#000000"
INK_MUTED = "#5f5f5c"
GRID = "#c9c9c9"
SURFACE = "#ffffff"
ALERT = "#b3261e"

# reference point sizes: the repo's single-panel (11, 10) CMS figure
_REF = dict(label=25.0, tick=17.0, header=18.0, title=16.0,
            legend=16.0, legend_title=15.0, ann=14.0)


def sizes(scale=1.0):
    """Point sizes for a panel `scale` times the repo's reference (11, 10) panel."""
    return {k: round(v * scale, 1) for k, v in _REF.items()}


def apply_style(scale=1.0):
    """CMS house style, with the repo's font/tick choices scaled to the panel size."""
    hep.style.use("CMS")
    s = sizes(scale)
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": s["label"],
        "axes.labelsize": s["label"],
        "axes.titlesize": s["title"],
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "axes.linewidth": max(1.0, 1.6 * scale),
        "xtick.labelsize": s["tick"], "ytick.labelsize": s["tick"],
        "xtick.color": INK, "ytick.color": INK,
        "xtick.major.size": 6.0, "ytick.major.size": 6.0,
        "xtick.minor.size": 3.0, "ytick.minor.size": 3.0,
        "xtick.major.width": max(0.8, 1.2 * scale), "ytick.major.width": max(0.8, 1.2 * scale),
        "xtick.minor.width": max(0.6, 0.9 * scale), "ytick.minor.width": max(0.6, 0.9 * scale),
        "legend.fontsize": s["legend"],
        "legend.frameon": True, "legend.framealpha": 0.95, "legend.edgecolor": GRID,
        "grid.color": GRID, "grid.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "figure.dpi": 110,
        "axes.formatter.useoffset": False,
        "axes.formatter.use_mathtext": False,
        "text.color": INK,
    })
    return s


def data_line(*parts):
    """The header's third line: WHAT DATA the figure shows, joined with the house separator.

    Every figure drawn from data states its run, its board combination and, where the figure is
    about one of them, its track or pixel - so a reader can tell from the header alone which
    subset of the campaign is on the page.  Cartoons and method diagrams take no data and so get
    no third line.
    """
    return u"  ·  ".join(str(p) for p in parts if p)


def cms_header(ax, name, subject=None, tag=None, data=None, scale=1.0, pad=None,
               campaign=CAMPAIGN, exp_text=None):
    """The repo header: experiment label left, figure name (+ campaign / conditions) right.

    name     the figure's name, e.g. "Beam Profile"
    subject  optional panel subject appended to the name, e.g. "H1 (HPK)"
    tag      optional conditions string, written after the campaign on the second title line
    data     optional third line naming the data drawn - build it with data_line()
    exp_text optional override for the left-hand experiment label; defaults to EXP_TEXT
    """
    s = sizes(scale)
    hep.cms.text(loc=0, ax=ax, text=EXP_TEXT if exp_text is None else exp_text,
                 fontsize=s["header"])
    line1 = name if subject is None else f"{name} · {subject}"
    line2 = campaign if tag is None else f"{campaign} · {tag}"
    lines = [line1, line2] if data is None else [line1, line2, data]
    kw = {} if pad is None else {"pad": pad}
    ax.set_title("\n".join(lines), loc="right", size=s["title"], color=INK, **kw)
    return s


def check_no_clipping(fig, name, min_overlap=0.16, pad_px=1.0):
    """Every text artist must sit inside the canvas and clear every other text artist.

    Reported, never silently fixed: these figures are laid out in inches by hand, so a report here
    means a placement has to change.  A tick label whose tick lies outside the drawn view is
    skipped - nothing of it is on the page - and the texts of one legend are not compared with each
    other, since the legend lays them out itself.  Returns the list of problems, and prints it.

    All THREE title artists of a panel are checked.  matplotlib keeps the loc="left" and
    loc="right" titles in `_left_title` / `_right_title`, apart from the centred `ax.title`, so a
    check that collected only `ax.title` saw neither a panel subtitle written with
    `set_title(..., loc="left")` nor a `cms_header` line - and a row title sitting on top of one of
    them passed silently.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []

    def collect(artist, owner, kind):
        try:
            if artist is None or not artist.get_visible():
                return
            if not str(artist.get_text()).strip():
                return
            bb = artist.get_window_extent(renderer=r)
        except Exception:
            return
        if bb.width <= 0 or bb.height <= 0:
            return
        items.append(dict(bb=bb, owner=owner, kind=kind,
                          text=" ".join(str(artist.get_text()).split())[:46]))

    for t in fig.texts:
        collect(t, "figure", "text")
    for i, ax in enumerate(fig.axes):
        who = "axes[%d]" % i
        for t in ax.texts:
            collect(t, who, "text")
        for attr in ("title", "_left_title", "_right_title"):
            collect(getattr(ax, attr, None), who, "title")
        collect(ax.xaxis.label, who, "xlabel")
        collect(ax.yaxis.label, who, "ylabel")
        ab = ax.get_window_extent(renderer=r)
        for t in ax.get_xticklabels():
            bb = t.get_window_extent(renderer=r)
            if bb.x0 >= ab.x0 - 2 and bb.x1 <= ab.x1 + 2:
                collect(t, who, "tick")
        for t in ax.get_yticklabels():
            bb = t.get_window_extent(renderer=r)
            if bb.y0 >= ab.y0 - 2 and bb.y1 <= ab.y1 + 2:
                collect(t, who, "tick")
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts() + ([leg.get_title()] if leg.get_title() else []):
                collect(t, who + ":legend", "legend")

    fb = fig.bbox
    problems = []
    for it in items:
        b = it["bb"]
        if (b.x0 < fb.x0 - pad_px or b.y0 < fb.y0 - pad_px
                or b.x1 > fb.x1 + pad_px or b.y1 > fb.y1 + pad_px):
            problems.append("outside the canvas: %s %s %r" % (it["owner"], it["kind"], it["text"]))

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            if a["kind"] == "legend" and b["kind"] == "legend" and a["owner"] == b["owner"]:
                continue
            ix = min(a["bb"].x1, b["bb"].x1) - max(a["bb"].x0, b["bb"].x0)
            iy = min(a["bb"].y1, b["bb"].y1) - max(a["bb"].y0, b["bb"].y0)
            if ix <= 0 or iy <= 0:
                continue
            area = ix * iy
            small = min(a["bb"].width * a["bb"].height, b["bb"].width * b["bb"].height)
            if area / small > min_overlap:
                problems.append("overlap %.0f %%: %s %s %r  vs  %s %s %r"
                                % (100 * area / small, a["owner"], a["kind"], a["text"],
                                   b["owner"], b["kind"], b["text"]))

    print("  overlap check on %s: %d text items, %s"
          % (name, len(items), "nothing clips" if not problems
             else "%d PROBLEM(S)" % len(problems)))
    for p in problems:
        print("      %s" % p)
    return problems


def style_axes(ax, scale=1.0, grid=True, minor=True, grid_alpha=0.45):
    """Repo tick/label treatment on one panel."""
    s = sizes(scale)
    ax.xaxis.label.set_fontsize(s["label"])
    ax.yaxis.label.set_fontsize(s["label"])
    ax.tick_params(axis="both", which="major", length=6, labelsize=s["tick"])
    ax.tick_params(axis="both", which="minor", length=3)
    if not minor:
        ax.minorticks_off()
    if grid:
        ax.grid(True, which="major", alpha=grid_alpha, zorder=0)
        ax.set_axisbelow(True)
    return s


def board_label(idx):
    """Figure label: the chip name, nothing else."""
    return DESY_BOARDS[idx][1]


def board_chip(idx):
    return DESY_BOARDS[idx][1]


def board_role(idx):
    """Pipeline bookkeeping label. For JSON provenance only, never for a figure."""
    return DESY_BOARDS[idx][0]


# ============================================================= single-panel export (--panel)
#
# Every compound-figure script keeps a module-level `PANELS = {name: (draw, title)}` registry.
# `draw` is always called as `draw(ax_or_fig, ctx)` - one matplotlib Axes for a panel that owns a
# single set of axes, or the Figure itself for a panel that needs several (a map with its
# colourbar, a grid of maps, a row of histograms), which then adds its own axes onto it exactly as
# the compound figure's assembly function does.  `ctx` is whatever small bundle of already-loaded
# data that script's panels need; building it is the script's own `_panel_context()`.  These three
# helpers are the CLI/registry plumbing shared by every script; the per-panel drawing and sizing
# stays with the script that owns the data.


def add_panel_arg(ap):
    """--panel NAME (repeatable), or --panel all: the one CLI option every plot script gets.

    Leaves the parser's other options untouched.  With no --panel the script's behaviour (and
    its output) is unchanged; --panel switches it to writing single-panel files instead of the
    compound figure.
    """
    ap.add_argument("--panel", action="append", default=None, metavar="NAME",
                    help="render just this one panel of PANELS (repeatable), or 'all' for "
                         "every panel; writes figures/panels/<stem>/<NN>_<name>.{png,pdf} "
                         "instead of the compound figure")
    return ap


def resolve_panels(requested, panels):
    """`requested` (the --panel list, or None/empty) against an ordered {name: (draw, title)}.

    Returns an ordered list of (1-based index, name, draw, title); an empty `requested` (falsy)
    resolves to nothing, so `if resolve_panels(...):` is the same switch as `if a.panel:`.  An
    unknown name is a hard error naming the valid choices, the same way argparse itself reports
    a bad --choice, so a typo is caught immediately rather than silently skipped.
    """
    if not requested:
        return []
    names = list(panels.keys())
    chosen = names if any(r == "all" for r in requested) else []
    if not chosen:
        for r in requested:
            if r not in panels:
                raise SystemExit("unknown --panel %r; choices are: %s, or 'all'"
                                % (r, ", ".join(names)))
            if r not in chosen:
                chosen.append(r)
    return [(names.index(n) + 1, n, panels[n][0], panels[n][1]) for n in chosen]


def panel_subject(index, title, base_subject=None):
    """The header's right-title subject for one panel: '(a) Title', or 'base · (a) Title'.

    Letters past (z) fall back to the plain index so a 27th panel does not raise.
    """
    letter = chr(ord("a") + index - 1) if 1 <= index <= 26 else str(index)
    tag = "(%s)  %s" % (letter, title)
    return tag if not base_subject else u"%s  ·  %s" % (base_subject, tag)


def save_panel(fig, out, stem, index, name, dpi=200):
    """Save one single-panel figure to figures/panels/<stem>/<NN>_<name>.{png,pdf}.

    `stem` is the compound figure's own stem (e.g. "fig21_coverage_run26"), so the panel files
    of a script that draws more than one compound figure (a --run switch, or fig01's --only)
    land in one directory per compound figure, never mixed together.  Returns the two paths.
    """
    d = os.path.join(out, "panels", stem)
    os.makedirs(d, exist_ok=True)
    base = os.path.join(d, "%02d_%s" % (index, name))
    paths = []
    for ext in ("png", "pdf"):
        p = "%s.%s" % (base, ext)
        fig.savefig(p, dpi=dpi if ext == "png" else None, facecolor=SURFACE)
        paths.append(p)
    return paths


def header_axis(fig, ml_frac, y_frac, width_frac):
    """An invisible full-width axis to carry cms_header, sitting above the real panel axes.

    Every compound-figure assembly function in this folder builds exactly this - a slim,
    tickless, spineless axis spanning the header band, so `cms_header`'s two-or-three-line
    title and the `hep.cms.text` label at its left never compete with a panel's own
    `ax.set_title`.  `ml_frac`/`width_frac` are the axis's x0/width in figure fraction (e.g.
    `ML / W`, `(W - ML - MR) / W`); `y_frac` is its y0 in figure fraction.
    """
    hax = fig.add_axes([ml_frac, y_frac, width_frac, 1e-6])
    hax.set_xticks([]); hax.set_yticks([])
    for sp in hax.spines.values():
        sp.set_visible(False)
    return hax
