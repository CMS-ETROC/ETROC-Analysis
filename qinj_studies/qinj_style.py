"""House style for the charge-injection figures of this folder.

This is a NEW file written for `qinj_studies/`, not a copy of anything in the repo. It follows
the ETROC-Analysis plotting convention of `TestBeam/BeamTestHelpers/plotting_basic.py` and of
the analysis-walkthrough figures — `mplhep` CMS style, the experiment label written at the
header left with `hep.cms.text(loc=0, ...)`, the figure name and campaign written at the header
right with `ax.set_title(..., loc="right")`, short ticks on all four sides — and adds the size
scaling and the multi-panel header axis that those single-panel helpers do not have.

The experiment label is this folder's own member of the family the repo already uses
("ETL ETROC Test Beam" in `plotting_basic.py`, "ETL ETROC IRRAD" in the IRRAD figures).

Multi-panel figures carry the header on a zero-height axis spanning the figure width, the same
mechanism the walkthrough figures use, so the panels themselves keep their own captions.
`apply_style(scale)` sizes the panel text; the header takes its own, larger scale so it stays
prominent on a wide multi-panel canvas.
"""
import matplotlib as mpl
import mplhep as hep

EXP_TEXT = "ETL ETROC Charge Injection"
CAMPAIGN = "CERN climate chamber, January 2024"

INK = "#000000"
GRID = "#c9c9c9"
SURFACE = "#ffffff"

# reference point sizes: the repo's single-panel (11, 10) CMS figure
_REF = dict(label=25.0, tick=17.0, header=18.0, title=16.0,
            legend=16.0, legend_title=15.0, ann=14.0)


def sizes(scale=1.0):
    return {k: round(v * scale, 1) for k, v in _REF.items()}


def apply_style(scale=1.0):
    """CMS house style with the repo's font/tick choices scaled to the panel size."""
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
        "axes.grid": True,
        "lines.linewidth": 2.0,
        "figure.dpi": 110, "savefig.dpi": 150,
        "axes.formatter.useoffset": False,
        "axes.formatter.use_mathtext": False,
        "text.color": INK,
    })
    return s


def cms_header(ax, name, subject=None, tag=None, scale=1.0, pad=None, campaign=CAMPAIGN):
    """Experiment label left, figure name (+ conditions / campaign) right.

    The right block is stacked one item per line — name, then the conditions, then the campaign —
    rather than run together on one line, so the longest line stays short enough that the right
    block never runs into the experiment label on a narrow canvas.
    """
    s = sizes(scale)
    hep.cms.text(loc=0, ax=ax, text=EXP_TEXT, fontsize=s["header"])
    lines = [name if subject is None else f"{name} · {subject}"]
    if tag:
        lines.append(tag)
    lines.append(campaign)
    kw = {} if pad is None else {"pad": pad}
    ax.set_title("\n".join(lines), loc="right", size=s["title"], color=INK, **kw)
    return s


def header(fig, name, tag=None, scale=1.0, y=1.012, pad=6):
    """The repo header on a zero-height axis spanning the figure width.

    Call AFTER any tight_layout so the header is not folded into the panel packing; the axis
    sits just above the figure and `bbox_inches="tight"` grows the saved canvas to include it.
    """
    hax = fig.add_axes([0.0, y, 1.0, 1e-6])
    hax.set_xticks([])
    hax.set_yticks([])
    hax.patch.set_visible(False)
    hax.grid(False)
    for sp in hax.spines.values():
        sp.set_visible(False)
    cms_header(hax, name, tag=tag, scale=scale, pad=pad)
    return hax
