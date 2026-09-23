"""Saved style for the V_gl-vs-fluence figures.

These figures are drawn at paper scale: 9.5 pt text, inward ticks on all four sides, and
the Okabe-Ito colourblind-safe palette. That is a different visual system from `_ivstyle`, which
styles the IV-curve figures (14x12 in, ~20 pt), so the two are kept apart on purpose:
importing the wrong one is then a visible mistake rather than a silent one.
"""
import matplotlib.pyplot as plt

# Okabe-Ito, by board POSITION rather than board name: H1 and F1 use these same four colours in
# this same order, each keyed by its own board names.
PALETTE = ('#0072B2', '#D55E00', '#009E73', '#CC79A7')

# Horizontal offsets that fan a telescope's four boards apart at a shared fluence, same order.
XOFFSETS = (-0.054, -0.018, 0.018, 0.054)

RCPARAMS = {
    "font.family": "sans-serif",
    "font.size": 9.5,
    "axes.linewidth": 0.9,
    "axes.labelsize": 10.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 2.8,
    "ytick.minor.size": 2.8,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "0.7",
    "savefig.dpi": 300,
}


def apply():
    """Apply the V_gl figure style to the current matplotlib session."""
    plt.rcParams.update(RCPARAMS)


def by_board(boards, values=PALETTE):
    """Map a campaign's board names onto the positional palette (or any positional sequence)."""
    if len(boards) > len(values):
        raise ValueError("%d boards but only %d values" % (len(boards), len(values)))
    return {b: values[i] for i, b in enumerate(boards)}
