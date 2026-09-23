"""Shared plot style: every figure is built from these constants."""

import mplhep as hep

hep.style.use(hep.style.CMS)

# ---------------------------------------------------------------- style block
FIGSIZE_IV    = (14, 12)     # single-panel IV
FIGSIZE_TIME  = (16, 12)     # time series (one or two stacked panels)
FIGSIZE_VGL   = (21, 7)      # three-panel V_gl / K-factor
PANEL_W       = 16           # per-channel panel view: width
PANEL_H       = 4.0          # per-channel panel view: height per channel
DPI_SCREEN    = 150
DPI_SAVE      = 300

MARKEVERY_N   = 50           # target number of markers along a dense line

FS_CMS        = 22           # "CMS  ETL ETROC"
FS_TITLE      = 22           # right-aligned run title
FS_LEGEND     = 20           # legend body and legend title
FS_MINOR      = 'xx-small'   # minor tick labels

# CMS rcParams are sized for one big figure; multi-panel grids need smaller text
FS_GRID_LABEL = 17           # axis labels inside a grid
FS_GRID_TICK  = 13           # tick labels inside a grid
FS_GRID_TAG   = 16           # per-panel device tag
FS_GRID_LEG   = 15           # legend inside a grid

MARKERSIZE    = 6
ALPHA_LINE    = 0.8
ALPHA_GRID    = 0.5
LW_LEGEND_KEY = 4            # swatch thickness in the "Conditions" legend

EXPERIMENT    = "ETL ETROC"  # subtitle after the bold CMS label

# twin-axis panel view: colour identifies the quantity, not the dataset
COLOR_V       = "tab:blue"
COLOR_I       = "tab:red"

MARKERS    = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']   # marker  -> channel
LINESTYLES = ['-', '--', ':', '-.']                     # style   -> channel
# colour -> dataset. 16 distinct hues so damage campaigns with many conditions
# don't wrap; ordered so early entries stay maximally separated. If a list ever
# exceeds this, dataset i additionally falls back to a dashed variant (see
# _scan_style) rather than silently repeating a solid colour.
COLORS     = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd',
              '#ff7f0e', '#8c564b', '#17becf', '#e377c2',
              '#bcbd22', '#7f7f7f', '#000075', '#800000',
              '#f58231', '#3cb44b', '#911eb4', '#42d4f4']
