"""Repository roots.

Only paths inside the repository live here. Where a campaign's data sit is set by its campaign
module (campaigns/), so the helpers carry no data and no personal path.
"""
import os

PKG_ROOT = os.path.dirname(os.path.abspath(__file__))   # .../TestBeam/etroc_plots
REPO_TESTBEAM = os.path.dirname(PKG_ROOT)               # .../TestBeam, where board_configs_yaml lives
