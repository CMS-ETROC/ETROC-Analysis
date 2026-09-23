"""Repository roots for the IV helpers.

Only paths inside the repository live here. Where a campaign's data sit is set by its campaign
module (iv/campaigns/), so the helpers carry no data and no personal path.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)                 # .../etroc_plots
REPO_TESTBEAM = os.path.dirname(PKG_ROOT)        # .../TestBeam, where board_configs_yaml lives
