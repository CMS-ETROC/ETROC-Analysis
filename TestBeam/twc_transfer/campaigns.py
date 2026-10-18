"""Campaign facts of the TWC transfer study: where each campaign's step-10 track files are, where
the study's outputs are read from and where a run of this package writes.

The only paths outside the repository live here, each behind an environment variable:

  ETROC_TWC_ROOT_<CAMPAIGN>   the step-10 analysis root of a campaign (MAR, JUL, DESY, AUG);
                              the track files are <root>/tracks_<run>/<combo>/time/track_*.parquet
  ETROC_TWC_TRANSFER_ROOT     the read root: the study's outputs (coefficient bank, cells, tables),
                              read-only here; the same variable and default as etroc_plots uses
  ETROC_TWC_OUT               the write root. It has no default: write_root() refuses to run
                              without it, and refuses the read root and every data root

JOB_LISTS names the study's seven job lists with the output root each one wrote into, and
CELL_VERSIONS the cell code each wave ran and whether cell.py reproduces it.

This module does not import the pipeline core.
"""
import os

# per-triple event cap (load_events, the leave-pair-out sources, the bank)
N_CAP = 6000
PS_PER_NS = 1000.0

# step-10 analysis roots (defaults: the campaigns' EOS trees). DESY August 2026 is the v2
# re-analysis tree.
_DEFAULT_ROOTS = {
    "mar": "/eos/user/m/musafdar/CERN_IRRAD_Mar2026/analysis",
    "jul": "/eos/user/m/musafdar/CERN_IRRAD_Jul2026/analysis",
    "desy": "/eos/user/m/musafdar/DESY_May2026/analysis",
    "aug": "/eos/user/m/musafdar/DESY_Aug2026/analysis_v2",
}
CAMPAIGNS = {c: (os.environ.get("ETROC_TWC_ROOT_%s" % c.upper()) or root)
             for c, root in _DEFAULT_ROOTS.items()}

ROLE_INIT = {"t": "trig", "d": "dut", "r": "ref", "e": "extra"}
NAME_ORDER = ("trig", "dut", "ref", "extra")   # the order step 8 writes into file names
ALL_ROLES = ("dut", "ref", "trig", "extra")

# The study's live output tree: bank/, cells*/, tables/, tier0/. Read, never written.
_DEFAULT_TRANSFER_ROOT = "/eos/user/m/musafdar/DESY_May2026/analysis/twc_transfer"
_RERUN_SEEDED_ROOT = "/eos/user/m/musafdar/DESY_May2026/analysis/twc_transfer_rerun_seeded_20260908"

# The seven job lists the study's cells ran from (copies in jobs/), in the order they ran: a
# later list that names a cell again redefines it. Each comes with the output root its jobs
# wrote into, the ${TT_OUT} of that run. cells_stage2_held.tsv was never submitted itself; its
# six jobs ran from cells_stage3.tsv. cells_rerun.tsv, the seeded rerun of the cells whose
# verdict sat within the reproducibility jitter, wrote into a separate tree.
JOB_LISTS = (
    ("cells.tsv", _DEFAULT_TRANSFER_ROOT),
    ("cells_stage1.tsv", _DEFAULT_TRANSFER_ROOT),
    ("cells_stage2.tsv", _DEFAULT_TRANSFER_ROOT),
    ("cells_stage2_held.tsv", _DEFAULT_TRANSFER_ROOT),
    ("cells_stage3.tsv", _DEFAULT_TRANSFER_ROOT),
    ("cells_rerun.tsv", _RERUN_SEEDED_ROOT),
    ("cells_guardfix_rerun.tsv", _DEFAULT_TRANSFER_ROOT),
)

# sha1 of the pipeline core files the study's published cells imported (a core last changed on
# 2026-08-24). With ETROC_TWC_CORE pointing at a core of these hashes, this package reproduces
# the study's numbers bit for bit. This repository's core differs: twc_solver streams its
# least-squares design in blocks through a QR factor. Coefficients then agree to about 4e-12
# relative, but a width is not a smooth function of them: width_gmm seeds its fit from the
# residual bytes, so a last-bit change can move a width by the GMM restart jitter (measured on
# one triple: up to 2.85 ps on one placebo pair, below 1e-12 ps on most arms).
CELLS_CORE_SHA1 = {
    "twc_solver.py": "da6caabffece62bb7a01bbf685bcea22dd45de0b",
    "bootstrap.py": "0e0b891ec6261749760955895c9b36655014c58e",
    "io_utils.py": "a737972dc0adea61c94c9da39f875be95eedff32",
}

# The cell code each wave of the study ran (the sha256 prefix of its tt.py, as in the wave's run
# manifest) and whether cell.py reproduces that wave's tables. Every wave imported the core of
# CELLS_CORE_SHA1. cell.py is the guardfix version; the stage-3 version differs from it only by
# the --lpo-fallback flag, whose default ("sibling") is the stage-3 behaviour. A reproducible
# wave's commands, run with ETROC_TWC_CORE at a core of those hashes, give its tables back
# except the ts column. The earlier waves ran an older cell (round 1 also an older column set):
# their commands parse and run here but give the current columns and estimators, a rerun and
# not a reproduction. cells_v2w1.tsv is not among the job lists in jobs/.
# (wave, job list, tt.py sha256 prefix, reproducible)
CELL_VERSIONS = (
    ("round1", "cells.tsv", "258cb1f8", False),
    ("stage1", "cells_stage1.tsv", "05793488", False),
    ("stage2", "cells_stage2.tsv", "05793488", False),
    ("stage3", "cells_stage3.tsv", "698f72f9", True),
    ("rerun_seeded", "cells_rerun.tsv", "698f72f9", True),
    ("guardfix", "cells_guardfix_rerun.tsv", "e83f71c1", True),
    ("v2w1", "cells_v2w1.tsv", "e83f71c1", True),
)


def transfer_root():
    """The read root: ETROC_TWC_TRANSFER_ROOT, else the study's live output tree."""
    return os.environ.get("ETROC_TWC_TRANSFER_ROOT") or _DEFAULT_TRANSFER_ROOT


def _same(a, b):
    return os.path.normpath(os.path.abspath(a)) == os.path.normpath(os.path.abspath(b))


def write_root():
    """The write root, ETROC_TWC_OUT. Raises when it is unset, or when it is the read root or the
    step-10 root of a campaign."""
    out = os.environ.get("ETROC_TWC_OUT")
    if not out:
        raise RuntimeError("ETROC_TWC_OUT is not set. Set it to the folder this run writes into; "
                           "there is no default, so nothing is ever written into the study's "
                           "own output tree (%s) by accident." % transfer_root())
    if _same(out, transfer_root()):
        raise RuntimeError("ETROC_TWC_OUT=%s is the read root (ETROC_TWC_TRANSFER_ROOT); "
                           "write somewhere else." % out)
    for c, root in CAMPAIGNS.items():
        if _same(out, root):
            raise RuntimeError("ETROC_TWC_OUT=%s is the step-10 data root of campaign %r; "
                               "write somewhere else." % (out, c))
    return out
