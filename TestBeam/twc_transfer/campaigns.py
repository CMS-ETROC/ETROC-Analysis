"""Campaign facts of the TWC transfer study: where each campaign's step-10 track files are, where
the study's outputs are read from and where a run of this package writes.

The only paths outside the repository live here, each behind an environment variable:

  ETROC_TWC_ROOT_<CAMPAIGN>   the step-10 analysis root of a campaign (MAR, JUL, DESY, AUG);
                              the track files are <root>/tracks_<run>/<combo>/time/track_*.parquet
  ETROC_TWC_TRANSFER_ROOT     the read root: the study's outputs (coefficient bank, cells, tables),
                              read-only here; the same variable and default as etroc_plots uses
  ETROC_TWC_OUT               the write root. It has no default: write_root() refuses to run
                              without it, and refuses the read root and every data root

JOB_LISTS names the study's seven job lists with the output root each one wrote into,
CELL_VERSIONS the cell code each wave ran and whether cell.py reproduces it, and F_CAL the
error inflation of the aggregation per campaign, at the two precisions the study used.

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


# The error inflation f of the aggregation (SPEC 4 errors, 3), per campaign: the ratio of a
# block-bootstrap error to the half-split error, measured by the study's bootstrap on one
# anchor run (the bootstrap is not part of this package; its outputs are tables/f_cal_*.json
# in the read root). The study's aggregation was run with four decimals: the numbers written
# into its drivers for DESY May, March and July, the measured August value formatted "%.4f".
# The round-1 recomputation of the verdicts used the full values. agg --f-cal <campaign> takes
# the four-decimal value, which reproduces the aggregated tables; --f-precision full the other.
# DESY May and August were also aggregated at f = 1.0, and August at the DESY value 0.8153 (its
# pre-registered fallback); those runs pass the number with --f-inflate.
# (four decimals, full value, provenance)
F_CAL = {
    "desy": (0.8153, 0.8152579445510614,
             "bootstrap of desy run26 ref1-trig2-dut3, tables/f_cal_desy_run26.json"),
    "aug": (0.8979, 0.8979111975615606,
            "bootstrap of aug run23 dut0-trig1-ref2, tables/f_cal_aug_run23.json (gate PASS)"),
    "mar": (0.9524, 0.9524,
            "the March H1 run h1_run1 cross-check; recorded to four decimals only"),
    "jul": (0.9259, 0.925913381797006,
            "bootstrap of jul h1_run10 extra0-dut1-ref2, tables/f_cal_jul_h1_run10.json"),
}
_F4 = {c: v[0] for c, v in F_CAL.items()}


# ------------------------------------------------------------------ round 1 (master.py)
# The aggregation tables of round 1 (job list cells.tsv) that its chain reads, a closed list:
# (folder under tables/, campaign group, the f the folder was aggregated with). DESY May and
# August were also aggregated at f = 1.0, and August at the DESY value; a cell's verdict must
# agree across its group's tables (rule R8 of master.py). The f here are the four-decimal
# values (the tables'); the chain's recomputation uses the full values of F_CAL.
ROUND1_AGG_TAGS = (
    ("agg_desy_f0p8153", "desy", _F4["desy"]),
    ("agg_desy_f1p0", "desy", 1.0),
    ("agg_aug_fmeas", "aug", _F4["aug"]),
    ("agg_aug_f0p8153", "aug", _F4["desy"]),
    ("agg_aug_f1p0", "aug", 1.0),
    ("agg_mar_f0p9524", "mar", _F4["mar"]),
    ("agg_jul_f0p9259", "jul", _F4["jul"]),
)
# the table of each group whose verdict is the cell's verdict
ROUND1_PRIMARY_TAG = {"desy": "agg_desy_f0p8153", "aug": "agg_aug_fmeas",
                      "mar": "agg_mar_f0p9524", "jul": "agg_jul_f0p9259"}
# the 40 cells pre-registered for round 1, per level, and the four added before it ran
ROUND1_PREREG = {
    "a": ("a1_twin_26to27", "a2_twin_27to26", "a3_floor_parity_run26", "a5_floor_block_run26"),
    "b": ("b1_30to0", "b2_30to10", "b3_30to20", "b4_30to45", "b5_ctrl_30to30"),
    "bp": ("bp1_off30to60_r44", "bp2_off30to60_r45", "bp3_ctrl_off60to60", "bp4_twin_off60"),
    "c": ("c1_pix2board_run26_0deg", "c2_pix2board_run33_30deg", "c3_pix2board_run44_60deg"),
    "d": ("d1_board2board_run26", "d2_board2board_run33"),
    "e": ("e1_pooled_aug_run23", "e2_pooled_aug_run18", "e3_pooled_aug_run8",
          "e4_pooled_aug_run5", "e5_pooled_may_run26"),
    "f": ("f1_tier0_map",),
    "g": ("g1_h1_7to10", "g2_h1_7to11", "g4_h1_7to19", "g5_h1_11to14", "g6_h1_14to19",
          "g7_f1_2to10", "g8_f1_2to11", "g9_f1_2to19"),
    "h": ("h1_mar19_to_jul3", "h2_mar19_to_jul4", "h3_mar19_to_jul5", "h4_mar17_to_jul10",
          "h5_f1_19to5", "h6_f1_19to6", "h7_f1_19to7", "h8_f1_17to5"),
}
ROUND1_ADDED = ("a6_twin_24to25", "a7_twin_20to21", "g10_f1_11to14", "g11_f1_14to19")
# target runs too thin for a headline claim, from the pre-registration (rule R7): a PASS on
# one of them is reported as SUPPORTING
THIN_RUNS = frozenset({("aug", "run5"), ("aug", "run8"), ("aug", "run18"), ("aug", "run23"),
                       ("jul", "h1_run3")})
LEVEL_TITLES = {"a": "(a) stability floor - same pixel, twin runs / in-run splits",
                "b": "(b) angle ladder, aligned holder, board-level",
                "bp": "(b') angle 30->60, offset holder, board-level",
                "c": "(c) pixel -> board, same run",
                "d": "(d) board -> board, same run",
                "e": "(e) pooled board TWC vs per-pixel",
                "f": "(f) Tier-0 cross grid (map only)",
                "g": "(g) across irradiation steps (March)",
                "h": "(h) across time, March -> July, same chips"}


# ----------------------------------------------------------------- axis tables (axes.py)
# The aggregation tables behind each axis table of the study, per axis: (folder under
# tables/, the label its rows carry as agg_tag, the f it was aggregated with, whether that f is
# the campaign's primary one). The eight axes of the second wave (stages 1 and 2) come first.
# Axis E holds the August cells (three f) and the July cells. Axis A: stage 3 reused the group
# name A_desy, so tables/agg_A_desy_* hold only its cells A8 and A9; the A1-A7 rows of the
# axis table are in agg_A_r2_desy_*, aggregated again from the stage-2 cells (by hand, with no
# recorded file list), and carry the label A_desy_*.
AXIS_TAGS = {
    "D": (("agg_D_mar_f0p9524", "D_mar_f0p9524", _F4["mar"], True),),
    "E": (("agg_E_aug_fmeas", "E_aug_fmeas", _F4["aug"], True),
          ("agg_E_aug_f0p8153", "E_aug_f0p8153", _F4["desy"], False),
          ("agg_E_aug_f1p0", "E_aug_f1p0", 1.0, False),
          ("agg_E_jul_f0p9259", "E_jul_f0p9259", _F4["jul"], True)),
    "I": (("agg_I_mar_f0p9524", "I_mar_f0p9524", _F4["mar"], True),),
    "J": (("agg_J_jul_f0p9259", "J_jul_f0p9259", _F4["jul"], True),),
    "A": (("agg_A_r2_desy_f0p8153", "A_desy_f0p8153", _F4["desy"], True),
          ("agg_A_r2_desy_f1p0", "A_desy_f1p0", 1.0, False)),
    "B": (("agg_B_desy_f0p8153", "B_desy_f0p8153", _F4["desy"], True),
          ("agg_B_desy_f1p0", "B_desy_f1p0", 1.0, False)),
    "C": (("agg_C_desy_f0p8153", "C_desy_f0p8153", _F4["desy"], True),
          ("agg_C_desy_f1p0", "C_desy_f1p0", 1.0, False)),
    "K": (("agg_K_desy_f0p8153", "K_desy_f0p8153", _F4["desy"], True),
          ("agg_K_desy_f1p0", "K_desy_f1p0", 1.0, False)),
    # axis F (stage 3): the xfer_lin re-score of eleven round-1 cells
    "F": (("agg_F_desy_f0p8153", "F_desy_f0p8153", _F4["desy"], True),
          ("agg_F_desy_f1p0", "F_desy_f1p0", 1.0, False),
          ("agg_F_mar_f0p9524", "F_mar_f0p9524", _F4["mar"], True),
          ("agg_F_jul_f0p9259", "F_jul_f0p9259", _F4["jul"], True)),
}
# The guardfix rerun (cells_guardfix_rerun.tsv, --lpo-fallback own) of eleven cells of axes E,
# J and K: their tables, whose rows replace those cells' rows of the axis table (axes.py).
GUARDFIX_AXIS_TAGS = {
    "E": (("agg_E_jul_guardfix_f0p9259", "E_jul_guardfix_f0p9259", _F4["jul"], True),),
    "J": (("agg_J_jul_guardfix_f0p9259", "J_jul_guardfix_f0p9259", _F4["jul"], True),),
    "K": (("agg_K_desy_guardfix_f0p8153", "K_desy_guardfix_f0p8153", _F4["desy"], True),
          ("agg_K_desy_guardfix_f1p0", "K_desy_guardfix_f1p0", 1.0, False)),
}
# The cells of axis F and what each compares: (label, the round-1 cell it re-scores and how,
# a note). The first word of the second field names the round-1 cell.
AXIS_F_CELLS = (
    ("F1", "c1, run26, all3, mode c, --src-parity-contrast (src-k 5, trkpop)",
     "also carries xfer_srcmatch/xfer_srccross, the only F cell with the parity flag"),
    ("F2", "d1, run26, board-to-board, mode d, role-map dut:ref,ref:trig,trig:dut", ""),
    ("F3", "d2, run33, board-to-board, mode d, role-map dut:ref,ref:trig,trig:dut", ""),
    ("F4", "b1, run26<-run9 (30deg->0deg), mode d", ""),
    ("F5", "b3, run20<-run9 (30deg->20deg), mode d", ""),
    ("F6", "b5, run33<-run9 (ctrl, 30deg->30deg), mode d", ""),
    ("F7", "e5, run26, pooled May, mode e", "pool-ntrip 40, pool-cap 1500, pool-lopo-scope board"),
    ("F8", "g2, March H1 h1_run11<-h1_run7 (7to11), mode a", ""),
    ("F9", "g6, March H1 h1_run19<-h1_run14 (14to19), mode a", ""),
    ("F10", "h2, mar h1_run19 -> jul h1_run4, mode a, role-map dut:ref,ref:dut,trig:trig", ""),
    ("F11", "h4, mar h1_run17 -> jul h1_run10, mode a, role-map dut:ref,ref:dut,trig:trig", ""),
)

# md5 of the study's tables that `master` and `axes` give back byte for byte, by the path the
# port writes them to under its output folder (measured 2026-10-18); and of the universality
# tables etroc_plots.universality.reduce composes from them (its master part copies
# master_cells.csv; its axes part composes the axis tables with the guardfix overlay and
# appends the self_transfer flag; its cells part reads the job lists and summary_F.md).
STUDY_TABLE_MD5 = {
    "master/master_cells.csv": "62cfc93f96be1dcbab6cad770da772fa",
    "results_r2/axis_A.csv": "f54ef48af42e2379c1fe8017f08466ef",
    "results_r2/axis_B.csv": "5103089c07b8d2b6b2551bfc2f90c61a",
    "results_r2/axis_C.csv": "332b93b9bcfd22dbe057d38adfda2eed",
    "results_r2/axis_D.csv": "1cf27fe9d4e66aa94479d159a986d4b3",
    "results_r2/axis_E.csv": "b8d9c6fe9a585f054cf85670aade857e",
    "results_r2/axis_I.csv": "4656d575599dfb9b7bcc68b21ef1a0e0",
    "results_r2/axis_J.csv": "281926f5e3c313fb0d08e55859de78e7",
    "results_r2/axis_K.csv": "d5b4825caf1f5483af1a69689759e9f3",
    "results_r3/axis_F.csv": "44c661a62fdbf95b05dd7d67cd86cb74",
    "results_r3/axis_F_long.csv": "8cead873b7cc767f21b356cbd74bd136",
    "results_r3/guardfix/axis_E_gf.csv": "2f0a77c5a5c58ffca95bb7fa2c24645f",
    "results_r3/guardfix/axis_J_gf.csv": "7e4758c43efdd7d51f079c06f9ed8681",
    "results_r3/guardfix/axis_K_gf.csv": "db411e3e2daa8a45eeb0dafb84a55361",
}
PUBLISHED_MD5 = {
    "univ_master_cells.csv": "62cfc93f96be1dcbab6cad770da772fa",
    "univ_axis_r2.csv": "15c688966ce406593454391e9d6120ff",
    "univ_axis_F_wide.csv": "911efb7d856178f24e60a87b5203a755",
    "univ_axis_F_long.csv": "51302099b643b24014141af7d860a059",
    "univ_cells.csv": "2fcde80b93bff79a4d9eb54292546318",
}


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
