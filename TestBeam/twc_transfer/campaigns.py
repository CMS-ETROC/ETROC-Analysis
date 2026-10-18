"""Campaign facts of the TWC transfer study: where each campaign's step-10 track files are, where
the study's outputs are read from and where a run of this package writes.

The only paths outside the repository live here, each behind an environment variable:

  ETROC_TWC_ROOT_<CAMPAIGN>   the step-10 analysis root of a campaign (MAR, JUL, DESY, AUG);
                              the track files are <root>/tracks_<run>/<combo>/time/track_*.parquet
  ETROC_TWC_TRANSFER_ROOT     the read root: the study's outputs (coefficient bank, cells, tables),
                              read-only here; the same variable and default as etroc_plots uses
  ETROC_TWC_OUT               the write root. It has no default: write_root() refuses to run
                              without it, and refuses it when it is, or lies inside, one of
                              protected_roots() (the read root, the study's output trees, the
                              data roots, this checkout; real paths compared); output_path()
                              applies the same rule to every output path, --out included
  ETROC_TWC_CORE              the pipeline core directory (read by _core, not here; default this
                              repository's TestBeam/condor_at_lxplus/core). A core of the hashes
                              of CELLS_CORE_SHA1 reproduces the study's cells; another reruns them

JOB_LISTS names the study's seven job lists with the output root each one wrote into; WAVES
the cell code each wave ran, whether cell.py reproduces it and which published tables it
feeds (CELL_VERSIONS is its short form); TABLE_VERSIONS the tables this package gives back
byte for byte and the one it corrects (axis_F.csv; the study's version is AXIS_F_AS_PUBLISHED,
the universality table composed from that UNIV_AXIS_F_WIDE_AS_PUBLISHED), with their md5 and
command (versions_markdown() renders both for the README);
F_CAL the error inflation of the aggregation per campaign, at the two precisions the study
used (March's is known to four decimals only), and AGG_F the f each campaign's groups are aggregated with; THIN_RULE and THIN_RUNS the
pre-registered thin rule; LEVEL_TITLES the round-1 levels; AXIS_TAGS and CELL_DESC the axis
tables and what their cells compare.

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
# Per wave: (wave, its name in words, job list, tt.py sha256 prefix, cells reproducible here,
# what this package does with the wave's cells, the wave's aggregation tags, what this package
# does with them, the published tables (PUBLISHED_MD5) the wave feeds). "Compared" names what
# was run against the study's own tables (the aggregation tags through the plan of `jobs --agg`
# on the study's cell tables); "not compared" rests on the code being the same.
WAVES = (
    ("round1", "round 1", "cells.tsv", "258cb1f8", False,
     "reruns them (older cell, older columns)",
     "agg_desy_*, agg_aug_*, agg_mar_f0p9524, agg_jul_f0p9259",
     "re-derives them (later rules, more columns): agg_desy_f0p8153 changes tost on 2 rows; "
     "master reads them as they are",
     ("univ_master_cells.csv", "univ_cells.csv")),
    ("stage1", "stage 1", "cells_stage1.tsv", "05793488", False,
     "reruns them (older cell)",
     "the groups of cells_stage1.tsv: axes D, E, I, J",
     "reproduces them (all 7 tags compared)",
     ("univ_axis_r2.csv", "univ_cells.csv")),
    ("stage2", "stage 2", "cells_stage2.tsv", "05793488", False,
     "reruns them (older cell)",
     "the groups of cells_stage2.tsv: axes A, B, C, K",
     "reproduces them (all 8 tags compared; its A_desy tags are the study's agg_A_r2_desy_*, "
     "since stage 3 reused the folder name)",
     ("univ_axis_r2.csv", "univ_cells.csv")),
    ("stage3", "stage 3", "cells_stage3.tsv", "698f72f9", True,
     "reproduces them up to the cross-node tolerance (ks_d, ks_p within 3.3e-14; one triple "
     "per cell in modes c, d, e and xfer-lin compared)",
     "the groups of cells_stage3.tsv: A8 and A9, F, G",
     "reproduces them (all 8 tags compared)",
     ("univ_axis_F_long.csv", "univ_axis_F_wide.csv", "univ_cells.csv")),
    ("rerun_seeded", "seeded rerun", "cells_rerun.tsv", "698f72f9", True,
     "not compared (the stage-3 code)",
     "the groups of cells_rerun.tsv, in the rerun's own tree",
     "reproduces them (all 14 tags compared)",
     ("univ_cells.csv",)),
    ("guardfix", "guardfix", "cells_guardfix_rerun.tsv", "e83f71c1", True,
     "reproduces them (the smoke cell compared, 240 rows)",
     "agg_E_jul_guardfix_*, agg_J_jul_guardfix_*, agg_K_desy_guardfix_*",
     "reproduces them (all 4 tags compared)",
     ("univ_axis_r2.csv", "univ_cells.csv")),
    ("v2w1", "v2 wave 1", "cells_v2w1.tsv", "e83f71c1", True,
     "not compared (the guardfix code); the list is not in jobs/",
     "agg_V2_*", "the guardfix code, not compared",
     ()),
)
# (wave, job list, tt.py sha256 prefix, reproducible)
CELL_VERSIONS = tuple((w[0], w[2], w[3], w[4]) for w in WAVES)


# The error inflation f of the aggregation (SPEC 4 errors, 3), per campaign: the ratio of a
# block-bootstrap error to the half-split error, measured by the study's bootstrap on one
# anchor run (the bootstrap is not part of this package; its outputs are tables/f_cal_*.json
# in the read root). The study's aggregation was run with four decimals: the numbers written
# into its drivers for DESY May, March and July, the measured August value formatted "%.4f".
# The round-1 recomputation of the verdicts used the full values. agg --f-cal <campaign> takes
# the four-decimal value, which re-derives the aggregated tables (their f_source then names the
# campaign, so only --f-inflate gives them back byte for byte); --f-precision full the other.
# DESY May and August were also aggregated at f = 1.0, and August at the DESY value 0.8153 (its
# pre-registered fallback); those runs pass the number with --f-inflate.
# (four decimals, full value, provenance)
# Each value is the rms convention of the bootstrap (the rms of the block-bootstrap deltas over
# the rms of the half-split ones) and passes the gate F_GATE; the median convention, also
# written by the bootstrap, is not used.
F_CAL = {
    "desy": (0.8153, 0.8152579445510614,
             "DESY May run26 ref1-trig2-dut3, 30 rows, rms convention; gate PASS 1.9 % above "
             "its lower edge (median convention 0.8418); tables/f_cal_desy_run26.json"),
    "aug": (0.8979, 0.8979111975615606,
            "August run23 dut0-trig1-ref2, measured before round 1 was aggregated, rms "
            "convention; gate PASS; tables/f_cal_aug_run23.json"),
    "mar": (0.9524, 0.9524,
            "IRRAD March h1_run1 extra0-ref1-dut2, 24 rows, rms convention; gate PASS (median "
            "convention 0.8582); an untouched cross-check, recorded to four decimals only"),
    "jul": (0.9259, 0.925913381797006,
            "IRRAD July h1_run10 extra0-dut1-ref2, 30 rows, rms convention; gate PASS; the median "
            "convention, 0.7449, is below the gate and never used; tables/f_cal_jul_h1_run10.json"),
}
_F4 = {c: v[0] for c, v in F_CAL.items()}
# the licensed range of f (SPEC.md 4): a campaign whose f falls outside it has no TOST verdict
F_GATE = (0.8, 1.25)

# The f each campaign's groups of a job list are aggregated with, as the study's aggregation
# drivers ran them, per campaign ((tag suffix, f), ...): the campaign's own f_cal first; DESY
# May and August also at f = 1.0, because 0.8153 sits at the gate's lower edge and a verdict
# that flips between the two is UNSTABLE, not PASS; August also at the DESY value, its
# pre-registered fallback from before its own f was measured (suffix fmeas: the measured value).
AGG_F = {
    "desy": (("f0p8153", _F4["desy"]), ("f1p0", 1.0)),
    "aug": (("fmeas", _F4["aug"]), ("f0p8153", _F4["desy"]), ("f1p0", 1.0)),
    "mar": (("f0p9524", _F4["mar"]),),
    "jul": (("f0p9259", _F4["jul"]),),
}


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
# The pre-registered thin rule, on the event counts of a (run, combo)'s triples (the nevt of
# `index`). THICK, per-pixel cells allowed, iff the median count is at least median_min AND at
# least n1500_per_ntrip x ntrip triples hold 1500 events or more (90 for a 30-triple cell).
# Otherwise THIN: pooled (mode e) and board-level cells only, and a thin per-pixel target is
# SUPPORTING, never headline (THIN_RUNS: the August runs, medians 74 to 116 events, and July
# h1_run3, median 524 with 98 triples at 1500). Fewer than exclude_below triples of 600
# events or more: excluded outright. DESY run32 ref1-trig2-dut3 (median 454) is thin too, and
# the study used it as a source only, so it is not among THIN_RUNS.
THIN_RULE = {"median_min": 600, "n1500_per_ntrip": 3, "exclude_below": 40}


def thin_verdict(median_n, n_ge_600, n_ge_1500, ntrip):
    """THICK, THIN or EXCLUDED for a (run, combo) and a cell of `ntrip` triples (THIN_RULE)."""
    if n_ge_600 < THIN_RULE["exclude_below"]:
        return "EXCLUDED"
    if median_n >= THIN_RULE["median_min"] and n_ge_1500 >= THIN_RULE["n1500_per_ntrip"] * ntrip:
        return "THICK"
    return "THIN"


def thin_table(index, ntrip):
    """The thin rule over an index frame (columns campaign, run, combo, nevt; one row per
    triple, as `index` writes it): one row per (campaign, run, combo) with median_n,
    n_ge_600, n_ge_1500 and verdict."""
    import pandas as pd
    rows = []
    for (camp, run, combo), g in index.groupby(["campaign", "run", "combo"], sort=True):
        n = g["nevt"]
        med, n600, n1500 = float(n.median()), int((n >= 600).sum()), int((n >= 1500).sum())
        rows.append(dict(campaign=camp, run=run, combo=combo, median_n=med, n_ge_600=n600,
                         n_ge_1500=n1500, verdict=thin_verdict(med, n600, n1500, ntrip)))
    return pd.DataFrame(rows, columns=["campaign", "run", "combo", "median_n", "n_ge_600",
                                       "n_ge_1500", "verdict"])


# the round-1 levels (the level column of cells.tsv), as the table of the level cells titles them
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
# The two other axis tables of stage 3, results_r3/axis_A8A9.csv and axis_G.csv (axis G with
# each cell's source cap, src-k and label from cells_stage3.tsv), in the AXIS_TAGS form. No
# published table reads them and axes does not build them. A8A9 reads tables/agg_A_desy_*,
# the folder stage 3 reused for its cells A8 and A9.
OTHER_AXIS_TAGS = {
    "A8A9": (("agg_A_desy_f0p8153", "A_desy_f0p8153", _F4["desy"], True),
             ("agg_A_desy_f1p0", "A_desy_f1p0", 1.0, False)),
    "G": (("agg_G_desy_f0p8153", "G_desy_f0p8153", _F4["desy"], True),
          ("agg_G_desy_f1p0", "G_desy_f1p0", 1.0, False)),
}

# What the cells of each axis compare, per axis: (label, the comparison, a note), as the study's
# axis summaries list them. For axis F the first word of the comparison names the round-1 cell
# it re-scores; its rows are the cell table of summary_F.md (AXIS_F_CELLS).
CELL_DESC = {
    "D": (
        ("D1/D2", "h1_run12 <-> h1_run14",
         "March H1, fluence 9e14, bias 350V<->410V, --arms ext, 996 common triples"),
        ("D3/D4", "h1_run17 <-> h1_run19",
         "March H1, fluence 1.5e15, bias 400V<->470V, --arms ext, 1041 common triples"),
        ("D5/D6", "f1_run12 <-> f1_run14",
         "March F1, fluence 9e14, bias 270V<->320V, --arms ext, 988 common triples"),
        ("D7/D8", "f1_run17 <-> f1_run19",
         "March F1, fluence 1.5e15, bias 265V<->350V, --arms ext, 1070 common triples"),
    ),
    "E": (
        ("E1/E2", "h1 offset 20<->10, early (41/51 h cooling down)",
         "July H1 3.5e15, HV 530/550/550/550 held, 650 common triples"),
        ("E3", "h1 offset 20<->10, late (141/152 h cooling down)",
         "July H1 3.5e15, 250 common triples"),
        ("E4/E5", "h1 RFSel 2<->1 (141/103 h cooling down)", "July H1 3.5e15, 734 common triples"),
        ("E6", "h1_run52 -> h1_run53 (12-22h apart)",
         "July H1 identical-settings floor, 966 common triples"),
        ("E7/E8", "f1 offset 20<->10, early", "July F1 3.5e15"),
        ("E9", "f1 offset 15->10, late", "July F1 3.5e15"),
        ("E10/E11", "f1 RFSel 2<->1", "July F1 3.5e15"),
        ("E12", "f1 floor", "July F1 identical-settings floor"),
        ("E13/E14", "aug 19->21 / 18->23 (offset+RFSel together)",
         "August, RFSel 3 AND low offset 6-10 move together - measures a configuration, not a "
         "knob; THIN (median 82-116 evt/triple), supporting only"),
        ("E15", "aug floor 5->8", "August identical-settings floor, THIN, supporting only"),
    ),
    "I": (
        ("I1", "h1_run7 (190V, ref1-dut2-trig3) <- f1_run2 (240V, extra0-dut1-ref2)",
         "March pre-irrad, mode d + --role-map dut:dut,ref:extra,trig:ref, matched TOT quantiles"),
        ("I2", "f1_run2 <- h1_run7 (reverse of I1)", "March pre-irrad, inverse role-map"),
        ("I3", "h1_run7 board-to-board, cyclic map",
         "same-vendor control - so 'different vendor' is quoted as an excess over "
         "'different board'"),
    ),
    "J": (
        ("J1/J2", "h1_run14 -> h1_run51, 0.0 -> 141.5 h",
         "HV 550/550/550, offset 20, RFSel 2 - identical"),
        ("J3", "h1_run17 -> h1_run51, 27.2 -> 141.5 h", "identical settings"),
        ("J4", "h1_run14 -> h1_run17, 0.0 -> 27.2 h",
         "identical settings - short-baseline control"),
        ("J5/J6", "f1_run32 -> f1_run52, 51.1 -> 174.8 h",
         "HV 480/450/480, offset 10 (NOT the standard 20 - data gap), RFSel 2 - identical"),
        ("J7", "f1_run32 -> f1_run53, 51.1 -> 197.0 h", "identical settings"),
    ),
    "A": (
        ("A1", "P3, run26 (0deg), IH2->IH2, ntrip all (1094)", "930 min est"),
        ("A2", "P3, run33 (30deg), IH2->IH2, ntrip 300", "255 min est"),
        ("A3", "P3, run44 (60deg, offset holder), IH2->IH2, ntrip 300", "255 min est"),
        ("A4", "P4, run26 (0deg), IH5->IH2, ntrip all (1094)", "930 min est"),
        ("A5", "P4, run33 (30deg), IH5->IH2, ntrip 300", "255 min est"),
        ("A6", "P4, run44 (60deg), IH5->IH2, ntrip 300", "255 min est"),
        ("A7", "P4 control, run26 (0deg), IH5->IH3 (ref-to-trig), ntrip 300", "255 min est"),
        ("A8/A9 (HELD)", "P3/P4 + --src-parity-contrast, run26, IH2/IH5->IH2, ntrip 100",
         "held out of stage 2 (cells_stage2_held.tsv) and run in stage 3: CELL_DESC A8A9"),
    ),
    "B": (
        ("B1/B2", "run26 <-> run3 (0deg, 6d18h apart, identical HV, role-mapped)",
         "same-pixel, 932 common triples"),
        ("B3/B4", "P3 / P4, source in run27 (3.1h twin)", "ntrip 100"),
        ("B5/B6", "P3 / P4, source in run3", "ntrip 100"),
    ),
    "C": (
        ("C1-C4", "P3 / P4 both ways, run33 (30deg) <-> run30 (20deg)", ""),
        ("C5/C6", "run30 -> run31 (both 20deg, 4.5h apart)",
         "same-angle control, 789 common triples"),
        ("C7", "run9 -> run20 (best available same-pixel pair)",
         "23 triples at a 250-event floor, ntrip all, data-limited, supporting only"),
    ),
    "K": (
        ("K1", "run13 (45deg, -25C chiller) -> run17 (45deg, -10C chiller)",
         "293 common triples at the 600-event floor"),
        ("K2", "run17 -> run13 (reverse of K1)", "same pair, reverse direction"),
        ("K3", "run17 -> run18 (both -10C, 45deg, consecutive)",
         "same-condition floor; run18 thinner (213 triples)"),
    ),
    "A8A9": (
        ("A8", "P3, run26 (0deg), IH2->IH2, ntrip 100, --mode c --xfer-board dut",
         "930-min-est cell dropped to ntrip=100 for the parity companion"),
        ("A9", "P4, run26 (0deg), IH5->IH2, ntrip 100, --mode cb --xfer-board dut --src-role ref",
         "same as A8, source role ref"),
    ),
    "F": (
        ("F1", "c1, run26, all3, mode c, --src-parity-contrast (src-k 5, trkpop)",
         "also carries xfer_srcmatch/xfer_srccross, the only F cell with the parity flag"),
        ("F2", "d1, run26, board-to-board, mode d, role-map dut:ref,ref:trig,trig:dut", ""),
        ("F3", "d2, run33, board-to-board, mode d, role-map dut:ref,ref:trig,trig:dut", ""),
        ("F4", "b1, run26<-run9 (30deg->0deg), mode d", ""),
        ("F5", "b3, run20<-run9 (30deg->20deg), mode d", ""),
        ("F6", "b5, run33<-run9 (ctrl, 30deg->30deg), mode d", ""),
        ("F7", "e5, run26, pooled May, mode e",
         "pool-ntrip 40, pool-cap 1500, pool-lopo-scope board"),
        ("F8", "g2, March H1 h1_run11<-h1_run7 (7to11), mode a", ""),
        ("F9", "g6, March H1 h1_run19<-h1_run14 (14to19), mode a", ""),
        ("F10", "h2, mar h1_run19 -> jul h1_run4, mode a, role-map dut:ref,ref:dut,trig:trig", ""),
        ("F11", "h4, mar h1_run17 -> jul h1_run10, mode a, role-map dut:ref,ref:dut,trig:trig", ""),
    ),
    "G": (
        ("G_P3_run26_n100..nall",
         "P3, run26, mode c, six rungs: src-cap 100/200/400/800/1600/(nall = follows --cap 6000)",
         "ntrip 60"),
        ("G_P4_run26_n100/nall", "P4, run26, mode cb, two extreme rungs: src-cap 100/(nall)",
         "ntrip 60"),
    ),
}
AXIS_F_CELLS = CELL_DESC["F"]

# md5 of the study's tables that `master` and `axes` give back byte for byte, by the path the
# port writes them to under its output folder (measured 2026-10-18), save axis_F.csv, whose md5
# is that of the corrected table `axes` writes from the archived agg_F tables (measured
# 2026-10-18; the study's is AXIS_F_AS_PUBLISHED); and of the universality
# tables etroc_plots.universality.reduce composes from them (its master part copies
# master_cells.csv; its axes part composes the axis tables with the guardfix overlay and
# appends the self_transfer flag; its cells part reads the job lists and summary_F.md), the
# wide axis-F table composed from the corrected axis_F.csv (published 2026-10-18).
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
    "results_r3/axis_F.csv": "d5425a11ff92f69c4528c57f5936ff0f",
    "results_r3/axis_F_long.csv": "8cead873b7cc767f21b356cbd74bd136",
    "results_r3/guardfix/axis_E_gf.csv": "2f0a77c5a5c58ffca95bb7fa2c24645f",
    "results_r3/guardfix/axis_J_gf.csv": "7e4758c43efdd7d51f079c06f9ed8681",
    "results_r3/guardfix/axis_K_gf.csv": "db411e3e2daa8a45eeb0dafb84a55361",
}
# The study's axis_F.csv: per wide key and arm it kept the last row of the long table, and on 12
# keys of F_desy_f0p8153 (cells F2 and F3, 48 arm entries) that was the self-transfer row
# beside a transfer row (SPEC 10). The port's axis_F.csv shows the transfer row, flags each arm
# with <arm>_self_transfer, and differs from the study's on those 12 rows and in the five flags.
AXIS_F_AS_PUBLISHED = ("44c661a62fdbf95b05dd7d67cd86cb74",
                       "as published (12 keys carry the self-transfer row)")
# univ_axis_F_wide.csv as composed from AXIS_F_AS_PUBLISHED, published until 2026-10-18, when
# the table composed from the port's axis_F.csv (PUBLISHED_MD5) replaced it
UNIV_AXIS_F_WIDE_AS_PUBLISHED = ("911efb7d856178f24e60a87b5203a755",
                                 "published until 2026-10-18 (12 rows carry the self-transfer "
                                 "row)")
PUBLISHED_MD5 = {
    "univ_master_cells.csv": "62cfc93f96be1dcbab6cad770da772fa",
    "univ_axis_r2.csv": "15c688966ce406593454391e9d6120ff",
    "univ_axis_F_wide.csv": "09418b9acd0a698f1c9f3717e25b6b35",
    "univ_axis_F_long.csv": "51302099b643b24014141af7d860a059",
    "univ_cells.csv": "2fcde80b93bff79a4d9eb54292546318",
}

# The tables this package gives back byte for byte (STUDY_TABLE_MD5, PUBLISHED_MD5), each with
# the command that writes it, the waves whose cells it is made from (WAVES) and what it reads.
# `reduce` is etroc_plots.universality.reduce, run on this package's write root (README). The
# cells of axis A were aggregated again outside any wave's driver, and the code version of that
# aggregation is not recorded; this package's aggregation of the stage-2 groups gives those
# tables back byte for byte.
_REDUCE = "python -m etroc_plots.universality.reduce --part %s"
_R2 = ("A", "B", "C", "D", "E", "I", "J", "K")
_R2_WAVE = {"A": "stage2", "B": "stage2", "C": "stage2", "K": "stage2",
            "D": "stage1", "E": "stage1", "I": "stage1", "J": "stage1"}
# (table, command, waves, what it is made from, the aggregation's code version recorded)
TABLE_VERSIONS = (
    ("master/master_cells.csv", "python -m twc_transfer master", ("round1",),
     "the seven round-1 aggregation tables, read as they are, and the round-1 cell tables", True),
) + tuple(
    ("results_r2/axis_%s.csv" % x, "python -m twc_transfer axes", (_R2_WAVE[x],),
     "agg_A_r2_desy_*, the stage-2 cells of axis A aggregated again (the plan of `jobs "
     "--agg` on cells_stage2.tsv gives them back)" if x == "A"
     else ", ".join(t[0] for t in AXIS_TAGS[x]), x != "A")
    for x in _R2
) + (
    ("results_r3/axis_F.csv", "python -m twc_transfer axes", ("stage3",),
     "agg_F_* (the four tags of AXIS_TAGS F); the study's table, md5 %s %s, differs on "
     "those 12 rows and lacks the five <arm>_self_transfer columns; univ_axis_F_wide.csv is "
     "composed from this table since 2026-10-18" % AXIS_F_AS_PUBLISHED,
     True),
    ("results_r3/axis_F_long.csv", "python -m twc_transfer axes", ("stage3",),
     "agg_F_* (the four tags of AXIS_TAGS F)", True),
) + tuple(
    ("results_r3/guardfix/axis_%s_gf.csv" % x, "python -m twc_transfer axes --guardfix",
     ("guardfix",), ", ".join(t[0] for t in GUARDFIX_AXIS_TAGS[x]), True)
    for x in sorted(GUARDFIX_AXIS_TAGS)
) + (
    ("univ_master_cells.csv", _REDUCE % "master", ("round1",),
     "a copy of master_cells.csv", True),
    ("univ_axis_r2.csv", _REDUCE % "axes", ("stage1", "stage2", "guardfix"),
     "the eight results_r2 axis tables, the guardfix rows replacing those of their 11 cells, "
     "and the self_transfer flag from the aggregation tables", True),
    ("univ_axis_F_long.csv", _REDUCE % "axes", ("stage3",),
     "results_r3/axis_F_long.csv and the self_transfer flag", True),
    ("univ_axis_F_wide.csv", _REDUCE % "axes", ("stage3",),
     "this package's results_r3/axis_F.csv (above) and the self_transfer flag, checked "
     "against the table's own; composed from the study's axis_F.csv it was md5 %s, %s"
     % UNIV_AXIS_F_WIDE_AS_PUBLISHED, True),
    ("univ_cells.csv", _REDUCE % "cells",
     ("round1", "stage1", "stage2", "stage3", "rerun_seeded", "guardfix"),
     "the seven job lists and the cell table of summary_F.md", True),
)


def table_md5(table):
    """The md5 of a table of TABLE_VERSIONS (STUDY_TABLE_MD5, else PUBLISHED_MD5)."""
    return STUDY_TABLE_MD5.get(table) or PUBLISHED_MD5[table]


def versions_markdown():
    """The two version tables of the README, rendered from WAVES and TABLE_VERSIONS."""
    sha = {w[0]: w[3] for w in WAVES}
    title = {w[0]: w[1] for w in WAVES}
    lines = ["| wave | job list | tt.py | cells here | aggregation tags | aggregation here "
             "| feeds |", "|---|---|---|---|---|---|---|"]
    for w, name, jl, code, _, cells, tags, agg, feeds in WAVES:
        lines.append("| %s | `%s` | %s | %s | %s | %s | %s |"
                     % (name, jl, code, cells, tags, agg,
                        ", ".join("`%s`" % f for f in feeds) or "none"))
    lines += ["", "| table | md5 | command | waves (tt.py) | made from |",
              "|---|---|---|---|---|"]
    for table, cmd, waves, made_from, recorded in TABLE_VERSIONS:
        codes = ", ".join("%s (%s)" % (title[w], sha[w]) for w in waves)
        if not recorded:
            codes += "; the aggregation's code version is not recorded"
        lines.append("| `%s` | `%s` | `%s` | %s | %s |"
                     % (table, table_md5(table), cmd, codes, made_from))
    return "\n".join(lines) + "\n"


def transfer_root():
    """The read root: ETROC_TWC_TRANSFER_ROOT, else the study's live output tree."""
    return os.environ.get("ETROC_TWC_TRANSFER_ROOT") or _DEFAULT_TRANSFER_ROOT


# the repository this package is part of
_CHECKOUT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def inside(path, root):
    """Whether `path` is `root` or lies inside it, both taken by their real paths (symbolic
    links resolved, so /eos/user/m/x and /eos/home-m/x are one folder)."""
    path, root = os.path.realpath(path), os.path.realpath(root)
    return os.path.commonpath([path, root]) == root


def protected_roots():
    """[(what, root)] that nothing of this package writes into: the read root, the output trees
    of the study's job lists (JOB_LISTS, the seeded rerun's included), the step-10 data root of
    every campaign and this package's checkout."""
    roots = [("the read root (ETROC_TWC_TRANSFER_ROOT)", transfer_root())]
    roots += [("an output tree of the study's job lists (campaigns.JOB_LISTS)", r)
              for r in (_DEFAULT_TRANSFER_ROOT, _RERUN_SEEDED_ROOT)]
    roots += [("the step-10 data root of campaign %r" % c, r) for c, r in CAMPAIGNS.items()]
    roots.append(("this package's checkout", _CHECKOUT))
    return roots


def refuse_inside(path, roots, name=None):
    """Raise ValueError when `path` is, or lies inside, one of `roots` ([(what, root)]); the
    message calls the path `name` (default: the path itself)."""
    name = name or path
    for what, root in roots:
        if inside(path, root):
            same = os.path.realpath(path) == os.path.realpath(root)
            raise ValueError("%s %s %s (%s); write somewhere else."
                             % (name, "is" if same else "lies inside", what, root))


def write_root():
    """The write root, ETROC_TWC_OUT, made absolute. Raises RuntimeError when it is unset, or
    when it is or lies inside one of protected_roots()."""
    out = os.environ.get("ETROC_TWC_OUT")
    if not out:
        raise RuntimeError("ETROC_TWC_OUT is not set. Set it to the folder this run writes into; "
                           "there is no default, so nothing is ever written into the study's "
                           "own output tree (%s) by accident." % transfer_root())
    try:
        refuse_inside(out, protected_roots(), "ETROC_TWC_OUT=%s" % out)
    except ValueError as e:
        raise RuntimeError(str(e))
    return os.path.abspath(out)


def output_path(out, root, *default):
    """The file or folder a writing command writes: `out` (its --out) when given, a relative one
    taken in the write root `root`, else os.path.join(root, *default). Exits (SystemExit, as the
    commands do) when it is or lies inside one of protected_roots(), before anything is made or
    removed."""
    path = out or os.path.join(root, *default)
    if not os.path.isabs(path):
        path = os.path.join(root, path)
    path = os.path.abspath(path)
    try:
        refuse_inside(path, protected_roots(), "the output %s" % path)
    except ValueError as e:
        raise SystemExit("twc_transfer: %s" % e)
    return path
