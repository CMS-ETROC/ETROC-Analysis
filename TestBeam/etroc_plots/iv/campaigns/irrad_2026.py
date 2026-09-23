"""IRRAD 2026 (CERN March + July) scan catalogue.

Everything the IV code needs to know about this campaign: which file is which scan, which runs
sit at which fluence, the chip maps, and where the inputs live. No data sit in the repository:
raw logs and scans are read from EOS, and the tables the figures read (binned scans, reduced
current logs, run lists, caches) from one folder, INPUTS below. irrad_2026_inputs.md lists those
files and where each came from; irrad_2026_inputs.md5 holds their checksums. Paths outside the
repository can be redirected with environment variables (ETROC_*).

To add a campaign: copy this file, change the catalogue and INPUTS, list your input files with
their checksums in <campaign>_inputs.md5 (md5sum format), point ETROC_IV_CAMPAIGN at it.
"""
import os

from .._paths import REPO_TESTBEAM

# The tables the figures read, 31 MB, kept out of the repository in one folder. Set
# ETROC_IV_INPUTS to use a copy of it.
INPUTS = (os.environ.get("ETROC_IV_INPUTS")
          or "/eos/user/m/musafdar/ETROC_plot_inputs/irrad_2026")
INPUTS_JULY = os.path.join(INPUTS, "july")
INPUTS_MARCH = os.path.join(INPUTS, "march")
INPUTS_VGL = os.path.join(INPUTS, "vgl")
# every file in INPUTS with its md5 (md5sum format); iv_data.check_inputs() compares them
INPUTS_MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "irrad_2026_inputs.md5")

# The raw March tree (IV scans and slow-control logs in week1/, week2/ and laptop_mirror_15e14/),
# read in place. Every raw path below is built from EOS_ROOT: set ETROC_IV_EOS_MARCH to use a copy
# of the whole tree. ETROC_IV_EOS_MARCH_WEEK2 and ETROC_IV_EOS_MARCH_15E14 move one subfolder,
# ETROC_PREIRRAD_LOG one file. irrad_2026_inputs.md lists every variable.
EOS_ROOT = (os.environ.get("ETROC_IV_EOS_MARCH")
            or "/eos/user/m/musafdar/CERN_IRRAD_Mar2026/IVCurves")


# ============================================================================================
# from iv_data.py
# ============================================================================================

MARCH_EOS_WEEK2 = (os.environ.get("ETROC_IV_EOS_MARCH_WEEK2")
                   or os.path.join(EOS_ROOT, "week2"))

MARCH_EOS_15E14 = (os.environ.get("ETROC_IV_EOS_MARCH_15E14")
                   or os.path.join(EOS_ROOT, "laptop_mirror_15e14"))

# HV channel 0-3 -> chip, per telescope (brief "Chips" section / talk_style.TELESCOPE_CHIPS)
TEL_CHIPS = {"h1": ["IH7", "IH11", "IH12", "IH13"], "f1": ["LF10", "LF12", "LF17", "LF18"]}

LINE1_PARKED = "CERN IRRAD 2026, parked, no beam"   # every IV-curve header override (brief)

RAD_STOP_UTC = {   # first post-irradiation IV scan for July steps is an upper bound (brief)
    3e14: "2026-03-16 17:07:00", 9e14: "2026-03-19 17:01:00", 1.5e15: "2026-03-22 22:30:00",
    2e15: "2026-07-20 18:37:00", 3.5e15: "2026-07-25 16:09:00",
}

# ---------------------------------------------------------------------------- March scan catalogue
# Explicit paths, not a naming rule: week2 stems carry "_interpolated_binned_iv_data.csv", the
# 1.5e15 H1 quick "last before next step" scan exists only as the "_overlay" csv, and the F1
# 1.5e15 scans have no binned CSV anywhere -- they are produced locally by bin_f1_15e14.py.
def _week2(stem):
    return os.path.join(MARCH_EOS_WEEK2, stem + "_interpolated_binned_iv_data.csv")

def _h1_15e14(stem, overlay=False):
    name = stem + "_interpolated_binned_iv_data" + ("_overlay.csv" if overlay else ".csv")
    return os.path.join(MARCH_EOS_15E14, "H1", name)

def _f1_local(stem):
    return os.path.join(INPUTS_MARCH, stem + "_binned_iv_data.csv")

MARCH_SCANS = {
    "h1": {
        0.0: {"fine_ra": _week2("fine_iv_H1_m25c_17cm_03162026_0313"),
              "quick_last": _week2("quick_iv_H1_m25c_17cm_03152026_1457")},
        3e14: {"fine_ra": _week2("fine_iv_H1_m25c_17cm_3e14_03162026_1843"),
               "fine_2d": _week2("fine_iv_H1_m25c_17cm_3e14_03182026_1654"),
               "quick_last": _week2("quick_iv_H1_m25c_17cm_3e14_03172026_2352")},
        9e14: {"fine_ra": _week2("fine_iv_H1_m25c_17cm_9e14_03192026_1933"),
               "fine_2d": _week2("fine_iv_H1_m25c_17cm_9e14_03212026_1857"),
               "quick_last": _week2("quick_iv_H1_m25c_17cm_9e14_03212026_1845")},
        1.5e15: {"fine_ra": _h1_15e14("fine_iv_H1_m25c_17cm_15e14_03232026_0037"),
                 "fine_2d": _h1_15e14("fine_iv_H1_m25c_17cm_15e14_03252026_0617"),
                 "quick_last": _h1_15e14("quick_iv_H1_m25c_17cm_15e14_03252026_0605", overlay=True)},
    },
    "f1": {
        0.0: {"fine_ra": _week2("fine_iv_F1_m25C_17cm_03162026_0316"),
              "quick_last": _week2("quick_iv_F1_m25C_17cm_03152026_1458")},
        3e14: {"fine_ra": _week2("fine_iv_F1_m25C_17cm_3e14_03162026_1843"),
               "fine_2d": _week2("fine_iv_F1_m25C_17cm_3e14_03182026_1654"),
               "quick_last": _week2("quick_iv_F1_m25C_17cm_3e14_03172026_2352")},
        9e14: {"fine_ra": _week2("fine_iv_F1_m25C_17cm_9e14_03192026_1933"),
               "fine_2d": _week2("fine_iv_F1_m25C_17cm_9e14_03212026_1858"),
               "quick_last": _week2("quick_iv_F1_m25C_17cm_9e14_03212026_1838")},
        1.5e15: {"fine_ra": _f1_local("fine_iv_F1_m25C_17cm_15e14_03232026_0037"),
                 "fine_2d": _f1_local("fine_iv_F1_m25C_17cm_15e14_03252026_0752"),
                 "quick_last": _f1_local("quick_iv_F1_m25C_17cm_15e14_03252026_0605")},
    },
}

# July fine (0.1 V) legacy slow-control logs, 2026-07-16, four months after the March 1.5e15
# step -- the campaign notebook's own after-rest source for vgl_points.csv (see load_july_fine).
# Each raw file spans many hours (the channel sits idle near 0 V for most of it); "start" is the
# scan's own timestamp from its filename, minus a two-minute buffer, so the multi-hour idle
# period and any earlier scan's tail (F1 ch0 shows a leftover ~350 V reading from a prior scan
# ending well before this window) never enter the bin.
INPUTS_JULY_FINE = os.path.join(INPUTS_JULY, "fine_legacy_20260716")

JULY_FINE_SCANS = {
    "4mo_1p5e15_fine": {
        "h1": dict(path=os.path.join(INPUTS_JULY_FINE,
                                     "H1_202607160020_fineIV_interpolated_claudefixed.csv"),
                   start="2026-07-16 00:18:00"),
        "f1": dict(path=os.path.join(INPUTS_JULY_FINE, "F1_202607160017_fineIV_interpolate.csv"),
                   start="2026-07-16 00:15:00"),
    },
}

# July scan keys, "<TEL>/<scan>" into INPUTS/july/iv_curves.json -- the ones the brief selects.
JULY_SCANS = {
    "4mo_1p5e15": {"h1": "H1/20260717_1517_quick_IVscan", "f1": "F1/20260717_1702_quick_IVscan"},
    "2e15_ra":    {"h1": "H1/20260720_183721_quick_IVscan", "f1": "F1/20260720_183825_quick_IVscan"},
    "2e15_2d":    {"h1": "H1/20260722_011656_fine_IVscan", "f1": "F1/20260722_011719_fine_IVscan"},
    "2e15_3d":    {"h1": "H1/20260723_042703_quick_IVscan", "f1": "F1/20260723_042715_quick_IVscan"},
    "3p5e15_ra":  {"h1": "H1/20260725_160827_quick_IVscan", "f1": "F1/20260725_160837_quick_IVscan"},
    "3p5e15_8h":  {"h1": "H1/20260726_040730_quick_IVscan", "f1": "F1/20260726_040758_quick_IVscan"},
    "3p5e15_1d":  {"h1": "H1/20260726_123407_quick_IVscan", "f1": "F1/20260726_123347_quick_IVscan"},
    "3p5e15_2d":  {"h1": "H1/20260727_220346_quick_IVscan", "f1": "F1/20260727_220400_quick_IVscan"},
    "3p5e15_3d":  {"h1": "H1/20260728_172052_quick_IVscan", "f1": "F1/20260728_172038_quick_IVscan"},
    "3p5e15_4d":  {"h1": "H1/20260729_121022_quick_IVscan", "f1": "F1/20260729_121830_quick_IVscan"},
    # "+10 days" look (2026-09-19 kink check, see fig31_iv_3p5e15 module docstring): kept at
    # this scan despite IH12's reproducible ~3.6x jump at 570->580 V -- confirmed real/isolated
    # (independent 20260805_021311 scan same night, same jump, only IH12; absent 6 days earlier
    # in 20260729_121022, the +4-day look) rather than swapped to a stale kink-free substitute.
    "3p5e15_end": {"h1": "H1/20260805_033621_quick_IVscan", "f1": "F1/20260805_033755_quick_IVscan"},

    # FINE (0.1 V) July scans -- the low-V knee source (2026-09-19 user redirect: a low-V figure
    # may never be drawn from a 10 V quick scan, the gain-layer depletion voltage moves by a few
    # volts).  Taken straight from INPUTS/july/iv_scan_inventory.csv rows with type == "fine";
    # the quick-scan keys above stay the full-range source.  Reach (v_max, H1/F1):
    #   2e15_ra_fine    540 / 500 V      3p5e15_ra_fine    35 / 35 V
    #   2e15_2d_fine    550 / 550 V      3p5e15_4d_fine    60 / 60 V
    #                                    3p5e15_10d_fine   35 / 35 V
    # 4mo_1p5e15_fine_0717 reaches only 10.9 / 12.8 V, far short of the 0-60 V knee window, so
    # the 1.5e15 "+4 months" low-V look is drawn from the 2026-07-16 legacy fine logs instead
    # (JULY_FINE_SCANS["4mo_1p5e15_fine"], the campaign notebook's own V_gl source, used by
    # fig27 and fig39d).  The key is catalogued here anyway so its short reach is a recorded
    # fact rather than a missing entry.
    "2e15_ra_fine":     {"h1": "H1/20260720_190819_fine_IVscan", "f1": "F1/20260720_190823_fine_IVscan"},
    "2e15_2d_fine":     {"h1": "H1/20260722_011656_fine_IVscan", "f1": "F1/20260722_011719_fine_IVscan"},
    "3p5e15_ra_fine":   {"h1": "H1/20260725_164721_fine_IVscan", "f1": "F1/20260725_165028_fine_IVscan"},
    "3p5e15_4d_fine":   {"h1": "H1/20260729_082633_fine_IVscan", "f1": "F1/20260729_082546_fine_IVscan"},
    "3p5e15_10d_fine":  {"h1": "H1/20260805_040908_fine_IVscan", "f1": "F1/20260805_041055_fine_IVscan"},
    "4mo_1p5e15_fine_0717": {"h1": "H1/20260717_0307_fine_IV_scan", "f1": "F1/20260717_0307_fine_IVscan"},
}

# ============================================================================================
# from iv_timeseries.py
# ============================================================================================

# the week-1 spark test: one 4-channel HV-supply log (the HPK boards IH18/IH19/IH21/IH22 at
# 3e15 p/cm2, before the setup swap), read in place like the other raw logs
MARCH_SPARK_CSV = os.path.join(EOS_ROOT, "week1",
                               "sparktest_T3_m25c_parking_3e15pcm2_interpolated_03112026.csv")

DISPLAY_RUNS_JUL_CSV = os.path.join(INPUTS_JULY, "display_runs_jul.csv")

CHIPS = {"h1": ["IH7", "IH11", "IH12", "IH13"], "f1": ["LF10", "LF12", "LF17", "LF18"]}

# ---------------------------------------------------------------- March: runs, files, references
# H1 windows/biases/references straight from the build brief (session etroc-analysis-61).
MARCH_H1_RUNS = {
    10: dict(fluence=3e14, start_utc="2026-03-16 20:39", plateau_h=12.4,
             bias={"IH7": 210, "IH11": 210, "IH12": 210, "IH13": 210},
             ref_uA={"IH7": 251.4, "IH11": 239.4, "IH12": 223.2, "IH13": 250.2}),
    11: dict(fluence=3e14, start_utc="2026-03-17 10:05", plateau_h=12.5,
             bias={"IH7": 280, "IH11": 280, "IH12": 280, "IH13": 280},
             ref_uA={"IH7": 709.3, "IH11": 663.4, "IH12": 631.5, "IH13": 746.5}),
    14: dict(fluence=9e14, start_utc="2026-03-20 16:32", plateau_h=12.3,
             bias={"IH7": 410, "IH11": 410, "IH12": 410, "IH13": 410},
             ref_uA={"IH7": 1767, "IH11": 1503, "IH12": 1314, "IH13": 1458}),
    15: dict(fluence=9e14, start_utc="2026-03-21 05:42", plateau_h=12.1,
             bias={"IH7": 380, "IH11": 380, "IH12": 380, "IH13": 380},
             ref_uA={"IH7": 860.3, "IH11": 799.4, "IH12": 744.9, "IH13": 786.2}),
    17: dict(fluence=1.5e15, start_utc="2026-03-23 15:29", plateau_h=12.5,
             bias={"IH7": 400, "IH11": 400, "IH12": 400, "IH13": 400},
             ref_uA={"IH7": 674.6, "IH11": 633.1, "IH12": 577.0, "IH13": 613.8}),
    19: dict(fluence=1.5e15, start_utc="2026-03-24 09:51", plateau_h=12.5,
             bias={"IH7": 465, "IH11": 470, "IH12": 470, "IH13": 470},
             ref_uA={"IH7": 1527, "IH11": 1348, "IH12": 1137, "IH13": 1265}),
}

MARCH_F1_RUN_NUMS = {10: "run10", 11: "run11", 14: "run14", 15: "run15", 17: "run17", 19: "run19"}

F1_WINDOWS_CSV = os.path.join(INPUTS_MARCH, "irrad_f1_inrun_currents_2026mar.csv")

# raw ~1 Hz logs: one file per telescope x fluence step, covering the two runs at that step
MARCH_RAW_FILES = {
    ("h1", 3e14): os.path.join(MARCH_EOS_WEEK2,
                                "fine_iv_H1_m25c_17cm_3e14_03182026_1654_interpolated.csv"),
    ("h1", 9e14): os.path.join(MARCH_EOS_WEEK2,
                                "fine_iv_H1_m25c_17cm_9e14_03212026_1857_interpolated.csv"),
    ("h1", 1.5e15): os.path.join(MARCH_EOS_15E14, "H1",
                                  "fine_iv_H1_m25c_17cm_15e14_03252026_0617_interpolated.csv"),
    ("f1", 3e14): os.path.join(MARCH_EOS_WEEK2,
                                "fine_iv_F1_m25C_17cm_3e14_03182026_1654_interpolated.csv"),
    ("f1", 9e14): os.path.join(MARCH_EOS_WEEK2,
                                "fine_iv_F1_m25C_17cm_9e14_03212026_1858_interpolated.csv"),
    ("f1", 1.5e15): os.path.join(MARCH_EOS_15E14, "F1",
                                  "quick_iv_F1_m25C_17cm_15e14_03252026_0605_interpolated.csv.gz"),
}

RUNS_AT_FLUENCE = {3e14: [10, 11], 9e14: [14, 15], 1.5e15: [17, 19]}

# ---------------------------------------------------------------- July: timeline + run selection
JULY_TIMELINE = os.path.join(INPUTS_JULY, "timeline_60s.csv.gz")

JULY_REF_CSV = os.path.join(INPUTS_JULY, "july_inrun_currents.csv")

JULY_TEL = {"h1": "H1", "f1": "F1"}

# team good-run list: campaign,telescope,run,good,reason.  The telescope label there
# ("H1_HPK"/"F1_FBK") differs from JULY_TEL.
GOOD_RUNS_CSV = (os.environ.get("ETROC_GOOD_RUNS_CSV")
                 or os.path.join(INPUTS_JULY, "talk_good_runs.csv"))

GOOD_RUNS_CAMPAIGN = "IRRAD_Jul2026"

GOOD_RUNS_TEL = {"h1": "H1_HPK", "f1": "F1_FBK"}

# ----------------------------------------------------------- July irradiation steps (2e15, 3.5e15)
# july_run_selection() draws, at each step, the runs of display_runs_jul.csv with RFSel
# JULY_STEPS_RFSEL and threshold offset JULY_STEPS_OFFSET that are on the good-run list, and of
# those only the boards whose HV plateau in JULY_REF_CSV lasts at least JULY_MIN_PLATEAU_H hours.
JULY_STEPS = (2e15, 3.5e15)

JULY_STEPS_RFSEL = 2

JULY_STEPS_OFFSET = 20

JULY_MIN_PLATEAU_H = 4.0

# ------------------------------------------------- July 1.5e15 (the runs after the four-month rest)
# july_run_selection() covers only the two July irradiation steps (2e15, 3.5e15).  The 1.5e15 runs
# of July are the sensors irradiated in March, measured again after the four-month rest, and the
# talk shows them as their own currents figure (user request 2026-09-19).  They are not on
# display_runs_jul.csv (that list is the resolution display runs at the July steps), so the gate
# here is the one the user stated: on the team good-run list, RFSel 2 and Disc offset 20 on EVERY
# board in the campaign yaml, and an HV-monitor plateau in the reference CSV (status "ok" -- runs
# 3 and 4 of H1 and run 3 of F1 ended before the July HV-monitor timeline starts and carry
# status "no_hv_log", so they cannot be drawn at all).
JULY_YAML = os.path.join(REPO_TESTBEAM, "board_configs_yaml", "CERN_Irrad_2026Jul.yaml")

JULY_1P5E15_RFSEL = 2

JULY_1P5E15_OFFSET = 20

# --------------------------------------------------------- v2: in-run spike/excursion scan (fig32)
# fig32 v2 review: identify runs with in-run current spikes so they can be kept out of the
# resolution plots -- scanned over EVERY July run in july_timeline.json's "per_run" table (both
# telescopes, all runs, not only the ones fig32 actually draws), never tuned to the runs fig32
# happens to plot.
JULY_TIMELINE_JSON = os.path.join(INPUTS_JULY, "july_timeline.json")

# ============================================================================================
# from iv_plot.py
# ============================================================================================

# ------------------------------------------------------ on-chip fluence label option (2026-09-20)
# On-chip fluence factor: 0.748 = 1.122e15 (ETROC foil DOS-005210.2, March dosimetry, +-7%) / 1.5e15
# facility bookkeeping. Two campaign figures show this axis instead of bookkeeping fluence
# (fig39b_iv_campaign_chipdose_v2, fig40_vgl_campaign_chipdose_v2); series/colours/timing labels
# are otherwise untouched -- only the printed/plotted fluence NUMBER changes.
CHIP_DOSE_FACTOR = 0.748

CHIP_DOSE_FOOTER = ("fluence on chip = %.3f x bookkeeping (March dosimetry, +-7 %%)"
                    % CHIP_DOSE_FACTOR)

_CHIP_DOSE_TEXT = {
    0.0: "pre-irradiation",
    3e14: r"$2.2\times10^{14}$",
    9e14: r"$6.7\times10^{14}$",
    1.5e15: r"$1.12\times10^{15}$",
    2e15: r"$1.50\times10^{15}$",
    3.5e15: r"$2.62\times10^{15}$",
}


# ==============================================================================================
# V_gl vs fluence
# ==============================================================================================
# Every entry must equal the value for the same chip, fluence and look in INPUTS/vgl/vgl_points.csv
# within 0.05 V; iv_data.check_vgl_table() checks this where figures 34-39 are drawn.
# Gain-layer depletion voltage read off the fine IV scans, per board, per irradiation step, per
# look: (bookkeeping fluence in 1e15 p/cm2, look label, [V_gl per board]). Board order matches
# TEL_CHIPS, so VGL_H1 rows index TEL_CHIPS["h1"] and VGL_F1 rows index TEL_CHIPS["f1"].
# Fluence is stored in units of 1e15 bookkeeping 24 GeV/c protons per cm2 so
# that the fitted decay constant is O(1) and the solve stays well conditioned.
# Each row: (fluence, timing label, [V_gl per board in the BOARDS order], V)
VGL_H1 = [
    (0.00, "right after", [35.0, 34.9, 35.0, 35.3]),   # pre-irradiation
    (0.30, "right after", [30.6, 30.2, 30.1, 30.5]),
    (0.30, "+2 d",        [31.7, 31.7, 31.4, 31.6]),
    (0.90, "right after", [26.3, 25.2, 25.0, 25.1]),
    (0.90, "+2 d",        [27.1, 26.9, 26.2, 26.6]),
    (1.50, "right after", [22.5, 22.4, 21.9, 21.9]),
    (1.50, "+2 d",        [23.6, 23.0, 22.7, 22.7]),
    (1.50, "4 months",    [22.8, 21.9, 21.8, 21.6]),
    (2.00, "right after", [18.5, 17.5, 17.1, 17.3]),
    (2.00, "+2 d",        [18.9, 18.2, 17.8, 18.1]),
    (3.50, "right after", [9.9, 8.9, 8.2, 8.1]),
    (3.50, "+4 d",        [10.9, 10.2, 9.9, 10.0]),
]

# Fluence is stored in units of 1e15 bookkeeping 24 GeV/c protons per cm2 so
# that the fitted decay constant is O(1) and the solve stays well conditioned.
# Each row: (fluence, timing label, [V_gl per board in the BOARDS order], V)
VGL_F1 = [
    (0.00, "right after", [45.3, 45.1, 45.0, 45.7]),   # pre-irradiation
    (0.30, "right after", [42.1, 41.9, 42.1, 42.0]),
    (0.30, "+2 d",        [44.6, 44.2, 44.1, 44.6]),
    (0.90, "right after", [39.9, 39.1, 39.4, 38.9]),
    (0.90, "+2 d",        [41.9, 41.9, 41.2, 41.5]),
    (1.50, "right after", [38.4, 37.3, 37.3, 37.1]),
    (1.50, "+2 d",        [39.5, 39.5, 38.9, 38.5]),
    (1.50, "4 months",    [39.8, 39.6, 37.8, 38.1]),
    (2.00, "right after", [33.5, 32.9, 32.3, 32.6]),
    (2.00, "+2 d",        [34.9, 33.4, 33.1, 33.4]),
    (3.50, "right after", [24.5, 23.7, 23.8, 23.7]),
    (3.50, "+4 d",        [26.7, 26.0, 24.3, 24.4]),
]

# Look-time vocabulary used by the V_gl figures.
PROMPT = "right after"
LATE_LABELS = ("+2 d", "+4 d")
VERY_LATE = "4 months"

# Vendor of each telescope's sensors: picks the published removal constants (vgl.REFERENCES).
VGL_VENDOR = {"h1": "HPK", "f1": "FBK"}
# What the published constants would predict under chip-hit fractions the gamma-spectroscopy
# measurement excludes: (reference name, chip-hit fraction). Drawn dotted on the comparison.
VGL_REF_ALTERNATES = {"h1": (("Curras 24 GeV/c p", 1.00), ("Kraus W25", 0.50), ("Kraus W36", 0.50)),
                      "f1": ()}

# Fluence at or above which a point is dropped for the "without the last step" fit variant.
VGL_DROP_ABOVE = 3.4

# Fluence conversion from one facility bookkeeping proton to what the sensor
# actually saw. The chip-hit fraction is MEASURED, not assumed: gamma
# spectroscopy of the ETROC target segment after the campaign gave
# 1.122e15 p/cm2 +- 7 percent against 1.5e15 bookkeeping, that is 0.75 +- 0.05.
# F1 and H1 sat back to back in the same box and the same beam, so the same
# number applies to both telescopes.
# 0.62 +- 0.04 is the 24 GeV/c proton NIEL hardness factor, so the three early
# steps correspond to 1.4, 4.2 and 7.0e14 n_eq.
# Provenance: ETL talk, 20 April 2026.
NIEL_24GEV = 0.62
CHIP_HIT_FRACTION = 0.75      # measured by gamma spectroscopy
CHIP_HIT_REL_ERR = 0.07       # +- 7 percent on the measured on-chip fluence
GAMMA_MEAS_ONCHIP = 1.122     # 1e15 p/cm2 measured at 1.5e15 bookkeeping
GAMMA_MEAS_BOOKKEEPING = 1.50
if abs(CHIP_DOSE_FACTOR - GAMMA_MEAS_ONCHIP / GAMMA_MEAS_BOOKKEEPING) > 1e-9:
    raise ValueError("irrad_2026.py: CHIP_DOSE_FACTOR = %r is not GAMMA_MEAS_ONCHIP / "
                     "GAMMA_MEAS_BOOKKEEPING = %r; update the two together"
                     % (CHIP_DOSE_FACTOR, GAMMA_MEAS_ONCHIP / GAMMA_MEAS_BOOKKEEPING))
CONV_FULL = 0.467          # n_eq per bookkeeping proton, NIEL times chip hit

VGL_XGRID_MAX = 3.72      # x extent of the drawn model curves, just past the last step
VGL_XGRID_N = 400


# ============================================================================================
# from the talk's build_current_vs_run.py (2026-09-22): July per-run bias-current panels
# ============================================================================================
# The resolution-vs-run figure's sidecar: run order and per-run class / flag / fluence / offset /
# RFSel (a copy of the one the talk's resolution builder wrote on 2026-09-21).
COMBO_CHECK_JSON = (os.environ.get("ETROC_COMBO_CHECK_JSON")
                    or os.path.join(INPUTS_JULY, "res_vs_run_combo_check_jul_values.json"))
# HV / LV cycle, irradiation-step and DAQ-restart marks per run (written by the resolution
# combo-check builder)
HV_CYCLES_JUL_CSV = (os.environ.get("ETROC_HV_CYCLES_JUL_CSV")
                     or os.path.join(INPUTS_JULY, "hv_cycles_jul.csv"))
# in-run bias-current spike runs (the July in-run current scan)
CURRENT_SPIKE_RUNS = {"h1": {11, 53, 56, 57, 64, 70}, "f1": {17, 18, 53, 59, 71, 86}}
FLUENCE_TEXT_PLAIN = {0.0: "pre", 3e14: "3e14", 9e14: "9e14", 1.5e15: "1.5e15", 2e15: "2e15",
                      3.5e15: "3.5e15"}

RAD_STOP_SOURCE_TEXT = (
    "logged radiation stop: rad_stop_utc of july/july_inrun_currents.csv (per-telescope stamp), "
    "LV-on time from lv_on_utc of july/hv_cycles_jul.csv, both in the inputs folder; shown on "
    "the first run after each LV-off window (H1 runs 11/14, F1 runs 10/13)."
)

RAD_STOP_SOURCE_TEXT_FIGURE = (
    "logged radiation stop: the per-run current table's rad_stop_utc (per-telescope stamp), "
    "LV-on time from the per-run HV/LV cycle table (lv_on_utc); shown on the first run after "
    "each LV-off window (H1 runs 11/14, F1 runs 10/13)."
)

DAQ_RESTART_SENTENCE = (
    "DAQ restart: H1 run 18 / F1 run 17, the 2026-07-27 DAQ restart. Source: the restart column "
    "of july/hv_cycles_jul.csv in the inputs folder."
)

DAQ_RESTART_SENTENCE_FIGURE = (
    "DAQ restart: H1 run 18 / F1 run 17, the 2026-07-27 DAQ restart. Source: the restart column "
    "of the per-run HV/LV cycle table."
)


# ============================================================================================
# from the March walkthrough's analyse/plot_fig2_current.py (2026-09-22): pre-irradiation bias
# and leakage-current stability of H1, iv.preirrad_current
# ============================================================================================
# The iseg slow-control log covering the whole pre-irradiation phase: the interpolated companion
# of the fine IV scan that closed it (~0.5 s sampling, continuous 2026-03-11 15:38 -> 03-16
# 07:04 local time, ending at the start of irradiation step 1).
PREIRRAD_LOG = (os.environ.get("ETROC_PREIRRAD_LOG")
                or os.path.join(MARCH_EOS_WEEK2, "fine_iv_H1_m25c_17cm_03162026_0313_interpolated.csv"))
# per-run conditions: run start (UTC, from the DAQ run metadata), fluence, bias, threshold offset
PREIRRAD_CONDITIONS_CSV = (os.environ.get("ETROC_PREIRRAD_CONDITIONS_CSV")
                           or os.path.join(INPUTS_MARCH, "conditions_March2026_H1_HPK.csv"))
# the three files the figure is drawn from; regenerate with iv.preirrad_current --rebuild
PREIRRAD_CACHE_DIR = os.path.join(INPUTS_MARCH, "preirrad_cache")
# Local-clock-to-UTC offset, hours, of every March slow-control log (this one, the in-run logs,
# the spark-test log) and of the time stamps in the March scan file names: local CERN time,
# CET = UTC+1, no daylight saving before 2026-03-29. UTC = local - PREIRRAD_LOG_UTC_OFFSET_H.
PREIRRAD_LOG_UTC_OFFSET_H = 1
# HV channel (the log's column index) -> (role, chip)
PREIRRAD_CHANNELS = {0: ("extra", "IH7"), 1: ("ref", "IH11"), 2: ("dut", "IH12"), 3: ("trig", "IH13")}
# DAQ run length, minutes (run metadata max_run_time_minutes): the run END comes from the DAQ,
# the log only supplies the HV plateau, and the window is the intersection of the two
PREIRRAD_MAX_RUN_MIN = {1: 720, 2: 540, 4: 720, 5: 720, 6: 720, 7: 720, 8: 720, 9: 720}
# Independent reference for the printed cross-check: in-run medians (uA; IH7/IH11/IH12/IH13) and
# HV-plateau lengths (h) from the campaign's IV analysis, computed separately from this code.
PREIRRAD_REF_MEDIANS = {
    1: (0.048, 0.048, 0.051, 0.070), 2: (0.090, 0.090, 0.092, 0.116),
    4: (0.114, 0.114, 0.115, 0.139), 5: (0.124, 0.124, 0.124, 0.145),
    6: (0.088, 0.087, 0.084, 0.094), 7: (0.447, 0.460, 0.514, 0.787),
    8: (0.096, 0.095, 0.091, 0.100), 9: (0.109, 0.107, 0.103, 0.113)}
PREIRRAD_REF_PLATEAU_H = {1: 12.5, 2: 9.0, 4: 12.5, 5: 12.0, 6: 12.0, 7: 12.0, 8: 12.0, 9: 12.0}
PREIRRAD_TEXT = {
    "exp_text": "ETL ETROC IRRAD",
    "campaign": "CERN IRRAD, March 2026",
    "subject": "H1 (HPK)",
    "hybrid": "H1 telescope\nHPK LGAD + ETROC2.02",
    "footer": "No temperature channel in the slow-control record; \u221225 \u00b0C is the cold-box "
              "set point.",
}
# the between-run IV ramps, named once: arrow tip at (end of the first run window + after_first_run_h,
# y V), text at text_dt_h further right and text_y V
PREIRRAD_RAMP_NOTE = {"text": "bracketing IV scans\n(quick / fine ramps, to ~240 V)",
                      "after_first_run_h": 1.6, "y": 214, "text_dt_h": 6, "text_y": 232}
# spike labels, one per run with spikes in time order: (label y, arrow-tip y) in uA, staggered so
# the boxes never touch; a campaign with more spike runs needs more entries
PREIRRAD_SPIKE_LABEL_Y = {0: (9.0, 0.33), 1: (1.5, 0.26)}


# ============================================================================================
# raw inputs, read in place (iv_data.check_inputs checks each one exists and is readable)
# ============================================================================================
def _outside_inputs(paths):
    """The paths not under INPUTS (the md5 manifest covers those), sorted, each once."""
    inside = os.path.join(os.path.abspath(INPUTS), "")
    return sorted({p for p in paths if not os.path.abspath(p).startswith(inside)})


# Every file this campaign reads outside the tables folder: the March scans and slow-control
# logs on EOS, the July board-config yaml in the repository, and any table moved by its own
# environment variable. PREIRRAD_LOG and PREIRRAD_CONDITIONS_CSV are read only by
# `python -m etroc_plots.iv.preirrad_current --rebuild` and are listed so the check covers them.
RAW_INPUTS = _outside_inputs(
    [p for tel in MARCH_SCANS.values() for look in tel.values() for p in look.values()]
    + list(MARCH_RAW_FILES.values())
    + [spec["path"] for scan in JULY_FINE_SCANS.values() for spec in scan.values()]
    + [MARCH_SPARK_CSV, PREIRRAD_LOG, PREIRRAD_CONDITIONS_CSV, JULY_YAML, GOOD_RUNS_CSV,
       COMBO_CHECK_JSON, HV_CYCLES_JUL_CSV])
