"""Campaign catalogues: which file is which scan, which runs sit at which fluence.

One module per campaign. `active` is the one in use, chosen by the environment variable
ETROC_IV_CAMPAIGN (default "irrad_2026"), which is read once, on the first import of
etroc_plots.iv: set it before that import. The helper modules copy the active campaign's names
when they are imported, so switching to another campaign needs a fresh Python process (in a
notebook: restart the kernel).

To plot a new campaign, copy irrad_2026.py, edit the catalogue to match the new run, and set
ETROC_IV_CAMPAIGN=<your module>. The module must define every name in REQUIRED; loading it
stops with the list of missing names otherwise.
"""
import importlib
import os

# every name the IV helpers and the IV notebook read from the campaign module
REQUIRED = (
    "CHIPS", "CHIP_DOSE_FACTOR", "CHIP_DOSE_FOOTER", "CHIP_HIT_FRACTION", "CHIP_HIT_REL_ERR",
    "COMBO_CHECK_JSON", "CONV_FULL", "CURRENT_SPIKE_RUNS", "DAQ_RESTART_SENTENCE",
    "DAQ_RESTART_SENTENCE_FIGURE", "DISPLAY_RUNS_JUL_CSV", "EOS_ROOT", "F1_WINDOWS_CSV",
    "FLUENCE_TEXT_PLAIN", "GOOD_RUNS_CAMPAIGN", "GOOD_RUNS_CSV", "GOOD_RUNS_TEL",
    "HV_CYCLES_JUL_CSV", "INPUTS", "INPUTS_JULY", "INPUTS_JULY_FINE", "INPUTS_MANIFEST",
    "INPUTS_MARCH", "INPUTS_VGL", "JULY_1P5E15_OFFSET", "JULY_1P5E15_RFSEL", "JULY_FINE_SCANS",
    "JULY_MIN_PLATEAU_H", "JULY_REF_CSV", "JULY_SCANS", "JULY_STEPS", "JULY_STEPS_OFFSET",
    "JULY_STEPS_RFSEL", "JULY_TEL", "JULY_TIMELINE", "JULY_TIMELINE_JSON", "JULY_YAML",
    "LATE_LABELS", "LINE1_PARKED", "MARCH_EOS_15E14", "MARCH_EOS_WEEK2", "MARCH_F1_RUN_NUMS",
    "MARCH_H1_RUNS", "MARCH_RAW_FILES", "MARCH_SCANS", "MARCH_SPARK_CSV", "NIEL_24GEV",
    "PREIRRAD_CACHE_DIR", "PREIRRAD_CHANNELS", "PREIRRAD_CONDITIONS_CSV", "PREIRRAD_LOG",
    "PREIRRAD_LOG_UTC_OFFSET_H", "PREIRRAD_MAX_RUN_MIN", "PREIRRAD_RAMP_NOTE",
    "PREIRRAD_REF_MEDIANS", "PREIRRAD_REF_PLATEAU_H", "PREIRRAD_SPIKE_LABEL_Y", "PREIRRAD_TEXT",
    "PROMPT", "RAD_STOP_SOURCE_TEXT", "RAD_STOP_SOURCE_TEXT_FIGURE", "RAD_STOP_UTC",
    "RAW_INPUTS", "RUNS_AT_FLUENCE", "TEL_CHIPS", "VERY_LATE", "VGL_DROP_ABOVE", "VGL_F1",
    "VGL_H1", "VGL_REF_ALTERNATES", "VGL_VENDOR", "VGL_XGRID_MAX", "VGL_XGRID_N",
    "_CHIP_DOSE_TEXT",
)


def check_required(module):
    """Raise RuntimeError naming every REQUIRED name `module` does not define."""
    missing = [n for n in REQUIRED if not hasattr(module, n)]
    if missing:
        raise RuntimeError("campaign module %s does not define %s; copy them from irrad_2026.py "
                           "and adapt them" % (module.__name__, ", ".join(missing)))


def check_chips(telescope_chips, module=None):
    """Raise ValueError unless `telescope_chips` ({telescope: [chip, ...]}, the chips every
    panel draws) equals the campaign's TEL_CHIPS and CHIPS, telescope by telescope, in order."""
    module = module or active
    want = {tel: list(chips) for tel, chips in telescope_chips.items()}
    for name in ("TEL_CHIPS", "CHIPS"):
        have = {tel: list(chips) for tel, chips in getattr(module, name).items()}
        if have != want:
            raise ValueError("talk_style.TELESCOPE_CHIPS = %s but campaign %s has %s = %s; "
                             "make them equal" % (want, module.__name__, name, have))


NAME = os.environ.get("ETROC_IV_CAMPAIGN", "irrad_2026")
active = importlib.import_module("." + NAME, __name__)
check_required(active)
