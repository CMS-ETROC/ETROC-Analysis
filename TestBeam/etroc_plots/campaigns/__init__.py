"""Campaign modules: what a campaign is and where its inputs live.

One module per campaign (irrad_2026.py is CERN IRRAD 2026). `active` is the one in use, chosen
by the environment variable ETROC_CAMPAIGN (default "irrad_2026") and read once, on the first
import of any etroc_plots module that needs it: set it before that import. The helper modules
copy the active campaign's names when they are imported, so switching to another campaign needs
a fresh Python process (in a notebook: restart the kernel).

The package README (etroc_plots/README.md, "A new campaign") gives the steps to plot a new
campaign. REQUIRED lists the names each part of the package reads: every figure needs the
"style" names, the IV helpers and the IV notebook also the "iv" names. Importing a part stops
with the list of names the campaign module is missing.
"""
import importlib
import os

REQUIRED = {
    # identity and look, read by etroc_plots.style
    "style": (
        "EXP_TEXT", "FLUENCE_COLOR", "FLUENCE_STEPS", "FLUENCE_TEXT", "FLUENCE_UNIT",
        "HEADER_LINE_1", "HEADER_LINE_2", "PREAMP_POWER", "TELESCOPE_CHIPS", "TELESCOPE_TITLE",
    ),
    # scan catalogue, run lists and inputs, read by etroc_plots.iv and notebooks/iv.py
    "iv": (
        "BINNED_FROM_RAW", "CHIP_FLUENCE_FACTOR", "CHIP_FLUENCE_FOOTER", "CHIP_FLUENCE_REL_ERR",
        "CHIP_FLUENCE_TEXT", "COMBO_CHECK_JSON", "CONV_FULL", "CURRENT_SPIKE_RUNS",
        "DAQ_RESTART_SENTENCE", "DAQ_RESTART_SENTENCE_FIGURE", "DISPLAY_RUNS_JUL_CSV",
        "EOS_ROOT", "F1_WINDOWS_CSV", "FLUENCE_TEXT_PLAIN", "GOOD_RUNS_CAMPAIGN",
        "GOOD_RUNS_CAMPAIGN_MARCH", "GOOD_RUNS_CSV", "GOOD_RUNS_TEL", "HV_CYCLES_JUL_CSV",
        "INPUTS", "INPUTS_JULY", "INPUTS_JULY_FINE", "INPUTS_MANIFEST", "INPUTS_MARCH",
        "INPUTS_VGL", "JULY_1P5E15_OFFSET", "JULY_1P5E15_RFSEL", "JULY_FINE_SCANS",
        "JULY_MIN_PLATEAU_H", "JULY_REF_CSV", "JULY_SCANS", "JULY_STEPS", "JULY_STEPS_OFFSET",
        "JULY_STEPS_RFSEL", "JULY_TEL", "JULY_TIMELINE", "JULY_TIMELINE_JSON", "JULY_YAML",
        "LATE_LABELS", "LINE1_PARKED", "MARCH_EOS_15E14", "MARCH_EOS_WEEK2",
        "MARCH_F1_RUN_NUMS", "MARCH_H1_RUNS", "MARCH_OFFSET", "MARCH_RAW_FILES", "MARCH_SCANS",
        "MARCH_SPARK_CSV", "MARCH_YAML", "NIEL_24GEV", "PREIRRAD_CACHE_DIR",
        "PREIRRAD_CHANNELS", "PREIRRAD_CONDITIONS_CSV", "PREIRRAD_LOG",
        "PREIRRAD_LOG_UTC_OFFSET_H", "PREIRRAD_MAX_RUN_MIN", "PREIRRAD_RAMP_NOTE",
        "PREIRRAD_REF_MEDIANS", "PREIRRAD_REF_PLATEAU_H", "PREIRRAD_SPIKE_LABEL_Y",
        "PREIRRAD_TEXT", "PROMPT", "RAD_STOP_SOURCE_TEXT", "RAD_STOP_SOURCE_TEXT_FIGURE",
        "RAD_STOP_UTC", "RAW_INPUTS", "RUNS_AT_FLUENCE", "VALUES_CONVENTIONS", "VERY_LATE",
        "VGL_DROP_ABOVE", "VGL_F1", "VGL_H1", "VGL_REF_ALTERNATES", "VGL_VENDOR",
        "VGL_XGRID_MAX", "VGL_XGRID_N",
    ),
}


def check_required(part, module=None):
    """Raise RuntimeError naming every REQUIRED[part] name the campaign module (default: the
    active one) does not define."""
    module = module or active
    missing = [n for n in REQUIRED[part] if not hasattr(module, n)]
    if missing:
        raise RuntimeError("campaign module %s does not define %s (read by the %r part of "
                           "etroc_plots); copy them from irrad_2026.py and adapt them"
                           % (module.__name__, ", ".join(missing), part))


NAME = os.environ.get("ETROC_CAMPAIGN") or "irrad_2026"     # unset or empty: the default
active = importlib.import_module("." + NAME, __name__)
