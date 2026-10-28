"""DESY 2026: the May and August electron test beams of the ETROC2 telescope, facts only.

The universality figures draw DESY runs beside the IRRAD ones, so they need the DESY run facts:
per board of a run its chip, role, bias, threshold offset, angle to the beam, temperature and
RFSel (run_settings, read from the campaign's board-config yaml), and the few campaign facts a
figure states. This module is not a campaign module of the campaigns.REQUIRED kind: it has no
REQUIRED parts, ETROC_CAMPAIGN never selects it, and the code that needs it imports it by name
(`from etroc_plots.campaigns import desy_2026`).

Campaign keys here are "may" (DESY May 2026) and "aug" (DESY Aug 2026). The TWC-transfer study's
coefficient bank and result tables write them "desy" (May) and "aug" (August): BANK_CAMPAIGN maps
the bank's codes to these keys.

What run_settings does with the yamls' quirks:

* May runs 1-3 carry no temperature (temp None), and they keep the default roles, in which
  PT-IH5 is the trigger and PT-IH3 the reference; every later May run swaps the two (PT-IH5
  reference, PT-IH3 trigger).
* May runs 1-3, 26 and 27 carry no angle key: the boards stood normal to the beam, angle 0.
* Temperatures are strings with a C suffix ("11C", "-20C", "1.5C"); run_settings returns them
  as floats in degrees C. They are the per-board sensor values the yaml records, not a set point.
* No key marks the offset holder, the second mounting of the angled telescope:
  OFFSET_HOLDER_RUNS lists its runs (in the yaml they are the only runs rotated the other way,
  rotation y +30 and +60).
* May runs 17-19 are the 45 deg bias step: boards 0-2 at 230/240/230 V and 7.5 C, PT-IH2 (board 3)
  at 200 V and -10 C.
* August: RFSel is a yaml key only on runs 20-23 (RFSel 3, threshold offsets 6-10); every other
  August run ran at DEFAULT_RFSEL["aug"]. The May yaml has no RFSel key and the May DAQ
  configuration records none, so a May RFSel is None (not recorded).
* August run 18: the yaml copies run 17 (board 3 at 140 V), but the DAQ run configuration records
  110 V on board 3; run_settings returns the DAQ value (HV_FROM_DAQ).
"""
import os
import re

from .._paths import REPO_TESTBEAM

CAMPAIGN_TEXT = {"may": "DESY May 2026", "aug": "DESY Aug 2026"}
# the bank's and the result tables' campaign codes -> the keys of this module
BANK_CAMPAIGN = {"desy": "may", "aug": "aug"}

# DESY-II electrons. May ran at 4 GeV throughout. In August the DAQ run names of runs 7-21 and 23
# carry "_3GeV" (AUG_3GEV_RUNS); runs 5, 6 and 22 ran at 4 GeV (AUG_4GEV_RUNS), as the August
# yaml's comments say too. beam_gev() and beam_text() read these; every DESY label and values
# file takes its beam from beam_text().
AUG_3GEV_RUNS = frozenset(list(range(7, 22)) + [23])
AUG_4GEV_RUNS = frozenset({5, 6, 22})

# The board-config yamls, located like the IRRAD campaign's (irrad_2026.MARCH_YAML). The August
# file is the corrected one: the same settings as DESY_TB_2026Aug.yaml, with the measured
# telescope geometry and the 60 deg default of the whole campaign.
_YAML_DIR = os.path.join(REPO_TESTBEAM, "board_configs_yaml")
MAY_YAML = os.path.join(_YAML_DIR, "DESY_TB_2026May_CE.yaml")
AUG_YAML = os.path.join(_YAML_DIR, "DESY_TB_2026Aug_corrected.yaml")
YAML = {"may": MAY_YAML, "aug": AUG_YAML}

# The RFSel of a run whose yaml has no RFSel key. Source: each campaign's DAQ run configuration.
# RFSel is the preamp feedback setting (R_f), which sets the preamp gain. August records RFSel 2
# on every chip of runs 2-19 and 3 on runs 20-23 (which the yaml carries as RFSel 3); May
# records no RFSel, so None: a figure omits the RFSel there.
DEFAULT_RFSEL = {"may": None, "aug": 2}
# The preamp power mode, from the same DAQ configuration: August records "high" on every run and
# chip; May records none (None: a figure omits it).
PREAMP_POWER = {"may": None, "aug": "high"}

# The May runs taken on the offset holder (see the module docstring); never pooled with the
# aligned angle grid.
OFFSET_HOLDER_RUNS = frozenset({42, 44, 45})

# (campaign, run, board) -> bias in V where the DAQ run configuration and the yaml disagree
HV_FROM_DAQ = {("aug", 18, 3): 110.0}

_YAML_CACHE = {}


def _config(campaign):
    if campaign not in YAML:
        raise KeyError("DESY campaign %r: use one of %s" % (campaign, sorted(YAML)))
    if campaign not in _YAML_CACHE:
        import yaml
        with open(YAML[campaign]) as fh:
            _YAML_CACHE[campaign] = yaml.safe_load(fh)
    return _YAML_CACHE[campaign]


def _temp(value):
    """'11C' / '-20C' / '1.5C' -> 11.0 / -20.0 / 1.5; None stays None."""
    if value is None:
        return None
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*C\s*", str(value))
    if not m:
        raise ValueError("temperature %r is not a number followed by C" % (value,))
    return float(m.group(1))


def _run_number(run):
    return int(str(run).rsplit("run", 1)[-1])


def beam_gev(campaign, run):
    """The electron energy of one DESY run in GeV: 4 in May; in August 3 for AUG_3GEV_RUNS and 4
    for AUG_4GEV_RUNS, and a ValueError for a run in neither. campaign "may" or "aug", or the
    bank's code ("desy" = May); run an int or "run26"."""
    campaign, run = BANK_CAMPAIGN.get(campaign, campaign), _run_number(run)
    if campaign == "may":
        return 4
    if campaign != "aug":
        raise KeyError("DESY campaign %r: use one of %s" % (campaign, sorted(YAML)))
    if run in AUG_3GEV_RUNS:
        return 3
    if run in AUG_4GEV_RUNS:
        return 4
    raise ValueError("no beam energy recorded for DESY Aug run %d" % run)


def beam_text(campaign, runs):
    """The beam of a set of runs of one DESY campaign, as a figure states it: "4 GeV electrons" or
    "3 GeV electrons" when every run shares it, else "3 and 4 GeV electrons (run 5 at 4 GeV)",
    naming the runs at 4 GeV. Arguments as for beam_gev."""
    by = {}
    for r in runs:
        by.setdefault(beam_gev(campaign, r), set()).add(_run_number(r))
    if not by:
        raise ValueError("beam_text needs at least one run")
    if len(by) == 1:
        return "%d GeV electrons" % next(iter(by))
    four = sorted(by[4])
    words = ("run %d" % four[0] if len(four) == 1 else
             "runs %s and %d" % (", ".join(str(r) for r in four[:-1]), four[-1]))
    return "3 and 4 GeV electrons (%s at 4 GeV)" % words


def run_settings(campaign, run):
    """Per board of one DESY run, from the campaign's yaml (module docstring: the quirks), as a
    list of dicts in board order:

        board_idx  0-3
        chip       the yaml's "short" name ("PT-IH3", "B1-P1-9")
        role       the yaml's role word ("ref", "trig", ...)
        hv         bias in V (float)
        offset     threshold offset (int): Disc threshold = baseline + offset
        angle      angle to the beam in deg (int), 0 where the yaml has no angle key
        temp       the board's temperature in deg C (float), None where the yaml has none
        rfsel      int, from the yaml's RFSel key or DEFAULT_RFSEL[campaign]; None when unknown

    campaign is "may" or "aug" (BANK_CAMPAIGN maps the bank's codes); run an int or "run26"."""
    run = _run_number(run)
    cfg = _config(campaign)
    key = "run%d" % run
    if key not in cfg:
        raise KeyError("%s has no %s" % (os.path.basename(YAML[campaign]), key))
    boards = []
    for idx in sorted(cfg[key]):
        b = cfg[key][idx]
        hv = HV_FROM_DAQ.get((campaign, run, int(idx)), b["HV"])
        rfsel = b.get("RFSel", DEFAULT_RFSEL[campaign])
        boards.append(dict(board_idx=int(idx), chip=b["short"], role=b["role"], hv=float(hv),
                           offset=int(b["offset"]), angle=int(b.get("angle", 0)),
                           temp=_temp(b.get("temp")),
                           rfsel=None if rfsel is None else int(rfsel)))
    return boards
