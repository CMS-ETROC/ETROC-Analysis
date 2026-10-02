"""Rebuild the frontend tables from their sources.

    python -m etroc_plots.frontend.rebuild --out DIR [--part blnw|tdc ...] [--runs RUNS ...]

run from TestBeam/, writes into DIR (copy the files into the frontend/ folder of the inputs and
update the checksum list to use them). The parts:

  blnw   blnw_by_run.csv from the scan histories (BLNW_SQLITES) and the run metadata
         (RUN_METADATA_DIRS), every run unless --runs says otherwise
  tdc    tdc_pixel_hists.csv and tdc_cal_modes.csv from the merged hits (DECODED_DIRS), by
         default the runs of REST_PAIRS; every loop file of a run is read, about 1 s per file on
         one lxplus core (5.5 min for the four REST_PAIRS runs, 328 files)

--runs takes "pairs" (REST_PAIRS), "frontend" (frontend_runs()) or campaign:telescope:run items
such as july:h1:4; the TDC part always adds the July run of each telescope's REST_PAIRS pair,
whose most populated pixels it histograms. The progress and every run's hit counts (before the
ea cut, after it, after the CAL window) go to standard output.
"""
import argparse
import os
import sys
import time

import pandas as pd

from ..campaigns import active as campaign
from ..resolution import tables
from . import blnw, tdc_codes


def frontend_runs():
    """{(campaign, telescope): [run, ...]}: FRONTEND_RUNS plus the July display runs of
    DISPLAY_RUNS_JUL_CSV at FRONTEND_JULY_SETTINGS."""
    out = {}
    for camp, by_tel in campaign.FRONTEND_RUNS.items():
        for tel, runs in by_tel.items():
            out[(camp, tel)] = set(runs)
    for tel, steps in tables.display_runs(campaign.DISPLAY_RUNS_JUL_CSV,
                                          **campaign.FRONTEND_JULY_SETTINGS).items():
        for runs in steps.values():
            out.setdefault(("july", tel), set()).update(runs)
    return {k: sorted(v) for k, v in sorted(out.items())}


def pair_runs():
    out = {}
    for tel, (march, july) in campaign.REST_PAIRS.items():
        out[("march", tel)] = [march]
        out[("july", tel)] = [july]
    return out


def parse_runs(items):
    """--runs items -> {(campaign, telescope): [run, ...]}."""
    out = {}
    for item in items:
        if item == "pairs":
            got = pair_runs()
        elif item == "frontend":
            got = frontend_runs()
        else:
            camp, tel, run = item.split(":")
            got = {(camp, tel): [int(run)]}
        for k, v in got.items():
            out[k] = sorted(set(out.get(k, [])) | set(v))
    return out


def _log(text):
    print("%s  %s" % (time.strftime("%H:%M:%S"), text), flush=True)


def build_blnw(out, runs):
    df = blnw.build_blnw_table(os.path.join(out, "blnw_by_run.csv"), runs=runs)
    _log("blnw_by_run.csv: %d rows; match %s" % (len(df), df["match"].value_counts().to_dict()))


def build_tdc(out, runs):
    runs = dict(runs or pair_runs())
    for tel, (_, july) in campaign.REST_PAIRS.items():   # the pixel choice needs the pair's July run
        if any(t == tel for _, t in runs):
            runs[("july", tel)] = sorted(set(runs.get(("july", tel), [])) | {july})
    hist_rows, cal_rows = [], []
    for tel in campaign.TELESCOPE_CHIPS:
        todo = [(c, r) for (c, t), rs in runs.items() if t == tel for r in rs]
        if not todo:
            continue
        july = campaign.REST_PAIRS[tel][1]
        todo.sort(key=lambda cr: (cr != ("july", july), cr))    # the pair's July run first
        pixels = None
        for camp, run in todo:
            g = tdc_codes.gather(camp, tel, run, log=_log)
            if pixels is None:
                pixels = tdc_codes.busiest_pixels(g)
                _log("%s pixels (board: row, col) from July run %d: %s" % (tel, run, pixels))
            hist_rows += tdc_codes.hist_rows(g, camp, tel, run, pixels)
            cal_rows += tdc_codes.cal_rows(g, camp, tel, run)
            for b in range(tdc_codes.NBOARDS):
                a, e, w = (int(g[k][b].sum()) for k in ("n_hits_all", "n_hits_ea0", "n_hits"))
                _log("%s %s run %d board %d: %d files, hits %d, ea == 0 %d (removed %d), "
                     "CAL window %d" % (camp, tel, run, b, g["n_files"], a, e, a - e, w))
    pd.DataFrame(hist_rows, columns=tdc_codes.TDC_HISTS_COLUMNS).to_csv(
        os.path.join(out, "tdc_pixel_hists.csv"), index=False)
    pd.DataFrame(cal_rows, columns=tdc_codes.TDC_CAL_COLUMNS).to_csv(
        os.path.join(out, "tdc_cal_modes.csv"), index=False, float_format="%.6g")
    _log("tdc_pixel_hists.csv: %d rows; tdc_cal_modes.csv: %d rows" % (len(hist_rows),
                                                                       len(cal_rows)))


PARTS = {"blnw": build_blnw, "tdc": build_tdc}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m etroc_plots.frontend.rebuild",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="folder the tables are written to")
    ap.add_argument("--part", action="append", choices=sorted(PARTS),
                    help="the tables to build (repeatable; default every part)")
    ap.add_argument("--runs", nargs="+", default=None,
                    help='"pairs", "frontend" or campaign:telescope:run items')
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    runs = parse_runs(a.runs) if a.runs else None
    for part in a.part or sorted(PARTS):
        _log("part %s, runs %s" % (part, runs or "default"))
        PARTS[part](a.out, runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
