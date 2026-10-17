"""Rebuild the frontend tables from their sources.

    python -m etroc_plots.frontend.rebuild --out DIR [--part blnw|tdc|tracks ...] [--runs RUNS ...]
                                           [--rank rows|bytes]

run from TestBeam/, writes into DIR (copy the files into the frontend/ folder of the inputs and
update the checksum list to use them). The parts:

  blnw    blnw_by_run.csv from the scan histories (BLNW_SQLITES) and the run metadata
          (RUN_METADATA_DIRS), every run unless --runs says otherwise
  tdc     tdc_pixel_hists.csv and tdc_cal_modes.csv from the merged hits (DECODED_DIRS), by
          default the runs of COOLDOWN_PAIRS; every loop file of a run is read, about 1 s per
          file on one lxplus core (5.5 min for the four COOLDOWN_PAIRS runs, 328 files)
  tracks  triples.csv, tot_pixel_medians.csv, tot_pooled_summary.csv, tot_pooled_hist.csv,
          dtoa_triples.csv, dtoa_pooled_hist.csv, pixel_hists.csv and rfsel_pair_hist.csv from
          the track files (ANALYSIS_DIRS; frontend/tot_shapes.py), by default the runs of
          frontend_runs(); about two minutes per run on one lxplus core. The fixed pixels of
          pixel_hists.csv are chosen over the runs of tot_shapes.shape_runs() that are in the run
          set, the RFSel histograms are made when every run of RFSEL_TOT_PAIRS is in it, and the
          time-control groups of TOT_TIME_CONTROL, on the board chosen over the RFSel pairs, when
          their runs are in it too

--runs takes "pairs" (COOLDOWN_PAIRS), "frontend" (frontend_runs()) or campaign:telescope:run
items such as july:h1:4; the TDC part always adds the July run of each telescope's COOLDOWN_PAIRS
pair, whose most populated pixels it histograms. --rank picks a pixel's triple by its events (rows,
the default) or by its file size (bytes, the rule of the CERN IRRAD 2026 talk figures, for
comparisons). The progress and every run's counts go to standard output.
"""
import argparse
import os
import sys
import time

import pandas as pd

from ..campaigns import active as campaign
from ..resolution import tables
from . import blnw, tdc_codes, tot_shapes


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
    for tel, (march, july) in campaign.COOLDOWN_PAIRS.items():
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
    # the pixel choice needs the pair's July run
    for tel, (_, july) in campaign.COOLDOWN_PAIRS.items():
        if any(t == tel for _, t in runs):
            runs[("july", tel)] = sorted(set(runs.get(("july", tel), [])) | {july})
    hist_rows, cal_rows = [], []
    for tel in campaign.TELESCOPE_CHIPS:
        todo = [(c, r) for (c, t), rs in runs.items() if t == tel for r in rs]
        if not todo:
            continue
        july = campaign.COOLDOWN_PAIRS[tel][1]
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


def build_tracks(out, runs, rank="rows"):
    runs = runs or frontend_runs()
    tri, res = [], {}
    for (camp, tel), rs in sorted(runs.items()):
        for run in rs:
            t0 = time.time()
            t = tot_shapes.scan_run(camp, tel, run)
            for k, v in tot_shapes.run_tables(camp, tel, run, t, rank).items():
                res.setdefault(k, []).extend(v)
            tri.append(t)
            _log("%s %s run %d: %d track files, %d pixel rows, %d dTOA triples, %.0f s"
                 % (camp, tel, run, len(t), sum(1 for r in res["pixel"] if r["run"] == run
                                                and r["tel"] == tel and r["campaign"] == camp),
                    sum(1 for r in res["dtoa"] if r["run"] == run and r["tel"] == tel
                        and r["campaign"] == camp), time.time() - t0))
    tri = pd.concat(tri, ignore_index=True)
    hist_rows = []
    for tel in campaign.TELESCOPE_CHIPS:
        have = [k for k in tot_shapes.shape_runs(tel) if k[1] in runs.get((k[0], tel), [])]
        if len(have) < len(tot_shapes.shape_runs(tel)):
            _log("%s: fixed pixels over %d of the %d shape runs (the run set lacks the others)"
                 % (tel, len(have), len(tot_shapes.shape_runs(tel))))
        if not have or not any(k[0] == "july" for k in have):
            continue
        ch = tot_shapes.choose(tri, tel, rank, runs=have)
        _log("%s fixed pixels: %s; pair 2-3 board 2 %s (%d candidates); July board %d %s"
             % (tel, {b: p[0] for b, p in ch["pixels"].items()},
                ch["pairs"][(2, 3)]["pixels"][2], ch["pairs"][(2, 3)]["n_candidates"],
                ch["july"]["board"], ch["july"]["pixel"]))
        hist_rows += tot_shapes.pixel_hist_rows(ch, log=_log)
    rfsel_rows, rfsel_boards = [], {}
    rs = pd.read_csv(campaign.RUNS_SUMMARY_CSV)
    for (camp, tel), pairs in campaign.RFSEL_TOT_PAIRS.items():
        if all(r in runs.get((camp, tel), []) for p in pairs for r in p):
            b, totals = tot_shapes.rfsel_board(tri, camp, tel, pairs, rank)
            _log("RFSel pairs %s %s: board %d (%s), events above the floor per board and run %s"
                 % (camp, tel, b, campaign.TELESCOPE_CHIPS[tel][b], totals))
            rfsel_rows += tot_shapes.rfsel_rows(tri, rs, camp, tel, pairs, b, rank, log=_log)
            rfsel_boards[(camp, tel)] = b
        else:
            _log("RFSel pairs %s %s: runs not all in the run set, skipped" % (camp, tel))
    for (camp, tel), groups in campaign.TOT_TIME_CONTROL.items():
        if ((camp, tel) in rfsel_boards
                and all(r in runs.get((camp, tel), []) for g in groups for r in g)):
            rfsel_rows += tot_shapes.rfsel_rows(tri, rs, camp, tel, groups,
                                                rfsel_boards[(camp, tel)], rank, log=_log)
        else:
            _log("time control %s %s: no RFSel pairs or runs not all in the run set, skipped"
                 % (camp, tel))
    for name, rows, cols in (
            ("triples.csv", tri, tot_shapes.TRIPLES_COLUMNS),
            ("tot_pixel_medians.csv", res.get("pixel", []), tot_shapes.PIXEL_COLUMNS),
            ("tot_pooled_summary.csv", res.get("summary", []), tot_shapes.SUMMARY_COLUMNS),
            ("tot_pooled_hist.csv", res.get("tot_hist", []), tot_shapes.TOT_HIST_COLUMNS),
            ("dtoa_triples.csv", res.get("dtoa", []), tot_shapes.DTOA_COLUMNS),
            ("dtoa_pooled_hist.csv", res.get("dtoa_hist", []), tot_shapes.DTOA_HIST_COLUMNS),
            ("pixel_hists.csv", hist_rows, tot_shapes.PIXEL_HISTS_COLUMNS),
            ("rfsel_pair_hist.csv", rfsel_rows, tot_shapes.RFSEL_COLUMNS)):
        df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, columns=cols)
        for c in df.columns:
            if ((c.startswith(("row_", "col_", "board_")) or c in ("partner_run", "cal_mode"))
                    and df[c].dtype != object):
                df[c] = df[c].astype("Int64")
        df[cols].to_csv(os.path.join(out, name), index=False)
        _log("%s: %d rows" % (name, len(df)))


PARTS = {"blnw": build_blnw, "tdc": build_tdc, "tracks": build_tracks}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m etroc_plots.frontend.rebuild",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="folder the tables are written to")
    ap.add_argument("--part", action="append", choices=sorted(PARTS),
                    help="the tables to build (repeatable; default every part)")
    ap.add_argument("--runs", nargs="+", default=None,
                    help='"pairs", "frontend" or campaign:telescope:run items')
    ap.add_argument("--rank", choices=sorted(tot_shapes.RANK_KEYS), default="rows",
                    help="a pixel's triple: the most events (rows) or the largest file (bytes)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    runs = parse_runs(a.runs) if a.runs else None
    for part in a.part or sorted(PARTS):
        _log("part %s, runs %s" % (part, runs or "default"))
        if part == "tracks":
            PARTS[part](a.out, runs, a.rank)
        else:
            PARTS[part](a.out, runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
