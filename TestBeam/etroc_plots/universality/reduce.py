"""Reduce the TWC-transfer study's outputs to the six tables the universality notebook reads.

    python -m etroc_plots.universality.reduce --out DIR --results-root S [--twc-root R]
                                              [--part cells|master|axes|bank ...]

run from TestBeam/ where the study's outputs can be read (lxplus: EOS and AFS), writes into DIR
(copy the files into the universality/ folder of the inputs and update the checksum list to use
them). The study asked whether a time-walk correction (TWC) fitted on one source (a run, a board,
a pixel) can be applied to another target without losing time resolution; each comparison it ran
is a "cell". S is the study's folder (also ETROC_TWC_RESULTS_ROOT; no default), which must hold
condor/ (the seven job lists below), results_r2/axis_{A,B,C,D,E,I,J,K}.csv and results_r3/
(axis_F.csv, axis_F_long.csv, summary_F.md). R (default TWC_TRANSFER_ROOT) holds the coefficient
bank, bank/coeffs_<campaign>_<run>.parquet, and the round-1 table of the level cells,
figs/transfer_2026-09-04/inputs/master_cells.csv. The parts:

  cells   univ_cells.csv, one row per cell, from the job lists that ran them, in the order
          cells.tsv, cells_stage1.tsv, cells_stage2.tsv, cells_stage2_held.tsv, cells_stage3.tsv,
          cells_rerun.tsv, cells_guardfix_rerun.tsv: a later list that names a cell again redefines
          it (the reruns). The runs come from each job's command; base_cell, for the axis-F
          cells, from the cell table of results_r3/summary_F.md; dbias_v, for the level-h cells,
          from runs_summary.csv. Checks that every cell of the other tables is in it. Seconds.
  master  univ_master_cells.csv, the level-cell table, copied byte for byte. Seconds.
  axes    univ_axis_r2.csv (the eight results_r2 axis tables, one header, rows in axis order, each
          line copied unchanged), univ_axis_F_wide.csv and univ_axis_F_long.csv (copies of
          results_r3/axis_F.csv and axis_F_long.csv). Seconds.
  bank    twc_bank_best.csv from the 63 bank files (102 MB): per (campaign, run, board_id, row,
          col) the fit with the most events (BEST_FILTER, then the first row of highest n_fit, the
          study's own tie-break). 12 s on one lxplus core (2026-10-02), 10.5 MB out.

Each part prints its row counts. The bank's hv, temp and angle_deg columns are empty in every
file (checked); the table leaves them out, and the figures take a run's conditions from the
campaign modules.
"""
import argparse
import csv
import glob
import os
import re
import shlex
import shutil
import sys
import time

import pandas as pd

from ..campaigns import active as campaign

# the job lists, in the order they ran: a later list that names a cell again redefines it
TSVS = ("cells.tsv", "cells_stage1.tsv", "cells_stage2.tsv", "cells_stage2_held.tsv",
        "cells_stage3.tsv", "cells_rerun.tsv", "cells_guardfix_rerun.tsv")
TSV_FIELDS = ("job_id", "cell_id", "level", "group", "est_min", "mem_mb", "command")
AXES_R2 = ("A", "B", "C", "D", "E", "I", "J", "K")
CELLS_COLUMNS = ["cell", "level_or_axis", "src_campaign", "src_run", "tgt_campaign", "tgt_run",
                 "base_cell", "dbias_v"]
# the bank's campaign codes: "desy" = DESY May 2026, "aug" = DESY Aug 2026, "mar" / "jul" = CERN
# IRRAD March / July 2026 (runs "h1_run19", "f1_run5"; the DESY runs are "run26")
IRRAD_CODES = {"mar": "march", "jul": "july"}
# the fits a figure may draw: the whole run (fold full), the full guard, not a placebo, fitted on
# the tracks of the pixel itself
BEST_FILTER = dict(fold="full", guard="full", placebo=0, src_kind="track")
BEST_KEYS = ["board_id", "row", "col"]
BANK_COLUMNS = ["campaign", "run", "telescope", "board_id", "role", "row", "col", "combo",
                "src_triple", "n_fit", "c2", "c1", "c0", "q05", "q95", "tot_min", "tot_max",
                "cal_mean", "hist_lo", "hist"]
BANK_EMPTY = ("hv", "temp", "angle_deg")       # empty in every bank file (checked)
MASTER_REL = os.path.join("figs", "transfer_2026-09-04", "inputs", "master_cells.csv")


def _log(text):
    print("%s  %s" % (time.strftime("%H:%M:%S"), text), flush=True)


def _results_paths(root):
    """The files of the results folder the reduction reads; stops naming every one missing."""
    paths = {"tsv:" + n: os.path.join(root, "condor", n) for n in TSVS}
    paths.update({"axis:" + a: os.path.join(root, "results_r2", "axis_%s.csv" % a)
                  for a in AXES_R2})
    paths.update({"F_wide": os.path.join(root, "results_r3", "axis_F.csv"),
                  "F_long": os.path.join(root, "results_r3", "axis_F_long.csv"),
                  "summary_F": os.path.join(root, "results_r3", "summary_F.md")})
    missing = [p for p in paths.values() if not os.path.isfile(p)]
    if missing:
        raise SystemExit("--results-root %s lacks %d of the files the reduction reads (it must "
                         "hold condor/ with the seven job lists, results_r2/ with the eight axis "
                         "tables, results_r3/ with axis_F.csv, axis_F_long.csv and summary_F.md):"
                         "\n  %s" % (root, len(missing), "\n  ".join(missing)))
    return paths


# ------------------------------------------------------------------------------------- cells
def _flag(tokens, name):
    if name in tokens:
        return tokens[tokens.index(name) + 1]
    return None


def cell_runs(command):
    """(src_campaign, src_run, tgt_campaign, tgt_run) of one job command. A "cell" job names the
    target with --campaign/--run and the source with --src-campaign/--src-run, each defaulting to
    the target's; a pooled cell (--mode e) has no single source run (src_run None). A "tier0" job
    reads one bank file (--bank .../coeffs_<campaign>_<run>.parquet), source and target alike."""
    tok = shlex.split(command)
    if tok[0] == "tier0":
        m = re.search(r"coeffs_([a-z0-9]+)_(.+)\.parquet$", _flag(tok, "--bank"))
        return m.group(1), m.group(2), m.group(1), m.group(2)
    if tok[0] != "cell":
        raise ValueError("job command %r is neither cell nor tier0" % command[:60])
    camp, run = _flag(tok, "--campaign"), _flag(tok, "--run")
    src_camp = _flag(tok, "--src-campaign") or camp
    src_run = None if _flag(tok, "--mode") == "e" else (_flag(tok, "--src-run") or run)
    return src_camp, src_run, camp, run


def read_tsv(path):
    """{cell: dict(level_or_axis, src_campaign, src_run, tgt_campaign, tgt_run)} of one job
    list. Smoke-test jobs are skipped; the chunks of one cell must agree."""
    out = {}
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            if parts[0] == "job_id":
                continue          # header line (cells_guardfix_rerun.tsv has none)
            if len(parts) != len(TSV_FIELDS):
                raise ValueError("%s line %d: %d fields, expected %d"
                                 % (path, n, len(parts), len(TSV_FIELDS)))
            job = dict(zip(TSV_FIELDS, parts))
            if job["level"] == "smoke":
                continue
            row = dict(zip(("src_campaign", "src_run", "tgt_campaign", "tgt_run"),
                           cell_runs(job["command"])), level_or_axis=job["level"])
            old = out.setdefault(job["cell_id"], row)
            if old != row:
                raise ValueError("%s: the jobs of cell %s disagree: %s / %s"
                                 % (path, job["cell_id"], old, row))
    return out


def axis_f_base_cells(summary_md, cells):
    """{F cell: round-1 cell} from the table "Cells and what each compares" of summary_F.md, whose
    rows read "| F4 | b1, run26<-run9 ... |": the first word names the round-1 cell, matched to
    the one cell of `cells` that starts with it ("b1" -> "b1_30to0")."""
    text = open(summary_md).read()
    sec = text.split("## Cells and what each compares", 1)[1].split("\n## ", 1)[0]
    by_num = {}
    for m in re.finditer(r"^\|\s*(F\d+)\s*\|\s*([a-z]+\d+),", sec, re.M):
        hits = [c for c in cells if c.startswith(m.group(2) + "_")]
        if len(hits) != 1:
            raise ValueError("summary_F.md %s names %s: %d cells start with it %s"
                             % (m.group(1), m.group(2), len(hits), hits))
        by_num[m.group(1)] = hits[0]
    out = {}
    for c in cells:
        m = re.match(r"(F\d+)_", c)
        if m:
            if m.group(1) not in by_num:
                raise ValueError("axis-F cell %s is not in the cell table of summary_F.md" % c)
            out[c] = by_num[m.group(1)]
    return out


def run_bias(rs, code, run):
    """The bias of an IRRAD run: the median over its boards in runs_summary (bank code "mar" /
    "jul", run "h1_run19")."""
    tel, num = run.split("_run")
    r = rs[(rs["campaign"] == campaign.TABLE_CAMPAIGN[IRRAD_CODES[code]])
           & (rs["telescope"] == campaign.TABLE_TEL[tel]) & (rs["run"] == int(num))]
    if len(r) != 1:
        raise ValueError("%d runs_summary rows for %s %s" % (len(r), code, run))
    hv = r[[c for c in rs.columns if re.fullmatch(r"board\d+_hv", c)]].iloc[0].dropna()
    return float(hv.median())


def build_cells(out, paths, twc_root):
    union, per_tsv = {}, {}
    for name in TSVS:
        got = read_tsv(paths["tsv:" + name])
        per_tsv[name] = len(got)
        redefined = sorted(c for c in got if c in union and union[c] != got[c])
        union.update(got)
        _log("%s: %d cells%s" % (name, len(got), ", redefines %s" % redefined if redefined
                                 else ""))
    base = axis_f_base_cells(paths["summary_F"], sorted(union))
    rs = pd.read_csv(campaign.RUNS_SUMMARY_CSV)
    rows = []
    for cell in sorted(union):
        r = dict(union[cell], cell=cell, base_cell=base.get(cell), dbias_v=None)
        if r["level_or_axis"] == "h":        # March -> July at 1.5e15: the bias step between them
            r["dbias_v"] = (run_bias(rs, r["tgt_campaign"], r["tgt_run"])
                            - run_bias(rs, r["src_campaign"], r["src_run"]))
        rows.append(r)
    df = pd.DataFrame(rows, columns=CELLS_COLUMNS)
    # every cell the result tables name is one row here
    named = {"master": pd.read_csv(os.path.join(twc_root, MASTER_REL)).cell}
    named.update({"axis_" + a: pd.read_csv(paths["axis:" + a]).cell for a in AXES_R2})
    named.update({"axis_F_wide": pd.read_csv(paths["F_wide"]).cell,
                  "axis_F_long": pd.read_csv(paths["F_long"]).cell})
    unresolved = {k: sorted(set(v) - set(df.cell)) for k, v in named.items()}
    unresolved = {k: v for k, v in unresolved.items() if v}
    if unresolved or df.cell.duplicated().any():
        raise SystemExit("cells the result tables name but univ_cells lacks: %s" % unresolved)
    df.to_csv(os.path.join(out, "univ_cells.csv"), index=False)
    _log("univ_cells.csv: %d rows (per job list %s); %d axis-F base cells, %d level-h bias steps;"
         " every cell of the master and axis tables resolves (%d names)"
         % (len(df), per_tsv, df.base_cell.notna().sum(), df.dbias_v.notna().sum(),
            sum(v.nunique() for v in named.values())))


# ------------------------------------------------------------------------------- master, axes
def _data_rows(path):
    with open(path, newline="") as fh:
        return sum(1 for _ in csv.reader(fh)) - 1


def build_master(out, twc_root):
    src = os.path.join(twc_root, MASTER_REL)
    dst = os.path.join(out, "univ_master_cells.csv")
    shutil.copyfile(src, dst)
    _log("univ_master_cells.csv: %d rows (copy of %s)" % (_data_rows(dst), src))


def build_axes(out, paths):
    header, lines, counts = None, [], {}
    for a in AXES_R2:
        p = paths["axis:" + a]
        text = open(p, newline="").read().splitlines()
        if header is None:
            header = text[0]
        elif text[0] != header:
            raise SystemExit("%s has another header than axis_A.csv" % p)
        if not header.startswith("axis,"):
            raise SystemExit("%s: the axis tables are expected to start with an axis column" % p)
        if _data_rows(p) != len(text) - 1:
            raise SystemExit("%s: a field spans lines; copy it with a CSV writer instead" % p)
        lines += text[1:]
        counts[a] = len(text) - 1
    dst = os.path.join(out, "univ_axis_r2.csv")
    with open(dst, "w", newline="") as fh:
        fh.write("\n".join([header] + lines) + "\n")
    _log("univ_axis_r2.csv: %d rows (per axis %s)" % (_data_rows(dst), counts))
    for key, name in (("F_wide", "univ_axis_F_wide.csv"), ("F_long", "univ_axis_F_long.csv")):
        dst = os.path.join(out, name)
        shutil.copyfile(paths[key], dst)
        _log("%s: %d rows (copy of %s)" % (name, _data_rows(dst), paths[key]))


# --------------------------------------------------------------------------------------- bank
def best_rows(df):
    """One row per (board_id, row, col) of one bank file: of the fits passing BEST_FILTER, the one
    with the most events (n_fit), the first such row on a tie."""
    sub = df
    for k, v in BEST_FILTER.items():
        sub = sub[sub[k] == v]
    idx = sub.groupby(BEST_KEYS)["n_fit"].idxmax()
    return sub.loc[idx].sort_values(BEST_KEYS)


def build_bank(out, twc_root):
    files = sorted(glob.glob(os.path.join(twc_root, "bank", "coeffs_*.parquet")))
    if not files:
        raise SystemExit("no coeffs_*.parquet in %s" % os.path.join(twc_root, "bank"))
    frames, per_file = [], {}
    for p in files:
        df = pd.read_parquet(p)
        full = df[list(BANK_EMPTY)].notna().sum()
        if full.any():
            raise SystemExit("%s: %s not empty; the table would drop them" % (p, full.to_dict()))
        if (df["telescope"].fillna("") != "").any():
            raise SystemExit("%s: the telescope column is filled; the table derives it" % p)
        if df[["campaign", "run"]].drop_duplicates().shape[0] != 1:
            raise SystemExit("%s holds more than one (campaign, run)" % p)
        best = best_rows(df).copy()
        camp, run = df["campaign"].iloc[0], df["run"].iloc[0]
        if os.path.basename(p) != "coeffs_%s_%s.parquet" % (camp, run):
            raise SystemExit("%s holds campaign %s run %s" % (p, camp, run))
        best["telescope"] = run.split("_run")[0] if "_run" in run else ""
        frames.append(best[BANK_COLUMNS])
        key = camp + ("_" + best["telescope"].iloc[0] if best["telescope"].iloc[0] else "")
        per_file.setdefault(key, []).append(len(best))
        _log("%s: %d rows, %d best pixels" % (os.path.basename(p), len(df), len(best)))
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(os.path.join(out, "twc_bank_best.csv"), index=False)
    _log("twc_bank_best.csv: %d rows, %d (campaign, run) groups; files and rows per campaign: %s;"
         " hv, temp and angle_deg empty in all %d files"
         % (len(table), table[["campaign", "run"]].drop_duplicates().shape[0],
            {k: (len(v), sum(v)) for k, v in sorted(per_file.items())}, len(files)))


PARTS = ("cells", "master", "axes", "bank")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m etroc_plots.universality.reduce",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="folder the tables are written to")
    ap.add_argument("--results-root", default=campaign.TWC_RESULTS_ROOT,
                    help="the study's folder (condor/, results_r2/, results_r3/); default "
                         "ETROC_TWC_RESULTS_ROOT")
    ap.add_argument("--twc-root", default=campaign.TWC_TRANSFER_ROOT,
                    help="the folder of bank/ and figs/ (default TWC_TRANSFER_ROOT)")
    ap.add_argument("--part", action="append", choices=PARTS,
                    help="the tables to build (repeatable; default every part)")
    a = ap.parse_args(argv)
    parts = a.part or list(PARTS)
    paths = None
    if {"cells", "axes"} & set(parts):
        if not a.results_root:
            ap.error("--results-root (or ETROC_TWC_RESULTS_ROOT) is required: the study's folder, "
                     "holding condor/ with the seven job lists, results_r2/ with the axis tables "
                     "and results_r3/ with axis_F.csv, axis_F_long.csv and summary_F.md")
        paths = _results_paths(a.results_root)
    os.makedirs(a.out, exist_ok=True)
    for part in parts:
        t0 = time.time()
        _log("part %s" % part)
        if part == "cells":
            build_cells(a.out, paths, a.twc_root)
        elif part == "master":
            build_master(a.out, a.twc_root)
        elif part == "axes":
            build_axes(a.out, paths)
        else:
            build_bank(a.out, a.twc_root)
        _log("part %s done in %.0f s" % (part, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
