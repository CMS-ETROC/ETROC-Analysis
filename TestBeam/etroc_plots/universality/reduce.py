"""Reduce the TWC-transfer study's outputs to the six tables the universality notebook reads.

    python -m etroc_plots.universality.reduce --out DIR [--results-root S] [--twc-root R]
                                              [--part cells|master|axes|bank ...]

run from TestBeam/ where the study's outputs can be read (lxplus: EOS and AFS), writes into DIR
(copy the files into the universality/ folder of the inputs and update the checksum list to use
them). The study asked whether a time-walk correction (TWC) fitted on one source (a run, a board,
a pixel) can be applied to another target without losing time resolution; each comparison it ran
is a "cell". S is the study's folder (default TWC_RESULTS_ROOT: the copy of these files in
universality/results/ of the inputs, or ETROC_TWC_RESULTS_ROOT), which must hold condor/ (the
seven job lists below), the axis tables campaign.UNIV_AXIS_SOURCES names
(results_r2/axis_{A,B,C,D,E,I,J,K}.csv and results_r3/guardfix/axis_{E,J,K}_gf.csv) and
results_r3/ (axis_F.csv, axis_F_long.csv, summary_F.md). R (default TWC_TRANSFER_ROOT) holds the
coefficient bank, bank/coeffs_<campaign>_<run>.parquet, the first summary table of the level cells,
figs/transfer_2026-09-04/inputs/master_cells.csv, and the aggregation tables the axis tables were
built from, tables/agg_<agg_tag>/agg_by_cell.csv (campaign.UNIV_AGG_DIRS names the folders that
hold a tag under another name). The parts:

  cells   univ_cells.csv, one row per cell, from the job lists that ran them, in the order
          cells.tsv, cells_stage1.tsv, cells_stage2.tsv, cells_stage2_held.tsv, cells_stage3.tsv,
          cells_rerun.tsv, cells_guardfix_rerun.tsv: a later list that names a cell again redefines
          it (the reruns). The runs come from each job's command; base_cell, for the axis-F
          cells, from the cell table of results_r3/summary_F.md; dbias_v, for the level-h cells,
          from runs_summary.csv. Checks that every cell of the other tables is in it. Seconds.
  master  univ_master_cells.csv, the level-cell table, copied byte for byte. Seconds.
  axes    univ_axis_r2.csv (the eight axes, one header, rows in axis order; per axis the files
          of UNIV_AXIS_SOURCES, a later file replacing every row of the cells it holds: the
          guard-fix rerun's rows of 11 cells of axes E, J and K), univ_axis_F_long.csv and
          univ_axis_F_wide.csv (from results_r3/axis_F_long.csv and axis_F.csv): each line
          copied unchanged (the columns a later file adds dropped), with the study's
          self_transfer flag appended (the wide table, whose row holds several arms side by side,
          gets one per arm, <arm>_self_transfer). An axis_F.csv that carries the five
          <arm>_self_transfer columns itself (the one twc_transfer's axes command writes, each
          after its <arm>_present) has every value of them checked against the flag of the
          join, and stops naming each arm entry where the two differ; its lines are copied
          without those columns and the flags appended, so the table has them once, at the end,
          as for the study's axis_F.csv. Seconds.
  bank    twc_bank_best.csv from the 63 bank files (102 MB): per (campaign, run, board_id, row,
          col) the fit with the most events (BEST_FILTER, then the first row of highest n_fit, the
          study's own tie-break). 12 s on one lxplus core (2026-10-02), 10.5 MB out.

self_transfer is 1 on a row whose source triple is its target triple (the same run, combination
and pixels): the study aggregates such rows as their own group, leaves them out of its per-cell
summaries, and its axis tables carry them with no column that marks them. The flag is recovered
by the study's own join: each row matches exactly one row of its aggregation table on (cell, arm,
kind, name, family, agg_tag, n_pix), with the same mu and se, or the reduction stops naming the
rows that do not. The join gives the flag of every table read, the wide axis-F table's flag
columns included when it carries them (they are checked, not taken).

Each part prints its row counts. The bank's hv, temp and angle_deg columns are empty in every
file (checked); the table leaves them out, and the figures take a run's conditions from the
campaign modules.
"""
import argparse
import csv
import glob
import io
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
# the study's join of an axis row to its aggregation row, in the axis tables' words (the
# aggregation table says name for board_or_role and extrap_family for family)
JOIN_KEYS = ["cell", "arm", "kind", "board_or_role", "family", "agg_tag", "n_pix"]
AGG_COLUMNS = ["cell", "arm", "kind", "name", "extrap_family", "n_pix", "self_transfer", "mu",
               "se"]
WIDE_ARMS = ("xfer", "xfer_lin", "xfer_lin_lpo", "xfer_clip", "xfer_rescale")   # axis_F.csv
WIDE_FLAGS = tuple("%s_self_transfer" % arm for arm in WIDE_ARMS)


def _log(text):
    print("%s  %s" % (time.strftime("%H:%M:%S"), text), flush=True)


def _results_paths(root):
    """The files of the results folder the reduction reads; stops naming every one missing."""
    paths = {"tsv:" + n: os.path.join(root, "condor", n) for n in TSVS}
    paths.update({"F_wide": os.path.join(root, "results_r3", "axis_F.csv"),
                  "F_long": os.path.join(root, "results_r3", "axis_F_long.csv"),
                  "summary_F": os.path.join(root, "results_r3", "summary_F.md")})
    files = list(paths.values())
    for a, rels in campaign.UNIV_AXIS_SOURCES.items():
        paths["axis:" + a] = [os.path.join(root, *rel.split("/")) for rel in rels]
        files += paths["axis:" + a]
    missing = [p for p in files if not os.path.isfile(p)]
    if missing:
        raise SystemExit("--results-root %s lacks %d of the files the reduction reads (it must "
                         "hold condor/ with the seven job lists, the axis tables of "
                         "UNIV_AXIS_SOURCES, results_r3/ with axis_F.csv, axis_F_long.csv and "
                         "summary_F.md):\n  %s" % (root, len(missing), "\n  ".join(missing)))
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
    """{F cell: level cell} from the table "Cells and what each compares" of summary_F.md, whose
    rows read "| F4 | b1, run26<-run9 ... |": the first word names the level cell, matched to
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
    named.update({"axis_" + a: axis_table(paths["axis:" + a])[2].cell
                  for a in campaign.UNIV_AXIS_SOURCES})
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


def table_lines(path):
    """(header, data lines, frame) of one axis table: its lines as written, and the same rows
    parsed, the floats read back exactly as written. Stops when a field spans lines or a line is
    blank (the lines and the rows would not pair up)."""
    text = open(path, newline="").read().splitlines()
    df = pd.read_csv(path, float_precision="round_trip")
    if _data_rows(path) != len(text) - 1 or len(df) != len(text) - 1:
        raise SystemExit("%s: a field spans lines or a line is blank; copy it with a CSV writer "
                         "instead" % path)
    return text[0], text[1:], df


def axis_table(files):
    """(header, data lines, frame, cells replaced) of one axis: the table of files[0], each later
    file replacing every row of the cells it holds, at the place of that cell's first row (a cell
    the earlier files lack goes at the end). A later file has the first file's columns first; the
    columns it adds are dropped, each line cut after the first file's last column. The frame is
    the composed lines parsed, the floats read back exactly as written."""
    header, lines, _ = table_lines(files[0])
    n = len(next(csv.reader([header])))
    replaced = {}
    for path in files[1:]:
        head, body, _ = table_lines(path)
        if not head.startswith(header + ","):
            raise SystemExit("%s: its columns do not start with those of %s" % (path, files[0]))
        new = {}
        for line in body:
            fields = next(csv.reader([line]))
            cut = _csv_line(fields[:n])
            if not line.startswith(cut + ","):
                raise SystemExit("%s: a line does not start with its first %d fields as a CSV "
                                 "writer writes them: %s" % (path, n, line))
            new.setdefault(fields[1], []).append(cut)
        cells = [next(csv.reader([line]))[1] for line in lines]
        composed, done = [], set()
        for cell, line in zip(cells, lines):
            if cell not in new:
                composed.append(line)
            elif cell not in done:
                composed += new[cell]
                done.add(cell)
        composed += [line for cell in new if cell not in done for line in new[cell]]
        replaced.update({c: (cells.count(c), len(new[c])) for c in new})
        lines = composed
    if not header.startswith("axis,cell,"):
        raise SystemExit("%s: the axis tables are expected to start with the columns axis, cell"
                         % files[0])
    df = pd.read_csv(io.StringIO("\n".join([header] + lines) + "\n"),
                     float_precision="round_trip")
    if len(df) != len(lines):
        raise SystemExit("%s: the composed lines do not parse to one row each" % files[0])
    return header, lines, df, replaced


def _csv_line(fields):
    buf = io.StringIO()
    csv.writer(buf, lineterminator="").writerow(fields)
    return buf.getvalue()


def agg_rows(twc_root, tags):
    """The aggregation rows behind the axis tables' `tags` (their agg_tag values): agg_by_cell.csv
    of each tag's folder under twc_root/tables (campaign.UNIV_AGG_DIRS, else agg_<tag>), in the
    axis tables' words (board_or_role, family), with agg_tag. They are read with pandas' default
    float parser, as the study's axis-table code read them, so mu and se equal the axis tables'
    numbers exactly."""
    frames, missing = [], []
    for tag in sorted(set(tags)):
        p = os.path.join(twc_root, "tables", campaign.UNIV_AGG_DIRS.get(tag, "agg_" + tag),
                         "agg_by_cell.csv")
        if not os.path.isfile(p):
            missing.append(p)
            continue
        d = pd.read_csv(p, usecols=AGG_COLUMNS).rename(
            columns={"name": "board_or_role", "extrap_family": "family"})
        d["agg_tag"] = tag
        frames.append(d)
    if missing:
        raise SystemExit("%d aggregation tables are missing:\n  %s"
                         % (len(missing), "\n  ".join(missing)))
    out = pd.concat(frames, ignore_index=True)
    out["n_pix"] = out["n_pix"].astype(int)
    return out


def self_transfer_flags(rows, agg, what):
    """The self_transfer flag (0 or 1) of each row of `rows` (columns JOIN_KEYS, mu_ps and se_ps),
    in order: that of the one row of `agg` (agg_rows) with the same JOIN_KEYS, whose mu and se
    equal the row's (nan equals nan). Stops naming every row of `what` that matches no
    aggregation row, more than one, or one with other numbers."""
    r = rows[JOIN_KEYS + ["mu_ps", "se_ps"]].reset_index(drop=True)
    r["n_pix"] = r["n_pix"].astype(int)
    r["_row"] = r.index
    m = r.merge(agg, on=JOIN_KEYS, how="left", indicator=True)
    m["_hit"] = m["_merge"] == "both"
    m["_same"] = (m["_hit"]
                  & ((m.mu_ps == m.mu) | (m.mu_ps.isna() & m.mu.isna()))
                  & ((m.se_ps == m.se) | (m.se_ps.isna() & m.se.isna())))
    per = m.groupby("_row")[["_hit", "_same"]].sum()
    bad = per[(per["_hit"] != 1) | (per["_same"] != 1)]
    if len(bad):
        raise SystemExit("%s: %d rows do not join exactly one aggregation row with the same mu and "
                         "se on (%s):\n  %s" % (what, len(bad), ", ".join(JOIN_KEYS), "\n  ".join(
                             "%s: %d rows on the key, %d with the same mu and se" % (
                                 "/".join(str(r.loc[i, k]) for k in JOIN_KEYS),
                                 per.loc[i, "_hit"], per.loc[i, "_same"]) for i in bad.index)))
    flags = m[m["_hit"]].set_index("_row")["self_transfer"].sort_index()
    if not flags.isin([0, 1]).all():
        raise SystemExit("%s: self_transfer values other than 0 and 1: %s"
                         % (what, sorted(flags.unique())))
    return flags.astype(int).tolist()


def wide_self_transfer(header, lines, frame, agg):
    """(header, lines, flags, n_held) of the wide axis-F table (table_lines gives header, lines
    and frame). flags is {<arm>_self_transfer: one value per line}: the flag of the row the arm
    holds, by the join (self_transfer_flags), and empty where the arm holds none (<arm>_present
    not True); n_held counts the arm entries that hold a row. A table without flag columns comes
    back as it is. A table that carries all five WIDE_FLAGS columns (the twc_transfer port's,
    each after its <arm>_present) has every value of them compared, as written ("0", "1" or
    empty), with the value the reduction writes, and comes back without those columns, every
    other field as written; the reduction stops naming each arm entry where the two differ, and
    on a table that carries only some of the five."""
    names = next(csv.reader([header]))
    flags, n_held = {}, 0
    for arm in WIDE_ARMS:
        has = (frame["%s_present" % arm] == True).values          # noqa: E712
        sub = frame[has]
        rows = pd.DataFrame(dict(cell=sub.cell, arm=arm, kind=sub.kind,
                                 board_or_role=sub.board_or_role, family=sub.family,
                                 agg_tag=sub.agg_tag, n_pix=sub["%s_n_pix" % arm],
                                 mu_ps=sub["%s_mu_ps" % arm], se_ps=sub["%s_se_ps" % arm]))
        got = iter(self_transfer_flags(rows, agg, "univ_axis_F_wide arm %s" % arm))
        flags["%s_self_transfer" % arm] = [next(got) if h else "" for h in has]
        n_held += int(has.sum())
    carried = [c for c in names if c in WIDE_FLAGS]
    if not carried:
        return header, lines, flags, n_held
    if sorted(carried) != sorted(WIDE_FLAGS):
        raise SystemExit("axis_F.csv carries %s; the reduction reads a table with all five of %s "
                         "or none" % (", ".join(carried), ", ".join(WIDE_FLAGS)))
    at = {c: names.index(c) for c in WIDE_FLAGS}
    keep = [i for i in range(len(names)) if i not in at.values()]
    key = [names.index(c) for c in ("cell", "kind", "board_or_role", "family")]
    if _csv_line(names) != header:
        raise SystemExit("axis_F.csv: the header is not as a CSV writer writes it: %s" % header)
    out, differ = [], []
    for n, line in enumerate(lines):
        fields = next(csv.reader([line]))
        if _csv_line(fields) != line:
            raise SystemExit("axis_F.csv: a line is not as a CSV writer writes its fields, so "
                             "they cannot be copied without the flag columns: %s" % line)
        for c, i in at.items():
            if fields[i] != str(flags[c][n]):
                differ.append("%s %s: the table holds %r, the join gives %r" % (
                    "/".join(fields[k] for k in key), c, fields[i], str(flags[c][n])))
        out.append(_csv_line([fields[i] for i in keep]))
    if differ:
        raise SystemExit("axis_F.csv: %d of its <arm>_self_transfer values differ from the "
                         "flag of the join:\n  %s" % (len(differ), "\n  ".join(differ)))
    return _csv_line([names[i] for i in keep]), out, flags, n_held


def _write_with_columns(dst, header, lines, columns):
    """`lines` under `header` as given, each with the values of `columns` ({name: values}, one
    value per line) appended."""
    names = list(columns)
    with open(dst, "w", newline="") as fh:
        fh.write(",".join([header] + names) + "\n")
        for i, line in enumerate(lines):
            fh.write(",".join([line] + [str(columns[n][i]) for n in names]) + "\n")


def build_axes(out, paths, twc_root):
    header, lines, frames, counts = None, [], [], {}
    for a, files in ((a, paths["axis:" + a]) for a in campaign.UNIV_AXIS_SOURCES):
        head, body, df, replaced = axis_table(files)
        if header is None:
            header = head
        elif head != header:
            raise SystemExit("%s has other columns than the first axis table" % files[0])
        if set(df.axis) != {a}:
            raise SystemExit("axis %s: its tables hold rows of axes %s" % (a, sorted(set(df.axis))))
        lines += body
        frames.append(df)
        counts[a] = len(body)
        if replaced:
            _log("axis %s: %d of %d cells from %s (rows before -> after: %s)"
                 % (a, len(replaced), df.cell.nunique(), ", ".join(files[1:]),
                    ", ".join("%s %d->%d" % (c, *replaced[c]) for c in sorted(replaced))))
    r2 = pd.concat(frames, ignore_index=True)
    long_head, long_lines, flong = table_lines(paths["F_long"])
    wide_head, wide_lines, fwide = table_lines(paths["F_wide"])
    agg = agg_rows(twc_root, set(r2.agg_tag) | set(flong.agg_tag) | set(fwide.agg_tag))

    st = self_transfer_flags(r2, agg, "univ_axis_r2")
    _write_with_columns(os.path.join(out, "univ_axis_r2.csv"), header, lines,
                        {"self_transfer": st})
    r2["self_transfer"] = st
    _log("univ_axis_r2.csv: %d rows (per axis %s), every row joined to one aggregation row; "
         "self_transfer 1 on %d (per axis %s)"
         % (len(r2), counts, sum(st), r2.groupby("axis", sort=False).self_transfer.sum().to_dict()))

    st = self_transfer_flags(flong, agg, "univ_axis_F_long")
    _write_with_columns(os.path.join(out, "univ_axis_F_long.csv"), long_head, long_lines,
                        {"self_transfer": st})
    _log("univ_axis_F_long.csv: %d rows (from %s), self_transfer 1 on %d"
         % (len(flong), paths["F_long"], sum(st)))

    carried = any(c in next(csv.reader([wide_head])) for c in WIDE_FLAGS)
    wide_head, wide_lines, per_arm, n_held = wide_self_transfer(wide_head, wide_lines, fwide, agg)
    _write_with_columns(os.path.join(out, "univ_axis_F_wide.csv"), wide_head, wide_lines, per_arm)
    flagged = [i for i in range(len(fwide)) if any(v == 1 for v in
                                                    (per_arm[c][i] for c in per_arm))]
    mixed = [i for i in flagged if any(v == 0 for v in (per_arm[c][i] for c in per_arm))]
    _log("univ_axis_F_wide.csv: %d rows (from %s), a self-transfer arm on %d of them (%d of "
         "these also hold an arm that is not); %s" % (
             len(fwide), paths["F_wide"], len(flagged), len(mixed),
             "the table's own <arm>_self_transfer columns equal the join's flag on all %d arm "
             "entries that hold a row and are empty on the other %d, written once, at the end"
             % (n_held, len(WIDE_ARMS) * len(fwide) - n_held) if carried else
             "the flags of the join appended (the table carries none)"))


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
                         "TWC_RESULTS_ROOT, universality/results/ of the inputs")
    ap.add_argument("--twc-root", default=campaign.TWC_TRANSFER_ROOT,
                    help="the folder of bank/, figs/ and tables/ (default TWC_TRANSFER_ROOT)")
    ap.add_argument("--part", action="append", choices=PARTS,
                    help="the tables to build (repeatable; default every part)")
    a = ap.parse_args(argv)
    parts = a.part or list(PARTS)
    paths = None
    if {"cells", "axes"} & set(parts):
        if not a.results_root:
            ap.error("--results-root (or ETROC_TWC_RESULTS_ROOT) is required: the study's folder, "
                     "holding condor/ with the seven job lists, the axis tables of "
                     "UNIV_AXIS_SOURCES and results_r3/ with axis_F.csv, axis_F_long.csv and "
                     "summary_F.md")
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
            build_axes(a.out, paths, a.twc_root)
        else:
            build_bank(a.out, a.twc_root)
        _log("part %s done in %.0f s" % (part, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
