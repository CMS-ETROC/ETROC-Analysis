"""The source-count ensemble (SPEC.md 3c): how many source events a borrowed time-walk
correction needs, read from many draws of the source per event count rather than from one.

From one template row of a job list, a cell command, ensemble_jobs() makes one cell per
(rung, seed): the source capped at N events (--src-cap N) by a draw without replacement seeded
by --src-seed s, for s = 0 .. K-1 at every capped rung and s = 0 alone at the rung 'all' (the
source as --cap reads it). Seed 0 is the draw the study's own ladder made. Everything else in
the command is the template's: the target run, combo and triples, the cut, the floors and the
arms, so every cell evaluates the same target events in the same folds. --src-range (full by
default) takes each target's coverage, lever and family from the full source, so the family of
a target does not move with the cap or the seed; the capped label is written beside it. Each
cell writes its own folder, <write root>/cells/<cell>/<cell>.csv, as the study's job lists do,
so that the AGG node of the DAG aggregates every cell table once.

write_list() writes the rows, with one smoke row (three target triples at the smallest rung,
seed 1: a subsampled read and a full-source read in a few minutes), as a job list of the
study's format; `jobs --write-dag`, or `ensemble --write-dag`, turns it into a DAG. Nothing
here submits.
"""
import os
import re
import shlex
import time

from . import campaigns, jobs

# the template's flags that every member sets itself
REPLACED = ("--cell", "--src-cap", "--src-seed", "--src-range", "--out")
_RUNG_TAG = re.compile(r"_n(\d+|all)$")
SMOKE_SEED = 1
SMOKE_NTRIP = 3
SMOKE_EST_MIN = 5
# the modes whose source floor (--nmin-src) is checked on the drawn source (cell.py); modes c,
# d and cb have none
SOURCE_FLOOR_MODES = ("a", "b", "f")


def parse_rungs(text):
    """'100,200,all' -> [100, 200, 0]; 0 is the rung 'all', the source as --cap reads it.
    Raises ValueError on an empty list, a repeated rung or one that is not a positive count."""
    out = []
    for tok in str(text).split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        n = 0 if tok == "all" else int(tok)
        if tok != "all" and n <= 0:
            raise ValueError("a rung is a positive event count or 'all' (got %r)" % tok)
        if n in out:
            raise ValueError("rung %r given twice" % tok)
        out.append(n)
    if not out:
        raise ValueError("no rung given")
    return out


def rung_tag(n):
    """n100 for a cap of 100 events, nall for the uncapped rung (0)."""
    return "nall" if n == 0 else "n%d" % n


def cell_id(stem, n, seed):
    """<stem>_<rung tag>_s<seed, two digits>: G_P3_run26_n100_s07, G_P3_run26_nall_s00."""
    return "%s_%s_s%02d" % (stem, rung_tag(n), seed)


def default_stem(cell):
    """The template's cell id without its rung tag: G_P3_run26_n100 and G_P3_run26_nall both
    give G_P3_run26."""
    return _RUNG_TAG.sub("", cell)


def default_group(group):
    """The template's group with 'ens' after its first part (G_desy -> Gens_desy): the
    aggregation tags of the ensemble are then never the study's, and the last part still names
    the campaign whose f aggregates it (jobs.group_campaign)."""
    parts = group.split("_")
    parts[0] += "ens"
    return "_".join(parts)


def _drop(tokens, flags):
    """`tokens` without the options `flags` and their values (`--flag v` or `--flag=v`)."""
    out, i = [], 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.split("=", 1)[0] in flags:
            i += 1 if "=" in tok else 2
            continue
        out.append(tok)
        i += 1
    return out


def _set(tokens, flag, value):
    """`tokens` with option `flag` set to `value`: its value replaced where the option is, else
    the option appended."""
    out = list(tokens)
    for i, tok in enumerate(out):
        if tok == flag and i + 1 < len(out):
            out[i + 1] = str(value)
            return out
        if tok.startswith(flag + "="):
            out[i] = "%s=%s" % (flag, value)
            return out
    return out + [flag, str(value)]


def member_command(template_command, cid, n, seed, src_range="full"):
    """The command of one member: the template's cell command with --cell, --src-cap (none at
    the rung 'all'), --src-seed, --src-range and --out set for this member, every other token
    as the template has it, placeholders included."""
    toks = shlex.split(template_command)
    if not toks or toks[0] not in jobs.CELL_COMMANDS:
        raise ValueError("the template is not a cell command: %r" % template_command[:60])
    rest = _drop(toks[1:], REPLACED)
    if "--chunk" in [t.split("=", 1)[0] for t in rest]:
        raise ValueError("the template is a chunk job (--chunk): a member is one job, so take "
                         "an unchunked row, or one with --ntrip small enough for one job")
    cmd = [toks[0], "--cell", cid] + rest
    if n:
        cmd += ["--src-cap", str(n)]
    cmd += ["--src-seed", str(seed), "--src-range", src_range,
            "--out", "%s/cells/%s/%s.csv" % (jobs.OUT_TOKEN, cid, cid)]
    bad = [t for t in cmd if not t or re.search(r"\s", t)]
    if bad:
        raise ValueError("tokens with white space cannot go into a job list: %r" % bad)
    return " ".join(cmd)


def ensemble_jobs(template, rungs, seeds, src_range="full", stem=None, group=None,
                  est_min=None):
    """The rows of the rung x seed grid from one template job (a dict of jobs.FIELDS): seeds
    0 .. seeds-1 at every capped rung, seed 0 at the rung 'all' (0), in rung order. Each row's
    job_id is its cell id; its level and memory are the template's, its estimate `est_min`
    (default the template's) and its group `group` (default default_group of the template's)."""
    if int(seeds) < 1:
        raise ValueError("--seeds must be at least 1 (got %r)" % seeds)
    stem = stem or default_stem(template["cell_id"])
    group = group or default_group(template["group"])
    est = template["est_min"] if est_min is None else int(est_min)
    rows = []
    for n in rungs:
        for s in (range(int(seeds)) if n else (0,)):
            cid = cell_id(stem, n, s)
            rows.append(dict(job_id=cid, cell_id=cid, level=template["level"], group=group,
                             est_min=est, mem_mb=template["mem_mb"],
                             command=member_command(template["command"], cid, n, s, src_range)))
    return rows


def smoke_job(template, rungs, src_range="full", stem=None):
    """The list's smoke row: SMOKE_NTRIP target triples at the smallest capped rung with seed
    SMOKE_SEED, so that the first node reads a subsampled source and its full source."""
    stem = stem or default_stem(template["cell_id"])
    capped = [n for n in rungs if n]
    n = min(capped) if capped else 0
    cid = "smoke_" + cell_id(stem, n, SMOKE_SEED if n else 0)
    cmd = member_command(template["command"], cid, n, SMOKE_SEED if n else 0, src_range)
    toks = _drop(cmd.split(" "), ("--max-triples", "--cap-triples"))
    cmd = " ".join(_set(toks, "--ntrip", SMOKE_NTRIP))
    return dict(job_id="SMOKE", cell_id=cid, level="smoke", group="none",
                est_min=SMOKE_EST_MIN, mem_mb=template["mem_mb"], command=cmd)


def write_list(path, rows, smoke, comments=()):
    """Write a job list: '#' comment lines, the header row, the smoke row, then `rows`."""
    with open(path, "w") as fh:
        for c in comments:
            fh.write("# %s\n" % c)
        fh.write("\t".join(jobs.FIELDS) + "\n")
        for j in [smoke] + list(rows):
            fh.write("\t".join([j["job_id"], j["cell_id"], j["level"], j["group"],
                                str(j["est_min"]), str(j["mem_mb"]), j["command"]]) + "\n")
    return path


def cmd_ensemble(a, root, parse):
    """`ensemble`: write the job list of the source-count ensemble of the template row
    `--template` of `--list` into the write root (default ensemble/cells_<stem>_ensemble.tsv).
    Every command is expanded and parsed first (`parse`, the cell parser); an existing list is
    not replaced. Returns the path written."""
    src = jobs.list_path(a.job_list)
    found = [j for j in jobs.read_jobs(src) if j["job_id"] == a.template]
    if not found:
        raise ValueError("%s has no job %r" % (src, a.template))
    tmpl = found[0]
    rungs = parse_rungs(a.rungs)
    stem = a.stem or default_stem(tmpl["cell_id"])
    rows = ensemble_jobs(tmpl, rungs, a.seeds, a.src_range, stem, a.group, a.est_min)
    smoke = smoke_job(tmpl, rungs, a.src_range, stem)
    read_root = campaigns.transfer_root()
    for j in [smoke] + rows:
        ns = parse(shlex.split(jobs.expand(j["command"], campaigns.CAMPAIGNS, read_root, root)))
        if ns.cell != j["cell_id"]:
            raise ValueError("%s: --cell %s is not the row's cell id" % (j["job_id"], ns.cell))
        if ns.mode in SOURCE_FLOOR_MODES and ns.src_cap and ns.nmin_src > ns.src_cap:
            raise ValueError("%s: in mode %s the source floor --nmin-src %d applies to the draw, "
                             "so a rung of %d events would refuse every source; give the "
                             "template --nmin-src %d or less"
                             % (j["job_id"], ns.mode, ns.nmin_src, ns.src_cap, ns.src_cap))
    dest = campaigns.output_path(a.out, root, "ensemble", "cells_%s_ensemble.tsv" % stem)
    if os.path.exists(dest):
        raise ValueError("%s exists: an ensemble list is written once (its DAG folder keeps "
                         "the copy it ran from); name a new --out" % dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    capped = [n for n in rungs if n]
    est = sum(j["est_min"] for j in rows)
    comments = [
        "TWC transfer: the source-count ensemble of one cell (SPEC.md 3c)",
        "written by python -m twc_transfer ensemble on %s" % time.strftime("%Y-%m-%d %H:%M:%S"),
        "template: job %s of %s" % (tmpl["job_id"], os.path.basename(src)),
        "rungs: %s; seeds 0..%d per capped rung, seed 0 at all; --src-range %s"
        % (", ".join(rung_tag(n)[1:] for n in rungs), int(a.seeds) - 1, a.src_range),
        "%d cell jobs (%d capped rungs x %d seeds%s) + 1 smoke; est %d CPU-min = %.1f CPU-h"
        % (len(rows), len(capped), int(a.seeds), " + 1 at all" if 0 in rungs else "", est,
           est / 60.0),
    ]
    write_list(dest, rows, smoke, comments)
    for c in comments[2:]:
        print("ensemble: %s" % c)
    print("ensemble: -> %s" % dest)
    return dest
