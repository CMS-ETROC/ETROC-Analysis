"""Job lists: reading the study's job tables, expanding their roots, checking their commands.

A job list is a text file of tab-separated rows with seven fields,

    job_id  cell_id  level  group  est_min  mem_mb  command

where `command` is the argument list of one `python -m twc_transfer` subcommand (`cell` or
`tier0` in the study's lists). Lines starting with "#" and empty lines are skipped, and a first
row whose job_id is the word "job_id" is a header (optional). A cell too long for one job is
split into chunk jobs (`--chunk k/N`) that share its cell_id.

The commands carry two kinds of placeholder:

  ${ETROC_TWC_ROOT_<CAMPAIGN>}  the step-10 root of a campaign (--root, --src-root), replaced
                                by campaigns.CAMPAIGNS[<campaign>] (the same variable name
                                overrides it in the environment)
  ${TT_OUT}                     the study's output tree, replaced per flag: an input flag
                                (--bank, --src-bank and the other READ_FLAGS) gets the read
                                root, --out gets the write root. A ${TT_OUT} under any other
                                flag is refused, and so is a write root inside the read root.

The seven lists of the study are in jobs/ beside this file (campaigns.JOB_LISTS gives their
order and the output root each wrote into). They are the study's own tables with their
comment lines dropped and the literal campaign roots replaced by the placeholders above:
expanding them with the default roots gives back the commands the study ran, byte for byte.

The condor driver (SPEC.md 6a) turns a list into a DAG without submitting anything:
write_dag() writes, in a folder of the write root, the job table with every placeholder
expanded, the DAG (SMOKE -> PREP -> one node per job row -> AGG), its four submit files, the
wrapper every node runs (the checks and exit codes 90 to 97 of the study's wrapper, then
PYTHONHASHSEED=0, one thread per pool, nice 19) and a manifest. PREP checks that every input
the table names exists (check_inputs); AGG aggregates each group of the table at the f of its
campaign (agg_plan, campaigns.AGG_F), as the study's aggregation drivers did.
"""
import hashlib
import os
import re
import shlex
import stat
import sys
import time

from . import campaigns

FIELDS = ("job_id", "cell_id", "level", "group", "est_min", "mem_mb", "command")
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(PKG_DIR, "jobs")
REPO_DIR = os.path.dirname(os.path.dirname(PKG_DIR))

OUT_TOKEN = "${TT_OUT}"
WRITE_FLAGS = ("--out",)
READ_FLAGS = ("--bank", "--src-bank", "--root", "--src-root", "--cells", "--tier0")
_ROOT_TOKEN = re.compile(r"\$\{ETROC_TWC_ROOT_([A-Z0-9_]+)\}")


def list_path(name):
    """A job list: `name` itself when it is an existing file, else jobs/<name>."""
    if os.path.isfile(name):
        return name
    path = os.path.join(JOBS_DIR, name)
    if not os.path.isfile(path):
        raise FileNotFoundError("no job list %r (neither a file nor one of jobs/: %s)"
                                % (name, ", ".join(sorted(os.listdir(JOBS_DIR)))))
    return path


def read_jobs(path):
    """The jobs of one list, in file order: dicts of FIELDS, est_min and mem_mb as int.

    Raises ValueError on a row without exactly seven fields or on a repeated job_id."""
    jobs, seen = [], set()
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            if parts[0] == "job_id":
                continue
            if len(parts) != len(FIELDS):
                raise ValueError("%s line %d: %d tab-separated fields, expected %d"
                                 % (path, n, len(parts), len(FIELDS)))
            job = dict(zip(FIELDS, parts))
            job["est_min"], job["mem_mb"] = int(job["est_min"]), int(job["mem_mb"])
            if job["job_id"] in seen:
                raise ValueError("%s line %d: job_id %s repeated" % (path, n, job["job_id"]))
            seen.add(job["job_id"])
            jobs.append(job)
    return jobs


def by_cell(jobs):
    """{cell_id: [jobs]} in file order; the chunk jobs of a cell share its cell_id."""
    out = {}
    for job in jobs:
        out.setdefault(job["cell_id"], []).append(job)
    return out


def expand(command, campaign_roots, read_root, write_root):
    """`command` with its placeholders replaced, spacing kept.

    campaign_roots  {campaign: step-10 root}, e.g. campaigns.CAMPAIGNS
    read_root       replaces ${TT_OUT} under an input flag (READ_FLAGS)
    write_root      replaces ${TT_OUT} under --out
    A root given as None leaves its ${TT_OUT} tokens as they are. Raises ValueError when the
    write root is, or lies inside, the read root (by their real paths)."""
    if read_root is not None and write_root is not None:
        campaigns.refuse_inside(write_root, [("the read root", read_root)],
                                "the write root %s" % write_root)
    out, flag, pos = [], None, 0
    for m in re.finditer(r"\S+", command):
        tok = m.group(0)
        if tok.startswith("--"):
            flag = tok.split("=", 1)[0]
        if "$" in _ROOT_TOKEN.sub("", tok.replace(OUT_TOKEN, "")):
            raise ValueError("unknown placeholder in %r: %s" % (tok, command))
        tok = _ROOT_TOKEN.sub(lambda r: _campaign_root(r.group(1), campaign_roots), tok)
        if OUT_TOKEN in tok:
            if flag in WRITE_FLAGS:
                root = write_root
            elif flag in READ_FLAGS:
                root = read_root
            else:
                raise ValueError("%s under %s, which is neither an input flag (%s) nor an "
                                 "output flag (%s): %s" % (OUT_TOKEN, flag, ", ".join(READ_FLAGS),
                                                           ", ".join(WRITE_FLAGS), command))
            if root is not None:
                tok = tok.replace(OUT_TOKEN, root.rstrip("/"))
        out.append(command[pos:m.start()])
        out.append(tok)
        pos = m.end()
    out.append(command[pos:])
    return "".join(out)


def _campaign_root(name, campaign_roots):
    key = name.lower()
    if key not in campaign_roots:
        raise ValueError("${ETROC_TWC_ROOT_%s}: no campaign %r (known: %s)"
                         % (name, key, ", ".join(sorted(campaign_roots))))
    return campaign_roots[key].rstrip("/")


def dry_run(path, campaign_roots, read_root, write_root, parse, only=None, stream=None):
    """Print every job of a list as it would run, and parse each command with `parse`
    (tokens -> argparse namespace; it raises on a command the parser refuses). A cell
    command must name its row's cell_id in --cell. Returns (number of jobs, list of
    (job_id, error) for the commands that failed)."""
    stream = stream or sys.stdout
    jobs = read_jobs(path)
    if only:
        missing = sorted(set(only) - {j["job_id"] for j in jobs})
        if missing:
            raise ValueError("%s has no job %s" % (path, ", ".join(missing)))
        jobs = [j for j in jobs if j["job_id"] in only]
    failed = []
    for job in jobs:
        try:
            cmd = expand(job["command"], campaign_roots, read_root, write_root)
            ns = parse(shlex.split(cmd))
            if ns.cmd in ("cell", "transfer") and ns.cell != job["cell_id"]:
                raise ValueError("--cell %s differs from the row's cell_id %s"
                                 % (ns.cell, job["cell_id"]))
        except (ValueError, SystemExit) as e:
            failed.append((job["job_id"], str(e)))
            stream.write("%s\tFAILED: %s\n" % (job["job_id"], e))
            continue
        stream.write("%s\t%s\n" % (job["job_id"], cmd))
    smoke = sum(1 for j in jobs if j["level"] == "smoke")
    stream.write("%s: %d jobs (%d smoke), %d parsed, %d failed\n"
                 % (os.path.basename(path), len(jobs), smoke, len(jobs) - len(failed),
                    len(failed)))
    return len(jobs), failed


# ------------------------------------------------------------------------- the condor driver
NODE_IDS = ("SMOKE", "PREP", "AGG")     # the DAG's own nodes; no job row of a list may use them
CELL_COMMANDS = ("cell", "transfer")
REPO_CORE = os.path.join(REPO_DIR, "TestBeam", "condor_at_lxplus", "core")


def expanded_jobs(path, campaign_roots, read_root, write_root, parse):
    """The jobs of a list with the placeholders of every command expanded (expand) and the
    parsed command under "ns". Raises ValueError naming every command that does not expand,
    that the parser refuses, or whose --cell is not its row's cell_id."""
    out, bad = [], []
    for job in read_jobs(path):
        try:
            cmd = expand(job["command"], campaign_roots, read_root, write_root)
            ns = parse(shlex.split(cmd))
            if ns.cmd in CELL_COMMANDS and ns.cell != job["cell_id"]:
                raise ValueError("--cell %s differs from the row's cell_id %s"
                                 % (ns.cell, job["cell_id"]))
        except (ValueError, SystemExit) as e:
            bad.append("%s: %s" % (job["job_id"], e))
            continue
        out.append(dict(job, command=cmd, ns=ns))
    if bad:
        raise ValueError("%s: %d command(s) refused:\n  %s" % (path, len(bad), "\n  ".join(bad)))
    return out


def job_inputs(ns, read_root):
    """The files and folders one parsed command reads: a bank file (tier0 --bank, cell
    --src-bank; a relative one in the read root), the target run's step-10 folder (for a cell,
    its combo's time/ folder) and, for a cell whose source is another run or campaign, the
    source run's folder (the root rule of cell.py)."""
    from .naming import run_dir

    def in_read_root(p):
        return p if os.path.isabs(p) else os.path.join(read_root, p)

    paths = []
    if ns.cmd == "tier0":
        paths.append(in_read_root(ns.bank))
    if ns.cmd in ("index", "bank", "fit") + CELL_COMMANDS:
        target = run_dir(ns.campaign, ns.run, ns.root)
        if ns.cmd in CELL_COMMANDS and ns.combo:
            target = os.path.join(target, ns.combo, "time")
        paths.append(target)
    if ns.cmd in CELL_COMMANDS:
        if ns.src_run or ns.src_campaign:
            camp = ns.src_campaign or ns.campaign
            root = ns.src_root or (ns.root if camp == ns.campaign else None)
            paths.append(run_dir(camp, ns.src_run or ns.run, root))
        if ns.src_bank:
            paths.append(in_read_root(ns.src_bank))
    return paths


def check_inputs(jobs, read_root, stream=None):
    """Write every input the expanded jobs name that does not exist (the PREP node). Returns
    (number of inputs, the missing ones)."""
    stream = stream or sys.stdout
    paths = sorted({p for job in jobs for p in job_inputs(job["ns"], read_root)})
    missing = [p for p in paths if not os.path.exists(p)]
    for p in missing:
        stream.write("MISSING\t%s\n" % p)
    stream.write("inputs: %d named, %d missing\n" % (len(paths), len(missing)))
    return len(paths), missing


def group_campaign(group):
    """The campaign whose f aggregates a group of a job list: the group itself in round 1
    (desy, aug, mar, jul), else its last part before an optional _guardfix (D_mar,
    K_desy_guardfix). None when it names none (none, the smoke row's group)."""
    from .campaigns import AGG_F
    parts = group.split("_")
    if len(parts) > 1 and parts[-1] == "guardfix":
        parts = parts[:-1]
    return parts[-1] if parts[-1] in AGG_F else None


def agg_plan(jobs, write_root):
    """The aggregations of the AGG node, the study's aggregation drivers as one rule: per group
    of the expanded cell jobs (levels other than smoke and f), one `agg` per f of the group's
    campaign (campaigns.AGG_F), over the folder each job's --out names (its *.csv: the chunk
    tables of one cell share the folder) or, without --out, the cell's default table, into
    <write root>/tables/agg_<group>_<ftag>. A group that names no campaign is refused: one
    campaign's f never aggregates another's cells. Returns dicts of group, tag, f, cells
    ({cell_id: [patterns]}) and out, groups in name order."""
    from .campaigns import AGG_F
    groups = {}
    for job in jobs:
        ns = job["ns"]
        if job["level"] in ("smoke", "f") or ns.cmd not in CELL_COMMANDS:
            continue
        out = ns.out or os.path.join(write_root, "cells", ns.cell + ".csv")
        pat = os.path.join(os.path.dirname(out), "*.csv") if ns.out else out
        groups.setdefault(job["group"], {}).setdefault(job["cell_id"], set()).add(pat)
    plan = []
    for group in sorted(groups):
        camp = group_campaign(group)
        if camp is None:
            raise ValueError("group %r names no campaign of campaigns.AGG_F (%s): its cells "
                             "would be aggregated with another campaign's f"
                             % (group, ", ".join(sorted(AGG_F))))
        cells = {c: sorted(p) for c, p in sorted(groups[group].items())}
        for ftag, f in AGG_F[camp]:
            tag = "agg_%s_%s" % (group, ftag)
            plan.append(dict(group=group, tag=tag, f=f, cells=cells,
                             out=os.path.join(write_root, "tables", tag)))
    return plan


def run_agg(plan, run, stream=None):
    """Run an agg_plan: `run(argv)` runs one `agg` command line and raises SystemExit when it
    fails. A cell with no table yet is listed and left out; a group with none is skipped.
    Returns the tags that failed."""
    import glob
    stream = stream or sys.stdout
    failed = []
    for p in plan:
        have = {c: [f for pat in pats for f in glob.glob(pat)] for c, pats in p["cells"].items()}
        missing = sorted(c for c, files in have.items() if not files)
        if missing:
            stream.write("%s: no table yet for %d cell(s): %s\n"
                         % (p["tag"], len(missing), ", ".join(missing)))
        pats = [pat for c, pats in p["cells"].items() if have[c] for pat in pats]
        if not pats:
            stream.write("%s: nothing to aggregate\n" % p["tag"])
            continue
        argv = ["agg", "--cells"] + pats + ["--f-inflate", repr(p["f"]), "--out", p["out"]]
        stream.write("%s: %s\n" % (p["tag"], " ".join(argv)))
        stream.flush()
        try:
            run(argv)
        except SystemExit as e:
            if e.code not in (None, 0):
                failed.append(p["tag"])
                stream.write("%s: FAILED (%s)\n" % (p["tag"], e.code))
    return failed


def package_sha(pkg_dir=PKG_DIR):
    """sha256[0:16] of the package's modules (its *.py files read in name order): the version
    a DAG is written for, which its wrapper recomputes before every node."""
    h = hashlib.sha256()
    for name in sorted(n for n in os.listdir(pkg_dir) if n.endswith(".py") and n[0] != "."):
        with open(os.path.join(pkg_dir, name), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:16]


CORE_FILES = ("twc_solver.py", "bootstrap.py", "io_utils.py")


def core_files_sha1(core_dir):
    """{file: sha1} of the pipeline core files (CORE_FILES) in `core_dir` (read, not
    imported)."""
    out = {}
    for name in CORE_FILES:
        p = os.path.join(core_dir, name)
        if os.path.isfile(p):
            with open(p, "rb") as fh:
                out[name] = hashlib.sha1(fh.read()).hexdigest()
    return out


def flavour_for(est_min, command=""):
    """+JobFlavour of a job: longlunch (2 h) covers most jobs by a factor of two or more;
    workday (8 h) when the job's own estimate exceeds 90 min, or when it rotates five or more
    source pixels (--src-k >= 5), whose fit count the estimate does not fully capture (the
    study's stage-3 cells of that kind overran longlunch)."""
    if est_min > 90:
        return "workday"
    m = re.search(r"--src-k\s+(\d+)", command)
    if m and int(m.group(1)) >= 5:
        return "workday"
    return "longlunch"


def default_dag_name(table):
    """transfer_<list name without its cells_ prefix>: transfer_cells for cells.tsv,
    transfer_guardfix_rerun for cells_guardfix_rerun.tsv, as the study named its DAGs."""
    stem = os.path.splitext(os.path.basename(table))[0]
    return "transfer_" + (stem[len("cells_"):] if stem.startswith("cells_") else stem)


SUB_HEADER = """\
# written by python -m twc_transfer jobs --write-dag on {stamp}; do not edit by hand
universe                = vanilla
executable              = {dag_dir}/wrapper.sh
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = {dag_dir}/{table}
transfer_output_files   = ""
MY.WantOS               = "el9"
request_cpus            = 1
request_disk            = 2 GB
JobBatchName            = "{batch}"
"""

WRAPPER = r"""#!/bin/bash
# wrapper.sh: the executable of every node of the DAG beside it, written by
# python -m twc_transfer jobs --write-dag on @STAMP@ for the job table @TABLE_NAME@.
# Do not edit by hand: write the DAG again.
#
#   $1  job_id   a row of the job table (its first field), or PREP or AGG
#   $2  retry    DAGMan's $(RETRY) for this node; > 0 means resume: --overwrite is dropped
#                and a cell continues its table where it stopped
#
# Checks, in this order, each with its own exit code: 90 no job_id; 91 the package cannot be
# read (AFS); 92 the package is not the version the DAG was written for; 98 the pipeline core
# is not the one it was written for; 93 no job table; 94 the job is not in it; 95 the write
# root is not writable; 96 the LCG environment does not source; 97 no TestBeam/ to run from.
# Any other code is the command's own. The roots below are the ones the table's commands were
# expanded with. `wrapper.sh --help` prints this text.
set -o pipefail

JOB="${1:-}"
RETRY="${2:-0}"
case "$JOB" in
    -h|--help) awk 'NR > 1 && /^#/ {sub(/^# ?/, ""); print; next} NR > 1 {exit}' "$0"; exit 0 ;;
esac
case "$RETRY" in ''|*[!0-9]*) RETRY=0 ;; esac

TWC_REPO=@REPO@
TWC_DAG_DIR=@DAG_DIR@
TWC_TSV=@TABLE@
TWC_EXPECT_SHA=@SHA@
TWC_CORE=@CORE@
TWC_EXPECT_CORE=@CORE_SHA@
@EXPORTS@
TWC_PKG="$TWC_REPO/TestBeam/twc_transfer"

say() { echo "[wrap $(date +%H:%M:%S)] $*"; }

[ -n "$JOB" ] || { say "FATAL no job_id given"; exit 90; }
say "job=$JOB retry=$RETRY host=$(hostname -s) pid=$$"

# 1. AFS: the package itself
if [ ! -r "$TWC_PKG/__main__.py" ]; then
    say "FATAL cannot read $TWC_PKG/__main__.py: the repository is not visible from this worker."
    exit 91
fi

# 2. one package version for every node: sha256 of its modules read in name order, as
#    jobs.package_sha() computed it when the DAG was written
SHA=$(cd "$TWC_PKG" && LC_ALL=C ls *.py | while read -r f; do cat "$f"; done \
      | sha256sum | cut -c1-16)
say "twc_transfer sha256[0:16]=$SHA"
if [ "$SHA" != "$TWC_EXPECT_SHA" ]; then
    say "FATAL the package changed since the DAG was written (expected $TWC_EXPECT_SHA, got $SHA)."
    say "      Every node of one DAG runs one version of the package: write the DAG again."
    exit 92
fi

# 2b. one pipeline core for every node: the sha1 of each of its files, as
#     jobs.core_files_sha1() computed them when the DAG was written
CORE_SHA=""
for f in @CORE_FILES@; do
    [ -f "$TWC_CORE/$f" ] || continue
    CORE_SHA="$CORE_SHA${CORE_SHA:+ }$f=$(sha1sum < "$TWC_CORE/$f" | cut -c1-40)"
done
say "core $TWC_CORE ${CORE_SHA:-(none of its files)}"
if [ "$CORE_SHA" != "$TWC_EXPECT_CORE" ]; then
    say "FATAL the core changed since the DAG was written."
    say "      expected '$TWC_EXPECT_CORE'"
    say "      got      '$CORE_SHA'"
    say "      Every node of one DAG runs one core: write the DAG again."
    exit 98
fi

# 3. the job table: the copy condor transferred, else the one beside the DAG
TSV="$TWC_TSV"
[ -r "$TSV" ] || TSV="$TWC_DAG_DIR/$TWC_TSV"
[ -r "$TSV" ] || { say "FATAL no job table ($TWC_TSV)"; exit 93; }
TSV="$(cd "$(dirname "$TSV")" && pwd)/$(basename "$TSV")"
case "$JOB" in
    PREP) CELL_ID=""; LEVEL=prep; ARGS="jobs --list '$TSV' --check-inputs" ;;
    AGG)  CELL_ID=""; LEVEL=agg; ARGS="jobs --list '$TSV' --agg" ;;
    *)    ROW=$(awk -F'\t' -v j="$JOB" '$1 == j {print; exit}' "$TSV")
          [ -n "$ROW" ] || { say "FATAL job_id '$JOB' not in $TSV"; exit 94; }
          CELL_ID=$(printf '%s\n' "$ROW" | cut -f2)
          LEVEL=$(printf '%s\n' "$ROW" | cut -f3)
          ARGS=$(printf '%s\n' "$ROW" | cut -f7) ;;
esac
say "cell_id=$CELL_ID level=$LEVEL"

# 4. the write root: the folder of this job's log
LOGDIR="$ETROC_TWC_OUT/cells/$CELL_ID"
[ -n "$CELL_ID" ] || LOGDIR="$ETROC_TWC_OUT/logs"
mkdir -p "$LOGDIR" || {
    say "FATAL cannot mkdir $LOGDIR: the write root is not writable from this worker"; exit 95; }
LOG="$LOGDIR/${JOB}.log"

# 5. the python environment (LCG_104d: python 3.9.12, pandas 1.5.3)
source "$TWC_REPO/TestBeam/condor_at_lxplus/envs/load_python39.sh" >/dev/null 2>&1 \
    || { say "FATAL cannot source the LCG env"; exit 96; }
say "python $(python3 -c 'import sys, pandas
print(sys.version.split()[0], "pandas", pandas.__version__)' 2>&1)"

# 6. determinism and one thread per pool
export PYTHONHASHSEED=0
@THREADS@
export PYTHONUNBUFFERED=1

# 7. a retry resumes: only the first attempt may --overwrite
if [ "$RETRY" -gt 0 ]; then
    ARGS=$(printf '%s\n' "$ARGS" | sed 's/ --overwrite//g')
    say "retry $RETRY: --overwrite dropped, the cell resumes from its table"
fi

cd "$TWC_REPO/TestBeam" || { say "FATAL cannot cd $TWC_REPO/TestBeam"; exit 97; }

# the table's commands are expanded already, no placeholder is left: eval only splits them
eval "set -- $ARGS"
say "run: nice -n 19 python3 -m twc_transfer $*"
{
  echo "=== $(date +%Y-%m-%d_%H:%M:%S) job=$JOB cell=$CELL_ID host=$(hostname -s) retry=$RETRY"
  echo "=== twc_transfer sha256[0:16]=$SHA"
  echo "=== argv: python3 -m twc_transfer $*"
} >> "$LOG"

nice -n 19 python3 -m twc_transfer "$@" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "=== $(date +%Y-%m-%d_%H:%M:%S) exit=$rc" >> "$LOG"
say "exit=$rc  log=$LOG"
exit "$rc"
"""


def wrapper_text(table, dag_dir, sha, campaign_roots, read_root, write_root, core_dir=None,
                 repo=REPO_DIR, stamp=""):
    """The wrapper script of a DAG, with its roots, its table, the package sha and the sha1 of
    the core files written in (the core is `core_dir`, else the repository's)."""
    core = core_dir or os.path.join(repo, "TestBeam", "condor_at_lxplus", "core")
    core_sha = " ".join("%s=%s" % kv for kv in sorted(core_files_sha1(core).items()))
    from .__main__ import THREAD_VARS
    exports = ["export ETROC_TWC_OUT=%s" % shlex.quote(write_root),
               "export ETROC_TWC_TRANSFER_ROOT=%s" % shlex.quote(read_root)]
    exports += ["export ETROC_TWC_ROOT_%s=%s" % (c.upper(), shlex.quote(r))
                for c, r in sorted(campaign_roots.items())]
    exports.append("export ETROC_TWC_CORE=%s" % shlex.quote(core_dir) if core_dir
                   else "unset ETROC_TWC_CORE")
    subs = {"@STAMP@": stamp, "@TABLE_NAME@": table, "@REPO@": shlex.quote(repo),
            "@DAG_DIR@": shlex.quote(dag_dir), "@TABLE@": shlex.quote(table),
            "@SHA@": sha, "@EXPORTS@": "\n".join(exports), "@CORE@": shlex.quote(core),
            "@CORE_SHA@": shlex.quote(core_sha), "@CORE_FILES@": " ".join(sorted(CORE_FILES)),
            "@THREADS@": "export " + " ".join("%s=1" % v for v in THREAD_VARS)}
    text = WRAPPER
    for k, v in subs.items():
        text = text.replace(k, v)
    return text


def write_dag(path, dag_dir, campaign_roots, read_root, write_root, parse, dag_name=None,
              retry=1, maxjobs=12, prep_node=True, stream=None):
    """Write the condor DAG of the job list `path` under `dag_dir` (a new or empty folder in the
    write root; a relative one is taken in it). Submits nothing. Every command is expanded (read root,
    write root, campaign roots) and parsed first; the list must hold exactly one smoke row.

      <dag_dir>/<list name>            the job table, every placeholder expanded
      <dag_dir>/<dag>.dag              SMOKE -> PREP -> one node per other row -> AGG
      <dag_dir>/sub/[<dag>_]{smoke,prep,cell,agg}.sub
      <dag_dir>/wrapper.sh             the executable of every node
      <dag_dir>/manifest[_<dag>].txt   the package sha, the core, the roots, the counts, the
                                       estimate and the aggregations of the AGG node
      <dag_dir>/logs/                  condor's output, error and event logs

    The bracketed <dag>_ is left out for the default name transfer_cells, as the study's
    generator did. Returns {name: path} of what it wrote."""
    stream = stream or sys.stdout
    if not os.path.isabs(dag_dir):
        dag_dir = os.path.join(write_root, dag_dir)
    dag_dir = os.path.abspath(dag_dir)
    if not campaigns.inside(dag_dir, write_root):
        raise ValueError("%s is not inside the write root %s" % (dag_dir, write_root))
    campaigns.refuse_inside(dag_dir, campaigns.protected_roots(), "the DAG folder %s" % dag_dir)
    if os.path.isdir(dag_dir) and os.listdir(dag_dir):
        raise ValueError("%s exists and is not empty: write each job list's DAG into a new, "
                         "empty folder (one DAG folder and one write root per list; DAGMan "
                         "keeps its rescue files beside a DAG)" % dag_dir)
    jobs = expanded_jobs(path, campaign_roots, read_root, write_root, parse)
    smoke = [j for j in jobs if j["level"] == "smoke"]
    cells = [j for j in jobs if j["level"] != "smoke"]
    if len(smoke) != 1:
        raise ValueError("%s holds %d level=smoke rows; a DAG needs exactly one (the node that "
                         "proves a worker can read the package and the data and write the "
                         "write root before any other runs)" % (path, len(smoke)))
    taken = sorted({j["job_id"] for j in cells} & set(NODE_IDS))
    if taken:
        raise ValueError("%s: job ids %s are the DAG's own node names" % (path, ", ".join(taken)))
    smoke = smoke[0]
    plan = agg_plan(jobs, write_root)
    table = os.path.basename(path)
    dagbase = dag_name or default_dag_name(table)
    default_dag = dagbase == "transfer_cells"
    pfx = "" if default_dag else dagbase + "_"
    bsuf = "" if default_dag else "_" + dagbase.replace("transfer_", "")
    sha = package_sha()
    core_env = os.environ.get("ETROC_TWC_CORE")
    core = core_env or REPO_CORE
    core_sha = core_files_sha1(core)
    from .campaigns import CELLS_CORE_SHA1
    core_is_cells = core_sha == CELLS_CORE_SHA1
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    est = sum(j["est_min"] for j in cells)
    by_level = {}
    for j in cells:
        by_level[j["level"]] = by_level.get(j["level"], 0) + 1
    levels = ", ".join("%s=%d" % kv for kv in sorted(by_level.items()))

    stream.write("write-dag: %d jobs + 1 smoke%s + 1 agg; by level %s\n"
                 % (len(cells), " + 1 prep" if prep_node else "", levels))
    stream.write("           est %d CPU-min = %.1f CPU-h; wall at -maxjobs %d: >= %.1f h plus the "
                 "longest job (%d min) and queue time\n"
                 % (est, est / 60.0, maxjobs, est / 60.0 / maxjobs,
                    max(j["est_min"] for j in cells)))
    stream.write("           twc_transfer sha256[0:16]=%s; core %s (%s)\n"
                 % (sha, core, "the study's cells' core" if core_is_cells
                    else "not the study's cells' core: the cells rerun, they do not reproduce"
                    if core_sha else "MISSING: none of the three core files is there"))
    check_inputs(jobs, read_root, stream)

    subdir, logdir = os.path.join(dag_dir, "sub"), os.path.join(dag_dir, "logs")
    os.makedirs(subdir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)
    written = {}

    def put(name, text, mode=None):
        p = os.path.join(dag_dir, name)
        with open(p, "w") as fh:
            fh.write(text)
        if mode:
            os.chmod(p, mode)
        written[name] = p

    put(table, "\n".join(["\t".join(FIELDS)] + [
        "\t".join([j["job_id"], j["cell_id"], j["level"], j["group"], str(j["est_min"]),
                   str(j["mem_mb"]), j["command"]]) for j in jobs]) + "\n")
    put("wrapper.sh", wrapper_text(table, dag_dir, sha, campaign_roots, read_root, write_root,
                                   core_env, stamp=stamp),
        stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)

    common_io = ("output                  = {log}/{jid}.$(ClusterId).out\n"
                 "error                   = {log}/{jid}.$(ClusterId).err\n"
                 "log                     = {log}/{dag}.condorlog\n")

    def sub(name, batch, body, jid):
        put(os.path.join("sub", pfx + name),
            SUB_HEADER.format(stamp=stamp, dag_dir=dag_dir, table=table, batch=batch + bsuf)
            + body + common_io.format(log=logdir, jid=jid, dag=dagbase) + "queue\n")

    sub("cell.sub", "twc_transfer_cells",
        'arguments               = "$(job_id) $(retry)"\n'
        '+JobFlavour             = "$(flavour)"\n'
        'request_memory          = $(mem)\n', "$(job_id)")
    sub("smoke.sub", "twc_transfer_smoke",
        'arguments               = "%s 0"\n'
        '+JobFlavour             = "microcentury"\n'
        'request_memory          = %d\n' % (smoke["job_id"], smoke["mem_mb"]), "SMOKE")
    if prep_node:
        sub("prep.sub", "twc_transfer_prep",
            'arguments               = "PREP 0"\n'
            '+JobFlavour             = "microcentury"\n'
            'request_memory          = 2000\n', "prep")
    sub("agg.sub", "twc_transfer_agg",
        'arguments               = "AGG 0"\n'
        '+JobFlavour             = "longlunch"\n'
        'request_memory          = 4000\n', "agg")

    L = ["# TWC transfer: the jobs of one job table",
         "# DAG %s, job table %s" % (dagbase, table),
         "# written by python -m twc_transfer jobs --write-dag on %s" % stamp,
         "# twc_transfer sha256[0:16] = %s" % sha,
         "# %d cell jobs, %d CPU-min estimated; submit with -maxjobs %d" % (len(cells), est,
                                                                            maxjobs),
         "# SMOKE -> %sthe cell jobs -> AGG" % ("PREP -> " if prep_node else ""), "",
         "JOB  SMOKE %s/sub/%ssmoke.sub" % (dag_dir, pfx),
         "RETRY SMOKE %d" % retry, ""]
    if prep_node:
        L += ["JOB  PREP %s/sub/%sprep.sub" % (dag_dir, pfx), "RETRY PREP %d" % retry,
              "PARENT SMOKE CHILD PREP", ""]
    for j in cells:
        L.append("JOB  %s %s/sub/%scell.sub" % (j["job_id"], dag_dir, pfx))
        L.append('VARS %s job_id="%s" cell_id="%s" mem="%d" flavour="%s" retry="$(RETRY)"'
                 % (j["job_id"], j["job_id"], j["cell_id"], j["mem_mb"],
                    flavour_for(j["est_min"], j["command"])))
        if retry:
            L.append("RETRY %s %d" % (j["job_id"], retry))
    L += ["", "JOB  AGG %s/sub/%sagg.sub" % (dag_dir, pfx), "RETRY AGG %d" % retry, ""]
    parent = "PREP" if prep_node else "SMOKE"
    ids = [j["job_id"] for j in cells]
    L += ["PARENT %s CHILD %s" % (parent, " ".join(ids[i:i + 8])) for i in range(0, len(ids), 8)]
    L.append("")
    L += ["PARENT %s CHILD AGG" % " ".join(ids[i:i + 8]) for i in range(0, len(ids), 8)]
    L.append("")
    put(dagbase + ".dag", "\n".join(L) + "\n")

    man = ["TWC transfer DAG manifest",
           "written        %s" % stamp,
           "twc_transfer   %s  sha256[0:16]=%s" % (PKG_DIR, sha),
           "core           %s  %s" % (core, " ".join("%s=%s" % kv
                                                    for kv in sorted(core_sha.items()))),
           "               %s" % ("the core of the study's cells (campaigns.CELLS_CORE_SHA1)"
                                  if core_is_cells else "not the core of the study's cells: "
                                  "the cells rerun, they do not reproduce its numbers"
                                  if core_sha else "MISSING: the folder holds none of the "
                                  "three core files"),
           "write root     %s" % write_root,
           "read root      %s" % read_root,
           "cell jobs      %d (%s)" % (len(cells), levels),
           "distinct cells %d" % len({j["cell_id"] for j in cells}),
           "smoke job      %s (%s)" % (smoke["job_id"], smoke["cell_id"]),
           "prep node      %s" % ("yes: every input the table names is checked first"
                                  if prep_node else "no"),
           "est CPU-min    %d (%.1f CPU-h; %.1f CPU-h with 20%% contingency)"
           % (est, est / 60.0, 1.2 * est / 60.0),
           "maxjobs        %d" % maxjobs,
           "dag            %s.dag" % dagbase,
           "job table      %s (every placeholder expanded)" % table,
           "sub files      sub/%s{cell,smoke,%sagg}.sub" % (pfx, "prep," if prep_node else "")]
    man += ["aggregation    %s: f=%s, %d cells" % (p["tag"], repr(p["f"]), len(p["cells"]))
            for p in plan]
    man += ["", "Every node runs the package version and the core above: wrapper.sh exits 92",
            "when the package's modules, 98 when the core's files, have changed since the DAG",
            "was written."]
    put("manifest.txt" if default_dag else "manifest_%s.txt" % dagbase, "\n".join(man) + "\n")

    for name in sorted(written):
        stream.write("wrote %s\n" % written[name])
    stream.write("nothing was submitted; TestBeam/twc_transfer/README.md (Condor) says what to "
                 "check before submitting by hand\n")
    return written
