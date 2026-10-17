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
                                flag is refused, and so is a write root equal to the read root.

The seven lists of the study are in jobs/ beside this file (campaigns.JOB_LISTS gives their
order and the output root each wrote into). They are the study's own tables with their
comment lines dropped and the literal campaign roots replaced by the placeholders above:
expanding them with the default roots gives back the commands the study ran, byte for byte.
"""
import os
import re
import shlex
import sys

FIELDS = ("job_id", "cell_id", "level", "group", "est_min", "mem_mb", "command")
JOBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs")

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


def _same(a, b):
    return os.path.normpath(os.path.abspath(a)) == os.path.normpath(os.path.abspath(b))


def expand(command, campaign_roots, read_root, write_root):
    """`command` with its placeholders replaced, spacing kept.

    campaign_roots  {campaign: step-10 root}, e.g. campaigns.CAMPAIGNS
    read_root       replaces ${TT_OUT} under an input flag (READ_FLAGS)
    write_root      replaces ${TT_OUT} under --out
    A root given as None leaves its ${TT_OUT} tokens as they are."""
    if read_root is not None and write_root is not None and _same(read_root, write_root):
        raise ValueError("the write root %s is the read root: outputs would land among the "
                         "inputs" % write_root)
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
