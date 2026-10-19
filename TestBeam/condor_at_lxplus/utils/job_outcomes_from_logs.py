#!/usr/bin/env python3
"""Summarise HTCondor job outcomes from the job event logs written under condor_logs/.

Every submit file in submit/ sets `log = <log_dir>/<step>.log`; that file records
each job of the cluster (000 submitted, 001 executing, 005 terminated with the
return value, 009 aborted, 012 held). Reading it answers "did my jobs finish and
with what exit code" without any schedd query: this script never calls
condor_history or condor_q.

Usage:
  job_outcomes_from_logs.py [PATH ...] [--cluster ID ...] [--jobs]
PATH is a .log file or a directory searched recursively (default: condor_logs).
--cluster restricts the summary to those ClusterIds; --jobs lists every job.
"""
import argparse
import collections
import os
import re
import sys

EVENT_RE = re.compile(r"^(\d{3}) \((\d+)\.(\d+)\.\d+\) (\d\d/\d\d \d\d:\d\d:\d\d) (.*)$")
TERM_RE = re.compile(r"^\s+\(1\) Normal termination \(return value (-?\d+)\)")
SIG_RE = re.compile(r"^\s+\(0\) Abnormal termination \(signal (\d+)\)")
STATUS = {"000": "submitted", "001": "running", "005": "terminated", "009": "aborted",
          "012": "held", "013": "released"}


def parse_log(path, jobs):
    pending = None
    with open(path, errors="replace") as fh:
        for line in fh:
            m = EVENT_RE.match(line)
            if m:
                code, cluster, proc = m.group(1), int(m.group(2)), int(m.group(3))
                job = jobs.setdefault((cluster, proc), {"status": "submitted", "exit": None, "log": path})
                job["when"] = m.group(4)
                if code in STATUS:
                    job["status"] = STATUS[code]
                pending = job if code == "005" else None
                continue
            if pending is not None:
                t = TERM_RE.match(line)
                if t:
                    pending["exit"] = int(t.group(1))
                    pending = None
                    continue
                s = SIG_RE.match(line)
                if s:
                    pending["exit"] = "signal %s" % s.group(1)
                    pending = None


def find_logs(paths):
    for p in paths:
        if os.path.isfile(p):
            yield p
        elif os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for f in sorted(files):
                    if f.endswith(".log"):
                        yield os.path.join(root, f)
        else:
            print("no such file or directory: %s" % p, file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", default=["condor_logs"])
    ap.add_argument("--cluster", type=int, nargs="*", default=None)
    ap.add_argument("--jobs", action="store_true", help="list every job, not only the per-cluster summary")
    args = ap.parse_args()

    jobs = {}
    nlogs = 0
    for log in find_logs(args.paths):
        nlogs += 1
        parse_log(log, jobs)
    if args.cluster:
        keep = set(args.cluster)
        jobs = {k: v for k, v in jobs.items() if k[0] in keep}
    if not jobs:
        print("no jobs found in %d log file(s)" % nlogs)
        return 1

    print("%-10s %5s %5s %5s %5s %5s  %-24s  %s" % ("cluster", "jobs", "done", "run", "held", "abort", "exit codes", "log"))
    for cluster in sorted({k[0] for k in jobs}):
        cj = [v for k, v in jobs.items() if k[0] == cluster]
        status = collections.Counter(j["status"] for j in cj)
        exits = collections.Counter(str(j["exit"]) for j in cj if j["status"] == "terminated")
        exit_str = ", ".join("%s x%d" % (code, n) for code, n in sorted(exits.items()))
        logs = sorted({j["log"] for j in cj})
        print("%-10d %5d %5d %5d %5d %5d  %-24s  %s" % (
            cluster, len(cj), status["terminated"], status["running"] + status["submitted"] + status["released"],
            status["held"], status["aborted"], exit_str, logs[0] if len(logs) == 1 else "%s (+%d)" % (logs[0], len(logs) - 1)))
    bad = sorted(k for k, v in jobs.items() if v["status"] != "terminated" or v["exit"] not in (0, None))
    if bad:
        print("\nnot finished cleanly (%d):" % len(bad))
        for k in bad:
            v = jobs[k]
            print("  %d.%d  %-10s exit=%s  last event %s" % (k[0], k[1], v["status"], v["exit"], v["when"]))
    if args.jobs:
        print("\nall jobs:")
        for k in sorted(jobs):
            v = jobs[k]
            print("  %d.%d  %-10s exit=%s  %s" % (k[0], k[1], v["status"], v["exit"], v["when"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
