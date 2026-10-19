#!/usr/bin/env bash
# Usage: ./get_job_completion_info.sh <job event log> [more logs ...]
#
# Summarises the jobs recorded in the local job event log(s), i.e. the file named on
# the `log = ...` line of the submit description (under condor_logs/). Uses
# condor_history -userlog, which reads that file only and never contacts the schedd;
# it works for finished and unfinished jobs alike.
#
# Never query the schedd for accounting: condor_history without -limit, or with a
# -limit larger than the number of matching jobs, scans every history file for hours
# (CERN HTCondor service notice, 2026-09-16).
set -e
[ $# -ge 1 ] || { echo "usage: $0 <job event log> [more logs ...]" >&2; exit 1; }
for log in "$@"; do
  echo "== $log"
  condor_history -userlog "$log" -af ClusterId ProcId JobStatus ExitCode ExitBySignal \
  | awk '{ n++; st[$3]++; if ($3 == 4) ec[($5 == "true") ? "signal" : "exit " $4]++ }
         END { printf "jobs %d  completed %d  running %d  idle %d  held %d  removed %d\n", n, st[4]+0, st[2]+0, st[1]+0, st[5]+0, st[3]+0;
               for (k in ec) printf "  %s x%d\n", k, ec[k] }'
done
