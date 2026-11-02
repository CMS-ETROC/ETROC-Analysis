#!/usr/bin/env bash
# Usage: ./get_my_job_list.sh [condor_logs directory, default ./condor_logs]
#
# Lists every cluster found in the job event logs under the given directory with the
# condor_q -userlog totals line (completed/idle/running/held). Reads local files only;
# it never contacts the schedd. See get_job_completion_info.sh for exit codes and
# job_outcomes_from_logs.py for the per-cluster table.
set -e
dir="${1:-condor_logs}"
[ -d "$dir" ] || { echo "no such directory: $dir" >&2; exit 1; }
find "$dir" -name "*.log" -type f | sort | while read -r log; do
  echo "== $log"
  condor_q -userlog "$log" 2>/dev/null | grep "^Total for query" || echo "  (no jobs parsed)"
done
