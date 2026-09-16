#!/usr/bin/env bash

# Usage: ./get_job_completion_info.sh <ClusterId> [max records, default 500]
# -limit stops the schedd scan once that many matching records are found;
# prefer utils/job_outcomes_from_logs.py, which reads the local job event logs and never queries the schedd.

condor_history $1 -limit "${2:-500}" -long -attributes ClusterID,ProcID,ToE | ./make_condor_summary.py

