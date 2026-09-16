#!/usr/bin/env bash

# Usage: ./get_my_job_list.sh [max records, default 200]
# -limit stops the schedd scan once that many records are found; never run condor_history unbounded.

condor_history $USER -limit "${1:-200}" | ./extract_jobs_from_list.py

