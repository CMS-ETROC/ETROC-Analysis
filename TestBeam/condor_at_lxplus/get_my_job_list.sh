#!/usr/bin/env bash

echo "Please be patient, this script may take a while to go through the full history"

condor_history $USER | ./extract_jobs_from_list.py

