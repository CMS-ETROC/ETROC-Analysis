#!/usr/bin/env bash

echo "Please be patient, this script may take a while to go through the full history"

condor_history $1 -long -attributes ClusterID,ProcID,ToE | ./make_condor_summary.py

