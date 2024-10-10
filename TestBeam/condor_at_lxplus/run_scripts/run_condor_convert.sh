#!/bin/bash

## -d: input path to directory 
## --run_name: name for output directory

run_name="Run_18"
python submit_decoding.py -d /eos/user/j/jongho/ETROC_CERN_Aug2024/${run_name} --run_name ${run_name}
