import argparse
import getpass
import subprocess
import sys
from pathlib import Path
from typing import Dict

import yaml
from jinja2 import Template
from natsort import natsorted

# --- Configuration & Templates ---

# Template for the shell script running on the worker node
BASH_TEMPLATE = """#!/bin/bash

ls -ltrh
echo ""
pwd

# Load python environment from work node
source /cvmfs/sft.cern.ch/lcg/views/LCG_104a/x86_64-el9-gcc13-opt/setup.sh

# Copy input data from EOS to local work node
xrdcp -r root://eosuser.cern.ch/{{ path }} ./

echo "Will process input file from {{ runname }} {{ filename }}"

# Run the python script
echo "python extract_events_by_path.py -f {{ filename }} -r {{ runname }} -t {{ track }} -c {{ config }} --trigID {{ trigID }} --cal_table {{ cal_table }} --neighbor_search_method {{ search_method }}"
python extract_events_by_path.py -f {{ filename }} -r {{ runname }} -t {{ track }} -c {{ config }} --trigID {{ trigID }} --cal_table {{ cal_table }} --neighbor_search_method {{ search_method }}

ls -ltrh
echo ""

# Delete input file so condor will not return it as output
rm {{ filename }}

ls -ltrh
echo ""
"""

# Template for the Condor JDL file
JDL_TEMPLATE = """universe              = vanilla
executable            = {{ script_dir }}/run_extract_events{{ run_append }}.sh
should_Transfer_Files = YES
whenToTransferOutput  = ON_EXIT
arguments             = $(fname) $(run) $(path)
transfer_Input_Files  = extract_events_by_path.py,{{ track_file }},{{ cal_table }},{{ config_file }}
output                = {{ log_dir }}/$(ClusterId).$(ProcId).extractEvents.stdout
error                 = {{ log_dir }}/$(ClusterId).$(ProcId).extractEvents.stderr
log                   = {{ log_dir }}/$(ClusterId).$(ProcId).extractEvents.log
MY.WantOS             = "el9"
MY.XRDCP_CREATE_DIR   = True
output_destination    = root://eosuser.cern.ch/{{ eos_base }}/{{ out_dir }}
+JobFlavour           = "microcentury"
Queue fname,run,path from {{ script_dir }}/input_list_for_extractEvents{{ run_append }}.txt
"""

# --- Helper Functions ---

def get_trigger_id_from_config(config_path: Path, run_name: str) -> int:
    """Parses the YAML config to find the board ID with role 'trig'."""
    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    if run_name not in config_data:
        raise ValueError(f"Run config '{run_name}' not found in {config_path}")

    for board_id, board_info in config_data[run_name].items():
        if board_info.get('role') == 'trig':
            return board_id

    raise ValueError(f"No board with role 'trig' found in config for {run_name}")

def create_submission_files(
    args: argparse.Namespace,
    trig_id: int,
    paths: Dict[str, Path],
    run_append: str,
    eos_base: str
):
    """Generates the Input List, Bash Script, and JDL file."""

    # 1. Generate Input List
    # Finds all feather files and writes them to a text file
    input_list_path = paths['scripts_dir'] / f'input_list_for_extractEvents{run_append}.txt'
    if input_list_path.exists():
        input_list_path.unlink()

    feather_files = natsorted(Path(args.dirname).glob('loop*feather'))
    if not feather_files:
        print(f"Warning: No feather files found in {args.dirname}")

    with open(input_list_path, 'a') as f:
        run_identifier = Path(args.dirname).name.split('_feather')[0]
        for file_path in feather_files:
            # Format: filename, run_identifier, full_path
            f.write(f"{file_path.name}, {run_identifier}, {file_path}\n")

    # 2. Generate Bash Script
    bash_content = Template(BASH_TEMPLATE).render({
        'path': '${3}',          # Condor Argument 3
        'runname': '${2}',       # Condor Argument 2
        'filename': '${1}',      # Condor Argument 1
        'track': args.track,
        'cal_table': Path(args.cal_table).name,
        'trigID': trig_id,
        'search_method': args.search_method,
        'config': Path(args.config).name,
    })

    bash_script_path = paths['scripts_dir'] / f'run_extract_events{run_append}.sh'
    with open(bash_script_path, 'w') as f:
        f.write(bash_content)

    # 3. Generate JDL File
    jdl_content = Template(JDL_TEMPLATE).render({
        'script_dir': paths['scripts_dir'],
        'run_append': run_append,
        'track_file': args.track,
        'cal_table': args.cal_table,
        'log_dir': paths['log_dir'],
        'eos_base': eos_base,
        'out_dir': args.outname,
        'config_file': args.config,
    })

    jdl_path = paths['scripts_dir'] / f'condor_extract_events{run_append}.jdl'
    with open(jdl_path, 'w') as f:
        f.write(jdl_content)

    return jdl_path, bash_script_path, input_list_path

def handle_resubmission(script_name: str, input_list_path: Path):
    """
    Identifies running/idle jobs, removes them, and repopulates the input list
    to re-queue them.
    """
    condor_output = subprocess.run(['condor_q', '-nobatch'], capture_output=True, text=True)

    # Filter lines relevant to our specific script
    relevant_jobs = [line for line in condor_output.stdout.splitlines() if script_name in line]

    if not relevant_jobs:
        print('No relevant Condor jobs found to resubmit.')
        sys.exit(0)

    jobs_to_kill = set()
    requeue_lines = []

    for line in relevant_jobs:
        fields = line.split()
        if len(fields) >= 12:
            job_id = fields[0].split('.')[0]
            status = fields[5]

            # Reconstruct the input arguments (fname run path)
            # Assumes arguments are the last 3 fields
            args = fields[-3:]
            requeue_lines.append(' '.join(args))

            # Mark for deletion if not already deleted ('X')
            if status != 'X':
                jobs_to_kill.add(job_id)

    # Rewrite the input list with only the jobs we found
    with open(input_list_path, 'w') as f:
        for line in requeue_lines:
            f.write(line + '\n')

    # Kill old jobs
    for job_id in jobs_to_kill:
        print(f"Removing old job {job_id}...")
        subprocess.run(['condor_rm', job_id])

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Submit Extract Events',
        description='Submit Condor jobs to extract event data for tracks.'
    )

    parser.add_argument('-d', '--inputdir', required=True, dest='dirname', help='Input directory containing feather files')
    parser.add_argument('-t', '--track', required=True, dest='track', help='CSV file with track candidates')
    parser.add_argument('-c', '--config', required=True, dest='config', help='YAML file with run config')
    parser.add_argument('-r', '--runName', required=True, dest='runName', help='Run name in YAML config')
    parser.add_argument('--cal_table', required=True, dest='cal_table', help='CSV file with CAL mode values')
    parser.add_argument('-o', '--outdir', default='extractEvents_outputs', dest='outname', help='Output directory on EOS')
    parser.add_argument('--neighbor_search_method', default="none", dest='search_method',
                        help="Search method for neighbor hit checking, default is 'none'. possible argument: 'row_only', 'col_only', 'cross', 'square'")
    parser.add_argument('--condor_tag', dest='condor_tag', help='Tag appended to filenames to avoid collisions')
    parser.add_argument('--dryrun', action='store_true', help='Generate files but do not submit')
    parser.add_argument('--resubmit', action='store_true', help='Kill matching jobs and re-submit them')

    args = parser.parse_args()

    # --- Setup Environments ---
    username = getpass.getuser()
    eos_base_dir = f'/eos/user/{username[0]}/{username}'
    run_append = f"_{args.condor_tag}" if args.condor_tag else ""

    # Directory setup
    paths = {
        'scripts_dir': Path('./condor_scripts') / f'extractEvents_job{run_append}',
        'log_dir': Path('./condor_logs') / 'extract_events' / f'extractEvents_job{run_append}'
    }

    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    # --- Validation ---
    if not Path('extract_events_by_path.py').is_file():
        sys.exit(f"Error: Worker script extract_events_by_path.py not found in current directory.")
    if not Path(args.track).is_file():
        sys.exit(f"Error: Track file '{args.track}' not found.")
    if not Path(args.cal_table).is_file():
        sys.exit(f"Error: Cal table '{args.cal_table}' not found.")
    if not Path(args.config).is_file():
        sys.exit(f"Error: Config file '{args.config}' not found.")

    # --- Logic ---
    try:
        trig_id = get_trigger_id_from_config(Path(args.config), args.runName)
    except Exception as e:
        sys.exit(f"Configuration Error: {e}")

    print('\n========= Submission Details =========')
    print(f'Input:       {args.dirname}')
    print(f'Trigger ID:  {trig_id}')
    print(f'Output:      {eos_base_dir}/{args.outname}')
    print('======================================\n')

    jdl_file, bash_file, list_file = create_submission_files(
        args, trig_id, paths, run_append, eos_base_dir
    )

    # --- Submission ---
    if args.dryrun:
        print("--- Dry Run: Files Generated ---")
        print(f"JDL:  {jdl_file}")
        print(f"Bash: {bash_file}")
        print(f"List: {list_file}")

    elif args.resubmit:
        print("--- Resubmission Mode ---")
        handle_resubmission(bash_file.name, list_file)

        if list_file.stat().st_size > 0:
            print("Submitting new jobs...")
            subprocess.run(['condor_submit', str(jdl_file)])
        else:
            print("Input list empty after filtering. Nothing to submit.")

    else:
        # Standard Submission
        if list_file.stat().st_size > 0:
            subprocess.run(['condor_submit', str(jdl_file)])
        else:
            print("No input files found in directory. Nothing submitted.")