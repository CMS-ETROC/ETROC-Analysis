import argparse
import subprocess
import sys
import re
from pathlib import Path
from jinja2 import Template
from natsort import natsorted
from typing import List, Dict, Optional

# --- Configuration & Templates ---

WORKER_SCRIPT_NAME = "bootstrap.py"

BASH_TEMPLATE = """#!/bin/bash

ls -ltrh
echo ""
pwd

# Note: Input file path is passed as argument 2 ($2).
# Assumes the 'path' argument is an EOS path to copy.

# Load python environment from work node
source /cvmfs/sft.cern.ch/lcg/views/LCG_104a/x86_64-el9-gcc13-opt/setup.sh

# Copy input data from EOS to local work node
xrdcp root://eosuser.cern.ch/{{ input_file }} ./

# Run Bootstrap Analysis
echo "Running: {{ command }}"
{{ command }}

# Cleanup: Delete the local copy of the input file
if [ -f {{ filename }} ]; then
    rm {{ filename }}
fi

ls -ltrh
echo ""
"""

JDL_TEMPLATE = """universe              = vanilla
executable            = {{ script_dir }}/run_bootstrap{{ unique_tag }}.sh
should_Transfer_Files = YES
whenToTransferOutput  = ON_EXIT
arguments             = $(ifile) $(path)
transfer_Input_Files  = {{ transfer_files }}
TransferOutputRemaps  = "$(ifile)_resolution.pkl={{ out_dir }}/$(ifile)_resolution.pkl; $(ifile)_gmmInfo.pkl={{ out_dir }}/$(ifile)_gmmInfo.pkl"
output                = {{ log_dir }}/$(ClusterId).$(ProcId).bootstrap.stdout
error                 = {{ log_dir }}/$(ClusterId).$(ProcId).bootstrap.stderr
log                   = {{ log_dir }}/$(ClusterId).$(ProcId).bootstrap.log
MY.WantOS             = "el9"
+JobFlavour           = "workday"
Queue ifile,path from {{ script_dir }}/input_list_for_bootstrap{{ unique_tag }}.txt
"""

# --- Helper Functions ---

def build_python_command(args: argparse.Namespace, filename_val: str) -> str:
    """Constructs the python command string dynamically."""
    neighbor_cut_str = " ".join(args.neighbor_cut)

    cmd_parts = [
        f"python {WORKER_SCRIPT_NAME}",
        f"-f {filename_val}",
        f"-n {args.num_bootstrap_output}",
        f"-s {args.sampling}",
        f"--minimum_nevt {args.minimum_nevt}",
        f"--iteration_limit {args.iteration_limit}",
        f"--neighbor_cut {neighbor_cut_str}",
        f"--neighbor_logic {args.neighbor_logic}",
    ]

    if args.reproducible: cmd_parts.append("--reproducible")
    if args.force_twc:    cmd_parts.append("--force-twc")
    if args.single:       cmd_parts.append("--single")
    if args.twc_coeffs:   cmd_parts.append(f"--twc_coeffs {args.twc_coeffs}")

    return " ".join(cmd_parts)

def create_submission_files(
    args: argparse.Namespace,
    paths: Dict[str, Path],
    unique_tag: str,
    input_dir: Path,
    group_out_dir: Path,
    group_log_dir: Path
):
    """
    Generates the input list, bash script, and JDL file using relative paths.
    """

    # 1. Generate Input List
    input_list_path = paths['scripts'] / f'input_list_for_bootstrap{unique_tag}.txt'
    if input_list_path.exists():
        input_list_path.unlink()

    pkl_files = list(input_dir.glob('*.pkl'))
    parquet_files = list(input_dir.glob('*.parquet'))

    files: List[Path] = []

    if pkl_files:
        print(f"    Found {len(pkl_files)} PKL files. Ignoring any Parquet files.")
        files = pkl_files
    elif parquet_files:
        print(f"    Found {len(parquet_files)} Parquet files. Proceeding with Parquet.")
        files = parquet_files
    else:
        # files list remains empty
        pass

    files = natsorted(files)
    if not files:
        print(f"Warning: No pickle files found in {input_dir.name}. Skipping group.")
        return None, None, None

    with open(input_list_path, 'a') as f:
        for file_path in files:
            name = file_path.stem

            # --- PATH FIX: Enforce /eos/user/ instead of /eos/home-X/ ---
            abs_path = str(file_path.resolve())

            # Regex match: start of string, /eos/home-, one alphanumeric char (group 1), /
            # Replace with: /eos/user/, group 1, /
            # Example: /eos/home-j/jongho -> /eos/user/j/jongho
            logical_path = re.sub(r'^/eos/home-([a-z0-9])/', r'/eos/user/\1/', abs_path)

            f.write(f"{name}, {logical_path}\n")

    # 2. Generate Bash Script
    filename_var = '${1}.pkl'
    command = build_python_command(args, filename_var)

    bash_content = Template(BASH_TEMPLATE).render({
        'input_file': '${2}',
        'filename': filename_var,
        'command': command
    })

    bash_script_name = f'run_bootstrap{unique_tag}.sh'
    bash_path = paths['scripts'] / bash_script_name
    with open(bash_path, 'w') as f:
        f.write(bash_content)

    # 3. Generate JDL File
    transfer_list = [WORKER_SCRIPT_NAME]
    if args.twc_coeffs:
        transfer_list.append(args.twc_coeffs)

    jdl_content = Template(JDL_TEMPLATE).render({
        'script_dir': str(paths['scripts']),
        'unique_tag': unique_tag,
        'out_dir': str(group_out_dir),
        'log_dir': str(group_log_dir),
        'transfer_files': ", ".join(transfer_list)
    })

    jdl_path = paths['scripts'] / f'condor_bootstrap{unique_tag}.jdl'
    with open(jdl_path, 'w') as f:
        f.write(jdl_content)

    return jdl_path, bash_script_name, input_list_path

def handle_resubmission(
    mode: str,
    script_name: str,
    list_path: Path,
    log_dir: Path
):
    """
    Handles logic for --resubmit (kill & rerun) and --resubmit_with_stderr (retry failed).
    """
    condor_cmd = ['condor_q', '-nobatch']
    res = subprocess.run(condor_cmd, capture_output=True, text=True)
    active_jobs = [l for l in res.stdout.splitlines() if script_name in l]

    if mode == 'stderr':
        if active_jobs:
            print(f"Skipping {script_name}: Cannot resubmit while jobs are running.")
            return

        failed_indices = []
        for log in log_dir.glob('*.stderr'):
            if log.stat().st_size > 0:
                try:
                    proc_id = int(log.name.split('.')[1])
                    failed_indices.append(proc_id)
                except IndexError:
                    pass

        if not failed_indices:
            print("No failed jobs found.")
            return

        with open(list_path, 'r') as f:
            all_lines = f.readlines()

        with open(list_path, 'w') as f:
            for idx in sorted(failed_indices):
                if idx < len(all_lines):
                    f.write(all_lines[idx])

        print(f"Resubmitting {len(failed_indices)} failed jobs.")

    elif mode == 'kill_and_rerun':
        requeue_lines = []
        jobs_to_kill = set()

        for line in active_jobs:
            parts = line.split()
            if len(parts) >= 10:
                job_id = parts[0].split('.')[0]
                status = parts[5]
                args = parts[-2:]
                requeue_lines.append(f"{args[0]}, {args[1]}")
                if status != 'X':
                    jobs_to_kill.add(job_id)

        if jobs_to_kill:
            for jid in jobs_to_kill:
                subprocess.run(['condor_rm', jid])

            with open(list_path, 'w') as f:
                for l in requeue_lines:
                    f.write(l + '\n')
            print(f"Killed {len(jobs_to_kill)} clusters. Resubmitting {len(requeue_lines)} jobs.")
        else:
            print("No active jobs found to kill/resubmit.")

# --- Main ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Submit Bootstrap Analysis to Condor')

    # Required
    parser.add_argument('-d', '--inputdir', required=True, dest='dirname', help='Mother directory containing time/time_group folders')
    parser.add_argument('-o', '--outputdir', required=True, dest='outputdir', help='Output directory base name')

    # Bootstrap Params
    parser.add_argument('-n', '--num_bootstrap_output', type=int, default=100)
    parser.add_argument('-s', '--sampling', type=int, default=75)
    parser.add_argument('--minimum_nevt', type=int, default=1000)
    parser.add_argument('--iteration_limit', type=int, default=7500)

    # Options
    parser.add_argument('--condor_tag', dest='condor_tag', help='Tag for filenames')
    parser.add_argument('--twc_coeffs', help='Pre-calculated TWC coeffs file')
    parser.add_argument('--reproducible', action='store_true')
    parser.add_argument('--force-twc', action='store_true')
    parser.add_argument('--single', action='store_true', help='Single shot mode')
    parser.add_argument('--neighbor_cut', dest='neighbor_cut', default=['none'], nargs='+',
                        help='Specify one or more **space-separated** board columns to be used for neighbor cuts. '
                        'The argument collects all values into a list. '
                        'Possible columns: HasNeighbor_dut, HasNeighbor_ref, HasNeighbor_extra, HasNeighbor_trig, trackNeighbor. '
                        'Default value is a list containing only "none".')
    parser.add_argument('--neighbor_logic', dest='neighbor_logic', default='OR',
                        help='Logic for multiple neighbor cuts on board. Default is OR. AND is possble.')

    # Execution Modes
    parser.add_argument('--dryrun', action='store_true')
    parser.add_argument('--resubmit', action='store_true', help='Kill & Rerun active jobs')
    parser.add_argument('--resubmit_with_stderr', action='store_true', help='Rerun failed jobs')

    args = parser.parse_args()

    # --- 1. Identify Groups ---
    mother_dir = Path(args.dirname).resolve()

    if mother_dir.name.find('time') != -1:
        time_dirs = [mother_dir]
    else:
        # Find all directories containing 'time'
        all_time_dirs = sorted([d for d in mother_dir.iterdir() if d.is_dir() and 'time' in d.name])

        # Check if we have split groups (e.g., time_group1)
        has_groups = any('group' in d.name for d in all_time_dirs)

        if has_groups:
            # If groups exist, filter out the bare 'time' folder to avoid duplicates
            time_dirs = [d for d in all_time_dirs if 'group' in d.name]
            # Optional: Print what was excluded
            excluded = set(all_time_dirs) - set(time_dirs)
            if excluded:
                print(f"Detected split groups. Excluding base folders: {[d.name for d in excluded]}")
        else:
            # If no groups, just process whatever was found (e.g. just 'time')
            time_dirs = all_time_dirs

    if not time_dirs:
        sys.exit(f"No directories containing 'time' found in {mother_dir}")

    print(f"Found {len(time_dirs)} groups: {[d.name for d in time_dirs]}")

    # --- 2. Setup Base Paths ---
    run_append = f"_{args.condor_tag}" if args.condor_tag else ""

    # Only script paths are global; Output/Log paths are calculated per group
    # Note: Using relative paths here to keep JDL portable
    paths = {
        'scripts': Path('condor_scripts') / f'bootstrap_job{run_append}',
        'logs':    Path('condor_logs') / 'bootstrap' / f'bootstrap_job{run_append}',
    }

    # Create Base Directories
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    if not Path(WORKER_SCRIPT_NAME).is_file():
        sys.exit(f"Error: Worker script '{WORKER_SCRIPT_NAME}' not found.")

    # --- 3. Process Each Group ---
    for group_dir in time_dirs:
        group_name = group_dir.name

        # Calculate Suffix: time -> "", time_group1 -> _group1
        group_suffix = group_name.replace('time', '')

        # 1. Output Directory: bootstrap_OutName_group1
        unique_out_name = f"bootstrap_{args.outputdir}{group_suffix}"
        group_out_dir = Path(unique_out_name)

        # 2. Log Directory: condor_logs/.../time_group1
        group_log_dir = paths['logs'] / group_name

        if not (args.dryrun or args.resubmit or args.resubmit_with_stderr):
            try:
                group_out_dir.mkdir(parents=True, exist_ok=True)
                group_log_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                print(f"Error creating directories for {group_name}: {e}")
                continue

        unique_group_tag = f"{run_append}_{group_name}"

        print(f"\n>>> Processing Group: {group_name}")
        print(f"    Out: {group_out_dir}")
        print(f"    Log: {group_log_dir}")

        jdl, bash, list_file = create_submission_files(
            args, paths, unique_group_tag, group_dir, group_out_dir, group_log_dir
        )

        if not jdl: continue

        # --- Submission Logic ---
        if args.dryrun:
            print(f"    [Dry Run] JDL: {jdl}")

        elif args.resubmit:
            handle_resubmission('kill_and_rerun', bash, list_file, group_log_dir)
            if list_file.stat().st_size > 0:
                print("    Resubmitting...")
                subprocess.run(['condor_submit', str(jdl)])

        elif args.resubmit_with_stderr:
            handle_resubmission('stderr', bash, list_file, group_log_dir)
            if list_file.stat().st_size > 0:
                print("    Resubmitting failed jobs...")
                subprocess.run(['condor_submit', str(jdl)])

        else:
            if list_file.stat().st_size > 0:
                print(f"    Submitting jobs...")
                subprocess.run(['condor_submit', str(jdl)])
            else:
                print("    Input list empty.")

    print("\nDone.")