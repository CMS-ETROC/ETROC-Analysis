import argparse
import subprocess
import sys
import re
from pathlib import Path
from jinja2 import Template
from natsort import natsorted
from typing import List, Dict, Optional

WORKER_SCRIPT_NAME = "bootstrap.py"

BASH_TEMPLATE = """#!/bin/bash

# $1: The full EOS path of the input file (e.g., /path/to/time/data_01.pkl)
INPUT_FILE_EOS="$1"

# Load python environment from work node
source /cvmfs/sft.cern.ch/lcg/views/LCG_104a/x86_64-el9-gcc13-opt/setup.sh

# 1. Determine local file name
LOCAL_FILENAME=$(basename "$INPUT_FILE_EOS")

# 2. Copy input data from EOS to local work node
echo "Transferring: $INPUT_FILE_EOS"
xrdcp root://eosuser.cern.ch/$INPUT_FILE_EOS ./$LOCAL_FILENAME

# 3. Construct and Run Bootstrap Analysis
echo "\nRunning: {{ command }} -f $LOCAL_FILENAME"
{{ command }} -f $LOCAL_FILENAME

# 4. Cleanup: Delete the local copy of the input file
if [ -f $LOCAL_FILENAME ]; then
    rm $LOCAL_FILENAME
fi

echo "\n--- Job finished successfully ---"
"""

JDL_TEMPLATE = """universe              = vanilla
executable            = {{ script_dir }}/{{ bash_script_name }}
should_Transfer_Files = YES
whenToTransferOutput  = ON_EXIT
transfer_Input_Files  = {{ transfer_files }}
Arguments             = $(path)
TransferOutputRemaps  = "$(stem)_boot.parquet={{ out_dir }}/$(stem)_boot.parquet"
output                = {{ log_dir }}/$(ClusterId).$(ProcId).bootstrap.stdout
error                 = {{ log_dir }}/$(ClusterId).$(ProcId).bootstrap.stderr
log                   = {{ log_dir }}/$(ClusterId).$(ProcId).bootstrap.log
MY.WantOS             = "el9"
+JobFlavour           = "workday"
Queue stem, path from {{ script_dir }}/{{ master_list_file_name }}
"""

def build_python_command(args: argparse.Namespace) -> str:
    """
    Constructs the python command string dynamically using a placeholder for the filename.
    The bash script will substitute 'FILENAME_PLACEHOLDER' with the local filename.
    """
    neighbor_cut_str = " ".join(args.neighbor_cut)

    cmd_parts = [
        f"python {WORKER_SCRIPT_NAME}",
        f"-n {args.num_bootstrap_output}",
        f"-s {args.sampling}",
        f"--minimum_nevt {args.minimum_nevt}",
        f"--iteration_limit {args.iteration_limit}",
        f"--neighbor_cut {neighbor_cut_str}",
        f"--neighbor_logic {args.neighbor_logic}",
    ]

    if args.reproducible: cmd_parts.append("--reproducible")

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
    Generates the input list (stem, full EOS path), bash script, and JDL file.
    """

    # 1. Generate Input List (stem, full EOS path)
    master_list_file_name = f'input_list_for_bootstrap{unique_tag}.txt'
    input_list_path = paths['scripts'] / master_list_file_name
    if input_list_path.exists():
        input_list_path.unlink()

    # File discovery (PKL or PARQUET exclusive)
    pkl_files = list(input_dir.glob('*.pkl'))
    parquet_files = list(input_dir.glob('*.parquet'))

    files: List[Path] = []
    if pkl_files:
        print(f"    Found {len(pkl_files)} PKL files. Ignoring any Parquet files.")
        files = pkl_files
    elif parquet_files:
        print(f"    Found {len(parquet_files)} Parquet files. Proceeding with Parquet.")
        files = parquet_files

    files = natsorted(files)
    if not files:
        print(f"Warning: No pickle or parquet files found in {input_dir.name}. Skipping group.")
        return None, None, None

    # Write: stem_name, logical_path
    with open(input_list_path, 'a') as f:
        for file_path in files:
            stem_name = file_path.stem

            # --- PATH FIX: Enforce /eos/user/ instead of /eos/home-X/ ---
            abs_path = str(file_path.resolve())
            logical_path = re.sub(r'^/eos/home-([a-z0-9])/', r'/eos/user/\1/', abs_path)

            # JDL arguments will be: stem, path
            f.write(f"{stem_name},{logical_path}\n")

    # 2. Generate Bash Script
    bash_script_name = f'run_bootstrap{unique_tag}.sh'

    # MODIFICATION: build_python_command no longer takes filename_val
    command = build_python_command(args)

    bash_content = Template(BASH_TEMPLATE).render({
        'command': command
    })

    bash_path = paths['scripts'] / bash_script_name
    with open(bash_path, 'w') as f:
        f.write(bash_content)

    # 3. Generate JDL File
    transfer_list = [WORKER_SCRIPT_NAME]

    jdl_content = Template(JDL_TEMPLATE).render({
        'script_dir': str(paths['scripts']),
        'bash_script_name': bash_script_name, # Pass bash script name for JDL execution
        'master_list_file_name': master_list_file_name, # Pass master list name for Queue
        'out_dir': str(group_out_dir),
        'log_dir': str(group_log_dir),
        'transfer_files': ", ".join(transfer_list),
        'unique_tag': unique_tag,
    })

    jdl_path = paths['scripts'] / f'condor_bootstrap{unique_tag}.jdl'
    with open(jdl_path, 'w') as f:
        f.write(jdl_content)

    return jdl_path, bash_script_name, input_list_path

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
    parser.add_argument('--reproducible', action='store_true')
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
            print(f"    [Dry Run] Bash: {bash}")
            print(f"    [Dry Run] Input: {list_file}")

        else:
            if list_file.stat().st_size > 0:
                print(f"    Submitting jobs...")
                subprocess.run(['condor_submit', str(jdl)])
            else:
                print("    Input list empty.")

    print("\nDone.")