import argparse
import getpass
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

from jinja2 import Template
from natsort import natsorted

BASH_TEMPLATE = """#!/bin/bash

clusterid="$1"
procid="$2"
INPUT_LIST_FILE="$3"
BATCH_SIZE="$4"

# Load python environment from work node
source /cvmfs/sft.cern.ch/lcg/views/LCG_104a/x86_64-el9-gcc13-opt/setup.sh

LOCAL_DIR="./input_chunk_${clusterid}_${procid}"
mkdir -p $LOCAL_DIR

echo "--- Starting job ${procid} using ${INPUT_LIST_FILE} ---" # Used {} for safety

# 1. Calculate the start and end line numbers for this job
# Batch size is explicitly 10.

# Start line is (JOB_ID * BATCH_SIZE) + 1 (because line numbers start at 1)
START_LINE=$(( ($procid * $BATCH_SIZE) + 1 ))

# End line is (START_LINE + BATCH_SIZE) - 1
END_LINE=$(( ($START_LINE + $BATCH_SIZE) - 1 ))

echo "Processing lines from $START_LINE to $END_LINE"

# 2. Extract the filenames for this job using 'sed'
# 'sed -n "${START_LINE},${END_LINE}p"' prints only lines between START_LINE and END_LINE
BATCH_FILENAMES=$(sed -n "${START_LINE},${END_LINE}p" "$INPUT_LIST_FILE")

echo "\n--- Print ---\n"
ls -ltrh .

# Check if any files were extracted (for the last, possibly partial, batch)
if [ -z "$BATCH_FILENAMES" ]; then
    echo "No files found in range $START_LINE-$END_LINE. Exiting gracefully."
    exit 0
fi

# Check if any files were extracted (for the last, possibly partial, batch)
if [ -z "$BATCH_FILENAMES" ]; then
    echo "No files found in range $START_LINE-$END_LINE. Exiting gracefully."
    exit 0
fi

# 3. Loop through the extracted filenames
for FILENAME in $BATCH_FILENAMES
do
    REMOTE_FILE_PATH="{{ remote_path }}/${FILENAME}"
    LOCAL_FILE_PATH="${LOCAL_DIR}/${FILENAME}"

    echo "Transferring: $FILENAME"
    xrdcp -s root://eosuser.cern.ch/{{ remote_path }}/${FILENAME} $LOCAL_FILE_PATH

done

# 4. Print input files
echo "\n--- Copied Files ---"
ls -ltrh ${LOCAL_DIR}

echo "\nRunning: {{ command }} -d ${LOCAL_DIR}"
{{ command }} -d ${LOCAL_DIR}

echo "\nCleanup: Removing local input files."
rm -rf $LOCAL_DIR

echo "Job Finished."
ls -ltrh

echo "\n--- Job ${procid} finished successfully ---"
"""

# Template for the Condor JDL file
JDL_TEMPLATE = """universe              = vanilla
executable            = {{ script_dir }}/{{ bash_script_name }}
should_Transfer_Files = YES
whenToTransferOutput  = ON_EXIT
transfer_Input_Files  = {{ transfer_files }}
Arguments             = $(ClusterId) $(ProcId) {{ master_list_file_name }} {{ batch_size }}
output_destination    = root://eosuser.cern.ch/{{ output_dir }}
output                = {{ log_dir }}/$(ClusterId).$(ProcId).tdc.stdout
error                 = {{ log_dir }}/$(ClusterId).$(ProcId).tdc.stderr
log                   = {{ log_dir }}/$(ClusterId).$(ProcId).tdc.log
MY.WantOS             = "el9"
+JobFlavour           = "workday"
Queue {{ num_of_jobs }}
"""

def build_python_command_args(args: argparse.Namespace) -> str:
    """Constructs the python command arguments string dynamically."""
    cmd_parts = [
        'python apply_tdc_cuts.py',
        f'-c {Path(args.config).name}', # Condor transfers the config, use only the filename
        f'-r {args.runName}',
        f'--distance_factor {args.distance_factor}',
        f'--TOALower {args.TOALower}',
        f'--TOAUpper {args.TOAUpper}',
        f'--dutTOTlower {args.dutTOTlower}',
        f'--dutTOTupper {args.dutTOTupper}',
        f'--TOALowerTime {args.TOALowerTime}',
        f'--TOAUpperTime {args.TOAUpperTime}',
        f"--exclude_role {args.exclude_role}",
    ]
    if args.convert_first: cmd_parts.append("--convert-first")
    return " ".join(cmd_parts)

def create_master_file_list(input_group_dir: Path, output_dir: Path) -> Optional[Path]:
    """
    Scans the input directory for files to process, creates a sorted list of their
    paths relative to the mother_dir, and saves it to a temporary file.

    Returns the path to the temporary list file.
    """
    # 1. Identify all track files (e.g., .root files)
    all_files = natsorted([f for f in input_group_dir.iterdir() if f.suffix == '.pkl'])
    absolute_filenames = [f.name for f in all_files]

    if not all_files:
        return None

    # 3. Save the list to a temporary file in the script directory
    list_file_name = f"{input_group_dir.name}_file_list.txt"
    list_file_path = output_dir / list_file_name # Save it one level above the tracks dir

    with open(list_file_path, 'w') as f:
        f.write('\n'.join(absolute_filenames) + '\n')

    print(f"    Generated master list with {len(absolute_filenames)} files: {list_file_path.name}")
    return list_file_path, len(absolute_filenames)

def create_jdl_file(args, master_list_path, run_append, group_name, njobs):
    jdl_content = Template(JDL_TEMPLATE).render({
        'script_dir': script_dir.as_posix(),
        'bash_script_name': f'applyTDC_job{run_append}_{group_name}.sh',
        'master_list_file_name': f'{master_list_path.name}',
        'transfer_files': f"apply_tdc_cuts.py, {Path(args.config).as_posix()}, {master_list_path.as_posix()}",
        'output_dir': f"{args.inputdir}/{group_name.replace('tracks','time')}",
        'log_dir': log_dir.as_posix(),
        'batch_size': args.batch_size,
        'num_of_jobs': njobs,
    })

    jdl_path = script_dir / f'condor_tdccuts_{group_name}.jdl'
    with open(jdl_path, 'w') as f:
        f.write(jdl_content)

    return jdl_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Submit TDC Cuts',
        description='Submit Condor jobs to apply cuts to track files.'
    )

    # Paths
    parser.add_argument('-d', '--inputdir', required=True, dest='inputdir',
                        help='Mother directory containing "tracks" or "tracks_groupX" folders')

    # Naming Logic (Retained for prefix/suffix customization of the output dir)
    parser.add_argument('--prefix', default='', help='Add a prefix to the output folder name')
    parser.add_argument('--suffix', default='', help='Add a suffix to the output folder name')

    # Config
    parser.add_argument('-c', '--config', required=True, help='YAML config file')
    parser.add_argument('-r', '--runName', required=True, help='Run name')

    # Cuts
    parser.add_argument('--distance_factor', type=float, default=3.0, help='Correlation cut sigma')
    parser.add_argument('--TOALower', type=int, default=100, help='Raw ToA Lower')
    parser.add_argument('--TOAUpper', type=int, default=500, help='Raw ToA Upper')
    parser.add_argument('--TOALowerTime', type=float, default=2, help='Time ToA Lower (ns)')
    parser.add_argument('--TOAUpperTime', type=float, default=10, help='Time ToA Upper (ns)')
    parser.add_argument('--dutTOTlower', type=int, default=1, help='DUT ToT Lower Percentile')
    parser.add_argument('--dutTOTupper', type=int, default=96, help='DUT ToT Upper Percentile')

    # Flags
    parser.add_argument('--exclude_role', default='trig', help='Role to exclude from CUT calculations')
    parser.add_argument('--convert-first', action='store_true', help='Convert to time before cutting')

    # Condor options
    parser.add_argument('--batch_size', default=10, dest='batch_size', help='Number of files per job')
    parser.add_argument('--condor_tag', dest='condor_tag', help='Tag appended to filenames to avoid collisions')
    parser.add_argument('--dryrun', action='store_true', help='Generate files but do not submit')
    parser.add_argument('--resubmit', action='store_true', help='Kill matching jobs and re-submit them')

    args = parser.parse_args()

    # --- Setup Environments ---
    username = getpass.getuser()

    # Determine the user's EOS base directory structure (e.g., /eos/user/j/jongho)
    # This assumes the input directory path is under this root.
    eos_base_dir = Path(f'/eos/user/{username[0]}/{username}')
    run_append = f"_{args.condor_tag}" if args.condor_tag else ""

    tag = f"applyTDC{run_append}"

    script_dir = Path('condor_scripts') / tag
    log_dir_base = Path('condor_logs') / tag

    script_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Identify Input/Output Groups ---
    mother_dir = Path(args.inputdir)
    track_dirs = sorted([d for d in mother_dir.iterdir() if d.is_dir() and d.name.startswith('tracks')])

    if not track_dirs:
        if mother_dir.name.startswith('tracks'):
            track_dirs = [mother_dir]
            mother_dir = mother_dir.parent
        else:
            sys.exit(f"No 'tracks*' directories found in {mother_dir}")

    python_cmd = build_python_command_args(args)

    # --- 3. Process Each Group ---
    print(f"\nScanning: {mother_dir}")
    print(f"Found {len(track_dirs)} track groups: {[d.name for d in track_dirs]}")

    for input_group_dir in track_dirs:
        dir_name = input_group_dir.name

        # Generate the master file list for this group
        list_info = create_master_file_list(input_group_dir, script_dir)

        if list_info is None:
            print(f"    No files found to process for {dir_name}. Skipping.")
            continue

        master_list_path, num_files = list_info

        # Calculate number of jobs
        batch_size = int(args.batch_size)
        num_of_jobs = (num_files + batch_size - 1) // batch_size # Ceiling division

        # Log directory (local)
        log_dir = log_dir_base / dir_name
        log_dir.mkdir(parents=True, exist_ok=True)

        bash_path = script_dir / f'applyTDC_job{run_append}_{dir_name}.sh'
        with open(bash_path, 'w') as f:
            f.write(Template(BASH_TEMPLATE).render({'command': python_cmd, 'remote_path': f'{args.inputdir}/{dir_name}'}))

        jdl_file = create_jdl_file(args, master_list_path, run_append, dir_name, num_of_jobs)
        print(f"\n>>> Preparing Group: {dir_name}")

        # --- Submission ---
        if args.dryrun:
            print(f"    [Dry Run] Generated JDL: {jdl_file}")
        else:
            # Standard Submission
            print(f"    Submitting {jdl_file}...")
            subprocess.run(['condor_submit', str(jdl_file)], check=True)

    print("\nSubmission process complete.")