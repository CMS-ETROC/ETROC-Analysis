import argparse
import getpass
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from jinja2 import Template
from natsort import natsorted

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'core'))
import io_utils

BASH_TEMPLATE = """#!/bin/bash
""" + io_utils.BASH_STRICT_HEADER + """
clusterid="$1"
procid="$2"
INPUT_LIST_FILE="$3"
BATCH_SIZE="$4"

# Load python environment from work node
# LCG's setup.sh references its own internal vars (e.g. COMPILER) without
# defaults, which trips our `set -u` above even though it's not a real error.
set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_104a/x86_64-el9-gcc13-opt/setup.sh
set -u

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

# Check if any files were extracted (for the last, possibly partial, batch)
if [ -z "$BATCH_FILENAMES" ]; then
    echo "No files found in range $START_LINE-$END_LINE. Exiting gracefully."
    exit 0
fi

# 3. Loop through the extracted filenames
for FILENAME in $BATCH_FILENAMES
do
    echo "Transferring: $FILENAME"
    xrdcp -s root://eosuser.cern.ch/{{ remote_path }}/${FILENAME} ${LOCAL_DIR}/${FILENAME}
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
log                   = {{ log_dir }}/tdc.log
MY.WantOS             = "el9"
+JobFlavour           = "workday"
{% if concurrency_limit -%}
concurrency_limits    = etroc_apply_tdc_cuts:{{ concurrency_limit }}
{% endif -%}
Queue {{ num_of_jobs }}
"""

def build_python_command_args(args: argparse.Namespace, script_to_run: str) -> str:
    """Constructs the python command arguments string dynamically."""
    cmd_parts = [
        f'python {script_to_run}',
        f'-c {Path(args.config).name}', # Condor transfers the config, use only the filename
        f'-r {args.runName}',
        f'--distance_factor {args.distance_factor}',
        f'--TOALower {args.TOALower}',
        f'--TOAUpper {args.TOAUpper}',
        f'--TOALowerTime {args.TOALowerTime}',
        f'--TOAUpperTime {args.TOAUpperTime}',
    ]
    if args.convert_first: cmd_parts.append("--convert-first")
    return " ".join(cmd_parts)

def create_master_file_list(input_group_dir: Path, output_dir: Path, label: str) -> Optional[Path]:
    """
    Scans the input directory for files to process, creates a sorted list of their
    paths relative to the mother_dir, and saves it to a temporary file.

    Returns the path to the temporary list file.
    """
    # 1. Identify all track files
    allowed_extensions = {'.parquet'}
    all_files = natsorted([
        f for f in input_group_dir.iterdir()
        if f.suffix in allowed_extensions
    ])

    absolute_filenames = [f.name for f in all_files]

    if not all_files:
        return None

    # 3. Save the list to a temporary file in the script directory. Keyed by
    # `label` (not just input_group_dir.name) so two board combos with a
    # same-named group (e.g. both have a plain "tracks" dir) don't overwrite
    # each other's master list within this run's script_dir.
    list_file_name = f"{label}_file_list.txt"
    list_file_path = output_dir / list_file_name # Save it one level above the tracks dir

    with open(list_file_path, 'w') as f:
        f.write('\n'.join(absolute_filenames) + '\n')

    print(f"    Generated master list with {len(absolute_filenames)} files: {list_file_path.name}")
    return list_file_path, len(absolute_filenames)

def create_jdl_file(args, group_parent_dir, master_list_path, group_label, dir_name, njobs, script_to_run):
    jdl_content = Template(JDL_TEMPLATE).render({
        'script_dir': script_dir.as_posix(),
        'bash_script_name': f'run_applyTDC_{group_label}.sh',
        'master_list_file_name': f'{master_list_path.name}',
        'transfer_files': io_utils.build_transfer_files(script_to_run, args.config, master_list_path),
        # group_parent_dir is the group's actual immediate parent -- either
        # the mother dir directly, or a combo subdirectory of it when -d
        # holds multiple board combos -- so the "time" output lands next to
        # the "tracks" dir it came from either way.
        'output_dir': f"{group_parent_dir}/{dir_name.replace('tracks','time')}",
        'log_dir': log_dir.as_posix(),
        'batch_size': args.batch_size,
        'num_of_jobs': njobs,
        'concurrency_limit': args.concurrency_limit,
    })

    jdl_path = script_dir / f'condor_applyTDC_{group_label}.jdl'
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
                        help='Mother directory containing "tracks" or "tracks_groupX" folders (output of '
                             'step 9). Can also be the combo mother directory (one subdirectory per board '
                             'combo, each with its own "tracks"/"tracks_groupX" folders) -- every combo is '
                             'then auto-detected and processed in one submission.')

    # Config
    parser.add_argument('-c', '--config', required=True, help='YAML config file')
    parser.add_argument('-r', '--runName', required=True, help='Run name')

    # Cuts
    parser.add_argument('--distance_factor', type=float, default=3.0, help='Correlation cut sigma')
    parser.add_argument('--TOALower', type=int, default=100, help='Raw ToA Lower')
    parser.add_argument('--TOAUpper', type=int, default=500, help='Raw ToA Upper')
    parser.add_argument('--TOALowerTime', type=float, default=2, help='Time ToA Lower (ns)')
    parser.add_argument('--TOAUpperTime', type=float, default=10, help='Time ToA Upper (ns)')

    # Flags
    parser.add_argument('--convert-first', action='store_true', help='Convert to time before cutting')

    # Condor options
    parser.add_argument('--batch_size', type=int, default=10, dest='batch_size', help='Number of files per job')
    parser.add_argument('--condor_tag', dest='condor_tag', help='Tag appended to filenames to avoid collisions')
    parser.add_argument('--concurrency_limit', type=int, default=None,
                        help='Cap on concurrently running jobs, shared pool-wide across every '
                             'submission of this script (via HTCondor concurrency_limits). Unset '
                             'by default -- a single run submission is not throttled. Set this when '
                             'you are about to submit several runs around the same time and want to '
                             'bound their combined memory footprint on the condor pool (e.g. 30).')
    parser.add_argument('--dryrun', action='store_true', help='Generate files but do not submit')

    args = parser.parse_args()

    # --- Setup Environments ---
    username = getpass.getuser()

    # Determine the user's EOS base directory structure (e.g., /eos/user/j/jongho)
    # This assumes the input directory path is under this root.
    eos_base_dir = io_utils.eos_base_dir(username)

    if args.condor_tag:
        run_append = args.condor_tag
    else:
        # Auto-generate a unique tag rather than falling back to a shared bucket name -
        # otherwise a second untagged submission can overwrite run_applyTDC_*.sh/file_list.txt
        # while an earlier untagged submission is still queued and hasn't been dispatched yet.
        run_append = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        print(f"No --condor_tag given; auto-generated tag '{run_append}' to avoid collisions with other submissions.")

    script_dir =  Path('.') / 'condor_scripts' / 'apply_TDC' / f'{run_append}'
    log_dir_base = Path ('.') / 'condor_logs' / 'apply_TDC' / f'{run_append}'

    script_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Identify Input/Output Groups ---
    def find_track_group_dirs(d: Path) -> List[Path]:
        return sorted([sub for sub in d.iterdir() if sub.is_dir() and sub.name.startswith('tracks')])

    mother_dir = Path(f'{eos_base_dir}/{args.inputdir}')
    track_dirs = find_track_group_dirs(mother_dir)

    if not track_dirs:
        # Distinguish "-d points directly at a single tracks/tracks_groupX
        # dir" from "-d is a combo mother directory" by content, not name --
        # a name check (e.g. mother_dir.name.startswith('tracks')) would
        # misfire on a perfectly natural combo mother dir name like
        # "tracks_run2" and skip the combo descent below entirely.
        if any(mother_dir.glob('*.parquet')):
            track_dirs = [mother_dir]
            mother_dir = mother_dir.parent
        else:
            # -d may be the combo mother directory step 9 can now write (one
            # subdirectory per board combo, each itself containing
            # tracks/tracks_groupX) -- descend one level and collect every
            # combo's track groups instead of requiring one submission per combo.
            for combo_dir in sorted(d for d in mother_dir.iterdir() if d.is_dir()):
                track_dirs.extend(find_track_group_dirs(combo_dir))
            if not track_dirs:
                sys.exit(f"No 'tracks*' directories found in {mother_dir} or its subdirectories")

    # --- 3. Process Each Group ---
    print(f"\nScanning: {mother_dir}")
    print(f"Found {len(track_dirs)} track groups: {[d.name for d in track_dirs]}")

    script_to_run = "apply_tdc_cuts.py"
    failures = 0

    for input_group_dir in track_dirs:
        dir_name = input_group_dir.name
        group_parent_dir = input_group_dir.parent

        # When -d holds multiple combos, input_group_dir.parent is the combo
        # subdirectory rather than mother_dir itself -- prefix local artifact
        # names (script/log/master-list) with the combo label so two combos'
        # same-named groups (e.g. both a plain "tracks" dir) don't overwrite
        # each other's files within this run's script_dir/log_dir.
        combo_label = group_parent_dir.name if group_parent_dir != mother_dir else None
        group_label = f'{combo_label}_{dir_name}' if combo_label else dir_name

        python_cmd = build_python_command_args(args, script_to_run)

        # Generate the master file list for this group
        list_info = create_master_file_list(input_group_dir, script_dir, group_label)

        if list_info is None:
            print(f"    No files found to process for {group_label}. Skipping.")
            continue

        master_list_path, num_files = list_info

        # Calculate number of jobs
        batch_size = args.batch_size
        num_of_jobs = (num_files + batch_size - 1) // batch_size # Ceiling division

        # Log directory (local)
        log_dir = log_dir_base / group_label
        log_dir.mkdir(parents=True, exist_ok=True)

        bash_path = script_dir / f'run_applyTDC_{group_label}.sh'
        with open(bash_path, 'w') as f:
            f.write(Template(BASH_TEMPLATE).render({
                'command': python_cmd,
                'remote_path': str(input_group_dir),
            }))

        jdl_file = create_jdl_file(args, group_parent_dir, master_list_path, group_label, dir_name, num_of_jobs, script_to_run)
        print(f">>> Preparing Group: {group_label}")

        # --- Submission ---
        if args.dryrun:
            print(f"    [Dry Run] Generated JDL: {jdl_file}")
            print(f"    [Dry Run] Generated Bash: {bash_path}")
            print(f"    [Dry Run] Generated Input text: {master_list_path}\n")
        else:
            # Standard Submission
            print(f"    Submitting {jdl_file}...")
            result = subprocess.run(['condor_submit', str(jdl_file)])
            if result.returncode != 0:
                print(f"    !!! ERROR: condor_submit failed for {group_label} with exit code {result.returncode}.")
                failures += 1

    if failures:
        print(f"\n{failures}/{len(track_dirs)} group(s) FAILED to submit.")
        sys.exit(1)

    print("\nSubmission process complete.")