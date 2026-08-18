import argparse
import subprocess
import sys
import re, getpass
import time
import uuid
from datetime import datetime
from pathlib import Path
from jinja2 import Template
from natsort import natsorted

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'core'))
import io_utils

WORKER_SCRIPT_NAME = "bootstrap.py"

BASH_TEMPLATE = """#!/bin/bash
""" + io_utils.BASH_STRICT_HEADER + """
# $1: The full EOS path of the input file (e.g., /path/to/time/data_01.parquet)
INPUT_FILE_EOS="$1"

# Load python environment from work node
# LCG's setup.sh references its own internal vars (e.g. COMPILER) without
# defaults, which trips our `set -u` above even though it's not a real error.
set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_104a/x86_64-el9-gcc13-opt/setup.sh
set -u

# 1. Determine local file name
LOCAL_FILENAME=$(basename "$INPUT_FILE_EOS")

# Cleanup the local copy on any exit -- success, a `set -e` abort from a
# failing command below, or bootstrap.py's own sys.exit (e.g. too few events
# to run bootstrap on). A plain step-4-at-the-end `rm` never ran in the
# failure cases, since `set -e` jumps straight past it to script exit.
trap 'rm -f "$LOCAL_FILENAME"' EXIT

# 2. Copy input data from EOS to local work node
echo "Transferring: $INPUT_FILE_EOS"
xrdcp root://eosuser.cern.ch/$INPUT_FILE_EOS ./$LOCAL_FILENAME

# 3. Construct and Run Bootstrap Analysis
echo "\nRunning: {{ command }} -f $LOCAL_FILENAME"
{{ command }} -f $LOCAL_FILENAME

echo "\n--- Job finished successfully ---"
"""

JDL_TEMPLATE = """universe              = vanilla
executable            = {{ script_dir }}/{{ bash_script_name }}
should_Transfer_Files = YES
whenToTransferOutput  = ON_EXIT
transfer_Input_Files  = {{ transfer_files }}
Arguments             = {{ logical_dir }}/$(stem){{ ext }}
TransferOutputRemaps  = "$(stem)_boot.parquet={{ out_dir }}/$(stem)_boot.parquet"
output                = {{ log_dir }}/$(ClusterId).$(ProcId).bootstrap.stdout
error                 = {{ log_dir }}/$(ClusterId).$(ProcId).bootstrap.stderr
log                   = {{ log_dir }}/bootstrap.log
MY.WantOS             = "el9"
+JobFlavour           = "workday"
{% if concurrency_limit -%}
JobBatchName          = "etroc_bootstrap"
max_materialize        = {{ concurrency_limit }}
{% endif -%}
Queue stem from {{ script_dir }}/{{ master_list_file_name }}
"""

CONCURRENCY_TAG = 'etroc_bootstrap'

def build_python_command(args: argparse.Namespace) -> str:
    """
    Constructs the python command string dynamically using a placeholder for the filename.
    The bash script will substitute 'FILENAME_PLACEHOLDER' with the local filename.
    """
    neighbor_cut_str = " ".join(args.neighbor_cut)

    cmd_parts = [
        f"python {WORKER_SCRIPT_NAME}",
        f"-n {args.num_bootstrap_output}",
        f"--ks_pmin {args.ks_pmin}",
        f"--ks_pmin_floor {args.ks_pmin_floor}",
        f"--ks_dmax {args.ks_dmax}",
        f"--gmm_tol {args.gmm_tol}",
        f"--gmm_max_iter {args.gmm_max_iter}",
        f"--minimum_nevt {args.minimum_nevt}",
        f"--iteration_limit {args.iteration_limit}",
        f"--neighbor_cut {neighbor_cut_str}",
        f"--neighbor_logic {args.neighbor_logic}",
    ]

    if args.reproducible: cmd_parts.append("--reproducible")

    return " ".join(cmd_parts)

def wait_for_condor_capacity(tag: str, limit: int, poll_interval: int = 60) -> None:
    """Blocks until this user has fewer than `limit` jobs queued (idle +
    running) under JobBatchName `tag`. Each cluster caps its own concurrency
    via max_materialize, but that only bounds a single condor_submit call --
    this additionally serializes separate calls (different groups, or
    separate runs of this script) sharing the same tag, so their per-cluster
    caps don't just stack on top of each other on the pool.
    """
    username = getpass.getuser()
    while True:
        result = subprocess.run(
            ['condor_q', username, '-constraint', f'JobBatchName=="{tag}"', '-af', 'ClusterId'],
            capture_output=True, text=True,
        )
        queued = len([line for line in result.stdout.splitlines() if line.strip()])
        if queued < limit:
            return
        print(f"    Waiting for condor capacity: {queued} job(s) already queued under '{tag}' "
              f"(limit {limit}). Rechecking in {poll_interval}s...")
        time.sleep(poll_interval)


def create_submission_files(
    args: argparse.Namespace,
    paths: dict[str, Path],
    unique_tag: str,
    input_dir: Path,
    group_out_dir: Path,
    group_log_dir: Path
):
    """
    Generates the input list (stem), bash script, and JDL file.
    """

    master_list_file_name = f'input_list{unique_tag}.txt'
    input_list_path = paths['scripts'] / master_list_file_name
    if input_list_path.exists():
        input_list_path.unlink()

    parquet_files = list(input_dir.glob('*.parquet'))

    files: list[Path] = []
    ext = "" # Store the extension for the JDL

    if parquet_files:
        print(f"    Found {len(parquet_files)} Parquet files. Proceeding with Parquet.")
        files = parquet_files
        ext = ".parquet"

    files = natsorted(files)
    if not files:
        print(f"Warning: No parquet files found in {input_dir.name}. Skipping group.")
        return None, None, None

    # --- PATH FIX: Calculate the base directory ONLY ONCE ---
    abs_dir = str(input_dir.resolve())
    logical_dir = re.sub(r'^/eos/home-([a-z0-9])/', r'/eos/user/\1/', abs_dir)

    # Write ONLY the stem_name
    with open(input_list_path, 'a') as f:
        for file_path in files:
            f.write(f"{file_path.stem}\n")

    # Generate Bash Script
    bash_script_name = f'run_bootstrap{unique_tag}.sh'
    command = build_python_command(args)

    bash_content = Template(BASH_TEMPLATE).render({
        'command': command
    })

    bash_path = paths['scripts'] / bash_script_name
    with open(bash_path, 'w') as f:
        f.write(bash_content)

    # Generate JDL File
    jdl_content = Template(JDL_TEMPLATE).render({
        'script_dir': str(paths['scripts']),
        'bash_script_name': bash_script_name,
        'master_list_file_name': master_list_file_name,
        'out_dir': str(group_out_dir),
        'log_dir': str(group_log_dir),
        'transfer_files': io_utils.build_transfer_files(WORKER_SCRIPT_NAME),
        'unique_tag': unique_tag,
        'logical_dir': logical_dir, # Pass the directory to Jinja
        'ext': ext,                 # Pass the extension to Jinja
        'concurrency_limit': args.concurrency_limit
    })

    jdl_path = paths['scripts'] / f'condor_bootstrap{unique_tag}.jdl'
    with open(jdl_path, 'w') as f:
        f.write(jdl_content)

    return jdl_path, bash_path, input_list_path

def process_group_dir(group_dir: Path, combo_label, args: argparse.Namespace, paths: dict[str, Path]) -> bool:
    """Submits (or dry-runs) the condor job for one time/time_groupX directory.
    Returns True on success (including a skipped/empty group), False on failure."""
    group_name = group_dir.name

    # Calculate Suffix: time -> "", time_group1 -> _group1
    group_suffix = group_name.replace('time', '')

    # Namespace output/log dirs and local artifact names (script/JDL/input list)
    # by combo so two combos with the same group_suffix (e.g. both a bare "time"
    # dir) never collide. Unlike step 11's CSVs, a collision here would silently
    # overwrite bootstrap .parquet output between combos via TransferOutputRemaps
    # (stem-based filenames can plausibly match across combos), not just mix up
    # human-readable filenames.
    if combo_label:
        unique_out_name = f"bootstrap_{args.outputdir}/{combo_label}{group_suffix}"
        group_log_dir = paths['logs'] / combo_label / group_name
        unique_group_tag = f'_{combo_label}{group_suffix}'
    else:
        unique_out_name = f"bootstrap_{args.outputdir}{group_suffix}"
        group_log_dir = paths['logs'] / group_name
        unique_group_tag = group_suffix

    group_out_dir = Path(unique_out_name)
    label = f'{combo_label}/{group_name}' if combo_label else group_name

    if not args.dryrun:
        try:
            group_out_dir.mkdir(parents=True, exist_ok=True)
            group_log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Error creating directories for {label}: {e}")
            return False

    print(f"\n>>> Processing Group: {label}")
    print(f"    Out: {group_out_dir}")
    print(f"    Log: {group_log_dir}")

    jdl, bash, list_file = create_submission_files(
        args, paths, unique_group_tag, group_dir, group_out_dir, group_log_dir
    )

    if not jdl:
        return True

    if args.dryrun:
        print(f"    [Dry Run] JDL: {jdl}")
        print(f"    [Dry Run] Bash: {bash}")
        print(f"    [Dry Run] Input: {list_file}")
        return True

    if list_file.stat().st_size > 0:
        if args.concurrency_limit:
            wait_for_condor_capacity(CONCURRENCY_TAG, args.concurrency_limit)
        print(f"    Submitting jobs...")
        result = subprocess.run(['condor_submit', str(jdl)])
        if result.returncode != 0:
            print(f"    !!! ERROR: condor_submit failed for {label} with exit code {result.returncode}.")
            return False
    else:
        print("    Input list empty.")

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Submit Bootstrap Analysis to Condor')

    # Required
    parser.add_argument('-d', '--inputdir', required=True, dest='dirname', help='Mother directory containing time/time_group folders. '
                        'Can also be the combo mother directory (one subdirectory per board combo, each with its own time/time_groupX '
                        'folders) -- every combo is then auto-detected and processed in one submission.')
    parser.add_argument('-o', '--outputdir', required=True, dest='outputdir', help='Output directory base name')

    # Bootstrap Params
    parser.add_argument('-n', '--num_bootstrap_output', type=int, default=100)
    parser.add_argument('--minimum_nevt', type=int, default=1000)
    parser.add_argument('--iteration_limit', type=int, default=7500)

    # Options
    parser.add_argument('--condor_tag', dest='condor_tag', help='Tag for filenames')
    parser.add_argument('--concurrency_limit', type=int, default=None,
                        help='Cap on concurrently queued (idle+running) jobs, shared across every '
                             'submission of this script: caps each cluster with max_materialize, and '
                             'self-throttles (polls condor_q, blocking) before submitting the next group/run '
                             'until earlier ones have dropped under the cap. Unset by default -- a single '
                             'run submission is not throttled. Set this when you are about to submit several '
                             'runs around the same time and want to bound their combined memory footprint '
                             'on the condor pool (e.g. 30).')
    parser.add_argument('--reproducible', action='store_true')
    parser.add_argument('--ks_pmin', type=float, default=1e-3,
                        help='bootstrap.py --ks_pmin: a pair mixture fit is accepted if KS p-value >= this OR KS distance <= --ks_dmax (default 1e-3).')
    parser.add_argument('--ks_pmin_floor', type=float, default=1e-6, help='bootstrap.py --ks_pmin_floor (default 1e-6).')
    parser.add_argument('--ks_dmax', type=float, default=0.03, help='bootstrap.py --ks_dmax (default 0.03).')
    parser.add_argument('--gmm_tol', type=float, default=1e-6, help='bootstrap.py --gmm_tol: EM convergence tolerance of the mixture fit (default 1e-6).')
    parser.add_argument('--gmm_max_iter', type=int, default=2000, help='bootstrap.py --gmm_max_iter (default 2000).')
    parser.add_argument('--neighbor_cut', dest='neighbor_cut', default=['none'], nargs='+',
                        help='Specify one or more **space-separated** board columns to be used for neighbor cuts. '
                        'The argument collects all values into a list. '
                        'Possible columns: HasNeighbor_dut, HasNeighbor_ref, HasNeighbor_extra, HasNeighbor_trig, trackNeighbor. '
                        'Default value is a list containing only "none".')
    parser.add_argument('--neighbor_logic', dest='neighbor_logic', default='OR',
                        help='Logic for multiple neighbor cuts on board. Default is OR. AND is possble.')

    # Execution Modes
    parser.add_argument('--dryrun', action='store_true')

    args = parser.parse_args()
    if args.ks_pmin_floor > args.ks_pmin:
        parser.error('--ks_pmin_floor must not exceed --ks_pmin')

    # --- 1. Identify Groups ---
    username = getpass.getuser()
    eos_base_dir = io_utils.eos_base_dir(username)
    mother_dir = Path(f'{eos_base_dir}/{args.dirname}').resolve()

    time_dirs = io_utils.discover_time_dirs(mother_dir)

    # --- 2. Setup Base Paths ---
    if args.condor_tag:
        run_append = args.condor_tag
    else:
        # Auto-generate a unique tag rather than falling back to a shared bucket name -
        # otherwise a second untagged submission can overwrite this run's scripts/logs
        # while an earlier untagged submission is still queued and hasn't been dispatched yet.
        run_append = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        print(f"No --condor_tag given; auto-generated tag '{run_append}' to avoid collisions with other submissions.")

    # Only script paths are global; Output/Log paths are calculated per group
    # Note: Using relative paths here to keep JDL portable
    paths = {
        'scripts': Path('.') / 'condor_scripts' / 'bootstrap' / f'{run_append}',
        'logs':    Path('.') / 'condor_logs' / 'bootstrap' / f'{run_append}',
    }

    # Create Base Directories
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    if not Path(f'core/{WORKER_SCRIPT_NAME}').is_file():
        sys.exit(f"Error: Worker script '{WORKER_SCRIPT_NAME}' not found.")

    # --- 3. Process Each Group ---
    failures = 0
    total_groups = 0

    if time_dirs:
        print(f"Found {len(time_dirs)} groups: {[d.name for d in time_dirs]}")
        for group_dir in time_dirs:
            total_groups += 1
            if not process_group_dir(group_dir, None, args, paths):
                failures += 1
    else:
        # -d may be the combo mother directory step 10 can write (one
        # subdirectory per board combo, each itself containing
        # time/time_groupX) -- descend one level and process every combo
        # instead of requiring one submission per combo.
        processed_any = False
        for combo_dir in sorted(d for d in mother_dir.iterdir() if d.is_dir()):
            combo_time_dirs = io_utils.discover_time_dirs(combo_dir)
            if not combo_time_dirs:
                continue
            processed_any = True
            print(f"Found {len(combo_time_dirs)} groups for combo {combo_dir.name}: "
                  f"{[d.name for d in combo_time_dirs]}")
            for group_dir in combo_time_dirs:
                total_groups += 1
                if not process_group_dir(group_dir, combo_dir.name, args, paths):
                    failures += 1

        if not processed_any:
            sys.exit(f"No directories containing 'time' found in {mother_dir} or its subdirectories")

    if failures:
        print(f"\n{failures}/{total_groups} group(s) FAILED to submit.")
        sys.exit(1)

    print("\nDone.")