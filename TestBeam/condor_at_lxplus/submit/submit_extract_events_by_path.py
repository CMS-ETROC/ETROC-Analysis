import argparse
import getpass
import subprocess
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from pathlib import Path
from jinja2 import Template
from natsort import natsorted
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'core'))
import io_utils
import extract_events_by_path

# --- Configuration & Templates ---

# Template for the shell script running on the worker node
BASH_TEMPLATE = """#!/bin/bash
""" + io_utils.BASH_STRICT_HEADER + """
ls -ltrh
echo ""
pwd

file_index=$1
input_file=$2
path_to_copy=$3

# Load python environment from work node
# LCG's setup.sh references its own internal vars (e.g. COMPILER) without
# defaults, which trips our `set -u` above even though it's not a real error.
set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_104a/x86_64-el9-gcc13-opt/setup.sh
set -u

# Copy input data from EOS to local work node
xrdcp -r root://eosuser.cern.ch/$path_to_copy ./

echo "Will process input file from {{ runname }} $input_file (index $file_index)"

# Run the python script
echo "python extract_events_by_path.py -f $input_file -r {{ runname }} -t {{ track }} -c {{ config }} --cal_table {{ cal_table }} --neighbor_search_method {{ search_method }} --file-index $file_index"
python extract_events_by_path.py -f $input_file -r {{ runname }} -t {{ track }} -c {{ config }} --cal_table {{ cal_table }} --neighbor_search_method {{ search_method }} --file-index $file_index

ls -ltrh
echo ""

# Delete input file so condor will not return it as output
rm $input_file

ls -ltrh
echo ""
"""

# Template for the Condor JDL file
JDL_TEMPLATE = """universe              = vanilla
executable            = {{ script_dir }}/run_extract_events.sh
should_Transfer_Files = YES
whenToTransferOutput  = ON_EXIT
# $1 is the file index, $2 is the clean filename, $3 is the clean path
arguments             = $(idx) $(fname) {{ input_dir }}/$(fname)
transfer_Input_Files  = {{ transfer_files }}
output                = {{ log_dir }}/$(ClusterId).$(ProcId).extractEvents.stdout
error                 = {{ log_dir }}/$(ClusterId).$(ProcId).extractEvents.stderr
log                   = {{ log_dir }}/extractEvents.log
MY.WantOS             = "el9"
MY.XRDCP_CREATE_DIR   = True
output_destination    = root://eosuser.cern.ch/{{ eos_base }}/{{ out_dir }}
+JobFlavour           = "microcentury"
Queue idx,fname from {{ script_dir }}/input_list.txt
"""

# --- Helper Functions ---

def find_track_files(track_arg: Path) -> list[Path]:
    """Resolves -t to the list of track files to submit jobs for.

    If track_arg is a file, it's a single combo's track file and is returned
    as the sole entry -- unchanged from before. If it's a directory (the
    per-run directory path_finder.py writes combo track files into, and
    select_tracks_by_coverage.py writes '_reduced' files alongside),
    every '*_reduced.parquet' file in it is auto-detected -- one submission
    per combo. All of them share the same --cal_table, since CAL mode values
    are computed once per run (not per combo) by path_finder.py.
    """
    if track_arg.is_file():
        return [track_arg]
    if track_arg.is_dir():
        track_files = natsorted(track_arg.glob('*_reduced.parquet'))
        if not track_files:
            sys.exit(f"Error: No '*_reduced.parquet' files found in {track_arg}.")
        return track_files
    sys.exit(f"Error: Track path '{track_arg}' not found.")

def build_indexed_file_list(final_input_dir: Path) -> list[tuple[int, Path]]:
    """(file_index, file_path) pairs for every loop*.feather file in
    final_input_dir, parsed the same way both condor's input_list.txt and
    local in-process runs need it -- matches extract_events_by_path.py's
    --file-index semantics either way."""
    indexed = []
    for file_path in natsorted(final_input_dir.glob('loop*feather')):
        try:
            file_idx = int(file_path.stem.split('_')[1])
        except (IndexError, ValueError):
            print(f"Warning: Could not parse index from {file_path.name}, skipping.")
            continue
        indexed.append((file_idx, file_path))
    return indexed


# Local runs share a real interactive node (not a dedicated condor slot), so a
# few cores is fine but grabbing all of them isn't -- same reasoning as
# reshape_event_to_track.py's own worker pool.


def _limit_pyarrow_threads():
    """ProcessPoolExecutor(initializer=...): runs once per worker process at
    startup. Without this, each of the _LOCAL_MAX_WORKERS processes would let
    its own parquet/feather reads spin up pyarrow's default multi-threaded
    pool (one per core), oversubscribing well past the cap. Same fix
    reshape_event_to_track.py already applies to its own worker processes."""
    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


def run_local(args, track_path: Path, eos_base: str, out_dir: str) -> int:
    """Runs extract_events_by_path.py's logic for every input file on a small
    local process pool, instead of submitting one condor job per file. No
    xrdcp transfer needed -- /eos is already mounted on lxplus interactive
    nodes -- and output is written directly to its final destination instead
    of relying on condor's output_destination copy-back. Returns the number
    of files that failed."""
    final_input_dir = Path(eos_base) / args.dirname
    indexed_files = build_indexed_file_list(final_input_dir)
    if not indexed_files:
        print(f"    No input files found in {final_input_dir}. Nothing to process.")
        return 0

    final_output_dir = str(Path(eos_base) / out_dir)

    def make_args(file_idx, file_path):
        return argparse.Namespace(
            inputfile=str(file_path),
            runinfo=args.runName,
            config=args.config,
            track=str(track_path),
            search_method=args.search_method,
            cal_table=args.cal_table,
            file_index=file_idx,
            outdir=final_output_dir,
        )

    failures = 0
    with ProcessPoolExecutor(max_workers=3, initializer=_limit_pyarrow_threads) as executor:
        futures = {
            executor.submit(extract_events_by_path.run, make_args(file_idx, file_path)): file_path
            for file_idx, file_path in indexed_files
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Local run"):
            file_path = futures[future]
            try:
                future.result()
            except SystemExit as e:
                if e.code not in (0, None):
                    print(f"    !!! ERROR processing {file_path.name}: exited with code {e.code}")
                    failures += 1
            except Exception as e:
                print(f"    !!! ERROR processing {file_path.name}: {e}")
                failures += 1

    return failures


def create_submission_files(args, script_dir, log_dir, eos_base, track_path, out_dir):

    config_path = Path(args.config)
    cal_path = Path(args.cal_table)
    final_input_dir = Path(eos_base) / args.dirname

    # 2. No unlink() needed
    input_list_path = script_dir / 'input_list.txt'
    indexed_files = build_indexed_file_list(final_input_dir)

    with open(input_list_path, 'w') as f:
        for file_idx, file_path in indexed_files:
            f.write(f"{file_idx},{file_path.name}\n")

    # 3. Pass Path objects directly to Template
    bash_content = Template(BASH_TEMPLATE).render(
        runname=args.runName,
        track=track_path.name,
        cal_table=cal_path.name,
        search_method=args.search_method,
        config=config_path.name,
    )

    bash_script_path = script_dir / f'run_extract_events.sh'
    with open(bash_script_path, 'w') as f:
        f.write(bash_content)

    # 3. Generate JDL File
    transfer_files = io_utils.build_transfer_files(
        'extract_events_by_path.py', track_path, cal_path, config_path
    )
    jdl_content = Template(JDL_TEMPLATE).render(
        script_dir=script_dir,
        input_dir=final_input_dir,
        transfer_files=transfer_files,
        log_dir=log_dir,
        eos_base=eos_base,
        out_dir=out_dir,
    )

    jdl_path = script_dir / f'condor_extract_events.jdl'
    with open(jdl_path, 'w') as f:
        f.write(jdl_content)

    return jdl_path, bash_script_path, input_list_path

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Submit Extract Events',
        description='Submit Condor jobs to extract event data for tracks.'
    )

    parser.add_argument('-d', '--inputdir', required=True, dest='dirname', help='Input directory containing feather files')
    parser.add_argument('-t', '--track', required=True, dest='track',
                        help='Parquet file with track candidates for one board combo. Can also be the '
                             'per-run directory path_finder.py / select_tracks_by_coverage.py '
                             'wrote them into -- every "*_reduced.parquet" file in it is then '
                             'auto-detected and submitted as a separate job, one per combo, all sharing '
                             'the same --cal_table.')
    parser.add_argument('-c', '--config', required=True, dest='config', help='YAML file with run config')
    parser.add_argument('-r', '--runName', required=True, dest='runName', help='Run name in YAML config')
    parser.add_argument('--cal_table', required=True, dest='cal_table', help='CSV file with CAL mode values')
    parser.add_argument('-o', '--outdir', default='extractEvents_outputs', dest='outname', help='Output directory on EOS')
    parser.add_argument('--neighbor_search_method', default="none", dest='search_method',
                        help="Search method for neighbor hit checking, default is 'none'. possible argument: 'row_only', 'col_only', 'cross', 'square'")
    parser.add_argument('--condor_tag', dest='condor_tag', help='Tag appended to filenames to avoid collisions')
    parser.add_argument('--dryrun', action='store_true', help='Generate files but do not submit')
    parser.add_argument('--local', action='store_true',
                        help='Run in-process on this machine instead of submitting to condor -- no JDL/bash '
                             'files, no xrdcp transfer (/eos is already mounted on lxplus interactive nodes), '
                             'output written directly to its final destination. Only worth it when '
                             'n_files x per_file_time is short enough to just wait out serially (a handful of '
                             'minutes); condor still wins once that stretches to tens of minutes or hours, or '
                             'you want it running unattended. Ignored together with --dryrun -- --local runs '
                             'immediately and does not generate condor submission files at all.')

    args = parser.parse_args()

    # --- Setup Environments ---
    username = getpass.getuser()
    eos_base_dir = str(io_utils.eos_base_dir(username))

    if args.condor_tag:
        run_append = args.condor_tag
    else:
        # Auto-generate a unique tag rather than falling back to a shared bucket name -
        # otherwise a second untagged submission can overwrite run_extract_events.sh/input_list.txt
        # while an earlier untagged submission is still queued and hasn't been dispatched yet.
        run_append = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        print(f"No --condor_tag given; auto-generated tag '{run_append}' to avoid collisions with other submissions.")

    # Directory setup
    base_scripts_dir = Path('.') / 'condor_scripts' / 'extract_events' / f'{run_append}'
    base_log_dir = Path('.') / 'condor_logs' / 'extract_events' / f'{run_append}'
    base_scripts_dir.mkdir(parents=True, exist_ok=True)
    base_log_dir.mkdir(parents=True, exist_ok=True)

    # --- Validation ---
    if not Path('core/extract_events_by_path.py').is_file():
        sys.exit(f"Error: Worker script extract_events_by_path.py not found in current directory.")
    track_arg = Path(args.track)
    if not track_arg.exists():
        sys.exit(f"Error: Track path '{args.track}' not found.")
    if not Path(args.cal_table).is_file():
        sys.exit(f"Error: Cal table '{args.cal_table}' not found.")
    if not Path(args.config).is_file():
        sys.exit(f"Error: Config file '{args.config}' not found.")

    # --- Logic ---
    track_files = find_track_files(track_arg)

    print('\n========= Submission Details =========')
    print(f'Input:       {args.dirname}')
    print(f'Input CAL table: {args.cal_table}')
    if args.search_method != 'none':
        print(f'Neighbor search method: {args.search_method}')
    if len(track_files) > 1:
        print(f'Found {len(track_files)} combo track file(s) in {track_arg}: {[f.name for f in track_files]}')
    print('======================================\n')

    base_outname = args.outname
    failures = 0

    for track_path in track_files:
        # Auto-namespace the output directory by board combo so two combos
        # submitted with the same -o can never collide on EOS or get merged
        # together by step 9's track_id-based gather (track_id restarts at 0
        # independently per combo, so mixing combos in one directory silently
        # corrupts the merge). Same reasoning for nesting each combo's own
        # script/log directory, so their input_list.txt/JDL/bash files (and
        # condor logs) don't overwrite each other either.
        try:
            combo_label = io_utils.combo_label_from_track_filename(track_path)
        except ValueError as e:
            sys.exit(f"Error: {e}")

        out_dir = str(Path(base_outname) / combo_label)

        print(f'>>> Combo: {combo_label}')
        print(f'    Track file: {track_path}')
        print(f'    Output:     {eos_base_dir}/{out_dir}')

        if args.local:
            local_failures = run_local(args, track_path, eos_base_dir, out_dir)
            if local_failures:
                print(f"    !!! {local_failures} file(s) FAILED for {combo_label}.")
                failures += local_failures
            print()
            continue

        combo_script_dir = base_scripts_dir / combo_label
        combo_log_dir = base_log_dir / combo_label
        combo_script_dir.mkdir(parents=True, exist_ok=True)
        combo_log_dir.mkdir(parents=True, exist_ok=True)

        jdl_file, bash_file, list_file = create_submission_files(
            args, combo_script_dir, combo_log_dir, eos_base_dir, track_path, out_dir
        )

        if args.dryrun:
            print(f"    [Dry Run] JDL:  {jdl_file}")
            print(f"    [Dry Run] Bash: {bash_file}")
            print(f"    [Dry Run] List: {list_file}\n")
            continue

        if list_file.stat().st_size > 0:
            result = subprocess.run(['condor_submit', str(jdl_file)])
            if result.returncode != 0:
                print(f"    !!! ERROR: condor_submit failed for {combo_label} with exit code {result.returncode}.")
                failures += 1
        else:
            print(f"    No input files found in directory. Nothing submitted for {combo_label}.")
        print()

    if failures:
        verb = 'process' if args.local else 'submit'
        print(f"{failures}/{len(track_files)} combo(s) FAILED to {verb}.")
        sys.exit(1)
