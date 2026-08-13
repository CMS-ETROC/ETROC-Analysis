import argparse
import getpass
import logging
import re
import subprocess
import sys
import time
import uuid
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import combinations

from pathlib import Path
from jinja2 import Template
from natsort import natsorted
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'core'))
import io_utils
import extract_events_by_path

# Mirrors path_finder.py's own constant -- combo indices below are only ever
# meaningful if both scripts agree on how board combos are generated.
MIN_BOARD_COMBO_SIZE = 3

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
{% if concurrency_limit -%}
JobBatchName          = "etroc_extract_events"
max_materialize        = {{ concurrency_limit }}
{% endif -%}
Queue idx,fname from {{ script_dir }}/input_list.txt
"""

CONCURRENCY_TAG = 'etroc_extract_events'

# --- Helper Functions ---

def find_track_files(track_arg: Path, combos: list[str] = None) -> list[Path]:
    """Resolves -t to the list of track files to submit jobs for.

    If track_arg is a file, it's a single combo's track file and is returned
    as the sole entry -- unchanged from before (combos is ignored in this
    case, since there's nothing to filter). If it's a directory (the per-run
    directory path_finder.py writes combo track files into, and
    select_tracks_by_coverage.py writes '_reduced' files alongside), every
    '*_reduced.parquet' file in it is auto-detected -- one submission per
    combo. All of them share the same --cal_table, since CAL mode values are
    computed once per run (not per combo) by path_finder.py.

    combos, when given, restricts that auto-detection to just the named
    board-combo labels (e.g. ['dut1-ref2-trig3', 'extra0-dut1-ref2']) instead
    of every combo found -- for processing a subset without having to move
    the other combos' files out of the directory first.
    """
    if track_arg.is_file():
        return [track_arg]
    if track_arg.is_dir():
        track_files = natsorted(track_arg.glob('*_reduced.parquet'))
        if not track_files:
            sys.exit(f"Error: No '*_reduced.parquet' files found in {track_arg}.")
        if combos:
            by_label = {io_utils.combo_label_from_track_filename(f): f for f in track_files}
            missing = [c for c in combos if c not in by_label]
            if missing:
                sys.exit(
                    f"Error: --combos requested label(s) not found in {track_arg}: {missing}. "
                    f"Available: {sorted(by_label)}"
                )
            track_files = [by_label[c] for c in combos]
        return track_files
    sys.exit(f"Error: Track path '{track_arg}' not found.")

def board_ids_from_combo_label(label: str) -> tuple[int, ...]:
    """'dut1-ref2-trig3' -> (1, 2, 3) -- parses the trailing board id off each
    role-tagged token in a combo label (see io_utils.combo_label_from_track_filename)."""
    return tuple(sorted(int(re.search(r'\d+$', tok).group()) for tok in label.split('-')))

def compute_expected_combos(config_path: str, run_name: str) -> list[tuple[int, ...]]:
    """Reproduces path_finder.py's own board-combo generation (same board ids,
    same combo sizes, same order: combinations() over every subset down to
    MIN_BOARD_COMBO_SIZE, largest first) so a --combos index is derived purely
    from the run's board config. That keeps index 0 always meaning the same
    board combo across every invocation of this run -- unlike an index into
    "whichever files happen to exist in the directory today", which would
    shift if a combo hasn't been reduced yet or was deleted."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if run_name not in config:
        sys.exit(f"Error: Run config '{run_name}' not found in {config_path}")
    ids_to_process = sorted(int(b) for b in config[run_name].keys())

    min_boards = min(MIN_BOARD_COMBO_SIZE, len(ids_to_process))
    max_boards = len(ids_to_process) - 1 if len(ids_to_process) > min_boards else len(ids_to_process)
    return [
        combo
        for size in range(max_boards, min_boards - 1, -1)
        for combo in combinations(ids_to_process, size)
    ]

def print_combo_legend(track_files: list[Path], expected_combos: list[tuple[int, ...]]) -> None:
    by_board_ids = {board_ids_from_combo_label(io_utils.combo_label_from_track_filename(f)): f for f in track_files}
    print('Board combos for this run (pass to --combos by index or by label):')
    for idx, board_ids in enumerate(expected_combos):
        tag = io_utils.combo_label_from_track_filename(by_board_ids[board_ids]) if board_ids in by_board_ids else '(not reduced yet)'
        print(f'  {idx}: {"-".join(map(str, board_ids))}  ({tag})')

def resolve_combo_tokens(tokens: list[str], expected_combos: list[tuple[int, ...]], track_files: list[Path]) -> list[str]:
    """Resolves --combos tokens to combo labels for find_track_files(). Each
    token is either a plain integer -- an index into expected_combos -- or a
    literal combo label, matched as-is (unchanged from before this existed)."""
    by_board_ids = {board_ids_from_combo_label(io_utils.combo_label_from_track_filename(f)): f for f in track_files}

    labels = []
    for token in tokens:
        if not token.isdigit():
            labels.append(token)
            continue
        idx = int(token)
        if idx < 0 or idx >= len(expected_combos):
            sys.exit(f"Error: combo index {idx} out of range 0-{len(expected_combos) - 1}.")
        board_ids = expected_combos[idx]
        if board_ids not in by_board_ids:
            sys.exit(
                f"Error: combo index {idx} (boards {'-'.join(map(str, board_ids))}) has no matching "
                f"'*_reduced.parquet' file in the input directory yet."
            )
        labels.append(io_utils.combo_label_from_track_filename(by_board_ids[board_ids]))
    return labels

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


def _init_local_worker():
    """ProcessPoolExecutor(initializer=...): runs once per worker process at
    startup.

    1. Caps pyarrow's own internal threading -- without this, each of the 3
       worker processes would let its own parquet/feather reads spin up
       pyarrow's default multi-threaded pool (one per core), oversubscribing
       well past the cap. Same fix reshape_event_to_track.py already applies
       to its own worker processes.
    2. Quiets extract_events_by_path.py's own per-file INFO logging (it calls
       logging.basicConfig(level=INFO) at import time) -- useful for a single
       condor job's own captured stdout, but with 3 workers running
       concurrently here it's an interleaved flood that buries the shared
       tqdm progress bar. WARNING+ (actual problems) still comes through per
       file.
    """
    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    logging.getLogger().setLevel(logging.WARNING)


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
    # Local runs share a real interactive node (not a dedicated condor slot),
    # so a few cores is fine but grabbing all of them isn't -- same reasoning
    # as reshape_event_to_track.py's own worker pool.
    with ProcessPoolExecutor(max_workers=3, initializer=_init_local_worker) as executor:
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


def wait_for_condor_capacity(tag: str, limit: int, poll_interval: int = 60) -> None:
    """Blocks until this user has fewer than `limit` jobs queued (idle +
    running) under JobBatchName `tag`. Each cluster caps its own concurrency
    via max_materialize, but that only bounds a single condor_submit call --
    this additionally serializes separate calls (different combos, or
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
        concurrency_limit=args.concurrency_limit,
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
                             'the same --cal_table. Use --combos to process only some of them.')
    parser.add_argument('--combos', dest='combos', default=None,
                        help='Comma-separated combos to process when -t points at a directory, instead of '
                             'every "*_reduced.parquet" file found -- restricts auto-detection to just these. '
                             'Each entry is either a combo label (e.g. "dut1-ref2-trig3") or a plain integer '
                             'index into the board-config-derived combo ordering printed at the top of every '
                             'run (index 0 always means the same board combo for a given -c/-r, regardless of '
                             'which combo files currently exist in the directory). Ignored when -t points '
                             'directly at a single file.')
    parser.add_argument('-c', '--config', required=True, dest='config', help='YAML file with run config')
    parser.add_argument('-r', '--runName', required=True, dest='runName', help='Run name in YAML config')
    parser.add_argument('--cal_table', required=True, dest='cal_table', help='CSV file with CAL mode values')
    parser.add_argument('-o', '--outdir', default='extractEvents_outputs', dest='outname', help='Output directory on EOS')
    parser.add_argument('--neighbor_search_method', default="none", dest='search_method',
                        help="Search method for neighbor hit checking, default is 'none'. possible argument: 'row_only', 'col_only', 'cross', 'square'")
    parser.add_argument('--condor_tag', dest='condor_tag', help='Tag appended to filenames to avoid collisions')
    parser.add_argument('--concurrency_limit', type=int, default=None,
                        help='Cap on concurrently queued (idle+running) jobs, shared across every '
                             'submission of this script: caps each cluster with max_materialize, and '
                             'self-throttles (polls condor_q, blocking) before submitting the next combo/run '
                             'until earlier ones have dropped under the cap. Unset by default -- a single '
                             'run submission is not throttled. Set this when you are about to submit several '
                             'runs around the same time and want to bound their combined memory footprint '
                             'on the condor pool (e.g. 30).')
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
    all_track_files = find_track_files(track_arg)

    combos = None
    if track_arg.is_dir():
        expected_combos = compute_expected_combos(args.config, args.runName)
        print_combo_legend(all_track_files, expected_combos)
        if args.combos:
            tokens = [c.strip() for c in args.combos.split(',')]
            combos = resolve_combo_tokens(tokens, expected_combos, all_track_files)

    track_files = find_track_files(track_arg, combos)

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
            if args.concurrency_limit:
                wait_for_condor_capacity(CONCURRENCY_TAG, args.concurrency_limit)
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
