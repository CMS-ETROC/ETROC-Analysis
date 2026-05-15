import argparse
import getpass
import subprocess
import sys, yaml

from pathlib import Path
from jinja2 import Template
from natsort import natsorted

# --- Configuration & Templates ---

# Template for the shell script running on the worker node
BASH_TEMPLATE = """#!/bin/bash

ls -ltrh
echo ""
pwd

input_file=$1

# Load python environment from work node
source /cvmfs/sft.cern.ch/lcg/views/LCG_104a/x86_64-el9-gcc13-opt/setup.sh

# Copy input data from EOS to local work node
xrdcp -r root://eosuser.cern.ch/{{ path }} ./

echo "Will process input file from {{ runname }} $input_file"

# Run the python script
echo "python extract_events_by_path.py -f $input_file -r {{ runname }} -t {{ track }} -c {{ config }} --trigID {{ trigID }} --cal_table {{ cal_table }} --neighbor_search_method {{ search_method }}"
python extract_events_by_path.py -f $input_file -r {{ runname }} -t {{ track }} -c {{ config }} --trigID {{ trigID }} --cal_table {{ cal_table }} --neighbor_search_method {{ search_method }}

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
# $1 is the clean filename, $2 is the clean path
arguments             = $(fname) {{ input_dir }}/$(fname)
transfer_Input_Files  = core/extract_events_by_path.py,{{ track_file }},{{ cal_table }},{{ config_file }}
output                = {{ log_dir }}/$(ClusterId).$(ProcId).extractEvents.stdout
error                 = {{ log_dir }}/$(ClusterId).$(ProcId).extractEvents.stderr
log                   = {{ log_dir }}/extractEvents.log
MY.WantOS             = "el9"
MY.XRDCP_CREATE_DIR   = True
output_destination    = root://eosuser.cern.ch/{{ eos_base }}/{{ out_dir }}
+JobFlavour           = "microcentury"
Queue fname from {{ script_dir }}/input_list.txt
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

def create_submission_files(args, trig_id, paths, eos_base):

    config_path = Path(args.config)
    cal_path = Path(args.cal_table)
    track_path = Path(args.track)
    final_input_dir = Path(eos_base) / args.dirname

    # 2. No unlink() needed
    input_list_path = paths['scripts_dir'] / 'input_list.txt'
    feather_files = natsorted(final_input_dir.glob('loop*feather'))

    with open(input_list_path, 'w') as f:
        for file_path in feather_files:
            f.write(f"{file_path.name}\n")

    # 3. Pass Path objects directly to Template
    bash_content = Template(BASH_TEMPLATE).render(
        runname=args.runName,
        track=track_path,
        cal_table=cal_path,
        trigID=trig_id,
        search_method=args.search_method,
        config=config_path,
    )

    bash_script_path = paths['scripts_dir'] / f'run_extract_events.sh'
    with open(bash_script_path, 'w') as f:
        f.write(bash_content)

    # 3. Generate JDL File
    jdl_content = Template(JDL_TEMPLATE).render(
        script_dir=paths['scripts_dir'],
        input_dir=final_input_dir,
        track_file=track_path,
        cal_table=cal_path,
        config_file=config_path,
        log_dir=paths['log_dir'],
        eos_base=eos_base,
        out_dir=args.outname,
    )

    jdl_path = paths['scripts_dir'] / f'condor_extract_events.jdl'
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
    run_append = f"{args.condor_tag}" if args.condor_tag else "subdir"

    # Directory setup
    paths = {
        'scripts_dir': Path('.') / 'condor_scripts' / 'extract_events' / f'{run_append}',
        'log_dir': Path('.') / 'condor_logs' / 'extract_events' / f'{run_append}'
    }

    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    # --- Validation ---
    if not Path('core/extract_events_by_path.py').is_file():
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
    print(f'Input CAL table: {args.cal_table}')
    print(f'Input track file: {args.track}')
    print(f'Trigger ID:  {trig_id}')
    print(f'Output:      {eos_base_dir}/{args.outname}')
    if args.search_method != 'none':
        print(f'Neighbor search method: {args.search_method}')
    print('======================================\n')

    jdl_file, bash_file, list_file = create_submission_files(
        args, trig_id, paths, eos_base_dir
    )

    # --- Submission ---
    if args.dryrun:
        print("--- Dry Run: Files Generated ---")
        print(f"[Dry Run]JDL:  {jdl_file}")
        print(f"[Dry Run] Bash: {bash_file}")
        print(f"[Dry Run] List: {list_file}")

    else:
        # Standard Submission
        if list_file.stat().st_size > 0:
            subprocess.run(['condor_submit', str(jdl_file)])
        else:
            print("No input files found in directory. Nothing submitted.")