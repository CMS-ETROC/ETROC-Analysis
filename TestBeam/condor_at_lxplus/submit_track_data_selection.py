from pathlib import Path
from jinja2 import Template
from natsort import natsorted
import yaml

def load_bash_template(args, id_roles):
    options = {
        'filename': '${1}',
        'runname': '${2}',
        'path': '${3}',
        'track': args.track,
        'cal_table': args.cal_table.split('/')[-1],
        'trigID': id_roles['trig'],
        'refID': id_roles['ref'],
        'dutID': id_roles['dut'],
        'extraID': id_roles['extra'],
    }

    bash_template = """#!/bin/bash

ls -ltrh
echo ""
pwd

# Load python environment from work node
source /cvmfs/sft.cern.ch/lcg/views/LCG_104a/x86_64-el9-gcc13-opt/setup.sh

# Copy input data from EOS to local work node
xrdcp -r root://eosuser.cern.ch/{{ path }} ./

echo "Will process input file from {{ runname }} {{ filename }}"

echo "python track_data_selection.py -f {{ filename }} -r {{ runname }} -t {{ track }} --trigID {{ trigID }} --refID {{ refID }} --dutID {{ dutID }} --extraID {{ extraID }} --cal_table {{ cal_table }}"
python track_data_selection.py -f {{ filename }} -r {{ runname }} -t {{ track }} --trigID {{ trigID }} --refID {{ refID }} --dutID {{ dutID }} --extraID {{ extraID }} --cal_table {{ cal_table }}

ls -ltrh
echo ""

# Delete input file so condor will not return
rm {{ filename }}

ls -ltrh
echo ""
"""

    return Template(bash_template).render(options)

def load_jdl_template(args, log_dir, eos_base_dir, condor_scripts_dir, runAppend):
    jdl = """universe              = vanilla
executable            = {5}/run_track_data_selection{6}.sh
should_Transfer_Files = YES
whenToTransferOutput  = ON_EXIT
arguments             = $(fname) $(run) $(path)
transfer_Input_Files  = track_data_selection.py,{1},{2}
output                = {0}/$(ClusterId).$(ProcId).trackSelection.stdout
error                 = {0}/$(ClusterId).$(ProcId).trackSelection.stderr
log                   = {0}/$(ClusterId).$(ProcId).trackSelection.log
MY.WantOS             = "el9"
MY.XRDCP_CREATE_DIR   = True
output_destination    = root://eosuser.cern.ch/{3}/{4}
+JobFlavour           = "microcentury"
Queue fname,run,path from {5}/input_list_for_trackDataSelection{6}.txt
""".format(log_dir, args.track, args.cal_table, eos_base_dir, args.outname, condor_scripts_dir, runAppend)

    return jdl

def make_jobs(args, id_roles, log_dir, eos_base_dir, condor_scripts_dir, runAppend):

    listfile = condor_scripts_dir / f'input_list_for_trackDataSelection{runAppend}.txt'
    if listfile.is_file():
        listfile.unlink()

    with open(listfile, 'a') as base_txt:
        # dirname="......../physics_run_N_feather" or ......./<any name>_feather
        files = natsorted(Path(args.dirname).glob('loop*feather'))
        for ifile in files:
            fname = ifile.name
            save_string = f"{fname}, {Path(args.dirname).name.split('_feather')[0]}, {ifile}"
            base_txt.write(save_string + '\n')

    # Render the template with the data
    bash_script = load_bash_template(args, id_roles)
    with open(condor_scripts_dir / f'run_track_data_selection{runAppend}.sh','w') as bashfile:
        bashfile.write(bash_script)

    jdl_scripts = load_jdl_template(args, log_dir, eos_base_dir, condor_scripts_dir, runAppend)
    with open(condor_scripts_dir / f'condor_track_data_selection{runAppend}.jdl','w') as jdlfile:
        jdlfile.write(jdl_scripts)


## --------------------------------------
if __name__ == "__main__":

    import getpass, subprocess
    import argparse

    parser = argparse.ArgumentParser(
            prog='PlaceHolder',
            description='submit condor job!',
        )

    parser.add_argument(
        '-d',
        '--inputdir',
        metavar = 'DIRNAME',
        type = str,
        help = 'input directory name',
        required = True,
        dest = 'dirname',
    )

    parser.add_argument(
        '-t',
        '--track',
        metavar = 'NAME',
        type = str,
        help = 'csv file including track candidates',
        required = True,
        dest = 'track',
    )

    parser.add_argument(
        '-c',
        '--config',
        metavar = 'NAME',
        type = str,
        help = 'YAML file including run information.',
        required = True,
        dest = 'config',
    )

    parser.add_argument(
        '-r',
        '--runName',
        metavar = 'NAME',
        type = str,
        help = 'Name of the run to process. It must be matched with the name defined in YAML.',
        required = True,
        dest = 'runName',
    )

    parser.add_argument(
        '--cal_table',
        metavar = 'NAME',
        type = str,
        help = 'csv file including CAL mode values per board, per pixel',
        required = True,
        dest = 'cal_table',
    )

    parser.add_argument(
        '-o',
        '--outdir',
        metavar = 'DIRNAME',
        type = str,
        help = 'output directory name',
        default = 'dataSelection_outputs',
        dest = 'outname',
    )

    parser.add_argument(
        '--condor_tag',
        metavar = 'NAME',
        type = str,
        help = 'Tag of the run to process on condor. If given, the tag will be used to avoid file collisions',
        dest = 'condor_tag',
    )

    parser.add_argument(
        '--dryrun',
        action = 'store_true',
        help = 'If set, condor submission will not happen',
        dest = 'dryrun',
    )

    parser.add_argument(
        '--resubmit',
        action = 'store_true',
        help = 'If set, condor resubmission for jobs in Running and IDLE. Will kill the old jobs.',
        dest = 'resubmit',
    )

    args = parser.parse_args()

    tag_for_condor = args.condor_tag
    if tag_for_condor is None:
        runAppend = ""
    else:
        runAppend = "_" + tag_for_condor

    with open(args.config) as input_yaml:
        config = yaml.safe_load(input_yaml)

    if args.runName not in config:
        raise ValueError(f"Run config {args.runName} not found")

    roles = {}
    for board_id, board_info in config[args.runName].items():
        roles[board_info.get('role')] = board_id

    condor_scripts_dir = Path('./') / 'condor_scripts' / f'trackDataSelect_job{runAppend}'
    condor_scripts_dir.mkdir(exist_ok=True, parents=True)

    log_dir = Path('./') / 'condor_logs' / 'track_data_selection' / f'trackDataSelect_job{runAppend}'
    log_dir.mkdir(exist_ok=True, parents=True)

    username = getpass.getuser()
    eos_base_dir = f'/eos/user/{username[0]}/{username}'

    print('\n========= Run option =========')
    print(f'Input dataset: {args.dirname}')
    print(f'Track csv file: {args.track}')
    print(f'Cal code mode table: {args.cal_table}')
    print(f'Output will be stored {eos_base_dir}/{args.outname}')
    print(f'Trigger board ID: {roles["trig"]}')
    print(f'DUT board ID: {roles["dut"]}')
    print(f'Reference board ID: {roles["ref"]}')
    print(f'Second reference (or will be ignored) board ID: {roles["extra"]}')
    # print(f'TOT cut is {args.trigTOTLower}-{args.trigTOTUpper} on board ID={roles["trig"]}')
    print('========= Run option =========\n')

    make_jobs(args=args, id_roles=roles, log_dir=log_dir, eos_base_dir=eos_base_dir, condor_scripts_dir=condor_scripts_dir, runAppend=runAppend)

    if args.dryrun:
        input_txt_path = condor_scripts_dir / f"input_list_for_trackDataSelection{runAppend}.txt"
        print('=========== Input text file ===========')
        subprocess.run(f"head -n 10 {input_txt_path}", shell=True)
        subprocess.run(f"tail -n 10 {input_txt_path}", shell=True)
        print()
        print('=========== Bash file ===========')
        with open(condor_scripts_dir / f"run_track_data_selection{runAppend}.sh") as f:
            print(f.read(), '\n')
        print()
        print('=========== Condor Job Description file ===========')
        with open(condor_scripts_dir / f"condor_track_data_selection{runAppend}.jdl") as f:
            print(f.read(), '\n')
        print()
        print()
    elif args.resubmit:
        input_txt_path = condor_scripts_dir / f"input_list_for_trackDataSelection{runAppend}.txt"
        condor_output = subprocess.run(['condor_q', '-nobatch'], capture_output=True, text=True)

        # Filter lines containing the target script
        filtered_lines = [line for line in condor_output.stdout.splitlines() if f'run_track_data_selection{runAppend}.sh' in line]

        if len(filtered_lines) == 0:
            print('No condor job found.')
            import sys
            sys.exit(1)

        with open(input_txt_path, 'w') as f:
            for line in filtered_lines:
                fields = line.split()
                if len(fields) == 12:
                    if fields[5] == 'X':
                        old_condor_job_id = -1
                        continue
                    old_condor_job_id = fields[0].split('.')[0]
                    last_three = fields[-3:]
                    sentence = ' '.join(last_three)
                    f.write(sentence + '\n')

        if not old_condor_job_id == -1:
            ## Kill old jobs before submit new one
            subprocess.run(['condor_rm', f'{old_condor_job_id}'])

            if input_txt_path.stat().st_size > 0:
                subprocess.run(['condor_submit', f'{condor_scripts_dir}/condor_track_data_selection{runAppend}.jdl'])
            else:
                print("No jobs to submit — input list is empty.")
    else:
        subprocess.run(['condor_submit', f'{condor_scripts_dir}/condor_track_data_selection{runAppend}.jdl'])