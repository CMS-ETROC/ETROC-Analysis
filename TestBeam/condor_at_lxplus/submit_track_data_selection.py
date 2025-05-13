import getpass, subprocess
import argparse
from pathlib import Path
from jinja2 import Template
from natsort import natsorted

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
    '-r',
    '--runName',
    metavar = 'NAME',
    type = str,
    help = 'Name of the run to process. If given, the run name will be used to avoid file collisions',
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
    '--trigID',
    metavar = 'ID',
    type = int,
    help = 'trigger board ID',
    required = True,
    dest = 'trigID',
)

parser.add_argument(
    '--refID',
    metavar = 'ID',
    type = int,
    help = 'reference board ID',
    default = 3,
    dest = 'refID',
)

parser.add_argument(
    '--dutID',
    metavar = 'ID',
    type = int,
    help = 'DUT board ID',
    default = 1,
    dest = 'dutID',
)

parser.add_argument(
    '--extraID',
    metavar = 'ID',
    type = int,
    help = 'board ID be ignored',
    default = 2,
    dest = 'extraID',
)

parser.add_argument(
    '--trigTOTLower',
    metavar = 'NUM',
    type = int,
    help = 'Lower TOT selection boundary for the trigger board',
    default = 100,
    dest = 'trigTOTLower',
)

parser.add_argument(
    '--trigTOTUpper',
    metavar = 'NUM',
    type = int,
    help = 'Upper TOT selection boundary for the trigger board',
    default = 200,
    dest = 'trigTOTUpper',
)

parser.add_argument(
    '--dryrun',
    action = 'store_true',
    help = 'If set, condor submission will not happen',
    dest = 'dryrun',
)

def load_bash_template(args):
    options = {
        'filename': '${1}',
        'runname': '${2}',
        'path': '${3}',
        'track': args.track,
        'cal_table': args.cal_table.split('/')[-1],
        'trigID': args.trigID,
        'refID': args.refID,
        'dutID': args.dutID,
        'extraID': args.extraID,
        'trigTOTLower': args.trigTOTLower,
        'trigTOTUpper': args.trigTOTUpper,
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

echo "python track_data_selection.py -f {{ filename }} -r {{ runname }} -t {{ track }} --trigID {{ trigID }} --refID {{ refID }} --dutID {{ dutID }} --extraID {{ extraID }} --trigTOTLower {{ trigTOTLower }} --trigTOTUpper {{ trigTOTUpper }} --cal_table {{ cal_table }}"
python track_data_selection.py -f {{ filename }} -r {{ runname }} -t {{ track }} --trigID {{ trigID }} --refID {{ refID }} --dutID {{ dutID }} --extraID {{ extraID }} --trigTOTLower {{ trigTOTLower }} --trigTOTUpper {{ trigTOTUpper }} --cal_table {{ cal_table }}

ls -ltrh
echo ""

# Delete input file so condor will not return
rm {{ filename }}

ls -ltrh
echo ""
"""

    return Template(bash_template).render(options)

args = parser.parse_args()
username = getpass.getuser()
eos_base_dir = f'/eos/user/{username[0]}/{username}'

runName = args.runName
if runName is None:
    runAppend = ""
else:
    runAppend = "_" + runName

condor_scripts_dir = Path('./') / 'condor_scripts' / f'trackDataSelect_job{runAppend}'
condor_scripts_dir.mkdir(exist_ok=True, parents=True)

listfile = condor_scripts_dir / f'input_list_for_trackDataSelection{runAppend}.txt'
if listfile.is_file():
    listfile.unlink()

with open(listfile, 'a') as base_txt:
    # dirname="......../physics_run_N_feather" or ......./<any name>_feather
    files = natsorted(Path(args.dirname).glob('loop*feather'))
    for ifile in files:
        fname = ifile.name
        save_string = f"{Path(args.dirname).name.split('_feather')[0]}, {fname}, {ifile}"
        base_txt.write(save_string + '\n')

log_dir = Path('./') / 'condor_logs' / 'track_data_selection' / f'trackDataSelect_job{runAppend}'
log_dir.mkdir(exist_ok=True, parents=True)

# if log_dir.exists():
#     # Remove files
#     subprocess.run(f'rm {log_dir}/*trackSelection*log', shell=True)
#     subprocess.run(f'rm {log_dir}/*trackSelection*stdout', shell=True)
#     subprocess.run(f'rm {log_dir}/*trackSelection*stderr', shell=True)

#     # Count files
#     result = subprocess.run(f'ls {log_dir}/*trackSelection*log | wc -l', shell=True, capture_output=True, text=True)
#     print("Log file count:", result.stdout.strip())

print('\n========= Run option =========')
print(f'Input dataset: {args.dirname}')
print(f'Track csv file: {args.track}')
print(f'Cal code mode table: {args.cal_table}')
print(f'Output will be stored {eos_base_dir}/{args.outname}')
print(f'Trigger board ID: {args.trigID}')
print(f'DUT board ID: {args.dutID}')
print(f'Reference board ID: {args.refID}')
print(f'Second reference (or will be ignored) board ID: {args.extraID}')
print(f"TOT cut is {args.trigTOTLower}-{args.trigTOTUpper} on board ID={args.trigID}")
print('========= Run option =========\n')

# Render the template with the data
bash_script = load_bash_template(args)
with open(condor_scripts_dir / f'run_track_data_selection{runAppend}.sh','w') as bashfile:
    bashfile.write(bash_script)

jdl = """universe              = vanilla
executable            = {5}/run_track_data_selection.sh
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
Queue run,fname,path from {5}/input_list_for_trackDataSelection.txt
""".format(log_dir, args.track, args.cal_table, eos_base_dir, args.outname, condor_scripts_dir)

with open(condor_scripts_dir / f'condor_track_data_selection{runAppend}.jdl','w') as jdlfile:
    jdlfile.write(jdl)

if args.dryrun:
    print('=========== Input text file ===========')
    subprocess.run(f"head -n 10 {listfile}", shell=True)
    subprocess.run(f"tail -n 10 {listfile}", shell=True)
    print()
    print('=========== Bash file ===========')
    with open(condor_scripts_dir / f"run_track_data_selection.sh{runAppend}") as f:
        print(f.read(), '\n')
    print()
    print('=========== Condor Job Description file ===========')
    with open(condor_scripts_dir / f"condor_track_data_selection{runAppend}.jdl") as f:
        print(f.read(), '\n')
    print()
    print()
else:
    subprocess.run(['condor_submit', f'{condor_scripts_dir}/condor_track_data_selection{runAppend}.jdl'])