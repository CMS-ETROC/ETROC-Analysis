import os, re
from pathlib import Path
import argparse
from glob import glob
from jinja2 import Template

parser = argparse.ArgumentParser(
            prog='PlaceHolder',
            description='submit condor job!',
        )

parser.add_argument(
    '-d',
    '--inputdir',
    nargs='+',
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
    '--ignoreID',
    metavar = 'ID',
    type = int,
    help = 'board ID be ignored',
    default = 2,
    dest = 'ignoreID',
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
    '--load_from_eos',
    action = 'store_true',
    help = 'If set, bash script and condor jdl will include EOS command',
    dest = 'load_from_eos',
)

parser.add_argument(
    '--dryrun',
    action = 'store_true',
    help = 'If set, condor submission will not happen',
    dest = 'dryrun',
)

args = parser.parse_args()
current_dir = Path('./')

dirs = args.dirname

listfile = current_dir / 'input_list_for_dataSelection.txt'
if listfile.is_file():
    listfile.unlink()

with open(listfile, 'a') as listfile:
    for idir in dirs:
        files = glob(f'{idir}/loop*feather')
        for ifile in files:
            pattern = r'Run_(\d+)'
            fname = ifile.split('/')[-1]
            loop_name = fname.split('.')[0]
            matches = re.findall(pattern, ifile)
            save_string = f"run{matches[0]}, {fname}, {loop_name}, {ifile}"
            listfile.write(save_string + '\n')

log_dir = current_dir / 'condor_logs'
log_dir.mkdir(exist_ok=True)

if log_dir.exists():
    os.system('rm condor_logs/*trackSelection*log')
    os.system('rm condor_logs/*trackSelection*stdout')
    os.system('rm condor_logs/*trackSelection*stderr')
    os.system('ls condor_logs/*trackSelection*log | wc -l')

out_dir = current_dir / args.outname
out_dir.mkdir(exist_ok=False)

if args.load_from_eos:
    bash_template = """#!/bin/bash

ls -ltrh
echo ""
pwd

# Load python environment from work node
source /cvmfs/sft.cern.ch/lcg/views/LCG_104a/x86_64-el9-gcc13-opt/setup.sh

# Copy input data from EOS to local work node
xrdcp -r root://eosuser.cern.ch/{{ path }} ./

echo "Will process input file from {{ runname }} {{ filename }}"

echo "python track_data_selection.py -f {{ filename }} -r {{ runname }} -t {{ track }} --trigID {{ trigID }} --refID {{ refID }} --dutID {{ dutID }} --ignoreID {{ ignoreID }} --trigTOTLower {{ trigTOTLower }} --trigTOTUpper {{ trigTOTUpper }}"
python track_data_selection.py -f {{ filename }} -r {{ runname }} -t {{ track }} --trigID {{ trigID }} --refID {{ refID }} --dutID {{ dutID }} --ignoreID {{ ignoreID }} --trigTOTLower {{ trigTOTLower }} --trigTOTUpper {{ trigTOTUpper }}

ls -ltrh
echo ""

# Delete input file so condor will not return
rm {{ filename }}

ls -ltrh
echo ""
"""

    # Prepare the data for the template
    options = {
        'filename': '${1}',
        'runname': '${2}',
        'path': '${3}',
        'track': args.track,
        'trigID': args.trigID,
        'refID': args.refID,
        'dutID': args.dutID,
        'ignoreID': args.ignoreID,
        'trigTOTLower': args.trigTOTLower,
        'trigTOTUpper': args.trigTOTUpper,
    }

else:
    bash_template = """#!/bin/bash

ls -ltrh
echo ""
pwd

# Load python environment from work node
source /cvmfs/sft.cern.ch/lcg/views/LCG_104a/x86_64-el9-gcc13-opt/setup.sh

echo "Will process input file from {{ runname }} {{ filename }}"

echo "python track_data_selection.py -f {{ filename }} -r {{ runname }} -t {{ track }} --trigID {{ trigID }} --refID {{ refID }} --dutID {{ dutID }} --ignoreID {{ ignoreID }} --trigTOTLower {{ trigTOTLower }} --trigTOTUpper {{ trigTOTUpper }}"
python track_data_selection.py -f {{ filename }} -r {{ runname }} -t {{ track }} --trigID {{ trigID }} --refID {{ refID }} --dutID {{ dutID }} --ignoreID {{ ignoreID }} --trigTOTLower {{ trigTOTLower }} --trigTOTUpper {{ trigTOTUpper }}

ls -ltrh
echo ""

# Delete input file so condor will not return
rm {{ filename }}

ls -ltrh
echo ""
"""

    # Prepare the data for the template
    options = {
        'filename': '${1}',
        'runname': '${2}',
        'track': args.track,
        'trigID': args.trigID,
        'refID': args.refID,
        'dutID': args.dutID,
        'ignoreID': args.ignoreID,
        'trigTOTLower': args.trigTOTLower,
        'trigTOTUpper': args.trigTOTUpper,
    }

# Render the template with the data
bash_script = Template(bash_template).render(options)

print('\n========= Run option =========')
print(f'Input dataset: {args.dirname}')
print(f'Track csv file: {args.track}')
print(f'Output will be stored {args.outname}')
print(f'Trigger board ID: {args.trigID}')
print(f'DUT board ID: {args.dutID}')
print(f'Reference board ID: {args.refID}')
print(f'Second reference (or will be ignored) board ID: {args.ignoreID}')
print(f"TOT cut is {args.trigTOTLower}-{args.trigTOTUpper} on board ID={args.trigID}")
if args.load_from_eos:
    print('Feather files will be load from EOS')
else:
    print('Feather files will be load from local area')
print('========= Run option =========\n')


with open('run_track_data_selection.sh','w') as bashfile:
    bashfile.write(bash_script)

if args.load_from_eos:
    jdl = """universe              = vanilla
executable            = run_track_data_selection.sh
should_Transfer_Files = YES
whenToTransferOutput  = ON_EXIT
arguments             = $(fname) $(run) $(path)
transfer_Input_Files  = track_data_selection.py,{1}
TransferOutputRemaps = "$(run)_$(loop).pickle={2}/$(run)_$(loop).pickle"
output                = {0}/$(ClusterId).$(ProcId).trackSelection.stdout
error                 = {0}/$(ClusterId).$(ProcId).trackSelection.stderr
log                   = {0}/$(ClusterId).$(ProcId).trackSelection.log
MY.WantOS             = "el9"
+JobFlavour           = "microcentury"
Queue run,fname,loop,path from input_list_for_dataSelection.txt
""".format(str(log_dir), args.track, str(out_dir))

else:
    jdl = """universe              = vanilla
executable            = run_track_data_selection.sh
should_Transfer_Files = YES
whenToTransferOutput  = ON_EXIT
arguments             = $(fname) $(run)
transfer_Input_Files  = track_data_selection.py,{1},$(path)
TransferOutputRemaps = "$(run)_$(loop).pickle={2}/$(run)_$(loop).pickle"
output                = {0}/$(ClusterId).$(ProcId).trackSelection.stdout
error                 = {0}/$(ClusterId).$(ProcId).trackSelection.stderr
log                   = {0}/$(ClusterId).$(ProcId).trackSelection.log
MY.WantOS             = "el9"
+JobFlavour           = "microcentury"
Queue run,fname,loop,path from input_list_for_dataSelection.txt
""".format(str(log_dir), args.track, str(out_dir))


with open(f'condor_track_data_selection.jdl','w') as jdlfile:
    jdlfile.write(jdl)

if args.dryrun:
    print('=========== Input text file ===========')
    os.system('cat input_list_for_dataSelection.txt')
    print()
    print('=========== Bash file ===========')
    os.system('cat run_track_data_selection.sh')
else:
    os.system(f'condor_submit condor_track_data_selection.jdl')
