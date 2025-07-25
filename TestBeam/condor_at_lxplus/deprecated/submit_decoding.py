import os
from pathlib import Path
import argparse
from glob import glob
from jinja2 import Template
from natsort import natsorted

parser = argparse.ArgumentParser(
            prog='PlaceHolder',
            description='submit decoding jobs on condor',
        )

parser.add_argument(
    '-d',
    '--input_dir',
    metavar = 'NAME',
    type = str,
    help = 'input directory containing .bin',
    required = True,
    dest = 'input_dir',
)

parser.add_argument(
    '--run_name',
    metavar = 'NAME',
    type = str,
    help = 'extra run information for output directory name. Example: Run_X. X can be any number.',
    required = True,
    dest = 'run_name',
)

parser.add_argument(
    '--dryrun',
    action = 'store_true',
    help = 'If set, condor submission will not happen',
    dest = 'dryrun',
)

args = parser.parse_args()

current_dir = Path('./')
dir_list = natsorted(list(Path(args.input_dir).glob('loop*')))

listfile = current_dir / 'input_list_for_decoding.txt'
if listfile.is_file():
    listfile.unlink()

with open(listfile, 'a') as listfile:
    for idir in dir_list:
        name = str(idir).split('/')[-1]
        save_string = f"{name}, {str(idir)}"
        listfile.write(save_string + '\n')

outdir = current_dir / f'{args.run_name}_feather'
outdir.mkdir(exist_ok = False)

# Define the bash script template
bash_template = """#!/bin/bash

# Check current directory to make sure that input files are transferred
ls -ltrh
echo ""

# Load python environment from work node
source /cvmfs/sft.cern.ch/lcg/views/LCG_104a/x86_64-el9-gcc13-opt/setup.sh

# Copy input data from EOS to local work node
xrdcp -r root://eosuser.cern.ch/{{ input_dir_path }} ./

# Untar python environment
tar -xf python_lib.tar

# Check untar output
ls -ltrh

# Set custom python environment
export PYTHONPATH=${PWD}/local/lib/python3.9/site-packages:$PYTHONPATH
echo "${PYTHONPATH}"
echo ""

echo "python decoding.py -d {{ input_dir_name }}"
python decoding.py -d {{ input_dir_name }}
"""

# Prepare the data for the template
options = {
    'input_dir_name': '${1}',
    'input_dir_path': '${2}'
}

# Render the template with the data
bash_script = Template(bash_template).render(options)

with open('run_decode.sh','w') as bashfile:
    bashfile.write(bash_script)

log_dir = current_dir / 'condor_logs'
log_dir.mkdir(exist_ok=True)

if log_dir.exists():
    os.system('rm condor_logs/*decoding*log')
    os.system('rm condor_logs/*decoding*stdout')
    os.system('rm condor_logs/*decoding*stderr')
    os.system('ls condor_logs/*decoding*log | wc -l')

jdl = """universe              = vanilla
executable            = run_decode.sh
should_Transfer_Files = YES
whenToTransferOutput  = ON_EXIT
arguments             = $(name) $(path)
transfer_Input_Files  = decoding.py, python_lib.tar
TransferOutputRemaps = "$(name).feather={1}/$(name).feather;filler_$(name).feather={1}/filler_$(name).feather"
output                = {0}/$(ClusterId).$(ProcId).decoding.stdout
error                 = {0}/$(ClusterId).$(ProcId).decoding.stderr
log                   = {0}/$(ClusterId).$(ProcId).decoding.log
MY.WantOS             = "el9"
+JobFlavour           = "longlunch"
Queue name, path from input_list_for_decoding.txt
""".format(str(log_dir), str(outdir))

with open(f'condor_decoding.jdl','w') as jdlfile:
    jdlfile.write(jdl)

if args.dryrun:
    print('=========== Input text file ===========')
    os.system('cat input_list_for_decoding.txt')
    print()
    print('=========== Bash file ===========')
    os.system('cat run_decode.sh')
else:
    os.system(f'condor_submit condor_decoding.jdl')
