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
    '--range',
    metavar='N',
    type=int,
    nargs='+',
    help='Range to decide how much files will be processed per job',
    default = [35, 46],
    dest = 'range',
)

parser.add_argument(
    '--dryrun',
    action = 'store_true',
    help = 'If set, condor submission will not happen',
    dest = 'dryrun',
)

args = parser.parse_args()

current_dir = Path('./')
file_list = natsorted(Path(args.input_dir).glob('file*bin'))

listfile = current_dir / 'input_list_for_decoding.txt'
if listfile.is_file():
    listfile.unlink()

files_per_job, remain = max(((v, len(file_list) % v) for v in range(args.range[0], args.range[1])), key=lambda x: x[1])
print('\nNumber of files per job:', files_per_job)
print(f'Last job will have {remain} files.\n')

with open(listfile, 'a') as listfile:

    point1 = int(file_list[0].name.split('.')[0].split('_')[1])
    point2 = int(file_list[-1].name.split('.')[0].split('_')[1])

    for idx, num in enumerate(range(point1, point2, files_per_job)):
        start = num
        end = min(num + files_per_job - 1, point2)
        save_string = f"{start}, {end}, {idx}"
        listfile.write(save_string + '\n')

outdir = current_dir / f'{args.run_name}_feather'
if not args.dryrun:
    outdir.mkdir(exist_ok = False)

# Define the bash script template
bash_template = """#!/bin/bash

# Check current directory to make sure that input files are transferred
ls -ltrh
echo ""

# Load python environment from work node
source /cvmfs/sft.cern.ch/lcg/views/LCG_104a/x86_64-el9-gcc13-opt/setup.sh

# Make temporary directory to save files
mkdir -p ./job_{{ ClusterID }}_{{ idx }}

# Copy input data from EOS to local work node
for num in $(seq {{ start }} {{ end }}); do
    xrdcp -s root://eosuser.cern.ch/{{ eos_path }}/file_${num}.bin ./job_{{ ClusterID }}_{{ idx }}
done

# Untar python environment
tar -xf python_lib.tar

# Check untar output
ls -ltrh

# Set custom python environment
export PYTHONPATH=${PWD}/local/lib/python3.9/site-packages:$PYTHONPATH
echo "${PYTHONPATH}"
echo ""

echo "python decoding.py -d job_{{ ClusterID }}_{{ idx }} -o loop_{{ idx }}"
python decoding.py -d job_{{ ClusterID }}_{{ idx }} -o loop_{{ idx }}

# Remove temporary directory
rm -r job_{{ ClusterID }}_{{ idx }}
"""

# Prepare the data for the template
options = {
    'eos_path': args.input_dir,
    'start': '${1}',
    'end': '${2}',
    'idx': '${3}',
    'ClusterID': '${4}',
}

# Render the template with the data
bash_script = Template(bash_template).render(options)

with open('run_decode.sh','w') as bashfile:
    bashfile.write(bash_script)

log_dir = current_dir / 'condor_logs' / 'decoding'
log_dir.mkdir(exist_ok=True, parents=True)

if log_dir.exists():
    os.system('rm condor_logs/decoding/*decoding*log')
    os.system('rm condor_logs/decoding/*decoding*stdout')
    os.system('rm condor_logs/decoding/*decoding*stderr')
    os.system('ls condor_logs/decoding/*decoding*log | wc -l')

jdl = """universe              = vanilla
executable            = run_decode.sh
should_Transfer_Files = YES
whenToTransferOutput  = ON_EXIT
arguments             = $(start) $(end) $(index) $(ClusterId)
transfer_Input_Files  = decoding.py, python_lib.tar
TransferOutputRemaps = "loop_$(index).feather={1}/loop_$(index).feather;filler_loop_$(index).feather={1}/filler_loop_$(index).feather"
output                = {0}/$(ClusterId).$(ProcId).decoding.stdout
error                 = {0}/$(ClusterId).$(ProcId).decoding.stderr
log                   = {0}/$(ClusterId).$(ProcId).decoding.log
MY.WantOS             = "el9"
+JobFlavour           = "longlunch"
Queue start, end, index from input_list_for_decoding.txt
""".format(str(log_dir), str(outdir))

with open(f'condor_decoding.jdl','w') as jdlfile:
    jdlfile.write(jdl)

if args.dryrun:
    print('\n=========== Input text file ===========')
    os.system('cat input_list_for_decoding.txt | tail -n 10')
    print()
    print('=========== Bash file ===========')
    os.system('cat run_decode.sh')
    print()
else:
    os.system(f'condor_submit condor_decoding.jdl')
