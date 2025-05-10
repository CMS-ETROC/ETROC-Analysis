import getpass, subprocess
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

username = getpass.getuser()
eos_base_dir = f'/eos/user/{username[0]}/{username}'
outdir = f'{args.run_name}_feather'

file_list = natsorted(Path(args.input_dir).glob('file*bin'))

listfile = Path('./') / 'input_list_for_decoding.txt'
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

log_dir = Path('./') / 'condor_logs' / 'decoding'
log_dir.mkdir(exist_ok=True, parents=True)

if log_dir.exists():
    # Remove files
    subprocess.run(f'rm {log_dir}/*decoding*log', shell=True)
    subprocess.run(f'rm {log_dir}/*decoding*stdout', shell=True)
    subprocess.run(f'rm {log_dir}/*decoding*stderr', shell=True)

    # Count files
    result = subprocess.run(f'ls {log_dir}/*decoding*log | wc -l', shell=True, capture_output=True, text=True)
    print("Log file count:", result.stdout.strip())

jdl = """universe              = vanilla
executable            = run_decode.sh
should_Transfer_Files = YES
whenToTransferOutput  = ON_EXIT
arguments             = $(start) $(end) $(index) $(ClusterId)
transfer_Input_Files  = decoding.py, python_lib.tar
output                = {0}/$(ClusterId).$(ProcId).decoding.stdout
error                 = {0}/$(ClusterId).$(ProcId).decoding.stderr
log                   = {0}/$(ClusterId).$(ProcId).decoding.log
output_destination    = root://eosuser.cern.ch/{1}/{2}
MY.XRDCP_CREATE_DIR   = True
MY.WantOS             = "el9"
+JobFlavour           = "longlunch"
Queue start, end, index from input_list_for_decoding.txt
""".format(log_dir, eos_base_dir, outdir)

with open(f'condor_decoding.jdl','w') as jdlfile:
    jdlfile.write(jdl)

if args.dryrun:
    print('\n=========== Input text file ===========')
    subprocess.run("head -n 10 input_list_for_decoding.txt", shell=True)
    subprocess.run("tail -n 10 input_list_for_decoding.txt", shell=True)
    print()
    print('=========== Bash file ===========')
    with open("run_decode.sh") as f:
        print(f.read(), '\n')
    print('=========== Condor Job Description file ===========')
    with open('condor_decoding.jdl') as f:
        print(f.read(), '\n')
    print()
else:
    subprocess.run(['condor_submit', '-spool', 'condor_decoding.jdl'])