from pathlib import Path
from jinja2 import Template
from natsort import natsorted

def load_bash_template(input_dir_path):

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
        'eos_path': input_dir_path,
        'start': '${1}',
        'end': '${2}',
        'idx': '${3}',
        'ClusterID': '${4}',
    }

    # Render the template with the data
    return Template(bash_template).render(options)

def load_jdl_template(condor_log_dir, output_dir, runName, condor_scripts_dir):

    import getpass
    username = getpass.getuser()
    eos_base_dir = f'/eos/user/{username[0]}/{username}'

    jdl = """universe              = vanilla
executable            = {4}/run_decode{3}.sh
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
Queue start, end, index from {4}/input_list_for_decoding{3}.txt
""".format(condor_log_dir, eos_base_dir, output_dir, runName, condor_scripts_dir)

    return jdl

def make_jobs(args, log_dir, condor_scripts_dir, runAppend):

    file_list = natsorted(Path(args.input_dir).glob('file*bin'))
    print(f'\nFirst file: {file_list[0].name}')
    print(f'Last file: {file_list[-1].name}')

    if len(file_list) < args.range[0]:
        print(f'Number of input files: {len(file_list)} which is smaller than the given argument: {args.range[0]}')
        files_per_job = 1
        num_jobs = len(file_list) // files_per_job
        remainder = len(file_list) % files_per_job
        print(f"\nNumber of jobs: {num_jobs}")
        print(f"Each job gets 1 file.\n")
    else:
        files_per_job, remainder = min(((v, len(file_list) % v) for v in range(args.range[0], args.range[1]+1)), key=lambda x: x[1])
        num_jobs = len(file_list) // files_per_job

        print(f"\nNumber of jobs: {num_jobs}")
        print(f"Each job gets {files_per_job} files, with some jobs getting 1 extra file.\n")

    listfile = condor_scripts_dir / f'input_list_for_decoding{runAppend}.txt'
    if listfile.is_file():
        listfile.unlink()

    idx = 0
    with open(listfile, 'a') as base_txt:
        for job_id in range(num_jobs):
            # Distribute the remainder among the first `remainder` jobs
            job_size = files_per_job + (1 if job_id < remainder else 0)
            chunk = file_list[idx:idx + job_size]
            start = int(chunk[0].name.split('.')[0].split('_')[1])
            end = int(chunk[-1].name.split('.')[0].split('_')[1])
            save_string = f"{start}, {end}, {job_id}"
            base_txt.write(save_string + '\n')
            idx += job_size

    bash_script = load_bash_template(args.input_dir)
    with open(condor_scripts_dir / f'run_decode{runAppend}.sh','w') as bashfile:
        bashfile.write(bash_script)

    outdir = f'{args.output}_feather'
    jdl_script = load_jdl_template(condor_log_dir=log_dir, output_dir=outdir, runName=runAppend, condor_scripts_dir=condor_scripts_dir)
    with open(condor_scripts_dir / f'condor_decoding{runAppend}.jdl','w') as jdlfile:
        jdlfile.write(jdl_script)


## --------------------------------------
if __name__ == "__main__":
    import argparse
    import subprocess

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
        '-o',
        '--output',
        metavar = 'NAME',
        type = str,
        help = 'Output directory including path',
        required = True,
        dest = 'output',
    )

    parser.add_argument(
        '--range',
        metavar='N',
        type=int,
        nargs='+',
        help='Range to decide the number of files per job with the smallest remainder',
        default = [5, 10],
        dest = 'range',
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

    runName = args.runName
    if runName is None:
        runAppend = ""
    else:
        runAppend = "_" + runName

    log_dir = Path('./') / 'condor_logs' / 'decoding' / f'decoding_job{runAppend}'
    log_dir.mkdir(exist_ok=True, parents=True)

    condor_scripts_dir = Path('./') / 'condor_scripts' / f'decoding_job{runAppend}'
    condor_scripts_dir.mkdir(exist_ok=True, parents=True)

    make_jobs(args=args, log_dir=log_dir, condor_scripts_dir=condor_scripts_dir, runAppend=runAppend)

    if args.dryrun:
        input_txt_path = condor_scripts_dir / f"input_list_for_decoding{runAppend}.txt"
        print('\n=========== Input text file ===========')
        print('First 10 lines:')
        subprocess.run(f"head -n 10 {input_txt_path}", shell=True)
        print('Last 10 lines:')
        subprocess.run(f"tail -n 10 {input_txt_path}", shell=True)
        print()
        print('=========== Bash file ===========')
        with open(condor_scripts_dir / f"run_decode{runAppend}.sh") as f:
            print(f.read(), '\n')
        print('=========== Condor Job Description file ===========')
        with open(condor_scripts_dir / f'condor_decoding{runAppend}.jdl') as f:
            print(f.read(), '\n')
        print()

    elif args.resubmit:
        input_txt_path = condor_scripts_dir / f"input_list_for_decoding{runAppend}.txt"
        condor_output = subprocess.run(['condor_q', '-nobatch'], capture_output=True, text=True)

        # Filter lines containing the target script
        filtered_lines = [line for line in condor_output.stdout.splitlines() if f'run_decode{runAppend}.sh' in line]
        with open(input_txt_path, 'w') as f:
            for line in filtered_lines:
                fields = line.split()
                if len(fields) == 13:
                    if fields[5] == 'X':
                        continue
                    old_condor_job_id = fields[0].split('.')[0]
                    last_three = fields[-4:-1]  # Extract 4th to last, 3rd to last, 2nd to last
                    sentence = ' '.join(last_three)
                    f.write(sentence + '\n')

        ## Kill old jobs before submit new one
        subprocess.run(['condor_rm', f'{old_condor_job_id}'])

        if input_txt_path.stat().st_size > 0:
            subprocess.run(['condor_submit', f'{condor_scripts_dir}/condor_decoding{runAppend}.jdl'])
        else:
            print("No jobs to submit — input list is empty.")

    else:
        subprocess.run(['condor_submit', f'{condor_scripts_dir}/condor_decoding{runAppend}.jdl'])