from pathlib import Path
from jinja2 import Template
from natsort import natsorted

import argparse
import subprocess
import sys

def load_bash_template(input_dir_path):
    bash_template = """#!/bin/bash
ls -ltrh
echo ""

source /cvmfs/sft.cern.ch/lcg/views/LCG_104a/x86_64-el9-gcc13-opt/setup.sh

# Directory name uses $2 (index) and $1 (ClusterID)
mkdir -p ./job_${1}_${2}

# IMPORTANT: We do NOT quote $3 so bash splits the file bundle into individual names
for fname in $3; do
    echo "Transferring ${fname}..."
    xrdcp -s root://eosuser.cern.ch/{{ eos_path }}/${fname} ./job_${1}_${2}
done

tar -xf python_lib.tar
export PYTHONPATH=${PWD}/local/lib/python3.9/site-packages:$PYTHONPATH

# Output loop index is now strictly numeric (no commas)
echo "python decoding.py -d job_${1}_${2} -o loop_${2}"
python decoding.py -d job_${1}_${2} -o loop_${2}

rm -r job_${1}_${2}
"""
    options = {
        'eos_path': input_dir_path,
    }
    return Template(bash_template).render(options)


def load_jdl_template(condor_log_dir, output_dir, condor_scripts_dir):

    import getpass
    username = getpass.getuser()
    eos_base_dir = f'/eos/user/{username[0]}/{username}'

    jdl = """universe              = vanilla
executable            = {3}/run_decode.sh
should_Transfer_Files = YES
whenToTransferOutput  = ON_EXIT
# $1:ClusterId, $2:Index, $3:FileBundle
arguments             = $(ClusterId) $(index) $(flist)
transfer_Input_Files  = core/decoding.py, utils/python_lib.tar
output                = {0}/$(ClusterId).$(ProcId).decoding.stdout
error                 = {0}/$(ClusterId).$(ProcId).decoding.stderr
log                   = {0}/decoding.log
transfer_Output_Files = hits, status
output_destination    = root://eosuser.cern.ch/{1}/{2}
MY.XRDCP_CREATE_DIR   = True
MY.WantOS             = "el9"
+JobFlavour           = "workday"
# We define the TAB (\t) as the delimiter to keep flist clean
Queue index, flist from {3}/input_list.txt DELIMITER=\\t
""".format(condor_log_dir, eos_base_dir, output_dir, condor_scripts_dir)
    return jdl

def make_jobs(args, log_dir, condor_scripts_dir):

    input_path = Path(args.input_dir)
    file_extension = None
    file_suffix = None
    file_list = None

    bin_files = natsorted(input_path.glob('file*.bin'))
    if bin_files:
        print("Found '.bin' files.")
        file_extension = 'bin'
        file_suffix = ''  # No suffix for .bin files
        file_list = bin_files
    else:
        # If no .bin files, check for '_CE.dat' files.
        dat_files = natsorted(input_path.glob('file*_CE.dat'))
        if dat_files:
            print("Found '_CE.dat' files.")
            file_extension = 'dat'
            file_suffix = '_CE' # Suffix is '_CE' for .dat files
            file_list = dat_files

    if file_list is None:
        print(f"!!! ERROR: No 'file*.bin' or 'file*_CE.dat' files found in '{input_path}'. Exiting.")
        sys.exit(1)

    print(f"--> Auto-detected and using file extension: '{file_extension}'")

    glob_pattern = f'file*.{file_extension}'
    file_list = natsorted(input_path.glob(glob_pattern))
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

    listfile = condor_scripts_dir / f'input_list.txt'
    if listfile.is_file():
        listfile.unlink()

    idx_ptr = 0
    with open(listfile, 'w') as f:
        for job_id in range(num_jobs):
            job_size = files_per_job + (1 if job_id < remainder else 0)
            chunk = file_list[idx_ptr:idx_ptr + job_size]

            # Use physical index from the filename to prevent overwrites
            physical_idx = int(chunk[0].name.split('.')[0].split('_')[1])

            # Space-separated bundle of filenames
            file_bundle = " ".join([f.name for f in chunk])

            # Write: Index [TAB] File1 File2 File3
            f.write(f"{physical_idx}\t{file_bundle}\n")

            idx_ptr += job_size

    bash_script = load_bash_template(args.input_dir)

    with open(condor_scripts_dir / f'run_decode.sh','w') as bashfile:
        bashfile.write(bash_script)

    outdir = f'{args.output}'
    jdl_script = load_jdl_template(condor_log_dir=log_dir, output_dir=outdir, condor_scripts_dir=condor_scripts_dir)
    with open(condor_scripts_dir / f'condor_decoding.jdl','w') as jdlfile:
        jdlfile.write(jdl_script)


## --------------------------------------
if __name__ == "__main__":


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
        runAppend = "subdir"
    else:
        runAppend = tag_for_condor

    log_dir = Path('./') / 'condor_logs' / 'decoding' / f'{runAppend}'
    log_dir.mkdir(exist_ok=True, parents=True)

    condor_scripts_dir = Path('./') / 'condor_scripts' / 'decoding' / f'{runAppend}'
    condor_scripts_dir.mkdir(exist_ok=True, parents=True)

    make_jobs(args=args, log_dir=log_dir, condor_scripts_dir=condor_scripts_dir)

    if args.dryrun:
        input_txt_path = condor_scripts_dir / f"input_list.txt"
        print('\n=========== Input text file ===========')
        print('First 3 lines:')
        subprocess.run(f"head -n 3 {input_txt_path}", shell=True)
        print('Last 3 lines:')
        subprocess.run(f"tail -n 3 {input_txt_path}", shell=True)
        print()
        print('=========== Bash file ===========')
        with open(condor_scripts_dir / f"run_decode.sh") as f:
            print(f.read(), '\n')
        print('=========== Condor Job Description file ===========')
        with open(condor_scripts_dir / f'condor_decoding.jdl') as f:
            print(f.read(), '\n')
        print()

    elif args.resubmit:
        input_txt_path = condor_scripts_dir / f"input_list.txt"
        condor_output = subprocess.run(['condor_q', '-nobatch'], capture_output=True, text=True)

        # Filter lines containing the target script
        filtered_lines = [line for line in condor_output.stdout.splitlines() if f'run_decode.sh' in line]

        if len(filtered_lines) == 0:
            print('No condor job found.')
            sys.exit(1)

        with open(input_txt_path, 'w') as f:
            for line in filtered_lines:
                fields = line.split()
                if len(fields) == 13:
                    if fields[5] == 'X':
                        old_condor_job_id = -1
                        continue
                    old_condor_job_id = fields[0].split('.')[0]
                    last_three = fields[-4:-1]  # Extract 4th to last, 3rd to last, 2nd to last
                    sentence = ' '.join(last_three)
                    f.write(sentence + '\n')

        if not old_condor_job_id == -1:
            ## Kill old jobs before submit new one
            subprocess.run(['condor_rm', f'{old_condor_job_id}'])

            if input_txt_path.stat().st_size > 0:
                subprocess.run(['condor_submit', f'{condor_scripts_dir}/condor_decoding.jdl'])
            else:
                print("No jobs to submit — input list is empty.")

    else:
        subprocess.run(['condor_submit', f'{condor_scripts_dir}/condor_decoding.jdl'])
