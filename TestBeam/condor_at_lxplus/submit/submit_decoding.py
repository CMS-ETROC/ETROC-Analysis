from pathlib import Path
from jinja2 import Template
from natsort import natsorted

import argparse
import subprocess
import sys
import getpass

def load_bash_template(input_dir_path):
    bash_template = """#!/bin/bash
ls -ltrh
echo ""

source /cvmfs/sft.cern.ch/lcg/views/LCG_104a/x86_64-el9-gcc13-opt/setup.sh

mkdir -p ./job_${1}_${2}

for fname in $3; do
    echo "Transferring ${fname}..."
    xrdcp -s root://eosuser.cern.ch/{{ eos_path }}/${fname} ./job_${1}_${2}
done

tar -xf python_lib.tar
export PYTHONPATH=${PWD}/local/lib/python3.9/site-packages:$PYTHONPATH

python decoding.py -d job_${1}_${2} -o loop_${2}
rm -r job_${1}_${2}
"""
    return Template(bash_template).render(eos_path=input_dir_path)


def load_jdl_template(log_dir, eos_base, out_dir, scripts_dir):
    jdl_template = """universe              = vanilla
executable            = {{ scripts_dir }}/run_decode.sh
should_Transfer_Files = YES
whenToTransferOutput  = ON_EXIT
arguments             = $(ClusterId) $(index) $(flist)
transfer_Input_Files  = core/decoding.py, utils/python_lib.tar
output                = {{ log_dir }}/$(ClusterId).$(ProcId).decoding.stdout
error                 = {{ log_dir }}/$(ClusterId).$(ProcId).decoding.stderr
log                   = {{ log_dir }}/decoding.log
transfer_Output_Files = hits, status
output_destination    = root://eosuser.cern.ch/{{ eos_base }}/{{ out_dir }}
MY.XRDCP_CREATE_DIR   = True
MY.WantOS             = "el9"
+JobFlavour           = "workday"
Queue index, flist from {{ scripts_dir }}/input_list.txt
"""
    return Template(jdl_template).render(
        log_dir=log_dir,
        eos_base=eos_base,
        out_dir=out_dir,
        scripts_dir=scripts_dir
    )

def make_jobs(args, log_dir, condor_scripts_dir):

    username = getpass.getuser()
    eos_base_dir = f'/eos/user/{username[0]}/{username}'
    input_path = Path(eos_base_dir) / args.input_dir

    # Detect files once and keep them
    file_list = natsorted(input_path.glob('file*.bin'))
    if file_list:
        file_extension = 'bin'
    else:
        file_list = natsorted(input_path.glob('file*_CE.dat'))
        file_extension = 'dat'

    if not file_list:
        print(f"!!! ERROR: No valid files found in '{input_path}'. Exiting.")
        sys.exit(1)

    print(f"--> Auto-detected and using file extension: '{file_extension}'")
    print(f'First file: {file_list[0].name}, Last file: {file_list[-1].name}')

    input_list = condor_scripts_dir / f'input_list.txt'
    with open(input_list, 'w') as f:
        for file_path in file_list:
            try:
                physical_idx = int(file_path.name.split('.')[0].split('_')[1])
            except (IndexError, ValueError):
                print(f"Warning: Could not parse index from {file_path.name}, skipping.")
                continue

            f.write(f"{physical_idx} {file_path.name}\n")

    bash_script = load_bash_template(input_path)

    with open(condor_scripts_dir / 'run_decode.sh', 'w') as bashfile:
        bashfile.write(bash_script)

    jdl_script = load_jdl_template(
        log_dir=log_dir,
        eos_base=eos_base_dir,
        out_dir=args.output,
        scripts_dir=condor_scripts_dir
    )

    with open(condor_scripts_dir / 'condor_decoding.jdl', 'w') as jdlfile:
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
        print(f"[Dry Run] JDL: {condor_scripts_dir / 'condor_decoding.jdl'}")
        print(f"[Dry Run] Bash: {condor_scripts_dir / 'run_decode.sh'}")
        print(f"[Dry Run] Input: {condor_scripts_dir / 'input_list.txt'}")

    else:
        subprocess.run(['condor_submit', f'{condor_scripts_dir}/condor_decoding.jdl'])