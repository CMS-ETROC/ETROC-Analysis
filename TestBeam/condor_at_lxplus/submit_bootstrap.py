import os
from pathlib import Path
import argparse
from glob import glob
from jinja2 import Template

parser = argparse.ArgumentParser(
            prog='PlaceHolder',
            description='offline translate script',
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
    '-o',
    '--outputdir',
    metavar = 'DIRNAME',
    type = str,
    help = 'output directory name',
    required = True,
    dest = 'outputdir',
)

parser.add_argument(
    '-n',
    '--num_bootstrap_output',
    metavar = 'NUM',
    type = int,
    help = 'Number of outputs after bootstrap',
    default = 100,
    dest = 'num_bootstrap_output',
)

parser.add_argument(
    '-s',
    '--sampling',
    metavar = 'SAMPLING',
    type = int,
    help = 'Random sampling fraction',
    default = 75,
    dest = 'sampling',
)

parser.add_argument(
    '--board_id_for_TOA_cut',
    metavar = 'NUM',
    type = int,
    help = 'TOA range cut will be applied to a given board ID',
    default = 1,
    dest = 'board_id_for_TOA_cut',
)

parser.add_argument(
    '--minimum_nevt',
    metavar = 'NUM',
    type = int,
    help = 'Minimum number of events for bootstrap',
    default = 1000,
    dest = 'minimum_nevt',
)

parser.add_argument(
    '--iteration_limit',
    metavar = 'NUM',
    type = int,
    help = 'Maximum iteration of sampling',
    default = 7500,
    dest = 'iteration_limit',
)

parser.add_argument(
    '--board_ids',
    metavar='N',
    type=int,
    nargs='+',
    help='board IDs to analyze'
)

parser.add_argument(
    '--trigTOALower',
    metavar = 'NUM',
    type = int,
    help = 'Lower TOA selection boundary for the trigger board',
    default = 100,
    dest = 'trigTOALower',
)

parser.add_argument(
    '--trigTOAUpper',
    metavar = 'NUM',
    type = int,
    help = 'Upper TOA selection boundary for the trigger board',
    default = 500,
    dest = 'trigTOAUpper',
)

parser.add_argument(
    '--board_id_rfsel0',
    metavar = 'NUM',
    type = int,
    help = 'board ID that set to RfSel = 0',
    dest = 'board_id_rfsel0',
)

parser.add_argument(
    '--autoTOTcuts',
    action = 'store_true',
    help = 'If set, select 80 percent of data around TOT median value of each board',
    dest = 'autoTOTcuts',
)

parser.add_argument(
    '--reproducible',
    action = 'store_true',
    help = 'If set, random seed will be set by counter and save random seed in the final output',
    dest = 'reproducible',
)

parser.add_argument(
    '--time_df_input',
    action = 'store_true',
    help = 'If set, time_df_bootstrap function will be used',
    dest = 'time_df_input',
)

parser.add_argument(
    '--dryrun',
    action = 'store_true',
    help = 'If set, condor submission will not happen',
    dest = 'dryrun',
)

args = parser.parse_args()
current_dir = Path('./')

files = glob(f'{args.dirname}/*pkl')
listfile = current_dir / 'input_list_for_bootstrap.txt'
if listfile.is_file():
    listfile.unlink()

with open(listfile, 'a') as listfile:
    for ifile in files:
        name = ifile.split('/')[-1].split('.')[0]
        save_string = f"{name}, {ifile}"
        listfile.write(save_string + '\n')

outdir = current_dir / f'resolution_{args.outputdir}'
outdir.mkdir(exist_ok = False)

#### Make python command
bash_command = "python bootstrap.py -f {{ filename }} -n {{ num_bootstrap_output }} -s {{ sampling }} \
--board_id_for_TOA_cut {{ board_id_for_TOA_cut }} --minimum_nevt {{ minimum_nevt }} --iteration_limit {{ iteration_limit }} \
--trigTOALower {{ trigTOALower }} --trigTOAUpper {{ trigTOAUpper }} --board_ids {{ board_ids }}"

conditional_args = {
    'autoTOTcuts': args.autoTOTcuts,
    'reproducible': args.reproducible,
    'time_df_input': args.time_df_input,
}

for arg, value in conditional_args.items():
    if value:
        bash_command += f" --{arg}"  # Add the argument if value is True

conditional_input_args = {
    'board_id_rfsel0': args.board_id_rfsel0,
}

for arg, value in conditional_input_args.items():
    if value:
        bash_command += f" --{arg} {value}"  # Add the argument if value is True

# Define the bash script template
bash_template = """#!/bin/bash

ls -ltrh
echo ""
pwd

# Load python environment from work node
source /cvmfs/sft.cern.ch/lcg/views/LCG_104a/x86_64-el9-gcc13-opt/setup.sh

echo "{0}"
{0}
""".format(bash_command)

# Prepare the data for the template
options = {
    'filename': '${1}.pkl',
    'num_bootstrap_output': args.num_bootstrap_output,
    'sampling': args.sampling,
    'board_id_for_TOA_cut': args.board_id_for_TOA_cut,
    'minimum_nevt': args.minimum_nevt,
    'iteration_limit': args.iteration_limit,
    'trigTOALower': args.trigTOALower,
    'trigTOAUpper': args.trigTOAUpper,
    'board_ids': ' '.join(map(str, args.board_ids))
}

# Render the template with the data
bash_script = Template(bash_template).render(options)

print('\n========= Run option =========')
print(f'Input dataset: {args.dirname}')
print(f'Output direcotry: resolution_{args.outputdir}')
print(f'Bootstrap iteration limit: {args.iteration_limit}')
print(f'Number of bootstrap outputs: {args.num_bootstrap_output}')
print(f'{args.sampling}% of random sampling')
print(f'Consider board IDs: {args.board_ids}')
print(f"TOA cut for a 'NEW' trigger is {args.trigTOALower}-{args.trigTOAUpper} on board ID={args.board_id_for_TOA_cut}")
print(f'Number of events larger than {args.minimum_nevt} will be considered')
if args.autoTOTcuts:
    print(f'Automatic TOT cuts will be applied')
if args.reproducible:
    print('Random seed will be set by counter. The final output will have seed information together')
print('========= Run option =========\n')

with open('run_bootstrap.sh','w') as bashfile:
    bashfile.write(bash_script)

log_dir = current_dir / 'condor_logs'
log_dir.mkdir(exist_ok=True)

if log_dir.exists():
    os.system('rm condor_logs/*bootstrap*log')
    os.system('rm condor_logs/*bootstrap*stdout')
    os.system('rm condor_logs/*bootstrap*stderr')
    os.system('ls condor_logs/*bootstrap*log | wc -l')

jdl = """universe              = vanilla
executable            = run_bootstrap.sh
should_Transfer_Files = YES
whenToTransferOutput  = ON_EXIT
arguments             = $(ifile)
transfer_Input_Files  = bootstrap.py,$(path)
TransferOutputRemaps = "$(ifile)_resolution.pkl={1}/$(ifile)_resolution.pkl"
output                = {0}/$(ClusterId).$(ProcId).bootstrap.stdout
error                 = {0}/$(ClusterId).$(ProcId).bootstrap.stderr
log                   = {0}/$(ClusterId).$(ProcId).bootstrap.log
MY.WantOS             = "el9"
+JobFlavour           = "microcentury"
Queue ifile,path from input_list_for_bootstrap.txt
""".format(str(log_dir), str(outdir))

with open(f'condor_bootstrap.jdl','w') as jdlfile:
    jdlfile.write(jdl)

if args.dryrun:
    print('=========== Input text file ===========')
    os.system('cat input_list_for_bootstrap.txt')
    print()
    print('=========== Bash file ===========')
    os.system('cat run_bootstrap.sh')
else:
    os.system(f'condor_submit condor_bootstrap.jdl')
