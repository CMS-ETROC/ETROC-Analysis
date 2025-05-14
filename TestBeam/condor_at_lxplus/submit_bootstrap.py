from pathlib import Path
from jinja2 import Template
from natsort import natsorted


def load_bash_script(args):

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

xrdcp root://eosuser.cern.ch/{1} ./

# Load python environment from work node
source /cvmfs/sft.cern.ch/lcg/views/LCG_104a/x86_64-el9-gcc13-opt/setup.sh

echo "{0}"
{0}

# Delete input file so condor will not return
rm {2}

ls -ltrh
echo ""

""".format(bash_command, '${2}', '${1}.pkl')

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
    return Template(bash_template).render(options)


def load_jdl_script(log_dir, condor_scripts_dir, outdir, runAppend):
    jdl = """universe              = vanilla
executable            = {2}/run_bootstrap{3}.sh
should_Transfer_Files = YES
whenToTransferOutput  = ON_EXIT
arguments             = $(ifile) $(path)
transfer_Input_Files  = bootstrap.py
TransferOutputRemaps = "$(ifile)_resolution.pkl={1}/$(ifile)_resolution.pkl"
output                = {0}/$(ClusterId).$(ProcId).bootstrap.stdout
error                 = {0}/$(ClusterId).$(ProcId).bootstrap.stderr
log                   = {0}/$(ClusterId).$(ProcId).bootstrap.log
MY.WantOS             = "el9"
+JobFlavour           = "microcentury"
Queue ifile,path from {2}/input_list_for_bootstrap{3}.txt
""".format(log_dir, outdir, condor_scripts_dir, runAppend)

    return jdl

def make_jobs(args, log_dir, condor_scripts_dir, outdir, runAppend):

    files = natsorted(Path(args.dirname).glob('*.pkl'))
    listfile = condor_scripts_dir / f'input_list_for_bootstrap{runAppend}.txt'
    if listfile.is_file():
        listfile.unlink()

    with open(listfile, 'a') as base_txt:
        for ifile in files:
            name = ifile.name.split('.')[0]
            save_string = f"{name}, {ifile}"
            base_txt.write(save_string + '\n')

    bash_template = load_bash_script(args)
    with open(condor_scripts_dir / f'run_bootstrap{runAppend}.sh','w') as bashfile:
        bashfile.write(bash_template)

    jdl_templalte = load_jdl_script(log_dir, condor_scripts_dir, outdir, runAppend)
    with open(condor_scripts_dir / f'condor_bootstrap{runAppend}.jdl','w') as jdlfile:
        jdlfile.write(jdl_templalte)

## --------------------------------------
if __name__ == "__main__":

    import subprocess
    import argparse

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
        '-r',
        '--runName',
        metavar = 'NAME',
        type = str,
        help = 'Name of the run to process. If given, the run name will be used to avoid file collisions',
        dest = 'runName',
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

    condor_scripts_dir = Path('./') / 'condor_scripts' / f'bootstrap_job{runAppend}'
    condor_scripts_dir.mkdir(exist_ok=True, parents=True)

    log_dir = Path('./') / 'condor_logs' / 'bootstrap' / f'bootstrap_job{runAppend}'
    log_dir.mkdir(exist_ok=True, parents=True)

    outdir = Path('./') / f'bootstrap_{args.outputdir}'
    if not args.dryrun and not args.resubmit:
        outdir.mkdir(exist_ok = False)

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

    make_jobs(args=args, log_dir=log_dir, condor_scripts_dir=condor_scripts_dir, outdir=outdir, runAppend=runAppend)

    if args.dryrun:
        input_txt_path = condor_scripts_dir / f"input_list_for_bootstrap{runAppend}.txt"
        print('\n=========== Input text file ===========')
        subprocess.run(f"head -n 10 {input_txt_path}", shell=True)
        subprocess.run(f"tail -n 10 {input_txt_path}", shell=True)
        print()
        print('=========== Bash file ===========')
        with open(condor_scripts_dir / f"run_bootstrap{runAppend}.sh") as f:
            print(f.read(), '\n')
        print('=========== Condor Job Description file ===========')
        with open(condor_scripts_dir / f'condor_bootstrap{runAppend}.jdl') as f:
            print(f.read(), '\n')
        print()

    elif args.resubmit:
        input_txt_path = condor_scripts_dir / f"input_list_for_bootstrap{runAppend}.txt"
        condor_output = subprocess.run(['condor_q', '-nobatch'], capture_output=True, text=True)

        # Filter lines containing the target script
        filtered_lines = [line for line in condor_output.stdout.splitlines() if f'run_bootstrap{runAppend}.sh' in line]
        with open(input_txt_path, 'w') as f:
            for line in filtered_lines:
                fields = line.split()
                if len(fields) == 11:
                    old_condor_job_id = fields[0].split('.')[0]
                    last_three = fields[-2:]
                    sentence = ' '.join(last_three)
                    f.write(sentence + '\n')

        ## Kill old jobs before submit new one
        subprocess.run(['condor_rm', f'{old_condor_job_id}'])

        if input_txt_path.stat().st_size > 0:
            subprocess.run(['condor_submit', f'{condor_scripts_dir}/condor_bootstrap{runAppend}.jdl'])
        else:
            print("No jobs to submit — input list is empty.")

    else:
        subprocess.run(['condor_submit', f'{condor_scripts_dir}/condor_bootstrap{runAppend}.jdl'])