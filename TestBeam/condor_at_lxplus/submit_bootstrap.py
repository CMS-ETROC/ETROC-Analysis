from pathlib import Path
from jinja2 import Template
from natsort import natsorted

def load_bash_script(args):

    #### Make python command
    bash_command = "python bootstrap.py -f {{ filename }} -n {{ num_bootstrap_output }} -s {{ sampling }} \
    --minimum_nevt {{ minimum_nevt }} --iteration_limit {{ iteration_limit }}"

    conditional_args = {
        'reproducible': args.reproducible,
        'twc_coeffs': args.twc_coeffs,
        'force-twc': args.force_twc,
    }

    for arg, value in conditional_args.items():
        # Skip arguments that are not set (i.e., value is False or None)
        if not value:
            continue

        # If the value is a boolean (and we already know it's True), just add the flag.
        if isinstance(value, bool):
            bash_command += f" --{arg}"
        # Otherwise, add the flag and its value.
        else:
            bash_command += f" --{arg} {value}"

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
        'minimum_nevt': args.minimum_nevt,
        'iteration_limit': args.iteration_limit,
    }

    # Render the template with the data
    return Template(bash_template).render(options)


def load_jdl_script(args, log_dir, condor_scripts_dir, outdir, runAppend):

    # Start with the base file and add twc_coeffs if it exists
    transfer_files = "bootstrap.py"
    if args.twc_coeffs:
        transfer_files += f", {args.twc_coeffs}"

    jdl = """universe              = vanilla
executable            = {2}/run_bootstrap{3}.sh
should_Transfer_Files = YES
whenToTransferOutput  = ON_EXIT
arguments             = $(ifile) $(path)
transfer_Input_Files  = {4}
TransferOutputRemaps = "$(ifile)_resolution.pkl={1}/$(ifile)_resolution.pkl; $(ifile)_gmmInfo.pkl={1}/$(ifile)_gmmInfo.pkl"
output                = {0}/$(ClusterId).$(ProcId).bootstrap.stdout
error                 = {0}/$(ClusterId).$(ProcId).bootstrap.stderr
log                   = {0}/$(ClusterId).$(ProcId).bootstrap.log
MY.WantOS             = "el9"
+JobFlavour           = "workday"
Queue ifile,path from {2}/input_list_for_bootstrap{3}.txt
""".format(log_dir, outdir, condor_scripts_dir, runAppend, transfer_files)

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

    jdl_templalte = load_jdl_script(args, log_dir, condor_scripts_dir, outdir, runAppend)
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
        '--condor_tag',
        metavar = 'NAME',
        type = str,
        help = 'Tag of the run to process on condor. If given, the tag will be used to avoid file collisions',
        dest = 'condor_tag',
    )

    parser.add_argument(
        '--twc_coeffs',
        metavar = 'FILE',
        type = str,
        help = 'pre-calculated TWC coefficients, it has to be pickle file',
        dest = 'twc_coeffs',
    )

    parser.add_argument(
        '--reproducible',
        action = 'store_true',
        help = 'If set, random seed will be set by counter and save random seed in the final output',
        dest = 'reproducible',
    )

    parser.add_argument(
        '--force-twc',
        action='store_true',
        help='Force use of provided TWC file for all samples.',
        dest='force_twc'
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

    parser.add_argument(
        '--resubmit_with_stderr',
        action = 'store_true',
        help = 'If set, condor resubmission for jobs when stderr is not empty.',
        dest = 'resubmit_with_stderr',
    )

    args = parser.parse_args()

    tag_for_condor = args.condor_tag
    if tag_for_condor is None:
        runAppend = ""
    else:
        runAppend = "_" + tag_for_condor

    condor_scripts_dir = Path('./') / 'condor_scripts' / f'bootstrap_job{runAppend}'
    condor_scripts_dir.mkdir(exist_ok=True, parents=True)

    log_dir = Path('./') / 'condor_logs' / 'bootstrap' / f'bootstrap_job{runAppend}'
    log_dir.mkdir(exist_ok=True, parents=True)

    outdir = Path('./') / f'bootstrap_{args.outputdir}'
    if not args.dryrun and not args.resubmit and not args.resubmit_with_stderr:
        outdir.mkdir(exist_ok = False)

    print('\n========= Run option =========')
    print(f'Input dataset: {args.dirname}')
    print(f'Output direcotry: resolution_{args.outputdir}')
    print(f'Bootstrap iteration limit: {args.iteration_limit}')
    print(f'Number of bootstrap outputs: {args.num_bootstrap_output}')
    print(f'{args.sampling}% of random sampling')
    print(f'Number of events larger than {args.minimum_nevt} will be considered')
    if args.reproducible:
        print('Random seed will be set by counter. The final output will have a seed information together')
    if args.twc_coeffs is not None:
        print('Pre-calculated TWC coeffs will be used')
    print('========= Run option =========\n')

    make_jobs(args=args, log_dir=log_dir, condor_scripts_dir=condor_scripts_dir, outdir=outdir, runAppend=runAppend)
    input_txt_path = condor_scripts_dir / f"input_list_for_bootstrap{runAppend}.txt"

    if args.dryrun:

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
        condor_output = subprocess.run(['condor_q', '-nobatch'], capture_output=True, text=True)

        # Filter lines containing the target script
        filtered_lines = [line for line in condor_output.stdout.splitlines() if f'run_bootstrap{runAppend}.sh' in line]

        if len(filtered_lines) == 0:
            print('No condor job found.')
            import sys
            sys.exit(1)

        with open(input_txt_path, 'w') as f:
            for line in filtered_lines:
                fields = line.split()
                if len(fields) == 11:
                    if fields[5] == 'X':
                        old_condor_job_id = -1
                        continue
                    old_condor_job_id = fields[0].split('.')[0]
                    last_three = fields[-2:]
                    sentence = ' '.join(last_three)
                    f.write(sentence + '\n')

        if not old_condor_job_id == -1:
            ## Kill old jobs before submit new one
            subprocess.run(['condor_rm', f'{old_condor_job_id}'])

            if input_txt_path.stat().st_size > 0:
                subprocess.run(['condor_submit', f'{condor_scripts_dir}/condor_bootstrap{runAppend}.jdl'])
            else:
                print("No jobs to submit — input list is empty.")

    elif args.resubmit_with_stderr:
        condor_output = subprocess.run(['condor_q', '-nobatch'], capture_output=True, text=True)
        filtered_lines = [line for line in condor_output.stdout.splitlines() if f'run_bootstrap{runAppend}.sh' in line]

        if filtered_lines:
            print('Found running jobs. This option is only allowed when no job with the same condor tag is running on condor')
            exit()

        print('No running jobs found. Checking stderr files...')
        stderr_files = log_dir.glob('*.stderr')

        line_numbers = []
        for ifile in stderr_files:
            if ifile.stat().st_size > 0:
                line_numbers.append(ifile.name.split('.')[1])

        with open(input_txt_path, 'r') as f:
            all_lines = f.readlines()

        selected_data = []
        for line_num in line_numbers:
            data = all_lines[int(line_num)].strip()
            selected_data.append(data)
        del all_lines

        print(f'Found {len(selected_data)} jobs to resubmit')
        with open(input_txt_path, 'w') as f:
            for item in selected_data:
                f.write(item + '\n')

        if input_txt_path.stat().st_size > 0:
            subprocess.run(['condor_submit', f'{condor_scripts_dir}/condor_bootstrap{runAppend}.jdl'])

    else:
        subprocess.run(['condor_submit', f'{condor_scripts_dir}/condor_bootstrap{runAppend}.jdl'])
