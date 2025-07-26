import pandas as pd
import argparse
import re
from yaml import safe_load
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from natsort import natsorted
from collections import defaultdict
from tqdm import tqdm
from tabulate import tabulate

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
    dest = 'inputdir',
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
    '-c',
    '--config',
    metavar = 'NAME',
    type = str,
    help = 'YAML file including run information.',
    required = True,
    dest = 'config',
)

parser.add_argument(
    '-r',
    '--runName',
    metavar = 'NAME',
    type = str,
    help = 'Name of the run to process. It must be matched with the name defined in YAML.',
    required = True,
    dest = 'runName',
)

parser.add_argument(
    '--tag',
    metavar = 'NAME',
    type = str,
    help = 'Tag for the output file name.',
    default = '',
    dest = 'tag',
)

args = parser.parse_args()

input_dir = Path(args.inputdir)
files = natsorted(list(input_dir.glob('excluded*track*pkl')))

with open(args.config) as input_yaml:
    config = safe_load(input_yaml)
selected_config = config[args.runName]

if len(files) == 0:
    print('No input file')
    exit()

final_dict = defaultdict(list)

def process_file(ifile):
    # Define the pattern to match "RxCx" part
    pattern = r'R(\d+)C(\d+)'

    # Find all occurrences of the pattern in the string
    matches = re.findall(pattern, str(ifile))

    if len(selected_config) != len(matches):
        print('Please check given inputs')
        print(f'Matches: {matches}')
        exit()

    file_dict = defaultdict(list)

    for board_id, board_info in selected_config.items():
        role = board_info.get('role')
        file_dict[f'row_{role}'].append(matches[board_id][0])
        file_dict[f'col_{role}'].append(matches[board_id][1])

    tmp_df = pd.read_pickle(ifile)
    file_dict['nevt'].append(tmp_df.shape[0])

    del tmp_df
    return file_dict

# Process files in parallel
with tqdm(files) as pbar:
    with ProcessPoolExecutor() as executor:
        future_to_file = [executor.submit(process_file, ifile) for ifile in files]

    for future in as_completed(future_to_file):
        pbar.update(1)
        result = future.result()
        for key, value in result.items():
            final_dict[key].extend(value)

track_nevt_df = pd.DataFrame(data=final_dict)
track_nevt_df.sort_values(by=['nevt'], ascending=False, inplace=True)
track_nevt_df.to_csv(f'{args.outputdir}_nevt_per_track{args.tag}.csv', index=False)

cuts = range(100, 1600, 100)
ntrk_survived = []
cut_name = [f'ntrk > {jcut}' for jcut in cuts]

for icut in cuts:
    tmp_df = track_nevt_df.loc[track_nevt_df['nevt'] > icut]
    ntrk_survived.append(tmp_df.shape[0])
del tmp_df

table_data = list(zip(cut_name, ntrk_survived))
print('\n================================================================\n')
print(tabulate(table_data, headers=['nTrk Cut', 'Number of survived track candidates for bootstrap']))
print('\n================================================================')
