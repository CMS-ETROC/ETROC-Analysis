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

if len(files) == 0:
    print('No input file')
    exit()

nickname_dict = {
    't': 'trig',
    'd': 'dut',
    'r': 'ref',
    'e': 'extra',
}

final_dict = defaultdict(list)

def process_file(ifile):
    # Define the pattern to match "RxCx" part
    # - (\w): Captures the nickname (e.g., 'r')
    # - R(\d+): Captures the row number
    # - C(\d+): Captures the column number
    pattern = re.compile(r"(\w)-R(\d+)C(\d+)")

    # Find all occurrences of the pattern in the string
    matches = re.findall(pattern, str(ifile))

    file_dict = defaultdict(list)
    for nickname, row, col in matches:
        # Use the reverse maps to find the original data
        full_role = nickname_dict[nickname]

        file_dict[f'row_{full_role}'].append(row)
        file_dict[f'col_{full_role}'].append(col)

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
cuts = [1, *cuts]

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
