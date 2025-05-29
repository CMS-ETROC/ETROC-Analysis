import pandas as pd
import argparse
import re
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
    "--ids",
    type=int,
    nargs='+',
    default=[0, 1, 2, 3],
    help="A list of board IDs (default: [0, 1, 2, 3]). How to use: --ids 0 1 2",
    dest = 'ids'
)

args = parser.parse_args()

input_dir = Path(args.inputdir)
files = natsorted(list(input_dir.glob('track*pkl')))

if len(files) == 0:
    import sys
    print('No input file')
    sys.exit(1)

final_dict = defaultdict(list)

def process_file(ifile):
    # Define the pattern to match "RxCx" part
    pattern = r'R(\d+)C(\d+)'

    # Find all occurrences of the pattern in the string
    matches = re.findall(pattern, str(ifile))

    if len(args.ids) != len(matches):
        print('Please check given inputs')
        print(f'Given board IDs: {args.ids}')
        print(f'Matches: {matches}')
        import sys
        sys.exit()

    file_dict = defaultdict(list)
    for id, pixel in zip(args.ids, matches):
        file_dict[f'row{id}'].append(pixel[0])
        file_dict[f'col{id}'].append(pixel[1])

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
track_nevt_df.to_csv(f'{args.outputdir}_nevt_per_track.csv', index=False)

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