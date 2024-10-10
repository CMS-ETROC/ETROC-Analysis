from pathlib import Path
from natsort import natsorted
from tqdm import tqdm

import pandas as pd
import argparse

parser = argparse.ArgumentParser(
            prog='PlaceHolder',
            description='Merge feather files',
        )

parser.add_argument(
    '-d',
    '--input_dir',
    metavar = 'NAME',
    type = str,
    help = 'input directory containing .feather',
    required = True,
    dest = 'input_dir',
)

parser.add_argument(
    '-n',
    '--number_of_merge',
    metavar = 'NUM',
    type = int,
    help = 'Decide how many files will be used per merge',
    default = 10,
    dest = 'number_of_merge',
)

args = parser.parse_args()

files = natsorted(list(Path(args.input_dir).glob('loop*feather')))
merge_size = args.number_of_merge

base_dir = Path('./')
outdir = base_dir / f'output_merged_{args.input_dir}'
outdir.mkdir(exist_ok=True)

groups = []

print(f'Number of input files: {len(files)}')
print(f'How many output files: Either {len(files)//merge_size} or {len(files)//merge_size + 1}')

for i in range(0, len(files), merge_size):

    chunk = files[i:i + merge_size]

    if len(chunk) == merge_size:
        groups.append(chunk)
    elif len(chunk) > 4:
        groups.append(chunk)
    elif len(chunk) <= 4:
        groups[-1].extend(chunk)

for idx, igroup in enumerate(tqdm(groups)):

    nevt_adder = 0
    dfs = []
    for ifile in igroup:
        df = pd.read_feather(ifile)
        df['evt'] += nevt_adder
        nevt_adder += df['evt'].nunique()
        dfs.append(df)

    final_df = pd.concat(dfs).reset_index(drop=True)
    final_df.to_feather(outdir / f'loop_{idx}.feather')

print('=================================')
print('Now remove 30s loop files')

for jfile in files:
    jfile.unlink(missing_ok=True)

print('Now move merged files')
new_files = outdir.glob('loop*feather')
for kfile in new_files:
    kfile.rename(f'{args.input_dir}/{kfile.name}')
print('=================================')

## Remove temp output directory
outdir.rmdir()
