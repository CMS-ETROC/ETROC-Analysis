import getpass, subprocess
from pathlib import Path
import argparse
from glob import glob
from natsort import natsorted
from math import ceil
from beamtest_analysis_helper import toSingleDataFrame_newEventModel
import tqdm

parser = argparse.ArgumentParser(
            prog='Nem to feather',
            description='convert nem runs to feather format',
        )

parser.add_argument(
    '-d',
    '--input_dir',
    metavar = 'NAME',
    type = str,
    help = 'input directory containing .nem',
    required = True,
    dest = 'input_dir',
)

parser.add_argument(
    '-o',
    '--output_dir',
    metavar = 'NAME',
    type = str,
    help = 'Output directory to save feather formatted data. The directory must not exist since it will be created',
    required = True,
    dest = 'output_dir',
)

parser.add_argument(
    '-g',
    '--group',
    type=int,
    help='Amount of nem files to group into one feather file',
    default = 100,
    dest = 'group',
)

args = parser.parse_args()

outdir = Path(f'{args.output_dir}_feather')
outdir.mkdir(exist_ok = False)

file_list = natsorted(Path(args.input_dir).glob('file*.nem'))

num_feather_files = ceil(len(file_list)/args.group)

for feather_idx in tqdm.tqdm(range(num_feather_files)):
    min_file = feather_idx*args.group
    max_file = min(len(file_list), (feather_idx+1)*args.group)
    current_files = file_list[min_file:max_file+1]

    df = toSingleDataFrame_newEventModel(files=current_files)

    df.to_feather(outdir / f'loop_{feather_idx}.feather')
