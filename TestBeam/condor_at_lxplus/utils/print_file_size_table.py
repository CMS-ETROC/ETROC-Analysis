#!/usr/bin/env python

from natsort import natsorted
from tabulate import tabulate
from pathlib import Path
from tqdm import tqdm
import argparse
import pandas as pd

parser = argparse.ArgumentParser(
        prog='PlaceHolder',
        description='Print number of events, rows of each dataframe',
    )

parser.add_argument(
    '-d',
    '--inputdir',
    metavar = 'DIRNAME',
    type = str,
    help = 'input directory name',
    required = True,
    dest = 'input_dir',
)

args = parser.parse_args()

columns_to_read = ['evt']
files = natsorted(Path(args.input_dir).glob('loop*feather'))
outputs = {}

total_nevt = 0
total_nhits = 0
for ifile in tqdm(files):
    tmp_df = pd.read_feather(ifile, columns=columns_to_read)
    nevt = tmp_df['evt'].nunique()
    nrows = tmp_df.shape[0]

    outputs[ifile.name] = [nevt, nrows]
    total_nevt += nevt
    total_nhits += nrows

outputs['total'] = [total_nevt, total_nhits]
table_data = [[x, outputs[x][0], outputs[x][1]] for x in outputs.keys()]
print(tabulate(table_data, headers=['File name', 'Number of events', 'Number of hits']))
