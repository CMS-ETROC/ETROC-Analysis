import pandas as pd
import numpy as np
import argparse, sys
from tabulate import tabulate

parser = argparse.ArgumentParser(
            prog='Reduce number of track candidates',
            description='Reduce number of track candidates!',
        )

parser.add_argument(
    '-f',
    '--file',
    metavar = 'FILE',
    type = str,
    help = 'Track combination table as csv format for input',
    required = True,
    dest = 'file',
)

parser.add_argument(
    '-m',
    '--minimum',
    metavar = 'NUM',
    type = int,
    help = 'Minimum threshold for track candidates',
    default = 1000,
    dest = 'minimum_ntracks',
)

parser.add_argument(
    '--ntrk_table',
    action = 'store_true',
    help = 'If set, print the cut table based on the given cut, range is fixed from 40 to 400 w/ step = 40',
    dest = 'ntrk_table',
)

args = parser.parse_args()

track_output_df = pd.read_csv(f'{args.file}')
previous_num = track_output_df.shape[0]


if args.ntrk_table:
    # SMART STEP 1: Determine cuts based on data distribution (deciles)
    # This prevents the table from being empty or uselessly small
    data_counts = track_output_df['count']
    cuts = np.percentile(data_counts, [10, 20, 30, 40, 50, 60, 70, 75, 80, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99])
    cuts = sorted(list(set(cuts.astype(int))))

    table_data = []
    for icut in cuts:
        survived = (data_counts > icut).sum()
        percent = (survived / previous_num) * 100

        table_data.append([f"> {icut}", f"{survived:,}", f"{percent:.1f}%"])

    print('\n=== Track Candidate Reduction Analysis ===')
    print(tabulate(table_data,
                   headers=['nTrk Cut', 'Survived', '% of Total'],
                   tablefmt='simple'))

    print('=== Track Candidate Reduction Analysis ===\n')
    sys.exit(0)

track_output_df = track_output_df.loc[track_output_df['count'] > args.minimum_ntracks]
track_output_df.reset_index(drop=True, inplace=True)

prefix = args.file.split('_tracks')[0] ## prefix will not be changed
suffix = '_tracks_reduced'
track_output_df.to_csv(f'{prefix}{suffix}.csv', index=False)

print(f'New file {prefix}{suffix}.csv is created')
print(f'Number of track combinations has been decreased from {previous_num} to {track_output_df.shape[0]}')
