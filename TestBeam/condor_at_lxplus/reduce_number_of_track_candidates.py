import pandas as pd
import argparse, sys

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
    from tabulate import tabulate
    import sys

    cuts = range(40, 600, 40)
    ntrk_survived = []
    cut_name = [f'ntrk > {jcut}' for jcut in cuts]

    for icut in cuts:
        tmp_df = track_output_df.loc[track_output_df['count'] > icut]
        ntrk_survived.append(tmp_df.shape[0])
    del tmp_df

    table_data = list(zip(cut_name, ntrk_survived))
    print('\n================================================================\n')
    print(tabulate(table_data, headers=['nTrk Cut', 'Number of survived track candidates']))
    print('\n================================================================')
    sys.exit(1)

track_output_df = track_output_df.loc[track_output_df['count'] > args.minimum_ntracks]
track_output_df.reset_index(drop=True, inplace=True)

prefix = args.file.split('_tracks')[0] ## prefix will not be changed
suffix = '_tracks_reduced'
track_output_df.to_csv(f'{prefix}{suffix}.csv', index=False)

print(f'New file {prefix}{suffix}.csv is created')
print(f'Number of track combinations has been decreased from {previous_num} to {track_output_df.shape[0]}')