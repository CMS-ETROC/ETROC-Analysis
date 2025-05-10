import pandas as pd
import argparse

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

args = parser.parse_args()

track_output_df = pd.read_csv(f'{args.file}')
previous_num = track_output_df.shape[0]
track_output_df = track_output_df.loc[track_output_df['count'] > args.minimum_ntracks]
track_output_df.reset_index(drop=True, inplace=True)
track_output_df.to_csv(f'{args.file}', index=False)
print(f'Number of track combinations has been decreased from {previous_num} to {track_output_df.shape[0]}')