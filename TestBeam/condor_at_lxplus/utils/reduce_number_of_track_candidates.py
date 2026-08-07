import pandas as pd
import numpy as np
import argparse, sys
from pathlib import Path
from tabulate import tabulate

parser = argparse.ArgumentParser(
            prog='Reduce number of track candidates',
            description='Reduce number of track candidates!',
        )

parser.add_argument(
    '-f',
    '--file',
    metavar = 'FILE_OR_DIR',
    type = str,
    help = 'Track combination table as Parquet format for input, or a directory of them '
           '(path_finder.py writes one file per board combo into a per-run directory -- '
           'point -f at that directory to reduce every combo in one call)',
    required = True,
    dest = 'file',
)

parser.add_argument(
    '-p',
    '--percentile',
    metavar = 'PCT',
    type = float,
    help = 'Keep only the top PCT%% of track candidates by occurrence count '
           '(e.g. 10 keeps candidates above the 90th percentile). If omitted, '
           'only the reduction-analysis table is printed -- no reduced file is written.',
    default = None,
    dest = 'top_percentile',
)

args = parser.parse_args()

input_path = Path(args.file)
if input_path.is_dir():
    # Skip already-reduced outputs so re-running against the same directory
    # doesn't try to reduce a reduced file again.
    input_files = sorted(f for f in input_path.glob('*.parquet') if not f.stem.endswith('_reduced'))
    if not input_files:
        sys.exit(f'No .parquet files found in directory {input_path}')
else:
    input_files = [input_path]


def build_tail_percentiles(n):
    """Keeps halving the gap to the 100th percentile (95, 97.5, 98.75, ...)
    until less than one row would separate it from the next step, so the
    finest cut shown always reaches close to a single surviving candidate --
    deeper for a file with more rows, shallower for a smaller one -- instead
    of a fixed cutoff like 99.99 that's meaningless for a small file and not
    fine-grained enough for a huge one."""
    tail = []
    gap = 10.0
    for _ in range(20):
        pct = 100 - gap / 2
        gap /= 2
        if n * (gap / 100) < 1:
            break
        tail.append(pct)
    return tail


def print_ntrk_table(label, data_counts, previous_num):
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90] + build_tail_percentiles(previous_num)

    table_data = []
    prev_min_count = None
    for pct in percentiles:
        min_count = int(round(np.percentile(data_counts, pct)))
        if min_count == prev_min_count:
            # Data is discrete -- nearby percentiles often round to the same
            # integer cut. Skip the repeat instead of printing a duplicate row.
            continue
        prev_min_count = min_count
        survived = (data_counts > min_count).sum()
        percent = (survived / previous_num) * 100
        table_data.append([f"{pct:g}th", f"{min_count}", f"{survived:,}", f"{percent:.2f}%"])

    print(f'\n=== Track Candidate Reduction Analysis: {label} ===')
    print(tabulate(table_data,
                   headers=['Percentile', 'Min nTrk Count', 'Survived', '% of Total'],
                   tablefmt='simple'))
    print(f'=== Track Candidate Reduction Analysis: {label} ===\n')


for file in input_files:
    track_output_df = pd.read_parquet(file)
    previous_num = track_output_df.shape[0]

    if args.top_percentile is None:
        print_ntrk_table(file.name, track_output_df['count'], previous_num)
        continue

    cut = int(round(np.percentile(track_output_df['count'], 100 - args.top_percentile)))
    track_output_df = track_output_df.loc[track_output_df['count'] > cut]
    track_output_df.reset_index(drop=True, inplace=True)

    ## Use the full stem (not just a '_tracks'-split prefix) so each board combo's
    ## file (e.g. tracks_0-1-2-3.parquet vs tracks_0-1-2.parquet) gets its own
    ## reduced output instead of colliding on the same name.
    output_file = file.with_name(f'{file.stem}_reduced.parquet')
    track_output_df.to_parquet(output_file, index=False)

    print(f'New file {output_file} is created (kept top {args.top_percentile}%, count > {cut})')
    print(f'Number of track combinations has been decreased from {previous_num} to {track_output_df.shape[0]}')
