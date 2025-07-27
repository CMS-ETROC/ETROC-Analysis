from natsort import natsorted
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import yaml
import warnings
import argparse

warnings.filterwarnings("ignore")

def save_single_track(track_key, track_parts, track_dir, nickname_dict, role_by_index):
    """
    Worker function to concatenate, name, and save a single track file.
    This function will be run in parallel.
    """
    concatenated_track_df = pd.concat(track_parts, ignore_index=True)

    # Robustness Check 1: Skip empty tracks
    if concatenated_track_df.empty:
        return f"Skipped empty track: {track_key}"

    # Robustness Check 2: Protect filename generation
    try:
        board_ids_for_naming = [
            b for b in concatenated_track_df.columns.get_level_values('board').unique()
            if isinstance(b, int)
        ]
        row_cols = {
            idx: (concatenated_track_df['row'][idx].unique()[0], concatenated_track_df['col'][idx].unique()[0])
            for idx in board_ids_for_naming
        }
        outname = f"track" + ''.join([f"{nickname_dict[role_by_index[key]]}R{row}C{col}" for key, (row, col) in row_cols.items()])
        concatenated_track_df.to_pickle(track_dir / f'{outname}.pkl')
        return f"Saved track: {outname}.pkl"

    except IndexError:
        return f"Warning: Failed to name track {track_key}. Skipping."

def reshape_to_tracks(args):

    # --- Configuration Loading ---
    with open(args.config) as input_yaml:
        config = yaml.safe_load(input_yaml)

    if args.runName not in config:
        raise ValueError(f"Run config {args.runName} not found")

    id_role_map = {}
    for board_id, board_info in config[args.runName].items():
        role = board_info.get('role')
        if role:
            id_role_map[role] = board_id
            id_role_map[board_id] = role

    nickname_dict = {'trig': '_t-', 'dut': '_d-', 'ref': '_r-', 'extra': '_e-'}

    output_dir = Path(args.outdir)
    track_dir = output_dir / 'tracks'
    track_dir.mkdir(exist_ok=True, parents=True)

    # --- File Discovery ---
    print(f'\nInput path is: {args.dirname}')
    print(f'Output path is: {args.outdir}\n')

    files = []
    for pattern in args.file_pattern.split():
        files.extend(natsorted(Path(args.dirname).glob(pattern)))

    if not files:
        print(f'No input files found for the given path: {args.dirname}')
        exit()

    # --- Data Grouping (Single-threaded, memory-intensive part) ---
    print('====== Reading and grouping data by track ======')
    track_data = defaultdict(list)
    for ifile in tqdm(files, desc="Reading Files"):
        data_dict = pd.read_pickle(ifile)
        for track_key, df in data_dict.items():
            if not df.empty:
                track_data[track_key].append(df)
                ### pd.concat usually work even if input includes empty dataframe
                ### BUT!! for the case when pd.concat with MultiIndex dataframe, different style (even empty) is not allowed.

    if args.debug:
        for key, parts in  track_data.items():
            save_single_track(key, parts, track_dir, nickname_dict, id_role_map)
            break

    else:
        # --- Saving Track Files (Parallelized for speed) ---
        print('\n====== Concatenating and saving individual track files in parallel ======')
        with ProcessPoolExecutor() as executor:
            # Submit each track to be processed by the worker function
            futures = [
                executor.submit(save_single_track, key, parts, track_dir, nickname_dict, id_role_map)
                for key, parts in track_data.items()
            ]
            # Use tqdm to show progress as jobs complete
            for future in tqdm(as_completed(futures), total=len(futures), desc="Saving Tracks"):
                try:
                    # This is the crucial line. It will re-raise any exception
                    # that happened in the worker process.
                    future.result()
                except Exception as exc:
                    print(f"A worker process generated an exception: {exc}")
                    # For a full error report, uncomment the next two lines
                    # import traceback
                    # traceback.print_exc()
                finally:
                    pass

    print(f"\nDone. Track files saved in {track_dir}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            prog='reshape_to_tracks',
            description='Reads file-based data, reshapes it, and saves as track-based files.',
    )

    parser.add_argument(
        '-d',
        '--inputdir',
        metavar = 'DIRNAME',
        type = str,
        help = 'input directory name',
        required = True,
        dest = 'dirname',
    )

    parser.add_argument(
        '-o',
        '--outdir',
        metavar = 'OUTNAME',
        type = str,
        help = 'output directory name',
        required = True,
        dest = 'outdir',
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
        '-c',
        '--config',
        metavar = 'NAME',
        type = str,
        help = 'YAML file including run information.',
        required = True,
        dest = 'config',
    )

    parser.add_argument(
        '--file_pattern',
        metavar = 'glob-pattern',
        help = "Put the file pattern for glob, if you want to process part of dataset. Example: 'run*_loop_[0-9].pickle run*_loop_1[0-9].pickle run*_loop_2[0-4].pickle'",
        default = '*.pickle',
        dest = 'file_pattern',
    )

    parser.add_argument(
        '--debug',
        action = 'store_true',
        help = 'If set, switch to loop mode to print error message',
        dest = 'debug',
    )

    args = parser.parse_args()
    reshape_to_tracks(args)