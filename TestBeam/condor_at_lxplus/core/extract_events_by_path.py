import argparse
import sys
import logging
import yaml
from pathlib import Path
from functools import reduce
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

import io_utils

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')

# --- Helper Functions ---
def get_parquet_rename_map_from_config(config_path: str, run_key: str) -> Dict[str, str]:
    """
    Loads the YAML config and dynamically builds the positional index -> role mapping
    for Parquet file renaming.
    """
    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    if run_key not in config_data:
        raise ValueError(f"Run config '{run_key}' not found in {config_path}")

    positional_role_map = {}
    for board_id, board_info in config_data[run_key].items():
        role = board_info.get('role', 'unknown')
        if role != 'unknown':
            # Map old suffix '_0' to new suffix '_trig'
            positional_role_map[f'_{board_id}'] = f'_{role}'

    return positional_role_map


def find_neighbor_hits(
        input_df: pd.DataFrame,
        search_method: str,
):

    # Initialize the new column as False
    input_df['HasNeighbor'] = False

    if search_method == "none":
        return input_df
    elif search_method == 'cross':
        offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    elif search_method == 'row_only':
        offsets = [(1, 0), (-1, 0)]
    elif search_method == 'col_only':
        offsets = [(0, 1), (0, -1)]
    elif search_method == 'square':
        offsets = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    else:
        raise ValueError(f'Unknown method: {search_method}')

    # Save index to keep track of rows after merge
    input_df = input_df.reset_index()

    # 3. Iterate through offsets and merge
    for r_off, c_off in offsets:
        # Create a temporary copy of the potential neighbors
        # We shift the coordinates of the copy. If the copy matches a real row
        # after shifting, it means the real row has a neighbor at that distance.
        temp_df = input_df[['evt', 'board', 'row', 'col', 'index']].copy()

        # "Look" at the position where a neighbor would be
        temp_df['row'] = temp_df['row'] + r_off
        temp_df['col'] = temp_df['col'] + c_off

        # Merge original DF with the shifted temp DF
        # We match on Evt, bd, and the *shifted* R and C
        matches = input_df.merge(
            temp_df,
            on=['evt', 'board', 'row', 'col'],
            how='inner',
            suffixes=('', '_neighbor')
        )

        # matches['index_neighbor'] contains the indices of rows that HAVE a neighbor
        # at this specific offset.
        if not matches.empty:
            input_df.loc[input_df['index'].isin(matches['index_neighbor']), 'HasNeighbor'] = True

    # Clean up
    final_df = input_df.set_index('index').rename_axis(None)

    return final_df


def select_calibrated_hits(input_df: pd.DataFrame, cal_cuts_dict: dict) -> pd.DataFrame:
    """Keeps only events where each required board has a hit with the correct CAL
    (cal_mode +/- 3) at its candidate pixel -- events where that board's only hit
    there has the wrong CAL code are dropped entirely, rather than just that one
    row. ToT-based quality cuts are intentionally left to apply_tdc_cuts.py
    (step 10), which runs after reshaping and can compute a robust per-role ToT
    range instead of a single-file, trig-only window."""
    masks = {}
    for board, (cal_min, cal_max) in cal_cuts_dict.items():
        mask = (
            (input_df['board'] == board) &
            input_df['cal'].between(cal_min, cal_max)
        )
        masks[board] = input_df.loc[mask, 'evt'].unique()

    if not masks:
        return input_df

    common_events = reduce(np.intersect1d, list(masks.values()))
    return input_df.loc[input_df['evt'].isin(common_events)].reset_index(drop=True)

def extract_events_for_track(
    hit_index: pd.core.groupby.DataFrameGroupBy, # Optimized Input
    cal_map: Dict[Tuple[int, int, int], float],
    pix_dict: Dict[int, List[int]],
    board_ids: List[int],
) -> pd.DataFrame:
    """
    Extracts events using pre-grouped hit index (Instant Lookup).
    """
    # 1. Retrieve specific hits from the index (O(1) operation)
    candidate_hits = []
    for bid in board_ids:
        row, col = pix_dict[bid]
        key = (bid, row, col)

        # Check if this pixel even has hits in the run data
        if key in hit_index.groups:
            candidate_hits.append(hit_index.get_group(key))
        else:
            # If a pixel required by the track has 0 hits in raw data, track is invalid
            return pd.DataFrame()

    # Combine the retrieved chunks
    track_tmp_df = pd.concat(candidate_hits)

    # 2. Define CAL Cuts per board
    cal_cuts = {}
    for bid in board_ids:
        row, col = pix_dict[bid]
        cal_mode = cal_map.get((bid, row, col))

        if cal_mode is None: return pd.DataFrame()

        cal_cuts[bid] = (cal_mode - 3, cal_mode + 3)

    # 3. Apply CAL filtering
    track_tmp_df = select_calibrated_hits(track_tmp_df, cal_cuts)

    if len(track_tmp_df['board'].unique()) != len(board_ids):
        return pd.DataFrame()

    # 4. Isolation Check
    counts = track_tmp_df.groupby(['evt', 'board']).size().unstack(fill_value=0)

    req_boards_exist = [b in counts.columns for b in board_ids]
    if not all(req_boards_exist):
        return pd.DataFrame()

    valid_mask = np.logical_and.reduce([counts[b] == 1 for b in board_ids])
    valid_events = counts.index[valid_mask]

    if len(valid_events) == 0:
        return pd.DataFrame()

    # 5. Retrieve final data and pivot
    isolated_df = track_tmp_df.loc[track_tmp_df['evt'].isin(valid_events)]

    pivot_table = isolated_df.pivot(
        index="evt",
        columns="board",
        values=["row", "col", "toa", "tot", "cal", "HasNeighbor"]
    ).reset_index(drop=True)

    return pivot_table

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description='Select detailed event data based on track candidates.')
    parser.add_argument('-f', '--inputfile', required=True, dest='inputfile', help='Input feather file')
    parser.add_argument('-r', '--runinfo', required=True, dest='runinfo', help='Run info string for output name')
    parser.add_argument('-c', '--config', required=True, dest='config', help='YAML file with run config')
    parser.add_argument('-t', '--track', required=True, dest='track', help='Parquet file with track candidates for one board combo')
    parser.add_argument('--neighbor_search_method', default="none", dest='search_method',
                        help="Search method for neighbor hit checking, default is 'none'. possible argument: 'row_only', 'col_only', 'cross', 'square'")
    parser.add_argument('--cal_table', required=True, dest='cal_table', help='CSV file with CAL mode values')
    parser.add_argument('--file-index', type=int, default=None, dest='file_index',
                        help="Numeric index tagged into each output row's 'file' column. "
                             "The submit script passes this explicitly; if omitted, it falls back "
                             "to parsing it from the input filename (expects 'loop_<N>...').")
    args = parser.parse_args()

    # NEW: Dynamically load the rename map from the config file
    try:
        rename_map = get_parquet_rename_map_from_config(args.config, args.runinfo)
    except Exception as e:
        logging.error(f"Failed to load or process config file '{args.config}': {e}")
        sys.exit(1)

    # 1. Load Main Data
    try:
        run_df = pd.read_feather(args.inputfile, columns=['evt', 'board', 'row', 'col', 'toa', 'tot', 'cal'])
        # Stored as uint16 upstream; cast to signed so any cal-vs-cal_mode arithmetic
        # here can't silently wrap around like the equivalent code in path_finder.py did.
        run_df['cal'] = run_df['cal'].astype('int32')
    except Exception as e:
        logging.error(f"Failed to read input file: {e}")
        sys.exit(1)

    if run_df.empty:
        logging.error('Empty input dataframe.')
        sys.exit(0)

    # 2. Load Track Candidates
    track_df = pd.read_parquet(args.track)
    if track_df.empty:
        logging.error('Track file is empty.')
        sys.exit(0)

    # 3. Automatic Board Detection
    present_boards = sorted([i for i in [0, 1, 2, 3] if f'row_{i}' in track_df.columns])
    logging.info(f"Detected boards in track file: {present_boards}")

    if len(present_boards) < 3:
        logging.error("Fewer than 3 boards found in track file. Cannot process.")
        sys.exit(1)

    # 4. Optimization: Pre-filter Main Data
    run_df = run_df.loc[run_df['board'].isin(present_boards)].reset_index(drop=True)

    logging.info(f"Determine neighbor hits in track file: {present_boards}")
    run_df = find_neighbor_hits(run_df, args.search_method)

    # 5. Build Fast Lookups (Optimization)

    # A. Calibration Map (Dict Lookup)
    logging.info("Building calibration map...")
    cal_df = pd.read_csv(args.cal_table)
    cal_map = cal_df.set_index(['board', 'row', 'col'])['cal_mode'].to_dict()

    # B. Hit Index (Groupby Lookup) - The new speed booster
    logging.info("Indexing hits by (board, row, col)...")
    hit_index = run_df.groupby(['board', 'row', 'col'])

    # 6. Process Tracks
    track_pivots = []
    if args.file_index is not None:
        file_indicator = args.file_index
    else:
        try:
            file_indicator = int(Path(args.inputfile).stem.split('_')[1])
        except (IndexError, ValueError):
            logging.error(
                f"Could not parse file index from '{args.inputfile}' (expected 'loop_<N>...'). "
                f"Pass --file-index explicitly to avoid relying on filename parsing."
            )
            sys.exit(1)

    logging.info(f"Processing {len(track_df)} tracks...")

    # Pull row_X/col_X once per track up front instead of repeated .iloc[itrack] lookups
    pix_cols = [f'{axis}_{bid}' for bid in present_boards for axis in ('row', 'col')]
    track_rows = track_df[pix_cols].to_dict('records')

    for itrack in tqdm(range(len(track_df))):
        row_data = track_rows[itrack]
        pix_dict = {
            bid: [row_data[f'row_{bid}'], row_data[f'col_{bid}']]
            for bid in present_boards
        }

        table = extract_events_for_track(
            hit_index=hit_index,  # Passing the index instead of the df
            cal_map=cal_map,
            pix_dict=pix_dict,
            board_ids=present_boards,
        )

        if not table.empty:
            table['track_id'] = itrack
            table['file'] = file_indicator
            track_pivots.append(table)

    # 7. Save Output
    fname = Path(args.inputfile).stem
    out_name = f'{args.runinfo}_{fname}.parquet'

    if track_pivots:
        logging.info(f"Concatenating {len(track_pivots)} tracks...")

        # 1. Flatten: Merge all small DataFrames into one massive DataFrame
        final_df = pd.concat(track_pivots, ignore_index=True)

        # 2. Flatten MultiIndex Columns (Parquet doesn't like Tuple columns)
        # Your pivot table created columns like ('row', 0), ('row', 1).
        # We flatten them to 'row_0', 'row_1'.
        final_df.columns = [f'{c[0]}_{c[1]}' if isinstance(c, tuple) and str(c[1]) != '' else c[0] for c in final_df.columns]

        logging.info("Applying column role renaming...")
        rename_dict = {}
        # Iterate through the columns and apply the map to the index part
        for col in final_df.columns:
            # Check if column ends with a positional index we need to replace
            for old_suffix, new_suffix in rename_map.items():
                if col.endswith(old_suffix):
                    rename_dict[col] = col.replace(old_suffix, new_suffix)
                    break

        if rename_dict:
            final_df = final_df.rename(columns=rename_dict)

        # 3. Downcast Types (Final Check)
        # Example: columns containing 'row', 'col' can be uint16 usually
        for col in final_df.columns:
            if 'row' in col or 'col' in col:
                final_df[col] = final_df[col].astype('uint8')
            elif 'toa' in col or 'tot' in col or 'cal' in col:
                final_df[col] = final_df[col].astype('uint16')
            elif 'HasNeighbor' in col:
                final_df[col] = final_df[col].astype(bool)
            elif 'track_id' in col or 'file' in col:
                final_df[col] = final_df[col].astype('uint16')

        logging.info(f"Saving to {out_name} with compression...")

        # 4. Save to Parquet with LZ4 compression (high compression ratio)
        io_utils.write_parquet(final_df, out_name, index=False, compression='lz4')


if __name__ == "__main__":
    main()