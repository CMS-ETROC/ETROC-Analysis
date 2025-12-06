import argparse
import pickle
import sys
import logging
from collections import defaultdict
from functools import reduce
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.signal import argrelextrema

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')

# --- Helper Functions ---

def determine_tot_cut_range_for_trig(input_df: pd.DataFrame, trig_id: int) -> List[float]:
    """Calculates Time-over-Threshold (ToT) cuts based on the valley after the first peak."""
    trig_tot = input_df.loc[input_df['board'] == trig_id, 'tot'].reset_index(drop=True)

    counts, bin_centers = np.histogram(trig_tot, bins=128, range=(0, 512))
    first_peak_index = np.argmax(counts)

    minima_indices = argrelextrema(counts, np.less)[0]
    valley_candidates = minima_indices[minima_indices > first_peak_index]

    if len(valley_candidates) > 0:
        valley_index = valley_candidates[0]
        filtered_tot = trig_tot.loc[trig_tot < bin_centers[valley_index]]
        return [filtered_tot.quantile(0.02), filtered_tot.quantile(0.98)]
    else:
        return [trig_tot.quantile(0.03), trig_tot.quantile(0.96)]

def tdc_event_selection(input_df: pd.DataFrame, tdc_cuts_dict: dict) -> pd.DataFrame:
    """Filters events based on calibration, ToA, and ToT cuts."""
    masks = {}
    for board, cuts in tdc_cuts_dict.items():
        # cuts = [cal_min, cal_max, toa_min, toa_max, tot_min, tot_max]
        mask = (
            (input_df['board'] == board) &
            input_df['cal'].between(cuts[0], cuts[1]) &
            input_df['toa'].between(cuts[2], cuts[3]) &
            input_df['tot'].between(cuts[4], cuts[5])
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
    trig_id: int,
    board_ids: List[int],
    tot_cuts: List[float]
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

    # 2. Define TDC Cuts per board
    tdc_cuts = {}
    for bid in board_ids:
        row, col = pix_dict[bid]
        cal_mode = cal_map.get((bid, row, col))

        if cal_mode is None: return pd.DataFrame()

        if bid == trig_id:
            tdc_cuts[bid] = [cal_mode - 3, cal_mode + 3, 0, 1100, tot_cuts[0], tot_cuts[1]]
        else:
            tdc_cuts[bid] = [cal_mode - 3, cal_mode + 3, 0, 1100, 0, 600]

    # 3. Apply TDC filtering
    track_tmp_df = tdc_event_selection(track_tmp_df, tdc_cuts)

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
        values=["row", "col", "toa", "tot", "cal"]
    ).reset_index(drop=True)

    return pivot_table

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description='Select detailed event data based on track candidates.')
    parser.add_argument('-f', '--inputfile', required=True, dest='inputfile', help='Input feather file')
    parser.add_argument('-r', '--runinfo', required=True, dest='runinfo', help='Run info string for output name')
    parser.add_argument('-t', '--track', required=True, dest='track', help='CSV file with track candidates')
    parser.add_argument('--cal_table', required=True, dest='cal_table', help='CSV file with CAL mode values')
    parser.add_argument('--trigID', type=int, required=True, dest='trigID', help='Trigger board ID')

    args = parser.parse_args()

    # 1. Load Main Data
    try:
        run_df = pd.read_feather(args.inputfile, columns=['evt', 'board', 'row', 'col', 'toa', 'tot', 'cal'])
    except Exception as e:
        logging.error(f"Failed to read input file: {e}")
        sys.exit(1)

    if run_df.empty:
        logging.error('Empty input dataframe.')
        sys.exit(0)

    # 2. Load Track Candidates
    track_df = pd.read_csv(args.track)
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

    # 5. Build Fast Lookups (Optimization)

    # A. Calibration Map (Dict Lookup)
    logging.info("Building calibration map...")
    cal_df = pd.read_csv(args.cal_table)
    cal_map = cal_df.set_index(['board', 'row', 'col'])['cal_mode'].to_dict()

    # B. Hit Index (Groupby Lookup) - The new speed booster
    logging.info("Indexing hits by (board, row, col)...")
    hit_index = run_df.groupby(['board', 'row', 'col'])

    # 6. Calculate Trigger Cuts
    tot_cuts = determine_tot_cut_range_for_trig(run_df, args.trigID)
    logging.info(f"Calculated ToT cuts for Trig ({args.trigID}): {tot_cuts}")

    # 7. Process Tracks
    track_pivots = defaultdict(pd.DataFrame)
    file_indicator = int(args.inputfile.split('.')[0].split('_')[1])

    logging.info(f"Processing {len(track_df)} tracks...")

    for itrack in tqdm(range(len(track_df))):
        pix_dict = {}
        for bid in present_boards:
            pix_dict[bid] = [
                track_df.iloc[itrack][f'row_{bid}'],
                track_df.iloc[itrack][f'col_{bid}']
            ]

        table = extract_events_for_track(
            hit_index=hit_index,  # Passing the index instead of the df
            cal_map=cal_map,
            pix_dict=pix_dict,
            trig_id=args.trigID,
            board_ids=present_boards,
            tot_cuts=tot_cuts
        )

        if not table.empty:
            table['file'] = file_indicator
            table['file'] = table['file'].astype('uint16')
            track_pivots[itrack] = table

    # 8. Save Output
    fname = args.inputfile.split('.')[0]
    out_name = f'{args.runinfo}_{fname}.pickle'

    logging.info(f"Saving {len(track_pivots)} tracks to {out_name}")
    with open(out_name, 'wb') as output:
        pickle.dump(track_pivots, output, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()