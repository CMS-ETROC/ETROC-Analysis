import argparse
import sys
import logging
import yaml
from pathlib import Path
from typing import List, Dict

import pandas as pd
import numpy as np

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


def _extract_batch_vectorized(
    run_df: pd.DataFrame,
    track_df: pd.DataFrame,
    board_ids: List[int],
    cal_df: pd.DataFrame,
) -> pd.DataFrame:
    """Vectorized replacement for the old one-track-at-a-time loop: instead of
    a Python for-loop calling hit_index.get_group()/concat/groupby/pivot once
    per track candidate (thousands of small pandas calls, each paying fixed
    per-call overhead), the same isolation + CAL-window logic is applied to
    every track at once via a handful of operations over the whole hit table.

    Equivalence with the old per-track logic (select_calibrated_hits() +
    isolation-check): an event only ever contributes to a track's output if
    every required board has an isolated (count == 1) raw hit at that board's
    candidate pixel in that event, and that hit's CAL falls within the pixel's
    calibrated mode +/- 3. That's what the old combination reduces to once
    traced through -- select_calibrated_hits() only ever drops whole events,
    never individual rows within a kept event, so the isolation count it fed
    into always counted every raw hit at the pixel regardless of CAL. Verified
    byte-for-byte identical output against the old implementation on real
    files across multiple board combos before this replaced it.

    ToT-based quality cuts are intentionally left to apply_tdc_cuts.py
    (step 10), which runs after reshaping and can compute a robust per-role
    ToT range instead of a single-file, trig-only window.

    Called per track batch by extract_all_tracks_vectorized() -- see there
    for why this is chunked rather than run once over the full track_df.
    """
    # 1. Long-format (track_id, board, row, col) -- tiny, n_tracks * n_boards rows.
    # Dtypes are pinned to match run_df's merge-key columns (board/row/col are
    # uint8 there): leaving these at the pandas default (int64, from
    # track_df.index and bare .to_numpy()) silently widens run_df's uint8
    # merge-key columns to int64 too once merged below, ~8x'ing 3 of the 9
    # columns in the resulting hits_long.
    pixels_long = pd.concat([
        pd.DataFrame({
            'track_id': np.asarray(track_df.index, dtype='uint16'),
            'board': np.uint8(bid),
            'row': track_df[f'row_{bid}'].to_numpy(dtype='uint8'),
            'col': track_df[f'col_{bid}'].to_numpy(dtype='uint8'),
        })
        for bid in board_ids
    ], ignore_index=True)

    # 2. One merge instead of len(track_df) separate hit_index.get_group() calls --
    # every raw hit (any CAL) at each requested (board,row,col), tagged with
    # which track(s) requested it.
    hits_long = pixels_long.merge(run_df, on=['board', 'row', 'col'], how='inner')
    if hits_long.empty:
        return pd.DataFrame()

    # 3. Isolation: exactly 1 raw hit for this (track, board) in this event.
    # Packed-key np.unique instead of groupby(...).transform('size'): same
    # result, but avoids pandas' multi-column Grouper/hashing machinery.
    # uint32 relies on two things holding together: track_id is batch-local
    # (bounded by BATCH_SIZE=100 below, not by track_df's full length), and
    # evt -- while declared uint32 -- never actually approaches its 4.3e9
    # ceiling in real data (it's a per-file event count, observed max ~1.94M
    # on this run). Packing 100 tracks * 4 boards * a ~2M-range evt keeps the
    # worst case around 5.8e8, safely under uint32's limit; a per-file evt
    # count anywhere near uint32's actual max would overflow this and needs
    # reverting to int64.
    n_boards_span = int(hits_long['board'].max()) + 1
    n_evt_span = int(hits_long['evt'].max()) + 1
    key = (hits_long['track_id'].to_numpy('uint32') * n_boards_span
           + hits_long['board'].to_numpy('uint32')) * n_evt_span + hits_long['evt'].to_numpy('uint32')
    _, inverse, key_counts = np.unique(key, return_inverse=True, return_counts=True)
    counts = key_counts[inverse]
    isolated = hits_long.loc[counts == 1]
    if isolated.empty:
        return pd.DataFrame()

    # 4. CAL validity of that single hit -- dense-array lookup, same trick
    # path_finder.py uses for its own CAL-deviation filter (board/row/col are
    # small bounded integers, so this beats a merge-based join here too).
    # cal_lookup uses -1 as the "no CAL-mode entry for this pixel" sentinel
    # (real cal_mode values are always >= 0) since int16 can't hold NaN --
    # int16 comfortably covers CAL's 0-1023 range at a quarter the size of
    # the float64 this used to be.
    max_board = max(int(cal_df['board'].max()), int(isolated['board'].max())) + 1
    max_row = max(int(cal_df['row'].max()), int(isolated['row'].max())) + 1
    max_col = max(int(cal_df['col'].max()), int(isolated['col'].max())) + 1
    cal_lookup = np.full((max_board, max_row, max_col), -1, dtype='int16')
    cal_lookup[cal_df['board'].to_numpy(), cal_df['row'].to_numpy(), cal_df['col'].to_numpy()] = cal_df['cal_mode'].to_numpy()

    cal_mode_vals = cal_lookup[isolated['board'].to_numpy(), isolated['row'].to_numpy(), isolated['col'].to_numpy()]
    cal_dev = isolated['cal'].to_numpy(dtype='int16') - cal_mode_vals
    valid_cal = (np.abs(cal_dev) <= 3) & (cal_mode_vals != -1)
    valid = isolated.loc[valid_cal]
    if valid.empty:
        return pd.DataFrame()

    # 5. Every required board must be valid for the same (track_id, evt).
    # Isolation (step 3) already guarantees at most one row per
    # (track_id, board, evt), so within a (track_id, evt) group, board
    # values can't repeat -- a row count is therefore equivalent to a
    # distinct-board count, without needing an actual nunique. uint32 is
    # safe for the same reason as step 3's key (batch-local track_id,
    # real-world evt range far under uint32's ceiling).
    n_evt_span2 = int(valid['evt'].max()) + 1
    key2 = valid['track_id'].to_numpy('uint32') * n_evt_span2 + valid['evt'].to_numpy('uint32')
    _, inverse2, key_counts2 = np.unique(key2, return_inverse=True, return_counts=True)
    board_counts = key_counts2[inverse2]
    survivors = valid.loc[board_counts == len(board_ids)]
    if survivors.empty:
        return pd.DataFrame()

    # 6. One pivot for every surviving (track_id, evt) x board, instead of one per track.
    pivot_table = survivors.pivot(
        index=['track_id', 'evt'],
        columns='board',
        values=['row', 'col', 'toa', 'tot', 'cal', 'HasNeighbor'],
    ).reset_index()
    pivot_table.columns = [
        f'{c[0]}_{c[1]}' if isinstance(c, tuple) and str(c[1]) != '' else c[0]
        for c in pivot_table.columns
    ]
    return pivot_table.drop(columns=['evt'])


# Tracks per _extract_batch_vectorized() call. Fixed rather than a caller
# parameter because the uint32 packed-key dtype inside that function is only
# safe for this specific bound (see step 3's comment there) -- raising this
# would need those keys widened back to int64.
BATCH_SIZE = 100


def extract_all_tracks_vectorized(
    run_df: pd.DataFrame,
    track_df: pd.DataFrame,
    board_ids: List[int],
    cal_df: pd.DataFrame,
) -> pd.DataFrame:
    """Runs _extract_batch_vectorized() over track_df in chunks of BATCH_SIZE
    tracks and concatenates the results.

    The (board,row,col)-only merge inside _extract_batch_vectorized() blows
    up superlinearly with the number of tracks processed together -- every
    hit at a pixel gets duplicated once per track that queries it, so more
    tracks in one call means both more duplication per hit and more hits
    matched overall. Measured on a real run (1083 tracks, ~6M hits after
    board filtering): unbatched (all 1083 tracks in one call) peaked at
    ~4.6GB RSS even after the dtype/np.unique fixes inside
    _extract_batch_vectorized; BATCH_SIZE=100 brings that down to roughly
    a third of that, with byte-for-byte identical output.
    """
    results = []
    for start in range(0, len(track_df), BATCH_SIZE):
        batch_result = _extract_batch_vectorized(
            run_df, track_df.iloc[start:start + BATCH_SIZE], board_ids, cal_df
        )
        if not batch_result.empty:
            results.append(batch_result)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)

# --- Main Execution ---

def build_arg_parser() -> argparse.ArgumentParser:
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
    parser.add_argument('-o', '--outdir', default='.', dest='outdir',
                        help="Directory to write the output Parquet into (default: current directory, "
                             "i.e. condor's own worker sandbox -- unchanged from before, relies on "
                             "condor's output_destination to copy it back). Local (non-condor) callers "
                             "pass this explicitly so output lands directly in its real destination.")
    return parser


def run(args: argparse.Namespace) -> None:
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
        # int16 (not uint16) is enough headroom -- CAL codes are 0-1023, so any
        # cal-cal_mode difference stays within +/-1023, far inside int16's +/-32767.
        run_df['cal'] = run_df['cal'].astype('int16')
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

    # 5. Load CAL table (used as a dense lookup array inside the vectorized extractor)
    logging.info("Loading CAL table...")
    cal_df = pd.read_csv(args.cal_table)
    cal_df['row'] = cal_df['row'].astype('uint8')
    cal_df['col'] = cal_df['col'].astype('uint8')
    cal_df['board'] = cal_df['board'].astype('uint8')
    # cal_mode is a mode over integer CAL codes, so it's always whole-valued
    # despite arriving as float64 from the CSV.
    cal_df['cal_mode'] = cal_df['cal_mode'].astype('int16')

    # 6. Process Tracks
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

    logging.info(f"Processing {len(track_df)} tracks against {len(run_df)} hits (vectorized)...")
    final_df = extract_all_tracks_vectorized(run_df, track_df, present_boards, cal_df)

    # 7. Save Output
    fname = Path(args.inputfile).stem
    out_name = Path(args.outdir) / f'{args.runinfo}_{fname}.parquet'

    if not final_df.empty:
        final_df['file'] = file_indicator

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
    else:
        logging.warning("No surviving tracks/events after CAL and isolation filtering. Nothing saved.")


if __name__ == "__main__":
    run(build_arg_parser().parse_args())