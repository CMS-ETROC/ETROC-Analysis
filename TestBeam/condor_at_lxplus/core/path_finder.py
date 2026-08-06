import argparse
import sys
import logging
import warnings
import random, getpass
from itertools import combinations
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple
from ruamel.yaml import YAML

import io_utils

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore")

# --- Constants & Defaults ---
PIXEL_PITCH = 1.3
PIXEL_OFFSET = 7.5
MAX_MEMORY_USAGE_MB = 2600
SINGLE_FILE_MEMORY_LIMIT_MB = 26
MIN_BOARD_COMBO_SIZE = 3  # extract_events_by_path.py (step 8) requires at least 3 boards

# --- Helper Functions ---

def check_empty_df(input_df: pd.DataFrame, context_msg: str = ""):

    """
    Checks if DataFrame is empty and exits if true.
    """

    if input_df.empty:
        logging.warning(f"DataFrame is empty after {context_msg}")
        sys.exit(1)

def reindex_events(df: pd.DataFrame) -> pd.DataFrame:

    """
    Re-indexes events sequentially to ensure continuity.
    """

    if df.empty:
        return df
    is_new_event = df['evt'] != df['evt'].shift()
    df['evt'] = is_new_event.cumsum() - 1
    return df

def get_transformation_params(board_id: int, config: dict) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:

    """
    Extracts rotation (radians) and translation from config with safe defaults.
    """

    # Safely get the specific board config, defaulting to empty dicts if missing
    board_conf = config.get(board_id, {})
    trans_conf = board_conf.get('transformation', {})

    rot_conf = trans_conf.get('rotation', {})
    tra_conf = trans_conf.get('translation', {})

    # Extract values with defaults (0.0) and convert rotation to radians
    rot = (
        np.deg2rad(rot_conf.get('x', 0.0)),
        np.deg2rad(rot_conf.get('y', 0.0)),
        np.deg2rad(rot_conf.get('z', 0.0))
    )

    tra = (
        tra_conf.get('x', 0.0),
        tra_conf.get('y', 0.0),
        tra_conf.get('z', 0.0)
    )

    return rot, tra

def get_rotation_matrix(rx, ry, rz):
    # Pre-calculate trig values once
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    # Construct the Z-Y-X rotation matrix
    R = np.array([
        [cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx],
        [sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx],
        [-sy,   cy*sx,            cy*cx]
    ])
    return R

def apply_geometric_transformation_matrix(df, board_ids, config):
    for bid in board_ids:
        if f'col_{bid}' not in df.columns: continue

        # 1. Local coordinates (Pre-calculated vectors)
        x_prime = (df[f'col_{bid}'] - PIXEL_OFFSET) * PIXEL_PITCH
        y_prime = (df[f'row_{bid}'] - PIXEL_OFFSET) * PIXEL_PITCH
        z_prime = np.zeros_like(x_prime) # Boards are 2D planes at z=0 locally

        # 2. Get Transformation Parameters
        rot, tra = get_transformation_params(bid, config)
        R = get_rotation_matrix(*rot)

        # 3. Stack into (N, 3) matrix and apply dot product
        coords = np.stack([x_prime, y_prime, z_prime], axis=1)
        transformed = coords @ R.T + np.array(tra)

        # 4. Re-assign to dataframe
        df[f'x_{bid}'] = transformed[:, 0]
        df[f'y_{bid}'] = transformed[:, 1]
        df[f'z_{bid}'] = transformed[:, 2]

# --- Core Logic Blocks ---

def load_and_sample_data(file_paths: List[Path], sampling_rate: float) -> pd.DataFrame:

    """
    Loads feather files, performs memory checks, and concatenates data.
    """

    columns_to_read = ['evt', 'board', 'row', 'col', 'toa', 'tot', 'cal']
    portion = sampling_rate * 0.01

    def sample_events(tmp_df: pd.DataFrame) -> pd.DataFrame:
        unique_evts = tmp_df['evt'].unique()
        if len(unique_evts) == 0:
            return tmp_df
        n = max(1, int(portion * len(unique_evts)))
        indices = np.random.choice(unique_evts, n, replace=False)
        return tmp_df.loc[tmp_df['evt'].isin(indices)]

    # 1. Memory Safety Check
    logging.info('Performing Memory Safety Check...')
    check_files = file_paths if len(file_paths) < 10 else random.sample(file_paths, 10)
    sum_use = 0
    # Cache these full reads so the main loading pass below doesn't hit disk
    # a second time for the same (up to 10) files.
    checked_full_dfs = {}

    for f in tqdm(check_files):
        full_df_f = pd.read_feather(f, columns=columns_to_read)
        checked_full_dfs[f] = full_df_f
        temp_df = sample_events(full_df_f)
        sum_use += temp_df.memory_usage(deep=True).sum() / (1024**2)

    avg_use = sum_use / len(check_files)
    total_est = avg_use * len(file_paths)

    logging.info(f'Avg usage: {avg_use:.2f} MB, Est total: {total_est:.2f} MB')

    if avg_use > SINGLE_FILE_MEMORY_LIMIT_MB or total_est > MAX_MEMORY_USAGE_MB:
        logging.error('Memory limit exceeded. Reduce sampling rate or file count.')
        sys.exit(1)

    # 2. Real Loading
    logging.info('Loading data...')
    dfs = []
    for f in tqdm(file_paths, desc="Reading Files"):
        tmp_df = checked_full_dfs[f] if f in checked_full_dfs else pd.read_feather(f, columns=columns_to_read)
        unique_evts = tmp_df['evt'].unique()
        if len(unique_evts) == 0:
            continue
        tmp_df = sample_events(tmp_df)
        dfs.append(tmp_df)

    if not dfs:
        logging.warning("No data loaded.")
        sys.exit()

    full_df = pd.concat(dfs, ignore_index=True)
    full_df = reindex_events(full_df)

    logging.info(f'Total memory usage: {full_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB')
    return full_df

def apply_masking(df: pd.DataFrame, mask_config_path: Path) -> pd.DataFrame:

    """
    Removes noisy pixels defined in the mask config.
    """

    if not mask_config_path:
        return df

    # Initialize the modern YAML loader
    yaml = YAML(typ='safe') # Use 'safe' for faster read-only performance

    with open(mask_config_path, 'r') as f:
        mask_info = yaml.load(f)

    bad_pixels = [
        (board_id, r, c)
        for board_id, val in mask_info.get("board_ids", {}).items()
        for (r, c) in val.get('pixels', [])
    ]

    if not bad_pixels:
        return df

    # Single combined membership check instead of one filter pass per pixel
    bad_index = pd.MultiIndex.from_tuples(bad_pixels, names=['board', 'row', 'col'])
    row_keys = pd.MultiIndex.from_arrays([df['board'], df['row'], df['col']])
    df = df[~row_keys.isin(bad_index)]

    return df.reset_index(drop=True)

def generate_cal_table(df: pd.DataFrame, output_name: str) -> pd.DataFrame:

    """
    Calculates mode for 'cal' values per pixel and saves to CSV.
    """

    logging.info('Generating CAL mode table...')

    # Efficient mode calculation using pivot_table
    cal_table = df.pivot_table(
        index=["row", "col"],
        columns=["board"],
        values=["cal"],
        aggfunc=lambda x: x.mode().iat[0] if not x.mode().empty else np.nan
    )

    # Flatten structure
    cal_table = cal_table.stack(level='board').reset_index()
    # Fix column names after stack
    cal_table.columns = ['row', 'col', 'board', 'cal_mode']

    # Save
    io_utils.write_csv(cal_table, f'{output_name}_cal_table.csv', index=False)
    return cal_table

def check_spatial_alignment(df: pd.DataFrame, roles: Dict[str, int], max_diff_pixel: float) -> pd.Series:

    """
    Checks if hits are spatially aligned across boards using Euclidean distance.
    Logic: r = sqrt((x1-x2)^2 + (y1-y2)^2) <= limit
    """

    trig_id = roles.get('trig')

    # We need a trigger board to compare against
    if trig_id is None:
        logging.warning("No 'trig' role found. Skipping spatial alignment check.")
        return pd.Series(True, index=df.index)

    # Identify other boards to compare with Trigger
    # We compare Trig vs Ref, Trig vs DUT, Trig vs Extra
    others = [r for r in ['ref', 'dut', 'extra'] if r in roles]

    conditions = []
    # Limit calculation: pixels * pitch (mm/pixel)
    limit = max_diff_pixel * PIXEL_PITCH

    for role_name in others:
        other_id = roles[role_name]

        # Check if global coordinates exist for both
        if f'x_{trig_id}' in df.columns and f'x_{other_id}' in df.columns:

            # Calculate deltas
            dx = df[f'x_{trig_id}'] - df[f'x_{other_id}']
            dy = df[f'y_{trig_id}'] - df[f'y_{other_id}']

            # Cartesian (Euclidean) Distance Check
            distance = np.sqrt(dx**2 + dy**2)
            conditions.append(distance <= limit)

    if not conditions:
        return pd.Series(True, index=df.index)

    # Combine all conditions (must satisfy distance check for ALL pairs)
    return np.logical_and.reduce(conditions)

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description='Find track candidates and Calibrate.')
    parser.add_argument('-p', '--path', required=True, help='Path to directory with feather files')
    parser.add_argument('--cal-label', required=True, help='Output name for CAL table', dest='cal_label')
    parser.add_argument('--track-label', required=True, help='Output name for Tracks', dest='track_label')
    parser.add_argument('-s', '--sampling', type=float, default=3, help='Sampling fraction (percent)')
    parser.add_argument('-m', '--minimum', type=int, default=1000, dest='ntracks', help='Min tracks')
    parser.add_argument('--max_diff_pixel', type=int, default=1, help='Max pixel diff')
    parser.add_argument('-c', '--config', required=True, help='YAML config file')
    parser.add_argument('-r', '--runName', required=True, help='Run name in YAML')
    parser.add_argument('--mask_config', type=Path, dest='mask_config_file', help='Mask config YAML')
    parser.add_argument('--cal_table_only', action='store_true', help='Only generate CAL table')
    parser.add_argument('--find_alignment', action='store_true', help='Find the board offset alignments refer to trigger board')

    args = parser.parse_args()

    # 1. Setup & Config with modern API
    yaml = YAML(typ='rt')  # 'rt' = Round Trip
    yaml.preserve_quotes = True
    yaml.default_flow_style = None  # Crucial: tells ruamel to respect existing flow/block style
    yaml.width = 4096  # Prevents ruamel from wrapping long lines into block style

    with open(args.config, 'r') as f:
        full_config = yaml.load(f)

    if args.runName not in full_config:
        raise ValueError(f"Run config {args.runName} not found")

    run_config = full_config[args.runName]

    # Map roles to IDs
    roles = {info['role']: bid for bid, info in run_config.items()}

    # 2. Load Data

    # --- Setup Environments ---
    username = getpass.getuser()
    eos_base_dir = io_utils.eos_base_dir(username)

    files = list((eos_base_dir / args.path).glob('loop*feather'))
    if len(files) > 100: files = random.sample(files, 100)

    if not files:
        logging.error("No input files found.")
        sys.exit(1)

    df = load_and_sample_data(files, args.sampling)

    # 3. Preprocessing
    if args.mask_config_file:
        df = apply_masking(df, args.mask_config_file)

    # 4. Calibration
    cal_table = generate_cal_table(df, args.cal_label)

    if args.cal_table_only:
        logging.info("Cal table only mode. Exiting.")
        sys.exit(0)

    # 5. Track Finding
    logging.info('Starting track finding...')

    # Filter based on CAL deviations.
    # board/row/col are small bounded integers (a handful of boards x a 16x16
    # pixel grid), so a dense lookup array + vectorized numpy indexing does the
    # same job as a merge()'d hash-join but without pandas' general-purpose
    # join machinery -- much faster for a multi-million-row hit table against
    # a ~1k-row cal_table. Bounds are taken from both cal_table and df so a
    # pixel present only in df (no cal_table entry) still indexes safely and
    # simply falls out via the NaN check below (same as the old how='left').
    max_board = max(int(cal_table['board'].max()), int(df['board'].max())) + 1
    max_row = max(int(cal_table['row'].max()), int(df['row'].max())) + 1
    max_col = max(int(cal_table['col'].max()), int(df['col'].max())) + 1
    cal_lookup = np.full((max_board, max_row, max_col), np.nan, dtype='float64')
    cal_lookup[cal_table['board'].to_numpy(), cal_table['row'].to_numpy(), cal_table['col'].to_numpy()] = cal_table['cal_mode'].to_numpy()

    cal_mode_vals = cal_lookup[df['board'].to_numpy(), df['row'].to_numpy(), df['col'].to_numpy()]
    # 'cal' is stored as uint16, so subtracting a float64 cal_mode promotes
    # safely (no wraparound risk, unlike a plain uint16-uint16 subtraction).
    cal_dev = df['cal'].astype('int32').to_numpy() - cal_mode_vals
    valid_cal = (np.abs(cal_dev) <= 3) & ~np.isnan(cal_mode_vals)
    df = df.loc[valid_cal].reset_index(drop=True)

    df = reindex_events(df) # Renumber after filtering
    check_empty_df(df, "CAL deviation filtering")

    ids_to_process = sorted(roles.values())
    df[['row', 'col']] = df[['row', 'col']].astype('int8') # Optimization

    # Per-board, per-event hit counts, once -- reused by every combo below.
    # evt is contiguous 0..n-1 after reindex_events, so a bincount over
    # (evt, board) pairs reproduces the old groupby+unstack matrix without
    # pandas' hashing/pivoting overhead -- much faster for a large event count.
    board_vals = df['board'].to_numpy()
    evt_vals = df['evt'].to_numpy()
    n_events = int(evt_vals.max()) + 1
    n_boards = int(board_vals.max()) + 1
    boards_with_hits = set(np.unique(board_vals).tolist())
    combined = evt_vals.astype(np.int64) * n_boards + board_vals.astype(np.int64)
    counts_2d = np.bincount(combined, minlength=n_events * n_boards).reshape(n_events, n_boards)
    single_hit = counts_2d == 1  # shape (n_events, n_boards): True where that board has exactly 1 hit

    # Board combinations to produce tracks for: the full board set, plus every
    # smaller subset down to 3 boards (e.g. for 4 boards, that's the 4-board
    # set and all four 3-board leave-one-out subsets). Each combo gets its own
    # single-hit requirement -- a board left out of a combo isn't required to
    # have a hit for that combo's tracks. extract_events_by_path.py requires
    # at least 3 boards, so we never go below that.
    min_boards = min(MIN_BOARD_COMBO_SIZE, len(ids_to_process))
    board_combos = [
        combo
        for size in range(len(ids_to_process), min_boards - 1, -1)
        for combo in combinations(ids_to_process, size)
    ]

    values_to_keep = ['row', 'col'] # minimal set for tracking
    trig_id = roles.get('trig')
    id_to_role = {v: k for k, v in roles.items()}
    n_written = 0
    alignment_results = {} # combo_label -> {board_id: translation dict}, written once after the loop

    for combo in board_combos:
        # Tag each board id with its role (from the config YAML) so output
        # filenames are legible without cross-referencing the config, e.g.
        # "trig0-ref1-dut2" instead of just "0-1-2".
        combo_label = '-'.join(f"{id_to_role[b]}{b}" for b in combo)
        logging.info(f"--- Board combo ({combo_label}) ---")

        # Single Hit Selection: every board in this combo must have exactly 1 hit
        missing = [b for b in combo if b not in boards_with_hits]
        if missing:
            logging.warning(f"Combo ({combo_label}): board(s) {missing} have no hits at all. Skipping.")
            continue

        valid_event_mask = np.logical_and.reduce([single_hit[:, b] for b in combo])
        valid_events = np.nonzero(valid_event_mask)[0]

        combo_df = df.loc[df['evt'].isin(valid_events) & df['board'].isin(combo)].reset_index(drop=True)
        if combo_df.empty:
            logging.warning(f"Combo ({combo_label}): no valid events. Skipping.")
            continue

        # Pivot to Wide Format (Events as rows, Boards as columns)
        track_df = combo_df.pivot(index='evt', columns='board', values=values_to_keep)
        # Flatten columns: (row, 0) -> row_0
        track_df.columns = [f"{v}_{b}" for v, b in track_df.columns]

        # Group identical hit patterns (finding "Hot Tracks" or frequent combinations)
        group_cols = list(track_df.columns)

        track_candidates = track_df.groupby(group_cols).size().reset_index(name='count')
        track_candidates = track_candidates[track_candidates['count'] > args.ntracks]

        if track_candidates.empty:
            logging.warning(f"Combo ({combo_label}): no track candidates above threshold. Skipping.")
            continue

        # 6. Geometric Transformation & Final Filtering
        apply_geometric_transformation_matrix(track_candidates, combo, run_config)

        # Alignment offsets are a property of the boards themselves, but the
        # estimate itself still depends on which combo's tracks it's computed
        # from -- so compute (and record) it for every combo that includes the
        # trigger board, keyed by combo, instead of overwriting one shared
        # result. Only the full board-set combo's estimate (processed first)
        # is ever fed back into run_config, so every combo's own track output
        # stays on the same, consistent geometry; smaller combos' numbers are
        # recorded purely so you can sanity-check that they agree.
        if args.find_alignment and trig_id in combo:
            shift_df = track_candidates.copy(deep=True)
            combo_alignment = {}
            for bid in combo:
                if bid == trig_id:
                    continue

                shift_df[f'x_{bid}'] = shift_df[f'x_{bid}'] - shift_df[f'x_{trig_id}']
                shift_df[f'y_{bid}'] = shift_df[f'y_{bid}'] - shift_df[f'y_{trig_id}']

                hist_counts, bin_edges = np.histogram(shift_df[f'x_{bid}'], weights=shift_df['count'], bins=30)
                max_index = np.argmax(hist_counts)
                max_bin_start = bin_edges[max_index]
                max_bin_end = bin_edges[max_index + 1]

                center_x = round(float(0.5*(max_bin_end+max_bin_start)), 2)

                hist_counts, bin_edges = np.histogram(shift_df[f'y_{bid}'], weights=shift_df['count'], bins=30)
                max_index = np.argmax(hist_counts)
                max_bin_start = bin_edges[max_index]
                max_bin_end = bin_edges[max_index + 1]

                center_y = round(float(0.5*(max_bin_end+max_bin_start)), 2)

                existing = run_config.get(bid, {}).get('transformation', {}).get('translation', {'x': 0.0, 'y': 0.0, 'z': 0.0})
                combo_alignment[bid] = {
                    'x': existing.get('x', 0.0) - center_x,
                    'y': existing.get('y', 0.0) - center_y,
                    'z': existing.get('z', 0.0),
                }

            alignment_results[combo_label] = combo_alignment

            # Only the full board-set combo's estimate feeds forward, applied
            # in-memory only (never saved back to args.config), so subsequent
            # combos' own track output uses the corrected geometry too.
            if len(combo) == len(ids_to_process):
                for bid, new_translation in combo_alignment.items():
                    full_config[args.runName][bid].setdefault('transformation', {})['translation'] = new_translation
                run_config = full_config[args.runName]
                apply_geometric_transformation_matrix(track_candidates, combo, run_config)

        spatial_mask = check_spatial_alignment(track_candidates, roles, args.max_diff_pixel)
        final_tracks = track_candidates[spatial_mask]

        # Remove duplicates if any remain based on pattern
        final_tracks = final_tracks.drop_duplicates(subset=group_cols)

        coord_cols = [c for c in final_tracks.columns if c.split('_')[0] in ['x', 'y', 'z']]
        final_tracks[coord_cols] = final_tracks[coord_cols].round(2)

        output_file = f'{args.track_label}_tracks_{combo_label}.parquet'
        io_utils.write_parquet(final_tracks, output_file, index=False)
        logging.info(f"Combo ({combo_label}): {len(final_tracks)} tracks saved to {output_file}")
        n_written += 1

    if alignment_results:
        alignment_file = f'{args.track_label}_alignment.yaml'
        with open(alignment_file, 'w') as f:
            yaml.dump({
                args.runName: {
                    combo_label: {bid: {'transformation': {'translation': t}} for bid, t in combo_vals.items()}
                    for combo_label, combo_vals in alignment_results.items()
                }
            }, f)
        logging.info(f"Alignment offsets for {len(alignment_results)} combo(s) written to {alignment_file} "
                     f"(not saved back to {args.config} -- merge in manually if desired).")

    if n_written == 0:
        logging.warning("No track candidates found for any board combo.")
        sys.exit(0)

    logging.info(f"Done. {n_written}/{len(board_combos)} board combo(s) produced track files.")

if __name__ == "__main__":
    main()