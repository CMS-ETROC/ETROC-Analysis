import argparse
import sys
import logging
import warnings
import random
import ruamel.yaml
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from functools import reduce
from typing import List, Dict, Tuple, Optional

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore")

# --- Constants & Defaults ---
PIXEL_PITCH = 1.3
PIXEL_OFFSET = 7.5
MAX_MEMORY_USAGE_MB = 2600
SINGLE_FILE_MEMORY_LIMIT_MB = 26

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

def apply_geometric_transformation(df: pd.DataFrame, board_ids: List[int], config: dict):

    """
    Applies rotation and translation to coordinate columns in place.
    """

    for bid in board_ids:
        # Check if columns exist before processing
        if f'col_{bid}' not in df.columns or f'row_{bid}' not in df.columns:
            continue

        # Local coordinates
        x_prime = (df[f'col_{bid}'] - PIXEL_OFFSET) * PIXEL_PITCH
        y_prime = (df[f'row_{bid}'] - PIXEL_OFFSET) * PIXEL_PITCH

        rot, tra = get_transformation_params(bid, config)
        rx, ry, rz = rot
        tx, ty, tz = tra

        # Rotation Matrix Application (Z-Y-X sequence based on original code logic)
        # Note: Optimization could be done using matrix multiplication, but keeping
        # explicit expansion to match original logic exactly.

        cos_rx, sin_rx = np.cos(rx), np.sin(rx)
        cos_ry, sin_ry = np.cos(ry), np.sin(ry)
        cos_rz, sin_rz = np.cos(rz), np.sin(rz)

        df[f'x_{bid}'] = (x_prime * (cos_rz * cos_ry) +
                          y_prime * (cos_rz * sin_ry * sin_rx - sin_rz * cos_rx) +
                          tx)

        df[f'y_{bid}'] = (x_prime * (sin_rz * cos_ry) +
                          y_prime * (sin_rz * sin_ry * sin_rx + cos_rz * cos_rx) +
                          ty)

        df[f'z_{bid}'] = (x_prime * (-sin_ry) +
                          y_prime * (cos_ry * sin_rx) +
                          tz)

# --- Core Logic Blocks ---

def load_and_sample_data(file_paths: List[Path], sampling_rate: float) -> pd.DataFrame:

    """
    Loads feather files, performs memory checks, and concatenates data.
    """

    columns_to_read = ['evt', 'board', 'row', 'col', 'toa', 'tot', 'cal']
    portion = sampling_rate * 0.01

    # 1. Memory Safety Check
    logging.info('Performing Memory Safety Check...')
    check_files = file_paths if len(file_paths) < 10 else random.sample(file_paths, 10)
    sum_use = 0

    for f in check_files:
        temp_df = pd.read_feather(f, columns=columns_to_read)
        # Simulate sampling
        n = int(portion * temp_df['evt'].nunique())
        if n > 0:
            indices = np.random.choice(temp_df['evt'].unique(), n, replace=False)
            temp_df = temp_df.loc[temp_df['evt'].isin(indices)]
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
        tmp_df = pd.read_feather(f, columns=columns_to_read)
        n = int(portion * tmp_df['evt'].nunique())
        if n > 0:
            indices = np.random.choice(tmp_df['evt'].unique(), n, replace=False)
            tmp_df = tmp_df.loc[tmp_df['evt'].isin(indices)]
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

    with open(mask_config_path, 'r') as f:
        mask_info = yaml.safe_load(f)

    for board_id, val in mask_info.get("board_ids", {}).items():
        pixels = val.get('pixels', [])
        if not pixels:
            continue

        # Vectorized masking is faster than iterating
        # Create a boolean mask
        for (r, c) in pixels:
             df = df[~((df['board'] == board_id) & (df['row'] == r) & (df['col'] == c))]

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
    cal_table.to_csv(f'{output_name}_cal_table.csv', index=False)
    return cal_table

def filter_by_tdc(df: pd.DataFrame, cuts: Dict[int, List[int]]) -> pd.DataFrame:

    """
    Filters events based on TDC/ToA/ToT cuts.
    """

    # Create masks per board
    masks = {}
    for board, c in cuts.items():
        # c = [cal_min, cal_max, toa_min, toa_max, tot_min, tot_max]
        mask = (
            (df['board'] == board) &
            df['cal'].between(c[0], c[1]) &
            df['toa'].between(c[2], c[3]) &
            df['tot'].between(c[4], c[5])
        )
        # Store the valid events for this board
        masks[board] = df.loc[mask, 'evt'].unique()

    # Find intersection of events present in ALL required boards logic could be complex
    # The original logic used intersection of unique events
    common_events = reduce(np.intersect1d, list(masks.values()))
    return df.loc[df['evt'].isin(common_events)].reset_index(drop=True)

def check_spatial_alignment(df: pd.DataFrame, roles: Dict[str, int], max_diff_pixel: float) -> pd.Series:

    """
    Checks if hits are spatially aligned across boards using Euclidean distance.
    Logic: r = sqrt((x1-x2)^2 + (y1-y2)^2) <= limit
    """

    trig_id = roles.get('trig')

    # We need a trigger board to compare against
    if trig_id is None:
        logging.warning("No 'trig' role found. Skipping spatial alignment check.")
        return pd.Series([True] * len(df))

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
        return pd.Series([True] * len(df))

    # Combine all conditions (must satisfy distance check for ALL pairs)
    return np.logical_and.reduce(conditions)

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description='Find track candidates and Calibrate.')
    parser.add_argument('-p', '--path', required=True, help='Path to directory with feather files')
    parser.add_argument('--out_calname', required=True, help='Output name for CAL table')
    parser.add_argument('--out_trackname', required=True, help='Output name for Tracks')
    parser.add_argument('-s', '--sampling', type=float, default=3, help='Sampling fraction (percent)')
    parser.add_argument('-m', '--minimum', type=int, default=1000, dest='ntracks', help='Min tracks')
    parser.add_argument('--max_diff_pixel', type=int, default=1, help='Max pixel diff')
    parser.add_argument('-c', '--config', required=True, help='YAML config file')
    parser.add_argument('-r', '--runName', required=True, help='Run name in YAML')
    parser.add_argument('--mask_config', type=Path, dest='mask_config_file', help='Mask config YAML')
    parser.add_argument('--exclude_role', help='Role to exclude')
    parser.add_argument('--cal_table_only', action='store_true', help='Only generate CAL table')
    parser.add_argument('--find_alignment', action='store_true', help='Find the board offset alignments refer to trigger board')

    args = parser.parse_args()

    # 1. Setup & Config
    with open(args.config) as f:
        full_config = ruamel.yaml.load(f, Loader=ruamel.yaml.RoundTripLoader)

    if args.runName not in full_config:
        raise ValueError(f"Run config {args.runName} not found")

    run_config = full_config[args.runName]

    # Map roles to IDs
    roles = {info['role']: bid for bid, info in run_config.items()}

    # 2. Load Data
    files = list(Path(args.path).glob('loop*feather'))
    if len(files) > 100: files = files[:100]

    if not files:
        logging.error("No input files found.")
        sys.exit(1)

    df = load_and_sample_data(files, args.sampling)

    # 3. Preprocessing
    if args.mask_config_file:
        df = apply_masking(df, args.mask_config_file)

    if args.exclude_role and args.exclude_role in roles:
        ex_id = roles[args.exclude_role]
        logging.info(f"Dropping board role: {args.exclude_role} (ID: {ex_id})")
        df = df[df['board'] != ex_id].reset_index(drop=True)
        # Remove from roles dict to prevent downstream errors
        del roles[args.exclude_role]

    # 4. Calibration
    cal_table = generate_cal_table(df, args.out_calname)

    if args.cal_table_only:
        logging.info("Cal table only mode. Exiting.")
        sys.exit(0)

    # 5. Track Finding
    logging.info('Starting track finding...')

    # Filter based on CAL deviations
    merged = pd.merge(df[['board', 'row', 'col', 'cal', 'evt']], cal_table, on=['board', 'row', 'col'])
    valid_cal = abs(merged['cal'] - merged['cal_mode']) <= 3
    df = df.loc[valid_cal].reset_index(drop=True)
    df = reindex_events(df) # Renumber after filtering
    check_empty_df(df, "CAL deviation filtering")

    # Define TDC Cuts (Moved from hardcoded logic)
    # Default fallback logic from original code
    ids_to_process = sorted(roles.values())
    tdc_cuts = {}
    for idx in ids_to_process:
        if idx == roles.get('trig'):
            tdc_cuts[idx] = [0, 1100, 100, 500, 50, 250]
        elif idx == roles.get('ref'):
            tdc_cuts[idx] = [0, 1100, 0, 1100, 50, 250]
        else:
            tdc_cuts[idx] = [0, 1100, 0, 1100, 0, 600]

    df = filter_by_tdc(df, tdc_cuts)
    check_empty_df(df, "TDC filtering")

    # Single Hit Selection
    # Ensure every required board has exactly 1 hit
    req_boards = [b for b in ['trig', 'ref', 'dut', 'extra'] if b in roles]

    # Group by event and board to count hits
    counts = df.groupby(['evt', 'board']).size().unstack(fill_value=0)

    # Create mask where all required boards have exactly 1 hit
    valid_event_mask = np.logical_and.reduce([counts[roles[r]] == 1 for r in req_boards])
    valid_events = counts[valid_event_mask].index

    df = df.loc[df['evt'].isin(valid_events)].reset_index(drop=True)
    df[['row', 'col']] = df[['row', 'col']].astype('int8') # Optimization
    check_empty_df(df, "Single hit filtering")

    # Pivot to Wide Format (Events as rows, Boards as columns)
    # Prepare columns to keep
    values_to_keep = ['row', 'col'] # minimal set for tracking

    # Pivot
    track_df = df.pivot(index='evt', columns='board', values=values_to_keep)
    # Flatten columns: (row, 0) -> row_0
    track_df.columns = [f"{v}_{b}" for v, b in track_df.columns]

    # Group identical hit patterns (finding "Hot Tracks" or frequent combinations)
    # Identify grouping columns (row_X, col_X for all boards)
    group_cols = list(track_df.columns)

    track_candidates = track_df.groupby(group_cols).size().reset_index(name='count')
    track_candidates = track_candidates[track_candidates['count'] > args.ntracks]

    if track_candidates.empty:
        logging.warning("No track candidates found above threshold.")
        sys.exit(0)

    # 6. Geometric Transformation & Final Filtering
    apply_geometric_transformation(track_candidates, ids_to_process, run_config)

    if args.find_alignment:
        shift_df = track_candidates.copy(deep=True)
        trig_id = roles['trig']
        for bid in range(4):
            if bid == trig_id:
                continue

            shift_df[f'x_{bid}'] = shift_df[f'x_{bid}'] - shift_df[f'x_{trig_id}']
            shift_df[f'y_{bid}'] = shift_df[f'y_{bid}'] - shift_df[f'y_{trig_id}']

            counts, bin_edges = np.histogram(shift_df[f'x_{bid}'], weights=shift_df['count'], bins=30)
            max_index = np.argmax(counts)
            max_bin_start = bin_edges[max_index]
            max_bin_end = bin_edges[max_index + 1]

            center_x = round(float(0.5*(max_bin_end+max_bin_start)), 2)

            counts, bin_edges = np.histogram(shift_df[f'y_{bid}'], weights=shift_df['count'], bins=30)
            max_index = np.argmax(counts)
            max_bin_start = bin_edges[max_index]
            max_bin_end = bin_edges[max_index + 1]

            center_y = round(float(0.5*(max_bin_end+max_bin_start)), 2)

            full_config[args.runName][bid]['transformation']['translation']['x'] += -center_x
            full_config[args.runName][bid]['transformation']['translation']['y'] += -center_y

        with open(args.config, 'w') as f:
            ruamel.yaml.dump(full_config, f, Dumper=ruamel.yaml.RoundTripDumper)

        run_config = full_config[args.runName]
        apply_geometric_transformation(track_candidates, ids_to_process, run_config)

    spatial_mask = check_spatial_alignment(track_candidates, roles, args.max_diff_pixel)
    final_tracks = track_candidates[spatial_mask]

    # Remove duplicates if any remain based on pattern
    final_tracks = final_tracks.drop_duplicates(subset=group_cols)

    output_file = f'{args.out_trackname}_tracks.csv'
    final_tracks.to_csv(output_file, index=False)
    logging.info(f"Done. Tracks saved to {output_file}")

if __name__ == "__main__":
    main()