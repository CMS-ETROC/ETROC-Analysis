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
MIN_BOARD_COMBO_SIZE = 3  # extract_events_by_path.py (step 8) requires at least 3 boards

# Default for --alignment_core_warn, the --apply_alignment peak-significance
# floor: the share of a combo's candidate weight that must sit within +/-1
# pixel of the modal integer-pixel shift for the estimate to count as a clear
# beam-spot peak. Below it the value is still applied (in the known low-purity
# failure mode the estimate degenerates to ~the yaml value anyway, so applying
# it is no worse than not applying it) but a WARNING flags that the mode may be
# the combinatorial-background peak, not the beam spot, so the number should be
# reviewed before the run is trusted.
# 0.10 is a conservative floor, NOT a calibrated discriminator: a pure
# triangular background can carry ~18 % of its weight in the modal core and
# stay silent. The operator sets the tolerance per campaign with the flag; a
# proper value needs the core-fraction distribution over one real campaign.
# Unused unless --apply_alignment is given.
ALIGNMENT_CORE_WARN_DEFAULT = 0.10

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

    if total_est > MAX_MEMORY_USAGE_MB:
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

def check_spatial_alignment(df: pd.DataFrame, combo: Tuple[int, ...], roles: Dict[str, int], max_diff_pixel: float) -> pd.Series:

    """
    Checks if every board in combo is spatially aligned with a single
    reference board, via Euclidean distance: the trig board when it's in
    this combo (unchanged from before), otherwise combo's own median board
    id -- same fallback used for the global_relative alignment estimate --
    so a combo that doesn't happen to include trig still gets a real
    geometric-coincidence requirement instead of skipping the check
    entirely (previously, looking for a trig board absent from this combo's
    columns always found nothing to compare, silently returning "everything
    passes").
    Logic: r = sqrt((x1-x2)^2 + (y1-y2)^2) <= limit
    """

    trig_id = roles.get('trig')
    if trig_id is not None and trig_id in combo:
        ref_id = trig_id
    else:
        sorted_combo = sorted(combo)
        ref_id = sorted_combo[len(sorted_combo) // 2]
        logging.info(f"No trig board in this combo; using board {ref_id} as the spatial-alignment reference instead.")

    conditions = []
    # Limit calculation: pixels * pitch (mm/pixel)
    limit = max_diff_pixel * PIXEL_PITCH

    for bid in combo:
        if bid == ref_id:
            continue

        # Check if global coordinates exist for both
        if f'x_{ref_id}' in df.columns and f'x_{bid}' in df.columns:

            # Calculate deltas
            dx = df[f'x_{ref_id}'] - df[f'x_{bid}']
            dy = df[f'y_{ref_id}'] - df[f'y_{bid}']

            # Cartesian (Euclidean) Distance Check
            distance = np.sqrt(dx**2 + dy**2)
            conditions.append(distance <= limit)

    if not conditions:
        return pd.Series(True, index=df.index)

    # Combine all conditions (must satisfy distance check for ALL pairs)
    return np.logical_and.reduce(conditions)

def compute_peak_offset(track_candidates: pd.DataFrame, bid: int, ref_id: int) -> Tuple[float, float]:
    """Histogram-peak estimate of (x_bid - x_ref, y_bid - y_ref), weighted by
    each hit pattern's count -- the mode of the shift distribution, not the
    mean, so a handful of outlier tracks can't pull the estimate off the bulk
    of the distribution.
    """
    dx = track_candidates[f'x_{bid}'] - track_candidates[f'x_{ref_id}']
    dy = track_candidates[f'y_{bid}'] - track_candidates[f'y_{ref_id}']

    hist_counts, bin_edges = np.histogram(dx, weights=track_candidates['count'], bins=30)
    max_index = np.argmax(hist_counts)
    center_x = round(float(0.5 * (bin_edges[max_index] + bin_edges[max_index + 1])), 2)

    hist_counts, bin_edges = np.histogram(dy, weights=track_candidates['count'], bins=30)
    max_index = np.argmax(hist_counts)
    center_y = round(float(0.5 * (bin_edges[max_index] + bin_edges[max_index + 1])), 2)

    return center_x, center_y

def corrected_translation(existing: Dict, dx: float, dy: float) -> Dict[str, float]:
    """New translation.{x,y,z} for a board whose current geometry shows an
    observed (dx, dy) offset from its reference -- keeps z untouched."""
    return {
        'x': round(existing.get('x', 0.0) - dx, 2),
        'y': round(existing.get('y', 0.0) - dy, 2),
        'z': round(existing.get('z', 0.0), 2),
    }

def compute_modal_offset(track_candidates: pd.DataFrame, bid: int, ref_id: int,
                         counts_w: np.ndarray) -> Tuple[float, float, float]:
    """Sub-pixel estimate of (x_bid - x_ref, y_bid - y_ref) used by
    --apply_alignment, plus the modal-core weight share that
    --alignment_core_warn tests.

    The shifts are exact multiples of PIXEL_PITCH (both boards use the same
    pixel-centre model), so this takes the count-weighted MODE of the integer
    pixel shift and refines it with the count-weighted centroid over mode +/- 1
    pixel (a sub-pixel estimate from the neighbour asymmetry). It is
    idempotent: re-run on already-corrected geometry it returns ~0.

    This is deliberately NOT compute_peak_offset(). That one histograms the
    continuous shift into 30 bins over its full range; on a full-grid candidate
    set the bin width is ~one pitch, so the modal-bin midpoint is quantized to
    +/-0.65 mm (half a pixel) with a sign set by where the range extremes fall
    -- observed as every estimate coming out exactly +/-0.65 mm, the same board
    flipping sign between combos. That is tolerable for a number a human reads
    out of a diagnostic yaml, and it is what the two diagnostic blocks have
    always reported, so it is left untouched; it is NOT tolerable for a value
    fed back into the geometry, which is why --apply_alignment uses this
    estimator instead. The two therefore do not agree in general.

    The returned core fraction is the min over the two axes of the share of
    candidate weight inside mode +/- 1 pixel: ~1 for a clean beam-spot peak,
    small when the mode is merely the crest of the flat combinatorial
    background (this estimator's low-purity failure mode, in which it returns
    ~0 = "no misalignment").
    """
    centers = []
    core_frac = 1.0
    for axis in ('x', 'y'):
        shift = (track_candidates[f'{axis}_{bid}'] - track_candidates[f'{axis}_{ref_id}']).to_numpy(dtype=float)
        k = np.rint(shift / PIXEL_PITCH).astype(int)               # integer pixel shift per candidate
        ks, inv = np.unique(k, return_inverse=True)
        mode_k = ks[np.argmax(np.bincount(inv, weights=counts_w))]  # count-weighted mode
        core = np.abs(k - mode_k) <= 1                              # mode +/- 1 pixel
        core_frac = min(core_frac, float(counts_w[core].sum() / counts_w.sum()))
        centers.append(float(np.average(shift[core], weights=counts_w[core])))
    return centers[0], centers[1], core_frac

def applied_translation(existing: Dict, dx: float, dy: float) -> Dict[str, float]:
    """corrected_translation()'s arithmetic at the precision --apply_alignment
    feeds forward: new translation = existing minus the measured residual, so
    re-running on already-aligned data leaves the translation unchanged. Kept
    separate from corrected_translation() (2 dp, the diagnostic blocks' long-
    standing rounding) because a value that is actually applied to the geometry
    is quoted to 3 dp = 1 um, below the 10 um scale the estimates repeat to.
    """
    return {
        'x': round(float(existing.get('x', 0.0)) - dx, 3),
        'y': round(float(existing.get('y', 0.0)) - dy, 3),
        'z': round(float(existing.get('z', 0.0)), 3),
    }

def solve_global_relative_alignment(edges: List[Tuple[int, int, float, float]], pin_id: int) -> Dict[int, Tuple[float, float]]:
    """Combines per-combo relative offset measurements (bid, ref_id, dx, dy
    meaning x_bid - x_ref ~= dx, same for y) from every combo into one
    least-squares fit of every board's position, instead of anchoring to a
    single combo's estimate. Solvable for any connected measurement graph
    (unlike the resolution-unfolding sum-of-variances problem, a difference
    system like this one doesn't need an odd cycle to be identifiable) -- but
    only up to one arbitrary global additive constant, since relative offsets
    alone can't fix an absolute origin. pin_id's fitted position is shifted
    to exactly (0, 0) to remove that ambiguity; the choice of which board is
    pinned doesn't affect any board's position *relative* to any other.
    """
    board_ids = sorted({b for e in edges for b in (e[0], e[1])} | {pin_id})
    index = {b: i for i, b in enumerate(board_ids)}
    n = len(board_ids)

    A = np.zeros((len(edges), n))
    bx = np.zeros(len(edges))
    by = np.zeros(len(edges))
    for row, (bid, ref_id, dx, dy) in enumerate(edges):
        A[row, index[bid]] = 1
        A[row, index[ref_id]] = -1
        bx[row] = dx
        by[row] = dy

    x_sol, *_ = np.linalg.lstsq(A, bx, rcond=None)
    y_sol, *_ = np.linalg.lstsq(A, by, rcond=None)

    # Gauge-fix: shift the whole solution so pin_id reads exactly zero.
    x_sol = x_sol - x_sol[index[pin_id]]
    y_sol = y_sol - y_sol[index[pin_id]]

    return {b: (float(x_sol[index[b]]), float(y_sol[index[b]])) for b in board_ids}

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description='Find track candidates and Calibrate.')
    parser.add_argument('-p', '--path', required=True, help='Path to directory with feather files')
    parser.add_argument('--cal-label', required=True, help='Output name for CAL table', dest='cal_label')
    parser.add_argument('--track-label', required=True, help='Output name for Tracks', dest='track_label')
    parser.add_argument('-s', '--sampling', type=float, default=3, help='Sampling fraction (percent)')
    parser.add_argument('--max_diff_pixel', type=int, default=1, help='Max pixel diff')
    parser.add_argument('-c', '--config', required=True, help='YAML config file')
    parser.add_argument('-r', '--runName', required=True, help='Run name in YAML')
    parser.add_argument('--mask_config', type=Path, dest='mask_config_file', help='Mask config YAML')
    parser.add_argument('--cal_table_only', action='store_true', help='Only generate CAL table')
    parser.add_argument('--find_alignment', action='store_true',
                        help='Compute board alignment offsets two ways for comparison, written to a '
                             'diagnostic YAML: "legacy" (each board vs. the trig board, per combo that '
                             'includes trig) and "global_relative" (every combo\'s boards vs. that combo\'s '
                             'own median board id, combined across all combos into one least-squares fit). '
                             'Purely diagnostic -- never mutates the config or affects saved track output. '
                             'Pass --apply_alignment as well to also feed the estimate forward into this '
                             'run\'s in-memory geometry.')
    parser.add_argument('--apply_alignment', action='store_true',
                        help='OPT-IN, requires --find_alignment. Feed the alignment estimate forward: each '
                             'non-trigger board\'s translation is estimated once (count-weighted modal pixel '
                             'shift plus sub-pixel centroid, not the diagnostic histogram estimator) from the '
                             'first trigger-containing combo that holds it, applied to the IN-MEMORY run '
                             'config, and used for every later combo\'s coincidence window. Trigger-containing '
                             'combos are processed first so no combo is cut on uncorrected geometry, a run with '
                             'no board of role "trig" is an error instead of a silent no-op, and the alignment '
                             'YAML gains an "applied" block recording what was used and which combo supplied '
                             'it. Nothing is ever written back to --config. Use it when the board-config yaml '
                             'has no (or wrong) translations; the durable fix is to measure the geometry once '
                             'and put it in the config. Track output DOES change: this is not a diagnostic.')
    parser.add_argument('--alignment_core_warn', type=float, default=ALIGNMENT_CORE_WARN_DEFAULT,
                        help='--apply_alignment peak-significance floor in [0, 1]: warn when less than this '
                             'share of a combo\'s candidate weight sits within +/-1 pixel of the modal shift, '
                             'i.e. the applied value may be the combinatorial-background mode rather than the '
                             'beam spot. Warn-only, never changes what is applied; 0 disables. Ignored without '
                             '--apply_alignment. Default %(default)s, a conservative floor rather than a '
                             'calibrated discriminator -- set it per campaign.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed for every random draw in this script (per-file event sampling, the memory-check '
                             'file sample and the --max_files subset). Default: unseeded, i.e. not reproducible run-to-run.')
    parser.add_argument('--max_files', type=int, default=100,
                        help='Cap on the number of input feather files read; above it a random subset is taken '
                             '(reported in the log). Default 100 keeps the previous hard-coded behaviour; '
                             'pass 0 to read every file.')

    args = parser.parse_args()

    if args.apply_alignment and not args.find_alignment:
        parser.error("--apply_alignment requires --find_alignment (it feeds that estimate forward).")
    if not 0.0 <= args.alignment_core_warn <= 1.0:
        parser.error(f"--alignment_core_warn must be in [0, 1], got {args.alignment_core_warn}")

    # Reproducibility: every random draw below (the per-file event sample in
    # load_and_sample_data, the memory-check file sample and the --max_files
    # subset) goes through the global `random` / `np.random` state. Seeding
    # both makes the CAL table, the candidate lists and the alignment
    # reproducible run-to-run; left unseeded (the default) the outputs differ
    # slightly on every invocation.
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    logging.info(f"Random seed: {args.seed if args.seed is not None else 'none (unseeded, not reproducible)'}")

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

    # sorted(): glob order is filesystem-dependent, and a deterministic order
    # is what makes the --max_files subset reproducible under --seed.
    input_dir = eos_base_dir / args.path
    files = sorted(input_dir.glob('loop*feather'))
    n_found = len(files)
    # Previously a silent, hard-coded `if len(files) > 100: files = random.sample(files, 100)`:
    # runs with more than 100 files had a random subset of their files dropped with no
    # log line. Same default, but configurable, seedable and always reported.
    if args.max_files > 0 and n_found > args.max_files:
        files = random.sample(files, args.max_files)
        logging.warning(f"{n_found} input files found in {input_dir} but --max_files={args.max_files}: "
                        f"reading a random subset of {len(files)} ({n_found - len(files)} not read). "
                        f"Pass --max_files 0 to read all, or --seed N to make the subset reproducible.")
    else:
        logging.info(f"Reading all {n_found} input files from {input_dir}")

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

    # Board combinations to produce tracks for: every subset down to 3 boards
    # (e.g. for 4 boards, that's the four 3-board leave-one-out subsets). Each
    # combo gets its own single-hit requirement -- a board left out of a combo
    # isn't required to have a hit for that combo's tracks. extract_events_by_
    # path.py requires at least 3 boards, so we never go below that.
    #
    # The full board-set combo is deliberately NOT generated: requiring every
    # board to have a hit only ever shrinks the qualifying event set relative
    # to any smaller combo, so the full combo's tracks are already a strict
    # subset of each leave-one-out combo's. apply_tdc_cuts.py/bootstrap.py no
    # longer have an exclude_role mechanism to exploit that extra coincidence
    # requirement, so producing it added a combo with no reachable use.
    # Skip this only when the full set IS the minimum size (nothing smaller
    # exists to subsume it).
    min_boards = min(MIN_BOARD_COMBO_SIZE, len(ids_to_process))
    max_boards = len(ids_to_process) - 1 if len(ids_to_process) > min_boards else len(ids_to_process)
    board_combos = [
        combo
        for size in range(max_boards, min_boards - 1, -1)
        for combo in combinations(ids_to_process, size)
    ]

    values_to_keep = ['row', 'col'] # minimal set for tracking
    trig_id = roles.get('trig')
    id_to_role = {v: k for k, v in roles.items()}
    n_written = 0
    alignment_results = {} # combo_label -> {board_id: translation dict} (legacy, trig-anchored), written once after the loop
    relative_edges = []    # (bid, ref_id, dx, dy) per non-median board per combo (global_relative, median-anchored), combined into one fit after the loop

    # --- --apply_alignment state (all of it inert without the flag) ---
    # board_id -> label of the combo whose estimate was applied to the in-memory
    # run_config for that board. A board is aligned EXACTLY ONCE, from the first
    # trigger-containing combo that contains it.
    #
    # SCOPE: everything here is PER RUN. This script processes one --runName per
    # invocation, the estimate is derived from that run's own candidates, it is
    # applied only to full_config[args.runName], and it is never written back to
    # --config (it goes to the separate alignment yaml). So a telescope geometry
    # change part-way through a campaign (DESY May 2026: new holder from run 5,
    # translations only carried by runs 42/44/45) cannot leak an alignment from
    # one geometry into runs taken in another. The residual risk is a HUMAN one:
    # merging a derived alignment into the board-config yaml by hand and letting
    # it be inherited (YAML anchors) by runs on the other side of a geometry
    # change. Record the geometry validity range next to any merged-in value.
    alignment_source = {}

    # Snapshot of the --config geometry, taken before any feed-forward can touch
    # it. The two diagnostic blocks below are always measured against THIS, so
    # legacy_per_combo and global_relative report the same absolute translations
    # whether or not --apply_alignment is on; without the flag it is simply the
    # unmutated run_config and no copy is ever made.
    baseline_config = {
        bid: {'transformation': {
            'rotation': dict(((conf or {}).get('transformation', {}) or {}).get('rotation', {}) or {}),
            'translation': dict(((conf or {}).get('transformation', {}) or {}).get('translation', {}) or {}),
        }}
        for bid, conf in run_config.items()
    } if args.apply_alignment else run_config

    if args.apply_alignment and trig_id is None:
        # This tests the RUN CONFIG, not any individual combo: `trig_id` is
        # roles.get('trig') for the whole run. Combos that happen not to contain
        # the trigger board are normal and stay fully supported (they use the
        # median-board anchor in check_spatial_alignment). What is caught here is
        # a run in which NO board carries the 'trig' role at all: the estimate is
        # gated on `trig_id in combo`, which is then False for every combo, so
        # --apply_alignment would silently become a no-op -- nothing applied, no
        # warning, every board left on its --config translation. (Plain
        # --find_alignment stays a no-op-plus-global_relative in that case, which
        # is origin's long-standing behaviour and is left alone.)
        logging.error(f"--apply_alignment requires a board with role 'trig'; run {args.runName} "
                      f"defines roles {sorted(roles)}. Nothing would be applied -- exiting.")
        sys.exit(1)

    # PROCESSING order. Unchanged (board_combos' own order) unless
    # --apply_alignment is on. With the flag, trigger-containing combos are
    # processed first: the estimate exists only for combos that contain the
    # trigger board and it is what centres the coincidence window, so a
    # trigger-less combo processed first would be cut on raw, uncorrected
    # geometry. That happens whenever the trigger is the highest-numbered board:
    # with trig = 3 the lexicographically first combo is (0,1,2) and it runs
    # before any estimate exists (CERN IRRAD Jul 2026 h1_run14: boards 0 and 2
    # sit 6.34 mm = 4.9 px apart in x uncorrected, so their true coincidence
    # falls OUTSIDE the 4-pixel window).
    # Running every trigger-containing combo first also makes each combo's own
    # boards aligned by the time it is cut -- a board is either aligned by an
    # earlier combo or by this one -- and by the time the first trigger-less
    # combo is reached every non-trigger board has a translation (a non-trigger
    # board is absent from exactly one of the N-1 leave-one-out combos, so it
    # appears in the others). No combo is ever processed before its geometry is
    # final, so nothing has to be re-cut afterwards.
    # sorted() is stable, so which combo aligns which board is still the
    # lexicographically first one containing it. board_combos itself is left
    # untouched: it is the --combos index order that
    # submit_extract_events_by_path.py (compute_expected_combos) re-derives, and
    # output files are named by combo label, so no index or filename changes.
    combo_order = sorted(board_combos, key=lambda c: trig_id not in c) if args.apply_alignment else board_combos

    # Each combo gets its own file, so with 5+ combos per run they'd otherwise
    # pile up flat in one directory across every run. Nest them under a
    # per-run directory named after --track-label's basename instead --
    # e.g. -t tracks_csv/desy2026aug_run1 writes into
    # tracks_csv/desy2026aug_run1/tracks_<combo>.parquet. The run name isn't
    # repeated in the filename itself since it's already the directory name.
    tracks_out_dir = Path(args.track_label)
    tracks_out_dir.mkdir(parents=True, exist_ok=True)

    for combo in combo_order:
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

        # 6. Geometric Transformation & Final Filtering
        apply_geometric_transformation_matrix(track_candidates, combo, run_config)

        # Alignment offsets are a property of the boards themselves, but the
        # estimate itself still depends on which combo's tracks it's computed
        # from -- so compute (and record) it for every eligible combo, keyed
        # by combo, instead of overwriting one shared result. Both methods
        # below are purely diagnostic: neither ever mutates run_config /
        # full_config or affects this combo's saved track output, so track
        # output is identical regardless of --find_alignment. (--apply_alignment
        # is the separate, opt-in block further down that DOES change the
        # geometry; the two diagnostic blocks here are unaffected by it because
        # they read `diag_candidates`/`baseline_config`, i.e. the --config
        # geometry, in both modes.)
        if args.find_alignment:
            if args.apply_alignment and alignment_source:
                # A previous combo already fed its estimate into run_config, so
                # track_candidates is in corrected geometry and would measure a
                # residual. Re-transform a copy on the --config geometry so the
                # diagnostic blocks keep reporting the same absolute numbers
                # they report without the flag. (No copy on the first combo, or
                # ever when the flag is off: nothing has been applied yet.)
                diag_candidates = track_candidates.copy(deep=True)
                apply_geometric_transformation_matrix(diag_candidates, combo, baseline_config)
            else:
                diag_candidates = track_candidates

            # Legacy method: every board's offset relative to the trig board
            # -- only computable for combos that include it.
            if trig_id in combo:
                combo_alignment = {}
                for bid in combo:
                    if bid == trig_id:
                        continue
                    center_x, center_y = compute_peak_offset(diag_candidates, bid, trig_id)
                    existing = baseline_config.get(bid, {}).get('transformation', {}).get('translation', {'x': 0.0, 'y': 0.0, 'z': 0.0})
                    combo_alignment[bid] = corrected_translation(existing, center_x, center_y)
                alignment_results[combo_label] = combo_alignment

            # global_relative method: every board's offset relative to *this
            # combo's own* median board id -- doesn't need trig, so it runs
            # for every combo. Collected here and solved once, globally,
            # after the loop instead of being anchored to any single board or
            # combo.
            sorted_combo = sorted(combo)
            median_id = sorted_combo[len(sorted_combo) // 2]
            for bid in combo:
                if bid == median_id:
                    continue
                dx, dy = compute_peak_offset(diag_candidates, bid, median_id)
                relative_edges.append((bid, median_id, dx, dy))

        # --- Alignment feed-forward (--apply_alignment, opt-in) ---
        # Estimates each non-trigger board's translation relative to the trigger
        # board from this combo's candidates and APPLIES it, so that the
        # coincidence window in check_spatial_alignment is centred on where
        # tracks actually land.
        # What the number IS: the count-weighted mean *projected* offset of
        # tracks between the two boards. It absorbs mechanical misalignment AND
        # the mean track angle times the lever arm between the planes (non-zero
        # for an off-axis telescope, e.g. the IRRAD setups) -- which is exactly
        # the quantity that centres the window; it is not a purely mechanical
        # survey number.
        #
        # Feed-forward is PER BOARD: each board's estimate is applied to the
        # in-memory run_config EXACTLY ONCE, from the first trigger-containing
        # combo that contains that board, and a board is never re-estimated on
        # top of its own already-applied translation. Two earlier designs failed
        # here and both are ruled out by construction now:
        #   - Applying on every combo (`len(combo) == max_boards`, true for
        #     EVERY generated combo once the full board-set combo stopped being
        #     generated) made each combo re-estimate on top of the previous
        #     combo's shift and overwrite it: order-dependent, oscillating.
        #   - A single run-level "applied" latch, firing on the first
        #     trigger-containing combo, went too far the other way: that combo
        #     holds only N-2 of the N-1 non-trigger boards, so on a 4-board
        #     telescope exactly one board kept its config translation for the
        #     whole run. With an all-zero config (DESY Aug 2026) that board's
        #     window stayed centred on zero while the board itself sat ~10 px
        #     away, so every combo containing it kept only combinatorial
        #     background (run 23 trig1-ref2-extra3: 1864 counts over 1840
        #     surviving patterns, 1.01 events each, against 4.24 for the aligned
        #     combo).
        # For a board an earlier combo already aligned, this combo's estimate is
        # a RESIDUAL cross-check and a disagreement is logged, not acted on.
        # Nothing is ever written back to --config.
        if args.apply_alignment and trig_id in combo:
            counts_w = track_candidates['count'].to_numpy(dtype=float)
            applied_here = {}
            combo_core = {}  # board_id -> min over axes of the modal-core weight share
            for bid in combo:
                if bid == trig_id:
                    continue
                existing = run_config.get(bid, {}).get('transformation', {}).get('translation', {}) or {}
                center_x, center_y, core_frac = compute_modal_offset(track_candidates, bid, trig_id, counts_w)
                applied_here[bid] = applied_translation(existing, center_x, center_y)
                combo_core[bid] = core_frac

            for bid in sorted(set(applied_here) & set(alignment_source)):
                # Already aligned by an earlier combo, so this combo's candidates
                # were transformed with that translation and what was just
                # measured is a RESIDUAL: applied_here[bid] should reproduce the
                # applied value. A real disagreement means the two combos do not
                # see the same board position -- typically one of the two modes
                # latched onto combinatorial background instead of the beam spot.
                # Reported, not acted on: the first estimate stands.
                already = run_config[bid].get('transformation', {}).get('translation', {}) or {}
                residual = max(abs(float(applied_here[bid][a]) - float(already.get(a, 0.0))) for a in ('x', 'y'))
                emit = logging.warning if residual > PIXEL_PITCH else logging.info
                emit(f"Combo ({combo_label}): board {bid} already aligned from ({alignment_source[bid]}) "
                     f"as {dict(already)}; this combo re-measures {applied_here[bid]} "
                     f"(max residual {residual:.3f} mm). Not re-applied.")

            newly_aligned = {bid: t for bid, t in applied_here.items() if bid not in alignment_source}
            if newly_aligned:
                for bid, new_translation in newly_aligned.items():
                    run_config[bid].setdefault('transformation', {})['translation'] = new_translation
                    alignment_source[bid] = combo_label
                logging.info(f"Alignment from combo ({combo_label}) applied in-memory to run_config "
                             f"(not saved to {args.config}): {newly_aligned}")
                for bid in sorted(newly_aligned):
                    if combo_core[bid] < args.alignment_core_warn:
                        logging.warning(f"Combo ({combo_label}): board {bid}'s modal pixel +/-1 carries only "
                                        f"{combo_core[bid]:.1%} of the candidate weight (floor "
                                        f"--alignment_core_warn {args.alignment_core_warn:.2f}), so the applied translation "
                                        f"{newly_aligned[bid]} is not a clear peak -- it may be the "
                                        f"combinatorial-background mode, not the beam spot. Review the per-combo "
                                        f"cross-checks in the alignment yaml before trusting combos that contain "
                                        f"this board.")
                # Re-transform this combo's own candidates so its cut below
                # already uses the corrected geometry. Boards aligned by an
                # earlier combo are recomputed from row/col with the translation
                # they already had, so this is idempotent for them.
                apply_geometric_transformation_matrix(track_candidates, combo, run_config)

        spatial_mask = check_spatial_alignment(track_candidates, combo, roles, args.max_diff_pixel)
        final_tracks = track_candidates[spatial_mask]

        # Remove duplicates if any remain based on pattern
        final_tracks = final_tracks.drop_duplicates(subset=group_cols)

        coord_cols = [c for c in final_tracks.columns if c.split('_')[0] in ['x', 'y', 'z']]
        final_tracks[coord_cols] = final_tracks[coord_cols].round(2)

        output_file = tracks_out_dir / f'tracks_{combo_label}.parquet'
        io_utils.write_parquet(final_tracks, output_file, index=False)
        logging.info(f"Combo ({combo_label}): {len(final_tracks)} tracks saved to {output_file}")
        n_written += 1

    if args.apply_alignment:
        unaligned = [b for b in ids_to_process
                     if b != trig_id and b in boards_with_hits and b not in alignment_source]
        if unaligned:
            logging.warning(f"--apply_alignment: board(s) {unaligned} never received a derived translation "
                            f"(no trigger-containing combo produced candidates for them). They kept their "
                            f"--config translation, so every combo containing them was cut on that geometry.")

    if alignment_results or relative_edges:
        # Own directory, separate from wherever --track-label's tracks/cal_table
        # output goes, since alignment output is a different kind of artifact
        # (a diagnostic to review/merge by hand, not pipeline input).
        alignment_dir = Path('alignment')
        alignment_dir.mkdir(parents=True, exist_ok=True)
        alignment_file = alignment_dir / f'{Path(args.track_label).name}_alignment.yaml'

        output = {}
        if alignment_results:
            output['legacy_per_combo'] = {
                combo_label: {bid: {'transformation': {'translation': t}} for bid, t in combo_vals.items()}
                for combo_label, combo_vals in alignment_results.items()
            }
        if relative_edges:
            # Any board works as the gauge pin -- trig is used here only so
            # the reported numbers line up with legacy_per_combo's convention
            # (both report "what to add to the pinned/trig board's existing
            # translation" for every other board), making the two sections
            # directly comparable board-by-board.
            pin_id = trig_id if trig_id is not None else min(ids_to_process)
            fitted = solve_global_relative_alignment(relative_edges, pin_id)
            global_alignment = {}
            for bid, (fx, fy) in fitted.items():
                if bid == pin_id:
                    continue
                existing = baseline_config.get(bid, {}).get('transformation', {}).get('translation', {'x': 0.0, 'y': 0.0, 'z': 0.0})
                global_alignment[bid] = corrected_translation(existing, fx, fy)
            output['global_relative'] = {
                'pinned_board': pin_id,
                'boards': {bid: {'transformation': {'translation': t}} for bid, t in global_alignment.items()},
            }

        # --apply_alignment only: what the tracks in this run were ACTUALLY cut
        # with, and which combo supplied it. The two diagnostic blocks alone
        # cannot say: they report every combo's own measurement against the
        # --config geometry, with nothing in the numbers to mark which one was
        # applied. dict() copies the values so ruamel emits them plainly instead
        # of as aliases to the run_config entries.
        if alignment_source:
            output['applied'] = {
                bid: {'from_combo': alignment_source[bid],
                      'transformation': {'translation': dict(
                          run_config[bid]['transformation']['translation'])}}
                for bid in sorted(alignment_source)
            }

        with open(alignment_file, 'w') as f:
            yaml.dump({args.runName: output}, f)
        logging.info(f"Alignment comparison (legacy_per_combo vs. global_relative) written to {alignment_file} "
                     f"(not saved back to {args.config} -- merge in manually if desired).")
        if alignment_source:
            logging.info(f"--apply_alignment: 'applied' block in {alignment_file} records the translation each "
                         f"board's tracks were actually cut with, and the combo it came from.")

    if n_written == 0:
        logging.warning("No track candidates found for any board combo.")
        sys.exit(0)

    logging.info(f"Done. {n_written}/{len(board_combos)} board combo(s) produced track files.")

if __name__ == "__main__":
    main()