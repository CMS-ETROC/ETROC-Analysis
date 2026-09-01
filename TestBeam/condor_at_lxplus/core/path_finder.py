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
# Default for --alignment_core_warn, the --find_alignment peak-significance
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

def check_spatial_alignment(df: pd.DataFrame, roles: Dict[str, int], max_diff_pixel: float) -> pd.Series:

    """
    Coincidence-consistency (radius) cut on one combo's track candidates.

    For every non-anchor board present in the combo, the transformed
    pixel-centre position must lie within a circle of radius
    max_diff_pixel * PIXEL_PITCH around the anchor board's pixel centre:
        r = sqrt((x_anchor - x_other)^2 + (y_anchor - y_other)^2) <= limit
    This is deliberately a window, not a straight-line fit: it tolerates
    inclined tracks and mechanical/projective offsets up to the window size,
    and rejects the random coincidences (a real hit on two boards plus noise,
    an after-pulse or a second particle on the third) whose pixel pattern is
    geometrically impossible for a single particle.

    Anchor choice: the trigger board when it is part of this combo (the
    established behaviour); otherwise the first present board in the order
    ref, dut, extra. Before this fallback existed, a combo WITHOUT the trigger
    board (e.g. the 'extra0-ref1-dut2' leave-one-out combo of a 4-board
    telescope) received no cut at all: `x_{trig_id}` was not among its
    columns, no condition was ever built, and the function returned all-True
    without any warning. That combo therefore carried every combinatorial
    background pattern into the downstream steps -- observed on real data as
    ~2.5x more candidates than the trigger-containing combos, 20-50% of them
    geometrically inconsistent -- and since step 7's coverage-based selection
    deliberately picks low-count candidates, that background was liable to be
    selected as tracks. Relative offsets between two non-trigger boards are
    already correct whenever each board's own yaml translation (defined
    relative to the trigger board) is correct, so the fallback anchor needs
    no additional alignment information.
    """

    # Boards actually present in this combo = those with transformed coordinates
    # (apply_geometric_transformation_matrix only adds x_/y_ for combo boards).
    present = {role: bid for role, bid in roles.items() if f'x_{bid}' in df.columns}
    if not present:
        logging.warning("No transformed coordinates found for any board. Skipping spatial alignment check.")
        return pd.Series(True, index=df.index)

    anchor_role = next((r for r in ['trig', 'ref', 'dut', 'extra'] if r in present), None)
    if anchor_role is None:
        # Non-standard role names only: fall back to the first present board.
        anchor_role = sorted(present)[0]
    anchor_id = present[anchor_role]
    if anchor_role != 'trig':
        logging.info(f"Trigger board not in this combo: spatial alignment check anchored on "
                     f"'{anchor_role}' (board {anchor_id}) instead.")

    # Limit calculation: pixels * pitch (mm/pixel)
    limit = max_diff_pixel * PIXEL_PITCH

    conditions = []
    for role_name, other_id in present.items():
        if other_id == anchor_id:
            continue
        dx = df[f'x_{anchor_id}'] - df[f'x_{other_id}']
        dy = df[f'y_{anchor_id}'] - df[f'y_{other_id}']
        # Cartesian (Euclidean) Distance Check
        distance = np.sqrt(dx**2 + dy**2)
        conditions.append(distance <= limit)

    if not conditions:
        # Only one board present -- nothing to compare against. Cannot happen
        # while MIN_BOARD_COMBO_SIZE >= 2; kept as a guard.
        return pd.Series(True, index=df.index)

    # Combine all conditions (must satisfy distance check for ALL pairs)
    return np.logical_and.reduce(conditions)

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description='Find track candidates and Calibrate.')
    parser.add_argument('-p', '--path', required=True,
                        help='Directory with the input feather files. A relative path is resolved under your EOS '
                             'base (/eos/user/<u>/<user>/); an absolute path is used as-is (e.g. another user\'s EOS area).')
    parser.add_argument('--cal-label', required=True, dest='cal_label',
                        help='Output PREFIX for the CAL table: writes <CAL_LABEL>_cal_table.csv, '
                             'relative to the current working directory (not to EOS).')
    parser.add_argument('--track-label', required=True, dest='track_label',
                        help='Output DIRECTORY for the per-combo track-candidate parquet files '
                             '(<TRACK_LABEL>/tracks_<combo>.parquet), relative to the current working directory (not to EOS).')
    parser.add_argument('-s', '--sampling', type=float, default=3, help='Sampling fraction (percent)')
    parser.add_argument('--max_diff_pixel', type=int, default=1, help='Max pixel diff')
    parser.add_argument('-c', '--config', required=True, help='YAML config file')
    parser.add_argument('-r', '--runName', required=True, help='Run name in YAML')
    parser.add_argument('--mask_config', type=Path, dest='mask_config_file', help='Mask config YAML')
    parser.add_argument('--cal_table_only', action='store_true', help='Only generate CAL table')
    parser.add_argument('--find_alignment', action='store_true', help='Find the board offset alignments refer to trigger board')
    parser.add_argument('--alignment_core_warn', type=float, default=ALIGNMENT_CORE_WARN_DEFAULT,
                        help='--find_alignment peak-significance floor in [0, 1]: warn when less than this share of a '
                             'combo\'s candidate weight sits within +/-1 pixel of the modal shift, i.e. when the applied '
                             'translation may be the combinatorial-background mode rather than the beam spot. Warn-only: '
                             'the value is still applied. 0 disables the warning. Default %(default)s is a conservative '
                             'floor, not a calibrated discriminator -- set it per campaign.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed for every random draw in this script (per-file event sampling, the memory-check '
                             'file sample and the --max_files subset). Default: unseeded, i.e. NOT reproducible run-to-run.')
    parser.add_argument('--max_files', type=int, default=100,
                        help='Cap on the number of input feather files read; above it a random subset is taken '
                             '(reported in the log). Default 100 keeps the previous hard-coded behaviour; '
                             'pass 0 to read every file.')

    args = parser.parse_args()
    if not 0.0 <= args.alignment_core_warn <= 1.0:
        parser.error(f"--alignment_core_warn must be in [0, 1], got {args.alignment_core_warn}")

    # Reproducibility: every random draw below (the per-file event sample in
    # load_and_sample_data, the <=10-file memory-check sample, and the
    # --max_files subset) goes through the global `random` / `np.random`
    # state. Seeding both makes the CAL table, the candidate lists and any
    # --find_alignment estimate bit-reproducible for a given input set, which
    # is what turns a re-run after a code change into a controlled comparison.
    # Left unseeded (the default) the outputs differ slightly on every
    # invocation, exactly as before this option existed.
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    logging.info(f"Random seed: {args.seed if args.seed is not None else 'none (unseeded -- not reproducible)'}")

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

    # Note: pathlib discards eos_base_dir when args.path is absolute, so an
    # absolute -p reads any location (e.g. a colleague's EOS area) unchanged.
    # sorted(): glob order is filesystem-dependent, and a deterministic order
    # is what makes the --max_files subset reproducible under --seed.
    input_dir = eos_base_dir / args.path
    files = sorted(input_dir.glob('loop*feather'))
    n_found = len(files)
    # Previously a silent, hard-coded `if len(files) > 100: files = random.sample(files, 100)`:
    # runs with >100 files had a random ~40% of their files dropped with no log
    # line and no way to reproduce which ones. Same default cap, but now
    # configurable, seedable and always reported.
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
    alignment_results = {} # combo_label -> {board_id: translation dict}, written once after the loop
    # --find_alignment: board_id -> label of the combo whose estimate was applied
    # to the in-memory run_config for that board. A board is aligned EXACTLY
    # ONCE, from the first trigger-containing combo that contains it (see below).
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

    if args.find_alignment and trig_id is None:
        # This tests the RUN CONFIG, not any individual combo: `trig_id` is
        # roles.get('trig') for the whole run. Combos that happen not to contain
        # the trigger board are normal and stay fully supported (they use the
        # anchor fallback in check_spatial_alignment). What is caught here is a
        # run in which NO board carries the 'trig' role at all.
        # The estimate below is gated on `trig_id in combo`, which is False for
        # every combo when no board has role 'trig' -- --find_alignment would
        # silently become a no-op: no estimate, no alignment yaml, no warning,
        # and every board left on its --config translation.
        logging.error(f"--find_alignment requires a board with role 'trig'; run {args.runName} "
                      f"defines roles {sorted(roles)}. Nothing would be estimated -- exiting.")
        sys.exit(1)

    # PROCESSING order -- deliberately not board_combos' own order. The alignment
    # estimate exists only for combos that contain the trigger board, and it is
    # what centres the radius cut, so a trigger-less combo processed first is cut
    # on raw, uncorrected geometry. That is today's behaviour whenever the trigger
    # is the highest-numbered board: with trig = 3 the lexicographically first
    # combo is (0,1,2) and it runs before any estimate exists (CERN IRRAD Jul 2026
    # h1_run14: boards 0 and 2 sit 6.34 mm = 4.9 px apart in x uncorrected, so
    # their true coincidence falls OUTSIDE the 4-pixel window).
    # Running every trigger-containing combo first also makes each combo's own
    # boards aligned by the time it is cut -- a board is either aligned by an
    # earlier combo or by this one -- and by the time the first trigger-less combo
    # is reached every non-trigger board has a translation (a non-trigger board is
    # absent from exactly one of the N-1 leave-one-out combos, so it appears in
    # the others). No combo is ever processed before its geometry is final, so
    # nothing has to be re-cut afterwards.
    # sorted() is stable, so which combo aligns which board is still the
    # lexicographically first one containing it. board_combos itself is left
    # untouched: it is the --combos index order that
    # submit_extract_events_by_path.py (compute_expected_combos) re-derives, and
    # output files are named by combo label, so no index or filename changes.
    combo_order = sorted(board_combos, key=lambda c: trig_id not in c)

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

        # --- Alignment estimate (--find_alignment) ---
        # Estimates each non-trigger board's translation relative to the
        # trigger board from this combo's candidates, so that the radius cut in
        # check_spatial_alignment is centred on where tracks actually land.
        # What the number IS: the count-weighted mean *projected* offset of
        # tracks between the two boards. It absorbs mechanical misalignment AND
        # the mean track angle times the lever arm between the planes (non-zero
        # for an off-axis telescope, e.g. the IRRAD setups) -- which is exactly
        # the quantity that centres the window; it is not a purely mechanical
        # survey number.
        #
        # Estimator (changed): the shifts x_bid - x_trig are exact multiples of
        # PIXEL_PITCH (both boards use the same pixel-centre model), so we take
        # the count-weighted MODE of the integer pixel shift and refine it with
        # the count-weighted centroid over mode +/- 1 pixel (a sub-pixel
        # estimate from the neighbour asymmetry). The previous implementation
        # histogrammed the continuous shift with bins=30 over its full range;
        # on a full-grid candidate set that bin width is ~one pitch, so the
        # modal-bin midpoint was quantized to +/-0.65 mm (half a pixel) with a
        # sign decided by where the range extremes fell -- observed as every
        # estimate being exactly +/-0.65 mm, the same board flipping sign
        # between combos.
        #
        # Feed-forward (per board): each board's estimate is applied to the
        # in-memory run_config EXACTLY ONCE, from the first trigger-containing
        # combo that contains that board, and a board is never re-estimated on
        # top of its own already-applied translation.
        #   - The condition before f7d135a, `len(combo) == max_boards`, became
        #     true for EVERY generated combo once the full board-set combo
        #     stopped being generated (max_boards == N-1), so each combo
        #     re-estimated on top of the previous combo's shift and overwrote it
        #     (order-dependent, oscillating values). The per-board record keeps
        #     that fixed: an already-aligned board is skipped below, never
        #     re-applied.
        #   - f7d135a's replacement, one global "applied" flag fired on the first
        #     trigger-containing combo, went too far the other way: that combo
        #     holds only N-2 of the N-1 non-trigger boards, so on a 4-board
        #     telescope exactly one board kept its config translation for the
        #     whole run. With an all-zero config (DESY Aug 2026) that board's
        #     radius window stayed centred on zero while the board itself sat
        #     ~10 px away, so every combo containing it kept only combinatorial
        #     background (run23 trig1-ref2-extra3: 1864 counts over 1840
        #     surviving patterns, 1.01 events each, against 4.24 for the
        #     aligned combo).
        # Every combo's estimate is still recorded, keyed by combo, so they can be
        # cross-checked in the alignment yaml; for a board an earlier combo
        # already aligned the estimate is a RESIDUAL check, and a disagreement is
        # now logged instead of passing silently. Nothing is written back to
        # --config.
        if args.find_alignment and trig_id in combo:
            counts_w = track_candidates['count'].to_numpy(dtype=float)
            combo_alignment = {}
            combo_core = {}  # board_id -> min over axes of the modal-core weight share
            for bid in combo:
                if bid == trig_id:
                    continue
                existing = run_config.get(bid, {}).get('transformation', {}).get('translation', {}) or {}
                combo_alignment[bid] = {}
                core_frac = 1.0
                for axis in ('x', 'y'):
                    shift = (track_candidates[f'{axis}_{bid}'] - track_candidates[f'{axis}_{trig_id}']).to_numpy(dtype=float)
                    k = np.rint(shift / PIXEL_PITCH).astype(int)               # integer pixel shift per candidate
                    ks, inv = np.unique(k, return_inverse=True)
                    mode_k = ks[np.argmax(np.bincount(inv, weights=counts_w))]  # count-weighted mode
                    core = np.abs(k - mode_k) <= 1                               # mode +/- 1 pixel
                    # Weight share of the modal core: ~1 for a clean beam-spot
                    # peak, small when the mode is merely the crest of the flat
                    # combinatorial background (the estimator's low-purity
                    # failure mode, which returns ~0 = "no misalignment").
                    core_frac = min(core_frac, float(counts_w[core].sum() / counts_w.sum()))
                    center = float(np.average(shift[core], weights=counts_w[core]))
                    # New translation = existing translation minus the measured
                    # residual, so re-running on already-aligned data yields a
                    # residual of ~0 and leaves the translation unchanged.
                    combo_alignment[bid][axis] = round(float(existing.get(axis, 0.0)) - center, 3)
                combo_alignment[bid]['z'] = round(float(existing.get('z', 0.0)), 3)
                combo_core[bid] = core_frac

            alignment_results[combo_label] = combo_alignment

            for bid in sorted(set(combo_alignment) & set(alignment_source)):
                # Already aligned by an earlier combo, so this combo's candidates
                # were transformed with that translation and what was just
                # measured is a RESIDUAL: combo_alignment[bid] should reproduce
                # the applied value. A real disagreement means the two combos do
                # not see the same board position -- typically one of the two
                # modes latched onto combinatorial background instead of the beam
                # spot. Reported, not acted on: the first estimate stands.
                applied = full_config[args.runName][bid].get('transformation', {}).get('translation', {}) or {}
                residual = max(abs(float(combo_alignment[bid][a]) - float(applied.get(a, 0.0))) for a in ('x', 'y'))
                emit = logging.warning if residual > PIXEL_PITCH else logging.info
                emit(f"Combo ({combo_label}): board {bid} already aligned from ({alignment_source[bid]}) "
                     f"as {dict(applied)}; this combo re-measures {combo_alignment[bid]} "
                     f"(max residual {residual:.3f} mm). Not re-applied.")

            newly_aligned = {bid: t for bid, t in combo_alignment.items() if bid not in alignment_source}
            if newly_aligned:
                for bid, new_translation in newly_aligned.items():
                    full_config[args.runName][bid].setdefault('transformation', {})['translation'] = new_translation
                    alignment_source[bid] = combo_label
                run_config = full_config[args.runName]
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
                # Re-transform this combo's own candidates so its radius cut below
                # already uses the corrected geometry. Boards aligned by an earlier
                # combo are recomputed from row/col with the translation they
                # already had, so this is idempotent for them.
                apply_geometric_transformation_matrix(track_candidates, combo, run_config)

        spatial_mask = check_spatial_alignment(track_candidates, roles, args.max_diff_pixel)
        final_tracks = track_candidates[spatial_mask]

        # Remove duplicates if any remain based on pattern
        final_tracks = final_tracks.drop_duplicates(subset=group_cols)

        coord_cols = [c for c in final_tracks.columns if c.split('_')[0] in ['x', 'y', 'z']]
        final_tracks[coord_cols] = final_tracks[coord_cols].round(2)

        output_file = tracks_out_dir / f'tracks_{combo_label}.parquet'
        io_utils.write_parquet(final_tracks, output_file, index=False)
        logging.info(f"Combo ({combo_label}): {len(final_tracks)} tracks saved to {output_file}")
        n_written += 1

    if args.find_alignment and trig_id is not None:
        unaligned = [b for b in ids_to_process
                     if b != trig_id and b in boards_with_hits and b not in alignment_source]
        if unaligned:
            logging.warning(f"--find_alignment: board(s) {unaligned} never received a derived translation "
                            f"(no trigger-containing combo produced candidates for them). They kept their "
                            f"--config translation, so every combo containing them was cut on that geometry.")

    if alignment_results:
        # Own directory, separate from the tracks/cal_table output, since the
        # alignment yaml is a different kind of artifact (a diagnostic to
        # review/merge by hand, not pipeline input). It sits NEXT TO the
        # --track-label tree (<TRACK_LABEL parent>/alignment/) rather than in
        # the bare current working directory, so it lands with the run's other
        # outputs regardless of where the script is invoked from. For a plain
        # --track-label with no directory part this is still ./alignment/.
        alignment_dir = Path(args.track_label).parent / 'alignment'
        alignment_dir.mkdir(parents=True, exist_ok=True)
        alignment_file = alignment_dir / f'{Path(args.track_label).name}_alignment.yaml'
        align_block = {
            combo_label: {bid: {'transformation': {'translation': t}} for bid, t in combo_vals.items()}
            for combo_label, combo_vals in alignment_results.items()
        }
        # What the tracks in this run were ACTUALLY cut with, and which combo
        # supplied it. The per-combo blocks alone are ambiguous: a board that
        # appears in several of them is recorded pre-correction in the combo that
        # aligned it and as a residual re-measurement in the later ones, with
        # nothing in the numbers to say which. dict() copies the value so ruamel
        # emits it plainly instead of an alias to the run_config entry.
        if alignment_source:
            align_block['applied'] = {
                bid: {'from_combo': alignment_source[bid],
                      'transformation': {'translation': dict(
                          full_config[args.runName][bid]['transformation']['translation'])}}
                for bid in sorted(alignment_source)
            }
        with open(alignment_file, 'w') as f:
            yaml.dump({args.runName: align_block}, f)
        logging.info(f"Alignment offsets for {len(alignment_results)} combo(s) written to {alignment_file} "
                     f"(not saved back to {args.config} -- merge in manually if desired).")

    if n_written == 0:
        logging.warning("No track candidates found for any board combo.")
        sys.exit(0)

    logging.info(f"Done. {n_written}/{len(board_combos)} board combo(s) produced track files.")

if __name__ == "__main__":
    main()