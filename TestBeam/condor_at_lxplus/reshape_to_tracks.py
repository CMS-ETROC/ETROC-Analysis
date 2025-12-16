import argparse
import sys
import yaml
import warnings
import pandas as pd
import gc
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from natsort import natsorted
from typing import List, Dict, Any, Tuple

# --- Configuration ---
warnings.filterwarnings("ignore")

# --- Helper Functions ---

def rename_parquet_columns(df: pd.DataFrame, rename_map: Dict[str, str]) -> pd.DataFrame:
    """
    Renames columns in a flat Parquet DataFrame from index suffix (e.g., 'row_0')
    to role suffix (e.g., 'row_trig') using the provided dynamic rename_map.
    """
    rename_dict = {}

    # Iterate through the provided map (e.g., '_0' : '_trig')
    for old_suffix, new_suffix in rename_map.items():
        # Find columns ending with the old suffix
        for col in df.columns:
            # We check if the column name ends with the old suffix
            if col.endswith(old_suffix):
                # Ensure the column isn't one of the track-level metadata columns (e.g., 'track_id')
                if not col.startswith('track_') and not col.startswith('file'):
                    new_col = col.replace(old_suffix, new_suffix)
                    rename_dict[col] = new_col

    if rename_dict:
        # Perform the renaming and return the modified DataFrame
        return df.rename(columns=rename_dict)

    return df

def load_config_and_roles(config_path: str, run_name: str) -> Tuple[Dict[int, str], Dict[str, str], Dict[str, str]]:
    """
    Loads YAML config and builds the board ID to Role mapping and the dynamic
    positional index -> role mapping for Parquet file renaming.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if run_name not in config:
        raise ValueError(f"Run config '{run_name}' not found in {config_path}")

    # Map ID -> Role (for PKL/MultiIndex filename generation)
    id_role_map = {}

    # Map Positional Index -> Role (for Parquet renaming)
    # The order of iteration over config[run_name] defines the positional index (0, 1, 2, 3...)
    parquet_positional_role_map = {}
    index = 0

    for board_id, board_info in config[run_name].items():
        role = board_info.get('role', 'unknown')

        # 1. Build the ID -> Role map (used for PKL/MultiIndex naming)
        try:
            board_id_int = int(board_id)
            id_role_map[board_id_int] = role
        except ValueError:
            id_role_map[str(board_id)] = role

        # 2. Build the Positional Index -> Role map (used for Parquet renaming)
        # We only need the first 4 (0-3) since the flat Parquet files currently only go up to _3.
        if index < 4 and role != 'unknown':
            # The rename map needs to map the old suffix (e.g., _0) to the new suffix (e.g., _trig)
            parquet_positional_role_map[f'_{index}'] = f'_{role}'
            index += 1

    # Nickname mapping (used for filename prefixes)
    nickname_dict = {'trig': '_t-', 'dut': '_d-', 'ref': '_r-', 'extra': '_e-'}

    # Add the roles found in the config to nickname_dict if they aren't standard
    for role in id_role_map.values():
        if role not in nickname_dict:
            # Use first 3 letters as prefix if not standard
            nickname_dict[role] = f'_{role[:3]}-'

    return id_role_map, nickname_dict, parquet_positional_role_map

def generate_track_filename(df: pd.DataFrame, id_map: Dict[int, str], nicknames: Dict[str, str]) -> str:
    """Generates a descriptive filename based on the track's coordinates."""

    filename_parts = ["track"]

    # --- Logic for PKL (MultiIndex) Format ---
    if isinstance(df.columns, pd.MultiIndex):

        # We look for level 'board' in columns
        board_ids = [
            b for b in df.columns.get_level_values('board').unique()
            if isinstance(b, int)
        ]

        for bid in board_ids:
            role = id_map.get(bid, 'unknown')
            prefix = nicknames.get(role, f'_{bid}-')

            try:
                # unique()[0] assumes the track is static across the event
                r_val = df['row'][bid].unique()[0]
                c_val = df['col'][bid].unique()[0]
                filename_parts.append(f"{prefix}R{r_val}C{c_val}")
            except KeyError:
                continue

    # --- Logic for Parquet (Flat Index/Renamed) Format ---
    else:
        # CORRECTED: Iterate over the keys (roles) of the nickname dictionary for simplicity
        known_roles = nicknames.keys()

        for role in known_roles:
            # Construct the expected column names based on the role (e.g., 'row_trig')
            row_col = f'row_{role}'
            col_col = f'col_{role}'

            # Determine the prefix using the role
            prefix = nicknames.get(role, f'_{role}-')

            if row_col in df.columns and col_col in df.columns:

                try:
                    # unique()[0] assumes the track is static across the event
                    r_val = df[row_col].unique()[0]
                    c_val = df[col_col].unique()[0]
                    filename_parts.append(f"{prefix}R{r_val}C{c_val}")
                except KeyError:
                    # Should not happen if columns exist, but kept for robustness
                    continue

    return "".join(filename_parts)

def process_and_save_track(
    track_key: Any,
    df_parts: List[pd.DataFrame],
    output_dir: Path,
    id_map: Dict[int, str],
    nicknames: Dict[str, str],
    is_parquet_format: bool = False
) -> str:
    """Worker function: Concatenates data parts for one track and saves to disk."""
    if not df_parts:
        return f"Skipped empty track list: {track_key}"

    try:
        full_df = pd.concat(df_parts, ignore_index=True)

        if full_df.empty:
            return f"Skipped empty dataframe: {track_key}"

        out_name = generate_track_filename(full_df, id_map, nicknames)

        # Save as Parquet if it originated from a Parquet file, otherwise save as Pickle (PKL)
        ext = ".parquet" if is_parquet_format else ".pkl"
        save_path = output_dir / f"{out_name}{ext}"

        if is_parquet_format:
            # Using to_parquet for the Parquet case
            full_df.to_parquet(save_path)
        else:
            # Using to_pickle for the PKL case
            full_df.to_pickle(save_path)

        return f"Saved: {save_path.name}"

    except Exception as e:
        return f"Error saving track {track_key}: {e}"

def determine_file_batches(files: List[Path]) -> List[List[Path]]:
    # ... (function body remains the same)
    n_files = len(files)

    size_threshold = 120
    if n_files < size_threshold:
        return [files]

    # Logic: Try groups 2, 3, 4, 5.
    # Pick the first one (minimum) that results in a chunk size < 100.
    # If none do (e.g. 1000 files), cap at 5 groups.
    num_groups = 5 # Default max
    for g in range(2, 6):
        # Ceiling division equivalent for roughly equal chunks
        chunk_size = (n_files + g - 1) // g
        if chunk_size < size_threshold:
            num_groups = g
            break

    # Calculate chunk size for the chosen number of groups
    k, m = divmod(n_files, num_groups)
    batches = []
    start_idx = 0
    for i in range(num_groups):
        # Distribute remainder to first few groups
        batch_size = k + 1 if i < m else k
        end_idx = start_idx + batch_size
        batches.append(files[start_idx:end_idx])
        start_idx = end_idx

    print(f"\nLarge dataset detected ({n_files} files). Splitting into {num_groups} processing groups.")
    for i, b in enumerate(batches):
        print(f"  Group {i+1}: {len(b)} files")

    return batches

# --- Main Logic ---

def main():
    parser = argparse.ArgumentParser(
        description='Reads file-based data, reshapes it, and saves as track-based files.'
    )
    parser.add_argument('-d', '--inputdir', required=True, dest='dirname', help='Input directory')
    parser.add_argument('-o', '--outdir', required=True, dest='outdir', help='Output directory')
    parser.add_argument('-r', '--runName', required=True, dest='runName', help='Run name')
    parser.add_argument('-c', '--config', required=True, dest='config', help='YAML config file')
    parser.add_argument('--file_pattern', default='*.pickle', help="Glob pattern for input files (e.g. '*.pkl *.parquet')")
    parser.add_argument('--debug', action='store_true', help='Run sequentially for debugging')

    args = parser.parse_args()

    # 1. Setup paths and config
    try:
        # CORRECT: Unpack three values, including the new parquet_rename_map
        id_role_map, nickname_dict, parquet_rename_map = load_config_and_roles(args.config, args.runName)
    except Exception as e:
        sys.exit(f"Config Error: {e}")
    base_out_path = Path(args.outdir)

    print(f'\nInput:  {args.dirname}')
    print(f'Output Base: {base_out_path}\n')

    # 2. Find Files
    files = []
    for pattern in args.file_pattern.split():
        files.extend(natsorted(Path(args.dirname).glob(pattern)))

    if not files:
        sys.exit(f"No files found matching '{args.file_pattern}' in {args.dirname}")

    # 3. Determine Batches
    batches = determine_file_batches(files)

    # Check if we have multiple batches to decide naming convention
    is_multi_group = len(batches) > 1

    # 4. Process Batches
    for batch_idx, batch_files in enumerate(batches):

        # --- Update Output Directory ---
        if is_multi_group:
            subdir_name = f'tracks_group{batch_idx + 1}'
        else:
            subdir_name = 'tracks'

        current_out_dir = base_out_path / subdir_name
        current_out_dir.mkdir(parents=True, exist_ok=True)

        print(f'\n====== Processing Group {batch_idx + 1}/{len(batches)} ({len(batch_files)} files) ======')
        print(f"Saving to: {current_out_dir}")

        track_data_pkl = defaultdict(list)
        track_data_pqt = [] # List to hold all DataFrames from Parquet files

        pkl_flag = False
        pqt_flag = False

        # A. Read Files into Memory
        for f in tqdm(batch_files, desc=f"Reading Group {batch_idx + 1}"):
            if '.pkl' in f.name:
                data_dict = pd.read_pickle(f)
                for key, df in data_dict.items():
                    if not df.empty:
                        track_data_pkl[key].append(df)
                pkl_flag = True
            elif '.parquet' in f.name:
                initial_df = pd.read_parquet(f)
                if not initial_df.empty:
                    # FIX: CALL RENAME FUNCTION HERE
                    renamed_df = rename_parquet_columns(initial_df, parquet_rename_map)
                    track_data_pqt.append(renamed_df)
                pqt_flag = True
            else:
                print(f"Warning: Failed to read {f.name}")

        # --- Handle input file type conflict ---
        if pkl_flag and pqt_flag:
            # We allow mixing input types but process them separately, so we just warn
            print('Warning: Input directory contains both .pkl and .parquet files. Both formats will be processed.')

        # B. Save Tracks
        print(f'Saving tracks for Group {batch_idx + 1}...')

        all_futures = []
        tracks_to_process = 0

        # 1. Process PKL files (Old Logic)
        if pkl_flag:
            total_tracks = len(track_data_pkl)
            print(f"  PKL: Found {total_tracks} unique tracks to process.")
            tracks_to_process += total_tracks

            if args.debug:
                print("(Debug Mode: PKL processing sequentially)")
                for key, parts in track_data_pkl.items():
                    res = process_and_save_track(key, parts, current_out_dir, id_role_map, nickname_dict, is_parquet_format=False)
                    print(res)
                    if "Saved" in res: break
            else:
                executor_pkl = ProcessPoolExecutor()
                pkl_futures = [
                    executor_pkl.submit(
                        process_and_save_track,
                        key, parts, current_out_dir, id_role_map, nickname_dict, is_parquet_format=False
                    )
                    for key, parts in track_data_pkl.items()
                ]
                all_futures.extend(pkl_futures)

        # 2. Process PARQUET files (New Logic)
        if pqt_flag:
            if track_data_pqt:
                # i. Concatenate all parquet DataFrames from the batch
                print("  PARQUET: Concatenating all DataFrames in the batch...")
                full_pqt_df = pd.concat(track_data_pqt, ignore_index=True)

                # ii. Group by track_id
                grouped_pqt_df = full_pqt_df.groupby('track_id')
                total_tracks = len(grouped_pqt_df)
                print(f"  PARQUET: Found {total_tracks} unique tracks to process.")
                tracks_to_process += total_tracks

                # iii. Create track parts and submit
                if args.debug:
                    print("(Debug Mode: PARQUET processing sequentially)")
                    for track_id, track_df in grouped_pqt_df:
                        res = process_and_save_track(track_id, [track_df], current_out_dir, id_role_map, nickname_dict, is_parquet_format=True)
                        print(res)
                        if "Saved" in res: break
                else:
                    executor_pqt = ProcessPoolExecutor()
                    pqt_futures = [
                        executor_pqt.submit(
                            process_and_save_track,
                            track_id, [track_df], current_out_dir, id_role_map, nickname_dict, is_parquet_format=True
                        )
                        for track_id, track_df in grouped_pqt_df
                    ]
                    all_futures.extend(pqt_futures)

                # Cleanup Parquet DataFrames in memory
                del full_pqt_df
                del grouped_pqt_df

        # 3. Wait for all futures to complete (if not in debug mode)
        if not args.debug and all_futures:
            print(f"  Starting parallel saving for a total of {tracks_to_process} tracks...")
            for future in tqdm(as_completed(all_futures), total=tracks_to_process, desc="Saving All Tracks"):
                try:
                    future.result()
                except Exception as e:
                    print(f"Worker Error: {e}")

            # Close the executor for this batch
            if 'executor_pkl' in locals():
                executor_pkl.shutdown(wait=True)
            if 'executor_pqt' in locals():
                executor_pqt.shutdown(wait=True)

        # C. Memory Cleanup
        print(f"Cleaning up memory for Group {batch_idx + 1}...")
        track_data_pkl.clear()
        del track_data_pkl
        track_data_pqt.clear()
        del track_data_pqt
        gc.collect()

    print(f"\nDone. All groups processed.")

if __name__ == "__main__":
    main()