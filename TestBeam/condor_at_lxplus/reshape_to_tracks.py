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

def load_config_and_roles(config_path: str, run_name: str) -> Tuple[Dict[int, str], Dict[str, str]]:
    """Loads YAML config and builds the board ID to Role mapping."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if run_name not in config:
        raise ValueError(f"Run config '{run_name}' not found in {config_path}")

    # Map ID -> Role
    id_role_map = {}
    for board_id, board_info in config[run_name].items():
        role = board_info.get('role')
        if role:
            id_role_map[board_id] = role

    # Nickname mapping
    nickname_dict = {'trig': '_t-', 'dut': '_d-', 'ref': '_r-', 'extra': '_e-'}

    return id_role_map, nickname_dict

def generate_track_filename(df: pd.DataFrame, id_map: Dict[int, str], nicknames: Dict[str, str]) -> str:
    """Generates a descriptive filename based on the track's coordinates."""
    # We look for level 'board' in columns
    board_ids = [
        b for b in df.columns.get_level_values('board').unique()
        if isinstance(b, int)
    ]

    filename_parts = ["track"]

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

    return "".join(filename_parts)

def process_and_save_track(
    track_key: Any,
    df_parts: List[pd.DataFrame],
    output_dir: Path,
    id_map: Dict[int, str],
    nicknames: Dict[str, str]
) -> str:
    """Worker function: Concatenates data parts for one track and saves to disk."""
    if not df_parts:
        return f"Skipped empty track list: {track_key}"

    try:
        full_df = pd.concat(df_parts, ignore_index=True)

        if full_df.empty:
            return f"Skipped empty dataframe: {track_key}"

        out_name = generate_track_filename(full_df, id_map, nicknames)
        save_path = output_dir / f"{out_name}.pickle"
        full_df.to_pickle(save_path)

        return f"Saved: {save_path.name}"

    except Exception as e:
        return f"Error saving track {track_key}: {e}"

def determine_file_batches(files: List[Path]) -> List[List[Path]]:
    """
    Splits files into batches based on the logic:
    - If < size_threshold files: 1 batch.
    - If >= size_threshold files: Split into 2-5 groups.
      Choose min groups such that batch size < 100 (if possible).
    """
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
    parser.add_argument('--file_pattern', default='*.pickle', help="Glob pattern for input files")
    parser.add_argument('--debug', action='store_true', help='Run sequentially for debugging')

    args = parser.parse_args()

    # 1. Setup paths and config
    try:
        id_role_map, nickname_dict = load_config_and_roles(args.config, args.runName)
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
        # If multiple groups: tracks_group1, tracks_group2...
        # If single group: tracks
        if is_multi_group:
            subdir_name = f'tracks_group{batch_idx + 1}'
        else:
            subdir_name = 'tracks'

        current_out_dir = base_out_path / subdir_name
        current_out_dir.mkdir(parents=True, exist_ok=True)

        print(f'\n====== Processing Group {batch_idx + 1}/{len(batches)} ({len(batch_files)} files) ======')
        print(f"Saving to: {current_out_dir}")

        track_data = defaultdict(list)

        if args.debug:
            batch_files = batch_files[:5]

        # A. Read Files into Memory
        for f in tqdm(batch_files, desc=f"Reading Group {batch_idx + 1}"):
            try:
                data_dict = pd.read_pickle(f)
                for key, df in data_dict.items():
                    if not df.empty:
                        track_data[key].append(df)
            except Exception as e:
                print(f"Warning: Failed to read {f.name}: {e}")

        total_tracks = len(track_data)
        print(f"Group {batch_idx + 1}: Found {total_tracks} unique tracks.")

        # B. Save Tracks
        print(f'Saving tracks for Group {batch_idx + 1}...')

        if args.debug:
            print("(Debug Mode: Processing sequentially)")
            for key, parts in track_data.items():
                res = process_and_save_track(key, parts, current_out_dir, id_role_map, nickname_dict)
                print(res)
                if "Saved" in res: break
        else:
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        process_and_save_track,
                        key, parts, current_out_dir, id_role_map, nickname_dict
                    )
                    for key, parts in track_data.items()
                ]

                for future in tqdm(as_completed(futures), total=total_tracks, desc="Saving"):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Worker Error: {e}")

        # C. Memory Cleanup
        print(f"Cleaning up memory for Group {batch_idx + 1}...")
        track_data.clear()
        del track_data
        gc.collect()

    print(f"\nDone. All groups processed.")

if __name__ == "__main__":
    main()
