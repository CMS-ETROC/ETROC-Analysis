import argparse
import sys
import yaml
import warnings
import pandas as pd
import shutil
import pyarrow.dataset as ds
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from natsort import natsorted
from typing import List, Dict, Tuple, Any

# --- Configuration ---
warnings.filterwarnings("ignore")

# --- Helper Functions ---
def load_config_and_roles(config_path: str, run_name: str) -> Tuple[Dict[int, str], Dict[str, str]]:
    """
    Loads YAML config and builds the board ID to Role mapping.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if run_name not in config:
        raise ValueError(f"Run config '{run_name}' not found in {config_path}")

    id_role_map = {}

    for board_id, board_info in config[run_name].items():
        role = board_info.get('role', 'unknown')
        try:
            board_id_int = int(board_id)
            id_role_map[board_id_int] = role
        except ValueError:
            id_role_map[str(board_id)] = role

    # Nickname mapping (used for filename prefixes)
    nickname_dict = {'trig': '_t-', 'dut': '_d-', 'ref': '_r-', 'extra': '_e-'}

    # Add the roles found in the config to nickname_dict if they aren't standard
    for role in id_role_map.values():
        if role not in nickname_dict:
            nickname_dict[role] = f'_{role[:3]}-'

    return id_role_map, nickname_dict

def generate_track_filename(df: pd.DataFrame, nicknames: Dict[str, str]) -> str:
    """Generates a descriptive filename based on the track's coordinates."""
    filename_parts = ["track"]

    # Flat Index/Role-based Columns Format
    known_roles = nicknames.keys()
    for role in known_roles:
        row_col = f'row_{role}'
        col_col = f'col_{role}'
        prefix = nicknames.get(role, f'_{role}-')

        if row_col in df.columns and col_col in df.columns:
            try:
                r_val = df[row_col].unique()[0]
                c_val = df[col_col].unique()[0]
                filename_parts.append(f"{prefix}R{r_val}C{c_val}")
            except KeyError:
                continue

    return "".join(filename_parts)

def determine_file_batches(files: List[Path], num_groups: int = 1) -> List[List[Path]]:
    """
    Equally spaces files into a fixed number of groups.
    """
    n_files = len(files)
    num_groups = min(num_groups, n_files)

    if num_groups <= 0:
        return [files]

    k, m = divmod(n_files, num_groups)
    batches = []
    start_idx = 0

    for i in range(num_groups):
        batch_size = k + 1 if i < m else k
        end_idx = start_idx + batch_size
        batches.append(files[start_idx:end_idx])
        start_idx = end_idx

    print(f"\nSplitting {n_files} files into {num_groups} equal groups.")
    return batches

def process_partitioned_track(
    track_dir: Path,
    output_dir: Path,
    nicknames: Dict[str, str]
) -> str:
    try:
        # 1. Safely read the actual parquet files INSIDE the partition folder
        pq_files = list(track_dir.glob("*.parquet"))
        if not pq_files:
            shutil.rmtree(track_dir, ignore_errors=True)
            return f"Skipped: No parquet files found in {track_dir.name}"

        # Read the file(s) and force the pyarrow engine
        df = pd.concat([pd.read_parquet(f, engine='pyarrow') for f in pq_files], ignore_index=True)

        if df.empty:
            shutil.rmtree(track_dir, ignore_errors=True)
            return f"Skipped empty track: {track_dir.name}"

        # 2. Generate your custom filename
        out_name = generate_track_filename(df, nicknames)

        # --- Safety Check: Prevent Overwriting ---
        # If generate_track_filename fails to find the right columns, it returns just "track".
        # We append the track_id so 695 files don't overwrite each other as "track.parquet"!
        if out_name == "track":
            out_name = f"track_{track_dir.name}"

        save_path = output_dir / f"{out_name}.parquet"

        # 3. Save the final file
        df.to_parquet(save_path, compression='lz4', engine='pyarrow')

        # 4. Clean up the temp directory
        shutil.rmtree(track_dir, ignore_errors=True)

        return f"Saved: {save_path.name}"

    except Exception as e:
        # Return the EXACT error so we can see what's going wrong
        return f"Error processing {track_dir.name}: {repr(e)}"

# --- Main Logic ---
def main():
    parser = argparse.ArgumentParser(
        description='Reads Parquet event data, reshapes it, and saves as track-based files out-of-core.'
    )
    parser.add_argument('-d', '--inputdir', required=True, dest='dirname', help='Input directory')
    parser.add_argument('-o', '--outdir', required=True, dest='outdir', help='Output directory')
    parser.add_argument('-r', '--runName', required=True, dest='runName', help='Run name')
    parser.add_argument('-c', '--config', required=True, dest='config', help='YAML config file')
    parser.add_argument('--groups', type=int, default=1, help='Number of processing groups to split files into')
    parser.add_argument('--file_pattern', default='*.parquet', help="Glob pattern for input files (e.g. '*.parquet')")
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
    batches = determine_file_batches(files, num_groups=args.groups)
    is_multi_group = len(batches) > 1

    # 4. Process Batches
    for batch_idx, batch_files in enumerate(batches):

        # --- Update Output Directory ---
        if is_multi_group:
            subdir_name = f'tracks_group{batch_idx + 1}'
        else:
            subdir_name = 'tracks'

        current_out_dir = base_out_path / subdir_name
        temp_dir = base_out_path / f'temp_partitions_group{batch_idx + 1}'

        current_out_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)

        print(f'\n====== Processing Group {batch_idx + 1}/{len(batches)} ({len(batch_files)} files) ======')
        print(f"Saving to: {current_out_dir}")

        # ==========================================
        # PHASE 1: Out-of-Core Streaming & Grouping
        # ==========================================
        print(f"Phase 1: Streaming and partitioning Group {batch_idx + 1} to disk (zero memory spikes)...")
        try:
            # Convert Path objects to strings for pyarrow
            file_paths_str = [str(f) for f in batch_files]
            dataset = ds.dataset(file_paths_str, format="parquet")

            ds.write_dataset(
                data=dataset,
                base_dir=temp_dir,
                format="parquet",
                partitioning=["track_id"],
                existing_data_behavior="overwrite_or_ignore"
            )
        except Exception as e:
            print(f"Failed during out-of-core partitioning for Group {batch_idx + 1}: {e}")
            continue

        # ==========================================
        # PHASE 2: Parallel Formatting & Cleanup
        # ==========================================
        track_dirs = [d for d in temp_dir.iterdir() if d.is_dir()]
        total_tracks = len(track_dirs)

        print(f"\nPhase 2: Found {total_tracks} unique tracks. Formatting and saving...")

        all_futures = []

        if args.debug:
            print("(Debug Mode: Processing sequentially)")
            for track_dir in track_dirs:
                res = process_partitioned_track(track_dir, current_out_dir, nickname_dict)
                print(res)
                if "Saved" in res: break
        else:
            with ProcessPoolExecutor(max_workers=6) as executor:
                for track_dir in track_dirs:
                    future = executor.submit(
                        process_partitioned_track,
                        track_dir, current_out_dir, nickname_dict
                    )
                    all_futures.append(future)

                for future in tqdm(as_completed(all_futures), total=total_tracks, desc=f"Saving Group {batch_idx + 1}"):
                    try:
                        res = future.result()
                        # CRITICAL: Print the error if the worker failed!
                        if "Error" in res:
                            tqdm.write(res) # tqdm.write prints without breaking the progress bar
                    except Exception as e:
                        tqdm.write(f"Worker Exception: {repr(e)}")

        # Clean up the temp directory for this specific group
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except OSError:
                pass

    print(f"\nDone. All groups processed successfully.")

if __name__ == "__main__":
    main()