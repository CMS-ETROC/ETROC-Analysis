import argparse, sys, yaml, shutil, gc, getpass

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from natsort import natsorted
from tqdm import tqdm

# ------------------------------------
def packer_worker(p_idx, b_idx, file_list, out_tmp_dir):
    try:
        dataset = ds.dataset(file_list, format="parquet")
        table = dataset.to_table()
        output_file = out_tmp_dir / f"p{p_idx}_b{b_idx}.parquet"
        pq.write_table(table, output_file, compression='lz4')

        return f"Done: P{p_idx+1} B{b_idx}"

    except Exception as e:
        return f"Error P{p_idx+1} B{b_idx}: {e}"

# ------------------------------------
def run_parallel_packing(flat_tasks):
    print(f"--- Step 1: Launching {len(flat_tasks)} Parallel Packing Tasks ---")

    # We use max_workers=4 as a safe default for Lxplus interactive nodes
    with ProcessPoolExecutor(max_workers=4) as executor:
        # We use submit() here because it can handle the unpacked arguments
        # without needing a non-picklable lambda function.
        futures = [executor.submit(packer_worker, *task) for task in flat_tasks]

        results = []
        # Use as_completed to update the progress bar as jobs finish
        for f in tqdm(as_completed(futures), total=len(flat_tasks), desc="Packing"):
            try:
                results.append(f.result())
            except Exception as e:
                results.append(f"Worker crashed: {e}")

    # Print any errors that occurred during the run
    for r in results:
        if "Error" in r or "crashed" in r:
            print(r)

# ------------------------------------
def run_safe_sequential_splitting(out_dir, out_tmp_dir, nicknames, num_partitions):
    print(f"--- Step 2: Safe Sequential Splitting ---")

    # 1. Create a SECOND local temp directory for the final tracks
    local_final_base = out_tmp_dir.parent / "final_tracks_local"
    if local_final_base.exists():
        shutil.rmtree(local_final_base)
    local_final_base.mkdir(parents=True, exist_ok=True)

    for p_idx in range(num_partitions):
        subdir_name = "tracks" if num_partitions == 1 else f"tracks_group{p_idx + 1}"
        local_track_dir = local_final_base / subdir_name
        local_track_dir.mkdir(parents=True, exist_ok=True)

        partition_masters = natsorted(out_tmp_dir.glob(f"p{p_idx}_b*.parquet"))
        if not partition_masters: continue

        for m_file in tqdm(partition_masters, desc=f"Merging {subdir_name} locally"):
            batch_df = pd.read_parquet(m_file)

            for _, track_df in batch_df.groupby('track_id'):
                fname = generate_track_filename(track_df, nicknames) + ".parquet"
                save_path = local_track_dir / fname # Writing to LOCAL SSD

                if save_path.exists():
                    # This Read-Append-Write is now 100x faster because it's on SSD
                    existing_df = pd.read_parquet(save_path)
                    combined_df = pd.concat([existing_df, track_df], ignore_index=True)
                    combined_df.to_parquet(save_path, compression='lz4')
                    del existing_df, combined_df
                else:
                    track_df.to_parquet(save_path, compression='lz4')

            del batch_df
            gc.collect()

    # 2. THE BULK MOVE: Copy from /tmp to EOS in one go
    print(f"--- Step 3: Moving final files to EOS: {out_dir} ---")
    final_eos_path = Path(out_dir)
    # shutil.copytree is usually more robust for EOS than shutil.move
    shutil.copytree(local_final_base, final_eos_path, dirs_exist_ok=True)

    # 3. Final cleanup of all local temp areas
    shutil.rmtree(local_final_base)


# ------------------------------------
def run_hybrid_fast_splitting(out_dir, out_tmp_dir, nicknames, num_partitions):
    print(f"--- Step 2: Native Scatter ---")

    for p_idx in range(num_partitions):

        partition_masters = natsorted(out_tmp_dir.glob(f"p{p_idx}_b*.parquet"))
        if not partition_masters: continue

        scatter_dir = out_tmp_dir / f"p{p_idx}"
        scatter_dir.mkdir(parents=True, exist_ok=True)

        for m_file in tqdm(partition_masters, desc=f"Scattering Partition {p_idx+1}"):
            table = pq.read_table(m_file)
            ds.write_dataset(
                table,
                base_dir=scatter_dir,
                format="parquet",
                partitioning=["track_id"],
                existing_data_behavior="overwrite_or_ignore",
                basename_template=f"{m_file.stem}_{{i}}.parquet",
                use_threads=False,
            )

    # 2. THE GATHER: Unifying the fragments into the final EOS directory
    print(f"\n--- Step 3: Final Unification ---")

    # Identify all numeric track directories (e.g., 0, 1, 2...)
    # In your output, PyArrow created directories directly by ID value
    subdir = "tracks" if num_partitions == 1 else f"tracks_group{p_idx + 1}"
    track_dirs = natsorted([d for d in scatter_dir.iterdir() if d.is_dir()])

    for track_folder in tqdm(track_dirs, desc=f"Unifying {subdir}"):
        track_fragments = ds.dataset(track_folder, format="parquet")
        combined_table = track_fragments.to_table()

        # Efficient coordinate lookup
        sample_df = combined_table.slice(0, 1).to_pandas()
        fname = generate_track_filename(sample_df, nicknames) + ".parquet"

        save_path = Path(out_dir) / subdir / fname
        save_path.parent.mkdir(parents=True, exist_ok=True)

        pq.write_table(combined_table, save_path, compression='lz4')

    # Clean up this partition's fragments before moving to the next one to save /tmp space
    shutil.rmtree(scatter_dir)

    print(f"Successfully unified {len(track_dirs)} tracks.")

# ------------------------------------
def load_config_and_roles(config_path: str, run_name: str) -> tuple[dict[int, str], dict[str, str]]:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if run_name not in config:
        sys.exit(f"Error: Run config '{run_name}' not found in {config_path}")
    id_role_map = {}

    for board_id, board_info in config[run_name].items():
        role = board_info.get('role', 'unknown')
        try:
            board_id_int = int(board_id)
            id_role_map[board_id_int] = role
        except ValueError:
            id_role_map[str(board_id)] = role

    nickname_dict = {'trig': '_t-', 'dut': '_d-', 'ref': '_r-', 'extra': '_e-'}

    for role in id_role_map.values():
        if role not in nickname_dict:
            nickname_dict[role] = f'_{role[:3]}-'

    return id_role_map, nickname_dict

# ------------------------------------
def generate_track_filename(df: pd.DataFrame, nicknames: dict[str, str]) -> str:
    filename_parts = ["track"]
    for role in nicknames.keys():
        row_col = f'row_{role}'
        col_col = f'col_{role}'
        prefix = nicknames.get(role, f'_{role}-')
        if row_col in df.columns and col_col in df.columns:
            try:
                r_val = df[row_col].unique()[0]
                c_val = df[col_col].unique()[0]
                filename_parts.append(f"{prefix}R{r_val}C{c_val}")
            except (KeyError, IndexError):
                continue
    return "".join(filename_parts)

# ------------------------------------
def determine_size(files, num_groups):
    """
    Equally spaces files into a fixed number of groups.
    """
    n_files = len(files)

    # Ensure we don't try to make more groups than there are files
    num_groups = min(num_groups, n_files)

    # Calculate base size (k) and the remainder (m)
    k, m = divmod(n_files, num_groups)

    batches = []
    start_idx = 0
    for i in range(num_groups):
        # Add 1 extra file to the first 'm' groups to handle the remainder
        batch_size = k + 1 if i < m else k
        end_idx = start_idx + batch_size
        batches.append(files[start_idx:end_idx])
        start_idx = end_idx

    return batches

# ------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Submit jobs to reshape event-based to track-based')
    parser.add_argument('-d', '--inputdir', required=True, dest='dirname', help='Input directory')
    parser.add_argument('-o', '--outdir', required=True, dest='outdir', help='Output base directory')
    parser.add_argument('-r', '--runName', required=True, dest='runName', help='Run name in config')
    parser.add_argument('-c', '--config', required=True, dest='config', help='YAML config file')
    parser.add_argument('-b', '--batches', type=int, default=30, dest='batches', help='Total batches to split input files into for safety')
    parser.add_argument('-p', '--partitions', type=int, default=1, dest='partitions', help='Number of output datasets (partitions)')
    parser.add_argument('--file_pattern', default='*.parquet', help="Glob pattern for inputs")

    args = parser.parse_args()

    # --- Setup Environments ---
    username = getpass.getuser()
    eos_base_dir = f'/eos/user/{username[0]}/{username}'

    all_files = natsorted(Path(f'{eos_base_dir}/{args.dirname}').glob(args.file_pattern))
    final_output_dir = f'{eos_base_dir}/{args.outdir}'

    if not all_files:
        print('No input files found')
        sys.exit(1)

    out_tmp_dir = Path('/tmp/reshape_events_to_tracks')
    if out_tmp_dir.exists():
        shutil.rmtree(out_tmp_dir)
    out_tmp_dir.mkdir(parents=True, exist_ok=True)

    _, nickname_dict = load_config_and_roles(args.config, args.runName)

    partitions = determine_size(all_files, args.partitions)
    flat_tasks = []
    for p_idx, partition_files in enumerate(partitions):
        batches = determine_size(partition_files, args.batches)
        for b_idx, batch_files in enumerate(batches):
            file_list_str = [str(f) for f in batch_files]
            flat_tasks.append((p_idx, b_idx, file_list_str, out_tmp_dir))

    run_parallel_packing(flat_tasks)

    run_hybrid_fast_splitting(final_output_dir, out_tmp_dir, nickname_dict, args.partitions)

    print(f"--- Final Step: Cleaning up temporary files in {out_tmp_dir} ---")
    try:
        shutil.rmtree(out_tmp_dir)
        # Also remove the 'tmp' parent folder if it's now empty
        if out_tmp_dir.parent.exists() and not any(out_tmp_dir.parent.iterdir()):
            out_tmp_dir.parent.rmdir()
    except Exception as e:
        print(f"Warning: Cleanup failed: {e}")

    print("\nReshaping Complete!")

if __name__ == "__main__":
    main()