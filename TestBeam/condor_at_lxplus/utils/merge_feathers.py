import argparse
import pandas as pd
from pathlib import Path
from natsort import natsorted
from tqdm import tqdm
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_balanced_sequential_groups(files, target_size):
    """
    Distributes leftovers across the first few groups to maintain
    uniformity and strict sequential order.
    """
    total_files = len(files)
    if total_files == 0:
        return []
    if total_files <= target_size:
        return [files]

    # Calculate how many groups to create based on the target size
    num_groups = total_files // target_size

    # Use divmod to get the base size and how many groups need +1 file
    base_size, remainder = divmod(total_files, num_groups)

    groups = []
    start = 0
    for i in range(num_groups):
        # The first 'remainder' groups get one extra file
        current_size = base_size + (1 if i < remainder else 0)
        groups.append(files[start : start + current_size])
        start += current_size

    return groups

def process_single_group(idx, igroup, tmp_outdir):
    """
    Worker function to process a single group of files.
    """
    try:
        nevt_adder = 0
        dfs = []

        for ifile in igroup:
            df = pd.read_feather(ifile)

            # Offset 'evt' column to maintain continuity within the new merged file
            df['evt'] += nevt_adder
            nevt_adder += df['evt'].nunique()

            dfs.append(df)

        # Concatenate and save to temporary location
        final_df = pd.concat(dfs, ignore_index=True)
        output_file = tmp_outdir / f'loop_{idx}.feather'
        final_df.to_feather(output_file)

        return True, idx
    except Exception as e:
        return False, f"Group {idx} failed: {e}"

def main():
    parser = argparse.ArgumentParser(
        prog='MergeFeathers',
        description='Merge feather files into balanced, sequential chunks using ProcessPoolExecutor.',
    )

    parser.add_argument(
        '-d', '--input_dir',
        type=str,
        required=True,
        help='Input directory containing .feather files',
    )

    parser.add_argument(
        '-n', '--number_of_merge',
        type=int,
        default=10,
        help='Target number of files per merge',
    )

    parser.add_argument(
        '--dryrun',
        action = 'store_true',
        help = 'If set, No merge happens',
        dest = 'dryrun',
    )

    args = parser.parse_args()
    input_path = Path(args.input_dir).resolve()

    # 1. Collect and Sort files
    files = natsorted([
        f for f in input_path.glob('loop_*.feather')
        if not (f.name.startswith('merged_') or f.name.startswith('new_'))
    ])

    if not files:
        print(f"No files found in {input_path} matching 'loop_*.feather'")
        return

    # 2. Generate Balanced Groups
    groups = get_balanced_sequential_groups(files, args.number_of_merge)

    # 3. Improved Group Printing
    counts = Counter(len(g) for g in groups)
    dist_str = ", ".join([f"{count} groups of {size}" for size, count in sorted(counts.items(), reverse=True)])

    # 4. Create a temporary output directory
    tmp_outdir = input_path.parents[0] / "merged_hits"
    tmp_outdir.mkdir(parents=True, exist_ok=True)

    print(f'Total input files: {len(files)}')
    print(f'Total output files: {len(groups)}')
    print(f'Group distribution: {dist_str}')
    print(f'Output directory: {tmp_outdir}')

    if args.dryrun:
        return

    # 5. Processing with ProcessPoolExecutor
    try:
        results_count = 0
        with ProcessPoolExecutor(max_workers=3) as executor:
            # Map each group to a future
            future_to_group = {
                executor.submit(process_single_group, idx, group, tmp_outdir): idx
                for idx, group in enumerate(groups)
            }

            # Wrap in tqdm to track progress as futures complete
            for future in tqdm(as_completed(future_to_group), total=len(groups), desc="Merging Groups"):
                success, info = future.result()
                if success:
                    results_count += 1
                else:
                    print(f"\n[ERROR] {info}")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        print("Original files preserved. Check temporary directory for partial results.")

if __name__ == "__main__":
    main()