import argparse
import pandas as pd
from pathlib import Path
from natsort import natsorted
from tqdm import tqdm
from collections import Counter

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

def main():
    parser = argparse.ArgumentParser(
        prog='MergeFeathers',
        description='Merge feather files into balanced, sequential chunks.',
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
    input_path = Path(args.input_dir)

    # 1. Collect and Sort files
    # Using a specific pattern 'loop_*.feather' prevents picking up
    # already merged files if they have a different naming scheme.
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
    # Sort by size so it reads logically: "3 groups of 11, 7 groups of 10"
    dist_str = ", ".join([f"{count} groups of {size}" for size, count in sorted(counts.items(), reverse=True)])

    print(f'Total input files: {len(files)}')
    print(f'Total output files: {len(groups)}')
    print(f'Group distribution: {dist_str}')

    if args.dryrun:
        return

    # 4. Create a temporary output directory
    # Placing it inside input_path ensures it's on the same drive (faster moves)
    input_path = Path(args.input_dir).resolve()
    tmp_outdir = input_path / "temp_merged_processing"

    # Parents=True ensures that even if input_path doesn't exist, it creates the tree
    tmp_outdir.mkdir(parents=True, exist_ok=True)

    # Double check for the user
    if not tmp_outdir.exists():
        raise RuntimeError(f"Failed to create directory at {tmp_outdir}")

    try:
        # 5. Processing Loop
        for idx, igroup in enumerate(tqdm(groups, desc="Merging Groups")):
            nevt_adder = 0
            dfs = []

            for ifile in igroup:
                df = pd.read_feather(ifile)

                # Offset 'evt' column to maintain continuity within the new merged file
                df['evt'] += nevt_adder
                nevt_adder += df['evt'].nunique()

                dfs.append(df)

            # Concatenate and save to temporary location
            # ignore_index=True replaces the need for .reset_index(drop=True)
            final_df = pd.concat(dfs, ignore_index=True)
            final_df.to_feather(tmp_outdir / f'merged_loop_{idx}.feather')

            # Explicitly clear memory
            del dfs
            del final_df

        print('\nProcessing complete. Finalizing files...')

        # 6. Move and Rename merged files
        for kfile in tmp_outdir.glob('merged_loop_*.feather'):
           # Rename 'merged_loop_X' to 'loop_X' in the original directory
           new_name = kfile.name.replace('merged_', '')
           kfile.rename(tmp_outdir / new_name)

        print('Done. All files merged and cleaned.')

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        print("Original files preserved. Check temporary directory for partial results.")

if __name__ == "__main__":
    main()