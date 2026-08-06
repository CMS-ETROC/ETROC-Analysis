import argparse
import re
import sys, getpass

import pandas as pd
import pyarrow.parquet as pq

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from natsort import natsorted
from collections import defaultdict
from tqdm import tqdm
from tabulate import tabulate

import io_utils

# --- Configuration ---
NICKNAME_DICT = {
    't': 'trig',
    'd': 'dut',
    'r': 'ref',
    'e': 'extra',
}

# Regex to capture role nickname and coordinates from filename
# Matches: "r-R12C34" -> ('r', '12', '34')
FILENAME_PATTERN = re.compile(r"(\w)-R(\d+)C(\d+)")

def process_single_file(filepath: Path) -> dict:
    """
    Worker function: Extracts track info from filename and event count from dataframe.
    """
    file_data = defaultdict(list)

    # 1. Parse Filename for Track Coordinates
    matches = re.findall(FILENAME_PATTERN, filepath.name)

    for nickname, row, col in matches:
        full_role = NICKNAME_DICT.get(nickname)
        if full_role:
            file_data[f'row_{full_role}'].append(int(row))
            file_data[f'col_{full_role}'].append(int(col))

    # 2. Optimized Row Counting
    if filepath.suffix == '.parquet':
        # ONLY read metadata, do not load columns!
        meta = pq.read_metadata(filepath)
        nevt = meta.num_rows
    else:
        nevt = 0

    file_data['nevt'].append(nevt)
    return file_data

def generate_summary_table(df: pd.DataFrame, group_name: str):
    """
    Prints the summary table of surviving tracks based on cuts.
    """
    cuts = [1] + list(range(100, 1600, 100))
    table_data = []

    for cut in cuts:
        count = len(df[df['nevt'] > cut])
        table_data.append((f'ntrk > {cut}', count))

    print(f'\n--- Summary for {group_name} ---')
    print(tabulate(table_data, headers=['nTrk Cut', 'Survived Candidates']))
    print('--------------------------------\n')

def process_time_dirs(time_dirs, dirname_part: str, combo_label, output_path: Path, args) -> int:
    """Runs extraction + summary + save for one combo's time/time_groupX dirs.
    Returns the number of files that failed to process."""
    total_failures = 0

    for group_dir in time_dirs:
        group_name = group_dir.name
        label = f'{combo_label}/{group_name}' if combo_label else group_name
        print(f">>> Processing Group: {label}")

        # A. Find Files
        files = natsorted(group_dir.glob('*.parquet'))

        if not files:
            print(f"    Warning: No .parquet files found in {label}. Skipping.")
            continue

        # B. Process Files in Parallel
        group_dict = defaultdict(list)
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_single_file, f) for f in files]

            group_failures = 0
            for future in tqdm(as_completed(futures), total=len(files), desc=f"    Extracting {label}"):
                try:
                    result = future.result()
                    for key, value in result.items():
                        group_dict[key].extend(value)
                except Exception as e:
                    print(f"Worker exception: {e}")
                    group_failures += 1

            if group_failures:
                print(f"    Warning: {group_failures}/{len(files)} file(s) FAILED to process in {label}")
            total_failures += group_failures

        # C. Save and Summarize Specific to this Group
        if group_dict:
            df = pd.DataFrame(data=group_dict)
            df.sort_values(by=['nevt'], ascending=False, inplace=True)

            # --- Filename Logic ---
            # Join non-empty parts with '_' so segments never run together regardless
            # of whether a combo label, group suffix, and/or tag are present. The combo
            # label (if any) comes right after "nevt" so two combos' same-named groups
            # (e.g. both a plain "time" dir) never collide on the same output filename.
            group_match = re.search(r'(group\d+)', group_name)
            name_parts = ["nevt"]
            if combo_label:
                name_parts.append(combo_label)
            name_parts.append(dirname_part)
            if group_match:
                name_parts.append(group_match.group(1))
            if args.tag:
                name_parts.append(args.tag)

            out_filename = "_".join(name_parts) + ".csv"
            final_out_path = output_path / out_filename

            print(f"    Saving to: {final_out_path}")
            io_utils.write_csv(df, final_out_path, index=False)

            generate_summary_table(df, label)
        else:
            print(f"    No data extracted for {label}.")

    return total_failures

def main():
    parser = argparse.ArgumentParser(
        description='Extract event counts per track from time-domain files.',
    )

    parser.add_argument(
        '-d', '--inputdir',
        required=True,
        dest='inputdir',
        help='Mother directory containing "time" or "time_groupX" folders. Can also be the '
             'combo mother directory (one subdirectory per board combo, each with its own '
             '"time"/"time_groupX" folders) -- every combo is then auto-detected and processed.'
    )

    parser.add_argument(
        '-o', '--outputdir',
        required=True,
        dest='outputdir',
        help='Output directory name (created inside inputdir parent usually)'
    )

    parser.add_argument(
        '--tag',
        default='',
        help='Additional tag for the output filename'
    )

    args = parser.parse_args()

    # --- Setup Environments ---
    username = getpass.getuser()
    eos_base_dir = io_utils.eos_base_dir(username)

    mother_dir = eos_base_dir / args.inputdir
    output_path = Path(args.outputdir)
    output_path.mkdir(exist_ok=True, parents=True)

    # 1. Identify the Base Dir Name (used in every combo's output filename)
    if 'time' in mother_dir.name:
        # User pointed directly to a specific time folder (e.g., .../Angle30.../time_group1)
        raw_dirname = mother_dir.parent.name
    else:
        # User pointed to mother dir; scan for subfolders (e.g., .../Angle30.../)
        raw_dirname = mother_dir.name

    # Clean the dirname: remove '_AfterCuts' if present
    dirname_part = raw_dirname.replace('_AfterCuts', '')

    # 2. Try -d as a single combo's mother dir (or a specific time folder itself) first
    time_dirs = io_utils.discover_time_dirs(mother_dir)

    total_failures = 0
    if time_dirs:
        print(f"Found {len(time_dirs)} input groups in base: {dirname_part}")
        print(f"Groups: {[d.name for d in time_dirs]}\n")
        total_failures += process_time_dirs(time_dirs, dirname_part, None, output_path, args)
    else:
        # -d may be the combo mother directory step 10 can write (one
        # subdirectory per board combo, each itself containing
        # time/time_groupX) -- descend one level and process every combo
        # instead of requiring one invocation per combo.
        processed_any = False
        for combo_dir in sorted(d for d in mother_dir.iterdir() if d.is_dir()):
            combo_time_dirs = io_utils.discover_time_dirs(combo_dir)
            if not combo_time_dirs:
                continue
            processed_any = True
            print(f"Found {len(combo_time_dirs)} input groups for combo {combo_dir.name}: "
                  f"{[d.name for d in combo_time_dirs]}\n")
            total_failures += process_time_dirs(combo_time_dirs, dirname_part, combo_dir.name, output_path, args)

        if not processed_any:
            print(f"Error: No directories containing 'time' found in {mother_dir} or its subdirectories")
            sys.exit(1)

    if total_failures:
        print(f"\nDone WITH {total_failures} FAILED FILE(S) -- output may be incomplete.")
        sys.exit(1)

    print("Done.")

if __name__ == "__main__":
    main()