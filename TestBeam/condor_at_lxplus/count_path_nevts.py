import pandas as pd
import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from natsort import natsorted
from collections import defaultdict
from tqdm import tqdm
from tabulate import tabulate

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

    # 2. Open Pickle to get Event Count
    if '.pkl' in filepath.name:
        df = pd.read_pickle(filepath)
        nevt = len(df)
        del df
    elif '.parquet' in filepath.name:
        df = pd.read_parquet(filepath)
        nevt = len(df)
        del df
    else:
        print(f"Error reading {filepath.name}")
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

def main():
    parser = argparse.ArgumentParser(
        description='Extract event counts per track from time-domain files.',
    )

    parser.add_argument(
        '-d', '--inputdir',
        required=True,
        dest='inputdir',
        help='Mother directory containing "time" or "time_groupX" folders'
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

    mother_dir = Path(args.inputdir)
    output_path = Path(args.outputdir)
    output_path.mkdir(exist_ok=True, parents=True)

    # 1. Identify Input Groups and Base Dir Name
    if mother_dir.name.find('time') != -1:
        # User pointed directly to a specific time folder (e.g., .../Angle30.../time_group1)
        time_dirs = [mother_dir]
        raw_dirname = mother_dir.parent.name
    else:
        # User pointed to mother dir; scan for subfolders (e.g., .../Angle30.../)
        time_dirs = sorted([d for d in mother_dir.iterdir() if d.is_dir() and 'time' in d.name])
        raw_dirname = mother_dir.name

    if not time_dirs:
        print(f"Error: No directories containing 'time' found in {mother_dir}")
        sys.exit(1)

    # Clean the dirname: remove '_AfterCuts' if present
    dirname_part = raw_dirname.replace('_AfterCuts', '')

    print(f"Found {len(time_dirs)} input groups in base: {dirname_part}")
    print(f"Groups: {[d.name for d in time_dirs]}\n")

    # 2. Loop through each group independently
    for group_dir in time_dirs:
        group_name = group_dir.name
        print(f">>> Processing Group: {group_name}")

        # A. Find Files
        files = natsorted(group_dir.glob('exclude*.pkl'))
        if not files:
            files = natsorted(group_dir.glob('*.parquet'))

        if not files:
            print(f"    Warning: No .pkl or .parquet files found in {group_name}. Skipping.")
            continue

        # B. Process Files in Parallel
        group_dict = defaultdict(list)
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_single_file, f) for f in files]

            for future in tqdm(as_completed(futures), total=len(files), desc=f"    Extracting {group_name}"):
                try:
                    result = future.result()
                    for key, value in result.items():
                        group_dict[key].extend(value)
                except Exception as e:
                    print(f"Worker exception: {e}")

        # C. Save and Summarize Specific to this Group
        if group_dict:
            df = pd.DataFrame(data=group_dict)
            df.sort_values(by=['nevt'], ascending=False, inplace=True)

            # --- Modified Filename Logic ---
            # 1. dirname_part is already calculated (e.g., Angle30Deg_HV160_os10)
            # 2. Determine suffix based on group name (e.g., time_group1 -> _group1)
            group_match = re.search(r'(group\d+)', group_name)
            if group_match:
                # If "group1" exists in the folder name, append "_group1"
                group_suffix = f"_{group_match.group(1)}"
            else:
                # If folder is just "time", no suffix
                group_suffix = ""

            # Final name: nevt_<dirname_part>_<group_suffix>.csv
            out_filename = f"nevt_{dirname_part}{group_suffix}{args.tag}.csv"
            final_out_path = output_path / out_filename

            print(f"    Saving to: {final_out_path}")
            df.to_csv(final_out_path, index=False)

            generate_summary_table(df, group_name)
        else:
            print(f"    No data extracted for {group_name}.")

    print("Done.")

if __name__ == "__main__":
    main()