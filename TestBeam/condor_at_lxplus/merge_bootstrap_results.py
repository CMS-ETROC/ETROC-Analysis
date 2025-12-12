import argparse
import sys
import re
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from natsort import natsorted
from tqdm import tqdm
from scipy.stats import norm

# Optional imports for binned fitting fallback
try:
    import hist
    from lmfit.models import GaussianModel
    HAS_LMFIT = True
except ImportError:
    HAS_LMFIT = False

# --- Configuration ---
warnings.filterwarnings("ignore")

NICKNAME_DICT = {
    't': 'trig',
    'd': 'dut',
    'r': 'ref',
    'e': 'extra',
}

# Regex to capture role nickname and coordinates from filename
# Matches: "r-R12C34" -> ('r', '12', '34')
FILENAME_PATTERN = re.compile(r"(\w)-R(\d+)C(\d+)")

# --- Helper Functions ---

def fit_binned_gaussian(data: pd.Series, bins: int):
    """
    Performs a binned Gaussian fit using lmfit/hist.
    Returns (mu, sigma, success_bool).
    """
    if not HAS_LMFIT:
        return 0, 0, False

    try:
        x_min, x_max = data.min() - 5, data.max() + 5
        h_temp = hist.Hist(hist.axis.Regular(bins, x_min, x_max))
        h_temp.fill(data)

        centers = h_temp.axes[0].centers
        values = h_temp.values()

        # Fit range: +/- 1.5 sigma (approx 17th-83rd percentile)
        lower = np.percentile(data, 17)
        upper = np.percentile(data, 83)
        mask = (centers > lower) & (centers < upper)

        if np.sum(mask) < 3: # Not enough points to fit
            return 0, 0, False

        mod = GaussianModel(nan_policy='omit')
        pars = mod.guess(values[mask], x=centers[mask])
        out = mod.fit(values[mask], pars, x=centers[mask], weights=1/np.sqrt(values[mask] + 1e-6))

        if out.success:
            return out.params['center'].value, abs(out.params['sigma'].value), True

    except Exception:
        pass

    return 0, 0, False

def calculate_statistics(data: pd.Series, hist_bins: int) -> dict:
    """
    Determines Mean and Sigma for a distribution using:
    1. Unbinned Gaussian Fit (Primary)
    2. Binned Gaussian Fit (Fallback 1)
    3. Simple Statistics (Fallback 2)
    """
    # Strategy 1: Unbinned Fit (Fastest & Standard)
    try:
        mu, sigma = norm.fit(data)
        return {'mu': mu, 'sigma': sigma}
    except Exception:
        pass

    # Strategy 2: Binned Fit (Robust against outliers/noise)
    mu, sigma, success = fit_binned_gaussian(data, hist_bins)
    if success:
        return {'mu': mu, 'sigma': sigma}

    # Strategy 3: Last Resort (Simple Stats)
    return {'mu': data.mean(), 'sigma': data.std()}

def parse_filename_metadata(filename: str) -> dict:
    """Extracts board coordinates from the standard filename format."""
    matches = re.findall(FILENAME_PATTERN, filename)
    pixel_dict = {}
    for nickname, row, col in matches:
        full_role = NICKNAME_DICT.get(nickname)
        if full_role:
            pixel_dict[full_role] = (int(row), int(col))
    return pixel_dict

def process_group(
    input_dir: Path,
    output_dir: Path,
    args: argparse.Namespace
):
    """
    Processes all pickle files in a specific directory (group) and saves a CSV.
    """
    # Find files (exclude_... resolution.pkl)
    files = natsorted(input_dir.glob('*track*'))

    if not files:
        print(f"Warning: No resolution files found in {input_dir.name}. Skipping.")
        return

    # Determine Excluded Role from the first file name convention
    # Expected format: "exclude_trig_..."
    try:
        excluded_role = files[0].name.split('_')[1]
    except IndexError:
        excluded_role = 'trig' # Default fallback

    final_dict = defaultdict(list)

    print(f"Processing {len(files)} files in {input_dir.name}...")

    for ifile in tqdm(files, desc=f"  Merging {input_dir.name}"):
        # 1. Load Data

        if '.pkl' in ifile.name:
            df = pd.read_pickle(ifile)
            # Filter rows that are all zeros (failed bootstraps often return 0s)
            df = df.loc[(df != 0).all(axis=1)].reset_index(drop=True)
            if df.empty:
                continue
        elif '.parquet' in ifile.name:
            df = pd.read_parquet(ifile)
            # Filter rows that are all zeros (failed bootstraps often return 0s)
            df = df.loc[(df != 0).all(axis=1)].reset_index(drop=True)
            if df.empty:
                continue
        else:
            print(f"Error reading {ifile.name}")
            continue

        # 2. Parse Coordinates
        pixel_dict = parse_filename_metadata(str(ifile))

        # 3. Calculate Results
        file_results = {}

        if df.shape[0] == 1:
            # Single-shot result
            row = df.iloc[0]
            for col in df.columns:
                file_results[col] = {'mu': row[col], 'sigma': 0.0}
        else:
            # Bootstrap distribution
            for col in df.columns:
                file_results[col] = calculate_statistics(df[col], args.hist_bins)

        # 4. Append to Final Dictionary
        # We always want the coordinates of the 'excluded' board as the primary key
        # if it exists in the filename metadata.
        if excluded_role in pixel_dict:
            final_dict[f'row_{excluded_role}'].append(pixel_dict[excluded_role][0])
            final_dict[f'col_{excluded_role}'].append(pixel_dict[excluded_role][1])
        else:
            # If excluded role isn't in filename (rare), append placeholders or skip
            pass

        # Append results for all columns found in the dataframe
        for val_name, stats in file_results.items():
            # Add coordinates for this specific board if we have them
            if val_name in pixel_dict:
                final_dict[f'row_{val_name}'].append(pixel_dict[val_name][0])
                final_dict[f'col_{val_name}'].append(pixel_dict[val_name][1])

            final_dict[f'res_{val_name}'].append(stats['mu'])
            final_dict[f'err_{val_name}'].append(stats['sigma'])

    # 5. Save Output
    if final_dict:
        out_filename = f"{input_dir.name}_resolution{args.tag}.csv"
        out_path = output_dir / out_filename

        pd.DataFrame(final_dict).to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")
    else:
        print(f"  No valid data merged for {input_dir.name}.")

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description='Merge individual bootstrap results into a summary CSV.')

    parser.add_argument('-d', '--inputdir', required=True, dest='inputdir',
                        help='Specific bootstrap directory (merges it + siblings) OR Mother directory (merges all)')
    parser.add_argument('-o', '--outputdir', required=True, dest='outputdir',
                        help='Output directory name')

    parser.add_argument('--minimum', type=int, default=50, help='Min bootstrap samples for fit')
    parser.add_argument('--hist_bins', type=int, default=35, help='Bins for binned fit')
    parser.add_argument('--tag', default='', help='Tag for output filename')

    args = parser.parse_args()

    # 1. Identify Groups
    # Resolve the path to handle relative paths (like '.') correctly
    mother_dir = Path(args.inputdir).resolve()

    if not mother_dir.exists():
        sys.exit(f"Error: Input directory {mother_dir} does not exist.")

    group_dirs = []

    # CASE A: User points to a specific bootstrap folder (e.g. bootstrap_Angle30Deg_HV230_os20)
    # We want to process this folder AND any split groups (e.g. ..._group1, ..._group2)
    # but NOT other runs (e.g. ..._HV220).
    if mother_dir.name.startswith('bootstrap'):
        parent_dir = mother_dir.parent
        base_name = mother_dir.name

        # Scan parent directory for matching groups
        for candidate in parent_dir.iterdir():
            if not candidate.is_dir():
                continue

            # Match if it is the base directory exactly
            # OR if it starts with base_name + "_group"
            # (The "_group" check ensures we don't match substrings like HV2 matching HV20)
            if candidate.name == base_name or candidate.name.startswith(base_name + "_group"):
                group_dirs.append(candidate)

        # Sort naturally to ensure group1, group2, group3 order
        group_dirs = natsorted(group_dirs)

    # CASE B: User points to a mother directory (e.g. "./" or "results/")
    # We process ALL bootstrap folders found inside.
    else:
        group_dirs = natsorted([d for d in mother_dir.iterdir() if d.is_dir() and d.name.startswith('bootstrap')])

    if not group_dirs:
        sys.exit(f"No valid 'bootstrap*' directories found based on input: {mother_dir}")

    # 2. Setup Output
    base_out_dir = Path(args.outputdir)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(group_dirs)} groups to merge.")
    print(f"Target Groups: {[d.name for d in group_dirs]}")
    print(f"Output Directory: {base_out_dir}\n")

    # 3. Process Each Group
    for group in group_dirs:
        process_group(group, base_out_dir, args)

    print("\nDone.")

if __name__ == "__main__":
    main()