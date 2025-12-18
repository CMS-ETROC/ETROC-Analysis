import argparse
import sys
import yaml
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from natsort import natsorted
from itertools import combinations

# --- Configuration ---
warnings.filterwarnings("ignore")

# --- Helper Functions ---

def calculate_toa_correlation(
    df: pd.DataFrame,
    col1: str,
    col2: str
) -> tuple[np.ndarray, pd.Series]:
    """Calculates linear fit and perpendicular distance for correlation cut."""
    x = df[col1]
    y = df[col2]

    # Fit a line to the TOA correlation between two boards
    slope, intercept = np.polyfit(x, y, 1)
    # Calculate distance of each point from the fit line
    dist = (x * slope - y + intercept) / np.sqrt(slope**2 + 1)

    return (slope, intercept), dist

def apply_correlation_cut(
    df: pd.DataFrame,
    factor: float,
    active_roles: list[str],
) -> pd.DataFrame:
    """Removes events that fall outside the TOA correlation distance factor."""
    if df.empty or len(active_roles) < 2:
        return df

    pairs = list(combinations(active_roles, 2))
    mask = pd.Series(True, index=df.index)

    for r1, r2 in pairs:
        c1, c2 = f'toa_{r1}', f'toa_{r2}'
        if c1 not in df.columns or c2 not in df.columns:
            continue

        _, dist = calculate_toa_correlation(df, c1, c2)

        # Median of the absolute deviations from the median
        median_val = np.nanmedian(dist)
        mad = np.nanmedian(np.abs(dist - median_val))

        # Convert MAD to a sigma-equivalent
        limit = factor * mad * 1.4826

        # Center the mask on the median, not zero, for robustness
        mask &= (np.abs(dist - median_val) < limit)

    return df.loc[mask].reset_index(drop=True)

def convert_to_time(df: pd.DataFrame, all_roles: dict[str, int]) -> pd.DataFrame:
    """Calculates physical time units, ensuring bin_size is calculated per-file."""
    processed_chunks = []

    # Group by 'file' to handle calibration drifts between data runs
    for _, group in df.groupby('file'):
        out_chunk = pd.DataFrame(index=group.index)

        for role in all_roles.keys():
            # Mean CAL is used to derive the bin_size for this specific file
            mean_cal = group[f'cal_{role}'].mean()
            bin_size = 3.125 / mean_cal if mean_cal != 0 else 0.2

            # Conversion formulas to nanoseconds and picoseconds
            raw_tot = group[f'tot_{role}']
            out_chunk[f'tot_{role}'] = ((2 * raw_tot - np.floor(raw_tot / 32.)) * bin_size) * 1e3

            raw_toa = group[f'toa_{role}']
            out_chunk[f'toa_{role}'] = (12.5 - raw_toa * bin_size) * 1e3

        processed_chunks.append(out_chunk)

    return pd.concat(processed_chunks).sort_index() if processed_chunks else pd.DataFrame()

def apply_raw_tdc_cuts(
    df: pd.DataFrame,
    all_roles: dict[str, int],
    cut_roles: list[str],
    args: argparse.Namespace
) -> pd.DataFrame:
    """Filters raw TDC values using file-specific thresholds and config roles."""
    dut_role = 'dut' if 'dut' in all_roles else 'extra'
    trig_role = 'ref' if 'ref' in all_roles else 'trig'
    chunks = []

    # Iterate per file to ensure quantile cuts are relative to local distributions
    for fid in df['file'].unique():
        sub = df.loc[df['file'] == fid]
        mask = pd.Series(True, index=sub.index)

        for role in all_roles.keys():
            if role not in cut_roles:
                continue

            # Determine TOT bounds for DUT vs others
            if role == dut_role:
                if args.dutTOTlowerVal != -1 or args.dutTOTupperVal != -1:
                    low = args.dutTOTlowerVal if args.dutTOTlowerVal != -1 else -np.inf
                    high = args.dutTOTupperVal if args.dutTOTupperVal != -1 else np.inf
                else:
                    low, high = sub[f'tot_{role}'].quantile([args.dutTOTlower * 0.01, args.dutTOTupper * 0.01])
            else:
                low, high = sub[f'tot_{role}'].quantile([0.01, 0.96])

            # Apply TOA window primarily for the trigger-assigned board
            toa_low = args.TOALower if role == trig_role else 0
            toa_high = args.TOAUpper if role == trig_role else 1100

            board_mask = (
                sub[f'cal_{role}'].between(0, 1100) &
                sub[f'toa_{role}'].between(toa_low, toa_high) &
                sub[f'tot_{role}'].between(low, high)
            )
            mask &= board_mask

        sub_cut = sub.loc[mask]
        if not sub_cut.empty:
            chunks.append(sub_cut)

    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

def apply_time_domain_cuts(
    df: pd.DataFrame,
    cut_roles: list[str],
    args: argparse.Namespace
) -> pd.DataFrame:
    """Applies cuts in the physical time domain (picoseconds)."""
    # Use 'toa_ref' if present, otherwise fallback to 'toa_trig'
    ref_col = 'toa_ref' if 'toa_ref' in df else 'toa_trig'
    low_ps = args.TOALowerTime * 1e3
    high_ps = args.TOAUpperTime * 1e3

    df = df.loc[df[ref_col].between(low_ps, high_ps)]
    if df.empty:
        return df

    mask = pd.Series(True, index=df.index)

    for col in [c for c in df.columns if c.startswith('tot_')]:
        role = col.replace('tot_', '')
        if role not in cut_roles:
            continue

        if role == 'dut':
            if args.dutTOTlowerTime != -1 or args.dutTOTupperTime != -1:
                low = args.dutTOTlowerTime * 1e3 if args.dutTOTlowerTime != -1 else -np.inf
                high = args.dutTOTupperTime * 1e3 if args.dutTOTupperTime != -1 else np.inf
            else:
                low, high = df[col].quantile([args.dutTOTlower * 0.01, args.dutTOTupper * 0.01])
        else:
            low, high = df[col].quantile([0.01, 0.96])

        mask &= df[col].between(low, high)

    df = df.loc[mask].reset_index(drop=True)
    return apply_correlation_cut(df, args.distance_factor, cut_roles)

def process_single_file(
    filepath: Path,
    args: argparse.Namespace,
    all_roles: dict[str, int],
    cut_roles: list[str],
) -> str:
    """Worker function for Parquet-only processing."""
    try:
        # Load flat Parquet table
        df = pd.read_parquet(filepath)
        if df.empty:
            return f"Empty: {filepath.name}"

        # 1. Routing logic for conversion and cuts
        if args.convert_first:
            time_df = convert_to_time(df, all_roles)
            final_df = apply_time_domain_cuts(time_df, cut_roles, args)
            raw_df_to_use = df.loc[final_df.index] if not final_df.empty else pd.DataFrame()
        else:
            cut_df = apply_raw_tdc_cuts(df, all_roles, cut_roles, args)
            final_df = convert_to_time(cut_df, all_roles)
            raw_df_to_use = cut_df

        # 2. Add Neighbor tracking
        if not final_df.empty and not raw_df_to_use.empty:
            for role in all_roles.keys():
                col_name = f'HasNeighbor_{role}'
                final_df[col_name] = raw_df_to_use[col_name].astype(bool)

            # Track-level neighbor flag
            neighbor_columns = [col for col in final_df.columns if col.startswith('HasNeighbor')]
            final_df['trackNeighbor'] = final_df[neighbor_columns].any(axis=1)

            prefix = f'exclude_{args.exclude_role}_'
            out_name = f"{prefix}{filepath.stem}.parquet"
            final_df.to_parquet(out_name, compression='lz4')
            return f"Saved: {out_name}"
        else:
            return f"Filtered to Empty: {filepath.name}"

    except Exception as e:
        return f"Error processing {filepath.name}: {str(e)}"

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Apply cuts to Parquet track files.")

    parser.add_argument('-d', '--inputdir', required=True, help='Directory containing track parquet files')
    parser.add_argument('-c', '--config', required=True, help='YAML config file')
    parser.add_argument('-r', '--runName', required=True, help='Run name in config')

    parser.add_argument('--distance_factor', type=float, default=3.0)
    parser.add_argument('--TOALower', type=int, default=100)
    parser.add_argument('--TOAUpper', type=int, default=500)
    parser.add_argument('--TOALowerTime', type=float, default=2)
    parser.add_argument('--TOAUpperTime', type=float, default=10)
    parser.add_argument('--dutTOTlower', type=int, default=1)
    parser.add_argument('--dutTOTupper', type=int, default=96)
    parser.add_argument('--dutTOTlowerVal', type=float, default=-1)
    parser.add_argument('--dutTOTupperVal', type=float, default=-1)
    parser.add_argument('--dutTOTlowerTime', type=float, default=-1)
    parser.add_argument('--dutTOTupperTime', type=float, default=-1)

    parser.add_argument('--exclude_role', default='trig')
    parser.add_argument('--convert-first', action='store_true')

    args = parser.parse_args()

    # 1. Load Config & Mapping
    with open(args.config) as f:
        config = yaml.safe_load(f)

    all_roles = {}
    cut_roles = []
    for bid, info in config[args.runName].items():
        role = info.get('role')
        if role:
            all_roles[role] = bid
            if role != args.exclude_role:
                cut_roles.append(role)

    # 2. Find Files (Parquet Only)
    input_path = Path(args.inputdir)
    files = natsorted(input_path.glob('track_*.parquet'))

    if not files:
        print("No parquet files found. Exit.")
        sys.exit(0)

    # 3. Process
    for f in tqdm(files, desc="Processing Tracks"):
        print(process_single_file(f, args, all_roles, cut_roles))

if __name__ == "__main__":
    main()