import argparse
import sys
import yaml
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from natsort import natsorted
from typing import Dict, List, Tuple, Optional
from itertools import combinations

# --- Configuration ---
warnings.filterwarnings("ignore")

# --- Helper Functions ---

def calculate_toa_correlation(
    df: pd.DataFrame,
    col1: str,
    col2: str
) -> Tuple[np.ndarray, pd.Series]:
    """Calculates linear fit and perpendicular distance for correlation cut."""
    x = df[col1]
    y = df[col2]

    slope, intercept = np.polyfit(x, y, 1)
    dist = (x * slope - y + intercept) / np.sqrt(slope**2 + 1)

    return (slope, intercept), dist

def apply_correlation_cut(
    df: pd.DataFrame,
    factor: float,
    active_roles: List[str],
    use_time_cols: bool = False
) -> pd.DataFrame:
    if df.empty or len(active_roles) < 2:
        return df

    pairs = list(combinations(active_roles, 2))
    mask = pd.Series(True, index=df.index)

    for r1, r2 in pairs:
        if use_time_cols:
            c1, c2 = f'toa_{r1}', f'toa_{r2}'
            if c1 not in df.columns or c2 not in df.columns: continue
            _, dist = calculate_toa_correlation(df, c1, c2)
        else:
            continue

        limit = factor * np.nanstd(dist)
        mask &= (dist < limit)

    return df.loc[mask].reset_index(drop=True)

def convert_to_time(df: pd.DataFrame, all_roles: Dict[str, int]) -> pd.DataFrame:
    out_df = pd.DataFrame()

    bins = {
        role: 3.125 / df['cal'][bid].mean()
        for role, bid in all_roles.items()
    }

    for role, bid in all_roles.items():
        bin_size = bins[role]
        raw_tot = df['tot'][bid]
        out_df[f'tot_{role}'] = ((2 * raw_tot - np.floor(raw_tot / 32.)) * bin_size) * 1e3

        raw_toa = df['toa'][bid]
        out_df[f'toa_{role}'] = (12.5 - raw_toa * bin_size) * 1e3

    return out_df

def apply_raw_tdc_cuts(
    df: pd.DataFrame,
    all_roles: Dict[str, int],
    cut_roles: List[str],
    args: argparse.Namespace
) -> pd.DataFrame:

    dut_id = all_roles.get('dut', all_roles.get('extra'))
    trig_id = all_roles.get('trig', all_roles.get('ref'))
    dut_pct = [args.dutTOTlower * 0.01, args.dutTOTupper * 0.01]

    mask = pd.Series(True, index=df.index)

    for role, bid in all_roles.items():
        if bid == dut_id:
            low, high = df['tot'][bid].quantile(dut_pct)
        else:
            low, high = df['tot'][bid].quantile([0.01, 0.96])

        toa_low = args.TOALower if bid == trig_id else 0
        toa_high = args.TOAUpper if bid == trig_id else 1100

        board_mask = (
            df['cal'][bid].between(0, 1100) &
            df['toa'][bid].between(toa_low, toa_high) &
            df['tot'][bid].between(low, high)
        )
        mask &= board_mask

    filtered_df = df.loc[mask].reset_index(drop=True)
    if filtered_df.empty: return filtered_df

    ids_for_corr = sorted([all_roles[r] for r in cut_roles])

    ### Need to fix
    if len(ids_for_corr) >= 2:
        pairs = list(combinations(ids_for_corr, 2))
        corr_mask = pd.Series(True, index=filtered_df.index)

        for bid1, bid2 in pairs:
            x = pd.to_numeric(filtered_df['toa'][bid1])
            y = pd.to_numeric(filtered_df['toa'][bid2])
            m, c = np.polyfit(x, y, 1)
            dist = (x * m - y + c) / np.sqrt(m**2 + 1)
            limit = args.distance_factor * np.nanstd(dist)
            corr_mask &= (dist < limit)

        filtered_df = filtered_df.loc[corr_mask]

    return filtered_df

def apply_time_domain_cuts(
    df: pd.DataFrame,
    cut_roles: List[str],
    args: argparse.Namespace
) -> pd.DataFrame:

    ref_col = 'toa_ref' if 'toa_ref' in df else ('toa_extra' if 'toa_extra' in df else 'toa_trig')
    low_ns = args.TOALowerTime * 1e3
    high_ns = args.TOAUpperTime * 1e3

    df = df.loc[df[ref_col].between(low_ns, high_ns)]
    if df.empty: return df

    mask = pd.Series(True, index=df.index)
    dut_pct = [args.dutTOTlower * 0.01, args.dutTOTupper * 0.01]

    for col in [c for c in df.columns if c.startswith('tot')]:
        if 'dut' in col:
            low, high = df[col].quantile(dut_pct)
        else:
            low, high = df[col].quantile([0.01, 0.96])
        mask &= df[col].between(low, high)

    df = df.loc[mask].reset_index(drop=True)

    if not df.empty:
        df = apply_correlation_cut(df, args.distance_factor, cut_roles, use_time_cols=True)

    return df

def process_single_file(
    filepath: Path,
    args: argparse.Namespace,
    all_roles: Dict[str, int],
    cut_roles: List[str],
) -> str:
    """Worker function."""
    try:
        df = pd.read_pickle(filepath)
        final_df = pd.DataFrame()

        if args.convert_first:
            time_df = convert_to_time(df, all_roles)
            if not time_df.empty:
                final_df = apply_time_domain_cuts(time_df, cut_roles, args)
        else:
            if df.shape[0] < 1000:
                cut_df = apply_raw_tdc_cuts(df, all_roles, cut_roles, args)
            else:
                chunks = []
                for fid in df['file'].unique():
                    sub = df.loc[df['file'] == fid]
                    sub_cut = apply_raw_tdc_cuts(sub, all_roles, cut_roles, args)
                    if not sub_cut.empty:
                        chunks.append(sub_cut)
                cut_df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

            if not cut_df.empty:
                final_df = convert_to_time(cut_df, all_roles)

                #### Add neighbor columns
                for role, board_id in all_roles.items():
                    try:
                        final_df[f'HasNeighbor_{role}'] = cut_df['HasNeighbor'][board_id].astype(bool)
                    except:
                        final_df[f'HasNeighbor_{role}'] = False

                ## Add board level neighbor column
                neighbor_columns = [col for col in final_df.columns if col.startswith('HasNeighbor')]
                final_df['trackNeighbor'] = final_df[neighbor_columns].any(axis=1)

        if not final_df.empty:
            prefix = f'exclude_{args.exclude_role}_'
            out_name = f"{prefix}{filepath.stem}.parquet"
            final_df.to_parquet(out_name)
            return f"Saved: {out_name}"
        else:
            return f"Empty: {filepath.name}"

    except Exception as e:
        return f"Error processing {filepath.name}: {str(e)}"

# --- Main ---

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Apply cuts to track files and save final output.")

    # Paths
    parser.add_argument('-d', '--inputdir', required=True, dest='inputdir',
                        help='Directory containing track files (passed by Condor)')

    # Config (REQUIRED for Roles)
    parser.add_argument('-c', '--config', required=True, help='YAML config file')
    parser.add_argument('-r', '--runName', required=True, help='Run name in config')

    # Cuts (simplified for worker)
    parser.add_argument('--distance_factor', type=float, default=3.0)
    parser.add_argument('--TOALower', type=int, default=100)
    parser.add_argument('--TOAUpper', type=int, default=500)
    parser.add_argument('--TOALowerTime', type=float, default=2)
    parser.add_argument('--TOAUpperTime', type=float, default=10)
    parser.add_argument('--dutTOTlower', type=int, default=1)
    parser.add_argument('--dutTOTupper', type=int, default=96)

    # Flags
    parser.add_argument('--exclude_role', default='trig')
    parser.add_argument('--convert-first', action='store_true')

    args = parser.parse_args()

    # 1. Load Config & Roles
    try:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file {args.config} not found locally on worker node.")
        sys.exit(1)

    if args.runName not in config:
        sys.exit(f"Error: Run {args.runName} not found in {args.config}")

    all_roles = {}
    cut_roles = []
    for bid, info in config[args.runName].items():
        r = info.get('role')
        if r:
            all_roles[r] = bid
            if r != args.exclude_role:
                cut_roles.append(r)

    # 2. Find Files
    input_path = Path(args.inputdir)
    # Condor will have transferred the directory contents, so input_path is now local.
    files = natsorted(input_path.glob('track_*.pkl'))

    if not files:
        print(f"Warning: No track files found in {input_path}. Exit.")
        # Exit 0 means successful completion, even if no files were found (prevents Condor errors)
        sys.exit(0)

    print(f"Processing {len(files)} files from {input_path}...")

    # 3. Loop sequentially on the worker node
    for f in tqdm(files, desc="Processing"):
        print(process_single_file(f, args, all_roles, cut_roles))
        gc.collect()

    print("\nAll files processed.")

if __name__ == "__main__":
    main()