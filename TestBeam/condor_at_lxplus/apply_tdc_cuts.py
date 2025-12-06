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
    if len(ids_for_corr) >= 2:
        pairs = list(combinations(ids_for_corr, 2))
        corr_mask = pd.Series(True, index=filtered_df.index)

        for bid1, bid2 in pairs:
            x = filtered_df['toa'][bid1]
            y = filtered_df['toa'][bid2]
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
    output_dir: Path
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

        if not final_df.empty:
            prefix = f'exclude_{args.exclude_role}_'
            out_name = f"{prefix}{filepath.stem}.pkl"
            final_df.to_pickle(output_dir / out_name)
            return f"Saved: {out_name}"
        else:
            return f"Empty: {filepath.name}"

    except Exception as e:
        return f"Error processing {filepath.name}: {str(e)}"

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Apply cuts to track files and save final output.")

    # Paths
    parser.add_argument('-d', '--inputdir', required=True, dest='inputdir',
                        help='Mother directory containing "tracks" or "tracks_groupX" folders')

    # Naming Logic
    parser.add_argument('--prefix', default='', help='Add a prefix to the output folder name')
    parser.add_argument('--suffix', default='', help='Add a suffix to the output folder name')

    # Config
    parser.add_argument('-c', '--config', required=True, help='YAML config file')
    parser.add_argument('-r', '--runName', required=True, help='Run name')

    # Cuts
    parser.add_argument('--distance_factor', type=float, default=3.0, help='Correlation cut sigma')
    parser.add_argument('--TOALower', type=int, default=100, help='Raw ToA Lower')
    parser.add_argument('--TOAUpper', type=int, default=500, help='Raw ToA Upper')
    parser.add_argument('--TOALowerTime', type=float, default=2, help='Time ToA Lower (ns)')
    parser.add_argument('--TOAUpperTime', type=float, default=10, help='Time ToA Upper (ns)')
    parser.add_argument('--dutTOTlower', type=int, default=1, help='DUT ToT Lower Percentile')
    parser.add_argument('--dutTOTupper', type=int, default=96, help='DUT ToT Upper Percentile')

    # Flags
    parser.add_argument('--exclude_role', default='trig', help='Role to exclude from CUT calculations')
    parser.add_argument('--convert-first', action='store_true', help='Convert to time before cutting')
    parser.add_argument('--debug', action='store_true', help='Run serial mode for debugging')

    args = parser.parse_args()

    # --- 1. Identify Input/Output Groups ---
    mother_dir = Path(args.inputdir)

    # Find directories starting with "tracks" inside the mother directory
    track_dirs = sorted([d for d in mother_dir.iterdir() if d.is_dir() and d.name.startswith('tracks')])

    if not track_dirs:
        if mother_dir.name.startswith('tracks'):
            track_dirs = [mother_dir]
            mother_dir = mother_dir.parent
        else:
            sys.exit(f"No 'tracks*' directories found in {mother_dir}")

    print(f"\nScanning: {mother_dir}")
    print(f"Found {len(track_dirs)} track groups: {[d.name for d in track_dirs]}")

    # --- 2. Load Config ---
    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.runName not in config:
        sys.exit(f"Run {args.runName} not found in config.")

    all_roles = {}
    cut_roles = []
    for bid, info in config[args.runName].items():
        r = info.get('role')
        if r:
            all_roles[r] = bid
            if r != args.exclude_role:
                cut_roles.append(r)

    # --- 3. Process Each Group ---
    for input_group_dir in track_dirs:
        dir_name = input_group_dir.name
        core_name = dir_name.replace('tracks', 'time')

        # --- NEW LOGIC: Intelligent Underscore Insertion ---
        folder_parts = []
        if args.prefix:
            folder_parts.append(args.prefix)

        folder_parts.append(core_name)

        if args.suffix:
            folder_parts.append(args.suffix)

        final_dirname = "_".join(folder_parts)
        # ---------------------------------------------------

        output_group_dir = mother_dir / final_dirname
        output_group_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n>>> Processing Group: {dir_name}")
        print(f"    Output: {output_group_dir.name}")

        files = natsorted(input_group_dir.glob('track_*.pkl'))

        if not files:
            print(f"    Warning: No track files found in {dir_name}. Skipping.")
            continue

        if args.debug:
            for f in files:
                res = process_single_file(f, args, all_roles, cut_roles, output_group_dir)
                print(res)
        else:
            with ProcessPoolExecutor() as exe:
                futures = {
                    exe.submit(process_single_file, f, args, all_roles, cut_roles, output_group_dir): f
                    for f in files
                }

                for future in tqdm(as_completed(futures), total=len(files), desc=f"    Converting {dir_name}"):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Critical Worker Error: {e}")

    print("\nAll groups processed.")

if __name__ == "__main__":
    main()