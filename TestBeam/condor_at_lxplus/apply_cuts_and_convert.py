from natsort import natsorted
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np
import warnings
import argparse, yaml

warnings.filterwarnings("ignore")

## --------------------------------------
def tdc_event_selection_pivot(
        input_df: pd.DataFrame,
        tdc_cuts_dict: dict
    ) -> pd.DataFrame:
    combined_mask = pd.Series(True, index=input_df.index)
    for board, cuts in tdc_cuts_dict.items():
        mask = (
            input_df['cal'][board].between(cuts[0], cuts[1]) &
            input_df['toa'][board].between(cuts[2], cuts[3]) &
            input_df['tot'][board].between(cuts[4], cuts[5])
        )
        combined_mask &= mask
    return input_df[combined_mask].reset_index(drop=True)

## --------------------------------------
def return_TOA_correlation_param(
        input_df: pd.DataFrame,
        board_id1: int,
        board_id2: int,
    ):

    x = input_df['toa'][board_id1]
    y = input_df['toa'][board_id2]

    params = np.polyfit(x, y, 1)
    distance = (x*params[0] - y + params[1])/(np.sqrt(params[0]**2 + 1))

    return params, distance

## --------------------------------------
def apply_toa_time_correlation_cut(
        input_df: pd.DataFrame,
        factor: float,
    ):

    from itertools import combinations

    distances = []
    roles = input_df.columns.str.split('_').str.get(-1).unique()
    role_pairs = list(combinations(roles, 2))

    for role1, role2 in role_pairs:
        if f'toa_{role1}' in input_df.columns and f'toa_{role2}' in input_df.columns:
            x = input_df[f'toa_{role1}']
            y = input_df[f'toa_{role2}']

            params = np.polyfit(x, y, 1)
            distance = (x*params[0] - y + params[1])/(np.sqrt(params[0]**2 + 1))
            distances.append(distance)
        else:
            # Handle the case where a column is missing
            print(f"Warning: Columns for {role1} or {role2} not found. Skipping.")
            continue

    dist_cut = pd.Series(True, index=input_df.index)

    for dist in distances:
        # Calculate the standard deviation and apply the cut
        std = np.nanstd(dist)
        cut_mask = (dist < factor * std)
        dist_cut &= cut_mask

    # Apply the final combined cut to the DataFrame
    return input_df.loc[dist_cut].reset_index(drop=True)

## --------------------------------------
def convert_code_to_time(
        input_df: pd.DataFrame,
        board_roles: dict,
        new_toa: bool = False,
):
    tmp_df = pd.DataFrame()
    board_bins = {idx: 3.125 / input_df['cal'][idx].mean() for _, idx in board_roles.items()}

    for role, idx in board_roles.items():
        bins = board_bins[idx]
        tmp_df[f'tot_{role}'] = ((2 * input_df['tot'][idx] - np.floor(input_df['tot'][idx] / 32.)) * bins) * 1e3

        if not new_toa:
            tmp_df[f'toa_{role}'] = (12.5 - input_df['toa'][idx] * bins) * 1e3
        else:
            origin_toa = (input_df['toa'][idx] * bins) * 1e3
            second_toa = ((input_df['toa'][idx]+input_df['cal'][idx]) * bins) * 1e3
            tmp_df[f'toa_{role}'] = 12500 - (0.5*(origin_toa + second_toa - 3125))

    return tmp_df

## --------------------------------------
def apply_TDC_cuts(
        args,
        input_df: pd.DataFrame,
        board_roles: dict,
    ):

    dut_lowerTOT = args.dutTOTlower * 0.01
    dut_upperTOT = args.dutTOTupper * 0.01

    df_in_time = pd.DataFrame()

    dut_id = board_roles.get('dut')
    if dut_id is None:
        dut_id = board_roles.get('extra')

    trig_id = board_roles.get('trig')
    if trig_id is None:
        trig_id = board_roles.get('ref')

    ### Apply TDC cut
    tot_cuts = {
        idx: list(input_df['tot'][idx].quantile(
            [dut_lowerTOT, dut_upperTOT] if dut_id is not None and idx == dut_id else [0.01, 0.96]
        ).values)
        for _, idx in board_roles.items()
    }

    tdc_cuts = {
        idx: [
            0, 1100,
            args.TOALower if trig_id is not None and idx == trig_id else 0,
            args.TOAUpper if trig_id is not None and idx == trig_id else 1100,
            *tot_cuts[idx]
        ] for _, idx in board_roles.items()
    }
    interest_df = tdc_event_selection_pivot(input_df, tdc_cuts_dict=tdc_cuts)

    if not interest_df.empty:

        board_ids = sorted(board_roles.values())

        # --- Apply TOA correlation cut
        _, distance1 = return_TOA_correlation_param(interest_df, board_id1=board_ids[0], board_id2=board_ids[1])
        _, distance2 = return_TOA_correlation_param(interest_df, board_id1=board_ids[0], board_id2=board_ids[2])
        _, distance3 = return_TOA_correlation_param(interest_df, board_id1=board_ids[1], board_id2=board_ids[2])

        std1 = np.nanstd(distance1)
        std2 = np.nanstd(distance2)
        std3 = np.nanstd(distance3)

        dist_cut = (distance1 < args.distance_factor * std1) & \
                    (distance2 < args.distance_factor * std2) & \
                    (distance3 < args.distance_factor * std3)

        reduced_df = interest_df.loc[dist_cut]

        if not reduced_df.empty:
            df_in_time = convert_code_to_time(reduced_df, board_roles, args.use_new_toa)

    return df_in_time


## --------------------------------------
def apply_TDC_time_cuts(
        args,
        input_df: pd.DataFrame,
    ):

    dut_lowerTOT = args.dutTOTlower * 0.01
    dut_upperTOT = args.dutTOTupper * 0.01

    lowerTOA = args.TOALowerTime * 1e3
    upperTOA = args.TOAUpperTime * 1e3

    ### Apply simple TOA cut
    toa_cut_on_board = 'ref'

    if not f'toa_ref' in input_df.columns:
        toa_cut_on_board = 'extra'

        if not 'toa_extra' in input_df.columns:
            toa_cut_on_board = 'trig'

    df_after_toa_cut = input_df.loc[(input_df[f'toa_{toa_cut_on_board}'] >= lowerTOA) & (input_df[f'toa_{toa_cut_on_board}'] <= upperTOA)]

    ### Apply percentile TOT cut
    tot_cols = [col for col in df_after_toa_cut.columns if col.startswith('tot')]
    tot_cuts = {}
    # Apply the percentile cut to each 'tot' column
    for col in tot_cols:
        tot_cuts[col] = list(df_after_toa_cut[col].quantile(
            [dut_lowerTOT, dut_upperTOT] if col == 'tot_dut' else [0.01, 0.96]
        ))

    combined_mask = pd.Series(True, index=df_after_toa_cut.index)
    for role, cuts in tot_cuts.items():
        mask = df_after_toa_cut[f'tot_{role}'].between(cuts[0], cuts[1])
        combined_mask &= mask

    interest_df = df_after_toa_cut.loc[combined_mask].reset_index(drop=True)
    del df_after_toa_cut

    if not interest_df.empty:

        reduced_df = apply_toa_time_correlation_cut(interest_df, args.distance_factor)

        if not reduced_df.empty:
            return reduced_df

# --- This would be your new worker function for parallel processing ---
def process_track_file(track_filepath, args, board_roles, final_output_dir):
    """
    Opens a single track file, applies cuts, and saves the final output.
    """
    track_df = pd.read_pickle(track_filepath)
    df_in_time = pd.DataFrame()

    if args.convert_first:
        tmp_df = convert_code_to_time(track_df, board_roles)
        if not tmp_df.empty:
            df_in_time = apply_TDC_time_cuts(args, tmp_df)

    else:
        if track_df.shape[0] < 1000:
            # use single mean
            df_in_time = apply_TDC_cuts(args, track_df, board_roles)

        else:
            dfs = []
            for file_id in track_df['file'].unique():
                partial_track_df = track_df.loc[track_df['file'] == file_id]
                partial_df_in_time = apply_TDC_cuts(args, partial_track_df, board_roles)
                if not partial_df_in_time.empty:
                    dfs.append(partial_df_in_time)

            df_in_time = pd.concat(dfs, ignore_index=True)

    if not df_in_time.empty:
        prefix = f'exclude_{args.exclude_role}_'
        output_name = f"{prefix}{track_filepath.stem}.pkl" # Use stem to get filename without .pkl
        final_output_path = final_output_dir / output_name
        df_in_time.to_pickle(final_output_path)
        return f"Processed {track_filepath.name}"

    else:
        return f"Skipped {track_filepath.name} (no data after cuts)"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Apply cuts to track files and save final output.")

    parser.add_argument(
        '-d',
        '--inputdir',
        metavar = 'INPUTNAME',
        type = str,
        help = 'input directory name',
        required = True,
        dest = 'inputdir',
    )

    parser.add_argument(
        '-o',
        '--outdir',
        metavar = 'OUTNAME',
        type = str,
        help = 'output directory path',
        required = True,
        dest = 'outdir',
    )

    parser.add_argument(
        '--dirname',
        metavar = 'NAME',
        type = str,
        help = 'output directory name',
        default = 'time',
        dest = 'dirname',
    )

    parser.add_argument(
        '-r',
        '--runName',
        metavar = 'NAME',
        type = str,
        help = 'Name of the run to process. It must be matched with the name defined in YAML.',
        required = True,
        dest = 'runName',
    )

    parser.add_argument(
        '-c',
        '--config',
        metavar = 'NAME',
        type = str,
        help = 'YAML file including run information.',
        required = True,
        dest = 'config',
    )

    parser.add_argument(
        '--distance_factor',
        metavar = 'NUM',
        type = float,
        help = 'A factor to set boundary cut size. e.g. factor*nan.std(distance)',
        default = 3.0,
        dest = 'distance_factor',
    )

    parser.add_argument(
        '--TOALower',
        metavar = 'NUM',
        type = int,
        help = 'Lower TOA code selection boundary',
        default = 100,
        dest = 'TOALower',
    )

    parser.add_argument(
        '--TOAUpper',
        metavar = 'NUM',
        type = int,
        help = 'Upper TOA code selection boundary',
        default = 500,
        dest = 'TOAUpper',
    )

    parser.add_argument(
        '--TOALowerTime',
        metavar = 'NUM',
        type = float,
        help = 'Lower TOA time (in ns) selection boundary',
        default = 2,
        dest = 'trigTOALowerTime',
    )

    parser.add_argument(
        '--TOAUpperTime',
        metavar = 'NUM',
        type = float,
        help = 'Upper TOA time (in ns) selection boundary',
        default = 10,
        dest = 'trigTOAUpperTime',
    )

    parser.add_argument(
        '--dutTOTlower',
        metavar = 'NUM',
        type = int,
        help = 'Lower TOT boundary for the DUT board. Only relevant when --reprocess option is on.',
        default = 4,
        dest = 'dutTOTlower',
    )

    parser.add_argument(
        '--dutTOTupper',
        metavar = 'NUM',
        type = int,
        help = 'Upper TOT boundary for the DUT board. Only relevant when --reprocess option is on.',
        default = 91,
        dest = 'dutTOTupper',
    )

    parser.add_argument(
        '--exclude_role',
        metavar = 'NAME',
        type = str,
        help = "Choose the board to exclude for calculating TWC coeffs. Possible option: 'trig', 'dut', 'ref', 'extra'",
        default = 'trig',
        dest = 'exclude_role',
    )

    parser.add_argument(
        '--use_new_toa',
        action = 'store_true',
        help = 'If set, use average of TOA and TOA+CAL as a new toa in time',
        dest = 'use_new_toa',
    )

    parser.add_argument(
        '--convert-first',
        action='store_true',
        help='If set, converts to time domain first, then applies cuts.',
        dest='convert_first',
    )

    parser.add_argument(
        '--debug',
        action = 'store_true',
        help = 'If set, switch to loop mode to print error message',
        dest = 'debug',
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    final_output_path = outdir / args.dirname
    final_output_path.mkdir(exist_ok=True)

    with open(args.config) as input_yaml:
        config = yaml.safe_load(input_yaml)

    if args.runName not in config:
        raise ValueError(f"Run config {args.runName} not found")

    id_role_map = {}
    for board_id, board_info in config[args.runName].items():
        role = board_info.get('role')
        if role != args.exclude_role:
            id_role_map[role] = board_id

    track_files = natsorted(Path(args.inputdir).glob('track_*.pkl'))

    if args.debug:
        for f in track_files:
            process_track_file(f, args, id_role_map, final_output_path)
            break
    else:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_track_file, f, args, id_role_map, final_output_path) for f in track_files]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Saving Tracks"):
                try:
                    # This is the crucial line. It will re-raise any exception
                    # that happened in the worker process.
                    future.result()
                except Exception as exc:
                    print(f"A worker process generated an exception: {exc}")
                    # For a full error report, uncomment the next two lines
                    # import traceback
                    # traceback.print_exc()
                finally:
                    pass