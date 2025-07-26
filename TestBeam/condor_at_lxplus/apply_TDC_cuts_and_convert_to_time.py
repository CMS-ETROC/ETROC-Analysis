from natsort import natsorted
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict

import pandas as pd
import numpy as np
import yaml
import warnings
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

    removed_role = board_roles.pop(args.exclude_role)

    dut_lowerTOT = args.dutTOTlower * 0.01
    dut_upperTOT = args.dutTOTupper * 0.01

    df_in_time = pd.DataFrame()

    dut_id = board_roles.get('dut')
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
            args.trigTOALower if trig_id is not None and idx == trig_id else 0,
            args.trigTOAUpper if trig_id is not None and idx == trig_id else 1100,
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
def process_single_track(args, track_dfs: dict, board_roles: dict, save_track_dir: Path, save_time_dir: Path):
    concatenated_track_df = pd.concat(track_dfs, ignore_index=True)
    time_dfs = []

    for file_id in sorted(concatenated_track_df['file'].unique()):
        df_file = concatenated_track_df.loc[concatenated_track_df['file'] == file_id]

        if df_file.empty:
            continue

        df_in_time = apply_TDC_cuts(args, df_file, board_roles)
        time_dfs.append(df_in_time)

    concatenated_time_df = pd.concat(time_dfs, ignore_index=True)

    ### Save dataframes
    board_ids_for_naming = [
        b for b in concatenated_track_df.columns.get_level_values('board').unique()
        if isinstance(b, int)
    ]

    row_cols = {
        idx: (concatenated_track_df['row'][idx].unique()[0], concatenated_track_df['col'][idx].unique()[0])
        for idx in board_ids_for_naming
    }

    role_by_index = {value: key for key, value in board_roles.items()}
    nickname_dict = {
        'trig': '_t-',
        'dut': '_d-',
        'ref': '_r-',
        'extra': '_e-',
    }

    prefix = f"excluded_{args.exclude_role}_"
    outname = f"track" + ''.join([f"{nickname_dict[role_by_index[key]]}R{row}C{col}" for key, (row, col) in row_cols.items()])

    concatenated_track_df.to_pickle(save_track_dir / f'{outname}.pkl')
    concatenated_time_df.to_pickle(save_time_dir / f'{prefix}{outname}.pkl')


## --------------------------------------
def reprocess_code_to_time_df(args, input_file, board_roles: dict):

    track_df = pd.read_pickle(input_file)
    time_dfs = []

    for file_id in sorted(track_df['file'].unique()):
        df_file = track_df.loc[track_df['file'] == file_id]

        if df_file.empty:
            continue

        df_in_time = apply_TDC_cuts(args, df_file, board_roles)
        time_dfs.append(df_in_time)

    concatenated_time_df = pd.concat(time_dfs, ignore_index=True)
    return concatenated_time_df

## --------------------------------------
if __name__ == "__main__":

    import argparse, sys

    parser = argparse.ArgumentParser(
            prog='PlaceHolder',
            description='merge individual dataSelection results',
    )

    parser.add_argument(
        '-d',
        '--inputdir',
        metavar = 'DIRNAME',
        type = str,
        help = 'input directory name',
        required = True,
        dest = 'dirname',
    )

    parser.add_argument(
        '-o',
        '--outdir',
        metavar = 'OUTNAME',
        type = str,
        help = 'output directory name',
        required = True,
        dest = 'outdir',
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
        '--file_pattern',
        metavar = 'glob-pattern',
        help = "Put the file pattern for glob, if you want to process part of dataset. Example: 'run*_loop_[0-9].pickle run*_loop_1[0-9].pickle run*_loop_2[0-4].pickle'",
        default = 'run*_loop*.pickle',
        dest = 'file_pattern',
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
        '--trigTOALower',
        metavar = 'NUM',
        type = int,
        help = 'Lower TOA selection boundary for the trigger board',
        default = 100,
        dest = 'trigTOALower',
    )

    parser.add_argument(
        '--trigTOAUpper',
        metavar = 'NUM',
        type = int,
        help = 'Upper TOA selection boundary for the trigger board',
        default = 500,
        dest = 'trigTOAUpper',
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
        '--reprocess',
        action = 'store_true',
        help = 'If set, reprocess track',
        dest = 'reprocess',
    )

    parser.add_argument(
        '--use_new_toa',
        action = 'store_true',
        help = 'If set, use average of TOA and TOA+CAL as a new toa in time',
        dest = 'use_new_toa',
    )

    args = parser.parse_args()

    with open(args.config) as input_yaml:
        config = yaml.safe_load(input_yaml)

    if args.runName not in config:
        raise ValueError(f"Run config {args.runName} not found")

    roles = {}
    for board_id, board_info in config[args.runName].items():
        roles[board_info.get('role')] = board_id

    if not args.reprocess:

        outputdir = Path(args.outdir)
        outputdir.mkdir(exist_ok=True, parents=True)

        track_dir = outputdir / 'tracks'
        track_dir.mkdir(exist_ok=False)

        time_dir = outputdir / 'time'
        time_dir.mkdir(exist_ok=False)

        print(f'\nInput path is: {args.dirname}')
        print(f'Output path is: {args.outdir}')
        print(f'Will process the files based on the pattern: {args.file_pattern}\n')

        files = []
        patterns = args.file_pattern.split()
        for pattern in patterns:
            files += natsorted(Path(args.dirname).glob(pattern))

        if len(files) == 0:
            print(f'No input files for the given path: {args.dirname}')
            sys.exit()

        print('====== Categorize data by track ======')
        track_data = defaultdict(list)
        for ifile in tqdm(files):
            data_dict = pd.read_pickle(ifile)

            for track_key, df in data_dict.items():
                track_data[track_key].append(df)

        print('\n====== Apply TDC cut and save track and time dataframes ======')
        with tqdm(track_data) as pbar:
            with ProcessPoolExecutor() as process_executor:
                # Each input results in multiple threading jobs being created:
                futures = [
                    process_executor.submit(process_single_track, args, itrack, roles, track_dir, time_dir)
                        for _, itrack in track_data.items()
                ]
                for future in as_completed(futures):
                    pbar.update(1)

    else:
        track_dir = Path(args.outdir) / 'tracks'
        time_dir = Path(args.outdir) / 'reprocessed_time'
        time_dir.mkdir(exist_ok=True)

        print(f'Reprocess track pkl file at {track_dir}')
        print(f'New quantile TOT cut for DUT: {args.dutTOTlower}% - {args.dutTOTupper}%')

        prefix = f"excluded_{args.exclude_role}_"
        files = natsorted(track_dir.glob('track*pkl'))

        print('\n====== Apply TDC cut and save track and time dataframes ======')

        with tqdm(files) as pbar:
            with ProcessPoolExecutor() as process_executor:
                # Each input results in multiple threading jobs being created:
                futures = {
                    process_executor.submit(reprocess_code_to_time_df, args, ifile, roles): ifile
                    for ifile in files
                }
                for future in as_completed(futures):
                    iresult = future.result()
                    iresult = iresult.reset_index(drop=True) ## Make dataframe to use RangeIndex for memory efficient
                    iresult.to_pickle(time_dir / f"{prefix}{futures[future].name}")
                    pbar.update(1)