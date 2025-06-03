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
def process_single_track(args, track_dfs: dict, board_ids: list[int], board_roles: dict, save_track_dir: Path, save_time_dir: Path):
    concatenated_track_df = pd.concat(track_dfs, ignore_index=True)
    time_dfs = []

    for file_id in sorted(concatenated_track_df['file'].unique()):
        df_file = concatenated_track_df.loc[concatenated_track_df['file'] == file_id]

        if df_file.empty:
            continue

        ### Apply TDC cut
        tot_cuts = {
            idx: list(df_file['tot'][idx].quantile(
                [0.04, 0.91] if idx == board_roles['dut'] else [0.01, 0.96]
            ).values)
            for idx in board_ids
        }

        tdc_cuts = {
            idx: [
                0, 1100,
                args.trigTOALower if idx == board_roles['ref'] else 0,
                args.trigTOAUpper if idx == board_roles['ref'] else 1100,
                *tot_cuts[idx]
            ] for idx in board_ids
        }

        interest_df = tdc_event_selection_pivot(df_file, tdc_cuts_dict=tdc_cuts)

        if interest_df.empty:
            continue

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
        if reduced_df.empty:
            continue

        ### Convert to time
        df_in_time = pd.DataFrame()
        board_bins = {idx: 3.125 / reduced_df['cal'][idx].mean() for idx in board_ids}
        for idx in board_ids:
            bins = board_bins[idx]
            df_in_time[f'toa_b{idx}'] = (12.5 - reduced_df['toa'][idx] * bins) * 1e3
            df_in_time[f'tot_b{idx}'] = ((2 * reduced_df['tot'][idx] - np.floor(reduced_df['tot'][idx] / 32.)) * bins) * 1e3

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
    outname = f"track" + ''.join([f"_R{row}C{col}" for _, (row, col) in row_cols.items()])

    concatenated_track_df.to_pickle(save_track_dir / f'{outname}.pkl')
    concatenated_time_df.to_pickle(save_time_dir / f'{outname}.pkl')


## --------------------------------------
def reprocess_code_to_time_df(input_file, newDUTtotLower, newDUTtotUpper):

    track_df = pd.read_pickle(input_file)

    ### Apply TDC cut
    tot_cuts = {
        idx: list(track_df['tot'][idx].quantile(
            [newDUTtotLower, newDUTtotUpper] if idx == roles['dut'] else [0.01, 0.96]
        ).values)
        for idx in board_ids
    }

    tdc_cuts = {
        idx: [
            0, 1100,
            args.trigTOALower if idx == roles['ref'] else 0,
            args.trigTOAUpper if idx == roles['ref'] else 1100,
            *tot_cuts[idx]
        ] for idx in board_ids
    }

    interest_df = tdc_event_selection_pivot(track_df, tdc_cuts_dict=tdc_cuts)

    if not interest_df.empty:
        ### Apply TOA correlation cut
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
            df_in_time = pd.DataFrame()
            board_bins = {idx: 3.125 / reduced_df['cal'][idx].mean() for idx in board_ids}
            for idx in board_ids:
                bins = board_bins[idx]
                df_in_time[f'toa_b{idx}'] = (12.5 - reduced_df['toa'][idx] * bins) * 1e3
                df_in_time[f'tot_b{idx}'] = ((2 * reduced_df['tot'][idx] - np.floor(reduced_df['tot'][idx] / 32.)) * bins) * 1e3

    return df_in_time

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
        '--autoTOTcuts',
        action = 'store_true',
        help = 'If set, select 80 percent of data around TOT median value of each board',
        dest = 'autoTOTcuts',
    )

    parser.add_argument(
        '--reprocess',
        action = 'store_true',
        help = 'If set, reprocess track ',
        dest = 'reprocess',
    )

    parser.add_argument(
        '--dutTOTlower',
        metavar = 'NUM',
        type = int,
        help = 'Lower TOT boundary for the DUT board. Only relevant when --reprocess option is on.',
        default = 1,
        dest = 'dutTOTlower',
    )

    parser.add_argument(
        '--dutTOTupper',
        metavar = 'NUM',
        type = int,
        help = 'Upper TOT boundary for the DUT board. Only relevant when --reprocess option is on.',
        default = 96,
        dest = 'dutTOTupper',
    )

    args = parser.parse_args()

    with open(args.config) as input_yaml:
        config = yaml.safe_load(input_yaml)

    if args.runName not in config:
        raise ValueError(f"Run config {args.runName} not found")

    roles = {}
    for board_id, board_info in config[args.runName].items():
        roles[board_info.get('role')] = board_id

    board_ids = sorted([roles['ref'], roles['dut'], roles['extra']])

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
                    process_executor.submit(process_single_track, args, itrack, board_ids, roles, track_dir, time_dir)
                        for _, itrack in track_data.items()
                ]
                for future in as_completed(futures):
                    pbar.update(1)

    else:
        new_dutTOTlower = round(args.dutTOTlower * 0.01, 2)
        new_dutTOTupper = round(args.dutTOTupper * 0.01, 2)

        track_dir = Path(args.outdir) / 'tracks'
        time_dir = Path(args.outdir) / 'reprocessed_time'
        time_dir.mkdir(exist_ok=True)

        print(f'Reprocess track pkl file at {track_dir}')
        print(f'New quantile TOT cut for DUT: {new_dutTOTlower} - {new_dutTOTupper}')

        files = natsorted(track_dir.glob('track*pkl'))

        print('\n====== Apply TDC cut and save track and time dataframes ======')

        with tqdm(files) as pbar:
            with ProcessPoolExecutor() as process_executor:
                # Each input results in multiple threading jobs being created:
                futures = {
                    process_executor.submit(reprocess_code_to_time_df, ifile, new_dutTOTlower, new_dutTOTupper): ifile
                    for ifile in files
                }
                for future in as_completed(futures):
                    iresult = future.result()
                    iresult = iresult.reset_index(drop=True) ## Make dataframe to use RangeIndex for memory efficient
                    iresult.to_pickle(time_dir / futures[future].name)
                    pbar.update(1)