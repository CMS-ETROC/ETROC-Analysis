from natsort import natsorted
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict

import pandas as pd
import numpy as np
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
def convert_to_time_df(input_file):

    data_in_time = {}
    with open(input_file, 'rb') as f:
        # data_dict = pickle.load(f)  # Load dictionary from file (assuming files are pickled)
        data_dict = pd.read_pickle(f)
        for key in data_dict.keys():

            if data_dict[key].empty:
                data_in_time[key] = pd.DataFrame()
                continue

            ### Apply TDC cut
            tot_cuts = {
                idx: (
                    [data_dict[key]['tot'][idx].quantile(0.04 if idx == args.setDUTBoardID else 0.01),
                    data_dict[key]['tot'][idx].quantile(0.91 if idx == args.setDUTBoardID else 0.96)]
                    if args.autoTOTcuts else [0, 600]
                ) for idx in board_ids
            }

            tdc_cuts = {
                idx: [
                    0, 1100,
                    args.trigTOALower if idx == args.setTrigBoardID else 0,
                    args.trigTOAUpper if idx == args.setTrigBoardID else 1100,
                    *tot_cuts[idx]
                ] for idx in board_ids
            }

            df_in_time = pd.DataFrame()
            data_in_time[key] = df_in_time ## Put empty dataframe first, in case one of "if" conditions not work
            interest_df = tdc_event_selection_pivot(data_dict[key], tdc_cuts_dict=tdc_cuts)
            if not interest_df.empty:
                 ### Apply TOA correlation cut
                _, distance1 = return_TOA_correlation_param(interest_df, board_id1=board_ids[0], board_id2=board_ids[1])
                _, distance2 = return_TOA_correlation_param(interest_df, board_id1=board_ids[0], board_id2=board_ids[2])
                _, distance3 = return_TOA_correlation_param(interest_df, board_id1=board_ids[1], board_id2=board_ids[2])

                dist_cut = (distance1 < args.distance_factor*np.nanstd(distance1)) \
                         & (distance2 < args.distance_factor*np.nanstd(distance2)) \
                         & (distance3 < args.distance_factor*np.nanstd(distance3))

                reduced_interest_df = interest_df.loc[dist_cut]
                if not reduced_interest_df.empty:
                    for idx in board_ids:
                        bins = 3.125/reduced_interest_df['cal'][idx].mean()
                        df_in_time[f'toa_b{str(idx)}'] = (12.5 - reduced_interest_df['toa'][idx] * bins)*1e3
                        df_in_time[f'tot_b{str(idx)}'] = ((2*reduced_interest_df['tot'][idx] - np.floor(reduced_interest_df['tot'][idx]/32.)) * bins)*1e3
                    data_in_time[key] = df_in_time

    return data_dict, data_in_time

## --------------------------------------
def save_data(ikey, merged_data, merged_data_in_time, track_dir, time_dir):
    if not merged_data[ikey].empty:
        board_ids = merged_data[ikey].columns.get_level_values('board').unique().tolist()
        row_cols = {
            board_id: (merged_data[ikey]['row'][board_id].unique()[0], merged_data[ikey]['col'][board_id].unique()[0])
            for board_id in board_ids
        }
        outname = f"track_{ikey}" + ''.join([f"_R{row}C{col}" for board_id, (row, col) in row_cols.items()])

        merged_data[ikey].to_pickle(track_dir / f'{outname}.pkl')
        merged_data_in_time[ikey].to_pickle(time_dir / f'{outname}.pkl')
    else:
        print('Empty dataframe found, skip')


## --------------------------------------
def reprocess_code_to_time_df(input_file, newDUTtotLower, newDUTtotUpper):

    track_df = pd.read_pickle(input_file)
    df_in_time = pd.DataFrame()

    ### Apply TDC cut
    tot_cuts = {
        idx: (
            [track_df['tot'][idx].quantile(newDUTtotLower if idx == args.setDUTBoardID else 0.01),
            track_df['tot'][idx].quantile(newDUTtotUpper if idx == args.setDUTBoardID else 0.96)]
        ) for idx in board_ids
    }

    tdc_cuts = {
        idx: [
            0, 1100,
            args.trigTOALower if idx == args.setTrigBoardID else 0,
            args.trigTOAUpper if idx == args.setTrigBoardID else 1100,
            *tot_cuts[idx]
        ] for idx in board_ids
    }

    interest_df = tdc_event_selection_pivot(track_df, tdc_cuts_dict=tdc_cuts)
    if not interest_df.empty:
        ### Apply TOA correlation cut
        _, distance1 = return_TOA_correlation_param(interest_df, board_id1=board_ids[0], board_id2=board_ids[1])
        _, distance2 = return_TOA_correlation_param(interest_df, board_id1=board_ids[0], board_id2=board_ids[2])
        _, distance3 = return_TOA_correlation_param(interest_df, board_id1=board_ids[1], board_id2=board_ids[2])

        dist_cut = (distance1 < args.distance_factor*np.nanstd(distance1)) \
                    & (distance2 < args.distance_factor*np.nanstd(distance2)) \
                    & (distance3 < args.distance_factor*np.nanstd(distance3))

        reduced_interest_df = interest_df.loc[dist_cut]

        if not reduced_interest_df.empty:
            for idx in board_ids:
                bins = 3.125/reduced_interest_df['cal'][idx].mean()
                df_in_time[f'toa_b{str(idx)}'] = (12.5 - reduced_interest_df['toa'][idx] * bins)*1e3
                df_in_time[f'tot_b{str(idx)}'] = ((2*reduced_interest_df['tot'][idx] - np.floor(reduced_interest_df['tot'][idx]/32.)) * bins)*1e3

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
        '--setTrigBoardID',
        metavar = 'NUM',
        type = int,
        help = 'Set the offline trigger board ID',
        required = True,
        dest = 'setTrigBoardID',
    )

    parser.add_argument(
        '--setDUTBoardID',
        metavar = 'NUM',
        type = int,
        help = 'Set the DUT board ID',
        required = True,
        dest = 'setDUTBoardID',
    )

    parser.add_argument(
        '--setRefBoardID',
        metavar = 'NUM',
        type = int,
        help = 'Set the offline reference board ID',
        required = True,
        dest = 'setRefBoardID',
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
    board_ids = sorted([args.setTrigBoardID, args.setDUTBoardID, args.setRefBoardID])

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

        print('====== Code to Time Conversion is started ======')
        results = []
        with tqdm(files) as pbar:
            with ProcessPoolExecutor() as process_executor:
                # Each input results in multiple threading jobs being created:
                futures = [
                    process_executor.submit(convert_to_time_df, ifile)
                        for ifile in files
                ]
                for future in as_completed(futures):
                    pbar.update(1)
                    results.append(future.result())
        print('====== Code to Time Conversion is finished ======\n')

        ## Structure of results array: nested three-level
        # First [] points output from each file
        # Second [0] is data in code, [1] is data in time
        # Third [] access single dataframe of each track

        print('====== Merging is started ======')
        merged_data = defaultdict(list)
        merged_data_in_time = defaultdict(list)

        for result in results:
            for key in result[0].keys():
                merged_data[key].append(result[0][key])
                merged_data_in_time[key].append(result[1][key])

        # Now concatenate the lists of DataFrames
        merged_data = {key: pd.concat(df_list, ignore_index=True) for key, df_list in tqdm(merged_data.items())}
        merged_data_in_time = {key: pd.concat(df_list, ignore_index=True) for key, df_list in tqdm(merged_data_in_time.items())}
        del results
        print('====== Merging is finished ======\n')

        print('====== Saving data by track ======')
        with ThreadPoolExecutor(max_workers=6) as executor:
            list(tqdm(executor.map(lambda ikey: save_data(ikey, merged_data, merged_data_in_time, track_dir, time_dir), merged_data.keys()), total=len(merged_data)))

    else:
        new_dutTOTlower = args.dutTOTlower * 0.01
        new_dutTOTupper = args.dutTOTupper * 0.01

        track_dir = args.outdir / 'tracks'
        time_dir = args.outdir / 'reprocessed_time'
        time_dir.mkdir(exist_ok=True)

        print(f'Reprocess track pkl file at {track_dir}')
        print(f'New quantile TOT cut for DUT: {new_dutTOTlower} - {new_dutTOTupper}')

        files = natsorted(track_dir.glob('track*pkl'))
        print('\n====== Code to Time Conversion is started ======')
        results = []
        with tqdm(files) as pbar:
            with ProcessPoolExecutor() as process_executor:
                # Each input results in multiple threading jobs being created:
                futures = {
                    process_executor.submit(reprocess_code_to_time_df, ifile, new_dutTOTlower, new_dutTOTupper): ifile
                    for ifile in files
                }
                for future in as_completed(futures):
                    iresult = future.result()
                    iresult.to_pickle(time_dir / futures[future].name)
                    pbar.update(1)