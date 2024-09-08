import pandas as pd
import numpy as np
import random
import yaml
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

## --------------------------------------
def tdc_event_selection(
        input_df: pd.DataFrame,
        tdc_cuts_dict: dict,
        select_by_hit: bool = False,
    ):

    if select_by_hit:

        # Create boolean masks for each board's filtering criteria
        masks = {}
        for board, cuts in tdc_cuts_dict.items():
            mask = (
                (input_df['board'] == board) &
                input_df['cal'].between(cuts[0], cuts[1]) &
                input_df['toa'].between(cuts[2], cuts[3]) &
                input_df['tot'].between(cuts[4], cuts[5])
            )
            masks[board] = mask

        # Combine the masks using logical OR
        combined_mask = pd.concat(masks, axis=1).any(axis=1)

        # Apply the combined mask to the DataFrame
        tdc_filtered_df = input_df[combined_mask].reset_index(drop=True)

        return tdc_filtered_df

    else:
        from functools import reduce

        # Create boolean masks for each board's filtering criteria
        masks = {}
        for board, cuts in tdc_cuts_dict.items():
            mask = (
                (input_df['board'] == board) &
                input_df['cal'].between(cuts[0], cuts[1]) &
                input_df['toa'].between(cuts[2], cuts[3]) &
                input_df['tot'].between(cuts[4], cuts[5])
            )
            masks[board] = input_df[mask]['evt'].unique()

        common_elements = reduce(np.intersect1d, list(masks.values()))
        tdc_filtered_df = input_df.loc[input_df['evt'].isin(common_elements)].reset_index(drop=True)

        return tdc_filtered_df

## --------------------------------------
def making_pivot(
        input_df: pd.DataFrame,
        index: str,
        columns: str,
        drop_columns: tuple,
        ignore_boards: list[int] = None
    ):
        ana_df = input_df
        if ignore_boards is not None:
            for board in ignore_boards:
                ana_df = ana_df.loc[ana_df['board'] != board].copy()
        pivot_data_df = ana_df.pivot(
        index = index,
        columns = columns,
        values = list(set(ana_df.columns) - drop_columns),
        )
        pivot_data_df.columns = ["{}_{}".format(x, y) for x, y in pivot_data_df.columns]

        return pivot_data_df

## --------------------------------------
def making_clean_track_df(
        input_file: Path,
        columns_to_read: list[str],
        mask_config_file: Path,
        trig_id: int = 0,
        dut_id: int = 1,
        ref_id: int = 3,
        red_2nd_id: int = 2,
        four_board_track: bool = False,
    ):

    df = pd.read_feather(input_file, columns=columns_to_read)

    if mask_config_file is not None:
        with open(mask_config_file, 'r') as file:
            config_info = yaml.safe_load(file)

            for key, val in dict(config_info["board_ids"]).items():
                ## There is no noisy pixel, so list is empty
                if len(val['pixels']) == 0:
                    continue
                else:
                    for ipixel in val['pixels']:
                        df = df.loc[~((df['board'] == key) & (df['row'] == ipixel[0]) & (df['col'] == ipixel[1]))]

    # noisy_pixels = {
    #     2: [(0, 15)],
    # }

    # for board in noisy_pixels:
    #     for pixel in noisy_pixels[board]:
    #         df = df.loc[~((df['board'] == board) & (df['col'] == pixel[1]) & (df['row'] == pixel[0]))]

    if df.empty:
        print('file is empty. Move on to the next file')
        return pd.DataFrame()

    if (four_board_track) and (df['board'].unique().size != 4):
        print('This file does not have a full data including all four boards. Move on to the next file')
        return pd.DataFrame()

    if (~four_board_track) and (df['board'].unique().size < 3):
        print('This file does not have data including at least three boards. Move on to the next file')
        return pd.DataFrame()

    ### CAL code filtering
    cal_table = df.pivot_table(index=["row", "col"], columns=["board"], values=["cal"], aggfunc=lambda x: x.mode().iat[0])
    cal_table = cal_table.reset_index().set_index([('row', ''), ('col', '')]).stack().reset_index()
    cal_table.columns = ['row', 'col', 'board', 'cal_mode']

    merged_df = pd.merge(df[['board', 'row', 'col', 'cal']], cal_table, on=['board', 'row', 'col'])
    merged_df['board'] = merged_df['board'].astype('uint8')
    merged_df['cal_mode'] = merged_df['cal_mode'].astype('int16')
    cal_condition = abs(merged_df['cal'] - merged_df['cal_mode']) <= 3
    del cal_table, merged_df
    cal_filtered_df = df[cal_condition].reset_index(drop=True)
    del df, cal_condition

    ## A wide TDC cuts
    tdc_cuts = {}
    if not four_board_track:
        ids_to_loop = sorted([trig_id, dut_id, ref_id])
    else:
        ids_to_loop = [0, 1, 2, 3]

    for idx in ids_to_loop:
        # board ID: [CAL LB, CAL UB, TOA LB, TOA UB, TOT LB, TOT UB]
        if idx == 0:
            tdc_cuts[idx] = [0, 1100,  100, 500, 50, 250]
        elif idx == ref_id:
            tdc_cuts[idx] = [0, 1100,  0, 1100, 50, 250]
        else:
            tdc_cuts[idx] = [0, 1100,  0, 1100, 0, 600]

    filtered_df = tdc_event_selection(cal_filtered_df, tdc_cuts_dict=tdc_cuts)
    del cal_filtered_df

    if filtered_df.empty:
        return pd.DataFrame()

    event_board_counts = filtered_df.groupby(['evt', 'board']).size().unstack(fill_value=0)
    event_selection_col = None

    if not four_board_track:
        trig_selection = (event_board_counts[trig_id] == 1)
        ref_selection = (event_board_counts[ref_id] == 1)
        dut_selection = (event_board_counts[dut_id] == 1)
        event_selection_col = trig_selection & ref_selection & dut_selection
    else:
        trig_selection = (event_board_counts[trig_id] == 1)
        ref_selection = (event_board_counts[ref_id] == 1)
        ref_2nd_selection = (event_board_counts[red_2nd_id] == 1)
        dut_selection = (event_board_counts[dut_id] == 1)
        event_selection_col = trig_selection & ref_selection & ref_2nd_selection & dut_selection

    selected_event_numbers = event_board_counts[event_selection_col].index
    selected_subset_df = filtered_df.loc[filtered_df['evt'].isin(selected_event_numbers)]
    selected_subset_df.reset_index(inplace=True, drop=True)

    try:
        selected_subset_df['evt'], _ = pd.factorize(selected_subset_df['evt'])
    except:
        unique_evt_values = selected_subset_df['evt'].unique()
        evt_mapping = {old: new for new, old in enumerate(unique_evt_values)}
        # Apply the mapping to the evt column
        selected_subset_df['evt'] = selected_subset_df['evt'].map(evt_mapping)

    del filtered_df
    return selected_subset_df

## --------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
                prog='finding tracks',
                description='find track candidates!',
            )

    parser.add_argument(
        '-p',
        '--path',
        metavar = 'PATH',
        type = str,
        help = 'path to directory including feather files',
        required = True,
        dest = 'path',
    )

    parser.add_argument(
        '-o',
        '--outfilename',
        metavar = 'NAME',
        type = str,
        help = 'name for output csv file',
        required = True,
        dest = 'outfilename',
    )

    parser.add_argument(
        '-i',
        '--iteration',
        metavar = 'ITERATION',
        type = int,
        help = 'Number of iteration to find tracks',
        default = 10,
        dest = 'iteration',
    )

    parser.add_argument(
        '-s',
        '--sampling',
        metavar = 'SAMPLING',
        type = int,
        help = 'Random sampling fraction',
        default = 20,
        dest = 'sampling',
    )

    parser.add_argument(
        '-m',
        '--minimum',
        metavar = 'NUM',
        type = int,
        help = 'Minimum number of tracks for selection',
        default = 1000,
        dest = 'track',
    )

    parser.add_argument(
        '--max_diff_pixel',
        metavar = 'NUM',
        type = int,
        help = 'Maximum difference to allow track construction by pixel positions',
        default = 1,
        dest = 'max_diff_pixel',
    )

    parser.add_argument(
        '--trigID',
        metavar = 'ID',
        type = int,
        help = 'trigger board ID',
        default = 0,
        dest = 'trigID',
    )

    parser.add_argument(
        '--refID',
        metavar = 'ID',
        type = int,
        help = 'reference board ID',
        default = 3,
        dest = 'refID',
    )

    parser.add_argument(
        '--dutID',
        metavar = 'ID',
        type = int,
        help = 'DUT board ID',
        default = 1,
        dest = 'dutID',
    )

    parser.add_argument(
        '--ignoreID',
        metavar = 'ID',
        type = int,
        help = 'board ID be ignored',
        default = 2,
        dest = 'ignoreID',
    )

    parser.add_argument(
        '--four_board',
        action = 'store_true',
        help = 'data will be selected based on 4-board combination',
        dest = 'four_board',
    )

    parser.add_argument(
        '--mask_config',
        type = Path,
        help = 'The YAML config file for masking noisy pixels',
        default = None,
        dest = 'mask_config_file',
    )

    args = parser.parse_args()

    input_files = list(Path(f'{args.path}').glob('loop*feather'))
    columns_to_read = ['evt', 'board', 'row', 'col', 'toa', 'tot', 'cal']

    if len(input_files) == 0:
        import sys
        print('No input files.')
        sys.exit()

    if not args.four_board:

        list_of_ignore_boards = [args.ignoreID]
        columns_want_to_drop = [f'toa_{i}' for i in set([0, 1, 2, 3])-set(list_of_ignore_boards)]

        columns_want_to_group = []
        for i in set([0, 1, 2, 3])-set(list_of_ignore_boards):
            columns_want_to_group.append(f'row_{i}')
            columns_want_to_group.append(f'col_{i}')

        print('*************** 3-board track finding ********************')
        print(f'Output csv file name is: {args.outfilename}')
        print(f'Number track finding iteration: {args.iteration}')
        print(f'Sampling fraction is: {args.sampling*0.01}')
        print(f'Minimum number of track for selection is: {args.track}')
        print(f'Trigger board ID is: {args.trigID}')
        print(f'Reference board ID is: {args.refID}')
        print(f'Device Under Test board ID is: {args.dutID}')
        print(f'Board ID {args.ignoreID} will be ignored')
        print('*************** 3-board track finding ********************')
    else:
        columns_want_to_drop = [f'toa_{i}' for i in [0,1,2,3]]

        columns_want_to_group = []
        for i in [0, 1, 2, 3]:
            columns_want_to_group.append(f'row_{i}')
            columns_want_to_group.append(f'col_{i}')

        print('*************** 4-board track finding ********************')
        print(f'Output csv file name is: {args.outfilename}')
        print(f'Number track finding iteration: {args.iteration}')
        print(f'Sampling fraction is: {args.sampling*0.01}')
        print(f'Minimum number of track for selection is: {args.track}')
        print(f'Trigger board ID is: {args.trigID}')
        print(f'Reference board ID is: {args.refID}')
        print(f'2nd reference board ID is: {args.ignoreID}')
        print(f'Device Under Test board ID is: {args.dutID}')
        print('*************** 4-board track finding ********************')

    sampling_fraction = args.sampling * 0.01
    final_list = []

    for isampling in tqdm(range(args.iteration)):
        files = random.sample(input_files, k=int(sampling_fraction*len(input_files)))

        results = []
        with tqdm(files) as pbar:
            with ProcessPoolExecutor() as process_executor:
                # Each input results in multiple threading jobs being created:
                futures = [
                    process_executor.submit(making_clean_track_df, ifile, columns_to_read, args.mask_config_file,
                                            args.trigID, args.dutID, args.refID, args.ignoreID, args.four_board)
                        for ifile in files
                ]
                for future in as_completed(futures):
                    pbar.update(1)
                    results.append(future.result())

        dfs = []
        nevt = 0
        for iframe in results:
            iframe['evt'] += nevt
            nevt += iframe['evt'].nunique()
            dfs.append(iframe)

        df = pd.concat(dfs)
        df.reset_index(inplace=True, drop=True)
        del results, dfs

        if not args.four_board:
            ignore_board_ids = list(set([0, 1, 2, 3]) - set([args.trigID, args.dutID, args.refID]))
        else:
            ignore_board_ids = None

        pivot_data_df = making_pivot(df, 'evt', 'board', set({'board', 'evt', 'cal', 'tot'}), ignore_boards=ignore_board_ids)
        del df

        min_hit_counter = args.track*(len(files)/len(input_files))
        combinations_df = pivot_data_df.groupby(columns_want_to_group).count()
        combinations_df['count'] = combinations_df[f'toa_{args.trigID}']
        combinations_df.drop(columns_want_to_drop, axis=1, inplace=True)
        track_df = combinations_df.loc[combinations_df['count'] > min_hit_counter]
        track_df.reset_index(inplace=True)
        del pivot_data_df, combinations_df

        if not args.four_board:
            row_delta_TR = np.abs(track_df[f'row_{args.trigID}'] - track_df[f'row_{args.refID}']) <= args.max_diff_pixel
            row_delta_TD = np.abs(track_df[f'row_{args.trigID}'] - track_df[f'row_{args.dutID}']) <= args.max_diff_pixel
            col_delta_TR = np.abs(track_df[f'col_{args.trigID}'] - track_df[f'col_{args.refID}']) <= args.max_diff_pixel
            col_delta_TD = np.abs(track_df[f'col_{args.trigID}'] - track_df[f'col_{args.dutID}']) <= args.max_diff_pixel
            track_condition = (row_delta_TR) & (col_delta_TR) & (row_delta_TD) & (col_delta_TD)

        else:
            row_delta_TR  = np.abs(track_df[f'row_{args.trigID}'] - track_df[f'row_{args.refID}']) <= args.max_diff_pixel
            row_delta_TR2 = np.abs(track_df[f'row_{args.trigID}'] - track_df[f'row_{args.ignoreID}']) <= args.max_diff_pixel
            row_delta_TD  = np.abs(track_df[f'row_{args.trigID}'] - track_df[f'row_{args.dutID}']) <= args.max_diff_pixel
            col_delta_TR  = np.abs(track_df[f'col_{args.trigID}'] - track_df[f'col_{args.refID}']) <= args.max_diff_pixel
            col_delta_TR2 = np.abs(track_df[f'col_{args.trigID}'] - track_df[f'col_{args.ignoreID}']) <= args.max_diff_pixel
            col_delta_TD  = np.abs(track_df[f'col_{args.trigID}'] - track_df[f'col_{args.dutID}']) <= args.max_diff_pixel
            track_condition = (row_delta_TR) & (col_delta_TR) & (row_delta_TD) & (col_delta_TD) & (row_delta_TR2) & (col_delta_TR2)

        track_df = track_df[track_condition]
        final_list.append(track_df)

    final_df = pd.concat(final_list)
    final_df.drop(columns=['count'], inplace=True)
    final_df = final_df.drop_duplicates(subset=columns_want_to_group, keep='first')
    final_df.to_csv(f'{args.outfilename}.csv', index=False)
