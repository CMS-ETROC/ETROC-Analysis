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
        '-s',
        '--sampling',
        metavar = 'SAMPLING',
        type = int,
        help = 'Random sampling fraction',
        default = 5,
        dest = 'sampling',
    )

    parser.add_argument(
        '-m',
        '--minimum',
        metavar = 'NUM',
        type = int,
        help = 'Minimum number of tracks for selection',
        default = 1000,
        dest = 'ntracks',
    )

    parser.add_argument(
        '--max_diff_pixel',
        metavar = 'NUM',
        type = int,
        help = 'Maximum difference to allow track construction by pixel IDs',
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

    parser.add_argument(
        '--cal_table',
        action = 'store_true',
        help = 'If argument is on, code only does making CAL code table for a given dataset',
        dest = 'cal_table',
    )

    args = parser.parse_args()

    input_files = list(Path(f'{args.path}').glob('loop*feather'))
    columns_to_read = ['evt', 'board', 'row', 'col', 'toa', 'tot', 'cal']

    if len(input_files) == 0:
        import sys
        print('No input files.')
        sys.exit()

    if args.sampling > 20:
        import sys
        print('This is protection not to consume too much memory. Please use a number less then 20')
        sys.exit()

    print('Load dataframes')
    print(f'Sampling fraction is: {args.sampling*0.01}')
    portion = args.sampling*0.01

    dfs = []
    for ifile in tqdm(input_files):
        tmp_df = pd.read_feather(ifile, columns=columns_to_read)
        n = int(portion*tmp_df.shape[0])
        indices = np.random.choice(tmp_df['evt'].unique(), n, replace=False)
        tmp_df = tmp_df.loc[tmp_df['evt'].isin(indices)]
        dfs.append(tmp_df)
        del tmp_df

    final_input_df = pd.concat(dfs)

    ### Remove noisy pixels
    if args.mask_config_file is not None:
        with open(args.mask_config_file, 'r') as file:
            config_info = yaml.safe_load(file)

            for key, val in dict(config_info["board_ids"]).items():
                ## There is no noisy pixel, so list is empty
                if len(val['pixels']) == 0:
                    continue
                else:
                    for ipixel in val['pixels']:
                        final_input_df = final_input_df.loc[~((final_input_df['board'] == key) & (final_input_df['row'] == ipixel[0]) & (final_input_df['col'] == ipixel[1]))]

    final_input_df.reset_index(drop=True, inplace=True)
    del dfs

    print('CAL code filtering and Save Cal mode table')
    ### CAL code filtering
    cal_table = final_input_df.pivot_table(index=["row", "col"], columns=["board"], values=["cal"], aggfunc=lambda x: x.mode().iat[0])
    cal_table = cal_table.reset_index().set_index([('row', ''), ('col', '')]).stack().reset_index()
    cal_table.columns = ['row', 'col', 'board', 'cal_mode']
    cal_table.to_csv(f'{args.outfilename}_cal_table.csv', index=False)

    if not args.cal_table:
        print('Find track combinations')

        merged_df = pd.merge(final_input_df[['board', 'row', 'col', 'cal']], cal_table, on=['board', 'row', 'col'])
        merged_df['board'] = merged_df['board'].astype('uint8')
        merged_df['cal_mode'] = merged_df['cal_mode'].astype('int16')
        cal_condition = abs(merged_df['cal'] - merged_df['cal_mode']) <= 3
        del cal_table, merged_df
        cal_filtered_df = final_input_df.loc[cal_condition].reset_index(drop=True)
        del final_input_df, cal_condition

        ### Re-define Evt numbers
        # Identify where a new event starts
        is_new_event = cal_filtered_df['evt'] != cal_filtered_df['evt'].shift()

        # Assign a unique sequential number to each event
        cal_filtered_df['evt'] = is_new_event.cumsum() - 1

        ## A wide TDC cuts
        tdc_cuts = {}
        if not args.four_board:
            ids_to_loop = sorted([args.trigID, args.dutID, args.refID])
        else:
            ids_to_loop = [0, 1, 2, 3]

        for idx in ids_to_loop:
            # board ID: [CAL LB, CAL UB, TOA LB, TOA UB, TOT LB, TOT UB]
            if idx == args.trigID:
                tdc_cuts[idx] = [0, 1100,  100, 500, 50, 250]
            elif idx == args.refID:
                tdc_cuts[idx] = [0, 1100,  0, 1100, 50, 250]
            else:
                tdc_cuts[idx] = [0, 1100,  0, 1100, 0, 600]

        filtered_df = tdc_event_selection(cal_filtered_df, tdc_cuts_dict=tdc_cuts)
        del cal_filtered_df

        event_board_counts = filtered_df.groupby(['evt', 'board']).size().unstack(fill_value=0)
        event_selection_col = None

        if not args.four_board:
            trig_selection = (event_board_counts[args.trigID] == 1)
            ref_selection = (event_board_counts[args.refID] == 1)
            dut_selection = (event_board_counts[args.dutID] == 1)
            event_selection_col = trig_selection & ref_selection & dut_selection
        else:
            trig_selection = (event_board_counts[args.trigID] == 1)
            ref_selection = (event_board_counts[args.refID] == 1)
            ref_2nd_selection = (event_board_counts[args.ignoreID] == 1)
            dut_selection = (event_board_counts[args.dutID] == 1)
            event_selection_col = trig_selection & ref_selection & ref_2nd_selection & dut_selection

        selected_event_numbers = event_board_counts[event_selection_col].index
        selected_subset_df = filtered_df.loc[filtered_df['evt'].isin(selected_event_numbers)]
        selected_subset_df.reset_index(inplace=True, drop=True)
        del filtered_df

        if not args.four_board:
            ignore_board_ids = list(set([0, 1, 2, 3]) - set([args.trigID, args.dutID, args.refID]))
            list_of_ignore_boards = [args.ignoreID]
            columns_want_to_drop = [f'toa_{i}' for i in set([0, 1, 2, 3])-set(list_of_ignore_boards)]

            columns_want_to_group = []
            for i in set([0, 1, 2, 3])-set(list_of_ignore_boards):
                columns_want_to_group.append(f'row_{i}')
                columns_want_to_group.append(f'col_{i}')
        else:
            ignore_board_ids = None
            columns_want_to_drop = [f'toa_{i}' for i in [0,1,2,3]]

            columns_want_to_group = []
            for i in [0, 1, 2, 3]:
                columns_want_to_group.append(f'row_{i}')
                columns_want_to_group.append(f'col_{i}')

        pivot_data_df = making_pivot(selected_subset_df, 'evt', 'board', set({'board', 'evt', 'cal', 'tot'}), ignore_boards=ignore_board_ids)

        combinations_df = pivot_data_df.groupby(columns_want_to_group).count()
        combinations_df['count'] = combinations_df[f'toa_{args.trigID}']
        combinations_df.drop(columns_want_to_drop, axis=1, inplace=True)
        track_df = combinations_df.loc[combinations_df['count'] > args.ntracks]
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

        track_df = track_df.loc[track_condition]
        track_df.drop(columns=['count'], inplace=True)
        track_df = track_df.drop_duplicates(subset=columns_want_to_group, keep='first')
        track_df.to_csv(f'{args.outfilename}.csv', index=False)