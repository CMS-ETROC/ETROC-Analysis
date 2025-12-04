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
def check_empty_df(input_df: pd.DataFrame, extraStr=""):
    import sys
    if input_df.empty:
        print(f"Warning: DataFrame is empty after {extraStr}")
        sys.exit(1)

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
def delta_within_limit(input_df: pd.DataFrame, id_roles: dict, axis: str, role1: str, role2: str) -> pd.Series:
    return (
        np.abs(input_df[f'{axis}_{id_roles[role1]}'] - input_df[f'{axis}_{id_roles[role2]}'])
        <= args.max_diff_pixel * 1.3
    )

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


def get_tranformation(
        board_id: int,
        config,
    ):
    # TODO: Add here a check if these values exist and if not return a default value of 0
    rot_x = config[board_id]['transformation']['rotation']['x']*np.pi/180
    rot_y = config[board_id]['transformation']['rotation']['y']*np.pi/180
    rot_z = config[board_id]['transformation']['rotation']['z']*np.pi/180
    tra_x = config[board_id]['transformation']['translation']['x']
    tra_y = config[board_id]['transformation']['translation']['y']
    tra_z = config[board_id]['transformation']['translation']['z']

    return (rot_x, rot_y, rot_z), (tra_x, tra_y, tra_z)

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
        '--out_calname',
        metavar = 'NAME',
        type = str,
        help = 'Name for cal table output csv file',
        required = True,
        dest = 'out_calname',
    )

    parser.add_argument(
        '--out_trackname',
        metavar = 'NAME',
        type = str,
        help = 'Name for track combination table output csv file',
        required = True,
        dest = 'out_trackname',
    )

    parser.add_argument(
        '-s',
        '--sampling',
        metavar = 'SAMPLING',
        type = float,
        help = 'Random sampling fraction',
        default = 3,
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
        '-c',
        '--config',
        metavar = 'NAME',
        type = str,
        help = 'YAML file including run information.',
        required = True,
        dest = 'config',
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
        '--mask_config',
        type = Path,
        help = 'The YAML config file for masking noisy pixels',
        default = None,
        dest = 'mask_config_file',
    )

    parser.add_argument(
        '--exclude_role',
        metavar = 'NAME',
        type = str,
        help = "Choose the board to exclude for track combination. Possible option: 'trig', 'dut', 'ref', 'extra'",
        dest = 'exclude_role',
    )

    parser.add_argument(
        '--cal_table_only',
        action = 'store_true',
        help = 'If argument is on, code only does making CAL code table for a given dataset',
        dest = 'cal_table_only',
    )

    args = parser.parse_args()

    input_files = list(Path(f'{args.path}').glob('loop*feather'))

    if len(input_files) > 100:
        input_files = input_files[:100]

    columns_to_read = ['evt', 'board', 'row', 'col', 'toa', 'tot', 'cal']

    with open(args.config) as input_yaml:
        config = yaml.safe_load(input_yaml)

    if args.runName not in config:
        raise ValueError(f"Run config {args.runName} not found")

    roles = {}
    for board_id, board_info in config[args.runName].items():
        roles[board_info.get('role')] = board_id

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

    ### Estimate memory usage as another safety check
    print('Memory Safety Check, using randomly selected 10 files')
    import random
    if len(input_files) < 10:
      random_files = input_files
    else:
      random_files = random.sample(input_files, 10)
    sum_use = 0
    for jfile in tqdm(random_files):
        check_df = pd.read_feather(jfile, columns=columns_to_read)
        n = int(portion*check_df['evt'].nunique())
        indices = np.random.choice(check_df['evt'].unique(), n, replace=False)
        check_df = check_df.loc[check_df['evt'].isin(indices)]
        sum_use += check_df.memory_usage(deep=True).sum() / (1024**2) ## memory usage in MB

    avg_use = round(sum_use/len(random_files))
    total_use = round(avg_use) * len(input_files)

    print(f'Average dataframe memory usage: {avg_use} MB')
    print(f'Estimated total memory usage: {total_use} MB')
    if (avg_use > 26) or (total_use > 2600):
        print('\nMemory Safety Check Fail, Memory usages are over limit.')
        print('Recommend: Single dataframe < 20 MB or Total dataframe < 2560 MB')
        import sys
        sys.exit()

    del check_df, n, indices, sum_use, total_use, random_files
    print('Memory Safety Check Pass\n')

    dfs = []
    for ifile in tqdm(input_files):
        tmp_df = pd.read_feather(ifile, columns=columns_to_read)
        n = int(portion*tmp_df['evt'].nunique())
        indices = np.random.choice(tmp_df['evt'].unique(), n, replace=False)
        tmp_df = tmp_df.loc[tmp_df['evt'].isin(indices)]
        dfs.append(tmp_df)
        del tmp_df

    final_input_df = pd.concat(dfs)
    total_use = round(final_input_df.memory_usage(deep=True).sum() / (1024**2))
    print(f'Real total memory usage: {total_use} MB')

    ### Re-define Evt numbers
    # Identify where a new event starts
    is_new_event = final_input_df['evt'] != final_input_df['evt'].shift()

    # Assign a unique sequential number to each event
    final_input_df['evt'] = is_new_event.cumsum() - 1
    del is_new_event

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

    if not args.exclude_role == None:
        print(f'Board {args.exclude_role}, ID: {roles[args.exclude_role]} will be dropped')
        drop_id_condition = ~(final_input_df['board'] == roles[args.exclude_role])
        final_input_df = final_input_df[drop_id_condition].reset_index(drop=True)

    three_board_flag = False
    if final_input_df['board'].nunique() == 3:
        three_board_flag = True

    print('CAL code filtering and Save Cal mode table')
    ### CAL code filtering
    cal_table = final_input_df.pivot_table(index=["row", "col"], columns=["board"], values=["cal"], aggfunc=lambda x: x.mode().iat[0])
    cal_table = cal_table.reset_index().set_index([('row', ''), ('col', '')]).stack().reset_index()
    cal_table.columns = ['row', 'col', 'board', 'cal_mode']
    cal_table.to_csv(f'{args.out_calname}_cal_table.csv', index=False)

    if not args.cal_table_only:
        print('Find track combinations')

        merged_df = pd.merge(final_input_df[['board', 'row', 'col', 'cal']], cal_table, on=['board', 'row', 'col'])
        merged_df = merged_df.astype({'board': 'uint8', 'cal_mode': 'int16'})
        cal_condition = abs(merged_df['cal'] - merged_df['cal_mode']) <= 3
        del cal_table, merged_df
        cal_filtered_df = final_input_df.loc[cal_condition].reset_index(drop=True)
        del final_input_df, cal_condition

        ### Re-define Evt numbers
        # Identify where a new event starts
        is_new_event = cal_filtered_df['evt'] != cal_filtered_df['evt'].shift()

        # Assign a unique sequential number to each event
        cal_filtered_df['evt'] = is_new_event.cumsum() - 1
        check_empty_df(cal_filtered_df, "CAL filtering.")

        ## A wide TDC cuts
        ids_to_loop = sorted([roles[k] for k in ('trig', 'dut', 'ref')]) if three_board_flag else [0, 1, 2, 3]
        tdc_cuts = {
            idx: ([0, 1100, 100, 500, 50, 250] if idx == roles['trig']
                else [0, 1100, 0, 1100, 50, 250] if idx == roles['ref']
                else [0, 1100, 0, 1100, 0, 600])
            for idx in ids_to_loop
        }

        filtered_df = tdc_event_selection(cal_filtered_df, tdc_cuts_dict=tdc_cuts)
        del cal_filtered_df
        check_empty_df(filtered_df, "TDC filtering.")

        event_board_counts = filtered_df.groupby(['evt', 'board']).size().unstack(fill_value=0)
        if three_board_flag:
            boards_needed = ['trig', 'ref', 'dut']
        else:
            boards_needed = ['trig', 'ref', 'extra', 'dut']

        event_mask = np.logical_and.reduce([event_board_counts[roles[r]] == 1 for r in boards_needed])
        selected_event_numbers = event_board_counts[event_mask].index

        selected_subset_df = filtered_df.loc[filtered_df['evt'].isin(selected_event_numbers)].reset_index(drop=True)
        selected_subset_df[['row', 'col']] = selected_subset_df[['row', 'col']].astype('int8')
        del filtered_df
        check_empty_df(selected_subset_df, "Single hit event filtering.")

        if three_board_flag:
            ignore_board_ids = list(set([0, 1, 2, 3]) - set([roles['trig'], roles['dut'], roles['ref']]))
            columns_want_to_drop = [f'toa_{i}' for i in set([0, 1, 2, 3])-set(ignore_board_ids)]

            columns_want_to_group = []
            for i in set([0, 1, 2, 3])-set(ignore_board_ids):
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
        combinations_df['count'] = combinations_df[f'toa_{roles["trig"]}']
        combinations_df.drop(columns_want_to_drop, axis=1, inplace=True)

        track_df = combinations_df.loc[combinations_df['count'] > args.ntracks]
        track_df.reset_index(inplace=True)

        del pivot_data_df, combinations_df

        for id in range(4):
            track_df[f'xprime_{id}'] = (track_df[f'col_{id}'] - 7.5) * 1.3
            track_df[f'yprime_{id}'] = (track_df[f'row_{id}'] - 7.5) * 1.3

            rot, tra = get_tranformation(id, config[args.runName])

            track_df[f'x_{id}'] = track_df[f'xprime_{id}'] * (np.cos(rot[2]) * np.cos(rot[1])) + track_df[f'yprime_{id}'] * (np.cos(rot[2])*np.sin(rot[1])*np.sin(rot[0]) - np.sin(rot[2])*np.cos(rot[0]))
            track_df[f'y_{id}'] = track_df[f'xprime_{id}'] * (np.sin(rot[2]) * np.cos(rot[1])) + track_df[f'yprime_{id}'] * (np.sin(rot[2])*np.sin(rot[1])*np.sin(rot[0]) + np.cos(rot[2])*np.cos(rot[0]))
            track_df[f'z_{id}'] = track_df[f'xprime_{id}'] * (- np.sin(rot[1]))                + track_df[f'yprime_{id}'] * (np.cos(rot[1])*np.sin(rot[0]))

            track_df[f'x_{id}'] += tra[0]
            track_df[f'y_{id}'] += tra[1]
            track_df[f'z_{id}'] += tra[2]

        if three_board_flag:
            role_pairs = [('y', 'trig', 'ref'), ('y', 'trig', 'dut'),
                          ('x', 'trig', 'ref'), ('x', 'trig', 'dut')]
        else:
            role_pairs = [('y', 'trig', 'ref'), ('y', 'trig', 'dut'), ('y', 'trig', 'extra'),
                          ('x', 'trig', 'ref'), ('x', 'trig', 'dut'), ('x', 'trig', 'extra')]

        track_condition = np.logical_and.reduce([delta_within_limit(track_df, roles, axis, r1, r2) for axis, r1, r2 in role_pairs])

        track_df = track_df.loc[track_condition]
        track_df = track_df.drop_duplicates(subset=columns_want_to_group, keep='first')
        track_df.to_csv(f'{args.out_trackname}_tracks.csv', index=False)

        print('Done: find track combinations\n')