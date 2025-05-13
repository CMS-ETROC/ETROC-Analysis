import pandas as pd
import numpy as np

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
def simple_hitmap(
        input_df: pd.DataFrame,
        ref_board_id: int,
        interest_board_id: int,
        pixel: list[int],
    ):
    input_df = input_df.loc[(input_df['board'] == ref_board_id) | (input_df['board'] == interest_board_id)]
    event_board_counts = input_df.groupby(['evt', 'board']).size().unstack(fill_value=0)
    event_selection_col = (event_board_counts[ref_board_id] == 1) & (event_board_counts[interest_board_id] == 1)

    selected_subset_df = input_df.loc[input_df['evt'].isin(event_board_counts[event_selection_col].index)]
    selected_subset_df.reset_index(inplace=True, drop=True)

    evts = selected_subset_df[(selected_subset_df['board'] == ref_board_id) & (selected_subset_df['row'] == pixel[0]) & (selected_subset_df['col'] == pixel[1])]['evt'].values
    interest_df = selected_subset_df[selected_subset_df['evt'].isin(evts)].reset_index(drop=True)
    pivot_data_df = making_pivot(interest_df, 'evt', 'board', set({'board', 'evt', 'cal', 'tot', 'ea', 'bcid', 'l1a_counter'}), ignore_boards=None)

    combinations_df = pivot_data_df.groupby([f'row_{interest_board_id}', f'col_{interest_board_id}']).count()
    combinations_df['count'] = combinations_df[f'toa_{interest_board_id}']
    combinations_df.drop([f'toa_{ref_board_id}', f'toa_{interest_board_id}', f'row_{ref_board_id}', f'col_{ref_board_id}'], axis=1, inplace=True)
    combinations_df = combinations_df.astype('int64')
    combinations_df.reset_index(inplace=True)

    return combinations_df

## --------------------------------------
def track_based_hitmap(
        input_df: pd.DataFrame,
        ref_board_id1: int,
        ref_board_id2: int,
        ref_board_id3: int,
        interest_board_id: int,
        pixel: list[int],
    ):

    drop_colums = ['toa_0', 'toa_1', 'toa_2', 'toa_3']
    for inum in [ref_board_id1, ref_board_id2, ref_board_id3]:
         drop_colums.append(f'row_{inum}')
         drop_colums.append(f'col_{inum}')

    event_board_counts = input_df.groupby(['evt', 'board']).size().unstack(fill_value=0)
    event_selection_col = (event_board_counts[0] == 1) & (event_board_counts[1] == 1) & (event_board_counts[2] == 1) & (event_board_counts[3] == 1)

    selected_subset_df = input_df.loc[input_df['evt'].isin(event_board_counts[event_selection_col].index)]
    selected_subset_df.reset_index(inplace=True, drop=True)

    board1_condition = (selected_subset_df['board'] == ref_board_id1) & (selected_subset_df['row'] == pixel[0]) & (selected_subset_df['col'] == pixel[1])
    board2_condition = (selected_subset_df['board'] == ref_board_id2) & (selected_subset_df['row'] == pixel[0]) & (selected_subset_df['col'] == pixel[1])
    board3_condition = (selected_subset_df['board'] == ref_board_id3) & (selected_subset_df['row'] == pixel[0]) & (selected_subset_df['col'] == pixel[1])

    evts1 = selected_subset_df[board1_condition]['evt'].values
    evts2 = selected_subset_df[board2_condition]['evt'].values
    evts3 = selected_subset_df[board3_condition]['evt'].values

    evts = np.intersect1d(np.intersect1d(evts1, evts2), evts3)

    interest_df = selected_subset_df[selected_subset_df['evt'].isin(evts)].reset_index(drop=True)
    pivot_data_df = making_pivot(interest_df, 'evt', 'board', set({'board', 'evt', 'cal', 'tot', 'ea', 'bcid', 'l1a_counter'}), ignore_boards=None)

    combinations_df = pivot_data_df.groupby([f'row_{interest_board_id}', f'col_{interest_board_id}']).count()
    combinations_df['count'] = combinations_df[f'toa_{interest_board_id}']
    combinations_df.drop(drop_colums, axis=1, inplace=True)
    combinations_df = combinations_df.astype('int64')
    combinations_df.reset_index(inplace=True)

    return combinations_df

if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser(
                prog='bootstrap',
                description='find time resolution!',
            )

    parser.add_argument(
        '-f',
        '--file',
        metavar = 'PATH',
        type = str,
        help = 'feather file',
        required = True,
        dest = 'file',
    )

    parser.add_argument(
        '-r',
        '--runinfo',
        metavar = 'NAME',
        type = str,
        help = 'run information for the output name',
        required = True,
        dest = 'runinfo',
    )

    parser.add_argument(
        '--row',
        metavar = 'NUM',
        type = int,
        help = 'Row of pixel',
        default = 8,
        dest = 'row',
    )

    parser.add_argument(
        '--col',
        metavar = 'NUM',
        type = int,
        help = 'Column of pixel',
        default = 11,
        dest = 'col',
    )

    parser.add_argument(
        '--ref_board_id1',
        metavar = 'NUM',
        type = int,
        help = 'Reference board ID',
        default = 0,
        dest = 'ref_board_id1',
    )

    parser.add_argument(
        '--ref_board_id2',
        metavar = 'NUM',
        type = int,
        help = 'Reference board ID',
        default = 2,
        dest = 'ref_board_id2',
    )

    parser.add_argument(
        '--ref_board_id3',
        metavar = 'NUM',
        type = int,
        help = 'Reference board ID',
        default = 3,
        dest = 'ref_board_id3',
    )

    parser.add_argument(
        '--interest_board_id',
        metavar = 'NUM',
        type = int,
        help = 'Interesting board ID',
        default = 1,
        dest = 'interest_board_id',
    )

    args = parser.parse_args()

    columns_to_read = ['evt', 'board', 'row', 'col', 'toa', 'tot', 'cal']
    run_df = pd.read_feather(args.file, columns=columns_to_read)

    if run_df.empty:
        print('Empty input file!')
        exit(0)

    results = {}

    results['simple_map'] = simple_hitmap(input_df=run_df, ref_board_id=args.ref_board_id1, interest_board_id=args.interest_board_id, pixel=[args.row, args.col])
<<<<<<< HEAD
    results['track_based_map'] = track_based_hitmap(input_df=run_df, ref_board_id1=args.ref_board_id1, ref_board_id2=args.ref_board_id2, ref_board_id3=args.ref_board_id3, interest_board_id=args.interest_board_id, pixel=[args.row, args.col])
=======
    results['track_based_map'] = track_based_hitmap(input_df=run_df, ref_board_id1=args.ref_board_id1, ref_board_id2=args.ref_board_id2, ref_board_id3=args.ref_board_id3,
                                                    interest_board_id=args.interest_board_id, pixel=[args.row, args.col])
>>>>>>> 0fadb2d329e8212b49983bb7c6aa93320fcd6ab4

    fname = args.file.split('.')[0]
    with open(f'{args.runinfo}_{fname}_hitmap.pickle', 'wb') as output:
        pickle.dump(results, output, protocol=pickle.HIGHEST_PROTOCOL)
