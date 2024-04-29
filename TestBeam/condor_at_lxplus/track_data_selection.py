import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm

## --------------------------------------
def pixel_filter(
        input_df: pd.DataFrame,
        pixel_dict: dict,
        filter_by_area: bool = False,
        pixel_buffer: int = 2,
    ):

    masks = {}
    if filter_by_area:
        for board, pix in pixel_dict.items():
            mask = (
                (input_df['board'] == board)
                & (input_df['row'] >= pix[0]-pixel_buffer) & (input_df['row'] <= pix[0]+pixel_buffer)
                & (input_df['col'] >= pix[1]-pixel_buffer) & (input_df['col'] <= pix[1]+pixel_buffer)
            )
            masks[board] = mask
    else:
        for board, pix in pixel_dict.items():
            mask = (
                (input_df['board'] == board) & (input_df['row'] == pix[0]) & (input_df['col'] == pix[1])
            )
            masks[board] = mask

    # Combine the masks using logical OR
    combined_mask = pd.concat(masks, axis=1).any(axis=1)

    # Apply the combined mask to the DataFrame
    filtered = input_df[combined_mask].reset_index(drop=True)
    return filtered

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
def data_4board_selection_by_track(
        input_df: pd.DataFrame,
        pix_dict: dict,
        trig_id: int,
        dut_id: int,
        ref_id: int,
        ref_2nd_id: int,
        board_to_analyze: list[int],
        tot_cuts: list[int],
    ):

    track_tmp_df = pixel_filter(input_df, pix_dict, filter_by_area=False)

    ## Selecting good hits with TDC cuts
    tdc_cuts = {}
    for idx in board_to_analyze:
        # board ID: [CAL LB, CAL UB, TOA LB, TOA UB, TOT LB, TOT UB]
        if idx == trig_id:
            tdc_cuts[idx] = [track_tmp_df.loc[track_tmp_df['board'] == idx]['cal'].mode()[0]-3, track_tmp_df.loc[track_tmp_df['board'] == idx]['cal'].mode()[0]+3,
                    0, 1100, tot_cuts[0], tot_cuts[1]]
        elif idx == ref_id:
            tdc_cuts[idx] = [track_tmp_df.loc[track_tmp_df['board'] == idx]['cal'].mode()[0]-3, track_tmp_df.loc[track_tmp_df['board'] == idx]['cal'].mode()[0]+3,
                    0, 1100, 0, 600]
        else:
            tdc_cuts[idx] = [track_tmp_df.loc[track_tmp_df['board'] == idx]['cal'].mode()[0]-3, track_tmp_df.loc[track_tmp_df['board'] == idx]['cal'].mode()[0]+3,
                    0, 1100, 0, 600]

    track_tmp_df = tdc_event_selection(track_tmp_df, tdc_cuts_dict=tdc_cuts)

    event_board_counts = track_tmp_df.groupby(['evt', 'board']).size().unstack(fill_value=0)
    event_selection_col = (event_board_counts[0] == 1) & (event_board_counts[1] == 1) & (event_board_counts[2] == 1) & (event_board_counts[3] == 1)

    isolated_df = track_tmp_df.loc[track_tmp_df['evt'].isin(event_board_counts[event_selection_col].index)]

    pivot_table = isolated_df.pivot(index=["evt"], columns=["board"], values=["row", "col", "toa", "tot", "cal"])
    pivot_table = pivot_table.reset_index(drop=True)

    ## Pivot Table to make tracks
    return pivot_table

## --------------------------------------
if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser(
                prog='data selection',
                description='save data by track',
            )

    parser.add_argument(
        '-f',
        '--inputfile',
        metavar = 'NAME',
        type = str,
        help = 'input feather file',
        required = True,
        dest = 'inputfile',
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
        '-t',
        '--track',
        metavar = 'NAME',
        type = str,
        help = 'csv file including track candidates',
        required = True,
        dest = 'track',
    )

    parser.add_argument(
        '--trigID',
        metavar = 'ID',
        type = int,
        help = 'trigger board ID',
        required = True,
        dest = 'trigID',
    )

    parser.add_argument(
        '--refID',
        metavar = 'ID',
        type = int,
        help = 'reference board ID',
        required = True,
        dest = 'refID',
    )

    parser.add_argument(
        '--dutID',
        metavar = 'ID',
        type = int,
        help = 'DUT board ID',
        required = True,
        dest = 'dutID',
    )

    parser.add_argument(
        '--ignoreID',
        metavar = 'ID',
        type = int,
        help = 'board ID be ignored',
        required = True,
        dest = 'ignoreID',
    )

    parser.add_argument(
        '--trigTOTLower',
        metavar = 'NUM',
        type = int,
        help = 'Lower TOT selection boundary for the trigger board',
        default = 100,
        dest = 'trigTOTLower',
    )

    parser.add_argument(
        '--trigTOTUpper',
        metavar = 'NUM',
        type = int,
        help = 'Upper TOT selection boundary for the trigger board',
        default = 200,
        dest = 'trigTOTUpper',
    )

    args = parser.parse_args()

    board_ids = [0,1,2,3]
    ignore_boards = [args.ignoreID]
    tot_cuts = [args.trigTOTLower, args.trigTOTUpper]

    trig_id = args.trigID
    ref_id = args.refID
    dut_id = args.dutID

    columns_to_read = ['evt', 'board', 'row', 'col', 'toa', 'tot', 'cal']

    run_df = pd.read_feather(args.inputfile, columns=columns_to_read)

    if run_df.empty:
        print('Empty input file!')
        exit(0)

    track_df = pd.read_csv(args.track)
    track_pivots = defaultdict(pd.DataFrame)

    if track_df.shape[1] == 8:
        board_to_analyze = board_ids

        for itrack in tqdm(range(track_df.shape[0])):
            pix_dict = {}
            for idx in board_to_analyze:
                pix_dict[idx] = [track_df.iloc[itrack][f'row_{idx}'], track_df.iloc[itrack][f'col_{idx}']]

            table = data_4board_selection_by_track(input_df=run_df, pix_dict=pix_dict, trig_id=trig_id, dut_id=dut_id, ref_id=ref_id, ref_2nd_id=args.ignoreID,
                                                   board_to_analyze=board_to_analyze, tot_cuts=tot_cuts)
            track_pivots[itrack] = table
    else:
        print('Under developing')
        exit(0)

    fname = args.inputfile.split('.')[0]
    ### Save python dictionary in pickle format
    with open(f'{args.runinfo}_{fname}.pickle', 'wb') as output:
        pickle.dump(track_pivots, output, protocol=pickle.HIGHEST_PROTOCOL)