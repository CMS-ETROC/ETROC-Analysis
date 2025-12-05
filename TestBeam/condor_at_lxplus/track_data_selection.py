import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm

## --------------------------------------
def pixel_filter(
        input_df: pd.DataFrame,
        pixel_dict: dict,
    ):

    masks = {}
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
def determine_tot_cut_range_for_trig(
        input_df: pd.DataFrame,
        trig_id: int,
):
    from scipy.signal import argrelextrema

    trig_tot = input_df.loc[input_df['board'] == trig_id]['tot'].reset_index(drop=True)
    counts, bin_centers = np.histogram(trig_tot, bins=128, range=(0, 512))

    first_peak_index = np.argmax(counts)
    minima_indices = argrelextrema(counts, np.less)[0]

    valley_candidates = minima_indices[minima_indices > first_peak_index]
    if len(valley_candidates) > 0:
        valley_index = valley_candidates[0]
    else:
        valley_index = None  # fallback

    if valley_index is not None:
        filtered_tot_trig = trig_tot.loc[trig_tot < bin_centers[valley_index]]
        tot_range = [filtered_tot_trig.quantile(0.02), filtered_tot_trig.quantile(0.98)]
    else:
        tot_range = [trig_tot.quantile(0.03), trig_tot.quantile(0.96)]

    return tot_range

## --------------------------------------
def data_3board_selection_by_track(
        input_df: pd.DataFrame,
        cal_mode_table: pd.DataFrame,
        pix_dict: dict,
        trig_id: int,
        ref_id: int,
        board_to_analyze: list[int],
        tot_cuts: list[int],
    ):

    track_tmp_df = pixel_filter(input_df, pix_dict)

    if sorted(track_tmp_df['board'].unique().tolist()) != board_to_analyze:
        print('dataframe is not sufficient after pixel selection. Exit the function')
        return pd.DataFrame()

    ## Selecting good hits with TDC cuts
    tdc_cuts = {}
    for idx in board_to_analyze:
        # board ID: [CAL LB, CAL UB, TOA LB, TOA UB, TOT LB, TOT UB]
        cal_mode = cal_mode_table.loc[(cal_mode_table['board'] == idx) & (cal_mode_table['row'] == pix_dict[idx][0]) & (cal_mode_table['col'] == pix_dict[idx][1])]['cal_mode'].values[0]
        if idx == trig_id:
            tdc_cuts[idx] = [cal_mode-3, cal_mode+3, 0, 1100, tot_cuts[0], tot_cuts[1]]
        elif idx == ref_id:
            tdc_cuts[idx] = [cal_mode-3, cal_mode+3, 0, 1100, 0, 600]
        else:
            tdc_cuts[idx] = [cal_mode-3, cal_mode+3, 0, 1100, 0, 600]

    track_tmp_df = tdc_event_selection(track_tmp_df, tdc_cuts_dict=tdc_cuts)

    if sorted(track_tmp_df['board'].unique().tolist()) != board_to_analyze:
        print('dataframe is not sufficient after TDC cut. Exit the function')
        return pd.DataFrame()

    event_board_counts = track_tmp_df.groupby(['evt', 'board']).size().unstack(fill_value=0)
    event_selection_col = (event_board_counts[board_to_analyze[0]] == 1) & (event_board_counts[board_to_analyze[1]] == 1) & (event_board_counts[board_to_analyze[2]] == 1)

    isolated_df = track_tmp_df.loc[track_tmp_df['evt'].isin(event_board_counts[event_selection_col].index)]

    pivot_table = isolated_df.pivot(index=["evt"], columns=["board"], values=["row", "col", "toa", "tot", "cal"])
    pivot_table = pivot_table.reset_index(drop=True)

    ## Pivot Table to make tracks
    return pivot_table


## --------------------------------------
def data_4board_selection_by_track(
        input_df: pd.DataFrame,
        cal_mode_table: pd.DataFrame,
        pix_dict: dict,
        trig_id: int,
        board_to_analyze: list[int],
        tot_cuts: list[int],
    ):

    track_tmp_df = pixel_filter(input_df, pix_dict)

    if sorted(track_tmp_df['board'].unique().tolist()) != board_to_analyze:
        print('dataframe is not sufficient after pixel selection. Exit the function')
        return pd.DataFrame()

    ## Selecting good hits with TDC cuts
    tdc_cuts = {}
    for idx in board_to_analyze:
        # board ID: [CAL LB, CAL UB, TOA LB, TOA UB, TOT LB, TOT UB]
        cal_mode = cal_mode_table.loc[(cal_mode_table['board'] == idx) & (cal_mode_table['row'] == pix_dict[idx][0]) & (cal_mode_table['col'] == pix_dict[idx][1])]['cal_mode'].values[0]
        if idx == trig_id:
            tdc_cuts[idx] = [cal_mode-3, cal_mode+3, 0, 1100, tot_cuts[0], tot_cuts[1]]
        else:
            tdc_cuts[idx] = [cal_mode-3, cal_mode+3, 0, 1100, 0, 600]

    track_tmp_df = tdc_event_selection(track_tmp_df, tdc_cuts_dict=tdc_cuts)

    if sorted(track_tmp_df['board'].unique().tolist()) != board_to_analyze:
        print('dataframe is not sufficient after TDC cut. Exit the function')
        return pd.DataFrame()

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
        '--cal_table',
        metavar = 'NAME',
        type = str,
        help = 'csv file including CAL mode values per board, per pixel',
        required = True,
        dest = 'cal_table',
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
        '--extraID',
        metavar = 'ID',
        type = int,
        help = 'board ID be ignored',
        dest = 'extraID',
    )
    args = parser.parse_args()

    trig_id = args.trigID
    ref_id = args.refID
    dut_id = args.dutID

    columns_to_read = ['evt', 'board', 'row', 'col', 'toa', 'tot', 'cal']

    run_df = pd.read_feather(args.inputfile, columns=columns_to_read)
    file_num_indicator = int(args.inputfile.split('.')[0].split('_')[1])

    tot_cuts = determine_tot_cut_range_for_trig(run_df, trig_id)

    if run_df.empty:
        print('Empty input file!')
        exit(0)

    if run_df['board'].nunique() < 3:
        print('Dataframe does not have at least 3 boards information')
        exit(0)

    cal_table = pd.read_csv(args.cal_table)
    track_pivots = defaultdict(pd.DataFrame)

    # Load data
    track_df = pd.read_csv(args.track)

    # Robustly identify which boards are actually in the file
    # We check if 'row_0', 'row_1', etc. exist in the columns
    present_boards = sorted([i for i in [0, 1, 2, 3] if f'row_{i}' in track_df.columns])

    # 2. Use the length of present_boards to decide logic
    if len(present_boards) == 4:
        print('Track separation with 4 boards combination')

        for itrack in tqdm(range(track_df.shape[0])):
            pix_dict = {}
            # Use present_boards instead of hardcoded [0, 1, 2, 3] to be safe
            for idx in present_boards:
                pix_dict[idx] = [track_df.iloc[itrack][f'row_{idx}'], track_df.iloc[itrack][f'col_{idx}']]

            table = data_4board_selection_by_track(input_df=run_df, cal_mode_table=cal_table, pix_dict=pix_dict, trig_id=trig_id,
                                                   board_to_analyze=present_boards, tot_cuts=tot_cuts)
            table['file'] = file_num_indicator
            table['file'] = table['file'].astype('uint16')

            track_pivots[itrack] = table

    else:
        print(f'Track separation with 3 boards combination. Found: {present_boards}')

        # Logic to determine which board is missing (the 'extra' or ignored one)
        # If args.extraID is None, we deduce it from what's missing in the CSV
        if args.extraID is None:
            # Find the ID that is in [0,1,2,3] but NOT in present_boards
            missing_ids = list(set([0, 1, 2, 3]) - set(present_boards))
            id_to_ignore = missing_ids[0] if missing_ids else None
        else:
            id_to_ignore = args.extraID

        reduced_run_df = run_df.loc[~(run_df['board'] == id_to_ignore)]

        # The boards to analyze are simply the ones we found in the CSV
        board_to_analyze = present_boards

        for itrack in tqdm(range(track_df.shape[0])):
            pix_dict = {}
            for idx in board_to_analyze:
                pix_dict[idx] = [track_df.iloc[itrack][f'row_{idx}'], track_df.iloc[itrack][f'col_{idx}']]

            table = data_3board_selection_by_track(input_df=reduced_run_df, cal_mode_table=cal_table, pix_dict=pix_dict, trig_id=trig_id, ref_id=ref_id,
                                                board_to_analyze=board_to_analyze, tot_cuts=tot_cuts)

            table['file'] = file_num_indicator
            table['file'] = table['file'].astype('uint16')

            track_pivots[itrack] = table

    fname = args.inputfile.split('.')[0]
    ### Save python dictionary in pickle format
    with open(f'{args.runinfo}_{fname}.pickle', 'wb') as output:
        pickle.dump(track_pivots, output, protocol=pickle.HIGHEST_PROTOCOL)
