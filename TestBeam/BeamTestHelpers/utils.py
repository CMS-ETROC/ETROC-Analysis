import numpy as np
import pandas as pd
import hist

from pathlib import Path


__all__ = [
    'efficiency_with_single_board',
    'efficiency_with_two_boards',
    'singlehit_event_clear',
    'tdc_event_selection',
    'tdc_event_selection_pivot',
    'pixel_filter',
    'making_pivot',
    'return_broadcast_dataframe',
    'save_TDC_summary_table',
    'return_TOA_correlation_param',
    'return_TWC_param',
    'return_resolution_three_board',
    'return_resolution_three_board_fromFWHM',
    'return_resolution_four_board',
    'return_resolution_four_board_fromFWHM',
    'return_board_resolution',
]

## --------------- Modify DataFrame -----------------------
## --------------------------------------
def efficiency_with_single_board(
        input_df: pd.DataFrame,
        pixel: set = (8, 8), # (row, col)
        board_id: int = 0,
    ):
    df_tmp = input_df.set_index('evt')
    selection = (df_tmp['board'] == board_id) & (df_tmp['row'] == pixel[0]) & (df_tmp['col'] == pixel[1])
    new_df = input_df.loc[input_df['evt'].isin(df_tmp.loc[selection].index)]

    del df_tmp, selection
    return new_df

## --------------------------------------
def efficiency_with_two_boards(
        input_df: pd.DataFrame,
        pixel: set = (8, 8), # (row, col)
        board_ids: set = (0, 3), #(board 1, board 2)
    ):

    df_tmp = input_df.set_index('evt')
    selection1 = (df_tmp['board'] == board_ids[0]) & (df_tmp['row'] == pixel[0]) & (df_tmp['col'] == pixel[1])
    selection2 = (df_tmp['board'] == board_ids[1]) & (df_tmp['row'] == pixel[0]) & (df_tmp['col'] == pixel[1])

    filtered_index = list(set(df_tmp.loc[selection1].index).intersection(df_tmp.loc[selection2].index))
    new_df = input_df.loc[input_df['evt'].isin(filtered_index)]

    del df_tmp, filtered_index, selection1, selection2
    return new_df

## --------------------------------------
def singlehit_event_clear(
        input_df: pd.DataFrame,
        ignore_boards: list[int] = None
    ):

    ana_df = input_df
    if ignore_boards is not None:
        for board in ignore_boards:
            ana_df = ana_df.loc[ana_df['board'] != board].copy()

    ## event has one hit from each board
    event_board_counts = ana_df.groupby(['evt', 'board']).size().unstack(fill_value=0)
    event_selection_col = None
    for board in event_board_counts:
        if event_selection_col is None:
            event_selection_col = (event_board_counts[board] == 1)
        else:
            event_selection_col = event_selection_col & (event_board_counts[board] == 1)
    selected_event_numbers = event_board_counts[event_selection_col].index
    selected_subset_df = ana_df[ana_df['evt'].isin(selected_event_numbers)]
    selected_subset_df.reset_index(inplace=True, drop=True)

    del ana_df, event_board_counts, event_selection_col

    return selected_subset_df

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
def tdc_event_selection_pivot(
        input_df: pd.DataFrame,
        tdc_cuts_dict: dict
    ):
    # Create boolean masks for each board's filtering criteria
    masks = {}
    for board, cuts in tdc_cuts_dict.items():
        mask = (
            input_df['cal'][board].between(cuts[0], cuts[1]) &
            input_df['toa'][board].between(cuts[2], cuts[3]) &
            input_df['tot'][board].between(cuts[4], cuts[5])
        )
        masks[board] = mask

    # Combine the masks using logical AND
    combined_mask = pd.concat(masks, axis=1).all(axis=1)
    del masks

    # Apply the combined mask to the DataFrame
    tdc_filtered_df = input_df[combined_mask].reset_index(drop=True)
    del combined_mask
    return tdc_filtered_df

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
def return_broadcast_dataframe(
        input_df: pd.DataFrame,
        reference_board_id: int,
        board_id_want_broadcast: int,
    ):

    tmp_df = input_df.loc[(input_df['board'] == reference_board_id) | (input_df['board'] == board_id_want_broadcast)]
    tmp_df = tmp_df.drop(columns=['ea', 'toa', 'tot', 'cal'])

    event_board_counts = tmp_df.groupby(['evt', 'board']).size().unstack(fill_value=0)
    event_selections = (event_board_counts[board_id_want_broadcast] == 1) & (event_board_counts[reference_board_id] == 1)
    single_hit_df = tmp_df.loc[tmp_df['evt'].isin(event_board_counts[event_selections].index)]
    single_hit_df.reset_index(inplace=True, drop=True)

    if 'identifier' in single_hit_df.columns:
        single_hit_df = single_hit_df.drop(columns=['identifier'])

    sub_single_df1 = single_hit_df.loc[single_hit_df['board'] == board_id_want_broadcast]
    sub_single_df2 = single_hit_df.loc[single_hit_df['board'] == reference_board_id]

    single_df = pd.merge(sub_single_df1, sub_single_df2, on='evt', suffixes=[f'_{board_id_want_broadcast}', f'_{reference_board_id}'])
    single_df = single_df.drop(columns=['evt'])
    del single_hit_df, sub_single_df1, sub_single_df2

    event_selections = (event_board_counts[board_id_want_broadcast] == 1) & (event_board_counts[reference_board_id] >= 2)
    multi_hit_df = tmp_df.loc[tmp_df['evt'].isin(event_board_counts[event_selections].index)]
    multi_hit_df.reset_index(inplace=True, drop=True)

    sub_multiple_df1 = multi_hit_df.loc[multi_hit_df['board'] == board_id_want_broadcast]
    sub_multiple_df2 = multi_hit_df.loc[multi_hit_df['board'] == reference_board_id]

    multi_df = pd.merge(sub_multiple_df1, sub_multiple_df2, on='evt', suffixes=[f'_{board_id_want_broadcast}', f'_{reference_board_id}'])
    multi_df = multi_df.drop(columns=['evt'])
    del multi_hit_df, tmp_df, sub_multiple_df1, sub_multiple_df2

    return single_df, multi_df


## --------------- Modify DataFrame -----------------------




## --------------- Extract results -----------------------
## --------------------------------------
def save_TDC_summary_table(
        input_df: pd.DataFrame,
        chipLabels: list[int],
        var: str,
        save_path: Path,
        fname_tag: str,
    ):

    for id in chipLabels:

        if input_df[input_df['board'] == id].empty:
            continue

        sum_group = input_df[input_df['board'] == id].groupby(["col", "row"]).agg({var:['mean','std']})
        sum_group.columns = sum_group.columns.droplevel()
        sum_group.reset_index(inplace=True)

        table_mean = sum_group.pivot_table(index='row', columns='col', values='mean')
        table_mean = table_mean.round(1)

        table_mean = table_mean.reindex(pd.Index(np.arange(0,16), name='')).reset_index()
        table_mean = table_mean.reindex(columns=np.arange(0,16))

        table_std = sum_group.pivot_table(index='row', columns='col', values='std')
        table_std = table_std.round(2)

        table_std = table_std.reindex(pd.Index(np.arange(0,16), name='')).reset_index()
        table_std = table_std.reindex(columns=np.arange(0,16))

        table_mean.to_pickle(save_path / f"{fname_tag}_mean.pkl")
        table_std.to_pickle(save_path / f"{fname_tag}_std.pkl")

        del sum_group, table_mean, table_std

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
def return_TWC_param(
        corr_toas: dict,
        input_df: pd.DataFrame,
        board_ids: list[int],
    ):

    results = {}

    del_toa_b0 = (0.5*(corr_toas[f'toa_b{board_ids[1]}'] + corr_toas[f'toa_b{board_ids[2]}']) - corr_toas[f'toa_b{board_ids[0]}'])
    del_toa_b1 = (0.5*(corr_toas[f'toa_b{board_ids[0]}'] + corr_toas[f'toa_b{board_ids[2]}']) - corr_toas[f'toa_b{board_ids[1]}'])
    del_toa_b2 = (0.5*(corr_toas[f'toa_b{board_ids[0]}'] + corr_toas[f'toa_b{board_ids[1]}']) - corr_toas[f'toa_b{board_ids[2]}'])

    coeff_b0 = np.polyfit(input_df[f'tot_b{board_ids[0]}'].values, del_toa_b0, 1)
    results[0] = (input_df[f'tot_b{board_ids[0]}'].values*coeff_b0[0] - del_toa_b0 + coeff_b0[1])/(np.sqrt(coeff_b0[0]**2 + 1))

    coeff_b1 = np.polyfit(input_df[f'tot_b{board_ids[1]}'].values, del_toa_b1, 1)
    results[1] = (input_df[f'tot_b{board_ids[1]}'].values*coeff_b1[0] - del_toa_b1 + coeff_b1[1])/(np.sqrt(coeff_b1[0]**2 + 1))

    coeff_b2 = np.polyfit(input_df[f'tot_b{board_ids[2]}'].values, del_toa_b2, 1)
    results[2] = (input_df[f'tot_b{board_ids[2]}'].values*coeff_b2[0] - del_toa_b2 + coeff_b2[1])/(np.sqrt(coeff_b2[0]**2 + 1))

    return results

## --------------- Extract results -----------------------


## --------------- Result -----------------------
## --------------------------------------
def return_resolution_three_board(
        fit_params: dict,
        var: list,
        board_ids:list,
    ):

    results = {
        board_ids[0]: np.sqrt(0.5*(fit_params[var[0]][0]**2 + fit_params[var[1]][0]**2 - fit_params[var[2]][0]**2))*1e3,
        board_ids[1]: np.sqrt(0.5*(fit_params[var[0]][0]**2 + fit_params[var[2]][0]**2 - fit_params[var[1]][0]**2))*1e3,
        board_ids[2]: np.sqrt(0.5*(fit_params[var[1]][0]**2 + fit_params[var[2]][0]**2 - fit_params[var[0]][0]**2))*1e3,
    }

    return results

## --------------------------------------
def return_resolution_three_board_fromFWHM(
        fit_params: dict,
        var: list,
        board_ids:list,
    ):

    results = {
        board_ids[0]: np.sqrt(0.5*(fit_params[var[0]]**2 + fit_params[var[1]]**2 - fit_params[var[2]]**2)),
        board_ids[1]: np.sqrt(0.5*(fit_params[var[0]]**2 + fit_params[var[2]]**2 - fit_params[var[1]]**2)),
        board_ids[2]: np.sqrt(0.5*(fit_params[var[1]]**2 + fit_params[var[2]]**2 - fit_params[var[0]]**2)),
    }

    return results

## --------------------------------------
def return_resolution_four_board(
        fit_params: dict,
    ):

    results = {
        0: np.sqrt((1/6)*(2*fit_params['01'][0]**2+2*fit_params['02'][0]**2+2*fit_params['03'][0]**2-fit_params['12'][0]**2-fit_params['13'][0]**2-fit_params['23'][0]**2))*1e3,
        1: np.sqrt((1/6)*(2*fit_params['01'][0]**2+2*fit_params['12'][0]**2+2*fit_params['13'][0]**2-fit_params['02'][0]**2-fit_params['03'][0]**2-fit_params['23'][0]**2))*1e3,
        2: np.sqrt((1/6)*(2*fit_params['02'][0]**2+2*fit_params['12'][0]**2+2*fit_params['23'][0]**2-fit_params['01'][0]**2-fit_params['03'][0]**2-fit_params['13'][0]**2))*1e3,
        3: np.sqrt((1/6)*(2*fit_params['03'][0]**2+2*fit_params['13'][0]**2+2*fit_params['23'][0]**2-fit_params['01'][0]**2-fit_params['02'][0]**2-fit_params['12'][0]**2))*1e3,
    }

    return results

## --------------------------------------
def return_resolution_four_board_fromFWHM(
        fit_params: dict,
    ):

    results = {
        0: np.sqrt((1/6)*(2*fit_params['01']**2+2*fit_params['02']**2+2*fit_params['03']**2-fit_params['12']**2-fit_params['13']**2-fit_params['23']**2)),
        1: np.sqrt((1/6)*(2*fit_params['01']**2+2*fit_params['12']**2+2*fit_params['13']**2-fit_params['02']**2-fit_params['03']**2-fit_params['23']**2)),
        2: np.sqrt((1/6)*(2*fit_params['02']**2+2*fit_params['12']**2+2*fit_params['23']**2-fit_params['01']**2-fit_params['03']**2-fit_params['13']**2)),
        3: np.sqrt((1/6)*(2*fit_params['03']**2+2*fit_params['13']**2+2*fit_params['23']**2-fit_params['01']**2-fit_params['02']**2-fit_params['12']**2)),
    }

    return results

## --------------------------------------
def return_board_resolution(
        input_df: pd.DataFrame,
        board_ids: list[int],
        key_names: list[str],
        hist_bins: int = 15,
    ):

    from collections import defaultdict
    from lmfit.models import GaussianModel
    mod = GaussianModel(nan_policy='omit')

    results = defaultdict(float)

    for key in range(len(board_ids)):
        hist_x_min = int(input_df[f'res{board_ids[key]}'].min())-5
        hist_x_max = int(input_df[f'res{board_ids[key]}'].max())+5
        h_temp = hist.Hist(hist.axis.Regular(hist_bins, hist_x_min, hist_x_max, name="time_resolution", label=r'Time Resolution [ps]'))
        h_temp.fill(input_df[f'res{board_ids[key]}'].values)
        mean = np.mean(input_df[f'res{board_ids[key]}'].values)
        std = np.std(input_df[f'res{board_ids[key]}'].values)
        centers = h_temp.axes[0].centers
        fit_range = centers[np.argmax(h_temp.values())-5:np.argmax(h_temp.values())+5]
        fit_vals = h_temp.values()[np.argmax(h_temp.values())-5:np.argmax(h_temp.values())+5]

        pars = mod.guess(fit_vals, x=fit_range)
        out = mod.fit(fit_vals, pars, x=fit_range, weights=1/np.sqrt(fit_vals))

        results[f'{key_names[key]}_mean'] = mean
        results[f'{key_names[key]}_std'] = std
        results[f'{key_names[key]}_res'] = out.params['center'].value
        results[f'{key_names[key]}_err'] = abs(out.params['sigma'].value)

    return results

## --------------- Result -----------------------