import pandas as pd
import numpy as np
import sys
import os
from glob import glob
from tqdm import tqdm
from collections import defaultdict
import warnings
import matplotlib.pyplot as plt
import pickle
import subprocess
import csv
import mplhep as hep
import matplotlib.colors as colors
import hist
from pathlib import Path
warnings.filterwarnings("ignore")

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

def three_board_iterative_timewalk_correction(
    input_df: pd.DataFrame,
    iterative_cnt: int,
    poly_order: int,
    board_list: list,
    RunNum: str,
    counter: int,
    output_name: str,
    RCid: str = '20_20',
):
    if args.plotting:
        if counter == 0:
            plot_TWC(input_df=input_df, board_ids=board_list, RunNum=RunNum, poly_order=poly_order , print_func=False, filename=f'Run_{RunNum}_TWC_plots_{output_name}')

    corr_toas = {}
    corr_b0 = input_df[f'toa_b{board_list[0]}'].values
    corr_b1 = input_df[f'toa_b{board_list[1]}'].values
    corr_b2 = input_df[f'toa_b{board_list[2]}'].values

    del_toa_b0 = (0.5*(input_df[f'toa_b{board_list[1]}'] + input_df[f'toa_b{board_list[2]}']) - input_df[f'toa_b{board_list[0]}']).values
    del_toa_b1 = (0.5*(input_df[f'toa_b{board_list[0]}'] + input_df[f'toa_b{board_list[2]}']) - input_df[f'toa_b{board_list[1]}']).values
    del_toa_b2 = (0.5*(input_df[f'toa_b{board_list[0]}'] + input_df[f'toa_b{board_list[1]}']) - input_df[f'toa_b{board_list[2]}']).values

    for i in range(iterative_cnt):
        coeff_b0 = np.polyfit(input_df[f'tot_b{board_list[0]}'].values, del_toa_b0, poly_order)
        poly_func_b0 = np.poly1d(coeff_b0)
        for order in range(poly_order+1):
            coeffs[f'{RunNum}_1_{i}_{order}_{RCid}'].append(coeff_b0[order].item())

        coeff_b1 = np.polyfit(input_df[f'tot_b{board_list[1]}'].values, del_toa_b1, poly_order)
        poly_func_b1 = np.poly1d(coeff_b1)
        for order in range(poly_order+1):
            coeffs[f'{RunNum}_2_{i}_{order}_{RCid}'].append(coeff_b1[order].item())

        coeff_b2 = np.polyfit(input_df[f'tot_b{board_list[2]}'].values, del_toa_b2, poly_order)
        poly_func_b2 = np.poly1d(coeff_b2)
        for order in range(poly_order+1):
            coeffs[f'{RunNum}_3_{i}_{order}_{RCid}'].append(coeff_b2[order].item())

        corr_b0 = corr_b0 + poly_func_b0(input_df[f'tot_b{board_list[0]}'].values)
        corr_b1 = corr_b1 + poly_func_b1(input_df[f'tot_b{board_list[1]}'].values)
        corr_b2 = corr_b2 + poly_func_b2(input_df[f'tot_b{board_list[2]}'].values)

        del_toa_b0 = (0.5*(corr_b1 + corr_b2) - corr_b0)
        del_toa_b1 = (0.5*(corr_b0 + corr_b2) - corr_b1)
        del_toa_b2 = (0.5*(corr_b0 + corr_b1) - corr_b2)

        if i == iterative_cnt-1:
            corr_toas[f'toa_b{board_list[0]}'] = corr_b0
            corr_toas[f'toa_b{board_list[1]}'] = corr_b1
            corr_toas[f'toa_b{board_list[2]}'] = corr_b2

    return corr_toas

def three_board_iterative_timewalk_correction_integrated(
    input_df: pd.DataFrame,
    iterative_cnt: int,
    poly_order: int,
    board_ids: list,
):

    corr_toas = {}
    corr_b0 = input_df[f'toa_b{board_ids[0]}'].values
    corr_b1 = input_df[f'toa_b{board_ids[1]}'].values
    corr_b2 = input_df[f'toa_b{board_ids[2]}'].values

    del_toa_b0 = (0.5*(input_df[f'toa_b{board_ids[1]}'] + input_df[f'toa_b{board_ids[2]}']) - input_df[f'toa_b{board_ids[0]}']).values
    del_toa_b1 = (0.5*(input_df[f'toa_b{board_ids[0]}'] + input_df[f'toa_b{board_ids[2]}']) - input_df[f'toa_b{board_ids[1]}']).values
    del_toa_b2 = (0.5*(input_df[f'toa_b{board_ids[0]}'] + input_df[f'toa_b{board_ids[1]}']) - input_df[f'toa_b{board_ids[2]}']).values

    for i in range(iterative_cnt):
        # print('-------------')
        # print(f'TWC loop {i}')

        ##### CHANGE THIS TO BE A CONSTANT ONLY ON THE REF RUN #####
        if RunNum == RunNums[0] and times == 0:
            coeff_b0 = np.polyfit(input_df[f'tot_b{board_ids[0]}'].values, del_toa_b0, poly_order)
            poly_fits[f'poly_func_b0_iter_{i}'] = np.poly1d(coeff_b0)

            coeff_b1 = np.polyfit(input_df[f'tot_b{board_ids[1]}'].values, del_toa_b1, poly_order)
            poly_fits[f'poly_func_b1_iter_{i}'] = np.poly1d(coeff_b1)

            coeff_b2 = np.polyfit(input_df[f'tot_b{board_ids[2]}'].values, del_toa_b2, poly_order)
            poly_fits[f'poly_func_b2_iter_{i}'] = np.poly1d(coeff_b2)

        #############################################################

        corr_b0 = corr_b0 + poly_fits[f'poly_func_b0_iter_{i}'](input_df[f'tot_b{board_ids[0]}'].values)
        corr_b1 = corr_b1 + poly_fits[f'poly_func_b1_iter_{i}'](input_df[f'tot_b{board_ids[1]}'].values)
        corr_b2 = corr_b2 + poly_fits[f'poly_func_b2_iter_{i}'](input_df[f'tot_b{board_ids[2]}'].values)

        del_toa_b0 = (0.5*(corr_b1 + corr_b2) - corr_b0)
        del_toa_b1 = (0.5*(corr_b0 + corr_b2) - corr_b1)
        del_toa_b2 = (0.5*(corr_b0 + corr_b1) - corr_b2)

        if i == iterative_cnt-1: # when on last itteration
            corr_toas[f'toa_b{board_ids[0]}'] = corr_b0
            corr_toas[f'toa_b{board_ids[1]}'] = corr_b1
            corr_toas[f'toa_b{board_ids[2]}'] = corr_b2

            ##########################

            if RunNum == RunNums[0] and times == 0:
                coeff_b0 = np.polyfit(input_df[f'tot_b{board_ids[0]}'].values, del_toa_b0, poly_order)
                poly_fits['poly_func_b0_Final'] = np.poly1d(coeff_b0)

                coeff_b1 = np.polyfit(input_df[f'tot_b{board_ids[1]}'].values, del_toa_b1, poly_order)
                poly_fits['poly_func_b1_Final'] = np.poly1d(coeff_b1)

                coeff_b2 = np.polyfit(input_df[f'tot_b{board_ids[2]}'].values, del_toa_b2, poly_order)
                poly_fits['poly_func_b2_Final'] = np.poly1d(coeff_b2)

            # print(poly_fits['poly_func_b2_Final'])
            ##########################

    return corr_toas

## --------------------------------------
def four_board_iterative_timewalk_correction(
    input_df: pd.DataFrame,
    iterative_cnt: int,
    poly_order: int,
    ):

    corr_toas = {}
    corr_b0 = input_df['toa_b0'].values
    corr_b1 = input_df['toa_b1'].values
    corr_b2 = input_df['toa_b2'].values
    corr_b3 = input_df['toa_b3'].values

    del_toa_b3 = ((1/3)*(input_df['toa_b0'] + input_df['toa_b1'] + input_df['toa_b2']) - input_df['toa_b3']).values
    del_toa_b2 = ((1/3)*(input_df['toa_b0'] + input_df['toa_b1'] + input_df['toa_b3']) - input_df['toa_b2']).values
    del_toa_b1 = ((1/3)*(input_df['toa_b0'] + input_df['toa_b3'] + input_df['toa_b2']) - input_df['toa_b1']).values
    del_toa_b0 = ((1/3)*(input_df['toa_b3'] + input_df['toa_b1'] + input_df['toa_b2']) - input_df['toa_b0']).values

    for i in range(iterative_cnt):
        coeff_b0 = np.polyfit(input_df['tot_b0'].values, del_toa_b0, poly_order)
        poly_func_b0 = np.poly1d(coeff_b0)

        coeff_b1 = np.polyfit(input_df['tot_b1'].values, del_toa_b1, poly_order)
        poly_func_b1 = np.poly1d(coeff_b1)

        coeff_b2 = np.polyfit(input_df['tot_b2'].values, del_toa_b2, poly_order)
        poly_func_b2 = np.poly1d(coeff_b2)

        coeff_b3 = np.polyfit(input_df['tot_b3'].values, del_toa_b3, poly_order)
        poly_func_b3 = np.poly1d(coeff_b3)

        corr_b0 = corr_b0 + poly_func_b0(input_df['tot_b0'].values)
        corr_b1 = corr_b1 + poly_func_b1(input_df['tot_b1'].values)
        corr_b2 = corr_b2 + poly_func_b2(input_df['tot_b2'].values)
        corr_b3 = corr_b3 + poly_func_b3(input_df['tot_b3'].values)

        del_toa_b3 = ((1/3)*(corr_b0 + corr_b1 + corr_b2) - corr_b3)
        del_toa_b2 = ((1/3)*(corr_b0 + corr_b1 + corr_b3) - corr_b2)
        del_toa_b1 = ((1/3)*(corr_b0 + corr_b3 + corr_b2) - corr_b1)
        del_toa_b0 = ((1/3)*(corr_b3 + corr_b1 + corr_b2) - corr_b0)

        if i == iterative_cnt-1:
            corr_toas[f'toa_b0'] = corr_b0
            corr_toas[f'toa_b1'] = corr_b1
            corr_toas[f'toa_b2'] = corr_b2
            corr_toas[f'toa_b3'] = corr_b3

    return corr_toas

## --------------------------------------
def fwhm_based_on_gaussian_mixture_model(
        input_data: np.array,
        n_components: int = 2,
        plotting: bool = False,
        plotting_each_component: bool = False,
    ):

    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
    from scipy.spatial import distance
    import matplotlib.pyplot as plt

    x_range = np.linspace(input_data.min(), input_data.max(), 1000).reshape(-1, 1)
    bins, edges = np.histogram(input_data, bins=30, density=True)
    centers = 0.5*(edges[1:] + edges[:-1])
    # print('Problem start')
    models = GaussianMixture(n_components=n_components,covariance_type='diag').fit(input_data.reshape(-1, 1))
    # print('Problem end')
    silhouette_eval_score = silhouette_score(centers.reshape(-1, 1), models.predict(centers.reshape(-1, 1)))

    logprob = models.score_samples(centers.reshape(-1, 1))
    pdf = np.exp(logprob)
    jensenshannon_score = distance.jensenshannon(bins, pdf)

    logprob = models.score_samples(x_range)
    pdf = np.exp(logprob)

    peak_height = np.max(pdf)

    # Find the half-maximum points.
    half_max = peak_height*0.5
    half_max_indices = np.where(pdf >= half_max)[0]

    # Calculate the FWHM.
    fwhm = x_range[half_max_indices[-1]] - x_range[half_max_indices[0]]

    ### Draw plot
    if plotting_each_component:
        # Compute PDF for each component
        responsibilities = models.predict_proba(x_range)
        pdf_individual = responsibilities * pdf[:, np.newaxis]

    if plotting:

        fig, ax = plt.subplots(figsize=(10,10))

        # Plot data histogram
        bins, _, _ = ax.hist(input_data, bins=30, density=True, histtype='stepfilled', alpha=0.4, label='Data')

        # Plot PDF of whole model
        ax.plot(x_range, pdf, '-k', label='Mixture PDF')

        if plotting_each_component:
            # Plot PDF of each component
            ax.plot(x_range, pdf_individual, '--', label='Component PDF')

        # Plot
        ax.vlines(x_range[half_max_indices[0]],  ymin=0, ymax=np.max(bins)*0.75, lw=1.5, colors='red', label='FWHM')
        ax.vlines(x_range[half_max_indices[-1]], ymin=0, ymax=np.max(bins)*0.75, lw=1.5, colors='red')

        ax.legend(loc='best', fontsize=14)

    return fwhm, [silhouette_eval_score, jensenshannon_score]

## --------------------------------------
def return_resolution_three_board_fromFWHM(
        fit_params: dict,
        var: list,
        board_list:list,
    ):

    results = {
        board_list[0]: np.sqrt(0.5*(fit_params[var[0]]**2 + fit_params[var[1]]**2 - fit_params[var[2]]**2)),
        board_list[1]: np.sqrt(0.5*(fit_params[var[0]]**2 + fit_params[var[2]]**2 - fit_params[var[1]]**2)),
        board_list[2]: np.sqrt(0.5*(fit_params[var[1]]**2 + fit_params[var[2]]**2 - fit_params[var[0]]**2)),
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
def bootstrap(
        input_df: pd.DataFrame,
        board_to_analyze: list[int],
        iteration: int = 100,
        sampling_fraction: int = 75,
        minimum_nevt_cut: int = 1000,
        do_reproducible: bool = False,
    ):

    resolution_from_bootstrap = defaultdict(list)
    random_sampling_fraction = sampling_fraction*0.01

    counter = 0
    resample_counter = 0

    while True:

        if counter > 15000:
            print("Loop is over maximum. Escaping bootstrap loop")
            break

        tdc_filtered_df = input_df

        if do_reproducible:
            np.random.seed(counter)

        n = int(random_sampling_fraction*tdc_filtered_df.shape[0])
        indices = np.random.choice(tdc_filtered_df['evt'].unique(), n, replace=False)
        tdc_filtered_df = tdc_filtered_df.loc[tdc_filtered_df['evt'].isin(indices)]

        if tdc_filtered_df.shape[0] < minimum_nevt_cut:
            print(f'Number of events in random sample is {tdc_filtered_df.shape[0]}')
            print('Warning!! Sampling size is too small. Skipping this track')
            break

        df_in_time = pd.DataFrame()

        for idx in board_to_analyze:
            bins = 3.125/tdc_filtered_df['cal'][idx].mean()
            df_in_time[f'toa_b{str(idx)}'] = (12.5 - tdc_filtered_df['toa'][idx] * bins)*1e3
            df_in_time[f'tot_b{str(idx)}'] = ((2*tdc_filtered_df['tot'][idx] - np.floor(tdc_filtered_df['tot'][idx]/32)) * bins)*1e3

        del tdc_filtered_df

        if(len(board_to_analyze)==3):
            corr_toas = three_board_iterative_timewalk_correction(df_in_time, 2, 2, board_list=board_to_analyze)
        elif(len(board_to_analyze)==4):
            corr_toas = four_board_iterative_timewalk_correction(df_in_time, 2, 2)
        else:
            print("You have less than 3 boards to analyze")
            break

        diffs = {}
        for board_a in board_to_analyze:
            for board_b in board_to_analyze:
                if board_b <= board_a:
                    continue
                name = f"{board_a}{board_b}"
                diffs[name] = np.asarray(corr_toas[f'toa_b{board_a}'] - corr_toas[f'toa_b{board_b}'])

        keys = list(diffs.keys())
        try:
            fit_params = {}
            scores = []
            for ikey in diffs.keys():
                params, eval_scores = fwhm_based_on_gaussian_mixture_model(diffs[ikey], n_components=3, plotting=False, plotting_each_component=False)
                fit_params[ikey] = float(params[0]/2.355)
                scores.append(eval_scores)

            if np.any(np.asarray(scores)[:,0] > 0.6) or np.any(np.asarray(scores)[:,1] > 0.075) :
                print('Redo the sampling')
                counter += 1
                resample_counter += 1
                continue

            if(len(board_to_analyze)==3):
                resolutions = return_resolution_three_board_fromFWHM(fit_params, var=keys, board_list=board_to_analyze)
            elif(len(board_to_analyze)==4):
                resolutions = return_resolution_four_board_fromFWHM(fit_params)
            else:
                print("You have less than 3 boards to analyze")
                break

            if any(np.isnan(val) for key, val in resolutions.items()):
                print('At least one of time resolution values is NaN. Skipping this iteration')
                counter += 1
                resample_counter += 1
                continue

            if do_reproducible:
                resolution_from_bootstrap['RandomSeed'].append(counter)

            for key in resolutions.keys():
                resolution_from_bootstrap[key].append(resolutions[key])

            counter += 1

        except Exception as inst:
            print(inst)
            counter += 1
            del diffs, corr_toas

        break_flag = False
        for key, val in resolution_from_bootstrap.items():
            if len(val) > iteration:
                break_flag = True
                break

        if break_flag:
            print('Escaping bootstrap loop')
            break

    print('How many times do resample?', resample_counter)

    ### Empty dictionary case
    if not resolution_from_bootstrap:
        return pd.DataFrame()
    else:
        resolution_from_bootstrap_df = pd.DataFrame(resolution_from_bootstrap)
        return resolution_from_bootstrap_df

## --------------------------------------
def time_df_bootstrap(
        input_df: pd.DataFrame,
        board_to_analyze: list[int],
        poly_order: int,
        RunNum: int,
        output_name: str,
        iteration: int = 100,
        sampling_fraction: int = 75,
        minimum_nevt_cut: int = 1000,
        do_reproducible: bool = False,
        RCid: str = '20_20'
    ):
    resolution_from_bootstrap = defaultdict(list)
    random_sampling_fraction = sampling_fraction*0.01

    counter = 0
    resample_counter = 0

    while True:

        # print(f'counter: {counter}')

        # end bootstrap if loop hits 15k
        if counter > 1000:
            print("Loop is over maximum. Escaping bootstrap loop")
            for board in range(1,4):
                resolution_individual[f'{RunNum}_{board}_{RCid}'] = [0.0]*100
                resolution_grouped[f'{RunNum}_{board}_{RCid}'] = [0.0]*100
                resolution_normal[f'normal_{RunNum}_{board}_{RCid}'] = [0.0]*100
                resolution_mod[f'mod_{RunNum}_{board}_{RCid}'] = [0.0]*100
                for i in range(2):
                    for order in range(poly_order+1):
                        coeffs[f'{RunNum}_{board}_{i}_{order}_{RCid}'] = [-1000.0]*100
            blank_list.append(RC)
            break

        # use specific random seed for reproducable results
        if do_reproducible:
            np.random.seed(counter)

        # random selection of n events from input file
        n = int(random_sampling_fraction*input_df.shape[0])
        indices = np.random.choice(input_df['evt'].unique(), n, replace=False)
        selected_df = input_df.loc[input_df['evt'].isin(indices)]

        # if empty array, skip the track
        if selected_df.shape[0] < minimum_nevt_cut:
            print(f'Number of events in random sample is {selected_df.shape[0]}')
            print('Warning!! Sampling size is too small. Skipping this track')
            break

        # time walk correction
        # ------------------- MAIN EDITS HERE ------------------------------
        # do two runs of TWC on the run
        if(len(board_to_analyze)==3):
            # first TWC is for the normal TWC using new params, second is for the corrected TWC using prepopulated params
            corr_toas = three_board_iterative_timewalk_correction(selected_df, iterative_cnt=2, poly_order=poly_order, board_list=board_to_analyze, RCid=RCid, RunNum=RunNum, counter=counter, output_name=output_name)
            corr_toas_M = three_board_iterative_timewalk_correction_integrated(selected_df, iterative_cnt=2, poly_order=poly_order, board_ids=board_to_analyze)

        # case for a 4 board analysis, not used here 
        elif(len(board_to_analyze)==4):
            corr_toas = four_board_iterative_timewalk_correction(selected_df, 2, 2)
        else:
            print("You have less than 3 boards to analyze")
            break
        # ------------------------------------------------------------------

        # difference of TOAs between each board
        diffs = {}
        diffs_M = {}
        for board_a in board_to_analyze:
            for board_b in board_to_analyze:
                # only want one set, as a-b = b-a, only use when b>a
                if board_b <= board_a:
                    continue
                # name example: "board_1board_2" for the deltaTOA of board 1 and 2
                name = f"{board_a}{board_b}"
                # dict of delta doas, stored in arrays where the key is the name from above diffs = { "key" : [deltaTOAs,...], ...}
                diffs[name] = np.asarray(corr_toas[f'toa_b{board_a}'] - corr_toas[f'toa_b{board_b}'])
                diffs_M[name] = np.asarray(corr_toas_M[f'toa_b{board_a}'] - corr_toas_M[f'toa_b{board_b}'])
                
        # key list is just list of keys from dict above (should only be 3 in the array for 3 board analysis/ 1-2,1-3,2-3)
        keys = list(diffs.keys())
        keys_M = list(diffs_M.keys())

        try:
            fit_params = {}
            fit_params_M = {}
            scores = []
            scores_M = []
            # gets fit parameters for the final poly fit
            for ikey in diffs.keys():
                params, eval_scores = fwhm_based_on_gaussian_mixture_model(diffs[ikey], n_components=3, plotting=False, plotting_each_component=False)
                fit_params[ikey] = float(params[0]/2.355)
                scores.append(eval_scores)
            
            for ikey in diffs_M.keys():
                params_M, eval_scores_M = fwhm_based_on_gaussian_mixture_model(diffs_M[ikey], n_components=3, plotting=False, plotting_each_component=False)
                fit_params_M[ikey] = float(params_M[0]/2.355)
                scores_M.append(eval_scores_M)

            # if any value in the 0th column is > 0.6 (params) or any value in the 1st column > 0.075 (eval_scores), skip itteration
            if np.any(np.asarray(scores)[:,0] > 0.6) or np.any(np.asarray(scores)[:,1] > 0.075):
                # print('Redo the sampling')
                counter += 1
                resample_counter += 1
                for board in range(1,4):
                    for i in range(2):
                        for order in range(poly_order+1):
                            coeffs[f'{RunNum}_{board}_{i}_{order}_{RCid}'].pop()
                continue
            
            # save resolutions from the TW corrected graph
            if(len(board_to_analyze)==3):
                resolutions = return_resolution_three_board_fromFWHM(fit_params, var=keys, board_list=board_to_analyze)
                resolutions_M = return_resolution_three_board_fromFWHM(fit_params_M, var=keys_M, board_list=board_to_analyze)
                # if resolutions != [] and resolutions != '':
                #     print('resolutions print:')
                #     print(resolutions)

                # save resolutions into dicts for later analysis
                for board in range(3):
                    resolution_individual[f'{RunNum}_{board+1}_{RCid}'].append(resolutions[board_to_analyze[board]].item())
                    resolution_grouped[f'{RunNum}_{board+1}_{RCid}'].append(resolutions_M[board_to_analyze[board]].item())
                    resolution_normal[f'normal_{RunNum}_{board+1}_{RCid}'].append(resolutions[board_to_analyze[board]].item())
                    resolution_mod[f'mod_{RunNum}_{board+1}_{RCid}'].append(resolutions_M[board_to_analyze[board]].item())

            elif(len(board_to_analyze)==4):
                resolutions = return_resolution_four_board_fromFWHM(fit_params)
            else:
                print("You have less than 3 boards to analyze")
                break

            # if there are any NaN in the data, skip the itteration
            if any(np.isnan(val) for key, val in resolutions.items()):
                # print('At least one of time resolution values is NaN. Skipping this iteration')
                counter += 1
                resample_counter += 1
                for board in range(3):
                    for i in range(2):
                        for order in range(poly_order+1):
                            coeffs[f'{RunNum}_{board+1}_{i}_{order}_{RCid}'].pop()
                    resolution_individual[f'{RunNum}_{board+1}_{RCid}'].pop()
                    resolution_grouped[f'{RunNum}_{board+1}_{RCid}'].pop()
                    resolution_normal[f'normal_{RunNum}_{board+1}_{RCid}'].pop()
                    resolution_mod[f'mod_{RunNum}_{board+1}_{RCid}'].pop()
                # print('NaN value found')

                continue

            # save the keys for later use
            if do_reproducible:
                resolution_from_bootstrap['RandomSeed'].append(counter)

            for key in resolutions.keys():
                resolution_from_bootstrap[key].append(resolutions[key])

            counter += 1

        # if resolutions fail, restart and recalc parameters
        except Exception as inst:
            print(inst)
            print('here')
            counter += 1
            del diffs, corr_toas

        break_flag = False
            
        for key, val in resolution_from_bootstrap.items():
            # print(len(val))
            if len(val) >= iteration:
                break_flag = True
                break

        if break_flag:
            # print('Escaping bootstrap loop')
            break

    # print('How many times do resample?', resample_counter)

    ### Empty dictionary case
    if not resolution_from_bootstrap:
        return pd.DataFrame()
    else:
        resolution_from_bootstrap_df = pd.DataFrame(resolution_from_bootstrap)
        return resolution_from_bootstrap_df

def track_to_bootstrap(
    directory: str,
    track: str,
    RunNum: int,
    output_name: str,
    poly_order: int,
    RCid: str = '20_20',
):
    for file in os.listdir(directory):
        if file.endswith(f'{track}.pkl'):
            df = pd.read_pickle(os.path.join(directory, file))
    
    df = df.reset_index(names='evt')

    if RunNum <= 54:
        board_ids = [1,2,3]
    if RunNum > 54:
        board_ids = [1,2,0]

    resolution_df = time_df_bootstrap(input_df=df, board_to_analyze=board_ids, iteration=args.iteration,
                                    sampling_fraction=args.sampling, minimum_nevt_cut=args.minimum_nevt,
                                    do_reproducible=args.reproducible, RunNum=RunNum, RCid=RCid, output_name=output_name, poly_order=poly_order)
    return resolution_df

def save_data(
        output_name: str
):
    
    with open(f'{output_name}_ratio_dict.pkl', 'wb') as f:
        pickle.dump(resolution_ratio, f)
    subprocess.run(['mv',f'{output_name}_ratio_dict.pkl','Resolution_Data'])
    with open(f'{output_name}_delta_dict.pkl', 'wb') as f:
        pickle.dump(resolution_delta, f)
    subprocess.run(['mv',f'{output_name}_delta_dict.pkl','Resolution_Data'])
    with open(f'{output_name}_normal_dict.pkl', 'wb') as f:
        pickle.dump(resolution_normal, f)
    subprocess.run(['mv',f'{output_name}_normal_dict.pkl','Resolution_Data'])
    with open(f'{output_name}_mod_dict.pkl', 'wb') as f:
        pickle.dump(resolution_mod, f)
    subprocess.run(['mv',f'{output_name}_mod_dict.pkl','Resolution_Data'])


    coeffs_df = pd.DataFrame(data=coeffs)
    coeffs_df.to_pickle(f'{output_name}_coeffs.pkl')
    subprocess.run(['mv',f'{output_name}_coeffs.pkl','Resolution_Data'])

    resolution_ratio_df = pd.DataFrame(data=resolution_ratio)
    resolution_delta_df = pd.DataFrame(data=resolution_delta)
    resolution_normal_df = pd.DataFrame(data=resolution_normal)
    resolution_mod_df = pd.DataFrame(data=resolution_mod)

    res_df = pd.concat([resolution_ratio_df,resolution_delta_df,resolution_normal_df,resolution_mod_df], axis=1, join='outer')

    res_df.to_pickle(f'{output_name}_resolutions.pkl')
    subprocess.run(['mv',f'{output_name}_resolutions.pkl','Resolution_Data'])

    # if not resolution_df.empty:
    #     # print(resolution_df.tail())
    #     resolution_df.to_pickle(f'{output_name}_resolution.pkl')
    # else:
    #     print(f'With {args.sampling}% sampling, number of events in sample is not enough to do bootstrap')

def plot_TWC(
        input_df: pd.DataFrame,
        board_ids: list[int],
        RunNum: str,
        poly_order: int,
        corr_toas: dict | None = None,
        filename: str | None = None,
        print_func: bool = False,
    ):

    plot_title = f'Run {RunNum} Plot'

    if corr_toas is not None:
        del_toa_b0 = (0.5*(corr_toas[f'toa_b{board_ids[1]}'] + corr_toas[f'toa_b{board_ids[2]}']) - corr_toas[f'toa_b{board_ids[0]}'])
        del_toa_b1 = (0.5*(corr_toas[f'toa_b{board_ids[0]}'] + corr_toas[f'toa_b{board_ids[2]}']) - corr_toas[f'toa_b{board_ids[1]}'])
        del_toa_b2 = (0.5*(corr_toas[f'toa_b{board_ids[0]}'] + corr_toas[f'toa_b{board_ids[1]}']) - corr_toas[f'toa_b{board_ids[2]}'])
    else:
        del_toa_b0 = (0.5*(input_df[f'toa_b{board_ids[1]}'] + input_df[f'toa_b{board_ids[2]}']) - input_df[f'toa_b{board_ids[0]}']).values
        del_toa_b1 = (0.5*(input_df[f'toa_b{board_ids[0]}'] + input_df[f'toa_b{board_ids[2]}']) - input_df[f'toa_b{board_ids[1]}']).values
        del_toa_b2 = (0.5*(input_df[f'toa_b{board_ids[0]}'] + input_df[f'toa_b{board_ids[1]}']) - input_df[f'toa_b{board_ids[2]}']).values

    def roundup(x):
        return int(np.ceil(x / 100.0)) * 100

    tot_ranges = {}
    for idx in board_ids:
        min_value = roundup(input_df[f'tot_b{idx}'].min()) - 500
        max_value = roundup(input_df[f'tot_b{idx}'].max()) + 500
        if min_value < 0:
            min_value = 0
        tot_ranges[idx] = [min_value, max_value]

    h_twc1 = hist.Hist(
        hist.axis.Regular(50, tot_ranges[board_ids[0]][0], tot_ranges[board_ids[0]][1], name=f'tot_b{board_ids[0]}', label=f'tot_b{board_ids[0]}'),
        hist.axis.Regular(50, -3000, 3000, name=f'delta_toa{board_ids[0]}', label=f'delta_toa{board_ids[0]}')
    )
    h_twc2 = hist.Hist(
        hist.axis.Regular(50, tot_ranges[board_ids[1]][0], tot_ranges[board_ids[1]][1], name=f'tot_b{board_ids[1]}', label=f'tot_b{board_ids[1]}'),
        hist.axis.Regular(50, -3000, 3000, name=f'delta_toa{board_ids[1]}', label=f'delta_toa{board_ids[1]}')
    )
    h_twc3 = hist.Hist(
        hist.axis.Regular(50, tot_ranges[board_ids[2]][0], tot_ranges[board_ids[2]][1], name=f'tot_b{board_ids[2]}', label=f'tot_b{board_ids[2]}'),
        hist.axis.Regular(50, -3000, 3000, name=f'delta_toa{board_ids[2]}', label=f'delta_toa{board_ids[2]}')
    )

    h_twc1.fill(input_df[f'tot_b{board_ids[0]}'], del_toa_b0)
    h_twc2.fill(input_df[f'tot_b{board_ids[1]}'], del_toa_b1)
    h_twc3.fill(input_df[f'tot_b{board_ids[2]}'], del_toa_b2)

    b1_xrange = np.linspace(input_df[f'tot_b{board_ids[0]}'].min(), input_df[f'tot_b{board_ids[0]}'].max(), 100)
    b2_xrange = np.linspace(input_df[f'tot_b{board_ids[1]}'].min(), input_df[f'tot_b{board_ids[1]}'].max(), 100)
    b3_xrange = np.linspace(input_df[f'tot_b{board_ids[2]}'].min(), input_df[f'tot_b{board_ids[2]}'].max(), 100)

    coeff_b0 = np.polyfit(input_df[f'tot_b{board_ids[0]}'].values, del_toa_b0, poly_order)
    poly_func_b0 = np.poly1d(coeff_b0)

    coeff_b1 = np.polyfit(input_df[f'tot_b{board_ids[1]}'].values, del_toa_b1, poly_order)
    poly_func_b1 = np.poly1d(coeff_b1)

    coeff_b2 = np.polyfit(input_df[f'tot_b{board_ids[2]}'].values, del_toa_b2, poly_order)
    poly_func_b2 = np.poly1d(coeff_b2)

    def make_legend(coeff, poly_order):
        legend_str = ""
        for i in range(poly_order + 1):
            if round(coeff[i], 2) == 0:
                # Use scientific notation
                coeff_str = f"{coeff[i]:.2e}"
            else:
                # Use fixed-point notation
                coeff_str = f"{coeff[i]:.2f}"

            # Add x
            coeff_str = rf"{coeff_str}$x^{poly_order-i}$"

            # Add sign
            if coeff[i] > 0:
                coeff_str = f"+{coeff_str}"
                legend_str += coeff_str
            else:
                legend_str += coeff_str
        return legend_str

    if print_func:
        print(poly_func_b0)
        print(poly_func_b1)
        print(poly_func_b2)

    fig, axes = plt.subplots(1, 3, figsize=(38, 10))
    hep.hist2dplot(h_twc1, ax=axes[0], norm=colors.LogNorm())
    hep.cms.text(loc=0, ax=axes[0], text="ETL ETROC Test Beam", fontsize=18)
    axes[0].plot(b1_xrange, poly_func_b0(b1_xrange), 'r-', lw=3, label=make_legend(coeff_b0, poly_order=poly_order))
    axes[0].set_xlabel('TOT1 [ps]', fontsize=25)
    axes[0].set_ylabel('0.5*(TOA2+TOA3)-TOA1 [ps]', fontsize=25)
    axes[0].set_title(plot_title+' Board 1', fontsize=16, loc='right')
    hep.hist2dplot(h_twc2, ax=axes[1], norm=colors.LogNorm())
    hep.cms.text(loc=0, ax=axes[1], text="ETL ETROC Test Beam", fontsize=18)
    axes[1].plot(b2_xrange, poly_func_b1(b2_xrange), 'r-', lw=3, label=make_legend(coeff_b1, poly_order=poly_order))
    axes[1].set_xlabel('TOT2 [ps]', fontsize=25)
    axes[1].set_ylabel('0.5*(TOA1+TOA3)-TOA2 [ps]', fontsize=25)
    axes[1].set_title(plot_title+' Board 2', fontsize=16, loc='right')
    hep.hist2dplot(h_twc3, ax=axes[2], norm=colors.LogNorm())
    hep.cms.text(loc=0, ax=axes[2], text="ETL ETROC Test Beam", fontsize=18)
    axes[2].plot(b3_xrange, poly_func_b2(b3_xrange), 'r-', lw=3, label=make_legend(coeff_b2, poly_order=poly_order))
    axes[2].set_xlabel('TOT3 [ps]', fontsize=25)
    axes[2].set_ylabel('0.5*(TOA1+TOA2)-TOA3 [ps]', fontsize=25)
    axes[2].set_title(plot_title+' Board 3', fontsize=16, loc='right')

    axes[0].legend(loc='best')
    axes[1].legend(loc='best')
    axes[2].legend(loc='best')

    axes[0].set_xlim(2000,6000)
    axes[1].set_xlim(2500,5500)
    axes[2].set_xlim(2500,7000)

    axes[0].set_ylim(-3000,3000)
    axes[1].set_ylim(-3000,3000)
    axes[2].set_ylim(-3000,3000)

    if filename is not None:
        fig.savefig(f'{filename}.png')
        plt.close(fig)


## --------------------------------------
# main execution block
if __name__ == "__main__": 
    import argparse
    # arguments

    parser = argparse.ArgumentParser(
                prog='bootstrap',
                description='find time resolution!',
            )

    parser.add_argument(
        '-f',
        '--file',
        metavar = 'PATH',
        type = str,
        help = 'pickle file with tdc data based on selected track',
        required = False,
        default = 'None',
        dest = 'file'
    )

    parser.add_argument(
        '-i',
        '--iteration',
        metavar = 'ITERATION',
        type = int,
        help = 'Number of bootstrapping',
        default = 100,
        dest = 'iteration',
    )

    parser.add_argument(
        '-s',
        '--sampling',
        metavar = 'SAMPLING',
        type = int,
        help = 'Random sampling fraction',
        default = 75,
        dest = 'sampling',
    )

    parser.add_argument(
        '--board_ids',
        metavar='N',
        type=int,
        nargs='+',
        help='board IDs to analyze'
    )

    parser.add_argument(
        '--board_id_for_TOA_cut',
        metavar = 'NUM',
        type = int,
        help = 'TOA range cut will be applied to a given board ID',
        default = 1,
        dest = 'board_id_for_TOA_cut',
    )

    parser.add_argument(
        '--minimum_nevt',
        metavar = 'NUM',
        type = int,
        help = 'Minimum number of events for bootstrap',
        default = 1000,
        dest = 'minimum_nevt',
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
        '--board_id_rfsel0',
        metavar = 'NUM',
        type = int,
        help = 'board ID that set to RfSel = 0',
        default = -1,
        dest = 'board_id_rfsel0',
    )

    parser.add_argument(
        '--autoTOTcuts',
        action = 'store_true',
        help = 'If set, select 80 percent of data around TOT median value of each board',
        dest = 'autoTOTcuts',
    )

    parser.add_argument(
        '--reproducible',
        action = 'store_true',
        help = 'If set, random seed will be set by counter and save random seed in the final output',
        dest = 'reproducible',
    )

    parser.add_argument(
        '--time_df_input',
        action = 'store_true',
        help = 'If set, time_df_bootstrap function will be used',
        dest = 'time_df_input',
    )

    parser.add_argument(
        '--runs_to_analyze',
        metavar='N',
        type=int,
        nargs='+',
        help='array of runs to analyze',
        dest = 'runs'
    )

    parser.add_argument(
        '-o',
        '--output_name',
        type = str,
        help = 'name of the output data file',
        dest='output_name'
    )

    parser.add_argument(
        '--poly_order',
        type = int,
        metavar = 'NUM',
        default = 2,
        help = 'order of the TWC fit function',
        dest = 'poly_order'
    )

    parser.add_argument(
        '--output_plot',
        action = 'store_true',
        help = 'If set, plot of the TWC fits will be saved',
        dest = 'plotting',
    )

    args = parser.parse_args()

    #read input file if using only 1 file input
    if args.file != 'None':
        output_name = args.file.split('.')[0]
        df = pd.read_pickle(args.file)

    #check if board setup matches input
    board_ids = args.board_ids
    if len(board_ids) != 3:
        print('Please double check inputs. It should be e.g. 0 1 2 or 1 2 3')
        sys.exit()

    #not used for this analysis
    if not args.time_df_input:
        df = df.reset_index(names='evt')
        tot_cuts = {}
        #tot cuts
        for idx in board_ids:
            if args.autoTOTcuts:
                lower_bound = df['tot'][idx].quantile(0.01)
                upper_bound = df['tot'][idx].quantile(0.96)
                tot_cuts[idx] = [round(lower_bound), round(upper_bound)]

                if idx == args.board_id_rfsel0:
                    condition = df['tot'][idx] < 470
                    lower_bound = df['tot'][idx][condition].quantile(0.07)
                    upper_bound = df['tot'][idx][condition].quantile(0.98)
                    tot_cuts[idx] = [round(lower_bound), round(upper_bound)]

            else:
                tot_cuts[idx] = [0, 600]

        print(f'TOT cuts: {tot_cuts}')

        ## Selecting good hits with TDC cuts
        tdc_cuts = {}
        for idx in board_ids:
            if idx == args.board_id_for_TOA_cut:
                tdc_cuts[idx] = [0, 1100, args.trigTOALower, args.trigTOAUpper, tot_cuts[idx][0], tot_cuts[idx][1]]
            else:
                tdc_cuts[idx] = [0, 1100, 0, 1100, tot_cuts[idx][0], tot_cuts[idx][1]]

        #event selection pivot
        interest_df = tdc_event_selection_pivot(df, tdc_cuts_dict=tdc_cuts)
        print('Size of dataframe after TDC cut:', interest_df.shape[0])

        #bootstrap function execution
        resolution_df = bootstrap(input_df=interest_df, board_to_analyze=board_ids, iteration=args.iteration,
                                sampling_fraction=args.sampling, minimum_nevt_cut=args.minimum_nevt, do_reproducible=args.reproducible)
    #main bootsrap execution block
    else:
        # initial parameter setup
        RunNums = args.runs
        output_name = args.output_name
        poly_order = args.poly_order
        timepath = {}
        poly_fits = {}
        blank_list = []
        track_selected = False
        boardscan = True

        
        # if scaning a single board by each pixel
        RCids = 1
        if boardscan == True:
            RCids = 16
        refpix = 'R9C9'

        # create data dictionaries for resolutions and coefficients
        resolution_ratio = {}
        resolution_delta = {}
        resolution_normal = {}
        resolution_mod = {}
        coeffs = {}
        for RunNum in RunNums:
            for board in range(1,4):
                for Row in range(RCids):
                    for Column in range(RCids):
                        # "type_runnum_board_row_column"
                        resolution_ratio[f'ratio_{RunNum}_{board}_{Row}_{Column}']=[]
                        resolution_delta[f'delta_{RunNum}_{board}_{Row}_{Column}']=[]
                        resolution_normal[f'normal_{RunNum}_{board}_{Row}_{Column}']=[]
                        resolution_mod[f'mod_{RunNum}_{board}_{Row}_{Column}']=[]

                        for i in range(2):
                            for order in range(poly_order+1):
                                # "itteration_row_column"
                                coeffs[f'{RunNum}_{board}_{i}_{order}_{Row}_{Column}'] = []

        # create array of timecodes if multiple tests of a run exist
        for RunNum in RunNums:
            for dirpath, dirnames, filenames in os.walk(f'./test_outputs/run_{RunNum}'):
                if dirnames != [] and dirnames != ['tracks','time']:
                    timepath[RunNum] = dirnames
                    timepath[RunNum].sort()

        for RunNum in tqdm(RunNums):
            # loop through each run

            # dictionary for resolution calculations
            # wiped after every run
            resolution_individual = {}
            resolution_grouped = {}
            for board in range (1,4):
                for Row in range(RCids):
                    for Column in range(RCids):
                        # "board_row_column"
                        resolution_individual[f'{RunNum}_{board}_{Row}_{Column}'] = []
                        resolution_grouped[f'{RunNum}_{board}_{Row}_{Column}'] = []
            
            for times in range(len(timepath[RunNum])):
                # loop through each time block within a run
                timenow = timepath[RunNum][times]
                # filter which blocks will be read
                if timenow.startswith('Sep'):
                    continue
                # print(f'RunNum: {RunNum}')
                # print(f'times: {times}')
                # print(f'timenow: {timenow}')

                directory = f"./test_outputs/run_{RunNum}/{timenow}/time/"

                # loop across pixels on a board
                for Row in range(RCids):
                    for Column in range(RCids):
                        RC = f'R{Row}C{Column}'
                        # for track name splicing
                        upperval = 17
                        if Row >= 12:
                            upperval = 18

                        # if running run analysis
                        if boardscan == False:
                            if track_selected == False:
                                # finds largest track file
                                largest_track_file = max(
                                (os.path.join(root, file) for root, dirs, files in os.walk(directory) for file in files),
                                key=os.path.getsize
                                )
                                track_file = largest_track_file.removesuffix('.pkl').removeprefix(directory).removeprefix('track_')
                                while not track_file.startswith('R'):
                                        track_file = track_file[1:]
                                print(f'Track used: {track_file}')
                                track_selected = True
                            
                            track_to_bootstrap(directory=directory, track=track_file, RunNum=RunNum, RCid=f'{Row}_{Column}',output_name=output_name, poly_order=poly_order)

                        # if running pixel analysis
                        else:                    
                            if track_selected == False:
                                try:
                                    # largest track file with desired pixel to reference
                                    largest_track_file = max(
                                    (os.path.join(root, file) for root, dirs, files in os.walk(directory) for file in files if f'{refpix}_' in file[0:upperval]),
                                    key=os.path.getsize
                                    )
                                    track_file = largest_track_file.removesuffix('.pkl').removeprefix(directory).removeprefix('track_')
                                    while not track_file.startswith('R'):
                                        track_file = track_file[1:]     
                                    print(f'Track used: {track_file}')
                                    track_selected = True

                                    track_to_bootstrap(directory=directory, track=track_file, RunNum=RunNum, RCid='9_9', output_name=output_name, poly_order=poly_order)

                                except Exception as inst:
                                    print('No reference track with name R9C9 on first board')
                                    print('Pick a different reference pixel')
                                    print(f'Exception: {inst}')
                                    exit()
                            
                            # skip reference pixel, dont want to run it twice
                            if RC == refpix:
                                continue

                            # run bootstrap for each pixel on a board
                            try:
                                largest_track_file = max(
                                (os.path.join(root, file) for root, dirs, files in os.walk(directory) for file in files if f'{RC}_' in file[0:upperval]),
                                key=os.path.getsize
                                )
                                track_file = largest_track_file.removesuffix('.pkl').removeprefix(directory).removeprefix('track_')
                                while not track_file.startswith('R'):
                                    track_file = track_file[1:]
                                print(f'Track used: {track_file}')
                            
                            except:
                                print(f'No track file with name {RC} on first board')
                                # fill in empty spots in dicts with blank values
                                for board in range(1,4):
                                    resolution_individual[f'{RunNum}_{board}_{Row}_{Column}'] = [0.0]*100
                                    resolution_grouped[f'{RunNum}_{board}_{Row}_{Column}'] = [0.0]*100
                                    resolution_normal[f'normal_{RunNum}_{board}_{Row}_{Column}'] = [0.0]*100
                                    resolution_mod[f'mod_{RunNum}_{board}_{Row}_{Column}'] = [0.0]*100
                                    for i in range(2):
                                        for order in range(poly_order+1):
                                            coeffs[f'{RunNum}_{board}_{i}_{order}_{Row}_{Column}'] = [-1000.0]*100
                                blank_list.append(RC)
                                continue

                            track_to_bootstrap(directory=directory, track=track_file, RunNum=RunNum, RCid=f'{Row}_{Column}', output_name=output_name, poly_order=poly_order)
                            
                            # for board in range(1,4):
                            #     for i in range(2):
                            #         for order in range(poly_order):
                            #             print(f'{RunNum}_{board}_{i}_{order}_{Row}_{Column}')
                            #             print(len(coeffs[f'{RunNum}_{board}_{i}_{order}_{Row}_{Column}']))

            # calculating ratio and delta
            for Row in range(RCids):
                for Column in range(RCids):
                    for board in range(1,4):
                        idstring = f'{RunNum}_{board}_{Row}_{Column}'
                        for i in range(len(resolution_grouped[idstring])):
                            try:
                                resolution_ratio[f'ratio_{idstring}'].append(resolution_grouped[idstring][i]/resolution_individual[idstring][i])
                                resolution_delta[f'delta_{idstring}'].append(resolution_grouped[idstring][i]-resolution_individual[idstring][i])
                            except:
                                resolution_ratio[f'ratio_{idstring}'].append(0.0)
                                resolution_delta[f'delta_{idstring}'].append(-10.0)

    # saves data in output file
    save_data(output_name=output_name)