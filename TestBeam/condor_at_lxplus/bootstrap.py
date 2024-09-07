import pandas as pd
import numpy as np
import sys
from collections import defaultdict
import warnings
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
):

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

        coeff_b1 = np.polyfit(input_df[f'tot_b{board_list[1]}'].values, del_toa_b1, poly_order)
        poly_func_b1 = np.poly1d(coeff_b1)

        coeff_b2 = np.polyfit(input_df[f'tot_b{board_list[2]}'].values, del_toa_b2, poly_order)
        poly_func_b2 = np.poly1d(coeff_b2)

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
    models = GaussianMixture(n_components=n_components).fit(input_data.reshape(-1, 1))
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
        limit: int = 7500,
        nouts: int = 100,
        sampling_fraction: int = 75,
        minimum_nevt_cut: int = 1000,
        do_reproducible: bool = False,
    ):

    resolution_from_bootstrap = defaultdict(list)
    random_sampling_fraction = sampling_fraction*0.01

    counter = 0
    resample_counter = 0

    while True:

        if counter > limit:
            print("Loop reaches the limit. Escaping bootstrap loop")
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
            if len(val) > nouts:
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
        limit: int = 7500,
        nouts: int = 100,
        sampling_fraction: int = 75,
        minimum_nevt_cut: int = 1000,
        do_reproducible: bool = False,
    ):
    resolution_from_bootstrap = defaultdict(list)
    random_sampling_fraction = sampling_fraction*0.01

    counter = 0
    resample_counter = 0

    while True:

        if counter > limit:
            print("Loop reaches the limit. Escaping bootstrap loop")
            break

        if do_reproducible:
            np.random.seed(counter)

        n = int(random_sampling_fraction*input_df.shape[0])
        indices = np.random.choice(input_df['evt'].unique(), n, replace=False)
        selected_df = input_df.loc[input_df['evt'].isin(indices)]

        if selected_df.shape[0] < minimum_nevt_cut:
            print(f'Number of events in random sample is {selected_df.shape[0]}')
            print('Warning!! Sampling size is too small. Skipping this track')
            break

        if(len(board_to_analyze)==3):
            corr_toas = three_board_iterative_timewalk_correction(selected_df, 2, 2, board_list=board_to_analyze)
        elif(len(board_to_analyze)==4):
            corr_toas = four_board_iterative_timewalk_correction(selected_df, 2, 2)
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
            if len(val) > nouts:
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
if __name__ == "__main__":
    import argparse

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
        required = True,
        dest = 'file',
    )

    parser.add_argument(
        '-n',
        '--num_bootstrap_output',
        metavar = 'NUM',
        type = int,
        help = 'Number of outputs after bootstrap',
        default = 100,
        dest = 'num_bootstrap_output',
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
        '--iteration_limit',
        metavar = 'NUM',
        type = int,
        help = 'Maximum iteration of sampling',
        default = 7500,
        dest = 'iteration_limit',
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

    args = parser.parse_args()

    output_name = args.file.split('.')[0]
    df = pd.read_pickle(args.file)

    board_ids = args.board_ids
    if len(board_ids) != 3:
        print('Please double check inputs. It should be e.g. 0 1 2 or 1 2 3')
        sys.exit()

    if not args.time_df_input:
        df = df.reset_index(names='evt')
        tot_cuts = {}
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

        interest_df = tdc_event_selection_pivot(df, tdc_cuts_dict=tdc_cuts)
        print('Size of dataframe after TDC cut:', interest_df.shape[0])

        resolution_df = bootstrap(input_df=interest_df, board_to_analyze=board_ids, limit=args.iteration_limit, nouts=args.num_bootstrap_output,
                                sampling_fraction=args.sampling, minimum_nevt_cut=args.minimum_nevt, do_reproducible=args.reproducible)
    else:
        df = df.reset_index(names='evt')
        resolution_df = time_df_bootstrap(input_df=df, board_to_analyze=board_ids, limit=args.iteration_limit, nouts=args.num_bootstrap_output,
                                          sampling_fraction=args.sampling, minimum_nevt_cut=args.minimum_nevt, do_reproducible=args.reproducible)


    if not resolution_df.empty:
        resolution_df.to_pickle(f'{output_name}_resolution.pkl')
    else:
        print(f'With {args.sampling}% sampling, number of events in sample is not enough to do bootstrap')
