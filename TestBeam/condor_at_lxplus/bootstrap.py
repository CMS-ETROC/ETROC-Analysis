import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

## --------------------------------------
def three_board_iterative_timewalk_correction(
    input_df: pd.DataFrame,
    iterative_cnt: int,
    poly_order: int,
    board_roles: list[str],
    twc_coeffs: dict,
):

    if len(board_roles) != 3:
        raise ValueError(f"This function's logic requires exactly 3 boards, but {len(board_roles)} were provided.")

    tots = {key: input_df[f'tot_{key}'].values for key in board_roles}
    corr_toas = {key: input_df[f'toa_{key}'].values for key in board_roles}

    def _calculate_deltas(current_toas):
        deltas = {}
        for current_key in board_roles:
            others_sum = sum(current_toas[other_key] for other_key in board_roles if other_key != current_key)
            deltas[current_key] = (0.5 * others_sum) - current_toas[current_key]
        return deltas

    delta_toas = _calculate_deltas(corr_toas)

    for i in range(iterative_cnt):
        corrections = {}

        for key in board_roles:

            if twc_coeffs is not None:
                coeff = twc_coeffs[f'iter{i+1}'][key]
            else:
                coeff = np.polyfit(tots[key], delta_toas[key], poly_order)

            poly_func = np.poly1d(coeff)
            corrections[key] = poly_func(tots[key])

        for key in board_roles:
            corr_toas[key] += corrections[key]

        delta_toas = _calculate_deltas(corr_toas)

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
        import matplotlib.pyplot as plt
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
        board_roles: list[str],
    ):

    if len(board_roles) != 3:
        raise ValueError(f"This function's logic requires exactly 3 boards, but {len(board_roles)} were provided.")

    def get_param_value(board1: str, board2: str):
        key1 = f"{board1}-{board2}"
        key2 = f"{board2}-{board1}"
        if key1 in fit_params:
            return fit_params[key1]
        if key2 in fit_params:
            return fit_params[key2]
        raise KeyError(f"Could not find resolution pair for '{board1}' and '{board2}' in fit_params.")

    results = {}
    for target_board in board_roles:
        other_boards = [b for b in board_roles if b != target_board]

        term1 = get_param_value(target_board, other_boards[0]) ** 2
        term2 = get_param_value(target_board, other_boards[1]) ** 2
        term3 = get_param_value(other_boards[0], other_boards[1]) ** 2

        # Apply the generalized formula
        # np.sqrt can return a nan if the value is negative, which indicates non-physical results
        with np.errstate(invalid='warn'): # Will warn if you take sqrt of a negative number
            resolution_sq = 0.5 * (term1 + term2 - term3)
            results[target_board] = np.sqrt(resolution_sq) if resolution_sq > 0 else 0

    return results

## --------------------------------------
def time_df_bootstrap(
        input_df: pd.DataFrame,
        board_to_analyze: list[str],
        twc_coeffs: dict,
        limit: int = 7500,
        nouts: int = 100,
        sampling_fraction: int = 75,
        do_reproducible: bool = False,
    ):
    resolution_from_bootstrap = defaultdict(list)
    random_sampling_fraction = sampling_fraction*0.01

    counter = 0
    resample_counter = 0
    successful_runs = 0

    while True:

        if counter > limit:
            print("Loop reaches the limit. Escaping bootstrap loop")
            break

        if do_reproducible:
            np.random.seed(counter)

        n = int(random_sampling_fraction*input_df.shape[0])
        indices = np.random.choice(input_df['evt'].unique(), n, replace=False)
        selected_df = input_df.loc[input_df['evt'].isin(indices)]

        corr_toas = three_board_iterative_timewalk_correction(selected_df, 2, 2, board_roles=board_to_analyze, twc_coeffs=twc_coeffs)

        diffs = {}
        for board_a in board_to_analyze:
            for board_b in board_to_analyze:
                if board_b <= board_a:
                    continue
                name = f"{board_a}-{board_b}"
                diffs[name] = corr_toas[board_a] - corr_toas[board_b]

        try:
            fit_params = {}
            scores = []
            for ikey in diffs.keys():
                params, eval_scores = fwhm_based_on_gaussian_mixture_model(diffs[ikey], n_components=3, plotting=False, plotting_each_component=False)
                fit_params[ikey] = float(params[0]/2.355)
                scores.append(eval_scores)

            if np.any(np.asarray(scores)[:,0] > 0.6) or np.any(np.asarray(scores)[:,1] > 0.075) :
                print('The result does not pass a fit evaluation cut. Redo the sampling')
                counter += 1
                resample_counter += 1
                continue

            resolutions = return_resolution_three_board_fromFWHM(fit_params, board_roles=board_to_analyze)

            if any(val <= 0 for val in resolutions.values()):
                print('At least one time resolution value is zero or non-physical. Skipping this iteration')
                counter += 1
                resample_counter += 1
                continue

            for key in resolutions.keys():
                resolution_from_bootstrap[key].append(resolutions[key])

            if do_reproducible:
                resolution_from_bootstrap['RandomSeed'].append(counter)

            counter += 1
            successful_runs += 1

        except Exception as inst:
            print(f"An error occurred during fitting: {inst}. Skipping this iteration.")
            counter += 1
            resample_counter += 1
            del diffs, corr_toas

        print(f"{successful_runs} / {nouts}")
        if successful_runs >= nouts:
            print(f'Collected {nouts} successful runs. Escaping bootstrap loop.')
            break

    print(f'\nTotal iterations: {counter}, Resampled/Skipped: {resample_counter}, Successful: {successful_runs}')

    ### Empty dictionary case
    if not resolution_from_bootstrap:
        return pd.DataFrame()
    else:
        return pd.DataFrame(resolution_from_bootstrap)

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
        '--minimum_nevt',
        metavar = 'NUM',
        type = int,
        help = 'Minimum number of events for bootstrap',
        default = 1000,
        dest = 'minimum_nevt',
    )

    parser.add_argument(
        '--twc_coeffs',
        metavar = 'FILE',
        type = str,
        help = 'pre-calculated TWC coefficients, it has to be pickle file',
        dest = 'twc_coeffs',
    )

    parser.add_argument(
        '--reproducible',
        action = 'store_true',
        help = 'If set, random seed will be set by counter and save random seed in the final output',
        dest = 'reproducible',
    )

    args = parser.parse_args()

    output_name = args.file.split('/')[-1].split('.')[0]
    track_name = args.file.split('/')[-1].split('.')[0].split('track_')[1]
    excluded_role = args.file.split('/')[-1].split('_')[1]
    all_roles = {'trig', 'dut', 'ref', 'extra'}
    board_roles = sorted(all_roles - {excluded_role})

    if args.twc_coeffs is not None:
        import pickle
        with open(args.twc_coeffs, 'rb') as input_coeff:
            calculated_twc_coeffs = pickle.load(input_coeff)[track_name]

            # A slightly more efficient and readable version
            coeff_keys = sorted(calculated_twc_coeffs['iter1'].keys())

            if coeff_keys != board_roles:
                print(f"Keys from TWC coefficient file: {coeff_keys}")
                print(f'Board roles for current run: {board_roles}')
                raise KeyError('Board roles in the provided TWC coefficient file do not match the current run.')
    else:
        calculated_twc_coeffs = None

    df = pd.read_pickle(args.file)
    df = df.reset_index(names='evt')

    if df.shape[0] < args.minimum_nevt:
        print(f'Number of events in the sample is {df.shape[0]}')
        print('Warning!! Sampling size is too small. Bootstrap will not happen for this track')
        exit()

    resolution_df = time_df_bootstrap(input_df=df, board_to_analyze=board_roles, twc_coeffs=calculated_twc_coeffs, limit=args.iteration_limit, nouts=args.num_bootstrap_output,
                                        sampling_fraction=args.sampling, do_reproducible=args.reproducible)

    if not resolution_df.empty:
        resolution_df.to_pickle(f'{output_name}_resolution.pkl')
    else:
        print(f'With {args.sampling}% sampling, number of events in sample is not enough to do bootstrap')