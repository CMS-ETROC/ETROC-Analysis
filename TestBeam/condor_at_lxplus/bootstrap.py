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
    twc_coeffs: dict = None,
    use_precomputed_coeffs: bool = False, # CHANGED: Added flag
):

    if len(board_roles) != 3:
        raise ValueError(f"This function's logic requires exactly 3 boards, but {len(board_roles)} were provided.")

    ## FIXED: Removed redundant size check. This logic is now handled in the calling function.
    if use_precomputed_coeffs and twc_coeffs is None:
        raise ValueError("Request to use pre-computed coefficients was made, but 'twc_coeffs' is None.")

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
            # CHANGED: Logic now depends on the boolean flag
            if use_precomputed_coeffs:
                if twc_coeffs is None:
                    raise ValueError("Cannot use pre-computed coefficients because 'twc_coeffs' is None.")
                try:
                    coeff = twc_coeffs[f'iter{i+1}'][key]
                except KeyError:
                    raise KeyError(f"Missing coefficients for iteration {i+1}, board '{key}' in 'twc_coeffs'.")
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
    ):

    from sklearn.mixture import GaussianMixture
    from scipy.spatial import distance

    ### Find the best number of bins
    n = len(input_data)
    iqr = np.percentile(input_data, 75) - np.percentile(input_data, 25)

    if iqr == 0:
        print("IQR is zero. Cannot apply Freedman-Diaconis rule.")
        hist_bins = 200
    else:
        bin_width = 2 * iqr / (n**(1/3))
        hist_bins = int(np.ceil((input_data.max() - input_data.min()) / bin_width))

        if hist_bins > 200:
            hist_bins = 200
        elif hist_bins < 30:
            hist_bins = 30

    x_range = np.linspace(input_data.min(), input_data.max(), 1000).reshape(-1, 1)
    bins, edges = np.histogram(input_data, bins=hist_bins, density=True)
    centers = 0.5*(edges[1:] + edges[:-1])
    models = GaussianMixture(n_components=n_components).fit(input_data.reshape(-1, 1))

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
    return fwhm, jensenshannon_score

## --------------------------------------
def get_optimal_bins(data: np.array):
    """Calculates the optimal number of histogram bins using the Freedman-Diaconis rule."""
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return int(np.sqrt(len(data)))
    bin_width = 2 * iqr * (len(data) ** (-1/3))
    if bin_width == 0:
        return int(np.sqrt(len(data)))
    data_range = np.max(data) - np.min(data)
    num_bins = int(data_range / bin_width)
    return num_bins

## --------------------------------------
def fwhm_from_histogram(input_data: np.array, bins: int):
    """Calculates FWHM directly from a histogram by finding the peak and interpolating."""
    from scipy.interpolate import interp1d
    counts, bin_edges = np.histogram(input_data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    peak_height = np.max(counts)
    peak_index = np.argmax(counts)
    half_max = peak_height / 2.0
    try:
        f_left = interp1d(counts[:peak_index+1], bin_centers[:peak_index+1])
        f_right = interp1d(counts[peak_index:], bin_centers[peak_index:])
        x_left = f_left(half_max)
        x_right = f_right(half_max)
        return float(x_right - x_left)
    except ValueError:
        return None

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
        minimum_nevt_cut: int = 1000,
        do_reproducible: bool = False,
        force_precomputed_coeffs: bool = False,
    ):
    resolution_from_bootstrap = defaultdict(list)

    # --- NEW: Initialize state variables ---
    current_sampling_fraction = sampling_fraction
    consecutive_failures = 0
    failure_threshold = int(limit * 0.02) # Consecutive failure threshold set to 2%

    counter = 0
    successful_runs = 0

    while True:

        if counter > limit:
            print("Loop reaches the limit. Escaping bootstrap loop")
            break

        if do_reproducible:
            np.random.seed(counter)

        n = int(0.01 * current_sampling_fraction * input_df.shape[0])
        indices = np.random.choice(input_df.index, n, replace=False)
        selected_df = input_df.loc[indices]

        use_coeffs_for_this_iteration = False
        if force_precomputed_coeffs:
            use_coeffs_for_this_iteration = True
        elif selected_df.shape[0] < minimum_nevt_cut:
            print(f"\nINFO: Subsample is small ({selected_df.shape[0]} rows). Forcing use of pre-computed TWC.", end="")
            use_coeffs_for_this_iteration = True

        corr_toas = three_board_iterative_timewalk_correction(
            selected_df, 2, 2,
            board_roles=board_to_analyze,
            twc_coeffs=twc_coeffs,
            use_precomputed_coeffs=use_coeffs_for_this_iteration,
        )

        diffs = {}
        for board_a in board_to_analyze:
            for board_b in board_to_analyze:
                if board_b <= board_a:
                    continue
                name = f"{board_a}-{board_b}"
                diffs[name] = corr_toas[board_a] - corr_toas[board_b]

        try:
            fit_params = {}
            gmm_failed = False
            for ikey in diffs.keys():
                params, jensenshannon_score = fwhm_based_on_gaussian_mixture_model(diffs[ikey], n_components=3)
                ## jensenshannon_score means how overall shape matches between data and fit

                # Check GMM quality
                if jensenshannon_score > 0.05:
                    gmm_failed = True
                    break # A failure in any fit invalidates this iteration

                fit_params[ikey] = float(params[0]/2.355)

            # If GMM failed, handle it according to the new strategy
            if gmm_failed:

                 # First, check if the sampling rate is already high. If so, move to fallback immediately.
                if current_sampling_fraction > 90:
                    print("\n--- STRATEGY: GMM failed at a high sampling rate. Giving up on bootstrap. ---")
                    print("--- Performing a single calculation on the full dataset using the histogram method. ---")

                    # 1. Use the FULL original dataframe for the calculation
                    full_corr_toas = three_board_iterative_timewalk_correction(
                        input_df, 2, 2,
                        board_roles=board_to_analyze,
                        twc_coeffs=twc_coeffs,
                        use_precomputed_coeffs=force_precomputed_coeffs
                    )

                    full_diffs = {}
                    for board_a in board_to_analyze:
                        for board_b in board_to_analyze:
                            if board_b <= board_a: continue
                            name = f"{board_a}-{board_b}"
                            full_diffs[name] = full_corr_toas[board_a] - full_corr_toas[board_b]

                    # 2. Try the histogram method first
                    fallback_params = {}
                    for ikey, data in full_diffs.items():
                        bins = get_optimal_bins(data)
                        fwhm = fwhm_from_histogram(data, bins=bins)
                        fallback_params[ikey] = fwhm / 2.355

                    final_resolution = return_resolution_three_board_fromFWHM(fallback_params, board_roles=board_to_analyze)
                    print(f"Final calculated resolution (FWHM method): {final_resolution}")

                    # 3. Check if the histogram result is physical. If not, try the "last resort" method.
                    if any(val <= 0 for val in final_resolution.values()):
                        print("--- Histogram method yielded non-physical results. Trying last resort: simple standard deviation. ---")

                        std_params = {}
                        for ikey, data in full_diffs.items():
                            std_params[ikey] = np.std(data)

                        # This is the absolute final calculation.
                        last_resort_resolution = return_resolution_three_board_fromFWHM(std_params, board_roles=board_to_analyze)
                        print(f"Final calculated resolution (STD method): {last_resort_resolution}")
                        return pd.DataFrame([last_resort_resolution])

                    # Return the single, valid result from the histogram method
                    return pd.DataFrame([final_resolution])

                # If the rate is NOT high, proceed with the normal consecutive failure logic
                else:
                    consecutive_failures += 1
                    print(f"GMM quality cut failed. Consecutive failures: {consecutive_failures}. Total run: {counter}")

                    if consecutive_failures >= failure_threshold:
                        current_sampling_fraction = min(95, current_sampling_fraction + 10)
                        print(f"--- STRATEGY: Increasing sampling rate to {current_sampling_fraction}% ---")
                        consecutive_failures = 0

                # If we haven't met the failure threshold yet, just continue the loop
                counter += 1
                continue

            resolutions = return_resolution_three_board_fromFWHM(fit_params, board_roles=board_to_analyze)

            if any(val <= 0 for val in resolutions.values()):
                print('At least one time resolution value is zero or non-physical. Skipping this iteration')
                counter += 1
                consecutive_failures += 1
                continue

            consecutive_failures = 0

            for key in resolutions.keys():
                resolution_from_bootstrap[key].append(resolutions[key])

            if do_reproducible:
                resolution_from_bootstrap['RandomSeed'].append(counter)

            successful_runs += 1

        except Exception as inst:
            print(f"An error occurred during fitting: {inst}. Skipping this iteration.")
            counter += 1
            del diffs, corr_toas

        counter += 1

        print(f"Success: {successful_runs} / {nouts}")
        if successful_runs >= nouts:
            print(f'Collected {nouts} successful runs. Escaping bootstrap loop.')
            break

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
        help='Minimum number of events to force TWC use',
        default = 100,
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

    parser.add_argument(
        '--force-twc',
        action='store_true',
        help='Force use of provided TWC file for all samples.',
        dest='force_twc'
    )

    args = parser.parse_args()

    output_name = args.file.split('/')[-1].split('.')[0]
    track_name = args.file.split('/')[-1].split('.')[0].split('track_')[1]
    excluded_role = args.file.split('/')[-1].split('_')[1]
    all_roles = {'trig', 'dut', 'ref', 'extra'}
    board_roles = sorted(all_roles - {excluded_role})

    # --- NEW: Conditional logic for loading TWC coefficients ---
    calculated_twc_coeffs = None
    if args.twc_coeffs:
        import pickle
        with open(args.twc_coeffs, 'rb') as input_coeff:
            all_coeffs = pickle.load(input_coeff)

        if args.force_twc:
            # When forcing, select coefficients by the specific track name
            track_name = args.file.split('/')[-1].split('.')[0].split('track_')[1]
            print(f"INFO: --force-twc enabled. Selecting TWC coefficients for track: {track_name}")
            try:
                calculated_twc_coeffs = all_coeffs[track_name]
            except KeyError:
                raise KeyError(f"Track '{track_name}' not found in the provided TWC coefficient file.")
        else:
            # In default mode, use the first available set of coefficients in the file.
            # This set will only be used if/when a small subsample is encountered.
            first_key = next(iter(all_coeffs))
            print(f"INFO: Using first available TWC key ('{first_key}') for potential small-sample corrections.")
            calculated_twc_coeffs = all_coeffs[first_key]

    if args.force_twc and not args.twc_coeffs:
        raise ValueError("--force-twc was used, but no coefficient file was provided via --twc_coeffs.")

    if calculated_twc_coeffs:
        coeff_keys = sorted(calculated_twc_coeffs['iter1'].keys())
        if coeff_keys != board_roles:
            raise KeyError('Board roles in the loaded TWC coefficients do not match the current run.')

    df = pd.read_pickle(args.file)

    resolution_df = time_df_bootstrap(input_df=df, board_to_analyze=board_roles, twc_coeffs=calculated_twc_coeffs, limit=args.iteration_limit,
                                      nouts=args.num_bootstrap_output, sampling_fraction=args.sampling, minimum_nevt_cut=args.minimum_nevt,
                                      do_reproducible=args.reproducible, force_precomputed_coeffs=args.force_twc)

    if not resolution_df.empty:
        resolution_df.to_pickle(f'{output_name}_resolution.pkl')
    else:
        print(f'With {args.sampling}% sampling, number of events in sample is not enough to do bootstrap')
