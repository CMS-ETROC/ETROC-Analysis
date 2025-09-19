import argparse, yaml
import pandas as pd
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from natsort import natsorted
from pathlib import Path
from tqdm import tqdm

## --------------------------------------
def three_board_iterative_timewalk_correction(
    input_df: pd.DataFrame,
    iterative_cnt: int,
    poly_order: int,
    board_roles: list[str],
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
            # CHANGED: Logic now depends on the boolean flag
            coeff = np.polyfit(tots[key], delta_toas[key], poly_order)

            poly_func = np.poly1d(coeff)
            corrections[key] = poly_func(tots[key])

        for key in board_roles:
            corr_toas[key] += corrections[key]
        delta_toas = _calculate_deltas(corr_toas)

    return corr_toas

## --------------------------------------
def find_optimal_gmm(input_data: np.array):
    from sklearn.mixture import GaussianMixture

    data = input_data.reshape(-1, 1)
    best_gmm = GaussianMixture(n_components=3).fit(data)

    return best_gmm


## --------------------------------------
def fwhm_based_on_gaussian_mixture_model(
        input_data: np.array,
    ):

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

    # --- Use the new function to find the optimal GMM ---
    models = find_optimal_gmm(input_data)

    # Calculate the Jensen-Shannon distance with the optimal model
    logprob = models.score_samples(centers.reshape(-1, 1))
    pdf = np.exp(logprob)
    jensenshannon_score = distance.jensenshannon(bins, pdf)

    # Calculate the FWHM
    logprob = models.score_samples(x_range)
    pdf = np.exp(logprob)
    peak_height = np.max(pdf)
    half_max = peak_height * 0.5
    half_max_indices = np.where(pdf >= half_max)[0]
    fwhm = x_range[half_max_indices[-1]] - x_range[half_max_indices[0]]

    return fwhm, jensenshannon_score

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
        input_file: Path,
        args: argparse,
        twc_coeffs: dict,
        minimum_nevt_cut: int = 1000,
    ):

    input_df = pd.read_pickle(input_file)
    excluded_role = input_file.name.split('_')[1]
    output_name = input_file.name.split('.')[0]
    all_roles = {'trig', 'dut', 'ref', 'extra'}
    board_to_analyze = sorted(all_roles - {excluded_role})

    ### Determine GMM quality cut based on the given sample size
    ### This is data-driven by separate study
    ### Log equation + margin
    gmm_quality_cut = (-0.0148 * np.log(input_df.shape[0]) + 0.1842) + 0.002

    ### Avoid to restrict cut
    if gmm_quality_cut < 0.05:
        gmm_quality_cut = 0.05

    if input_df.shape[0] < minimum_nevt_cut:
        exit()

    corr_toas = three_board_iterative_timewalk_correction(
        input_df, 2, 2,
        board_roles=board_to_analyze,
        twc_coeffs=twc_coeffs,
    )

    diffs = {}
    for board_a in board_to_analyze:
        for board_b in board_to_analyze:
            if board_b <= board_a:
                continue
            name = f"{board_a}-{board_b}"
            diffs[name] = corr_toas[board_a] - corr_toas[board_b]

    resolution_from_bootstrap = defaultdict(list)
    while True:

        fit_params = {}
        js_scores = {}

        for ikey in diffs.keys():
            params, jensenshannon_score = fwhm_based_on_gaussian_mixture_model(diffs[ikey])
            ## jensenshannon_score means how overall shape matches between data and fit

            fit_params[ikey] = float(params[0]/2.355)
            js_scores[ikey] = jensenshannon_score

        # --- Check for failure AFTER all fits are done ---
        gmm_failed = False

        for ikey in js_scores.keys():
            if js_scores[ikey] > gmm_quality_cut:
                gmm_failed = True
                break

        if gmm_failed:
            continue

        else:
            resolutions = return_resolution_three_board_fromFWHM(fit_params, board_roles=board_to_analyze)

            if any(val <= 0 for val in resolutions.values()):
                continue

            for key in resolutions.keys():
                resolution_from_bootstrap[key].append(resolutions[key])

        if resolution_from_bootstrap:
            break

    final_result = pd.DataFrame(resolution_from_bootstrap)
    final_result.to_pickle(Path(args.outdir) / f'{output_name}_single.pkl')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Apply cuts to track files and save final output.")

    parser.add_argument(
        '-d',
        '--inputdir',
        metavar = 'INPUTNAME',
        type = str,
        help = 'input directory name',
        required = True,
        dest = 'inputdir',
    )

    parser.add_argument(
        '-o',
        '--outdir',
        metavar = 'OUTNAME',
        type = str,
        help = 'output directory path',
        required = True,
        dest = 'outdir',
    )

    args = parser.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    files = natsorted(Path(args.inputdir).glob('exclude_*.pkl'))

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(time_df_bootstrap, f, args) for f in files]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Single Time Resolution analysis"):
            try:
                # This is the crucial line. It will re-raise any exception
                # that happened in the worker process.
                future.result()
            except Exception as exc:
                print(f"A worker process generated an exception: {exc}")
                # For a full error report, uncomment the next two lines
                # import traceback
                # traceback.print_exc()
            finally:
                pass