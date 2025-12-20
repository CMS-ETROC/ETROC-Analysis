import argparse
import sys
import warnings
import logging
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.stats import norm, kstest
from itertools import combinations
from pathlib import Path

# --- Configuration & Logging ---
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Modular Logic: Data Filtering ---

def apply_neighbor_cut(df: pd.DataFrame, requested_cols: list, logic: str):
    """Filters events based on neighbor occupancy logic."""
    neighbor_cols = set(requested_cols)
    if 'none' in neighbor_cols:
        logger.info("No neighbor cut applied.")
        return df

    valid_cols = {'HasNeighbor_dut', 'HasNeighbor_ref', 'HasNeighbor_extra',
                  'HasNeighbor_trig', 'trackNeighbor'}
    cols_to_use = list(neighbor_cols.intersection(valid_cols).intersection(df.columns))

    if not cols_to_use:
        logger.warning("No valid neighbor columns found. Skipping cut.")
        return df

    # Vectorized reduction for speed
    neighbor_matrix = df[cols_to_use].values.astype(bool)
    if logic.upper() == 'OR':
        mask = np.any(neighbor_matrix, axis=1)
    elif logic.upper() == 'AND':
        mask = np.all(neighbor_matrix, axis=1)
    else:
        sys.exit(f"Error: Invalid neighbor_logic '{logic}'.")

    filtered_df = df[~mask].reset_index(drop=True)
    logger.info(f"Neighbor Cut ({logic}) on {cols_to_use}: {len(df)} -> {len(filtered_df)} events.")
    return filtered_df

# --- Core Physics Logic ---

def apply_timewalk_correction(df: pd.DataFrame, roles: list[str]):
    """Iteratively corrects Time Walk."""
    tots = {r: df[f'tot_{r}'].values for r in roles}
    toas = {r: df[f'toa_{r}'].values.copy() for r in roles}

    def get_deltas(current_toas):
        d = {}
        for r in roles:
            others = [current_toas[o] for o in roles if o != r]
            d[r] = (0.5 * sum(others)) - current_toas[r]
        return d

    for _ in range(2):
        delta_toas = get_deltas(toas)
        for r in roles:
            coeff = np.polyfit(tots[r], delta_toas[r], 2)
            toas[r] += np.poly1d(coeff)(tots[r])
    return toas

def calculate_gmm_cdf(x: np.ndarray, weights: np.ndarray, means: np.ndarray, covariances: np.ndarray):
    """Standalone vectorized GMM CDF calculation to avoid re-definition in loops."""
    cdf_val = np.zeros_like(x)
    stds = np.sqrt(covariances.flatten())
    means = means.flatten()

    for w, m, s in zip(weights, means, stds):
        cdf_val += w * norm.cdf(x, m, s)
    return cdf_val

def fit_gmm_and_get_fwhm(data: np.ndarray):
    """Fits GMM and returns FWHM and KS score."""
    data_reshaped = data.reshape(-1, 1)
    data_sorted = np.sort(data)
    n_events = len(data)
    components_to_try = [1, 2, 3] if n_events < 1500 else [3]
    best_fwhm, best_ks = 0.0, 1.0

    for n_comp in components_to_try:
        try:
            gmm = GaussianMixture(n_components=n_comp, n_init=3).fit(data_reshaped)
            y_theoretical = calculate_gmm_cdf(data_sorted, gmm.weights_, gmm.means_, gmm.covariances_)
            ks_score, _ = kstest(data_sorted, y_theoretical)

            x_range = np.linspace(data.min(), data.max(), 1000).reshape(-1, 1)
            pdf_range = np.exp(gmm.score_samples(x_range))
            peak_val = np.max(pdf_range)
            half_max_indices = np.where(pdf_range >= peak_val / 2.0)[0]

            if len(half_max_indices) > 1:
                fwhm = float(x_range[half_max_indices[-1], 0] - x_range[half_max_indices[0], 0])

            if ks_score < best_ks:
                    best_ks, best_fwhm = ks_score, fwhm

        except Exception:
            continue

    return best_fwhm, best_ks

def calculate_resolution_from_fit(fit_params: dict, roles: list[str]):
    """Solves 3-board resolution equations."""
    results = {}
    for target in roles:
        others = [r for r in roles if r != target]
        s_t_o1 = fit_params.get(f"{target}-{others[0]}", fit_params.get(f"{others[0]}-{target}", 0))**2
        s_t_o2 = fit_params.get(f"{target}-{others[1]}", fit_params.get(f"{others[1]}-{target}", 0))**2
        s_o1_o2 = fit_params.get(f"{others[0]}-{others[1]}", fit_params.get(f"{others[1]}-{others[0]}", 0))**2
        val_sq = 0.5 * (s_t_o1 + s_t_o2 - s_o1_o2)
        results[target] = np.sqrt(val_sq) if val_sq > 0 else 0.0
    return results

def run_sample_analysis(sample_df: pd.DataFrame, roles: list[str], threshold: float, is_boot: bool):
    """Single pipeline run for a data sample."""
    label = "Bootstrap" if is_boot else "SINGLE-SHOT"

    corr_toas = apply_timewalk_correction(sample_df, roles)
    diffs = {f"{r1}-{r2}": corr_toas[r1] - corr_toas[r2] for r1, r2 in combinations(roles, 2)}

    fit_sigmas = {}
    for pair, data in diffs.items():
        fwhm, ks = fit_gmm_and_get_fwhm(data)
        if ks > threshold:
            logger.info(f"{label}] Rejecting pair {pair}: KS score {ks:.4f} > {threshold}")
            return None, False

        if fwhm == 0:
            logger.info(f"[{label}] Rejecting pair {pair}: FWHM fit failed.")
            return None, False

        fit_sigmas[pair] = fwhm / 2.355

    res = calculate_resolution_from_fit(fit_sigmas, roles)

    for role, val in res.items():
        if val <= 0:
            logger.warning(f"[{label}] Physics Failure: Result for {role} is imaginary/zero ({val}).")
            return None, False

    return res, True

# --- Main Workflow ---

def main():
    parser = argparse.ArgumentParser(description='Unified Analysis with Neighbor Logic')
    parser.add_argument('-f', '--file', required=True, help='Input file')
    parser.add_argument('-n', '--num_bootstrap_output', type=int, default=200)
    parser.add_argument('-s', '--sampling', type=int, default=75)
    parser.add_argument('--iteration_limit', type=int, default=7500)
    parser.add_argument('--minimum_nevt', type=int, default=100)
    parser.add_argument('--reproducible', action='store_true')
    parser.add_argument('--neighbor_cut', dest='neighbor_cut', default=['none'], nargs='+',
                        help='Specify one or more **space-separated** board columns to be used for neighbor cuts. '
                        'The argument collects all values into a list. '
                        'Possible columns: HasNeighbor_dut, HasNeighbor_ref, HasNeighbor_extra, HasNeighbor_trig, trackNeighbor. '
                        'Default value is a list containing only "none".')
    parser.add_argument('--neighbor_logic', dest='neighbor_logic', default='OR',
                        help='Logic for multiple neighbor cuts on board. Default is OR. AND is possble.')

    args = parser.parse_args()

    # 1. Metadata & Data Loading
    input_path = Path(args.file)
    try:
        active_roles = sorted({'trig', 'dut', 'ref', 'extra'} - {input_path.stem.split('_')[1]})
        logger.info(f"Processing {input_path.stem} | Active: {active_roles}")
    except:
        sys.exit("Error: Check filename format.")

    df = pd.read_pickle(input_path) if input_path.suffix == '.pkl' else pd.read_parquet(input_path)

    # 2. Apply Neighbor Cut
    df = apply_neighbor_cut(df, args.neighbor_cut, args.neighbor_logic)

    if df.shape[0] < args.minimum_nevt:
        sys.exit(f"Input dataframe is not enough to run bootstrap. Size: {df.shape[0]} < Cut: {args.minimum_nevt}")

    # 3. Unified Phase Execution
    final_results = []
    phases = [(1.0, 1, False), (args.sampling * 0.01, args.num_bootstrap_output, True)]

    for fraction, target, is_boot in phases:
        n_success, attempts = 0, 0
        current_threshold = 0.03
        logger.info(f"Starting Phase: {'Bootstrap' if is_boot else 'Single-Shot'}")

        while n_success < target and attempts < args.iteration_limit:
            attempts += 1

            if attempts % 100 == 0 and n_success == 0 and current_threshold < 0.05:
                current_threshold += 0.001
                logger.warning(f"[Relaxation] Low success rate. Increasing threshold to {current_threshold:.4f}")

            sample = df.sample(frac=fraction) if fraction < 1.0 else df
            res, success = run_sample_analysis(sample, active_roles, threshold=current_threshold, is_boot=is_boot)

            if success:
                res['is_bootstrap'] = is_boot
                final_results.append(res)
                n_success += 1
                if is_boot and n_success == 1:
                    logger.info(f"[Stability] Found first success at {current_threshold:.3f}. Locking threshold.")
                if is_boot and n_success % 20 == 0:
                    logger.info(f"Success: {n_success}/{target}")

    # 4. Save Output
    if final_results:
        res_df = pd.DataFrame(final_results)
        output_name = f"{input_path.stem}_boot.parquet"
        res_df.to_parquet(output_name, compression='lz4')
        logger.info(f"Results saved to {output_name}")

if __name__ == "__main__":
    main()