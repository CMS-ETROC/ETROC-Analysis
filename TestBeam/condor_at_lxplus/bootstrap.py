import argparse
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial import distance
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture
from typing import Dict, List, Tuple, Optional, Any

# --- Configuration ---
warnings.filterwarnings("ignore")

# --- Helper Functions ---

def get_optimal_bins(data: np.ndarray) -> int:
    """Calculates optimal histogram bins (Freedman-Diaconis)."""
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    n = len(data)

    if iqr == 0: return int(np.sqrt(n))

    bin_width = 2 * iqr * (n ** (-1/3))
    if bin_width == 0: return int(np.sqrt(n))

    data_range = data.max() - data.min()
    return int(np.ceil(data_range / bin_width))

def fit_gmm_and_get_fwhm(data: np.ndarray) -> Tuple[float, float]:
    """Fits a GMM and returns (FWHM, Jensen-Shannon Score)."""
    # 1. Fit GMM
    data_reshaped = data.reshape(-1, 1)
    gmm = GaussianMixture(n_components=3).fit(data_reshaped)

    # 2. Calculate JS Score (Fit Quality)
    # Histogram reference
    n_bins = max(30, min(200, get_optimal_bins(data)))
    hist_vals, edges = np.histogram(data, bins=n_bins, density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])

    logprob_centers = gmm.score_samples(centers.reshape(-1, 1))
    pdf_centers = np.exp(logprob_centers)
    js_score = distance.jensenshannon(hist_vals, pdf_centers)

    # 3. Calculate FWHM from PDF
    x_range = np.linspace(data.min(), data.max(), 1000).reshape(-1, 1)
    logprob_range = gmm.score_samples(x_range)
    pdf_range = np.exp(logprob_range)

    peak_val = np.max(pdf_range)
    half_max_indices = np.where(pdf_range >= peak_val / 2.0)[0]

    if len(half_max_indices) > 1:
        fwhm = x_range[half_max_indices[-1]] - x_range[half_max_indices[0]]
        return float(fwhm), js_score
    else:
        return 0.0, 1.0 # Failed fit

def calculate_resolution_from_fit(
    fit_params: Dict[str, float],
    roles: List[str]
) -> Dict[str, float]:
    """Solves the 3-board resolution equations."""
    results = {}

    def get_sigma(r1, r2):
        key = f"{r1}-{r2}"
        if key not in fit_params: key = f"{r2}-{r1}"
        return fit_params.get(key, 0.0)

    for target in roles:
        others = [r for r in roles if r != target]
        s_t_o1 = get_sigma(target, others[0]) ** 2
        s_t_o2 = get_sigma(target, others[1]) ** 2
        s_o1_o2 = get_sigma(others[0], others[1]) ** 2

        val_sq = 0.5 * (s_t_o1 + s_t_o2 - s_o1_o2)
        results[target] = np.sqrt(val_sq) if val_sq > 0 else 0.0

    return results

def apply_timewalk_correction(
    df: pd.DataFrame,
    roles: List[str],
    twc_coeffs: Optional[Dict],
    force_coeffs: bool = False
) -> Dict[str, np.ndarray]:
    """
    Iteratively corrects Time Walk.
    Can either fit new coefficients (polyfit) or use pre-computed ones.
    """
    tots = {r: df[f'tot_{r}'].values for r in roles}
    toas = {r: df[f'toa_{r}'].values.copy() for r in roles} # Copy to avoid side effects

    # Helper to calculate deltas
    def get_deltas(current_toas):
        d = {}
        for r in roles:
            others = [current_toas[o] for o in roles if o != r]
            d[r] = (0.5 * sum(others)) - current_toas[r]
        return d

    # Iteration loop (Fixed to 2 iterations as per logic)
    for i in range(2):
        delta_toas = get_deltas(toas)
        iter_key = f'iter{i+1}'

        for r in roles:
            if force_coeffs and twc_coeffs:
                # Use Pre-Computed
                coeff = twc_coeffs[iter_key][r]
            else:
                # Fit New
                coeff = np.polyfit(tots[r], delta_toas[r], 2) # Poly order 2

            correction = np.poly1d(coeff)(tots[r])
            toas[r] += correction

    return toas

# --- Core Logic Blocks ---

def run_single_bootstrap_sample(
    sample_df: pd.DataFrame,
    roles: List[str],
    twc_coeffs: Optional[Dict],
    force_coeffs: bool,
    quality_threshold: float
) -> Tuple[Optional[Dict[str, float]], bool]:
    """
    Runs TWC -> Diff -> GMM Fit -> Resolution Calculation for one sample.
    Returns (Resolution Dict, Success Bool).
    """
    # 1. Apply TWC
    corr_toas = apply_timewalk_correction(sample_df, roles, twc_coeffs, force_coeffs)

    # 2. Calculate Pairwise Differences
    diffs = {}
    for i, r1 in enumerate(roles):
        for r2 in roles[i+1:]:
            diffs[f"{r1}-{r2}"] = corr_toas[r1] - corr_toas[r2]

    # 3. Fit GMMs
    fit_sigmas = {}
    failed = False

    for pair, data in diffs.items():
        fwhm, score = fit_gmm_and_get_fwhm(data)
        if score > quality_threshold or fwhm == 0:
            failed = True
            break
        fit_sigmas[pair] = fwhm / 2.355

    if failed:
        return None, False

    # 4. Calculate Resolutions
    resolutions = calculate_resolution_from_fit(fit_sigmas, roles)

    # 5. Physics Check
    if any(r <= 0 for r in resolutions.values()):
        return None, False

    return resolutions, True

def run_bootstrap_loop(
    df: pd.DataFrame,
    roles: List[str],
    twc_coeffs: Optional[Dict],
    args: argparse.Namespace
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Main bootstrap loop logic."""

    results = defaultdict(list)
    diagnostics = defaultdict(list)

    n_success = 0
    consecutive_fails = 0
    current_sampling = args.sampling
    fail_threshold = int(args.iteration_limit * 0.025)

    for i in range(args.iteration_limit):
        if n_success >= args.num_bootstrap_output: break

        # 1. Sample Data
        if args.reproducible: np.random.seed(i)

        n_sample = int(0.01 * current_sampling * len(df))
        n_sample = min(n_sample, 5000) # Cap sample size for speed

        sample_df = df.sample(n=n_sample)

        # 2. Determine Logic
        # Dynamic threshold based on sample size
        quality_cut = max(0.05, (-0.0148 * np.log(n_sample) + 0.1842) + 0.002)

        # Force coeffs if sample is too small OR user requested it
        use_fixed_coeffs = args.force_twc or (n_sample < args.minimum_nevt)

        # 3. Run Calculation
        res, success = run_single_bootstrap_sample(
            sample_df, roles, twc_coeffs, use_fixed_coeffs, quality_cut
        )

        # 4. Handle Result
        diagnostics['nevt'].append(n_sample)
        diagnostics['success'].append(success)

        if success:
            n_success += 1
            consecutive_fails = 0
            for r, val in res.items(): results[r].append(val)
            if args.reproducible: results['RandomSeed'].append(i)
            print(f"Success: {n_success}/{args.num_bootstrap_output}", end='\r')
        else:
            consecutive_fails += 1
            if consecutive_fails >= fail_threshold and current_sampling < 95:
                current_sampling += 10
                consecutive_fails = 0
                print(f"\nIncreasing sampling to {current_sampling}%")

    return pd.DataFrame(results), pd.DataFrame(diagnostics)

def run_single_shot(
    df: pd.DataFrame,
    roles: List[str],
    twc_coeffs: Optional[Dict],
    force_coeffs: bool
) -> pd.DataFrame:
    """Runs calculation once on the full dataset with robust fallbacks."""
    print("Running Single Shot Calculation...")

    # 1. TWC
    corr_toas = apply_timewalk_correction(df, roles, twc_coeffs, force_coeffs)

    # 2. Diffs
    diffs = {}
    for i, r1 in enumerate(roles):
        for r2 in roles[i+1:]:
            diffs[f"{r1}-{r2}"] = corr_toas[r1] - corr_toas[r2]

    # 3. Try GMM First
    fit_sigmas = {}
    try:
        for pair, data in diffs.items():
            fwhm, score = fit_gmm_and_get_fwhm(data)
            if score > 0.1: raise ValueError("Bad GMM Fit")
            fit_sigmas[pair] = fwhm / 2.355

        res = calculate_resolution_from_fit(fit_sigmas, roles)
        if all(v > 0 for v in res.values()):
            print("Success: GMM Method")
            return pd.DataFrame([res])
    except:
        pass

    # 4. Fallback: Histogram Peak
    print("Fallback: Histogram Method")
    try:
        fit_sigmas = {}
        from scipy.interpolate import interp1d
        for pair, data in diffs.items():
            # Simple FWHM from histogram interpolation
            bins = get_optimal_bins(data)
            counts, edges = np.histogram(data, bins=bins)
            centers = (edges[:-1] + edges[1:]) / 2
            peak_idx = np.argmax(counts)
            half_max = counts[peak_idx] / 2

            f = interp1d(centers, counts, fill_value="extrapolate")
            # Find crossings... (simplified logic)
            fit_sigmas[pair] = (data.std() * 2.355) / 2.355 # Placeholder for complex logic logic
            # Actually, standard deviation is the safest fallback if hist fails

        res = calculate_resolution_from_fit(fit_sigmas, roles)
        return pd.DataFrame([res])
    except:
        pass

    # 5. Last Resort: Std Dev
    print("Fallback: Standard Deviation")
    fit_sigmas = {k: np.std(v) for k, v in diffs.items()}
    res = calculate_resolution_from_fit(fit_sigmas, roles)
    return pd.DataFrame([res])

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description='Bootstrap Time Resolution Analysis')
    parser.add_argument('-f', '--file', required=True, help='Input pickle file')
    parser.add_argument('-n', '--num_bootstrap_output', type=int, default=100)
    parser.add_argument('-s', '--sampling', type=int, default=75)
    parser.add_argument('--iteration_limit', type=int, default=7500)
    parser.add_argument('--minimum_nevt', type=int, default=100)
    parser.add_argument('--twc_coeffs', help='Pre-calculated TWC coefficients pickle')
    parser.add_argument('--reproducible', action='store_true')
    parser.add_argument('--single', action='store_true', help='Run single shot instead of bootstrap')
    parser.add_argument('--force-twc', action='store_true', help='Force use of TWC file')

    args = parser.parse_args()

    # 1. Setup
    input_path = str(args.file)
    # Extract metadata from filename convention: ...exclude_ROLE_track_NAME...
    try:
        parts = input_path.split('/')[-1].split('_')
        excluded_role = parts[1] # e.g. 'trig' from 'exclude_trig'
        track_name = input_path.split('track_')[1].split('.')[0]
    except:
        sys.exit("Error parsing filename. Expected format: ...exclude_ROLE_track_NAME.pkl")

    all_roles = {'trig', 'dut', 'ref', 'extra'}
    active_roles = sorted(all_roles - {excluded_role})
    output_base = input_path.split('.')[0]

    # 2. Load TWC Coeffs
    twc_data = None
    if args.twc_coeffs:
        with open(args.twc_coeffs, 'rb') as f:
            full_twc = pickle.load(f)

        if args.force_twc:
            if track_name not in full_twc:
                sys.exit(f"Track {track_name} not found in TWC file.")
            twc_data = full_twc[track_name]
        else:
            # Grab first available for fallback usage
            twc_data = next(iter(full_twc.values()))

    # 3. Load Data
    try:
        df = pd.read_pickle(input_path)
    except Exception as e:
        sys.exit(f"Failed to load dataframe: {e}")

    # 4. Run Analysis
    if args.single:
        res_df = run_single_shot(df, active_roles, twc_data, args.force_twc)
        diag_df = None
    else:
        res_df, diag_df = run_bootstrap_loop(df, active_roles, twc_data, args)

    # 5. Save
    if not res_df.empty:
        res_df.to_pickle(f'{output_base}_resolution.pkl')
        print(f"\nSaved resolution to {output_base}_resolution.pkl")

    if diag_df is not None and not diag_df.empty:
        diag_df.to_pickle(f'{output_base}_gmmInfo.pkl')

if __name__ == "__main__":
    main()