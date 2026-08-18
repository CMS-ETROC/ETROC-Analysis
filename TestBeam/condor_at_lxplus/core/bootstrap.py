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

import io_utils

# --- Configuration & Logging ---
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

ALL_ROLES = {'trig', 'dut', 'ref', 'extra'}

# Convergence of the Gaussian-mixture fit. sklearn's defaults (tol=1e-3 on the
# per-sample lower-bound gain, max_iter=100) stop EM after ~10 iterations on
# these TOA-difference distributions with converged_=True, but the mixture is
# not converged: the broad component has not finished absorbing the tails, the
# peak is too high and narrow, and FWHM/2.355 comes out ~12% LOW on tracks with
# thousands of events (measured: 43.4 -> 49.2 ps on a 21k-event pair,
# sigma_dut 31.6 -> 35.6 ps), with KS p-values of ~0.05-0.1 that jump to
# ~0.9-0.99 once converged. tol=1e-6 with max_iter=2000 (~40 iterations at 300
# events, 130-200 at 20k; 2-3x the fit time) recovers the width to ~1%; the residual creep between
# 1e-6 and 1e-7 (~0.7%) is the intrinsic softness of an FWHM-of-a-mixture
# definition, quoted as a systematic rather than chased further.
GMM_TOL = 1e-6
GMM_MAX_ITER = 2000
KS_PMIN_DEFAULT = 1e-3     # see run_sample_analysis()
KS_DMAX_DEFAULT = 0.03     # see run_sample_analysis()

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
    """Fits GMM and returns (FWHM, KS distance, KS p-value) of the kept mixture.

    The kept mixture is still the candidate with the smallest KS distance; the
    p-value of that same test is returned alongside because the acceptance in
    run_sample_analysis() is on the p-value (see there for why)."""
    data_reshaped = data.reshape(-1, 1)
    data_sorted = np.sort(data)
    n_events = len(data)
    components_to_try = [1, 2, 3] if n_events < 1500 else [3]
    best_fwhm, best_ks, best_p = 0.0, 1.0, np.nan

    for n_comp in components_to_try:
        try:
            # random_state intentionally left unseeded even under --reproducible: n_init=3
            # already does multi-restart to find a good fit, and pinning a seed here would
            # just lock in whichever local optimum that seed lands on rather than the best one.
            gmm = GaussianMixture(n_components=n_comp, n_init=3, tol=GMM_TOL, max_iter=GMM_MAX_ITER).fit(data_reshaped)
            ks_score, ks_p = kstest(data_sorted, lambda x: calculate_gmm_cdf(x, gmm.weights_, gmm.means_, gmm.covariances_))

            x_range = np.linspace(data.min(), data.max(), 1000).reshape(-1, 1)
            pdf_range = np.exp(gmm.score_samples(x_range))
            peak_val = np.max(pdf_range)
            half_max_indices = np.where(pdf_range >= peak_val / 2.0)[0]

            if len(half_max_indices) <= 1:
                # No usable half-max width from this component count - skip it rather
                # than falling through with a stale fwhm from a previous n_comp's fit.
                continue
            fwhm = float(x_range[half_max_indices[-1], 0] - x_range[half_max_indices[0], 0])

            if ks_score < best_ks:
                best_ks, best_fwhm, best_p = ks_score, fwhm, ks_p

        except Exception:
            continue

    return best_fwhm, best_ks, best_p

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

def run_sample_analysis(sample_df: pd.DataFrame, roles: list[str], threshold: float, is_boot: bool,
                        ks_dmax: float = KS_DMAX_DEFAULT):
    """Single pipeline run for a data sample.

    A pair's mixture fit is ACCEPTED if its KS p-value is at least `threshold`
    OR its KS distance is at most `ks_dmax`; it is rejected only when both
    fail. History and reasoning:
    - the original rule was "KS distance <= 0.03". The distance of a perfectly
      good fit scales like ~0.83/sqrt(N) (0.048 median at N=300, 0.006 at
      N=20000), so that rule rejected ~80% of good fits at 300 events (the
      accepted resamples being the lucky fluctuations -- a biased selection)
      and tested nothing at 20000 events;
    - the p-value of the same test is N-independent for a correct, converged
      model (with the GMM_TOL convergence above, good fits sit at p ~ 0.9 at
      any N), so `threshold` (default KS_PMIN_DEFAULT = 1e-3, lenient on
      purpose) is the quality gate: it removes broken fits -- spikes and
      degenerate mixtures -- whose distance is far above the statistical
      floor. In distance terms p = 1e-3 corresponds to D ~ 0.19 / 0.11 / 0.06 /
      0.014 at N = 100 / 300 / 1000 / 20000;
    - the `ks_dmax` clause (default KS_DMAX_DEFAULT = 0.03) keeps the accepted
      set a superset of the old rule at its default threshold at every N (the
      old code relaxed the distance upward to 0.05 when nothing passed; here
      the p-value threshold relaxes instead): a fit that is within 0.03 of
      the data is never thrown away because a very large sample makes a small
      model imperfection statistically visible.

    Besides the per-board resolutions, the returned dict carries the pair
    widths (FWHM/2.355 of each pair) under 'pair_<a>-<b>' and the smallest KS
    p-value over the pairs under 'ksp_min', so downstream checks can use the
    pipeline's own numbers and monitor the acceptance.
    """
    label = "Bootstrap" if is_boot else "SINGLE-SHOT"

    corr_toas = apply_timewalk_correction(sample_df, roles)
    diffs = {f"{r1}-{r2}": corr_toas[r1] - corr_toas[r2] for r1, r2 in combinations(roles, 2)}

    fit_sigmas = {}
    ksp_min = 1.0
    for pair, data in diffs.items():
        fwhm, ks, ks_p = fit_gmm_and_get_fwhm(data)
        if fwhm == 0 or not np.isfinite(ks_p):
            logger.info(f"[{label}] Rejecting pair {pair}: FWHM fit failed.")
            return None, False

        if ks_p < threshold and ks > ks_dmax:
            logger.info(f"[{label}] Rejecting pair {pair}: KS p-value {ks_p:.2e} < {threshold:.1e} and D {ks:.4f} > {ks_dmax}")
            return None, False

        fit_sigmas[pair] = fwhm / 2.355
        ksp_min = min(ksp_min, ks_p)

    res = calculate_resolution_from_fit(fit_sigmas, roles)

    for role, val in res.items():
        if val <= 0:
            logger.warning(f"[{label}] Physics Failure: Result for {role} is imaginary/zero ({val}).")
            return None, False

    for pair, val in fit_sigmas.items():
        res[f"pair_{pair}"] = val
    res["ksp_min"] = ksp_min

    return res, True

# --- Main Workflow ---

def main():
    global GMM_TOL, GMM_MAX_ITER
    parser = argparse.ArgumentParser(description='Unified Analysis with Neighbor Logic')
    parser.add_argument('-f', '--file', required=True, help='Input file')
    parser.add_argument('-n', '--num_bootstrap_output', type=int, default=200)
    parser.add_argument('--iteration_limit', type=int, default=7500)
    parser.add_argument('--minimum_nevt', type=int, default=100)
    parser.add_argument('--reproducible', action='store_true',
                        help='Seed resampling (not the GMM fit) so results are reproducible run-to-run.')
    parser.add_argument('--ks_pmin', type=float, default=KS_PMIN_DEFAULT,
                        help='A pair\'s mixture fit is accepted if its KS p-value is at least this (default 1e-3) '
                             'OR its KS distance is at most --ks_dmax; see run_sample_analysis() for why the old '
                             '"distance <= 0.03" rule alone was a statistical rather than a quality gate. Relaxed by '
                             'halving after each failed single-shot attempt / every 100 fruitless bootstrap attempts, '
                             'down to --ks_pmin_floor (which is tried once); the bootstrap phase starts at the value the '
                             'single-shot was accepted at.')
    parser.add_argument('--ks_pmin_floor', type=float, default=1e-6,
                        help='Lowest p-value threshold tried (default 1e-6). The single-shot gives up after failing at '
                             'the floor; the bootstrap phase keeps sampling at the floor until --iteration_limit.')
    parser.add_argument('--ks_dmax', type=float, default=KS_DMAX_DEFAULT,
                        help='A fit within this KS distance of the data is always accepted (default 0.03, the old rule), '
                             'so the accepted set is a superset of the old one at every N.')
    parser.add_argument('--gmm_tol', type=float, default=GMM_TOL,
                        help='EM convergence tolerance of the Gaussian-mixture fit (default 1e-6; sklearn\'s default 1e-3 '
                             'leaves the fit ~12%% too narrow, see the module constants).')
    parser.add_argument('--gmm_max_iter', type=int, default=GMM_MAX_ITER, help='EM iteration cap (default 2000).')
    parser.add_argument('--neighbor_cut', dest='neighbor_cut', default=['none'], nargs='+',
                        help='Specify one or more **space-separated** board columns to be used for neighbor cuts. '
                        'The argument collects all values into a list. '
                        'Possible columns: HasNeighbor_dut, HasNeighbor_ref, HasNeighbor_extra, HasNeighbor_trig, trackNeighbor. '
                        'Default value is a list containing only "none".')
    parser.add_argument('--neighbor_logic', dest='neighbor_logic', default='OR',
                        help='Logic for multiple neighbor cuts on board. Default is OR. AND is possble.')

    args = parser.parse_args()
    if args.ks_pmin_floor > args.ks_pmin:
        parser.error('--ks_pmin_floor must not exceed --ks_pmin')
    GMM_TOL, GMM_MAX_ITER = args.gmm_tol, args.gmm_max_iter

    # 1. Metadata & Data Loading
    input_path = Path(args.file)
    df = pd.read_parquet(input_path)

    # Derive active roles from the columns actually present, not the hardcoded
    # ALL_ROLES universe -- path_finder.py only ever produces 3-board (leave-
    # one-out) track files, each missing exactly one role entirely.
    active_roles = sorted(role for role in ALL_ROLES if f'toa_{role}' in df.columns)
    logger.info(f"Processing {input_path.stem} | Active: {active_roles}")

    # 2. Apply Neighbor Cut
    df = apply_neighbor_cut(df, args.neighbor_cut, args.neighbor_logic)

    if df.shape[0] < args.minimum_nevt:
        sys.exit(f"Input dataframe is not enough to run bootstrap. Size: {df.shape[0]} < Cut: {args.minimum_nevt}")

    # 3. Unified Phase Execution
    final_results = []
    # Single-shot: the actual central value, computed once on the untouched full
    # dataset. Bootstrap: proper resamples of the same size n, WITH replacement, so
    # their spread estimates Var of the full-n estimator (not a smaller subsample's).
    phases = [(1, False), (args.num_bootstrap_output, True)]

    # One independent seed per (phase, attempt) instead of a single shared/advancing
    # random state, so any specific attempt's resample can be regenerated directly and
    # doesn't depend on the exact sequence of every prior draw. Does not seed the GMM
    # fit itself - that's a numerical fitting detail, not part of the resampling design.
    root_seed_seq = np.random.SeedSequence(42) if args.reproducible else None
    phase_seed_seqs = root_seed_seq.spawn(len(phases)) if root_seed_seq is not None else [None] * len(phases)

    ss_accept_threshold = args.ks_pmin   # the bootstrap phase starts where the single-shot was accepted
    for phase_idx, (target, is_boot) in enumerate(phases):
        n_success, attempts = 0, 0
        current_threshold = min(args.ks_pmin, ss_accept_threshold) if is_boot else args.ks_pmin
        logger.info(f"Starting Phase: {'Bootstrap' if is_boot else 'Single-Shot'} "
                    f"(accept if KS p-value >= {current_threshold:.1e} or D <= {args.ks_dmax})")

        max_attempts = args.iteration_limit if is_boot else 30

        phase_seed_seq = phase_seed_seqs[phase_idx]
        attempt_seeds = phase_seed_seq.spawn(max_attempts) if phase_seed_seq is not None else None

        while n_success < target and attempts < max_attempts:
            attempt_rng = np.random.default_rng(attempt_seeds[attempts]) if attempt_seeds is not None else None
            attempts += 1

            sample = df.sample(frac=1.0, replace=True, random_state=attempt_rng) if is_boot else df
            res, success = run_sample_analysis(sample, active_roles, threshold=current_threshold, is_boot=is_boot,
                                               ks_dmax=args.ks_dmax)

            if success:
                res['is_bootstrap'] = is_boot
                final_results.append(res)
                n_success += 1
                if not is_boot:
                    ss_accept_threshold = current_threshold

                if is_boot and n_success == 1:
                    logger.info(f"[Stability] Found first bootstrap success at p-value threshold {current_threshold:.1e}. Locking threshold.")

                if is_boot and n_success % 20 == 0:
                    logger.info(f"Success: {n_success}/{target}")

            if not is_boot:
                # Relax the p-value threshold after every failed full-sample attempt; the floor itself is
                # tried once, then the phase gives up (-1 placeholders below).
                if n_success == 0:
                    if current_threshold <= args.ks_pmin_floor:
                        break
                    current_threshold = max(current_threshold * 0.5, args.ks_pmin_floor)
            else:
                # Standard relaxation for Bootstrap resamples: only while nothing succeeded yet; at the
                # floor it keeps sampling until iteration_limit.
                if attempts % 100 == 0 and n_success == 0 and current_threshold > args.ks_pmin_floor:
                    current_threshold = max(current_threshold * 0.5, args.ks_pmin_floor)
                    logger.warning(f"[Relaxation] Lowering p-value threshold to {current_threshold:.1e}")

        logger.info(f"[Summary] {'Bootstrap' if is_boot else 'Single-Shot'}: {n_success}/{attempts} attempts accepted, "
                    f"final p-value threshold {current_threshold:.1e}")
        if is_boot and attempts > 0 and n_success / attempts < 0.7:
            logger.warning(f"[Summary] Bootstrap acceptance rate {n_success / attempts:.2f} is low: the accepted resamples "
                           f"may be a biased subset (check the KS gate for this track).")

        # --- ADDED LOGIC FOR SINGLE-SHOT FAILURE ---
        if n_success == 0:
            label = "Bootstrap" if is_boot else "Single-Shot"
            logger.error(f"FAILED {label} phase after {attempts} attempts. Appending -1 placeholders.")

            # For Bootstrap failure, we only add 1 row of -1.0 to signify the failure
            # (roles and pair widths alike, so the output schema does not depend on success)
            fail_res = {role: -1.0 for role in active_roles}
            for r1, r2 in combinations(active_roles, 2):
                fail_res[f"pair_{r1}-{r2}"] = -1.0
            fail_res['ksp_min'] = -1.0
            fail_res['is_bootstrap'] = is_boot
            final_results.append(fail_res)
        elif n_success < target:
            label = "Bootstrap" if is_boot else "Single-Shot"
            logger.warning(f"[{label}] Only reached {n_success}/{target} successes after {attempts} attempts "
                            f"(hit the {'iteration_limit' if is_boot else 'attempt'} cap) -- output has fewer samples than requested.")

    # 4. Save Output
    if final_results:
        res_df = pd.DataFrame(final_results)
        output_name = f"{input_path.stem}_boot.parquet"
        io_utils.write_parquet(res_df, output_name, compression='lz4')
        logger.info(f"Results saved to {output_name}")

if __name__ == "__main__":
    main()