import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from scipy.stats import norm, kstest


__all__ = [
    'poly2D',
    'poly3D',
    'three_board_iterative_timewalk_correction',
    'fit_gmm_and_get_fwhm',
    'calculate_resolution_from_fit',
]

## --------------- Time Walk Correction -----------------------
## --------------------------------------
def poly2D(max_order, x, y, *args):
    if max_order < 0:
        raise RuntimeError("The polynomial order must be non-negative")

    ret_val = None

    linear_idx = 0
    for i in range(max_order+1):
        for j in range(max_order - i + 1):
            this_val = args[linear_idx] * x**j * y**i
            linear_idx += 1

            if ret_val is None:
                ret_val = this_val
            else:
                ret_val += this_val

    return ret_val

## --------------------------------------
def poly3D(max_order, x, y, z, *args):
    if max_order < 0:
        raise RuntimeError("The polynomial order must be non-negative")

    ret_val = None

    linear_idx = 0
    for i in range(max_order+1):
        for j in range(max_order - i + 1):
            for k in range(max_order - i - j + 1):
                this_val = args[linear_idx] * x**k * y**j + z**i
                linear_idx += 1

                if ret_val is None:
                    ret_val = this_val
                else:
                    ret_val += this_val

    return ret_val

## --------------------------------------
def three_board_iterative_timewalk_correction(df: pd.DataFrame, roles: list[str]):
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

## --------------------------------------
def calculate_gmm_cdf(x: np.ndarray, weights: np.ndarray, means: np.ndarray, covariances: np.ndarray):
    """Standalone vectorized GMM CDF calculation to avoid re-definition in loops."""
    cdf_val = np.zeros_like(x)
    stds = np.sqrt(covariances.flatten())
    means = means.flatten()

    for w, m, s in zip(weights, means, stds):
        cdf_val += w * norm.cdf(x, m, s)
    return cdf_val

## --------------------------------------
def fit_gmm_and_get_fwhm(data: np.ndarray, pair: str, plot_result: bool = False, plot_cdf: bool = False):
    """Fits GMM and returns FWHM and KS score."""
    data_reshaped = data.reshape(-1, 1)
    data_sorted = np.sort(data)
    n_events = len(data)
    components_to_try = [1, 2, 3] if n_events < 1500 else [3]
    best_fwhm, best_ks = 0.0, 1.0
    best_gmm = None
    best_x_range, best_pdf = None, None

    for n_comp in components_to_try:
        try:
            gmm = GaussianMixture(n_components=n_comp, n_init=3).fit(data_reshaped)
            ks_score, _ = kstest(data_sorted, lambda x: calculate_gmm_cdf(x, gmm.weights_, gmm.means_, gmm.covariances_))

            x_range = np.linspace(data.min(), data.max(), 1000).reshape(-1, 1)
            pdf_range = np.exp(gmm.score_samples(x_range))
            peak_val = np.max(pdf_range)
            half_max_indices = np.where(pdf_range >= peak_val / 2.0)[0]

            if len(half_max_indices) > 1:
                fwhm = float(x_range[half_max_indices[-1], 0] - x_range[half_max_indices[0], 0])

            if ks_score < best_ks:
                best_ks, best_fwhm = ks_score, fwhm
                best_gmm = gmm
                best_x_range, best_pdf = x_range, pdf_range

        except Exception:
            continue

    # --- Optional Plotting Logic ---
    if plot_result and best_gmm is not None:
        plt.figure(figsize=(11, 9))
        plt.hist(data, bins=50, range=[-1000, 1000], density=True, alpha=0.4, color='teal', label='Data')
        plt.plot(best_x_range, best_pdf, color='black', lw=2, label=f'Best GMM (subGaussian={best_gmm.n_components})')

        # # Plot individual Gaussian components (Sub-distributions)
        # for i in range(best_gmm.n_components):
        #     weight = best_gmm.weights_[i]
        #     mean = best_gmm.means_[i, 0]
        #     variance = best_gmm.covariances_[i, 0, 0]
        #     std_dev = np.sqrt(variance)

        #     # Weighted Gaussian: weight * Normal PDF
        #     component_pdf = weight * norm.pdf(best_x_range.flatten(), mean, std_dev)
        #     plt.plot(best_x_range, component_pdf, '--', lw=1.5, label=f'Comp {i+1} (w={weight:.2f})')

        # Visualize FWHM
        peak_y = np.max(best_pdf)
        plt.hlines(y=peak_y/2, xmin=best_x_range[np.where(best_pdf >= peak_y/2)[0][0]],
                   xmax=best_x_range[np.where(best_pdf >= peak_y/2)[0][-1]],
                   colors='red', linestyles='--', label=f'FWHM: {best_fwhm:.2f}')

        plt.title(f"GMM Fit for {pair}")
        plt.xlabel(f'$\Delta$ TOA (ps)')
        plt.ylabel("Density")
        plt.legend(fontsize=17)
        plt.tight_layout()

    if plot_cdf and best_gmm is not None:
        plt.figure(figsize=(11, 9))

        # 1. Empirical CDF - The "Background"
        # Use a thicker solid line to act as a border for the model line
        cdf_empirical = np.arange(1, n_events + 1) / n_events
        plt.step(data_sorted, cdf_empirical, color='black', alpha=0.3, lw=4,
                label='Empirical (Data)', where='post')

        # 2. GMM Theoretical CDF - The "Foreground"
        # Use a thinner, dashed line in a bright color to cut through the black
        gmm_cdf_y = calculate_gmm_cdf(best_x_range.flatten(), best_gmm.weights_,
                                    best_gmm.means_, best_gmm.covariances_)
        plt.plot(best_x_range, gmm_cdf_y, color='red', lw=2, linestyle='--',
                dash_capstyle='round', dashes=(5, 2), label='GMM Fit')

        plt.title(f"CDF Comparison: {pair} (KS: {best_ks:.4f})")
        plt.xlabel(r'$\Delta$ TOA [ps]')
        plt.ylabel("Cumulative Probability")
        plt.xlim(-1000, 1000)
        plt.ylim(-0.05, 1.05)
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.legend(loc='lower right')
        plt.tight_layout()

    return best_fwhm, best_ks

## --------------------------------------
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

# ## --------------------------------------
# def four_board_iterative_timewalk_correction(
#     input_df: pd.DataFrame,
#     iterative_cnt: int,
#     poly_order: int,
#     ):

#     corr_toas = {}
#     corr_b0 = input_df['toa_b0'].values
#     corr_b1 = input_df['toa_b1'].values
#     corr_b2 = input_df['toa_b2'].values
#     corr_b3 = input_df['toa_b3'].values

#     del_toa_b3 = ((1/3)*(input_df['toa_b0'] + input_df['toa_b1'] + input_df['toa_b2']) - input_df['toa_b3']).values
#     del_toa_b2 = ((1/3)*(input_df['toa_b0'] + input_df['toa_b1'] + input_df['toa_b3']) - input_df['toa_b2']).values
#     del_toa_b1 = ((1/3)*(input_df['toa_b0'] + input_df['toa_b3'] + input_df['toa_b2']) - input_df['toa_b1']).values
#     del_toa_b0 = ((1/3)*(input_df['toa_b3'] + input_df['toa_b1'] + input_df['toa_b2']) - input_df['toa_b0']).values

#     for i in range(iterative_cnt):
#         coeff_b0 = np.polyfit(input_df['tot_b0'].values, del_toa_b0, poly_order)
#         poly_func_b0 = np.poly1d(coeff_b0)

#         coeff_b1 = np.polyfit(input_df['tot_b1'].values, del_toa_b1, poly_order)
#         poly_func_b1 = np.poly1d(coeff_b1)

#         coeff_b2 = np.polyfit(input_df['tot_b2'].values, del_toa_b2, poly_order)
#         poly_func_b2 = np.poly1d(coeff_b2)

#         coeff_b3 = np.polyfit(input_df['tot_b3'].values, del_toa_b3, poly_order)
#         poly_func_b3 = np.poly1d(coeff_b3)

#         corr_b0 = corr_b0 + poly_func_b0(input_df['tot_b0'].values)
#         corr_b1 = corr_b1 + poly_func_b1(input_df['tot_b1'].values)
#         corr_b2 = corr_b2 + poly_func_b2(input_df['tot_b2'].values)
#         corr_b3 = corr_b3 + poly_func_b3(input_df['tot_b3'].values)

#         del_toa_b3 = ((1/3)*(corr_b0 + corr_b1 + corr_b2) - corr_b3)
#         del_toa_b2 = ((1/3)*(corr_b0 + corr_b1 + corr_b3) - corr_b2)
#         del_toa_b1 = ((1/3)*(corr_b0 + corr_b3 + corr_b2) - corr_b1)
#         del_toa_b0 = ((1/3)*(corr_b3 + corr_b1 + corr_b2) - corr_b0)

#         if i == iterative_cnt-1:
#             corr_toas[f'toa_b0'] = corr_b0
#             corr_toas[f'toa_b1'] = corr_b1
#             corr_toas[f'toa_b2'] = corr_b2
#             corr_toas[f'toa_b3'] = corr_b3

#     return corr_toas