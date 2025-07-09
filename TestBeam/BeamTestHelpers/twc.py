import numpy as np
import pandas as pd

from pathlib import Path

__all__ = [
    'poly2D',
    'poly3D',
    'three_board_iterative_timewalk_correction',
    'four_board_iterative_timewalk_correction',
    'fwhm_based_on_gaussian_mixture_model',
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
def three_board_iterative_timewalk_correction(
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
        coeff_b0 = np.polyfit(input_df[f'tot_b{board_ids[0]}'].values, del_toa_b0, poly_order)
        poly_func_b0 = np.poly1d(coeff_b0)

        coeff_b1 = np.polyfit(input_df[f'tot_b{board_ids[1]}'].values, del_toa_b1, poly_order)
        poly_func_b1 = np.poly1d(coeff_b1)

        coeff_b2 = np.polyfit(input_df[f'tot_b{board_ids[2]}'].values, del_toa_b2, poly_order)
        poly_func_b2 = np.poly1d(coeff_b2)

        corr_b0 = corr_b0 + poly_func_b0(input_df[f'tot_b{board_ids[0]}'].values)
        corr_b1 = corr_b1 + poly_func_b1(input_df[f'tot_b{board_ids[1]}'].values)
        corr_b2 = corr_b2 + poly_func_b2(input_df[f'tot_b{board_ids[2]}'].values)

        del_toa_b0 = (0.5*(corr_b1 + corr_b2) - corr_b0)
        del_toa_b1 = (0.5*(corr_b0 + corr_b2) - corr_b1)
        del_toa_b2 = (0.5*(corr_b0 + corr_b1) - corr_b2)

        if i == iterative_cnt-1:
            corr_toas[f'toa_b{board_ids[0]}'] = corr_b0
            corr_toas[f'toa_b{board_ids[1]}'] = corr_b1
            corr_toas[f'toa_b{board_ids[2]}'] = corr_b2

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
        tb_loc: str,
        tag: str,
        hist_bins: int = 30,
        n_components: int = 3,
        show_plot: bool = False,
        show_sub_gaussian: bool = False,
        show_fwhm_guideline: bool = False,
        show_number: bool = False,
        save_mother_dir: Path | None = None,
        fname_tag: str = '',
    ):
    """Find the sigma of delta TOA distribution and plot the distribution.

    Parameters
    ----------
    input_data: np.array,
        A numpy array includes delta TOA values.
    tb_loc: str,
        Test Beam location for the title. Available argument: desy, cern, fnal.
    tag: str,
        Additional string to show which boards are used for delta TOA calculation.
    hist_bins: int
        Bins for histogram.
    n_components: int
        Number of sub-gaussian to be considered for the Gaussian Mixture Model
    show_sub_gaussian: bool, optional
        If it is True, show sub-gaussian in the plot.
    show_fwhm_guideline: bool, optional
        If it is True, show horizontal and vertical lines to show how FWHM has been performed.
    show_number: bool, optional
        If it is True, FWHM and sigma will be shown in the plot.
    save_mother_dir: Path, optional
        Plot will be saved at save_mother_dir/'fwhm'.
    fname_tag: str, optional
        Additional tag for the file name.
    """

    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
    from scipy.spatial import distance


    x_range = np.linspace(input_data.min(), input_data.max(), 1000).reshape(-1, 1)
    bins, edges = np.histogram(input_data, bins=hist_bins, density=True)
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
    xval = x_range[np.argmax(pdf)][0]

    if show_sub_gaussian:
        # Compute PDF for each component
        responsibilities = models.predict_proba(x_range)
        pdf_individual = responsibilities * pdf[:, np.newaxis]

    if show_plot:
        from plotting import load_fig_title
        import matplotlib.pyplot as plt
        import mplhep as hep
        hep.style.use('CMS')

        loc_title = load_fig_title(tb_loc)
        fig, ax = plt.subplots(figsize=(11,10))

        # Plot data histogram
        bins, _, _ = ax.hist(input_data, bins=hist_bins, density=True, histtype='stepfilled', alpha=0.4, label='Data')

        # Plot PDF of whole model
        hep.cms.text(loc=0, ax=ax, text="ETL ETROC Test Beam", fontsize=18)
        ax.set_title(loc_title, loc="right", fontsize=16)
        ax.set_xlabel(rf'$\Delta \mathrm{{TOA}}_{{{tag}}}$ [ps]', fontsize=25)
        ax.yaxis.label.set_fontsize(25)
        if show_number:
            ax.plot(x_range, pdf, '-k', label=f'Mixture PDF, mean: {xval:.2f}')
            ax.plot(np.nan, np.nan, linestyle='none', label=f'FWHM:{fwhm[0]:.2f}, sigma:{fwhm[0]/2.355:.2f}')
        else:
            ax.plot(x_range, pdf, '-k', label=f'Mixture PDF')

        if show_sub_gaussian:
            # Plot PDF of each component
            ax.plot(x_range, pdf_individual, '--', label='Component PDF')

        if show_fwhm_guideline:
            ax.vlines(x_range[half_max_indices[0]],  ymin=0, ymax=np.max(bins)*0.75, lw=1.5, colors='red')
            ax.vlines(x_range[half_max_indices[-1]], ymin=0, ymax=np.max(bins)*0.75, lw=1.5, colors='red')
            ax.hlines(y=peak_height, xmin=x_range[0], xmax=x_range[-1], lw=1.5, colors='crimson', label='Max')
            ax.hlines(y=half_max, xmin=x_range[0], xmax=x_range[-1], lw=1.5, colors='deeppink', label='Half Max')

        ax.legend(loc='best', fontsize=14)
        plt.tight_layout()

        if save_mother_dir is not None:
            save_dir = save_mother_dir / 'fwhm'
            save_dir.mkdir(exist_ok=True)
            fig.savefig(save_dir / f"fwhm_{tag}_{fname_tag}.png")
            fig.savefig(save_dir / f"fwhm_{tag}_{fname_tag}.pdf")
            plt.close(fig)

    return fwhm, [silhouette_eval_score, jensenshannon_score]

## --------------- Time Walk Correction -----------------------