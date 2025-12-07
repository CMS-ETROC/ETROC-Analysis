import pandas as pd
import numpy as np
import hist
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import mplhep as hep
hep.style.use('CMS')

from .utils import load_fig_title
from lmfit.models import GaussianModel
from lmfit.lineshapes import gaussian
from natsort import natsorted
from pathlib import Path
from matplotlib import rcParams
rcParams["axes.formatter.useoffset"] = False
rcParams["axes.formatter.use_mathtext"] = False

__all__ = [
    'fit_resolution_data',
    'calculate_weighted_mean_std_for_every_pixel',
    'preprocess_ranking_data',
    'plot_resolution_with_pulls',
    'plot_resolution_table',
    'plot_resolutions_per_row',
    'plot_avg_resolution_per_row',
]

## --------------- Plotting -----------------------
def save_plot(fig, save_dir, file_name):
    """Helper function to finalize and save a plot."""
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"{file_name}.png")
    fig.savefig(save_dir / f"{file_name}.pdf")
    plt.close(fig)


## --------------------------------------
def plot_TWC(
        input_df: pd.DataFrame,
        board_ids: list[int],
        tb_loc: str,
        poly_order: int = 2,
        corr_toas: dict | None = None,
        save_mother_dir: Path | None = None,
        print_func: bool = False,
    ):

    plot_title = load_fig_title(tb_loc)

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
    axes[0].set_title(plot_title, fontsize=16, loc='right')
    hep.hist2dplot(h_twc2, ax=axes[1], norm=colors.LogNorm())
    hep.cms.text(loc=0, ax=axes[1], text="ETL ETROC Test Beam", fontsize=18)
    axes[1].plot(b2_xrange, poly_func_b1(b2_xrange), 'r-', lw=3, label=make_legend(coeff_b1, poly_order=poly_order))
    axes[1].set_xlabel('TOT2 [ps]', fontsize=25)
    axes[1].set_ylabel('0.5*(TOA1+TOA3)-TOA2 [ps]', fontsize=25)
    axes[1].set_title(plot_title, fontsize=16, loc='right')
    hep.hist2dplot(h_twc3, ax=axes[2], norm=colors.LogNorm())
    hep.cms.text(loc=0, ax=axes[2], text="ETL ETROC Test Beam", fontsize=18)
    axes[2].plot(b3_xrange, poly_func_b2(b3_xrange), 'r-', lw=3, label=make_legend(coeff_b2, poly_order=poly_order))
    axes[2].set_xlabel('TOT3 [ps]', fontsize=25)
    axes[2].set_ylabel('0.5*(TOA1+TOA2)-TOA3 [ps]', fontsize=25)
    axes[2].set_title(plot_title, fontsize=16, loc='right')

    axes[0].legend(loc='best')
    axes[1].legend(loc='best')
    axes[2].legend(loc='best')

    if save_mother_dir is not None:
        save_dir = save_mother_dir / 'twc_fit'
        save_dir.mkdir(exist_ok=True)
        fig.savefig(save_dir / f"twc_fit.png")
        fig.savefig(save_dir / f"twc_fit.pdf")
        plt.close(fig)


## --------------------------------------
def fit_resolution_data(
    input_df: pd.DataFrame,
    list_of_boards: list[str],
    fig_config: dict,
    hist_range: list[int] = [20, 75],
    hist_bins: int = 15,
) -> dict:
    """
    Performs Gaussian fits on board resolution data.

    Returns
    -------
    dict
        A dictionary containing fit results keyed by board_id.
        Structure: { board_id: {'hist': ..., 'fit_result': ..., 'pulls': ..., 'config': ...} }
    """
    mod = GaussianModel(nan_policy='omit')
    results = {}

    for _, board_info in fig_config.items():
        role = board_info.get('role')

        # validation checks
        column_to_check = f'res_{role}'
        if column_to_check not in input_df.columns:
            continue
        if role not in list_of_boards:
            continue

        data_to_hist = input_df[f'res_{role}'].dropna().values

        # 1. Create Histogram
        h = hist.Hist(hist.axis.Regular(hist_bins, hist_range[0], hist_range[1],
                                        name="time_resolution", label='Time Resolution [ps]'))
        h.fill(data_to_hist)

        # Check if empty
        if h.sum() == 0:
            print(f"WARNING: Histogram for role: {role} is empty. Skipping fit.")
            continue

        centers = h.axes[0].centers

        # 2. Slice Data for Fitting
        peak_bin_index = np.argmax(h.values())
        fit_window_half_width = 5
        start_index = max(0, peak_bin_index - fit_window_half_width)
        end_index   = min(len(centers), peak_bin_index + fit_window_half_width + 1)

        fit_range = centers[start_index:end_index]
        fit_vals = h.values()[start_index:end_index]

        if len(fit_vals) == 0:
            print(f"WARNING: Could not create a valid fit slice for role: {role}. Skipping fit.")
            continue

        # 3. Perform Fit
        pars = mod.guess(fit_vals, x=fit_range)
        out = mod.fit(fit_vals, pars, x=fit_range, weights=1/np.sqrt(fit_vals))

        # 4. Calculate Pulls
        # Note: We evaluate the model over all centers for the pull plot, not just the fit range
        model_vals = out.eval(x=centers)
        pulls = (h.values() - model_vals) / np.sqrt(model_vals)
        pulls[np.isnan(pulls) | np.isinf(pulls)] = 0

        # 5. Store Results
        results[role] = {
            'hist': h,
            'fit_result': out,
            'pulls': pulls,
            'config': board_info, # Store config here for easy access in plotter
            'centers': centers
        }

    return results


## --------------------------------------
def plot_resolution_with_pulls(
    fit_results: dict,
    tb_loc: str,
    constraint_ylim: bool = False,
    save_mother_dir: Path | None = None,
):
    """
    Plots the resolution fit results generated by fit_resolution_data.
    """

    # Assuming load_fig_title is a helper function defined elsewhere in your scope
    # If not, you might need to pass the full title string instead of tb_loc
    try:
        plot_title = load_fig_title(tb_loc)
    except NameError:
        plot_title = f"Test Beam: {tb_loc}"

    for board_id, data in fit_results.items():
        h = data['hist']
        fit_out = data['fit_result']
        pulls = data['pulls']
        config = data['config']
        centers = data['centers']

        # Setup Canvas
        fig = plt.figure(figsize=(11.5, 10))
        grid = fig.add_gridspec(2, 1, hspace=0, height_ratios=[3, 1])

        main_ax = fig.add_subplot(grid[0])
        sub_ax = fig.add_subplot(grid[1], sharex=main_ax)
        plt.setp(main_ax.get_xticklabels(), visible=False)

        # ---------------------
        # MAIN PLOT
        # ---------------------
        hep.cms.text(loc=0, ax=main_ax, text="ETL ETROC Test Beam", fontsize=18)
        main_ax.set_title(f'{plot_title}\n{config['title']}', loc="right", size=14)

        # Plot Data Points
        main_ax.errorbar(centers, h.values(), np.sqrt(h.variances()),
                        ecolor="steelblue", mfc="steelblue", mec="steelblue", fmt="o",
                        ms=6, capsize=1, capthick=2, alpha=0.8)

        # Plot Fit Line and Uncertainty Band
        x_min, x_max = centers[0], centers[-1]
        x_range = np.linspace(x_min, x_max, 500)

        # Calculate fit uncertainty band
        popt = [par for name, par in fit_out.best_values.items()]
        pcov = fit_out.covar

        if pcov is not None and np.isfinite(pcov).all():
            n_samples = 100
            vopts = np.random.multivariate_normal(popt, pcov, n_samples)
            # Reconstruct gaussian function manually or use model eval
            # Using manual gaussian import from lmfit.lineshapes
            sampled_ydata = np.vstack([gaussian(x_range, *vopt).T for vopt in vopts])
            model_uncert = np.nanstd(sampled_ydata, axis=0)
        else:
            # Fallback if covariance is bad
            model_uncert = np.zeros_like(x_range)

        main_ax.plot(x_range, fit_out.eval(x=x_range), color="hotpink", ls="-", lw=2, alpha=0.8,
                    label=fr"$\mu$:{fit_out.params['center'].value:.2f} $\pm$ {fit_out.params['center'].stderr:.2f} ps")

        # Dummy plot for Sigma label
        main_ax.plot(np.NaN, np.NaN, color='none',
                    label=fr"$\sigma$: {abs(fit_out.params['sigma'].value):.2f} $\pm$ {abs(fit_out.params['sigma'].stderr):.2f} ps")

        main_ax.fill_between(
            x_range,
            fit_out.eval(x=x_range) - model_uncert,
            fit_out.eval(x=x_range) + model_uncert,
            color="hotpink",
            alpha=0.2,
            label='Fit Uncertainty'
        )

        main_ax.set_ylabel('Counts', fontsize=25)
        main_ax.tick_params(axis='x', labelsize=20)
        main_ax.tick_params(axis='y', labelsize=20)
        main_ax.legend(fontsize=18, loc='best')

        if constraint_ylim:
            main_ax.set_ylim(-5, 190)

        # ---------------------
        # PULL PLOT
        # ---------------------
        width = (x_max - x_min) / len(pulls)
        sub_ax.axhline(1, c='black', lw=0.75)
        sub_ax.axhline(0, c='black', lw=1.2)
        sub_ax.axhline(-1, c='black', lw=0.75)
        sub_ax.bar(centers, pulls, width=width, fc='royalblue')

        sub_ax.set_ylim(-2, 2)
        sub_ax.set_yticks(ticks=np.arange(-1, 2), labels=[-1, 0, 1])
        sub_ax.tick_params(axis='y', labelsize=20)
        sub_ax.set_xlabel('Pixel Time Resolution [ps]', fontsize=25)
        sub_ax.tick_params(axis='x', which='both', labelsize=20)
        sub_ax.set_ylabel('Pulls', fontsize=20, loc='center')

        fig.tight_layout()

        # ---------------------
        # SAVING
        # ---------------------
        if save_mother_dir is not None:
            save_dir = save_mother_dir / 'time_resolution_results'
            save_dir.mkdir(exist_ok=True, parents=True)
            # Using 'short' key from config if available, otherwise board_id
            name_tag = config.get('short', board_id)
            fig.savefig(save_dir / f"board_res_{name_tag}.png")
            fig.savefig(save_dir / f"board_res_{name_tag}.pdf")
            plt.close(fig)


## --------------------------------------
def plot_resolution_table(
        input_df: pd.DataFrame,
        tb_loc: str,
        fig_config: dict,
        min_resolution: float = 25.0,
        max_resolution: float = 75.0,
        missing_pixel_info: dict | None = None,
        show_number: bool = False,
        save_mother_dir: Path | None = None,
    ):

    from matplotlib import colormaps
    cmap = colormaps['viridis']
    cmap.set_under(color='lightgrey')

    plot_title = load_fig_title(tb_loc)

    tables = {}
    for board_id, board_info in fig_config.items():
        role = board_info.get('role')

        column_to_check = f'res_{role}'
        if not column_to_check in input_df.columns:
            continue

        board_info = input_df[[f'row_{role}', f'col_{role}', f'res_{role}', f'err_{role}']]

        res = board_info.groupby([f'row_{role}', f'col_{role}']).apply(lambda x: np.average(x[f'res_{role}'], weights=1/x[f'err_{role}']**2), include_groups=False).reset_index()
        err = board_info.groupby([f'row_{role}', f'col_{role}']).apply(lambda x: np.sqrt(1/(np.sum(1/x[f'err_{role}']**2))), include_groups=False).reset_index()

        res_table = res.pivot_table(index=f'row_{role}', columns=f'col_{role}', values=0, fill_value=-1)
        err_table = err.pivot_table(index=f'row_{role}', columns=f'col_{role}', values=0, fill_value=-1)

        res_table = res_table.reindex(pd.Index(np.arange(0,16), name='')).reset_index()
        res_table = res_table.reindex(columns=np.arange(0,16))
        res_table = res_table.fillna(-1)

        err_table = err_table.reindex(pd.Index(np.arange(0,16), name='')).reset_index()
        err_table = err_table.reindex(columns=np.arange(0,16))
        err_table = err_table.fillna(-1)

        tables[board_id] = [res_table, err_table]

    for idx in tables.keys():
        # Create a heatmap to visualize the count of hits
        fig, ax = plt.subplots(dpi=100, figsize=(15, 15))
        ax.cla()
        im = ax.imshow(tables[idx][0], cmap=cmap, interpolation="nearest", vmin=min_resolution, vmax=max_resolution)

        # Add color bar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Time Resolution (ps)', fontsize=18)
        cbar.ax.tick_params(labelsize=18)

        if show_number:
            for i in range(16):
                for j in range(16):
                    value = tables[idx][0].iloc[i, j]
                    error = tables[idx][1].iloc[i, j]
                    if value == -1: continue
                    text_color = 'black' if value > 0.66*(min_resolution + max_resolution) else 'white'
                    text = str(rf"{value:.1f}""\n"fr"$\pm$ {error:.1f}")
                    plt.text(j, i, text, va='center', ha='center', color=text_color, fontsize=15, rotation=45)

        if missing_pixel_info is not None:
            for jdx in range(len(missing_pixel_info[idx]['res'])):
                text = str(rf"{float(missing_pixel_info[idx]['res'][jdx]):.1f}""\n"fr"$\pm$ {float(missing_pixel_info[idx]['err'][jdx]):.1f}")
                plt.text(int(missing_pixel_info[idx]['col'][jdx]), int(missing_pixel_info[idx]['row'][jdx]), text, va='center', ha='center', color='black', fontsize=15 , rotation=45)

        hep.cms.text(loc=0, ax=ax, text="ETL ETROC Test Beam", fontsize=18)
        ax.set_xlabel('Column', fontsize=25)
        ax.set_ylabel('Row', fontsize=25)
        ticks = range(0, 16)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        sup_title = fig_config[idx]['title']
        ax.set_title(f"{plot_title}\n{sup_title}", loc="right", size=16)
        ax.tick_params(axis='x', which='both', length=5, labelsize=18)
        ax.tick_params(axis='y', which='both', length=5, labelsize=18)
        ax.invert_xaxis()
        ax.invert_yaxis()
        plt.minorticks_off()
        plt.tight_layout()

        if save_mother_dir is not None:
            save_dir = save_mother_dir / 'time_resolution_results'
            save_dir.mkdir(exist_ok=True)
            fig.savefig(save_dir / f"resolution_map_{fig_config[idx]['short']}.png")
            fig.savefig(save_dir / f"resolution_map_{fig_config[idx]['short']}.pdf")
            plt.close(fig)

    del tables

## --------------------------------------
def preprocess_ranking_data(
    input_df: pd.DataFrame,
    row_col: str,
    col_col: str,
    rank_by_col: str = 'nevt',
    num_cols: int = 16
):
    """
    Prepares a DataFrame for plotting by ranking and categorizing.

    Sorts by 'rank_by_col', groups by 'row_col' and 'col_col' to create ranks,
    and adds 'category' and 'subgroup' columns.
    Also returns mappings for plotting.

    Returns:
        A tuple containing:
        (df_proc, subgroup_x_map, tick_positions)
    """
    # 1. Create Subgroup/Column Mappings
    all_subgroups = [f"Col {i}" for i in range(num_cols)]
    subgroup_x_map = {subgroup: i for i, subgroup in enumerate(all_subgroups)}
    tick_positions = list(subgroup_x_map.values())

    # 2. Vectorized Data Processing
    df_proc = input_df.copy()

    # Sort by the ranking column
    df_proc = df_proc.sort_values(by=[rank_by_col], ascending=False)

    # Group by row and col, then rank by the ranking column
    ranks = df_proc.groupby([row_col, col_col])[rank_by_col].rank(ascending=False, method='first')

    # 3. Create new columns
    df_proc['category'] = "Path " + ranks.astype(int).astype(str)
    df_proc['subgroup'] = "Col " + df_proc[col_col].astype(str)

    return df_proc, subgroup_x_map, tick_positions


## --------------------------------------
def plot_resolutions_per_row(
    input_df: pd.DataFrame,
    board_role: str,
    tb_loc: str,
    fig_config: dict,
    rows_to_draw: list[int] = [0, 15],
):

    row_col = f'row_{board_role}'
    col_col = f'col_{board_role}'
    res_col = f'res_{board_role}'
    err_col = f'err_{board_role}'

    loc_title = load_fig_title(tb_loc)

    for _, val in fig_config.items():
        if val['role'] == board_role:
            selected_fig_config = val
            break

    # All the data processing happens in one line
    df_proc, subgroup_x_map, tick_positions = preprocess_ranking_data(
        input_df,
        row_col,
        col_col,
        rank_by_col='nevt',
        num_cols=16
    )

    for irow in range(rows_to_draw[0], rows_to_draw[1]+1):

        df_summary = df_proc.loc[df_proc[row_col] == irow].reset_index(drop=True)

        if df_summary.empty:
            print(f"No data for {board_role} row {irow}, skipping plot.")
            continue

        # 2. Get unique subgroups and categories for plotting
        unique_categories = natsorted(df_summary['category'].unique())
        n_categories = len(unique_categories)

        if n_categories == 0:
            continue # Skip if, for some reason, categories are empty

        # Offsets for categories (e.g., -0.2, 0, 0.2)
        # We create a small "dodge" for each category
        offsets = np.linspace(-0.2, 0.2, n_categories)
        category_offset_map = {category: offset for category, offset in zip(unique_categories, offsets)}

        # Get colors for categories
        colors = plt.cm.viridis(np.linspace(0, 1, n_categories))
        category_color_map = {category: color for category, color in zip(unique_categories, colors)}

        # 4. Create the plot
        fig, ax = plt.subplots(figsize=(25, 7))
        hep.cms.text(loc=0, ax=ax, text="ETL ETROC Test Beam", fontsize=20)

        # Plot the points for each category one by one
        for category in unique_categories:
            df_cat = df_summary[df_summary['category'] == category]

            if df_cat.empty:
                continue

            x_base = df_cat['subgroup'].map(subgroup_x_map)
            offset = category_offset_map[category]
            x_final = x_base + offset

            # Plot this category's points
            ax.errorbar(
                x=x_final,
                y=df_cat[res_col],
                yerr=df_cat[err_col],
                fmt='o',  # 'o' specifies a circular marker
                linestyle='None', # No connecting line
                capsize=5,
                markersize=8,
                color=category_color_map[category],
                label=category # Label for the legend
            )

        # 5. Format the plot
        ax.set_ylabel('Time resolution [ps]')
        ax.set_ylim(10, 90)
        ax.set_xticks(ticks=list(subgroup_x_map.values()), labels=list(subgroup_x_map.keys()))

        # Loop until the second-to-last tick
        for i in range(len(tick_positions) - 1):
            line_pos = tick_positions[i] + 0.5
            ax.axvline(x=line_pos, color='black', linestyle='dashed', linewidth=1)

        # Plot threshold lines for 'Path 1'
        thres_df = df_summary.loc[df_summary['category'] == 'Path 1']
        for _, row in thres_df.iterrows():
            x_pos = subgroup_x_map.get(row['subgroup'])
            if x_pos is None:
                continue

            ax.hlines(
                y=row[res_col], # <-- Use variable
                xmin=x_pos - 0.5,
                xmax=x_pos + 0.5,
                color='red',
                linestyle='solid',
                linewidth=1
            )

        ax.set_title(f'{loc_title}\n{selected_fig_config['title']} - Row {irow}', loc='right', fontsize=20)
        ax.set_xlim(-0.5, 15.5)
        ax.legend(title='Ordered paths', bbox_to_anchor=(1, 1), loc='upper left')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        fig.tight_layout() # Adjust layout to make room for the legend
        plt.minorticks_off()


## --------------------------------------
def plot_avg_resolution_per_row(
    input_df: pd.DataFrame,
    board_role: str,
    tb_loc: str,
    fig_config: dict,
    ylims: list[float] = [None, None],
):

    row_col = f'row_{board_role}'
    col_col = f'col_{board_role}'
    res_col = f'res_{board_role}'
    err_col = f'err_{board_role}'

    loc_title = load_fig_title(tb_loc)

    for _, val in fig_config.items():
        if val['role'] == board_role:
            selected_fig_config = val
            break

    # All the data processing happens in one line
    df_proc, _, _ = preprocess_ranking_data(
        input_df,
        row_col,
        col_col,
        rank_by_col='nevt',
        num_cols=16
    )

    summary_array = []
    for irow in range(0, 16):
        thres_df = df_proc.loc[(df_proc['category'] == 'Path 1') & (df_proc[row_col] == irow)]

        if thres_df.empty:
            summary_array.append((irow, np.NaN, np.NaN))
        else:
            summary_array.append((irow, np.average(thres_df[res_col], weights=1/thres_df[err_col]**2), np.sqrt(1/(np.sum(1/thres_df[err_col]**2)))))

    summary_df = pd.DataFrame(summary_array, columns=['row', res_col, err_col])

    fig, ax = plt.subplots(figsize=(12, 10))
    hep.cms.text(loc=0, ax=ax, text="ETL ETROC Test Beam", fontsize=20)
    ax.set_title(f'{loc_title}\n{selected_fig_config['title']}', loc='right', fontsize=17)

    ax.errorbar(
        x=summary_df['row'],
        y=summary_df[res_col],
        yerr=summary_df[err_col],
        fmt='o',  # 'o' specifies a circular marker
        linestyle='None', # No connecting line
        capsize=5,
        markersize=8,
        # label=category # Label for the legend
    )

    ax.set_xticks(ticks=range(16))
    ax.set_xlabel('Row')
    ax.set_ylabel('Avg. time resolution [ps]')
    ax.grid(axis='y')
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylim(ylims[0], ylims[1])

    fig.tight_layout()

## --------------------------------------
def calculate_weighted_mean_std_for_every_pixel(
        input_df: pd.DataFrame,
        board_role: str,
):

    row_col = f'row_{board_role}'
    col_col = f'col_{board_role}'
    res_col = f'res_{board_role}'
    err_col = f'err_{board_role}'

    ranked_df, _, _ = preprocess_ranking_data(input_df, row_col, col_col)
    ranked_df = ranked_df[[row_col, col_col, res_col, err_col, 'nevt']].reset_index(drop=True)

    # Calculate weight
    ranked_df['weight'] = 1 / (ranked_df['err_dut'] ** 2)

    # Formula: value * weight
    ranked_df['weighted_res'] = ranked_df['res_dut'] * ranked_df['weight']

    # 2. Group by row and col, then sum the components
    grouped = ranked_df.groupby([row_col, col_col])[['weighted_res', 'weight']].sum()

    # 3. Calculate final metrics from the sums
    # Mean = Sum(val * w) / Sum(w)
    grouped['weighted_mean'] = grouped['weighted_res'] / grouped['weight']

    # Error = sqrt(1 / Sum(w))
    grouped['weighted_error'] = np.sqrt(1 / grouped['weight'])

    # 4. Clean up result
    final_df = grouped[['weighted_mean', 'weighted_error']].reset_index()
    final_df.columns = [row_col, col_col, res_col, err_col]

    return final_df

## --------------------------------------
# def plot_TDC_correlation_scatter_matrix(
#         input_df: pd.DataFrame,
#         chip_names: list[str],
#         single_hit: bool = False,
#         colinear: bool = False,
#         colinear_cut: int = 1,
#         save: bool = False,
#     ):

#     import plotly.express as px

#     input_df['identifier'] = input_df.groupby(['evt', 'board']).cumcount()
#     board_ids = input_df['board'].unique()
#     val_names = [f'toa_{board_ids[0]}', f'toa_{board_ids[1]}', f'tot_{board_ids[0]}', f'tot_{board_ids[1]}', f'cal_{board_ids[0]}', f'cal_{board_ids[1]}']
#     val_labels = {
#         f'toa_{board_ids[0]}':f'TOA_{chip_names[int(board_ids[0])]}',
#         f'toa_{board_ids[1]}':f'TOA_{chip_names[int(board_ids[1])]}',
#         f'tot_{board_ids[0]}':f'TOT_{chip_names[int(board_ids[0])]}',
#         f'tot_{board_ids[1]}':f'TOT_{chip_names[int(board_ids[1])]}',
#         f'cal_{board_ids[0]}':f'CAL_{chip_names[int(board_ids[0])]}',
#         f'cal_{board_ids[1]}':f'CAL_{chip_names[int(board_ids[1])]}',
#     }
#     extra_tag = ''

#     if single_hit:
#         extra_tag = '_singleHit'
#         input_df['count'] = (0.5*input_df.groupby('evt')['evt'].transform('count')).astype(int)
#         new_df = input_df.pivot(index=['evt', 'identifier'], columns=['board'], values=['row', 'col', 'toa', 'tot', 'cal', 'count'])
#         new_df.columns = ['{}_{}'.format(x, y) for x, y in new_df.columns]
#         new_df['single_hit'] = (new_df[f'count_{board_ids[0]}'] == 1)
#         new_df = new_df.sort_values(by='single_hit', ascending=False) # Make sure True always draw first

#         fig = px.scatter_matrix(
#             new_df.reset_index(),
#             dimensions=val_names,
#             color='single_hit',
#             labels=val_labels,
#             width=1920,
#             height=1080,
#         )

#     elif colinear:
#         extra_tag = '_colinear_pixels'
#         new_df = input_df.pivot(index=['evt', 'identifier'], columns=['board'], values=['row', 'col', 'toa', 'tot', 'cal'])
#         new_df.columns = ['{}_{}'.format(x, y) for x, y in new_df.columns]
#         new_df['colinear'] = (abs(new_df[f'row_{board_ids[0]}']-new_df[f'row_{board_ids[1]}']) <= colinear_cut) & (abs(new_df[f'col_{board_ids[0]}']-new_df[f'col_{board_ids[1]}']) <= colinear_cut)
#         new_df = new_df.sort_values(by='colinear', ascending=False) # Make sure True always draw first

#         fig = px.scatter_matrix(
#             new_df.reset_index(),
#             dimensions=val_names,
#             color='colinear',
#             labels=val_labels,
#             width=1920,
#             height=1080,
#         )

#     else:
#         new_df = input_df.pivot(index=['evt', 'identifier'], columns=['board'], values=['row', 'col', 'toa', 'tot', 'cal'])
#         new_df.columns = ['{}_{}'.format(x, y) for x, y in new_df.columns]
#         fig = px.scatter_matrix(
#             new_df,
#             dimensions=val_names,
#             labels=val_labels,
#             width=1920,
#             height=1080,
#         )

#     fig.update_traces(
#         diagonal_visible=False,
#         showupperhalf=False,
#         marker = {'size': 3},
#     )

#     for k in range(len(fig.data)):
#         fig.data[k].update(
#             selected = dict(
#                 marker = dict(
#                 )
#             ),
#             unselected = dict(
#                 marker = dict(
#                     color="grey"
#                 )
#             ),
#         )

#     if save:
#         fig.write_html(
#             'scatter_matrix_{}_vs_{}{}.html'.format(chip_names[board_ids[0]], chip_names[board_ids[1]], extra_tag),
#             full_html = False,
#             include_plotlyjs = 'cdn',
#         )
#     else:
#         fig.show()

## --------------- Plotting -----------------------
