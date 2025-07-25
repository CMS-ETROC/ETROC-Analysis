import pandas as pd
import numpy as np
import hist
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import mplhep as hep
hep.style.use('CMS')

from pathlib import Path
from matplotlib import rcParams
rcParams["axes.formatter.useoffset"] = False
rcParams["axes.formatter.use_mathtext"] = False

__all__ = [
    'return_hist',
    'return_event_hist',
    'return_crc_hist',
    'return_hist_pivot',
    'plot_BL_and_NW',
    'plot_number_of_fired_board',
    'plot_number_of_hits_per_event',
    'plot_2d_nHits_nBoard',
    'plot_occupany_map',
    'plot_3d_occupany_map',
    'plot_TDC_summary_table',
    'plot_1d_TDC_histograms',
    'plot_1d_event_CRC_histogram',
    'plot_1d_CRC_histogram',
    'plot_correlation_of_pixels',
    'plot_difference_of_pixels',
    'plot_distance',
    'plot_TOA_correlation',
    'plot_TWC',
    'plot_resolution_with_pulls',
    'plot_resolution_table',
]

## --------------- Plotting -----------------------
def save_plot(fig, save_dir, file_name):
    """Helper function to finalize and save a plot."""
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"{file_name}.png")
    fig.savefig(save_dir / f"{file_name}.pdf")
    plt.close(fig)

def load_fig_title(
    tb_loc:str
):
    """Load figure title (beam info and testing location)

    Parameters
    ----------
    tb_loc: str,
        Location of the test beam.
        Input argument for test beam facility
            1. 'desy',
            2. 'cern',
            3. 'fnal',
        Input argument for SEU test facility
            1. 'northwestern',
            2. 'louvain-{ion type}'
                - ion type: C, Ne, Al, Ar, Cr, Ni, Kr, Rh, Xe
    """
    if tb_loc == 'desy':
        plot_title = r'4 GeV $e^{-}$ at DESY TB'
    elif tb_loc == 'cern':
        plot_title = r'120 GeV (1/3 p; 2/3 $\pi^{+}$) at CERN SPS H6'
    elif tb_loc == 'cern_mu':
        plot_title = r'$\mu^{\pm}$ at CERN SPS H6'
    elif tb_loc == 'cern_h8':
        plot_title = r'180 GeV ($\pi^{+}$) at CERN SPS H8'
    elif tb_loc == 'fnal':
        plot_title = r'120 GeV p at Fermilab TB'
    elif tb_loc == 'northwestern':
        plot_title = r'217 MeV p at Northwestern Medicine Proton Center'
    elif tb_loc == 'wh14':
        plot_title = 'Wilson Hall 14th floor lab'
    # The assumption for louvain is the the tb_loc will specify location and ion with the following format:
    # louvain-Kr  - for example for louvain with Krypton ion beam
    # louvain-Xe  - for example for louvain with Xenon ion beam
    elif tb_loc[:7] == 'louvain':
        ion = tb_loc[8:]
        energy = "xx"
        if ion == "C":
            energy = 131
        elif ion == "Ne":
            energy = 238
        elif ion == "Al":
            energy = 250
        elif ion == "Ar":
            energy = 353
        elif ion == "Cr":
            energy = 505
        elif ion == "Ni":
            energy = 582
        elif ion == "Kr":
            energy = 769
        elif ion == "Rh":
            energy = 957
        elif ion == "Xe":
            energy = 995
        plot_title = rf'{energy} MeV {ion} at Heavy Ion Facility'
    else:
        print('Unknown location. Please add info into the function. Return empty string')
        plot_title = ""

    return plot_title

## --------------------------------------
def return_hist(
        input_df: pd.DataFrame,
        board_info: dict | None = None, # Prioritized argument
        board_ids: list[int] | None = None,
        board_names: list[str] | None = None,
        hist_bins: list = [70, 64, 64]
):
    if board_info:

        h = {val['short']: hist.Hist(hist.axis.Regular(hist_bins[0], 120, 260, name="CAL", label="CAL [LSB]"),
                hist.axis.Regular(hist_bins[1], 0, 512,  name="TOT", label="TOT [LSB]"),
                hist.axis.Regular(hist_bins[2], 0, 1024, name="TOA", label="TOA [LSB]"),
                hist.axis.Regular(4, 0, 3, name="EA", label="EA"),
            )
        for _, val in board_info.items()}

        for ikey, val in board_info.items():
            tmp_df = input_df.loc[input_df['board'] == ikey]
            h[val['short']].fill(tmp_df['cal'].values, tmp_df['tot'].values, tmp_df['toa'].values, tmp_df['ea'].values)

    elif board_ids and board_names:

        if len(board_ids) != len(board_names):
            raise ValueError("The lists 'board_ids' and 'board_names' must have the same length.")

        h = {board_names[board_idx]: hist.Hist(hist.axis.Regular(hist_bins[0], 120, 260, name="CAL", label="CAL [LSB]"),
                    hist.axis.Regular(hist_bins[1], 0, 512,  name="TOT", label="TOT [LSB]"),
                    hist.axis.Regular(hist_bins[2], 0, 1024, name="TOA", label="TOA [LSB]"),
                    hist.axis.Regular(4, 0, 3, name="EA", label="EA"),
            )
        for board_idx in range(len(board_ids))}

        for board_idx in range(len(board_ids)):
            tmp_df = input_df.loc[input_df['board'] == board_ids[board_idx]]
            h[board_names[board_idx]].fill(tmp_df['cal'].values, tmp_df['tot'].values, tmp_df['toa'].values, tmp_df['ea'].values)

    else:
        raise ValueError("You must provide either 'board_info' or both 'board_ids' and 'board_names'.")

    return h

## --------------------------------------
def return_event_hist(
        input_df: pd.DataFrame,
):
    h = hist.Hist(hist.axis.Regular(8, 0, 8, name="HA", label="Hamming Count"),
                  hist.axis.Regular(2, 0, 2, name="CRC_mismatch", label="CRC Mismatch"))

    h.fill(input_df["hamming_count"].values, input_df["CRC_mismatch"].values)

    return h

## --------------------------------------
def return_crc_hist(
        input_df: pd.DataFrame,
        chipNames: list[str],
        chipLabels: list[int],
):
    h = {chipNames[board_idx]: hist.Hist(
            hist.axis.Regular(2, 0, 2, name="CRC_mismatch", label="CRC Mismatch"),
        )
    for board_idx in range(len(chipLabels))}


    for board_idx in range(len(chipLabels)):
        tmp_df = input_df.loc[input_df['board'] == chipLabels[board_idx]]
        h[chipNames[board_idx]].fill(tmp_df['CRC_mismatch'].values)

    return h

## --------------------------------------
def return_hist_pivot(
        input_df: pd.DataFrame,
        board_ids: list[int],
        board_names: list[str],
        hist_bins: list = [50, 64, 64]
):
    h = {board_names[board_idx]: hist.Hist(hist.axis.Regular(hist_bins[0], 140, 240, name="CAL", label="CAL [LSB]"),
                hist.axis.Regular(hist_bins[1], 0, 512,  name="TOT", label="TOT [LSB]"),
                hist.axis.Regular(hist_bins[2], 0, 1024, name="TOA", label="TOA [LSB]"),
        )
    for board_idx in range(len(board_ids))}

    for idx, board_id in enumerate(board_ids):
        h[board_names[idx]].fill(input_df['cal'][board_id].values, input_df['tot'][board_id].values, input_df['toa'][board_id].values)

    return h

## --------------------------------------
def plot_BL_and_NW(
        run_time_df: pd.DataFrame,
        which_run: int,
        baseline_df: pd.DataFrame,
        config_dict: dict,
        which_val: str,
        save_mother_dir: Path | None = None,
    ):
    """Make Basline of Noise Width 2d map.

    Parameters
    ----------
    run_time_df: pd.DataFrame,
        Include TB run information in csv. Check reading_history/tb_run_info.
    which_run: int,
        Run number.
    baseline_df: pd.DataFrame,
        Baseline and Noise Width dataframe. Saved in SQL format.
    config_dict: dict,
        A dictionary of user config. Should include title for plot, chip_type (determine the color range), and channel (determine HV value).
    which_val: str,
        Either which_val = 'baseline' or which_val = 'noise_width.
    save_mother_dir: Path, optional
        Plot will be saved at save_mother_dir/'occupancy_map'.
    """

    cut_time1 = run_time_df.loc[run_time_df['Run'] == which_run-1 ,'Start_Time'].values[0]
    cut_time2 = run_time_df.loc[run_time_df['Run'] == which_run ,'Start_Time'].values[0]

    selected_run_df = baseline_df.loc[(baseline_df['timestamp'] > cut_time1) & (baseline_df['timestamp'] < cut_time2)]

    single_run_df = run_time_df.loc[run_time_df['Run'] == which_run, ["HV0", "HV1", "HV2", "HV3"]]
    HVs = single_run_df.iloc[0, 0:].to_numpy()

    if selected_run_df.shape[0] != 1024:
        selected_run_df = selected_run_df.loc[selected_run_df.groupby(['row', 'col', 'chip_name'])['timestamp'].idxmax()].reset_index(drop=True)

    for iboard in selected_run_df['chip_name'].unique():
        tmp_df = selected_run_df.loc[selected_run_df['chip_name']==iboard]

        # Create a pivot table to reshape the data for plotting
        pivot_table = tmp_df.pivot(index='row', columns='col', values=which_val)

        if pivot_table.empty:
            continue

        if (pivot_table.shape[0] != 16) or (pivot_table.shape[1]!= 16):
            pivot_table = pivot_table.reindex(pd.Index(np.arange(0,16), name='')).reset_index()
            pivot_table = pivot_table.reindex(columns=np.arange(0,16))
            pivot_table = pivot_table.fillna(-1)

        # # Create a heatmap to visualize the count of hits
        fig, ax = plt.subplots(dpi=100, figsize=(12, 12))

        if which_val == 'baseline':
            if config_dict[iboard]['chip_type'] == "T":
                im = ax.imshow(pivot_table, interpolation="nearest", vmin=300, vmax=500)
            elif config_dict[iboard]['chip_type'] == "F":
                im = ax.imshow(pivot_table, interpolation="nearest", vmin=50, vmax=250)
            # # Add color bar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, extend='both')
            cbar.set_label('Baseline', fontsize=25)
            cbar.ax.tick_params(labelsize=18)
        elif which_val == 'noise_width':
            im = ax.imshow(pivot_table, interpolation="nearest", vmin=0, vmax=16)

            # # Add color bar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Noise Width', fontsize=25)
            cbar.ax.tick_params(labelsize=18)

        for i in range(16):
            for j in range(16):
                value = pivot_table.iloc[i, j]
                if value == -1: continue
                text = str("{:.0f}".format(value))
                plt.text(j, i, text, va='center', ha='center', color='white', fontsize=14)

        hep.cms.text(loc=0, ax=ax, text="ETL ETROC Test Beam", fontsize=18)
        ax.set_xlabel('Column', fontsize=25)
        ax.set_ylabel('Row', fontsize=25)
        ticks = range(0, 16)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_title(f"{config_dict[iboard]['plot_title'].replace('_', ' ')} HV{HVs[config_dict[iboard]['channel']]}V", loc="right", size=16)
        ax.tick_params(axis='x', which='both', length=5, labelsize=17)
        ax.tick_params(axis='y', which='both', length=5, labelsize=17)
        ax.invert_xaxis()
        ax.invert_yaxis()
        plt.minorticks_off()
        plt.tight_layout()

        if save_mother_dir is not None:
            pass

## --------------------------------------
def plot_number_of_fired_board(
        input_df: pd.DataFrame,
        tb_loc: str,
        fig_tag: str = '',
        do_logy: bool = False,
        save_mother_dir: Path | None = None,
    ):
    """Make a plot of number of fired boards in events.

    Parameters
    ----------
    input_df: pd.DataFrame,
        Input dataframe.
    tb_loc: str,
        Test Beam location for the title. Available argument: desy, cern, fnal.
    fig_tag: str, optional
        Additional figure tag to put in the title.
    do_logy: str, optional
        Log y-axis.
    save_mother_dir: Path, optional
        Plot will be saved at save_mother_dir/'misc'.
    """

    plot_title = load_fig_title(tb_loc)
    h = hist.Hist(hist.axis.Regular(5, 0, 5, name="nBoards", label="nBoards"))
    h.fill(input_df.groupby('evt')['board'].nunique())

    fig = plt.figure(figsize=(11,10))
    gs = fig.add_gridspec(1,1)
    ax = fig.add_subplot(gs[0,0])
    hep.cms.text(loc=0, ax=ax, text="ETL ETROC Test Beam", fontsize=18)
    ax.set_title(f"{plot_title} {fig_tag}", loc="right", size=16)
    ax.xaxis.label.set_fontsize(25)
    ax.yaxis.label.set_fontsize(25)
    h.plot1d(ax=ax, lw=2)
    ax.get_yaxis().get_offset_text().set_position((-0.05, 0))
    if do_logy:
        ax.set_yscale('log')
    plt.tight_layout()

    if save_mother_dir is not None:
        save_dir = save_mother_dir / 'misc'
        save_dir.mkdir(exist_ok=True)
        fig.savefig(save_dir / f"number_of_fired_board.png")
        fig.savefig(save_dir / f"number_of_fired_board.pdf")
        plt.close(fig)

## --------------------------------------
def plot_number_of_hits_per_event(
        input_df: pd.DataFrame,
        tb_loc: str,
        board_names: list[str],
        do_logy: bool = False,
        save_mother_dir: Path | None = None,
    ):
    """Make a plot of number of hits per event.

    Parameters
    ----------
    input_df: pd.DataFrame,
        Input dataframe.
    tb_loc: str,
        Test Beam location for the title. Available argument: desy, cern, fnal
    board_names: list[str],
        A list of board names.
    do_logy: str, optional
        Log y-axis.
    save_mother_dir: Path, optional
        Plot will be saved at save_mother_dir/'misc'.
    """

    plot_title = load_fig_title(tb_loc)
    hit_df = input_df.groupby(['evt', 'board']).size().unstack(fill_value=0)
    hists = {}

    for key in hit_df.columns:
        max_hit = hit_df[key].unique().max()
        hists[key] = hist.Hist(hist.axis.Regular(max_hit, 0, max_hit, name="nHits", label='Number of Hits'))
        hists[key].fill(hit_df[key])

    for idx, ikey in enumerate(hists.keys()):
        fig, ax = plt.subplots(figsize=(11, 10))
        hep.cms.text(loc=0, ax=ax, text="ETL ETROC Test Beam", fontsize=18)
        hists[ikey].plot1d(ax=ax, lw=2)
        ax.set_title(f"{plot_title} | {board_names[idx]}", loc="right", size=16)
        ax.get_yaxis().get_offset_text().set_position((-0.05, 0))

        if do_logy:
            ax.set_yscale('log')

        plt.tight_layout()
        if save_mother_dir is not None:
            save_dir = save_mother_dir / 'misc'
            save_dir.mkdir(exist_ok=True)
            fig.savefig(save_dir / f"number_of_hits_{board_names[idx]}.png")
            fig.savefig(save_dir / f"number_of_hits_{board_names[idx]}.pdf")
            plt.close(fig)

## --------------------------------------
def plot_2d_nHits_nBoard(
        input_df: pd.DataFrame,
        fig_titles: list[str],
        fig_tag: str = '',
        bins: int = 15,
        hist_range: tuple = (0, 15),

    ):
    nboard_df = input_df.groupby('evt')['board'].nunique()
    hit_df = input_df.groupby(['evt', 'board']).size().unstack(fill_value=0)
    hit_df.dropna(subset=[0], inplace=True)
    hists = {}

    for key in hit_df.columns:
        hists[key] = hist.Hist(
            hist.axis.Regular(bins, hist_range[0], hist_range[1], name="nHits", label='nHits'),
            hist.axis.Regular(5, 0, 5, name="nBoards", label="nBoards")
        )
        hists[key].fill(hit_df[key], nboard_df)

    fig = plt.figure(dpi=100, figsize=(30,13))
    gs = fig.add_gridspec(2,2)

    for i, plot_info in enumerate(gs):

        if i not in hists.keys():
            continue

        ax = fig.add_subplot(plot_info)
        hep.cms.text(loc=0, ax=ax, text="ETL ETROC Test Beam", fontsize=20)
        hep.hist2dplot(hists[i], ax=ax, norm=colors.LogNorm())
        ax.set_title(f"{fig_titles[i]} {fig_tag}", loc="right", size=18)

    plt.tight_layout()
    del hists, hit_df, nboard_df

## --------------------------------------
def plot_occupany_map(
        input_df: pd.DataFrame,
        tb_loc: str,
        board_info: dict | None = None, # Prioritized argument
        board_ids: list[int] | None = None,
        board_names: list[str] | None = None,
        extra_cms_title: str = 'ETL ETROC Test Beam',
        fname_tag: str = '',
        save_mother_dir: Path | None = None,
    ):
    """Make occupancy plot.

    Parameters
    ----------
    input_df: pd.DataFrame,
        Pandas dataframe of data.
    tb_loc: str,
        Test Beam location for the title. Available argument: desy, cern, fnal.
    board_info: dict, optional
        Primary way to pass board data. A dictionary where keys are board IDs
        and values are another dict containing board info, e.g., {'title': 'Board_Name'}.
        If provided, 'board_ids' and 'board_names' are ignored.
    board_ids: list[int], optional
        A list of integer board IDs. Used if 'board_info' is not provided.
    board_names: list[str], optional
        A list of board names for plot titles. Used if 'board_info' is not provided.
    extra_cms_title: str,
        Default is "ETL ETROC Test Beam".
    fname_tag: str, optional
        Tag to be added to the output filename.
    save_mother_dir: Path, optional
        Plot will be saved at save_mother_dir/'occupancy_map'.
    """

    cmap = plt.get_cmap('viridis').copy()
    cmap.set_under(color='lightgrey')

    loc_title = load_fig_title(tb_loc)
    hits_df = input_df.groupby(['board', 'col', 'row'])['evt'].count().reset_index(name='hits')

    board_configs = {}
    if board_info:
        board_configs = board_info
    elif board_ids and board_names:
        if len(board_ids) != len(board_names):
            raise ValueError("The lists 'board_ids' and 'board_names' must have the same length.")
        for idx, board_id in enumerate(board_ids):
            board_configs[board_id] = {
                'name': board_names[idx],
            }
    else:
        raise ValueError("You must provide either 'board_info' or both 'board_ids' and 'board_names'.")

    for ikey, config in board_configs.items():
        board_hits = hits_df.loc[hits_df['board'] == ikey]
        if board_hits.empty:
            print(f"Skipping Board ID {ikey}: No data available.")
            continue

        pivot_table = board_hits.pivot_table(index='row', columns='col', values='hits')

        # Ensure the pivot table is always 16x16, filling missing data
        all_indices = np.arange(16)
        pivot_table = pivot_table.reindex(index=all_indices, columns=all_indices).fillna(-1)

        # --- Plotting ---
        fig, ax = plt.subplots(dpi=100, figsize=(12, 12))
        im = ax.imshow(pivot_table, cmap=cmap, interpolation="nearest", vmin=0)

        max_val = pivot_table.values.max()
        min_val = pivot_table.values.min()
        text_color_threshold = min_val + (max_val - min_val) * 0.45
        for i in range(16):
            for j in range(16):
                value = pivot_table.iloc[i, j]
                if value == -1: continue # Don't label cells with no data
                text_color = 'black' if value > text_color_threshold else 'white'
                ax.text(j, i, f"{value:.0f}", va='center', ha='center', color=text_color, fontsize=12)

        # --- Styling and Labels ---
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Hits', fontsize=25)
        cbar.ax.tick_params(labelsize=18)

        hep.cms.text(loc=0, ax=ax, text=extra_cms_title, fontsize=18)
        ax.set_title(f"{loc_title}\n{config['name']}", loc="right", size=16)
        ax.set_xlabel('Column', fontsize=25)
        ax.set_ylabel('Row', fontsize=25)

        ax.set_xticks(np.arange(16))
        ax.set_yticks(np.arange(16))
        ax.tick_params(axis='both', which='major', length=5, labelsize=17)
        ax.invert_xaxis()
        ax.invert_yaxis()
        plt.minorticks_off()
        plt.tight_layout()

        # --- Saving ---
        if save_mother_dir:
            save_dir = Path(save_mother_dir) / 'occupancy_map'
            save_dir.mkdir(parents=True, exist_ok=True)
            output_base = save_dir / f"occupancy_{config['name']}_{fname_tag}"
            fig.savefig(f"{output_base}.png")
            fig.savefig(f"{output_base}.pdf")
            plt.close(fig) # Close the figure to free memory

## --------------------------------------
def plot_3d_occupany_map(
        input_df: pd.DataFrame,
        board_ids: list[int],
        board_names: list[str],
        tb_loc: str,
        fname_tag: str = '',
        save_mother_dir: Path | None = None,
    ):
    """Make 3D occupancy plot.

    Parameters
    ----------
    input_df: pd.DataFrame,
        Pandas dataframe of data.
    board_ids: list[int],
        A list of integer (board ID) that wants to make plots.
    board_names: list[str],
        A list of board name that will use for the file name.
    tb_loc: str,
        Test Beam location for the title. Available argument: desy, cern, fnal.
    fname_tag: str, optional
        Draw boundary cut in the plot.
    save_mother_dir: Path, optional
        Plot will be saved at save_mother_dir/'occupancy_map'.
    """

    plot_title = load_fig_title(tb_loc)
    hits_count_by_col_row_board = input_df.groupby(['col', 'row', 'board'])['evt'].count().reset_index()

    # Rename the 'evt' column to 'hits'
    hits_count_by_col_row_board = hits_count_by_col_row_board.rename(columns={'evt': 'hits'})

    for idx ,board_id in enumerate(board_ids):
        # Create a pivot table to reshape the data for plotting
        pivot_table = hits_count_by_col_row_board[hits_count_by_col_row_board['board'] == board_id].pivot_table(
            index='row',
            columns='col',
            values='hits',
            fill_value=0  # Fill missing values with 0 (if any)
        )
        fig = plt.figure(figsize=(11, 10))
        ax = fig.add_subplot(111, projection='3d')
        hep.cms.text(loc=0, ax=ax, text="ETL ETROC Test Beam", fontsize=18)

        # Create a meshgrid for the 3D surface
        x, y = np.meshgrid(np.arange(16), np.arange(16))
        z = pivot_table.values
        dx = dy = 0.75  # Width and depth of the bars

        # Create a 3D surface plot
        ax.bar3d(x.flatten(), y.flatten(), np.zeros_like(z).flatten(), dx, dy, z.flatten(), shade=True)

        # Customize the 3D plot settings as needed
        ax.set_xlabel('COL', fontsize=15, labelpad=15)
        ax.set_ylabel('ROW', fontsize=15, labelpad=15)
        ax.set_zlabel('Hits', fontsize=15, labelpad=-35)
        ax.invert_xaxis()
        ticks = range(0, 16)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticks(ticks=range(16), labels=[], minor=True)
        ax.set_yticks(ticks=range(16), labels=[], minor=True)
        ax.tick_params(axis='x', labelsize=8)  # You can adjust the 'pad' value
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='z', labelsize=8)
        ax.set_title(f"{plot_title} | {board_names[idx]}", fontsize=14, loc='right')
        plt.tight_layout()

        if save_mother_dir is not None:
            save_dir = save_mother_dir / 'occupancy_map'
            save_dir.mkdir(exist_ok=True)
            fig.savefig(save_dir / f"3D_occupancy_{board_names[board_id]}_{fname_tag}.png")
            fig.savefig(save_dir / f"3D_occupancy_{board_names[board_id]}_{fname_tag}.pdf")
            plt.close(fig)

## --------------------------------------
def plot_TDC_summary_table(
        input_df: pd.DataFrame,
        board_ids: list[int],
        varname: str,
        avg_min_bound: float = 0.0,
        avg_max_bound: float = 1024.0,
        std_min_bound: float = 0.0,
        std_max_bound: float = 10.0,
    ):
    """Make plots of TDC variable mean and std table per pixel in 2D map.

    Parameters
    ----------
    input_df: pd.DataFrame,
        Pandas dataframe of data.
    board_ids: list[int],
        Board IDs in integer.
    varname: str,
        TDC variable name. Must be all lower case: 'toa', 'tot', or 'cal'.
    avg_min_bound: float,
        A number to set minimum boundary for mean table plot.
    avg_max_bound: float,
        A number to set maximum boundary for mean table plot.
    std_min_bound: float,
        A number to set minimum boundary for std table plot.
    std_max_bound: float,
        A number to set maximum boundary for std table plot.
    """

    from matplotlib import colormaps
    cmap = colormaps['viridis']
    cmap.set_under(color='lightgrey')

    for id in board_ids:

        if input_df[input_df['board'] == id].empty:
            continue

        sum_group = input_df[input_df['board'] == id].groupby(["col", "row"]).agg({varname:['mean','std']})
        sum_group.columns = sum_group.columns.droplevel()
        sum_group.reset_index(inplace=True)

        table_mean = sum_group.pivot_table(index='row', columns='col', values='mean', fill_value=-1)
        table_mean = table_mean.round(1)

        table_mean = table_mean.reindex(pd.Index(np.arange(0,16), name='')).reset_index()
        table_mean = table_mean.reindex(columns=np.arange(0,16))

        table_std = sum_group.pivot_table(index='row', columns='col', values='std', fill_value=-1)
        table_std = table_std.round(2)

        table_std = table_std.reindex(pd.Index(np.arange(0,16), name='')).reset_index()
        table_std = table_std.reindex(columns=np.arange(0,16))

        plt.rcParams["xtick.major.size"] = 2.5
        plt.rcParams["ytick.major.size"] = 2.5
        plt.rcParams['xtick.minor.visible'] = False
        plt.rcParams['ytick.minor.visible'] = False

        fig, axes = plt.subplots(1, 2, figsize=(20, 20))

        im1 = axes[0].imshow(table_mean, cmap=cmap, vmin=avg_min_bound, vmax=avg_max_bound)
        im2 = axes[1].imshow(table_std, cmap=cmap, vmin=std_min_bound, vmax=std_max_bound)

        hep.cms.text(loc=0, ax=axes[0], text="ETL ETROC Test Beam", fontsize=25)
        hep.cms.text(loc=0, ax=axes[1], text="ETL ETROC Test Beam", fontsize=25)

        axes[0].set_title(f'{varname.upper()} Mean', loc="right")
        axes[1].set_title(f'{varname.upper()} Std', loc="right")

        axes[0].set_xticks(np.arange(0,16))
        axes[0].set_yticks(np.arange(0,16))
        axes[1].set_xticks(np.arange(0,16))
        axes[1].set_yticks(np.arange(0,16))

        # i for col, j for row
        for i in range(16):
            for j in range(16):
                if np.isnan(table_mean.iloc[i,j]) or table_mean.iloc[i,j] < 0.:
                    continue
                text_color = 'black' if table_mean.iloc[i,j] > 0.5*(table_mean.stack().max() + table_mean.stack().min()) else 'white'
                axes[0].text(j, i, table_mean.iloc[i,j], ha="center", va="center", rotation=45, fontweight="bold", fontsize=12, color=text_color)

        for i in range(16):
            for j in range(16):
                if np.isnan(table_std.iloc[i,j]) or table_std.iloc[i,j] < 0.:
                    continue
                text_color = 'black' if table_std.iloc[i,j] > 0.5*(table_std.stack().max() + table_std.stack().min()) / 2 else 'white'
                axes[1].text(j, i, table_std.iloc[i,j], ha="center", va="center", rotation=45, color=text_color, fontweight="bold", fontsize=12)

        axes[0].invert_xaxis()
        axes[0].invert_yaxis()
        axes[1].invert_xaxis()
        axes[1].invert_yaxis()

        plt.minorticks_off()
        plt.tight_layout()

## --------------------------------------
def plot_1d_TDC_histograms(
    input_hist: dict,
    tb_loc: str,
    extra_cms_title: str = 'ETL ETROC Test Beam',
    fig_tag: list[str] | None = None,
    slide_friendly: bool = False,
    do_logy: bool = False,
    event_hist: hist.Hist | None = None,
    save_mother_dir: Path | None = None,
    no_errorbar: bool = False,
    tag: str = '',
):
    """Make plots of 1D TDC histograms.

    Parameters
    ----------
    input_hist: dict,
        A dictionary of TDC histograms, which returns from return_hist, return_hist_pivot
    tb_loc: str,
        Test Beam location for the title. Available argument: desy, cern, fnal.
    extra_cms_title: str,
        Default is "ETL ETROC Test Beam". Please change it based on test source.
    fig_tag: str, optional
        Additional board information to show in the plot.
    slide_friendly: bool, optional
        If it is True, draw plots in a single figure. Recommend this option, when you try to add plots on the slides.
    do_logy: bool, optional
        Set log y-axis on 1D histograms.
    event_hist: hist.Hist, optional
        A dictionary of TDC histograms, which returns from return_event_hist
    save_mother_dir: Path, optional
        Plot will be saved at save_mother_dir/'1d_tdc_hists'.
    no_errorbar: bool, optional.
        no_errorbar=False will omit errorbar when plotting.
    tag: str, optional (recommend),
        Additional tag for the file name.
    """

    loc_title = load_fig_title(tb_loc)
    save_dir = save_mother_dir / '1d_tdc_hists' if save_mother_dir else None

    if not slide_friendly:
        for idx, (board_name, ihist) in enumerate(input_hist.items()):
            pass
            for ival in ["CAL", "TOT", "TOA", "EA"]:
                try:
                    fig, ax = plt.subplots(figsize=(11, 10))
                    ax.set_title(loc_title, loc="right", size=16)
                    hep.cms.text(loc=0, ax=ax, text=extra_cms_title, fontsize=18)
                    ihist.project(ival).plot1d(ax=ax, lw=2, yerr=not no_errorbar)
                    ax.xaxis.label.set_fontsize(25)
                    ax.yaxis.label.set_fontsize(25)
                    if fig_tag[idx]:
                        ax.text(0.98, 0.97, fig_tag[idx], transform=ax.transAxes, fontsize=17, verticalalignment='top', horizontalalignment='right')
                    if do_logy:
                        ax.set_yscale('log')
                    plt.tight_layout()
                    if save_dir:
                        save_plot(fig, save_dir, f'{board_name}_{ival}_{tag}')
                except Exception as e:
                    print(f'No {ival} histogram is found: {e}')

            # 2D TOA-TOT plot
            fig, ax = plt.subplots(figsize=(11, 10))
            ax.set_title(loc_title, loc="right", size=16)
            hep.cms.text(loc=0, ax=ax, text=extra_cms_title, fontsize=18)
            hep.hist2dplot(ihist.project("TOA", "TOT")[::2j, ::2j], ax=ax)
            ax.xaxis.label.set_fontsize(25)
            ax.yaxis.label.set_fontsize(25)
            if fig_tag[idx]:
                ax.text(0.98, 0.97, fig_tag[idx], transform=ax.transAxes, fontsize=17, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white'))
            plt.tight_layout()
            if save_dir:
                save_plot(fig, save_dir, f'{board_name}_TOA_TOT_{tag}')

            # Hamming Count plot
            if event_hist:
                fig, ax = plt.subplots(figsize=(11, 10))
                ax.set_title(loc_title, loc="right", size=16)
                hep.cms.text(loc=0, ax=ax, text=extra_cms_title, fontsize=18)
                event_hist.project("HA").plot1d(ax=ax, lw=2, yerr=not no_errorbar)
                ax.xaxis.label.set_fontsize(25)
                ax.yaxis.label.set_fontsize(25)
                if fig_tag[idx]:
                    ax.text(0.98, 0.97, fig_tag[idx], transform=ax.transAxes, fontsize=17, verticalalignment='top', horizontalalignment='right')
                if do_logy:
                    ax.set_yscale('log')
                plt.tight_layout()
                if save_dir:
                    save_plot(fig, save_dir, f'{board_name}_Hamming_Count_{tag}')
    else:
        for idx, (board_name, ihist) in enumerate(input_hist.items()):
            fig = plt.figure(dpi=100, figsize=(30, 13))
            gs = fig.add_gridspec(2, 2)
            plot_vars = ["CAL", "TOA", "TOT"]

            for i, plot_info in enumerate(gs):
                ax = fig.add_subplot(plot_info)
                hep.cms.text(loc=0, ax=ax, text=extra_cms_title, fontsize=18)

                if i < len(plot_vars):
                    ax.set_title(f"{loc_title} | {fig_tag[idx]}", loc="right", size=16)
                    ihist.project(plot_vars[i]).plot1d(ax=ax, lw=2, yerr=not no_errorbar)
                    if do_logy:
                        ax.set_yscale('log')
                else:
                    if event_hist:
                        event_hist.project("HA").plot1d(ax=ax, lw=2, yerr=no_errorbar)
                        if do_logy:
                            ax.set_yscale('log')
                    else:
                        # Hide the original axis frame from the main gridspec
                        ax.set_frame_on(False)
                        ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

                        # 1. Create a sub-grid in the bottom-right cell
                        sub_gs = plot_info.subgridspec(1, 2, width_ratios=[20, 1], wspace=0.1)

                        # 2. Create axes for the plot and the colorbar using the sub-grid
                        ax_2d = fig.add_subplot(sub_gs[0, 0])
                        cax = fig.add_subplot(sub_gs[0, 1])

                        # Set the title on the new plot axis
                        ax_2d.set_title(f"{loc_title} | {fig_tag[idx]}", loc="right", size=16)

                        # 3. Plot and capture the container object
                        artists = ihist.project("TOA", "TOT")[::2j, ::2j].plot2d(
                            ax=ax_2d,
                            cbar=False  # Keep this to prevent automatic resizing
                        )

                        # 4. Manually create the colorbar using the .mesh attribute of the returned object
                        fig.colorbar(artists[0], cax=cax)

            plt.tight_layout()
            if save_dir:
                save_plot(fig, save_dir, f'{board_name}_combined_{tag}')

## --------------------------------------
def plot_1d_event_CRC_histogram(
        input_hist: hist.Hist,
        fig_path: Path = Path('./'),
        save: bool = False,
        tag: str = '',
        fig_tag: str = '',
        do_logy: bool = False,
    ):
    fig = plt.figure(dpi=50, figsize=(20,10))
    gs = fig.add_gridspec(1,1)
    ax = fig.add_subplot(gs[0,0])
    ax.set_title(f"Event CRC Check{fig_tag}", loc="right", size=16)
    hep.cms.text(loc=0, ax=ax, text="ETL ETROC Test Beam", fontsize=18)
    input_hist.project("CRC_mismatch")[:].plot1d(ax=ax, lw=2)
    if do_logy:
        ax.set_yscale('log')
    plt.tight_layout()
    if(save):
        plt.savefig(fig_path/f'Event_CRCCheck_{tag}.pdf')
        plt.clf()
        plt.close(fig)

## ------------------------------------
def plot_1d_CRC_histogram(
        input_hist: hist.Hist,
        chip_name: str,
        chip_figname: str,
        fig_title: str,
        fig_path: Path = Path('./'),
        save: bool = False,
        tag: str = '',
        fig_tag: str = '',
        do_logy: bool = False,
    ):
    fig = plt.figure(dpi=50, figsize=(20,10))
    gs = fig.add_gridspec(1,1)
    ax = fig.add_subplot(gs[0,0])
    ax.set_title(f"{fig_title}, CRC Check{fig_tag}", loc="right", size=25)
    hep.cms.text(loc=0, ax=ax, text="ETL ETROC Test Beam", fontsize=25)
    input_hist[chip_name].project("CRC_mismatch")[:].plot1d(ax=ax, lw=2)
    if do_logy:
        ax.set_yscale('log')
    plt.tight_layout()
    if(save):
        plt.savefig(fig_path/f'{chip_figname}_CRCCheck_{tag}.pdf')
        plt.clf()
        plt.close(fig)

## --------------------------------------
def plot_correlation_of_pixels(
        input_df: pd.DataFrame,
        board_ids: list[int],
        board_name1: str,
        board_name2: str,
        tb_loc: str,
        fname_tag: str = '',
        save_mother_dir: Path | None = None,
    ):
    """Make pixel row-column correlation plot.

    Parameters
    ----------
    input_df: pd.DataFrame,
        Pandas dataframe of data.
    board_ids: list[int],
        A list of integer (board ID) that wants to make plots.
    board_name1: str,
        Board 1 name.
    board_name2: str,
        Board 2 name.
    tb_loc: str,
        Test Beam location for the title. Available argument: desy, cern, fnal.
    fname_tag: str, optional (recommend)
        Additiional tag for the file name.
    save_mother_dir: Path, optional
        Plot will be saved at save_mother_dir/'spatial_correlation'.
    """

    plot_title = load_fig_title(tb_loc)
    axis_name1 = board_name1.replace('_', ' ')
    axis_name2 = board_name2.replace('_', ' ')

    save_dir = save_mother_dir / 'spatial_correlation' if save_mother_dir else None

    h_row = hist.Hist(
        hist.axis.Regular(16, 0, 16, name='row1', label=f'Row of {axis_name1}'),
        hist.axis.Regular(16, 0, 16, name='row2', label=f'Row of {axis_name2}'),
    )
    h_col = hist.Hist(
        hist.axis.Regular(16, 0, 16, name='col1', label=f'Column of {axis_name1}'),
        hist.axis.Regular(16, 0, 16, name='col2', label=f'Column of {axis_name2}'),
    )

    h_row.fill(input_df[f'row_{board_ids[0]}'].values, input_df[f'row_{board_ids[1]}'].values)
    h_col.fill(input_df[f'col_{board_ids[0]}'].values, input_df[f'col_{board_ids[1]}'].values)

    location = np.arange(0, 16) + 0.5
    tick_labels = np.char.mod('%d', np.arange(0, 16))
    fig, ax = plt.subplots(1, 2, dpi=100, figsize=(23, 11))

    hep.hist2dplot(h_row, ax=ax[0], norm= colors.LogNorm())
    hep.cms.text(loc=0, ax=ax[0], text="ETL ETROC Test Beam", fontsize=18)
    ax[0].set_title(plot_title, loc="right", size=16)
    ax[0].xaxis.label.set_fontsize(25)
    ax[0].yaxis.label.set_fontsize(25)
    ax[0].xaxis.set_major_formatter(ticker.NullFormatter())
    ax[0].xaxis.set_minor_locator(ticker.FixedLocator(location))
    ax[0].xaxis.set_minor_formatter(ticker.FixedFormatter(tick_labels))
    ax[0].yaxis.set_major_formatter(ticker.NullFormatter())
    ax[0].yaxis.set_minor_locator(ticker.FixedLocator(location))
    ax[0].yaxis.set_minor_formatter(ticker.FixedFormatter(tick_labels))
    ax[0].tick_params(axis='both', which='major', length=0)

    hep.hist2dplot(h_col, ax=ax[1], norm= colors.LogNorm())
    hep.cms.text(loc=0, ax=ax[1], text="ETL ETROC Test Beam", fontsize=18)
    ax[1].set_title(plot_title, loc="right", size=16)
    ax[1].xaxis.label.set_fontsize(25)
    ax[1].yaxis.label.set_fontsize(25)
    ax[1].xaxis.set_major_formatter(ticker.NullFormatter())
    ax[1].xaxis.set_minor_locator(ticker.FixedLocator(location))
    ax[1].xaxis.set_minor_formatter(ticker.FixedFormatter(tick_labels))
    ax[1].yaxis.set_major_formatter(ticker.NullFormatter())
    ax[1].yaxis.set_minor_locator(ticker.FixedLocator(location))
    ax[1].yaxis.set_minor_formatter(ticker.FixedFormatter(tick_labels))
    ax[1].tick_params(axis='both', which='major', length=0)

    plt.tight_layout()

    if save_dir:
        save_plot(fig, save_dir, f"spatial_correlation_{board_name1}_{board_name2}_{fname_tag}")

## --------------------------------------
def plot_difference_of_pixels(
        input_df: pd.DataFrame,
        board_ids: list[int],
        board_name1: str,
        board_name2: str,
        tb_loc: str,
        fname_tag: str = '',
        save_mother_dir: Path | None = None,
    ):
    """Make 2D map of delta Row and delta Column.

    Parameters
    ----------
    input_df: pd.DataFrame,
        Pandas dataframe of data.
    board_ids: list[int],
        A list of integer (board ID) that wants to make plots.
    board_name1: str,
        Board 1 name.
    board_name2: str,
        Board 2 name.
    tb_loc: str,
        Test Beam location for the title. Available argument: desy, cern, fnal.
    fname_tag: str, optional (recommend)
        Additiional tag for the file name.
    save_mother_dir: Path, optional
        Plot will be saved at save_mother_dir/'spatial_correlation'.
    """

    plot_title = load_fig_title(tb_loc)
    diff_row = (input_df[f'row_{board_ids[0]}'].astype(np.int8) - input_df[f'row_{board_ids[1]}'].astype(np.int8)).values
    diff_col = (input_df[f'col_{board_ids[0]}'].astype(np.int8) - input_df[f'col_{board_ids[1]}'].astype(np.int8)).values

    save_dir = save_mother_dir / 'spatial_correlation' if save_mother_dir else None

    h = hist.Hist(
        hist.axis.Regular(32, -16, 16, name='delta_row', label=r"$\Delta$Row"),
        hist.axis.Regular(32, -16, 16, name='delta_col', label=r"$\Delta$Col"),
    )

    h.fill(diff_row, diff_col)

    fig, ax = plt.subplots(dpi=100, figsize=(11, 11))

    hep.hist2dplot(h, ax=ax, norm=colors.LogNorm())
    hep.cms.text(loc=0, ax=ax, text="ETL ETROC Test Beam", fontsize=18)
    ax.set_title(plot_title, loc="right", size=16)
    ax.xaxis.label.set_fontsize(25)
    ax.yaxis.label.set_fontsize(25)
    ax.tick_params(axis='x', which='both', length=5, labelsize=17)
    ax.tick_params(axis='y', which='both', length=5, labelsize=17)
    plt.minorticks_off()
    plt.tight_layout()

    if save_dir:
        save_plot(fig, save_dir, f"spatial_difference_{board_name1}_{board_name2}_{fname_tag}")

## --------------------------------------
def plot_distance(
        input_df: pd.DataFrame,
        board_ids: np.array,
        xaxis_label_board_name: str,
        fig_title: str,
        fig_tag: str = '',
        do_logy: bool = False,
        no_show: bool = False,
    ):
    h_dis = hist.Hist(hist.axis.Regular(32, 0, 32, name='dis', label=f'Distance (Trigger - {xaxis_label_board_name})'))

    diff_row = (input_df.loc[input_df['board'] == board_ids[0]]['row'].reset_index(drop=True) - input_df.loc[input_df['board'] == board_ids[1]]['row'].reset_index(drop=True)).values
    diff_col = (input_df.loc[input_df['board'] == board_ids[0]]['col'].reset_index(drop=True) - input_df.loc[input_df['board'] == board_ids[1]]['col'].reset_index(drop=True)).values
    dis = np.sqrt(diff_row**2 + diff_col**2)
    h_dis.fill(dis)
    del diff_row, diff_col, dis

    fig, ax = plt.subplots(dpi=100, figsize=(15, 8))
    hep.histplot(h_dis, ax=ax)
    hep.cms.text(loc=0, ax=ax, text="ETL ETROC Test Beam", fontsize=18)
    ax.set_title(f"{fig_title} {fig_tag}", loc="right", size=16)

    if do_logy:
        ax.set_yscale('log')

    if no_show:
        plt.close(fig)

    return h_dis

## --------------------------------------
def plot_TOA_correlation(
        input_df: pd.DataFrame,
        board_id1: int,
        board_id2: int,
        boundary_cut: float,
        board_names: list[str],
        tb_loc: str,
        draw_boundary: bool = False,
        save_mother_dir: Path | None = None,
    ):
    """Make plot of TOA correlation between selected two boards.

    Parameters
    ----------
    input_df: pd.DataFrame,
        Pandas dataframe of a single track.
    board_id1: int,
        Board 1 ID.
    board_id2: int,
        Board 2 ID.
    boundary_cut: float,
        Size of boundary. boundary_cut * standard devition of distance arrays
    board_names: list[str],
        A string list including board names.
    tb_loc: str,
        Test Beam location for the title. Available argument: desy, cern, fnal.
    draw_boundary: bool, optional
        Draw boundary cut in the plot.
    save_mother_dir: Path, optional
        Plot will be saved at save_mother_dir/'temporal_correlation'.
    """

    plot_title = load_fig_title(tb_loc)
    x = input_df['toa'][board_id1]
    y = input_df['toa'][board_id2]

    axis_name1 = board_names[board_id1].replace('_', ' ')
    axis_name2 = board_names[board_id2].replace('_', ' ')

    save_dir = save_mother_dir / 'temporal_correlation' if save_mother_dir else None


    h = hist.Hist(
        hist.axis.Regular(128, 0, 1024, name=f'{board_names[board_id1]}', label=f'TOA of {axis_name1} [LSB]'),
        hist.axis.Regular(128, 0, 1024, name=f'{board_names[board_id2]}', label=f'TOA of {axis_name2} [LSB]'),
    )
    h.fill(x, y)
    params = np.polyfit(x, y, 1)
    distance = (x*params[0] - y + params[1])/(np.sqrt(params[0]**2 + 1))

    fig, ax = plt.subplots(figsize=(11, 10))
    hep.cms.text(loc=0, ax=ax, text="ETL ETROC Test Beam", fontsize=18)
    ax.set_title(plot_title, loc='right', fontsize=16)
    ax.xaxis.label.set_fontsize(25)
    ax.yaxis.label.set_fontsize(25)
    hep.hist2dplot(h, ax=ax, norm=colors.LogNorm())

    # calculate the trendline
    trendpoly = np.poly1d(params)
    x_range = np.linspace(x.min(), x.max(), 500)

    # plot the trend line
    ax.plot(x_range, trendpoly(x_range), 'r-', label='linear fit')
    if draw_boundary:
        ax.plot(x_range, trendpoly(x_range)-boundary_cut*np.std(distance), 'r--', label=fr'{boundary_cut}$\sigma$ boundary')
        ax.plot(x_range, trendpoly(x_range)+boundary_cut*np.std(distance), 'r--')
        # ax.fill_between(x_range, y1=trendpoly(x_range)-boundary_cut*np.std(distance), y2=trendpoly(x_range)+boundary_cut*np.std(distance),
        #                 facecolor='red', alpha=0.35, label=fr'{boundary_cut}$\sigma$ boundary')
    ax.legend()

    if save_dir:
        save_plot(fig, save_dir, f"toa_correlation_{board_names[board_id1]}_{board_names[board_id2]}")

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
def plot_resolution_with_pulls(
        input_df: pd.DataFrame,
        board_ids: list[int],
        tb_loc: str,
        fig_config: dict,
        hist_range: list[int] = [20, 95],
        hist_bins: int = 15,
        slides_friendly: bool = False,
        print_fit_results: bool = False,
        constraint_ylim: bool = False,
        save_mother_dir: Path | None = None,
    ):
    """Make summary plot of the board resolution plot with a gaussian fit.

    Parameters
    ----------
    input_df: pd.DataFrame
        Pandas dataframe includes bootstrap results.
    board_ids: list[int]
        A list of board IDs to make plots.
    tb_loc: str,
        Test Beam location for the title. Available argument: desy, cern, fnal.
    fig_config: dict,
        Dictionary with board ID as a dict key. It contains a figure title.
    hist_range: list[int], optional
        Set the histogram range. Default value is [20, 95].
    hist_bins: int, optional
        Adjust the histogram bins. Default value is 15.
    slide_friendly: bool, optional
        If it is True, draw plots in a single figure. Recommend this option, when you try to add plots on the slides.
    save_mother_dir: Path, optional
        Plot will be saved at save_mother_dir/'time_resolution_results'.
    """

    import matplotlib.gridspec as gridspec
    from lmfit.models import GaussianModel
    from lmfit.lineshapes import gaussian

    plot_title = load_fig_title(tb_loc)
    mod = GaussianModel(nan_policy='omit')

    hists = {}
    fit_params = {}
    pulls_dict = {}
    means = {}

    for key in board_ids:
        hists[key] = hist.Hist(hist.axis.Regular(hist_bins, hist_range[0], hist_range[1], name="time_resolution", label=r'Time Resolution [ps]'))
        hists[key].fill(input_df[f'res{key}'].values)
        means[key] = np.mean(input_df[f'res{key}'].values)
        centers = hists[key].axes[0].centers
        fit_range = centers[np.argmax(hists[key].values())-5:np.argmax(hists[key].values())+5]
        fit_vals = hists[key].values()[np.argmax(hists[key].values())-5:np.argmax(hists[key].values())+5]

        pars = mod.guess(fit_vals, x=fit_range)
        out = mod.fit(fit_vals, pars, x=fit_range, weights=1/np.sqrt(fit_vals))
        fit_params[key] = out

        if print_fit_results:
            print(key)
            print(out.fit_report())

        ### Calculate pull
        pulls = (hists[key].values() - out.eval(x=centers))/np.sqrt(out.eval(x=centers))
        pulls[np.isnan(pulls) | np.isinf(pulls)] = 0
        pulls_dict[key] = pulls


    if slides_friendly:

        # Create a figure with a 2x2 grid
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(2, 2)

        for i, plot in enumerate(gs):
            global_ax = fig.add_subplot(plot)
            inner_plot = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=global_ax, hspace=0, height_ratios=[3, 1])
            main_ax = fig.add_subplot(inner_plot[0])
            sub_ax = fig.add_subplot(inner_plot[1], sharex=main_ax)
            global_ax.xaxis.set_visible(False)
            global_ax.yaxis.set_visible(False)
            main_ax.xaxis.set_visible(False)

            if i not in hists:
                global_ax.set_axis_off()
                main_ax.set_axis_off()
                sub_ax.set_axis_off()
                continue

            sup_title = fig_config[idx]['title']
            centers = hists[i].axes[0].centers
            hep.cms.text(loc=0, ax=main_ax, text="ETL ETROC Test Beam", fontsize=20)
            main_ax.set_title(f'{plot_title} {sup_title}', loc="right", size=18)

            main_ax.errorbar(centers, hists[i].values(), np.sqrt(hists[i].variances()),
                            ecolor="steelblue", mfc="steelblue", mec="steelblue", fmt="o",
                            ms=6, capsize=1, capthick=2, alpha=0.8)

            main_ax.set_ylabel('Counts', fontsize=18)
            main_ax.set_ylim(-5, 190)
            main_ax.tick_params(axis='x', labelsize=18)
            main_ax.tick_params(axis='y', labelsize=18)

            x_min = centers[0]
            x_max = centers[-1]

            x_range = np.linspace(x_min, x_max, 200)
            popt = [par for name, par in fit_params[i].best_values.items()]
            pcov = fit_params[i].covar

            if np.isfinite(pcov).all():
                n_samples = 100
                vopts = np.random.multivariate_normal(popt, pcov, n_samples)
                sampled_ydata = np.vstack([gaussian(x_range, *vopt).T for vopt in vopts])
                model_uncert = np.nanstd(sampled_ydata, axis=0)
            else:
                model_uncert = np.zeros_like(np.sqrt(hists[i].variances()))

            main_ax.plot(x_range, fit_params[i].eval(x=x_range), color="hotpink", ls="-", lw=2, alpha=0.8,
                        label=fr"$\mu$:{fit_params[i].params['center'].value:.2f} $\pm$ {fit_params[i].params['center'].stderr:.2f} ps")
            main_ax.plot(np.NaN, np.NaN, color='none',
                        label=fr"$\sigma$: {abs(fit_params[i].params['sigma'].value):.2f} $\pm$ {abs(fit_params[i].params['sigma'].stderr):.2f} ps")

            main_ax.fill_between(
                x_range,
                fit_params[i].eval(x=x_range) - model_uncert,
                fit_params[i].eval(x=x_range) + model_uncert,
                color="hotpink",
                alpha=0.2,
                label='Fit Uncertainty'
            )
            main_ax.legend(fontsize=18, loc='best', title=fig_config[idx]['title'], title_fontsize=18)

            width = (x_max - x_min) / len(pulls_dict[i])
            sub_ax.axhline(1, c='black', lw=0.75)
            sub_ax.axhline(0, c='black', lw=1.2)
            sub_ax.axhline(-1, c='black', lw=0.75)
            sub_ax.bar(centers, pulls_dict[i], width=width, fc='royalblue')
            sub_ax.set_ylim(-2, 2)
            sub_ax.set_yticks(ticks=np.arange(-1, 2), labels=[-1, 0, 1], fontsize=20)
            sub_ax.set_xlabel('Pixel Time Resolution [ps]', fontsize=20)
            sub_ax.tick_params(axis='x', which='both', labelsize=20)
            sub_ax.set_ylabel('Pulls', fontsize=20, loc='center')

        fig.tight_layout()

        if save_mother_dir is not None:
            save_dir = save_mother_dir / 'time_resolution_results'
            save_dir.mkdir(exist_ok=True)
            fig.savefig(save_dir / f"board_res.png")
            fig.savefig(save_dir / f"board_res.pdf")
            plt.close(fig)

        del hists, fit_params, pulls_dict, mod

    else:
        for idx in hists.keys():
            fig = plt.figure(figsize=(11, 10))
            grid = fig.add_gridspec(2, 1, hspace=0, height_ratios=[3, 1])

            main_ax = fig.add_subplot(grid[0])
            sub_ax = fig.add_subplot(grid[1], sharex=main_ax)
            plt.setp(main_ax.get_xticklabels(), visible=False)

            centers = hists[idx].axes[0].centers
            hep.cms.text(loc=0, ax=main_ax, text="ETL ETROC Test Beam", fontsize=18)
            main_ax.set_title(f'{plot_title}', loc="right", size=16)

            main_ax.errorbar(centers, hists[idx].values(), np.sqrt(hists[idx].variances()),
                            ecolor="steelblue", mfc="steelblue", mec="steelblue", fmt="o",
                            ms=6, capsize=1, capthick=2, alpha=0.8)

            main_ax.set_ylabel('Counts', fontsize=25)
            main_ax.tick_params(axis='x', labelsize=20)
            main_ax.tick_params(axis='y', labelsize=20)
            if constraint_ylim:
                main_ax.set_ylim(-5, 190)

            x_min = centers[0]
            x_max = centers[-1]

            x_range = np.linspace(x_min, x_max, 500)
            popt = [par for name, par in fit_params[idx].best_values.items()]
            pcov = fit_params[idx].covar

            if np.isfinite(pcov).all():
                n_samples = 100
                vopts = np.random.multivariate_normal(popt, pcov, n_samples)
                sampled_ydata = np.vstack([gaussian(x_range, *vopt).T for vopt in vopts])
                model_uncert = np.nanstd(sampled_ydata, axis=0)
            else:
                model_uncert = np.zeros_like(np.sqrt(hists[i].variances()))

            main_ax.plot(x_range, fit_params[idx].eval(x=x_range), color="hotpink", ls="-", lw=2, alpha=0.8,
                        label=fr"$\mu$:{fit_params[idx].params['center'].value:.2f} $\pm$ {fit_params[idx].params['center'].stderr:.2f} ps")
            main_ax.plot(np.NaN, np.NaN, color='none',
                        label=fr"$\sigma$: {abs(fit_params[idx].params['sigma'].value):.2f} $\pm$ {abs(fit_params[idx].params['sigma'].stderr):.2f} ps")

            main_ax.fill_between(
                x_range,
                fit_params[idx].eval(x=x_range) - model_uncert,
                fit_params[idx].eval(x=x_range) + model_uncert,
                color="hotpink",
                alpha=0.2,
                label='Fit Uncertainty'
            )
            main_ax.legend(fontsize=18, loc='best', title=fig_config[idx]['title'], title_fontsize=18)

            width = (x_max - x_min) / len(pulls_dict[idx])
            sub_ax.axhline(1, c='black', lw=0.75)
            sub_ax.axhline(0, c='black', lw=1.2)
            sub_ax.axhline(-1, c='black', lw=0.75)
            sub_ax.bar(centers, pulls_dict[idx], width=width, fc='royalblue')
            sub_ax.set_ylim(-2, 2)
            sub_ax.set_yticks(ticks=np.arange(-1, 2), labels=[-1, 0, 1], fontsize=20)
            sub_ax.set_xlabel('Pixel Time Resolution [ps]', fontsize=25)
            sub_ax.tick_params(axis='x', which='both', labelsize=20)
            sub_ax.set_ylabel('Pulls', fontsize=20, loc='center')

            fig.tight_layout()

            if save_mother_dir is not None:
                save_dir = save_mother_dir / 'time_resolution_results'
                save_dir.mkdir(exist_ok=True)
                fig.savefig(save_dir / f"board_res_{fig_config[idx]['short']}.png")
                fig.savefig(save_dir / f"board_res_{fig_config[idx]['short']}.pdf")
                plt.close(fig)

        del hists, fit_params, pulls_dict, mod

## --------------------------------------
def plot_resolution_table(
        input_df: pd.DataFrame,
        board_ids: list[int],
        tb_loc: str,
        fig_config: dict,
        min_resolution: float = 25.0,
        max_resolution: float = 75.0,
        missing_pixel_info: dict | None = None,
        slides_friendly: bool = False,
        show_number: bool = False,
        save_mother_dir: Path | None = None,
    ):

    from matplotlib import colormaps
    cmap = colormaps['viridis']
    cmap.set_under(color='lightgrey')

    plot_title = load_fig_title(tb_loc)

    tables = {}
    for board_id in board_ids:
        board_info = input_df[[f'row{board_id}', f'col{board_id}', f'res{board_id}', f'err{board_id}']]

        res = board_info.groupby([f'row{board_id}', f'col{board_id}']).apply(lambda x: np.average(x[f'res{board_id}'], weights=1/x[f'err{board_id}']**2)).reset_index()
        err = board_info.groupby([f'row{board_id}', f'col{board_id}']).apply(lambda x: np.sqrt(1/(np.sum(1/x[f'err{board_id}']**2)))).reset_index()

        res_table = res.pivot_table(index=f'row{board_id}', columns=f'col{board_id}', values=0, fill_value=-1)
        err_table = err.pivot_table(index=f'row{board_id}', columns=f'col{board_id}', values=0, fill_value=-1)

        res_table = res_table.reindex(pd.Index(np.arange(0,16), name='')).reset_index()
        res_table = res_table.reindex(columns=np.arange(0,16))
        res_table = res_table.fillna(-1)

        err_table = err_table.reindex(pd.Index(np.arange(0,16), name='')).reset_index()
        err_table = err_table.reindex(columns=np.arange(0,16))
        err_table = err_table.fillna(-1)

        tables[board_id] = [res_table, err_table]

    if slides_friendly:
        fig = plt.figure(figsize=(35, 35))
        gs = fig.add_gridspec(2, 2)

        for idx, plot in enumerate(gs):
            ax = fig.add_subplot(plot)

            if idx not in tables:
                ax.set_axis_off()
                continue
            im = ax.imshow(tables[idx][0], cmap=cmap, interpolation="nearest", vmin=min_resolution, vmax=max_resolution)

            # Add color bar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Time Resolution (ps)', fontsize=20)
            cbar.ax.tick_params(labelsize=18)

            if show_number:
                for i in range(16):
                    for j in range(16):
                        value = tables[idx][0].iloc[i, j]
                        error = tables[idx][1].iloc[i, j]
                        if value == -1: continue
                        text_color = 'black' if value > 0.66*(min_resolution + max_resolution) else 'white'
                        text = str(rf"{value:.1f}""\n"fr"$\pm$ {error:.1f}")
                        plt.text(j, i, text, va='center', ha='center', color=text_color, fontsize=18, rotation=45)

            hep.cms.text(loc=0, ax=ax, text="ETL ETROC Test Beam", fontsize=18)
            ax.set_xlabel('Column (col)', fontsize=18)
            ax.set_ylabel('Row (row)', fontsize=18)
            ticks = range(0, 16)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            sup_title = fig_config[idx]['title']
            ax.set_title(f"{plot_title} | {sup_title}", loc="right", size=18)
            ax.tick_params(axis='x', which='both', length=5, labelsize=18)
            ax.tick_params(axis='y', which='both', length=5, labelsize=18)
            ax.invert_xaxis()
            ax.invert_yaxis()
            plt.minorticks_off()

        plt.tight_layout()

        if save_mother_dir is not None:
            save_dir = save_mother_dir / 'time_resolution_results'
            save_dir.mkdir(exist_ok=True)
            fig.savefig(save_dir / f"resolution_map.png")
            fig.savefig(save_dir / f"resolution_map.pdf")
            plt.close(fig)
        del tables

    else:
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
            ax.set_title(f"{plot_title} | {sup_title}", loc="right", size=16)
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
