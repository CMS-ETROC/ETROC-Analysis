import sqlite3, hist
import pandas as pd
import mplhep as hep
hep.style.use('CMS')
import matplotlib.pyplot as plt
from pathlib import Path

__all__ = [
    'read_BLNW_history_chips',
    'read_BLNW_history_chip_measurements',
    'read_BLNW_history_chip_measurement_df',
    'make_BLNW_history_1DNW_plot',
    'make_BLNW_history_2D_plots',
    'make_BLNW_history_chip_measurement_plots'
]

## --------------- BL and NW plotting made by Cristovao -----------------------
## --------------------------------------
def read_BLNW_history_chips(sqlite_file: Path):
    with sqlite3.connect(sqlite_file) as sqlite3_connection:
        chip_df = pd.read_sql_query("SELECT chip_name FROM baselines", sqlite3_connection)

        return chip_df.chip_name.unique()

## --------------------------------------
def read_BLNW_history_chip_measurements(sqlite_file: Path, chip_name: str):
    with sqlite3.connect(sqlite_file) as sqlite3_connection:
        timestamp_df = pd.read_sql_query(f"SELECT timestamp, save_notes FROM baselines WHERE chip_name='{chip_name}' AND ROW=0 AND COL=0", sqlite3_connection)
        max_timestamp_df = pd.read_sql_query(f"SELECT timestamp FROM baselines WHERE chip_name='{chip_name}' AND ROW=15 AND COL=15", sqlite3_connection)
        timestamp_df['min_timestamp'] = pd.to_datetime(timestamp_df['timestamp'], format='mixed')
        timestamp_df['max_timestamp'] = pd.to_datetime(max_timestamp_df['timestamp'], format='mixed')

        return timestamp_df.drop('timestamp', axis=1).copy()

## --------------------------------------
def read_BLNW_history_chip_measurement_df(sqlite_file: Path, chip_name: str, min_timestamp, max_timestamp):
    with sqlite3.connect(sqlite_file) as sqlite3_connection:
        data_df = pd.read_sql_query(f"SELECT * FROM baselines WHERE chip_name='{chip_name}'", sqlite3_connection)
        data_df['timestamp'] = pd.to_datetime(data_df['timestamp'], format='mixed')
        data_df = data_df.loc[data_df.timestamp >= min_timestamp]
        data_df = data_df.loc[data_df.timestamp <= max_timestamp]

        return data_df.copy()

## --------------------------------------
def make_BLNW_history_1DNW_plot(data_df: pd.DataFrame, chip_name, chip_figname, save_note: str, save: bool, fig_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 9))
    hep.cms.text(loc=0, ax=ax, fontsize=17, text="ETL ETROC")
    tmp_hist = hist.Hist(hist.axis.Regular(16, 0, 16, name='nw', label='NW [DAC]'))
    nw_array = data_df['noise_width'].to_numpy().flatten()
    tmp_hist.fill(nw_array)
    mean, std = nw_array.mean(), nw_array.std()
    tmp_hist.plot1d(ax=ax, yerr=False, label=f'Mean: {mean:.2f}, Std: {std:.2f}')
    ax.set_title(f"{chip_figname}: NW (DAC LSB)\n{save_note}", size=17, loc="right")
    ax.legend(fontsize=16)
    plt.xticks(range(16), range(16))

    plt.tight_layout()
    plt.show()

    if save:
        keepcharacters = (' ','.','_')
        save_note_safe = "".join(c for c in save_note if c.isalnum() or c in keepcharacters).rstrip()
        save_note_safe = "_".join(save_note_safe.split())
        fig.savefig(fig_dir / ("NW_1D_"+chip_name+"_"+save_note_safe+".png"))

## --------------------------------------
def make_BLNW_history_2D_plots(data_df: pd.DataFrame, chip_name, chip_figname, save_note: str, save: bool, fig_dir: Path):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig = plt.figure(dpi=200, figsize=(20,10))
    gs = fig.add_gridspec(1,2)

    ax0 = fig.add_subplot(gs[0,0])
    BL_vmin = data_df.baseline.min()
    BL_vmax = data_df.baseline.max()
    NW_vmin = 0
    NW_vmax = 16

    pivot_data_df = data_df.pivot(
        index = ['row'],
        columns = ['col'],
        values = ['baseline', 'noise_width'],
    )

    ax0.set_title(f"{chip_figname}: BL (DAC LSB)\n{save_note}", size=17, loc="right")
    img0 = ax0.imshow(pivot_data_df.baseline, interpolation='none',vmin=BL_vmin,vmax=BL_vmax)
    ax0.set_aspect("equal")
    ax0.invert_xaxis()
    ax0.invert_yaxis()
    plt.xticks(range(16), range(16), rotation="vertical")
    plt.yticks(range(16), range(16))
    hep.cms.text(loc=0, ax=ax0, fontsize=17, text="ETL ETROC")
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes('right', size="5%", pad=0.05)
    fig.colorbar(img0, cax=cax, orientation="vertical")#,boundaries=np.linspace(vmin,vmax,int((vmax-vmin)*30)))

    ax1 = fig.add_subplot(gs[0,1])
    ax1.set_title(f"{chip_figname}: NW (DAC LSB)\n{save_note}", size=17, loc="right")
    img1 = ax1.imshow(pivot_data_df.noise_width, interpolation='none',vmin=NW_vmin,vmax=NW_vmax)
    ax1.set_aspect("equal")
    ax1.invert_xaxis()
    ax1.invert_yaxis()
    plt.xticks(range(16), range(16), rotation="vertical")
    plt.yticks(range(16), range(16))
    hep.cms.text(loc=0, ax=ax1, fontsize=17, text="ETL ETROC")
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size="5%", pad=0.05)
    fig.colorbar(img1, cax=cax, orientation="vertical")#,boundaries=np.linspace(vmin,vmax,int((vmax-vmin)*5)))

    for col in range(16):
        for row in range(16):
            ax0.text(col,row,f"{pivot_data_df.baseline[col][row]:.0f}", c="white", size=10, rotation=45, fontweight="bold", ha="center", va="center")
            ax1.text(col,row,f"{pivot_data_df.noise_width[col][row]:.0f}", c="white", size=10, rotation=45, fontweight="bold", ha="center", va="center")
    fig.tight_layout()

    if save:
        keepcharacters = (' ','.','_')
        save_note_safe = "".join(c for c in save_note if c.isalnum() or c in keepcharacters).rstrip()
        save_note_safe = "_".join(save_note_safe.split())
        fig.savefig(fig_dir / ("BL_NW_"+chip_name+"_"+save_note_safe+".png"))

## --------------------------------------
def make_BLNW_history_chip_measurement_plots(sqlite_file: Path, chip_name: str, measurement_idx: int, save: bool = False, save_dir: Path = None, figname_map = {}):
    measurement_df = read_BLNW_history_chip_measurements(sqlite_file, chip_name)
    if measurement_idx >= len(measurement_df) or measurement_idx < 0:
        print("You selected a measurement which does not exist, exiting")
        return
    if save and save_dir is None:
        save = False

    save_note     = measurement_df.save_notes[measurement_idx]
    min_timestamp = measurement_df.min_timestamp[measurement_idx]
    max_timestamp = measurement_df.max_timestamp[measurement_idx]
    chip_figname = chip_name
    if chip_name in figname_map:
        chip_figname = figname_map[chip_name]

    data_df = read_BLNW_history_chip_measurement_df(sqlite_file, chip_name, min_timestamp, max_timestamp)

    make_BLNW_history_2D_plots(data_df, chip_name, chip_figname, save_note, save, save_dir)
    make_BLNW_history_1DNW_plot(data_df, chip_name, chip_figname, save_note, save, save_dir)

## --------------- BL and NW plotting made by Cristovao -----------------------