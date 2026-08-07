from pathlib import Path
from natsort import natsorted
from tqdm import tqdm

import argparse
import random
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mplhep as hep

hep.style.use('CMS')

parser = argparse.ArgumentParser(
            prog='Plot CAL per file',
            description='Plot per-pixel CAL trend across files, from find_cal_per_file.py sqlite output',
        )

parser.add_argument(
    '-i',
    '--input',
    metavar = 'NAME',
    type = str,
    help = 'sqlite file produced by find_cal_per_file.py',
    required = True,
    dest = 'input',
)

parser.add_argument(
    '-o',
    '--outdir',
    metavar = 'NAME',
    type = str,
    help = 'Output directory to save the plot',
    required = True,
    dest = 'outdir',
)

parser.add_argument(
    '--value',
    choices = ['cal_mode', 'cal_mean', 'both'],
    default = 'both',
    help = 'Which CAL statistic to plot. Default: both',
    dest = 'value',
)

parser.add_argument(
    '--seed',
    type = int,
    default = None,
    help = 'Random seed for reproducible pixel selection. Default: unseeded (different pixels each run)',
    dest = 'seed',
)

args = parser.parse_args()

rng = random.Random(args.seed)

# Averaging over all pixels hides board-specific behavior at a single pixel, so
# instead pick two actual pixels -- one from each half of the column range --
# and plot their raw per-board CAL trend as-is, no aggregation.
pixels = {
    'col0-7': (rng.randrange(16), rng.randrange(0, 8)),
    'col8-15': (rng.randrange(16), rng.randrange(8, 16)),
}

with sqlite3.connect(args.input) as sqlconn:
    cursor = sqlconn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    table_names = natsorted(row[0] for row in cursor.fetchall())

    if not table_names:
        raise SystemExit(f'No tables found in {args.input}')

    file_index = {name: i for i, name in enumerate(table_names)}

    pixel_dfs = []
    for table in tqdm(table_names):
        for label, (row_sel, col_sel) in pixels.items():
            df = pd.read_sql(
                f'SELECT board, cal_mode, cal_mean FROM "{table}" WHERE "row" = ? AND "col" = ?',
                sqlconn, params=(row_sel, col_sel),
            )
            df['file'] = table
            df['pixel'] = label
            pixel_dfs.append(df)

pixel_df = pd.concat(pixel_dfs, ignore_index=True)
pixel_df['file_index'] = pixel_df['file'].map(file_index)

out_dir = Path(args.outdir)
out_dir.mkdir(parents=True, exist_ok=True)

values_to_plot = ['cal_mode', 'cal_mean'] if args.value == 'both' else [args.value]
value_labels = {'cal_mode': 'CAL mode', 'cal_mean': 'CAL mean'}

for label, (row_sel, col_sel) in pixels.items():
    sub_df = pixel_df.loc[pixel_df['pixel'] == label]

    for value in values_to_plot:
        fig, ax = plt.subplots(figsize=(12, 8))

        for board in natsorted(sub_df['board'].unique()):
            board_df = sub_df.loc[sub_df['board'] == board].sort_values('file_index')
            ax.plot(
                board_df['file_index'], board_df[value],
                marker='o', markersize=4, linestyle='-', label=f'Board {board}',
            )

        hep.cms.text(loc=0, ax=ax, text='ETL ETROC Test Beam', fontsize=18)
        ax.set_xlabel('File', fontsize=25)
        ax.set_ylabel(f'{value_labels[value]} [LSB]', fontsize=25)
        ax.set_title(f'Pixel (row={row_sel}, col={col_sel})', loc='right', fontsize=16)
        ax.set_xticks(list(file_index.values()))
        ax.set_xticklabels(table_names, rotation=45, ha='right', fontsize=12)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.tick_params(axis='y', which='major', labelsize=17)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=16, loc='upper left', bbox_to_anchor=(1.01, 1.0))
        plt.tight_layout()

        output_base = out_dir / f'cal_trend_{value}_row{row_sel}_col{col_sel}'
        fig.savefig(f'{output_base}.png')
        fig.savefig(f'{output_base}.pdf')
        plt.close(fig)

        print(f'Saved {output_base}.png/.pdf')
