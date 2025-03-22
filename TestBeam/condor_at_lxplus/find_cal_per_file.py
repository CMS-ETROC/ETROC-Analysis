from pathlib import Path
from natsort import natsorted
from tqdm import tqdm

import argparse
import sqlite3
import pandas as pd

parser = argparse.ArgumentParser(
            prog='PlaceHolder',
            description='figure cal code mean and mode per file',
        )

parser.add_argument(
    '-d',
    '--input_dir',
    metavar = 'NAME',
    type = str,
    help = 'input directory containing .bin',
    required = True,
    dest = 'input_dir',
)

parser.add_argument(
    '-o',
    '--output',
    metavar = 'NAME',
    type = str,
    help = 'Name for sqlite file',
    required = True,
    dest = 'output',
)

parser.add_argument(
    '--make_plot',
    action = 'store_true',
    help = 'If set, make the plot and save it',
    dest = 'make_plot',
)

args = parser.parse_args()

file_list = natsorted(Path(args.input_dir).glob('loop*.feather'))
outname = f'{args.output}.sqlite'

for ifile in tqdm(file_list):
    num = ifile.name.split('.')[0]
    tmp_df = pd.read_feather(ifile)

    cal_table = (
        tmp_df.pivot_table(index=["row", "col"], columns="board", values="cal", aggfunc=lambda x: x.mode().iat[0] if not x.mode().empty else None)
        .reset_index()
        .melt(id_vars=["row", "col"], var_name="board", value_name="cal_mode")
        .dropna()  # Drop rows where cal_mode is NaN
    )

    # Merge the original dataframe with the calculated mode values
    merged_df = pd.merge(tmp_df, cal_table, on=["board", "row", "col"], how="inner")

    # # Convert types for efficiency
    merged_df["board"] = merged_df["board"].astype("uint8")
    merged_df["cal_mode"] = merged_df["cal_mode"].astype("int16")

    # # Compute absolute difference and filter
    merged_df["cal_diff"] = (merged_df["cal"] - merged_df["cal_mode"]).abs()
    filtered_df = merged_df.loc[merged_df["cal_diff"] <= 3][['board', 'row', 'col', 'cal', 'cal_mode']].reset_index(drop=True)

    cal_mean_df = filtered_df.groupby(["board", "row", "col"], as_index=False)["cal"].mean()
    cal_mean_df.rename(columns={"cal": "cal_mean"}, inplace=True)

    cal_summary_df = pd.merge(cal_table, cal_mean_df, on=["board", "row", "col"], how="left")
    cal_summary_df['board'] = cal_summary_df['board'].astype('uint8')
    cal_summary_df['cal_mode'] = cal_summary_df['cal_mode'].astype('int16')

    with sqlite3.connect(outname) as sqlconn:
        cal_summary_df.to_sql(f'{num}', sqlconn, if_exists='append', index=False)