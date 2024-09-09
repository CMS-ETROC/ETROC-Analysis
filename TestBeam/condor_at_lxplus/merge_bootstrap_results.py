from collections import defaultdict
from natsort import natsorted
from tqdm import tqdm
from lmfit.models import GaussianModel
from glob import glob

import mplhep as hep
hep.style.use('CMS')
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import re
import hist
import numpy as np
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(
            prog='PlaceHolder',
            description='merge individual bootstrap results',
        )

parser.add_argument(
    '-d',
    '--inputdir',
    metavar = 'DIRNAME',
    type = str,
    help = 'input directory name',
    required = True,
    dest = 'dirname',
)

parser.add_argument(
    '-o',
    '--outputname',
    metavar = 'OUTNAME',
    type = str,
    help = 'output file name',
    required = True,
    dest = 'outname',
)

parser.add_argument(
    '--minimum',
    metavar = 'VALUE',
    type = int,
    help = 'minimum number of bootstrap results to do a fit',
    dest = 'minimum',
    default = 50,
)

parser.add_argument(
    '--hist_bins',
    metavar = 'VALUE',
    type = int,
    help = 'Set a histogram bins',
    dest = 'hist_bins',
    default = 35,
)

args = parser.parse_args()
files = natsorted(glob(args.dirname+'/*pkl'))

final_dict = defaultdict(list)
mod = GaussianModel(nan_policy='omit')

for ifile in tqdm(files):

    # Define the pattern to match "RxCx" part
    pattern = r'R(\d+)C(\d+)'

    # Find all occurrences of the pattern in the string
    matches = re.findall(pattern, ifile)

    df = pd.read_pickle(ifile)
    columns = df.columns

    # There is a bug in bootstrap code. Time resolution result is 0, should be dropped.
    df = df.loc[(df != 0).all(axis=1)]

    if df.shape[0] < args.minimum:
        # print('Bootstrap result is not correct. Do not process!')
        # print(df.shape[0])
        continue

    if len(matches) == 4:
        final_dict['row0'].append(matches[0][0])
        final_dict['col0'].append(matches[0][1])

        for val in columns:

            x_min = df[val].min()-5
            x_max = df[val].max()+5

            h_temp = hist.Hist(hist.axis.Regular(args.hist_bins, x_min, x_max, name="time_resolution", label=r'Time Resolution [ps]'))
            h_temp.fill(df[val])
            centers = h_temp.axes[0].centers

            fit_constrain = (centers > df[val].astype(int).mode()[0]-7) & (centers < df[val].astype(int).mode()[0]+7)

            final_dict[f'row{val}'].append(matches[val][0])
            final_dict[f'col{val}'].append(matches[val][1])

            try:
                pars = mod.guess(h_temp.values()[fit_constrain], x=centers[fit_constrain])
                out = mod.fit(h_temp.values()[fit_constrain], pars, x=centers[fit_constrain], weights=1/np.sqrt(h_temp.values()[fit_constrain]))

                if out.success:
                    final_dict[f'res{val}'].append(out.params['center'].value)
                    final_dict[f'err{val}'].append(abs(out.params['sigma'].value))
                else:
                    final_dict[f'res{val}'].append(np.mean(df[val]))
                    final_dict[f'err{val}'].append(np.std(df[val]))
            except Exception as inst:
                print(inst)

    else:
        for idx, val in enumerate(columns):

            x_min = df[val].min()-5
            x_max = df[val].max()+5

            h_temp = hist.Hist(hist.axis.Regular(args.hist_bins, x_min, x_max, name="time_resolution", label=r'Time Resolution [ps]'))
            h_temp.fill(df[val])
            centers = h_temp.axes[0].centers

            fit_constrain = (centers > df[val].astype(int).mode()[0]-7) & (centers < df[val].astype(int).mode()[0]+7)

            final_dict[f'row{val}'].append(matches[idx][0])
            final_dict[f'col{val}'].append(matches[idx][1])

            try:
                pars = mod.guess(h_temp.values()[fit_constrain], x=centers[fit_constrain])
                out = mod.fit(h_temp.values()[fit_constrain], pars, x=centers[fit_constrain], weights=1/np.sqrt(h_temp.values()[fit_constrain]))

                if out.success:
                    final_dict[f'res{val}'].append(out.params['center'].value)
                    final_dict[f'err{val}'].append(abs(out.params['sigma'].value))
                else:
                    final_dict[f'res{val}'].append(np.mean(df[val]))
                    final_dict[f'err{val}'].append(np.std(df[val]))
            except Exception as inst:
                print(inst)

final_df = pd.DataFrame(final_dict)
final_df.to_csv(args.outname+'.csv', index=False)