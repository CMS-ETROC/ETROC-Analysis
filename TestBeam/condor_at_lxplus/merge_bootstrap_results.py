from collections import defaultdict
from natsort import natsorted
from tqdm import tqdm
from pathlib import Path
from scipy.stats import norm
import argparse
import pandas as pd
import re
import numpy as np
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='merge individual bootstrap results')
parser.add_argument('-d', '--inputdir', required=True, type=str, dest='inputdir')
parser.add_argument('-o', '--output', required=True, type=str, dest='output')
parser.add_argument('--trigID', required=True, type=int, help='original Trigger board ID', dest='trigID')
parser.add_argument('--minimum', type=int, default=50, help='Minimum number of bootstrap results to perform a fit', dest='minimum')
parser.add_argument('--hist_bins', type=int, default=35, help='Number of bins for binned fit', dest='hist_bins')
args = parser.parse_args()

files = natsorted(Path(args.inputdir).glob('*pkl'))
final_dict = defaultdict(list)

def fit_unbinned(data):
    try:
        mu, sigma = norm.fit(data)

        ### Future development: Set threshold for unbinned fit
        # Calculate the MSE between data and fitted PDF
        # pdf_values = norm.pdf(data, mu, sigma)
        # mse = np.mean((data - pdf_values)**2)

        # Log-Likelihood
        # log_likelihood = np.sum(np.log(pdf_values))

        #print(f" {mu}, {sigma}")
        #print(f"   MSE: {mse}, Log-Likelihood: {log_likelihood}")

        return mu, sigma, True

    except Exception as e:
        print(e)
        return np.mean(data), np.std(data), False

def fit_binned(data):
    try:
        import hist
        x_min = data.min()-5
        x_max = data.max()+5
        h_temp = hist.Hist(hist.axis.Regular(args.hist_bins, x_min, x_max))
        h_temp.fill(data)
        centers = h_temp.axes[0].centers
        values = h_temp.values()
        lower_bound = np.percentile(data, 17) # left percentile for 1.5 sigma
        upper_bound = np.percentile(data, 83) # right percentile for 1.5 sigma
        fit_mask = (centers > lower_bound) & (centers < upper_bound)

        from lmfit.models import GaussianModel
        mod = GaussianModel(nan_policy='omit')
        pars = mod.guess(values[fit_mask], x=centers[fit_mask])
        out = mod.fit(values[fit_mask], pars, x=centers[fit_mask], weights=1/np.sqrt(values[fit_mask]))

        if out.success:
            return out.params['center'].value, abs(out.params['sigma'].value), True

    except Exception as e:
        print(f"Error: {e}")
        return np.mean(data), np.std(data), False

for ifile in tqdm(files):
    pattern = r'R(\d+)C(\d+)'
    match_dict = {i: val for i, val in enumerate(re.findall(pattern, str(ifile)))}
    df = pd.read_pickle(ifile)
    df = df.loc[(df != 0).all(axis=1)]

    if df.shape[0] < args.minimum:
        continue

    df.reset_index(drop=True, inplace=True)
    columns = df.columns

    if len(match_dict.keys()) == 4:
        final_dict[f'row{args.trigID}'].append(match_dict[args.trigID][0])
        final_dict[f'col{args.trigID}'].append(match_dict[args.trigID][1])

    for val in columns:
        final_dict[f'row{val}'].append(match_dict[val][0])
        final_dict[f'col{val}'].append(match_dict[val][1])

        mu, sigma, unbinned_check = fit_unbinned(df[val])
        if not unbinned_check:
            mu, sigma, _ = fit_binned(df[val])

        final_dict[f'res{val}'].append(mu)
        final_dict[f'err{val}'].append(sigma)

pd.DataFrame(final_dict).to_csv('resolution_' + args.output + '.csv', index=False)