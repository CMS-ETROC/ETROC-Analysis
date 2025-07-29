from collections import defaultdict
from natsort import natsorted
from tqdm import tqdm
from pathlib import Path
from scipy.stats import norm
import argparse
import pandas as pd
import re, yaml
import numpy as np
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='merge individual bootstrap results')
parser.add_argument('-d', '--inputdir', required=True, type=str, dest='inputdir')
parser.add_argument('-o', '--output', required=True, type=str, dest='output')
parser.add_argument('--minimum', type=int, default=50, help='Minimum number of bootstrap results to perform a fit', dest='minimum')
parser.add_argument('--hist_bins', type=int, default=35, help='Number of bins for binned fit', dest='hist_bins')
parser.add_argument(
    '--tag',
    metavar = 'NAME',
    type = str,
    help = 'Tag for the output file name.',
    default = '',
    dest = 'tag',
)

args = parser.parse_args()

nickname_dict = {
    't': 'trig',
    'd': 'dut',
    'r': 'ref',
    'e': 'extra',
}

files = natsorted(Path(args.inputdir).glob('*_resolution.pkl'))
final_dict = defaultdict(list)

excluded_role = files[0].name.split('_')[1]

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

if len(files) == 0:
    print('No input file found')

for ifile in tqdm(files):
    pattern = re.compile(r"(\w)-R(\d+)C(\d+)")
    matches = re.findall(pattern, str(ifile))

    pixel_dict = {}
    for nickname, row, col in matches:
        pixel_dict[nickname_dict[nickname]] = (row, col)

    df = pd.read_pickle(ifile)
    df = df.loc[(df != 0).all(axis=1)].reset_index(drop=True)

    if df.empty:
        continue

    file_results = {}
    # CASE 1: Single-row file from bootstrap fallback
    if df.shape[0] == 1:
        # print(f"\nINFO: Found single-row file (fallback result): {ifile.name}")
        single_row_values = df.iloc[0]
        for col_name in df.columns:
            # For a single result, mu is the value and sigma (error) is 0
            file_results[col_name] = {'mu': single_row_values[col_name], 'sigma': 0}

    # CASE 2: Standard file with enough bootstrap results to fit
    else:
        df.reset_index(drop=True, inplace=True)
        for col_name in df.columns:
            mu, sigma, unbinned_check = fit_unbinned(df[col_name])
            if not unbinned_check:
                # If unbinned fit fails, use the more robust binned fit
                mu, sigma, _ = fit_binned(df[col_name])
            file_results[col_name] = {'mu': mu, 'sigma': sigma}

    # --- This block now handles appending for all successful cases ---
    if len(pixel_dict.keys()) == 4:
        final_dict[f'row_{excluded_role}'].append(pixel_dict[excluded_role][0])
        final_dict[f'col_{excluded_role}'].append(pixel_dict[excluded_role][1])

        for val_name, results in file_results.items():
            final_dict[f'row_{val_name}'].append(pixel_dict[val_name][0])
            final_dict[f'col_{val_name}'].append(pixel_dict[val_name][1])
            final_dict[f'res_{val_name}'].append(results['mu'])
            final_dict[f'err_{val_name}'].append(results['sigma'])
    else:
         for val_name, results in file_results.items():
            final_dict[f'row_{val_name}'].append(pixel_dict[val_name][0])
            final_dict[f'col_{val_name}'].append(pixel_dict[val_name][1])
            final_dict[f'res_{val_name}'].append(results['mu'])
            final_dict[f'err_{val_name}'].append(results['sigma'])

pd.DataFrame(final_dict).to_csv('resolution_' + args.output + args.tag + '.csv', index=False)
