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
    '-c',
    '--config',
    metavar = 'NAME',
    type = str,
    help = 'YAML file including run information.',
    required = True,
    dest = 'config',
)
parser.add_argument(
    '-r',
    '--runName',
    metavar = 'NAME',
    type = str,
    help = 'Name of the run to process. It must be matched with the name defined in YAML.',
    required = True,
    dest = 'runName',
)
parser.add_argument(
    '--tag',
    metavar = 'NAME',
    type = str,
    help = 'Tag for the output file name.',
    default = '',
    dest = 'tag',
)

args = parser.parse_args()

with open(args.config) as input_yaml:
    config = yaml.safe_load(input_yaml)

if args.runName not in config:
    raise ValueError(f"Run config {args.runName} not found")

files = natsorted(Path(args.inputdir).glob('*_resolution.pkl'))
final_dict = defaultdict(list)

excluded_role = files[0].name.split('_')[1]

roles = {}
for board_id, board_info in config[args.runName].items():
    roles[board_info.get('role')] = board_id

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
    pattern = r'R(\d+)C(\d+)'
    match_dict = {i: val for i, val in enumerate(re.findall(pattern, str(ifile)))}
    df = pd.read_pickle(ifile)
    df = df.loc[(df != 0).all(axis=1)]

    if df.shape[0] < args.minimum:
        continue

    df.reset_index(drop=True, inplace=True)
    columns = df.columns

    if len(match_dict.keys()) == 4:
        final_dict[f'row_{excluded_role}'].append(match_dict[roles[excluded_role]][0])
        final_dict[f'col_{excluded_role}'].append(match_dict[roles[excluded_role]][1])

    for val in columns:
        final_dict[f'row_{val}'].append(match_dict[roles[val]][0])
        final_dict[f'col_{val}'].append(match_dict[roles[val]][1])

        mu, sigma, unbinned_check = fit_unbinned(df[val])
        if not unbinned_check:
            mu, sigma, _ = fit_binned(df[val])

        final_dict[f'res_{val}'].append(mu)
        final_dict[f'err_{val}'].append(sigma)

pd.DataFrame(final_dict).to_csv('resolution_' + args.output + args.tag + '.csv', index=False)