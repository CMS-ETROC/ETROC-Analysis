import argparse, re, warnings

import pandas as pd
import numpy as np

from pathlib import Path
from collections import defaultdict
from natsort import natsorted
from tqdm import tqdm
from scipy.stats import norm
from iminuit import Minuit

# --- Configuration ---
warnings.filterwarnings("ignore")

NICKNAME_DICT = {'t': 'trig', 'd': 'dut', 'r': 'ref', 'e': 'extra'}
FILENAME_PATTERN = re.compile(r"(\w)-R(\d+)C(\d+)")

# --- Core Fitting Logic ---

def nll_gaussian(data, mu, sigma):
    """Negative Log-Likelihood for a Gaussian distribution."""
    # Using logpdf is numerically stable for unbinned fits
    return -np.sum(norm.logpdf(data, loc=mu, scale=sigma))

def calculate_fit_quality(data, mu_fit, sigma_fit, bins=30):
    """Calculates Reduced Chi2 and Pull statistics for an unbinned result."""
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    # Expected counts based on the unbinned fit result
    expected = len(data) * bin_width * norm.pdf(bin_centers, mu_fit, sigma_fit)

    # Calculate Chi2 (only on bins with content)
    mask = expected > 0
    pulls = (counts[mask] - expected[mask]) / np.sqrt(expected[mask])
    chi2 = np.sum(pulls**2)
    ndf = len(counts[mask]) - 2

    return chi2 / ndf if ndf > 0 else 999

def perform_robust_unbinned_fit(data: pd.Series, sigma_cut: float):
    """
    Two-step process:
    1. Use Median/IQR for robust scouting (ignores outliers).
    2. Perform Unbinned ML fit on the filtered 'signal region'.
    """
    vals = data.values

    # 1. Robust Estimators
    mu_init = np.median(vals)
    q75, q25 = np.percentile(vals, [75, 25])
    sigma_init = (q75 - q25) / 1.349
    if sigma_init == 0: sigma_init = vals.std() + 1e-6

    # 2. Filtering Outliers (The 'Shield')
    mask = (vals >= mu_init - sigma_cut * sigma_init) & \
           (vals <= mu_init + sigma_cut * sigma_init)
    masked_data = vals[mask]

    if len(masked_data) < 5:
        return {'mu': mu_init, 'sigma': sigma_init, 'err_sigma': 0, 'chi2_red': 999, 'div': 999, 'valid': 0}

    # 3. Iminuit Fit
    try:
        m = Minuit(lambda mu, sigma: nll_gaussian(masked_data, mu, sigma),
                   mu=mu_init, sigma=sigma_init)
        m.errordef = Minuit.LIKELIHOOD
        m.limits["sigma"] = (1e-6, None)

        m.migrad()
        m.hesse()

        mu_f, sigma_f = m.values['mu'], m.values['sigma']
        sigma_err = m.errors['sigma']

        # 4. Audit Metrics
        red_chi2 = calculate_fit_quality(masked_data, mu_f, sigma_f)
        # Divergence: How far did the fit move from the median?
        divergence = abs(mu_f - mu_init) / sigma_f if sigma_f > 0 else 999

        return {
            'mu': mu_f,
            'sigma': sigma_f,
            'err_sigma': sigma_err,
            'chi2_red': red_chi2,
            'div': divergence,
            'valid': 1 if m.fmin.is_valid else 0,
        }

    except Exception:
        return {'mu': mu_init, 'sigma': sigma_init, 'err_sigma': 0, 'chi2_red': 999, 'div': 999, 'valid': 0}

# --- Data Processing ---

def parse_filename_metadata(filename: str) -> dict:
    matches = re.findall(FILENAME_PATTERN, filename)
    pixel_dict = {}
    for nickname, row, col in matches:
        full_role = NICKNAME_DICT.get(nickname)
        if full_role:
            pixel_dict[full_role] = (int(row), int(col))
    return pixel_dict

def process_group(input_dir: Path, output_dir: Path, args: argparse.Namespace):
    files = natsorted(input_dir.glob('*_boot.parquet'))
    if not files: return

    excluded_role = files[0].name.split('_')[1] if '_' in files[0].name else 'trig'
    boot_dict = defaultdict(list)

    for ifile in tqdm(files, desc=f"  Merging {input_dir.name}"):
        df = pd.read_parquet(ifile)
        if df.empty: continue

        pixel_dict = parse_filename_metadata(ifile.name)
        boot_df = df.loc[df['is_bootstrap'] == True]
        anchor_df = df.loc[df['is_bootstrap'] == False]

        if excluded_role in pixel_dict:
            boot_dict[f'row_{excluded_role}'].append(pixel_dict[excluded_role][0])
            boot_dict[f'col_{excluded_role}'].append(pixel_dict[excluded_role][1])

        for col in boot_df.columns:
            if col == 'is_bootstrap': continue

            # FIT STEP
            stats = perform_robust_unbinned_fit(boot_df[col], args.sigma_cut)

            if col in pixel_dict:
                boot_dict[f'row_{col}'].append(pixel_dict[col][0])
                boot_dict[f'col_{col}'].append(pixel_dict[col][1])

            # Standard Results
            boot_dict[f'res_{col}'].append(stats['mu'])
            boot_dict[f'err_{col}'].append(stats['sigma'])

            # AUDIT METRICS
            boot_dict[f'rel_error_{col}'].append(stats['err_sigma']/stats['sigma'])
            boot_dict[f'red_chi2_{col}'].append(stats['chi2_red'])
            boot_dict[f'div_{col}'].append(stats['div'])
            boot_dict[f'fit_valid_{col}'].append(stats['valid'])

        for col in anchor_df.columns:
            if col == 'is_bootstrap': continue
            boot_dict[f'single_shot_res_{col}'].append(anchor_df[col].iloc[0])

    if boot_dict:
        out_path = output_dir / f"resolution_table{args.tag}.csv"
        pd.DataFrame(boot_dict).to_csv(out_path, index=False)

# --- Main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--inputdir', required=True)
    parser.add_argument('-o', '--outputdir', required=True)
    parser.add_argument('--sigma_cut', type=float, default=2.5, help='Sigma window for filtered fit')
    parser.add_argument('--tag', default='')
    args = parser.parse_args()

    group_dirs = [Path(args.inputdir).resolve()]

    out_dir = Path(args.outputdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for group in group_dirs:
        process_group(group, out_dir, args)

if __name__ == "__main__":
    main()