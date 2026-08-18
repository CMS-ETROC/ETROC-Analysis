import argparse, re, sys, warnings

import pandas as pd
import numpy as np

from pathlib import Path
from collections import defaultdict
from natsort import natsorted
from tqdm import tqdm
from scipy.stats import norm
from iminuit import Minuit

import io_utils

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

def process_single_boot_file(ifile: Path, args: argparse.Namespace) -> dict:
    """
    Processes one *_boot.parquet file and returns its contribution to the merged
    result dict as a plain dict (one file = one row), or {} if there's nothing to
    contribute. Built as a standalone dict rather than appending straight into the
    caller's column-lists, so a failure partway through never leaves some columns
    with more entries than others.
    """
    df = pd.read_parquet(ifile)
    if df.empty:
        return {}

    pixel_dict = parse_filename_metadata(ifile.name)
    boot_df = df.loc[df['is_bootstrap'] == True]
    anchor_df = df.loc[df['is_bootstrap'] == False]

    contribution = {}

    for col in boot_df.columns:
        if col == 'is_bootstrap': continue
        if col.startswith('ksp'):
            # the per-sample KS p-value floor is a diagnostic, not a resolution: carry its
            # single-shot value and its median over the accepted resamples, no Gaussian fit
            contribution['ksp_min_single_shot'] = float(anchor_df[col].iloc[0]) if len(anchor_df) else float('nan')
            contribution['ksp_min_boot_median'] = float(boot_df[col].median())
            continue

        # FIT STEP
        stats = perform_robust_unbinned_fit(boot_df[col], args.sigma_cut)

        if col in pixel_dict:
            contribution[f'row_{col}'] = pixel_dict[col][0]
            contribution[f'col_{col}'] = pixel_dict[col][1]

        # Standard Results
        contribution[f'res_{col}'] = stats['mu']
        contribution[f'err_{col}'] = stats['sigma']

        # AUDIT METRICS
        contribution[f'rel_error_{col}'] = stats['err_sigma']/stats['sigma']
        contribution[f'red_chi2_{col}'] = stats['chi2_red']
        contribution[f'div_{col}'] = stats['div']
        contribution[f'fit_valid_{col}'] = stats['valid']

    for col in anchor_df.columns:
        if col == 'is_bootstrap' or col.startswith('ksp'): continue
        contribution[f'single_shot_res_{col}'] = anchor_df[col].iloc[0]
        # A -1 single-shot means step 12 failed every full-sample attempt for this
        # track (KS or imaginary 3-board solve); its res_/err_ above still come from
        # whatever bootstrap resamples were accepted and can be far from physical
        # (e.g. an imaginary solve barely turned real) -- flag it so downstream
        # tables/plots can exclude or mark such tracks instead of trusting them.
        contribution[f'single_shot_failed_{col}'] = int(anchor_df[col].iloc[0] < 0)
    boot_failed = int(len(boot_df) == 1 and (boot_df.iloc[0].drop(labels=['is_bootstrap'], errors='ignore') < 0).all())
    contribution['boot_failed'] = boot_failed
    contribution['n_boot'] = 0 if boot_failed else int(len(boot_df))   # accepted resamples (the -1 placeholder is not one)

    return contribution

def find_input_dirs(mother_dir: Path):
    """Resolves -d to the list of boot-file directories to process.

    If mother_dir itself directly contains *_boot.parquet files, it's already
    a single group directory (step 12 writes one such directory per
    combo/group, e.g. bootstrap_<outputdir>/<combo_label>[_groupX]) and is
    returned as the sole entry, with is_direct=True so the output filename
    stays exactly as before (no label). Otherwise mother_dir is treated as
    the combo mother directory: every subdirectory that actually contains
    boot files is entered, is_direct=False, and each gets a labeled output
    filename (mirroring step 11) so results from different combos/groups
    saved to the same -o never collide.
    """
    if list(mother_dir.glob('*_boot.parquet')):
        return [mother_dir], True

    sub_dirs = [d for d in sorted(mother_dir.iterdir()) if d.is_dir() and list(d.glob('*_boot.parquet'))]
    if not sub_dirs:
        sys.exit(f"Error: No '*_boot.parquet' files found in {mother_dir} or its subdirectories.")
    return sub_dirs, False

def process_group(input_dir: Path, output_dir: Path, label, args: argparse.Namespace) -> int:
    files = natsorted(input_dir.glob('*_boot.parquet'))
    if not files: return 0

    # One dict per file, framed at the end: files with different column sets (e.g. boot
    # files written before/after bootstrap.py started recording pair widths and ksp_min)
    # then simply get NaN in the columns they lack instead of a ragged-lists crash.
    rows = []

    desc = f'{label}' if label else input_dir.name
    failures = 0
    for ifile in tqdm(files, desc=f"  Merging {desc}"):
        try:
            contribution = process_single_boot_file(ifile, args)
        except Exception as e:
            print(f"  Warning: failed to process {ifile.name}: {e}")
            failures += 1
            continue

        if not contribution:
            continue

        rows.append(contribution)

    if failures:
        print(f"  Warning: {failures}/{len(files)} file(s) FAILED to process in {desc}")

    if rows:
        # Combo/group label (if any) goes right after "resolution_table" so two
        # combos/groups saved to the same -o never overwrite each other's CSV --
        # same convention as step 11's nevt_<combo_label>_... filenames.
        base = f"resolution_table_{label}" if label else "resolution_table"
        out_path = output_dir / f"{base}{args.tag}.csv"
        io_utils.write_csv(pd.DataFrame(rows), out_path, index=False)

    return failures

# --- Main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--inputdir', required=True,
                        help='Directory of step 12 boot-file output for one combo/group. Can also be the '
                             'combo mother directory (one subdirectory per board combo/group, each holding '
                             '*_boot.parquet files) -- every one is then auto-detected and processed, with '
                             'output CSVs distinguished by label, same as step 11.')
    parser.add_argument('-o', '--outputdir', required=True)
    parser.add_argument('--sigma_cut', type=float, default=2.5, help='Sigma window for filtered fit')
    parser.add_argument('--tag', default='')
    args = parser.parse_args()

    input_dir = Path(args.inputdir).resolve()

    out_dir = Path(args.outputdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    group_dirs, is_direct = find_input_dirs(input_dir)

    total_failures = 0
    for group_dir in group_dirs:
        label = None if is_direct else group_dir.name
        total_failures += process_group(group_dir, out_dir, label, args)

    if total_failures:
        sys.exit(1)

if __name__ == "__main__":
    main()