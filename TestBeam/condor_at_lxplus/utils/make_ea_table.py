from pathlib import Path
from natsort import natsorted
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

#def process_single_file(ifile):
#    """Worker function to process a single feather file."""
#    # Reading only the necessary columns to save memory
#    df = pd.read_feather(ifile, columns=['board', 'ea'])
#    # Group locally within the file to reduce data size before returning
#    return df.groupby(['board', 'ea']).size().reset_index(name='count')


def process_single_file(ifile):
    """Worker function to process a single feather file."""
    df = pd.read_feather(ifile)
    df = df.loc[df['ea'] != 0].reset_index(drop=True)
    return df

def main():
    run_dirs = natsorted(Path('/eos/home-j/jongho/ETROC_DESY_Dec2025/greybox').glob('Run_*_feather'))
    outdir = Path('EA_tables')
    outdir.mkdir(exist_ok=True)

    for idir in run_dirs:
        files = natsorted(idir.glob('loop*feather'))
        prefix = idir.name.split('_feather')[0]

        print(f"Processing {prefix}...")

        # Use ProcessPoolExecutor for CPU-bound tasks (Pandas grouping)
        # It defaults to the number of processors on the machine
        with ProcessPoolExecutor(max_workers=8) as executor:
            # We use list() to force the execution and wrap it in tqdm
            all_dfs = list(tqdm(executor.map(process_single_file, files), total=len(files)))

        if all_dfs:
            # Combine all small count tables and aggregate
            #final_df = pd.concat(all_dfs).groupby(['board', 'ea'], as_index=False)['count'].sum()
            final_df = pd.concat(all_dfs)
            final_df.to_csv(outdir / f'{prefix}_table.csv', index=False)

if __name__ == "__main__":
    main()
