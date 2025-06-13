import pandas as pd
import numpy as np

## --------------------------------------
def return_coefficient_three_board_iterative_TWC(
    input_df: pd.DataFrame,
    iterative_cnt: int,
    poly_order: int,
    board_roles: dict,
):

    board_keys = sorted(board_roles.keys())
    corr_toas = {key: input_df[f'toa_{key}'].values for key in board_keys}

    ### Calculate delta of avg TOA
    delta_toas = {}
    for current_key in board_keys:
        others_sum = sum(corr_toas[other_key] for other_key in board_keys if other_key != current_key)
        delta_toas[current_key] = (0.5 * others_sum) - corr_toas[current_key]

    coeff_dict = {}
    for i in range(iterative_cnt):

        coeff_dict[f'iter{i+1}'] = {}
        corrections = {}

        for key in board_keys:
            coeff = np.polyfit(input_df[f'tot_{key}'].values, delta_toas[key], poly_order)
            poly_func = np.poly1d(coeff)

            coeff_dict[f'iter{i+1}'][key] = coeff.tolist()
            corrections[key] = poly_func(input_df[f'tot_{key}'].values)

        for key in board_keys:
            corr_toas[key] += corrections[key]

        for current_key in board_keys:
            others_sum = sum(corr_toas[other_key] for other_key in board_keys if other_key != current_key)
            delta_toas[current_key] = (0.5 * others_sum) - corr_toas[current_key]

    return coeff_dict

## --------------------------------------
if __name__ == "__main__":
    import argparse, pickle, yaml
    from natsort import natsorted
    from pathlib import Path

    parser = argparse.ArgumentParser(
                prog='bootstrap',
                description='find time resolution!',
            )

    parser.add_argument(
        '-d',
        '--input_dir',
        metavar = 'PATH',
        type = str,
        help = 'Path to input directory including time dataframes',
        required = True,
        dest = 'input_dir',
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
        '-c',
        '--config',
        metavar = 'NAME',
        type = str,
        help = 'YAML file including run information.',
        required = True,
        dest = 'config',
    )

    parser.add_argument(
        '--poly_order',
        metavar = 'NUM',
        type = int,
        help = "Decide the order of polynominal for TWC fit",
        default = 2,
        dest = 'poly_order',
    )

    parser.add_argument(
        '--iteration',
        metavar = 'NUM',
        type = int,
        help = 'Maximum iteration of TWC',
        default = 2,
        dest = 'iteration',
    )

    args = parser.parse_args()
    files = natsorted(Path(args.input_dir).glob('*.pkl'))

    with open(args.config) as input_yaml:
        config = yaml.safe_load(input_yaml)

    if args.runName not in config:
        raise ValueError(f"Run config {args.runName} not found")

    excluded_role = files[0].name.split('_')[1]
    roles = {}
    for board_id, board_info in config[args.runName].items():
        if excluded_role == board_info.get('role'):
            continue
        roles[board_info.get('role')] = board_id

    output = {}
    for ifile in files:
        track_name = ifile.name.split('.')[0].split('track_')[1]

        df = pd.read_pickle(ifile)
        single_track_twc_coeffs = return_coefficient_three_board_iterative_TWC(df, args.iteration, args.poly_order, roles)
        output[track_name] = single_track_twc_coeffs

    output_filename = f"{args.config.split('/')[-1].split('.')[0]}_{args.runName}_TWC_coeffs.pickle"

    with open(output_filename, 'wb') as pkl_file:
        pickle.dump(output, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)