import pandas as pd
import numpy as np

## --------------------------------------
def return_coefficient_three_board_iterative_TWC(
    input_df: pd.DataFrame,
    iterative_cnt: int,
    poly_order: int,
    board_list: list,
):

    coeff_dict = {}

    corr_b0 = input_df[f'toa_b{board_list[0]}'].values
    corr_b1 = input_df[f'toa_b{board_list[1]}'].values
    corr_b2 = input_df[f'toa_b{board_list[2]}'].values

    del_toa_b0 = (0.5*(input_df[f'toa_b{board_list[1]}'] + input_df[f'toa_b{board_list[2]}']) - input_df[f'toa_b{board_list[0]}']).values
    del_toa_b1 = (0.5*(input_df[f'toa_b{board_list[0]}'] + input_df[f'toa_b{board_list[2]}']) - input_df[f'toa_b{board_list[1]}']).values
    del_toa_b2 = (0.5*(input_df[f'toa_b{board_list[0]}'] + input_df[f'toa_b{board_list[1]}']) - input_df[f'toa_b{board_list[2]}']).values

    for i in range(iterative_cnt):

        coeff_dict[f'iter{i+1}'] = {
            f'board_{board_list[0]}': None,
            f'board_{board_list[1]}': None,
            f'board_{board_list[2]}': None,
        }

        coeff_b0 = np.polyfit(input_df[f'tot_b{board_list[0]}'].values, del_toa_b0, poly_order)
        poly_func_b0 = np.poly1d(coeff_b0)

        coeff_b1 = np.polyfit(input_df[f'tot_b{board_list[1]}'].values, del_toa_b1, poly_order)
        poly_func_b1 = np.poly1d(coeff_b1)

        coeff_b2 = np.polyfit(input_df[f'tot_b{board_list[2]}'].values, del_toa_b2, poly_order)
        poly_func_b2 = np.poly1d(coeff_b2)

        coeff_dict[f'iter{i+1}'][f'board_{board_list[0]}'] = coeff_b0.tolist()
        coeff_dict[f'iter{i+1}'][f'board_{board_list[1]}'] = coeff_b1.tolist()
        coeff_dict[f'iter{i+1}'][f'board_{board_list[2]}'] = coeff_b2.tolist()

        corr_b0 = corr_b0 + poly_func_b0(input_df[f'tot_b{board_list[0]}'].values)
        corr_b1 = corr_b1 + poly_func_b1(input_df[f'tot_b{board_list[1]}'].values)
        corr_b2 = corr_b2 + poly_func_b2(input_df[f'tot_b{board_list[2]}'].values)

        del_toa_b0 = (0.5*(corr_b1 + corr_b2) - corr_b0)
        del_toa_b1 = (0.5*(corr_b0 + corr_b2) - corr_b1)
        del_toa_b2 = (0.5*(corr_b0 + corr_b1) - corr_b2)

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
        '--exclude_role',
        metavar = 'NAME',
        type = str,
        help = "Choose the board to exclude for calculating TWC coeffs. Possible option: 'trig', 'dut', 'ref', 'extra'",
        required = True,
        dest = 'exclude_role',
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
        default = 5,
        dest = 'iteration',
    )

    args = parser.parse_args()
    files = natsorted(Path(args.input_dir).glob('*.pkl'))

    with open(args.config) as input_yaml:
        config = yaml.safe_load(input_yaml)

    if args.runName not in config:
        raise ValueError(f"Run config {args.runName} not found")

    board_ids = []
    for board_id, board_info in config[args.runName].items():
        if args.exclude_role == board_info.get('role'):
            continue
        board_ids.append(board_id)
    board_ids = sorted(board_ids)

    output = {}
    for ifile in files:
        track_name = ifile.name.split('.')[0].split('track_')[1]

        df = pd.read_pickle(ifile)
        single_track_twc_coeffs = return_coefficient_three_board_iterative_TWC(df, args.iteration, args.poly_order, board_ids)
        output[track_name] = single_track_twc_coeffs

    output_filename = f"{args.config.split('/')[-1].split('.')[0]}_{args.runName}_TWC_coeffs.pickle"

    with open(output_filename, 'wb') as pkl_file:
        pickle.dump(output, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)