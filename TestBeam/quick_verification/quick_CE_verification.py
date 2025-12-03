import os, sys
path2add = os.path.normpath(os.path.abspath(os.path.join(os.path.curdir, os.path.pardir)))
print(path2add)

if (not (path2add in sys.path)) :
    sys.path.append(path2add)

import BeamTestHelpers as helper
from pathlib import Path
from natsort import natsorted
import yaml

with open("./board_configs_yaml/CERN_TB_2025Oct_Suitcase.yaml", "r") as file:
    fig_configs = yaml.safe_load(file)

print(fig_configs.keys())

given_run = 'run19'
selected_fig_config = fig_configs[given_run]

for id in selected_fig_config.keys():
    selected_fig_config[id]['title'] = f"{selected_fig_config[id]['name']} HV{selected_fig_config[id]['HV']}V OS:{selected_fig_config[id]['offset']}"

roles = {}
for board_id, board_info in selected_fig_config.items():
    roles[board_info.get('role')] = board_id

for key, val in selected_fig_config.items():
    print(key, val)

path_to_files = "./"
files = natsorted(Path(path_to_files).glob('file*dat'))
df = helper.process_tamalero_outputs(files)
df.info()

# helper.plot_occupany_map(df, tb_loc='cern', board_ids=ids, board_names=names)
h_inclusive = helper.return_hist(input_df=df, board_info=selected_fig_config, hist_bins=[140, 128, 128])
chip_figtitles = [val['title'] for _, val in selected_fig_config.items()]
helper.plot_1d_TDC_histograms(h_inclusive, tb_loc='desy', fig_tag=chip_figtitles, slide_friendly=True)#, save_mother_dir=output_campaign_dir, tag='inclusive')