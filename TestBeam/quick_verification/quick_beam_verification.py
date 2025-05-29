from natsort import natsorted
from pathlib import Path
import sys, os

path2add = os.path.normpath(os.path.abspath(os.path.join(os.path.curdir, os.path.pardir)))
print(path2add)

if (not (path2add in sys.path)) :
    sys.path.append(path2add)

from beamtest_analysis_helper import DecodeBinary, plot_occupany_map, return_hist, plot_1d_TDC_histograms

# ### In case, you're using "nem" files
# files = Path('/media/daq/X9/Run_nominal_alignment_12').glob('loop_*/*nem')
# print(files[0])
# print(files[-1])
# df = toSingleDataFrame_newEventModel(files=files)

### In case, you're converting from binary files
input_path = "/media/daq/X9/Run_DESYFeb2024_Run_34/"
pattern = 'Run_*/file*.bin'
files = natsorted(list(Path(input_path).glob(pattern)))
print(files[0], files[-1])
decode = DecodeBinary(firmware_key=0b0001, board_id=[0x17f0f, 0x17f0f, 0x17f0f, 0x17f0f], file_list=files, skip_crc_df=True, skip_event_df=True, skip_fw_filler=True)
df, _, _, _ = decode.decode_files() # hit dataframe, event dataframe, CRC dataframe, fillter dataframe
print(f'Nhits: {df.shape[0]}, Nevts: {df['evt'].nunique()}')

chip_labels = [0]
chip_names = ["ET2p01_Bar19"]
offsets = [20]
high_voltages = [230]

chip_fignames = chip_names
chip_figtitles = [
    f"(Trigger) Bar 19 HV{high_voltages[0]}V OS:{offsets[0]}",
]

output_mother_dir = Path('./') / 'etroc_TB_figs'
output_mother_dir.mkdir(exist_ok=True, parents=True)

### Now you need change the directory name per campaign
### Naming rule is this:
### <TB location>_TB_MonthYear
### E.g. desy_TB_Apr2024, cern_TB_Sep2023, fnal_TB_Jul2024

output_campaign_dir = output_mother_dir / 'cern_TB_May2025'
output_campaign_dir.mkdir(exist_ok=True)

### Make plots
plot_occupany_map(df, board_ids=chip_labels, board_names=chip_names, tb_loc='cern', save_mother_dir=output_campaign_dir)

h_inclusive = return_hist(input_df=df, board_names=chip_names, board_ids=chip_labels, hist_bins=[100, 128, 128])
for iboard in chip_labels:
    plot_1d_TDC_histograms(input_hist=h_inclusive, board_name=chip_names[iboard], tb_loc='cern', fig_tag=chip_figtitles[iboard],
                           slide_friendly=True, save_mother_dir=output_campaign_dir)
del h_inclusive
