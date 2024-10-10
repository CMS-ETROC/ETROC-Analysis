#!/bin/bash

### Every python script supports "-h". If you want to know details, please use it:
### example: python submit_track_data_selection.py -h

###################################################
### Step 0: Find track candidates - No need to run several times
### -p: path to input directories
### -o: output file name (in csv format)
### -i: how many iteration
### -s: random sampling fraction, 20 means 20% of full dataset
### -m: minimum number of track cut
### You can keep other options

#python finding_good_track_candidates.py -p <path> -o <output name> -i 5 -s 20 -m 500 --trigID 0 --refID 3 --dutID 1 --ignoreID 2 --four_board

###################################################


###################################################
### Step 1: Separate data by the given track
### -d: path to input directory
### -t: input track info (csv file)
### -o: output directory name
### --trigTOTLower, --trigTOTUpper: TOT cut for the trigger hit
### You can keep other options

#python submit_track_data_selection.py -d /eos/user/j/jongho/ETROC_CERN_Aug2024/MarcosBoardHV${voltage}/${feathers} -t CERN_TB_Aug2024_good_track_candidates_4boards_v2.csv --trigID 0 --refID 3 --dutID 1 --ignoreID 2 --trigTOTLower 80 --trigTOTUpper 160 --load_from_eos -o CERNAug2024_210V_offset20_cooling_23C
###################################################


###################################################
### Step 2: Separate data by the given track
### -d: input directory which is output directory from step 1
### -o: output directory name
### --file_pattern: if you want to select the file (e.g. first 2 hours, specific run, etc...), then use this option.
### You can keep other options

#python apply_TDC_cuts_and_convert_to_time.py -d CERNAug2024_210V_offset20_cooling_23C -o ${dname}_4boards_${voltage} --setTrigBoardID 1 --setDUTBoardID 2 --setRefBoardID 3 --trigTOALower 250 --trigTOAUpper 500 --autoTOTcuts --file_pattern "run*.pickle"
###################################################


###################################################
### Step 3: Find number of events in each track
### -d: input directory <name>/tracks or <name>/time (tracks: before cut, times: after cut / I usually use tracks)
### -o: output name

#python extract_nevt_per_track.py -d <name>/tracks -o <output>
###################################################


###################################################
### Step 4: Run bootstrap by submitting on condor 
### -d: input directory
### -o: output directory (name: resolution_<argument>)
### -n: number of bootstrap output
### --minimum_nevt: minimum number of events to do bootstrap after 75% of random sampling
### --trigTOALower, --trigTOAUpper: TOA range to analyze
### --time_df_input: if input directory is <name>/time, you must give this option
### --board_ids: board IDs to analyze
### --board_id_for_TOA_cut: which board ID to apply TOA range cut
### --iteration_limit: Maximum number of sampling

#python submit_bootstrap.py -d ${dname}_4boards_${voltage}/time -o ${dname}_4boards_${voltage} -n 100 --minimum_nevt 250 --trigTOALower 250 --trigTOAUpper 500 --time_df_input --board_ids 1 2 3 --board_id_for_TOA_cut 1 --iteration_limit 10000
###################################################


###################################################
### Step 5: Merge bootstrap results 
### -d: input directory
### -o: output name
### --minimum: minimum cut for # of bootstrap results
### --hist_bins: histogram bin

### toa="TOA250to500"
### mv resolution_${dname}_4boards_${voltage} resolution_${dname}_4boards_${voltage}_${toa}
### python merge_bootstrap_results.py -d resolution_${dname}_4boards_${voltage}_${toa} -o resolution_${dname}_4boards_${voltage}_${toa} --minimum 40 --hist_bins 35
###################################################
