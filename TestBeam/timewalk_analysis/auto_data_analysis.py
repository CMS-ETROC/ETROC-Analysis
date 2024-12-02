import subprocess
import os
import fnmatch
from datetime import datetime
import numpy as np

# RunNums = [62,63,64,65]
# RunNums = [60]
RunNums = [61,62,63,64,65]
RunNums2 = [58,59,61]

# print('Finding Good Track Candidates')
# print('-----------------------------')
# subprocess.run(['python','finding_good_track_candidates.py','-p',f'../Test_Data/Run_{RunNums[0]}/','-o','FeatherData','-i','5','-s','20','-m','100','--trigID','0','--refID','3','--dutID','1','--ignoreID','2','--four_board'])

# for RunNum in RunNums:

    # print(' ')
    # print(f'Running analysis for run {RunNum}')

    # ################################################################################

    # print('Track Data Selection')
    # print('-----------------------------')

    # ffiledir = f'/media/quantumtyrant/ODRIVE/FeatherData/Run_{RunNum}_feather/'
    # ffiles = fnmatch.filter(os.listdir(ffiledir), 'loop_*.feather')
    # for i in range(len(ffiles)):
    #     ffiles[i] = int(ffiles[i].removesuffix('.feather').removeprefix('loop_'))
    # ffiles.sort()

    # for i in ffiles:
    #     subprocess.run(['cp',f'{ffiledir}loop_{i}.feather','./temp'])
    
    # if RunNum <= 54:
    #     subprocess.run(['python','track_data_selection_futures.py','-d',f'./temp/','-r',f'run_{RunNum}','-t','FeatherData.csv','--trigID','0','--refID','3','--dutID','1','--ignoreID','2','--trigTOTLower','80','--trigTOTUpper','160'])
    # if RunNum > 54:
    #     subprocess.run(['python','track_data_selection_futures.py','-d',f'./temp/','-r',f'run_{RunNum}','-t','FeatherData.csv','--trigID','3','--refID','0','--dutID','1','--ignoreID','2','--trigTOTLower','100','--trigTOTUpper','200'])
    # # subprocess.run(['python','track_data_selection.py','-f',f'loop_{i}.feather','-r',f'run_{RunNum}','-t','FeatherData.csv','--trigID','0','--refID','3','--dutID','1','--ignoreID','2','--trigTOTLower','80','--trigTOTUpper','160'])
    # for i in ffiles:
    #     subprocess.run(['mv',f'run_{RunNum}_loop_{i}.pickle',f'./test_inputs/run_{RunNum}/'])
    #     subprocess.run(['rm',f'./temp/loop_{i}.feather'])

    
    ################################################################################

for RunNum in RunNums2:

    print(' ')
    print(f'Running analysis for run {RunNum}')

    pfiledir = f'./test_inputs/run_{RunNum}/'
    pfiles = fnmatch.filter(os.listdir(pfiledir), f'run_{RunNum}_loop_*.pickle')
    for i in range(len(pfiles)):
        pfiles[i] = int(pfiles[i].removesuffix('.pickle').removeprefix(f'run_{RunNum}_loop_'))
    pfiles.sort()

    # print(pfiles)

    split = [pfiles[i:i + 7] for i in range(0, len(pfiles), 7)]
    split[-2] = split[-2] + split[-1]
    del split[-1]

    for x in split:
        feather_to_read = x
        # feather_to_read = pfiles

        now = datetime.now()
        timenow = f'Sep_{now.day}_{now.hour}-{now.minute}'

        for i in feather_to_read:
            subprocess.run(['cp',f'./test_inputs/run_{RunNum}/run_{RunNum}_loop_{i}.pickle','./temp'])

        print('Applying TDC Cuts and Converting to Time')
        print('-----------------------------')
        # Runs 45-53
        if RunNum <= 54:
            subprocess.run(['python','apply_TDC_cuts_and_convert_to_time.py','-d','./temp','-o',f'test_outputs/run_{RunNum}/{timenow}','--setTrigBoardID','1','--setDUTBoardID','2','--setRefBoardID','3','--trigTOALower','250','--trigTOAUpper','500','--autoTOTcuts'])
        # Runs 58-65
        if RunNum > 54:
            subprocess.run(['python','apply_TDC_cuts_and_convert_to_time.py','-d','./temp','-o',f'test_outputs/run_{RunNum}/{timenow}','--setTrigBoardID','0','--setDUTBoardID','1','--setRefBoardID','2','--trigTOALower','250','--trigTOAUpper','500','--autoTOTcuts'])

        for i in feather_to_read:
            subprocess.run(['rm',f'./temp/run_{RunNum}_loop_{i}.pickle'])

    ################################################################################


        # directory = f'./test_outputs/run_{RunNum}/{timenow}/time/'
        # largest_pickle_file = max(
        # (os.path.join(root, file) for root, dirs, files in os.walk(directory) for file in files),
        # key=os.path.getsize
        # )
        # largest_pickle_file = largest_pickle_file.removesuffix('.pkl').removeprefix(directory)
        # #print(largest_pickle_file)

        # print('Running Bootstrap')
        # print('-----------------------------')
        # subprocess.run(['cp',f'{directory}/{largest_pickle_file}.pkl','./'])
        # subprocess.run(['python','bootstrap.py','-f',f'{largest_pickle_file}.pkl','-i','100','-s','75','--board_id_for_TOA_cut','1','--minimum_nevt','100','--trigTOALower','250','--trigTOAUpper','500','--board_ids','1','2','3','--time_df_input'])
        # subprocess.run(['rm',f'{largest_pickle_file}.pkl'])

        # subprocess.run(['mkdir',f'Resolution_Data/run_{RunNum}/{timenow}'])
        # subprocess.run(['mv',f'{largest_pickle_file}_resolution.pkl',f'Resolution_Data/run_{RunNum}/{timenow}'])

        # print('Merging Bootstrap Results')
        # print('-----------------------------')
        # subprocess.run(['python','merge_bootstrap_results.py','-d',f'Resolution_Data/{timenow}','-o',f'Resolution_{RunNum}_{timenow}'])
        # subprocess.run(['mv',f'Resolution_{RunNum}_{timenow}.csv','Final_Resolutions'])