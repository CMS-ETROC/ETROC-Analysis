from tqdm import tqdm
from natsort import natsorted
from os import stat
from glob import glob
import pandas as pd
import json

__all__ = [
    'process_qinj_nem_files',
    'toSingleDataFrame_newEventModel_moneyplot',
    'toSingleDataFrame_secondEventModel',
    'toSingleDataFrame_newEventModel',
    'toSingleDataFrame_eventModel_underlyingFunction',
    'toSingleDataFramePerDirectory_newEventModel'
]

## --------------- Text converting to DataFrame -----------------------
## --------------------------------------
def process_qinj_nem_files(idir):
        data_format = {
            'bcid': [],
            'l1a_counter': [],
            'board': [],
            'ea': [],
            'charge': [],
            'threshold': [],
            'row': [],
            'col': [],
            'toa': [],
            'tot': [],
            'cal': [],
        }

        info = idir.name.split('_')
        thres = int(info[-1])
        charge = int(info[-3])
        files = list(idir.glob('TDC*nem'))

        for ifile in files:
            with open(ifile, 'r') as infile:
                for line in infile:
                    parts = line.split()
                    if parts[0] == 'EH' or parts[0] == 'T' or parts[0] == 'ET':
                        continue
                    elif parts[0] == 'H':
                        bcid = int(parts[-1])
                        l1a_counter = int(parts[2])
                    elif parts[0] == 'D':
                        data_format['bcid'].append(bcid)
                        data_format['l1a_counter'].append(l1a_counter)
                        data_format['board'].append(int(parts[1]))
                        data_format['ea'].append(int(parts[2]))
                        data_format['charge'].append(charge)
                        data_format['threshold'].append(thres)
                        data_format['row'].append(int(parts[-5]))
                        data_format['col'].append(int(parts[-4]))
                        data_format['toa'].append(int(parts[-3]))
                        data_format['tot'].append(int(parts[-2]))
                        data_format['cal'].append(int(parts[-1]))

        single_df = pd.DataFrame(data_format).astype({
            'bcid': 'uint16',
            'l1a_counter': 'uint8',
            'board': 'uint8',
            'ea': 'uint8',
            'charge': 'uint8',
            'threshold': 'uint16',
            'row': 'uint8',
            'col': 'uint8',
            'toa': 'uint16',
            'tot': 'uint16',
            'cal': 'uint16'
        })

        return single_df

## --------------------------------------
def toSingleDataFrame_newEventModel_moneyplot(
        directories: list,
        minimum_stats: int = 100,
    ):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    results = []
    with tqdm(directories) as pbar:
        with ProcessPoolExecutor() as process_executor:
            # Each input results in multiple threading jobs being created:
            futures = [
                process_executor.submit(process_qinj_nem_files, idir)
                    for idir in directories
            ]
            for future in as_completed(futures):
                pbar.update(1)
                tmp_df = future.result()
                if tmp_df.shape[0] >= minimum_stats:
                    results.append(tmp_df)

    df = pd.concat(results, ignore_index=True)
    return df

## --------------------------------------
def toSingleDataFrame_secondEventModel(
        files: list,
        do_blockMix: bool = False,
        do_savedf: bool = False,
    ):
    return toSingleDataFrame_eventModel_underlyingFunction(
        files,
        do_blockMix,
        do_savedf,
        event_number_col = 1,
    )

def toSingleDataFrame_newEventModel(
        files: list,
        do_blockMix: bool = False,
        do_savedf: bool = False,
    ):
    return toSingleDataFrame_eventModel_underlyingFunction(
        files,
        do_blockMix,
        do_savedf,
        event_number_col = 2,
    )

def toSingleDataFrame_eventModel_underlyingFunction(
        files: list,
        do_blockMix: bool = False,
        do_savedf: bool = False,
        event_number_col: int = 0
    ):
    if event_number_col <= 0:
      raise RuntimeError("You need to define stuff correctly for the translation to work")
    evt = -1
    previous_evt = -1
    d = {
        'evt': [],
        'bcid': [],
        'l1a_counter': [],
        'board': [],
        'ea': [],
        'row': [],
        'col': [],
        'toa': [],
        'tot': [],
        'cal': [],
    }

    files = natsorted(files)
    df = pd.DataFrame(d)

    if do_blockMix:
        files = files[1:]

    in_event = False
    for ifile in files:
        file_d = json.loads(json.dumps(d))
        with open(ifile, 'r') as infile:
            for line in infile:
                if line.split(' ')[0] == 'EH':
                    tmp_evt = int(line.split(' ')[event_number_col])
                    if previous_evt != tmp_evt:
                        evt += 1
                        previous_evt = tmp_evt
                    in_event = True
                elif line.split(' ')[0] == 'H':
                    bcid = int(line.split(' ')[-1])
                    l1a_counter = int(line.split(' ')[2])
                elif line.split(' ')[0] == 'D':
                    id  = int(line.split(' ')[1])
                    ea  = int(line.split(' ')[2])
                    col = int(line.split(' ')[-4])
                    row = int(line.split(' ')[-5])
                    toa = int(line.split(' ')[-3])
                    tot = int(line.split(' ')[-2])
                    cal = int(line.split(' ')[-1])
                    if in_event:
                        file_d['evt'].append(evt)
                        file_d['bcid'].append(bcid)
                        file_d['l1a_counter'].append(l1a_counter)
                        file_d['board'].append(id)
                        file_d['ea'].append(ea)
                        file_d['row'].append(row)
                        file_d['col'].append(col)
                        file_d['toa'].append(toa)
                        file_d['tot'].append(tot)
                        file_d['cal'].append(cal)
                elif line.split(' ')[0] == 'T':
                    pass
                elif line.split(' ')[0] == 'ET':
                    in_event = False
                    pass
        if len(file_d['evt']) > 0:
            file_df = pd.DataFrame(file_d)
            df = pd.concat((df, file_df), ignore_index=True)
            del file_df
        del file_d

    df = df.astype('int')
    ## Under develop
    if do_savedf:
        pass

    return df

## --------------------------------------
def toSingleDataFramePerDirectory_newEventModel(
        path_to_dir: str,
        dir_name_pattern: str,
        data_qinj: bool = False,
        save_to_csv: bool = False,
        debugging: bool = False,
    ):

    evt = -1
    previous_evt = -1
    name_pattern = "*translated*.nem"

    dirs = glob(f"{path_to_dir}/{dir_name_pattern}")
    dirs = natsorted(dirs)
    print(dirs[:3])

    if debugging:
        dirs = dirs[:1]

    d = {
        'evt': [],
        'bcid': [],
        'l1a_counter': [],
        'board': [],
        'ea': [],
        'row': [],
        'col': [],
        'toa': [],
        'tot': [],
        'cal': [],
    }

    for dir in tqdm(dirs):
        df = pd.DataFrame(d)
        name = dir.split('/')[-1]
        files = glob(f"{dir}/{name_pattern}")

        for ifile in files:
            file_d = json.loads(json.dumps(d))

            if stat(ifile).st_size == 0:
                continue

            with open(ifile, 'r') as infile:
                for line in infile:
                    if line.split(' ')[0] == 'EH':
                        tmp_evt = int(line.split(' ')[2])
                        if previous_evt != tmp_evt:
                            evt += 1
                            previous_evt = tmp_evt
                    elif line.split(' ')[0] == 'H':
                        bcid = int(line.split(' ')[-1])
                        l1a_counter = int(line.split(' ')[2])
                    elif line.split(' ')[0] == 'D':
                        id  = int(line.split(' ')[1])
                        ea  = int(line.split(' ')[2])
                        col = int(line.split(' ')[-4])
                        row = int(line.split(' ')[-5])
                        toa = int(line.split(' ')[-3])
                        tot = int(line.split(' ')[-2])
                        cal = int(line.split(' ')[-1])
                        file_d['evt'].append(evt)
                        file_d['bcid'].append(bcid)
                        file_d['l1a_counter'].append(l1a_counter)
                        file_d['board'].append(id)
                        file_d['ea'].append(ea)
                        file_d['row'].append(row)
                        file_d['col'].append(col)
                        file_d['toa'].append(toa)
                        file_d['tot'].append(tot)
                        file_d['cal'].append(cal)
                    elif line.split(' ')[0] == 'T':
                        pass
                    elif line.split(' ')[0] == 'ET':
                        pass
            if len(file_d['evt']) > 0:
                file_df = pd.DataFrame(file_d)
                df = pd.concat((df, file_df), ignore_index=True)
                del file_df
            del file_d

        if not df.empty:
            df = df.astype('int')
            if data_qinj:
                df.drop(columns=['evt', 'board'], inplace=True)
            if save_to_csv:
                df.to_csv(name+'.csv', index=False)
            else:
                df.to_feather(name+'.feather')
            del df

## --------------- Text converting to DataFrame -----------------------