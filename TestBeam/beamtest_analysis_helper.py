from natsort import natsorted
from glob import glob
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import hist
import os
import pickle
import json

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import mplhep as hep
plt.style.use(hep.style.CMS)

from crc import Calculator, Crc8, Configuration

matplotlib.rcParams["axes.formatter.useoffset"] = False
matplotlib.rcParams["axes.formatter.use_mathtext"] = False

PeriCfg = {
    0: "PLL Config",
    1: "PLL Config",
    2: "PLL Config",
    3: "VRef & PLL Config",
    4: "TS & PS Config",
    5: "PS Config",
    6: "RefStrSel",
    7: "GRO & CLK40 Config",
    8: "GRO & CLK1280 Config",
    9: "GRO & FC Config",
    10: "BCIDOffset",
    11: "emptySlotBCID & BCIDOffset",
    12: "emptySlotBCID",
    13: "asy* & readoutClockDelayPixel",
    14: "asy* & readoutClockWidthPixel",
    15: "asy* & readoutClockDelayGlobal",
    16: "LTx_AmplSel & readoutClockWidthGlobal",
    17: "RTx_AmplSel & chargeInjectionDelay",
    18: "various",
    19: "various",
    20: "eFuse & trigger config",
    21: "eFuse Config",
    22: "eFuse Prog",
    23: "eFuse Prog",
    24: "eFuse Prog",
    25: "eFuse Prog",
    26: "linkResetFixedPattern",
    27: "linkResetFixedPattern",
    28: "linkResetFixedPattern",
    29: "linkResetFixedPattern",
    30: "ThrCounter Config",
    31: "TDCTest Config & ThrCounter Config",
}

PeriSta = {
    0: "PS_Late & AFCcalCap & AFCBusy",
    1: "fcAlignFinal and controller States",
    2: "FC Status",
    3: "invalidFCCount",
    4: "pllUnlockCount & invalidFCCount",
    5: "pllUnlockCount",
    6: "EFuseQ",
    7: "EFuseQ",
    8: "EFuseQ",
    9: "EFuseQ",
    10: "Unused",
    11: "Unused",
    12: "Unused",
    13: "Unused",
    14: "Unused",
    15: "Unused",
}

PixCfg = {
    0: "RF, IB & CL Selection",
    1: "QInj & QSel",
    2: "PD_DACDiscri & HysSel",
    3: "various",
    4: "DAC",
    5: "TH_Offset & DAC",
    6: "TDC Config",
    7: "various",
    8: "L1Adelay & selfTestOccupancy",
    9: "L1Adelay",
    10: "lowerCal",
    11: "upperCal & lowerCal",
    12: "lowerTOA & upperCal",
    13: "upperTOA & lowerTOA",
    14: "upperTOA",
    15: "lowerTOT",
    16: "upperTOT & lowerTOT",
    17: "lowerCalTrig & upperTOT",
    18: "upperCalTrig & lowerCalTrig",
    19: "lowerTOATrig & upperCalTrig",
    20: "lowerTOATrig",
    21: "upperTOATrig",
    22: "lowerTOTTrig & upperTOATrig",
    23: "upperTOTTrig & lowerTOTTrig",
    24: "upperTOTTrig",
    25: "unused",
    26: "unused",
    27: "unused",
    28: "unused",
    29: "unused",
    30: "unused",
    31: "unused",
}

PixSta = {
    0: "PixelID",
    1: "THState & NW & ScanDone",
    2: "BL",
    3: "TH & BL",
    4: "TH",
    5: "ACC",
    6: "ACC",
    7: "Copy of PixCfg31?",
    8: "unused",
    9: "unused",
    10: "unused",
    11: "unused",
    12: "unused",
    13: "unused",
    14: "unused",
    15: "unused",
    16: "unused",
    17: "unused",
    18: "unused",
    19: "unused",
    20: "unused",
    21: "unused",
    22: "unused",
    23: "unused",
    24: "unused",
    25: "unused",
    26: "unused",
    27: "unused",
    28: "unused",
    29: "unused",
    30: "unused",
    31: "unused",
}


## --------------- Compare Chip Configs -----------------------
## --------------------------------------
def compare_chip_configs(config_file1: Path, config_file2: Path, dump_i2c:bool = False):
    with open(config_file1, 'rb') as f:
        loaded_obj1 = pickle.load(f)
    with open(config_file2, 'rb') as f:
        loaded_obj2 = pickle.load(f)

    if loaded_obj1['chip'] != loaded_obj2['chip']:
        print("The config files are for different chips.")
        print(f"  - config 1: {loaded_obj1['chip']}")
        print(f"  - config 2: {loaded_obj2['chip']}")
        return

    chip1: dict = loaded_obj1['object']
    chip2: dict = loaded_obj2['object']

    common_keys = []
    for key in chip1.keys():
        if key not in chip2:
            print(f"Address Space \"{key}\" in config file 1 and not in config file 2")
        else:
            if key not in common_keys:
                common_keys += [key]
    for key in chip2.keys():
        if key not in chip1:
            print(f"Address Space \"{key}\" in config file 2 and not in config file 1")
        else:
            if key not in common_keys:
                common_keys += [key]

    for address_space_name in common_keys:
        if len(chip1[address_space_name]) != len(chip2[address_space_name]):
            print(f"The length of the \"{address_space_name}\" memory for config file 1 ({len(chip1[address_space_name])}) is not the same as for config file 2 ({len(chip2[address_space_name])})")

        length = min(len(chip1[address_space_name]), len(chip2[address_space_name]))

        for idx in range(length):
            if idx <= 0x001f:
                register = idx
                reg_info = f" (PeriCfg{register} contains {PeriCfg[register]})"
            elif idx == 0x0020:
                reg_info = " (Magic Number)"
            elif idx < 0x0100:
                reg_info = " (Unnamed Register Blk1)"
            elif idx <= 0x010f:
                register = idx - 0x0100
                reg_info = f" (PeriSta{register} contains {PeriSta[register]})"
            elif idx < 0x0120:
                reg_info = " (Unnamed Register BLk2)"
            elif idx <= 0x0123:
                reg_info = f" (SEUCounter{idx - 0x0120})"
            elif idx < 0x8000:
                reg_info = f" (Unnamed Register Blk3)"
            else:
                space = "Cfg"
                if (idx & 0x4000) != 0:
                    space = "Sta"
                broadcast = ""
                if (idx & 0x2000) != 0:
                    broadcast = " broadcast"
                register = idx & 0x1f
                row = (idx >> 5) & 0xf
                col = (idx >> 9) & 0xf
                contains = ""
                if space == "Cfg":
                    contains = PixCfg[register]
                else:
                    contains = PixSta[register]
                reg_info = f" (Pix{space}{register} - Pixel R{row} C{col}{broadcast} contains {contains})"
            if chip1[address_space_name][idx] != chip2[address_space_name][idx]:
                print(f"The register at address {idx} of {address_space_name}{reg_info} is different between config file 1 and config file 2: {chip1[address_space_name][idx]:#010b} vs {chip2[address_space_name][idx]:#010b}")

            if dump_i2c:
                print(f"The register at address {idx} of {address_space_name}{reg_info} is: {chip1[address_space_name][idx]:#010b}")


    print("Done comparing!")


## --------------- Decoding Class -----------------------
## --------------------------------------
class DecodeBinary:
    def copy_dict_by_json(self, d):
            return json.loads(json.dumps(d))

    def __init__(self, firmware_key,
                 board_id: list[int],
                 file_list: list[Path],
                 save_nem: Path | None = None,
                 skip_fw_filler: bool = False,
                 skip_event_df: bool = False,
                 skip_crc_df: bool = False,
                 verbose: bool = False,
                 ):
        self.firmware_key            = firmware_key
        self.header_pattern          = 0xc3a3c3a
        self.trailer_pattern         = 0b001011
        self.channel_header_pattern  = 0x3c5c0 >> 2
        self.firmware_filler_pattern = 0x5555
        self.firmware_filler_pattern_new = 0x556
        self.check_link_filler_pattern = 0x559
        self.previous_event          = -1
        self.event_counter           = 0
        self.board_ids               = board_id
        self.files_to_process        = file_list
        self.save_nem                = save_nem
        self.nem_file                = None
        self.skip_fw_filler          = skip_fw_filler
        self.skip_event_df           = skip_event_df
        self.skip_crc_df             = skip_crc_df
        self.verbose                 = verbose

        self.file_count = 0
        self.line_count = 0
        self.max_file_lines = 1e6

        self.in_event                = False
        self.eth_words_in_event      = -1
        self.words_in_event          = -1
        self.current_word            = -1
        self.event_number            = -1
        self.enabled_channels        = -1
        self.running_word            = None
        self.position_40bit          = 0
        self.current_channel         = -1
        self.in_40bit                = False
        self.data                    = {}
        self.version                 = None
        self.event_type              = None
        self.reset_params()

        self.data_template = {
            'evt': [],
            'bcid': [],
            'l1a_counter': [],
            'ea': [],
            'board': [],
            'row': [],
            'col': [],
            'toa': [],
            'tot': [],
            'cal': [],
        }

        self.crc_data_template = {
            'evt': [],
            'bcid': [],
            'l1a_counter': [],
            'board': [],
            'CRC': [],
            'CRC_calc': [],
            'CRC_mismatch': [],
        }

        self.event_data_template = {
            'evt': [],
            'bcid': [],
            'hits_counter': [],
            'l1a_counter': [],
            'fpga_evt_number': [],
            'hamming_count': [],
            'overflow_count': [],
            'CRC': [],
            'CRC_calc': [],
            'CRC_mismatch': [],
        }

        self.data_to_load = self.copy_dict_by_json(self.data_template)
        self.crc_data_to_load = self.copy_dict_by_json(self.crc_data_template)
        self.event_data_to_load = self.copy_dict_by_json(self.event_data_template)

        #self.CRCcalculator = Calculator(Crc8.AUTOSAR, optimized=True)

        config = Configuration(
            width=8,
            polynomial=0x2F, # Normal representation
            #polynomial=0x97, # Reversed reciprocal representation (the library uses normal representation, so do not use this)
            init_value=0x00,
            final_xor_value=0x00,
            reverse_input=False,
            reverse_output=False,
        )

        self.CRCcalculator = Calculator(config, optimized=True)

        self.event_in_filler_counter = 0  # To count events between fillers
        self.filler_idx = 0
        self.filler_prev_event = -1

        self.event_in_filler_40_counter = 0  # To count events between fillers
        self.filler_40_idx = 0
        self.filler_40_prev_event = -1

        self.filler_data_template = {
            'idx': [],
            'type': [],
            'events': [],
            'prev_event': [],
            'last_event': [],
            'filler_data': [],
        }

        self.filler_data = self.copy_dict_by_json(self.filler_data_template)

    def set_dtype(self):
        tmp = self.data_to_load

        self.data_to_load = {
            'evt': np.array(tmp['evt'], dtype=np.uint64),
            'bcid': np.array(tmp['bcid'], dtype=np.uint16),
            'l1a_counter': np.array(tmp['l1a_counter'], dtype=np.uint8),
            'ea': np.array(tmp['ea'], dtype=np.uint8),
            'board': np.array(tmp['board'], dtype=np.uint8),
            'row': np.array(tmp['row'], dtype=np.uint8),
            'col': np.array(tmp['col'], dtype=np.uint8),
            'toa': np.array(tmp['toa'], dtype=np.uint16),
            'tot': np.array(tmp['tot'], dtype=np.uint16),
            'cal': np.array(tmp['cal'], dtype=np.uint16),
        }

    def set_crc_dtype(self):
        tmp = self.crc_data_to_load

        self.crc_data_to_load = {
            'evt': np.array(tmp['evt'], dtype=np.uint64),
            'bcid': np.array(tmp['bcid'], dtype=np.uint16),
            'l1a_counter': np.array(tmp['l1a_counter'], dtype=np.uint8),
            'board': np.array(tmp['board'], dtype=np.uint8),
            'CRC': np.array(tmp['CRC'], dtype=np.uint8),
            'CRC_calc': np.array(tmp['CRC_calc'], dtype=np.uint8),
            'CRC_mismatch': np.array(tmp['CRC_mismatch'], dtype=np.bool_),
        }

    def set_event_dtype(self):
        tmp = self.event_data_to_load

        self.event_data_to_load = {
            'evt': np.array(tmp['evt'], dtype=np.uint64),
            'bcid': np.array(tmp['bcid'], dtype=np.uint16),
            'hits_counter': np.array(tmp['hits_counter'], dtype=np.uint32),
            'l1a_counter': np.array(tmp['l1a_counter'], dtype=np.uint8),
            'fpga_evt_number': np.array(tmp['fpga_evt_number'], dtype=np.uint64),
            'hamming_count': np.array(tmp['hamming_count'], dtype=np.uint8),
            'overflow_count': np.array(tmp['overflow_count'], dtype=np.uint8),
            'CRC': np.array(tmp['CRC'], dtype=np.uint8),
            'CRC_calc': np.array(tmp['CRC_calc'], dtype=np.uint8),
            'CRC_mismatch': np.array(tmp['CRC_mismatch'], dtype=np.bool_),
        }

    def set_filler_dtype(self):
        tmp = self.filler_data

        self.filler_data = {
            'idx': np.array(tmp['idx'], dtype=np.uint64),
            'type': np.array(tmp['type'], dtype=np.string_),
            'events': np.array(tmp['events'], dtype=np.uint32),
            'prev_event': np.array(tmp['prev_event'], dtype=np.int64),
            'last_event': np.array(tmp['last_event'], dtype=np.int64),
            'filler_data': np.array(tmp['filler_data'], dtype=np.string_),
        }

    def reset_params(self):
        self.in_event                = False
        self.eth_words_in_event      = -1
        self.words_in_event          = -1
        self.current_word            = -1
        self.event_number            = -1
        self.enabled_channels        = -1
        self.running_word            = None
        self.position_40bit          = 0
        self.current_channel         = -1
        self.in_40bit                = False
        self.data                    = {}
        self.crc_data                = {}
        self.event_data              = {}
        self.version                 = None
        self.event_type              = None
        self.CRCdata_40bit           = []
        self.CRCdata                 = []  # Datao mentions the initial value for the event CRC is the CRC output value of the previous event... so it is hard to implement a CRC check for events if this is true

    def open_next_file(self):
        if self.save_nem is not None:
            if self.nem_file is not None:
                self.nem_file.close()

            base_dir = self.save_nem.parent
            file_name = self.save_nem.stem
            suffix = self.save_nem.suffix

            file_name = f'{file_name}_{self.file_count}{suffix}'

            self.nem_file = open(base_dir / file_name, "w")
            self.file_count += 1
            self.line_count = 0
        else:
            self.nem_file = None
            self.line_count = 0
            self.file_count = 0

    def close_file(self):
        if self.save_nem is not None:
            if self.nem_file is not None:
                self.nem_file.close()
        self.nem_file = None
        self.line_count = 0
        self.file_count = 0

    def write_to_nem(self, write_str: str):
        if self.nem_file is not None:
            self.nem_file.write(write_str)
            self.line_count += 1

            if self.line_count >= self.max_file_lines:
                self.open_next_file()

    def div_ceil(self, x,y):
        return -(x//(-y))

    def decode_40bit(self, word):
        # Header
        if word >> 22 == self.channel_header_pattern and not self.in_40bit:
            self.current_channel += 1
            while not ((self.enabled_channels >> self.current_channel) & 0b1):
                self.current_channel += 1
                if self.current_channel > 3:
                    print('Found more headers than number of channels')
                    if self.nem_file is not None:
                        self.write_to_nem(f"THIS IS A BROKEN EVENT SINCE MORE HEADERS THAN MASK FOUND\n")
                    self.reset_params()
                    return
            self.bcid = (word & 0xfff)
            self.l1acounter = ((word >> 14) & 0xff)
            self.data[self.current_channel] = self.copy_dict_by_json(self.data_template)
            self.crc_data[self.current_channel] = self.copy_dict_by_json(self.crc_data_template)
            self.in_40bit = True
            Type = (word >> 12) & 0x3

            self.CRCdata_40bit = [
                (word >> 32) & 0xff,
                (word >> 24) & 0xff,
                (word >> 16) & 0xff,
                (word >> 8) & 0xff,
                (word ) & 0xff,
                ]

            if self.nem_file is not None:
                self.write_to_nem(f"H {self.current_channel} {self.l1acounter} 0b{Type:02b} {self.bcid}\n")
        # Data
        elif (word >> 39) == 1 and self.in_40bit:
            EA = (word >> 37) & 0b11
            ROW = (word >> 29) & 0b1111
            COL = (word >> 33) & 0b1111
            TOA = (word >> 19) & 0x3ff
            TOT = (word >> 10) & 0x1ff
            CAL = (word) & 0x3ff
            #self.data[self.current_channel]['evt_number'].append(self.event_number)
            self.data[self.current_channel]['bcid'].append(self.bcid)
            self.data[self.current_channel]['l1a_counter'].append(self.l1acounter)
            self.data[self.current_channel]['evt'].append(self.event_counter)
            self.data[self.current_channel]['ea'].append(EA)
            self.data[self.current_channel]['board'].append(self.current_channel)
            self.data[self.current_channel]['row'].append(ROW)
            self.data[self.current_channel]['col'].append(COL)
            self.data[self.current_channel]['toa'].append(TOA)
            self.data[self.current_channel]['tot'].append(TOT)
            self.data[self.current_channel]['cal'].append(CAL)

            self.CRCdata_40bit += [
                (word >> 32) & 0xff,
                (word >> 24) & 0xff,
                (word >> 16) & 0xff,
                (word >> 8) & 0xff,
                (word ) & 0xff,
                ]

            if self.nem_file is not None:
                self.write_to_nem(f"D {self.current_channel} 0b{EA:02b} {ROW} {COL} {TOA} {TOT} {CAL}\n")

        # Trailer
        elif (word >> 22) & 0x3ffff == self.board_ids[self.current_channel] and self.in_40bit:
            hits   = (word >> 8) & 0xff
            status = (word >> 16) & 0x3f
            CRC    = (word) & 0xff
            self.in_40bit = False

            if len(self.data[self.current_channel]['evt']) != hits:
                print('Number of hits does not match!')
                self.reset_params()
                return

            self.CRCdata_40bit += [
                (word >> 32) & 0xff,
                (word >> 24) & 0xff,
                (word >> 16) & 0xff,
                (word >> 8) & 0xff,
                #(word ) & 0xff,
                ]

            mismatch = ""
            if not self.skip_crc_df:
                data = bytes(self.CRCdata_40bit)
                check = self.CRCcalculator.checksum(data)

                #print("Raw data:")
                #print_string = ""
                #for dat in self.CRCdata_40bit:
                #    print_string += f"{dat:08b} "
                #print(print_string)
                #print(f"CRC: {CRC:08b}")
                #print(f"CRC Check: {check:08b}")

                if CRC != check:
                    mismatch = " CRC Mismatch"

                self.crc_data[self.current_channel]['bcid'].append(self.bcid)
                self.crc_data[self.current_channel]['l1a_counter'].append(self.l1acounter)
                self.crc_data[self.current_channel]['evt'].append(self.event_counter)
                self.crc_data[self.current_channel]['board'].append(self.current_channel)
                self.crc_data[self.current_channel]['CRC'].append(CRC)
                self.crc_data[self.current_channel]['CRC_calc'].append(check)
                self.crc_data[self.current_channel]['CRC_mismatch'].append(bool(CRC != check))

            if self.nem_file is not None:
                self.write_to_nem(f"T {self.current_channel} {status} {hits} 0b{CRC:08b}{mismatch}\n")


        # Something else
        else:
            binary = format(word, '040b')
            print(f'Warning! Found 40 bits word which is not matched with the pattern {binary}')
            self.reset_params()
            return

    def decode_files(self):
        if self.save_nem is not None:
            self.open_next_file()

        self.data_to_load = self.copy_dict_by_json(self.data_template)
        self.set_dtype()
        df = pd.DataFrame(self.data_to_load)
        self.data_to_load = self.copy_dict_by_json(self.data_template)

        self.crc_data = self.copy_dict_by_json(self.crc_data_template)
        self.set_crc_dtype()
        crc_df = pd.DataFrame(self.crc_data_to_load)
        self.crc_data_to_load = self.copy_dict_by_json(self.crc_data_template)

        self.event_data_to_load = self.copy_dict_by_json(self.event_data_template)
        self.set_event_dtype()
        event_df = pd.DataFrame(self.event_data_to_load)
        self.event_data_to_load = self.copy_dict_by_json(self.event_data_template)

        self.filler_data = self.copy_dict_by_json(self.filler_data_template)
        self.set_filler_dtype()
        filler_df = pd.DataFrame(self.filler_data)
        self.filler_data = self.copy_dict_by_json(self.filler_data_template)

        decoding = False
        for ifile in self.files_to_process:
            with open(file=ifile, mode='rb') as infile:
                while True:
                    in_data = infile.read(4)
                    # print(in_data)
                    if in_data == b'':
                        break
                    word = int.from_bytes(in_data, byteorder='little')
                    if not decoding and word == 0:
                        continue
                    if not decoding:
                        decoding = True

                    ## Event header
                    if (word >> 4) == self.header_pattern:
                        self.reset_params()
                        self.enabled_channels = word & 0b1111
                        self.in_event = True
                        # print('Event header')
                        self.CRCdata = [
                            (word >> 24) & 0xff,
                            (word >> 16) & 0xff,
                            (word >> 8) & 0xff,
                            (word ) & 0xff,
                        ]
                        continue

                    # Event Header Line Two Found
                    elif(self.in_event and (self.words_in_event == -1) and (word >> 28 == self.firmware_key)):
                        self.current_word       = 0
                        self.event_type         = (word) & 0x3
                        self.event_number       = (word >> 12) & 0xffff
                        self.words_in_event     = (word >> 2) & 0x3ff
                        self.version            = (word >> 28) & 0xf
                        self.eth_words_in_event = self.div_ceil(40*self.words_in_event, 32)
                        # print(f"Num Words {self.words_in_event} & Eth Words {self.eth_words_in_event}")
                        # Set valid_data to true once we see fresh data
                        if(self.event_number==1 or self.event_number==0):
                            self.valid_data = True
                        self.event_data = self.copy_dict_by_json(self.event_data_template)
                        self.CRCdata += [
                            (word >> 24) & 0xff,
                            (word >> 16) & 0xff,
                            (word >> 8) & 0xff,
                            (word ) & 0xff,
                        ]
                        # print('Event Header Line Two Found')
                        # print(self.event_number)
                        if self.nem_file is not None:
                            self.write_to_nem(f"EH 0b{self.version:04b} {self.event_number} {self.words_in_event} {self.event_type:02b}\n")
                        continue

                    # Event Header Line Two NOT Found after the Header
                    elif(self.in_event and (self.words_in_event == -1) and (word >> 28 != self.firmware_key)):
                        # print('Event Header Line Two NOT Found after the Header')
                        self.reset_params()
                        if self.verbose:
                            print('Event Header Line Two NOT Found after the Header')
                        continue

                    # Event Trailer NOT Found after the required number of ethernet words was read
                    elif(self.in_event and (self.eth_words_in_event==self.current_word) and (word >> 26 != self.trailer_pattern)):
                        # print('Event Trailer NOT Found after the required number of ethernet words was read')
                        self.reset_params()
                        if self.verbose:
                            print('Event Trailer NOT Found')
                        continue

                    # Event Trailer Found - DO NOT CONTINUE
                    elif(self.in_event and (self.eth_words_in_event==self.current_word) and (word >> 26 == self.trailer_pattern)):
                        for key in self.data_to_load:
                            for board in self.data:
                                self.data_to_load[key] += self.data[board][key]
                        for key in self.crc_data_to_load:
                            for board in self.crc_data:
                                self.crc_data_to_load[key] += self.crc_data[board][key]
                        # print(self.event_number)
                        # print(self.data)

                        self.CRCdata += [
                            (word >> 24) & 0xff,
                            (word >> 16) & 0xff,
                            (word >> 8) & 0xff,
                        ]

                        crc = (word) & 0xff

                        mismatch = ""
                        if not self.skip_event_df:
                            data = bytes(self.CRCdata)
                            check = self.CRCcalculator.checksum(data)

                            hits_count = (word >> 14) & 0xfff
                            overflow_count = (word >> 11) & 0x7
                            hamming_count  = (word >> 8) & 0x7

                            if crc != check:
                                mismatch = " CRC Mismatch"

                            self.event_data['evt'].append(self.event_counter)
                            self.event_data['bcid'].append(self.bcid)
                            self.event_data['hits_counter'].append(hits_count)
                            self.event_data['l1a_counter'].append(self.l1acounter)
                            self.event_data['fpga_evt_number'].append(self.event_number)
                            self.event_data['hamming_count'].append(hamming_count)
                            self.event_data['overflow_count'].append(overflow_count)
                            self.event_data['CRC'].append(crc)
                            self.event_data['CRC_calc'].append(check)
                            self.event_data['CRC_mismatch'].append(bool(crc != check))

                        for key in self.event_data_to_load:
                            self.event_data_to_load[key] += self.event_data[key]
                        self.event_counter += 1
                        self.event_in_filler_counter += 1
                        self.event_in_filler_40_counter += 1

                        if len(self.data_to_load['evt']) >= 10000:
                            self.set_dtype()
                            df = pd.concat([df, pd.DataFrame(self.data_to_load)], ignore_index=True)
                            self.data_to_load = self.copy_dict_by_json(self.data_template)

                            if not self.skip_crc_df:
                                self.set_crc_dtype()
                                crc_df = pd.concat([crc_df, pd.DataFrame(self.crc_data_to_load)], ignore_index=True)
                                self.crc_data_to_load = self.copy_dict_by_json(self.crc_data_template)

                            if not self.skip_event_df:
                                self.set_event_dtype()
                                event_df = pd.concat([event_df, pd.DataFrame(self.event_data_to_load)], ignore_index=True)
                                self.event_data_to_load = self.copy_dict_by_json(self.event_data_template)

                        if self.nem_file is not None:
                            self.write_to_nem(f"ET {self.event_number} {overflow_count} {hamming_count} 0b{crc:08b}{mismatch}\n")

                    # Event Data Word
                    elif(self.in_event):
                        # print(self.current_word)
                        # print(format(word, '032b'))

                        self.CRCdata += [
                            (word >> 24) & 0xff,
                            (word >> 16) & 0xff,
                            (word >> 8) & 0xff,
                            (word ) & 0xff,
                        ]

                        if self.position_40bit == 4:
                            self.word_40 = self.word_40 | word
                            self.decode_40bit(self.word_40)
                            self.position_40bit = 0
                            self.current_word += 1
                            continue
                        if self.position_40bit >= 1:
                            self.word_40 = self.word_40 | (word >> (8*(4-self.position_40bit)))
                            self.decode_40bit(self.word_40)
                        self.word_40 = (word << ((self.position_40bit + 1)*8)) & 0xffffffffff
                        self.position_40bit += 1
                        self.current_word += 1
                        continue

                    # If Firmware filler
                    elif (word >> 16) == self.firmware_filler_pattern:
                        if self.nem_file is not None and not self.skip_fw_filler:
                            self.write_to_nem(f"Filler: 0b{word & 0xffff:016b}\n")

                    # New firmware filler
                    elif (word >> 20) == self.firmware_filler_pattern_new:
                        if not self.skip_fw_filler:
                            self.filler_data['idx'].append(self.filler_idx)
                            self.filler_data['type'].append("FW")
                            self.filler_data['events'].append(self.event_in_filler_counter)
                            self.filler_data['prev_event'].append(self.filler_prev_event)
                            self.filler_data['last_event'].append(self.event_counter)
                            self.filler_data['filler_data'].append(f"0b{word & 0xfffff:020b}")
                            self.filler_idx += 1
                            self.event_in_filler_counter = 0
                            self.filler_prev_event = self.event_counter
                        if self.nem_file is not None and not self.skip_fw_filler:
                            self.write_to_nem(f"FW Filler: 0b{word & 0xfffff:020b}\n")

                    # Check link filler
                    elif (word >> 20) == self.check_link_filler_pattern:
                        self.filler_data['idx'].append(self.filler_40_idx)
                        self.filler_data['type'].append("40")
                        self.filler_data['events'].append(self.event_in_filler_40_counter)
                        self.filler_data['prev_event'].append(self.filler_40_prev_event)
                        self.filler_data['last_event'].append(self.event_counter)
                        self.filler_data['filler_data'].append(f"0b{word & 0xfffff:020b}")
                        self.filler_40_idx += 1
                        self.event_in_filler_40_counter = 0
                        self.filler_40_prev_event = self.event_counter
                        if self.nem_file is not None:
                            self.write_to_nem(f"40Hz Filler: 0b{word & 0xfffff:020b}\n")

                    if len(self.filler_data['idx']) > 10000:
                        self.set_filler_dtype()
                        filler_df = pd.concat([filler_df, pd.DataFrame(self.filler_data)], ignore_index=True)
                        self.filler_data= self.copy_dict_by_json(self.filler_data_template)

                    # Reset anyway!
                    self.reset_params()

                if len(self.data_to_load['evt']) > 0:
                    self.set_dtype()
                    df = pd.concat([df, pd.DataFrame(self.data_to_load)], ignore_index=True)
                    self.data_to_load = self.copy_dict_by_json(self.data_template)

                if len(self.crc_data_to_load['evt']) > 0:
                    if not self.skip_crc_df:
                        self.set_crc_dtype()
                        crc_df = pd.concat([crc_df, pd.DataFrame(self.crc_data_to_load)], ignore_index=True)
                        self.crc_data_to_load = self.copy_dict_by_json(self.crc_data_template)

                if len(self.event_data_to_load['evt']) > 0:
                    if not self.skip_event_df:
                        self.set_event_dtype()
                        event_df = pd.concat([event_df, pd.DataFrame(self.event_data_to_load)], ignore_index=True)
                        self.event_data_to_load = self.copy_dict_by_json(self.event_data_template)

                if len(self.filler_data['idx']) > 0:
                    self.set_filler_dtype()
                    filler_df = pd.concat([filler_df, pd.DataFrame(self.filler_data)], ignore_index=True)
                    self.filler_data= self.copy_dict_by_json(self.filler_data_template)

        self.close_file()
        return df, event_df, crc_df, filler_df

## --------------- Decoding Class -----------------------



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
def toSingleDataFrame_newEventModel(
        files: list,
        do_blockMix: bool = False,
        do_savedf: bool = False,
    ):
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

    for ifile in files:
        file_d = json.loads(json.dumps(d))
        with open(ifile, 'r') as infile:
            for line in infile:
                if line.split(' ')[0] == 'EH':
                    tmp_evt = int(line.split(' ')[2])
                    if previous_evt != tmp_evt:
                        evt += 1
                        previous_evt = tmp_evt
                        pass
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

            if os.stat(ifile).st_size == 0:
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


## --------------- Modify DataFrame -----------------------
## --------------------------------------
def efficiency_with_single_board(
        input_df: pd.DataFrame,
        pixel: set = (8, 8), # (row, col)
        board_id: int = 0,
    ):
    df_tmp = input_df.set_index('evt')
    selection = (df_tmp['board'] == board_id) & (df_tmp['row'] == pixel[0]) & (df_tmp['col'] == pixel[1])
    new_df = input_df.loc[input_df['evt'].isin(df_tmp.loc[selection].index)]

    del df_tmp, selection
    return new_df

## --------------------------------------
def efficiency_with_two_boards(
        input_df: pd.DataFrame,
        pixel: set = (8, 8), # (row, col)
        board_ids: set = (0, 3), #(board 1, board 2)
    ):

    df_tmp = input_df.set_index('evt')
    selection1 = (df_tmp['board'] == board_ids[0]) & (df_tmp['row'] == pixel[0]) & (df_tmp['col'] == pixel[1])
    selection2 = (df_tmp['board'] == board_ids[1]) & (df_tmp['row'] == pixel[0]) & (df_tmp['col'] == pixel[1])

    filtered_index = list(set(df_tmp.loc[selection1].index).intersection(df_tmp.loc[selection2].index))
    new_df = input_df.loc[input_df['evt'].isin(filtered_index)]

    del df_tmp, filtered_index, selection1, selection2
    return new_df

## --------------------------------------
def singlehit_event_clear(
        input_df: pd.DataFrame,
        ignore_boards: list[int] = None
    ):

    ana_df = input_df
    if ignore_boards is not None:
        for board in ignore_boards:
            ana_df = ana_df.loc[ana_df['board'] != board].copy()

    ## event has one hit from each board
    event_board_counts = ana_df.groupby(['evt', 'board']).size().unstack(fill_value=0)
    event_selection_col = None
    for board in event_board_counts:
        if event_selection_col is None:
            event_selection_col = (event_board_counts[board] == 1)
        else:
            event_selection_col = event_selection_col & (event_board_counts[board] == 1)
    selected_event_numbers = event_board_counts[event_selection_col].index
    selected_subset_df = ana_df[ana_df['evt'].isin(selected_event_numbers)]
    selected_subset_df.reset_index(inplace=True, drop=True)

    del ana_df, event_board_counts, event_selection_col

    return selected_subset_df

## --------------------------------------
def tdc_event_selection(
        input_df: pd.DataFrame,
        tdc_cuts_dict: dict,
        select_by_hit: bool = False,
    ):

    if select_by_hit:

        # Create boolean masks for each board's filtering criteria
        masks = {}
        for board, cuts in tdc_cuts_dict.items():
            mask = (
                (input_df['board'] == board) &
                input_df['cal'].between(cuts[0], cuts[1]) &
                input_df['toa'].between(cuts[2], cuts[3]) &
                input_df['tot'].between(cuts[4], cuts[5])
            )
            masks[board] = mask

        # Combine the masks using logical OR
        combined_mask = pd.concat(masks, axis=1).any(axis=1)

        # Apply the combined mask to the DataFrame
        tdc_filtered_df = input_df[combined_mask].reset_index(drop=True)

        return tdc_filtered_df

    else:
        from functools import reduce

        # Create boolean masks for each board's filtering criteria
        masks = {}
        for board, cuts in tdc_cuts_dict.items():
            mask = (
                (input_df['board'] == board) &
                input_df['cal'].between(cuts[0], cuts[1]) &
                input_df['toa'].between(cuts[2], cuts[3]) &
                input_df['tot'].between(cuts[4], cuts[5])
            )
            masks[board] = input_df[mask]['evt'].unique()

        common_elements = reduce(np.intersect1d, list(masks.values()))
        tdc_filtered_df = input_df.loc[input_df['evt'].isin(common_elements)].reset_index(drop=True)

        return tdc_filtered_df

## --------------------------------------
def tdc_event_selection_pivot(
        input_df: pd.DataFrame,
        tdc_cuts_dict: dict
    ):
    # Create boolean masks for each board's filtering criteria
    masks = {}
    for board, cuts in tdc_cuts_dict.items():
        mask = (
            input_df['cal'][board].between(cuts[0], cuts[1]) &
            input_df['toa'][board].between(cuts[2], cuts[3]) &
            input_df['tot'][board].between(cuts[4], cuts[5])
        )
        masks[board] = mask

    # Combine the masks using logical AND
    combined_mask = pd.concat(masks, axis=1).all(axis=1)
    del masks

    # Apply the combined mask to the DataFrame
    tdc_filtered_df = input_df[combined_mask].reset_index(drop=True)
    del combined_mask
    return tdc_filtered_df

## --------------------------------------
def pixel_filter(
        input_df: pd.DataFrame,
        pixel_dict: dict,
        filter_by_area: bool = False,
        pixel_buffer: int = 2,
    ):

    masks = {}
    if filter_by_area:
        for board, pix in pixel_dict.items():
            mask = (
                (input_df['board'] == board)
                & (input_df['row'] >= pix[0]-pixel_buffer) & (input_df['row'] <= pix[0]+pixel_buffer)
                & (input_df['col'] >= pix[1]-pixel_buffer) & (input_df['col'] <= pix[1]+pixel_buffer)
            )
            masks[board] = mask
    else:
        for board, pix in pixel_dict.items():
            mask = (
                (input_df['board'] == board) & (input_df['row'] == pix[0]) & (input_df['col'] == pix[1])
            )
            masks[board] = mask

    # Combine the masks using logical OR
    combined_mask = pd.concat(masks, axis=1).any(axis=1)

    # Apply the combined mask to the DataFrame
    filtered = input_df[combined_mask].reset_index(drop=True)
    return filtered

## --------------------------------------
def making_pivot(
        input_df: pd.DataFrame,
        index: str,
        columns: str,
        drop_columns: tuple,
        ignore_boards: list[int] = None
    ):
        ana_df = input_df
        if ignore_boards is not None:
            for board in ignore_boards:
                ana_df = ana_df.loc[ana_df['board'] != board].copy()
        pivot_data_df = ana_df.pivot(
            index = index,
            columns = columns,
            values = list(set(ana_df.columns) - drop_columns),
        )
        pivot_data_df.columns = ["{}_{}".format(x, y) for x, y in pivot_data_df.columns]

        return pivot_data_df

## --------------------------------------
def return_broadcast_dataframe(
        input_df: pd.DataFrame,
        reference_board_id: int,
        board_id_want_broadcast: int,
    ):

    tmp_df = input_df.loc[(input_df['board'] == reference_board_id) | (input_df['board'] == board_id_want_broadcast)]
    tmp_df = tmp_df.drop(columns=['ea', 'toa', 'tot', 'cal'])

    event_board_counts = tmp_df.groupby(['evt', 'board']).size().unstack(fill_value=0)
    event_selections = (event_board_counts[board_id_want_broadcast] == 1) & (event_board_counts[reference_board_id] == 1)
    single_hit_df = tmp_df.loc[tmp_df['evt'].isin(event_board_counts[event_selections].index)]
    single_hit_df.reset_index(inplace=True, drop=True)

    if 'identifier' in single_hit_df.columns:
        single_hit_df = single_hit_df.drop(columns=['identifier'])

    sub_single_df1 = single_hit_df.loc[single_hit_df['board'] == board_id_want_broadcast]
    sub_single_df2 = single_hit_df.loc[single_hit_df['board'] == reference_board_id]

    single_df = pd.merge(sub_single_df1, sub_single_df2, on='evt', suffixes=[f'_{board_id_want_broadcast}', f'_{reference_board_id}'])
    single_df = single_df.drop(columns=['evt'])
    del single_hit_df, sub_single_df1, sub_single_df2

    event_selections = (event_board_counts[board_id_want_broadcast] == 1) & (event_board_counts[reference_board_id] >= 2)
    multi_hit_df = tmp_df.loc[tmp_df['evt'].isin(event_board_counts[event_selections].index)]
    multi_hit_df.reset_index(inplace=True, drop=True)

    sub_multiple_df1 = multi_hit_df.loc[multi_hit_df['board'] == board_id_want_broadcast]
    sub_multiple_df2 = multi_hit_df.loc[multi_hit_df['board'] == reference_board_id]

    multi_df = pd.merge(sub_multiple_df1, sub_multiple_df2, on='evt', suffixes=[f'_{board_id_want_broadcast}', f'_{reference_board_id}'])
    multi_df = multi_df.drop(columns=['evt'])
    del multi_hit_df, tmp_df, sub_multiple_df1, sub_multiple_df2

    return single_df, multi_df


## --------------- Modify DataFrame -----------------------


## --------------- Extract results -----------------------
## --------------------------------------
def save_TDC_summary_table(
        input_df: pd.DataFrame,
        chipLabels: list[int],
        var: str,
        save_path: Path,
        fname_tag: str,
    ):

    for id in chipLabels:

        if input_df[input_df['board'] == id].empty:
            continue

        sum_group = input_df[input_df['board'] == id].groupby(["col", "row"]).agg({var:['mean','std']})
        sum_group.columns = sum_group.columns.droplevel()
        sum_group.reset_index(inplace=True)

        table_mean = sum_group.pivot_table(index='row', columns='col', values='mean')
        table_mean = table_mean.round(1)

        table_mean = table_mean.reindex(pd.Index(np.arange(0,16), name='')).reset_index()
        table_mean = table_mean.reindex(columns=np.arange(0,16))

        table_std = sum_group.pivot_table(index='row', columns='col', values='std')
        table_std = table_std.round(2)

        table_std = table_std.reindex(pd.Index(np.arange(0,16), name='')).reset_index()
        table_std = table_std.reindex(columns=np.arange(0,16))

        table_mean.to_pickle(save_path / f"{fname_tag}_mean.pkl")
        table_std.to_pickle(save_path / f"{fname_tag}_std.pkl")

        del sum_group, table_mean, table_std

## --------------------------------------
def return_TOA_correlation_param(
        input_df: pd.DataFrame,
        board_id1: int,
        board_id2: int,
    ):

    x = input_df['toa'][board_id1]
    y = input_df['toa'][board_id2]

    params = np.polyfit(x, y, 1)
    distance = (x*params[0] - y + params[1])/(np.sqrt(params[0]**2 + 1))

    return params, distance

## --------------------------------------
def return_TWC_param(
        corr_toas: dict,
        input_df: pd.DataFrame,
        board_ids: list[int],
    ):

    results = {}

    del_toa_b0 = (0.5*(corr_toas[f'toa_b{board_ids[1]}'] + corr_toas[f'toa_b{board_ids[2]}']) - corr_toas[f'toa_b{board_ids[0]}'])
    del_toa_b1 = (0.5*(corr_toas[f'toa_b{board_ids[0]}'] + corr_toas[f'toa_b{board_ids[2]}']) - corr_toas[f'toa_b{board_ids[1]}'])
    del_toa_b2 = (0.5*(corr_toas[f'toa_b{board_ids[0]}'] + corr_toas[f'toa_b{board_ids[1]}']) - corr_toas[f'toa_b{board_ids[2]}'])

    coeff_b0 = np.polyfit(input_df[f'tot_b{board_ids[0]}'].values, del_toa_b0, 1)
    results[0] = (input_df[f'tot_b{board_ids[0]}'].values*coeff_b0[0] - del_toa_b0 + coeff_b0[1])/(np.sqrt(coeff_b0[0]**2 + 1))

    coeff_b1 = np.polyfit(input_df[f'tot_b{board_ids[1]}'].values, del_toa_b1, 1)
    results[1] = (input_df[f'tot_b{board_ids[1]}'].values*coeff_b1[0] - del_toa_b1 + coeff_b1[1])/(np.sqrt(coeff_b1[0]**2 + 1))

    coeff_b2 = np.polyfit(input_df[f'tot_b{board_ids[2]}'].values, del_toa_b2, 1)
    results[2] = (input_df[f'tot_b{board_ids[2]}'].values*coeff_b2[0] - del_toa_b2 + coeff_b2[1])/(np.sqrt(coeff_b2[0]**2 + 1))

    return results

## --------------- Extract results -----------------------


## --------------- Plotting -----------------------
def load_fig_title(
    tb_loc:str
):
    """Load figure title (beam info and testing location)

    Parameters
    ----------
    tb_loc: str,
        Location of the test beam.
        Input argument for test beam facility
            1. 'desy',
            2. 'cern',
            3. 'fnal',
        Input argument for SEU test facility
            1. 'northwestern',
            2. 'louvain-{ion type}'
                - ion type: C, Ne, Al, Ar, Cr, Ni, Kr, Rh, Xe
    """
    if tb_loc == 'desy':
        plot_title = r'4 GeV $e^{-}$ at DESY TB'
    elif tb_loc == 'cern':
        plot_title = r'120 GeV (1/3 p; 2/3 $\pi^{+}$) at CERN SPS'
    elif tb_loc == 'fnal':
        plot_title = r'120 GeV p at Fermilab TB'
    elif tb_loc == 'northwestern':
        plot_title = r'217 MeV p at Northwestern Medicine Proton Center'
    # The assumption for louvain is the the tb_loc will specify location and ion with the following format:
    # louvain-Kr  - for example for louvain with Krypton ion beam
    # louvain-Xe  - for example for louvain with Xenon ion beam
    elif tb_loc[:7] == 'louvain':
        ion = tb_loc[8:]
        energy = "xx"
        if ion == "C":
            energy = 131
        elif ion == "Ne":
            energy = 238
        elif ion == "Al":
            energy = 250
        elif ion == "Ar":
            energy = 353
        elif ion == "Cr":
            energy = 505
        elif ion == "Ni":
            energy = 582
        elif ion == "Kr":
            energy = 769
        elif ion == "Rh":
            energy = 957
        elif ion == "Xe":
            energy = 995
        plot_title = rf'{energy} MeV {ion} at Heavy Ion Facility'
    else:
        print('Unknown location. Please add info into the function. Return empty string')
        plot_title = ""

    return plot_title

## --------------------------------------
def return_hist(
        input_df: pd.DataFrame,
        board_ids: list[int],
        board_names: list[str],
        hist_bins: list = [50, 64, 64]
):
    h = {board_names[board_idx]: hist.Hist(hist.axis.Regular(hist_bins[0], 140, 240, name="CAL", label="CAL [LSB]"),
                hist.axis.Regular(hist_bins[1], 0, 512,  name="TOT", label="TOT [LSB]"),
                hist.axis.Regular(hist_bins[2], 0, 1024, name="TOA", label="TOA [LSB]"),
                hist.axis.Regular(4, 0, 3, name="EA", label="EA"),
        )
    for board_idx in range(len(board_ids))}

    for board_idx in range(len(board_ids)):
        tmp_df = input_df.loc[input_df['board'] == board_ids[board_idx]]
        h[board_names[board_idx]].fill(tmp_df['cal'].values, tmp_df['tot'].values, tmp_df['toa'].values, tmp_df['ea'].values)

    return h

## --------------------------------------
def return_event_hist(
        input_df: pd.DataFrame,
):
    h = hist.Hist(hist.axis.Regular(8, 0, 8, name="HA", label="Hamming Count"),
                  hist.axis.Regular(2, 0, 2, name="CRC_mismatch", label="CRC Mismatch"))

    h.fill(input_df["hamming_count"].values, input_df["CRC_mismatch"].values)

    return h

## --------------------------------------
def return_crc_hist(
        input_df: pd.DataFrame,
        chipNames: list[str],
        chipLabels: list[int],
):
    h = {chipNames[board_idx]: hist.Hist(
            hist.axis.Regular(2, 0, 2, name="CRC_mismatch", label="CRC Mismatch"),
        )
    for board_idx in range(len(chipLabels))}


    for board_idx in range(len(chipLabels)):
        tmp_df = input_df.loc[input_df['board'] == chipLabels[board_idx]]
        h[chipNames[board_idx]].fill(tmp_df['CRC_mismatch'].values)

    return h

## --------------------------------------
def return_hist_pivot(
        input_df: pd.DataFrame,
        board_ids: list[int],
        board_names: list[str],
        hist_bins: list = [50, 64, 64]
):
    h = {board_names[board_idx]: hist.Hist(hist.axis.Regular(hist_bins[0], 140, 240, name="CAL", label="CAL [LSB]"),
                hist.axis.Regular(hist_bins[1], 0, 512,  name="TOT", label="TOT [LSB]"),
                hist.axis.Regular(hist_bins[2], 0, 1024, name="TOA", label="TOA [LSB]"),
        )
    for board_idx in range(len(board_ids))}

    for idx, board_id in enumerate(board_ids):
        h[board_names[idx]].fill(input_df['cal'][board_id].values, input_df['tot'][board_id].values, input_df['toa'][board_id].values)

    return h

## --------------------------------------
def plot_BL_and_NW(
        run_time_df: pd.DataFrame,
        which_run: int,
        baseline_df: pd.DataFrame,
        config_dict: dict,
        which_val: str,
        save_mother_dir: Path | None = None,
    ):
    """Make Basline of Noise Width 2d map.

    Parameters
    ----------
    run_time_df: pd.DataFrame,
        Include TB run information in csv. Check reading_history/tb_run_info.
    which_run: int,
        Run number.
    baseline_df: pd.DataFrame,
        Baseline and Noise Width dataframe. Saved in SQL format.
    config_dict: dict,
        A dictionary of user config. Should include title for plot, chip_type (determine the color range), and channel (determine HV value).
    which_val: str,
        Either which_val = 'baseline' or which_val = 'noise_width.
    save_mother_dir: Path, optional
        Plot will be saved at save_mother_dir/'occupancy_map'.
    """

    cut_time1 = run_time_df.loc[run_time_df['Run'] == which_run-1 ,'Start_Time'].values[0]
    cut_time2 = run_time_df.loc[run_time_df['Run'] == which_run ,'Start_Time'].values[0]

    selected_run_df = baseline_df.loc[(baseline_df['timestamp'] > cut_time1) & (baseline_df['timestamp'] < cut_time2)]

    single_run_df = run_time_df.loc[run_time_df['Run'] == which_run, ["HV0", "HV1", "HV2", "HV3"]]
    HVs = single_run_df.iloc[0, 0:].to_numpy()

    if selected_run_df.shape[0] != 1024:
        selected_run_df = selected_run_df.loc[selected_run_df.groupby(['row', 'col', 'chip_name'])['timestamp'].idxmax()].reset_index(drop=True)

    for iboard in selected_run_df['chip_name'].unique():
        tmp_df = selected_run_df.loc[selected_run_df['chip_name']==iboard]

        # Create a pivot table to reshape the data for plotting
        pivot_table = tmp_df.pivot(index='row', columns='col', values=which_val)

        if pivot_table.empty:
            continue

        if (pivot_table.shape[0] != 16) or (pivot_table.shape[1]!= 16):
            pivot_table = pivot_table.reindex(pd.Index(np.arange(0,16), name='')).reset_index()
            pivot_table = pivot_table.reindex(columns=np.arange(0,16))
            pivot_table = pivot_table.fillna(-1)

        # # Create a heatmap to visualize the count of hits
        fig, ax = plt.subplots(dpi=100, figsize=(12, 12))

        if which_val == 'baseline':
            if config_dict[iboard]['chip_type'] == "T":
                im = ax.imshow(pivot_table, interpolation="nearest", vmin=300, vmax=500)
            elif config_dict[iboard]['chip_type'] == "F":
                im = ax.imshow(pivot_table, interpolation="nearest", vmin=50, vmax=250)
            # # Add color bar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, extend='both')
            cbar.set_label('Baseline', fontsize=25)
            cbar.ax.tick_params(labelsize=18)
        elif which_val == 'noise_width':
            im = ax.imshow(pivot_table, interpolation="nearest", vmin=0, vmax=16)

            # # Add color bar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Noise Width', fontsize=25)
            cbar.ax.tick_params(labelsize=18)

        for i in range(16):
            for j in range(16):
                value = pivot_table.iloc[i, j]
                if value == -1: continue
                text = str("{:.0f}".format(value))
                plt.text(j, i, text, va='center', ha='center', color='white', fontsize=14)

        hep.cms.text(loc=0, ax=ax, text="ETL ETROC Test Beam", fontsize=18)
        ax.set_xlabel('Column', fontsize=25)
        ax.set_ylabel('Row', fontsize=25)
        ticks = range(0, 16)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_title(f"{config_dict[iboard]['plot_title'].replace('_', ' ')} HV{HVs[config_dict[iboard]['channel']]}V", loc="right", size=16)
        ax.tick_params(axis='x', which='both', length=5, labelsize=17)
        ax.tick_params(axis='y', which='both', length=5, labelsize=17)
        ax.invert_xaxis()
        ax.invert_yaxis()
        plt.minorticks_off()
        plt.tight_layout()

        if save_mother_dir is not None:
            pass

## --------------------------------------
def plot_number_of_fired_board(
        input_df: pd.DataFrame,
        tb_loc: str,
        fig_tag: str = '',
        do_logy: bool = False,
        save_mother_dir: Path | None = None,
    ):
    """Make a plot of number of fired boards in events.

    Parameters
    ----------
    input_df: pd.DataFrame,
        Input dataframe.
    tb_loc: str,
        Test Beam location for the title. Available argument: desy, cern, fnal.
    fig_tag: str, optional
        Additional figure tag to put in the title.
    do_logy: str, optional
        Log y-axis.
    save_mother_dir: Path, optional
        Plot will be saved at save_mother_dir/'misc'.
    """

    plot_title = load_fig_title(tb_loc)
    h = hist.Hist(hist.axis.Regular(5, 0, 5, name="nBoards", label="nBoards"))
    h.fill(input_df.groupby('evt')['board'].nunique())

    fig = plt.figure(figsize=(11,10))
    gs = fig.add_gridspec(1,1)
    ax = fig.add_subplot(gs[0,0])
    hep.cms.text(loc=0, ax=ax, text="ETL ETROC Test Beam", fontsize=18)
    ax.set_title(f"{plot_title} {fig_tag}", loc="right", size=16)
    ax.xaxis.label.set_fontsize(25)
    ax.yaxis.label.set_fontsize(25)
    h.plot1d(ax=ax, lw=2)
    ax.get_yaxis().get_offset_text().set_position((-0.05, 0))
    if do_logy:
        ax.set_yscale('log')
    plt.tight_layout()

    if save_mother_dir is not None:
        save_dir = save_mother_dir / 'misc'
        save_dir.mkdir(exist_ok=True)
        fig.savefig(save_dir / f"number_of_fired_board.png")
        fig.savefig(save_dir / f"number_of_fired_board.pdf")
        plt.close(fig)

## --------------------------------------
def plot_number_of_hits_per_event(
        input_df: pd.DataFrame,
        tb_loc: str,
        board_names: list[str],
        do_logy: bool = False,
        save_mother_dir: Path | None = None,
    ):
    """Make a plot of number of hits per event.

    Parameters
    ----------
    input_df: pd.DataFrame,
        Input dataframe.
    tb_loc: str,
        Test Beam location for the title. Available argument: desy, cern, fnal
    board_names: list[str],
        A list of board names.
    do_logy: str, optional
        Log y-axis.
    save_mother_dir: Path, optional
        Plot will be saved at save_mother_dir/'misc'.
    """

    plot_title = load_fig_title(tb_loc)
    hit_df = input_df.groupby(['evt', 'board']).size().unstack(fill_value=0)
    hists = {}

    for key in hit_df.columns:
        max_hit = hit_df[key].unique().max()
        hists[key] = hist.Hist(hist.axis.Regular(max_hit, 0, max_hit, name="nHits", label='Number of Hits'))
        hists[key].fill(hit_df[key])

    for idx, ikey in enumerate(hists.keys()):
        fig, ax = plt.subplots(figsize=(11, 10))
        hep.cms.text(loc=0, ax=ax, text="ETL ETROC Test Beam", fontsize=18)
        hists[ikey].plot1d(ax=ax, lw=2)
        ax.set_title(f"{plot_title} | {board_names[idx]}", loc="right", size=16)
        ax.get_yaxis().get_offset_text().set_position((-0.05, 0))

        if do_logy:
            ax.set_yscale('log')

        plt.tight_layout()
        if save_mother_dir is not None:
            save_dir = save_mother_dir / 'misc'
            save_dir.mkdir(exist_ok=True)
            fig.savefig(save_dir / f"number_of_hits_{board_names[idx]}.png")
            fig.savefig(save_dir / f"number_of_hits_{board_names[idx]}.pdf")
            plt.close(fig)

## --------------------------------------
def plot_2d_nHits_nBoard(
        input_df: pd.DataFrame,
        fig_titles: list[str],
        fig_tag: str = '',
        bins: int = 15,
        hist_range: tuple = (0, 15),

    ):
    nboard_df = input_df.groupby('evt')['board'].nunique()
    hit_df = input_df.groupby(['evt', 'board']).size().unstack(fill_value=0)
    hit_df.dropna(subset=[0], inplace=True)
    hists = {}

    for key in hit_df.columns:
        hists[key] = hist.Hist(
            hist.axis.Regular(bins, hist_range[0], hist_range[1], name="nHits", label='nHits'),
            hist.axis.Regular(5, 0, 5, name="nBoards", label="nBoards")
        )
        hists[key].fill(hit_df[key], nboard_df)

    fig = plt.figure(dpi=100, figsize=(30,13))
    gs = fig.add_gridspec(2,2)

    for i, plot_info in enumerate(gs):

        if i not in hists.keys():
            continue

        ax = fig.add_subplot(plot_info)
        hep.cms.text(loc=0, ax=ax, text="ETL ETROC Test Beam", fontsize=20)
        hep.hist2dplot(hists[i], ax=ax, norm=colors.LogNorm())
        ax.set_title(f"{fig_titles[i]} {fig_tag}", loc="right", size=18)

    plt.tight_layout()
    del hists, hit_df, nboard_df

## --------------------------------------
def plot_occupany_map(
        input_df: pd.DataFrame,
        board_ids: list[int],
        board_names: list[str],
        tb_loc: str,
        fname_tag: str = '',
        exclude_noise: bool = False,
        save_mother_dir: Path | None = None,
    ):
    """Make occupancy plot.

    Parameters
    ----------
    input_df: pd.DataFrame,
        Pandas dataframe of data.
    board_ids: list[int],
        A list of integer (board ID) that wants to make plots.
    board_names: list[str],
        A list of board name that will use for the file name.
    tb_loc: str,
        Test Beam location for the title. Available argument: desy, cern, fnal.
    fname_tag: str, optional
        Draw boundary cut in the plot.
    exclude_noise: bool, optional
        Remove hits when TOT < 10 before plotting.
    save_mother_dir: Path, optional
        Plot will be saved at save_mother_dir/'occupancy_map'.
    """

    from matplotlib import colormaps
    cmap = colormaps['viridis']
    cmap.set_under(color='lightgrey')

    plot_title = load_fig_title(tb_loc)

    if exclude_noise:
        ana_df = input_df.loc[input_df['tot'] > 10].copy()
    else:
        ana_df = input_df

    # Group the DataFrame by 'col,' 'row,' and 'board,' and count the number of hits in each group
    hits_count_by_col_row_board = ana_df.groupby(['col', 'row', 'board'])['evt'].count().reset_index()
    del ana_df

    # Rename the 'evt' column to 'hits'
    hits_count_by_col_row_board = hits_count_by_col_row_board.rename(columns={'evt': 'hits'})

    for idx ,board_id in enumerate(board_ids):
        # Create a pivot table to reshape the data for plotting
        pivot_table = hits_count_by_col_row_board[hits_count_by_col_row_board['board'] == board_id].pivot_table(
            index='row',
            columns='col',
            values='hits',
            fill_value=0  # Fill missing values with 0 (if any)
        )

        if pivot_table.empty:
            continue

        if (pivot_table.shape[0] != 16) or (pivot_table.shape[1]!= 16):
            pivot_table = pivot_table.reindex(pd.Index(np.arange(0,16), name='')).reset_index()
            pivot_table = pivot_table.reindex(columns=np.arange(0,16))
            pivot_table = pivot_table.fillna(-1)

        # Create a heatmap to visualize the count of hits
        fig, ax = plt.subplots(dpi=100, figsize=(12, 12))
        ax.cla()
        im = ax.imshow(pivot_table, cmap=cmap, interpolation="nearest", vmin=0)

        # Add color bar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Hits', fontsize=25)
        cbar.ax.tick_params(labelsize=18)

        for i in range(16):
            for j in range(16):
                value = pivot_table.iloc[i, j]
                if value == -1: continue
                text_color = 'black' if value > 0.5*(pivot_table.values.max() + pivot_table.values.min()) else 'white'
                text = str("{:.0f}".format(value))
                plt.text(j, i, text, va='center', ha='center', color=text_color, fontsize=12)

        hep.cms.text(loc=0, ax=ax, text="ETL ETROC Test Beam", fontsize=18)
        ax.set_xlabel('Column', fontsize=25)
        ax.set_ylabel('Row', fontsize=25)
        ticks = range(0, 16)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_title(f"{plot_title} | {board_names[idx].replace('_', ' ')}", loc="right", size=16)
        ax.tick_params(axis='x', which='both', length=5, labelsize=17)
        ax.tick_params(axis='y', which='both', length=5, labelsize=17)
        ax.invert_xaxis()
        ax.invert_yaxis()
        plt.minorticks_off()
        plt.tight_layout()

        if save_mother_dir is not None:
            save_dir = save_mother_dir / 'occupancy_map'
            save_dir.mkdir(exist_ok=True)
            fig.savefig(save_dir / f"occupancy_{board_names[board_id]}_{fname_tag}.png")
            fig.savefig(save_dir / f"occupancy_{board_names[board_id]}_{fname_tag}.pdf")
            plt.close(fig)

## --------------------------------------
def plot_3d_occupany_map(
        input_df: pd.DataFrame,
        board_ids: list[int],
        board_names: list[str],
        tb_loc: str,
        fname_tag: str = '',
        save_mother_dir: Path | None = None,
    ):
    """Make 3D occupancy plot.

    Parameters
    ----------
    input_df: pd.DataFrame,
        Pandas dataframe of data.
    board_ids: list[int],
        A list of integer (board ID) that wants to make plots.
    board_names: list[str],
        A list of board name that will use for the file name.
    tb_loc: str,
        Test Beam location for the title. Available argument: desy, cern, fnal.
    fname_tag: str, optional
        Draw boundary cut in the plot.
    save_mother_dir: Path, optional
        Plot will be saved at save_mother_dir/'occupancy_map'.
    """

    plot_title = load_fig_title(tb_loc)
    hits_count_by_col_row_board = input_df.groupby(['col', 'row', 'board'])['evt'].count().reset_index()

    # Rename the 'evt' column to 'hits'
    hits_count_by_col_row_board = hits_count_by_col_row_board.rename(columns={'evt': 'hits'})

    for idx ,board_id in enumerate(board_ids):
        # Create a pivot table to reshape the data for plotting
        pivot_table = hits_count_by_col_row_board[hits_count_by_col_row_board['board'] == board_id].pivot_table(
            index='row',
            columns='col',
            values='hits',
            fill_value=0  # Fill missing values with 0 (if any)
        )
        fig = plt.figure(figsize=(11, 10))
        ax = fig.add_subplot(111, projection='3d')
        hep.cms.text(loc=0, ax=ax, text="ETL ETROC Test Beam", fontsize=18)

        # Create a meshgrid for the 3D surface
        x, y = np.meshgrid(np.arange(16), np.arange(16))
        z = pivot_table.values
        dx = dy = 0.75  # Width and depth of the bars

        # Create a 3D surface plot
        ax.bar3d(x.flatten(), y.flatten(), np.zeros_like(z).flatten(), dx, dy, z.flatten(), shade=True)

        # Customize the 3D plot settings as needed
        ax.set_xlabel('COL', fontsize=15, labelpad=15)
        ax.set_ylabel('ROW', fontsize=15, labelpad=15)
        ax.set_zlabel('Hits', fontsize=15, labelpad=-35)
        ax.invert_xaxis()
        ticks = range(0, 16)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticks(ticks=range(16), labels=[], minor=True)
        ax.set_yticks(ticks=range(16), labels=[], minor=True)
        ax.tick_params(axis='x', labelsize=8)  # You can adjust the 'pad' value
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='z', labelsize=8)
        ax.set_title(f"{plot_title} | {board_names[idx]}", fontsize=14, loc='right')
        plt.tight_layout()

        if save_mother_dir is not None:
            save_dir = save_mother_dir / 'occupancy_map'
            save_dir.mkdir(exist_ok=True)
            fig.savefig(save_dir / f"3D_occupancy_{board_names[board_id]}_{fname_tag}.png")
            fig.savefig(save_dir / f"3D_occupancy_{board_names[board_id]}_{fname_tag}.pdf")
            plt.close(fig)

## --------------------------------------
def plot_TDC_summary_table(
        input_df: pd.DataFrame,
        chipLabels: list[int],
        var: str
    ):

    from matplotlib import colormaps
    cmap = colormaps['viridis']
    cmap.set_under(color='lightgrey')

    for id in chipLabels:

        if input_df[input_df['board'] == id].empty:
            continue

        sum_group = input_df[input_df['board'] == id].groupby(["col", "row"]).agg({var:['mean','std']})
        sum_group.columns = sum_group.columns.droplevel()
        sum_group.reset_index(inplace=True)

        table_mean = sum_group.pivot_table(index='row', columns='col', values='mean', fill_value=-1)
        table_mean = table_mean.round(1)

        table_mean = table_mean.reindex(pd.Index(np.arange(0,16), name='')).reset_index()
        table_mean = table_mean.reindex(columns=np.arange(0,16))

        table_std = sum_group.pivot_table(index='row', columns='col', values='std', fill_value=-1)
        table_std = table_std.round(2)

        table_std = table_std.reindex(pd.Index(np.arange(0,16), name='')).reset_index()
        table_std = table_std.reindex(columns=np.arange(0,16))

        plt.rcParams["xtick.major.size"] = 2.5
        plt.rcParams["ytick.major.size"] = 2.5
        plt.rcParams['xtick.minor.visible'] = False
        plt.rcParams['ytick.minor.visible'] = False

        fig, axes = plt.subplots(1, 2, figsize=(20, 20))

        im1 = axes[0].imshow(table_mean, cmap=cmap, vmin=0)
        im2 = axes[1].imshow(table_std, cmap=cmap, vmin=0)

        hep.cms.text(loc=0, ax=axes[0], text="ETL ETROC Test Beam", fontsize=25)
        hep.cms.text(loc=0, ax=axes[1], text="ETL ETROC Test Beam", fontsize=25)

        axes[0].set_title(f'{var.upper()} Mean', loc="right")
        axes[1].set_title(f'{var.upper()} Std', loc="right")

        axes[0].set_xticks(np.arange(0,16))
        axes[0].set_yticks(np.arange(0,16))
        axes[1].set_xticks(np.arange(0,16))
        axes[1].set_yticks(np.arange(0,16))

        # i for col, j for row
        for i in range(16):
            for j in range(16):
                if np.isnan(table_mean.iloc[i,j]) or table_mean.iloc[i,j] < 0.:
                    continue
                text_color = 'black' if table_mean.iloc[i,j] > 0.5*(table_mean.stack().max() + table_mean.stack().min()) else 'white'
                axes[0].text(j, i, table_mean.iloc[i,j], ha="center", va="center", rotation=45, fontweight="bold", fontsize=12, color=text_color)

        for i in range(16):
            for j in range(16):
                if np.isnan(table_std.iloc[i,j]) or table_std.iloc[i,j] < 0.:
                    continue
                text_color = 'black' if table_std.iloc[i,j] > 0.5*(table_std.stack().max() + table_std.stack().min()) / 2 else 'white'
                axes[1].text(j, i, table_std.iloc[i,j], ha="center", va="center", rotation=45, color=text_color, fontweight="bold", fontsize=12)

        axes[0].invert_xaxis()
        axes[0].invert_yaxis()
        axes[1].invert_xaxis()
        axes[1].invert_yaxis()

        plt.minorticks_off()
        plt.tight_layout()

## --------------------------------------
def plot_1d_TDC_histograms(
        input_hist: dict,
        board_name: str,
        tb_loc: str,
        extra_cms_title: str = 'ETL ETROC Test Beam',
        fig_tag: str | None = None,
        slide_friendly: bool = False,
        do_logy: bool = False,
        event_hist: hist.Hist | None = None,
        save_mother_dir: Path | None = None,
        tag: str = '',
    ):
    """Make plots of 1D TDC histograms.

    Parameters
    ----------
    input_hist: dict,
        A dictionary of TDC histograms, which returns from return_hist, return_hist_pivot
    board_name: str,
        Board name.
    tb_loc: str,
        Test Beam location for the title. Available argument: desy, cern, fnal.
    fig_tag: str, optional
        Additional board information to show in the plot.
    slide_friendly: bool, optional
        If it is True, draw plots in a single figure. Recommend this option, when you try to add plots on the slides.
    do_logy: bool, optional
        Set log y-axis on 1D histograms.
    event_hist: hist.Hist, optional
        A dictionary of TDC histograms, which returns from return_event_hist
    save_mother_dir: Path, optional
        Plot will be saved at save_mother_dir/'1d_tdc_hists'.
    tag: str, optional (recommend),
        Additional tag for the file name.
    """

    plot_title = load_fig_title(tb_loc)
    if not slide_friendly:

        vals = ["CAL", "TOT", "TOA", "EA"]
        for ival in vals:
            try:
                fig, ax = plt.subplots(figsize=(11, 10))
                ax.set_title(plot_title, loc="right", size=16)
                hep.cms.text(loc=0, ax=ax, text=extra_cms_title, fontsize=18)
                input_hist[board_name].project(ival)[:].plot1d(ax=ax, lw=2)
                ax.xaxis.label.set_fontsize(25)
                ax.yaxis.label.set_fontsize(25)

                if fig_tag is not None:
                    ax.text(0.98, 0.97, fig_tag, transform=ax.transAxes, fontsize=17, verticalalignment='top', horizontalalignment='right')

                if do_logy:
                    ax.set_yscale('log')
                plt.tight_layout()

                if save_mother_dir is not None:
                    save_dir = save_mother_dir / '1d_tdc_hists'
                    save_dir.mkdir(exist_ok=True)
                    fig.savefig(save_dir / f'{board_name}_{ival}_{tag}.png')
                    fig.savefig(save_dir / f'{board_name}_{ival}_{tag}.pdf')
                    plt.close(fig)
            except Exception as e:
                plt.close(fig)
                print(f'No {ival} histogram is found')

        ## 2D TOA-TOT
        fig, ax = plt.subplots(figsize=(11, 10))
        ax.set_title(plot_title, loc="right", size=16)
        ax.xaxis.label.set_fontsize(25)
        ax.yaxis.label.set_fontsize(25)
        hep.cms.text(loc=0, ax=ax, text=extra_cms_title, fontsize=18)
        hep.hist2dplot(input_hist[board_name].project("TOA","TOT")[::2j,::2j], ax=ax)

        if fig_tag is not None:
            ax.text(0.98, 0.97, fig_tag, transform=ax.transAxes, fontsize=17, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white'))

        plt.tight_layout()

        if save_mother_dir is not None:
            save_dir = save_mother_dir / '1d_tdc_hists'
            save_dir.mkdir(exist_ok=True)
            fig.savefig(save_dir / f'{board_name}_TOA_TOT_{tag}.png')
            fig.savefig(save_dir / f'{board_name}_TOA_TOT_{tag}.pdf')
            plt.close(fig)

        if event_hist is not None:
            fig, ax = plt.subplots(figsize=(11, 10))
            ax.set_title(plot_title, loc="right", size=16)
            hep.cms.text(loc=0, ax=ax, text=extra_cms_title, fontsize=18)
            event_hist.project("HA")[:].plot1d(ax=ax, lw=2)
            ax.xaxis.label.set_fontsize(25)
            ax.yaxis.label.set_fontsize(25)

            if fig_tag is not None:
                ax.text(0.98, 0.97, fig_tag, transform=ax.transAxes, fontsize=17, verticalalignment='top', horizontalalignment='right')

            if do_logy:
                ax.set_yscale('log')

            plt.tight_layout()

            if save_mother_dir is not None:
                save_dir = save_mother_dir / '1d_tdc_hists'
                save_dir.mkdir(exist_ok=True)
                fig.savefig(save_dir / f'{board_name}_Hamming_Count_{tag}.png')
                fig.savefig(save_dir / f'{board_name}_Hamming_Count_{tag}.pdf')
                plt.close(fig)

    else:
        fig = plt.figure(dpi=100, figsize=(30,13))
        gs = fig.add_gridspec(2,2)

        for i, plot_info in enumerate(gs):
            ax = fig.add_subplot(plot_info)
            hep.cms.text(loc=0, ax=ax, text=extra_cms_title, fontsize=18)
            if i == 0:
                ax.set_title(plot_title, loc="right", size=16)
                input_hist[board_name].project("CAL")[:].plot1d(ax=ax, lw=2)
                if do_logy:
                    ax.set_yscale('log')
            elif i == 1:
                ax.set_title(plot_title, loc="right", size=16)
                input_hist[board_name].project("TOA")[:].plot1d(ax=ax, lw=2)
                if do_logy:
                    ax.set_yscale('log')
            elif i == 2:
                ax.set_title(plot_title, loc="right", size=16)
                input_hist[board_name].project("TOT")[:].plot1d(ax=ax, lw=2)
                if do_logy:
                    ax.set_yscale('log')
            elif i == 3:
                if event_hist is None:
                    ax.set_title(plot_title, loc="right", size=16)
                    input_hist[board_name].project("TOA","TOT")[::2j,::2j].plot2d(ax=ax)
                    if do_logy:
                        #pcm = plt.pcolor(self._data, norm = colors.LogNorm())
                        #plt.colorbar(pcm)
                        pass
                else:
                    ax.set_title(plot_title, loc="right", size=16)
                    event_hist.project("HA")[:].plot1d(ax=ax, lw=2)
                    if do_logy:
                        ax.set_yscale('log')

        plt.tight_layout()
        if save_mother_dir is not None:
            save_dir = save_mother_dir / '1d_tdc_hists'
            save_dir.mkdir(exist_ok=True)
            fig.savefig(save_dir / f'{board_name}_combined_{tag}.png')
            fig.savefig(save_dir / f'{board_name}_combined_{tag}.pdf')
            plt.close(fig)

## --------------------------------------
def plot_1d_event_CRC_histogram(
        input_hist: hist.Hist,
        fig_path: Path = Path('./'),
        save: bool = False,
        tag: str = '',
        fig_tag: str = '',
        do_logy: bool = False,
    ):
    fig = plt.figure(dpi=50, figsize=(20,10))
    gs = fig.add_gridspec(1,1)
    ax = fig.add_subplot(gs[0,0])
    ax.set_title(f"Event CRC Check{fig_tag}", loc="right", size=16)
    hep.cms.text(loc=0, ax=ax, text="ETL ETROC Test Beam", fontsize=18)
    input_hist.project("CRC_mismatch")[:].plot1d(ax=ax, lw=2)
    if do_logy:
        ax.set_yscale('log')
    plt.tight_layout()
    if(save):
        plt.savefig(fig_path/f'Event_CRCCheck_{tag}.pdf')
        plt.clf()
        plt.close(fig)

## ------------------------------------
def plot_1d_CRC_histogram(
        input_hist: hist.Hist,
        chip_name: str,
        chip_figname: str,
        fig_title: str,
        fig_path: Path = Path('./'),
        save: bool = False,
        tag: str = '',
        fig_tag: str = '',
        do_logy: bool = False,
    ):
    fig = plt.figure(dpi=50, figsize=(20,10))
    gs = fig.add_gridspec(1,1)
    ax = fig.add_subplot(gs[0,0])
    ax.set_title(f"{fig_title}, CRC Check{fig_tag}", loc="right", size=25)
    hep.cms.text(loc=0, ax=ax, text="ETL ETROC Test Beam", fontsize=25)
    input_hist[chip_name].project("CRC_mismatch")[:].plot1d(ax=ax, lw=2)
    if do_logy:
        ax.set_yscale('log')
    plt.tight_layout()
    if(save):
        plt.savefig(fig_path/f'{chip_figname}_CRCCheck_{tag}.pdf')
        plt.clf()
        plt.close(fig)

## --------------------------------------
def plot_correlation_of_pixels(
        input_df: pd.DataFrame,
        board_ids: list[int],
        board_name1: str,
        board_name2: str,
        tb_loc: str,
        fname_tag: str = '',
        save_mother_dir: Path | None = None,
    ):
    """Make pixel row-column correlation plot.

    Parameters
    ----------
    input_df: pd.DataFrame,
        Pandas dataframe of data.
    board_ids: list[int],
        A list of integer (board ID) that wants to make plots.
    board_name1: str,
        Board 1 name.
    board_name2: str,
        Board 2 name.
    tb_loc: str,
        Test Beam location for the title. Available argument: desy, cern, fnal.
    fname_tag: str, optional (recommend)
        Additiional tag for the file name.
    save_mother_dir: Path, optional
        Plot will be saved at save_mother_dir/'spatial_correlation'.
    """

    plot_title = load_fig_title(tb_loc)
    axis_name1 = board_name1.replace('_', ' ')
    axis_name2 = board_name2.replace('_', ' ')

    h_row = hist.Hist(
        hist.axis.Regular(16, 0, 16, name='row1', label=f'Row of {axis_name1}'),
        hist.axis.Regular(16, 0, 16, name='row2', label=f'Row of {axis_name2}'),
    )
    h_col = hist.Hist(
        hist.axis.Regular(16, 0, 16, name='col1', label=f'Column of {axis_name1}'),
        hist.axis.Regular(16, 0, 16, name='col2', label=f'Column of {axis_name2}'),
    )

    h_row.fill(input_df[f'row_{board_ids[0]}'].values, input_df[f'row_{board_ids[1]}'].values)
    h_col.fill(input_df[f'col_{board_ids[0]}'].values, input_df[f'col_{board_ids[1]}'].values)

    location = np.arange(0, 16) + 0.5
    tick_labels = np.char.mod('%d', np.arange(0, 16))
    fig, ax = plt.subplots(1, 2, dpi=100, figsize=(23, 11))

    hep.hist2dplot(h_row, ax=ax[0], norm= colors.LogNorm())
    hep.cms.text(loc=0, ax=ax[0], text="ETL ETROC Test Beam", fontsize=18)
    ax[0].set_title(plot_title, loc="right", size=16)
    ax[0].xaxis.label.set_fontsize(25)
    ax[0].yaxis.label.set_fontsize(25)
    ax[0].xaxis.set_major_formatter(ticker.NullFormatter())
    ax[0].xaxis.set_minor_locator(ticker.FixedLocator(location))
    ax[0].xaxis.set_minor_formatter(ticker.FixedFormatter(tick_labels))
    ax[0].yaxis.set_major_formatter(ticker.NullFormatter())
    ax[0].yaxis.set_minor_locator(ticker.FixedLocator(location))
    ax[0].yaxis.set_minor_formatter(ticker.FixedFormatter(tick_labels))
    ax[0].tick_params(axis='both', which='major', length=0)

    hep.hist2dplot(h_col, ax=ax[1], norm= colors.LogNorm())
    hep.cms.text(loc=0, ax=ax[1], text="ETL ETROC Test Beam", fontsize=18)
    ax[1].set_title(plot_title, loc="right", size=16)
    ax[1].xaxis.label.set_fontsize(25)
    ax[1].yaxis.label.set_fontsize(25)
    ax[1].xaxis.set_major_formatter(ticker.NullFormatter())
    ax[1].xaxis.set_minor_locator(ticker.FixedLocator(location))
    ax[1].xaxis.set_minor_formatter(ticker.FixedFormatter(tick_labels))
    ax[1].yaxis.set_major_formatter(ticker.NullFormatter())
    ax[1].yaxis.set_minor_locator(ticker.FixedLocator(location))
    ax[1].yaxis.set_minor_formatter(ticker.FixedFormatter(tick_labels))
    ax[1].tick_params(axis='both', which='major', length=0)

    plt.tight_layout()

    if save_mother_dir is not None:
        save_dir = save_mother_dir / 'spatial_correlation'
        save_dir.mkdir(exist_ok=True)
        fig.savefig(save_dir / f"spatial_correlation_{board_name1}_{board_name2}_{fname_tag}.png")
        fig.savefig(save_dir / f"spatial_correlation_{board_name1}_{board_name2}_{fname_tag}.pdf")
        plt.close(fig)

## --------------------------------------
def plot_difference_of_pixels(
        input_df: pd.DataFrame,
        board_ids: list[int],
        board_name1: str,
        board_name2: str,
        tb_loc: str,
        fname_tag: str = '',
        save_mother_dir: Path | None = None,
    ):
    """Make 2D map of delta Row and delta Column.

    Parameters
    ----------
    input_df: pd.DataFrame,
        Pandas dataframe of data.
    board_ids: list[int],
        A list of integer (board ID) that wants to make plots.
    board_name1: str,
        Board 1 name.
    board_name2: str,
        Board 2 name.
    tb_loc: str,
        Test Beam location for the title. Available argument: desy, cern, fnal.
    fname_tag: str, optional (recommend)
        Additiional tag for the file name.
    save_mother_dir: Path, optional
        Plot will be saved at save_mother_dir/'spatial_correlation'.
    """

    plot_title = load_fig_title(tb_loc)
    diff_row = (input_df[f'row_{board_ids[0]}'].astype(np.int8) - input_df[f'row_{board_ids[1]}'].astype(np.int8)).values
    diff_col = (input_df[f'col_{board_ids[0]}'].astype(np.int8) - input_df[f'col_{board_ids[1]}'].astype(np.int8)).values

    h = hist.Hist(
        hist.axis.Regular(32, -16, 16, name='delta_row', label=r"$\Delta$Row"),
        hist.axis.Regular(32, -16, 16, name='delta_col', label=r"$\Delta$Col"),
    )

    h.fill(diff_row, diff_col)

    fig, ax = plt.subplots(dpi=100, figsize=(11, 11))

    hep.hist2dplot(h, ax=ax, norm=colors.LogNorm())
    hep.cms.text(loc=0, ax=ax, text="ETL ETROC Test Beam", fontsize=18)
    ax.set_title(plot_title, loc="right", size=16)
    ax.xaxis.label.set_fontsize(25)
    ax.yaxis.label.set_fontsize(25)
    ax.tick_params(axis='x', which='both', length=5, labelsize=17)
    ax.tick_params(axis='y', which='both', length=5, labelsize=17)
    plt.minorticks_off()
    plt.tight_layout()

    if save_mother_dir is not None:
        save_dir = save_mother_dir / 'spatial_correlation'
        save_dir.mkdir(exist_ok=True)
        fig.savefig(save_dir / f"spatial_difference_{board_name1}_{board_name2}_{fname_tag}.png")
        fig.savefig(save_dir / f"spatial_difference_{board_name1}_{board_name2}_{fname_tag}.pdf")
        plt.close(fig)

## --------------------------------------
def plot_distance(
        input_df: pd.DataFrame,
        board_ids: np.array,
        xaxis_label_board_name: str,
        fig_title: str,
        fig_tag: str = '',
        do_logy: bool = False,
        no_show: bool = False,
    ):
    h_dis = hist.Hist(hist.axis.Regular(32, 0, 32, name='dis', label=f'Distance (Trigger - {xaxis_label_board_name})'))

    diff_row = (input_df.loc[input_df['board'] == board_ids[0]]['row'].reset_index(drop=True) - input_df.loc[input_df['board'] == board_ids[1]]['row'].reset_index(drop=True)).values
    diff_col = (input_df.loc[input_df['board'] == board_ids[0]]['col'].reset_index(drop=True) - input_df.loc[input_df['board'] == board_ids[1]]['col'].reset_index(drop=True)).values
    dis = np.sqrt(diff_row**2 + diff_col**2)
    h_dis.fill(dis)
    del diff_row, diff_col, dis

    fig, ax = plt.subplots(dpi=100, figsize=(15, 8))
    hep.histplot(h_dis, ax=ax)
    hep.cms.text(loc=0, ax=ax, text="ETL ETROC Test Beam", fontsize=18)
    ax.set_title(f"{fig_title} {fig_tag}", loc="right", size=16)

    if do_logy:
        ax.set_yscale('log')

    if no_show:
        plt.close(fig)

    return h_dis

## --------------------------------------
def plot_TOA_correlation(
        input_df: pd.DataFrame,
        board_id1: int,
        board_id2: int,
        boundary_cut: float,
        board_names: list[str],
        tb_loc: str,
        draw_boundary: bool = False,
        save_mother_dir: Path | None = None,
    ):
    """Make plot of TOA correlation between selected two boards.

    Parameters
    ----------
    input_df: pd.DataFrame,
        Pandas dataframe of a single track.
    board_id1: int,
        Board 1 ID.
    board_id2: int,
        Board 2 ID.
    boundary_cut: float,
        Size of boundary. boundary_cut * standard devition of distance arrays
    board_names: list[str],
        A string list including board names.
    tb_loc: str,
        Test Beam location for the title. Available argument: desy, cern, fnal.
    draw_boundary: bool, optional
        Draw boundary cut in the plot.
    save_mother_dir: Path, optional
        Plot will be saved at save_mother_dir/'temporal_correlation'.
    """

    plot_title = load_fig_title(tb_loc)
    x = input_df['toa'][board_id1]
    y = input_df['toa'][board_id2]

    axis_name1 = board_names[board_id1].replace('_', ' ')
    axis_name2 = board_names[board_id2].replace('_', ' ')

    h = hist.Hist(
        hist.axis.Regular(128, 0, 1024, name=f'{board_names[board_id1]}', label=f'TOA of {axis_name1} [LSB]'),
        hist.axis.Regular(128, 0, 1024, name=f'{board_names[board_id2]}', label=f'TOA of {axis_name2} [LSB]'),
    )
    h.fill(x, y)
    params = np.polyfit(x, y, 1)
    distance = (x*params[0] - y + params[1])/(np.sqrt(params[0]**2 + 1))

    fig, ax = plt.subplots(figsize=(11, 10))
    hep.cms.text(loc=0, ax=ax, text="ETL ETROC Test Beam", fontsize=18)
    ax.set_title(plot_title, loc='right', fontsize=16)
    ax.xaxis.label.set_fontsize(25)
    ax.yaxis.label.set_fontsize(25)
    hep.hist2dplot(h, ax=ax, norm=colors.LogNorm())

    # calculate the trendline
    trendpoly = np.poly1d(params)
    x_range = np.linspace(x.min(), x.max(), 500)

    # plot the trend line
    ax.plot(x_range, trendpoly(x_range), 'r-', label='linear fit')
    if draw_boundary:
        ax.plot(x_range, trendpoly(x_range)-boundary_cut*np.std(distance), 'r--', label=fr'{boundary_cut}$\sigma$ boundary')
        ax.plot(x_range, trendpoly(x_range)+boundary_cut*np.std(distance), 'r--')
        # ax.fill_between(x_range, y1=trendpoly(x_range)-boundary_cut*np.std(distance), y2=trendpoly(x_range)+boundary_cut*np.std(distance),
        #                 facecolor='red', alpha=0.35, label=fr'{boundary_cut}$\sigma$ boundary')
    ax.legend()

    if save_mother_dir is not None:
        save_dir = save_mother_dir / 'temporal_correlation'
        save_dir.mkdir(exist_ok=True)
        fig.savefig(save_dir / f"toa_correlation_{board_names[board_id1]}_{board_names[board_id2]}.png")
        fig.savefig(save_dir / f"toa_correlation_{board_names[board_id1]}_{board_names[board_id2]}.pdf")
        plt.close(fig)

## --------------------------------------
def plot_TWC(
        input_df: pd.DataFrame,
        board_ids: list[int],
        tb_loc: str,
        poly_order: int = 2,
        corr_toas: dict | None = None,
        save_mother_dir: Path | None = None,
        print_func: bool = False,
    ):

    plot_title = load_fig_title(tb_loc)

    if corr_toas is not None:
        del_toa_b0 = (0.5*(corr_toas[f'toa_b{board_ids[1]}'] + corr_toas[f'toa_b{board_ids[2]}']) - corr_toas[f'toa_b{board_ids[0]}'])
        del_toa_b1 = (0.5*(corr_toas[f'toa_b{board_ids[0]}'] + corr_toas[f'toa_b{board_ids[2]}']) - corr_toas[f'toa_b{board_ids[1]}'])
        del_toa_b2 = (0.5*(corr_toas[f'toa_b{board_ids[0]}'] + corr_toas[f'toa_b{board_ids[1]}']) - corr_toas[f'toa_b{board_ids[2]}'])
    else:
        del_toa_b0 = (0.5*(input_df[f'toa_b{board_ids[1]}'] + input_df[f'toa_b{board_ids[2]}']) - input_df[f'toa_b{board_ids[0]}']).values
        del_toa_b1 = (0.5*(input_df[f'toa_b{board_ids[0]}'] + input_df[f'toa_b{board_ids[2]}']) - input_df[f'toa_b{board_ids[1]}']).values
        del_toa_b2 = (0.5*(input_df[f'toa_b{board_ids[0]}'] + input_df[f'toa_b{board_ids[1]}']) - input_df[f'toa_b{board_ids[2]}']).values

    def roundup(x):
        return int(np.ceil(x / 100.0)) * 100

    tot_ranges = {}
    for idx in board_ids:
        min_value = roundup(input_df[f'tot_b{idx}'].min()) - 500
        max_value = roundup(input_df[f'tot_b{idx}'].max()) + 500
        if min_value < 0:
            min_value = 0
        tot_ranges[idx] = [min_value, max_value]

    h_twc1 = hist.Hist(
        hist.axis.Regular(50, tot_ranges[board_ids[0]][0], tot_ranges[board_ids[0]][1], name=f'tot_b{board_ids[0]}', label=f'tot_b{board_ids[0]}'),
        hist.axis.Regular(50, -3000, 3000, name=f'delta_toa{board_ids[0]}', label=f'delta_toa{board_ids[0]}')
    )
    h_twc2 = hist.Hist(
        hist.axis.Regular(50, tot_ranges[board_ids[1]][0], tot_ranges[board_ids[1]][1], name=f'tot_b{board_ids[1]}', label=f'tot_b{board_ids[1]}'),
        hist.axis.Regular(50, -3000, 3000, name=f'delta_toa{board_ids[1]}', label=f'delta_toa{board_ids[1]}')
    )
    h_twc3 = hist.Hist(
        hist.axis.Regular(50, tot_ranges[board_ids[2]][0], tot_ranges[board_ids[2]][1], name=f'tot_b{board_ids[2]}', label=f'tot_b{board_ids[2]}'),
        hist.axis.Regular(50, -3000, 3000, name=f'delta_toa{board_ids[2]}', label=f'delta_toa{board_ids[2]}')
    )

    h_twc1.fill(input_df[f'tot_b{board_ids[0]}'], del_toa_b0)
    h_twc2.fill(input_df[f'tot_b{board_ids[1]}'], del_toa_b1)
    h_twc3.fill(input_df[f'tot_b{board_ids[2]}'], del_toa_b2)

    b1_xrange = np.linspace(input_df[f'tot_b{board_ids[0]}'].min(), input_df[f'tot_b{board_ids[0]}'].max(), 100)
    b2_xrange = np.linspace(input_df[f'tot_b{board_ids[1]}'].min(), input_df[f'tot_b{board_ids[1]}'].max(), 100)
    b3_xrange = np.linspace(input_df[f'tot_b{board_ids[2]}'].min(), input_df[f'tot_b{board_ids[2]}'].max(), 100)

    coeff_b0 = np.polyfit(input_df[f'tot_b{board_ids[0]}'].values, del_toa_b0, poly_order)
    poly_func_b0 = np.poly1d(coeff_b0)

    coeff_b1 = np.polyfit(input_df[f'tot_b{board_ids[1]}'].values, del_toa_b1, poly_order)
    poly_func_b1 = np.poly1d(coeff_b1)

    coeff_b2 = np.polyfit(input_df[f'tot_b{board_ids[2]}'].values, del_toa_b2, poly_order)
    poly_func_b2 = np.poly1d(coeff_b2)

    def make_legend(coeff, poly_order):
        legend_str = ""
        for i in range(poly_order + 1):
            if round(coeff[i], 2) == 0:
                # Use scientific notation
                coeff_str = f"{coeff[i]:.2e}"
            else:
                # Use fixed-point notation
                coeff_str = f"{coeff[i]:.2f}"

            # Add x
            coeff_str = rf"{coeff_str}$x^{poly_order-i}$"

            # Add sign
            if coeff[i] > 0:
                coeff_str = f"+{coeff_str}"
                legend_str += coeff_str
            else:
                legend_str += coeff_str
        return legend_str

    if print_func:
        print(poly_func_b0)
        print(poly_func_b1)
        print(poly_func_b2)

    fig, axes = plt.subplots(1, 3, figsize=(38, 10))
    hep.hist2dplot(h_twc1, ax=axes[0], norm=colors.LogNorm())
    hep.cms.text(loc=0, ax=axes[0], text="ETL ETROC Test Beam", fontsize=18)
    axes[0].plot(b1_xrange, poly_func_b0(b1_xrange), 'r-', lw=3, label=make_legend(coeff_b0, poly_order=poly_order))
    axes[0].set_xlabel('TOT1 [ps]', fontsize=25)
    axes[0].set_ylabel('0.5*(TOA2+TOA3)-TOA1 [ps]', fontsize=25)
    axes[0].set_title(plot_title, fontsize=16, loc='right')
    hep.hist2dplot(h_twc2, ax=axes[1], norm=colors.LogNorm())
    hep.cms.text(loc=0, ax=axes[1], text="ETL ETROC Test Beam", fontsize=18)
    axes[1].plot(b2_xrange, poly_func_b1(b2_xrange), 'r-', lw=3, label=make_legend(coeff_b1, poly_order=poly_order))
    axes[1].set_xlabel('TOT2 [ps]', fontsize=25)
    axes[1].set_ylabel('0.5*(TOA1+TOA3)-TOA2 [ps]', fontsize=25)
    axes[1].set_title(plot_title, fontsize=16, loc='right')
    hep.hist2dplot(h_twc3, ax=axes[2], norm=colors.LogNorm())
    hep.cms.text(loc=0, ax=axes[2], text="ETL ETROC Test Beam", fontsize=18)
    axes[2].plot(b3_xrange, poly_func_b2(b3_xrange), 'r-', lw=3, label=make_legend(coeff_b2, poly_order=poly_order))
    axes[2].set_xlabel('TOT3 [ps]', fontsize=25)
    axes[2].set_ylabel('0.5*(TOA1+TOA2)-TOA3 [ps]', fontsize=25)
    axes[2].set_title(plot_title, fontsize=16, loc='right')

    axes[0].legend(loc='best')
    axes[1].legend(loc='best')
    axes[2].legend(loc='best')

    if save_mother_dir is not None:
        save_dir = save_mother_dir / 'twc_fit'
        save_dir.mkdir(exist_ok=True)
        fig.savefig(save_dir / f"twc_fit.png")
        fig.savefig(save_dir / f"twc_fit.pdf")
        plt.close(fig)

## --------------------------------------
def plot_resolution_with_pulls(
        input_df: pd.DataFrame,
        board_ids: list[int],
        board_names: list[str],
        tb_loc: str,
        fig_tag: list[str],
        hist_range: list[int] = [20, 95],
        hist_bins: int = 15,
        slides_friendly: bool = False,
        print_fit_results: bool = False,
        constraint_ylim: bool = False,
        save_mother_dir: Path | None = None,
    ):
    """Make summary plot of the board resolution plot with a gaussian fit.

    Parameters
    ----------
    input_df: pd.DataFrame
        Pandas dataframe includes bootstrap results.
    board_ids: list[int]
        A list of board IDs to make plots.
    board_names: list[str]
        A list of board names.
    tb_loc: str,
        Test Beam location for the title. Available argument: desy, cern, fnal.
    fig_tag: list[str]
        Additional information to show in the plot as legend title.
    hist_range: list[int], optional
        Set the histogram range. Default value is [20, 95].
    hist_bins: int, optional
        Adjust the histogram bins. Default value is 15.
    slide_friendly: bool, optional
        If it is True, draw plots in a single figure. Recommend this option, when you try to add plots on the slides.
    save_mother_dir: Path, optional
        Plot will be saved at save_mother_dir/'time_resolution_results'.
    """

    import matplotlib.gridspec as gridspec
    from lmfit.models import GaussianModel
    from lmfit.lineshapes import gaussian

    plot_title = load_fig_title(tb_loc)
    mod = GaussianModel(nan_policy='omit')

    hists = {}
    fit_params = {}
    pulls_dict = {}
    means = {}

    for key in board_ids:
        hists[key] = hist.Hist(hist.axis.Regular(hist_bins, hist_range[0], hist_range[1], name="time_resolution", label=r'Time Resolution [ps]'))
        hists[key].fill(input_df[f'res{key}'].values)
        means[key] = np.mean(input_df[f'res{key}'].values)
        centers = hists[key].axes[0].centers
        fit_range = centers[np.argmax(hists[key].values())-5:np.argmax(hists[key].values())+5]
        fit_vals = hists[key].values()[np.argmax(hists[key].values())-5:np.argmax(hists[key].values())+5]

        pars = mod.guess(fit_vals, x=fit_range)
        out = mod.fit(fit_vals, pars, x=fit_range, weights=1/np.sqrt(fit_vals))
        fit_params[key] = out

        if print_fit_results:
            print(key)
            print(out.fit_report())

        ### Calculate pull
        pulls = (hists[key].values() - out.eval(x=centers))/np.sqrt(out.eval(x=centers))
        pulls[np.isnan(pulls) | np.isinf(pulls)] = 0
        pulls_dict[key] = pulls


    if slides_friendly:

        # Create a figure with a 2x2 grid
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(2, 2)

        for i, plot in enumerate(gs):
            global_ax = fig.add_subplot(plot)
            inner_plot = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=global_ax, hspace=0, height_ratios=[3, 1])
            main_ax = fig.add_subplot(inner_plot[0])
            sub_ax = fig.add_subplot(inner_plot[1], sharex=main_ax)
            global_ax.xaxis.set_visible(False)
            global_ax.yaxis.set_visible(False)
            main_ax.xaxis.set_visible(False)

            if i not in hists:
                global_ax.set_axis_off()
                main_ax.set_axis_off()
                sub_ax.set_axis_off()
                continue

            centers = hists[i].axes[0].centers
            hep.cms.text(loc=0, ax=main_ax, text="ETL ETROC Test Beam", fontsize=20)
            main_ax.set_title(f'{plot_title} {fig_tag[i]}', loc="right", size=18)

            main_ax.errorbar(centers, hists[i].values(), np.sqrt(hists[i].variances()),
                            ecolor="steelblue", mfc="steelblue", mec="steelblue", fmt="o",
                            ms=6, capsize=1, capthick=2, alpha=0.8)

            main_ax.set_ylabel('Counts', fontsize=18)
            main_ax.set_ylim(-5, 190)
            main_ax.tick_params(axis='x', labelsize=18)
            main_ax.tick_params(axis='y', labelsize=18)

            x_min = centers[0]
            x_max = centers[-1]

            x_range = np.linspace(x_min, x_max, 200)
            popt = [par for name, par in fit_params[i].best_values.items()]
            pcov = fit_params[i].covar

            if np.isfinite(pcov).all():
                n_samples = 100
                vopts = np.random.multivariate_normal(popt, pcov, n_samples)
                sampled_ydata = np.vstack([gaussian(x_range, *vopt).T for vopt in vopts])
                model_uncert = np.nanstd(sampled_ydata, axis=0)
            else:
                model_uncert = np.zeros_like(np.sqrt(hists[i].variances()))

            main_ax.plot(x_range, fit_params[i].eval(x=x_range), color="hotpink", ls="-", lw=2, alpha=0.8,
                        label=fr"$\mu$:{fit_params[i].params['center'].value:.2f} $\pm$ {fit_params[i].params['center'].stderr:.2f} ps")
            main_ax.plot(np.NaN, np.NaN, color='none',
                        label=fr"$\sigma$: {abs(fit_params[i].params['sigma'].value):.2f} $\pm$ {abs(fit_params[i].params['sigma'].stderr):.2f} ps")

            main_ax.fill_between(
                x_range,
                fit_params[i].eval(x=x_range) - model_uncert,
                fit_params[i].eval(x=x_range) + model_uncert,
                color="hotpink",
                alpha=0.2,
                label='Fit Uncertainty'
            )
            main_ax.legend(fontsize=18, loc='best', title=fig_tag[i], title_fontsize=18)

            width = (x_max - x_min) / len(pulls_dict[i])
            sub_ax.axhline(1, c='black', lw=0.75)
            sub_ax.axhline(0, c='black', lw=1.2)
            sub_ax.axhline(-1, c='black', lw=0.75)
            sub_ax.bar(centers, pulls_dict[i], width=width, fc='royalblue')
            sub_ax.set_ylim(-2, 2)
            sub_ax.set_yticks(ticks=np.arange(-1, 2), labels=[-1, 0, 1], fontsize=20)
            sub_ax.set_xlabel('Pixel Time Resolution [ps]', fontsize=20)
            sub_ax.tick_params(axis='x', which='both', labelsize=20)
            sub_ax.set_ylabel('Pulls', fontsize=20, loc='center')

        plt.tight_layout()

        if save_mother_dir is not None:
            save_dir = save_mother_dir / 'time_resolution_results'
            save_dir.mkdir(exist_ok=True)
            fig.savefig(save_dir / f"board_res.png")
            fig.savefig(save_dir / f"board_res.pdf")
            plt.close(fig)

        del hists, fit_params, pulls_dict, mod

    else:
        for idx in hists.keys():
            fig = plt.figure(figsize=(11, 10))
            grid = fig.add_gridspec(2, 1, hspace=0, height_ratios=[3, 1])

            main_ax = fig.add_subplot(grid[0])
            sub_ax = fig.add_subplot(grid[1], sharex=main_ax)
            plt.setp(main_ax.get_xticklabels(), visible=False)

            centers = hists[idx].axes[0].centers
            hep.cms.text(loc=0, ax=main_ax, text="ETL ETROC Test Beam", fontsize=18)
            main_ax.set_title(f'{plot_title}', loc="right", size=16)

            main_ax.errorbar(centers, hists[idx].values(), np.sqrt(hists[idx].variances()),
                            ecolor="steelblue", mfc="steelblue", mec="steelblue", fmt="o",
                            ms=6, capsize=1, capthick=2, alpha=0.8)

            main_ax.set_ylabel('Counts', fontsize=25)
            main_ax.tick_params(axis='x', labelsize=20)
            main_ax.tick_params(axis='y', labelsize=20)
            if constraint_ylim:
                main_ax.set_ylim(-5, 190)

            x_min = centers[0]
            x_max = centers[-1]

            x_range = np.linspace(x_min, x_max, 500)
            popt = [par for name, par in fit_params[idx].best_values.items()]
            pcov = fit_params[idx].covar

            if np.isfinite(pcov).all():
                n_samples = 100
                vopts = np.random.multivariate_normal(popt, pcov, n_samples)
                sampled_ydata = np.vstack([gaussian(x_range, *vopt).T for vopt in vopts])
                model_uncert = np.nanstd(sampled_ydata, axis=0)
            else:
                model_uncert = np.zeros_like(np.sqrt(hists[i].variances()))

            main_ax.plot(x_range, fit_params[idx].eval(x=x_range), color="hotpink", ls="-", lw=2, alpha=0.8,
                        label=fr"$\mu$:{fit_params[idx].params['center'].value:.2f} $\pm$ {fit_params[idx].params['center'].stderr:.2f} ps")
            main_ax.plot(np.NaN, np.NaN, color='none',
                        label=fr"$\sigma$: {abs(fit_params[idx].params['sigma'].value):.2f} $\pm$ {abs(fit_params[idx].params['sigma'].stderr):.2f} ps")

            main_ax.fill_between(
                x_range,
                fit_params[idx].eval(x=x_range) - model_uncert,
                fit_params[idx].eval(x=x_range) + model_uncert,
                color="hotpink",
                alpha=0.2,
                label='Fit Uncertainty'
            )
            main_ax.legend(fontsize=18, loc='best', title=fig_tag[idx], title_fontsize=18)

            width = (x_max - x_min) / len(pulls_dict[idx])
            sub_ax.axhline(1, c='black', lw=0.75)
            sub_ax.axhline(0, c='black', lw=1.2)
            sub_ax.axhline(-1, c='black', lw=0.75)
            sub_ax.bar(centers, pulls_dict[idx], width=width, fc='royalblue')
            sub_ax.set_ylim(-2, 2)
            sub_ax.set_yticks(ticks=np.arange(-1, 2), labels=[-1, 0, 1], fontsize=20)
            sub_ax.set_xlabel('Pixel Time Resolution [ps]', fontsize=25)
            sub_ax.tick_params(axis='x', which='both', labelsize=20)
            sub_ax.set_ylabel('Pulls', fontsize=20, loc='center')

            plt.tight_layout()

            if save_mother_dir is not None:
                save_dir = save_mother_dir / 'time_resolution_results'
                save_dir.mkdir(exist_ok=True)
                fig.savefig(save_dir / f"board_res_{board_names[idx]}.png")
                fig.savefig(save_dir / f"board_res_{board_names[idx]}.pdf")
                plt.close(fig)

        del hists, fit_params, pulls_dict, mod

## --------------------------------------
def plot_resolution_table(
        input_df: pd.DataFrame,
        board_ids: list[int],
        board_names: list[str],
        tb_loc: str,
        fig_tag: str = '',
        min_resolution: float = 25.0,
        max_resolution: float = 75.0,
        missing_pixel_info: dict | None = None,
        slides_friendly: bool = False,
        show_number: bool = False,
        save_mother_dir: Path | None = None,
    ):

    from matplotlib import colormaps
    cmap = colormaps['viridis']
    cmap.set_under(color='lightgrey')

    plot_title = load_fig_title(tb_loc)

    tables = {}
    for board_id in board_ids:
        board_info = input_df[[f'row{board_id}', f'col{board_id}', f'res{board_id}', f'err{board_id}']]

        res = board_info.groupby([f'row{board_id}', f'col{board_id}']).apply(lambda x: np.average(x[f'res{board_id}'], weights=1/x[f'err{board_id}']**2)).reset_index()
        err = board_info.groupby([f'row{board_id}', f'col{board_id}']).apply(lambda x: np.sqrt(1/(np.sum(1/x[f'err{board_id}']**2)))).reset_index()

        res_table = res.pivot_table(index=f'row{board_id}', columns=f'col{board_id}', values=0, fill_value=-1)
        err_table = err.pivot_table(index=f'row{board_id}', columns=f'col{board_id}', values=0, fill_value=-1)

        res_table = res_table.reindex(pd.Index(np.arange(0,16), name='')).reset_index()
        res_table = res_table.reindex(columns=np.arange(0,16))
        res_table = res_table.fillna(-1)

        err_table = err_table.reindex(pd.Index(np.arange(0,16), name='')).reset_index()
        err_table = err_table.reindex(columns=np.arange(0,16))
        err_table = err_table.fillna(-1)

        tables[board_id] = [res_table, err_table]

    if slides_friendly:
        fig = plt.figure(figsize=(35, 35))
        gs = fig.add_gridspec(2, 2)

        for idx, plot in enumerate(gs):
            ax = fig.add_subplot(plot)

            if idx not in tables:
                ax.set_axis_off()
                continue
            im = ax.imshow(tables[idx][0], cmap=cmap, interpolation="nearest", vmin=min_resolution, vmax=max_resolution)

            # Add color bar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Time Resolution (ps)', fontsize=20)
            cbar.ax.tick_params(labelsize=18)

            if show_number:
                for i in range(16):
                    for j in range(16):
                        value = tables[idx][0].iloc[i, j]
                        error = tables[idx][1].iloc[i, j]
                        if value == -1: continue
                        text_color = 'black' if value > 0.66*(min_resolution + max_resolution) else 'white'
                        text = str(rf"{value:.1f}""\n"fr"$\pm$ {error:.1f}")
                        plt.text(j, i, text, va='center', ha='center', color=text_color, fontsize=18, rotation=45)

            hep.cms.text(loc=0, ax=ax, text="ETL ETROC Test Beam", fontsize=18)
            ax.set_xlabel('Column (col)', fontsize=18)
            ax.set_ylabel('Row (row)', fontsize=18)
            ticks = range(0, 16)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_title(f"{plot_title} | {fig_tag[idx]}", loc="right", size=18)
            ax.tick_params(axis='x', which='both', length=5, labelsize=18)
            ax.tick_params(axis='y', which='both', length=5, labelsize=18)
            ax.invert_xaxis()
            ax.invert_yaxis()
            plt.minorticks_off()

        plt.tight_layout()

        if save_mother_dir is not None:
            save_dir = save_mother_dir / 'time_resolution_results'
            save_dir.mkdir(exist_ok=True)
            fig.savefig(save_dir / f"resolution_map.png")
            fig.savefig(save_dir / f"resolution_map.pdf")
            plt.close(fig)
        del tables

    else:
        for idx in tables.keys():
            # Create a heatmap to visualize the count of hits
            fig, ax = plt.subplots(dpi=100, figsize=(15, 15))
            ax.cla()
            im = ax.imshow(tables[idx][0], cmap=cmap, interpolation="nearest", vmin=min_resolution, vmax=max_resolution)

            # Add color bar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Time Resolution (ps)', fontsize=18)
            cbar.ax.tick_params(labelsize=18)


            if show_number:
                for i in range(16):
                    for j in range(16):
                        value = tables[idx][0].iloc[i, j]
                        error = tables[idx][1].iloc[i, j]
                        if value == -1: continue
                        text_color = 'black' if value > 0.66*(min_resolution + max_resolution) else 'white'
                        text = str(rf"{value:.1f}""\n"fr"$\pm$ {error:.1f}")
                        plt.text(j, i, text, va='center', ha='center', color=text_color, fontsize=15, rotation=45)

            if missing_pixel_info is not None:
                for jdx in range(len(missing_pixel_info[idx]['res'])):
                    text = str(rf"{float(missing_pixel_info[idx]['res'][jdx]):.1f}""\n"fr"$\pm$ {float(missing_pixel_info[idx]['err'][jdx]):.1f}")
                    plt.text(int(missing_pixel_info[idx]['col'][jdx]), int(missing_pixel_info[idx]['row'][jdx]), text, va='center', ha='center', color='black', fontsize=15 , rotation=45)

            hep.cms.text(loc=0, ax=ax, text="ETL ETROC Test Beam", fontsize=18)
            ax.set_xlabel('Column', fontsize=25)
            ax.set_ylabel('Row', fontsize=25)
            ticks = range(0, 16)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_title(f"{plot_title} | {fig_tag[idx]}", loc="right", size=16)
            ax.tick_params(axis='x', which='both', length=5, labelsize=18)
            ax.tick_params(axis='y', which='both', length=5, labelsize=18)
            ax.invert_xaxis()
            ax.invert_yaxis()
            plt.minorticks_off()
            plt.tight_layout()

            if save_mother_dir is not None:
                save_dir = save_mother_dir / 'time_resolution_results'
                save_dir.mkdir(exist_ok=True)
                fig.savefig(save_dir / f"resolution_map_{board_names[idx]}.png")
                fig.savefig(save_dir / f"resolution_map_{board_names[idx]}.pdf")
                plt.close(fig)

        del tables

## --------------------------------------
def plot_TDC_correlation_scatter_matrix(
        input_df: pd.DataFrame,
        chip_names: list[str],
        single_hit: bool = False,
        colinear: bool = False,
        colinear_cut: int = 1,
        save: bool = False,
    ):

    import plotly.express as px

    input_df['identifier'] = input_df.groupby(['evt', 'board']).cumcount()
    board_ids = input_df['board'].unique()
    val_names = [f'toa_{board_ids[0]}', f'toa_{board_ids[1]}', f'tot_{board_ids[0]}', f'tot_{board_ids[1]}', f'cal_{board_ids[0]}', f'cal_{board_ids[1]}']
    val_labels = {
        f'toa_{board_ids[0]}':f'TOA_{chip_names[int(board_ids[0])]}',
        f'toa_{board_ids[1]}':f'TOA_{chip_names[int(board_ids[1])]}',
        f'tot_{board_ids[0]}':f'TOT_{chip_names[int(board_ids[0])]}',
        f'tot_{board_ids[1]}':f'TOT_{chip_names[int(board_ids[1])]}',
        f'cal_{board_ids[0]}':f'CAL_{chip_names[int(board_ids[0])]}',
        f'cal_{board_ids[1]}':f'CAL_{chip_names[int(board_ids[1])]}',
    }
    extra_tag = ''

    if single_hit:
        extra_tag = '_singleHit'
        input_df['count'] = (0.5*input_df.groupby('evt')['evt'].transform('count')).astype(int)
        new_df = input_df.pivot(index=['evt', 'identifier'], columns=['board'], values=['row', 'col', 'toa', 'tot', 'cal', 'count'])
        new_df.columns = ['{}_{}'.format(x, y) for x, y in new_df.columns]
        new_df['single_hit'] = (new_df[f'count_{board_ids[0]}'] == 1)
        new_df = new_df.sort_values(by='single_hit', ascending=False) # Make sure True always draw first

        fig = px.scatter_matrix(
            new_df.reset_index(),
            dimensions=val_names,
            color='single_hit',
            labels=val_labels,
            width=1920,
            height=1080,
        )

    elif colinear:
        extra_tag = '_colinear_pixels'
        new_df = input_df.pivot(index=['evt', 'identifier'], columns=['board'], values=['row', 'col', 'toa', 'tot', 'cal'])
        new_df.columns = ['{}_{}'.format(x, y) for x, y in new_df.columns]
        new_df['colinear'] = (abs(new_df[f'row_{board_ids[0]}']-new_df[f'row_{board_ids[1]}']) <= colinear_cut) & (abs(new_df[f'col_{board_ids[0]}']-new_df[f'col_{board_ids[1]}']) <= colinear_cut)
        new_df = new_df.sort_values(by='colinear', ascending=False) # Make sure True always draw first

        fig = px.scatter_matrix(
            new_df.reset_index(),
            dimensions=val_names,
            color='colinear',
            labels=val_labels,
            width=1920,
            height=1080,
        )

    else:
        new_df = input_df.pivot(index=['evt', 'identifier'], columns=['board'], values=['row', 'col', 'toa', 'tot', 'cal'])
        new_df.columns = ['{}_{}'.format(x, y) for x, y in new_df.columns]
        fig = px.scatter_matrix(
            new_df,
            dimensions=val_names,
            labels=val_labels,
            width=1920,
            height=1080,
        )

    fig.update_traces(
        diagonal_visible=False,
        showupperhalf=False,
        marker = {'size': 3},
    )

    for k in range(len(fig.data)):
        fig.data[k].update(
            selected = dict(
                marker = dict(
                )
            ),
            unselected = dict(
                marker = dict(
                    color="grey"
                )
            ),
        )

    if save:
        fig.write_html(
            'scatter_matrix_{}_vs_{}{}.html'.format(chip_names[board_ids[0]], chip_names[board_ids[1]], extra_tag),
            full_html = False,
            include_plotlyjs = 'cdn',
        )
    else:
        fig.show()

## --------------- Plotting -----------------------



## --------------- Time Walk Correction -----------------------
## --------------------------------------
def poly2D(max_order, x, y, *args):
    if max_order < 0:
        raise RuntimeError("The polynomial order must be non-negative")

    ret_val = None

    linear_idx = 0
    for i in range(max_order+1):
        for j in range(max_order - i + 1):
            this_val = args[linear_idx] * x**j * y**i
            linear_idx += 1

            if ret_val is None:
                ret_val = this_val
            else:
                ret_val += this_val

    return ret_val

## --------------------------------------
def poly3D(max_order, x, y, z, *args):
    if max_order < 0:
        raise RuntimeError("The polynomial order must be non-negative")

    ret_val = None

    linear_idx = 0
    for i in range(max_order+1):
        for j in range(max_order - i + 1):
            for k in range(max_order - i - j + 1):
                this_val = args[linear_idx] * x**k * y**j + z**i
                linear_idx += 1

                if ret_val is None:
                    ret_val = this_val
                else:
                    ret_val += this_val

    return ret_val

## --------------------------------------
def three_board_iterative_timewalk_correction(
    input_df: pd.DataFrame,
    iterative_cnt: int,
    poly_order: int,
    board_ids: list,
):

    corr_toas = {}
    corr_b0 = input_df[f'toa_b{board_ids[0]}'].values
    corr_b1 = input_df[f'toa_b{board_ids[1]}'].values
    corr_b2 = input_df[f'toa_b{board_ids[2]}'].values

    del_toa_b0 = (0.5*(input_df[f'toa_b{board_ids[1]}'] + input_df[f'toa_b{board_ids[2]}']) - input_df[f'toa_b{board_ids[0]}']).values
    del_toa_b1 = (0.5*(input_df[f'toa_b{board_ids[0]}'] + input_df[f'toa_b{board_ids[2]}']) - input_df[f'toa_b{board_ids[1]}']).values
    del_toa_b2 = (0.5*(input_df[f'toa_b{board_ids[0]}'] + input_df[f'toa_b{board_ids[1]}']) - input_df[f'toa_b{board_ids[2]}']).values

    for i in range(iterative_cnt):
        coeff_b0 = np.polyfit(input_df[f'tot_b{board_ids[0]}'].values, del_toa_b0, poly_order)
        poly_func_b0 = np.poly1d(coeff_b0)

        coeff_b1 = np.polyfit(input_df[f'tot_b{board_ids[1]}'].values, del_toa_b1, poly_order)
        poly_func_b1 = np.poly1d(coeff_b1)

        coeff_b2 = np.polyfit(input_df[f'tot_b{board_ids[2]}'].values, del_toa_b2, poly_order)
        poly_func_b2 = np.poly1d(coeff_b2)

        corr_b0 = corr_b0 + poly_func_b0(input_df[f'tot_b{board_ids[0]}'].values)
        corr_b1 = corr_b1 + poly_func_b1(input_df[f'tot_b{board_ids[1]}'].values)
        corr_b2 = corr_b2 + poly_func_b2(input_df[f'tot_b{board_ids[2]}'].values)

        del_toa_b0 = (0.5*(corr_b1 + corr_b2) - corr_b0)
        del_toa_b1 = (0.5*(corr_b0 + corr_b2) - corr_b1)
        del_toa_b2 = (0.5*(corr_b0 + corr_b1) - corr_b2)

        if i == iterative_cnt-1:
            corr_toas[f'toa_b{board_ids[0]}'] = corr_b0
            corr_toas[f'toa_b{board_ids[1]}'] = corr_b1
            corr_toas[f'toa_b{board_ids[2]}'] = corr_b2

    return corr_toas

## --------------------------------------
def four_board_iterative_timewalk_correction(
    input_df: pd.DataFrame,
    iterative_cnt: int,
    poly_order: int,
    ):

    corr_toas = {}
    corr_b0 = input_df['toa_b0'].values
    corr_b1 = input_df['toa_b1'].values
    corr_b2 = input_df['toa_b2'].values
    corr_b3 = input_df['toa_b3'].values

    del_toa_b3 = ((1/3)*(input_df['toa_b0'] + input_df['toa_b1'] + input_df['toa_b2']) - input_df['toa_b3']).values
    del_toa_b2 = ((1/3)*(input_df['toa_b0'] + input_df['toa_b1'] + input_df['toa_b3']) - input_df['toa_b2']).values
    del_toa_b1 = ((1/3)*(input_df['toa_b0'] + input_df['toa_b3'] + input_df['toa_b2']) - input_df['toa_b1']).values
    del_toa_b0 = ((1/3)*(input_df['toa_b3'] + input_df['toa_b1'] + input_df['toa_b2']) - input_df['toa_b0']).values

    for i in range(iterative_cnt):
        coeff_b0 = np.polyfit(input_df['tot_b0'].values, del_toa_b0, poly_order)
        poly_func_b0 = np.poly1d(coeff_b0)

        coeff_b1 = np.polyfit(input_df['tot_b1'].values, del_toa_b1, poly_order)
        poly_func_b1 = np.poly1d(coeff_b1)

        coeff_b2 = np.polyfit(input_df['tot_b2'].values, del_toa_b2, poly_order)
        poly_func_b2 = np.poly1d(coeff_b2)

        coeff_b3 = np.polyfit(input_df['tot_b3'].values, del_toa_b3, poly_order)
        poly_func_b3 = np.poly1d(coeff_b3)

        corr_b0 = corr_b0 + poly_func_b0(input_df['tot_b0'].values)
        corr_b1 = corr_b1 + poly_func_b1(input_df['tot_b1'].values)
        corr_b2 = corr_b2 + poly_func_b2(input_df['tot_b2'].values)
        corr_b3 = corr_b3 + poly_func_b3(input_df['tot_b3'].values)

        del_toa_b3 = ((1/3)*(corr_b0 + corr_b1 + corr_b2) - corr_b3)
        del_toa_b2 = ((1/3)*(corr_b0 + corr_b1 + corr_b3) - corr_b2)
        del_toa_b1 = ((1/3)*(corr_b0 + corr_b3 + corr_b2) - corr_b1)
        del_toa_b0 = ((1/3)*(corr_b3 + corr_b1 + corr_b2) - corr_b0)

        if i == iterative_cnt-1:
            corr_toas[f'toa_b0'] = corr_b0
            corr_toas[f'toa_b1'] = corr_b1
            corr_toas[f'toa_b2'] = corr_b2
            corr_toas[f'toa_b3'] = corr_b3

    return corr_toas

## --------------------------------------
def fwhm_based_on_gaussian_mixture_model(
        input_data: np.array,
        tb_loc: str,
        tag: str,
        hist_bins: int = 30,
        n_components: int = 3,
        show_plot: bool = False,
        show_sub_gaussian: bool = False,
        show_fwhm_guideline: bool = False,
        show_number: bool = False,
        save_mother_dir: Path | None = None,
        fname_tag: str = '',
    ):
    """Find the sigma of delta TOA distribution and plot the distribution.

    Parameters
    ----------
    input_data: np.array,
        A numpy array includes delta TOA values.
    tb_loc: str,
        Test Beam location for the title. Available argument: desy, cern, fnal.
    tag: str,
        Additional string to show which boards are used for delta TOA calculation.
    hist_bins: int
        Bins for histogram.
    n_components: int
        Number of sub-gaussian to be considered for the Gaussian Mixture Model
    show_sub_gaussian: bool, optional
        If it is True, show sub-gaussian in the plot.
    show_fwhm_guideline: bool, optional
        If it is True, show horizontal and vertical lines to show how FWHM has been performed.
    show_number: bool, optional
        If it is True, FWHM and sigma will be shown in the plot.
    save_mother_dir: Path, optional
        Plot will be saved at save_mother_dir/'fwhm'.
    fname_tag: str, optional
        Additional tag for the file name.
    """

    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
    from scipy.spatial import distance

    plot_title = load_fig_title(tb_loc)

    x_range = np.linspace(input_data.min(), input_data.max(), 1000).reshape(-1, 1)
    bins, edges = np.histogram(input_data, bins=hist_bins, density=True)
    centers = 0.5*(edges[1:] + edges[:-1])
    models = GaussianMixture(n_components=n_components).fit(input_data.reshape(-1, 1))

    silhouette_eval_score = silhouette_score(centers.reshape(-1, 1), models.predict(centers.reshape(-1, 1)))

    logprob = models.score_samples(centers.reshape(-1, 1))
    pdf = np.exp(logprob)
    jensenshannon_score = distance.jensenshannon(bins, pdf)

    logprob = models.score_samples(x_range)
    pdf = np.exp(logprob)
    peak_height = np.max(pdf)

    # Find the half-maximum points.
    half_max = peak_height*0.5
    half_max_indices = np.where(pdf >= half_max)[0]

    # Calculate the FWHM.
    fwhm = x_range[half_max_indices[-1]] - x_range[half_max_indices[0]]

    xval = x_range[np.argmax(pdf)][0]

    if show_sub_gaussian:
        # Compute PDF for each component
        responsibilities = models.predict_proba(x_range)
        pdf_individual = responsibilities * pdf[:, np.newaxis]

    if show_plot:
        fig, ax = plt.subplots(figsize=(11,10))

        # Plot data histogram
        bins, _, _ = ax.hist(input_data, bins=hist_bins, density=True, histtype='stepfilled', alpha=0.4, label='Data')

        # Plot PDF of whole model
        hep.cms.text(loc=0, ax=ax, text="ETL ETROC Test Beam", fontsize=18)
        ax.set_title(plot_title, loc="right", fontsize=16)
        ax.set_xlabel(rf'$\Delta \mathrm{{TOA}}_{{{tag}}}$ [ps]', fontsize=25)
        ax.yaxis.label.set_fontsize(25)
        if show_number:
            ax.plot(x_range, pdf, '-k', label=f'Mixture PDF, mean: {xval:.2f}')
            ax.plot(np.nan, np.nan, linestyle='none', label=f'FWHM:{fwhm[0]:.2f}, sigma:{fwhm[0]/2.355:.2f}')
        else:
            ax.plot(x_range, pdf, '-k', label=f'Mixture PDF')

        if show_sub_gaussian:
            # Plot PDF of each component
            ax.plot(x_range, pdf_individual, '--', label='Component PDF')

        if show_fwhm_guideline:
            ax.vlines(x_range[half_max_indices[0]],  ymin=0, ymax=np.max(bins)*0.75, lw=1.5, colors='red')
            ax.vlines(x_range[half_max_indices[-1]], ymin=0, ymax=np.max(bins)*0.75, lw=1.5, colors='red')
            ax.hlines(y=peak_height, xmin=x_range[0], xmax=x_range[-1], lw=1.5, colors='crimson', label='Max')
            ax.hlines(y=half_max, xmin=x_range[0], xmax=x_range[-1], lw=1.5, colors='deeppink', label='Half Max')

        ax.legend(loc='best', fontsize=14)
        plt.tight_layout()

        if save_mother_dir is not None:
            save_dir = save_mother_dir / 'fwhm'
            save_dir.mkdir(exist_ok=True)
            fig.savefig(save_dir / f"fwhm_{tag}_{fname_tag}.png")
            fig.savefig(save_dir / f"fwhm_{tag}_{fname_tag}.pdf")
            plt.close(fig)

    return fwhm, [silhouette_eval_score, jensenshannon_score]

## --------------- Time Walk Correction -----------------------



## --------------- Result -----------------------
## --------------------------------------
def return_resolution_three_board(
        fit_params: dict,
        var: list,
        board_ids:list,
    ):

    results = {
        board_ids[0]: np.sqrt(0.5*(fit_params[var[0]][0]**2 + fit_params[var[1]][0]**2 - fit_params[var[2]][0]**2))*1e3,
        board_ids[1]: np.sqrt(0.5*(fit_params[var[0]][0]**2 + fit_params[var[2]][0]**2 - fit_params[var[1]][0]**2))*1e3,
        board_ids[2]: np.sqrt(0.5*(fit_params[var[1]][0]**2 + fit_params[var[2]][0]**2 - fit_params[var[0]][0]**2))*1e3,
    }

    return results

## --------------------------------------
def return_resolution_three_board_fromFWHM(
        fit_params: dict,
        var: list,
        board_ids:list,
    ):

    results = {
        board_ids[0]: np.sqrt(0.5*(fit_params[var[0]]**2 + fit_params[var[1]]**2 - fit_params[var[2]]**2)),
        board_ids[1]: np.sqrt(0.5*(fit_params[var[0]]**2 + fit_params[var[2]]**2 - fit_params[var[1]]**2)),
        board_ids[2]: np.sqrt(0.5*(fit_params[var[1]]**2 + fit_params[var[2]]**2 - fit_params[var[0]]**2)),
    }

    return results

## --------------------------------------
def return_resolution_four_board(
        fit_params: dict,
    ):

    results = {
        0: np.sqrt((1/6)*(2*fit_params['01'][0]**2+2*fit_params['02'][0]**2+2*fit_params['03'][0]**2-fit_params['12'][0]**2-fit_params['13'][0]**2-fit_params['23'][0]**2))*1e3,
        1: np.sqrt((1/6)*(2*fit_params['01'][0]**2+2*fit_params['12'][0]**2+2*fit_params['13'][0]**2-fit_params['02'][0]**2-fit_params['03'][0]**2-fit_params['23'][0]**2))*1e3,
        2: np.sqrt((1/6)*(2*fit_params['02'][0]**2+2*fit_params['12'][0]**2+2*fit_params['23'][0]**2-fit_params['01'][0]**2-fit_params['03'][0]**2-fit_params['13'][0]**2))*1e3,
        3: np.sqrt((1/6)*(2*fit_params['03'][0]**2+2*fit_params['13'][0]**2+2*fit_params['23'][0]**2-fit_params['01'][0]**2-fit_params['02'][0]**2-fit_params['12'][0]**2))*1e3,
    }

    return results

## --------------------------------------
def return_resolution_four_board_fromFWHM(
        fit_params: dict,
    ):

    results = {
        0: np.sqrt((1/6)*(2*fit_params['01']**2+2*fit_params['02']**2+2*fit_params['03']**2-fit_params['12']**2-fit_params['13']**2-fit_params['23']**2)),
        1: np.sqrt((1/6)*(2*fit_params['01']**2+2*fit_params['12']**2+2*fit_params['13']**2-fit_params['02']**2-fit_params['03']**2-fit_params['23']**2)),
        2: np.sqrt((1/6)*(2*fit_params['02']**2+2*fit_params['12']**2+2*fit_params['23']**2-fit_params['01']**2-fit_params['03']**2-fit_params['13']**2)),
        3: np.sqrt((1/6)*(2*fit_params['03']**2+2*fit_params['13']**2+2*fit_params['23']**2-fit_params['01']**2-fit_params['02']**2-fit_params['12']**2)),
    }

    return results

## --------------------------------------
def return_board_resolution(
        input_df: pd.DataFrame,
        board_ids: list[int],
        key_names: list[str],
        hist_bins: int = 15,
    ):

    from collections import defaultdict
    from lmfit.models import GaussianModel
    mod = GaussianModel(nan_policy='omit')

    results = defaultdict(float)

    for key in range(len(board_ids)):
        hist_x_min = int(input_df[f'res{board_ids[key]}'].min())-5
        hist_x_max = int(input_df[f'res{board_ids[key]}'].max())+5
        h_temp = hist.Hist(hist.axis.Regular(hist_bins, hist_x_min, hist_x_max, name="time_resolution", label=r'Time Resolution [ps]'))
        h_temp.fill(input_df[f'res{board_ids[key]}'].values)
        mean = np.mean(input_df[f'res{board_ids[key]}'].values)
        std = np.std(input_df[f'res{board_ids[key]}'].values)
        centers = h_temp.axes[0].centers
        fit_range = centers[np.argmax(h_temp.values())-5:np.argmax(h_temp.values())+5]
        fit_vals = h_temp.values()[np.argmax(h_temp.values())-5:np.argmax(h_temp.values())+5]

        pars = mod.guess(fit_vals, x=fit_range)
        out = mod.fit(fit_vals, pars, x=fit_range, weights=1/np.sqrt(fit_vals))

        results[f'{key_names[key]}_mean'] = mean
        results[f'{key_names[key]}_std'] = std
        results[f'{key_names[key]}_res'] = out.params['center'].value
        results[f'{key_names[key]}_err'] = abs(out.params['sigma'].value)

    return results

## --------------- Result -----------------------

