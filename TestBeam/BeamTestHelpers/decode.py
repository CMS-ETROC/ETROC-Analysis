from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import json
from crc import Calculator, Configuration

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
        self.check_link_filler_pattern_2 = 0x553
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
                    elif (word >> 20) == self.check_link_filler_pattern or (word >> 20) == self.check_link_filler_pattern_2:
                        self.filler_data['idx'].append(self.filler_40_idx)
                        if (word >> 20) == self.check_link_filler_pattern:
                            self.filler_data['type'].append("40")
                        elif (word >> 20) == self.check_link_filler_pattern_2:
                            self.filler_data['type'].append("40_2")
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