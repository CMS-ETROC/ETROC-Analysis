from pathlib import Path
from crc import Calculator, Configuration
from tqdm import tqdm
from natsort import natsorted

import numpy as np
import pandas as pd
import json
import struct
import argparse

## --------------- Tamalero Decoding Class -----------------------
## --------------------------------------
format_dict = {
    'nbits': 40,
    'bitorder': 'reversed',
    'identifiers': {
        'header': {
            'frame': 0x3C5C000000,
            'mask': 0xFFFFC00000,
        },
        'data': {
          'frame': 0x8000000000,
          'mask': 0x8000000000,
        },
        'filler': {
          'frame': 0x3C5C800000,
          'mask': 0xFFFFC00000,
        },
        'trailer': {
          'frame': 0x0000000000,
          'mask': 0x8000000000,
        },
    },
    'types': ['ea', 'col_id', 'row_id', 'toa', 'cal', 'tot', 'elink', 'full', 'any_full', 'global_full'],
    'data': {
        'header': {
            'elink': {
                'mask': 0xFF0000000000,
                'shift': 40,
            },
            'sof': {
                'mask': 0x1000000000000,
                'shift': 48,
            },
            'eof': {
                'mask': 0x2000000000000,
                'shift': 49,
            },
            'full': {
                'mask': 0x4000000000000,
                'shift': 50,
            },
            'any_full': {
                'mask': 0x8000000000000,
                'shift': 51,
            },
            'global_full': {
                'mask': 0xF0000000000000,
                'shift': 52,
            },
            'l1counter': {
                'mask': 0x00003FC000,
                'shift': 14,
            },
            'type': {
                'mask': 0x0000003000,
                'shift': 12,
            },
            'bcid': {
                'mask': 0x0000000FFF,
                'shift': 0,
            },
        },
        'data': {
            'elink': {
                'mask': 0xFF0000000000,
                'shift': 40
            },
            'sof': {
                'mask': 0x1000000000000,
                'shift': 48
            },
            'eof': {
                'mask': 0x2000000000000,
                'shift': 49
            },
            'full': {
                'mask': 0x4000000000000,
                'shift': 50
            },
            'any_full': {
                'mask': 0x8000000000000,
                'shift': 51
            },
            'global_full': {
                'mask': 0xF0000000000000,
                'shift': 52
            },
            'ea': {
                'mask': 0x6000000000,
                'shift': 37
            },
            'col_id': {
                'mask': 0x1E00000000,
                'shift': 33
            },
            'row_id': {
                'mask': 0x01E0000000,
                'shift': 29
            },
            # random test pattern specific
            'col_id2': {
                'mask': 0x001E000000,
                'shift': 25
            },
            'row_id2': {
                'mask': 0x0001E00000,
                'shift': 21
            },
            'bcid': {
                'mask': 0x00001FFE00,
                'shift': 9
            },
            'counter_a': {
                'mask': 0x00000001FF,
                'shift': 0
            },
            'counter_b': {
                'mask': 0x1FFE00,
                'shift': 9
            },
            'test_pattern': {
                'mask': 0xff,  # should be 9 bits according to manual, but bit 9 is (almost) always 1
                'shift': 0
            },
            # generic portion of the data
            'data': {
                'mask': 0x001FFFFFFF,
                'shift': 0
            },
            'toa': {
                'mask': 0x1ff80000,
                'shift': 19
            },
            'tot': {
                'mask': 0x0007fc00,
                'shift': 10
            },
            'cal': {
                'mask': 0x000003ff,
                'shift': 0
            }
        },
        'trailer': {
            'elink': {
                'mask': 0xFF0000000000,
                'shift': 40
            },
            'sof': {
                'mask': 0x1000000000000,
                'shift': 48
            },
            'eof': {
                'mask': 0x2000000000000,
                'shift': 49
            },
            'full': {
                'mask': 0x4000000000000,
                'shift': 50
            },
            'any_full': {
                'mask': 0x8000000000000,
                'shift': 51
            },
            'global_full': {
                'mask': 0xF0000000000000,
                'shift': 52
            },
            'chipid': {
                'mask': 0x7FFFC00000,
                'shift': 24
            },
            'status': {
                'mask': 0x00003F0000,
                'shift': 16
            },
            'hits': {
                'mask': 0x000000FF00,
                'shift': 8
            },
            'crc': {
                'mask': 0x00000000FF,
                'shift': 0
            }
        },
        'filler': {
            'elink': {
                'mask': 0xFF0000000000,
                'shift': 40
            },
            'sof': {
                'mask': 0x1000000000000,
                'shift': 48
            },
            'eof': {
                'mask': 0x2000000000000,
                'shift': 49
            },
            'full': {
                'mask': 0x4000000000000,
                'shift': 50
            },
            'any_full': {
                'mask': 0x8000000000000,
                'shift': 51
            },
            'global_full': {
                'mask': 0xF0000000000000,
                'shift': 52
            },
            'l1counter': {
                'mask': 0x00003FC000,
                'shift': 14
            },
            'ebs': {
                'mask': 0x0000003000,
                'shift': 12
            },
            'bcid': {
                'mask': 0x0000000FFF,
                'shift': 0
            }
        }
    }
}

## --------------------------------------
class TamaleroDF:
    def __init__(self):
        self.format = format_dict

    def get_bytes(self, word, format_order):
        output_bytes = []
        if self.format_order['bitorder'] == 'normal':
            shifts = [32, 24, 16, 8, 0]
        elif self.format_order['bitorder'] == 'reversed':
            shifts = [0, 8, 16, 24, 32]
        for shift in shifts:
            output_bytes.append((word >> shift) & 0xFF)
        if format_order:
            return [ '{0:0{1}x}'.format(b,2) for b in output_bytes ]
        else:
            return output_bytes

    def get_trigger_words(self, format=False):
        return \
            self.get_bytes(self.format['identifiers']['header']['frame'], format=format)  # FIXME check that this still works with FW > v1.2.0

    def get_trigger_masks(self, format=False):
        return \
            self.get_bytes(self.format['identifiers']['header']['mask'], format=format)  # FIXME check that this still works with FW > v1.2.0

    def read(self, val, quiet=True):
        data_type = None
        for id in self.format['identifiers']:
            if self.format['identifiers'][id]['frame'] == (val & self.format['identifiers'][id]['mask']):
                data_type = id
                break

        res = {}
        if data_type == None:
            if not quiet:
                print ("Found data of type None:", val)
            return None, res

        if data_type == 'data':
            datatypelist = self.format['types']
        else:
            datatypelist = self.format['data'][data_type]

        for d in datatypelist:
            res[d] = (val & self.format['data'][data_type][d]['mask']) >> self.format['data'][data_type][d]['shift']

        if data_type == 'header':
            self.type = res['type']
        res['raw'] = hex(val&0xFFFFFFFFFF)
        res['raw_full'] = hex(val)
        res['meta'] = hex((val>>40)&0xFFFFFF)

        if not quiet:
            print (f"Found data of type {data_type}:", res)
        return data_type, res

## --------------------------------------
def merge_words(res):
    empty_frame_mask = np.array(res[0::2]) > (2**8)
    len_cut = min(len(res[0::2]), len(res[1::2]))
    if len(res) > 0:
        return list(np.array(res[0::2])[:len_cut][empty_frame_mask[:len_cut]] | (np.array(res[1::2]) << 32)[:len_cut][empty_frame_mask[:len_cut]])
    else:
        return []

## --------------------------------------
# NEW: Replaces the old event builder with a sequential, stateful approach.
def build_events_sequentially(unpacked_data):
    """
    Builds events by processing the data stream sequentially. An event is defined
    by a consistent BCID. The event number increments only when the BCID of a new,
    complete packet changes from the previous one.

    Args:
        unpacked_data (iterable): An iterable that yields tuples of
                                  (record_type, record_data).

    Returns:
        dict: A dictionary of lists ready for conversion to a Pandas DataFrame.
    """
    df_data = {
        'evt': [], 'bcid': [], 'l1a_counter': [], 'ea': [], 'row': [], 'col': [],
        'toa': [], 'tot': [], 'cal': [], 'elink': []
    }

    pending_packets = {}
    # State variables for the new event counting logic
    event_counter = -1
    current_bcid = -1  # Sentinel value, assuming no real bcid is -1

    for record_type, record_data in unpacked_data:
        if not record_type or not record_data:
            continue

        elink = record_data.get('elink')
        if elink is None:
            continue

        if record_type == 'header':
            # Start a new packet for this elink
            pending_packets[elink] = {'header': record_data, 'data': []}

        elif record_type == 'data':
            # Add data to an existing packet
            if elink in pending_packets:
                pending_packets[elink]['data'].append(record_data)

        elif record_type == 'trailer':
            # A packet is complete if it's in pending_packets and has data
            if elink in pending_packets and len(pending_packets[elink]['data']) > 0:
                packet = pending_packets.pop(elink)
                packet_bcid = packet['header'].get('bcid')

                if packet_bcid is None:
                    continue # Ignore packets with no bcid

                # ---- CORE EVENT COUNTING LOGIC ----
                # If the bcid of this valid packet is different from the
                # bcid of the last valid packet, increment the event counter.
                if packet_bcid != current_bcid:
                    event_counter += 1
                    current_bcid = packet_bcid
                # ------------------------------------

                l1a_for_event = packet['header'].get('l1counter')

                # Add all hits from this completed packet to the final dictionary
                for data_hit in packet['data']:
                    df_data['evt'].append(event_counter)
                    df_data['bcid'].append(current_bcid)
                    df_data['l1a_counter'].append(l1a_for_event)
                    df_data['ea'].append(data_hit.get('ea'))
                    df_data['row'].append(data_hit.get('row_id'))
                    df_data['col'].append(data_hit.get('col_id'))
                    df_data['toa'].append(data_hit.get('toa'))
                    df_data['tot'].append(data_hit.get('tot'))
                    df_data['cal'].append(data_hit.get('cal'))
                    df_data['elink'].append(data_hit.get('elink'))

            elif elink in pending_packets:
                # Discard packet that has a header but no data.
                del pending_packets[elink]

    return df_data

## --------------------------------------
# UPDATED: The main processing function is now simpler and more robust.
def process_tamalero_outputs(input_files: list):

    all_merged_data = []
    print("Reading and merging data from input files...")
    for ifile in tqdm(input_files):
        with open(ifile, 'rb') as f:
            bin_data = f.read()
            raw_data = struct.unpack(f'<{int(len(bin_data)/4)}I', bin_data)

        # Merge data and add to a single list for unified processing
        all_merged_data.extend(merge_words(raw_data))

    print("Decoding data stream...")
    df_decoder = TamaleroDF()
    unpacked_data = [df_decoder.read(x) for x in tqdm(all_merged_data)]

    print("Building events sequentially...")
    events = build_events_sequentially(unpacked_data)

    final_df = pd.DataFrame(events)

    # The board mapping and final type casting remain useful
    board_map = {
        0: 0,
        4: 1,
        8: 2,
        12: 3
    }
    if not final_df.empty:
        final_df['board'] = final_df['elink'].map(board_map)
        final_df.drop(columns=['elink'], inplace=True)

        # Set data types for memory efficiency
        dtype_map = {
            'evt': np.uint32,
            'bcid': np.uint16,
            'l1a_counter': np.uint16,
            'ea': np.uint8,
            'board': np.uint8,
            'row': np.int8,
            'col': np.int8,
            'toa': np.uint16,
            'tot': np.uint16,
            'cal': np.uint16,
        }
        final_df = final_df.astype(dtype_map)

    return final_df




## --------------- Decoding Class -----------------------
## --------------------------------------
class DecodeBinary:
    def copy_dict_by_json(self, d):
        return json.loads(json.dumps(d))

    def __init__(self, firmware_key,
                 board_id: list[int],
                 file_list: list[Path],
                 save_nem: Path = None,
                 skip_fw_filler: bool = False,
                 skip_event_df: bool = False,
                 skip_crc_df: bool = False,
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

        if not self.skip_crc_df:
            config = Configuration(
                width=8,
                polynomial=0x2F, # Normal representation
                #polynomial=0x97, # Reversed reciprocal representation (the library uses normal representation, so do not use this)
                init_value=0x00,
                final_xor_value=0x00,
                reverse_input=False,
                reverse_output=False,
            )

        if not self.skip_crc_df:
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
            'prev_event': np.array(tmp['prev_event'], dtype=np.int32),
            'last_event': np.array(tmp['last_event'], dtype=np.int32),
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

            if not self.skip_crc_df:
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

            if not self.skip_crc_df:
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

            if not self.skip_crc_df:
                self.CRCdata_40bit += [
                    (word >> 32) & 0xff,
                    (word >> 24) & 0xff,
                    (word >> 16) & 0xff,
                    (word >> 8) & 0xff,
                    #(word ) & 0xff,
                    ]
                data = bytes(self.CRCdata_40bit)
                check = self.CRCcalculator.checksum(data)

            if not self.skip_crc_df:
                self.crc_data[self.current_channel]['bcid'].append(self.bcid)
                self.crc_data[self.current_channel]['l1a_counter'].append(self.l1acounter)
                self.crc_data[self.current_channel]['evt'].append(self.event_counter)
                self.crc_data[self.current_channel]['board'].append(self.current_channel)
                self.crc_data[self.current_channel]['CRC'].append(CRC)
                self.crc_data[self.current_channel]['CRC_calc'].append(check)
                self.crc_data[self.current_channel]['CRC_mismatch'].append(bool(CRC != check))

            if self.nem_file is not None:
                mismatch = ""
                if CRC != check:
                    mismatch = " CRC Mismatch"
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
                        if not self.skip_crc_df:
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
                        if not self.skip_crc_df:
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
                        continue

                    # Event Trailer NOT Found after the required number of ethernet words was read
                    elif(self.in_event and (self.eth_words_in_event==self.current_word) and (word >> 26 != self.trailer_pattern)):
                        # print('Event Trailer NOT Found after the required number of ethernet words was read')
                        self.reset_params()
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

                        if not self.skip_crc_df:
                            self.CRCdata += [
                                (word >> 24) & 0xff,
                                (word >> 16) & 0xff,
                                (word >> 8) & 0xff,
                            ]

                            data = bytes(self.CRCdata)
                            check = self.CRCcalculator.checksum(data)

                        crc            = (word) & 0xff
                        overflow_count = (word >> 11) & 0x7
                        hamming_count  = (word >> 8) & 0x7

                        if not self.skip_event_df:
                            self.event_data['evt'].append(self.event_counter)
                            self.event_data['bcid'].append(self.bcid)
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
                            mismatch = ""
                            if crc != check:
                                mismatch = " CRC Mismatch"
                            self.write_to_nem(f"ET {self.event_number} {overflow_count} {hamming_count} 0b{crc:08b}{mismatch}\n")

                    # Event Data Word
                    elif(self.in_event):
                        # print(self.current_word)
                        # print(format(word, '032b'))

                        if not self.skip_crc_df:
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

## --------------------------------------
if __name__ == "__main__":


    parser = argparse.ArgumentParser(
                prog='convert',
                description='converting binary to feather',
            )

    parser.add_argument(
        '-d',
        '--input_dir',
        metavar = 'NAME',
        type = str,
        help = 'input directory containing .bin',
        required = True,
        dest = 'input_dir',
    )

    parser.add_argument(
        '-o',
        '--output_name',
        metavar = 'NAME',
        help = 'Name for output file',
        default = None,
        dest = 'output_name',
    )

    args = parser.parse_args()

    constellation_files = natsorted(Path(args.input_dir).glob('file*bin'))
    ce_files = natsorted(Path(args.input_dir).glob('file*dat'))

    if len(constellation_files) != 0:

        files = constellation_files

        decoder = DecodeBinary(
            firmware_key = 0b0001,
            board_id = [0x17f0f, 0x17f0f, 0x17f0f, 0x17f0f],
            file_list = files,
            save_nem = None,
            skip_fw_filler = True,
            skip_event_df = True,
            skip_crc_df = True,
        )
        df, _, _, filler_df = decoder.decode_files()

        name = Path(args.input_dir).name
        if args.output_name is not None:
            name = args.output_name

        if not df.empty:
            df.to_feather(f'{name}.feather')
        else:
            print('No data is recorded!')

        if not filler_df.empty:
            filler_df.to_feather(f'filler_{name}.feather')
        else:
            print('No filler information is recorded!')

    elif len(ce_files) != 0:

        files = ce_files

        if len(files) == 0:
            print('Input files not found')
            exit()

        df = process_tamalero_outputs(files)

        if not df.empty:
            df.to_feather(f'{args.output_name}.feather')
        else:
            print('No data is recorded!')
