import pandas as pd
import numpy as np
import struct
from tqdm import tqdm

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

                l1a_for_event = packet['header'].get('l1a_counter')

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

## --------------------------------------
if __name__ == "__main__":

    from natsort import natsorted
    from pathlib import Path

    import argparse

    parser = argparse.ArgumentParser(
                prog='convert',
                description='converting tamalero output to feather',
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

    files = natsorted(Path(args.input_dir).glob('file*dat'))

    if len(files) == 0:
        print('Input files not found')
        exit()

    df = process_tamalero_outputs(files)

    if not df.empty:
        df.to_feather(f'{args.output_name}.feather')
    else:
        print('No data is recorded!')
