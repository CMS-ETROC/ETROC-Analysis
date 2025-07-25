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
def build_events_to_dataframe_dict(unpacked_data):
    """
    Builds events from an iterable of unpacked data and returns a single
    dictionary in a format ready for direct conversion to a Pandas DataFrame.
    This version includes the l1a_counter from the header and uses a robust
    two-pass approach to ensure the event counter is correct and deterministic.

    Args:
        unpacked_data (iterable): An iterable that yields tuples of
                                  (record_type, record_data).

    Returns:
        dict: A dictionary of lists (e.g., {'event': [...], 'bcid': [...], ...})
              that can be passed directly to pd.DataFrame().
    """
    pending_packets = {}

    # --- Pass 1: Group all hits and l1a_counter by bcid, preserving order ---
    # The dictionary will now store the l1a_counter and a list of hits.
    events_by_bcid = {}
    # This list will store bcids in the order they are first completed.
    ordered_bcids = []

    for record_type, record_data in unpacked_data:
        if not record_type or not record_data:
            continue

        elink = record_data.get('elink')
        if elink is None:
            continue

        if record_type == 'header':
            if elink in pending_packets:
                pass # Discard previous incomplete packet for this elink.
            pending_packets[elink] = {'header': record_data, 'data': []}

        elif record_type == 'data':
            if elink in pending_packets:
                pending_packets[elink]['data'].append(record_data)

        elif record_type == 'trailer':
            if elink in pending_packets and pending_packets[elink]['data']:
                packet = pending_packets.pop(elink)
                bcid = packet['header'].get('bcid')
                l1counter = packet['header'].get('l1counter')

                if bcid is not None:
                    # If this is the first time we've seen this bcid,
                    # add it to our ordered list and create its event structure.
                    if bcid not in events_by_bcid:
                        events_by_bcid[bcid] = {
                            'l1a_counter': l1counter, # Store the l1a_counter
                            'hits': []
                        }
                        ordered_bcids.append(bcid)

                    # Add all data hits from the completed packet to the event.
                    events_by_bcid[bcid]['hits'].extend(packet['data'])

            elif elink in pending_packets:
                # Discard packet that has a header but no data.
                del pending_packets[elink]

    # --- Pass 2: Build the final dictionary for the DataFrame ---
    all_hits_data = {
        'evt': [], 'bcid': [], 'l1a_counter': [], 'ea': [], 'row': [], 'col': [],
        'toa': [], 'tot': [], 'cal': [], 'elink': []
    }

    # Create a mapping from bcid to its final, sequential event number.
    bcid_to_event_id = {bcid: i for i, bcid in enumerate(ordered_bcids)}

    # Iterate through the bcids in the order they appeared.
    for bcid in ordered_bcids:
        event_id = bcid_to_event_id[bcid]
        event_data = events_by_bcid[bcid]
        l1a_counter_for_event = event_data['l1a_counter']

        # Add all the hits for this event to the final dictionary.
        for data_hit in event_data['hits']:
            all_hits_data['evt'].append(event_id)
            all_hits_data['bcid'].append(bcid)
            all_hits_data['l1a_counter'].append(l1a_counter_for_event) # Add the event's l1a_counter
            all_hits_data['ea'].append(data_hit.get('ea'))
            all_hits_data['row'].append(data_hit.get('row_id'))
            all_hits_data['col'].append(data_hit.get('col_id'))
            all_hits_data['toa'].append(data_hit.get('toa'))
            all_hits_data['tot'].append(data_hit.get('tot'))
            all_hits_data['cal'].append(data_hit.get('cal'))
            all_hits_data['elink'].append(data_hit.get('elink'))

    return all_hits_data

## --------------------------------------
def merge_words(res):
    empty_frame_mask = np.array(res[0::2]) > (2**8)
    len_cut = min(len(res[0::2]), len(res[1::2]))
    if len(res) > 0:
        return list(np.array(res[0::2])[:len_cut][empty_frame_mask[:len_cut]] | (np.array(res[1::2]) << 32)[:len_cut][empty_frame_mask[:len_cut]])
    else:
        return []

## --------------------------------------
def process_tamalero_outputs(input_files: list):

    list_of_dfs = []

    for ifile in tqdm(input_files):

        with open(ifile, 'rb') as f:
            bin_data = f.read()
            raw_data = struct.unpack(f'<{int(len(bin_data)/4)}I', bin_data)

        merged_data = merge_words(raw_data)

        df_decoder = TamaleroDF()
        unpacked_data = [df_decoder.read(x) for x in merged_data]
        events = build_events_to_dataframe_dict(unpacked_data)

        df = pd.DataFrame(events)

        board_map = {
            0: 0,
            4: 1,
            8: 2,
            12: 3
        }

        df['board'] = df['elink'].map(board_map)
        df.drop(columns=['elink'], inplace=True)
        list_of_dfs.append(df)

    final_df = pd.concat(list_of_dfs)
    is_new_event = final_df['evt'] != final_df['evt'].shift()

    # Assign a unique sequential number to each event
    final_df['evt'] = is_new_event.cumsum() - 1
    final_df.reset_index(drop=True, inplace=True)

    # # Set data type
    dtype_map = {
        'evt': np.uint32,
        'bcid': np.uint16,
        'l1a_counter': np.uint8,
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