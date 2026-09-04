#!/usr/bin/env python3
"""Independent, from-scratch re-decode of one TDC point directory, written only from the
ETROC2 / KCU105 word-format description -- it does NOT import DecodeBinary. Used to check
the production reduction (reduce_tid.py, which does use DecodeBinary) on the same bytes.

Format (established by inspection of the byte stream, see DECODE_README.md):
  * the .bin is a little-endian stream of 32-bit FPGA words
  * event header  : (w >> 4) == 0xc3a3c3a ; w & 0xf = enabled-channel mask
  * firmware word : (w >> 28) == firmware_key(=1) ; n40 = (w >> 2) & 0x3ff  (40-bit words in event)
  * payload       : ceil(40*n40/32) words; their bytes, taken big-endian per 32-bit word and
                    concatenated, form a byte stream that is cut into 5-byte (40-bit) ETROC words
  * event trailer : (w >> 26) == 0b001011
  * ETROC 40-bit words:  header  (w >> 22) == 0x3c5c0 >> 2 ; bcid = w & 0xfff
                         data    (w >> 39) == 1 ; col=(w>>33)&0xf row=(w>>29)&0xf
                                  toa=(w>>19)&0x3ff tot=(w>>10)&0x1ff cal=w&0x3ff
"""
import sys, statistics, glob, os

FW_KEY = 1
EVT_HDR = 0xc3a3c3a
CH_HDR = 0x3c5c0 >> 2
TRAILER = 0b001011
BOARD_ID = 0x17f0f


def decode(path):
    b = open(path, 'rb').read()
    n = len(b) // 4
    words = [int.from_bytes(b[4 * i:4 * i + 4], 'little') for i in range(n)]
    hits = []
    i = 0
    n_events = 0
    while i < n:
        w = words[i]
        if (w >> 4) != EVT_HDR:
            i += 1
            continue
        if i + 1 >= n:
            break
        fw = words[i + 1]
        if (fw >> 28) != FW_KEY:
            i += 1
            continue
        n40 = (fw >> 2) & 0x3ff
        neth = -(-(40 * n40) // 32)
        payload = words[i + 2:i + 2 + neth]
        if len(payload) < neth:
            break
        stream = b''.join(x.to_bytes(4, 'big') for x in payload)
        in40 = False
        for k in range(n40):
            chunk = stream[5 * k:5 * k + 5]
            if len(chunk) < 5:
                break
            W = int.from_bytes(chunk, 'big')
            if (not in40) and (W >> 22) == CH_HDR:
                in40 = True                      # ETROC channel header
            elif in40 and (W >> 39) == 1:
                hits.append(((W >> 29) & 0xf, (W >> 33) & 0xf,
                             (W >> 19) & 0x3ff, (W >> 10) & 0x1ff, W & 0x3ff))
            elif in40 and ((W >> 22) & 0x3ffff) == BOARD_ID:
                in40 = False                     # ETROC trailer closes the channel
        n_events += 1
        i += 2 + neth + 1   # +1 for the event trailer
    return hits, n_events


def main():
    d = sys.argv[1]
    files = sorted(glob.glob(os.path.join(d, 'TDC_Data_*.bin')),
                   key=lambda p: int(p.rsplit('_', 1)[1].split('.')[0]))
    all_hits, nev = [], 0
    for f in files:
        h, e = decode(f)
        all_hits += h
        nev += e
    print(f'dir      : {os.path.basename(d)}')
    print(f'files    : {len(files)}   events: {nev}   hits: {len(all_hits)}')
    if not all_hits:
        return
    rows = sorted(set((h[0], h[1]) for h in all_hits))
    print(f'pixels   : {rows}')
    for name, idx in (('toa', 2), ('tot', 3), ('cal', 4)):
        v = [h[idx] for h in all_hits]
        print(f'{name}_mean = {statistics.fmean(v):.10f}   {name}_std(ddof=1) = {statistics.stdev(v):.10f}   '
              f'min={min(v)} max={max(v)}')


if __name__ == '__main__':
    main()
