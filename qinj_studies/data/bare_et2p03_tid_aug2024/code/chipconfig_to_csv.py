#!/usr/bin/env python3
"""Decode the ETROC2 ChipConfig register dumps (*.pckl) shipped beside the
2024-08 X9 run directories into a flat CSV of named register fields.

The pickle is a plain dict {'object': {'ETROC2': [65536 ints], 'Waveform Sampler': [48 ints]},
'chip': 'ETROC2', 'version': '0.0.1'} -- the full I2C address space as read back.
Field names / bit positions come from i2c_gui2.chips.etroc2_chip (copied verbatim).
"""
import pickle, sys, csv, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from etroc2_regmap import register_model, register_decoding

BLOCKS = ['Peripheral Config', 'Peripheral Status']


def reg_addr_map():
    m = {}
    for blk, spec in register_model['ETROC2']['Register Blocks'].items():
        if blk not in BLOCKS:
            continue
        base = spec['Base Address']
        for name, r in spec['Registers'].items():
            m[name] = base + r['offset']
    return m


def bits_of(rng):
    if '-' in rng:
        a, b = rng.split('-')
        return list(range(int(a), int(b) - 1, -1))
    return [int(rng)]


def decode(space, addrs, blk):
    out = {}
    for field, spec in register_decoding['ETROC2']['Register Blocks'][blk].items():
        val = 0
        ok = True
        for pos in spec['position']:
            reg, src, dst = pos[0], pos[1], pos[2]
            if reg not in addrs:
                ok = False
                break
            byte = space[addrs[reg]]
            sb, db = bits_of(src), bits_of(dst)
            for s, d in zip(sb, db):
                val |= ((byte >> s) & 1) << d
        if ok:
            out[field] = val
    return out


def main():
    rows = []
    for p in sys.argv[2:]:
        p = Path(p)
        d = pickle.load(open(p, 'rb'))
        space = d['object']['ETROC2']
        addrs = reg_addr_map()
        rec = {'file': p.name, 'chip_dump': d.get('chip'), 'dump_version': d.get('version'),
               'ws_dump_bytes': len(d['object'].get('Waveform Sampler', []))}
        for blk in BLOCKS:
            if blk in register_decoding['ETROC2']['Register Blocks']:
                for k, v in decode(space, addrs, blk).items():
                    rec[k] = v
        # raw PeriCfg bytes, so nothing is lost
        for i in range(32):
            rec[f'PeriCfg{i}_raw'] = space[i]
        rows.append(rec)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(sys.argv[1], 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('wrote', sys.argv[1], len(rows), 'rows', len(keys), 'cols')
    for r in rows:
        print(r['file'], '| VrefGen_PD =', r.get('VrefGen_PD'), '(0 = internal Vref generator UP)',
              '| TS_PD =', r.get('TS_PD'), '| disPowerSequence =', r.get('disPowerSequence'))


if __name__ == '__main__':
    main()
