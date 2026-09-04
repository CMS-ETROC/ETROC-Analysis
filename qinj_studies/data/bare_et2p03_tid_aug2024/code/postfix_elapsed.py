#!/usr/bin/env python3
"""Recompute the name-parsed `elapsed_h` column in place across the already-written tables,
using the corrected lenient regex in reduce_tid.py. Touches ONE name-parsed column and
re-decodes nothing; every other value is left byte-for-byte as the decode produced it.
Needed because one 2024 run directory is typo'd `_37hm00s_` (blank minutes)."""
import sys, pandas as pd
from pathlib import Path
sys.path.insert(0, '/uscms_data/d3/murtazas/scratch/qinj_tid_et2p03/code')
from reduce_tid import RE_ELAP

B = Path('/uscms_data/d3/murtazas/scratch/qinj_tid_et2p03')


def elapsed(name):
    m = RE_ELAP.search(str(name))
    if not m:
        return float('nan')
    return (int(m.group(1)) + (int(m.group(2)) / 60.0 if m.group(2) else 0.0)
            + (int(m.group(3)) / 3600.0 if m.group(3) else 0.0))


for p in sorted(list((B / 'out').glob('tid_et2p03_*')) + list((B / 'out_x9').glob('x9_et2p03_*'))):
    if p.suffix == '.feather':
        d = pd.read_feather(p)
    elif p.suffix == '.csv':
        d = pd.read_csv(p)
    else:
        continue
    if 'elapsed_h' not in d.columns or 'run_dir' not in d.columns:
        continue
    sub = d['sub_run_dir'].fillna('').astype(str)
    src = sub.where(sub.str.len() > 0, d['run_dir'].fillna('').astype(str))
    new = src.map(elapsed)
    n = int((new.notna() & d['elapsed_h'].isna()).sum())
    if n:
        d['elapsed_h'] = new
        d.to_feather(p) if p.suffix == '.feather' else d.to_csv(p, index=False)
    print(f'{p.parent.name}/{p.name:34s} filled {n:5d} previously-NaN elapsed_h')
