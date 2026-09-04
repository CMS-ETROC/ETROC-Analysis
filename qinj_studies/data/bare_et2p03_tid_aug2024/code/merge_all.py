#!/usr/bin/env python3
"""Merge the TID_August (working-copy) and X9 (master-copy) reductions into one set of
union tables, adding a `source_tree` column. The TID_August-only tables are left in place
untouched; nothing here re-decodes anything."""
import pandas as pd, hashlib, os
from pathlib import Path

B = Path('/uscms_data/d3/murtazas/scratch/qinj_tid_et2p03')
OUT = B / 'out'
TID, X9 = 'TID_August_working_copy', 'X9_master_copy'


def tag(df, t):
    df = df.copy()
    df.insert(0, 'source_tree', t)
    return df


def union(name, reader, writer, ext):
    a = reader(OUT / f'tid_et2p03_{name}.{ext}')
    b = reader(B / 'out_x9' / f'x9_et2p03_{name}.{ext}')
    m = pd.concat([tag(a, TID), tag(b, X9)], ignore_index=True, sort=False)
    p = OUT / f'tid_et2p03_{name}_all.{ext}'
    writer(m, p)
    print(f'{name:10s} {len(a):6d} + {len(b):6d} = {len(m):6d} rows x {m.shape[1]} cols -> {p.name}')
    return m


pts = union('points', pd.read_feather, lambda d, p: d.reset_index(drop=True).to_feather(p), 'feather')
for n in ('turnoff', 'noiseonly', 'configs'):
    union(n, pd.read_csv, lambda d, p: d.to_csv(p, index=False), 'csv')

print()
print('=== union points: dose x chip x kind ===')
print(pts.groupby(['chip', 'dose_MRad', 'kind'], dropna=False).size().to_string())
print()
print('=== money-plot (TDC) coverage per condition ===')
t = pts[pts.kind == 'TDC']
g = t.groupby(['chip', 'dose_MRad', 'va_V', 'vd_V'], dropna=False).agg(
    n_points=('dac', 'size'), dac_min=('dac', 'min'), dac_max=('dac', 'max'),
    charges=('charge_fC', lambda s: sorted(int(x) for x in s.unique())),
    n_pix=('point_dir', lambda s: 0), hits=('n_hits', 'sum'),
    elapsed_h=('elapsed_h', lambda s: sorted(set(round(x, 2) for x in s.dropna()))))
g['n_pix'] = t.groupby(['chip', 'dose_MRad', 'va_V', 'vd_V'], dropna=False).apply(
    lambda d: len(set(zip(d.row, d.col))))
print(g.to_string())
print()
print('total TDC points', len(t), ' total hits', int(pts.n_hits.sum()),
      ' zero-hit points', int((pts.n_hits == 0).sum()), ' decode errors', int(pts.decode_error.astype(bool).sum()))
print()
for f in sorted(OUT.glob('tid_et2p03_*')):
    print(f'| `{f.name}` | {f.stat().st_size} | `{hashlib.md5(f.read_bytes()).hexdigest()}` |')
