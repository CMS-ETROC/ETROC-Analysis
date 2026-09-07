#!/usr/bin/env python3
"""
reduce_bench.py -- reduce the ETROC2 FNAL bench charge-injection ("money plot") campaign of
July 2024 to per-point tables, in the same schema as the August TID campaign's reduce_tid.py.

Source tree (flat, no run_dir level, no dose, no rail scan):
    <root>/<date>_Array_Test_Results/<point_dir>/{TDC_Data_*.bin | TDC_Data_translated_*.nem
                                                    | FPGA_Data.dat}

Adapted, not rewritten, from reduce_tid.py (qinj_tid_et2p03/code/reduce_tid.py, 2026-09-04):
the per-hit CAL window, the moment definitions, decode_bins/decode_nems, do_point and the
FPGA_Data.dat parser are copied verbatim -- same definitions, so the two campaigns' tables can
be compared directly. Only the directory walk changes, because the July layout has one fewer
level of nesting and none of August's dose / VA / VD / temperature-token axes.

Binary decoding uses the ETROC-Analysis repo class TestBeam/condor_at_lxplus/core/decoding.py
:: DecodeBinary (copied verbatim into code/decoding.py, see the August DECODE_README.md),
firmware_key=0b0001, board_id=0x17f0f.
"""
import os, re, sys, argparse, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decoding import DecodeBinary

FIRMWARE_KEY = 0b0001
BOARD_ID = 0x17f0f

# ------------------------------------------------------------- per-hit CAL window
# Verbatim from reduce_tid.py: a hit is kept if |CAL - cal_mode| <= 3, cal_mode the modal CAL
# of that pixel at that scanned point. Same criterion as
# TestBeam/condor_at_lxplus/core/path_finder.py:446 / core/extract_events_by_path.py:155.
CAL_WINDOW = 3

# ---------------------------------------------------------------- name parsing
# Same point-directory regexes as reduce_tid.py -- the July DAQ scripts wrote the same
# "<chip>_VRef_SCurve_..." naming convention as the August money-plot runs.
RE_TDC   = re.compile(r'_VRef_SCurve_TDC_Pixel_C(\d+)_R(\d+)_QInj_(\d+)_Threshold_(\d+)$')
RE_TOFF  = re.compile(r'_VRef_SCurve_BinarySearch_TurnOff_Pixel_C(\d+)_R(\d+)_QInj_(\d+)_data$')
RE_NOISE = re.compile(r'_VRef_SCurve_NoiseOnly_Pixel_C(\d+)_R(\d+)_quietCB$')
RE_CHIP  = re.compile(r'^(ET2p\d+_BARE\d+)')

# Directories present in the July day trees that are NOT scan data:
#  - 'clear_fifo': a leftover FIFO-clear acquisition (translated .nem files), not a scan point.
#  - 'ws_testing_qinj_*': wafer-probe-station debris (rawdataWS_*.csv, WS_Aout/Dout_*.html),
#    a different measurement entirely, no TDC_Data files at all.
SKIP_POINT_PREFIXES = ('ws_testing',)
SKIP_POINT_NAMES = {'clear_fifo'}


# ---------------------------------------------------------------- decoding (verbatim from reduce_tid.py)
def decode_bins(files):
    dec = DecodeBinary(firmware_key=FIRMWARE_KEY, board_id=[BOARD_ID] * 4,
                       file_list=files, save_nem=None,
                       skip_fw_filler=True, skip_event_df=True, skip_crc_df=True)
    out = dec.decode_files()
    df = out[0] if isinstance(out, tuple) else out
    return df


NEM_D = re.compile(r'^D\s+(\S+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$')


def decode_nems(files):
    """Parse the D lines of already-translated .nem files. Field order is
    'D <channel> <EA> <ROW> <COL> <TOA> <TOT> <CAL>' (DecodeBinary.write_to_nem)."""
    rows = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                if line[0] != 'D':
                    continue
                m = NEM_D.match(line)
                if m:
                    rows.append((int(m.group(3)), int(m.group(4)),
                                 int(m.group(5)), int(m.group(6)), int(m.group(7))))
    if not rows:
        return pd.DataFrame(columns=['row', 'col', 'toa', 'tot', 'cal'])
    return pd.DataFrame(rows, columns=['row', 'col', 'toa', 'tot', 'cal'])


def empty_moments():
    out = {'n_hits': 0}
    for c in ('toa', 'tot', 'cal'):
        out[c + '_mean'] = np.nan
        out[c + '_std'] = np.nan
    out.update(cal_windowed(None, 0))
    return out


def cal_windowed(df, n):
    """Same moments over the CAL-windowed hits: |CAL - per-point CAL mode| <= CAL_WINDOW.
    Verbatim from reduce_tid.py."""
    out = {'cal_mode': np.nan, 'n_hits_cal': 0, 'frac_rejected': np.nan}
    for c in ('toa', 'tot', 'cal'):
        out[c + '_mean_cal'] = np.nan
        out[c + '_std_cal'] = np.nan
    if not n:
        return out
    cal = np.asarray(df['cal'], dtype='int64')
    npix = df.groupby(['row', 'col']).ngroups if {'row', 'col'} <= set(df.columns) else 1
    if npix > 1:
        m = df.groupby(['row', 'col'])['cal'].transform(lambda t: t.mode().iat[0])
        keep = np.abs(cal - np.asarray(m, dtype='int64')) <= CAL_WINDOW
    else:
        vals, cnts = np.unique(cal, return_counts=True)
        mode = int(vals[cnts.argmax()])
        keep = np.abs(cal - mode) <= CAL_WINDOW
        out['cal_mode'] = float(mode)
    k = int(keep.sum())
    out['n_hits_cal'] = k
    out['frac_rejected'] = float(1.0 - k / float(n))
    if k:
        d = df[keep]
        for c in ('toa', 'tot', 'cal'):
            out[c + '_mean_cal'] = float(d[c].mean())
            out[c + '_std_cal'] = float(d[c].std(ddof=1)) if k > 1 else np.nan
    return out


def moments(df, row=None, col=None):
    """Plain mean / sample std (ddof=1) over every hit; no cuts. Verbatim from reduce_tid.py."""
    if row is not None and 'row' in df.columns and len(df):
        df = df[(df['row'] == row) & (df['col'] == col)]
    n = len(df)
    out = {'n_hits': n}
    for c in ('toa', 'tot', 'cal'):
        if n:
            out[c + '_mean'] = float(df[c].mean())
            out[c + '_std'] = float(df[c].std(ddof=1)) if n > 1 else np.nan
        else:
            out[c + '_mean'] = np.nan
            out[c + '_std'] = np.nan
    out.update(cal_windowed(df, n))
    return out


# ---------------------------------------------------------------- workers (verbatim from reduce_tid.py)
def do_point(task):
    kind, path, meta = task
    p = Path(path)
    try:
        bins = sorted(p.glob('TDC_Data_*.bin'), key=lambda x: int(re.search(r'_(\d+)\.bin$', x.name).group(1)))
        nems = sorted(p.glob('TDC_Data_translated_*.nem'), key=lambda x: int(re.search(r'_(\d+)\.nem$', x.name).group(1)))
        rec = dict(meta)
        if bins:
            df = decode_bins(bins)
            rec['n_files'] = len(bins)
            rec['source_format'] = 'bin'
        elif nems:
            df = decode_nems(nems)
            rec['n_files'] = len(nems)
            rec['source_format'] = 'nem'
        else:
            rec.update(empty_moments())
            rec.update({'n_files': 0, 'source_format': 'none',
                        'n_hits_all_pixels': 0, 'n_pixels_seen': 0, 'decode_error': ''})
            return rec
        rec['n_hits_all_pixels'] = int(len(df))
        rec['n_pixels_seen'] = int(df.groupby(['row', 'col']).ngroups) if len(df) else 0
        r, c = meta.get('row'), meta.get('col')
        if r is not None and not (isinstance(r, float) and np.isnan(r)):
            rec.update(moments(df, int(r), int(c)))
        else:
            rec.update(moments(df))
        rec['decode_error'] = ''
        return rec
    except Exception:
        rec = dict(meta)
        rec.update(empty_moments())
        rec.update({'n_files': -1, 'source_format': 'error',
                    'n_hits_all_pixels': 0, 'n_pixels_seen': 0,
                    'decode_error': traceback.format_exc(limit=2)})
        return rec


def read_fpga_dat(path):
    """FPGA_Data.dat: 7 comma-separated integers per line. Verbatim from reduce_tid.py.
      [0] fpga_state  [1] ?  [2] ?  [3] tdc_data  [4] l1a_max  [5] triggerbits  [6] dac
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            t = line.strip().split(',')
            if len(t) != 7:
                continue
            try:
                rows.append([int(x) for x in t])
            except ValueError:
                continue
    return rows


def _classify(p, cfg, tasks, turnoff, noiseonly):
    n = p.name
    if cfg.get('chip') in (None, 'not recorded'):
        m = RE_CHIP.match(n)
        if m:
            cfg = dict(cfg); cfg['chip'] = m.group(1)
    m = RE_TDC.search(n)
    if m:
        meta = dict(cfg)
        meta.update({'kind': 'TDC', 'col': int(m.group(1)), 'row': int(m.group(2)),
                     'charge_fC': float(m.group(3)), 'dac': int(m.group(4)),
                     'dac_source': 'point_dir_name', 'point_dir': n})
        tasks.append(('TDC', str(p), meta))
        return
    m = RE_TOFF.search(n)
    if m:
        turnoff.append((dict(cfg), int(m.group(1)), int(m.group(2)), float(m.group(3)), str(p), n))
        return
    m = RE_NOISE.search(n)
    if m:
        noiseonly.append((dict(cfg), int(m.group(1)), int(m.group(2)), str(p), n))
        return
    # unrecognised directory: still decode it, with unknown coordinates
    meta = dict(cfg)
    meta.update({'kind': 'other', 'row': np.nan, 'col': np.nan,
                 'charge_fC': cfg.get('charge_tag_fC', np.nan), 'dac': np.nan,
                 'dac_source': 'not recorded', 'point_dir': n})
    tasks.append(('other', str(p), meta))


# ---------------------------------------------------------------- the July-specific walk
def build_tasks_flat(root):
    """Walk <root>/<date>_Array_Test_Results/<point_dir> directly -- the July bench layout has
    no run_dir level and no voltage-scan sub-runs. One chip per day (verified by inspection:
    every day directory holds exactly one ET2p0x_BAREn prefix among its scan points, plus
    'clear_fifo' / 'ws_testing_qinj_*' debris that is skipped here, never decoded)."""
    root = Path(root)
    tasks, configs, turnoff, noiseonly = [], [], [], []
    for day in sorted(p for p in root.iterdir() if p.is_dir()):
        cfg = {
            'chip': 'not recorded', 'vd_V': np.nan, 'va_V': np.nan,
            'temperature_C': np.nan, 'dose_MRad': np.nan, 'elapsed_h': np.nan,
            'charge_tag_fC': np.nan, 'repeat': 1,
            'temperature_source': 'not recorded', 'dose_source': 'not recorded',
            'day': day.name, 'run_dir': day.name, 'sub_run_dir': '',
            'config': day.name, 'scan_family': 'money_plot',
        }
        # Peek at one scan point-dir name to fix the day-level 'chip' up front (each day holds
        # exactly one chip). _classify() already re-derives 'chip' per point independently, so
        # this only affects the summary row this function appends to `configs` -- the per-point
        # rows in `tasks`/`turnoff`/`noiseonly` were always correct regardless of this.
        for p in sorted(day.iterdir()):
            if not p.is_dir() or p.name in SKIP_POINT_NAMES or p.name.startswith(SKIP_POINT_PREFIXES):
                continue
            m = RE_CHIP.match(p.name)
            if m:
                cfg['chip'] = m.group(1)
                break
        n_pts = 0
        for p in sorted(day.iterdir()):
            if not p.is_dir():
                continue
            if p.name in SKIP_POINT_NAMES or p.name.startswith(SKIP_POINT_PREFIXES):
                continue
            _classify(p, cfg, tasks, turnoff, noiseonly)
            n_pts += 1
        cfg['n_point_dirs'] = n_pts
        configs.append(cfg)
    return tasks, configs, turnoff, noiseonly


# ---------------------------------------------------------------- BL/NW autocal maps
# Present for exactly one day in this campaign: 2024-07-17, ET2p03_BARE4 ('<chip>_BL_map.pickle'
# / '<chip>_NW_map.pickle', a pair of 16x16 float64 numpy arrays -- same format as the August
# X9 2024-08-03 pickles). No ChipConfig register-dump pickles exist anywhere in the July tree
# (checked: find -iname '*.pckl' -o -iname 'ChipConfig*' across all 7 days, zero hits), so unlike
# August there is no chipconfig.csv for July -- VRef and the rest of the register map are simply
# not recorded for this campaign.
def find_blnw(root):
    import numpy as _np
    root = Path(root)
    rows = []
    for day in sorted(p for p in root.iterdir() if p.is_dir()):
        for bl in sorted(day.glob('*_BL_map.pickle')):
            chip = bl.name[:-len('_BL_map.pickle')]
            nw = day / f'{chip}_NW_map.pickle'
            if not nw.exists():
                continue
            bl_arr = _np.load(bl, allow_pickle=True) if bl.suffix != '.pickle' else __import__('pickle').load(open(bl, 'rb'))
            nw_arr = __import__('pickle').load(open(nw, 'rb'))
            for row in range(bl_arr.shape[0]):
                for col in range(bl_arr.shape[1]):
                    rows.append({'chip': chip, 'day': day.name, 'row': row, 'col': col,
                                 'bl_dac': float(bl_arr[row, col]), 'nw_dac': float(nw_arr[row, col])})
    return rows


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--prefix', default='jul_et2p03')
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tasks, configs, turnoff, noiseonly = build_tasks_flat(a.root)
    print(f'point dirs to decode: {len(tasks)}   turnoff: {len(turnoff)}   noiseonly: {len(noiseonly)}   configs: {len(configs)}', flush=True)
    if a.limit:
        tasks = tasks[:a.limit]

    # --- BL/NW autocal maps
    blnw_rows = find_blnw(a.root)
    if blnw_rows:
        pd.DataFrame(blnw_rows).to_csv(out / f'{a.prefix}_blnw.csv', index=False)
        print(f'wrote {a.prefix}_blnw.csv: {len(blnw_rows)} rows', flush=True)

    # --- turn-off (no binary decoding needed)
    trows = []
    for cfg, col, row, q, path, n in turnoff:
        lines = read_fpga_dat(Path(path) / 'FPGA_Data.dat')
        rec = dict(cfg)
        rec.update({'kind': 'TurnOff', 'row': row, 'col': col, 'charge_fC': q, 'point_dir': n,
                    'n_search_steps': len(lines)})
        if lines:
            last = lines[-1]
            rec['turnoff_dac'] = last[6]
            rec['turnoff_triggerbits'] = last[5]
            rec['turnoff_tdc_data'] = last[3]
            rec['l1a_max'] = lines[0][4]
            rec['dac_first'] = lines[0][6]
            rec['dac_min'] = min(l[6] for l in lines)
            rec['dac_max'] = max(l[6] for l in lines)
            rec['search_trace'] = ';'.join(f'{l[6]}:{l[5]}' for l in lines)
        else:
            for k in ('turnoff_dac', 'turnoff_triggerbits', 'turnoff_tdc_data', 'l1a_max',
                      'dac_first', 'dac_min', 'dac_max'):
                rec[k] = np.nan
            rec['search_trace'] = ''
        trows.append(rec)
    if trows:
        pd.DataFrame(trows).to_csv(out / f'{a.prefix}_turnoff.csv', index=False)

    # --- noise only
    nrows = []
    for cfg, col, row, path, n in noiseonly:
        lines = read_fpga_dat(Path(path) / 'FPGA_Data.dat')
        rec = dict(cfg)
        rec.update({'kind': 'NoiseOnly', 'row': row, 'col': col, 'point_dir': n,
                    'n_steps': len(lines)})
        if lines:
            rec['dac_min'] = min(l[6] for l in lines)
            rec['dac_max'] = max(l[6] for l in lines)
            rec['triggerbits_max'] = max(l[5] for l in lines)
            rec['dac_at_triggerbits_max'] = max(lines, key=lambda l: l[5])[6]
            rec['scan_trace'] = ';'.join(f'{l[6]}:{l[5]}' for l in lines)
        else:
            for k in ('dac_min', 'dac_max', 'triggerbits_max', 'dac_at_triggerbits_max'):
                rec[k] = np.nan
            rec['scan_trace'] = ''
        nrows.append(rec)
    if nrows:
        pd.DataFrame(nrows).to_csv(out / f'{a.prefix}_noiseonly.csv', index=False)

    pd.DataFrame(configs).to_csv(out / f'{a.prefix}_configs.csv', index=False)

    # --- binary decoding
    recs = []
    import time
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(do_point, t) for t in tasks]
        for i, f in enumerate(as_completed(futs)):
            recs.append(f.result())
            if (i + 1) % 200 == 0:
                el = time.time() - t0
                print(f'  {i+1}/{len(tasks)}  {el:.0f}s  ({el/(i+1)*1000:.1f} ms/dir)  '
                      f'projected total {el/(i+1)*len(tasks)/60:.1f} min', flush=True)
    dt = time.time() - t0
    print(f'decoded {len(recs)} dirs in {dt:.1f} s ({dt/max(len(recs),1)*1000:.1f} ms/dir)', flush=True)

    cols = ['config', 'day', 'run_dir', 'sub_run_dir', 'scan_family', 'chip', 'kind',
            'temperature_C', 'temperature_source', 'dose_MRad', 'dose_source',
            'va_V', 'vd_V', 'elapsed_h', 'repeat',
            'row', 'col', 'charge_fC', 'dac', 'dac_source',
            'n_hits', 'n_files', 'source_format',
            'toa_mean', 'toa_std', 'tot_mean', 'tot_std', 'cal_mean', 'cal_std',
            'n_hits_cal', 'frac_rejected', 'cal_mode',
            'toa_mean_cal', 'toa_std_cal', 'tot_mean_cal', 'tot_std_cal',
            'cal_mean_cal', 'cal_std_cal',
            'n_hits_all_pixels', 'n_pixels_seen', 'point_dir', 'decode_error']
    df = pd.DataFrame(recs)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols].sort_values(['config', 'row', 'col', 'charge_fC', 'dac']).reset_index(drop=True)
    name = f'{a.prefix}_points.feather' if not a.limit else f'{a.prefix}_points_TEST.feather'
    df.to_feather(out / name)
    print(f'wrote {out / name}: {df.shape}', flush=True)
    print(df['decode_error'].astype(bool).sum(), 'dirs with decode errors', flush=True)


if __name__ == '__main__':
    main()
