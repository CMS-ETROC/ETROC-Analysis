# `qinj_bench_jul2024` — ETROC2 FNAL bench charge-injection ("money plot") set, July 2024, reduced to per-point tables

**What this is.** A reduction of the FNAL **bench** charge-injection campaign of **July 2024**,
seven days across four bare dies (three ET2.03, one ET2.01). It uses the **same definitions**
as the August 2024 TID reduction (`qinj_tid_et2p03/out/DECODE_README.md`, `tid_et2p03_*`
tables) — same per-hit CAL window, same moment definitions, same decoder — so the two
campaigns' tables can be compared directly. This document follows that one's structure and
calls out only what differs.

`fermilab-etroc3` was treated as **read-only** throughout: only `ls`, `find`, `du`, `md5sum`
and one `tar cf -` to stdout per day were run there, all `nice -n 19`d; nothing was written,
moved or deleted on that machine, and the removable X9 disk was never unmounted. Working
copies live on LPC under `/uscms_data/d3/murtazas/scratch/qinj_bench_jul2024/`.

Every quantity below is labelled **RAW** or **DERIVED**. Where the campaign did not record
something, the tables say **`not recorded`** — never a guess.

---

## 1. Provenance

Source: `fermilab-etroc3:/run/media/daq/X9/et2p03_qinj_data/2024-07-{15,16,17,18,22,29,30}_Array_Test_Results/`.
Streamed to LPC one day at a time with `ssh fermilab-etroc3 'nice -n 19 tar cf - -C
.../et2p03_qinj_data <day>' | ssh lpc 'tar xf - -C raw_x9/'`, through the laptop (lpc cannot
resolve `fermilab-etroc3` directly — no direct lpc-to-DAQ-host route exists).

**Integrity, per day**: file count (`find -type f | wc -l`) and content-byte sum
(`find -type f -printf '%s\n' | awk '{s+=$1} END{print s}'` — **not** `du -sb`, which
double-counts directory-inode overhead differently across the source exFAT and destination
NFS filesystems and produced a false ~10 MB mismatch on the first day before this was
diagnosed) matched exactly on all 7 days, plus a 20-21 file MD5 spot check per day
(`LC_ALL=C`-sorted file list, every Nth file) that was identical on both ends, 146 files total.

| day | chip | files | bytes | md5 sample |
|---|---|---:|---:|---|
| 2024-07-15 | ET2p03_BARE1 | 10021 | 7403534992 | 21/21 match |
| 2024-07-16 | ET2p03_BARE4 | 10550 | 7879911381 | 20/20 match |
| 2024-07-17 | ET2p03_BARE4 | 3090 | 2275852634 | 21/21 match |
| 2024-07-18 | ET2p03_BARE4 | 8526 | 6340174208 | 21/21 match |
| 2024-07-22 | **ET2p01_BARE10** | 9725 | 7079871883 | 21/21 match |
| 2024-07-29 | ET2p03_BARE9 | 3457 | 4061548929 | 21/21 match |
| 2024-07-30 | ET2p03_BARE9 | 3290 | 3728778770 | 21/21 match |
| **total** | | **48659** | **38769672797 = 38.8 GB** | |

**2024-07-22 is the ET2.01 die** (`ET2p01_BARE10`); the other six days are ET2.03 bare dies —
this is the DAC-scale comparison point the August reduction flagged as out of scope for it.

EOS upload: `xrdcp -r --parallel 4` per day to
`root://cmseos.fnal.gov//eos/uscms/store/user/lpcmtdstudies/fnal_Jul2024_bench/raw/`, verified
with a full recursive `xrdfs ls -l -R` file-count and byte-sum against the LPC copy — exact
match, all 7 days (see `PROVENANCE.md` there for the numbers).

---

## 2. What differs from the August (`TID_August` / X9) layout

1. **Flat, one level shallower.** August is `<root>/<run_dir>/<point_dir>` (or
   `<root>/<day>/<run_dir>/<point_dir>` for the X9 master copy). **July is
   `<root>/<date>_Array_Test_Results/<point_dir>` directly** — no `run_dir` wrapper, because
   there is only one "run" (one chip, one continuous sweep) per day. `reduce_bench.py`'s
   `build_tasks_flat()` walks this layout; everything downstream of a point directory
   (`do_point`, `moments`, `cal_windowed`, `read_fpga_dat`) is copied verbatim from
   `reduce_tid.py`.
2. **No dose, no rail scan, no temperature token, no elapsed-time token.** July's point/turn-off/
   noise-only directory names carry only `Pixel_C<col>_R<row>_QInj_<q>_Threshold_<dac>` (or the
   turn-off / quietCB equivalents) — none of `_<n>MRad`, `<n>p<nn>VA`, `<n>p<nn>VD`, `_m<n>C`, or
   `_<n>h<n>m<n>s` appear anywhere in the July tree. `dose_MRad`, `dose_source`, `va_V`, `vd_V`,
   `temperature_C`, `temperature_source`, `elapsed_h` are therefore **NaN / `not recorded` for
   every row** — this is a property of the campaign (a bench characterization run, not a
   dose/rail-scan campaign), not a gap in the reduction.
3. **No voltage-scan sub-runs at all.** Every July directory is `scan_family = money_plot`.
4. **Charges differ from August's set.** July charges are **{5, 6, 8, 15, 22, 27} fC** (six
   values); August's money plots use {6, 8, 10, 15, 22} fC. **10 fC is not in July; 5 and 27 fC
   are not in August.** Compare charge-by-charge, not index-by-index.
5. **File formats: mixed .bin and .nem, split by day, not intermixed within a day.** Days
   2024-07-{15,16,17,18,22} are **entirely `.nem`** (already translated at acquisition time,
   like August's `ET2p03_BARE9_Voltage_scan` sub-runs); days 2024-07-{29,30} are **entirely
   `.bin`** (decoded here with the same `DecodeBinary`, byte-identical copy of
   `TestBeam/condor_at_lxplus/core/decoding.py`, md5 `289ef6cb7b7e48589e02d817de5060ed`). Both
   code paths already existed in `reduce_tid.py`'s `do_point()` and are reused unchanged.
6. **No ChipConfig register dumps anywhere in the July tree** (checked: `find -iname '*.pckl'
   -o -iname 'ChipConfig*'` across all 7 days found only the two BL/NW map files below). **There
   is therefore no `jul_et2p03_chipconfig.csv`** — VRef and the rest of the register map are
   simply not recorded for this campaign. Do not infer VRef state from the August dumps; they
   are a different die family and a different date.
7. **BL/NW autocal maps exist for exactly one day.** `2024-07-17` carries
   `ET2p03_BARE4_BL_map.pickle` / `ET2p03_BARE4_NW_map.pickle` (16x16 float64 arrays, same
   format as the August `2024-08-03` pickles). Melted to `jul_et2p03_blnw.csv` (256 rows: chip,
   day, row, col, `bl_dac`, `nw_dac`). **No BL/NW maps exist for BARE1, BARE9 or BARE10** — not
   invented, not carried over from another day.
8. **Junk directories differ.** August's non-scan debris is `ChipConfig/` and `clear_fifo/`
   (its `SKIP_RUN_DIRS`). July's is **`clear_fifo/`** (present on 5 of 7 days; leftover
   FIFO-clear `.nem` files, not scan data) and **`ws_testing_qinj_*/`** (wafer-probe-station
   debris — `rawdataWS_*.csv`, `WS_Aout/Dout_*.html`; present only on 2024-07-15,17,18, 9/1/1
   directories respectively). Both are excluded from decoding by name (`SKIP_POINT_NAMES`,
   `SKIP_POINT_PREFIXES` in `reduce_bench.py`) but were still copied into `raw_x9/` for
   completeness — they are on disk, just never decoded.
9. **`config` has no sub-run structure.** Since there is one chip and no voltage-scan sub-runs,
   `config` = the day directory name (`sub_run_dir` is always `''`).
10. **Single tree, no `_all` union.** August has two table families because the working copy
    (`TID_August`) and the X9 master copy overlap and had to be merged. July comes from one tree
    only (X9), so there is exactly one table family, `jul_et2p03_*` — no `_all` variant.

---

## 3. What conditions the set actually contains

| day | chip | pixels | charges (fC) | DAC range | TDC dirs | TurnOff dirs | NoiseOnly dirs |
|---|---|---|---|---|---:|---:|---:|
| 2024-07-15 | ET2p03_BARE1 | (0,0)(0,14)(3,0)(3,14)(12,0)(12,14) | 5,6,8,15,22,27 | 603–924 | 3670 | 36 | 6 |
| 2024-07-16 | ET2p03_BARE4 | same six | 5,6,8,15,22,27 | 585–935 | 4057 | 36 | 6 |
| 2024-07-17 | ET2p03_BARE4 | same six | 5,6,8,15,22,27 | 585–857 | 1101 | 36 | 6 |
| 2024-07-18 | ET2p03_BARE4 | same six | 5,6,8,15,22,27 | 587–936 | 3168 | 36 | 6 |
| 2024-07-22 | **ET2p01_BARE10** | same six | 5,6,8,15,22,27 | 376–673 | 4113 | 36 | 6 |
| 2024-07-29 | ET2p03_BARE9 | same six | 5,6,8,15,22,27 | 344–610 | 3399 | 36 | 6 |
| 2024-07-30 | ET2p03_BARE9 | same six | 5,6,8,15,22,27 | 585–846 | 3248 | 36 | 6 |
| **total** | 4 dies | | | | **22756** | **252** | **42** |

All 7 days use the **same six pixels** as the August campaign — (row, col) ∈ {(0,0), (0,14),
(3,0), (3,14), (12,0), (12,14)} — and the **same six charges**, 5/6/8/15/22/27 fC (not August's
five). 36 turn-off dirs/day = 6 pixels × 6 charges; 6 noise-only dirs/day = 6 pixels.

---

## 4. The deliverables

Written by `reduce_bench.py` (`code/`, adapted from `reduce_tid.py`; the CAL window, moment
definitions, `decode_bins`/`decode_nems`, `do_point` and the `FPGA_Data.dat` parser are copied
verbatim). One table family, prefix `jul_et2p03`:

### `jul_et2p03_points.feather` — one row per decoded TDC point directory

Same 42 columns and meanings as `tid_et2p03_points.feather` (see the August README §2 for the
full column table). For July: `dose_MRad`, `dose_source`, `va_V`, `vd_V`, `temperature_C`,
`temperature_source`, `elapsed_h` are NaN/`not recorded` for every row (§2.2 above); `kind` is
always `TDC`; `scan_family` is always `money_plot`; `source_format` is `nem` for
2024-07-{15,16,17,18,22} and `bin` for 2024-07-{29,30}.

**22756 rows × 42 columns. 0 decode errors. 585 zero-hit points (2.6%)** — near-baseline and
near-turnoff threshold points, the same pattern as the August set's zero-hit rows.
**378 051 805 hits** behind the table.

### `jul_et2p03_turnoff.csv` — 252 rows (7 days × 6 pixels × 6 charges)

Same parsing of `FPGA_Data.dat` and the same columns as `tid_et2p03_turnoff.csv`
(`turnoff_dac` = DAC on the last line, the campaign's own turn-off definition; `search_trace`
carries the full binary search).

### `jul_et2p03_noiseonly.csv` — 42 rows (7 days × 6 pixels)

Same `quietCB` noise-scan parsing as `tid_et2p03_noiseonly.csv`.

### `jul_et2p03_configs.csv` — 7 rows, one per day

`chip`, `day`, `run_dir` (= day dir name), `sub_run_dir` (always `''`), `config` (= day dir
name), `scan_family` (always `money_plot`), `n_point_dirs` (TDC+TurnOff+NoiseOnly count for
that day; the six `not recorded`/NaN columns from §2.2), and `repeat` (always 1 — no `_v<n>`
retakes were found in July's names).

### `jul_et2p03_blnw.csv` — 256 rows (one day only, §2.7)

`chip`, `day`, `row`, `col`, `bl_dac`, `nw_dac` — the 16×16 BL/NW autocal maps for
`ET2p03_BARE4` on 2024-07-17. **Not available for the other three dies/six days.**

### No `jul_et2p03_chipconfig.csv`

No ChipConfig register-dump pickles exist anywhere in the July tree (§2.6). VRef and the rest
of the register map are `not recorded` for this campaign — there is nothing to decode.

---

## 5. Verification

1. **Decode failures: zero.** `decode_error` is empty for all 22756 rows.
2. **Pixel identity.** By construction, `row`/`col` come from the same `Pixel_C<col>_R<row>`
   token in the point-directory name that also drives the pixel selection in `moments()` — same
   mechanism as August, not independently re-checked here (time did not allow a hand-decode
   cross-check equivalent to the August README's §6.1; flagged as a gap, not silently skipped).
3. **Codes lie in physical ranges**: `toa_mean`, `tot_mean`, `cal_mean` all fall inside the
   10-bit TOA / 9-bit TOT / 10-bit CAL fields (checked by construction of `moments()`, the same
   function used and range-checked for August).
4. **DecodeBinary's own internal checks** (`Number of hits does not match!`, and a
   "40 bits word which is not matched with the pattern" warning not seen in the August log)
   fired during decoding of the `.bin` days (2024-07-29/30) at the same low rate as August's
   22 occurrences — reported, not suppressed; `decode_error` stayed empty throughout because
   `DecodeBinary` discards the offending event and continues, exactly as it does for August.

**Caveat, stated plainly:** unlike the August reduction, this one did **not** run an
independent from-scratch re-decoder (`verify_hand.py`) against a July point directory. The
decoder is the identical, byte-verified `DecodeBinary` class used and independently verified
for August, so this is a smaller risk than a from-scratch decoder would carry, but it is a real
gap relative to the August verification standard and is recorded here rather than implied away.

---

## 6. Reproducing this

```bash
# on LPC
source /cvmfs/sft.cern.ch/lcg/views/LCG_104a/x86_64-el9-gcc11-opt/setup.sh
export PYTHONUSERBASE=/uscms_data/d3/murtazas/scratch/qinj_tid_et2p03/pyenv   # shared with the Aug reduction; crc + natsort
cd /uscms_data/d3/murtazas/scratch/qinj_bench_jul2024
python3 code/reduce_bench.py --root raw_x9 --out out --prefix jul_et2p03 --workers 6
```

**Cost.** 6 worker processes, `nice -n 10`, interactive on `cmslpc`, run **concurrently with two
~35 GB EOS uploads** (heavy shared-NFS contention slowed both): **22756 directories in 3194 s
(53.2 min, 140.4 ms/dir)** — about 2.2x the per-directory cost of the uncontended August run
(72-224 ms/dir); a quiet re-run would likely be faster.

## 7. File manifest

| file | rows × cols |
|---|---|
| `jul_et2p03_points.feather` | 22756 × 42 |
| `jul_et2p03_turnoff.csv` | 252 × 29 |
| `jul_et2p03_noiseonly.csv` | 42 × 25 |
| `jul_et2p03_configs.csv` | 7 × 16 |
| `jul_et2p03_blnw.csv` | 256 × 6 |

**Where the copies live.**
`lpc:/uscms_data/d3/murtazas/scratch/qinj_bench_jul2024/` (`raw_x9` + `code` + `out`; 38.8 GB
of raw) · EOS `root://cmseos.fnal.gov//eos/uscms/store/user/lpcmtdstudies/fnal_Jul2024_bench/`
(`raw/` + `reduced_qinj_feather/`) · `~/ETROC_viz_mirror/qinj_bench_jul2024/` (laptop) ·
`lxplus:/afs/cern.ch/user/m/musafdar/CMS_ETROC/ETROC-Analysis/notes/qinj_bench_jul2024/`.
