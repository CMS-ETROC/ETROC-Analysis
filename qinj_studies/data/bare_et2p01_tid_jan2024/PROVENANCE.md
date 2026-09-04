# Provenance — bare ET2.01 die, CERN TID campaign, January 2024

**Chip** `ET2p01_W36_6` — an ETROC2 version 2.01 **bare die, with no LGAD bonded**.
**Campaign** CERN, January 2024, total-ionising-dose test at 0, 200 and 400 MRad.
**Pixels** 12: rows {0, 1, 3, 12} × columns {2, 8, 14}. **Charges** 5, 6, 8, 15, 22, 27 fC.

Everything in this directory was read from the Fermilab LPC EOS area

```
root://cmseos.fnal.gov//store/group/lpcmtdstudies/cern_Jan2024/TID/
```

on 2026-09-04, read-only, via `xrdcp` and `xrdfs ls -l` from an LPC interactive node. Nothing on
EOS was written, moved or deleted.

## Staged files

| file | bytes | md5 |
|---|---|---|
| `ET2p01_W36_6_TDC_summary_allMRad.feather` | 1,186,578 | `333d13522d479f39f3d5f097934532f8` |
| `blnw_map_pickles.tar.gz` | 2,797 | `b7a3c0e732a16e9a5f5ab070ec55f7b8` |
| `blnw_maps.csv` | 26,421 | `5fe12a58705b98073304e50948415115` |
| `fpga_turnoff_dat.tar.gz` | 34,184 | `686370c1e9ec3b994d60bb257fdd03c2` |
| `fpga_turnoff_probes.csv` | 117,133 | `37d8b6cb5c3b8b18a9e5e90b772e9e8d` |
| `sweep_point_dirs.csv.gz` | 143,772 | `c8a24b708d9b788eadaa607f5ad14f59` |

### `ET2p01_W36_6_TDC_summary_allMRad.feather` — RAW, copied byte for byte

The per-point TOA/TOT/CAL summary, 22 055 rows × 12 columns. One row per
(dose, temperature, pixel, charge, threshold DAC), carrying the mean and the standard deviation
of each TDC code over every hit recorded at that point. **No cuts, no CAL window and no outlier
rejection were applied when it was built** — it is a plain per-point mean and standard deviation.

Its md5 on EOS is `333d13522d479f39f3d5f097934532f8`, identical to the staged copy; notebook 01
prints both.

Source path: `<TID>/ET2p01_W36_6_TDC_summary_allMRad.feather` (1 186 578 B, mtime 2024-05-29).

### `fpga_turnoff_dat.tar.gz` — RAW, the original bytes

The 288 `FPGA_Data.dat` files written by the discriminator binary search, one per
(day, pixel, charge), copied verbatim from

```
<TID>/<day>/ET2p01_W36_6_VRef_SCurve_BinarySearch_TurnOff_Pixel_C<col>_R<row>_QInj_<Q>[_<tag>]/FPGA_Data.dat
```

for the four days that have them: `2024-01-11` (72 files), `2024-01-12` (72), `2024-01-15` (72)
and `2024-01-18` (72). Members are named `<day>__<original directory name>.dat`.

Each line is one probe of the binary search, seven comma-separated integers. The columns that
matter: index 3 is the TDC-hit counter, index 4 is `header_max` (the number of Level-1 accepts
issued at that probe, ≈ 11 223 — a real denominator), index 5 is the trigger-bit counter, and
index 6 is the threshold DAC probed.

### `fpga_turnoff_probes.csv` — RAW, transcribed

The same 288 files parsed into one table, 2 584 rows. Notebook 01 re-parses the tarball above and
asserts an exact match on every counter, so the transcription does not have to be taken on trust.

### `blnw_map_pickles.tar.gz` and `blnw_maps.csv` — RAW

The autocalibration baseline and noise-width maps, 16 × 16 pixels each, for the four day
directories on EOS that carry them: `2024-01-11`, `2024-01-12`, `2024-01-14` and `2024-01-15`.
The tarball holds the original pickles (2 176 B each, NumPy 16 × 16 float arrays); the CSV is the
same content flattened to `day, row, col, baseline_dac, noise_width_dac`.

**There is no map on EOS for `2024-01-16`, `2024-01-18` or `2024-01-19`**, which is why the
400 MRad baseline in notebook 01 is reconstructed from the sweep instead.

### `sweep_point_dirs.csv.gz` — RAW

One row per scanned sweep point directory: `day, row, col, charge, dac, dir_bytes`, 23 416 rows,
from a single `xrdfs ls -l` per day directory. It is what makes two things recoverable without
reading the 16 GB of hit files:

* the **hit count** per point, from the directory size, at 72.46 bytes per hit (calibrated in the
  original study against 19 exact line counts obtained server-side; residual RMS 61 hits over the
  3 500–9 400 hit range, about 1 %);
* the **censoring flag**, because the highest threshold the sweep *scanned* is then known
  independently of the highest threshold at which it recorded hits.

1 168 of the 23 416 directories are zero-byte, which is the genuine turn-off.

## What is deliberately NOT staged

* **The sweep itself** — about 16 GB of `TDC_Data_translated_*.nem` across 23 416 point
  directories under `<TID>/<day>/ET2p01_W36_6_VRef_SCurve_TDC_Pixel_*_Threshold_*/`. The summary
  feather is its per-point reduction and was verified against it exactly (six digits on all six
  mean/std quantities for one spot-checked point).
* **The per-dose overview PNGs** `ET2p01_W36_6_{0,200,400}MRad.png` in the same EOS directory.
* **`PowerHistory_v2.sqlite`** and `BaselineHistory_TID_Jan2024_CERN.sqlite` under
  `<TID>/2024-01-14_Array_Test_Results/` — not used by notebook 01.

## Which day directory is which condition

Established by matching pixel sets, charge sets and point counts.

| dose | temperature | day directories | what they contain |
|---|---|---|---|
| 0 MRad | +12 °C | `2024-01-11_Array_Test_Results` (and its identical `_PlateTemp12C_DryAir` twin) | full sweep + turn-off |
| 0 MRad | −25 °C | `2024-01-12_Array_Test_Results` | turn-off only |
| 200 MRad | −25 °C | `2024-01-15` + `2024-01-16` | full sweep; turn-off on 01-15 |
| 400 MRad | −25 °C | `2024-01-18` + `2024-01-19` | full sweep; turn-off on 01-18 |

The 0 MRad block is at **+12 °C** while both irradiated blocks are at **−25 °C**, so the summary
alone cannot separate dose from temperature. The `2024-01-12` turn-off files are the fourth corner
that breaks the degeneracy, and they are the reason they are staged.

## Not recorded anywhere in this campaign

**RFSel**, **power mode**, injections per point, and any temperature more precise than the day
directory's own tag. RFSel sets the preamplifier feedback resistance and therefore directly
affects gain and TOT; its absence is the largest unquantified systematic in notebook 01.

## How to re-fetch from source

```bash
# on an LPC interactive node, with a valid ticket
B=root://cmseos.fnal.gov//store/group/lpcmtdstudies/cern_Jan2024/TID
xrdcp -f $B/ET2p01_W36_6_TDC_summary_allMRad.feather .
xrdcp -f $B/2024-01-11_Array_Test_Results/ET2p01_W36_6_BL_map.pickle .
xrdfs root://cmseos.fnal.gov ls -l $B/2024-01-11_Array_Test_Results   # the directory census
```
