# Provenance — ET2.01 + LGAD hybrid, CERN climate chamber, January 2024

**Chip** `ET2p01_Bar4` — an ETROC2 version 2.01 **bonded to an LGAD** (a hybrid module).
**Campaign** CERN climate chamber, January 2024, at −25, 0 and +22 °C, with the sensor bias at
210 V, at 0 V with the cable still plugged in, or not recorded.
**Pixels** 9 on the flagship configurations: rows {2, 8, 14} × columns {2, 8, 14}.
**Charges** 5, 6, 8, 15, 22, 27 fC.

The raw tree is on Fermilab LPC EOS at

```
root://cmseos.fnal.gov//store/group/lpcmtdstudies/cern_Jan2024/CC/
```

and a 12.2 GB working copy lives on LPC at `/uscms_data/d3/murtazas/scratch/qinj_cc/`. Neither is
staged here. What is staged is the **reduction** of that tree plus two auxiliary logs.

## Staged files

| file | bytes | md5 |
|---|---|---|
| `cc_points.feather` | 1,187,466 | `a0f99500063c9f99c2215f96e1b624ee` |
| `cc_fpga_probes.csv.gz` | 126,377 | `ac52fd05c8a0b1e80c8310ae72d088df` |
| `cc_tdc_dirs.csv.gz` | 118,901 | `6b7f1c13738436760bd6e958c13f3932` |
| `cc_turnoff.csv` | 101,248 | `51a919ee1bfe5653270389a894b7494b` |
| `cc_blnw.csv.gz` | 44,320 | `5ce2d9cb41b57815c046042b58dece69` |
| `cc_noiseonly.csv` | 44,755 | `313eb6c0dda21af7f6aff7facf8d4988` |
| `cc_configs.csv` | 10,995 | `0418c23834af8933eaa2f3f0ffcdff02` |
| `aux_lv_timeseries.csv.gz` | 161,004 | `df851467b6e05bec42f13e571c700e1d` |
| `aux_baseline_history.csv.gz` | 103,395 | `56d65ceec8df9e24eb29a4ba6e50ca8a` |
| `table_aux_power_lv.csv` | 1,237 | `2f1d6e88c512609fe42e46fcf2702488` |
| `summary_numbers.json` | 39,915 | `9cfd699fc55d6a6a9a815afba4ab3c0e` |

The `cc_*` files were produced by the campaign reduction (`reduce_cc.py` and `dump_probes.py`,
which live with the LPC working copy) and are copied here unchanged apart from gzip on the four
largest. The three `aux_*` files were extracted for this notebook set on 2026-09-04 by the
recipes given below. `table_aux_power_lv.csv` and `summary_numbers.json` are the previously
published derived summaries and are used **only** as cross-check targets in notebook 02.

### `cc_points.feather` — RAW

13 281 rows × 22. One row per (config, pixel, charge, threshold DAC), carrying the per-point TOA,
TOT and CAL mean and standard deviation in raw codes, the recorded hit count, and the parsed
configuration metadata. No cuts were applied when it was built, mirroring the bare-die summary.

### `cc_fpga_probes.csv.gz` — RAW

Every probe point of every binary search: DAC, trigger-bit count, TDC count, and `header_max`
(the Level-1 accept denominator). This is what lets notebook 02 fit S-curves and recompute the
turn-off itself rather than take `cc_turnoff.csv` on trust.

### `cc_tdc_dirs.csv.gz` — RAW

One row per scanned point directory, **including the ones that produced no hits** (869 of 14 150
readable directories, 6.1 %). This is what makes the censoring flag computable: the highest
threshold scanned is known independently of the highest threshold with hits.

### `cc_blnw.csv.gz`, `cc_noiseonly.csv` — RAW

The autocalibration baseline and noise-width maps (all 256 pixels of 19 maps) and the
in-configuration NoiseOnly scans. The campaign **renamed the chip whenever the bias changed**, so
a map found at a day's root may belong to a different bias state than the scan taken that day;
`applies_to_configs` records the attribution and notebook 02 cross-checks every one of them.

### `cc_configs.csv` — RAW metadata + DERIVED attribution

The configuration inventory. Temperature, bias state and power tag are **parsed from directory
names only**; each carries its own `*_source` column saying so.

### `aux_lv_timeseries.csv.gz` — RAW, downsampled

The low-voltage rails from the two `PowerHistory_v2.sqlite` files (table `power_v2`, two channels
`Analog` and `Digital`, sampled about every 5 s). Together they span
**2024-01-12 17:53 → 2024-01-19 13:54, 233 576 samples**. Unit strings (`1.279V`, `0.4944A`) are
stripped at extraction. **One sample per rail per minute is kept**, 19 470 rows, which reproduces
every published per-day median to within 0.001 A.

Extraction (run on LPC, against the working copy):

```python
import sqlite3, glob
for f in sorted(glob.glob("raw/*/PowerHistory_v2.sqlite")):
    con = sqlite3.connect(f)
    for ts, V, I, ch in con.execute("select timestamp,V,I,Channel from power_v2 order by timestamp"):
        v = float(str(V).rstrip("V")); i = float(str(I).rstrip("A"))
        # keep the first sample of each (channel, minute)
```

### `aux_baseline_history.csv.gz` — RAW, aggregated

`BaselineHistory_CC_Jan2024_CERN.sqlite`, table `baselines`
(`row, col, baseline, noise_width, timestamp, chip_name, note`), 33 170 rows, averaged per
(`note`, minute) over all pixels and separately over the nine primary-configuration pixels;
8 459 rows result.

The log covers **2024-01-17 → 2024-01-19 only**, so it reaches the two flagship configurations
and none of the −25 °C ones. Its schema has **no HV column and no temperature column**, and no
`note` value contains an HV token; the `note` records the run and a coarse thermal condition
(`Run<n>_m25_Jan_17_2024`, `Run<n>_Overnight0C_...`, `..._CCoff_dooropen_...`) and nothing else.
Its absolute baselines are not directly comparable with the per-configuration maps, because the
autocalibration settings differ (the `time_3s` in the note is the accumulator time).

## What the campaign did NOT record

Searched: every directory name in the raw tree, both `PowerHistory_v2.sqlite` files, the
`BaselineHistory` sqlite, and every `*.json / *.yaml / *.cfg / *.txt / *.log` in the tree.

| quantity | present? |
|---|---|
| Low-voltage rails (V, I) | **yes** |
| Autocalibration baseline / noise-width history | **yes** |
| High voltage (sensor bias) | **no** — no HV channel, column or token anywhere |
| On-chip temperature sensor (VTemp) | **no** |
| Chamber or plate temperature log | **no** — only the coarse directory tag |
| Voltage reference as a monitored value | **no** — ET2.01 took its 1.0 V reference externally |
| RFSel | **no** |
| Injections per point | **no** — the sweep acquires for a fixed time |

**Scan windows are not recoverable.** Directory modification times are 2024-01-27 on EOS (the
upload) and 2026-09 on the LPC copy (the copy), not acquisition times, and no scan file carries a
wall-clock stamp. The two sqlite logs hold the only genuine timestamps in the campaign, which is
why the auxiliary figures in notebook 02 anchor on those and on calendar days rather than drawing
a scan window.

## Known holes in the reduction

* **2024-01-13 and 2024-01-14 wrote only undecoded binary** `TDC_Data_*.bin`, which the reduction
  does not decode. Their turn-off and NoiseOnly data is present, which is why those
  configurations appear in the gain table with no sweep-derived σ and no jitter.
* **The 2024-01-13 `m25c_v7` tag (8.30 GB) was not copied** — a third −25 °C repeat, binary
  anyway, so it would have produced no rows.
* **`temperature_C` is a mixed-type column** (integer or the string `not recorded`) in every CSV.
  In `cc_points.feather` it happens to be clean `int64`; do not assume that on a re-run.
