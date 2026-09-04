# Provenance — the data census and the auxiliary logs

Everything notebook `00_census_and_aux_data.ipynb` reads is in this directory. Two of the files are
**inventories** (a transcription of what two read-only storage scans found); five are **extractions
or copies** of auxiliary data that no other notebook in this folder plots.

Nothing here was produced by writing to a facility. LPC EOS was read with `ls` and `xrdcp`; the
sqlite files were copied to LPC scratch with `xrdcp` and queried read-only
(`sqlite3.connect("file:…?mode=ro", uri=True)`). No DAQ machine was touched for this notebook.

## Files

| file | bytes | md5 | RAW / DERIVED |
|---|---:|---|---|
| `qinj_census.csv` | 12,751 | `c7b4d74c93118ee7e8c81e62f269de39` | DERIVED — inventory |
| `qinj_aux_matrix.csv` | 5,412 | `267d5af1ff9e49f7299f0686c7f790af` | DERIVED — inventory |
| `aux_tid_lv_timeseries.csv.gz` | 153,597 | `6672b01ca9962224c36ba6873b8e0150` | RAW, downsampled |
| `aux_tid_lv_actions.csv` | 522 | `175bc9762152207a1afe27adac3115fe` | RAW |
| `aux_tid_baseline_history.csv` | 84,803 | `60f056bf91b1444efce65208a7a994ef` | RAW |
| `aux_et2p03_chipconfig.csv` | 3,164 | `5aef7758fd020bddf206430880301a20` | RAW, decoded |
| `aux_et2p03_blnw.csv` | 14,833 | `96da68fed9fef08cf920002f4bfa5023` | RAW |
| `aux_ccts_vol.csv.gz` | 175,344 | `2e1b161c93115e84448e89672496a37c` | RAW, downsampled |
| `aux_ccts_chamber.csv.gz` | 59,554 | `a4e0ee163be1683124e9f7cf1a30c1ff` | RAW, downsampled |
| `aux_ccts_lv.csv.gz` | 224,893 | `b8e8731283c5dac6e56f219690935964` | RAW, downsampled |
| `aux_ccts_lv_actions.csv` | 21,962 | `90a1ac41ea65c22cd22fb7adbce3bb61` | RAW |
| `aux_ccts_blnw.csv.gz` | 390,706 | `421f81df91dcad68ac075e627fd510b1` | RAW |
| `aux_tid_et2p03_timeline.csv` | 738 | `bfa86d85a080a291e4e6e559fdc938db` | RAW, transcribed |
| `census_x9_history_folders.csv` | 1,436 | `f53ffc3f2aa1ebe12f204c5d1a901f3e` | RAW |

*(The two md5 values in the first two rows are recomputed by the notebook itself, cell 4.1, so a
disagreement is visible in the executed output rather than taken on trust.)*

---

## `qinj_census.csv` — 32 rows × 24 columns

One row per data set, including the negative rows: places where charge-injection data would be
expected and there is none. Negatives are load-bearing, so they are rows rather than footnotes.

**Where the content comes from.** Rows **1-16** are a transcription of a read-only scan of
**LPC EOS `/store/group/lpcmtdstudies`** and the FNAL bench PC **`fermilab-etroc3`**, completed
2026-09-02. Rows **17-24** are a transcription of a second read-only scan, completed 2026-09-04,
covering **lxplus EOS and AFS** and **`fermilab-etroc5`**. Rows **25-31** are the same second
scan's pass over the seven **LPC EOS `SEU_*` trees**, split here into one row per tree so that the
five clean negatives are as visible as the two positives.

**What was re-measured for this notebook**, on LPC EOS on 2026-09-04, and recorded in the
`verified_here` column:

* the dose tag and scan-kind counts of **every one of the eleven day directories** of
  `cern_Jan2024/TID/`, by parsing the scan-directory names (`… _QInj_<n>_<tag>`);
* that `2024-01-11_Array_Test_Results` and `2024-01-11_Array_Test_Results_PlateTemp12C_DryAir`
  have **identical listings and are 4.9 GB each** — the same day stored twice;
* that the **DAC 48-59** scans of `2024-01-05` are on chip **`Pair1`**, not on `ET2p01_W36_6`.

**Column conventions.** `size_GB` is blank where no size was measured (`size_note` says why);
`unknown` means a scan found no evidence and is never a guess; `has_threshold_scan` distinguishes
`yes` from `no` and from `no (single point)`; `analysed_in_notebook` names the notebook in this
folder that uses the set, and is blank where nothing here does.

---

## `qinj_aux_matrix.csv` — 15 rows × 13 columns

One row per campaign, one column per kind of monitoring: `lv_supply_log`, `blnw_maps`, `vtemp`,
`hv_log`, `vref_record`, `chamber_log`, `register_dump`, `logbook`. Values are
**`yes` / `partial` / `no` / `na` / `unknown`**.

`na` and `unknown` are not the same thing and neither is `no`. `na` means the question does not
arise (an HV log on a bare die). **`unknown` means the scan did not check** — it is a hole in the
census, not a negative result, and it must not be read as `no`. The `files` column names the
artefacts, and `note` carries the caveat that goes with the row.

---

## `aux_tid_lv_timeseries.csv.gz` — RAW, downsampled

The **analog and digital low-voltage rails of the CERN January-2024 TID campaign**, from the three
`PowerHistory_v2.sqlite` files inside that tree:

```
root://cmseos.fnal.gov//store/group/lpcmtdstudies/cern_Jan2024/TID/
    2024-01-10_Array_Test_Results/PowerHistory_v2.sqlite   180,224 B  md5 86bc446cdedea69940465b055adfabb2
    2024-01-12_Array_Test_Results/PowerHistory_v2.sqlite 5,066,752 B  md5 002875a41fb33facbd90d6940a240cb0
    2024-01-14_Array_Test_Results/PowerHistory_v2.sqlite 9,981,952 B  md5 4ff9c12dc72b078da14871a8b5532b77
```

Table `power_v2(timestamp, V, I, Instrument, Channel, channel_id)`, **231,120 rows** in total
(2,574 + 77,758 + 150,788), covering 2024-01-10 22:40 → 2024-01-19 06:53 with one gap
(2024-01-11 00:27 → 2024-01-12 16:02). Exactly two channels ever appear, `Analog` (`channel_id` 1)
and `Digital` (2), at about **5 s** cadence; `Instrument` is the constant `'Power'`, so there is no
high-voltage, chiller or temperature channel in these files — the logger only ever had the two
low-voltage rails wired.

**The values are strings with the unit baked in** (`'1.238V'`, `'0.3394A'`), so they are parsed,
not cast. Extraction, run on LPC against the `xrdcp`'d copies:

```python
import sqlite3
con = sqlite3.connect("file:PowerHistory_v2.sqlite?mode=ro", uri=True)
for ts, V, I, inst, ch, cid in con.execute(
        "select timestamp,V,I,Instrument,Channel,channel_id from power_v2 order by timestamp"):
    v = float(str(V).rstrip("V")); i = float(str(I).rstrip("A"))
    # keep the FIRST sample of each (Channel, minute)
```

**One sample per rail per minute is kept**, 19,268 rows of 231,120. The downsampling was validated
against the full tables: the per-day median current, computed over powered samples
(`V > 0.5 V`) on both the full and the kept series, agrees for all twenty (day, rail) pairs to
**within 0.0001 A**, the worst case being 2024-01-10 Analog.

Samples below 0.5 V are power-down stretches. They are **kept in the series** — the notebook plots
them — and excluded only where a median is taken, never silently.

## `aux_tid_lv_actions.csv` — RAW

The `actions_v2(timestamp, system, type, message)` rows of the same three files, 7 rows. These are
what date the power-down stretches, and they show the logger **outliving the data**: the last Stop
is 2024-01-22 18:00, three days after the last scan day.

## `aux_tid_baseline_history.csv` — RAW

`BaselineHistory_TID_Jan2024_CERN.sqlite`
(`/…/cern_Jan2024/TID/2024-01-14_Array_Test_Results/`, 98,304 B,
md5 `b4d88ec07735aef6aeeb8a8e81642cff`), table
`baselines(row, col, baseline, noise_width, timestamp, chip_name, note)`, **all 1,024 rows**
verbatim. It is two full-array autocalibration repeats on `ET2p01_W36_6`, 2024-01-14 22:23 →
22:46, noted `Run{0,1}_postTID_overnight_BL_test`. That is the whole of this campaign's baseline
log: **23 minutes out of a 15-day campaign.** It has no HV and no temperature column.

## `aux_et2p03_chipconfig.csv`, `aux_et2p03_blnw.csv` — RAW

Copies, **byte for byte**, of two tables of the ET2.03 August-2024 reduction (the input to
notebook 03), staged here so that notebook 00 reads only files under `data/`. The md5 values above
are the reduction's own. The first is the decode of the three `ChipConfig/*.pckl` register
read-backs taken on 2024-08-08, -14 and -21; the second is the `ET2p03_BARE9` autocalibration
baseline and noise-width maps of 2024-08-03, all 256 pixels.

---

## What is deliberately NOT here

* **The auxiliary logs of the climate-chamber campaign** (its own two `PowerHistory_v2.sqlite`
  files and its baseline sqlite). Those are staged under
  `data/hybrid_et2p01_cc_jan2024/` and plotted by notebook 02; notebook 00 does not duplicate them.
* **Any raw scan data.** This directory holds inventories and monitoring only.
* **The `SEU_*` auxiliary logs** — the largest low-voltage log in the census, the only three-rail
  log, and the only numerical `vref` column all live in those trees and are recorded in the aux
  matrix as `yes` without being extracted. They belong to campaigns that took no threshold scan,
  so nothing in this folder analyses them yet.

---

## The CERN August-2024 climate-chamber (CTS) monitoring set — census row 32

`aux_ccts_vol.csv.gz`, `aux_ccts_chamber.csv.gz`, `aux_ccts_lv.csv.gz`,
`aux_ccts_lv_actions.csv`, `aux_ccts_blnw.csv.gz`.

**Origin.** Four SQLite files in the per-campaign history folder

```
fermilab-etroc3:/run/media/daq/X9/etroclaptop3_backup_20251127/ETROC-History/CERN_Aug_2024_CC_CTS/
```

a backup of the `etroclaptop3` `ETROC-History` tree taken on 2025-11-27 onto the removable X9
disk. **The copy did not preserve timestamps**, so every mtime on that disk is meaningless; all
dates below come from timestamp columns inside the files. The four files were staged read-only
(no writes on the DAQ machine, no `du`, listing and copying only) and their checksums are:

| source file | bytes | md5 |
|---|---:|---|
| `vol_monitoring.sqlite` | 14,082,048 | `9ad5677688ce386a2a1338fdcd070ead` |
| `PowerHistory_v2.sqlite` | 22,913,024 | `7dc1fcdf14107f28ad4716bd5ebccecc` |
| `ti1080_sensor_data.sqlite` | 3,289,088 | `3d8a72a03b79ace094da539b8355144a` |
| `BaselineHistory.sqlite` | 3,194,880 | `5c2166f3901cb626c0b5fa993e69a366` |

**What the campaign is.** `ET2p03_BARE4`, a **bare ET2.03 die**, in the CERN climate chamber
2024-08-04 17:27 → 2024-08-14 16:39, cycled +23 → −30 → +24 °C. **It took no charge-injection
data, and it was never irradiated** — all 140 autocalibration maps carry `0 MRad` in their
condition text. It is staged here because it is the only place in either census where an ET2.03
chip's **internal voltage reference** and its **on-chip temperature sensor** were logged against a
real temperature sweep.

**Tables and downsampling.**

* `vol_monitoring(timestamp, voltage, Instrument)` — `Instrument` ∈ {`VRef`, `VTemp`}, 168,083
  samples each at about 5 s, 336,166 rows in total. **One sample per instrument per minute kept**,
  28,030 rows. Voltages are bench-multimeter readings in volts and are stored as REAL, so unlike
  the January-2024 power logs there is no unit string to strip.
* `power_v2(timestamp, V, I, Instrument, Channel, channel_id)` — 350,489 rows. **Three**
  instruments appear and they do not overlap in time: `Power1` (2024-08-04 → 08-13) and then
  `Power` together with `WS_Power` (08-13 → 08-14), and `channel_id` is not consistent between
  them, so the series must be grouped by `(Instrument, Channel)` and never by `channel_id`.
  `WS_Power` has an `Analog` channel only. Values are strings; the unit is stripped at extraction.
  **One sample per (instrument, rail) per minute kept**, 29,295 rows. `actions_v2` is staged whole,
  469 rows.
* `ti_hdc1080(timestamp, temperature, humidity, which_hdc1080)` — 85,779 rows, one sensor
  (`which_hdc1080 = 1`). **One sample per minute kept**, 14,325 rows. This is chamber *air*, not a
  cold plate and not the die.
* `baselines(col, row, baseline, noise_width, timestamp, chip_name, save_notes)` — 35,840 rows =
  **140 complete 16 × 16 maps**, staged whole. The `save_notes` free text is the campaign's own
  condition record (`+20.3C, 0 MRad, A: 1.2V, D: 0.9V`): 90 distinct values, **all 90 parse**, and
  the four parsed fields are added as `note_temp_C`, `note_dose_MRad`, `note_va_nominal_V`,
  `note_vd_nominal_V`. The rail figures in the note are **not** the supply-terminal readings —
  notebook 00 cell 7.6h pairs each map with the supply reading nearest in time and shows the offset
  between them is constant over all seven settings on each rail.

Samples outside a physical window (`VRef` 0.5–1.2 V, `VTemp` 0.3–1.0 V) are the multimeter reading
an open channel or a powered-down board. They are **kept in the staged series and plotted**, and
excluded only where a median or a fit is taken.

## `aux_tid_et2p03_timeline.csv` — RAW, transcribed

The irradiation start/stop markers of the ET2.03 August-2024 dose campaign, read off
`ETROC-Analysis/reading_history/cern_TID.ipynb` on `fermilab-etroc3` (last modified 2025-06-02,
saved with no outputs). That notebook reads five SQLite files of a
`ETROC-History/CERN_Aug_2024_TID_history/` folder that this census has not located; notebook 00
section 7.7 lists where it was searched. The `source_in_notebook` column names the cell each marker
came from, and records the one place where the notebook disagrees with itself (the end of the
third step: 01:30 in its prose comment, 02:30 in a commented-out plotting array). Both are kept.

## `census_x9_history_folders.csv` — RAW

One `ls` per per-campaign history folder of the same X9 backup, run read-only on 2026-09-04:
folder name, entry count, CSV count, the SQLite basenames present, and the subdirectories. Nineteen
folders. Only `CERN_Aug_2024_CC_CTS` is analysed anywhere in this repository; the rest are listed
so that "was anything monitored during campaign X" is answerable without repeating the search.
