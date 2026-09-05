# Provenance — bare ET2.03 dies, CERN X-ray TID campaign, August 2024 (raw data parked on the FNAL bench PC)

**Chips** `ET2p03_BARE9` and `ET2p03_BARE7` — two ETROC2 version **2.03 bare dies, with no LGAD
bonded to either**.
**Campaign** CERN X-ray total-ionising-dose test, August 2024, at 0, 200 and 400 MRad (BARE9; BARE7's dose after 0 MRad is unknown, see the addendum).
**Temperature** −23.2 to −23.9 °C, from directory names only; `info.txt` calls it a "−25 C plate"
setpoint. The plate temperature is recorded per autocalibration map in the monitoring logs (see the addendum); there is no continuous temperature log.
**Pixels** 6: (row, col) ∈ {(0,0), (0,14), (3,0), (3,14), (12,0), (12,14)}.
**Charges** 6, 8, 10, 15, 22 fC. **Rails** VD 1.20 V for every threshold scan; VA 0.85 / 1.00 /
1.20 / 1.40 V, plus a separate 94-point VA × VD grid at a fixed 10 fC.

Everything in this directory is the output of a reduction run on the Fermilab LPC in September
2026. The reduction's own documentation is shipped here verbatim as **`DECODE_README.md`**, and
the scripts that produced the tables are in **`code/`**. This note records where the reduction's
*inputs* came from, what was deliberately left out of them, and the integrity checks that were
run — it does not repeat the format description, which `DECODE_README.md` gives in full.

## The two raw trees, and what was read

**Source A — the working copy, complete.** `fermilab-etroc3:/home/daq/tmp/TID_August/`

Streamed to LPC scratch with a single read-only `tar cf -` to stdout. `fermilab-etroc3` was
treated as **read-only throughout**: only `ls`, `du`, `find`, `head`, `cat`, `md5sum` and that one
`tar` were run there, all `nice`d; nothing on that machine was written, moved or deleted.

| | source (`fermilab-etroc3`) | copy (LPC) |
|---|---:|---:|
| files | 4 761 | **4 761** |
| bytes | 6 247 714 636 | **6 247 714 636** |

Exact match on both file count and total bytes, plus a nine-file md5 spot check that was identical
on both ends.

**Source B — the master copy, partial.** The removable exFAT USB disk **`X9`**
(`/dev/sdc1`, mounted at `/run/media/daq/X9`), tree `et2p03_qinj_data/2024-08-*_Array_Test_Results/`.
**This is where the 200 and 400 MRad money plots live** — `TID_August` has full threshold × charge
scans at 0 MRad only. Seven day directories were copied, 19 681 files and 25 103 146 145 B in
total, and the file count and byte count agree **per day directory** on both ends, with an
eleven-file md5 spot check that was identical.

Working copies of both trees, and of the reduction's outputs, are on LPC at
`/uscms_data/d3/murtazas/scratch/qinj_tid_et2p03/` (`raw/`, `raw_x9/`, `aux/`, `code/`, `out/`,
`out_x9/`; about 31 GB).

> A first md5 spot check appeared to show three of nine files differing. It did not: `find | sort`
> had been run without a fixed locale, so the two hosts' collation picked *different* files as
> "every Nth". Re-run with `LC_ALL=C` and the same explicit path list on both ends, all matched.
> It is recorded because the first result looks like data corruption and is not.

## What was NOT copied, and why

The complete `2024-08-*` set on X9 is 39.9 GB. Two day directories were dropped, deliberately, and
**both are 0 MRad** — the dose `TID_August` already covers in full:

1. **`2024-08-04_Array_Test_Results`** — 4 509 files, 5 864 572 201 B = 5.86 GB. `BARE9` at
   0 MRad, VA 1.00 / 1.20 / 1.40 V, plus a `_RoomT` control run and two `noise_Pixel` runs. The
   `TID_August` copy already holds `BARE9` 0 MRad at VA 0.85 and 1.20 V, which is what notebook 03
   uses. **The `_RoomT` control run and the 0 MRad VA 1.40 V point exist only on X9.**
2. **`2024-08-14_Array_Test_Results` (bulk)** — 7 117 files, 8 880 999 274 B = 8.88 GB. `BARE7` at
   **0 MRad**, VA 0.85 / 1.00 / 1.20 V. Its `ChipConfig` register dump *was* taken (131 408 B) and
   is one of the three in `tid_et2p03_chipconfig.csv`.

**The consequence is stated in notebook 03 and repeated here: there is no `BARE7` 0 MRad money
plot in these tables, so a BARE7 dose *difference* cannot be formed.** Only `BARE9` has 0, 200 and
400 MRad. The July day directories (an ET2.03-versus-ET2.01 DAC-scale comparison) were out of
scope and were not touched.

**X9 is removable and may not be attached next session.** The 14.7 GB above is the only part of
the August master copy that is not on LPC.

## Staged files

| file | bytes | md5 |
|---|---:|---|
| `tid_et2p03_points_all.feather` | 4,949,330 | `c1a669ff60e88b8697ef233cf0402a21` |
| `tid_et2p03_turnoff_all.csv` | 181,335 | `96883ad1b698caba6a9fa94b9cc4807b` |
| `tid_et2p03_noiseonly_all.csv` | 43,642 | `7da9e2675ef3b38aa343a5e64c69be41` |
| `tid_et2p03_configs_all.csv` | 29,923 | `1fea689e9f16a8bfb9cb2858f82a3fb0` |
| `tid_et2p03_points.feather` | 610,234 | `11832f9bef784d13102d9625180c068d` |
| `tid_et2p03_turnoff.csv` | 21,696 | `a0cf7d157ac24337bd223c57c0cc42ca` |
| `tid_et2p03_noiseonly.csv` | 5,411 | `99257c0ca258a409cf814da5a2ee80d3` |
| `tid_et2p03_configs.csv` | 24,809 | `d857637ee4f399ffab5353dd119ad860` |
| `tid_et2p03_blnw.csv` | 14,833 | `96da68fed9fef08cf920002f4bfa5023` |
| `tid_et2p03_chipconfig.csv` | 3,164 | `5aef7758fd020bddf206430880301a20` |
| `info_BARE9_voltage_scan.txt` | 1,703 | `d79ee484fd73b8d85b6c3824c96dce71` |
| `info_BARE7_voltage_scan.txt` | 1,694 | `b579ec8c8eb3ba7866e46eafb745f3e3` |
| `DECODE_README.md` | 38,569 | `d230a08548ff4d42fc75529613aa9c82` |
| `code/decoding.py` | 46,365 | `289ef6cb7b7e48589e02d817de5060ed` |
| `code/etroc2_regmap.py` | 70,593 | `6c96a43262fb8a864ebf33d5a8d752ca` |
| `code/reduce_tid.py` | 19,451 | `1ae0093c079d7e3a07d94c4b6a11da87` |
| `code/verify_hand.py` | 3,445 | `7f4012cdcaae12f0a159d26835ac2d03` |
| `code/chipconfig_to_csv.py` | 2,958 | `58499d0095f835d1e7685b7e818c8f4e` |
| `code/merge_all.py` | 2,239 | `9e66c757d9d2eb0ba6a742fb1412aee5` |
| `code/postfix_elapsed.py` | 1,568 | `91e86f3ba3549a2ee13c962c7855f473` |

3.4 MB in total. Notebook 03 prints every one of these md5s in its section 1, so the table above
is checkable rather than merely asserted.

### `tid_et2p03_points_all.feather` — RAW columns and DERIVED moments, 29 922 × 43

One row per decoded point directory. The first 22 columns carry the same names and the same
meanings as `data/hybrid_et2p01_cc_jan2024/cc_points.feather`, with `hv_state` / `power_tag`
(meaningless on a bare die) replaced by the axes this campaign actually varied — `dose_MRad`,
`va_V`, `vd_V`, `elapsed_h`.

`toa_mean`, `toa_std`, `tot_mean`, `tot_std`, `cal_mean`, `cal_std` are a **plain mean and sample
standard deviation (`ddof = 1`) over every hit recorded at that pixel** — no cuts, no CAL window,
no outlier rejection, no time ordering, no BCID selection — in **raw TDC codes, not nanoseconds**.
That is the same definition the two January feathers use, which is what makes the three
differenceable, and it is unchanged.

Nine further columns carry the same moments over a **per-hit CAL window**: `n_hits_cal`,
`frac_rejected`, `cal_mode` and `toa_mean_cal`, `toa_std_cal`, `tot_mean_cal`, `tot_std_cal`,
`cal_mean_cal`, `cal_std_cal`. The criterion is the repository test-beam pipeline's own,
`|CAL − cal_mode| ≤ 3` codes against the modal CAL of that pixel at that point
(`core/path_finder.py`, `core/extract_events_by_path.py`). It exists because a small fraction of
this campaign's hits carry a grossly wrong CAL — in contiguous blocks of a few threshold DAC
values at a fixed absolute DAC, the same block at every injected charge, which is a decode
artefact and not front-end behaviour — and one TOA code is `3.125 ns / CAL`, so those hits carry a
wrong TOA. Over all 29 828 money-plot points the window removes **2.68 % of hits**, and 19.4 % of
points carry `cal_std > 2` codes on the raw moments. `DECODE_README.md` gives the per-condition
and per-pixel before/after; notebook 03 draws it in section 4.4 and quotes the windowed moments.

**The only selection anywhere in the reduction** is `row == <row in the directory name> and
col == <col in the directory name>`, and it is recorded rather than hidden: `n_hits_all_pixels`
and `n_pixels_seen` say exactly how much it removed. Over all 29 828 money-plot points it removes
**49 194 hits out of 239 969 849, i.e. 0.021 %**, and notebook 03 recomputes both totals.

The 94 `scan_family == 'voltage_scan'` rows are a different animal: one fixed, **unrecorded**
threshold each (`dac` is `NaN`), sixteen pixels pooled with no selection applied (`row`/`col` are
`NaN`). They must never be mixed with the threshold-scan rows in one fit.

### `tid_et2p03_turnoff_all.csv` — 450 binary searches

`turnoff_dac` is **RAW**: the DAC on the last line of `FPGA_Data.dat`, which is exactly the
campaign's own definition — `notebook_helpers.charge_peakDAC_plot` reads
`int(last_line.split(',')[-1])` and plots that against charge. `search_trace` carries the whole
search as `dac:triggerbits;…`, which is what lets notebook 03 re-derive the 50 % crossing with the
same estimator notebooks 01 and 02 use and show where the two definitions agree.

### `tid_et2p03_noiseonly_all.csv` — 90 `quietCB` noise scans

A dense one-DAC-step scan of the trigger-bit counter across the noise peak, taken inside every
configuration. This is what gives notebook 03 a baseline and a noise width at **every** condition,
which neither January campaign has.

### `info_BARE9_voltage_scan.txt`, `info_BARE7_voltage_scan.txt` — the campaign logbook, RAW

Copied byte for byte from `fermilab-etroc3:/home/daq/tmp/TID_August/ET2p03_BARE{9,7}_Voltage_scan/
info.txt` (via the LPC copy of that tree). They are the campaign's **only** contemporaneous prose
record, and their first two lines are load-bearing for every voltage label in this folder:

    Nominal Analog 1.2V = 1.4V on power supply
    Nominal Digital 1.2V = 1.25V on power supply

i.e. `va_V` and `vd_V` in every table here are **power-supply settings**, and the intended
operating point on both rails was 1.2 V *at the chip*: a 0.20 V analog and 0.05 V digital IR drop
over cables and board. The rail currents were never logged, so converting a supply setting to an
at-chip voltage by subtracting a constant drop is an approximation good to roughly ±50 mV.
The rest of each file is the contemporaneous record of the VD 0.90 / 0.95 V (supply) failures that
notebook 03 section 12 reproduces by decoding those sub-runs blind. The two files differ only in
their 0 MRad VD 0.90 V line.

### `tid_et2p03_blnw.csv` and `tid_et2p03_chipconfig.csv`

Byte-identical to `data/census/aux_et2p03_blnw.csv` and `data/census/aux_et2p03_chipconfig.csv`,
which notebook 00 reads. They are staged in both places so that each notebook's data directory is
self-contained; notebook 03 prints both md5s side by side to prove the duplication is exact.

The register dumps say `VrefGen_PD = 0`, `RefStrSel = 3` and `TS_PD = 0`, identically on all
three. **`VrefGen_PD = 0` proves the internal reference generator was powered up. It does not
prove the internal reference was used** — internal versus external VRef is board wiring and has no
register bit. `TS_PD = 0` likewise says the on-chip temperature sensor had power, not that anyone
read it; **no VTemp value is stored anywhere in this campaign.**

### `code/` — the reduction scripts, shipped for provenance

`decoding.py` is this repository's own `DecodeBinary`, copied byte for byte from
`TestBeam/condor_at_lxplus/core/decoding.py` (md5 `289ef6cb7b7e48589e02d817de5060ed`), and it is
what decoded the `.bin` stream. Notebook 03 prints that md5 against the repository's.

The reduction is verified three ways, all documented in `DECODE_README.md` §6: an **independent
decoder written from the format description alone** reproduces the production TOA/TOT/CAL means to
every digit printed on a 0 MRad point and on a dosed `BARE7` point (and differs by one hit in
19 389 on one dosed `BARE9` point, where `DecodeBinary`'s trailer hit-count check rejects an event
the unvalidated reference decoder keeps); the decoded pixel identity matches the directory name on
all 29 828 money-plot points; and decoding the low-VD rail sub-runs blind returns **exactly zero
hits**, reproducing what `info.txt` independently says happened there.

## What this campaign did not record

No power-supply or LV-current log. No temperature log. No `PowerHistory` sqlite. No baseline
history — `ETROC-History/BaselineHistory.sqlite` on `etroc3` was checked directly and its earliest
row is 2025-01-30, with no `BARE7`/`BARE9` rows at all. No per-run json, toml, yaml or csv of any
kind. No VTemp. **No RFSel**, on any die, in any name or any table.

The only record of the supply currents anywhere is one sentence of prose in `info.txt`: *"WS
current is down at 9.X mA level … it was not so during the Voltage / money plots after 200 MRad
when we saw large large noise peak measurements. Is this a related phenomenon?"* That is the whole
record, and it is why the auxiliary sections of notebooks 00 and 02 have no counterpart here.

## Raw data that is NOT staged, and where it lives

| what | size | where |
|---|---|---|
| `TID_August`, the complete working copy (0 MRad money plots + all rail scans, undecoded) | 6.2 GB | `fermilab-etroc3:/home/daq/tmp/TID_August/`; LPC copy at `/uscms_data/d3/murtazas/scratch/qinj_tid_et2p03/raw/` |
| The X9 day directories that carry the dosed money plots | 25.1 GB | X9 `et2p03_qinj_data/2024-08-{03,07,08,11,12,21}`; LPC copy at `…/raw_x9/` |
| `2024-08-04` — `BARE9` 0 MRad, VA 1.00/1.20/1.40 V, the `_RoomT` control run | 5.86 GB | **X9 only** |
| `2024-08-14` — `BARE7` 0 MRad money plots | 8.88 GB | **X9 only** |


## Addendum 2026-09-04 — monitoring logs staged, campaign relabelled, BARE7 dose corrected

**Campaign location.** The X-ray irradiation was done at **CERN** (the history folder is named `CERN_Aug_2024_TID`, its
sister climate-chamber folders are `CERN_Aug_2024_CC_*`); the raw data were later parked on the FNAL bench PC, which is
why earlier text here called it an FNAL bench campaign. Figure headers and prose now say CERN X-ray.

**BARE7 dose.** The `2024-08-21` acquisition of `ET2p03_BARE7` is tagged `200MRad` in its directory name and in the
baseline-history notes, but the X-ray source failed before BARE7 reached 200 MRad (operator statement, 2026-09-04). The
staged `*_all` tables were edited in place on 2026-09-04: for those rows (3023 points, 30 turn-off searches, 6 noise scans,
1 config) `dose_MRad` is `NaN`, `dose_source` is `unknown_xray_failure`, and a `dose_note` column carries the reason. No other
value changed. Notebook 03 treats BARE7 in isolation (section 14).

**`aux/` — the campaign's monitoring logs.** Supplied by the campaign operator as `CERN_Aug_2024_TID.zip` (md5
`d9074110da3bf4f15a12120f65dd0d5d`, 23.5 MB) on 2026-09-04; the unpacked folder is on LPC EOS at
`/eos/uscms/store/user/lpcmtdstudies/cern_Aug2024_TID/ETROC-History/CERN_Aug_2024_TID/` (8 files, 133 MB). What is staged here
is downsampled to one sample per minute (medians) or grouped per autocalibration map; the full-rate originals are on EOS.

| file | bytes | md5 | what it is |
|---|---|---|---|
| `aux/aux_tid_adc_1min.csv.gz` | 217,172 | `71fe7d2633261b777457f54b59ff34c7` | RAW, downsampled. External ADC channels 0/1/7/8, 1-minute medians of voltage and the logger's calibrated value; channel 7 = VTemp pin, channel 8 = room sensor |
| `aux/aux_tid_baseline_maps.csv` | 25,860 | `f07f05c3231ad51240c7c79cdfe7dffc` | DERIVED. One row per autocalibration map (rows grouped by chip, note and 10-minute window): n_pix, mean/std of baseline and noise width |
| `aux/aux_tid_baselines_full.csv.gz` | 380,071 | `21f9411b14eb98845d41ea123976571f` | RAW, verbatim. All 34 304 rows of BaselineHistory.sqlite (per-pixel baseline and noise width with the save_notes) |
| `aux/aux_tid_baselines_pre_campaign.csv.gz` | 50,393 | `827473a742635a23dd8d398ad2bfa5ef` | RAW, verbatim. BaselineHistory_need_to_be_merged_later.sqlite: pre-campaign maps of eleven chips, 07-23..08-03 |
| `aux/aux_tid_irradiation_timeline_plot_ipynb.csv` | 150 | `f2e2b479993f5aa54af760cfbc334708` | RAW, transcribed. The four irradiation start/stop markers from the campaign's plot.ipynb |
| `aux/aux_tid_power_1min.csv.gz` | 375,414 | `cebc3dd8c377510657afeb429ffb7023` | RAW, downsampled. Analog/digital/WS rail V and I, 1-minute medians, from the three PowerHistory files (source_file column) |
| `aux/aux_tid_power_actions.csv` | 26,408 | `2708d6e5b9298258c69be0838b0c33f2` | RAW, verbatim. The 578 supply on/off/logging actions |
| `aux/aux_tid_vref_1min.csv.gz` | 184,059 | `87bbde4d5cfb12019023f87e058ca517` | RAW, downsampled. VRef pin of the chip under test, 1-minute medians (dropouts ≤ 0.5 V counted separately), from vol_monitoring.sqlite |

**Edited table md5s after the BARE7 relabel:**

| file | bytes | md5 |
|---|---|---|
| `tid_et2p03_points_all.feather` | 4,965,266 | `34ec6001da66d9a48103cc9981a70614` |
| `tid_et2p03_turnoff_all.csv` | 186,895 | `482b9e457aae1d32a385fa23f3316e14` |
| `tid_et2p03_noiseonly_all.csv` | 44,762 | `e18213b72eed3cbd5d27ff793955cd6b` |
| `tid_et2p03_configs_all.csv` | 30,212 | `11a3cbc4035b63c6a6c42a11f7f3870e` |
