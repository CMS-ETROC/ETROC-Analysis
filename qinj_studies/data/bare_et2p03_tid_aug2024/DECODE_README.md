# `TID_August` — ETROC2 **ET2.03** bare-chip TID charge-injection set, reduced to per-point tables

**What this is.** A reduction of the FNAL bench **August-2024 TID (X-ray dose) charge-injection
campaign** on two **ETROC2 v2.03 bare dies**, `ET2p03_BARE9` and `ET2p03_BARE7`. It is the ET2.03
counterpart of the ET2.01 dose series in row 1 of `notes/qinj-money-plot-census.md`, and it is built
to the **same definitions** as `~/ETROC_viz_mirror/qinj_hybrid_cc/cc_points.feather`, so the three
can be differenced directly.

Reduction date **2026-09-04** (extended the same day to the X9 dosed money plots). `fermilab-etroc3` was treated as **read-only** throughout: only `ls`,
`du`, `find`, `head`, `cat`, `md5sum` and one `tar cf -` to stdout were run there, all `nice`d; nothing
was written, moved or deleted on that machine. Working copies live on LPC under
`/uscms_data/d3/murtazas/scratch/qinj_tid_et2p03/`.

Every quantity below is labelled **RAW** (a field of, or a byte count of, the source) or **DERIVED**
(computed by `reduce_tid.py`). Where the campaign did not record something, the tables say
**`not recorded`** — never a guess, never an inferred value.

---

## 1. Provenance — exactly what was read and what was copied

**Source A (the working copy, complete).** `fermilab-etroc3:/home/daq/tmp/TID_August/`

Streamed to LPC with `ssh fermilab-etroc3 'nice -n 19 tar cf - -C /home/daq/tmp TID_August' | ssh lpc 'tar xf - -C .../raw/'`
(11 min wall). Local copy, kept: `/uscms_data/d3/murtazas/scratch/qinj_tid_et2p03/raw/TID_August/`

| | source (etroc3) | copy (LPC) |
|---|---:|---:|
| files | 4 761 | **4 761** |
| bytes | 6 247 714 636 | **6 247 714 636** |

**Integrity: exact match on both file count and total bytes**, plus a 9-file md5 spot check
(every 600th file in sorted order) that was **identical on both ends**. RAW.

**Source B (the master copy, partial).** The removable **exFAT USB disk `X9`** (`/dev/sdc1`,
mounted at `/run/media/daq/X9`), tree `et2p03_qinj_data/2024-08-*_Array_Test_Results/`.
This is where the **200 and 400 MRad money plots live** — `TID_August` has full charge scans at
**0 MRad only** (see §3). Copied to `.../raw_x9/`:

| day dir | contents | files | bytes |
|---|---|---:|---:|
| `2024-08-03` | `ET2p03_BARE9` BL/NW autocal maps (pickles) | 38 | 13 633 |
| `2024-08-07` | BARE9 **200 MRad**, VA 0.85 / 1.00 V | 3 220 | 4 175 587 194 |
| `2024-08-08` | BARE9 **200 MRad**, VA 1.20 / 1.40 V + `ChipConfig` | 5 034 | 6 426 276 980 |
| `2024-08-11` | BARE9 **400 MRad**, VA 0.85 / 1.00 / 1.20 V | 5 633 | 7 159 604 619 |
| `2024-08-12` | BARE9 **400 MRad**, VA 1.40 V | 2 682 | 3 452 428 566 |
| `2024-08-21` | BARE7 **200 MRad** + `ChipConfig` | 3 073 | 3 889 103 745 |
| `2024-08-14/ChipConfig` | BARE7 0 MRad register dump only | 1 | 131 408 |
| `2024-08-14` (bulk, **added later**) | BARE7 **0 MRad**, VA 0.85 / 1.00 / 1.20 V | 7 117 | 8 880 999 274 |
| **total** | | **26 798** | **33 984 145 419 = 34.0 GB** |

**Integrity of the X9 copy: exact match, per day directory, on both file count and total bytes**
(the table above was measured on *both* ends and agrees line for line), plus an **11-file md5 spot
check that was identical on both ends**. RAW.

> A first md5 spot check appeared to show 3 of 9 files differing. It did not: `find | sort` had been
> run without a fixed locale, so the two hosts' collation picked *different* files as "every Nth".
> Re-run with `LC_ALL=C` on both ends and the **same explicit path list**, all 11 matched. Recorded
> here because the first result looks like data corruption and is not.

### What was NOT copied from X9, and why

The complete `2024-08-*` set is **39.9 GB**, over the ~30 GB budget for this task. Two day
directories were dropped, deliberately, and both are **0 MRad** (the dose `TID_August` already
covers in full):

1. **`2024-08-04_Array_Test_Results`** — 4 509 files, **5 864 572 201 B = 5.86 GB**. BARE9 at
   **0 MRad**, VA 1.00 / 1.20 / 1.40 V, plus a `_RoomT` control run and two `noise_Pixel` runs.
   `TID_August` already holds BARE9 0 MRad at VA 0.85 and 1.20 V. **The `_RoomT` control run and
   the VA = 1.40 V 0 MRad point are therefore still only on X9** — see §7.
2. **`2024-08-14_Array_Test_Results`** (bulk) — 7 117 files, **8 880 999 274 B = 8.88 GB**. BARE7
   at **0 MRad**, VA 0.85 / 1.00 / 1.20 V. **This day has since been copied and decoded** (see the
   table above): re-verified with `LC_ALL=C`-sorted `name + size` listings on both ends, which
   matched line for line at **7 117 files / 8 880 999 274 B**, plus a 6-file md5 spot check that
   was identical on both ends. It is what gives BARE7 a **0 → 200 MRad dose step**.
3. The **July** day directories (rows 5 of the census, the ET2.03-vs-ET2.01 DAC-scale comparison)
   are out of scope for this task and were not touched.

**Urgency note, unchanged from the census:** X9 is removable and may not be attached next session.
After the `2024-08-14` copy, the only part of the August master copy not on LPC is the 5.86 GB
`2024-08-04` day (BARE9 0 MRad at VA 1.00 / 1.20 / 1.40 V, the `_RoomT` control run and two
`noise_Pixel` runs).

---

## 2. The deliverables

Written by one script, `reduce_tid.py` (shipped alongside), plus `chipconfig_to_csv.py` for the
register dumps, `merge_all.py` for the union tables and `postfix_elapsed.py` for one name-parsing
repair (§7.11). All are in `code/` next to the tables.

**There are two families of table**, with identical schemas:

* **`tid_et2p03_*`** — the `TID_August` working copy alone (0 MRad money plots + all rail scans).
  Kept unchanged so earlier work against it stays reproducible.
* **`tid_et2p03_*_all`** — **the union of `TID_August` and the X9 master copy**, i.e. the one that
  contains the **200 and 400 MRad money plots**. This is the table to use for a money-plot-vs-dose
  analysis. It carries one extra first column, **`source_tree`** ∈ {`TID_August_working_copy`,
  `X9_master_copy`}, and every row carries **`day`** (the `<date>_Array_Test_Results` directory, or
  `not recorded` for the working copy, which has no day directories).

The X9-only intermediate tables are in `out_x9/` as `x9_et2p03_*`; the union is built from them
without re-decoding.

### `tid_et2p03_points.feather` — one row per decoded point directory

The first 22 columns are **the same names and the same meanings** as
`qinj_hybrid_cc/cc_points.feather`, with `hv_state` / `power_tag` (meaningless on a bare die with
no sensor and no `highpower` tag in any name) replaced by the axes this campaign actually varied —
dose and the two supply rails.

| column | RAW/DERIVED | meaning |
|---|---|---|
| `source_tree` | DERIVED | `_all` tables only: which copy the row came from |
| `config` | DERIVED | `<run_dir>` for a money-plot run, `<run_dir>__<sub_run_dir>` for a voltage-scan sub-run |
| `day` | RAW | the `<date>_Array_Test_Results` directory (X9 layout), else `not recorded` |
| `run_dir`, `sub_run_dir`, `point_dir` | RAW | the three source directory names, verbatim |
| `scan_family` | DERIVED | `money_plot` (a threshold scan over charges) or `voltage_scan` (one fixed-threshold acquisition per rail setting) |
| `chip` | RAW (name-parsed) | `ET2p03_BARE9` or `ET2p03_BARE7` |
| `kind` | DERIVED | `TDC` (threshold-scan point) or `VoltageScan` |
| `temperature_C` | RAW (name-parsed) | from the `m23p5C`…`m23p9C` token; `-23.5`…`-23.9` °C |
| `temperature_source` | DERIVED | `run_dir_name` or `not recorded` |
| `dose_MRad` | RAW (name-parsed) | 0 / 200 / 400, from the `_<n>MRad` token |
| `dose_source` | DERIVED | `run_dir_name` or `not recorded` |
| `va_V`, `vd_V` | RAW (name-parsed) | analog and digital **POWER-SUPPLY SETTINGS**, from `<n>p<nn>VA` / `<n>p<nn>VD` — **not** voltages at the chip. The campaign logbook (`ET2p03_BARE{9,7}_Voltage_scan/info.txt`, staged with these tables) records `Nominal Analog 1.2V = 1.4V on power supply` and `Nominal Digital 1.2V = 1.25V on power supply`, i.e. a 0.20 V analog / 0.05 V digital IR drop over cables and board. So VA 0.85 / 1.00 / 1.20 / 1.40 V (supply) is ~0.65 / 0.80 / 1.00 / 1.20 V at the chip and only VA 1.40 is nominal; VD 1.20 (supply) is ~1.15 V at the chip. The drop is I*R and no rail current was logged, so at-chip values are approximate to ~50 mV. |
| `elapsed_h` | RAW (name-parsed) | hours since the start of the dose step, from `_<hh>h<mm>m<ss>s`; `NaN` where the name carries none |
| `repeat` | RAW (name-parsed) | 1, or *n* from a trailing `_v<n>` re-take |
| `row`, `col` | RAW | pixel, from `Pixel_C<col>_R<row>` in the point-directory name |
| `charge_fC` | RAW | injected charge, from `QInj_<n>` (= the `QSel` register value in fC, §5) |
| `dac` | RAW | absolute threshold DAC, from `Threshold_<n>` |
| `dac_source` | DERIVED | `point_dir_name`, or **`not recorded`** for every voltage-scan point (§4) |
| `n_hits` | DERIVED | hits **at the pixel named in the directory**; the moments below are over exactly these |
| `n_files` | RAW | how many `.bin` (or `.nem`) files the point directory held |
| `source_format` | DERIVED | `bin` (decoded here) or `nem` (already translated in 2024, parsed here) |
| `toa_mean`, `toa_std`, `tot_mean`, `tot_std`, `cal_mean`, `cal_std` | DERIVED | moments over **every** hit at the pixel; see below |
| `n_hits_cal` | DERIVED | hits surviving the per-hit CAL window (see below) |
| `frac_rejected` | DERIVED | `1 − n_hits_cal / n_hits`, the fraction the window drops at this point |
| `cal_mode` | DERIVED | the modal CAL code of that pixel at that point — the centre of the window; `NaN` on the voltage-scan rows, where hits are pooled over pixels and each pixel is windowed against its own mode |
| `toa_mean_cal`, `toa_std_cal`, `tot_mean_cal`, `tot_std_cal`, `cal_mean_cal`, `cal_std_cal` | DERIVED | the same moments over the CAL-windowed hits only |
| `n_hits_all_pixels`, `n_pixels_seen` | DERIVED | hits and distinct pixels **before** the pixel selection — the audit trail for §6 |
| `decode_error` | DERIVED | empty string on success; a traceback if the directory failed |

**Definition of the moments, and it is the one the two companion feathers use.** `*_mean` and
`*_std` are a **plain mean and standard deviation over every hit recorded at that pixel** — no cuts,
no CAL window, no outlier rejection, no time-ordering, no BCID selection. The standard deviation is
**`ddof = 1`** (the sample std, pandas' default). All values are **raw TDC codes, not nanoseconds.**
Where a point directory held more than one file the files are concatenated first and the moments
taken over the union.

#### The per-hit CAL window, and the `*_cal` moments

Those raw moments are **kept unchanged and are still the primary columns**. Alongside them the
reduction now writes a second set, `*_cal`, over the hits that pass a per-hit CAL window:

> a hit is kept if `|CAL − cal_mode| ≤ 3` codes, where `cal_mode` is the **modal CAL code of that
> pixel at that scanned point**.

That is this repository's own test-beam criterion — `TestBeam/condor_at_lxplus/core/path_finder.py`
and `core/extract_events_by_path.py` both apply `abs(cal − cal_mode) <= 3` — reused here verbatim.
The mode is taken per point rather than per campaign because CAL drifts with the analog rail, with
temperature and with dose across this campaign, so a campaign-wide mode would reject good hits at
the rail extremes. `CAL_GUARD = (100, 220)`, which appears elsewhere in the analysis, is a
different object: an absolute sanity band on the CAL *mean*, not a per-hit window. It does not
catch what this window catches.

**Why it exists.** One TOA code is `3.125 ns / CAL`, so a hit whose CAL is wrong has a wrong TOA
and a wrong TOT no matter how well the TDC performed. A small fraction of this campaign's hits
carry a grossly wrong CAL, and they are not spread evenly: **on each pixel they occupy a contiguous
block of a few threshold DAC values at a fixed absolute DAC, the same block for every injected
charge in that pixel.** The sweep's inner loop is charge at fixed threshold, so a feature that is
identical across charge at one absolute DAC is an acquisition/decode artefact, not front-end
behaviour. It hides in the mean — `cal_mean` moves by a couple of codes — and shows in the spread,
where `cal_std` jumps from well under one code to tens of codes.

**What it removes.** Over all 29828 money-plot points: **2.6795 % of hits** (6428591 of 239920655), median per-point
rejected fraction **0.0000 %**. 5795 points (19.4 %) carry `cal_std > 2` codes on the raw moments.

The block at the reference condition (`ET2p03_BARE9`, 0 MRad, VA 1.20 V, VD 1.20 V), before and
after the window. "Block" here means the longest run of consecutive DAC values flagged at three or
more injected charges, taken **above the pixel's baseline** (offset ≥ 5, where the sweep starts).
Two other things also raise `cal_std` and are deliberately excluded from that definition: points
below the baseline, where noise fires the discriminator, and points above a pixel's own turn-off,
where almost no hits remain.

| pixel (row, col) | absolute DAC block | width | offset above BL | charges flagged | mean `cal_std` raw → windowed | mean `toa_std` raw → windowed | hits dropped there |
|---|---|---|---|---|---|---|---|
| (0, 0) | 885–885 | 1 | +63 … +63 | 3 | 27.7 → 0.46 | 33.3 → 2.1 | 0.51 % |
| (0, 14) | 861–863 | 3 | +34 … +36 | 4 | 28.7 → 0.42 | 27.8 → 1.2 | 0.35 % |
| (3, 0) | 879–879 | 1 | +54 … +54 | 3 | 33.9 → 0.40 | 28.9 → 0.9 | 0.54 % |
| (3, 14) | 839–840 | 2 | +5 … +6 | 5 | 22.7 → 0.47 | 41.2 → 30.0 | 0.35 % |
| (12, 0) | 891–896 | 6 | +60 … +65 | 3 | 23.5 → 0.39 | 25.3 → 0.5 | 0.37 % |
| (12, 14) | 904–908 | 5 | +57 … +61 | 3 | 23.1 → 0.27 | 30.0 → 0.4 | 0.43 % |

The same, condition by condition:

| chip | dose | VA | points | points with `cal_std > 2` | hits | hits rejected by the window |
|---|---|---|---|---|---|---|
| `ET2p03_BARE7` | 0 MRad | 0.85 V | 1656 | 291 (17.6 %) | 13336805 | 3.1104 % |
| `ET2p03_BARE7` | 0 MRad | 1.00 V | 2374 | 407 (17.1 %) | 19024944 | 3.2278 % |
| `ET2p03_BARE7` | 0 MRad | 1.20 V | 2964 | 379 (12.8 %) | 23994179 | 2.0101 % |
| `ET2p03_BARE7` | 200 MRad | 1.20 V | 3023 | 426 (14.1 %) | 24272067 | 1.1787 % |
| `ET2p03_BARE9` | 0 MRad | 0.85 V | 1168 | 261 (22.3 %) | 9587407 | 4.4956 % |
| `ET2p03_BARE9` | 0 MRad | 1.20 V | 2376 | 393 (16.5 %) | 19325945 | 1.5052 % |
| `ET2p03_BARE9` | 200 MRad | 0.85 V | 1153 | 385 (33.4 %) | 9176547 | 2.3049 % |
| `ET2p03_BARE9` | 200 MRad | 1.00 V | 1995 | 593 (29.7 %) | 15886492 | 6.2240 % |
| `ET2p03_BARE9` | 200 MRad | 1.20 V | 2425 | 495 (20.4 %) | 19468104 | 1.6388 % |
| `ET2p03_BARE9` | 200 MRad | 1.40 V | 2536 | 448 (17.7 %) | 20351331 | 1.1472 % |
| `ET2p03_BARE9` | 400 MRad | 0.85 V | 1123 | 238 (21.2 %) | 9046476 | 5.3896 % |
| `ET2p03_BARE9` | 400 MRad | 1.00 V | 1927 | 533 (27.7 %) | 15369465 | 4.3994 % |
| `ET2p03_BARE9` | 400 MRad | 1.20 V | 2475 | 519 (21.0 %) | 19888501 | 3.4688 % |
| `ET2p03_BARE9` | 400 MRad | 1.40 V | 2633 | 427 (16.2 %) | 21192392 | 1.4296 % |

Both sets of columns ship, so any consumer can compare them; nothing is dropped, and no threshold
point, pixel, charge or condition is removed by the window — only individual hits inside a point.

**The only selection that changes the primary columns anywhere in this reduction** is
`row == <row in the directory name> and col == <col in the directory name>` (the CAL window above
adds columns, it does not alter the raw ones). It is recorded, not hidden: `n_hits_all_pixels` and
`n_pixels_seen` say exactly how much it removed, point by point. For the money-plot runs one pixel
is enabled and it removes essentially nothing (§6); for the voltage-scan runs sixteen pixels are
enabled and no directory name identifies one, so **no selection is applied there** and `n_hits`
equals `n_hits_all_pixels` (`row`/`col` are `NaN`).

### `tid_et2p03_turnoff.csv` — the binary-search turn-off points (the money plot itself)

One row per (`config`, pixel, charge). Parsed from `FPGA_Data.dat`, which has **seven
comma-separated integers per line**; the column meanings are read off
`notebook_helpers.pixel_turnoff_points`, which is the code that wrote them:
`[0]` fpga state/time, `[1]`, `[2]` unused here, `[3]` TDC data count, `[4]` L1A count (the
`header_max` the search compares against), `[5]` trigger-bit count, `[6]` **DAC**.

* `turnoff_dac` — **RAW**, the DAC on the **last line** of the file. This is *exactly* the campaign's
  own definition: `notebook_helpers.charge_peakDAC_plot` reads `int(last_line.split(',')[-1])` and
  plots that against charge. Reproduced rather than re-derived, on purpose.
* `l1a_max`, `turnoff_triggerbits`, `turnoff_tdc_data`, `dac_first`, `dac_min`, `dac_max`,
  `n_search_steps` — RAW.
* `search_trace` — RAW, the whole binary search as `dac:triggerbits;dac:triggerbits;…`, so the
  turn-off point can be re-derived under any other definition without going back to the raw tree.

### `tid_et2p03_noiseonly.csv` — the `quietCB` noise scans

One row per (`config`, pixel), same `FPGA_Data.dat` parsing. `dac_min`/`dac_max`,
`triggerbits_max`, `dac_at_triggerbits_max` and the full `scan_trace`. These are the only in-campaign
measurement of the noise S-curve as *data* rather than as an autocal summary.

### `tid_et2p03_configs.csv` — one row per run / sub-run

The conditions grid (`chip`, `vd_V`, `va_V`, `temperature_C`, `dose_MRad`, `elapsed_h`,
`charge_tag_fC`, `repeat`, the two `*_source` columns, `run_dir`, `sub_run_dir`, `config`,
`scan_family`, `n_point_dirs`). All RAW name-parsed except `config`, `scan_family` and the
`*_source` columns.

### `tid_et2p03_chipconfig.csv` — the register dumps (this is where VRef lives)

Decoded from the three `ChipConfig/*.pckl` files by `chipconfig_to_csv.py`. See §5.

---

## 3. What conditions the set actually contains

### 3a. `TID_August` (the working copy) — 96 runs / sub-runs

| top-level directory | kind | sub-runs | what it is |
|---|---|---:|---|
| `ET2p03_BARE9_1p2VD_0p85VA_m23p8C_0MRad` | money plot | 1 | full threshold × charge scan, **0 MRad**, VA 0.85 V |
| `ET2p03_BARE9_1p2VD_1p2VA_m23p5C_0MRad` | money plot | 1 | full threshold × charge scan, **0 MRad**, VA 1.20 V |
| `ET2p03_BARE9_Voltage_scan` | rail scan | 84 | VD 0.90–1.45 V × VA 0.85–1.40 V at **0 / 200 / 400 MRad**, fixed **10 fC** |
| `ET2p03_BARE7_Voltage_scan` | rail scan | 10 | VD 0.90–1.40 V × VA 0.85–1.40 V at **0 MRad**, fixed **10 fC** |

VA ∈ {0.85, 0.90, 0.95, 1.00, 1.10, 1.20, 1.30, 1.40} V; VD ∈ {0.90, 0.95, 1.00, 1.05, 1.10, 1.15,
1.20, 1.30, 1.40, 1.45} V. **In this tree the 200/400 MRad data are the 10 fC rail scans only.**

### 3b. The X9 day directories — day → chip / dose / rail map

| day | chip | dose | VA (at VD 1.20 V) | elapsed since dose |
|---|---|---|---|---|
| `2024-08-03` | BARE9 | 0 MRad (untagged, §7.10) | 1.20 | — |
| `2024-08-07` | BARE9 | **200 MRad** | 0.85, 1.00 | 16 h, 19 h |
| `2024-08-08` | BARE9 | **200 MRad** | 1.20, 1.40 | 23 h, 33 h |
| `2024-08-11` | BARE9 | **400 MRad** | 0.85, 1.00, 1.20 | 19.5 h, 22 h, 25.9 h |
| `2024-08-12` | BARE9 | **400 MRad** | 1.40 | 37 h |
| `2024-08-14` | BARE7 | 0 MRad | 0.85, 1.00, 1.20 | — (not tagged) |
| `2024-08-21` | BARE7 | **200 MRad** | 1.20 | — (not tagged) |

### 3c. Money-plot (threshold-scan) coverage now available, per condition

All 14 conditions have the **same six pixels** — (row, col) ∈ {(0,0), (0,14), (3,0), (3,14),
(12,0), (12,14)} — and the **same five charges, 6 / 8 / 10 / 15 / 22 fC**.

| chip | dose | VA | VD | points | DAC range | hits |
|---|---|---|---|---:|---|---:|
| BARE7 | 0 MRad | 0.85 | 1.20 | 1 656 | 593–750 | 13 336 805 |
| BARE7 | 0 MRad | 1.00 | 1.20 | 2 374 | 660–881 | 19 024 944 |
| BARE7 | 0 MRad | 1.20 | 1.20 | 2 964 | 733–1018 | 23 994 179 |
| BARE7 | **200 MRad** | 1.20 | 1.20 | 3 023 | 744–1018 | 24 272 067 |
| BARE9 | 0 MRad | 0.85 | 1.20 | 1 168 | 693–797 | 9 587 407 |
| BARE9 | 0 MRad | 1.20 | 1.20 | 2 376 | 821–1018 | 19 325 945 |
| BARE9 | **200 MRad** | 0.85 | 1.20 | 1 153 | 609–732 | 9 176 547 |
| BARE9 | **200 MRad** | 1.00 | 1.20 | 1 995 | 676–866 | 15 886 492 |
| BARE9 | **200 MRad** | 1.20 | 1.20 | 2 425 | 774–1011 | 19 468 104 |
| BARE9 | **200 MRad** | 1.40 | 1.20 | 2 536 | 852–1018 | 20 351 331 |
| BARE9 | **400 MRad** | 0.85 | 1.20 | 1 123 | 589–724 | 9 046 476 |
| BARE9 | **400 MRad** | 1.00 | 1.20 | 1 927 | 627–841 | 15 369 465 |
| BARE9 | **400 MRad** | 1.20 | 1.20 | 2 475 | 715–989 | 19 888 501 |
| BARE9 | **400 MRad** | 1.40 | 1.20 | 2 633 | 806–1018 | 21 192 392 |
| **total** | | | | **29 828** | | **239 920 655** |

**BARE7 now spans 0 → 200 MRad** at VA 1.20 V (and 0 MRad at VA 0.85 and 1.00 V as well), from the
`2024-08-14` day copied and decoded after the first pass. It does **not** reach 400 MRad: only
BARE9 has all three doses, so every 400 MRad statement still rests on the single die BARE9.

Turn-off scans: **450** rows (15 conditions × 6 pixels × 5 charges), noise scans **90**.

### 3d. The result the campaign was taken for

The turn-off DAC vs charge at VA = 1.20 V, pixel (0,0), BARE9 — the money plot, versus dose:

| dose | 6 fC | 8 fC | 10 fC | 15 fC | 22 fC | span / 16 fC |
|---|---:|---:|---:|---:|---:|---:|
| 0 MRad | 858 | 877 | 895 | 950 | 1019 | 10.1 DAC/fC |
| 200 MRad | 838 | 860 | 882 | 934 | 1012 | 10.9 DAC/fC |
| 400 MRad | 818 | 832 | 858 | 910 | 990 | 10.8 DAC/fC |

The whole curve shifts **down** with dose while the slope is preserved to ~8 %. This is one pixel at
one rail setting, shown to demonstrate the tables work end to end — **it is not an analysis**, and
the DAC-LSB-to-mV scale is not established here.

**Temperature.** −23.2 to −23.9 °C, from the directory names only, on both trees. `info.txt` calls
it a "−25 C plate" setpoint. There is **no temperature log** anywhere.

## 4. File formats found, and the decoder

**Formats in `TID_August`** (4 761 files): **3 554 `TDC_Data_<n>.bin`**, **1 133
`TDC_Data_translated_<n>.nem`**, **72 `FPGA_Data.dat`**, **2 `info.txt`**. The X9 tree
(19 681 files) is the same, with `.bin` throughout, plus 3 `ChipConfig/*.pckl` and 2 BL/NW pickles.
There is **no json, toml, yaml, sqlite or csv in either tree** — see §5.

### The `.bin` format

Established by direct inspection of the byte stream and confirmed against the repo decoder:

* a little-endian stream of **32-bit FPGA words** (every file size is a multiple of 4);
* **event header** `(w >> 4) == 0xc3a3c3a`, low nibble = enabled-channel mask (always `0b0001` here,
  one ETROC on channel 0);
* **firmware word** `(w >> 28) == 0b0001` (the firmware key), carrying the event number
  `(w >> 12) & 0xffff` and `n40 = (w >> 2) & 0x3ff`, the number of 40-bit ETROC words in the event;
* **payload**: `ceil(40·n40 / 32)` words. Their bytes, taken **big-endian within each 32-bit word**
  and concatenated, form a byte stream that is cut into **5-byte (40-bit) ETROC2 words**;
* **event trailer** `(w >> 26) == 0b001011`.

The 40-bit ETROC2 words are the standard ones: **channel header** `(W >> 22) == 0x3c5c0 >> 2`
with `bcid = W & 0xfff` and `l1a = (W >> 14) & 0xff`; **data** `(W >> 39) == 1` with
`col = (W >> 33) & 0xf`, `row = (W >> 29) & 0xf`, `toa = (W >> 19) & 0x3ff`,
`tot = (W >> 10) & 0x1ff`, `cal = W & 0x3ff`; **trailer** `((W >> 22) & 0x3ffff) == board_id`
with `hits = (W >> 8) & 0xff`, `status = (W >> 16) & 0x3f`.

A typical single-hit event is therefore **28 bytes** = firmware word + 4 payload words
(header + data + trailer = 120 bits, padded to 128) + event trailer + the next event header.

### Decoder provenance

The production reduction uses **`DecodeBinary` from this repository**, copied **verbatim** from
`TestBeam/condor_at_lxplus/core/decoding.py`
(md5 `289ef6cb7b7e48589e02d817de5060ed`, byte-identical on both ends) into `code/decoding.py`,
called with `firmware_key = 0b0001`, `board_id = [0x17f0f] × 4`,
`skip_fw_filler / skip_event_df / skip_crc_df = True`.

That it is the right decoder is not an assumption. `notebook_helpers.run_daq` — the function the
2024 acquisition scripts call — drives `run_script.main_process` with
`--compressed_translation`, and the `.nem` files sitting in this very tree begin
`EH 0001 1 14 16`, which is `DecodeBinary.write_to_nem`'s event-header line with **firmware key
0001**. The `.nem` and the `.bin` in this campaign are two outputs of the same translator.

### The `.nem` files

1 133 of them, all inside `ET2p03_BARE9_Voltage_scan` sub-runs (that day's DAQ was run with
translation on and binary kept off; the BARE7 sub-runs and every money-plot point kept the binary
instead). They are parsed here, not re-decoded, from their `D` lines:
`D <channel> <EA> <row> <col> <toa> <tot> <cal>`.

### `dac` is `not recorded` for the voltage-scan points

A voltage-scan sub-run is **one acquisition at one fixed threshold**, and no directory name, and no
file in the tree, records what that threshold was. It is therefore **`NaN`, with
`dac_source = 'not recorded'`** — not a baseline-plus-offset guess. (The acquisition sets it from
the autocal baseline plus an offset; the offset is in the notebook call, which was not saved.)

---

## 5. Aux data — VRef, temperature, currents: what exists and what does not

**This campaign stored almost no machine-readable metadata.** The complete list of non-`.bin`/`.nem`
artefacts, everywhere, is:

| artefact | where | size | what it gives |
|---|---|---:|---|
| `info.txt` ×2 | `*_Voltage_scan/` in `TID_August` | 1 703 / 1 694 B | **the campaign logbook** — prose, see below |
| `FPGA_Data.dat` ×72 | turn-off and noise-scan point dirs | 224–~500 B | the DAC / trigger-bit search traces (§2) |
| `ChipConfig/*.pckl` ×3 | X9 `2024-08-08`, `-14`, `-21` | 131 408 B each | **the full ETROC2 register read-back** |
| `ET2p03_BARE9_{BL,NW}_map.pickle` | X9 `2024-08-03` | 2 176 B each | autocal baseline / noise-width maps, 16×16 |

There is **no** per-run json / toml / yaml / csv, **no** power-supply or LV-current log, **no**
temperature log, and **no** `PowerHistory` sqlite for August 2024 — unlike the CERN Jan-2024
campaigns, which have both. `ETROC-History/BaselineHistory.sqlite` on `etroc3` was checked directly:
its earliest row is **2025-01-30**, and it contains **no `BARE7`/`BARE9` rows at all**, so it is not
this campaign's baseline record. The BL/NW pickles above are.

### VRef — the ET2.03 question, answered

ET2.03 uses the chip's **internal** voltage-reference generator (ET2.01 used an external 1.0 V).
The three register dumps confirm it directly and identically:

| field | value | meaning |
|---|---|---|
| `VrefGen_PD` (PeriCfg3 bit 7) | **0** | **the internal voltage-reference generator is POWERED UP** (the field is active-high power-down) |
| `RefStrSel` | **3** | reference-strobe selection, all three dumps |
| `TS_PD` | **0** | the on-chip temperature sensor is powered up |
| `CLKSel` | 1 | |
| `PLL_BiasGen_CONFIG` | 8 | |
| `chargeInjectionDelay` | 10 | |

All three dumps (BARE9 at 200 MRad, BARE7 at 0 MRad, BARE7 at the end of its 200 MRad step) agree
on every one of these. The full decoded field set (135 columns, including the raw `PeriCfg0..31`
bytes so nothing is lost) is `tid_et2p03_chipconfig.csv`.

**Caveat, stated plainly:** `TS_PD = 0` means the temperature sensor was *powered*, not that anyone
read it. **No VTemp / ADC reading is stored anywhere in this campaign.** The only temperature
information is the `m23p<n>C` token in the directory names.

### Baseline / noise width at the operating point

`tid_et2p03_blnw.csv` — the 16×16 autocal maps, `ET2p03_BARE9`, `2024-08-03`. At the six scanned
pixels: **BL = 822 / 828 / 824 / 834 / 831 / 849** DAC LSB and **NW = 3 / 4 / 5 / 4 / 5 / 4** LSB
for (0,0) / (0,14) / (3,0) / (3,14) / (12,0) / (12,14). Consistent with the DAC 821–1018 sweep of
the VA = 1.20 V money-plot run.

### `info.txt` — the only record of the LV currents, and it is prose

It is worth reading in full; it is the campaign's own account of what went wrong. It opens with the
bench-offset line quoted in §3, and then records, per dose step: that pixel auto-calibration failed
at VD = 0.90 V after 200 MRad; that at VD = 0.95 V links and QInj fail *"even though pixels
calibrate"* and *"QInj events are empty but at right BCIDs"*; that `clear_fifo` at VDD 1.3 V breaks
the PLL lock but not at 1.4 V; that after 400 MRad calibration took ~50 s longer; and — the only
current measurement anywhere — *"WS current is down at 9.X mA level … it was not so during the
Voltage / money plots after 200 MRad when we saw large large noise peak measurements. Is this a
related phenomenon? **Perhaps money plots will reveal all.**"*

**The LV currents are therefore `not recorded` as data.** That sentence is the whole record.

---

## 6. Verification

**1. An independent decoder, written from the format description, reproduces the production
numbers exactly.** `code/verify_hand.py` re-decodes a point directory from raw bytes without
importing `DecodeBinary` at all (its own 32-bit word walk, its own 40-bit repacking, its own field
masks). On `ET2p03_BARE9_1p2VD_0p85VA_m23p8C_0MRad / …_Pixel_C0_R0_QInj_10_Threshold_700`:

| | independent decoder | `reduce_tid.py` (`DecodeBinary`) |
|---|---|---|
| hits at (0,0) | **8 225** | **8 225** |
| `toa_mean` | 202.744 559 270 5 | 202.744 559 270 516 7 |
| `tot_mean` | 68.559 270 516 7 | 68.559 270 516 717 3 |
| `cal_mean` | 188.785 167 173 3 | 188.785 167 173 252 3 |

Agreement to every digit printed. The independent walk additionally finds **2 stray hits** at
pixels (7,8) and (8,15) out of 8 227 — words inside events that `DecodeBinary` rejects (its CRC /
hit-count checks) and that the pixel selection would remove regardless.

**1b. The same check on two *dosed* points, from the X9 tree.**

| point | independent decoder | `reduce_tid.py` |
|---|---|---|
| BARE7 **200 MRad**, `2024-08-21`, (3,14), 22 fC, DAC 1000 | 7 954 hits; toa 239.839 200 4, tot 18.849 258 2, cal 184.474 226 8 | **7 954**; 239.839 200, 18.849 258, 184.474 227 |
| BARE9 **400 MRad**, `2024-08-11`, (0,0), 15 fC, DAC 774 | 19 389 hits; toa 506.394 657 | 19 388; toa 506.408 087 |

The BARE7 point agrees **exactly**. The BARE9 point differs by **one hit in 19 389 (5 × 10⁻⁵)**:
`DecodeBinary` rejects one event on its trailer hit-count check, which the deliberately unvalidated
reference decoder keeps. The production number is the stricter one. This is reported rather than
smoothed over — it is also evidence that the check in §6.6 is actually discriminating.

**2. Decoded pixel identity matches the directory name.** For all **29 828** money-plot points
across both trees, the `row`/`col` decoded out of the binary agrees with the `Pixel_C<col>_R<row>`
in the path. `n_pixels_seen > 1` in only **293 of 29 828** directories, and the pixel selection
removes **38 219 hits out of 183 602 946 — 0.021 %**.

**3. The decoding reproduces an independently recorded physical fact.** `info.txt` states that at
VD = 0.90 V (and, after 200 MRad, 0.95 V) the chip calibrates but produces **no QInj events**.
Decoding those sub-runs blind gives **exactly zero hits**: 9 of the 94 voltage-scan points are
empty, and 8 of the 9 are VD ∈ {0.90, 0.95, 1.00} V. This was not put in by hand.

**4. Codes lie in their physical ranges.** Across all points `toa_mean ≤ 702.3`, `tot_mean ≤ 333.7`,
`cal_mean ≤ 571.7` — inside the 10-bit TOA / 9-bit TOT / 10-bit CAL fields.

**5. The turn-off DAC rises monotonically with injected charge**, as it must for a money plot
(e.g. BARE9, VA 0.85 V, pixel (0,0): 718 / 729 / 739 / 754 / 793 DAC for 6 / 8 / 10 / 15 / 22 fC).

**6. Decode failures: zero.** `decode_error` is empty for all **29 922** rows of the union table. `DecodeBinary` printed
`Number of hits does not match!` **22 times** on `TID_August`, and a similar handful on X9 — events whose
trailer hit-count disagreed with the data words counted, which it discards; that is a 10⁻⁵-level effect
(and §6.1b shows one of those events directly), recorded here rather than suppressed.

---

## 7. Caveats — read these before using the tables

1. **Use `tid_et2p03_points_all.feather` for anything involving dose.** The plain
   `tid_et2p03_points.feather` is `TID_August` only, where the 200/400 MRad data are the 10 fC rail
   scans and the money plots are 0 MRad. Both are shipped; only the `_all` one has dosed money plots.
2. **BARE7 spans 0 → 200 MRad, BARE9 spans 0 → 400.** The `2024-08-14` day was copied and decoded
   after the first pass, so a **BARE7 dose difference can now be formed** — at VA 1.20 V, the one
   rail both of its doses share. Nothing on BARE7 reaches 400 MRad, so every 400 MRad number in
   these tables rests on the single die BARE9.
3. **`dac` is `NaN` for every voltage-scan point**, a property of the campaign, not of this
   reduction (§4). Do not treat those 94 rows as threshold-scan points.
4. **The 94 voltage-scan rows pool 16 pixels.** No directory name names a pixel there, so no pixel
   selection is applied and the moments are over all sixteen. `n_pixels_seen` separates the two
   kinds of row; never mix them in one fit.
5. **`charge_fC` is `NaN` for 8 voltage-scan sub-runs** whose names carry no `_10fC` tag. The
   campaign almost certainly ran them at 10 fC like the rest; the name does not say so, so neither
   does the column.
6. **113 points decode to zero hits.** The eight genuine low-VD failures of §6.3 are real physics —
   keep them. The rest are near-baseline threshold points and a few failed re-takes.
7. **Temperature is name-parsed only**, to 0.1 °C, and is a plate reading, not a die reading.
8. **`repeat` (`_v1`…`_v4`) re-takes are not merged** — separate rows, deliberately, since the chip
   was re-calibrated between some of them.
9. **TOT/TOA/CAL are raw TDC codes**, not nanoseconds; no time-walk or CAL correction is applied.
10. **X9 `2024-08-03` carries no dose or temperature token** (its run directory is just
    `1p2VD_1p2VA`), so those 36 rows have `dose_MRad = NaN`, `dose_source = 'not recorded'`. By its
    position in the campaign it is the 0 MRad set, and its `day` column says so — but the reduction
    does not guess. Its 30 turn-off + 6 noise scans are a **separate acquisition** from the
    same-named `TID_August` run, not a byte-duplicate (checked by md5), so both are kept.
11. **One `elapsed_h` was repaired by re-parsing, not re-decoding.** The run directory
    `ET2p03_BARE9_1p40VA_1p20VD_400MRad_37hm00s_m23p5` is typo'd in the 2024 original — `37hm00s`,
    blank minutes — so the strict regex returned `NaN` for its 2 633 points. `postfix_elapsed.py`
    re-derives **only that one name-parsed column** with a lenient regex (`reduce_tid.py` now carries
    the same one); every decoded value is untouched. `elapsed_h = 37.0` there.
12. **X9 is removable, and 5.86 GB of the August master copy is still only on it** — the 0 MRad
    `2024-08-04` day (BARE9 at VA 1.00 / 1.20 / 1.40 V, the `_RoomT` control run and two
    `noise_Pixel` runs). `2024-08-14` has since been copied.

## 8. Reproducing this

```bash
# on LPC
source /cvmfs/sft.cern.ch/lcg/views/LCG_104a/x86_64-el9-gcc11-opt/setup.sh
export PYTHONUSERBASE=/uscms_data/d3/murtazas/scratch/qinj_tid_et2p03/pyenv   # crc + natsort
cd /uscms_data/d3/murtazas/scratch/qinj_tid_et2p03
python3 code/reduce_tid.py --root raw/TID_August --out out --prefix tid_et2p03 --workers 6
python3 code/reduce_tid.py --root raw_x9 --day-dirs --out out_x9 --prefix x9_et2p03 --workers 6
python3 code/merge_all.py                       # builds the *_all union tables
python3 code/postfix_elapsed.py                 # the one name-parsing repair, caveat 11
python3 code/chipconfig_to_csv.py out/tid_et2p03_chipconfig.csv aux/*.pckl
python3 code/verify_hand.py <a point directory>  # the independent check
```

LCG_104a supplies pandas / pyarrow / numpy; `crc` and `natsort` (needed by `decoding.py`) are
pip-installed into the scratch `pyenv/` so that nothing outside this directory was modified.

**Cost.** 6 worker processes, `nice -n 10`, interactive on `cmslpc`. First pass: `TID_August`
**3 638 directories in 779 s (13 min, 214 ms/dir)**; X9 **19 290 directories in 4 326 s (72 min,
224 ms/dir)**. The later `2024-08-14` day: **6 994 directories in 1 389 s (23 min, 199 ms/dir)**,
folded into the X9 intermediate tables and re-merged without re-decoding anything else. No batch
submission was needed.

## 9. File manifest

| file | rows × cols | bytes | md5 |
|---|---|---:|---|
| `tid_et2p03_points_all.feather` | **29 922 × 43** | 4 949 330 | `c1a669ff60e88b8697ef233cf0402a21` |
| `tid_et2p03_turnoff_all.csv` | 450 × 30 | 181 335 | `96883ad1b698caba6a9fa94b9cc4807b` |
| `tid_et2p03_noiseonly_all.csv` | 90 × 26 | 43 642 | `7da9e2675ef3b38aa343a5e64c69be41` |
| `tid_et2p03_configs_all.csv` | 109 × 17 | 29 923 | `1fea689e9f16a8bfb9cb2858f82a3fb0` |
| `tid_et2p03_points.feather` | 3 638 × 42 | 610 234 | `11832f9bef784d13102d9625180c068d` |
| `tid_et2p03_turnoff.csv` | 60 × 29 | 21 696 | `a0cf7d157ac24337bd223c57c0cc42ca` |
| `tid_et2p03_noiseonly.csv` | 12 × 25 | 5 411 | `99257c0ca258a409cf814da5a2ee80d3` |
| `tid_et2p03_configs.csv` | 96 × 16 | 24 809 | `d857637ee4f399ffab5353dd119ad860` |
| `tid_et2p03_chipconfig.csv` | 3 × 135 | 3 164 | `5aef7758fd020bddf206430880301a20` |
| `tid_et2p03_blnw.csv` | 256 × 5 | 14 833 | `96da68fed9fef08cf920002f4bfa5023` |
| `reduce_tid.py` | — | — | see `code/` |
| `decoding.py` (repo, verbatim) | — | 46 365 | `289ef6cb7b7e48589e02d817de5060ed` |
| `merge_all.py`, `postfix_elapsed.py`, `chipconfig_to_csv.py`, `etroc2_regmap.py`, `verify_hand.py` | — | — | see `code/` |

**Union table contents.** 29 828 money-plot (TDC) points + 94 voltage-scan points across
**14 conditions**, **2 dies** and **3 doses**; **239 920 655 hits** behind the TDC rows (295 583 698 including
the pooled voltage scans); **135 zero-hit points**; **0 decode errors**.

**Where the copies live.**
`lpc:/uscms_data/d3/murtazas/scratch/qinj_tid_et2p03/` (`raw` + `raw_x9` + `aux` + `code` + `out` +
`out_x9`; 31 GB of raw) ·
`~/ETROC_viz_mirror/qinj_et2p03_tid/` (laptop) ·
`lxplus:/afs/cern.ch/user/m/musafdar/CMS_ETROC/ETROC-Analysis/notes/qinj_et2p03_tid/`


## Addendum 2026-09-04 — BARE7 dose relabel

The `2024-08-21_Array_Test_Results` BARE7 acquisition is tagged `200MRad`, but the X-ray source failed before BARE7 reached 200 MRad (operator statement). In the `*_all` tables those rows now carry `dose_MRad = NaN`, `dose_source = unknown_xray_failure` and a `dose_note` column; nothing else changed. The campaign was at CERN (X-ray), not on the FNAL bench; the raw data were parked on the FNAL bench PC afterwards. The campaign's monitoring logs (VRef, VTemp pin, rails, 256-pixel baseline maps) are on LPC EOS under cern_Aug2024_TID/ETROC-History/CERN_Aug_2024_TID/.
