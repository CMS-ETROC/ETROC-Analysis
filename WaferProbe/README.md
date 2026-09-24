# WaferProbe plotting

`plot_wafer.py` turns the run folders of one tested wafer into three tables
and a set of figures. It only reads results and never talks to the station,
the supplies or the chip, so it can run on a copy of the results folder on
any computer, independent of the wafer-probe station.

The wafer test itself runs from the station repo `ETROC-WaferProbe`, which
writes the results and grades the dies. `station.py` carries a copy of its
grading (`src/grading.py`) and of two small readers, so this folder runs on
its own; when the station's grading changes, `station.py` follows it.

## Getting the results

The station laptop writes results under
`<path>/BatchID_<id>_Name_<batch>/WaferID_<id>_Name_<wafer>/`, with the batch
and wafer IDs of the station's `configs/wafers.csv` (`X` for a wafer it does
not list, such as a test wafer). The two folder names are the wafer's
labels, which the station also stamps on the prober (Velox lot id and wafer
id) and on its saved wafer map; the plots name the wafer by them too. Copy
that folder here, keeping both levels, e.g. with `rsync`:

```
rsync -av <station host>:<path>/BatchID_0_Name_N62M23/WaferID_3_Name_08A5/ \
    ./BatchID_0_Name_N62M23/WaferID_3_Name_08A5/
```

Results written before the station took these labels (ETROC-WaferProbe
before commit fcae979, 2026-09-23) sat in `<path>/<batch>/<wafer>/`. The
station's two wafers of that time were renamed once, by hand
(`FFF2p00/N60R91` to `BatchID_X_Name_FFF2p00/WaferID_X_Name_N60R91`,
`N62M23/08A5` to `BatchID_0_Name_N62M23/WaferID_3_Name_08A5`); a copy
taken before then needs the same rename, because `plot_wafer.py` finds a
wafer folder by its labels only.

## Environment

Python >= 3.9 with `numpy`, `pandas`, `pyarrow` and `matplotlib`. On lxplus,
`source TestBeam/condor_at_lxplus/envs/load_python39.sh` (from the
`ETROC-Analysis` repo root) gives an environment with all four.

## Running it

```
python plot_wafer.py --path <results dir> --batchName FFF2p00 --waferName N60R91
python plot_wafer.py --path <results dir> --batchID 0 --waferID 3
```

It finds the wafer folder by the name, the ID or both of each level
(`--batchName`, `--batchID`; `--waferName`, `--waferID`): given both, both
must match, and an ID never finds a folder with `X`, so a wafer without IDs
is found by its names. It refuses to guess when there is not exactly one
match, and lists the folders that match (or, when none does, every wafer
folder under `--path`). It then prints the grade counts and writes into
the folder's `plots/` (`--out` for another folder). A die is represented by its newest run folder
that holds a `summary.json`, passing over runs aborted with Ctrl+C: the run
whose grade the station map shows. Two exceptions: a run made by hand with
`master_run_script.py` counts here but never reaches the map, and a pass
stopped with Ctrl+C during a die's retry leaves the first attempt standing
here but no grade on the map. That run stands for the die whatever it
tested: a newest run without the full scan or charge injection leaves the
die out of the full-scan maps and blank on the QInj maps, even when an
older run has them. The station therefore keeps a wafer's full-protocol
runs under one `--path` and lighter runs (a 9-pixel test, an eFuse verify,
a data-phase scan) under another; give that other `--path` here to plot
them apart.
`--before "2026-09-22 15:30"` (DAQ computer clock) takes the newest run that
started before that time instead, to see the wafer as it stood then. A die of
the wafer map without a run grades `NOT_TESTED`. `--tables-only` skips the
figures (and warns when `--out` already holds some, which it leaves as they
are), `--no-qinj` skips reading the QInj data. `--waferMap` overrides the
die-position map, otherwise the bundled `wafer_map.csv` is used.

## What it writes

The tables, as csv:

- `dies.csv`, one row per die: grade and map text, run folder, status and
  error; the median current and voltage of every rail at power-on, at high
  power and during QInj (`<rail>_I_on`, `<rail>_I_high`, `<rail>_I_qinj` and
  `_V_`; the first sweep of each phase is dropped, it can still catch the
  rails switching: the power-on current at high power, vref ramping at
  power-on; a run without a power log gets its power-on rail-check reading
  as `_on`); the rail-check thresholds the run used; the I2C verdicts and
  failure lists; baseline and noise-width mean and std over the pixels that
  did not read zero, with the zero readings listed apart; the QInj verdict
  of the station's check (`qinj_check_*`; over the files after the first,
  or the last file alone in runs without `qinj_seconds`), the events of all
  the files read (`qinj_events`), the trailers with a nonzero chip
  status (`qinj_flagged_trailers`), the EA-flagged hit words and the lowest
  efficiency over the injected pixels; the chuck position at contact minus
  the station's map position (`dx_um`, `dy_um`); a note on a baseline or
  noise width that stands out (`bl_nw_note`, see "Baseline and noise-width
  notes"). Power-on logs only two
  sweeps, so `vref_V_on` is a single reading and can still be settling
  (1.01-1.11 V on 25 dies of N60R91, against 1.00 V at high power).
- `pixels.csv`, one row per calibrated pixel: baseline and noise width.
- `qinj.csv`, one row per pixel with hits in the QInj data (the files of
  `qinj/` after the first, which can hold malformed events from the start of
  the run; for older runs with a flush run `qinj_run1/`, all of the data run
  `qinj_run2/`; a `qinj/` run that wrote a single file, as a broken readout
  can, leaves nothing, and its QInj columns in `dies.csv` stay empty): hits,
  efficiency (hits per event), the most common CAL code, and the mean and
  sample std of CAL, TOA and TOT over the hits with |CAL - that code| < 3.
  Hits whose EA field is not 0 (the chip's own flag) are left out. The
  injected pixels are the list the run recorded (`qinj_pixels` in
  `summary.json`); for runs from before that record, the pixels hit in at
  least half of the events of at least half of the QInj dies, used only when
  the run's check expected that many hits per event. A die whose readout
  broke can add rows for pixels it was never injected at (junk words at
  (0,0) with EA 0: 6 hits, CAL 0, on die 16 of N60R91); the figures show the
  injected pixels only.

The figures, as png. A figure whose data the runs did not take (no full scan,
no QInj) is left out, and an older copy of it is deleted, so the folder never
mixes two plots:

| figure | shows |
|--------|-------|
| `grades` | grade per die, the counts and the yield; `*` = graded on the retry; a superscript letter = a baseline or noise-width note, listed under the map |
| `currents` | analog and digital current per die at power-on and at high power, and the difference |
| `currents_small_rails` | the other rails at high power |
| `current_hists` | the analog and digital currents as histograms, with the check thresholds the runs used |
| `baseline_maps` | baseline and noise-width mean and std per die; red frame = some pixels read zero; letters and notes as in `grades` |
| `baseline_hists` | baseline and noise width of all calibrated pixels, quick-test and full-scan dies apart |
| `alignment` | chuck position at contact minus the station's map position, per die, with a plane fitted over the wafer |
| `fullscan_baseline`, `fullscan_noise_width` | the 16 x 16 map of every full-scan die at its place on the wafer |
| `pixel_issues` | per pixel: dies failing the pixel-ID check, dies reading zero, baseline mean and std over the full-scan dies |
| `qinj_overview` | per QInj die: events, lowest pixel efficiency, trailers with a nonzero chip status, EA-flagged hits |
| `qinj_cal`, `qinj_toa`, `qinj_tot` | mean code per die, one wafer map per injected pixel |
| `qinj_pixels` | CAL, TOA and TOT mean and std per injected pixel, one dot per die |

Wafer maps draw each die at its `wafer_map.csv` place, row 0 at the top as on
the station display, and print the value in the cell; a die without a value
is hatched. Colour ranges span the 2nd to 98th percentile of the PASSED dies
(of all dies in the alignment maps and the full-scan galleries; fixed for
efficiencies and counts), so one broken die does not flatten the rest. Every
figure title starts with the wafer's labels
(`BatchID_0_Name_N62M23 / WaferID_3_Name_08A5`), on a line of their own when
the figure is too narrow for the whole first line (`pixel_issues` without
full-scan dies, say). Under every figure a note names the wafer by its
labels, which run of each die was used (the newest, or the newest before
`--before`) and when it was plotted.

### Baseline and noise-width notes

A die can pass every check of the station and still have a baseline or
noise width unlike the rest: a pixel whose baseline sits hundreds of codes
from its neighbours, or a whole die noisier than the wafer. The station
fails a pixel only when it reads exactly zero, so such a die keeps its
grade; the plots mark it instead. A die, whatever its grade, gets a note
(`bl_nw_note` in `dies.csv`) when

- a pixel's baseline is more than 100 DAC codes from the median of its
  die, or its noise width more than 8; the note names the three furthest
  such pixels and counts the rest;
- the die's baseline or noise-width mean is more than 5 robust sigma
  (1.4826 x the median absolute deviation) from the median of the wafer's
  PASSED dies calibrated over as many pixels (a mean over the 9 pixels of
  a quick test scatters more than one over the 256 of a full scan, so each
  kind is compared with its own). This needs at least 20 such PASSED dies
  whose means spread; a line under the maps names the kind of die it could
  not check, notes or not.

Zero readings stay out, since the grade already takes them. `grades` and
`baseline_maps` put a superscript letter (a, b, ..., in die order) on each
such die and print the notes under the map, the first 8 of them;
`dies.csv` has them all. A note never changes a grade, a bin or the
station's map. The limits (`NOTE_*` in `wafer_tables.py`) were set on the
calibrated dies of the two N62M23 wafers of September 2026 (114 of 08A5,
115 of 07B2, all full scans): the only die they flag is 07B2 die 15, by
both checks (pixel (8,8) at baseline 257 against the die's 554; noise-width
mean 9.36, +7.7 sigma from the wafer's 7.18). On every other die no pixel
is more than 80 codes from its die's median baseline, or more than 7 from
its median noise width.

## Tests

```
python -m unittest discover -s tests
```

run from this folder (`WaferProbe/`). Needs `numpy`, `pandas` and `pyarrow`
for `tests/test_wafer_tables.py`; also `matplotlib` for
`tests/test_plot_wafer.py`, else that file's tests are skipped. Both test
files build synthetic run folders (`tests/wafer_results.py`) laid out as the
station writes them and do not need real wafer data.
