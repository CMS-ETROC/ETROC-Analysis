# ETROC Test-Beam Analysis — HTCondor / lxplus Pipeline

Decodes, selects, and fits ETROC test-beam data using HTCondor batch jobs on lxplus.
Steps 4, 8, 10, and 12 submit condor jobs (`submit/`); the rest run locally (`core/`, `utils/`).

## Repository layout

| Path | Contents |
|---|---|
| `core/` | Worker scripts — either run locally or shipped to condor workers as `transfer_Input_Files`. |
| `submit/` | Condor submission scripts: generate the JDL/bash/input-list files and call `condor_submit`. |
| `utils/` | Standalone helpers: merging, candidate reduction, job monitoring (see [Utilities](#utilities)). |
| `envs/` | Environment setup scripts (`load_python39.sh`, `load_python311.sh`). |
| `mask_pixel_configs/` | Per-setup noisy-pixel mask YAMLs, used by step 6's `--mask_config`. |
| `../board_configs_yaml/` | Board-role config YAMLs (the `-c/--config` argument throughout). |
| `deprecated/` | Superseded scripts kept for reference only — not part of the current pipeline. |

## Pipeline flow

```
binary files
  -> [4]  decode                       (condor)
  -> [5]  merge feathers               (optional, local)
  -> [6]  find paths                   (local)
  -> [7]  reduce path candidates       (local)
  -> [8]  extract events by path       (condor)
  -> [9]  reshape event -> track       (local)
  -> [10] apply TDC cuts               (condor)
  -> [11] count events per track       (local)
  -> [12] bootstrap                    (condor)
  -> [13] merge bootstrap fit results  (local)
```

> Unless noted otherwise, every `submit/` script knows your EOS base path (`/eos/user/<n>/<name>/`)
> — path arguments (`-d`, `-o`, etc.) only need the part **after** that base.
> `--condor_tag` is optional everywhere it appears: if omitted, a unique tag is
> auto-generated so concurrent/sequential untagged submissions don't collide.

## Diagnostics at a glance

The fixes that came out of the first end-to-end pass on H1 run 1 (August 2026), each with its evidence and the commit that carries it, are written up in [docs/fixes_2026-08_h1_run1.md](docs/fixes_2026-08_h1_run1.md).

The pipeline steps above are unchanged in what they do; four optional tools in `utils/` look at their outputs. Each is one `summarize` (or read) plus one `plot` command, writes only figures (plus a parquet/CSV/JSON of the numbers) into its output folder, and every figure carries its own caption saying what is plotted and how each number is computed. All plot commands accept `--format png|pdf` and `--split` (one file per panel next to the compound figure). The dictionary of figures for each tool is in the section named in the table.

| Tool | Run after | What it answers | Section |
|---|---|---|---|
| `utils/telescope_diagnostics.py` | step 7 | Where the beam hits each board, occupancy, CAL/TOT per pixel and per chip half, where tracks land, beam tilt, board rotation; cross-run trends. | [7b](#7b-optional-telescope--beam-diagnostics-per-run) |
| `utils/track_diagnostics.py` | step 10 | Stability along the run (events, cut survival, CAL, TOT, inter-board offsets, per pixel), the time-walk correction across the array and across boards, its stability in time, a resolution proxy in time. | [10b](#10b-optional-per-track-diagnostics-stability-along-the-run-and-the-time-walk-correction) |
| `utils/bootstrap_diagnostics.py` | step 12 | Anatomy of one bootstrap trial with the fits on the data, statistical behaviour over all tracks, the KS gate vs statistics, resolution maps and chip halves, the same board and the same pixel pair across board combinations, two step-12 configurations track by track. | [12b](#12b-optional-bootstrap-diagnostics-anatomy-of-a-trial-statistical-health-partner-consistency) |
| `utils/quote_resolution.py` | step 13 | Quotable per-board numbers with one fixed recipe: cleaning, per-pixel table, robust mean over pixels, partner and definition systematics, chip halves, illumination split (average and central-hit maps). | [13b](#13b-quoting-the-resolutions---one-recipe-for-any-beam-campaign-and-number-of-boards) |

## Contents

- [Prerequisites](#prerequisites)
  - [0. Set the file-descriptor limit](#0-set-the-file-descriptor-limit)
  - [1. Load the Python 3.9 environment](#1-load-the-python-39-environment)
  - [2. Locate binary files on EOS](#2-locate-binary-files-on-eos)
  - [3. Check the Condor server](#3-check-the-condor-server)
- [Pipeline steps](#pipeline-steps)
  - [4. Submit decoding jobs](#4-submit-decoding-jobs)
  - [5. (Optional) Merge output feather files](#5-optional-merge-output-feather-files)
  - [6. Path finding](#6-path-finding)
  - [7. Reduce path-finding output](#7-reduce-path-finding-output)
    - [7b. (Optional) Telescope / beam diagnostics per run](#7b-optional-telescope--beam-diagnostics-per-run)
  - [8. Submit jobs for event selection by path](#8-submit-jobs-for-event-selection-by-path)
  - [9. Reshape output from event-based to track-based](#9-reshape-output-from-event-based-to-track-based)
  - [10. Submit jobs to apply TDC cuts](#10-submit-jobs-to-apply-tdc-cuts)
    - [10b. (Optional) Per-track diagnostics](#10b-optional-per-track-diagnostics-stability-along-the-run-and-the-time-walk-correction)
  - [11. Count events per track](#11-count-events-per-track)
  - [12. Submit jobs for bootstrap](#12-submit-jobs-for-bootstrap)
    - [12b. (Optional) Bootstrap diagnostics](#12b-optional-bootstrap-diagnostics-anatomy-of-a-trial-statistical-health-partner-consistency)
  - [13. Merge bootstrap results (unbinned Gaussian fit)](#13-merge-bootstrap-results-unbinned-gaussian-fit)
    - [13b. Quoting the resolutions](#13b-quoting-the-resolutions---one-recipe-for-any-beam-campaign-and-number-of-boards)
- [Diagnostics at a glance](#diagnostics-at-a-glance)
- [Utilities](#utilities)

---

## Prerequisites

### 0. Set the file-descriptor limit

Recommended to put in your `~/.bashrc`:
```bash
ulimit -n 4096
```

### 1. Load the Python 3.9 environment
```bash
source envs/load_python39.sh
```

### 2. Locate binary files on EOS

Copy from the DAQ PC to CernBox using `scp`, `rsync`, or `xrdcp`.

### 3. Check the Condor server
```bash
condor_q
```
If the worker has too many jobs, switch server:
```bash
myschedd bump
```

---

## Pipeline steps

### 4. Submit decoding jobs
```bash
python submit/submit_decoding.py -d <INPUT_DIR> -o <OUTPUT_DIR> --condor_tag <CONDOR_TAG> --dryrun
```
| Flag | Default | Description |
|---|---|---|
| `-d`, `--input_dir` | *required* | Input directory, after the EOS base path. |
| `-o`, `--output` | *required* | Output directory, after the EOS base path. |
| `--condor_tag` | auto-generated | String to identify the job submission. |
| `--dryrun` | off | Generate the input list, bash script, and condor JDL, but skip actual submission. |

### 5. (Optional) Merge output feather files
```bash
python utils/merge_feathers.py -d <INPUT_DIR> -n <NUMBER_OF_MERGE> --dryrun
```
| Flag | Default | Description |
|---|---|---|
| `-d`, `--input_dir` | *required* | Input directory, after the EOS base path. |
| `-n`, `--number_of_merge` | `10` | Target number of files per merged group. |
| `--dryrun` | off | Show the resulting number of groups without performing the merge. |

### 6. Path finding
```bash
python core/path_finder.py -p <PATH> --cal-label <CAL_LABEL> --track-label <TRACK_LABEL> -c <CONFIG> -r <RUNNAME> -s <SAMPLING> [--max_diff_pixel <N>] [--mask_config <MASK_YAML>] [--cal_table_only] [--find_alignment] [--seed <N>] [--max_files <N>]
```
Produces one Parquet track-candidates file per board combo: every subset down to 3 boards (the minimum `extract_events_by_path.py` can process), e.g. for 4 boards (ids 0-3, with roles trig/ref/dut/extra) that's the four 3-board leave-one-out combos (`trig0-ref1-dut2`, `trig0-ref1-extra3`, `trig0-dut2-extra3`, `ref1-dut2-extra3`). The full board-set combo is deliberately not produced -- its single-hit requirement on every board only ever shrinks the qualifying event set relative to a leave-one-out combo, so its tracks are already a strict subset of each of those, and there's nothing downstream that made use of the extra coincidence requirement. Each board id is tagged with its role from the config YAML so filenames are legible without cross-referencing the config. Each combo's single-hit requirement only applies to the boards in that combo. To avoid every run's combo files piling up flat in one directory, they're nested under a per-run directory named after `--track-label`'s basename, e.g. `-t tracks_csv/desy2026aug_run1` writes to `tracks_csv/desy2026aug_run1/tracks_trig0-ref1-dut2.parquet` (created automatically) -- the run name isn't repeated in the filename since it's already the directory name. No count threshold is applied here -- every surviving track candidate is written; use step 7 to cut on occurrence count. Pick whichever combo file fits the analysis at step 7/8.

Where things are read from and written to. The input `-p` is resolved under your EOS base when relative, but an *absolute* path is used as-is - so you can read a colleague's decoded feathers directly (e.g. `-p /eos/user/j/jongho/.../Run_1_feather`) without copying them. The outputs, on the other hand, are written relative to the current working directory, not to EOS: `--cal-label` is a filename *prefix* (`<CAL_LABEL>_cal_table.csv`), `--track-label` is a *directory*, and the `--find_alignment` yaml goes to `./alignment/`. Run this step from the directory where you want the outputs to live (e.g. a work directory on your EOS), or pass absolute output paths.

Spatial (radius) cut. `--max_diff_pixel` is a coincidence-consistency window, not a straight-line fit: for every board in the combo, its transformed pixel centre must lie within `max_diff_pixel x 1.3 mm` of the *anchor* board's pixel centre. The anchor is the trigger board when it is in the combo, else the combo's median board id. The window tolerates inclined tracks and board offsets up to its size; the yaml `transformation` (translation and, if given by hand, rotation) is what centres it.

Reproducibility. Event sampling and the `--max_files` file subset are random; pass `--seed` to make the CAL table, the candidate lists and the alignment estimate bit-reproducible for a given input set. Runs with more than `--max_files` feather files (default 100) have a random subset read - this is now reported in the log; pass `--max_files 0` to read everything.

| Flag | Default | Description |
|---|---|---|
| `-p`, `--path` | *required* | Directory containing feather files (output of step 4 or 5). Relative: after the EOS base path; absolute: used as-is (any readable location). |
| `--cal-label` | *required* | Output *prefix* for the CAL-code table CSV: writes `<CAL_LABEL>_cal_table.csv`, relative to the current working directory (parent directories are created). |
| `--track-label` | *required* | Per-run output *directory* for the per-combo track-candidates Parquet files (`<TRACK_LABEL>/tracks_<combo>.parquet`), relative to the current working directory. |
| `-c`, `--config` | *required* | Path to the board-config YAML file. |
| `-r`, `--runName` | *required* | Key of the run's entry in the config YAML. |
| `-s`, `--sampling` | `3` | Percent of data to read from each file. |
| `--max_diff_pixel` | `1` | Radius of the spatial coincidence window, in pixels (x 1.3 mm), around the anchor board's pixel (trigger board if in the combo, else the combo's median board id). |
| `--mask_config` | none | Path to a noisy-pixel mask YAML (see `mask_pixel_configs/`). |
| `--cal_table_only` | off | Stop right after building the CAL-code table. |
| `--find_alignment` | off | Estimate board translations two ways and write them, keyed by run, to `./alignment/{TRACK_LABEL basename}_alignment.yaml`: `legacy_per_combo` (each board against the trigger board, from the peak of the count-weighted shift histogram, for every combo that contains the trigger) and `global_relative` (every combo's boards against that combo's median board, combined over all combos in one least-squares fit, reported relative to the pinned trigger board, or the lowest board id when the run has no trigger board). Diagnostic only: nothing is applied to the track output and nothing is saved back into `--config`; merge in manually. |
| `--seed` | none | Seed for all random draws (event sampling, memory-check file sample, `--max_files` subset). Unseeded runs are not reproducible. |
| `--max_files` | `100` | Cap on the number of input feather files read (random subset above it, reported in the log). `0` reads every file. |

### 7. Reduce path-finding output

Step 6 writes every surviving track candidate for each combo unfiltered so this step is required before step 8, not optional; skipping it means step 8 processes the full unreduced candidate set.
```bash
python utils/select_tracks_by_coverage.py -f <FILE_OR_DIR> [-d <TARGET_DEPTH>] [-n <MAX_CANDIDATES>]
```
| Flag | Default | Description |
|---|---|---|
| `-f`, `--file` | *required* | Track-candidates Parquet file for one board combo (from step 6), or the per-run directory step 6 wrote them all into -- every combo file in the directory is reduced in one call. |
| `-d`, `--target-depth` | `4` | Minimum number of candidates required at every pixel on each board's 16x16 grid. Higher values guarantee deeper per-pixel statistics at the cost of more total candidates; the total needed to satisfy a given depth differs per combo/run since it's derived from that file's own data, not a fixed count. |
| `-n`, `--max-candidates` | none | Hard ceiling on candidates kept per file. If `-d` would exceed this, the depth is lowered by 1 and retried (down to depth 1, i.e. coverage only) until it fits. If even depth 1 exceeds `N`, that result is kept anyway with a warning -- coverage is never sacrificed to force a fit. |

Selects a subset of candidates that guarantees every pixel is backed by at least `-d` candidates, instead of a plain top-N-by-count cut (which can leave whole regions of the grid uncovered, especially on lower-statistics runs, while piling many redundant candidates onto a few hot pixels). Candidates are walked in descending occurrence-count order to first reach full coverage using as few as possible, then water-filled -- always topping up whichever pixel currently has the fewest candidates -- until every reachable pixel meets the target depth. Prints per-board pixel coverage, min/max depth achieved, and any pixels never reached by any candidate (worth checking against `mask_pixel_configs/` if any show up). Writes `<same file stem>_reduced.parquet` next to each input file. Already-reduced files (`*_reduced.parquet`) are skipped when scanning a directory, so re-running against the same directory is safe.

#### 7b. (Optional) Telescope / beam diagnostics per run
Once steps 6 and 7 have run for a run, `utils/telescope_diagnostics.py` condenses what the track candidates say about the telescope into one small parquet, and draws it - per run, or across runs to check consistency of an HV/fluence scan:
```bash
python utils/telescope_diagnostics.py summarize --tracks-dir <TRACK_LABEL dir from step 6> --cal-table <CAL_LABEL>_cal_table.csv -c <CONFIG> -r <RUNNAME> --label <RUNNAME> -o <OUTDIR> [--alignment <alignment yaml>] [--raw-dir <feather dir> --n-raw-files 5]
python utils/telescope_diagnostics.py plot -i <OUTDIR>/<RUNNAME>_diagnostics.parquet -o <OUTDIR> [--format png|pdf] [--split]   # one run: figures in <OUTDIR>/<RUNNAME>/
python utils/telescope_diagnostics.py plot -i <OUTDIR>/*_diagnostics.parquet -o <OUTDIR>                    # many runs: <OUTDIR>/compare/trends.png + summary_table.csv
```
Worked example (H1, run 1, outputs under the user's EOS `CERN_IRRAD_Mar2026/analysis/`; run from `condor_at_lxplus/` with `source envs/load_python39.sh`; the alignment yaml is the one step 6 wrote into `alignment/` of the directory it was run from):
```bash
E=/eos/user/m/musafdar/CERN_IRRAD_Mar2026/analysis
python utils/telescope_diagnostics.py summarize --tracks-dir $E/tracks/irradMar_run1_v2 --cal-table $E/cal_tables/irradMar_run1_v2_cal_table.csv -c ../board_configs_yaml/CERN_Irrad_2026Mar.yaml -r h1_run1 --label h1_run1 -o $E/diagnostics --alignment alignment/irradMar_run1_v2_alignment.yaml --raw-dir /eos/user/j/jongho/ETROC_Irrad_2026Mar/H1/Run_1_feather --n-raw-files 5
python utils/telescope_diagnostics.py plot -i $E/diagnostics/h1_run1_diagnostics.parquet -o $E/diagnostics
```

The parquet holds every number the figures need - candidate/selection statistics and geometric consistency, count-weighted hit maps, per-pixel CAL modes, per-pixel TOT medians in raw code and in ns (`tot_ns_median`, the code converted with that pixel's own CAL), raw occupancy (if `--raw-dir`), pairwise pixel offsets, per-track slopes per plane gap, rotation profiles, a relative plane-spacing estimate and the alignment values - and one input turns it into one figure per plain question, drawn in the physical pixel orientation (pixel (0,0) bottom-right when looking down at the ETROC with its wire-bond pads at the bottom edge, so `col<8` is the physical right half): `hit_maps.png`, `occupancy.png`, `cal_tot_halves.png`, `track_landing.png`, `beam_tilt.png`, plus `board_rotation.png` with `--extended`, each with its question as the title and a caption in ordinary words. Several inputs instead write `compare/` - `trends.png` and `summary_table.csv` - which show how the same quantities trend from run to run.

Every caption is written as a reading guide that holds for any run of any campaign - what a flat map, a step at the chip-half line, a shifted peak or a wide distribution would each mean - and run-specific facts appear only as measured numbers annotated on the figure. Each caption sits in a band spanning the full width of the figure so it can be read at a normal font size, and no annotation is drawn on top of data.

Two conventions run through all the figures. TOA and TOT codes are bin counts, not times: CAL is the number of delay-cell gates that flip in the fixed 3.125 ns reference interval, so a pixel's TDC bin is 3.125 ns / CAL and a higher CAL means a smaller bin, i.e. the *same* physical time reads a *higher* code. A code is therefore only comparable between pixels (or chip halves) after conversion to ns. And a narrow, centred beam limits what is measurable: the plane-spacing (straightness) estimate needs angular spread and the rotation fit needs a lit range of anchor rows/columns, so when the beam does not provide them the tool stores NaN plus a `status` row and says so on the figure instead of quoting a number (`summarize` also prints those lines).

How to read each figure (the same guidance, in full sentences, is on the figure itself):
- `hit_maps.png`: a flat map means the plane sits in a diffuse halo rather than in a focused beam (what a telescope off the beam axis sees); a bright compact spot means a focused beam crosses there; a gradient, or one brighter chip half, means a flux gradient across the telescope and/or the chip-half difference shown in `cal_tot_halves.png`; dark edge rows/columns are the geometric acceptance of the coincidence with the neighbouring boards. What the centroid cross means depends on the illumination: with a compact spot (a narrow beam centred on the array) the centre of gravity is the beam centre; under flat halo illumination it is not a beam position at all and mostly reflects the gradient - the count-weighted spread quoted in the caption tells the two apart.
- `occupancy.png`: hits/event around 1 with some 10-20% of events at 2+ hits is a busy halo; hits/event well above 1, or single pixels far brighter than their neighbours, points at noise or a badly behaved channel; a large no-hit fraction is expected rather than alarming - at IRRAD the trigger is the AND of two boards (which two varies by run; boards 1 and 3 for `h1_run1`) and it is one whole-chip trigger bit, not a per-pixel one, so a triggered board can still record nothing in an event, and on top of that the DAQ drops or truncates events and the chip's fast-readout mode omits an event's data words once the L1 buffer is 96 or more deep; since the analysis requires a hit on the pixels of interest, incomplete events fall out downstream. A step in hit rate at the col 7 | 8 line is a chip-half effect (the halves' TDCs sit on different supplies and their discriminators can differ slightly); a step in the TOT code at that line is only the change of TDC bin size and says nothing about threshold.
- `cal_tot_halves.png`: three maps per board - CAL, the TOT code, and the same TOT converted to ns with each pixel's own bin - plus the three column averages. A CAL step exactly at the col 7 | 8 line is expected: per the ETROC2 Reference Manual (rev 0.6, sec 3.7.1, Table 20) the TDCs of the right half (`col<8`) are powered from the digital supply and those of the left half (`col>=8`) from the analog (discriminator) supply. A smooth column-to-column or row-to-row grade in CAL is fabrication process variation of the delay line, since the CAL code counts the delay-cell gates that flip in a fixed interval. A TOT-code step that follows the CAL step is just the change of unit: convert with the per-pixel CAL and it disappears - that is what the ns map and the flat ns trace show - whereas a step that *remains* in ns would be a real threshold/gain difference and is the only version worth chasing. (Verified on `h1_run1`: the TOT-code right/left ratio equals the CAL right/left ratio, and TOT in ns agrees across the halves to about 1% or better (0.3-1.1% board to board).) Isolated odd pixels are ones to watch for masking, and pixels with no measurement are left grey - on the TOT maps, pixels no hit landed on, which is most of the array when the beam is a narrow spot. The pipeline uses the per-pixel CAL, so neither the step nor the grade biases the timing.
- `track_landing.png`: a peak at (0,0) means the boards are aligned and the tracks perpendicular; a shifted peak means a mechanical offset and/or tracks crossing at an angle (not separable without a survey; the pipeline's radius cut is centred by the yaml translation, and this peak is the value to put there); a wider blob for a pair with a longer lever arm is angular spread; a tail on one side only means tracks tilted predominantly one way; a peak far from (0,0) with a narrow blob is a pure mechanical offset.
- `beam_tilt.png`: the tilt is quoted per plane gap, in px and in mm, since an angle in mrad would need the plane spacing along the beam; the caption quotes the run's measured count-weighted average tilt in col and row. Rule of thumb: |tilt| below about 0.1 px/gap means the tracks run along the telescope axis to within the pixel resolution, around 0.3 px/gap (what a telescope standing beside the beam sees) is clearly tilted, above 1 px/gap means a large angle or a board out of place; agreement between the board sets means roughly equally spaced planes and the same beam seen by all, disagreement means unequal spacing or a misplaced board; wide bars mean a large angular spread among the tracks, bars concentrated in one or two positions a narrow, nearly parallel beam.
- `board_rotation.png`: flat profiles mean no relative rotation; a slope means that board is rotated about the beam axis (the slope read as radians is the angle); slopes of opposite sign but clearly unequal size mean a shear/perspective term (tilted planes, say) rather than a rigid rotation. To scale an angle: a rotation `theta` shifts the far edge of the array by `16 px * tan(theta)`, so 0.5 deg -> 0.14 px, 2 deg -> 0.56 px, 4 deg -> 1.1 px; below about 1 deg it is negligible against a radius window of a few pixels (path_finder's `--max_diff_pixel`) and matters only if that window is 1 px. Each panel also states how many of the 16 anchor rows/columns the beam lights up; with fewer than 8 no line is fitted and the panel reads *beam too narrow for a rotation estimate*.

How the numbers on the figures are computed (the same explanations are written on each figure):
- Centroid (red x on `hit_maps.png`): the centre of gravity of that board's map - the mean col and the mean row of the map, weighted by the counts, quoted in pixels and in mm from the array centre (pixel 7.5).
- Landing peak and spread (`track_landing.png`): peak = the most common whole-pixel shift, refined by the count-weighted average of the shifts within +-1 pixel of it (a peak estimator, so the combinatorial tail does not pull it; `path_finder.py --find_alignment` reports the peak bin of a 30-bin histogram of the same shifts, which agrees to within a bin); spread = the count-weighted standard deviation of those shifts.
- Beam tilt mean and spread (`beam_tilt.png`): the outermost-board pixel shift divided by the number of plane gaps, converted with 1 pixel = 1.3 mm; the quoted mean is the count-weighted mean over the selected tracks and the `+-` is the count-weighted standard deviation of that distribution - the spread of the tracks, not the uncertainty on the mean. The figure's own average over board sets weights each set by its candidate statistics.
- Plane spacing and rotation, when the beam is narrow: the spacing estimate is reported only if the count-weighted spread of the first-gap offset exceeds 0.3 px (`spread_d1_<axis>` in the parquet), and the rotation fit only if at least 8 anchor rows/columns are lit (`profile_span_*`); otherwise the value is NaN and a `status` row carries the reason.
- CAL (`cal_tot_halves.png`): the per-pixel mode, i.e. the commonest CAL code on that pixel, which is exactly the per-pixel value the pipeline uses; the TDC bin it implies is 3.125 ns / CAL. Per-half numbers are medians over the pixels of a half and the quoted step is left half minus right half.
- TOT (`cal_tot_halves.png`): the per-pixel median of the raw TOT code (a count of that pixel's TDC bins), and `tot_ns_median`, the median over that pixel's hits of `(2*tot - floor(tot/32)) * 3.125 ns / CAL` using that pixel's own CAL mode - NaN where the pixel has no CAL entry. The ns version is the one comparable across pixels and chip halves.
- hits/event and the 0 / 1 / 2+ fractions (`occupancy.png`): hits/event = all hits that board recorded in the files read, divided by the number of events in those files; the 0 / 1 / 2+ fractions are fractions of that same event count, so they add to 1. Both are raw decoded hits, with no track requirement.
- Everything derived from track candidates is count-weighted (a candidate counts as many times as the events that share its pixel pattern) and comes from the event sample step 6 read, not the whole run.

### 8. Submit jobs for event selection by path
```bash
python submit/submit_extract_events_by_path.py -d <DIRNAME> -t <TRACK> -o <OUTNAME> -c <CONFIG> -r <RUNNAME> --cal_table <CAL_TABLE> --condor_tag <CONDOR_TAG> [--combos <INDICES_OR_LABELS>] [--neighbor_search_method <METHOD>] [--dryrun | --local]
```
Per-track extraction is vectorized across every track candidate at once rather than looping one at a time, so a single file's worth of work is now typically a couple of seconds rather than several -- see `--local` below for when that's fast enough to skip condor entirely.

Whenever `-t` points at a directory, a combo legend is printed first, e.g. for a 4-board run:
```
Board combos for this run (pass to --combos by index or by label):
  0: 0-1-2  (extra0-dut1-ref2)
  1: 0-1-3  (extra0-dut1-trig3)
  2: 0-2-3  (extra0-ref2-trig3)
  3: 1-2-3  (dut1-ref2-trig3)
```
The index is derived from the run's board config (the same combo generation `path_finder.py` itself uses), not from which `*_reduced.parquet` files happen to exist that day -- so index `0` always means board combo `0-1-2` for a given `-c`/`-r`, and stays correct even if that combo hasn't been reduced yet (shown as `(not reduced yet)` in the legend, and `--combos` errors clearly if you ask for it anyway).

| Flag | Default | Description |
|---|---|---|
| `-d`, `--inputdir` | *required* | Directory of step 6 output. |
| `-t`, `--track` | *required* | Reduced track-candidates Parquet file for one board combo, from step 7. Can also be the per-run directory step 6/7 wrote them into -- every `*_reduced.parquet` file in it is then auto-detected and submitted as a separate job, one per combo, all sharing the same `--cal_table` (CAL values are computed once per run, not per combo). Use `--combos` to only process some of them. |
| `--combos` | none | Comma-separated combos to process, restricting `-t` directory auto-detection instead of every combo found. Each entry is either the printed integer index (e.g. `0,2`) or a literal combo label (e.g. `dut1-ref2-trig3`) -- freely mixable (`1,extra0-ref2-trig3`). Ignored when `-t` points directly at a single file. |
| `-o`, `--outdir` | `extractEvents_outputs` | Output directory, after the EOS base path. The board-combo label is parsed from each track file's filename and appended automatically (e.g. `extractEvents_outputs/dut0-trig1-ref2`), so different combos submitted with the same `-o` never collide or get merged together by step 9. |
| `-c`, `--config` | *required* | Path to the board-config YAML file. |
| `-r`, `--runName` | *required* | Key of the run's entry in the config YAML. |
| `--cal_table` | *required* | CAL-code table CSV from step 6, shared by every combo of the same run. |
| `--neighbor_search_method` | `none` | Neighbor-hit search method: `row_only`, `col_only`, `cross`, or `square`. |
| `--condor_tag` | auto-generated | String to identify the job submission. |
| `--dryrun` | off | Generate the input list, bash script, and condor JDL, but skip actual submission. |
| `--local` | off | Skip condor entirely and process every file on this machine instead, using a small local process pool (3 workers) -- no JDL/xrdcp, output written straight to its final directory (`/eos` is already mounted on lxplus interactive nodes). Worth it once processing got fast enough that `n_files x per_file_time` is only a few minutes serially; condor still wins once that stretches to tens of minutes/hours, or you want it running unattended. |

### 9. Reshape output from event-based to track-based
```bash
python core/reshape_event_to_track.py -d <DIRNAME> -o <OUTDIR> -c <CONFIG> -r <RUNNAME> -b <BATCHES> -p <PARTITIONS> [--file_pattern <GLOB>]
```
| Flag | Default | Description |
|---|---|---|
| `-d`, `--inputdir` | *required* | Directory of step 8 output for one board combo (the combo-labeled subdirectory step 8 wrote). Can also be the combo mother directory (one subdirectory per board combo, as step 8 now writes when given a directory for `-t`) -- every combo is then auto-detected and processed in one call. |
| `-o`, `--outdir` | *required* | Output base directory. The combo label (from `-d`'s basename, or each auto-detected combo subdirectory) is appended automatically (e.g. `<outdir>/dut0-trig1-ref2/tracks`), so different combos never collide or get their tracks silently mixed together. |
| `-c`, `--config` | *required* | Path to the board-config YAML file. |
| `-r`, `--runName` | *required* | Key of the run's entry in the config YAML. |
| `-b`, `--batches` | `30` | Total batches to split input files into, for safety. |
| `-p`, `--partitions` | `1` | Number of output partitions (datasets). |
| `--file_pattern` | `*.parquet` | Glob pattern for input files. |

### 10. Submit jobs to apply TDC cuts
```bash
python submit/submit_apply_tdc_cuts.py -d <INPUTDIR> -c <CONFIG> -r <RUNNAME> --TOALower <TOALOWER> --TOAUpper <TOAUPPER> --distance_factor <DISTANCE_FACTOR> --condor_tag <CONDOR_TAG> [--TOALowerTime <NS>] [--TOAUpperTime <NS>] [--convert-first] [--batch_size <N>] [--dryrun]
```
| Flag | Default | Description |
|---|---|---|
| `-d`, `--inputdir` | *required* | Mother directory containing `tracks` / `tracks_groupX` folders (output of step 9). Can also be the combo mother directory (containing one subdirectory per board combo, each with its own `tracks`/`tracks_groupX` folders) -- every combo is then auto-detected and processed in one submission. |
| `-c`, `--config` | *required* | Path to the board-config YAML file. |
| `-r`, `--runName` | *required* | Key of the run's entry in the config YAML. |
| `--TOALower` | `100` | Lower raw-TDC TOA cut boundary. |
| `--TOAUpper` | `500` | Upper raw-TDC TOA cut boundary. |
| `--TOALowerTime` | `2` | Lower TOA cut boundary in physical time (ns). |
| `--TOAUpperTime` | `10` | Upper TOA cut boundary in physical time (ns). |
| `--distance_factor` | `3.0` | Allowed spread (in MAD-derived sigma) for the TOA correlation cut. |
| `--convert-first` | off | Convert to physical time before applying cuts, instead of after. |
| `--batch_size` | `10` | Number of files per condor job. |
| `--condor_tag` | auto-generated | String to identify the job submission. |
| `--dryrun` | off | Generate the input list, bash script, and condor JDL, but skip actual submission. |

Output: one `track_*.parquet` per surviving track in a `time/` directory next to the `tracks/` it came from (`<combo>/time/`, or `time_groupX` for `tracks_groupX`), same filename as the input. Columns: `toa_<role>` / `tot_<role>` in picoseconds; `file` (the step-8 feather index) and the raw `cal_<role>` codes, passed straight through from the input rows; `HasNeighbor_<role>` and `trackNeighbor`. `file` is the only time-ordering handle a track file has (rows are events from the whole run), so keeping it is what allows a resolution-vs-time-in-run study after this step; `cal_<role>` is the one input to the CAL->ps conversion that cannot be recovered from the ps values (the bin size is a per-file *mean* CAL), so it lets the conversion be audited or redone per event. Steps 11-13 ignore both columns. Everything else about a track that steps 8/9 carried (`track_id`, `row_*`/`col_*`) is not in the table; the pixel triple lives in the filename, which steps 11 and 13 parse.

#### 10b. (Optional) Per-track diagnostics: stability along the run and the time-walk correction
Once step 10 has run for a combo, `utils/track_diagnostics.py` reads its `tracks/` (step 9) and `time/` (step 10) directories track by track, condenses what they say about the run's stability and about the time-walk correction into one tidy parquet, and draws it:
```bash
python utils/track_diagnostics.py summarize --tracks-dir <combo>/tracks --time-dir <combo>/time --label <RUNNAME>_<combo> -o <OUTDIR> [--files-per-bin 5] [--min-events-twc 3000] [--workers 4]
python utils/track_diagnostics.py plot -i <OUTDIR>/<RUNNAME>_<combo>_track_diagnostics.parquet -o <OUTDIR> [--t0 <run start ISO> --duration-min <D>] [--file-times <csv>] [--format png|pdf] [--split]   # figures in <OUTDIR>/<RUNNAME>_<combo>/
```
Worked example (H1, run 1, one combo; repeat per combo):
```bash
E=/eos/user/m/musafdar/CERN_IRRAD_Mar2026/analysis; C=ref1-dut2-trig3
python utils/track_diagnostics.py summarize --tracks-dir $E/tracks_h1_run1/$C/tracks --time-dir $E/tracks_h1_run1/$C/time --label h1_run1_$C -o $E/diagnostics/tracks --workers 4
python utils/track_diagnostics.py plot -i $E/diagnostics/tracks/h1_run1_${C}_track_diagnostics.parquet -o $E/diagnostics/tracks --t0 2026-03-11T19:02:56Z --duration-min 720
```

`summarize` is the slow part (every track file is read twice over EOS; ~10 min per combo of ~1100 tracks with 4 workers); `plot` takes seconds and can be re-run freely. The parquet holds, per track: events per raw file entering and surviving step 10; per file and board the mean CAL (on step 10's default cut-then-convert path exactly the CAL it used for that file's bin size; with `--convert-first` step 10 averaged over all rows before the cuts), the median TOA and the mean TOT (ps; the mean because TOT is quantised in ~40 ps steps and a per-file median only hops between codes), the raw inter-board TOA offsets and each board's raw offset against the mean of its two partners; the full-run time-walk-correction (TWC) polynomial of every board - computed with exactly the fit step 12 (`bootstrap.py`) applies on the same events (a step-12 run without `--neighbor_cut`): 2nd-order in the board's own TOT against the mean of the other two boards, two iterations, the two summed into one effective polynomial - reduced to three comparable numbers (slope at the board's median TOT in ps/ns, curvature in ps/ns², and the correction span across the central 80% of the TOT distribution in ps); the same fit redone in bins of `--files-per-bin` raw files for tracks with at least `--min-events-twc` surviving events, and a resolution proxy per bin (robust IQR width of the TWC-corrected pairwise TOA differences, 3-board solve) both with the full-run TWC applied and with the TWC refitted in the bin.

Time axis. The pipeline never records wall-clock time per event; the raw file index (`file`, kept through steps 8-10) is the only ordering, and the DAQ closes a raw file at a fixed *size*, so file index is "accumulated data". `plot` therefore labels the x axis in file index and, given `--t0` and `--duration-min` (the run start and duration from the DAQ's `run_metadata.yaml`, e.g. `max_run_time_minutes` when the run ran to its cap), adds a top axis in hours under the stated assumption of a constant data rate. If the raw binaries' timestamps are available, `--file-times <csv>` (columns `file,time`) replaces the assumption by the measured mapping.

How to read each figure (the same guidance, in full sentences, is on each figure):
- `events_per_file.png`: a flat entering count means a steady beam and DAQ (files are equal-size chunks); a flat survival fraction means the per-file cut windows and the underlying distributions are stable. The last file is normally a partial chunk.
- `cal_drift.png`: per board, the mean CAL of each file's surviving events minus the track's run average - on the default cut-then-convert path exactly the CAL step 10 used for that file's bin (3.125 ns / mean CAL). CAL counts delay-line cells flipping in the fixed 3.125 ns reference, so a drift here is the delay line (the bin) changing with temperature/supply, not the clock. One code at CAL ~155 is 0.13 ps of bin, up to ~80 ps at the far end of the TOA range; a slow drift is absorbed by the per-file mean, a change within a file is not.
- `toa_tot_drift.png`: top, per board, the mean TOT per file relative to the run (charge / threshold drift); bottom, per board pair, the raw inter-board TOA offset per file relative to the run (relative clock phase / latency / temperature of one chip). Common-mode TOA drift cancels and is invisible. A static per-track offset is absorbed by the TWC constant term; a drift within the run is not - step 12 fits one constant per track over the whole run, so a drift adds its RMS in quadrature to the pair widths (a linear end-to-end drift D adds D/sqrt(12)).
- `twc_curves.png`: one TWC curve per track, each drawn over its own central 80% of TOT and shifted to 0 at the median over tracks of the per-track median TOT (the constant term is a per-track offset and is not comparable, the shape is), coloured by pixel column from blue (col 0, physical right edge) to red (col 15, physical left edge). Tight bundle = one correction serves the whole array; colour order = position dependence; wide bundle without colour order = pixel scatter or fit noise.
- `twc_array_maps.png`: per board whose pixel varies, maps of the slope, span and curvature over the array (event-weighted mean over the tracks sharing a pixel; colour scale clipped at the 2nd-98th percentile), plus the spread (sample std) of the slope over the tracks that share a pixel - if that spread is as large as the pixel-to-pixel structure, the structure is fit noise. Sign of the slope: the correction is added to the TOA and fitted to (mean of the other two boards) - (this board), so a positive slope means the board's TOA comes earlier with increasing TOT relative to its partners, the usual time walk. Caveat of the coupled fit: each board is corrected against the mean of the other two, so a real feature of one chip is partly mirrored with the opposite sign onto its partners; compare the same board across the combos it appears in - a feature that persists with different partners belongs to that chip, one that flips sign or migrates to a partner is the fit's bookkeeping.
- `twc_board_compare.png`: box plots over tracks of the same summaries and of the resolution proxy, one box per board; overlapping boxes = compatible corrections.
- `twc_vs_time.png`: for each bin of files, the change of the TWC slope and span relative to the full-run fit (rows 1-2) and the change of the board's raw TOA offset against the mean of the other two boards (row 3; shown directly rather than via the fitted constant, because the coupled 2-iteration fit maps a drift D of one board to -D/2 on that board and +D/4 on each partner - the fitted change at the median TOT is kept in the parquet as `d_corr_med`), median over tracks with 16-84% band. Flat rows 1-2 within a few ps = the correction shape is stable and one full-run fit is adequate; a trend in row 3 alone = the boards' relative TOA offset drifted (clocks/latencies), not the time walk.
- `resolution_proxy_vs_time.png`: the resolution proxy per bin with the full-run TWC applied (what step 12 effectively does) and with the TWC refitted in the bin; if the two agree the TWC's drift is irrelevant to the resolution, and a systematically lower refit is an upper bound on what a time-dependent TWC could recover (the in-bin refit also fits noise: a few tenths of a ps at the smallest bins). The dotted full-run reference is over the same tracks; bin values slightly below it are the cost of the slow inter-board offset drift, which adds in quadrature over the run but not within a bin. This proxy is not the step-12 number (GMM FWHM, bootstrap): read it for trends and comparisons, not for the absolute value.

How the numbers are computed: the effective TWC polynomial P(TOT) = a2·TOT² + a1·TOT + a0 [ps, TOT in ps] is the sum of the two iterations' `np.polyfit(tot, 0.5·(toa_other1 + toa_other2) − toa, 2)` fits, exactly as `bootstrap.py` applies them; slope at the median TOT = 2·a2·TOT_med + a1 (×1000 → ps/ns); curvature = 2·a2 (×10⁶ → ps/ns²); span = P(TOT_p90) − P(TOT_p10) with the percentiles of that board's TOT over the whole run; the resolution proxy uses IQR/1.349 of each TWC-corrected pairwise TOA difference and σ_a² = (s_ab² + s_ac² − s_bc²)/2. All "per file" numbers use the events that survived step 10 (the CAL mean is therefore the one step 10 used); the entering counts come from the step-9 files. Medians and 16-84% bands are over tracks (blank where fewer than 5 tracks have data); the per-bin TWC study is restricted to tracks with ≥ `--min-events-twc` events and bins with ≥ 150 events. Tracks that step 10 filtered to nothing (no `time/` file) are counted as entering with 0 survivors; the entering counts come from the step-9 files, everything else from the step-10 survivors.

### 11. Count events per track
```bash
python core/count_path_nevts.py -d <INPUTDIR> -o <OUTPUTDIR> [--tag <TAG>]
```
| Flag | Default | Description |
|---|---|---|
| `-d`, `--inputdir` | *required* | Directory containing step 10 output (a `time`/`time_groupX` folder or its parent). Can also be the combo mother directory (one subdirectory per board combo, each with its own `time`/`time_groupX` folders) -- every combo is then auto-detected and processed, with output CSVs distinguished by combo label. |
| `-o`, `--outputdir` | *required* | Output directory. |
| `--tag` | none | Additional string appended to the output filename. |

### 12. Submit jobs for bootstrap
```bash
python submit/submit_bootstrap.py -d <DIRNAME> -o <OUTPUTDIR> -n <NUM_BOOTSTRAP_OUTPUT> --minimum_nevt <MINIMUM_NEVT> --iteration_limit <ITERATION_LIMIT> --condor_tag <CONDOR_TAG> [--reproducible] [--ks_pmin <P>] [--ks_pmin_floor <P>] [--ks_dmax <D>] [--gmm_tol <TOL>] [--gmm_max_iter <N>] [--neighbor_cut <COL> ...] [--neighbor_logic <OR|AND>] [--dryrun]
```
| Flag | Default | Description |
|---|---|---|
| `-d`, `--inputdir` | *required* | Directory containing step 10 output (a `time`/`time_groupX` folder or its parent). Can also be the combo mother directory (one subdirectory per board combo, each with its own `time`/`time_groupX` folders) -- every combo is then auto-detected and processed, each writing to its own `bootstrap_<outputdir>/<combo_label>[_groupX]` output directory. |
| `-o`, `--outputdir` | *required* | Output directory base name. |
| `-n`, `--num_bootstrap_output` | `100` | Target number of bootstrap results. |
| `--minimum_nevt` | `1000` | Minimum event count required to run bootstrap. |
| `--iteration_limit` | `7500` | Maximum number of bootstrap trials. |
| `--reproducible` | off | Seed resampling (not the GMM fit) so results are reproducible run-to-run. |
| `--ks_pmin` | `1e-3` | A pair's mixture fit is accepted if its KS p-value is at least this or its KS distance is at most `--ks_dmax`. The threshold is halved after each failed single-shot attempt and after each 100 fruitless bootstrap attempts, down to `--ks_pmin_floor` (`1e-6`; the single-shot tries the floor once, the bootstrap phase keeps drawing at the floor until `--iteration_limit`); the bootstrap phase starts at the threshold the single-shot was accepted at. The former rule was the distance alone (D ≤ 0.03, relaxed upward in steps of 0.001 to 0.05 when nothing passed), a statistical rather than a quality gate: the distance of a good fit shrinks like 0.83/√N, so it rejected most good fits below ~800 events (a biased selection of resamples, bootstrap error too small) and nothing above ~3000. The p-value is the same test on an N-independent scale; 1e-3 is lenient on purpose and still removes broken fits (spikes, degenerate mixtures). |
| `--ks_dmax` | `0.03` | A fit within this KS distance of the data is always accepted, so the accepted set is a superset of the old rule's at its default threshold at every N (a very large sample cannot reject a fit for a small model imperfection); the old upward relaxation of the distance is not kept, the p-value threshold relaxes instead. |
| `--gmm_tol`, `--gmm_max_iter` | `1e-6`, `2000` | EM convergence of the Gaussian-mixture fit. sklearn's defaults (`1e-3`, `100`) leave the fit unconverged on these distributions - EM stops after ~10 iterations with `converged_ = True`, the peak is too high and narrow, and FWHM/2.355 comes out ~12 % low on tracks with thousands of events (measured 43.4 → 49.2 ps on a 21k-event pair, σ_dut 31.6 → 35.6 ps), with KS p-values ~0.05-0.1 that become ~0.9-0.99 once converged. The residual change between `1e-6` and `1e-7` (~0.7 %) is the intrinsic softness of an FWHM-of-a-mixture width and should be quoted as a systematic. `--gmm_max_iter` is a ceiling: converged fits take ~40 iterations at 300 events and 130-200 at 20k, 2-3× the default cost. |
| `--neighbor_cut` | `none` | Space-separated board columns for neighbor cuts, e.g. `HasNeighbor_dut HasNeighbor_ref`. |
| `--neighbor_logic` | `OR` | Combine multiple `--neighbor_cut` columns with `OR` or `AND`. |
| `--condor_tag` | auto-generated | String to identify the job submission. |
| `--dryrun` | off | Generate the input list, bash script, and condor JDL, but skip actual submission. |

#### 12b. (Optional) Bootstrap diagnostics: anatomy of a trial, statistical health, partner consistency
Once step 12 has run, `utils/bootstrap_diagnostics.py` shows what the resolution extraction actually did. It imports and re-uses `bootstrap.py`'s and `fit_bootstrap_results.py`'s own functions and module defaults (gate thresholds, EM convergence), so the fits it draws are the pipeline's at its default settings; a boot file does not record the flags its job ran with, so a replay of an output made with non-default `--ks_pmin`/`--gmm_tol` differs accordingly (and one more caveat: the Gaussian-mixture fit is unseeded in the pipeline, so a replay reproduces the procedure, not bit-for-bit numbers: typically ~0.1 ps apart for tracks with thousands of events, many ps where the mixture is unstable (a few hundred events) - both numbers are printed on the figure, together with the accept/reject verdict step 12 would give the replayed attempt).
```bash
python utils/bootstrap_diagnostics.py anatomy -f <combo>/time/track_X.parquet --boot-file bootstrap_<run>/<combo>/track_X_boot.parquet [--diag-parquet <track_diagnostics parquet>] [--replay-boot N] -o <OUTDIR>
python utils/bootstrap_diagnostics.py stats -d bootstrap_<run> --time-base <EOS .../tracks_<run>> [--log-dir condor_logs/bootstrap/<tag>] -o <OUTDIR>
python utils/bootstrap_diagnostics.py consistency -d bootstrap_<run> --time-base <EOS .../tracks_<run>> [--diag-dir <dir of track_diagnostics parquets>] -o <OUTDIR>
python utils/bootstrap_diagnostics.py compare -a <step-12 output dir A> -b <step-12 output dir B> --time-base <EOS .../tracks_<run>> [--log-dir-a <condor_logs/bootstrap/<tagA>> --log-dir-b <...tagB>] -o <OUTDIR>   # two configurations track by track
```
Worked example (H1, run 1, all four combos; `bootstrap_h1_run1` is the local step-12 output dir):
```bash
E=/eos/user/m/musafdar/CERN_IRRAD_Mar2026/analysis; O=$E/diagnostics/bootstrap
python utils/bootstrap_diagnostics.py stats -d bootstrap_h1_run1 --time-base $E/tracks_h1_run1 --log-dir condor_logs/bootstrap/h1run1_step12_conv -o $O/stats
python utils/bootstrap_diagnostics.py consistency -d bootstrap_h1_run1 --time-base $E/tracks_h1_run1 --diag-dir $E/diagnostics/tracks -o $O/consistency
python utils/bootstrap_diagnostics.py anatomy --tag highest-stat -f $E/tracks_h1_run1/ref1-dut2-trig3/time/track_t-R7C1_d-R7C1_r-R7C1.parquet --boot-file bootstrap_h1_run1/ref1-dut2-trig3/track_t-R7C1_d-R7C1_r-R7C1_boot.parquet --diag-parquet $E/diagnostics/tracks/h1_run1_ref1-dut2-trig3_track_diagnostics.parquet -o $O/anatomy
python utils/bootstrap_diagnostics.py compare -a bootstrap_h1_run1_ksD030 -b bootstrap_h1_run1_pval_deftol --time-base $E/tracks_h1_run1 --log-dir-a condor_logs/bootstrap/h1run1_step12 --log-dir-b condor_logs/bootstrap/h1run1_step12_pval -o $O/compare   # old vs new acceptance rule (same convergence)
python utils/bootstrap_diagnostics.py compare -a bootstrap_h1_run1_pval_deftol -b bootstrap_h1_run1 --time-base $E/tracks_h1_run1 -o $O/compare                                   # default vs converged mixture fit
```
Every figure carries its full caption (what is plotted, how each number is computed, how to read it).

- `anatomy` (one track; three figures in `<OUTDIR>/<track>/`): `anatomy_twc.png` - per board, the quantity step 12 fits (half the sum of the other two boards' TOA minus this board's TOA) against this board's TOT with the iteration-1 polynomial, the same quantity after one correction with the iteration-2 polynomial, and the iteration-2 fit residual with its median per TOT bin (flat = time walk removed; centred on 0 by construction of the fit - a constant offset survives the coupled iterations but enters no pair width); `anatomy_pairs.png` - per board pair, the TWC-corrected TOA-difference histogram with every candidate mixture (1/2/3 components, KS distance each), the kept one, its FWHM bar and FWHM/2.355, the KS threshold verdict, and the 3-board solve with step 12's accept/reject verdict for this attempt (KS above threshold, no usable FWHM, imaginary solve) next to what the actual step-12 job wrote (single-shot or its -1 placeholder, bootstrap median ± std of the accepted resamples) and the IQR proxy; a -1 track still gets `res_`/`err_` from its bootstrap rows in step 13, which is worth checking here; `anatomy_bootstrap.png` - the `-n` bootstrap values per board with the step-13 Gaussian fit (median/IQR-seeded, ±2.5σ window, unbinned ML: μ → `res_<role>`, σ → `err_<role>`), the single-shot value, and, with `--replay-boot N`, N resamples replayed here for comparison.
- `stats` (all tracks; `boot_summary.csv` + per combo `stats_errors.png`, `stats_boards.png`, `stats_ks_gate.png`, `stats_sigma_vs_n.png`, `stats_maps.png`): bootstrap std vs event count against a 1/√N line, relative error, where the single-shot sits in its bootstrap distribution (expected centred at 0 with a width of ~0.1, the noise of the median of 200 values; a width near 1 means a track-dependent bootstrap bias), skewness, completeness (rows per track, -1 placeholders) and, from the condor logs, the number of rejected bootstrap resamples per track (mixture KS above threshold vs imaginary solve, and how many tracks needed more than one single-shot attempt); and the distribution over tracks of the per-track resolutions per board (single-shot and bootstrap median) with the robust Gaussian fit step 13 applies per track, here applied across tracks - a preview of a board-level number (μ; the pipeline itself quotes none, and a downstream fit over pixels would use `res_<role>`, the bootstrap μ) and of the pixel-to-pixel spread (σ; the bootstrap errors of the tracks are not propagated into it). `stats_ks_gate.png`: the fraction of rejected attempts per track vs event count against what a fixed KS-distance gate does to a perfect fit (the Kolmogorov curve P(K > 0.03·√N)) and the 0.83/√N median line - the reason for the p-value gate; `stats_sigma_vs_n.png`: per-board σ vs event count with running medians (a healthy estimator is flat); `stats_maps.png`: per-board σ over the array (1/err²-weighted per pixel, failed tracks excluded) and the split by chip half (col < 8 physical right, col ≥ 8 left) with the error-weighted left − right difference.
- `consistency` (all combos found under `-d`; `consistency_summary.png` = per board one box of per-pixel σ per combo it appears in; one figure per board and per pair of combos it appears in; one pair-width figure per pair of combos, from the step-12 pair widths when the boot files carry them, else from the track_diagnostics proxy): for each pixel of a board, the inverse-variance-weighted (by bootstrap error) mean single-shot σ over the tracks using it, in combo A and in combo B - scatter with errors, difference over the array, pull with the two errors combined as if independent (they are not: the combos share most events, so the null width is below 1, ~0.6-0.9 - a width of ~1 or more, or a mean far from 0, is a firm systematic), and per combo the spread of σ over the different partner pixels sharing a pixel next to the per-track bootstrap error. `consistency_pairs_*.png` uses the `track_diagnostics` IQR proxy to compare the width of the SAME pixel pair (e.g. dut-ref) between the two combos, i.e. between two event samples that differ in which third board also fired (and, with it, in the per-combo step-10 windows/correlation cut and the coupled TWC) - the test that separates a third-board bias through event selection (pair widths shift) from one through the 3-board solve (pair widths stable while the per-board σ shifts: correlated jitter, a partner's residual time walk); tracks with fewer than 500 events are left out.
- `compare` (two step-12 output sets for the same tracks): per-track ratio B/A of the single-shot resolutions and pair widths against event count (`compare_*.png`) and, when both condor log dirs are given, `compare_gate_*.png` - the fraction of resamples each acceptance rule threw away vs N against the Kolmogorov expectation for a perfect fit, and the effect of that selection on the bootstrap error and on the central value. Used to quantify the mixture-convergence fix (×1.12 on every board and pair) and the acceptance-rule change (old rule: most resamples rejected below ~800 events (94 % at 300-600 events in run 1) → bootstrap error up to ~8 % too small there, central values unchanged).

Reading guide: a healthy extraction has flat TWC residuals, mixtures with KS ≪ 0.03, near-Gaussian bootstrap distributions a few % wide, errors on the 1/√N line, single-shot pulls centred at 0, and per-board σ agreeing between combos within the pulls; a per-board σ that shifts between combos while the shared pair widths do not is a partner effect inside the 3-board solve, not a property of the board.

Each `track_*_boot.parquet` holds one single-shot row (`is_bootstrap == False`) and up to `-n` bootstrap rows: the resolution per role (columns named by role), the three pair widths `pair_<a>-<b>` (FWHM/2.355 of the kept mixture) that the 3-board solve used, and `ksp_min`, the smallest KS p-value over the three pair fits of that sample. A phase that failed every attempt writes a single row of `-1` placeholders (all columns). The job log carries a `[Summary]` line per phase (accepted/attempts, final threshold) and warns when the bootstrap acceptance rate is below 0.7 - the regime where the accepted resamples are a biased subset.

### 13. Merge bootstrap results (unbinned Gaussian fit)
```bash
python core/fit_bootstrap_results.py -d <INPUTDIR> -o <OUTPUTDIR> --sigma_cut <COEFF> [--tag <TAG>]
```
| Flag | Default | Description |
|---|---|---|
| `-d`, `--inputdir` | *required* | Directory containing bootstrap output files (step 12) for one combo/group. Can also be the combo mother directory (one subdirectory per board combo/group, each holding `*_boot.parquet` files) -- every one is then auto-detected and processed, with output CSVs distinguished by label, same as step 11. |
| `-o`, `--outputdir` | *required* | Output directory. **Recommended: reuse the same directory as step 11.** |
| `--sigma_cut` | `2.5` | Sigma multiplier used to determine the fit range. |
| `--tag` | none | Additional string appended to the output filename. |

---


Besides `res_*`, `err_*`, `single_shot_res_*` and the audit metrics, the table carries `single_shot_failed_<col>` (1 when step 12 wrote a `-1` placeholder for that track's single-shot, i.e. every full-sample attempt failed by KS or by an imaginary 3-board solve; the `res_/err_` of such a track come from whatever resamples were accepted and can be unphysical, e.g. an imaginary solve barely turned real - exclude or mark them downstream), `n_boot` (accepted bootstrap rows; 0 when the bootstrap phase failed) and `boot_failed` (1 when the bootstrap phase itself never succeeded). Note that a near-degenerate 3-board solve can also come out barely real (e.g. σ of a few ps next to pair widths of 60-100 ps) without any flag: check `res_pair_*` against `res_<role>` downstream. Pair widths appear as `res_pair_<a>-<b>` etc.

#### 13b. Quoting the resolutions - one recipe for any beam, campaign and number of boards
`utils/quote_resolution.py` turns the step-13 table(s) into quotable numbers with a fixed recipe, so a three-plane telescope (one table) and a four-board telescope (one table per leave-one-out combo) are treated identically. It produces two per-pixel maps and two board-level numbers per board: the average map (canonical - every track using the pixel, 1/err²-weighted, i.e. the operating resolution under this illumination) and the central-hit map (only tracks whose partner pixels sit at the modal offset - one sub-region of the pixel, the more geometry-independent number for chip-to-chip comparisons); both maps are also written as CSV (`<label>_map_<board>_{average,central}.csv`):
```bash
python utils/quote_resolution.py -i final_<run>/resolution_table_*.csv -o final_<run>/quote --label <run>     # writes <run>_resolution_quote.{md,json,png}
```
Recipe: (1) drop tracks with `single_shot_failed`/`boot_failed`, fewer than `--min-boot` resamples, or a near-degenerate solve (solved σ below `--margin-lo` = 0.35 or above `--margin-hi` = 0.95 of the smallest pair width it came from); (2) per pixel, the 1/err²-weighted mean of `res_` over the tracks using it (the standard resolution table) plus the spread of `res_` over those tracks; (3) per board and table, the robust Gaussian mean over pixels (the same `perform_robust_unbinned_fit` step 13 uses) = the value, its width = pixel-to-pixel spread (quoted as spread), stat error = width/√N_pix; (4) systematics with the same names everywhere: partner = per pixel, half the range of that pixel's per-table values, median over the pixels present in ≥ 2 tables (so different pixel coverage per table - an angled beam fires different rows in different combos - cannot fake a partner shift), or, with one table or no pixel in two tables, the median excess of the track-to-track spread over the bootstrap error per pixel (how much the answer depends on which partner pixels were used - exists with three boards); definition = `--def-syst` (1 %, the convergence softness of the FWHM-of-a-mixture width; the core-vs-RMS convention is stated, not folded in); chip halves = error-weighted left − right, reported; illumination = the same pixel measured by tracks whose partners sit at the modal offset (one sub-region of the pixel, set by the fractional inter-plane alignment) versus off-nominal partners (the complementary side sub-region, a few hundred μm at 1.3 mm pitch, not a charge-sharing edge): the quoted value is the event-weighted average of both (the operating resolution under this alignment/beam), and the central-only value, the off-nominal − central difference and the off-nominal event share are reported (event weights from the step-11 `nevt_*.csv` if present next to the tables); (5) the quoted number = the robust Gaussian mean over the COMBINED per-pixel map (all tables' cleaned tracks pooled per pixel; a pixel present in one table only just gets that table's tracks) ± stat ± partner ± definition; coverage (pixels per table, in the union, in ≥ 2 tables) is reported. With ≥ 4 boards the least-squares solve of all pair widths is printed as a consistency check (never as the number). Worked example: `python utils/quote_resolution.py -i "/eos/user/m/musafdar/CERN_IRRAD_Mar2026/analysis/final_h1_run1/resolution_table_*.csv" -o .../final_h1_run1/quote --label h1_run1`.

## Utilities

Standalone helpers in `utils/`, not part of the numbered pipeline above. Run with `--help`
for the full flag list.

| Script | Purpose |
|---|---|
| `get_my_job_list.sh` | Dumps your condor job history into a summary table. |
| `get_job_completion_info.sh <cluster_id>` | Summarizes exit/termination info for a given condor cluster. |
| `print_file_size_table.py` | Prints row/event counts per dataframe in a directory. |
| `find_cal_per_file.py` | Computes CAL-code mean/mode per input file, saved to sqlite. |
| `extract_twc_coeffs.py` | Fits time-walk-correction (TWC) polynomial coefficients per board. |
| `telescope_diagnostics.py` | Per-run telescope diagnostics: `summarize` condenses one run's step-6/step-7 outputs into one tidy parquet, `plot` draws the per-run dashboard or cross-run trends. |
| `track_diagnostics.py` | Per-track diagnostics after step 10: `summarize` condenses one combo's step-9/step-10 outputs (events per file, CAL/TOT/TOA drift, the time-walk correction per track and per bin of files, a resolution proxy along the run) into one tidy parquet, `plot` draws the figures (per-pixel maps are built from the per-track rows at plot time). |
| `bootstrap_diagnostics.py` | Step-12/13 diagnostics: `anatomy` replays one track through the pipeline's own TWC / mixture / 3-board-solve / step-13 fit and draws every stage with the fit on the data; `stats` checks the bootstrap's statistical behaviour over all tracks (errors vs N, pulls, completeness, rejected attempts from the logs); `consistency` compares the same board and the same pixel pair across board combos. |
| `quote_resolution.py` | Quoted per-board resolutions from the step-13 table(s) with one fixed recipe (cleaning, per-pixel weighted table, robust Gaussian over pixels, partner / definition systematics, chip halves) that applies to a 3-plane telescope and to a multi-combo one alike; markdown + JSON + one figure. |
