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
  -> [7]  reduce path candidates       (optional, local)
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
  - [7. Check path-finding output (optional)](#7-check-path-finding-output-optional)
  - [8. Submit jobs for event selection by path](#8-submit-jobs-for-event-selection-by-path)
  - [9. Reshape output from event-based to track-based](#9-reshape-output-from-event-based-to-track-based)
  - [10. Submit jobs to apply TDC cuts](#10-submit-jobs-to-apply-tdc-cuts)
  - [11. Count events per track](#11-count-events-per-track)
  - [12. Submit jobs for bootstrap](#12-submit-jobs-for-bootstrap)
  - [13. Merge bootstrap results (unbinned Gaussian fit)](#13-merge-bootstrap-results-unbinned-gaussian-fit)
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
python core/path_finder.py -p <PATH> --cal-label <CAL_LABEL> --track-label <TRACK_LABEL> -c <CONFIG> -r <RUNNAME> -s <SAMPLING> -m <NTRACKS> [--max_diff_pixel <N>] [--mask_config <MASK_YAML>] [--cal_table_only] [--find_alignment] [--exclude_role <EXCLUDE_ROLE>]
```
| Flag | Default | Description |
|---|---|---|
| `-p`, `--path` | *required* | Directory containing feather files (output of step 4 or 5), after the EOS base path. |
| `--cal-label` | *required* | Output filename for the CAL-code table CSV. |
| `--track-label` | *required* | Output filename for the track-candidates CSV. |
| `-c`, `--config` | *required* | Path to the board-config YAML file. |
| `-r`, `--runName` | *required* | Key of the run's entry in the config YAML. |
| `-s`, `--sampling` | `3` | Percent of data to read from each file. |
| `-m`, `--minimum` | `1000` | Minimum occurrence threshold for a track candidate. |
| `--max_diff_pixel` | `1` | Max allowed pixel-position spread across boards for spatial alignment. |
| `--mask_config` | none | Path to a noisy-pixel mask YAML (see `mask_pixel_configs/`). |
| `--cal_table_only` | off | Stop right after building the CAL-code table. |
| `--find_alignment` | off | Report per-board pixel offsets relative to the trigger board. |
| `--exclude_role` | none | Board role to exclude from path finding. One of `trig`, `dut`, `ref`, `extra`. |

### 7. Check path-finding output (optional)

Recommended if the number of paths from step 6 exceeds ~1.5k.
```bash
python utils/reduce_number_of_track_candidates.py -f <FILE> -m <NUMBER> [--ntrk_table]
```
| Flag | Default | Description |
|---|---|---|
| `-f`, `--file` | *required* | Track-candidates CSV from step 6. |
| `-m`, `--minimum` | `1000` | New, tighter minimum occurrence threshold. |
| `--ntrk_table` | off | Print a table of surviving paths for thresholds 40–400 (step 40). |

### 8. Submit jobs for event selection by path
```bash
python submit/submit_extract_events_by_path.py -d <DIRNAME> -t <TRACK> -o <OUTNAME> -c <CONFIG> -r <RUNNAME> --cal_table <CAL_TABLE> --condor_tag <CONDOR_TAG> [--neighbor_search_method <METHOD>] [--dryrun]
```
| Flag | Default | Description |
|---|---|---|
| `-d`, `--inputdir` | *required* | Directory of step 6 output. |
| `-t`, `--track` | *required* | Track-candidates CSV from step 7 (or step 6 if step 7 was skipped). |
| `-o`, `--outdir` | `extractEvents_outputs` | Output directory, after the EOS base path. |
| `-c`, `--config` | *required* | Path to the board-config YAML file. |
| `-r`, `--runName` | *required* | Key of the run's entry in the config YAML. |
| `--cal_table` | *required* | CAL-code table CSV from step 6. |
| `--neighbor_search_method` | `none` | Neighbor-hit search method: `row_only`, `col_only`, `cross`, or `square`. |
| `--condor_tag` | auto-generated | String to identify the job submission. |
| `--dryrun` | off | Generate the input list, bash script, and condor JDL, but skip actual submission. |

### 9. Reshape output from event-based to track-based
```bash
python core/reshape_event_to_track.py -d <DIRNAME> -o <OUTDIR> -c <CONFIG> -r <RUNNAME> -b <BATCHES> -p <PARTITIONS> [--file_pattern <GLOB>]
```
| Flag | Default | Description |
|---|---|---|
| `-d`, `--inputdir` | *required* | Directory of step 8 output. |
| `-o`, `--outdir` | *required* | Output base directory. |
| `-c`, `--config` | *required* | Path to the board-config YAML file. |
| `-r`, `--runName` | *required* | Key of the run's entry in the config YAML. |
| `-b`, `--batches` | `30` | Total batches to split input files into, for safety. |
| `-p`, `--partitions` | `1` | Number of output partitions (datasets). |
| `--file_pattern` | `*.parquet` | Glob pattern for input files. |

### 10. Submit jobs to apply TDC cuts
```bash
python submit/submit_apply_tdc_cuts.py -d <INPUTDIR> -c <CONFIG> -r <RUNNAME> --TOALower <TOALOWER> --TOAUpper <TOAUPPER> --distance_factor <DISTANCE_FACTOR> --condor_tag <CONDOR_TAG> [--TOALowerTime <NS>] [--TOAUpperTime <NS>] [--exclude_role <ROLE>] [--convert-first] [--batch_size <N>] [--dryrun]
```
| Flag | Default | Description |
|---|---|---|
| `-d`, `--inputdir` | *required* | Mother directory containing `tracks` / `tracks_groupX` folders (output of step 9). |
| `-c`, `--config` | *required* | Path to the board-config YAML file. |
| `-r`, `--runName` | *required* | Key of the run's entry in the config YAML. |
| `--TOALower` | `100` | Lower raw-TDC TOA cut boundary. |
| `--TOAUpper` | `500` | Upper raw-TDC TOA cut boundary. |
| `--TOALowerTime` | `2` | Lower TOA cut boundary in physical time (ns). |
| `--TOAUpperTime` | `10` | Upper TOA cut boundary in physical time (ns). |
| `--distance_factor` | `3.0` | Allowed spread (in MAD-derived sigma) for the TOA correlation cut. |
| `--exclude_role` | `trig` | Role excluded from cut calculations. |
| `--convert-first` | off | Convert to physical time before applying cuts, instead of after. |
| `--batch_size` | `10` | Number of files per condor job. |
| `--condor_tag` | auto-generated | String to identify the job submission. |
| `--dryrun` | off | Generate the input list, bash script, and condor JDL, but skip actual submission. |

### 11. Count events per track
```bash
python core/count_path_nevts.py -d <INPUTDIR> -o <OUTPUTDIR> [--tag <TAG>]
```
| Flag | Default | Description |
|---|---|---|
| `-d`, `--inputdir` | *required* | Directory containing step 10 output (a `time`/`time_groupX` folder or its parent). |
| `-o`, `--outputdir` | *required* | Output directory. |
| `--tag` | none | Additional string appended to the output filename. |

### 12. Submit jobs for bootstrap
```bash
python submit/submit_bootstrap.py -d <DIRNAME> -o <OUTPUTDIR> -n <NUM_BOOTSTRAP_OUTPUT> -s <SAMPLING> --minimum_nevt <MINIMUM_NEVT> --iteration_limit <ITERATION_LIMIT> --condor_tag <CONDOR_TAG> [--reproducible] [--neighbor_cut <COL> ...] [--neighbor_logic <OR|AND>] [--dryrun]
```
| Flag | Default | Description |
|---|---|---|
| `-d`, `--inputdir` | *required* | Directory containing step 10 output. |
| `-o`, `--outputdir` | *required* | Output directory base name. |
| `-n`, `--num_bootstrap_output` | `100` | Target number of bootstrap results. |
| `-s`, `--sampling` | `75` | Percent of events resampled per bootstrap draw. |
| `--minimum_nevt` | `1000` | Minimum event count required to run bootstrap. |
| `--iteration_limit` | `7500` | Maximum number of bootstrap trials. |
| `--reproducible` | off | Seed resampling (not the GMM fit) so results are reproducible run-to-run. |
| `--neighbor_cut` | `none` | Space-separated board columns for neighbor cuts, e.g. `HasNeighbor_dut HasNeighbor_ref`. |
| `--neighbor_logic` | `OR` | Combine multiple `--neighbor_cut` columns with `OR` or `AND`. |
| `--condor_tag` | auto-generated | String to identify the job submission. |
| `--dryrun` | off | Generate the input list, bash script, and condor JDL, but skip actual submission. |

### 13. Merge bootstrap results (unbinned Gaussian fit)
```bash
python core/fit_bootstrap_results.py -d <INPUTDIR> -o <OUTPUTDIR> --sigma_cut <COEFF> [--tag <TAG>]
```
| Flag | Default | Description |
|---|---|---|
| `-d`, `--inputdir` | *required* | Directory containing bootstrap output files (step 12). |
| `-o`, `--outputdir` | *required* | Output directory. **Recommended: reuse the same directory as step 11.** |
| `--sigma_cut` | `2.5` | Sigma multiplier used to determine the fit range. |
| `--tag` | none | Additional string appended to the output filename. |

---

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
