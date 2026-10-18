# IRRAD 2026 inputs

The notebooks (`../notebooks/iv.ipynb`, `merged.ipynb`, `maps.ipynb`, `overview.ipynb`,
`frontend.ipynb` and `universality.ipynb`) draw
the CERN IRRAD 2026 figures (March and July; the universality notebook draws DESY 2026 runs beside
them) from two kinds of input, in two EOS locations. Raw data (the March IV
scans, the slow-control logs and the week-1 spark-test log, read by the IV notebook only) are read
in place from the March tree:

    /eos/user/m/musafdar/CERN_IRRAD_Mar2026/IVCurves

(`EOS_ROOT` in `irrad_2026.py`; its `week1/`, `week2/` and `laptop_mirror_15e14/` subfolders). The
tables listed below (binned scans, reduced HV-monitor logs, run lists, small caches and the
test-beam result tables: 50 files, 129 MB) sit in one folder outside the repository:

    /eos/user/m/musafdar/ETROC_plot_inputs/irrad_2026

`irrad_2026.py` reads from there by default (its `INPUTS`). Both locations are in a personal EOS
area, so reading them needs a share from its owner (Murtaza Safdari). Rebuilding some of the
tables reads more places, which the notebooks never read: the LV power-log copies (`LV_LOG_ROOT`,
in the same personal area) for `july/lv_spans_jul.csv`; the DAQ's run folders
(`RUN_METADATA_ROOT`, another personal EOS area) for `tables/run_times.csv` and the two BL/NW
tables of `frontend/`; the merged hits (`DECODED_ROOT`, the same area) for the two TDC tables
of `frontend/`; the reconstructed tracks (`ANALYSIS_ROOT`, the personal area of the tables) for the
eight track tables of `frontend/`; and the autocalibration scan histories (`BLNW_ROOT`, a group
area on the LPC EOS) for the two BL/NW tables of `frontend/`; the TWC-transfer study's coefficient bank
(`TWC_TRANSFER_ROOT`, the personal area of the tables) and its own folder (`TWC_RESULTS_ROOT`, by
default the copy in `universality/results/`) for `universality/`. Each needs a share from its owner. To work from your own copy,
copy the folders and point `ETROC_INPUTS` and `ETROC_IV_EOS_MARCH` at them (all the variables
are in the table at the end):

    cp -r /eos/user/m/musafdar/ETROC_plot_inputs/irrad_2026 /some/where/
    export ETROC_INPUTS=/some/where/irrad_2026
    cp -r /eos/user/m/musafdar/CERN_IRRAD_Mar2026/IVCurves /some/where/march_ivcurves
    export ETROC_IV_EOS_MARCH=/some/where/march_ivcurves

Each notebook's first code cell checks the inputs it reads (`inputs.check_inputs`). The tables are
checked against `irrad_2026_inputs.md5`, which lists every file with its checksum: a missing or
unreadable file stops the notebook, and a file that differs from the published one (a cache you
rebuilt, say) is reported. Every raw file the campaign reads (`RAW_INPUTS` in `irrad_2026.py`:
the March scans and logs, the July board-config yaml, and any table moved by its own variable)
must exist and be readable. All problems are listed together, missing files apart from files
without read permission. To check a copy of the tables by hand:

    cd $ETROC_INPUTS && md5sum -c /path/to/TestBeam/etroc_plots/campaigns/irrad_2026_inputs.md5

## Data quality

The IRRAD 2026 loops were recorded with the KCU105 readout saturated: 47 to 81 % of the ETROC2
packets reached disk without their header, and the decoding drops those hits. Every table here
built from readout hits (the TDC tables of `frontend/`, from the merged hits; its track tables and
the test-beam result tables of `tables/`, from the reconstructed tracks) therefore holds the hits
that survived, selected by the state of the readout buffers: its hit, event and pixel counts are
populations of that sample, not of the sensor, and whether the selection biases the TOT, the
time walk or the resolution is open. The autocalibration tables (`frontend/blnw_by_run.csv`,
`frontend/blnw_by_pixel.csv.gz`), the IV, bias-current, HV-monitor and LV tables and the run lists
are not readout hits and are unaffected. In `universality/` the fits and pair penalties of the
IRRAD runs are measured on the surviving hits in the same way; its DESY 2026 rows come from clean
readout and are unaffected. The package README, "Data quality", gives the numbers, the loops that
are exceptions and what this means per figure type.

## Curated by hand

Edit these to change what the figures show.

| file | contents |
|---|---|
| `july/display_runs_jul.csv` | the July runs drawn at the 1.5e15, 2e15 and 3.5e15 steps. One row per run: telescope, fluence, rfsel, os (threshold offset), run, preferred (1 marks the run used where a figure needs one run per telescope, fluence, RFSel and offset), combo (`anointed`, its name on disk for the reference combination, the fixed board combination; `band` = the min-max over all combinations), bias_note, note |
| `july/good_runs.csv` | a good/bad verdict with a one-line reason for every run (campaign, telescope, run, good, reason); the IV figures read its `IRRAD_Mar2026` and `IRRAD_Jul2026` rows |

## Rebuilt by this package

Run from `TestBeam/`. The rebuild is the authority for these files.

| file | contents | rebuild |
|---|---|---|
| `march/march_inrun_60s.csv` | March in-run leakage current per run and board: 60 s medians inside each run's plateau window, from the six raw IV-monitor logs on EOS | `python -m etroc_plots.iv.iv_timeseries --reduce-all --out FILE` |
| `march/preirrad_cache/` (3 files) | the pre-irradiation stability figure's cache: the slow-control log as 10 s medians, per-run and per-board statistics, the run windows. The rebuild writes the same bytes when nothing has changed, so `md5sum -c` shows whether the published cache was stale | `python -m etroc_plots.iv.preirrad_current --out FIGDIR --cache DIR --rebuild` |
| `march/fine_iv_F1_m25C_17cm_15e14_*_binned_iv_data.csv`, `march/quick_iv_F1_m25C_17cm_15e14_*_binned_iv_data.csv` (3 files) | the F1 IV scans of the 1.5e15 step, on the same voltage bins as the other March scans (the March tree holds no binned version of them): each slow-control log of that step (a `*_interpolated.csv.gz` file in `laptop_mirror_15e14/F1/` of the March tree; `BINNED_FROM_RAW` in `irrad_2026.py` names each with its window) cut to its scan's window, then each channel to its up-sweep; mean voltage and median current per bin (`FINE_BINS` or `QUICK_BINS` in `iv/legacy.py`). The command prints each table's md5, which matches its line in `irrad_2026_inputs.md5` | `python -m etroc_plots.iv.legacy --out DIR` |
| `july/lv_spans_jul.csv` | the July LV power log, one row per file: campaign, telescope, file name, first and last record (UTC, to the second) and number of records. The log records only while the LV is on, so every gap between files is an LV cycle (overview notebook, figure 6, draws those longer than 1 h; the LV-on time on the IV notebook's figures 24 and 25 is the end of the last such gap before the run). Read from the log folders `LV_LOG_DIRS` of the campaign module (`<LV_LOG_ROOT>/H1` and `<LV_LOG_ROOT>/F1`), copies of the DAQ's folders `/store/user/lpcmtdstudies/IRRad_CERN_2026Jul/{H1,F1}_telescope/power_history/lv_history/` on the LPC EOS | `python -c "from etroc_plots import lv_log; lv_log.build_lv_spans('FILE')"`, or with `dirs={("july", "h1"): H1_FOLDER, ("july", "f1"): F1_FOLDER}` for other folders |
| `frontend/` (11 files) | the front-end notebook's tables: the baseline and noise width of every board in every run, the TDC codes of one pixel per chip and the CAL code of every pixel in the `COOLDOWN_PAIRS` runs, and the TOT and TOA difference of the tracks (per triple, per pixel, per board, fixed pixels, the RFSel pairs and their time control). Columns, sources and run times in "Front-end tables" below | `python -m etroc_plots.frontend.rebuild --out DIR` |

## Made outside this package

The code that made these is not in this repository; the origin column says what each came from.

| file | contents | origin |
|---|---|---|
| `july/timeline_60s.csv.gz` | the whole July HV-monitor log, both telescopes and all four channels, in 60 s bins (UTC) | the July HV-log extraction of 2026-08-22 (its `timeline_60s.csv`, gzipped). Its source files are named in `july_timeline.json` (`source_host`, `source_dirs`) |
| `july/july_timeline.json` | that extraction in one object: clock finding, board map, radiation stops, irradiation windows, logging gaps, HV-monitor file index, every IV scan at six reference biases, every per-board run | the same extraction. Edited after the extraction: two keys and one sentence renamed to "cooldown", numbers unchanged |
| `july/july_inrun_currents.csv` | per run and board: in-run bias current (plateau window, median, first and last hour, 5-95 % range), nominal bias, logging status, a per-telescope radiation-stop proxy (the first IV scan after the step; not read, the figures take the campaign's `RAD_STOP_UTC`), the IV scans before and after | the same extraction |
| `july/july_iv_scans.csv` | every July IV scan and board: abs(I) at 150, 400, 450, 480, 530 and 550 V | the same extraction |
| `july/iv_curves.json` | the full binned IV curve of every July scan | the same extraction |
| `july/iv_scan_inventory.csv` | the July IV scans: type, label, start and end (UTC), raw and clean point counts, maximum voltage | the same extraction |
| `july/logging_gaps.csv` | gaps in the July HV-monitor log, with the bias before and after each | the same extraction |
| `july/hv_cycles_jul.csv` | per run: HV and LV cycles since the previous run, irradiation steps, DAQ restarts | the July per-run time-resolution analysis (not in this repository) |
| `july/res_vs_run_combo_check_jul_values.json` | the resolution-vs-run figure's values file: run order and per-run class, flag, fluence, threshold offset and RFSel | the same analysis, 2026-09-21 |
| `july/fine_legacy_20260716/` (2 files) | the iseg slow-control logs holding the 2026-07-16 fine IV scans, H1 and F1 | copies, with the telescope added to the name (and the H1 file's suffix changed to `_corrected`), of `202607160020_fineIV_interpolated_claudefixed.csv` (H1) and `202607160017_fineIV_interpolate.csv` (F1) in `/eos/uscms/store/user/lpcmtdstudies/IRRad_CERN_2026Jul/{H1,F1}_telescope/power_history/hv_history/`. The H1 file is a corrected interpolation stored there beside the original; what was corrected is not recorded with it |
| `march/irrad_f1_inrun_currents_2026mar.csv` | F1 in-run leakage current per run: median, MAD, first and last hour, drift, window | the H1 in-run recipe run on the F1 slow-control logs, 2026-09-02 |
| `march/conditions_March2026_H1_HPK.csv` | H1 run conditions: run start (UTC), chip, role, HV and threshold offset per board, fluence label | the DAQ run metadata |
| `vgl/vgl_points.csv` | V_gl (gain-layer depletion voltage) per chip and fluence | the campaign IV notebooks (`ExtractIVCurves.ipynb`), k-factor peak of the fine low-V scans; the file's header records the cross-check |
| `universality/univ_cells.csv` | every cell of the TWC-transfer study (one comparison of a source and a target): its level or axis, its source and target runs | the study's job lists (its toolkit is not in this repository; specification dated 2026-09-09), reduced by `python -m etroc_plots.universality.reduce`. Columns and command in "Universality tables" below |
| `universality/univ_master_cells.csv` | the study's verdict table of the level cells | the inputs of the study's round-1 report of 2026-09-04 (its `master_cells.csv`), copied byte for byte by the same command |
| `universality/univ_axis_r2.csv` | the follow-up axes A-E and I-K, every arm of every cell | the study's eight round-2 axis tables, the guard-fix rerun's rows replacing those of its 11 cells of axes E, J and K, with the self-transfer flag from the study's aggregation tables, by the same command |
| `universality/univ_axis_F_wide.csv`, `universality/univ_axis_F_long.csv` | axis F, the source correction with its linear term refitted on the target: the arms side by side, and in the shape of the other axes | the study's two axis-F tables, copied with the self-transfer flag by the same command |
| `universality/twc_bank_best.csv` | one time-walk correction per pixel and run, DESY May and August and IRRAD March and July 2026: the quadratic's coefficients, and the TOT range and histogram behind it | the study's coefficient bank (63 files, 102 MB), reduced to the best fit per pixel by the same command |

## Test-beam result tables

The time-resolution results of the test-beam chain (`TestBeam/condor_at_lxplus`), as flat tables
in `tables/`; the merged-runs notebook reads them. `../resolution/tables.py` describes their
columns, and each folder carries a `PIXEL_TABLES_README.md` from its build. The code that made
them is not in this repository yet: the build scripts are named in the origin column, and the
run merging itself (the merged step-13 outputs) was done by the run-merging study's scripts
(`TestBeam/condor_at_lxplus/merge_runs/`), also not in this repository yet.

| file | contents | origin |
|---|---|---|
| `tables/irrad/board_table_irrad.csv` | one row per run, 3-board combination, board, track variant and event floor, March and July: the board's time resolution, pixel spread and pixel count, and whether the combination is the reference one for that board (the column `anointed`, its name on disk) | `build_pixel_tables.py --family irrad`, 2026-09-15/16, from each run's step-13 `resolution_table_<combo>.csv` with the functions of `TestBeam/condor_at_lxplus/utils/quote_resolution.py` |
| `tables/irrad/pixel_table_irrad.csv.gz` | the same per pixel: resolution, fit error, events behind the pixel | the same build |
| `tables/merged/board_table_merged.csv`, `tables/merged/pixel_table_merged.csv` | the same two tables for merged runs (`run` is the merge's name, such as `h1_grp_run3_4`), with the share of tracks dropped as degenerate per combination | the same build, on the merged step-13 outputs of the run-merging study |
| `tables/merged/per_pixel_cost_summary.csv` | per merge and board: the cost of merging, per pixel (median, 16th and 84th percentile) and on the board value | `merge_runs/merge_cost/build_per_pixel_cost.py` of the run-merging study, 2026-09-15 |
| `tables/runs_summary.csv` | one row per run: status, and per board the role, chip, bias, bookkeeping fluence, threshold offset and RFSel | `build_irrad_summary.py`: the board settings from the run-config yamls (`TestBeam/board_configs_yaml/CERN_Irrad_2026Mar.yaml` and `CERN_Irrad_2026Jul.yaml`), the rest from each run's quote. The copy of 2026-10-02 has RFSel 3 for H1 runs 64, 67 and 70 after the yaml correction (the yaml block of these runs carried no RFSel key and inherited the chip default 2; the run names, the autocalibration scan's gain_mode and `rfsel_src` = scan in `frontend/blnw_by_run.csv` say 3): those 12 cells changed from 2.0 to 3.0 and nothing else; the copy of 2026-09-16 is kept as `tables/superseded/runs_summary_2026-09-16.csv`. The frontend and maps notebooks also read this table and draw no RFSel of those runs; the manifests of older branches report the checksum difference |
| `tables/irrad/PIXEL_TABLES_README.md`, `tables/merged/PIXEL_TABLES_README.md` | the notes each build wrote beside its tables | the same builds |

One more table in `tables/` is not a result of the chain: `tables/run_times.csv`, read by the
overview notebook's timeline (figure 6). One row per DAQ run folder of both campaigns, 97 rows:
campaign, telescope, run number, run name, start (UTC, to the second) and the configured run
length in minutes, taken from each run's `run_metadata.yaml` (`run_info.timestamp` and
`acquisition_settings.max_run_time_minutes`) in the DAQ's run folders on EOS
(`RUN_METADATA_DIRS` in the campaign module). Made 2026-09-25 by `build_run_times()` of
`../overview/timeline.py`; its docstring gives the command.

## Front-end tables

The front-end notebook (`../notebooks/frontend.ipynb`) reads the tables in `frontend/`. They are
made by

    python -m etroc_plots.frontend.rebuild --out DIR [--part blnw|tdc|tracks] [--runs ...] [--rank rows|bytes]

run from `TestBeam/` (`../frontend/rebuild.py`; copy the files into `frontend/` and update the
checksum lines to use them). `--runs` takes `pairs` (the runs of `COOLDOWN_PAIRS`), `frontend` (the
runs of `FRONTEND_RUNS` and the July display runs at RFSel 2, threshold offset 20) or
`campaign:telescope:run` items such as `july:h1:4`.

* The BL/NW part reads every DAQ run folder (`RUN_METADATA_DIRS`: each run's `run_metadata.yaml`,
  the run start and per chip the threshold offset and the threshold DAC of every pixel), the
  run-config yamls (`RUN_LIST_YAML`: the bias of each board) and the autocalibration scan
  histories of the telescope PCs (`BLNW_SQLITES`, under `BLNW_ROOT`, default
  `/eos/uscms/store/group/lpcmtdstudies`: `BaselineHistory.sqlite` per telescope, July in two
  files). sqlite cannot open the histories in place on EOS: copy them, keeping the folders below
  `BLNW_ROOT`, and point `ETROC_BLNW_ROOT` at the copy. It joins each run to the scan the DAQ set
  its thresholds from (`../frontend/blnw.py` gives the rule) and writes `blnw_by_run.csv` and
  `blnw_by_pixel.csv.gz`.
* The TDC part reads every loop file of the merged hits of each run (`DECODED_DIRS`, under
  `DECODED_ROOT`, default `/eos/user/j/jongho`): about 1 s per file on one lxplus core, 5.5 min
  for the four `COOLDOWN_PAIRS` runs (328 files, 812 million hits), and prints each run's hit counts
  before and after each cut.
* The tracks part reads the track files of each run (`ANALYSIS_DIRS`, under `ANALYSIS_ROOT`,
  default `/eos/user/m/musafdar`: `CERN_IRRAD_Mar2026/analysis/tracks_<tel>_run<N>/<combo>/time/`
  and the same under `CERN_IRRAD_Jul2026`, one parquet file per triple of pixels) and the
  run-config yamls (`RUN_LIST_YAML`: each board's role, which names the columns and folders). By
  default it reads the runs of `FRONTEND_RUNS` and the July display runs at RFSel 2, threshold
  offset 20 (30 runs; in July `FRONTEND_RUNS` holds H1 4, 19 and 47 and F1 9, 18, 32, 47, 52
  and 86, F1 18 also a display run): the footer of every file, then the columns of every file
  that is some pixel's triple. It took 87 min on one lxplus core (2026-10-09), about
  three minutes per run. The TOA difference uses the pipeline's time-walk solver
  (`condor_at_lxplus/core/twc_solver.py`, `apply_timewalk_correction_arrays`: one polynomial in
  TOT per board, solved jointly over the triple's three boards), so the rebuild runs from
  `TestBeam/`. `--rank bytes` picks each pixel's triple by file size, the rule of the first
  presented figures (kept for comparisons); the default, `rows`, by its events.

A pixel's triple: the track files of a run hold one triple of pixels each, on boards 0-1-2 or
1-2-3 (the folder name gives the boards' roles and indices, `extra0-dut1-ref2`). Each pixel of a
board takes one triple, the one with the most events among the triples of its board's
combination (boards 0-1-2 for boards 0-2, boards 1-2-3 for board 3; ties to the larger file, then
the first name), and counts when that triple holds at least 300 events (`tot_shapes.FLOOR`).
Nothing is pooled across the triples of one pixel.

The two BL/NW tables were made by the rebuild (`--part blnw`, every run) on 2026-10-18 on the
LPC, from xrootd copies of the six scan histories and copies of the 97 runs' `run_metadata.yaml`
(`ETROC_BLNW_ROOT` and `ETROC_RUN_METADATA_ROOT` pointing at the copies), in 16 s on one core.
`blnw_by_run.csv` replaced the 2026-09-02 build of the same join (made outside this repository,
from the same histories and run folders, its columns renamed to the ones below; kept as
`frontend/superseded/blnw_by_run_2026-09-02.csv`, out of the checksum list): on its 24 columns
the two agree in every cell of all 384 rows, and the rebuild adds the six median and percentile
columns. The two TDC tables were made by the rebuild (`--part tdc`,
default runs) on 2026-10-02, the eight track tables by the rebuild (`--part tracks`, default runs
and rank) on 2026-10-09.

| table | rows | made by | md5 (first 8; full in `irrad_2026_inputs.md5`) |
|---|---|---|---|
| `blnw_by_run.csv` | 384 | `--part blnw`: `blnw.build_blnw_table`, `board_stats` | `e669866c` |
| `blnw_by_pixel.csv.gz` | 98304 | `--part blnw`: `blnw.build_blnw_table`, `pixel_rows` | `80a3dc8c` |
| `tdc_pixel_hists.csv` | 19162 | `--part tdc`: `tdc_codes.gather`, `busiest_pixels`, `hist_rows` | `82d783ec` |
| `tdc_cal_modes.csv` | 4096 | `--part tdc`: `tdc_codes.gather`, `cal_rows` | `15f8c86c` |
| `triples.csv` | 72653 | `--part tracks`: `tot_shapes.scan_run` (the footer of every track file) | `677a2777` |
| `tot_pixel_medians.csv` | 30623 | `--part tracks`: `tot_shapes.run_tables` | `a2c585da` |
| `tot_pooled_summary.csv` | 120 | `--part tracks`: `tot_shapes.run_tables` | `4bdb1451` |
| `tot_pooled_hist.csv` | 12964 | `--part tracks`: `tot_shapes.run_tables` | `417980ec` |
| `dtoa_triples.csv` | 28409 | `--part tracks`: `tot_shapes.run_tables` (time-walk solver of the pipeline) | `f92e4a67` |
| `dtoa_pooled_hist.csv` | 11836 | `--part tracks`: `tot_shapes.run_tables` | `c1a4a605` |
| `pixel_hists.csv` | 8939 | `--part tracks`: `tot_shapes.choose`, `pixel_hist_rows` (`tot_codes`) | `2557f250` |
| `rfsel_pair_hist.csv` | 812 | `--part tracks`: `tot_shapes.rfsel_board`, `rfsel_rows` | `ab3ce7ab` |

`frontend/blnw_by_run.csv`, 384 rows (97 runs). One row = one board in one run.

| column | contents |
|---|---|
| `campaign`, `tel`, `run` | `march` or `july`, `h1` or `f1`, the run number |
| `board_idx`, `chip` | the board (0-3, the DAQ's elink / 4) and its chip as `TELESCOPE_CHIPS` names it (`IH7`) |
| `start_utc` | the run start, UTC |
| `rfsel`, `rfsel_src` | the RFSel of the run and where it was read: `scan` (the RFSel the scan history records), `run_name` (`RFSel<N>` in the run name), `run_yaml` (the run-config yaml) or `none` (every March run: the March records carry none; they ran at RFSel 2) |
| `offset` | the threshold offset of the chip in the run (DAC) |
| `hv` | the sensor bias of the board in the run-config yaml (V) |
| `scan_utc`, `scan_gap_min` | the start of the scan joined to the run (UTC; the March PCs wrote CET, converted) and the minutes from it to the run start |
| `match`, `n_mismatch` | how well the scan's baseline map equals the run's threshold DAC minus its offset: `exact` (no pixel differs), `near_exact` (1-3 pixels), `nearest` (more; the closest earlier scan is used) or `none` (no scan before the run); the number of pixels that differ, railed pixels left out |
| `bl_mean_allpix`, `bl_std_allpix` | mean and standard deviation of the scan's baseline over all 256 pixels (DAC), a railed or dead pixel as stored |
| `nw_mean_allpix`, `nw_std_allpix` | the same for the noise width (DAC), a pixel without a noise width counted as 0 |
| `bl_mean`, `bl_std`, `n_pixels_bl` | the same over the pixels with 50 < baseline < 1020 DAC and not railed in the run, and their number |
| `nw_mean`, `nw_std`, `n_pixels_nw` | the same over the pixels with a noise width above 0, and their number |
| `bl_median`, `bl_p16`, `bl_p84` | median and 16th and 84th percentiles of the baseline over the pixels of `bl_mean` (DAC; numpy's linear interpolation between pixels): the pixel-to-pixel spread, not an uncertainty |
| `nw_median`, `nw_p16`, `nw_p84` | the same for the noise width over the pixels of `nw_mean` (the noise width is an integer DAC, so these are coarse) |

`frontend/blnw_by_pixel.csv.gz`, 98304 rows (gzip-compressed CSV; read by
`blnw.read_blnw_pixels`, by no notebook here). One row = one pixel of one board in one run: the 256
pixels of every board of `blnw_by_run.csv` that has a scan, from the scan that row names. Pooling
the pixels of several runs (a merged run) takes their rows together; runs that reused one scan
(the same `scan_utc`, such as H1 July runs 18 and 19) hold the same pixels.

| column | contents |
|---|---|
| `campaign`, `tel`, `run`, `board_idx`, `chip` | as in `blnw_by_run.csv` |
| `row`, `col` | the pixel, chip frame: pixel (0, 0) at the bottom right of a map |
| `bl`, `nw` | the scan's baseline and noise width of the pixel as stored (DAC): a failed autocalibration stores a baseline of 0, the rail is 1020 and above; empty where the scan has no row for the pixel |
| `live` | 1 when the pixel counts in `bl_mean` of its board (50 < baseline < 1020 DAC and not railed in the run's threshold DAC, `blnw.live_pixels`), else 0; the noise-width statistics take the pixels with `nw` above 0 instead |
| `scan_utc` | the start of the scan (UTC), as in `blnw_by_run.csv` |

`frontend/tdc_pixel_hists.csv`, 19162 rows. One row = one non-empty bin of one histogram of one
pixel in one run. The pixel of each chip is its most populated in the July run of `COOLDOWN_PAIRS`
(hits after the cuts), the same pixel in the March run; the hits are those with `ea == 0` and a CAL
code within 3 of the pixel's CAL mode in that loop file.

| column | contents |
|---|---|
| `campaign`, `tel`, `run`, `board_idx`, `chip` | as above |
| `row`, `col` | the pixel (chip frame) |
| `quantity` | `cal_code`, `toa_code`, `tot_code` (the 10-bit codes), `toa_ns` (12.5 - toa x 3.125 / m) or `tot_ns` ((2 tot - floor(tot / 32)) x 3.125 / m), m the pixel's CAL mode in that loop file |
| `bin_lo` | the code, or the lower edge of the 0.1 ns bin (bins not clipped to any range) |
| `count` | hits in the bin |

`frontend/tdc_cal_modes.csv`, 4096 rows. One row = one pixel of one board in one run of
`COOLDOWN_PAIRS`.

| column | contents |
|---|---|
| `campaign`, `tel`, `run`, `board_idx`, `chip`, `row`, `col` | as above |
| `cal_mode` | the median over the loop files of the pixel's most frequent CAL code in each file (hits with `ea == 0`) |
| `n_hits_all`, `n_hits_ea0`, `n_hits` | the pixel's hits: all, with `ea == 0`, and also inside the CAL window |
| `n_files` | the loop files of the run, all read |

`frontend/triples.csv`, 72653 rows. One row = one track file (one triple of pixels) of one
run.

| column | contents |
|---|---|
| `campaign`, `tel`, `run` | as above |
| `boards`, `combo` | the boards of the triple (`0-1-2` or `1-2-3`) and its folder (roles and indices, `extra0-dut1-ref2`) |
| `chip_<b>`, `row_<b>`, `col_<b>` | for each board b = 0-3, its chip and pixel in the triple; empty for the board outside it |
| `n_rows`, `n_bytes` | the events of the triple (rows of the file) and the file size (bytes) |
| `file` | the file name |

`frontend/tot_pixel_medians.csv`, 30623 rows. One row = one pixel of one board in one run:
every pixel with a triple in its board's combination.

| column | contents |
|---|---|
| `campaign`, `tel`, `run`, `board_idx`, `chip`, `row`, `col` | as above |
| `n` | the events of the pixel's triple |
| `tot_median_ns` | the median TOT of the pixel's events in that triple (ns; events without a TOT left out) |
| `passes_floor` | `n` >= 300: the pixel counts |

`frontend/tot_pooled_summary.csv`, 120 rows. One row = one board in one run.

| column | contents |
|---|---|
| `campaign`, `tel`, `run`, `board_idx`, `chip` | as above |
| `n_pixels`, `n_events` | the counted pixels and all their events |
| `pixmed_median_ns`, `pixmed_p16_ns`, `pixmed_p84_ns` | the median and the 16th and 84th percentiles of the counted pixels' `tot_median_ns` |
| `median_ns`, `p16_ns`, `p84_ns` | the same over every event of the counted pixels together |
| `mode_ns_0.05`, `mode_ns_0.10`, `mode_ns_0.20` | the centre of the fullest bin of those events in 0.05, 0.1 and 0.2 ns bins from 0 |
| `n_outside_hist` | those events outside 0-20 ns (not in `tot_pooled_hist.csv`) |

`frontend/tot_pooled_hist.csv`, 12964 rows. One row = one non-empty 0.05 ns bin of the TOT of
every event of one board's counted pixels in one run.

| column | contents |
|---|---|
| `campaign`, `tel`, `run`, `board_idx`, `chip` | as above |
| `bin_lo_ns`, `count` | the lower edge of the bin (0-20 ns) and its events |

`frontend/dtoa_triples.csv`, 28409 rows. One row = one board pair of one triple in one run.
Pairs 0-1 and 0-2 come from board 0's triple of every counted board-0 pixel, 1-2 from board 1's,
2-3 from board 2's triple among the triples of boards 1-2-3 (the one with the most events); a
triple is used when it holds at least 300 events with TOT, TOA and CAL on all three boards. The
TOA difference is taken after the time-walk correction of the triple (above).

| column | contents |
|---|---|
| `campaign`, `tel`, `run` | as above |
| `pair`, `board_a`, `chip_a`, `board_b`, `chip_b` | the pair (`0-1`, `0-2`, `1-2` or `2-3`), its first and second board |
| `combo`, `row_<b>`, `col_<b>` | the triple's folder and pixels, empty for the board outside it |
| `n_events` | the events with every value present |
| `dt_median_ps` | the median of TOA(a) - TOA(b) after the correction (ps, 3 decimals) |
| `n_outside` | events more than 2000 ps from that median |
| `cal_a`, `cal_b` | the median CAL code of each board of the pair in the triple (one TOA code = 3.125 ns / CAL) |

`frontend/dtoa_pooled_hist.csv`, 11836 rows. One row = one non-empty 25 ps bin of one pair in
one run: every triple of `dtoa_triples.csv`, each centred on its own median.

| column | contents |
|---|---|
| `campaign`, `tel`, `run`, `pair`, `board_a`, `chip_a`, `board_b`, `chip_b` | as above |
| `bin_lo_ps`, `count` | the lower edge of the bin (-2000 to 2000 ps) and its events |

`frontend/pixel_hists.csv`, 8939 rows. One row = one non-empty bin of one histogram of one
fixed pixel or pixel pair in one run. The fixed pixels are chosen per telescope over the runs of
`tot_shapes.shape_runs` (one per fluence step: March the runs of `MARCH_DISPLAY_RUNS`, July the
preferred display run at RFSel 2, offset 20): per board, among the pixels counted in every one of
those runs, the one whose smallest count over the runs is largest (ties to the smallest row,
column). Pairs 0-1, 0-2 and 1-2 join those pixels, through the triple with the most events holding
both in each run; pair 2-3 keeps board 3's pixel and takes the board-2 pixel tracked with it in
every run whose best such triple has the largest smallest count. The board-1 pixel of the July
figures is chosen the same way over the July runs alone.

The TOT of a fixed pixel is stored twice: in ns (`tot_ns`, 0.05 ns bins) and per TOT code
(`tot_code`). The codes are the exact inverse of the pipeline's conversion (step 10,
`core/apply_tdc_cuts.convert_to_time`): per loop file of the run (the `file` column of the track
file) the bin is 3.125 ns / the mean CAL code of the board over that file's saved events, and
every TOT divided by it is m = 2 code - floor(code / 32) for one code of the 9-bit TOT
(`tot_shapes.tot_codes`, which stops on any value off that ladder). The events are the same as
those of the `tot_ns` rows, after the cuts of step 10; the `tracks/` folder next to `time/`
holds the raw codes before those cuts and is not read.

| column | contents |
|---|---|
| `campaign`, `tel`, `run` | as above |
| `pick` | `pixel` (each board's fixed pixel: its TOT), `pixel_pair` (each pair's fixed pixels: their TOA difference), `july_pixel` (the board-1 pixel of the July runs: its TOT and its TOA difference with board 2) |
| `quantity` | `tot_ns`, `tot_code` or `dtoa_ps` |
| `board_a`, `chip_a`, `row_a`, `col_a` | the board and pixel histogrammed (for `dtoa_ps` the first of the pair) |
| `board_b`, `chip_b`, `row_b`, `col_b` | the second board and its pixel in the triple (`dtoa_ps` only) |
| `combo`, `file` | the triple's folder and file |
| `n_events`, `median`, `n_outside` | the events, their median (TOT in ns; for `tot_code` the lower median code; for `dtoa_ps` the median the values were centred on, ps) and the events outside the histogram (0 for `tot_code`) |
| `bin_lo`, `count` | the lower edge of a 0.05 ns bin over 0-20 ns, the TOT code (`tot_code`), or the lower edge of a 25 ps bin over +-2000 ps (centred), and its events |
| `cal_mode` | `tot_code` rows only: the board's most frequent CAL code over the triple's events (ties to the smallest); a TOT code is 2 x 3.125 ns / `cal_mode` wide, except the code at each multiple of 32 (32, 64, ...), half of that; empty on the other rows |

`frontend/rfsel_pair_hist.csv`, 812 rows. One row = one non-empty 0.05 ns TOT bin of one board
in one run of `RFSEL_TOT_PAIRS` (July F1: runs 47 at RFSel 1 and 18 at RFSel 2, threshold offset
20; runs 52 at RFSel 2 and 86 at RFSel 3, offset 10) or of the time-control group of
`TOT_TIME_CONTROL` (July F1: runs 32 and 52 at RFSel 2 and 86 at RFSel 3, offset 10). The board
is the one whose smallest total of counted events over the four runs of the pairs is largest
(board 1, LF12), for the group too; per pair or group, the pixels of that board counted in every
one of its runs, every event of each.

| column | contents |
|---|---|
| `campaign`, `tel`, `run`, `board_idx`, `chip` | as above |
| `rfsel`, `offset`, `partner_run` | the run's RFSel and threshold offset (`tables/runs_summary.csv`) and the other run of its pair (empty in the time-control group) |
| `n_pixels`, `n_events`, `median_ns` | the pixels counted in every run of the pair or group, their events in this run and the median TOT (ns) |
| `bin_lo_ns`, `count` | the lower edge of the bin (0-20 ns) and its events |
| `group` | the runs of the pair or group, joined by `+` (`47+18`, `52+86`, `32+52+86`) |

## Universality tables

The universality notebook (`../notebooks/universality.ipynb`) reads the six tables in
`universality/`. They hold the outputs of the TWC-transfer study, which asked whether a time-walk
correction (TWC) fitted on one source (a run, a board, a pixel) can be applied to a target without
losing time resolution. The correction is a quadratic per pixel, `delta(TOT) = c2 TOT^2 + c1 TOT +
c0` (TOT and delta in ps). Each comparison is a cell; within a cell each way of correcting the
target is an arm, compared with the baseline `self_cf` (the target's own correction, fitted on the
other half of its events). The study's code (its toolkit) and its specification (dated 2026-09-09)
are not in this repository. The six tables are made by

    python -m etroc_plots.universality.reduce --out DIR [--results-root S] [--twc-root R] [--part cells|master|axes|bank ...]

run from `TestBeam/` on lxplus (`../universality/reduce.py`; copy the files into `universality/`
and update the checksum lines to use them). R is `TWC_TRANSFER_ROOT` (default
`/eos/user/m/musafdar/DESY_May2026/analysis/twc_transfer`): the coefficient bank,
`bank/coeffs_<campaign>_<run>.parquet`, the round-1 report's inputs,
`figs/transfer_2026-09-04/inputs/master_cells.csv`, and the aggregation tables the axis tables
were built from, `tables/agg_<agg_tag>/agg_by_cell.csv`. S is the study's own folder
(`TWC_RESULTS_ROOT`, by default the copy in `universality/results/`, below): `condor/` (the seven
job lists),
the axis tables `UNIV_AXIS_SOURCES` names (`results_r2/axis_{A,B,C,D,E,I,J,K}.csv` and
`results_r3/guardfix/axis_{E,J,K}_gf.csv`) and `results_r3/` (`axis_F.csv`, `axis_F_long.csv`,
`summary_F.md`). On one lxplus core the cells, master and axes parts take 3 s together and the
bank part 12 s. Published 2026-10-02 (cells, master, bank) and 2026-10-18 (the three axis tables,
with the guard-fix rows of axes E, J and K and the `self_transfer` flag; their 2026-10-02 versions
are kept as `universality/superseded/<file>_2026-10-02.csv`):

| file | rows | columns | md5 |
|---|---|---|---|
| `univ_cells.csv` | 131 | 8 | `2fcde80b93bff79a4d9eb54292546318` |
| `univ_master_cells.csv` | 43 | 105 | `62cfc93f96be1dcbab6cad770da772fa` |
| `univ_axis_r2.csv` | 6011 | 25 | `15c688966ce406593454391e9d6120ff` |
| `univ_axis_F_wide.csv` | 167 | 51 | `911efb7d856178f24e60a87b5203a755` |
| `univ_axis_F_long.csv` | 2863 | 25 | `51302099b643b24014141af7d860a059` |
| `twc_bank_best.csv` | 18414 | 20 | `001872b9be344ec9d9aa252bf1b9a011` |

The files the reduction reads are copied unchanged, at their relative paths, to
`universality/results/` (108 files, 114 MB, copied 2026-10-18; checksums in
`universality/results/files.md5`): the study folder's `condor/`, `results_r2/` and `results_r3/`
(with `guardfix/`), and from `TWC_TRANSFER_ROOT` the bank files, the round-1 table of the level
cells and the aggregation tables. `TWC_RESULTS_ROOT` points there by default, and `--twc-root`
can point there too: run on that copy for both roots, the reduction remakes the six tables above
with the checksums listed.

The study's words, used in every table:

* Campaign `desy` (DESY May 2026), `aug` (DESY August 2026), `mar` and `jul` (CERN IRRAD March
  and July 2026); run `run26` (DESY) or `h1_run19` (IRRAD, telescope first). The DESY runs'
  settings are in `desy_2026.py`.
* A target triple is one pixel on each of the three boards of a combination, with the tracks
  through all three. Its three pair widths are measured, its board resolutions solved from them;
  a row's difference is the arm's width minus the baseline's on the same triple and events.
* The arms: `notwc` (no correction), `notwc_1b` (no correction on the transferred board alone),
  `self_in` (the target's own correction fitted on the same events: in-sample, optimistic),
  `self_cf` (the baseline), `xfer` (the source's correction), `placebo_self` and `placebo` (the
  controls: corrections fitted with the TOT values shuffled, on the target and on the source),
  and the leave-pair-out arms
  `self_cf_lpo`, `xfer_lpo` and `xfer_lin_lpo`, whose fits leave out the other board of the pair
  scored (the guard against a correction that has learned that pair). The variants of `xfer`:
  `xfer_clip` (TOT clipped to the source's 0.5-99.5 % range), `xfer_shift` (TOT shifted by the
  difference of the medians), `xfer_rescale` (TOT shifted and scaled to the source's median and
  spread), `xfer_calfix` (TOT scaled by the ratio of the mean CAL codes), and `xfer_lin` (axis F:
  the source's `c2` kept, the linear term refitted on the target, fold-matched).
* The family: `interp` when the target's TOT lies within the source's (coverage `C`, the target's
  share inside the source's 0.5-99.5 % range, at least 0.98, and the lever, the rms change of the
  correction from clipping over all target events, at most 5 ps), else `extrap`. Transfer is
  claimed on `interp` only.
* The decision per row: TOST equivalence at alpha 0.05 against the margin max(1.0 ps, 2 % of the
  baseline width), PASS when the 90 % interval of the difference lies inside +-margin. A transfer
  row whose leave-pair-out guard is absent or uninformative reads `NA-unguarded`. No claim is made
  below 0.45 ps.
* Retention R = 1 - difference / (`notwc` width - `self_cf` width): the share of the correction's
  improvement the transfer keeps.

`universality/univ_cells.csv`, 131 rows. One row = one cell, from the job lists that ran it, read
in the order `cells.tsv`, `cells_stage1.tsv`, `cells_stage2.tsv`, `cells_stage2_held.tsv`,
`cells_stage3.tsv`, `cells_rerun.tsv`, `cells_guardfix_rerun.tsv` (a later list that names a cell
again redefines it; the two reruns name each of their cells with the runs it already had). Every
cell the other tables name is a row here. Which run of a rerun cell its axis rows come from is set
by the axis files, below, not by this order.

| column | contents |
|---|---|
| `cell` | the cell's name, such as `b1_30to0` (level b, from 30 deg to 0 deg) or `F4_lin_b1_30to0` (axis F) |
| `level_or_axis` | the level (`a` stability floor, `b` angle, `bp` angle on the offset holder, `c` pixel to board, `d` board to board, `e` pooled source, `f` the analytic comparison of the curves of one bank file, `g` fluence, `h` March to July) or the follow-up axis (`A`-`K`) |
| `src_campaign`, `src_run`, `tgt_campaign`, `tgt_run` | the source and the target, from each job's command; `src_run` is empty for a pooled source (level e), and a level-f cell has its bank file's run on both sides |
| `base_cell` | for an axis-F cell, the level cell it re-scores (from the cell table of `summary_F.md`); else empty |
| `dbias_v` | for a level-h cell, the July target's bias minus the March source's, each the median over the run's boards in `tables/runs_summary.csv` (V); else empty |

`universality/univ_master_cells.csv`, 43 rows. One row = one level cell (levels a to e, g and h)
at its campaign's primary error factor, sorted by level and cell. The binding row of a cell is
its primary row (`xfer`, pair, `interp`) furthest from equivalence (largest abs(mu) / margin).
Widths and differences in ps.

| column | contents |
|---|---|
| `cell`, `level` | the cell and its level |
| `ftag`, `group`, `f` | the aggregation (such as `agg_desy_f0p8153`), its campaign, and the factor f that scales the half-split error (measured per campaign: 0.8153 DESY May, 0.8979 August, 0.9524 March, 0.9259 July) |
| `campaign`, `run` | the target |
| `thin_target` | 1 when the target is one of the thin runs listed before the study ran (its PASS is supporting only) |
| `n_pair_interp`, `n_pair_extrap`, `npix_interp`, `npix_extrap` | the primary pair rows in each family and the differences behind them |
| `bind_name`, `mu`, `se`, `tau`, `ci_lo`, `ci_hi`, `margin`, `tost`, `n_pix`, `sigma_base`, `R_median`, `frac_over`, `chi2_red`, `p95` | the binding row: its pair, the pooled difference (random effects) and its error, the spread between triples, the 90 % interval, the margin, the verdict, the differences pooled, the median baseline width, the median retention, the share of differences over the margin, the reduced chi2 and the 95th percentile of the differences |
| `mu_min`, `mu_max`, `mu_ivw`, `se_ivw` | over the primary rows: the smallest and largest difference, their inverse-variance mean and its error |
| `n_row_fail`, `n_row_unguarded`, `n_row_pass`, `n_row_underpowered` | the primary rows that FAIL, are unguarded, PASS, and FAIL without showing non-equivalence (p above 0.5) |
| `p_nonequiv_min`, `p_nonequiv_max`, `holm_missing` | the range of the non-equivalence p values over the primary rows; 1 when none carries a Holm correction |
| `verdict_pair_interp`, `verdict_pair_extrap`, `verdict_board_interp`, `verdict_board_extrap` | the cell's verdict over the `xfer` rows of each kind and family: PASS only when every row passes, else FAIL, NA-UNGUARDED, FAIL+UNGUARDED (a guarded FAIL beside an unguarded row), MIXED or NO-ROWS |
| `mu_extrap`, `se_extrap`, `npix_e` | the binding `extrap` pair row |
| `placebo_present`, `placebo_R_absmax`, `placebo_mu_med`, `placebo_self_R_absmax`, `notwc_mu_med`, `placebo_ok` | the controls: whether a placebo row exists, its largest abs(R) and median difference, the largest abs(R) of `placebo_self`, the median `notwc` difference, and 1 when the placebo's abs(R) is below 0.05 |
| `has_xfer_lpo`, `guard_verdict`, `cheat_absmax`, `self_lpo_absmax`, `margin_min`, `cheat_over_margin`, `cheat_missing`, `lpo_ok` | the leave-pair-out guard over the primary rows (the `extrap` rows when there are none): whether `xfer_lpo` exists, its verdicts (`ok`, `absent`, or `uninformative` when the baseline's own leave-pair-out cost exceeds the margin), the largest abs of the guard statistic (`xfer` - `xfer_lpo`) - (`self_cf` - `self_cf_lpo`) and of the baseline's cost, the smallest margin, and three flags: the statistic over that margin, the statistic missing, the guard clear |
| `gate_fail`, `k_cal` | the median share of width fits that failed their goodness-of-fit gate, and the median calibration k of the half-split error |
| `lever_X` | the median lever over all target events (the toolkit's `lever_X_all`) |
| `n_src`, `mu_src_spread` | the source triples, and the spread of the difference between them |
| `xfer_lpo_verdict`, `xfer_lpo_mu` | the verdict and median difference of `xfer_lpo` (pair, `interp`) |
| `final`, `below_noclaim` | the cell's final verdict (the pair-`interp` verdict; UNSTABLE when it changes with the error factor; SUPPORTING for a thin target's PASS) and 1 when abs(`mu`) is below 0.45 ps |
| `all_bind_name`, `all_mu`, `all_se`, `all_margin`, `all_tost`, `all_npix`, `all_ivw_mu`, `all_ivw_se`, `all_verdict`, `all_sigma_base` | the same over all triples, both families pooled (descriptive): the binding pair row and its numbers, the inverse-variance mean and its error, the verdict over the rows, the median baseline width |
| `n_interp`, `n_extrap`, `n_censored`, `n_below_nmin` | the differences in each family, the triples censored as degenerate, and those under the 600-event floor |
| `allb_name`, `allb_mu`, `allb_se`, `allb_tost`, `allb_margin` | the binding board row over all triples |
| `n_tgt_triples`, `src_run`, `interp_frac`, `f_verdicts` | the target triples, the source run, the share of differences in `interp`, and the pair-`interp` verdict at each error factor of the campaign (`;`-joined) |
| `n_pair_rows`, `n_interp_pair_rows`, `n_triples`, `n_tri_any_interp`, `n_tri_all_interp` | the `xfer` pair rows (all, `interp`), the target triples, and those with at least one or all of their pairs in `interp` |
| `n_same_triple`, `n_same_triple_interp`, `all_mu_excl`, `all_se_excl`, `int_mu_excl`, `int_se_excl` | the rows whose source triple is the target triple (all, `interp`), and the inverse-variance mean difference and its error without them (all triples, `interp`) |
| `all_mu_spec`, `all_se_spec`, `int_mu_spec`, `int_se_spec` | the same means with each row's error as the specification first wrote it (half the difference of the two halves' differences, times k), all triples and `interp` |

`universality/univ_axis_r2.csv`, 6011 rows (axis A 800, B 620, C 660, D 1089, E 1395, I 393, J 370,
K 684), and `universality/univ_axis_F_long.csv`, 2863 rows, have the same 25 columns: the study's
24 and `self_transfer`. One row = one cell, arm, kind, pair or board, family, error factor and
`self_transfer`, every arm included. Per axis the reduction reads the files of
`UNIV_AXIS_SOURCES` (`irrad_2026.py`) in order, a later file replacing every row of the cells it
holds at the place of the cell's first row; the guard-fix files' two extra columns (`lpo_shape`,
`lpo_pairs`) are dropped. Each line is otherwise the study's line unchanged. The code behind each
axis (the toolkit's `tt.py`, sha256 first 16 hex digits, as the job lists' DAG headers record it):

| axis | cells | file | `tt.py` |
|---|---|---|---|
| A, B, C, K | condor stage 2 | `results_r2/axis_X.csv` | `057934883be9ad88` |
| D, E, I, J | condor stage 1 | `results_r2/axis_X.csv` | `057934883be9ad88` |
| E, J, K | E2, E4, E5, E10, J1-J4, K1-K3: the guard-fix rerun of 2026-09-10 (`--lpo-fallback own`), replacing their stage-1 and stage-2 rows | `results_r3/guardfix/axis_X_gf.csv` | `e83f71c18ed40c56` |
| F | condor stage 3 | `results_r3/axis_F.csv`, `results_r3/axis_F_long.csv` | `698f72f97efb5a01` |

The seeded rerun of 2026-09-08 (`cells_rerun.tsv`, `tt.py` `698f72f97efb5a01`) wrote no axis
table: of its cells, A4, A6, B6, C7, E12, E13 and I2 keep their round-2 rows and J1-J4 and K3
take the guard-fix rows.

| column | contents |
|---|---|
| `axis`, `cell`, `arm`, `kind` | the axis, the cell, the arm, `pair` or `board` |
| `board_or_role` | the pair (its two roles, hyphen-joined) or the board's role |
| `family` | `interp`, `extrap`, or `na` for the arms that use no source |
| `n_pix`, `n_tgt`, `n_src` | the differences pooled, the target triples and the source triples |
| `mu_ps`, `se_ps`, `tau`, `margin_ps`, `tost`, `guard_verdict` | the pooled difference from `self_cf` and its error, the spread between triples, the margin, the verdict and the guard's verdict |
| `spec_primary_row` | 1 for the rows the specification makes primary (pair rows, except `xfer_lpo` on the pair without the transferred board, a guard control), else 0 |
| `R_median` | the median retention |
| `placebo_minus_notwc1b_ps` | `placebo` minus `notwc_1b`: on a cell that transfers one board, the control that says whether the cell is readable |
| `cov_C`, `lever_X`, `lever_X_all` | the mean coverage, and the lever over the target events outside the source's range and over all of them |
| `f_inflate`, `f_variant_is_primary`, `agg_tag` | the error factor; True at the campaign's measured factor (the DESY May cells are also given at 1.0, the August cells at 0.8153 and 1.0); the aggregation |
| `self_transfer` | 1 when the row's source triple is its target triple (the same run, combination and pixels), else 0. The study aggregates these rows as a group of their own and leaves them out of its per-cell summaries; its axis tables carry them unmarked. The reduction takes the flag from the row's aggregation table, `tables/agg_<agg_tag>/agg_by_cell.csv` (for A1-A7, `agg_A_r2_desy_f0p8153` and `agg_A_r2_desy_f1p0`: `UNIV_AGG_DIRS`), joined on cell, arm, kind, pair or board, family, `agg_tag` and `n_pix`, and stops unless each row joins exactly one row with the same `mu` and `se`. 110 rows of `univ_axis_r2.csv` (axis A 104, I 6) and 276 of `univ_axis_F_long.csv` carry 1; the notebook draws none of them |

`universality/univ_axis_F_wide.csv`, 167 rows. One row = one axis-F cell, kind, pair or board and
family at the primary error factor, with the arms side by side.

| column | contents |
|---|---|
| `cell`, `kind`, `board_or_role`, `family`, `agg_tag`, `f_inflate` | as in the long table |
| `<arm>_n_pix`, `<arm>_mu_ps`, `<arm>_se_ps`, `<arm>_tau`, `<arm>_tost`, `<arm>_guard_verdict`, `<arm>_present` | for each of `xfer`, `xfer_lin`, `xfer_lin_lpo`, `xfer_clip` and `xfer_rescale`, the columns of the long table and whether the arm has the row (35 columns) |
| `cov_C`, `lever_X`, `lever_X_all`, `spec_primary_row` | of the `xfer` row |
| `delta_xferlin_minus_xfer_mu_ps` | `xfer_lin_mu_ps` minus `xfer_mu_ps` |
| `<arm>_self_transfer` | for each of the five arms, the `self_transfer` flag of the long-table row the arm holds (as in the long table); empty when the arm has no row (5 columns, appended by the reduction) |

The study's builder keeps one long-table row per arm and key. On 12 rows (F2 and F3) the `xfer`,
`xfer_lin`, `xfer_clip` and `xfer_rescale` arms hold the self-transfer row (1-5 pixels) beside a
`xfer_lin_lpo` arm that is not; the same arms' rows over the other pixels (3-70) are in the long
table only.

`universality/twc_bank_best.csv`, 18414 rows from the 63 bank files (DESY May 17 files and 3878
rows; August 10 and 2286; March H1 13 and 4620, F1 15 and 5448; July H1 4 and 884, F1 4 and
1298). One row = one board pixel in one run: of its fits on the whole run with every board in
the solve, not a placebo, fitted on the pixel's own tracks, the one with the most events
(`n_fit`; the first such row on a tie, the study's own rule). The bank's `hv`, `temp` and
`angle_deg` columns are empty in every file and left out; the conditions come from the campaign
modules.

| column | contents |
|---|---|
| `campaign`, `run`, `telescope` | the study's codes; the telescope (`h1`, `f1`) for IRRAD, empty for DESY |
| `board_id`, `role`, `row`, `col` | the board, its role in the combination, and the pixel |
| `combo`, `src_triple` | the combination and the triple the fit was made on |
| `n_fit` | the events of the fit (at most 6000) |
| `c2`, `c1`, `c0` | the correction delta(TOT) = c2 TOT^2 + c1 TOT + c0, TOT and delta in ps |
| `q05`, `q95`, `tot_min`, `tot_max` | the 5 % and 95 % quantiles, the smallest and the largest TOT of those events (ps) |
| `cal_mean` | the board's mean CAL code in those events |
| `hist_lo`, `hist` | the TOT histogram of those events: the lower edge (ps) and the counts in 25 ps bins, as a JSON list |

## Environment variables

Every variable the package and the notebooks read. Set them before the notebook's first import of
`etroc_plots` (in a notebook: restart the kernel after changing one).

| variable | default | what it moves |
|---|---|---|
| `ETROC_CAMPAIGN` | `irrad_2026` | the campaign module in `campaigns/` |
| `ETROC_INPUTS` | `/eos/user/m/musafdar/ETROC_plot_inputs/irrad_2026` | the tables folder (`INPUTS`) |
| `ETROC_IV_EOS_MARCH` | `/eos/user/m/musafdar/CERN_IRRAD_Mar2026/IVCurves` | the raw March tree (`EOS_ROOT`); every raw default below is built from it |
| `ETROC_IV_EOS_MARCH_WEEK2` | `$ETROC_IV_EOS_MARCH/week2` | the week-2 scans and logs alone |
| `ETROC_IV_EOS_MARCH_15E14` | `$ETROC_IV_EOS_MARCH/laptop_mirror_15e14` | the 1.5e15-step scans and logs alone |
| `ETROC_PREIRRAD_LOG` | `$ETROC_IV_EOS_MARCH_WEEK2/fine_iv_H1_m25c_17cm_03162026_0313_interpolated.csv` | the pre-irradiation slow-control log (read only by `preirrad_current --rebuild`) |
| `ETROC_PREIRRAD_CONDITIONS_CSV` | `$ETROC_INPUTS/march/conditions_March2026_H1_HPK.csv` | the H1 run conditions (read only by `preirrad_current --rebuild`) |
| `ETROC_GOOD_RUNS_CSV` | `$ETROC_INPUTS/july/good_runs.csv` | the good-run list |
| `ETROC_COMBO_CHECK_JSON` | `$ETROC_INPUTS/july/res_vs_run_combo_check_jul_values.json` | the July run order and settings of the IV notebook's figures 24 and 25 |
| `ETROC_HV_CYCLES_JUL_CSV` | `$ETROC_INPUTS/july/hv_cycles_jul.csv` | the July HV / LV cycle table of the IV notebook's figures 24 and 25 and the overview notebook's figure 6 (its DAQ restarts) |
| `ETROC_LV_SPANS_JUL_CSV` | `$ETROC_INPUTS/july/lv_spans_jul.csv` | the LV power log's files: the LV-off periods of the overview notebook's figure 6, the LV-on time of the IV notebook's figures 24 and 25 |
| `ETROC_LV_LOG_ROOT` | `/eos/user/m/musafdar/ETROC_TB_2026_combo_tables/lv_history_jul` | the LV power log folders (`LV_LOG_DIRS`), read only by `lv_log.build_lv_spans` |
| `ETROC_RUN_METADATA_ROOT` | `/eos/user/j/jongho` | the DAQ's run folders (`RUN_METADATA_DIRS`), read only by `overview.timeline.build_run_times` and the BL/NW part of `frontend.rebuild` |
| `ETROC_BLNW_ROOT` | `/eos/uscms/store/group/lpcmtdstudies` | the autocalibration scan histories (`BLNW_SQLITES`), read only by the BL/NW part of `frontend.rebuild` |
| `ETROC_DECODED_ROOT` | `/eos/user/j/jongho` | the merged hits (`DECODED_DIRS`), read only by the TDC part of `frontend.rebuild` |
| `ETROC_ANALYSIS_ROOT` | `/eos/user/m/musafdar` | the reconstructed tracks (`ANALYSIS_DIRS`), read only by the tracks part of `frontend.rebuild` |
| `ETROC_TWC_TRANSFER_ROOT` | `/eos/user/m/musafdar/DESY_May2026/analysis/twc_transfer` | the TWC-transfer study's coefficient bank and round-1 report inputs (`TWC_TRANSFER_ROOT`), read only by `universality.reduce` (its bank and master parts, and the cells part's check against the master table) |
| `ETROC_TWC_RESULTS_ROOT` | `$ETROC_INPUTS/universality/results` | the TWC-transfer study's own folder (`TWC_RESULTS_ROOT`: its job lists and axis tables; by default the copy of the files the reduction reads), read only by the cells and axes parts of `universality.reduce` (or `--results-root`) |
| `ETROC_FIGURES` | `figures` (next to the notebook) | where the notebooks write; figures go to `$ETROC_FIGURES/<notebook>/`, such as `$ETROC_FIGURES/iv/` |
