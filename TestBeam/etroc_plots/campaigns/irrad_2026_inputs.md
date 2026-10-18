# IRRAD 2026 inputs

The notebooks (`../notebooks/iv.ipynb`, `merged.ipynb`, `maps.ipynb`, `overview.ipynb` and
`frontend.ipynb`) draw
the CERN IRRAD 2026 figures (March and July) from two kinds of input, in two EOS locations. Raw data (the March IV
scans, the slow-control logs and the week-1 spark-test log, read by the IV notebook only) are read
in place from the March tree:

    /eos/user/m/musafdar/CERN_IRRAD_Mar2026/IVCurves

(`EOS_ROOT` in `irrad_2026.py`; its `week1/`, `week2/` and `laptop_mirror_15e14/` subfolders). The
tables listed below (binned scans, reduced HV-monitor logs, run lists, small caches and the
test-beam result tables: 44 files, 117 MB) sit in one folder outside the repository:

    /eos/user/m/musafdar/ETROC_plot_inputs/irrad_2026

`irrad_2026.py` reads from there by default (its `INPUTS`). Both locations are in a personal EOS
area, so reading them needs a share from its owner (Murtaza Safdari). Rebuilding some of the
tables reads more places, which the notebooks never read: the LV power-log copies (`LV_LOG_ROOT`,
in the same personal area) for `july/lv_spans_jul.csv`; the DAQ's run folders
(`RUN_METADATA_ROOT`, another personal EOS area) for `tables/run_times.csv` and the two BL/NW
tables of `frontend/`; the merged hits (`DECODED_ROOT`, the same area) for the two TDC tables
of `frontend/`; the reconstructed tracks (`ANALYSIS_ROOT`, the personal area of the tables) for the
eight track tables of `frontend/`; and the autocalibration scan histories (`BLNW_ROOT`, a group
area on the LPC EOS) for the two BL/NW tables of `frontend/`. Each needs a share from its owner. To work from your own copy,
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
are not readout hits and are unaffected. The package README, "Data quality", gives the numbers,
the loops that are exceptions and what this means per figure type.

## Curated by hand

Edit these to change what the figures show.

| file | contents |
|---|---|
| `july/display_runs_jul.csv` | the July runs drawn at the 1.5e15, 2e15 and 3.5e15 steps. One row per run: telescope, fluence, rfsel, os (threshold offset), run, preferred (1 marks the run used where a figure needs one run per telescope, fluence, RFSel and offset), combo (anointed = the fixed board combination, band = the min-max over all combinations), bias_note, note |
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

## Test-beam result tables

The time-resolution results of the test-beam chain (`TestBeam/condor_at_lxplus`), as flat tables
in `tables/`; the merged-runs notebook reads them. `../resolution/tables.py` describes their
columns, and each folder carries a `PIXEL_TABLES_README.md` from its build. The code that made
them is not in this repository yet: the build scripts are named in the origin column, and the
run merging itself (the merged step-13 outputs) was done by the run-merging study's scripts
(`TestBeam/condor_at_lxplus/merge_runs/`), also not in this repository yet.

| file | contents | origin |
|---|---|---|
| `tables/irrad/board_table_irrad.csv` | one row per run, 3-board combination, board, track variant and event floor, March and July: the board's time resolution, pixel spread and pixel count, and whether the combination is the anointed one for that board | `build_pixel_tables.py --family irrad`, 2026-09-15/16, from each run's step-13 `resolution_table_<combo>.csv` with the functions of `TestBeam/condor_at_lxplus/utils/quote_resolution.py` |
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
  `TestBeam/`. `--rank bytes` picks each pixel's triple by file size, the rule of the CERN IRRAD
  2026 talk figures (kept for comparisons); the default, `rows`, by its events.

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
| `ETROC_FIGURES` | `figures` (next to the notebook) | where the notebooks write; figures go to `$ETROC_FIGURES/<notebook>/`, such as `$ETROC_FIGURES/iv/` |
