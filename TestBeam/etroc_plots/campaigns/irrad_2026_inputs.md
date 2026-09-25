# IRRAD 2026 inputs

The notebooks (`../notebooks/iv.ipynb`, `../notebooks/merged.ipynb`) draw the CERN IRRAD 2026
figures (March and July) from two kinds of input, in two EOS locations. Raw data (the March IV
scans, the slow-control logs and the week-1 spark-test log, read by the IV notebook only) are read
in place from the March tree:

    /eos/user/m/musafdar/CERN_IRRAD_Mar2026/IVCurves

(`EOS_ROOT` in `irrad_2026.py`; its `week1/`, `week2/` and `laptop_mirror_15e14/` subfolders). The
tables listed below (binned scans, reduced HV-monitor logs, run lists, small caches and the
test-beam result tables: 31 files, 100 MB) sit in one folder outside the repository:

    /eos/user/m/musafdar/ETROC_plot_inputs/irrad_2026

`irrad_2026.py` reads from there by default (its `INPUTS`). Both locations are in a personal EOS
area, so reading them needs a share from its owner (Murtaza Safdari). To work from your own copy,
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

## Made outside this package

The code that made these is not in this repository; the origin column says what each came from.

| file | contents | origin |
|---|---|---|
| `july/timeline_60s.csv.gz` | the whole July HV-monitor log, both telescopes and all four channels, in 60 s bins (UTC) | the July HV-log extraction of 2026-08-22 (its `timeline_60s.csv`, gzipped). Its source files are named in `july_timeline.json` (`source_host`, `source_dirs`) |
| `july/july_timeline.json` | that extraction in one object: clock finding, board map, radiation stops, irradiation windows, logging gaps, HV-monitor file index, every IV scan at six reference biases, every per-board run | the same extraction. Edited after the extraction: two keys and one sentence renamed to "cooldown", numbers unchanged |
| `july/july_inrun_currents.csv` | per run and board: in-run bias current (plateau window, median, first and last hour, 5-95 % range), nominal bias, logging status, radiation-stop time, the IV scans before and after | the same extraction |
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
| `tables/runs_summary.csv` | one row per run: status, and per board the role, chip, bias, bookkeeping fluence, threshold offset and RFSel | `build_irrad_summary.py`: the board settings from the run-config yamls (`TestBeam/board_configs_yaml/CERN_Irrad_2026Mar.yaml` and `CERN_Irrad_2026Jul.yaml`), the rest from each run's quote |
| `tables/irrad/PIXEL_TABLES_README.md`, `tables/merged/PIXEL_TABLES_README.md` | the notes each build wrote beside its tables | the same builds |

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
| `ETROC_HV_CYCLES_JUL_CSV` | `$ETROC_INPUTS/july/hv_cycles_jul.csv` | the July HV / LV cycle table of the IV notebook's figures 24 and 25 |
| `ETROC_FIGURES` | `figures` (next to the notebook) | where the notebooks write; figures go to `$ETROC_FIGURES/iv/` and `$ETROC_FIGURES/merged/` |
