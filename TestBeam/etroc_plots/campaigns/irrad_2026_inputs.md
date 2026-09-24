# IRRAD 2026 IV inputs

The IV notebook (`../notebooks/iv.ipynb`) draws the CERN IRRAD 2026 figures (March and July)
from two kinds of input, in two EOS locations. Raw data (the March IV scans, the slow-control logs
and the week-1 spark-test log) are read in place from the March tree:

    /eos/user/m/musafdar/CERN_IRRAD_Mar2026/IVCurves

(`EOS_ROOT` in `irrad_2026.py`; its `week1/`, `week2/` and `laptop_mirror_15e14/` subfolders). The
tables listed below (binned scans, reduced HV-monitor logs, run lists and small caches: 23 files,
31 MB) sit in one folder outside the repository:

    /eos/user/m/musafdar/ETROC_plot_inputs/irrad_2026

`irrad_2026.py` reads from there by default (its `INPUTS`). Both locations are in a personal EOS
area, so reading them needs a share from its owner (Murtaza Safdari). To work from your own copy,
copy the folders and point `ETROC_IV_INPUTS` and `ETROC_IV_EOS_MARCH` at them (all the variables
are in the table at the end):

    cp -r /eos/user/m/musafdar/ETROC_plot_inputs/irrad_2026 /some/where/
    export ETROC_IV_INPUTS=/some/where/irrad_2026
    cp -r /eos/user/m/musafdar/CERN_IRRAD_Mar2026/IVCurves /some/where/march_ivcurves
    export ETROC_IV_EOS_MARCH=/some/where/march_ivcurves

The notebook's first code cell checks every input (`iv_data.check_inputs()`). The tables are
checked against `irrad_2026_inputs.md5`, which lists every file with its checksum: a missing or
unreadable file stops the notebook, and a file that differs from the published one (a cache you
rebuilt, say) is reported. Every raw file the campaign reads (`RAW_INPUTS` in `irrad_2026.py`:
the March scans and logs, the July board-config yaml, and any table moved by its own variable)
must exist and be readable. All problems are listed together, missing files apart from files
without read permission. To check a copy of the tables by hand:

    cd $ETROC_IV_INPUTS && md5sum -c /path/to/TestBeam/etroc_plots/campaigns/irrad_2026_inputs.md5

## Curated by hand

Edit these to change what the figures show.

| file | contents |
|---|---|
| `july/display_runs_jul.csv` | the July runs drawn at the 2e15 and 3.5e15 steps. One row per run: telescope, fluence, rfsel, os (threshold offset), run, preferred (1 marks the run used where a figure needs one run per telescope, fluence, RFSel and offset), combo (anointed = the fixed board combination, band = the min-max over all combinations), bias_note, note |
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

## Environment variables

Every variable the IV package and notebook read. Set them before the notebook's first import of
`etroc_plots` (in a notebook: restart the kernel after changing one).

| variable | default | what it moves |
|---|---|---|
| `ETROC_CAMPAIGN` | `irrad_2026` | the campaign module in `campaigns/` |
| `ETROC_IV_INPUTS` | `/eos/user/m/musafdar/ETROC_plot_inputs/irrad_2026` | the tables folder (`INPUTS`) |
| `ETROC_IV_EOS_MARCH` | `/eos/user/m/musafdar/CERN_IRRAD_Mar2026/IVCurves` | the raw March tree (`EOS_ROOT`); every raw default below is built from it |
| `ETROC_IV_EOS_MARCH_WEEK2` | `$ETROC_IV_EOS_MARCH/week2` | the week-2 scans and logs alone |
| `ETROC_IV_EOS_MARCH_15E14` | `$ETROC_IV_EOS_MARCH/laptop_mirror_15e14` | the 1.5e15-step scans and logs alone |
| `ETROC_PREIRRAD_LOG` | `$ETROC_IV_EOS_MARCH_WEEK2/fine_iv_H1_m25c_17cm_03162026_0313_interpolated.csv` | the pre-irradiation slow-control log (read only by `preirrad_current --rebuild`) |
| `ETROC_PREIRRAD_CONDITIONS_CSV` | `$ETROC_IV_INPUTS/march/conditions_March2026_H1_HPK.csv` | the H1 run conditions (read only by `preirrad_current --rebuild`) |
| `ETROC_GOOD_RUNS_CSV` | `$ETROC_IV_INPUTS/july/good_runs.csv` | the good-run list |
| `ETROC_COMBO_CHECK_JSON` | `$ETROC_IV_INPUTS/july/res_vs_run_combo_check_jul_values.json` | the July run order and settings of figures 24 and 25 |
| `ETROC_HV_CYCLES_JUL_CSV` | `$ETROC_IV_INPUTS/july/hv_cycles_jul.csv` | the July HV / LV cycle table of figures 24 and 25 |
| `ETROC_FIGURES` | `figures` (next to the notebook) | where the notebook writes; figures go to `$ETROC_FIGURES/iv/` |
