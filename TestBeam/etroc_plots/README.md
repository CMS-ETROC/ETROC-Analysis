# etroc_plots

    source TestBeam/condor_at_lxplus/envs/load_python39.sh

Run that first, on lxplus, from the repository root. It sets up LCG_104d, the environment this
code is tested in (Python 3.9.12, numpy 1.23.5, pandas 1.5.3, matplotlib 3.7.1, scipy 1.10.1,
mplhep 0.3.12, Pillow 9.5.0, PyYAML 6.0, pyarrow 11.0.0, nbconvert 7.6.0).

Plotting code for ETROC test-beam and irradiation campaigns: a small helper package and one
Jupyter notebook per topic. The notebooks hold the choices made per figure, the package holds what
the figures share (style, loaders, checks), and a campaign module holds what one campaign is (run
lists, chips, fluence steps, input locations). No data lives in the repository.

## What is here

| path | what it is |
|---|---|
| `notebooks/iv.ipynb` (and `iv.py`) | IV scans, bias currents and gain-layer depletion voltages of CERN IRRAD 2026, 42 figures |
| `notebooks/merged.ipynb` (and `merged.py`) | the July IRRAD 2026 run merges against their single runs, and the cost of merging, 7 figures |
| `notebooks/maps.ipynb` (and `maps.py`) | per-pixel time-resolution maps of CERN IRRAD 2026, one run per fluence step, 18 figures |
| `notebooks/overview.ipynb` (and `overview.py`) | the CERN IRRAD 2026 setup and method as schematics (telescope, beam profile, boards, track selection, the RFSel setting) and the timeline of both campaigns, 6 figures |
| `notebooks/frontend.ipynb` (and `frontend.py`) | the CERN IRRAD 2026 front end: baseline and noise width of every board against fluence and before / after cooling down, the CAL, TOA and TOT codes and CAL-code maps at 1.5e15 before / after cooling down, and the TOT and the time-walk-corrected TOA difference of the tracks against fluence (boards and single pixels), and the TOT of one chip at two RFSel settings and at fixed settings over time, 16 figures; about 80 s on one lxplus core |
| `notebooks/universality.ipynb` (and `universality.py`) | whether a time-walk correction fitted on one run, board or pixel carries over to another: one pixel's correction against TOT across angle (DESY May 2026), bias and fluence (CERN IRRAD March 2026) and from March to July; then the pair penalty of transferring: against how much of the target's TOT the source covers and how far it reaches beyond it, per main question of the study, before and after a linear refit, the August settings steps and three controls; 6 figures; about 50 s on one lxplus core |
| `notebooks/settings.ipynb` (and `settings.py`) | the time resolution of every board at the last July step of CERN IRRAD 2026 (bookkeeping fluence 3.5e15) against the front-end settings its runs stepped: the threshold offset at RFSel 2, the settings class (RFSel, offset), and the step from RFSel 2 to 3; then the threshold over the noise of every board (offset over the median noise width of its pixels, with their 16-84 % range), the resolution against it, the baseline and noise width against RFSel, and what RFSel changes by design beside what one chip shows when it steps; 7 figures; about 45 s on one lxplus core |
| `notebooks/resolution.ipynb` (and `resolution.py`) | the time resolution of every board of CERN IRRAD 2026 against its bias and against time: the March bias scans before irradiation and after each step, 1.5e15 in March and in July, July 1.5e15 against the hours since the first run there (a stability check after the reinstall), July 2e15 against the hours after the end of the irradiation with each run's in-run bias currents, July 3.5e15 against bias and against the hours after the end of the irradiation, every July step against bias; then the fluence ladder: the resolution of every board at each step, one run per campaign and step, its numbers as a table (also .csv and .md), and the pixels with a resolution at each step; then the combination split at 3.5e15: each board's resolution in every combination containing it, run by run (an accepted merge in place of its runs, as on the settings figures); then an appendix of the July talk's figures drawn from the same points: the summary bias scan at fewer steps (July 1.5e15 and 2e15; July 1.5e15 to 3.5e15; March) and July 3.5e15 against the hours after the end of the irradiation; and the combination checks: each board's resolution in its anointed combination, in the combination without its HV partner and at the pair-corrected midpoint, against the bias (July and March) and run by run (every July run, with the HV cycles, LV cycles, irradiation steps and shared autocalibration scans between runs); 18 figures; about 90 s on one lxplus core (88 and 89 s, two runs) |
| `campaigns/irrad_2026.py` | that campaign: header text, telescopes and chips, fluence ladder, scan catalogue, run settings, input locations |
| `campaigns/desy_2026.py` | facts of the DESY May and August 2026 runs that the universality figures draw beside IRRAD: beam, run settings from the run-config yamls, the offset-holder runs |
| `campaigns/irrad_2026_inputs.md` | every input table (contents, origin), where the inputs live, the environment variables that move them; checksums in `irrad_2026_inputs.md5` |
| `style.py` | the house look: header, footer, legends, colours, saving figures and values files |
| `etroc_style.py` | the mplhep CMS style underneath it, the overlap audit (texts, arrows and tagged marks), single-panel export |
| `checks.py` | the house rules as code, run on every figure as it is drawn and on an output folder afterwards |
| `iv/` | IV loaders and drawing helpers: March slow-control logs, July HV-monitor extracts, k-factor and V_gl |
| `resolution/` | the test-beam result tables: readers, the reference-combination selection, the combination band, run settings, the settings classes of one fluence step; the pixel-map layouts; the resolution figures (`curves.py`: points of single and merged runs, the standard operating point, the merge rule and its per-figure override, the good-run list, every display run as single runs, the clocks after a stop and since a first run, the colour rule, one run per condition, the low-pixel threshold, the bias-scan and time-series primitives, the summary bias scan drawn in part with its colours); the fluence ladder (`ladder.py`: one run per campaign and step, the ladder and pixel-yield figures, the floor-"none" pixel lookup); the numbers table (`numbers_table.py`: the ladder's points as a table, .csv, .md and a table figure); the settings figures (`settings.py`: the settings guard, the threshold over the noise and its figures, the slot layouts, the layout audit); the combination split (`combo_split.py`: the references and every display run of a step, an accepted merge in place of its runs, each board's value in every combination, the figure); the combination checks (`combo_check.py`: the partner-excluded combination and the pair-corrected midpoint from the combination rows, a missing combination reported, the HV cycles between runs, the shared autocalibration scans, the bias and run figures) |
| `overview/` | the setup and method schematics (among them the RFSel block diagram, alone or beside a measured RFSel pair), and the campaign timeline with its run-time table builder |
| `frontend/` | the 12 front-end tables, their rebuild and drawing. The tables: `blnw_by_run` (baseline and noise width per board and run) and `blnw_by_pixel` (per pixel), `tdc_pixel_hists` and `tdc_cal_modes` (TDC codes of one pixel per chip, CAL mode of every pixel), and from the track files `triples`, `tot_pixel_medians`, `tot_pooled_summary`, `tot_pooled_hist`, `dtoa_triples`, `dtoa_pooled_hist`, `pixel_hists` (the fixed pixels' TOT in ns and in TOT codes, and their TOA difference) and `rfsel_pair_hist` (the RFSel pairs and their time-control group) (columns in `campaigns/irrad_2026_inputs.md`). The rebuild: `python -m etroc_plots.frontend.rebuild --out DIR [--part blnw\|tdc\|tracks] [--runs ...] [--rank rows\|bytes]`, run from `TestBeam/` (the track part imports the pipeline's time-walk solver, `condor_at_lxplus/core/twc_solver.py`); on one lxplus core the TDC part took 5.5 min and the track part 87 min for 30 runs (about three minutes per run). `--rank rows` (the default) gives each pixel the triple with the most events, `--rank bytes` the largest file, the rule of the first presented figures. The BL/NW part reads the scan histories on the LPC EOS: `--part blnw` with `ETROC_BLNW_ROOT` pointing at copies of them (and `ETROC_RUN_METADATA_ROOT` at copies of the run folders where those are out of reach) rebuilds both BL/NW tables in seconds; the published ones (2026-10-18) agree with the earlier build of `blnw_by_run.csv` in every cell |
| `universality/` | the TWC-transfer study's tables, their reduction and the TWC curves. The six tables (`univ_cells`, `univ_master_cells`, `univ_axis_r2`, `univ_axis_F_wide`, `univ_axis_F_long`, `twc_bank_best`; columns in `campaigns/irrad_2026_inputs.md`) are made by `python -m etroc_plots.universality.reduce --out DIR` (the study's files read by default from their copy in `universality/results/` of the inputs), run from `TestBeam/` on lxplus, in about 15 s on one core; `transfer.py` reads them (with the study's verdict override) and holds the curve arithmetic, `curves.py` draws the curves, `levels.py` the pair-penalty figures |
| `inputs.py` | the input check every notebook runs first |
| `lv_log.py` | the July LV power log: its file-spans table builder, the LV-off periods and the LV-on time before a run |
| `tests/` | pytest tests (how to run them: under Running a notebook): `test_frontend.py` (14 tests) of the code that builds the frontend tables: the TOT code inversion and its bins, the RFSel group rows, the runs of the fluence ladder, the run set of the rebuild, the hits stated in the code boxes, the colour range of the CAL difference maps and the BL/NW tables built from synthetic scans; `test_universality.py` (19 tests) of the universality readers: the study's verdict override, the axis tables the reduction composes, the self-transfer flag and the rows left out by it, the axis-F rows, the DESY beam of a set of runs, the DESY chip names and the reduction's default study folder; `test_settings.py` (25 tests) of the settings helpers (the settings classes drawing a merge only when the good-run list marks it good, the settings guard, the threshold over the noise from the per-pixel noise widths (a merge's pooled scans too), the small open marker of a board under 128 pixels, the slot positions, the offset markers, the layout audit, the BL/NW point offsets, the RFSel pair readers); `test_resolution.py` (24 tests) of the resolution curves helpers (the merge rule and its override, the good-run list with merges, every display run as single runs, the points of single and merged runs, the standard operating point, the clocks with their bounds, the colour rule, the low-pixel threshold, the fluence ladder's picks, one run per condition, the floor-"none" pixel lookup, the numbers table, the combination split's runs, points and layout with and without a merge, the summary bias scan drawn in part, the combination checks' partner-excluded combination, midpoint, missing combinations, HV cycles and autocalibration groups) |
| `CONVENTIONS.md` | the rules every figure follows |

## Data quality: the IRRAD 2026 readout was saturated

The IRRAD 2026 beam loops (H1 and F1 telescopes, March and July campaigns) were recorded with the
KCU105 readout saturated. Measured on the raw loop files in October 2026: 47 to 81 % of the ETROC2
packets reached disk without their header, the DAQ FIFO reported `global_full` on 14 to 23 % of
the records, and 5 to 14 % of the packets that kept their header fail their CRC. The decoding step
discards every headerless hit, so the hit tables behind every IRRAD figure drawn from readout hits
hold roughly a quarter to a half of the hits the chips sent, selected by the state of the readout
buffers at the time. Three loops are exceptions (H1 March run 21, clean; F1 July run 59 and H1
July run 51, 8 and 17 % of the packets without their header); the DESY May and August 2026 data
are clean.

What this means for the figures here:

- Time-resolution, time-walk, TOT, TOA and CAL figures are measurements on the surviving hits.
  The TDC values of those hits are not corrupted: every complete packet of the clean files passes
  its CRC (the decoding does not check it, so the IRRAD packets that fail it are kept). Whether
  the selection by buffer state biases the resolution, the TOT or the time walk is open and needs
  a dedicated comparison (resolution against the headerless share per loop); until it runs, read
  these results as valid for the surviving hits with the selection bias unquantified.
- Anything count-, rate- or efficiency-like on the IRRAD 2026 data (hit counts, occupancy, events
  per pixel, the `n_pixels` of a board fit when it is limited by hits) measures DAQ loss, not the
  sensor. No figure in this package quotes an efficiency; the pixel counts it prints are
  populations of the surviving sample.
- The baseline and noise width come from the autocalibration scans, not from readout hits, and
  are unaffected; so are the IV scans, the bias currents and V_gl.

Every frontend figure drawn from readout hits (fe03-fe16: the TDC codes and CAL maps from the
merged hits, the TOT and TOA difference from the track files) ends its footer with the line
"Hits and pixel counts are the surviving readout sample, selected by buffer state; see README,
Data quality." (`READOUT_FOOTER` in the campaign module). Figures fe01-fe02 do not carry it: they
are drawn from the autocalibration scans, not from event readout.

The universality figures with IRRAD runs end their footer with the same line, after the part it
applies to: un01 (its bias, fluence and March-to-July panels), un02 (the rows of axes D and I),
un03 (the cells of levels g and h), un04 (the rows of cells F8 to F11, re-scored fluence and
March-to-July cells) and un06 (its March-to-July panel). Figure un05 draws DESY runs only and does
not carry it; the DESY 2026 data are clean.

The maps figures (maps01-maps18) are drawn from the resolution tables, not from the hits directly.
Their only count of readout data is a pixel count, so each of their footers ends with the shorter
line "Pixel counts are the surviving readout sample." (`SURVIVING_TEXT` in the maps notebook).

The settings figures st01-st05 are drawn from the resolution tables too, and give the shorter
line after the pixel counts of their small open markers. Figure st07 also draws the TOT of the
track files, so its footer ends with the full line. Figure st06 is drawn from the
autocalibration scans and carries neither.

The per-loop census and the readout mechanism are documented with the DAQ work and will be
published with it.

## Running a notebook

    cd TestBeam/etroc_plots/notebooks
    export ETROC_FIGURES=$PWD/figures      # where the figures go, as an absolute path
    jupyter nbconvert --to notebook --execute iv.ipynb --output-dir $ETROC_FIGURES \
        --ExecutePreprocessor.timeout=600
    cd ../.. && python3 -m etroc_plots.checks $ETROC_FIGURES/iv

or open `iv.ipynb` in Jupyter and run all cells; the same for `merged.ipynb`, `maps.ipynb`,
`overview.ipynb`, `frontend.ipynb`, `universality.ipynb`, `settings.ipynb` and `resolution.ipynb`,
whose figures go to `$ETROC_FIGURES/merged`, `$ETROC_FIGURES/maps`, `$ETROC_FIGURES/overview`,
`$ETROC_FIGURES/frontend`, `$ETROC_FIGURES/universality`, `$ETROC_FIGURES/settings` and
`$ETROC_FIGURES/resolution`. Without `ETROC_FIGURES` the figures go to `figures/iv/` in the notebook's
folder; check them from `TestBeam/` with
`python3 -m etroc_plots.checks etroc_plots/notebooks/figures/iv`. The inputs are on EOS, in an area
that needs a share from its owner; `campaigns/irrad_2026_inputs.md` says where they are and how to
point the notebook at your own copy. Each notebook's first code cell checks the inputs it reads and
stops with a list of whatever is missing or unreadable.

Each figure is written as PNG (200 dpi) and PDF, with `<stem>_values.json` beside it: the numbers
drawn, the input files read, the commit of the code (with `-dirty` when it had uncommitted changes)
and what the layout audits found. The last command above reads those files back and measures every
PNG; it exits with status 1 when a figure breaks a rule of `CONVENTIONS.md`, and when the folder
holds no figures at all.

The tests run from `TestBeam/` in LCG_104d: `python -m pytest -p no:cacheprovider etroc_plots/tests`
(the tests that read the published tables, axis F and the corrected rows of axes E, J and K, are
skipped where the universality tables cannot be read).

## Editing a notebook

Each `.ipynb` and the `.py` of the same name are one notebook, paired by jupytext (`py:percent`).
Edit either one,
then sync the pair and strip the outputs before committing. Neither tool is in LCG_104d, so install
the tested versions into a folder of their own:

    python3 -m pip install --target $HOME/nbtools jupytext==1.19.5 nbstripout==0.8.2
    export PYTHONPATH=$HOME/nbtools:$PYTHONPATH
    cd TestBeam/etroc_plots/notebooks
    python3 -m jupytext --sync iv.ipynb
    python3 -m nbstripout iv.ipynb

If `~/.local` holds packages that clash with LCG_104d (a second pandas, say), run everything with
`PYTHONNOUSERSITE=1`.

## A new campaign

1. Copy `campaigns/irrad_2026.py` to `campaigns/<name>.py` and change what differs: header text,
   telescopes and chips, fluence ladder and colours, run settings, input locations. List the input
   tables with their checksums in `campaigns/<name>_inputs.md5` (`md5sum` output, paths relative to
   the tables folder), point `INPUTS_MANIFEST` at it, and describe the tables in
   `campaigns/<name>_inputs.md`.
2. Start the campaign's notebook from a copy of `notebooks/iv.py` (or `merged.py`, `maps.py`,
   `overview.py`, `frontend.py`) and set `CAMPAIGN = "<name>"` in its setup cell. The notebooks themselves are
   written for IRRAD 2026: their per-figure choices name that campaign's scans, chips and runs (in
   `merged.py`: the constants cell under "The tables and the merges", `TABLE_CAMPAIGN`, `DATA_TEXT`
   and `MERGES`; in `maps.py`: the constants cell under "The tables and the runs", `JULY_SETTINGS`
   (the settings of the display runs taken from the display-run list; `RUNS` is made from these and
   the campaign's `MARCH_DISPLAY_RUNS`), `PAIR_FLUENCE`, `PAIR_RUNS` (made from the campaign's
   `COOLDOWN_PAIRS`), `DEFAULT_RFSEL`,
   `CAMPAIGN_TEXT`, `RUNS_TEXT`, `PAIR_RUNS_TEXT` and `BLOCK_TITLE`; the maps' colour scale and fit
   threshold are `VMIN`, `VMAX` and `N_MIN_FIT` in `resolution/pixel_maps.py`, whose layouts draw
   four boards per telescope; in `overview.py`: the words and setup facts in each figure's cell and
   `TIMELINE_FOOTER`, while the sensors, beam, run folders, LV log folders and March display runs
   are in the campaign module; in `frontend.py`: the words in each figure's cell and the CAL colour
   scales, while the runs (`MARCH_DISPLAY_RUNS`, `COOLDOWN_PAIRS`, `FRONTEND_JULY_SETTINGS`) and the
   sources of the front-end tables are in the campaign module). An exported `ETROC_CAMPAIGN` must match `CAMPAIGN` (the notebook
   stops otherwise); for the command-line modules, such as
   `python3 -m etroc_plots.iv.preirrad_current`, it alone chooses the campaign. The campaign is
   read once, on the first `etroc_plots` import.
3. On import, the package checks that the campaign module defines every name its code reads
   (`REQUIRED` in `campaigns/__init__.py`) and lists the missing ones.

The IV helpers read the formats this campaign was recorded in (the March slow-control CSV logs and
the July HV-monitor extracts); the resolution helpers read the result tables of the test-beam chain
(`resolution/tables.py` gives their columns). The scripts that build those tables are not in this
repository yet (`campaigns/irrad_2026_inputs.md` names them and what they read), so a new
campaign's tables have to be made with the same columns. A campaign recorded in other formats
needs loaders of its own; the style, the checks and the saving code do not depend on the format.
