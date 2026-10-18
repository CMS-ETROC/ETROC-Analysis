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
| `campaigns/irrad_2026.py` | that campaign: header text, telescopes and chips, fluence ladder, scan catalogue, run settings, input locations |
| `campaigns/irrad_2026_inputs.md` | every input table (contents, origin), where the inputs live, the environment variables that move them; checksums in `irrad_2026_inputs.md5` |
| `style.py` | the house look: header, footer, legends, colours, saving figures and values files |
| `etroc_style.py` | the mplhep CMS style underneath it, the overlap audit (texts, arrows and tagged marks), single-panel export |
| `checks.py` | the house rules as code, run on every figure as it is drawn and on an output folder afterwards |
| `iv/` | IV loaders and drawing helpers: March slow-control logs, July HV-monitor extracts, k-factor and V_gl |
| `resolution/` | the test-beam result tables: readers, the anointed-combination selection, the combination band, run settings; the pixel-map layouts |
| `overview/` | the setup and method schematics, and the campaign timeline with its run-time table builder |
| `frontend/` | the 12 front-end tables, their rebuild and drawing. The tables: `blnw_by_run` (baseline and noise width per board and run) and `blnw_by_pixel` (per pixel), `tdc_pixel_hists` and `tdc_cal_modes` (TDC codes of one pixel per chip, CAL mode of every pixel), and from the track files `triples`, `tot_pixel_medians`, `tot_pooled_summary`, `tot_pooled_hist`, `dtoa_triples`, `dtoa_pooled_hist`, `pixel_hists` (the fixed pixels' TOT in ns and in TOT codes, and their TOA difference) and `rfsel_pair_hist` (the RFSel pairs and their time-control group) (columns in `campaigns/irrad_2026_inputs.md`). The rebuild: `python -m etroc_plots.frontend.rebuild --out DIR [--part blnw\|tdc\|tracks] [--runs ...] [--rank rows\|bytes]`, run from `TestBeam/` (the track part imports the pipeline's time-walk solver, `condor_at_lxplus/core/twc_solver.py`); on one lxplus core the TDC part took 5.5 min and the track part 87 min for 30 runs (about three minutes per run). `--rank rows` (the default) gives each pixel the triple with the most events, `--rank bytes` the largest file, the rule of the talk figures. The BL/NW part reads the scan histories on the LPC EOS: `--part blnw` with `ETROC_BLNW_ROOT` pointing at copies of them (and `ETROC_RUN_METADATA_ROOT` at copies of the run folders where those are out of reach) rebuilds both BL/NW tables in seconds; the published ones (2026-10-18) agree with the earlier build of `blnw_by_run.csv` in every cell |
| `inputs.py` | the input check every notebook runs first |
| `lv_log.py` | the July LV power log: its file-spans table builder, the LV-off periods and the LV-on time before a run |
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

The per-loop census and the readout mechanism are documented with the DAQ work and will be
published with it.

## Running a notebook

    cd TestBeam/etroc_plots/notebooks
    export ETROC_FIGURES=$PWD/figures      # where the figures go, as an absolute path
    jupyter nbconvert --to notebook --execute iv.ipynb --output-dir $ETROC_FIGURES \
        --ExecutePreprocessor.timeout=600
    cd ../.. && python3 -m etroc_plots.checks $ETROC_FIGURES/iv

or open `iv.ipynb` in Jupyter and run all cells; the same for `merged.ipynb`, `maps.ipynb`,
`overview.ipynb` and `frontend.ipynb`, whose figures go to `$ETROC_FIGURES/merged`,
`$ETROC_FIGURES/maps`, `$ETROC_FIGURES/overview` and `$ETROC_FIGURES/frontend`. Without `ETROC_FIGURES` the figures go to `figures/iv/` in the notebook's
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
