# etroc_plots

    source TestBeam/condor_at_lxplus/envs/load_python39.sh

Run that first, on lxplus, from the repository root. It sets up LCG_104d, the environment this
code is tested in (Python 3.9.12, numpy 1.23.5, pandas 1.5.3, matplotlib 3.7.1, scipy 1.10.1,
mplhep 0.3.12, Pillow 9.5.0, PyYAML 6.0, nbconvert 7.6.0).

Plotting code for ETROC test-beam and irradiation campaigns: a small helper package and one
Jupyter notebook per topic. The notebooks hold the choices made per figure, the package holds what
the figures share (style, loaders, checks), and a campaign module holds what one campaign is (run
lists, chips, fluence steps, input locations). No data lives in the repository.

## What is here

| path | what it is |
|---|---|
| `notebooks/iv.ipynb` (and `iv.py`) | IV scans, bias currents and gain-layer depletion voltages of CERN IRRAD 2026, 42 figures |
| `notebooks/merged.ipynb` (and `merged.py`) | the July IRRAD 2026 run merges against their single runs, and the cost of merging, 7 figures |
| `campaigns/irrad_2026.py` | that campaign: header text, telescopes and chips, fluence ladder, scan catalogue, run settings, input locations |
| `campaigns/irrad_2026_inputs.md` | every input table (contents, origin), where the inputs live, the environment variables that move them; checksums in `irrad_2026_inputs.md5` |
| `style.py` | the house look: header, footer, legends, colours, saving figures and values files |
| `etroc_style.py` | the mplhep CMS style underneath it, the text-overlap audit, single-panel export |
| `checks.py` | the house rules as code, run on every figure as it is drawn and on an output folder afterwards |
| `iv/` | IV loaders and drawing helpers: March slow-control logs, July HV-monitor extracts, k-factor and V_gl |
| `resolution/` | the test-beam result tables: readers, the anointed-combination selection, the combination band, run settings |
| `inputs.py` | the input check every notebook runs first |
| `CONVENTIONS.md` | the rules every figure follows |

## Running a notebook

    cd TestBeam/etroc_plots/notebooks
    export ETROC_FIGURES=$PWD/figures      # where the figures go, as an absolute path
    jupyter nbconvert --to notebook --execute iv.ipynb --output-dir $ETROC_FIGURES \
        --ExecutePreprocessor.timeout=600
    cd ../.. && python3 -m etroc_plots.checks $ETROC_FIGURES/iv

or open `iv.ipynb` in Jupyter and run all cells; the same for `merged.ipynb`, whose figures go to
`$ETROC_FIGURES/merged`. Without `ETROC_FIGURES` the figures go to `figures/iv/` in the notebook's
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
2. Start the campaign's notebook from a copy of `notebooks/iv.py` (or `merged.py`) and set
   `CAMPAIGN = "<name>"` in its setup cell. The notebooks themselves are written for IRRAD 2026:
   their per-figure choices name that campaign's scans, chips and runs (in `merged.py`: the
   constants cell under "The tables and the merges", `TABLE_CAMPAIGN`, `DATA_TEXT` and
   `MERGES`). An exported `ETROC_CAMPAIGN` must match `CAMPAIGN` (the
   notebook stops otherwise); for the command-line modules, such as
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
