# twc_transfer

    source TestBeam/condor_at_lxplus/envs/load_python39.sh

Run that first, on lxplus, from the repository root (LCG_104d: Python 3.9.12, numpy 1.23.5,
pandas 1.5.3, scipy 1.10.1, scikit-learn 1.2.2, pyarrow 11.0.0). Pin the thread
pools and run at low priority; the width fits are single-threaded by design and a free thread
pool only adds load:

    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
           NUMEXPR_NUM_THREADS=1 ARROW_NUM_THREADS=1

The library of the TWC transfer study: a time-walk correction (TWC) fitted on one sample, the
source, applied to another, the target, and the time resolution that costs. The solve, the
event cut and the width are the pipeline's own (`condor_at_lxplus/core/twc_solver.py` and
`bootstrap.py`, imported unmodified); this package adds the bookkeeping around them: frozen
evaluation folds, per-event leakage checks, a coefficient bank, the arms, the analytic Tier-0
screen, the transfer cell, the statistics, the aggregation of cells, the round-1 chain to the
table of the level cells and the axis tables. `SPEC.md` beside this file defines all of it.

## Environment variables

| variable | meaning | default |
|---|---|---|
| `ETROC_TWC_CORE` | the pipeline core directory to import `bootstrap` and `twc_solver` from; a core of the hashes of `campaigns.CELLS_CORE_SHA1` reproduces the study's cells, another reruns them (Versions) | this repository's `TestBeam/condor_at_lxplus/core` |
| `ETROC_TWC_ROOT_<CAMPAIGN>` | the step-10 analysis root of a campaign (`MAR`, `JUL`, `DESY`, `AUG`); track files are `<root>/tracks_<run>/<combo>/time/track_*.parquet` | the campaign trees in `campaigns.py` |
| `ETROC_TWC_TRANSFER_ROOT` | the read root: the study's outputs (bank, cells, tables), never written; the same variable and default as `etroc_plots` | the study's output tree in `campaigns.py` |
| `ETROC_TWC_OUT` | the write root; a relative one is made absolute | none: `campaigns.write_root()` raises when it is unset, or when it is or lies inside the read root, an output tree of the study's job lists, a data root or this checkout (`campaigns.protected_roots()`, real paths compared) |

`ETROC_TWC_CORE` is read once, at the first import of the package; the others are read when
`campaigns` is first imported (the roots) or when `transfer_root()` / `write_root()` is called.

## Modules

| module | what it holds |
|---|---|
| `campaigns` | step-10 roots per campaign, the read root, the write root, the event cap, the core sha1 of the study's cells, the waves and the version tables, f per campaign (`F_CAL`, `F_GATE`, `AGG_F`), the thin rule, the levels, the axis tags and what their cells compare |
| `_core` | loads the pipeline core, exposes `BS` (bootstrap) and `TS` (twc_solver), `core_versions()` |
| `naming` | combos and roles, pixel triples, run directories, sibling combos, track ranking, `ensure_dirs(root)` |
| `events` | `load_events`, `Triple`, per-event identity (`event_keys`), the leakage asserts |
| `solve` | the joint solve, the gauge, the apply line, raw and centred bases, coefficient covariance, transfer variants |
| `bank` | `bank_rows_for_triple` and `BANK_COLUMNS`, TOT moments, leave-pair-out source events, TOT overlap metrics |
| `widths` | `width_gmm`, the three-board sigma, `evaluate_arm`, own-TOT slope, the leaky positive control |
| `arms` | `self_coeff_maps`, `transferred_map`, `xfer_lin_map`, `xfer_lin_lpo_map`, the arm names |
| `tier0` | `tier0_delta`, the analytic change of RMS^2 per pair and board |
| `stats` | `kcal_from_cells`, `random_effects`, `tost`, `margin_for`, `source_random_effect`, `holm`, `p_nonequiv` |
| `cell` | the transfer cell: the source of each mode, the board-identity checks, the pixel-to-board protocols, `cmd_cell`, the cell table `CELL_COLUMNS` |
| `agg` | the aggregation of cell tables: one function per rule, `aggregate` (in memory), `cmd_agg`, the table columns `DELTA_COLUMNS`, `AGG_COLUMNS`, `CALIBRATION_COLUMNS` |
| `master` | the round-1 chain: `recon`, `cell_verdicts`, `f_stability`, `cell_final`, `primary_rows`, `recompute`, `master_table`, `level_verdicts`, `run`, `cmd_master` |
| `axes` | the axis tables: `merge_triples`, `axis_table`, `wide_table`, `overlay` (the guardfix overlay), `f_cell_table`, `build`, `cmd_axes`, `AXIS_COLUMNS` |
| `cli`, `__main__` | the command line, `python -m twc_transfer` (below) |
| `jobs` | the job lists: reader, root expansion, `--dry-run`, and the condor driver (`write_dag`, `check_inputs`, `agg_plan`); the study's seven lists are in `jobs/` |

The package re-exports nothing; import the modules. `etroc_plots` never imports it.

## Use

From `TestBeam/`:

```python
from twc_transfer import arms, bank, events, naming, widths

camp, run, combo = "mar", "h1_run1", "extra0-ref1-dut2"
path = naming.top_triples(camp, run, combo, 1)[0]           # the largest track file
tri = events.Triple.from_file(path, camp, run, combo)       # cut, dropna, cap at 6000 events
rows = bank.bank_rows_for_triple(tri, folds="both", guards=("full", "lpo"), placebo=True)
maps = arms.self_coeff_maps(tri)                            # notwc, self_in, self_cf, ...
res = {a: widths.evaluate_arm(tri, m, src_keys=maps.src_keys.get(a), arm=a)
       for a, m in maps.items()}
```

`res[arm]["pooled"][pair]["w"]` is the pair width (FWHM/2.355 of the GMM fit, ps) and
`res[arm]["sigma"][role]` the board sigma; every arm is evaluated on the same events. Nothing
here writes a file. A caller that writes passes `ensure_dirs(campaigns.write_root())`.

## Command line

From `TestBeam/`, with a folder of your own as the write root. The recipe is index, then bank
(and the Tier-0 screen of it), then cells, then their aggregation, then the round-1 chain and
the axis tables, then the universality reduction. Each line reads the study's tree, not the
output of the line before it; "Chaining on your own outputs" below says how to chain them:

    export ETROC_TWC_OUT=/path/to/your/output
    python -m twc_transfer index --campaign mar --run h1_run1 --combo extra0-ref1-dut2
    python -m twc_transfer fit --campaign mar --run f1_run12 --ntrip 40 --folds both --placebo
    python -m twc_transfer tier0 --bank $ETROC_TWC_OUT/bank/coeffs_mar_f1_run12.parquet --tag f1_run12
    python -m twc_transfer cell --cell jul_51to48 --mode a --campaign jul --run h1_run48 \
        --combo dut1-ref2-trig3 --src-run h1_run51 --ntrip 5 --lpo --lpo-fallback own --dry-run
    python -m twc_transfer agg --cells 'cells_guardfix/J?_h1_*/*.csv' --f-inflate 0.9259 \
        --out $ETROC_TWC_OUT/tables/agg_J_jul_guardfix_f0p9259
    python -m twc_transfer jobs --list cells.tsv --dry-run
    python -m twc_transfer jobs --list cells_guardfix_rerun.tsv --write-dag dag_guardfix --retry 3
    python -m twc_transfer master
    python -m twc_transfer axes
    python -m twc_transfer axes --guardfix
    mkdir -p $ETROC_TWC_OUT/condor && cp twc_transfer/jobs/*.tsv $ETROC_TWC_OUT/condor/
    python -m etroc_plots.universality.reduce --out $ETROC_TWC_OUT/univ \
        --results-root $ETROC_TWC_OUT --part axes --part cells

`index` lists a run's track files with their event counts (`bank/index_<campaign>_<run>.parquet`),
`fit` (or `bank`) writes the coefficient bank of a run (`bank/coeffs_<campaign>_<run>.parquet`;
the line above is how the study built its banks), and `tier0` screens one bank file
(`tier0/transfer_<tag>.parquet`); a relative `--bank` is read from the read root, so
`--bank bank/coeffs_desy_run13.parquet` screens the study's own bank of that run. `cell`
evaluates one transfer cell, every arm on every target triple, into `cells/<cell>.csv`; the
line above prints its plan (the study's smoke job of `cells_guardfix_rerun.tsv` under another
name: five July H1 triples of run 48 corrected with run 51's coefficients), and without
`--dry-run` it runs in about two minutes. `--src-bank` takes a source's coefficients from a bank file instead
of refitting them (modes a, b, c, cb and f). A cell that stops can be started again with the
same command: the target triples already in its table are skipped. A target that is its own
source (`self_transfer` 1) gets one log line naming the arms of its rows that are in-sample:
`xfer_1b:<board>` and `xfer_lin`, which take the source's fit on all its events (Limits).
`agg` aggregates a group of
cell tables into `delta_by_triple.csv`, `agg_by_cell.csv` and `calibration.csv` (default
`$ETROC_TWC_OUT/tables/`); a relative `--cells` pattern is taken in the read root, so the line
above re-aggregates the four cells of the study's group `J_jul_guardfix` as its driver did.
`--f-cal jul` gives the same f from `campaigns.F_CAL` (with
`--f-precision full`, the bootstrap's full value) and records that in `f_source`; the study
passed f with `--f-inflate`, so only that gives its tables back byte for byte. Without either,
`agg` takes the f of the read root's `tables/f_cal.json` only when that file names its
campaign (`"campaign"`) and every cell aggregated is of that campaign; it refuses a file that
names none or another, and the study's file (the DESY May run26 bootstrap) names none, so in
the study's tree `agg` needs `--f-cal` or `--f-inflate`. `jobs --dry-run`
prints every command of a job list with its roots filled in and checks that it parses;
`jobs --write-dag` writes the condor DAG of a list and submits nothing (Condor, below).

`master` runs round 1's chain (recon, verdicts, recompute, levels) from its seven aggregation
tables, read as they are, and its cell tables, into `$ETROC_TWC_OUT/master/`:
`master_cells.csv`, the table of the level cells, `level_verdicts.csv` and thirteen intermediate
files (about 10 s). `axes` writes the axis tables of stages 1 to 3 from the aggregation tables
behind them, `results_r2/axis_<X>.csv` and `results_r3/axis_F.csv`, `axis_F_long.csv`,
`summary_F.md`; `axes --guardfix` the guardfix tables of axes E, J and K,
`results_r3/guardfix/axis_<X>_gf.csv` (seconds each). The last two commands need
`etroc_plots.universality`, the universality notebook's package, which this tree does not
hold yet: its reduction composes these tables into the ones the notebook reads, and it takes
the job lists from `<results root>/condor/`, hence the copy. `SPEC.md` sections 3 and 6 to 10
list every mode, every option, every rule of the aggregation and the chain, and every output
column.

What refuses to run, and why:

- A writing subcommand (`index`, `bank`/`fit`, `tier0`, `cell`/`transfer`, `agg`, `master`,
  `axes`) without `ETROC_TWC_OUT`: there is no default output folder, so nothing lands in the
  study's own tree by accident.
- `ETROC_TWC_OUT` that is, or lies inside, the read root (`ETROC_TWC_TRANSFER_ROOT`), one of
  the study's output trees (`campaigns.JOB_LISTS`, the seeded rerun's included), a campaign's
  step-10 root or this checkout, compared by real path (`/eos/user/m/<user>` and
  `/eos/home-m/<user>` are one folder): outputs would mix with the inputs.
- An output path inside one of those folders, `--out` included, on every writing subcommand;
  nothing is made or removed first. A relative `--out` is taken in the write root, never in
  the working folder.
- In a job list, `${TT_OUT}` under a flag that is neither an input nor `--out`, an unknown
  placeholder, or a write root inside the read root.
- A cell whose flags would change its meaning without saying so (SPEC.md 3): a source board that
  is not the target's board without a `--role-map`, `--role-map` in modes e and inrun, mode cb
  without both `--src-role` and `--xfer-board`, `--xfer-lin` on a one-board cell or without
  `--arms ext`, `--src-parity-contrast` with a source in another run.
- Resuming a cell table whose columns are not `cell.CELL_COLUMNS`.
- An aggregation whose k_cal pool of a one-board pair holds a row identical to the baseline
  (SPEC.md 8, rule 7): it would pull the error scale down; nothing is written.
- In the round-1 chain, an aggregation table that is not one of the seven of round 1
  (`campaigns.ROUND1_AGG_TAGS`).
- An axis-F cell table whose labels do not name the level-F cells of `cells_stage3.tsv` one to
  one, and `axes --guardfix` with an axis other than E, J or K.
- `jobs --write-dag` into a folder outside the write root or inside one of those folders, or
  one that holds the DAG already, on a list without exactly one smoke row, with a job id named
  SMOKE, PREP or AGG on another row, or with a group that names no campaign (its cells would be
  aggregated with another campaign's f).

`python -m twc_transfer` pins the thread pools to one thread and, if `PYTHONHASHSEED` is not
0, restarts itself once with `PYTHONHASHSEED=0` and says so: the placebo of a cell whose source
is refitted from events, and of the in-run split, is seeded from Python's `hash()` of strings,
which is salted per process otherwise. Those seeds reproduce the study's cells only under
LCG_104d (Python 3.9.12) with `PYTHONHASHSEED=0`; under another Python they are not verified.

### Chaining on your own outputs

Every command writes into the write root (`ETROC_TWC_OUT`) and takes the study's tables from
the read root (`ETROC_TWC_TRANSFER_ROOT`, by default the study's tree): a relative `--cells`,
`--bank`, `--src-bank` or `--tier0` is taken in the read root, and `master` and `axes` read only
the read root. A chain of steps on your own
outputs therefore points the read root of each step at the write root of the step before it,
and writes each step into a new write root (one may not lie inside the read root). For example,
the guardfix table of axis J from your own aggregation of the study's four J cells:

    export ETROC_TWC_OUT=/path/to/step1
    python -m twc_transfer agg --cells 'cells_guardfix/J?_h1_*/*.csv' --f-inflate 0.9259 \
        --out tables/agg_J_jul_guardfix_f0p9259
    export ETROC_TWC_TRANSFER_ROOT=/path/to/step1 ETROC_TWC_OUT=/path/to/step2
    python -m twc_transfer axes --guardfix --axis J

The first step reads the study's cells and writes `tables/agg_J_jul_guardfix_f0p9259/` under
step 1; the second reads that folder and writes `results_r3/guardfix/axis_J_gf.csv` under step 2
(byte for byte the study's, md5 `7e4758c43efdd7d51f079c06f9ed8681`). Cells of your own enter
the same way: write them under one write root, then aggregate them with that root as the read
root, or with absolute `--cells` patterns.

## How etroc_plots reads these

The universality notebook's reduction (`etroc_plots.universality.reduce`, last line of the
recipe) composes this package's tables into the ones the notebook reads:

- `--results-root` is the write root, `$ETROC_TWC_OUT`: `results_r2/axis_<A..K>.csv`,
  `results_r3/axis_F.csv`, `axis_F_long.csv`, `summary_F.md`,
  `results_r3/guardfix/axis_<E,J,K>_gf.csv`, and the seven job lists in `condor/`, which are
  byte for byte `twc_transfer/jobs/*.tsv` (hence the copy in the recipe);
- `--twc-root` (default the read root) still supplies `tables/agg_<tag>/agg_by_cell.csv`, for
  the self_transfer flag of the axis rows, and `bank/`;
- the round-1 table of the level cells: `master` writes it to
  `$ETROC_TWC_OUT/master/master_cells.csv`, with the md5 of the copy the reduction takes from
  the read root.

## Job lists

`jobs/` holds the seven job lists the study's cells ran from, in the order of
`campaigns.JOB_LISTS` (a later list that names a cell again redefines it), each with the
output root it wrote into. They are the study's tables with their comment lines dropped and
the campaign roots replaced by `${ETROC_TWC_ROOT_<CAMPAIGN>}`; `${TT_OUT}` stays as it was.
Expanded with the default roots they give back the study's commands byte for byte (a test
checks the digests). 247 rows: 236 cell jobs (6 of them smoke tests) and 11 Tier-0 jobs.

Run each list into its own write root. The lists name the same outputs under `${TT_OUT}`, and
every cell row passes `--overwrite` (236 of 247, all but the Tier-0 rows): the 26 rows of `cells_rerun.tsv` write the
tables of 8 rows of `cells_stage1.tsv` (`cells/<cell>/<cell>[_cNofM].csv` of E12, E13, the two
chunks of I2, J1 to J4) and of 18 rows of `cells_stage2.tsv` (the 11 chunks of A4, the 3 of A6,
B6, C7, K3 and the smoke cell `smoke_B1_same_26to3_n3`), with the same job ids; the 6 rows of
`cells_stage2_held.tsv` write the chunk tables of A8 and A9 that `cells_stage3.tsv` writes
again. The study ran the seeded rerun in a tree of its own for that reason
(`campaigns.JOB_LISTS`). Every list but `cells_stage2_held.tsv` has a smoke row named SMOKE.

## Condor

`jobs --write-dag DIR` turns a job list into an HTCondor DAG, as the study's DAGs were laid
out, and submits nothing:

    python -m twc_transfer jobs --list cells_guardfix_rerun.tsv --write-dag dag_guardfix --retry 3

A relative DIR is taken in the write root; an absolute one must lie inside it. DIR must be new
or empty: one DAG folder per list, and one write root per list (Job lists). It receives:

- the job table, under the list's name, with every `${TT_OUT}` and `${ETROC_TWC_ROOT_<CAMPAIGN>}`
  filled in as `jobs --dry-run` prints it: the read root under the input flags (`--bank`,
  `--src-bank`, `--root`, `--src-root`, `--cells`, `--tier0`), the write root under `--out`;
- `<dag>.dag`: SMOKE, then PREP, then one node per other row of the table (chunk rows are
  separate nodes of one cell), then AGG. The name is `transfer_` and the list's name without
  `cells_` (`--dag-name` changes it); `--retry` sets every node's RETRY (the study's DAG had 3 for
  the guardfix rerun, 1 is the default) and `--no-prep-node` leaves PREP out;
- `sub/[<dag>_]{smoke,prep,cell,agg}.sub`: el9, one CPU, 2 GB of disk; a cell node takes its
  row's memory and the flavour longlunch, or workday when its estimate exceeds 90 min or it
  rotates five or more source pixels (`--src-k` 5 or more); smoke and PREP run as
  microcentury, AGG as longlunch;
- `wrapper.sh`, the executable of every node. In order, each with its own exit code, it checks
  the job id (90), that the package can be read (91, AFS), that the package's sha256 is the one
  the DAG was written for (92: every node of a DAG runs one version), that the sha1 of the
  core files is the one recorded when the DAG was written (98: one core per DAG), the job
  table and the job's row in it (93, 94), that the write root is writable from the worker
  (95), the LCG
  environment (96) and `TestBeam/` (97). It then exports `PYTHONHASHSEED=0`, the six thread
  variables at 1 and `PYTHONUNBUFFERED=1`, drops `--overwrite` on a retry so that a cell resumes
  its table, and runs `nice -n 19 python3 -m twc_transfer <command>`, logging to
  `$ETROC_TWC_OUT/cells/<cell>/<job>.log`. `wrapper.sh --help` prints the list;
- `manifest[_<dag>].txt`: the package sha, the core and whether it is the core of the study's
  cells, the roots, the counts, the CPU estimate and the aggregations of the AGG node;
- `logs/`, for condor's output, error and event logs.

The PREP node is `jobs --list <table> --check-inputs`: it fails when a bank file, a run's track
folder or a combo the commands name does not exist. The study's PREP node also computed the
f_cal bootstraps and built banks; here f is a campaign fact (`campaigns.F_CAL`) and the banks
are read from the read root. The AGG node is `jobs --list <table> --agg`: per group of the
table (its group column; smoke and level-f rows excluded), one `agg` per f of the group's
campaign (`campaigns.AGG_F`), over the tables in the folder of each job's `--out`, into
`$ETROC_TWC_OUT/tables/agg_<group>_<f suffix>/`; a cell with no table yet is listed and left
out. Both also run by hand, with the same commands.

Before submitting, on lxplus:

1. Read the summary `--write-dag` prints and the manifest: jobs per level, the CPU estimate,
   `inputs: N named, 0 missing`, and the core line. With this repository's core the cells
   rerun and do not reproduce the study's numbers (Versions); export `ETROC_TWC_CORE` before
   writing the DAG to run another core, and the wrapper carries it.
2. Keep the write root, and with it the DAG folder and the condor logs, on AFS (a work area):
   CERN's batch service does not take a submit folder or job logs on EOS.
3. Run the smoke node by hand, then read its log; the DAG runs it first as well:

       bash $ETROC_TWC_OUT/dag_guardfix/wrapper.sh --help
       bash $ETROC_TWC_OUT/dag_guardfix/wrapper.sh SMOKE 0

4. Leave the package and the core alone while the DAG runs: a node exits 92 when a module,
   98 when a core file, changed after the DAG was written. Write the DAG again, into a new
   folder, after any change.

Then submit from the DAG folder by hand, with the `-maxjobs` of the manifest:

    cd $ETROC_TWC_OUT/dag_guardfix && condor_submit_dag -maxjobs 12 transfer_guardfix_rerun.dag

and follow it in `transfer_guardfix_rerun.dag.dagman.out`, `logs/` and the cell logs rather
than by querying the scheduler in a loop.

Compared with the study's DAG of the same list, with the default roots: the same 19 nodes in
the same order, the same VARS keys and values on the 16 cell nodes (cell id, memory, flavour),
the same PARENT lines, and the same wrapper steps in the same order (exit codes 90 to 96, the
seed, the threads, the retry rule, 97, the run), to which this wrapper adds the core check (98)
after 92. The 17 commands are the study's byte for byte
once the `--out` root, here the write root, is mapped back to the study's tree. What differs:
RETRY is 1 unless `--retry 3` is given, as the study's reruns had it; the submit files carry no
`environment` line, since the roots are written into the wrapper; the batch names start with
`twc_transfer_` rather than `tt_transfer_`; and PREP and AGG run the wrapper (the input check,
the aggregation plan) where the study ran `prep.sh` (bootstraps and banks, workday, 4000 MB)
and `agg_guardfix.sh`, so PREP is microcentury with 2000 MB. The wrapper also checks the
package's sha unconditionally, where the study's skipped the check when no sha was given.

## Tests

    cd TestBeam && python -m pytest twc_transfer/tests -q

Synthetic data only: the tests read no data file but the package's own job lists and the toy
track files and cell tables the tests write under pytest's temporary folder; they touch nothing under
`/eos` and need no network.

## Parity with the study's toolkit

The port was checked against the toolkit the study ran, on the largest track file of mar
`h1_run1` `extra0-ref1-dut2` (6000 events), with `PYTHONHASHSEED=0`: naming, the triple's
arrays and event keys, the leave-pair-out source events, 60 bank rows (`folds="both"`, full
and leave-pair-out guards, placebo), the self maps (both `lpo_fallback` rules), a bank-backed
transfer and every variant built from it, `evaluate_arm` on 16 arms, the leaky control,
Tier-0 with and without residual moments, the overlap metrics and every statistic.

- With `ETROC_TWC_CORE` set to a core of the hashes in `campaigns.CELLS_CORE_SHA1`, all 2367
  compared items are bit for bit equal (bank frame compared with `check_exact=True`); the
  bank's `ts` and `git_sha` columns are excluded.
- With this repository's core, quantities that do not go through the solve stay bit for bit
  equal; coefficients move by at most 3.8e-12 relative; most widths agree to 1e-12 ps, but a
  few move by the GMM restart jitter, up to 2.85 ps on one placebo pair (see below).

The cell was checked on two cells of the study, against their tables, with `PYTHONHASHSEED=0`:

- The smoke cell of `cells_guardfix_rerun.tsv` (mode a, five triples, `--lpo --lpo-fallback
  own`), with a core of the hashes below: all 240 rows and 50 columns equal the study's table
  as text, except `ts`, and the side-car is equal. With this repository's core, 165 of the 240
  `value` entries move, by up to 3.85 ps, while `rms`, `iqr` and the overlap columns move by
  less than 3e-12: the GMM restart jitter of a last-bit change.
- One triple (`--triples`) of the stage-3 P3 cell `G_P3_run26_n100`: its 54 rows equal a rerun of
  the study's stage-3 toolkit on the same node except `ts`. Both differ from the study's
  table, which was made on a batch node, in `ks_d` (13 entries, at most 2e-16) and `ks_p` (12
  entries, at most 1.5e-14).

The aggregation was checked by re-aggregating groups of the study's cell tables, read from the
read root, with the f its driver passed (`--f-inflate`), and comparing the three files with the
study's `tables/agg_<tag>/`:

- `J_jul_guardfix_f0p9259` (four cells with side-cars): `delta_by_triple.csv` (14280 rows, 63
  columns), `agg_by_cell.csv` (214 rows, 47 columns, `lpo_shape` included) and
  `calibration.csv` (56 rows) are byte for byte the study's. The study's guardfix toolkit, run
  on the same node on the same files, gives the same three files.
- Stage 1 `J_jul_f0p9259` (7 cells; 23316, 369 and 98 rows) and `I_mar_f0p9524` (99000, 393, 90
  rows), stage 3 `G_desy_f0p8153` (21330, 434, 128 rows): byte for byte the study's.
- The AGG node's plan (`jobs --agg`, run on the study's cell tables into a folder of its own)
  over `cells_stage1.tsv` (7 tags), `cells_stage2.tsv` (8), `cells_stage3.tsv` (8),
  `cells_rerun.tsv` (14, in the seeded rerun's tree) and `cells_guardfix_rerun.tsv` (4): all
  123 files byte for byte the study's. The stage-2 tags of group `A_desy` equal the study's
  `agg_A_r2_desy_*`, whose aggregation has no recorded code version.
- Round 1 is re-derived, not reproduced (Versions, below). On `agg_mar_f0p9524` the rows and
  every shared column are equal; the port adds `placebo_minus_notwc1b_ps` to `agg_by_cell.csv`
  and the seven protocol columns, empty, to `delta_by_triple.csv`, since the round-1 cell tables
  have 43 columns. On `agg_desy_f0p8153` the k_cal exclusions (215 same-pixel and 234
  self-transfer rows, absent in round 1) change `k_cal` in 30 of 302 groups, by up to 15 %, and
  with it `mu`, `se` and the interval: `tost` changes on 2 rows and `fail_holm` on 1.

The round-1 chain and the axis tables were checked against the study's tables, run from the
read root on lxplus:

- `master`: `master_cells.csv` (43 cells, 105 columns) and the other thirteen CSV tables are
  byte for byte the study's. `recon.json` has the same entries less `prereg_coverage_cells`,
  which the study took from its pre-registration notes rather than from a table. The
  self-check joins 2143 rows, with differences up to 4.3e-5 ps in mu and 5.9e-5 ps in se and 3
  TOST verdicts that differ, as in the study's own run.
- `axes`: the eight tables of `results_r2/`, `axis_F_long.csv` and, with `--guardfix`, the
  three guardfix tables are byte for byte the study's (`campaigns.STUDY_TABLE_MD5`).
  `axis_F.csv` is the study's table corrected on 12 rows (Versions): its other 155 rows and
  every column of the study's are equal as text, and it adds the five `<arm>_self_transfer`
  columns. The cell table of `summary_F.md` is line for line the study's; the rest of the
  study's file (its hypothesis, verdict tables and anomalies) is prose and not produced.
- Through the universality reduction (the recipe above), run on a write root whose
  `axis_F.csv` was still the study's table, `univ_axis_r2.csv` (6011 rows),
  `univ_axis_F_long.csv`, `univ_axis_F_wide.csv` and `univ_cells.csv` are byte for byte the
  published tables (`campaigns.PUBLISHED_MD5`), and `master_cells.csv` is byte for byte
  `univ_master_cells.csv`, which the reduction's master part copies from the study's tree. The
  reduction has not been run on the corrected `axis_F.csv`.

## Versions

The study's published cells imported a pipeline core of these sha1 (`campaigns.CELLS_CORE_SHA1`):

    twc_solver.py  da6caabffece62bb7a01bbf685bcea22dd45de0b
    bootstrap.py   0e0b891ec6261749760955895c9b36655014c58e
    io_utils.py    a737972dc0adea61c94c9da39f875be95eedff32

This repository's core differs in two places. `twc_solver.py` streams its least-squares design
through a QR factor in blocks of events instead of one dense `lstsq`, which changes the
coefficients in their last bits; `bootstrap.py` differs only in the default of its
`--minimum_nevt` option, which this package does not use. Because `width_gmm` seeds its fit
from the bytes of the residuals, a last-bit change in a coefficient can move a width by the
fit's restart jitter rather than by the float difference. To reproduce a published number,
point `ETROC_TWC_CORE` at a core of the hashes above; `_core.core_versions()` reports what was
loaded, and the manifest of a DAG says which core its cells run on.

The core of these hashes is the one of commit `ac97d8d6` (2026-08-24) of this repository. A
detached worktree of that commit gives it without touching your checkout:

    git worktree add --detach ../ETROC-Analysis-core-ac97d8d6 ac97d8d6
    export ETROC_TWC_CORE=$PWD/../ETROC-Analysis-core-ac97d8d6/TestBeam/condor_at_lxplus/core
    sha1sum $ETROC_TWC_CORE/twc_solver.py $ETROC_TWC_CORE/bootstrap.py $ETROC_TWC_CORE/io_utils.py

(from the repository's top folder; on macOS `shasum -a 1`). The three sums must be those of
`campaigns.CELLS_CORE_SHA1` above. Run the package from your own checkout as before; only the
core comes from the worktree.

What the repository's core changes, measured on one node with nothing but the core changed:

- one triple of mar `h1_run1` (6000 events, 16 arms): the coefficients by at most 3.8e-12
  relative and the widths of 10 arms by at most 5e-13 ps, but the placebo pair's width by
  2.85 ps (4.6 %) and its board sigma by 2.12 ps;
- the smoke cell of `cells_guardfix_rerun.tsv` (240 rows): 165 `value` entries move, by up to
  3.85 ps, while `rms`, `iqr` and the overlap columns move by less than 3e-12;
- one triple of the stage-3 cell F2 (mode d, 390 rows): 368 `value` entries move, by up to
  0.481 ps, and the smooth columns by at most 1e-11;
- the aggregation fits no width: its tables are byte for byte the same with either core
  (`J_jul_guardfix_f0p9259`).

The archived core reproduces the study's cells because it gives the residuals the study's
bytes, so `width_gmm` takes the study's restarts; the repository's core reruns them, with
differences of the size of the restart jitter. Neither is wrong; only the first gives back a
published table.

The study ran in waves, each from its own job list and cell code (`campaigns.WAVES`: the
sha256 prefix of its `tt.py`, all on the core above). The first table says what this package
does with each wave's cells and aggregation tables and which published tables the wave feeds;
the second lists every table this package gives back byte for byte, and `axis_F.csv`, which it
corrects (below), with its md5 and the command that writes it (`campaigns.TABLE_VERSIONS`). Both are rendered by
`campaigns.versions_markdown()`, and a test checks that they are the ones below.

| wave | job list | tt.py | cells here | aggregation tags | aggregation here | feeds |
|---|---|---|---|---|---|---|
| round 1 | `cells.tsv` | 258cb1f8 | reruns them (older cell, older columns) | agg_desy_*, agg_aug_*, agg_mar_f0p9524, agg_jul_f0p9259 | re-derives them (later rules, more columns): agg_desy_f0p8153 changes tost on 2 rows; master reads them as they are | `univ_master_cells.csv`, `univ_cells.csv` |
| stage 1 | `cells_stage1.tsv` | 05793488 | reruns them (older cell) | the groups of cells_stage1.tsv: axes D, E, I, J | reproduces them (all 7 tags compared) | `univ_axis_r2.csv`, `univ_cells.csv` |
| stage 2 | `cells_stage2.tsv` | 05793488 | reruns them (older cell) | the groups of cells_stage2.tsv: axes A, B, C, K | reproduces them (all 8 tags compared; its A_desy tags are the study's agg_A_r2_desy_*, since stage 3 reused the folder name) | `univ_axis_r2.csv`, `univ_cells.csv` |
| stage 3 | `cells_stage3.tsv` | 698f72f9 | reproduces them up to the cross-node tolerance (ks_d, ks_p within 3.3e-14; one triple per cell in modes c, d, e and xfer-lin compared) | the groups of cells_stage3.tsv: A8 and A9, F, G | reproduces them (all 8 tags compared) | `univ_axis_F_long.csv`, `univ_axis_F_wide.csv`, `univ_cells.csv` |
| seeded rerun | `cells_rerun.tsv` | 698f72f9 | not compared (the stage-3 code) | the groups of cells_rerun.tsv, in the rerun's own tree | reproduces them (all 14 tags compared) | `univ_cells.csv` |
| guardfix | `cells_guardfix_rerun.tsv` | e83f71c1 | reproduces them (the smoke cell compared, 240 rows) | agg_E_jul_guardfix_*, agg_J_jul_guardfix_*, agg_K_desy_guardfix_* | reproduces them (all 4 tags compared) | `univ_axis_r2.csv`, `univ_cells.csv` |
| v2 wave 1 | `cells_v2w1.tsv` | e83f71c1 | not compared (the guardfix code); the list is not in jobs/ | agg_V2_* | the guardfix code, not compared | none |

| table | md5 | command | waves (tt.py) | made from |
|---|---|---|---|---|
| `master/master_cells.csv` | `62cfc93f96be1dcbab6cad770da772fa` | `python -m twc_transfer master` | round 1 (258cb1f8) | the seven round-1 aggregation tables, read as they are, and the round-1 cell tables |
| `results_r2/axis_A.csv` | `f54ef48af42e2379c1fe8017f08466ef` | `python -m twc_transfer axes` | stage 2 (05793488); the aggregation's code version is not recorded | agg_A_r2_desy_*, the stage-2 cells of axis A aggregated again (the plan of `jobs --agg` on cells_stage2.tsv gives them back) |
| `results_r2/axis_B.csv` | `5103089c07b8d2b6b2551bfc2f90c61a` | `python -m twc_transfer axes` | stage 2 (05793488) | agg_B_desy_f0p8153, agg_B_desy_f1p0 |
| `results_r2/axis_C.csv` | `332b93b9bcfd22dbe057d38adfda2eed` | `python -m twc_transfer axes` | stage 2 (05793488) | agg_C_desy_f0p8153, agg_C_desy_f1p0 |
| `results_r2/axis_D.csv` | `1cf27fe9d4e66aa94479d159a986d4b3` | `python -m twc_transfer axes` | stage 1 (05793488) | agg_D_mar_f0p9524 |
| `results_r2/axis_E.csv` | `b8d9c6fe9a585f054cf85670aade857e` | `python -m twc_transfer axes` | stage 1 (05793488) | agg_E_aug_fmeas, agg_E_aug_f0p8153, agg_E_aug_f1p0, agg_E_jul_f0p9259 |
| `results_r2/axis_I.csv` | `4656d575599dfb9b7bcc68b21ef1a0e0` | `python -m twc_transfer axes` | stage 1 (05793488) | agg_I_mar_f0p9524 |
| `results_r2/axis_J.csv` | `281926f5e3c313fb0d08e55859de78e7` | `python -m twc_transfer axes` | stage 1 (05793488) | agg_J_jul_f0p9259 |
| `results_r2/axis_K.csv` | `d5b4825caf1f5483af1a69689759e9f3` | `python -m twc_transfer axes` | stage 2 (05793488) | agg_K_desy_f0p8153, agg_K_desy_f1p0 |
| `results_r3/axis_F.csv` | `d5425a11ff92f69c4528c57f5936ff0f` | `python -m twc_transfer axes` | stage 3 (698f72f9) | agg_F_* (the four tags of AXIS_TAGS F); the study's table, md5 44c661a62fdbf95b05dd7d67cd86cb74 as published (12 keys carry the self-transfer row), differs on those 12 rows and lacks the five <arm>_self_transfer columns |
| `results_r3/axis_F_long.csv` | `8cead873b7cc767f21b356cbd74bd136` | `python -m twc_transfer axes` | stage 3 (698f72f9) | agg_F_* (the four tags of AXIS_TAGS F) |
| `results_r3/guardfix/axis_E_gf.csv` | `2f0a77c5a5c58ffca95bb7fa2c24645f` | `python -m twc_transfer axes --guardfix` | guardfix (e83f71c1) | agg_E_jul_guardfix_f0p9259 |
| `results_r3/guardfix/axis_J_gf.csv` | `7e4758c43efdd7d51f079c06f9ed8681` | `python -m twc_transfer axes --guardfix` | guardfix (e83f71c1) | agg_J_jul_guardfix_f0p9259 |
| `results_r3/guardfix/axis_K_gf.csv` | `db411e3e2daa8a45eeb0dafb84a55361` | `python -m twc_transfer axes --guardfix` | guardfix (e83f71c1) | agg_K_desy_guardfix_f0p8153, agg_K_desy_guardfix_f1p0 |
| `univ_master_cells.csv` | `62cfc93f96be1dcbab6cad770da772fa` | `python -m etroc_plots.universality.reduce --part master` | round 1 (258cb1f8) | a copy of master_cells.csv |
| `univ_axis_r2.csv` | `15c688966ce406593454391e9d6120ff` | `python -m etroc_plots.universality.reduce --part axes` | stage 1 (05793488), stage 2 (05793488), guardfix (e83f71c1) | the eight results_r2 axis tables, the guardfix rows replacing those of their 11 cells, and the self_transfer flag from the aggregation tables |
| `univ_axis_F_long.csv` | `51302099b643b24014141af7d860a059` | `python -m etroc_plots.universality.reduce --part axes` | stage 3 (698f72f9) | results_r3/axis_F_long.csv and the self_transfer flag |
| `univ_axis_F_wide.csv` | `911efb7d856178f24e60a87b5203a755` | `python -m etroc_plots.universality.reduce --part axes` | stage 3 (698f72f9) | the study's results_r3/axis_F.csv, md5 44c661a62fdbf95b05dd7d67cd86cb74 as published (12 keys carry the self-transfer row), and the self_transfer flag |
| `univ_cells.csv` | `2fcde80b93bff79a4d9eb54292546318` | `python -m etroc_plots.universality.reduce --part cells` | round 1 (258cb1f8), stage 1 (05793488), stage 2 (05793488), stage 3 (698f72f9), seeded rerun (698f72f9), guardfix (e83f71c1) | the seven job lists and the cell table of summary_F.md |

"Reproduces" means byte for byte for an aggregation or a derived table, and for a cell the
same table except `ts` when its command runs on a core of the hashes above, on the node the
compared table was made on; "up to the cross-node tolerance" adds that on another node `ks_d`
and `ks_p` differ too, by at most 3.3e-14 (6.1e-14 relative; the largest of the stage-3
comparisons, Acceptance evidence); "reruns" means the
command runs but computes with the current cell; "not compared" means the code is the same as
a compared wave's and no table of that wave was run against the study's. `cell.py` is the
guardfix version; the stage-3 version differs from it only by `--lpo-fallback`, whose default
is the stage-3 behaviour. `agg.py` is the guardfix version too; stages 1 to 3 lack only its
merge of the `lpo_shape` side-cars, which acts only on cells that have one. Round 1 lacks the
one-board identical-pair drop, the k_cal exclusions, the structural-zero stop, the `notwc_1b`
retention gain, the `xfer_lpo` primary rule and the column `placebo_minus_notwc1b_ps`: its
tables are read as they are, and re-aggregating its cells changes the DESY May numbers
(Parity, above), so `master` run on such tables would not give the study's
`master_cells.csv`. Which cells form a group, and the f of each tag, were the study's drivers'
choice, which `jobs --agg` follows (`campaigns.AGG_F`). Axis A is byte for byte the study's once
its rows carry the label the study gave them, `A_desy_*`, rather than the name of the folder
they are read from (`campaigns.AXIS_TAGS`). Neither `master` nor `axes` reads
`calibration.csv`. The tags `agg_dt_*` and `agg_V2_*` were not compared.

`axis_F.csv` is corrected. A cell whose source triple is one of its own targets has, under
one key of the wide table (cell, kind, pair or board, family, tag, f), two rows of an arm: the
transfer row (`self_transfer` 0) and the self-transfer row (1: the target is its own source),
which the aggregation keeps apart (SPEC 8). The study's wide table kept
whichever came last; on 12 keys of `F_desy_f0p8153` (cells F2 and F3; the arms `xfer`,
`xfer_lin`, `xfer_clip` and `xfer_rescale`, 48 entries) that was the self-transfer row: F3
`trig-dut` extrapolating `xfer` reads n_pix 5, mu 0.000 ps, PASS where the transfer row is
n_pix 69, mu 5.32 ps, FAIL. `axes` now shows the transfer row of an arm when the key has one,
else its self-transfer row, and says which in `<arm>_self_transfer`. The study's table is
`campaigns.AXIS_F_AS_PUBLISHED` (md5 `44c661a62fdbf95b05dd7d67cd86cb74`, as published: 12 keys
carry the self-transfer row); `univ_axis_F_wide.csv` was composed from it.

## Acceptance evidence

What was run against what, on lxplus under LCG_104d with `PYTHONHASHSEED=0`:

| part | compared against | counts | result |
|---|---|---|---|
| library (naming, events, bank, arms, widths, Tier-0, statistics) | the study's toolkit on one triple, archived core | 2367 items | bit for bit |
| job lists | the study's seven lists, expanded with the default roots | 7 lists, 247 rows | byte for byte; 247 of 247 parse |
| cell, mode a | the smoke cell of `cells_guardfix_rerun.tsv`, archived core | 240 rows x 50 columns, 5 triples | equal except `ts`; side-car equal |
| cell, modes c, d, e, xfer-lin | one triple each of G_P3_run26_n100, F2, F7, F8; the stage-3 toolkit rerun on the same node | 54, 390, 108, 108 rows | equal to the rerun except `ts`; `ks_d`, `ks_p` differ from the batch node's table by at most 3.3e-14, as the rerun does |
| aggregation | every tag of stages 1 to 3, the seeded rerun (in its own tree) and the guardfix rerun, through the plan of `jobs --agg` on the study's cell tables | 41 tags x 3 files = 123 files | byte for byte; round 1 re-derived (`agg_desy_f0p8153`: 2 of its TOST verdicts change) |
| round-1 chain | `master_cells.csv` and 13 intermediate tables | 43 cells x 105 columns, 14 files | byte for byte |
| axis tables | `results_r2/` (8), `axis_F.csv`, `axis_F_long.csv`, the 3 guardfix tables | 13 tables | 12 byte for byte; `axis_F.csv` equal but for the 12 rows it corrects (48 arm entries) and its 5 added flag columns |
| universality reduction | the published tables, from axis tables whose `axis_F.csv` was the study's | 4 tables, and the master copy | byte for byte |
| condor driver | the study's DAG of `cells_guardfix_rerun.tsv` | 19 nodes, 16 VARS lines, 5 PARENT lines, 17 commands | the same nodes in the same order, VARS keys and values, PARENT lines and wrapper steps in order; 17 of 17 commands byte for byte once the `--out` root is mapped back (Condor) |

## Limits

- Modes b, cb, f and inrun of the cell were never run against the study's tables; nor was the
  in-run placebo's seed.
- The cell comparisons of modes c, d, e and xfer-lin took one triple per cell; mode a took the
  five triples of one smoke cell.
- The placebo seeds come from Python's `hash()` of strings under `PYTHONHASHSEED=0`; they are
  verified under Python 3.9.12 (LCG_104d) only.
- With the repository's core the cells rerun; only `ETROC_TWC_CORE` at the archived core
  reproduces them (Versions).
- Round 1's aggregation tables are re-derived by `agg`, not reproduced; `master` reads them as
  they are.
- The long axis tables (`results_r2/axis_<X>.csv`, `axis_F_long.csv`) keep the study's 24
  columns, without `self_transfer`. Where a cell's source triple is one of its targets a key
  holds two rows of an arm that only the flag tells apart: 208 rows of axis A (cells A4 to A7),
  12 of axis I (I3), 552 of axis F (F1, F2, F3). The flag is in the tag's `agg_by_cell.csv`,
  which the universality reduction joins; `axis_F.csv` carries it per arm.
- On a self-transfer target (`self_transfer` 1: the source triple is the target) the arms
  `xfer_1b:<board>` (`--arms ext`) and `xfer_lin` (`--xfer-lin`) evaluate the target's own fit
  on all its events, in-sample; only `xfer`, `placebo`, `xfer_lpo` and the variants of `xfer`
  take the fold-matched fits (SPEC 3). The cell names those arms in a log line per target and
  writes the rows with their flag; nothing else marks them.
- The f_cal bootstrap is not part of this package: f is a campaign fact (`campaigns.F_CAL`).
- The condor driver's DAG was compared with the study's by its files; no DAG written by this
  package has been submitted.
- `etroc_plots.universality`, which the last line of the recipe runs, is not on this branch.
