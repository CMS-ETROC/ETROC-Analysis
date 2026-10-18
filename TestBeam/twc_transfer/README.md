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
screen, the transfer cell, the statistics and the aggregation of cells. `SPEC.md` beside this file defines all of it.

## Environment variables

| variable | meaning | default |
|---|---|---|
| `ETROC_TWC_CORE` | the pipeline core directory to import `bootstrap` and `twc_solver` from | this repository's `TestBeam/condor_at_lxplus/core` |
| `ETROC_TWC_ROOT_<CAMPAIGN>` | the step-10 analysis root of a campaign (`MAR`, `JUL`, `DESY`, `AUG`); track files are `<root>/tracks_<run>/<combo>/time/track_*.parquet` | the campaign trees in `campaigns.py` |
| `ETROC_TWC_TRANSFER_ROOT` | the read root: the study's outputs (bank, cells, tables), never written; the same variable and default as `etroc_plots` | the study's output tree in `campaigns.py` |
| `ETROC_TWC_OUT` | the write root | none: `campaigns.write_root()` raises when it is unset, or when it is the read root or a data root |

`ETROC_TWC_CORE` is read once, at the first import of the package; the others are read when
`campaigns` is first imported (the roots) or when `transfer_root()` / `write_root()` is called.

## Modules

| module | what it holds |
|---|---|
| `campaigns` | step-10 roots per campaign, the read root, the write root, the event cap, the core sha1 of the study's cells, the cell versions |
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
| `cli`, `__main__` | the command line, `python -m twc_transfer` (below) |
| `jobs` | the job lists: reader, root expansion, `--dry-run`; the study's seven lists are in `jobs/` |

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
(and the Tier-0 screen of it), then cells, then their aggregation:

    export ETROC_TWC_OUT=/path/to/your/output
    python -m twc_transfer index --campaign mar --run h1_run1 --combo extra0-ref1-dut2
    python -m twc_transfer fit --campaign mar --run f1_run12 --ntrip 40 --folds both --placebo
    python -m twc_transfer tier0 --bank $ETROC_TWC_OUT/bank/coeffs_mar_f1_run12.parquet --tag f1_run12
    python -m twc_transfer cell --cell jul_51to48 --mode a --campaign jul --run h1_run48 \
        --combo dut1-ref2-trig3 --src-run h1_run51 --ntrip 5 --lpo --lpo-fallback own --dry-run
    python -m twc_transfer agg --cells 'cells_guardfix/J?_h1_*/*.csv' --f-inflate 0.9259 \
        --out $ETROC_TWC_OUT/tables/agg_J_jul_guardfix_f0p9259
    python -m twc_transfer jobs --list cells.tsv --dry-run

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
same command: the target triples already in its table are skipped. `agg` aggregates a group of
cell tables into `delta_by_triple.csv`, `agg_by_cell.csv` and `calibration.csv` (default
`$ETROC_TWC_OUT/tables/`); a relative `--cells` pattern is taken in the read root, so the line
above re-aggregates the four cells of the study's group `J_jul_guardfix` as its driver did.
`--f-cal jul` gives the same f from `campaigns.F_CAL` (with
`--f-precision full`, the bootstrap's full value) and records that in `f_source`; the study
passed f with `--f-inflate`, so only that gives its tables back byte for byte. `jobs --dry-run`
prints every command of a job list with its roots filled in and checks that it parses.
`SPEC.md` sections 3, 6, 7 and 8 list every mode, every option, every rule of the aggregation
and every output column.

What refuses to run, and why:

- A writing subcommand (`index`, `bank`/`fit`, `tier0`, `cell`/`transfer`, `agg`) without
  `ETROC_TWC_OUT`: there is no default output folder, so nothing lands in the study's own tree
  by accident.
- `ETROC_TWC_OUT` equal to the read root (`ETROC_TWC_TRANSFER_ROOT`) or to a campaign's
  step-10 root: outputs would mix with the inputs.
- In a job list, `${TT_OUT}` under a flag that is neither an input nor `--out`, an unknown
  placeholder, or a write root equal to the read root.
- A cell whose flags would change its meaning without saying so (SPEC.md 3): a source board that
  is not the target's board without a `--role-map`, `--role-map` in modes e and inrun, mode cb
  without both `--src-role` and `--xfer-board`, `--xfer-lin` on a one-board cell or without
  `--arms ext`, `--src-parity-contrast` with a source in another run.
- Resuming a cell table whose columns are not `cell.CELL_COLUMNS`.
- An aggregation whose k_cal pool of a one-board pair holds a row identical to the baseline
  (SPEC.md 8, rule 7): it would pull the error scale down; nothing is written.

`python -m twc_transfer` pins the thread pools to one thread and, if `PYTHONHASHSEED` is not
0, restarts itself once with `PYTHONHASHSEED=0` and says so: the placebo of a cell whose source
is refitted from events, and of the in-run split, is seeded from Python's `hash()` of strings,
which is salted per process otherwise. Those seeds reproduce the study's cells only under
LCG_104d (Python 3.9.12) with `PYTHONHASHSEED=0`; under another Python they are not verified.

## Job lists

`jobs/` holds the seven job lists the study's cells ran from, in the order of
`campaigns.JOB_LISTS` (a later list that names a cell again redefines it), each with the
output root it wrote into. They are the study's tables with their comment lines dropped and
the campaign roots replaced by `${ETROC_TWC_ROOT_<CAMPAIGN>}`; `${TT_OUT}` stays as it was.
Expanded with the default roots they give back the study's commands byte for byte (a test
checks the digests). 247 rows: 236 cell jobs (6 of them smoke tests) and 11 Tier-0 jobs.

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
- Round 1 is re-derived, not reproduced (Versions, below). On `agg_mar_f0p9524` the rows and
  every shared column are equal; the port adds `placebo_minus_notwc1b_ps` to `agg_by_cell.csv`
  and the seven protocol columns, empty, to `delta_by_triple.csv`, since the round-1 cell tables
  have 43 columns. On `agg_desy_f0p8153` the k_cal exclusions (215 same-pixel and 234
  self-transfer rows, absent in round 1) change `k_cal` in 30 of 302 groups, by up to 15 %, and
  with it `mu`, `se` and the interval: `tost` changes on 2 rows and `fail_holm` on 1.

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
loaded.

The study's cells ran under five versions of its cell code (`campaigns.CELL_VERSIONS`, the
sha256 prefix of its `tt.py`), all on the core above:

| wave | job list | tt.py | this package |
|---|---|---|---|
| round 1 | `cells.tsv` | 258cb1f8 | reruns it (older cell, older columns) |
| stage 1, stage 2 | `cells_stage1.tsv`, `cells_stage2.tsv` | 05793488 | reruns them (older cell) |
| stage 3, seeded rerun | `cells_stage3.tsv`, `cells_rerun.tsv` | 698f72f9 | reproduces them |
| guardfix, v2 wave 1 | `cells_guardfix_rerun.tsv`, `cells_v2w1.tsv` | e83f71c1 | reproduces them |

`cell.py` is the guardfix version; the stage-3 version differs from it only by
`--lpo-fallback`, whose default is the stage-3 behaviour. "Reproduces" means the same command
on a core of the hashes above gives the same table except `ts`; "reruns" means the command runs
but computes with the current cell. `cells_v2w1.tsv` is not among the lists in `jobs/`.

The study's aggregation tables (`tables/agg_<tag>/`, one per group and f) were written by four
versions of its `agg`:

| wave | tags | tt.py | this package |
|---|---|---|---|
| round 1 | `agg_desy_*`, `agg_aug_*`, `agg_mar_f0p9524`, `agg_jul_f0p9259` | 258cb1f8 | re-derives them (later rules, more columns) |
| stage 1, stage 2 | the groups of `cells_stage1.tsv`, `cells_stage2.tsv` | 05793488 | reproduces them |
| stage 3 | the groups of `cells_stage3.tsv` | 698f72f9 | reproduces them |
| guardfix | `agg_*_guardfix_*` | e83f71c1 | reproduces them |

`agg.py` is the guardfix version; stages 1 to 3 lack only its merge of the `lpo_shape`
side-cars, which acts only on cells that have one. Round 1 lacks the one-board identical-pair
drop, the k_cal exclusions, the structural-zero stop, the `notwc_1b` retention gain, the
`xfer_lpo` primary rule and the column `placebo_minus_notwc1b_ps`: its tables are read as they
are, and re-aggregating its cells changes the DESY May numbers (Parity, above). "Reproduces"
means byte for byte on the tags compared above. The tables of the seeded rerun's tree and the
tags `agg_A_r2_desy_*`, `agg_dt_*` and `agg_V2_*` were not compared. Which cells form a group,
and the f of each tag, were the study's drivers' choice (the group column of the job list; the
f named by the tag's suffix); `agg` takes the patterns and the f as arguments.
