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
screen and the statistics. `SPEC.md` beside this file defines all of it.

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
| `campaigns` | step-10 roots per campaign, the read root, the write root, the event cap, the core sha1 of the study's cells |
| `_core` | loads the pipeline core, exposes `BS` (bootstrap) and `TS` (twc_solver), `core_versions()` |
| `naming` | combos and roles, pixel triples, run directories, sibling combos, track ranking, `ensure_dirs(root)` |
| `events` | `load_events`, `Triple`, per-event identity (`event_keys`), the leakage asserts |
| `solve` | the joint solve, the gauge, the apply line, raw and centred bases, coefficient covariance, transfer variants |
| `bank` | `bank_rows_for_triple` and `BANK_COLUMNS`, TOT moments, leave-pair-out source events, TOT overlap metrics |
| `widths` | `width_gmm`, the three-board sigma, `evaluate_arm`, own-TOT slope, the leaky positive control |
| `arms` | `self_coeff_maps`, `transferred_map`, `xfer_lin_map`, `xfer_lin_lpo_map`, the arm names |
| `tier0` | `tier0_delta`, the analytic change of RMS^2 per pair and board |
| `stats` | `kcal_from_cells`, `random_effects`, `tost`, `margin_for`, `source_random_effect`, `holm`, `p_nonequiv` |

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

## Tests

    cd TestBeam && python -m pytest twc_transfer/tests -q

Synthetic data only: the tests read no file, touch nothing under `/eos` and need no network.

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

This commit holds the library. The command-line tools (bank, Tier-0 screen, cells,
aggregation, quote) are added by later commits.
