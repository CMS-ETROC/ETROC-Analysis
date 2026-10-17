# twc_transfer: specification of the library

What the package computes and the rules it keeps. Sections 0, 1, 2, 4 and 5 define the library
in this directory; sections 6 and 7 the commands that write the index, the bank and the Tier-0
screen, and the tables they write. Section 3 (the transfer operations and the two
pixel-to-board protocols) belongs to the cell command and is specified with it; section 3 below
holds only the terms of it that the library code uses. The code cites sections as "SPEC 1",
"SPEC 5.1" and so on.

## 0. Layout and invariants

The library is the ten modules listed in `README.md`. It imports the pipeline core unmodified
(`_core.py`; the directory is `ETROC_TWC_CORE` or this repository's
`condor_at_lxplus/core`):

- `twc_solver.solve_timewalk_coefficients`, the joint least-squares TWC of the pipeline (`TS`);
- from `bootstrap` (`BS`, logger at level 40): `fit_gmm_and_get_fwhm`,
  `calculate_resolution_from_fit`, `apply_neighbor_cut`, `KS_PMIN_DEFAULT`, `KS_DMAX_DEFAULT`.

The estimator is `BS.fit_gmm_and_get_fwhm` itself, with the regularisation it applies; no
re-implementation of the width is used anywhere, and `utils/extract_twc_coeffs.py` is not used.
The one new primitive is the pipeline's apply line, `toa_r + np.polyval(c_r, tot_r)`
(`solve.apply_c`). Two small functions of the study's TWC verification module are vendored
unchanged, with their origin stated: `naming.roles_of` and `widths.own_tot_slope`.

A step-10 track file (`time/track_*.parquet`) has the columns `tot_<role>`, `toa_<role>`,
`file`, `cal_<role>`, `HasNeighbor_<role>`, `trackNeighbor`, and no event id. A four-board run
has four leave-one-out combos; on March H1 each holds about 1090 to 1210 triples per run.

Nothing in the library writes a file. A caller that writes, the command line of section 6
included, uses `campaigns.write_root()` (the folder `ETROC_TWC_OUT`, which has no default and
may be neither the read root nor a data root) and `naming.ensure_dirs(root)`.

## 1. Data model

**The unit of work is one target triple:** one step-10 track file, whose pixel triple is
parsed from its name, in one (campaign, run, combo). `events.load_events` applies, in this
order, the pipeline's neighbour cut (`BS.apply_neighbor_cut` with the run's own arguments), a
drop of rows missing a TOT, a TOA or the file index, and a cap at `N_CAP = 6000` events drawn
with a fixed seed (original order kept). Every arm sees exactly these events.

**The frozen evaluation set E is every kept event of the target,** shared by all arms. Arms are
therefore N-matched by construction, and the component branch of `fit_gmm_and_get_fwhm` (1, 2
or 3 components below 1500 events, 3 at or above) is the same for every arm.

**Folds are `fold = file % 2`:** file-stratified, never contiguous. The per-file inter-board
common mode was measured at 0.7 to 6.6 ps rms, so contiguous halves would charge drift to the
self arm and make every transfer look better than it is. Contiguous file blocks
(`Triple.idx_block("A")`, `"B"`) are kept as a transfer cell, the in-run stability split of
section 3, not as a baseline.

**Fold-matched sourcing.** When the source triple is the target itself, the coefficients
applied to fold-k events come from a fit that excludes fold k. A source in another run, angle
or board is disjoint by construction. The hard invariant is that no event a source fit used is
evaluated:

- `events.event_keys` gives every event an identity: a 64-bit FNV-style hash over its file
  index and the exact bit pattern of the TOA of every role compared. Two different events of
  one run colliding on all of them has probability of order n^2 * 2^-64.
- `widths.evaluate_arm` calls `events.assert_keys_disjoint` on every arm whose source read
  events, and raises `LeakageError` on a single shared event. A leave-pair-out source lives in
  another combo, so it is compared on the two roles it shares with the target.
- A bank-backed source reads no events; it is checked structurally by `events.assert_disjoint`
  (same triple implies different folds), and the evaluation returns `n_leak_checks = 0`.

**Keying is on the board id, not the role** (`naming.boards_of`, `naming.board_pixel_key`):
roles can be reassigned between runs. The role travels as metadata.

## 2. The coefficient bank

Fit once, evaluate many: the `bank` command (section 6) writes one parquet per run, and
`bank.bank_rows_for_triple` returns its rows for one triple, one row per (campaign, run,
angle_deg, telescope, combo, src_triple, board_id, role, fold, guard, placebo), the key
`bank.BANK_META`. `fold` is `full` plus, for `folds="parity"`, `0` and `1`; for `"block"`,
`blockA` and `blockB`; for `"both"`, all four. A fold of fewer than 50 events is skipped.
`guard` is `full` or `lpo:<dropped role>`; `src_kind` is `track` (a pooled source is `pooled`).

The 56 columns, `bank.BANK_COLUMNS`, in the order written:

| group | columns | meaning |
|---|---|---|
| identity | `campaign`, `run`, `angle_deg`, `telescope`, `combo`, `src_triple`, `src_file`, `board_id`, `role`, `fold`, `guard`, `placebo`, `src_kind` | the key above, plus the track name |
| coefficients | `c2`, `c1`, `c0` | raw-TOT polynomial in ps, applied as `toa + polyval([c2, c1, c0], tot)` |
| | `mu_tot`, `sd_tot`, `b2`, `b1`, `b0` | the same polynomial in `u = (tot - mu_tot) / sd_tot`, the comparable form; `mu_tot`, `sd_tot` are those of the fit events |
| | `cov_c`, `n_fit` | the 3x3 covariance of `(c2, c1, c0)` (JSON, flattened) and the number of fit events |
| metadata | `rho_pairwise`, `cal_mean`, `hv`, `temp`, `row`, `col` | mean pairwise TOT correlation of the triple, the board's mean CAL code, run conditions, the pixel |
| | `pair_moments`, `resid_moments` | JSON: per pair `E[t_a t_b]`, `E[t_a^2 t_b]`, `E[t_a t_b^2]`, `E[t_a^2 t_b^2]` of the triple's TOT; per pair the moments `E[r]`, `E[r t_a]`, `E[r t_a^2]`, `E[r t_b]`, `E[r t_b^2]` of the residual of the triple's in-sample solve |
| | `order`, `solver`, `git_sha`, `ts`, `neighbor`, `neighbor_logic` | 2, `twc_solver`, the caller's code version, the time written, the neighbour cut |
| TOT summary | `tot_min`, `q005`, `q01`, `q05`, `q25`, `q50`, `q75`, `q95`, `q99`, `q995`, `tot_max`, `mean_tot`, `sd_tot_emp`, `m1`..`m4`, `hist_lo`, `hist` | of this board's TOT in the row's fold of the triple: quantiles (q005 = 0.5 %), the first four raw moments, and a 25 ps histogram starting at `hist_lo` (JSON counts) |

`cov_c` is an approximate covariance (`solve.coeff_cov`: the OLS form on the solver's own
design, which ignores that the three pair rows of one event are correlated). It is used as a
scale, never as a quoted error.

**Placebo rows** (`placebo = 1`) are the same solve with each board's TOT permuted among the
fold's events. The permutation seed is a blake2b hash of the track name and the fold, so it
does not depend on Python's per-process hash salt.

**Leave-pair-out rows** (`guard = lpo:<d>`) come from the sibling combo of the same run that
omits role d and keeps the triple's other two pixels (`bank.lpo_events`): every matching file
(the fourth board's pixel is free), largest first, until `N_CAP` events, then capped with a
fixed seed. The fold subset is taken on the sibling's own file parity (or its file median for
the blocks), the sibling's roles are solved jointly, and rows are written for the two kept
roles only. A role with no sibling combo gets no rows.

**Gauge.** The constant of each board's polynomial cannot be observed in a width. The stored
`c0` is the solver's own (minimum-norm) choice and is never compared across sources. At
application every arm, both self baselines included, removes the mean of each board's
correction over the events it is applied to (`solve.gauge_to`, `solve._gauge_corr`), so no arm
differs from another by a location. Coefficients of different sources are compared only after
re-expressing them in one declared basis, the target's TOT mean and sd (`solve.raw_to_centred`).

`cov_c` and the TOT moments are what let Tier-0 (section 4) and the analytic fit-noise
prediction run without re-reading events.

## 3. Transfer operations: the terms the library uses

A cell is (source, target, mode). In an `all3` transfer, the headline, all three boards take the
source's coefficients. In a `one_board` transfer one board takes them and the other two keep
their own fold-matched `self_cf` coefficients (`arms.transferred_map` with `base` and
`roles_sub`); a pair that does not contain the transferred board then has a residual array
identical to its `self_cf` one, and because the width fit is seeded from the residual bytes
(section 4) its width is identical too. The in-run stability split is a transfer from one
contiguous file block of a triple to the other. The pixel-to-board protocols pick their source
as "the most populous pixel, its coefficients from its most populous track":
`naming.pixel_tracks` ranks a board's pixels by the total size of the track files that carry
them and names each pixel's largest track.

## 4. Estimator and errors

### Three tiers

**Tier-0, analytic and GMM-free** (`tier0.tier0_delta`). For a change of coefficients
`dW_r(t) = W_src,r(t) - W_self_in,r(t)` on each board, with `t` the target's TOT:

    pair  d(sigma^2_ab) = Var[dW_a] + Var[dW_b] - 2 Cov[dW_a, dW_b]
    board d(sigma^2_r)  = 1/2 (d_rb + d_rc - d_bc)  over the pair changes d

Every term is a polynomial in the stored moments, so the whole cross-product of sources and
targets is arithmetic on the bank. Without the residual moments the prediction is exact only
for the sum over the three pairs (the joint least-squares residual is orthogonal to the design
only in that sum); with `resid_moments` it is exact pair by pair. The limitation is stated, not
glossed: this is exact for the change in RMS^2 of the pair difference against `self_in`, not
against `self_cf`, and not for the GMM core. A null from two independent split-half self fits,
`P_null`, converts it into an excess `P_exc = sqrt(max(P^2 - P_null^2, 0))`. Tier-0 is a screen
and a predictor, never a quoted result.

**Tier-1, free.** `IQR/1.349` and the RMS, on the same pooled residuals as every Tier-2 fit
(`width_gmm` returns both; `widths.width_iqr`). They cost nothing and calibrate Tier-0 against
Tier-2.

**Tier-2, the pipeline's own width** (`widths.evaluate_arm`). Per arm: apply the coefficients
with the gauge, form the three pair differences, subtract each fold's own median, pool the
folds, then `BS.fit_gmm_and_get_fwhm`, `w = FWHM / 2.355`, and the three-board sigma solve
`BS.calculate_resolution_from_fit`. A board whose radicand
`1/2 (w_ab^2 + w_ac^2 - w_bc^2)` is not positive gets `sigma = NaN` and `neg_radicand = 1`;
the pipeline's floored 0.0 is never used as a value. Per fit the evaluation records `ks_d`,
`ks_p`, `accepted` (`ks_p >= KS_PMIN` or `ks_d <= KS_DMAX`), the components branch, `n`,
`grid_step = (max - min) / 999`, `iqr`, `rms`, and per board `own_slope`, the slope of the
corrected residual against the board's own TOT (zero for a converged self solve; nonzero says
the transferred shape is wrong). Widths are computed pooled and per fold.

**Seeding of the width.** `bootstrap.py` leaves its `GaussianMixture` unseeded, so it draws from
numpy's global random state. `widths.width_gmm` seeds that state with a blake2b hash of the
residual array's bytes before each fit. A width is then reproducible, and two arms with
bit-identical residuals get bit-identical widths. Unseeded, two runs on the same input differed
by mean 0.16 ps, 99th percentile 0.70 ps, maximum 17 ps (thin triples just above 1500 events).
The other side of the same rule: a change in the last bit of a coefficient changes the seed and
can move a width by the restart jitter, not only by the float difference.

### Arms

The core arms (`arms.ARMS_CORE`):

| arm | coefficients applied to fold f |
|---|---|
| `notwc` | zero on every board |
| `self_in` | the target's fit on all its events: in-sample, reported as optimism only |
| `self_cf` | the target's fit on the complementary fold: **the baseline**, and the self-transfer null of 5.5 |
| `self_cf_lpo` | `self_cf`, with each pair (a, b) using for a a fit that excluded b (5.1) |
| `placebo_self` | the target's complementary-fold fit on permuted TOT |
| `xfer` | the source's coefficients (all three boards, or one board in a one-board cell) |
| `xfer_lpo` | `xfer`, with each pair (a, b) using for a the source's fit that excluded b |
| `placebo` | the source's permuted-TOT fit, transferred |
| `notwc_1b` | the reference of a one-board cell: the transferred board uncorrected, the other two on `self_cf` |

The extended arms (`arms.ARMS_EXT`): `xfer_clip` (the source polynomial evaluated at the target's
TOT clipped to the source's [q0.5, q99.5]), `xfer_shift` (the target's TOT shifted so its median
matches the source's), `xfer_rescale` (the target's TOT mapped onto the source's median and
sd), `xfer_calfix` (the target's TOT rescaled by the ratio of the source's to the target's mean
CAL code), `self_in_rep` (the GMM repeat probe of 5.7), and `xfer_lin` / `xfer_lin_lpo` (below).
The one-board diagnostic `xfer_1b:<board>` replaces one board of `self_cf` by the source's.

**`xfer_lin`, shape versus slope** (`arms.xfer_lin_map`). The source's quadratic coefficient
`c2` is held at its transferred value on every board; the linear term and the constant are refit
on the target's own events, fold-matched like `self_cf` (fold f's from the complementary fold),
jointly across the three boards, by one call of the unmodified solver at `order=1` on
`toa_r + c2_r * tot_r^2`. The sign is `+` because the apply line adds the correction:
`toa_r + c2_r t^2 + c1_r t + c0_r = (toa_r + c2_r t^2) + c1_r t + c0_r`. Degrees of freedom per
board, counted comparably: the raw fits have 3 (`self_cf`), 2 (`xfer_lin`) and 0 (`xfer`), but
the gauge (section 2) and the per-fold median removal each take one location degree of freedom
from every arm alike, so the effective ladder is `self_cf` / `xfer_lin` / `xfer` = 2 / 1 / 0.
`xfer_lin` is defined for `all3` cells only.

**`xfer_lin_lpo`, its leave-pair-out guard** (`arms.xfer_lin_lpo_map`). For pair (x, y), x's
coefficients must come from a source fit that excluded y. For each pair and each direction a
fresh `xfer_lin` solve is made with only the guarded board's `c2` replaced by the leave-pair-out
source's (`lpo[y][x]`), every other board keeping the plain source's, and only the guarded
board's resulting coefficients are kept: 3 pairs x 2 directions x 2 folds = 12 joint solves per
triple. Reusing the plain fit's `(c1, c0)` would leave the guard partial, since board y's fixed
`c2` enters the same joint design that sets x's `(c1, c0)`. A second-order coupling remains and
is stated: inside each guarded solve, y still contributes through its own unguarded `c2`, a
coupling `xfer_lpo` does not have, because its guarded coefficient is copied from the
leave-pair-out source fit with no target-side refit.

### Statistics

The primary quantities are in variance: `p_pair = w^2_xfer - w^2_self_cf` and
`p_board = sigma^2_xfer - sigma^2_self_cf`, reported as `D = sigma_xfer - sigma_self_cf` and as
the retention `R = 1 - D / G`, with `G = sigma_notwc - sigma_self_cf` what the correction buys
on that pair or board (comparable across angles, since the walk amplitude grows with the angle).
`sigma_xfer - sigma_self_in` is reported separately and labelled in-sample optimism; quoting it as
transfer loss is the main trap of the study (measured `self_in - self_cf`: +2.8 ps at N = 300,
+0.1 ps at N = 5000). Pair widths are primary; board sigmas are derived, with negative radicands
and the 0.35 / 0.95 margins of `quote_resolution.py` recorded as flags, never as silent zeros
or drops.

Everything is paired per triple (the pixel-to-pixel spread is about ten times the effect and
cancels in the pairing) and compared within one combo, so the pairing covariance g (+60 ps^2 on
H1 at 1.5e15, +619 ps^2 on F1 at 3.5e15, untouched by any TOT correction) cancels. Agreement of
D across combos is a reported consistency check, not an assumption.

**Errors.**

1. Primary: the half-split applied to the paired difference, not to the widths:
   `sd(D_i) = k_cal * s_i` with `s_i = w_self_cf,i / sqrt(n_i)` (`stats.kcal_from_cells`:
   `k_cal` is the median of `|D_f0 - D_f1| / s_i` over triples, divided by `2 * 0.6745`, pooled
   once per run). This carries `quote_resolution.py`'s calibration to the paired difference.
2. Verification: on anchor triples, a paired bootstrap over blocks of two files with the self
   coefficients refit inside each resample, B = 100; `f = sd_boot / sd_half` must be close to 1,
   and the iid bootstrap is reported beside it so serial-correlation inflation is visible.
3. Aggregation: random effects `D_i ~ N(mu, f^2 sd_i^2 + tau^2)`, DerSimonian-Laird
   (`stats.random_effects`, which also returns the fixed-effect cross-check, `chi2` per degree
   of freedom, the median and the 95th percentile). With more than one source pixel,
   `stats.source_random_effect` adds the source-to-source spread, so a claim generalises over
   sources and not only over targets; a cell with one source states that its claim is
   conditional on it.

**Source event count.** The source fit is scanned over n_fit in {250, 500, 1000, 2000, 4000}
and evaluated on one fixed held-out set, `p = alpha / n_fit` is fitted, and each cell's penalty
splits into `p_measured - alpha / n_fit,S` (genuine mismatch) and the fit-noise part; `alpha` is
cross-checked against the analytic `tr(M_T Cov(c_S))` from the bank.

**Decision.** TOST at alpha = 0.05 (`stats.tost`) against the pre-registered margin
`max(1.0 ps, 2 % of sigma_self_cf)` (`stats.margin_for`): PASS when the 90 % interval of D lies
inside the margin. No claim is made below 0.45 ps or 1 %, the definition systematic of the FWHM of
a mixture. Reported beside it: the fraction of triples over the margin and the 95th percentile
of D. Holm's step-down (`stats.holm`) is applied to the non-equivalence claims only, with
`stats.p_nonequiv` as their p-value; a set of equivalence claims that must all pass is an
intersection-union test and needs no correction.

## 5. Controls (all mandatory)

1. **Leave-pair-out.** For target pair (a, b), a's coefficient comes from a source solve that
   excluded b: with four boards, the leave-one-out combo that omits b. The cheat statistic is a
   difference of differences with its own null,
   `[sigma_xfer - sigma_xfer_lpo] - [sigma_self_cf - sigma_self_cf_lpo]`, which regroups as
   `[sigma_xfer - sigma_self_cf] - [sigma_xfer_lpo - sigma_self_cf_lpo]`: the transfer effect
   without the guard minus the same effect under it. The guard is enforced at the source as well,
   since sources share boards with targets. The measured cost of the protocol ranges from
   +0.22 ps to +2.31 ps. Where a run is effectively three-board, the guard
   degrades to a two-board solve: the self guard cost is reported, and the guard is declared
   uninformative when that cost exceeds the margin.

   **Where no sibling combo exists** (`self_coeff_maps(lpo_fallback=...)`): the default
   (`None` or `"sibling"`) builds `self_cf_lpo` only if all three dropped roles have a same-pixel
   sibling combo, and otherwise not at all. `"own"` builds the guard per dropped role: a sibling
   when there is one, and otherwise a joint solve on the target's own fold-matched events
   restricted to the two remaining roles, a design that never sees the dropped role. The cheat
   statistic stays well posed, because each of its two brackets differences arms identical on
   every unguarded board, whatever built the guarded entries; a two-board guard is noisier, which
   the uninformative rule above censors. `maps.lpo_shape` records, per dropped role, which
   construction was used.

   The positive control is `widths.leaky_pairwidths`: per pair, a correction `p(t_a) + p(t_b)`
   least-squares fitted directly on the pair difference, on the complementary fold. It leaks no
   event, only partner information, which is the channel the guard exists to forbid (the channel
   that exposed a neural-network TWC, at -4.26 ps). A guard with teeth shows this arm improving
   while `xfer` does not; if both are near zero the guard is untested, not clean.
2. **Permuted-TOT placebo.** The source is solved on shuffled TOT and transferred; it must
   reproduce `notwc` (R near 0; in a one-board cell, `notwc_1b`). It bounds "any smooth polynomial
   narrows this", not "a polynomial fitted on correlated TOT induces covariance"; the
   leave-pair-out arm covers that, and the remaining exposure is stated.
3. **TOT overlap** (`bank.overlap_metrics` on events, `bank.overlap_from_bank` from two bank rows
   only, exact up to the 25 ps binning). Per (source, target, board): the coverage C, the target
   fraction inside the source's [q0.5, q99.5]; the histogram intersection O; the standardised
   shift and the sd ratio; and the extrapolation lever X, the rms in ps of
   `|W_S(t) - W_S(clip(t))|` over the out-of-support events. A cell is flagged EXTRAP when
   C < 0.98 or X > 5 ps; interpolating and extrapolating families are reported separately, and a
   transferability claim is made on the interpolating family only. `xfer_clip` separates "the
   quadratic explodes outside its support" from "the curve is wrong".
4. **Noise floors.** The in-run block split, and run-to-run pairs at identical conditions where
   they exist; every other cell is quoted raw and as an excess over that floor.
5. **Nulls.** The self-transfer null (`self_cf`: the source is the complementary fold of the
   same triple) in every cell; a shuffled source-label null for the aggregate; a random-source
   null (a board with incompatible TOT support).
6. **KS gate.** A triple is never dropped because an arm failed the gate, which would keep exactly
   the transfers that happened to work. The verdict is recorded (`accepted`) and the failure rate
   quoted per arm; a symmetric KS-cleaned re-run is a robustness check only. Cleaning is on
   quality only (event floors, coverage, degeneracy flags), never on D, and always symmetric
   across arms.
7. **GMM repeat probe `self_in_rep`.** It is evaluated on `self_in`'s identical coefficients, so
   seeding by content alone would make it bit-identical to `self_in`, a probe turned into a
   structural zero. `evaluate_arm` therefore passes `seed_salt=1` for this arm alone: a
   deliberate, reproducible re-draw of the same estimator on the same residuals. If the jitter is
   comparable to D, the remedy is the median of three refits (salts 1 to 3); the library
   evaluates salt 1 only.
8. **Fidelity guards.** `grid_step` is recorded per fit and D is checked to be uncorrelated with
   the step difference; cells are stratified by N and compared cells subsampled to a common n, so
   the 1500-event component switch never differs between them; `cal_mean` travels with every
   bank row and `xfer_calfix` measures how much of an apparent loss is calibration.

Pre-registered floors: n_eval >= 600, n_fit >= 1000, headline n >= 1500, `N_CAP = 6000`.

## 6. Commands

`python -m twc_transfer <subcommand>`, run from `TestBeam/` (`cli.py`). `__main__.py` first
pins every thread pool to one thread and, when `PYTHONHASHSEED` is not 0, restarts the
interpreter once with `PYTHONHASHSEED=0`, saying so on stderr. Argument names, defaults and
types are those of the study's own command line, so the commands of its job lists parse
unchanged.

    index  --campaign C --run R [--combo X[,Y]] [--root DIR] [--ntrip 100000]
           -> bank/index_<C>_<R>.parquet
    bank   --campaign C --run R [--combo X[,Y]] [--root DIR] [--ntrip 40] [--nmin 600]
           [--folds parity|full|block|both] [--guards full[,lpo]] [--placebo] [--cap 6000]
           [--neighbor none] [--neighbor-logic OR] [--angle A] [--telescope T] [--hv V]
           [--temp T] [--chunk k/N] [--append]
           -> bank/coeffs_<C>_<R>.parquet          (alias: fit)
    tier0  --bank FILE [--tag all] [--max-src 0]
           -> tier0/transfer_<tag>.parquet
    jobs   --list LIST [--job ID ...] --dry-run
    cell   the arguments of one transfer cell (alias: transfer); parsed, not run, by this
           version

- `--combo` absent means every combo of the run that has a `time/` folder. `--ntrip` takes the
  largest track files of each combo (file size as the proxy for event count); `bank` skips a
  triple with fewer than `--nmin` events after the cut, and `--chunk k/N` keeps every N-th of
  those files from the k-th on.
- The study's banks were written with `fit --campaign C --run R --ntrip 40 --folds both
  --placebo`, the remaining options at their defaults.
- **The write root.** `index`, `bank` and `tier0` need `ETROC_TWC_OUT` and write under it (the
  arrows above are relative to it); `--out` names another output file. They refuse to start
  when it is unset, when it is the read root `ETROC_TWC_TRANSFER_ROOT`, and when it is the
  step-10 root of a campaign, so no run writes into the study's tree or among the track files.
  Only these subcommands create the output folders.
- **Inputs.** `tier0 --bank` is read from the read root when the path is relative.
- **Job lists** (`jobs.py`). The seven lists the study ran are in `jobs/`, in the order of
  `campaigns.JOB_LISTS`, which also records the output root each wrote into (the seeded rerun,
  `cells_rerun.tsv`, wrote into its own tree). A row has seven tab-separated fields, `job_id
  cell_id level group est_min mem_mb command`; the header row is optional, `#` lines are
  skipped, and the chunk jobs of a long cell share its `cell_id`. In the commands,
  `${ETROC_TWC_ROOT_<CAMPAIGN>}` stands for a campaign's step-10 root and `${TT_OUT}` for the
  study's tree: under an input flag (`--bank`, `--src-bank`, `--root`, `--src-root`, `--cells`,
  `--tier0`) it becomes the read root, under `--out` the write root, and anywhere else it is
  refused, as is a write root equal to the read root. Expanded with the default roots and
  `${TT_OUT}` left in place, the lists reproduce the study's commands byte for byte.
  `jobs --dry-run` prints every command expanded and parses it with the parser above; a cell
  command must name its row's `cell_id` in `--cell`.

## 7. Output tables

- `bank/index_<C>_<R>.parquet`: one row per track file of the combos read: `campaign`, `run`,
  `combo`, `path`, `name` (the file name without `.parquet`), `nevt` (the file's row count,
  before any cut), `triple` (`naming.triple_key`), `bytes`; sorted by `nevt`, largest first.
- `bank/coeffs_<C>_<R>.parquet`: the bank rows of section 2, `bank.BANK_COLUMNS`. `git_sha` is
  empty and `ts` is the time the row was computed.
- `tier0/transfer_<tag>.parquet` (`tier0.bank_screen`): one row per (target triple, source
  triple, kind, name). Targets and sources are the bank's `guard = full`, `placebo = 0`,
  `fold = full` rows; every other triple of the same combo is a source of a target, the target
  itself excluded, and a source must carry all of the target's roles (`--max-src N` keeps the
  first N source triples of a combo by track name). Columns: `run`, `combo`, `tgt`, `src_run`,
  `src` (track names), `kind` (`board` or `pair`), `name` (a role, or `a-b`), `dvar` (the
  signed Tier-0 change in RMS^2, ps^2, with the residual moments), `P = sqrt(max(dvar, 0))`,
  `P_signed`, `P_null` (the same statistic between the target's own fold-0 and fold-1 fits),
  `P_exc = sqrt(max(P^2 - P_null^2, 0))`, `own_var` (board rows: the variance of the
  coefficient change over the target's TOT), and the overlap metrics `cov_C`, `hist_O`,
  `lever_X`, `lever_X_all`, `extrap` of `bank.overlap_from_bank` (a pair takes the smaller
  coverage and overlap and the larger lever of its two boards).
