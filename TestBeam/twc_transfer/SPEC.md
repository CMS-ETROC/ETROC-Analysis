# twc_transfer: specification of the library

What the package computes and the rules it keeps. Sections 0, 1, 2, 4 and 5 define the library
in this directory; section 3 the transfer operations of the cell command (cell.py), its modes
and the two pixel-to-board protocols; sections 6 and 7 the commands and the tables they write;
section 8 the aggregation of the cell tables (agg.py); section 9 the round-1 chain from the
aggregation tables to the table of the level cells (master.py); section 10 the axis tables
(axes.py).
The code cites sections as "SPEC 1", "SPEC 5.1" and so on.

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

## 3. Transfer operations

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

**Modes** (`cell --mode`). Each names where a target triple's source comes from.

| mode | source of a target triple |
|---|---|
| `a` run to run | the triple with the target's own pixels in `--src-run` (and `--src-campaign`, `--src-combo`), named by the target's role -> pixel map |
| `b` angle to angle | as `a`; the source run is another angle |
| `f` cross grid | as `a` |
| `c` pixel to board | a named source triple applied to every target triple of the combo: whole-triple (`all3`) by default, protocol P3 (3b) with `--xfer-board` |
| `cb` pixel to other board | protocol P4 (3b): one board's polynomial of a named source triple onto another board |
| `d` board to board | a named source triple, its boards mapped onto the target's by `--role-map 'tgt:src,...'` |
| `e` pooled board | per board, the `--pool-ntrip` largest triples of the target run, leave-one-pixel-out |
| `inrun` in-run split | one half of the target triple itself; the other half is evaluated |

In modes `a`, `b` and `f` the source is read from the source run's track file of the same pixels
(or its bank row, with `--src-bank`; so is a named source in modes `c` and `cb`). In modes `c`,
`cb` and `d` the sources are named by
`--src-triple` (comma-separated) and completed to `--src-k` sources by `--src-pick`: `trkpop`
takes the largest track files whose pixels are new on every board, `pixpop` ranks the pixels of
the board the polynomial is fitted on (`--src-role`, else `--xfer-board`) by population and
takes each one's largest track, `auto` is `pixpop` when a board is named and `trkpop`
otherwise; every target is evaluated against every source. `--src-cap` raises the source's own
event cap (0 follows `--cap`).

Mode `e` (`cell.pooled_source`) assembles the pool one board at a time: for board r every pool
triple whose pixel on r is the target's is dropped (`--pool-lopo-scope triple` drops only the
target's own triple, the biased variant kept for comparison), each triple contributes at most
`--pool-cap` events with its TOA median-centred per board, and only board r's coefficients are
taken from the pooled solve. `xfer_avg` (with `--arms ext`) is the solve-then-average
alternative in the centred basis, n-weighted, with the gauge constant set to zero. The
leave-pair-out guard of the pool (`cell.pooled_lpo_source`) pools the sibling-combo events of
the first `--pool-lpo-ntrip` pool triples under the same exclusion.

Mode `inrun` fits the source on one half of the target triple and evaluates the other:
`--split block` the contiguous file blocks A and B (drift is charged to the transfer),
`--split parity` the two file parities, with the evaluation half's folds re-derived from the
rank of its files. Its leave-pair-out guard is the sibling combo's events restricted to the
source half's files.

`--xfer-lin` adds `xfer_lin` and `xfer_lin_lpo` (section 4) to a whole-triple cell; it needs
`--arms ext` and is refused with `--xfer-board`. `--src-parity-contrast` adds `xfer_srcmatch`
and `xfer_srccross`: the source refitted on the file parity that matches the evaluation fold
and on the other one, whose difference is the drift a same-run source shares with its target;
it is refused unless the source is in the target's campaign and run.

**Board identity.** Role sets are not board identity: two campaigns can give the same role names
to different boards (March H1 `extra0-ref1-dut2`, July H1 `extra0-dut1-ref2`). Every cell
asserts that the source puts the same board under each role as the target and stops with
`cell.BoardIdentityError` otherwise, naming the complete `--role-map` that would fix it when one
exists. Mode `d` is exempt once a `--role-map` declares the mapping; mode `cb` declares its
substitution with `--src-role` and `--xfer-board`, which it requires. `--role-map` is refused in
modes `e` and `inrun`, which have no source with a role map of its own, and in modes `a`, `b`
and `f` with a source in the target's own run and combo, where it would make a cross-board
transfer inside one run under a same-pixel label (`LeakageError`). A role map permutes the
source's coefficients, placebo, statistics, pixels and leave-pair-out fits onto the target's
role names. The per-event keys of section 1 are compared only when both sides put the same
board under each role name; the row records this as `keys_comparable`.

**Self-transfer.** In modes `a`, `b`, `c`, `cb`, `d` and `f` a target's `self_transfer` is 1
exactly when the source triple's name is the target's, the source run (`--src-run`, else the
target's) is the target's run and the source combo (`--src-combo`, else the target's) is its
combo; the campaign is not compared. Such a target takes its own fold-matched fits in `xfer`
= `self_cf`, `placebo` = `placebo_self` and `xfer_lpo` = `self_cf_lpo` (in a one-board cell,
board `sr`'s fold-matched fit on board `xb`), and the `xfer_clip` / `shift` / `rescale` /
`calfix` variants are built from that `xfer`: the self-transfer null of 5.5. The diagnostics
`xfer_1b:<board>` and `xfer_lin` take the source's coefficients as they are, which for such a
target are its own fit on all its events, so its rows of those two arms are in-sample. In modes
`e` and `inrun` the flag is always 0. The self arms are written with `self_transfer` = 0 and
blank source columns.

### 3b. The pixel-to-board protocols

- P3, pixel to board (`--mode c --xfer-board B1`): the coefficients of source pixel P1 on board
  B1, from its most populous track, applied to board B1 of every other target triple; B2 and B3
  keep their own `self_cf` coefficients. The cell's own source triple is not a target (its
  `xfer` arm would be `self_cf` itself).
- P4, pixel to other board (`--mode cb --src-role B1 --xfer-board B2`): P1's coefficients from
  B1 applied to board B2 of every target triple; B1 and B3 keep their own. The two flags must
  name different boards; mode `c` with two different boards is refused and pointed at `cb`.

`--xfer-board` and `--src-role` are target role names (a `--role-map` has re-keyed the source
onto them); the source pixels are ranked on the source's own board. In a one-board cell every
transfer-derived arm (`xfer`, `placebo`, `xfer_lpo`, the extended variants) replaces board
`xb` only, and the reference arm `notwc_1b` (section 4) is added. The leave-pair-out guard of
pair (`xb`, y) needs a source solve that excluded y, `lpo[y][sr]`; for y = `sr` there is none,
so P4 guards one of `xb`'s two pairs and P3 both. The number of guarded pairs is written as
`lpo_pairs` (under `--lpo-fallback own` also for an `all3` cell, counting the guard entries
filled, up to 6).

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
    cell   --cell ID --mode a|b|c|cb|d|e|f|inrun --campaign C --run R --combo X [--root DIR]
           [--src-campaign C] [--src-run R] [--src-combo X] [--src-root DIR] [--src-triple T]
           [--src-bank FILE] [--src-k 5] [--src-pick auto|pixpop|trkpop] [--src-cap 0]
           [--role-map T:S,...] [--xfer-board B] [--src-role B] [--split block|parity]
           [--pool-ntrip 40] [--pool-cap 1500] [--pool-lopo-scope board|triple]
           [--pool-lpo-ntrip 6] [--arms core|ext] [--lpo] [--lpo-fallback sibling|own]
           [--xfer-lin] [--src-parity-contrast] [--ntrip 12|all] [--triples T,...]
           [--chunk k/N] [--nmin 600] [--nmin-src 1000] [--nmin-fold 250] [--cap 6000]
           [--max-triples N] [--timebudget S] [--flush 2] [--fast] [--overwrite] [--dry-run]
           -> cells/<ID>.csv                          (alias: transfer)
    agg    --cells PATTERN [PATTERN ...] [--baseline self_cf]
           [--f-inflate F | --f-cal C [--f-precision agg|full]] [--nmin-eval 600]
           [--tier0 FILE] [--out DIR]
           -> tables/{delta_by_triple,agg_by_cell,calibration}.csv   (section 8)
    master [--jobs cells.tsv] [--out DIR]
           -> master/master_cells.csv and its intermediate tables     (section 9)
    axes   [--axis A|B|C|D|E|F|I|J|K ...] [--guardfix] [--out DIR]
           -> results_r2/axis_<X>.csv, results_r3/axis_F.csv, axis_F_long.csv, summary_F.md;
              with --guardfix results_r3/guardfix/axis_<E,J,K>_gf.csv   (section 10)

- `--combo` absent means every combo of the run that has a `time/` folder. `--ntrip` takes the
  largest track files of each combo (file size as the proxy for event count); `bank` skips a
  triple with fewer than `--nmin` events after the cut, and `--chunk k/N` keeps every N-th of
  those files from the k-th on.
- The study's banks were written with `fit --campaign C --run R --ntrip 40 --folds both
  --placebo`, the remaining options at their defaults.
- **The write root.** `index`, `bank`, `tier0`, `cell`, `agg`, `master` and `axes` need
  `ETROC_TWC_OUT` and write under it (the arrows above are relative to it); `--out` names
  another output file or folder. They refuse to start when it is unset, when it is the read
  root `ETROC_TWC_TRANSFER_ROOT`, and when it is the step-10 root of a campaign, so no run
  writes into the study's tree or among the track files.
  Only these subcommands create the output folders, and `cell --dry-run` creates none.
- **Inputs.** `tier0 --bank`, `cell --src-bank`, `agg --cells` and `agg --tier0` are read from
  the read root when the path is relative. `master` and `axes` read only the read root and the
  job lists in `jobs/`.
- **Cell targets.** `--ntrip` takes the largest track files of the target combo (`all`: every
  one); `--triples` names them instead (track names without `.parquet`, in the target combo's
  `time/` folder; a name with no file is dropped); `--chunk k/N` then keeps every N-th from the
  k-th on. A target with fewer than `--nmin` events, or a fold with fewer than `--nmin-fold`, is
  skipped, and every skip is printed at the end. `--max-triples` stops after that many measured
  triples and `--timebudget` after that many seconds; either leaves a resumable table.
- **`cell --dry-run`** prints the plan (the number of targets, sources and arms, the mode, the
  target, the source, the target names, the output file and the write root) and writes
  nothing; it reads only file listings and, when the output table exists, its `tgt_triple`
  column.
- The placebo seeds of a cell need `PYTHONHASHSEED=0` (section 7).
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
- `cells/<ID>.csv` (`cell.CELL_COLUMNS`, 50 columns in a fixed order): one row per (target
  triple, source, arm, kind, name). `kind = pair` rows carry the pair width (`name` = `a-b`,
  `value` the pooled-fold width, `v_f0` / `v_f1` per fold, the width fit's `ks_d`, `ks_p`,
  `accepted`, `comps`, `grid_step`, `iqr`, `rms`); `kind = board` rows the board sigma solved
  from the three pair widths (`own_slope`, `neg_radicand`). Every row carries the cell's
  identity (`cell`, `mode`, `campaign`, `run`, `combo`, `tgt_triple`, `tgt_key`), the source
  (`src_run`, `src_combo`, `src_triple`, `n_fit_src`, `self_transfer`, `same_pixel`,
  `same_board`, `src_file_overlap`, `src_keys_checked`, `keys_comparable`, `n_pool_excluded`),
  the protocol columns (`xfer_board`, `src_role`, `src_pick`, `src_pix_rc`, `src_pix_ntrk`,
  `src_pix_pop`, `lpo_pairs`), the overlap metrics on transfer arms (`cov_C`, `hist_O`,
  `lever_X`, `lever_X_all`, `extrap`), `rho`, and `ts`, the time the target's rows were made.
  There is no `git_sha` column.
- **Resume, keyed on `tgt_triple`.** Rows are appended after every `--flush` target triples
  (default 2) and at the end, with the fixed column list. A table that exists is resumed unless
  `--overwrite` is given: its header must be `CELL_COLUMNS` (otherwise the cell stops before
  reading an event), the target triples already in it are skipped and the others appended, so a
  rerun of the same command never writes a triple twice. `--overwrite` removes the table
  first.
- `<table without .csv>.lpo_shape.json`, under `--lpo-fallback own` only: for each
  `tgt_triple`, which construction built each dropped role's leave-pair-out guard,
  `{"self": {role: "sibling"|"own"}, "xfer": {role: ...}}` (target side, source side). It is
  merged into an existing side-car, under `--overwrite` too (which removes only the table), and
  is not locked: two chunk jobs of one cell writing the same side-car can overwrite each
  other's entries.
- **Placebo seeds.** A source refitted from events seeds its placebo with
  `abs(hash(("placebo", source triple name, source run))) % 2**31`, and the in-run split with
  `abs(hash((target triple name, "pl"))) % 2**31`: Python's string hash, fixed only by
  `PYTHONHASHSEED`. With `PYTHONHASHSEED=0` (set by `python -m twc_transfer`) under LCG_104d
  (Python 3.9.12) these are the seeds of the study's cells; other Python versions are not
  verified. A bank source's placebo was seeded when the bank was made (section 2) and does not
  depend on it.

## 8. Aggregation

`python -m twc_transfer agg` (`agg.py`) turns the cell tables of one group of cells into three
tables: the paired difference of every triple against the baseline arm, the aggregate per
(cell, arm, quantity) and the error calibration. `aggregate()` does it in memory and writes
nothing; `cmd_agg` reads and writes.

**Inputs.** The cell tables (section 7) matched by `--cells`, a list of paths or glob patterns
(relative ones in the read root), sorted by path and stacked (`read_cells`). A file without the
columns `arm`, `kind`, `name` and `value` is skipped with a line; a table with another column
set (the 43-column tables of round 1) is aggregated on the shared columns, the missing ones NaN.
The side-car of a cell run with `--lpo-fallback own` is merged in memory as a column
`lpo_shape`, the JSON text of its entry for the row's `tgt_triple`. The study aggregated the
cells of one `group` of a job list together (smoke and Tier-0 rows left out), every table of
each cell (chunks included), once per f.

**The error inflation f** (section 4, errors 3), `f_inflate_for`, in this order:

1. `--f-inflate F`.
2. `--f-cal C`: `campaigns.F_CAL[C]` at `--f-precision`. `agg` (the default) is the value the
   study's aggregation ran with, four decimals: 0.8153 (desy), 0.8979 (aug), 0.9524 (mar),
   0.9259 (jul); with it the study's numbers come back. `full` is the bootstrap's
   value: 0.8152579445510614, 0.8979111975615606, 0.9524 (known to four decimals only),
   0.925913381797006; the round-1 recomputation of the verdicts used these.
3. Neither: the `f_cal` of `<read root>/tables/f_cal.json` (in the study's tree the DESY May
   run26 bootstrap), whatever the campaign of the cells; without that file 1.0, the uninflated
   half-split error, with a warning.

The study passed every f with `--f-inflate` (its four-decimal values, and 1.0 and 0.8153 for the
sensitivity runs of DESY May and August). The value is written into `f_inflate` and its origin
into `calibration.csv`'s `f_source`: `explicit` for `--f-inflate`, `f_cal <campaign>, as
aggregated` or `f_cal <campaign>, full precision` for `--f-cal`. The study's tables therefore
read `explicit`, and only `--f-inflate` gives them back byte for byte.

**Rules**, applied in this order, each a function of `agg.py`:

1. `drop_below_nmin_eval`: every row of a (cell, run, combo, tgt_triple) one of whose board
   rows has `n` below `--nmin-eval` (600, the n_eval floor of section 5; 0 disables).
2. `censor_degenerate`: a (cell, run, combo, tgt_triple, kind, name) whose `value` is not
   finite, or whose `neg_radicand` is 1, in any arm leaves every arm (section 5.6). The arms that
   triggered it are counted and printed.
3. The one-board identical-pair drop, with the `self_cf` baseline only (`xfer_board_by_cell`,
   `identical_pair_mask`, `drop_identical_pairs`). A cell whose rows name exactly one non-blank
   `xfer_board` is a one-board cell (a cell naming two is left alone, with a line). Its pair rows
   whose pair does not contain that board, in `placebo`, `notwc_1b` and every arm whose name
   starts with `xfer` except `xfer_lpo` (`is_one_board_arm`), are dropped: they are identical to
   `self_cf` by construction (section 3b) and would read delta = 0 and d_f0 - d_f1 = 0 exactly.
   Against another baseline that pair's delta is a measurement, and nothing is dropped.
4. `deltas`: every arm but the baseline inner-joined to the baseline on (cell, mode, run,
   combo, tgt_triple, kind, name), the first baseline row of a key: `value_b`, `v_f0_b`,
   `v_f1_b`, `n_b`; `delta = value - value_b`, `d_f0 = v_f0 - v_f0_b`, `d_f1 = v_f1 - v_f1_b`,
   `scale = value_b / sqrt(max(n_b, 1))`. No row with a baseline stops the command.
5. `same_pixel_mask`: a row of an `xfer*` arm with `same_pixel = 1`, `src_run = run` and
   `src_combo = combo` is a refit of the board against itself filed as a transfer. It leaves
   every aggregate and the k_cal pool and stays in `delta_by_triple.csv`. Across runs or angles
   the same pixel is the comparison itself, so the rule is restricted to the same run and combo.
6. The k_cal pool (`kcal_exclusion_mask`, `kcal_by_group`). `k_cal` is `stats.kcal_from_cells`
   per (cell, run, arm, kind) over the group's rows except the rule-5 rows and the rows of a
   non-self arm with `self_transfer = 1` (the source triple is one of the cell's targets, where
   `xfer` is `self_cf`); a self arm (`agg.SELF_ARMS`) has no source and is never excluded on the
   flag. A group with no row left uses all of its rows and is named in a line. Every row gets its
   group's `k_cal`, `se = f * k_cal * scale` and `se_halfraw = |d_f0 - d_f1| / 2`.
7. `structural_zero_census`: the rows in the pool with delta == 0 and d_f0 - d_f1 == 0 exactly
   are counted, with their groups, and those outside it beside them. One such row in a pair group
   of an `is_one_board_arm` arm of a one-board cell stops the command (`StructuralZeroError`)
   before any file is written: it would pull the group's median down. The usual cause is an older
   cell table with its own source triple among its targets and no `self_transfer` flag.
8. `retention_gain`: `G = value(notwc) - value(baseline)` per (cell, run, combo, tgt_triple,
   kind, name), `value(notwc_1b) - value(baseline)` where the cell has `notwc_1b` (a one-board
   cell, where `notwc` would add the two boards that keep their own fit); `R = 1 - delta / G`.
9. The aggregate, per (cell, mode, arm, kind, name, self_transfer, extrapolation family), over
   the rows rule 5 keeps, with `self_transfer` set to 0 on the self arms (`delta_by_triple.csv`
   keeps each row's own flag): `stats.random_effects` and `stats.source_random_effect` on
   (delta, se, src_triple); `se = sqrt(se_within^2 + se_src^2)`; `sigma_base` the median
   `value_b`, `margin = stats.margin_for(sigma_base)`, `stats.tost(mu, se, margin)`.
10. The guard (`guard_verdict`, `tost_label`), per (cell, kind, name) from the mean delta of
    `xfer` (xf), `xfer_lpo` (xl) and `self_cf_lpo` (sl) over the rows rule 5 keeps:
    `has_xfer_lpo`, `cheat_stat = (xf - xl) + sl` (section 5.1, NaN unless all three exist),
    `self_lpo_cost = sl`, and `guard_verdict` = `absent` without xl, `uninformative` when |sl|
    exceeds the row's margin, `ok` otherwise. `tost` is `NA-unguarded` on an `xfer*` row whose
    guard is not `ok` and whose `self_transfer` is 0, the TOST verdict otherwise; `tost_raw` is
    always the TOST verdict.
11. `primary_flag`: 1 on pair rows, 0 on board rows, and 0 on the `xfer_lpo` row of the pair of
    a one-board cell without the transferred board (identical to the `self_cf_lpo` control, it
    cannot fail; the row is kept).
12. `holm_board_fails`: Holm's step-down (`stats.holm`) on `p_nonequiv` over each cell's board
    rows with `tost = FAIL` gives `p_ne_holm` and `fail_holm` (NaN and 0 on every other row).

**Outputs**, in `--out` (default `<write root>/tables/`):

- `delta_by_triple.csv`: one row per (arm other than the baseline, triple, quantity) that rules
  1 to 3 keep, rule-5 rows included. Columns: (cell, mode, run, combo, tgt_triple, kind, name),
  the other cell-table columns in their order, `lpo_shape` when the group has side-cars, then
  `agg.DELTA_COLUMNS`: `value_b`, `v_f0_b`, `v_f1_b`, `n_b`, `delta`, `d_f0`, `d_f1`, `scale`,
  `k_cal`, `f_inflate`, `se`, `se_halfraw`.
- `agg_by_cell.csv`: one row per aggregate (rule 9), sorted by cell, kind, arm, name; columns
  `agg.AGG_COLUMNS`, with `lpo_shape` (the distinct non-empty shape texts of the (cell, kind,
  name), joined by ";") before `p_ne_holm` when the group has side-cars. `extrap_family` is
  `interp`, `extrap` or `na` (no `extrap` flag); `n_pix` the rows with a finite delta and a
  positive se, `n_tgt` the distinct target triples, `n_src` the distinct sources; `mu`, `se`,
  `se_within`, `tau`, `median`, `p95`, `chi2_red` (per degree of freedom), `mu_fe`, `se_fe`
  from the random-effects fit; `mu_src_spread`, `se_src` from the source term; `ci_lo`,
  `ci_hi` the 90 % interval; `has_placebo`, `has_self_in_rep` whether the (cell, kind, name)
  has such rows; `placebo_minus_notwc1b_ps` = mean delta of `placebo` minus that of `notwc_1b`
  (the one-board placebo control of section 5.2; NaN without `notwc_1b`); `frac_over_margin` the
  share of |delta| above the margin; `R_median`, `gate_fail` (share with `accepted = 0`),
  `k_cal`, `src_file_overlap`, `n_pool_excluded`, `lever_X_all` medians over the rows.
- `calibration.csv`: one row per k_cal group, `agg.CALIBRATION_COLUMNS`: cell, run, arm, kind,
  `k_cal`, `f_inflate`, `f_source`.
- With `--tier0 FILE` (a tier0 table, section 7): `tier0_calibration.csv`, the least-squares
  line of delta on `P_exc` over the rows matched on (run, combo, target, source, kind, name)
  with both finite (`n`, `slope`, `intercept`, `resid_rms`, `corr`; at least 4 rows), and
  `tier0_vs_tier2.csv`, those rows.

**Versions of the study's aggregation.** Stages 1 and 2 (`tt.py` 05793488) and stage 3
(698f72f9) ran this code less the side-car merge, which acts only where a side-car exists; the
guardfix wave (e83f71c1) ran this code. Round 1 (258cb1f8) had rules 1, 2, 4, 5 (in the
aggregate only), 9, 10, the plain primary flag and rule 12; it had no rule 3, 7, the k_cal
exclusions of rule 6, the `notwc_1b` retention gain G of rule 8, the `xfer_lpo` primary rule,
or the column `placebo_minus_notwc1b_ps` (45 columns), and its cell tables lack the seven
protocol columns.

## 9. The round-1 chain

`python -m twc_transfer master` (`master.py`) runs the four stages by which the study turned
round 1 (job list `cells.tsv`) into the table of its level cells, `master_cells.csv`, and the
verdict of each level. `run()` runs them in order; each stage reads the tables of the stages
before it back from the output folder as written, as the study's scripts did, so the CSV round
trip is part of the result.

**Inputs**, all in the read root and read as they are:

- The seven round-1 aggregation tables `tables/<tag>/agg_by_cell.csv`, the closed list
  `campaigns.ROUND1_AGG_TAGS` of (tag, campaign group, f): `agg_desy_f0p8153`,
  `agg_desy_f1p0`, `agg_aug_fmeas`, `agg_aug_f0p8153`, `agg_aug_f1p0`, `agg_mar_f0p9524`,
  `agg_jul_f0p9259`. Any other tag is refused (`MasterError`). Their schema is round 1's (45
  columns, section 8, versions); the chain reads its columns by name (`ROUND1_AGG_READ`) and
  never re-aggregates, since the later rules of section 8 change k_cal on the DESY May cells.
  The table of each group whose verdict is the cell's is `campaigns.ROUND1_PRIMARY_TAG`.
- `cells/<cell>/*.csv` and `*.log` of every cell of the job list: round 1's cell tables (43
  columns, `ROUND1_CELL_COLUMNS`: `cell.CELL_COLUMNS` less the seven protocol columns) and their
  job logs. A table of another schema is read with a line.
- The job list, `jobs/cells.tsv` by default (`--jobs`): each cell's level and campaign group.

**f, at two precisions** (`campaigns.F_CAL`). The verdicts carry each table's f as its folder
names it, four decimals; the recomputation uses the bootstrap's full value of the cell's group,
as the study's own recomputation did.

**Stages and outputs** (`master.OUTPUTS`, under `<write root>/master/` or `--out`):

1. recon: `recon_cells.csv`, per cell its jobs, files, exit codes, rows, target triples, arms,
   event counts, source run and the round-1 tables that hold it; `recon.json`, the totals and
   the cells of the pre-registration (`campaigns.ROUND1_PREREG`, 40 cells, and the four of
   `ROUND1_ADDED`) found and missing.
2. verdicts: `cell_verdicts.csv`, per (cell, table) the binding primary row (largest |mu| over
   the margin) and the verdict by the rules below; `f_stability.csv`, per cell its verdicts
   across the tables of its group; `cell_final.csv`, the final verdict on the group's primary
   table; `primary_rows.csv`, the primary (xfer, pair, interp) rows with Holm within each cell.
3. recompute: every cell of the job list but the Tier-0 and smoke cells, recomputed from its
   cell tables by code of its own (it imports neither `agg` nor `stats`): the n_eval floor (a
   triple goes when one of its board rows has n < 600), the symmetric censoring (a triple and
   quantity go from every arm when one arm's value is not finite or has `neg_radicand`), the
   paired delta against `self_cf`, k_cal per (cell, run, arm, kind) as in section 8, se =
   f k_cal scale, the self arms set to `self_transfer` 0, the same-pixel transfers of the same
   run and combo dropped, DerSimonian-Laird with the source term, TOST against max(1 ps, 2 % of
   the median baseline). It writes `recheck_by_group.csv` (the aggregation's grouping),
   `selfcheck_join.csv` and `selfcheck_spot.csv` (that grouping joined with the primary tables
   on (cell, arm, kind, name, self_transfer, extrap_family); a line gives the largest
   differences, it is not a gate), `alltriples_by_cell.csv` (pooled over both extrapolation
   families), and three audits: `support_by_cell.csv` (interpolating rows and triples),
   `same_pixel_audit.csv` (each cell with and without the transfers whose source triple is the
   target triple) and `spec_literal.csv` (the error from the half-split of the paired
   difference, section 4, beside the one the aggregation uses).
4. levels: `master_cells.csv`, one row per cell of `cell_final.csv`, sorted by level and cell:
   its final verdict, the all-triples headline of the pairs (`all_*`, the binding xfer pair row
   among the rows that are not self transfers, and the inverse-variance mean over the pairs)
   and of the boards (`allb_*`), the target triples and source run, the interpolating
   fraction, the verdicts across f and the audit columns (`AUDIT_COLUMNS`);
   `level_verdicts.csv`, per level (`campaigns.LEVEL_TITLES`) its counts and verdict.

**Rules** (the pre-registered rules of round 1 where the pre-registration is explicit, the
conservative reading where it is not):

- R1. The headline arm is `xfer` against the `self_cf` baseline; `self_in` is optimism only.
- R2. Pair widths are primary (section 4); the board sigma is secondary and reported.
- R3. TOST at alpha 0.05 against max(1.0 ps, 2 % of sigma(`self_cf`)), read from the table:
  PASS iff the 90 % interval lies inside the margin.
- R4. Transferability is claimed on the interpolating family only; the extrapolating family is
  reported beside it and never carries a PASS.
- R5. A cell passes only if every primary (pair, interp) row passes, an intersection-union test
  with no multiplicity correction on PASS. A guarded FAIL row takes precedence over an
  unguarded row, giving "FAIL (also unguarded on one pair)"; the pre-registration orders
  neither way, and this order follows its rule that one FAIL row makes the cell FAIL (post hoc).
- R5b. The aggregation applies Holm to a cell's board FAIL claims only; `primary_rows.csv` adds
  Holm over the primary rows within each cell as a sensitivity.
- R6. A row whose guard is not `ok` cannot pass (NA-UNGUARDED); "uninformative" means the self
  leave-pair-out cost exceeds the margin. The pre-registration states it for level e only; it is
  applied to every level (post hoc).
- R7. A cell on a thin target run (`campaigns.THIN_RUNS`) is SUPPORTING, never PASS.
- R8. A verdict that changes between the f of DESY May (0.8153, 1.0) or of DESY August (0.8979,
  0.8153, 1.0) is UNSTABLE, not PASS.
- R9. No claim below 0.45 ps: `below_noclaim` flags |mu| under it.
- R10. The placebo gate: |R_median| < 0.05 on the placebo arm (`placebo_ok`).

The level verdict: PASS when every cell passes; "PASS (supporting cells excluded from the
claim)" when the others pass; INCONCLUSIVE when every cell is unguarded or has no rows; FAIL
when a cell fails and none passes, MIXED when a cell fails and one passes; MIXED otherwise.

## 10. The axis tables

`python -m twc_transfer axes` (`axes.py`) condenses the aggregation tables behind each axis of
the study into one table per axis, the input of `etroc_plots.universality.reduce --part axes`.

**Inputs**, in the read root: `tables/<folder>/agg_by_cell.csv` and `delta_by_triple.csv` of
every folder of the closed lists `campaigns.AXIS_TAGS` (per axis: folder, the label its rows
carry as `agg_tag`, the f, whether that f is the campaign's primary one) and
`campaigns.GUARDFIX_AXIS_TAGS`; the job lists `jobs/cells_stage3.tsv` and
`jobs/cells_guardfix_rerun.tsv`. Axis A reads `agg_A_r2_desy_*` and labels its rows `A_desy_*`:
stage 3 reused the group name `A_desy`, so `tables/agg_A_desy_*` hold only its cells A8 and A9,
and the A1 to A7 rows are in the `_r2` folders, aggregated again from the stage-2 cells.

**A row** (`axis_row`, `AXIS_COLUMNS`, 24 columns) is one `agg_by_cell` row, read by name,
joined (`merge_triples`) with the mean over its target triples of `delta_by_triple`'s `cov_C`,
`lever_X` and `lever_X_all` on (cell, arm, kind, name, extrap_family); the per-row `extrap` 0/1
maps to `interp`/`extrap`, a row without one to `na`. `agg_by_cell` has a `lever_X_all` column
of its own, so the join keeps both copies under other names and the axis row's `lever_X_all` is
empty on every row, as in the study's tables. Rows are sorted by cell, arm (`ARM_ORDER`; axis F
`ARM_ORDER_F`), kind, board or pair and f. The guardfix tables add `lpo_shape` (the side-car
text of `agg_by_cell`, collapsed by `classify_lpo_shape`) and the mean `lpo_pairs` of
`delta_by_triple`, and hold only the cells of `cells_guardfix_rerun.tsv` (smoke left out).

**Outputs**, under the write root or `--out`, with the study's names:

- `results_r2/axis_<A,B,C,D,E,I,J,K>.csv`, the axes of stages 1 and 2;
- `results_r3/axis_F_long.csv` and `axis_F.csv`: axis F long, and wide (the rows of the
  primary f only, one per cell, kind, pair or board, family and tag, the arms of `WIDE_ARMS`
  side by side, and the `xfer_lin` mu minus the `xfer` mu);
- `results_r3/summary_F.md`: the cell table of axis F, `campaigns.AXIS_F_CELLS`, under the
  heading "Cells and what each compares". Each label must name exactly one level-F cell of
  `cells_stage3.tsv` and every such cell must have a label, else `AxesError`;
- with `--guardfix`, `results_r3/guardfix/axis_<E,J,K>_gf.csv`.

**The guardfix overlay** (`overlay`, as the universality reduction applies it): every row of a
cell of the guardfix table replaces that cell's rows of the axis table, placed where the cell's
first row was; the other rows keep their place; a guardfix cell absent from the axis table is
appended; `lpo_shape` and `lpo_pairs` are dropped. On the study's tables it replaces 4 cells of
E, 4 of J and 3 of K (axis E 1383 rows to 1395, J 369 to 370, K 576 to 684). `axes --guardfix`
reports the counts when the axis table is in the output folder.
