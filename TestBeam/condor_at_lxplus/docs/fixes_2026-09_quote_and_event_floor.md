# Quote-resolution and event-floor fixes, September 2026

This note carries the numbers behind three commits that changed `utils/quote_resolution.py`,
`core/bootstrap.py` and `submit/submit_bootstrap.py` this month. They previously lived only in the
commit messages; this puts them in the repo where the README can point at them.

## Central-hit modal offset (commit 1190dea)

`central_mask()`'s old definition of the "nominal offset" for a pixel had two independent defects,
both fixed by the same rewrite:

1. It picked the nominal offset by an unweighted per-track vote (`Series.mode()`). Every pixel has one
   partner pixel at the nominal offset but several at side offsets, so as step 7's water-fill adds
   mostly side-offset tracks with selection depth, an unweighted vote can flip to a side offset even
   though the nominal offset still carries the most events. Run 26 depth 8 showed this directly:
   central-hit ref moved to 46.46 ps on a 97 % off-nominal share, trig to 44.87 ps on 95 %, both far
   from the depth-4/16 pattern (about 44.7 / 44.2-44.5 ps).
2. It took the row-offset mode and the column-offset mode separately (`dr.mode()` and `dc.mode()`,
   ANDed) instead of the mode of the (row, col) offset pair - these need not agree, and the mismatch is
   not a weighting effect: on run 23 (canonical, depth 4) the marginal offset (5, 0) for extra-vs-trig
   in combo `extra0-trig2-dut3` disagreed with both the unweighted and the weighted joint-pair mode,
   which agreed with each other at (6, 0).

Fix: group tracks by the (row, col) offset pair and pick the one with the largest summed weight (nevt
when the step-11 CSV is present, else the same 1/err^2 fallback `illumination()` already used).

Regression, re-run of the shipped outputs after the fix (not a fresh study - central.value and .value
for every board on run 26 and run 23 are bit-identical to the pre-fix-up scratch run except where noted):
run26 dut 43.03 -> 43.47 ps, ref 44.74 -> 44.55 ps; run23 extra 44.84 -> 44.51 ps, trig 42.34 -> 42.23 ps
(both at depth 4, where the note that raised the symptom had assumed depth 4 was unaffected).
depthtest_run26_d8: ref central-hit 46.46 -> 44.56 +/- 0.03 ps (share 97 % -> 51 %), trig
44.87 -> 44.51 +/- 0.03 ps (share 95 % -> 74 %); all four boards then track d16 to within 0.05 ps.

## Event floor (commit 2a12e3a)

Width-vs-events study (run 26, notes/pipeline-pr-list.md 2026-09-07): the quoted resolution divided by
its full-sample value is 0.91 / 0.95 / 0.99 at 100 / 300 / 1000 events, and the fraction of trials
discarded for a non-positive solved variance is 40 % at 100 events versus 2 % at 300. On run 26 itself
almost nothing moves in practice at the new floor: 4422 of 4426 kept tracks already carry >= 300 events.

`core/bootstrap.py`'s own `--minimum_nevt` default moved 100 -> 300 and `submit/submit_bootstrap.py`'s
moved 1000 -> 300, so the two are no longer mismatched. That mismatch had been silent in production:
every campaign driver (`notes/scripts/run_full_chain*.sh`, `resume_chain*.sh`) passed `--minimum_nevt
100` explicitly, so the submitter's 1000 default was never actually the production floor. Pass
`--minimum_nevt 1000` to the submitter to restore its old cut.

## reg_covar floor and the joint TWC solve (commit ac97d8d)

The time-walk correction used to be a two-pass alternating loop (`for _ in range(2)`), i.e. a
fixed-point iteration whose leftover own-TOT slope after k passes is `s1 * rho^(k-1)`, with `rho` the
inter-board TOT correlation. Stopped at k = 2: at rho ~ 0.35 (heavily irradiated boards, correlated TOT)
two passes left 26-30 ps/ns of un-corrected slope and pair widths 7-16 ps too wide, while at rho ~ 0.1
the same defect is worth < 0.25 ps. The joint least-squares solve (`core/twc_solver.py`) removes the
iteration count entirely and agrees with the loop run to convergence.

Same commit added the `--gmm_reg_covar` variance floor (default 4.0 ps^2, i.e. a 2 ps sigma floor) on
every Gaussian-mixture component: the converged EM otherwise collapses a component onto one quantised
point on about 2 % of low-N fits, which then owns the PDF maximum so no half-maximum width can be read
off and the resample is silently discarded.

## Commits

- `1190dea` - quote_resolution: event-weight the central-hit modal offset, add vote-margin diagnostics,
  `--alignment` check.
- `2a12e3a` - Raise the per-track event floor from 100 to 300.
- `ac97d8d` - TWC joint least-squares solve + GMM reg_covar floor + trial bookkeeping + dual-threshold
  and pairing-covariance quote.
