"""TWC transfer: can a time-walk correction (TWC) fitted on one sample, the source, be applied to
another, the target, without losing time resolution? The source can be the same pixels in another
run or at another angle, another pixel of the same board, another board, or a fit pooled over a
board.

The package measures this on the pipeline's own terms. The joint least-squares TWC solve
(condor_at_lxplus/core/twc_solver.py), the event cut, the GMM width (FWHM/2.355) and the
three-board sigma solve (condor_at_lxplus/core/bootstrap.py) are imported unmodified; the only new
primitive is the pipeline's apply line, toa + polyval(c, tot), and the bookkeeping around it.

The unit of work is one target pixel triple: one step-10 track file of one run and combo, its
events split into two folds by file parity. Every arm (no correction, the target's own in-sample
and cross-fitted coefficients, the leave-pair-out guard, the permuted-TOT placebo, the transferred
coefficients and their variants) is a map of coefficients applied to the same frozen events, so
arms differ only in the coefficients. Source fits never see an evaluated event; that is asserted
on per-event identities. The coefficient bank stores the coefficients of every triple with the TOT
moments and histograms that let the Tier-0 screen and the TOT overlap metrics run without
re-reading events. Statistics: k_cal from the fold half-split of the paired difference,
DerSimonian-Laird random effects, TOST against max(1 ps, 2 % of the baseline sigma), Holm across
non-equivalence claims.

Modules (imported one by one; this package re-exports nothing):

  campaigns   step-10 data roots per campaign, the read root, the write root (ETROC_TWC_*)
  _core       the pipeline core: where it is loaded from, and the sha1 of what was loaded
  naming      combos, roles, pixel triples, run directories, track ranking
  events      loading a track file, Triple, per-event identity, the leakage asserts
  solve       the solve, the gauge, the apply line, basis changes, transfer variants
  bank        bank rows of a triple, the leave-pair-out source events, TOT overlap metrics
  widths      the width, the three-board sigma, the evaluation of one arm, controls
  arms        the arms' coefficient maps
  tier0       the analytic change of RMS^2 per pair and board
  stats       k_cal, random effects, TOST, the margin, Holm

SPEC.md beside this file defines the data model, the bank, the estimator and the controls;
README.md says how to run it. etroc_plots never imports this package: it reads its output tables.
"""
