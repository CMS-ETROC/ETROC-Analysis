# Step 6 alignment feed-forward: one board per 4-board telescope was never aligned (fixed 2026-09-01)

## Symptom

On the DESY August 2026 campaign (four boards, 60 degrees, offset holder, every yaml translation zero)
every run produced exactly one resolution table, `dut0-trig1-ref2`. The three combos containing board 3
reached step 12 with hundreds of time files each and produced zero bootstrap files. Compared with an
independent processing of the same raw data (pre-fix code), our `trig1-ref2-extra3` step-8 output held
3,049 rows against 356,050 (run 23, 45 files each) while our `dut0-trig1-ref2` was healthy.

## Cause

`core/path_finder.py --find_alignment` estimates each non-trigger board's translation relative to the
trigger board, per combo. Since commit f7d135a the estimate was applied to the in-memory run config
**once, from the first trigger-containing combo**, and never again (a run-level `alignment_applied`
latch). That combo holds only its own boards. On a telescope of N >= 4 boards every generated combo has
N-1 boards, so the anchor combo covers N-2 of the N-1 non-trigger boards and **exactly one board keeps
its `--config` translation for the whole run**. With an all-zero yaml that board is matched against an
aim point 10 pixels away from where its hits are, inside a 4-pixel window, so only accidental
coincidences survive.

Evidence, DESY Aug run 23: the step-6 log applied `{0: (0.296, 5.532), 2: (-0.14, -6.102)}` and nothing
for board 3, whose own estimate in the alignment yaml is `(-0.68, -13.167)` = 10.1 px. Reconstructing the
radius cut from the surviving candidates gives a maximum radius of exactly 4.00 px with board 3 at zero
translation. Raw hits with no coincidence selection show board 3 illuminated in 83 % of events, ten rows
from the trigger board. The survivors' pattern purity (sum of counts over number of patterns) was 1.01 on
every board-3 combo against 4.24 on the healthy one: once-only patterns, i.e. background.

A second defect with the same root: when the trigger is the highest-numbered board (IRRAD, trig = 3)
the lexicographically first combo `(0,1,2)` contains no trigger and was cut before any estimate existed,
on raw geometry.

## How this landed in main

The feed-forward described below is opt-in on main, behind `--apply_alignment` (which requires
`--find_alignment`). Nothing changes for anyone who does not pass it: `--find_alignment` alone stays
diagnostic-only and writes the same `legacy_per_combo` + `global_relative` yaml it always has, the
coincidence window keeps its main-line anchor (the trigger board, else the combo's own median board
id), the combo processing order is unchanged, and the track output of a run without the new flag is
byte-identical to before. With `--apply_alignment` the per-board registry, the trigger-first ordering,
the sub-pixel estimator, the "no board has role trig" error, `--alignment_core_warn` and the `applied:`
block all switch on together, and the two diagnostic blocks keep being measured against the `--config`
geometry so their numbers mean the same thing in both modes. `utils/quote_resolution.py --alignment`
reads the `applied:` block when it is there and rejects a diagnostic-only yaml with a message saying
why; `utils/telescope_diagnostics.py` reads `applied:` in either layout.

The estimator is the one place where the two branches genuinely disagree rather than merely differing
in scope. The diagnostic blocks use a 30-bin histogram of the continuous shift, whose bin width is
about one pixel pitch on a full-grid candidate set, so their values are quantised to +/-0.65 mm; the
applied value uses the count-weighted modal pixel plus a sub-pixel centroid. That quantisation is
tolerable in a number a human reads out of a yaml and is not tolerable in a number fed back into the
geometry, so the diagnostic blocks were left exactly as they are and `--apply_alignment` carries its
own estimator. Expect the two to differ by up to half a pixel for the same board.

## Fix (all in `core/path_finder.py`)

1. **Per-board registry instead of a run-level latch.** `alignment_source[board] = combo` records
   which combo supplied each board's translation; a board is aligned exactly once, from the first
   trigger-containing combo that contains it. Re-estimating a board on top of its own applied value -
   the order dependence f7d135a was written to remove - is now structurally unreachable.
2. **Trigger-containing combos are processed first** (a stable sort of the unchanged combo list), so by
   the time any combo is cut every one of its boards has its translation. Output file names and the
   `--combos` index order are unchanged.
3. **`--find_alignment` with no board of role `trig` is an error** (exit 1) instead of a silent no-op that
   wrote no estimate, no yaml and no warning.
4. **`--alignment_core_warn` (default 0.10, warn-only).** When less than this share of a combo's
   candidate weight sits within +/-1 pixel of the modal shift, the applied value is flagged as possibly
   the combinatorial-background mode rather than the beam spot (this happens when a board sits outside
   the trigger's acceptance, e.g. DESY May runs 5-7 board 0). 0 disables. The floor is conservative, not
   calibrated - set it per campaign.

The alignment yaml gains an `applied:` block recording, per board, the value applied and the combo it
came from. `utils/telescope_diagnostics.py` reads both layouts.

Net code change: +60 / -12 lines; the rest of the diff is comments. Nothing changes for a 3-board
telescope, for a run without `--find_alignment`, or for a telescope whose yaml already carries correct
translations (the estimate then measures a residual of ~0).

## Validation

Step 6 re-run on DESY Aug run 23 with the same seed (42) and inputs into a fresh directory:

| check | result |
|---|---|
| board 3 applied from `dut0-trig1-extra3` | exactly `(-0.68, -13.167)`, the value the old yaml had recorded |
| `tracks_dut0-trig1-ref2.parquet` (anchor combo) | byte-identical to the pre-fix file (md5 `e571db1e7464`) |
| `trig1-ref2-extra3` candidates | 1,840 patterns at purity 1.01 -> 7,603 at purity 5.37 |
| warnings | none |

Independently: the two combos that measure board 3 agree to 0.2 mm; the value is constant to 0.05 mm
across all 19 August runs (one geometry). IRRAD March/July: the orphaned board's translation is a
constant 0.3-0.8 px of mounting offset, identical to 0.01 mm over 85 runs; the 11 July runs where it
reaches 2.0-2.3 px coincide exactly with the documented halo-illumination state (both telescopes, same
run blocks), i.e. a change in mean track angle, not geometry.

## What has to be reprocessed (from step 6)

- DESY Aug 2026: all 19 runs (5-23).
- DESY May 2026: runs 5, 6, 7 and 41 only (yaml translations zero at 60 degrees); the other 31 processed
  runs carry correct yaml translations and are unaffected (residual < 0.7 px).
- IRRAD March/July 2026: nothing. The effect is below one pixel except in the 11 halo-state runs, where
  it costs a few per cent of acceptance edge on the board-2 combos; left as is.

Re-runs must go to a fresh output tree: the chain drivers skip any step whose outputs already exist and
would otherwise mix pre- and post-fix results in one chain.

## Practice going forward

Measure the telescope geometry once per holder configuration, write the translations into the board
config yaml (as the May 2026 yaml already does, scaling with tan(angle)), and keep `--find_alignment` as
a residual check that should read ~0. The derived correction is then a safety net, not the source of
the geometry.

For August 2026 that is now done, in a SEPARATE file rather than by editing the campaign config:
`board_configs_yaml/DESY_TB_2026Aug_corrected.yaml` rides alongside `DESY_TB_2026Aug.yaml`, which is left
untouched so nobody's existing command changes behaviour. Pass the corrected file with `-c` to supply the
geometry; the original stays the default. It differs in exactly two ways: the runs 5-23 translations
(board 0 x +0.300 y +5.451, board 2 x -0.174 y -6.117, board 3 x -0.617 y -13.303 mm, board 1 being the
trigger at the origin), and the defaults `angle` 30 -> 60, which August ran at throughout. The values are
the median of 38 independent per-combo estimates over all 19 runs of the post-fix reprocessing, spread
0.03-0.28 mm across the campaign, i.e. one geometry. Runs 2-4 keep their zero translations and their
`rotation y: +30`: they are the wrong-orientation, undecoded runs, and guessing their orientation was
deliberately avoided.

A caveat on the guard this fix added: `--alignment_core_warn` at its 0.10 default did NOT fire on a
genuine background latch (DESY May 2026 run 5, board 0, whose anchor-combo estimate is ~13 mm from the
value every other combo measures). The warning that DID fire, and the one to trust, is the existing
cross-combo disagreement message - `board N already aligned from (combo) as ...; this combo re-measures
... (max residual X mm)`. Treat a large residual there as the signal that a board's applied translation is
not to be believed, and supply that run's geometry instead.

## Validation of the reconciled code (2026-09-10)

Branch merge-check-2026-09-10 at f5c0d86 (main plus this PR), step 6 re-run interactively with the same
flags as the shipped runs (`--max_diff_pixel 4 -s 10 --seed 42`):

- DESY Aug run 23, `--find_alignment --apply_alignment`, original DESY_TB_2026Aug.yaml: the `applied:` block
  is byte-identical to the shipped analysis_v2 yaml (boards 0 and 2 from dut0-trig1-ref2, board 3 from
  dut0-trig1-extra3 at x -0.680 y -13.167 mm), and all four track-candidate files match the shipped ones by
  md5 (dut0-trig1-ref2 14805 rows, dut0-trig1-extra3 3750, trig1-ref2-extra3 7603, dut0-ref2-extra3 3336).
  The trig-less combo matches too, so dropping the branch's radius cut in favour of main's combo-anchored
  spatial check changed nothing on this data.
- DESY Aug run 23, `--find_alignment` only: yaml and all four track files byte-identical to origin/main
  (c463a76) run with the same command.
- CERN IRRAD March H1 run 12, `--find_alignment --max_files 20`: yaml and all four track files
  byte-identical to origin/main.

Scratch outputs: /eos/user/m/musafdar/scratch_mergecheck_20260910/{A,B,C,D}_*.
