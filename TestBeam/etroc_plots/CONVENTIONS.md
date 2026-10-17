# Figure conventions

Every figure made with this package follows these rules. `checks.py` tests the ones marked
(checked): the layout rules as each figure is drawn, with the findings written into its values
file, and the top band on the saved PNG (`python3 -m etroc_plots.checks`, see `README.md`). The
rest are for whoever writes the next figure.

## Layout

- Header (checked): "CMS" and the campaign's experiment text (`EXP_TEXT`, "ETL ETROC IRRAD" for
  IRRAD 2026) alone on the left, immediately above the axes. Everything else goes on the right, at
  most two lines: the facility line on top, the subject (telescope or panel, conditions, data)
  below. No left titles. `style.header` and `style.compound_header` write it.
- Footer (checked): selection, vetoes and counts go in a footer, in a band of its own below the
  axes (`style.footer`, then `style.lower_footer` adds the 0.5 in band), so it can be cropped away
  without touching the plot.
- Top band (checked): at most 60 px of blank above the topmost ink, at 200 dpi.
- Text (checked): no text touches an axes frame, overlaps other text or runs off the canvas.
  An annotation arrow is not text: it may reach its target, a frame included, but it never
  crosses another text. A mark drawn outside the axes (a flag tick, a symbol above a panel) is
  tagged `_etroc_mark` so the audit sees it: it never crosses a text and never touches another
  mark.
- No callbacks (checked): no figure or slide numbers, notes folders, script or notebook names,
  or facility and home paths (/eos, /afs, /store, home folders) on a figure. It has to stand on its
  own wherever it is shown.
- State the selection on the image itself: guards, vetoes and counts.
- Titles fit the width of the figure.

## Words

- Fluence, never dose: 24 GeV protons, in p/cm2. Say which fluence a figure draws: the bookkeeping
  fluence (the irradiation facility's number) or the fluence on chip (bookkeeping times the
  dosimetry factor, `CHIP_FLUENCE_FACTOR` in the campaign module).
- After irradiation the sensors are cooling down; never "annealing".
- Preamp gain or charge-to-threshold, never a bare "gain".
- Chips by name (IH7, LF10), never by their role in a run.

## Colour and markers

- Colour = fluence, one ladder per campaign (`FLUENCE_COLOR` in the campaign module). Within one
  fluence step, later measurements of the same chips (after cooling down, for instance) take hues
  of their own from `iv_plot.TIMING_COLORS`, never a lighter or darker shade of the step's
  colour. Six exceptions: the V_gl fit figures (`iv/vgl_plot.py`: fluence is their x axis, one
  fit per board), the baseline / noise-width figures (`frontend/blnw.py`: the fluence step, the
  measurement before and after cooling down or the RFSel category is their x axis) and the board
  TOT figures (`tot_shapes.board_vs_fluence` in `frontend/tot_shapes.py`: the fluence step is
  their x axis) colour by board (`etroc_style.BOARD_COLOR`), and the bias-current timelines drawn
  one line per chip (`iv/preirrad_current.py`, `iv/current_vs_run.py` and the current-limit-hold
  figures of the IV notebook) colour by chip; there the fluence is a shaded band or a label,
  never a line colour.
  The RFSel comparison (`tot_shapes.rfsel_pairs`: runs of one fluence step that differ in RFSel)
  colours by RFSel (`etroc_style.RFSEL_COLOR`, no orange, which is a fluence colour), and so do the
  settings figures (below). The schematic pulses of the RFSel explainer use a grey ramp of their
  own (`cartoons.RF_GREY`): they are not measured series.
- Marker and line style = chip, by its slot in the telescope. Legends name chips only; the vendor
  goes in the panel title.

## Resolution and TDC figures

- Every figure or panel showing resolution or TDC data states the front-end operating point in its
  title: RFSel with the feedback resistor, the discriminator threshold as baseline + offset, and the
  preamp power. `style.settings_text` writes it.
- A board's value comes from one 3-board combination, the reference combination: boards 0-1-2
  for boards 0, 1 and 2, boards 1-2-3 for board 3. It is fixed, the same in every run, so that a
  board has one number on every figure. Where the value depends on the combination
  (`BAND_STEPS` in the campaign module: 3.5e15 for IRRAD 2026), the board is drawn as a band: a
  shaded box over its value in every combination containing it, the marker at the box's centre,
  the bar the reference combination's pixel spread (`resolution.tables.band`). The footer says so
  (`tables.BAND_FOOTER`). A figure that compares the combinations themselves shows each
  combination's value and no band. A figure built on maps pooled over every combination (the
  merge cost, figure 7 of the merged notebook) says so in its footer.

- Baseline and noise width are drawn with both statistics over a board's pixels: all 256 pixels as
  the scan stored them, and the pixels with a value (`frontend/blnw.py` says which); the figure
  states how many pixels the second leaves out. The bar is the pixel-to-pixel spread.
- Distributions of TDC codes, TOT and TOA difference are drawn per bin as the fraction of the
  curve's events, with the bin width on the axis label: one code for codes (CAL, and the
  single-pixel TOT of fe10, fe13 and fe14, whose ns axis carries the code ladder of the pixel's
  CAL mode), 0.3 ns for the TOA and 0.2 ns for the TOT of fe03-fe04, 0.1 ns for the pooled TOT,
  25 ps (about one TOA code) for the TOA difference.
- A pixel's TOT and TOA difference come from one triple of pixels: among the track files of its
  board's reference combination, the one with the most events, counted when it holds at least 300
  (`tot_shapes.FLOOR`). Nothing is pooled across the triples of a pixel. The TOT depends on which
  triple is read (the partner pixels fix where the track crosses the pixel), so a figure of TOT
  says which triple it took.

## Settings figures

- The settings figures (`resolution/settings.py`, the settings notebook) draw the time resolution
  at one fluence step against the front-end settings its runs stepped: the preamp feedback setting
  RFSel and the threshold offset. Colour = RFSel (`etroc_style.RFSEL_COLOR`), marker = chip; the
  fluence step is in the header.
- A run sits in the slot of its settings class (RFSel, offset) from the display-run list, each
  board in the cell of its own offset (`runs_summary.csv`) where the axis is the offset: a run
  whose boards sit at two offsets (class "8/10") stands in both cells. The runs of a slot stand
  side by side in run order, the boards of a run in board order. Under each run: its number, bold
  for the preferred run of its class, and the time from the end of the step's irradiation to the
  run start; the bias goes in the footer. The runs of one class differ in both, and the figures
  and the notebook say so.
- Nothing is drawn unless `settings.settings_guard` passes: the display list, `runs_summary.csv`
  and the autocalibration scan before the run agree on every board's RFSel and on the offsets.
- A class whose runs on a telescope include every run of an accepted merge (the merged runs'
  board table: `runs_merged`, `accepted`) draws the merge in place of those runs on the
  resolution-against-class and x_B figures (st02, st04, st05): one band per board from the merged
  table by the rule of the single runs, labelled with its runs joined by "+"
  (`tables.classes_at_step`, `settings.class_points`). Its runs each had their own
  autocalibration scan: its x_B and bar pool the pixels of its runs' scans before the median and
  the percentiles, a scan the runs share once (`settings.pixel_x_over_nw`). The figures of one
  scan per run (st06) keep the runs and say so in the footer.
- A board whose resolution comes from fewer than `tables.LOW_PIXELS` pixels (128, half the chip:
  the board table's `n_pixels` of the anointed combination, `settings.low_pixels`) is a small
  open marker on st02, st04 and st05, where the notebook sets `p["low"]`; the footer gives the
  rule, how many boards and each one's count. On st04 no band stands behind the marker, so its
  edge keeps the RFSel colour.
- The threshold over the noise of board B in a run is x_B = offset / median noise width: its
  threshold offset over the median noise width of its live pixels (noise width above 0, the
  pixels of `n_pixels_nw`) from the autocalibration scan before the run, both in DAC codes
  (`settings.pixel_x_over_nw`: the offset from `blnw_by_run.csv`, the noise width of every pixel
  from `blnw_by_pixel.csv.gz`). Its bar runs from the 16th to the 84th percentile of offset /
  noise width over the same pixels: a spread, not an uncertainty. The noise width is an integer
  DAC, so the medians are coarse and the bars asymmetric; they are drawn as they are.
- Threshold over the noise (`settings.threshold_over_noise`): x_B of every board with its bar,
  x = chip, the settings classes side by side in each chip's slot. Colour = RFSel as everywhere,
  but the marker is the offset class (`settings.offset_markers`, in offset order), not the chip:
  the chip is already the x axis, and the marker then tells the offset apart where two classes
  share an RFSel. The only settings figure whose marker is not the chip.
- Resolution against x_B (`settings.res_vs_x`): the band of every board at its x_B, with the
  x_B bar as an x bar in the RFSel colour (the 16-84 % range of offset / noise width over the
  pixels, not an error on x_B); the y bar stays the resolution's pixel-to-pixel spread.
- Baseline and noise width against RFSel (st06): `frontend.blnw.ladder_vs_fluence` with the RFSel
  categories on the x axis in place of the fluence steps, colour = board as in the front-end
  figures; both statistics (all 256 pixels, and the live pixels) are drawn, the live one named
  with the range of pixels it leaves out. The runs of a category stand side by side at one pitch
  for every category (`blnw.points(pitch=...)`).
- What RFSel changes (st07, `overview.cartoons.rfsel_measured`): the block diagram, one measured
  RFSel pair (`frontend.tot_shapes.rfsel_curves`, the pixels counted in both runs) and the design
  lines with their sources; the measured lines under them are computed in the notebook (median TOT
  shift per pair, the chip's resolution in the runs, the time after the irradiation and the bias
  of both runs). A design line without a source in the design description is left out and listed
  in the values file.
- The setting is named "preamp feedback setting RFSel r (R_f = k kOhm)" (`style.rfsel_text`
  writes the RFSel part); what it changes is the preamp gain. Never "gain mode", "high gain" or
  "low gain".

## Pixel maps

- Every map states its frame on the figure. The resolution maps (`resolution/pixel_maps.py`) and
  the CAL-code maps (`frontend/tdc_codes.py`) are drawn in the chip frame: both axes inverted,
  pixel (0, 0) at the bottom right.
- One colour scale for every map of a notebook (0-120 ps for the IRRAD 2026 resolution maps,
  `VMIN` and `VMAX` in `resolution/pixel_maps.py`), so maps compare by colour; a pixel outside it
  is counted as saturated, a pixel without a value is grey. A map of a difference (the CAL-code
  change of the front-end notebook) takes its own symmetric range, set from the data by one rule
  (`DIFF_RANGE` in `frontend/tdc_codes.py`, pooled over both telescopes by default, so the
  telescope with the smaller change reads nearly flat); the footer states the rule and the range.

## Schematics

- The setup and method figures of the overview notebook draw no measured data. Every number on
  them (sensor names, beam width, event counts) comes from the campaign module, the notebook or
  `overview/cartoons.py` (its illustrative tracks and pulses) and is listed in the values file.
  A figure that could be read as data says so on the image: the beam profile is a model, the
  track counts and the pulse shapes are illustrative.
- They carry no fluence, so their colours mean other things: the board index
  (`etroc_style.BOARD_COLOR`), the track chosen for a pixel, R_f on a grey ramp, and the RFSel
  switch in the alert colour.

## IV figures

- Currents are drawn as abs(I). A binned scan is the mean voltage and the median current per
  voltage bin (the median rejects beam-spill spikes): 10 V bins for quick scans, down to 0.1 V at
  low voltage for fine scans (`FINE_BINS` and `QUICK_BINS` in `iv/legacy.py`). The figure inputs
  this package bins from raw logs (`legacy.binned_table`, `iv_data.load_july_fine`) keep only each
  channel's up-sweep, since the ramps down before and after it pass the same voltages again; the
  general helpers `build_iv_curve` and `kfactor.plot_kfactor` bin every reading in the window they
  are given.
- Full-range IV figures share the x range 0-620 V (`iv_plot.XLIM_FULL`); every linear V_gl axis
  spans 0-55 V (`YLIM_LINEAR` in `iv/vgl_plot.py`), and a log view sets its own range.
- Scan times are UTC; the March raw logs were recorded in local time (CET, UTC+1).

## Time-walk transfer figures

- A transfer is scored on pairs of boards: the penalty is the width of a pair with the
  transferred correction minus its width with the target's own, on the same events, in ps. Axis
  titles say pair penalty (or pair width), never a board's resolution. Intervals are 90 %
  (mu +- 1.645 se); each row's own margin is drawn behind it; the verdicts are the study's, after
  the override `transfer.read_master` applies: a cell whose leave-pair-out check (a refit without
  the other board of the pair) is missing or cannot decide, and that is not a FAIL, reads "no
  leave-pair-out check".
- A pair is named by the chips of its two boards in the target run; the study's role words stay
  in the values file.
- Comparisons that touch an IRRAD run at a fluence step in `UNIV_EXCLUDED_STEPS` (the campaign
  module; 2e15 and 3.5e15 for IRRAD 2026) are not drawn. Where this leaves something out, the
  figure or the notebook says what, as a fact about the data (at which steps those comparisons
  exist), with the counts left out.
- The TOT under a correction curve is drawn in 100 ps bins (four 25 ps bins of the coefficient
  bank summed; one TOT code is about 40 to 45 ps, so finer bins show the comb of the codes), the
  bin on the axis label.

## Values files

Next to every figure, `<stem>_values.json` holds the numbers drawn, the input files, the commit of
the code, a conventions block (units, binning, selection) and the audit findings. A figure is only
as reproducible as this file: when a number shown on the figure is not in it, add it.

## Data quality

- A figure drawn from IRRAD 2026 readout hits (merged hits, track files) ends its footer with
  the campaign's `READOUT_FOOTER`, which sends the reader to the README's "Data quality"
  section.
- A figure drawn from the IRRAD 2026 resolution tables, whose only count of readout data is a
  pixel count, ends its footer with the one line "Pixel counts are the surviving readout
  sample." (`SURVIVING_TEXT` in its notebook: the maps notebook).
- A figure drawn from autocalibration scans, from IV or slow-control logs, or from DESY data
  only carries neither.
