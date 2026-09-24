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
  crosses another text.
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
  fluence step, later looks at the same chips (after cooling down, for instance) take hues of their
  own from `iv_plot.TIMING_COLORS`, never a lighter or darker shade of the step's colour. Two
  exceptions: the V_gl fit figures (`iv/vgl_plot.py`: fluence is their x axis, one fit per board)
  colour by board, and the bias-current timelines drawn one line per chip (`iv/preirrad_current.py`,
  `iv/current_vs_run.py` and the current-limit-hold figures of the IV notebook) colour by chip;
  there the fluence is a shaded band or a label, never a line colour.
- Marker and line style = chip, by its slot in the telescope. Legends name chips only; the vendor
  goes in the panel title.

## Resolution and TDC figures

- Every figure or panel showing resolution or TDC data states the front-end operating point in its
  title: RFSel with the feedback resistor, the discriminator threshold as baseline + offset, and the
  preamp power. `style.settings_text` writes it.

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

## Values files

Next to every figure, `<stem>_values.json` holds the numbers drawn, the input files, the commit of
the code, a conventions block (units, binning, selection) and the audit findings. A figure is only
as reproducible as this file: when a number shown on the figure is not in it, add it.
