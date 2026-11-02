"""Universality figures: can a time-walk correction (TWC) fitted on one source (a run, a board,
a pixel) be applied to another target without losing time resolution? transfer reads the
study's tables and holds the arithmetic the figures share, curves draws the TWC curves, and
`python -m etroc_plots.universality.reduce` makes the tables. Import submodules directly; no
re-exports by design."""
from ..campaigns import check_required

check_required("universality")
