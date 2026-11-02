"""Front-end figures: the autocalibration baseline and noise width of every board (blnw), and the
raw TDC codes of one pixel per chip (tdc_codes). `python -m etroc_plots.frontend.rebuild` makes
their tables. Import submodules directly; no re-exports by design."""
from ..campaigns import check_required

check_required("style")
check_required("frontend")
