"""Time-resolution helpers: the result tables and the selections the figures share. Import
submodules directly; no re-exports by design."""
from ..campaigns import check_required

check_required("style")
check_required("resolution")
