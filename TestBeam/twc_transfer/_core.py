"""The pipeline core this package imports, unmodified: condor_at_lxplus/core/twc_solver.py (the
joint least-squares time-walk solve) and condor_at_lxplus/core/bootstrap.py (the event cut, the
GMM width and the three-board sigma solve).

The core directory is ETROC_TWC_CORE when that is set, else this repository's
TestBeam/condor_at_lxplus/core, found from this file. It is read once, when the package is first
imported, and put on sys.path before the import because bootstrap.py imports io_utils and
twc_solver by their bare names. core_versions() gives the sha1 of every core file actually loaded,
so a result can state which core produced it (campaigns.CELLS_CORE_SHA1 holds the core the study's
published cells ran against).
"""
import hashlib
import os
import sys
import warnings

_REPO_CORE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "condor_at_lxplus", "core")


def core_dir():
    """The core directory: ETROC_TWC_CORE, else the repository's condor_at_lxplus/core."""
    return os.environ.get("ETROC_TWC_CORE") or _REPO_CORE


CORE_DIR = core_dir()
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

# thousands of GMM fits per cell: their convergence and numpy warnings are silenced, as
# bootstrap.py itself does at import
warnings.filterwarnings("ignore")

import bootstrap as BS        # noqa: E402
import twc_solver as TS       # noqa: E402

BS.logger.setLevel(40)

for _m in (BS, TS):
    if os.path.dirname(os.path.abspath(_m.__file__)) != os.path.abspath(CORE_DIR):
        raise ImportError("twc_transfer: %s was loaded from %s, not from the core directory %s "
                          "(another copy was imported first)" % (_m.__name__, _m.__file__, CORE_DIR))

KS_PMIN = BS.KS_PMIN_DEFAULT
KS_DMAX = BS.KS_DMAX_DEFAULT
ORDER = TS.TWC_ORDER


def _sha1(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def core_versions():
    """{file name: sha1} of the loaded core modules, plus "core_dir"."""
    out = {"core_dir": os.path.abspath(CORE_DIR)}
    for name in ("bootstrap", "twc_solver", "io_utils"):
        m = sys.modules.get(name)
        if m is not None and getattr(m, "__file__", None):
            out[os.path.basename(m.__file__)] = _sha1(m.__file__)
    return out
