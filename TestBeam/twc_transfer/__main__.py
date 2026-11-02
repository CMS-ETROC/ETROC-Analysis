"""python -m twc_transfer <subcommand>: the command line (cli.py) under the study's run rules.

Before anything numeric is imported, every thread pool is pinned to one thread (the width
fits are single-threaded by design; a free pool only adds load), and the interpreter is
restarted once with PYTHONHASHSEED=0 when the variable holds anything else, saying so on
stderr. Draws seeded from Python's hash() of a string are reproducible only with the hash
salt fixed; the study's condor jobs and bank fits ran with PYTHONHASHSEED=0 (Python 3.9.12,
LCG_104d; the same seeds under other Python versions are not verified).

Run from TestBeam/ (or with TestBeam/ on PYTHONPATH): the restart runs `-m twc_transfer` again
in the same folder with the same environment.
"""
import os
import sys

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "ARROW_NUM_THREADS")


def pin_threads(environ):
    """Every thread-pool variable set to 1, as the study's condor jobs had them."""
    for k in THREAD_VARS:
        environ[k] = "1"


def ensure_hashseed(environ, argv, execv=os.execv, executable=None):
    """Restart the interpreter with PYTHONHASHSEED=0 unless it is already 0.

    Returns False when nothing was needed. Otherwise it sets the variable in `environ` (with
    os.environ the restarted process inherits it), writes one line to stderr and replaces the
    process with `executable -m twc_transfer argv[1:]` through `execv`, so it returns only when
    `execv` does (a test's stand-in)."""
    was = environ.get("PYTHONHASHSEED")
    if was == "0":
        return False
    environ["PYTHONHASHSEED"] = "0"
    executable = executable or sys.executable
    sys.stderr.write("twc_transfer: PYTHONHASHSEED was %s; restarting with PYTHONHASHSEED=0\n"
                     % ("unset" if was is None else repr(was)))
    sys.stderr.flush()
    execv(executable, [executable, "-m", "twc_transfer"] + list(argv[1:]))
    return True


def main():
    pin_threads(os.environ)
    ensure_hashseed(os.environ, sys.argv)
    from twc_transfer.cli import main as cli_main
    return cli_main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
