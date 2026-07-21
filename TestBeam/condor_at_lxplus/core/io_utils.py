"""Shared I/O helpers for core/ and utils/ scripts.

core/ scripts import this directly (same-directory import -- works both
locally and after condor's transfer_Input_Files drops this file into the
worker sandbox alongside the main script, as long as it's added to that
script's transfer list; see build_transfer_files/COMMON_TRANSFER_FILES
below).

utils/ scripts (never shipped to condor, always run from the repo
checkout) reach it with a small path bootstrap first:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'core'))
    from io_utils import ...
"""
import getpass
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import pandas as pd


def eos_base_dir(username: Optional[str] = None) -> Path:
    """/eos/user/<first-letter>/<username> -- the implicit base every
    submit script's relative -d/-o path arguments are resolved against."""
    username = username or getpass.getuser()
    return Path(f'/eos/user/{username[0]}/{username}')


# Files every condor-transferred core/ script depends on beyond its own
# source file. Extend this list as io_utils.py grows more dependencies
# (e.g. if it starts importing another shared module).
COMMON_TRANSFER_FILES = ['core/io_utils.py']


def build_transfer_files(worker_script: str, *extra_paths: Union[str, Path]) -> str:
    """Builds a condor transfer_Input_Files value for a core/ worker script.

    Validates that every referenced file actually exists before submission,
    so a missing dependency fails loudly at submit time instead of deep
    inside a condor job's stderr log.
    """
    files = [f'core/{worker_script}', *COMMON_TRANSFER_FILES,
             *(Path(p).as_posix() for p in extra_paths)]
    missing = [f for f in files if not Path(f).is_file()]
    if missing:
        raise FileNotFoundError(f"transfer_Input_Files references missing file(s): {missing}")
    return ", ".join(files)


def _warn_if_exists(path: Path) -> None:
    if path.exists():
        prev_mtime = datetime.fromtimestamp(path.stat().st_mtime)
        logging.warning(f"Overwriting existing output: {path} (previous mtime: {prev_mtime})")


def write_parquet(df: pd.DataFrame, path: Union[str, Path], **kwargs) -> None:
    """df.to_parquet, but logs a warning instead of silently clobbering a
    pre-existing output (e.g. left over from an earlier, possibly buggy,
    run of the same stage). Passes kwargs straight through -- callers keep
    specifying index=False etc. exactly as they did before."""
    path = Path(path)
    _warn_if_exists(path)
    df.to_parquet(path, **kwargs)


def write_csv(df: pd.DataFrame, path: Union[str, Path], **kwargs) -> None:
    """df.to_csv, with the same overwrite warning as write_parquet."""
    path = Path(path)
    _warn_if_exists(path)
    df.to_csv(path, **kwargs)


def record_manifest(manifest_path: Union[str, Path], **fields) -> None:
    """Appends one JSON line to a per-run manifest.

    Call once per output file from condor-facing scripts, right after the
    write, so later you can answer "which outputs came from which
    commit/input" without mtime archaeology. Safe to call concurrently
    from many condor tasks writing into the same manifest path, since each
    call is a single append of one line.
    """
    fields.setdefault('timestamp', datetime.now(timezone.utc).isoformat())
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'a') as f:
        f.write(json.dumps(fields) + '\n')
