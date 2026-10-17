"""Tests of the write root: never unset, never the read root, never a data root.

Run from TestBeam/:  python -m pytest twc_transfer/tests
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "..")))
from twc_transfer import campaigns as C   # noqa: E402


def test_write_root_refuses_unset_read_root_and_data_roots(monkeypatch, tmp_path):
    monkeypatch.delenv("ETROC_TWC_OUT", raising=False)
    monkeypatch.delenv("ETROC_TWC_TRANSFER_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="ETROC_TWC_OUT is not set"):
        C.write_root()
    monkeypatch.setenv("ETROC_TWC_OUT", "")
    with pytest.raises(RuntimeError, match="ETROC_TWC_OUT is not set"):
        C.write_root()
    monkeypatch.setenv("ETROC_TWC_OUT", C.transfer_root() + "/")    # the default read root
    with pytest.raises(RuntimeError, match="read root"):
        C.write_root()
    monkeypatch.setenv("ETROC_TWC_TRANSFER_ROOT", str(tmp_path / "T"))
    monkeypatch.setenv("ETROC_TWC_OUT", str(tmp_path / "T"))
    with pytest.raises(RuntimeError, match="read root"):
        C.write_root()
    for root in C.CAMPAIGNS.values():
        monkeypatch.setenv("ETROC_TWC_OUT", root)
        with pytest.raises(RuntimeError, match="data root"):
            C.write_root()
    monkeypatch.setenv("ETROC_TWC_OUT", str(tmp_path / "out"))
    assert C.write_root() == str(tmp_path / "out")
