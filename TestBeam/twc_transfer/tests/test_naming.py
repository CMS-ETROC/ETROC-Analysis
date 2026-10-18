"""Tests of the track-file names and the pixel keys.

Run from TestBeam/:  python -m pytest twc_transfer/tests
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "..")))
from twc_transfer import naming as N     # noqa: E402

PIX = {"dut": (14, 1), "ref": (3, 12), "extra": (0, 15)}


def test_parse_pixels_and_track_name_are_inverse():
    name = N.track_name(PIX, ["extra", "ref", "dut"])
    assert name == "track_d-R14C1_r-R3C12_e-R0C15"           # the pipeline's file order
    assert N.parse_pixels("/any/dir/" + name + ".parquet") == PIX
    pix4 = {"trig": (7, 7), "dut": (1, 2), "ref": (15, 0)}
    name4 = N.track_name(pix4, ["dut", "ref", "trig"])
    assert name4 == "track_t-R7C7_d-R1C2_r-R15C0"
    assert N.parse_pixels(name4 + ".parquet") == pix4
    assert N.track_name(PIX, ["dut"]) == "track_d-R14C1"     # only the roles asked for


def test_triple_key_and_board_pixel_key():
    assert N.triple_key(PIX) == "dut:R14C1|extra:R0C15|ref:R3C12"
    boards = N.boards_of("extra0-ref1-dut2")
    assert boards == {"extra": 0, "ref": 1, "dut": 2}
    assert N.roles_of("extra0-ref1-dut2") == ["extra", "ref", "dut"]
    assert N.board_pixel_key(PIX, boards) == "b0:R0C15|b1:R3C12|b2:R14C1"
    # the board key follows the board, not the role name: the same pixels under swapped roles
    swapped = {"ref": PIX["dut"], "dut": PIX["ref"], "extra": PIX["extra"]}
    assert N.board_pixel_key(swapped, N.boards_of("extra0-dut1-ref2")) == "b0:R0C15|b1:R3C12|b2:R14C1"


def test_ensure_dirs_needs_an_explicit_root(tmp_path):
    with pytest.raises(TypeError):
        N.ensure_dirs()
    assert N.ensure_dirs(str(tmp_path)) == str(tmp_path)
    assert sorted(os.listdir(tmp_path)) == sorted(
        ["bank", "tier0", "cells", "logs", "tables"])
