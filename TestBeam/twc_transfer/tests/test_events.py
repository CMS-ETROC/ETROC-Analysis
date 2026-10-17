"""Tests of the per-event identity and the leakage asserts on a synthetic triple.

Run from TestBeam/:  python -m pytest twc_transfer/tests
"""
import os
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "..")))
from twc_transfer import events as E     # noqa: E402

ROLES = ["extra", "ref", "dut"]


def toy_triple(n=400, seed=3):
    """400 events in 40 files of 10; the path is only parsed for its pixels, never read."""
    rng = np.random.default_rng(seed)
    tot = {r: rng.uniform(1500.0, 5000.0, n) for r in ROLES}
    toa = {r: rng.normal(10000.0, 40.0, n) for r in ROLES}
    return E.Triple(tot, toa, ROLES, file=np.arange(n) // 10,
                    path="track_d-R14C1_r-R14C1_e-R14C1.parquet",
                    campaign="toy", run="toy", combo="extra0-ref1-dut2")


def _fnv(file_index, toas):
    """The key by hand: 64-bit FNV over the file index and the TOA bit patterns."""
    h = 14695981039346656037
    for c in [file_index] + [struct.unpack("=Q", struct.pack("=d", t))[0] for t in toas]:
        h = ((h ^ c) * 1099511628211) % 2 ** 64
    return h


def test_assert_disjoint_raises_on_an_overlap():
    with pytest.raises(E.LeakageError):
        E.assert_disjoint("dut:R1C1|extra:R2C2|ref:R3C3", "dut:R1C1|extra:R2C2|ref:R3C3", 0, 0)
    with pytest.raises(E.LeakageError):
        E.assert_disjoint("k", "k", None, 1)
    assert E.assert_disjoint("k", "k", 0, 1) is True
    assert E.assert_disjoint("k1", "k2", 0, 0) is True


def test_assert_keys_disjoint_raises_on_one_shared_event():
    tri = toy_triple()
    f0, f1 = tri.idx_fold(0), tri.idx_fold(1)
    assert E.assert_keys_disjoint(tri, {0: [tri.key_entry(f1)], 1: [tri.key_entry(f0)]}) == 2
    assert E.assert_keys_disjoint(tri, None) == 0
    with pytest.raises(E.LeakageError):
        E.assert_keys_disjoint(tri, {0: [tri.key_entry(f0)]})
    with pytest.raises(E.LeakageError):
        E.assert_keys_disjoint(tri, {1: [tri.key_entry(np.r_[f0, f1[:1]])]}, arm="xfer")


def test_event_keys_are_stable_and_slice_invariant():
    tri = toy_triple()
    k = E.event_keys(tri.file, tri.toa, tri.roles)
    copies = {r: tri.toa[r].copy() for r in tri.roles}
    np.testing.assert_array_equal(E.event_keys(tri.file.copy(), copies, ROLES[::-1]), k)
    np.testing.assert_array_equal(tri.keys(), k)
    np.testing.assert_array_equal(tri.keys(), k)              # the cached second call
    idx = np.arange(0, tri.n, 7)
    np.testing.assert_array_equal(
        E.event_keys(tri.file[idx], {r: tri.toa[r][idx] for r in tri.roles}, tri.roles), k[idx])
    assert len(np.unique(k)) == tri.n
    for i in (0, 1, tri.n - 1):
        assert int(k[i]) == _fnv(int(tri.file[i]), [tri.toa[r][i] for r in sorted(ROLES)])
    with pytest.raises(ValueError):
        E.event_keys(tri.file[:-1], tri.toa, tri.roles)
