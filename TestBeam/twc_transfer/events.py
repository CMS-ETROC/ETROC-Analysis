"""Events: loading a step-10 track file, the Triple (one target pixel triple with its file-parity
folds), per-event identity and the leakage asserts of SPEC.md 1.
"""
import os
from itertools import combinations

import numpy as np
import pandas as pd

from ._core import BS
from .campaigns import N_CAP
from .naming import boards_of, parse_pixels, roles_of, triple_key


def load_events(path, roles, cap=N_CAP, seed=0, neighbor="none", neighbor_logic="OR"):
    """Step-10 time parquet -> DataFrame, pipeline cut first, then dropna, then cap.

    The cut is core/bootstrap.apply_neighbor_cut, verbatim, with the run's own
    args; every arm of every cell sees exactly these events (SPEC 1).
    """
    need = [f"{p}_{r}" for r in roles for p in ("tot", "toa")] + ["file"]
    opt = [f"cal_{r}" for r in roles] + [f"HasNeighbor_{r}" for r in roles] + ["trackNeighbor"]
    import pyarrow.parquet as pq
    have = set(pq.ParquetFile(path).schema.names)
    cols = [c for c in need + opt if c in have]
    df = pd.read_parquet(path, columns=cols)
    df = BS.apply_neighbor_cut(df, [s.strip() for s in str(neighbor).split(",")], neighbor_logic)
    df = df.dropna(subset=[c for c in need if c in df.columns]).reset_index(drop=True)
    if cap and len(df) > cap:
        df = df.sample(n=cap, replace=False, random_state=seed).sort_index().reset_index(drop=True)
    return df


class LeakageError(AssertionError):
    """SPEC 1 hard invariant violated: a source fit saw an evaluated event."""


_FNV_PRIME = np.uint64(1099511628211)
_FNV_OFFSET = np.uint64(14695981039346656037)


def event_keys(file_arr, toa, roles):
    """Per-event identity over (file index, exact TOA bit pattern of every role).

    A step-10 table carries no event id, so the identity is built from what the
    row actually contains.  The file index alone is not unique (a file holds many
    tracks) and a TOA alone is not either (two boards can coincide), but the tuple
    (file, toa_r for every role of the role set) is: TOA is a full-precision float
    in ps, so two DIFFERENT events colliding on all of them inside one run has
    probability ~n^2 * 2^-64.  Taking the raw bit pattern (`view`) rather than a
    rounded value means the key of an event is invariant under any slicing, which
    is what makes the intersection test exact.

    `roles` is the role subset the comparison is made over: a leave-pair-out source
    lives in a DIFFERENT combo with a different third board, so only the kept roles
    can be compared, and comparing on them is exactly the right test (the two boards
    that carry the information are the ones that must not be shared).
    """
    roles = tuple(sorted(roles))
    f = np.ascontiguousarray(np.asarray(file_arr, np.int64))
    h = np.full(len(f), _FNV_OFFSET, np.uint64)
    cols = [f.astype(np.uint64)]
    for r in roles:
        t = np.ascontiguousarray(np.asarray(toa[r], float))
        if len(t) != len(f):
            raise ValueError("event_keys: role %s has %d rows but `file` has %d - the "
                             "file array and the TOA arrays must be the SAME subset"
                             % (r, len(t), len(f)))
        cols.append(t.view(np.uint64))
    with np.errstate(over="ignore"):
        for c in cols:
            h = (h ^ c) * _FNV_PRIME
    return h


class Triple(object):
    """One step-10 parquet = one pixel triple = the unit of work."""

    def __init__(self, tot, toa, roles, file=None, path=None, campaign=None, run=None,
                 combo=None, cal=None, meta=None):
        self._keycache = {}
        self.roles = list(roles)
        self.tot = {r: np.asarray(tot[r], float) for r in self.roles}
        self.toa = {r: np.asarray(toa[r], float) for r in self.roles}
        self.n = len(self.toa[self.roles[0]])
        self.file = np.zeros(self.n, int) if file is None else np.asarray(file, int)
        self.fold = self.file % 2
        self.path, self.campaign, self.run, self.combo = path, campaign, run, combo
        self.cal = cal or {}
        self.meta = meta or {}
        self.boards = boards_of(combo) if combo else {r: i for i, r in enumerate(self.roles)}
        self.pix = parse_pixels(path) if path else {r: (-1, -1) for r in self.roles}
        self.key = triple_key(self.pix) if path else "synthetic"
        self.name = os.path.basename(path)[:-8] if path else "synthetic"

    # -- constructors
    @classmethod
    def from_file(cls, path, campaign, run, combo, cap=N_CAP, seed=0, neighbor="none",
                  neighbor_logic="OR", root=None):
        roles = roles_of(combo)
        df = load_events(path, roles, cap=cap, seed=seed, neighbor=neighbor, neighbor_logic=neighbor_logic)
        cal = {r: float(df[f"cal_{r}"].mean()) for r in roles if f"cal_{r}" in df.columns}
        return cls({r: df[f"tot_{r}"].to_numpy(float) for r in roles},
                   {r: df[f"toa_{r}"].to_numpy(float) for r in roles},
                   roles, file=df["file"].to_numpy(int) if "file" in df else None,
                   path=path, campaign=campaign, run=run, combo=combo, cal=cal)

    # -- views
    def idx_fold(self, f):
        return np.where(self.fold == f)[0]

    def refold_by_rank(self):
        """Re-derive the fold from the RANK of the file index among the files actually
        present, not from file % 2.

        Needed whenever a Triple is a subset that already selected on parity (the
        half-split inside the bootstrap): there, file % 2 is constant, one fold comes
        out empty and its cross-fit coefficients would silently be zeros - i.e. the
        `self_cf` arm would degrade to `notwc` without failing.
        """
        u = {v: i for i, v in enumerate(np.unique(self.file))}
        self.fold = np.array([u[v] % 2 for v in self.file], int)
        return self

    def idx_block(self, which):
        """Split the target's own events into a source half and an evaluation half.

        which in {A, blockA} / {B, blockB}
            CONTIGUOUS file blocks - the in-run stability cell of SPEC 3a.  Any
            inter-file drift (measured: 0.7-6.6 ps rms common mode) is charged
            to the transfer, so this is an upper bound on the same-config floor.
        which in {parityA, parityB}
            FILE-STRATIFIED halves, i.e. SPEC 1's own fold rule lifted to the
            source/target split: the two halves interleave in time, so the drift
            term cancels and what is left is the pure fit-noise floor.  A parity
            half has a constant file parity, so a Triple built from one MUST be
            refold_by_rank()ed before it is used as an evaluation set (otherwise
            one cross-fit fold is empty and `self_cf` silently becomes `notwc`).
        """
        if which in ("parityA", "parityB"):
            u = np.unique(self.file)
            keep = u[0::2] if which == "parityA" else u[1::2]
            return np.where(np.isin(self.file, keep))[0]
        cut = np.median(self.file)
        return np.where(self.file <= cut)[0] if which in ("A", "blockA") else np.where(self.file > cut)[0]

    def sub(self, idx):
        return ({r: self.tot[r][idx] for r in self.roles},
                {r: self.toa[r][idx] for r in self.roles})

    def rho(self):
        v = [np.corrcoef(self.tot[a], self.tot[b])[0, 1] for a, b in combinations(self.roles, 2)]
        return float(np.mean(v))

    # -- event identity (the SPEC 1 hard invariant is enforced on these)
    def keys(self, roles=None):
        """Per-event identity hashes over `roles` (default: all of them), cached."""
        rk = tuple(sorted(roles or self.roles))
        if rk not in self._keycache:
            self._keycache[rk] = event_keys(self.file, self.toa, rk)
        return self._keycache[rk]

    def key_entry(self, idx=None, roles=None):
        """(roles, key array) for a subset - what a source path hands to evaluate_arm."""
        rk = tuple(sorted(roles or self.roles))
        k = self.keys(rk)
        return (rk, k if idx is None else k[np.asarray(idx, int)])


# ------------------------------------------------------------------ leakage
def assert_disjoint(src_id, tgt_id, src_fold, tgt_fold):
    """SPEC 1 hard invariant: source fit events INTERSECT evaluated events = empty.

    There is no event id in a step-10 table, so the invariant is enforced
    structurally, which is exact here: events are partitioned by (campaign, run,
    combo, pixel triple) - a track file IS its pixel triple - and, inside one
    triple, by the file-parity fold.  Different triple/run/combo => disjoint by
    construction; same triple => the folds must differ.

    This structural test is the CHEAP half of the invariant and it is only a
    pre-filter: it fires before any event is read (so a mis-specified in-run split
    dies at once), but it cannot see a source that carries the same physical events
    under another name.  `assert_keys_disjoint` below is the real test and runs on
    every arm of every mode.
    """
    if src_id != tgt_id:
        return True
    if src_fold is None or tgt_fold is None or src_fold == tgt_fold:
        raise LeakageError("leakage: source %s fold %s would be evaluated on fold %s"
                           % (src_id, src_fold, tgt_fold))
    return True


def assert_keys_disjoint(tri, src_keys, folds=(0, 1), arm="", where=""):
    """The SPEC 1 hard invariant, enforced on real event identities.

    src_keys : {fold: [(roles_tuple, key_array), ...]} - the events the coefficients
               applied to that evaluation fold were fitted on.  A list, because a
               leave-pair-out arm has one source fit per dropped board and each is
               comparable only on the roles it shares with the target.
               None/{} means the source read no events (a bank-backed transfer); the
               caller must then satisfy `assert_disjoint` structurally instead, and
               the row records src_keys_checked = 0 so the claim is never implicit.

    Returns the number of (fold, source) pairs actually compared.
    """
    if not src_keys:
        return 0
    nchk = 0
    for f in folds:
        ent = src_keys.get(f)
        if ent is None:
            continue
        if isinstance(ent, tuple) and len(ent) == 2 and not isinstance(ent[0], (list, tuple)):
            ent = [ent]
        elif isinstance(ent, tuple):
            ent = [ent]
        m = np.where(tri.fold == f)[0] if len(folds) > 1 else np.arange(tri.n)
        for roles_used, ks in ent:
            ks = np.asarray(ks, np.uint64)
            if not len(ks) or not len(m):
                continue
            ev = tri.keys(tuple(roles_used))[m]
            n_shared = int(np.intersect1d(ev, ks).size)
            if n_shared:
                raise LeakageError(
                    "leakage: arm '%s'%s - %d of %d evaluated events in fold %d were also "
                    "in the source fit sample (roles %s)"
                    % (arm, (" [%s]" % where) if where else "", n_shared, len(ev), f,
                       ",".join(roles_used)))
            nchk += 1
    return nchk
