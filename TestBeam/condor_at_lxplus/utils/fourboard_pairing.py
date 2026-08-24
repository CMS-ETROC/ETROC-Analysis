#!/usr/bin/env python3
"""Build the four-board per-quadruple pair-width table that feeds the
pairing-covariance fit in utils/quote_resolution.py.

WHY THIS EXISTS
---------------
The 3-board algebra the pipeline quotes with assumes
sigma_ab^2 = sigma_a^2 + sigma_b^2.  Measured on four-board coincidences that
additivity is rejected (p = 8e-11 and 1e-30 on the two runs studied), and the
one covariance model six pair widths can constrain - a single g shared by the
two pairs of ONE PAIRING of the four boards - fits them at chi2/dof 0.59 and
0.35, with g = +60 ps^2 (H1, 1.5e15) and +619 ps^2 (F1, 3.5e15).  Under that
model each board's three 3-board solves are v-g, v-g, v+g, so the combo-averaged
quote is low by g/3: +0.2 ps at 1.5e15, +2 ps on F1 at 3.5e15.  Full derivation,
controls and caveats: notes/fourboard-lsq-test.md.

HOW THE SAMPLE IS BUILT
-----------------------
Step 6 only ever makes leave-one-out TRIPLES, and step 9/10 rows carry no event
id, so the four-board sample is rebuilt by INTERSECTING TWO 3-BOARD COMBOS on
the raw hit codes of the two boards they share, e.g.

    A = extra0-dut1-ref2      B = extra0-dut1-trig3      shared = {extra, dut}

Two rows, one per side, with the same (file, k_shared1, k_shared2) - k being the
full raw identity (board,row,col,toa,tot,cal) of a hit packed into an int64 -
are the same event.  Keys that are not unique inside one side are dropped as
ambiguous (1-5 % of rows; the bias that costs is bounded at 0.5 ps on a pair
width, see the note).

The raw codes are recovered from step 10's own ps values by inverting its
conversion exactly: it writes `file` and the per-row raw cal_<role>, and on its
default cut-then-convert path the bin is 3.125 ns / mean(cal) over the SURVIVING
rows of that file, so the bin is recomputable from the step-10 table itself and
    toa_code = (12.5 - toa_ps/1e3) / bin ,  tot_ps/1e3/bin = 2*tot_code - floor(tot_code/32)
invert to integers.  No step-9 read is needed.  (Step 10 now also writes the
`src_row` provenance column - see core/apply_tdc_cuts.join_step9_rows - but the
inversion is kept because it works on tables written before that column existed.)

THE FIT is the pipeline's own: core/twc_solver's joint least-squares time-walk
correction over all FOUR boards, core/bootstrap's converged Gaussian mixture
(tol 1e-6, max_iter 2000, n_init 3, reg_covar floor) and its acceptance gate
(KS p >= 1e-3 OR D <= 0.03) with a retry, per pixel quadruple.  Each quadruple
is fitted on one event sample and again on its two disjoint halves; the halves
are what calibrates the pair-width errors (sd(w_n) = sd(w_h0 - w_h1)/2 under
Var ~ 1/n), without which no chi2 downstream would mean anything.

COST.  This is a per-quadruple GMM sweep - one run is tens of CPU-minutes.  Run
it under condor (or a nice'd batch), never interactively on a login node.

USAGE
  python utils/fourboard_pairing.py -t <tracks_<run>> -n <final_<run>> -o final_<run>/fourboard_pairs_<run>.csv
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import warnings
from glob import glob
from itertools import combinations
from multiprocessing import Pool

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))
import bootstrap as BS          # noqa: E402  the pipeline's GMM + acceptance gate
import twc_solver as TS         # noqa: E402  the pipeline's joint time-walk solve

BS.logger.setLevel(40)

FNORD = ("trig", "dut", "ref", "extra")     # generate_track_filename's fixed order
_TT = np.arange(512)
_LUT = 2.0 * _TT - np.floor(_TT / 32.0)     # tot_ps/1e3/bin as a function of tot_code


def die(msg):
    sys.exit("fourboard_pairing: " + msg)


def pair_key(a, b):
    return "-".join(sorted((a, b)))


def parse_combo(c):
    """'extra0-dut1-ref2' -> (['extra','dut','ref'], {'extra':0, ...})"""
    rs, bid = [], {}
    for m in re.finditer(r"([a-z]+)(\d)", c):
        rs.append(m.group(1))
        bid[m.group(1)] = int(m.group(2))
    return rs, bid


def hitkey(bd, rw, cl, toa, tot, cal):
    return ((((np.int64(bd) * 16 + np.int64(rw)) * 16 + np.int64(cl)) * 1024 + toa) * 512 + tot) * 1024 + cal


def track_name(pix, roles):
    return "track_" + "_".join("%s-R%dC%d" % (r[0], pix[r][0], pix[r][1])
                               for r in FNORD if r in roles) + ".parquet"


def raw_codes(t10, r):
    """Exact inverse of apply_tdc_cuts.convert_to_time for one role."""
    cal = t10["cal_%s" % r].to_numpy()
    binns = 3.125 / t10.groupby("file")["cal_%s" % r].transform("mean").to_numpy()
    toac = np.round((12.5 - t10["toa_%s" % r].to_numpy() / 1e3) / binns)
    totc = np.round(np.interp(t10["tot_%s" % r].to_numpy() / 1e3 / binns, _LUT, _TT))
    resid = float(np.abs((12.5 - t10["toa_%s" % r].to_numpy() / 1e3) / binns - toac).max())
    return toac.astype(np.int64), totc.astype(np.int64), cal.astype(np.int64), resid


def load_combo_file(tracks_dir, combo, roles, pix, shared, bid):
    p10 = os.path.join(tracks_dir, combo, "time", track_name(pix, roles))
    if not os.path.exists(p10):
        return None, 0.0
    t10 = pd.read_parquet(p10)
    if not len(t10):
        return None, 0.0
    d = pd.DataFrame({"file": t10["file"].to_numpy().astype(np.int32)})
    worst = 0.0
    for r in roles:
        toac, totc, cal, resid = raw_codes(t10, r)
        worst = max(worst, resid)
        if r in shared:
            d["k_%s" % r] = hitkey(bid[r], pix[r][0], pix[r][1], toac, totc, cal)
        d["toa_%s" % r] = t10["toa_%s" % r].to_numpy().astype(np.float64)
        d["tot_%s" % r] = t10["tot_%s" % r].to_numpy().astype(np.float64)
    return d, worst


# --------------------------------------------------------------------------
# the fit, per quadruple
# --------------------------------------------------------------------------
_G = {}


def _init(path, roles):
    d = pd.read_parquet(path)
    _G["by"] = {int(k): v.reset_index(drop=True) for k, v in d.groupby("quad")}
    _G["roles"] = roles


def fit_gated(x, tries=6):
    """The pipeline's acceptance gate with a retry.  If it never passes, the
    best-KS attempt is returned with gate=0 so that one stubborn pair does not
    delete a whole quadruple; the failure is recorded in the ksp sign."""
    best = None
    for _ in range(tries):
        fwhm, ks, ksp = BS.fit_gmm_and_get_fwhm(np.asarray(x, float))
        if fwhm == 0 or not np.isfinite(ksp):
            continue
        cand = dict(sigma=fwhm / 2.355, ks=ks, ksp=ksp, gate=1)
        if not (ksp < BS.KS_PMIN_DEFAULT and ks > BS.KS_DMAX_DEFAULT):
            return cand
        if best is None or ks < best["ks"]:
            cand["gate"] = 0
            best = cand
    return best


def six_widths(toa, tot, idx, roles):
    """Joint four-board time-walk correction on `idx`, then a gated fit per pair."""
    t = {r: toa[r][idx] for r in roles}
    o = {r: tot[r][idx] for r in roles}
    corr = TS.apply_timewalk_correction_arrays(t, o, roles)
    w, ksp = {}, {}
    for a, b in combinations(roles, 2):
        res = fit_gated(corr[a] - corr[b])
        if res is None:
            return None, None
        w[pair_key(a, b)] = res["sigma"]
        ksp[pair_key(a, b)] = res["ksp"] if res["gate"] else -res["ksp"]   # negative marks a gate failure
    return w, ksp


def one_quad(args):
    qid, nmin, ncap, seed = args
    roles = _G["roles"]
    df = _G["by"].get(int(qid))
    if df is None or len(df) < nmin:
        return None
    rng = np.random.default_rng(seed)
    n_all = len(df)
    take = rng.choice(n_all, min(n_all, ncap), replace=False) if n_all > ncap else np.arange(n_all)
    df = df.iloc[take].reset_index(drop=True)
    n = len(df)
    toa = {r: df["toa_%s" % r].to_numpy(float) for r in roles}
    tot = {r: df["tot_%s" % r].to_numpy(float) for r in roles}
    row = dict(quad=int(qid), n=n, n_all=n_all)
    w, ksp = six_widths(toa, tot, np.arange(n), roles)
    if w is None:
        return dict(quad=int(qid), n=n, n_all=n_all, status="fitfail")
    row["status"] = "ok"
    for k in w:
        row["w_" + k] = w[k]
        row["ksp_" + k] = ksp[k]
    # two disjoint halves: the placebo AND the source of the pair-width errors
    perm = rng.permutation(n)
    for lab, ix in (("h0", perm[: n // 2]), ("h1", perm[n // 2:])):
        wh, _ = six_widths(toa, tot, ix, roles)
        if wh is None:
            continue
        for k in wh:
            row["%s_w_%s" % (lab, k)] = wh[k]
    return row


# --------------------------------------------------------------------------
def discover_combos(tracks_dir, roles_wanted=4):
    """Two combos sharing exactly two boards, covering all four."""
    combos = sorted(os.path.basename(p) for p in glob(os.path.join(tracks_dir, "*"))
                    if os.path.isdir(os.path.join(p, "time")))
    for ca, cb in combinations(combos, 2):
        ra, _ = parse_combo(ca)
        rb, _ = parse_combo(cb)
        if len(set(ra) & set(rb)) == 2 and len(set(ra) | set(rb)) == roles_wanted:
            return ca, cb
    return None, None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-t", "--tracks-dir", required=True, dest="tracks_dir",
                   help="step-10 mother directory of one run (<combo>/time/track_*.parquet inside)")
    p.add_argument("-n", "--nevt-dir", required=True, dest="nevt_dir",
                   help="directory holding step-11's nevt_<combo>_*.csv for this run (usually final_<run>)")
    p.add_argument("-o", "--out", required=True, help="output CSV (quote_resolution reads fourboard_pairs_*.csv)")
    p.add_argument("--combo-a", default=None, dest="combo_a")
    p.add_argument("--combo-b", default=None, dest="combo_b")
    p.add_argument("--na", type=int, default=80, help="top A triples by nevt to load (default 80)")
    p.add_argument("--nmin", type=int, default=500,
                   help="minimum four-board events per quadruple to fit (default 500, the quote step's cut)")
    p.add_argument("--ncap", type=int, default=6000, help="events used per quadruple (default 6000)")
    p.add_argument("--maxq", type=int, default=0, help="fit at most this many quadruples (0 = all)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--keep-sample", default=None, dest="keep_sample",
                   help="also write the intersected four-board event table to this parquet")
    a = p.parse_args()

    t0 = time.time()
    ca, cb = a.combo_a, a.combo_b
    if ca is None or cb is None:
        ca, cb = discover_combos(a.tracks_dir)
        if ca is None:
            die("could not find two combos in %s sharing exactly two boards and covering four; "
                "pass --combo-a/--combo-b" % a.tracks_dir)
    ra, bida = parse_combo(ca)
    rb, bidb = parse_combo(cb)
    bid = dict(bida)
    bid.update(bidb)
    shared = [r for r in ra if r in rb]
    if len(shared) != 2:
        die("combos %s and %s share %d board(s), need exactly 2" % (ca, cb, len(shared)))
    ua = [r for r in ra if r not in shared][0]
    ub = [r for r in rb if r not in shared][0]
    roles4 = shared + [ua, ub]
    print("A = %s (unique %s), B = %s (unique %s), shared %s" % (ca, ua, cb, ub, shared), flush=True)

    def nevt_csv(combo):
        g = sorted(glob(os.path.join(a.nevt_dir, "nevt_%s_*.csv" % combo)))
        if not g:
            die("no nevt_%s_*.csv in %s (step 11 output)" % (combo, a.nevt_dir))
        return g[0]

    nA = pd.read_csv(nevt_csv(ca)).sort_values("nevt", ascending=False)
    nB = pd.read_csv(nevt_csv(cb))
    selA = nA.head(a.na)
    shc = [c for r in shared for c in ("row_%s" % r, "col_%s" % r)]
    shkeys = set(map(tuple, selA[shc].to_numpy()))
    selB = nB[[tuple(x) in shkeys for x in nB[shc].to_numpy()]]
    print("A triples %d (%d evt), B triples %d (%d evt)"
          % (len(selA), selA["nevt"].sum(), len(selB), selB["nevt"].sum()), flush=True)

    def pixof(row, roles):
        return {r: (int(row["row_%s" % r]), int(row["col_%s" % r])) for r in roles}

    worst = 0.0
    sides = {}
    for side, sel, combo, roles in (("a", selA, ca, ra), ("b", selB, cb, rb)):
        parts, ids = [], []
        for _, row in sel.iterrows():
            pix = pixof(row, roles)
            d, w = load_combo_file(a.tracks_dir, combo, roles, pix, shared, bid)
            worst = max(worst, w)
            if d is None:
                continue
            d["i%s" % side] = np.int32(len(ids))
            ids.append({"row_%s" % r: pix[r][0] for r in roles} | {"col_%s" % r: pix[r][1] for r in roles})
            parts.append(d)
        if not parts:
            die("no step-10 track files found for combo %s under %s" % (combo, a.tracks_dir))
        sides[side] = (pd.concat(parts, ignore_index=True), pd.DataFrame(ids))
    A, aid = sides["a"]
    B, bidf = sides["b"]
    print("loaded A %d rows / B %d rows  (max |toa_code - int| = %.2e)  %.0fs"
          % (len(A), len(B), worst, time.time() - t0), flush=True)

    kcols = ["file"] + ["k_%s" % r for r in shared]
    dupA, dupB = A.duplicated(kcols, keep=False), B.duplicated(kcols, keep=False)
    print("ambiguous keys dropped: A %.3f %%  B %.3f %%" % (100 * dupA.mean(), 100 * dupB.mean()), flush=True)
    M = A[~dupA].merge(B[~dupB][kcols + ["ib", "toa_%s" % ub, "tot_%s" % ub]], on=kcols, how="inner")
    if not len(M):
        die("the two combos have no common events after the key join")
    print("four-board rows %d (%.3f of A)" % (len(M), len(M) / max(len(A), 1)), flush=True)

    M = M.merge(aid.reset_index().rename(columns={"index": "ia"})[["ia", "row_%s" % ua, "col_%s" % ua]], on="ia")
    M = M.merge(bidf.reset_index().rename(columns={"index": "ib"})[["ib", "row_%s" % ub, "col_%s" % ub]], on="ib")
    M["quad"] = M.groupby(["ia", "ib"]).ngroup().astype(np.int32)
    q = M.groupby("quad").agg(ia=("ia", "first"), ib=("ib", "first"), n=("quad", "size")).reset_index()
    for r in roles4:
        q = q.merge(M.groupby("quad")[["row_%s" % r, "col_%s" % r]].first().reset_index(), on="quad")
    print("quadruples %d;  n>=%d: %d" % (len(q), a.nmin, int((q["n"] >= a.nmin).sum())), flush=True)

    keep = ["quad"] + ["%s_%s" % (v, r) for r in roles4 for v in ("toa", "tot")]
    sample_path = a.keep_sample or (os.path.splitext(a.out)[0] + "_sample.parquet")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    M[keep].to_parquet(sample_path, index=False, compression="lz4")

    qs = q[q["n"] >= a.nmin].sort_values("n", ascending=False)
    if a.maxq:
        qs = qs.head(a.maxq)
    jobs = [(int(x), a.nmin, a.ncap, 90000 + int(x)) for x in qs["quad"]]
    print("fitting %d quadruples with %d worker(s)" % (len(jobs), a.workers), flush=True)
    rows = []
    if jobs:
        with Pool(a.workers, initializer=_init, initargs=(sample_path, roles4)) as pool:
            for i, r in enumerate(pool.imap_unordered(one_quad, jobs)):
                if r is not None:
                    rows.append(r)
                if (i + 1) % 10 == 0:
                    print("   %d/%d  %.0fs" % (i + 1, len(jobs), time.time() - t0), flush=True)
    out = pd.DataFrame(rows)
    if len(out):
        out = out.merge(q, on="quad", suffixes=("", "_q"))
    out.to_csv(a.out, index=False)
    if not a.keep_sample:
        os.remove(sample_path)
    print("wrote %s (%d quadruples, %.0fs)" % (a.out, len(out), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
