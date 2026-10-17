"""The coefficient bank (SPEC.md 2, "fit once, evaluate many"): the bank rows of one triple
(coefficients per board, fold and guard, TOT quantiles, moments and the 25-ps TOT histogram),
the leave-pair-out source events, bank lookups, and the TOT overlap metrics of SPEC.md 5.3 from
events or from bank rows alone.
"""
import glob
import hashlib
import json
import os
import time
from itertools import combinations

import numpy as np
import pandas as pd

from ._core import ORDER, TS
from .campaigns import N_CAP, NAME_ORDER, ROLE_INIT
from .events import load_events
from .naming import roles_of, run_dir, sibling_combo
from .solve import apply_c, coeff_cov, raw_to_centred, solve


def tot_stats(t):
    t = np.asarray(t, float)
    q = np.percentile(t, [0.5, 1, 5, 25, 50, 75, 95, 99, 99.5])
    m = [float(np.mean(t ** k)) for k in (1, 2, 3, 4)]
    lo = float(np.floor(t.min() / 25.0) * 25.0)
    hi = float(np.ceil(t.max() / 25.0) * 25.0)
    nb = max(int(round((hi - lo) / 25.0)), 1)
    h, _ = np.histogram(t, bins=nb, range=(lo, hi))
    return dict(tot_min=float(t.min()), q005=q[0], q01=q[1], q05=q[2], q25=q[3], q50=q[4],
                q75=q[5], q95=q[6], q99=q[7], q995=q[8], tot_max=float(t.max()),
                mean_tot=float(t.mean()), sd_tot_emp=float(t.std()),
                m1=m[0], m2=m[1], m3=m[2], m4=m[3], hist_lo=lo, hist=json.dumps(h.tolist()))


def pair_moments(tri):
    """E[t_a t_b], E[t_a^2 t_b], E[t_a t_b^2], E[t_a^2 t_b^2] for every pair (Tier-0)."""
    out = {}
    for a, b in combinations(tri.roles, 2):
        ta, tb = tri.tot[a], tri.tot[b]
        out["%s-%s" % (a, b)] = [float(np.mean(ta * tb)), float(np.mean(ta ** 2 * tb)),
                                 float(np.mean(ta * tb ** 2)), float(np.mean(ta ** 2 * tb ** 2))]
    return out


def resid_moments(tri, coeffs):
    """Moments of the SELF-SOLVE residual against each board's TOT (Tier-0 exactness).

    The joint least-squares residual is orthogonal to the design only in the SUM
    over the pairs a board takes part in, not pair by pair: measured on real data,
    the analytic Var-only prediction reproduces the SUM of the three
    pair changes exactly and each individual pair only up to the cross term.  Storing
    E[r], E[r*t], E[r*t^2] per pair per board makes Tier-0 exact pair by pair too.
    """
    out = {}
    corr = {r: apply_c(tri.toa[r], tri.tot[r], coeffs[r]) for r in tri.roles}
    for a, b in combinations(tri.roles, 2):
        r_ab = corr[a] - corr[b]
        out["%s-%s" % (a, b)] = dict(
            e0=float(np.mean(r_ab)),
            ea1=float(np.mean(r_ab * tri.tot[a])), ea2=float(np.mean(r_ab * tri.tot[a] ** 2)),
            eb1=float(np.mean(r_ab * tri.tot[b])), eb2=float(np.mean(r_ab * tri.tot[b] ** 2)))
    return out


# ------------------------------------------------------------------- LPO sources
def lpo_events(tri, drop_role, cap=N_CAP, root=None, neighbor="none", neighbor_logic="OR", seed=0):
    """Events of the sibling combo that OMITS `drop_role` and keeps this triple's
    other two pixels (SPEC 5.1: the leave-pair-out guard).

    The fourth board's pixel is free, so every matching file is concatenated.
    Returns (tot, toa, file, roles) or None when the sibling combo is absent.
    """
    sib = sibling_combo(tri.campaign, tri.run, tri.combo, drop_role, root)
    if sib is None:
        return None
    sroles = roles_of(sib)
    keep = [r for r in tri.roles if r != drop_role]
    inv = {v: k for k, v in ROLE_INIT.items()}
    toks = []
    for r in NAME_ORDER:
        if r not in sroles:
            continue
        if r in keep:
            toks.append("%s-R%dC%d" % (inv[r], tri.pix[r][0], tri.pix[r][1]))
        else:
            toks.append("%s-R*C*" % inv[r])
    pat = os.path.join(run_dir(tri.campaign, tri.run, root), sib, "time", "track_" + "_".join(toks) + ".parquet")
    files = sorted(glob.glob(pat), key=os.path.getsize, reverse=True)
    if not files:
        return None
    frames, tot_n = [], 0
    for f in files:
        d = load_events(f, sroles, cap=0, seed=seed, neighbor=neighbor, neighbor_logic=neighbor_logic)
        if len(d) == 0:
            continue
        frames.append(d)
        tot_n += len(d)
        if tot_n >= cap:
            break
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    if cap and len(df) > cap:
        df = df.sample(n=cap, replace=False, random_state=seed).sort_index().reset_index(drop=True)
    return (dict((r, df["tot_%s" % r].to_numpy(float)) for r in sroles),
            dict((r, df["toa_%s" % r].to_numpy(float)) for r in sroles),
            df["file"].to_numpy(int) if "file" in df else np.zeros(len(df), int),
            sroles, sib, len(df))


# ------------------------------------------------------------------ the coeff bank
BANK_META = ("campaign", "run", "angle_deg", "telescope", "combo", "src_triple", "board_id",
             "role", "fold", "guard", "placebo", "src_kind")
# every column of a bank row, in the order bank_rows_for_triple writes them: identity,
# coefficients and their metadata, then the TOT summary of tot_stats
BANK_COLUMNS = (
    "campaign", "run", "angle_deg", "telescope", "combo", "src_triple", "src_file",
    "board_id", "role", "fold", "guard", "placebo", "src_kind",
    "c2", "c1", "c0", "mu_tot", "sd_tot", "b2", "b1", "b0", "cov_c", "n_fit",
    "rho_pairwise", "cal_mean", "hv", "temp", "row", "col", "pair_moments", "resid_moments",
    "order", "solver", "git_sha", "ts", "neighbor", "neighbor_logic",
    "tot_min", "q005", "q01", "q05", "q25", "q50", "q75", "q95", "q99", "q995", "tot_max",
    "mean_tot", "sd_tot_emp", "m1", "m2", "m3", "m4", "hist_lo", "hist",
)
def bank_rows_for_triple(tri, folds="parity", guards=("full",), placebo=False, root=None,
                         angle_deg=np.nan, telescope="", neighbor="none", neighbor_logic="OR",
                         hv=np.nan, temp=np.nan, git_sha=""):
    """Every coefficient row this triple contributes to the bank (SPEC 2)."""
    rows = []
    fsets = {"full": np.arange(tri.n)}
    if folds in ("parity", "2", 2):
        fsets["0"] = tri.idx_fold(0)
        fsets["1"] = tri.idx_fold(1)
    elif folds == "block":
        fsets["blockA"] = tri.idx_block("A")
        fsets["blockB"] = tri.idx_block("B")
    elif folds == "both":
        fsets.update({"0": tri.idx_fold(0), "1": tri.idx_fold(1),
                      "blockA": tri.idx_block("A"), "blockB": tri.idx_block("B")})
    pm = pair_moments(tri)
    rm_json = json.dumps(resid_moments(tri, solve(tri.tot, tri.toa, tri.roles)))
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")

    def emit(fold, guard, idx, coeffs, roles_used, placebo_flag, n_fit, tots_used, toas_used, src_kind="track"):
        try:
            cov, mu, sd = coeff_cov(tots_used, toas_used, roles_used, coeffs)
        except Exception:
            cov, mu, sd = {r: np.full((3, 3), np.nan) for r in roles_used}, {}, {}
        for r in roles_used:
            if r not in tri.roles:
                continue
            c = np.asarray(coeffs[r], float)
            tstat = tot_stats(tri.tot[r][idx] if idx is not None and len(idx) else tri.tot[r])
            mu_r = mu.get(r, float(np.mean(tots_used[r])))
            sd_r = sd.get(r, float(np.std(tots_used[r])) or 1.0)
            b = raw_to_centred(c, mu_r, sd_r)
            row = dict(campaign=tri.campaign, run=tri.run, angle_deg=angle_deg, telescope=telescope,
                       combo=tri.combo, src_triple=tri.key, src_file=tri.name,
                       board_id=tri.boards.get(r, -1), role=r, fold=fold, guard=guard,
                       placebo=int(placebo_flag), src_kind=src_kind,
                       c2=c[0], c1=c[1], c0=c[2], mu_tot=mu_r, sd_tot=sd_r,
                       b2=b[0], b1=b[1], b0=b[2],
                       cov_c=json.dumps(np.asarray(cov[r], float).ravel().tolist()),
                       n_fit=int(n_fit), rho_pairwise=tri.rho(),
                       cal_mean=tri.cal.get(r, np.nan), hv=hv, temp=temp,
                       row=tri.pix.get(r, (-1, -1))[0], col=tri.pix.get(r, (-1, -1))[1],
                       pair_moments=json.dumps(pm), resid_moments=rm_json,
                       order=ORDER, solver="twc_solver",
                       git_sha=git_sha, ts=ts, neighbor=neighbor, neighbor_logic=neighbor_logic)
            # the TOT summary of the events this board's coefficients will be compared
            # against: quantiles, the four raw moments (Tier-0) and the 25-ps histogram
            # (the overlap metrics).  Without it the bank cannot feed tier0 or
            # source_from_bank, so it is merged in here rather than recomputed later.
            row.update(tstat)
            rows.append(row)

    for fold, idx in fsets.items():
        if len(idx) < 50:
            continue
        tots, toas = tri.sub(idx)
        if "full" in guards:
            c = solve(tri.tot, tri.toa, tri.roles, idx=idx)
            emit(fold, "full", idx, c, tri.roles, 0, len(idx), tots, toas)
        if placebo:
            # content-addressed seed (blake2b of the track name and the fold): the
            # placebo does not depend on Python's per-process hash() salt
            rng = np.random.default_rng(
                int(hashlib.blake2b(repr((tri.name, fold)).encode(), digest_size=8).hexdigest(), 16))
            c = solve(tri.tot, tri.toa, tri.roles, idx=idx, permute=True, rng=rng)
            emit(fold, "full", idx, c, tri.roles, 1, len(idx), tots, toas)
        if "lpo" in guards:
            for drop in tri.roles:
                got = lpo_events(tri, drop, cap=N_CAP, root=root, neighbor=neighbor,
                                 neighbor_logic=neighbor_logic)
                if got is None:
                    continue
                ltot, ltoa, lfile, lroles, sib, nl = got
                if fold in ("0", "1"):
                    m = np.where((lfile % 2) == int(fold))[0]
                elif fold in ("blockA", "blockB"):
                    cut = np.median(lfile)
                    m = np.where(lfile <= cut)[0] if fold == "blockA" else np.where(lfile > cut)[0]
                else:
                    m = np.arange(len(lfile))
                if len(m) < 50:
                    continue
                sub_t = {r: ltot[r][m] for r in lroles}
                sub_a = {r: ltoa[r][m] for r in lroles}
                c = TS.solve_timewalk_coefficients(sub_t, sub_a, lroles)
                emit(fold, "lpo:%s" % drop, idx, c, [r for r in lroles if r in tri.roles], 0,
                     len(m), sub_t, sub_a)
    return rows


def bank_lookup(bank, **kw):
    """Rows of the bank matching every keyword (exact match, NaN-safe)."""
    d = bank
    for k, v in kw.items():
        if v is None:
            continue
        d = d[d[k] == v]
    return d


def coeffs_from_row(row):
    return np.array([row["c2"], row["c1"], row["c0"]], float)


# ------------------------------------------------------------- overlap / coverage
def overlap_metrics(src_row, tot_tgt, c_src):
    """SPEC 5.3 - coverage C, histogram intersection O, shift, sd ratio, lever X."""
    t = np.asarray(tot_tgt, float)
    lo, hi = float(src_row["q005"]), float(src_row["q995"])
    C = float(np.mean((t >= lo) & (t <= hi)))
    out_mask = (t < lo) | (t > hi)
    if out_mask.any():
        w_raw = np.polyval(c_src, t[out_mask])
        w_clip = np.polyval(c_src, np.clip(t[out_mask], lo, hi))
        X = float(np.sqrt(np.mean((w_raw - w_clip) ** 2)))      # SPEC 5.3, verbatim
    else:
        X = 0.0
    # X is an rms over the OUT-OF-SUPPORT events only (SPEC 5.3).  Measured: the
    # source window [q0.5, q99.5] discards 1 % of the source by construction, so an
    # identically distributed target still has ~1 % outside it and a quadratic
    # evaluated 1 % into a TOT tail always moves by tens of ps - X > 5 ps therefore
    # fires on EVERY cell, the in-run self-transfer floor included, which makes the
    # flag carry no information.  X_all is the
    # same rms taken over ALL target events (= sqrt(frac_out) * X), i.e. the actual
    # ps the extrapolation can move this target, and it is what the flag uses.
    # Both travel; X is never dropped.
    frac_out = float(out_mask.mean())
    X_all = X * float(np.sqrt(frac_out))
    hs = np.asarray(json.loads(src_row["hist"]), float)
    slo = float(src_row["hist_lo"])
    edges_s = slo + 25.0 * np.arange(len(hs) + 1)
    tlo = float(np.floor(t.min() / 25.0) * 25.0)
    thi = float(np.ceil(t.max() / 25.0) * 25.0)
    ht, edges_t = np.histogram(t, bins=max(int(round((thi - tlo) / 25.0)), 1), range=(tlo, thi))
    glo, ghi = min(edges_s[0], edges_t[0]), max(edges_s[-1], edges_t[-1])
    nb = max(int(round((ghi - glo) / 25.0)), 1)
    gs = np.zeros(nb)
    gt = np.zeros(nb)
    i0 = int(round((edges_s[0] - glo) / 25.0))
    gs[i0:i0 + len(hs)] = hs
    j0 = int(round((edges_t[0] - glo) / 25.0))
    gt[j0:j0 + len(ht)] = ht
    gs = gs / max(gs.sum(), 1.0)
    gt = gt / max(gt.sum(), 1.0)
    O = float(np.minimum(gs, gt).sum())
    mu_s, sd_s = float(src_row["mean_tot"]), float(src_row["sd_tot_emp"]) or 1.0
    shift = (float(t.mean()) - mu_s) / sd_s
    sdr = (float(t.std()) or 1.0) / sd_s
    return dict(cov_C=C, hist_O=O, shift=shift, sd_ratio=sdr, lever_X=X,
                lever_X_all=X_all, frac_out=frac_out,
                extrap=int(C < 0.98 or X_all > 5.0))


def overlap_from_bank(src_row, tgt_row, c_src):
    """SPEC 5.3 overlap metrics from BANK ROWS ONLY - no event is re-read.

    Tier-0 is the GMM-free exhaustive screen, so its coverage/extrapolation numbers
    have to come out of the stored 25-ps TOT histograms and quantiles rather than
    the target's event array.  C, O, shift and sd_ratio are then exact up to the
    25-ps binning; X and X_all are the same rms as `overlap_metrics` with the
    out-of-support events replaced by their bin centres weighted by the bin counts.
    """
    lo, hi = float(src_row["q005"]), float(src_row["q995"])
    ht = np.asarray(json.loads(tgt_row["hist"]), float)
    tlo = float(tgt_row["hist_lo"])
    ctr = tlo + 25.0 * (np.arange(len(ht)) + 0.5)
    tot_w = max(ht.sum(), 1.0)
    inside = (ctr >= lo) & (ctr <= hi)
    C = float(ht[inside].sum() / tot_w)
    out = ~inside
    frac_out = float(ht[out].sum() / tot_w)
    if ht[out].sum() > 0:
        w_raw = np.polyval(c_src, ctr[out])
        w_clip = np.polyval(c_src, np.clip(ctr[out], lo, hi))
        X = float(np.sqrt(np.sum(ht[out] * (w_raw - w_clip) ** 2) / ht[out].sum()))
    else:
        X = 0.0
    X_all = X * float(np.sqrt(frac_out))
    hs = np.asarray(json.loads(src_row["hist"]), float)
    slo = float(src_row["hist_lo"])
    glo = min(slo, tlo)
    ghi = max(slo + 25.0 * len(hs), tlo + 25.0 * len(ht))
    nb = max(int(round((ghi - glo) / 25.0)), 1)
    gs, gt = np.zeros(nb), np.zeros(nb)
    gs[int(round((slo - glo) / 25.0)):][:len(hs)] = hs
    gt[int(round((tlo - glo) / 25.0)):][:len(ht)] = ht
    gs, gt = gs / max(gs.sum(), 1.0), gt / max(gt.sum(), 1.0)
    O = float(np.minimum(gs, gt).sum())
    sd_s = float(src_row["sd_tot_emp"]) or 1.0
    shift = (float(tgt_row["mean_tot"]) - float(src_row["mean_tot"])) / sd_s
    sdr = (float(tgt_row["sd_tot_emp"]) or 1.0) / sd_s
    return dict(cov_C=C, hist_O=O, shift=shift, sd_ratio=sdr, lever_X=X,
                lever_X_all=X_all, frac_out=frac_out,
                extrap=int(C < 0.98 or X_all > 5.0))
