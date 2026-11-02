"""Analytic, GMM-free Tier-0 (SPEC.md 4): the change in RMS^2 of every pair difference and every
board sigma^2 when one set of coefficients replaces another, from stored TOT moments only, and
the screen of a whole bank file built on it (`bank_screen`, the `tier0` command).
"""
import json
from itertools import combinations

import numpy as np

from .bank import coeffs_from_row, overlap_from_bank


def _var_quad(d, m):
    """Var[d2 t^2 + d1 t + d0] from the stored raw moments m1..m4."""
    d2, d1 = float(d[0]), float(d[1])
    m1, m2, m3, m4 = m
    v_t = m2 - m1 ** 2
    v_t2 = m4 - m2 ** 2
    c_t2_t = m3 - m2 * m1
    return d2 ** 2 * v_t2 + d1 ** 2 * v_t + 2 * d1 * d2 * c_t2_t


def _cov_quad(da, db, ma, mb, cross):
    """Cov[da(t_a), db(t_b)] from marginal and cross moments (SPEC 4 Tier-0)."""
    a2, a1 = float(da[0]), float(da[1])
    b2, b1 = float(db[0]), float(db[1])
    e_ab, e_a2b, e_ab2, e_a2b2 = cross
    c_tt = e_ab - ma[0] * mb[0]
    c_t2t = e_a2b - ma[1] * mb[0]
    c_tt2 = e_ab2 - ma[0] * mb[1]
    c_t2t2 = e_a2b2 - ma[1] * mb[1]
    return a2 * b2 * c_t2t2 + a2 * b1 * c_t2t + a1 * b2 * c_tt2 + a1 * b1 * c_tt


def tier0_delta(dcoef, moments, cross, roles, resid=None):
    """Analytic change in RMS^2 of every pair difference and every board sigma^2.

    dcoef   : {role: c_src - c_self_in} (raw TOT)
    moments : {role: (m1,m2,m3,m4)} of the TARGET's TOT
    cross   : {'a-b': (E[ab],E[a2b],E[ab2],E[a2b2])} of the TARGET
    resid   : optional {'a-b': {e0,ea1,ea2,eb1,eb2}} from resid_moments() - without
              it the prediction is exact only for the SUM over the three pairs
              (the least-squares residual is orthogonal to the design only in that
              sum); with it, exact pair by pair (measured on real data).
    """
    V = {r: _var_quad(dcoef[r], moments[r]) for r in roles}
    C = {}
    for a, b in combinations(roles, 2):
        key = "%s-%s" % (a, b)
        C[key] = _cov_quad(dcoef[a], dcoef[b], moments[a], moments[b], cross[key])
    pair = {}
    for a, b in combinations(roles, 2):
        key = "%s-%s" % (a, b)
        pair[key] = V[a] + V[b] - 2 * C[key]
        if resid and key in resid:
            q = resid[key]
            da, db = dcoef[a], dcoef[b]
            e_a = float(da[0]) * q["ea2"] + float(da[1]) * q["ea1"] + float(da[2]) * q["e0"]
            e_b = float(db[0]) * q["eb2"] + float(db[1]) * q["eb1"] + float(db[2]) * q["e0"]
            m_a = float(da[0]) * moments[a][1] + float(da[1]) * moments[a][0] + float(da[2])
            m_b = float(db[0]) * moments[b][1] + float(db[1]) * moments[b][0] + float(db[2])
            mean_r = q["e0"]
            pair[key] += 2.0 * ((e_a - e_b) - mean_r * (m_a - m_b))
    board = {}
    for r in roles:
        o = [q for q in roles if q != r]
        pk = lambda x, y: pair.get("%s-%s" % (x, y), pair.get("%s-%s" % (y, x), 0.0))
        board[r] = 0.5 * (pk(r, o[0]) + pk(r, o[1]) - pk(o[0], o[1]))
    return pair, board, V


def bank_screen(bank, max_src=0):
    """The Tier-0 screen of one bank frame (SPEC 4 Tier-0): every target triple against every
    other source triple of the same combo, with its floor and its TOT coverage.

    Targets and sources are the bank's full-guard, non-placebo, fold "full" rows; a source
    triple is used only when it has every role of the target. Per board and per pair:

      dvar     the signed analytic change in RMS^2 (tier0_delta, with the residual moments)
      P        sqrt(max(dvar, 0)), the curve distance; P_signed keeps the sign of dvar
      P_null   the same statistic between the target's own two split-half self fits (folds 0
               and 1): what two independent fits of the same thing differ by
      P_exc    sqrt(max(P^2 - P_null^2, 0)), the distance in excess of that floor
      own_var  (board rows) the variance of the coefficient change on the target's TOT
      cov_C, hist_O, lever_X, lever_X_all, extrap
               SPEC 5.3 from the stored 25-ps TOT histograms (overlap_from_bank), so the
               screen reads no event; a pair takes the worse of its two boards

    max_src > 0 keeps the first max_src source triples of the combo (sorted by track name).
    Returns a pandas DataFrame, one row per (target, source, kind, name).
    """
    import pandas as pd
    b = bank[(bank.guard == "full") & (bank.placebo == 0) & (bank.fold == "full")]
    halves = bank[(bank.guard == "full") & (bank.placebo == 0) & (bank.fold.isin(["0", "1"]))]
    rows = []
    for (run, combo, tgt), g in b.groupby(["run", "combo", "src_file"]):
        roles = list(g["role"])
        gi = g.set_index("role")
        mom = {r: (float(gi.loc[r, "m1"]), float(gi.loc[r, "m2"]),
                   float(gi.loc[r, "m3"]), float(gi.loc[r, "m4"])) for r in roles}
        cross = {k: tuple(v) for k, v in json.loads(g.iloc[0]["pair_moments"]).items()}
        rm = json.loads(g.iloc[0]["resid_moments"]) if "resid_moments" in g.columns else None
        c_self = {r: coeffs_from_row(gi.loc[r]) for r in roles}

        # the floor: the target's own two split-half self fits, differenced
        null_pair, null_board = {}, {}
        h = halves[(halves.run == run) & (halves.combo == combo) & (halves.src_file == tgt)]
        h0 = h[h.fold == "0"].set_index("role")
        h1 = h[h.fold == "1"].set_index("role")
        if set(h0.index) >= set(roles) and set(h1.index) >= set(roles):
            dn = {r: coeffs_from_row(h0.loc[r]) - coeffs_from_row(h1.loc[r]) for r in roles}
            null_pair, null_board, _ = tier0_delta(dn, mom, cross, roles, resid=rm)

        srcs = b[(b.combo == combo)]
        if max_src:
            srcs = srcs[srcs.src_file.isin(sorted(srcs.src_file.unique())[:max_src])]
        for (srun, sfile), sg in srcs.groupby(["run", "src_file"]):
            if sfile == tgt and srun == run:
                continue
            sgi = sg.set_index("role")
            cs = {r: coeffs_from_row(sgi.loc[r]) for r in sgi.index}
            if set(cs) != set(roles):
                continue
            d = {r: cs[r] - c_self[r] for r in roles}
            pair, board, V = tier0_delta(d, mom, cross, roles, resid=rm)
            ovb = {}
            for r in roles:
                try:
                    ovb[r] = overlap_from_bank(sgi.loc[r], gi.loc[r], cs[r])
                except Exception:
                    ovb[r] = {}

            def emit(kind, name, dv, nullv, ov):
                P = np.sqrt(max(dv, 0.0))
                Pn = np.sqrt(max(nullv, 0.0)) if np.isfinite(nullv) else np.nan
                rows.append(dict(
                    run=run, combo=combo, tgt=tgt, src_run=srun, src=sfile, kind=kind,
                    name=name, dvar=dv, P=P,
                    P_signed=float(np.sign(dv) * np.sqrt(abs(dv))),
                    P_null=Pn, P_exc=(np.sqrt(max(P ** 2 - Pn ** 2, 0.0))
                                      if np.isfinite(Pn) else np.nan),
                    own_var=V.get(name, np.nan) if kind == "board" else np.nan,
                    cov_C=ov.get("cov_C", np.nan), hist_O=ov.get("hist_O", np.nan),
                    lever_X=ov.get("lever_X", np.nan),
                    lever_X_all=ov.get("lever_X_all", np.nan),
                    extrap=ov.get("extrap", np.nan)))

            for r in roles:
                emit("board", r, board[r], null_board.get(r, np.nan), ovb.get(r, {}))
            for k, v in pair.items():
                x, y = k.split("-")
                o = {}
                if ovb.get(x) and ovb.get(y):
                    o = dict(cov_C=min(ovb[x]["cov_C"], ovb[y]["cov_C"]),
                             hist_O=min(ovb[x]["hist_O"], ovb[y]["hist_O"]),
                             lever_X=max(ovb[x]["lever_X"], ovb[y]["lever_X"]),
                             lever_X_all=max(ovb[x]["lever_X_all"], ovb[y]["lever_X_all"]),
                             extrap=max(ovb[x]["extrap"], ovb[y]["extrap"]))
                emit("pair", k, v, null_pair.get(k, np.nan), o)
    return pd.DataFrame(rows)
