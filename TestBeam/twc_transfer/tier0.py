"""Analytic, GMM-free Tier-0 (SPEC.md 4): the change in RMS^2 of every pair difference and every
board sigma^2 when one set of coefficients replaces another, from stored TOT moments only.
"""
from itertools import combinations


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
