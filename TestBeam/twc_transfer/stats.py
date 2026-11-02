"""Statistics of the aggregation (SPEC.md 4): k_cal, DerSimonian-Laird random effects, TOST against
the pre-registered margin, the source-to-source random effect, Holm, the non-equivalence p.
"""
import numpy as np


def kcal_from_cells(df, value="w", by=("run",)):
    """Port of quote_resolution._calibrate to the PAIRED difference (SPEC 4 errors).

    err(delta_i) = k_cal * s_i with s_i = w_self_cf,i / sqrt(n_i); k_cal is the
    robust median of |delta_f0 - delta_f1| / s_i over triples, / (2 * 0.6745).
    """
    u = df.dropna(subset=["d_f0", "d_f1", "scale"])
    if not len(u):
        return np.nan
    r = (np.abs(u["d_f0"] - u["d_f1"]) / u["scale"]).to_numpy()
    r = r[np.isfinite(r)]
    return float(np.median(r) / (2 * 0.6745)) if len(r) else np.nan


def random_effects(delta, se):
    """DerSimonian-Laird mu, tau, se(mu) plus the fixed-effect cross-check."""
    d = np.asarray(delta, float)
    s = np.asarray(se, float)
    m = np.isfinite(d) & np.isfinite(s) & (s > 0)
    d, s = d[m], s[m]
    if len(d) == 0:
        return dict(n=0, mu=np.nan, se=np.nan, tau=np.nan, mu_fe=np.nan, se_fe=np.nan, chi2=np.nan)
    w = 1.0 / s ** 2
    mu_fe = float(np.sum(w * d) / np.sum(w))
    Q = float(np.sum(w * (d - mu_fe) ** 2))
    k = len(d)
    C = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2 = max((Q - (k - 1)) / C, 0.0) if C > 0 else 0.0
    wr = 1.0 / (s ** 2 + tau2)
    mu = float(np.sum(wr * d) / np.sum(wr))
    se_mu = float(np.sqrt(1.0 / np.sum(wr)))
    return dict(n=k, mu=mu, se=se_mu, tau=float(np.sqrt(tau2)), mu_fe=mu_fe,
                se_fe=float(np.sqrt(1.0 / np.sum(w))), chi2=Q / max(k - 1, 1),
                median=float(np.median(d)), p95=float(np.percentile(d, 95)))


def tost(mu, se, margin, alpha=0.05):
    """Two one-sided tests; PASS iff the 90 % CI lies inside +-margin."""
    from scipy.stats import norm
    if not np.isfinite(mu) or not np.isfinite(se) or se <= 0:
        return dict(verdict="NA", p=np.nan, lo=np.nan, hi=np.nan, margin=margin)
    z = norm.ppf(1 - alpha)
    lo, hi = mu - z * se, mu + z * se
    p1 = 1 - norm.cdf((mu + margin) / se)
    p2 = norm.cdf((mu - margin) / se)
    return dict(verdict="PASS" if (lo > -margin and hi < margin) else "FAIL",
                p=float(max(p1, p2)), lo=float(lo), hi=float(hi), margin=float(margin))


def margin_for(sigma_self):
    """Pre-registered equivalence margin: max(1.0 ps, 2 % of sigma_self_cf)."""
    return float(max(1.0, 0.02 * sigma_self)) if np.isfinite(sigma_self) else 1.0


def source_random_effect(delta, se, src):
    """Add the SOURCE-to-source component the (c)-(f) hierarchy needs.

    `random_effects` treats the target pixels as independent replicates, which is
    right for generalising over TARGETS but says nothing about generalising over
    SOURCES: with one source pixel the shared part of the effect ("this particular
    pixel transfers badly") is the estimand, not noise, and it carries no error bar
    at all.  Here the per-source means are formed first, their spread is the
    between-source variance, and se(mu) gains sqrt(tau_src^2 / n_src).

    Returns n_src, mu_src_spread (sd of the per-source means), se_src, and the list
    of per-source (source, n, mu).
    """
    d = np.asarray(delta, float)
    s = np.asarray(se, float)
    src = np.asarray(src, object)
    m = np.isfinite(d) & np.isfinite(s) & (s > 0)
    d, s, src = d[m], s[m], src[m]
    per = []
    for u in sorted(set(map(str, src))):
        sel = np.array([str(x) == u for x in src])
        if not sel.any():
            continue
        w = 1.0 / s[sel] ** 2
        per.append((u, int(sel.sum()), float(np.sum(w * d[sel]) / np.sum(w))))
    k = len(per)
    if k <= 1:
        return dict(n_src=k, mu_src_spread=np.nan, se_src=0.0, per_source=per)
    mus = np.array([p[2] for p in per], float)
    sd = float(np.std(mus, ddof=1))
    return dict(n_src=k, mu_src_spread=sd, se_src=sd / np.sqrt(k), per_source=per)


def holm(pvals, alpha=0.05):
    """Holm-Bonferroni step-down: returns (adjusted p, reject flag) in input order.

    Applied to the NON-equivalence (FAIL) claims only.  A set of "all three boards
    are equivalent" statements is an intersection-union test - the family-wise error
    is already controlled at alpha by requiring every component to pass - so PASS
    claims take no correction and FAIL claims do.
    """
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    idx = np.where(ok)[0]
    adj = np.full(len(p), np.nan)
    if not len(idx):
        return adj, np.zeros(len(p), bool)
    order = idx[np.argsort(p[idx])]
    k = len(order)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, float(min(1.0, (k - i) * p[j])))
        adj[j] = run
    return adj, np.nan_to_num(adj, nan=1.0) < alpha


def p_nonequiv(mu, se, margin):
    """One-sided p for the claim |mu| > margin (the evidence a FAIL asserts)."""
    from scipy.stats import norm
    if not (np.isfinite(mu) and np.isfinite(se) and se > 0):
        return np.nan
    return float(1.0 - norm.cdf((abs(mu) - margin) / se))
