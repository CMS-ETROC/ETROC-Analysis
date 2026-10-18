"""The round-1 chain (SPEC.md 9): from the round-1 aggregation tables and cell tables to the
table of the level cells, master_cells.csv, and the verdict of each level.

The study's round 1 (job list cells.tsv) aggregated its cells into seven tables, one per
campaign group and f (campaigns.ROUND1_AGG_TAGS, a closed list), and then ran four stages on
them. This module is those four stages, in order, each a function:

  recon      per cell: its jobs, files, exit codes, rows, triples, arms, event counts, the
             aggregation tables that hold it -> recon_cells.csv, recon.json
  verdicts   per (cell, table): the binding primary row and the verdict by the rules R1-R10
             below -> cell_verdicts.csv; per cell the verdict across the f variants
             -> f_stability.csv; the final verdict on each group's primary f -> cell_final.csv;
             the primary rows with Holm within each cell -> primary_rows.csv
  recompute  an independent recomputation of every round-1 cell from its cell tables (it
             re-implements the aggregation's rules here and imports neither agg.py nor
             stats.py): the self-check against the agg tables (recheck_by_group.csv,
             selfcheck_join.csv, selfcheck_spot.csv), the delta pooled over both
             extrapolation families (alltriples_by_cell.csv) and three audits:
             support_by_cell.csv (interpolating rows and triples), same_pixel_audit.csv (every
             cell with and without the transfers whose source triple is the target triple),
             spec_literal.csv (the error from the half-split of the paired difference,
             SPEC 4, beside the error the aggregation uses)
  levels     the per-cell master table -> master_cells.csv, and per level -> level_verdicts.csv

Each stage reads the tables of the stages before it from the output folder, as written, as
the study's scripts did; run() runs the four in order.

Inputs, all in the read root (campaigns.transfer_root()), read as they are:

  tables/<tag>/agg_by_cell.csv  the seven round-1 tables. Their schema is round 1's: 45
                                columns, agg.AGG_COLUMNS without placebo_minus_notwc1b_ps,
                                which round 1 did not write; the chain reads its columns by
                                name (ROUND1_AGG_READ) and never re-aggregates (agg.py's later
                                rules change k_cal on the DESY May cells, README parity)
  cells/<cell>/*.csv, *.log     the cell tables and job logs of every cell of the job list:
                                round 1's cell schema, 43 columns (ROUND1_CELL_COLUMNS)
  the job list                  jobs/cells.tsv: the cells, their level and campaign group

f enters twice, at the two precisions the study used (campaigns.F_CAL): the verdicts carry
each table's f as its folder names it (four decimals), the recomputation uses the bootstrap's
full value, as the study's own recomputation did.

The rules (R1-R10, the study's pre-registered rules where the pre-registration is explicit,
the conservative reading where it is not):

  R1   the headline arm is xfer against the self_cf baseline; self_in is optimism only
  R2   pair widths are primary (SPEC 4); the per-board sigma is secondary and reported
  R3   TOST at alpha 0.05 against max(1.0 ps, 2 % of sigma(self_cf)): PASS iff the 90 % CI
       lies inside the margin (the aggregation's verdict, read from the table)
  R4   transferability is claimed on the interpolating family only; the extrapolating family
       is reported beside it and never carries a PASS
  R5   a cell passes only if every primary (pair, interp) row passes, an intersection-union
       test, so no multiplicity correction on PASS (verdict_from). A guarded FAIL row takes
       precedence over an unguarded row: such a cell is "FAIL (also unguarded on one pair)".
       The pre-registration orders neither way; this order follows its rule that one FAIL
       row makes the cell FAIL, and it is post hoc
  R5b  Holm: the aggregation applies Holm across a cell's board FAIL claims only, so the
       primary (xfer, pair, interp) rows carry no Holm correction; primary_rows.csv adds
       Holm over them within each cell as a sensitivity (holm)
  R6   a row whose guard is not "ok" cannot pass (the aggregation writes NA-unguarded);
       "uninformative" = the self leave-pair-out cost exceeds the margin. The
       pre-registration states it for level e only; it is applied study-wide (post hoc)
  R7   a cell on a thin target run (campaigns.THIN_RUNS) is SUPPORTING, never a PASS
  R8   a verdict that changes between the f variants of DESY May (0.8153, 1.0) or DESY
       August (0.8979, 0.8153, 1.0) is UNSTABLE, not PASS (f_stability, final_verdict)
  R9   no claim below 0.45 ps (NOCLAIM): below_noclaim flags |mu| under it
  R10  the placebo gate: |R_median| < 0.05 on the placebo arm (placebo_ok)

The level verdict (level_verdict): PASS when every cell passes; PASS with the supporting cells
excluded when the others pass; INCONCLUSIVE when every cell is unguarded or has no rows; FAIL
or MIXED (FAIL with a passing cell) when a cell fails; MIXED otherwise.
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import campaigns
from . import jobs as joblists
from .agg import AGG_COLUMNS
from .cell import CELL_COLUMNS
from .naming import log

# round 1's agg_by_cell.csv: the current columns less the one round 1 did not write
ROUND1_AGG_COLUMNS = [c for c in AGG_COLUMNS if c != "placebo_minus_notwc1b_ps"]
# round 1's cell table: the current columns less the seven added after it
ROUND1_CELL_COLUMNS = [c for c in CELL_COLUMNS
                       if c not in ("xfer_board", "src_role", "src_pick", "src_pix_rc",
                                    "src_pix_ntrk", "src_pix_pop", "lpo_pairs")]
# the agg_by_cell columns the chain reads
ROUND1_AGG_READ = ["cell", "arm", "kind", "name", "extrap_family", "self_transfer", "n_pix",
                   "n_tgt", "n_src", "mu_src_spread", "sigma_base", "mu", "se", "tau", "p95",
                   "chi2_red", "margin", "tost", "tost_raw", "ci_lo", "ci_hi", "p_nonequiv",
                   "has_xfer_lpo", "cheat_stat", "self_lpo_cost", "guard_verdict",
                   "frac_over_margin", "R_median", "gate_fail", "k_cal", "lever_X_all",
                   "p_ne_holm"]
NOCLAIM = 0.45                 # R9, ps
BASELINE = "self_cf"
NMIN_EVAL = 600                # the aggregation's n_eval floor (agg --nmin-eval)
# the arms without a source in round 1 (no notwc_1b yet): never split by self_transfer
SELF_ARMS = ("notwc", "self_in", "self_cf", "self_cf_lpo", "placebo_self", "self_in_rep")
FINAL_LABEL = {"FAIL+UNGUARDED": "FAIL (also unguarded on one pair)"}
# the headline rows the self-check prints beside the aggregation's numbers
SPOT = [("a1_twin_26to27", "pair", "ref-trig", "interp"),
        ("bp4_twin_off60", "pair", "ref-trig", "interp"),
        ("b4_30to45", "pair", "trig-dut", "interp"),
        ("c3_pix2board_run44_60deg", "pair", "ref-dut", "interp"),
        ("e5_pooled_may_run26", "pair", "ref-dut", "interp"),
        ("h4_mar17_to_jul10", "pair", "dut-ref", "interp"),
        ("g6_h1_14to19", "board", "dut", "extrap"),
        ("h2_mar19_to_jul4", "board", "ref", "extrap")]
OUTPUTS = ("recon_cells.csv", "recon.json", "cell_verdicts.csv", "f_stability.csv",
           "cell_final.csv", "primary_rows.csv", "recheck_by_group.csv",
           "alltriples_by_cell.csv", "support_by_cell.csv", "same_pixel_audit.csv",
           "spec_literal.csv", "selfcheck_join.csv", "selfcheck_spot.csv", "master_cells.csv",
           "level_verdicts.csv")


class MasterError(RuntimeError):
    """An input of the round-1 chain is missing or not what the chain reads."""


# ----------------------------------------------------------------------------- inputs
def round1_tag(tag):
    """(group, f) of a round-1 table folder; MasterError for a folder not in the closed list."""
    for t, group, f in campaigns.ROUND1_AGG_TAGS:
        if t == tag:
            return group, f
    raise MasterError("%s is not one of the round-1 aggregation tables (%s)"
                      % (tag, ", ".join(t for t, _, _ in campaigns.ROUND1_AGG_TAGS)))


def read_round1_table(read_root, tag):
    """tables/<tag>/agg_by_cell.csv of one round-1 table, as written; MasterError when the tag
    is not in the closed list or the table lacks a column the chain reads."""
    round1_tag(tag)
    path = os.path.join(read_root, "tables", tag, "agg_by_cell.csv")
    if not os.path.isfile(path):
        raise MasterError("no round-1 aggregation table %s" % path)
    d = pd.read_csv(path)
    miss = [c for c in ROUND1_AGG_READ if c not in d.columns]
    if miss:
        raise MasterError("%s lacks the columns %s" % (path, miss))
    if list(d.columns) != ROUND1_AGG_COLUMNS:
        log("master: %s has %d columns, not round 1's %d; read by name"
            % (path, d.shape[1], len(ROUND1_AGG_COLUMNS)))
    return d


def job_table(path):
    """The job list as a frame: job_id, cell, level, group (jobs.read_jobs)."""
    rows = [dict(job_id=j["job_id"], cell=j["cell_id"], level=j["level"], group=j["group"])
            for j in joblists.read_jobs(path)]
    return pd.DataFrame(rows)


def cell_files(cells_dir, cell, ext):
    return sorted(glob.glob(os.path.join(cells_dir, cell, "*." + ext)))


def load_cell(cells_dir, cell):
    """The cell tables of one cell stacked, or None when it has none. A table whose columns
    are not round 1's (ROUND1_CELL_COLUMNS) is read by name, with a line."""
    fs = cell_files(cells_dir, cell, "csv")
    if not fs:
        return None
    frames = [pd.read_csv(f) for f in fs]
    for f, t in zip(fs, frames):
        if list(t.columns) != ROUND1_CELL_COLUMNS:
            log("master: %s has %d columns, not round 1's %d; read by name"
                % (f, t.shape[1], len(ROUND1_CELL_COLUMNS)))
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------------------ recon
def prereg_label(cell):
    """prereg (one of the 40 pre-registered cells), added, level-f-expansion, smoke or EXTRA."""
    if cell in {c for cs in campaigns.ROUND1_PREREG.values() for c in cs}:
        return "prereg"
    if cell in campaigns.ROUND1_ADDED:
        return "added"
    if cell.startswith("f1_tier0_"):
        return "level-f-expansion"
    if cell.startswith("smoke_"):
        return "smoke"
    return "EXTRA"


def recon(jobs, cells_dir, agg_cells):
    """recon_cells.csv as a frame, and the recon.json summary.

    `jobs` is job_table(), `cells_dir` the read root's cells/, `agg_cells` {tag: set of the
    cells its agg_by_cell holds} over the round-1 tables."""
    recs = []
    for cell, gj in jobs.groupby("cell"):
        lev = gj.level.iloc[0]
        csvs = cell_files(cells_dir, cell, "csv")
        logs = cell_files(cells_dir, cell, "log")
        exits = []
        for lg in logs:
            m = re.findall(r"exit=(\d+)", open(lg, errors="replace").read())
            exits += [int(x) for x in m]
        r = dict(cell=cell, level=lev, group=gj.group.iloc[0], n_jobs=len(gj),
                 n_csv=len(csvs), n_log=len(logs),
                 exits=",".join(str(e) for e in exits) or "-",
                 all_exit0=bool(exits) and all(e == 0 for e in exits))
        if csvs:
            t = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
            r["n_rows"] = len(t)
            r["n_tgt"] = int(t.tgt_triple.nunique())
            r["n_src"] = int(t.src_triple.dropna().nunique())
            r["arms"] = ",".join(sorted(t.arm.unique()))
            r["has_placebo"] = int("placebo" in set(t.arm))
            r["has_xfer_lpo"] = int("xfer_lpo" in set(t.arm))
            b = t[t.kind == "board"]
            n_per = b.groupby("tgt_triple")["n"].median()
            r["n_eval_median"] = float(n_per.median())
            r["n_eval_min"] = float(n_per.min())
            r["n_tgt_below600"] = int((n_per < 600).sum())
            r["n_tgt_below1500"] = int((n_per < 1500).sum())
            v = pd.to_numeric(t["value"], errors="coerce")
            r["n_nan_value"] = int((~np.isfinite(v)).sum())
            r["n_neg_radicand"] = int((pd.to_numeric(t.get("neg_radicand"),
                                                     errors="coerce") == 1).sum())
            acc = pd.to_numeric(t.get("accepted"), errors="coerce")
            r["gate_fail_frac"] = float(np.mean(acc == 0)) if acc.notna().any() else np.nan
            r["src_keys_checked_med"] = float(pd.to_numeric(t.get("src_keys_checked"),
                                                            errors="coerce").median())
            r["keys_comparable_med"] = float(pd.to_numeric(t.get("keys_comparable"),
                                                           errors="coerce").median())
            r["same_pixel_frac"] = float(np.nanmean(pd.to_numeric(t.get("same_pixel"),
                                                                  errors="coerce")))
            r["campaign"] = t.campaign.iloc[0]
            r["run"] = t.run.iloc[0]
            r["combo"] = t.combo.iloc[0]
            r["src_run"] = str(t.src_run.dropna().iloc[0]) if t.src_run.notna().any() else "-"
        else:
            for k in ("n_rows", "n_tgt", "n_src", "n_eval_median"):
                r[k] = np.nan
            r["arms"] = "-"
        r["in_agg"] = ",".join(sorted(t_ for t_, s in agg_cells.items() if cell in s)) or "-"
        recs.append(r)
    rec = pd.DataFrame(recs)
    rec["prereg"] = rec.cell.map(prereg_label)
    found = set(rec.cell)
    prereg_ids = [c for cs in campaigns.ROUND1_PREREG.values() for c in cs]
    missing = [c for c in prereg_ids if c not in found and not c.startswith("f1_tier0_map")]
    rec = rec.sort_values(["level", "cell"])
    summ = dict(
        n_jobs=len(jobs), n_cells=int(rec.cell.nunique()),
        prereg_cells_expected=len(prereg_ids),
        prereg_found=int((rec.prereg == "prereg").sum()),
        added_found=int((rec.prereg == "added").sum()),
        f_expansion_jobs=int((rec.prereg == "level-f-expansion").sum()),
        smoke=int((rec.prereg == "smoke").sum()),
        missing_prereg=missing,
        f1_tier0_map_as_named="f1_tier0_map" in found,
        all_jobs_exit0=bool(rec.all_exit0.all()),
        cells_with_nan_value=rec.loc[rec.n_nan_value.fillna(0) > 0, "cell"].tolist(),
        cells_with_neg_radicand=rec.loc[rec.n_neg_radicand.fillna(0) > 0, "cell"].tolist(),
        agg_tables={t: sorted(s) for t, s in agg_cells.items()},
    )
    return rec, summ


# --------------------------------------------------------------------------- verdicts
def cell_rows(d, cell, arm, kind, fam=None, st=0):
    q = (d.cell == cell) & (d.arm == arm) & (d.kind == kind) & (d.self_transfer == st)
    if fam is not None:
        q &= (d.extrap_family == fam)
    return d[q]


def verdict_from(rows):
    """R5 over `rows`: NO-ROWS, FAIL+UNGUARDED (a guarded FAIL and an unguarded row), FAIL,
    NA-UNGUARDED, PASS (every row passes) or MIXED."""
    if not len(rows):
        return "NO-ROWS"
    v = set(rows.tost)
    guarded_fail = "FAIL" in v
    unguarded = ("NA-unguarded" in v) or ("NA" in v)
    if guarded_fail and unguarded:
        return "FAIL+UNGUARDED"
    if guarded_fail:
        return "FAIL"
    if unguarded:
        return "NA-UNGUARDED"
    return "PASS" if v == {"PASS"} else "MIXED"


def holm(ps):
    """Holm step-down adjusted p-values (R5b); a non-finite p counts as 1."""
    ps = np.asarray(ps, float)
    m = len(ps)
    if m == 0:
        return ps
    order = np.argsort(np.where(np.isfinite(ps), ps, 1.0))
    out = np.full(m, np.nan)
    run = 0.0
    for i, idx in enumerate(order):
        pv = ps[idx] if np.isfinite(ps[idx]) else 1.0
        run = max(run, (m - i) * pv)
        out[idx] = min(run, 1.0)
    return out


def stacked_tables(tables):
    """The round-1 tables stacked in the order of the closed list, each row with its ftag;
    `tables` is {tag: agg_by_cell frame}. MasterError for a tag outside the list."""
    for tag in tables:
        round1_tag(tag)
    frames = []
    for tag, _, _ in campaigns.ROUND1_AGG_TAGS:
        if tag in tables:
            d = tables[tag].copy()
            d["ftag"] = tag
            frames.append(d)
    return pd.concat(frames, ignore_index=True)


def cell_verdicts(A, level, meta):
    """cell_verdicts.csv: one row per (round-1 table, cell). `A` is stacked_tables(), `level`
    {cell: level}, `meta` recon_cells indexed by cell."""
    out = []
    for ftag, d in A.groupby("ftag"):
        grp, fval = round1_tag(ftag)
        for cell in sorted(d.cell.unique()):
            lev = level.get(cell, "?")
            m = meta.loc[cell] if cell in meta.index else None
            camp = str(m["campaign"]) if m is not None else ""
            run = str(m["run"]) if m is not None else ""
            thin_tgt = (camp, run) in campaigns.THIN_RUNS
            pi = cell_rows(d, cell, "xfer", "pair", "interp")
            pe = cell_rows(d, cell, "xfer", "pair", "extrap")
            bi = cell_rows(d, cell, "xfer", "board", "interp")
            be = cell_rows(d, cell, "xfer", "board", "extrap")
            # the binding primary row: the one furthest from equivalence
            rec = dict(cell=cell, level=lev, ftag=ftag, group=grp, f=fval,
                       campaign=camp, run=run, thin_target=int(thin_tgt),
                       n_pair_interp=len(pi), n_pair_extrap=len(pe),
                       npix_interp=int(pi.n_pix.sum()) if len(pi) else 0,
                       npix_extrap=int(pe.n_pix.sum()) if len(pe) else 0)
            if len(pi):
                z = (np.abs(pi.mu) / pi.margin)
                b = pi.loc[z.idxmax()]
                rec.update(bind_name=b["name"], mu=b["mu"], se=b["se"], tau=b["tau"],
                           ci_lo=b["ci_lo"], ci_hi=b["ci_hi"], margin=b["margin"],
                           tost=b["tost"], n_pix=b["n_pix"], sigma_base=b["sigma_base"],
                           R_median=b["R_median"], frac_over=b["frac_over_margin"],
                           chi2_red=b["chi2_red"], p95=b["p95"],
                           mu_min=pi.mu.min(), mu_max=pi.mu.max(),
                           mu_ivw=float(np.sum(pi.mu / pi.se**2) / np.sum(1 / pi.se**2)),
                           se_ivw=float(np.sqrt(1 / np.sum(1 / pi.se**2))))
            else:
                for k in ("bind_name", "mu", "se", "tau", "ci_lo", "ci_hi", "margin", "tost",
                          "n_pix", "sigma_base", "R_median", "frac_over", "chi2_red", "p95",
                          "mu_min", "mu_max", "mu_ivw", "se_ivw"):
                    rec[k] = np.nan
            rec["n_row_fail"] = int((pi.tost == "FAIL").sum()) if len(pi) else 0
            rec["n_row_unguarded"] = (int(pi.tost.isin(["NA-unguarded", "NA"]).sum())
                                      if len(pi) else 0)
            rec["n_row_pass"] = int((pi.tost == "PASS").sum()) if len(pi) else 0
            # underpowered: a FAIL row that does not show non-equivalence either
            rec["n_row_underpowered"] = int(((pi.tost == "FAIL") &
                                             (pi.p_nonequiv > 0.5)).sum()) if len(pi) else 0
            rec["p_nonequiv_min"] = float(np.nanmin(pi.p_nonequiv)) if len(pi) else np.nan
            rec["p_nonequiv_max"] = float(np.nanmax(pi.p_nonequiv)) if len(pi) else np.nan
            rec["holm_missing"] = int(pi.p_ne_holm.isna().all()) if len(pi) else 1
            rec["verdict_pair_interp"] = verdict_from(pi)
            rec["verdict_pair_extrap"] = verdict_from(pe)
            rec["verdict_board_interp"] = verdict_from(bi)
            rec["verdict_board_extrap"] = verdict_from(be)
            if len(pe):
                z = (np.abs(pe.mu) / pe.margin)
                b = pe.loc[z.idxmax()]
                rec.update(mu_extrap=b["mu"], se_extrap=b["se"], npix_e=b["n_pix"])
            else:
                rec.update(mu_extrap=np.nan, se_extrap=np.nan, npix_e=np.nan)
            # controls (R10)
            pl = d[(d.cell == cell) & (d.arm == "placebo") & (d.kind == "pair")]
            pls = d[(d.cell == cell) & (d.arm == "placebo_self") & (d.kind == "pair")]
            nw = d[(d.cell == cell) & (d.arm == "notwc") & (d.kind == "pair")]
            rec["placebo_present"] = int(len(pl) > 0)
            rec["placebo_R_absmax"] = (float(np.nanmax(np.abs(pl.R_median))) if len(pl)
                                       else np.nan)
            rec["placebo_mu_med"] = float(np.nanmedian(pl.mu)) if len(pl) else np.nan
            rec["placebo_self_R_absmax"] = (float(np.nanmax(np.abs(pls.R_median))) if len(pls)
                                            else np.nan)
            rec["notwc_mu_med"] = float(np.nanmedian(nw.mu)) if len(nw) else np.nan
            rec["placebo_ok"] = int(rec["placebo_R_absmax"] < 0.05) if len(pl) else 0
            src = pi if len(pi) else pe
            if len(src):
                rec["has_xfer_lpo"] = int(src.has_xfer_lpo.max())
                rec["guard_verdict"] = ",".join(sorted(set(src.guard_verdict.astype(str))))
                rec["cheat_absmax"] = float(np.nanmax(np.abs(src.cheat_stat)))
                rec["self_lpo_absmax"] = float(np.nanmax(np.abs(src.self_lpo_cost)))
                rec["margin_min"] = float(np.nanmin(src.margin))
                rec["cheat_over_margin"] = int(np.isfinite(rec["cheat_absmax"]) and
                                               rec["cheat_absmax"] > rec["margin_min"])
                rec["cheat_missing"] = int(not np.isfinite(rec["cheat_absmax"]))
                rec["lpo_ok"] = int((rec["cheat_absmax"] < src.margin.min()) and
                                    (rec["guard_verdict"] == "ok"))
                rec["gate_fail"] = float(np.nanmedian(src.gate_fail))
                rec["k_cal"] = float(np.nanmedian(src.k_cal))
                rec["lever_X"] = float(np.nanmedian(src.lever_X_all))
                rec["n_src"] = int(np.nanmax(src.n_src))
                rec["mu_src_spread"] = float(np.nanmedian(src.mu_src_spread))
            # the guarded transfer, xfer_lpo
            xl = cell_rows(d, cell, "xfer_lpo", "pair", "interp")
            rec["xfer_lpo_verdict"] = verdict_from(xl)
            rec["xfer_lpo_mu"] = float(np.nanmedian(xl.mu)) if len(xl) else np.nan
            out.append(rec)
    return pd.DataFrame(out)


def f_variants(group):
    """The tables whose verdicts must agree for a cell of `group` (R8): the f variants of DESY
    May and August, else the group's primary table alone."""
    tags = [t for t, g, _ in campaigns.ROUND1_AGG_TAGS if g == group]
    return tags if group in ("desy", "aug") else [campaigns.ROUND1_PRIMARY_TAG[group]]


def f_stability(C):
    """f_stability.csv: per cell, its pair-interp verdict in each f variant (R8)."""
    stab = []
    for cell, g in C.groupby("cell"):
        vs = dict(zip(g.ftag, g.verdict_pair_interp))
        grp = g.group.iloc[0]
        pair = f_variants(grp)
        seen = [vs.get(p) for p in pair if p in vs]
        stab.append(dict(cell=cell, group=grp, variants=";".join(pair),
                         verdicts=";".join(map(str, seen)),
                         stable=int(len(set(seen)) == 1),
                         unstable_flag=int(len(set(seen)) > 1)))
    return pd.DataFrame(stab)


def final_verdict(row, st):
    """R8, then R7, then R5's label: UNSTABLE, SUPPORTING (thin target, no PASS claimed), or
    the pair-interp verdict."""
    v = row["verdict_pair_interp"]
    if st["unstable_flag"]:
        return "UNSTABLE"
    if row["thin_target"] and v == "PASS":
        return "SUPPORTING (thin target, no PASS claimed)"
    return FINAL_LABEL.get(v, v)


def cell_final(C, S):
    """cell_final.csv: the rows of each group's primary table with the final verdict and R9."""
    prim = C[C.ftag.isin(campaigns.ROUND1_PRIMARY_TAG.values())].copy()
    sm = S.set_index("cell")
    P = prim.copy()
    P["final"] = [final_verdict(r, sm.loc[r["cell"]]) for _, r in P.iterrows()]
    P["below_noclaim"] = (P.mu.abs() < NOCLAIM).astype(int)
    return P.sort_values(["level", "cell"])


PRIMARY_ROW_COLUMNS = ["cell", "level", "name", "n_pix", "n_tgt", "n_src", "mu", "se", "tau",
                       "margin", "ci_lo", "ci_hi", "tost", "tost_raw", "guard_verdict",
                       "frac_over_margin", "p95", "chi2_red", "p_nonequiv", "p_ne_holm",
                       "p_ne_holm_recomputed", "underpowered", "nonequiv_holm_005",
                       "below_noclaim", "R_median", "sigma_base"]


def primary_rows(A, level):
    """primary_rows.csv: the (xfer, pair, interp, self_transfer 0) rows of the primary tables,
    with Holm over each cell's rows (R5b)."""
    PR = A[(A.ftag.isin(campaigns.ROUND1_PRIMARY_TAG.values())) & (A.arm == "xfer") &
           (A.kind == "pair") & (A.extrap_family == "interp") & (A.self_transfer == 0)].copy()
    PR["level"] = PR.cell.map(level)
    PR["p_ne_holm_recomputed"] = np.nan
    for c, g in PR.groupby("cell"):
        PR.loc[g.index, "p_ne_holm_recomputed"] = holm(g.p_nonequiv.to_numpy())
    PR["underpowered"] = ((PR.tost == "FAIL") & (PR.p_nonequiv > 0.5)).astype(int)
    PR["nonequiv_holm_005"] = ((PR.tost == "FAIL") &
                               (PR.p_ne_holm_recomputed < 0.05)).astype(int)
    PR["below_noclaim"] = (PR.mu.abs() < NOCLAIM).astype(int)
    return PR[PRIMARY_ROW_COLUMNS].sort_values(["level", "cell", "name"])


# -------------------------------------------------------------------------- recompute
def censor(df):
    """Drop a (triple, quantity) from every arm when one arm's value is not finite or has
    neg_radicand == 1. Returns (frame, rows dropped)."""
    bad = ~np.isfinite(pd.to_numeric(df["value"], errors="coerce"))
    bad |= (pd.to_numeric(df["neg_radicand"], errors="coerce") == 1)
    kk = ["cell", "run", "combo", "tgt_triple", "kind", "name"]
    dead = set(map(tuple, df.loc[bad, kk].drop_duplicates().to_numpy().tolist()))
    if not dead:
        return df, 0
    tup = list(map(tuple, df[kk].to_numpy().tolist()))
    keep = np.array([t not in dead for t in tup])
    return df[keep].reset_index(drop=True), int((~keep).sum())


def nfloor(df):
    """Drop every row of a triple one of whose board rows has n < NMIN_EVAL. Returns (frame,
    rows dropped)."""
    small = (pd.to_numeric(df["n"], errors="coerce") < NMIN_EVAL) & (df["kind"] == "board")
    kk = ["cell", "run", "combo", "tgt_triple"]
    dead = set(map(tuple, df.loc[small, kk].drop_duplicates().to_numpy().tolist()))
    if not dead:
        return df, 0
    tup = list(map(tuple, df[kk].to_numpy().tolist()))
    keep = np.array([t not in dead for t in tup])
    return df[keep].reset_index(drop=True), int((~keep).sum())


def deltas(df):
    """Every arm against the baseline per (cell, mode, run, combo, tgt_triple, kind, name):
    delta, the fold deltas d_f0/d_f1 and scale = value_b / sqrt(n_b)."""
    key = ["cell", "mode", "run", "combo", "tgt_triple", "kind", "name"]
    base = df[df.arm == BASELINE].set_index(key)
    base = base[~base.index.duplicated(keep="first")]
    out = []
    for arm, g in df.groupby("arm"):
        if arm == BASELINE:
            continue
        m = g.set_index(key).join(base[["value", "v_f0", "v_f1", "n"]],
                                  rsuffix="_b", how="inner").reset_index()
        m["delta"] = m["value"] - m["value_b"]
        m["d_f0"] = m["v_f0"] - m["v_f0_b"]
        m["d_f1"] = m["v_f1"] - m["v_f1_b"]
        m["scale"] = m["value_b"] / np.sqrt(m["n_b"].clip(lower=1))
        out.append(m)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def kcal(g):
    """k_cal of one (cell, run, arm, kind): median |d_f0 - d_f1| / scale over 2 * 0.6745."""
    u = g.dropna(subset=["d_f0", "d_f1", "scale"])
    if not len(u):
        return np.nan
    r = (np.abs(u["d_f0"] - u["d_f1"]) / u["scale"]).to_numpy()
    r = r[np.isfinite(r)]
    return float(np.median(r) / (2 * 0.6745)) if len(r) else np.nan


def dl(delta, se):
    """DerSimonian-Laird random effects: n, mu, se, tau."""
    d, s = np.asarray(delta, float), np.asarray(se, float)
    m = np.isfinite(d) & np.isfinite(s) & (s > 0)
    d, s = d[m], s[m]
    if not len(d):
        return dict(n=0, mu=np.nan, se=np.nan, tau=np.nan)
    w = 1 / s**2
    mu_fe = np.sum(w * d) / np.sum(w)
    Q = np.sum(w * (d - mu_fe)**2)
    k = len(d)
    C = np.sum(w) - np.sum(w**2) / np.sum(w)
    tau2 = max((Q - (k - 1)) / C, 0.0) if C > 0 else 0.0
    wr = 1 / (s**2 + tau2)
    return dict(n=k, mu=float(np.sum(wr * d) / np.sum(wr)),
                se=float(np.sqrt(1 / np.sum(wr))), tau=float(np.sqrt(tau2)))


def se_src_of(delta, se, src):
    """The between-source term: the standard error of the per-source inverse-variance means
    (0 with one source), and the number of sources."""
    d, s = np.asarray(delta, float), np.asarray(se, float)
    src = np.asarray(src, object)
    m = np.isfinite(d) & np.isfinite(s) & (s > 0)
    d, s, src = d[m], s[m], src[m]
    per = []
    for u in sorted(set(map(str, src))):
        sel = np.array([str(x) == u for x in src])
        w = 1 / s[sel]**2
        per.append(float(np.sum(w * d[sel]) / np.sum(w)))
    if len(per) <= 1:
        return 0.0, len(per)
    return float(np.std(per, ddof=1) / np.sqrt(len(per))), len(per)


def tost(mu, se, marg, alpha=0.05):
    """(PASS or FAIL or NA, ci_lo, ci_hi): PASS iff the 1 - 2 alpha interval lies inside
    +-marg."""
    if not (np.isfinite(mu) and np.isfinite(se) and se > 0):
        return "NA", np.nan, np.nan
    z = norm.ppf(1 - alpha)
    lo, hi = mu - z * se, mu + z * se
    return ("PASS" if (lo > -marg and hi < marg) else "FAIL"), lo, hi


def ivw_pairs(g, secol="se"):
    """Inverse-variance mean over the pair rows of one cell, each pair row being the
    DerSimonian-Laird mean over its target triples with the between-source term."""
    mus, ses = [], []
    for _, gg in g.groupby("name"):
        re_ = dl(gg.delta, gg[secol])
        s2, _ = se_src_of(gg.delta, gg[secol], gg.src_triple.fillna(""))
        se_t = float(np.sqrt(re_["se"]**2 + s2**2)) if np.isfinite(re_["se"]) else np.nan
        if np.isfinite(re_["mu"]) and np.isfinite(se_t) and se_t > 0:
            mus.append(re_["mu"])
            ses.append(se_t)
    if not mus:
        return np.nan, np.nan, 0
    mus, ses = np.asarray(mus), np.asarray(ses)
    w = 1.0 / ses**2
    return float(np.sum(w * mus) / np.sum(w)), float(np.sqrt(1.0 / np.sum(w))), len(mus)


def recompute_deltas(df, f_inf):
    """One cell's rows through the chain: the n_eval floor, the symmetric censoring, the paired
    delta, k_cal per (cell, run, arm, kind), se = f * k_cal * scale, self_transfer 0 on the
    self arms, the same-pixel transfers of the same run and combo dropped, the extrapolation
    family. Returns (deltas, rows censored, rows below the floor), deltas None without rows."""
    df, n_small = nfloor(df)
    df, n_cens = censor(df)
    d = deltas(df)
    if not len(d):
        return None, n_cens, n_small
    kc = {k: kcal(g) for k, g in d.groupby(["cell", "run", "arm", "kind"])}
    d["k_cal"] = [kc.get((c, r, a_, k_), np.nan)
                  for c, r, a_, k_ in zip(d.cell, d.run, d.arm, d["kind"])]
    d["se"] = f_inf * d["k_cal"] * d["scale"]
    d.loc[d.arm.isin(SELF_ARMS), "self_transfer"] = 0
    sp = pd.to_numeric(d.get("same_pixel"), errors="coerce")
    same_run = (d.src_run.astype(str) == d.run.astype(str)) & \
               (d.src_combo.astype(str) == d.combo.astype(str))
    drop_sp = d.arm.astype(str).str.startswith("xfer") & (sp == 1) & same_run
    d = d[~drop_sp]
    d["fam"] = d["extrap"].fillna(-1).astype(int).map({1: "extrap", 0: "interp", -1: "na"})
    return d, n_cens, n_small


def _pooled_row(g):
    re_ = dl(g.delta, g.se)
    ses, nsrc = se_src_of(g.delta, g.se, g.src_triple.fillna(""))
    se_tot = float(np.sqrt(re_["se"]**2 + ses**2)) if np.isfinite(re_["se"]) else np.nan
    sig = float(g.value_b.median())
    marg = float(max(1.0, 0.02 * sig)) if np.isfinite(sig) else 1.0
    v, lo, hi = tost(re_["mu"], se_tot, marg)
    return re_, se_tot, nsrc, sig, marg, v, lo, hi


def recompute_cell(cell, grp, d, n_cens, n_small):
    """The rows one cell adds to the recomputation's tables: {table: [rows]} for recheck,
    alltriples, support, same_pixel, spec."""
    out = dict(check=[], all=[], sup=[], sp=[], spec=[])
    # audits (not part of the chain)
    d["se_spec"] = d["k_cal"] * np.abs(d["d_f0"] - d["d_f1"]) / 2.0
    xp = d[(d.arm == "xfer") & (d["kind"] == "pair") & (d.self_transfer == 0)].copy()
    if len(xp):
        # distinct pair names that interpolate on each target triple: on a cell with several
        # sources, "all three pairs" means all three pairs, not three interpolating rows
        n_names_all = xp.groupby("tgt_triple")["name"].nunique()
        n_names_int = (xp[xp.fam == "interp"].groupby("tgt_triple")["name"].nunique()
                       .reindex(n_names_all.index).fillna(0))
        out["sup"].append(dict(
            cell=cell, level="", n_pair_rows=int(len(xp)),
            n_interp_pair_rows=int((xp.fam == "interp").sum()),
            n_extrap_pair_rows=int((xp.fam == "extrap").sum()),
            n_triples=int(xp.tgt_triple.nunique()),
            n_tri_any_interp=int((n_names_int > 0).sum()),
            n_tri_all_interp=int((n_names_int >= n_names_all).sum()),
            n_pairs_per_triple=float(n_names_all.median()),
            n_src_rows=int(xp.src_triple.nunique())))

        same_tri = xp.src_triple.astype(str) == xp.tgt_triple.astype(str)
        kept = xp[~same_tri]
        a0 = ivw_pairs(xp)
        a1 = ivw_pairs(kept)
        i0 = ivw_pairs(xp[xp.fam == "interp"])
        i1 = ivw_pairs(kept[kept.fam == "interp"])
        out["sp"].append(dict(
            cell=cell, n_xfer_pair_rows=int(len(xp)),
            n_same_triple=int(same_tri.sum()),
            n_interp_rows=int((xp.fam == "interp").sum()),
            n_same_triple_interp=int((same_tri & (xp.fam == "interp")).sum()),
            same_pixel_col_all_nan=bool(pd.to_numeric(xp.get("same_pixel"),
                                                      errors="coerce").isna().all()),
            all_mu=a0[0], all_se=a0[1], all_mu_excl=a1[0], all_se_excl=a1[1],
            int_mu=i0[0], int_se=i0[1], int_mu_excl=i1[0], int_se_excl=i1[1]))

        t0 = ivw_pairs(xp, "se")
        t1 = ivw_pairs(xp, "se_spec")
        u0 = ivw_pairs(xp[xp.fam == "interp"], "se")
        u1 = ivw_pairs(xp[xp.fam == "interp"], "se_spec")
        out["spec"].append(dict(
            cell=cell, all_mu_tk=t0[0], all_se_tk=t0[1], all_mu_spec=t1[0], all_se_spec=t1[1],
            int_mu_tk=u0[0], int_se_tk=u0[1], int_mu_spec=u1[0], int_se_spec=u1[1],
            n_rows_tk=int(np.isfinite(pd.to_numeric(xp.se, errors="coerce")).sum()),
            n_rows_spec=int((pd.to_numeric(xp.se_spec, errors="coerce") > 0).sum())))

    # the aggregation's grouping, for the self-check
    for (arm, kind, name, st, fam), g in d.groupby(["arm", "kind", "name",
                                                    "self_transfer", "fam"]):
        re_, se_tot, nsrc, sig, marg, v, lo, hi = _pooled_row(g)
        out["check"].append(dict(cell=cell, group=grp, arm=arm, kind=kind, name=name,
                                 self_transfer=int(st), extrap_family=fam,
                                 n_pix=re_["n"], mu=re_["mu"], se=se_tot, tau=re_["tau"],
                                 margin=marg, tost=v, ci_lo=lo, ci_hi=hi, n_src=nsrc))
    # descriptive: pooled over the extrapolation families (all triples)
    for (arm, kind, name, st), g in d.groupby(["arm", "kind", "name", "self_transfer"]):
        re_, se_tot, _, sig, marg, v, lo, hi = _pooled_row(g)
        out["all"].append(dict(cell=cell, group=grp, arm=arm, kind=kind, name=name,
                               self_transfer=int(st), n_pix=re_["n"], mu=re_["mu"],
                               se=se_tot, tau=re_["tau"], margin=marg, tost=v,
                               ci_lo=lo, ci_hi=hi, sigma_base=sig,
                               n_interp=int((g.fam == "interp").sum()),
                               n_extrap=int((g.fam == "extrap").sum()),
                               n_censored=n_cens, n_below_nmin=n_small))
    return out


def recompute(jobs, cells_dir, primary_tables):
    """The recomputation's tables as frames: recheck_by_group, alltriples_by_cell,
    support_by_cell, same_pixel_audit, spec_literal, selfcheck_join, selfcheck_spot.

    Every cell of the job list except the Tier-0 and smoke cells, in cell order, with the full
    f of its group (campaigns.F_CAL); `primary_tables` is {group: agg_by_cell of the group's
    primary round-1 table}, for the self-check."""
    cellgrp = dict(zip(jobs.cell, jobs.group))
    celllev = dict(zip(jobs.cell, jobs.level))
    rows = dict(check=[], all=[], sup=[], sp=[], spec=[])
    for cell, grp in sorted(cellgrp.items()):
        if cell.startswith(("f1_tier0", "smoke")):
            continue
        df = load_cell(cells_dir, cell)
        if df is None:
            continue
        d, n_cens, n_small = recompute_deltas(df, campaigns.F_CAL[grp][1])
        if d is None:
            continue
        for k, v in recompute_cell(cell, grp, d, n_cens, n_small).items():
            rows[k] += v
    RC = pd.DataFrame(rows["check"])
    RA = pd.DataFrame(rows["all"])
    SUP = pd.DataFrame(rows["sup"])
    SUP["level"] = SUP.cell.map(lambda c: celllev.get(c, "?"))
    SPX = pd.DataFrame(rows["sp"])
    SPX["d_all"] = SPX.all_mu_excl - SPX.all_mu
    SPX["d_int"] = SPX.int_mu_excl - SPX.int_mu
    SPC = pd.DataFrame(rows["spec"])
    SPC["d_all"] = SPC.all_mu_spec - SPC.all_mu_tk
    SPC["d_int"] = SPC.int_mu_spec - SPC.int_mu_tk

    A = []
    for grp, tag in campaigns.ROUND1_PRIMARY_TAG.items():
        a = primary_tables[grp].copy()
        a["group"] = grp
        A.append(a)
    A = pd.concat(A, ignore_index=True)
    key = ["cell", "arm", "kind", "name", "self_transfer", "extrap_family"]
    J = RC.merge(A[key + ["n_pix", "mu", "se", "tau", "margin", "tost_raw"]],
                 on=key, how="inner", suffixes=("_mine", "_agg"))
    J["d_mu"] = J.mu_mine - J.mu_agg
    J["d_se"] = J.se_mine - J.se_agg
    if len(J):
        log("master: self-check matched %d rows (recomputed %d, agg %d): max |d_mu| = %.3e "
            "ps, max |d_se| = %.3e ps, %d n_pix and %d TOST mismatches"
            % (len(J), len(RC), len(A), np.nanmax(np.abs(J.d_mu)), np.nanmax(np.abs(J.d_se)),
               int((J.n_pix_mine != J.n_pix_agg).sum()), int((J.tost != J.tost_raw).sum())))
    else:
        log("master: self-check matched no row (recomputed %d, agg %d)" % (len(RC), len(A)))
    sp = []
    for c, k, n, f in SPOT:
        r = J[(J.cell == c) & (J.arm == "xfer") & (J.kind == k) & (J["name"] == n) &
              (J.extrap_family == f) & (J.self_transfer == 0)]
        if len(r):
            r = r.iloc[0]
            sp.append(dict(cell=c, kind=k, name=n, family=f, n_pix=int(r.n_pix_mine),
                           mu_recomputed=r.mu_mine, mu_agg=r.mu_agg,
                           se_recomputed=r.se_mine, se_agg=r.se_agg,
                           tost_recomputed=r.tost, tost_agg=r.tost_raw))
    return dict(recheck_by_group=RC, alltriples_by_cell=RA, support_by_cell=SUP,
                same_pixel_audit=SPX, spec_literal=SPC, selfcheck_join=J,
                selfcheck_spot=pd.DataFrame(sp))


# ----------------------------------------------------------------------------- levels
def all_triples_headline(alltr, kind):
    """Per cell, the binding (largest |mu| / margin) xfer row of `kind` among the rows that are
    not self transfers, from alltriples_by_cell: the all_* columns (pair, with the
    inverse-variance mean over the pairs) or the allb_* columns (board)."""
    recs = []
    for cell, g in alltr[(alltr.arm == "xfer") & (alltr.kind == kind) &
                         (alltr.self_transfer == 0)].groupby("cell"):
        z = (g.mu.abs() / g.margin)
        b = g.loc[z.idxmax()]
        if kind == "board":
            recs.append(dict(cell=cell, allb_name=b["name"], allb_mu=b.mu, allb_se=b.se,
                             allb_tost=b.tost, allb_margin=b.margin))
            continue
        ivw_mu = float(np.sum(g.mu / g.se**2) / np.sum(1 / g.se**2))
        ivw_se = float(np.sqrt(1 / np.sum(1 / g.se**2)))
        v = set(g.tost)
        recs.append(dict(cell=cell, all_bind_name=b["name"], all_mu=b.mu, all_se=b.se,
                         all_margin=b.margin, all_tost=b.tost, all_npix=int(b.n_pix),
                         all_ivw_mu=ivw_mu, all_ivw_se=ivw_se,
                         all_verdict=("FAIL" if any(str(x).startswith("FAIL") for x in v)
                                      else "PASS"),
                         n_interp=int(g.n_interp.sum()), n_extrap=int(g.n_extrap.sum()),
                         all_sigma_base=float(g.sigma_base.median()),
                         n_censored=int(g.n_censored.iloc[0]),
                         n_below_nmin=int(g.n_below_nmin.iloc[0])))
    return pd.DataFrame(recs)


AUDIT_COLUMNS = (("support_by_cell", ["n_pair_rows", "n_interp_pair_rows", "n_triples",
                                      "n_tri_any_interp", "n_tri_all_interp"]),
                 ("same_pixel_audit", ["n_same_triple", "n_same_triple_interp", "all_mu_excl",
                                       "all_se_excl", "int_mu_excl", "int_se_excl"]),
                 ("spec_literal", ["all_mu_spec", "all_se_spec", "int_mu_spec",
                                   "int_se_spec"]))


def master_table(fin, alltr, recon_cells, stab, audits=None):
    """master_cells.csv: cell_final with the all-triples headlines, the target triples and
    source run (recon), the interpolating fraction, the verdicts across f and the audit columns
    ({table name: frame} of AUDIT_COLUMNS, each optional)."""
    recon_cells = recon_cells.set_index("cell")
    stab = stab.set_index("cell")
    M = fin.merge(all_triples_headline(alltr, "pair"), on="cell", how="left").merge(
        all_triples_headline(alltr, "board"), on="cell", how="left")
    M["n_tgt_triples"] = M.cell.map(recon_cells.n_tgt)
    M["src_run"] = M.cell.map(recon_cells.src_run)
    M["interp_frac"] = M.n_interp / (M.n_interp + M.n_extrap).replace(0, np.nan)
    M["f_verdicts"] = M.cell.map(stab.verdicts)
    for name, cols in AUDIT_COLUMNS:
        d = (audits or {}).get(name)
        if d is not None:
            d = d.set_index("cell")
            for c in cols:
                if c in d.columns:
                    M[c] = M.cell.map(d[c])
    return M.sort_values(["level", "cell"])


def is_fail(x):
    """FAIL and "FAIL (also unguarded on one pair)" are both FAIL (R5)."""
    return str(x).startswith("FAIL")


def level_verdict(finals):
    """The verdict of a level from the final verdicts of its cells (module docstring)."""
    v = list(finals)
    if all(x == "PASS" for x in v):
        return "PASS"
    if any(x.startswith("SUPPORTING") for x in v) and \
       all(x in ("PASS",) or x.startswith("SUPPORTING") for x in v):
        return "PASS (supporting cells excluded from the claim)"
    if all(x in ("NA-UNGUARDED", "NO-ROWS") for x in v):
        return "INCONCLUSIVE"
    if any(is_fail(x) for x in v):
        return "MIXED" if any(x == "PASS" for x in v) else "FAIL"
    return "MIXED"


def level_verdicts(M):
    """level_verdicts.csv: per level, its cells' counts and verdict (campaigns.LEVEL_TITLES)."""
    rows = []
    for lev, g in M.groupby("level"):
        rows.append(dict(level=lev, title=campaigns.LEVEL_TITLES.get(lev, lev),
                         n_cells=len(g),
                         n_pass=int((g["final"] == "PASS").sum()),
                         n_fail=int(g["final"].map(is_fail).sum()),
                         n_fail_also_unguarded=int((g["final"] ==
                                                    "FAIL (also unguarded on one pair)").sum()),
                         n_narows=int((g["final"] == "NO-ROWS").sum()),
                         n_unguarded=int((g["final"] == "NA-UNGUARDED").sum()),
                         interp_frac_med=float(g.interp_frac.median()),
                         all_mu_med=float(g.all_mu.median()),
                         verdict=level_verdict(g["final"])))
    return pd.DataFrame(rows).sort_values("level")


# -------------------------------------------------------------------------------- run
def run(read_root, job_list, outdir):
    """The four stages in order: reads the read root and the job list, writes OUTPUTS into
    `outdir`, each stage reading the tables of the stages before it back from `outdir`."""
    jobs = job_table(job_list)
    cells_dir = os.path.join(read_root, "cells")
    tables = {t: read_round1_table(read_root, t) for t, _, _ in campaigns.ROUND1_AGG_TAGS}
    os.makedirs(outdir, exist_ok=True)

    def out(name):
        return os.path.join(outdir, name)

    # recon
    agg_cells = {t: set(tables[t].cell.unique()) for t in sorted(tables)}
    rec, summ = recon(jobs, cells_dir, agg_cells)
    rec.to_csv(out("recon_cells.csv"), index=False)
    with open(out("recon.json"), "w") as fh:
        json.dump(summ, fh, indent=1)
    log("master: recon_cells.csv %d cells (%d jobs), every job exit 0: %s"
        % (len(rec), len(jobs), summ["all_jobs_exit0"]))

    # verdicts
    level = {}
    for j in joblists.read_jobs(job_list):
        level[j["cell_id"]] = j["level"]
    meta = pd.read_csv(out("recon_cells.csv")).set_index("cell")
    A = stacked_tables(tables)
    C = cell_verdicts(A, level, meta)
    C.to_csv(out("cell_verdicts.csv"), index=False)
    S = f_stability(C)
    S.to_csv(out("f_stability.csv"), index=False)
    P = cell_final(C, S)
    P.to_csv(out("cell_final.csv"), index=False)
    PR = primary_rows(A, level)
    PR.to_csv(out("primary_rows.csv"), index=False)
    log("master: cell_verdicts.csv %d rows, cell_final.csv %d cells, primary_rows.csv %d rows"
        % (len(C), len(P), len(PR)))

    # recompute
    prim = {g: tables[t] for g, t in campaigns.ROUND1_PRIMARY_TAG.items()}
    for name, frame in recompute(jobs, cells_dir, prim).items():
        frame.to_csv(out(name + ".csv"), index=False)

    # levels
    fin = pd.read_csv(out("cell_final.csv"))
    alltr = pd.read_csv(out("alltriples_by_cell.csv"))
    rc = pd.read_csv(out("recon_cells.csv"))
    stab = pd.read_csv(out("f_stability.csv"))
    audits = {n: pd.read_csv(out(n + ".csv")) for n, _ in AUDIT_COLUMNS
              if os.path.exists(out(n + ".csv"))}
    M = master_table(fin, alltr, rc, stab, audits)
    M.to_csv(out("master_cells.csv"), index=False)
    LV = level_verdicts(M)
    LV.to_csv(out("level_verdicts.csv"), index=False)
    log("master: master_cells.csv %d cells x %d columns; levels %s"
        % (len(M), M.shape[1], ", ".join("%s %s" % (r.level, r.verdict)
                                          for r in LV.itertuples())))
    return M


def cmd_master(a, root):
    """`master`: the round-1 chain from the read root into <write root>/master/ (or --out, a
    relative one in the write root)."""
    outdir = campaigns.output_path(a.out, root, "master")
    try:
        run(campaigns.transfer_root(), joblists.list_path(a.jobs), outdir)
    except MasterError as e:
        raise SystemExit("twc_transfer master: %s" % e)
    print("-> %s" % outdir)
