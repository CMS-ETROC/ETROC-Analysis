"""The axis tables (SPEC.md 10): one condensed row per (cell, arm, kind, board or pair,
extrapolation family, f) of the aggregation tables behind each axis, the wide side-by-side
table of axis F, the guardfix tables of axes E, J and K, and the cell table of axis F.

Inputs, in the read root (campaigns.transfer_root()): tables/<folder>/agg_by_cell.csv and
delta_by_triple.csv of every folder campaigns.AXIS_TAGS and GUARDFIX_AXIS_TAGS name (closed
lists; axis A reads agg_A_r2_desy_* and labels its rows A_desy_*), and the job lists
jobs/cells_stage3.tsv and jobs/cells_guardfix_rerun.tsv.

One row of an axis table is one agg_by_cell row (read by name: cell, arm, kind, name,
extrap_family, n_pix, n_tgt, n_src, mu, se, tau, margin, tost, guard_verdict, primary,
R_median, placebo_minus_notwc1b_ps, f_inflate) joined (merge_triples) with the mean over its
target triples of delta_by_triple's cov_C, lever_X and lever_X_all on (cell, arm, kind, name,
extrap_family), the per-row extrap flag 0/1 mapped to interp/extrap and a row without one to
"na". agg_by_cell has a lever_X_all column of its own, so the join names the two copies
lever_X_all_x and lever_X_all_y and the axis row's lever_X_all is empty on every row, as in the
study's tables. 24 columns (AXIS_COLUMNS), rows sorted by cell, arm (ARM_ORDER), kind, board or
pair and f. The guardfix tables add lpo_shape (the leave-pair-out shape of agg_by_cell's
side-car merge, collapsed by classify_lpo_shape) and the mean lpo_pairs of delta_by_triple.

Outputs, under the output folder, with the study's names (what etroc_plots' universality
reduction reads with --results-root):

  results_r2/axis_<A,B,C,D,E,I,J,K>.csv    the eight axes of the second wave
  results_r3/axis_F_long.csv, axis_F.csv   axis F, long and wide (one row per cell, kind, pair
                                           or board, family and primary f; the arms of
                                           WIDE_ARMS side by side)
  results_r3/summary_F.md                  the cell table of axis F (campaigns.AXIS_F_CELLS)
  results_r3/guardfix/axis_<E,J,K>_gf.csv  with --guardfix: the guardfix cells only

The guardfix tables are an overlay (overlay): their cells' rows replace that cell's rows of the
axis table, at the place of the cell's first row, the other cells keep theirs, and the two
columns the guardfix tables add are dropped.
"""
import json
import os

import numpy as np
import pandas as pd

from . import campaigns
from . import jobs as joblists
from .naming import log

TRIPLE_JOIN = ["cell", "arm", "kind", "name", "extrap_family"]
TRIPLE_MEANS = ["cov_C", "lever_X", "lever_X_all"]
AXIS_COLUMNS = ["axis", "cell", "arm", "kind", "board_or_role", "family", "n_pix", "n_tgt",
                "n_src", "mu_ps", "se_ps", "tau", "margin_ps", "tost", "guard_verdict",
                "spec_primary_row", "R_median", "placebo_minus_notwc1b_ps", "cov_C", "lever_X",
                "lever_X_all", "f_inflate", "f_variant_is_primary", "agg_tag"]
GUARDFIX_COLUMNS = ["lpo_shape", "lpo_pairs"]
# row order within a cell: the primary arms, the optimism and leave-pair-out controls, the
# controls, the variant transfers (axes of the second wave and the guardfix tables)
ARM_ORDER = ["xfer", "xfer_lpo",
             "self_in", "self_cf_lpo", "self_in_rep",
             "placebo", "placebo_self", "notwc",
             "xfer_1b:dut", "xfer_1b:ref", "xfer_1b:trig", "xfer_1b:extra",
             "xfer_calfix", "xfer_clip", "xfer_rescale", "xfer_shift"]
# axis F: the one-board reference, the shape-versus-slope pair, the parity-contrast pair and the
# pooled-mode arm added
ARM_ORDER_F = ["xfer", "xfer_lpo", "xfer_lin", "xfer_lin_lpo",
               "self_in", "self_cf_lpo", "self_in_rep",
               "placebo", "placebo_self", "notwc", "notwc_1b",
               "xfer_1b:dut", "xfer_1b:ref", "xfer_1b:trig", "xfer_1b:extra",
               "xfer_calfix", "xfer_clip", "xfer_rescale", "xfer_shift",
               "xfer_srcmatch", "xfer_srccross", "xfer_avg"]
WIDE_ARMS = ["xfer", "xfer_lin", "xfer_lin_lpo", "xfer_clip", "xfer_rescale"]
WIDE_KEY = ["cell", "kind", "board_or_role", "family", "agg_tag", "f_inflate"]
R2_AXES = ("A", "B", "C", "D", "E", "I", "J", "K")
F_CELL_HEADING = "## Cells and what each compares"


class AxesError(RuntimeError):
    """An input of the axis tables is missing or not what they read."""


# ------------------------------------------------------------------------------ rows
def merge_triples(agg, trip, extra=()):
    """agg_by_cell joined with the mean over target triples of delta_by_triple's TRIPLE_MEANS
    (and `extra`) on TRIPLE_JOIN: the per-row extrap 0/1 becomes interp/extrap, no flag "na"."""
    trip = trip.copy()
    trip["extrap_family"] = trip["extrap"].map({0: "interp", 1: "extrap", 0.0: "interp",
                                                1.0: "extrap"})
    trip.loc[trip["extrap_family"].isna(), "extrap_family"] = "na"
    cov = trip.groupby(TRIPLE_JOIN, dropna=False)[TRIPLE_MEANS + list(extra)].mean()
    return agg.merge(cov.reset_index(), on=TRIPLE_JOIN, how="left")


def read_tag(read_root, folder, extra=()):
    """merge_triples of one aggregation folder of the read root."""
    d = os.path.join(read_root, "tables", folder)
    for f in ("agg_by_cell.csv", "delta_by_triple.csv"):
        if not os.path.isfile(os.path.join(d, f)):
            raise AxesError("no %s" % os.path.join(d, f))
    agg = pd.read_csv(os.path.join(d, "agg_by_cell.csv"))
    trip = pd.read_csv(os.path.join(d, "delta_by_triple.csv"),
                       usecols=["cell", "arm", "kind", "name", "extrap"] + TRIPLE_MEANS
                       + list(extra))
    return merge_triples(agg, trip, extra)


def classify_lpo_shape(raw):
    """One leave-pair-out shape of an aggregation row: agg_by_cell's lpo_shape is a ";"-joined
    list of the distinct {"self": {board: "sibling"|"own"}, "xfer": {...}} records of its
    target triples. The union of every board's value over every record: all "sibling" ->
    "sibling" (every dropped pair guarded by a same-pixel sibling combination), all "own" ->
    "own" (every one fell back to the triple's own events), both -> "mixed"; no record (a row
    with no leave-pair-out guard, notwc or self_in) -> NaN."""
    if not isinstance(raw, str) or not raw or raw in ("nan", "{}"):
        return np.nan
    vals = set()
    for blob in raw.split(";"):
        blob = blob.strip()
        if not blob:
            continue
        try:
            d = json.loads(blob)
        except Exception:
            continue
        for side in ("self", "xfer"):
            for v in d.get(side, {}).values():
                vals.add(v)
    if not vals:
        return np.nan
    if vals == {"sibling"}:
        return "sibling"
    if vals == {"own"}:
        return "own"
    return "mixed"


def axis_row(axis, r, label, is_primary, guardfix=False):
    """One axis-table row (AXIS_COLUMNS, + GUARDFIX_COLUMNS) from one merged row `r`."""
    row = dict(
        axis=axis, cell=r["cell"], arm=r["arm"], kind=r["kind"],
        board_or_role=r["name"], family=r["extrap_family"],
        n_pix=int(r["n_pix"]), n_tgt=r.get("n_tgt", np.nan), n_src=r.get("n_src", np.nan),
        mu_ps=r["mu"], se_ps=r["se"], tau=r["tau"], margin_ps=r["margin"],
        tost=r["tost"], guard_verdict=r.get("guard_verdict", np.nan),
        spec_primary_row=(int(r.get("primary", -1)) if pd.notna(r.get("primary", np.nan))
                          else np.nan),
        R_median=r.get("R_median", np.nan),
        placebo_minus_notwc1b_ps=r.get("placebo_minus_notwc1b_ps", np.nan),
        cov_C=r.get("cov_C", np.nan), lever_X=r.get("lever_X", np.nan),
        lever_X_all=r.get("lever_X_all", np.nan),
        f_inflate=r["f_inflate"], f_variant_is_primary=is_primary, agg_tag=label)
    if guardfix:
        row.update(lpo_shape=classify_lpo_shape(r.get("lpo_shape", np.nan)),
                   lpo_pairs=r.get("lpo_pairs", np.nan))
    return row


def sort_axis(D, arm_order):
    """Rows by cell, arm (in `arm_order`, others last), kind, board or pair, f."""
    D = D.copy()
    D["_armkey"] = D["arm"].apply(lambda a: arm_order.index(a) if a in arm_order
                                  else len(arm_order))
    return D.sort_values(["cell", "_armkey", "kind", "board_or_role",
                          "f_inflate"]).drop(columns="_armkey")


def axis_table(axis, merged, arm_order=ARM_ORDER, cells=None, guardfix=False):
    """The table of one axis. `merged` is [(label, is_primary, merge_triples frame)] in the
    order of its tags; `cells`, when given, keeps only those cells."""
    rows = []
    for label, is_primary, m in merged:
        if cells is not None:
            m = m[m["cell"].isin(cells)]
        for _, r in m.iterrows():
            rows.append(axis_row(axis, r, label, is_primary, guardfix))
    return sort_axis(pd.DataFrame(rows), arm_order)


def wide_table(FL):
    """axis_F.csv from the long table of axis F: per (cell, kind, pair or board, family,
    agg_tag, f) of the primary f, the n_pix, mu, se, tau, TOST, guard and presence of each arm
    of WIDE_ARMS, the xfer row's cov_C, lever_X, lever_X_all and primary flag, and the xfer_lin
    mu minus the xfer mu. Rows by cell, kind (pair, board), pair or board, family (interp,
    extrap, na)."""
    prim = FL[FL.f_variant_is_primary == True].copy()          # noqa: E712
    wide_rows = []
    for key, g in prim.groupby(WIDE_KEY, dropna=False):
        rec = dict(zip(WIDE_KEY, key))
        by_arm = {r.arm: r for _, r in g.iterrows()}
        for arm in WIDE_ARMS:
            r = by_arm.get(arm)
            rec["%s_n_pix" % arm] = int(r.n_pix) if r is not None else np.nan
            rec["%s_mu_ps" % arm] = r.mu_ps if r is not None else np.nan
            rec["%s_se_ps" % arm] = r.se_ps if r is not None else np.nan
            rec["%s_tau" % arm] = r.tau if r is not None else np.nan
            rec["%s_tost" % arm] = r.tost if r is not None else np.nan
            rec["%s_guard_verdict" % arm] = r.guard_verdict if r is not None else np.nan
            rec["%s_present" % arm] = r is not None
        xr = by_arm.get("xfer")
        rec["cov_C"] = xr.cov_C if xr is not None else np.nan
        rec["lever_X"] = xr.lever_X if xr is not None else np.nan
        rec["lever_X_all"] = xr.lever_X_all if xr is not None else np.nan
        rec["spec_primary_row"] = xr.spec_primary_row if xr is not None else np.nan
        if rec.get("xfer_mu_ps") is not None and rec.get("xfer_lin_mu_ps") is not None \
                and pd.notna(rec["xfer_mu_ps"]) and pd.notna(rec["xfer_lin_mu_ps"]):
            rec["delta_xferlin_minus_xfer_mu_ps"] = rec["xfer_lin_mu_ps"] - rec["xfer_mu_ps"]
        else:
            rec["delta_xferlin_minus_xfer_mu_ps"] = np.nan
        wide_rows.append(rec)
    W = pd.DataFrame(wide_rows)
    W["_kindkey"] = W["kind"].map({"pair": 0, "board": 1}).fillna(2)
    W["_famkey"] = W["family"].map({"interp": 0, "extrap": 1, "na": 2}).fillna(3)
    return W.sort_values(["cell", "_kindkey", "board_or_role", "_famkey"]).drop(
        columns=["_kindkey", "_famkey"])


def overlay(base, gf):
    """The axis table `base` with the guardfix table `gf` laid over it: every cell of `gf`
    replaces all of that cell's rows of `base`, at the place of its first row (a cell `base`
    lacks goes at the end), the other cells keep theirs; the columns `gf` adds are dropped.
    Returns (frame, {cell: (rows before, rows after)})."""
    extra = [c for c in gf.columns if c not in base.columns]
    gf = gf.drop(columns=extra)[list(base.columns)]
    new = {c: g for c, g in gf.groupby("cell", sort=False)}
    parts, done, keep = [], set(), []
    for i, cell in enumerate(base["cell"].tolist()):
        if cell not in new:
            keep.append(i)
            continue
        if keep:
            parts.append(base.iloc[keep])
            keep = []
        if cell not in done:
            parts.append(new[cell])
            done.add(cell)
    if keep:
        parts.append(base.iloc[keep])
    parts += [g for c, g in new.items() if c not in done]
    counts = {c: (int((base.cell == c).sum()), len(g)) for c, g in new.items()}
    return pd.concat(parts, ignore_index=True), counts


# ---------------------------------------------------------------------- facts checks
def f_cell_table(stage3_jobs):
    """The lines of the cell table of axis F (campaigns.AXIS_F_CELLS), from its heading to the
    blank line after it; AxesError unless its labels are the level-F cells of the stage-3 job
    list (F<n> names the cell F<n>_<...>), one to one."""
    cells = sorted({j["cell_id"] for j in joblists.read_jobs(stage3_jobs) if j["level"] == "F"})
    labels = [lab for lab, _, _ in campaigns.AXIS_F_CELLS]
    hits = {lab: [c for c in cells if c.startswith(lab + "_")] for lab in labels}
    bad = {lab: h for lab, h in hits.items() if len(h) != 1}
    unnamed = [c for c in cells if not any(c in h for h in hits.values())]
    if bad or unnamed:
        raise AxesError("the cell table of axis F and the level-F cells of %s disagree: labels "
                        "naming not exactly one cell %s, cells without a label %s"
                        % (stage3_jobs, bad, unnamed))
    lines = [F_CELL_HEADING, "", "| cell | round-1 provenance | what `xfer_lin` decides |",
             "|---|---|---|"]
    lines += ["| %s | %s | %s |" % row for row in campaigns.AXIS_F_CELLS]
    return lines + [""]


def guardfix_cells(guardfix_jobs):
    """The cells of the guardfix job list, smoke tests left out."""
    return sorted({j["cell_id"] for j in joblists.read_jobs(guardfix_jobs)
                   if j["level"] != "smoke"})


# ------------------------------------------------------------------------------- run
def _write(D, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    D.to_csv(path, index=False)
    log("axes: %s %d rows, %d cells" % (path, len(D), D["cell"].nunique()))


def build(read_root, outdir, axes=None, guardfix=False):
    """Write the axis tables under `outdir`: without `guardfix`, the axes `axes` (default R2_AXES
    and F; F writes axis_F_long.csv, axis_F.csv and summary_F.md); with it, the guardfix tables
    of `axes` (default E, J, K). Returns {path: frame}."""
    written = {}
    if guardfix:
        cells = guardfix_cells(joblists.list_path("cells_guardfix_rerun.tsv"))
        for axis in axes or sorted(campaigns.GUARDFIX_AXIS_TAGS):
            merged = [(label, prim, read_tag(read_root, folder, extra=("lpo_pairs",)))
                      for folder, label, _, prim in campaigns.GUARDFIX_AXIS_TAGS[axis]]
            D = axis_table(axis, merged, cells=cells, guardfix=True)
            path = os.path.join(outdir, "results_r3", "guardfix", "axis_%s_gf.csv" % axis)
            _write(D, path)
            written[path] = D
            base = os.path.join(outdir, "results_r2", "axis_%s.csv" % axis)
            if os.path.isfile(base):
                _, counts = overlay(pd.read_csv(base), D)
                log("axes: the overlay on %s replaces %s (rows before -> after)"
                    % (base, ", ".join("%s %d->%d" % (c, *n) for c, n in sorted(counts.items()))))
        return written
    for axis in axes or R2_AXES + ("F",):
        merged = [(label, prim, read_tag(read_root, folder))
                  for folder, label, _, prim in campaigns.AXIS_TAGS[axis]]
        if axis != "F":
            D = axis_table(axis, merged)
            path = os.path.join(outdir, "results_r2", "axis_%s.csv" % axis)
            _write(D, path)
            written[path] = D
            continue
        FL = axis_table("F", merged, arm_order=ARM_ORDER_F)
        W = wide_table(FL)
        r3 = os.path.join(outdir, "results_r3")
        for name, D in (("axis_F_long.csv", FL), ("axis_F.csv", W)):
            _write(D, os.path.join(r3, name))
            written[os.path.join(r3, name)] = D
        md = os.path.join(r3, "summary_F.md")
        lines = f_cell_table(joblists.list_path("cells_stage3.tsv"))
        with open(md, "w") as fh:
            fh.write("\n".join(["# Axis F: the cells", ""] + lines) + "\n")
        log("axes: %s, the cell table of %d cells" % (md, len(campaigns.AXIS_F_CELLS)))
    return written


def cmd_axes(a, root):
    """`axes`: the axis tables from the read root into the write root (or --out, a relative one
    in the write root)."""
    outdir = campaigns.output_path(a.out, root)
    try:
        build(campaigns.transfer_root(), outdir, axes=a.axis, guardfix=a.guardfix)
    except AxesError as e:
        raise SystemExit("twc_transfer axes: %s" % e)
    print("-> %s" % outdir)
