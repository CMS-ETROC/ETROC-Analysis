"""The TWC-transfer tables, read, and the arithmetic the universality figures share.

The study fitted a time-walk correction (TWC) for every pixel of a run, a quadratic in TOT,

    delta(TOT) = c2 TOT^2 + c1 TOT + c0        (TOT and delta in ps)

and asked whether the correction of a source (a run, a board, a pixel) can be applied to a target
without losing time resolution; each such comparison is a "cell". The readers below return the
six tables of the universality/ inputs folder (made by `python -m
etroc_plots.universality.reduce`, which describes their sources); beside them is what more than one
figure needs: row filters, the good-run check of the IRRAD runs, the per-pixel curve arithmetic,
and each board's chip and conditions. Campaign codes are the study's: "desy" (DESY May 2026),
"aug" (DESY Aug 2026), "mar" and "jul" (CERN IRRAD March and July 2026); runs are "run26" (DESY)
and "h1_run19" (IRRAD, telescope first).
"""
import json
import re

import numpy as np
import pandas as pd

from .. import style
from ..campaigns import active as campaign, desy_2026
from ..resolution import tables
from .reduce import BEST_KEYS, IRRAD_CODES

HIST_BIN_WIDTH_PS = 25.0       # the bank's TOT histograms: 25 ps bins from hist_lo
# the order of the level cells: (a) floor, (b) angle, (b') angle on the offset holder, (c) pixel
# to board, (d) board to board, (e) pooled source, (g) fluence, (h) March to July
LEVORDER = {"a": 0, "b": 1, "bp": 2, "c": 3, "d": 4, "e": 5, "g": 6, "h": 7}


# ------------------------------------------------------------------------------------ readers
def read_cells():
    """univ_cells.csv: one row per cell (cell, level_or_axis, the source and target campaign and
    run, base_cell, dbias_v). src_run is empty for a pooled source."""
    return pd.read_csv(campaign.UNIV_CELLS_CSV)


def natkey(cell):
    """'a1', 'a2', ..., 'g11', 'E13_...': letters, then the number as a number."""
    m = re.match(r"([A-Za-z]+)(\d+)", cell)
    return (m.group(1), int(m.group(2))) if m else (cell, 0)


def override_verdicts(df):
    """The study's verdict override on master-cell rows (columns guard_verdict, final,
    all_verdict, verdict_pair_interp), as a new frame: a cell whose leave-pair-out guard was
    absent or uninformative and whose final verdict is not a FAIL reads NA-UNGUARDED in
    all_verdict (a guarded FAIL outranks "unguarded"); the pair-interp verdict FAIL+UNGUARDED
    (a guarded FAIL beside an unguarded row) reads FAIL."""
    out = df.copy()
    unguarded = (out.guard_verdict.astype(str).str.contains("absent|uninformative")
                 & ~out["final"].astype(str).str.startswith("FAIL"))
    out["all_verdict"] = np.where(unguarded, "NA-UNGUARDED", out.all_verdict)
    out["verdict_pair_interp"] = out.verdict_pair_interp.replace({"FAIL+UNGUARDED": "FAIL"})
    return out


def read_master():
    """univ_master_cells.csv, in level order (LEVORDER, then natkey), with the study's verdict
    override (override_verdicts). interp_pct = 100 n_interp / (n_interp + n_extrap)."""
    df = pd.read_csv(campaign.UNIV_MASTER_CSV)
    df["_lev"] = df.level.map(LEVORDER)
    df["_nat"] = df.cell.map(natkey)
    df = override_verdicts(df.sort_values(["_lev", "_nat"]).reset_index(drop=True))
    df["interp_pct"] = 100.0 * df.n_interp / (df.n_interp + df.n_extrap).replace(0, np.nan)
    return df.drop(columns=["_lev", "_nat"])


def read_axis_r2():
    """univ_axis_r2.csv: one row per axis, cell, arm, kind, board_or_role, family, f_inflate and
    self_transfer (without_self_transfer drops the rows whose source triple is the target
    triple)."""
    return pd.read_csv(campaign.UNIV_AXIS_R2_CSV)


def read_axis_f():
    """univ_axis_F_wide.csv (one row per cell, kind, board_or_role and family, the arms side by
    side, each with its <arm>_self_transfer flag) with margin_ps from the long table's xfer row of
    the same cell, kind, board_or_role, family, pixel count and flag (xfer_n_pix = n_pix,
    xfer_self_transfer = self_transfer: the long table holds a second xfer row on some pairs, the
    self-transfer one), and base_cell from univ_cells."""
    wide = pd.read_csv(campaign.UNIV_AXIS_F_WIDE_CSV)
    long = pd.read_csv(campaign.UNIV_AXIS_F_LONG_CSV)
    for table in (wide, long):
        self_transfer_columns(table)       # raises on a table reduced before the flag
    keys = ["cell", "kind", "board_or_role", "family"]
    marg = (long[long.arm == "xfer"][keys + ["n_pix", "self_transfer", "margin_ps"]]
            .drop_duplicates()
            .rename(columns={"n_pix": "xfer_n_pix", "self_transfer": "xfer_self_transfer"}))
    marg[["xfer_n_pix", "xfer_self_transfer"]] = marg[["xfer_n_pix", "xfer_self_transfer"]].astype(
        float)
    wide = wide.merge(marg, on=keys + ["xfer_n_pix", "xfer_self_transfer"], how="left",
                      validate="many_to_one")
    base = read_cells().set_index("cell")["base_cell"]
    wide["base_cell"] = wide.cell.map(base)
    return wide


def read_bank_best():
    """twc_bank_best.csv: one row per campaign, run and pixel (board_id, row, col), the fit with
    the most events; the floats read back exactly as written."""
    df = pd.read_csv(campaign.TWC_BANK_BEST_CSV, float_precision="round_trip",
                     dtype={"campaign": str, "run": str, "telescope": str})
    df["telescope"] = df["telescope"].fillna("")
    return df


# ------------------------------------------------------------------------------------ filters
def primary_rows(df, kind="pair", arm="xfer"):
    """The rows a figure reads from an axis table: the cell's primary f variant
    (f_variant_is_primary), one kind (pair or board) and one arm (xfer, the transferred
    correction)."""
    return df[(df.f_variant_is_primary == True) & (df.kind == kind) & (df.arm == arm)]  # noqa: E712


def family_rows(df, family):
    """Rows whose target TOT lies inside the source's TOT range (family "interp") or reaches
    beyond it ("extrap")."""
    if family not in ("interp", "extrap"):
        raise ValueError("family %r: interp or extrap" % family)
    return df[df.family == family]


def self_transfer_columns(df):
    """The self-transfer flags of an axis table: self_transfer, or in the wide axis-F table one
    per arm, <arm>_self_transfer. Raises when there is none (a table reduced before the flag)."""
    cols = [c for c in df.columns if c == "self_transfer" or c.endswith("_self_transfer")]
    if not cols:
        raise KeyError("the table has no self_transfer column: reduce it again with python -m "
                       "etroc_plots.universality.reduce")
    return cols


def without_self_transfer(df):
    """`df` without the rows whose source triple is the target triple (flag 1 in any of its
    self_transfer_columns; an arm a wide row does not hold has no flag). The study leaves these
    rows out of its per-cell summaries; its axis tables carry them, flagged by the reduction."""
    cols = self_transfer_columns(df)
    flags = df[cols]
    odd = ~(flags.isin([0, 1]) | flags.isna()).all(axis=1)
    if odd.any():
        raise ValueError("self_transfer values other than 0, 1 and empty: %s"
                         % sorted(set(flags[odd].stack())))
    return df[~(flags == 1).any(axis=1)]


def is_irrad(code):
    return code in IRRAD_CODES


def irrad_run(code, run):
    """("march" / "july", telescope, run number) of an IRRAD run ("mar", "h1_run19")."""
    tel, num = run.split("_run")
    return IRRAD_CODES[code], tel, int(num)


_GOOD = {}


def good_run_drops(rows, pairs=(("src_campaign", "src_run"), ("tgt_campaign", "tgt_run"))):
    """{(code, run): reason} for every IRRAD run of `rows` (any frame with the `pairs` columns)
    that is not marked good in the good-run list (GOOD_RUNS_CSV, its IRRAD_Mar2026 and
    IRRAD_Jul2026 rows). DESY runs are not on that list and are not checked."""
    from ..iv.iv_timeseries import good_runs
    drops = {}
    for _, r in rows.iterrows():
        for c, k in pairs:
            code, run = r[c], r[k]
            if not is_irrad(code) or not isinstance(run, str):
                continue
            camp, tel, num = irrad_run(code, run)
            if (camp, tel) not in _GOOD:
                _GOOD[(camp, tel)] = good_runs(tel, good_runs_campaign=(
                    campaign.GOOD_RUNS_CAMPAIGN_MARCH if camp == "march"
                    else campaign.GOOD_RUNS_CAMPAIGN))
            good, reasons = _GOOD[(camp, tel)]
            if num not in good:
                drops[(code, run)] = reasons.get(num, "not on the good-run list")
    return drops


# ------------------------------------------------------------------------ per-pixel curves
def best_rows(bank, code, run):
    """One run's rows of read_bank_best(), indexed by (board_id, row, col)."""
    sub = bank[(bank.campaign == code) & (bank.run == run)]
    if sub.empty:
        raise KeyError("no %s %s in the bank table" % (code, run))
    return sub.set_index(list(BEST_KEYS)).sort_index()


def pick_pixel(best_by_run, require_keys=None, prefer_keys=None):
    """Rank the pixels of {key: best_rows(...)}: most runs carrying the pixel first, then the
    larger n_fit summed over those runs. `require_keys`: runs that must carry it (the source);
    `prefer_keys`: the runs counted (default all). Returns [(coverage, summed n_fit, (board_id,
    row, col)), ...], best first; a tie keeps the order the pixels were met in."""
    runs = list(prefer_keys or best_by_run.keys())
    require_keys = require_keys or []
    all_pix = set()
    for r in runs:
        all_pix |= set(best_by_run[r].index)
    rows = []
    for pix in all_pix:
        if any(pix not in best_by_run[r].index for r in require_keys):
            continue
        cov = sum(1 for r in runs if pix in best_by_run[r].index)
        nsum = sum(best_by_run[r].loc[pix, "n_fit"] for r in runs if pix in best_by_run[r].index)
        rows.append((cov, nsum, pix))
    rows.sort(key=lambda t: (-t[0], -t[1]))
    return rows


def delta_tot(row, tot):
    """The correction delta(TOT) in ps of one fit, TOT in ps."""
    return row.c2 * tot ** 2 + row.c1 * tot + row.c0


def mean_over_range(row, lo, hi, n=500):
    """The mean of delta(TOT) over a uniform grid of n points on [lo, hi]: the constant removed
    from a curve so that curves compare in shape (lo, hi = the source's q05 and q95)."""
    tot = np.linspace(lo, hi, n)
    return float(delta_tot(row, tot).mean())


def curve_xy(row, lo, hi, n=400):
    tot = np.linspace(lo, hi, n)
    return tot, delta_tot(row, tot)


def hist_edges_counts(row):
    """The fit's TOT histogram: bin edges in ps (HIST_BIN_WIDTH_PS from hist_lo) and counts."""
    counts = np.array(json.loads(row["hist"]), dtype=float)
    edges = row["hist_lo"] + HIST_BIN_WIDTH_PS * np.arange(len(counts) + 1)
    return edges, counts


def merge_bins(edges, counts, k):
    """The histogram with every k neighbouring bins summed, from the first edge; a last bin
    short of k bins gets empty ones added (its edge extends past the data)."""
    counts = np.asarray(counts, dtype=float)
    width = float(edges[1] - edges[0])
    n = -(-len(counts) // k)
    padded = np.concatenate([counts, np.zeros(n * k - len(counts))])
    return edges[0] + k * width * np.arange(n + 1), padded.reshape(n, k).sum(axis=1)


def curve(row, src_lo, src_hi, merge=1):
    """One pixel's curve over its own TOT range [tot_min, tot_max], mean-removed over the
    source's [src_lo, src_hi], with its TOT histogram (`merge` neighbouring 25 ps bins summed,
    merge_bins): dict(x, y in ps, mean_removed, tot_min, tot_max, edges, counts, counts_table
    (the table's 25 ps bins), row)."""
    lo, hi = float(row.tot_min), float(row.tot_max)
    x, y = curve_xy(row, lo, hi)
    m = mean_over_range(row, src_lo, src_hi)
    edges, counts_table = hist_edges_counts(row)
    edges, counts = merge_bins(edges, counts_table, merge)
    return dict(x=x, y=y - m, mean_removed=m, tot_min=lo, tot_max=hi, edges=edges,
                counts=counts, counts_table=counts_table, row=row)


# ------------------------------------------------------------------- chips and conditions
def role_boards(bank, code, run):
    """{role word: board_id} of one run, from the role column of read_bank_best() (the combo the
    study's fits of that run used). Raises when a board carries two roles or a role two boards."""
    sub = bank[(bank.campaign == code) & (bank.run == run)][["board_id", "role"]].drop_duplicates()
    if sub.empty:
        raise KeyError("no %s %s in the bank table" % (code, run))
    if sub.board_id.duplicated().any() or sub.role.duplicated().any():
        raise ValueError("%s %s: the bank's roles and boards do not pair one to one: %s"
                         % (code, run, sorted(zip(sub.board_id, sub.role))))
    return {r: int(b) for b, r in zip(sub.board_id, sub.role)}


def pair_chips(rs, bank, code, run, board_or_role, sep=" / "):
    """The chips of a row's role words in one run: 'ref-dut' of DESY May run 26 ->
    'PT-IH5 / PT-IH2' (role_boards of that run, then board_conditions; the separator is not a
    hyphen, which the DESY chip names carry). A word that is not a role of the run is kept as it
    is."""
    roles = role_boards(bank, code, run)
    return sep.join(board_conditions(rs, code, run, roles[w])["chip"] if w in roles else w
                    for w in str(board_or_role).split("-"))


def run_conditions(rs, code, run):
    """board_conditions of every board of one run, in board order."""
    if is_irrad(code):
        camp, tel, num = irrad_run(code, run)
        ids = [b["board_idx"] for b in tables.run_settings(rs, campaign.TABLE_CAMPAIGN[camp],
                                                           campaign.TABLE_TEL[tel], num)]
    else:
        ids = [b["board_idx"] for b in desy_2026.run_settings(desy_2026.BANK_CAMPAIGN[code], run)]
    return [board_conditions(rs, code, run, i) for i in sorted(ids)]


def run_steps(rs, code, run):
    """The fluence steps (style.fluence_key) of the boards of an IRRAD run, sorted; empty for a
    DESY run and for no run (a pooled source)."""
    if not is_irrad(code) or not isinstance(run, str):
        return []
    return sorted({style.fluence_key(c["fluence"]) for c in run_conditions(rs, code, run)})


def cells_at_steps(cells, rs, steps):
    """{cell: [(code, run, step), ...]} for every cell of `cells` (univ_cells rows) whose source
    or target is an IRRAD run with a board at one of the fluence `steps`."""
    out = {}
    for _, r in cells.iterrows():
        hits = [(code, run, st) for code, run in ((r.src_campaign, r.src_run),
                                                 (r.tgt_campaign, r.tgt_run))
                for st in run_steps(rs, code, run) if st in steps]
        if hits:
            out[r.cell] = hits
    return out


def board_conditions(rs, code, run, board_id):
    """One board of one run: dict(campaign (the module key: "march", "july", "may", "aug"), chip
    (as legends print it), hv, fluence (IRRAD bookkeeping, None for DESY), angle and temp (DESY,
    None for IRRAD), offset, rfsel, rfsel_source, power). IRRAD boards come from runs_summary
    (`rs`), whose March rows carry no RFSel: those read MARCH_RFSEL; their chip is
    style.chip_short of its name (IH11, LF10). DESY boards come from desy_2026.run_settings, their
    chip the yaml's short name as it is (PT-IH5, PT-IH3, PT-IH2, B1-P1-9): style.chip_short would
    drop the PT- of the DESY chips and make them read like IRRAD ones."""
    board_id = int(board_id)
    if is_irrad(code):
        camp, tel, num = irrad_run(code, run)
        b = [x for x in tables.run_settings(rs, campaign.TABLE_CAMPAIGN[camp],
                                            campaign.TABLE_TEL[tel], num)
             if x["board_idx"] == board_id]
        if len(b) != 1:
            raise KeyError("runs_summary has no board %d in %s %s" % (board_id, code, run))
        b = b[0]
        rfsel, source = b["rfsel"], "runs_summary"
        if rfsel is None and camp == "march":
            rfsel, source = campaign.MARCH_RFSEL, "MARCH_RFSEL (not in runs_summary)"
        return dict(campaign=camp, chip=style.chip_short(b["chip"]), hv=b["hv"],
                    fluence=b["fluence"], angle=None, temp=None, offset=b["offset"],
                    rfsel=rfsel, rfsel_source=source, power=campaign.PREAMP_POWER)
    camp = desy_2026.BANK_CAMPAIGN[code]
    b = [x for x in desy_2026.run_settings(camp, run) if x["board_idx"] == board_id]
    if len(b) != 1:
        raise KeyError("the %s yaml has no board %d in %s" % (camp, board_id, run))
    b = b[0]
    return dict(campaign=camp, chip=b["chip"], hv=b["hv"], fluence=None, angle=b["angle"],
                temp=b["temp"], offset=b["offset"], rfsel=b["rfsel"],
                rfsel_source="desy_2026 (yaml RFSel key, else DEFAULT_RFSEL)",
                power=desy_2026.PREAMP_POWER[camp])


def settings_text(conds, sep=", "):
    """The operating point of some boards of one campaign (board_conditions dicts), worded by
    style.settings_text: their RFSel and threshold offset (a list where they differ) and the
    preamp power. A clause the campaign did not record (None) is left out."""
    rfsels = {c["rfsel"] for c in conds}
    if None in rfsels and len(rfsels) > 1:
        raise ValueError("RFSel recorded for some boards only: %s" % rfsels)
    rfsels = sorted(r for r in rfsels if r is not None)
    offsets = sorted({c["offset"] for c in conds})
    powers = {c["power"] for c in conds}
    if len(powers) != 1:
        raise ValueError("boards at different preamp power %s" % powers)
    power = powers.pop()
    rfsel = None if not rfsels else (rfsels[0] if len(rfsels) == 1 else rfsels)
    offset = offsets[0] if len(offsets) == 1 else offsets
    if power is not None:
        return style.settings_text(rfsel=rfsel, offset=offset, power=power, sep=sep)
    tail = sep + "preamp - power"
    text = style.settings_text(rfsel=rfsel, offset=offset, power="-", sep=sep)
    if not text.endswith(tail):
        raise ValueError("style.settings_text no longer ends with the power clause: %r" % text)
    return text[:-len(tail)]
