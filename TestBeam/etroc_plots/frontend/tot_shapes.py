"""TOT and time-difference (dTOA) shapes from the analysis pipeline's track files, per board,
per pixel and per board pair.

The track files (ANALYSIS_DIRS in the campaign module) hold, per run and per combo of three
boards, one parquet file per pixel triple (one pixel on each board): every event in it is a
three-board track through that triple. A file is named after the boards' roles in that run,
track_<role initial>-R<row>C<col>_..._.parquet (d dut, r ref, e extra, t trig), and carries per
board the columns tot_<role>, toa_<role> and cal_<role>: TOT and TOA in ps, converted with one
bin per loop file (3.125 ns over the mean CAL of that file's events, the pipeline's
core/apply_tdc_cuts.convert_to_time), CAL in codes per event. A combo folder is named by the roles at its board indices in the
run's config yaml (extra0-dut1-ref2; the role at an index changes between runs, the chip does
not), so the chip of a board is always TELESCOPE_CHIPS[telescope][board index].

Combos and triples. Boards 0, 1 and 2 are read from the combo of boards 0-1-2, board 3 from the
combo 1-2-3 (REFERENCE). A pixel's triple is one file of that combo holding the pixel: the one
with the most events (rank="rows"; ties go to the larger file, then to the first file name), or
the largest file (rank="bytes", ties to the first file name; the rule of the first presented
figures). The events of a file are its row count, read from the parquet footer. A pixel
counts for a board when its triple holds at least FLOOR events.

What the tables hold (`python -m etroc_plots.frontend.rebuild --part tracks`; the columns are
described in the campaign's inputs document):

  triples           every track file of every run: its pixels, rows and bytes
  per pixel         the median TOT of the pixel's triple (TOT_PIXEL_MEDIANS_CSV)
  per board         over the pixels above the floor: the median of their medians with the 16th
                    and 84th percentiles, and the pooled TOT of all their events (median, 16th
                    and 84th percentiles, mode at 0.05, 0.10 and 0.20 ns, histogram in 0.05 ns
                    bins over 0-20 ns) (TOT_POOLED_SUMMARY_CSV, TOT_POOLED_HIST_CSV)
  dTOA per pair     for every pixel of board A above the floor, its triple: TOA corrected for
                    time walk by the joint solve over the three boards of the triple
                    (core/twc_solver.py), dt = TOA_A - TOA_B, centred on its own median and
                    histogrammed in 25 ps bins over +-2 ns, then summed over the triples. Pairs
                    0-1 and 0-2 from board 0's triples, 1-2 from board 1's, 2-3 from board 2's
                    triples in the combo 1-2-3 (DTOA_TRIPLES_CSV, DTOA_POOLED_HIST_CSV)
  fixed pixels      choose(): one pixel per board kept fixed over the runs of shape_runs(), one
                    pixel pair per board pair, one board-1 pixel over the July runs; their TOT
                    (in ns, and per TOT code by tot_codes()) and dTOA histograms
                    (PIXEL_HISTS_CSV)
  RFSel pairs       rfsel_rows(): per pair of runs of RFSEL_TOT_PAIRS and per group of
                    TOT_TIME_CONTROL, the pooled TOT of one chip over the pixels above the floor
                    in every run of the pair or group (RFSEL_PAIR_HIST_CSV)

The RFSel pairs are read back by rfsel_curves (one panel: the two curves of a pair, checked against
the runs summary and, when given the pooled tables, against each run's pooled TOT) and
median_shift (the two stored medians and their difference).

The drawing (board_vs_fluence, shape_grid, run_columns, rfsel_pairs and its per-axes
draw_rfsel_pair) takes its words as arguments.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from ..campaigns import active as campaign
from .. import style
from ..etroc_style import BOARD_COLOR, RFSEL_COLOR
from ..resolution import tables
from . import blnw

FLOOR = 300                       # events in a pixel's triple for the pixel to count
NBOARDS = 4
REFERENCE = {0: (0, 1, 2), 1: (0, 1, 2), 2: (0, 1, 2), 3: (1, 2, 3)}
COMBOS = ((0, 1, 2), (1, 2, 3))
PAIRS = ((0, 1), (0, 2), (1, 2), (2, 3))
PAIR_COMBO = {(0, 1): (0, 1, 2), (0, 2): (0, 1, 2), (1, 2): (0, 1, 2), (2, 3): (1, 2, 3)}
RANK_KEYS = {"rows": ["n_rows", "n_bytes"], "bytes": ["n_bytes"]}
TOT_BIN_NS, TOT_HI_NS = 0.05, 20.0
NB_TOT = int(round(TOT_HI_NS / TOT_BIN_NS))
MODE_BINS_NS = (0.05, 0.10, 0.20)
DT_BIN_PS, DT_HALF_PS = 25.0, 2000.0
NB_DT = int(round(2 * DT_HALF_PS / DT_BIN_PS))
TOA_CODE_PS = 3125.0              # one TOA code = 3.125 ns / CAL
# The TOT code ladder of the pipeline's step 10 (TOT is 9 bits): code c is converted to
# m(c) = 2 c - floor(c / 32) times the bin of its loop file, 3.125 ns / the mean CAL. The same
# arrays as condor_at_lxplus/utils/fourboard_pairing.py (_TT, _LUT), defined here so the package
# does not import the pipeline.
TOT_CODES = np.arange(512)
TOT_LADDER = 2 * TOT_CODES - np.floor(TOT_CODES / 32)
# colour = RFSel on the RFSel comparison (one fluence step, so the ladder colour carries nothing
# there): etroc_style.RFSEL_COLOR, Okabe-Ito hues, no orange (the 3.5e15 colour of the ladder)
ROLE_INITIAL = {"d": "dut", "r": "ref", "e": "extra", "t": "trig"}

_PIX = ["%s_%d" % (k, b) for b in range(NBOARDS) for k in ("chip", "row", "col")]
_KEY = ["campaign", "tel", "run"]
TRIPLES_COLUMNS = _KEY + ["boards", "combo"] + _PIX + ["n_rows", "n_bytes", "file"]
PIXEL_COLUMNS = _KEY + ["board_idx", "chip", "row", "col", "n", "tot_median_ns", "passes_floor"]
SUMMARY_COLUMNS = (_KEY + ["board_idx", "chip", "n_pixels", "n_events", "pixmed_median_ns",
                           "pixmed_p16_ns", "pixmed_p84_ns", "median_ns", "p16_ns", "p84_ns"]
                   + ["mode_ns_%.2f" % w for w in MODE_BINS_NS] + ["n_outside_hist"])
TOT_HIST_COLUMNS = _KEY + ["board_idx", "chip", "bin_lo_ns", "count"]
DTOA_COLUMNS = (_KEY + ["pair", "board_a", "chip_a", "board_b", "chip_b", "combo"]
                + ["%s_%d" % (k, b) for b in range(NBOARDS) for k in ("row", "col")]
                + ["n_events", "dt_median_ps", "n_outside", "cal_a", "cal_b"])
DTOA_HIST_COLUMNS = _KEY + ["pair", "board_a", "chip_a", "board_b", "chip_b", "bin_lo_ps",
                            "count"]
PIXEL_HISTS_COLUMNS = (_KEY + ["pick", "quantity", "board_a", "chip_a", "row_a", "col_a",
                               "board_b", "chip_b", "row_b", "col_b", "combo", "file",
                               "n_events", "median", "n_outside", "bin_lo", "count", "cal_mode"])
RFSEL_COLUMNS = (_KEY + ["board_idx", "chip", "rfsel", "offset", "partner_run", "n_pixels",
                         "n_events", "median_ns", "bin_lo_ns", "count", "group"])


# ------------------------------------------------------------------------------- naming
_CONFIGS = {}
_TRACK = re.compile(r"([tdre])-R(\d+)C(\d+)")


def run_roles(camp, tel, run):
    """The role of each board index in a run ([role of board 0, ...]), from the run-config yaml
    (RUN_LIST_YAML); a run without an entry takes the telescope's default block. Raises
    ValueError when the yaml's chip at an index is not TELESCOPE_CHIPS'."""
    if camp not in _CONFIGS:
        _CONFIGS[camp] = blnw.board_configs(campaign.RUN_LIST_YAML[camp])
    per_run, defaults = _CONFIGS[camp]
    out = []
    for idx, chip in enumerate(campaign.TELESCOPE_CHIPS[tel]):
        b = per_run.get((tel, int(run), idx)) or defaults.get((tel, chip))
        if b is None or style.chip_short(b["short"]) != chip:
            raise ValueError("%s %s run %s board %d: no yaml entry for chip %s"
                             % (camp, tel, run, idx, chip))
        out.append(str(b["role"]))
    return out


def boards_text(boards):
    return "-".join(str(b) for b in boards)


def combo_name(roles, boards):
    """The combo folder of `boards` in a run with `roles` ('extra0-dut1-ref2')."""
    return "-".join("%s%d" % (roles[b], b) for b in boards)


def track_dir(camp, tel, run, combo):
    return os.path.join(campaign.ANALYSIS_DIRS[camp].format(tel=tel, run=int(run)), combo, "time")


def track_path(t):
    """Full path of a triples row (a Series or namedtuple with campaign, tel, run, combo, file)."""
    return os.path.join(track_dir(t.campaign, t.tel, t.run, t.combo), t.file)


def track_pixels(name):
    """{role: (row, col)} from a track file name."""
    return {ROLE_INITIAL[m.group(1)]: (int(m.group(2)), int(m.group(3)))
            for m in _TRACK.finditer(os.path.basename(name))}


# ------------------------------------------------------------------------------- triples
def scan_run(camp, tel, run):
    """Every track file of the two combos of a run as rows of the triples table (one row per
    file: the pixel of each of its boards, the row count from the parquet footer, the size)."""
    import pyarrow.parquet as pq
    roles = run_roles(camp, tel, run)
    chips = campaign.TELESCOPE_CHIPS[tel]
    rows = []
    for boards in COMBOS:
        combo = combo_name(roles, boards)
        files = sorted(glob.glob(os.path.join(track_dir(camp, tel, run, combo), "track_*.parquet")))
        if not files:
            raise FileNotFoundError("no track files in %s" % track_dir(camp, tel, run, combo))
        for p in files:
            px = track_pixels(p)
            rec = dict(campaign=camp, tel=tel, run=int(run), boards=boards_text(boards),
                       combo=combo)
            for b in range(NBOARDS):
                inside = b in boards
                rec["chip_%d" % b] = chips[b] if inside else None
                rec["row_%d" % b] = px[roles[b]][0] if inside else None
                rec["col_%d" % b] = px[roles[b]][1] if inside else None
            rec.update(n_rows=int(pq.read_metadata(p).num_rows), n_bytes=int(os.path.getsize(p)),
                       file=os.path.basename(p))
            rows.append(rec)
    return as_triples(pd.DataFrame(rows, columns=TRIPLES_COLUMNS))


def as_triples(df):
    """The triples table with integer pixel columns (empty where the board is not in the combo)."""
    for b in range(NBOARDS):
        for k in ("row", "col"):
            df["%s_%d" % (k, b)] = df["%s_%d" % (k, b)].astype("Int64")
    return df


def read_triples(path=None):
    return as_triples(pd.read_csv(path or campaign.TRIPLES_CSV))


def ranked(t, rank):
    """Triples rows best first: most rows (then most bytes) or most bytes; ties to the first file."""
    keys = RANK_KEYS[rank]
    return t.sort_values(keys + ["file"], ascending=[False] * len(keys) + [True], kind="mergesort")


def best_triples(tri, board, rank="rows", boards=None):
    """One row per pixel of `board`: its triple among the triples of one run (rows of the triples
    table) in the combo `boards` (default REFERENCE[board]), sorted by pixel."""
    t = tri[tri["boards"] == boards_text(boards or REFERENCE[board])]
    t = ranked(t, rank).drop_duplicates(["row_%d" % board, "col_%d" % board], keep="first")
    return t.sort_values(["row_%d" % board, "col_%d" % board], kind="mergesort").reset_index(drop=True)


def pixel_counts(tri, board, rank="rows"):
    """{(row, col): events of the pixel's triple} for the pixels of `board` at or above FLOOR."""
    bt = best_triples(tri, board, rank)
    bt = bt[bt["n_rows"] >= FLOOR]
    return {(int(r), int(c)): int(n) for r, c, n in
            zip(bt["row_%d" % board], bt["col_%d" % board], bt["n_rows"])}


def _pixel_of(t, board):
    return (int(t["row_%d" % board]), int(t["col_%d" % board]))


# ------------------------------------------------------------------------------- reading
def read_tot(path, role, with_cal=False):
    """TOT of one board of a track file, ps, missing values dropped; with `with_cal`, the arrays
    (tot, cal, file) of those events: the board's CAL code and the loop-file index of each."""
    if not with_cal:
        v = pd.read_parquet(path, columns=["tot_" + role])["tot_" + role].to_numpy(dtype=float)
        return v[~np.isnan(v)]
    df = pd.read_parquet(path, columns=["tot_" + role, "cal_" + role, "file"])
    df = df[df["tot_" + role].notna()]
    return (df["tot_" + role].to_numpy(dtype=float), df["cal_" + role].to_numpy(dtype=float),
            df["file"].to_numpy())


def tot_codes(tot_ps, file_idx, cal):
    """The TOT codes of one board's events, the exact inverse of step 10's conversion
    (core/apply_tdc_cuts.convert_to_time, as utils/fourboard_pairing.raw_codes): per loop file
    (`file_idx`) the bin is 3.125 ns / the mean of `cal` over that file's events (0.2 ns when the
    mean is 0), m = TOT / bin, code = the code whose m(code) is m. Raises ValueError naming the
    loop file when a value is not on the ladder (|m - round(m)| > 1e-6, or no code has that m)."""
    tot_ps = np.asarray(tot_ps, dtype=float)
    file_idx = np.asarray(file_idx)
    mean = pd.Series(np.asarray(cal, dtype=float)).groupby(file_idx).transform("mean").to_numpy()
    bin_ns = np.full(len(mean), 0.2)
    nz = mean != 0
    bin_ns[nz] = 3.125 / mean[nz]
    m = tot_ps / 1e3 / bin_ns
    r = np.round(m)
    on = np.isfinite(m) & (np.abs(m - r) <= 1e-6)
    code = np.round(np.interp(np.where(on, m, 0.0), TOT_LADDER, TOT_CODES)).astype(np.int64)
    on &= TOT_LADDER[code] == r
    if not on.all():
        i = int(np.argmin(on))
        raise ValueError("loop file %s: TOT %.4f ps is not on the TOT code ladder (m = %.6f)"
                         % (file_idx[i], tot_ps[i], m[i]))
    return code


def code_ladder_ns(codes, cal_mode):
    """TOT in ns of TOT codes at one CAL code: m(code) x 3.125 ns / cal_mode."""
    return TOT_LADDER[np.asarray(codes, dtype=np.int64)] * 3.125 / float(cal_mode)


def code_edges_ns(lo_code, hi_code, cal_mode):
    """Edges in ns of one bin per TOT code from lo_code to hi_code at one CAL code: the bin of
    code c runs from m(c - 1) to m(c) on the ladder, times 3.125 ns / cal_mode (below code 0,
    m(-1) = -1 by the same formula), so a code's bin does not depend on the range. Every bin is
    2 x 3.125 ns / cal_mode wide except the code at each multiple of 32, half that: the ladder
    skips one half-code there, and the hits say the skipped half belongs to that code, not to the
    one below."""
    lo = int(lo_code)
    below = TOT_LADDER[lo - 1] if lo > 0 else 2 * (lo - 1) - (lo - 1) // 32
    m = TOT_LADDER[np.arange(lo, int(hi_code) + 1)]
    return np.concatenate(([below], m)) * 3.125 / float(cal_mode)


def read_track(path, roles):
    """tot, toa and cal of the three boards of a track file, events with any missing value
    dropped (none in the 2026 files)."""
    cols = ["%s_%s" % (q, r) for r in roles for q in ("tot", "toa", "cal")]
    return pd.read_parquet(path, columns=cols).dropna().reset_index(drop=True)


def corrected_toa(df, roles):
    """{role: TOA corrected for time walk, ps}: the joint least-squares solve over the boards of
    the track (core/twc_solver.py, reached as a namespace package from TestBeam/)."""
    from condor_at_lxplus.core import twc_solver
    tots = {r: df["tot_" + r].to_numpy() for r in roles}
    toas = {r: df["toa_" + r].to_numpy() for r in roles}
    return twc_solver.apply_timewalk_correction_arrays(tots, toas, roles)


def tot_hist(v_ns):
    return np.histogram(v_ns, bins=NB_TOT, range=(0.0, TOT_HI_NS))[0]


def dt_hist(dt_ps):
    return np.histogram(dt_ps, bins=NB_DT, range=(-DT_HALF_PS, DT_HALF_PS))[0]


def tot_bin_lo(i):
    return round(i * TOT_BIN_NS, 3)


def dt_bin_lo(i):
    return round(-DT_HALF_PS + i * DT_BIN_PS, 1)


def mode_from_hist(counts, bin_w, store_w=TOT_BIN_NS):
    """Centre of the peak bin of a histogram rebinned to `bin_w` (a multiple of the stored bin)."""
    factor = int(round(bin_w / store_w))
    n = (len(counts) // factor) * factor
    c = np.asarray(counts[:n]).reshape(-1, factor).sum(axis=1)
    return (int(np.argmax(c)) + 0.5) * bin_w


def centred_dt(corr, ra, rb):
    """(dt - its median, the median) of corrected TOA of role ra minus role rb, ps."""
    dt = np.asarray(corr[ra] - corr[rb], dtype=float)
    med = float(np.median(dt))
    return dt - med, med


# ------------------------------------------------------------------------------- one run
def run_tables(camp, tel, run, tri, rank="rows", log=None):
    """The per-pixel, per-board and dTOA rows of one run (dict of row lists keyed pixel,
    summary, tot_hist, dtoa, dtoa_hist), from that run's triples rows."""
    roles = run_roles(camp, tel, run)
    chips = campaign.TELESCOPE_CHIPS[tel]
    key = dict(campaign=camp, tel=tel, run=int(run))
    out = {k: [] for k in ("pixel", "summary", "tot_hist", "dtoa", "dtoa_hist")}
    acc = {p: np.zeros(NB_DT, dtype=np.int64) for p in PAIRS}
    dt_pairs = {0: [(0, 1), (0, 2)], 1: [(1, 2)], 2: [(2, 3)]}

    def dtoa(t, a, df):
        boards = PAIR_COMBO[dt_pairs[a][0]]
        croles = [roles[b] for b in boards]
        if len(df) < FLOOR:
            return
        corr = corrected_toa(df, croles)
        for pa, pb in dt_pairs[a]:
            dtc, med = centred_dt(corr, roles[pa], roles[pb])
            c = dt_hist(dtc)
            acc[(pa, pb)] += c
            rec = dict(key, pair="%d-%d" % (pa, pb), board_a=pa, chip_a=chips[pa], board_b=pb,
                       chip_b=chips[pb], combo=t.combo)
            for b in range(NBOARDS):
                inside = b in boards
                rec["row_%d" % b] = int(t["row_%d" % b]) if inside else None
                rec["col_%d" % b] = int(t["col_%d" % b]) if inside else None
            rec.update(n_events=int(len(dtc)), dt_median_ps=round(med, 3),
                       n_outside=int(len(dtc) - int(c.sum())),
                       cal_a=float(np.median(df["cal_" + roles[pa]])),
                       cal_b=float(np.median(df["cal_" + roles[pb]])))
            out["dtoa"].append(rec)

    for b in range(NBOARDS):
        croles = [roles[x] for x in REFERENCE[b]]
        chunks, medians = [], []
        for _, t in best_triples(tri, b, rank).iterrows():
            passes = int(t["n_rows"]) >= FLOOR
            if passes and b in (0, 1):            # the same triple gives board b's dTOA pairs
                df = pd.read_parquet(track_path(t), columns=[
                    "%s_%s" % (q, r) for r in croles for q in ("tot", "toa", "cal")])
                v = df["tot_" + roles[b]].to_numpy(dtype=float)
                v = v[~np.isnan(v)]
                dtoa(t, b, df.dropna().reset_index(drop=True))
            else:
                v = read_tot(track_path(t), roles[b])
            med = float(np.median(v)) / 1000.0 if len(v) else float("nan")
            r, c = _pixel_of(t, b)
            out["pixel"].append(dict(key, board_idx=b, chip=chips[b], row=r, col=c,
                                     n=int(t["n_rows"]), tot_median_ns=med, passes_floor=passes))
            if passes and len(v):
                chunks.append(v / 1000.0)
                medians.append(med)
        if not chunks:
            continue
        vals = np.concatenate(chunks)
        counts = tot_hist(vals)
        p16, p50, p84 = np.quantile(vals, [0.16, 0.5, 0.84])
        m50 = float(np.median(medians))
        m16, m84 = np.quantile(medians, [0.16, 0.84])
        rec = dict(key, board_idx=b, chip=chips[b], n_pixels=len(chunks), n_events=int(len(vals)),
                   pixmed_median_ns=m50, pixmed_p16_ns=float(m16), pixmed_p84_ns=float(m84),
                   median_ns=float(p50), p16_ns=float(p16), p84_ns=float(p84))
        for w in MODE_BINS_NS:
            rec["mode_ns_%.2f" % w] = mode_from_hist(counts, w)
        rec["n_outside_hist"] = int((vals < 0.0).sum() + (vals >= TOT_HI_NS).sum())
        out["summary"].append(rec)
        out["tot_hist"] += [dict(key, board_idx=b, chip=chips[b], bin_lo_ns=tot_bin_lo(i),
                                 count=int(counts[i])) for i in np.nonzero(counts)[0]]
    # pair 2-3: board 2's triples in the combo 1-2-3
    croles = [roles[x] for x in (1, 2, 3)]
    for _, t in best_triples(tri, 2, rank, boards=(1, 2, 3)).iterrows():
        if int(t["n_rows"]) >= FLOOR:
            dtoa(t, 2, read_track(track_path(t), croles))
    for p in PAIRS:
        out["dtoa_hist"] += [dict(key, pair="%d-%d" % p, board_a=p[0], chip_a=chips[p[0]],
                                  board_b=p[1], chip_b=chips[p[1]], bin_lo_ps=dt_bin_lo(i),
                                  count=int(acc[p][i])) for i in np.nonzero(acc[p])[0]]
    if log:
        log("%s %s run %d: %d pixels, %d dTOA triples" % (camp, tel, run, len(out["pixel"]),
                                                           len(out["dtoa"])))
    return out


# ------------------------------------------------------------------------------- fixed pixels
def shape_runs(tel):
    """[(campaign, run), ...], one run per fluence step, lowest first: March the runs of
    MARCH_DISPLAY_RUNS, July the preferred display run at FRONTEND_JULY_SETTINGS."""
    july = tables.preferred_runs(campaign.DISPLAY_RUNS_JUL_CSV, **campaign.FRONTEND_JULY_SETTINGS)
    return ([("march", r) for r in campaign.MARCH_DISPLAY_RUNS[tel]]
            + [("july", r) for r in july[tel]])


def pick_pixel(by_run, board, rank="rows"):
    """(pixel, {run key: events}, number of candidates): among the pixels of `board` at or above
    FLOOR in every run of `by_run` ({run key: triples rows of that run}), the one whose smallest
    count over the runs is largest; ties to the smallest (row, col)."""
    per = {k: pixel_counts(t, board, rank) for k, t in by_run.items()}
    common = set.intersection(*(set(d) for d in per.values()))
    if not common:
        raise ValueError("board %d: no pixel above the floor in all of %s" % (board, list(per)))
    best = max(sorted(common), key=lambda px: min(d[px] for d in per.values()))
    return best, {k: d[best] for k, d in per.items()}, len(common)


def _with(t, pix):
    m = pd.Series(True, index=t.index)
    for b, (r, c) in pix.items():
        m &= (t["row_%d" % b] == r) & (t["col_%d" % b] == c)
    return t[m]


def pair_tracks(by_run, pixels, rank="rows"):
    """{pair: dict(pixels={board: (row, col)}, tracks={run key: triples row}, n_candidates)}: per
    board pair one pixel pair fixed over the runs and, per run, the triple with the most events
    holding both. Pairs inside 0-1-2 take the two boards' fixed pixels (`pixels`). For 2-3 board 3
    keeps its pixel and board 2 takes, among the board-2 pixels tracked with it in every run, the
    one whose best such triple has the largest smallest count over the runs (the two boards' own
    pixels are rarely tracked together)."""
    out = {}
    pop = RANK_KEYS[rank][0]
    for pair in PAIRS[:3]:
        pix = {b: pixels[b] for b in pair}
        tracks = {}
        for k, t in by_run.items():
            sel = _with(t[t["boards"] == boards_text(PAIR_COMBO[pair])], pix)
            if not len(sel):
                raise ValueError("pair %s, run %s: no triple holds pixels %s" % (pair, k, pix))
            tracks[k] = ranked(sel, rank).iloc[0]
        out[pair] = dict(pixels=pix, tracks=tracks, n_candidates=1)
    per = {}
    for k, t in by_run.items():
        sel = _with(t[t["boards"] == boards_text(PAIR_COMBO[(2, 3)])], {3: pixels[3]})
        sel = ranked(sel, rank).drop_duplicates(["row_2", "col_2"], keep="first")
        per[k] = {_pixel_of(r, 2): r for _, r in sel.iterrows()}
    common = set.intersection(*(set(d) for d in per.values()))
    if not common:
        raise ValueError("pair 2-3: no board-2 pixel tracked with %s in every run" % (pixels[3],))
    px2 = max(sorted(common), key=lambda px: min(int(d[px][pop]) for d in per.values()))
    out[(2, 3)] = dict(pixels={2: px2, 3: pixels[3]}, tracks={k: d[px2] for k, d in per.items()},
                       n_candidates=len(common))
    return out


def choose(tri, tel, rank="rows", runs=None, july_board=1):
    """The fixed pixels of the shape figures of one telescope, from the triples table:
    runs (default shape_runs(tel)); pixels {board: (pixel, {run: events}, candidates)} over those
    runs; pairs (pair_tracks); july: the board-`july_board` pixel over the July runs among them,
    with its triple per run (its TOT and the dTOA with the next board come from that triple)."""
    runs = list(runs or shape_runs(tel))
    by_run = {k: tri[(tri["campaign"] == k[0]) & (tri["tel"] == tel) & (tri["run"] == k[1])]
              for k in runs}
    for k, t in by_run.items():
        if not len(t):
            raise ValueError("%s %s run %d: not in the triples table" % (k[0], tel, k[1]))
    pixels = {b: pick_pixel(by_run, b, rank) for b in range(NBOARDS)}
    best = {k: {b: best_triples(t, b, rank) for b in range(NBOARDS)} for k, t in by_run.items()}
    tracks = {b: {k: _with(best[k][b], {b: pixels[b][0]}).iloc[0] for k in runs}
              for b in range(NBOARDS)}
    july_runs = [k for k in runs if k[0] == "july"]
    jp = pick_pixel({k: by_run[k] for k in july_runs}, july_board, rank)
    jt = {k: _with(best[k][july_board], {july_board: jp[0]}).iloc[0] for k in july_runs}
    return dict(tel=tel, rank=rank, runs=runs, pixels=pixels, tracks=tracks,
                pairs=pair_tracks(by_run, {b: p[0] for b, p in pixels.items()}, rank),
                july=dict(board=july_board, pair=(july_board, july_board + 1), runs=july_runs,
                          pixel=jp[0], counts=jp[1], n_candidates=jp[2], tracks=jt))


def _hist_row(key, pick, quantity, t, a, b, n, med, nout, lo, cnt):
    chips = campaign.TELESCOPE_CHIPS[key["tel"]]
    return dict(key, pick=pick, quantity=quantity, board_a=a, chip_a=chips[a],
                row_a=int(t["row_%d" % a]), col_a=int(t["col_%d" % a]),
                board_b=b, chip_b=None if b is None else chips[b],
                row_b=None if b is None else int(t["row_%d" % b]),
                col_b=None if b is None else int(t["col_%d" % b]),
                combo=t["combo"], file=t["file"], n_events=n, median=med, n_outside=nout,
                bin_lo=lo, count=cnt)


def tot_rows(key, pick, t, board):
    """The TOT rows of one board's pixel in its triple: quantity "tot_ns" (0.05 ns bins) and
    "tot_code" (one row per TOT code, tot_codes(); cal_mode = the board's most frequent CAL code
    over the triple's events, ties to the smallest; median = the lower median code)."""
    roles = run_roles(key["campaign"], key["tel"], key["run"])
    tot, cal, f = read_tot(track_path(t), roles[board], with_cal=True)
    v = tot / 1000.0
    c = tot_hist(v)
    med = float(np.median(v)) if len(v) else float("nan")
    rows = [_hist_row(key, pick, "tot_ns", t, board, None, int(len(v)), med,
                      int(len(v) - c.sum()), tot_bin_lo(i), int(c[i])) for i in np.nonzero(c)[0]]
    if not len(v):
        return rows
    codes = tot_codes(tot, f, cal)
    cc = np.bincount(codes, minlength=len(TOT_CODES))
    mode = int(np.argmax(np.bincount(cal[np.isfinite(cal)].astype(np.int64))))
    cmed = int(np.sort(codes)[(len(codes) - 1) // 2])
    return rows + [dict(_hist_row(key, pick, "tot_code", t, board, None, int(len(v)), cmed, 0,
                                  int(i), int(cc[i])), cal_mode=mode) for i in np.nonzero(cc)[0]]


def dt_rows(key, pick, t, pair):
    roles = run_roles(key["campaign"], key["tel"], key["run"])
    croles = [roles[b] for b in PAIR_COMBO[pair]]
    df = read_track(track_path(t), croles)
    dtc, med = centred_dt(corrected_toa(df, croles), roles[pair[0]], roles[pair[1]])
    c = dt_hist(dtc)
    return [_hist_row(key, pick, "dtoa_ps", t, pair[0], pair[1], int(len(dtc)), round(med, 3),
                      int(len(dtc) - c.sum()), dt_bin_lo(i), int(c[i])) for i in np.nonzero(c)[0]]


def pixel_hist_rows(choice, log=None):
    """The rows of PIXEL_HISTS_CSV for one telescope's choose() result: per run, the TOT of each
    board's fixed pixel (pick "pixel"), the dTOA of each fixed pixel pair ("pixel_pair"), and in
    the July runs the TOT and the dTOA with the next board of the board-1 pixel ("july_pixel")."""
    rows = []
    tel = choice["tel"]
    for k in choice["runs"]:
        key = dict(campaign=k[0], tel=tel, run=int(k[1]))
        for b in range(NBOARDS):
            rows += tot_rows(key, "pixel", choice["tracks"][b][k], b)
        for pair in PAIRS:
            rows += dt_rows(key, "pixel_pair", choice["pairs"][pair]["tracks"][k], pair)
        if k in choice["july"]["runs"]:
            t = choice["july"]["tracks"][k]
            rows += tot_rows(key, "july_pixel", t, choice["july"]["board"])
            rows += dt_rows(key, "july_pixel", t, choice["july"]["pair"])
        if log:
            log("%s %s run %d: pixel histograms" % (k[0], tel, k[1]))
    return rows


# ------------------------------------------------------------------------------- RFSel pairs
def rfsel_board(tri, camp, tel, pairs, rank="rows"):
    """(board, {board: {run: events}}): over the runs of `pairs`, the board whose smallest total
    of events in its pixels above the floor is largest; ties to the first chip name."""
    runs = sorted({r for p in pairs for r in p})
    totals = {}
    for b in range(NBOARDS):
        totals[b] = {r: int(sum(pixel_counts(tri[(tri["campaign"] == camp) & (tri["tel"] == tel)
                                                 & (tri["run"] == r)], b, rank).values()))
                     for r in runs}
    chips = campaign.TELESCOPE_CHIPS[tel]
    order = sorted(range(NBOARDS), key=lambda b: chips[b])
    return max(order, key=lambda b: min(totals[b].values())), totals


def rfsel_rows(tri, rs, camp, tel, groups, board, rank="rows", log=None):
    """The rows of RFSEL_PAIR_HIST_CSV: per group of runs (an RFSel pair, or a longer group such
    as the time control), for board `board` (rfsel_board() over the RFSel pairs), the pooled TOT
    (0.05 ns bins) of the pixels above the floor in every run of the group, each run read from
    its own triples; RFSel and threshold offset from the runs summary `rs`. Every row names its
    group ("52+86"); partner_run is the other run of a pair, empty in a longer group."""
    chips = campaign.TELESCOPE_CHIPS[tel]
    rows = []
    for group in groups:
        sub = {r: tri[(tri["campaign"] == camp) & (tri["tel"] == tel) & (tri["run"] == r)]
               for r in group}
        above = {r: set(pixel_counts(t, board, rank)) for r, t in sub.items()}
        common = set.intersection(*above.values())
        name = "+".join(str(r) for r in group)
        if log:
            log("RFSel group %s, board %d (%s): %d pixels above the floor in every run (%s)"
                % (name, board, chips[board], len(common),
                   ", ".join("run %d: %d" % (r, len(above[r])) for r in group)))
        for k, r in enumerate(group):
            partner = int(group[1 - k]) if len(group) == 2 else None
            roles = run_roles(camp, tel, r)
            bt = best_triples(sub[r], board, rank)
            bt = bt[[_pixel_of(t, board) in common for _, t in bt.iterrows()]]
            v = [read_tot(track_path(t), roles[board]) for _, t in bt.iterrows()]
            v = np.concatenate(v) / 1000.0 if v else np.zeros(0)
            c = tot_hist(v)
            st = [s for s in tables.run_settings(rs, campaign.TABLE_CAMPAIGN[camp],
                                                 campaign.TABLE_TEL[tel], r)
                  if s["board_idx"] == board][0]
            rows += [dict(campaign=camp, tel=tel, run=int(r), board_idx=board, chip=chips[board],
                          rfsel=st["rfsel"], offset=st["offset"], partner_run=partner,
                          n_pixels=len(common), n_events=int(len(v)),
                          median_ns=float(np.median(v)), bin_lo_ns=tot_bin_lo(i), count=int(c[i]),
                          group=name)
                     for i in np.nonzero(c)[0]]
            if log:
                log("RFSel group %s: run %d %s, %d pixels, %d events, median %.4f ns"
                    % (name, r, chips[board], len(common), len(v), float(np.median(v))))
    return rows


def counted_pixels(pixels, camp, tel, run, board):
    """{(row, col): events of its triple} for the pixels of `board` counted in one run: the rows
    of TOT_PIXEL_MEDIANS_CSV (`pixels`) that pass the floor."""
    p = pixels[(pixels["campaign"] == camp) & (pixels["tel"] == tel) & (pixels["run"] == run)
               & (pixels["board_idx"] == board) & pixels["passes_floor"]]
    return {(int(r), int(c)): int(n) for r, c, n in zip(p["row"], p["col"], p["n"])}


def _pair_rows(hist_df, camp, tel, run, partner):
    g = hist_df[(hist_df["campaign"] == camp) & (hist_df["tel"] == tel)
                & (hist_df["run"] == run) & (hist_df["partner_run"] == partner)]
    if not len(g):
        raise ValueError("RFSel pair table: no rows for %s %s run %d with partner %d"
                         % (camp, tel, run, partner))
    return g


def median_shift(hist_df, camp, tel, pair):
    """The median TOT of the pair's chip in each run of `pair` (the stored, unbinned medians of
    RFSEL_PAIR_HIST_CSV, `hist_df`) and the shift from the first run to the second. Returns
    dict(runs, chip, board_idx, rfsel, offset, n_pixels, n_events, median_ns, shift_ns): rfsel,
    offset, n_events and median_ns as [first run, second run]; n_pixels the pixels counted in
    both runs."""
    rows = [_pair_rows(hist_df, camp, tel, r, p).iloc[0]
            for r, p in ((pair[0], pair[1]), (pair[1], pair[0]))]
    npix = {int(r["n_pixels"]) for r in rows}
    boards = {int(r["board_idx"]) for r in rows}
    if len(npix) != 1 or len(boards) != 1:
        raise ValueError("RFSel pair %s: pixels %s, boards %s" % (pair, npix, boards))
    med = [float(r["median_ns"]) for r in rows]
    return dict(runs=[int(pair[0]), int(pair[1])], chip=str(rows[0]["chip"]),
                board_idx=boards.pop(), rfsel=[int(r["rfsel"]) for r in rows],
                offset=[int(r["offset"]) for r in rows], n_pixels=npix.pop(),
                n_events=[int(r["n_events"]) for r in rows], median_ns=med,
                shift_ns=med[1] - med[0])


def rfsel_curves(hist_df, campaign_key, tel, pair, rs, run_times, stop_utc, *, label, note,
                 width=0.1, pooled=None):
    """One panel of rfsel_pairs / draw_rfsel_pair from RFSEL_PAIR_HIST_CSV (`hist_df`): the TOT
    of the pair's chip in each run of `pair` over the pixels counted in both runs, rebinned to
    `width` ns, as the fraction of the curve's events, with its stored (unbinned) median.

    rs, run_times, stop_utc  the runs summary (the chip's RFSel and threshold offset per run must
                equal the table's; the bias goes to the records), the run start times and the
                end of the step's irradiation (frontend.blnw.time_after_stop)
    label       format string per curve, fields rfsel (style.rfsel_text), run, time (after the
                stop), median (ns) and n (events)
    note        format string heading the legend, fields chip and n_pixels
    pooled      None, or dict(pixels, summary, hists): TOT_PIXEL_MEDIANS_CSV, TOT_POOLED_SUMMARY_CSV
                indexed by (campaign, tel, run, board_idx), and {(campaign, tel, run, board_idx):
                rows of TOT_POOLED_HIST_CSV}. When given, each curve must be its run's pooled TOT
                of the chip less the pixels not counted in the partner run, bin by bin (raises
                otherwise); the numbers of that check go to "controls"
    Returns dict(subject, note, curves, records, controls, hists, hv): curves (label, color,
    edges, y, median, rfsel) sorted by RFSel; records (one per run: the values-file row),
    controls, hists (the rebinned (edges, counts)) and hv (the chip's bias) in the order of
    `pair`."""
    curves, records, controls, hists, hv = [], [], [], [], []
    npix, offsets, boards = set(), set(), set()
    for run, partner in ((pair[0], pair[1]), (pair[1], pair[0])):
        g = _pair_rows(hist_df, campaign_key, tel, run, partner)
        row = g.iloc[0]
        board = int(row["board_idx"])
        boards.add(board)
        chip = campaign.TELESCOPE_CHIPS[tel][board]
        (s,) = [x for x in tables.run_settings(rs, campaign.TABLE_CAMPAIGN[campaign_key],
                                               campaign.TABLE_TEL[tel], run)
                if x["board_idx"] == board]
        if (s["rfsel"], s["offset"]) != (int(row["rfsel"]), int(row["offset"])):
            raise ValueError("run %d: RFSel / offset %s in the run summary, %s in the table" % (
                run, (s["rfsel"], s["offset"]), (int(row["rfsel"]), int(row["offset"]))))
        e, c = hist_of(g, "bin_lo_ns", TOT_BIN_NS, 0.0, TOT_HI_NS)
        if pooled is not None:
            # the chip's pooled TOT in this run less the pixels not counted in the partner run
            # is this curve, bin by bin
            mine = counted_pixels(pooled["pixels"], campaign_key, tel, run, board)
            other = counted_pixels(pooled["pixels"], campaign_key, tel, partner, board)
            left = sorted(set(mine) - set(other))
            summ = pooled["summary"].loc[(campaign_key, tel, run, board)]
            _, c_all = hist_of(pooled["hists"][(campaign_key, tel, run, board)], "bin_lo_ns",
                               TOT_BIN_NS, 0.0, TOT_HI_NS)
            ctl = dict(run=int(run), partner_run=int(partner),
                       board_pixels=int(summ["n_pixels"]), board_events=int(summ["n_events"]),
                       pixels_left_out=len(left), events_left_out=sum(mine[p] for p in left),
                       common_pixels=len(set(mine) & set(other)),
                       min_bin_difference=int((c_all - c).min()),
                       difference_events=int((c_all - c).sum()), curve_events=int(c.sum()))
            if (ctl["min_bin_difference"] < 0 or ctl["common_pixels"] != int(row["n_pixels"])
                    or ctl["difference_events"] != ctl["events_left_out"]
                    or ctl["curve_events"] != int(row["n_events"])
                    or ctl["board_pixels"] - ctl["pixels_left_out"] != int(row["n_pixels"])
                    or ctl["board_events"] - ctl["events_left_out"] != int(row["n_events"])):
                raise ValueError("run %d: the RFSel curve is not the pooled TOT less the pixels "
                                 "left out: %s, table %d pixels %d events"
                                 % (run, ctl, row["n_pixels"], row["n_events"]))
            controls.append(ctl)
        text, hours = blnw.time_after_stop(run_times, campaign_key, tel, [run], stop_utc)
        ew, cw = rebin(e, c, width)
        hists.append((ew, cw))
        med = float(row["median_ns"])
        curves.append(dict(rfsel=s["rfsel"], color=RFSEL_COLOR[s["rfsel"]], edges=ew,
                           y=cw / float(row["n_events"]), median=med,
                           label=label % dict(rfsel=style.rfsel_text(s["rfsel"]), run=run,
                                              time=text, median=med, n=int(row["n_events"]))))
        records.append(dict(pair=list(pair), run=int(run), partner_run=int(partner), chip=chip,
                            board_idx=board, rfsel=s["rfsel"],
                            rf_kohm=style.RFSEL_RF_KOHM[s["rfsel"]], offset=s["offset"],
                            hv=s["hv"], hours=round(hours[run], 2),
                            n_pixels=int(row["n_pixels"]), n_events=int(row["n_events"]),
                            median_ns=round(med, 4)))
        npix.add(int(row["n_pixels"]))
        offsets.add(s["offset"])
        hv.append(s["hv"])
    if len(npix) != 1 or len(offsets) != 1 or len(boards) != 1:
        raise ValueError("pair %s: pixels %s, offsets %s, boards %s" % (pair, npix, offsets,
                                                                        boards))
    curves.sort(key=lambda cv: cv["rfsel"])
    chip = campaign.TELESCOPE_CHIPS[tel][boards.pop()]
    return dict(subject=style.settings_text(rfsel=[cv["rfsel"] for cv in curves],
                                            offset=offsets.pop()),
                note=note % dict(chip=chip, n_pixels=npix.pop()), curves=curves,
                records=records, controls=controls, hists=hists, hv=hv)


# ------------------------------------------------------------------------------- readers
def hist_of(df, lo_col, store_w, lo, hi):
    """(edges, counts) of one stored histogram (rows with `lo_col`, count) rebinned to the grid
    from `lo` to `hi` in steps of `store_w` (the stored bin)."""
    n = int(round((hi - lo) / store_w))
    counts = np.zeros(n, dtype=np.int64)
    i = np.round((df[lo_col].to_numpy(dtype=float) - lo) / store_w).astype(int)
    ok = (i >= 0) & (i < n)
    np.add.at(counts, i[ok], df["count"].to_numpy()[ok])
    return lo + np.arange(n + 1) * store_w, counts


def rebin(edges, counts, width):
    """Sum adjacent bins to `width` (a multiple of the bin); a last partial group is dropped."""
    f = int(round(width / (edges[1] - edges[0])))
    n = (len(counts) // f) * f
    return edges[:n + 1:f], counts[:n].reshape(-1, f).sum(axis=1)


def hist_quantile(edges, counts, q):
    """The q quantile of a binned distribution, linear within the bin."""
    c = np.cumsum(counts, dtype=float)
    if not len(c) or c[-1] <= 0:
        return float("nan")
    i = min(int(np.searchsorted(c, q * c[-1], side="left")), len(counts) - 1)
    prev = c[i - 1] if i else 0.0
    return float(edges[i] + (q * c[-1] - prev) / max(counts[i], 1) * (edges[i + 1] - edges[i]))


def common_range(hists, width, symmetric=False, lo_q=0.005, hi_q=0.995, floor=None):
    """One x range for several histograms [(edges, counts), ...]: the outer envelope of their
    0.5 and 99.5 % points, rounded out to `width` plus one bin each side; symmetric about 0 if
    asked; not below `floor`."""
    lo = min(hist_quantile(e, c, lo_q) for e, c in hists)
    hi = max(hist_quantile(e, c, hi_q) for e, c in hists)
    if symmetric:
        m = (np.ceil(max(abs(lo), abs(hi)) / width) + 1) * width
        return (-float(m), float(m))
    lo = np.floor(lo / width) * width - width
    hi = np.ceil(hi / width) * width + width
    return (float(lo if floor is None else max(lo, floor)), float(hi))


def crop(edges, counts, x0, x1):
    """The bins of (edges, counts) from x0 to x1 (both on the bin grid)."""
    i0 = int(np.argmin(np.abs(edges - x0)))
    i1 = int(np.argmin(np.abs(edges - x1)))
    return edges[i0:i1 + 1], counts[i0:i1]


def group_hists(hist_df, camp, tel, group, width):
    """The curves of one group of runs of the RFSel table (RFSEL_PAIR_HIST_CSV rows; `group` as
    stored, "32+52+86"), per run in the group's order: {run: dict(edges, y, median_ns,
    n_events, n_pixels, rfsel, offset)}, y the fraction of the run's events per bin of `width`
    (summed from the stored 0.05 ns bins). A run of the group without rows raises ValueError."""
    d = hist_df[(hist_df["campaign"] == camp) & (hist_df["tel"] == tel)
                & (hist_df["group"].astype(str) == group)]
    out = {}
    for run in (int(r) for r in group.split("+")):
        g = d[d["run"] == run]
        if not len(g):
            raise ValueError("RFSel table: no rows of %s %s run %d in group %s"
                             % (camp, tel, run, group))
        first = g.iloc[0]
        e, c = rebin(*hist_of(g, "bin_lo_ns", TOT_BIN_NS, 0.0, TOT_HI_NS), width)
        n = int(first["n_events"])
        out[run] = dict(edges=e, y=c / float(n), median_ns=float(first["median_ns"]), n_events=n,
                        n_pixels=int(first["n_pixels"]), rfsel=int(first["rfsel"]),
                        offset=int(first["offset"]))
    return out


# ------------------------------------------------------------------------------- drawing
BOARD_DX = 0.11          # x offset between the boards of one run, in step units


def board_vs_fluence(points, steps_by_tel, series, *, tick_labels, xlabel, ylabel, tel_titles,
                     tag, data, footer, boundary_after=None, board_dx=BOARD_DX, scale=0.72):
    """A board statistic against the fluence step, one panel per telescope (columns): x = the
    steps (categories, `tick_labels[tel]`), colour = board, marker = chip.

    points  [dict(tel, step, board_idx, chip, value, lo, hi, kind, dx), ...]: lo/hi the bar ends
            (None: no bar); dx an extra x offset in step units (runs side by side)
    series  {kind: dict(label, open, line, size)}: the legend words of each kind of point (a
            string, or {tel: string} when the words differ by panel), an open marker, a line
            joining the points of one board over the steps, marker size factor; drawn in the
            order given
    board_dx  x offset between the boards of one run, in step units
    Returns the figure."""
    tels = list(tel_titles)
    fig, axes = blnw._grid(1, len(tels), aw=7.4, ah=4.6, ml=1.25, mr=0.3, mt=0.62,
                           mb=1.15 + style.footer_extra_in(footer, scale, fits=4),
                           gx=0.85, gy=0.0, scale=scale)
    s = style.sizes(scale)
    vals = [v for p in points for v in (p["value"], p.get("lo"), p.get("hi")) if v is not None]
    span = (max(vals) - min(vals)) or 1.0
    for j, tel in enumerate(tels):
        ax = axes[0][j]
        n = len(steps_by_tel[tel])
        for k, (kind, sp) in enumerate(series.items()):
            pts = [p for p in points if p["tel"] == tel and p["kind"] == kind]
            for b in sorted({p["board_idx"] for p in pts}):
                bp = sorted([p for p in pts if p["board_idx"] == b], key=lambda p: p["step"])
                kw = style.point_style(bp[0]["chip"], 0.0, scale=scale * sp.get("size", 1.0),
                                       line=False, open_marker=bool(sp.get("open")),
                                       color=BOARD_COLOR[b])
                xs = [p["step"] + (b - 1.5) * board_dx + p.get("dx", 0.0) for p in bp]
                if sp.get("line"):
                    ax.plot(xs, [p["value"] for p in bp], color=BOARD_COLOR[b], zorder=2,
                            linewidth=1.3 * scale, linestyle=style.chip_linestyle(bp[0]["chip"]))
                for x, p in zip(xs, bp):
                    yerr = None
                    if p.get("lo") is not None:
                        yerr = [[p["value"] - p["lo"]], [p["hi"] - p["value"]]]
                    ax.errorbar([x], [p["value"]], yerr=yerr, zorder=3 + k, **kw)
        ax.set_ylim(min(vals) - 0.06 * span, max(vals) + 0.32 * span)
        if boundary_after is not None:
            ax.axvline(boundary_after + 0.5, color=style.INK_MUTED, linewidth=0.9, linestyle=":",
                       zorder=1)
        ax.set_xlim(-0.6, n - 0.4)
        ax.set_xticks(range(n))
        ax.set_xticklabels(tick_labels[tel], fontsize=s["tick"] * 0.78)
        ax.minorticks_off()
        ax.set_ylabel(ylabel if j == 0 else "")
        ax.set_xlabel(xlabel)
        style.style_axes(ax, scale, minor=False)
        ax.tick_params(axis="x", labelsize=s["tick"] * 0.78)
        lg = ax.legend(handles=blnw.board_handles(tel, scale), loc="upper left", ncol=4,
                       fontsize=s["legend"] * 0.9, handletextpad=0.2, columnspacing=0.9,
                       borderaxespad=0.4)
        ax.add_artist(lg)
        ax.legend(handles=series_handles(series, scale, tel), loc="upper right",
                  fontsize=s["legend"] * 0.9, handletextpad=0.2, borderaxespad=0.4)
    style.compound_header(axes[0], tag=tag, data=data, scale=scale,
                          subjects=[tel_titles[t] for t in tels])
    style.footer(fig, footer, scale)
    return fig


def series_handles(series, scale=1.0, tel=None):
    out = []
    for sp in series.values():
        label = sp["label"][tel] if isinstance(sp["label"], dict) else sp["label"]
        out.append(Line2D([], [], color=style.INK_MUTED, marker="o",
                          markersize=8.0 * scale * sp.get("size", 1.0),
                          markerfacecolor=style.SURFACE if sp.get("open") else style.INK_MUTED,
                          markeredgecolor=style.INK_MUTED, markeredgewidth=1.6 * scale,
                          linestyle="-" if sp.get("line") else "none", linewidth=1.3 * scale,
                          label=label))
    return out


# 2 x 4 compound: columns 0-1 the first telescope, 2-3 the second; panel geometry in inches
CW, CH = 3.30, 2.95
CGAP, HALFGAP, RGAP = 0.62, 0.95, 0.95
ML, MR, MT, MB = 1.10, 0.32, 0.63, 0.95
YTOP = 1.15              # y max = this times the tallest curve of a telescope's four panels


def shape_grid(panels, *, xlabel, ylabel, ranges, tag, data, footer, scale=0.52,
               legend_scale=0.95, ytop=YTOP):
    """Two telescopes side by side, four panels each (two rows of two), one step curve per run.

    panels  {tel: [dict(subject, note, curves=[dict(label, color, edges, y)]), x 4]}: in the
            order row 0 left, row 0 right, row 1 left, row 1 right; y the value per bin
    ranges  {tel: (x0, x1)}; one y scale per telescope (ytop times its tallest curve)
    The subject of a panel is its right title; the note heads its legend. Returns the figure."""
    import matplotlib.pyplot as plt
    style.apply_style(scale)
    W = ML + 4 * CW + 3 * CGAP + HALFGAP + MR
    H = MB + 2 * CH + RGAP + MT
    fig = plt.figure(figsize=(W, H))
    axes = [[None] * 4 for _ in range(2)]
    for i in range(2):
        for j in range(4):
            x0 = ML + j * (CW + CGAP) + (HALFGAP - CGAP if j >= 2 else 0.0)
            axes[i][j] = fig.add_axes([x0 / W, (MB + (1 - i) * (CH + RGAP)) / H, CW / W, CH / H])
    s = style.sizes(scale)
    for half, tel in enumerate(panels):
        top = ytop * max(float(np.max(c["y"])) for p in panels[tel] for c in p["curves"])
        for k, p in enumerate(panels[tel]):
            ax = axes[k // 2][2 * half + k % 2]
            for c in p["curves"]:
                x = np.repeat(c["edges"], 2)[1:-1]
                ax.plot(np.concatenate(([x[0]], x, [x[-1]])),
                        np.concatenate(([0.0], np.repeat(c["y"], 2), [0.0])),
                        color=c["color"], linewidth=1.25, zorder=3)
            ax.set_xlim(*ranges[tel])
            ax.set_ylim(0.0, top)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            style.style_axes(ax, scale)
            if k // 2 == 1:
                ax.set_xlabel(xlabel)
            if k % 2 == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.tick_params(axis="y", labelleft=False)
            h = [Line2D([], [], color=c["color"], linewidth=2.0, label=c["label"])
                 for c in p["curves"]]
            lg = ax.legend(handles=h, loc="upper left", fontsize=s["legend"] * legend_scale,
                           title=p.get("note"), title_fontsize=s["legend"] * legend_scale,
                           frameon=True, framealpha=0.88, borderpad=0.28, labelspacing=0.26,
                           handlelength=1.5, handletextpad=0.5, borderaxespad=0.35)
            lg._legend_box.align = "left"
            if k >= 2:
                style.right_lines(ax, [p["subject"]], s["title"])
    style.compound_header(axes[0], tag=tag, data=data, scale=scale,
                          subjects=[panels[t][k]["subject"] for t in panels for k in (0, 1)])
    style.footer(fig, footer, scale)
    return fig


def run_columns(columns, rows, *, tag, data, footer, scale=0.6):
    """One column per run, one row per quantity: a filled histogram per panel.

    columns  [dict(subject, color, hists={row key: (edges, counts)}, notes={row key: text}), ...]
    rows     [dict(key, xlabel, ylabel, range, zero_line), ...]: the quantities, top to bottom,
             one x range per row
    Returns the figure."""
    ncol, nrow = len(columns), len(rows)
    fig, axes = blnw._grid(nrow, ncol, aw=4.6, ah=2.9, ml=1.25, mr=0.3, mt=0.62, mb=1.0,
                           gx=0.85, gy=0.95, scale=scale)
    s = style.sizes(scale)
    for j, col in enumerate(columns):
        for i, rw in enumerate(rows):
            ax = axes[i][j]
            edges, counts = col["hists"][rw["key"]]
            x = np.repeat(edges, 2)[1:-1]
            y = np.repeat(counts, 2).astype(float)
            ax.fill_between(x, y, 0, color=col["color"], alpha=0.55, linewidth=0, zorder=2)
            ax.plot(x, y, color=col["color"], linewidth=1.1, zorder=3)
            if rw.get("zero_line"):
                ax.axvline(0.0, color=style.INK_MUTED, linewidth=0.8, linestyle="--", zorder=1)
            ax.set_xlim(*rw["range"])
            ax.set_ylim(0, (counts.max() or 1) * 1.30)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.set_xlabel(rw["xlabel"])
            ax.set_ylabel(rw["ylabel"] if j == 0 else "")
            style.style_axes(ax, scale)
            ax.text(0.97, 0.95, col["notes"][rw["key"]], transform=ax.transAxes, ha="right",
                    va="top", fontsize=s["ann"], color=style.INK_MUTED, zorder=5)
            if i > 0:
                style.right_lines(ax, [col["subject"]], s["title"])
    style.compound_header(axes[0], tag=tag, data=data, scale=scale,
                          subjects=[c["subject"] for c in columns])
    style.footer(fig, footer, scale)
    return fig


def draw_rfsel_pair(ax, panel, *, xrange, top, xlabel, ylabel, scale=0.72, prune=None):
    """One panel of rfsel_curves on `ax`: one step curve per run, closed to 0 at both ends, and
    a short tick at each curve's median on the x axis (7 % of `top`, the y maximum); the legend
    (one line per curve, headed by the panel's note) in the upper left. `prune` goes to the x
    tick locator ("lower" drops the first tick, clear of the y tick labels)."""
    s = style.sizes(scale)
    for c in panel["curves"]:
        x = np.repeat(c["edges"], 2)[1:-1]
        ax.plot(np.concatenate(([x[0]], x, [x[-1]])),
                np.concatenate(([0.0], np.repeat(c["y"], 2), [0.0])),
                color=c["color"], linewidth=1.5, linestyle=c.get("linestyle", "-"), zorder=3)
        ax.plot([c["median"], c["median"]], [0.0, 0.07 * top], color=c["color"],
                linewidth=2.6, solid_capstyle="butt", zorder=4)
    ax.set_xlim(*xrange)
    ax.set_ylim(0.0, top)
    ax.xaxis.set_major_locator(MaxNLocator(6, prune=prune))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    style.style_axes(ax, scale)
    h = [Line2D([], [], color=c["color"], linewidth=2.2, linestyle=c.get("linestyle", "-"),
                label=c["label"]) for c in panel["curves"]]
    lg = ax.legend(handles=h, loc="upper left", fontsize=s["legend"] * 0.9,
                   title=panel.get("note"), title_fontsize=s["legend"] * 0.9, frameon=True,
                   framealpha=0.9, borderpad=0.35, labelspacing=0.45, handlelength=1.6,
                   handletextpad=0.5, borderaxespad=0.4)
    lg._legend_box.align = "left"
    return lg


def rfsel_pairs(panels, *, xlabel, ylabel, xrange, tag, data, footer, scale=0.72):
    """Pairs of runs that differ in RFSel only, side by side: one panel per pair
    (draw_rfsel_pair). One x range and one y scale for every panel, so the panels compare.

    panels  [dict(subject, note, curves=[dict(label, color, edges, y, median)]), ...]
            (rfsel_curves): y the value per bin, an optional linestyle (default "-"); the subject
            is the panel's right title, the note heads its legend
    Returns the figure."""
    # 8.2 in panels: the subject (settings_text of two RFSels) beside the experiment text
    fig, axes = blnw._grid(1, len(panels), aw=8.2, ah=4.4, ml=1.2, mr=0.3, mt=0.62,
                           mb=1.0 + style.footer_extra_in(footer, scale, fits=4),
                           gx=0.75, gy=0.0, scale=scale)
    top = 1.55 * max(float(np.max(c["y"])) for p in panels for c in p["curves"])
    for j, p in enumerate(panels):
        draw_rfsel_pair(axes[0][j], p, xrange=xrange, top=top, xlabel=xlabel,
                        ylabel=ylabel if j == 0 else "", scale=scale)
    style.compound_header(axes[0], tag=tag, data=data, scale=scale,
                          subjects=[p["subject"] for p in panels])
    style.footer(fig, footer, scale)
    return fig
