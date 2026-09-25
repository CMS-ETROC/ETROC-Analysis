"""The resolution result tables: readers and the selections the resolution figures share.

The test-beam chain (TestBeam/condor_at_lxplus) quotes a time resolution for every board of a
3-board combo, per pixel and per board; the quote is utils/quote_resolution.py there. A track is
one path of hit pixels through the combo's three boards, one pixel per board, as the chain's
step 13 writes it; its events are the particles recorded along it. The results are kept as flat
tables (the campaign module's BOARD_TABLE, PIXEL_TABLE, and BOARD_TABLE_MERGED,
PIXEL_TABLE_MERGED for merged runs):

board table   one row per campaign, telescope, run, combo, board (board_idx, board_chip), track
              variant and event floor: value_ps (the board value, a robust fit over the board's
              pixel map), pixel_spread_ps (pixel-to-pixel sigma), n_pixels, and `anointed`, True
              on the combo that stands for the board (boards 0-1-2 for boards 0, 1 and 2;
              boards 1-2-3 for board 3). Merged tables add degenerate_share: the share of the
              combo's tracks the quote drops as degenerate (a solved width below 0.35 or above
              0.95 of the smallest pair width it came from; condor_at_lxplus/README.md).
pixel table   one row per pixel (row, col) of each of those: value_ps, err_ps, nevt_sum (the
              events behind the pixel).

variant is how a pixel's tracks are combined: "top" (the most-populated track through the pixel),
"central" or "average". floor is the minimum number of events per track, read as text ("none",
"100", "300"). A merged run's `run` is the merge's name, merge_name(telescope, runs).

runs_summary.csv (the campaign's RUNS_SUMMARY_CSV) holds one row per run (campaign, telescope,
run) with, per board i, the chip, bias, bookkeeping fluence, threshold offset and RFSel in the
columns board<i>_chip, board<i>_hv, board<i>_irrad, board<i>_offset, board<i>_rfsel:
run_settings() reads it.

per_pixel_cost_summary.csv (MERGE_COST_CSV) holds one row per merge (group, the merge's name)
and chip: status ("ok" when both maps exist), median_cost_px_ps, p16_cost_px_ps,
p84_cost_px_ps (the cost per pixel over the pixels common to the merge and its runs),
board_cost_official_ps (the same difference on the board values) and n_pixels_common.
"""
import re

import pandas as pd
from matplotlib.patches import Rectangle

from ..campaigns import active as _campaign
from .. import style

VARIANT_TEXT = {"top": "most-populated track per pixel",
                "central": "central track per pixel",
                "average": "every track per pixel, inverse-variance mean"}
ANOINTED_TEXT = "anointed combo (0-1-2 for boards 0-2, 1-2-3 for board 3)"

BAND_FOOTER = ("shaded box = min to max of the board's value over every combination containing "
               "it, marker at its centre, bar = pixel-to-pixel spread of the anointed combination")

PIXEL_COLUMNS = ["campaign", "telescope", "run", "combo", "board_idx", "board_role", "board_chip",
                 "row", "col", "variant", "floor", "value_ps", "err_ps", "n_paths", "nevt_sum",
                 "anointed"]


def conventions(variant, floor):
    """The conventions block of a resolution figure's values file."""
    return {"selection": "%s, %s, one track per pixel per board, event floor %s"
                         % (VARIANT_TEXT[variant], ANOINTED_TEXT, floor),
            "board_value": "robust Gaussian mean over the pixel map "
                           "(quote_resolution.board_summary)",
            "error_bar": "pixel-to-pixel sigma; stat = sigma/sqrt(n_pixels)",
            "fluence": _campaign.FLUENCE_CONVENTION}


def merge_name(telescope, runs):
    """The `run` of a merge in the merged tables: ("h1", (3, 4)) -> "h1_grp_run3_4"."""
    return "%s_grp_run%s" % (telescope, "_".join(str(r) for r in runs))


def combo_chips(combo, telescope):
    """A combo named by board role and index ("extra0-dut1-ref2"; the role at an index changes
    from run to run) as its chips, by board index ("IH7-IH11-IH12")."""
    chips = _campaign.TELESCOPE_CHIPS[telescope]
    return "-".join(chips[int(re.search(r"\d+$", part).group())] for part in combo.split("-"))


# ---------------------------------------------------------------------------- readers
def read_board_table(path):
    return pd.read_csv(path, dtype={"floor": str})


def read_pixel_table(path, runs, variant, floor, campaign, anointed=True, chunksize=200_000):
    """The rows of a pixel table for `runs` ({telescope: run numbers or merge names}) at one
    variant, floor and campaign, anointed combos only unless `anointed` is None. Read in chunks:
    the single-run table holds every pixel of every run."""
    keep = []
    for chunk in pd.read_csv(path, usecols=PIXEL_COLUMNS, chunksize=chunksize,
                             dtype={"floor": str}):
        sub = chunk[(chunk["campaign"] == campaign) & (chunk["variant"] == variant)
                    & (chunk["floor"] == str(floor))]
        if anointed is not None:
            sub = sub[sub["anointed"] == anointed]
        mask = pd.Series(False, index=sub.index)
        for tel, rr in runs.items():
            mask |= (sub["telescope"] == tel) & sub["run"].astype(str).isin([str(r) for r in rr])
        if mask.any():
            keep.append(sub[mask])
    return (pd.concat(keep, ignore_index=True) if keep
            else pd.DataFrame(columns=PIXEL_COLUMNS))


def read_runs_summary(path):
    return pd.read_csv(path)


# ---------------------------------------------------------------------------- selections
def board_rows(bt, telescope, run, variant, floor, campaign=None, board_idx=None, anointed=True):
    """The board-table rows of one run (a run number or a merge name), sorted by board; anointed
    combos only unless `anointed` is None."""
    sel = ((bt["telescope"] == telescope) & (bt["run"].astype(str) == str(run))
           & (bt["variant"] == variant) & (bt["floor"] == str(floor)))
    if campaign is not None:
        sel &= bt["campaign"] == campaign
    if anointed is not None:
        sel &= bt["anointed"] == anointed
    if board_idx is not None:
        sel &= bt["board_idx"] == board_idx
    return bt[sel].sort_values("board_idx")


def band(bt, telescope, run, board_idx, variant, floor, campaign=None):
    """The combo band of one board in one run: band_lo_ps..band_hi_ps = min..max of the board
    value over every combo containing the board; marker_ps its centre; bar_spread_ps and n_pixels
    from the anointed combo, whose own value is kept as anointed_ps."""
    rows = board_rows(bt, telescope, run, variant, floor, campaign, board_idx, anointed=None)
    if len(rows) < 2:
        raise ValueError("%s run %s board %d is in %d combo(s): a band needs two or more"
                         % (telescope, run, board_idx, len(rows)))
    anointed = rows[rows["anointed"] == True]      # noqa: E712  (a pandas mask)
    if not len(anointed):
        raise ValueError("no anointed row for %s run %s board %d" % (telescope, run, board_idx))
    a = anointed.iloc[0]
    lo, hi = float(rows["value_ps"].min()), float(rows["value_ps"].max())
    return dict(marker_ps=(lo + hi) / 2.0, band_lo_ps=lo, band_hi_ps=hi,
                bar_spread_ps=float(a["pixel_spread_ps"]), n_pixels=int(a["n_pixels"]),
                anointed_ps=float(a["value_ps"]), n_combo_rows=len(rows))


def draw_band(ax, x, band, color, half_width=0.032, capsize=3.0, zorder=1, alpha=0.22,
              marker="_", markersize=10.0):
    """One band at x: a filled rectangle (low alpha) from band_lo_ps to band_hi_ps, behind a thin
    ink pixel-spread bar on the band centre. `marker` marks the centre: a dash, or the chip's
    marker where several chips' bands stand side by side."""
    rect = Rectangle((x - half_width, band["band_lo_ps"]), 2.0 * half_width,
                     band["band_hi_ps"] - band["band_lo_ps"], facecolor=color, edgecolor="none",
                     alpha=alpha, zorder=zorder)
    ax.add_patch(rect)
    ax.errorbar([x], [band["marker_ps"]], yerr=[band["bar_spread_ps"]], color=style.INK,
                ecolor=style.INK, marker=marker, markersize=markersize, markeredgewidth=1.6,
                linestyle="none", elinewidth=1.4, capsize=capsize, zorder=zorder + 2)
    return rect


# ---------------------------------------------------------------------------- run settings
def _fluence(text):
    """runs_summary's fluence text ("1.5e15", "pre-irrad") as p/cm2."""
    try:
        return float(text)
    except (TypeError, ValueError):
        if str(text).strip().lower().startswith("pre"):
            return 0.0
        raise ValueError("fluence %r in runs_summary is not a number" % (text,))


def run_settings(rs, campaign, telescope, run):
    """Per board of one run, from runs_summary: chip, bias (V), bookkeeping fluence, threshold
    offset and RFSel (bias and RFSel None where the table has none). `telescope` is the table's key ("H1_HPK")."""
    r = rs[(rs["campaign"] == campaign) & (rs["telescope"] == telescope) & (rs["run"] == int(run))]
    if len(r) != 1:
        raise ValueError("%d runs_summary rows for %s %s run %s" % (len(r), campaign, telescope, run))
    r = r.iloc[0]
    boards = []
    for i in range(sum(1 for c in rs.columns if re.fullmatch(r"board\d+_chip", c))):
        chip = r.get("board%d_chip" % i)
        if not isinstance(chip, str):
            continue
        rfsel, hv = r["board%d_rfsel" % i], r["board%d_hv" % i]
        boards.append(dict(board_idx=i, chip=chip, hv=None if pd.isna(hv) else float(hv),
                           fluence=_fluence(r["board%d_irrad" % i]),
                           offset=int(r["board%d_offset" % i]),
                           rfsel=None if pd.isna(rfsel) else int(rfsel)))
    return boards


def common_settings(rs, campaign, telescope, runs):
    """The one bookkeeping fluence, RFSel and threshold offset shared by every board of every run
    in `runs` (a merge, say): dict(fluence, rfsel, offset). Raises ValueError when they differ, or
    when a board's chip or bias differs between the runs."""
    seen = {k: set() for k in ("fluence", "rfsel", "offset")}
    per_board = {}
    for run in runs:
        for b in run_settings(rs, campaign, telescope, run):
            for k in seen:
                seen[k].add(b[k])
            per_board.setdefault(b["board_idx"], set()).add((b["chip"], b["hv"]))
    split = {k: sorted(v, key=str) for k, v in seen.items() if len(v) != 1}
    split.update({"board%d chip, bias" % i: sorted(v, key=str) for i, v in per_board.items()
                  if len(v) != 1})
    if split:
        raise ValueError("%s runs %s do not share one setting: %s" % (telescope, runs, split))
    return {k: v.pop() for k, v in seen.items()}
