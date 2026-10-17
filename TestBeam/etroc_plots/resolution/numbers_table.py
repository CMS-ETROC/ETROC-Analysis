"""The numbers table (re09): one row per point of the fluence ladder (resolution.ladder,
step_winners), that is per (telescope, campaign, fluence step, board), written as .csv and .md
beside a table figure (PNG and PDF through style.save_figure).

Columns, named in words with the unit in brackets (COLUMNS):

  telescope, campaign, bookkeeping fluence [p/cm2], board, chip, run ("3+4" for a merge),
  merged, bias [V], threshold offset, RFSel, pixels (the points' event floor, anointed
  combination)
  top value [ps], top pixel spread [ps]
      the value drawn on the ladder: most-populated track per pixel; at a step of BAND_STEPS
      the band centre, with the anointed combination's spread
  central value [ps], central pixel spread [ps]
      the same board and run with the central track per pixel (central_values): the anointed
      value, or the band centre at a step of BAND_STEPS; empty where the table has no row
  band low [ps], band high [ps], anointed value [ps], central anointed value [ps]
      at a step of BAND_STEPS: the band's ends and the anointed combination's own value, with
      the most-populated and with the central track per pixel; empty elsewhere
  preferred run
      July: "yes" when the run, or the merge holding it, is the step's preferred run; empty for
      March (March has no display-run list)
"""
import os

import pandas as pd

from ..campaigns import active as campaign, check_required
from .. import style
from . import curves, tables
from .settings import DIVIDER_GID, LABEL_GID

check_required("resolution")

COLUMNS = ["telescope", "campaign", "bookkeeping fluence [p/cm2]", "board", "chip", "run",
           "merged", "bias [V]", "threshold offset", "RFSel", "pixels", "top value [ps]",
           "top pixel spread [ps]", "central value [ps]", "central pixel spread [ps]",
           "band low [ps]", "band high [ps]", "anointed value [ps]",
           "central anointed value [ps]", "preferred run"]
PS_COLUMNS = [c for c in COLUMNS if c.endswith("[ps]")]


def _key(p):
    return (p["tel"], p["camp"], p["label"], p["board_idx"])


def central_values(bt, bm, pts, floor, variant="central"):
    """{(tel, camp, label, board_idx): dict(value_ps, spread_ps, anointed_ps) or None}: each
    point's run and board with track variant `variant` at event floor `floor`, from the point's
    table (the merged table for a merge): at a step of BAND_STEPS the band centre (tables.band),
    the anointed spread and the anointed combination's own value; elsewhere the anointed
    combination's value and spread (anointed_ps None). None where the table has no anointed
    row."""
    out = {}
    for p in pts:
        table = bm if p["merged"] else bt
        tc = campaign.TABLE_CAMPAIGN[p["camp"]]
        rows = tables.board_rows(table, p["tel"], p["run"], variant, floor, campaign=tc,
                                 board_idx=p["board_idx"])
        if len(rows) > 1:
            raise ValueError("%s %s run %s board %d: %d anointed %s rows"
                             % (p["tel"], p["camp"], p["label"], p["board_idx"], len(rows),
                                variant))
        if not len(rows):
            out[_key(p)] = None
        elif p["band"]:
            b = tables.band(table, p["tel"], p["run"], p["board_idx"], variant, floor,
                            campaign=tc)
            out[_key(p)] = dict(value_ps=b["marker_ps"], spread_ps=b["bar_spread_ps"],
                                anointed_ps=b["reference_ps"])
        else:
            r = rows.iloc[0]
            out[_key(p)] = dict(value_ps=float(r["value_ps"]),
                                spread_ps=float(r["pixel_spread_ps"]), anointed_ps=None)
    return out


def numbers_table(points, central):
    """The table of `points` (step_winners points: curves.points dicts with step and
    preferred), one row per point, columns COLUMNS, ordered by telescope (TELESCOPE_CHIPS),
    campaign (curves.CAMP_ORDER), step and board. `central` is central_values of the same
    points."""
    tels = list(campaign.TELESCOPE_CHIPS)
    rows = []
    for p in sorted(points, key=lambda p: (tels.index(p["tel"]),
                                           curves.CAMP_ORDER.index(p["camp"]), p["step"],
                                           p["board_idx"])):
        band, c = p["band"], central.get(_key(p))
        rows.append(dict(zip(COLUMNS, [
            p["tel"].upper(), p["camp"].capitalize(), p["step"], p["board_idx"], p["chip"],
            p["label"], "yes" if p["merged"] else "no", p["hv"], p["offset"], p["rfsel"],
            p["n_pixels"], p["y_ps"], p["yerr_ps"], c["value_ps"] if c else None,
            c["spread_ps"] if c else None,
            band["band_lo_ps"] if band else None, band["band_hi_ps"] if band else None,
            band["reference_ps"] if band else None, c["anointed_ps"] if c and band else None,
            "" if p["preferred"] is None else ("yes" if p["preferred"] else "no")])))
    return pd.DataFrame(rows, columns=COLUMNS)


def records(df):
    """The rows of `df` as a list of dicts of plain Python values (a missing value None), for a
    values file."""
    out = []
    for row in df.itertuples(index=False):
        rec = {}
        for c, v in zip(df.columns, row):
            v = v.item() if hasattr(v, "item") else v
            rec[c] = None if isinstance(v, float) and pd.isna(v) else v
        out.append(rec)
    return out


def _md_cell(col, v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if col in PS_COLUMNS:
        return "%.1f" % v
    if col == "bookkeeping fluence [p/cm2]":
        return "%g" % v
    if col == "bias [V]":
        return "%g" % v
    return str(v)


def write_table(df, out_dir, stem):
    """Write `df` (numbers_table) as <stem>.csv (every digit) and <stem>.md (ps to 0.1) in
    `out_dir`. Returns the two paths."""
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, stem + ".csv")
    md_path = os.path.join(out_dir, stem + ".md")
    df.to_csv(csv_path, index=False)
    lines = ["| " + " | ".join(df.columns) + " |",
             "|" + "|".join("---" for _ in df.columns) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_md_cell(c, r[c]) for c in df.columns) + " |")
    with open(md_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return [csv_path, md_path]


def cell_text(r):
    """The lines of one table-figure cell for a row of numbers_table."""
    run = u"run %s%s, %g V" % (r["run"], u" merged" if r["merged"] == "yes" else u"",
                              r["bias [V]"])
    lines = [u"top %.1f ± %.1f ps" % (r["top value [ps]"], r["top pixel spread [ps]"])]
    if pd.notna(r["central value [ps]"]):
        lines.append(u"central %.1f ± %.1f ps" % (r["central value [ps]"],
                                                     r["central pixel spread [ps]"]))
    else:
        lines.append(u"central: no row")
    lines += [run, u"%d pixels" % r["pixels"]]
    if pd.notna(r["band low [ps]"]):
        lines += [u"band %.1f to %.1f ps" % (r["band low [ps]"], r["band high [ps]"]),
                  u"anointed %.1f ps" % r["anointed value [ps]"]]
        if pd.notna(r["central anointed value [ps]"]):
            lines.append(u"central anointed %.1f ps" % r["central anointed value [ps]"])
    return lines


def table_figure(df, *, name, tag, data, footer, col_w=2.5, row_head_w=1.25,
                 scale=curves.SCALE):
    """The table figure of `df` (numbers_table): one row per (telescope, chip), one column per
    look (campaign, step), each cell the lines of cell_text; the header on the table's axes
    (style.header: `name`, `tag`, `data`), `footer` below. Every text is an in-axes label (the
    layout audit checks that none touch); the rules between rows are dividers. Returns (figure,
    dict(cells={"<tel> <chip> | <campaign> <step>": [lines]}, columns, rows))."""
    import matplotlib.pyplot as plt
    style.apply_style(scale)
    s = style.sizes(scale)
    tels = list(campaign.TELESCOPE_CHIPS)
    looks = sorted({(r["campaign"], r["bookkeeping fluence [p/cm2]"]) for _, r in df.iterrows()},
                   key=lambda k: (k[1], curves.CAMP_ORDER.index(k[0].lower())))
    rows = []
    for _, r in df.iterrows():
        k = (r["telescope"], r["chip"], r["board"])
        if k not in rows:
            rows.append(k)
    rows.sort(key=lambda k: (tels.index(k[0].lower()), k[2]))
    line_in = 1.25 * s["ann"] / 72.0
    n_lines = max(len(cell_text(r)) for _, r in df.iterrows())
    row_h = n_lines * line_in + 0.22
    head_h = 2 * 1.25 * s["legend"] / 72.0 + 0.22
    tw, th = row_head_w + len(looks) * col_w, head_h + len(rows) * row_h
    ml, mr, mt = 0.3, 0.3, 0.62
    mb = 0.3 + (footer.count("\n") + 1) * 1.2 * s["ann"] / 72.0
    W, H = ml + tw + mr, mb + th + mt
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([ml / W, mb / H, tw / W, th / H])
    ax.set_xlim(0.0, tw)
    ax.set_ylim(th, 0.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    def text(x, y, t, **kw):
        ax.text(x, y, t, ha="center", va="center", color=style.INK, gid=LABEL_GID,
                linespacing=1.25, **kw)

    def rule(y, lw):
        ax.plot([0.0, tw], [y, y], color=style.INK_MUTED, linewidth=lw, gid=DIVIDER_GID)

    for j, (camp, f) in enumerate(looks):
        text(row_head_w + (j + 0.5) * col_w, head_h / 2.0,
             u"%s\n%s" % (camp, style.fluence_label(f)), fontsize=s["legend"],
             fontweight="bold")
    rule(head_h, 1.4)
    cells = {}
    for i, (tel, chip, _) in enumerate(rows):
        y0 = head_h + i * row_h
        text(row_head_w / 2.0, y0 + row_h / 2.0, u"%s\n%s" % (tel, chip),
             fontsize=s["legend"], fontweight="bold")
        for j, (camp, f) in enumerate(looks):
            sel = df[(df["telescope"] == tel) & (df["chip"] == chip) & (df["campaign"] == camp)
                     & (df["bookkeeping fluence [p/cm2]"] == f)]
            if len(sel) > 1:
                raise ValueError("%s %s %s %g: %d rows" % (tel, chip, camp, f, len(sel)))
            lines = cell_text(sel.iloc[0]) if len(sel) else [u"no point"]
            text(row_head_w + (j + 0.5) * col_w, y0 + row_h / 2.0, u"\n".join(lines),
                 fontsize=s["ann"])
            cells[u"%s %s | %s %g" % (tel, chip, camp, f)] = lines
        last = i + 1 == len(rows) or rows[i + 1][0] != tel
        if i + 1 < len(rows):
            rule(y0 + row_h, 1.4 if last else 0.6)
    style.header(ax, name=name, tag=tag, data=data, scale=scale)
    style.footer(fig, footer, scale)
    return fig, dict(cells=cells, columns=[u"%s %g" % k for k in looks],
                     rows=[u"%s %s" % (t, c) for t, c, _ in rows])
