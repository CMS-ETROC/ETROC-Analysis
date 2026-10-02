"""plot_qinj.py -- the TDC of the injected pixels, in time units, on every
wafer of the lots given.

Per pixel it shows the CAL, TOA and TOT of every die, and where the TOA of
each pixel sits against the H-tree delays of the ETROC2 design.

    python plot_qinj.py --path <path> --lot "N62M23 bare=N62M23:08A5,07B2,06B7,05C4" --lot N62H30

with --path and --lot as for plot_power.py. Of each stage of each wafer it
reads the qinj.csv and dies.csv that plot_wafer.py wrote into its plots/
folder, and keeps the injected pixels on its PASSED valid dies: the
pixels hit in at least half of the events on at least half of the stage's
dies (wafer_tables.injected_pixels). A row of
qinj.csv holds a die's mean codes over the hits of one pixel; they convert
with that pixel's bin, 3.125 ns / CAL, as the test-beam pipeline converts
each hit (TestBeam/condor_at_lxplus/core/apply_tdc_cuts.py):

    TOA = 12.5 ns - TOA code x bin,  TOT = (2 TOT code - floor(TOT code / 32)) x bin.

TOA is linear in the code, so the converted mean is the mean of the
converted hits; TOT takes the floor of the mean code, which differs from
the mean over the hits by less than one bin where a pixel's hits straddle
a multiple of 32.

The TDC latches the TOA code at the first rising edge of the reference
strobe after the hit, so a pixel the strobe reaches later reads an earlier
TOA. The strobe and the charge-injection pulse reach the pixels through
H-trees whose delays the ETROC2 Reference Manual (rev 0.6, Fig. 28) gives
for each pixel; htree_toa_ns is what they add to a pixel's TOA. dtoa_ns is
a pixel's TOA minus that of REFERENCE_PIXEL on the same die, so whatever
the die shares cancels, and htree_dtoa_ns is the H-tree prediction for it.
The prediction covers the H-trees only: whatever else differs between
pixels, such as the supply their TDC runs on, stays in dtoa_ns. A stage
without the tables, without charge injection, without a PASSED valid die
or without REFERENCE_PIXEL among its injected pixels is named and left
out.

It writes into --out (default <path>/qinj_summary/): qinj_pixels_ns.csv,
one row per die and injected pixel of every wafer and stage read, and the
figure of each lot and stage (qinj_plots.py), <label>_<stage>_qinj.png:
per pixel, boxes of CAL, TOT, TOA and dtoa over the dies, with the H-tree
prediction beside dtoa. Like plot_wafer.py it only reads results.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from plot_lots import parse_lot, read_lots, slug
from stage_compare import STAGE_ORDER
from wafer_tables import injected_pixels

REFERENCE_PIXEL = (2, 2)
CLOCK_NS = 3.125  # the period CAL counts the delay cells over
TOA_WINDOW_NS = 12.5


def build_arg_parser():
    parser = argparse.ArgumentParser(description=" ".join(__doc__.split("\n\n")[0].split()))
    parser.add_argument('--path', required=True,
                        help='The --path of the wafer runs: the mother directory of all results')
    parser.add_argument('--lot', action='append', required=True, dest='lots', metavar='LABEL=BATCH:WAFER,...',
                        help='The wafers of batch BATCH (all, or the ones listed after the colon), labelled '
                             'LABEL (default: the batch name); repeat for each lot')
    parser.add_argument('--out', default=None, help='Output folder (default: <path>/qinj_summary)')
    parser.add_argument('--tables-only', action='store_true', dest='tables_only',
                        help='Write qinj_pixels_ns.csv only, no figures')
    return parser


def strobe_delay_ns(row, col):
    """The H-tree delay of the TDC reference strobe at a pixel, ETROC2
    Reference Manual rev 0.6, Fig. 28(d)."""
    if row % 2:
        return 4.89 if col % 2 else 4.82
    if col % 2:
        return 4.77 if col % 4 == 3 else 4.78
    return 4.71


def qinj_delay_ns(row, col):
    """The H-tree delay of the charge-injection pulse at a pixel, Fig. 28(b)."""
    return 4.79 if row % 4 == 0 and col % 4 == 0 else 4.80


def htree_toa_ns(row, col):
    """What the H-trees add to a pixel's TOA: a later injection pulse makes
    it later, a later strobe earlier."""
    return qinj_delay_ns(row, col) - strobe_delay_ns(row, col)


def to_ns(qinj):
    """bin_ps, toa_ns and tot_ns of each row of qinj.csv, from its mean
    codes and the bin of its own CAL."""
    cal = pd.to_numeric(qinj["cal_mean"], errors="coerce")
    tdc_bin = CLOCK_NS / cal
    tot = pd.to_numeric(qinj["tot_mean"], errors="coerce")
    return pd.DataFrame({"bin_ps": tdc_bin * 1e3,
                         "toa_ns": TOA_WINDOW_NS - pd.to_numeric(qinj["toa_mean"], errors="coerce") * tdc_bin,
                         "tot_ns": (2 * tot - np.floor(tot / 32)) * tdc_bin}, index=qinj.index)


def read_wafer(wafer_dir):
    """(the rows of each stage of the wafer with charge injection,
    [(stage, reason left out)])."""
    tables, left_out = [], []
    for stage in STAGE_ORDER:
        if not (wafer_dir / stage).is_dir():
            continue
        paths = [wafer_dir / stage / "plots" / name for name in ("qinj.csv", "dies.csv")]
        absent = [p for p in paths if not p.is_file()]
        if absent:
            left_out.append((stage, f"no {absent[0].relative_to(wafer_dir)}: run plot_wafer.py on it first"))
            continue
        qinj, dies = (pd.read_csv(p) for p in paths)
        if qinj.empty:
            left_out.append((stage, "no charge injection in its runs"))
            continue
        invalid = dies["invalid"].fillna(False).astype(bool)
        good = dies.loc[(dies["grade"] == "PASSED") & ~invalid, ["die", "start"]]
        injected = injected_pixels(qinj)
        rows = qinj[qinj["die"].isin(good["die"])
                    & pd.Series(list(zip(qinj["pix_row"], qinj["pix_col"])), index=qinj.index).isin(injected)]
        rows = rows[["die", "pix_row", "pix_col", "cal_mean", "toa_mean", "tot_mean"]].merge(good, on="die")
        if rows.empty:
            left_out.append((stage, "no PASSED valid die with charge injection"))
            continue
        if REFERENCE_PIXEL not in injected:
            left_out.append((stage, f"the reference pixel {REFERENCE_PIXEL} is not among its injected pixels"))
            continue
        rows = pd.concat([rows, to_ns(rows)], axis=1)
        reference = rows[(rows["pix_row"] == REFERENCE_PIXEL[0]) & (rows["pix_col"] == REFERENCE_PIXEL[1])]
        rows["dtoa_ns"] = rows["toa_ns"] - rows["die"].map(reference.set_index("die")["toa_ns"])
        rows["htree_dtoa_ns"] = [htree_toa_ns(r, c) - htree_toa_ns(*REFERENCE_PIXEL)
                                 for r, c in zip(rows["pix_row"], rows["pix_col"])]
        rows.insert(0, "stage", stage)
        tables.append(rows)
    return tables, left_out


def report(qinj):
    """One line per wafer and stage: its PASSED valid dies, the median CAL
    of each column half and the median dtoa of each pixel, the H-tree
    prediction in brackets."""
    for (lot, wafer, stage), rows in qinj.groupby(["lot", "wafer", "stage"], sort=False):
        halves = rows.groupby(rows["pix_col"] >= 8)["cal_mean"].median()
        offsets = [f"({r},{c}) {g['dtoa_ns'].median():+.3f} [{g['htree_dtoa_ns'].iloc[0]:+.2f}]"
                   for (r, c), g in rows.groupby(["pix_row", "pix_col"]) if (r, c) != REFERENCE_PIXEL]
        print(f"  {lot} {wafer} {stage}: {rows['die'].nunique()} PASSED valid dies, median CAL columns 0-7 "
              f"{halves.get(False, np.nan):.1f}, 8-15 {halves.get(True, np.nan):.1f}; median TOA minus "
              f"{REFERENCE_PIXEL} [H-tree prediction] (ns) {', '.join(offsets)}")


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    out_dir = Path(args.out) if args.out else Path(args.path) / "qinj_summary"
    specs = [parse_lot(s) for s in args.lots]
    labels = [label for label, *_ in specs]
    if len({slug(label) for label in labels}) != len(labels):
        print(f"two lots share a label, or its file-name form: {', '.join(labels)}; give each --lot its own LABEL=")
        return 2
    qinj = read_lots(args.path, specs, read_wafer)
    if qinj is None:
        return 2
    report(qinj)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [out_dir / "qinj_pixels_ns.csv"]
    qinj.to_csv(written[0], index=False)
    if not args.tables_only:
        from qinj_plots import plot_qinj  # matplotlib is needed from here on only
        note = (f"lots {', '.join(labels)}: qinj.csv and dies.csv of each wafer and stage (plot_wafer.py, newest run "
                f"of each die); plot_qinj.py {datetime.now():%Y-%m-%d %H:%M}")
        written += plot_qinj(out_dir, qinj, note=note)
    print(f"{len(written)} files written to {out_dir}:")
    for path in written:
        print(f"  {path.name}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
