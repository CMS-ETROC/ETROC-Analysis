"""plot_lots.py -- what UBM and bumping did to the wafers of each lot.

    python plot_lots.py --path <path> --lot N62C72 --lot N62H30 --lot N62M23
    python plot_lots.py --path <path> --lot "N62C72 earlier UBM=N62C72:02G4,03F5"

with the --path of the wafer runs, as for plot_wafer.py. Each --lot names a
lot as LABEL=BATCH:WAFER,WAFER,...: the wafers of the batch (lot number)
BATCH, all of them or the ones listed, under the label LABEL (the batch
name when no label is given). Wafers of one batch that went through UBM
at different times are separate lots: give each set its own --lot and
label. Only the wafers with both stage folders, pre_ubm and post_ubm, are
compared; the others are named and left out. The label names the files;
the figures name each lot by its batch and the months its stages were
tested (lot_compare.lot_names), and lots.png puts the lots in the order
they were tested.

It reads both stages of every wafer as plot_wafer.py does (the newest run
of each die), grades each die in both stages with the test both of its
runs made (lot_compare.like_with_like: pixels, QInj, eFuse and high-power
check), and writes into --out (default <path>/lot_summary/): lot_dies.csv,
one row per die of every wafer of the lots given, the figures of each lot,
<label>_yield.png, <label>_transitions.png, <label>_edges.png,
<label>_currents.png and <label>_shorts.png (lot_compare_plots.py), and,
with two lots or more, lots.png comparing them. Like plot_wafer.py it only
reads results.
"""
import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from lot_compare import lot_names, wafer_rows
from plot_wafer import find_wafer_dirs
from stage_compare import STAGE_ORDER
from station import load_wafer_map
from wafer_tables import invalid_dies

REPO = Path(__file__).resolve().parent


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--path', required=True,
                        help='The --path of the wafer runs: the mother directory of all results')
    parser.add_argument('--lot', action='append', required=True, dest='lots', metavar='LABEL=BATCH:WAFER,...',
                        help='A lot: the wafers of batch BATCH (all, or the ones listed after the colon), '
                             'labelled LABEL (default: the batch name); repeat for each lot')
    parser.add_argument('--waferMap', default=str(REPO / 'wafer_map.csv'), dest='wafer_map',
                        help='CSV mapping the die number (location_id) to its row and col on the wafer map, '
                             'and, in an optional invalid column, 1 for a die never to be used')
    parser.add_argument('--out', default=None, help='Output folder (default: <path>/lot_summary)')
    parser.add_argument('--tables-only', action='store_true', dest='tables_only',
                        help='Write lot_dies.csv only, no figures')
    return parser


def parse_lot(spec):
    """(label, batch, [wafer names] or None for all) from
    "LABEL=BATCH:WAFER,WAFER", "BATCH:WAFER,WAFER" or "BATCH"."""
    label, _, rest = spec.rpartition("=")
    batch, _, wafers = rest.partition(":")
    return (label or batch).strip(), batch.strip(), [w.strip() for w in wafers.split(",") if w.strip()] or None


def slug(label):
    """The label as a file-name prefix."""
    return re.sub(r"[^A-Za-z0-9.-]+", "_", label).strip("_")


def lot_wafers(path, batch, wafers):
    """(wafer folders with both stages, [(wafer folder name, reason left out)])
    for a lot; raises ValueError for a wafer asked for that is not there."""
    found = find_wafer_dirs(path, batch_name=batch)
    if wafers is not None:
        names = {re.fullmatch(r"WaferID_(?:\d+|X)_Name_(.+)", w.name).group(1): w for w in found}
        missing = [w for w in wafers if w not in names]
        if missing:
            raise ValueError(f"no wafer folder {', '.join(missing)} of batch {batch} under {path}")
        found = [names[w] for w in wafers]
    usable, left_out = [], []
    for w in found:
        absent = [s for s in STAGE_ORDER if not any((w / s).glob("die*"))]
        if absent:
            left_out.append((w.name, f"no {' or '.join(absent)} runs"))
        else:
            usable.append(w)
    return usable, left_out


def wafer_label(wafer_dir):
    return re.fullmatch(r"WaferID_(?:\d+|X)_Name_(.+)", wafer_dir.name).group(1)


def read_lot(label, wafer_dirs, wafer_map, invalid):
    """(lot rows, {wafer: setpoints}, [(wafer, stage, die, measured, regraded, test)])
    for the wafer folders of one lot, printing their warnings."""
    tables, setpoints, regraded = [], {}, []
    for w in wafer_dirs:
        name = wafer_label(w)
        changes, volts, changed, warnings = wafer_rows(w, wafer_map, invalid)
        for warning in warnings:
            print(f"warning: {name}: {warning}")
        changes.insert(0, "wafer", name)
        changes.insert(0, "batch", w.parent.name)
        changes.insert(0, "lot", label)
        tables.append(changes)
        setpoints[name] = volts
        regraded += [(name, *c) for c in changed]
    return pd.concat(tables, ignore_index=True), setpoints, regraded


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    wafer_map = load_wafer_map(args.wafer_map)
    invalid = invalid_dies(args.wafer_map)
    out_dir = Path(args.out) if args.out else Path(args.path) / "lot_summary"
    specs = [parse_lot(s) for s in args.lots]
    labels = [label for label, *_ in specs]
    if len({slug(label) for label in labels}) != len(labels):
        print(f"two lots share a label, or its file-name form: {', '.join(labels)}; give each --lot its own LABEL=")
        return 2
    lots, notes, taken = {}, {}, {}
    for label, batch, wafers in specs:
        try:
            usable, left_out = lot_wafers(args.path, batch, wafers)
        except ValueError as err:
            print(err)
            return 2
        for name, reason in left_out:
            print(f"{label}: {name} left out: {reason}")
        for w in usable:
            if w in taken:
                print(f"{label}: {w.name} is in {taken[w]} already; a wafer can be in one lot once only")
                return 2
            taken[w] = label
        if not usable:
            print(f"{label}: no wafer of batch {batch} has both stages under {args.path}")
            return 2
        print(f"{label}: reading {len(usable)} wafers of {usable[0].parent.name}: "
              + ", ".join(wafer_label(w) for w in usable))
        lot, setpoints, regraded = read_lot(label, usable, wafer_map, invalid)
        for wafer, stage, die, measured, regrade, test in regraded:
            print(f"  {wafer} {stage} die {die}: {measured} as measured, {regrade} with the test both stages "
                  f"made ({test})")
        lots[label] = (lot, setpoints, regraded)
        common = "; ".join(f"{w} {s.split('_')[0]} die {d} {m} → {r} ({t})" for w, s, d, m, r, t in regraded)
        notes[label] = (f"{label}: {usable[0].parent.name}, wafers {', '.join(wafer_label(w) for w in usable)}; "
                        "newest run of each die in each stage"
                        + (f"; regraded with the test both stages made: {common}" if common else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [out_dir / "lot_dies.csv"]
    pd.concat([lot for lot, *_ in lots.values()], ignore_index=True).to_csv(written[0], index=False)
    if not args.tables_only:
        from lot_compare_plots import plot_lot, plot_lots  # matplotlib is needed from here on only
        stamp = f"plot_lots.py {datetime.now():%Y-%m-%d %H:%M}"
        titles = lot_names({label: lot for label, (lot, *_) in lots.items()})
        for label, (lot, setpoints, _) in lots.items():
            written += plot_lot(out_dir, slug(label), lot, setpoints, title=titles[label],
                                note=f"{notes[label]}; {stamp}")
        if len(lots) > 1:
            written.append(plot_lots(out_dir, {label: lot for label, (lot, *_) in lots.items()},
                                     note=f"{' | '.join(notes.values())}; {stamp}"))
    print(f"{len(written)} files written to {out_dir}:")
    for path in written:
        print(f"  {path.name}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
