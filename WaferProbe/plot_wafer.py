"""plot_wafer.py -- tables and figures for one tested wafer.

    python plot_wafer.py --path <path> --batchName <batch> --waferName <wafer>
    python plot_wafer.py --path <path> --batchID <id> --waferID <id>

with the --path of the wafer run. It reads the die folders of the wafer
folder the station wrote, <path>/BatchID_<id>_Name_<batch>/WaferID_<id>_Name_<wafer>/
(X for an ID the wafer does not have), found by the name, the ID or both at
each level (a folder with X only by its name), and writes into its plots/
the tables dies.csv, pixels.csv and qinj.csv (wafer_tables.py) and the
figures (wafer_plots.py), named by the two folder names, the wafer's labels.
It only reads
results and never talks to the station, the supplies or the chip, so it can
run while a wafer is being tested, or on any computer with a copy of the
results folder.
"""
import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

from station import load_wafer_map
from wafer_tables import collect, grade_counts, write_tables

REPO = Path(__file__).resolve().parent


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--path', required=True,
                        help='The --path of the wafer run: the mother directory of all results')
    parser.add_argument('--batchName', default=None, help='Batch (lot) name of the wafer run, e.g. N62M23')
    parser.add_argument('--batchID', type=int, default=None,
                        help='Batch ID of the wafer run, e.g. 0 (with --batchName, both must match)')
    parser.add_argument('--waferName', default=None, help='Wafer name of the wafer run, e.g. 08A5')
    parser.add_argument('--waferID', type=int, default=None,
                        help='Wafer ID of the wafer run, e.g. 3 (with --waferName, both must match)')
    parser.add_argument('--waferMap', default=str(REPO / 'wafer_map.csv'), dest='wafer_map',
                        help='CSV mapping the die number (location_id) to its row and col on the wafer map')
    parser.add_argument('--before', default=None,
                        help='Use the newest run of each die that started before this time, e.g. '
                             '"2026-09-22 15:30" on the DAQ computer clock (default: the newest run)')
    parser.add_argument('--out', default=None, help='Output folder (default: <wafer folder>/plots)')
    parser.add_argument('--tables-only', action='store_true', dest='tables_only',
                        help='Write the CSV tables only, no figures')
    parser.add_argument('--no-qinj', action='store_true', dest='no_qinj',
                        help='Do not read the QInj .nem files (faster; no QInj tables or figures)')
    return parser


def find_wafer_dirs(path, batch_name=None, batch_id=None, wafer_name=None, wafer_id=None):
    """The wafer folders under `path`, BatchID_<id>_Name_<batch>/WaferID_<id>_Name_<wafer>
    with a number or X for each ID, that match what is given of each level:
    the name, the ID, or both, which must then both match. An ID never
    matches X; a level given neither matches every folder."""
    def matching(parent, kind, name, number):
        label = re.compile(rf"{kind}ID_(\d+|X)_Name_(.+)")
        found = []
        for p in sorted(parent.iterdir()):
            m = label.fullmatch(p.name)
            if (m and p.is_dir() and (name is None or m.group(2) == name)
                    and (number is None or m.group(1) == str(number))):
                found.append(p)
        return found
    root = Path(path)
    if not root.is_dir():
        return []
    return [w for b in matching(root, "Batch", batch_name, batch_id)
            for w in matching(b, "Wafer", wafer_name, wafer_id)]


def _wanted(kind, name, number):
    """The folder name asked for, * for what was not given."""
    return f"{kind}ID_{'*' if number is None else number}_Name_{'*' if name is None else name}"


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    for level, name, number in (("batch", args.batchName, args.batchID),
                                ("wafer", args.waferName, args.waferID)):
        if name is None and number is None:
            parser.error(f"give --{level}Name, --{level}ID or both")
    found = find_wafer_dirs(args.path, args.batchName, args.batchID, args.waferName, args.waferID)
    if len(found) != 1:
        wanted = (f"{_wanted('Batch', args.batchName, args.batchID)}/"
                  f"{_wanted('Wafer', args.waferName, args.waferID)}")
        print(f"{len(found)} results folders match {wanted} under {args.path}, expected 1")
        if not found:
            found = find_wafer_dirs(args.path)
            print("the wafer folders there:" if found else "no wafer folder there at all")
        for folder in found:
            print(f"  {folder}")
        return 2
    wafer_dir = found[0]
    label = f"{wafer_dir.parent.name} / {wafer_dir.name}"
    print(f"reading {wafer_dir}")
    try:
        before = datetime.fromisoformat(args.before) if args.before else None
    except ValueError:
        print(f'--before {args.before!r}: expected a time such as "2026-09-22 15:30"')
        return 2
    out_dir = Path(args.out) if args.out else wafer_dir / "plots"

    dies, pixels, qinj, warnings = collect(wafer_dir, load_wafer_map(args.wafer_map), before=before,
                                           with_qinj=not args.no_qinj)
    for warning in warnings:
        print(f"warning: {warning}")
    counts, passed, tested = grade_counts(dies)
    share = f" ({100 * passed / tested:.1f} %)" if tested else ""
    print(f"{label}: {tested} of {len(dies)} dies tested, PASSED {passed}{share}")
    for name, n in counts:
        print(f"  {name:16s} {n}")

    out_dir.mkdir(parents=True, exist_ok=True)
    written = write_tables(out_dir, dies, pixels, qinj)
    if args.tables_only:
        stale = sorted(p.name for p in out_dir.glob("*.png"))
        if stale:
            print(f"warning: --tables-only: the {len(stale)} figures already in {out_dir} "
                  "were not redrawn and may not match these tables")
    else:
        from wafer_plots import plot_all  # matplotlib is needed from here on only
        selection = f"newest run before {args.before}" if before else "newest run"
        note = f"{label}: {selection} of each die; plot_wafer.py {datetime.now():%Y-%m-%d %H:%M}"
        written += plot_all(out_dir, dies, pixels, qinj, title=label, note=note)
    print(f"{len(written)} files written to {out_dir}:")
    for path in written:
        print(f"  {path.name}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
