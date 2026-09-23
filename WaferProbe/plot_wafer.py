"""plot_wafer.py -- tables and figures for one tested wafer.

    python plot_wafer.py --path <path> --batchName <batch> --waferName <wafer>

with the --path, --batchName and --waferName of the wafer run. It reads the
die folders under <path>/<batch>/<wafer>/ and writes into
<path>/<batch>/<wafer>/plots/ the tables dies.csv, pixels.csv and qinj.csv
(wafer_tables.py) and the figures (wafer_plots.py). It only reads
results and never talks to the station, the supplies or the chip, so it can
run while a wafer is being tested, or on any computer with a copy of the
results folder.
"""
import argparse
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
    parser.add_argument('--batchName', help='Batch name, e.g. N62C72 (else BatchID_<batchID>)')
    parser.add_argument('--waferName', help='Wafer name, e.g. 02G4 (else WaferID_<waferID>)')
    parser.add_argument('--batchID', type=int, default=-1, help='Batch ID, for a run given no --batchName')
    parser.add_argument('--waferID', type=int, default=-1, help='Wafer ID, for a run given no --waferName')
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


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    batch = args.batchName or (f"BatchID_{args.batchID}" if args.batchID >= 0 else None)
    wafer = args.waferName or (f"WaferID_{args.waferID}" if args.waferID >= 0 else None)
    if not batch or not wafer:
        print("give --batchName (or --batchID) and --waferName (or --waferID), as for the wafer run")
        return 2
    wafer_dir = Path(args.path) / batch / wafer
    if not wafer_dir.is_dir():
        print(f"no results folder {wafer_dir}")
        return 2
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
    print(f"{batch} / {wafer}: {tested} of {len(dies)} dies tested, PASSED {passed}{share}")
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
        note = f"{batch} / {wafer}: {selection} of each die; plot_wafer.py {datetime.now():%Y-%m-%d %H:%M}"
        written += plot_all(out_dir, dies, pixels, qinj, title=f"{batch} / {wafer}", note=note)
    print(f"{len(written)} files written to {out_dir}:")
    for path in written:
        print(f"  {path.name}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
