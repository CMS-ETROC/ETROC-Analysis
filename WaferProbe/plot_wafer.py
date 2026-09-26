"""plot_wafer.py -- tables and figures for one tested wafer.

    python plot_wafer.py --path <path> --waferStage <stage> --batchName <batch> --waferName <wafer>
    python plot_wafer.py --path <path> --waferStage <stage> --batchID <id> --waferID <id>

with the --path of the wafer run. It reads the die folders of one stage of
the wafer folder the station wrote,
<path>/BatchID_<id>_Name_<batch>/WaferID_<id>_Name_<wafer>/<stage>/
(X for an ID the wafer does not have; the stage pre_ubm or post_ubm), found
by the name, the ID or both at each level (a folder with X only by its
name), and writes into the stage folder's plots/
the tables dies.csv, pixels.csv and qinj.csv (wafer_tables.py) and the
figures (wafer_plots.py), named by the two folder names, the wafer's labels.
When the wafer folder holds the other stage as well, it reads that one too
(the newest run of each die) and writes what changed between them into
<wafer folder>/pre_vs_post/: changes.csv (stage_compare.py) and its
figures (stage_compare_plots.py); --no-compare skips it. It only reads
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
from stage_compare import MAIN_RAILS, STAGE_ORDER, compare, qinj_compared, transitions, unchecked_text
from wafer_tables import collect, grade_counts, invalid_dies, write_tables

REPO = Path(__file__).resolve().parent


STAGES = ("pre_ubm", "post_ubm")


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--path', required=True,
                        help='The --path of the wafer run: the mother directory of all results')
    parser.add_argument('--waferStage', required=True, choices=STAGES, dest='wafer_stage',
                        help='Stage the wafer was probed at: pre_ubm (bare, before UBM and bumps) or '
                             'post_ubm (after UBM and bumping), a folder of its own under the wafer folder')
    parser.add_argument('--batchName', default=None, help='Batch (lot) name of the wafer run, e.g. N62M23')
    parser.add_argument('--batchID', type=int, default=None,
                        help='Batch ID of the wafer run, e.g. 0 (with --batchName, both must match)')
    parser.add_argument('--waferName', default=None, help='Wafer name of the wafer run, e.g. 08A5')
    parser.add_argument('--waferID', type=int, default=None,
                        help='Wafer ID of the wafer run, e.g. 3 (with --waferName, both must match)')
    parser.add_argument('--waferMap', default=str(REPO / 'wafer_map.csv'), dest='wafer_map',
                        help='CSV mapping the die number (location_id) to its row and col on the wafer map, '
                             'and, in an optional invalid column, 1 for a die never to be used')
    parser.add_argument('--before', default=None,
                        help='Use the newest run of each die that started before this time, e.g. '
                             '"2026-09-22 15:30" on the DAQ computer clock (default: the newest run)')
    parser.add_argument('--out', default=None, help='Output folder (default: <wafer folder>/<stage>/plots)')
    parser.add_argument('--tables-only', action='store_true', dest='tables_only',
                        help='Write the CSV tables only, no figures')
    parser.add_argument('--no-qinj', action='store_true', dest='no_qinj',
                        help='Do not read the QInj .nem files (faster; no QInj tables or figures)')
    parser.add_argument('--no-compare', action='store_true', dest='no_compare',
                        help='Do not compare with the other stage of the wafer, even when its folder is there '
                             '(default: compare, into <wafer folder>/pre_vs_post, or <out>/pre_vs_post with --out)')
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


def ran_as_another_stage(dies, stage):
    """([(die, stage recorded)] of the dies whose run recorded a stage other
    than `stage`, the number of runs that recorded none)."""
    ran = dies[dies["run"].notna()] if "run" in dies else dies.iloc[0:0]
    stages = ran["wafer_stage"] if "wafer_stage" in ran else ran["die"].map(lambda d: None)
    other = ran[stages.notna() & (stages != stage)]
    return list(zip(other["die"], stages[other.index])), int(stages.isna().sum())


def warn_stale(out_dir):
    """Under --tables-only, warn about the figures already in out_dir."""
    stale = sorted(p.name for p in Path(out_dir).glob("*.png"))
    if stale:
        print(f"warning: --tables-only: the {len(stale)} figures already in {out_dir} "
              "were not redrawn and may not match these tables")


def compare_stages(args, wafer_dir, tables):
    """Compare the stage just read, `tables` = (dies, pixels, qinj), with
    the wafer's other stage when its folder is there, and write changes.csv
    and the figures into pre_vs_post/. Returns the paths written."""
    other = next(s for s in STAGES if s != args.wafer_stage)
    if not (wafer_dir / other).is_dir():
        return []
    print(f"reading {wafer_dir / other} to compare the two stages (the newest run of each die)")
    dies, pixels, qinj, warnings = collect(wafer_dir / other, load_wafer_map(args.wafer_map),
                                           with_qinj=not args.no_qinj, invalid=invalid_dies(args.wafer_map))
    for warning in warnings:
        print(f"warning: {other}: {warning}")
    if "run" not in dies or not dies["run"].notna().any():
        print(f"no die ran in {wafer_dir / other}: the stages are not compared")
        return []
    wrong, unrecorded = ran_as_another_stage(dies, other)
    if wrong:
        print(f"{len(wrong)} dies in {wafer_dir / other} ran as another stage, the stages are not compared: "
              + ", ".join(f"die {d} ({s})" for d, s in wrong))
        return []
    if unrecorded:
        print(f"warning: {other}: {unrecorded} dies ran without a recorded stage")
    changes, pixel_rows, qinj_rows, volts = compare({args.wafer_stage: tables, other: (dies, pixels, qinj)})
    out_dir = (Path(args.out) if args.out else wafer_dir) / "pre_vs_post"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [out_dir / "changes.csv"]
    changes.to_csv(written[0], index=False)
    moves = transitions(changes)
    print(f"pre_ubm vs post_ubm: {sum(len(d) for *_, d in moves)} valid dies changed grade")
    for before, after, moved in moves:
        print(f"  {before} -> {after} ({len(moved)}): {', '.join(map(str, moved))}")
    for stage, tag in zip(STAGE_ORDER, ("pre", "post")):
        if not any(changes[f"{rail}_I_high_{tag}"].notna().any() for rail in MAIN_RAILS):
            print(f"  {stage} logged no power phases: its currents are not compared")
        unchecked = unchecked_text(changes, tag)
        if unchecked:
            print(f"  {stage}: {unchecked}")
    if pixel_rows.empty:
        print("  no pixel was calibrated in both stages: no baseline comparison")
    if not qinj_compared(changes).any():
        print("  no die ran QInj in both stages: no QInj comparison")
    if args.tables_only:
        warn_stale(out_dir)
    else:
        from stage_compare_plots import plot_changes  # matplotlib is needed from here on only
        label = f"{wafer_dir.parent.name} / {wafer_dir.name} / pre_ubm vs post_ubm"
        selection = (f"{args.wafer_stage}: newest run before {args.before}, {other}: newest run"
                     if args.before else "newest run of each die in each stage")
        note = f"{label}: {selection}; plot_wafer.py {datetime.now():%Y-%m-%d %H:%M}"
        written += plot_changes(out_dir, changes, pixel_rows, qinj_rows, volts, title=label, note=note)
    return written


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
    stage_dir = wafer_dir / args.wafer_stage
    loose = [p for p in wafer_dir.glob("die*") if p.is_dir()]
    if loose:
        print(f"warning: {len(loose)} die folders directly in {wafer_dir}, outside a stage folder, "
              "are not read")
    if not stage_dir.is_dir():
        there = [s for s in STAGES if (wafer_dir / s).is_dir()]
        print(f"no {args.wafer_stage} folder in {wafer_dir}; the stages there: {', '.join(there) or 'none'}")
        return 2
    label = f"{wafer_dir.parent.name} / {wafer_dir.name} / {args.wafer_stage}"
    print(f"reading {stage_dir}")
    try:
        before = datetime.fromisoformat(args.before) if args.before else None
    except ValueError:
        print(f'--before {args.before!r}: expected a time such as "2026-09-22 15:30"')
        return 2
    out_dir = Path(args.out) if args.out else stage_dir / "plots"

    dies, pixels, qinj, warnings = collect(stage_dir, load_wafer_map(args.wafer_map), before=before,
                                           with_qinj=not args.no_qinj, invalid=invalid_dies(args.wafer_map))
    for warning in warnings:
        print(f"warning: {warning}")
    other, unrecorded = ran_as_another_stage(dies, args.wafer_stage)
    if other:
        print(f"{len(other)} dies in {stage_dir} ran as another stage, nothing written: "
              + ", ".join(f"die {d} ({s})" for d, s in other))
        return 2
    if unrecorded:
        print(f"warning: {unrecorded} dies ran without a recorded stage")
    counts, passed, tested = grade_counts(dies)
    share = f" ({100 * passed / tested:.1f} %)" if tested else ""
    invalid = dies.loc[dies["invalid"], "die"].tolist()
    if invalid:
        print(f"{label}: {tested} of {len(dies) - len(invalid)} valid dies tested, PASSED {passed}{share}; "
              f"invalid, not counted: {', '.join(map(str, invalid))}")
    else:
        print(f"{label}: {tested} of {len(dies)} dies tested, PASSED {passed}{share}")
    for name, n in counts:
        print(f"  {name:16s} {n}")

    out_dir.mkdir(parents=True, exist_ok=True)
    written = write_tables(out_dir, dies, pixels, qinj)
    if args.tables_only:
        warn_stale(out_dir)
    else:
        from wafer_plots import plot_all  # matplotlib is needed from here on only
        selection = f"newest run before {args.before}" if before else "newest run"
        note = f"{label}: {selection} of each die; plot_wafer.py {datetime.now():%Y-%m-%d %H:%M}"
        written += plot_all(out_dir, dies, pixels, qinj, title=label, note=note)
    print(f"{len(written)} files written to {out_dir}:")
    for path in written:
        print(f"  {path.name}")
    if not args.no_compare:
        compared = compare_stages(args, wafer_dir, (dies, pixels, qinj))
        if compared:
            print(f"{len(compared)} files written to {compared[0].parent}:")
            for path in compared:
                print(f"  {path.name}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
