"""plot_positions.py -- do the dies at one place on the wafer behave alike on
every wafer of a lot?

    python plot_positions.py --path <path> --stage pre_ubm --lot N62M23
    python plot_positions.py --path <path> --stage pre_ubm --lot "N62C72=N62C72:02G4,03F7,04F2"

with the --path of the wafer runs, as for plot_wafer.py. Each --lot names
the wafers as plot_lots.py does, LABEL=BATCH:WAFER,WAFER,... (all wafers of
the batch when none are listed). Of each wafer it reads the --stage folder
(the newest run of each die, as plot_wafer.py), and compares the dies at
each place across the wafers that have full scans or QInj data there
(position_compare.py): the die-level baseline, noise width, CAL, TOA and TOT
by place and the pairwise r of every two wafers, and the pixel maps of
the dies at the same place against those at different places, with the
same die measured in the other stage as the reference where both stages
have its full scan, and the tilt across each die at each place. A wafer
without such data is named and left out; a lot needs two wafers.

Wafers tested at different times or setups can go in one lot: every
measure takes each wafer's own offset out. The figures name the months
each wafer was tested.

It writes into --out (default <path>/position_summary/), per lot,
<label>_<stage>_positions.csv (one row per die: its values and each minus
its wafer's median) and <label>_<stage>_position_maps.png,
_position_pairs.png and _pixel_patterns.png (position_plots.py). Like
plot_wafer.py it only reads results.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from plot_lots import parse_lot, slug, wafer_label
from plot_wafer import find_wafer_dirs
from position_compare import (DIE_QUANTITIES, MAP_QUANTITIES, centred, common_injected, die_values, full_scan,
                              map_pairs, mean_r, measured, months_tested, pairwise_r, place_tilts, residual_maps,
                              same_die_r, shuffled_r, tilts)
from stage_compare import STAGE_ORDER
from station import load_wafer_map
from wafer_tables import collect, invalid_dies

REPO = Path(__file__).resolve().parent
STAGE_TEXT = {"pre_ubm": "before UBM", "post_ubm": "after UBM"}


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--path', required=True,
                        help='The --path of the wafer runs: the mother directory of all results')
    parser.add_argument('--stage', required=True, choices=STAGE_ORDER, help='The stage folder of each wafer to read')
    parser.add_argument('--lot', action='append', required=True, dest='lots', metavar='LABEL=BATCH:WAFER,...',
                        help='The wafers of batch BATCH (all, or the ones listed after the colon), labelled '
                             'LABEL (default: the batch name); repeat for each lot')
    parser.add_argument('--waferMap', default=str(REPO / 'wafer_map.csv'), dest='wafer_map',
                        help='CSV mapping the die number (location_id) to its row and col on the wafer map, '
                             'and, in an optional invalid column, 1 for a die never to be used')
    parser.add_argument('--no-qinj', action='store_true', dest='no_qinj',
                        help='Leave the QInj data out (faster: the .nem files are not read)')
    parser.add_argument('--out', default=None, help='Output folder (default: <path>/position_summary)')
    parser.add_argument('--tables-only', action='store_true', dest='tables_only',
                        help='Write the csv files only, no figures')
    return parser


def read_stage(wafer_dirs, stage, wafer_map, invalid, with_qinj):
    """({wafer: (dies, pixels, qinj)} of the wafers with full scans or QInj
    data in `stage`, [(wafer, reason left out)]), printing the collect
    warnings."""
    tables, left_out = {}, []
    for w in wafer_dirs:
        name = wafer_label(w)
        if not any((w / stage).glob("die*")):
            left_out.append((name, f"no {stage} runs"))
            continue
        dies, pixels, qinj, warnings = collect(w / stage, wafer_map, with_qinj=with_qinj, invalid=invalid)
        for warning in warnings:
            print(f"warning: {name} {stage}: {warning}")
        if not full_scan(dies) and qinj.empty:
            left_out.append((name, f"no full scan or QInj data in {stage}"))
            continue
        tables[name] = (dies, pixels, qinj)
    return tables, left_out


def title_of(label, stage, tables):
    """"<label> <stage text>, <n> wafers: <wafers> (<months>); ...", the
    wafers grouped by the months they were tested."""
    months = {}
    for wafer, (dies, _, _) in tables.items():
        months.setdefault(months_tested(dies), []).append(wafer)
    groups = "; ".join(f"{' '.join(ws)} ({m})" for m, ws in months.items())
    return f"{label} {STAGE_TEXT[stage]}, {len(tables)} wafers: {groups}"


def compare_lot(label, wafer_dirs, stage, wafer_map, invalid, with_qinj):
    """Everything plot_positions writes for one lot, or None with a message
    when fewer than two wafers have data."""
    tables, left_out = read_stage(wafer_dirs, stage, wafer_map, invalid, with_qinj)
    for name, reason in left_out:
        print(f"{label}: {name} left out: {reason}")
    if len(tables) < 2:
        print(f"{label}: {len(tables)} wafer{'s' if len(tables) != 1 else ''} with data in {stage}; "
              "a comparison needs two")
        return None
    values = die_values(tables)
    for q in DIE_QUANTITIES:
        values[f"{q}_centred"] = centred(values, q)
    stats, matrices = {}, {}
    for q in measured(values):
        matrices[q] = pairwise_r(values, q)
        stats[q] = (mean_r(matrices[q]), shuffled_r(values, q))
    other_stage = next(s for s in STAGE_ORDER if s != stage)
    others, _ = read_stage([w for w in wafer_dirs if wafer_label(w) in tables], other_stage, wafer_map, invalid,
                           with_qinj=False)
    positions = next(iter(tables.values()))[0][["die", "die_row", "die_col", "invalid"]].reset_index(drop=True)
    maps = {}
    for q in MAP_QUANTITIES:
        index, residuals, pattern = residual_maps(tables, q)
        if not len(index):
            continue
        same, other, by_die = map_pairs(index, residuals)
        slopes, flat = tilts(residuals)
        o_index, o_residuals, _ = residual_maps(others, q)
        maps[q] = {"pattern": pattern, "same": same, "other": other, "by_die": by_die,
                   "same_die": same_die_r(index, residuals, o_index, o_residuals),
                   "same_untilted": map_pairs(index, flat)[0], "tilt": place_tilts(index, slopes, positions)}
    return {"tables": tables, "values": values, "stats": stats, "matrices": matrices, "maps": maps,
            "positions": positions, "title": title_of(label, stage, tables),
            "injected": common_injected(tables) if with_qinj else []}


def report(label, result):
    """The numbers behind the figures, printed."""
    for q, (mean, shuffled) in result["stats"].items():
        lo, hi = pd.Series(shuffled).quantile([0.025, 0.975])
        print(f"  {DIE_QUANTITIES[q]:17s} mean pairwise r {mean:+.3f} (shuffled positions: 95 % within "
              f"{lo:+.3f} to {hi:+.3f})")
    for q, m in result["maps"].items():
        line = (f"  {MAP_QUANTITIES[q]:17s} map r median: same place {pd.Series(m['same']).median():+.3f} "
                f"({len(m['same'])} pairs), different places {pd.Series(m['other']).median():+.3f} "
                f"({len(m['other'])} pairs)")
        if len(m["same_die"]):
            line += f", same die other stage {pd.Series(m['same_die']).median():+.3f} ({len(m['same_die'])} dies)"
        print(line)
        tilt = m["tilt"]
        print(f"  {MAP_QUANTITIES[q]:17s} tilt across each die (median over wafers at each place): median rise "
              f"{15 * np.hypot(tilt['col'], tilt['row']).median():.2f} DAC code across the die, cosine with the outward radius "
              f"median {tilt['outward'].median():+.2f}; same-place map r with the tilt taken out "
              f"{pd.Series(m['same_untilted']).median():+.3f}")


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    wafer_map = load_wafer_map(args.wafer_map)
    invalid = invalid_dies(args.wafer_map)
    out_dir = Path(args.out) if args.out else Path(args.path) / "position_summary"
    specs = [parse_lot(s) for s in args.lots]
    labels = [label for label, *_ in specs]
    if len({slug(label) for label in labels}) != len(labels):
        print(f"two lots share a label, or its file-name form: {', '.join(labels)}; give each --lot its own LABEL=")
        return 2
    results = {}
    for label, batch, wafers in specs:
        found = find_wafer_dirs(args.path, batch_name=batch)
        if wafers is not None:
            names = {wafer_label(w): w for w in found}
            missing = [w for w in wafers if w not in names]
            if missing:
                print(f"no wafer folder {', '.join(missing)} of batch {batch} under {args.path}")
                return 2
            found = [names[w] for w in wafers]
        if not found:
            print(f"{label}: no wafer folder of batch {batch} under {args.path}")
            return 2
        print(f"{label}: reading {args.stage} of {len(found)} wafers of batch {batch}")
        result = compare_lot(label, found, args.stage, wafer_map, invalid, not args.no_qinj)
        if result is not None:
            results[label] = result
            report(label, result)
    if not results:
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    stamp = f"plot_positions.py {datetime.now():%Y-%m-%d %H:%M}"
    for label, result in results.items():
        prefix = f"{slug(label)}_{args.stage}"
        path = out_dir / f"{prefix}_positions.csv"
        result["values"].to_csv(path, index=False)
        written.append(path)
        if not args.tables_only:
            from position_plots import plot_positions  # matplotlib is needed from here on only
            setups = result["values"].groupby("wafer")["imported"].any()
            imported = [w for w, i in setups.items() if i]
            note = (f"{label} {args.stage}: newest run of each die; invalid dies left out"
                    + (f"; QInj means over the pixels every wafer injected: "
                       f"{', '.join(f'({r},{c})' for r, c in result['injected'])}" if result["injected"] else "")
                    + (f"; imported from an earlier test: {', '.join(imported)}" if imported else "")
                    + f"; {stamp}")
            written += plot_positions(out_dir, prefix, result["values"], result["positions"], result["stats"],
                                      result["matrices"], result["maps"], title=result["title"], note=note)
    print(f"{len(written)} files written to {out_dir}:")
    for path in written:
        print(f"  {path.name}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
