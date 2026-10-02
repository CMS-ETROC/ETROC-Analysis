"""plot_power.py -- the power each chip draws from the supplies, on every
wafer of the lots given.

    python plot_power.py --path <path> --lot "N62M23 bare=N62M23:08A5,07B2,06B7,05C4" --lot N62H30

with the --path of the wafer runs, as for plot_wafer.py, and each --lot as
for plot_lots.py (LABEL=BATCH:WAFER,...; all wafers of the batch when none
are listed). Of each wafer it reads the dies.csv that plot_wafer.py wrote
into the plots/ folder of each stage it has (pre_ubm, post_ubm), so run
plot_wafer.py on every wafer and stage first; a stage without that table is
named and left out. A die's power in a run phase is NOMINAL_V, the 1.2 V
the chip is designed for, times the sum over the rails POWER_RAILS of the
phase-median current, at power-on and at high power, as dies.csv has
them: the same on every rail and every test, whatever the supplies were
set to and whatever the cables and the probe card drop. A die whose run
did not log every one of these rails in a phase has no power in that
phase. The eFuse rail is left out: not every run logs it, and where one
does it reads at most 10 mA on a PASSED die. Power-on runs the
preamplifiers at their lower power setting, high power at their high
one (IBSel); the figures draw what the ETROC2 Reference Manual (rev 0.6,
Table 21) expects for the two, 0.77 W and 0.97 W per chip within +-20 %,
an estimate from simulation and the earlier ETROC0 and ETROC1 chips.

It writes into --out (default <path>/power_summary/): power_dies.csv, one
row per die of every wafer and stage read, with each rail's power and the
sums, and the figures (power_plots.py) power.png, the power of the PASSED
valid dies of each wafer at power-on and at high power, one panel per
stage, and power_rails.png, the mean power of each rail per lot and stage.
Like plot_wafer.py it only reads results.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from plot_lots import parse_lot, read_lots, slug
from stage_compare import STAGE_ORDER

POWER_RAILS = ("analog", "digital", "ws_analog", "ws_digital", "vref")
NOMINAL_V = 1.2
PHASES = {"on": "power-on", "high": "high power"}
DIE_COLUMNS = ["die", "die_row", "die_col", "invalid", "grade", "start",
               "analog_V_high", "digital_V_high"]


def build_arg_parser():
    parser = argparse.ArgumentParser(description=" ".join(__doc__.split("\n\n")[0].split()))
    parser.add_argument('--path', required=True,
                        help='The --path of the wafer runs: the mother directory of all results')
    parser.add_argument('--lot', action='append', required=True, dest='lots', metavar='LABEL=BATCH:WAFER,...',
                        help='The wafers of batch BATCH (all, or the ones listed after the colon), labelled '
                             'LABEL (default: the batch name); repeat for each lot')
    parser.add_argument('--out', default=None, help='Output folder (default: <path>/power_summary)')
    parser.add_argument('--tables-only', action='store_true', dest='tables_only',
                        help='Write power_dies.csv only, no figures')
    return parser


def _column(dies, name):
    if name in dies:
        return pd.to_numeric(dies[name], errors="coerce")
    return pd.Series(float("nan"), index=dies.index)


def die_power(dies):
    """Per die, the power of each rail in each phase at NOMINAL_V
    (<rail>_P_<phase>, W) and their sum over POWER_RAILS (power_<phase>_W),
    NaN where the run did not log one of the rails."""
    out = pd.DataFrame(index=dies.index)
    for phase in PHASES:
        parts = [(NOMINAL_V * _column(dies, f"{rail}_I_{phase}")).rename(rail) for rail in POWER_RAILS]
        for part in parts:
            out[f"{part.name}_P_{phase}"] = part
        out[f"power_{phase}_W"] = pd.concat(parts, axis=1).sum(axis=1, min_count=len(POWER_RAILS))
    return out


def read_wafer(wafer_dir):
    """(the power rows of each stage of the wafer that has a dies.csv,
    [(stage, reason left out)])."""
    tables, left_out = [], []
    for stage in STAGE_ORDER:
        if not (wafer_dir / stage).is_dir():
            continue
        path = wafer_dir / stage / "plots" / "dies.csv"
        if not path.is_file():
            left_out.append((stage, f"no {path.relative_to(wafer_dir)}: run plot_wafer.py on it first"))
            continue
        dies = pd.read_csv(path)
        rows = dies.reindex(columns=DIE_COLUMNS)
        rows["invalid"] = rows["invalid"].fillna(False).astype(bool)
        rows.insert(0, "stage", stage)
        tables.append(pd.concat([rows, die_power(dies)], axis=1))
    return tables, left_out


def usable(power):
    """The rows the figures use: valid dies graded PASSED."""
    return power[(power["grade"] == "PASSED") & ~power["invalid"]]


def report(power):
    """One line per wafer and stage: its PASSED valid dies and their median
    power in each phase, and how many of them have no power in a phase."""
    for (lot, wafer, stage), rows in power.groupby(["lot", "wafer", "stage"], sort=False):
        good = usable(rows)
        parts = []
        for phase, text in PHASES.items():
            values = good[f"power_{phase}_W"].dropna()
            missing = len(good) - len(values)
            parts.append((f"{text} {values.median():.3f} W" if len(values) else f"no power at {text}")
                         + (f" ({missing} without)" if missing and len(values) else ""))
        print(f"  {lot} {wafer} {stage}: {len(good)} PASSED valid dies, median {', '.join(parts)}")


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    out_dir = Path(args.out) if args.out else Path(args.path) / "power_summary"
    specs = [parse_lot(s) for s in args.lots]
    labels = [label for label, *_ in specs]
    if len({slug(label) for label in labels}) != len(labels):
        print(f"two lots share a label, or its file-name form: {', '.join(labels)}; give each --lot its own LABEL=")
        return 2
    power = read_lots(args.path, specs, read_wafer)
    if power is None:
        return 2
    report(power)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [out_dir / "power_dies.csv"]
    power.to_csv(written[0], index=False)
    if not args.tables_only and usable(power)[[f"power_{p}_W" for p in PHASES]].isna().all(axis=None):
        print(f"{written[0]} written; no PASSED valid die of the lots has a power, so no figures")
        return 2
    if not args.tables_only:
        from power_plots import plot_power  # matplotlib is needed from here on only
        note = (f"lots {', '.join(labels)}: dies.csv of each wafer and stage (plot_wafer.py, newest run of each die); "
                f"plot_power.py {datetime.now():%Y-%m-%d %H:%M}")
        written += plot_power(out_dir, power, note=note)
    print(f"{len(written)} files written to {out_dir}:")
    for path in written:
        print(f"  {path.name}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
