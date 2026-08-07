import argparse, getpass, shutil, sys

from pathlib import Path
from natsort import natsorted


def find_combo_dirs(mother_dir: Path, file_pattern: str) -> list[Path]:
    """Resolves one -s source to the list of per-combo directories it contains.

    Mirrors core/reshape_event_to_track.py's find_combo_dirs(): if mother_dir
    itself contains files matching file_pattern, it's already a single combo
    directory (step 8 writes one such directory per board combo) and is
    returned as the sole entry. Otherwise mother_dir is treated as the shared
    parent of several combo dirs (e.g. a step 8 -o outdir containing
    "trig0-ref1-dut2/" and "extra0-ref1-dut2/"), and each subdirectory that
    actually contains matching files is entered.
    """
    if list(mother_dir.glob(file_pattern)):
        return [mother_dir]

    combo_dirs = [d for d in sorted(mother_dir.iterdir()) if d.is_dir() and list(d.glob(file_pattern))]
    if not combo_dirs:
        sys.exit(f"Error: No files matching '{file_pattern}' found in {mother_dir} or its subdirectories.")
    return combo_dirs


def collect_files_by_combo(sources: list[Path], file_pattern: str) -> dict[str, list[Path]]:
    """Maps combo_label -> every source file to move into <dest>/<combo_label>/,
    across all -s directories."""
    files_by_combo: dict[str, list[Path]] = {}
    for source in sources:
        for combo_dir in find_combo_dirs(source, file_pattern):
            combo_label = combo_dir.name
            files_by_combo.setdefault(combo_label, []).extend(natsorted(combo_dir.glob(file_pattern)))
    return files_by_combo


def find_collisions(files_by_combo: dict[str, list[Path]], dest: Path) -> list[str]:
    """Filename collisions within a combo's target directory -- either between
    two source files sharing a name, or a source file matching one already
    sitting in <dest>/<combo_label>/ from a prior merge. Reported as
    '<combo_label>/<filename>' so it can be traced back to the offending pair."""
    collisions = []
    for combo_label, files in files_by_combo.items():
        seen: dict[str, int] = {}
        target_dir = dest / combo_label
        if target_dir.is_dir():
            for existing in target_dir.glob('*'):
                seen[existing.name] = seen.get(existing.name, 0) + 1

        for f in files:
            seen[f.name] = seen.get(f.name, 0) + 1

        for fname, count in seen.items():
            if count > 1:
                collisions.append(f"{combo_label}/{fname}")

    return collisions


def main():
    parser = argparse.ArgumentParser(
        prog='Merge combo directories',
        description="Move a board combo's parquet files from multiple step 8 output "
                     'directories (e.g. one per run) into a single directory, so step 9 '
                     '(reshape_event_to_track.py) can reshape the full combined dataset.',
    )

    parser.add_argument(
        '-s', '--sources',
        nargs='+',
        required=True,
        help='Two or more step 8 output directories to merge. Each can be a single combo '
             "directory or a combo mother directory (one subdirectory per board combo) -- "
             'auto-detected the same way step 9 does.',
    )

    parser.add_argument(
        '-d', '--dest',
        required=True,
        help='Destination mother directory. Each combo is merged into <dest>/<combo_label>/.',
    )

    parser.add_argument(
        '--file_pattern',
        default='*.parquet',
        help='Glob pattern for input files. Default: *.parquet',
    )

    parser.add_argument(
        '--dryrun',
        action='store_true',
        help='If set, only print the planned moves and collision check -- nothing is moved.',
    )

    args = parser.parse_args()

    if len(args.sources) < 2:
        sys.exit('Error: Give at least two -s/--sources to merge.')

    username = getpass.getuser()
    eos_base_dir = Path(f'/eos/user/{username[0]}/{username}')

    sources = [Path(f'{eos_base_dir}/{s}').resolve() for s in args.sources]
    dest = Path(f'{eos_base_dir}/{args.dest}').resolve()

    for source in sources:
        if not source.is_dir():
            sys.exit(f'Error: Source directory not found: {source}')

    files_by_combo = collect_files_by_combo(sources, args.file_pattern)

    collisions = find_collisions(files_by_combo, dest)
    if collisions:
        print(f'Error: {len(collisions)} filename collision(s) found -- aborting before moving anything:')
        for c in natsorted(collisions):
            print(f'  {c}')
        sys.exit(1)

    for combo_label, files in files_by_combo.items():
        target_dir = dest / combo_label
        if args.dryrun:
            print(f'[Dry Run] {combo_label}: {len(files)} file(s) -> {target_dir}/')
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.move(str(f), str(target_dir / f.name))
        print(f'{combo_label}: moved {len(files)} file(s) -> {target_dir}/')

    if args.dryrun:
        print('No collisions found.')
    else:
        print('\nDone.')


if __name__ == '__main__':
    main()
