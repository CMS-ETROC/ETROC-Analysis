import argparse, sys
from pathlib import Path
from collections import defaultdict, deque

import pandas as pd
from tabulate import tabulate

GRID = 16  # 16x16 pixel plane per board

parser = argparse.ArgumentParser(
    prog='Select track candidates by coverage',
    description="Select track candidates that guarantee every pixel on each board's 16x16 "
                'grid is backed by at least --target-depth candidates, instead of a plain '
                'top-N-by-count cut (which can leave whole regions of the grid with only 1 '
                'candidate, or none, while piling many redundant candidates onto a few hot '
                'pixels). Candidates are walked in descending occurrence-count order to first '
                'reach full coverage, then water-filled -- always topping up whichever pixel '
                'currently has the fewest candidates -- until every reachable pixel meets the '
                'target depth (or genuinely runs out of candidates for it). The resulting '
                'candidate count is exactly however many that depth guarantee costs -- no '
                'arbitrary fixed budget.',
)

parser.add_argument(
    '-f',
    '--file',
    metavar='FILE_OR_DIR',
    type=str,
    help='Track combination table as Parquet format for input, or a directory of them '
         '(path_finder.py writes one file per board combo into a per-run directory -- '
         'point -f at that directory to select tracks for every combo in one call)',
    required=True,
    dest='file',
)

parser.add_argument(
    '-d',
    '--target-depth',
    metavar='D',
    type=int,
    default=4,
    help='Minimum number of track candidates required at every pixel (default: 4). Higher '
         'values guarantee deeper per-pixel statistics at the cost of more total candidates. '
         'Since different combos/runs need different totals to satisfy the same D, this can '
         'blow past --max-candidates if given -- see that flag for the fallback behavior.',
    dest='target_depth',
)

parser.add_argument(
    '-n',
    '--max-candidates',
    metavar='N',
    type=int,
    default=None,
    help='Hard ceiling on the number of candidates kept per file. If --target-depth would '
         'exceed this, --target-depth is lowered by 1 and retried (repeating down to depth 1, '
         'i.e. coverage only) until the result fits under the ceiling. If even depth 1 '
         "(bare coverage) exceeds N, the depth-1 result is kept anyway with a warning -- N is "
         'too small to guarantee full coverage at all, and coverage is never sacrificed to '
         'force a fit.',
    dest='max_candidates',
)

args = parser.parse_args()

input_path = Path(args.file)
if input_path.is_dir():
    # Skip already-reduced outputs so re-running against the same directory doesn't try to
    # select from an already-selected file again. Output uses the '_reduced' suffix (not
    # '_selected') so it's a drop-in for submit_extract_events_by_path.py's find_track_files(),
    # which globs '*_reduced.parquet'.
    input_files = sorted(
        f for f in input_path.glob('*.parquet')
        if not f.stem.endswith('_reduced')
    )
    if not input_files:
        sys.exit(f'No .parquet files found in directory {input_path}')
else:
    input_files = [input_path]


def board_ids_from_columns(df: pd.DataFrame) -> list[int]:
    return sorted(int(c.split('_')[1]) for c in df.columns if c.startswith('row_'))


def select_by_depth(df: pd.DataFrame, board_ids: list[int], target_depth: int):
    """Two-phase greedy selection.

    Phase 1 (coverage): walk candidates in descending 'count' order, keep one only if it
    touches a pixel (on any board) not yet touched by an already-kept candidate -- reaches
    the maximum achievable coverage using as few candidates as possible.

    Phase 2 (depth water-filling): repeatedly top up whichever covered pixel currently has
    the fewest kept candidates, using the best (highest-count) remaining candidate that
    touches it, until every covered pixel reaches target_depth or genuinely runs out of
    candidates for it. This raises the floor everywhere instead of piling more candidates
    onto pixels that are already well represented, which is what a plain count-ranked
    top-N cut does."""
    df_sorted = df.sort_values('count', ascending=False).reset_index(drop=True)
    full_coverage = GRID * GRID

    # Pre-extract as plain arrays -- avoids per-row pandas overhead in the loops below,
    # which matters here since files can have 100k+ candidates.
    row_arrays = {b: df_sorted[f'row_{b}'].to_numpy() for b in board_ids}
    col_arrays = {b: df_sorted[f'col_{b}'].to_numpy() for b in board_ids}

    def cells_of(i):
        return [(b, (int(row_arrays[b][i]), int(col_arrays[b][i]))) for b in board_ids]

    # --- Phase 1: coverage ---
    depth = defaultdict(int)  # (board, (row,col)) -> appearance count among selected
    covered = {b: set() for b in board_ids}
    selected = []
    leftover = []

    for i in range(len(df_sorted)):
        cells = cells_of(i)
        is_new = any(cell not in covered[b] for b, cell in cells)
        if is_new:
            selected.append(i)
            for b, cell in cells:
                covered[b].add(cell)
                depth[(b, cell)] += 1
        else:
            leftover.append(i)
        if all(len(covered[b]) >= full_coverage for b in board_ids):
            leftover.extend(range(i + 1, len(df_sorted)))
            break

    # --- Phase 2: depth water-filling ---
    active = {(b, px) for b in board_ids for px in covered[b]}
    pixel_queue = defaultdict(deque)
    for i in leftover:
        for cell in cells_of(i):
            pixel_queue[cell].append(i)

    used = set(selected)
    under = {p for p in active if depth[p] < target_depth}
    while under:
        p = min(under, key=lambda p: depth[p])
        q = pixel_queue[p]
        while q and q[0] in used:
            q.popleft()
        if not q:
            under.discard(p)  # genuinely exhausted -- can't reach target for this pixel
            continue
        idx = q.popleft()
        used.add(idx)
        selected.append(idx)
        for cell in cells_of(idx):
            depth[cell] += 1
        under = {p for p in active if depth[p] < target_depth}

    all_pixels = {(r, c) for r in range(GRID) for c in range(GRID)}
    missing_pixels = {b: sorted(all_pixels - covered[b]) for b in board_ids}

    depth_by_board = {b: [] for b in board_ids}
    for (b, px), d in depth.items():
        depth_by_board[b].append(d)

    stats = {
        'coverage': {b: len(covered[b]) / full_coverage for b in board_ids},
        'missing': missing_pixels,
        'min_depth': {b: min(depth_by_board[b]) if depth_by_board[b] else 0 for b in board_ids},
        'max_depth': {b: max(depth_by_board[b]) if depth_by_board[b] else 0 for b in board_ids},
    }
    return df_sorted.loc[selected].reset_index(drop=True), stats


for file in input_files:
    track_df = pd.read_parquet(file)
    previous_num = track_df.shape[0]
    board_ids = board_ids_from_columns(track_df)

    # Different combos/runs need a different total to satisfy the same target depth (that's
    # the whole point -- it's derived from the data, not fixed), so with a ceiling given, walk
    # the depth down one step at a time until the result fits, rather than truncating the
    # depth-D selection (which would undo the water-filling and reintroduce the exact
    # imbalance -- a few candidates over the ceiling -- this flag exists to avoid).
    depth_used = args.target_depth
    selected_df, stats = select_by_depth(track_df, board_ids, depth_used)
    while args.max_candidates is not None and selected_df.shape[0] > args.max_candidates and depth_used > 1:
        depth_used -= 1
        selected_df, stats = select_by_depth(track_df, board_ids, depth_used)

    if args.max_candidates is not None and selected_df.shape[0] > args.max_candidates:
        print(f'\nWarning: {file.name}: even depth 1 (bare coverage) needs {selected_df.shape[0]} '
              f'candidates, above --max-candidates {args.max_candidates}. Keeping it anyway -- '
              'coverage is never sacrificed to force a fit under the ceiling.')

    output_file = file.with_name(f'{file.stem}_reduced.parquet')
    selected_df.to_parquet(output_file, index=False)

    table_data = [
        [
            f'board {b}',
            f'{stats["coverage"][b]:.2%}',
            stats['min_depth'][b],
            stats['max_depth'][b],
            ', '.join(f'({r},{c})' for r, c in stats['missing'][b]) or '-',
        ]
        for b in board_ids
    ]
    print(f'\n=== Track Selection by Coverage: {file.name} ===')
    depth_note = f' (requested {args.target_depth}, lowered to fit --max-candidates)' if depth_used != args.target_depth else ''
    print(f'Target depth: {depth_used}{depth_note}')
    print(f'Candidates: {previous_num} -> {selected_df.shape[0]}')
    print(tabulate(
        table_data,
        headers=['Board', 'Pixel coverage', 'Min depth', 'Max depth', 'Missing pixels (row,col)'],
        tablefmt='simple',
    ))
    print(f'New file {output_file} is created')
