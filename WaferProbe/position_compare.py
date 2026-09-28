"""position_compare.py -- do the dies at one place on the wafer behave alike
on the wafers of a lot? The tables and numbers behind plot_positions.py
(position_plots.py draws them). Needs numpy and pandas, no matplotlib.

For one stage of each wafer (wafer_tables.collect: the newest run of each
die), per die: the mean baseline and noise width of its pixels (full scans
only; a pixel reading 0 is left out) and the mean CAL, TOA and TOT of the
pixels every wafer of the lot injected (common_injected). Two measures:

  pairwise r  for two wafers, the correlation of a die-level quantity over
              the positions both measured. Dies that behave alike at one
              place make it positive; without that it stays at the level of
              the same wafers with each wafer's values moved at random
              among the places it measured (shuffled_r). An offset or a scale factor between
              two wafers, or two test setups, does not change it.
  map r       for two full-scan dies, the correlation over the pixels both
              read of each pixel's baseline (or noise width) minus its
              die's mean and minus the lot's pattern, the median of that
              over all dies, which every chip shares. Dies at the same
              place on two wafers against dies at two different places;
              the same die measured in the other stage (where both stages
              have its full scan) shows what one chip measured twice gives.
  tilt        of each full-scan die's residual map, the least-squares plane
              over its pixels: its rise per pixel along the pixel columns
              and rows (tilts), the median over the wafers at each place
              and the cosine of that with the outward radius of the wafer
              map (place_tilts). The map r of the same place with the tilt
              taken out tells a shared tilt from a finer shared pattern.
              The pixel maps are seen notch up, as the wafer map, so the
              pixel rows and columns run the way the map's rows and
              columns do.

Every wafer is tested in die-number order, which runs through the wafer
map row by row from the top, serpentine. A drift during a pass would
therefore repeat by position on every wafer: in die-level values it shows
as a trend with the die number, a top-to-bottom gradient on the map. The
map r takes each die's mean out, so a drift that moves a whole die leaves
it alone.
"""
import warnings

import numpy as np
import pandas as pd

from stage_compare import _good_pixels
from wafer_tables import injected_pixels

FULL_SCAN_PIXELS = 256
DIE_QUANTITIES = {"baseline": "baseline mean", "noise_width": "noise-width mean",
                  "cal": "CAL mean", "toa": "TOA mean", "tot": "TOT mean"}
MAP_QUANTITIES = {"baseline": "baseline", "noise_width": "noise width"}
MIN_POSITIONS = 20    # positions two wafers must both have measured for a pairwise r
MIN_MAP_PIXELS = 200  # pixels two dies must both have read for a map r
MIN_QINJ_HITS = 10    # hits a pixel needs (n_sel) for its QInj means to count
SHUFFLES = 200
VALUE_COLUMNS = ["wafer", "die", "die_row", "die_col", "start", "imported"] + list(DIE_QUANTITIES)


def full_scan(dies):
    """The valid dies whose calibration read every pixel."""
    n = pd.to_numeric(dies["n_pixels"], errors="coerce") if "n_pixels" in dies else pd.Series(np.nan, dies.index)
    return set(dies.loc[(n >= FULL_SCAN_PIXELS) & ~dies["invalid"], "die"])


def common_injected(tables):
    """The pixels that every wafer with QInj data injected (injected_pixels),
    sorted; [] when no wafer has QInj data."""
    sets = [set(injected_pixels(qinj)) for _, _, qinj in tables.values() if not qinj.empty]
    return sorted(set.intersection(*sets)) if sets else []


def injected_notes(tables):
    """Lines naming what common_injected leaves out: a wafer's injected
    pixels beyond the ones every wafer injected, or, when no pixel is
    common to all, that and any wafer with QInj data but no injected
    pixel."""
    sets = {w: set(injected_pixels(qinj)) for w, (_, _, qinj) in tables.items() if not qinj.empty}
    common = set.intersection(*sets.values()) if sets else set()
    lines = []
    for wafer, pixels in sets.items():
        if not pixels:
            lines.append(f"{wafer}: no pixel hit in half the events of half its QInj dies")
        elif common and pixels - common:
            extra = ", ".join(f"({r},{c})" for r, c in sorted(pixels - common))
            lines.append(f"{wafer}: injected {extra} too, left out of the QInj means")
    if sets and not common:
        lines.append("no pixel injected on every wafer with QInj data: CAL, TOA and TOT left out")
    return lines


def die_values(tables):
    """One row per valid die with a value, of every wafer in `tables`
    ({wafer: (dies, pixels, qinj)} as collect returns them): the
    DIE_QUANTITIES, NaN where the die was not measured. The QInj means are
    the mean over the common injected pixels of each pixel's mean; a die
    missing one of them, or with fewer than MIN_QINJ_HITS hits on one, gets
    none."""
    common = set(common_injected(tables))
    rows = []
    for wafer, (dies, pixels, qinj) in tables.items():
        good = _good_pixels(pixels)
        means = good[good["die"].isin(full_scan(dies))].groupby("die")[["baseline", "noise_width"]].mean()
        injected = np.array([(r, c) in common for r, c in zip(qinj["pix_row"], qinj["pix_col"])], dtype=bool)
        q = qinj[(qinj["n_sel"].to_numpy(dtype=float) >= MIN_QINJ_HITS) & injected]
        tdc = q.groupby("die").agg(n=("pix_row", "size"), cal=("cal_mean", "mean"),
                                   toa=("toa_mean", "mean"), tot=("tot_mean", "mean"))
        tdc = tdc[tdc["n"] == len(common)] if common else tdc.iloc[0:0]
        for d in dies[~dies["invalid"]].itertuples(index=False):
            record = {"wafer": wafer, "die": d.die, "die_row": d.die_row, "die_col": d.die_col,
                      "start": getattr(d, "start", None), "imported": bool(getattr(d, "imported", False))}
            if d.die in means.index:
                record.update(means.loc[d.die].to_dict())
            if d.die in tdc.index:
                record.update(tdc.loc[d.die, ["cal", "toa", "tot"]].to_dict())
            if any(np.isfinite(record.get(q, np.nan)) for q in DIE_QUANTITIES):
                rows.append(record)
    return pd.DataFrame(rows, columns=VALUE_COLUMNS)


def measured(values):
    """The DIE_QUANTITIES that at least two wafers measured."""
    return [q for q in DIE_QUANTITIES
            if values.loc[np.isfinite(values[q].astype(float)), "wafer"].nunique() >= 2]


def centred(values, quantity):
    """The quantity minus its wafer's median."""
    v = values[quantity].astype(float)
    return v - v.groupby(values["wafer"]).transform("median")


def _wide(values, quantity):
    return values.pivot(index="die", columns="wafer", values=quantity).astype(float)


def pairwise_r(values, quantity):
    """wafer x wafer: the correlation of the quantity over the positions both
    wafers measured; NaN below MIN_POSITIONS of them."""
    return _wide(values, quantity).corr(min_periods=MIN_POSITIONS)


def mean_r(r):
    """The mean of the off-diagonal entries of a pairwise r matrix."""
    m = np.asarray(r, dtype=float)
    v = m[~np.eye(len(m), dtype=bool)]
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def shuffled_r(values, quantity, n=SHUFFLES, seed=0):
    """The mean pairwise r, n times, with each wafer's values moved at
    random among the places it measured: what the lot shows with no
    position effect."""
    wide = _wide(values, quantity)
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for k in range(n):
        shuffled = wide.copy()
        for w in wide:
            v = wide[w].to_numpy(copy=True)
            read = np.isfinite(v)
            v[read] = rng.permutation(v[read])
            shuffled[w] = v
        out[k] = mean_r(shuffled.corr(min_periods=MIN_POSITIONS))
    return out


def residual_maps(tables, quantity, dies_of=None):
    """(index, maps, pattern) over every full-scan die of every wafer (or,
    with `dies_of` {wafer: dies}, those dies only): index the (wafer, die) of
    each, maps an array with one row of 256 pixels (row * 16 + column) per
    die, each pixel's quantity minus its die's mean and minus `pattern`,
    NaN where not read; pattern (16 x 16) the median over the dies of each
    pixel's quantity minus its die's mean."""
    keys, rows = [], []
    for wafer, (dies, pixels, _) in tables.items():
        keep = full_scan(dies) if dies_of is None else full_scan(dies) & set(dies_of.get(wafer, ()))
        good = _good_pixels(pixels)
        for die, g in good[good["die"].isin(keep)].groupby("die"):
            m = np.full(256, np.nan)
            m[g["pix_row"].to_numpy(int) * 16 + g["pix_col"].to_numpy(int)] = g[quantity].to_numpy(float)
            keys.append((wafer, die))
            rows.append(m - np.nanmean(m))
    maps = np.array(rows) if rows else np.empty((0, 256))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a pixel no die read
        pattern = np.nanmedian(maps, axis=0) if len(maps) else np.full(256, np.nan)
    return pd.DataFrame(keys, columns=["wafer", "die"]), maps - pattern, pattern.reshape(16, 16)


def _standard(maps):
    """(z, read, flat): each row centred and scaled over its read pixels, 0
    where not read; read as 0/1; flat marks rows without spread."""
    read = np.isfinite(maps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(maps, axis=1, keepdims=True)
        sd = np.nanstd(maps, axis=1, keepdims=True)
    flat = ~(sd[:, 0] > 0)
    z = np.where(read, (maps - mean) / np.where(sd > 0, sd, 1.0), 0.0)
    return np.nan_to_num(z), read.astype(float), flat


def map_r(a, b):
    """Matrix of the map r of every row of `a` with every row of `b`, over
    the pixels both read; NaN below MIN_MAP_PIXELS of them or for a flat
    map."""
    za, ra, fa = _standard(a)
    zb, rb, fb = _standard(b)
    n = ra @ rb.T
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (za @ zb.T) / n
    r[n < MIN_MAP_PIXELS] = np.nan
    r[fa, :] = np.nan
    r[:, fb] = np.nan
    return r


def map_pairs(index, maps):
    """(same, other, by_die): the map r of every two dies of different
    wafers at the same position (same die number), of every two at
    different positions, and per die number the median of its same-position
    ones."""
    r = map_r(maps, maps)
    wafer = index["wafer"].to_numpy()
    die = index["die"].to_numpy()
    upper = np.triu(np.ones(r.shape, dtype=bool), 1) & (wafer[:, None] != wafer[None, :])
    same = upper & (die[:, None] == die[None, :])
    other = upper & (die[:, None] != die[None, :])
    by_die = {}
    for d in np.unique(die):
        k = np.flatnonzero(die == d)
        v = r[np.ix_(k, k)][np.triu(np.ones((len(k), len(k)), dtype=bool), 1)]
        v = v[np.isfinite(v)]
        if v.size:
            by_die[int(d)] = float(np.median(v))
    finite = np.isfinite(r)
    return r[same & finite], r[other & finite], pd.Series(by_die, dtype=float)


def tilts(maps):
    """(slopes, flat): of each map (one row of 256 pixels, row * 16 +
    column) the least-squares plane over the pixels read, as its rise per
    pixel along the pixel columns and along the pixel rows (NaN below
    MIN_MAP_PIXELS read), and the maps with that plane taken out."""
    row, col = np.divmod(np.arange(256), 16)
    design = np.column_stack([np.ones(256), col - 7.5, row - 7.5])
    slopes, flat = np.full((len(maps), 2), np.nan), np.full(np.shape(maps), np.nan)
    for k, m in enumerate(maps):
        read = np.isfinite(m)
        if read.sum() < MIN_MAP_PIXELS:
            continue
        coef = np.linalg.lstsq(design[read], m[read], rcond=None)[0]
        slopes[k] = coef[1:]
        flat[k] = m - design @ coef
    return slopes, flat


def place_tilts(index, slopes, positions):
    """Per die number: the median over the wafers of the rise per pixel
    along the pixel columns (col) and rows (row), and the cosine of that
    tilt with the outward radius from the centre of the wafer map, the
    mean place of `positions` (outward; NaN at the centre)."""
    tilt = index.assign(col=slopes[:, 0], row=slopes[:, 1]).dropna().groupby("die")[["col", "row"]].median()
    places = positions.set_index("die")
    dx = (places["die_col"] - places["die_col"].mean()).reindex(tilt.index)
    dy = (places["die_row"] - places["die_row"].mean()).reindex(tilt.index)
    with np.errstate(invalid="ignore", divide="ignore"):
        tilt["outward"] = (tilt["col"] * dx + tilt["row"] * dy) / (np.hypot(tilt["col"], tilt["row"])
                                                                    * np.hypot(dx, dy))
    return tilt


def same_die_r(index, maps, other_index, other_maps):
    """The map r of each die with itself in the other stage (same wafer and
    die number), for the dies both index tables hold."""
    both = index.reset_index().merge(other_index.reset_index(), on=["wafer", "die"], suffixes=("", "_other"))
    if both.empty:
        return np.array([])
    za, ra, fa = _standard(maps[both["index"].to_numpy()])
    zb, rb, fb = _standard(other_maps[both["index_other"].to_numpy()])
    n = (ra * rb).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (za * zb).sum(axis=1) / n
    r[(n < MIN_MAP_PIXELS) | fa | fb] = np.nan
    return r[np.isfinite(r)]


def months_tested(dies):
    """When a wafer's stage was tested: "MM/YYYY" or "MM/YYYY-MM/YYYY", from
    the start times of its dies; "?" without any."""
    days = pd.to_datetime(dies.get("start"), errors="coerce").dropna()
    if days.empty:
        return "?"
    first, last = f"{days.min():%m/%Y}", f"{days.max():%m/%Y}"
    return first if first == last else f"{first}-{last}"
