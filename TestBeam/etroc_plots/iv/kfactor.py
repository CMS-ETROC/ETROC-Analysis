"""k-factor: depletion (V_gl) and breakdown extraction."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.signal import find_peaks

from ._ivstyle import *
from ._helpers import (_lims, _apply_lims, _style_axes, _ch_label, _mfactor, _as_scan_list, _panel_tag, _grid_fonts, _smooth_volts, _scan_load_kw, _scan_clean_kw, _scan_color, _scan_dash, _place_legend)
from ._loading import load_iv, clean_channel

CURRENT_EPS = 1e-12   # below this, k is masked instead of computed
MIN_PLOT_V  = 0.1     # drop start/idle bookkeeping points sitting at ~0 V


def _pick_current_unit(i_max):
    """Choose a display unit from the largest current in the trace."""
    if not np.isfinite(i_max) or i_max <= 0:
        return 1e-6, "uA"
    if i_max < 1e-6:
        return 1e-9, "nA"
    if i_max < 1e-3:
        return 1e-6, "uA"
    if i_max < 1.0:
        return 1e-3, "mA"
    return 1.0, "A"


def k_factor(v, i, v_grid=None, smooth_v=0.0, low_range=None):
    """
    k = (V/I)(dI/dV) on |V|, |I| in acquisition order.

    v_grid    : voltages used for the derivative spacing. Pass the demand voltage
                (v_set) when you have it: readback jitter on a 0.1 V step scan can
                shrink an interval to 0.009 V, which inflates dI/dV tenfold at that
                point. Defaults to v itself.
    smooth_v  : boxcar width in VOLTS applied to I before differentiating, and to k
                after. A no-op on coarse scans; see _smooth_volts.
    low_range : the ammeter's `low_current_range` flag. The one interval that
                straddles a range change compares readings taken on two different
                ranges, with different quantisation, so its derivative is not
                trustworthy -- those points are dropped and bridged by
                interpolation. On the quick scan that single interval was
                manufacturing a fake k peak at 20 V.
    """
    v = np.abs(np.asarray(v, dtype=float))
    i = np.abs(np.asarray(i, dtype=float))
    x = v if v_grid is None else np.abs(np.asarray(v_grid, dtype=float))
    if len(v) < 3:
        return np.full_like(v, np.nan)

    i_s = _smooth_volts(x, i, smooth_v)
    didv = np.gradient(i_s, x)                   # central difference at the samples
    if low_range is not None:
        lr = np.asarray(low_range).astype(int)
        for j in np.where(np.diff(lr) != 0)[0]:
            didv[max(0, j - 1):min(len(didv), j + 3)] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        k = (v / i_s) * didv
    k[i_s < CURRENT_EPS] = np.nan
    k[~np.isfinite(k)] = np.nan
    ok = np.isfinite(k)
    if ok.sum() >= 2:                            # bridge the masked gap
        k = np.interp(np.arange(len(k)), np.where(ok)[0], k[ok])
    return _smooth_volts(x, k, smooth_v)


def fine_step_span(v_set, max_step=0.5, min_run=10):
    """
    |V| range over which a scan is finely stepped: the LONGEST contiguous run
    of voltages spaced <= max_step, at least min_run steps long.

    The campaign scans are not uniformly stepped -- 20260720_190823 runs 0.1 V
    from 10 to 60 V then opens out to 10 V steps, while 20260717_0307 is fine
    only between 10.1 and 12.8 V. A gain-layer peak found in the coarse region
    is not resolvable, so the search is confined to the fine part.

    Contiguity matters: near breakdown the voltage readback sags under load,
    so auto-binned legacy scans sprout clusters of bins < 0.5 V apart at
    480-500 V. Taking the global (min, max) of every sub-max_step gap -- the
    old behaviour -- stretched the V_gl search window across the whole sweep,
    and the breakdown ramp's curvature then beat the real gain-layer peak
    (202607160017 ch0/2/3 reported V_gl ~ 490 V). The longest fine run is the
    deliberately fine-stepped region; incidental clusters lose to it. min_run
    also rejects scans with no genuine fine region (quick scans return NaN).
    """
    u = np.unique(np.abs(np.asarray(v_set, dtype=float)))
    u = u[np.isfinite(u)]
    if len(u) < 3:
        return np.nan, np.nan
    fine = np.diff(u) <= max_step
    best_lo = best_hi = None
    best_len = 0
    i = 0
    while i < len(fine):
        if fine[i]:
            j = i
            while j < len(fine) and fine[j]:
                j += 1
            if j - i > best_len:
                best_lo, best_hi, best_len = u[i], u[j], j - i
            i = j
        else:
            i += 1
    if best_len < min_run:
        return np.nan, np.nan
    return float(best_lo), float(best_hi)


def range_switch_voltages(good):
    """|V| where the ammeter changed range during a scan."""
    lr = good["low_range"].to_numpy().astype(int)
    j = np.where(np.diff(lr) != 0)[0]
    return good["v_meas"].abs().to_numpy()[j]


def _first_crossing(v, k, level, i=None, min_points=2):
    """
    Interpolated voltage where k first rises through `level` AND stays there.

    A genuine breakdown crossing is sustained -- k holds above the threshold
    for consecutive samples while |I| rises. The low-|V| artifact usually is
    not: below depletion the current sits at the ammeter floor, V/I is
    enormous, and dI/dV is noise, so k spikes through thresholds erratically.
    Requiring `min_points` consecutive samples at or above `level` with |I|
    net-rising across them rejects isolated spikes -- but it is a raised bar,
    not a fence: a FLICKERING sub-depletion region can pass a short run with
    a chance net-rise in floor-quantized current. The deterministic guard is
    the per-scan "min_v" in the scan dict, which removes those voltages
    before k is computed at all; this test is defense-in-depth behind it.
    min_points=1 (or i=None, which skips the rising test) restores the
    single-point behaviour.
    """
    ok = np.isfinite(k) & np.isfinite(v)
    if i is not None:
        ok &= np.isfinite(i)
    v, k = np.asarray(v)[ok], np.asarray(k)[ok]
    ii = np.asarray(i)[ok] if i is not None else None
    n = len(k)
    for j in range(1, n):
        run = 0
        while j + run < n and k[j + run] >= level:
            run += 1
        if run == 0:
            continue
        # a run truncated by the end of the scan is genuine: entering
        # breakdown and tripping compliance on the next dwell is exactly how
        # these scans end (20260720 fine crosses k=8 on its final bin)
        sustained = run >= min_points or (j + run == n)
        rising = True
        if ii is not None:
            last = min(j + run, n) - 1
            rising = ii[last] > ii[j] if last > j else ii[j] > ii[j - 1]
        if sustained and rising:
            if k[j] == k[j - 1]:
                return v[j]
            return v[j - 1] + (level - k[j - 1]) * (v[j] - v[j - 1]) / (k[j] - k[j - 1])
    return np.nan


def plot_kfactor(scans, channel_names=None, title="", unit="auto",
                 xlims=None, ylims_iv=None, ylims_k=None, ylims_dlogi=None,
                 channels=None, k_breakdown=8.0, vgl_search=None,
                 vgl_prominence=0.3, min_v=1.0, smooth_v=1.0, deriv_x="auto",
                 mask_range_switch=True, show_range_switch=True,
                 show_dlogi=False, scale_iv="log", col_width=6.0, row_height=5.0,
                 figsize=None, output_fig=None, verbose=False, **clean_kw):
    """
    One column per channel: |I| vs |V| on top, k-factor below.

    unit           : "auto" picks per channel from the data, or force "nA"/"uA"/"mA"
    k_breakdown    : level whose first crossing is reported as the breakdown voltage
    vgl_search     : (lo, hi) window in |V| to hunt for the gain-layer peak in k,
                     or "auto" to use each scan's own finely-stepped region --
                     the campaign files step 0.1 V over quite different spans
                     (10-13 V, 10-70 V, 1-35 V), so one fixed window cannot serve
                     them all
    vgl_prominence : minimum peak prominence in k; nothing clearing it returns NaN
                     rather than falling back to argmax
    min_v          : ignore |V| below this when computing k; a scan dict may
                     carry its own "min_v" to override per condition (a
                     pre-irradiation scan wants ~15 V, an irradiated one
                     whose fine region starts at 10 V wants less). Near zero bias the
                     current sits on the noise floor and V/I blows up -- on the fine
                     scan that produced a spurious k of 207 at 0.17 V, which buried
                     the real feature at 25 V.
    smooth_v       : boxcar width in volts (not samples). 1 V is a no-op on a 10 V
                     step scan and a 10-sample average on a 0.1 V one.
    deriv_x        : "auto" uses v_set spacing when recorded, else v_meas
    mask_range_switch : drop the derivative across an ammeter autorange change
    show_range_switch : draw a faint grey line where the range changed
    """
    scans = _as_scan_list(scans)
    loaded = [(s, load_iv(s["file"], **_scan_load_kw(s))) for s in scans]
    all_ch = sorted({c for _, t in loaded for c in t["channel"].unique()
                     if channels is None or c in channels})
    if not all_ch:
        raise ValueError("No channels to plot")

    nrows = 3 if show_dlogi else 2
    ncols = len(all_ch)
    fig, grid = plt.subplots(nrows, ncols, sharex="col", squeeze=False,
                             figsize=figsize or (max(14.0, col_width * ncols),
                                                 row_height * nrows),
                             dpi=DPI_SCREEN)
    results = []

    for col, ch in enumerate(all_ch):
        ax_iv, ax_k = grid[0][col], grid[1][col]
        ax_d = grid[2][col] if show_dlogi else None
        col_imax = 0.0

        # --- pass 1: clean every scan in this column and find the largest current.
        # The unit has to be chosen before anything is drawn: picking it inside the
        # loop scaled the first scan by its own maximum and later ones by the
        # running maximum, so two traces on one axis ended up in different units.
        traces = []
        for s_idx, (s, tidy) in enumerate(loaded):
            if ch not in set(tidy["channel"]):
                continue
            kw = dict(clean_kw)
            kw.update(_scan_clean_kw(s))
            good, _, _ = clean_channel(tidy, ch, label=_ch_label(channel_names, ch),
                                       verbose=verbose, **kw)
            if good.empty:
                continue

            if s.get("bins") is not None or good["v_set"].isna().all():
                # Legacy slow-control log: the supply dwells at each set point
                # while the logger samples at ~1 Hz, so the raw trace has
                # hundreds of zero-voltage-spacing pairs and np.gradient on it
                # is 0/0 noise. Collapse onto the voltage binning first, exactly
                # as the IV plot does, and differentiate the binned curve.
                from .legacy import bin_iv, infer_bins, QUICK_BINS
                b = s.get("bins", "auto")
                if isinstance(b, str) and b == "auto":
                    b = infer_bins(good)
                curve = bin_iv(good, b if b is not None else QUICK_BINS)
                if curve is None or len(curve) < 3:
                    continue
                v = curve["V_mean"].abs().to_numpy(dtype=float)
                i = curve["I_filtered"].abs().to_numpy(dtype=float)
                order = np.argsort(v)          # binned curve has no sweep order
                v, i = v[order], i[order]
                vset = np.full_like(v, np.nan)
                gk = None                      # no per-reading flags after binning
            else:
                # acquisition order preserved -- do NOT sort by voltage
                good = good.sort_values("timestamp")
                v = good["v_meas"].abs().to_numpy(dtype=float)
                vset = good["v_set"].abs().to_numpy(dtype=float)
                i = good["i_meas"].abs().to_numpy(dtype=float)
                gk = good

            mv = float(s.get("min_v", min_v))   # per-scan override, like i_max
            keep = v > max(MIN_PLOT_V, mv)
            v, vset, i = v[keep], vset[keep], i[keep]
            gk = gk[keep] if gk is not None else None
            if len(v) < 3:
                continue
            # unit choice must reflect what the axis will SHOW: a scan running
            # to 540 V at 4 mA would otherwise force mA on a 0-35 V window
            # whose visible currents are tens of uA, squashing every curve
            # onto the axis floor
            xw = _lims(xlims, "x")
            vis = np.ones(len(v), bool)
            if xw.get("left") is not None:
                vis &= v >= xw["left"]
            if xw.get("right") is not None:
                vis &= v <= xw["right"]
            if vis.any():
                col_imax = max(col_imax, float(np.nanmax(i[vis])))
            traces.append((s_idx, s, v, vset, i, gk))

        # one unit for the whole column, shared by every scan drawn in it
        scale, uname = ((_pick_current_unit(col_imax)) if unit == "auto"
                        else (1.0 / _mfactor(unit), unit))

        # --- pass 2: draw ---
        for s_idx, s, v, vset, i, gk in traces:
            color = _scan_color(s_idx)

            use_set = (deriv_x == "v_set" or
                       (deriv_x == "auto" and np.isfinite(vset).all()))
            grid_x = vset if use_set else None
            lr = (gk["low_range"].to_numpy()
                  if (mask_range_switch and gk is not None) else None)

            k = k_factor(v, i, v_grid=grid_x, smooth_v=smooth_v, low_range=lr)

            if show_range_switch and gk is not None:
                for vsw in range_switch_voltages(gk):
                    for a in (ax_iv, ax_k):
                        a.axvline(vsw, color=color, ls="-", lw=1.2, alpha=0.35, zorder=0)
            ax_iv.plot(v, i / scale, marker=".", ls="-", color=color, alpha=ALPHA_LINE)
            ax_k.plot(v, k, marker=".", ls="-", color=color, alpha=ALPHA_LINE)

            # breakdown lives above the finely-stepped region: on low-fluence
            # scans the gain-layer peak itself can exceed k_breakdown (the
            # March 3e14 scan's V_gl peak tops 8), so the crossing hunt starts
            # where the fine region ends. Scans with no fine region (quick
            # scans) search everywhere, as before.
            span_src = vset if np.isfinite(vset).any() else v
            f_lo, f_hi = fine_step_span(span_src, max_step=0.5)
            bd_sel = v > f_hi if np.isfinite(f_hi) else np.ones(len(v), bool)
            v_bd = _first_crossing(v[bd_sel], k[bd_sel], k_breakdown,
                                   i=i[bd_sel])
            if np.isfinite(v_bd):
                ax_k.axvline(v_bd, color=color, ls="--", alpha=0.6)
                ax_iv.axvline(v_bd, color=color, ls="--", alpha=0.6)

            v_gl = np.nan
            if vgl_search is not None:
                if isinstance(vgl_search, str) and vgl_search == "auto":
                    # legacy files carry no demand voltage: v_set is all-NaN,
                    # which silently empties the fine-step span and NaNs every
                    # V_gl. Fall back to the measured (binned) voltages, which
                    # step exactly as the sweep did.
                    span_src = vset if np.isfinite(vset).any() else v
                    lo, hi = fine_step_span(span_src, max_step=0.5)
                    lo = max(lo, float(s.get("min_v", min_v))) if np.isfinite(lo) else np.nan
                    if not np.isfinite(hi) or hi - lo < 5.0:
                        lo = hi = np.nan     # too short a fine region to trust
                else:
                    lo, hi = vgl_search
            if vgl_search is not None and np.isfinite(lo) and np.isfinite(hi):
                sel = (v >= lo) & (v <= hi) & np.isfinite(k)
                if sel.sum() >= 3:
                    # a real peak or nothing -- no argmax fallback, which would
                    # always hand back a number even for a flat, featureless k
                    pk, _ = find_peaks(k[sel], prominence=vgl_prominence)
                    if len(pk):
                        v_gl = v[sel][pk[np.argmax(k[sel][pk])]]
                        for a in (ax_k, ax_iv):
                            a.axvline(v_gl, color=color, ls=":", lw=2.5, alpha=0.9)
                    elif verbose:
                        print(f"    ch{ch}: no k peak with prominence >= "
                              f"{vgl_prominence:g} in {lo:g}-{hi:g} V")

            if ax_d is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    li = np.log10(np.where(i > 0, i, np.nan))
                x = grid_x if grid_x is not None else v
                dlog = np.gradient(_smooth_volts(x, li, smooth_v), x)
                ax_d.plot(v, _smooth_volts(x, dlog, smooth_v), marker=".", ls="-",
                          color=color, alpha=ALPHA_LINE)

            results.append({"scan": s["label"], "channel": ch,
                            "device": _ch_label(channel_names, ch),
                            "V_gl [V]": v_gl,
                            f"V @ k={k_breakdown:g} [V]": v_bd})

        ax_iv.set_yscale(scale_iv)
        if scale_iv == "linear":
            ax_iv.ticklabel_format(axis="y", useOffset=False, style="plain")
        ax_iv.set_ylabel(f"|I| [{uname}]")
        _style_axes(ax_iv, cms=(col == 0))
        _apply_lims(ax_iv, xlims, ylims_iv)
        # device name inside the panel, so the title strip is free for the CMS
        # header on the first column and the run title on the last
        _panel_tag(ax_iv, _ch_label(channel_names, ch), loc="upper left")
        _grid_fonts(ax_iv)

        ax_k.axhline(k_breakdown, color="red", ls="-", lw=2, alpha=0.8)
        ax_k.set_ylabel("k = (V/I)(dI/dV)")
        _style_axes(ax_k, xlabel=None if show_dlogi else "|V| [V]")
        _apply_lims(ax_k, xlims, ylims_k)
        _grid_fonts(ax_k)

        if ax_d is not None:
            ax_d.set_ylabel(r"d(log$_{10}$|I|) / d|V|")
            _style_axes(ax_d, xlabel="|V| [V]")
            _apply_lims(ax_d, xlims, ylims_dlogi)
            _grid_fonts(ax_d)

    if title:
        grid[0][-1].set_title(title, loc="right", size=FS_TITLE)
    fig.tight_layout()
    # legends after layout: the placer must score the final axes geometry
    if len(scans) > 1:
        handles = [mlines.Line2D([], [], color=_scan_color(i), ls=_scan_dash(i),
                                 lw=LW_LEGEND_KEY, label=s["label"])
                   for i, s in enumerate(scans)]
        if len(scans) <= 2:
            _place_legend(grid[1][0], handles, title="Conditions",
                          fontsize=FS_GRID_LEG)
        else:
            # k-factor panels are dense everywhere; >2 conditions go to a
            # caption-style strip under the grid, matching plot_iv's grid
            fig.legend(handles=handles, title="Conditions",
                       fontsize=FS_GRID_LEG, title_fontsize=FS_GRID_LEG,
                       loc="upper center", bbox_to_anchor=(0.5, 0.0),
                       ncols=2, frameon=True)
    if output_fig:
        plt.savefig(output_fig, dpi=DPI_SAVE, bbox_inches="tight")
        print(f"Saved plot to: {output_fig}")
    plt.show()

    # returned, not printed: the cell renders it once as a table. This is the
    # only genuinely new information the function produces -- the curves
    # themselves are already on screen.
    out = pd.DataFrame(results).drop_duplicates(subset=["scan", "channel"])
    if out.empty:
        return out
    if out["V_gl [V]"].isna().all():
        out = out.drop(columns=["V_gl [V]"])
    # Evolution view: rows follow the order the scans were listed (put them in
    # campaign order and each column reads as one board's history top-to-
    # bottom); one column per device.
    # Evolution view: one row per scan in listed order (list them oldest-first
    # and each device block reads top-to-bottom as that board's history), one
    # column block per device. Built with set_index/unstack rather than
    # pivot_table: the latter either drops all-NaN scans (a fine scan too short
    # to reach V_gl silently vanishes, reading as "not analysed") or, with
    # dropna=False, manufactures the full scan x device x value cartesian
    # product.
    value_cols = [c for c in out.columns if c not in ("scan", "channel", "device")]
    devs = list(dict.fromkeys(out["device"]))
    labels = [s["label"] for s in scans if s["label"] in set(out["scan"])]
    wide = (out.set_index(["scan", "device"])[value_cols]
               .unstack("device")
               .swaplevel(axis=1))
    wide = wide.reindex(index=labels)
    wide = wide.reindex(columns=pd.MultiIndex.from_product([devs, value_cols]))
    wide.index.name = "scan"
    return wide.round(1)
