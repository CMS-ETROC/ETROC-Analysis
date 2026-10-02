"""qinj_plots.py -- the figures of plot_qinj.py, drawn from its
qinj_pixels_ns.csv rows: for each lot and stage, per injected pixel, the
CAL, TOT and TOA of its dies and the TOA against the reference pixel of the
same die, beside what the H-trees of the design predict for it."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

from plot_lots import slug  # noqa: E402
from plot_qinj import REFERENCE_PIXEL  # noqa: E402
from position_compare import months_tested  # noqa: E402
from power_plots import WHISKERS  # noqa: E402
from stage_compare import STAGE_ORDER, STAGE_TEXT  # noqa: E402
from wafer_plots import add_note  # noqa: E402

# ETROC2 Reference Manual rev 0.6, Table 20: the TDCs of the right half of
# the matrix (column 0 is the rightmost, in the chip's own frame) run on the
# digital supply, those of the left half on the discriminator supply of the
# analog group
HALF_COLOURS = {False: "#1f77b4", True: "#ff7f0e"}
HALF_TEXT = {False: "pixel columns 0-7: TDC on the digital supply",
             True: "pixel columns 8-15: TDC on the discriminator (analog) supply"}
PANELS = (("cal_mean", "CAL (code)"), ("tot_ns", "TOT (ns)"), ("toa_ns", "TOA (ns)"),
          ("dtoa_ns", f"TOA minus TOA of {REFERENCE_PIXEL}, same die (ns)"))
PREDICTION_TEXT = (f"H-tree prediction of the TOA minus that of the reference pixel {REFERENCE_PIXEL}: injection-pulse "
                   "minus reference-strobe delay, ETROC2 manual rev 0.6 Fig. 28")


def fig_qinj(rows, title):
    """Per injected pixel, boxes of CAL, TOT, TOA and dtoa over the dies,
    the H-tree prediction beside dtoa."""
    pixels = sorted(set(zip(rows["pix_row"], rows["pix_col"])))
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8), layout="constrained")
    for ax, (column, label) in zip(axes.flat, PANELS):
        for k, (r, c) in enumerate(pixels):
            values = rows.loc[(rows["pix_row"] == r) & (rows["pix_col"] == c), column].dropna()
            if values.empty or (column == "dtoa_ns" and (r, c) == REFERENCE_PIXEL):
                continue
            ax.boxplot([values], positions=[k], widths=0.55, whis=WHISKERS, patch_artist=True, manage_ticks=False,
                       boxprops={"facecolor": HALF_COLOURS[c >= 8], "edgecolor": "#333333", "lw": 0.6},
                       medianprops={"color": "black", "lw": 1.0}, whiskerprops={"lw": 0.6}, capprops={"lw": 0.6},
                       flierprops={"marker": ".", "markersize": 3, "markeredgecolor": "#555555"})
        ax.set_xticks(range(len(pixels)), [f"({r},{c})" for r, c in pixels], fontsize=10, rotation=30)
        ax.set_xlim(-0.6, len(pixels) - 0.4)
        ax.set_ylabel(label, fontsize=11)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", color="#e5e5e5", lw=0.5)
        ax.set_axisbelow(True)
    low, high = axes[0, 0].get_ylim()
    axes[0, 0].set_ylim(low - 0.08 * (high - low), high)
    counts = blended_transform_factory(axes[0, 0].transData, axes[0, 0].transAxes)
    for k, (r, c) in enumerate(pixels):
        n = rows.loc[(rows["pix_row"] == r) & (rows["pix_col"] == c), "cal_mean"].notna().sum()
        axes[0, 0].text(k, 0.02, n, transform=counts, ha="center", va="bottom", fontsize=8, color="#555555")
    predicted = rows.groupby(["pix_row", "pix_col"])["htree_dtoa_ns"].first()
    axes[1, 1].scatter([k + 0.42 for k, p in enumerate(pixels) if p != REFERENCE_PIXEL],
                       [predicted[p] for p in pixels if p != REFERENCE_PIXEL],
                       marker="<", s=40, color="#d62728", zorder=3)
    axes[1, 1].axhline(0, color="#999999", lw=0.6)
    if REFERENCE_PIXEL in pixels:
        axes[1, 1].text(pixels.index(REFERENCE_PIXEL), 0, "reference pixel", ha="center", va="bottom", fontsize=8,
                        color="#555555")
    for ax in axes[1]:
        ax.set_xlabel("injected pixel (row, column)", fontsize=11)
    handles = [Patch(facecolor=HALF_COLOURS[h], edgecolor="#333333", lw=0.6, label=HALF_TEXT[h]) for h in HALF_COLOURS]
    handles.append(Line2D([], [], marker="<", ls="none", color="#d62728", markersize=7,
                          label=PREDICTION_TEXT))
    fig.legend(handles=handles, loc="outside lower center", ncol=1, fontsize=10, frameon=False,
               title="boxes: quartiles and median; whiskers: 2nd to 98th percentile; small numbers: the dies\n"
                     "TOA = 12.5 ns - code x bin, TOT = (2 code - floor(code / 32)) x bin, bin = 3.125 ns / CAL",
               title_fontsize=10)
    fig.suptitle(title, fontsize=12.5)
    return fig


def plot_qinj(out_dir, qinj, note="", dpi=130):
    """Write <label>_<stage>_qinj.png into out_dir for each lot and stage;
    returns their paths."""
    written = []
    for lot in dict.fromkeys(qinj["lot"]):
        for stage in STAGE_ORDER:
            rows = qinj[(qinj["lot"] == lot) & (qinj["stage"] == stage)]
            if rows.empty:
                continue
            wafers = list(dict.fromkeys(rows["wafer"]))
            title = (f"{lot}, {STAGE_TEXT[stage]}, {months_tested(rows)}: charge injection on\n"
                     f"{rows['die'].groupby(rows['wafer']).nunique().sum()} PASSED valid dies of "
                     f"{len(wafers)} wafers ({', '.join(wafers)})")
            fig = fig_qinj(rows, title)
            if note:
                add_note(fig, note, wrap=True)
            path = out_dir / f"{slug(lot)}_{stage}_qinj.png"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            written.append(path)
    return written
