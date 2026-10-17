"""The command line of the TWC transfer study: python -m twc_transfer <subcommand>.

  index   the target triples of a (campaign, run[, combo]) with their event counts
          -> <write root>/bank/index_<campaign>_<run>.parquet
  bank    (alias fit) the coefficient bank of a run (SPEC.md 2)
          -> <write root>/bank/coeffs_<campaign>_<run>.parquet
  tier0   the analytic Tier-0 screen over one bank file (SPEC.md 4)
          -> <write root>/tier0/transfer_<tag>.parquet
  cell    (alias transfer) one transfer cell; its arguments are parsed, so that job lists can
          be checked, but this version does not run it
  jobs    a job list (jobs.py): with --dry-run, every command expanded and parsed

Argument names, defaults and types are those of the study's own command line, so the
commands of its job lists parse unchanged. `--out` overrides an output file name.

Every subcommand that writes needs the write root, ETROC_TWC_OUT (campaigns.write_root()):
there is no default, and the read root and the step-10 data roots are refused, so nothing is
ever written into the study's own tree or among the track files. Only those subcommands make
the output folders under it. `tier0 --bank` is read from the read root when it is a relative
path. python -m twc_transfer (__main__.py) also pins the thread pools to one thread and fixes
PYTHONHASHSEED=0; main() called from Python does neither.
"""
import argparse
import os
import time

from . import campaigns

WRITING = ("index", "bank", "fit", "tier0")
CELL_MODES = ["a", "b", "c", "cb", "d", "e", "f", "inrun"]


class _RaisingParser(argparse.ArgumentParser):
    """An ArgumentParser that raises ValueError instead of printing usage and exiting."""

    def error(self, message):
        raise ValueError(message)


def build_parser(raising=False):
    cls = _RaisingParser if raising else argparse.ArgumentParser
    p = cls(prog="python -m twc_transfer", description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q, run_required=True):
        q.add_argument("--campaign", default="mar")
        q.add_argument("--run", required=run_required)
        q.add_argument("--combo", default=None)
        q.add_argument("--root", default=None, help="override the campaign's step-10 root")
        q.add_argument("--cap", type=int, default=campaigns.N_CAP)
        q.add_argument("--neighbor", default="none")
        q.add_argument("--neighbor-logic", default="OR", dest="neighbor_logic")
        q.add_argument("--out", default=None)

    q = sub.add_parser("index", help="the target triples of a run, with event counts")
    common(q)
    q.add_argument("--ntrip", type=int, default=100000)
    q.set_defaults(func=cmd_index)
    q.set_defaults(combo=None)

    for nm in ("bank", "fit"):
        q = sub.add_parser(nm, help="extract coefficients into the bank")
        common(q)
        q.add_argument("--ntrip", type=int, default=40)
        q.add_argument("--nmin", type=int, default=600)
        q.add_argument("--folds", default="parity", choices=["full", "parity", "block", "both"])
        q.add_argument("--guards", default="full")
        q.add_argument("--placebo", action="store_true")
        q.add_argument("--angle", type=float, default=float("nan"))
        q.add_argument("--telescope", default="")
        q.add_argument("--hv", type=float, default=float("nan"))
        q.add_argument("--temp", type=float, default=float("nan"))
        q.add_argument("--chunk", default=None, help="k/N: every N-th file of each combo")
        q.add_argument("--append", action="store_true")
        q.set_defaults(func=cmd_bank)

    for nm in ("cell", "transfer"):
        q = sub.add_parser(nm, help="one transfer cell (SPEC.md 3); parsed, not run here")
        common(q)
        q.add_argument("--cell", required=True, help="cell id (output file name)")
        q.add_argument("--mode", required=True, choices=CELL_MODES)
        q.add_argument("--src-campaign", dest="src_campaign", default=None)
        q.add_argument("--src-run", dest="src_run", default=None)
        q.add_argument("--src-combo", dest="src_combo", default=None)
        q.add_argument("--src-triple", dest="src_triple", default=None)
        q.add_argument("--src-root", dest="src_root", default=None)
        q.add_argument("--src-bank", dest="src_bank", default=None,
                       help="bank parquet to read the source coefficients from")
        q.add_argument("--role-map", dest="role_map", default=None,
                       help="'tgt_role:src_role,...'")
        q.add_argument("--src-parity-contrast", dest="src_parity_contrast",
                       action="store_true")
        q.add_argument("--split", default="block", choices=["block", "parity"])
        q.add_argument("--xfer-board", dest="xfer_board", default=None)
        q.add_argument("--src-role", dest="src_role", default=None)
        q.add_argument("--src-pick", dest="src_pick", default="auto",
                       choices=["auto", "pixpop", "trkpop"])
        q.add_argument("--src-cap", dest="src_cap", type=int, default=0)
        q.add_argument("--arms", default="core", choices=["core", "ext"])
        q.add_argument("--xfer-lin", dest="xfer_lin_arm", action="store_true")
        q.add_argument("--lpo", action="store_true", help="build the leave-pair-out arms")
        q.add_argument("--lpo-fallback", dest="lpo_fallback", default="sibling",
                       choices=["sibling", "own"])
        q.add_argument("--ntrip", default="12", help="number of target triples, or 'all'")
        q.add_argument("--triples", default=None)
        q.add_argument("--nmin", type=int, default=600)
        q.add_argument("--nmin-src", dest="nmin_src", type=int, default=1000)
        q.add_argument("--nmin-fold", dest="nmin_fold", type=int, default=250)
        q.add_argument("--src-k", dest="src_k", type=int, default=5)
        q.add_argument("--pool-ntrip", dest="pool_ntrip", type=int, default=40)
        q.add_argument("--pool-cap", dest="pool_cap", type=int, default=1500)
        q.add_argument("--pool-lopo-scope", dest="pool_lopo_scope", default="board",
                       choices=["board", "triple"])
        q.add_argument("--pool-lpo-ntrip", dest="pool_lpo_ntrip", type=int, default=6)
        q.add_argument("--chunk", default=None)
        q.add_argument("--flush", type=int, default=2)
        q.add_argument("--fast", action="store_true")
        q.add_argument("--overwrite", action="store_true")
        q.add_argument("--dry-run", dest="dry_run", action="store_true")
        q.add_argument("--max-triples", "--cap-triples", dest="max_triples", type=int,
                       default=0)
        q.add_argument("--timebudget", "--cap-seconds", type=float, default=0.0)
        q.set_defaults(func=cmd_cell)

    q = sub.add_parser("tier0", help="the Tier-0 screen over one bank file")
    q.add_argument("--bank", required=True,
                   help="bank parquet; a relative path is taken in the read root")
    q.add_argument("--tag", default="all")
    q.add_argument("--max-src", dest="max_src", type=int, default=0)
    q.add_argument("--out", default=None)
    q.set_defaults(func=cmd_tier0)

    q = sub.add_parser("jobs", help="check a job list")
    q.add_argument("--list", required=True, dest="job_list",
                   help="a job list file, or the name of one in jobs/")
    q.add_argument("--job", action="append", default=None,
                   help="only this job id (repeatable)")
    q.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="print every command expanded and parse it (required)")
    q.set_defaults(func=cmd_jobs)
    return p


def parse_command(tokens):
    """The namespace of one command (a token list), ValueError when the parser refuses it."""
    return build_parser(raising=True).parse_args(tokens)


def _write_root():
    try:
        return campaigns.write_root()
    except RuntimeError as e:
        raise SystemExit("twc_transfer: %s" % e)


# ---------------------------------------------------------------------------- index / bank
def cmd_index(a, root):
    import pandas as pd
    import pyarrow.parquet as pq
    from . import naming
    rows = []
    for combo in (a.combo.split(",") if a.combo else naming.combos_in(a.campaign, a.run, a.root)):
        for p in naming.top_triples(a.campaign, a.run, combo, a.ntrip, a.root):
            n = pq.ParquetFile(p).metadata.num_rows
            rows.append(dict(campaign=a.campaign, run=a.run, combo=combo, path=p,
                             name=os.path.basename(p)[:-8], nevt=int(n),
                             triple=naming.triple_key(naming.parse_pixels(p)),
                             bytes=os.path.getsize(p)))
    df = pd.DataFrame(rows).sort_values("nevt", ascending=False)
    dest = a.out or os.path.join(root, "bank", "index_%s_%s.parquet" % (a.campaign, a.run))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    df.to_parquet(dest, index=False)
    print(df.head(15).to_string(index=False))
    print("%d triples -> %s" % (len(df), dest))


def cmd_bank(a, root):
    """`fit`: extract per-triple coefficients into the bank (SPEC 2)."""
    import pandas as pd
    from . import bank, events, naming
    dest = a.out or os.path.join(root, "bank", "coeffs_%s_%s.parquet" % (a.campaign, a.run))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    combos = a.combo.split(",") if a.combo else naming.combos_in(a.campaign, a.run, a.root)
    guards = tuple(a.guards.split(","))
    rows = []
    t0 = time.time()
    for combo in combos:
        files = naming.top_triples(a.campaign, a.run, combo, a.ntrip, a.root)
        if a.chunk:
            k, n = (int(x) for x in a.chunk.split("/"))
            files = files[k - 1::n]
        for i, p in enumerate(files):
            tri = events.Triple.from_file(p, a.campaign, a.run, combo, cap=a.cap,
                                          neighbor=a.neighbor, neighbor_logic=a.neighbor_logic)
            if tri.n < a.nmin:
                continue
            rows += bank.bank_rows_for_triple(tri, folds=a.folds, guards=guards,
                                              placebo=a.placebo, root=a.root,
                                              angle_deg=a.angle, telescope=a.telescope,
                                              neighbor=a.neighbor,
                                              neighbor_logic=a.neighbor_logic,
                                              hv=a.hv, temp=a.temp)
            if (i + 1) % 5 == 0:
                naming.log("%s %s: %d/%d triples, %d rows, %.0fs"
                           % (a.run, combo, i + 1, len(files), len(rows), time.time() - t0))
    df = pd.DataFrame(rows)
    if a.append and os.path.exists(dest):
        df = pd.concat([pd.read_parquet(dest), df], ignore_index=True)
    df.to_parquet(dest, index=False)
    print("bank: %d rows (%d triples) -> %s  [%.0fs]"
          % (len(df), df["src_file"].nunique() if len(df) else 0, dest, time.time() - t0))


# ----------------------------------------------------------------------------------- tier0
def cmd_tier0(a, root):
    """GMM-free analytic screen over one bank file (tier0.bank_screen)."""
    import numpy as np
    import pandas as pd
    from . import tier0
    path = a.bank if os.path.isabs(a.bank) else os.path.join(campaigns.transfer_root(), a.bank)
    df = tier0.bank_screen(pd.read_parquet(path), max_src=a.max_src)
    dest = a.out or os.path.join(root, "tier0", "transfer_%s.parquet" % a.tag)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    df.to_parquet(dest, index=False)
    print("tier0: %d rows -> %s" % (len(df), dest))
    if len(df):
        bb = df[df.kind == "board"]
        print(bb.groupby("name")[["P", "P_null", "P_exc"]].median().round(2).to_string())
        print("EXTRAP fraction: %.2f ; median coverage C = %.3f"
              % (float(np.nanmean(bb["extrap"])), float(np.nanmedian(bb["cov_C"]))))
        print("P_exc = 0 (i.e. within the split-half floor) on %.0f %% of board rows"
              % (100.0 * float(np.nanmean(bb["P_exc"] <= 0))))


# ------------------------------------------------------------------------------ cell / jobs
def cmd_cell(a, root):
    raise SystemExit("twc_transfer cell: not available in this version of the package; "
                     "`jobs --dry-run` parses cell commands without running them")


def cmd_jobs(a, root):
    from . import jobs
    if not a.dry_run:
        raise SystemExit("twc_transfer jobs: only --dry-run is available")
    njobs, failed = jobs.dry_run(jobs.list_path(a.job_list), campaigns.CAMPAIGNS,
                                 campaigns.transfer_root(), _write_root(), parse_command,
                                 only=a.job)
    if failed:
        raise SystemExit("twc_transfer jobs: %d of %d commands failed" % (len(failed), njobs))


def main(argv=None):
    a = build_parser().parse_args(argv)
    root = None
    if a.cmd in WRITING:
        from .naming import ensure_dirs
        root = ensure_dirs(_write_root())
    a.func(a, root)
    return 0
