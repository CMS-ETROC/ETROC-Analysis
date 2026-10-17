"""Names and places: combos, roles, pixel triples, run directories, and the ranking of track files.

A step-10 track file is one pixel triple of one combo, named after its pixels in the pipeline's
file order, e.g. track_d-R14C1_r-R14C1_e-R14C1.parquet (dut, ref, extra). Keys are built on the
board id where roles can move between runs.
"""
import glob
import os
import re
import time

from .campaigns import CAMPAIGNS, NAME_ORDER, ROLE_INIT


def log(msg):
    print("[tt %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def ensure_dirs(root):
    """Make the output folders under `root`, the write root (campaigns.write_root())."""
    for d in ("bank", "tier0", "cells", "toys", "quote", "figs", "logs", "tables"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    return root


# --------------------------------------------------------------- naming / parsing
def roles_of(combo):
    """'extra0-ref1-dut2' -> ['extra','ref','dut'] (board order).

    Vendored unchanged from roles_of in tv_core.py, the module of the study's TWC verification
    (twcverify/), which is not in this repository; copied 2026-10-18.
    """
    out = []
    for tok in combo.split('-'):
        out.append(''.join(ch for ch in tok if not ch.isdigit()))
    return out


def boards_of(combo):
    """'extra0-ref1-dut2' -> {'extra':0,'ref':1,'dut':2}.  Keying is on board id."""
    out = {}
    for tok in combo.split("-"):
        role = "".join(ch for ch in tok if not ch.isdigit())
        num = "".join(ch for ch in tok if ch.isdigit())
        out[role] = int(num) if num else -1
    return out


def parse_pixels(path):
    """track_d-R14C1_r-R14C1_e-R14C1.parquet -> {'dut':(14,1),'ref':(14,1),'extra':(14,1)}."""
    base = os.path.basename(path)
    out = {}
    for m in re.finditer(r"([tdre])-R(\d+)C(\d+)", base):
        out[ROLE_INIT[m.group(1)]] = (int(m.group(2)), int(m.group(3)))
    return out


def triple_key(pix):
    """Canonical, role-sorted pixel key: 'dut:R14C1|extra:R14C1|ref:R14C1'."""
    return "|".join("%s:R%dC%d" % (r, pix[r][0], pix[r][1]) for r in sorted(pix))


def board_pixel_key(pix, boards):
    """Board-id keyed variant (roles are reassigned between DESY runs)."""
    return "|".join("b%d:R%dC%d" % (boards[r], pix[r][0], pix[r][1]) for r in sorted(pix, key=lambda x: boards[x]))


def track_name(pix, roles):
    """Inverse of parse_pixels for the roles given, in the pipeline's file order."""
    inv = {v: k for k, v in ROLE_INIT.items()}
    toks = ["%s-R%dC%d" % (inv[r], pix[r][0], pix[r][1]) for r in NAME_ORDER if r in roles]
    return "track_" + "_".join(toks)


def run_dir(campaign, run, root=None):
    base = root or CAMPAIGNS[campaign]
    return os.path.join(base, "tracks_%s" % run)


def combos_in(campaign, run, root=None):
    d = run_dir(campaign, run, root)
    return sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x, "time")))


def sibling_combo(campaign, run, combo, drop_role, root=None):
    """The leave-one-out combo of the SAME run that omits `drop_role` and keeps the
    other two roles of `combo` (the 4-board telescope's fourth board joins it)."""
    keep = set(roles_of(combo)) - {drop_role}
    for c in combos_in(campaign, run, root):
        rs = set(roles_of(c))
        if drop_role not in rs and keep <= rs:
            return c
    return None


def triple_files(campaign, run, combo, root=None):
    return sorted(glob.glob(os.path.join(run_dir(campaign, run, root), combo, "time", "track_*.parquet")))


def top_triples(campaign, run, combo, ntrip, root=None, min_bytes=0):
    """Largest-first (file size as a proxy for event count)."""
    fs = [p for p in triple_files(campaign, run, combo, root) if os.path.getsize(p) >= min_bytes]
    fs.sort(key=os.path.getsize, reverse=True)
    return fs[:ntrip]


def pixel_tracks(campaign, run, combo, role, root=None, min_bytes=0):
    """Rank ONE board's pixels by population, each with its own most populous track.

    The source-picking rule of the pixel -> board protocols (SPEC.md 3b): *the most
    populous pixel, its coefficients from its most populous track*.  The step-10 tree
    stores one parquet per pixel TRIPLE, so there is no per-pixel file to weigh: a
    board-pixel's population is the total over every track that carries that pixel on
    that board, in the same file-size proxy for event count that `top_triples`
    uses.  The two levels are therefore

      * `pop`  - sum of the sizes of every track carrying this pixel on `role`, which
                 is what "the most populous pixel" ranks on;
      * `best` - the single largest of those tracks, which is the track the TWC
                 coefficients are then solved on ("its most populous track").

    They can and do disagree with `top_triples`: the largest single track need not
    carry the pixel with the largest total, which is why this is a different rule.

    Returns a list of dicts, most populous pixel first, deterministically tie-broken
    on (total, best, pixel) so a re-run picks the same source:
        {"pix": (row, col), "pop": int, "ntrk": int, "best": path, "best_pop": int}
    """
    agg = {}
    for p in triple_files(campaign, run, combo, root):
        sz = os.path.getsize(p)
        if sz < min_bytes:
            continue
        px = parse_pixels(p).get(role)
        if px is None:                       # the combo does not carry this role
            continue
        e = agg.setdefault(px, dict(pix=px, pop=0, ntrk=0, best=None, best_pop=-1))
        e["pop"] += sz
        e["ntrk"] += 1
        if sz > e["best_pop"]:
            e["best_pop"], e["best"] = sz, p
    return sorted(agg.values(), key=lambda e: (-e["pop"], -e["best_pop"], e["pix"]))
