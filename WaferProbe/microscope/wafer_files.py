"""Where the tools find their inputs: the wafer images under UBM_ROOT, and the data files kept next to the code."""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GRIDS = os.path.join(HERE, "grids_L4.json")
TEMPLATES = os.path.join(HERE, "templates.npz")
WAFER_MAP = os.path.join(HERE, "..", "wafer_map.csv")


def image_path(wafer):
    """the .ets file of one wafer image, e.g. N62H30_02C7 -> $UBM_ROOT/_UBM_N62H30_02C7_/stack10000/frame_t_0.ets"""
    root = os.environ.get("UBM_ROOT")
    if not root:
        sys.exit("UBM_ROOT is not set: point it at the folder that holds _UBM_<wafer>_/stack10000/frame_t_0.ets")
    return os.path.join(root, "_UBM_%s_" % wafer, "stack10000", "frame_t_0.ets")


def wafer_map():
    """die number -> (map row, map column)"""
    with open(WAFER_MAP) as f:
        return {int(r["location_id"]): (int(r["row"]), int(r["col"])) for r in csv.DictReader(f)}


def grid_fit(wafer):
    """the level-4 grid fit of one wafer (grid.py): wafer circle cx, cy, R and die lattice Px, Py, phx, phy"""
    with open(GRIDS) as f:
        grids = json.load(f)
    if wafer not in grids:
        sys.exit("no grid fit for %s in grids_L4.json: run grid.py %s first" % (wafer, wafer))
    return grids[wafer]
