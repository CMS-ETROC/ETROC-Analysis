# Microscope images of wafers after UBM

Tools for the Olympus cellSens microscope images (`.vsi` + `.ets`) of ETROC wafers after UBM:

- a reader for the `.ets` tile files (`ets.py`);
- a pipeline that registers every die of a wafer and compares it with the dies that passed wafer probing both
  before and after UBM.

They were written in September 2026 to look for a visible cause of the new post-UBM probe failures. They were run on
N62H30_02C7, N62C72_24C5 and N62C72_22D7. At about 4.2 µm per pixel they found no visible defect (bumps, pads, probe
marks) that sets the failing dies apart.

## The images

Each image is a small `.vsi` file plus a folder `_<name>_/stack10000/frame_t_0.ets` that holds the pixels. Only the
`.ets` is read. The wafer images are about 20 GB each, stored as `_UBM_<wafer>_/stack10000/frame_t_0.ets` in
`/eos/user/m/musafdar/HighResImagesDSF/`. That is a personal EOS area, so ask Murtaza Safdari for read access.

Throughout, `<wafer>` is the image name, lot and wafer together, such as `N62H30_02C7` (`$W` in the setup below).

An `.ets` is a pyramid of levels. Level 0 is full resolution: 72192 x 71168 px for a wafer, about 4.2 µm per pixel.
Each level after that is half the size of the one before. The pixels are raw 8-bit RGB tiles of 512 x 512, stored in
scan order rather than position order, so reading a region means random seeks.

## Setup (lxplus)

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_104d/x86_64-el9-gcc13-opt/setup.sh
export OMP_NUM_THREADS=1
M=<path to ETROC-Analysis>/WaferProbe/microscope
W=N62H30_02C7
mkdir -p /tmp/$USER/ubm/_UBM_${W}_/stack10000
xrdcp -s root://eosuser.cern.ch//eos/user/m/musafdar/HighResImagesDSF/_UBM_${W}_/stack10000/frame_t_0.ets \
    /tmp/$USER/ubm/_UBM_${W}_/stack10000/
export UBM_ROOT=/tmp/$USER/ubm
```

Copy the image to node-local disk before running a whole-wafer step. Random tile reads through the EOS mount run at
about 3.5 MB/s, against about 70 MB/s for sequential reads, and the copy takes about 3 minutes.
- `/tmp` belongs to one node, so run every step on the node that holds the copy, and delete the copy when you are
  done.
- For a look at a few dies, `UBM_ROOT=/eos/user/m/musafdar/HighResImagesDSF` works without the copy.
- On the shared lxplus nodes, run the heavy steps with `nice -n 19`, one at a time.

## Reading any `.ets` file

```bash
python $M/ets.py info  <frame_t_0.ets>                          # header and the pyramid levels
python $M/ets.py level <frame_t_0.ets> 4 wafer_L4.png           # one whole level as a PNG
python $M/ets.py level <frame_t_0.ets> 0 box.png 1000 2000 3000 3000   # x0 y0 x1 y1 in that level's pixels
```

```python
from ets import ETS                            # with $M on PYTHONPATH
img = ETS(path).read(level, x0, y0, x1, y1)   # uint8 array (y1 - y0, x1 - x0, 3)
```

- Missing tiles read as white (255).
- The wafer scans have one extra tile coordinate between (x, y) and the level. The gel-pack scans of diced chips (the
  AOI images) have two. The reader handles both, and `mid=` picks another plane if a file ever has one.

## Whole-wafer pipeline

Run the pipeline from a work directory outside the repository, calling the scripts by path as below.
- **Inputs shipped with the code** are found next to the scripts: `grids_L4.json`, `templates.npz` and
  `../wafer_map.csv`.
- **The die grades** come from the per-die table `lot_dies.csv`, written by `WaferProbe/plot_lots.py` (in its output
  folder, `lot_summary/` by default). That script comes with the wafer-probe plotting code of PR #5; its columns
  `wafer`, `die`, `grade_pre`, `grade_post` and `detail_post` are what `anomaly.py` reads.
- **Outputs** go to `./out/<wafer>_peri/`. `grid.py` is the exception: it updates `grids_L4.json` in the repository.
- **Reference die:** die 64 on every wafer so far.

0. **`python $M/grid.py <wafer>`**: fits the die lattice at level 4 and adds it to `grids_L4.json`.
   - Skip this step for wafers already in the file (02C7, 24C5 and 22D7).
   - The lattice phase is only known modulo one die pitch. Check a new fit with `$M/crop_die.py` on a few dies: the
     partial dies 57 and 58 sit in column 0, on the left. If the crops come out one die off, add or subtract `Py` to
     `phy` (or `Px` to `phx`) in `grids_L4.json`.
1. **`python $M/anomaly.py <wafer> peri 64 <lot_summary>/lot_dies.csv`**: registers every die onto the reference die
   and builds the per-pixel median and spread of the dies that passed before and after UBM. It writes:
   - `dies.csv`: per-die corners, grades and anomaly counts;
   - `blobs.csv`;
   - `median.npy` and `sigma.npy`;
   - a z-map PNG per die.

   `peri` is the top strip of the die at level 0 (pad row and first bump row). `full` is the whole die at level 1 and
   writes to `out/<wafer>_full/` instead. The run needs 0.9 GB (`peri`) or 1.9 GB (`full`) of space under `$TMPDIR`.
2. **`python $M/register.py <wafer> 64`**: registers every die again on its pad row, which does not repeat along the
   die.
   - Run it whenever step 1 prints outliers in its lattice fit. On 24C5 that fit failed, leaving 99 of 116 corners
     off by more than 20 px.
   - It rewrites `dies.csv` and keeps step 1's version as `dies_anomaly.csv`. On later runs it reads
     `dies_anomaly.csv`, so delete that file whenever you rerun step 1.
   - With `REG_OUT=<name>` it writes to that file instead of `dies.csv`, for a trial run.
3. **`python $M/sites.py <wafer> 64 auto`**: finds, on the reference die, the 124 pad-row octagons and the pixel bump
   domes, using the templates in `templates.npz`.
4. **`python $M/census.py <wafer> 64`**: measures every dome of every die and compares it with the same dome on the
   passing dies (match quality, brightness, colour, shift). With `DIES=64,65` it runs on a subset. Then run
   **`python $M/census_report.py out/<wafer>_peri <label>`**. It prints failing against passing dies and draws
   `census_map.png`.

Looking at single dies:

- `python $M/crop_die.py <wafer> <level> <margin> <die>...`: crops whole dies from the lattice fit, writing to
  `./out/`.
- `python $M/padends.py <wafer> <die>...`: both ends of the pad row at level 0.
- `python $M/padrow.py <wafer> <die>`: the octagon sequence along the pad row.
- `python $M/patches.py <wafer> peri 64 <n> <half> <die>...`: the strongest anomaly blobs next to the same spot on
  the reference die.

`python $M/templates.py` remakes `templates.npz` from die 64 of N62H30_02C7, overwriting the copy in the repository.
Rerun it only to change the templates. It needs that wafer's `out/N62H30_02C7_peri/dies.csv` (step 1) and the
`sites_pix_die064.csv` of `python $M/sites.py N62H30_02C7 64 pix`, which is how the saved 02C7 pixel sites were made
before the templates existed. Step 3 writes the same file, so run the `pix` mode right before `templates.py`.

## Frames and coordinates

- The image is the wafer seen from the front with the notch at the top. The chips appear rotated by 180 degrees (the
  laser mark reads upside down). Image die cell (i, j) is `wafer_map.csv` row i, column j, the same notch-up wafer
  view as the other WaferProbe plots.
- Die coordinates (u, v) are level-0 pixels from the die corner (X, Y), where X = x0 - sx + 400 and
  Y = y0 - sy + 60 from `dies.csv`.
- The pad row is fitted per die by `sites.py`. On die 64 of 02C7, pad n has its octagon at u = 15 + 34.16 (124 - n),
  a pitch of 143 µm, with pad 124 leftmost.
- The octagons alternate between a near row (v = 205) and a far row (v = 258).

## Traps

- **Mosaic seams.** The microscope stitches fields of view of about 2000 x 1450 px (level 0), each placed with its
  own error.
  - Up to about 15 px can be lost or duplicated at a seam.
  - Deep in a die, domes can sit up to about 25 px from where the pad-row registration puts them.
  - Every "defect" flagged in the 2026 study turned out to be a seam, so check a flag against the seams before
    believing it.
- **Registration.** The bump array repeats every 1.3 mm (310 px), so registering a die on the bumps alone can lock one
  period off. `anomaly.py` fits the whole lattice to avoid this, and `register.py` uses the pad row.
- **Image timing.** Some wafers were imaged before their post-UBM probing and some after. Check the order before
  reading probe marks into an image.

## Tests

```bash
cd WaferProbe/microscope && python -m unittest
```

The tests write small synthetic `.ets` files and read them back. They need only numpy.
