# Microscope images of wafers after UBM

Tools for the Olympus cellSens microscope images (`.vsi` + `.ets`) of ETROC wafers after UBM:

- a reader for the `.ets` tile files (`ets.py`);
- a pipeline that registers every die of a wafer and compares it with the dies that passed wafer probing both
  before and after UBM;
- bump checks on the gel-pack images of diced chips (`gelpack.py`, see [Gel-pack images](#gel-pack-images-of-diced-chips)).

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

## Gel-pack images of diced chips

The gel-pack scans show diced chips face up, pads at the bottom, 3 x 3 chips per pack (or a single chip).
- The packs so far all hold chips of wafer N62C72 14A2: `Accepted_N62C72_14_775um` and `_Num2` to `_Num7`,
  `BadChips_111_N62C72_14` and `Missimg_002_N62C72_14`. The numbers are inspection codes: 111 is a power short or
  an I2C error, 002 is bad bumps.
- The `.ets` is in `_<name>_/stack10000/` (`stack1/` for `Missimg_002`), about 13 GB for a pack.
- Level 0 is about 1.25 µm per pixel. `gelpack.py` assumes that scale: the 1.3 mm bump pitch is 64.2 px at level 4
  (`PITCH` and the chip size limits at the top of the script).

```bash
python $M/gelpack.py bumps <frame_t_0.ets>              # level 4, a few minutes per pack
python $M/gelpack.py domes <frame_t_0.ets> [chip ...]   # level 1, needs bumps; 1-5 minutes per pack via the EOS mount
```

Outputs go to `./out/gelpack_<name>/`. Chips are numbered from 0 along the rows of the scan, top to bottom and left
to right, as labelled in `bumps_L4.png`. Bump sites (j, i) count rows from the top and columns from the left, both
from 0, as the chip lies in the scan, and row 16 is the bump row along the pad row.

1. **`bumps`** finds the chips, fits each chip's 16 x 16 bump grid (with its small rotation) and the bump row along
   the pad row, and scores every site against the pack's median bump. It writes `chips.csv`, `bumps.csv` and
   `bumps_L4.png`, where empty sites are circled and chips without solder bumps are boxed in red.
   - An empty site scores below 0.3. On the blurrier packs single bumps score as low as 0.31, so the cut has almost
     no margin there: look at a flagged site before calling it empty.
   - The score alone does not tell a bump from a bare pad. The `highlight` column does: the brightest pixel at the
     site above the local mean, which only a solder dome gives. A chip whose median highlight is below 0.6 of the
     pack median is flagged `unbumped`. With fewer than 3 chips in the scan the flag is left blank: compare the
     highlight with a pack of bumped chips scanned the same way (84 to 96 on the 2026 scans).
   - `x4, y4` in `bumps.csv` are level-4 pixels of the whole image: the fitted site for rows 0 to 15, the best match
     for row 16. `bumps_L4.png` is cropped to the scanned area, so its pixels are offset from these.
2. **`domes`** cuts a patch of every bump at level 1, re-centres it on the chip's median bump, and measures the
   match (`ncc`) and the colour of the bump top (`b_minus_r`). A top that is redder than the chip median by more
   than 4 robust sigma is flagged `discoloured`. A site with `ncc` below 0.3 has no bump to judge and is listed as
   "no bump found" instead. `score4` repeats the `bumps` score; `dx, dy` are the level-1 pixels from the level-4
   site to the dome, a few pixels on every site plus any real displacement. `sheet_chip<c>.png` shows every bump of
   the chip in place, row 16 at the bottom: the quickest way to look at a chip. It does not mark the flagged ones.

Results on 14A2 (October 2026):
- Chip 6 of `BadChips_111` (bottom left in the scan) has no solder bumps: every site is a flat pad (median
  highlight 45, against 87 to 95 on the other bad chips and 84 to 96 on all 63 accepted ones).
- `Missimg_002` has 12 discoloured bump tops, in its lower left (rows 12 to 16, columns 0 to 8); on the sheet these
  tops are brown or rainbow-coloured, some striated. None of the bad chips has a discoloured top, and of the
  accepted chips checked at level 1 (`_Num7` chip 5) none either. The cut sits in a tail: a few more Missimg tops
  look off by eye, and a top or two at the cut can come and go with small changes to the method.
- Apart from chip 6, the bad chips look like the accepted ones at both levels.

Traps:
- **Stitching.** In `Missimg_002` a block of bumps sits about 175 µm off the grid along a field-of-view seam, so
  `bumps` reports 11 empty sites there (rows 13 to 15, columns 7 to 11) that hold displaced bumps. `domes` finds
  them, at the edge of its search window, and two of them, (15, 7) and (15, 8), are among the discoloured tops. Look
  at the sheet or a crop before believing an empty site.
- **Scan quality.** The older accepted packs (no suffix and `_Num2` to `_Num5`) are blurrier: bump scores around
  0.65, against 0.83 for the packs scanned on 2026-09-28. Compare chips within one pack, or with packs scanned the
  same day.

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

- `test_ets.py` writes small synthetic `.ets` files and reads them back. It needs only numpy.
- `test_gelpack.py` runs the analysis of both gel-pack steps on synthetic images (not the file reading and writing):
  an empty site, a chip without solder bumps, a single chip, a chip cut by the edge of the scan, the chip order, a
  discoloured bump top and a site with no bump. It needs numpy, scipy and Pillow, as the tools do.
