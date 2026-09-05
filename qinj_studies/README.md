# ETROC2 charge-injection studies

Reproducible, reviewer-facing notebooks for the ETROC2 internal charge-injection ("Q_inj",
"money plot") measurements: **what the front end does with a charge we know the size of.**

Charge injection is the only measurement in the ETROC programme where the input charge is a
number we set rather than whatever a sensor happened to produce. That is what makes it the right
place to measure the **preamp gain** (the front end's charge-to-threshold response, in
threshold-DAC LSB per fC), the threshold-edge width and the equivalent noise charge, the
time-over-threshold response to charge, and the time-of-arrival jitter — and to ask how each of
those moves with dose, with temperature, with the supply rails, with sensor bias, with the chip
revision, and with a sensor attached at all.

Every number and every figure in this folder is recomputed from staged data by the notebooks
themselves — the one exception is notebook 04, which deliberately recomputes nothing and reads
only the tables the other three wrote, so that both sides of every cross-campaign comparison went
through the same code. Nothing is quoted that a cell above it did not print or that a staged table
does not carry. Quantities are labelled **RAW** (a column, or a direct byte, of a source file) or
**DERIVED** (computed here), and where a campaign did not record something the text says
`not recorded` — never a guess.

Two conventions that decide how every number below should be read:

* **"Gain" always means the preamp gain**, the charge-to-threshold response of the front end in
  DAC LSB per fC. **No sensor multiplication enters any measurement in this folder**, notebook
  02's hybrid included: the charge is injected by each chip's own internal injector directly at
  the preamplifier input, so a bonded LGAD contributes its capacitance to that node and nothing
  else. No LGAD gain is measured or assumed anywhere.
* **Every ET2.03 `VA` and `VD` is a power-supply setting, not a voltage at the chip.** The August
  campaign's logbook (staged as `data/bare_et2p03_tid_aug2024/info_BARE{7,9}_voltage_scan.txt`)
  records `Nominal Analog 1.2V = 1.4V on power supply` and
  `Nominal Digital 1.2V = 1.25V on power supply` — a 0.20 V analog and 0.05 V digital IR drop over
  cables and board. So VA 0.85 / 1.00 / 1.20 / 1.40 V (supply) ≈ **0.65 / 0.80 / 1.00 / 1.20 V at
  the chip**, and only VA 1.40 V (supply) is nominal. The drop is I·R and no rail current was
  logged, so the at-chip values are approximate to roughly ±50 mV; the supply setting is the RAW
  quantity every table carries.

---

## Folder map

```
qinj_studies/
├── README.md                          this file
├── qinj_common.py                     shared loaders, constants and fits — every notebook imports it
├── qinj_style.py                      this folder's CMS plotting style, following the repo's convention
├── _build_nb.py, _src_0*.py           authoring scaffolding: the notebooks' percent-format source
├── 00_census_and_aux_data.ipynb       every Q_inj data set that exists, and what was monitored
├── 01_bare_et2p01_tid_jan2024.ipynb   bare ET2.01 die through 400 MRad
├── 02_hybrid_et2p01_cc_jan2024.ipynb  ET2.01 + LGAD hybrid, and bare-versus-hybrid
├── 03_bare_et2p03_tid_aug2024.ipynb   bare ET2.03 die through 400 MRad, on a supply-rail axis
├── 04_cross_campaign_synthesis.ipynb  the three data sets on one axis; reads only outputs/
├── data/
│   ├── bare_et2p01_tid_jan2024/       + PROVENANCE.md
│   ├── hybrid_et2p01_cc_jan2024/      + PROVENANCE.md
│   ├── bare_et2p03_tid_aug2024/       + PROVENANCE.md, DECODE_README.md and code/
│   └── census/                        + PROVENANCE.md — the inventory and the aux logs
├── figures/                           every figure as PNG (screen) and PDF (print), written by the notebooks
└── outputs/                           every table the notebooks write; 01, 02 and 03 write what 04 reads
```

`qinj_common.py` exists so that **the three measurement campaigns are analysed by literally the
same functions**
— the same turn-off estimator, the same weighted gain fit, the same shared-σ S-curve fit, the same
TOT log law, the same quantisation floor. Any difference between the three sets of results is then
a difference in the data, not a difference in the analysis. Notebook 03 needed exactly one
addition to that path and it is measured rather than asserted: the August campaign's binary search
converges on a different feature of the trigger-bit counter, so the turn-off it carries is the
campaign's own converged DAC — which notebook 03 shows agrees with the estimator used in 01 and 02
to **0.0 ± 0.7 DAC** on the 185 of 360 searches where both are computable.

`qinj_style.py` is a **new file written for this folder**, not a copy of anything in the repo. It
follows the ETROC-Analysis plotting convention of `TestBeam/BeamTestHelpers/plotting_basic.py` and
of the analysis-walkthrough figures — `mplhep` CMS style, the experiment label at the header left
via `hep.cms.text(loc=0, …)`, the figure name and conditions at the header right via
`ax.set_title(…, loc="right")` — and adds the size scaling and the multi-panel header axis that
those single-panel helpers do not have. Its experiment label, `ETL ETROC Charge Injection`, is this
folder's own member of the family the repo already uses (`ETL ETROC Test Beam`,
`ETL ETROC IRRAD`). `qinj_common.fig_header` is a thin wrapper that lets each notebook name its own
campaign in the header's right block without editing the style file.

## Notebooks

| # | notebook | one-line hook |
|---|---|---|
| 00 | `00_census_and_aux_data.ipynb` | **The census.** Every Q_inj data set that exists — 33 of them, positive and negative — what each can and cannot answer, where it lives, and what each campaign was monitoring while it took the data. Carries the auxiliary logs that the measurement notebooks do not. |
| 01 | `01_bare_et2p01_tid_jan2024.ipynb` | **The bare-die reference.** `ET2p01_W36_6`, no sensor, through 0/200/400 MRad: gain, the noise-set threshold edge, ENC, TOT-versus-charge, the money plot, and the dose-versus-temperature confound resolved by a fourth measurement corner. |
| 02 | `02_hybrid_et2p01_cc_jan2024.ipynb` | **What the sensor costs.** `ET2p01_Bar4` with an LGAD bonded, at −25/0/+22 °C and two bias states: the same measurements, the campaign's auxiliary logs, a confound-free bias bracket, and the bare-versus-hybrid comparison — where the answer inverts the usual expectation. |
| 03 | `03_bare_et2p03_tid_aug2024.ipynb` | **The later revision, a supply-rail axis, and the campaign's own monitoring.** `ET2p03_BARE9`, no sensor, through 0/200/400 MRad at four analog-rail settings with an **internal** voltage reference that the campaign **logged**: gain, ENC, TOT, the money plot, baseline against dose, the VA × VD rail grid, VRef/VTemp/rail currents through the irradiation with the 256-pixel baseline maps and their annealing, and the reference-scale decomposition of the baseline fall. `ET2p03_BARE7` is characterised in isolation (its post-0 MRad dose is unknown: the X-ray failed). |
| 04 | `04_cross_campaign_synthesis.ipynb` | **The three on one axis.** Reads only the tables 01, 02 and 03 wrote. What survives the revision change, what does not, everything that differs between the three campaigns in one table, and eight open questions ranked by how much confounding each would remove. |

**Notebook 00 is standalone** — it reads only `data/census/` and can be run first or last.
**Run 01 before 02, and 01–03 before 04.** Notebook 01 writes `outputs/bare_*.csv` and
`outputs/bare_results.json`, which notebook 02's bare-versus-hybrid section reads; notebook 03
writes `outputs/et2p03_*`; notebook 04 reads all of them and **recomputes nothing**, so both sides
of every comparison always went through the same code. Notebooks 02 and 04 stop with a clear
message listing what is missing.

---

## Results at a glance

### 00 — the census

32 data sets across six storage locations (the count is printed by the notebook's cell 5.0, as is
the list). Every count below is printed by the notebook, except
the last, which is its section 3 statement rather than a column tally.

| | |
|---|---|
| data sets with a real charge × threshold scan | **7** of 32 |
| hybrid (sensor-loaded) threshold scans anywhere | **1** — the campaign of notebook 02 |
| ET2.02 threshold scans anywhere | **0** — the ET2.02 data that exists is single-point, 30 fC at BL+20 |
| campaigns with a low-voltage supply log | 5 `yes`, 3 `partial` of 15 |
| campaigns with on-chip VTemp | **2** of 15 |
| campaigns with a sensor-bias log | **0** `yes`, 2 `partial` of 15 |
| data sets recording RFSel | **0** of 32, on any of the six locations |

Four findings there are new rather than inventory, and all are in the notebook's section 7:

* **The CERN January-2024 TID campaign had a supply log all along** — three `PowerHistory_v2.sqlite`
  files inside the tree notebook 01 analyses, 231 120 samples of analog and digital rail voltage
  and current at 5 s, never previously read. **The ~37 % analog-current step in it is not a dose
  effect:** both sides of the step are at 0 MRad (the scan-directory dose tags are re-parsed in the
  notebook to prove it), and it coincides instead with the cooldown from +12 to −25 °C *and* with a
  change in the supply terminal voltage itself. Across the dose series proper, all at −25 °C, the
  analog current moves by −1.5 % and *downward* (the digital rail by −0.2 %). What caused the step is not established, and
  the notebook says so rather than picking one of the three simultaneous changes.
* **The ET2.03 August-2024 register dumps say the on-chip reference generator was powered up**
  (`VrefGen_PD = 0`, `RefStrSel = 3`, identically on three dumps and two dies) — and that is all
  they say. Internal versus external VRef is board wiring with no register bit, so the dump proves
  the generator was **on**, not that it was **used**.
* **The ET2.03 internal reference has now been measured, on a different die** (census row 32,
  section 7.6). `ET2p03_BARE4`, a **bare ET2.03 die that took no charge-injection data and was
  never irradiated**, sat in the CERN climate chamber for ten days in August 2024 with its VRef
  pin and its on-chip temperature sensor on a multimeter at 5 s. The reference sits near **0.878 V
  at 0 °C — not 1.0 V** — and drifts **+0.573 ± 0.007 mV/°C**; the on-chip sensor responds at
  **+2.270 ± 0.020 mV/°C**, which is the only VTemp calibration in the census. The same set gives
  the only direct temperature coefficient of ETROC2 analog supply current anywhere:
  **+0.112 %/°C**, i.e. *cooling lowers the current* — the opposite sign to the January-2024 step
  above, which therefore stays unexplained rather than being attributed to the cooldown.
* **The ET2.03 dose campaign's monitoring logs were written, and have now been found**
  (section 7.7). An analysis notebook on the bench PC opened five databases of a
  `CERN_Aug_2024_TID_history` folder by name; the folder was absent from the bench PC, its backup
  disk, the newer bench PC, lxplus and LPC EOS, and was supplied by the campaign's operator on
  2026-09-04. It now lives on LPC EOS under `cern_Aug2024_TID/ETROC-History/` (with a second
  climate-chamber set, `CERN_Aug_2024_CC_Spirale3`, under `cern_Aug2024_CC/ETROC-History/`).
  Notebook 03 section 15 analyses it: **VRef rose +4.7 % over 400 MRad**, the VTemp pin shifted
  with dose, both rail currents fell a few per cent, and the baseline maps show a day of annealing
  after each step. The aux matrix no longer carries a `logged-missing` cell.

All values are recomputed by the notebooks. "Pixel RMS" is the pixel-to-pixel spread and is the
dominant uncertainty throughout; the ± is the uncertainty on the mean over pixels.

### 01 — bare `ET2p01_W36_6`, no sensor, 0 MRad, +12 °C, 12 pixels

| # | quantity | value | pixel RMS | label |
|---|---|---|---|---|
| 1 | Preamp gain (charge-to-threshold), 8–27 fC | **9.42 ± 0.89 DAC/fC** | 3.09 | DERIVED |
| 2 | the same in volts | 3.68 ± 0.35 mV/fC | 1.21 | DERIVED |
| 3 | Turn-off above baseline at 15 fC | 116.0 DAC | 37.2 | DERIVED from RAW FPGA counters |
| 4 | TOT at BL+20, 15 fC | 49.9 code (1.83 ns) | 2.2 | DERIVED |
| 5 | TOA jitter at BL+20, 15 fC | 6.4 ps | 3.0 | DERIVED — **at the quantisation floor** |
| 6 | TOA jitter at BL+20, 5 fC | 23.9 ps | 13.1 | DERIVED — above the floor, 9/12 pixels |
| 7 | S-curve width σ (72 curves, shared σ) | **0.92 ± 0.05 DAC** = 0.359 mV | — | DERIVED |
| 8 | ENC = σ / gain | **610 e⁻** | — | DERIVED — bare die, no input load |
| 9 | Autocal noise width NW | 3.17 ± 0.17 DAC | 0.58 | RAW (`NW_map`) |
| 10 | Minimum charge reaching BL+20 | 4.3 ± 0.4 fC | 1.5 | DERIVED, extrapolated |
| 11 | TOT-to-charge law at BL+20 | TOT = 19.3 + 10.7 ln(Q/fC), χ²/ndf 5.6 | — | DERIVED, valid 8–27 fC |
| 12 | TDC quantisation floor, LSB/√12 | 5.37 ps (TOA LSB 18.6 ps) | — | DERIVED |
| 13 | Walk with charge, 5→27 fC at BL+20 | +493 ps | — | DERIVED |
| 14 | Gain change on cooling +12 → −25 °C | −7.7 % | — | DERIVED |
| 15 | Gain change 0 → 400 MRad at −25 °C, ratio of pooled means | −5.5 % | — | DERIVED — carries no error |
| 16 | the same, **paired per pixel** | **−6.5 ± 3.2 %** | 11.0 | DERIVED — same 12 pixels at both doses; what notebook 04 compares |

Baseline shifts with dose: **+77 DAC** for the cooling, then **+25** and **+21 DAC** for the first
and second 200 MRad. Referred to each condition's own baseline the four turn-off curves coincide,
which is the operational result: **to 400 MRad, re-running autocalibration and working at a fixed
offset above the new baseline recovers essentially the same response.**

### 02 — hybrid `ET2p01_Bar4` + LGAD, 0 °C, HV 210 V, `highpower`, 9 pixels

| # | quantity | value | pixel RMS | label |
|---|---|---|---|---|
| 1 | Preamp gain (charge-to-threshold), 8–27 fC | **5.96 ± 0.37 DAC/fC** | 1.10 | DERIVED |
| 2 | the same in volts | 2.33 ± 0.14 mV/fC | 0.43 | DERIVED |
| 3 | Turn-off above baseline at 15 fC | 72.9 DAC | 12.3 | DERIVED from RAW FPGA counters |
| 4 | TOT at BL+20, 15 fC | 84.9 code (3.04 ns) | 6.4 | DERIVED |
| 5 | TOA jitter at BL+20, 15 fC | 10.6 ps | 2.4 | DERIVED — near the quantisation floor |
| 6 | TOA jitter at BL+20, **8 fC** | **27.7 ps** | 8.5 | DERIVED — **above the floor, the useful one** |
| 7 | S-curve width σ (54 curves, shared σ) | **0.93 ± 0.07 DAC** = 0.363 mV | — | DERIVED |
| 8 | ENC = σ / gain | **974 ± 97 e⁻** | — | DERIVED — the sensor-loaded number |
| 9 | Autocal noise width NW | 4.44 ± 0.24 DAC | 0.73 | RAW (`NW_map`) |
| 10 | Minimum charge reaching BL+20 | 5.5 ± 0.1 fC | 0.4 | DERIVED, extrapolated |
| 11 | TOT-to-charge law at BL+20 | TOT = −40.3 + 43.4 ln(Q/fC), χ²/ndf 9.0 | — | DERIVED — **a poor fit** |
| 12 | TDC quantisation floor, LSB/√12 | 5.24 ps (TOA LSB 18.14 ps) | — | DERIVED |
| 13 | Walk with charge, 5→27 fC at BL+20 | +1384 ps | — | DERIVED |
| 14 | Bias bracket, gain ratio HV nominal → 0 V (cable in) | **0.89** | — | DERIVED — the only confound-free capacitance handle |
| 15 | Gain ratio hybrid / bare | **0.63** | — | DERIVED — conditions differ, see notebook |
| 16 | ENC ratio hybrid / bare | **1.60** | — | DERIVED — entirely through the gain |

### 02 — the three things worth carrying away from the hybrid

1. **The sensor costs gain, not edge width.** The threshold-edge width in DAC is unchanged
   (0.93 against 0.92, ratio 1.01) while the gain falls to 63 %. ENC is the ratio of the two, so
   it grows by 1.60 **entirely through the gain**. The usual capacitance argument predicts a wider
   edge; the edge did not move. Referred to charge the front end is noisier, referred to threshold
   DAC it is not.
2. **The bias bracket reproduces that signature inside one day, on one chip.** At −25 °C on
   2024-01-16, the same nine pixels hours apart, turning the bias down to 0 V with the cable still
   plugged in moves gain ×0.89, σ ×1.20, ENC ×1.34 — the sign and the size the capacitance
   argument predicts, free of the chip, temperature and pixel-set confounds that limit the
   bare-versus-hybrid comparison.
3. **TOT is not transferable, and TOT-versus-charge is not a log law on a hybrid.** At BL+20 the
   hybrid sits at 1.7–1.8× the bare TOT **above 8 fC** (at 5 fC the ratio inverts, to 0.90) and
   the bare chip's well-behaved `19.3 + 10.7 ln Q` is
   replaced by a steep, badly fitting law (χ²/ndf 9.0). **RFSel is not recorded on either data
   set** and it sets the feedback resistance that fixes TOT, so this cannot be attributed to the
   sensor.

### 03 — bare `ET2p03_BARE9`, no sensor, 0 MRad, −23.5 °C, VA 1.20 V, VD 1.20 V (**supply settings**), 6 pixels

**This reference condition is under-volted.** VA 1.20 V on the supply is ≈ **1.00 V at the chip,
about 0.2 V below the nominal 1.2 V**; it is the reference only because it is the only analog
setting with a 0 MRad money plot. The one nominal setting is VA 1.40 V (supply) ≈ 1.20 V at the
chip, which has 200 and 400 MRad but no 0 MRad. Notebook 03 section 6 quotes the 200 → 400 MRad
step at both: at the nominal rail it costs **−4.96 ± 2.04 %** of preamp gain and **−19.2 ± 8.2**
DAC of baseline, against **−5.34 ± 2.76 %** and **−33.8 ± 7.5** DAC at the under-volted reference
— neither difference is resolved (0.1 σ and 1.3 σ).

The reference condition is one corner of a fourteen-condition grid over two dies. `pixel RMS` is over six pixels
here rather than twelve, so it is a noisier estimate of the spread than notebook 01's.

| # | quantity | value | pixel RMS | label |
|---|---|---|---|---|
| 1 | Preamp gain (charge-to-threshold), 8–22 fC | **8.26 ± 1.19 DAC/fC** | 2.91 | DERIVED |
| 2 | the same in volts, under the assumed LSB | 3.22 ± 0.46 mV/fC | 1.14 | DERIVED — the LSB is an assumption on this revision |
| 3 | Preamp gain over the analog rail, supply 0.85 → 1.40 V (~0.65 → ~1.20 V at chip), at 200 MRad | **3.11 → 8.69 DAC/fC** | — | DERIVED |
| 4 | Turn-off above baseline at 15 fC | 100.5 DAC | 30.3 | RAW search DAC, referred to the noise-peak baseline |
| 5 | TOT at BL+20, 15 fC | 46.0 code (1.55 ns) | 2.7 | DERIVED |
| 6 | TOA jitter at BL+20, 15 fC | 8.3 ps (median 8.4) | 2.1 | DERIVED — near the quantisation floor |
| 7 | TOA jitter at BL+20, 6 fC | 22.7 ps (median 15.2) | 12.4 | DERIVED — above the floor; the mean is pulled by the widest pixel, hence the median |
| 8 | S-curve width σ (27 curves, shared σ) | **1.12 ± 0.10 DAC** = 0.438 mV | — | DERIVED |
| 9 | ENC = σ / gain | **847 e⁻** | — | DERIVED — bare die, no input load |
| 10 | Noise-peak FWHM | 2.99 ± 0.45 DAC | 1.11 | DERIVED; the one autocal map gives NW 4.17 DAC |
| 11 | Minimum charge reaching BL+20 | 4.9 ± 0.8 fC | — | DERIVED, extrapolated |
| 12 | TOT-to-charge law at BL+20 | TOT = 9.5 + 13.1 ln(Q/fC), χ²/ndf 1.7 | — | DERIVED, valid 6–22 fC |
| 13 | TDC quantisation floor, LSB/√12 | 4.88 ps (TOA LSB 16.9 ps at CAL 185) | — | DERIVED |
| 14 | Walk with charge, 6→22 fC at BL+20 | +518 ps | — | DERIVED |
| 15 | **Baseline shift, 0 → 400 MRad at VA 1.20 V (supply), `BARE9`** | **−65.5 DAC** (6-pixel in-config); **−80.1 DAC** on the 256-pixel autocal maps | — | DERIVED — sign opposite to ET2.01; row 17 attributes about half of it to the reference |
| 16 | Paired preamp-gain change, 0 → 400 MRad at VA 1.20 V (supply) | **−5.4 ± 4.3 %** | 10.4 | DERIVED — same six pixels, paired ratio |
| 16b | the same 200 → 400 MRad step at **VA 1.40 V (supply) = nominal at the chip** | **−4.96 ± 2.04 %**, baseline **−19.2 ± 8.2 DAC** | — | DERIVED — the nominal-rail version; no 0 MRad exists there |
| 17 | **VRef (internal reference) of `BARE9`, 0 → 200 → 400 MRad** | **0.864 → 0.883 → 0.904 V (+2.3 %, +4.7 %)** at constant plate temperature | — | MEASURED from the campaign's VRef log (section 15) |
| 18 | Reference-scale part of row 15, assuming DAC full scale ∝ VRef | **−37 DAC** of the −65.5 (6-pixel) / −80.1 (map); the rest, **−29 / −43 DAC**, is front-end drift | — | DERIVED (section 15) |
| 20 | **X-ray footprint** (256-pixel maps): rows 0–4 outside the beam vs rows 7–15 inside, 0 → 400 MRad | **−37 vs −104 DAC**; the outside rows match the reference-only prediction (−37) | — | DERIVED (section 15b) — the reference-scale assumption checked from inside the data; four of the six money-plot pixels were outside the beam |
| 19 | Preamp-gain change 0 → 400 MRad, **reference-corrected** | **−0.7 ± 4.3 %** (measured −5.4 ± 4.3 % in DAC/fC, + 4.7 % scale) | — | DERIVED — consistent with no loss on this die |

This campaign is also the only one with a **decode artefact big enough to matter**, and section
4.4 is given over to it: on each pixel a contiguous run of one to six threshold DAC values, at a
fixed absolute DAC and the same run at every injected charge, carries hits with a grossly wrong
CAL — `cal_std` 23 to 34 codes against a normal 0.4, and a TOA standard deviation of 25 to 33
codes against 0.4 to 2 once the bad hits are gone. The blocks land at different absolute DAC on
different pixels, so the median over six pixels smears them into one broad charge-independent bump
in the middle of the money plot. Every TOA and TOT number in this notebook is therefore taken over
the CAL-windowed hits (`|CAL − cal_mode| ≤ 3`, 2.68 % of hits dropped); the raw moments are kept
beside them and the before/after is drawn rather than asserted. Turn-off, gain, σ, the TOT law and
the walk are untouched by it — they come from the binary searches and from points with
`cal_std < 0.7` — and the number that moves is row 7 above.

Three further results are specific to this campaign because no other one has the axis to see them.

* **Under-volting the analog rail is a bigger lever than the dose, by a wide margin — and
  under-voltage is what this axis measures.** Moving VA from 0.85 to 1.40 V *on the supply* changes
  the DAC-referred preamp gain by a factor of about 2.8 and the DAC-referred baseline by about
  430 DAC/V. Those settings are **~0.65 → ~1.20 V at the chip**: the scan runs from roughly half
  the nominal analog rail *up to* nominal and never above it, so this is a statement about
  under-voltage sensitivity, not about a range around the operating point. That is *not* a demonstration that the DAC scale moves: the internal reference
  is powered from VA, so the millivolts per code may track it, but so are the preamplifier bias
  currents, so the front-end gain may genuinely change. A ×2.8 swing from the reference alone
  would need the reference itself to move by ×2.8, which is not plausible, so a real front-end
  change is at least as likely and probably the larger part. Separating them needs the
  threshold-DAC transfer function — open question 1.
* **On `BARE9` the baseline falls with dose on every rail**, by −65 DAC over 400 MRad at both
  VA 1.20 V and VA 0.85 V (supply), and by −80 DAC on the 256-pixel autocalibration maps. On ET2.01
  it *rose*. The campaign's own VRef log now says why part of it happens: **the internal reference
  rose by +4.7 % over the same dose**, and a larger reference lowers every threshold code of an
  unchanged analog level. Under the one assumption that the DAC full scale follows VRef, that is
  −37 DAC of the fall; the remaining −29 (6-pixel) to −43 DAC (map) is the front end itself drifting
  *down*, where the ET2.01 front end drifted *up* on a fixed external reference. So the sign
  difference survives the correction and stays a one-die-per-revision observation: the only other
  irradiated ET2.03 die, `BARE7`, has an unknown dose (the X-ray source failed before the tagged
  200 MRad) and is kept out of every comparison. Notebook 03 sections 11, 14 and 15.
* **The X-ray spot covered only rows ~6–15 of `BARE9`.** Rows 0–4 shifted by exactly the amount the
  VRef rise predicts (−37 vs −37 DAC), rows 7–15 by −104 DAC; so the un-irradiated rows validate the
  reference-scale assumption from inside the data, and the irradiated rows carry a front-end drift of
  about −67 DAC. Four of the six money-plot pixels (rows 0 and 3) were outside the beam, so every
  six-pixel dose number is a 4:2 mixture; section 15b quotes the split. The January ET2.01 maps show
  the same thing the other way round: the spot covered rows 0–3 of `W36_6` (nine of its twelve pixels),
  the un-irradiated rows did not move on the fixed external reference, and the irradiated rows rose
  by +33 DAC at 200 MRad (notebook 01 section 11.2).
* **Corrected for the reference, `BARE9`'s preamp gain did not measurably change**: −5.4 ± 4.3 % in
  DAC/fC becomes −0.7 ± 4.3 % in mV/fC. Its VTemp pin rose ~30 mV during the first step at
  constant plate temperature and stayed there — the temperature sensor shifts with dose — and both
  rail currents fell by a few per cent. The baseline maps anneal back by several DAC in the day after
  each step; the quoted dose numbers are taken ≥ 12 h after a step.
* **Running well below the nominal analog rail costs more than the dose does.** At VA 0.85 V
  (supply) ≈ 0.65 V at the chip the same 400 MRad costs **−17.1 ± 4.8 %** of preamp gain against
  −5.4 ± 4.3 % at VA 1.20 V (supply) ≈ 1.00 V at the chip; the noise scan at that corner
  degenerates; and on the digital rail every sub-run at VD ≤ 1.00 V (supply) after irradiation
  decodes to **exactly zero hits**. The campaign's `info.txt` says precisely which failure was seen where,
  written down at the time: after 200 MRad, **pixel auto-calibration failed at VD = 0.90 V
  (supply, ~0.85 V at the chip)**, and **at VD = 0.95 V (supply, ~0.90 V at the chip) the links and
  the injector failed even though the pixels calibrated** —
  `QInj` events empty but at the right BCIDs. Decoding those sub-runs blind reproduces both.

### 04 — the three data sets on one axis

Notebook 04 recomputes nothing; it reads the tables 01, 02 and 03 wrote.

Notebook 04's summary now carries a fifth column, `et2p03_nominal_rail`: the same quantity on
`ET2p03_BARE9` at **VA 1.40 V (supply) ≈ 1.20 V at the chip, the one nominal setting**, at
200 MRad. The `ET2.03 bare` column below is at VA 1.20 V (supply) ≈ 1.00 V at the chip and 0 MRad
— under-volted, and used because it is the only analog setting with a 0 MRad scan. The two differ
in *both* rail and dose, so the nominal column is a companion, not a correction.

| # | quantity | unit | ET2.01 bare | ET2.01 + LGAD | ET2.03 bare | comparable across the three? |
|---|---|---|---|---|---|---|
| 1 | Preamplifier gain at each reference condition | DAC/fC | 9.42 ± 0.89 | 5.96 ± 0.37 | 8.26 ± 1.19 | **no** — the DAC scale differs between revisions |
| 2 | Gain change, 0 → 400 MRad, **paired per pixel** | % | −6.5 ± 3.2 (12 px) | no dose series | −5.4 ± 4.3 (6 px) in DAC/fC; **−0.7 ± 4.3 after the measured +4.7 % reference rise** | **yes** as a fractional change — but the ET2.03 value needs the reference correction |
| 3 | Baseline shift, 0 → 400 MRad | DAC | +46.2 at −25 °C (front end; reference fixed) | no dose series | −65.5 at VA 1.20 V supply (`BARE9`): −37 reference scale (measured VRef +4.7 %) + −29 front end | front end against front end: **opposite signs**, one die per revision |
| 4 | Turn-off above baseline at 15 fC | DAC | 116.0 | 72.9 | 100.5 | the offset cancels, the scale does not |
| 5 | TOT at BL+20, 15 fC | ns | 1.83 | 3.04 | 1.55 | yes in ns — but RFSel is unrecorded on all three |
| 6 | TOA jitter at BL+20, 15 fC | ps | 6.4 | 10.6 | 8.3 | near the floor, ×1.18 / ×2.02 / ×1.69 of it — limits, not resolutions |
| 7 | TDC quantisation floor | ps | 5.37 | 5.24 | 4.88 | a property of the TDC and CAL |
| 8 | S-curve width σ | DAC | 0.92 | 0.93 | 1.12 | **no** — the DAC scale differs between revisions |
| 9 | **ENC = σ / gain** | electrons | **610** | **974** | **847** | **yes** — the only scale-free noise number |
| 10 | Autocalibration NW (map) | DAC | 3.17 | 4.44 | 4.17 | same definition, different DAC scale |
| 11 | Preamp gain range over VA 0.85 → 1.40 V (supply) ≈ 0.65 → 1.20 V at chip | DAC/fC | not recorded | not recorded | 3.11 → 8.69 | only ET2.03 measured it — and it is an under-voltage scan up to nominal |

**The five things worth carrying away**

1. **The baseline moves the opposite way with dose on the two dies compared, and the measured
   reference explains only half of the ET2.03 side.** ET2.01 bare (`W36_6`, fixed external
   reference): baseline +46 DAC over 400 MRad, preamp gain −6.5 ± 3.2 % (paired, same 12 pixels).
   ET2.03 bare (`BARE9`): baseline −65 DAC (6-pixel) / −80 DAC (256-pixel maps) over the same dose,
   while its internal reference **rose by +4.7 %** in the campaign's own VRef log. If the DAC full
   scale follows VRef, that accounts for −37 DAC; the remaining −29 to −43 DAC is the ET2.03 front end
   drifting *down* where the ET2.01 front end drifted *up*. The ET2.03 preamp gain, corrected for the
   same scale change, is −0.7 ± 4.3 %, consistent with no loss. Both campaigns' 256-pixel maps show a
   **partial X-ray spot** (rows 0–3 in January, rows ~6–15 in August): the un-irradiated rows are a null
   test both dies pass, and the irradiated rows give front-end drifts of +33 DAC (ET2.01, 200 MRad) and
   −67 DAC (ET2.03, 400 MRad). No second die of either revision has a usable dose step (`BARE7`'s dose
   is unknown), so every sign here is a one-die statement. Notebook 04 section 2b.
2. **In DAC units both revisions lose of order five per cent of gain per 400 MRad, and on ET2.03 the
   second dose step repeats at every rail setting the campaign used** (on ET2.03 that is largely the
   reference-scale effect of item 1; the consistency across rails still holds). The four 200 → 400 MRad steps are
   −5.25 ± 3.87, −5.39 ± 1.94, −5.34 ± 2.76 and −4.96 ± 2.04 % at VA 0.85, 1.00, 1.20 and 1.40 V —
   a spread of 0.4 points against per-step standard errors of 1.9 to 3.9 points. That is a
   consistency check, **not four independent measurements**: one die, the same six pixels and the
   same two dose states, re-read at four supply settings, so the four steps share their pixel
   sample and their irradiation entirely. The ±0.7 % repeatability floor comes from the two
   independent 0 MRad acquisitions and bounds acquisition-to-acquisition scatter, nothing else.
3. **Under-volting the analog rail is the largest lever in the folder.** One ET2.03 die's preamp
   gain changes by ×2.8 across its own VA range — more than bare-versus-hybrid, more than the
   revision change, more than 400 MRad. That range is ~0.65 → ~1.20 V *at the chip*, i.e. from
   about half the nominal rail up to nominal, so it bounds under-voltage sensitivity rather than
   describing behaviour around the operating point. Neither January campaign recorded its rails at
   all, so no absolute gain comparison between them is safe.
4. **ENC is the only noise number that survives the revision change**, because the DAC scale
   cancels in σ/gain. The σ values themselves cannot be compared across revisions at all until
   someone measures the threshold-DAC transfer function, which is open question 1.
5. **Referred to each condition's own baseline, the turn-off curves coincide on both revisions.**
   Whichever way the baseline moved, re-running autocalibration and working at `BL + 20` recovers
   the response to 400 MRad. That is the operational result, and it is the one statement that
   survives every confound above.

**Eight open questions, ranked** by how much confounding each would remove, are in notebook 04's
section 9 and in `outputs/xcamp_open_questions.csv`. The first three:

1. **What is the threshold-DAC LSB in millivolts on each revision, and how does it move with VA on
   ET2.03?** A DAC transfer-function measurement against an external voltmeter. It is the single
   largest blocker: it would make σ, NW, baseline shifts and gain-in-mV comparable across
   revisions.
2. **What was RFSel, and how much does it move gain and TOT?** Unrecorded on all three data sets,
   and it fixes TOT directly, so every TOT comparison here bounds a difference rather than
   measuring one.
3. **Does the baseline-versus-dose sign flip follow the reference, the revision or the die?** Two
   dies of one revision through the same dose steps with the reference switchable on the board.
   This is the measurement that would turn the folder's most striking observation into a result.

### Open discrepancies, stated rather than smoothed over

* **The two campaigns disagree on the sign of the temperature coefficient of gain.** The hybrid
  gains about +24 % on cooling from +22 to −25 °C; the bare die *loses* 7.7 % on cooling from +12
  to −25 °C. Neither data set can resolve it: the hybrid's −25 °C bias is `not recorded`, and the
  bare die's −25 °C point comes from turn-off files only.
* **σ = 0.92 DAC, not the 0.88 previously published for the bare die.** Notebook 01 implements
  both estimators — the shared-σ profile fit and the align-then-stack recipe — and both return
  0.92 on this data. The hybrid analysis independently re-fitted the same bare probe table and
  also obtained 0.92 ± 0.05. Two independent implementations agreeing is the stronger evidence,
  so 0.92 DAC and 610 e⁻ are what these notebooks carry; 0.88 and 584 e⁻ are noted as the earlier
  published values. Against 584 e⁻ the hybrid ENC ratio would read 1.67 instead of 1.60.
* **The turn-off estimator.** These notebooks interpolate the efficiency to 0.5 between the
  highest probe at or above 50 % and the next one above it; the previously published hybrid tables
  took simply the highest probe at or above 50 %. The interpolated point is sub-LSB and is the
  estimator the bare-die tables were built with, so using it everywhere is what makes the two
  campaigns comparable — but it lands about 0.44 DAC higher, which shifts the hybrid's turn-off
  offsets, low-charge gain cross-check, fit intercept and `Q_min` slightly. Notebook 02
  quantifies the shift and shows that the gain itself, being a slope, is untouched (5.957 against
  5.955 DAC/fC).
* **`highpower` is a directory tag, not a measurement.** Both hybrid flagship configurations carry
  it, yet the logged median analog supply current on their two days differs by about half
  (0.342 A against 0.499 A), and the 2024-01-18 rail visibly steps mid-day. Unresolved.
* **The August campaign's binary search converges on a different feature.** Its trigger-bit counter
  plateaus at two counts per Level-1 accept immediately above the baseline, where noise as well as
  the injected charge fires the discriminator, and the search often settles on the top of that
  shelf rather than on the efficiency edge. Notebook 03 measures the consequence rather than
  assuming it away: on the 185 of 360 searches where the efficiency crossing is bracketed to two
  DAC or less, the estimator used in notebooks 01 and 02 and the campaign's own converged DAC agree
  to **0.0 ± 0.7 DAC**; on the other 175 the bracket averages 25 DAC and the interpolated estimator
  is not usable. The campaign's own definition is what notebook 03 carries, and the threshold sweep
  cannot arbitrate because its upper bound was set to the search's answer — every sweep stops
  exactly one DAC short with the hit rate still on its plateau.
* **Twenty of the 360 August searches ran into the 10-bit DAC ceiling** and are lower limits rather
  than measurements. Notebook 03 quotes the gain fit both with and without them; at VA 1.40 V (supply),
  where seven of thirty are censored, putting them back in drags the slope down by about a third.
* **σ = 1.12 DAC on ET2.03 against 0.92 on ET2.01, and the folder cannot say what that means.**
  Both are the same estimator on the same kind of data. Whether the ET2.03 edge is really wider, or
  simply measured in finer DAC steps, needs the threshold-DAC transfer function that no campaign
  measured. The ENC ratio is what notebook 04 compares instead.

### Reproduction ledgers

Every headline number is compared, in the cell that computes it, against the previously published
value, and the comparison is collected into a ledger at the end of each notebook.

| notebook | checks | agree | disagree |
|---|---|---|---|
| 01 | 37 | 35 | S-curve σ (0.92 vs 0.88) and the ENC that follows — see above |
| 02 | 38 | 35 | low-charge gain cross-check, fit intercept, `Q_min` — all the turn-off-estimator shift |
| 03 | 12 | 12 | — |

Notebook 03's ledger is shorter because the August reduction documents fewer of its own numbers:
the point and hit counts, the pixel-selection loss, the autocalibration baselines, and the
money-plot turn-off DACs of one pixel at one rail. All twelve reproduce exactly. **Notebook 04 has
no ledger** — it recomputes nothing, so there is nothing to reproduce.

`outputs/bare_reproduction_ledger.csv`, `outputs/hybrid_reproduction_ledger.csv` and
`outputs/et2p03_reproduction_ledger.csv` carry them machine-readably.

---

## Data provenance

Full detail, including how each file was extracted and what was deliberately left out, is in
`data/*/PROVENANCE.md`. Summary:

### `data/bare_et2p01_tid_jan2024/` — read from `root://cmseos.fnal.gov//store/group/lpcmtdstudies/cern_Jan2024/TID/`

| file | bytes | md5 | what it is |
|---|---|---|---|
| `ET2p01_W36_6_TDC_summary_allMRad.feather` | 1,186,578 | `333d13522d479f39f3d5f097934532f8` | RAW. Per-point TOA/TOT/CAL mean and std, 22 055 × 12. The backbone. Same md5 as on EOS. |
| `fpga_turnoff_dat.tar.gz` | 34,184 | `686370c1e9ec3b994d60bb257fdd03c2` | RAW bytes. The 288 `FPGA_Data.dat` binary-search logs. |
| `fpga_turnoff_probes.csv` | 117,133 | `37d8b6cb5c3b8b18a9e5e90b772e9e8d` | RAW, transcribed. The same 288 files parsed; notebook 01 re-parses the tarball and proves the match. |
| `blnw_map_pickles.tar.gz` | 2,797 | `b7a3c0e732a16e9a5f5ab070ec55f7b8` | RAW bytes. Autocalibration BL/NW maps for the four days on EOS that have one. |
| `blnw_maps.csv` | 26,421 | `5fe12a58705b98073304e50948415115` | RAW, flattened. The same maps as a table. |
| `sweep_point_dirs.csv.gz` | 143,772 | `c8a24b708d9b788eadaa607f5ad14f59` | RAW. One row per scanned point directory with its size in bytes, 23 416 rows. Gives hit counts and the censoring flag. |

### `data/hybrid_et2p01_cc_jan2024/` — reduction of `root://cmseos.fnal.gov//store/group/lpcmtdstudies/cern_Jan2024/CC/`

| file | bytes | md5 | what it is |
|---|---|---|---|
| `cc_points.feather` | 1,187,466 | `a0f99500063c9f99c2215f96e1b624ee` | RAW. Per-point TOA/TOT/CAL mean and std plus hit count, 13 281 × 22. |
| `cc_fpga_probes.csv.gz` | 126,377 | `ac52fd05c8a0b1e80c8310ae72d088df` | RAW. Every binary-search probe point. |
| `cc_tdc_dirs.csv.gz` | 118,901 | `6b7f1c13738436760bd6e958c13f3932` | RAW. One row per scanned point directory, including those with no hits. |
| `cc_turnoff.csv` | 101,248 | `51a919ee1bfe5653270389a894b7494b` | DERIVED. The reduction's own turn-off; recomputed from the probes in notebook 02, not trusted. |
| `cc_blnw.csv.gz` | 44,320 | `5ce2d9cb41b57815c046042b58dece69` | RAW. Autocalibration maps, 256 pixels × 19 maps. |
| `cc_noiseonly.csv` | 44,755 | `313eb6c0dda21af7f6aff7facf8d4988` | RAW. In-configuration NoiseOnly scans. |
| `cc_configs.csv` | 10,995 | `0418c23834af8933eaa2f3f0ffcdff02` | RAW + DERIVED. Configuration inventory with the provenance of each metadata field. |
| `aux_lv_timeseries.csv.gz` | 161,004 | `df851467b6e05bec42f13e571c700e1d` | RAW, downsampled. LV rails from `PowerHistory_v2.sqlite`, 1 sample/rail/minute of 233 576. |
| `aux_baseline_history.csv.gz` | 103,395 | `56d65ceec8df9e24eb29a4ba6e50ca8a` | RAW, aggregated. `BaselineHistory_CC_Jan2024_CERN.sqlite`, 33 170 rows averaged per run and minute. |
| `table_aux_power_lv.csv` | 1,237 | `2f1d6e88c512609fe42e46fcf2702488` | DERIVED. Published per-day rail summary; cross-check target only. |
| `summary_numbers.json` | 39,915 | `9cfd699fc55d6a6a9a815afba4ab3c0e` | DERIVED. Published headline numbers; cross-check target only. |

### `data/bare_et2p03_tid_aug2024/` — reduction of `fermilab-etroc3:/home/daq/tmp/TID_August/` and of the removable disk `X9`

3.4 MB in total, and unlike the two January directories it also ships the reduction's own
documentation (`DECODE_README.md`) and the scripts that produced it (`code/`), because the raw
trees are on a bench machine and a removable disk rather than on EOS. Notebook 03 prints every md5
below in its section 1.

| file | bytes | md5 | what it is |
|---|---|---|---|
| `tid_et2p03_points_all.feather` | 4,965,266 | `34ec6001da66d9a48103cc9981a70614` | RAW + DERIVED. Per-point TOA/TOT/CAL mean and standard deviation in raw codes, plus the CAL-windowed repeat of them, 29 922 × 43. **The union table** — the only one carrying the dosed money plots. |
| `tid_et2p03_turnoff_all.csv` | 186,895 | `482b9e457aae1d32a385fa23f3316e14` | RAW. The 450 discriminator binary searches, each written out in full as `search_trace`. |
| `tid_et2p03_noiseonly_all.csv` | 44,762 | `e18213b72eed3cbd5d27ff793955cd6b` | RAW. The 90 `quietCB` noise scans — a dense threshold scan of the trigger-bit counter, taken inside every configuration. This is what gives a baseline at all fourteen conditions. |
| `tid_et2p03_configs_all.csv` | 30,212 | `11a3cbc4035b63c6a6c42a11f7f3870e` | RAW + DERIVED. The conditions grid, one row per run or rail sub-run. |
| `tid_et2p03_points.feather` | 610,234 | `11832f9bef784d13102d9625180c068d` | RAW + DERIVED. The `TID_August` working copy alone, where the dosed data are the 10 fC rail scans only. Staged so the difference between the two table families is demonstrable. |
| `tid_et2p03_turnoff.csv` | 21,696 | `a0cf7d157ac24337bd223c57c0cc42ca` | RAW. The same, working copy only. |
| `tid_et2p03_noiseonly.csv` | 5,411 | `99257c0ca258a409cf814da5a2ee80d3` | RAW. The same, working copy only. |
| `tid_et2p03_configs.csv` | 24,809 | `d857637ee4f399ffab5353dd119ad860` | RAW + DERIVED. The same, working copy only. |
| `tid_et2p03_blnw.csv` | 14,833 | `96da68fed9fef08cf920002f4bfa5023` | RAW. The one autocalibration BL/NW map the campaign left behind, `BARE9` 2024-08-03, 256 pixels. Byte-identical to `data/census/aux_et2p03_blnw.csv`. |
| `tid_et2p03_chipconfig.csv` | 3,164 | `5aef7758fd020bddf206430880301a20` | RAW, decoded. The three register read-backs, 135 fields plus the raw `PeriCfg0..31` bytes. Byte-identical to `data/census/aux_et2p03_chipconfig.csv`. |
| `info_BARE9_voltage_scan.txt` | 1,703 | `d79ee484fd73b8d85b6c3824c96dce71` | RAW, verbatim. The campaign logbook of the BARE9 voltage scan. Its first two lines are the **supply-versus-chip mapping** every VA/VD label in this folder depends on; the rest is the contemporaneous record of the VD failures. |
| `info_BARE7_voltage_scan.txt` | 1,694 | `b579ec8c8eb3ba7866e46eafb745f3e3` | RAW, verbatim. The same for BARE7 (its 0 MRad VD 0.90 V note differs: "stable links but no QInj" against BARE9's "highly unstable links"). |
| `aux/aux_tid_adc_1min.csv.gz` | 217,172 | `71fe7d2633261b777457f54b59ff34c7` | RAW, downsampled. External ADC ch 0/1/7/8, 1-minute medians; ch 7 = VTemp pin, ch 8 = room sensor. Origin: the operator's `CERN_Aug_2024_TID.zip` (2026-09-04), full-rate originals on LPC EOS `cern_Aug2024_TID/ETROC-History/CERN_Aug_2024_TID/`. |
| `aux/aux_tid_baseline_maps.csv` | 25,860 | `f07f05c3231ad51240c7c79cdfe7dffc` | DERIVED. One row per 256-pixel autocalibration map: mean/std of baseline and noise width. Origin: the operator's `CERN_Aug_2024_TID.zip` (2026-09-04), full-rate originals on LPC EOS `cern_Aug2024_TID/ETROC-History/CERN_Aug_2024_TID/`. |
| `aux/aux_tid_baselines_full.csv.gz` | 380,071 | `21f9411b14eb98845d41ea123976571f` | RAW. All 34 304 rows of `BaselineHistory.sqlite`, per pixel, with the `save_notes`. Origin: the operator's `CERN_Aug_2024_TID.zip` (2026-09-04), full-rate originals on LPC EOS `cern_Aug2024_TID/ETROC-History/CERN_Aug_2024_TID/`. |
| `aux/aux_tid_baselines_pre_campaign.csv.gz` | 50,393 | `827473a742635a23dd8d398ad2bfa5ef` | RAW. Pre-campaign maps of eleven chips, 07-23 → 08-03. Origin: the operator's `CERN_Aug_2024_TID.zip` (2026-09-04), full-rate originals on LPC EOS `cern_Aug2024_TID/ETROC-History/CERN_Aug_2024_TID/`. |
| `aux/aux_tid_irradiation_timeline_plot_ipynb.csv` | 150 | `f2e2b479993f5aa54af760cfbc334708` | RAW, transcribed. The four irradiation markers from the campaign's `plot.ipynb`. Origin: the operator's `CERN_Aug_2024_TID.zip` (2026-09-04), full-rate originals on LPC EOS `cern_Aug2024_TID/ETROC-History/CERN_Aug_2024_TID/`. |
| `aux/aux_tid_power_1min.csv.gz` | 375,414 | `cebc3dd8c377510657afeb429ffb7023` | RAW, downsampled. Analog/digital/WS rails V and I, 1-minute medians, three logger files. Origin: the operator's `CERN_Aug_2024_TID.zip` (2026-09-04), full-rate originals on LPC EOS `cern_Aug2024_TID/ETROC-History/CERN_Aug_2024_TID/`. |
| `aux/aux_tid_power_actions.csv` | 26,408 | `2708d6e5b9298258c69be0838b0c33f2` | RAW. The 578 supply on/off/logging actions. Origin: the operator's `CERN_Aug_2024_TID.zip` (2026-09-04), full-rate originals on LPC EOS `cern_Aug2024_TID/ETROC-History/CERN_Aug_2024_TID/`. |
| `aux/aux_tid_vref_1min.csv.gz` | 184,059 | `87bbde4d5cfb12019023f87e058ca517` | RAW, downsampled. VRef pin, 1-minute medians (`vol_monitoring.sqlite`). Origin: the operator's `CERN_Aug_2024_TID.zip` (2026-09-04), full-rate originals on LPC EOS `cern_Aug2024_TID/ETROC-History/CERN_Aug_2024_TID/`. |
| `DECODE_README.md` | 38,569 | `d230a08548ff4d42fc75529613aa9c82` | The reduction's own documentation, verbatim: formats, decoder provenance, verification, caveats. |
| `code/decoding.py` | 46,365 | `289ef6cb7b7e48589e02d817de5060ed` | **This repository's own `DecodeBinary`**, copied byte for byte from `TestBeam/condor_at_lxplus/core/decoding.py`. Notebook 03 prints that md5 against the repository's. |
| `code/etroc2_regmap.py`, `reduce_tid.py`, `verify_hand.py`, `chipconfig_to_csv.py`, `merge_all.py`, `postfix_elapsed.py` | 100,254 | see `PROVENANCE.md` | The rest of the reduction, shipped for provenance. `verify_hand.py` is the independent decoder written from the format description alone. |

### `data/census/` — the inventory and the auxiliary logs no other notebook plots

| file | bytes | md5 | what it is |
|---|---|---|---|
| `qinj_census.csv` | 14,608 | `a9b347d8a3467067a1865f7e354eb5fe` | DERIVED. One row per data set, 32 × 24, positives and negatives. |
| `qinj_aux_matrix.csv` | 6,769 | `401cb5c396b8bb4ebe1f02ad6a44c3a6` | DERIVED. 16 campaigns × 8 kinds of monitoring, `na`/`no`/`partial`/`unknown`/`yes` (`logged-missing` is defined but no longer used); `yes`. `logged-missing` is its own value and means *written, and not located* — it is the ET2.03 dose campaign of section 7.7, not a `no`. |
| `aux_tid_lv_timeseries.csv.gz` | 153,597 | `6672b01ca9962224c36ba6873b8e0150` | RAW, downsampled. The TID campaign's analog and digital rails, 1 sample/rail/minute of 231 120. |
| `aux_tid_lv_actions.csv` | 522 | `175bc9762152207a1afe27adac3115fe` | RAW. The logger's own Start/Stop rows. |
| `aux_tid_baseline_history.csv` | 84,803 | `60f056bf91b1444efce65208a7a994ef` | RAW. That campaign's entire baseline log: 1 024 rows, 23 minutes. |
| `aux_et2p03_chipconfig.csv` | 3,164 | `5aef7758fd020bddf206430880301a20` | RAW, decoded. The three ET2.03 August-2024 register read-backs. |
| `aux_et2p03_blnw.csv` | 14,833 | `96da68fed9fef08cf920002f4bfa5023` | RAW. ET2.03 `BARE9` BL/NW maps, 256 pixels. |
| `aux_ccts_vol.csv.gz` | 175,344 | `2e1b161c93115e84448e89672496a37c` | RAW, downsampled. The ET2.03 internal VRef and on-chip VTemp of census row 32, 1 sample/instrument/minute of 336 166. |
| `aux_ccts_chamber.csv.gz` | 59,554 | `a4e0ee163be1683124e9f7cf1a30c1ff` | RAW, downsampled. That campaign's chamber air temperature and humidity. |
| `aux_ccts_lv.csv.gz` | 224,893 | `b8e8731283c5dac6e56f219690935964` | RAW, downsampled. Its supply rails, three instruments, 1/instrument/rail/minute of 350 489. |
| `aux_ccts_lv_actions.csv` | 21,962 | `90a1ac41ea65c22cd22fb7adbce3bb61` | RAW. Its power logger's action rows. |
| `aux_ccts_blnw.csv.gz` | 390,706 | `421f81df91dcad68ac075e627fd510b1` | RAW. Its 140 full 16 × 16 autocalibration maps, with the condition text parsed. |
| `aux_tid_et2p03_timeline.csv` | 738 | `bfa86d85a080a291e4e6e559fdc938db` | RAW, transcribed. The ET2.03 irradiation start/stop markers of section 7.7. |
| `census_x9_history_folders.csv` | 1,436 | `f53ffc3f2aa1ebe12f204c5d1a901f3e` | RAW. The nineteen per-campaign history folders of the X9 backup disk. |

The five `aux_ccts_*` files come from the `CERN_Aug_2024_CC_CTS` folder of the `etroclaptop3`
backup on the removable X9 disk; that copy did not preserve timestamps, so every date in them
comes from a timestamp column inside a file and never from an mtime.

The three `aux_tid_*` files were extracted for notebook 00 from
`root://cmseos.fnal.gov//store/group/lpcmtdstudies/cern_Jan2024/TID/`; the two `aux_et2p03_*`
files are byte-for-byte copies of the ET2.03 reduction's own tables. The downsampling of the LV
series reproduces every per-day median current of the full 231 120-row tables to within 0.0001 A,
which the provenance note records.

### Raw data that is NOT staged, and where it lives

| what | size | where |
|---|---|---|
| The bare-die threshold sweep itself, 23 416 point directories of `TDC_Data_translated_*.nem` | ~16 GB | `root://cmseos.fnal.gov//store/group/lpcmtdstudies/cern_Jan2024/TID/<day>/ET2p01_W36_6_VRef_SCurve_TDC_Pixel_*_Threshold_*/` |
| The hybrid raw tree | ~12 GB | `root://cmseos.fnal.gov//store/group/lpcmtdstudies/cern_Jan2024/CC/`; a working copy is on LPC at `/uscms_data/d3/murtazas/scratch/qinj_cc/` |
| `PowerHistory_v2.sqlite` (two files) | 15 MB | in the hybrid raw tree; `aux_lv_timeseries.csv.gz` is its reduction |
| `BaselineHistory_CC_Jan2024_CERN.sqlite` | 3.2 MB | with the LPC working copy; `aux_baseline_history.csv.gz` is its reduction |
| The ET2.03 working copy (`TID_August`: 0 MRad money plots + all 94 rail scans, undecoded binary) | 6.2 GB | `fermilab-etroc3:/home/daq/tmp/TID_August/`; LPC copy at `/uscms_data/d3/murtazas/scratch/qinj_tid_et2p03/raw/` |
| The ET2.03 dosed money plots (six `2024-08-*` day directories, undecoded binary) | 25.1 GB | the removable exFAT disk `X9`, `et2p03_qinj_data/`; LPC copy at `…/qinj_tid_et2p03/raw_x9/` |
| `2024-08-04` — `BARE9` 0 MRad at VA 1.00/1.20/1.40 V plus a room-temperature control run | 5.86 GB | **`X9` only**, not copied |
| the reduced points feather, tables, reducer/decoder and `DECODE_README` | 6 MB | **also on LPC EOS** at `/eos/uscms/store/user/lpcmtdstudies/cern_Aug2024_TID/reduced_qinj_feather/` (2026-09-04), with the hybrid CC reduction under `cern_Jan2024/CC/reduced_qinj_feather/` and the bare ET2.01 derived points under `cern_Jan2024/TID/reduced_qinj_feather/` |
| `2024-08-14` — `BARE7` 0 MRad money plots | 8.88 GB | on `X9`, **now also copied to LPC** at `…/qinj_tid_et2p03/raw_x9/2024-08-14_Array_Test_Results` and decoded — it is what gives BARE7 a 0 → 200 MRad dose step |

## Not recorded by any of the three campaigns

**RFSel** (sets the preamplifier feedback resistance, and therefore gain and TOT — the single
largest unquantified systematic in all three notebooks), **power mode** on both bare-die sides,
on-chip temperature (`VTemp`), any chamber temperature beyond a directory tag, sensor bias on the
hybrid −25 °C series, injections per point, and — on the hybrid — the scan windows, because every
directory modification time in that tree is an upload or copy time rather than an acquisition
time.

**The August campaign additionally has no monitoring at all**: no low-voltage supply log, no
temperature log, no `PowerHistory` sqlite, no baseline history. Its only record of the supply
currents anywhere is one sentence of prose in `info.txt`. Its `TS_PD = 0` register field says the
on-chip temperature sensor had power, not that it was read, and no `VTemp` value is stored.

**The chip revisions differ, and so does the voltage reference.** Notebooks 01 and 02 are
**ET2.01** driving the threshold DAC from an external 1.0 V reference; notebook 03 is **ET2.03**
driving it from the chip's own generator (`VrefGen_PD = 0` on all three surviving register dumps —
which proves the generator was powered up, not that it was used, since internal versus external is
board wiring with no register bit). The 2.01 → 2.02 revision fixed that on-chip reference, which
had read about 167 mV low. **So the absolute DAC-to-millivolt scale is not guaranteed to carry
between the notebooks**, and notebook 03 shows directly that on ET2.03 it also moves with the
analog supply. That is why every figure in this folder uses threshold **offset above baseline**
rather than absolute DAC, and why notebook 04 labels each comparison with whether it is scale-free
or not.

---

## Environment, and how to execute

The notebooks need only `numpy`, `pandas`, `pyarrow`, `scipy`, `matplotlib` and `mplhep`, plus
`jupyter`/`nbconvert` to run headless. They read only the files under `data/` — nothing in them
touches EOS, AFS or any remote system.

### On lxplus (the repo's own environment) — verified

```bash
cd /afs/cern.ch/user/m/musafdar/CMS_ETROC/ETROC-Analysis
source TestBeam/condor_at_lxplus/envs/load_python39.sh
cd qinj_studies
jupyter nbconvert --to notebook --execute --inplace 00_census_and_aux_data.ipynb
jupyter nbconvert --to notebook --execute --inplace 01_bare_et2p01_tid_jan2024.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_hybrid_et2p01_cc_jan2024.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_bare_et2p03_tid_aug2024.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_cross_campaign_synthesis.ipynb
```

**Nothing needs installing.** The repo's prescribed environment (`load_python39.sh`, LCG_104d,
Python 3.9.12) already provides every dependency — pandas 1.5.3, pyarrow 11.0.0, numpy 1.23.5,
scipy 1.10.1, matplotlib 3.7.1, mplhep 0.3.12, nbformat 5.9.1, nbconvert 7.6.0, ipykernel 6.0.2.
All five notebooks were executed end to end there on 2026-09-04, after the CAL-window
re-reduction, in 28, 69, 96, 93 and 28 s: 00, 01, 02, 03, 04. Every ledger value and every quoted
number matched the laptop run, despite the older pandas, numpy and mplhep. Two harmless
differences remain between the two platforms and neither is numerical: the rendered PNG/PDF bytes
differ (matplotlib version), and `outputs/et2p03_tot_vs_charge_offsets.csv` comes out in a
different **row order**, because `groupby` orders its groups differently in pandas 1.5.3 — sorted
on `(config, offset, charge_fC)` the two files agree to zero relative difference. If a future LCG
view were ever to drop one of these packages, the fallback is
`python -m pip install --user <package>` — but it is not needed today.

To execute without touching the committed copies, copy the folder to `$TMPDIR` first and run there
(the notebooks write into `figures/` and `outputs/` relative to `qinj_common.py`), or use
`--output /tmp/test_nb.ipynb` instead of `--inplace` and accept that `figures/` and `outputs/`
will be rewritten in place.

### Locally

Any Python 3.9+ with the six packages above:

```bash
python -m pip install jupyter nbconvert ipykernel nbformat pandas pyarrow scipy matplotlib mplhep
cd qinj_studies
jupyter nbconvert --to notebook --execute --inplace 00_census_and_aux_data.ipynb
jupyter nbconvert --to notebook --execute --inplace 01_bare_et2p01_tid_jan2024.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_hybrid_et2p01_cc_jan2024.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_bare_et2p03_tid_aug2024.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_cross_campaign_synthesis.ipynb
```

### Editing the notebooks

The committed `.ipynb` files carry their outputs so they can be read without running anything. The
authoring source is the percent-format `_src_00.py` … `_src_04.py`, which is far easier to review
and diff than notebook JSON:

```bash
python _build_nb.py _src_01.py 01_bare_et2p01_tid_jan2024.ipynb
jupyter nbconvert --to notebook --execute --inplace 01_bare_et2p01_tid_jan2024.ipynb
```

Keep inline figures at `dpi=110` (`qinj_common.save_fig` does) so the executed notebooks stay
under about 6 MB each; the print-quality copies go to `figures/*.pdf`. Figure names are prefixed by
notebook: `census_*`, `bare_*`, `hyb_*`, `et2p03_*`, `xcamp_*`, and so are the tables in
`outputs/`.

Executing a notebook **rewrites `figures/` and `outputs/` in place**, which is normally what you
want. To execute without touching the committed copies, copy the folder to `$TMPDIR` first and run
there — the notebooks resolve `figures/` and `outputs/` relative to `qinj_common.py`.

## Conventions

* Thresholds are plotted as **`offset = dac − BL`** everywhere. The beam operating point is
  `threshold = BL + 20`, and the offset axis survives a chip-revision change where absolute DAC
  does not. `DAC LSB = 400 mV / 2¹⁰ = 0.390625 mV`.
  On ET2.03 that LSB is an **assumption**, not a measurement: the threshold DAC there is driven
  from the chip's own reference generator, and notebook 03 shows the DAC scale moving with the
  analog supply. Every millivolt number in notebook 03 carries that caveat, and notebook 04 marks
  each cross-revision comparison as scale-free or not.
* **No point, pixel, charge or condition is cut at any stage** in any notebook. What varies is
  *coverage* — a low charge simply turns off below BL+20 on some pixels — so every pooled quantity
  reports its own `n`, and the figures annotate partial coverage. The one place a point is set
  aside is stated as such: notebook 03's twenty DAC-ceiling turn-offs are lower limits rather than
  measurements, and its gain fits are quoted both with and without them.
* **There is exactly one per-hit selection in the folder, and it is a data-integrity criterion.**
  The ET2.03 reduction carries a **CAL window**: a hit counts towards a point's moments only if
  `|CAL − cal_mode| ≤ 3` codes against that pixel's modal CAL at that point — the repository test
  beam pipeline's own criterion (`core/path_finder.py`, `core/extract_events_by_path.py`). One TOA
  code is `3.125 ns / CAL`, so a hit with a wrong CAL carries a wrong TOA by construction, and on
  this campaign a small fraction of hits do: in contiguous blocks of one to six threshold DAC
  values at a **fixed absolute DAC, the same block at every injected charge** — a decode artefact,
  since the sweep's inner loop is charge at fixed threshold. It costs 2.68 % of the money-plot
  hits; 19.4 % of money-plot points carry `cal_std > 2` codes on the raw moments and notebook 03
  carries that flag as notebook 02 already did. **The window does not catch everything**: a
  smaller, TOA-only class survives it — CAL clean after windowing, TOA still corrupt — and is
  flagged (`toa_rms_gt3`, `toa_spike_50ps`), counted in notebook 03 section 4.4 and ringed on
  `et2p03_fig4_money_plot_per_pixel`, never cut. It is 585 of 24 882 points at offset ≥ 10 and
  13 of 378 at BL+20, isolated points rather than blocks: in **98.9 %** of (condition, charge,
  offset) groups the six-pixel median is taken over a majority of unflagged pixels, so the money
  plot and the BL+20 numbers are safe at that level — the notebook computes that fraction rather
  than asserting it. The reduction writes **both** sets of moments, the
  raw ones are staged under a `_raw` suffix, and notebook 03 section 4.4 draws the before and
  after with the criterion stated. The bare ET2.01 data needs no such window — **not one** of its
  22 055 points has `cal_std > 2`, the largest being 1.41 codes — while the hybrid has 865 of
  13 281; notebook 02 carries those as a `cal_rms_gt2` flag rather than windowing them, because
  that campaign's staged table is per-point moments with the hits behind them not staged.
* **The pixel-to-pixel spread is the dominant uncertainty** on every chip (33 % on the bare
  ET2.01, 18 % on the hybrid, 35 % on the bare ET2.03, on 12, 9 and 6 pixels). A single chip-level
  gain is close to meaningless; the per-pixel values are in `outputs/*_gain_fits.csv`. Notebook 03
  is the only campaign that measured the **same six pixels at every condition**, which is what lets
  it form a paired per-pixel ratio and a measured repeatability floor of ±0.7 % on the gain.
* **The money plot's high-charge branch is a floor measurement, not a resolution.** The injection
  is synchronous with the clock and, bare or bonded, the sensor is not generating the signal, so
  above about 8 fC the measured jitter closes on the TDC quantisation floor `LSB/√12` — within a
  factor of 1.18, 2.02 and 1.69 of it at 15 fC on the three data sets, per
  `outputs/xcamp_jitter_at_bl20.csv`. It must not
  be compared with a beam per-board time resolution. The physics lives in the low-charge rise and
  in the threshold-edge width. The floor itself differs between the data sets, because one TOA code
  is `3.125 ns / CAL`: 5.37, 5.24 and 4.88 ps on the bare ET2.01, the hybrid and the bare ET2.03.
* **Notebook 04 recomputes nothing.** It reads the tables 01, 02 and 03 wrote, so both sides of
  every comparison went through the same functions. If any of those three is re-run with a changed
  definition, 04 must be re-run too.
