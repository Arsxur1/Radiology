# Clinical references and encoded criteria

CardioLens encodes established ECG diagnostic criteria as **data** — in the
age/sex norm tables (`configs/norms/`) and the pathophysiology rule base
(`configs/rules/pathophysiology.yaml`) — not as hard-coded logic (TZ §0, §8).
This file records the clinical sources those criteria are drawn from and maps
each encoded criterion to its source.

> **Honesty note.** The references below were provided as learning material; they
> were **not** fetched or scraped. The criteria encoded here are **standard,
> well-established textbook ECG values** (medical facts, not copyrighted expression)
> that these sources teach; nothing was copied verbatim. Where a criterion is
> threshold-based it lives in a config file and can be edited to match a given source
> or local protocol.

## Sources

- **ECG Library** — ecglibrary.com (Dean Jenkins & Stephen Gerred), a standard
  online teaching reference for 12-lead ECG diagnoses.
- **150 ECG Problems** (Hampton), case-based ECG interpretation.
- **Master Visual Diagnosis of ECG — a short atlas**, pattern-based ECG atlas.
- Wiley ECG textbook front-matter (onlinelibrary.wiley.com, 10.1002/9781119536475).
- **ECG-Image-Database** — Reyna et al., *arXiv:2409.16612* — dataset of ECG
  images with real-world imaging/scanning artifacts; the foundation for the
  PhysioNet 2024 image-digitization challenge. Relevant to the paper-ECG
  digitiser (`layer1_ingestion/digitize.py`): the intended benchmark for
  hardening digitisation against dirty photos in v2.

## Encoded criteria → source

| Criterion (config) | Value | Basis |
|--------------------|-------|-------|
| First-degree AV block: PR | > 200 ms (adult) | standard |
| QRS wide | > 110–120 ms | standard |
| QTc upper (Bazett) | 450 ms male / 460 ms female | standard |
| LVH (Sokolow-Lyon SV1+RV5/6) | > 3.5 mV | Sokolow-Lyon |
| Low QRS voltage (limb) | < 0.5 mV | standard |
| Left-axis / right-axis deviation | axis < −30° / > +90° | standard |
| **Left atrial enlargement (P-mitrale)** | P duration > 120 ms | ecglibrary / 150 ECG Problems |
| **Right atrial enlargement (P-pulmonale)** | P amplitude > 0.25 mV (lead II) | ecglibrary / 150 ECG Problems |
| **Pericarditis vs STEMI** | diffuse ST elevation across ≥2 territories, no reciprocal (+ PR depression) → pericarditis; focal single-territory → STEMI | classic STEMI-mimic teaching |
| ST-elevation territory → coronary artery + reciprocal | inferior/anterior/lateral | standard |
| Pediatric juvenile T-inversion (V1–V4) is normal | age-conditional | standard pediatric caveat |
| **Hyperkalemia chain** | peaked T → wide QRS → sine wave; K⁺ ≥ 5.5 | standard |
| **Hypokalemia chain** | flat T, U waves, ST depression, long QT; K⁺ ≤ 3.5 | standard |
| **Hypocalcemia** | long QT via prolonged ST; Ca²⁺ ≤ 2.1 mmol/L | standard |
| **Hypercalcemia** | short QT via shortened ST; Ca²⁺ ≥ 2.6 mmol/L | standard |
| LVH strain: secondary ST-T, not primary ischemia | — | standard |
| BBB / WPW explain a wide QRS (not hyperkalemia) | RBBB by V1 terminal sign; WPW by short PR | standard |
| LA/RA electrode reversal → artifact hypothesis | lead I inverted + aVR upright | standard |
| **QT end (tangent method), QTc formula** | steepest post-peak tangent ∩ baseline; record QT = 75th percentile of per-lead medians; Fridericia default (`qtc_ms.formula`) | Postema & Wilde 2014 (tangent); Fridericia; D27 |
| **STEMI thresholds V2–V3** | J ≥ 2.5 mV·10⁻¹ men < 40, 2.0 men ≥ 40, 1.5 women; other leads 1.0 mm; ≥ 2 contiguous leads; plateau at J+60 | ESC/ACC/AHA universal MI definition |
| **Atrial flutter band** | atrial rate 200–350 bpm (`rhythm.atrial_flutter_bpm`), F-wave autocorrelation or spectral peak on ≥ 2 leads | standard; D24/D28 |
| **AF gates** | RR-CV ≥ 0.10 and organised-P fraction ≤ 0.5 (`rhythm.af_*`) | calibrated on 964 records, train/test halves (D31) |
| **Complete AV block** | dissociated P train, atrial/ventricular ratio ≥ 1.2, unstable PR, organised P | standard; gates from 16 PTB-XL cases (D28) |

Bold rows are the criteria added in the clinical-knowledge increment; the rest
predate it. Editing a threshold in `configs/norms/*.yaml` or adding a chain to
`configs/rules/pathophysiology.yaml` updates the behaviour without code changes.

## Validation datasets

- **PTB-XL** (physionet.org/content/ptb-xl) — 12-lead, SCP-ECG labels. Adapter:
  `validation.build_ptbxl_dataset`.
- **ecg-arrhythmia / Chapman-Shaoxing + Ningbo**
  (physionet.org/content/ecg-arrhythmia/1.0.0) — ~45k 10-second 12-lead records
  (WFDB `.hea`+`.mat`) with SNOMED-CT diagnostic labels. Adapter:
  `validation.build_ecg_arrhythmia_dataset`; SNOMED→finding mapping in
  `validation.SNOMED_TO_CODE`.

To validate, obtain a local checkout and point the harness at it:
`python -m cardiolens.validate --ecg-arrhythmia <local-root>` (or `--ptbxl`). The
dataset itself is licensed and is never committed (`data/ecg-arrhythmia/` is
git-ignored). If the environment's network policy allows `physionet.org`, a subset can
be fetched directly (see README).

### Measured results — ecg-arrhythmia (real 12-lead)

Run on a **300-record subset** (`WFDBRecords/01/010–012`, 500 Hz, SNOMED-labelled;
brady-skewed, one region — **not** representative of the full ~45k set). Real
numbers; the "before" column is the first run, "after" is the current code:

| Finding | Before (sens / spec) | After (sens / spec) | n |
|---------|:--------------------:|:-------------------:|--:|
| sinus_rhythm | 0.97 / 0.78 | 0.95 / 0.93 | 39 |
| sinus_bradycardia | 0.94 / 0.98 | 0.96 / 0.97 | 155 |
| sinus_tachycardia | 1.00 / 0.91 | 0.97 / 0.96 | 30 |
| atrial_fibrillation | 0.62 / 0.94 | 0.62 / 0.95 | 61 |
| atrial_flutter | 0.00 / 1.00 | 0.20 / 1.00 | 15 |
| first_degree_av_block | 0.09 / 0.95 | 0.83 / 0.95 | 12 |
| rbbb | 0.18 / 0.98 | 1.00 / 0.99 | 11 |
| lbbb | 0.17 / 0.79 | 0.50 / 0.98 | 6 |
| left_axis_deviation | 1.00 / 0.94 | 1.00 / 0.94 | 10 |
| right_axis_deviation | 0.75 / 0.98 | 0.75 / 0.98 | 4 |
| low_qrs_voltage | 0.86 / 0.89 | 0.86 / 0.90 | 7 |
| st_elevation | 0.67 / 0.76 | 0.67 / 0.99 | 3 |
| st_depression | 0.83 / 0.67 | 0.75 / 0.97 | 12 |
| long_qt | 1.00 / 0.66 | 0.00 / 0.98 | 1 |

Macro sensitivity 0.65 → **0.72**. (PR/QT interval MAE is not reported for this
dataset — it carries diagnostic labels only, no gold interval annotations.)

**What real data revealed, and what was done about it:**

- **Atrial fibrillation** initially scored 0.0: neurokit's delineation interpolates a
  P onset for nearly every beat, so the onset-only `p_wave_ratio` saturated at 1.0.
  Fixed by gating P-presence on the P-region amplitude (`P_PRESENCE_FRACTION`,
  calibrated here: sinus P/R ≈ 0.21 vs AF ≈ 0.10) → 0.62 at 0.95 specificity. The
  remaining misses are AF with a controlled, near-regular ventricular response
  (RR-CV below the 0.15 gate).
- **Bundle branch block** read as *narrow* QRS: dwt under-estimates width when the
  terminal forces are slurred (RBBB measured 62–74 ms). QRS bounds now come from the
  multi-lead spatial-velocity envelope (`layer3_measure/delineation.py`,
  `QRS_ENVELOPE_FRACTION = 0.05`, calibrated on this set: RBBB/LBBB 112–190 ms,
  normal ≤ 102 ms) → RBBB 1.00. LBBB stays at 0.50 partly because the dataset gives
  the fascicular blocks (LFBBB/LBBBB, narrow QRS) the same SNOMED code as complete
  LBBB — label noise, not a detector miss.
- **First-degree AV block**: dwt P onsets land late. P onsets are now refined by
  amplitude (closest supra-threshold run before the QRS, smoothed 15 % onset
  criterion; a fixed fallback offset was found to inflate PR by 60–80 ms and was
  removed) → 0.83 sensitivity at 0.95 specificity. PR is not reported in AF or
  under AV dissociation.
- **Atrial flutter & complete AV block** had rules but no measurement behind them.
  `layer3_measure/atrial.py` subtracts the average beat, blanks the QRS, band-limits
  the residual and reads its autocorrelation: an F-wave search (2–8 Hz) for flutter
  and a P-band search (1–6 Hz, 45–200 bpm) for dissociated P waves. Flutter with
  variable conduction is detected (3/15 real cases; the rest are fixed 2:1 at high
  ventricular rates or read as AF); a fixed k:1 ratio locks the F-waves to the QRS
  and is a documented miss (a second pass on a QRS-only residual was built, caught
  1 of 5 such cases and mimicked sinus P–T spacing as 3:1, so it was removed).
  Complete AV block is validated on synthetic stimuli only (no gold cases in this
  subset): 6 of 7 variants; an atrial rate within 10 % of k × ventricular rate is
  rejected by the harmonic guard on purpose — a false complete-AV-block is a
  critical alert, so the conservative side is chosen.
- **ST** was over-called (72 false elevations / 297): J point from dwt, a single
  baseline sample, one threshold for every lead, no contiguity, and secondary
  changes in BBB/LVH counted as ischemia. Now: J = envelope QRS offset, baseline =
  PR-segment median, sex/age-specific V2–V3 STEMI thresholds (ESC/ACC), elevation
  must persist to J+60, depression judged at J+60/J+80 (upsloping excluded),
  `st_elevation` requires two contiguous leads of one territory (else
  `st_elevation_nonspecific`, never critical), discordant deviations under
  BBB/LVH/WPW become `secondary_st_t_change` — discordant elevation is never a
  STEMI, discordant depression stays reported (strain) and is explained by the
  causal chains. Elevation specificity 0.76 → 0.99, depression 0.67 → 0.97.
- **Sinus codes in AF/flutter** were the main reason for sinus_rhythm's 0.78
  specificity: a detected non-sinus rhythm now replaces the sinus_* rate codes and
  suppresses P-wave findings.
- **R-peak over-sensing** at slow rates (a tall T or a P wave after a long pause
  counted as a beat) is filtered; the synthetic 40-bpm case and the real
  bradycardia class both improved.
- **long_qt**: the single labelled case now measures QTc ≤ 450 with the earlier,
  more accurate QRS onset; specificity rose from 0.66 to 0.98 (QT is no longer
  inflated by a late onset). n = 1, so no conclusion either way.

### Measured results — balanced subset (664 records)

`fetch_physionet --folders 40 --per-class 80 --seed 0`: headers from 40 folders
spread over all 46 top-level directories (Chapman **and** Ningbo), 4000 records
screened, 664 selected (rarest classes first; 80 unlabeled negatives kept).
`manifest.json` lists every record and its labels.

| Finding | Sens | Spec | n |
|---------|-----:|-----:|--:|
| sinus_rhythm | 0.89 | 0.77 | 80 |
| sinus_bradycardia | 0.96 | 0.93 | 115 |
| sinus_tachycardia | 0.91 | 0.87 | 93 |
| atrial_fibrillation | 0.49 | 0.82 | 80 |
| atrial_flutter | 0.01 | 1.00 | 187 |
| *AF or flutter (one family)* | *0.49* | *0.96* | 267 |
| first_degree_av_block | 0.73 | 0.89 | 85 |
| rbbb | 0.73 | 0.98 | 71 |
| lbbb | 0.74 | 0.94 | 83 |
| left_axis_deviation | 0.81 | 0.86 | 83 |
| right_axis_deviation | 0.75 | 0.95 | 80 |
| low_qrs_voltage | 0.69 | 0.85 | 81 |
| lvh | 0.88 | 0.91 | 51 |
| wpw_preexcitation | 0.00 | 0.98 | 3 |
| st_elevation | 0.40 | 0.98 | 58 |
| st_depression | 0.48 | 0.82 | 85 |
| long_qt | 0.17 | 0.93 | 23 |
| complete_av_block | — (no gold cases) | 1.00 | 0 |

Macro sensitivity **0.60** — lower than on the 300-record slice (0.72) because the
set is harder and more diverse, not because the code regressed (the 300-set
numbers above are from the same code). What the misses are:

- **AF vs flutter.** 91 of the 107 AF false positives are flutter-labelled records:
  flutter with an irregular ventricular response and no organised P is what the AF
  rule describes. None of the 187 flutter records is recognised as flutter (median
  RR-CV 0.21 → mostly irregular; the F-wave autocorrelation gates are not met), 91
  are called AF and 96 nothing. Treated as one family the detector is precise
  (spec 0.96, 132 TP / 16 FP) but catches half (sens 0.49). AF misses: 21 records
  with irregular RR but coarse f-waves that pass the P-amplitude gate, 20 with a
  near-regular response (RR-CV ≤ 0.15).
- **Long QT** fails on measurement, not thresholds: for many gold records the
  measured QT is implausibly short (196–290 ms) because the dwt T offset lands
  early on the rhythm lead. Next step: a multi-lead T offset (tangent method on
  the lead with the longest QT).
- **ST elevation.** The dataset label "ST tilt up" is broader than the STEMI
  definition the alert implements: the misses are discordant elevation under LVH
  (reported as `secondary_st_t_change`), J-point elevation with no plateau at J+60
  (early-repolarisation-like) and V2–V3 elevation under the sex/age threshold. The
  alert keeps its precision (spec 0.98); a broader "any ST elevation" descriptor
  could be added as a non-critical finding if the label semantics are wanted.
- **ST depression** 0.48 / 0.82 ("ST drop down" is likewise broad); not analysed
  further in this increment.
- **Complete AV block.** The first balanced run produced one false positive (sinus
  rhythm with first-degree AV block and long QT: a periodic P-band residual at 112
  bpm). A physiologic gate closed it — AV dissociation means no fixed PR, so a
  candidate with a refined P onset on ≥ 50 % of beats and a beat-to-beat PR
  standard deviation < 40 ms (`pr_std_ms`) is conducted, not dissociated. 0 false
  positives in 664 afterwards; synthetic complete AV block still detected. No 3AVB
  label exists in the 4000 screened headers, so real-data sensitivity is unknown.
- **LBBB** improves from 0.50 to 0.74 here (the hemiblock share of the LBBB code
  is smaller in this sample); **LVH** 0.88 / 0.91 on a Sokolow-Lyon criterion.



The adapters and SNOMED/SCP mappings are unit-tested; the balanced fetch
(`cardiolens.validation.fetch_physionet`) exists precisely because this 300-record
slice is brady-skewed. Rerun on a larger sample before drawing clinical conclusions.

### Measured results — balanced 664 after items 0–3 and the conducted-P gate (rerun)

Same 664 records, current code (tangent T offset + Fridericia, spectral flutter
search, AF gates from the norm table, `st_elevation_any`, conducted-P veto). The
first-run column is the table above.

| Finding | First run (sens / spec) | Current (sens / spec) | n |
|---------|:-:|:-:|--:|
| sinus_rhythm | 0.89 / 0.77 | 0.89 / 0.84 | 80 |
| sinus_bradycardia | 0.96 / 0.93 | 0.93 / 0.98 | 115 |
| sinus_tachycardia | 0.91 / 0.87 | 0.91 / 0.89 | 93 |
| atrial_fibrillation | 0.49 / 0.82 | 0.80 / 0.73 | 80 |
| atrial_flutter | 0.01 / 1.00 | 0.01 / 1.00 | 187 |
| *AF or flutter (family)* | *0.49 / 0.96* | *0.77 / 0.95* | 267 |
| first_degree_av_block | 0.73 / 0.89 | 0.68 / 0.94 | 85 |
| rbbb / lbbb | 0.73 / 0.98 · 0.74 / 0.94 | 0.72 / 0.98 · 0.71 / 0.94 | 71 / 83 |
| left / right axis deviation | 0.81 / 0.86 · 0.75 / 0.95 | 0.87 / 0.86 · 0.69 / 0.96 | 83 / 80 |
| low_qrs_voltage | 0.69 / 0.85 | 0.72 / 0.86 | 81 |
| lvh | 0.88 / 0.91 | 0.90 / 0.91 | 51 |
| st_elevation (alert) | 0.40 / 0.98 | 0.38 / 0.98 | 58 |
| *st_elevation_any (family)* | — | *0.81 / 0.84* | 58 |
| st_depression | 0.48 / 0.82 | 0.48 / 0.82 | 85 |
| long_qt | 0.17 / 0.93 | 0.70 / 0.86 | 23 |

Macro sensitivity 0.60 → **0.65**. Reading:

- **Long QT** 0.17 → 0.70 is the tangent-method T offset aggregated over leads
  (D27); the 23 gold records no longer measure 196–290 ms. Specificity fell 0.93 →
  0.86 — the 75th-percentile aggregate over-reads some normals; the maximum over
  leads (textbook "longest lead") was worse (spec 0.44–0.57) and the median missed
  half the cases, so this is the chosen trade-off, made on the train half.
- **AF** 0.49 → 0.80 at 0.73 specificity: the gates moved from (0.15, 0.30) to
  (0.10, 0.5) (calibration table below). Most of the remaining false AF are
  flutter-labelled records (family spec 0.95), so the per-class specificity
  understates the detector. The conducted-P veto (D36) cost 5 points of AF
  sensitivity here (0.85 → 0.80) and bought 2 points of family specificity; it was
  introduced for children, where it removed two thirds of the false AF.
- **Flutter** stays at 0.01 on the Ningbo "flutter" label: the spectral F-wave
  search finds no 3.3–6 Hz peak on ≥ 2 leads in these records, while it does on
  PTB-XL AFLT (0.51). Either the label is broad here or the F-waves are not
  visible at 500 Hz surface leads with an irregular response; reported as is.
- **First-degree AV block** 0.73 → 0.68 / 0.89 → 0.94: the AF/flutter/dissociation
  gates suppress PR in more records; the trade favours specificity.
- **ST elevation (alert)** unchanged at spec 0.98; the neutral `st_elevation_any`
  descriptor (D32) covers 0.81 of the "ST tilt up" label at 0.84 specificity —
  the first version without the J+60 plateau fired on half the set (spec 0.51).

### Pediatric stratum — PTB-XL, 64 records aged ≤ 15 (item 5b)

`validate --ptbxl data/ptbxl --age-max 15`. All 64 children of PTB-XL are in the
local subset; the pediatric norm table (`configs/norms/pediatric.yaml`) applies to
every one of them (HR bands by age, PR 90–170, QTc 460, wide normal axis, juvenile
T pattern). Labels: NORM 54, SR 39, SARRH 20, SBRAD 4, STACH 2, IRBBB 6, LVH 5,
WPW 2, LNGQT 1, 1AVB 1, one inferior MI pattern.

| Finding | Sens | Spec | n |
|---|:-:|:-:|--:|
| sinus_rhythm (SR ∪ SARRH) | 0.86 | 0.67 | 58 |
| sinus_bradycardia | 1.00 | 0.95 | 4 |
| sinus_tachycardia | 0.00 | 1.00 | 2 |
| atrial_fibrillation | — | 0.92 | 0 |
| first_degree_av_block | 1.00 | 0.91 | 1 |
| wpw_preexcitation | 0.50 | 1.00 | 2 |
| right_axis_deviation | 0.00 | 1.00 | 2 |
| lvh | 0.00 | 1.00 | 5 |
| long_qt | 0.00 | 1.00 | 1 |
| complete AV block, flutter, BBB, ST codes | — | 0.86–1.00 | 0 |

What the stratum taught, in order of importance:

- **Sinus arrhythmia read as AF** — the biggest finding of this exercise. In the
  first run 15 of 64 children were called AF: marked respiratory RR variation
  (RR-CV 0.10–0.21) plus a P/R amplitude ratio near zero (tall pediatric R waves
  defeat the adult-calibrated `P_PRESENCE_FRACTION` gate), while a refined P onset
  existed on 92–100 % of beats with a stable PR (SD 8–20 ms). The AF rule now
  vetoes on conducted P (D36): false AF 15 → 5, sinus_rhythm sensitivity 0.69 →
  0.86. The 5 remaining are records where the P onset is not found or the PR is
  unstable (SD 55–95 ms).
- **SARRH is sinus rhythm.** 20 of the 25 children without an SR label carry SARRH
  (sinus arrhythmia); the adapter now maps it to `sinus_rhythm`. Before that the
  stratum showed a meaningless 0.32 sinus-rhythm specificity.
- **Sinus_rhythm specificity 0.67** is now label semantics: the only children
  without a sinus-rhythm label are the 6 with SBRAD/STACH by adult thresholds,
  and 4 of them fall inside the pediatric HR band (60–110 at 9–15 y), which calls
  them normal;
  the two STACH children (HR ≈ 105–110) are the mirror image (sens 0.00 for
  sinus_tachycardia). The pediatric norms are doing what they should.
- **LVH 0 / 5**: the pediatric norm table has no voltage criterion for LVH
  (Sokolow-Lyon 3.5 mV is an adult criterion; children need Davignon/Rijnbeek
  age-band percentiles), so `lvh` cannot fire. Deliberately not filled in with the
  adult number.
- **WPW 1 / 2, RAD 0 / 2, long QT 0 / 1**: too few to conclude; the RAD misses are
  the intentionally wide pediatric normal axis (−30° … 180°).

Pediatric **fine-tuning** (TZ §17) was not done and is not claimable here: no
labelled pediatric training set exists in the reachable data (64 children in PTB-XL,
a handful in ecg-arrhythmia). What it needs: a pediatric dataset with ages
(e.g. a children's-hospital 12-lead set), age-band norm tables for voltage/axis/QT,
and a stratified hold-out per band. Until then the system relies on the
rule-based pediatric norm table plus this evaluation.

### Calibration of the AF gates — train/test halves (item 2)

The AF rule fires on `rr_cv >= rhythm.af_rr_cv_min` and `p_wave_ratio <=
rhythm.af_p_ratio_max` (both in `configs/norms/*.yaml`). The old values (0.15 / 0.30)
were tuned on the 300-record slice. The new ones were chosen on the **train half**
of 300 + 664 ecg-arrhythmia records (split by record-name hash), objective = Youden
index of the AF-or-flutter family with a family-specificity floor of 0.93, then
scored on the untouched **test half** and on PTB-XL. Family = AF rule ∪ flutter
detector vs gold AFIB ∪ AFLT.

| Gates (cv / p_ratio) | ecg-arrhythmia train (n+ 162) | ecg-arrhythmia test (n+ 181) | PTB-XL (n+ 179 / n− 1513) |
|---|:-:|:-:|:-:|
| 0.15 / 0.30 (old) | 0.54 / 0.97 | 0.52 / 0.97 | 0.50 / 0.97 |
| **0.10 / 0.50 (chosen)** | 0.81 / 0.94 | **0.78 / 0.94** | **0.64 / 0.92** |

Train and test agree within 0.03, so the gates did not over-fit the half they were
chosen on. The cost is 3–5 points of family specificity, spent knowingly: the
alternative (missing half of the AF) is the worse clinical error, and AF is not a
critical alert in this system.

### Paper-page digitisation benchmark (item 5d)

Synthetic 12-lead pages (3×4 layout + lead-II rhythm strip, 6 px/mm, 25 mm/s,
10 mm/mV) rendered by `render_ecg_page`, degraded by `degrade_image`, recovered by
`digitize_page`; per-lead Pearson r between the 2.5-s panel trace and the source:

| Degradation | mean r (12 leads) | min r |
|---|:-:|:-:|
| clean scan | 0.93 | 0.89 |
| rotation +3° | 0.93 | 0.89 |
| Gaussian blur 2 px | 0.91 | 0.87 |
| JPEG q = 40 | 0.93 | 0.89 |
| shadow gradient 50 % | 0.93 | 0.89 |
| salt noise 0.5 % | 0.93 | 0.89 |
| all of the above (−2°, blur 1 px, JPEG 40, shadow 40 %, salt 0.3 %) | 0.92 | 0.89 |

The ceiling (~0.93) is the render itself: 150 samples/s and a 2-px stroke quantise
the QRS. Downstream on a degraded STEMI page: HR within 3 bpm, QRS axis within 15°,
`st_elevation` preserved (`tests/test_digitize_page.py`). Real photographs and the
calibration pulse remain untested/unimplemented; `provenance.confidence` is the panel
column coverage, so a bad photo lowers confidence rather than inventing a trace.

## Not a medical device
These criteria support interpretation for a decision-support tool; CardioLens is
not clinically validated and is not a medical device. Every report carries the
disclaimer "Инструмент поддержки. Не заменяет врача."
