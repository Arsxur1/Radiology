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
| *st_elevation_any (family)* | — | *0.83 / 0.82* | 58 |
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
  descriptor (D32, thresholds revised in D37) covers 0.83 of the "ST tilt up" label
  at 0.82 specificity — the first version without the J+60 plateau fired on half
  the set (spec 0.51).

### Measured results — PTB-XL (1692 records, item 4)

`fetch_ptbxl` selects from `ptbxl_database.csv` (SCP codes with likelihood ≥ 50,
rhythm codes at likelihood 0 are labels too, `heart_axis` LAD/RAD): rare classes in
full (3AVB 16, 2AVB 14, AFLT 73, WPW 79, LNGQT 117, STE_ 28, LVOLT 182), 80 per
common class, 80 unlabeled normals, plus every record aged ≤ 15 (64). 500 Hz
`records500/*_hr`. Lead names AVR/AVL/AVF are canonicalised on ingestion (D30) —
before that the axis and ST-territory logic silently ran on 9 leads.

| Finding | First run | Current (sens / spec) | n |
|---|:-:|:-:|--:|
| sinus_rhythm (SR ∪ SARRH) | — | 0.83 / 0.78 | 1135 |
| sinus_bradycardia / sinus_tachycardia | — | 0.96 / 0.88 · 0.92 / 0.96 | 106 / 110 |
| atrial_fibrillation | 0.41 | 0.55 / 0.97 | 123 |
| atrial_flutter | 0.23 | 0.51 / 0.997 | 73 |
| *AF or flutter (family)* | — | *0.59 / 0.97* | 179 |
| **complete_av_block** | 1 / 16 | **7 / 16 (0.44) / 0.996** | 16 |
| first_degree_av_block (1AVB ∪ LPR) | — | 0.63 / 0.92 | 188 |
| rbbb / lbbb | — | 0.85 / 0.93 · 0.57 / 0.88 | 86 / 85 |
| left / right axis deviation | 0.00 · 0.89 | 0.36 / 0.84 · 0.89 / 0.96 | 305 / 93 |
| low_qrs_voltage | — | 0.74 / 0.91 | 182 |
| lvh | — | 0.35 / 0.97 | 198 |
| wpw_preexcitation | — | 0.35 / 0.98 | 79 |
| st_elevation (alert) | — | 0.00 / 0.97 | 28 |
| *st_elevation_any (family)* | — | *0.61 / 0.79* | 28 |
| st_depression | — | 0.23 / 0.80 | 107 |
| long_qt | — | 0.49 / 0.89 | 117 |

Macro sensitivity 0.44 → **0.57**. Reading, by clinical weight:

- **Complete AV block, 16 real cases** (the headline check of item 4): 1 → 7 found,
  one false positive in 1676 negatives. The AV-dissociation gates were rewritten on
  these records (D28): 9 of 16 have an irregular ventricular response, so
  regularity is no longer required; atrial/ventricular ratio ≥ 1.2; "P rate ≈ k ×
  HR" is allowed when the PR is unstable (a 2:1 block has a constant PR and is
  rejected on purpose — it is not complete). The 9 misses are records where the
  P-band residual has no dominant period (low P amplitude, wide QRS escape with
  large T) — a critical alert stays on the conservative side.
- **Flutter with a regular response** 0.23 → 0.51 at 0.997: the spectral F-wave
  search (Welch PSD of the QRS-blanked signal, sharp 3.3–6 Hz peak on ≥ 2 leads
  with a band-power ratio ≥ 1 vs 0.5–3 Hz). Thresholds were chosen here (AFLT
  0.63 / AFIB 0.02 / SR 0.00 for the detector alone); the looser setting caught
  0.86 but flagged 9 % of sinus records.
- **AF** 0.41 → 0.55 at 0.97 specificity: the norm-table gates (D31) took it to
  0.61 / 0.92, the conducted-P veto (D36) trades 6 points of sensitivity for 4.5
  of specificity on this set — the veto is kept because it removes two thirds of
  the pediatric false AF and AF is not a critical alert.
- **Left-axis deviation 0.00 → 0.36**: the 0.00 was the lead-name bug (aVL/aVF
  absent → axis undefined). 0.36 / 0.84 against PTB-XL's `heart_axis` field, which
  its own algorithm assigns; not investigated further.
- **ST elevation**: none of the 28 STE_ records reaches the STEMI alert — and
  should not: STE_ is "non-specific ST elevation" (13 of 28 otherwise NORM), not an
  MI label. The descriptor `st_elevation_any` went from 0.18 to 0.61 at unchanged
  specificity once it stopped inheriting the STEMI sex/age V2–V3 thresholds (D37,
  analysis in its own section below).
- **LVH 0.35 / 0.97** on Sokolow-Lyon alone (PTB-XL's LVH is a mixed voltage +
  strain label); **WPW 0.35**: the short-PR + delta rule misses intermittent
  pre-excitation; **long QT 0.49 / 0.89** with the tangent method (LNGQT includes
  borderline QTc).

### ST elevation on PTB-XL STE_ — what the label is and what was changed (D37)

`STE_` is a PTB-XL *form* statement ("non-specific ST elevation", likelihood 0 like all
form/rhythm codes; 28 of 1692 records, 13 of them otherwise NORM, 2 with digitalis, 1
paced, 2 with LVH). It is not a STEMI label — PTB-XL has essentially no acute-MI
labels in this subset (AMI 1, ASMI 2) — so the STEMI-grade alert's 0.00 sensitivity on
it is not the failure it looks like; the right target is the neutral descriptor
`st_elevation_any`, which reached only 0.18.

Per-lead measurements of the 28 records: the elevation sits in V2 (25 records) and V3
(18), J-point 50–190 µV, J+60 100–430 µV — i.e. 1–2 mm below the sex/age STEMI
thresholds for V2–V3 (150 / 200 / 250 µV) that the descriptor had inherited.

Is the measurement biased? The PR-segment baseline lies 15–55 µV *below* the TP
baseline in V2–V3 (atrial repolarisation), and 60 % of all records show ≥ 1 mm at
J+60 in V2 over the PR baseline (median 96 µV) — the normal upsloping ST of the
right precordial leads. Re-measuring every record under three baselines:

| Baseline | descriptor J ≥ 100 & J+60 ≥ 100, concordant | alert proxy spec | V2 J / J+60 median (negatives) |
|---|:-:|:-:|:-:|
| PR segment [−50, −10] ms (current) | 0.57 / 0.88 | 0.971 | 32 / 96 |
| PQ junction [−24, −4] ms | 0.54 / 0.89 | 0.974 | 30 / 94 |
| TP segment [−300, −200] ms | 0.43 / 0.91 | 0.980 | 7 / 66 |

The baseline only slides along the same sensitivity/specificity curve, so the PR
segment stays (it is the reference of the universal MI definition). The label is
subjective and rare, and cannot be separated cleanly from normal-variant V2–V3
elevation by J / J+60 levels; the achievable operating points, evaluated on **both**
sets (PTB-XL STE_ n = 28 and ecg-arrhythmia "ST tilt up" n = 58):

| Descriptor rule (any lead but aVR) | PTB-XL STE_ | 664 "ST tilt up" |
|---|:-:|:-:|
| previous: STEMI V2–V3 thresholds, discordant included | 0.18 / 0.80 | 0.81 / 0.84 |
| J ≥ 100 & J+60 ≥ 100, concordant only | 0.57 / 0.88 | 0.67 / 0.91 |
| J ≥ 80 & J+60 ≥ 100, concordant only | 0.71 / 0.84 | 0.69 / 0.85 |
| **J ≥ 100 & J+60 ≥ 80, discordant only when J ≥ 200 (chosen)** | **0.61 / 0.80** | **0.83 / 0.82** |

Chosen: a flat 1 mm for the descriptor (`st_uv.any_elevation_uv`), the STEMI-grade
sex/age thresholds stay with the alert; discordant elevation under BBB/LVH/WPW counts
from 2 mm (`any_elevation_discordant_uv`, a softened Sgarbossa criterion) because the
datasets' broad labels include secondary elevation. Same specificity as before on
both sets, sensitivity 0.18 → 0.61 on PTB-XL and 0.81 → 0.83 on the 664. The alert
was not touched: its 54 PTB-XL false positives are mostly elderly patients with V2–V4
elevation, some under an LBBB the detector missed (no secondary context → the
discordant elevation counts as primary); there is no STEMI gold here to trade against.

### Flutter on the Ningbo label — the records, seen (open item 3, D40)

187 records of the balanced ecg-arrhythmia set carry the SNOMED flutter code and
score 0.01 on `atrial_flutter`. Montages (II, V1, aVF, 5 s) of all 187 plus zoomed
3-s views of 12 were rendered and ~55 inspected by eye:

| What the traces show | Share of the inspected records |
|---|:-:|
| wide-QRS rhythms (LBBB-like / paced morphology), flat baseline, no sawtooth | ~40 % (a contiguous block JS131xx–JS132xx) |
| narrow QRS with visible P waves before each beat (sinus, some irregular) | ~35 % |
| narrow regular tachycardia 130–150 /min where 2:1 flutter is *possible* but no F-waves are visible | ~10 % |
| AF-like irregular narrow QRS / low voltage, undecidable | ~15 % |

Quantitatively, the F-wave spectral features (Welch peak in 3–6.5 Hz after QRS
blanking, peak sharpness, band-power ratio) are no higher in these records than in
the same set's sinus records at any gate setting; on PTB-XL the same features
separate the classes:

| Gate (sharpness / band ratio / leads agreeing) | PTB-XL AFLT 73 | PTB-XL sinus 1127 | Ningbo "flutter" 186 | Ningbo sinus 80 |
|---|:-:|:-:|:-:|:-:|
| ≥ 10 / ≥ 1.0 / ≥ 2 (current) | 0.68 | 0.048 | 0.03 | 0.037 |
| ≥ 10 / ≥ 1.0 / ≥ 3 | 0.59 | 0.020 | 0.03 | 0.025 |
| ≥ 6 / ≥ 0.5 / ≥ 2 | 0.89 | 0.245 | 0.07 | 0.188 |
| ≥ 4 / ≥ 0.2 / ≥ 3 | 0.96 | 0.486 | 0.19 | 0.438 |

Conclusion: the Ningbo flutter label does not describe visible atrial flutter in
this subset, and no detector setting can recover it without flagging sinus records
first. The detector keeps its D28 gates (PTB-XL AFLT 0.51 at spec 0.997 through the
full pipeline); the AF-or-flutter family remains the honest number on this set.

### ST depression — re-measured, thresholds kept (open item 5, D42)

Same method as for elevation: per-lead J / J+60 / J+80 (PR baseline, refined J)
on every PTB-XL and 664 record, rule variants scored on train/test halves (record
hash). Gold: PTB-XL `STD_` (107; `INJ*` subendocardial injury, 34, now added to the
gold), ecg-arrhythmia "ST drop down" (85).

| Depression rule (any lead but aVR) | PTB-XL train | PTB-XL test | 664 train | 664 test |
|---|:-:|:-:|:-:|:-:|
| **current: ≥ 1 mm at J+60 and J+80, not upsloping, ≥ 2 leads** | 0.23 / 0.82 | 0.24 / 0.82 | 0.51 / 0.85 | 0.45 / 0.81 |
| 0.8 mm | 0.38 / 0.76 | 0.41 / 0.77 | 0.61 / 0.78 | 0.57 / 0.80 |
| 0.5 mm | 0.73 / 0.64 | 0.75 / 0.64 | 0.80 / 0.68 | 0.80 / 0.72 |
| 1 mm, ≥ 1 lead | 0.41 / 0.75 | 0.39 / 0.75 | 0.61 / 0.74 | 0.59 / 0.74 |
| 1 mm at J+60 only | 0.20 / 0.84 | 0.24 / 0.84 | 0.46 / 0.87 | 0.41 / 0.86 |
| 1 mm, concordant leads only (no BBB/LVH strain) | 0.07 / 0.95 | 0.14 / 0.96 | 0.15 / 0.98 | 0.14 / 0.95 |
| 0.8 mm, concordant, ≥ 1 lead | 0.25 / 0.88 | 0.35 / 0.89 | 0.20 / 0.90 | 0.34 / 0.88 |

Every variant sits on the same poor curve, so the rule stays. The reason is the
label, not the measurement: of 288 PTB-XL records the rule calls depressed without
an `STD_` label, 63 are LBBB, 37 RBBB and 47 LVH (secondary / strain depression that
PTB-XL does not code as `STD_`), 64 carry `ISC*` and 25 `NST_` (ischaemic or
non-specific ST changes under another code), 10 `INJ*`. Yet 40 of the 107 `STD_`
records themselves carry an LVH label — the cardiologists did label strain
depression as `STD_` sometimes — so "concordant only" (spec 0.95) loses two thirds
of the true positives. The system keeps reporting discordant depression as
`st_depression` flagged secondary (confidence 0.55) with the LVH/BBB causal chain;
what would move this number is an ischaemia label set with lead localisation.

### STEMI alert sensitivity — PTB Diagnostic DB (open item 1, D38)

PhysioNet **ptbdb** (549 records, 290 subjects, 15 leads at 1 kHz) is the only
reachable set with ECGs recorded *days* after an acute myocardial infarction; the
header gives the admission diagnosis, the acute localisation and both dates.
`fetch_ptbdb` downloads the first 10 s of every record (HTTP Range); the adapter
takes MI ≤ 3 days after the infarction as the STEMI proxy (not an ST-elevation
label: some are NSTEMI or post-thrombolysis) and scores the alert by day bin, by
territory agreement (inferior II/III/aVF, anterior V1–V4, lateral I/aVL/V5/V6) and
by false positives per admission reason.

| Days from infarction to ECG | n | `st_elevation` alert | `st_elevation_any` | territory agreement |
|---|--:|:-:|:-:|:-:|
| 0–1 | 61 | **0.23** (14) | 0.39 | 13 / 14 |
| 2–3 | 35 | 0.23 (8) | 0.37 | 8 / 8 |
| 4–7 | 68 | 0.04 | 0.38 | 3 / 3 |
| 8–14 | 87 | 0.05 | 0.30 | 3 / 4 |
| ≥ 15 | 91 | 0.01 | 0.24 | 1 / 1 |
| healthy controls | 80 | 3 false alerts (spec 0.96) | — | — |
| bundle branch block | 17 | 0 false alerts | — | — |
| other non-MI (cardiomyopathy, dysrhythmia, valvular, …) | 84 | 4 false alerts | — | — |

The day gradient is the expected one (elevation fades), the territory is right
whenever the alert fires, and the false-alert rate on controls matches PTB-XL's
(spec 0.97–0.98). The sensitivity at day 0–1 is low, and the misses were looked at
one by one (montage + per-lead J / J+60):

- **A third of the misses have real but sub-threshold J-point elevation** (0.5–1.9
  mm) with the ST segment well above it at J+60 (convex "tombstone" ST): the
  guideline measures at J, and V2–V3 need 2–2.5 mm in men. Re-measuring under the
  refined per-lead J point (the multi-lead envelope offset can sit on the S-wave
  upstroke in V1–V3) did not change the sensitivity but raised PTB-XL specificity
  0.971 → 0.981, so it was kept.
- **A J+60-based rule** ("J+60 ≥ threshold with J ≥ half of it") lifts day 0–1 to
  0.31–0.39 but produces 17–22 false alerts in 80 healthy controls and drops PTB-XL
  specificity to 0.85–0.91 — rejected for a critical alert.
- **Discordant elevation under RBBB was vetoed** (3.8 mm in V1–V3 with RBBB did
  not alert). RBBB's secondary change is depression / T inversion in V1–V3, so
  elevation with RBBB is primary; RBBB was removed from the elevation-mimic
  context (LBBB, pre-excitation and LVH remain): +2 points of sensitivity.
- **Inferior MIs with J ≤ 0.5 mm** are NSTEMI or already resolved — nothing to
  detect on the ECG at guideline thresholds.

What is now measured, and honestly stated: the alert catches about a quarter of
PTBDB's "MI within 3 days" ECGs at the guideline thresholds with 4 % false alerts on
healthy controls and the right territory. A true STEMI-labelled set (elevation
annotated by a cardiologist) would separate the NSTEMI/resolved share from the
detector's own misses; without it the number above is the ceiling that can be
claimed.

### LVH — three criteria, pediatric voltage bands (open item 2, D39)

Sokolow-Lyon alone: PTB-XL 0.35 / 0.97. New measurements (`layer3_measure/voltages.py`):
R aVL, S V3, R V5/V6, S V1/V2, R V1, S V6, tallest limb deflection, intrinsicoid
deflection V5/V6, terminal P force in V1. Criteria sweep on train/test halves
(record hash), BBB/WPW records excluded from the voltage criteria:

| Criterion | PTB-XL train | PTB-XL test | 664 train | 664 test |
|---|:-:|:-:|:-:|:-:|
| Sokolow-Lyon > 3.5 mV | 0.31 / 0.99 | 0.26 / 0.99 | 0.85 / 0.94 | 0.72 / 0.93 |
| Cornell > 2.8 / 2.0 mV (m / f) | 0.32 / 0.94 | 0.24 / 0.94 | 0.23 / 0.92 | 0.32 / 0.93 |
| Romhilt-Estes ≥ 5 (strict strain, PTF ≥ 6000 µV·ms) | 0.13 / 0.94 | 0.15 / 0.94 | 0.62 / 0.95 | 0.48 / 0.95 |
| **Sokolow-Lyon ∪ Cornell (chosen for `lvh`)** | 0.53 / 0.93 | **0.45 / 0.93** | 0.88 / 0.86 | **0.80 / 0.88** |
| any of the three | 0.54 / 0.89 | 0.47 / 0.88 | 0.88 / 0.85 | 0.80 / 0.84 |

Two Romhilt-Estes inputs had to be tightened on the data: strain requires T
inversion *and* ≥ 1 mm depression in V5/V6 (either alone fired on a quarter of
PTB-XL's non-LVH records), and the terminal P force threshold is 6000 µV·ms (the
textbook 1 mm × 40 ms = 4000 flagged 22 % of non-LVH). The point score alone
reports `lvh_probable`. Cornell fired on most LBBB records, so all voltage
criteria are silent under BBB / WPW / wide QRS.

Children: adult voltage criteria are not applied (a tall R is normal). LVH / RVH
use age-band 98th-percentile voltages (Davignon 1979/80 as reproduced in pediatric
ECG references, transcribed approximately and marked so in
`configs/norms/pediatric.yaml`) with a supporting sign (strain or axis) or both
voltages above the band.

Full-pipeline results: PTB-XL LVH 0.35 / 0.97 → **0.48 / 0.93**; 664 LVH 0.90 / 0.91
→ 0.84 / 0.87 (Cornell adds little there and the BBB gate removes a few true
positives); PTB-XL children 0 / 5 → 1 / 5 at specificity 1.00.

### STEMI alert against ST-elevation labels — GE 12SL "STEMI" on PTB-XL+ and CPSC-2018 (open item 1, D50)

PTBDB gave only a proxy (days from infarction, D38). Two label sets closer to the
alert's own definition are reachable: the GE 12SL **STEMI** statement on PTB-XL+
(178 records in PTB-XL; the 141 not in the balanced sample were fetched, 148
records 12SL calls normal serve as controls) and the physician label **ST
elevation** of CPSC-2018 (220 records from the 2020 challenge training set, with
300 ST-depression and 300 sinus-rhythm records as controls;
`fetch_challenge.py`). Neither is a clinical STEMI adjudication — 12SL applies its
own criteria, CPSC's "STE" is a 12-lead reading — but both are ST-elevation
labels, not dates.

| Set | alert sens | false alerts | `st_elevation_any` |
|---|:-:|:-:|:-:|
| 12SL STEMI, 178 — before | 0.34 | 0 / 148 normal | 0.75 |
| 12SL STEMI — **after D50** | **0.40** | 0 / 148 | 0.76 |
| CPSC-2018 STE, 220 — before | 0.33 | spec 0.993 (STD + NSR, 600) | 0.80 |
| CPSC-2018 STE — **after D50** | **0.36** | spec 0.990 | 0.80 |

What the 12SL misses showed (118 before): 42 of the 178 STEMI records carried our
`lbbb` call while 12SL calls none of them LBBB — our QRS read 118–128 ms where
12SL reads 98–112 (the ST take-off merges with the QRS end and the envelope
offset lands late), and "wide + V1 terminal negative" (a normal rS is terminal
negative) made it LBBB, whose discordant-ST context silenced the elevation on 37
misses. True CLBBB on PTB-XL is 143–179 ms wide (p10–p90) with an intrinsicoid
deflection of 54–102 ms in V5/V6, so LBBB now needs QRS ≥ 135 ms (120 in
children) or ≥ the block threshold with intrinsicoid ≥ 60 ms; otherwise the QRS is
`wide_qrs_nonspecific`, which is not an elevation-mimic context. On the data this
keeps 72/72 called CLBBB and frees 22/42 STEMI records. Cost and benefit on the
other sets (before → after, sens / spec):

| | PTB-XL lbbb (85) | 664 lbbb (83) | PTB-XL st_elevation spec (STE_ gold) | 664 st_elevation | PTBDB day 0–1 alert / controls |
|---|:-:|:-:|:-:|:-:|:-:|
| before | 0.85 / 0.87 | 0.77 / 0.94 | 0.974 | 0.41 / 0.985 | 0.23 (14/61) / 3 of 80 |
| after D50 | 0.84 / **0.94** | 0.70 / 0.97 | 0.966 | 0.41 / 0.979 | 0.23 (14/61) / 3 of 80 (unchanged; territory 13/14) |

LBBB specificity on PTB-XL rises by 7 points (≈ 110 fewer false LBBB) at 1–7
points of LBBB sensitivity; the alert gains 6 points on the 12SL set with no new
false alert among its normals and loses 0.6–0.8 points of specificity on the
broader PTB-XL / 664 negatives. The remaining 106 misses: sub-threshold J
elevation (median 148 µV at the maximal lead), 26 % under an LVH call (the
strain-discordance context), 16 % still LBBB — the next layer is the LVH context,
already at the edge of the false-alert budget.

### WPW by delta wave (open item 4, D51)

PTB-XL, 79 WPW records: the classic rule (PR < 120 **and** QRS > 110) found 0.35.
The misses have PR 109 / 131 / 176 ms (p10 / median / p90; Glasgow reads
98 / 122 / 142) and QRS 88 / 106 / 147 — only 15 of 51 have a short PR, 21 a wide
QRS, none both; Glasgow's per-lead delta-wave confidence is ≥ 50 % on 71 % of
the WPW records and 12SL calls WPW on 52 %. A delta-wave descriptor was added:
per lead, the time from the QRS onset until the smoothed slope first reaches
half of the beat's maximal QRS slope (a slurred pre-excited upstroke is slow for
a long stretch), median over beats and then over leads. On PTB-XL it reads
54 ms (p10–p90 26–88) on WPW and 24 ms (18–35) otherwise. Rule sweep on 1688
records (1609 non-WPW), the classic rule OR delta ≥ T with PR ≤ P and QRS ≥ Q:

| rule | sens | spec | false |
|---|:-:|:-:|--:|
| classic only | 0.35 | 0.984 | 26 |
| **delta ≥ 45, PR ≤ 140, QRS ≥ 90** (chosen) | **0.51** | 0.981 | 31 |
| delta ≥ 40, PR ≤ 140, QRS ≥ 90 | 0.56 | 0.978 | 35 |
| delta ≥ 35, PR ≤ 140 | 0.57 | 0.973 | 43 |

An RSR' in V1 with a wide QRS keeps the RBBB reading. Validation: PTB-XL
0.35 → **0.51 / 0.98**; 664 (3 WPW) 0 / 3 at spec 0.973; CPSC (no WPW label)
spec 0.993; pediatric cohort (1591 children, 6 WPW) 0 / 6 → **2 / 6** at spec 1.000.
The four missed children: one is a near miss (PR 108, delta 42 ms — 3 ms under
the threshold), one has delta 68 ms but PR 164 ms and reads as LBBB (QRS 114),
two have PR 159–188 ms with delta 27 and 2 ms (no pre-excitation visible on this
tracing — intermittent conduction or a chart label). Lowering the delta
threshold would cost adult specificity (table above) and recovers one child.

### LVH and ST depression against a second algorithm — GE 12SL on PTB-XL+ (open item 6)

The question was whether LVH 0.48 and ST depression 0.28 on PTB-XL are our
measurement or the label. PTB-XL+ carries the GE 12SL statements and per-lead
measurements for the same records, so a commercial algorithm can be scored
against the same cardiologist labels, and our measurements against its (1692
local records):

| | sens / spec vs the PTB-XL label | n+ |
|---|:-:|--:|
| LVH — ours (Sokolow ∪ Cornell, D39) | 0.48 / 0.93 | 198 |
| LVH — **12SL** (LVH / LVH2 / LVH3 statements) | **0.42 / 0.95** | 198 |
| ST depression — ours | 0.28 / 0.81 | 125 |
| ST depression — **12SL** (STDEP2, subendocardial-injury statements) | **0.06 / 0.98** | 125 |

The commercial algorithm does no better than the rules against the LVH label
and calls almost none of the STD_ records ST depression: both labels encode
something the ECG measurements alone do not carry (clinical LVH; "non-specific
ST depression" read by a cardiologist). That settles the item as label
semantics, with an independent witness.

Measurements agree where they should: R V5/V6, S V1, R aVL, S V3 read 4 %
lower than 12SL at the median (ratio 0.96, p10 0.87 — different beat selection
and baselines); the Sokolow-Lyon sum differs by −115 µV at the median (MAE 350 µV)
and the Cornell sum by −58 µV (MAE 180 µV), so of the 135 records 12SL flags
Sokolow-positive our sum reaches 3.5 mV on 92 (the voltage criterion sits on a
continuum and a 4 % amplitude difference moves the borderline cases); the ST
amplitude at J differs from 12SL's STJ by a median of −9
(II), +33 (V2), +9 (V3), −29 (V5) µV with MAE 50–100 µV — a third to a whole
threshold, which is why the elevation/depression descriptors cannot be tuned
much further against labels of this kind. Our J+80 reads 120–140 µV above
Glasgow's "ST80" in V2/V3 (near zero in II and V5); the definitions differ (the
Glasgow value is not referenced to the same baseline), so this is recorded, not
corrected. Agreement of our LVH call with 12SL's: 0.45 / 0.92 (n+ 163) — the two
algorithms disagree with each other as much as either disagrees with the label.

### Infants and toddlers — what the reachable data holds (open item 5)

Searched for 12-lead records aged under 4 in every PhysioNet set reachable from
this environment: ecg-arrhythmia (45 152 headers: none aged 1–3, "age 0" is
unknown, D44), PTB-XL (21 799: none under 2, three aged 2–3, all NORM/SR),
CPSC-2018 (6 868 with an age: none under 1, three aged 1–3). The six toddler
records were run through the pipeline with the 1–3-year band (PR 80–150, QRS
45–85, QT 270–420): HR 98–174, PR 96–127 ms, QRS 70–80 ms, QTc(F) 327–370 ms,
axis 53–78°, juvenile T pattern flagged as expected; the one 1-year-old labelled
RBBB (CPSC A4546) has PR 229 ms (1° AV block, not in the label), QRS 94 ms and a
left axis — a 94-ms QRS is wide at that age but sits under the flat 110-ms block
threshold, so the RBBB is missed. One record is not evidence for a band-relative
block threshold; it is noted as the first thing to test when an infant set
appears. Conclusion unchanged: the <1 y and 1–3 y bands cannot be measured or
calibrated from reachable data; their norm-table entries remain literature
values (Davignon / Rijnbeek) and are marked so in `pediatric.yaml`.

### QRS axis against GE 12SL and Glasgow; what PTB-XL's axis label means (open item 7, D49)

PTB-XL+ (Strodthoff et al., PhysioNet) publishes the measurements of two
commercial algorithms for every PTB-XL record — GE 12SL (`R_AxisFrontal_Global`)
and the University of Glasgow program (`QRS_AxisFront_Global`). On the 1691 local
records with a measurable axis:

| Estimator | vs 12SL: MAE / median / within 15° | vs Glasgow |
|---|:-:|:-:|
| largest peak in I and aVF (before) | 13.3° / 8.4° / 72 % | 14.4° / 9.0° / 69 % |
| net deflection (max + min in the QRS) in I and aVF | 6.9° / 4.7° / 91 % | 7.6° / 4.8° / 90 % |
| **net deflection, all six limb leads, hexaxial least squares (now)** | **4.7° / 2.6° / 94 %** | **5.6° / 2.7° / 93 %** |
| QRS area, I and aVF / six leads | 12.1° / 6.5° / 78 % · 12.2° / 6.4° / 77 % | 11.5° / 5.4° / 81 % |
| 12SL vs Glasgow themselves | 4.7° | — |

The six-lead net deflection agrees with either reference as well as the two
references agree with each other; it is the conventional definition and is now
`_qrs_axis` (the peak estimate remains the fallback when no QRS bounds exist).

The label: PTB-XL's `heart_axis` has the categories MID, LAD, ALAD, RAD, ARAD,
AXL, AXR. The 12SL axis per category (p25 / median / p75): **LAD −30 / −16 / −3**,
**ALAD −58 / −48 / −39**, MID 20 / 41 / 61, **RAD 92 / 96 / 103**, **ARAD 112 / 130 /
207**. So "LAD" is the mild leftward axis inside the normal −30 … 90 range —
even the 12SL and Glasgow axes call it left axis deviation on only 76 of 305 —
and "ALAD" is the deviation. The adapter had used LAD as the
`left_axis_deviation` gold (hence 0.36); it now takes ALAD, and RAD ∪ ARAD for
`right_axis_deviation`. PTB-XL, 1692 records, before → after (axis estimator and
gold together):

| Finding | before | after | n |
|---|:-:|:-:|--:|
| left_axis_deviation | 0.36 / 0.84 (gold LAD, 305) | **0.86 / 0.92** (gold ALAD) | 130 |
| right_axis_deviation | 0.89 / 0.96 (gold RAD, 93) | 0.75 / 0.98 (gold RAD ∪ ARAD) | 130 |

The right-axis misses are ARAD records (axis beyond 120°, many north-west) that
the rules call `extreme_axis` instead — a different code, not a miss in substance.
Macro sensitivity on PTB-XL 0.596 → 0.617.

### LBBB / RBBB typing — terminal forces inside the lead's own QRS (open item 4, D41)

26 of the 37 LBBB records PTB-XL that were missed were called RBBB: the terminal-force
window ended at the multi-lead envelope QRS offset, which in V1 lies after this
lead's own QRS end, so the window swallowed the discordant ST elevation of the LBBB
and read it as a positive R'. The lead's QRS end is now the last sample whose slope
is ≥ 15 % of that lead's QRS slope, with a PR-segment baseline
(`delineation.terminal_qrs_sign`).

| | PTB-XL LBBB (85) | PTB-XL RBBB (86) | 664 LBBB (83) | 664 RBBB (71) | PTB-XL STEMI-alert spec |
|---|:-:|:-:|:-:|:-:|:-:|
| before | 0.57 / 0.88 | 0.85 / 0.93 | 0.71 / 0.94 | 0.72 / 0.98 | 0.968 |
| after | **0.85 / 0.86** | **0.87 / 0.96** | 0.77 / 0.94 | 0.72 / 0.99 | **0.974** |

Remaining LBBB misses: QRS measured ≤ 110 ms (8 on PTB-XL, 9 on the 664) and a
short measured PR that routes the wide QRS to WPW (3 / 6). The fuller morphology
criteria of the plan (notched R in I/aVL/V5/V6, absent q) were not needed for
this step and stay open.

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
| lvh | 0.00 → 0.20 (D39) | 1.00 | 5 |
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
rule-based pediatric norm table plus this evaluation. Re-run after D39/D44: LVH
1 / 5 (the pediatric voltage bands now exist), macro sensitivity 0.51, nothing else
moved. The larger pediatric cohort is the next section.

### Pediatric cohort — ecg-arrhythmia, 1838 records aged ≤ 15 (open item 7, D44)

`fetch_physionet --folders 0 --age-max 15` read all 45 152 headers of the
ecg-arrhythmia database and downloaded every record with `#Age ≤ 15`:
1838 records into `data/ecg-arrhythmia-pediatric/` (git-ignored, manifest with
age/sex/labels). Age distribution: **0** → 247, **1–3 y** → 0, **4–8 y** → 971,
**9–15 y** → 620. The 247 "age 0" records measure like adults (HR median 78, QRS
median 90 ms, PR 156 ms, R V6 1.1 mV) — the database uses 0 for *unknown*, so the
adapter now maps 0 to "no age" and the pediatric evaluation is the 1591 records
aged 1–15. Labels are adult-style SNOMED (sinus tachycardia 762, sinus rhythm 168,
sinus bradycardia 68, RAD 35, RBBB 10, LAD 10, 1° AV block 7, WPW 6, ST depression
4, ST elevation 3, long QT 2, LBBB 1, no LVH/AF/flutter/CAVB).

**What the cohort measured, per band (p2 / p50 / p98), before any change:**

| Band | n | HR | PR ms | QRS ms | QTc(F) ms | axis ° | R V6 / S V1 p98 mV |
|---|--:|:-:|:-:|:-:|:-:|:-:|:-:|
| 4–8 y | 971 | 65 / 115 / 169 | 88 / 122 / 183 | 66 / 76 / 101 | 323 / 364 / 417 | −126 / 66 / 145 | 1.9 / 1.9 |
| 9–15 y | 620 | 53 / 88 / 184 | 87 / 128 / 245 | 68 / 80 / 102 | 321 / 375 / 428 | −62 / 68 / 142 | 2.2 / 2.3 |

The measured voltages sit under the Davignon 98th percentiles written into
`pediatric_98th_by_age` (R V6 2.4–2.6 mV, S V1 2.3–2.5 mV at these ages), so the
LVH/RVH bands from D39 are consistent with a general pediatric population.
The intervals showed four things the norm table got wrong for children, each
fixed as a **norm-table entry**, not a model:

| Symptom (before) | Cause | Change | After |
|---|---|---|---|
| `lbbb` on 38 / 971 (4–8 y, spec 0.961) and 78 / 620 (9–15 y) | QRS upper normal 90 ms was also the BBB threshold; the 98th percentile of the cohort is 101–102 ms | `intervals_ms_by_age` upper limits per band + `intervals_ms.qrs_bbb: 110` | 6 / 971 (spec 0.994), 7 / 620 (0.989); after D45 (zero leads excluded) 0 / 971 and 2 / 620 (0.997) |
| `lead_reversal_la_ra` on 96 / 971 and 49 / 620 | I inverted + aVR upright is the signature of a physiological rightward axis in a child | the reversal hypothesis is raised only when the axis lies outside the pediatric normal range | 21 / 971, 10 / 620 |
| `flat_t_waves` on 684 / 971 (70 %) and 234 / 620 | first diagnosis "small pediatric T waves vs a 100 µV cut" was **wrong**: the relative condition (T < 15 % QRS) removed 15 of 684. The flat leads were V4–V6 in 113 of 120 records checked — **the precordial channels are all zero** in 673 / 971 and 166 / 620 records (limb-only recordings stored as 12 leads) | D45: a lead with no signal (< 10 µV span, the Layer-2 lead-off criterion) is not measured by Layer 3 — no T morphology, no ST, no R/S voltage (None, not 0), no terminal sign | 15 / 971, 44 / 620 |
| `short_qt` on 120 / 971 (12 %) and 47 / 620 | 2nd percentile of QTc(F) in the cohort is 321–329 ms | `qtc_ms.short_lower` 340 → 320 | 14 / 971, 13 / 620 |

D45 also explains the "R V6 median 0" in the 4–8 y row of the distribution table
(half of that band has no V6) and removed the last false LBBB calls: the V1/V6
terminal sign was being read from a zero channel. On the adult sets D45 is
neutral (15 of the 664 records have a zero lead; PTB-XL none): 664 macro 0.653 and
PTB-XL 0.596 / lvh 0.48 / lbbb 0.85 / rbbb 0.87 / st_elevation spec 0.974, all as
before.

Per-class results after the change, `validate --ecg-arrhythmia
data/ecg-arrhythmia-pediatric --age-min 1 --age-max 15` (1591 records; the bands
in brackets are 4–8 y / 9–15 y):

| Finding | Sens | Spec | n | note |
|---|:-:|:-:|--:|---|
| sinus_tachycardia | 0.87 (0.88 / 0.85) | 0.98 | 762 | adult label (HR > 100); pediatric HR bands call 100–110 normal at 4–8 y |
| sinus_rhythm | 0.90 (0.84 / 0.96) | 0.59 | 168 | the "false" sinus rhythms are the adult-labelled tachycardias inside the pediatric band |
| sinus_bradycardia | 0.90 (0.80 / 0.93) | 0.98 | 68 | |
| first_degree_av_block | 0.71 (1 / 3 · 4 / 4) | 0.97 | 7 | |
| rbbb | 0.50 (3 / 5 · 2 / 5) | 0.999 | 10 | |
| lbbb | 0 / 1 | 0.999 | 1 | before D44/D45: 0.961 / 0.87 by band |
| left_axis_deviation | 0.40 | 0.98 | 10 | |
| right_axis_deviation | 0.00 | 1.00 | 35 | adult label (> +90°); the pediatric normal axis reaches +180°, deliberately |
| wpw_preexcitation | 0 / 6 | 1.00 | 6 | real gap: the delta-wave/short-PR rule misses the pediatric WPW (also 1 / 2 in PTB-XL) |
| st_elevation (alert) / st_elevation_any | 1 / 3 · 3 / 3 | 0.98 / 0.89 | 3 | |
| st_depression | 1 / 4 | 0.94 | 4 | |
| long_qt | 0 / 2 | 0.996 | 2 | |
| atrial_fibrillation ∪ flutter | — | 0.96 | 0 | after the conducted-P gate of D36 |
| lvh / rvh | — | 1.00 | 0 | no LVH label in the cohort; PTB-XL children: 1 / 5 |

Macro sensitivity 0.37 over the labels present (unchanged by the fixes: the
fixes removed false positives of codes that are not scored labels). What is *not*
claimed: no pediatric fine-tuning, no calibration on annotated pediatric intervals
(the cohort has rhythm/conduction labels only), and the 1–3-year and infant bands
remain unmeasured — the database has no records there once "age 0" is read
correctly. The norm-table entries were changed only where the cohort's own
percentiles contradicted the table, and each entry is a documented number in
`configs/norms/pediatric.yaml`.

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

### Learnable layer — hold-out and cross-dataset (item 5a)

Per-class `HistGradientBoostingClassifier` on 79 Layer-3 features (intervals, axis,
voltages, per-lead J/J+60/J+80 and net QRS, T-morphology flags, atrial-band
spectrum, RR statistics). Training: 1584 ecg-arrhythmia records (`fetch_physionet
--seed 1`, 1322 records from other folders + the 300 slice; 38 hold-out duplicates
removed). Hold-out: the balanced 664 (same database, disjoint records). Cross-dataset:
PTB-XL 1692. Threshold 0.5. Numbers are sens / spec for rules alone, model alone,
and rules ∪ model (the pipeline's effective behaviour when a suggestion is allowed).

| Class | hold-out rules | model | ∪ | PTB-XL rules | model | ∪ |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| atrial_fibrillation | 0.80 / 0.73 | 0.60 / 0.87 | 0.86 / 0.70 | 0.55 / 0.97 | 0.44 / 0.97 | 0.72 / 0.94 |
| atrial_flutter | 0.01 / 1.00 | 0.62 / 0.87 | 0.62 / 0.87 | 0.51 / 1.00 | 0.51 / 0.93 | 0.71 / 0.93 |
| sinus_rhythm | 0.89 / 0.84 | 0.76 / 0.97 | 0.91 / 0.84 | 0.83 / 0.78 | 0.60 / 0.91 | 0.84 / 0.78 |
| sinus_bradycardia | 0.93 / 0.98 | 0.89 / 0.99 | 0.94 / 0.97 | 0.96 / 0.88 | 0.95 / 0.89 | 0.97 / 0.87 |
| sinus_tachycardia | 0.91 / 0.89 | 0.85 / 0.98 | 0.95 / 0.89 | 0.92 / 0.96 | 0.83 / 0.97 | 0.95 / 0.95 |
| first_degree_av_block | 0.68 / 0.94 | 0.54 / 0.98 | 0.72 / 0.93 | 0.63 / 0.92 | 0.47 / 0.95 | 0.64 / 0.91 |
| rbbb | 0.72 / 0.98 | 0.86 / 1.00 | 0.87 / 0.98 | 0.85 / 0.93 | 0.93 / 0.96 | 0.97 / 0.92 |
| lbbb | 0.71 / 0.94 | 0.70 / 0.98 | 0.87 / 0.92 | 0.57 / 0.88 | 0.73 / 0.97 | 0.88 / 0.86 |
| lvh | 0.90 / 0.91 | 0.77 / 0.97 | 0.92 / 0.91 | 0.35 / 0.97 | 0.10 / 1.00 | 0.36 / 0.97 |
| low_qrs_voltage | 0.72 / 0.86 | 0.65 / 0.96 | 0.86 / 0.84 | 0.74 / 0.91 | 0.50 / 0.91 | 0.81 / 0.85 |
| long_qt | 0.70 / 0.86 | 0.30 / 0.97 | 0.70 / 0.84 | 0.49 / 0.89 | 0.14 / 0.99 | 0.50 / 0.88 |
| st_depression | 0.48 / 0.82 | 0.55 / 0.93 | 0.65 / 0.79 | 0.23 / 0.80 | 0.27 / 0.90 | 0.36 / 0.77 |
| st_elevation | 0.38 / 0.98 | 0.64 / 0.98 | 0.66 / 0.97 | 0.00 / 0.97 | 0.00 / 0.96 | 0.00 / 0.94 |
| left_axis_deviation | 0.87 / 0.86 | 0.76 / 0.91 | 0.94 / 0.84 | 0.36 / 0.84 | 0.27 / 0.89 | 0.41 / 0.82 |
| right_axis_deviation | 0.69 / 0.96 | 0.71 / 0.97 | 0.84 / 0.96 | 0.89 / 0.96 | 0.84 / 0.97 | 0.91 / 0.96 |

Honest reading:

- The model beats the rules where the rule is a crude threshold over morphology —
  RBBB, LBBB, ST elevation on the hold-out (0.38 → 0.64 at equal specificity) — and
  loses where the rule computes a physical quantity (LVH, long QT, low voltage,
  first-degree AV block): 79 summary features do not see the waveform.
- Flutter 0.62 on the hold-out is the model learning the Ningbo label's semantics
  from the same database (irregular response, no organised P); on PTB-XL it equals
  the rule (0.51) at lower specificity. Cross-dataset, the model degrades more than
  the rules on almost every class — the rules encode physiology, the model encodes
  a database.
- Rules ∪ model buys 5–15 points of sensitivity for 1–5 of specificity nearly
  everywhere, which is why the integration is "evidence, never authority" (D33):
  `ml_p` on rule findings, a confidence bump on agreement, and `ml_suggested_*` only
  for classes with spec ≥ 0.95 on **both** sets and sens ≥ rules on PTB-XL —
  `rbbb`, `lbbb`, `st_elevation` (the last passes the PTB-XL test vacuously, 0.00 ≥
  0.00; its real value is the hold-out). The safety layer never reads ML output.
- Reproducible: `python -m cardiolens.ml.train --train data/ecg-arrhythmia-train
  data/ecg-arrhythmia --holdout data/ecg-arrhythmia-balanced --cross data/ptbxl
  --report configs/models/rhythm_hgb.report.json`; the JSON report is committed,
  the 7 MB joblib is not.

### Learnable layer v2 — waveform features, mixed sources, the transfer gap (open item 8, D46)

What changed: (1) **feature set v2** = the 79 summary features + 300 shape
features (per lead the median beat, −250 … +450 ms around R, PR baseline,
amplitude-normalised by the lead's QRS peak, 24 points, plus QRS/T energy
ratios); the set is versioned in the artifact and in the feature cache, an
artifact of another version is refused; (2) **mixed-source training**:
ecg-arrhythmia 1322 + 300 and PTB-XL strat-folds 1–8 (1352) → 2936 records;
hold-outs = the balanced 664 (same source) and PTB-XL folds 9–10 (340,
patient-disjoint); cross = the pediatric cohort (1838); (3) a **transfer report**:
train on one database only, test on the other, both directions — that is the
transfer gap the item asked for. Command:
`python -m cardiolens.ml.train --train data/ecg-arrhythmia-train data/ecg-arrhythmia
data/ptbxl --holdout data/ecg-arrhythmia-balanced --cross data/ecg-arrhythmia-pediatric
--transfer ecgarr=data/ecg-arrhythmia-train ptbxl=data/ptbxl --report
configs/models/rhythm_hgb.report.json` (the JSON is committed, the 18 MB joblib is
not; features re-extracted after D45).

**Transfer gap, v1 → v2 features, model trained on ecg-arrhythmia only and tested on
PTB-XL (1692):**

| Class | rules | v1 model (79 features) | v2 model (379) |
|---|:-:|:-:|:-:|
| rbbb | 0.87 / 0.96 | 0.93 / 0.96 | 0.93 / 0.97 |
| lbbb | 0.85 / 0.87 | 0.73 / 0.97 | **0.84 / 0.97** |
| st_depression | 0.28 / 0.81 | 0.27 / 0.90 | 0.35 / 0.93 |
| first_degree_av_block | 0.63 / 0.92 | 0.47 / 0.95 | 0.52 / 0.93 |
| low_qrs_voltage | 0.74 / 0.91 | 0.50 / 0.91 | 0.59 / 0.92 |
| lvh | 0.48 / 0.93 | 0.10 / 1.00 | 0.12 / 0.99 |
| long_qt | 0.49 / 0.89 | 0.14 / 0.99 | 0.11 / 0.99 |
| atrial_fibrillation | 0.55 / 0.97 | 0.44 / 0.97 | 0.29 / 0.99 |
| sinus_rhythm | 0.83 / 0.78 | 0.60 / 0.91 | 0.54 / 0.93 |
| right_axis_deviation | 0.89 / 0.96 | 0.84 / 0.97 | 0.82 / 0.97 |

The beat shape helps exactly where a morphology class was being read from summary
numbers (LBBB +11 points across databases at unchanged specificity, ST depression
+8, low voltage +9) and does nothing for classes that are a physical quantity (LVH,
long QT — the rule measures them, a normalised beat shape does not contain them),
and it costs on AF and sinus rhythm (−15 and −6: more features fitted to one
database's beat shapes). The **rules still transfer better than any model** on
10 of 15 classes; the model is evidence, not a second reader.

<!-- ML_V2_TABLES -->
**Mixed-source model (2936 records) — hold-out ecg-arrhythmia 664 (rules / model / rules ∪ model, sens / spec):**
| class | n | rules | model | ∪ |
|---|--:|:-:|:-:|:-:|
| atrial_fibrillation | 80 | 0.80 / 0.73 | 0.71 / 0.81 | 0.91 / 0.69 |
| atrial_flutter | 187 | 0.01 / 1.00 | 0.47 / 0.88 | 0.47 / 0.88 |
| sinus_rhythm | 80 | 0.90 / 0.84 | 0.89 / 0.92 | 0.95 / 0.82 |
| sinus_bradycardia | 115 | 0.93 / 0.98 | 0.86 / 0.99 | 0.94 / 0.98 |
| sinus_tachycardia | 93 | 0.91 / 0.89 | 0.87 / 0.99 | 0.95 / 0.89 |
| first_degree_av_block | 85 | 0.68 / 0.94 | 0.72 / 0.98 | 0.80 / 0.93 |
| rbbb | 71 | 0.72 / 0.99 | 0.79 / 1.00 | 0.86 / 0.99 |
| lbbb | 83 | 0.77 / 0.94 | 0.68 / 0.99 | 0.85 / 0.93 |
| lvh | 51 | 0.84 / 0.87 | 0.82 / 0.95 | 0.96 / 0.86 |
| low_qrs_voltage | 81 | 0.72 / 0.86 | 0.58 / 0.97 | 0.80 / 0.85 |
| long_qt | 23 | 0.70 / 0.86 | 0.48 / 0.98 | 0.74 / 0.85 |
| st_depression | 85 | 0.49 / 0.82 | 0.52 / 0.95 | 0.67 / 0.79 |
| st_elevation | 58 | 0.41 / 0.98 | 0.40 / 0.99 | 0.60 / 0.98 |
| left_axis_deviation | 83 | 0.87 / 0.86 | 0.68 / 0.94 | 0.93 / 0.84 |
| right_axis_deviation | 80 | 0.69 / 0.96 | 0.68 / 0.97 | 0.78 / 0.96 |
| wpw_preexcitation | 3 | 0.00 / 0.98 | 0.00 / 1.00 | 0.00 / 0.97 |

**Hold-out PTB-XL strat-folds 9–10 (340, patient-disjoint):**
| class | n | rules | model | ∪ |
|---|--:|:-:|:-:|:-:|
| atrial_fibrillation | 32 | 0.53 / 0.97 | 0.59 / 0.97 | 0.72 / 0.95 |
| atrial_flutter | 14 | 0.43 / 1.00 | 0.79 / 0.98 | 0.79 / 0.98 |
| sinus_rhythm | 219 | 0.81 / 0.76 | 0.87 / 0.82 | 0.91 / 0.74 |
| sinus_bradycardia | 24 | 0.96 / 0.88 | 0.83 / 0.93 | 0.96 / 0.88 |
| sinus_tachycardia | 23 | 0.91 / 0.96 | 0.87 / 0.97 | 0.91 / 0.95 |
| first_degree_av_block | 37 | 0.62 / 0.91 | 0.70 / 0.94 | 0.81 / 0.89 |
| rbbb | 15 | 0.87 / 0.97 | 0.80 / 0.97 | 0.93 / 0.97 |
| lbbb | 18 | 0.72 / 0.88 | 0.61 / 0.98 | 0.94 / 0.87 |
| lvh | 52 | 0.44 / 0.93 | 0.52 / 0.96 | 0.61 / 0.90 |
| low_qrs_voltage | 37 | 0.76 / 0.91 | 0.76 / 0.94 | 0.84 / 0.89 |
| long_qt | 23 | 0.56 / 0.90 | 0.39 / 0.98 | 0.70 / 0.89 |
| st_depression | 29 | 0.24 / 0.81 | 0.38 / 0.93 | 0.45 / 0.78 |
| st_elevation | 6 | 0.00 / 0.97 | 0.00 / 0.98 | 0.00 / 0.96 |
| left_axis_deviation | 63 | 0.35 / 0.87 | 0.51 / 0.87 | 0.64 / 0.79 |
| right_axis_deviation | 17 | 1.00 / 0.95 | 1.00 / 0.96 | 1.00 / 0.95 |
| wpw_preexcitation | 15 | 0.40 / 0.97 | 0.40 / 0.99 | 0.67 / 0.97 |

**Cross set — pediatric cohort (1838):**
| class | n | rules | model | ∪ |
|---|--:|:-:|:-:|:-:|
| atrial_fibrillation | 0 | — / 0.95 | — / 0.99 | — / 0.95 |
| atrial_flutter | 19 | 0.00 / 0.99 | 0.53 / 0.98 | 0.53 / 0.97 |
| sinus_rhythm | 329 | 0.90 / 0.61 | 0.63 / 0.91 | 0.91 / 0.60 |
| sinus_bradycardia | 84 | 0.92 / 0.98 | 0.81 / 0.99 | 0.94 / 0.98 |
| sinus_tachycardia | 772 | 0.87 / 0.96 | 0.47 / 0.97 | 0.97 / 0.94 |
| first_degree_av_block | 11 | 0.73 / 0.97 | 0.46 / 1.00 | 0.82 / 0.96 |
| rbbb | 14 | 0.64 / 0.99 | 0.64 / 1.00 | 0.64 / 0.99 |
| lbbb | 2 | 0.50 / 0.99 | 0.50 / 0.99 | 0.50 / 0.98 |
| lvh | 4 | 0.25 / 0.98 | 0.50 / 0.98 | 0.75 / 0.97 |
| low_qrs_voltage | 3 | 0.67 / 0.98 | 0.33 / 0.99 | 0.67 / 0.98 |
| long_qt | 6 | 0.17 / 0.99 | 0.33 / 1.00 | 0.33 / 0.99 |
| st_depression | 7 | 0.57 / 0.93 | 0.29 / 1.00 | 0.57 / 0.93 |
| st_elevation | 7 | 0.43 / 0.97 | 0.43 / 0.99 | 0.71 / 0.97 |
| left_axis_deviation | 16 | 0.56 / 0.97 | 0.56 / 0.99 | 0.69 / 0.97 |
| right_axis_deviation | 40 | 0.07 / 0.99 | 0.65 / 0.91 | 0.68 / 0.91 |
| wpw_preexcitation | 6 | 0.00 / 1.00 | 0.17 / 1.00 | 0.17 / 1.00 |

**Transfer: trained on ecg-arrhythmia only (1322), tested on PTB-XL (1692):**
| class | n | rules | model |
|---|--:|:-:|:-:|
| atrial_fibrillation | 123 | 0.55 / 0.97 | 0.29 / 0.99 |
| atrial_flutter | 73 | 0.51 / 1.00 | 0.44 / 0.93 |
| sinus_rhythm | 1135 | 0.83 / 0.78 | 0.54 / 0.93 |
| sinus_bradycardia | 106 | 0.96 / 0.88 | 0.93 / 0.90 |
| sinus_tachycardia | 110 | 0.92 / 0.96 | 0.84 / 0.98 |
| first_degree_av_block | 188 | 0.63 / 0.92 | 0.52 / 0.93 |
| rbbb | 86 | 0.87 / 0.96 | 0.93 / 0.96 |
| lbbb | 85 | 0.85 / 0.87 | 0.83 / 0.98 |
| lvh | 198 | 0.48 / 0.93 | 0.12 / 0.99 |
| low_qrs_voltage | 182 | 0.74 / 0.91 | 0.59 / 0.92 |
| long_qt | 117 | 0.49 / 0.89 | 0.11 / 0.99 |
| st_depression | 125 | 0.28 / 0.81 | 0.35 / 0.93 |
| st_elevation | 28 | 0.00 / 0.97 | 0.04 / 0.96 |
| left_axis_deviation | 305 | 0.36 / 0.84 | 0.25 / 0.89 |
| right_axis_deviation | 93 | 0.89 / 0.96 | 0.82 / 0.97 |

**Transfer: trained on PTB-XL only (1692), tested on ecg-arrhythmia (1322):**
| class | n | rules | model |
|---|--:|:-:|:-:|
| atrial_fibrillation | 150 | 0.77 / 0.80 | 0.83 / 0.78 |
| atrial_flutter | 319 | 0.05 / 0.99 | 0.13 / 0.97 |
| sinus_rhythm | 150 | 0.95 / 0.86 | 0.97 / 0.67 |
| sinus_bradycardia | 325 | 1.00 / 0.94 | 0.20 / 1.00 |
| sinus_tachycardia | 164 | 0.91 / 0.92 | 0.82 / 0.98 |
| first_degree_av_block | 153 | 0.74 / 0.93 | 0.56 / 0.96 |
| rbbb | 154 | 0.82 / 0.98 | 0.75 / 0.98 |
| lbbb | 56 | 0.57 / 0.95 | 0.48 / 1.00 |
| lvh | 138 | 0.85 / 0.88 | 0.91 / 0.90 |
| low_qrs_voltage | 151 | 0.68 / 0.86 | 0.41 / 0.94 |
| long_qt | 76 | 0.49 / 0.90 | 0.13 / 0.98 |
| st_depression | 157 | 0.64 / 0.84 | 0.26 / 0.97 |
| st_elevation | 156 | 0.40 / 0.98 | 0.00 / 1.00 |
| left_axis_deviation | 156 | 0.86 / 0.90 | 0.36 / 0.94 |
| right_axis_deviation | 119 | 0.87 / 0.94 | 0.74 / 0.97 |
| wpw_preexcitation | 5 | 0.20 / 0.98 | 0.40 / 0.99 |
<!-- /ML_V2_TABLES -->

Reading the mixed-source model:

- **Same-source hold-out (664)** vs the v1 single-source model: AF 0.60 → 0.71,
  sinus rhythm 0.76 → 0.89, 1° AV block 0.54 → 0.72, LVH 0.77 → 0.82 improved;
  **flutter 0.62 → 0.47 and ST elevation 0.64 → 0.40 dropped** — mixing in PTB-XL
  taught the model PTB-XL's AFLT / STE_ semantics, which disagree with the Ningbo
  labels ("ST tilt up", flutter without F waves, D40/D37). Label semantics differ
  between databases and a mixed model averages them; that is the price of the
  smaller transfer gap, not a bug.
- **PTB-XL folds 9–10 (340)**: the mixed model reaches LVH 0.52 / 0.96 where the
  ecg-arrhythmia-only model had 0.12 — it learned PTB-XL's LVH label from folds
  1–8, not LVH physiology (the transfer table shows that: 0.12 the moment the
  database is unseen).
- **Pediatric cross set (1838)**: the model is useless for the adult-labelled
  tachycardias (0.47 vs rules 0.87) and reads RAD in children at 0.65 where the
  rules deliberately say 0.07 (pediatric normal axis); rules ∪ model would undo
  the pediatric norm table. Another reason the model is not authority.
- **Suggestion policy (D46)**: a class may raise `ml_suggested_*` on its own only if
  spec ≥ 0.95 on every set **and** sens ≥ rules on every foreign set with ≥ 20
  positives, *in both transfer directions*; a class no large foreign set ever
  voted on is not eligible. Under that criterion **no class qualifies** (RBBB
  loses to the rule in the PTB-XL → ecg-arrhythmia direction, 0.75 < 0.83; LBBB
  loses both ways by 1–9 points; ST elevation 0.00 < 0.40 towards
  ecg-arrhythmia). The earlier list (rbbb, lbbb, st_elevation) rested on a single
  transfer direction. `ml_p` on rule findings and the agreement bump stay; the
  safety layer never reads ML output.

### Paper-page digitisation v3 — self-calibration and photo perspective (open item 6, D43)

Added: grid-pitch estimation (px/mm from the grid's periodicity, tile-wise, with
the layout's 32-mm row pitch as the primary scale), calibration-pulse reading
(mm/mV from the 1 mV × 200 ms rectangle), perspective rectification from the grid's
vanishing points, and a phone-photo degradation model (perspective, vignette, paper
texture). Synthetic pages, auto-calibrated (no px/mm or mm/mV given), 3 seeds:

| Degradation | 6 px/mm: mean r / min r | 8 px/mm: mean r / min r | scale error | mm/mV read |
|---|:-:|:-:|:-:|:-:|
| clean, rotation 3°, JPEG 40, shadow 50 %, salt 0.5 % | 0.93 / 0.89 | 0.96 / 0.93 | < 1 % | 9.8–9.95 |
| blur 2 px | 0.91 / 0.87 | 0.96 / 0.93 | < 1 % | 9.8 |
| scan combo (−2°, blur, JPEG 40, shadow, salt) | 0.92 / 0.88 | 0.96 / 0.93 | < 1 % | 9.8 |
| photo: perspective 2 %, vignette, texture, blur, JPEG 45, 1.5° | 0.76 (0.44 / 0.92 / 0.92) / 0.63 | 0.94 / 0.89 | 2–3 % | 9.9 |
| perspective 3.5 % alone | 0.23 | 0.29 | — | — |
| photo strong (3.5 %, vignette 0.6, texture 0.5, JPEG 40, −2.5°, salt) | 0.27 | 0.07 | — | — |

The vanishing-point rectification works to about 2 % corner displacement and fails
beyond it (the segment-angle fit is no longer reliable and the self-check rejects or
mis-accepts the warp) — a stated limit, not tuned around. A JPEG-specific bug was
found on the way: Hough skew estimation chained 8×8 block edges into "lines" and
read a 3° rotation as 0.3° (maxLineGap 20 → 5).

**Strong perspective — the page quadrilateral (D48).** When the whole page is in
the frame, the grid-textured region (residual after a local mean, so a vignette
or shadow does not count as paper) is a quadrilateral whose hull gives the four
corners; each side is snapped to the outermost grid line (Hough in a band, the
nearest long parallel line) and the quadrilateral is mapped to a rectangle. The
step is taken only when the projective part of that homography changes the scale
across the page by ≥ 1 % (a rotation or a scan is affine and stays with the skew
step, exact for a rotation); the vanishing-point step then removes the few-pixel
corner residual, accepted only if the grid stays square and the angle spread
drops further. Synthetic pages, auto-calibrated, three seeds, mean r / min r:

| Degradation | seed 1 | seed 2 | seed 3 | before D48 |
|---|:-:|:-:|:-:|:-:|
| perspective 2 % | 0.93 / 0.89 | 0.94 / 0.90 | 0.85 / 0.69 | 0.76 |
| perspective 3.5 % | 0.39 / 0.02 | 0.85 / 0.64 | 0.89 / 0.76 | 0.2–0.3 |
| perspective 5 % | 0.91 / 0.81 | 0.91 / 0.84 | 0.84 / 0.65 | ~0.2 |
| perspective 8 % | 0.88 / 0.75 | 0.93 / 0.85 | 0.85 / 0.68 | — |
| photo 2 % (vignette, texture, JPEG 45, 1.5°) | 0.94 / 0.88 | 0.93 / 0.88 | 0.90 / 0.79 | 0.76–0.94 |
| photo 3.5 % (vignette 0.6, texture 0.5, −2.5°, salt) | 0.58 / 0.22 | 0.06 / −0.17 | 0.21 / −0.10 | 0.27 |
| rotation 3° / scan combo | 0.93 / 0.88 | 0.92 / 0.82 | 0.92 / 0.82 | unchanged |

Pure perspective up to 8 % is now digitised in 10 of 12 cases (one 3.5 % seed
fails on a mis-snapped side). Photos with a vignette and paper grain at ≥ 3.5 %
still fail: on a grainy page the line snap is unreliable and the residual is
beyond the vanishing-point step. Real photographs remain the open item — the
ECG-Image-Database and the 2024 challenge images are not reachable (404) from
this environment.

**Pages rendered from real PTB-XL signals (100 random records of the 1692, 6 px/mm,
calibration pulse, auto-calibrated).** The digital record goes through the pipeline,
its filtered signal is rendered as a 3×4 page, the page is degraded and digitised,
and the digitised signal goes through the same pipeline. Agreement digital ↔ page:
per-lead r, HR / axis / QRS-width deltas, and the Jaccard overlap of the scored
finding codes — the clinically meaningful number.

| Page | extraction | fail | mean r | r p10 | HR MAE (≤ 3 bpm) | axis MAE (≤ 15°) | QRS MAE | codes Jaccard | identical |
|---|---|--:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| clean | column median (v3) | 0 | 0.90 | 0.82 | 4.1 (80 %) | 11° (87 %) | 25 ms | 0.57 | 18 % |
| clean | **tracking (D47)** | 0 | **0.94** | 0.89 | 2.6 (80 %) | 9° (90 %) | **12 ms** | **0.69** | **31 %** |
| scan (1°, blur 1 px, JPEG 60) | column median | 0 | 0.89 | 0.82 | 3.2 (81 %) | 12° (84 %) | 24 ms | 0.48 | 9 % |
| scan | tracking | 0 | 0.92 | 0.88 | 4.0 (77 %) | 13° (84 %) | 14 ms | 0.60 | 21 % |
| photo (2 % perspective, vignette, texture, JPEG 45, 1.5°) | column median | 0 | 0.61 | 0.18 | 9.8 (39 %) | 31° (46 %) | 40 ms | 0.33 | 0 % |
| photo | tracking | 0 | 0.62 | 0.18 | 9.0 (39 %) | 29° (46 %) | 30 ms | 0.38 | 3 % |
| photo strong (3.5 %, −2.5°, salt) | column median | 11 | 0.24 | 0.02 | 11 (37 %) | 64° (16 %) | 45 ms | 0.31 | 0 % |
| photo strong | tracking | 6 | 0.24 | 0.01 | 13 (30 %) | 62° (18 %) | 39 ms | 0.30 | 4 % |

Reading, first run (column median): on a clean page from a real record the
waveform and the rhythm survive (r 0.90, HR within 3 bpm in 80 %, axis within 15°
in 87 %) but the **finding set does not**: only 18 % of pages reproduce the digital
record's codes exactly and the average overlap is 0.57. The dominant cause was a
systematic **QRS widening of +24 ms (median; +31 mean)** — 15 clean pages checked
column by column: a 3–4-px stroke drawn along a steep edge spreads that edge's rows
into the neighbouring columns (each column's ink run is the true sub-column
segment plus slope × half the stroke width above and below), so the column before
the QRS onset already reads 10–25 px up and the QRS comes out 1–2 columns wider on
each side; normal records crossed the 120-ms BBB threshold (`lbbb` on 6 of 15) and
QT shifted +15–25 ms. A second cause showed up in the same pages: deep S waves
(V2/V3 at −2.5 mV) run into the row below, and the neighbour's column median
swallowed them (spikes in V3 and in the rhythm strip, missed R peaks, false AF).

**D47 — tracking instead of the median.** Candidates per column come from the ink
skeleton (a steep edge stays in its own column; the lateral spread next to a
skeleton column stays empty; the raw runs serve only at line ends), a Viterbi pass
follows each panel's own trace (continuity plus a weak pull to the panel's
baseline, forced far-off choices dropped, the box extended half a row up and down
so a deep S is followed rather than clipped), and the R/S apex is read from the raw
ink's symmetric extreme profile (the skeleton's apex is off-centre on a thick
spike). Two alternatives were measured and rejected: horizontal erosion by the
stroke width erases steep strokes outright (they are 3 px wide horizontally, 4 px
tall), and a distance-transform ridge with a "skip" state is symmetric but skips
genuine spikes (R under blur 1200 → 200 µV). Result on the same 100 records:
QRS MAE 25 → 12 ms on clean pages (15-record check: median bias +24 → +8 ms,
mean +31 → +15), r 0.90 → 0.94, finding-set overlap 0.57 → 0.69, identical sets
18 → 31 %; scan 0.48 → 0.60. The photo rows barely move: at 2 % perspective the
limit is the de-warp, not the extraction. The residual +8 ms bias is the flank
columns next to an apex, which are read from the raw run midpoints (the skeleton
sits at the base of a thick spike there); reading them from the skeleton removes
the bias (median +1 ms) but costs the blur and combo pages (min r 0.75) — the
current setting keeps every synthetic page at or above the median's numbers. What
still separates a page from its record on the remaining 69 % of clean pages is
rhythm semantics at the margins (a P/R ratio at the AF gate, one dropped R peak)
and QRS width on wide-complex records; real photographs remain unmeasured.

### Pages: amplitude under-read and the AF gate (open item 3, D52)

After D47 the code disagreements on the 100 clean pages were dominated by
voltage, not width: `low_qrs_voltage` appeared on 19 pages, `lvh` was lost on 7,
and AF appeared on 6. Twenty pages checked column by column, plus the synthetic
page, gave three independent causes; none needed a threshold.

1. **Adjacent apex columns demoted each other.** A spike tip two columns wide
   is two adjacent apexes (each the middle of its own three-column plateau);
   each marked the other as a flank, and the flank rule overwrote both with the
   raw-run midpoint. A 26-px R on the synthetic page read as 17 px (lead III
   514 → 328 µV). Apex columns are now excluded from the flank pass.
2. **The P/T oversense gate compared beats measured on different lead sets.**
   On a 3×4 page each time column carries four leads and the other eight are
   zero; the spatial-velocity envelope of a beat under aVR/aVL/aVF is a
   fraction of one under V1–V3, so a whole column of true beats was rejected
   against the record-wide 90th percentile — HR fell by a quarter, RR
   variability tripled, false AF (3 of the 6 AF pages). Beats are now grouped by
   the leads valid at their time and judged within the group.
3. **The limb amplitude was a pooled median over (lead, beat) pairs.** On a
   page lead II contributes ten beats and I/III two each, so the "limb"
   amplitude was the lead-II amplitude (one page read 881 → 225 µV with the
   waveform itself at 0.83 of the record). It is now the median of per-lead
   medians; on digital records the change is within a point (PTB-XL
   `low_qrs_voltage` 0.74 / 0.91 → 0.73 / 0.91, 664 0.72 / 0.86 → 0.73 / 0.86).

Twenty clean pages, page / record (6 px/mm):

| | limb amplitude ratio, median (p10) | per-lead peak-to-peak ratio | QRS bias | mean r | codes Jaccard | `low_qrs_voltage` added | `lvh` lost |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| D47 | 0.87 (0.67) | 0.92 | +13.5 ms | 0.930 | 0.75 | 3 | 2 |
| + gate per lead set, balanced limb amplitude | 0.83 (0.68) | 0.92 | +15.1 ms | 0.930 | 0.78 | 1 | 2 |
| + apex fix (D52) | **0.91 (0.80)** | **0.97** | **+9.7 ms** | 0.932 | 0.78 | 1 | 2 |
| relaxed apex condition (no outer ±3 px margin), rejected | 0.86 (0.67) | 0.93 | +19.5 ms | 0.928 | 0.76 | 3 | 2 |
| 12 px/mm (≈ 300 dpi), before the apex fix | 0.92 (0.84) | 0.96 | +12.7 ms | 0.969 | 0.77 | 1 | 2 |
| 12 px/mm, apex fix | **0.94 (0.85)** | **0.97** | +12.3 ms | **0.969** | 0.76 | 1 | 2 |

What remains is the render resolution: a narrow low complex (an S of −2 mm
lasting 20 ms) is one column at 6 px/mm and loses up to half its depth after
resampling and the band-pass; the 150-Hz-equivalent page, not the tracker, is
the limit there. The remaining AF pages were already at the gate on the digital
record (RR CV 0.17–0.24 with a P/R ratio that the page moves from 0.5 to 0.3).
**100 clean pages, re-run with all of D52** (same records and script as the D47
code comparison): finding sets identical on **36 → 51** of 100 pages; QRS bias
median **+8 → +4 ms** (mean +9 → +6, MAE 12.4 → 12.2); `low_qrs_voltage` added on
**19 → 5** pages; `lvh` lost 7 → 6; `lbbb` added 4 → 1; AF added 6 → 5. The five
AF pages: three records are at the AF gate on the digital record itself (RR CV
0.17–0.24 with a P/R ratio the page lowers from 0.5 to 0.3), two lose several
beats to the rhythm-strip detector (a near-isoelectric lead II of 0.2 mV with a
20-ms S, and an LBBB record whose P waves are taken for R) — the one-beat case
(RR CV 0.07 → 0.28) is healed by the regularity guard — an isolated RR of
1.8–2.2 medians counts as two intervals for the regularity statistic when such
intervals are at most a quarter of the series; its cost on digital records is one
PTB-XL AF record (0.545 / 0.969 → 0.537 / 0.969; 664 AF 0.80 / 0.728 → 0.80 /
0.733).

Same 100 records, all four page variants, D47 → D47 + D48 + D52 (the photo rows
also carry the D48 quadrilateral de-warp, which the D47 run predates):

| Page | fail | mean r | r p10 | HR MAE (≤ 3 bpm) | axis MAE (≤ 15°) | QRS MAE | codes Jaccard | identical |
|---|--:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| clean | 0 | 0.94 → 0.94 | 0.89 | 2.6 (80 %) → **1.5 (88 %)** | 9° (90 %) → **7° (93 %)** | 12.4 → 12.2 ms | 0.69 → **0.78** | 31 → **51 %** |
| scan (1°, blur 1 px, JPEG 60) | 0 | 0.92 → 0.92 | 0.87 | 4.0 (77 %) → **1.8 (88 %)** | 13° (84 %) → **7° (93 %)** | 14.1 → 12.6 ms | 0.60 → **0.66** | 21 → **30 %** |
| photo (2 %, vignette, texture, JPEG 45, 1.5°) | 0 | 0.62 → 0.65 | 0.18 → 0.20 | 9.0 (39 %) → **6.2 (52 %)** | 29° (46 %) → **14° (80 %)** | 29.7 → 24.9 ms | 0.38 → **0.46** | 3 → 10 % |
| photo strong (3.5 %, −2.5°, salt) | 6 | 0.24 → 0.23 | 0.01 | 13.5 (30 %) → 8.5 (35 %) | 62° (18 %) → 26° (60 %) | 39.1 → 32.6 ms | 0.30 → 0.37 | 4 → 7 % |

The waveform correlation does not move (the trace was already right where it
was found); what moves is what the measurements make of it — HR within 3 bpm on
88 % of clean and scanned pages (the per-set oversense gate and the missed-beat
guard), axis within 15° on 93 % (the six-lead net-deflection axis of D49 is
robust to a single under-read lead), and the finding sets. The 2 % photo is now
usable for rhythm and axis (axis MAE 29° → 14°) but not yet for the finding set
(0.46); the strong photo remains beyond the de-warp.



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
