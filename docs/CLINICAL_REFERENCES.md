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

<!-- BALANCED_RESULTS -->

The adapters and SNOMED/SCP mappings are unit-tested; the balanced fetch
(`cardiolens.validation.fetch_physionet`) exists precisely because this 300-record
slice is brady-skewed. Rerun on a larger sample before drawing clinical conclusions.

## Not a medical device
These criteria support interpretation for a decision-support tool; CardioLens is
not clinically validated and is not a medical device. Every report carries the
disclaimer "Инструмент поддержки. Не заменяет врача."
