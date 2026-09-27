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
brady-skewed, one region — **not** representative of the full ~45k set). These are real
numbers, optimistic bounds do **not** apply:

| Finding | Sens | Spec | n |
|---------|-----:|-----:|--:|
| sinus_rhythm | 0.97 | 0.78 | 39 |
| sinus_bradycardia | 0.94 | 0.98 | 155 |
| sinus_tachycardia | 1.00 | 0.91 | 30 |
| atrial_fibrillation | 0.62 | 0.94 | 61 |
| atrial_flutter | 0.00 | 1.00 | 15 |
| first_degree_av_block | 0.09 | 0.95 | 11 |
| rbbb | 0.18 | 0.98 | 11 |
| lbbb | 0.17 | 0.79 | 6 |
| left_axis_deviation | 1.00 | 0.94 | 10 |
| right_axis_deviation | 0.75 | 0.98 | 4 |
| low_qrs_voltage | 0.86 | 0.89 | 7 |
| st_elevation | 0.67 | 0.76 | 3 |
| st_depression | 0.83 | 0.67 | 12 |
| long_qt | 1.00 | 0.66 | 1 |

Macro sensitivity **0.65**. (PR/QT interval MAE is not reported for this dataset — it
carries diagnostic labels only, no gold interval annotations.)

**What real data revealed (honest limitations):**

- **Atrial fibrillation** initially scored **0.0** sensitivity: neurokit's delineation
  interpolates a P onset for nearly every beat, so the onset-only `p_wave_ratio`
  saturated at 1.0 and the AF gate never fired. Fixed by gating P-presence on the
  P-region **amplitude** (`P_PRESENCE_FRACTION` in `layer3_measure/measure.py`,
  calibrated on this set: sinus P/R ≈ 0.21 vs AF ≈ 0.10) → **0.62** sensitivity at
  0.94 specificity. Remaining misses are mostly AF with a controlled, near-regular
  ventricular response that doesn't clear the RR-irregularity gate.
- **Atrial flutter & complete AV block** score 0: the rules exist but the
  `atrial_rate_bpm` / `av_dissociation` measurements that feed them are not yet
  implemented (declared in the contract, always `None`). Detecting the atrial (F-wave)
  rate independently of the QRS is the next signal-processing task.
- **RBBB/LBBB** (0.18/0.17) and **first-degree AV block** (0.09) underperform on real
  morphology/PR delineation — the synthetic unit tests pass, but real signals need more
  robust criteria.
- **ST elevation/depression** are over-called (spec 0.76/0.67); the J-point threshold
  needs per-lead calibration on real data.

The adapters and SNOMED/SCP mappings are unit-tested; rerun on a larger, more balanced
sample (or the full dataset) before drawing clinical conclusions.

## Not a medical device
These criteria support interpretation for a decision-support tool; CardioLens is
not clinically validated and is not a medical device. Every report carries the
disclaimer "Инструмент поддержки. Не заменяет врача."
