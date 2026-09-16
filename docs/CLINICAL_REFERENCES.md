# Clinical references and encoded criteria

CardioLens encodes established ECG diagnostic criteria as **data** — in the
age/sex norm tables (`configs/norms/`) and the pathophysiology rule base
(`configs/rules/pathophysiology.yaml`) — not as hard-coded logic (TZ §0, §8).
This file records the clinical sources those criteria are drawn from and maps
each encoded criterion to its source.

> **Honesty note.** The references below were provided as learning material. In
> this environment outbound network egress is restricted to package registries,
> so these sources could **not** be fetched/scraped. The criteria encoded here
> are **standard, well-established textbook ECG values** (medical facts, not
> copyrighted expression) that these sources teach; nothing was copied verbatim.
> Where a criterion is threshold-based it lives in a config file and can be
> edited to match a given source or local protocol.

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

## Not a medical device
These criteria support interpretation for a decision-support tool; CardioLens is
not clinically validated and is not a medical device. Every report carries the
disclaimer "Инструмент поддержки. Не заменяет врача."
