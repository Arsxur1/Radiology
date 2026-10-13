"""CardioLens — ECG detection and interpretation system (TZ CardioLens).

Six replaceable layers behind data contracts (TZ §3):
  layer1_ingestion  -> layer2_preprocess -> layer3_measure
  -> layer4_findings -> layer5_causal -> layer6_interpret
with a cross-cutting ``safety`` module (TZ §10) and ``outputs`` (TZ §11).
"""

__version__ = "0.1.0"
