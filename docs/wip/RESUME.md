# Resume point — D48 (critical-alert burden: ST elevation)

Branch state: clean, all validated work committed (last shipped: D45; D46/D47 documented negatives).

## Critical-alert census (current code)
Records with any §10 critical alert: fold 10 30.7%, fresh Georgia 29.3%, fresh CPSC 30.6%.
Top sources: st_elevation 9–17%, peaked_t_waves 10–15%, long_qt 6–16% (already tiered, D44).

## D48 in progress — ST measured on the 12-lead median beat
- Dev (PTB-XL dev, 16,356 leads) vs GE 12SL J-point ST: current per-beat MAE 138 uV (28% of leads
  off > 100 uV); median-beat MAE 37 uV (5.8%). Big measurement gain.
- Fold 10: 12SL J-point UDMI criterion met on 6.1% of records vs our STE alert on 16.7%.
- Dev rule comparison (1363 records, 33 labelled STE_/acute MI; 12SL-UDMI positive 11.6%):
  current per-beat J+60 rule: label sens 0.42, alerts on unlabelled 20.1%;
  UDMI J-point on median beat: label sens 0.27, alerts 12.6%, vs 12SL-UDMI sens 0.89 / false 2.9%.
- Proposed (product decision needed after held-out): keep the st_elevation finding as today,
  raise the CRITICAL STEMI alert only when the 4th-UDMI J-point criterion holds on the median beat.
- Pre-registration: `d48_prereg.txt` (held-out = PTB-XL fold 9, never used for ST).
- Scripts: `d48_gst_dump.py` (median-beat vs per-beat vs 12SL ST dump), `d48_gst_eval.py`,
  `d48_ste_eval.py`. They expect PTB-XL+ `12sl_features.csv` and PTB-XL records locally.

## Next steps
1. Download PTB-XL fold 9 (records500), run `d48_gst_dump.py` on it, score against the
   pre-registered criteria (MAE <= 50 uV; agreement with 12SL-UDMI sens >= 0.80, false <= 5%).
2. Ask the product owner whether to tier the STEMI critical alert; implement + document D48.
3. Then peaked-T (hyperkalaemia pattern) critical alert, 10–15% of records; no hyperK labels exist,
   so only measurement/tiering is possible.

## Open product decision
D42/D43: pre-QRS P consistency cuts false AF ~70% but costs 3–5% AF sensitivity — adopt or not.
