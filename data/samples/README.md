# Sample ECG data

Small bundled samples so the parsers and pipeline have real files to run on
offline. These are **synthetic** (generated deterministically), not clinical.

| File | Format | Load with |
|------|--------|-----------|
| `sample_ecg.hea` + `.dat` | WFDB (mV) | `load_wfdb("data/samples/sample_ecg")` |
| `sample_ecg.csv` | CSV (µV, header row) | `load_csv(path, sampling_rate_hz=500, unit="uv")` |

12 leads, 500 Hz, 5 s.

Regenerate with the snippet in the repo README, or point the demo at a real
clinical record:

```bash
python -m cardiolens.demo --wfdb /path/to/PTBXL/records500/00000/00001_hr
```

Real clinical data (PTB-XL, MIT-BIH, LUDB) is **not** bundled — see the top-level
README (TZ §15).
