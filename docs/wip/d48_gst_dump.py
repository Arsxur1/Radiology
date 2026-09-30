"""Global (median-beat) ST vs current per-beat ST vs 12SL. argv: out.csv root"""
import sys, csv, ast, os, warnings
import numpy as np
warnings.filterwarnings("ignore"); sys.path.insert(0, "/home/user/Radiology/src")
from cardiolens.layer1_ingestion import load_wfdb
from cardiolens.pipeline import run_pipeline
from cardiolens.layer6_interpret.stub import StubProvider
from cardiolens.layer3_measure.global_beat import median_beats, global_fiducials
SP = "/tmp/claude-0/-home-user-Radiology/1ba6cb74-4ecb-5b0a-873b-e7131460a121/scratchpad/"
L = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
sl = {r["ecg_id"]: r for r in csv.DictReader(open(SP + "ptbxlplus/12sl_features.csv"))}
root = sys.argv[2]; w = csv.writer(open(sys.argv[1], "w", newline="")); w.writerow(["ecg_id", "lead", "sl_j", "sl_j80", "cur_j", "cur_j60", "cur_j80", "g_j", "g_j60", "g_j80", "g_j40"])
for r in csv.DictReader(open(root + "/ptbxl_database.csv")):
    p = root + "/" + r["filename_hr"]
    if not os.path.exists(p + ".dat") or r["ecg_id"] not in sl: continue
    s = load_wfdb(p); s.patient.age_years = float(r["age"]) if r["age"] else None; s.patient.sex = "female" if r["sex"] == "1" else "male"
    try: res = run_pipeline(s, mode="A", provider=StubProvider())
    except Exception: continue
    sig, m = res.signal, res.measurements; fs = sig.sampling_rate_hz
    X, a = median_beats(sig.signal, m.r_peaks_samples, fs)
    if X is None: continue
    rp = np.asarray(m.r_peaks_samples, float); rr = float(np.median(np.diff(rp))) / fs
    fid = global_fiducials(X, a, fs, rr); off = fid["qrs_off"]
    lo, hi = max(0, a - int(0.30 * fs)), a - int(0.03 * fs); w20 = max(2, int(0.02 * fs))
    cur = {x.lead: x for x in m.morphology.st}
    for li, lead in enumerate(sig.lead_names):
        if lead not in L: continue
        x = X[li]; best, base = np.inf, 0.0
        for s0 in range(lo, max(lo + 1, hi - w20)):
            seg = x[s0:s0 + w20]; e = float(np.ptp(seg))
            if e < best: best, base = e, float(np.mean(seg))
        g = lambda ms: round(float(x[min(len(x) - 1, off + int(ms / 1000 * fs))] - base), 1)
        c = cur.get(lead)
        f = lambda v: None if v in ("", None) else round(float(v) * 1000, 1)
        w.writerow([r["ecg_id"], lead, f(sl[r["ecg_id"]].get(f"ST_Amp_{lead}")), f(sl[r["ecg_id"]].get(f"ST_Amp80ms_{lead}")),
                    c.j_point_uv if c else None, c.j60_uv if c else None, c.j80_uv if c else None, g(0), g(60), g(80), g(40)])
