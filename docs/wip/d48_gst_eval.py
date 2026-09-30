import csv, sys, collections
SP = "/tmp/claude-0/-home-user-Radiology/1ba6cb74-4ecb-5b0a-873b-e7131460a121/scratchpad/"
exec(open(SP + "ste_eval.py").read().split("def load")[0].split("full = ")[0])
exec("PAIRS" + open(SP + "ste_eval.py").read().split("PAIRS", 1)[1].split("def load")[0])
full = {r["ecg_id"]: r for r in csv.DictReader(open(SP + "ptbxl_full.csv"))}
f = lambda x: float(x) if x not in ("", None, "None") else None
by = collections.defaultdict(dict)
for r in csv.DictReader(open(SP + sys.argv[1])):
    by[r["ecg_id"]][r["lead"]] = r
def cur_rule(st):   # current rule: J+60 >= 100 (200 in V2-V3) in 2 leads of one territory
    terr = [{"II", "III", "aVF"}, {"V1", "V2", "V3", "V4"}, {"I", "aVL", "V5", "V6"}]
    el = {l for l, v in st.items() if v is not None and v >= (200 if l in ("V2", "V3") else 100)}
    return any(len(el & t) >= 2 for t in terr)
rows = []
for eid, leads in by.items():
    fr = full[eid]; c = set(eval(fr["scp_codes"])); sex = "female" if fr["sex"] == "1" else "male"; age = f(fr["age"])
    lab = "STE_" in c or fr.get("infarction_stadium1", "") in ("Stadium I", "Stadium I-II")
    cur = {l: f(v["cur_j60"]) for l, v in leads.items()}; g60 = {l: f(v["g_j60"]) for l, v in leads.items()}
    gj = {l: [f(v["g_j"]), None] for l, v in leads.items()}; slj = {l: [f(v["sl_j"]), None] for l, v in leads.items()}
    rows.append(dict(lab=lab, cur=cur_rule(cur), g60=cur_rule(g60), gudmi=udmi(gj, sex, age, 0), sl=udmi(slj, sex, age, 0)))
pos = [r for r in rows if r["lab"]]; neg = [r for r in rows if not r["lab"]]; slp = [r for r in rows if r["sl"]]; sln = [r for r in rows if not r["sl"]]
print(f"{sys.argv[1]}: records {len(rows)}, labelled STE {len(pos)}, 12SL-UDMI positive {len(slp)} ({len(slp)/len(rows):.1%})")
for k, name in (("cur", "current rule, per-beat J+60"), ("g60", "current rule, median-beat J+60"), ("gudmi", "UDMI J-point, median beat")):
    tp = sum(r[k] for r in pos); fp = sum(r[k] for r in neg)
    a = sum(r[k] for r in slp); b = sum(r[k] for r in sln)
    print(f"  {name:32s}: label sens {tp}/{len(pos)} ({tp/max(len(pos),1):.2f}), alerts on unlabelled {fp/len(neg):.1%} | vs 12SL-UDMI: sens {a/max(len(slp),1):.2f}, false {b/len(sln):.1%}")
