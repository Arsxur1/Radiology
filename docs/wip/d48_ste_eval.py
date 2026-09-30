import csv, ast, sys, collections
SP = "/tmp/claude-0/-home-user-Radiology/1ba6cb74-4ecb-5b0a-873b-e7131460a121/scratchpad/"
full = {r["ecg_id"]: r for r in csv.DictReader(open(SP + "ptbxl_full.csv"))}
PAIRS = [("I", "aVL"), ("II", "III"), ("III", "aVF"), ("II", "aVF"), ("V1", "V2"), ("V2", "V3"), ("V3", "V4"), ("V4", "V5"), ("V5", "V6")]
RECIP = {"inferior": ({"II", "III", "aVF"}, {"I", "aVL"}), "anterior": ({"V1", "V2", "V3", "V4"}, {"II", "III", "aVF"}), "lateral": ({"I", "aVL", "V5", "V6"}, {"III", "aVF"})}
def udmi(st, sex, age, key=0, recip=False):
    el = set(); dep = set()
    for lead, v in st.items():
        x = v[key]
        if x is None: continue
        if lead in ("V2", "V3"):
            thr = 150 if sex == "female" else (250 if (age or 50) < 40 else 200)
        else: thr = 100
        if x >= thr: el.add(lead)
        if x <= -50: dep.add(lead)
    hit = [p for p in PAIRS if p[0] in el and p[1] in el]
    if not hit: return False
    if not recip: return True
    for wall, (leads, rec) in RECIP.items():
        if len(el & leads) >= 2 and dep & rec: return True
    return False
def load(n, kind):
    out = []
    for r in csv.DictReader(open(SP + f"ac_{n}.csv")):
        st = ast.literal_eval(r["st_by_lead"]) if r["st_by_lead"] else {}
        codes = set(r["codes"].split("|"))
        if kind == "ptbxl":
            fr = full.get(r["id"], {}); acute = fr.get("infarction_stadium1", "") in ("Stadium I", "Stadium I-II")
            lab = "STE_" in codes or acute
        else: lab = "164931005" in codes
        age = float(r["age"]) if r["age"] not in ("", "None") else None
        out.append((lab, r["sex"] if r["sex"] not in ("", "None") else None, age, st, "st_elevation" in r["alerts"].split()))
    return out
for n, kind in (("f10", "ptbxl"), ("ptbxl", "ptbxl"), ("gaho", "sn"), ("cpf", "sn")):
    R = load(n, kind); pos = [r for r in R if r[0]]; neg = [r for r in R if not r[0]]
    def rep(name, fn):
        tp = sum(fn(r) for r in pos); fp = sum(fn(r) for r in neg)
        return f"{name}: sens {tp}/{len(pos)} ({tp/max(len(pos),1):.2f}), alerts on unlabelled {fp/len(neg):.1%}"
    print(f"{n} ({len(R)}, labelled STE {len(pos)})")
    print("   " + rep("current (J+60)", lambda r: r[4]))
    print("   " + rep("UDMI J-point", lambda r: udmi(r[3], r[1], r[2], 0)))
    print("   " + rep("UDMI J-point + reciprocal", lambda r: udmi(r[3], r[1], r[2], 0, True)))
    print("   " + rep("UDMI thresholds at J+60", lambda r: udmi(r[3], r[1], r[2], 1)))
