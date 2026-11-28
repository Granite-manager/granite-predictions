"""Why some companies are predicted well and others not: per-company drop-group ablation on the holdout,
per-feature univariate AUC within each company, and spread of win rates across sources and reps."""
import sys, json, pickle; sys.path.insert(0, "src")
from common import *
cfg = pickle.load(open("data/out/final_lean.pkl", "rb"))
SN = load(strict=True); G = groups(SN)
F = cfg["feats"]
CO = {"GMQ": [4, 12, 13], "Indy": [1], "Columbus": [3], "Lex": [2], "Empire": [7]}
tr, va = SN[SN.created_at < HOLDOUT_FROM], SN[SN.created_at >= HOLDOUT_FROM]
def by_co(p):
    v = va.assign(p=p); out = {}
    for k, cs in CO.items():
        x = v[v.company_id.astype(int).isin(cs)]
        out[k] = roc_auc_score(x.label, x.p, sample_weight=x.w)
    return out
base = by_co(fit_predict(tr, va, F, cfg["rounds"], cfg["params"])[0])
print("full", {k: round(v, 3) for k, v in base.items()})
for g, cols in G.items():
    drop = [f for f in F if f in cols and f not in G["base"]]
    if not drop: continue
    r = by_co(fit_predict(tr, va, [f for f in F if f not in drop], cfg["rounds"], cfg["params"])[0])
    print(f"drop {g:18s}", {k: round(base[k] - r[k], 3) for k in r}, flush=True)
