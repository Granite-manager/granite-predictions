"""Group ablation: each group alone on top of base, and full model minus each group."""
import sys, json; sys.path.insert(0, "src")
from common import *
strict = "--loose" not in sys.argv
SN = load(strict=strict)
G = groups(SN)
G = {k: v for k, v in G.items() if v}
full = sorted(set(sum(G.values(), [])))
res = []
def run(name, kind, feats):
    cv = cv_score(SN, feats, seeds=(0, 1))
    ho = holdout_score(SN, feats)[0]
    row = dict(group=name, kind=kind, n_feats=len(feats), cv_auc=cv["auc"], cv_sd=cv["auc_sd"], cv_ll=cv["ll"], ho_auc=ho["auc"], ho_ll=ho["ll"])
    res.append(row); print(json.dumps(row), flush=True)
run("base", "ref", G["base"]); run("full", "ref", full)
for k, v in G.items():
    if k == "base": continue
    run(k, "alone", sorted(set(G["base"] + v)))
    run(k, "drop", [f for f in full if f not in v])
pd.DataFrame(res).to_csv(f"data/out/ablation_{'strict' if strict else 'loose'}.csv", index=False)
