import sys, json, pickle; sys.path.insert(0, "src")
from common import *
cfg = pickle.load(open("data/out/final_lean.pkl", "rb"))
SN = load(strict=True)
rank = list(pd.read_csv("data/out/feature_rank.csv", index_col=0).index[:20])
A = [f for f in rank if f not in ("off_h", "lead_campaign")]
V = {"A": A, "B": [f for f in A if f != "user_id"], "C": [f for f in A if f != "last_any_ago_d"], "D": [f for f in A if f not in ("user_id", "last_any_ago_d")]}
name = sys.argv[1]; f = V[name]
cv = cv_score(SN, f, seeds=(0, 1), rounds=cfg["rounds"], params=cfg["params"])
ho = holdout_score(SN, f, rounds=cfg["rounds"], params=cfg["params"])[0]
print(json.dumps(dict(v=name, n=len(f), cv=round(cv["auc"], 4), ho=round(ho["auc"], 4), ll=round(ho["ll"], 4), feats=f)))
