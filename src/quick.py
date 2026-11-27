import sys; sys.path.insert(0,"src")
from common import *
SN = load()
G = groups(SN)
print("rows", len(SN), "deals", SN.deal_id.nunique(), "pos rate (deal)", SN.groupby("deal_id").label.first().mean().round(3))
print({k: len(v) for k,v in G.items()})
full = sorted(set(sum(G.values(), [])))
for name, f in [("base", G["base"]), ("base+stage", G["base"]+G["stage"]), ("full", full)]:
    r = cv_score(SN, f, seeds=(0,))
    h = holdout_score(SN, f)[0]
    print(name, {k: round(v,4) for k,v in r.items()}, "holdout", {k: round(v,4) for k,v in h.items()})
