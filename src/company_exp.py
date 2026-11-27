import sys, json; sys.path.insert(0, "src")
from common import *
SN = load(strict=True, company=True); G = groups(SN)
full = sorted(set(sum([v for k, v in G.items() if k != "company_aware"], [])))
for name, f in [("full", full), ("full+company_aware", full + G["company_aware"]),
                ("full-company_id", [c for c in full if c != "company_id"]),
                ("full+company_aware-company_id", [c for c in full + G["company_aware"] if c != "company_id"])]:
    cv = cv_score(SN, f, seeds=(0,)); ho = holdout_score(SN, f)[0]
    print(json.dumps(dict(cfg=name, cv_auc=round(cv["auc"], 4), ho_auc=round(ho["auc"], 4), ho_ll=round(ho["ll"], 4))), flush=True)
for name, f in [("loco full-company_id", [c for c in full if c != "company_id"]),
                ("loco +company_aware-company_id", [c for c in full + G["company_aware"] if c != "company_id"])]:
    print(name, {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in loco_score(SN, f).items()}, flush=True)
