"""Smallest feature set that keeps holdout AUC: rank lean features by mean |SHAP| (on CV folds), test top-k."""
import sys, json, pickle; sys.path.insert(0, "src")
import shap
from common import *
cfg = pickle.load(open("data/out/final_lean.pkl", "rb"))
SN = load(strict=True)
k = int(sys.argv[1]) if len(sys.argv) > 1 else 0
if k == 0:
    tr = SN[SN.created_at < HOLDOUT_FROM]
    cust = tr.customer_id.unique(); perm = pd.Series(np.random.default_rng(0).permutation(len(cust)), index=cust)
    imp = None
    for a, b in GroupKFold(5).split(tr, groups=tr.customer_id.map(perm)):
        p, m, Xva = fit_predict(tr.iloc[a], tr.iloc[b], cfg["feats"], cfg["rounds"], cfg["params"])
        s = pd.Series(np.abs(shap.TreeExplainer(m).shap_values(Xva)).mean(0), index=Xva.columns)
        imp = s if imp is None else imp + s
    # fold _te/_cnt back to the source column
    src = imp.groupby(lambda c: c[:-3] if c.endswith("_te") else c[:-4] if c.endswith("_cnt") else c).sum().sort_values(ascending=False)
    src.to_csv("data/out/feature_rank.csv"); print(src.head(45).round(4).to_string())
else:
    rank = pd.read_csv("data/out/feature_rank.csv", index_col=0).iloc[:, 0]
    f = list(rank.index[:k])
    cv = cv_score(SN, f, seeds=(0,), rounds=cfg["rounds"], params=cfg["params"])
    ho = holdout_score(SN, f, rounds=cfg["rounds"], params=cfg["params"])[0]
    print(json.dumps(dict(k=k, cv=round(cv["auc"], 4), ho=round(ho["auc"], 4), ll=round(ho["ll"], 4))))
