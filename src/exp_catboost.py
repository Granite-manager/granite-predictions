"""CatBoost on the strict feature set, time holdout, high-card columns as native cat features."""
import sys, time, json; sys.path.insert(0, "src")
from common import *
from catboost import CatBoostClassifier, Pool
SN = load(strict=True); G = groups(SN)
feats = sorted(set(sum([v for k, v in G.items() if k != "text"], [])))
cats = [f for f in feats if f in HIGH_CARD or f in LOW_CARD]
X = SN[feats].copy()
for c in cats:
    X[c] = X[c].astype(str).fillna("nan")
tr = SN.created_at < HOLDOUT_FROM; va = ~tr
t0 = time.time()
m = CatBoostClassifier(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5, thread_count=4, verbose=0, random_seed=0)
m.fit(Pool(X[tr], SN.label[tr], cat_features=cats, weight=SN.w[tr]))
p = m.predict_proba(Pool(X[va], cat_features=cats))[:, 1]
res = dict(model="catboost", ho_auc=roc_auc_score(SN.label[va], p, sample_weight=SN.w[va]),
           ho_ll=log_loss(SN.label[va], np.clip(p, 1e-4, 1 - 1e-4), sample_weight=SN.w[va]), secs=round(time.time() - t0))
print(json.dumps(res))
json.dump(res, open("data/out/exp_catboost.json", "w"))
