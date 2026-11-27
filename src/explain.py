"""Per-deal explanation: calibrated probability + top SHAP reasons for one holdout snapshot (anonymised)."""
import sys, pickle, json; sys.path.insert(0, "src")
import shap
from sklearn.isotonic import IsotonicRegression
from common import *
cfg = pickle.load(open("data/out/final_lean.pkl", "rb"))
SN = load(strict=True)
tr, va = SN[SN.created_at < HOLDOUT_FROM], SN[SN.created_at >= HOLDOUT_FROM]
raw, m, Xva = fit_predict(tr, va, cfg["feats"], cfg["rounds"], cfg["params"])
pred = pd.read_parquet("data/out/holdout_pred_lean.parquet").set_index("sid")
va = va.assign(p=va.sid.map(pred.p).values)
sv = shap.TreeExplainer(m).shap_values(Xva)
for target in [0.45, 0.08, 0.75]:
    cand = va[(va.off_h == 72)]
    i = (cand.p - target).abs().idxmin()
    j = va.index.get_loc(i)
    s = pd.Series(sv[j], index=Xva.columns)
    top = pd.concat([s.sort_values(ascending=False).head(4), s.sort_values().head(4)])
    print(json.dumps(dict(p=round(float(va.p.loc[i]), 3), label=int(va.label.loc[i]), company=int(va.company_id.loc[i]),
        reasons={k: [round(float(v), 3), str(Xva.iloc[j][k])] for k, v in top.items()})))
