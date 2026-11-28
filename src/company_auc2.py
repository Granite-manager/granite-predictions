import sys, pickle; sys.path.insert(0, "src")
from common import *
cfg = pickle.load(open("data/out/final_lean.pkl", "rb"))
SN = load(strict=True)
P = pd.read_parquet("data/out/holdout_pred_lean.parquet").set_index("sid")
va = SN[SN.sid.isin(P.index)].copy(); va["p"] = va.sid.map(P.p)
CO = {"GMQ": [4, 12, 13], "Indy": [1], "Columbus": [3], "Lex": [2], "Empire": [7]}
A = lambda x, s: roc_auc_score(x.label, s, sample_weight=x.w) if x.label.nunique() > 1 else np.nan
for k, cs in CO.items():
    x = va[va.company_id.astype(int).isin(cs)]
    per = [A(g, g.p) for _, g in x.groupby("off_h") if g.label.nunique() > 1 and g.label.sum() >= 5]
    first = x[x.off_h == 1]
    lk = x.groupby("label_kind").deal_id.nunique().to_dict()
    print(f"{k:9s} pooled={A(x, x.p):.3f}  age_only={A(x, -x.age_days):.3f}  mean_within_age={np.mean(per):.3f}  at_1h={A(first, first.p):.3f}  outcomes={lk}")
