import sys; sys.path.insert(0,"src")
from common import *
import shap
SN = load(strict=True); G = groups(SN); full = sorted(set(sum(G.values(), [])))
h, p, va, m, Xva = holdout_score(SN, full)
sv = shap.TreeExplainer(m).shap_values(Xva)
imp = pd.Series(np.abs(sv).mean(0), index=Xva.columns).sort_values(ascending=False)
print(imp.head(40).round(4).to_string())
