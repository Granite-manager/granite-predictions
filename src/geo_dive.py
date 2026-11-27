"""Value of high-cardinality location / identity fields at deal level, among deals where the field is known.
AUC of logistic(company dummies + out-of-fold target encoding of the field) vs company only,
and vs a null where the field is shuffled within company (20 shuffles)."""
import sys, warnings, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from common import EVAL_MAX_CREATED

D = pd.read_parquet("data/out/deals_raw.parquet")
S = pd.read_parquet("data/out/static.parquet").drop(columns="company_id")
X = D.merge(S, left_on="id", right_on="deal_id")
X = X[~X.instant & X.label.isin([0, 1]) & (X.created_at <= EVAL_MAX_CREATED)].reset_index(drop=True)
X["user_id"] = X.user_id.astype("Int64").astype(str)
X["geo_dist_b"] = pd.cut(X.geo_dist_shop_mi, [0, 5, 10, 20, 40, 1e9]).astype(str)
X["geo_income_q"] = pd.qcut(X.geo_income_rel, 5, duplicates="drop").astype(str)
X["geo_homeval_q"] = pd.qcut(X.geo_home_value_rel, 5, duplicates="drop").astype(str)
X["geo_density_q"] = pd.qcut(X.geo_pop_density, 5, duplicates="drop").astype(str)


def te_oof(col, y, folds, m=20):
    out = np.zeros(len(col))
    prior = y.mean()
    for tr, va in folds:
        s = pd.DataFrame({"k": col[tr], "y": y[tr]}).groupby("k").y.agg(["sum", "count"])
        enc = (s["sum"] + m * prior) / (s["count"] + m)
        out[va] = pd.Series(col[va]).map(enc).fillna(prior).values
    return out


def auc_with(F, col, reps=5, shuffle=False, seed=0):
    rng = np.random.default_rng(seed)
    y = F.label.values
    comp = pd.get_dummies(F.company_id.astype(str)).values.astype(float)
    aucs, base = [], []
    for r in range(reps):
        c = F[col].values.copy()
        if shuffle:
            c = F.groupby("company_id")[col].transform(lambda s: rng.permutation(s.values)).values
        folds = list(StratifiedKFold(5, shuffle=True, random_state=r).split(F, y))
        te = te_oof(c, y, folds)
        lg = np.log(te / (1 - te))
        p = np.zeros(len(F)); pb = np.zeros(len(F))
        for tr, va in folds:
            Z = np.column_stack([comp, lg])
            p[va] = LogisticRegression(max_iter=500).fit(Z[tr], y[tr]).predict_proba(Z[va])[:, 1]
            pb[va] = LogisticRegression(max_iter=500).fit(comp[tr], y[tr]).predict_proba(comp[va])[:, 1]
        aucs.append(roc_auc_score(y, p)); base.append(roc_auc_score(y, pb))
    return np.mean(aucs), np.mean(base)


rows = []
for col, label in [("geo_zip", "zip (5-digit)"), ("geo_zip3", "zip3 (first 3 digits)"), ("geo_city", "city"), ("geo_county", "county"),
                   ("geo_dist_b", "distance band"), ("geo_income_q", "zip income quintile"), ("geo_homeval_q", "zip home value quintile"),
                   ("geo_density_q", "zip density quintile"), ("ph_area", "phone area code"), ("ph_exchange", "phone area+exchange (6 digits)"),
                   ("em_domain", "email domain"), ("user_id", "salesperson"), ("lead_campaign", "ad campaign"), ("lead_adset", "ad set"), ("lead_ad", "ad")]:
    F = X[X[col].notna() & ~X[col].isin(["nan", "<NA>", "None"])].reset_index(drop=True)
    if len(F) < 200 or F.label.nunique() < 2:
        continue
    a, b = auc_with(F, col)
    nulls = [auc_with(F, col, reps=1, shuffle=True, seed=s)[0] for s in range(20)]
    rows.append(dict(field=label, deals_known=len(F), levels=F[col].nunique(), median_deals_per_level=float(F[col].value_counts().median()),
                     auc_company_only=b, auc_with_field=a, gain=a - b, null_gain_mean=np.mean(nulls) - b, null_gain_p95=np.percentile(nulls, 95) - b,
                     real=bool(a > np.percentile(nulls, 95))))
    print(rows[-1], flush=True)
R = pd.DataFrame(rows).sort_values("gain", ascending=False)
R.to_csv("data/out/geo_dive.csv", index=False)
print(R.round(4).to_string(index=False))
