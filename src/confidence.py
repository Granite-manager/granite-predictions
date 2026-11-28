"""When can we trust a prediction? Holdout accuracy by what was known at the snapshot (evidence),
per company group, and whether disagreement between bootstrap models flags weak predictions."""
import sys, pickle; sys.path.insert(0, "src")
import lightgbm as lgb
from common import *
cfg = pickle.load(open("data/out/final_lean.pkl", "rb"))
orig = pd.read_parquet
pd.read_parquet = lambda p, *a, **k: orig("data/out/snaps_after.parquet" if p == "data/out/snaps.parquet" else p, *a, **k)
SN = load(strict=True); G = groups(SN)
gmq = SN.company_id.astype(int).isin([4, 12, 13])
for c in G["stage"]:
    SN.loc[gmq, c] = np.nan if SN[c].dtype.kind in "fi" else None
F = cfg["feats"]
tr, va = SN[SN.created_at < HOLDOUT_FROM], SN[SN.created_at >= HOLDOUT_FROM].copy()
p, _, Xva = fit_predict(tr, va, F, cfg["rounds"], cfg["params"])
va["p"] = p
# bootstrap disagreement: 8 models on resampled deals
Xtr, Xva2 = prepare(tr, va, F)
deals = tr.deal_id.unique(); preds = []
for s in range(8):
    pick = pd.Series(np.random.default_rng(s).choice(deals, len(deals))).value_counts()
    w = tr.deal_id.map(pick).fillna(0).values * tr.w.values
    m = lgb.train(dict(cfg["params"], seed=s, num_threads=12), lgb.Dataset(Xtr, tr.label, weight=w), cfg["rounds"])
    preds.append(m.predict(Xva2))
va["sd"] = np.std(preds, axis=0)
va["grp"] = np.select([va.company_id.astype(int).isin([4, 12, 13]), va.company_id.astype(int) == 1, va.company_id.astype(int) == 3],
                      ["GMQ", "Indy", "Columbus"], "Lex+Empire")
ev = pd.DataFrame({
    "contacted": va.n_out > 0, "replied": va.n_in > 0, "call": (va.n_call_in + va.n_call_out) > 0,
    "talked": va.call_answered > 0, "email": (va.n_email_in + va.n_email_out) > 0,
    "quote": va.n_quote > 0, "quote_sent": va.n_quote_sent > 0, "quote_viewed": (va.n_quote_view + va.n_quote_first_view) > 0,
    "stage_moved": va.stage_n_changes > 0, "notes": va.n_note > 0})
va["evidence"] = ev[["replied", "talked", "email", "quote", "stage_moved"]].sum(axis=1).clip(upper=3)


def stats(x):
    if len(x) < 40 or x.label.nunique() < 2 or x.label.sum() < 5:
        return pd.Series(dict(snaps=len(x), deals=x.deal_id.nunique(), auc=np.nan, win=x.label.mean(), brier_skill=np.nan))
    br = brier_score_loss(x.label, x.p, sample_weight=x.w)
    base = np.average(x.label, weights=x.w)
    return pd.Series(dict(snaps=len(x), deals=x.deal_id.nunique(), auc=roc_auc_score(x.label, x.p, sample_weight=x.w),
                          win=base, brier_skill=1 - br / (base * (1 - base))))


pd.set_option("display.width", 200)
for g in ["Indy", "Columbus", "GMQ", "Lex+Empire"]:
    x = va[va.grp == g]; e = ev[va.grp == g]
    rows = {"all": stats(x)}
    for k in ev.columns:
        rows[f"{k}: yes"] = stats(x[e[k]]); rows[f"{k}: no"] = stats(x[~e[k]])
    for k in range(4):
        rows[f"evidence types = {k}{'+' if k == 3 else ''}"] = stats(x[x.evidence == k])
    for lo, hi in [(0, 1.5), (1.5, 8), (8, 31), (31, 1e9)]:
        rows[f"age {lo:g}-{hi:g} d"] = stats(x[(x.age_days >= lo) & (x.age_days < hi)])
    q = pd.qcut(x.sd, 3, labels=["models agree", "mid", "models disagree"])
    for k in q.cat.categories:
        rows[f"bootstrap: {k}"] = stats(x[q == k])
    print(f"\n######## {g}"); print(pd.DataFrame(rows).T.round(3).to_string())
va[["sid", "grp", "label", "p", "sd", "evidence", "age_days", "w"]].join(ev).to_parquet("data/out/confidence_holdout.parquet")
