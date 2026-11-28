"""Shared harness: does message content help where there are customer messages?

Population: every labeled snapshot (won / lost / stale open, any creation date; closed recent deals included
for more text). S = snapshots with at least one customer message (SMS in, email in, call transcript) before t.
CV: 5-fold grouped by customer, 3 seeds. The base LightGBM (lean strict features) trains on all rows of the
train fold; extra features (NaN outside S) are added as columns. Reported: deal-weighted AUC on the
validation rows in S, overall and for Indianapolis.

Use: from text_cv import load_all, evaluate, customer_messages
  evaluate(extra: DataFrame indexed by sid, name)            # plain feature columns
  evaluate(None, name, stacker=fn(train_df, val_df) -> (train_vec, val_vec))   # out-of-fold text score
"""
import sys, pickle, json
sys.path.insert(0, "src")
import numpy as np, pandas as pd, lightgbm as lgb
import common
from common import prepare, PARAMS, load
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

CFG = pickle.load(open("data/out/final_lean.pkl", "rb"))
_cache = {}


def load_all():
    if "SN" in _cache:
        return _cache["SN"], _cache["S"]
    common.EVAL_MAX_CREATED = pd.Timestamp("2026-11-22")
    SN = load(strict=True)
    SN = SN[SN.label_kind.isin(["won", "lost", "stale_open"])].reset_index(drop=True)
    m = SN[["sid", "deal_id", "t"]].merge(customer_messages()[["deal_id", "ts"]], on="deal_id")
    S = set(m[m.ts < m.t].sid)
    _cache["SN"], _cache["S"] = SN, S
    return SN, S


def customer_messages(kinds=("sms_in", "email_in", "transcript")):
    """Text units of the given kinds, deduplicated (SMS appear in two provider tables)."""
    T = pd.read_parquet("data/out/texts.parquet")
    T = T[T.kind.isin(kinds)].copy()
    T["norm"] = T.text.fillna("").str.strip().str.lower()
    T["minute"] = T.ts.dt.floor("min")
    T = T.sort_values("ts").drop_duplicates(["deal_id", "norm", "minute"]).drop(columns="minute")
    return T


def evaluate(extra, name, stacker=None, seeds=(0, 1, 2), params=None, rounds=None, base_feats=None):
    SN, S = load_all()
    feats = list(base_feats or CFG["feats"])
    X = SN
    if extra is not None:
        extra = extra.reindex(SN.sid)
        extra.index = SN.index
        X = pd.concat([SN, extra.add_prefix("x_")], axis=1)
        feats += ["x_" + c for c in extra.columns]
    inS = X.sid.isin(S).values
    indy = (X.company_id.astype(int) == 1).values
    p = dict(CFG["params"], **(params or {}))
    res = []
    for seed in seeds:
        cust = X.customer_id.unique()
        perm = pd.Series(np.random.default_rng(seed).permutation(len(cust)), index=cust)
        oof = np.full(len(X), np.nan)
        for a, b in GroupKFold(5).split(X, groups=X.customer_id.map(perm)):
            tr, va = X.iloc[a], X.iloc[b]
            Xtr, Xva = prepare(tr, va, feats)
            if stacker is not None:
                s_tr, s_va = stacker(tr, va)
                Xtr["txt"], Xva["txt"] = s_tr, s_va
            m = lgb.train(p, lgb.Dataset(Xtr, tr.label, weight=tr.w), rounds or CFG["rounds"])
            oof[b] = m.predict(Xva)
        auc = lambda mask: roc_auc_score(X.label[mask], oof[mask], sample_weight=X.w[mask])
        res.append((auc(inS), auc(inS & indy)))
    r = np.array(res)
    out = dict(name=name, S_auc=round(r[:, 0].mean(), 4), S_sd=round(r[:, 0].std(), 4), indy_S_auc=round(r[:, 1].mean(), 4),
               S_deals=int(X[inS].deal_id.nunique()), indy_deals=int(X[inS & indy].deal_id.nunique()))
    print(json.dumps(out), flush=True)
    return out


if __name__ == "__main__":
    evaluate(None, "base (no text)")
