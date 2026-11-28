"""Sentiment features and embedding variants on snapshots with customer messages (harness: text_cv.py)."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from text_cv import load_all, evaluate, customer_messages

SN, S = load_all()
snaps = SN[["sid", "deal_id", "t"]]
cust = customer_messages()
rep = customer_messages(("sms_out", "email_out"))
sent = pd.read_parquet("data/out/sentiment.parquet").set_index("uid")[["s_negative", "s_neutral", "s_positive"]]


def per_snapshot(msgs):
    m = snaps.merge(msgs[["deal_id", "ts", "uid"]], on="deal_id")
    return m[m.ts < m.t].sort_values(["sid", "ts"])


# ---------------------------------------------------------------- sentiment
def sentiment_features(msgs, prefix):
    m = per_snapshot(msgs).join(sent, on="uid")
    m["net"] = m.s_positive - m.s_negative
    g = m.groupby("sid")
    f = pd.DataFrame({
        f"{prefix}_n": g.size(), f"{prefix}_pos_mean": g.s_positive.mean(), f"{prefix}_neg_mean": g.s_negative.mean(),
        f"{prefix}_neg_max": g.s_negative.max(), f"{prefix}_net_mean": g.net.mean(),
        f"{prefix}_share_neg": g.s_negative.apply(lambda s: (s > 0.5).mean()),
        f"{prefix}_share_pos": g.s_positive.apply(lambda s: (s > 0.5).mean()),
        f"{prefix}_last_net": g.net.last(), f"{prefix}_last_neg": g.s_negative.last(), f"{prefix}_last_pos": g.s_positive.last(),
        f"{prefix}_last3_net": g.net.apply(lambda s: s.tail(3).mean()),
    })
    f[f"{prefix}_trend"] = f[f"{prefix}_last3_net"] - f[f"{prefix}_net_mean"]
    return f


import os
ONLY_EMB = os.environ.get("ONLY_EMB")
fc = sentiment_features(cust, "cs")
fr = sentiment_features(rep, "rs")
X = SN[SN.sid.isin(S)].set_index("sid").join(fc)
print("customer sentiment vs outcome (snapshots in S):")
print(X.groupby("label")[["cs_pos_mean", "cs_neg_mean", "cs_last_net", "cs_share_neg"]].mean().round(3).to_string())
if not ONLY_EMB:
    evaluate(fc, "customer sentiment")
    evaluate(fc.join(fr, how="outer"), "customer + rep sentiment")
    evaluate(fc[["cs_n"]], "customer message count only (control)")


# ---------------------------------------------------------------- embedding variants (stacked logistic score)
T = pd.read_parquet("data/out/texts.parquet").reset_index(drop=True)
row_of = pd.Series(np.arange(len(T)), index=pd.MultiIndex.from_arrays([T.deal_id, T.uid]))


def vectors(tag, msgs, mode):
    E = np.load(f"data/out/emb_{tag}.npy").astype(np.float32)
    assert (pd.read_parquet(f"data/out/emb_{tag}_uids.parquet").uid.values == T.uid.values).all()
    m = per_snapshot(msgs)
    m["row"] = row_of.reindex(pd.MultiIndex.from_arrays([m.deal_id, m.uid])).values
    if mode == "last":
        last = m.groupby("sid").row.last()
        V = pd.DataFrame(E[last.values], index=last.index)
    else:
        m = m.groupby("sid").tail(20)
        V = pd.DataFrame(E[m.row.values], index=m.sid.values).groupby(level=0).mean()
    return V


def make_stacker(V, C=0.5):
    def fit(df):
        rows = df[df.sid.isin(V.index)]
        return rows, V.loc[rows.sid].values
    def stack(tr, va):
        trS, Xtr = fit(tr); vaS, Xva = fit(va)
        out_tr = pd.Series(np.nan, index=tr.index); out_va = pd.Series(np.nan, index=va.index)
        mdl = lambda a, y, w: LogisticRegression(C=C, max_iter=500).fit(a, y, sample_weight=w * len(w) / w.sum())
        if len(vaS):
            out_va[vaS.index] = mdl(Xtr, trS.label.values, trS.w.values).predict_proba(Xva)[:, 1]
        f = trS.deal_id.map(pd.Series(np.random.default_rng(0).integers(0, 5, trS.deal_id.nunique()), index=trS.deal_id.unique())).values
        for k in range(5):
            out_tr[trS.index[f == k]] = mdl(Xtr[f != k], trS.label.values[f != k], trS.w.values[f != k]).predict_proba(Xtr[f == k])[:, 1]
        return out_tr.values, out_va.values
    return stack


for tag in ["minilm", "bgesmall", "bgebase", "qwen06"]:
    for who, msgs in [("customer", cust), ("rep", rep)]:
        for mode in ["mean", "last"]:
            evaluate(None, f"{tag} {who} {mode}", stacker=make_stacker(vectors(tag, msgs, mode)))
