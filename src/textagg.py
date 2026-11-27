"""Per-snapshot text vectors: mean embedding of the last 20 units per channel known before t.
Usage: textagg.py <tag>  (tag of data/out/emb_<tag>.npy, or 'tfidf')"""
import sys, numpy as np, pandas as pd, scipy.sparse as sp
tag = sys.argv[1]
SN = pd.read_parquet("data/out/snaps.parquet", columns=["sid", "deal_id", "t"])
T = pd.read_parquet("data/out/texts.parquet")
CH = {"sms_in": "cust", "email_in": "cust", "transcript": "cust", "sms_out": "rep", "email_out": "rep", "note": "note", "lead": "lead"}
T["ch"] = T.kind.map(CH)
T["row"] = np.arange(len(T))
X = SN.merge(T[["deal_id", "ts", "ch", "row"]], on="deal_id")
X = X[X.ts < X.t].sort_values(["sid", "ch", "ts"])
X = X.groupby(["sid", "ch"]).tail(20)
chans = ["cust", "rep", "note", "lead"]
if tag == "tfidf":
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(min_df=5, max_features=20000, ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode")
    E = vec.fit_transform(T.text.fillna("").str.slice(0, 3000))
    E = sp.csr_matrix(E, dtype=np.float32)
else:
    E = np.load(f"data/out/emb_{tag}.npy").astype(np.float32)
uids = pd.read_parquet(f"data/out/emb_{'minilm' if tag == 'tfidf' else tag}_uids.parquet").uid.values
assert (uids == T.uid.values).all()
n = len(SN); sid_pos = pd.Series(np.arange(n), index=SN.sid)
blocks, flags = [], []
for ch in chans:
    x = X[X.ch == ch]
    r = sid_pos[x.sid].values
    cnt = np.bincount(r, minlength=n).astype(np.float32)
    A = sp.csr_matrix((np.ones(len(x), np.float32), (r, x.row.values)), shape=(n, E.shape[0]))
    A = sp.diags(1 / np.maximum(cnt, 1)) @ A
    M = A @ E
    blocks.append(M if sp.issparse(M) else sp.csr_matrix(M) if tag == "tfidf" else M)
    flags.append(np.log1p(cnt))
if tag == "tfidf":
    M = sp.hstack(blocks + [sp.csr_matrix(np.array(flags).T)]).tocsr()
    sp.save_npz("data/out/txtagg_tfidf.npz", M)
else:
    M = np.hstack(blocks + [np.array(flags).T]).astype(np.float16)
    np.save(f"data/out/txtagg_{tag}.npy", M)
SN[["sid"]].to_parquet(f"data/out/txtagg_{tag}_sids.parquet")
print(tag, M.shape, "snapshots with any text:", (np.array(flags).sum(0) > 0).mean().round(3))
