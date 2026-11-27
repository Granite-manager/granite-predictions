"""Which words in customer / rep messages (known before t) go with winning: L1-free logistic on TF-IDF blocks."""
import sys, numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, "src")
from common import load
T = pd.read_parquet("data/out/texts.parquet")
vec = TfidfVectorizer(min_df=5, max_features=20000, ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode")
vec.fit(T.text.fillna("").str.slice(0, 3000))
voc = np.array(vec.get_feature_names_out())
M = sp.load_npz("data/out/txtagg_tfidf.npz").tocsr()
sids = pd.read_parquet("data/out/txtagg_tfidf_sids.parquet").sid.values
pos = pd.Series(np.arange(len(sids)), index=sids)
SN = load(strict=True)
X = M[pos[SN.sid].values]
nv = len(voc)
for b, name in enumerate(["customer messages", "rep messages", "notes", "lead form text"]):
    Xb = X[:, b * nv:(b + 1) * nv]
    has = np.asarray(Xb.sum(1)).ravel() > 0
    m = LogisticRegression(C=1.0, max_iter=500).fit(Xb[has], SN.label.values[has], sample_weight=SN.w.values[has])
    co = m.coef_[0]
    df = np.asarray((Xb[has] > 0).sum(0)).ravel()
    ok = df >= 30
    order = np.argsort(co)
    pos_terms = [f"{voc[i]}" for i in order[::-1] if ok[i]][:20]
    neg_terms = [f"{voc[i]}" for i in order if ok[i]][:20]
    print(f"\n## {name} (snapshots with text: {has.sum()})\n  + buy : {', '.join(pos_terms)}\n  - lost: {', '.join(neg_terms)}")
