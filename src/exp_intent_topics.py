"""Interpretable text features on customer messages: intent-prototype similarity and k-means topics."""
import sys, json
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from text_cv import load_all, evaluate, customer_messages

INTENTS = {
    "ready_buy": ["I'm ready to move forward", "When can you come measure?", "Let's schedule the template", "We want to go ahead with the order", "Book the install"],
    "deposit_payment": ["How much is the deposit?", "Can I pay with a card?", "I sent the payment", "Do you take checks?"],
    "price_question": ["How much would it cost?", "Can you send me a price?", "Is there any discount?", "What's the price per square foot?"],
    "too_expensive": ["That's too expensive", "It's out of our budget", "The price is too high for us", "We can't afford that right now"],
    "competitor": ["We got a cheaper quote elsewhere", "We went with another company", "Another company offered a better price", "We decided to go with someone else"],
    "delay": ["Not right now, maybe later", "We're putting the project on hold", "I'll get back to you next month", "We need more time to decide"],
    "color_material": ["Do you have white quartz?", "What colors do you have in granite?", "Can I see the slabs?", "I like the Calacatta"],
    "positive": ["Sounds great, thank you!", "We are so excited", "Perfect, that works", "Awesome, love it"],
    "complaint": ["I'm very disappointed", "This is unacceptable", "Nobody called me back", "There is a problem with the countertop"],
    "not_interested": ["Please stop texting me", "Not interested", "Remove me from your list", "Wrong number"],
    "visit_showroom": ["We'll stop by the showroom", "What are your hours?", "Can we come in Saturday?", "Where are you located?"],
    "send_info": ["Here are the measurements", "I'll send you pictures of the kitchen", "Here is the drawing", "Attached are photos"],
}

T = pd.read_parquet("data/out/texts.parquet")
uids = pd.read_parquet("data/out/emb_bgebase_uids.parquet").uid.values
assert (uids == T.uid.values).all()
E = np.load("data/out/emb_bgebase.npy").astype(np.float32)
C = customer_messages()  # index = row in texts.parquet
EC = E[C.index.values]
EC /= np.linalg.norm(EC, axis=1, keepdims=True) + 1e-9

from sentence_transformers import SentenceTransformer
m = SentenceTransformer("BAAI/bge-base-en-v1.5", device="mps")
sims = {}
for k, phrases in INTENTS.items():
    P = m.encode(phrases, normalize_embeddings=True)
    sims[k] = (EC @ P.T).max(1)
M = pd.DataFrame(sims, index=C.index)
M["deal_id"], M["ts"] = C.deal_id.values, C.ts.values

for K in (12, 24):
    km = KMeans(K, n_init=4, random_state=0).fit(EC)
    M[f"topic{K}"] = km.labels_
    if K == 12:
        print("\n## topics k=12")
        for t in range(12):
            ex = C.text[km.labels_ == t].drop_duplicates().head(3).str.slice(0, 90).str.replace("\n", " ").tolist()
            print(t, (km.labels_ == t).sum(), ex)

SN, S = load_all()
X = SN[SN.sid.isin(S)][["sid", "deal_id", "t", "label"]].merge(M, on="deal_id")
X = X[X.ts < X.t].sort_values(["sid", "ts"])
g = X.groupby("sid")
ik = list(INTENTS)
intent = pd.concat([g[ik].max().add_suffix("_max"), g[ik].mean().add_suffix("_mean"), g[ik].last().add_suffix("_last")], axis=1)
intent["n_msgs"] = g.size()


def topics(K):
    share = pd.crosstab(X.sid, X[f"topic{K}"], normalize="index").add_prefix(f"t{K}_share_")
    last = pd.get_dummies(g[f"topic{K}"].last()).astype(float).add_prefix(f"t{K}_last_")
    return pd.concat([share, last], axis=1)


res = []
res.append(evaluate(intent, "intent prototypes"))
res.append(evaluate(topics(12), "topics k=12"))
res.append(evaluate(topics(24), "topics k=24"))
res.append(evaluate(pd.concat([intent, topics(12)], axis=1), "intent + topics k=12"))
json.dump(res, open("data/out/exp_intent_topics.json", "w"), indent=1)

# which intents carry signal: deal-level mean by outcome (last snapshot of each deal in S)
lastsnap = X.sort_values("t").groupby("deal_id").sid.last()
d = intent.loc[intent.index.isin(lastsnap.values)].join(SN.set_index("sid").label)
cmp = pd.DataFrame({"won": d[d.label == 1][[f"{k}_max" for k in ik]].mean(), "lost": d[d.label == 0][[f"{k}_max" for k in ik]].mean()})
cmp["diff"] = cmp.won - cmp.lost
print("\n## intent max similarity by outcome (deal level)")
print(cmp.sort_values("diff").round(3).to_string())
cmp.to_csv("data/out/exp_intent_by_outcome.csv")
