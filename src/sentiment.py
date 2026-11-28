"""Sentiment of every customer and rep message (cardiffnlp twitter-roberta sentiment, MPS) -> data/out/sentiment.parquet"""
import time, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
T = pd.read_parquet("data/out/texts.parquet")
T = T[T.kind.isin(["sms_in", "email_in", "transcript", "sms_out", "email_out"])].copy()
tok = AutoTokenizer.from_pretrained(name); model = AutoModelForSequenceClassification.from_pretrained(name).to("mps").eval()
texts = T.text.fillna("").str.slice(0, 1500).tolist()
order = np.argsort([len(t) for t in texts]); out = np.zeros((len(texts), 3), np.float32)
t0 = time.time()
with torch.no_grad():
    for i in range(0, len(texts), 128):
        idx = order[i:i + 128]
        enc = tok([texts[j] for j in idx], truncation=True, max_length=256, padding=True, return_tensors="pt").to("mps")
        out[idx] = torch.softmax(model(**enc).logits, -1).float().cpu().numpy()
labels = [model.config.id2label[i].lower() for i in range(3)]
for k, l in enumerate(labels):
    T[f"s_{l}"] = out[:, k]
T[["uid", "deal_id", "ts", "kind"] + [f"s_{l}" for l in labels]].to_parquet("data/out/sentiment.parquet")
print(labels, len(T), f"{time.time() - t0:.0f}s")
cust = T[T.kind.isin(["sms_in", "email_in"])]
print("customer messages: mean", cust[[f"s_{l}" for l in labels]].mean().round(3).to_dict())
for l in ["negative", "positive"]:
    print(f"\nmost {l} customer messages:")
    for t in T.loc[cust.sort_values(f"s_{l}").index[-5:], "text"].str.slice(0, 120):
        print("  -", t.replace("\n", " "))
