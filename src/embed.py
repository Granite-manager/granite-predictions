"""Embed every text unit with a sentence-transformers model on MPS. Usage: embed.py <hf-model> <tag> [max_len]"""
import sys, time, numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer
name, tag = sys.argv[1], sys.argv[2]
max_len = int(sys.argv[3]) if len(sys.argv) > 3 else 256
T = pd.read_parquet("data/out/texts.parquet")
T["text"] = T.text.fillna("").str.slice(0, 3000)
m = SentenceTransformer(name, device="mps", trust_remote_code=True)
m.max_seq_length = max_len
order = T.text.str.len().argsort().values  # length-sorted batches are much faster
t0 = time.time()
E = m.encode(T.text.values[order].tolist(), batch_size=64, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
out = np.zeros_like(E); out[order] = E
np.save(f"data/out/emb_{tag}.npy", out.astype(np.float16))
T[["uid"]].to_parquet(f"data/out/emb_{tag}_uids.parquet")
print(tag, out.shape, f"{time.time()-t0:.0f}s", f"{(time.time()-t0)/len(T)*1000:.2f} ms/text")
