"""End-to-end fine-tuned MiniLM on deal messages, stacked into the text_cv harness.

Three steps in separate processes (torch and LightGBM in one process segfault on OpenMP):
  uv run python src/exp_finetune.py prep 0      # texts per snapshot in S + outer/inner fold ids (no torch)
  uv run python src/exp_finetune.py train 0     # torch only: per-fold logits -> data/out/ft_logits_s0.parquet
  uv run python src/exp_finetune.py eval 0      # LightGBM: text_cv.evaluate with the logits as stacker
"""
import sys, time
import numpy as np, pandas as pd

mode, seed = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
PREP = f"data/out/ft_prep_s{seed}.parquet"
LOGITS = f"data/out/ft_logits_s{seed}.parquet"

if mode == "prep":
    sys.path.insert(0, "src")
    from sklearn.model_selection import GroupKFold
    from text_cv import load_all, customer_messages
    SN, S = load_all()
    X = SN
    cust = X.customer_id.unique()
    perm = pd.Series(np.random.default_rng(seed).permutation(len(cust)), index=cust)
    fold = np.zeros(len(X), int)
    for k, (a, b) in enumerate(GroupKFold(5).split(X, groups=X.customer_id.map(perm))):
        fold[b] = k
    X = X.assign(fold=fold)
    msgs = pd.concat([customer_messages().assign(who="CUSTOMER"),
                      customer_messages(("sms_out", "email_out")).assign(who="REP")])
    msgs["line"] = msgs.who + ": " + msgs.text.fillna("").str.replace(r"\s+", " ", regex=True).str.slice(0, 300)
    rows = X[X.sid.isin(S)][["sid", "deal_id", "customer_id", "t", "label", "w", "fold"]]
    m = rows[["sid", "deal_id", "t"]].merge(msgs[["deal_id", "ts", "line"]], on="deal_id")
    m = m[m.ts < m.t].sort_values(["sid", "ts"]).groupby("sid").tail(8)
    text = m.groupby("sid").line.apply(lambda s: "\n".join(s))
    rows = rows.assign(text=rows.sid.map(text).fillna(""))
    # inner 3 folds by deal inside each outer train part
    deals = rows.deal_id.unique()
    rows["inner"] = rows.deal_id.map(pd.Series(np.random.default_rng(100 + seed).integers(0, 3, len(deals)), index=deals))
    rows.drop(columns="t").to_parquet(PREP)
    print("rows in S", len(rows), "deals", rows.deal_id.nunique(), "avg chars", int(rows.text.str.len().mean()))

elif mode == "train":
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.manual_seed(seed)
    R = pd.read_parquet(PREP)
    name = "sentence-transformers/all-MiniLM-L6-v2"
    tok = AutoTokenizer.from_pretrained(name)
    dev = "mps"

    def enc(texts):
        return tok(list(texts), truncation=True, max_length=256, padding=True, return_tensors="pt")

    def fit_predict(train, pred, epochs=3, lr=3e-5, bs=32):
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        y = torch.tensor(train.label.values, dtype=torch.long)
        w = torch.tensor(train.w.values / train.w.values.mean(), dtype=torch.float32)
        texts = train.text.values
        g = torch.Generator().manual_seed(seed)
        model.train()
        for _ in range(epochs):
            order = torch.randperm(len(train), generator=g)
            for i in range(0, len(order), bs):
                idx = order[i:i + bs]
                e = enc(texts[idx.numpy()]).to(dev)
                logits = model(**e).logits
                loss = (torch.nn.functional.cross_entropy(logits, y[idx].to(dev), reduction="none") * w[idx].to(dev)).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); out = []
        with torch.no_grad():
            pt = pred.text.values
            for i in range(0, len(pt), 64):
                lg = model(**enc(pt[i:i + 64]).to(dev)).logits.float().cpu().numpy()
                out.append(lg[:, 1] - lg[:, 0])
        del model; torch.mps.empty_cache()
        return np.concatenate(out) if out else np.zeros(0)

    t0 = time.time(); res = []
    for k in range(5):
        tr, va = R[R.fold != k], R[R.fold == k]
        res.append(pd.DataFrame({"sid": va.sid, "outer": k, "role": "val", "logit": fit_predict(tr, va)}))
        for j in range(3):
            a, b = tr[tr.inner != j], tr[tr.inner == j]
            res.append(pd.DataFrame({"sid": b.sid, "outer": k, "role": "train", "logit": fit_predict(a, b)}))
        print(f"outer fold {k} done {time.time() - t0:.0f}s", flush=True)
    pd.concat(res).to_parquet(LOGITS)
    print("train seconds", round(time.time() - t0))

elif mode == "eval":
    sys.path.insert(0, "src")
    from text_cv import evaluate
    L = pd.read_parquet(LOGITS)
    tr_map = {k: g.set_index("sid").logit for k, g in L[L.role == "train"].groupby("outer")}
    va_map = {k: g.set_index("sid").logit for k, g in L[L.role == "val"].groupby("outer")}
    R = pd.read_parquet(PREP)
    fold_of = R.set_index("sid").fold

    def stacker(tr, va):
        # which outer fold is this: the one whose validation sids are in va
        k = int(fold_of.reindex(va.sid).dropna().mode().iloc[0])
        return tr.sid.map(tr_map[k]).values, va.sid.map(va_map[k]).values

    evaluate(None, f"fine-tuned MiniLM (seed {seed})", stacker=stacker, seeds=(seed,))
