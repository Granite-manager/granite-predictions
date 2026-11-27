"""Sequence model (GRU over the deal's event stream before t + static strict features) on MPS,
time holdout like common.holdout_score, and stacking its OOF logit into LightGBM."""
import sys, time, json; sys.path.insert(0, "src")
import torch, torch.nn as nn
from common import *

torch.set_num_threads(2)
torch.manual_seed(0); np.random.seed(0)
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
L = 128

SN = load(strict=True)
G = groups(SN)
feats = sorted(set(sum([v for k, v in G.items() if k != "text"], [])))
cats = [f for f in feats if f in HIGH_CARD or f in LOW_CARD]
nums = [f for f in feats if f not in cats]

# ---------------------------------------------------------------- event sequences
EV = pd.read_parquet("data/out/events.parquet")
EV = EV[EV.deal_id.isin(SN.deal_id)].copy()
kinds = sorted(EV.kind.unique())
kid = {k: i + 2 for i, k in enumerate(kinds)}  # 0 pad, 1 BOS
EV["tok"] = EV.kind.map(kid)
EV["dir"] = np.where(EV.kind.str.endswith("_in"), 1.0, np.where(EV.kind.str.endswith("_out"), -1.0, 0.0))
EV["ltalk"] = np.log1p(EV.talk.fillna(0).clip(lower=0))
EV["llen"] = np.log1p(EV["len"].fillna(0).clip(lower=0))
EV["auto_"] = EV.auto.fillna(0).astype(float)
EV["ans"] = EV.answered.fillna(0).astype(float)
EV = EV.sort_values(["deal_id", "ts"])
EV["gap"] = EV.groupby("deal_id").ts.diff().dt.total_seconds().fillna(0) / 3600
EVF = ["dir", "ltalk", "llen", "auto_", "ans"]
NE = 3 + len(EVF)

t0 = time.time()
M = SN[["sid", "deal_id", "t", "created_at"]].reset_index().rename(columns={"index": "row"}).merge(
    EV[["deal_id", "ts", "tok", "gap"] + EVF], on="deal_id")
M = M[M.ts < M.t].sort_values(["row", "ts"])
M = M.groupby("row").tail(L - 1)
M["pos"] = M.groupby("row").cumcount() + 1
slog = lambda x: np.sign(x) * np.log1p(np.abs(x))
M["f_gap"] = np.log1p(M.gap.clip(lower=0))
M["f_since"] = slog((M.ts - M.created_at).dt.total_seconds() / 3600)
M["f_ago"] = np.log1p((M.t - M.ts).dt.total_seconds() / 3600)
N = len(SN)
TOK = np.zeros((N, L), np.int64); TOK[:, 0] = 1
EVX = np.zeros((N, L, NE), np.float32)
TOK[M.row.values, M.pos.values] = M.tok.values
EVX[M.row.values, M.pos.values] = M[["f_gap", "f_since", "f_ago"] + EVF].values.astype(np.float32)
LEN = np.bincount(M.row.values, minlength=N) + 1
print(f"sequences built {time.time()-t0:.0f}s, mean len {LEN.mean():.1f}, max {LEN.max()}", flush=True)


# ---------------------------------------------------------------- static encoding (fit on train rows only)
def encode_static(tr_idx):
    Xn = SN[nums].astype(float)
    mu, sd = Xn.iloc[tr_idx].mean(), Xn.iloc[tr_idx].std().replace(0, 1).fillna(1)
    Z = ((Xn - mu) / sd).clip(-5, 5)
    miss = Xn.isna().astype(np.float32).values
    NUM = np.hstack([Z.fillna(0).values.astype(np.float32), miss])
    CAT, sizes = [], []
    for c in cats:
        s = SN[c].astype(str)
        vc = s.iloc[tr_idx].value_counts()
        voc = {v: i + 1 for i, v in enumerate(vc[vc >= 5].index)}
        CAT.append(s.map(voc).fillna(0).astype(np.int64).values)
        sizes.append(len(voc) + 1)
    return NUM, np.stack(CAT, 1), sizes


class Net(nn.Module):
    def __init__(self, n_num, cat_sizes, h=64):
        super().__init__()
        self.tok = nn.Embedding(len(kid) + 2, 32, padding_idx=0)
        self.evp = nn.Linear(NE, 32)
        self.gru = nn.GRU(64, h, batch_first=True)
        self.cats = nn.ModuleList([nn.Embedding(s, min(8, 1 + s // 2)) for s in cat_sizes])
        cdim = sum(min(8, 1 + s // 2) for s in cat_sizes)
        self.st = nn.Sequential(nn.Linear(n_num + cdim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, h), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * h, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, tok, evx, ln, num, cat):
        x = torch.cat([self.tok(tok), self.evp(evx)], -1)
        o, _ = self.gru(x)
        last = o[torch.arange(o.shape[0], device=o.device), ln - 1]
        c = torch.cat([e(cat[:, i]) for i, e in enumerate(self.cats)], -1)
        s = self.st(torch.cat([num, c], -1))
        return self.head(torch.cat([last, s], -1)).squeeze(-1)


def T(a, idx):
    return torch.from_numpy(a[idx]).to(DEV)


def train_predict(tr_idx, pred_idx, epochs=12, bs=256, seed=0):
    torch.manual_seed(seed)
    NUM, CAT, sizes = encode_static(tr_idx)
    y = SN.label.values.astype(np.float32); w = SN.w.values.astype(np.float32)
    # inner early stopping on 10% of train deals
    deals = SN.deal_id.values[tr_idx]
    ud = np.unique(deals); rng = np.random.default_rng(seed)
    hold = set(rng.choice(ud, max(1, len(ud) // 10), replace=False))
    inner_va = tr_idx[np.isin(deals, list(hold))]; inner_tr = tr_idx[~np.isin(deals, list(hold))]
    net = Net(NUM.shape[1], sizes).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state = 1e9, None
    def predict(idx):
        net.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(idx), 1024):
                b = idx[i:i + 1024]
                out.append(net(T(TOK, b), T(EVX, b), T(LEN, b), T(NUM, b), T(CAT, b)).float().cpu().numpy())
        return np.concatenate(out)
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(inner_tr)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            logit = net(T(TOK, b), T(EVX, b), T(LEN, b), T(NUM, b), T(CAT, b))
            wb = T(w, b); wb = wb / wb.mean()
            loss = (nn.functional.binary_cross_entropy_with_logits(logit, T(y, b), reduction="none") * wb).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        lv = predict(inner_va)
        pv = 1 / (1 + np.exp(-lv))
        ll = log_loss(y[inner_va], np.clip(pv, 1e-4, 1 - 1e-4), sample_weight=w[inner_va])
        if ll < best:
            best = ll; best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return predict(pred_idx)


tr_idx = np.where(SN.created_at < HOLDOUT_FROM)[0]
va_idx = np.where(SN.created_at >= HOLDOUT_FROM)[0]
n_params = sum(p.numel() for p in Net(len(nums) * 2, [len(SN[c].astype(str).unique()) for c in cats]).parameters())
t0 = time.time()
lv = train_predict(tr_idx, va_idx)
pv = 1 / (1 + np.exp(-lv))
va = SN.iloc[va_idx]
res = dict(seq_ho_auc=roc_auc_score(va.label, pv, sample_weight=va.w),
           seq_ho_ll=log_loss(va.label, np.clip(pv, 1e-4, 1 - 1e-4), sample_weight=va.w), seq_secs=round(time.time() - t0), params_approx=int(n_params))
print(json.dumps(res), flush=True)

# OOF logits on train rows (customer-grouped 5-fold), holdout rows from the full-train model
oof = np.zeros(len(tr_idx))
trdf = SN.iloc[tr_idx]
cust = trdf.customer_id.unique()
perm = pd.Series(np.random.default_rng(1).permutation(len(cust)), index=cust)
t0 = time.time()
for k, (a, b) in enumerate(GroupKFold(5).split(trdf, groups=trdf.customer_id.map(perm))):
    oof[b] = train_predict(tr_idx[a], tr_idx[b], seed=k + 1)
    print("oof fold", k, flush=True)
SN["seq_logit"] = np.nan
SN.loc[SN.index[tr_idx], "seq_logit"] = oof
SN.loc[SN.index[va_idx], "seq_logit"] = lv
res["oof_secs"] = round(time.time() - t0)
res["seq_oof_train_auc"] = roc_auc_score(trdf.label, oof, sample_weight=trdf.w)
SN[["sid", "seq_logit"]].to_parquet("data/out/seq_logit.parquet")
json.dump(res, open("data/out/exp_seq.json", "w"))
print(json.dumps(res))
# LightGBM stacking runs in a separate process (torch + LightGBM OpenMP in one process segfaults): exp_seq_stack.py
