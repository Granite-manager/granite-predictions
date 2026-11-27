"""Company-aware features (one shared model, but each company has its own norms):
time-aware company / company+source / company+referral win rates from deals closed before t,
rep win rate relative to company, and engagement relative to the company's typical value."""
import numpy as np, pandas as pd

SN = pd.read_parquet("data/out/snaps.parquet")
D = pd.read_parquet("data/out/deals_raw.parquet")
S = pd.read_parquet("data/out/static.parquet")[["deal_id", "lead_source", "lead_referral_source"]]
D = D.merge(S, left_on="id", right_on="deal_id")
cl = D[D.label_kind.isin(["won", "lost"]) & ~D.instant].copy()
cl["w"] = (cl.label == 1).astype(int)
F = SN[["sid", "t", "company_id", "lead_source", "lead_referral_source"]].copy()
F["lead_source"] = F.lead_source.astype(str)
F["lead_referral_source"] = F.lead_referral_source.astype(str)
cl["lead_source"] = cl.lead_source.astype(str)
cl["lead_referral_source"] = cl.lead_referral_source.astype(str)
glob = cl.w.mean()


def prior_rate(keys, name, m=30, window_days=None):
    c = cl.sort_values("close_t").copy()
    c["cw"] = c.groupby(keys).w.cumsum()
    c["cn"] = c.groupby(keys).cumcount() + 1
    out = pd.merge_asof(F.sort_values("t"), c[["close_t", "cw", "cn"] + keys], left_on="t", right_on="close_t", by=keys,
                        allow_exact_matches=False)
    if window_days:
        lag = F[["sid", "t"] + keys].copy()
        lag["t"] = lag.t - pd.Timedelta(days=window_days)
        o2 = pd.merge_asof(lag.sort_values("t"), c[["close_t", "cw", "cn"] + keys], left_on="t", right_on="close_t", by=keys,
                           allow_exact_matches=False).set_index("sid")
        out = out.set_index("sid")
        out["cw"] = out.cw.fillna(0) - o2.cw.reindex(out.index).fillna(0)
        out["cn"] = out.cn.fillna(0) - o2.cn.reindex(out.index).fillna(0)
        out = out.reset_index()
    out[name] = (out.cw.fillna(0) + m * glob) / (out.cn.fillna(0) + m)
    out[name + "_n"] = out.cn.fillna(0)
    return out.set_index("sid")[[name, name + "_n"]]


R = pd.concat([prior_rate(["company_id"], "co_wr"), prior_rate(["company_id"], "co_wr_180d", window_days=180),
               prior_rate(["company_id", "lead_source"], "co_src_wr"), prior_rate(["company_id", "lead_referral_source"], "co_ref_wr")], axis=1)
R = R.reindex(SN.sid)
R["co_src_wr_rel"] = R.co_src_wr - R.co_wr
R["co_ref_wr_rel"] = R.co_ref_wr - R.co_wr
R["rep_wr_rel"] = SN.set_index("sid").rep_prior_wr.reindex(R.index) - R.co_wr
# engagement relative to the company's typical value at the same snapshot age
X = SN.set_index("sid")
for col in ["n_out", "n_in", "first_out_h", "rep_reply_lat_med_h", "last_in_ago_d", "active_days", "n_activity_created", "n_note", "company_leads_30d"]:
    med = X.groupby(["company_id", "off_h"])[col].transform("median")
    R[f"rel_{col}"] = (X[col] - med) if col.endswith(("_h", "_d")) else X[col] / (med + 1)
    R[f"pct_{col}"] = X.groupby(["company_id", "off_h"])[col].rank(pct=True)
R.reset_index().rename(columns={"index": "sid"}).to_parquet("data/out/company_feats.parquet")
print(R.shape, R.notna().mean().round(2).to_dict())
