"""What raises the odds: doubly-robust (AIPW) effect estimates of rep actions, cross-fitted, clustered by deal.

Each lever defines decision rows (deal state at time t), a treatment T (what the rep did in a window after t)
and the outcome Y = deal eventually won. Confounders = strict point-in-time features at t (no text).
Only companies/periods where calls and texts are logged are used."""
import sys, json, warnings, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from common import load, groups, HIGH_CARD

SN = load(strict=True)
logged = SN.company_id.astype(int).isin([4, 12, 13]) | ((SN.company_id.astype(int) == 1) & (SN.created_at >= "2026-04-26"))
SN = SN[logged].reset_index(drop=True)
EV = pd.read_parquet("data/out/events.parquet")
EV = EV[EV.deal_id.isin(SN.deal_id)]
OUT = EV[EV.kind.isin(["sms_out", "call_out", "email_out"]) & (EV.auto.fillna(0) == 0)]
IN = EV[EV.kind.isin(["sms_in", "call_in", "email_in"])]
G = groups(SN)
conf = sorted(set(sum([v for k, v in G.items() if k != "text"], [])))
conf = [c for c in conf if c not in HIGH_CARD] + ["user_id"]
SN["user_id"] = SN.user_id.astype("category")
print("rows in logged scope", len(SN), "deals", SN.deal_id.nunique(), "win", SN.groupby("deal_id").label.first().mean().round(3))


def first_in_window(ev, rows, start, hours):
    m = rows[["sid", "deal_id", start]].merge(ev[["deal_id", "ts", "kind"]], on="deal_id")
    m = m[(m.ts > m[start]) & (m.ts <= m[start] + pd.Timedelta(hours=hours))]
    f = m.sort_values("ts").groupby("sid").first()
    if f.empty:
        f = pd.DataFrame({"ts": pd.Series(dtype="datetime64[ns]"), "kind": pd.Series(dtype=object)})
    return f


P = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=50, feature_fraction=0.7,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=12)


def aipw(R, T, name, desc, cols=None):
    R = R.assign(T=T.astype(int)).reset_index(drop=True)
    if R["T"].nunique() < 2 or R.label.nunique() < 2:
        return
    X = R[cols or conf]
    y, t, w = R.label.values, R["T"].values, R.w.values
    e = np.zeros(len(R)); m1 = np.zeros(len(R)); m0 = np.zeros(len(R))
    for tr, va in GroupKFold(5).split(R, groups=R.customer_id):
        e[va] = lgb.train(P, lgb.Dataset(X.iloc[tr], t[tr]), 300).predict(X.iloc[va])
        a1, a0 = tr[t[tr] == 1], tr[t[tr] == 0]
        m1[va] = lgb.train(P, lgb.Dataset(X.iloc[a1], y[a1]), 300).predict(X.iloc[va])
        m0[va] = lgb.train(P, lgb.Dataset(X.iloc[a0], y[a0]), 300).predict(X.iloc[va])
    keep = (e > 0.05) & (e < 0.95)
    ec = np.clip(e, 0.05, 0.95)
    psi = m1 - m0 + t * (y - m1) / ec - (1 - t) * (y - m0) / (1 - ec)
    df = pd.DataFrame(dict(deal=R.deal_id, psi=psi, w=w, keep=keep))
    df = df[df.keep]
    ate = np.average(df.psi, weights=df.w)
    per_deal = df.groupby("deal").apply(lambda g: np.average(g.psi - ate, weights=g.w) * g.w.sum())
    se = np.sqrt((per_deal ** 2).sum()) / df.w.sum()
    naive = np.average(y[t == 1], weights=w[t == 1]) - np.average(y[t == 0], weights=w[t == 0])
    base = np.average(y[t == 0], weights=w[t == 0])
    out = dict(lever=name, desc=desc, rows=len(R), deals=R.deal_id.nunique(), treated_share=round(float(np.average(t, weights=w)), 3),
               win_untreated=round(float(base), 3), naive_diff_pp=round(100 * naive, 1), aipw_effect_pp=round(100 * ate, 1),
               ci95_pp=[round(100 * (ate - 1.96 * se), 1), round(100 * (ate + 1.96 * se), 1)], overlap=round(float(keep.mean()), 2))
    print(json.dumps(out), flush=True)
    RES.append(out)


RES = []
# 1. chasing a silent customer: silent >= 2 days (or never replied), we have not reached out for >= 2 days
R = SN[SN.off_h.isin([72, 168, 336, 720]) & (SN.last_in_ago_d.isna() | (SN.last_in_ago_d >= 2)) & (SN.last_out_ago_d.isna() | (SN.last_out_ago_d >= 2))]
f = first_in_window(OUT, R, "t", 72)
f_in = first_in_window(IN, R, "t", 72)
# treatment only if our touch came before any customer message in the window (rep-initiated)
T = R.sid.map(f.ts).notna() & (R.sid.map(f_in.ts).isna() | (R.sid.map(f.ts) < R.sid.map(f_in.ts)))
aipw(R, T, "chase_silent_3d", "customer silent 2+ days: rep reaches out in the next 3 days")
# 2. call vs text as that touch
Rt = R[T.values]
k = Rt.sid.map(f.kind)
aipw(Rt[k.isin(["call_out", "sms_out"]).values], (k[k.isin(["call_out", "sms_out"])] == "call_out").values, "call_not_text", "that touch is a call rather than a text")
# 3. two or more touches in the 3 days vs one
n3 = R[["sid", "deal_id", "t"]].merge(OUT[["deal_id", "ts"]], on="deal_id")
n3 = n3[(n3.ts > n3.t) & (n3.ts <= n3.t + pd.Timedelta(hours=72))].groupby("sid").size()
Rt2 = R[T.values]
aipw(Rt2, (Rt2.sid.map(n3).fillna(0) >= 2).values, "two_plus_touches", "2+ touches in the 3 days instead of 1")
# 4. fast reply: customer wrote in the last 24 h and we have not answered yet -> we answer within 30 min of t
R4 = SN[(SN.last_in_ago_d < 1) & ((SN.last_out_ago_d.isna()) | (SN.last_out_ago_d > SN.last_in_ago_d))]
f4 = first_in_window(OUT, R4, "t", 24)
lat = (R4.sid.map(f4.ts) - R4.t).dt.total_seconds() / 3600
aipw(R4, (lat <= 2).values, "reply_within_2h", "customer wrote, unanswered: rep answers within 2 h (vs later/never in 24 h)")
# 5. quote emailed within 3 days when a quote exists and was not emailed
R5 = SN[(SN.n_quote > 0) & (SN.n_quote_sent == 0)]
QS = EV[EV.kind == "quote_sent"]
aipw(R5, R5.sid.isin(first_in_window(QS, R5, "t", 72).index).values, "email_quote_3d", "quote exists, not emailed: email it within 3 days")
# 6. appointment scheduled within 7 days
R6 = SN[SN.off_h.isin([24, 72, 168, 336]) & (SN.n_appt_created == 0)]
AP = EV[EV.kind == "appt_created"]
aipw(R6, R6.sid.isin(first_in_window(AP, R6, "t", 168).index).values, "book_appointment_7d", "no appointment yet: book one within 7 days")
# 7. quote created within 7 days (stage contacted, no quote yet)
R7 = SN[SN.off_h.isin([24, 72, 168]) & (SN.n_quote == 0)]
QQ = EV[EV.kind == "quote"]
aipw(R7, R7.sid.isin(first_in_window(QQ, R7, "t", 168).index).values, "quote_within_7d", "no quote yet: build a quote within 7 days")
# 8. speed to lead: at creation (1 h snapshot uses state at 1 h, so use static-only rows at off 1): contact within first hour
R8 = SN[SN.off_h == 1]
fo = R8.first_out_h
static = [c for c in conf if c.startswith(("lead_", "cust_", "name_", "ph_", "em_", "geo_", "rep_"))] + ["company_id", "user_id", "company_leads_30d"]
aipw(R8, (fo.notna() & (fo <= 1)).values, "contact_first_hour", "new lead: first outbound contact within 1 h", cols=static)
R9 = SN[(SN.off_h == 24) & SN.first_out_h.isna()]
f9 = first_in_window(OUT, R9, "t", 48)
aipw(R9, R9.sid.map(f9.ts).notna().values, "contact_day2_if_missed", "not contacted on day 1: contact in the next 2 days")
pd.DataFrame(RES).to_csv("data/out/actions.csv", index=False)
