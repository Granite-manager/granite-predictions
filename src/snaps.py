"""Point-in-time snapshot features: one row per (deal, offset). Only events with ts < t are used."""
import numpy as np, pandas as pd

OFFSETS_H = [1, 24, 72, 168, 336, 720, 1440, 2160]
D = pd.read_parquet("data/out/deals_raw.parquet")
S = pd.read_parquet("data/out/static.parquet")
EV = pd.read_parquet("data/out/events.parquet")

D = D[~D.instant].copy()
rows = []
for off in OFFSETS_H:
    t = D.created_at + pd.Timedelta(hours=off)
    keep = t < D.end_t
    rows.append(pd.DataFrame({"deal_id": D.id[keep], "off_h": off, "t": t[keep]}))
SN = pd.concat(rows, ignore_index=True)
SN = SN.merge(D[["id", "created_at", "end_t", "label", "label_kind", "company_id", "user_id", "imported"]].rename(columns={"id": "deal_id"}), on="deal_id")
SN["age_days"] = (SN.t - SN.created_at).dt.total_seconds() / 86400
SN["days_to_end"] = (SN.end_t - SN.t).dt.total_seconds() / 86400  # NOT a feature, diagnostics only
SN["sid"] = np.arange(len(SN))
print("snapshots", len(SN), SN.groupby("off_h").size().to_dict())

EV = EV[EV.deal_id.isin(D.id)].copy()
# latencies: for message-like events, time since previous opposite-direction message
msg = EV[EV.kind.isin(["sms_in", "sms_out", "call_in", "call_out", "email_in", "email_out"])].copy()
msg = msg[~((msg.kind == "sms_out") & (msg.auto.fillna(0) == 1))]
msg["dir"] = np.where(msg.kind.str.endswith("_in"), "in", "out")
msg = msg.sort_values(["deal_id", "ts"])
for side, other in [("in", "out"), ("out", "in")]:
    last_other = msg.ts.where(msg.dir == other).groupby(msg.deal_id).ffill()
    prev_dir = msg.groupby("deal_id").dir.shift()
    lat = (msg.ts - last_other).dt.total_seconds() / 3600
    msg[f"lat_{side}"] = lat.where((msg.dir == side) & (prev_dir == other))
EV = EV.merge(msg[["lat_in", "lat_out"]], left_index=True, right_index=True, how="left")

X = SN[["sid", "deal_id", "t", "created_at"]].merge(EV, on="deal_id")
X = X[X.ts < X.t]
import os
if os.environ.get("AFTER_CREATE_ONLY"):
    X = X[X.ts >= X.created_at]  # experiment: drop everything dated before the deal
X["ago_d"] = (X.t - X.ts).dt.total_seconds() / 86400
X["since_create_h"] = (X.ts - X.created_at).dt.total_seconds() / 3600
g = X.groupby("sid")

F = pd.DataFrame(index=SN.sid)
KINDS = ["sms_in", "sms_out", "call_in", "call_out", "email_in", "email_out", "email_read", "note", "activity_created",
         "activity_done", "quote", "quote_rev", "quote_sent", "quote_view", "quote_first_view", "appt_created", "reassigned", "stage"]
cnt = X.groupby(["sid", "kind"]).size().unstack(fill_value=0).reindex(columns=KINDS, fill_value=0)
F = F.join(cnt.add_prefix("n_"))
cnt7 = X[X.ago_d <= 7].groupby(["sid", "kind"]).size().unstack(fill_value=0).reindex(columns=KINDS, fill_value=0)
F = F.join(cnt7.add_prefix("n7_"))
F = F.fillna({c: 0 for c in F.columns})

def first_h(kinds, name):
    s = X[X.kind.isin(kinds) & (X.since_create_h >= -1)].groupby("sid").since_create_h.min()
    F[name] = s

def last_ago(kinds, name):
    F[name] = X[X.kind.isin(kinds)].groupby("sid").ago_d.min()

OUT = ["sms_out", "call_out", "email_out"]
IN = ["sms_in", "call_in", "email_in"]
first_h(OUT, "first_out_h")
first_h(["call_out"], "first_call_out_h")
first_h(["sms_out"], "first_sms_out_h")
first_h(IN, "first_in_h")
first_h(["quote"], "first_quote_h")
first_h(["quote_sent"], "first_quote_sent_h")
last_ago(IN, "last_in_ago_d")
last_ago(OUT, "last_out_ago_d")
last_ago(["quote_view", "quote_first_view"], "last_quote_view_ago_d")
last_ago(IN + OUT + ["note", "activity_done", "stage"], "last_any_ago_d")
F["n_in"] = F[[f"n_{k}" for k in IN]].sum(axis=1)
F["n_out"] = F[[f"n_{k}" for k in OUT]].sum(axis=1)
F["in_out_ratio"] = F.n_in / (F.n_out + 1)
F["n7_in"] = F[[f"n7_{k}" for k in IN]].sum(axis=1)
F["n7_out"] = F[[f"n7_{k}" for k in OUT]].sum(axis=1)
F["replied"] = (F.n_in > 0).astype(int)
F["outbound_unanswered_streak"] = np.nan
# consecutive outbound messages since last inbound
m2 = X[X.kind.isin(IN + OUT)].sort_values(["sid", "ts"])
m2["is_in"] = m2.kind.isin(IN).astype(int)
m2["grp"] = m2.groupby("sid").is_in.cumsum()
last_grp = m2.groupby("sid").grp.transform("max")
F["outbound_unanswered_streak"] = m2[(m2.grp == last_grp) & (m2.is_in == 0)].groupby("sid").size().reindex(F.index).fillna(0)

calls = X[X.kind.isin(["call_in", "call_out"])]
F["call_talk_sum"] = calls.groupby("sid").talk.sum().reindex(F.index).fillna(0)
F["call_talk_max"] = calls.groupby("sid").talk.max()
F["call_answered"] = calls.groupby("sid").answered.sum().reindex(F.index).fillna(0)
F["call_answer_rate"] = F.call_answered / (F.n_call_in + F.n_call_out).replace(0, np.nan)
F["call_voicemail"] = calls.groupby("sid").voicemail.sum().reindex(F.index).fillna(0)
F["call_long_5m"] = calls[calls.talk >= 300].groupby("sid").size().reindex(F.index).fillna(0)
si = X[X.kind == "sms_in"]
F["sms_in_len_mean"] = si.groupby("sid")["len"].mean()
F["sms_in_len_sum"] = si.groupby("sid")["len"].sum().reindex(F.index).fillna(0)
F["sms_in_media"] = si.groupby("sid").media.sum().reindex(F.index).fillna(0)
so = X[X.kind == "sms_out"]
F["sms_out_auto"] = so.groupby("sid").auto.sum().reindex(F.index).fillna(0)
F["sms_out_failed"] = so.groupby("sid").failed.sum().reindex(F.index).fillna(0)
F["sms_out_len_mean"] = so.groupby("sid")["len"].mean()
F["rep_reply_lat_med_h"] = X.groupby("sid").lat_out.median()
F["cust_reply_lat_med_h"] = X.groupby("sid").lat_in.median()
F["cust_reply_lat_min_h"] = X.groupby("sid").lat_in.min()
F["n_rep_ids"] = so.groupby("sid").sender_user_id.nunique().reindex(F.index).fillna(0)
F["active_days"] = X[X.kind.isin(IN + OUT)].assign(day=lambda x: x.ts.dt.floor("D")).groupby("sid").day.nunique().reindex(F.index).fillna(0)
F["in_active_days"] = X[X.kind.isin(IN)].assign(day=lambda x: x.ts.dt.floor("D")).groupby("sid").day.nunique().reindex(F.index).fillna(0)
F["inbound_hour_mean"] = X[X.kind.isin(IN)].assign(h=lambda x: x.ts.dt.hour).groupby("sid").h.mean()

# quotes (latest quote before t)
qq = X[X.kind == "quote"].sort_values("ts").groupby("sid").last()
for col in ["q_total_max", "q_total_min", "q_product", "q_color", "q_n_options", "q_sqin", "q_counters", "q_cutouts", "q_sinks", "q_splash_edges", "q_runs", "q_items", "disc"]:
    F[col] = qq[col]
F["q_total_spread"] = (F.q_total_max - F.q_total_min) / F.q_total_max.replace(0, np.nan)
F["q_price_per_sqft"] = F.q_total_max / (F.q_sqin / 144).replace(0, np.nan)
F["n_quote_distinct"] = X[X.kind == "quote"].groupby("sid").qid.nunique().reindex(F.index).fillna(0)
F["q_views_per_sent"] = F.n_quote_view / F.n_quote_sent.replace(0, np.nan)

# activities overdue at t
ad = X[X.kind == "activity_deadline"]
od = ad[(ad.done_at.isna()) | (ad.done_at > ad.t)]
F["act_overdue"] = od.groupby("sid").size().reindex(F.index).fillna(0)
F["act_done_rate"] = F.n_activity_done / F.n_activity_created.replace(0, np.nan)
# appointments scheduled in the future at t
ap = X[X.kind == "appt_created"]
F["appt_future"] = ap[ap.appt_start > ap.t].groupby("sid").size().reindex(F.index).fillna(0)
F["note_len_sum"] = X[X.kind == "note"].groupby("sid")["len"].sum().reindex(F.index).fillna(0)

# stage at t
STAGE_ORD = {"new": 0, "attempted": 1, "contacted": 2, "short_term": 3, "long_term": 3, "ready_estimate": 4, "quoted": 5, "on_hold": 4, "other": 2}
st = X[X.kind == "stage"].sort_values("ts")
lastst = st.groupby("sid").last()
F["stage_now"] = lastst.stage
F["stage_days_in"] = lastst.ago_d
F["stage_n_changes"] = st.groupby("sid").size().reindex(F.index).fillna(0)
st = st.assign(o=st.stage.map(STAGE_ORD))
F["stage_max_ord"] = st.groupby("sid").o.max()
st["back"] = st.groupby("sid").o.diff() < 0
F["stage_back_moves"] = st.groupby("sid").back.sum().reindex(F.index).fillna(0)
F["stage_reached_quoted"] = (F.stage_max_ord >= 5).astype(int)
F["stage_known"] = F.stage_now.notna().astype(int)

SN = SN.set_index("sid").join(F)

# rep features, time-aware: win rate of deals that rep closed before t (shrunk)
Dall = pd.read_parquet("data/out/deals_raw.parquet")
cl = Dall[Dall.label_kind.isin(["won", "lost"]) & Dall.user_id.notna()].sort_values("close_t")
cl["w"] = (cl.label == 1).astype(int)
cl["cw"] = cl.groupby("user_id").w.cumsum()
cl["cn"] = cl.groupby("user_id").cumcount() + 1
SN = SN.reset_index().sort_values("t")
cl = cl.sort_values("close_t")
m = pd.merge_asof(SN[["sid", "t", "user_id"]].dropna(subset=["user_id"]), cl[["close_t", "user_id", "cw", "cn"]], left_on="t", right_on="close_t", by="user_id", allow_exact_matches=False)
prior = 0.38
m["rep_prior_wr"] = (m.cw.fillna(0) + 20 * prior) / (m.cn.fillna(0) + 20)
m["rep_prior_n"] = m.cn.fillna(0)
SN = SN.merge(m[["sid", "rep_prior_wr", "rep_prior_n"]], on="sid", how="left")
# rep load: open deals assigned at t (created before t, end after t)
Dl = Dall[Dall.user_id.notna()][["user_id", "created_at", "end_t"]]
loads = []
for uid, grp in SN.dropna(subset=["user_id"]).groupby("user_id"):
    dd = Dl[Dl.user_id == uid]
    cs = np.sort(dd.created_at.values); es = np.sort(dd.end_t.values)
    tt = grp.t.values
    loads.append(pd.Series(np.searchsorted(cs, tt) - np.searchsorted(es, tt), index=grp.sid))
SN["rep_open_load"] = SN.sid.map(pd.concat(loads))
# company leads that week (volume / seasonality)
Dc = Dall[["company_id", "created_at"]]
SN["company_leads_30d"] = [((Dc.company_id == cid) & (Dc.created_at < tt) & (Dc.created_at >= tt - pd.Timedelta(days=30))).sum() for cid, tt in zip(SN.company_id, SN.t)]

SN = SN.merge(S.drop(columns=["company_id"]), on="deal_id", how="left")
# prior won deals of the same customer, counted only once they were won before t (static version used final outcomes)
cust = Dall.set_index("id").customer_id
SN["customer_id"] = SN.deal_id.map(cust)
pw = SN[["sid", "customer_id", "created_at", "t"]].merge(
    Dall[Dall.label_kind == "won"][["customer_id", "created_at", "close_t"]].rename(columns={"created_at": "p_created"}), on="customer_id")
pw = pw[(pw.p_created < pw.created_at) & (pw.close_t < pw.t)]
SN["cust_prior_won"] = SN.sid.map(pw.groupby("sid").size()).fillna(0)
SN = SN.drop(columns=["customer_id"])
SN.to_parquet(os.environ.get("SNAPS_OUT", "data/out/snaps.parquet"))
print(SN.shape)
