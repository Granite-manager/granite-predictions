"""Build deal-level table, linked event tables and point-in-time snapshot features.

Outputs (in out/):
  deals.parquet     one row per deal: label, close time, static lead/geo/contact fields
  events.parquet    every timestamped event linked to a deal (sms/call/email/note/activity/quote/...)
  texts.parquet     text units (for embeddings) with deal_id, ts, kind
  snaps.parquet     one row per (deal, snapshot offset) with all point-in-time features
"""
import json, re, math
import numpy as np, pandas as pd

CUTOFF = pd.Timestamp("2026-11-22")
STALE_DAYS = 120
OFFSETS_H = [1, 24, 72, 168, 336, 720, 1440, 2160]
r = lambda k: pd.read_parquet(f"data/raw/{k}.parquet")


def l10(s):
    s = re.sub(r"\D", "", str(s or ""))
    return s[-10:] if len(s) >= 10 else None


# ---------------------------------------------------------------- deals
d = r("deals")
d = d[d.deleted_at.isna()].copy()
c = r("customers").set_index("id")
c["p1"] = c.phone.map(l10)
c["p2"] = c.phone_2.map(l10)
d = d.join(c.drop(columns=["deleted_at"]).add_prefix("c_"), on="customer_id")
d = d[(d.c_company_id.notna()) & (d.c_company_id != 5) & (d.created_at < CUTOFF)].copy()
d["company_id"] = d.c_company_id.astype(int)

dl = r("deals_list").set_index("id")
gl = r("groups_list").set_index("id")
dl["group_name"] = dl.group_id.map(gl.name)
STAGE_MAP = {
    "Not Contacted Yet": "new", "No Contact Yet": "new", "No Contact": "new", "New Prospect": "new",
    "Attempted Contact": "attempted", "Contacted": "contacted", "Contact Made": "contacted",
    "In conversation": "contacted", "Contact Made - Awaiting Choices": "contacted",
    "Short Term Leads": "short_term", "Long Terms Lead": "long_term", "Long Term": "long_term",
    "Ready For Estimate": "ready_estimate", "Got a Quote": "quoted", "Awaiting Decision": "quoted",
    "On Hold": "on_hold", "On-Hold": "won_pipe", "Closed Won": "closed_won", "Closed Lost": "closed_lost",
}
dl["stage"] = dl.name.map(STAGE_MAP)
dl.loc[dl.group_name.eq("Won Deals"), "stage"] = "won_pipe"
dl["stage"] = dl.stage.fillna("other")

sales = r("sales")
first_sale = sales.groupby("deal_id").sale_date.min()
sh = r("stage_history")
sh = sh[sh.deal_id.isin(d.id)].copy()
sh["stage"] = sh.list_id.map(dl.stage)
closing_entry = sh[sh.stage.isin(["closed_won", "closed_lost", "won_pipe"])].groupby("deal_id").entered_at.min()

d["close_t"] = d.updated_at
won = d.is_won.eq(1)
d.loc[won, "close_t"] = pd.concat([d.loc[won, "updated_at"], d.loc[won, "id"].map(first_sale)], axis=1).min(axis=1)
ce = d.id.map(closing_entry)
closed = d.is_won.notna()
d.loc[closed, "close_t"] = pd.concat([d.loc[closed, "close_t"], ce[closed]], axis=1).min(axis=1)
d.loc[d.close_t < d.created_at, "close_t"] = d.created_at
d.loc[~closed, "close_t"] = pd.NaT

age_cut = (CUTOFF - d.created_at).dt.days
d["label"] = np.where(d.is_won.eq(1), 1, np.where(d.is_won.eq(0), 0, np.where(age_cut > STALE_DAYS, 0, -1)))
d["label_kind"] = np.where(d.is_won.eq(1), "won", np.where(d.is_won.eq(0), "lost", np.where(age_cut > STALE_DAYS, "stale_open", "young_open")))
d["end_t"] = d.close_t.fillna(CUTOFF)
d["instant"] = (d.end_t - d.created_at) < pd.Timedelta(hours=1)
d["imported"] = d.created_by.eq("Pipedrive import") | d.c_created_by.eq("Pipedrive import") | d.c_created_by.eq("Slabware import")

# ---------------------------------------------------------------- static fields
def norm(s):
    return s.fillna("").astype(str).str.strip().str.lower().str.replace(r"[\s_]+", " ", regex=True).replace("", np.nan)

S = pd.DataFrame(index=d.index)
S["deal_id"] = d.id
S["company_id"] = d.company_id
for col in ["source", "referral_source", "form_name", "remodal_type", "when_start", "sink", "backsplash",
            "kitchen_stove", "remove_and_dispose", "contact_time", "customer_temperature", "state"]:
    S[f"lead_{col}"] = norm(d[f"c_{col}"])
S["lead_project_size"] = pd.to_numeric(d.c_project_size.astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")
S["lead_has_campaign"] = d.c_compaign_name.notna().astype(int)
S["lead_campaign"] = norm(d.c_compaign_name)
S["lead_adset"] = norm(d.c_adset_name)
S["lead_ad"] = norm(d.c_ad_name)
S["lead_has_message"] = d.c_your_message.fillna("").str.len().gt(0).astype(int)
S["lead_message_len"] = d.c_your_message.fillna("").str.len()
S["lead_has_file"] = d.c_attached_file.notna().astype(int)
S["lead_details_len"] = d.c_details.fillna("").str.len()
S["lead_desc_len"] = d.description.fillna("").str.len()
S["lead_imported"] = d.imported.astype(int)
S["lead_created_by_rep"] = (d.created_by.notna() & ~d.imported).astype(int)
S["lead_hour"] = d.created_at.dt.hour
S["lead_dow"] = d.created_at.dt.dayofweek
S["lead_month"] = d.created_at.dt.month
S["lead_is_weekend"] = (S.lead_dow >= 5).astype(int)
# GMQ Pipedrive descriptions: "Project Type (rough value): Kitchen $3500", "Platform: fb", "Form Type: ..."
desc = d.description.fillna("")
S["lead_desc_platform"] = norm(desc.str.extract(r"Platform:\s*([^\n]+)")[0])
S["lead_desc_formtype"] = norm(desc.str.extract(r"Form Type:\s*([^\n]+)")[0])
S["lead_desc_projtype"] = norm(desc.str.extract(r"Project Type[^:]*:\s*([A-Za-z ]+)")[0])
S["lead_desc_rough_value"] = pd.to_numeric(desc.str.extract(r"Project Type[^:]*:[^\n$]*\$\s*([\d,]+)")[0].str.replace(",", ""), errors="coerce")
S["lead_desc_how_found"] = norm(desc.str.extract(r"How did they find us\?:\s*([^\n]+)")[0])

# customer history: repeat customers / customers created before the deal
cust_first = c.created_date
S["cust_age_days_at_deal"] = (d.created_at - d.customer_id.map(cust_first)).dt.total_seconds() / 86400
dd = d.sort_values("created_at")
S["cust_prior_deals"] = dd.groupby("customer_id").cumcount().reindex(d.index)
prior_won = dd.assign(w=dd.is_won.eq(1).astype(int)).groupby("customer_id").w.transform(lambda s: s.shift().fillna(0).cumsum())
S["cust_prior_won"] = prior_won.reindex(d.index)
S["cust_has_parent"] = d.c_parent_id.notna().astype(int)

# name
name = d.c_name.fillna("").str.strip()
S["name_words"] = name.str.split().str.len().fillna(0)
S["name_len"] = name.str.len()
S["name_all_lower"] = (name.str.len().gt(0) & name.eq(name.str.lower())).astype(int)
S["name_all_upper"] = (name.str.len().gt(1) & name.eq(name.str.upper())).astype(int)
S["name_has_digit"] = name.str.contains(r"\d").astype(int)
S["has_company_name"] = d.c_company_name.notna().astype(int)

# phone
S["ph_has"] = d.c_p1.notna().astype(int)
S["ph_has2"] = d.c_p2.notna().astype(int)
S["ph_area"] = d.c_p1.str[:3]
S["ph_exchange"] = d.c_p1.str[:6]
S["ph_tollfree"] = S.ph_area.isin(["800", "833", "844", "855", "866", "877", "888"]).astype(int)
top_area = S.groupby("company_id").ph_area.agg(lambda s: s.value_counts().index[0] if s.notna().any() else None)
S["ph_area_is_company_top"] = (S.ph_area == S.company_id.map(top_area)).astype(int)
area_share = S.groupby(["company_id", "ph_area"]).size() / S.groupby("company_id").size()
S["ph_area_share_in_company"] = [area_share.get((a, b), np.nan) for a, b in zip(S.company_id, S.ph_area)]

# email
ce_ = r("customers_emails").sort_values("id").groupby("customer_id").email.first()
em = d.customer_id.map(ce_).fillna("").str.lower().str.strip()
S["em_has"] = em.str.contains("@").astype(int)
dom = em.str.split("@").str[-1].where(S.em_has.eq(1))
FREE = {"gmail.com", "yahoo.com", "icloud.com", "hotmail.com", "aol.com", "outlook.com", "comcast.net", "live.com", "msn.com", "att.net", "bellsouth.net", "me.com", "charter.net", "sbcglobal.net", "ymail.com", "protonmail.com", "mac.com"}
S["em_domain"] = dom.where(dom.isin(FREE), np.where(dom.isna(), None, np.where(dom.str.endswith(".edu", na=False), "edu", np.where(dom.str.endswith(".gov", na=False) | dom.str.endswith(".mil", na=False), "gov", "custom"))))
local = em.str.split("@").str[0]
S["em_local_digits"] = local.str.count(r"\d").where(S.em_has.eq(1))
S["em_local_len"] = local.str.len().where(S.em_has.eq(1))
first = name.str.split().str[0].fillna("").str.lower()
S["em_has_first_name"] = [int(bool(f) and len(f) > 2 and f in l) for f, l in zip(first, local)]
S.loc[S.em_has.eq(0), "em_has_first_name"] = np.nan
S["em_n"] = d.customer_id.map(r("customers_emails").groupby("customer_id").size()).fillna(0)
S["em_created_at"] = d.customer_id.map(r("customers_emails").groupby("customer_id").created_at.min())

# geo: zip from postal_code or address; city
z = d.c_postal_code.fillna("").astype(str).str.extract(r"(\d{5})")[0]
z2 = d.c_address.fillna("").str.extract(r"\b(\d{5})(?:-\d{4})?\s*$")[0]
z3 = d.c_address.fillna("").str.extract(r"\b(\d{5})\b")[0]
S["geo_zip"] = z.fillna(z2).fillna(z3)
S["geo_zip3"] = S.geo_zip.str[:3]
S["geo_city"] = norm(d.c_city)
S["geo_has_address"] = d.c_address.notna().astype(int)
S["geo_addr_has_unit"] = d.c_address.fillna("").str.contains(r"\b(apt|unit|#|suite|ste)\b", case=False).astype(int)
S["geo_addr_len"] = d.c_address.fillna("").str.len()

d[["id", "customer_id", "company_id", "user_id", "created_at", "close_t", "end_t", "label", "label_kind", "instant",
   "imported", "is_won", "amount", "lost_reason", "list_id", "c_p1", "c_p2"]].to_parquet("data/out/deals_raw.parquet")
S.to_parquet("data/out/static.parquet")
print("deals", len(d), d.label_kind.value_counts().to_dict(), "instant", d.instant.sum())

# ---------------------------------------------------------------- events (deal_id, ts, kind, ...)
deal_cust = d[["id", "customer_id", "company_id", "created_at", "end_t", "c_p1", "c_p2"]]
phone_map = pd.concat([deal_cust[["id", "company_id", "c_p1"]].rename(columns={"c_p1": "ph"}),
                       deal_cust[["id", "company_id", "c_p2"]].rename(columns={"c_p2": "ph"})]).dropna().drop_duplicates()
WINDOW_BEFORE = pd.Timedelta(days=60)


def attach_phone(ev, phone_col, ts_col):
    ev = ev.dropna(subset=[phone_col]).drop(columns=["deal_id", "customer_id"], errors="ignore").copy()
    ev["ph"] = ev[phone_col].astype(str).str[-10:]
    pm = phone_map.rename(columns={"id": "deal_id"})
    m = ev.merge(pm, on=["company_id", "ph"])
    m = m.merge(deal_cust[["id", "created_at"]].rename(columns={"id": "deal_id"}), on="deal_id")
    m = m[m[ts_col] >= m.created_at - WINDOW_BEFORE]
    return m.rename(columns={ts_col: "ts"})


E = []
sms = pd.concat([r(k).assign(chan=k.split("_")[0]) for k in ["cloudtalk_sms", "ringcentral_sms", "telnyx_sms"]], ignore_index=True)
sms = sms[sms.is_echo.fillna(0).eq(0)]
sms["sms_uid"] = sms.chan + ":" + sms.id.astype(str)
m = attach_phone(sms, "phone_digits", "created_date")
m["kind"] = np.where(m.direction.eq("inbound"), "sms_in", "sms_out")
m["auto"] = m.is_automated.fillna(0)
m["len"] = m.text.fillna("").str.len()
m["media"] = m.get("has_media", pd.Series(0, index=m.index)).fillna(0).astype(float)
m["failed"] = m.status.astype(str).str.contains("fail|undeliv|error", case=False).astype(int)
E.append(m[["deal_id", "ts", "kind", "auto", "len", "media", "failed", "sender_user_id"]].assign(uid=m.sms_uid))
texts = [m[["deal_id", "ts", "kind", "text"]].assign(uid=m.sms_uid)]

pc = r("provider_calls")
pc["ts"] = pd.to_datetime(pc.started_at)
for col in ["billsec", "talking_time", "waiting_time"]:
    pc[col] = pd.to_numeric(pc[col], errors="coerce")
m = attach_phone(pc, "customer_digits", "ts")
m["kind"] = np.where(m.type.eq("incoming"), "call_in", "call_out")
m["talk"] = m.talking_time.fillna(m.billsec).fillna(0)
m["answered"] = (m.talk > 0).astype(int)
m["voicemail"] = m.is_voicemail.astype(str).eq("true").astype(int)
E.append(m[["deal_id", "ts", "kind", "talk", "answered", "voicemail"]])
pcl = r("phone_calls")
pcl["company_id"] = pcl.company_id
m = attach_phone(pcl, "other_last10", "started_at")
m["kind"] = np.where(m.direction.astype(str).str.startswith("in"), "call_in", "call_out")
m["talk"] = m.duration_secs.fillna(0).where(m.answered_at.notna(), 0)
m["answered"] = m.answered_at.notna().astype(int)
m["voicemail"] = 0
E.append(m[["deal_id", "ts", "kind", "talk", "answered", "voicemail"]])
tr = m[m.transcript.fillna("").str.len() > 20]
texts.append(tr[["deal_id", "ts", "kind", "transcript"]].rename(columns={"transcript": "text"}).assign(kind="transcript", uid="pc:" + tr.id.astype(str)))

# emails: participants with customer ids or deal_id
e = r("emails")
e = e[e.deleted_at.isna() & e.is_draft.fillna(0).eq(0)]
ep = r("email_participants")
cust2deal = d[["id", "customer_id", "created_at"]]
frm = ep[ep.type.eq("from")].drop_duplicates("email_id").set_index("email_id")
inb = frm[frm.customer_id.notna() & frm.user_id.isna()][["customer_id"]].reset_index()
inb["dir"] = "in"
to = ep[ep.type.isin(["to", "cc"]) & ep.customer_id.notna() & ep.user_id.isna()][["email_id", "customer_id"]]
to = to[to.email_id.isin(frm[frm.user_id.notna() | frm.customer_id.isna()].index)]
to["dir"] = "out"
ec = pd.concat([inb, to]).drop_duplicates(["email_id", "customer_id"])
ec = ec.merge(cust2deal, on="customer_id").rename(columns={"id": "deal_id"})
byd = e[e.deal_id.notna()][["id", "deal_id", "sender_user_id"]].rename(columns={"id": "email_id"})
byd["dir"] = np.where(byd.sender_user_id.notna(), "out", "in")
ec = pd.concat([ec[["email_id", "deal_id", "dir"]], byd[["email_id", "deal_id", "dir"]]]).drop_duplicates(["email_id", "deal_id"])
ec = ec.merge(e[["id", "sent_at", "subject", "body", "is_automated", "message_id"]], left_on="email_id", right_on="id")
ec = ec.merge(d[["id", "created_at"]].rename(columns={"id": "deal_id"}), on="deal_id")
ec = ec[ec.sent_at >= ec.created_at - WINDOW_BEFORE]
reads = r("email_reads").groupby("message_id").read_at.min()
ec["read_at"] = ec.message_id.map(reads)
ec["kind"] = np.where(ec.dir.eq("in"), "email_in", "email_out")
ec["auto"] = ec.is_automated.fillna(0)
ec["len"] = ec.body.fillna("").str.len()
E.append(ec[["deal_id", "sent_at", "kind", "auto", "len"]].rename(columns={"sent_at": "ts"}))
rd = ec[ec.read_at.notna() & ec.dir.eq("out")]
E.append(rd[["deal_id", "read_at"]].rename(columns={"read_at": "ts"}).assign(kind="email_read"))
ec["text"] = (ec.subject.fillna("") + "\n" + ec.body.fillna("").str[:2000])
texts.append(ec[["deal_id", "sent_at", "kind", "text"]].rename(columns={"sent_at": "ts"}).assign(uid="em:" + ec.email_id.astype(str)))

# notes / activities / stage changes / reassignments
n = r("notes")
n = n[n.deleted_at.isna() & n.deal_id.isin(d.id)]
E.append(n[["deal_id", "created_at"]].rename(columns={"created_at": "ts"}).assign(kind="note", len=n.content.fillna("").str.len()))
texts.append(n[["deal_id", "created_at", "content"]].rename(columns={"created_at": "ts", "content": "text"}).assign(kind="note", uid="n:" + n.id.astype(str)))
a = r("activities")
a = a[a.deleted_at.isna() & a.deal_id.isin(d.id)]
E.append(a[["deal_id", "created_at"]].rename(columns={"created_at": "ts"}).assign(kind="activity_created"))
E.append(a[a.completed_at.notna()][["deal_id", "completed_at"]].rename(columns={"completed_at": "ts"}).assign(kind="activity_done"))
E.append(a[a.deadline.notna()][["deal_id", "deadline"]].rename(columns={"deadline": "ts"}).assign(kind="activity_deadline", done_at=a.completed_at))
ch = r("cust_history").merge(cust2deal, on="customer_id")
E.append(ch[["id_y", "updated_at"]].rename(columns={"id_y": "deal_id", "updated_at": "ts"}).assign(kind="reassigned"))

# quotes: linked by deal_id or customer_id (time window)
q = r("quotes")
q = q[q.deleted_at.isna()]
qr = r("quote_revisions")


def qtot(s):
    try:
        v = [o.get("total") or 0 for o in json.loads(s)]
        return (max(v), min(v), len(v)) if v else (np.nan, np.nan, 0)
    except Exception:
        return (np.nan, np.nan, 0)


def qdesign(s):
    out = dict(q_sqin=np.nan, q_counters=0, q_cutouts=0, q_sinks=0, q_splash_edges=0, q_runs=0, q_items=0)
    try:
        j = json.loads(s)
    except Exception:
        return out
    area = 0
    for ctr in j.get("counters", []) or []:
        out["q_counters"] += 1
        for run in ctr.get("runs", []) or []:
            out["q_runs"] += 1
            area += (run.get("length") or 0) * (run.get("depth") or 0)
            for co in run.get("cutouts", []) or []:
                out["q_cutouts"] += 1
                out["q_sinks"] += int(co.get("kind") == "sink")
        out["q_splash_edges"] += sum(1 for ed in ctr.get("edges", []) or [] if ed.get("type") == "splash")
    out["q_sqin"] = area if out["q_counters"] else np.nan
    out["q_items"] = len(j.get("items", []) or [])
    return out


def qmat(s):
    try:
        j = json.loads(s)
        a = j[0]["byArea"]
        first = next(iter(a.values()))
        return first.get("product"), first.get("color"), len(j)
    except Exception:
        return (None, None, 0)


qr1 = qr.sort_values("revision_number").groupby("quote_id").first()
tt = qr1.totals_json.map(qtot)
qr1["q_total_max"] = tt.str[0]
qr1["q_total_min"] = tt.str[1]
des = pd.DataFrame(list(qr1.drawing_json.map(qdesign)), index=qr1.index)
mt = qr1.options_json.map(qmat)
qr1["q_product"] = mt.str[0]
qr1["q_color"] = mt.str[1]
qr1["q_n_options"] = mt.str[2]
qinfo = pd.concat([qr1[["q_total_max", "q_total_min", "q_product", "q_color", "q_n_options", "has_cutout"]], des], axis=1)
q = q.join(qinfo, on="id")
qd = pd.concat([q[q.deal_id.notna()].merge(d[["id", "created_at"]], left_on="deal_id", right_on="id", suffixes=("", "_d")),
                q.merge(d[["id", "customer_id", "created_at"]].rename(columns={"id": "deal_id2"}), on="customer_id").assign(deal_id=lambda x: x.deal_id2)])
qd["deal_id"] = qd.deal_id.astype(int)
qd = qd[qd.created_date >= qd.created_at - WINDOW_BEFORE].drop_duplicates(["id", "deal_id"])
qd["disc"] = qd.discount_pct.fillna(0)
E.append(qd[["deal_id", "created_date", "q_total_max", "q_total_min", "q_product", "q_color", "q_n_options", "q_sqin", "q_counters",
             "q_cutouts", "q_sinks", "q_splash_edges", "q_runs", "q_items", "disc"]].rename(columns={"created_date": "ts"}).assign(kind="quote", qid=qd.id))
revs = qr.merge(qd[["id", "deal_id"]], left_on="quote_id", right_on="id")
E.append(revs[["deal_id", "created_at"]].rename(columns={"created_at": "ts"}).assign(kind="quote_rev"))
qe = r("quote_emails").merge(qd[["id", "deal_id"]], left_on="quote_id", right_on="id", suffixes=("", "_q"))
E.append(qe[["deal_id", "sent_at"]].rename(columns={"sent_at": "ts"}).assign(kind="quote_sent"))
qv = r("quote_email_views").merge(qe[["id", "deal_id"]], left_on="quote_email_id", right_on="id")
E.append(qv[["deal_id", "viewed_at"]].rename(columns={"viewed_at": "ts"}).assign(kind="quote_view"))
# quote_emails has first/last view and view_count even without view rows (older): first view as event
fv = qe[qe.first_viewed_at.notna()]
E.append(fv[["deal_id", "first_viewed_at"]].rename(columns={"first_viewed_at": "ts"}).assign(kind="quote_first_view"))

# events (appointments) linked by customer
ev = r("events")
ev = ev[ev.deleted_date.isna() & ev.customer_id.notna()].merge(cust2deal, on="customer_id")
ev = ev[ev.created_date >= ev.created_at - WINDOW_BEFORE]
E.append(ev[["id_y", "created_date"]].rename(columns={"id_y": "deal_id", "created_date": "ts"}).assign(kind="appt_created", appt_start=ev.start_date))

# stage changes
E.append(sh[["deal_id", "entered_at"]].rename(columns={"entered_at": "ts"}).assign(kind="stage", stage=sh.stage, list_id=sh.list_id))

EV = pd.concat(E, ignore_index=True)
EV = EV[EV.deal_id.isin(d.id) & EV.ts.notna()]
EV.to_parquet("data/out/events.parquet")
print(EV.kind.value_counts().to_string())
T = pd.concat(texts, ignore_index=True)
T = T[T.text.fillna("").str.strip().str.len() > 1]
# lead text as a text unit at deal creation
# only what the lead form sends at creation: deal description and customer notes are edited later (leak)
lead_txt = d.c_your_message.fillna("").str.strip()  # customers.details is edited by reps later (leak)
lt = pd.DataFrame({"deal_id": d.id, "ts": d.created_at, "kind": "lead", "text": lead_txt, "uid": "lead:" + d.id.astype(str)})
T = pd.concat([T, lt[lt.text.str.len() > 1]], ignore_index=True)
T.to_parquet("data/out/texts.parquet")
print("texts", T.kind.value_counts().to_dict())
