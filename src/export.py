"""Train the serving model (16 features), calibrate it, export it as compact JSON for the Rust Lambda,
and write parity cases: requests in the Lambda's input format with the probability Python expects.

  uv run python src/export.py v1            -> data/out/deal_odds_v1.json, data/out/parity_v1.json
  aws s3 cp data/out/deal_odds_v1.json s3://<bucket>/deal-odds/v1/model.json
"""
import sys, json, pickle, re; sys.path.insert(0, "src")
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from common import *

VERSION = sys.argv[1] if len(sys.argv) > 1 else "v1"
FEATURES = ["lead_source", "rep_prior_n", "age_days", "stage_max_ord", "lead_referral_source", "rep_prior_wr",
            "rep_open_load", "lead_project_size", "n_quote", "lead_month", "cust_age_days_at_deal", "note_len_sum",
            "lead_hour", "stage_days_in", "cust_prior_won", "company_id"]
CATEGORICAL = ["lead_source", "lead_referral_source", "company_id"]
# must match build.py / snaps.py
STAGE_MAP = {
    "Not Contacted Yet": "new", "No Contact Yet": "new", "No Contact": "new", "New Prospect": "new",
    "Attempted Contact": "attempted", "Contacted": "contacted", "Contact Made": "contacted",
    "In conversation": "contacted", "Contact Made - Awaiting Choices": "contacted",
    "Short Term Leads": "short_term", "Long Terms Lead": "long_term", "Long Term": "long_term",
    "Ready For Estimate": "ready_estimate", "Got a Quote": "quoted", "Awaiting Decision": "quoted",
    "On Hold": "on_hold", "On-Hold": "won_pipe", "Closed Won": "closed_won", "Closed Lost": "closed_lost",
}
STAGE_ORD = {"new": 0, "attempted": 1, "contacted": 2, "short_term": 3, "long_term": 3, "ready_estimate": 4, "quoted": 5, "on_hold": 4, "other": 2}

cfg = pickle.load(open("data/out/final_lean.pkl", "rb"))
params, rounds = dict(cfg["params"], num_threads=12), cfg["rounds"]
SN = load(strict=True)

# holdout check of exactly this configuration
tr, va = SN[SN.created_at < HOLDOUT_FROM], SN[SN.created_at >= HOLDOUT_FROM]
p, _, _ = fit_predict(tr, va, FEATURES, rounds, params)
ho_auc = roc_auc_score(va.label, p, sample_weight=va.w)
print("holdout AUC", round(ho_auc, 4))

# ---------------------------------------------------------------- confidence: holdout AUC by store group x what has happened
# How well the model ranks buyers above non-buyers depends on the store and on what has already happened with the deal,
# so each answer carries the holdout AUC of its cell. The Lambda computes the level from the request the same way.
CONF_GROUPS = {1: "Indianapolis", 3: "Columbus", 4: "GMQ", 12: "GMQ", 13: "GMQ", 2: "Lexington, Empire", 7: "Lexington, Empire"}
CONF_MIN_AUC, CONF_MIN_DEALS = 0.70, 50


def conf_level(df):
    """replied: the customer wrote or called, or a call was answered; worked: the store wrote, called or moved the
    stage without an answer; untouched: nothing yet."""
    replied = (df.n_in > 0) | (df.call_answered > 0)
    worked = (df.n_out > 0) | (df.stage_n_changes > 0)
    return pd.Series(np.select([replied, worked], ["replied", "worked"], "untouched"), index=df.index)


def conf_cell(x, boot=300):
    deals = x.deal_id.nunique()
    if deals < 10 or x.label.nunique() < 2:
        return dict(auc=None, auc_low=None, auc_high=None, deals=int(deals), snapshots=int(len(x)))
    auc = roc_auc_score(x.label, x.p, sample_weight=x.w)
    ids = x.deal_id.unique(); by = {d: g for d, g in x.groupby("deal_id")}
    rng = np.random.default_rng(0); bs = []
    for _ in range(boot):
        b = pd.concat([by[d] for d in rng.choice(ids, len(ids))])
        if b.label.nunique() == 2:
            bs.append(roc_auc_score(b.label, b.p, sample_weight=b.w))
    lo, hi = np.percentile(bs, [5, 95])
    return dict(auc=round(float(auc), 3), auc_low=round(float(lo), 3), auc_high=round(float(hi), 3), deals=int(deals), snapshots=int(len(x)))


cv = va.assign(p=p, level=conf_level(va), grp=va.company_id.astype(int).map(CONF_GROUPS))
cells = {g: {lv: conf_cell(cv[(cv.grp == g) & (cv.level == lv)]) for lv in ["replied", "worked", "untouched"]}
         for g in dict.fromkeys(CONF_GROUPS.values())}
confidence = dict(min_auc=CONF_MIN_AUC, min_deals=CONF_MIN_DEALS, groups={str(k): v for k, v in CONF_GROUPS.items()}, cells=cells)
for g, row in cells.items():
    print(f"{g:18s}", "  ".join(f"{lv}: {c['auc']} ({c['auc_low']}-{c['auc_high']}) {c['deals']}d" for lv, c in row.items()))

# final model on every labeled snapshot; isotonic calibration from out-of-fold predictions
X = SN[FEATURES]
oof = np.zeros(len(SN))
cust = SN.customer_id.unique()
perm = pd.Series(np.random.default_rng(0).permutation(len(cust)), index=cust)
for a, b in GroupKFold(5).split(SN, groups=SN.customer_id.map(perm)):
    m = lgb.train(params, lgb.Dataset(X.iloc[a], SN.label.iloc[a], weight=SN.w.iloc[a]), rounds)
    oof[b] = m.predict(X.iloc[b])
cv_auc = roc_auc_score(SN.label, oof, sample_weight=SN.w)
print("cv AUC", round(cv_auc, 4))
iso = IsotonicRegression(out_of_bounds="clip", y_min=0.005, y_max=0.995).fit(oof, SN.label, sample_weight=SN.w)
model = lgb.train(params, lgb.Dataset(X, SN.label, weight=SN.w), rounds)
raw_score = model.predict(X, raw_score=True)

cats = {c: [str(v) for v in X[c].cat.categories] for c in CATEGORICAL}
dump = model.dump_model()
assert dump["objective"].startswith("binary"), dump["objective"]
sig = float(re.search(r"sigmoid:([\d.]+)", dump["objective"]).group(1)) if "sigmoid" in dump["objective"] else 1.0
names = dump["feature_names"]
assert names == FEATURES, names


def flat(tree):
    """Pre-order node list. Split: [feature, threshold | [left categories], default_left, missing, left, right, value].
    Leaf: value only (a number)."""
    nodes = []
    def walk(n):
        i = len(nodes)
        nodes.append(None)
        if "leaf_value" in n:
            nodes[i] = n["leaf_value"]
            return i
        l = walk(n["left_child"]); r = walk(n["right_child"])
        if n["decision_type"] == "==":
            thr = [int(x) for x in str(n["threshold"]).split("||")]
        else:
            thr = float(n["threshold"])
        nodes[i] = [n["split_feature"], thr, bool(n["default_left"]), n["missing_type"], l, r, n["internal_value"]]
        return i
    walk(tree["tree_structure"])
    return nodes


out = dict(
    version=VERSION, created=pd.Timestamp.now(tz="UTC").isoformat(), objective="binary", sigmoid=sig,
    metrics=dict(holdout_auc=round(ho_auc, 4), cv_auc=round(cv_auc, 4), training_snapshots=int(len(SN)), training_deals=int(SN.deal_id.nunique())),
    features=[dict(name=f, kind="categorical" if f in CATEGORICAL else "numeric", categories=cats.get(f)) for f in FEATURES],
    trees=[flat(t) for t in dump["tree_info"]],
    calibration=dict(x=[float(v) for v in iso.X_thresholds_], y=[float(v) for v in iso.y_thresholds_]),
    verdicts=dict(likely=0.5, unlikely=0.1),
    confidence=confidence,
    rules=dict(stage_map=STAGE_MAP, won_group="Won Deals", stage_order=STAGE_ORD, rep_prior=0.38, rep_prior_weight=20, quote_window_days=60),
)
path = f"data/out/deal_odds_{VERSION}.json"
json.dump(out, open(path, "w"), separators=(",", ":"))
print("wrote", path, f"{len(json.dumps(out)) / 1024:.0f} KB", len(out["trees"]), "trees")

# ---------------------------------------------------------------- parity cases in the Lambda's request format
raw = lambda k: pd.read_parquet(f"data/raw/{k}.parquet")
D = pd.read_parquet("data/out/deals_raw.parquet").set_index("id")
C = raw("customers").set_index("id")
dl = raw("deals_list").set_index("id"); gl = raw("groups_list").set_index("id")
sh = raw("stage_history")
q = raw("quotes"); q = q[q.deleted_at.isna()]
nt = raw("notes"); nt = nt[nt.deleted_at.isna()]
clean = lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else v
rng = np.random.default_rng(7)
pick = rng.choice(len(SN), 400, replace=False)
fmt = lambda ts: None if pd.isna(ts) else pd.Timestamp(ts).isoformat()
cases = []
for i in pick:
    row = SN.iloc[i]
    did, t = int(row.deal_id), row.t
    d = D.loc[did]; cu = C.loc[d.customer_id]
    st = sh[(sh.deal_id == did) & (sh.entered_at < t)]
    stages = [dict(list_name=dl.name.get(l), group_name=gl.name.get(dl.group_id.get(l)), entered_at=fmt(e)) for l, e in zip(st.list_id, st.entered_at)]
    qs = q[(q.deal_id == did) | (q.customer_id == d.customer_id)].drop_duplicates("id")
    notes = nt[nt.deal_id == did]
    rep = None if pd.isna(row.rep_prior_n) else dict(closed_deals=int(row.rep_prior_n),
                                                     won_deals=int(round(row.rep_prior_wr * (row.rep_prior_n + 20) - 20 * 0.38)),
                                                     open_deals=int(row.rep_open_load))
    req = dict(as_of=fmt(t), deal=dict(id=did, company_id=int(row.company_id), created_at=fmt(d.created_at)),
               customer=dict(created_at=fmt(cu.created_date), source=clean(cu.source), referral_source=clean(cu.referral_source),
                             project_size=None if pd.isna(cu.project_size) else str(cu.project_size), prior_won_deals=int(row.cust_prior_won)),
               rep=rep, stages=stages,
               quotes=[fmt(x) for x in qs.created_date],
               notes=[dict(created_at=fmt(a), chars=len(c or "")) for a, c in zip(notes.created_at, notes.content)],
               activity=dict(inbound=int(row.n_in), outbound=int(row.n_out), answered_calls=int(row.call_answered)))
    feats = {f: (None if pd.isna(row[f]) else (str(row[f]) if f in CATEGORICAL else float(row[f]))) for f in FEATURES}
    cases.append(dict(request=req, features=feats, level=conf_level(SN.iloc[[i]]).iloc[0], raw_score=float(raw_score[i]), probability=float(iso.predict([1 / (1 + np.exp(-sig * raw_score[i]))])[0])))
json.dump(cases, open(f"data/out/parity_{VERSION}.json", "w"), allow_nan=False)
print("parity cases", len(cases))
