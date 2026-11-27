"""Shared data loading, feature groups, target encoding and CV for experiments."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

EVAL_MAX_CREATED = pd.Timestamp("2026-07-24")  # deals younger than 120 days at cutoff are censored
HOLDOUT_FROM = pd.Timestamp("2026-04-01")

HIGH_CARD = ["geo_zip", "geo_zip3", "geo_city", "geo_county", "ph_area", "ph_exchange", "lead_campaign", "lead_adset",
             "lead_ad", "q_color", "user_id", "lead_desc_how_found"]
LOW_CARD = ["company_id", "lead_source", "lead_referral_source", "lead_form_name", "lead_remodal_type", "lead_when_start",
            "lead_sink", "lead_backsplash", "lead_kitchen_stove", "lead_remove_and_dispose", "lead_contact_time",
            "lead_customer_temperature", "lead_state", "lead_desc_platform", "lead_desc_formtype", "lead_desc_projtype",
            "em_domain", "q_product", "stage_now", "geo_zip_state"]


# address fields have no history and reps add them while selling; only the website form sends them at creation
GEO_AT_CREATION_FORMS = {"wordpress-form"}
TIMING_UNSAFE_PREFIXES = ("geo_",)


def load(text_feats=None, strict=True, company=False):
    SN = pd.read_parquet("data/out/snaps.parquet")
    if company:
        SN = SN.merge(pd.read_parquet("data/out/company_feats.parquet"), on="sid", how="left")
    if strict:
        # contact/address fields have no history: trust them only where they are captured when the lead is created
        late = ~SN.lead_form_name.isin(GEO_AT_CREATION_FORMS)
        for c in [c for c in SN.columns if c.startswith(TIMING_UNSAFE_PREFIXES)]:
            SN.loc[late, c] = np.nan
        # quote revisions before 2026-10-03 were migrated in their final (post-sale) state: content leaks the outcome
        SN = SN.drop(columns=["ph_has2", "has_company_name", "disc"] + [c for c in SN.columns if c.startswith("q_")], errors="ignore")
        # emails have created_at: known only once added
        em_late = SN.em_created_at.isna() | (SN.em_created_at >= SN.t)
        for c in [c for c in SN.columns if c.startswith("em_") and c != "em_created_at"]:
            SN.loc[em_late, c] = np.nan
        SN["em_has"] = (~em_late).astype(int)
        SN = SN.drop(columns=[c for c in SN.columns if c.startswith("lead_desc_")] + ["lead_desc_len"], errors="ignore")
    SN = SN.drop(columns=["em_created_at"], errors="ignore")
    D = pd.read_parquet("data/out/deals_raw.parquet").set_index("id")
    SN["customer_id"] = SN.deal_id.map(D.customer_id)
    SN = SN[SN.label.isin([0, 1]) & (SN.created_at <= EVAL_MAX_CREATED)].reset_index(drop=True)
    SN["w"] = 1.0 / SN.groupby("deal_id").deal_id.transform("size")
    if text_feats is not None:
        SN = SN.merge(text_feats, on="sid", how="left")
    for c in [c for c in LOW_CARD if c in SN.columns]:
        SN[c] = SN[c].astype(str).replace({"nan": np.nan, "None": np.nan}).astype("category")
    for c in [c for c in HIGH_CARD if c in SN.columns]:
        SN[c] = SN[c].astype(str).replace({"nan": np.nan, "None": np.nan, "nan.0": np.nan})
    return SN


def cols(SN, prefixes=(), exact=()):
    out = [c for c in SN.columns if any(c.startswith(p) for p in prefixes)]
    return out + [c for c in exact if c in SN.columns and c not in out]


def groups(SN):
    G = {
        "base": ["age_days", "off_h", "company_id"],
        "lead_form": cols(SN, ["lead_"], ["has_company_name"]),
        "customer_history": cols(SN, ["cust_"]),
        "name": cols(SN, ["name_"]),
        "phone_number": cols(SN, ["ph_"]),
        "email_address": cols(SN, ["em_"]),
        "zip": ["geo_zip"], "zip3": ["geo_zip3"], "city": ["geo_city"], "county": ["geo_county"],
        "zip_demographics": ["geo_pop_density", "geo_population", "geo_home_value", "geo_income", "geo_occupancy",
                             "geo_land_sqmi", "geo_income_rel", "geo_home_value_rel"],
        "distance_latlng": ["geo_dist_shop_mi", "geo_lat", "geo_lng"],
        "address_text": ["geo_has_address", "geo_addr_has_unit", "geo_addr_len"],
        "salesperson": ["user_id", "rep_prior_wr", "rep_prior_n", "rep_open_load"],
        "sms": cols(SN, ["n_sms_", "n7_sms_", "sms_"], ["first_sms_out_h"]),
        "calls": cols(SN, ["n_call_", "n7_call_", "call_"], ["first_call_out_h"]),
        "email_msgs": cols(SN, ["n_email_", "n7_email_"]),
        "engagement": ["first_out_h", "first_in_h", "last_in_ago_d", "last_out_ago_d", "last_any_ago_d", "n_in", "n_out",
                       "in_out_ratio", "n7_in", "n7_out", "replied", "outbound_unanswered_streak", "rep_reply_lat_med_h",
                       "cust_reply_lat_med_h", "cust_reply_lat_min_h", "n_rep_ids", "active_days", "in_active_days",
                       "inbound_hour_mean"],
        "quote_activity": cols(SN, ["n_quote", "n7_quote", "first_quote", "last_quote"]),
        "quote_content": cols(SN, ["q_"], ["disc"]),
        "stage": cols(SN, ["stage_", "n_stage", "n7_stage"]),
        "tasks_notes": cols(SN, ["n_activity", "n7_activity", "act_", "n_note", "n7_note", "note_", "n_appt", "n7_appt",
                                 "appt_", "n_reassigned", "n7_reassigned"]),
        "volume": ["company_leads_30d"],
        "text": cols(SN, ["txt_"]),
        "company_aware": cols(SN, ["co_", "rel_", "pct_"], ["rep_wr_rel"]),
    }
    return {k: [c for c in v if c in SN.columns] for k, v in G.items()}


def target_encode(tr, va, col, y="label", m=30, inner=5, seed=0):
    """Deal-level smoothed target encoding; out-of-fold inside train, train stats for validation."""
    prior = tr.groupby("deal_id")[y].first().mean()
    def fit(df):
        dl = df.groupby("deal_id").agg(k=(col, "first"), y=(y, "first")).dropna()
        s = dl.groupby("k").y.agg(["sum", "count"])
        return (s["sum"] + m * prior) / (s["count"] + m), s["count"]
    enc, cnt = fit(tr)
    va_enc = va[col].map(enc).astype(float).fillna(prior)
    va_cnt = va[col].map(cnt).astype(float).fillna(0)
    tr_enc = pd.Series(prior, index=tr.index, dtype=float)
    tr_cnt = pd.Series(0.0, index=tr.index)
    deals = tr.deal_id.unique()
    rng = np.random.default_rng(seed)
    fold_of = pd.Series(rng.integers(0, inner, len(deals)), index=deals)
    f = tr.deal_id.map(fold_of)
    for k in range(inner):
        e, c = fit(tr[f != k])
        idx = f == k
        tr_enc[idx] = tr.loc[idx, col].map(e).astype(float).fillna(prior)
        tr_cnt[idx] = tr.loc[idx, col].map(c).astype(float).fillna(0)
    return tr_enc, va_enc, tr_cnt, va_cnt


PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=12, cat_smooth=20,
              min_data_per_group=50)


def prepare(tr, va, feats):
    Xtr, Xva = tr[[f for f in feats if f not in HIGH_CARD]].copy(), va[[f for f in feats if f not in HIGH_CARD]].copy()
    for c in [f for f in feats if f in HIGH_CARD]:
        a, b, ca, cb = target_encode(tr, va, c)
        Xtr[c + "_te"], Xva[c + "_te"] = a.values, b.values
        Xtr[c + "_cnt"], Xva[c + "_cnt"] = ca.values, cb.values
    return Xtr, Xva


def fit_predict(tr, va, feats, rounds=500, params=None):
    Xtr, Xva = prepare(tr, va, feats)
    p = dict(PARAMS, **(params or {}))
    m = lgb.train(p, lgb.Dataset(Xtr, tr.label, weight=tr.w), num_boost_round=rounds)
    return m.predict(Xva), m, Xva


def cv_score(SN, feats, n_splits=5, seeds=(0, 1), rounds=500, params=None, return_oof=False):
    aucs, lls = [], []
    oof_all = []
    for seed in seeds:
        cust = SN.customer_id.unique()
        rng = np.random.default_rng(seed)
        perm = pd.Series(rng.permutation(len(cust)), index=cust)
        grp = SN.customer_id.map(perm)
        oof = np.zeros(len(SN))
        for tri, vai in GroupKFold(n_splits).split(SN, groups=grp):
            tr, va = SN.iloc[tri], SN.iloc[vai]
            oof[vai], _, _ = fit_predict(tr, va, feats, rounds, params)
        aucs.append(roc_auc_score(SN.label, oof, sample_weight=SN.w))
        lls.append(log_loss(SN.label, np.clip(oof, 1e-4, 1 - 1e-4), sample_weight=SN.w))
        oof_all.append(oof)
    res = dict(auc=np.mean(aucs), auc_sd=np.std(aucs), ll=np.mean(lls))
    if return_oof:
        res["oof"] = np.mean(oof_all, axis=0)
    return res


def holdout_score(SN, feats, rounds=500, params=None):
    tr = SN[SN.created_at < HOLDOUT_FROM]
    va = SN[SN.created_at >= HOLDOUT_FROM]
    p, m, Xva = fit_predict(tr, va, feats, rounds, params)
    return dict(auc=roc_auc_score(va.label, p, sample_weight=va.w),
                ll=log_loss(va.label, np.clip(p, 1e-4, 1 - 1e-4), sample_weight=va.w),
                brier=brier_score_loss(va.label, p, sample_weight=va.w)), p, va, m, Xva


# ---------------------------------------------------------------- text stacking
TEXT = {"M": None, "pos": None, "C": 0.5}


def set_text(tag, C=0.5):
    import scipy.sparse as sp
    if tag == "tfidf":
        TEXT["M"] = sp.load_npz("data/out/txtagg_tfidf.npz").tocsr()
    else:
        TEXT["M"] = np.load(f"data/out/txtagg_{tag}.npy").astype(np.float32)
    sids = pd.read_parquet(f"data/out/txtagg_{tag}_sids.parquet").sid.values
    TEXT["pos"] = pd.Series(np.arange(len(sids)), index=sids)
    TEXT["C"] = C


def _text_model(Mtr, ytr, wtr):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=TEXT["C"], max_iter=400, solver="lbfgs")
    m.fit(Mtr, ytr, sample_weight=wtr * len(wtr) / wtr.sum())
    return m


def text_stack(tr, va, inner=5, seed=0):
    M = TEXT["M"]
    Mtr, Mva = M[TEXT["pos"][tr.sid].values], M[TEXT["pos"][va.sid].values]
    ytr, wtr = tr.label.values, tr.w.values
    va_s = _text_model(Mtr, ytr, wtr).predict_proba(Mva)[:, 1]
    tr_s = np.zeros(len(tr))
    deals = tr.deal_id.unique()
    fold_of = pd.Series(np.random.default_rng(seed).integers(0, inner, len(deals)), index=deals)
    f = tr.deal_id.map(fold_of).values
    for k in range(inner):
        tr_s[f == k] = _text_model(Mtr[f != k], ytr[f != k], wtr[f != k]).predict_proba(Mtr[f == k])[:, 1]
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    return lg(tr_s), lg(va_s)


_prepare_base = prepare


def prepare(tr, va, feats):
    use_text = "txt_score" in feats
    Xtr, Xva = _prepare_base(tr, va, [f for f in feats if f != "txt_score"])
    if use_text:
        Xtr["txt_score"], Xva["txt_score"] = text_stack(tr, va)
    return Xtr, Xva


def loco_score(SN, feats, rounds=500, params=None):
    """Leave-one-company-out: train on every other company, test on the held-out one (a brand-new client)."""
    out = {}
    for cid in sorted(SN.company_id.astype(int).unique()):
        tr, va = SN[SN.company_id.astype(int) != cid], SN[SN.company_id.astype(int) == cid]
        if va.label.nunique() < 2:
            continue
        p, _, _ = fit_predict(tr, va, feats, rounds, params)
        out[cid] = dict(n=va.deal_id.nunique(), auc=roc_auc_score(va.label, p, sample_weight=va.w))
    return out
