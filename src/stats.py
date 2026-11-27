"""Deal-level descriptive statistics: win rates by factor with Wilson CIs, chi-square tests and
company-adjusted effects (logistic regression with company + source controls)."""
import sys, warnings, numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from common import EVAL_MAX_CREATED

SN = pd.read_parquet("data/out/snaps.parquet")
SN = SN[SN.label.isin([0, 1]) & (SN.created_at <= EVAL_MAX_CREATED)]
# deal-level frame: static fields + whole-life engagement measured at the latest snapshot (activity summary)
last = SN.sort_values("off_h").groupby("deal_id").last()
d1 = SN[SN.off_h == 24].set_index("deal_id")  # state one day in
d7 = SN[SN.off_h == 168].set_index("deal_id")
D = last.copy()
D["won"] = D.label
out = []


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def table(col, frame=D, min_n=30, label=None, order=None):
    f = frame[list(dict.fromkeys([col, "won", "company_id"]))].copy()
    f[col] = f[col].astype(object).where(f[col].notna(), "(missing)")
    g = f.groupby(col).won.agg(["sum", "count"])
    g = g[g["count"] >= min_n]
    if len(g) < 2:
        return
    ct = np.array([g["sum"], g["count"] - g["sum"]])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    # Cramer's V as effect size
    v = np.sqrt(chi2 / ct.sum() / (min(ct.shape) - 1))
    g = g.assign(win=g["sum"] / g["count"])
    g[["lo", "hi"]] = [wilson(k, n) for k, n in zip(g["sum"], g["count"])]
    if order is not None:
        g = g.reindex([o for o in order if o in g.index])
    else:
        g = g.sort_values("win", ascending=False)
    print(f"\n### {label or col}  (chi2 p={p:.2g}, Cramer V={v:.3f}, n={int(g['count'].sum())})")
    print(g[["count", "win", "lo", "hi"]].round(3).to_string())
    out.append(dict(factor=label or col, p=p, cramer_v=v, n=int(g["count"].sum()), levels=len(g)))


def bucket(s, edges, labels):
    return pd.cut(s, edges, labels=labels, right=False).astype(object)


print("deals", len(D), "win rate", D.won.mean().round(3))
for col in ["company_id", "lead_source", "lead_referral_source", "lead_form_name", "lead_imported", "lead_created_by_rep",
            "lead_dow", "lead_is_weekend", "lead_remodal_type", "lead_sink", "lead_backsplash", "lead_kitchen_stove",
            "lead_contact_time", "lead_when_start", "lead_has_message", "lead_has_file", "em_domain", "ph_area_is_company_top",
            "ph_tollfree", "name_words", "cust_prior_deals", "has_company_name"]:
    table(col)
D["hour_b"] = bucket(D.lead_hour, [0, 7, 10, 13, 16, 19, 22, 25], ["0-6", "7-9", "10-12", "13-15", "16-18", "19-21", "22-23"])
table("hour_b", label="lead hour", order=["0-6", "7-9", "10-12", "13-15", "16-18", "19-21", "22-23"])

# speed to lead: first outbound contact (only deals with any linked outbound contact)
D["first_out_b"] = bucket(D.first_out_h, [-1e9, 0.0833, 0.5, 1, 4, 24, 72, 1e9], ["<5m", "5-30m", "30-60m", "1-4h", "4-24h", "1-3d", ">3d"])
table("first_out_b", D[D.first_out_h.notna()], label="speed to first outbound contact", order=["<5m", "5-30m", "30-60m", "1-4h", "4-24h", "1-3d", ">3d"])
D["first_call_b"] = bucket(D.first_call_out_h, [-1e9, 0.0833, 1, 24, 1e9], ["<5m", "5-60m", "1-24h", ">1d"])
table("first_call_b", D[D.first_call_out_h.notna()], label="speed to first call", order=["<5m", "5-60m", "1-24h", ">1d"])
D["replied_b"] = np.where(D.n_in > 0, "customer replied", "never replied")
table("replied_b", D[(D.n_out > 0)], label="customer ever replied (deals we contacted)")
D["rep_lat_b"] = bucket(D.rep_reply_lat_med_h, [0, 0.25, 1, 4, 24, 1e9], ["<15m", "15-60m", "1-4h", "4-24h", ">1d"])
table("rep_lat_b", label="rep median reply time to customer", order=["<15m", "15-60m", "1-4h", "4-24h", ">1d"])
D["out_b"] = bucket(D.n_out, [0, 1, 3, 6, 11, 21, 1e9], ["0", "1-2", "3-5", "6-10", "11-20", "21+"])
table("out_b", D[D.lead_source.isin(["leads"])], label="outbound touches (web leads)", order=["0", "1-2", "3-5", "6-10", "11-20", "21+"])
D["call_talk_b"] = bucket(D.call_talk_sum, [0, 1, 60, 300, 900, 1e9], ["0", "<1m", "1-5m", "5-15m", "15m+"])
table("call_talk_b", D[(D.n_call_out + D.n_call_in) > 0], label="total talk time", order=["0", "<1m", "1-5m", "5-15m", "15m+"])
D["media_b"] = np.where(D.sms_in_media > 0, "sent photo", "no photo")
table("media_b", D[D.n_sms_in > 0], label="customer texted a photo")
D["quote_b"] = np.where(D.n_quote > 0, "quote", "no quote")
table("quote_b", label="has a quote")
D["qsent_b"] = np.where(D.n_quote_sent > 0, np.where((D.n_quote_view + D.n_quote_first_view) > 0, "sent+viewed", "sent, not viewed"), "not emailed")
table("qsent_b", D[D.n_quote > 0], label="quote emailed / viewed")
D["qview_b"] = bucket(D.n_quote_view, [0, 1, 2, 4, 1e9], ["0", "1", "2-3", "4+"])
table("qview_b", D[D.n_quote_sent > 0], label="quote views", order=["0", "1", "2-3", "4+"])
D["price_b"] = bucket(D.q_total_max, [-1e9, 1000, 2500, 4000, 6000, 9000, 1e9], ["<1k", "1-2.5k", "2.5-4k", "4-6k", "6-9k", "9k+"])
table("price_b", D[D.q_total_max > 0], label="quote price", order=["<1k", "1-2.5k", "2.5-4k", "4-6k", "6-9k", "9k+"])
table("q_product", D[D.n_quote > 0], label="quoted material", min_n=40)
D["qopt_b"] = D.q_n_options.clip(upper=3)
table("qopt_b", D[D.n_quote > 0], label="color options on quote (3=3+)")
D["appt_b"] = np.where(D.n_appt_created > 0, "appointment", "none")
table("appt_b", label="appointment scheduled")
D["notes_b"] = bucket(D.n_note, [0, 1, 3, 1e9], ["0", "1-2", "3+"])
table("notes_b", label="notes", order=["0", "1-2", "3+"])
D["dist_b"] = bucket(D.geo_dist_shop_mi, [0, 5, 10, 20, 40, 1e9], ["<5mi", "5-10", "10-20", "20-40", "40+"])
table("dist_b", label="distance to shop (when zip known, any source)", order=["<5mi", "5-10", "10-20", "20-40", "40+", "(missing)"])
D["inc_b"] = pd.qcut(D.geo_income_rel, 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"]).astype(object)
table("inc_b", D[D.geo_income_rel.notna()], label="zip income vs company median (quartiles)")
D["hv_b"] = pd.qcut(D.geo_home_value_rel, 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"]).astype(object)
table("hv_b", D[D.geo_home_value_rel.notna()], label="zip home value vs company median (quartiles)")

# state one day / one week in
for frame, tag in [(d1, "day 1"), (d7, "day 7")]:
    f = frame.copy(); f["won"] = f.label
    f["eng"] = np.where(f.n_in > 0, "replied", np.where(f.n_out > 0, "contacted, no reply", "no contact logged"))
    table("eng", f, label=f"engagement at {tag}")
    f["stg"] = f.stage_now
    table("stg", f, label=f"stage at {tag}")

res = pd.DataFrame(out).sort_values("cramer_v", ascending=False)
res.to_csv("data/out/univariate.csv", index=False)
print("\n## factors ranked by effect size (Cramer V)")
print(res.round(4).to_string(index=False))

# adjusted effects: logistic regression with company + source fixed effects
print("\n## adjusted (logit, company + source FE), odds ratios")
A = D.copy()
A["fast5"] = (A.first_out_h < 0.0833).astype(float).where(A.first_out_h.notna())
A["fast60"] = (A.first_out_h < 1).astype(float).where(A.first_out_h.notna())
A["replied"] = (A.n_in > 0).astype(float)
A["photo"] = (A.sms_in_media > 0).astype(float)
A["quote_viewed"] = ((A.n_quote_view + A.n_quote_first_view) > 0).astype(float)
A["appt"] = (A.n_appt_created > 0).astype(float)
A["log_price"] = np.log1p(A.q_total_max.clip(lower=0))
A["src"] = A.lead_source.astype(str)
for term, sub in [("fast5", A.first_out_h.notna()), ("fast60", A.first_out_h.notna()), ("replied", A.n_out > 0),
                  ("photo", A.n_sms_in > 0), ("quote_viewed", A.n_quote_sent > 0), ("appt", None), ("log_price", A.q_total_max > 0)]:
    f = A if sub is None else A[sub]
    try:
        m = smf.logit(f"won ~ {term} + C(company_id) + C(src)", f).fit(disp=0)
        ci = np.exp(m.conf_int().loc[term])
        print(f"{term:14s} OR={np.exp(m.params[term]):.2f} [{ci[0]:.2f}, {ci[1]:.2f}] p={m.pvalues[term]:.2g} n={int(m.nobs)}")
    except Exception as e:
        print(term, "failed", e)
