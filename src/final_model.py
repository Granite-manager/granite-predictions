"""Tune, calibrate and evaluate the deal-win model on the time holdout. Usage: final_model.py <feature-set-json> [text-tag]"""
import sys, json, time, pickle; sys.path.insert(0, "src")
import optuna, lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
import common
from common import *

fs = json.load(open(sys.argv[1]))  # {"name":..., "features":[...]}
tag = sys.argv[2] if len(sys.argv) > 2 else None
if tag:
    common.set_text(tag)
SN = load(strict=True)
feats = [f for f in fs["features"] if f in SN.columns or f == "txt_score"]
if tag and "txt_score" not in feats:
    feats.append("txt_score")
print(fs["name"], len(feats), "features")

# precompute fold matrices once (target encoding + text stacking), then tune boosting params on them
cust = SN.customer_id.unique()
perm = pd.Series(np.random.default_rng(0).permutation(len(cust)), index=cust)
folds = []
for tri, vai in GroupKFold(5).split(SN, groups=SN.customer_id.map(perm)):
    tr, va = SN.iloc[tri], SN.iloc[vai]
    Xtr, Xva = prepare(tr, va, feats)
    folds.append((Xtr, tr.label.values, tr.w.values, Xva, va.label.values, va.w.values))


def objective(trial):
    p = dict(PARAMS, learning_rate=trial.suggest_float("lr", 0.01, 0.1, log=True),
             num_leaves=trial.suggest_int("num_leaves", 4, 64, log=True),
             min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 20, 300, log=True),
             feature_fraction=trial.suggest_float("feature_fraction", 0.3, 1.0),
             lambda_l2=trial.suggest_float("lambda_l2", 0.1, 50, log=True),
             max_depth=trial.suggest_int("max_depth", 3, 10))
    rounds = trial.suggest_int("rounds", 100, 1500, log=True)
    oof_auc = []
    for Xtr, ytr, wtr, Xva, yva, wva in folds:
        m = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=wtr), num_boost_round=rounds)
        oof_auc.append(roc_auc_score(yva, m.predict(Xva), sample_weight=wva))
    return float(np.mean(oof_auc))


optuna.logging.set_verbosity(optuna.logging.WARNING)
# parallel tuning: several processes share one study via a journal file (WORKER=1..N tune only, WORKER=0 tunes then evaluates)
import os
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
PARAMS["num_threads"] = int(os.environ.get("THREADS", PARAMS["num_threads"]))
storage = JournalStorage(JournalFileBackend(f"data/out/optuna_{fs['name']}.log"))
study = optuna.create_study(study_name=fs["name"], storage=storage, direction="maximize", load_if_exists=True,
                            sampler=optuna.samplers.TPESampler(seed=int(os.environ.get("WORKER", 0))))
if os.environ.get("WORKER", "0") == "0" and len(study.trials) == 0:
    study.enqueue_trial(dict(lr=0.03, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.8, lambda_l2=5.0, max_depth=10, rounds=500))
study.optimize(objective, n_trials=int(fs.get("trials", 60)))
if os.environ.get("WORKER", "0") != "0":
    sys.exit(0)
bp = study.best_params
print("best cv auc", round(study.best_value, 4), bp)
params = dict(PARAMS, learning_rate=bp["lr"], num_leaves=bp["num_leaves"], min_data_in_leaf=bp["min_data_in_leaf"],
              feature_fraction=bp["feature_fraction"], lambda_l2=bp["lambda_l2"], max_depth=bp["max_depth"])

# holdout: train on deals created before HOLDOUT_FROM; calibrate with isotonic on OOF predictions of the train part
tr = SN[SN.created_at < HOLDOUT_FROM].reset_index(drop=True)
va = SN[SN.created_at >= HOLDOUT_FROM].reset_index(drop=True)
oof = np.zeros(len(tr))
cust = tr.customer_id.unique()
perm = pd.Series(np.random.default_rng(1).permutation(len(cust)), index=cust)
for a, b in GroupKFold(5).split(tr, groups=tr.customer_id.map(perm)):
    p_, _, _ = fit_predict(tr.iloc[a], tr.iloc[b], feats, bp["rounds"], params)
    oof[b] = p_
iso = IsotonicRegression(out_of_bounds="clip", y_min=0.005, y_max=0.995).fit(oof, tr.label, sample_weight=tr.w)
t0 = time.time()
raw, model, Xva = fit_predict(tr, va, feats, bp["rounds"], params)
cal = iso.predict(raw)
res = dict(name=fs["name"], n_feats=len(feats), text=tag, cv_auc=study.best_value,
           ho_auc=roc_auc_score(va.label, raw, sample_weight=va.w),
           ho_ll_raw=log_loss(va.label, np.clip(raw, 1e-4, 1 - 1e-4), sample_weight=va.w),
           ho_ll_cal=log_loss(va.label, cal, sample_weight=va.w),
           ho_brier_cal=brier_score_loss(va.label, cal, sample_weight=va.w),
           base_rate=float(np.average(va.label, weights=va.w)))
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}))
va = va.assign(p=cal, raw=raw)
print("\nAUC by snapshot age (hours since lead):")
print(va.groupby("off_h").apply(lambda g: pd.Series(dict(n=len(g), auc=roc_auc_score(g.label, g.p) if g.label.nunique() > 1 else np.nan, win=g.label.mean()))).round(3).to_string())
print("\nAUC by company:")
print(va.groupby("company_id", observed=True).apply(lambda g: pd.Series(dict(n=len(g), auc=roc_auc_score(g.label, g.p) if g.label.nunique() > 1 else np.nan, win=g.label.mean()))).round(3).to_string())
print("\nCalibration (holdout, deal-weighted):")
va["bin"] = pd.cut(va.p, [0, .05, .1, .2, .3, .4, .5, .6, .8, 1.0])
print(va.groupby("bin", observed=True).apply(lambda g: pd.Series(dict(n=len(g), predicted=np.average(g.p, weights=g.w), actual=np.average(g.label, weights=g.w)))).round(3).to_string())
print("\nDecile lift (holdout):")
va["dec"] = pd.qcut(va.p.rank(method="first"), 10, labels=False)
print(va.groupby("dec").apply(lambda g: pd.Series(dict(predicted=g.p.mean(), actual=g.label.mean()))).round(3).to_string())
# verdict thresholds
for thr in [0.3, 0.4, 0.5, 0.6]:
    yes = va.p >= thr
    if yes.sum():
        print(f"verdict 'will buy' at p>={thr}: {yes.mean():.1%} of snapshots, precision {va.label[yes].mean():.2f}, recall {va.label[yes].sum()/va.label.sum():.2f}")
no = va.p < 0.1
print(f"verdict 'won't buy' at p<0.1: {no.mean():.1%} of snapshots, actually bought {va.label[no].mean():.3f}")
# latency of one prediction (tree part only)
x1 = Xva.iloc[[0]]
t0 = time.time()
for _ in range(200):
    model.predict(x1)
print(f"\nmodel trees: {model.num_trees()}, one prediction {1000*(time.time()-t0)/200:.2f} ms, model text size {len(model.model_to_string())/1024:.0f} KB")
va[["sid", "deal_id", "off_h", "label", "p", "raw"]].to_parquet(f"data/out/holdout_pred_{fs['name']}.parquet")
pickle.dump(dict(params=params, rounds=bp["rounds"], feats=feats, res=res), open(f"data/out/final_{fs['name']}.pkl", "wb"))
