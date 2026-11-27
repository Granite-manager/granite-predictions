"""Stack the GRU OOF logit (data/out/seq_logit.parquet) into strict LightGBM, time holdout."""
import sys, json; sys.path.insert(0, "src")
from common import *
SN = load(strict=True).merge(pd.read_parquet("data/out/seq_logit.parquet"), on="sid")
G = groups(SN); feats = sorted(set(sum([v for k, v in G.items() if k != "text"], [])))
ref = holdout_score(SN, feats)[0]; stk = holdout_score(SN, feats + ["seq_logit"])[0]
res = json.load(open("data/out/exp_seq.json"))
res.update(lgb_ref_ho_auc=ref["auc"], lgb_ref_ho_ll=ref["ll"], lgb_stack_ho_auc=stk["auc"], lgb_stack_ho_ll=stk["ll"])
json.dump(res, open("data/out/exp_seq.json", "w")); print(json.dumps(res))
