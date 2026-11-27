"""Text value: text-only score, base+text, full(strict)+text vs full. Usage: text_exp.py <tag> [C]"""
import sys, json; sys.path.insert(0, "src")
import common
from common import *
tag = sys.argv[1]; C = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
common.set_text(tag, C)
SN = load(strict=True); G = groups(SN); full = sorted(set(sum(G.values(), [])))
for name, f in [("base+text", G["base"] + ["txt_score"]), ("full+text", full + ["txt_score"])]:
    cv = cv_score(SN, f, seeds=(0,)); ho = holdout_score(SN, f)[0]
    print(json.dumps(dict(tag=tag, C=C, cfg=name, cv_auc=round(cv["auc"], 4), ho_auc=round(ho["auc"], 4), ho_ll=round(ho["ll"], 4))), flush=True)
