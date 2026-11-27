# granite-predictions

Machine-learning pipeline for Granite Manager. First model: the chance that an open deal ends in a sale,
for any deal at any point in its life, with a calibrated probability and the main reasons.

Nothing in this repo holds customer data. Everything under `data/` is git-ignored.

## Pipeline

| Step | Script | Output |
|---|---|---|
| Copy tables (read-only session) | `src/extract.py` | `data/raw/*.parquet` |
| Labels, close times, linked events, text units | `src/build.py` | `data/out/{deals_raw,static,events,texts}.parquet` |
| Zip geography (open uszipcode db) | `src/geo.py` | adds `geo_*` to `static.parquet` |
| Point-in-time snapshots | `src/snaps.py` | `data/out/snaps.parquet` |
| Company-relative features | `src/company_feats.py` | `data/out/company_feats.parquet` |
| Text embeddings / per-snapshot text vectors | `src/embed.py`, `src/textagg.py` | `data/out/emb_*`, `txtagg_*` |

```sh
uv sync
export DATABASE_URL='mysql://user:pass@host:3306/db'   # a read replica or the weekly copy
curl -sSL -o data/ext/simple_db.sqlite https://github.com/MacHu-GWU/uszipcode-project/releases/download/1.0.1.db/simple_db.sqlite
uv run python src/extract.py && uv run python src/build.py && uv run python src/geo.py && uv run python src/snaps.py
```

## How the data is framed

- **Label**: deal won = 1; lost = 0; an open deal older than 120 days = 0; younger open deals are censored (not used).
  Deals closed within an hour of creation are records of past sales and are left out.
- **Snapshots**: each deal is scored at 1 h, 1, 3, 7, 14, 30, 60 and 90 days after creation while it is still open.
  Only events with a timestamp before the snapshot are used. Each deal has total weight 1.
- **Evaluation**: 5-fold cross-validation grouped by customer, plus a time holdout
  (train on deals created before 2026-04-01, test on 2026-04-01 .. 2026-07-24). Deal-weighted AUC and log loss.
- **High-cardinality fields** (zip, city, rep, area code, ...) use smoothed target encoding fitted out of fold.

## Leakage rules (`load(strict=True)` in `src/common.py`)

Fields without history that reps fill in during or after the sale are not used at times they were not known:

- address, zip, city and the geography built on them: only for website-form leads (the form sends them at creation);
- quote revision content (price, material, colors) before 2026-10-03: migrated in its final state, excluded;
- deal description, customer details and notes, deal amount, customer company name: excluded;
- customer email: used only after its `customers_emails.created_at`.

## Experiments

| Script | What it measures |
|---|---|
| `src/ablate.py` | each feature group alone on top of a base, and the full model without it |
| `src/geo_dive.py` | location / identity fields vs a within-company shuffled null |
| `src/stats.py` | win rates by factor with Wilson intervals, chi-square, adjusted odds ratios |
| `src/text_exp.py`, `src/words.py` | TF-IDF and embedding text scores (stacked out of fold), top words |
| `src/company_exp.py` | company-relative features, leave-one-company-out |
| `src/exp_seq.py`, `src/exp_seq_stack.py`, `src/exp_catboost.py` | GRU over the event timeline (MPS), CatBoost |
| `src/final_model.py` | Optuna tuning (parallel workers: `WORKER=1..N THREADS=2`), isotonic calibration, holdout report |
| `src/actions.py` | doubly robust (AIPW) effects of rep actions |
| `src/explain.py` | per-deal SHAP reasons |

## First results (2026-11-27)

- LightGBM on 116 clean features: holdout AUC 0.771, CV AUC 0.822, well calibrated after isotonic regression.
  154 trees, about 1 ms per prediction on CPU: fits in a Lambda. A GRU over the event timeline stacked on top gives 0.778.
- Text (TF-IDF, MiniLM, bge-small/base, Qwen3-Embedding-0.6B) adds nothing on top of the structured features.
- Useful: lead source and form fields, quote activity, pipeline stage, customer history, salesperson, engagement.
  Noise: city, zip3, zip demographics, phone area code, name, email address.
- No rep action has a statistically proven effect yet; calls/texts are linked to only about 15% of deals.
