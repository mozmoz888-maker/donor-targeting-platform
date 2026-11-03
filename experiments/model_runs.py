"""Forest runs: RandomForest / ExtraTrees / HistGradientBoosting / LogisticRegression.

Two targets on the 60k send sample (built by churn_analysis.py -> send_frame_sample.parquet):

* donated       - will this send lead to a donation? (the production target)
* unsubscribed  - will this send cause an unsubscribe? (the churn target)

Out-of-time split (holdout = sends from 2026-08-01, like the repo's train.py). Metrics:
ROC AUC, average precision, plus a policy simulation - rank by score, email the top k%,
count donations kept and unsubscribes caused vs BAU (random 75%).

Writes reports/model_runs.json.

    uv run python experiments/model_runs.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from donor_targeting.config import REPORTS_DIR, WAREHOUSE_DIR

HOLDOUT_FROM = "2026-08-01"
TOP_FRACTIONS = (0.75, 0.50, 0.25)

FEATURES = [
    "centroid_cosine_similarity",
    "eng_events_30d",
    "days_since_eng",
    "sends_30d",
    "tenure_days",
    "prior_donations",
    "prior_amount",
]


def models() -> dict[str, object]:
    return {
        "logistic (baseline)": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=20,
            n_jobs=4, random_state=0,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=20,
            n_jobs=4, random_state=0,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31, random_state=0
        ),
    }


def policy_row(y, score: np.ndarray, frac: float) -> dict:
    """Email the top `frac` by score: share of positives reached."""
    y = np.asarray(y, dtype=float)
    k = int(frac * len(score))
    top = np.argsort(-score)[:k]
    return {
        "emailed": frac,
        "kept": float(y[top].sum() / max(y.sum(), 1.0)),
    }


def evaluate(name: str, model, X_tr, y_tr, X_te, y_te, target: str) -> dict:
    model.fit(X_tr, y_tr)
    score = model.predict_proba(X_te)[:, 1]
    out = {
        "model": name,
        "roc_auc": float(roc_auc_score(y_te, score)),
        "average_precision": float(average_precision_score(y_te, score)),
    }
    for frac in TOP_FRACTIONS:
        row = policy_row(y_te, score, frac)
        key = "donations_kept" if target == "donated" else "unsubs_caused"
        out[f"top{frac:.0%}_{key}"] = row["kept"]
    forest = getattr(model, "feature_importances_", None)
    if forest is not None:
        out["feature_importances"] = {
            f: round(float(v), 4) for f, v in zip(FEATURES, forest)
        }
    return out


def main() -> None:
    frame = pd.read_parquet(WAREHOUSE_DIR / "send_frame_sample.parquet")
    frame["was_null_sim"] = frame["centroid_cosine_similarity"].isna().astype(int)
    frame["centroid_cosine_similarity"] = frame["centroid_cosine_similarity"].fillna(-2.0)
    frame["days_since_eng"] = frame["days_since_eng"].fillna(99.0)
    feats = FEATURES + ["was_null_sim"]

    holdout = frame["sent_at"] >= pd.Timestamp(HOLDOUT_FROM, tz="UTC")
    X_tr, X_te = frame.loc[~holdout, feats], frame.loc[holdout, feats]
    print(f"train {len(X_tr):,} rows, holdout {len(X_te):,} rows")

    results = {"holdout_from": HOLDOUT_FROM, "features": feats, "targets": {}}
    for target in ("donated", "unsubscribed"):
        y_tr, y_te = frame.loc[~holdout, target], frame.loc[holdout, target]
        rows = []
        for name, model in models().items():
            row = evaluate(name, model, X_tr, y_tr, X_te, y_te, target)
            rows.append(row)
            print(f"[{target}] {name}: auc={row['roc_auc']:.3f} ap={row['average_precision']:.3f}")
        results["targets"][target] = {
            "base_rate_holdout": float(y_te.mean()),
            "models": rows,
        }

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "model_runs.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {REPORTS_DIR / 'model_runs.json'}")


if __name__ == "__main__":
    main()
