"""
upay Shield - Risk Scoring Model
-----------------------------------
Trains a gradient-boosted classifier to estimate the probability that a
transaction is suspicious, using the behavioral features from features.py.

Note on model choice: LightGBM/XGBoost were not installable in this sandbox
(network-restricted), so this uses scikit-learn's GradientBoostingClassifier,
which is algorithmically the same family (boosted decision trees) and is a
reasonable, fully-explainable drop-in. Swapping to LightGBM later is a
one-line change (see comment at the bottom) if you have an unrestricted
environment (e.g. your own laptop via Claude Code).

Outputs:
  - model.pkl              : trained model (pickle)
  - evaluation.json        : precision/recall/ROC-AUC + confusion matrix
  - feature_importance.json: which signals matter most (for explainability)
"""

import json
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve, average_precision_score
)

from features import FEATURE_COLUMNS

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "..", "data")


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))
    X = df[FEATURE_COLUMNS].fillna(0)
    y = df["is_scam"]

    # keep a clean, untouched test split (rulebook / responsible-AI requirement:
    # "keep a clean test set that is not used to train the model")
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.25, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.8,
        min_samples_leaf=3,
        random_state=42,
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y_test, probs)
    ap = average_precision_score(y_test, probs)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds).tolist()

    # also report recall at a stricter threshold (0.7) -- this is closer to
    # what "high risk" would mean operationally (hold for review)
    preds_strict = (probs >= 0.7).astype(int)
    recall_strict = recall_score(y_test, preds_strict, zero_division=0)
    precision_strict = precision_score(y_test, preds_strict, zero_division=0)

    evaluation = {
        "test_set_size": int(len(y_test)),
        "test_set_positives": int(y_test.sum()),
        "roc_auc": round(float(auc), 4),
        "average_precision": round(float(ap), 4),
        "threshold_0.5": {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
        },
        "threshold_0.7_high_risk": {
            "precision": round(float(precision_strict), 4),
            "recall": round(float(recall_strict), 4),
        },
        "confusion_matrix": {
            "labels": ["actual_normal", "actual_scam"],
            "columns": ["pred_normal", "pred_scam"],
            "matrix": cm,
        },
    }

    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist()))
    importances = dict(sorted(importances.items(), key=lambda kv: -kv[1]))

    with open(os.path.join(DATA_DIR, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(DATA_DIR, "evaluation.json"), "w") as f:
        json.dump(evaluation, f, indent=2)
    with open(os.path.join(DATA_DIR, "feature_importance.json"), "w") as f:
        json.dump(importances, f, indent=2)

    # also score the FULL dataset (train+test) so the dashboard/demo has
    # risk scores for every transaction, not just the held-out test rows
    all_probs = model.predict_proba(X)[:, 1]
    df_scored = df.copy()
    df_scored["risk_score"] = all_probs
    df_scored.to_csv(os.path.join(DATA_DIR, "scored_transactions.csv"), index=False)

    print("=== Evaluation (on held-out test set) ===")
    print(json.dumps(evaluation, indent=2))
    print("\n=== Feature importance ===")
    for k, v in importances.items():
        print(f"  {k:28s} {v:.3f}")
    print(f"\nSaved model.pkl, evaluation.json, feature_importance.json, "
          f"scored_transactions.csv to {DATA_DIR}")

    # --- To use LightGBM instead (recommended once you have full internet,
    # e.g. running this locally via Claude Code), swap the model line for:
    #
    #   import lightgbm as lgb
    #   model = lgb.LGBMClassifier(n_estimators=300, max_depth=5,
    #                               learning_rate=0.05, class_weight="balanced")
    #
    # everything else in this file stays the same.


if __name__ == "__main__":
    main()
