"""Train several models with patient-grouped cross-validation and save the best one.

Usage:  python src/train.py --data data/diabetic_data.csv [--fast]
"""
import argparse, json, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, brier_score_loss)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import (CATEGORICAL, NUMERIC, clean_training_frame, load_raw,
                      make_target, prepare)

SEED = 42


def preprocessor(scale):
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("num", num, NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100,
                              sparse_output=False), CATEGORICAL),
    ])


def get_models(pos_weight):
    models = {
        "Logistic Regression": Pipeline([("prep", preprocessor(True)),
            ("clf", LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced"))]),
        "Random Forest": Pipeline([("prep", preprocessor(False)),
            ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=-1,
                                           class_weight="balanced_subsample", random_state=SEED))]),
        "HistGradientBoosting": Pipeline([("prep", preprocessor(False)),
            ("clf", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300,
                                                   max_depth=4, l2_regularization=1.0,
                                                   class_weight="balanced", random_state=SEED))]),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = Pipeline([("prep", preprocessor(False)),
            ("clf", XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=4,
                                  subsample=0.8, colsample_bytree=0.8,
                                  scale_pos_weight=pos_weight, eval_metric="aucpr",
                                  n_jobs=-1, random_state=SEED))])
    except ImportError:
        print("xgboost not installed -> skipping XGBoost (pip install xgboost)")
    return models


def pick_threshold(y, p, min_recall=0.60):
    """Highest threshold that still catches >= min_recall of readmissions."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= min_recall)[0]
    return float(thr[ok[-1]]) if len(ok) else 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/diabetic_data.csv")
    ap.add_argument("--out", default="models")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-recall", type=float, default=0.60)
    ap.add_argument("--fast", action="store_true", help="20%% sample for a quick test run")
    a = ap.parse_args()

    raw = clean_training_frame(load_raw(a.data))
    if a.fast:
        raw = raw.sample(frac=0.2, random_state=SEED).reset_index(drop=True)
    X, y, groups = prepare(raw), make_target(raw), raw["patient_nbr"]
    print(f"rows={len(X):,}  patients={groups.nunique():,}  positive rate={y.mean():.3f}")

    # 1) hold-out test set, split BY PATIENT so no patient appears in both sets
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(X, y, groups))
    Xtr, Xte, ytr, yte, gtr = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te], groups.iloc[tr]

    # 2) grouped K-fold CV on the training part for every model
    cv = GroupKFold(n_splits=a.folds)
    pos_weight = (ytr == 0).sum() / (ytr == 1).sum()
    rows, oof = [], {}
    for name, model in get_models(pos_weight).items():
        t0 = time.time()
        p = cross_val_predict(model, Xtr, ytr, groups=gtr, cv=cv, method="predict_proba")[:, 1]
        oof[name] = p
        fold_auc = [roc_auc_score(ytr.iloc[i], p[i]) for _, i in cv.split(Xtr, ytr, gtr)]
        rows.append({"model": name, "cv_roc_auc": np.mean(fold_auc), "cv_roc_auc_std": np.std(fold_auc),
                     "cv_pr_auc": average_precision_score(ytr, p), "seconds": time.time() - t0})
        print(f"{name:22s} ROC-AUC {rows[-1]['cv_roc_auc']:.3f} ± {rows[-1]['cv_roc_auc_std']:.3f}"
              f"  PR-AUC {rows[-1]['cv_pr_auc']:.3f}  ({rows[-1]['seconds']:.0f}s)")
    res = pd.DataFrame(rows).sort_values("cv_pr_auc", ascending=False)
    best = res.iloc[0]["model"]
    print(f"\nBest by CV PR-AUC: {best}")

    # 3) threshold chosen from out-of-fold predictions (recall-first), never from the test set
    thr = pick_threshold(ytr, oof[best], a.min_recall)

    # 4) refit best model on all training data, score ONCE on the untouched test set
    model = get_models(pos_weight)[best].fit(Xtr, ytr)
    pte = model.predict_proba(Xte)[:, 1]
    pred = pte >= thr
    metrics = {
        "best_model": best, "threshold": thr, "target_recall_cv": a.min_recall,
        "test_roc_auc": roc_auc_score(yte, pte), "test_pr_auc": average_precision_score(yte, pte),
        "test_brier": brier_score_loss(yte, pte), "base_rate": float(yte.mean()),
        "test_recall": float((pred & (yte == 1)).sum() / (yte == 1).sum()),
        "test_precision": float((pred & (yte == 1)).sum() / max(pred.sum(), 1)),
        "flagged_share": float(pred.mean()),
    }
    print(json.dumps(metrics, indent=2))

    # 5) final model for users: refit on ALL data, save with everything the app needs
    final = get_models(pos_weight)[best].fit(X, y)
    defaults = {c: (X[c].median() if c in NUMERIC else X[c].mode()[0]) for c in X.columns}
    out = Path(a.out); out.mkdir(exist_ok=True)
    joblib.dump({"pipeline": final, "threshold": thr, "defaults": defaults,
                 "model_name": best, "metrics": metrics,
                 "categories": {c: sorted(X[c].unique().tolist()) for c in CATEGORICAL}},
                out / "readmission_model.joblib", compress=3)
    res.to_csv(out / "cv_results.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved -> {out/'readmission_model.joblib'}")


if __name__ == "__main__":
    main()
