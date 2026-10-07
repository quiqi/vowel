"""Data-free reference implementations of the four historical model families."""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import VOICE

CLINICAL = (
    "clinical_age", "clinical_sex_male", "clinical_bmi", "clinical_neck_cm",
    "clinical_tmd_cm", "clinical_mouth_open_cm", "clinical_mallampati_grade",
    "clinical_neck_extension_abnormal", "clinical_snore", "clinical_osahs",
)

SCHEMES = {
    "1": CLINICAL,
    "2": ("age", *VOICE),
    # The historical concatenation retained age and clinical_age as separate columns.
    "3": ("age", *VOICE, *CLINICAL),
    "4": ("age", *VOICE, "clinical_osahs", "clinical_tmd_cm",
          "clinical_mallampati_grade", "clinical_neck_extension_abnormal"),
}
BASE_KINDS = ("rf", "extra", "svm")


def estimator(scheme: str, kind: str, seed: int):
    """Training-fold imputation/scaling and historical model hyperparameters."""
    if kind == "logistic":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1, class_weight="balanced", max_iter=5000, random_state=seed)),
        ])
    if kind == "svm":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", SVC(C=1.2, gamma="scale", kernel="rbf", class_weight="balanced",
                          probability=True, random_state=seed)),
        ])
    if kind not in ("rf", "extra"):
        raise ValueError(kind)
    cls = RandomForestClassifier if kind == "rf" else ExtraTreesClassifier
    trees = 260 if scheme in ("2", "3") else (320 if kind == "rf" else 400)
    base = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", cls(n_estimators=trees, min_samples_leaf=2,
                       class_weight="balanced", random_state=seed, n_jobs=1)),
    ])
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


def _inner_oof(X, y, kind, scheme, seed):
    folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    prob = np.full(len(y), np.nan)
    for fold, (train, valid) in enumerate(folds.split(X, y), start=1):
        model = estimator(scheme, kind, seed + fold)
        model.fit(X.iloc[train], y[train])
        prob[valid] = model.predict_proba(X.iloc[valid])[:, 1]
    return prob


def _threshold(y, prob, metric):
    score = accuracy_score if metric == "accuracy" else lambda a, b: (
        (b[a == 1] == 1).mean() + (b[a == 0] == 0).mean()) / 2
    choices = np.linspace(.15, .85, 71)
    # Stable first-maximum rule, as in the historical threshold scans.
    return float(max(choices, key=lambda t: score(y, (prob >= t).astype(int))))


def _stack_predict(X_train, y_train, X_test, seed):
    members = np.column_stack([
        _inner_oof(X_train, y_train, kind, "4", seed + offset * 97)
        for offset, kind in enumerate(BASE_KINDS, start=1)
    ])
    meta = LogisticRegression(C=1, class_weight="balanced", max_iter=5000, random_state=seed)
    meta.fit(members, y_train)
    # Historical scheme 4 chose its accuracy threshold on meta-fit predictions.
    # This is not an independent meta-level threshold validation set.
    threshold = _threshold(y_train, meta.predict_proba(members)[:, 1], "accuracy")
    test_members = []
    for offset, kind in enumerate(BASE_KINDS, start=1):
        model = estimator("4", kind, seed + offset * 173)
        model.fit(X_train, y_train)
        test_members.append(model.predict_proba(X_test)[:, 1])
    return meta.predict_proba(np.column_stack(test_members))[:, 1], threshold


def evaluate(frame: pd.DataFrame, scheme: str, *, folds: int = 5, seed: int = 1001) -> dict:
    """Aggregate-only patient-level internal cross-validation; no predictions saved."""
    if scheme not in SCHEMES:
        raise ValueError("Scheme must be 1, 2, 3 or 4")
    required = ("patient_id", "target", *SCHEMES[scheme])
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    if frame.patient_id.isna().any() or not frame.patient_id.is_unique:
        raise ValueError("patient_id must be present and unique (one row per person)")
    y = pd.to_numeric(frame.target, errors="raise").to_numpy()
    if not np.isin(y, [0, 1]).all():
        raise ValueError("target must contain only 0 or 1")
    y = y.astype(int)
    smaller_class = min(np.bincount(y, minlength=2))
    if folds < 2 or smaller_class - int(np.ceil(smaller_class / folds)) < 9:
        raise ValueError("Each outer training fold needs at least 9 examples per class")
    X = frame.loc[:, SCHEMES[scheme]].apply(pd.to_numeric, errors="raise")
    if X.notna().sum().min() == 0:
        raise ValueError("Every input column needs at least one observed value")
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    probabilities = np.full(len(y), np.nan)
    predictions = np.zeros(len(y), dtype=int)
    for fold, (train, test) in enumerate(outer.split(X, y), start=1):
        train_X, test_X, train_y = X.iloc[train], X.iloc[test], y[train]
        if min(np.bincount(train_y, minlength=2)) < 9:
            raise ValueError("Training fold requires at least 9 examples per class")
        run_seed = seed + fold * 100
        if scheme == "4":
            prob, threshold = _stack_predict(train_X, train_y, test_X, run_seed)
        else:
            kind = "logistic" if scheme == "1" else "rf"
            inner = _inner_oof(train_X, train_y, kind, scheme, run_seed + 11)
            threshold = _threshold(train_y, inner, "balanced")
            model = estimator(scheme, kind, run_seed + 29)
            model.fit(train_X, train_y)
            prob = model.predict_proba(test_X)[:, 1]
        probabilities[test] = prob
        predictions[test] = prob >= threshold
    tn, fp, fn, tp = map(int, confusion_matrix(y, predictions, labels=[0, 1]).ravel())
    return {
        "scheme": scheme, "n": int(len(y)), "events": int(y.sum()), "folds": folds, "seed": seed,
        "auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "accuracy": float(accuracy_score(y, predictions)),
        "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
