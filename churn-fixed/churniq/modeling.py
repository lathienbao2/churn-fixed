"""Model training/evaluation helpers shared by Bronze, Silver and Gold."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    preprocessor = ColumnTransformer([("num", numeric, numeric_features), ("cat", categorical, categorical_features)])
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def churn_class_index(model: Any) -> int:
    classes = list(getattr(model, "classes_", []))
    if not classes and hasattr(model, "named_steps"):
        classes = list(getattr(model.named_steps.get("classifier"), "classes_", []))
    for label in (1, "1", "Yes", "yes", "True.", "true.", "True", "true"):
        if label in classes:
            return classes.index(label)
    return 1 if len(classes) > 1 else 0


def predict_with_threshold(model: Any, X, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    proba = model.predict_proba(X)
    idx = churn_class_index(model)
    churn_prob = proba[:, idx]
    pred = (churn_prob >= threshold).astype(int)
    return churn_prob, pred


def evaluate_thresholds(y_true, y_prob, thresholds: list[float] | None = None) -> dict[str, dict[str, float]]:
    thresholds = thresholds or [0.3, 0.4, 0.5, 0.6, 0.7]
    result = {}
    for threshold in thresholds:
        pred = (y_prob >= threshold).astype(int)
        result[f"{threshold:.2f}"] = {
            "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        }
    return result


def evaluate_model(model: Pipeline, X_train, X_test, y_train, y_test, cv_folds: int = 5) -> tuple[Pipeline, dict[str, Any]]:
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv = cross_validate(
        model,
        X_train,
        y_train,
        cv=skf,
        scoring={"auc": "roc_auc", "f1": "f1", "accuracy": "accuracy"},
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_prob, y_pred = predict_with_threshold(model, X_test, threshold=0.5)

    metrics = {
        "cv_auc_mean": round(float(cv["test_auc"].mean()), 4),
        "cv_auc_std": round(float(cv["test_auc"].std()), 4),
        "cv_f1_mean": round(float(cv["test_f1"].mean()), 4),
        "cv_f1_std": round(float(cv["test_f1"].std()), 4),
        "cv_accuracy_mean": round(float(cv["test_accuracy"].mean()), 4),
        "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "test_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "test_precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "test_recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "test_f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, target_names=["Ở lại (0)", "Rời đi (1)"], zero_division=0),
        "threshold_metrics": evaluate_thresholds(y_test, y_prob),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }
    return model, metrics


def metric_explanations_vi() -> dict[str, str]:
    return {
        "Precision": "Trong các khách hàng bị dự đoán rời đi, tỷ lệ thật sự rời đi.",
        "Recall": "Trong các khách hàng thật sự rời đi, tỷ lệ model phát hiện được.",
        "F1": "Trung bình điều hòa giữa Precision và Recall, hữu ích khi dữ liệu lệch lớp.",
        "AUC": "Khả năng phân biệt khách hàng rời đi và ở lại trên mọi ngưỡng.",
    }
