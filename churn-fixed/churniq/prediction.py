"""Prediction helpers with correct churn probability and threshold support."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import joblib
import pandas as pd

from .modeling import predict_with_threshold


def load_model_entry(dataset_key: str, config: dict[str, Any]) -> dict[str, Any] | None:
    path = config["model_path"]
    if not os.path.exists(path):
        return None
    return {"model": joblib.load(path), "config": config, "version": model_version(path)}


def load_models(configs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    loaded = {}
    for key, cfg in configs.items():
        entry = load_model_entry(key, cfg)
        if entry:
            loaded[key] = entry
    return loaded


def model_version(path: str) -> str:
    try:
        stat = os.stat(path)
        raw = f"{os.path.basename(path)}:{int(stat.st_mtime)}:{stat.st_size}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    except OSError:
        return "unknown"


def input_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def prepare_features_df(features: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    rename_map = config.get("column_rename", {})
    renamed = {rename_map.get(k, k): v for k, v in features.items()}
    df = pd.DataFrame([renamed])
    for col in config["numeric_features"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[config["numeric_features"] + config["categorical_features"]]


def predict_single(model_entry: dict[str, Any], features_df: pd.DataFrame, threshold: float = 0.5) -> dict[str, Any]:
    model = model_entry["model"]
    probs, preds = predict_with_threshold(model, features_df, threshold)
    prob = float(probs[0])
    is_churn = bool(preds[0] == 1)
    return {
        "churn_probability": prob,
        "is_churn": is_churn,
        "threshold": float(threshold),
        "risk_level": "HIGH" if prob > 0.7 else ("MEDIUM" if prob > 0.4 else "LOW"),
        "model_version": model_entry.get("version", "unknown"),
    }


def predict_batch(model_entry: dict[str, Any], df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    config = model_entry["config"]
    prepared = df.rename(columns=config.get("column_rename", {})).copy()
    for col in config["numeric_features"]:
        if col in prepared.columns:
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce")
    feature_cols = config["numeric_features"] + config["categorical_features"]
    missing = [c for c in feature_cols if c not in prepared.columns]
    if missing:
        raise ValueError(f"Thiếu cột: {missing}")
    prepared = prepared.dropna(subset=feature_cols)
    probs, preds = predict_with_threshold(model_entry["model"], prepared[feature_cols], threshold)
    prepared["Churn_Probability"] = probs
    prepared["Churn_Prediction"] = preds.astype(int)
    prepared["Model_Version"] = model_entry.get("version", "unknown")
    return prepared


def feature_importances(model_entry: dict[str, Any]) -> dict[str, float]:
    try:
        model = model_entry["model"]
        clf = model.named_steps["classifier"]
        pre = model.named_steps["preprocessor"]
        names = pre.get_feature_names_out()
        importances = clf.feature_importances_
    except Exception:
        return {}

    grouped: dict[str, float] = {}
    for name, importance in zip(names, importances):
        clean = name.split("__", 1)[-1]
        base = clean
        for original in model_entry["config"]["numeric_features"] + model_entry["config"]["categorical_features"]:
            if clean == original or clean.startswith(f"{original}_"):
                base = original
                break
        grouped[base] = grouped.get(base, 0.0) + float(importance)
    return {k: round(v, 4) for k, v in sorted(grouped.items(), key=lambda item: -item[1])}
