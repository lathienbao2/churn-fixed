"""Silver/Gold drift monitoring helpers."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


def numeric_baseline(df: pd.DataFrame, numeric_features: list[str], bins: int = 10) -> dict[str, Any]:
    baseline = {}
    for col in numeric_features:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        quantiles = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1))).tolist()
        baseline[col] = {"bins": quantiles, "mean": float(values.mean()), "std": float(values.std(ddof=0))}
    return baseline


def population_stability_index(expected: pd.Series, actual: pd.Series, bins: list[float] | None = None) -> float:
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()
    if expected.empty or actual.empty:
        return 0.0
    if bins is None or len(bins) < 2:
        bins = np.unique(np.quantile(expected, np.linspace(0, 1, 11))).tolist()
    if len(bins) < 2:
        return 0.0
    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)
    expected_pct = np.maximum(expected_counts / max(expected_counts.sum(), 1), 0.0001)
    actual_pct = np.maximum(actual_counts / max(actual_counts.sum(), 1), 0.0001)
    return round(float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))), 4)


def drift_report(training_df: pd.DataFrame, inference_df: pd.DataFrame, config: dict[str, Any], threshold: float = 0.2) -> dict[str, Any]:
    psi = {}
    for col in config["numeric_features"]:
        if col in training_df.columns and col in inference_df.columns:
            psi[col] = population_stability_index(training_df[col], inference_df[col])
    max_psi = max(psi.values()) if psi else 0.0
    return {
        "dataset": config["name"],
        "created_at": datetime.utcnow().isoformat(),
        "psi": psi,
        "max_psi": max_psi,
        "drift_detected": bool(max_psi > threshold),
        "threshold": threshold,
    }


def save_drift_report(report: dict[str, Any], logs_dir: str = "logs") -> str:
    os.makedirs(logs_dir, exist_ok=True)
    date = datetime.utcnow().strftime("%Y%m%d")
    safe_dataset = str(report.get("dataset", "dataset")).replace(" ", "_").lower()
    path = os.path.join(logs_dir, f"drift_{safe_dataset}_{date}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return path


def should_trigger_retrain(new_records: int, report: dict[str, Any], min_records: int = 500, psi_threshold: float = 0.25) -> bool:
    return bool(new_records >= min_records or float(report.get("max_psi", 0.0)) > psi_threshold)
