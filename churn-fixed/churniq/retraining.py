"""Gold retraining compare/promote helpers."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from typing import Any


def load_metrics(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def should_promote(old_metrics: dict[str, Any], new_metrics: dict[str, Any], tolerance: float = 0.02) -> bool:
    old_auc = float(old_metrics.get("test_auc", 0.0))
    new_auc = float(new_metrics.get("test_auc", 0.0))
    return new_auc >= old_auc - tolerance


def promote_model(candidate_path: str, production_path: str) -> dict[str, Any]:
    os.makedirs(os.path.dirname(production_path), exist_ok=True)
    backup_path = f"{production_path}.{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.bak"
    if os.path.exists(production_path):
        shutil.copy2(production_path, backup_path)
    shutil.copy2(candidate_path, production_path)
    return {"production_path": production_path, "backup_path": backup_path if os.path.exists(backup_path) else None}


def compare_and_promote(
    candidate_model_path: str,
    production_model_path: str,
    candidate_metrics: dict[str, Any],
    production_metrics_path: str,
    tolerance: float = 0.02,
) -> dict[str, Any]:
    old_metrics = load_metrics(production_metrics_path)
    decision = should_promote(old_metrics, candidate_metrics, tolerance=tolerance)
    result = {
        "decision": "promote" if decision else "rollback",
        "old_auc": old_metrics.get("test_auc"),
        "new_auc": candidate_metrics.get("test_auc"),
        "tolerance": tolerance,
    }
    if decision:
        result.update(promote_model(candidate_model_path, production_model_path))
    return result
