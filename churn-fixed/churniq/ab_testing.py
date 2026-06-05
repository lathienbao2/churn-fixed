"""Gold A/B and shadow traffic helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class ABDecision:
    model_key: str
    bucket: int
    shadow: bool


def stable_bucket(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def choose_model(key: str, production_model: str = "A", candidate_model: str = "B", shadow_percent: int = 20) -> ABDecision:
    bucket = stable_bucket(key)
    shadow = bucket < shadow_percent
    return ABDecision(model_key=candidate_model if shadow else production_model, bucket=bucket, shadow=shadow)


def should_auto_promote(days_observed: int, b_better: bool, error_rate_delta: float) -> bool:
    return days_observed >= 7 and b_better and error_rate_delta <= 0.05


def should_rollback(error_rate_delta: float) -> bool:
    return error_rate_delta > 0.05
