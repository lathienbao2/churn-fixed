"""Bronze ETL utilities: CSV load, schema validation, cleaning and reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COL = "Churn"


class SchemaValidationError(ValueError):
    """Raised when a dataset does not match the configured schema."""


@dataclass(frozen=True)
class PreparedDataset:
    df: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    report: dict[str, Any]


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise SchemaValidationError(f"CSV rỗng: {path}")
    return df


def apply_column_rename(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    return df.rename(columns=config.get("column_rename", {}))


def validate_schema(df: pd.DataFrame, config: dict[str, Any], require_target: bool = True) -> None:
    required = set(config["required_columns"])
    if require_target:
        required.add(TARGET_COL)
    missing = sorted(required - set(df.columns))
    if missing:
        guidance = ", ".join(missing)
        raise SchemaValidationError(f"Dataset thiếu cột bắt buộc: {guidance}. Kiểm tra file CSV hoặc mapping column_rename.")


def encode_target(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.strip().str.lower().map(
        {"yes": 1, "no": 0, "true.": 1, "false.": 0, "true": 1, "false": 0, "1": 1, "0": 0}
    )
    return mapped


def clean_dataset(df: pd.DataFrame, config: dict[str, Any], require_target: bool = True) -> pd.DataFrame:
    cleaned = apply_column_rename(df, config).copy()
    validate_schema(cleaned, config, require_target=require_target)

    for col in config["numeric_features"]:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    if require_target:
        cleaned[TARGET_COL] = encode_target(cleaned[TARGET_COL])
        cleaned = cleaned.dropna(subset=[TARGET_COL])
        cleaned[TARGET_COL] = cleaned[TARGET_COL].astype(int)

    feature_cols = config["numeric_features"] + config["categorical_features"]
    cleaned = cleaned.dropna(subset=[c for c in feature_cols if c in cleaned.columns])
    return cleaned


def feature_engineering(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    engineered = df.copy()
    if {"MonthlyCharges", "TotalCharges"}.issubset(engineered.columns):
        tenure = engineered.get("tenure", 0).replace(0, 1)
        engineered["AvgChargesPerTenure"] = engineered["TotalCharges"] / tenure
    if {"Total day minutes", "Total eve minutes", "Total night minutes"}.issubset(engineered.columns):
        engineered["Total domestic minutes"] = (
            engineered["Total day minutes"] + engineered["Total eve minutes"] + engineered["Total night minutes"]
        )
    return engineered


def data_quality_report(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "null_rate": {col: round(float(rate), 4) for col, rate in df.isna().mean().items()},
        "class_distribution": {},
        "outlier_warnings": [],
    }
    if TARGET_COL in df.columns:
        counts = df[TARGET_COL].value_counts(dropna=False).to_dict()
        report["class_distribution"] = {str(k): int(v) for k, v in counts.items()}

    for col in config["numeric_features"]:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((df[col] < lower) | (df[col] > upper)).sum())
        if count:
            report["outlier_warnings"].append({"feature": col, "count": count, "lower": float(lower), "upper": float(upper)})
    return report


def prepare_dataset(path: str, config: dict[str, Any], test_size: float = 0.2, random_state: int = 42) -> PreparedDataset:
    raw = load_csv(path)
    cleaned = clean_dataset(raw, config, require_target=True)
    engineered = feature_engineering(cleaned, config)
    report = data_quality_report(engineered, config)

    feature_cols = config["numeric_features"] + config["categorical_features"]
    X = engineered[feature_cols]
    y = engineered[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return PreparedDataset(engineered, X_train, X_test, y_train, y_test, report)


def print_quality_report(report: dict[str, Any]) -> None:
    print(f"  📄 Rows: {report['rows']:,} | Columns: {report['columns']:,}")
    print(f"  📊 Class distribution: {report.get('class_distribution', {})}")
    high_nulls = {k: v for k, v in report["null_rate"].items() if v > 0}
    print(f"  🧹 Null rates: {high_nulls if high_nulls else 'không có null'}")
    warnings = report.get("outlier_warnings", [])
    if warnings:
        for item in warnings:
            print(f"  ⚠️  Outlier: {item['feature']} có {item['count']} dòng ngoài IQR")
    else:
        print("  ✅ Không phát hiện outlier đáng kể theo IQR")
