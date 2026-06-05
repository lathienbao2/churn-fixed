import os

from churniq.etl import clean_dataset, data_quality_report, prepare_dataset, validate_schema
from datasets_config import DATASET_CONFIGS


BASE_DIR = os.path.join(os.path.dirname(__file__), "..")


def test_prepare_telco_dataset_split():
    prepared = prepare_dataset(
        os.path.join(BASE_DIR, "data", "Telco_customer_churn.csv"),
        DATASET_CONFIGS["telco_ibm"],
    )
    assert len(prepared.X_train) > len(prepared.X_test)
    assert len(prepared.X_train) + len(prepared.X_test) == len(prepared.df)
    assert set(prepared.y_train.unique()).issubset({0, 1})


def test_clean_call_dataset_with_column_rename():
    import pandas as pd

    raw = pd.read_csv(os.path.join(BASE_DIR, "data", "Churn.csv"))
    cleaned = clean_dataset(raw, DATASET_CONFIGS["call_details"], require_target=True)
    validate_schema(cleaned, DATASET_CONFIGS["call_details"], require_target=True)
    assert "Account length" in cleaned.columns
    assert "Total day minutes" in cleaned.columns


def test_quality_report_contains_required_sections():
    prepared = prepare_dataset(
        os.path.join(BASE_DIR, "data", "Churn.csv"),
        DATASET_CONFIGS["call_details"],
    )
    report = data_quality_report(prepared.df, DATASET_CONFIGS["call_details"])
    assert "null_rate" in report
    assert "class_distribution" in report
    assert "outlier_warnings" in report
