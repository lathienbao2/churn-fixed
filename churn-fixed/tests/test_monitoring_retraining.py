import os

import pandas as pd

from churniq.ab_testing import choose_model, should_auto_promote, should_rollback
from churniq.monitoring import drift_report, population_stability_index, save_drift_report, should_trigger_retrain
from churniq.retraining import should_promote
from datasets_config import DATASET_CONFIGS


def test_psi_non_negative():
    expected = pd.Series([1, 2, 3, 4, 5, 6])
    actual = pd.Series([1, 2, 3, 4, 8, 9])
    assert population_stability_index(expected, actual) >= 0


def test_drift_report_and_save(tmp_path):
    config = DATASET_CONFIGS["telco_ibm"]
    training = pd.DataFrame({"tenure": [1, 2, 3], "MonthlyCharges": [10, 20, 30], "TotalCharges": [10, 40, 90]})
    inference = pd.DataFrame({"tenure": [10, 20, 30], "MonthlyCharges": [100, 120, 130], "TotalCharges": [1000, 2000, 3000]})
    report = drift_report(training, inference, config)
    path = save_drift_report(report, str(tmp_path))
    assert os.path.exists(path)
    assert "psi" in report


def test_retrain_promotion_gate():
    assert should_promote({"test_auc": 0.8}, {"test_auc": 0.79}, tolerance=0.02)
    assert not should_promote({"test_auc": 0.8}, {"test_auc": 0.75}, tolerance=0.02)


def test_retrain_trigger_conditions():
    assert should_trigger_retrain(500, {"max_psi": 0.0})
    assert should_trigger_retrain(10, {"max_psi": 0.26})
    assert not should_trigger_retrain(10, {"max_psi": 0.1})


def test_ab_testing_deterministic():
    first = choose_model("customer-123")
    second = choose_model("customer-123")
    assert first == second
    assert should_auto_promote(7, True, 0.01)
    assert should_rollback(0.06)
