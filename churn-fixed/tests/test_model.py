import os

import joblib
import pandas as pd

from churniq.modeling import predict_with_threshold
from churniq.prediction import load_models, predict_single
from datasets_config import DATASET_CONFIGS


def test_load_pkl_models_if_present():
    models = load_models(DATASET_CONFIGS)
    assert isinstance(models, dict)
    for entry in models.values():
        assert "model" in entry
        assert "version" in entry


def test_prediction_probability_range(sample_telco_features):
    models = load_models(DATASET_CONFIGS)
    if "telco_ibm" not in models:
        return
    config = DATASET_CONFIGS["telco_ibm"]
    df = pd.DataFrame([sample_telco_features])
    result = predict_single(models["telco_ibm"], df[config["numeric_features"] + config["categorical_features"]])
    assert 0 <= result["churn_probability"] <= 1
    assert isinstance(result["is_churn"], bool)


def test_threshold_behavior(sample_telco_features):
    models = load_models(DATASET_CONFIGS)
    if "telco_ibm" not in models:
        return
    config = DATASET_CONFIGS["telco_ibm"]
    df = pd.DataFrame([sample_telco_features])
    low = predict_single(models["telco_ibm"], df[config["numeric_features"] + config["categorical_features"]], threshold=0.3)
    high = predict_single(models["telco_ibm"], df[config["numeric_features"] + config["categorical_features"]], threshold=0.7)
    assert low["threshold"] == 0.3
    assert high["threshold"] == 0.7
