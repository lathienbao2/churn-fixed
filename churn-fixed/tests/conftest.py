"""
conftest.py — Shared fixtures cho pytest
"""

import sys
import os
import pytest

# Đảm bảo import được project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def app():
    """Flask test app"""
    from api import app as flask_app
    flask_app.config['TESTING'] = True
    return flask_app


@pytest.fixture
def client(app):
    """Flask test client"""
    return app.test_client()


@pytest.fixture
def sample_telco_features():
    """Dữ liệu mẫu Telco IBM"""
    return {
        'tenure': 24,
        'MonthlyCharges': 65.0,
        'TotalCharges': 1500.0,
        'Contract': 'Month-to-month',
        'PaymentMethod': 'Electronic check',
        'InternetService': 'Fiber optic',
        'TechSupport': 'No',
        'OnlineSecurity': 'No'
    }


@pytest.fixture
def sample_call_features():
    """Dữ liệu mẫu Call Details"""
    return {
        'Account length': 100,
        'Total day minutes': 200.0,
        'Total eve minutes': 180.0,
        'Total night minutes': 150.0,
        'Total intl minutes': 10.0,
        'Number vmail messages': 5,
        'Customer service calls': 2,
        'International plan': 'No',
        'Voice mail plan': 'No'
    }


@pytest.fixture
def telco_config():
    """Config cho Telco dataset"""
    from datasets_config import DATASET_CONFIGS
    return DATASET_CONFIGS['telco_ibm']


@pytest.fixture
def call_config():
    """Config cho Call Details dataset"""
    from datasets_config import DATASET_CONFIGS
    return DATASET_CONFIGS['call_details']
