"""
Tests cho datasets_config.py
"""

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datasets_config import DATASET_CONFIGS, detect_dataset_type, get_retention_recommendations


class TestDatasetConfigs:
    """Test DATASET_CONFIGS structure"""

    def test_configs_exist(self):
        assert 'telco_ibm' in DATASET_CONFIGS
        assert 'call_details' in DATASET_CONFIGS

    def test_required_fields(self):
        for key, cfg in DATASET_CONFIGS.items():
            assert 'name' in cfg, f"{key} thiếu 'name'"
            assert 'numeric_features' in cfg, f"{key} thiếu 'numeric_features'"
            assert 'categorical_features' in cfg, f"{key} thiếu 'categorical_features'"
            assert 'model_path' in cfg, f"{key} thiếu 'model_path'"

    def test_model_paths_end_with_pkl(self):
        for key, cfg in DATASET_CONFIGS.items():
            assert cfg['model_path'].endswith('.pkl'), f"{key} model_path không phải .pkl"


class TestDetectDatasetType:
    """Test auto-detection của dataset type"""

    def test_detect_telco(self):
        df = pd.DataFrame({'tenure': [1], 'MonthlyCharges': [50]})
        assert detect_dataset_type(df) == 'telco_ibm'

    def test_detect_call_details(self):
        df = pd.DataFrame({'Account length': [100], 'Total day minutes': [200]})
        assert detect_dataset_type(df) == 'call_details'

    def test_detect_call_renamed(self):
        """Detect sau khi cần rename"""
        df = pd.DataFrame({'AccountLength': [100], 'DayMins': [200]})
        assert detect_dataset_type(df) == 'call_details'

    def test_detect_unknown(self):
        df = pd.DataFrame({'random_col': [1]})
        assert detect_dataset_type(df) is None


class TestRetentionRecommendations:
    """Test recommendation generation"""

    def test_telco_high_risk(self):
        features = {
            'Contract': 'Month-to-month',
            'tenure': 5,
            'MonthlyCharges': 90,
            'OnlineSecurity': 'No',
            'TechSupport': 'No',
            'InternetService': 'Fiber optic'
        }
        result = get_retention_recommendations('telco_ibm', features, 0.85)
        assert result['priority'] == 'high'
        assert len(result['recommendations']) > 0

    def test_telco_low_risk(self):
        features = {
            'Contract': 'Two year',
            'tenure': 50,
            'MonthlyCharges': 40,
            'OnlineSecurity': 'Yes',
            'TechSupport': 'Yes',
            'InternetService': 'DSL'
        }
        result = get_retention_recommendations('telco_ibm', features, 0.1)
        assert result['priority'] == 'low'

    def test_call_high_service_calls(self):
        features = {'Customer service calls': 5, 'Total day minutes': 100,
                    'International plan': 'No', 'Total intl minutes': 5}
        result = get_retention_recommendations('call_details', features, 0.8)
        assert result['priority'] == 'high'
        # Phải có recommendation về chăm sóc khách hàng
        actions = [r['action'] for r in result['recommendations']]
        assert any('CSKH' in a or 'chăm sóc' in a.lower() or 'Chăm sóc' in a for a in actions)

    def test_always_has_recommendations(self):
        """Luôn trả về ít nhất 1 recommendation"""
        result = get_retention_recommendations('unknown_type', {}, 0.5)
        assert len(result['recommendations']) >= 1

    def test_recommendation_has_required_fields(self):
        features = {'Contract': 'Month-to-month', 'tenure': 5,
                    'MonthlyCharges': 90, 'OnlineSecurity': 'No',
                    'TechSupport': 'No', 'InternetService': 'Fiber optic'}
        result = get_retention_recommendations('telco_ibm', features, 0.8)
        for rec in result['recommendations']:
            assert 'action' in rec
            assert 'detail' in rec
            assert 'impact' in rec
            assert 'effort' in rec
