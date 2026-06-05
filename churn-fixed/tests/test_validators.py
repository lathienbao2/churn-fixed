"""
Tests cho Input Validators
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.validators import (
    ValidationError,
    sanitize_string,
    validate_numeric,
    validate_categorical,
    validate_features,
    validate_batch_records,
)


class TestSanitizeString:
    """Test string sanitization"""

    def test_basic_strip(self):
        assert sanitize_string("  hello  ") == "hello"

    def test_control_chars_removed(self):
        result = sanitize_string("hello\x00world\x07")
        assert '\x00' not in result
        assert '\x07' not in result
        assert 'helloworld' == result

    def test_max_length(self):
        long_str = "a" * 500
        result = sanitize_string(long_str, max_length=100)
        assert len(result) == 100

    def test_non_string_input(self):
        assert sanitize_string(123) == "123"
        assert sanitize_string(None) == "None"

    def test_empty_string(self):
        assert sanitize_string("") == ""


class TestValidateNumeric:
    """Test numeric validation"""

    def test_valid_int(self):
        assert validate_numeric(42, 'test') == 42.0

    def test_valid_float(self):
        assert validate_numeric(3.14, 'test') == 3.14

    def test_string_number(self):
        assert validate_numeric("65.5", 'test') == 65.5

    def test_invalid_string(self):
        with pytest.raises(ValidationError):
            validate_numeric("abc", 'test')

    def test_none_not_allowed(self):
        with pytest.raises(ValidationError):
            validate_numeric(None, 'test')

    def test_none_allowed(self):
        result = validate_numeric(None, 'test', allow_none=True)
        assert result is None

    def test_min_bound(self):
        with pytest.raises(ValidationError):
            validate_numeric(-5, 'test', min_val=0)

    def test_max_bound(self):
        with pytest.raises(ValidationError):
            validate_numeric(999, 'test', max_val=100)

    def test_within_bounds(self):
        result = validate_numeric(50, 'test', min_val=0, max_val=100)
        assert result == 50.0


class TestValidateCategorical:
    """Test categorical validation"""

    def test_valid_value(self):
        result = validate_categorical('Yes', 'plan', ['Yes', 'No'])
        assert result == 'Yes'

    def test_invalid_value(self):
        with pytest.raises(ValidationError):
            validate_categorical('Maybe', 'plan', ['Yes', 'No'])

    def test_none_value(self):
        with pytest.raises(ValidationError):
            validate_categorical(None, 'plan', ['Yes', 'No'])


class TestValidateFeatures:
    """Test full feature validation"""

    def test_valid_telco_features(self, telco_config, sample_telco_features):
        cleaned, warnings = validate_features(sample_telco_features, telco_config)
        assert 'tenure' in cleaned
        assert 'MonthlyCharges' in cleaned
        assert 'Contract' in cleaned

    def test_valid_call_features(self, call_config, sample_call_features):
        cleaned, warnings = validate_features(sample_call_features, call_config)
        assert 'Account length' in cleaned
        assert 'International plan' in cleaned

    def test_missing_features(self, telco_config):
        with pytest.raises(ValidationError) as exc_info:
            validate_features({'tenure': 24}, telco_config)
        assert 'Thiếu' in str(exc_info.value)

    def test_invalid_numeric_value(self, telco_config, sample_telco_features):
        bad_features = sample_telco_features.copy()
        bad_features['tenure'] = -100  # Ngoài phạm vi
        with pytest.raises(ValidationError):
            validate_features(bad_features, telco_config)

    def test_invalid_categorical_value(self, telco_config, sample_telco_features):
        bad_features = sample_telco_features.copy()
        bad_features['Contract'] = 'NotAContract'
        with pytest.raises(ValidationError):
            validate_features(bad_features, telco_config)


class TestValidateBatchRecords:
    """Test batch records validation"""

    def test_valid_records(self):
        records = [{'a': 1}, {'a': 2}]
        result = validate_batch_records(records, max_records=100)
        assert len(result) == 2

    def test_empty_records(self):
        with pytest.raises(ValidationError):
            validate_batch_records([])

    def test_not_a_list(self):
        with pytest.raises(ValidationError):
            validate_batch_records("not a list")

    def test_too_many_records(self):
        records = [{'a': i} for i in range(101)]
        with pytest.raises(ValidationError):
            validate_batch_records(records, max_records=100)

    def test_non_dict_record(self):
        with pytest.raises(ValidationError):
            validate_batch_records([{'a': 1}, "not a dict"])
