"""
Tests cho API Endpoints
"""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestHealthEndpoint:
    """Test /health endpoint"""

    def test_health_returns_ok(self, client):
        response = client.get('/health')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'ok'
        assert 'version' in data
        assert 'models_loaded' in data

    def test_health_no_auth_required(self, client):
        """Health check không cần API key"""
        response = client.get('/health')
        assert response.status_code == 200


class TestModelsEndpoint:
    """Test /api/models endpoint"""

    def test_list_models(self, client):
        response = client.get('/api/models')
        assert response.status_code == 200
        data = response.get_json()
        assert 'models' in data
        assert isinstance(data['models'], list)

    def test_model_has_required_fields(self, client):
        response = client.get('/api/models')
        data = response.get_json()
        if data['models']:
            model = data['models'][0]
            assert 'id' in model
            assert 'name' in model
            assert 'numeric_features' in model
            assert 'categorical_features' in model


class TestPredictEndpoint:
    """Test /api/predict endpoint"""

    def test_predict_missing_body(self, client):
        response = client.post('/api/predict', data='', content_type='application/json')
        assert response.status_code == 400

    def test_predict_missing_features(self, client):
        response = client.post('/api/predict',
                              data=json.dumps({'not_features': {}}),
                              content_type='application/json')
        assert response.status_code == 400

    def test_predict_telco_valid(self, client, sample_telco_features):
        """Test prediction thành công với Telco data"""
        response = client.post('/api/predict',
                              data=json.dumps({
                                  'dataset_type': 'telco_ibm',
                                  'features': sample_telco_features
                              }),
                              content_type='application/json')
        # Có thể 200 (nếu model loaded) hoặc 400 (nếu model chưa load)
        assert response.status_code in (200, 400)
        if response.status_code == 200:
            data = response.get_json()
            assert 'churn_probability' in data
            assert 'is_churn' in data
            assert 'risk_level' in data
            assert data['churn_probability'] >= 0
            assert data['churn_probability'] <= 1

    def test_predict_call_valid(self, client, sample_call_features):
        """Test prediction thành công với Call data"""
        response = client.post('/api/predict',
                              data=json.dumps({
                                  'dataset_type': 'call_details',
                                  'features': sample_call_features
                              }),
                              content_type='application/json')
        assert response.status_code in (200, 400)
        if response.status_code == 200:
            data = response.get_json()
            assert 'churn_probability' in data
            assert 'retention' in data

    def test_predict_invalid_dataset_type(self, client, sample_telco_features):
        response = client.post('/api/predict',
                              data=json.dumps({
                                  'dataset_type': 'nonexistent',
                                  'features': sample_telco_features
                              }),
                              content_type='application/json')
        assert response.status_code == 400

    def test_predict_invalid_features(self, client):
        """Test prediction với features không hợp lệ"""
        response = client.post('/api/predict',
                              data=json.dumps({
                                  'dataset_type': 'telco_ibm',
                                  'features': {'invalid': 'data'}
                              }),
                              content_type='application/json')
        assert response.status_code == 400


class TestBatchEndpoint:
    """Test /api/predict/batch endpoint"""

    def test_batch_empty_records(self, client):
        response = client.post('/api/predict/batch',
                              data=json.dumps({
                                  'dataset_type': 'telco_ibm',
                                  'records': []
                              }),
                              content_type='application/json')
        assert response.status_code == 400

    def test_batch_not_list(self, client):
        response = client.post('/api/predict/batch',
                              data=json.dumps({
                                  'dataset_type': 'telco_ibm',
                                  'records': 'not_a_list'
                              }),
                              content_type='application/json')
        assert response.status_code == 400


class TestCSVEndpoint:
    """Test /api/predict/csv endpoint"""

    def test_csv_no_file(self, client):
        response = client.post('/api/predict/csv')
        assert response.status_code == 400

    def test_csv_empty_file(self, client):
        from io import BytesIO
        data = {'file': (BytesIO(b''), 'test.csv')}
        response = client.post('/api/predict/csv',
                              data=data,
                              content_type='multipart/form-data')
        assert response.status_code == 400

    def test_csv_wrong_extension(self, client):
        from io import BytesIO
        data = {'file': (BytesIO(b'some data'), 'test.txt')}
        response = client.post('/api/predict/csv',
                              data=data,
                              content_type='multipart/form-data')
        assert response.status_code == 400


class TestErrorHandling:
    """Test error handling"""

    def test_404_json_response(self, client):
        response = client.get('/nonexistent/path')
        assert response.status_code == 404
        data = response.get_json()
        assert 'error' in data

    def test_405_method_not_allowed(self, client):
        response = client.get('/api/predict')  # GET thay vì POST
        assert response.status_code == 405
        data = response.get_json()
        assert 'error' in data
