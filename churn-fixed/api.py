from flask import Flask, request, jsonify, g, Response
from flask_cors import CORS
import pandas as pd
import joblib
import os
import warnings
import time
import uuid
from io import StringIO

warnings.filterwarnings('ignore')

# ─── Import project modules ─────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets_config import DATASET_CONFIGS, detect_dataset_type, get_retention_recommendations
from utils.config_loader import config
from utils.logger import get_logger
from utils.validators import (
    ValidationError, validate_features, validate_file_upload,
    validate_batch_records, sanitize_string
)
from utils.auth import require_api_key, rate_limit, log_request
from utils.error_handlers import (
    register_error_handlers, AppError, BadRequestError,
    DatasetNotFoundError, ModelNotReadyError
)
from churniq.metrics import metrics
from churniq.prediction import (
    feature_importances,
    input_hash,
    model_version,
    predict_batch as shared_predict_batch,
    predict_single as shared_predict_single,
)


logger = get_logger('api')
START_TIME = time.time()

# ─── Flask App ───────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# Giới hạn kích thước request body
app.config['MAX_CONTENT_LENGTH'] = config.MAX_UPLOAD_SIZE_BYTES

# Đăng ký error handlers
register_error_handlers(app)


@app.before_request
def before_request():
    g.request_id = request.headers.get('X-Request-ID') or str(uuid.uuid4())
    g.request_started_at = time.time()


@app.after_request
def after_request(response):
    latency_ms = (time.time() - getattr(g, 'request_started_at', time.time())) * 1000
    response.headers['X-Request-ID'] = getattr(g, 'request_id', '')
    metrics.record(request.path, response.status_code, latency_ms)
    return response


# ─── Load models khi khởi động ───────────────────────────────────────────────
_models_cache = {}


def load_all_models():
    """Load tất cả models từ disk vào cache"""
    global _models_cache
    loaded = 0
    failed = 0

    for key, cfg in DATASET_CONFIGS.items():
        path = cfg['model_path']
        if os.path.exists(path):
            try:
                _models_cache[key] = {
                    'model': joblib.load(path),
                    'config': cfg,
                    'version': model_version(path)
                }
                loaded += 1
                logger.info(f"Model loaded: {key} ({cfg['name']})")
            except Exception as e:
                failed += 1
                logger.error(f"Failed to load model {key}: {e}",
                           extra={'error_type': 'MODEL_LOAD_FAILED'})
        else:
            logger.warning(f"Model file not found: {path}")

    logger.info(f"Models loaded: {loaded} success, {failed} failed")
    return _models_cache


load_all_models()


def _make_prediction(model_entry, input_df):
    """Helper: chạy model và trả về prob + prediction + feature importances"""
    try:
        result = shared_predict_single(model_entry, input_df, threshold=float(getattr(g, 'threshold', 0.5)))
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise AppError(f"Lỗi khi chạy model: {str(e)}", status_code=500)

    return result['churn_probability'], result['is_churn'], feature_importances(model_entry)


def _parse_threshold(value):
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        raise BadRequestError('threshold phải là số trong khoảng 0.3–0.7')
    if threshold < 0.3 or threshold > 0.7:
        raise BadRequestError('threshold phải nằm trong khoảng 0.3–0.7')
    return threshold


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint — không cần auth"""
    return jsonify({
        'status': 'ok',
        'models_loaded': list(_models_cache.keys()),
        'model_versions': {key: value.get('version', 'unknown') for key, value in _models_cache.items()},
        'uptime_seconds': round(time.time() - START_TIME, 1),
        'version': config.APP_VERSION,
        'environment': config.APP_ENV
    })


@app.route('/metrics', methods=['GET'])
def prometheus_metrics():
    return Response(metrics.render_prometheus(), mimetype='text/plain; version=0.0.4')


@app.route('/docs', methods=['GET'])
def docs():
    return Response("""
<!doctype html>
<html lang="vi">
<head><title>ChurnIQ API Docs</title></head>
<body style="font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto;">
<h1>ChurnIQ API</h1>
<p>Swagger-style lightweight documentation for local/Silver deployments.</p>
<h2>GET /health</h2><p>Trả trạng thái, model versions và uptime.</p>
<h2>GET /metrics</h2><p>Prometheus metrics: request count, error count, latency p50/p95/p99.</p>
<h2>POST /api/predict</h2>
<pre>{"dataset_type":"telco_ibm","threshold":0.5,"features":{...}}</pre>
<h2>POST /api/predict/batch</h2>
<pre>{"dataset_type":"telco_ibm","threshold":0.5,"records":[{...}]}</pre>
<h2>POST /api/predict/csv</h2><p>multipart/form-data field <code>file</code>.</p>
</body></html>
""", mimetype='text/html')


@app.route('/api/models', methods=['GET'])
@rate_limit
@log_request
def list_models():
    """Danh sách models có sẵn"""
    result = []
    for key, info in _models_cache.items():
        cfg = info['config']
        result.append({
            'id': key,
            'name': cfg['name'],
            'numeric_features': cfg['numeric_features'],
            'categorical_features': cfg['categorical_features'],
            'categorical_options': cfg.get('categorical_options', {})
        })
    return jsonify({'models': result})


@app.route('/api/predict', methods=['POST'])
@require_api_key
@rate_limit
@log_request
def predict_single():
    """
    Dự đoán đơn lẻ.

    Body JSON:
    {
        "dataset_type": "telco_ibm",   // optional, auto-detect if omitted
        "features": { "tenure": 24, "MonthlyCharges": 65, ... }
    }
    """
    # ── Parse JSON ────────────────────────────────────────────────────────
    try:
        data = request.get_json(force=True)
    except Exception:
        raise BadRequestError('Body phải là JSON hợp lệ')

    if not data or not isinstance(data, dict):
        raise BadRequestError('Body phải là JSON object')

    if 'features' not in data:
        raise BadRequestError('Thiếu trường "features"')

    threshold = _parse_threshold(data.get('threshold', config.DEFAULT_THRESHOLD))
    g.threshold = threshold

    features = data['features']
    if not isinstance(features, dict):
        raise BadRequestError('"features" phải là JSON object')

    # ── Detect dataset type ───────────────────────────────────────────────
    dataset_type = data.get('dataset_type')
    if dataset_type:
        dataset_type = sanitize_string(dataset_type, max_length=50)

    if not dataset_type:
        try:
            tmp_df = pd.DataFrame([features])
            dataset_type = detect_dataset_type(tmp_df)
        except Exception:
            raise BadRequestError('Không thể tự động nhận diện dataset. Hãy truyền "dataset_type".')

    if not dataset_type or dataset_type not in _models_cache:
        raise DatasetNotFoundError(dataset_type or 'unknown')

    model_entry = _models_cache[dataset_type]
    model_config = model_entry['config']

    # ── Rename columns nếu cần ────────────────────────────────────────────
    rename_map = model_config.get('column_rename', {})
    renamed_features = {rename_map.get(k, k): v for k, v in features.items()}

    # ── Validate features ─────────────────────────────────────────────────
    try:
        validated_features, val_warnings = validate_features(renamed_features, model_config)
    except ValidationError as e:
        raise BadRequestError(e.message, errors=e.errors)

    # ── Prediction ────────────────────────────────────────────────────────
    input_df = pd.DataFrame([validated_features])

    # Coerce numeric (sau validation, đảm bảo type đúng)
    for col in model_config['numeric_features']:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

    churn_prob, is_churn, feature_importance = _make_prediction(model_entry, input_df)

    # Recommendations
    recs = get_retention_recommendations(dataset_type, validated_features, churn_prob)

    latency_ms = (time.time() - getattr(g, 'request_started_at', time.time())) * 1000
    logger.info(f"Prediction: {dataset_type} → churn_prob={churn_prob:.4f}",
               extra={
                   'endpoint': '/api/predict',
                   'request_id': g.request_id,
                   'latency_ms': round(latency_ms, 2),
                   'model_version': model_entry.get('version', 'unknown'),
                   'input_hash': input_hash(validated_features)
               })

    return jsonify({
        'dataset_type': dataset_type,
        'churn_probability': round(churn_prob, 4),
        'churn_probability_pct': f"{churn_prob * 100:.1f}%",
        'is_churn': is_churn,
        'threshold': threshold,
        'model_version': model_entry.get('version', 'unknown'),
        'risk_level': 'HIGH' if churn_prob > 0.7 else ('MEDIUM' if churn_prob > 0.4 else 'LOW'),
        'feature_importance': feature_importance,
        'retention': recs
    })


@app.route('/api/predict/batch', methods=['POST'])
@require_api_key
@rate_limit
@log_request
def predict_batch_json():
    """
    Dự đoán hàng loạt từ JSON.

    Body JSON:
    {
        "dataset_type": "telco_ibm",
        "records": [ {...}, {...} ]
    }
    """
    # ── Parse JSON ────────────────────────────────────────────────────────
    try:
        data = request.get_json(force=True)
    except Exception:
        raise BadRequestError('Body phải là JSON hợp lệ')

    if not data or not isinstance(data, dict):
        raise BadRequestError('Body phải là JSON object')

    # ── Validate records ──────────────────────────────────────────────────
    records = data.get('records', [])
    threshold = _parse_threshold(data.get('threshold', config.DEFAULT_THRESHOLD))
    try:
        validate_batch_records(records, max_records=config.MAX_BATCH_RECORDS)
    except ValidationError as e:
        raise BadRequestError(e.message)

    # ── Detect dataset type ───────────────────────────────────────────────
    df = pd.DataFrame(records)
    dataset_type = data.get('dataset_type') or detect_dataset_type(df)

    if not dataset_type or dataset_type not in _models_cache:
        raise DatasetNotFoundError(dataset_type or 'unknown')

    model_entry = _models_cache[dataset_type]
    model_config = model_entry['config']
    model = model_entry['model']

    # ── Prepare data ──────────────────────────────────────────────────────
    rename_map = model_config.get('column_rename', {})
    df = df.rename(columns=rename_map)

    for col in model_config['numeric_features']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'TotalCharges' in df.columns:
        df = df.dropna(subset=['TotalCharges'])

    feature_cols = model_config['numeric_features'] + model_config['categorical_features']
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise BadRequestError(f'Thiếu cột: {missing}')

    # ── Predict ───────────────────────────────────────────────────────────
    try:
        df = shared_predict_batch(model_entry, df[feature_cols], threshold=threshold)
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        raise AppError(f"Lỗi khi chạy batch prediction: {str(e)}")

    df['churn_probability'] = df['Churn_Probability']
    df['is_churn'] = df['Churn_Prediction'].astype(bool)

    total = len(df)
    if total == 0:
        raise BadRequestError('Không còn dòng hợp lệ sau khi làm sạch dữ liệu')
    churn_count = int(df['is_churn'].sum())
    high_risk = int((df['churn_probability'] > 0.75).sum())

    # Tránh chia cho 0
    churn_rate = round(churn_count / total * 100, 2) if total > 0 else 0

    results = df[['churn_probability', 'is_churn']].to_dict(orient='records')

    logger.info(f"Batch prediction: {dataset_type} → {total} records, {churn_count} churn",
               extra={'endpoint': '/api/predict/batch'})

    return jsonify({
        'dataset_type': dataset_type,
        'total_records': total,
        'churn_count': churn_count,
        'churn_rate_pct': churn_rate,
        'high_risk_count': high_risk,
        'avg_churn_probability': round(float(df['churn_probability'].mean()), 4),
        'threshold': threshold,
        'model_version': model_entry.get('version', 'unknown'),
        'results': results
    })


@app.route('/api/predict/csv', methods=['POST'])
@require_api_key
@rate_limit
@log_request
def predict_batch_csv():
    """Upload file CSV — multipart/form-data field: file"""

    # ── Validate file upload ──────────────────────────────────────────────
    if 'file' not in request.files:
        raise BadRequestError('Thiếu file CSV trong form-data (field: file)')

    file = request.files['file']

    try:
        content, file_warnings = validate_file_upload(
            file,
            max_size_bytes=config.MAX_UPLOAD_SIZE_BYTES,
            allowed_extensions=config.ALLOWED_EXTENSIONS
        )
    except ValidationError as e:
        raise BadRequestError(e.message)

    # ── Parse CSV ─────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(StringIO(content))
    except Exception as e:
        raise BadRequestError(f'Không thể parse CSV: {str(e)}')

    if len(df) == 0:
        raise BadRequestError('File CSV rỗng (không có dữ liệu)')

    if len(df) > config.MAX_BATCH_RECORDS:
        raise BadRequestError(
            f'File quá nhiều dòng: {len(df)}. Giới hạn: {config.MAX_BATCH_RECORDS}'
        )

    # ── Detect dataset type ───────────────────────────────────────────────
    dataset_type = detect_dataset_type(df)
    if not dataset_type or dataset_type not in _models_cache:
        raise DatasetNotFoundError(dataset_type or 'unknown')

    model_entry = _models_cache[dataset_type]
    model_config = model_entry['config']
    model = model_entry['model']

    # ── Prepare data ──────────────────────────────────────────────────────
    rename_map = model_config.get('column_rename', {})
    df = df.rename(columns=rename_map)

    for col in model_config['numeric_features']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'TotalCharges' in df.columns:
        df = df.dropna(subset=['TotalCharges'])

    feature_cols = model_config['numeric_features'] + model_config['categorical_features']
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise BadRequestError(f'File CSV thiếu cột: {missing}')

    # ── Predict ───────────────────────────────────────────────────────────
    try:
        df = shared_predict_batch(model_entry, df[feature_cols], threshold=config.DEFAULT_THRESHOLD)
    except Exception as e:
        logger.error(f"CSV prediction failed: {e}", exc_info=True)
        raise AppError(f"Lỗi khi chạy prediction từ CSV: {str(e)}")

    total = len(df)
    if total == 0:
        raise BadRequestError('Không còn dòng hợp lệ sau khi làm sạch dữ liệu')
    churn_count = int(df['Churn_Prediction'].sum())
    churn_rate = round(churn_count / total * 100, 2) if total > 0 else 0

    logger.info(f"CSV prediction: {dataset_type} → {total} records, {churn_count} churn",
               extra={'endpoint': '/api/predict/csv'})

    return jsonify({
        'dataset_type': dataset_type,
        'total_records': total,
        'churn_count': churn_count,
        'churn_rate_pct': churn_rate,
        'high_risk_count': int((df['Churn_Probability'] > 0.75).sum()),
        'avg_churn_probability': round(float(df['Churn_Probability'].mean()), 4),
        'threshold': config.DEFAULT_THRESHOLD,
        'model_version': model_entry.get('version', 'unknown'),
        'results': df[['Churn_Probability', 'Churn_Prediction']].round(4).to_dict(orient='records'),
        'warnings': file_warnings
    })


# ═════════════════════════════════════════════════════════════════════════════
# RUN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    logger.info(f"Starting ChurnIQ API v{config.APP_VERSION}")
    logger.info(f"Environment: {config.APP_ENV}")
    logger.info(f"Models loaded: {list(_models_cache.keys())}")
    logger.info(f"Rate limit: {config.RATE_LIMIT_PER_MINUTE} req/min")
    logger.info(f"Auth: {'ENABLED' if config.API_KEY and config.API_KEY != 'changeme-generate-a-secure-key-here' else 'DISABLED (dev mode)'}")

    print(f"\n🚀 ChurnIQ API v{config.APP_VERSION}")
    print(f"   http://{config.API_HOST}:{config.API_PORT}")
    print(f"   Models: {list(_models_cache.keys())}")
    print(f"   Auth: {'ENABLED' if config.API_KEY and config.API_KEY != 'changeme-generate-a-secure-key-here' else 'DISABLED'}\n")

    app.run(
        host=config.API_HOST,
        port=config.API_PORT,
        debug=config.APP_DEBUG
    )
