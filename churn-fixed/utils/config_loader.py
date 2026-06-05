"""
Config Loader — Đọc cấu hình từ .env và cung cấp giá trị mặc định an toàn.
Sử dụng:
    from utils.config_loader import config
    port = config.API_PORT
"""

import os
from pathlib import Path

# ─── Tìm và load .env file ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

def _load_env_file(env_path: Path) -> dict:
    """Parse .env file thủ công (không phụ thuộc python-dotenv)"""
    env_vars = {}
    if not env_path.exists():
        return env_vars

    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Bỏ qua comment và dòng trống
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue

            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip()

            # Loại bỏ dấu quote nếu có
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]

            env_vars[key] = value
    return env_vars


# Load .env — biến môi trường OS được ưu tiên cao hơn .env file
_env_file_vars = _load_env_file(BASE_DIR / '.env')


def _get(key: str, default: str = '') -> str:
    """Lấy giá trị: OS env > .env file > default"""
    return os.environ.get(key, _env_file_vars.get(key, default))


def _get_int(key: str, default: int = 0) -> int:
    val = _get(key, str(default))
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _get_float(key: str, default: float = 0.0) -> float:
    val = _get(key, str(default))
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _get_bool(key: str, default: bool = False) -> bool:
    val = _get(key, str(default)).lower()
    return val in ('true', '1', 'yes', 'on')


def _get_list(key: str, default: str = '') -> list:
    val = _get(key, default)
    return [x.strip() for x in val.split(',') if x.strip()]


# ─── Config Object ──────────────────────────────────────────────────────────

class _Config:
    """Immutable config singleton — tất cả cấu hình tập trung ở đây"""

    # Application
    APP_ENV: str          = _get('APP_ENV', 'development')
    APP_DEBUG: bool       = _get_bool('APP_DEBUG', False)
    APP_VERSION: str      = _get('APP_VERSION', '2.1.0')
    BASE_DIR: Path        = BASE_DIR

    # API Server
    API_HOST: str         = _get('API_HOST', '0.0.0.0')
    API_PORT: int         = _get_int('API_PORT', 8001)

    # Authentication
    API_KEY: str          = _get('API_KEY', '')
    API_KEY_HEADER: str   = _get('API_KEY_HEADER', 'X-API-Key')

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = _get_int('RATE_LIMIT_PER_MINUTE', 60)
    RATE_LIMIT_BURST: int      = _get_int('RATE_LIMIT_BURST', 10)

    # File Upload
    MAX_UPLOAD_SIZE_MB: int    = _get_int('MAX_UPLOAD_SIZE_MB', 10)
    MAX_UPLOAD_SIZE_BYTES: int = _get_int('MAX_UPLOAD_SIZE_MB', 10) * 1024 * 1024
    ALLOWED_EXTENSIONS: list   = _get_list('ALLOWED_EXTENSIONS', 'csv')

    # Batch
    MAX_BATCH_RECORDS: int     = _get_int('MAX_BATCH_RECORDS', 5000)

    # Logging
    LOG_LEVEL: str             = _get('LOG_LEVEL', 'INFO')
    LOG_FILE: str              = _get('LOG_FILE', 'logs/churniq.log')
    LOG_MAX_BYTES: int         = _get_int('LOG_MAX_BYTES', 10 * 1024 * 1024)
    LOG_BACKUP_COUNT: int      = _get_int('LOG_BACKUP_COUNT', 5)

    # MLflow
    MLFLOW_URI: str            = _get('MLFLOW_URI', 'http://host.docker.internal:5000')

    # Silver/Gold services
    REDIS_URL: str             = _get('REDIS_URL', 'redis://localhost:6379/0')
    DATABASE_URL: str          = _get('DATABASE_URL', 'postgresql://churniq:churniq@localhost:5432/churniq')
    ENABLE_REDIS_RATE_LIMIT: bool = _get_bool('ENABLE_REDIS_RATE_LIMIT', False)
    DEFAULT_THRESHOLD: float   = _get_float('DEFAULT_THRESHOLD', 0.5)

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == 'production'

    @property
    def is_development(self) -> bool:
        return self.APP_ENV == 'development'

    def __repr__(self):
        safe_key = '***' if self.API_KEY else '(not set)'
        return (
            f"Config(env={self.APP_ENV}, port={self.API_PORT}, "
            f"api_key={safe_key}, rate_limit={self.RATE_LIMIT_PER_MINUTE}/min)"
        )


config = _Config()
