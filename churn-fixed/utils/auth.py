"""
Authentication & Rate Limiting middleware cho Flask API.
Bao gồm:
  - API Key authentication
  - Token bucket rate limiting (per-IP)
  - Request logging decorator
"""

import time
import functools
import threading
from collections import defaultdict
from flask import request, jsonify, g, current_app

from utils.config_loader import config
from utils.logger import get_logger

logger = get_logger('auth')

try:
    import redis
except ImportError:
    redis = None


# ═════════════════════════════════════════════════════════════════════════════
# API KEY AUTHENTICATION
# ═════════════════════════════════════════════════════════════════════════════

def require_api_key(f):
    """
    Decorator: yêu cầu API key hợp lệ trong header.
    Bỏ qua nếu API_KEY chưa được cấu hình (dev mode).
    """
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        # Nếu chưa cấu hình API key -> bỏ qua (dev mode)
        if not config.API_KEY or config.API_KEY == 'changeme-generate-a-secure-key-here':
            return f(*args, **kwargs)

        # Lấy key từ header
        provided_key = request.headers.get(config.API_KEY_HEADER, '')

        if not provided_key:
            logger.warning("API request without key",
                         extra={'client_ip': _get_client_ip(), 'endpoint': request.path})
            return jsonify({
                'error': 'Unauthorized',
                'message': f'Thiếu API key. Thêm header: {config.API_KEY_HEADER}'
            }), 401

        # Constant-time comparison để chống timing attack
        if not _constant_time_compare(provided_key, config.API_KEY):
            logger.warning("Invalid API key attempt",
                         extra={'client_ip': _get_client_ip(), 'endpoint': request.path})
            return jsonify({
                'error': 'Forbidden',
                'message': 'API key không hợp lệ'
            }), 403

        return f(*args, **kwargs)

    return decorated


def _constant_time_compare(a: str, b: str) -> bool:
    """So sánh constant-time để chống timing attack"""
    if len(a) != len(b):
        # Vẫn cần tốn thời gian tương đương
        result = False
        for x, y in zip(a.ljust(len(b)), b):
            result |= (x != y)
        return not result

    result = 0
    for x, y in zip(a, b):
        result |= ord(x) ^ ord(y)
    return result == 0


# ═════════════════════════════════════════════════════════════════════════════
# RATE LIMITING (Token Bucket per IP)
# ═════════════════════════════════════════════════════════════════════════════

class RateLimiter:
    """
    Token bucket rate limiter.
    Mỗi IP có bucket riêng với:
    - capacity = RATE_LIMIT_BURST
    - refill rate = RATE_LIMIT_PER_MINUTE tokens / 60 giây
    """

    def __init__(self, rate_per_minute: int = 60, burst: int = 10):
        self.rate = rate_per_minute / 60.0  # tokens per second
        self.burst = burst
        self._buckets = defaultdict(lambda: {'tokens': burst, 'last': time.time()})
        self._lock = threading.Lock()

    def allow_request(self, key: str) -> bool:
        """Kiểm tra và consume 1 token. Trả về True nếu cho phép."""
        with self._lock:
            bucket = self._buckets[key]
            now = time.time()

            # Refill tokens
            elapsed = max(0.0, now - bucket['last'])
            bucket['tokens'] = min(self.burst, bucket['tokens'] + elapsed * self.rate)
            bucket['last'] = now

            if bucket['tokens'] >= 1:
                bucket['tokens'] -= 1
                return True
            return False

    def get_retry_after(self, key: str) -> float:
        """Trả về số giây cần chờ"""
        bucket = self._buckets.get(key)
        if bucket and bucket['tokens'] < 1:
            return (1 - bucket['tokens']) / self.rate
        return 0

    def cleanup(self, max_age_seconds: int = 3600):
        """Dọn dẹp buckets cũ để tránh memory leak"""
        with self._lock:
            now = time.time()
            expired = [k for k, v in self._buckets.items()
                      if now - v['last'] > max_age_seconds]
            for k in expired:
                del self._buckets[k]


# Global rate limiter instance
_rate_limiter = RateLimiter(
    rate_per_minute=config.RATE_LIMIT_PER_MINUTE,
    burst=config.RATE_LIMIT_BURST
)

_redis_client = None


def _get_redis_client():
    global _redis_client
    if not config.ENABLE_REDIS_RATE_LIMIT or redis is None:
        return None
    if _redis_client is None:
        try:
            _redis_client = redis.from_url(config.REDIS_URL, socket_timeout=1, socket_connect_timeout=1)
            _redis_client.ping()
        except Exception as exc:
            logger.warning(f"Redis rate limit unavailable, using local limiter: {exc}")
            _redis_client = False
    return _redis_client if _redis_client is not False else None


def _allow_redis_request(key: str) -> bool:
    client = _get_redis_client()
    if client is None:
        return _rate_limiter.allow_request(key)
    bucket_key = f"rate:{key}:{int(time.time() // 60)}"
    count = client.incr(bucket_key)
    if count == 1:
        client.expire(bucket_key, 70)
    return int(count) <= config.RATE_LIMIT_PER_MINUTE


def rate_limit(f):
    """Decorator: áp dụng rate limiting theo IP"""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if current_app.config.get('TESTING'):
            return f(*args, **kwargs)

        client_ip = _get_client_ip()

        api_key = request.headers.get(config.API_KEY_HEADER, "")
        limit_key = api_key or client_ip

        if not _allow_redis_request(limit_key):
            retry_after = _rate_limiter.get_retry_after(client_ip)
            logger.warning("Rate limit exceeded",
                         extra={'client_ip': client_ip, 'endpoint': request.path})
            response = jsonify({
                'error': 'Too Many Requests',
                'message': f'Vượt quá giới hạn {config.RATE_LIMIT_PER_MINUTE} requests/phút',
                'retry_after_seconds': round(retry_after, 1)
            })
            response.status_code = 429
            response.headers['Retry-After'] = str(int(retry_after) + 1)
            return response

        return f(*args, **kwargs)
    return decorated


# ═════════════════════════════════════════════════════════════════════════════
# REQUEST LOGGING
# ═════════════════════════════════════════════════════════════════════════════

def log_request(f):
    """Decorator: log mỗi request với timing"""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        start = time.time()
        g.request_start = start

        logger.info(f"→ {request.method} {request.path}",
                   extra={
                       'client_ip': _get_client_ip(),
                       'method': request.method,
                       'endpoint': request.path
                   })

        response = f(*args, **kwargs)

        duration_ms = (time.time() - start) * 1000
        status = response[1] if isinstance(response, tuple) else 200

        logger.info(f"← {request.method} {request.path} [{status}] {duration_ms:.0f}ms",
                   extra={
                       'client_ip': _get_client_ip(),
                       'method': request.method,
                       'endpoint': request.path,
                       'status_code': status,
                       'duration_ms': round(duration_ms, 2)
                   })

        return response
    return decorated


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _get_client_ip() -> str:
    """Lấy IP client (hỗ trợ proxy)"""
    forwarded = request.headers.get('X-Forwarded-For', '')
    if forwarded:
        return forwarded.split(',')[0].strip()
    return request.remote_addr or '0.0.0.0'
