"""
Logger — Structured logging cho ChurnIQ system.
Sử dụng:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Processing request", extra={"user": "abc"})
"""

import logging
import logging.handlers
import os
import json
from datetime import datetime, timezone
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter cho structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Thêm extra fields
        for key in ('request_id', 'client_ip', 'endpoint', 'method',
                     'status_code', 'duration_ms', 'user', 'error_type'):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        # Thêm exception info nếu có
        if record.exc_info and record.exc_info[0] is not None:
            log_entry['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


class PrettyFormatter(logging.Formatter):
    """Human-readable formatter cho console output"""

    COLORS = {
        'DEBUG':    '\033[36m',   # Cyan
        'INFO':     '\033[32m',   # Green
        'WARNING':  '\033[33m',   # Yellow
        'ERROR':    '\033[31m',   # Red
        'CRITICAL': '\033[41m',   # Red background
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        ts = datetime.now().strftime('%H:%M:%S')
        msg = record.getMessage()
        return f"{color}[{ts}] {record.levelname:8s}{self.RESET} | {record.name} | {msg}"


_loggers_cache = {}


def get_logger(name: str = 'churniq') -> logging.Logger:
    """
    Tạo hoặc lấy logger đã có.
    - Console: PrettyFormatter (dev) hoặc JSON (production)
    - File: JSON formatter (luôn luôn)
    """
    if name in _loggers_cache:
        return _loggers_cache[name]

    # Lazy import config để tránh circular
    from utils.config_loader import config

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
    logger.propagate = False

    if not logger.handlers:
        # ── Console handler ──────────────────────────────────────────────
        console = logging.StreamHandler()
        if config.is_production:
            console.setFormatter(JSONFormatter())
        else:
            console.setFormatter(PrettyFormatter())
        console.setLevel(logging.DEBUG)
        logger.addHandler(console)

        # ── File handler (rotating) ──────────────────────────────────────
        try:
            log_path = Path(config.BASE_DIR) / config.LOG_FILE
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                str(log_path),
                maxBytes=config.LOG_MAX_BYTES,
                backupCount=config.LOG_BACKUP_COUNT,
                encoding='utf-8'
            )
            file_handler.setFormatter(JSONFormatter())
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Cannot create log file: {e}")

    _loggers_cache[name] = logger
    return logger
