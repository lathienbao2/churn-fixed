"""
Error Handlers — Xử lý lỗi tập trung cho Flask API.
Bao gồm:
  - Custom exception classes
  - Flask error handlers
  - Safe error response formatting
"""

import traceback
from flask import jsonify, request
from utils.logger import get_logger
from utils.config_loader import config

logger = get_logger('error_handler')


# ═════════════════════════════════════════════════════════════════════════════
# CUSTOM EXCEPTIONS
# ═════════════════════════════════════════════════════════════════════════════

class AppError(Exception):
    """Base exception cho ứng dụng"""
    def __init__(self, message: str, status_code: int = 500, error_type: str = 'APP_ERROR'):
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        super().__init__(self.message)


class NotFoundError(AppError):
    """Không tìm thấy resource"""
    def __init__(self, message: str = 'Resource không tồn tại'):
        super().__init__(message, status_code=404, error_type='NOT_FOUND')


class BadRequestError(AppError):
    """Request không hợp lệ"""
    def __init__(self, message: str = 'Request không hợp lệ', errors: list = None):
        super().__init__(message, status_code=400, error_type='BAD_REQUEST')
        self.errors = errors or []


class ModelNotReadyError(AppError):
    """Model chưa sẵn sàng"""
    def __init__(self, message: str = 'Model chưa được load'):
        super().__init__(message, status_code=503, error_type='MODEL_NOT_READY')


class DatasetNotFoundError(AppError):
    """Không nhận diện được dataset"""
    def __init__(self, dataset_type: str = ''):
        msg = f'Không nhận diện được dataset hoặc model chưa sẵn sàng. dataset_type={dataset_type}'
        super().__init__(msg, status_code=400, error_type='DATASET_NOT_FOUND')


# ═════════════════════════════════════════════════════════════════════════════
# ERROR RESPONSE BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def _build_error_response(error_type: str, message: str, status_code: int,
                          errors: list = None, trace: str = None):
    """
    Tạo error response JSON chuẩn.
    Chỉ include traceback trong development mode.
    """
    response = {
        'error': error_type,
        'message': message,
        'status_code': status_code,
    }

    if errors:
        response['errors'] = errors

    # Chỉ show traceback trong development
    if trace and config.is_development:
        response['trace'] = trace

    return jsonify(response), status_code


# ═════════════════════════════════════════════════════════════════════════════
# FLASK ERROR HANDLERS — Đăng ký vào app
# ═════════════════════════════════════════════════════════════════════════════

def register_error_handlers(app):
    """Đăng ký tất cả error handlers vào Flask app"""

    @app.errorhandler(AppError)
    def handle_app_error(error):
        logger.error(f"{error.error_type}: {error.message}",
                    extra={
                        'error_type': error.error_type,
                        'endpoint': request.path if request else '',
                        'status_code': error.status_code
                    })
        errors = getattr(error, 'errors', None)
        return _build_error_response(error.error_type, error.message, error.status_code, errors)

    @app.errorhandler(400)
    def handle_400(error):
        return _build_error_response('BAD_REQUEST', 'Request không hợp lệ', 400)

    @app.errorhandler(404)
    def handle_404(error):
        return _build_error_response('NOT_FOUND',
                                     f'Endpoint không tồn tại: {request.path}', 404)

    @app.errorhandler(405)
    def handle_405(error):
        return _build_error_response('METHOD_NOT_ALLOWED',
                                     f'Method {request.method} không được hỗ trợ cho {request.path}', 405)

    @app.errorhandler(413)
    def handle_413(error):
        max_mb = config.MAX_UPLOAD_SIZE_MB
        return _build_error_response('PAYLOAD_TOO_LARGE',
                                     f'Request body quá lớn. Giới hạn: {max_mb}MB', 413)

    @app.errorhandler(429)
    def handle_429(error):
        return _build_error_response('TOO_MANY_REQUESTS',
                                     'Quá nhiều requests. Vui lòng thử lại sau.', 429)

    @app.errorhandler(500)
    def handle_500(error):
        logger.critical(f"Unhandled 500 error: {error}",
                       extra={'endpoint': request.path if request else ''})
        trace = traceback.format_exc()
        return _build_error_response('INTERNAL_ERROR',
                                     'Lỗi hệ thống. Vui lòng thử lại sau.', 500, trace=trace)

    @app.errorhandler(Exception)
    def handle_unhandled(error):
        logger.critical(f"Unhandled exception: {type(error).__name__}: {error}",
                       exc_info=True,
                       extra={
                           'error_type': type(error).__name__,
                           'endpoint': request.path if request else ''
                       })
        trace = traceback.format_exc()
        return _build_error_response(
            'INTERNAL_ERROR',
            'Lỗi không xác định. Vui lòng thử lại sau.',
            500,
            trace=trace
        )
    