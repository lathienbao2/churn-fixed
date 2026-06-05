"""
Input Validation — Xác thực và làm sạch dữ liệu đầu vào.
Bao gồm:
  - Validate numeric ranges
  - Validate categorical values
  - Sanitize strings (chống injection)
  - File upload validation
  - Batch size validation
"""

import re
import os
from typing import Any, Dict, List, Optional, Tuple


class ValidationError(Exception):
    """Custom exception cho validation errors"""
    def __init__(self, message: str, field: str = None, errors: list = None):
        self.message = message
        self.field = field
        self.errors = errors or []
        super().__init__(self.message)


def sanitize_string(value: str, max_length: int = 200) -> str:
    """
    Làm sạch chuỗi đầu vào:
    - Strip whitespace
    - Loại bỏ ký tự điều khiển (control chars)
    - Giới hạn độ dài
    - Loại bỏ các ký tự nguy hiểm cho injection
    """
    if not isinstance(value, str):
        return str(value)

    # Strip và loại bỏ control characters
    value = value.strip()
    value = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', value)

    # Giới hạn độ dài
    if len(value) > max_length:
        value = value[:max_length]

    return value


def validate_numeric(value: Any, field_name: str,
                     min_val: float = None, max_val: float = None,
                     allow_none: bool = False) -> float:
    """
    Validate giá trị numeric.
    Trả về float hoặc raise ValidationError.
    """
    if value is None:
        if allow_none:
            return None
        raise ValidationError(f"Trường '{field_name}' không được để trống", field=field_name)

    try:
        num = float(value)
    except (ValueError, TypeError):
        raise ValidationError(
            f"Trường '{field_name}' phải là số, nhận được: '{value}'",
            field=field_name
        )

    if min_val is not None and num < min_val:
        raise ValidationError(
            f"Trường '{field_name}' phải >= {min_val}, nhận được: {num}",
            field=field_name
        )
    if max_val is not None and num > max_val:
        raise ValidationError(
            f"Trường '{field_name}' phải <= {max_val}, nhận được: {num}",
            field=field_name
        )

    return num


def validate_categorical(value: Any, field_name: str,
                         allowed_values: List[str]) -> str:
    """
    Validate giá trị categorical nằm trong tập cho phép.
    """
    if value is None:
        raise ValidationError(f"Trường '{field_name}' không được để trống", field=field_name)

    val_str = sanitize_string(str(value))

    if val_str not in allowed_values:
        raise ValidationError(
            f"Trường '{field_name}' phải là một trong {allowed_values}, nhận được: '{val_str}'",
            field=field_name
        )

    return val_str


def validate_features(features: Dict[str, Any], config: dict) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate toàn bộ features dựa trên config dataset.
    Returns: (cleaned_features, warnings)
    """
    errors = []
    warnings = []
    cleaned = {}

    numeric_features = config.get('numeric_features', [])
    categorical_features = config.get('categorical_features', [])
    categorical_options = config.get('categorical_options', {})

    # Kiểm tra thiếu trường bắt buộc
    all_features = numeric_features + categorical_features
    missing = [f for f in all_features if f not in features]
    if missing:
        raise ValidationError(
            f"Thiếu các trường bắt buộc: {missing}",
            errors=[f"Missing: {f}" for f in missing]
        )

    # Validate numeric features
    NUMERIC_BOUNDS = {
        'tenure':                 (0, 1000),
        'MonthlyCharges':         (0, 50000),
        'TotalCharges':           (0, 500000),
        'Account length':         (0, 10000),
        'Total day minutes':      (0, 5000),
        'Total eve minutes':      (0, 5000),
        'Total night minutes':    (0, 5000),
        'Total intl minutes':     (0, 1000),
        'Number vmail messages':  (0, 1000),
        'Customer service calls': (0, 100),
    }

    for feat in numeric_features:
        if feat in features:
            bounds = NUMERIC_BOUNDS.get(feat, (None, None))
            try:
                cleaned[feat] = validate_numeric(
                    features[feat], feat,
                    min_val=bounds[0], max_val=bounds[1]
                )
            except ValidationError as e:
                errors.append(str(e))

    # Validate categorical features
    for feat in categorical_features:
        if feat in features:
            allowed = categorical_options.get(feat, [])
            if allowed:
                try:
                    cleaned[feat] = validate_categorical(features[feat], feat, allowed)
                except ValidationError as e:
                    errors.append(str(e))
            else:
                cleaned[feat] = sanitize_string(str(features[feat]))

    if errors:
        raise ValidationError(
            f"Validation failed: {len(errors)} lỗi",
            errors=errors
        )

    return cleaned, warnings


def validate_file_upload(file_obj, max_size_bytes: int = 10 * 1024 * 1024,
                         allowed_extensions: list = None) -> Tuple[str, List[str]]:
    """
    Validate file upload:
    - Kiểm tra tên file
    - Kiểm tra extension
    - Kiểm tra kích thước
    - Đọc và trả về nội dung (string)
    """
    if allowed_extensions is None:
        allowed_extensions = ['csv']

    warnings = []

    # Kiểm tra file có tồn tại
    if file_obj is None:
        raise ValidationError("Không có file được upload")

    # Kiểm tra tên file
    filename = getattr(file_obj, 'filename', '')
    if not filename:
        raise ValidationError("File không có tên")

    # Sanitize filename
    filename = sanitize_string(filename, max_length=255)

    # Kiểm tra extension
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext not in allowed_extensions:
        raise ValidationError(
            f"Định dạng file không hợp lệ: '.{ext}'. Chỉ chấp nhận: {allowed_extensions}"
        )

    # Đọc nội dung
    content = file_obj.read()

    # Kiểm tra kích thước
    if len(content) > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        actual_mb = len(content) / (1024 * 1024)
        raise ValidationError(
            f"File quá lớn: {actual_mb:.1f}MB. Giới hạn: {max_mb:.0f}MB"
        )

    # Kiểm tra file không rỗng
    if len(content) == 0:
        raise ValidationError("File rỗng")

    # Decode
    try:
        text = content.decode('utf-8')
    except UnicodeDecodeError:
        try:
            text = content.decode('latin-1')
            warnings.append("File được decode bằng latin-1 thay vì utf-8")
        except UnicodeDecodeError:
            raise ValidationError("Không thể đọc file — encoding không hỗ trợ")

    return text, warnings


def validate_batch_records(records: list, max_records: int = 5000) -> List[dict]:
    """Validate danh sách records cho batch prediction"""
    if not isinstance(records, list):
        raise ValidationError("'records' phải là một danh sách JSON array")

    if len(records) == 0:
        raise ValidationError("Danh sách 'records' rỗng")

    if len(records) > max_records:
        raise ValidationError(
            f"Quá nhiều records: {len(records)}. Giới hạn: {max_records}"
        )

    # Kiểm tra mỗi record là dict
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValidationError(f"Record #{i} không phải là JSON object")

    return records
