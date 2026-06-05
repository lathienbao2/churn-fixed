"""
Tests cho Config Loader
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestConfigLoader:
    """Test config loading từ .env"""

    def test_config_loads(self):
        """Config object được tạo thành công"""
        from utils.config_loader import config
        assert config is not None
        assert config.APP_VERSION is not None

    def test_config_defaults(self):
        """Kiểm tra giá trị mặc định"""
        from utils.config_loader import config
        assert config.API_PORT == 8001 or isinstance(config.API_PORT, int)
        assert config.RATE_LIMIT_PER_MINUTE > 0
        assert config.MAX_UPLOAD_SIZE_MB > 0
        assert config.MAX_BATCH_RECORDS > 0

    def test_config_repr(self):
        """Config repr không leak API key"""
        from utils.config_loader import config
        repr_str = repr(config)
        assert 'Config(' in repr_str
        # API key phải bị ẩn
        assert config.API_KEY not in repr_str or config.API_KEY == ''

    def test_is_production(self):
        """Test production check"""
        from utils.config_loader import config
        # Trong test, nên là development
        assert config.is_production == (config.APP_ENV == 'production')

    def test_allowed_extensions_is_list(self):
        """ALLOWED_EXTENSIONS phải là list"""
        from utils.config_loader import config
        assert isinstance(config.ALLOWED_EXTENSIONS, list)
        assert 'csv' in config.ALLOWED_EXTENSIONS

    def test_env_file_parser(self):
        """Test _load_env_file parser"""
        from utils.config_loader import _load_env_file
        from pathlib import Path
        import tempfile

        # Tạo file .env tạm
        tmp_dir = os.path.join(os.path.dirname(__file__), '..', '_test_tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_file = os.path.join(tmp_dir, '.env.test')

        try:
            with open(tmp_file, 'w') as f:
                f.write("# Comment line\n")
                f.write("\n")
                f.write("TEST_KEY=test_value\n")
                f.write('QUOTED_KEY="quoted_value"\n')
                f.write("NUMBER_KEY=42\n")

            result = _load_env_file(Path(tmp_file))
            assert result['TEST_KEY'] == 'test_value'
            assert result['QUOTED_KEY'] == 'quoted_value'
            assert result['NUMBER_KEY'] == '42'
        finally:
            os.remove(tmp_file)
            os.rmdir(tmp_dir)
