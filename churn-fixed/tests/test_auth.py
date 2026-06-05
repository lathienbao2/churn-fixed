"""
Tests cho Authentication & Rate Limiting
"""

import os
import sys
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.auth import RateLimiter, _constant_time_compare


class TestRateLimiter:
    """Test rate limiter (token bucket)"""

    def test_allows_initial_requests(self):
        """Burst cho phép request ban đầu"""
        limiter = RateLimiter(rate_per_minute=60, burst=5)
        for _ in range(5):
            assert limiter.allow_request('test_ip') is True

    def test_blocks_after_burst(self):
        """Block sau khi hết burst"""
        limiter = RateLimiter(rate_per_minute=60, burst=3)
        for _ in range(3):
            limiter.allow_request('test_ip')
        assert limiter.allow_request('test_ip') is False

    def test_refills_over_time(self):
        """Tokens được refill theo thời gian"""
        limiter = RateLimiter(rate_per_minute=6000, burst=1)  # 100/sec
        assert limiter.allow_request('test_ip') is True
        assert limiter.allow_request('test_ip') is False
        time.sleep(0.02)  # chờ refill
        assert limiter.allow_request('test_ip') is True

    def test_separate_ips(self):
        """Mỗi IP có bucket riêng"""
        limiter = RateLimiter(rate_per_minute=60, burst=1)
        assert limiter.allow_request('ip_1') is True
        assert limiter.allow_request('ip_2') is True
        assert limiter.allow_request('ip_1') is False
        assert limiter.allow_request('ip_2') is False

    def test_retry_after(self):
        """get_retry_after trả về giá trị hợp lý"""
        limiter = RateLimiter(rate_per_minute=60, burst=1)
        limiter.allow_request('test_ip')
        limiter.allow_request('test_ip')
        retry = limiter.get_retry_after('test_ip')
        assert retry >= 0

    def test_cleanup(self):
        """Cleanup dọn buckets cũ"""
        limiter = RateLimiter(rate_per_minute=60, burst=5)
        limiter.allow_request('old_ip')
        limiter._buckets['old_ip']['last'] = time.time() - 7200  # 2h trước
        limiter.cleanup(max_age_seconds=3600)
        assert 'old_ip' not in limiter._buckets


class TestConstantTimeCompare:
    """Test constant-time string comparison"""

    def test_equal_strings(self):
        assert _constant_time_compare('abc123', 'abc123') is True

    def test_different_strings(self):
        assert _constant_time_compare('abc123', 'def456') is False

    def test_different_lengths(self):
        assert _constant_time_compare('short', 'longer_string') is False

    def test_empty_strings(self):
        assert _constant_time_compare('', '') is True

    def test_one_empty(self):
        assert _constant_time_compare('', 'notempty') is False
