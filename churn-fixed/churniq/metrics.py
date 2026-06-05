"""Small in-process Prometheus metrics collector."""

from __future__ import annotations

import time
from collections import defaultdict


class MetricsRegistry:
    def __init__(self) -> None:
        self.request_count = defaultdict(int)
        self.error_count = defaultdict(int)
        self.latencies = defaultdict(list)
        self.started_at = time.time()

    def record(self, endpoint: str, status_code: int, latency_ms: float) -> None:
        self.request_count[endpoint] += 1
        if status_code >= 400:
            self.error_count[endpoint] += 1
        values = self.latencies[endpoint]
        values.append(float(latency_ms))
        if len(values) > 1000:
            del values[: len(values) - 1000]

    @staticmethod
    def _percentile(values: list[float], pct: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
        return ordered[idx]

    def render_prometheus(self) -> str:
        lines = [
            "# HELP churniq_uptime_seconds Process uptime.",
            "# TYPE churniq_uptime_seconds gauge",
            f"churniq_uptime_seconds {time.time() - self.started_at:.0f}",
            "# HELP churniq_request_count Total requests by endpoint.",
            "# TYPE churniq_request_count counter",
        ]
        for endpoint, count in self.request_count.items():
            lines.append(f'churniq_request_count{{endpoint="{endpoint}"}} {count}')
        lines.extend(["# HELP churniq_error_count Total errors by endpoint.", "# TYPE churniq_error_count counter"])
        for endpoint, count in self.error_count.items():
            lines.append(f'churniq_error_count{{endpoint="{endpoint}"}} {count}')
        lines.extend(["# HELP churniq_latency_ms Request latency percentiles.", "# TYPE churniq_latency_ms gauge"])
        for endpoint, values in self.latencies.items():
            for pct in (50, 95, 99):
                lines.append(f'churniq_latency_ms{{endpoint="{endpoint}",quantile="p{pct}"}} {self._percentile(values, pct):.2f}')
        return "\n".join(lines) + "\n"


metrics = MetricsRegistry()
