"""PostgreSQL storage helpers for Gold tier audit tables."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://churniq:churniq@postgres:5432/churniq")


@contextmanager
def connection() -> Iterator[object]:
    try:
        import psycopg2
    except ImportError as exc:
        raise RuntimeError("psycopg2-binary is required for PostgreSQL storage") from exc
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def insert_prediction(record: dict) -> None:
    sql = """
    INSERT INTO predictions
      (request_id, dataset, model_version, probability, prediction, latency_ms, input_hash)
    VALUES (%(request_id)s, %(dataset)s, %(model_version)s, %(probability)s, %(prediction)s, %(latency_ms)s, %(input_hash)s)
    """
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, record)
