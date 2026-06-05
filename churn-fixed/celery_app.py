"""Gold Celery worker/beat tasks."""

from __future__ import annotations

import os
import subprocess

from celery import Celery
from celery.schedules import crontab


REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
celery_app = Celery("churniq", broker=REDIS_URL, backend=REDIS_URL)

celery_app.conf.beat_schedule = {
    "weekly-retrain-telco": {
        "task": "celery_app.scheduled_retrain",
        "schedule": crontab(minute=0, hour=2, day_of_week="sunday"),
        "args": ("telco_ibm", "scheduled"),
    },
    "weekly-retrain-calls": {
        "task": "celery_app.scheduled_retrain",
        "schedule": crontab(minute=0, hour=2, day_of_week="sunday"),
        "args": ("call_details", "scheduled"),
    },
}
celery_app.conf.timezone = "Asia/Saigon"


@celery_app.task
def scheduled_retrain(dataset: str, reason: str = "scheduled") -> dict:
    result = subprocess.run(
        ["python", "retrain.py", "--dataset", dataset, "--reason", reason],
        text=True,
        capture_output=True,
        check=False,
    )
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
