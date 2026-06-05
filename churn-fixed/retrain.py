"""Gold retrain CLI.

Usage:
    python retrain.py --dataset telco_ibm --reason "drift"
"""

from __future__ import annotations

import argparse
import os
import tempfile

from churniq.retraining import compare_and_promote
from datasets_config import DATASET_CONFIGS
from train_model import train_model_for_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain and promote ChurnIQ model if metrics pass the gate.")
    parser.add_argument("--dataset", required=True, choices=DATASET_CONFIGS.keys())
    parser.add_argument("--reason", default="manual")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DATASET_CONFIGS[args.dataset].copy()
    production_model_path = config["model_path"]
    production_metrics_path = production_model_path.replace(".pkl", "_metrics.json")

    with tempfile.TemporaryDirectory() as tmp:
        candidate_path = os.path.join(tmp, os.path.basename(production_model_path))
        config["model_path"] = candidate_path
        _, candidate_metrics = train_model_for_dataset(args.dataset, config, use_mlflow=True)
        decision = compare_and_promote(
            candidate_model_path=candidate_path,
            production_model_path=production_model_path,
            candidate_metrics=candidate_metrics,
            production_metrics_path=production_metrics_path,
            tolerance=0.02,
        )
        print(f"Retrain reason: {args.reason}")
        print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
