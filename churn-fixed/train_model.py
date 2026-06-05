"""
train_model.py — Bronze/Silver training entrypoint for ChurnIQ.

Keeps the existing command:
    python train_model.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime

import joblib

from churniq.etl import prepare_dataset, print_quality_report
from churniq.modeling import build_pipeline, evaluate_model
from churniq.monitoring import numeric_baseline
from datasets_config import DATASET_CONFIGS

warnings.filterwarnings("ignore")

try:
    import mlflow
    import mlflow.sklearn

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


BASE_DIR = os.getcwd()

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def setup_mlflow() -> bool:
    if not MLFLOW_AVAILABLE:
        print("ℹ️  MLflow không có sẵn — bỏ qua logging MLflow")
        return False
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_URI", "http://host.docker.internal:5000"))
        mlflow.set_experiment("churniq")
        return True
    except Exception as exc:
        print(f"⚠️  MLflow không kết nối được: {exc}")
        return False


def _data_path(dataset_key: str) -> str:
    filename = "Telco_customer_churn.csv" if dataset_key == "telco_ibm" else "Churn.csv"
    return os.path.join(BASE_DIR, "data", filename)


def _metrics_path(model_path: str) -> str:
    return model_path.replace(".pkl", "_metrics.json")


def train_model_for_dataset(dataset_key: str, config: dict, file_path: str | None = None, use_mlflow: bool = False):
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  🚀 Bronze ETL + Training: {config['name']}")
    print(sep)

    file_path = file_path or _data_path(dataset_key)
    prepared = prepare_dataset(file_path, config, test_size=0.2, random_state=42)
    print_quality_report(prepared.report)

    pipeline = build_pipeline(config["numeric_features"], config["categorical_features"])
    trained_model, metrics = evaluate_model(
        pipeline,
        prepared.X_train,
        prepared.X_test,
        prepared.y_train,
        prepared.y_test,
        cv_folds=5,
    )

    print("\n  📈 KẾT QUẢ ĐÁNH GIÁ")
    print(f"     Cross-val AUC-ROC : {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
    print(f"     Cross-val F1      : {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")
    print(f"     Test AUC-ROC      : {metrics['test_auc']:.4f}")
    print(f"     Test Precision    : {metrics['test_precision']:.4f}")
    print(f"     Test Recall       : {metrics['test_recall']:.4f}")
    print(f"     Test F1           : {metrics['test_f1']:.4f}")
    print("\n  📋 Classification report:")
    for line in metrics["classification_report"].splitlines():
        if line.strip():
            print(f"     {line}")

    feature_cols = config["numeric_features"] + config["categorical_features"]
    final_model = build_pipeline(config["numeric_features"], config["categorical_features"])
    final_model.fit(prepared.df[feature_cols], prepared.df["Churn"])

    os.makedirs(os.path.dirname(config["model_path"]), exist_ok=True)
    joblib.dump(final_model, config["model_path"], compress=3)
    print(f"\n  ✅ Model lưu tại: {config['model_path']}")

    baseline = numeric_baseline(prepared.df, config["numeric_features"])
    metrics_out = {
        "dataset_key": dataset_key,
        "dataset": config["name"],
        "trained_at": datetime.utcnow().isoformat(),
        "train_samples": int(len(prepared.df)),
        "baseline_churn_rate": round(float(prepared.df["Churn"].mean()), 4),
        "feature_baseline": baseline,
        "data_quality": prepared.report,
        **metrics,
    }
    with open(_metrics_path(config["model_path"]), "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, ensure_ascii=False, indent=2)
    print(f"  📝 Metrics lưu tại: {_metrics_path(config['model_path'])}")

    if use_mlflow and MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name=f"train_{dataset_key}"):
            mlflow.log_params({"dataset": dataset_key, "model_type": "RandomForest", "samples": len(prepared.df)})
            mlflow.log_metrics(
                {
                    "test_auc": metrics["test_auc"],
                    "test_f1": metrics["test_f1"],
                    "test_precision": metrics["test_precision"],
                    "test_recall": metrics["test_recall"],
                    "baseline_churn_rate": metrics_out["baseline_churn_rate"],
                }
            )
            mlflow.log_artifact(_metrics_path(config["model_path"]))
            mlflow.sklearn.log_model(final_model, artifact_path=f"model_{dataset_key}")
        print("  📡 Đã log run lên MLflow")

    return final_model, metrics_out


def main() -> None:
    print("\n" + "═" * 72)
    print("  🔮 CHURNIQ — BRONZE TRAINING PIPELINE")
    print("═" * 72)
    use_mlflow = setup_mlflow()
    results = {}
    for dataset_key, config in DATASET_CONFIGS.items():
        try:
            _, metrics = train_model_for_dataset(dataset_key, config, use_mlflow=use_mlflow)
            results[dataset_key] = metrics
        except Exception as exc:
            print(f"  ❌ {dataset_key}: {exc}")

    print("\n" + "═" * 72)
    print("  📊 TỔNG KẾT")
    print("═" * 72)
    for dataset_key, metrics in results.items():
        print(
            f"  {DATASET_CONFIGS[dataset_key]['name']}: "
            f"AUC={metrics['test_auc']:.4f} | Precision={metrics['test_precision']:.4f} | "
            f"Recall={metrics['test_recall']:.4f} | F1={metrics['test_f1']:.4f}"
        )
    print("\n  Streamlit: streamlit run app.py")
    print("  API:       python api.py")


if __name__ == "__main__":
    main()
