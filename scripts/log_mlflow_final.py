"""
Log final YOLO26n Step 3 960 metrics to a local MLflow run.

Run from the project root after installing optional MLOps dependencies:
    python3 -m pip install -r requirements-mlops.txt
    python3 scripts/log_mlflow_final.py
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "metrics" / "final_yolo26n_step3_960.json"
PARAMS_PATH = ROOT / "params.yaml"
TRACKING_DB = ROOT / "mlflow.db"
ARTIFACT_ROOT = ROOT / "mlruns"
EXPERIMENT_NAME = "smart-traffic-agadir-yolo26n"
BASE_RUN_NAME = "YOLO26n-Step3-960-final"


def flatten(prefix, value):
    if isinstance(value, dict):
        flattened = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten(child_prefix, child))
        return flattened
    return {prefix: value}


def file_fingerprint(*paths):
    digest = hashlib.md5()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:8]


def main():
    try:
        import mlflow
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "MLflow/YAML dependencies are not installed. "
            "Install them with: python3 -m pip install -r requirements-mlops.txt"
        ) from exc

    metrics = json.loads(METRICS_PATH.read_text())
    params = yaml.safe_load(PARAMS_PATH.read_text())
    run_name = f"{BASE_RUN_NAME}-{file_fingerprint(PARAMS_PATH, METRICS_PATH)}"

    tracking_uri = f"sqlite:///{TRACKING_DB}"
    artifact_uri = f"file://{ARTIFACT_ROOT}"
    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    if client.get_experiment_by_name(EXPERIMENT_NAME) is None:
        client.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    existing_runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    run_context = (
        mlflow.start_run(run_id=existing_runs[0].info.run_id)
        if existing_runs
        else mlflow.start_run(run_name=run_name)
    )

    with run_context as active_run:
        for key, value in flatten("", params).items():
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(key, value)

        mlflow.log_metric("val_precision", metrics["validation"]["precision"])
        mlflow.log_metric("val_recall", metrics["validation"]["recall"])
        mlflow.log_metric("val_mAP50", metrics["validation"]["mAP50"])
        mlflow.log_metric("val_mAP50_95", metrics["validation"]["mAP50_95"])
        mlflow.log_metric("test_precision", metrics["test"]["precision"])
        mlflow.log_metric("test_recall", metrics["test"]["recall"])
        mlflow.log_metric("test_mAP50", metrics["test"]["mAP50"])
        mlflow.log_metric("test_mAP50_95", metrics["test"]["mAP50_95"])
        mlflow.log_metric(
            "test_person_recall",
            metrics["test_per_class"]["person"]["recall"],
        )
        mlflow.log_metric(
            "test_person_mAP50",
            metrics["test_per_class"]["person"]["mAP50"],
        )
        deployment = metrics["deployment_benchmark"]
        mlflow.log_metric(
            "raspberry_pi_yolo26n_family_fps",
            deployment["fps"],
        )
        if "step3_960_fps" in deployment:
            mlflow.log_metric("raspberry_pi_step3_960_fps", deployment["step3_960_fps"])
            mlflow.log_metric(
                "raspberry_pi_step3_960_latency_ms",
                deployment["step3_960_latency_ms"],
            )
            mlflow.log_metric(
                "raspberry_pi_step3_960_anonymized_detections",
                deployment["step3_960_anonymized_detections"],
            )
        if "step3_800_from960_fps" in deployment:
            mlflow.log_metric(
                "raspberry_pi_step3_800_from960_fps",
                deployment["step3_800_from960_fps"],
            )
            mlflow.log_metric(
                "raspberry_pi_step3_800_from960_latency_ms",
                deployment["step3_800_from960_latency_ms"],
            )
        for variants_key, variants in deployment.items():
            if not variants_key.endswith("_variants") or not isinstance(variants, dict):
                continue
            model_prefix = variants_key[: -len("_variants")]
            for variant_name, variant in variants.items():
                metric_prefix = f"raspberry_pi_{model_prefix}_{variant_name}"
                for metric_name, metric_value in variant.items():
                    if isinstance(metric_value, bool):
                        continue
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(
                            f"{metric_prefix}_{metric_name}",
                            metric_value,
                        )

        mlflow.log_artifact(str(METRICS_PATH))
        mlflow.log_artifact(str(PARAMS_PATH))
        for artifact in (
            ROOT / "models" / "downloads" / "YOLO26n_step3_960_best.pt",
            ROOT / "models" / "downloads" / "YOLO26n_step3_960_results.csv",
            ROOT / "models" / "downloads" / "YOLO26n_step3_960_best.onnx",
            ROOT / "models" / "downloads" / "YOLO26n_step3_800_from960_best.onnx",
            ROOT / "models" / "downloads" / "YOLO26n_step3_800_from960_int8.onnx",
        ):
            if artifact.exists():
                mlflow.log_artifact(str(artifact))

    print(f"MLflow run {active_run.info.run_id} logged in {TRACKING_DB}")
    print(f"Artifacts stored in {ARTIFACT_ROOT}")


if __name__ == "__main__":
    main()
