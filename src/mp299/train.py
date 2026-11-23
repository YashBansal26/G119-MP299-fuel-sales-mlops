"""Train and track v1: run from the repository root with python -m mp299.train."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.tracking import MlflowClient
from mlflow.types.schema import ColSpec, Schema, TensorSpec
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from mp299.config import ROOT, load_params
from mp299.modeling import build_pipeline, repair_training_rows


def model_signature(params):
    """All eight columns are required; numeric values may still be missing."""
    return ModelSignature(
        inputs=Schema([ColSpec("double", c) for c in params["features"]["numeric"]]
                      + [ColSpec("string", c) for c in params["features"]["categorical"]]),
        outputs=Schema([TensorSpec(np.dtype("float64"), (-1,))]),
    )


def train(root: Path = ROOT, params: dict | None = None) -> dict:
    """Use one fixed split for every run and retain the exact scored winner."""
    root = Path(root).resolve()
    if Path.cwd().resolve() != root:
        raise ValueError("Run training from the repository root so artifact paths remain portable")
    params = params or load_params(root / "params.yaml")
    cfg, data = params["training"], params["data"]
    raw_path = root / data["train"]
    data_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    if data_sha256 != cfg["expected_data_sha256"]:
        raise ValueError("Original train.csv SHA256 does not match params.yaml")
    frame = repair_training_rows(pd.read_csv(raw_path), params)
    features = params["features"]["numeric"] + params["features"]["categorical"]
    train_idx, valid_idx = train_test_split(
        np.arange(len(frame)), test_size=cfg["validation_fraction"],
        random_state=cfg["random_state"], stratify=frame[data["group_col"]],
    )
    X_train, X_valid = frame.iloc[train_idx][features], frame.iloc[valid_idx][features]
    y_train, y_valid = frame.iloc[train_idx][data["target"]], frame.iloc[valid_idx][data["target"]]
    split = {"training_ids": frame.iloc[train_idx][data["id_col"]].tolist(),
             "validation_ids": frame.iloc[valid_idx][data["id_col"]].tolist()}
    split_hash = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
    baseline = DummyRegressor(strategy="median").fit(X_train, y_train)
    baseline_mae = float(mean_absolute_error(y_valid, baseline.predict(X_valid)))

    # Relative artifact paths make the database and registry portable with the repo.
    mlflow.set_tracking_uri("sqlite:///" + (root / "mlflow.db").as_posix())
    client = MlflowClient()
    existing = client.search_model_versions(f"name='{cfg['registered_model_name']}'")
    if existing:
        raise ValueError("v1 already exists. Preserve it; use a separate retraining command for v2.")
    experiment = client.get_experiment_by_name(cfg["experiment_name"])
    experiment_id = (experiment.experiment_id if experiment else client.create_experiment(
        cfg["experiment_name"], artifact_location=cfg["artifact_location"]))
    versions = {name: importlib.metadata.version(name) for name in
                ["pandas", "numpy", "scikit-learn", "scipy", "mlflow-skinny", "joblib", "cloudpickle"]}
    results, best_model = [], None
    for candidate in cfg["candidates"]:
        with mlflow.start_run(experiment_id=experiment_id, run_name=candidate["name"]) as run:
            model = build_pipeline(params, candidate)
            model.fit(X_train, y_train)
            predictions = model.predict(X_valid)
            mae = float(mean_absolute_error(y_valid, predictions))
            mlflow.log_params({"model": candidate["model"], **candidate["params"],
                               "data_sha256": data_sha256, "split_sha256": split_hash,
                               "random_state": cfg["random_state"],
                               "validation_fraction": cfg["validation_fraction"],
                               "training_rows": len(train_idx), "validation_rows": len(valid_idx)})
            mlflow.set_tags({"group_id": "G119", "problem_id": "MP299", "stage": "2",
                             "preprocessing_fit": "training_partition_only"})
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("baseline_mae", baseline_mae)
            mlflow.log_dict(split, "split.json")
            mlflow.log_dict(versions, "versions.json")
            mlflow.log_dict(params, "settings.json")
            row = {"name": candidate["name"], "run_id": run.info.run_id,
                   "model": candidate["model"], "params": candidate["params"], "mae": mae}
            if not results or mae < min(r["mae"] for r in results):
                best_model = model
            results.append(row)
            print(f"{candidate['name']}: MAE={mae:.3f}")

    best = min(results, key=lambda r: r["mae"])
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.sklearn.log_model(
            best_model, artifact_path="model", code_paths=[str(ROOT / "src" / "mp299")],
            signature=model_signature(params),
            pip_requirements=[f"{name}=={version}" for name, version in versions.items()],
        )
    client.create_registered_model(cfg["registered_model_name"])
    registered = client.create_model_version(
        cfg["registered_model_name"],
        source=f"{cfg['artifact_location']}/{best['run_id']}/artifacts/model",
        run_id=best["run_id"],
    )
    if int(registered.version) != 1:
        raise RuntimeError("Expected the initial registered model to be version 1")
    summary = {"experiment_name": cfg["experiment_name"], "data_sha256": data_sha256,
               "split_sha256": split_hash, "training_rows": len(train_idx),
               "validation_rows": len(valid_idx), "baseline_mae": baseline_mae,
               "best": best, "registered_model_name": cfg["registered_model_name"],
               "champion_version": int(registered.version), "runs": results, "versions": versions}
    return save_champion(root, params, client, registered, best_model, X_valid, summary)


def save_champion(root, params, client, registered, best_model, X_valid, summary):
    """Publish and verify an already trained v1 without fitting it again."""
    cfg, best = params["training"], summary["best"]
    client.set_registered_model_alias(cfg["registered_model_name"], "champion", registered.version)
    model_path = root / cfg["model_output"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path, compress=3)
    # Verify the serialized artifact and the registry point to the scored pipeline.
    expected = best_model.predict(X_valid)
    np.testing.assert_allclose(joblib.load(model_path).predict(X_valid), expected)
    champion = mlflow.sklearn.load_model(f"models:/{cfg['registered_model_name']}@champion")
    np.testing.assert_allclose(champion.predict(X_valid), expected)
    summary["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    (root / "reports" / "training_results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    findings_path = root / "reports" / "findings.yaml"
    findings = findings_path.read_text(encoding="utf-8")
    tracking = {"experiment_name": cfg["experiment_name"], "best_run_id": best["run_id"],
                "registered_model_name": cfg["registered_model_name"], "champion_version": int(registered.version)}
    # Replace just tracking values, preserving teammates' comments and later-stage fields.
    for key, value in tracking.items():
        rendered = json.dumps(value)
        findings = re.sub(rf"(?m)^(  {key}:).*?$", rf"\g<1> {rendered}", findings)
    findings_path.write_text(findings, encoding="utf-8")
    print(f"Champion v{registered.version}: {best['name']}; MAE={best['mae']:.3f}")
    return summary


if __name__ == "__main__":
    train()
