"""Check shipped Stage 2 evidence after copying it to another directory."""
import hashlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
import yaml
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="module")
def shipped_results():
    return json.loads((ROOT / "reports" / "training_results.json").read_text())


def test_runs_have_comparable_metrics_and_original_data_hash(shipped_results):
    results = shipped_results
    with sqlite3.connect(f"file:{(ROOT / 'mlflow.db').as_posix()}?mode=ro", uri=True) as db:
        for run in results["runs"]:
            params = dict(db.execute("SELECT key, value FROM params WHERE run_uuid=?", (run["run_id"],)))
            assert params["data_sha256"] == results["data_sha256"]
            assert params["split_sha256"] == results["split_sha256"]
            split_path = ROOT / "tracking_artifacts" / run["run_id"] / "artifacts" / "split.json"
            split = json.loads(split_path.read_text())
            assert hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest() == results["split_sha256"]
            assert set(split["training_ids"]).isdisjoint(split["validation_ids"])
            assert len(split["training_ids"]) == results["training_rows"]
            assert len(split["validation_ids"]) == results["validation_rows"]
            score = db.execute("SELECT value FROM latest_metrics WHERE run_uuid=? AND key='mae'",
                               (run["run_id"],)).fetchone()[0]
            assert score == pytest.approx(run["mae"])
        assert len(results["runs"]) >= 3
        assert results["best"]["mae"] == min(r["mae"] for r in results["runs"])
    findings = yaml.safe_load((ROOT / "reports" / "findings.yaml").read_text())
    assert findings["tracking"]["best_run_id"] == results["best"]["run_id"]
    assert hashlib.sha256((ROOT / "models" / "model.joblib").read_bytes()).hexdigest() == results["model_sha256"]


def test_registry_champion_loads_after_relocation(tmp_path, monkeypatch, shipped_results):
    shutil.copy2(ROOT / "mlflow.db", tmp_path / "mlflow.db")
    shutil.copytree(ROOT / "tracking_artifacts", tmp_path / "tracking_artifacts")
    monkeypatch.chdir(tmp_path)
    old_uri = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        client = MlflowClient()
        champion = client.get_model_version_by_alias(shipped_results["registered_model_name"], "champion")
        assert int(champion.version) == 1
        assert champion.run_id == shipped_results["best"]["run_id"]
        download_uri = client.get_model_version_download_uri(champion.name, champion.version)
        assert Path(download_uri).resolve().is_relative_to(tmp_path)
        assert Path(champion.source).resolve().is_relative_to(tmp_path)
        registry_model = mlflow.sklearn.load_model(f"models:/{champion.name}@champion")
        local_model = joblib.load(ROOT / "models" / "model.joblib")
        sample = pd.DataFrame({"highway_distance_km": [3.], "nearby_pumps": [5.],
                               "price_rs": [100.], "trucks_passing": [500.],
                               "temp_c": [25.], "staff": [999.],
                               "ownership": ["private "], "zone": ["City"]})
        np.testing.assert_allclose(registry_model.predict(sample), local_model.predict(sample))
        assert np.isfinite(registry_model.predict(sample)).all()
    finally:
        mlflow.set_tracking_uri(old_uri)
