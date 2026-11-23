"""Protect the data split, learned preprocessing and inference contract."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mp299.config import load_params
from mp299.modeling import build_pipeline, repair_training_rows


def example_frame():
    return pd.DataFrame({
        "record_id": ["R1", "R2", "R3", "R4"],
        "date": ["2026-01-01", "2026-01-02", "2026-04-10", "2026-06-01"],
        "highway_distance_km": [2., 4., 6., 8.], "nearby_pumps": [2., 4., 6., 8.],
        "price_rs": [90., 95., 100., 105.], "trucks_passing": [100, 200, 300, 400],
        "temp_c": [20., 22., 24., 26.], "staff": [3, 5, 999, 9],
        "ownership": ["private", "private ", "cooperative", "public_sector"],
        "zone": ["City", "City", "Highway", "Industrial"],
        "litres_per_day": [10000, 12000, 150000, 20000],
    })


def test_no_global_imputation_before_split():
    params = load_params()
    raw = example_frame()
    repaired = repair_training_rows(raw, params)
    assert np.isnan(repaired.loc[2, "nearby_pumps"])
    assert repaired.loc[2, "litres_per_day"] == 15000
    model = build_pipeline(params, params["training"]["candidates"][0])
    model.fit(repaired.iloc[:2], repaired.iloc[:2]["litres_per_day"])
    imputer = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
    assert imputer.statistics_[-1] == 4  # only training staff 3 and 5
    stats = imputer.statistics_.copy()
    pred = model.predict(repaired.iloc[2:])
    assert np.isfinite(pred).all()
    np.testing.assert_array_equal(imputer.statistics_, stats)
    assert raw.loc[2, "nearby_pumps"] == 6
    assert raw.loc[2, "litres_per_day"] == 150000


def test_only_eight_features_and_round_trip(tmp_path):
    params = load_params()
    repaired = repair_training_rows(example_frame(), params)
    model = build_pipeline(params, params["training"]["candidates"][0])
    model.fit(repaired, repaired["litres_per_day"])
    sample = repaired.iloc[:1].copy()
    sample["zone"] = "unseen_zone"
    sample["staff"] = 999
    sample["record_id"] = "different"
    sample["date"] = "2099-01-01"
    sample["litres_per_day"] = -999999
    features = params["features"]["numeric"] + params["features"]["categorical"]
    np.testing.assert_allclose(model.predict(sample), model.predict(sample[features]))
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_allclose(joblib.load(path).predict(sample), model.predict(sample))
    assert len(model.feature_names_in_) == 8


def test_missing_required_feature_is_rejected():
    params = load_params()
    frame = repair_training_rows(example_frame(), params)
    model = build_pipeline(params, params["training"]["candidates"][0])
    model.fit(frame, frame["litres_per_day"])
    with pytest.raises(KeyError):
        model.predict(frame.drop(columns="staff"))
