"""Tests for the data contract. Small data built in memory, no CSVs needed."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mp299.validation import validate_features  # noqa: E402


def _good():
    return pd.DataFrame([{
        "highway_distance_km": 6.8, "nearby_pumps": 8, "price_rs": 108.37,
        "trucks_passing": 501, "temp_c": 29.7, "staff": 3,
        "ownership": "private", "zone": "Highway",
    }])


def test_good_row_passes():
    rep = validate_features(_good())
    assert rep.ok and rep.warnings == []


def test_client_rules_are_errors():
    df = _good().assign(temp_c=80.0, highway_distance_km=-1.0)
    rep = validate_features(df)
    assert not rep.ok
    assert any("temp_c" in e for e in rep.errors)
    assert any("highway_distance_km" in e for e in rep.errors)


def test_sentinel_unknown_category_and_missing_column():
    rep = validate_features(_good().assign(staff=999, zone="city "))
    assert any("sentinel" in e for e in rep.errors)
    assert any("unknown categories" in e for e in rep.errors)
    assert not validate_features(_good().drop(columns=["zone"])).ok


def test_hot_july_day_is_only_a_warning():
    rep = validate_features(_good().assign(temp_c=63.2))
    assert rep.ok
    assert any("temp_c" in w for w in rep.warnings)
