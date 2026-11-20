"""Tests for the seven data checks. They build tiny DataFrames in memory,
so they do not need the CSVs (which are tracked by DVC, not git)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mp299 import data_checks as dc  # noqa: E402


def _frame(n=30):
    return pd.DataFrame({
        "record_id": [f"R{i:03d}" for i in range(n)],
        "date": pd.date_range("2026-01-01", periods=n).strftime("%Y-%m-%d"),
        "staff": [5 + i % 10 for i in range(n)],
        "nearby_pumps": [i % 7 for i in range(n)],
        "zone": ["City", "Highway", "Industrial"] * (n // 3),
        "litres_per_day": [10000 + 300 * i for i in range(n)],
    })


def test_sentinel_found():
    df = _frame()
    df.loc[:5, "staff"] = 999
    r = dc.check_sentinel_value(df, ["staff"], [-1, 999, 9999], min_count=5)
    assert r == {"present": True, "column": "staff", "value": 999, "count": 6}


def test_stuck_sensor_dates():
    df = _frame()
    df.loc[10:21, "nearby_pumps"] = 6
    r = dc.check_stuck_sensor(df, ["nearby_pumps"], "date", min_rows=10)
    assert r["present"] and r["column"] == "nearby_pumps"
    assert (r["start_date"], r["end_date"]) == ("2026-01-11", "2026-01-22")


def test_target_typos_count():
    df = _frame()
    df.loc[[3, 7], "litres_per_day"] *= 10
    assert dc.check_target_typos(df, "litres_per_day") == {"present": True, "count": 2}


def test_clean_frame_has_no_duplicates_or_spelling_issues():
    df = _frame()
    assert dc.check_duplicate_rows(df, "record_id")["present"] is False
    assert dc.check_spelling_variants(df, ["zone"])["present"] is False
    df.loc[0, "zone"] = "city "
    r = dc.check_spelling_variants(df, ["zone"])
    assert r == {"present": True, "column": "zone", "true_levels": 3}
