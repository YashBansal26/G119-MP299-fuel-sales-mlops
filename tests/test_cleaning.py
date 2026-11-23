"""Tests for the cleaning code. Small data built in memory, no CSVs needed."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mp299.cleaning import clean  # noqa: E402

PARAMS = {
    "data": {"id_col": "record_id", "date_col": "date", "target": "litres_per_day"},
    "features": {"numeric": ["nearby_pumps", "staff", "temp_c"],
                 "categorical": ["ownership", "zone"]},
    "cleaning": {"sentinels": {"staff": 999},
                 "stuck_sensor": {"column": "nearby_pumps",
                                  "start_date": "2026-04-06", "end_date": "2026-04-24"},
                 "target_typo_threshold": 60000, "target_typo_factor": 10},
}


def _raw():
    return pd.DataFrame({
        "record_id": ["R1", "R2", "R3", "R4", "R5", "R6"],
        "date": ["2026-04-01", "2026-04-06", "2026-04-20", "2026-04-24", "2026-05-01", "2026-05-01"],
        "nearby_pumps": [2, 6, 6, 6, 4, 4],
        "staff": [10, 999, 12, 14, 8, 8],
        "temp_c": [20.0, 21.0, 22.0, 23.0, 24.0, 24.0],
        "ownership": ["private", "public_sector ", "private", "cooperative", "private", "private"],
        "zone": ["City", "Highway", "Industrial", "City", "City", "City"],
        "litres_per_day": [15000, 182000, 16000, 17000, 18000, 18000],
    })


def test_cleaning_fixes_all_three_problems():
    out, medians = clean(_raw(), PARAMS)
    assert len(out) == 5                                   # duplicate R6 dropped
    assert (out["staff"] != 999).all()                     # sentinel gone
    assert out["litres_per_day"].max() == 18200            # 182000 / 10
    assert out.isna().sum().sum() == 0                     # nothing missing
    stuck = out[out["date"].between("2026-04-06", "2026-04-24")]
    assert (stuck["nearby_pumps"] == medians["nearby_pumps"]).all()
    assert set(out["ownership"]) == {"private", "public_sector", "cooperative"}


def test_cleaning_does_not_touch_the_input():
    raw = _raw()
    clean(raw, PARAMS)
    assert raw.loc[1, "staff"] == 999


def test_given_medians_are_reused():
    _, medians = clean(_raw(), PARAMS)
    later = _raw().assign(date="2026-07-01")
    out, used = clean(later, PARAMS, medians)
    assert used == medians
