"""Model Lead acceptance check for the integrated Stage 3 service."""
from mp299.service_smoke import smoke_test


def test_service_prediction_matches_v1_champion_and_model_file():
    result = smoke_test()
    assert result["prediction_status"] == 200
    assert result["predictions_match"]
    assert result["champion_version"] == 1
