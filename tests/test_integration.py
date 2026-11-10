"""
Slow tests: load the real models. Run with:  pytest -m slow
"""

from pathlib import Path

import pytest

from basha.fusion.english_fusion import EnglishFusion

FINETUNED = (Path(__file__).resolve().parents[1]
             / "evaluation" / "models" / "gauneg_basha_v1" / "best_model")


@pytest.fixture(scope="session")
def fusion():
    path = str(FINETUNED) if FINETUNED.exists() else None
    return EnglishFusion(path)


@pytest.mark.slow
def test_camera_battery_is_mixed(fusion):
    result = fusion.analyze(
        "The camera is terrible, but the battery life is excellent."
    )
    assert result["sentiment"]["label"] == "mixed"


@pytest.mark.slow
def test_clearly_positive(fusion):
    result = fusion.analyze("I absolutely love this phone. It is fantastic.")
    assert result["sentiment"]["label"] == "positive"


@pytest.mark.slow
def test_clearly_negative(fusion):
    result = fusion.analyze("This was a complete waste of money. Terrible.")
    assert result["sentiment"]["label"] == "negative"


@pytest.mark.slow
def test_output_has_expected_keys(fusion):
    result = fusion.analyze("The food is delicious, but the service is terrible.")
    for key in ("sentiment", "decision_source", "aspects", "mixed_analysis"):
        assert key in result