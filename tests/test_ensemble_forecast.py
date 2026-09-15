from pathlib import Path


def test_ensemble_script_exists():
    assert Path("examples/nino34_ensemble_forecast.py").exists()

