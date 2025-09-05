import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless test env
import pytest

from student_success_tool.modeling.h2o_modeling import evaluation, training


def test_create_and_log_h2o_model_comparison(monkeypatch, tmp_path):
    # Fake leaderboard with only GBM models
    fake_lb = pd.DataFrame(
        {
            "model_id": [
                "GBM_lr_annealing_selection_AutoML_2_20250823_00331_select_model",
                "GBM_grid_1_AutoML_2_20250823_00331_model_119",
                "GBM_grid_1_AutoML_2_20250823_00331_model_167",
            ],
            "logloss": [0.5538, 0.5539, 0.5544],
            "auc": [0.7906, 0.7900, 0.7896],
        }
    )

    class DummyAML:
        leaderboard = fake_lb

    # monkeypatch utils._to_pandas to return our fake lb
    monkeypatch.setattr(evaluation.utils, "_to_pandas", lambda _: fake_lb)

    # monkeypatch mlflow.log_figure so it doesn’t try to actually log
    called = {}

    def fake_log_figure(fig, artifact_path):
        called["artifact_path"] = artifact_path

    monkeypatch.setattr(evaluation.mlflow, "log_figure", fake_log_figure)
    monkeypatch.setattr(evaluation.mlflow, "active_run", lambda: True)

    # Run the function
    best = evaluation.create_and_log_h2o_model_comparison(
        DummyAML(), artifact_path="model_comparison.png"
    )

    # Assertions
    assert "framework" in best.columns
    assert set(best["framework"]) <= training.VALID_H2O_FRAMEWORKS
    # Best logloss should be the first row (lowest value)
    assert best.iloc[0]["logloss"] == pytest.approx(min(fake_lb["logloss"]))
    # MLflow log was called with expected path
    assert called["artifact_path"] == "model_comparison.png"
