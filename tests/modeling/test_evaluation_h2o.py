import numpy as np
import unittest.mock as mock

from student_success_tool.modeling import evaluation_h2o


@mock.patch(
    "student_success_tool.modeling.evaluation_h2o.generate_all_classification_plots"
)
@mock.patch(
    "student_success_tool.modeling.evaluation_h2o.get_metrics_near_threshold_all_splits"
)
@mock.patch("student_success_tool.modeling.evaluation_h2o.h2o.save_model")
@mock.patch("student_success_tool.modeling.evaluation_h2o.mlflow.active_run")
@mock.patch("student_success_tool.modeling.evaluation_h2o.mlflow.start_run")
@mock.patch("student_success_tool.modeling.evaluation_h2o.h2o.get_model")
def test_evaluate_and_log_model_success(
    mock_get_model,
    mock_start_run,
    mock_active_run,
    mock_save_model,
    mock_get_metrics,
    mock_generate_plots,
):
    # Mock model prediction
    model_mock = mock.MagicMock()
    model_mock.predict.return_value.col_names = ["p0", "p1"]
    model_mock.predict.return_value.__getitem__.return_value.as_data_frame.return_value.values.flatten.return_value = np.array(
        [0.6, 0.7, 0.8]
    )
    mock_get_model.return_value = model_mock

    # Mock metrics and plots
    mock_get_metrics.return_value = {"accuracy": 0.91}
    mock_generate_plots.return_value = None

    # Mock MLflow run
    mock_active_run.return_value.info.run_id = "run-xyz"

    # Call function under test
    result = evaluation_h2o.evaluate_and_log_model(
        aml=mock.MagicMock(),
        model_id="model1",
        train=mock.MagicMock(),
        valid=mock.MagicMock(),
        test=mock.MagicMock(),
        threshold=0.5,
        client=mock.MagicMock(),
    )

    # Assertions
    assert isinstance(result, dict)
    assert "mlflow_run_id" in result
    assert result["mlflow_run_id"] == "run-xyz"
    assert result["accuracy"] == 0.91
    mock_save_model.assert_called_once()
    mock_get_metrics.assert_called_once()

    expected_calls = [
        mock.call(mock.ANY, mock.ANY, mock.ANY, prefix="train"),
        mock.call(mock.ANY, mock.ANY, mock.ANY, prefix="val"),
        mock.call(mock.ANY, mock.ANY, mock.ANY, prefix="test"),
    ]
    mock_generate_plots.assert_has_calls(expected_calls, any_order=False)
