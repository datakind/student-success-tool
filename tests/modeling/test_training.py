from unittest.mock import MagicMock, patch

import pandas as pd

from student_success_tool.modeling.training import (
    run_automl_classification,
)


def test_run_automl_classification_uses_correct_args_and_format():
    mymodule = MagicMock()

    train_df = pd.DataFrame(
        {"id": [1, 2, 3], "semester_start": [1, 1, 2], "didntgrad": [1, 0, 1]}
    )
    outcome_col = "didntgrad"
    automl_metric = "log_loss"
    student_id_col = "id"
    input_kwargs = {"time_col": "semester_start", "timeout_minutes": 20}

    with patch.dict("sys.modules", databricks=mymodule):
        run_automl_classification(
            institution_id="test_inst",
            job_run_id="test",
            train_df=train_df,
            outcome_col=outcome_col,
            optimization_metric=automl_metric,
            student_id_col=student_id_col,
            **input_kwargs,
        )
        _, kwargs = mymodule.automl.classify.call_args
        pd.testing.assert_frame_equal(kwargs.get("dataset"), train_df)
        assert kwargs.get("target_col") == outcome_col
        assert kwargs.get("primary_metric") == automl_metric
        # particularly interested in making sure this was called as a list
        assert kwargs.get("exclude_cols") == [student_id_col]
        assert kwargs.get("time_col") == input_kwargs["time_col"]
        assert kwargs.get("timeout_minutes") == input_kwargs["timeout_minutes"]
        assert kwargs.get("pos_label") == True