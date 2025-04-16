from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from student_success_tool.modeling import training


def test_run_automl_classification_uses_correct_args_and_format():
    mymodule = MagicMock()

    train_df = pd.DataFrame(
        {"id": [1, 2, 3], "semester_start": [1, 1, 2], "didntgrad": [1, 0, 1]}
    )
    target_col = "didntgrad"
    automl_metric = "log_loss"
    student_id_col = "id"
    input_kwargs = {"time_col": "semester_start", "timeout_minutes": 20}

    with patch.dict("sys.modules", databricks=mymodule):
        training.run_automl_classification(
            df=train_df,
            target_col=target_col,
            primary_metric=automl_metric,
            institution_id="test_inst",
            job_run_id="test",
            student_id_col=student_id_col,
            **input_kwargs,
        )
        _, kwargs = mymodule.automl.classify.call_args
        pd.testing.assert_frame_equal(kwargs.get("dataset"), train_df)
        assert kwargs.get("target_col") == target_col
        assert kwargs.get("primary_metric") == automl_metric
        # particularly interested in making sure this was called as a list
        assert kwargs.get("exclude_cols") == [student_id_col]
        assert kwargs.get("time_col") == input_kwargs["time_col"]
        assert kwargs.get("timeout_minutes") == input_kwargs["timeout_minutes"]
        assert kwargs.get("pos_label") == True


@pytest.mark.parametrize(
    ["params", "exp_prefix"],
    [
        (
            {
                "institution_id": "inst_id",
                "job_run_id": "interactive",
                "primary_metric": "log_loss",
                "timeout_minutes": 10,
            },
            "inst_id__job_run_id='interactive'__primary_metric='log_loss'__timeout_minutes=10__",
        ),
        (
            {
                "institution_id": "other_inst_id",
                "job_run_id": "12345",
                "primary_metric": "f1",
                "timeout_minutes": 5,
                "exclude_frameworks": ["xgboost", "lightgbm"],
            },
            "other_inst_id__job_run_id='12345'__primary_metric='f1'__timeout_minutes=5__exclude_frameworks=xgboost,lightgbm__",
        ),
    ],
)
def test_get_experiment_name(params, exp_prefix):
    obs = training.get_experiment_name(**params)
    assert isinstance(obs, str)
    assert obs.startswith(exp_prefix)
