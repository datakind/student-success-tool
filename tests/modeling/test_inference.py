import numpy as np
import pandas as pd
import pytest

from student_success_tool.modeling.inference import select_top_features_for_display


@pytest.mark.parametrize(
    [
        "features",
        "unique_ids",
        "predicted_probabilities",
        "shap_values",
        "n_features",
        "features_table",
        "exp",
    ],
    [
        (
            pd.DataFrame(
                {
                    "x1": ["val1", "val2", "val3"],
                    "x2": [True, False, True],
                    "x3": [2.0, 1.0, 0.5],
                    "x4": [1, 2, 3],
                }
            ),
            pd.Series([1, 2, 3]),
            [0.9, 0.1, 0.5],
            np.array(
                [[1.0, 0.9, 0.8, 0.7], [0.0, -1.0, 0.9, -0.8], [0.25, 0.0, -0.5, 0.75]]
            ),
            3,
            {
                "x1": {"name": "feature #1"},
                "x2": {"name": "feature #2"},
                "x3": {"name": "feature #3"},
            },
            pd.DataFrame(
                {
                    "Student ID": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                    "Support Score": [0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5],
                    "Top Indicators": [
                        "feature #1",
                        "feature #2",
                        "feature #3",
                        "feature #2",
                        "feature #3",
                        "x4",
                        "x4",
                        "feature #3",
                        "feature #1",
                    ],
                    "Indicator Value": [
                        "val1",
                        True,
                        2.0,
                        False,
                        1.0,
                        2,
                        3,
                        0.5,
                        "val3",
                    ],
                    "SHAP Value": [1.0, 0.9, 0.8, -1.0, 0.9, -0.8, 0.75, -0.5, 0.25],
                    "Rank": [1, 2, 3, 1, 2, 3, 1, 2, 3],
                }
            ),
        ),
        (
            pd.DataFrame(
                {
                    "x1": ["val1", "val2", "val3"],
                    "x2": [True, False, True],
                    "x3": [2.0, 1.0, 0.5],
                    "x4": [1, 2, 3],
                }
            ),
            pd.Series([1, 2, 3]),
            [0.9, 0.1, 0.5],
            np.array(
                [[1.0, 0.9, 0.8, 0.7], [0.0, -1.0, 0.9, -0.8], [0.25, 0.0, -0.5, 0.75]]
            ),
            1,
            None,
            pd.DataFrame(
                {
                    "Student ID": [1, 2, 3],
                    "Support Score": [0.9, 0.1, 0.5],
                    "Top Indicators": ["x1", "x2", "x4"],
                    "Indicator Value": ["val1", False, 3],
                    "SHAP Value": [1.0, -1.0, 0.75],
                    "Rank": [1, 1, 1],
                }
            ),
        ),
    ],
)
def test_select_top_features_for_display(
    features,
    unique_ids,
    predicted_probabilities,
    shap_values,
    n_features,
    features_table,
    exp,
):
    obs = select_top_features_for_display(
        features,
        unique_ids,
        predicted_probabilities,
        shap_values,
        n_features=n_features,
        features_table=features_table,
    )
    assert isinstance(obs, pd.DataFrame) and not obs.empty
    assert pd.testing.assert_frame_equal(obs, exp) is None
