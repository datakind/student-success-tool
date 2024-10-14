import numpy as np
import pandas as pd
import pytest

from student_success_tool.modeling import feature_selection as fs


@pytest.fixture(scope="module")
def df():
    return pd.DataFrame(
        {
            "id": ["01", "02", "03", "04", "05"],
            "categorical_feature": ["a", "a", "b", "c", "b"],
            "include_feature": [3.142, 1.414, 1.618, 2.718, 0.693],
            "good_feature": [1, 2, 3, 4, 5],
            "partially_incomplete_feature": [np.nan, np.nan, 8, 5, 3],
            "incomplete_feature": [np.nan, 0.0, np.nan, np.nan, np.nan],
            "low_variance_feature": [1, 1, 1, 1, 2],
            "zero_variance_feature": [1, 1, 1, 1, 1],
            "high_collinear_feature": [6, 7, 8, 9, 10],
            "low_collinear_feature": [6.5, 1.5, 4, 3.5, 2],
        }
    ).astype({"id": "string", "categorical_feature": "category"})


# TODO: figure out why collinaer features col keeps raising errors / gives bad results
# @pytest.mark.parametrize(
#     ["non_feature_cols", "force_include_cols", "exp_dropped_cols"],
#     [
#         (
#             ["id"],
#             ["include_feature"],
#             ["incomplete_feature", "zero_variance_feature", "high_collinear_feature"],
#         ),
#     ],
# )
# def test_select_features(df, non_feature_cols, force_include_cols, exp_dropped_cols):
#     df_result = fs.select_features(
#         df, non_feature_cols=non_feature_cols, force_include_cols=force_include_cols
#     )
#     assert df.columns.difference(df_result.columns).tolist() == exp_dropped_cols
#     assert df_result.shape[0] == df.shape[0]


@pytest.mark.parametrize(
    ["threshold", "exp_dropped_cols"],
    [
        (0.5, ["incomplete_feature"]),
        (0.4, ["incomplete_feature", "partially_incomplete_feature"]),
    ],
)
def test_drop_incomplete_features(df, threshold, exp_dropped_cols):
    df_result = fs.drop_incomplete_features(df, threshold=threshold)
    assert df.columns.difference(df_result.columns).tolist() == exp_dropped_cols
    assert df_result.shape[0] == df.shape[0]


@pytest.mark.parametrize(
    ["threshold", "exp_dropped_cols"],
    [
        (0.0, ["incomplete_feature", "zero_variance_feature"]),
        (0.2, ["incomplete_feature", "low_variance_feature", "zero_variance_feature"]),
    ],
)
def test_drop_low_variance_features(df, threshold, exp_dropped_cols):
    df_result = fs.drop_low_variance_features(df, threshold=threshold)
    assert df.columns.difference(df_result.columns).tolist() == exp_dropped_cols
    assert df_result.shape[0] == df.shape[0]


def test_drop_collinear_features_iteratively():
    df = pd.DataFrame(
        {"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10], "C": [6.5, 1.5, 4, 3.5, 2]}
    )
    df_result = fs.drop_collinear_features_iteratively(
        df, threshold=10.0, force_include_cols=[]
    )
    assert df_result.columns.tolist() == ["A", "C"]
    assert df_result.shape[0] == df.shape[0]


def test_drop_collinear_features_iteratively_force_include():
    df = pd.DataFrame(
        {"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10], "C": [6.5, 1.5, 4, 3.5, 2]}
    )
    df_result = fs.drop_collinear_features_iteratively(
        df, threshold=10.0, force_include_cols=["B"]
    )
    assert df_result.columns.tolist() == ["B", "C"]
    assert df_result.shape[0] == df.shape[0]


def test_drop_collinear_features_iteratively_fails_with_one_feature_left():
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]})
    with pytest.raises(Exception):
        fs.drop_collinear_features_iteratively(df)
