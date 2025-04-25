import pandas as pd
import pytest

from student_success_tool.targets.pdp import retention


@pytest.mark.parametrize(
    ["df", "student_id_cols", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "02", "04", "03"],
                    "retention": [1, 1, 0, 0, pd.NA],
                },
            ).astype({"student_id": "string", "retention": "Int8"}),
            "student_id",
            pd.Series(
                data=[False, True, True, False],
                index=pd.Index(
                    ["01", "02", "04", "03"], dtype="string", name="student_id"
                ),
                name="target",
            ),
        ),
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "02", "04", "03"],
                    "retention": [True, True, False, False, pd.NA],
                },
            ).astype({"student_id": "string", "retention": "boolean"}),
            "student_id",
            pd.Series(
                data=[False, True, True, False],
                index=pd.Index(
                    ["01", "02", "04", "03"], dtype="string", name="student_id"
                ),
                name="target",
            ),
        ),
        (
            pd.DataFrame(
                {
                    "student_id_pt1": ["01", "02"],
                    "student_id_pt2": ["A", "B"],
                    "retention": [True, False],
                },
            ).astype(
                {
                    "student_id_pt1": "string",
                    "student_id_pt2": "string",
                    "retention": "bool",
                }
            ),
            ["student_id_pt1", "student_id_pt2"],
            pd.Series(
                data=[False, True],
                index=pd.MultiIndex.from_frame(
                    pd.DataFrame(
                        {"student_id_pt1": ["01", "02"], "student_id_pt2": ["A", "B"]},
                        dtype="string",
                    )
                ),
                name="target",
            ),
        ),
        # this is a pathological case: retention varies across student-terms
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "01"],
                    "retention": [True, False, pd.NA],
                },
            ).astype({"student_id": "string", "retention": "boolean"}),
            "student_id",
            pd.Series(
                data=[False],
                index=pd.Index(["01"], dtype="string", name="student_id"),
                name="target",
            ),
        ),
    ],
)
def test_compute_target(df, student_id_cols, exp):
    obs = retention.compute_target(
        df, student_id_cols=student_id_cols, retention_col="retention"
    )
    assert isinstance(obs, pd.Series)
    assert pd.testing.assert_series_equal(obs, exp) is None
