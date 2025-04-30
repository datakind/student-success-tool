import pandas as pd
import pytest

from student_success_tool.dataio.converters import pdp


@pytest.fixture(scope="module")
def df_test():
    return pd.DataFrame(
        {
            "student_id": ["01", "01", "01", "01", "02", "02"],
            "academic_year": [
                "2024-25",
                "2024-25",
                "2024-25",
                "2024-25",
                "2020-21",
                "2020-21",
            ],
            "academic_term": ["FALL", "FALL", "FALL", "SPRING", "SUMMER", "SUMMER"],
            "course_prefix": ["PHYS", "PHYS", "MATH", "MATH", "ENGL", "ENGL"],
            "course_number": ["101", "101", "101", "202", "102", "102"],
            "section_id": ["001", "001", "123", "123", "456", "456"],
            "number_of_credits_attempted": [1.0, 3.0, 3.0, 3.0, 4.0, 1.0],
        },
        dtype="string",
    ).astype({"number_of_credits_attempted": "Float32"})


@pytest.mark.parametrize(
    "exp",
    [
        pd.DataFrame(
            {
                "student_id": ["01", "01", "01", "01", "02", "02"],
                "academic_year": [
                    "2024-25",
                    "2024-25",
                    "2024-25",
                    "2024-25",
                    "2020-21",
                    "2020-21",
                ],
                "academic_term": ["FALL", "FALL", "FALL", "SPRING", "SUMMER", "SUMMER"],
                "course_prefix": ["PHYS", "PHYS", "MATH", "MATH", "ENGL", "ENGL"],
                "course_number": ["101-2", "101-1", "101", "202", "102-1", "102-2"],
                "section_id": ["001", "001", "123", "123", "456", "456"],
                "number_of_credits_attempted": [1.0, 3.0, 3.0, 3.0, 4.0, 1.0],
            },
            dtype="string",
        ).astype({"number_of_credits_attempted": "Float32"})
    ],
)
def test_dedupe_by_renumbering_courses(df_test, exp):
    obs = pdp.dedupe_by_renumbering_courses(df_test)
    assert isinstance(obs, pd.DataFrame)
    assert pd.testing.assert_frame_equal(obs, exp) is None
