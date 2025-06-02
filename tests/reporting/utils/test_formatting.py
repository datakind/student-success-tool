import pytest
from student_success_tool.reporting.utils.formatting import Formatting


@pytest.fixture
def formatter():
    return Formatting()


@pytest.mark.parametrize(
    "input_text,expected",
    [
        ("bachelor's degree", "Bachelor's Degree"),
        ("full-time", "Full-Time"),
        ("associate's-degree", "Associate's-Degree"),
        ("first_generation", "First Generation"),
        ("1st_semester", "1st Semester"),
        ("STEM", "Stem"),  # no acronym preservation for now
        (0.0, "0.0"),
        (3, "3"),
        ("3.0", "3.0"),
        ("4", "4"),
    ],
)
def test_friendly_case_capitalize_true(formatter, input_text, expected):
    result = formatter.friendly_case(input_text)
    assert result == expected


@pytest.mark.parametrize(
    "input_text,expected",
    [
        ("bachelor's degree", "bachelor's degree"),
        ("full-time", "full-time"),
        ("first_term", "first term"),
        ("1st_semester", "1st semester"),
    ],
)
def test_friendly_case_capitalize_false(formatter, input_text, expected):
    result = formatter.friendly_case(input_text, capitalize=False)
    assert result == expected
