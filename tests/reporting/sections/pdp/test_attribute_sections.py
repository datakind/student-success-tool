import pytest
from unittest.mock import MagicMock
from student_success_tool.reporting.sections.registry import SectionRegistry
from student_success_tool.reporting.sections.pdp import (
    attribute_sections as pdp_attribute_sections,
)
from student_success_tool.reporting.utils.formatting import Formatting


@pytest.fixture
def mock_card():
    card = MagicMock()
    card.format = Formatting()
    return card


@pytest.mark.parametrize(
    "outcome_type, time_limits, extra_config, expected_snippet",
    [
        (
            "retention",
            {},
            {},
            "The model predicts the risk of non-retention into the student's second academic year based on student, course, and academic data.",
        ),
        (
            "graduation",
            {"FULL-TIME": (2.0, "year"), "PART-TIME": (3.0, "year")},
            {},
            "The model predicts the risk of not graduating on time within 2 years for full-time students, and within 3 years for part-time students, based on student, course, and academic data.",
        ),
        (
            "credits_earned",
            {"FULL-TIME": (1.5, "year")},
            {"min_num_credits": 45},
            "The model predicts the risk of not earning 45 credits within 1.5 years for full-time students, based on student, course, and academic data.",
        ),
    ],
)
def test_outcome_variants(
    mock_card, outcome_type, time_limits, extra_config, expected_snippet
):
    mock_card.cfg.preprocessing.target.type_ = outcome_type
    mock_card.cfg.preprocessing.selection.intensity_time_limits = time_limits

    # Patching checkpoint since we are testing outcome
    mock_card.cfg.preprocessing.checkpoint.type_ = "all"

    if outcome_type == "credits_earned" and "min_num_credits" in extra_config:
        mock_card.cfg.preprocessing.target.min_num_credits = extra_config[
            "min_num_credits"
        ]

    registry = SectionRegistry()
    pdp_attribute_sections.register_attribute_sections(mock_card, registry)
    rendered = registry.render_all()

    assert expected_snippet in rendered["outcome_section"]


def test_target_population_section(mock_card):
    mock_card.cfg.preprocessing.selection.student_criteria = {
        "degree": ["bachelor's", "associate's"],
        "status": "full-time",
    }

    # Patching checkpoint since we are testing target population
    mock_card.cfg.preprocessing.checkpoint.type_ = "all"

    registry = SectionRegistry()
    pdp_attribute_sections.register_attribute_sections(mock_card, registry)
    rendered = registry.render_all()

    assert "Degree" in rendered["target_population_section"]
    assert "- Bachelor's" in rendered["target_population_section"]
    assert "- Associate's" in rendered["target_population_section"]
    assert "Status" in rendered["target_population_section"]
    assert "- Full-Time" in rendered["target_population_section"]


@pytest.mark.parametrize(
    "checkpoint_type,expected_output",
    [
        (
            "all",
            "The model makes this prediction when the student has completed their 3rd term",
        ),
        (
            "num_credits_earned",
            "The model makes this prediction when the student has earned 30 credits",
        ),
        (
            "within_cohort",
            "The model makes this prediction when the student has completed their first term within their cohort",
        ),
        (
            "enrollment_year",
            "The model makes this prediction when the student has completed their 2nd year of enrollment",
        ),
    ],
)
def test_checkpoint_variants(mock_card, checkpoint_type, expected_output):
    mock_card.cfg.preprocessing.checkpoint.type_ = checkpoint_type
    if checkpoint_type == "all":
        mock_card.cfg.preprocessing.checkpoint.n = 3
        mock_card.cfg.preprocessing.checkpoint.n = 1
    if checkpoint_type == "num_credits_earned":
        mock_card.cfg.preprocessing.checkpoint.min_num_credits = 30
        mock_card.cfg.preprocessing.checkpoint.n = 1
    if checkpoint_type == "enrollment_year":
        mock_card.cfg.preprocessing.checkpoint.enrollment_year = 2
        mock_card.cfg.preprocessing.checkpoint.n = -1

    registry = SectionRegistry()
    pdp_attribute_sections.register_attribute_sections(mock_card, registry)
    rendered = registry.render_all()

    assert expected_output in rendered["checkpoint_section"]


def test_checkpoint_unknown_type_raises_error(mock_card):
    mock_card.cfg.preprocessing.checkpoint.type_ = "unknown_metric"

    registry = SectionRegistry()
    pdp_attribute_sections.register_attribute_sections(mock_card, registry)

    with pytest.raises(ValueError, match="Unknown checkpoint type: unknown_metric"):
        registry.render_all()
