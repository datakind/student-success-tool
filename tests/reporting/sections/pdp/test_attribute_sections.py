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
            "The model predicts the likelihood of non-retention into the student's second academic year based on student, course, and academic data.",
        ),
        (
            "graduation",
            {"FULL-TIME": (2.0, "year"), "PART-TIME": (3.0, "year")},
            {},
            "The model predicts the likelihood of not graduating on time within 2 years for full-time students, and within 3 years for part-time students, based on student, course, and academic data.",
        ),
        (
            "credits_earned",
            {"FULL-TIME": (1.5, "year")},
            {"min_num_credits": 45},
            "The model predicts the likelihood of not earning 45 credits within 1.5 years for full-time students, based on student, course, and academic data.",
        ),
    ],
)
def test_outcome_variants(
    mock_card, outcome_type, time_limits, extra_config, expected_snippet
):
    mock_card.cfg.preprocessing.target.type_ = outcome_type
    mock_card.cfg.preprocessing.selection.intensity_time_limits = time_limits

    # Patching checkpoint since we are testing outcome
    mock_card.cfg.preprocessing.checkpoint.type_ = "nth"
    mock_card.cfg.preprocessing.checkpoint.n = 1
    mock_card.cfg.preprocessing.checkpoint.exclude_pre_cohort_terms = False
    mock_card.cfg.preprocessing.checkpoint.exclude_non_core_terms = False
    mock_card.cfg.preprocessing.checkpoint.valid_enrollment_year = None

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
    mock_card.cfg.preprocessing.checkpoint.type_ = "nth"
    mock_card.cfg.preprocessing.checkpoint.n = 1
    mock_card.cfg.preprocessing.checkpoint.exclude_pre_cohort_terms = False
    mock_card.cfg.preprocessing.checkpoint.exclude_non_core_terms = False
    mock_card.cfg.preprocessing.checkpoint.valid_enrollment_year = None

    registry = SectionRegistry()
    pdp_attribute_sections.register_attribute_sections(mock_card, registry)
    rendered = registry.render_all()

    assert "Degree" in rendered["target_population_section"]
    assert "- Bachelor's" in rendered["target_population_section"]
    assert "- Associate's" in rendered["target_population_section"]
    assert "Status" in rendered["target_population_section"]
    assert "- Full-Time" in rendered["target_population_section"]


@pytest.mark.parametrize(
    "checkpoint_type,n,exclude_pre,exclude_non_core,valid_year,expected_output",
    [
        # Basic nth
        (
            "nth",
            1,
            True,
            True,
            None,
            "The model makes this prediction when the student has completed their 2nd term.",
        ),
        # Include pre-cohort terms only
        (
            "nth",
            3,
            False,
            True,
            None,
            "The model makes this prediction when the student has completed their 4th term including pre-cohort terms.",
        ),
        # Include non-core terms only
        (
            "nth",
            3,
            True,
            False,
            None,
            "The model makes this prediction when the student has completed their 4th term including non-core terms.",
        ),
        # Include both
        (
            "nth",
            3,
            False,
            False,
            None,
            "The model makes this prediction when the student has completed their 4th term including pre-cohort terms and non-core terms.",
        ),
        # Include both + valid enrollment year
        (
            "nth",
            -1,
            False,
            False,
            1,
            "The model makes this prediction when the student has completed their last term including pre-cohort terms and non-core terms, provided the term occurred in their 1st year of enrollment.",
        ),
        # Only valid enrollment year
        (
            "nth",
            6,
            True,
            True,
            3,
            "The model makes this prediction when the student has completed their 7th term, provided the term occurred in their 3rd year of enrollment.",
        ),
        (
            "first",
            0,
            False,
            False,
            None,
            "The model makes this prediction when the student has completed their first term.",
        ),
        (
            "last",
            -1,
            False,
            False,
            None,
            "The model makes this prediction when the student has completed their last term.",
        ),
        (
            "first_at_num_credits_earned",
            None,
            None,
            None,
            None,
            "The model makes this prediction when the student has earned 30 credits.",
        ),
        (
            "last_in_enrollment_year",
            None,
            None,
            None,
            None,
            "The model makes this prediction when the student has completed their 2nd year of enrollment.",
        ),
        (
            "first_within_cohort",
            None,
            None,
            None,
            None,
            "The model makes this prediction when the student has completed their first term within their cohort.",
        ),
    ],
)
def test_checkpoint_variants(
    mock_card,
    checkpoint_type,
    n,
    exclude_pre,
    exclude_non_core,
    valid_year,
    expected_output,
):
    mock_card.cfg.preprocessing.checkpoint.type_ = checkpoint_type

    if checkpoint_type == "nth":
        mock_card.cfg.preprocessing.checkpoint.n = n
        mock_card.cfg.preprocessing.checkpoint.exclude_pre_cohort_terms = exclude_pre
        mock_card.cfg.preprocessing.checkpoint.exclude_non_core_terms = exclude_non_core
        mock_card.cfg.preprocessing.checkpoint.valid_enrollment_year = valid_year
    elif checkpoint_type == "first_at_num_credits_earned":
        mock_card.cfg.preprocessing.checkpoint.min_num_credits = 30
    elif checkpoint_type == "last_in_enrollment_year":
        mock_card.cfg.preprocessing.checkpoint.enrollment_year = 2

    registry = SectionRegistry()
    pdp_attribute_sections.register_attribute_sections(mock_card, registry)
    rendered = registry.render_all()

    assert rendered["checkpoint_section"].strip().endswith(expected_output)


def test_checkpoint_unknown_type_raises_error(mock_card):
    mock_card.cfg.preprocessing.checkpoint.type_ = "unknown_metric"

    registry = SectionRegistry()
    pdp_attribute_sections.register_attribute_sections(mock_card, registry)

    with pytest.raises(ValueError, match="Unknown checkpoint type: unknown_metric"):
        registry.render_all()
