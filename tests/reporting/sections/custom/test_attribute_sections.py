import pytest
from unittest.mock import MagicMock
from student_success_tool.reporting.sections.registry import SectionRegistry
from student_success_tool.reporting.sections.custom import (
    attribute_sections as custom_attribute_sections,
)
from student_success_tool.reporting.utils.formatting import Formatting


@pytest.fixture
def mock_card():
    card = MagicMock()
    card.format = Formatting()
    card.context = {}
    return card


def test_development_note_with_version(mock_card):
    mock_card.context["version_number"] = "1.2.3"
    registry = SectionRegistry()
    custom_attribute_sections.register_attribute_sections(mock_card, registry)
    rendered = registry.render_all()
    result = rendered["development_note_section"]

    assert "Model Version 1.2.3" in result
    assert "Developed by DataKind" in result


def test_development_note_without_version(mock_card):
    mock_card.context = {}  # no version_number
    registry = SectionRegistry()
    custom_attribute_sections.register_attribute_sections(mock_card, registry)
    rendered = registry.render_all()
    result = rendered["development_note_section"]

    assert "Model Version" not in result
    assert "Developed by DataKind" in result


@pytest.mark.parametrize(
    "category,unit,value,expected_snippet",
    [
        (
            "retention",
            None,
            None,
            "non-retention into the student's second academic year",
        ),
        ("graduation", "year", 2, "not graduating on time within 2 years"),
        (
            "graduation",
            "credit",
            30,
            "not graduating on time in achieving 30 credits required for graduation",
        ),
        ("graduation", "term", 1, "not graduating on time within 1 term"),
        ("graduation", "semester", 3, "not graduating on time within 3 semesters"),
        (
            "graduation",
            "pct_completion",
            85,
            "not graduating on time at 85% completion",
        ),
        (
            "graduation",
            "custom_metric",
            5,
            "not graduating on time within 5 custom_metric",
        ),  # fallback
    ],
)
def test_outcome_section_variants(mock_card, category, unit, value, expected_snippet):
    mock_card.cfg.preprocessing.target.category = category

    if category == "graduation":
        mock_card.cfg.preprocessing.target.unit = unit
        mock_card.cfg.preprocessing.target.value = value

    registry = SectionRegistry()
    custom_attribute_sections.register_attribute_sections(mock_card, registry)

    rendered = registry.render_all()
    result = rendered["outcome_section"]
    assert expected_snippet in result


def test_outcome_section_fallback_on_missing_config(mock_card, caplog):
    # simulate failure (missing target)
    del mock_card.cfg.preprocessing.target

    registry = SectionRegistry()
    custom_attribute_sections.register_attribute_sections(mock_card, registry)

    with caplog.at_level("WARNING"):
        rendered = registry.render_all()
        result = rendered["outcome_section"]
        assert "Unable to retrieve model outcome information" in result
        assert any(
            "Failed to generate outcome description" in msg for msg in caplog.messages
        )


def test_target_population_valid_dict(mock_card):
    mock_card.cfg.preprocessing.selection.student_criteria = {
        "status": "full-time",
        "degree": ["bachelor's", "associate's"],
    }
    mock_card.cfg.preprocessing.selection.student_criteria_aliases = {
        "status": "Enrollment Status",
        "degree": "Degree Type",
    }

    registry = SectionRegistry()
    custom_attribute_sections.register_attribute_sections(mock_card, registry)
    rendered = registry.render_all()
    result = rendered["target_population_section"]

    assert "- Enrollment Status" in result
    assert "- Degree Type" in result
    assert "- Full-Time" in result
    assert "- Bachelor's" in result
    assert "- Associate's" in result


def test_target_population_empty(mock_card, caplog):
    mock_card.cfg.preprocessing.selection.student_criteria = {}
    mock_card.cfg.preprocessing.selection.student_criteria_aliases = {}

    registry = SectionRegistry()
    with caplog.at_level("WARNING"):
        custom_attribute_sections.register_attribute_sections(mock_card, registry)
        result = registry.render_all()["target_population_section"]

    assert "No specific student criteria were applied." in result
    assert any("No student criteria provided in config." in m for m in caplog.messages)


def test_target_population_non_dict(mock_card):
    mock_card.cfg.preprocessing.selection.student_criteria_aliases = [
        "status",
        "degree",
    ]

    registry = SectionRegistry()
    custom_attribute_sections.register_attribute_sections(mock_card, registry)
    rendered = registry.render_all()
    result = rendered["target_population_section"]

    assert "Student criteria should be provided as a dictionary." in result


@pytest.mark.parametrize(
    "unit,value,expected_snippet",
    [
        ("credit", 30, "earned 30 credits"),
        ("year", 2, "completed 2 years"),
        ("term", 1, "completed 1 term"),
        ("semester", 3, "completed 3 semesters"),
    ],
)
def test_checkpoint_valid_variants(mock_card, unit, value, expected_snippet):
    mock_card.cfg.preprocessing.checkpoint.unit = unit
    mock_card.cfg.preprocessing.checkpoint.value = value

    registry = SectionRegistry()
    custom_attribute_sections.register_attribute_sections(mock_card, registry)
    rendered = registry.render_all()
    result = rendered["checkpoint_section"]

    assert expected_snippet in result


def test_checkpoint_section_missing_config(mock_card, caplog):
    # Simulate exception due to missing checkpoint
    del mock_card.cfg.preprocessing.checkpoint

    registry = SectionRegistry()
    custom_attribute_sections.register_attribute_sections(mock_card, registry)

    with caplog.at_level("WARNING"):
        rendered = registry.render_all()
        result = rendered["checkpoint_section"]
        assert "Unable to retrieve model checkpoint information" in result
        assert "Failed to generate checkpoint description" in caplog.text
