import pytest
from unittest.mock import MagicMock

from student_success_tool.reporting.sections.registry import SectionRegistry
from student_success_tool.reporting.sections.attribute_sections import (
    register_attribute_sections,
)
from student_success_tool.reporting.utils.formatting import Formatting


@pytest.fixture
def mock_card():
    card = MagicMock()
    card.format = Formatting()
    return card


def test_outcome_graduation_and_full_time_limit(mock_card):
    mock_card.cfg.preprocessing.target.name = "graduation"
    mock_card.cfg.preprocessing.selection.intensity_time_limits = {
        "FULL-TIME": (2.0, "year"),
        "PART-TIME": (3, "year"),
    }

    registry = SectionRegistry()
    register_attribute_sections(mock_card, registry)
    rendered = registry.render_all()

    assert (
        "graduation within 2 years for full-time students"
        in rendered["outcome_section"]
    )
    assert "within 3 years for part-time students" in rendered["outcome_section"]


def test_outcome_missing_target_or_limits(mock_card):
    mock_card.cfg.preprocessing.target.name = ""
    mock_card.cfg.preprocessing.selection.intensity_time_limits = {}

    registry = SectionRegistry()
    register_attribute_sections(mock_card, registry)
    rendered = registry.render_all()

    assert (
        rendered["outcome_section"] == "**Target or Time Limit Information Not Found**"
    )


def test_target_population_section(mock_card):
    mock_card.cfg.preprocessing.selection.student_criteria = {
        "degree": ["bachelor's", "associate's"],
        "status": "full-time",
    }

    registry = SectionRegistry()
    register_attribute_sections(mock_card, registry)
    rendered = registry.render_all()

    assert "Degree" in rendered["target_population_section"]
    assert "- Bachelor's" in rendered["target_population_section"]
    assert "- Associate's" in rendered["target_population_section"]
    assert "Status" in rendered["target_population_section"]
    assert "- Full-Time" in rendered["target_population_section"]


def test_checkpoint_section_with_credit(mock_card):
    mock_card.cfg.preprocessing.checkpoint.name = "credit_45"
    mock_card.cfg.preprocessing.checkpoint.params = {"min_num_credits": 45}

    registry = SectionRegistry()
    register_attribute_sections(mock_card, registry)
    rendered = registry.render_all()

    assert "45 credits" in rendered["checkpoint_section"]


def test_checkpoint_section_with_semester(mock_card):
    mock_card.cfg.preprocessing.checkpoint.name = "first_semester"
    mock_card.cfg.preprocessing.checkpoint.params = {}

    registry = SectionRegistry()
    register_attribute_sections(mock_card, registry)
    rendered = registry.render_all()

    assert "first semester" in rendered["checkpoint_section"]


def test_checkpoint_section_unknown(mock_card):
    mock_card.cfg.preprocessing.checkpoint.name = "unknown_metric"
    mock_card.cfg.preprocessing.checkpoint.params = {}

    registry = SectionRegistry()
    register_attribute_sections(mock_card, registry)
    rendered = registry.render_all()

    assert rendered["checkpoint_section"] == "**Checkpoint Information Not Found**"
