import pytest
from unittest.mock import MagicMock
from student_success_tool.reporting.sections.registry import SectionRegistry
from student_success_tool.reporting.sections.custom import (
    bias_sections as custom_bias_sections,
)
from student_success_tool.reporting.utils.formatting import Formatting
import logging


@pytest.fixture
def mock_card():
    card = MagicMock()
    formatter = Formatting()
    card.format.indent_level.side_effect = formatter.indent_level
    card.format.friendly_case.side_effect = formatter.friendly_case
    return card

@pytest.fixture
def registry():
    return SectionRegistry()

def test_bias_groups_section_with_valid_aliases(mock_card, registry):
    mock_card.cfg.student_group_aliases = {
        "firstgenflag": "First-Generation Status",
        "gender": "Gender",
        "ethnic": "Ethnicity",
        "race_demo": "Race",
    }

    custom_bias_sections.register_bias_sections(mock_card, registry)

    rendered = registry.render_all()
    result = rendered["bias_groups_section"]
    print(result)

    assert "- Our assessment for FNR Parity was conducted across the following student groups." in result
    assert "- First-Generation Status" in result
    assert "- Gender" in result
    assert "- Ethnicity" in result
    assert "- Race" in result

def test_bias_groups_section_with_aliases_that_need_friendlycase(mock_card):
    mock_card.cfg.student_group_aliases = {
        "firstgenflag": "first_generation_status",
        "disabilityflag": "disability_status"
    }

    registry = SectionRegistry()
    custom_bias_sections.register_bias_sections(mock_card, registry)

    rendered = registry.render_all()
    result = rendered["bias_groups_section"]
    print(result)

    assert "- First Generation Status" in result
    assert "- Disability Status" in result

def test_bias_groups_section_with_missing_aliases(mock_card, caplog):
    from student_success_tool.reporting.sections.custom import bias_sections as custom_bias_sections
    from student_success_tool.reporting.sections.registry import SectionRegistry

    mock_card.cfg.student_group_aliases = None
    mock_card.format.friendly_case.side_effect = lambda x: x.replace("_", " ").title()

    registry = SectionRegistry()
    custom_bias_sections.register_bias_sections(mock_card, registry)

    with caplog.at_level("WARNING"):
        rendered = registry.render_all()

    result = rendered["bias_groups_section"]
    assert "- Unable to extract student groups" in result
    assert any("Failed to extract student groups" in message for message in caplog.messages)
