import pytest
from unittest.mock import MagicMock
from student_success_tool.reporting.sections.registry import SectionRegistry
from student_success_tool.reporting.sections.custom import (
    bias_sections as custom_bias_sections,
)
import logging

@pytest.fixture
def mock_card():
    card = MagicMock()
    card.format.indent_level.side_effect = lambda level: "  " * level  # 2-space indentation
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

    result = registry.get("bias_groups_section")()

    assert "- Our assessment for FNR Parity was conducted across the following student groups." in result
    assert "- First-Generation Status" in result
    assert "- Gender" in result
    assert "- Ethnicity" in result
    assert "- Race" in result

def test_bias_groups_section_with_missing_aliases(mock_card, mock_registry, caplog):
    # Simulate missing or malformed config
    mock_card.cfg.student_group_aliases = None

    custom_bias_sections.register_bias_sections(mock_card, mock_registry)

    with caplog.at_level(logging.WARNING):
        result = mock_registry.register.call_args_list[0][0][1]()

    assert "- Unable to extract student groups" in result

    assert any("Failed to extract student groups" in message for message in caplog.messages)
