import pytest
from unittest.mock import MagicMock
from student_success_tool.reporting.sections.registry import SectionRegistry
from student_success_tool.reporting.sections.pdp import (
    bias_sections as pdp_bias_sections,
)


@pytest.fixture
def mock_card():
    card = MagicMock()
    card.format.indent_level = lambda level: "  " * level
    return card


def test_pdp_bias_groups_override(mock_card):
    registry = SectionRegistry()
    pdp_bias_sections.register_bias_sections(mock_card, registry)

    rendered = registry.render_all()

    # Check that the overridden bias_groups_section contains the pdp-specific groups
    bias_groups_output = rendered["bias_groups_section"]
    assert "Ethnicity" in bias_groups_output
    assert "First Generation Status" in bias_groups_output
    assert "Gender" in bias_groups_output
    assert "Race" in bias_groups_output
    assert "Student Age" in bias_groups_output

    # Optional: Confirm that other base sections (e.g., bias_summary_section) still exist
    assert "bias_summary_section" in rendered
