import pytest
from unittest.mock import MagicMock
from student_success_tool.reporting.sections.registry import SectionRegistry
from student_success_tool.reporting.sections.custom import (
    bias_sections as custom_bias_sections,
)
from student_success_tool.reporting.utils.formatting import Formatting


@pytest.fixture
def mock_card():
    card = MagicMock()
    formatter = Formatting()
    card.format.indent_level.side_effect = formatter.indent_level
    card.format.friendly_case.side_effect = formatter.friendly_case
    card.format.header_level.side_effect = formatter.header_level
    card.format.bold.side_effect = formatter.bold
    card.format.italic.side_effect = formatter.italic
    card.assets_folder = "/tmp/assets"
    card.run_id = "dummy_run_id"
    return card


@pytest.fixture
def registry():
    return SectionRegistry()


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: bias_groups_section
# ─────────────────────────────────────────────────────────────────────────────

def test_bias_groups_section_with_valid_aliases(mock_card, registry):
    mock_card.cfg.student_group_aliases = {
        "firstgenflag": "First-Generation Status",
        "gender": "Gender",
        "ethnicity_ipeds": "Ethnicity",
        "demo_race": "Race",
    }

    custom_bias_sections.register_bias_sections(mock_card, registry)

    rendered = registry.render_all()
    result = rendered["bias_groups_section"]
    print(result)

    assert (
        "- Our assessment for FNR Parity was conducted across the following student groups."
        in result
    )
    assert "- First-Generation Status" in result
    assert "- Gender" in result
    assert "- Ethnicity" in result
    assert "- Race" in result


def test_bias_groups_section_with_aliases_that_need_friendlycase(mock_card):
    mock_card.cfg.student_group_aliases = {
        "firstgenflag": "first_generation_status",
        "disabilityflag": "disability_status",
    }

    registry = SectionRegistry()
    custom_bias_sections.register_bias_sections(mock_card, registry)

    rendered = registry.render_all()
    result = rendered["bias_groups_section"]
    print(result)

    assert "- First Generation Status" in result
    assert "- Disability Status" in result


def test_bias_groups_section_with_missing_aliases(mock_card, caplog):
    mock_card.cfg.student_group_aliases = None
    mock_card.format.friendly_case.side_effect = lambda x: x.replace("_", " ").title()

    registry = SectionRegistry()
    custom_bias_sections.register_bias_sections(mock_card, registry)

    with caplog.at_level("WARNING"):
        rendered = registry.render_all()

    result = rendered["bias_groups_section"]
    assert "- Unable to extract student groups" in result
    assert any(
        "Failed to extract student groups" in message for message in caplog.messages
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST: bias_summary_section uses aliases in header
# ─────────────────────────────────────────────────────────────────────────────

def test_bias_summary_section_uses_aliases(mock_card, tmp_path):
    import pandas as pd
    from student_success_tool.reporting.utils import utils

    # Setup alias
    mock_card.cfg.student_group_aliases = {
        "firstgenflag": "First-Generation Students",
    }

    # Mock formatting
    mock_card.format.friendly_case.side_effect = lambda x: x.replace("_", " ").title()

    # Patch asset folder
    mock_card.assets_folder = tmp_path
    mock_card.run_id = "run123"

    # Create bias_flags CSV
    df = pd.DataFrame({
        "group": ["firstgenflag"],
        "split_name": ["test"],
        "subgroups": ["yes vs no"],
        "fnr_percentage_difference": [0.12],
        "type": ["p < 0.05, 95% CI [0.05, 0.19]"]
    })
    bias_path = tmp_path / "bias_flags"
    bias_path.mkdir()
    df.to_csv(bias_path / "high_bias_flags.csv", index=False)

    # Patch utils.download_artifact to just return file path
    def mock_download_artifact(run_id, local_folder, artifact_path, description=None):
        return tmp_path / artifact_path

    utils.download_artifact = mock_download_artifact

    registry = SectionRegistry()
    custom_bias_sections.register_bias_sections(mock_card, registry)

    rendered = registry.render_all()
    result = rendered["bias_summary_section"]

    print(result)

    assert "First-Generation Students" in result
    assert "12% difference" in result
    assert "False Negative Parity Rate for First-Generation Students" in result
